#include "wp-tp-trace.cuh"
#include "ggml-cuda.h" // ggml_backend_is_cuda

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

// See wp-tp-trace.cuh for the full contract. This file owns all state; every
// other TU reaches it either directly (allreduce.cu, ggml-cuda.cu -- both
// already CUDA/HIP translation units, same library) or through the
// wp_tp_trace_mark/_global/_gpu_mark proc-address bridges (ggml-backend-meta.cpp,
// which is backend-agnostic and must not depend on CUDA types).

namespace {

constexpr int WP_TPT_MAX_DEVICES = GGML_CUDA_MAX_DEVICES;
// Generous headroom over the 2 rolling slots this codebase actually uses
// today (WP_META_SLOT_STREAMS / n_graph_slots) -- indices out of range are
// dropped rather than clamped, so a future widening degrades to "no
// cross-referencing for the extra slots", not a crash.
constexpr int WP_TPT_MAX_SLOTS = 8;

struct Row {
    int          kind         = 0;
    long long    ubatch_idx   = -1;
    long long    slot         = -1;
    long long    subgraph_idx = -1;
    int          device       = -1;
    int64_t      host_t_us    = 0;
    double       gpu_t_us     = -1.0;  // -1 => n/a until resolved (or forever, for host-only rows)
    cudaEvent_t  gpu_event    = nullptr; // owned by this row until flush resolves+frees it
    char         extra[64]    = {0};
};

struct DeviceState {
    bool                     inited        = false;
    cudaEvent_t              base_event    = nullptr;
    int64_t                  base_host_us  = 0;
    std::vector<cudaEvent_t> free_events;
};

struct State {
    std::mutex   mu;
    std::vector<Row> rows;
    DeviceState  devs[WP_TPT_MAX_DEVICES];
    long long    ubatch_of_slot[WP_TPT_MAX_SLOTS];
    long long    subgraph_of_slot[WP_TPT_MAX_SLOTS];
    std::string  path;
    size_t       flush_n        = 200000;
    bool         wrote_header   = false;
    bool         atexit_armed   = false;
};

State & state() {
    static State s;
    return s;
}

int64_t now_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * kind_name(int kind) {
    switch (kind) {
        case WP_TPT_UBATCH_BEGIN:         return "UBATCH_BEGIN";
        case WP_TPT_UBATCH_END:           return "UBATCH_END";
        case WP_TPT_COMPUTE_SUBMIT_BEGIN: return "COMPUTE_SUBMIT_BEGIN";
        case WP_TPT_COMPUTE_SUBMIT_END:   return "COMPUTE_SUBMIT_END";
        case WP_TPT_COMPUTE_GPU_START:    return "COMPUTE_GPU_START";
        case WP_TPT_COMPUTE_GPU_END:      return "COMPUTE_GPU_END";
        case WP_TPT_AR_BEGIN_HOST:        return "AR_BEGIN_HOST";
        case WP_TPT_AR_END_HOST:          return "AR_END_HOST";
        case WP_TPT_AR_PACK_DONE:         return "AR_PACK_DONE";
        case WP_TPT_AR_SENT:              return "AR_SENT";
        case WP_TPT_AR_RECVD:             return "AR_RECVD";
        case WP_TPT_AR_UNPACK_DONE:       return "AR_UNPACK_DONE";
        case WP_TPT_FENCE_WAIT:           return "FENCE_WAIT";
        case WP_TPT_HEADER:               return "HEADER";
        default:                          return "UNKNOWN";
    }
}

// Must be called with state().mu held.
void ensure_device_inited_locked(int device, cudaStream_t stream) {
    if (device < 0 || device >= WP_TPT_MAX_DEVICES) {
        return;
    }
    DeviceState & d = state().devs[device];
    if (d.inited) {
        return;
    }
    cudaEvent_t ev = nullptr;
    if (cudaEventCreate(&ev) != cudaSuccess) {
        return;
    }
    if (cudaEventRecord(ev, stream) != cudaSuccess) {
        cudaEventDestroy(ev);
        return;
    }
    d.base_event   = ev;
    d.base_host_us = now_us();
    d.inited       = true;

    Row r;
    r.kind      = WP_TPT_HEADER;
    r.device    = device;
    r.host_t_us = d.base_host_us;
    r.gpu_t_us  = 0.0;
    snprintf(r.extra, sizeof(r.extra), "base_event");
    state().rows.push_back(r);
}

// Must be called with state().mu held.
cudaEvent_t acquire_event_locked(int device) {
    DeviceState & d = state().devs[device];
    if (!d.free_events.empty()) {
        cudaEvent_t ev = d.free_events.back();
        d.free_events.pop_back();
        return ev;
    }
    cudaEvent_t ev = nullptr;
    if (cudaEventCreate(&ev) != cudaSuccess) {
        return nullptr;
    }
    return ev;
}

// Must be called with state().mu held. Resolves every pending GPU event to a
// gpu_t_us (syncing on it -- this is the one place tracing can stall the
// host, and only at flush granularity, never per row), appends everything to
// WP_TP_TRACE_FILE, and clears the in-memory buffer.
void flush_locked() {
    State & s = state();
    if (s.rows.empty()) {
        return;
    }
    if (s.path.empty()) {
        s.rows.clear();
        return;
    }

    for (Row & r : s.rows) {
        if (r.gpu_event == nullptr) {
            continue;
        }
        const int dev = r.device;
        if (dev < 0 || dev >= WP_TPT_MAX_DEVICES || !s.devs[dev].inited) {
            r.gpu_event = nullptr; // orphaned (shouldn't happen); drop rather than leak the query
            continue;
        }
        ggml_cuda_set_device(dev);
        if (cudaEventSynchronize(r.gpu_event) == cudaSuccess) {
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, s.devs[dev].base_event, r.gpu_event) == cudaSuccess) {
                r.gpu_t_us = s.devs[dev].base_host_us + (double) ms * 1000.0;
            }
        } else {
            (void) cudaGetLastError(); // clear sticky error; row keeps gpu_t_us == -1
        }
        s.devs[dev].free_events.push_back(r.gpu_event);
        r.gpu_event = nullptr;
    }

    FILE * f = fopen(s.path.c_str(), "a");
    if (f != nullptr) {
        if (!s.wrote_header) {
            fprintf(f, "kind,ubatch_idx,slot,subgraph_idx,device,host_t_us,gpu_t_us,extra\n");
            s.wrote_header = true;
        }
        char gbuf[32];
        for (const Row & r : s.rows) {
            if (r.gpu_t_us < 0.0) {
                gbuf[0] = '\0';
            } else {
                snprintf(gbuf, sizeof(gbuf), "%.1f", r.gpu_t_us);
            }
            fprintf(f, "%s,%lld,%lld,%lld,%d,%lld,%s,%s\n",
                    kind_name(r.kind), r.ubatch_idx, r.slot, r.subgraph_idx, r.device,
                    (long long) r.host_t_us, gbuf, r.extra);
        }
        fclose(f);
    }
    s.rows.clear();
}

void init_once_locked_by_caller() {
    // Called only from within a function already holding state().mu, on the
    // first row of the process -- guarded by State::atexit_armed rather than
    // std::call_once so it can run under the same lock as the row push.
    State & s = state();
    if (s.atexit_armed) {
        return;
    }
    s.atexit_armed = true;
    const char * path = getenv("WP_TP_TRACE_FILE");
    s.path = path != nullptr ? path : "";
    if (const char * n = getenv("WP_TP_TRACE_FLUSH_N")) {
        const long v = atol(n);
        if (v > 0) {
            s.flush_n = (size_t) v;
        }
    }
    for (int i = 0; i < WP_TPT_MAX_SLOTS; i++) {
        s.ubatch_of_slot[i]   = -1;
        s.subgraph_of_slot[i] = -1;
    }
    s.rows.reserve(s.flush_n + 1024);
    atexit([] { wp_tp_trace_flush(); });
}

} // namespace

bool wp_tp_trace_enabled() {
    static const bool enabled = [] {
        const char * e = getenv("WP_TP_TRACE_FILE");
        return e != nullptr && e[0] != '\0';
    }();
    return enabled;
}

void wp_tp_trace_log_host(int kind, long long ubatch_idx, long long slot, long long subgraph_idx,
                           int device, const char * extra) {
    if (!wp_tp_trace_enabled()) {
        return;
    }
    const int64_t t = now_us();
    std::lock_guard<std::mutex> lock(state().mu);
    init_once_locked_by_caller();

    Row r;
    r.kind         = kind;
    r.ubatch_idx   = ubatch_idx;
    r.slot         = slot;
    r.subgraph_idx = subgraph_idx;
    r.device       = device;
    r.host_t_us    = t;
    if (extra != nullptr) {
        snprintf(r.extra, sizeof(r.extra), "%s", extra);
    }
    state().rows.push_back(r);

    if (kind == WP_TPT_UBATCH_BEGIN && slot >= 0 && slot < WP_TPT_MAX_SLOTS) {
        state().ubatch_of_slot[slot]   = ubatch_idx;
        state().subgraph_of_slot[slot] = -1;
    }
    if (kind == WP_TPT_COMPUTE_SUBMIT_BEGIN && slot >= 0 && slot < WP_TPT_MAX_SLOTS) {
        state().subgraph_of_slot[slot] = subgraph_idx;
    }
    if (state().rows.size() >= state().flush_n) {
        flush_locked();
    }
}

void wp_tp_trace_log_gpu(int kind, long long ubatch_idx, long long slot, long long subgraph_idx,
                          int device, cudaStream_t stream, const char * extra) {
    if (!wp_tp_trace_enabled()) {
        return;
    }
    if (device < 0 || device >= WP_TPT_MAX_DEVICES || stream == nullptr) {
        wp_tp_trace_log_host(kind, ubatch_idx, slot, subgraph_idx, device, extra);
        return;
    }

    std::lock_guard<std::mutex> lock(state().mu);
    init_once_locked_by_caller();
    // events are per-device objects: create/record them with THAT device current
    // (an event created while device 0 is current and recorded on device 1's stream
    // poisons the context: "invalid resource handle" on the next launch)
    ggml_cuda_set_device(device);
    ensure_device_inited_locked(device, stream);

    cudaEvent_t ev = acquire_event_locked(device);
    const int64_t t = now_us();
    if (ev != nullptr && cudaEventRecord(ev, stream) != cudaSuccess) {
        cudaEventDestroy(ev);
        ev = nullptr;
    }

    Row r;
    r.kind         = kind;
    r.ubatch_idx   = ubatch_idx;
    r.slot         = slot;
    r.subgraph_idx = subgraph_idx;
    r.device       = device;
    r.host_t_us    = t;
    r.gpu_event    = ev;
    if (extra != nullptr) {
        snprintf(r.extra, sizeof(r.extra), "%s", extra);
    }
    state().rows.push_back(r);

    if (kind == WP_TPT_COMPUTE_SUBMIT_BEGIN && slot >= 0 && slot < WP_TPT_MAX_SLOTS) {
        state().subgraph_of_slot[slot] = subgraph_idx;
    }
    if (state().rows.size() >= state().flush_n) {
        flush_locked();
    }
}

void wp_tp_trace_note_ubatch(long long slot, long long ubatch_idx) {
    if (!wp_tp_trace_enabled() || slot < 0 || slot >= WP_TPT_MAX_SLOTS) {
        return;
    }
    std::lock_guard<std::mutex> lock(state().mu);
    state().ubatch_of_slot[slot]   = ubatch_idx;
    state().subgraph_of_slot[slot] = -1;
}

void wp_tp_trace_note_subgraph(long long slot, long long subgraph_idx) {
    if (!wp_tp_trace_enabled() || slot < 0 || slot >= WP_TPT_MAX_SLOTS) {
        return;
    }
    std::lock_guard<std::mutex> lock(state().mu);
    state().subgraph_of_slot[slot] = subgraph_idx;
}

long long wp_tp_trace_current_ubatch(long long slot) {
    if (!wp_tp_trace_enabled() || slot < 0 || slot >= WP_TPT_MAX_SLOTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state().mu);
    return state().ubatch_of_slot[slot];
}

long long wp_tp_trace_current_subgraph(long long slot) {
    if (!wp_tp_trace_enabled() || slot < 0 || slot >= WP_TPT_MAX_SLOTS) {
        return -1;
    }
    std::lock_guard<std::mutex> lock(state().mu);
    return state().subgraph_of_slot[slot];
}

void wp_tp_trace_mark(ggml_backend_t backend, int kind, long long ubatch_idx, long long slot,
                       long long subgraph_idx, const char * extra) {
    if (!wp_tp_trace_enabled() || backend == nullptr) {
        return;
    }
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    wp_tp_trace_log_host(kind, ubatch_idx, slot, subgraph_idx, cuda_ctx->device, extra);
}

void wp_tp_trace_mark_global(int kind, long long ubatch_idx, long long slot, long long subgraph_idx,
                              const char * extra) {
    if (!wp_tp_trace_enabled()) {
        return;
    }
    wp_tp_trace_log_host(kind, ubatch_idx, slot, subgraph_idx, -1, extra);
}

void wp_tp_trace_gpu_mark(ggml_backend_t backend, int kind, long long ubatch_idx, long long slot,
                           long long subgraph_idx, const char * extra) {
    if (!wp_tp_trace_enabled() || backend == nullptr) {
        return;
    }
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    auto * cuda_ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    wp_tp_trace_log_gpu(kind, ubatch_idx, slot, subgraph_idx, cuda_ctx->device, cuda_ctx->stream(), extra);
}

void wp_tp_trace_flush() {
    if (!wp_tp_trace_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(state().mu);
    flush_locked();
}
