#include "wp-gpu-transport.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace wp {

#if defined(GGML_USE_HIP)

// --- HIP implementation ----------------------------------------------------

GpuTransport::~GpuTransport() {
    shutdown();
}

bool GpuTransport::init(int device_idx, int n_events) {
    if (initialized_) {
        LLAMA_LOG_WARN("wp::GpuTransport: init called twice — ignoring\n");
        return false;
    }
    if (n_events <= 0) {
        LLAMA_LOG_WARN("wp::GpuTransport::init: n_events must be > 0 (got %d)\n", n_events);
        return false;
    }

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipError_t err = hipSetDevice(device_idx);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::init: hipSetDevice(%d) failed: %s\n",
                       device_idx, hipGetErrorString(err));
        return false;
    }

    // Use hipStreamPerThread (== cudaStreamPerThread under HIP) instead of
    // a freshly-created stream. Reason: ggml-cuda's compute kernels run on
    // the per-thread default stream; H2D copies on a separate stream are
    // not ordered against those kernels and can overwrite VRAM that an
    // in-flight kernel is still reading. Phase 1e may revisit this with
    // explicit hipStreamWaitEvent to recover transfer/compute overlap.
    hipStream_t s = hipStreamPerThread;
    (void) err;

    events_.reserve(n_events);
    free_events_.reserve(n_events);
    for (int i = 0; i < n_events; ++i) {
        hipEvent_t ev = nullptr;
        // hipEventDisableTiming reduces overhead since we only care about
        // signalling, not timestamps.
        err = hipEventCreateWithFlags(&ev, hipEventDisableTiming);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("wp::GpuTransport::init: hipEventCreate[%d] failed: %s\n",
                           i, hipGetErrorString(err));
            // Tear down what we already built.
            for (void * e : events_) {
                hipEventDestroy((hipEvent_t) e);
            }
            events_.clear();
            free_events_.clear();
            hipStreamDestroy(s);
            hipSetDevice(prev_device);
            return false;
        }
        events_.push_back((void *) ev);
        free_events_.push_back(i);
    }

    stream_      = (void *) s;
    device_idx_  = device_idx;
    initialized_ = true;

    hipSetDevice(prev_device);
    LLAMA_LOG_INFO("wp::GpuTransport: device %d, stream + %d events ready\n",
                   device_idx, n_events);
    return true;
}

void GpuTransport::shutdown() {
    if (!initialized_) return;

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipSetDevice(device_idx_);

    if (stream_) {
        // We don't own hipStreamPerThread; just sync, don't destroy.
        hipStreamSynchronize((hipStream_t) stream_);
        stream_ = nullptr;
    }
    for (void * ev : events_) {
        if (ev) hipEventDestroy((hipEvent_t) ev);
    }
    events_.clear();
    free_events_.clear();
    initialized_ = false;
    device_idx_  = -1;

    hipSetDevice(prev_device);
}

int GpuTransport::stage_in(void * dst, const void * src_pinned,
                           size_t payload_size, size_t slot_size) {
    if (!initialized_ || dst == nullptr || src_pinned == nullptr) return -1;
    if (payload_size > slot_size) return -1;
    if (free_events_.empty()) {
        LLAMA_LOG_WARN("wp::GpuTransport::stage_in: event pool exhausted (queue depth too small?)\n");
        return -1;
    }

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipSetDevice(device_idx_);

    hipStream_t s  = (hipStream_t) stream_;
    hipError_t  err;

    // MAD-230 follow-up: use async memcpy + memset on this transport's own
    // stream, then synchronize that stream at the end to preserve the
    // "data is in VRAM on return" contract page_in_sync_ relies on. The
    // previous implementation used synchronous hipMemcpy on the default
    // stream, which on AMD HIP does NOT auto-serialize with GGML's
    // non-blocking compute stream (common.cuh:1439). That created a
    // torn-write race against MMQ kernels reading the same slot AND
    // contributed to compute/graphics ring scheduling pressure that
    // wedged the display GPU under MoE-decode load. Stream-scoped sync
    // is bounded to this stream's work, doesn't stall the device, and
    // keeps the host blocked only as long as the actual transfer takes.
    err = hipMemcpyAsync(dst, src_pinned, payload_size, hipMemcpyHostToDevice, s);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::stage_in: hipMemcpyAsync failed: %s\n",
                       hipGetErrorString(err));
        hipSetDevice(prev_device);
        return -1;
    }

    if (slot_size > payload_size) {
        err = hipMemsetAsync((char *) dst + payload_size, 0,
                             slot_size - payload_size, s);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("wp::GpuTransport::stage_in: hipMemsetAsync (padding) failed: %s\n",
                           hipGetErrorString(err));
            // Best-effort: don't fail the whole call.
        }
    }

    // Record the completion event on the transport stream BEFORE we
    // synchronize so a future async-aware caller can hipStreamWaitEvent
    // on it from another stream (e.g., have the GGML compute stream wait
    // on the transport event instead of blocking the CPU). For now
    // page_in_sync_ uses the simpler model: we sync here and the caller
    // immediately release_event()s the handle. Phase 1e can flip this
    // to truly pipelined behaviour without changing this function's
    // signature.
    int evt_idx = free_events_.back();
    free_events_.pop_back();
    hipEvent_t ev = (hipEvent_t) events_[evt_idx];
    err = hipEventRecord(ev, s);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::stage_in: hipEventRecord failed: %s\n",
                       hipGetErrorString(err));
        // Continue — we still need to sync to preserve the in-VRAM-on-return
        // contract; the event is just a handle for future async wiring.
    }

    // Preserve the "data is in VRAM when this returns" contract that
    // page_in_sync_ relies on. Stream-scoped — does NOT stall other
    // streams (compute, graphics) on the device.
    err = hipStreamSynchronize(s);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::stage_in: hipStreamSynchronize failed: %s\n",
                       hipGetErrorString(err));
        hipSetDevice(prev_device);
        return -1;
    }

    hipSetDevice(prev_device);
    return evt_idx;
}

bool GpuTransport::query(int event_handle) const {
    if (!initialized_ || event_handle < 0 || event_handle >= (int) events_.size()) return false;
    hipEvent_t ev = (hipEvent_t) events_[event_handle];
    if (ev == nullptr) return false;

    hipError_t st = hipEventQuery(ev);
    if (st == hipSuccess)         return true;
    if (st == hipErrorNotReady)   return false;

    // Any other error: treat as not signalled. Caller can promote to
    // synchronize() to surface the error if needed.
    return false;
}

bool GpuTransport::synchronize(int event_handle) {
    if (!initialized_ || event_handle < 0 || event_handle >= (int) events_.size()) return false;
    hipEvent_t ev = (hipEvent_t) events_[event_handle];
    if (ev == nullptr) return false;

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipSetDevice(device_idx_);

    hipError_t err = hipEventSynchronize(ev);

    hipSetDevice(prev_device);
    return err == hipSuccess;
}

void GpuTransport::release_event(int event_handle) {
    if (!initialized_ || event_handle < 0 || event_handle >= (int) events_.size()) return;
    // Guard against double-free.
    for (int idx : free_events_) {
        if (idx == event_handle) return;
    }
    free_events_.push_back(event_handle);
}

#else  // !GGML_USE_HIP

// --- Stub for non-HIP builds ----------------------------------------------
//
// Phase 1 is HIP-only. CUDA support shares the same logic with cuda*
// rather than hip* names but is deferred until a CUDA test environment
// is available. On non-HIP builds the transport silently fails its init,
// and every method is a no-op returning failure.

GpuTransport::~GpuTransport() = default;

bool GpuTransport::init(int /*device_idx*/, int /*n_events*/) {
    LLAMA_LOG_WARN("wp::GpuTransport: HIP support not compiled in; transport disabled\n");
    return false;
}

void GpuTransport::shutdown() {}

int  GpuTransport::stage_in(void * /*dst*/, const void * /*src_pinned*/,
                            size_t /*payload_size*/, size_t /*slot_size*/) { return -1; }
bool GpuTransport::query(int /*event_handle*/) const                       { return false; }
bool GpuTransport::synchronize(int /*event_handle*/)                       { return false; }
void GpuTransport::release_event(int /*event_handle*/)                     {}

#endif  // GGML_USE_HIP

}  // namespace wp
