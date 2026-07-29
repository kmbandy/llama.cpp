#include "wp-gpu-transport.h"

#include "llama-impl.h"  // LLAMA_LOG_*
#include "wp-gpu-runtime.h"

#include "ggml-backend.h"

// Compile-time: the Vulkan bridge declarations are absent from other builds.
#if defined(GGML_USE_VULKAN)
#include "ggml-vulkan.h"
#endif

#include <cstring>

namespace wp {

// Compile-time: Vulkan buffer identification needs the Vulkan backend name.
#if defined(GGML_USE_VULKAN)
static bool is_vulkan_buffer(ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    const char * name = dev ? ggml_backend_dev_name(dev) : nullptr;
    return name != nullptr && std::strncmp(name, GGML_VK_NAME, std::strlen(GGML_VK_NAME)) == 0;
}
#endif

// Ambiguous: CUDA-family types require this guard, but it also encloses Vulkan support.
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)

// --- HIP implementation ----------------------------------------------------

GpuTransport::~GpuTransport() {
    shutdown();
}

bool GpuTransport::init(int device_idx, int n_events, bool async_transfer_stream, ggml_backend_buffer_t buffer) {
    if (initialized_) {
        LLAMA_LOG_WARN("wp::GpuTransport: init called twice — ignoring\n");
        return false;
    }
    if (n_events <= 0) {
        LLAMA_LOG_WARN("wp::GpuTransport::init: n_events must be > 0 (got %d)\n", n_events);
        return false;
    }

    // Compile-time: the runtime Vulkan init route calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_buffer(buffer)) {
        events_.assign(n_events, nullptr);
        free_events_.reserve(n_events);
        for (int i = 0; i < n_events; ++i) {
            free_events_.push_back(i);
        }
        buffer_ = buffer;
        device_idx_ = device_idx;
        initialized_ = true;
        is_vulkan_ = true;
        LLAMA_LOG_INFO("wp::GpuTransport: Vulkan device %d, fence-before-return + %d events ready\n",
                       device_idx, n_events);
        return true;
    }
#endif

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipError_t err = hipSetDevice(device_idx);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::init: hipSetDevice(%d) failed: %s\n",
                       device_idx, hipGetErrorString(err));
        return false;
    }

    hipStream_t s = nullptr;
    bool owns_stream = false;
    if (async_transfer_stream) {
        err = hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
        if (err != hipSuccess) {
            LLAMA_LOG_WARN("wp::GpuTransport::init: hipStreamCreateWithFlags failed: %s\n",
                           hipGetErrorString(err));
            hipSetDevice(prev_device);
            return false;
        }
        owns_stream = true;
    } else {
        // Use hipStreamPerThread (== cudaStreamPerThread under HIP) instead
        // of a freshly-created stream. Reason: ggml-cuda's compute kernels
        // run on the per-thread default stream; H2D copies on a separate
        // stream are not ordered against those kernels unless the async
        // ensure path explicitly inserts hipStreamWaitEvent.
        s = hipStreamPerThread;
    }

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
            if (owns_stream) {
                hipStreamDestroy(s);
            }
            hipSetDevice(prev_device);
            return false;
        }
        events_.push_back((void *) ev);
        free_events_.push_back(i);
    }

    stream_      = (void *) s;
    device_idx_  = device_idx;
    initialized_ = true;
    owns_stream_ = owns_stream;

    hipSetDevice(prev_device);
    LLAMA_LOG_INFO("wp::GpuTransport: device %d, %s stream + %d events ready\n",
                   device_idx, owns_stream_ ? "dedicated" : "per-thread", n_events);
    return true;
}

void GpuTransport::shutdown() {
    if (!initialized_) return;

    // Compile-time: Vulkan event destruction calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        for (void * event : events_) {
            ggml_backend_vk_wp_event_free(event);
        }
        events_.clear();
        free_events_.clear();
        buffer_ = nullptr;
        initialized_ = false;
        is_vulkan_ = false;
        device_idx_ = -1;
        return;
    }
#endif

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipSetDevice(device_idx_);

    if (stream_) {
        hipStreamSynchronize((hipStream_t) stream_);
        if (owns_stream_) {
            hipStreamDestroy((hipStream_t) stream_);
        }
        stream_ = nullptr;
    }
    for (void * ev : events_) {
        if (ev) hipEventDestroy((hipEvent_t) ev);
    }
    events_.clear();
    free_events_.clear();
    initialized_ = false;
    device_idx_  = -1;
    owns_stream_ = false;

    hipSetDevice(prev_device);
}

int GpuTransport::stage_in(void * dst, const void * src_pinned,
                           size_t payload_size, size_t slot_size) {
    int evt_idx = stage_in_async(dst, src_pinned, payload_size, slot_size);
    if (evt_idx < 0) return -1;

    // Compile-time: Vulkan synchronization calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        return synchronize(evt_idx) ? evt_idx : -1;
    }
#endif

    int prev_device = 0;
    hipError_t device_err = hipGetDevice(&prev_device);
    if (device_err != hipSuccess) {
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in hipGetDevice failed: %s\n",
                        hipGetErrorString(device_err));
        return -1;
    }
    device_err = hipSetDevice(device_idx_);
    if (device_err != hipSuccess) {
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in hipSetDevice(%d) failed: %s\n",
                        device_idx_, hipGetErrorString(device_err));
        return -1;
    }

    hipStream_t s  = (hipStream_t) stream_;
    hipError_t err = hipStreamSynchronize(s);
    if (err != hipSuccess) {
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in hipStreamSynchronize failed: stream=%p dst=%p src=%p bytes=%zu err=%s\n",
                        (void *) s, dst, src_pinned, payload_size, hipGetErrorString(err));
        hipSetDevice(prev_device);
        return -1;
    }

    device_err = hipSetDevice(prev_device);
    if (device_err != hipSuccess) {
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in restore hipSetDevice(%d) failed: %s\n",
                        prev_device, hipGetErrorString(device_err));
        return -1;
    }
    return evt_idx;
}

int GpuTransport::stage_in_async(void * dst, const void * src_pinned,
                                 size_t payload_size, size_t slot_size) {
    if (!initialized_ || dst == nullptr || src_pinned == nullptr) return -1;
    if (payload_size > slot_size) return -1;
    if (free_events_.empty()) {
        LLAMA_LOG_WARN("wp::GpuTransport::stage_in_async: event pool exhausted (queue depth too small?)\n");
        return -1;
    }

    // Compile-time: Vulkan staging calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        const int evt_idx = free_events_.back();
        free_events_.pop_back();
        void * event = nullptr;
        // Genuinely async: record the fence and return without waiting. This
        // used to call ggml_backend_vk_wp_event_wait() right here, which made
        // every "async" stage a submit-and-block and collapsed the batch paths
        // to one page in flight at a time. Callers wait via synchronize() (which
        // stage_in() does for them) or poll query().
        if (!ggml_backend_vk_wp_stage_in(buffer_, dst, src_pinned, payload_size, slot_size, &event)) {
            ggml_backend_vk_wp_event_free(event);
            free_events_.push_back(evt_idx);
            return -1;
        }
        events_[evt_idx] = event;
        return evt_idx;
    }
#endif

    int prev_device = 0;
    hipError_t device_err = hipGetDevice(&prev_device);
    if (device_err != hipSuccess) {
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in_async hipGetDevice failed: %s\n",
                        hipGetErrorString(device_err));
        return -1;
    }
    device_err = hipSetDevice(device_idx_);
    if (device_err != hipSuccess) {
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in_async hipSetDevice(%d) failed: %s\n",
                        device_idx_, hipGetErrorString(device_err));
        return -1;
    }

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
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in_async hipMemcpyAsync failed: dst=%p src=%p bytes=%zu stream=%p err=%s\n",
                        dst, src_pinned, payload_size, (void *) s, hipGetErrorString(err));
        hipSetDevice(prev_device);
        return -1;
    }

    if (slot_size > payload_size) {
        err = hipMemsetAsync((char *) dst + payload_size, 0,
                             slot_size - payload_size, s);
        if (err != hipSuccess) {
            LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in_async hipMemsetAsync failed: dst=%p offset=%zu bytes=%zu stream=%p err=%s\n",
                            dst, payload_size, slot_size - payload_size, (void *) s,
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
        LLAMA_LOG_ERROR("[WP_HIP_DIAG] stage_in_async hipEventRecord failed: event=%d stream=%p dst=%p src=%p bytes=%zu err=%s\n",
                        evt_idx, (void *) s, dst, src_pinned, payload_size, hipGetErrorString(err));
        release_event(evt_idx);
        hipSetDevice(prev_device);
        return -1;
    }

    hipSetDevice(prev_device);
    return evt_idx;
}

bool GpuTransport::read_to_host(void * dst_host, const void * src_device, size_t n) {
    if (dst_host == nullptr || src_device == nullptr || n == 0) {
        return false;
    }
    // Compile-time: Vulkan readback calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        return ggml_backend_vk_wp_read(buffer_, src_device, dst_host, n);
    }
#endif
    // Compile-time: raw D2H is unavailable without a CUDA-family runtime.
#if defined(GGML_USE_HIP) || defined(GGML_USE_CUDA)
    hipError_t err = hipMemcpy(dst_host, src_device, n, hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::read_to_host: D2H(%zu) failed: %s\n",
                       n, hipGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

void * GpuTransport::host_alloc(size_t size) {
    // Compile-time: registered Vulkan host allocation calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        return ggml_backend_vk_wp_host_alloc(buffer_, size);
    }
#endif
    (void) size;
    return nullptr;
}

void GpuTransport::host_free(void * ptr) {
    // Compile-time: registered Vulkan host free calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        ggml_backend_vk_wp_host_free(buffer_, ptr);
        return;
    }
#endif
    (void) ptr;
}

bool GpuTransport::wait_event_on_stream(int event_handle, void * stream) {
    // Compile-time: Vulkan has no CUDA-family stream-wait event.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        return false;
    }
#endif
    if (!initialized_ || stream == nullptr) return false;
    if (event_handle < 0 || event_handle >= (int) events_.size()) return false;
    hipEvent_t ev = (hipEvent_t) events_[event_handle];
    if (ev == nullptr) return false;

    int prev_device = 0;
    hipGetDevice(&prev_device);
    hipSetDevice(device_idx_);

    hipError_t err = hipStreamWaitEvent((hipStream_t) stream, ev, 0);

    hipSetDevice(prev_device);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::GpuTransport::wait_event_on_stream: hipStreamWaitEvent failed: %s\n",
                       hipGetErrorString(err));
        return false;
    }
    return true;
}

bool GpuTransport::query(int event_handle) const {
    // Compile-time: Vulkan event polling calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        return event_handle >= 0 && event_handle < (int) events_.size() &&
               events_[event_handle] != nullptr && ggml_backend_vk_wp_event_query(events_[event_handle]);
    }
#endif
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
    // Compile-time: Vulkan event waiting calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        return event_handle >= 0 && event_handle < (int) events_.size() &&
               events_[event_handle] != nullptr && ggml_backend_vk_wp_event_wait(events_[event_handle]);
    }
#endif
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
    // Compile-time: Vulkan event release calls the Vulkan bridge.
#if defined(GGML_USE_VULKAN)
    if (is_vulkan_) {
        if (events_[event_handle] == nullptr) return;
        ggml_backend_vk_wp_event_free(events_[event_handle]);
        events_[event_handle] = nullptr;
        free_events_.push_back(event_handle);
        return;
    }
#endif
    // Guard against double-free.
    for (int idx : free_events_) {
        if (idx == event_handle) return;
    }
    free_events_.push_back(event_handle);
}

#else  // !GGML_USE_HIP && !GGML_USE_CUDA

// --- Stub for non-HIP builds ----------------------------------------------
//
// Phase 1 is HIP-only. CUDA support shares the same logic with cuda*
// rather than hip* names but is deferred until a CUDA test environment
// is available. On non-HIP builds the transport silently fails its init,
// and every method is a no-op returning failure.

GpuTransport::~GpuTransport() = default;

bool GpuTransport::init(int /*device_idx*/, int /*n_events*/, bool /*async_transfer_stream*/,
                        ggml_backend_buffer_t /*buffer*/) {
    LLAMA_LOG_WARN("wp::GpuTransport: HIP support not compiled in; transport disabled\n");
    return false;
}

void GpuTransport::shutdown() {}

int  GpuTransport::stage_in(void * /*dst*/, const void * /*src_pinned*/,
                            size_t /*payload_size*/, size_t /*slot_size*/) { return -1; }
int  GpuTransport::stage_in_async(void * /*dst*/, const void * /*src_pinned*/,
                                  size_t /*payload_size*/, size_t /*slot_size*/) { return -1; }
// Declared in the header and called from wp-pager, but defined only inside the
// HIP/CUDA block above -- without these the CPU-only build fails to link
// libllama.so. Fail closed: no host staging buffer, no device readback.
void * GpuTransport::host_alloc(size_t /*size*/)                           { return nullptr; }
void   GpuTransport::host_free(void * /*ptr*/)                             {}
bool   GpuTransport::read_to_host(void * /*dst_host*/, const void * /*src_device*/,
                                  size_t /*n*/)                            { return false; }

bool GpuTransport::wait_event_on_stream(int /*event_handle*/, void * /*stream*/) { return false; }
bool GpuTransport::query(int /*event_handle*/) const                       { return false; }
bool GpuTransport::synchronize(int /*event_handle*/)                       { return false; }
void GpuTransport::release_event(int /*event_handle*/)                     {}

#endif  // GGML_USE_HIP || GGML_USE_CUDA

}  // namespace wp
