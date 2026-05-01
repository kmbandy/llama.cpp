#pragma once

// GpuTransport — host-pinned → VRAM transfer with completion fences.
//
// One transport per device. Owns:
//   - a dedicated transfer stream (so transfers can overlap with compute
//     on the default stream),
//   - a recyclable pool of completion events.
//
// stage_in() issues the H2D copy AND the padding-zero (B-P2) AND the
// completion event in one call. Callers cannot accidentally skip the
// padding zero — block-quantized kernels rely on that tail being zero
// for safety past the tensor's payload size.
//
// HIP-only in Phase 1. CUDA support is the same pattern with cuda* names
// and is deferred. On non-HIP builds, init() returns false and every
// other method is a no-op returning failure.

#include <cstddef>
#include <vector>

namespace wp {

class GpuTransport {
public:
    GpuTransport() = default;
    ~GpuTransport();

    GpuTransport(const GpuTransport &)             = delete;
    GpuTransport & operator=(const GpuTransport &) = delete;

    // Initialise the transfer stream and pre-allocate `n_events` completion
    // events. n_events should be at least the prefetch queue depth so that
    // every in-flight stage_in has its own event without blocking on event
    // recycling. Returns false on HIP error or non-HIP builds.
    bool init(int device_idx, int n_events);

    // Tear down the stream and events. Safe to call multiple times.
    // Called automatically by the destructor.
    void shutdown();

    // Issue the staging-to-VRAM copy. After this returns successfully:
    //   - hipMemcpyAsync(dst, src_pinned, payload_size, H2D) has been queued
    //     on the transfer stream;
    //   - hipMemsetAsync(dst + payload_size, 0, slot_size - payload_size)
    //     has been queued on the same stream (B-P2 padding zero);
    //   - a recyclable event has been recorded after both ops.
    //
    // Returns the event handle (>= 0) for query()/synchronize()/release_event().
    // Returns -1 on any failure (transport not init'd, no free events, HIP error).
    //
    // NOTE: payload_size must be <= slot_size. dst must point into the
    // PoolAllocator's backing buffer for the device this transport was
    // init'd on. src_pinned must be host memory; ideally allocated with
    // hipHostMalloc for DMA performance.
    int stage_in(void * dst, const void * src_pinned,
                 size_t payload_size, size_t slot_size);

    // Non-blocking: returns true if the event has signalled, false if not
    // yet or on error. Safe to call repeatedly.
    bool query(int event_handle) const;

    // Blocks until the event signals or HIP returns an error.
    // Returns true on success, false on error or invalid handle.
    bool synchronize(int event_handle);

    // Return the event handle to the recyclable pool. After this call the
    // handle is invalid; subsequent query/synchronize calls return false.
    void release_event(int event_handle);

    // Inspect transport state.
    bool is_initialized() const { return initialized_; }
    int  device_idx()     const { return device_idx_; }
    int  n_events()       const { return (int) events_.size(); }
    int  n_free_events()  const { return (int) free_events_.size(); }

private:
    int                  device_idx_   = -1;
    void *               stream_       = nullptr;  // hipStream_t under HIP
    std::vector<void *>  events_;                  // hipEvent_t under HIP
    std::vector<int>     free_events_;             // free-list of event indices
    bool                 initialized_  = false;
};

}  // namespace wp
