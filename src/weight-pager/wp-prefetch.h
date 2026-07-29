#pragma once

// PrefetchScheduler — two-stage async pipeline for paging weights into VRAM.
//
// Stage 1: NVMe -> pinned host (via FileIOLayer; io_uring or pread).
// Stage 2: pinned host -> VRAM (via GpuTransport; hipMemcpyAsync + event).
//
// Each in-flight prefetch occupies one of `queue_depth` slots; each slot
// has its own pinned host staging buffer. The state machine progresses
// strictly forward (SUBMITTED -> STAGE1_DONE -> STAGE2_RUNNING -> DONE)
// and is driven by tick() (non-blocking) or wait_for() (blocking on a
// specific page).
//
// Replaces the previous pager's broken s_last_processed_page static and
// the page-index keying of io_uring user_data (B-P6). Internally we use
// a monotonic req_id for FileIOLayer routing; page_idx is the
// caller-facing handle.

#include "wp-file-io.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace wp {

class GpuTransport;

// MAD-235 — one prefetch request in a batch. Mirror of submit()'s args.
struct PrefetchBatchRequest {
    int      page_idx;
    int      fd_idx;
    uint64_t file_offset;
    size_t   payload_size;
    void *   dst_vram;
    size_t   slot_size;
};

class PrefetchScheduler {
public:
    PrefetchScheduler() = default;
    ~PrefetchScheduler();

    PrefetchScheduler(const PrefetchScheduler &)             = delete;
    PrefetchScheduler & operator=(const PrefetchScheduler &) = delete;

    // Bind to an already-initialised FileIOLayer + GpuTransport. Allocates
    // queue_depth pinned host staging buffers of max_page_size each.
    // Returns false on allocation failure or invalid args.
    bool init(FileIOLayer * file_io, GpuTransport * gpu,
              size_t max_page_size, int queue_depth,
              bool async_stage2 = false);

    // Drain in-flight requests, free pinned buffers. Safe to call multiple
    // times. Called automatically by the destructor.
    void shutdown();

    // Submit a prefetch. The page is identified by the caller's page_idx,
    // which is opaque to the scheduler — the caller maintains its own
    // mapping from name to page_idx (typically the PageCatalog).
    //
    // Returns false if:
    //   - the scheduler is not init'd,
    //   - all queue slots are in use,
    //   - page_idx is already in flight (resubmission is rejected),
    //   - any FileIOLayer submit failure.
    bool submit(int page_idx,
                int fd_idx,
                uint64_t file_offset,
                size_t payload_size,
                void * dst_vram,
                size_t slot_size);

    // MAD-235 — batch-submit a vector of prefetches. Atomic semantics:
    // either all reqs are queued (and one io_uring_submit covers them all
    // via FileIOLayer::submit_batch) or none are (and the function returns
    // false leaving scheduler state unchanged). The all-or-nothing contract
    // lets the caller fall back to per-page submit cleanly when the
    // queue can't fit the whole batch — no half-queued mess to unwind.
    //
    // Returns true iff every request was queued.
    //
    // Reasons for failure (returns false, no state change):
    //   - scheduler not initialized
    //   - any req has invalid args (dst null, payload 0 or > max_page_size,
    //     slot_size < payload_size, page_idx negative)
    //   - any page_idx is already in flight
    //   - reqs.size() > free queue slots
    //   - FileIOLayer::submit_batch returns less than reqs.size()
    bool submit_batch(const std::vector<PrefetchBatchRequest> & reqs);

    // Non-blocking: drive the pipeline forward by reaping any completed
    // file reads and any signalled GPU events. Idempotent — safe to call
    // anywhere in the eval callback.
    void tick();

    // Block until the page reaches DONE or FAILED. timeout_ms < 0 = wait
    // indefinitely. Returns true iff DONE.
    bool wait_for(int page_idx, int timeout_ms = -1);

    // Non-blocking: is the page DONE? (page must still be in flight; once
    // reap()ed it returns false.)
    bool is_loaded(int page_idx) const;

    // Non-blocking: page still has a scheduler slot (any state including Done).
    bool has_page(int page_idx) const;

    // Non-blocking: page slot is Failed (caller should tear down pool reservation).
    bool is_failed(int page_idx) const;

    // Release every Done/Failed scheduler slot. WeightPager must commit Done
    // pages to page_loaded_ *before* calling this, or call harvest first.
    // Frees free_queue_slots that were stuck holding completed prefetches.
    int reap_finished();

    // If stage 2 is already running, detach and return its GPU completion
    // event. The caller owns the event and must eventually release it via
    // GpuTransport. Returns -1 if the page has no stage-2 event yet.
    int take_stage2_event(int page_idx);

    // Release all internal state for `page_idx`. After this, queries for
    // page_idx return false. Called by the WeightPager once the page has
    // been successfully consumed (i.e., the eval callback redirected
    // tensor->data to the slot).
    void reap(int page_idx);

    // Block until all in-flight requests reach DONE or FAILED. Used by
    // shutdown and by tests.
    void drain();

    // Inspect.
    int  pending()      const { return queue_depth_ - (int) free_slots_.size(); }
    int  queue_depth()  const { return queue_depth_; }
    // Scheduler queue slots that can still accept a submit without rejection.
    // WeightPager must consult this *before* pool alloc_slot so speculative
    // prefetch cannot LRU-evict resident pages for work the SQ cannot take.
    int  free_queue_slots() const { return (int) free_slots_.size(); }
    bool is_initialized() const { return initialized_; }

private:
    enum class HostAllocKind : uint8_t {
        Malloc,
        Backend,
        Transport,
    };

    enum class State : uint8_t {
        Free,
        Submitted,       // file IO submitted, waiting for stage-1 completion
        Stage1Done,      // file IO complete, ready for stage 2
        Stage2Running,   // hipMemcpyAsync in flight, gpu_event recorded
        Done,            // both stages complete
        Failed,          // either stage errored; caller will treat as miss
    };

    struct Slot {
        State    state           = State::Free;
        int      page_idx        = -1;
        uint64_t req_id          = 0;
        size_t   payload_size    = 0;
        size_t   slot_size       = 0;
        void *   dst_vram        = nullptr;
        int      gpu_event       = -1;
        bool     direct_to_device = false;
    };

    int  alloc_slot_();
    void release_slot_(int handle);
    void process_io_(const IoResult & r);
    void promote_stage2_();
    void poll_stage2_();

    FileIOLayer *  file_io_       = nullptr;
    GpuTransport * gpu_           = nullptr;
    size_t         max_page_size_ = 0;
    int            queue_depth_   = 0;
    bool           initialized_   = false;
    bool           async_stage2_  = false;

    std::vector<Slot>   slots_;
    std::vector<void *> staging_;          // pinned host buffer per slot
    std::vector<HostAllocKind> staging_alloc_kinds_;
    std::vector<int>    free_slots_;       // free-list of slot handles

    uint64_t            next_req_id_ = 1;  // 0 reserved for "no request"

    std::unordered_map<int, int>      page_to_slot_;
    std::unordered_map<uint64_t, int> req_to_slot_;

    // req_ids of reads whose slots were rolled back on a partial submit_batch
    // while the file read was already in flight. tick() reaps-and-drops them
    // from the shared FileIOLayer demux buffer once they land, so they don't
    // accumulate unclaimed. Normally empty (pre-flight makes partials rare).
    std::vector<uint64_t> abandoned_reqs_;
};

}  // namespace wp
