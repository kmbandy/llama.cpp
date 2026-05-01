#include "wp-prefetch.h"

#include "wp-gpu-transport.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <chrono>
#include <cstdlib>
#include <cstring>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace wp {

// ---------------------------------------------------------------------------
// pinned host staging — small wrapper that uses hipHostMalloc on HIP and
// plain malloc elsewhere (in which case the layer can still function for
// SyncPread + a stub GpuTransport, though the latter currently fails init
// on non-HIP builds).
// ---------------------------------------------------------------------------

namespace {

void * alloc_pinned(size_t bytes) {
    if (bytes == 0) return nullptr;
#if defined(GGML_USE_HIP)
    void * p = nullptr;
    hipError_t err = hipHostMalloc(&p, bytes, hipHostMallocDefault);
    if (err != hipSuccess) {
        LLAMA_LOG_WARN("wp::alloc_pinned: hipHostMalloc(%zu) failed: %s — falling back to malloc\n",
                       bytes, hipGetErrorString(err));
        return std::malloc(bytes);
    }
    return p;
#else
    return std::malloc(bytes);
#endif
}

void free_pinned(void * p) {
    if (p == nullptr) return;
#if defined(GGML_USE_HIP)
    // hipHostFree on a malloc'd ptr is undefined. We pessimistically try
    // hipHostFree first; if it fails we leak (rare — only on the fallback
    // path). To avoid this, callers should track which alloc path was used.
    // For Phase 1b correctness we accept the leak risk on the fallback.
    if (hipHostFree(p) != hipSuccess) {
        std::free(p);
    }
#else
    std::free(p);
#endif
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// PrefetchScheduler
// ---------------------------------------------------------------------------

PrefetchScheduler::~PrefetchScheduler() {
    shutdown();
}

bool PrefetchScheduler::init(FileIOLayer * file_io, GpuTransport * gpu,
                             size_t max_page_size, int queue_depth) {
    if (initialized_) {
        LLAMA_LOG_WARN("wp::PrefetchScheduler: init called twice\n");
        return false;
    }
    if (file_io == nullptr || gpu == nullptr || max_page_size == 0 || queue_depth <= 0) {
        LLAMA_LOG_WARN("wp::PrefetchScheduler::init: invalid args\n");
        return false;
    }

    file_io_       = file_io;
    gpu_           = gpu;
    max_page_size_ = max_page_size;
    queue_depth_   = queue_depth;

    slots_.assign((size_t) queue_depth, Slot{});
    staging_.assign((size_t) queue_depth, nullptr);
    free_slots_.clear();
    free_slots_.reserve(queue_depth);

    for (int i = 0; i < queue_depth; ++i) {
        staging_[i] = alloc_pinned(max_page_size);
        if (staging_[i] == nullptr) {
            LLAMA_LOG_WARN("wp::PrefetchScheduler::init: alloc_pinned[%d] failed\n", i);
            // Tear down what we built.
            for (int j = 0; j < i; ++j) {
                free_pinned(staging_[j]);
                staging_[j] = nullptr;
            }
            slots_.clear();
            staging_.clear();
            return false;
        }
        free_slots_.push_back(i);
    }

    initialized_ = true;
    LLAMA_LOG_INFO("wp::PrefetchScheduler: queue_depth=%d, max_page_size=%zu (%.1f MiB pinned per slot)\n",
                   queue_depth, max_page_size, max_page_size / 1048576.0);
    return true;
}

void PrefetchScheduler::shutdown() {
    if (!initialized_) {
        // Even if init failed, free any partial staging.
        for (void * p : staging_) free_pinned(p);
        staging_.clear();
        slots_.clear();
        return;
    }

    // Drain in-flight to avoid leaving io_uring SQEs / GPU events orphaned.
    drain();

    for (void * p : staging_) free_pinned(p);
    staging_.clear();
    slots_.clear();
    free_slots_.clear();
    page_to_slot_.clear();
    req_to_slot_.clear();

    file_io_       = nullptr;
    gpu_           = nullptr;
    max_page_size_ = 0;
    queue_depth_   = 0;
    initialized_   = false;
}

int PrefetchScheduler::alloc_slot_() {
    if (free_slots_.empty()) return -1;
    int h = free_slots_.back();
    free_slots_.pop_back();
    return h;
}

void PrefetchScheduler::release_slot_(int handle) {
    if (handle < 0 || handle >= queue_depth_) return;
    Slot & s = slots_[handle];
    if (s.state == State::Free) return;  // already released

    if (s.gpu_event >= 0 && gpu_) {
        gpu_->release_event(s.gpu_event);
    }
    if (s.req_id != 0) {
        req_to_slot_.erase(s.req_id);
    }
    if (s.page_idx >= 0) {
        page_to_slot_.erase(s.page_idx);
    }
    s = Slot{};
    free_slots_.push_back(handle);
}

bool PrefetchScheduler::submit(int page_idx, int fd_idx, uint64_t file_offset,
                               size_t payload_size, void * dst_vram, size_t slot_size) {
    if (!initialized_)                                        return false;
    if (page_idx < 0 || dst_vram == nullptr)                  return false;
    if (payload_size == 0 || payload_size > max_page_size_)   return false;
    if (slot_size < payload_size)                             return false;
    if (page_to_slot_.find(page_idx) != page_to_slot_.end())  return false;  // already in flight

    int handle = alloc_slot_();
    if (handle < 0) return false;  // queue full

    Slot & s = slots_[handle];
    s.state        = State::Submitted;
    s.page_idx     = page_idx;
    s.req_id       = next_req_id_++;
    s.payload_size = payload_size;
    s.slot_size    = slot_size;
    s.dst_vram     = dst_vram;
    s.gpu_event    = -1;

    if (!file_io_->submit(s.req_id, fd_idx, file_offset, payload_size, staging_[handle])) {
        // FileIOLayer rejected. Release the slot and report failure.
        s.state = State::Free;
        free_slots_.push_back(handle);
        return false;
    }

    page_to_slot_[page_idx] = handle;
    req_to_slot_[s.req_id]  = handle;

    // Submit may batch in io_uring; flush now to make the read live.
    file_io_->flush();
    return true;
}

void PrefetchScheduler::process_io_(const IoResult & r) {
    auto it = req_to_slot_.find(r.req_id);
    if (it == req_to_slot_.end()) return;  // unknown req — stale or already reaped
    int handle = it->second;
    Slot & s   = slots_[handle];

    req_to_slot_.erase(it);
    s.req_id = 0;

    if (r.status == IoStatus::Ok && r.bytes_read == (int) s.payload_size) {
        s.state = State::Stage1Done;
    } else {
        LLAMA_LOG_WARN("wp::PrefetchScheduler: stage 1 failed for page %d (status=%d, bytes_read=%d, expected=%zu)\n",
                       s.page_idx, (int) r.status, r.bytes_read, s.payload_size);
        s.state = State::Failed;
    }
}

void PrefetchScheduler::promote_stage2_() {
    for (int h = 0; h < queue_depth_; ++h) {
        Slot & s = slots_[h];
        if (s.state != State::Stage1Done) continue;

        int evt = gpu_->stage_in(s.dst_vram, staging_[h], s.payload_size, s.slot_size);
        if (evt < 0) {
            LLAMA_LOG_WARN("wp::PrefetchScheduler: stage 2 stage_in failed for page %d\n", s.page_idx);
            s.state = State::Failed;
            continue;
        }
        s.gpu_event = evt;
        s.state     = State::Stage2Running;
    }
}

void PrefetchScheduler::poll_stage2_() {
    for (int h = 0; h < queue_depth_; ++h) {
        Slot & s = slots_[h];
        if (s.state != State::Stage2Running) continue;
        if (gpu_->query(s.gpu_event)) {
            gpu_->release_event(s.gpu_event);
            s.gpu_event = -1;
            s.state     = State::Done;
        }
    }
}

void PrefetchScheduler::tick() {
    if (!initialized_) return;

    // 1. Drain any completed file reads (non-blocking).
    while (file_io_->pending() > 0) {
        IoResult r = file_io_->wait_any(0);
        if (r.status == IoStatus::Timeout) break;
        process_io_(r);
    }
    // 2. Promote any newly-ready stage 1 to stage 2.
    promote_stage2_();
    // 3. Reap signalled GPU events.
    poll_stage2_();
}

bool PrefetchScheduler::wait_for(int page_idx, int timeout_ms) {
    if (!initialized_) return false;
    auto it = page_to_slot_.find(page_idx);
    if (it == page_to_slot_.end()) return false;
    int handle = it->second;
    Slot & s   = slots_[handle];

    using clock = std::chrono::steady_clock;
    const bool          have_deadline = (timeout_ms >= 0);
    clock::time_point   deadline      = have_deadline
        ? clock::now() + std::chrono::milliseconds(timeout_ms)
        : clock::time_point{};

    while (true) {
        if (s.state == State::Done)   return true;
        if (s.state == State::Failed) return false;

        // Compute remaining time budget for any blocking call below.
        int remaining_ms;
        if (!have_deadline) {
            remaining_ms = -1;
        } else {
            const auto now = clock::now();
            if (now >= deadline) return false;
            remaining_ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - now).count();
        }

        if (s.state == State::Submitted) {
            // Block on the file IO layer for this specific request.
            // wait_any may return a different req — process it and loop.
            IoResult r = file_io_->wait_any(remaining_ms);
            if (r.status == IoStatus::Timeout) return false;
            process_io_(r);
            continue;
        }
        if (s.state == State::Stage1Done) {
            promote_stage2_();
            continue;
        }
        if (s.state == State::Stage2Running) {
            // Block on the specific event.
            if (!gpu_->synchronize(s.gpu_event)) {
                s.state = State::Failed;
                return false;
            }
            poll_stage2_();
            continue;
        }
        // State::Free — request was reaped while we waited (shouldn't happen
        // on a single-threaded eval callback path). Treat as miss.
        return false;
    }
}

bool PrefetchScheduler::is_loaded(int page_idx) const {
    auto it = page_to_slot_.find(page_idx);
    if (it == page_to_slot_.end()) return false;
    return slots_[it->second].state == State::Done;
}

void PrefetchScheduler::reap(int page_idx) {
    auto it = page_to_slot_.find(page_idx);
    if (it == page_to_slot_.end()) return;
    release_slot_(it->second);
}

void PrefetchScheduler::drain() {
    if (!initialized_) return;

    // First reap anything already complete.
    tick();

    // Then block on each remaining in-flight slot.
    for (int h = 0; h < queue_depth_; ++h) {
        Slot & s = slots_[h];
        if (s.state == State::Free || s.state == State::Done || s.state == State::Failed) continue;
        // Use the page-keyed wait so we go through the same state machine.
        const int page = s.page_idx;
        wait_for(page, /*timeout_ms=*/-1);
    }

    // Release any successfully-completed slots so they don't linger.
    // Failed slots also get released so callers don't see stale entries.
    for (int h = 0; h < queue_depth_; ++h) {
        Slot & s = slots_[h];
        if (s.state == State::Done || s.state == State::Failed) {
            release_slot_(h);
        }
    }
}

}  // namespace wp
