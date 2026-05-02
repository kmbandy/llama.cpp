#include "wp-pager.h"

#include "ggml-backend.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <cstdlib>      // getenv, setenv, unsetenv, malloc, free
#include <cstring>
#include <new>          // placement new
#include <unistd.h>     // close()

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#endif

namespace wp {

// ---------------------------------------------------------------------------
// EnvSnapshot — record current value of an env var, then restore it later.
//
// Used for GGML_CUDA_DISABLE_GRAPHS: the pager forces it to "1" during init
// (the eval callback's per-step pointer rewrites are incompatible with
// hipGraph capture, which bakes pointers at capture time). On shutdown we
// restore whatever value (or absence) the user had before. This fixes
// B-P5: the previous pager set the var unconditionally and never restored
// it, leaking the "graphs disabled" state into any subsequent model load
// in the same process.
// ---------------------------------------------------------------------------

namespace {

void env_snapshot(const char * var, bool & present_out, std::string & prior_out) {
    const char * v = std::getenv(var);
    if (v != nullptr) {
        present_out = true;
        prior_out   = v;
    } else {
        present_out = false;
        prior_out.clear();
    }
}

void env_restore(const char * var, bool present, const std::string & prior) {
    if (present) {
        setenv(var, prior.c_str(), /*overwrite=*/1);
    } else {
        unsetenv(var);
    }
}

constexpr const char * kEnvDisableGraphs = "GGML_CUDA_DISABLE_GRAPHS";

}  // anonymous namespace

// ---------------------------------------------------------------------------
// WeightPager
// ---------------------------------------------------------------------------

WeightPager::~WeightPager() {
    shutdown();
}

int WeightPager::add_page(const std::string & name, uint16_t file_idx,
                          uint64_t file_offset, size_t size, int n_experts) {
    // Non-MoE / per-expert tensor: add as-is.
    if (n_experts <= 1) {
        return catalog_.add(name, file_idx, file_offset, size);
    }
    // Consolidated MoE tensor: register N sub-pages, one per expert.
    // Returns the index of the FIRST sub-page (subsequent experts are at
    // sequential indices). Per-expert size is the consolidated size
    // divided by n_experts; per-expert offset is base_offset + e * size_e.
    return catalog_.add_consolidated_experts(name, file_idx, file_offset, size, n_experts);
}

bool WeightPager::init(const Config &             cfg,
                       ggml_backend_buffer_type_t device_buft,
                       int                        device_idx,
                       std::vector<int>           fds,
                       const std::vector<int> &   devices_used) {
    if (initialized_) {
        LLAMA_LOG_WARN("wp::WeightPager: init called twice — ignoring\n");
        return false;
    }
    if (catalog_.size() == 0) {
        LLAMA_LOG_WARN("wp::WeightPager::init: no pages registered\n");
        return false;
    }

    // Multi-GPU guard (B-P7). Phase 1 is single-device by explicit design.
    // This is the *only* defence — the rest of the pager assumes a single
    // pool on a single device.
    if (devices_used.size() > 1) {
        LLAMA_LOG_ERROR(
            "wp::WeightPager::init: multi-device configurations are not supported by the "
            "weight pager (got %zu devices). Use --device with a single ROCm/CUDA index "
            "for paging, or run without --weight-paging.\n",
            devices_used.size());
        // Caller passed fds; close them so they don't leak.
        for (int fd : fds) {
            if (fd >= 0) close(fd);
        }
        return false;
    }
    if (!devices_used.empty() && devices_used.front() != device_idx) {
        LLAMA_LOG_WARN("wp::WeightPager::init: device mismatch (used=%d, configured=%d)\n",
                       devices_used.front(), device_idx);
    }

    cfg_ = cfg;
    if (cfg_.n_slots <= 0) cfg_.n_slots         = catalog_.size();  // pin everything if user didn't pick
    if (cfg_.prefetch_depth <= 0) cfg_.prefetch_depth = 4;

    // Snapshot env BEFORE we touch it.
    env_snapshot(kEnvDisableGraphs, env_was_present_, env_prior_value_);
    setenv(kEnvDisableGraphs, "1", /*overwrite=*/1);

    const size_t slot_size = catalog_.max_page_size();
    if (slot_size == 0) {
        LLAMA_LOG_WARN("wp::WeightPager::init: catalog max_page_size is 0\n");
        return false;
    }

    // 1. VRAM pool.
    if (!pool_.init(device_buft, cfg_.n_slots, slot_size)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: pool allocation failed\n");
        env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
        return false;
    }
    pool_.set_eviction_callback([this](int slot_idx) { on_pool_evict_(slot_idx); });

    // 2. Per-device transfer stream + event pool. Size events generously
    //    so prefetch never blocks waiting for an event.
    if (!transport_.init(device_idx, cfg_.prefetch_depth + 2)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: gpu transport init failed\n");
        pool_.~PoolAllocator();   // explicit teardown via dtor (RAII'd)
        new (&pool_) PoolAllocator{};
        env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
        return false;
    }

    // 3. File IO layer (io_uring or pread).
    file_io_ = create_file_io(std::move(fds), cfg_.prefer_async_io,
                              cfg_.prefetch_depth);
    if (!file_io_) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: file IO layer init failed\n");
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
        return false;
    }

    // 4. Prefetch scheduler bound to the above.
    if (!prefetch_.init(file_io_.get(), &transport_, slot_size, cfg_.prefetch_depth)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: prefetch scheduler init failed\n");
        file_io_.reset();
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
        return false;
    }

    // 5. Page-to-slot map + per-page loaded flag.
    page_to_slot_.assign((size_t) catalog_.size(), -1);
    page_loaded_.assign((size_t)  catalog_.size(), false);
    slot_to_page_.assign((size_t) cfg_.n_slots,    -1);

    // 6. Shared sync staging buffer (max_page_size pinned host). Allocated
    //    ONCE here so page_in_sync_ doesn't pay hipHostMalloc latency per
    //    call. For a 540 MB token-embed tensor that's ~10+ seconds saved
    //    per access.
    sync_staging_size_ = slot_size;
    sync_staging_      = nullptr;
#if defined(GGML_USE_HIP)
    if (hipHostMalloc(&sync_staging_, sync_staging_size_, hipHostMallocDefault) == hipSuccess) {
        sync_staging_pinned_ = true;
    } else {
        LLAMA_LOG_WARN("wp::WeightPager: hipHostMalloc(%zu) for shared sync staging failed; falling back to malloc\n",
                       sync_staging_size_);
        sync_staging_        = std::malloc(sync_staging_size_);
        sync_staging_pinned_ = false;
    }
#else
    sync_staging_ = std::malloc(sync_staging_size_);
#endif
    if (sync_staging_ == nullptr) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: shared sync staging allocation failed\n");
        prefetch_.shutdown();
        file_io_.reset();
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
        return false;
    }

    initialized_ = true;
    LLAMA_LOG_INFO("wp::WeightPager: %d pages, %d slots x %zu B (%.1f MiB), prefetch_depth=%d, sync_staging_pinned=%d\n",
                   catalog_.size(), cfg_.n_slots, slot_size,
                   (double) cfg_.n_slots * (double) slot_size / 1048576.0,
                   cfg_.prefetch_depth, (int) sync_staging_pinned_);
    return true;
}

void WeightPager::shutdown() {
    if (!initialized_) {
        // If init partially completed, there's no live state — but the env
        // snapshot may have been taken. Restore it defensively.
        if (env_was_present_ || !env_prior_value_.empty()) {
            env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
            env_was_present_ = false;
            env_prior_value_.clear();
        }
        return;
    }

    // Tear down in reverse construction order.
    prefetch_.shutdown();
    file_io_.reset();
    transport_.shutdown();
    if (sync_staging_ != nullptr) {
#if defined(GGML_USE_HIP)
        if (sync_staging_pinned_) {
            hipHostFree(sync_staging_);
        } else {
            std::free(sync_staging_);
        }
#else
        std::free(sync_staging_);
#endif
        sync_staging_       = nullptr;
        sync_staging_size_  = 0;
        sync_staging_pinned_ = false;
    }
    // PoolAllocator dtor frees the ggml buffer.
    pool_.~PoolAllocator();
    new (&pool_) PoolAllocator{};

    page_to_slot_.clear();
    page_loaded_.clear();
    slot_to_page_.clear();
    catalog_.clear();

    env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
    env_was_present_ = false;
    env_prior_value_.clear();

    initialized_ = false;
}

void WeightPager::on_pool_evict_(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= (int) slot_to_page_.size()) return;
    int page = slot_to_page_[slot_idx];
    if (page >= 0 && page < (int) page_to_slot_.size()) {
        page_to_slot_[page] = -1;
        page_loaded_[page]  = false;
    }
    slot_to_page_[slot_idx] = -1;
}

int WeightPager::slot_for_page(int page_idx) const {
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return -1;
    if (!page_loaded_[page_idx])                                 return -1;
    return page_to_slot_[page_idx];
}

void * WeightPager::ensure(int page_idx) {
    if (!initialized_)                                         return nullptr;
    if (page_idx < 0 || page_idx >= catalog_.size())           return nullptr;

    // Already committed? Bump LRU and return.
    if (page_loaded_[page_idx]) {
        const int slot = page_to_slot_[page_idx];
        pool_.mark_used(slot);
        return slot_ptr_(slot);
    }

    // Slot reserved by an in-flight prefetch? Wait for it.
    int slot = page_to_slot_[page_idx];
    if (slot >= 0) {
        if (prefetch_.wait_for(page_idx, /*timeout_ms=*/-1)) {
            // Stage 2 done; commit and reap.
            page_loaded_[page_idx] = true;
            pool_.mark_used(slot);
            prefetch_.reap(page_idx);
            return slot_ptr_(slot);
        }
        // Prefetch failed; tear down the reservation so sync fallback can
        // start fresh.
        prefetch_.reap(page_idx);
        page_to_slot_[page_idx] = -1;
        slot_to_page_[slot]     = -1;
        pool_.release_slot(slot);
        // fall through to sync fallback
    }

    // Synchronous fallback: read directly into a slot.
    slot = page_in_sync_(page_idx);
    if (slot < 0) return nullptr;
    return slot_ptr_(slot);
}

void WeightPager::prefetch_page(int page_idx) {
    if (!initialized_)                                          return;
    if (page_idx < 0 || page_idx >= catalog_.size())            return;
    if (page_to_slot_[page_idx] >= 0)                            return;  // loaded or in flight

    // Allocate (or evict) a slot now so the prefetch knows where to land.
    const int slot = pool_.alloc_slot();
    if (slot < 0) return;
    void * dst = slot_ptr_(slot);

    // Track ownership BEFORE submitting so eviction-callbacks resolve right.
    // page_loaded_ stays false until ensure() commits after stage 2.
    page_to_slot_[page_idx]     = slot;
    page_loaded_[page_idx]      = false;
    slot_to_page_[slot]         = page_idx;

    const PageMeta & m = catalog_.at(page_idx);
    if (!prefetch_.submit(page_idx, (int) m.file_idx, m.file_offset,
                          m.size, dst, pool_.slot_size())) {
        // Rejected — likely queue full. Roll back our reservation.
        page_to_slot_[page_idx] = -1;
        slot_to_page_[slot]     = -1;
        pool_.release_slot(slot);
    }
}

void WeightPager::tick() {
    if (!initialized_) return;
    prefetch_.tick();
}

int WeightPager::page_in_sync_(int page_idx) {
    // Synchronous read into a fresh slot, bypassing the prefetch pipeline.
    // Used by ensure() on miss. Tries the fast staging path through the
    // FileIOLayer (sync if iouring, but pinned still helps DMA on async),
    // then hands off to GpuTransport for the H2D + padding zero.

    static int s_diag_count = 0;
    const bool diag = (s_diag_count < 5);
    if (diag) {
        const PageMeta & dm = catalog_.at(page_idx);
        LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: ENTER page=%d name=%s file_idx=%u offset=%lu size=%zu\n",
                        s_diag_count, page_idx, dm.tensor_name.c_str(),
                        (unsigned) dm.file_idx, (unsigned long) dm.file_offset, dm.size);
    }

    const int slot = pool_.alloc_slot();
    if (slot < 0) return -1;
    void * dst = slot_ptr_(slot);
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: alloc_slot ok, slot=%d dst=%p\n", s_diag_count, slot, dst);

    const PageMeta & m = catalog_.at(page_idx);

    // Use the shared pinned staging buffer allocated at init time. Pinning
    // a fresh buffer per call costs hundreds of ms for hundred-MB tensors
    // and would dominate the paging path; the shared buffer is sized to
    // max_page_size so any individual page fits.
    void * staging = sync_staging_;
    if (staging == nullptr || m.size > sync_staging_size_) {
        LLAMA_LOG_ERROR("wp::WeightPager::page_in_sync_: page %d size %zu exceeds shared staging size %zu\n",
                        page_idx, m.size, sync_staging_size_);
        pool_.release_slot(slot);
        return -1;
    }

    // Stage 1: blocking read into staging via the file IO layer.
    const uint64_t req_id = (uint64_t) -1;  // synthetic; not pipelined
    bool ok = file_io_->submit(req_id, (int) m.file_idx, m.file_offset, m.size, staging);
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: submit returned ok=%d\n", s_diag_count, (int)ok);
    if (ok) file_io_->flush();
    while (ok) {
        IoResult r = file_io_->wait_any(/*timeout_ms=*/-1);
        if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: wait_any returned req_id=%lu status=%d bytes=%d\n",
                                  s_diag_count, (unsigned long) r.req_id, (int) r.status, r.bytes_read);
        if (r.req_id == req_id) {
            ok = (r.status == IoStatus::Ok && r.bytes_read == (int) m.size);
            break;
        }
        // Unrelated completion (could be a stale prefetch). Drop it; the
        // prefetch path treats unknown req_ids as no-ops in process_io_.
    }
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: stage1 done ok=%d\n", s_diag_count, (int)ok);
    if (!ok) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: file IO failed for page %d\n", page_idx);
        pool_.release_slot(slot);
        return -1;
    }

    // Stage 2: H2D + padding zero, blocking.
    int evt = transport_.stage_in(dst, staging, m.size, pool_.slot_size());
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: stage_in returned evt=%d\n", s_diag_count, evt);
    if (evt < 0 || !transport_.synchronize(evt)) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: gpu stage_in failed for page %d\n", page_idx);
        if (evt >= 0) transport_.release_event(evt);
        pool_.release_slot(slot);
        return -1;
    }
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: synchronize ok\n", s_diag_count);
    transport_.release_event(evt);

    // Shared sync_staging_ is owned by the WeightPager; no per-call free.

    page_to_slot_[page_idx] = slot;
    page_loaded_[page_idx]  = true;
    slot_to_page_[slot]     = page_idx;
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT slot=%d\n", s_diag_count, slot);
    ++s_diag_count;
    return slot;
}

}  // namespace wp
