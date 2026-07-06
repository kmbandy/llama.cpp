#include "wp-pager.h"

#include "ggml-backend.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <chrono>
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
// unless WP_HIP_GRAPHS=1 (default off). With WP_HIP_GRAPHS unset or 0 this
// preserves today's safety path exactly: graph capture stays disabled because
// the eval callback rewrites tensor->data before execution. With
// WP_HIP_GRAPHS=1 we leave the user's graph setting alone so MAD-P1 graph
// update handling can be tested. On shutdown we restore whatever value (or
// absence) the user had before when we forced it. This fixes B-P5: the
// previous pager set the var unconditionally and never restored it, leaking
// the "graphs disabled" state into any subsequent model load in the same
// process.
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
constexpr const char * kEnvWpHipGraphs   = "WP_HIP_GRAPHS";
constexpr const char * kEnvWpAsyncEnsure = "WP_ASYNC_ENSURE";

bool env_flag_is_one(const char * var) {
    const char * v = std::getenv(var);
    return v != nullptr && std::strcmp(v, "1") == 0;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// WeightPager
// ---------------------------------------------------------------------------

WeightPager::~WeightPager() {
    shutdown();
}

int WeightPager::register_pinned(const std::string & name, void * device_ptr, size_t bytes) {
    return catalog_.add_pinned(name, device_ptr, bytes);
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

    stats_ = Stats{};
    cfg_ = cfg;
    if (cfg_.n_slots <= 0) cfg_.n_slots         = catalog_.size();  // pin everything if user didn't pick
    if (cfg_.prefetch_depth <= 0) cfg_.prefetch_depth = 4;

    // Snapshot env BEFORE we touch it.
    env_snapshot(kEnvDisableGraphs, env_was_present_, env_prior_value_);
    hip_graphs_enabled_ = env_flag_is_one(kEnvWpHipGraphs);
    async_ensure_enabled_ = env_flag_is_one(kEnvWpAsyncEnsure);
    env_disable_graphs_forced_ = !hip_graphs_enabled_;
    if (env_disable_graphs_forced_) {
        setenv(kEnvDisableGraphs, "1", /*overwrite=*/1);
    } else {
        LLAMA_LOG_WARN("wp::WeightPager: WP_HIP_GRAPHS=1, leaving GGML_CUDA_DISABLE_GRAPHS unchanged\n");
    }

    const size_t slot_size = catalog_.max_page_size();
    if (slot_size == 0) {
        LLAMA_LOG_WARN("wp::WeightPager::init: catalog max_page_size is 0\n");
        restore_disable_graphs_env_();
        return false;
    }

    // 1. VRAM pool. Pass device_idx so the pool can detect UMA / APU devices
    //    (Strix Halo etc.) and refuse oversized requests up front rather than
    //    silently allocating from system RAM and triggering a swap storm.
    //    MAD-234.
    if (!pool_.init(device_buft, cfg_.n_slots, slot_size, device_idx)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: pool allocation failed\n");
        restore_disable_graphs_env_();
        return false;
    }
    pool_.set_eviction_callback([this](int slot_idx) { on_pool_evict_(slot_idx); });

    // MAD-237: opt-in hot-slot protection. Off by default (threshold=0 ⇒
    // pure LRU = MAD-231 behaviour). Operators enable via env when their
    // workload has measurable expert reuse (typically MoE with skewed
    // routing distribution). Cheap to leave off; nothing else changes.
    if (const char * env = std::getenv("WP_HOT_HIT_THRESHOLD")) {
        long t = std::strtol(env, nullptr, 10);
        if (t < 0) t = 0;
        pool_.set_hot_hit_threshold((uint32_t) t);
        if (t > 0) {
            LLAMA_LOG_INFO("wp::WeightPager: WP_HOT_HIT_THRESHOLD=%ld "
                           "(slots with hit_count > %ld are skipped in LRU "
                           "eviction Pass A)\n", t, t);
        }
    }

    // 2. Per-device transfer stream + event pool. Size events generously
    //    so prefetch never blocks waiting for an event.
    const int n_transport_events = async_ensure_enabled_
        ? cfg_.prefetch_depth * 2 + 8
        : cfg_.prefetch_depth + 2;
    if (!transport_.init(device_idx, n_transport_events, async_ensure_enabled_)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: gpu transport init failed\n");
        pool_.~PoolAllocator();   // explicit teardown via dtor (RAII'd)
        new (&pool_) PoolAllocator{};
        restore_disable_graphs_env_();
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
        restore_disable_graphs_env_();
        return false;
    }

    // 4. Prefetch scheduler bound to the above.
    if (!prefetch_.init(file_io_.get(), &transport_, slot_size, cfg_.prefetch_depth,
                        async_ensure_enabled_)) {
        LLAMA_LOG_ERROR("wp::WeightPager::init: prefetch scheduler init failed\n");
        file_io_.reset();
        transport_.shutdown();
        pool_.~PoolAllocator();
        new (&pool_) PoolAllocator{};
        restore_disable_graphs_env_();
        return false;
    }

    // 5. Page-to-slot map + per-page loaded flag.
    page_to_slot_.assign((size_t) catalog_.size(), -1);
    page_loaded_.assign((size_t)  catalog_.size(), false);
    cross_layer_prefetch_candidate_.assign((size_t) catalog_.size(), false);
    prefetch_started_at_.assign((size_t) catalog_.size(), std::chrono::steady_clock::time_point{});
    page_async_event_.assign((size_t) catalog_.size(), -1);
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
        restore_disable_graphs_env_();
        return false;
    }

    initialized_ = true;
    LLAMA_LOG_INFO("wp::WeightPager: %d pages, %d slots x %zu B (%.1f MiB), prefetch_depth=%d, sync_staging_pinned=%d, WP_ASYNC_ENSURE=%d\n",
                   catalog_.size(), cfg_.n_slots, slot_size,
                   (double) cfg_.n_slots * (double) slot_size / 1048576.0,
                   cfg_.prefetch_depth, (int) sync_staging_pinned_,
                   (int) async_ensure_enabled_);
    return true;
}

void WeightPager::shutdown() {
    if (!initialized_) {
        // If init partially completed, there's no live state — but the env
        // snapshot may have been taken. Restore it defensively.
        restore_disable_graphs_env_();
        env_was_present_ = false;
        env_prior_value_.clear();
        hip_graphs_enabled_ = false;
        async_ensure_enabled_ = false;
        return;
    }

    log_stats_summary();
    release_graph_pins_();
    for (int evt : page_async_event_) {
        if (evt >= 0) {
            transport_.synchronize(evt);
            transport_.release_event(evt);
        }
    }
    page_async_event_.clear();

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
    cross_layer_prefetch_candidate_.clear();
    prefetch_started_at_.clear();
    slot_to_page_.clear();
    catalog_.clear();

    restore_disable_graphs_env_();
    env_was_present_ = false;
    env_prior_value_.clear();
    hip_graphs_enabled_ = false;
    async_ensure_enabled_ = false;

    initialized_ = false;
}

void WeightPager::restore_disable_graphs_env_() {
    if (!env_disable_graphs_forced_) {
        return;
    }
    env_restore(kEnvDisableGraphs, env_was_present_, env_prior_value_);
    env_disable_graphs_forced_ = false;
}

void WeightPager::release_graph_pins_() {
    if (graph_pin_slots_.empty()) {
        return;
    }
    for (const auto & kv : graph_pin_slots_) {
        for (int slot : kv.second) {
            pool_.unpin_slot(slot);
        }
    }
    graph_pin_slots_.clear();
}

void WeightPager::on_pool_evict_(int slot_idx) {
    if (slot_idx < 0 || slot_idx >= (int) slot_to_page_.size()) return;
    int page = slot_to_page_[slot_idx];
    if (page >= 0 && page < (int) page_to_slot_.size()) {
        if (page < (int) page_async_event_.size() && page_async_event_[page] >= 0) {
            transport_.synchronize(page_async_event_[page]);
            transport_.release_event(page_async_event_[page]);
            page_async_event_[page] = -1;
        }
        page_to_slot_[page] = -1;
        page_loaded_[page]  = false;
    }
    slot_to_page_[slot_idx] = -1;
    ++stats_.evictions;
}

const WeightPager::Stats & WeightPager::stats() const {
    stats_.lru_walk_hot_skips    = pool_.lru_walk_hot_skips();
    stats_.lru_walk_pinned_skips = pool_.lru_walk_pinned_skips();
    return stats_;
}

int WeightPager::loaded_pages() const {
    int n = 0;
    for (bool loaded : page_loaded_) {
        if (loaded) ++n;
    }
    return n;
}

void WeightPager::record_page_in_(size_t bytes, double seconds) {
    ++stats_.page_ins;
    stats_.io_bytes += (uint64_t) bytes;
    if (seconds > 0.0) {
        stats_.io_seconds += seconds;
    }
}

void WeightPager::log_stats_summary() {
    const Stats & s = stats();
    const uint64_t prefetch_total = s.prefetch_hits + s.prefetch_misses;
    const double hit_rate = prefetch_total > 0
        ? 100.0 * (double) s.prefetch_hits / (double) prefetch_total
        : 0.0;
    const double gb_read = (double) s.io_bytes / 1000000000.0;
    const double gbps = s.io_seconds > 0.0 ? gb_read / s.io_seconds : 0.0;

    LLAMA_LOG_INFO(
        "wp::WeightPager summary:\n"
        "  page_ins: %lu\n"
        "  evictions: %lu\n"
        "  prefetch_hits: %lu\n"
        "  prefetch_misses: %lu\n"
        "  prefetch_hit_rate: %.2f%%\n"
        "  io_gb_read: %.3f\n"
        "  io_effective_gb_s: %.3f\n"
        "  sync_fallbacks: %lu\n"
        "  lru_walk_hot_skips: %lu\n"
        "  lru_walk_pinned_skips: %lu\n"
        "  cross_layer_prefetch_submitted: %lu\n"
        "  cross_layer_hit_in_ensure: %lu\n",
        (unsigned long) s.page_ins,
        (unsigned long) s.evictions,
        (unsigned long) s.prefetch_hits,
        (unsigned long) s.prefetch_misses,
        hit_rate,
        gb_read,
        gbps,
        (unsigned long) s.sync_fallbacks,
        (unsigned long) s.lru_walk_hot_skips,
        (unsigned long) s.lru_walk_pinned_skips,
        (unsigned long) s.cross_layer_prefetch_submitted,
        (unsigned long) s.cross_layer_hit_in_ensure);
}

int WeightPager::slot_for_page(int page_idx) const {
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return -1;
    if (!page_loaded_[page_idx])                                 return -1;
    return page_to_slot_[page_idx];
}

void WeightPager::update_graph_pins(const void * graph_key, const std::vector<int> & page_indices) {
    if (!initialized_ || !hip_graphs_enabled_ || graph_key == nullptr) {
        return;
    }

    std::vector<int> slots;
    slots.reserve(page_indices.size());
    for (int page_idx : page_indices) {
        const int slot = slot_for_page(page_idx);
        if (slot < 0) {
            continue;
        }
        bool seen = false;
        for (int s : slots) {
            if (s == slot) {
                seen = true;
                break;
            }
        }
        if (!seen) {
            slots.push_back(slot);
        }
    }

    auto it = graph_pin_slots_.find(graph_key);
    if (it != graph_pin_slots_.end() && it->second == slots) {
        return;
    }

    if (it != graph_pin_slots_.end()) {
        for (int old_slot : it->second) {
            pool_.unpin_slot(old_slot);
        }
        graph_pin_slots_.erase(it);
    }

    if (slots.empty()) {
        return;
    }

    for (int slot : slots) {
        pool_.pin_slot(slot);
    }
    graph_pin_slots_.emplace(graph_key, std::move(slots));
}

void * WeightPager::ensure(int page_idx) {
    if (!initialized_)                                         return nullptr;
    if (page_idx < 0 || page_idx >= catalog_.size())           return nullptr;
    if (page_idx < (int) page_async_event_.size() && page_async_event_[page_idx] >= 0) {
        transport_.synchronize(page_async_event_[page_idx]);
        transport_.release_event(page_async_event_[page_idx]);
        page_async_event_[page_idx] = -1;
    }

    // MAD-236 — pinned (always-resident) pages live in caller-owned VRAM,
    // not the pool. No slot lookup, no LRU update, no IO. Just return the
    // registered device pointer. Cheap and short-circuits the whole pipeline.
    const PageMeta & m_check = catalog_.at(page_idx);
    if (m_check.is_pinned) {
        return m_check.resident_ptr;
    }

    const bool cross_layer_candidate =
        page_idx >= 0 &&
        page_idx < (int) cross_layer_prefetch_candidate_.size() &&
        cross_layer_prefetch_candidate_[page_idx];

    // Already committed? Bump LRU and return.
    if (page_loaded_[page_idx]) {
        if (cross_layer_candidate) {
            ++stats_.cross_layer_hit_in_ensure;
            cross_layer_prefetch_candidate_[page_idx] = false;
        }
        const int slot = page_to_slot_[page_idx];
        pool_.mark_used(slot);
        return slot_ptr_(slot);
    }

    // Slot reserved by an in-flight prefetch? Wait for it.
    bool counted_prefetch_miss = false;
    int slot = page_to_slot_[page_idx];
    if (slot >= 0) {
        if (async_ensure_enabled_) {
            prefetch_.tick();
        }
        const bool loaded_before_wait = prefetch_.is_loaded(page_idx);
        if (loaded_before_wait) {
            ++stats_.prefetch_hits;
            if (cross_layer_candidate) {
                ++stats_.cross_layer_hit_in_ensure;
                cross_layer_prefetch_candidate_[page_idx] = false;
            }
        } else {
            ++stats_.prefetch_misses;
            counted_prefetch_miss = true;
        }
        if (async_ensure_enabled_ && !loaded_before_wait) {
            const int evt = prefetch_.take_stage2_event(page_idx);
            if (evt >= 0) {
                ++stats_.prefetch_hits;
                if (counted_prefetch_miss) {
                    --stats_.prefetch_misses;
                    counted_prefetch_miss = false;
                }
                if (cross_layer_candidate) {
                    ++stats_.cross_layer_hit_in_ensure;
                    cross_layer_prefetch_candidate_[page_idx] = false;
                }

                page_loaded_[page_idx] = true;
                pool_.mark_used(slot);
                prefetch_.reap(page_idx);
                double seconds = 0.0;
                if (page_idx < (int) prefetch_started_at_.size() &&
                    prefetch_started_at_[page_idx] != std::chrono::steady_clock::time_point{}) {
                    seconds = seconds_since(prefetch_started_at_[page_idx]);
                    prefetch_started_at_[page_idx] = std::chrono::steady_clock::time_point{};
                }
                record_page_in_(m_check.size, seconds);
                page_async_event_[page_idx] = evt;
                return slot_ptr_(slot);
            }
        }
        if (prefetch_.wait_for(page_idx, /*timeout_ms=*/-1)) {
            // Stage 2 done; commit and reap.
            page_loaded_[page_idx] = true;
            pool_.mark_used(slot);
            prefetch_.reap(page_idx);
            double seconds = 0.0;
            if (page_idx < (int) prefetch_started_at_.size() &&
                prefetch_started_at_[page_idx] != std::chrono::steady_clock::time_point{}) {
                seconds = seconds_since(prefetch_started_at_[page_idx]);
                prefetch_started_at_[page_idx] = std::chrono::steady_clock::time_point{};
            }
            record_page_in_(m_check.size, seconds);
            if (cross_layer_candidate && !loaded_before_wait) {
                cross_layer_prefetch_candidate_[page_idx] = false;
            }
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
    if (page_idx < (int) cross_layer_prefetch_candidate_.size()) {
        cross_layer_prefetch_candidate_[page_idx] = false;
    }
    if (!counted_prefetch_miss) {
        ++stats_.prefetch_misses;
    }
    ++stats_.sync_fallbacks;
    slot = page_in_sync_(page_idx);
    if (slot < 0) return nullptr;
    return slot_ptr_(slot);
}

int WeightPager::take_async_transfer_event(int page_idx) {
    if (!initialized_) return -1;
    if (page_idx < 0 || page_idx >= (int) page_async_event_.size()) return -1;
    const int evt = page_async_event_[page_idx];
    page_async_event_[page_idx] = -1;
    return evt;
}

bool WeightPager::enqueue_async_transfer_wait(int event_handle, void * stream) {
    return transport_.wait_event_on_stream(event_handle, stream);
}

bool WeightPager::synchronize_async_transfer_event(int event_handle) {
    return transport_.synchronize(event_handle);
}

void WeightPager::release_async_transfer_event(int event_handle) {
    transport_.release_event(event_handle);
}

void WeightPager::prefetch_page(int page_idx) {
    if (!initialized_)                                          return;
    if (page_idx < 0 || page_idx >= catalog_.size())            return;
    if (catalog_.at(page_idx).is_pinned)                        return;  // MAD-236: already resident, no slot needed
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
    } else {
        if (page_idx < (int) prefetch_started_at_.size()) {
            prefetch_started_at_[page_idx] = std::chrono::steady_clock::now();
        }
        if (page_idx < (int) cross_layer_prefetch_candidate_.size() &&
            cross_layer_prefetch_candidate_[page_idx]) {
            ++stats_.cross_layer_prefetch_submitted;
        }
    }
}

void WeightPager::tick() {
    if (!initialized_) return;
    prefetch_.tick();
}

bool WeightPager::prefetch_pages_batch(const std::vector<int> & page_indices) {
    if (!initialized_) return false;
    if (page_indices.empty()) return true;

    // Filter to pages that actually need prefetching (skip already-resident
    // and already-in-flight). Skipping is not a failure — common case is the
    // MoE re-uses some experts from the previous layer.
    std::vector<int> needed;
    needed.reserve(page_indices.size());
    for (int p : page_indices) {
        if (p < 0 || p >= catalog_.size()) continue;
        if (catalog_.at(p).is_pinned)   continue;  // MAD-236: pinned needs no prefetch
        if (page_to_slot_[p] >= 0)      continue;  // resident or in flight
        // Dedupe within input.
        bool dup = false;
        for (int q : needed) {
            if (q == p) { dup = true; break; }
        }
        if (!dup) needed.push_back(p);
    }
    if (needed.empty()) return true;

    // Reserve N slots up-front. On failure (any unable to alloc, e.g. all
    // currently pinned per MAD-231), release the prefix and report.
    std::vector<int> slots;
    slots.reserve(needed.size());
    for (size_t i = 0; i < needed.size(); ++i) {
        const int s = pool_.alloc_slot();
        if (s < 0) {
            for (int prev : slots) {
                pool_.release_slot(prev);
            }
            return false;
        }
        slots.push_back(s);
    }

    // Wire the provisional page→slot bookkeeping (matches what prefetch_page
    // does pre-submit). The scheduler's submit_batch will see clean state —
    // it checks ITS OWN page_to_slot_ map, not ours — but populating ours
    // here prevents a concurrent caller (none today, but defensive) from
    // double-prefetching the same page.
    std::vector<PrefetchBatchRequest> reqs;
    reqs.reserve(needed.size());
    for (size_t i = 0; i < needed.size(); ++i) {
        const int page_idx = needed[i];
        const int slot     = slots[i];
        page_to_slot_[page_idx] = slot;
        page_loaded_[page_idx]  = false;
        slot_to_page_[slot]     = page_idx;

        const PageMeta & m = catalog_.at(page_idx);
        reqs.push_back(PrefetchBatchRequest{
            page_idx, (int) m.file_idx, m.file_offset, m.size,
            slot_ptr_(slot), pool_.slot_size()
        });
    }

    if (!prefetch_.submit_batch(reqs)) {
        // Roll back EVERYTHING — page maps, slot_to_page maps, pool slots.
        // The all-or-nothing contract makes this clean: scheduler made no
        // promises about half-completed state, so we don't need to chase
        // partial scheduler progress.
        for (size_t i = 0; i < needed.size(); ++i) {
            page_to_slot_[needed[i]] = -1;
            slot_to_page_[slots[i]]  = -1;
            pool_.release_slot(slots[i]);
        }
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    for (int page_idx : needed) {
        if (page_idx < (int) prefetch_started_at_.size()) {
            prefetch_started_at_[page_idx] = now;
        }
        if (page_idx < (int) cross_layer_prefetch_candidate_.size() &&
            cross_layer_prefetch_candidate_[page_idx]) {
            ++stats_.cross_layer_prefetch_submitted;
        }
    }
    return true;
}

void WeightPager::pin_page(int page_idx) {
    if (!initialized_) return;
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return;
    const int slot = page_to_slot_[page_idx];
    if (slot < 0) return;  // page not resident; nothing to pin
    pool_.pin_slot(slot);
}

void WeightPager::unpin_page(int page_idx) {
    if (!initialized_) return;
    if (page_idx < 0 || page_idx >= (int) page_to_slot_.size()) return;
    const int slot = page_to_slot_[page_idx];
    if (slot < 0) return;  // page evicted between pin and unpin — refcount was on a
                            // slot that's been reassigned; not our problem (and the
                            // refcount on the new owner would be wrong if we touched it).
    pool_.unpin_slot(slot);
}

std::vector<AdviseRange> compute_advise_ranges(const PageCatalog & catalog,
                                                int                 block_idx,
                                                int                 k) {
    std::vector<AdviseRange> out;
    if (k <= 0 || block_idx < 0) return out;

    // Walk forward k layers. pages_for_block already filters by block_idx;
    // we additionally drop consolidated parents — they carry the full
    // consolidated size for other consumers (eval-cb lookup), but the disk
    // bytes are already covered by the per-expert sub-pages, so advising
    // them again would double-count the file range.
    out.reserve((size_t) k * 8);  // ~3 dense / ~16 MoE per block, headroom
    for (int b = block_idx + 1; b <= block_idx + k; ++b) {
        const std::vector<int> pages = catalog.pages_for_block(b);
        for (int page_idx : pages) {
            const PageMeta & m = catalog.at(page_idx);
            if (m.is_consolidated) continue;    // parents — children cover the bytes
            if (m.size == 0)       continue;    // defence-in-depth for malformed entries
            out.push_back(AdviseRange{m.file_idx, m.file_offset, m.size});
        }
    }
    return out;
}

void WeightPager::advise_layer_lookahead(int block_idx, int k) {
    if (!initialized_ || file_io_ == nullptr) return;
    if (k <= 0 || block_idx < 0)              return;

    const std::vector<AdviseRange> ranges = compute_advise_ranges(catalog_, block_idx, k);
    for (const auto & r : ranges) {
        file_io_->advise_prefetch((int) r.fd_idx, r.offset, r.size);
    }
}

void WeightPager::mark_cross_layer_prefetch_candidates(const std::vector<int> & page_indices) {
    if (!initialized_) return;
    for (int page_idx : page_indices) {
        if (page_idx < 0 || page_idx >= (int) cross_layer_prefetch_candidate_.size()) continue;
        if (catalog_.at(page_idx).is_pinned) continue;
        cross_layer_prefetch_candidate_[page_idx] = true;
    }
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

    const auto io_t0 = std::chrono::steady_clock::now();

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

    // Stage 2: H2D + padding zero. stage_in() preserves the synchronous
    // "resident on return" contract even when WP_ASYNC_ENSURE selects a
    // dedicated transport stream for prefetch stage 2.
    int evt = transport_.stage_in(dst, staging, m.size, pool_.slot_size());
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: stage_in returned evt=%d\n", s_diag_count, evt);
    if (evt < 0) {
        LLAMA_LOG_WARN("wp::WeightPager::page_in_sync_: gpu stage_in failed for page %d\n", page_idx);
        pool_.release_slot(slot);
        return -1;
    }
    transport_.release_event(evt);

    // Shared sync_staging_ is owned by the WeightPager; no per-call free.

    page_to_slot_[page_idx] = slot;
    page_loaded_[page_idx]  = true;
    slot_to_page_[slot]     = page_idx;
    record_page_in_(m.size, seconds_since(io_t0));
    if (diag) LLAMA_LOG_ERROR("[DIAG] page_in_sync_[%d]: EXIT slot=%d\n", s_diag_count, slot);
    ++s_diag_count;
    return slot;
}

}  // namespace wp
