#pragma once

// WeightPager — facade composing PageCatalog + FileIOLayer + PoolAllocator
// + GpuTransport + PrefetchScheduler into a single subsystem the rest of
// llama.cpp talks to.
//
// Lifetime:
//   1. Construct (cheap).
//   2. add_page() N times during model load (typically from the model
//      loader integration in Phase 1d).
//   3. init() once after the catalog is built. Pool, transport, prefetch
//      scheduler all come up here. GGML_CUDA_DISABLE_GRAPHS is snapshotted
//      and forced to "1" unless WP_HIP_GRAPHS=1 (ggml's hipGraph capture
//      bakes tensor->data pointers; the eval callback's per-step rewrites
//      need MAD-P1 graph update handling). The original env-var state is
//      restored on shutdown - fixes B-P5.
//   4. ensure() / prefetch_next() / tick() during inference, called from
//      the eval callback adapter.
//   5. shutdown() (or destructor) tears everything down in reverse order.

#include "wp-page-catalog.h"
#include "wp-file-io.h"
#include "wp-host-tier.h"
#include "wp-pool.h"
#include "wp-gpu-transport.h"
#include "wp-prefetch.h"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

namespace wp {

class WeightPager {
public:
    struct Config {
        int  n_slots         = 0;     // size of the VRAM ring; -1 / 0 = auto (one per layer)
        int  prefetch_depth  = 4;     // PrefetchScheduler queue depth
        int  io_uring_depth  = 0;     // FileIOLayer SQ depth; 0 = prefetch_depth
        bool prefer_async_io = true;  // try io_uring for stage 1 before SyncPread
    };

    struct Stats {
        uint64_t page_ins                       = 0;
        uint64_t evictions                      = 0;
        uint64_t prefetch_hits                  = 0;
        uint64_t prefetch_misses                = 0;
        uint64_t sync_fallbacks                 = 0;
        uint64_t io_bytes                       = 0;
        double   io_seconds                     = 0.0;
        uint64_t lru_walk_hot_skips             = 0;
        uint64_t lru_walk_pinned_skips          = 0;
        uint64_t dense_prefetch_submitted       = 0;
        uint64_t cross_layer_prefetch_submitted = 0;
        uint64_t cross_layer_hit_in_ensure      = 0;
        uint64_t host_tier_hits                 = 0;
        uint64_t routing_ptrs_set                  = 0;
        uint64_t routing_ptrs_consumed             = 0;
        uint64_t routing_ptrs_discarded_unconsumed = 0;
    };

    WeightPager() = default;
    ~WeightPager();

    WeightPager(const WeightPager &)             = delete;
    WeightPager & operator=(const WeightPager &) = delete;

    // MAD-236 — register a tensor whose bytes already live in caller-owned
    // VRAM (e.g. token_embd from the regular model loader buffer). The
    // pager doesn't allocate a pool slot or read from disk for these —
    // ensure(page_idx) returns `device_ptr` directly. Useful for unified
    // VRAM telemetry (paged + resident in one view) and for mixed-mode
    // workloads where some weights stay pinned and others page.
    //
    // Must be called BEFORE init(), like add_page(). Returns the page index.
    int register_pinned(const std::string & name, void * device_ptr, size_t bytes);

    // Catalog population. Must be called before init().
    //
    // n_experts > 1 marks a consolidated MoE expert tensor (Qwen3-MoE
    // style: blk.<N>.ffn_<role>_exps.weight packs all experts as a 3D
    // tensor with ne[2] == n_experts). The catalog will register one
    // sub-page per expert with synthetic names of the form
    // "<base_name>#expert.<E>", letting the pager and prefetch scheduler
    // page individual experts rather than the full consolidated tensor.
    // (MAD-88 Phase 2.) Use n_experts = 1 (default) for non-MoE tensors
    // and Mixtral-style per-expert tensors, which are already separate
    // pages by name.
    int add_page(const std::string & name,
                 uint16_t            file_idx,
                 uint64_t            file_offset,
                 size_t              size,
                 int                 n_experts = 1);

    // Initialise the pool, transport, file-io layer, and prefetch scheduler.
    //
    // `fds` are pre-prepared file descriptors (one per split GGUF file).
    // The caller should obtain them via dup_clear_o_direct() to fix B-P3.
    // The pager takes ownership of the fds; they are closed on shutdown.
    //
    // `devices_used` is the set of HIP/CUDA device indices the model's
    // weights are allocated on, taken from the model loader. If size > 1
    // init returns false with a clear error message — the pager is single-
    // device by design in Phase 1 (per-device pools are a future extension).
    // This guard fixes B-P7.
    //
    // `device_buft` is the buffer type for the device the pager will
    // allocate VRAM on. Should match devices_used.front().
    bool init(const Config &             cfg,
              ggml_backend_buffer_type_t device_buft,
              int                        device_idx,
              std::vector<int>           fds,
              const std::vector<int> &   devices_used);

    // Tear down in reverse order. Restores GGML_CUDA_DISABLE_GRAPHS to its
    // pre-init value if init forced it. Safe to call multiple times.
    void shutdown();

    // Lookup helpers.
    int    find_page(const std::string & name) const { return catalog_.find(name); }
    int    n_pages()                            const { return catalog_.size(); }
    size_t max_page_size()                      const { return catalog_.max_page_size(); }
    bool   is_initialized()                     const { return initialized_; }
    bool   hip_graphs_enabled()                 const { return hip_graphs_enabled_; }
    bool   async_ensure_enabled()               const { return async_ensure_enabled_; }
    const Stats & stats() const;
    int    loaded_pages() const;
    int    pending_prefetches() const { return prefetch_.pending(); }
    bool   async_prefetch_enabled() const { return cfg_.prefer_async_io; }

    // Ensure a page is in VRAM, returning the slot pointer. Synchronous
    // fallback if the page is not (yet) prefetched. Returns nullptr if
    // page_idx is out of range or any underlying op fails.
    void * ensure(int page_idx);

    // WP_ASYNC_ENSURE handoff. ensure() stashes the transfer event here when
    // it returns before stage 2 has completed; the eval callback takes it,
    // queues a compute-stream wait, and releases it after that op completes.
    int  take_async_transfer_event(int page_idx);
    bool enqueue_async_transfer_wait(int event_handle, void * stream);
    bool synchronize_async_transfer_event(int event_handle);
    void finish_async_transfer_event(int page_idx, int event_handle);
    void release_async_transfer_event(int event_handle);

    // Submit a prefetch hint for a page. No-op if the page is already in
    // flight or already loaded. Errors are logged but do not propagate —
    // the eval callback's ensure() will fall back to sync on miss.
    bool prefetch_page(int page_idx, bool count_dense_prefetch = false);

    // MAD-235 — batch-prefetch N pages atomically. Reserves N pool slots
    // up-front, builds the file-IO batch, issues one batched submit. If
    // any step fails (pool can't supply N slots, scheduler rejects the
    // batch), returns false and leaves no partial state. Caller falls
    // back to per-page prefetch_page() on false.
    //
    // Skips page indices that are already resident or already in flight
    // — those are no-ops, not failures. Returns true iff every NEEDED
    // request was queued (or no requests were needed).
    bool prefetch_pages_batch(const std::vector<int> & page_indices,
                              bool count_dense_prefetch = false);

    // Hint the kernel (via POSIX_FADV_WILLNEED on the file_io layer) that
    // we will soon need every paged tensor in layers [block_idx+1,
    // block_idx+k]. Cheap (one syscall per range), idempotent at kernel
    // level. Subsequent reads against these ranges hit warm page cache
    // instead of cold NVMe.
    //
    // No-op when not initialized, when k <= 0, or when block_idx < 0.
    //
    // The eval callback invokes this once per layer boundary in the forward
    // pass. RAM cost: each advised range is `size` bytes of page cache
    // pressure; on RAM-tight systems keep k low or disable via
    // WP_FADVISE_LOOKAHEAD=0.
    void advise_layer_lookahead(int block_idx, int k);

    // MAD-233 aggregate instrumentation. The eval callback marks candidate
    // pages before issuing cross-layer prefetches; successful scheduler
    // submissions and later ensure-time hits are folded into Stats.
    void mark_cross_layer_prefetch_candidates(const std::vector<int> & page_indices);

    // Drive the prefetch pipeline forward. Idempotent and non-blocking.
    void tick();

    // Pin / unpin the slot currently backing `page_idx` so eviction can't
    // reclaim it while an in-flight op references the VRAM. Refcounted via
    // PoolAllocator::pin_slot — safe under overlapping ops that touch the
    // same page. No-op if page_idx is out of range or the page is not
    // currently resident (no slot to pin).
    //
    // Lifecycle in the eval-callback (MAD-231): call pin_page on each page
    // the op references after ensure() succeeds; call unpin_page in the
    // NEXT eval-callback before submitting any new prefetches. This window
    // is a superset of the actual GPU-side read but stream ordering on the
    // compute stream guarantees correctness for the conservative bound.
    void pin_page(int page_idx);
    void unpin_page(int page_idx);

    // Backing buffer for the pool — used by the eval-cb adapter when
    // patching tensor->buffer (B-P4 requires a valid ggml backend buffer).
    ggml_backend_buffer_t pool_buf() const { return pool_.vram_buf(); }

    // Slot-and-page metadata (read-only public view).
    const PageMeta & page_meta(int page_idx) const { return catalog_.at(page_idx); }

    // Where a page currently lives in the pool, or -1 if not loaded.
    int slot_for_page(int page_idx) const;

    // MAD-P1: graph-lifetime pins for slots captured by CUDA/HIP graphs.
    // Replaces the pin set for graph_key with the slots currently backing
    // page_indices, unpinning the old exact slots first. No-op unless
    // WP_HIP_GRAPHS=1.
    void update_graph_pins(const void * graph_key, const std::vector<int> & page_indices);
    bool try_add_graph_pin_page(const void * graph_key, int page_idx, std::vector<int> & page_indices) const;

    // Stable slot base address for capture. Valid for the pool lifetime.
    void * slot_base_for_capture(int slot_idx) const { return pool_.slot_base_for_capture(slot_idx); }

private:
    // Internal helper: synchronous page-in (used by ensure() on miss).
    // Reads the page's bytes via FileIOLayer (sync path), copies to VRAM,
    // and zeros the padding. Returns the slot index or -1 on failure.
    int  page_in_sync_(int page_idx);

    // Resolve a slot index to a VRAM pointer.
    void * slot_ptr_(int slot_idx) const { return pool_.slot_ptr(slot_idx); }

    // PoolAllocator's eviction callback — clears page_to_slot_[evicted].
    void on_pool_evict_(int slot_idx);

    void log_stats_summary();
    void record_page_in_(size_t bytes, double seconds);
    void restore_disable_graphs_env_();
    void release_graph_pins_();
    int  graph_pin_max_slots_() const;
    int  graph_pin_slot_count_except_(const void * graph_key) const;

    // Catalog of all pages. Built before init().
    PageCatalog catalog_;

    // Owned subsystems.
    std::unique_ptr<FileIOLayer> file_io_;
    PoolAllocator                pool_;
    GpuTransport                 transport_;
    PrefetchScheduler            prefetch_;
    std::unique_ptr<HostTier>    host_tier_;

    // page_idx -> slot_idx (or -1). Set both for in-flight prefetches and
    // for committed (data-ready) pages — distinguished by page_loaded_.
    std::vector<int> page_to_slot_;
    // page_idx -> true iff the slot's data is committed (sync page-in done
    // OR prefetch stage 2 completed and reaped). False means slot is
    // reserved but the bytes aren't there yet.
    std::vector<bool> page_loaded_;
    std::vector<bool> cross_layer_prefetch_candidate_;
    std::vector<std::chrono::steady_clock::time_point> prefetch_started_at_;
    std::vector<int> page_async_event_;
    // Reverse map: slot_idx -> page_idx (or -1 if free). Used by the
    // eviction callback to clear page_to_slot_ / page_loaded_ correctly.
    std::vector<int> slot_to_page_;

    Config cfg_;
    bool   initialized_ = false;
    bool   hip_graphs_enabled_ = false;
    bool   async_ensure_enabled_ = false;
    mutable Stats stats_;

    // GGML_CUDA_DISABLE_GRAPHS lifecycle (B-P5).
    bool        env_was_present_ = false;
    bool        env_disable_graphs_forced_ = false;
    std::string env_prior_value_;

    // MAD-P1 graph_key -> exact pool slots pinned for captured graph args.
    // Stored as slots, not pages, so unpin releases the same refcounts even
    // if a page later moves to another slot.
    std::unordered_map<const void *, std::vector<int>> graph_pin_slots_;

    // Shared pinned staging buffer for page_in_sync_. Allocated once at
    // init, sized to max_page_size, reused across every sync page-in.
    // Pinning a fresh buffer per call (the original design) costs
    // hundreds of ms per allocation for hundred-MB tensors and dominates
    // the paging path for dense layers. Single shared buffer matches
    // the OLD pager's pool.pinned_staging behaviour.
    void * sync_staging_       = nullptr;
    size_t sync_staging_size_  = 0;
    bool   sync_staging_pinned_ = false;  // true if hipHostMalloc, false if malloc fallback
};

// File-range descriptor for advise_prefetch — one per paged tensor in the
// catalog walk. Exposed at namespace scope so unit tests can validate the
// catalog-walk logic (compute_advise_ranges) without standing up a real
// WeightPager.
struct AdviseRange {
    uint16_t fd_idx;
    uint64_t offset;
    size_t   size;
};

// Walk `catalog` for every page whose block_idx is in [block_idx+1,
// block_idx+k] AND whose size > 0 (i.e. excludes consolidated parents,
// which have no own bytes). Skips negative inputs. Caller owns the
// returned vector. Pure function — no I/O, no global state. Unit-tested
// directly without a real pager.
std::vector<AdviseRange> compute_advise_ranges(const PageCatalog & catalog,
                                                int                 block_idx,
                                                int                 k);

}  // namespace wp
