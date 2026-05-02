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
//      and forced to "1" (ggml's hipGraph capture bakes tensor->data
//      pointers; the eval callback's per-step rewrites are incompatible).
//      The original env-var state is restored on shutdown — fixes B-P5.
//   4. ensure() / prefetch_next() / tick() during inference, called from
//      the eval callback adapter.
//   5. shutdown() (or destructor) tears everything down in reverse order.

#include "wp-page-catalog.h"
#include "wp-file-io.h"
#include "wp-pool.h"
#include "wp-gpu-transport.h"
#include "wp-prefetch.h"

#include <cstddef>
#include <memory>
#include <string>
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
        bool prefer_async_io = true;  // try io_uring for stage 1 before SyncPread
    };

    WeightPager() = default;
    ~WeightPager();

    WeightPager(const WeightPager &)             = delete;
    WeightPager & operator=(const WeightPager &) = delete;

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
    // pre-init value. Safe to call multiple times.
    void shutdown();

    // Lookup helpers.
    int  find_page(const std::string & name) const { return catalog_.find(name); }
    int  n_pages()                            const { return catalog_.size(); }
    bool is_initialized()                     const { return initialized_; }

    // Ensure a page is in VRAM, returning the slot pointer. Synchronous
    // fallback if the page is not (yet) prefetched. Returns nullptr if
    // page_idx is out of range or any underlying op fails.
    void * ensure(int page_idx);

    // Submit a prefetch hint for a page. No-op if the page is already in
    // flight or already loaded. Errors are logged but do not propagate —
    // the eval callback's ensure() will fall back to sync on miss.
    void prefetch_page(int page_idx);

    // Drive the prefetch pipeline forward. Idempotent and non-blocking.
    void tick();

    // Backing buffer for the pool — used by the eval-cb adapter when
    // patching tensor->buffer (B-P4 requires a valid ggml backend buffer).
    ggml_backend_buffer_t pool_buf() const { return pool_.vram_buf(); }

    // Slot-and-page metadata (read-only public view).
    const PageMeta & page_meta(int page_idx) const { return catalog_.at(page_idx); }

    // Where a page currently lives in the pool, or -1 if not loaded.
    int slot_for_page(int page_idx) const;

private:
    // Internal helper: synchronous page-in (used by ensure() on miss).
    // Reads the page's bytes via FileIOLayer (sync path), copies to VRAM,
    // and zeros the padding. Returns the slot index or -1 on failure.
    int  page_in_sync_(int page_idx);

    // Resolve a slot index to a VRAM pointer.
    void * slot_ptr_(int slot_idx) const { return pool_.slot_ptr(slot_idx); }

    // PoolAllocator's eviction callback — clears page_to_slot_[evicted].
    void on_pool_evict_(int slot_idx);

    // Catalog of all pages. Built before init().
    PageCatalog catalog_;

    // Owned subsystems.
    std::unique_ptr<FileIOLayer> file_io_;
    PoolAllocator                pool_;
    GpuTransport                 transport_;
    PrefetchScheduler            prefetch_;

    // page_idx -> slot_idx (or -1). Set both for in-flight prefetches and
    // for committed (data-ready) pages — distinguished by page_loaded_.
    std::vector<int> page_to_slot_;
    // page_idx -> true iff the slot's data is committed (sync page-in done
    // OR prefetch stage 2 completed and reaped). False means slot is
    // reserved but the bytes aren't there yet.
    std::vector<bool> page_loaded_;
    // Reverse map: slot_idx -> page_idx (or -1 if free). Used by the
    // eviction callback to clear page_to_slot_ / page_loaded_ correctly.
    std::vector<int> slot_to_page_;

    Config cfg_;
    bool   initialized_ = false;

    // GGML_CUDA_DISABLE_GRAPHS lifecycle (B-P5).
    bool        env_was_present_ = false;
    std::string env_prior_value_;

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

}  // namespace wp
