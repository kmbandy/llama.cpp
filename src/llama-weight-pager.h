#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <stdint.h>
#include <unistd.h>

struct ggml_tensor;
struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

#ifdef LLAMA_HAVE_IO_URING
struct io_uring;
#endif

// Forward declarations
struct llama_vram_pool;

/// Represents an in-flight async prefetch request
struct prefetch_req {
    int          page_idx;
    int          slot_idx;
    void       * dst;
};

/// Represents a single weight tensor page that can be paged in/out of VRAM
struct llama_weight_page {
    std::string  tensor_name;   // ggml tensor name (e.g. "blk.0.attn_q.weight")
    uint16_t     file_idx;      // source file index (for split GGUFs)
    size_t       file_offset;   // byte offset in GGUF file
    size_t       size;          // tensor size in bytes
    int          slot_idx;      // VRAM slot index, -1 = not in VRAM
    uint64_t     last_used;     // monotonic counter for LRU
    void        *vram_ptr;      // pointer into slot pool, null if not loaded

    llama_weight_page() : file_idx(0), file_offset(0), size(0), slot_idx(-1), last_used(0), vram_ptr(nullptr) {}
};

/// VRAM slot pool for managing contiguous weight allocations
struct llama_vram_pool {
    void   *base;              // device base pointer for the pool
    ggml_backend_buffer_t ggml_buf; // ggml CUDA buffer wrapping the pool
    void   *pinned_staging;    // pinned host staging buffer
    size_t  slot_size;         // bytes per slot (= max single-layer weight size)
    int     n_slots;           // total slots = floor(pool_bytes / slot_size)
    std::vector<bool>      used;
    std::vector<uint64_t>  lru_tick; // per-slot last-used tick

    llama_vram_pool() : base(nullptr), ggml_buf(nullptr), pinned_staging(nullptr), slot_size(0), n_slots(0) {}

    /// Allocate a slot, evicting LRU if necessary. Returns slot index.
    int alloc_slot();

    /// Free a slot, making it available for reuse
    void free_slot(int idx);

    /// Get pointer to slot memory
    void* slot_ptr(int idx) { return (uint8_t*)base + (size_t)idx * slot_size; }

    /// Check if pool is initialized
    bool is_valid() const { return base != nullptr; }
};

/// Weight pager: manages paging model weights between NVMe and VRAM
struct llama_weight_pager {
    std::string              model_path;
    std::vector<int>         fds;          // file descriptors for each split file
    llama_vram_pool          pool;
    std::vector<llama_weight_page> pages;  // one per weight tensor
    std::unordered_map<std::string, int> name_to_page;  // tensor_name -> page index
    std::vector<ggml_tensor*> weight_tensor_ptrs;  // actual model tensor pointers
    uint64_t                 tick = 0;

#ifdef LLAMA_HAVE_IO_URING
    // Async io_uring reader for prefetch
    struct llama_io_uring * io_reader = nullptr;
    std::unordered_map<int, prefetch_req> in_flight;  // page_idx -> in-flight request
    bool                         async_prefetch = false; // TODO: re-enable after fixing page-index tracking in complete_prefetch
#endif

    // Async GPU transfer stream for overlapping data transfer with compute
    void * transfer_stream = nullptr;  // hipStream_t

    llama_weight_pager() {}
    ~llama_weight_pager();

    /// Ensure tensor is in VRAM. Returns VRAM pointer.
    /// If tensor already in VRAM, updates tick and returns pointer.
    /// If not loaded, initiates page_in (stub in Phase 1, returns nullptr).
    void* ensure(const std::string & name);

    /// Evict the least-recently-used slot
    void evict_lru();

    /// Find page index by tensor name. Returns -1 if not found.
    int find_page(const std::string & name) const;

    /// Page in a tensor from NVMe to VRAM (stub in Phase 1)
    void page_in(llama_weight_page & page);

    /// Initialize the VRAM pool with given parameters
    bool init_pool(size_t slot_size, int n_slots);

    /// Get the LRU slot index (for testing/debugging)
    int get_lru_slot() const;

#ifdef LLAMA_HAVE_IO_URING
    /// Initialize io_uring for async prefetch. Must be called after fd is set.
    bool init_io_uring(int queue_depth = 64);

    /// Submit an async prefetch for the given page index.
    /// If the page is already in VRAM, this is a no-op.
    void submit_prefetch(int page_idx);

    /// Complete any in-flight prefetches for the given page index.
    /// Waits for the IO if still in flight; returns true if the page is now loaded.
    bool complete_prefetch(int page_idx);

    /// Drain all in-flight prefetch requests (wait for completion).
    void drain_prefetches();
#endif
};

/// Eval callback for weight pager. Called before each graph node is executed.
/// Ensures weight tensors are paged in to VRAM before use.
/// Returns true to continue the callback, false to stop.
bool weight_pager_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);
