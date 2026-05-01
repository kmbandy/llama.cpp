#pragma once

// PoolAllocator — a fixed-size VRAM slot ring with LRU eviction.
//
// One pool per ggml_backend_buffer_type_t. Phase 1 is single-device by
// design (the WeightPager rejects multi-device configs at init), but the
// API takes a buffer-type so per-device pools are a drop-in extension.
//
// Allocation strategy:
//   - All n_slots slots are co-allocated in a single ggml_backend_buffer
//     (fixes B-P4 in docs/dev/memory-tier-bug-catalog.md: tensors paged in
//     must have a valid `tensor->buffer` matching the device's buffer-type
//     so ggml_cuda_mul_mat's assertion passes).
//   - When alloc_slot() is called and all slots are in use, the LRU slot
//     is evicted: the pool invokes the caller-supplied eviction callback
//     (so the caller can clear the page metadata that owned the slot)
//     and then hands the slot to the new owner.
//   - The pool itself does not know what a "page" is. It only knows
//     "slot i was last used at tick T."

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

namespace wp {

class PoolAllocator {
public:
    // Called once, with the slot index that the pool is about to overwrite.
    // The callee MUST clear any external state pointing at that slot (page
    // table entries etc.) before returning.
    using EvictionCallback = std::function<void(int slot_idx)>;

    PoolAllocator() = default;
    ~PoolAllocator();

    PoolAllocator(const PoolAllocator &)             = delete;
    PoolAllocator & operator=(const PoolAllocator &) = delete;

    // Allocate the underlying ggml backend buffer. Must be called once
    // before any slot operations. Returns false on failure (allocation
    // failed; pool unusable).
    //
    // After success: vram_buf() and slot_ptr(i) for 0 <= i < n_slots() are
    // valid. The buffer is freed when the PoolAllocator is destroyed.
    bool init(ggml_backend_buffer_type_t buft,
              int                        n_slots,
              size_t                     slot_size);

    // Register the eviction callback. Optional; default is a no-op.
    void set_eviction_callback(EvictionCallback cb) { on_evict_ = std::move(cb); }

    // Acquire a slot. If all slots are in use, the LRU slot is evicted via
    // the registered eviction callback and reused. The returned slot index
    // is marked as used and its LRU tick is bumped.
    //
    // Returns -1 only if the pool is uninitialised or n_slots == 0.
    int alloc_slot();

    // Bump the LRU tick of an already-allocated slot, signalling a cache
    // hit. Caller must ensure the slot was previously returned by
    // alloc_slot() and has not been evicted.
    void mark_used(int slot_idx);

    // Explicit free. Rarely needed — LRU eviction is the primary path.
    // Useful for tests and for the pager's shutdown flow.
    void release_slot(int slot_idx);

    // Pointer into the slot's memory region. Valid for the lifetime of the
    // pool. Returns nullptr if slot_idx is out of range.
    void * slot_ptr(int slot_idx) const;

    // Backing buffer; pass to tensor->buffer when paging a tensor in.
    ggml_backend_buffer_t vram_buf() const { return buf_; }

    int    n_slots()   const { return n_slots_;   }
    size_t slot_size() const { return slot_size_; }

    // Inspect LRU state for tests / metrics.
    int lru_slot() const;

private:
    ggml_backend_buffer_t buf_       = nullptr;
    void *                base_      = nullptr;
    int                   n_slots_   = 0;
    size_t                slot_size_ = 0;
    uint64_t              tick_      = 0;

    std::vector<bool>     used_;
    std::vector<uint64_t> last_used_;
    EvictionCallback      on_evict_;
};

}  // namespace wp
