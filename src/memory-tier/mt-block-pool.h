#pragma once

// BlockPool — physical block allocation pool for the paged tier refactor.
//
// Design adapted from vLLM (Apache 2.0, vllm/v1/core/block_pool.py +
// vllm/v1/worker/block_table.py). Code is independent (C++ vs Python),
// architecture is theirs. See docs/memory-tier/CREDITS.md when this
// lands as a complete subsystem.
//
// A "physical block" is a fixed-size chunk of KV cache storage capable
// of holding `block_size` tokens worth of K/V for one layer. Blocks are
// allocated from one of two pools — GPU (hot tier) or CPU (warm tier) —
// distinguished by their ID range:
//
//   [0,                total_gpu_blocks)                       GPU blocks
//   [total_gpu_blocks, total_gpu_blocks + total_cpu_blocks)    CPU blocks
//
// Cold tier (NVMe) is handled outside this pool — KvtcStore writes by
// (layer, position) and doesn't need a physical block ID.
//
// Allocation is LIFO from a free stack; cache-line reuse on freshly-
// freed blocks helps decode locality. Watermark gating reserves a small
// fraction of free blocks against admission spikes so the scheduler
// never starves itself by accepting one more request than it can
// retire.
//
// PHASE 1: This is a pure data structure with no live wiring. The
// existing position-keyed warm_pos_to_slot_ / cold_positions_ paths
// in mt-tiered.cpp continue to work. PHASE 2 wires this in behind a
// feature flag; PHASE 3 makes it the only path.
//
// Single-threaded — caller must serialize. mt:: is single-seq today
// and the scheduler runs single-threaded, so no internal locking.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace mt {

// Sentinel for "no block." Chosen as ~0u so it sticks out in logs and
// any accidental use as an array index will explode loudly.
static constexpr uint32_t kInvalidBlockId = ~0u;

class BlockPool {
public:
    // Initialize the pool with `n_gpu` GPU blocks and `n_cpu` CPU blocks.
    // `watermark` is the fraction (0.0..1.0) of each pool kept in reserve
    // before has_free_*_blocks() returns true for new admissions. 0.0
    // disables the watermark; vLLM defaults to ~0.05 (5%).
    void init(uint32_t n_gpu, uint32_t n_cpu, float watermark);

    // Allocate one physical block from the GPU pool. Returns
    // kInvalidBlockId when the pool is exhausted (caller should evict
    // first). Watermark is NOT enforced here — callers that care about
    // admission control should consult has_free_gpu_blocks() up-front.
    // alloc_*() always allocates if anything is free.
    uint32_t alloc_gpu();
    uint32_t alloc_cpu();

    // Return a block to its pool. Idempotent on already-free blocks
    // (logs a warning); double-free does NOT corrupt the pool. Asserts
    // on out-of-range IDs.
    void free_block(uint32_t block_id);

    // True if `block_id` is a GPU block (vs CPU). Inferred from the ID
    // range — no per-block bool stored.
    bool is_gpu(uint32_t block_id) const;

    // Pool size queries. n_free_*() is the actual count of free blocks
    // (ignores watermark); has_free_*_blocks(N) tests whether N blocks
    // can be allocated WITHOUT crossing the watermark reserve.
    size_t n_free_gpu() const;
    size_t n_free_cpu() const;
    bool   has_free_gpu_blocks(uint32_t n) const;
    bool   has_free_cpu_blocks(uint32_t n) const;

    uint32_t total_gpu_blocks() const { return total_gpu_blocks_; }
    uint32_t total_cpu_blocks() const { return total_cpu_blocks_; }

    // Reset to all-free state. Used by whole-cache wipes (mt::clear()).
    // Block IDs are NOT renumbered; existing block-table entries that
    // pointed into the freed range become dangling and the caller must
    // also clear those before next use.
    void reset();

private:
    // Free stacks (LIFO). gpu_free_ holds IDs in [0, total_gpu_blocks_),
    // cpu_free_ holds IDs in [total_gpu_blocks_, total_gpu_blocks_ +
    // total_cpu_blocks_). Stack push/pop is the eviction order under
    // pure LRU; smarter eviction policies layer above.
    std::vector<uint32_t> gpu_free_;
    std::vector<uint32_t> cpu_free_;

    uint32_t total_gpu_blocks_ = 0;
    uint32_t total_cpu_blocks_ = 0;
    uint32_t watermark_gpu_    = 0;  // reserve count, NOT a fraction
    uint32_t watermark_cpu_    = 0;
};

}  // namespace mt
