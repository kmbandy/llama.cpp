#pragma once

// BlockTable — per-sequence logical→physical block mapping for the
// paged tier refactor.
//
// Design adapted from vLLM (Apache 2.0, vllm/v1/worker/block_table.py).
// Code is independent (C++ vs Python), architecture is theirs. See
// docs/memory-tier/CREDITS.md when this lands as a complete subsystem.
//
// For each sequence:
//
//   table_[seq] = [physical_block_0, physical_block_1, ..., physical_block_N]
//                  └─ logical block 0 ┘└─ logical block 1 ┘
//
// To find which physical block holds position `p` for sequence `s`:
//
//   logical_block_idx = p / block_size
//   physical_block_id = table_[s][logical_block_idx]
//   token_in_block    = p % block_size
//
// Block size is fixed at construction (typical: 16 tokens). A sequence's
// logical block list grows monotonically as the sequence is appended;
// blocks are released only on whole-seq wipe (clear_seq) or via tier
// migration paths that swap a logical block's physical mapping (e.g.
// hot→warm copy that updates the table to point at the new CPU block).
//
// PHASE 1: Pure data structure with no live wiring. PHASE 2 wires this
// into the eviction trigger + restore path; PHASE 3 makes it the only
// path. Until then the existing position-keyed warm_pos_to_slot_ /
// cold_positions_ paths remain authoritative.
//
// Single-threaded.

#include "llama.h"  // llama_seq_id, llama_pos

#include <cstdint>
#include <vector>

namespace mt {

class BlockTable {
public:
    // Construct + size for `max_seqs` sequences with `block_size` tokens
    // per block. Per-seq capacity grows on demand (no upper bound at
    // construction — the BlockPool's exhaustion check is what bounds
    // total allocation).
    void init(uint32_t max_seqs, uint32_t block_size);

    // Append a physical block to the end of `seq`'s logical sequence.
    // The new logical_block_idx is `num_blocks(seq) - 1` after this
    // call. Caller is responsible for pulling the physical block from
    // BlockPool first.
    void append_block(llama_seq_id seq, uint32_t physical_block_id);

    // Replace the physical block at logical position `idx` for `seq`.
    // Used by tier migration: when a hot block is evicted to warm, the
    // table entry rewrites from the old GPU id to the new CPU id while
    // logical_block_idx stays put. Returns the OLD physical_block_id so
    // the caller can free it back to BlockPool.
    uint32_t swap_block(llama_seq_id seq, uint32_t logical_idx,
                        uint32_t new_physical_block_id);

    // Lookup: which physical block holds logical block `idx` of `seq`?
    // Returns kInvalidBlockId (from mt-block-pool.h) if `seq` has no
    // mapping or `idx` is out of range. Does NOT bounds-assert — paged
    // attention kernels read past the live range during prefill prep
    // and must get a sentinel rather than UB.
    uint32_t get_physical(llama_seq_id seq, uint32_t logical_idx) const;

    // For position-based callers. Same as get_physical(seq, pos /
    // block_size). Convenience wrapper.
    uint32_t get_physical_for_pos(llama_seq_id seq, llama_pos pos) const;

    // Number of physical blocks currently held by `seq`. Always equals
    // ceil(live_token_count / block_size).
    uint32_t num_blocks(llama_seq_id seq) const;

    // Clear all blocks for `seq` and return the freed physical block
    // IDs in the order they were appended. Caller passes these back to
    // BlockPool::free_block(). Used on task-lifecycle wipe (whole-seq
    // seq_rm with sentinel range).
    std::vector<uint32_t> clear_seq(llama_seq_id seq);

    // Reset everything. Used by mt::clear() (whole-cache wipe). Caller
    // is responsible for releasing block IDs back to BlockPool first.
    void reset();

    uint32_t block_size() const { return block_size_; }
    uint32_t max_seqs()   const { return max_seqs_;   }

private:
    // table_[seq][logical_idx] = physical_block_id
    std::vector<std::vector<uint32_t>> table_;
    uint32_t                           max_seqs_   = 0;
    uint32_t                           block_size_ = 0;
};

}  // namespace mt
