#pragma once

// mt_pagedattn — paged attention kernel for the mt:: tier system.
//
// Design adapted from vLLM's csrc/attention/attention_kernels.cuh
// (Apache 2.0). Code is independent (HIP/CUDA via ggml backend, not
// PyTorch/TorchCompile). See docs/memory-tier/CREDITS.md.
//
// Reads K/V from a block-indexed buffer ("paged blocks") rather than
// a contiguous KV cache. Each sequence has a `block_table` mapping
// logical block index → physical block ID, and the kernel walks
// blocks in logical order to compute attention.
//
// Layout (matches vLLM for memory-coalescing on the K side):
//
//   K cache: [num_blocks, n_kv_heads, head_dim/x, block_size, x]
//     where x = 16 / sizeof(scalar_t). For fp16 (2 bytes): x = 8.
//     This interleaves the inner head_dim dimension into chunks of
//     `x` elements that sit adjacent across `block_size` tokens —
//     so a warp loading a single token's K row reads `x` contiguous
//     elements per thread group.
//
//   V cache: [num_blocks, n_kv_heads, head_dim, block_size]
//     Transposed relative to K. For each (block, kv_head, head_dim)
//     row, all `block_size` tokens are contiguous. Lets the V@logits
//     phase load `V_VEC_SIZE` tokens worth of one head_dim row in
//     one coalesced read per thread.
//
// Threading model (per CUDA/HIP thread block):
//   grid:   (num_heads, num_seqs)
//   block:  NUM_THREADS = some multiple of WARP_SIZE
//   Each thread block handles one (head, seq) pair, computing the
//   complete attention output for all query tokens of that seq's
//   batch slot against the seq's full block-mapped context.
//
// PHASE 3.1: this header declares the kernel and dispatch entry; .cu
// implements them. Decode + prefill share the same kernel by
// generalizing the query-token loop. Prefill from scratch (no cached
// blocks yet) is handled by the caller writing K/V into freshly
// allocated blocks before launching.

#include "common.cuh"

namespace mt {

// Sentinel matching mt::kInvalidBlockId (~0u). Kept here as int32_t
// because the block_table tensor's element type is i32.
static constexpr int32_t kInvalidBlockTableEntry = -1;

// Kernel entry. Templated on compile-time constants for dispatch
// specialization — the wrapper picks the right template instantiation
// based on op_params.
//
// scalar_t  : compute precision for Q (typically half/__half)
// cache_t   : KV cache element type (half today; future quantized variants)
// HEAD_SIZE : head_dim (e.g. 128). Must match q->ne[0].
// BLOCK_SIZE: tokens per block (e.g. 16). Must be a power of 2.
// NUM_THREADS : threads per CUDA/HIP block (e.g. 128). Multiple of WARP_SIZE.
// PARTITION_SIZE : 0 = no partition (single thread block per (head, seq));
//                  >0 = split long contexts across grid.z partitions for
//                  better SM utilization. 0 for our v1 — partitioning is
//                  a follow-up.
template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE = 0>
__global__ void mt_paged_attention_kernel(
    scalar_t       * __restrict__ out,           // [num_seqs, num_heads, num_q_tokens_per_seq, head_size]
    const scalar_t * __restrict__ q,             // [num_seqs, num_heads, num_q_tokens_per_seq, head_size]
    const cache_t  * __restrict__ k_cache,       // [num_blocks, n_kv_heads, head_size/x, block_size, x]
    const cache_t  * __restrict__ v_cache,       // [num_blocks, n_kv_heads, head_size, block_size]
    const int32_t  * __restrict__ block_tables,  // [num_seqs, max_blocks_per_seq]
    const int32_t  * __restrict__ context_lens,  // [num_seqs]
    const int32_t  * __restrict__ q_lens,        // [num_seqs] — number of query tokens per seq (1 for decode, N for prefill)
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    float           scale);

// Host-side dispatch. Picks the right kernel template + grid/block
// dims given runtime tensor metadata. Implemented in mt_pagedattn.cu.
// Inputs come from the GGML_OP_PAGED_ATTN_MT op's source tensors.
void ggml_cuda_op_paged_attn_mt(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

}  // namespace mt
