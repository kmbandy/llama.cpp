// mt_pagedattn_tile — WMMA tile-based paged attention kernel.
//
// Replaces the per-token serial QK scan in mt_paged_attention_kernel with a
// tile-based flash-attention computation that uses tensor cores (WMMA on
// RDNA3/RDNA4, MFMA on CDNA, MMA on NVIDIA). The page-table indirection is
// resolved at tile-load granularity (gather a tile's worth of K from
// potentially-non-contiguous physical blocks into shared memory) instead of
// the inner loop, so the matmul itself runs over contiguous staged memory.
//
// Targets HEAD_SIZE=128, BLOCK_SIZE=16, CACHE_TYPE in {F16, TURBO4_0}.
// Q_TILE_M=16, K_TILE_N=16, K_INNER=16 — single warp (32 threads) per block,
// HEAD_SIZE/K_INNER = 8 mma ops per (Q tile, K tile) pair.
//
// Dispatch gate (in mt_pagedattn.cu): q_len >= 16 AND amd_wmma_available(cc)
// AND HEAD_SIZE == 128 AND BLOCK_SIZE == 16. Existing kernel handles
// decode (q_len < 16) and non-WMMA hardware (RDNA2, etc.) as fallback.

#pragma once

#include "mt_pagedattn.cuh"
#include "mma.cuh"
#include "turbo-quant.cuh"

#include <cmath>

namespace mt {

// Online softmax / accumulator state per Q row in the tile.
//
// Memory layout chosen so each thread in the 32-thread warp owns a 16-element
// f32 slice of accumulator + a single (max, sum) pair for one Q row, with the
// 16 rows distributed round-robin across the 2 half-warps. Concrete mapping
// is derived inside the kernel from RDNA4 WMMA lane→element semantics.

// Forward decl — body in .cu instantiations file to keep header light.
template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
__global__ void mt_paged_attention_tile_kernel(
    __half         * __restrict__ out,
    const __half   * __restrict__ q,
    const void     * __restrict__ k_cache,
    const void     * __restrict__ v_cache,
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    float           scale);

// Dispatch entry called from ggml_cuda_op_paged_attn_mt after the dispatch
// gate decides this kernel is applicable. Caller is responsible for ensuring
// scatter has already happened on the same stream (DO_SCATTER=false equivalent).
template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
void launch_paged_attn_tile(
    __half         * out,
    const __half   * q,
    const void     * k_cache,
    const void     * v_cache,
    const int32_t  * block_tables,
    const int32_t  * context_lens,
    const int32_t  * q_lens,
    int             num_seqs,
    int             n_heads,
    int             n_kv_heads,
    int             max_blocks_per_seq,
    int             max_q_len,
    float           scale,
    cudaStream_t    stream);

} // namespace mt
