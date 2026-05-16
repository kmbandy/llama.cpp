// mt_pagedattn_tile — WMMA tile FA kernel implementations.
//
// STATUS: scaffolding only — kernel body is a stub. Dispatch in
// mt_pagedattn.cu must remain gated off until this is fleshed out.
//
// Iteration plan:
//   1. (this commit) Scaffolding + dispatch hook, stub kernel returns 0.
//      Build green, llama-server runs through the existing kernel.
//   2. F16 path: Q/K/V stage to smem → mma::tile<16,8,half2> loads →
//      mma() into tile<16,16,float,DATA_LAYOUT_J_MAJOR> accumulator →
//      online softmax (per-row reductions via __shfl_xor with mask=16) →
//      writeback. Validate vs existing kernel.
//   3. TURBO4_0 path: tile dequant of paged K/V blocks (block_turbo4_0 →
//      half tile in smem). Reuse F16 mma path.
//   4. Perf tuning (multi-warp, smem layout, register pressure).
//
// Targets HEAD_SIZE=128, BLOCK_SIZE=16, CACHE_TYPE in {F16, TURBO4_0}.
// AMD wave32 WMMA only (RDNA3/RDNA4). Existing kernel handles decode
// (q_len < 16) and non-WMMA hardware as fallback.

#include "mt_pagedattn_tile.cuh"

#include <cmath>

namespace mt {

// Stub: identity-passthrough kernel that writes zeros to out.
// Replaced incrementally as the real body is built up.
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
    float           scale) {
    GGML_UNUSED(out); GGML_UNUSED(q); GGML_UNUSED(k_cache); GGML_UNUSED(v_cache);
    GGML_UNUSED(block_tables); GGML_UNUSED(context_lens); GGML_UNUSED(q_lens);
    GGML_UNUSED(max_blocks_per_seq); GGML_UNUSED(n_kv_heads); GGML_UNUSED(n_heads);
    GGML_UNUSED(scale);
    // Intentional no-op until kernel body lands.
}

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
    cudaStream_t    stream) {
    // Stub launch geometry — final geometry will be (n_heads, num_seqs, n_q_tiles).
    constexpr int Q_TILE_M = 16;
    const int n_q_tiles = (max_q_len + Q_TILE_M - 1) / Q_TILE_M;

    dim3 grid(n_heads, num_seqs, n_q_tiles);
    dim3 block(32);  // single wave32 warp

    mt_paged_attention_tile_kernel<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>
        <<<grid, block, 0, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, context_lens, q_lens,
            max_blocks_per_seq, n_kv_heads, n_heads, scale);
}

// Explicit instantiations for the dispatch-supported combos.
template void launch_paged_attn_tile<128, 16, GGML_TYPE_F16>(
        __half *, const __half *, const void *, const void *,
        const int32_t *, const int32_t *, const int32_t *,
        int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attn_tile<128, 16, GGML_TYPE_TURBO4_0>(
        __half *, const __half *, const void *, const void *,
        const int32_t *, const int32_t *, const int32_t *,
        int, int, int, int, int, float, cudaStream_t);

} // namespace mt
