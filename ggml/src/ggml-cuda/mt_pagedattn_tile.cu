// mt_pagedattn_tile — WMMA tile FA kernel implementations.
//
// Replaces the per-token serial QK scan in mt_paged_attention_kernel with
// a tile-based flash attention that uses WMMA (RDNA3/RDNA4 wave32) or MFMA
// (CDNA). Page-table indirection is resolved at tile-load granularity:
// gather a tile's worth of K/V from potentially-non-contiguous physical
// blocks into shared memory, then run the matmul over contiguous smem.
//
// Tile geometry: Q_TILE_M=16, K_TILE_N=16, K_INNER=16.
// HEAD_SIZE must be a multiple of K_INNER (HS=128 → 8 mma ops per tile pair).
//
// Lane mapping for tile<16, 16, float, DATA_LAYOUT_I_MAJOR> on AMD WMMA
// (from mma.cuh):
//   ne = 8 elements per thread
//   get_i(l) = threadIdx.x % 16        ← row (same for all l)
//   get_j(l) = 8 * (threadIdx.x / 16) + l  ← col 0..7 for low half-warp,
//                                            cols 8..15 for high half-warp
//
// Row pairs share threads (T, T+16) — cross-lane reduction within a row
// is one __shfl_xor_sync(..., mask=16) call.
//
// Targets HEAD_SIZE=128, BLOCK_SIZE=16, CACHE_TYPE in {F16, TURBO4_0}.
// AMD wave32 WMMA only. Existing scalar kernel handles decode (q_len<16),
// non-WMMA hardware, and other (HEAD_SIZE, BLOCK_SIZE) cases.
//
// STATUS: F16 path drafted; awaiting llama-server validation cycle.
// TURBO4_0 path is staged with a tile-dequant via the existing
// paged_cache_ops<TURBO4_0>::k_load/v_load (one element at a time during
// stage — slower than a cooperative tile dequant but algorithmically
// correct; bulk dequant follows once F16 path validates).

#include "mt_pagedattn_tile.cuh"
#include "mt_pagedattn_ops.cuh"

#include "common.cuh"
#include "mma.cuh"

#include <cmath>

namespace mt {

using namespace ggml_cuda_mma;

// Single wave32 warp per block (RDNA3/RDNA4 native wave size).
static constexpr int TILE_NUM_THREADS = 32;

static constexpr int Q_TILE_M = 16;
static constexpr int K_TILE_N = 16;
static constexpr int K_INNER  = 16;

// ── helpers ─────────────────────────────────────────────────────────────

// Stage Q[q_tile_actual, HEAD_SIZE] from global into smem as contiguous
// row-major half. Q global layout is [head_dim, n_heads, n_tokens]; we
// rebuild [Q_TILE_M, HEAD_SIZE] in smem (padded rows beyond q_tile_actual
// zeroed so the WMMA op produces zero for them).
template <int HEAD_SIZE>
static __device__ __forceinline__ void stage_q_tile(
        __half       * __restrict__ smem_q,
        const __half * __restrict__ q,
        size_t        q_global_base,
        int           q_tile_actual,
        int           n_heads,
        int           tid) {
    #pragma unroll
    for (int idx = tid; idx < Q_TILE_M * HEAD_SIZE; idx += TILE_NUM_THREADS) {
        const int row = idx / HEAD_SIZE;
        const int col = idx % HEAD_SIZE;
        if (row < q_tile_actual) {
            smem_q[idx] = q[q_global_base + (size_t) row * (size_t) n_heads * (size_t) HEAD_SIZE + (size_t) col];
        } else {
            smem_q[idx] = __float2half(0.0f);
        }
    }
}

// Stage K[K_TILE_N tokens, HEAD_SIZE dims] for a given k_tile_start.
// Walks block_table per row to translate logical→physical, then uses
// paged_cache_ops::k_load to fetch the element (handles dequant for
// quantized cache types). Padded with zero beyond valid_ctx.
template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
static __device__ __forceinline__ void stage_k_tile(
        __half        * __restrict__ smem_k,
        const void    * __restrict__ k_cache,
        const int     * __restrict__ seq_block_table,
        int            k_tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            tid) {
    using ops = paged_cache_ops<CACHE_TYPE, HEAD_SIZE, BLOCK_SIZE>;
    #pragma unroll
    for (int idx = tid; idx < K_TILE_N * HEAD_SIZE; idx += TILE_NUM_THREADS) {
        const int row   = idx / HEAD_SIZE;
        const int col   = idx % HEAD_SIZE;
        const int token = k_tile_start + row;
        float val = 0.0f;
        if (token < valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical != kInvalidBlockTableEntry) {
                val = ops::k_load(k_cache, physical, kv_head_idx, n_kv_heads, tok_in_block, col);
            }
        }
        smem_k[idx] = __float2half(val);
    }
}

// V staging: same shape and walk as K_tile (V layout in paged cache is the
// same family; the difference is in ops::v_load resolving the right offset
// and dequant). We stage V in its natural [K_TILE_N tokens, HEAD_SIZE]
// layout in smem; the V matmul uses load_ldmatrix_trans to transpose on
// load (since scores·V wants the K dim as inner, V_smem's K dim is row).
template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
static __device__ __forceinline__ void stage_v_tile(
        __half        * __restrict__ smem_v,
        const void    * __restrict__ v_cache,
        const int     * __restrict__ seq_block_table,
        int            k_tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            tid) {
    using ops = paged_cache_ops<CACHE_TYPE, HEAD_SIZE, BLOCK_SIZE>;
    #pragma unroll
    for (int idx = tid; idx < K_TILE_N * HEAD_SIZE; idx += TILE_NUM_THREADS) {
        const int row   = idx / HEAD_SIZE;
        const int col   = idx % HEAD_SIZE;
        const int token = k_tile_start + row;
        float val = 0.0f;
        if (token < valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical != kInvalidBlockTableEntry) {
                val = ops::v_load(v_cache, physical, kv_head_idx, n_kv_heads, tok_in_block, col);
            }
        }
        smem_v[idx] = __float2half(val);
    }
}

// ── kernel ──────────────────────────────────────────────────────────────

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
#if defined(AMD_WMMA_AVAILABLE)
    static_assert(HEAD_SIZE % K_INNER == 0, "HEAD_SIZE must be multiple of K_INNER=16");
    constexpr int N_INNER = HEAD_SIZE / K_INNER;  // 8 for HS=128

    const int head_idx   = blockIdx.x;
    const int seq_idx    = blockIdx.y;
    const int q_tile_idx = blockIdx.z;
    const int tid        = threadIdx.x;

    const int q_len = q_lens[seq_idx];
    const int q_tile_start = q_tile_idx * Q_TILE_M;
    if (q_tile_start >= q_len) {
        return;
    }
    const int q_tile_actual = (q_tile_start + Q_TILE_M <= q_len) ? Q_TILE_M : (q_len - q_tile_start);

    const int kv_head_idx       = head_idx / (n_heads / n_kv_heads);
    const int ctx_len_after_q   = context_lens[seq_idx];
    const int * seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Per-seq offset into packed Q / out tensors.
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) {
        seq_q_offset += (size_t) q_lens[s];
    }

    // Absolute position of the first Q row in this tile (for causal mask).
    const int q_pos_base = (ctx_len_after_q - q_len) + q_tile_start;

    // Shared memory layout (single warp per block):
    //   smem_q [Q_TILE_M * HEAD_SIZE]  half  = 4 KiB
    //   smem_k [K_TILE_N * HEAD_SIZE]  half  = 4 KiB
    //   smem_v [K_TILE_N * HEAD_SIZE]  half  = 4 KiB
    // Total = 12 KiB / block. Well within LDS budget on RDNA4.
    extern __shared__ unsigned char smem_raw[];
    __half * smem_q = (__half *)(smem_raw);
    __half * smem_k = smem_q + Q_TILE_M * HEAD_SIZE;
    __half * smem_v = smem_k + K_TILE_N * HEAD_SIZE;

    // Stage Q once (it's fixed across the K-tile loop).
    const size_t q_global_base = ((seq_q_offset + (size_t) q_tile_start) * (size_t) n_heads + (size_t) head_idx)
                                 * (size_t) HEAD_SIZE;
    stage_q_tile<HEAD_SIZE>(smem_q, q, q_global_base, q_tile_actual, n_heads, tid);
    __syncthreads();

    // Load Q tiles into registers (reused across all K iterations).
    // Q[16 Q rows, HEAD_SIZE] tiled into 8 tile<16, 8, half2> each covering
    // HEAD_SIZE cols [n*16 .. n*16+15] for the 16 Q rows.
    tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> Q_tiles[N_INNER];
    #pragma unroll
    for (int n = 0; n < N_INNER; ++n) {
        const half2 * src = (const half2 *)(smem_q + n * K_INNER);
        // Stride in half2: HEAD_SIZE / 2 (since each row is HEAD_SIZE halfs = HEAD_SIZE/2 half2).
        load_ldmatrix(Q_tiles[n], src, HEAD_SIZE / 2);
    }

    // Online softmax state. Each thread owns 8 elements of one Q row (the row
    // = tid % 16); pair lane (tid + 16) owns the other 8 cols of the same row.
    // Per-row running_max / running_sum live in scalars, shared between the
    // pair via the row-reduce shfl. acc[N_INNER] holds the 16×16 output tiles
    // (one per HEAD_SIZE 16-col block).
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    tile<16, 16, float, DATA_LAYOUT_I_MAJOR> acc[N_INNER];
    #pragma unroll
    for (int n = 0; n < N_INNER; ++n) {
        #pragma unroll
        for (int e = 0; e < acc[n].ne; ++e) {
            acc[n].x[e] = 0.0f;
        }
    }

    // Maximum K position any Q row in this tile attends to (causal):
    // last Q row's absolute pos. Tokens [0, valid_ctx) are at most visible.
    const int q_pos_last = q_pos_base + (q_tile_actual - 1);
    const int valid_ctx  = q_pos_last + 1;

    // ── K-tile loop ─────────────────────────────────────────────────────
    for (int k_tile_start = 0; k_tile_start < valid_ctx; k_tile_start += K_TILE_N) {
        // Stage K tile (with paged-block gather + optional dequant via ops).
        stage_k_tile<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_k, k_cache, seq_block_table, k_tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, tid);
        __syncthreads();

        // scores[16, 16] = Q[16, HEAD_SIZE] · K^T[HEAD_SIZE, 16]
        // (WMMA's natural C = A·B^T; A=Q, B=K both [16, K_INNER], so the
        // K_INNER dim becomes the contracted dim.)
        tile<16, 16, float, DATA_LAYOUT_I_MAJOR> scores;
        #pragma unroll
        for (int e = 0; e < scores.ne; ++e) {
            scores.x[e] = 0.0f;
        }

        #pragma unroll
        for (int n = 0; n < N_INNER; ++n) {
            tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> K_tile;
            const half2 * src = (const half2 *)(smem_k + n * K_INNER);
            load_ldmatrix(K_tile, src, HEAD_SIZE / 2);
            mma(scores, Q_tiles[n], K_tile);
        }

        // Apply scale + causal mask. Each thread inspects its 8 elements:
        //   row = scores.get_i(l) = tid % 16  (same for all l)
        //   col = scores.get_j(l) = 8*(tid/16) + l
        const int row = tid % 16;          // Q row this lane owns
        const int q_pos = q_pos_base + row;
        const bool row_valid = (row < q_tile_actual);

        #pragma unroll
        for (int l = 0; l < scores.ne; ++l) {
            const int col   = 8 * (tid / 16) + l;
            const int k_pos = k_tile_start + col;
            const bool visible = row_valid && (k_pos <= q_pos) && (k_pos < valid_ctx);
            scores.x[l] = visible ? (scores.x[l] * scale) : -INFINITY;
        }

        // Per-row max: 8-wide local max + shfl_xor(mask=16) with the
        // pair-lane that owns cols 8..15 (vs 0..7) of the same row.
        float local_max = -INFINITY;
        #pragma unroll
        for (int l = 0; l < scores.ne; ++l) {
            local_max = max(local_max, scores.x[l]);
        }
        const float row_max = max(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 16));

        const float new_max = max(running_max, row_max);

        // Rescale running state if needed.
        float rescale = 1.0f;
        if (running_max > -INFINITY) {
            rescale = __expf(running_max - new_max);
            running_sum *= rescale;
            #pragma unroll
            for (int n = 0; n < N_INNER; ++n) {
                #pragma unroll
                for (int e = 0; e < acc[n].ne; ++e) {
                    acc[n].x[e] *= rescale;
                }
            }
        }

        // exp(scores - new_max), local sum + shfl_xor for per-row sum.
        float local_sum = 0.0f;
        #pragma unroll
        for (int l = 0; l < scores.ne; ++l) {
            const float e = (scores.x[l] == -INFINITY) ? 0.0f : __expf(scores.x[l] - new_max);
            scores.x[l]   = e;
            local_sum   += e;
        }
        const float row_sum = local_sum + __shfl_xor_sync(0xFFFFFFFF, local_sum, 16);
        running_sum += row_sum;
        running_max  = new_max;

        // Stage V tile.
        stage_v_tile<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_v, v_cache, seq_block_table, k_tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, tid);
        __syncthreads();

        // Convert scores (f32) to half tile for the matmul. Note: scores
        // is in tile<16,16,float,I_MAJOR>; we pack to tile<16,8,half2,I_MAJOR>
        // by reinterpreting pairs of consecutive cols (l, l+ne/2 in mem
        // layout) — but get_j(l) is contiguous, so pair l and l+1 in mem.
        tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> scores_h;
        static_assert(decltype(scores_h)::ne == 4, "expected 4 half2 per thread for tile<16,8,half2>");
        // scores.ne == 8 (one f32 per col 0..7 of this lane's row half).
        // scores_h.ne == 4 (one half2 per 2-col pair).
        // Lane mapping for tile<16, 8, half2>: get_j(l) = 2*(tid/16) + l for l in [0..??]
        // Wait — looking at mma.cuh L139-140 for tile<16,8,half2,I_MAJOR>:
        //   get_j(l) = 2*(tid/16) + l    (AMD_MFMA branch, but is this AMD_WMMA?)
        //
        // For AMD_WMMA tile<16,8,T>: get_j(l) = ne*(tid/16) + l = 4*(tid/16) + l (ne=4).
        // So thread tid owns cols 4*(tid/16) + 0..3 for half2 layout, which is
        // cols 8*(tid/16) + 0..7 for half layout. Matches scores's cols.
        // Pack: scores_h.x[l] = make_half2(scores.x[2*l], scores.x[2*l+1]).
        #pragma unroll
        for (int l = 0; l < scores_h.ne; ++l) {
            scores_h.x[l] = __floats2half2_rn(scores.x[2*l], scores.x[2*l+1]);
        }

        // acc[N_INNER] += scores_h · V[K, HEAD_SIZE].
        // We want output[i, d] = sum_k scores[i,k] * V[k,d]; mma computes
        // D = A · B^T, so B[d,k] = V[k,d] — i.e., we load V transposed.
        #pragma unroll
        for (int n = 0; n < N_INNER; ++n) {
            tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> V_tile;
            // V_smem layout: [K_TILE_N tokens, HEAD_SIZE cols], half stride.
            // For load_ldmatrix_trans, xs0 is (const half2 *) base; the
            // transposed load picks up V[k, d] as V_tile[d-block, k] —
            // exactly what we need for B in the matmul.
            const half2 * src = (const half2 *)(smem_v + n * K_INNER);
            load_ldmatrix_trans(V_tile, src, HEAD_SIZE / 2);
            mma(acc[n], scores_h, V_tile);
        }

        __syncthreads();  // before next iter overwrites smem_k / smem_v
    }

    // ── output writeback ───────────────────────────────────────────────
    // acc[n] holds 16 Q rows × 16 cols of HEAD_SIZE[n*16..n*16+15] in f32.
    // Layout: get_i(l) = tid%16 (Q row), get_j(l) = 8*(tid/16) + l (HEAD col).
    const float inv_sum = 1.0f / (running_sum + 1e-6f);
    const int row_out   = tid % 16;
    if (row_out < q_tile_actual) {
        const int q_row_global = q_tile_start + row_out;
        const size_t out_row_base =
            ((seq_q_offset + (size_t) q_row_global) * (size_t) n_heads + (size_t) head_idx) * (size_t) HEAD_SIZE;
        #pragma unroll
        for (int n = 0; n < N_INNER; ++n) {
            #pragma unroll
            for (int l = 0; l < acc[n].ne; ++l) {
                const int d = n * K_INNER + 8 * (tid / 16) + l;
                out[out_row_base + (size_t) d] = __float2half(acc[n].x[l] * inv_sum);
            }
        }
    }
#else
    // Non-WMMA hardware should not be dispatched here; existing scalar
    // kernel is the fallback. NO_DEVICE_CODE traps if we somehow are.
    GGML_UNUSED(out); GGML_UNUSED(q); GGML_UNUSED(k_cache); GGML_UNUSED(v_cache);
    GGML_UNUSED(block_tables); GGML_UNUSED(context_lens); GGML_UNUSED(q_lens);
    GGML_UNUSED(max_blocks_per_seq); GGML_UNUSED(n_kv_heads); GGML_UNUSED(n_heads);
    GGML_UNUSED(scale);
    NO_DEVICE_CODE;
#endif // AMD_WMMA_AVAILABLE
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

    const int n_q_tiles = (max_q_len + Q_TILE_M - 1) / Q_TILE_M;

    dim3 grid(n_heads, num_seqs, n_q_tiles);
    dim3 block(TILE_NUM_THREADS);

    const size_t smem_bytes = (size_t)(Q_TILE_M + 2 * K_TILE_N) * (size_t) HEAD_SIZE * sizeof(__half);

    mt_paged_attention_tile_kernel<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>
        <<<grid, block, smem_bytes, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, context_lens, q_lens,
            max_blocks_per_seq, n_kv_heads, n_heads, scale);
}

// Explicit instantiations.
template void launch_paged_attn_tile<128, 16, GGML_TYPE_F16>(
        __half *, const __half *, const void *, const void *,
        const int32_t *, const int32_t *, const int32_t *,
        int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attn_tile<128, 16, GGML_TYPE_TURBO4_0>(
        __half *, const __half *, const void *, const void *,
        const int32_t *, const int32_t *, const int32_t *,
        int, int, int, int, int, float, cudaStream_t);
// HEAD_SIZE=256 — Qwen3.5/3.6 paged-attn shape (n_embd_head_v=128 doubled
// per-q-row by the layout). Uses 16 mma ops per (Q tile, K tile) pair.
template void launch_paged_attn_tile<256, 16, GGML_TYPE_F16>(
        __half *, const __half *, const void *, const void *,
        const int32_t *, const int32_t *, const int32_t *,
        int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attn_tile<256, 16, GGML_TYPE_TURBO4_0>(
        __half *, const __half *, const void *, const void *,
        const int32_t *, const int32_t *, const int32_t *,
        int, int, int, int, int, float, cudaStream_t);

} // namespace mt
