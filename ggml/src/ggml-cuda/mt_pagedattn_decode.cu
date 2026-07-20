// Flash-decode paged-attention kernel — see mt_pagedattn_decode.cuh.
//
// Design (MAD-185):
//   • Grid (n_heads, n_seqs, num_chunks). Each block handles one (head,
//     seq, kv_chunk) and produces a per-chunk partial via online softmax.
//   • A separate reduce kernel merges the chunk partials per (head, seq).
//   • CHUNK_KV = 1024 tokens/block, scanned in K_TILE_N=16 sub-chunks.
//   • Q-vector loaded once per block into smem and broadcast.
//   • K/V sub-chunks staged into smem via the same paged-block walk
//     used by the tile kernel; TURBO4_0 path is cooperative dequant
//     (one norm load per qblock, broadcast across 32 lanes).
//
// Why a new kernel instead of extending tile.cu:
//   • At q_len=1 WMMA tiles waste 15/16 of throughput; decode is memory-
//     bound on KV, not compute-bound on FLOPS.
//   • The tile kernel uses a single grid block per (head, seq) — at
//     400K ctx that's 400000/K_TILE_N = 25000 serial sub-chunks per
//     block. Split-K (num_chunks blocks per head) is the real win.
//
// Open follow-ups (not in this kernel yet):
//   • GQA fanout: today each head_idx is its own block, so the same KV
//     is read GQA× redundantly. Reading once per kv_head and computing
//     for all sharing heads in one block would 4× memory efficiency on
//     Qwen3.6 (n_heads=32, n_kv_heads=8). Follow-up Jira.
//   • Vectorized half2 / f16x4 loads for smem_q / smem_k / smem_v.

#include "mt_pagedattn_decode.cuh"
#include "mt_pagedattn_ops.cuh"
#include "turbo-quant.cuh"
#include "mma.cuh"

#include <cmath>
#include <cstdlib>

namespace mt {

using namespace ggml_cuda_mma;

// Finite softmax-mask sentinel (mirrors mt_pagedattn_tile.cu). This TU is built
// with -ffast-math (-ffinite-math-only), so IEEE negative-infinity is UB — the compiler
// may delete -inf materialization / equality tests. -1e30f is finite and safely
// below any real logit; expf(-1e30f - x) underflows to 0.
static constexpr float SOFTMAX_MASK_VAL = -1.0e30f;

// MAD-180 follow-up (2026-05-18): WMMA decode kernel for HEAD_SIZE=128/256.
//
// The original mt_paged_attention_decode_kernel was designed BEFORE the GQA
// fanout fix — its header comment explicitly noted "At q_len=1 WMMA tiles
// waste 15/16 of throughput; decode is memory-bound on KV, not compute-bound
// on FLOPS." That rationale no longer holds: with GQA fanout (one block per
// kv_head × num_queries_per_kv q-heads × q_len) the per-block query count is
// up to 16, which is exactly the WMMA tile height. WMMA is now fully utilized.
//
// Toggle via env var GGML_PAGED_DECODE_WMMA (default ON when the host has
// WMMA — i.e. RDNA4 gfx1201). Falls back to the scalar kernel automatically
// on gfx1030 / older where amd_wmma_available is false.
static int get_paged_decode_wmma_mode() {
    static int mode = -1;
    if (mode < 0) {
        const char * env = std::getenv("GGML_PAGED_DECODE_WMMA");
        mode = (env == nullptr || env[0] != '0') ? 1 : 0;
    }
    return mode;
}

// Sub-tile size for staging KV — matches the tile kernel so the
// cooperative TURBO4 dequant logic (32 lanes × 4 elements per qblock,
// HEAD_SIZE/128 qblocks per token) maps cleanly.
static constexpr int DECODE_K_TILE_N = 16;

// Per-block KV chunk. Each grid block scans CHUNK_KV tokens in
// DECODE_K_TILE_N steps.
//
// Trade-off: larger → fewer blocks, less reduction overhead, less
// scratch memory. Smaller → more parallelism (better CU coverage).
// 1024 = 64 sub-chunks/block. At 400K ctx → ~400 chunks × n_heads = 12800
// blocks (well above RDNA4 R9700's ~240 concurrent block ceiling, so we
// run in waves and amortize scratch).
static constexpr int CHUNK_KV = 128;  // MAD-301A: was 1024 (tuned for 400K ctx / RDNA4 240-block ceiling); too coarse for gfx803's 36 CUs at mid ctx (ctx 1751 -> only 2 chunks -> few blocks -> CU under-utilization). CHUNK_KV sweep on gfx803: 128 (56/38 t/s) > 256 (52/37) > 512 (45/23) @ctx 1751/10501. TODO: make arch-tunable.

// Threads/block. 4 warps (32 lanes each) — fits LDS budget for 2× tile
// (smem_k + smem_v at HEAD_SIZE=128 = 4 KiB each = 8 KiB) plus Q+logits.
static constexpr int DECODE_NUM_THREADS = 128;
static constexpr int DECODE_NUM_WARPS   = DECODE_NUM_THREADS / WARP_SIZE;

int paged_attn_decode_num_chunks(int max_ctx_len) {
    return (max_ctx_len + CHUNK_KV - 1) / CHUNK_KV;
}

// ── KV staging helpers (file-private) ──────────────────────────────────
//
// Mirror stage_k_tile / stage_v_tile / coop_stage_turbo4_tile in the
// tile kernel. Duplicated rather than shared via a header to avoid
// modifying the tile kernel — kept tight and reviewed against the
// originals.

template <int HEAD_SIZE, int BLOCK_SIZE>
static __device__ __forceinline__ void decode_stage_kv_f16(
        __half        * __restrict__ smem_dst,
        const __half  * __restrict__ src_cache_as_half,  // for both K and V calls
        const int     * __restrict__ seq_block_table,
        int            tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        bool           is_v,
        int            tid) {
    // Coalesced staging for F16 K/V caches.
    //
    // K cache layout: [blocks, kv_heads, HEAD_SIZE/K_X, BLOCK_SIZE, K_X], K_X=8.
    //   Innermost K_X=8 fp16s are contiguous. For a 32-lane warp to coalesce
    //   into 64 contiguous bytes, adjacent lanes must vary across (token_in_block,
    //   K_X) — i.e. each warp reads 4 tokens × 8 K_X per d_outer iteration.
    //
    // V cache layout: [blocks, kv_heads, HEAD_SIZE, BLOCK_SIZE]
    //   Innermost is token_in_block (16 fp16s = 32 bytes contiguous per d). For
    //   32-lane coalescing, each warp packs 2 d values × 16 tokens = 64 bytes.
    //
    // Why this matters (MAD-188 perf hunt, 2026-05-18): the prior version had
    // each thread load 1 element with iteration order (token, d), so within a
    // warp 32 lanes read 32 elements scattered across 4 cache lines (each line
    // only ~8 lanes used). That's ~4× the BW transactions vs the data volume,
    // and matched the observed 4× decode regression on paged-tiered vs stock.
    static_assert(DECODE_K_TILE_N == BLOCK_SIZE,
                  "decode_stage_kv_f16 assumes one staged tile == one logical block "
                  "(chunk_start and sub_start are both block-aligned because "
                  "CHUNK_KV % BLOCK_SIZE == 0 and DECODE_K_TILE_N == BLOCK_SIZE)");
    constexpr int K_X = 16 / sizeof(__half);  // 8 fp16 = 16 contiguous bytes
    static_assert(HEAD_SIZE % K_X == 0, "HEAD_SIZE must be a multiple of K_X");
    constexpr int NUM_WARPS = DECODE_NUM_THREADS / WARP_SIZE;
    static_assert(NUM_WARPS * 4 == BLOCK_SIZE,
                  "K stage assumes 4 tokens per warp × NUM_WARPS warps == BLOCK_SIZE (=16 tokens)");

    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    // All DECODE_K_TILE_N=16 tokens in this tile are in one logical block
    // (asserted above). Resolve the physical block once per thread; reuse.
    const int logical_block = tile_start / BLOCK_SIZE;
    const int physical      = seq_block_table[logical_block];
    const bool block_ok     = (physical != kInvalidBlockTableEntry);

    if (!is_v) {
        // K: 4 tokens × K_X=8 elements per warp per d_outer iter.
        //   lane 0..7    → token_in_warp=0, d_inner=0..7
        //   lane 8..15   → token_in_warp=1, d_inner=0..7
        //   lane 16..23  → token_in_warp=2, d_inner=0..7
        //   lane 24..31  → token_in_warp=3, d_inner=0..7
        // These 32 elements are contiguous in memory (only differ in the inner
        // (token_in_block * K_X + d_inner) range). 64-byte coalesced load.
        const int t_in_warp    = lane / K_X;
        const int d_inner      = lane % K_X;
        const int t            = warp * 4 + t_in_warp;          // 0..15 within tile
        const int token        = tile_start + t;
        const int tok_in_block = t;                              // tile is block-aligned
        const bool token_ok    = block_ok && (token < valid_ctx);

        const size_t kv_head_base = (size_t) physical    * n_kv_heads * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                                  + (size_t) kv_head_idx              * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                                  + (size_t) tok_in_block * K_X
                                  + (size_t) d_inner;

        #pragma unroll
        for (int d_outer = 0; d_outer < HEAD_SIZE / K_X; ++d_outer) {
            __half val = (__half) 0;
            if (token_ok) {
                const size_t off = kv_head_base + (size_t) d_outer * BLOCK_SIZE * K_X;
                val = src_cache_as_half[off];
            }
            const int d = d_outer * K_X + d_inner;
            smem_dst[t * HEAD_SIZE + d] = val;
        }
    } else {
        // V: 2 d values × BLOCK_SIZE=16 tokens per warp per outer iter.
        //   lane 0..15  → d_in_warp=0, t=0..15
        //   lane 16..31 → d_in_warp=1, t=0..15
        // Per warp iter: 32 contiguous fp16 (64 bytes) — coalesced.
        // 4 warps × 2 d = 8 d per outer iter. Loop HEAD_SIZE/8 outer iters.
        constexpr int D_PER_WARP   = 2;
        constexpr int D_PER_OUTER  = NUM_WARPS * D_PER_WARP;   // 8
        constexpr int OUTER_ITERS  = HEAD_SIZE / D_PER_OUTER;
        static_assert(HEAD_SIZE % D_PER_OUTER == 0, "HEAD_SIZE must be multiple of NUM_WARPS*2");

        const int d_in_warp = lane / BLOCK_SIZE;   // 0 or 1
        const int t         = lane % BLOCK_SIZE;   // 0..15
        const int token     = tile_start + t;
        const bool token_ok = block_ok && (token < valid_ctx);

        const size_t kv_head_base = (size_t) physical    * n_kv_heads * HEAD_SIZE * BLOCK_SIZE
                                  + (size_t) kv_head_idx              * HEAD_SIZE * BLOCK_SIZE;

        #pragma unroll
        for (int outer = 0; outer < OUTER_ITERS; ++outer) {
            const int d = outer * D_PER_OUTER + warp * D_PER_WARP + d_in_warp;
            __half val = (__half) 0;
            if (token_ok) {
                const size_t off = kv_head_base + (size_t) d * BLOCK_SIZE + (size_t) t;
                val = src_cache_as_half[off];
            }
            smem_dst[t * HEAD_SIZE + d] = val;
        }
    }
}

template <int HEAD_SIZE, int BLOCK_SIZE>
static __device__ __forceinline__ void decode_coop_stage_turbo4(
        __half        * __restrict__ smem_dst,
        const void    * __restrict__ cache,
        const int     * __restrict__ seq_block_table,
        int            tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            warp_id,
        int            lane_id) {
    constexpr int Q_BLOCK            = QK_TURBO4;
    constexpr int QBLOCKS_PER_TOKEN  = HEAD_SIZE / Q_BLOCK;
    constexpr int N_QBLOCKS_PER_TILE = DECODE_K_TILE_N * QBLOCKS_PER_TOKEN;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be multiple of QK_TURBO4=128");
    static_assert(Q_BLOCK == 128, "cooperative dequant expects QK_TURBO4=128 (32 lanes × 4 elements)");

    const block_turbo4_0 * blocks = (const block_turbo4_0 *) cache;

    #pragma unroll
    for (int qb = warp_id; qb < N_QBLOCKS_PER_TILE; qb += DECODE_NUM_WARPS) {
        const int row         = qb / QBLOCKS_PER_TOKEN;
        const int qb_in_token = qb % QBLOCKS_PER_TOKEN;
        const int token       = tile_start + row;

        const block_turbo4_0 * blk = nullptr;
        float norm_f = 0.0f;

        if (token < valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical != kInvalidBlockTableEntry) {
                const int64_t ib = ((int64_t) physical * n_kv_heads + kv_head_idx) * BLOCK_SIZE * QBLOCKS_PER_TOKEN
                                 + (int64_t) tok_in_block * QBLOCKS_PER_TOKEN
                                 + (int64_t) qb_in_token;
                blk = &blocks[ib];
                if (lane_id == 0) {
                    norm_f = __half2float(blk->norm);
                }
            }
        }
        norm_f = __shfl_sync(0xFFFFFFFF, norm_f, 0, WARP_SIZE);

        uint16_t packed = 0;
        if (blk != nullptr) {
            packed = *(const uint16_t *)(blk->qs + 2 * lane_id);
        }

        const int smem_row_base = row * HEAD_SIZE;
        const int smem_col_base = qb_in_token * Q_BLOCK + lane_id * 4;

        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t idx_nib = (packed >> (l * 4)) & 0xF;
            const float val = TURBO_CENTROIDS_4BIT[idx_nib] * norm_f;
            smem_dst[smem_row_base + smem_col_base + l] = __float2half(val);
        }
    }
}

// TURBO4_64 cooperative dequant for the decode tile (MAD-301C Lever B).
// Native head_dim-64: 64-element block => 32 lanes × 2 elements (1 qs byte/lane,
// 2 nibbles). Matches mt_scatter_kv_turbo4_64_kernel packing.
template <int HEAD_SIZE, int BLOCK_SIZE>
static __device__ __forceinline__ void decode_coop_stage_turbo4_64(
        __half        * __restrict__ smem_dst,
        const void    * __restrict__ cache,
        const int     * __restrict__ seq_block_table,
        int            tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            warp_id,
        int            lane_id) {
    constexpr int Q_BLOCK            = QK_TURBO4_64;  // 64
    constexpr int QBLOCKS_PER_TOKEN  = HEAD_SIZE / Q_BLOCK;
    constexpr int N_QBLOCKS_PER_TILE = DECODE_K_TILE_N * QBLOCKS_PER_TOKEN;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be multiple of QK_TURBO4_64=64");
    static_assert(Q_BLOCK == 64, "cooperative dequant expects QK_TURBO4_64=64 (32 lanes × 2 elements)");

    const block_turbo4_64 * blocks = (const block_turbo4_64 *) cache;

    #pragma unroll
    for (int qb = warp_id; qb < N_QBLOCKS_PER_TILE; qb += DECODE_NUM_WARPS) {
        const int row         = qb / QBLOCKS_PER_TOKEN;
        const int qb_in_token = qb % QBLOCKS_PER_TOKEN;
        const int token       = tile_start + row;

        const block_turbo4_64 * blk = nullptr;
        float norm_f = 0.0f;

        if (token < valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical != kInvalidBlockTableEntry) {
                const int64_t ib = ((int64_t) physical * n_kv_heads + kv_head_idx) * BLOCK_SIZE * QBLOCKS_PER_TOKEN
                                 + (int64_t) tok_in_block * QBLOCKS_PER_TOKEN
                                 + (int64_t) qb_in_token;
                blk = &blocks[ib];
                if (lane_id == 0) {
                    norm_f = __half2float(blk->norm);
                }
            }
        }
        norm_f = __shfl_sync(0xFFFFFFFF, norm_f, 0, WARP_SIZE);

        uint8_t packed = 0;
        if (blk != nullptr) {
            packed = blk->qs[lane_id];  // 1 byte = 2 nibbles (elements 2*lane, 2*lane+1)
        }

        const int smem_row_base = row * HEAD_SIZE;
        const int smem_col_base = qb_in_token * Q_BLOCK + lane_id * 2;

        #pragma unroll
        for (int l = 0; l < 2; ++l) {
            const uint8_t idx_nib = (packed >> (l * 4)) & 0xF;
            const float val = TURBO_CENTROIDS_4BIT_N64[idx_nib] * norm_f;
            smem_dst[smem_row_base + smem_col_base + l] = __float2half(val);
        }
    }
}

// TURBO3_0 cooperative dequant for the decode tile. Same threading shape as
// decode_coop_stage_turbo4 (32 lanes × 4 elements per qblock), but unpacks
// the 3-bit index as (qs low-2 | signs hi-1).
template <int HEAD_SIZE, int BLOCK_SIZE>
static __device__ __forceinline__ void decode_coop_stage_turbo3(
        __half        * __restrict__ smem_dst,
        const void    * __restrict__ cache,
        const int     * __restrict__ seq_block_table,
        int            tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            warp_id,
        int            lane_id) {
    constexpr int Q_BLOCK            = QK_TURBO3;
    constexpr int QBLOCKS_PER_TOKEN  = HEAD_SIZE / Q_BLOCK;
    constexpr int N_QBLOCKS_PER_TILE = DECODE_K_TILE_N * QBLOCKS_PER_TOKEN;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be multiple of QK_TURBO3=128");
    static_assert(Q_BLOCK == 128, "cooperative dequant expects QK_TURBO3=128 (32 lanes × 4 elements)");

    const block_turbo3_0 * blocks = (const block_turbo3_0 *) cache;

    #pragma unroll
    for (int qb = warp_id; qb < N_QBLOCKS_PER_TILE; qb += DECODE_NUM_WARPS) {
        const int row         = qb / QBLOCKS_PER_TOKEN;
        const int qb_in_token = qb % QBLOCKS_PER_TOKEN;
        const int token       = tile_start + row;

        const block_turbo3_0 * blk = nullptr;
        float norm_f = 0.0f;

        if (token < valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical != kInvalidBlockTableEntry) {
                const int64_t ib = ((int64_t) physical * n_kv_heads + kv_head_idx) * BLOCK_SIZE * QBLOCKS_PER_TOKEN
                                 + (int64_t) tok_in_block * QBLOCKS_PER_TOKEN
                                 + (int64_t) qb_in_token;
                blk = &blocks[ib];
                if (lane_id == 0) {
                    norm_f = __half2float(blk->norm);
                }
            }
        }
        norm_f = __shfl_sync(0xFFFFFFFF, norm_f, 0, WARP_SIZE);

        uint8_t qs_byte    = 0;
        uint8_t signs_byte = 0;
        if (blk != nullptr) {
            qs_byte    = blk->qs[lane_id];
            signs_byte = blk->signs[lane_id / 2];
        }
        const int signs_shift_base = (lane_id % 2) * 4;

        const int smem_row_base = row * HEAD_SIZE;
        const int smem_col_base = qb_in_token * Q_BLOCK + lane_id * 4;

        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint8_t low2 = (qs_byte    >> (l * 2)) & 0x3;
            const uint8_t hi1  = (signs_byte >> (signs_shift_base + l)) & 0x1;
            const uint8_t idx  = low2 | (hi1 << 2);
            const float   val  = TURBO_CENTROIDS_3BIT[idx] * norm_f;
            smem_dst[smem_row_base + smem_col_base + l] = __float2half(val);
        }
    }
}

template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
static __device__ __forceinline__ void decode_stage_k(
        __half        * __restrict__ smem_dst,
        const void    * __restrict__ cache,
        const int     * __restrict__ seq_block_table,
        int            tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            tid,
        int            warp_id,
        int            lane_id) {
    if constexpr (CACHE_TYPE == GGML_TYPE_TURBO4_0) {
        decode_coop_stage_turbo4<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, warp_id, lane_id);
    } else if constexpr (CACHE_TYPE == GGML_TYPE_TURBO3_0) {
        decode_coop_stage_turbo3<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, warp_id, lane_id);
    } else if constexpr (CACHE_TYPE == GGML_TYPE_TURBO4_64) {
        decode_coop_stage_turbo4_64<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, warp_id, lane_id);
    } else {
        decode_stage_kv_f16<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, (const __half *) cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, /*is_v=*/false, tid);
    }
}

template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
static __device__ __forceinline__ void decode_stage_v(
        __half        * __restrict__ smem_dst,
        const void    * __restrict__ cache,
        const int     * __restrict__ seq_block_table,
        int            tile_start,
        int            valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            tid,
        int            warp_id,
        int            lane_id) {
    if constexpr (CACHE_TYPE == GGML_TYPE_TURBO4_0) {
        decode_coop_stage_turbo4<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, warp_id, lane_id);
    } else if constexpr (CACHE_TYPE == GGML_TYPE_TURBO3_0) {
        decode_coop_stage_turbo3<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, warp_id, lane_id);
    } else if constexpr (CACHE_TYPE == GGML_TYPE_TURBO4_64) {
        decode_coop_stage_turbo4_64<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, warp_id, lane_id);
    } else {
        decode_stage_kv_f16<HEAD_SIZE, BLOCK_SIZE>(
            smem_dst, (const __half *) cache, seq_block_table, tile_start, valid_ctx,
            kv_head_idx, n_kv_heads, /*is_v=*/true, tid);
    }
}

// ── reductions (file-private, lifted from mt_pagedattn.cu) ─────────────

template <typename T>
__device__ __forceinline__ T decode_warp_reduce_sum(T v) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T decode_warp_reduce_max(T v) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        v = max(v, __shfl_xor_sync(0xFFFFFFFF, v, mask, WARP_SIZE));
    }
    return v;
}

static __device__ __forceinline__ float decode_block_reduce_sum(
        float v, float * red_smem) {
    const int wid  = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    v = decode_warp_reduce_sum(v);
    if (lane == 0) red_smem[wid] = v;
    __syncthreads();
    float partial = (threadIdx.x < DECODE_NUM_WARPS) ? red_smem[lane] : 0.0f;
    if (wid == 0) {
        partial = decode_warp_reduce_sum(partial);
        if (lane == 0) red_smem[0] = partial;
    }
    __syncthreads();
    return red_smem[0];
}

static __device__ __forceinline__ float decode_block_reduce_max(
        float v, float * red_smem) {
    const int wid  = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    v = decode_warp_reduce_max(v);
    if (lane == 0) red_smem[wid] = v;
    __syncthreads();
    float partial = (threadIdx.x < DECODE_NUM_WARPS) ? red_smem[lane] : SOFTMAX_MASK_VAL;
    if (wid == 0) {
        partial = decode_warp_reduce_max(partial);
        if (lane == 0) red_smem[0] = partial;
    }
    __syncthreads();
    return red_smem[0];
}

// Compile-time max QUERIES handled per block = num_queries_per_kv * q_len.
// With GQA fanout (2026-05-18), one block processes all q_heads sharing a
// kv_head AND all q_tokens in the batch. For Qwen3.5/3.6:
//   GQA=4 + q_len=4 (MTP spec-decode draft) → 16 queries  ← MUST fit
//   GQA=8 + q_len=1 (35B-A3B pure decode)    →  8 queries
//   GQA=8 + q_len=4 (35B + MTP)              → 32 queries ← won't fit, scalar
// Bumped from 8 → 16 to cover the Qwen3.5/3.6 MTP common case. Beyond 16
// the dispatch gate sends it to the scalar fallback. Larger q_len batches
// are prefill territory and go to the tile kernel.

// ── Pass 1: per-chunk partial kernel ───────────────────────────────────
//
// Processes ALL q_len query positions in a single (head, seq, chunk)
// block. KV reads are shared across queries; each query gets its own
// causal mask, online-softmax state, and partial output slot.
//
// Partial layout:
//   partials[((h * n_seqs + s) * num_chunks + c) * q_len + qi] →
//     (V_acc[HEAD_SIZE] float, running_max float, running_sum float)
// Inner stride = HEAD_SIZE + 2. The reducer fans out the same way.

template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
__global__ void mt_paged_attention_decode_kernel(
    float          * __restrict__ partials,
    const __half   * __restrict__ q,
    const void     * __restrict__ k_cache,
    const void     * __restrict__ v_cache,
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    int             n_seqs,
    int             num_chunks,
    int             max_q_len,    // uniform stride for partials inner-dim
    float           scale) {
    // GQA fanout (2026-05-18): grid is now (n_kv_heads, n_seqs, num_chunks).
    // Each block handles ALL `num_queries_per_kv` q-heads that share this
    // kv_head, so the staged K/V tile is reused across the GQA group instead
    // of being read GQA× redundantly (one block per q_head, as before). For
    // Qwen3.5-4B (GQA=4) this is a 4× HBM-traffic reduction on decode.
    //
    // Inner query index: qhqi = qh * q_len + qi (qh in 0..nq_per_kv-1,
    // qi in 0..q_len-1). The total per block is num_queries_per_kv * q_len
    // and must fit in DECODE_MAX_Q — the dispatch gate enforces this.
    const int kv_head_idx        = blockIdx.x;
    const int seq_idx            = blockIdx.y;
    const int chunk_idx          = blockIdx.z;
    const int tid                = threadIdx.x;
    const int wid                = tid / WARP_SIZE;
    const int lane               = tid % WARP_SIZE;

    const int q_len              = q_lens[seq_idx];   // 1..DECODE_MAX_Q (dispatch-enforced)
    const int ctx_len_after_q    = context_lens[seq_idx];
    const int num_queries_per_kv = n_heads / n_kv_heads;
    const int head_base          = kv_head_idx * num_queries_per_kv;
    const int total_q            = num_queries_per_kv * q_len;
    const int * seq_block_table  = block_tables + seq_idx * max_blocks_per_seq;

    if (q_len == 0) return;

    // Per-block check — defensive. Dispatch gate enforces both q_len and
    // total_q caps; clamp here in case a future op_param flow sneaks
    // something through.
    if (q_len > DECODE_MAX_Q || total_q > DECODE_MAX_Q) return;

    // Per-seq Q offset (sum of preceding q_lens).
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];

    // Per-query absolute position. qi=0 is the FIRST new query token in
    // the batch, qi=q_len-1 is the trailing one. Each qi has its own
    // causal mask: it sees tokens [0, q_pos_first + qi].
    const int q_pos_first    = ctx_len_after_q - q_len;
    const int valid_ctx_max  = ctx_len_after_q;   // highest mask any qi can reach

    const int chunk_start = chunk_idx * CHUNK_KV;

    // Per-head partial base (chunk-relative). Each block writes
    // num_queries_per_kv slots in this dimension. Layout matches the old
    // "one head per block" world so the reduce kernel is unchanged.
    auto partial_chunk_base_for_head = [&](int head_idx) -> size_t {
        return ((((size_t) head_idx * n_seqs + seq_idx) * num_chunks) + (size_t) chunk_idx)
             * (size_t) max_q_len * (size_t) (HEAD_SIZE + 2);
    };

    if (chunk_start >= valid_ctx_max) {
        // MAD-301A: this chunk is entirely beyond the seq's real context
        // (num_chunks is sized by ALLOCATED ctx, not actual — and with a small
        // CHUNK_KV + large --ctx-size there are thousands of these). The reduce
        // kernel now bounds its loop to ceil(context_lens/CHUNK_KV), so it never
        // reads partials from these chunks — skip the neutral-partial writes and
        // just retire the block. Makes decode cost scale with actual depth.
        return;
    }
    const int chunk_end = min(chunk_start + CHUNK_KV, valid_ctx_max);

    // Shared memory:
    //   smem_q       [q_len * HEAD_SIZE]            __half
    //   smem_k       [DECODE_K_TILE_N * HEAD_SIZE]  __half
    //   smem_v       [DECODE_K_TILE_N * HEAD_SIZE]  __half
    //   smem_logits  [DECODE_MAX_Q * DECODE_K_TILE_N] float
    //   red_smem     [DECODE_NUM_WARPS]             float
    // At HEAD_SIZE=128, q_len=8: 2K + 4K + 4K + 512 + 16 ≈ 11 KiB.
    extern __shared__ unsigned char smem_raw[];
    __half * smem_q      = (__half *)(smem_raw);
    __half * smem_k      = smem_q + DECODE_MAX_Q * HEAD_SIZE;
    __half * smem_v      = smem_k + DECODE_K_TILE_N * HEAD_SIZE;
    float  * smem_logits = (float *)(smem_v + DECODE_K_TILE_N * HEAD_SIZE);
    float  * red_smem    = smem_logits + DECODE_MAX_Q * DECODE_K_TILE_N;

    // ── Stage all (q_head_in_group × q_token) query vectors ──
    // smem_q[qhqi, d] for qhqi = qh * q_len + qi, qh = head_in_group.
    for (int idx = tid; idx < total_q * HEAD_SIZE; idx += DECODE_NUM_THREADS) {
        const int qhqi = idx / HEAD_SIZE;
        const int qh   = qhqi / q_len;
        const int qi   = qhqi % q_len;
        const int d    = idx % HEAD_SIZE;
        const int head_idx = head_base + qh;
        const size_t q_off = ((seq_q_offset + (size_t) qi) * (size_t) n_heads + (size_t) head_idx)
                             * (size_t) HEAD_SIZE + (size_t) d;
        smem_q[qhqi * HEAD_SIZE + d] = q[q_off];
    }
    __syncthreads();

    // Per-thread online-softmax state per (q_head_in_group, q_token) pair.
    // Indexed by qhqi = qh * q_len + qi, range 0..total_q-1, total_q ≤ DECODE_MAX_Q.
    constexpr int VEC_PER_THREAD = (HEAD_SIZE + DECODE_NUM_THREADS - 1) / DECODE_NUM_THREADS;
    float v_acc[VEC_PER_THREAD][DECODE_MAX_Q];
    float running_max[DECODE_MAX_Q];
    float running_sum[DECODE_MAX_Q];
    #pragma unroll
    for (int qhqi = 0; qhqi < DECODE_MAX_Q; ++qhqi) {
        running_max[qhqi] = SOFTMAX_MASK_VAL;
        running_sum[qhqi] = 0.0f;
        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) v_acc[v][qhqi] = 0.0f;
    }

    // Sub-chunk loop: stage K, QK (all queries), softmax (per query),
    // stage V, V@logits (all queries).
    for (int sub_start = chunk_start; sub_start < chunk_end; sub_start += DECODE_K_TILE_N) {
        const int sub_end = min(sub_start + DECODE_K_TILE_N, chunk_end);
        const int sub_len = sub_end - sub_start;

        // ── Stage K (shared by all queries) ──
        decode_stage_k<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_k, k_cache, seq_block_table,
            sub_start, valid_ctx_max,
            kv_head_idx, n_kv_heads, tid, wid, lane);
        __syncthreads();

        // ── QK: 1 warp per (token, all queries in GQA group). For each
        // token slot, one warp computes total_q = num_queries_per_kv * q_len
        // dot products against the same K row. ──
        #pragma unroll
        for (int t_base = 0; t_base < DECODE_K_TILE_N; t_base += DECODE_NUM_WARPS) {
            const int t     = t_base + wid;
            const int token = sub_start + t;
            if (t < DECODE_K_TILE_N) {
                for (int qhqi = 0; qhqi < total_q; ++qhqi) {
                    float qk = 0.0f;
                    if (t < sub_len && token < valid_ctx_max) {
                        #pragma unroll
                        for (int d = lane; d < HEAD_SIZE; d += WARP_SIZE) {
                            const float qv = __half2float(smem_q[qhqi * HEAD_SIZE + d]);
                            const float kv = __half2float(smem_k[t * HEAD_SIZE + d]);
                            qk += qv * kv;
                        }
                        qk = decode_warp_reduce_sum(qk);
                    }
                    if (lane == 0) {
                        const int qi       = qhqi % q_len;   // all q_heads in a group share q_pos
                        const int q_pos_qi = q_pos_first + qi;
                        const bool visible = (t < sub_len) && (token <= q_pos_qi);
                        smem_logits[qhqi * DECODE_K_TILE_N + t] = visible ? (qk * scale) : SOFTMAX_MASK_VAL;
                    }
                }
            }
        }
        __syncthreads();

        // ── Per-query softmax update — one update per (q_head, q_token) pair.
        // total_q ≤ DECODE_MAX_Q, so this loop is short and stable.
        for (int qhqi = 0; qhqi < total_q; ++qhqi) {
            float local_max = (tid < DECODE_K_TILE_N)
                              ? smem_logits[qhqi * DECODE_K_TILE_N + tid]
                              : SOFTMAX_MASK_VAL;
            const float sub_max = decode_block_reduce_max(local_max, red_smem);

            const float new_max = max(running_max[qhqi], sub_max);
            float rescale = 1.0f;
            if (running_max[qhqi] > SOFTMAX_MASK_VAL) {
                rescale = __expf(running_max[qhqi] - new_max);
                running_sum[qhqi] *= rescale;
                #pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; ++v) v_acc[v][qhqi] *= rescale;
            }

            float local_sum = 0.0f;
            if (tid < DECODE_K_TILE_N) {
                const float lg = smem_logits[qhqi * DECODE_K_TILE_N + tid];
                const float e  = (lg == SOFTMAX_MASK_VAL) ? 0.0f : __expf(lg - new_max);
                smem_logits[qhqi * DECODE_K_TILE_N + tid] = e;
                local_sum = e;
            }
            const float sub_sum = decode_block_reduce_sum(local_sum, red_smem);
            running_sum[qhqi] += sub_sum;
            running_max[qhqi]  = new_max;
        }

        // ── Stage V (shared by all queries) ──
        decode_stage_v<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_v, v_cache, seq_block_table,
            sub_start, valid_ctx_max,
            kv_head_idx, n_kv_heads, tid, wid, lane);
        __syncthreads();

        // ── V matmul: v_acc[qhqi, d] += Σ_t softmax[qhqi, t] · V[t, d] ──
        // V is shared across all queries in the GQA group; the inner total_q
        // loop hits registers (smem_logits) and the pre-loaded v_col.
        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * DECODE_NUM_THREADS;
            if (d < HEAD_SIZE) {
                float v_col[DECODE_K_TILE_N];
                #pragma unroll
                for (int t = 0; t < DECODE_K_TILE_N; ++t) {
                    v_col[t] = __half2float(smem_v[t * HEAD_SIZE + d]);
                }
                for (int qhqi = 0; qhqi < total_q; ++qhqi) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < DECODE_K_TILE_N; ++t) {
                        acc += smem_logits[qhqi * DECODE_K_TILE_N + t] * v_col[t];
                    }
                    v_acc[v][qhqi] += acc;
                }
            }
        }
        __syncthreads();  // before next sub's stage_k reuses smem_k
    }

    // ── Write per-(chunk, head, query) partials ──
    // Each (qh, qi) lands in a different head_idx slot of the partials
    // buffer — matches the original "one head per block" layout, so the
    // reduce kernel doesn't need to change.
    for (int qhqi = 0; qhqi < total_q; ++qhqi) {
        const int qh = qhqi / q_len;
        const int qi = qhqi % q_len;
        const int head_idx = head_base + qh;
        const size_t off = partial_chunk_base_for_head(head_idx)
                         + (size_t) qi * (HEAD_SIZE + 2);
        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * DECODE_NUM_THREADS;
            if (d < HEAD_SIZE) partials[off + d] = v_acc[v][qhqi];
        }
        if (tid == 0) {
            partials[off + HEAD_SIZE]     = running_max[qhqi];
            partials[off + HEAD_SIZE + 1] = running_sum[qhqi];
        }
    }
}

// ── Pass 1b: WMMA-accelerated decode kernel ────────────────────────────
//
// Same dispatch contract as mt_paged_attention_decode_kernel (same partials
// layout, same scatter expectations). Uses the ggml_cuda_mma::tile pattern
// from the tile kernel (MAD-180) — 16×16×16 fp16 WMMA on RDNA4 — to compute
// QK and V@logits as proper matmuls instead of scalar dot products with
// per-pair warp_reduce_sum.
//
// One warp per block. Grid (n_kv_heads, n_seqs, num_chunks). With GQA fanout
// the per-block query count `total_q = num_queries_per_kv * q_len` is handled
// by one or more 16-row WMMA tiles. Rows beyond total_q in the final tile produce
// don't-care output that the write-back loop masks.
//
// HEAD_SIZE: 128 (N_INNER=8) and 256 (N_INNER=16). Register footprint:
//   Q_tiles[N_INNER] : 4 half2/lane each
//   acc    [N_INNER] : 8 fp32/lane each → 64 (HS=128) / 128 (HS=256) fp32/lane
//   scores           : 8 fp32/lane
// RDNA4 has 256 VGPR/lane (wave32), so HS=256 fits with slack.
//
// CACHE_TYPE: F16 only for v1. TURBO4_0 still uses the scalar fallback path
// (the cooperative-dequant smem layout doesn't map straight onto load_ldmatrix
// — a follow-up).

template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
__global__ void mt_paged_attention_decode_kernel_wmma(
    float          * __restrict__ partials,
    const __half   * __restrict__ q,
    const void     * __restrict__ k_cache,
    const void     * __restrict__ v_cache,
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    int             n_seqs,
    int             num_chunks,
    int             max_q_len,
    float           scale) {
    static_assert(HEAD_SIZE % 16 == 0, "WMMA decode kernel requires HEAD_SIZE % 16 == 0");
    constexpr int K_INNER = 16;
    constexpr int N_INNER = HEAD_SIZE / K_INNER;
    constexpr int K_TILE_N = DECODE_K_TILE_N;  // 16, matches the staging helper

    const int kv_head_idx = blockIdx.x;
    const int seq_idx     = blockIdx.y;
    const int chunk_idx   = blockIdx.z;
    const int tid         = threadIdx.x;  // 0..127 — 4 warps for staging
    const int wid         = tid / 32;
    const int lane        = tid % 32;

    const int q_len              = q_lens[seq_idx];
    const int ctx_len_after_q    = context_lens[seq_idx];
    const int num_queries_per_kv = n_heads / n_kv_heads;
    const int head_base          = kv_head_idx * num_queries_per_kv;
    const int total_q            = num_queries_per_kv * q_len;
    if (q_len == 0) return;
    if (q_len > DECODE_MAX_Q || total_q > DECODE_MAX_Q) return;

    const int * seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];

    const int q_pos_first   = ctx_len_after_q - q_len;
    const int valid_ctx_max = ctx_len_after_q;
    const int chunk_start   = chunk_idx * CHUNK_KV;

    // DEBUG: lane-0 trace for WMMA decode bug ([[wmma-decode-kernel-bug-2026-05-19]]).
    // Gate fires once per kernel call (head=0, seq=0, chunk=0, warp=0, lane=0,
    // first sub-tile only — see inside the sub-tile loop for the actual dumps).
    // Set to true to re-enable while debugging.
    constexpr bool MT_WMMA_DEBUG_DUMP = false;
    const bool debug_origin =
        MT_WMMA_DEBUG_DUMP &&
        (kv_head_idx == 0) && (seq_idx == 0) && (chunk_idx == 0) &&
        (wid == 0) && (lane == 0);
    if (debug_origin) {
        printf("[WMMA] >>> step ctx=%d q_len=%d total_q=%d chunk_start=%d chunk_end=%d\n",
            ctx_len_after_q, q_len, total_q, chunk_start,
            min(chunk_start + CHUNK_KV, valid_ctx_max));
    }

    auto partial_chunk_base_for_head = [&](int head_idx) -> size_t {
        return ((((size_t) head_idx * n_seqs + seq_idx) * num_chunks) + (size_t) chunk_idx)
             * (size_t) max_q_len * (size_t) (HEAD_SIZE + 2);
    };

    if (chunk_start >= valid_ctx_max) {
        for (int qh = 0; qh < num_queries_per_kv; ++qh) {
            const int head_idx = head_base + qh;
            const size_t base = partial_chunk_base_for_head(head_idx);
            for (int qi = 0; qi < q_len; ++qi) {
                const size_t off = base + (size_t) qi * (HEAD_SIZE + 2);
                for (int d = tid; d < HEAD_SIZE; d += 128) partials[off + d] = 0.0f;
                if (tid == 0) {
                    partials[off + HEAD_SIZE]     = SOFTMAX_MASK_VAL;
                    partials[off + HEAD_SIZE + 1] = 0.0f;
                }
            }
        }
        return;
    }
    const int chunk_end = min(chunk_start + CHUNK_KV, valid_ctx_max);

    // ── Shared memory layout ──
    //   smem_q  [16 * HEAD_SIZE]            __half  — re-staged per row block
    //   smem_k  [DECODE_K_TILE_N * HEAD_SIZE] __half
    //   smem_v  [DECODE_K_TILE_N * HEAD_SIZE] __half
    // At HS=256: 16*256*2 + 16*256*2*2 = 8 KiB + 16 KiB = 24 KiB. Fine.
    extern __shared__ unsigned char smem_raw[];
    __half * smem_q = (__half *)(smem_raw);
    __half * smem_k = smem_q + 16 * HEAD_SIZE;     // 16 rows even when total_q < 16 (WMMA pads)
    __half * smem_v = smem_k + K_TILE_N * HEAD_SIZE;

    // Multi-warp design: warp 0 owns the WMMA compute + softmax state +
    // partials write. Warps 1-3 only help with staging via decode_stage_k/v
    // (which use the tid as a flat index 0..127). Wastes warp 1-3 compute
    // but recovers the per-warp staging BW that single-warp was missing,
    // which was hurting smaller models where attention is a bigger fraction
    // of decode time.
    const bool is_compute_warp = (wid == 0);

    // These register tiles are reused for each row block; only the 16 rows
    // currently staged in smem_q are live at once.
    tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> Q_tiles[N_INNER];
    tile<16, 16, float, DATA_LAYOUT_I_MAJOR> acc[N_INNER];
    float running_max;
    float running_sum;

    const int n_row_blocks = (total_q + 15) / 16;
    const int row_local = lane % 16;
    for (int rb = 0; rb < n_row_blocks; ++rb) {
        const int row_base  = rb * 16;
        const int row_g     = row_base + row_local;

        // ── Stage Q: this block's 16 rows × HEAD_SIZE. Rows beyond total_q
        // are padded with 0. All 4 warps cooperate on staging for max BW.
        for (int idx = tid; idx < 16 * HEAD_SIZE; idx += 128) {
            const int row_local_q = idx / HEAD_SIZE;
            const int d           = idx % HEAD_SIZE;
            const int row_g_q     = row_base + row_local_q;
            __half val = (__half) 0;
            if (row_g_q < total_q) {
                const int qh = row_g_q / q_len;
                const int qi = row_g_q % q_len;
                const int head_idx = head_base + qh;
                const size_t q_off = ((seq_q_offset + (size_t) qi) * (size_t) n_heads + (size_t) head_idx)
                                     * (size_t) HEAD_SIZE + (size_t) d;
                val = q[q_off];
            }
            smem_q[idx] = val;
        }
        __syncwarp();
        __syncthreads();

        // Load Q tiles into registers — only warp 0.
        if (is_compute_warp) {
            #pragma unroll
            for (int n = 0; n < N_INNER; ++n) {
                const half2 * src = (const half2 *)(smem_q + n * K_INNER);
                load_ldmatrix(Q_tiles[n], src, HEAD_SIZE / 2);
            }
        }

        // Online-softmax state is private to this row block. Each lane owns 8
        // cols of one Q row; its pair (tid ^ 16) owns the other 8 cols.
        running_max = SOFTMAX_MASK_VAL;
        running_sum = 0.0f;

        #pragma unroll
        for (int n = 0; n < N_INNER; ++n) {
            #pragma unroll
            for (int e = 0; e < acc[n].ne; ++e) acc[n].x[e] = 0.0f;
        }

        const int qi_row   = (row_g < total_q) ? (row_g % q_len) : 0;
        const int q_pos_qi = q_pos_first + qi_row;
        const bool row_valid = is_compute_warp && (row_g < total_q);

    // ── Sub-tile loop ──
    for (int sub_start = chunk_start; sub_start < chunk_end; sub_start += K_TILE_N) {
        const int sub_end = min(sub_start + K_TILE_N, chunk_end);
        const int sub_len = sub_end - sub_start;

        // Stage K — all 4 warps cooperate. The helper indexes by tid (0..127).
        decode_stage_k<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_k, k_cache, seq_block_table, sub_start, valid_ctx_max,
            kv_head_idx, n_kv_heads, tid, wid, lane);
        __syncthreads();

        // scores = Q · K^T (16 × 16, fp32 acc) — warp 0 only.
        tile<16, 16, float, DATA_LAYOUT_I_MAJOR> scores;
        if (is_compute_warp) {
            #pragma unroll
            for (int e = 0; e < scores.ne; ++e) scores.x[e] = 0.0f;
            #pragma unroll
            for (int n = 0; n < N_INNER; ++n) {
                tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> K_tile;
                const half2 * src = (const half2 *)(smem_k + n * K_INNER);
                load_ldmatrix(K_tile, src, HEAD_SIZE / 2);
                // RDNA4 WMMA: A operand has K-in-lane / M-in-slot layout; the
                // I-major load + (Q,K) call order computes (Q·K^T)^T, so the
                // kernel reads scores transposed. Swapping operands to (K,Q)
                // computes K·Q^T, which the kernel's I-major read decodes as
                // scores_true[Q-row][K-token]. See probe_v3 for ISA layout.
                mma(scores, K_tile, Q_tiles[n]);
            }
        }

        // DEBUG: trace post-Q·K scores. Fires once per kernel call.
        // Also dump for lane 16 (owns row=0 cols 8..15) to see the half-warp pair.
        const bool debug_here   = debug_origin && (sub_start == chunk_start);
        const bool debug_lane16 = MT_WMMA_DEBUG_DUMP &&
            (kv_head_idx == 0) && (seq_idx == 0) && (chunk_idx == 0) &&
            (wid == 0) && (lane == 16) && (sub_start == chunk_start);
        if (debug_here) {
            printf("[WMMA] scores       lane0  [%g %g %g %g  %g %g %g %g]\n",
                scores.x[0], scores.x[1], scores.x[2], scores.x[3],
                scores.x[4], scores.x[5], scores.x[6], scores.x[7]);
        }
        if (debug_lane16) {
            printf("[WMMA] scores       lane16 [%g %g %g %g  %g %g %g %g]\n",
                scores.x[0], scores.x[1], scores.x[2], scores.x[3],
                scores.x[4], scores.x[5], scores.x[6], scores.x[7]);
        }

        tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> scores_h;
        if (is_compute_warp) {
            // Scale + causal + row/sub_len mask
            #pragma unroll
            for (int l = 0; l < scores.ne; ++l) {
                const int col   = 8 * (lane / 16) + l;
                const int k_pos = sub_start + col;
                const bool visible = row_valid && (col < sub_len) && (k_pos < valid_ctx_max) && (k_pos <= q_pos_qi);
                scores.x[l] = visible ? (scores.x[l] * scale) : SOFTMAX_MASK_VAL;
            }

            // Per-row max
            float local_max = SOFTMAX_MASK_VAL;
            #pragma unroll
            for (int l = 0; l < scores.ne; ++l) local_max = max(local_max, scores.x[l]);
            const float row_max = max(local_max, __shfl_xor_sync(0xFFFFFFFF, local_max, 16, WARP_SIZE));

            const float new_max = max(running_max, row_max);
            float rescale = 1.0f;
            if (running_max > SOFTMAX_MASK_VAL) {
                rescale = __expf(running_max - new_max);
                running_sum *= rescale;
                #pragma unroll
                for (int n = 0; n < N_INNER; ++n) {
                    #pragma unroll
                    for (int e = 0; e < acc[n].ne; ++e) acc[n].x[e] *= rescale;
                }
            }

            float local_sum = 0.0f;
            #pragma unroll
            for (int l = 0; l < scores.ne; ++l) {
                const float e = (scores.x[l] == SOFTMAX_MASK_VAL) ? 0.0f : __expf(scores.x[l] - new_max);
                scores.x[l] = e;
                local_sum  += e;
            }
            const float row_sum = local_sum + __shfl_xor_sync(0xFFFFFFFF, local_sum, 16, WARP_SIZE);
            running_sum += row_sum;
            running_max  = new_max;

            // Pack fp32 scores → half2 for V@logits matmul (precompute before V is staged)
            #pragma unroll
            for (int l = 0; l < scores_h.ne; ++l) {
                scores_h.x[l] = __floats2half2_rn(scores.x[2*l], scores.x[2*l + 1]);
            }
        }

        // Stage V — all 4 warps cooperate.
        decode_stage_v<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_v, v_cache, seq_block_table, sub_start, valid_ctx_max,
            kv_head_idx, n_kv_heads, tid, wid, lane);
        __syncthreads();

        // V@logits — warp 0 only.
        if (is_compute_warp) {
            #pragma unroll
            for (int n = 0; n < N_INNER; ++n) {
                tile<16, 8, half2, DATA_LAYOUT_I_MAJOR> V_tile;
                const half2 * src = (const half2 *)(smem_v + n * K_INNER);
                load_ldmatrix_trans(V_tile, src, HEAD_SIZE / 2);

                // DEBUG: dump V_tile + scores_h + acc inputs for n=0 only.
                // For lane 0 (owns row=0, cols 0..7) AND lane 16 (row=0, cols 8..15).
                if (n == 0 && (debug_here || debug_lane16)) {
                    const char * tag = debug_here ? "lane0 " : "lane16";
                    printf("[WMMA] V_tile[0]    %s  half2[0]=(%g,%g) [1]=(%g,%g) [2]=(%g,%g) [3]=(%g,%g)\n",
                        tag,
                        __low2float(V_tile.x[0]), __high2float(V_tile.x[0]),
                        __low2float(V_tile.x[1]), __high2float(V_tile.x[1]),
                        __low2float(V_tile.x[2]), __high2float(V_tile.x[2]),
                        __low2float(V_tile.x[3]), __high2float(V_tile.x[3]));
                    printf("[WMMA] scores_h     %s  half2[0]=(%g,%g) [1]=(%g,%g) [2]=(%g,%g) [3]=(%g,%g)\n",
                        tag,
                        __low2float(scores_h.x[0]), __high2float(scores_h.x[0]),
                        __low2float(scores_h.x[1]), __high2float(scores_h.x[1]),
                        __low2float(scores_h.x[2]), __high2float(scores_h.x[2]),
                        __low2float(scores_h.x[3]), __high2float(scores_h.x[3]));
                }
                // Same RDNA4 WMMA operand-swap as the Q·K mma above.
                mma(acc[n], V_tile, scores_h);
                if (n == 0 && (debug_here || debug_lane16)) {
                    const char * tag = debug_here ? "lane0 " : "lane16";
                    printf("[WMMA] acc[0] post  %s  [%g %g %g %g  %g %g %g %g]\n",
                        tag,
                        acc[0].x[0], acc[0].x[1], acc[0].x[2], acc[0].x[3],
                        acc[0].x[4], acc[0].x[5], acc[0].x[6], acc[0].x[7]);
                }
            }
        }
        __syncthreads();
    }

    // ── Write partials: each lane has 8 floats × N_INNER per row it owns. ──
    // tile<16,16,float,I_MAJOR>::get_i(l) = tid % 16
    //                          ::get_j(l) = 8 * (tid/16) + l  (l=0..7)
    #pragma unroll
    for (int n = 0; n < N_INNER; ++n) {
        #pragma unroll
        for (int l = 0; l < acc[n].ne; ++l) {
            if (row_valid) {
                const int col_in_tile = 8 * (tid / 16) + l;
                const int d           = n * 16 + col_in_tile;
                const int qh          = row_g / q_len;
                const int qi          = row_g % q_len;
                const int head_idx    = head_base + qh;
                const size_t off = partial_chunk_base_for_head(head_idx)
                                 + (size_t) qi * (HEAD_SIZE + 2)
                                 + (size_t) d;
                partials[off] = acc[n].x[l];
            }
        }
    }
    // Lanes 0..15 own one row of running state each (mask=16 shfl_xor merged
    // them with their pair lane). One write per row.
    if (tid < 16 && row_valid) {
        const int qh = row_g / q_len;
        const int qi = row_g % q_len;
        const int head_idx = head_base + qh;
        const size_t base = partial_chunk_base_for_head(head_idx)
                          + (size_t) qi * (HEAD_SIZE + 2);
        partials[base + HEAD_SIZE]     = running_max;
        partials[base + HEAD_SIZE + 1] = running_sum;
    }
    }
}

// ── Pass 2: reduce chunk partials ──────────────────────────────────────
//
// Online-softmax merge across num_chunks partials, per query position.
// Each block handles one (head, seq); HEAD_SIZE threads write q_len
// output vectors (one per query position in the seq's batch).
template <int HEAD_SIZE>
__global__ void mt_paged_attention_decode_reduce_kernel(
    __half        * __restrict__ out,
    const float   * __restrict__ partials,
    const int32_t * __restrict__ q_lens,
    const int32_t * __restrict__ context_lens,
    int             num_chunks,
    int             chunk_kv,
    int             n_heads,
    int             n_seqs,
    int             max_q_len) {
    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    // Per-seq Q offset for the output write — output mirrors Q layout.
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];
    const int q_len = q_lens[seq_idx];

    // MAD-301A: num_chunks is sized by ALLOCATED ctx (max_blocks_per_seq*block_size).
    // With small CHUNK_KV + a large --ctx-size, that's thousands of chunks while a
    // decode step only fills context_lens[seq] tokens. Bound the reduction to the
    // chunks that actually hold data so cost scales with real depth, not allocated
    // capacity. Pass-1 chunk blocks beyond this range early-return without writing
    // neutral partials, so they must never be read here.
    const int ctx_len      = context_lens[seq_idx];
    const int chunks_full  = (ctx_len + chunk_kv - 1) / chunk_kv;
    const int valid_chunks = chunks_full < num_chunks ? chunks_full : num_chunks;

    // Stride from chunk c → chunk c+1 in the partials buffer.
    const size_t chunk_stride_q     = (size_t) max_q_len * (size_t) (HEAD_SIZE + 2);
    // (head, seq) base.
    const size_t partial_seq_base   = ((size_t) head_idx * n_seqs + seq_idx)
                                    * (size_t) num_chunks * chunk_stride_q;

    for (int qi = 0; qi < q_len; ++qi) {
        const size_t qi_stride = (size_t) qi * (size_t) (HEAD_SIZE + 2);

        // Pass 1: global max across chunks for this query position.
        float global_max = SOFTMAX_MASK_VAL;
        for (int c = 0; c < valid_chunks; ++c) {
            const float m = partials[partial_seq_base + (size_t) c * chunk_stride_q + qi_stride + HEAD_SIZE];
            global_max = max(global_max, m);
        }

        // Pass 2: merge across chunks.
        float global_sum = 0.0f;
        float v_d        = 0.0f;
        for (int c = 0; c < valid_chunks; ++c) {
            const size_t cbase = partial_seq_base + (size_t) c * chunk_stride_q + qi_stride;
            const float  c_max = partials[cbase + HEAD_SIZE];
            if (c_max == SOFTMAX_MASK_VAL) continue;
            const float c_sum = partials[cbase + HEAD_SIZE + 1];
            const float w     = __expf(c_max - global_max);
            global_sum += c_sum * w;
            if (tid < HEAD_SIZE) {
                const float c_v = partials[cbase + (size_t) tid];
                v_d += c_v * w;
            }
        }

        if (tid < HEAD_SIZE) {
            const float inv_sum = 1.0f / (global_sum + 1e-6f);
            const size_t out_off = ((seq_q_offset + (size_t) qi) * (size_t) n_heads + (size_t) head_idx)
                                   * (size_t) HEAD_SIZE
                                 + (size_t) tid;
            out[out_off] = __float2half(v_d * inv_sum);
        }
    }
}

// ── launch ─────────────────────────────────────────────────────────────

template <int HEAD_SIZE, int BLOCK_SIZE, ggml_type CACHE_TYPE>
void launch_paged_attn_decode(
    __half         * out,
    const __half   * q,
    const void     * k_cache,
    const void     * v_cache,
    const int32_t  * block_tables,
    const int32_t  * context_lens,
    const int32_t  * q_lens,
    float          * partials_scratch,
    int             num_seqs,
    int             n_heads,
    int             n_kv_heads,
    int             max_blocks_per_seq,
    int             max_ctx_len,
    int             max_q_len,
    float           scale,
    cudaStream_t    stream) {
    const int num_chunks = paged_attn_decode_num_chunks(max_ctx_len);

    // Pass 1: per-chunk partials.
    // Grid is (n_kv_heads, num_seqs, num_chunks): one block per
    // (kv_head, seq, chunk). The kernel itself iterates over the
    // num_queries_per_kv q_heads in the group + the q_len query tokens,
    // sharing the K/V tiles across them. GQA fanout fix — see kernel
    // comments and MAD-180 follow-up.
    dim3 grid1(n_kv_heads, num_seqs, num_chunks);
    dim3 block1(DECODE_NUM_THREADS);

    // Smem sizing matches the kernel's layout — see the header comment
    // there. smem_q sized for DECODE_MAX_Q to handle the multi-query case.
    const size_t smem_bytes = sizeof(__half) * DECODE_MAX_Q * HEAD_SIZE           // smem_q
                            + sizeof(__half) * DECODE_K_TILE_N * HEAD_SIZE * 2    // smem_k + smem_v
                            + sizeof(float)  * DECODE_MAX_Q * DECODE_K_TILE_N     // smem_logits
                            + sizeof(float)  * DECODE_NUM_WARPS;                  // red_smem

    // WMMA gate: HS in {128, 256}, F16/TURBO4_0/TURBO3_0 cache, WMMA-capable device, env on.
    int dev = 0; cudaGetDevice(&dev);
    const int cc = ggml_cuda_info().devices[dev].cc;
    const bool wmma_path = get_paged_decode_wmma_mode() != 0
                        && amd_wmma_available(cc)
                        && (HEAD_SIZE == 128 || HEAD_SIZE == 256)
                        && (CACHE_TYPE == GGML_TYPE_F16
                         || CACHE_TYPE == GGML_TYPE_TURBO4_0
                         || CACHE_TYPE == GGML_TYPE_TURBO3_0);

    if (wmma_path) {
        // 4-warp block: warp 0 does WMMA compute, warps 1-3 help with K/V
        // staging (cooperative stage_k/v helpers index by tid 0..127).
        dim3 grid_w(n_kv_heads, num_seqs, num_chunks);
        dim3 block_w(128);
        const size_t smem_wmma = sizeof(__half) * 16 * HEAD_SIZE                  // smem_q
                              + sizeof(__half) * DECODE_K_TILE_N * HEAD_SIZE * 2;  // smem_k + smem_v
        mt_paged_attention_decode_kernel_wmma<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>
            <<<grid_w, block_w, smem_wmma, stream>>>(
                partials_scratch, q, k_cache, v_cache,
                block_tables, context_lens, q_lens,
                max_blocks_per_seq, n_kv_heads, n_heads, num_seqs, num_chunks,
                max_q_len, scale);
    } else {
        mt_paged_attention_decode_kernel<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>
            <<<grid1, block1, smem_bytes, stream>>>(
                partials_scratch, q, k_cache, v_cache,
                block_tables, context_lens, q_lens,
                max_blocks_per_seq, n_kv_heads, n_heads, num_seqs, num_chunks,
                max_q_len, scale);
    }

    // Pass 2: reduce — one block per (head, seq); HEAD_SIZE threads
    // collaborate to write q_len output vectors in serial.
    dim3 grid2(n_heads, num_seqs);
    dim3 block2(HEAD_SIZE);
    mt_paged_attention_decode_reduce_kernel<HEAD_SIZE>
        <<<grid2, block2, 0, stream>>>(
            out, partials_scratch, q_lens, context_lens, num_chunks, CHUNK_KV,
            n_heads, num_seqs, max_q_len);
}

// ── explicit instantiations ────────────────────────────────────────────
// We use HS=128 (Qwen3.6 and most modern models) + BLOCK_SIZE=16 + the
// two K/V types we support (F16, TURBO4_0). HS=256 only if/when we hit
// a model that needs it (mirroring the tile kernel's gate).

template void launch_paged_attn_decode<128, 16, GGML_TYPE_F16>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

template void launch_paged_attn_decode<128, 16, GGML_TYPE_TURBO4_0>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

template void launch_paged_attn_decode<128, 16, GGML_TYPE_TURBO3_0>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

template void launch_paged_attn_decode<256, 16, GGML_TYPE_F16>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

template void launch_paged_attn_decode<256, 16, GGML_TYPE_TURBO4_0>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

template void launch_paged_attn_decode<256, 16, GGML_TYPE_TURBO3_0>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);
// HEAD_SIZE=64 — MAD-301C Lever B native head_dim-64 turbo4 flash-decode.
template void launch_paged_attn_decode<64, 16, GGML_TYPE_TURBO4_64>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

} // namespace mt
