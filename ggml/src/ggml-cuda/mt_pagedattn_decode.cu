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

#include <cmath>

namespace mt {

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
static constexpr int CHUNK_KV = 1024;

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
        norm_f = __shfl_sync(0xFFFFFFFF, norm_f, 0);

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
        v += __shfl_xor_sync(0xFFFFFFFF, v, mask);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T decode_warp_reduce_max(T v) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        v = max(v, __shfl_xor_sync(0xFFFFFFFF, v, mask));
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
    float partial = (threadIdx.x < DECODE_NUM_WARPS) ? red_smem[lane] : -INFINITY;
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
static constexpr int DECODE_MAX_Q = 16;

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
        // No visible tokens for any query — write neutral partials so the
        // reducer's pass over all chunks doesn't see uninitialized memory.
        // Loop over (qh, qi) so we cover every head this block owns.
        for (int qh = 0; qh < num_queries_per_kv; ++qh) {
            const int head_idx = head_base + qh;
            const size_t base = partial_chunk_base_for_head(head_idx);
            for (int qi = 0; qi < q_len; ++qi) {
                const size_t off = base + (size_t) qi * (HEAD_SIZE + 2);
                for (int d = tid; d < HEAD_SIZE; d += DECODE_NUM_THREADS) partials[off + d] = 0.0f;
                if (tid == 0) {
                    partials[off + HEAD_SIZE]     = -INFINITY;
                    partials[off + HEAD_SIZE + 1] = 0.0f;
                }
            }
        }
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
        running_max[qhqi] = -INFINITY;
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
                        smem_logits[qhqi * DECODE_K_TILE_N + t] = visible ? (qk * scale) : -INFINITY;
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
                              : -INFINITY;
            const float sub_max = decode_block_reduce_max(local_max, red_smem);

            const float new_max = max(running_max[qhqi], sub_max);
            float rescale = 1.0f;
            if (running_max[qhqi] > -INFINITY) {
                rescale = __expf(running_max[qhqi] - new_max);
                running_sum[qhqi] *= rescale;
                #pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; ++v) v_acc[v][qhqi] *= rescale;
            }

            float local_sum = 0.0f;
            if (tid < DECODE_K_TILE_N) {
                const float lg = smem_logits[qhqi * DECODE_K_TILE_N + tid];
                const float e  = (lg == -INFINITY) ? 0.0f : __expf(lg - new_max);
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
    int             n_heads,
    int             n_seqs,
    int             num_chunks,
    int             max_q_len) {
    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    // Per-seq Q offset for the output write — output mirrors Q layout.
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];
    const int q_len = q_lens[seq_idx];

    // Stride from chunk c → chunk c+1 in the partials buffer.
    const size_t chunk_stride_q     = (size_t) max_q_len * (size_t) (HEAD_SIZE + 2);
    // (head, seq) base.
    const size_t partial_seq_base   = ((size_t) head_idx * n_seqs + seq_idx)
                                    * (size_t) num_chunks * chunk_stride_q;

    for (int qi = 0; qi < q_len; ++qi) {
        const size_t qi_stride = (size_t) qi * (size_t) (HEAD_SIZE + 2);

        // Pass 1: global max across chunks for this query position.
        float global_max = -INFINITY;
        for (int c = 0; c < num_chunks; ++c) {
            const float m = partials[partial_seq_base + (size_t) c * chunk_stride_q + qi_stride + HEAD_SIZE];
            global_max = max(global_max, m);
        }

        // Pass 2: merge across chunks.
        float global_sum = 0.0f;
        float v_d        = 0.0f;
        for (int c = 0; c < num_chunks; ++c) {
            const size_t cbase = partial_seq_base + (size_t) c * chunk_stride_q + qi_stride;
            const float  c_max = partials[cbase + HEAD_SIZE];
            if (c_max == -INFINITY) continue;
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

    mt_paged_attention_decode_kernel<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>
        <<<grid1, block1, smem_bytes, stream>>>(
            partials_scratch, q, k_cache, v_cache,
            block_tables, context_lens, q_lens,
            max_blocks_per_seq, n_kv_heads, n_heads, num_seqs, num_chunks,
            max_q_len, scale);

    // Pass 2: reduce — one block per (head, seq); HEAD_SIZE threads
    // collaborate to write q_len output vectors in serial.
    dim3 grid2(n_heads, num_seqs);
    dim3 block2(HEAD_SIZE);
    mt_paged_attention_decode_reduce_kernel<HEAD_SIZE>
        <<<grid2, block2, 0, stream>>>(
            out, partials_scratch, q_lens, n_heads, num_seqs, num_chunks, max_q_len);
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

template void launch_paged_attn_decode<256, 16, GGML_TYPE_F16>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

template void launch_paged_attn_decode<256, 16, GGML_TYPE_TURBO4_0>(
    __half *, const __half *, const void *, const void *,
    const int32_t *, const int32_t *, const int32_t *,
    float *, int, int, int, int, int, int, float, cudaStream_t);

} // namespace mt
