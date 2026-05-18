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
    // We pre-cast to __half * but the layout helpers still want a void *.
    // Splitting K vs V here lets us select the right ops::*_load.
    using ops = paged_cache_ops<GGML_TYPE_F16, HEAD_SIZE, BLOCK_SIZE>;
    const void * src = (const void *) src_cache_as_half;
    #pragma unroll
    for (int idx = tid; idx < DECODE_K_TILE_N * HEAD_SIZE; idx += DECODE_NUM_THREADS) {
        const int row   = idx / HEAD_SIZE;
        const int col   = idx % HEAD_SIZE;
        const int token = tile_start + row;
        float val = 0.0f;
        if (token < valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical != kInvalidBlockTableEntry) {
                val = is_v
                    ? ops::v_load(src, physical, kv_head_idx, n_kv_heads, tok_in_block, col)
                    : ops::k_load(src, physical, kv_head_idx, n_kv_heads, tok_in_block, col);
            }
        }
        smem_dst[idx] = __float2half(val);
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

// Compile-time max q_len handled per block. q_len > MAX_Q falls to the
// scalar path (the dispatch gate checks this). 8 covers MTP spec-decode
// draft-verification batches (MAD-174 / MAD-176); larger q_len batches
// are prefill territory and go to the tile kernel.
static constexpr int DECODE_MAX_Q = 8;

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
    const int head_idx  = blockIdx.x;
    const int seq_idx   = blockIdx.y;
    const int chunk_idx = blockIdx.z;
    const int tid       = threadIdx.x;
    const int wid       = tid / WARP_SIZE;
    const int lane      = tid % WARP_SIZE;

    const int q_len           = q_lens[seq_idx];      // 1..DECODE_MAX_Q (dispatch-enforced)
    const int ctx_len_after_q = context_lens[seq_idx];
    const int kv_head_idx     = head_idx / (n_heads / n_kv_heads);
    const int * seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // Per-block check — defensive against host/device q_len skew. Dispatch
    // promises q_len <= DECODE_MAX_Q; clamp in case a future op_param flow
    // sneaks something through.
    if (q_len > DECODE_MAX_Q) return;

    // Per-seq Q offset (sum of preceding q_lens).
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];

    // Per-query absolute position. qi=0 is the FIRST new query token in
    // the batch, qi=q_len-1 is the trailing one. Each qi has its own
    // causal mask: it sees tokens [0, q_pos_first + qi].
    const int q_pos_first    = ctx_len_after_q - q_len;
    const int valid_ctx_max  = ctx_len_after_q;   // highest mask any qi can reach

    const int chunk_start = chunk_idx * CHUNK_KV;

    // Partial-output base for this (head, seq, chunk). Inner stride is
    // max_q_len so the buffer has a single uniform stride across seqs;
    // unused slots (qi >= q_len for this seq) are simply left untouched
    // and never read by the reducer (it loops to this seq's q_len).
    const size_t partial_chunk_base =
        ((((size_t) head_idx * n_seqs + seq_idx) * num_chunks) + (size_t) chunk_idx)
        * (size_t) max_q_len * (size_t) (HEAD_SIZE + 2);

    if (chunk_start >= valid_ctx_max) {
        // No visible tokens for any query — write neutral partials so the
        // reducer's pass over all chunks doesn't see uninitialized memory.
        for (int qi = 0; qi < q_len; ++qi) {
            const size_t off = partial_chunk_base + (size_t) qi * (HEAD_SIZE + 2);
            for (int d = tid; d < HEAD_SIZE; d += DECODE_NUM_THREADS) partials[off + d] = 0.0f;
            if (tid == 0) {
                partials[off + HEAD_SIZE]     = -INFINITY;
                partials[off + HEAD_SIZE + 1] = 0.0f;
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

    // ── Stage all q_len query vectors ──
    for (int idx = tid; idx < q_len * HEAD_SIZE; idx += DECODE_NUM_THREADS) {
        const int qi = idx / HEAD_SIZE;
        const int d  = idx % HEAD_SIZE;
        const size_t q_off = ((seq_q_offset + (size_t) qi) * (size_t) n_heads + (size_t) head_idx)
                             * (size_t) HEAD_SIZE + (size_t) d;
        smem_q[qi * HEAD_SIZE + d] = q[q_off];
    }
    __syncthreads();

    // Per-thread online-softmax state per query. HEAD_SIZE=128 and
    // DECODE_NUM_THREADS=128 → VEC_PER_THREAD=1; v_acc stays small even at
    // MAX_Q=8 (8 floats/thread).
    constexpr int VEC_PER_THREAD = (HEAD_SIZE + DECODE_NUM_THREADS - 1) / DECODE_NUM_THREADS;
    float v_acc[VEC_PER_THREAD][DECODE_MAX_Q];
    float running_max[DECODE_MAX_Q];
    float running_sum[DECODE_MAX_Q];
    #pragma unroll
    for (int qi = 0; qi < DECODE_MAX_Q; ++qi) {
        running_max[qi] = -INFINITY;
        running_sum[qi] = 0.0f;
        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) v_acc[v][qi] = 0.0f;
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

        // ── QK: 1 warp per (token, all queries). For each token slot,
        // one warp computes q_len dot products against the same K row. ──
        #pragma unroll
        for (int t_base = 0; t_base < DECODE_K_TILE_N; t_base += DECODE_NUM_WARPS) {
            const int t     = t_base + wid;
            const int token = sub_start + t;
            if (t < DECODE_K_TILE_N) {
                for (int qi = 0; qi < q_len; ++qi) {
                    float qk = 0.0f;
                    if (t < sub_len && token < valid_ctx_max) {
                        #pragma unroll
                        for (int d = lane; d < HEAD_SIZE; d += WARP_SIZE) {
                            const float qv = __half2float(smem_q[qi * HEAD_SIZE + d]);
                            const float kv = __half2float(smem_k[t * HEAD_SIZE + d]);
                            qk += qv * kv;
                        }
                        qk = decode_warp_reduce_sum(qk);
                    }
                    if (lane == 0) {
                        const int q_pos_qi = q_pos_first + qi;
                        const bool visible = (t < sub_len) && (token <= q_pos_qi);
                        smem_logits[qi * DECODE_K_TILE_N + t] = visible ? (qk * scale) : -INFINITY;
                    }
                }
            }
        }
        __syncthreads();

        // ── Per-query softmax update + V matmul prep ──
        // We need sub_max[qi] and sub_sum[qi] per query. Reduce via the
        // block primitives — DECODE_K_TILE_N (16) ≤ NUM_THREADS so the
        // first 16 threads' logit values cover the sub-chunk.
        for (int qi = 0; qi < q_len; ++qi) {
            float local_max = (tid < DECODE_K_TILE_N)
                              ? smem_logits[qi * DECODE_K_TILE_N + tid]
                              : -INFINITY;
            const float sub_max = decode_block_reduce_max(local_max, red_smem);

            const float new_max = max(running_max[qi], sub_max);
            float rescale = 1.0f;
            if (running_max[qi] > -INFINITY) {
                rescale = __expf(running_max[qi] - new_max);
                running_sum[qi] *= rescale;
                #pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; ++v) v_acc[v][qi] *= rescale;
            }

            float local_sum = 0.0f;
            if (tid < DECODE_K_TILE_N) {
                const float lg = smem_logits[qi * DECODE_K_TILE_N + tid];
                const float e  = (lg == -INFINITY) ? 0.0f : __expf(lg - new_max);
                smem_logits[qi * DECODE_K_TILE_N + tid] = e;
                local_sum = e;
            }
            const float sub_sum = decode_block_reduce_sum(local_sum, red_smem);
            running_sum[qi] += sub_sum;
            running_max[qi]  = new_max;
        }

        // ── Stage V (shared by all queries) ──
        decode_stage_v<HEAD_SIZE, BLOCK_SIZE, CACHE_TYPE>(
            smem_v, v_cache, seq_block_table,
            sub_start, valid_ctx_max,
            kv_head_idx, n_kv_heads, tid, wid, lane);
        __syncthreads();

        // ── V matmul: v_acc[qi, d] += Σ_t softmax[qi, t] · V[t, d] ──
        // Pre-load V column for this thread's d into registers so the
        // q_len inner loop hits registers, not LDS.
        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * DECODE_NUM_THREADS;
            if (d < HEAD_SIZE) {
                float v_col[DECODE_K_TILE_N];
                #pragma unroll
                for (int t = 0; t < DECODE_K_TILE_N; ++t) {
                    v_col[t] = __half2float(smem_v[t * HEAD_SIZE + d]);
                }
                for (int qi = 0; qi < q_len; ++qi) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < DECODE_K_TILE_N; ++t) {
                        acc += smem_logits[qi * DECODE_K_TILE_N + t] * v_col[t];
                    }
                    v_acc[v][qi] += acc;
                }
            }
        }
        __syncthreads();  // before next sub's stage_k reuses smem_k
    }

    // ── Write per-(chunk, query) partials ──
    for (int qi = 0; qi < q_len; ++qi) {
        const size_t off = partial_chunk_base + (size_t) qi * (HEAD_SIZE + 2);
        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * DECODE_NUM_THREADS;
            if (d < HEAD_SIZE) partials[off + d] = v_acc[v][qi];
        }
        if (tid == 0) {
            partials[off + HEAD_SIZE]     = running_max[qi];
            partials[off + HEAD_SIZE + 1] = running_sum[qi];
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
    dim3 grid1(n_heads, num_seqs, num_chunks);
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
