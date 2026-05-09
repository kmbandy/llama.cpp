// mt_pagedattn — paged attention kernel implementation.
//
// See mt_pagedattn.cuh for layout and threading model docs.
// Design adapted from vLLM (Apache 2.0). Code is independent.

#include "mt_pagedattn.cuh"
#include "turbo-quant.cuh"   // TURBO_CENTROIDS_4BIT, TURBO_WHT_SIGNS{1,2}, turbo_nearest_centroid_4bit

#include <cmath>
#include <cstdlib>

namespace mt {

// ───────────────── env-var: fused vs separate scatter ─────────────────
//
// Default (unset or GGML_PAGED_FUSED=0): mt_scatter_kv_kernel runs first
//   (non-redundant, n_kv_heads blocks per seq), then mt_paged_attention_kernel
//   runs with DO_SCATTER=false. Two launches per attn layer per token.
//   Chosen as default because separate kernels keep MAD-116 quant-aware write
//   paths isolated from the hot attn kernel and tune independently for
//   bandwidth (scatter) vs compute (attn) on speculative-decode batches.
//
// GGML_PAGED_FUSED=1 (toggle): scatter+attn fused in mt_paged_attention_kernel
//   (Phase 1 inside the attn kernel, intra-block __syncthreads). Redundant
//   scatter writes per head block (n_heads / n_kv_heads × idempotent), but
//   only one kernel launch per attn layer per token. Kept as a debug toggle
//   and a fallback if a future workload tilts the perf trade-off.
//
// On Qwen3.6-27B-Q6_K at decode batch=1, fused vs separate is parity within
// run-to-run noise (<1% delta on prefill or decode tps). See bench logs in
// the paged-attn-fusion branch commit message.
//
// Read once on first dispatch; cached for the process lifetime.
static int get_paged_fused_mode() {
    static int mode = -1;
    if (mode < 0) {
        const char * env = std::getenv("GGML_PAGED_FUSED");
        mode = (env != nullptr && env[0] == '1') ? 1 : 0;
        GGML_LOG_INFO("mt_paged_attn: GGML_PAGED_FUSED=%d (%s scatter)\n",
                      mode, mode ? "fused" : "separate");
    }
    return mode;
}

// ───────────────────────── helpers ─────────────────────────

// Warp-level reduce across 32 lanes (HIP wavefront is 64 on some
// GPUs but ggml-cuda's WARP_SIZE is fixed at 32 — works correctly on
// gfx1xxx because __shfl_xor_sync over the active mask).
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, WARP_SIZE);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        v = max(v, __shfl_xor_sync(0xffffffffu, v, mask, WARP_SIZE));
    }
    return v;
}

// Block-level sum: warp reduce, write per-warp partials to smem,
// last warp reduces. One __syncthreads. red_smem must be sized to at
// least NUM_WARPS floats.
template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float v, float * red_smem) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) red_smem[warp] = v;
    __syncthreads();

    float partial = (lane < NUM_WARPS) ? red_smem[lane] : 0.0f;
    if (warp == 0) {
        partial = warp_reduce_sum(partial);
        if (lane == 0) red_smem[0] = partial;
    }
    __syncthreads();
    return red_smem[0];
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float v, float * red_smem) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    v = warp_reduce_max(v);
    if (lane == 0) red_smem[warp] = v;
    __syncthreads();

    float partial = (lane < NUM_WARPS) ? red_smem[lane] : -INFINITY;
    if (warp == 0) {
        partial = warp_reduce_max(partial);
        if (lane == 0) red_smem[0] = partial;
    }
    __syncthreads();
    return red_smem[0];
}

// ──────────────────── paged_cache_ops: per-cache-type layout + dequant ────────────────────
//
// Encapsulates offset math, dequant on read, and quantize on write for the
// paged K/V cache. The kernel and scatter call these statically; actual
// cache layout is specialization-private. Kernel/scatter math stays
// cache-type agnostic — adding a new quant type means adding a specialization
// here, not touching the hot kernel body.
//
// Layouts per (paged_block, kv_head):
//
//   F16 K:       [HEAD_SIZE/K_X, BLOCK_SIZE, K_X]   K_X = 16 / sizeof(__half) = 8
//                (X-stride coalesces F16 reads; original mt:: layout)
//   F16 V:       [HEAD_SIZE, BLOCK_SIZE]
//                (head_dim-major; 16 contiguous tokens per fixed d)
//   Q8_0 K/V:    [BLOCK_SIZE, HEAD_SIZE/QK8_0]      of block_q8_0  (added in MAD-116)
//                (token-first; each token holds HEAD_SIZE/32 q8_0 blocks
//                 contiguously — natural for per-element dequant)
//   Turbo4 K/V:  [BLOCK_SIZE, HEAD_SIZE/QK_TURBO4]  of block_turbo4_0  (added in MAD-116)
//                (same shape as Q8_0; HEAD_SIZE/128 = 2 turbo4 blocks per token)
//
// Primary template — instantiations fail to compile if a needed type
// isn't specialized (lookups force linker error rather than silent fallback).
template <ggml_type T, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops;

// F16 specialization: existing layout, identity load/store via __half.
template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops<GGML_TYPE_F16, HEAD_SIZE, BLOCK_SIZE> {
    static constexpr int K_X = 16 / sizeof(__half);  // 8 — F16 16-byte coalesce stride
    static_assert(HEAD_SIZE % K_X == 0, "HEAD_SIZE must be divisible by K_X for F16 layout");
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");

    __device__ __forceinline__ static float k_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const __half * k = (const __half *) buf;
        const int dim_outer = d / K_X;
        const int dim_inner = d % K_X;
        const size_t off = ((size_t) paged_block * n_kv_heads + kv_head) * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                         + (size_t) dim_outer * BLOCK_SIZE * K_X
                         + (size_t) token_in_block * K_X
                         + (size_t) dim_inner;
        return (float) k[off];
    }

    __device__ __forceinline__ static float v_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const __half * v = (const __half *) buf;
        const size_t off = ((size_t) paged_block * n_kv_heads + kv_head) * HEAD_SIZE * BLOCK_SIZE
                         + (size_t) d * BLOCK_SIZE
                         + (size_t) token_in_block;
        return (float) v[off];
    }

    // Scatter: store one element of K and one of V. Element-granularity store
    // is correct for F16 (no per-block scale). Quant specializations override
    // this to do per-quant-block cooperative quantization.
    __device__ __forceinline__ static void kv_store(
            void * k_buf, void * v_buf,
            int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d,
            float k_val, float v_val) {
        __half * k = (__half *) k_buf;
        __half * v = (__half *) v_buf;
        const int dim_outer = d / K_X;
        const int dim_inner = d % K_X;
        const size_t k_off = ((size_t) paged_block * n_kv_heads + kv_head) * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                           + (size_t) dim_outer * BLOCK_SIZE * K_X
                           + (size_t) token_in_block * K_X
                           + (size_t) dim_inner;
        const size_t v_off = ((size_t) paged_block * n_kv_heads + kv_head) * HEAD_SIZE * BLOCK_SIZE
                           + (size_t) d * BLOCK_SIZE
                           + (size_t) token_in_block;
        k[k_off] = (__half) k_val;
        v[v_off] = (__half) v_val;
    }
};

// Q8_0 specialization: 8-bit symmetric quant, 32-element blocks.
// Layout per (paged_block, kv_head): [BLOCK_SIZE, HEAD_SIZE/QK8_0] of block_q8_0.
// kv_store intentionally omitted — Q8_0 quantization needs cooperative
// per-block scale, so scatter for Q8_0 is handled by a dedicated kernel
// (mt_scatter_kv_q8_0_kernel below) rather than a per-element store. Fused
// scatter inside the attn kernel is therefore not supported for Q8_0;
// dispatch forces separate scatter mode.
template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops<GGML_TYPE_Q8_0, HEAD_SIZE, BLOCK_SIZE> {
    static constexpr int Q_BLOCK = QK8_0;  // 32
    static constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK8_0");

    __device__ __forceinline__ static int64_t element_block_index(
            int paged_block, int kv_head, int n_kv_heads, int token_in_block, int d) {
        // Index into the buffer of block_q8_0:
        //   [paged_block, kv_head, token_in_block, qblock_idx]
        return ((int64_t) paged_block * n_kv_heads + kv_head) * BLOCK_SIZE * N_QBLOCKS_PER_TOKEN
             + (int64_t) token_in_block * N_QBLOCKS_PER_TOKEN
             + (int64_t) (d / Q_BLOCK);
    }

    __device__ __forceinline__ static float k_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const block_q8_0 * blocks = (const block_q8_0 *) buf;
        const int64_t ib = element_block_index(paged_block, kv_head, n_kv_heads, token_in_block, d);
        const int     iqs = d % Q_BLOCK;
        const float   d_scale = (float) blocks[ib].d;
        return (float) blocks[ib].qs[iqs] * d_scale;
    }

    __device__ __forceinline__ static float v_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        // Same layout as K for Q8_0.
        return k_load(buf, paged_block, kv_head, n_kv_heads, token_in_block, d);
    }
};

// Turbo4_0 specialization: 4-bit PolarQuant with WHT rotation, 128-element blocks.
// Layout per (paged_block, kv_head): [BLOCK_SIZE, HEAD_SIZE/QK_TURBO4] of block_turbo4_0.
// Same shape as Q8_0; just different (smaller, more complex) block type.
//
// Like Q8_0, kv_store is omitted because turbo4 quantization needs cooperative
// per-block work (norm reduction + WHT + centroid + nibble-pack). Scatter is
// handled by a dedicated kernel (mt_scatter_kv_turbo4_0_kernel).
template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops<GGML_TYPE_TURBO4_0, HEAD_SIZE, BLOCK_SIZE> {
    static constexpr int Q_BLOCK = QK_TURBO4;  // 128
    static constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO4");

    __device__ __forceinline__ static int64_t element_block_index(
            int paged_block, int kv_head, int n_kv_heads, int token_in_block, int d) {
        return ((int64_t) paged_block * n_kv_heads + kv_head) * BLOCK_SIZE * N_QBLOCKS_PER_TOKEN
             + (int64_t) token_in_block * N_QBLOCKS_PER_TOKEN
             + (int64_t) (d / Q_BLOCK);
    }

    __device__ __forceinline__ static float k_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const block_turbo4_0 * blocks = (const block_turbo4_0 *) buf;
        const int64_t ib  = element_block_index(paged_block, kv_head, n_kv_heads, token_in_block, d);
        const int     iqs = d % Q_BLOCK;
        const float   norm = __half2float(blocks[ib].norm);
        return turbo4_dequant_element(&blocks[ib], iqs, norm);
    }

    __device__ __forceinline__ static float v_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        return k_load(buf, paged_block, kv_head, n_kv_heads, token_in_block, d);
    }
};

// ──────────────────── separate scatter kernel ────────────────────
//
// Used only when GGML_PAGED_FUSED=0. Mirrors the math of Phase 1 in
// mt_paged_attention_kernel but with non-redundant work: one block per
// (kv_head, seq), so each (token, kv_head) pair is written exactly once
// — vs the fused kernel's (n_heads / n_kv_heads)x redundancy. After this
// kernel, the attn kernel is launched with DO_SCATTER=false.
//
// Same-stream submission ordering between this kernel and the subsequent
// attn kernel is guaranteed by CUDA/HIP within a stream (no fence needed
// after PATH A — we now own per-graph metadata tensors and the WAR hazard
// that motivated the host-side stream sync no longer exists).

template <typename scalar_t, ggml_type CACHE_TYPE,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mt_scatter_kv_kernel(
    void           * __restrict__ k_cache,      // type-erased; layout owned by paged_cache_ops<CACHE_TYPE>
    void           * __restrict__ v_cache,
    const scalar_t * __restrict__ k_cur,        // [head_dim, n_kv_heads, n_tokens]
    const scalar_t * __restrict__ v_cur,        // [head_dim, n_kv_heads, n_tokens]
    const int32_t  * __restrict__ slot_mapping, // [n_tokens]
    const int32_t  * __restrict__ q_lens,       // [num_seqs]
    int             n_kv_heads) {
    using ops = paged_cache_ops<CACHE_TYPE, HEAD_SIZE, BLOCK_SIZE>;

    const int kv_head_idx = blockIdx.x;
    const int seq_idx     = blockIdx.y;
    const int tid         = threadIdx.x;

    constexpr int VEC_PER_THREAD = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;

    const int q_len = q_lens[seq_idx];
    const size_t seq_q_offset = 0;  // v1: single-seq batches start at 0
    GGML_UNUSED(seq_idx);            // multi-seq offset accounting is a follow-up

    for (int t = 0; t < q_len; ++t) {
        const int global_token_idx = (int)(seq_q_offset + t);
        const int slot = slot_mapping[global_token_idx];
        if (slot < 0) continue;  // padding token

        const int block_idx     = slot / BLOCK_SIZE;
        const int slot_in_block = slot % BLOCK_SIZE;

        const size_t src_base = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                              + (size_t) kv_head_idx * HEAD_SIZE;

        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                const float k_val = (float) k_cur[src_base + (size_t) d];
                const float v_val = (float) v_cur[src_base + (size_t) d];
                ops::kv_store(k_cache, v_cache,
                              block_idx, kv_head_idx, n_kv_heads, slot_in_block, d,
                              k_val, v_val);
            }
        }
    }
}

// Q8_0 scatter: per-q8_0-block cooperative quantization.
//
// Each warp of 32 threads owns one q8_0 block (32 elements, sharing one
// scale). Warp-level reduction finds max(abs); lane 0 stores the scale,
// each lane writes its quantized int8. Iterates over the token's
// HEAD_SIZE/QK8_0 q8_0 blocks N_WARPS at a time.
template <int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mt_scatter_kv_q8_0_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    const int32_t  * __restrict__ q_lens,
    int             n_kv_heads) {

    constexpr int Q_BLOCK             = QK8_0;
    constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    constexpr int N_WARPS             = NUM_THREADS / WARP_SIZE;
    static_assert(NUM_THREADS % WARP_SIZE == 0, "NUM_THREADS must be multiple of WARP_SIZE");
    static_assert(WARP_SIZE == Q_BLOCK, "this kernel assumes WARP_SIZE == QK8_0 (32)");
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK8_0");

    const int kv_head_idx = blockIdx.x;
    const int seq_idx     = blockIdx.y;
    const int tid         = threadIdx.x;
    const int warp_id     = tid / WARP_SIZE;
    const int lane_id     = tid % WARP_SIZE;

    block_q8_0 * k_blocks = (block_q8_0 *) k_cache;
    block_q8_0 * v_blocks = (block_q8_0 *) v_cache;

    const int q_len = q_lens[seq_idx];
    const size_t seq_q_offset = 0;  // v1: single-seq batches start at 0
    GGML_UNUSED(seq_idx);

    for (int t = 0; t < q_len; ++t) {
        const int global_token_idx = (int)(seq_q_offset + t);
        const int slot = slot_mapping[global_token_idx];
        if (slot < 0) continue;

        const int paged_block   = slot / BLOCK_SIZE;
        const int slot_in_block = slot % BLOCK_SIZE;

        for (int qb_iter = 0; qb_iter < N_QBLOCKS_PER_TOKEN; qb_iter += N_WARPS) {
            const int qb_idx = qb_iter + warp_id;
            if (qb_idx >= N_QBLOCKS_PER_TOKEN) continue;

            const int d = qb_idx * Q_BLOCK + lane_id;

            const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                                 + (size_t) kv_head_idx * HEAD_SIZE
                                 + (size_t) d;
            const float k_val = (float) k_cur[src_off];
            const float v_val = (float) v_cur[src_off];

            // Per-block max(abs) via warp reduction (one warp == one q8_0 block).
            float k_amax = fabsf(k_val);
            float v_amax = fabsf(v_val);
            #pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                k_amax = fmaxf(k_amax, __shfl_xor_sync(0xffffffffu, k_amax, mask, WARP_SIZE));
                v_amax = fmaxf(v_amax, __shfl_xor_sync(0xffffffffu, v_amax, mask, WARP_SIZE));
            }

            const float k_scale     = k_amax / 127.0f;
            const float v_scale     = v_amax / 127.0f;
            const float k_inv_scale = (k_scale > 0.0f) ? (1.0f / k_scale) : 0.0f;
            const float v_inv_scale = (v_scale > 0.0f) ? (1.0f / v_scale) : 0.0f;

            const int k_q = __float2int_rn(k_val * k_inv_scale);
            const int v_q = __float2int_rn(v_val * v_inv_scale);

            // Output index (must match paged_cache_ops<GGML_TYPE_Q8_0>::element_block_index).
            const int64_t block_ib = ((int64_t) paged_block * n_kv_heads + kv_head_idx) * BLOCK_SIZE * N_QBLOCKS_PER_TOKEN
                                   + (int64_t) slot_in_block * N_QBLOCKS_PER_TOKEN
                                   + (int64_t) qb_idx;

            if (lane_id == 0) {
                k_blocks[block_ib].d = __float2half(k_scale);
                v_blocks[block_ib].d = __float2half(v_scale);
            }
            k_blocks[block_ib].qs[lane_id] = (int8_t) k_q;
            v_blocks[block_ib].qs[lane_id] = (int8_t) v_q;
        }
    }
}

// Turbo4_0 scatter: per-128-element-block cooperative quantize.
//
// Each CUDA block handles ONE turbo4 block of work (one (token, kv_head, qb_idx,
// kv_select) tuple). 128 threads, one per element of the turbo4 block. Pattern
// adapted from set-rows.cu's CUDA-side turbo4 quantization:
//   1. Load element into smem
//   2. Block-wide L2 norm (warp reduce + cross-warp accumulator)
//   3. Normalize, apply Randomized Hadamard Transform (signs1 → 7-stage
//      butterfly → signs2 / sqrt(128))
//   4. Nearest-centroid lookup (16 centroids), nibble-pack qs[]
//   5. Block-wide reconstruction norm; lane 0 writes corrected norm
template <int HEAD_SIZE, int BLOCK_SIZE>
__launch_bounds__(QK_TURBO4)
__global__ void mt_scatter_kv_turbo4_0_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    int             n_kv_heads) {

    constexpr int Q_BLOCK             = QK_TURBO4;          // 128
    constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    constexpr int N_WARPS             = Q_BLOCK / WARP_SIZE;  // 4
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO4");
    static_assert(Q_BLOCK == 128, "this kernel assumes QK_TURBO4 == 128");

    const int j                = threadIdx.x;  // 0..127, element index within turbo4 block
    const int global_token_idx = blockIdx.x;
    const int y_idx            = blockIdx.y;
    const int kv_select        = blockIdx.z;   // 0 = K, 1 = V
    const int kv_head_idx      = y_idx / N_QBLOCKS_PER_TOKEN;
    const int qb_idx           = y_idx % N_QBLOCKS_PER_TOKEN;

    const int slot = slot_mapping[global_token_idx];
    if (slot < 0) return;  // padding token

    const int paged_block   = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    const int    d = qb_idx * Q_BLOCK + j;
    const __half * src = (kv_select == 0) ? k_cur : v_cur;
    const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                         + (size_t) kv_head_idx * HEAD_SIZE
                         + (size_t) d;

    // Output block pointer (matches paged_cache_ops<TURBO4_0>::element_block_index)
    void * dst_buf = (kv_select == 0) ? k_cache : v_cache;
    const int64_t block_ib = ((int64_t) paged_block * n_kv_heads + kv_head_idx) * BLOCK_SIZE * N_QBLOCKS_PER_TOKEN
                           + (int64_t) slot_in_block * N_QBLOCKS_PER_TOKEN
                           + (int64_t) qb_idx;
    block_turbo4_0 * blk = (block_turbo4_0 *) dst_buf + block_ib;

    // ---- Step 1: Load into smem (explicit __half2float for HIP safety) ----
    __shared__ float x[Q_BLOCK];
    x[j] = __half2float(src[src_off]);
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    __shared__ float warp_accum[N_WARPS];
    {
        float v_sq = x[j] * x[j];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v_sq += __shfl_xor_sync(0xffffffffu, v_sq, offset);
        }
        if (j % WARP_SIZE == 0) warp_accum[j / WARP_SIZE] = v_sq;
    }
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < N_WARPS; ++w) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: (intentionally NO Randomized Hadamard Transform) ----
    //
    // Canonical turbo4 (ggml-cuda set-rows + fattn-vec) applies RHT
    // (signs1 → Hadamard butterfly → signs2/sqrt(d)) during quantization,
    // then dequant returns centroid*norm — which is RHT(K), not K. For
    // attention with an unrotated Q, that pair only approximates <K, Q>
    // via the Johnson-Lindenstrauss random-projection bound (~6% RMS
    // error at d=256). Acceptable for fattn-vec where it has been
    // empirically tuned.
    //
    // Our paged path stores and reads K end-to-end within mt::, so we can
    // skip RHT: just centroid-quant K_normalized in place. The dequant
    // (turbo4_dequant_element) returns centroid*norm ≈ K_norm * ||K|| = K.
    // Dot with Q gives <K, Q> directly — no JL approximation. Storage
    // savings are unchanged (same 4-bit + per-block scale, ~3.76× vs F16).
    //
    // Quality note: turbo4 centroids were tuned for N(0, 1/d) — the
    // distribution of unit-norm vectors in high-d space, which matches
    // K_norm regardless of RHT. So skipping RHT does NOT degrade the
    // quant fit for Gaussian-like K (verified on Qwen3.6 9/9 math).

    // ---- Step 5: Quantize element j ----
    const float   rv  = x[j];
    const uint8_t idx = turbo_nearest_centroid_4bit(rv);

    // ---- Step 6: Pack qs (nibble packed, warp-cooperative) ----
    const int      lane            = j % WARP_SIZE;
    const uint8_t  my_nibble       = idx & 0xF;
    const uint8_t  partner_nibble  = __shfl_sync(0xffffffffu, my_nibble, lane ^ 1);
    if ((j & 1) == 0) {
        blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
    }

    // ---- Step 7: Reconstruction norm (parallel) ----
    {
        const float c = TURBO_CENTROIDS_4BIT[idx];
        float rc = c * c;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            rc += __shfl_xor_sync(0xffffffffu, rc, offset);
        }
        if (j % WARP_SIZE == 0) warp_accum[j / WARP_SIZE] = rc;
    }
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < N_WARPS; ++w) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    // ---- Step 8: Lane 0 writes per-block scalars ----
    if (j == 0) {
        blk->norm  = __float2half(corrected_norm);
        blk->rnorm = __float2half(0.0f);  // reserved/unused in 4-bit mode
    }
}

template <ggml_type CACHE_TYPE, typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
static void launch_scatter_kv(
    void           * k_cache,
    void           * v_cache,
    const scalar_t * k_cur,
    const scalar_t * v_cur,
    const int32_t  * slot_mapping,
    const int32_t  * q_lens,
    int             num_seqs,
    int             num_tokens_total,    // sum(q_lens); used by per-token-grid kernels (turbo4)
    int             n_kv_heads,
    cudaStream_t    stream) {

    if constexpr (CACHE_TYPE == GGML_TYPE_F16) {
        constexpr int NUM_THREADS = 128;
        dim3 grid(n_kv_heads, num_seqs);
        dim3 block(NUM_THREADS);
        GGML_UNUSED(num_tokens_total);
        mt_scatter_kv_kernel<scalar_t, CACHE_TYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
            <<<grid, block, 0, stream>>>(
                k_cache, v_cache, k_cur, v_cur, slot_mapping, q_lens, n_kv_heads);
    } else if constexpr (CACHE_TYPE == GGML_TYPE_Q8_0) {
        constexpr int NUM_THREADS = 128;
        dim3 grid(n_kv_heads, num_seqs);
        dim3 block(NUM_THREADS);
        GGML_UNUSED(num_tokens_total);
        mt_scatter_kv_q8_0_kernel<HEAD_SIZE, BLOCK_SIZE, NUM_THREADS>
            <<<grid, block, 0, stream>>>(
                k_cache, v_cache, k_cur, v_cur, slot_mapping, q_lens, n_kv_heads);
    } else if constexpr (CACHE_TYPE == GGML_TYPE_TURBO4_0) {
        // 1 CUDA block per (token, (kv_head, qb_idx), K-or-V); 128 threads per block.
        constexpr int Q_BLOCK             = QK_TURBO4;
        constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
        dim3 grid(num_tokens_total, n_kv_heads * N_QBLOCKS_PER_TOKEN, 2);
        dim3 block(Q_BLOCK);
        GGML_UNUSED(q_lens);
        GGML_UNUSED(num_seqs);
        mt_scatter_kv_turbo4_0_kernel<HEAD_SIZE, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(
                k_cache, v_cache, k_cur, v_cur, slot_mapping, n_kv_heads);
    }
    // Fall-through for unsupported types is intentional — dispatch in
    // ggml_cuda_op_paged_attn_mt switches only over types that have a
    // launch path, so this branch is unreachable at runtime.
}

// ──────────────────── kernel ────────────────────
//
// Threading: each thread block handles one (head, seq) pair, running
// over all query tokens of that seq's batch slice. NUM_THREADS threads
// cooperate per token using warp shuffles for QK and per-row softmax
// + V@logits accumulation. Designed for HEAD_SIZE divisible by
// NUM_THREADS for the V-accumulator stride; we assert this in dispatch.
//
// Per-query iteration (chunked online-softmax, MAD-115):
//   1. Each thread loads HEAD_SIZE/NUM_THREADS elements of Q into
//      registers (q_reg).
//   2. Outer loop over the seq's valid_ctx in CHUNK_SIZE-token chunks.
//      Per-chunk maintain running (max, sum, acc[]) state in registers:
//        A. QK over the chunk → chunk-local smem logits[i], track chunk_max
//        B. new_max = max(running_max, chunk_max); rescale running state by
//           exp(running_max - new_max)
//        C. exp(qk - new_max), accumulate chunk_sum into running_sum,
//           store back into logits[i]
//        D. V @ logits accumulation into per-thread acc[VEC_PER_THREAD]
//   3. Final: out[d] = acc[v] / running_sum, write to global memory.
//
// Per-block smem footprint: (NUM_WARPS + CHUNK_SIZE) * sizeof(float),
// independent of valid_ctx — supports per-attn-call ctx well beyond
// the previous full-pass kernel's ~16k LDS-overflow ceiling.
//
// Memory access pattern:
//   K cache laid out as [num_blocks, n_kv_heads, HEAD_SIZE/x, BLOCK_SIZE, x]
//   For a given (block, kv_head) and our thread reading element
//   d in [0, HEAD_SIZE), the offset within the block is:
//     (d / x) * BLOCK_SIZE * x  +  token_in_block * x  +  (d % x)
//   With x = 16/sizeof(scalar_t), neighboring threads (different d)
//   read adjacent x-elements — coalesced.
//
//   V cache: [num_blocks, n_kv_heads, HEAD_SIZE, BLOCK_SIZE]
//   For (block, kv_head, head_dim_d), all BLOCK_SIZE tokens are
//   contiguous. Accumulating this thread's d means reading V at
//   (kv_head_idx * HEAD_SIZE + d) * BLOCK_SIZE + token_in_block.
//
// GQA: n_heads can be > n_kv_heads. kv_head = head_idx / (n_heads /
// n_kv_heads). n_heads % n_kv_heads == 0 enforced at dispatch.

template <typename scalar_t, ggml_type CACHE_TYPE,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, bool DO_SCATTER = true>
__global__ void mt_paged_attention_kernel(
    scalar_t       * __restrict__ out,
    const scalar_t * __restrict__ q,
    void           * __restrict__ k_cache,    // type-erased; layout owned by paged_cache_ops<CACHE_TYPE>
    void           * __restrict__ v_cache,
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
    const scalar_t * __restrict__ k_cur,        // [head_dim, n_kv_heads, n_tokens]
    const scalar_t * __restrict__ v_cur,        // [head_dim, n_kv_heads, n_tokens]
    const int32_t  * __restrict__ slot_mapping, // [n_tokens]
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    float           scale) {
    using ops = paged_cache_ops<CACHE_TYPE, HEAD_SIZE, BLOCK_SIZE>;

    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    constexpr int NUM_WARPS         = NUM_THREADS / WARP_SIZE;
    constexpr int VEC_PER_THREAD    = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(NUM_THREADS % WARP_SIZE == 0, "NUM_THREADS must be multiple of WARP_SIZE");

    const int kv_head_idx       = head_idx / (n_heads / n_kv_heads);
    const int q_len             = q_lens[seq_idx];
    const int ctx_len_after_q   = context_lens[seq_idx];   // total tokens in seq's context AFTER this batch's Q is applied
    const int * seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // ── Phase 1: scatter K_cur/V_cur into the K/V cache (fused) ──
    //
    // MAD-114: the previous design had scatter and attn as SEPARATE kernels,
    // relying on same-stream submission ordering to make scatter's writes
    // visible to attn's reads. On HIP/RDNA (gfx1201, ROCm 7.2.x) that
    // ordering isn't reliably enforced (cf. ROCm/hip#3882, #3887), so attn
    // would read stale cache. Fusing the two phases into one kernel,
    // separated by an intra-block __syncthreads(), eliminates the dependency
    // on inter-kernel ordering — __syncthreads is a hardware barrier that
    // always works.
    //
    // GQA: with n_heads > n_kv_heads, multiple (head_idx, seq_idx) blocks
    // share the same kv_head_idx. EACH such block scatters its kv_head's
    // slots redundantly. Writes are idempotent (same source values to same
    // destination addresses), so the redundancy is correctness-preserving.
    // No cross-block synchronization is needed: each block reads in phase 2
    // only the slots it itself wrote in phase 1.
    //
    // When DO_SCATTER=false, scatter has already been done by a separate
    // kernel launched on the same stream — Phase 1 here is elided.
    if constexpr (DO_SCATTER) {
        const size_t seq_q_offset = 0;  // v1: single-seq batches start at 0

        for (int t = 0; t < q_len; ++t) {
            const int global_token_idx = (int)(seq_q_offset + t);
            const int slot = slot_mapping[global_token_idx];
            if (slot < 0) continue;  // padding token

            const int block_idx     = slot / BLOCK_SIZE;
            const int slot_in_block = slot % BLOCK_SIZE;

            // K_cur / V_cur: ne[0]=head_dim, ne[1]=n_kv_heads, ne[2]=n_tokens
            const size_t src_base = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                                  + (size_t) kv_head_idx * HEAD_SIZE;

            #pragma unroll
            for (int v = 0; v < VEC_PER_THREAD; ++v) {
                const int d = tid + v * NUM_THREADS;
                if (d < HEAD_SIZE) {
                    const float k_val = (float) k_cur[src_base + (size_t) d];
                    const float v_val = (float) v_cur[src_base + (size_t) d];
                    ops::kv_store(k_cache, v_cache,
                                  block_idx, kv_head_idx, n_kv_heads, slot_in_block, d,
                                  k_val, v_val);
                }
            }
        }
        __syncthreads();  // makes this block's scatter writes visible to its own attn reads
    }
    // ── Phase 2: attention math (uses the cache slots this block just wrote) ──

    // Shared memory layout:
    //   [0 .. NUM_WARPS)        — red_smem (reduction scratch)
    //   [NUM_WARPS .. ...)      — logits buffer (per query-token, max ctx)
    extern __shared__ float smem[];
    float * red_smem = smem;
    float * logits   = smem + NUM_WARPS;

    // Per-token iteration.
    for (int qi = 0; qi < q_len; ++qi) {
        // q_pos is the absolute position of this query token in the
        // sequence (0-indexed). Causal mask: only attend to k tokens
        // with kj <= q_pos.
        const int q_pos = (ctx_len_after_q - q_len) + qi;
        const int valid_ctx = q_pos + 1;  // tokens [0, valid_ctx) are visible

        // Load Q slice into registers, scale-applied.
        // Q layout matches ggml's natural [head_dim, n_heads, n_tokens]
        // (head_dim fastest, n_tokens slowest in memory). For seq 0,
        // head H, query token Q, dim D:
        //   offset = (Q * n_heads + H) * HEAD_SIZE + D
        // (Multi-seq batches concat sequentially in the n_tokens axis;
        // qi here is the local-to-this-seq index, so we add the seq's
        // start offset = sum of preceding q_lens. v1 single-seq path
        // has seq_q_offset == 0.)
        scalar_t q_reg[VEC_PER_THREAD];
        const size_t seq_q_offset = 0;  // v1: single-seq batches start at 0
#pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                const size_t q_off = ((seq_q_offset + (size_t) qi) * n_heads + head_idx) * HEAD_SIZE + d;
                q_reg[v] = q[q_off];
            } else {
                q_reg[v] = scalar_t(0);
            }
        }
        GGML_UNUSED(seq_idx);  // multi-seq offset accounting is a follow-up

        // ── Chunked online-softmax (FlashAttention-style) ──
        //
        // Process K/V in chunks of CHUNK_SIZE tokens, maintaining running
        // (max, sum, acc[]) state across chunks via the standard recurrence:
        //
        //   new_max  = max(running_max, chunk_max)
        //   rescale  = exp(running_max - new_max)   // 1 on first chunk
        //   acc_new  = acc_old * rescale + Σ(exp(qk - new_max) * v)
        //   sum_new  = sum_old * rescale + Σ exp(qk - new_max)
        //   running_max = new_max
        //
        // Final: out = acc / sum.
        //
        // Per-chunk smem footprint = CHUNK_SIZE * sizeof(float), independent
        // of valid_ctx — supports per-attn-call ctx > 16k (was the LDS-
        // overflow limit on the previous full-pass kernel).
        constexpr int CHUNK_SIZE = 256;

        float running_max = -INFINITY;
        float running_sum = 0.0f;
        float acc[VEC_PER_THREAD];
#pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) acc[v] = 0.0f;

        for (int chunk_start = 0; chunk_start < valid_ctx; chunk_start += CHUNK_SIZE) {
            const int chunk_end = (chunk_start + CHUNK_SIZE < valid_ctx)
                                  ? (chunk_start + CHUNK_SIZE)
                                  : valid_ctx;
            const int chunk_len = chunk_end - chunk_start;

            // ── Phase A: QK over chunk → logits[i], track chunk_max ──
            //
            // Per-token cooperation: every thread contributes its slice of d
            // to partial_qk, then block_reduce_sum aggregates across the
            // block. tid==0 stores qk into the chunk-local logits[]. The
            // block_reduce_max at the end of this phase provides the
            // visibility barrier before Phase C reads logits[].
            float chunk_max = -INFINITY;
            for (int i = 0; i < chunk_len; ++i) {
                const int token         = chunk_start + i;
                const int logical_block = token / BLOCK_SIZE;
                const int tok_in_block  = token % BLOCK_SIZE;
                const int physical      = seq_block_table[logical_block];

                float partial_qk = 0.0f;
                if (physical != kInvalidBlockTableEntry) {
#pragma unroll
                    for (int v = 0; v < VEC_PER_THREAD; ++v) {
                        const int d = tid + v * NUM_THREADS;
                        if (d < HEAD_SIZE) {
                            const float k_val = ops::k_load(k_cache, physical, kv_head_idx, n_kv_heads, tok_in_block, d);
                            partial_qk += (float) q_reg[v] * k_val;
                        }
                    }
                }
                // All threads see the same physical → ternary is uniform;
                // either all call block_reduce_sum or all skip it.
                const float qk = (physical == kInvalidBlockTableEntry)
                    ? -INFINITY
                    : block_reduce_sum<NUM_WARPS>(partial_qk, red_smem) * scale;

                if (tid == 0) logits[i] = qk;
                chunk_max = max(chunk_max, qk);
            }
            chunk_max = block_reduce_max<NUM_WARPS>(chunk_max, red_smem);

            // ── Phase B: rescale running state to new_max ──
            const float new_max = max(running_max, chunk_max);
            if (running_max != -INFINITY) {
                const float rescale = __expf(running_max - new_max);
                running_sum *= rescale;
#pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; ++v) acc[v] *= rescale;
            }

            // ── Phase C: exp(qk - new_max), accumulate chunk_sum ──
            float chunk_sum = 0.0f;
            for (int i = tid; i < chunk_len; i += NUM_THREADS) {
                const float e = __expf(logits[i] - new_max);
                logits[i] = e;
                chunk_sum += e;
            }
            chunk_sum = block_reduce_sum<NUM_WARPS>(chunk_sum, red_smem);
            running_sum += chunk_sum;

            // ── Phase D: V @ logits accumulation for this chunk ──
            for (int i = 0; i < chunk_len; ++i) {
                const int token         = chunk_start + i;
                const int logical_block = token / BLOCK_SIZE;
                const int tok_in_block  = token % BLOCK_SIZE;
                const int physical      = seq_block_table[logical_block];
                if (physical == kInvalidBlockTableEntry) continue;
                const float w = logits[i];

#pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; ++v) {
                    const int d = tid + v * NUM_THREADS;
                    if (d < HEAD_SIZE) {
                        const float v_val = ops::v_load(v_cache, physical, kv_head_idx, n_kv_heads, tok_in_block, d);
                        acc[v] += w * v_val;
                    }
                }
            }

            running_max = new_max;
            __syncthreads();  // logits[] reuse safe for next chunk's Phase A
        }

        // ── Final: normalize and write output slice ──
        // Output mirrors Q's layout: ggml [head_dim, n_heads, n_tokens]
        // (same shape as q since the constructor copies q->ne).
        const float inv_sum = 1.0f / (running_sum + 1e-6f);
#pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                const size_t out_off = ((seq_q_offset + (size_t) qi) * n_heads + head_idx) * HEAD_SIZE + d;
                out[out_off] = (scalar_t) (acc[v] * inv_sum);
            }
        }

        __syncthreads();  // ensure smem reuse safe between qi iterations
    }
}

// ──────────────────── dispatch ────────────────────

// op_params layout (set by graph builder when emitting GGML_OP_PAGED_ATTN_MT):
//   [0]: float scale
//   [1]: int32_t block_size
//   [2]: int32_t max_blocks_per_seq
//   [3]: int32_t n_kv_heads
//
// src tensors:
//   src[0] = Q     [head_size, n_heads, sum(q_lens), 1]   — packed across seqs
//   src[1] = K cache [paged layout]
//   src[2] = V cache [paged layout]
//   src[3] = block_tables [max_blocks_per_seq, num_seqs]
//   src[4] = context_lens [num_seqs]
//   src[5] = q_lens       [num_seqs]
// dst:
//   out [head_size, n_heads, sum(q_lens), 1]
//
// For the v1 single-batch case sum(q_lens) collapses to q_len * num_seqs
// when all seqs in the batch have the same q_len (typical decode batch).

template <typename scalar_t, ggml_type CACHE_TYPE,
          int HEAD_SIZE, int BLOCK_SIZE, bool DO_SCATTER>
static void launch_paged_attn(
    scalar_t       * out,
    const scalar_t * q,
    void           * k_cache,
    void           * v_cache,
    const int32_t  * block_tables,
    const int32_t  * context_lens,
    const int32_t  * q_lens,
    const scalar_t * k_cur,
    const scalar_t * v_cur,
    const int32_t  * slot_mapping,
    int             num_seqs,
    int             n_heads,
    int             n_kv_heads,
    int             max_blocks_per_seq,
    int             max_ctx_len,
    float           scale,
    cudaStream_t    stream) {
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_WARPS   = NUM_THREADS / WARP_SIZE;
    constexpr int CHUNK_SIZE  = 256;  // mirror the kernel's CHUNK_SIZE

    dim3 grid(n_heads, num_seqs);
    dim3 block(NUM_THREADS);

    // smem: NUM_WARPS reduction floats + CHUNK_SIZE chunk-local logits floats.
    // MAD-115: rewrite to chunked online-softmax — smem is now bounded by
    // CHUNK_SIZE, not max_ctx_len. Per-block footprint ≈ 1 KiB, fits any
    // GPU's LDS regardless of context length.
    const size_t smem_bytes = (NUM_WARPS + CHUNK_SIZE) * sizeof(float);
    GGML_UNUSED(max_ctx_len);  // retained in signature for callers; chunking makes it irrelevant for smem sizing

    mt_paged_attention_kernel<scalar_t, CACHE_TYPE, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, /*PARTITION_SIZE=*/0, DO_SCATTER>
        <<<grid, block, smem_bytes, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, context_lens, q_lens,
            k_cur, v_cur, slot_mapping,
            max_blocks_per_seq, n_kv_heads, n_heads, scale);
}

void ggml_cuda_op_paged_attn_mt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_cache       = dst->src[1];
    const ggml_tensor * v_cache       = dst->src[2];
    const ggml_tensor * block_tables  = dst->src[3];
    const ggml_tensor * context_lens  = dst->src[4];
    const ggml_tensor * q_lens        = dst->src[5];
    const ggml_tensor * k_cur         = dst->src[6];
    const ggml_tensor * v_cur         = dst->src[7];
    const ggml_tensor * slot_mapping  = dst->src[8];

    const float * op_params_f = (const float *)(dst->op_params);
    const float   scale       = op_params_f[0];
    const int32_t block_size  = ((const int32_t *)(op_params_f + 1))[0];
    const int32_t max_bps     = ((const int32_t *)(op_params_f + 2))[0];
    const int32_t n_kv_heads  = ((const int32_t *)(op_params_f + 3))[0];

    const int head_size = q->ne[0];
    const int n_heads   = q->ne[1];
    const int num_seqs  = block_tables->ne[1];

    GGML_ASSERT(n_heads % n_kv_heads == 0 && "n_heads must be divisible by n_kv_heads");
    GGML_ASSERT(q->type == GGML_TYPE_F16 && "PagedAttn supports F16 Q only");
    GGML_ASSERT(k_cache->type == v_cache->type && "PagedAttn requires K and V cache to share type");
    GGML_ASSERT(k_cur && k_cur->type == GGML_TYPE_F16);
    GGML_ASSERT(v_cur && v_cur->type == GGML_TYPE_F16);
    GGML_ASSERT(slot_mapping && slot_mapping->type == GGML_TYPE_I32);
    GGML_ASSERT(k_cur->ne[1] == n_kv_heads);
    GGML_ASSERT(v_cur->ne[1] == n_kv_heads);
    GGML_ASSERT(slot_mapping->ne[0] == k_cur->ne[2]);

    // For smem sizing we need the longest context in this batch.
    // Cheap upper bound: max_blocks_per_seq * block_size.
    const int max_ctx_len = max_bps * block_size;

    cudaStream_t stream = ctx.stream();
    const bool do_fused = (get_paged_fused_mode() != 0);

    // Dispatch on (head_size, block_size, cache_type). Add quant cases by
    // adding a paged_cache_ops<TYPE> specialization above and a switch arm
    // here.
    auto run = [&](auto head_size_const, auto block_size_const) {
        constexpr int HS = decltype(head_size_const)::value;
        constexpr int BS = decltype(block_size_const)::value;

        auto run_typed = [&](auto cache_type_const) {
            constexpr ggml_type CT = decltype(cache_type_const)::value;

            // Cache-type / head-size compatibility: TURBO4_0 needs
            // HEAD_SIZE divisible by QK_TURBO4 (128). For smaller heads
            // (e.g. HS=64), abort at runtime — we don't even instantiate
            // the kernel templates for invalid combinations.
            if constexpr (CT == GGML_TYPE_TURBO4_0 && HS % QK_TURBO4 != 0) {
                GGML_ABORT("mt_paged_attn: HEAD_SIZE=%d too small for TURBO4_0 (requires multiple of QK_TURBO4=%d)",
                           HS, (int) QK_TURBO4);
            } else {

            // Fused scatter requires per-element kv_store, which only the F16
            // path provides. Quant types use a dedicated cooperative scatter
            // kernel; force separate mode for them.
            const bool fused = do_fused && (CT == GGML_TYPE_F16);

            if (!fused) {
                // Experiment path: separate scatter kernel + attn-only kernel.
                // Same-stream submission ordering is sufficient now that PATH A
                // removed the metadata-tensor WAR hazard.
                launch_scatter_kv<CT, __half, HS, BS>(
                    k_cache->data,
                    v_cache->data,
                    (const __half *) k_cur->data,
                    (const __half *) v_cur->data,
                    (const int32_t *) slot_mapping->data,
                    (const int32_t *) q_lens->data,
                    num_seqs, (int) k_cur->ne[2], n_kv_heads, stream);
                launch_paged_attn<__half, CT, HS, BS, /*DO_SCATTER=*/false>(
                    (__half *) dst->data,
                    (const __half *) q->data,
                    k_cache->data,
                    v_cache->data,
                    (const int32_t *) block_tables->data,
                    (const int32_t *) context_lens->data,
                    (const int32_t *) q_lens->data,
                    (const __half *) k_cur->data,
                    (const __half *) v_cur->data,
                    (const int32_t *) slot_mapping->data,
                    num_seqs, n_heads, n_kv_heads, max_bps, max_ctx_len,
                    scale, stream);
            } else if constexpr (CT == GGML_TYPE_F16) {
                launch_paged_attn<__half, CT, HS, BS, /*DO_SCATTER=*/true>(
                    (__half *) dst->data,
                    (const __half *) q->data,
                    k_cache->data,
                    v_cache->data,
                    (const int32_t *) block_tables->data,
                    (const int32_t *) context_lens->data,
                    (const int32_t *) q_lens->data,
                    (const __half *) k_cur->data,
                    (const __half *) v_cur->data,
                    (const int32_t *) slot_mapping->data,
                    num_seqs, n_heads, n_kv_heads, max_bps, max_ctx_len,
                    scale, stream);
            }
            }  // closes else of (CT == TURBO4_0 && HS % QK_TURBO4 != 0)
        };

        switch (k_cache->type) {
            case GGML_TYPE_F16:
                run_typed(std::integral_constant<ggml_type, GGML_TYPE_F16>{});
                break;
            case GGML_TYPE_Q8_0:
                run_typed(std::integral_constant<ggml_type, GGML_TYPE_Q8_0>{});
                break;
            case GGML_TYPE_TURBO4_0:
                run_typed(std::integral_constant<ggml_type, GGML_TYPE_TURBO4_0>{});
                break;
            default:
                GGML_ABORT("mt_paged_attn: unsupported cache type %s — add a paged_cache_ops specialization",
                           ggml_type_name(k_cache->type));
        }
    };

    // Most common cases first; fall through to a runtime error for
    // unsupported (head_size, block_size) so we discover them loudly.
    if (head_size == 128 && block_size == 16) {
        run(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 64 && block_size == 16) {
        run(std::integral_constant<int, 64>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 256 && block_size == 16) {
        run(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 128 && block_size == 32) {
        run(std::integral_constant<int, 128>{}, std::integral_constant<int, 32>{});
    } else {
        GGML_ABORT("mt_paged_attn: unsupported (head_size=%d, block_size=%d) — add a template instantiation",
                   head_size, block_size);
    }
}


}  // namespace mt
