// mt_pagedattn_aiter.cu — AITER-backed paged attention path. See the header
// for design notes.
//
// Only compiled when ggml-hip is built with -DGGML_HIP_AITER=ON. The
// non-AITER builds get the inline no-op stubs from the header.

#include "mt_pagedattn_aiter.cuh"

#ifdef GGML_HIP_AITER

#include "common.cuh"
#include "turbo-quant.cuh"   // MAD-199: TURBO_CENTROIDS_{3,4}BIT, turbo_nearest_centroid_{3,4}bit

#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <mutex>

// The runtime AITER wrapper. Lives in aiter-integration's static library
// (libaiter_triton_aot.a), linked into ggml-hip when GGML_HIP_AITER=ON.
// Header propagated via aiter_triton_aot's PUBLIC target_include_directories.
#include "mt_aiter_unified_attn.h"
#include "mt_turbo_fp8_lut_registry.h"  // MAD-214: per-(layer, kv-dir) centroid LUT lookup
#include "turbo_fp8_hadamard.cuh"      // MAD-227: fp16 FWHT for Q pre-rotation

#include <cstring>
#include <sys/stat.h>  // MAD-214 Option F: mkdir for dump dir
#include <vector>

namespace mt {

// Helper: parse layer index from a tensor name like "cache_k_l<N>" or
// "cache_v_l<N>". Returns -1 if the pattern doesn't match. Used to bind
// the right per-layer LUT to each AITER paged-attn invocation.
static int parse_layer_from_kv_cache_name(const char * name) {
    if (!name) return -1;
    const char * p = std::strstr(name, "_l");
    if (!p) return -1;
    p += 2;
    int n = 0;
    bool any = false;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); ++p; any = true; }
    return any ? n : -1;
}

// ─────────────────────────────────────────────────────────────────────────
// AITER-format scatter kernel (F16 cache only for v1)
//
// Layout: [num_blocks, block_size, n_kv_heads, head_size], no interleaving.
// Equivalent to vLLM/AITER's `unified_attention` K/V cache shape — keeps the
// scatter dead-simple compared to ggml's vectorized K layout.
// ─────────────────────────────────────────────────────────────────────────
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mt_scatter_kv_kernel_aiter(
    scalar_t       * __restrict__ k_cache,
    scalar_t       * __restrict__ v_cache,
    const scalar_t * __restrict__ k_cur,        // [head_dim, n_kv_heads, n_tokens]
    const scalar_t * __restrict__ v_cur,        // [head_dim, n_kv_heads, n_tokens]
    const int32_t  * __restrict__ slot_mapping, // [n_tokens]
    const int32_t  * __restrict__ q_lens,       // [num_seqs]
    int             n_kv_heads) {

    const int kv_head_idx = blockIdx.x;
    const int seq_idx     = blockIdx.y;
    const int tid         = threadIdx.x;

    constexpr int VEC_PER_THREAD = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;

    const int q_len = q_lens[seq_idx];
    // Per-seq offset into the packed k_cur/v_cur tensor — seq tokens are
    // concatenated in seq_id order on the ne[2] axis.
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];

    for (int t = 0; t < q_len; ++t) {
        const int global_token_idx = (int)(seq_q_offset + t);
        const int slot = slot_mapping[global_token_idx];
        if (slot < 0) continue;  // padding token

        const int block_idx     = slot / BLOCK_SIZE;
        const int slot_in_block = slot % BLOCK_SIZE;

        const size_t src_base = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                              + (size_t) kv_head_idx     * HEAD_SIZE;
        const size_t dst_base = (size_t) block_idx     * BLOCK_SIZE * n_kv_heads * HEAD_SIZE
                              + (size_t) slot_in_block * n_kv_heads * HEAD_SIZE
                              + (size_t) kv_head_idx   * HEAD_SIZE;

        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                k_cache[dst_base + d] = k_cur[src_base + d];
                v_cache[dst_base + d] = v_cur[src_base + d];
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AITER-layout TURBO3 scatter (MAD-199).
//
// Block geometry: 128 threads per block (one block per (token, kv_head,
// qb_idx)). Pipeline mirrors mt_scatter_kv_turbo3_0_kernel in
// mt_pagedattn.cu — same load, parallel L2 norm, normalize, nearest-3-bit
// centroid, pack qs (32-byte) + signs (16-byte), reconstruction norm. The
// ONLY difference is the destination block-index math: AITER layout is
// `[num_paged_blocks, BLOCK_SIZE, n_kv_heads, N_QBLOCKS_PER_TOKEN]` while
// paged-tile uses `[num_paged_blocks, n_kv_heads, BLOCK_SIZE, ...]`. Same
// content per element, different memory ordering — the Triton load
// helpers in unified_attention.py expect THIS layout.
//
// As with paged-tile, RHT is intentionally skipped: TURBO_CENTROIDS_3BIT
// is Lloyd-Max for N(0, 1/d) which matches normalized K vectors directly.
// ─────────────────────────────────────────────────────────────────────────
template <int HEAD_SIZE, int BLOCK_SIZE>
__launch_bounds__(QK_TURBO3)
__global__ void mt_scatter_kv_turbo3_aiter_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    int             n_kv_heads) {

    constexpr int Q_BLOCK             = QK_TURBO3;            // 128
    constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;  // 1 at HS=128
    constexpr int N_WARPS             = Q_BLOCK / WARP_SIZE;  // 4
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO3");
    static_assert(Q_BLOCK == 128, "this kernel assumes QK_TURBO3 == 128");

    const int j                = threadIdx.x;
    const int global_token_idx = blockIdx.x;
    const int y_idx            = blockIdx.y;
    const int kv_select        = blockIdx.z;   // 0 = K, 1 = V
    const int kv_head_idx      = y_idx / N_QBLOCKS_PER_TOKEN;
    const int qb_idx           = y_idx % N_QBLOCKS_PER_TOKEN;

    const int slot = slot_mapping[global_token_idx];
    if (slot < 0) return;

    const int paged_block   = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    const int    d   = qb_idx * Q_BLOCK + j;
    const __half * src = (kv_select == 0) ? k_cur : v_cur;
    const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                         + (size_t) kv_head_idx     * HEAD_SIZE
                         + (size_t) d;

    // AITER-layout block index: [paged_block, slot_in_block, kv_head, qb_idx]
    void * dst_buf = (kv_select == 0) ? k_cache : v_cache;
    const int64_t block_ib =
          ((int64_t) paged_block * BLOCK_SIZE * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) slot_in_block * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) kv_head_idx * N_QBLOCKS_PER_TOKEN)
        + (int64_t) qb_idx;
    block_turbo3_0 * blk = (block_turbo3_0 *) dst_buf + block_ib;

    __shared__ float x[Q_BLOCK];
    x[j] = __half2float(src[src_off]);
    __syncthreads();

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

    x[j] *= inv_norm;
    __syncthreads();

    const float   rv  = x[j];
    const uint8_t idx = turbo_nearest_centroid_3bit(rv);  // 0..7

    // Pack qs: 4 elements per byte, low 2 bits each (per-warp).
    {
        const int     lane    = j % WARP_SIZE;
        const int     warp_id = j / WARP_SIZE;
        const uint8_t my2     = idx & 0x3;
        uint8_t byte_val      = my2 << ((lane & 3) * 2);
        byte_val |= __shfl_xor_sync(0xffffffffu, byte_val, 1);
        byte_val |= __shfl_xor_sync(0xffffffffu, byte_val, 2);
        if ((lane & 3) == 0) {
            blk->qs[warp_id * (WARP_SIZE / 4) + lane / 4] = byte_val;
        }
    }

    // Pack signs: 8 elements per byte, high 1 bit each (per-warp).
    {
        const int     lane    = j % WARP_SIZE;
        const int     warp_id = j / WARP_SIZE;
        const uint8_t my1     = (idx >> 2) & 0x1;
        uint8_t bits          = my1 << (lane & 7);
        bits |= __shfl_xor_sync(0xffffffffu, bits, 1);
        bits |= __shfl_xor_sync(0xffffffffu, bits, 2);
        bits |= __shfl_xor_sync(0xffffffffu, bits, 4);
        if ((lane & 7) == 0) {
            blk->signs[warp_id * (WARP_SIZE / 8) + lane / 8] = bits;
        }
    }

    // Reconstruction norm: ||centroid·norm|| should equal ||K||; correct norm
    // if drift > 1e-10.
    {
        const float c = TURBO_CENTROIDS_3BIT[idx];
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

    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AITER-layout TURBO4 scatter (MAD-199). Same shape as turbo3 above; only
// the index packing (4-bit nibble) and block struct (block_turbo4_0) differ.
// ─────────────────────────────────────────────────────────────────────────
template <int HEAD_SIZE, int BLOCK_SIZE>
__launch_bounds__(QK_TURBO4)
__global__ void mt_scatter_kv_turbo4_aiter_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    int             n_kv_heads) {

    constexpr int Q_BLOCK             = QK_TURBO4;            // 128
    constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    constexpr int N_WARPS             = Q_BLOCK / WARP_SIZE;  // 4
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO4");
    static_assert(Q_BLOCK == 128, "this kernel assumes QK_TURBO4 == 128");

    const int j                = threadIdx.x;
    const int global_token_idx = blockIdx.x;
    const int y_idx            = blockIdx.y;
    const int kv_select        = blockIdx.z;
    const int kv_head_idx      = y_idx / N_QBLOCKS_PER_TOKEN;
    const int qb_idx           = y_idx % N_QBLOCKS_PER_TOKEN;

    const int slot = slot_mapping[global_token_idx];
    if (slot < 0) return;

    const int paged_block   = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    const int    d   = qb_idx * Q_BLOCK + j;
    const __half * src = (kv_select == 0) ? k_cur : v_cur;
    const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                         + (size_t) kv_head_idx     * HEAD_SIZE
                         + (size_t) d;

    void * dst_buf = (kv_select == 0) ? k_cache : v_cache;
    const int64_t block_ib =
          ((int64_t) paged_block * BLOCK_SIZE * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) slot_in_block * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) kv_head_idx * N_QBLOCKS_PER_TOKEN)
        + (int64_t) qb_idx;
    block_turbo4_0 * blk = (block_turbo4_0 *) dst_buf + block_ib;

    __shared__ float x[Q_BLOCK];
    x[j] = __half2float(src[src_off]);
    __syncthreads();

    __shared__ float warp_accum[N_WARPS];
    {
        float v_sq = x[j] * x[j];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v_sq += __shfl_xor_sync(0xffffffffu, v_sq, offset, WARP_SIZE);
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

    x[j] *= inv_norm;
    __syncthreads();

    const float   rv  = x[j];
    const uint8_t idx = turbo_nearest_centroid_4bit(rv);  // 0..15

    // Pack qs: 2 nibbles per byte (warp-cooperative).
    {
        const int      lane            = j % WARP_SIZE;
        const uint8_t  my_nibble       = idx & 0xF;
        const uint8_t  partner_nibble  = __shfl_sync(0xffffffffu, my_nibble, lane ^ 1, WARP_SIZE);
        if ((j & 1) == 0) {
            blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
        }
    }

    {
        const float c = TURBO_CENTROIDS_4BIT[idx];
        float rc = c * c;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            rc += __shfl_xor_sync(0xffffffffu, rc, offset, WARP_SIZE);
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

    if (j == 0) {
        blk->norm  = __float2half(corrected_norm);
        blk->rnorm = __float2half(0.0f);  // reserved/unused in 4-bit mode
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MAD-214 Phase 1G-G: AITER-layout turbo-FP8 BS=256 scatter.
//
// Same paged-cache layout as turbo4 above, but the block is 162 bytes
// (2-byte fp16 scale | 128-byte 4-bit indices | 32-byte sign bits) and
// the centroid LUT comes in as a runtime device pointer (one for K, one
// for V) instead of the compile-time TURBO_CENTROIDS_4BIT table.
//
// Grid: (num_tokens, n_kv_heads, 2_for_K_and_V). One (token, kv_head)
// row = one 162-byte block (HEAD_SIZE=256 = Q_BLOCK=256). 256 threads
// per block, one element per thread.
// ─────────────────────────────────────────────────────────────────────────

// Device-side E4M3 → fp32 (same mapping as set-rows.cu and the CPU packer).
static __device__ __forceinline__ float fp8_e4m3_to_fp32_aiter(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e    = (b >> 3) & 0xF;
    int m    = b & 0x7;
    float v  = (e == 0) ? (1.0f / 64.0f) * (m / 8.0f)
                        : __builtin_amdgcn_ldexp(1.0f + m / 8.0f, e - 7);
    return sign ? -v : v;
}

template <int HEAD_SIZE, int BLOCK_SIZE, bool APPLY_HADAMARD>
__launch_bounds__(256)
__global__ void mt_scatter_kv_turbo4_fp8_aiter_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    const uint8_t  * __restrict__ centroids_k,
    const uint8_t  * __restrict__ centroids_v,
    int             n_kv_heads) {

    static_assert(HEAD_SIZE == 256, "turbo4_fp8 AITER scatter requires HEAD_SIZE=256");
    constexpr int N_CENT = 16;
    constexpr int BYTES_PER_BLOCK = 162;

    const int j                = threadIdx.x;     // 0..255 element idx
    const int global_token_idx = blockIdx.x;
    const int kv_head_idx      = blockIdx.y;
    const int kv_select        = blockIdx.z;      // 0 = K, 1 = V

    const int slot = slot_mapping[global_token_idx];
    if (slot < 0) return;

    const int paged_block   = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    const __half * src = (kv_select == 0) ? k_cur : v_cur;
    const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                         + (size_t) kv_head_idx     * HEAD_SIZE
                         + (size_t) j;

    uint8_t * dst_buf = (uint8_t *) ((kv_select == 0) ? k_cache : v_cache);
    const int64_t block_byte_off =
          (int64_t) paged_block * BLOCK_SIZE * n_kv_heads * BYTES_PER_BLOCK
        + (int64_t) slot_in_block * n_kv_heads * BYTES_PER_BLOCK
        + (int64_t) kv_head_idx * BYTES_PER_BLOCK;
    uint8_t * blk = dst_buf + block_byte_off;

    const uint8_t * lut_bytes = (kv_select == 0) ? centroids_k : centroids_v;

    // ── Stage 1: load element + decode LUT into shared mem ──
    __shared__ float x[256];
    __shared__ float lut_f[N_CENT];

    x[j] = __half2float(src[src_off]);
    if (j < N_CENT) lut_f[j] = fp8_e4m3_to_fp32_aiter(lut_bytes[j]);
    __syncthreads();

    // ── MAD-227 Stage 1.5: optional in-place FWHT on K only ──
    // Identity QK^T = (QH)·(KH)^T holds → rotating K at scatter requires
    // rotating Q at attention. V is NOT rotated (see turbo_fp8_hadamard.cuh
    // for the K-only-vs-K+V tradeoff rationale). At HEAD_SIZE=256 the
    // butterfly is 8 stages; one __syncthreads pair per stage.
    if constexpr (APPLY_HADAMARD) {
        if (kv_select == 0) {  // K only
            constexpr int D = HEAD_SIZE;  // 256
            #pragma unroll
            for (int stage = 0; (1 << stage) < D; ++stage) {
                const int stride  = 1 << stage;
                const int partner = j ^ stride;
                const float a = x[j];        // our value
                const float b = x[partner];  // partner's value
                __syncthreads();
                if ((j & stride) == 0) {
                    x[j] = a + b;            // lower partner: a + b
                } else {
                    x[j] = b - a;            // upper partner: lower - upper = b - a
                }
                __syncthreads();
            }
            x[j] *= (1.0f / 16.0f);          // 1/sqrt(256)
            __syncthreads();
        }
    }

    // ── Stage 2: per-block max-abs scale ──
    float v_abs = fabsf(x[j]);
    for (int off = 16; off > 0; off >>= 1) {
        v_abs = fmaxf(v_abs, __shfl_xor_sync(0xffffffffffffffffull, v_abs, off, WARP_SIZE));
    }
    __shared__ float warp_max[8];
    if ((j % 32) == 0) warp_max[j / 32] = v_abs;
    __syncthreads();
    __shared__ float blk_max;
    if (j == 0) {
        float m = warp_max[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) m = fmaxf(m, warp_max[w]);
        blk_max = m;
    }
    __syncthreads();

    // ── Stage 3: cast scale → fp16, broadcast ──
    const float scale_f = blk_max;
    const __half scale_h = __float2half(scale_f);
    const float scale_eff = __half2float(scale_h);
    const float inv_scale = (scale_eff > 0.0f) ? (1.0f / scale_eff) : 0.0f;

    // ── Stage 4: quantize ──
    const float v   = x[j];
    const int   sgn = (v < 0.0f) ? 1 : 0;
    const float mag = fabsf(v) * inv_scale;

    int   best_idx = 0;
    float best_err = fabsf(mag - lut_f[0]);
    #pragma unroll
    for (int k = 1; k < N_CENT; ++k) {
        float e = fabsf(mag - lut_f[k]);
        if (e < best_err) { best_idx = k; best_err = e; }
    }

    // ── Stage 5: cooperative pack ──
    if (j == 0) {
        blk[0] = ((const uint8_t *) &scale_h)[0];
        blk[1] = ((const uint8_t *) &scale_h)[1];
    }

    const uint8_t my_nib      = (uint8_t)(best_idx & 0xF);
    const uint8_t partner_nib = (uint8_t) __shfl_xor_sync(0xffffffffffffffffull, (int) my_nib, 1, WARP_SIZE);
    if ((j & 1) == 0) {
        blk[2 + j / 2] = my_nib | (uint8_t)(partner_nib << 4);
    }

    const uint64_t sign_mask = __ballot_sync(0xffffffffffffffffull, sgn);
    if ((j & 7) == 0) {
        const int byte_idx = j / 8;
        const int hw_lane_base = (threadIdx.x % warpSize) & ~(WARP_SIZE - 1);
        const int bit_off = hw_lane_base + ((j % WARP_SIZE) & ~7);
        blk[130 + byte_idx] = (uint8_t)((sign_mask >> bit_off) & 0xFF);
    }
}

// Build the AITER `query_start_len` cu-seqlens tensor [num_seqs+1] on device
// from q_lens [num_seqs]. Tiny — one thread block.
__global__ void mt_build_cu_seqlens_kernel(
    int32_t       * __restrict__ cu_seqlens,
    const int32_t * __restrict__ q_lens,
    int             num_seqs) {
    if (threadIdx.x != 0) return;  // 1 thread; trivial sequential prefix sum
    int32_t acc = 0;
    cu_seqlens[0] = 0;
    for (int s = 0; s < num_seqs; ++s) {
        acc += q_lens[s];
        cu_seqlens[s + 1] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Runtime gate
// ─────────────────────────────────────────────────────────────────────────
bool aiter_backend_enabled() {
    static std::atomic<int> cached{-1};  // -1 = unset, 0 = off, 1 = on
    int v = cached.load(std::memory_order_relaxed);
    if (v < 0) {
        const char * env = std::getenv("MAD_USE_AITER");
        v = (env && *env && env[0] != '0') ? 1 : 0;
        cached.store(v, std::memory_order_relaxed);
        if (v) {
            std::fprintf(stderr, "mt_pagedattn: AITER backend ENABLED (MAD_USE_AITER=%s)\n", env);
        }
    }
    return v == 1;
}

// ─────────────────────────────────────────────────────────────────────────
// One-time-allocated 1.0f device buffer for q/k/v/out descale (unquantized
// path passes ones).
// ─────────────────────────────────────────────────────────────────────────
static float * descale_ones_device() {
    static float * ptr = nullptr;
    static std::atomic<bool> ready{false};
    if (ready.load(std::memory_order_acquire)) return ptr;
    // Init under a coarse lock — rare, on first AITER call.
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (ready.load(std::memory_order_relaxed)) return ptr;
    cudaMalloc((void**) &ptr, sizeof(float));
    const float one = 1.0f;
    cudaMemcpy(ptr, &one, sizeof(float), cudaMemcpyHostToDevice);
    ready.store(true, std::memory_order_release);
    return ptr;
}

// ─────────────────────────────────────────────────────────────────────────
// AITER dispatch entry
// ─────────────────────────────────────────────────────────────────────────
void ggml_cuda_op_paged_attn_mt_aiter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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

    const int head_size      = (int) q->ne[0];
    const int n_heads        = (int) q->ne[1];
    const int num_seqs       = (int) block_tables->ne[1];
    const int num_q_tokens   = (int) k_cur->ne[2];

    // Shape gate — the wrapper builds a Triton signature from these at first
    // call and the runtime registry compiles a matching kernel.
    GGML_ASSERT(q->type == GGML_TYPE_F16 && "AITER backend requires F16 Q");
    GGML_ASSERT(k_cache->type == v_cache->type && "AITER backend requires K and V cache to be the same type");
    GGML_ASSERT(n_heads % n_kv_heads == 0 && "n_heads must be divisible by n_kv_heads");
    GGML_ASSERT(head_size > 0 && (head_size & (head_size - 1)) == 0 && "AITER backend requires power-of-2 head_size");

    // MAD-199: route the cache ggml_type to the wrapper's mt_aiter_cache_type
    // enum. Triton-side dispatch on this value selects the right K/V load +
    // dequant path inside kernel_unified_attention_3d.
    int cache_type;
    switch (k_cache->type) {
        case GGML_TYPE_F16:               cache_type = MT_AITER_CACHE_F16;            break;
        case GGML_TYPE_TURBO3_0:          cache_type = MT_AITER_CACHE_TURBO3;         break;
        case GGML_TYPE_TURBO4_0:          cache_type = MT_AITER_CACHE_TURBO4;         break;
        case GGML_TYPE_TURBO4_FP8_BS256:  cache_type = MT_AITER_CACHE_TURBO4_FP8;     break;
        default:
            GGML_ABORT("AITER backend: unsupported KV cache type %d", (int) k_cache->type);
    }

    // MAD-214 Phase 1G-G: for turbo-FP8, look up per-(layer, kv-dir) centroid
    // LUTs from the runtime registry. Layer index is parsed from the cache
    // tensor name (set by llama_kv_cache as "cache_k_l<N>" / "cache_v_l<N>").
    const uint8_t * d_centroids_k = nullptr;
    const uint8_t * d_centroids_v = nullptr;
    if (cache_type == MT_AITER_CACHE_TURBO4_FP8) {
        const int il = parse_layer_from_kv_cache_name(k_cache->name);
        GGML_ASSERT(il >= 0 && "turbo4_fp8: failed to parse layer index from k_cache tensor name");
        d_centroids_k = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_K);
        d_centroids_v = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_V);
        GGML_ASSERT(d_centroids_k && d_centroids_v && "turbo4_fp8: centroid LUT lookup returned null");
    }

    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = head_size;
    shape.num_q_heads  = n_heads;
    shape.num_kv_heads = n_kv_heads;
    shape.block_size   = block_size;
    shape.cache_type   = cache_type;

    cudaStream_t stream = ctx.stream();

    // ── MAD-214 Option F: calibration dump hook ──
    // When MT_TURBO_FP8_DUMP_DIR is set and the cache type is turbo-FP8,
    // copy the fp16 K_cur and V_cur tensors to disk for offline Lloyd-Max
    // fitting (scripts/calibration/fit_centroids_from_dump.py). Layer index
    // is parsed from the k_cache tensor name. The actual scatter still runs
    // (using the fallback LUT), but only the dumps are consumed for fitting.
    if (cache_type == MT_AITER_CACHE_TURBO4_FP8) {
        const char * dump_dir = std::getenv("MT_TURBO_FP8_DUMP_DIR");
        if (dump_dir && *dump_dir) {
            const int il = parse_layer_from_kv_cache_name(k_cache->name);
            if (il >= 0) {
                const size_t bytes_per_tensor = (size_t) num_q_tokens * n_kv_heads * head_size * sizeof(__half);
                std::vector<__half> host_buf(num_q_tokens * n_kv_heads * head_size);
                static std::mutex dump_mu;
                std::lock_guard<std::mutex> g(dump_mu);
                // Ensure dump dir exists.
                ::mkdir(dump_dir, 0755);
                // K_cur dump
                hipMemcpy(host_buf.data(), k_cur->data, bytes_per_tensor, hipMemcpyDeviceToHost);
                char path[512];
                std::snprintf(path, sizeof(path), "%s/l%d_k.fp16", dump_dir, il);
                if (FILE *f = std::fopen(path, "ab")) {
                    std::fwrite(host_buf.data(), 1, bytes_per_tensor, f);
                    std::fclose(f);
                }
                // V_cur dump
                hipMemcpy(host_buf.data(), v_cur->data, bytes_per_tensor, hipMemcpyDeviceToHost);
                std::snprintf(path, sizeof(path), "%s/l%d_v.fp16", dump_dir, il);
                if (FILE *f = std::fopen(path, "ab")) {
                    std::fwrite(host_buf.data(), 1, bytes_per_tensor, f);
                    std::fclose(f);
                }
            }
        }
    }

    // ── 1. Scatter K_cur/V_cur into AITER-layout cache ──
    // Dispatch on (cache_type, head_size, block_size) at compile time so the
    // kernel can unroll its inner loops. Add instantiations here when a new
    // model needs an unsupported shape.
    if (cache_type == MT_AITER_CACHE_F16) {
        constexpr int NUM_THREADS = 128;
        dim3 grid(n_kv_heads, num_seqs);
        dim3 block(NUM_THREADS);
        auto launch_aiter_scatter = [&](auto HS_const, auto BS_const) {
            constexpr int HS = decltype(HS_const)::value;
            constexpr int BS = decltype(BS_const)::value;
            mt_scatter_kv_kernel_aiter<__half, HS, BS, NUM_THREADS>
                <<<grid, block, 0, stream>>>(
                    (__half*) k_cache->data,
                    (__half*) v_cache->data,
                    (const __half*) k_cur->data,
                    (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data,
                    (const int32_t*) q_lens->data,
                    n_kv_heads);
        };
        if (head_size == 128 && block_size == 16) {
            launch_aiter_scatter(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
        } else if (head_size == 64 && block_size == 16) {
            launch_aiter_scatter(std::integral_constant<int, 64>{}, std::integral_constant<int, 16>{});
        } else if (head_size == 256 && block_size == 16) {
            launch_aiter_scatter(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
        } else {
            GGML_ABORT("AITER F16 scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
        }
    } else {
        // MAD-199: TURBO3 / TURBO4 scatter. 128 threads per block (= QK_TURBO).
        // Grid: (num_tokens, n_kv_heads * N_QBLOCKS_PER_TOKEN, 2_for_K_and_V).
        // The kernel templates derive N_QBLOCKS_PER_TOKEN from HEAD_SIZE.
        const int n_qblocks_per_token = head_size / 128;  // QK_TURBO3 == QK_TURBO4 == 128
        dim3 grid(num_q_tokens, n_kv_heads * n_qblocks_per_token, 2);
        dim3 block(128);

        if (cache_type == MT_AITER_CACHE_TURBO3) {
            if (head_size == 128 && block_size == 16) {
                mt_scatter_kv_turbo3_aiter_kernel<128, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else if (head_size == 256 && block_size == 16) {
                mt_scatter_kv_turbo3_aiter_kernel<256, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else {
                GGML_ABORT("AITER TURBO3 scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
            }
        } else if (cache_type == MT_AITER_CACHE_TURBO4) {
            if (head_size == 128 && block_size == 16) {
                mt_scatter_kv_turbo4_aiter_kernel<128, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else if (head_size == 256 && block_size == 16) {
                mt_scatter_kv_turbo4_aiter_kernel<256, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else {
                GGML_ABORT("AITER TURBO4 scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
            }
        } else {  // MT_AITER_CACHE_TURBO4_FP8 — MAD-214 Phase 1G-G
            // Grid: (num_tokens, n_kv_heads, 2_for_K_and_V), 256 threads.
            // Block topology differs from turbo3/4 (one (token, kv_head) row
            // is one 162-byte BS=256 block) so it uses its own grid shape.
            // MAD-227: registry-served hadamard flag picks the kernel
            // template — runtime branch outside the kernel, no perf cost.
            const bool apply_h = mt_turbo_fp8::hadamard_required();
            dim3 fp8_grid(num_q_tokens, n_kv_heads, 2);
            dim3 fp8_block(256);
            if (head_size == 256 && block_size == 16) {
                if (apply_h) {
                    mt_scatter_kv_turbo4_fp8_aiter_kernel<256, 16, true><<<fp8_grid, fp8_block, 0, stream>>>(
                        k_cache->data, v_cache->data,
                        (const __half*) k_cur->data, (const __half*) v_cur->data,
                        (const int32_t*) slot_mapping->data,
                        d_centroids_k, d_centroids_v,
                        n_kv_heads);
                } else {
                    mt_scatter_kv_turbo4_fp8_aiter_kernel<256, 16, false><<<fp8_grid, fp8_block, 0, stream>>>(
                        k_cache->data, v_cache->data,
                        (const __half*) k_cur->data, (const __half*) v_cur->data,
                        (const int32_t*) slot_mapping->data,
                        d_centroids_k, d_centroids_v,
                        n_kv_heads);
                }
            } else {
                GGML_ABORT("AITER TURBO4_FP8 scatter: only (head_size=256, block_size=16) wired (got %d, %d)",
                           head_size, block_size);
            }
        }
    }

    // ── 2. Allocate AITER workspace + cu_seqlens ──
    const size_t segm_out_n   = mt_aiter_uattn_segm_output_bytes(&shape, num_q_tokens) / sizeof(float);
    const size_t segm_ms_n    = mt_aiter_uattn_segm_max_bytes(&shape, num_q_tokens)    / sizeof(float);
    ggml_cuda_pool_alloc<float>   segm_out_buf(ctx.pool(), segm_out_n);
    ggml_cuda_pool_alloc<float>   segm_max_buf(ctx.pool(), segm_ms_n);
    ggml_cuda_pool_alloc<float>   segm_exp_buf(ctx.pool(), segm_ms_n);
    ggml_cuda_pool_alloc<int32_t> cu_seqlens_buf(ctx.pool(), (size_t)(num_seqs + 1));

    mt_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(
        cu_seqlens_buf.get(), (const int32_t*) q_lens->data, num_seqs);

    // ── MAD-227: optional Q pre-rotation for Hadamard-mode FP8 ──
    // Identity (QH)·(HK)^T = QK^T requires rotating BOTH Q and K. K is
    // rotated in the FP8 scatter kernel above (APPLY_HADAMARD=true variant);
    // Q is rotated here into a pool-allocated scratch. q->data is untouched.
    // Allocation only happens when both (a) registry says hadamard mode AND
    // (b) cache is turbo-FP8 — non-FP8 paths bypass entirely.
    const __half * q_ptr = (const __half *) q->data;
    ggml_cuda_pool_alloc<__half> q_rot_scratch(ctx.pool());
    if (cache_type == MT_AITER_CACHE_TURBO4_FP8 && mt_turbo_fp8::hadamard_required()) {
        const size_t q_elts = (size_t) num_q_tokens * n_heads * head_size;
        q_rot_scratch.alloc(q_elts);
        hipMemcpyAsync(q_rot_scratch.get(), q->data,
                       q_elts * sizeof(__half), hipMemcpyDeviceToDevice, stream);
        const hipError_t herr = mt_turbo_fp8_fwht_half(
            stream, q_rot_scratch.get(),
            (int)(num_q_tokens * n_heads), head_size, head_size);
        if (herr != hipSuccess) {
            GGML_ABORT("mt_turbo_fp8_fwht_half(Q) launch failed: %s", hipGetErrorString(herr));
        }
        q_ptr = q_rot_scratch.get();
    }

    // ── 3. Launch AITER attention via the runtime wrapper ──
    mt_aiter_uattn_args_t args = {};
    args.shape        = shape;
    args.q            = (void *) q_ptr;
    args.k_cache      = k_cache->data;
    args.v_cache      = v_cache->data;
    args.out          = dst->data;
    args.segm_output  = segm_out_buf.get();
    args.segm_max     = segm_max_buf.get();
    args.segm_expsum  = segm_exp_buf.get();
    args.block_tables = (const int32_t*) block_tables->data;
    args.seq_lens     = (const int32_t*) context_lens->data;
    args.query_start_len = cu_seqlens_buf.get();

    float * ones = descale_ones_device();
    args.q_descale   = ones;
    args.k_descale   = ones;
    args.v_descale   = ones;
    args.out_scale   = ones;

    // MAD-214: pass per-(layer, kv-dir) centroid LUTs for the Triton FP8 path.
    // null for non-FP8 cache types (the kernel ignores them under constexpr).
    args.centroids_k = d_centroids_k;
    args.centroids_v = d_centroids_v;

    args.scale              = scale;
    args.num_seqs           = num_seqs;
    args.num_q_tokens       = num_q_tokens;
    args.block_table_stride = max_bps;
    args.q_stride_0         = (int64_t) n_heads * head_size;
    args.output_stride_0    = args.q_stride_0;
    args.k_stride_0         = (int64_t) block_size * n_kv_heads * head_size;
    args.k_stride_1         = (int64_t) n_kv_heads * head_size;
    args.k_stride_2         = head_size;
    args.v_stride_0         = args.k_stride_0;
    args.v_stride_1         = args.k_stride_1;
    args.v_stride_2         = args.k_stride_2;

    hipError_t err = mt_aiter_unified_attn(stream, &args);
    if (err != hipSuccess) {
        GGML_ABORT("mt_aiter_unified_attn launch failed: %s", hipGetErrorString(err));
    }
}

}  // namespace mt

#endif  // GGML_HIP_AITER
