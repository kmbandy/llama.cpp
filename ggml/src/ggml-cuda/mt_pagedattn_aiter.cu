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
#include <ctime>
#include <atomic>
#include <mutex>
#include <set>
#include <map>
#include <array>

// The runtime AITER wrapper. Lives in aiter-integration's static library
// (libaiter_triton_aot.a), linked into ggml-hip when GGML_HIP_AITER=ON.
// Header propagated via aiter_triton_aot's PUBLIC target_include_directories.
#include "mt_aiter_unified_attn.h"
#include "mt_turbo_fp8_lut_registry.h"  // MAD-214: per-(layer, kv-dir) centroid LUT lookup
#include "turbo_fp8_hadamard.cuh"      // MAD-227: fp16 FWHT for Q pre-rotation

#include <cstring>
#include <sys/stat.h>  // MAD-214 Option F: mkdir for dump dir
#include <unordered_map>
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

// Non-static export of parse_layer_from_kv_cache_name (declared in
// mt_pagedattn_aiter.cuh) for the R4D adapter's turbo4kv path
// (mt_pagedattn_r4d.cu), which needs to bind the same per-layer centroid
// LUTs this file uses without duplicating the tensor-name parse.
int mt_aiter_parse_layer_from_kv_cache_name(const char * name) {
    return parse_layer_from_kv_cache_name(name);
}

// Non-static export of the turbo4_fp8 scatter launch (declared in
// mt_pagedattn_aiter.cuh) for the R4D adapter's turbo4kv path. Reproduces the
// AITER call site's launch exactly (mt_pagedattn_aiter.cu, MT_AITER_CACHE_
// TURBO4_FP8 scatter branch above): LUT pointers via get_lut_device_ptr,
// Hadamard flag via hadamard_required() picking the kernel template, grid
// (n_tokens, n_kv_heads, 2_for_K_and_V), 256 threads.
void mt_aiter_scatter_kv_turbo4_fp8_launch(
        void * k_cache, void * v_cache,
        const __half * k_cur, const __half * v_cur,
        const int32_t * slot_mapping,
        int n_tokens, int n_kv_heads, int head_size, int block_size,
        int layer, cudaStream_t stream) {
    GGML_ASSERT(head_size == 256 && block_size == 16 &&
                "mt_aiter_scatter_kv_turbo4_fp8_launch: only (head_size=256, block_size=16) wired");

    const uint8_t * d_centroids_k = mt_turbo_fp8::get_lut_device_ptr(layer, mt_turbo_fp8::KV_K);
    const uint8_t * d_centroids_v = mt_turbo_fp8::get_lut_device_ptr(layer, mt_turbo_fp8::KV_V);
    GGML_ASSERT(d_centroids_k && d_centroids_v &&
                "mt_aiter_scatter_kv_turbo4_fp8_launch: centroid LUT lookup returned null");

    const bool apply_h = mt_turbo_fp8::hadamard_required();
    dim3 grid(n_tokens, n_kv_heads, 2);
    dim3 block(256);
    if (apply_h) {
        mt_scatter_kv_turbo4_fp8_aiter_kernel<256, 16, true><<<grid, block, 0, stream>>>(
            k_cache, v_cache, k_cur, v_cur, slot_mapping, d_centroids_k, d_centroids_v, n_kv_heads);
    } else {
        mt_scatter_kv_turbo4_fp8_aiter_kernel<256, 16, false><<<grid, block, 0, stream>>>(
            k_cache, v_cache, k_cur, v_cur, slot_mapping, d_centroids_k, d_centroids_v, n_kv_heads);
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

// MAD-2026-09-12 predequant-scratch (predequant-scratch-0912.txt): build a
// compacted block table for the gfx1030 fp8 pre-dequant scratch cache
// (ggml-cuda/aiter-integration/wrappers/mt_aiter_unified_attn.cpp's
// ensure_predequant_scratch() + the dequant_turbo4_fp8_bs256_to_f16_2d
// kernel). The old scheme sized that scratch cache by the paged cache's
// TOTAL physical block count (args.num_blocks, from `cap` below) — 512
// MiB/cache x2, permanently, at production ctx=524288/block=256/2 kv
// heads/head=256. Only kernel A/B below fix that; num_blocks_fp8 (still
// computed a few lines down) is unrelated (feeds the diagnostic block-table
// scan) and is left alone.
//
// Kernel A: per-seq block count (ceil(seq_len/block_size)) + exclusive
// prefix sum + grand total, single-thread — num_seqs is tiny in production
// (1-2; a prefill batch), so a single-workgroup sequential scan is simply
// not worth parallelizing (mirrors mt_build_cu_seqlens_kernel above).
// q_lens must gate this scan the same way paged_max_ctx_len() (which sizes
// num_scratch_blocks, src/llama-graph.cpp:569) gates its max: an idle slot's
// seq_lens entry stays real and nonzero across ubatches it doesn't
// participate in, so counting it here without the same filter can inflate
// the total past a bound that was never sized to include it.
__global__ void mt_aiter_predequant_scan_kernel(
    const int32_t * __restrict__ seq_lens,
    const int32_t * __restrict__ q_lens,
    int32_t         block_size,
    int32_t         num_seqs,
    int32_t       * __restrict__ out_counts,   // [num_seqs]
    int32_t       * __restrict__ out_prefix,   // [num_seqs], exclusive
    int32_t       * __restrict__ out_total) {  // [1]
    if (threadIdx.x != 0) return;
    int32_t running = 0;
    for (int s = 0; s < num_seqs; ++s) {
        int32_t sl  = seq_lens[s];
        int32_t cnt = (q_lens[s] > 0 && sl > 0) ? (sl + block_size - 1) / block_size : 0;
        out_counts[s] = cnt;
        out_prefix[s] = running;
        running += cnt;
    }
    *out_total = running;
}

// Kernel B: fill the compacted [num_seqs, block_table_stride] scratch table
// — scratch_table[s][j] = prefix[s]+j for a live, valid slot (j < counts[s]
// AND the physical block_tables[s][j] entry isn't kInvalidBlockTableEntry),
// -1 otherwise. The validity check against the ORIGINAL table (not just the
// live-range cutoff) matters: it keeps this compacted table's -1 convention
// exactly as defensive as block_tables_ptr's own -1 check inside
// dequant_turbo4_fp8_bs256_to_f16_2d, so a hole in the physical table still
// makes the dequant/F16-shadow pair skip that slot instead of touching an
// uninitialized scratch block.
//
// MAD-2026-09-12 predequant-overflow-guard (draft-ubatch-split-0912.txt,
// "VERIFY-BATCH FAULT" / FIX): `counts`/`prefix` are derived from the REAL,
// per-token-refreshed context_lens tensor (mt_aiter_predequant_scan_kernel
// above), but the destination f16 scratch cache the compacted index feeds
// (dequant_turbo4_fp8_bs256_to_f16_2d, kernels/unified_attention.py) is
// allocated from a DIFFERENT, separately-derived bound
// (`num_scratch_blocks`, computed from op_params[5] / max_ctx_len_param —
// see mt_pagedattn_aiter.cu's caller below). If those two disagree (the
// real live block count exceeds what the scratch cache was allocated for),
// the old code wrote `prefix[s]+j` unconditionally, and the dequant kernel
// would later store at that compact index into a too-small buffer — an
// out-of-bounds device write ("Page not present"). Clamp here instead: a
// compact index >= num_scratch_blocks is never written into scratch_table
// (stays -1, the same "skip this slot" convention already used for a hole
// in the physical table), so dequant_turbo4_fp8_bs256_to_f16_2d's own
// `if (scratch_block_idx_i32 < 0) return;` (kernels/unified_attention.py)
// early-exits on it — no OOB write can happen. `overflow_flag` (may be
// nullptr — a failed guard-state allocation degrades to "no flag", not a
// crash) is set to 1 (plain store, not atomic — every thread that hits this
// writes the same value 1, so a race here cannot produce a wrong result)
// so the host can report the disagreement without polling every kernel
// launch or adding a synchronous device readback.
__global__ void mt_aiter_predequant_fill_table_kernel(
    const int32_t * __restrict__ orig_table,    // [num_seqs, block_table_stride], physical
    const int32_t * __restrict__ counts,        // [num_seqs]
    const int32_t * __restrict__ prefix,        // [num_seqs], exclusive
    int32_t         block_table_stride,
    int32_t         num_scratch_blocks,         // MAD-2026-09-12: scratch cache's real capacity
    int32_t       * __restrict__ overflow_flag, // MAD-2026-09-12: 1 int32, may be nullptr
    int32_t       * __restrict__ scratch_table) { // [num_seqs, block_table_stride]
    const int s = blockIdx.x;
    const int j = blockIdx.y * (int) blockDim.x + threadIdx.x;
    if (j >= block_table_stride) return;
    const size_t idx = (size_t) s * (size_t) block_table_stride + (size_t) j;
    int32_t out = -1;
    if (j < counts[s] && orig_table[idx] >= 0) {
        const int32_t compact = prefix[s] + j;
        if (compact < num_scratch_blocks) {
            out = compact;
        } else if (overflow_flag != nullptr) {
            *overflow_flag = 1;
        }
    }
    scratch_table[idx] = out;
}

// MAD-2026-09-12 predequant-overflow-guard: per-stream device flag + pinned
// host mirror + event that lets the host learn, WITHOUT ever blocking on
// this stream, whether mt_aiter_predequant_fill_table_kernel above had to
// clamp (skip) a compacted index because num_scratch_blocks disagreed with
// the real live block count. Keyed by stream for the same reason
// CachedHandles::scratch_by_stream is (mt_aiter_unified_attn.cpp) — the
// target and an independent draft llama_context each own their own
// hipStream_t even on the same physical device.
//
// Checking is deliberately deferred to the START of the NEXT call on this
// stream (not the end of this one): the D2H copy issued at the end of a
// call is only ORDERED after this call's kernels on the stream, not
// necessarily COMPLETE by the time the host issues the next call — HOST
// issue order never implies device completion. A hipEventQuery (never
// hipEventSynchronize/hipStreamSynchronize/hipDeviceSynchronize) is the
// only non-blocking way to learn "has this specific copy landed yet";
// blocking here would serialize the issuing host thread in front of
// whatever else it still has to submit, which for the meta backend's
// single-host-thread AllReduce dispatch is a deadlock, not just a stall.
namespace {
struct mt_predequant_overflow_guard {
    int32_t  * flag_dev             = nullptr; // device-resident, 1 int32
    int32_t  * flag_host            = nullptr; // pinned host mirror
    hipEvent_t copy_event           = nullptr; // recorded right after the D2H copy
    bool       pending              = false;   // a copy is in flight, not yet checked
    int32_t    num_scratch_blocks   = 0;       // for the abort message
    int32_t    requested_total      = 0;
    int32_t    max_ctx_len_param    = 0;
};
std::mutex g_predequant_overflow_mu;
std::unordered_map<hipStream_t, mt_predequant_overflow_guard> g_predequant_overflow_by_stream;
} // namespace

// Call right BEFORE issuing this call's fill-table kernel. Non-blockingly
// checks the PREVIOUS call's overflow flag (if its D2H copy has landed) and
// GGML_ABORTs if it fired; then resets the flag to 0 for THIS call (async,
// same stream) and returns the device pointer the fill kernel should write
// to. Returns nullptr only if the one-time allocation of the guard's own
// state failed — the caller must treat a null return as "no guard this
// call" (degrades to the pre-existing, unguarded behavior; never itself a
// fault).
static int32_t * mt_aiter_predequant_overflow_guard_begin(
        hipStream_t stream, int32_t num_scratch_blocks, int32_t requested_total,
        int32_t max_ctx_len_param) {
    std::lock_guard<std::mutex> lock(g_predequant_overflow_mu);
    mt_predequant_overflow_guard & st = g_predequant_overflow_by_stream[stream];

    if (st.pending && st.copy_event != nullptr) {
        if (hipEventQuery(st.copy_event) == hipSuccess) {
            if (st.flag_host != nullptr && *st.flag_host != 0) {
                int dev = 0;
                (void) hipGetDevice(&dev);
                GGML_ABORT(
                    "mt_aiter_unified_attn: predequant-scratch compacted index "
                    "exceeded num_scratch_blocks on a previous call -- the write "
                    "was skipped (no out-of-bounds write occurred), but the scratch "
                    "cache is undersized for this stream's real live context length "
                    "(num_scratch_blocks=%d requested_total=%d max_ctx_len_param=%d "
                    "stream=%p device=%d)\n",
                    st.num_scratch_blocks, st.requested_total, st.max_ctx_len_param,
                    (void *) stream, dev);
            }
            st.pending = false;
        }
        // else: hipErrorNotReady -- the copy hasn't landed yet. Do not wait;
        // just try again at the start of the next call on this stream.
    }

    if (st.flag_dev == nullptr) {
        if (hipMalloc(&st.flag_dev, sizeof(int32_t)) != hipSuccess) {
            st.flag_dev = nullptr;
            return nullptr;
        }
    }
    if (st.flag_host == nullptr) {
        if (hipHostMalloc(&st.flag_host, sizeof(int32_t)) != hipSuccess) {
            st.flag_host = nullptr; // guard degrades to "reset only, never checked"
        } else {
            *st.flag_host = 0;
        }
    }
    if (st.copy_event == nullptr) {
        if (hipEventCreateWithFlags(&st.copy_event, hipEventDisableTiming) != hipSuccess) {
            st.copy_event = nullptr;
        }
    }

    (void) hipMemsetAsync(st.flag_dev, 0, sizeof(int32_t), stream);

    st.num_scratch_blocks = num_scratch_blocks;
    st.requested_total    = requested_total;
    st.max_ctx_len_param  = max_ctx_len_param;

    return st.flag_dev;
}

// Call right AFTER issuing this call's fill-table kernel: async-copies the
// flag back to the pinned host mirror on the SAME stream and records the
// event mt_aiter_predequant_overflow_guard_begin() will non-blockingly poll
// on the NEXT call. No-op if this stream's guard state isn't usable (no
// flag_host / no event) -- the same "degrade, never fault" contract as
// _begin().
static void mt_aiter_predequant_overflow_guard_end(hipStream_t stream) {
    std::lock_guard<std::mutex> lock(g_predequant_overflow_mu);
    auto it = g_predequant_overflow_by_stream.find(stream);
    if (it == g_predequant_overflow_by_stream.end()) {
        return;
    }
    mt_predequant_overflow_guard & st = it->second;
    if (st.flag_dev == nullptr || st.flag_host == nullptr || st.copy_event == nullptr) {
        return;
    }
    (void) hipMemcpyAsync(st.flag_host, st.flag_dev, sizeof(int32_t), hipMemcpyDeviceToHost, stream);
    (void) hipEventRecord(st.copy_event, stream);
    st.pending = true;
}

// ─────────────────────────────────────────────────────────────────────────

// MAD-XXX diag (2026-09-10, temporary): env-gated per-launch sync probe for the
// turbo4_fp8 gfx1030 memory fault. Prints "-> <what>" before a launch and
// "   ok <what>" once it has drained. The last "->" with no matching "ok" names
// the faulting launch. GPU memory faults kill the process, so the localisation
// comes from the log tail, not from the returned status.
static bool mt_aiter_sync_probe_on() {
    static int on = -1;
    if (on < 0) { const char * e = std::getenv("MAD_AITER_SYNC_PROBE"); on = (e && *e && e[0] != '0') ? 1 : 0; }
    return on != 0;
}
static void mt_aiter_sync_probe(cudaStream_t stream, const char * what, int layer) {
    if (!mt_aiter_sync_probe_on()) { return; }
    int dev = ggml_cuda_get_device();
    hipError_t e = hipStreamSynchronize(stream);
    std::fprintf(stderr, "[sync-probe] dev=%d l=%d   ok %s%s%s\n", dev, layer, what,
                 e == hipSuccess ? "" : " ERR=", e == hipSuccess ? "" : hipGetErrorString(e));
}
static void mt_aiter_sync_probe_pre(const char * what, int layer) {
    if (!mt_aiter_sync_probe_on()) { return; }
    std::fprintf(stderr, "[sync-probe] dev=%d l=%d -> %s\n", ggml_cuda_get_device(), layer, what);
}


// MAD-XXX diag (2026-09-10, v2): BLOCK-TABLE VALIDITY SCAN.
//
// Answers: does this device ever READ a kInvalidBlockTableEntry (-1) at an
// index the kernel treats as valid? That is the difference between "gfx1030
// faults" and "gfx1201 silently reads wild memory", since BOTH targets compile
// the turbo-FP8 loads to raw global_load with no bounds check.
//
// v1 was INVALID: it did hipMemcpyAsync + hipStreamSynchronize on the op's
// stream, which fails under HIP graph capture ("operation not permitted when
// stream is capturing") and returned early, silently skipping every captured
// call. v2 copies on a DEDICATED non-blocking stream that is never captured.
// Safe because the block table is written before the op and is not mutated
// during it. Counters are PER DEVICE (v1 shared one counter across devices,
// so only dev 0 ever hit the print condition).
static hipStream_t mt_aiter_scan_stream(int dev) {
    static std::mutex mu;
    static std::map<int, hipStream_t> streams;
    std::lock_guard<std::mutex> g(mu);
    auto it = streams.find(dev);
    if (it != streams.end()) { return it->second; }
    hipStream_t st = nullptr;
    if (hipStreamCreateWithFlags(&st, hipStreamNonBlocking) != hipSuccess) { st = nullptr; }
    streams[dev] = st;
    return st;
}

static void mt_aiter_scan_block_table(const ggml_tensor * block_tables,
                                      const ggml_tensor * context_lens,
                                      int num_seqs, int max_bps, int block_size,
                                      long blocks_capacity) {
    static int on = -1;
    if (on < 0) { const char * e = std::getenv("MAD_AITER_BT_SCAN"); on = (e && *e && e[0] != '0') ? 1 : 0; }
    if (!on) { return; }

    const int dev = ggml_cuda_get_device();
    hipStream_t st = mt_aiter_scan_stream(dev);
    if (!st) { return; }

    std::vector<int32_t> bt((size_t) num_seqs * max_bps);
    std::vector<int32_t> cl((size_t) num_seqs);
    if (hipMemcpyAsync(bt.data(), block_tables->data, bt.size()*sizeof(int32_t),
                       hipMemcpyDeviceToHost, st) != hipSuccess) { return; }
    if (hipMemcpyAsync(cl.data(), context_lens->data, cl.size()*sizeof(int32_t),
                       hipMemcpyDeviceToHost, st) != hipSuccess) { return; }
    if (hipStreamSynchronize(st) != hipSuccess) {
        static std::atomic<int> warned{0};
        if (warned.fetch_add(1) == 0) {
            std::fprintf(stderr, "[bt-scan] dev=%d scan stream sync FAILED - results invalid\n", dev);
        }
        return;
    }

    struct Acc { long calls=0, bad_calls=0, neg=0, oor=0, worst_neg_run=0; };
    static std::mutex mu; static std::map<int, Acc> per_dev;
    std::lock_guard<std::mutex> g(mu);
    Acc & a = per_dev[dev];
    ++a.calls;

    int neg = 0, oor = 0, fb_seq = -1, fb_idx = -1; int32_t fb_val = 0;
    for (int sq = 0; sq < num_seqs; ++sq) {
        const int n_used = (cl[sq] + block_size - 1) / block_size;
        for (int b = 0; b < n_used && b < max_bps; ++b) {
            const int32_t v = bt[(size_t) sq * max_bps + b];
            if (v < 0)                          { ++neg; if (fb_seq<0){fb_seq=sq;fb_idx=b;fb_val=v;} }
            else if (v >= (int32_t) blocks_capacity) { ++oor; if (fb_seq<0){fb_seq=sq;fb_idx=b;fb_val=v;} }
        }
    }
    if (neg || oor) {
        ++a.bad_calls; a.neg += neg; a.oor += oor;
        if (a.bad_calls <= 8 || a.bad_calls % 256 == 0) {
            std::fprintf(stderr,
                "[bt-scan] dev=%d *** BAD *** neg=%d oor=%d first(seq=%d blk=%d val=%d) ctx=%d cap=%ld "
                "| calls=%ld bad=%ld neg_tot=%ld oor_tot=%ld\n",
                dev, neg, oor, fb_seq, fb_idx, fb_val,
                fb_seq >= 0 ? cl[fb_seq] : -1, blocks_capacity,
                a.calls, a.bad_calls, a.neg, a.oor);
        }
    }
    if (a.calls % 1024 == 1) {
        std::fprintf(stderr, "[bt-scan] dev=%d summary: calls=%ld bad_calls=%ld neg_tot=%ld oor_tot=%ld\n",
                     dev, a.calls, a.bad_calls, a.neg, a.oor);
    }
}

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
    // Per-device: a single process-wide allocation is the same class of bug
    // as the AITER kernel-handle cache (d41c22e1d). Under TP the second card
    // must not read a 1.0f that lives on the first card.
    static std::mutex mu;
    static std::unordered_map<int, float *> ptrs;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        dev = 0;
    }
    std::lock_guard<std::mutex> g(mu);
    auto it = ptrs.find(dev);
    if (it != ptrs.end()) return it->second;
    float * ptr = nullptr;
    cudaMalloc((void**) &ptr, sizeof(float));
    const float one = 1.0f;
    cudaMemcpy(ptr, &one, sizeof(float), cudaMemcpyHostToDevice);
    ptrs[dev] = ptr;
    return ptr;
}

// ─────────────────────────────────────────────────────────────────────────
// AITER dispatch entry
// ─────────────────────────────────────────────────────────────────────────
// ── MAD-288 replay-safe scratch (2026-09-16) ─────────────────────────────
// Every scratch buffer this op needs used to come from ctx.pool() per call.
// Captured HIP graphs bake those pool addresses into their launches, and the
// pool hands the same memory to other callers between replays (the 09-12
// pre-dequant table alone churns it by num_seqs*max_bps ints per prefill
// call), so a replayed decode graph could run attention against memory that
// now belongs to something else -- the two-slot freeze of 09-13. The 09-13
// answer was to exclude PAGED_ATTN_MT from graph capture entirely
// (WP_HIP_GRAPHS_PAGED_ATTN, ggml-cuda.cu), which cost every decode step its
// graph replay. This is the fix that exclusion stood in for: scratch that is
// persistent and never moves.
//   * keyed by (device, stream): the two overlapping meta contexts on one
//     device (GGML_META_OVERLAP) each get their own set, so in-flight work
//     on one stream never shares a buffer with the other.
//   * grow-only, and a grown buffer's predecessor is deliberately kept
//     alive: a graph captured against the old address still owns memory
//     that fits the shape it was captured for.
//   * growth happens only on an eager visit. Capturable graphs containing
//     this op get one eager warm-up visit before capture (ggml-cuda.cu,
//     "eager warm-up visit ... PAGED_ATTN_MT"), and a shape that would need
//     more is a different graph key, which warms up eagerly again.
struct mt_aiter_persist_buf {
    void * ptr   = nullptr;
    size_t bytes = 0;
};
enum mt_aiter_persist_slot {
    MT_AITER_PERSIST_SEGM_OUT = 0,
    MT_AITER_PERSIST_SEGM_MAX,
    MT_AITER_PERSIST_SEGM_EXP,
    MT_AITER_PERSIST_CU_SEQLENS,
    MT_AITER_PERSIST_PREDQ_COUNTS,
    MT_AITER_PERSIST_PREDQ_PREFIX,
    MT_AITER_PERSIST_PREDQ_TOTAL,
    MT_AITER_PERSIST_PREDQ_TABLE,
    MT_AITER_PERSIST_Q_ROT,
    MT_AITER_PERSIST_COUNT
};
static std::mutex g_mt_aiter_persist_mutex;
static std::map<std::pair<int, cudaStream_t>, std::array<mt_aiter_persist_buf, MT_AITER_PERSIST_COUNT>> g_mt_aiter_persist;

static const char * mt_aiter_persist_slot_name(mt_aiter_persist_slot slot) {
    switch (slot) {
        case MT_AITER_PERSIST_SEGM_OUT:       return "MT_AITER_PERSIST_SEGM_OUT";
        case MT_AITER_PERSIST_SEGM_MAX:       return "MT_AITER_PERSIST_SEGM_MAX";
        case MT_AITER_PERSIST_SEGM_EXP:       return "MT_AITER_PERSIST_SEGM_EXP";
        case MT_AITER_PERSIST_CU_SEQLENS:     return "MT_AITER_PERSIST_CU_SEQLENS";
        case MT_AITER_PERSIST_PREDQ_COUNTS:   return "MT_AITER_PERSIST_PREDQ_COUNTS";
        case MT_AITER_PERSIST_PREDQ_PREFIX:   return "MT_AITER_PERSIST_PREDQ_PREFIX";
        case MT_AITER_PERSIST_PREDQ_TOTAL:    return "MT_AITER_PERSIST_PREDQ_TOTAL";
        case MT_AITER_PERSIST_PREDQ_TABLE:    return "MT_AITER_PERSIST_PREDQ_TABLE";
        case MT_AITER_PERSIST_Q_ROT:          return "MT_AITER_PERSIST_Q_ROT";
        case MT_AITER_PERSIST_COUNT:          return "MT_AITER_PERSIST_COUNT";
    }
    return "MT_AITER_PERSIST_?";
}

// MAD-2026-09-20 diag: WP_ALLOC_LOG=1 attribution for this grow-only, never-
// freed cache (see the design note above). Mirrors ggml-cuda.cu's
// wp_alloc_log() format exactly ("wp alloc-log HH:MM:SS.mmm <what>
// device=N size=..MiB extra=..MiB") so the same journal grep finds both;
// that helper has internal linkage there and isn't reachable from this
// translation unit, so this is a small local twin, diagnostic-only (no
// behavior change -- the old buffer is still deliberately leaked, per the
// comment above mt_aiter_persist_buf).
static void mt_aiter_persist_alloc_log(int device, size_t old_bytes, size_t new_bytes) {
    static const bool enabled = [] {
        const char * e = std::getenv("WP_ALLOC_LOG");
        return e != nullptr && e[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv; localtime_r(&ts.tv_sec, &tmv);
    std::fprintf(stderr, "wp alloc-log %02d:%02d:%02d.%03ld mt_aiter_persist_grow device=%d size=%.1fMiB extra=%.1fMiB\n",
                 tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ts.tv_nsec / 1000000, device,
                 new_bytes / 1048576.0, old_bytes / 1048576.0);
}

// Per-slot cap (MT_AITER_PERSIST_MAX_MB, default 1024): a hard ceiling on a
// single slot's per-(device,stream) buffer. Returns nullptr when need
// exceeds it (logged once per slot) instead of growing without bound -- the
// 2026-07-16 serving OOM: SEGM_OUT alone reached ~864 MB/slot on the R9700
// at num_q_tokens=2048, and the doubling growth below (now exact-fit)
// leaked ~1x that again per grow step. Callers must treat nullptr as a
// clean, explicit abort with the slot name, never dereference it.
static size_t mt_aiter_persist_max_bytes() {
    static const size_t max_bytes = [] {
        const char * e = std::getenv("MT_AITER_PERSIST_MAX_MB");
        long mb = (e && e[0]) ? std::atol(e) : 1024;
        if (mb <= 0) {
            mb = 1024;
        }
        return (size_t) mb * 1024u * 1024u;
    }();
    return max_bytes;
}

template <typename T>
static T * mt_aiter_persist_get(int device, cudaStream_t stream, mt_aiter_persist_slot slot, size_t n_elems) {
    const size_t need = n_elems * sizeof(T);
    const size_t max_bytes = mt_aiter_persist_max_bytes();
    if (need > max_bytes) {
        static std::array<bool, MT_AITER_PERSIST_COUNT> cap_logged = {};
        if (!cap_logged[slot]) {
            cap_logged[slot] = true;
            std::fprintf(stderr, "mt_aiter_persist_get: slot %d needs %zu B > MT_AITER_PERSIST_MAX_MB cap on device %d\n",
                         (int) slot, need, device);
        }
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mt_aiter_persist_mutex);
    mt_aiter_persist_buf & b = g_mt_aiter_persist[std::make_pair(device, stream)][slot];
    if (need > b.bytes) {
        // Exact-fit growth with a +12.5% pad (rounded up to 256) -- mirrors
        // ensure_predequant_scratch's predequant-cap (mt_aiter_unified_attn.cpp,
        // MAD-2026-09-20). The old doubling growth (max(need, b.bytes*2))
        // permanently overshot: every later still-growing-but-smaller call
        // re-doubled off an already-inflated floor instead of off what THIS
        // call actually needs, and each step leaked the old buffer on top.
        size_t bytes = need + need / 8;
        bytes = (bytes + 255) & ~(size_t) 255;
        void * ptr = nullptr;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        mt_aiter_persist_alloc_log(device, b.bytes, bytes);
        b.ptr   = ptr;
        b.bytes = bytes;
    }
    return (T *) b.ptr;
}


// MT_AITER_UATTN_PROFILE=1 (diagnostic, synchronous): time the op's own
// phases -- scatter, table/workspace prep, unified_attn call -- per device.
namespace {
struct mt_op_prof_acc { double scatter_ms = 0, prep_ms = 0, attn_ms = 0; uint64_t n = 0; };
mt_op_prof_acc g_mt_op_prof[16];
bool mt_op_prof_enabled() {
    static const bool e = [] { const char * v = std::getenv("MT_AITER_UATTN_PROFILE"); return v && v[0] == '1'; }();
    return e;
}
struct mt_op_prof_timer {
    bool on = false; int dev = 0; cudaStream_t st = nullptr; cudaEvent_t e0 = nullptr, e1 = nullptr;
    mt_op_prof_timer(int d, cudaStream_t s) : on(mt_op_prof_enabled()), dev(d), st(s) {
        if (on) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, st); }
    }
    float lap() {
        if (!on) return 0.f;
        cudaEventRecord(e1, st); cudaEventSynchronize(e1);
        float ms = 0.f; cudaEventElapsedTime(&ms, e0, e1); cudaEventRecord(e0, st);
        return ms;
    }
    ~mt_op_prof_timer() { if (on) { cudaEventDestroy(e0); cudaEventDestroy(e1); } }
};
} // namespace

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
    // MAD-2026-09-12 predequant-sync-fix (predequant-sync-fix-0912.txt):
    // op_params[5] (max live context, across all seqs in this ubatch) is
    // populated host-side by llm_graph_input_attn_kv::update_paged_attn_max_ctx_len()
    // from set_input() -- see src/llama-graph.cpp:550-574 (MAD-378) -- i.e.
    // it is already known on the host before this op ever launches, with no
    // device readback required. Used below to size the fp8 pre-dequant
    // scratch cache without the D2H + hipStreamSynchronize this file used
    // to do per call. 0 means "unset" (e.g. a cold graph executed before
    // its first set_input, such as during warmup) -- same convention as
    // op_params[4]/[5] in mt_pagedattn.cu.
    const int32_t max_ctx_len_param = ((const int32_t *)(op_params_f + 5))[0];
    // MAD-LAB (draft-KV-paged num_seqs fix, 2026-09-12): op_params[6] is the
    // REAL number of live/active sequences in this ubatch
    // (llama_ubatch::n_seqs_unq), populated the same way as op_params[5]
    // above by llm_graph_input_attn_kv::update_paged_attn_n_seqs_active()
    // (src/llama-graph.cpp, MAD-378-style per-device-clone propagation). 0
    // means "unset" (same convention as op_params[4]/[5]) -- num_seqs_dispatch
    // then falls back to num_seqs (== block_tables->ne[1], the cache's
    // static n_seq_max), i.e. exactly today's (pre-fix) behavior.
    const int32_t n_seqs_active_param = ((const int32_t *)(op_params_f + 6))[0];

    const int head_size      = (int) q->ne[0];
    const int n_heads        = (int) q->ne[1];
    // block_tables->ne[1] is the paged cache's STATIC n_seq_max -- the
    // block_table/context_lens/q_lens tensors are always allocated at this
    // width (src/llama-kv-cache-paged.cpp:482, src/llama-graph.cpp:3330/3334)
    // regardless of how many of those slots are actually live this call, and
    // slots are seq-id-indexed, NOT compacted -- a single live sequence can
    // sit at any slot index, so this value is the correct (and only safe)
    // bound for anything that INDEXES into those arrays (grid dims, the
    // reduce-segments phase, ALL_DECODE/g3_x sizing, cu_seqlens
    // construction) or that must conservatively cover every possible live
    // slot. Keep using it for all of that -- see num_seqs_dispatch below for
    // the one place (the 2D/3D + large-prefill dispatch HEURISTIC, which
    // only ever consumes num_seqs as a divisor/scale factor, never as an
    // array bound) that needs the real count instead.
    const int num_seqs       = (int) block_tables->ne[1];
    // Real live count for dispatch-only decisions (avg_q_len, 2D-vs-3D
    // occupancy, the large-prefill cutover below) -- falls back to the old,
    // static `num_seqs` when unset (cold graph / pre-set_input warmup),
    // reproducing today's behavior exactly in that case.
    //
    // MAD-LAB (draft-KV-paged fault-analysis fix, 2026-09-12): also HARD-CLAMP
    // to `num_seqs` (never exceed it). A cache can never legitimately have
    // MORE live sequences than its own n_seq_max, so op_params[6] reading
    // back larger than that is always invalid input for THIS cache instance
    // -- e.g. a stale/reserve-time ubatch value, or any future bug in
    // ubatch->n_seqs_unq's population -- and must never be trusted over the
    // known-safe static value. This is the critical direction to guard: an
    // over-large num_seqs_dispatch UNDER-estimates avg_q_len
    // (num_q_tokens/num_seqs_dispatch), which can push a genuinely large
    // (e.g. ~2048-token) batch below the large-prefill threshold and
    // misroute it onto the 3D split-K kernel with workspace/grid math sized
    // for a much smaller call -- see draft-kv-paged-0912.txt "FAULT
    // ANALYSIS" for the observed HSA_STATUS_ERROR_MEMORY_FAULT this class of
    // mistake produces. The clamp is a no-op whenever n_seqs_active_param is
    // correct (it is always <= num_seqs by construction in the intended
    // case), so it changes nothing for the calls this optimization targets.
    const int num_seqs_dispatch = n_seqs_active_param > 0
        ? (n_seqs_active_param < num_seqs ? (int) n_seqs_active_param : num_seqs)
        : num_seqs;
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

        // MAD-XXX diag (2026-09-10): ASTRA CANDIDATES 4 AND 5, both fp8-only.
        // Runs ONCE per process (function-local static), then costs a guard-byte
        // test. Prints a definite PASS/FAIL either way, so the answer does not
        // depend on whether the fault happens to land during the run.
        //  C4: are both centroid-LUT bases valid, device-accessible, and on THIS device?
        //  C5: does q actually hold the q_elts the Hadamard Q-copy reads out of it?
        // PER-DEVICE: the previous function-local static ran once per PROCESS and
        // only ever sampled dev 0. dev 1 (gfx1030) is the card that faults.
        static std::atomic<bool> fp8_checked_dev[8] = {};
        const int dv_chk = ggml_cuda_get_device();
        if (dv_chk >= 0 && dv_chk < 8 && !fp8_checked_dev[dv_chk].load(std::memory_order_relaxed)
            && !fp8_checked_dev[dv_chk].exchange(true)) {
            const int dv = dv_chk;
            const bool had = mt_turbo_fp8::hadamard_required();
            const size_t q_elts = (size_t) num_q_tokens * n_heads * head_size;
            const size_t q_want = q_elts * sizeof(__half);
            const size_t q_have = (size_t) ggml_nbytes(q);
            std::fprintf(stderr,
                "[c5-qsize] dev=%d hadamard=%d  q_want=%zu q_have=%zu  %s  "
                "(q->ne=[%ld,%ld,%ld,%ld] k_cur->ne[2]=%d n_heads=%d hs=%d)\n",
                dv, (int) had, q_want, q_have,
                had ? (q_want > q_have ? "*** OVERREAD ***" : "PASS") : "PASS (copy inactive)",
                (long)q->ne[0],(long)q->ne[1],(long)q->ne[2],(long)q->ne[3],
                num_q_tokens, n_heads, head_size);
            for (int k = 0; k < 2; ++k) {
                const void * lut = (k == 0) ? (const void *) d_centroids_k : (const void *) d_centroids_v;
                hipPointerAttribute_t at {};
                const hipError_t e = hipPointerGetAttributes(&at, lut);
                if (e != hipSuccess) {
                    std::fprintf(stderr, "[c4-lut]  dev=%d %s=%p *** hipPointerGetAttributes FAILED: %s ***\n",
                                 dv, k ? "V" : "K", lut, hipGetErrorString(e));
                } else {
                    const bool dev_ok = (at.device == dv);
                    std::fprintf(stderr, "[c4-lut]  dev=%d %s=%p type=%d owner_dev=%d  %s\n",
                                 dv, k ? "V" : "K", lut, (int) at.type, at.device,
                                 dev_ok ? "PASS" : "*** WRONG DEVICE ***");
                }
            }
        }
    }

    // MAD-XXX diag (2026-09-10, temporary): one-shot per-device arena audit for
    // the gfx1030 memory fault at tsa 1,1 + turbo4_fp8. Prints what the Triton
    // address math assumes vs what is actually allocated.
    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = head_size;
    shape.num_q_heads  = n_heads;
    shape.num_kv_heads = n_kv_heads;
    shape.block_size   = block_size;
    shape.cache_type   = cache_type;

    cudaStream_t stream = ctx.stream();
    mt_op_prof_timer mt_prof(ctx.device, stream);

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
    mt_aiter_sync_probe(stream, "scatter", parse_layer_from_kv_cache_name(k_cache->name));
    const float mt_prof_scatter = mt_prof.lap();

    // ── 2. Allocate AITER workspace + cu_seqlens ──
    // MAD-2026-09-12 dispatch-fix: 3D split-K workspace is unused on the 2D
    // path. This MUST call the exact same predicate (same three arguments)
    // as mt_aiter_unified_attn()'s own launch-gate below, or the workspace
    // sizing and the kernel actually launched can disagree — see the
    // comment on mt_aiter_uattn_should_use_2d() in mt_aiter_unified_attn.h.
    // MAD-LAB (draft-KV-paged num_seqs fix): uses num_seqs_dispatch (the
    // real live-seq count when known), NOT num_seqs, and propagates that
    // same value to mt_aiter_unified_attn() via args.num_seqs_active below
    // so its own internal recompute of this exact predicate agrees --
    // required by the "MUST call the exact same predicate" invariant above.
    const bool use_2d = mt_aiter_uattn_should_use_2d(num_q_tokens, num_seqs_dispatch, n_kv_heads) != 0;

    // MAD-LAB (draft-KV-paged fault-analysis-2 fix, 2026-09-12): fail LOUDLY,
    // before any kernel launch, if the 3D split-K path was somehow selected
    // for a batch large enough that it would have qualified for 2D-large
    // under the cache's own STATIC (n_seq_max, always safe) num_seqs — i.e.
    // avg_q_len computed with num_seqs (not num_seqs_dispatch) is still
    // >= the large-prefill threshold. This is a pure sanity re-check, not a
    // new decision: with num_seqs_dispatch now hard-clamped to <= num_seqs
    // (see its declaration above), this condition is believed unreachable
    // -- num_seqs_dispatch <= num_seqs implies
    // avg_q_len(num_seqs_dispatch) >= avg_q_len(num_seqs), so if the
    // static-num_seqs avg_q_len already clears the threshold, the
    // dispatch-num_seqs avg_q_len must too, and use_2d must be true. It's
    // kept as a second, independent layer specifically because the 3D
    // kernel's own internal tiling was never designed or validated for a
    // batch this wide (see draft-kv-paged-0912.txt "FAULT ANALYSIS" /
    // "FAULT ANALYSIS 2" for the HSA_STATUS_ERROR_MEMORY_FAULT this exact
    // grid shape — 1024x2x32 on a ~2044-token batch — produced before the
    // num_seqs clamp existed): a 3D split-K launch this wide is unvalidated
    // territory regardless of how it got selected, so if some future change
    // reopens a path to it, this converts that into a clean, diagnosable
    // process abort instead of a repeat of the same GPU memory fault.
    if (!use_2d && num_seqs > 0 &&
        (num_q_tokens / num_seqs) >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD) {
        GGML_ABORT("AITER paged-attn: 3D split-K selected for an oversized batch "
                   "(num_q_tokens=%d, num_seqs_dispatch=%d, static num_seqs=%d) -- "
                   "this batch would have qualified for the 2D-large path under the "
                   "cache's own static n_seq_max (avg_q_len=%d >= threshold=%d). "
                   "Refusing to launch an unvalidated large-grid 3D dispatch rather "
                   "than risk a repeat of the HSA_STATUS_ERROR_MEMORY_FAULT documented "
                   "in draft-kv-paged-0912.txt 'FAULT ANALYSIS 2'.",
                   num_q_tokens, num_seqs_dispatch, num_seqs,
                   num_q_tokens / num_seqs, (int) MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD);
    }

    // MAD-288: persistent scratch (see mt_aiter_persist_get above); nullptr
    // where the 2D path does not use the segment buffers, as before. The
    // SEGM_* split-KV partials are ONLY requested here, inside the 3D
    // (!use_2d) branch: the 2D prefill paths (use_2d / use_2d_large) never
    // touch those slots, so a large prefill ubatch cannot inflate them --
    // the 2026-07-16 serving OOM grew them via 3D decode-shaped calls and
    // the old doubling growth; mt_aiter_persist_get now grows exact-fit and
    // caps per slot (MT_AITER_PERSIST_MAX_MB). A nullptr return (slot cap
    // hit) aborts cleanly with the slot name instead of dereferencing.
    const int dev = ctx.device;
    float * segm_out_ptr = nullptr;
    float * segm_max_ptr = nullptr;
    float * segm_exp_ptr = nullptr;
    if (!use_2d) {
        segm_out_ptr = mt_aiter_persist_get<float>(dev, stream, MT_AITER_PERSIST_SEGM_OUT, mt_aiter_uattn_segm_output_bytes(&shape, num_q_tokens) / sizeof(float));
        segm_max_ptr = mt_aiter_persist_get<float>(dev, stream, MT_AITER_PERSIST_SEGM_MAX, mt_aiter_uattn_segm_max_bytes(&shape, num_q_tokens)    / sizeof(float));
        segm_exp_ptr = mt_aiter_persist_get<float>(dev, stream, MT_AITER_PERSIST_SEGM_EXP, mt_aiter_uattn_segm_expsum_bytes(&shape, num_q_tokens) / sizeof(float));
        if (!segm_out_ptr || !segm_max_ptr || !segm_exp_ptr) {
            GGML_ABORT("AITER paged-attn: 3D split-KV persist scratch unavailable on device %d "
                       "(MT_AITER_PERSIST_MAX_MB cap; null slot: %s%s%s) -- refusing to launch",
                       dev,
                       segm_out_ptr ? "" : "MT_AITER_PERSIST_SEGM_OUT ",
                       segm_max_ptr ? "" : "MT_AITER_PERSIST_SEGM_MAX ",
                       segm_exp_ptr ? "" : "MT_AITER_PERSIST_SEGM_EXP");
        }
    }
    int32_t * cu_seqlens_ptr = mt_aiter_persist_get<int32_t>(dev, stream, MT_AITER_PERSIST_CU_SEQLENS, (size_t)(num_seqs + 1));
    if (!cu_seqlens_ptr) {
        GGML_ABORT("AITER paged-attn: persist scratch %s unavailable "
                   "(MT_AITER_PERSIST_MAX_MB cap) on device %d",
                   mt_aiter_persist_slot_name(MT_AITER_PERSIST_CU_SEQLENS), dev);
    }

    mt_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(
        cu_seqlens_ptr, (const int32_t*) q_lens->data, num_seqs);
    mt_aiter_sync_probe(stream, "cu_seqlens", parse_layer_from_kv_cache_name(k_cache->name));
    // Total physical blocks (capacity) of the paged turbo4_fp8 cache — feeds
    // ONLY the diagnostic block-table validity scan (mt_aiter_scan_block_table,
    // MAD_AITER_BT_SCAN-gated) below. MAD-2026-09-12 predequant-scratch: this
    // is no longer used to size the gfx1030 pre-dequant scratch cache — see
    // num_scratch_blocks / mt_aiter_predequant_scan_kernel further down,
    // which sizes it by the blocks THIS call actually touches instead.
    long num_blocks_fp8 = 0;
    {
        const long bpb_scan = (k_cache->type == GGML_TYPE_TURBO4_FP8_BS256) ? 162 : 0;
        const long cap = bpb_scan ? ((long) ggml_nbytes(k_cache) / ((long) block_size * n_kv_heads * bpb_scan)) : 0;
        num_blocks_fp8 = cap;
        if (cap > 0) {
            mt_aiter_scan_block_table(block_tables, context_lens, num_seqs, max_bps, block_size, cap);
        }
    }

    // MAD-2026-09-12 predequant-scratch (predequant-scratch-0912.txt):
    // compacted block table + the actual slot count THIS call needs for the
    // gfx1030 fp8 pre-dequant scratch cache — replaces num_blocks_fp8 (the
    // paged cache's total physical capacity) as the scratch-sizing input.
    // Gated to exactly the calls that can take that path (turbo4_fp8 cache +
    // the 2D-large-prefill tile) so decode and every other cache type pay
    // nothing extra. use_2d_large mirrors mt_aiter_unified_attn()'s own
    // large-tile cutover (avg_q_len >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD)
    // exactly — both must agree on which calls are prefill-shaped, same as
    // the use_2d predicate above.
    int32_t * predq_counts_ptr        = nullptr;
    int32_t * predq_prefix_ptr        = nullptr;
    int32_t * predq_total_ptr         = nullptr;
    int32_t * predq_scratch_table_ptr = nullptr;
    int32_t num_scratch_blocks = 0;
    // MAD-LAB (draft-KV-paged num_seqs fix): num_seqs_dispatch, not num_seqs
    // -- see the comment on num_seqs_dispatch's declaration above.
    const bool use_2d_large =
        use_2d && (mt_aiter_uattn_avg_q_len(num_q_tokens, num_seqs_dispatch) >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD);
    const bool want_predequant_scratch = (cache_type == MT_AITER_CACHE_TURBO4_FP8) && use_2d_large;
    if (want_predequant_scratch) {
        // MAD-2026-09-12 predequant-sync-fix (predequant-sync-fix-0912.txt):
        // this used to be `mt_aiter_predequant_scan_kernel` + a D2H
        // hipMemcpyAsync + hipStreamSynchronize(stream) to learn the exact
        // live-block TOTAL on the host before sizing the scratch cache. That
        // sync blocked the issuing host thread on THIS device's stream --
        // under GGML_META_OVERLAP=1 the meta backend's single host thread
        // submits device subgraphs one at a time (ggml_backend_meta_graph_
        // runner::compute(), ggml/src/ggml-backend-meta.cpp:2649-2657), so
        // blocking here stalls it before it can ever submit the OTHER
        // device's subgraph -- including that device's half of any
        // AllReduce whose duplex handshake (ggml/src/ggml-cuda/allreduce.cu,
        // ggml_cuda_ar_dx_slot, "asynchronous, event-driven, no host syncs"
        // by design at allreduce.cu:724-750) this device's stream is
        // already waiting on. Host-level deadlock, not a device fault --
        // see predequant-sync-fix-0912.txt for the full trace.
        //
        // Fix: size the scratch cache from a bound that is host-visible
        // WITHOUT any device readback. op_params[5] (max_ctx_len_param,
        // read above) already carries the per-ubatch max live context
        // length across all seqs, computed host-side at set_input() from
        // the paged cache's own host mirrors (src/llama-graph.cpp:550-574,
        // MAD-378) -- strictly earlier than this op ever launches. Bound:
        //   num_scratch_blocks = num_seqs_dispatch * ceil(max_ctx_len / block_size)
        // (originally num_seqs here -- see MAD-2026-09-20 predequant-cap
        // note just below for why that over-counted). For the production
        // shape (num_seqs_dispatch==1 prefill) this equals the exact
        // live-block total the old scan kernel computed -- no VRAM
        // regression in the case that matters. 0 (unset -- a cold graph run
        // before its first set_input, e.g. warmup) falls back to the old
        // pre-0912 conservative bound (num_seqs_dispatch * max_bps, i.e. the
        // paged cache's full allocated capacity for these live seqs).
        // MAD-2026-09-20 predequant-cap: bound by the REAL live-sequence
        // count (num_seqs_dispatch — the same op_params[6]-derived value the
        // 2D/3D dispatch heuristic above already uses, analogous to the R4D
        // adapter's `num_active`), NOT the cache's static num_seqs (n_seq_max).
        // A cache provisioned for parallel=N but currently serving fewer live
        // slots (e.g. one active prefill under n_seq_max=3) used to size this
        // scratch buffer at N x the true requirement — observed 3x inflation
        // at parallel=3 with a single live slot (predequant-cap-0920.txt).
        // num_seqs_dispatch <= num_seqs always (clamped at its declaration
        // above), so this can only ever shrink the bound relative to the old
        // formula, never grow it.
        const int32_t ctx_len_bound =
            max_ctx_len_param > 0 ? max_ctx_len_param : (int32_t) max_bps * block_size;
        const int32_t blocks_per_seq_bound = (ctx_len_bound + block_size - 1) / block_size;
        num_scratch_blocks = num_seqs_dispatch * blocks_per_seq_bound;

        if (num_scratch_blocks > 0) {
            predq_counts_ptr = mt_aiter_persist_get<int32_t>(dev, stream, MT_AITER_PERSIST_PREDQ_COUNTS, (size_t) num_seqs);
            predq_prefix_ptr = mt_aiter_persist_get<int32_t>(dev, stream, MT_AITER_PERSIST_PREDQ_PREFIX, (size_t) num_seqs);
            predq_total_ptr  = mt_aiter_persist_get<int32_t>(dev, stream, MT_AITER_PERSIST_PREDQ_TOTAL, 1);
            if (!predq_counts_ptr || !predq_prefix_ptr || !predq_total_ptr) {
                GGML_ABORT("AITER paged-attn: fp8-predequant persist scratch unavailable on device %d "
                           "(MT_AITER_PERSIST_MAX_MB cap; null slot: %s%s%s) -- refusing to launch",
                           dev,
                           predq_counts_ptr ? "" : "MT_AITER_PERSIST_PREDQ_COUNTS ",
                           predq_prefix_ptr ? "" : "MT_AITER_PERSIST_PREDQ_PREFIX ",
                           predq_total_ptr  ? "" : "MT_AITER_PERSIST_PREDQ_TOTAL");
            }
            // Kernel A still runs, fully device-side: it produces the exact
            // per-seq counts/prefix the fill kernel below needs to keep the
            // compacted table DENSE (packed by actual live blocks, not the
            // worst-case bound) even though the ALLOCATION above is sized by
            // the bound. out_total is written but deliberately never copied
            // back to host -- that copyback + sync was the bug.
            mt_aiter_predequant_scan_kernel<<<1, 1, 0, stream>>>(
                (const int32_t*) context_lens->data, (const int32_t*) q_lens->data,
                block_size, num_seqs,
                predq_counts_ptr, predq_prefix_ptr, predq_total_ptr);

            predq_scratch_table_ptr = mt_aiter_persist_get<int32_t>(dev, stream, MT_AITER_PERSIST_PREDQ_TABLE, (size_t) num_seqs * (size_t) max_bps);
            if (!predq_scratch_table_ptr) {
                GGML_ABORT("AITER paged-attn: persist scratch %s unavailable on device %d "
                           "(MT_AITER_PERSIST_MAX_MB cap) -- refusing to launch",
                           mt_aiter_persist_slot_name(MT_AITER_PERSIST_PREDQ_TABLE), dev);
            }
            const dim3 fill_grid((unsigned) num_seqs, (unsigned) ((max_bps + 255) / 256));
            // MAD-2026-09-12 predequant-overflow-guard: `counts`/`prefix`
            // above are exact (real context_lens), but num_scratch_blocks is
            // the separately-derived, possibly-stale bound this scratch
            // cache was actually ALLOCATED for. Clamp the fill kernel's
            // writes against it (see the kernel's own comment) and check
            // asynchronously, without ever blocking this stream, whether a
            // clamp fired on the PREVIOUS call.
            // requested_total is unavailable without a host readback (predq_total
            // is intentionally never copied back — see the comment above); -1
            // marks it as "not tracked" rather than duplicating num_scratch_blocks.
            int32_t * predequant_overflow_flag = mt_aiter_predequant_overflow_guard_begin(
                stream, num_scratch_blocks, -1, max_ctx_len_param);
            mt_aiter_predequant_fill_table_kernel<<<fill_grid, 256, 0, stream>>>(
                (const int32_t*) block_tables->data,
                predq_counts_ptr, predq_prefix_ptr, max_bps,
                num_scratch_blocks, predequant_overflow_flag,
                predq_scratch_table_ptr);
            mt_aiter_predequant_overflow_guard_end(stream);
        }
    }

    // MAD-2026-09-20 predequant-pool: per-call f16 K/V dequant scratch from
    // the ggml CUDA pool, replacing the old persistent, never-freed per-
    // (device, stream) grow-only cache that used to live inside
    // mt_aiter_unified_attn.cpp's ensure_predequant_scratch() (deleted) —
    // measured root cause of unbounded per-round VRAM growth under
    // long-running TP serving (WP_ALLOC_LOG=1: predequant_scratch_grow
    // leaking every grow step, two (device, stream) keys, ~2 GB over 7
    // rounds). This scratch is live ONLY for the duration of THIS call: the
    // predequant fill kernel writes it and the immediately-following
    // h_2d_large_f16 launch on the SAME stream reads it inside
    // mt_aiter_unified_attn() below; nothing references it afterwards. That
    // makes it exactly the kind of per-call temporary ctx.pool() is for
    // (unlike the persistent MT_AITER_PERSIST_* slots above, which exist
    // because THOSE buffers must keep a stable address across HIP graph
    // replays — see the MAD-288 comment on mt_aiter_persist_get). Declared
    // here (not deeper in this function) so the ggml_cuda_pool_alloc
    // destructors release the memory back to the pool right after this
    // function's mt_aiter_unified_attn() call returns, on every path
    // (including the size-cap-skip and non-predequant paths below, where
    // they simply stay unallocated).
    ggml_cuda_pool_alloc<uint8_t> predq_scratch_k(ctx.pool());
    ggml_cuda_pool_alloc<uint8_t> predq_scratch_v(ctx.pool());
    if (want_predequant_scratch && num_scratch_blocks > 0) {
        const size_t predq_bytes_per_cache =
            mt_aiter_predequant_scratch_bytes_per_cache(&shape, num_scratch_blocks);
        const size_t predq_cap = mt_aiter_predequant_max_bytes_per_cache();
        if (predq_cap > 0 && predq_bytes_per_cache > predq_cap) {
            // Mirrors the old ensure_predequant_scratch cap message: log once,
            // skip allocating (predq_scratch_k/v stay null), so
            // mt_aiter_unified_attn() falls back to the in-kernel fp8 dequant
            // path for this call instead of aborting.
            static std::atomic<bool> cap_logged{false};
            if (!cap_logged.exchange(true)) {
                std::fprintf(stderr,
                    "AITER paged-attn: fp8-predequant scratch would need %zu B/cache "
                    "(num_scratch_blocks=%d) > MT_AITER_PREDEQUANT_MAX_MB cap (%zu B) on "
                    "device %d -- skipping predequant for this call, falling back to the "
                    "in-kernel fp8 dequant path. (Logged once; this can repeat silently "
                    "for later calls.)\n",
                    predq_bytes_per_cache, num_scratch_blocks, predq_cap, dev);
            }
        } else {
            // Bucket the request to a power of two (>= 16 MiB). The legacy
            // CUDA pool never frees and only reuses a parked buffer that is
            // >= the request, so an exact-fit size that grows with the live
            // context would park one buffer per distinct size (measured:
            // 28 hipMallocs / 1.9 GB retained in 4 minutes as K/V requests
            // stepped 43,65 -> 47,70 -> 50,75 -> ... MiB). Power-of-two
            // buckets bound the distinct sizes to O(log n) and the retained
            // total to < 2x the largest request.
            size_t predq_bucket = (size_t) 16u << 20;
            while (predq_bucket < predq_bytes_per_cache) predq_bucket <<= 1;
            predq_scratch_k.alloc(predq_bucket);
            predq_scratch_v.alloc(predq_bucket);
        }
    }

    // ── MAD-227: optional Q pre-rotation for Hadamard-mode FP8 ──
    // Identity (QH)·(HK)^T = QK^T requires rotating BOTH Q and K. K is
    // rotated in the FP8 scatter kernel above (APPLY_HADAMARD=true variant);
    // Q is rotated here into a pool-allocated scratch. q->data is untouched.
    // Allocation only happens when both (a) registry says hadamard mode AND
    // (b) cache is turbo-FP8 — non-FP8 paths bypass entirely.
    const __half * q_ptr = (const __half *) q->data;
    __half * q_rot_ptr = nullptr;
    if (cache_type == MT_AITER_CACHE_TURBO4_FP8 && mt_turbo_fp8::hadamard_required()) {
        const size_t q_elts = (size_t) num_q_tokens * n_heads * head_size;
        // MAD-XXX diag (2026-09-10): num_q_tokens comes from k_cur->ne[2] but
        // n_heads/head_size come from q, and nothing asserts q actually holds
        // that many elements. An overread walks off the END of q, which lives in
        // the compute buffer -- far from the KV arena, matching the measured
        // fault address. Pure rare-path: compares two already-computed numbers
        // and prints ONLY on violation, so it has no hot-path cost and cannot
        // mask the fault the way per-call instrumentation did.
        {
            static std::atomic<bool> q_warned{false};
            const size_t q_have = (size_t) ggml_nbytes(q);
            const size_t q_want = q_elts * sizeof(__half);
            if (q_want > q_have && !q_warned.exchange(true)) {
                std::fprintf(stderr,
                    "[q-overread] dev=%d WANT %zu B from q but ggml_nbytes(q)=%zu B  "
                    "(OVERREAD %zu B)  q->ne=[%ld,%ld,%ld,%ld] k_cur->ne[2]=%d n_heads=%d hs=%d\n",
                    ggml_cuda_get_device(), q_want, q_have, q_want - q_have,
                    (long)q->ne[0],(long)q->ne[1],(long)q->ne[2],(long)q->ne[3],
                    num_q_tokens, n_heads, head_size);
            }
        }
        q_rot_ptr = mt_aiter_persist_get<__half>(dev, stream, MT_AITER_PERSIST_Q_ROT, q_elts);
        const hipError_t q_cpy = hipMemcpyAsync(q_rot_ptr, q->data,
                       q_elts * sizeof(__half), hipMemcpyDeviceToDevice, stream);
        if (q_cpy != hipSuccess) {
            std::fprintf(stderr, "[q-overread] dev=%d Q copy FAILED: %s\n",
                         ggml_cuda_get_device(), hipGetErrorString(q_cpy));
        }
        const hipError_t herr = mt_turbo_fp8_fwht_half(
            stream, q_rot_ptr,
            (int)(num_q_tokens * n_heads), head_size, head_size);
        if (herr != hipSuccess) {
            GGML_ABORT("mt_turbo_fp8_fwht_half(Q) launch failed: %s", hipGetErrorString(herr));
        }
        q_ptr = q_rot_ptr;
    }

    // ── 3. Launch AITER attention via the runtime wrapper ──
    mt_aiter_uattn_args_t args = {};
    args.shape        = shape;
    args.q            = (void *) q_ptr;
    args.k_cache      = k_cache->data;
    args.v_cache      = v_cache->data;
    args.out          = dst->data;
    args.segm_output  = use_2d ? nullptr : segm_out_ptr;
    args.segm_max     = use_2d ? nullptr : segm_max_ptr;
    args.segm_expsum  = use_2d ? nullptr : segm_exp_ptr;
    args.block_tables = (const int32_t*) block_tables->data;
    args.seq_lens     = (const int32_t*) context_lens->data;
    args.query_start_len = cu_seqlens_ptr;

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
    // MAD-LAB (draft-KV-paged num_seqs fix): dispatch-only real live count,
    // consumed ONLY by mt_aiter_unified_attn()'s own avg_q_len / 2D-vs-3D
    // predicate recompute (must agree with use_2d above -- see that
    // comment). Every other use of a->num_seqs in that file (grid dims,
    // ALL_DECODE, reduce-segments, indexing) intentionally keeps reading
    // args.num_seqs (static n_seq_max) unchanged.
    args.num_seqs_active    = (int32_t) num_seqs_dispatch;
    args.num_q_tokens       = num_q_tokens;
    args.block_table_stride = max_bps;
    // MAD-2026-09-12 predequant-scratch: NULL/0 for every call that isn't
    // turbo4_fp8 2D-large-prefill (the wrapper only reads these when
    // cache_type == TURBO4_FP8_BS256 && the 2D-large tile is selected).
    args.scratch_block_tables = num_scratch_blocks > 0 ? predq_scratch_table_ptr : nullptr;
    args.num_scratch_blocks   = num_scratch_blocks;
    // MAD-2026-09-20 predequant-pool: per-call pool scratch allocated above;
    // null whenever it wasn't requested (not turbo4_fp8/2D-large this call)
    // or was skipped over the MT_AITER_PREDEQUANT_MAX_MB cap — either way
    // mt_aiter_unified_attn() treats a null pointer as "take the in-kernel
    // fp8 dequant path instead," exactly like num_scratch_blocks == 0.
    args.predq_scratch_k      = predq_scratch_k.get();
    args.predq_scratch_v      = predq_scratch_v.get();
    args.q_stride_0         = (int64_t) n_heads * head_size;
    args.output_stride_0    = args.q_stride_0;
    args.k_stride_0         = (int64_t) block_size * n_kv_heads * head_size;
    args.k_stride_1         = (int64_t) n_kv_heads * head_size;
    args.k_stride_2         = head_size;
    args.v_stride_0         = args.k_stride_0;
    args.v_stride_1         = args.k_stride_1;
    args.v_stride_2         = args.k_stride_2;

    mt_aiter_sync_probe_pre("uattn", parse_layer_from_kv_cache_name(k_cache->name));
    const float mt_prof_prep = mt_prof.lap();
    hipError_t err = mt_aiter_unified_attn(stream, &args);
    if (mt_prof.on) {
        mt_op_prof_acc & pa = g_mt_op_prof[ctx.device & 15];
        pa.attn_ms += mt_prof.lap(); pa.scatter_ms += mt_prof_scatter; pa.prep_ms += mt_prof_prep; pa.n++;
        if (pa.n % 200 == 0) {
            fprintf(stderr, "mt_pagedattn-op-profile dev=%d n=%llu scatter=%.2fms/call prep=%.2fms/call unified_attn=%.2fms/call (q=%d kv_heads=%d seqs=%d)\n",
                    ctx.device, (unsigned long long) pa.n, pa.scatter_ms / pa.n, pa.prep_ms / pa.n, pa.attn_ms / pa.n,
                    (int) num_q_tokens, (int) n_kv_heads, (int) num_seqs);
            fflush(stderr);
        }
    }
    if (err != hipSuccess) {
        GGML_ABORT("mt_aiter_unified_attn launch failed: %s", hipGetErrorString(err));
    }
    mt_aiter_sync_probe(stream, "uattn", parse_layer_from_kv_cache_name(k_cache->name));
}

}  // namespace mt

#endif  // GGML_HIP_AITER
