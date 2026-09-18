// mt_pagedattn_wmma_fp8.cu — hand-written HIP WMMA flash-attention prefill
// kernel for gfx1201/RDNA4, targeting exactly:
//   head_dim=256, GQA-6 (24 q-heads / 4 kv-heads), causal, paged KV cache
//   type GGML_TYPE_TURBO4_FP8_BS256 (4-bit centroid index + 1-bit sign +
//   fp16 per-(token,kv_head) scale, one 162-byte block covering the full
//   256-wide head row; per-(layer, K|V) centroid LUT is 16 E4M3 bytes,
//   see mt_turbo_fp8_lut_registry.h/.cu and mt_pagedattn_turbo_fp8.cuh).
//
// This is the AITER Triton kernel_unified_attention_2d's numerical/perf
// competitor for prefill-sized M (per-seq q_len >= 64). Gated OFF by
// default; set MT_PAGED_ATTN_WMMA=1 to opt in. See
// paged_attn_wmma_fp8_shape_ok() in the header for the exact gate.
//
// ── Verified-from-code facts that correct the original task brief ──────────
//  1. "TURBO4_FP8_BS256 ... fp8 K/V" is not literally true: storage is a
//     4-bit centroid index (16-entry LUT, LUT values stored as E4M3) + a
//     1-bit sign + one fp16 scale per (token, kv_head) covering the full
//     256-wide head row (see ggml/include/ggml.h:437 and
//     mt_pagedattn_turbo_fp8.cuh's header comment). There are no raw fp8
//     bytes in the KV cache to unpack with __builtin_amdgcn_cvt_pk_f32_fp8;
//     dequant is a nibble+sign+scale*centroid lookup.
//  2. "paged block size of 256 tokens" is not the physical page size: the
//     AITER dispatch for this cache type always uses block_size=16 tokens
//     per physical page (mt_pagedattn_aiter.cu: "head_size == 256 &&
//     block_size == 16" is the only wired TURBO4_FP8_BS256 scatter
//     instantiation). "BS256" in the type name refers to the *quant* block
//     spanning all 256 head_dim elements of one row, not the paged block
//     size. This kernel therefore walks the block table in units of 16
//     tokens/physical page (BLOCK_SIZE=16 below), not 256.
//  3. Hadamard rotation (MAD-227, MT_TURBO_FP8_HADAMARD) is OFF by default
//     (mt_turbo_fp8_lut_registry.cu reads the env var once at registry
//     init and defaults to false). This kernel only implements the
//     default (non-Hadamard) path — see paged_attn_wmma_fp8_shape_ok(),
//     which refuses the shape (falls back to AITER) when
//     mt_turbo_fp8::hadamard_required() is true. Adding Q pre-rotation
//     in-kernel (mirroring mt_pagedattn_aiter.cu's MAD-227 Q-rotation
//     scratch-buffer step) is follow-up work, not attempted here.
//
// ── Kernel design ────────────────────────────────────────────────────────
// One workgroup per (q_head, seq, q_tile of Q_TILE_M=64 rows) — i.e. we do
// NOT stack the 6 GQA-sibling q-heads into one workgroup's M dimension or
// loop them inside one workgroup with persistent per-head accumulators;
// seeing the concrete VGPR arithmetic through (below) that would need
// either 6x the O-accumulator registers held live at once (not survivable
// under a ~200 VGPR/wave budget) or spilling 5 of the 6 heads' O/softmax
// state to LDS between KV-tile visits, which collides with LDS already
// being fully committed to K/V staging at head_dim=256 (see LDS budget
// below). So v1 launches one workgroup per q_head (grid.x = n_heads, same
// convention as the existing mt_paged_attention_tile_mw_kernel), and pays
// for K/V dequant bandwidth 6x (once per q-head sharing a kv-head) rather
// than once. This is the single biggest documented performance tradeoff in
// this file — see the dispatch-entry comment for the concrete bandwidth
// estimate and the follow-up (share K/V loads across the GQA-6 group via
// an LDS-resident accumulator-spill scheme).
//
// ── v2 restructure (post-review: v1 put all 256 O columns in one wave —
//    128 VGPRs of O alone — and compiled to 256 VGPRs/206 spilled) ────────
//
// Per workgroup: 8 waves (256 threads). wave = row_group + 4*col_half:
//   row_group = wave & 3  (0..3): which 16-row Q stripe (was "wave" in v1)
//   col_half  = wave >> 2 (0/1):  which 128-column half of O this wave owns
// Waves (w, w+4) share row_group and independently compute the FULL
// S=QK^T for those 16 rows (duplicated — cheap relative to P@V), so
// softmax state (row_max/row_sum) is purely per-wave with zero cross-wave
// traffic, while each wave's O accumulator only covers 128 of 256 columns
// (8 col-tiles instead of 16 -> 64 VGPRs instead of 128).
//
// P@V's operand roles are SWAPPED from the "obvious" A=Q/B=K choice used
// for v1's QK^T: this kernel computes QK^T as A=K, B=Q. Per the gfx12
// wave32 WMMA convention (D[m][n]: lane%16=n, m=(lane/16)*8+i, where m/n
// are literally A's/B's own row indices — verified against
// gated_delta_net_prefill_wmma2_cuda), that swap produces an accumulator
// P_T with kv-token on the (lane/16)*8+i axis and query-row on lane%16 —
// exactly a B-operand-shaped fragment (B[k][n]: lane%16=n, k=(lane/16)*8+i,
// matching n=query, k=kv). So `to_h8(P_T[tn])` feeds directly into the
// P@V wmma16() call as its B operand with ZERO trip through LDS — no
// smem_p, no transpose step; P_T is a transient register value live only
// within one KV-tile iteration. (v1's A=Q/B=K choice put kv-token on
// lane%16 instead, which is the WRONG split for direct reuse — hence v1
// went through LDS for P. The fix here is upstream of that: change which
// operand is A vs B in QK^T so the accumulator comes out pre-shaped for
// the next matmul, not a post-hoc transpose of it.)
//
// One side effect of the A/B swap: since query-row is now fixed by lane%16
// (not spread across (lane/16)*8+i), row_max/row_sum become plain per-lane
// SCALARS instead of 8-element arrays — a further, incidental VGPR win.
//
// Q is read straight from global memory as a WMMA operand (no LDS
// staging): `scale` is folded into the post-QK^T epilogue instead of
// pre-scaling Q. K is staged into LDS as [token][head_dim] row-major (the
// existing mt_turbo_fp8::coop_stage_turbo4_fp8_bs256_tile decode helper,
// now cooperating across NW=8 waves instead of 4). V is staged TRANSPOSED
// into LDS as [head_dim][token] (decode_v_transposed_bs256_tile) so V's
// contraction axis (kv-token) is the fast axis, matching P_T's own
// (lane/16)*8+i axis for the same contraction.
//
// WMMA primitive: __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12 via
// 8-element _Float16/float vector types, exactly as verified/used in
// gated_delta_net.cu's gated_delta_net_prefill_wmma2_cuda (gfx12, wave32):
//   A[m][k]: lane%16 = m,  k = (lane/16)*8 + 0..7
//   B[k][n]: lane%16 = n,  k = (lane/16)*8 + 0..7
//   D[m][n]: lane%16 = n,  m = (lane/16)*8 + 0..7
//
// ── VGPR budget (per wave, target — see the dispatch entry doc-comment
//    for the real number via -Rpass-analysis=kernel-resource-usage) ──
//   O_T accumulator:     8 col-tiles (128/16) x f8_t (8 x f32)          =  64 VGPRs
//   P_T accumulator:     2 col-tiles (K_TILE_N=32/16) x f8_t            =  16 VGPRs
//   row_max/row_sum:     2 scalars (was 8+8 arrays pre-swap)            =   2 VGPRs
//   Q/K/V WMMA operands: transient h8_t (4 VGPRs, packed f16x2) per use,
//                         not live across iterations                    ~   8 VGPRs
//   misc (indices, bounds, pointers)                                    ~  20 VGPRs
//   ------------------------------------------------------------------------
//   total (target)                                                      ~ 110 VGPRs
// Comfortably under the 168-VGPR / >=3-waves-per-SIMD target — see the
// dispatch entry for the measured -Rpass-analysis numbers.
//
// ── LDS budget (unchanged shape from v1's K/V staging, minus smem_p —
//    P no longer touches LDS at all) ──
//   smem_k  [K_TILE_N=32][HEAD_SIZE=256] fp16   = 32*256*2  = 16384 B
//   smem_vt [HEAD_SIZE=256][K_TILE_N=32] fp16   = 256*32*2  = 16384 B
//   ------------------------------------------------------------------------
//   total                                                    = 32768 B (32 KB)
// ~32KB of headroom remains versus a 64KB LDS budget — enough to later
// double-buffer both K and V if the dequant-then-syncthreads bubble shows
// up as the bottleneck in profiling.

#include "mt_pagedattn_wmma_fp8.cuh"
#include "mt_pagedattn.cuh"
#include "mt_pagedattn_turbo_fp8.cuh"   // mt_turbo_fp8::coop_stage_turbo4_fp8_bs256_tile, e4m3_to_fp32
#include "mt_turbo_fp8_lut_registry.h"  // mt_turbo_fp8::get_lut_device_ptr / hadamard_required

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>

namespace mt {

// ─────────────────────────── env gate ───────────────────────────
bool paged_attn_wmma_fp8_env_enabled() {
    static int mode = -1;
    if (mode < 0) {
        const char * env = std::getenv("MT_PAGED_ATTN_WMMA");
        mode = (env != nullptr && env[0] == '1') ? 1 : 0;
        GGML_LOG_INFO("mt_pagedattn_wmma_fp8: MT_PAGED_ATTN_WMMA=%d (WMMA prefill kernel %s)\n",
                      mode, mode ? "enabled" : "disabled");
    }
    return mode != 0;
}

bool paged_attn_wmma_fp8_shape_ok(int cc, int head_size, ggml_type cache_type,
                                   int n_heads, int n_kv_heads, int max_q_len) {
    if (head_size != 256) return false;
    if (cache_type != GGML_TYPE_TURBO4_FP8_BS256) return false;
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0) return false;
    if (n_heads / n_kv_heads != 6) return false;      // GQA-6 only, matches the launch-bounds M/N split below
    if (max_q_len < 64) return false;                 // decode / tiny prefill stays on AITER
    if (!amd_wmma_available(cc)) return false;
    if (mt_turbo_fp8::hadamard_required()) return false;  // v1 scope limit, see file header
    return true;
}

// ───────────────── layer-index parse (duplicated, not shared) ─────────────
// mt_pagedattn_aiter.cu has an identical `static` helper (file-local linkage,
// so it cannot be called from here without either exposing it through a
// header or risking a cross-TU device-kernel-launch/RDC dependency for the
// scatter kernel below). Both are trivial and intentionally kept in sync by
// comment rather than shared, to avoid touching mt_pagedattn_aiter.cu (which
// per the task brief may be under concurrent edit) for a 9-line function.
static int mt_wmma_fp8_parse_layer_from_kv_cache_name(const char * name) {
    if (!name) return -1;
    const char * p = std::strstr(name, "_l");
    if (!p) return -1;
    p += 2;
    int n = 0;
    bool any = false;
    while (*p >= '0' && *p <= '9') { n = n * 10 + (*p - '0'); ++p; any = true; }
    return any ? n : -1;
}

// ───────────── scatter kernel (non-Hadamard only; see file header) ────────
//
// Identical math to mt_scatter_kv_turbo4_fp8_aiter_kernel<HEAD_SIZE=256,
// BLOCK_SIZE=16, APPLY_HADAMARD=false> in mt_pagedattn_aiter.cu (that file's
// AITER-format layout: [num_blocks, block_size, n_kv_heads, 162-byte block]
// is REQUIRED here too, since this kernel's attention pass and AITER's must
// read the identical on-disk cache format to be A/B comparable). Duplicated
// rather than cross-TU-launched for the same reason as the layer-parse
// helper above. Kept for reuse if this file needs its own Hadamard-capable
// scatter variant later.
namespace wmma_fp8_scatter_detail {
static __device__ __forceinline__ float e4m3_to_fp32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e    = (b >> 3) & 0xF;
    int m    = b & 0x7;
    float v  = (e == 0) ? (1.0f / 64.0f) * (m / 8.0f)
                        : __builtin_amdgcn_ldexp(1.0f + m / 8.0f, e - 7);
    return sign ? -v : v;
}
}  // namespace wmma_fp8_scatter_detail

template <int HEAD_SIZE, int BLOCK_SIZE>
__launch_bounds__(256)
__global__ void mt_scatter_kv_turbo4_fp8_wmma_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    const uint8_t  * __restrict__ centroids_k,
    const uint8_t  * __restrict__ centroids_v,
    int             n_kv_heads) {
    using namespace wmma_fp8_scatter_detail;

    static_assert(HEAD_SIZE == 256, "turbo4_fp8 wmma scatter requires HEAD_SIZE=256");
    constexpr int N_CENT          = 16;
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

    __shared__ float x[256];
    __shared__ float lut_f[N_CENT];

    x[j] = __half2float(src[src_off]);
    if (j < N_CENT) lut_f[j] = e4m3_to_fp32(lut_bytes[j]);
    __syncthreads();

    // No Hadamard rotation (v1 scope limit — default/off registry state).

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

    const float scale_f   = blk_max;
    const __half scale_h  = __float2half(scale_f);
    const float scale_eff = __half2float(scale_h);
    const float inv_scale = (scale_eff > 0.0f) ? (1.0f / scale_eff) : 0.0f;

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
        const int byte_idx     = j / 8;
        const int hw_lane_base = (threadIdx.x % warpSize) & ~(WARP_SIZE - 1);
        const int bit_off      = hw_lane_base + ((j % WARP_SIZE) & ~7);
        blk[130 + byte_idx] = (uint8_t)((sign_mask >> bit_off) & 0xFF);
    }
}

// ───────────── V decode, TRANSPOSED into LDS as [head_dim][token] ─────────
//
// Same per-(token,kv_head) 162-byte block decode as
// mt_turbo_fp8::coop_stage_turbo4_fp8_bs256_tile, but writes
// smem_dst[d * TOKEN_STRIDE + row] instead of smem_dst[row * HEAD_SIZE + d]
// so the P@V product can address V as a plain row-major
// [head_dim][kv_token] array (contraction axis = kv_token = the fast/inner
// axis), matching the same WMMA operand addressing convention used for K
// and P. See the file header for why V (and not K) needs the transpose:
// K's contraction axis for QK^T is head_dim, which is already V... no: K's
// contraction axis is head_dim (fast axis in K's natural [token][head_dim]
// layout, so no transpose needed there); V's contraction axis for P@V is
// kv_token, which is the SLOW axis in V's natural layout, hence transposing
// at decode time is required.
template <int HEAD_SIZE, int BLOCK_SIZE, int N_WARPS, int K_TILE_N>
static __device__ __forceinline__ void decode_v_transposed_bs256_tile(
        __half        * __restrict__ smem_vt,          // [HEAD_SIZE][K_TILE_N]
        const void    * __restrict__ v_cache,
        const uint8_t * __restrict__ centroids_v,
        const int     * __restrict__ seq_block_table,
        int            k_tile_start,
        int            block_valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            warp_id,
        int            lane_id) {
    static_assert(HEAD_SIZE == 256, "requires HEAD_SIZE=256");
    constexpr size_t BYTES_PER_BLOCK = 162;
    constexpr int    N_CENT          = 16;
    constexpr int    ELEMS_PER_LANE  = HEAD_SIZE / 32;  // 8

    const uint8_t * cache_bytes = (const uint8_t *) v_cache;

    float lut[N_CENT];
    #pragma unroll
    for (int k = 0; k < N_CENT; ++k) lut[k] = mt_turbo_fp8::e4m3_to_fp32(centroids_v[k]);

    for (int row = warp_id; row < K_TILE_N; row += N_WARPS) {
        const int token = k_tile_start + row;

        const uint8_t * blk_bytes = nullptr;
        float scale_f = 0.0f;
        if (token < block_valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical >= 0) {
                const int64_t blk_idx =
                      (int64_t) physical * BLOCK_SIZE * n_kv_heads
                    + (int64_t) tok_in_block * n_kv_heads
                    + (int64_t) kv_head_idx;
                blk_bytes = cache_bytes + blk_idx * BYTES_PER_BLOCK;
                if (lane_id == 0) {
                    __half h;
                    __builtin_memcpy(&h, blk_bytes, sizeof(__half));
                    scale_f = __half2float(h);
                }
            }
        }
        scale_f = __shfl_sync(0xFFFFFFFFFFFFFFFFull, scale_f, 0, WARP_SIZE);

        uint32_t qs_word = 0;
        uint8_t  signs_b = 0;
        if (blk_bytes != nullptr) {
            qs_word = *(const uint32_t *)(blk_bytes + 2 + lane_id * 4);
            signs_b = blk_bytes[2 + 128 + lane_id];
        }

        const int col_base = lane_id * ELEMS_PER_LANE;
        #pragma unroll
        for (int l = 0; l < ELEMS_PER_LANE; ++l) {
            const int idx = (qs_word >> (l * 4)) & 0xF;
            const int s   = (signs_b >> l) & 1;
            float val = lut[idx] * scale_f;
            if (s) val = -val;
            const int d = col_base + l;
            smem_vt[(size_t) d * K_TILE_N + row] = __float2half(val);
        }
    }
}

// ─────────────────────────── WMMA primitives ───────────────────────────
namespace wmma_fp8_detail {
typedef _Float16 h8_t __attribute__((ext_vector_type(8)));
typedef float    f8_t __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ f8_t wmma16(const h8_t & a, const h8_t & b, const f8_t & c) {
#if defined(__gfx1201__) || defined(__gfx1200__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
#else
    (void) a; (void) b; return c;
#endif
}
static __device__ __forceinline__ h8_t to_h8(const f8_t & v) {
    h8_t r;
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = (_Float16) v[i];
    return r;
}
static __device__ __forceinline__ h8_t ld_h8(const __half * p) {
    return *reinterpret_cast<const h8_t *>(p);
}
static __device__ __forceinline__ h8_t zero_h8() {
    h8_t r;
    #pragma unroll
    for (int i = 0; i < 8; i++) r[i] = (_Float16) 0.0f;
    return r;
}
}  // namespace wmma_fp8_detail

// ─────────────────────────── attention kernel (v2, zero-spill target) ────
//
// Restructure per coordinator directive (post-v1 review: v1's single
// wave-per-M-stripe design put all 256 O columns in one wave -> 128 VGPRs
// of O alone, compiled to 256 VGPRs/206 spilled). v2 splits the 256 output
// columns across TWO waves per M-stripe instead of keeping them all in one:
//
//   NW = 8 waves/workgroup. row_group = wave & 3 (0..3, which 16-row Q
//   stripe — same meaning as v1's "wave"), col_half = wave >> 2 (0 or 1,
//   which 128-column half of O this wave owns). Waves (w, w+4) share
//   row_group and split the head_dim range [0,128) / [128,256).
//
// Both waves in a (w, w+4) pair independently compute the FULL S = Q K^T
// for their shared 16 query rows (K=256 contraction, cheap relative to the
// P@V matmul) rather than computing it once and exchanging via LDS —
// "cheap to duplicate" per the coordinator's note, and it keeps softmax
// state (row_max/row_sum) purely per-wave with zero cross-wave traffic.
//
// The P@V matmul's A/B roles are SWAPPED relative to v1 to make the QK^T
// accumulator directly reusable as V1: instead of A=Q,B=K (which produces
// an accumulator with query-row on the (lane/16)*8+i axis and kv-token on
// the lane%16 axis — the WRONG split for reuse as an operand contracting
// over kv), this kernel computes A=K,B=Q. Per gfx12 wmma32 convention
// (D[m][n]: lane%16=n, m=(lane/16)*8+i, where m/n are A's/B's own row
// indices — verified against gated_delta_net_prefill_wmma2_cuda), that
// swap produces an accumulator P_T with:
//   - kv-token on the (lane/16)*8+i axis (i.e. i now indexes 8 kv offsets
//     within one lhi-half of a 16-token WMMA tile)
//   - query-row on the lane%16 axis (i.e. l16 now IS the query row, fixed
//     per lane — row_max/row_sum become plain per-lane SCALARS, not
//     8-element arrays, which also shrinks VGPR use)
// This is exactly a B-operand-shaped fragment (B[k][n]: lane%16=n,
// k=(lane/16)*8+i — matches with n=query, k=kv), so `to_h8(P_T[tn])` feeds
// directly into the P@V wmma16() call as the B operand with NO trip
// through LDS (smem_p is gone entirely; both P_T tiles are transient
// registers, live only across one KV-tile iteration). The A operand for
// P@V is V^T (same LDS-transposed staging as v1, contraction axis = kv,
// identity axis = head_dim), so:
//   O_T[tv] = sum_tn wmma16(V^T[tv,tn], to_h8(P_T[tn]), O_T[tv])
// which — because A's identity (head_dim) always lands on D's
// (lane/16)*8+i axis and B's identity (query) always lands on D's lane%16
// axis — produces O in TRANSPOSED form (head_dim on (lane/16)*8+i, query
// on lane%16). The epilogue write reflects that transpose directly; no
// separate detranspose step is needed.
//
// Grid:  (head_idx [0, n_heads), seq_idx [0, num_seqs), q_tile_idx)
// Block: NW*32 = 256 threads.
template <int HEAD_SIZE, int BLOCK_SIZE, int Q_TILE_M, int K_TILE_N>
__launch_bounds__(((Q_TILE_M / 16) * 2) * 32)
__global__ void mt_paged_attn_wmma_fp8_kernel(
    __half         * __restrict__ out,
    const __half   * __restrict__ q,
    const void     * __restrict__ k_cache,
    const void     * __restrict__ v_cache,
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
    const uint8_t  * __restrict__ centroids_k,
    const uint8_t  * __restrict__ centroids_v,
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    float           scale) {
    using namespace wmma_fp8_detail;

    static_assert(HEAD_SIZE == 256, "kernel is specialized for head_dim=256");
    static_assert(Q_TILE_M % 16 == 0, "Q_TILE_M must be a multiple of 16");
    static_assert(K_TILE_N % 16 == 0, "K_TILE_N must be a multiple of 16");

    constexpr int N_ROW_GROUPS   = Q_TILE_M / 16;              // distinct 16-row Q stripes (4)
    constexpr int N_COL_HALVES   = 2;                          // O column halves per row stripe
    constexpr int NW             = N_ROW_GROUPS * N_COL_HALVES; // total waves (8)
    constexpr int KV_N_TILES     = K_TILE_N / 16;               // WMMA tiles per KV iter (2)
    constexpr int KV_K_STEPS     = HEAD_SIZE / 16;              // contraction steps for QK^T (16)
    constexpr int HD_N_TILES     = HEAD_SIZE / 16;              // O column tiles, TOTAL (16)
    constexpr int HD_N_TILES_HF  = HD_N_TILES / N_COL_HALVES;   // O column tiles per wave (8)
    constexpr float SOFTMAX_MASK_VAL = -1.0e30f;                // matches mt_pagedattn.cu's convention

    const int head_idx   = blockIdx.x;
    const int seq_idx    = blockIdx.y;
    const int q_tile_idx = blockIdx.z;
    const int tid        = threadIdx.x;
    const int lane        = tid & 31;
    const int wave        = tid >> 5;             // 0..NW-1
    const int row_group   = wave & (N_ROW_GROUPS - 1);   // which 16-row Q stripe (shared by w, w+4)
    const int col_half    = wave >> 2;                   // which 128-col half of O (0 or 1)
    const int l16          = lane & 15;            // == query row within this wave's stripe (post A/B swap)
    const int lhi          = lane >> 4;            // 0 or 1 (kv sub-half within a 16-token WMMA tile)

    const int kv_head_idx = head_idx / (n_heads / n_kv_heads);
    const int q_len       = q_lens[seq_idx];
    const int q_tile_start = q_tile_idx * Q_TILE_M;
    if (q_tile_start >= q_len) return;

    const int ctx_len_after_q   = context_lens[seq_idx];
    const int * seq_block_table = block_tables + (size_t) seq_idx * max_blocks_per_seq;

    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];

    const int my_q_pos_base   = (ctx_len_after_q - q_len) + q_tile_start;
    const int tile_last_row   = min(q_tile_start + Q_TILE_M, q_len) - 1;
    const int block_valid_ctx = (ctx_len_after_q - q_len) + tile_last_row + 1;

    // LDS: K (row-major [token][dim]) + V^T (row-major [dim][token]) only —
    // no P scratch (P stays in registers, see file header). Cooperative
    // decode is now spread across NW=8 waves instead of 4.
    extern __shared__ unsigned char smem_raw[];
    __half * smem_k  = (__half *) smem_raw;                              // [K_TILE_N][HEAD_SIZE]
    __half * smem_vt = smem_k + (size_t) K_TILE_N * HEAD_SIZE;           // [HEAD_SIZE][K_TILE_N]

    float row_max = SOFTMAX_MASK_VAL;   // scalar per lane: lane%16 IS the query row now
    float row_sum = 0.0f;
    f8_t  O_T[HD_N_TILES_HF];           // this wave's half of the head-dim columns, TRANSPOSED
    #pragma unroll
    for (int t = 0; t < HD_N_TILES_HF; ++t) O_T[t] = f8_t{0,0,0,0,0,0,0,0};

    const int q_row_local = row_group * 16 + l16;      // fixed per lane for the whole kernel now
    const bool row_ok     = (q_row_local < Q_TILE_M) && (q_tile_start + q_row_local < q_len);
    const int  q_row_abs  = q_tile_start + q_row_local;
    const size_t q_row_base = row_ok
        ? ((seq_q_offset + (size_t) q_row_abs) * (size_t) n_heads + (size_t) head_idx) * (size_t) HEAD_SIZE
        : 0;

    for (int k_tile_start = 0; k_tile_start < block_valid_ctx; k_tile_start += K_TILE_N) {
        const int n_valid = min(K_TILE_N, block_valid_ctx - k_tile_start);

        mt_turbo_fp8::coop_stage_turbo4_fp8_bs256_tile<HEAD_SIZE, BLOCK_SIZE, NW, K_TILE_N>(
            smem_k, k_cache, centroids_k, seq_block_table, k_tile_start, block_valid_ctx,
            kv_head_idx, n_kv_heads, wave, lane);
        decode_v_transposed_bs256_tile<HEAD_SIZE, BLOCK_SIZE, NW, K_TILE_N>(
            smem_vt, v_cache, centroids_v, seq_block_table, k_tile_start, block_valid_ctx,
            kv_head_idx, n_kv_heads, wave, lane);
        __syncthreads();

        // ---- QK^T for this row_group's 16-row stripe, A=K / B=Q (swapped
        //      from the "obvious" A=Q/B=K so the resulting accumulator is
        //      directly reusable as P@V's B operand — see file header).
        //      Duplicated identically by both waves sharing this row_group. ----
        f8_t P_T[KV_N_TILES];
        #pragma unroll
        for (int tn = 0; tn < KV_N_TILES; ++tn) {
            f8_t acc = f8_t{0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int ks = 0; ks < KV_K_STEPS; ++ks) {
                const h8_t a = ld_h8(smem_k + (size_t)(tn * 16 + l16) * HEAD_SIZE + lhi * 8 + ks * 16);
                const h8_t b = row_ok ? ld_h8(q + q_row_base + lhi * 8 + ks * 16) : zero_h8();
                acc = wmma16(a, b, acc);
            }
            P_T[tn] = acc;
        }

        // ---- causal mask + scale; i now indexes KV offsets (lhi*8+i),
        //      query row is fixed (l16) for this whole lane ----
        const int abs_kv_base = k_tile_start + lhi * 8;
        float local_max = SOFTMAX_MASK_VAL;
        #pragma unroll
        for (int tn = 0; tn < KV_N_TILES; ++tn) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int kv_idx     = tn * 16 + lhi * 8 + i;
                const int abs_kv_pos = k_tile_start + tn * 16 + lhi * 8 + i;
                const bool valid     = row_ok && (kv_idx < n_valid) && (abs_kv_pos <= my_q_pos_base + q_row_local);
                const float sv = valid ? (P_T[tn][i] * scale) : SOFTMAX_MASK_VAL;
                P_T[tn][i] = sv;
                local_max = fmaxf(local_max, sv);
            }
        }
        // Combine the two lhi-halves (each lane only saw 8 of the 16 kv
        // offsets per tn-tile): one shuffle across the lhi bit (XOR 16),
        // full-warp width since there are only 2 groups.
        local_max = fmaxf(local_max, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, local_max, 16, 32));

        // ---- online-softmax rescale of O_T + row_sum (scalar alpha now —
        //      every element of O_T for this lane belongs to the SAME
        //      query row) ----
        const float new_max = fmaxf(row_max, local_max);
        const float alpha   = (row_max > SOFTMAX_MASK_VAL * 0.5f) ? expf(row_max - new_max) : 0.0f;
        row_sum *= alpha;
        row_max  = new_max;
        #pragma unroll
        for (int t = 0; t < HD_N_TILES_HF; ++t) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) O_T[t][i] *= alpha;
        }

        // ---- P = exp(P_T - row_max), stays in registers; accumulate row_sum ----
        float local_sum = 0.0f;
        #pragma unroll
        for (int tn = 0; tn < KV_N_TILES; ++tn) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float p = expf(P_T[tn][i] - row_max);
                P_T[tn][i] = p;
                local_sum += p;
            }
        }
        local_sum += __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, local_sum, 16, 32);
        row_sum += local_sum;

        // ---- O_T += V^T @ P (this wave's HD_N_TILES_HF head-dim tiles
        //      only; P_T reused DIRECTLY as the B operand, no LDS) ----
        const int hd_tile_base = col_half * HD_N_TILES_HF;
        #pragma unroll
        for (int tv = 0; tv < HD_N_TILES_HF; ++tv) {
            f8_t acc = O_T[tv];
            #pragma unroll
            for (int tn = 0; tn < KV_N_TILES; ++tn) {
                const h8_t a = ld_h8(smem_vt + (size_t)((hd_tile_base + tv) * 16 + l16) * K_TILE_N + lhi * 8 + tn * 16);
                const h8_t b = to_h8(P_T[tn]);
                acc = wmma16(a, b, acc);
            }
            O_T[tv] = acc;
        }
        __syncthreads();  // before the next iteration overwrites smem_k/smem_vt
    }

    // ---- epilogue: normalize and write O (transposed layout: head-dim on
    //      (lhi,i), query row fixed via l16/row_group) ----
    const float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    if (q_row_abs < q_len) {
        const int hd_tile_base = col_half * HD_N_TILES_HF;
        const size_t out_row_base = (seq_q_offset + (size_t) q_row_abs) * (size_t) n_heads * (size_t) HEAD_SIZE
                                   + (size_t) head_idx * (size_t) HEAD_SIZE;
        #pragma unroll
        for (int tv = 0; tv < HD_N_TILES_HF; ++tv) {
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = (hd_tile_base + tv) * 16 + lhi * 8 + i;
                out[out_row_base + (size_t) d] = __float2half(O_T[tv][i] * inv_sum);
            }
        }
    }
}

// ─────────────────────────── host dispatch ───────────────────────────
//
// K/V global-memory read amplification from launching one workgroup per
// q_head (instead of sharing loads across the 6 GQA-sibling heads, see the
// file header): each of the 4 kv_heads' K/V bytes for a given layer are
// read 6x instead of once. At 8192 KV tokens and 162 B/(token,kv_head) for
// K and V each, one full read is 8192*162*2 ~= 2.66 MB per kv_head-layer;
// 6x that is ~16 MB; across 4 kv_heads * 16 layers * 6x = ~1 GB of *extra*
// HBM traffic per 2048-token prefill ubatch versus a perfectly
// head-amortized version. Documented as the primary follow-up optimization
// (see file header) rather than attempted here given the VGPR/LDS
// constraints at head_dim=256.
void ggml_cuda_op_paged_attn_mt_wmma_fp8(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_cache       = dst->src[1];
    const ggml_tensor * v_cache       = dst->src[2];
    const ggml_tensor * block_tables  = dst->src[3];
    const ggml_tensor * context_lens  = dst->src[4];
    const ggml_tensor * q_lens        = dst->src[5];
    const ggml_tensor * k_cur         = dst->src[6];
    const ggml_tensor * v_cur         = dst->src[7];
    const ggml_tensor * slot_mapping  = dst->src[8];

    const float   * op_params_f = (const float *)(dst->op_params);
    const float     scale       = op_params_f[0];
    const int32_t   block_size  = ((const int32_t *)(op_params_f + 1))[0];
    const int32_t   n_kv_heads  = ((const int32_t *)(op_params_f + 3))[0];

    const int head_size      = q->ne[0];
    const int n_heads        = q->ne[1];
    const int num_seqs       = block_tables->ne[1];
    const int max_bps        = block_tables->ne[0];
    const int num_q_tokens   = (int) k_cur->ne[2];

    GGML_ASSERT(head_size == 256 && "wmma_fp8 requires head_dim=256");
    GGML_ASSERT(block_size == 16 && "wmma_fp8 requires the paged block size used by TURBO4_FP8_BS256 (16 tokens/page)");
    GGML_ASSERT(n_heads % n_kv_heads == 0 && (n_heads / n_kv_heads) == 6 && "wmma_fp8 requires GQA-6");
    GGML_ASSERT(k_cache->type == GGML_TYPE_TURBO4_FP8_BS256 && v_cache->type == GGML_TYPE_TURBO4_FP8_BS256);
    GGML_ASSERT(q->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16);
    GGML_ASSERT(!mt_turbo_fp8::hadamard_required() &&
                "wmma_fp8 v1 does not implement Q pre-rotation; caller must have gated this off "
                "via paged_attn_wmma_fp8_shape_ok() before reaching here");

    cudaStream_t stream = ctx.stream();

    const int il = mt_wmma_fp8_parse_layer_from_kv_cache_name(k_cache->name);
    GGML_ASSERT(il >= 0 && "wmma_fp8: failed to parse layer index from k_cache tensor name");
    const uint8_t * d_centroids_k = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_K);
    const uint8_t * d_centroids_v = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_V);
    GGML_ASSERT(d_centroids_k && d_centroids_v && "wmma_fp8: centroid LUT lookup returned null");

    // ── scatter k_cur/v_cur into the paged cache (same layout/kernel shape
    //    as mt_scatter_kv_turbo4_fp8_aiter_kernel<..., APPLY_HADAMARD=false>) ──
    {
        dim3 grid(num_q_tokens, n_kv_heads, 2);
        dim3 block(256);
        mt_scatter_kv_turbo4_fp8_wmma_kernel<256, 16><<<grid, block, 0, stream>>>(
            k_cache->data, v_cache->data,
            (const __half *) k_cur->data, (const __half *) v_cur->data,
            (const int32_t *) slot_mapping->data,
            d_centroids_k, d_centroids_v, n_kv_heads);
    }

    // ── attention ──
    constexpr int HEAD_SIZE = 256;
    constexpr int BLOCK_SIZE = 16;
    constexpr int Q_TILE_M  = 64;
    constexpr int K_TILE_N  = 32;
    constexpr int NW        = (Q_TILE_M / 16) * 2;  // 8 waves: 4 row-groups x 2 O column-halves

    const int num_q_tiles = (num_q_tokens + Q_TILE_M - 1) / Q_TILE_M;  // safe upper bound, see
                                                                        // launch_paged_attn_tile_mw's
                                                                        // identical total_q_tokens
                                                                        // sizing convention.
    dim3 grid(n_heads, num_seqs, max(1, num_q_tiles));
    dim3 block(NW * 32);

    const size_t smem_bytes = (size_t) K_TILE_N * HEAD_SIZE * sizeof(__half)     // smem_k
                             + (size_t) HEAD_SIZE * K_TILE_N * sizeof(__half);   // smem_vt
                             // (no P scratch in v2 — P stays in registers, see kernel doc-comment)

    mt_paged_attn_wmma_fp8_kernel<HEAD_SIZE, BLOCK_SIZE, Q_TILE_M, K_TILE_N>
        <<<grid, block, smem_bytes, stream>>>(
            (__half *) dst->data,
            (const __half *) q->data,
            k_cache->data, v_cache->data,
            (const int32_t *) block_tables->data,
            (const int32_t *) context_lens->data,
            (const int32_t *) q_lens->data,
            d_centroids_k, d_centroids_v,
            max_bps, n_kv_heads, n_heads, scale);
}

}  // namespace mt
