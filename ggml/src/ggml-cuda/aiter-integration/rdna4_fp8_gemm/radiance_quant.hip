// radiance_quant.hip — fused ml8-4 activation epilogue kernels. See
// radiance_quant.h for the API contract (rot_kind/a_dim/b_dim, the TILED
// layout, and why the older radiance_add_rms_quant/silu_mul_quant/
// gdn_norm_quant ports are gone from that header -- they never applied to
// this repo's rotated ml8-4 activation path and nothing called them).
//
// WHAT THIS FILE ACTUALLY DOES: ggml-cuda/ml8.cu owns GGML_OP_FP8_QUANT_ROT's
// reference implementation (ml8_fp8_qrot_v3_kernel: Kronecker H_a-mix or
// block-Hadamard rotation of each fp32 row, optional fused RMSNorm*w, then
// per-row e4m3 quant). ml8.cu is NOT edited here (owned by another agent) --
// everything device-side that kernel needs is copied verbatim below, cited
// by source line, and kept as `ml8_fp8_qrot_v3_kernel` (the numerics
// ORACLE the bench diffs against). Below that, `radp_ml8_qrot_kernel<...>`
// is OUR OWN restructuring of the same math (this directory's own kernel,
// a rewrite for speed is explicitly in scope) that:
//   (a) fuses a residual add or a split gate/up silu*mul at the load point,
//   (b) stores straight into radiance's fragment-TILED A layout instead of
//       row-major (skipping a separate retile pass), and
//   (c) restructures the Kronecker mix to scale with more waves/threads per
//       row instead of being pinned to one 32-lane wavefront.
//
// ── WHY THE ORIGINAL WAS ~6x OFF BANDWIDTH (measured: K=5120 M=4096 fp32-in/
// fp8-out qrot kernel ~1.2 ms vs a ~0.2 ms 600 GB/s roofline) ──────────────
// ml8_launch_qrot_v3's own dispatch (ml8.cu:5629-5692) is forced to NW==1
// (`static_assert(!HAS_HA || NW == 1, ...)`) for every Kronecker shape --
// b_dim==1024 (K=5120 a_dim=5, K=6144 a_dim=6, both b_dim==1024), b_dim==512,
// b_dim==256. NW==1 means exactly ONE 32-lane wavefront is launched per row
// (blockDim==32). For M=4096 rows that is 4096 blocks of 32 threads each --
// with only 32 threads/CU doing any work per row-block, a CU cannot keep
// enough independent memory requests in flight to hide HBM latency, so the
// GPU sits mostly idle waiting on loads/stores instead of streaming at
// ~600 GB/s. The forced NW==1 is a genuine algorithmic constraint in the
// ORIGINAL kernel, not an oversight: `ml8_fp8_qrot_v3_kernel`'s Kronecker
// mix (`y[k][e] = sum_i h_a[i,k] * v[i][e]`) reads OTHER waves' `v[i]`
// registers directly -- registers are private per-thread, so that mix is
// only expressible in-kernel if every `a` slot the mix needs already lives
// in the SAME wave's registers, i.e. NW==1, MAXAW==a_dim. A second, smaller
// effect: MAXAW==a_dim (5 or 8) times E==B/32==32 means each of those 32
// threads carries `float v[8][32]` (256 registers) resident for the whole
// kernel body -- on top of the amax/scale bookkeeping -- which by itself
// caps how many wavefronts of ANY kind can be co-resident on a SIMD, adding
// a second, independent brake on occupancy.
//
// ── THE FIX: move the Kronecker mix through LDS instead of registers ──────
// `radp_ml8_qrot_kernel` below still does the per-wave, per-`a` in-register
// block-Hadamard/FWHT butterfly exactly as the oracle does (that part was
// never the bottleneck -- it's already fully vectorized float4 loads,
// warp-shuffle butterfly, no LDS round trip, and it stays that way). What
// changes is the CROSS-`a` Kronecker step: each wave writes its own
// (post-Hadamard, post 1/sqrt(B)-scaled) `v[a]` into a shared-memory row
// buffer sized `a_dim_max * B` floats (<=32 KiB for every dispatched shape
// here -- comfortably inside RDNA4's 64 KiB LDS/CU, and the buffer is
// written and read back exactly ONCE per row, not round-tripped
// repeatedly), one `__syncthreads()`, then every wave reads the OTHER
// waves' contributions back out of LDS to compute its own output `a` slot.
// This removes the "the whole row must live in one wave's registers"
// constraint entirely, so NW can now be picked to make MAXAW==1 for every
// Kronecker shape this file dispatches (see radp_pick_qrot_shape below):
// b_dim=1024,a_dim<=5 -> NW=5 (160 threads/row, was 32); a_dim<=8 -> NW=8
// (256 threads/row); b_dim=512,a_dim<=16 / b_dim=256,a_dim<=16 -> NW=16
// (512 threads/row). Per-thread register footprint for `v`/`y` drops from
// `float[8][32]` to `float[1][32]` -- both the "too few threads" and the
// "too many registers per thread" causes of the 6x gap are addressed by the
// same restructuring. The block-Hadamard-only shapes (b_dim in {128,64,32},
// HAS_HA==false) were never register/occupancy-starved this badly since
// ml8_launch_qrot_v3 already used NW up to 16 there; this file still widens
// that to NW up to 32 (see radp_pick_qrot_hadamard) purely to shrink MAXAW
// further for the largest a_dim shapes (K=17408, a_dim=136), same
// register-pressure argument, smaller effect.
//
// The load, the within-block Hadamard/FWHT, the amax/scale bookkeeping, and
// the packed hardware fp8 store (fp8_quant_rot_pack4_hw /
// __builtin_amdgcn_cvt_pk_fp8_f32, via the verbatim-copied
// fp8_quant_rot_f32_to_e4m3) are UNCHANGED math from the oracle -- only the
// Kronecker mix's DATA PATH (registers -> LDS) and its THREAD MAPPING
// (NW==1 forced -> NW chosen so MAXAW==1) changed, so the plain and
// add-fused variants remain bit-identical to
// `rdna4_ml8_qrot_oracle_rowmajor` (the bench's numerics gate).
//
// TILED STORE: identical addressing to gemm_ml84_radiance.h's AT operand --
// see radiance_quant.h's TILED LAYOUT doc comment and `radp_tiled_off`
// below (cross-checked byte-for-byte against gemm_ml84_radiance.hip's
// `radp_retile_a_kernel`).
#include "radiance_quant.h"
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>

namespace ml8qrot {

// ---- ml8.cu:2036, 2041, verbatim ----
static constexpr float ML8_FP8_E4M3_MAX = 448.0f;
static constexpr float ML8_ACT_SCALE_EPS = 1e-12f;

// ---- ml8.cu:4784-4843, verbatim ----
static __device__ __forceinline__ uint8_t fp8_quant_rot_f32_to_e4m3(float xv) {
  uint32_t bits;
  memcpy(&bits, &xv, 4);
  const uint32_t sign = (bits >> 31) & 1u;
  const uint32_t exp_b = (bits >> 23) & 0xFFu;
  const uint32_t mant = bits & 0x7FFFFFu;

  if (exp_b == 0xFFu) {
    return (uint8_t)((sign << 7) | 0x7Fu);
  }
  if (exp_b == 0) {
    return (uint8_t)(sign << 7);
  }

  const int32_t e_un = (int32_t)exp_b - 127;

  if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u)) {
    return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
  }

  if (e_un >= -6) {
    const uint32_t e_e4m3 = (uint32_t)(e_un + 7);
    const uint32_t guard = (mant >> 19) & 1u;
    const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
    const uint32_t lsb = (mant >> 20) & 1u;
    uint32_t m_e4m3 = (mant >> 20) & 0x7u;
    if (guard && (sticky || lsb)) m_e4m3 += 1;
    uint32_t e_out = e_e4m3;
    if (m_e4m3 == 8) {
      m_e4m3 = 0;
      e_out += 1;
      if (e_out > 15) {
        return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
      }
    }
    if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
    return (uint8_t)((sign << 7) | (e_out << 3) | m_e4m3);
  }

  const int32_t shift = 23 - (e_un + 9);
  if (shift > 31) {
    return (uint8_t)(sign << 7);
  }
  const uint32_t implicit = (1u << 23) | mant;
  const uint32_t guard = (implicit >> (shift - 1)) & 1u;
  const uint32_t sticky = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
  uint32_t m_e4m3 = implicit >> shift;
  const uint32_t lsb = m_e4m3 & 1u;
  if (guard && (sticky || lsb)) m_e4m3 += 1;
  if (m_e4m3 >= 8) {
    return (uint8_t)((sign << 7) | (1u << 3));
  }
  return (uint8_t)((sign << 7) | m_e4m3);
}

// ---- ml8.cu:5164-5174, verbatim ----
static __device__ __forceinline__ uint32_t fp8_quant_rot_pack4_hw(float a, float b, float c,
                                                                    float d) {
  return (uint32_t)fp8_quant_rot_f32_to_e4m3(a) | ((uint32_t)fp8_quant_rot_f32_to_e4m3(b) << 8) |
         ((uint32_t)fp8_quant_rot_f32_to_e4m3(c) << 16) |
         ((uint32_t)fp8_quant_rot_f32_to_e4m3(d) << 24);
}

// ---- common.cuh:471-478 / 531-538, verbatim (width=32 only) ----
template <int width>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor(x, offset, width);
  }
  return x;
}
template <int width>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor(x, offset, width));
  }
  return x;
}

// ---- ml8.cu:5208-5392, verbatim (Tin=float only). This is the NUMERICS
// ORACLE the bench diffs the fused/tiled variants against -- do not modify.
template <int B, int NW, int MAXAW, bool HAS_HA>
__launch_bounds__(32 * NW)
__global__ void ml8_fp8_qrot_v3_kernel(
    const float* __restrict__ x, const float* __restrict__ h_a,
    const float* __restrict__ norm_w, float norm_eps, uint8_t* __restrict__ a_fp8,
    float* __restrict__ a_scale, int K, int a_dim, int n_rows) {
  constexpr int E = B / 32;
  static_assert(E >= 1 && (E & (E - 1)) == 0, "b_dim must be a power of two >= 32");
  static_assert(!HAS_HA || NW == 1, "kronecker mixing needs the whole row in one wave");

  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n_rows) return;
  const float* xrow = x + (size_t)row * (size_t)K;

  float v[MAXAW][E];
  float local_max = 0.0f;

#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const float* blk = xrow + (size_t)a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const float4 t = *reinterpret_cast<const float4*>(blk + e);
        v[j][e] = t.x; v[j][e + 1] = t.y; v[j][e + 2] = t.z; v[j][e + 3] = t.w;
      }
    }
  }
  if (norm_w != nullptr) {
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < MAXAW; j++) {
      const int a = wave + j * NW;
      if (a < a_dim) {
#pragma unroll
        for (int e = 0; e < E; e++) ss += v[j][e] * v[j][e];
      }
    }
    ss = warp_reduce_sum<32>(ss);
    float tot = ss;
    if constexpr (NW > 1) {
      __shared__ float s_ss[NW];
      if (lane == 0) s_ss[wave] = ss;
      __syncthreads();
      tot = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; w++) tot += s_ss[w];
      __syncthreads();
    }
    const float rms = rsqrtf(tot / (float)K + norm_eps);
#pragma unroll
    for (int j = 0; j < MAXAW; j++) {
      const int a = wave + j * NW;
      if (a < a_dim) {
        const float* wb = norm_w + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e++) v[j][e] = v[j][e] * rms * wb[e];
      }
    }
  }
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
#pragma unroll
      for (int stride = 1; stride < E; stride <<= 1) {
#pragma unroll
        for (int e = 0; e < E; e++) {
          if ((e & stride) == 0) {
            const float lo = v[j][e];
            const float hi = v[j][e + stride];
            v[j][e] = lo + hi;
            v[j][e + stride] = lo - hi;
          }
        }
      }
#pragma unroll
      for (int ls = 1; ls < 32; ls <<= 1) {
        const bool upper = (lane & ls) != 0;
#pragma unroll
        for (int e = 0; e < E; e++) {
          const float mine = v[j][e];
          const float p = __shfl_xor(mine, ls, 32);
          v[j][e] = upper ? (p - mine) : (mine + p);
        }
      }
      const float inv_sqrt_b = rsqrtf((float)B);
#pragma unroll
      for (int e = 0; e < E; e++) {
        v[j][e] *= inv_sqrt_b;
        if constexpr (!HAS_HA) local_max = fmaxf(local_max, fabsf(v[j][e]));
      }
    }
  }

  if constexpr (HAS_HA) {
    float y[MAXAW][E];
#pragma unroll
    for (int k = 0; k < MAXAW; k++) {
      if (k < a_dim) {
#pragma unroll
        for (int e = 0; e < E; e++) y[k][e] = 0.0f;
#pragma unroll
        for (int i = 0; i < MAXAW; i++) {
          if (i < a_dim) {
            const float h = h_a[i * a_dim + k];
#pragma unroll
            for (int e = 0; e < E; e++) y[k][e] = fmaf(h, v[i][e], y[k][e]);
          }
        }
#pragma unroll
        for (int e = 0; e < E; e++) local_max = fmaxf(local_max, fabsf(y[k][e]));
      }
    }
#pragma unroll
    for (int k = 0; k < MAXAW; k++) {
#pragma unroll
      for (int e = 0; e < E; e++) v[k][e] = y[k][e];
    }
  }

  local_max = warp_reduce_max<32>(local_max);
  float row_max = local_max;
  if constexpr (NW > 1) {
    __shared__ float s_max[NW];
    if (lane == 0) s_max[wave] = local_max;
    __syncthreads();
    row_max = s_max[0];
#pragma unroll
    for (int w = 1; w < NW; w++) row_max = fmaxf(row_max, s_max[w]);
  }
  const float scale = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) a_scale[row] = scale;

  uint8_t* orow = a_fp8 + (size_t)row * (size_t)K;
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      uint8_t* ob = orow + (size_t)a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale,
                                                        v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
        *reinterpret_cast<uint32_t*>(ob + e) = packed;
      }
    }
  }
}

// ── the restructured, TILED-store kernel (this directory's own rewrite) ──
// Store-index helper: column k of row `row` -> byte offset in the TILED
// A buffer. Verbatim formula from radiance's own TILED store / gemm_ml84_
// radiance.hip's radp_retile_a_kernel.
static __device__ __forceinline__ size_t radp_tiled_off(int row, int k, int K) {
  const int g = k >> 3, kt = g >> 1, half = g & 1;
  const int mt = row >> 4, lrow = row & 15, KT = K >> 4;
  return (size_t)(mt * KT + kt) * 256 + (size_t)lrow * 8 + (size_t)half * 128 + (size_t)(k & 7);
}

// MODE: 0 = plain (read x), 1 = residual-add (x + optional residual),
// 2 = split silu*mul (silu(gate) * up, two SEPARATE row pointers, each
// stride K), 3 = fused silu*mul (ggml's [M,2*K] gate_up layout, ONE row
// pointer of stride 2*K, gate/up picked out by `gate_first`).
enum { QMODE_PLAIN = 0, QMODE_ADD = 1, QMODE_SILU_SPLIT = 2, QMODE_SILU_FUSED = 3 };

// Unified fused+TILED kernel. Template params match the oracle above
// (B, NW, MAXAW, HAS_HA) plus MODE selecting the load-time fusion. The
// load, within-block Hadamard/FWHT, amax/scale bookkeeping and packed fp8
// store are bit-identical in structure to the oracle; the Kronecker mix
// (HAS_HA branch) is restructured to go through LDS (`s_row`) instead of
// requiring every `a` slot in one wave's registers, so NW need not be 1 --
// see this file's header comment for the full rationale.
//
// __launch_bounds__'s minBlocksPerMultiprocessor (2nd arg): the per-thread
// `v[MAXAW][E]` register array (E = B/32) is this kernel's whole working
// set outside LDS; for MAXAW*E<=32 (every Kronecker shape here, since
// MAXAW==1 there, and the smallest block-Hadamard bucket) that array is
// small enough that requesting 2 co-resident blocks/CU does not force a
// register cap tighter than the compiler would already pick on its own --
// it only helps shapes where the default single-block-oriented heuristic
// under-requests occupancy. For the large-MAXAW block-Hadamard buckets
// (MAXAW 12, 34: a 48- or 136-register array before any other locals)
// forcing 2 blocks would cap registers below what the array itself needs
// and spill to scratch, so those keep minBlocks==1 -- textually identical
// to this kernel's un-hinted behavior before this change (CUDA/HIP treat
// an omitted 2nd `__launch_bounds__` arg as minBlocksPerMultiprocessor==1).
template <int B, int NW, int MAXAW, bool HAS_HA, int MODE>
__launch_bounds__(32 * NW, (MAXAW * (B / 32) <= 32) ? 2 : 1)
__global__ void radp_ml8_qrot_kernel(
    const float* __restrict__ in0,       // PLAIN/ADD: x row. SILU_SPLIT: gate row.
    const float* __restrict__ in1,       // ADD: residual (nullable). SILU_SPLIT: up row.
    float* __restrict__ residual_out,    // ADD only, nullable.
    const float* __restrict__ h_a,       // Kronecker matrix, or null for block-Hadamard.
    const float* __restrict__ norm_w,    // PLAIN/ADD only, nullable.
    float norm_eps,
    uint8_t* __restrict__ a_tiled, float* __restrict__ a_scale,
    float* __restrict__ y_out,           // PLAIN/ADD only, nullable.
    int gate_first,                      // QMODE_SILU_FUSED only.
    int K, int a_dim, int n_rows) {
  constexpr int E = B / 32;
  static_assert(E >= 4 && (E & (E - 1)) == 0,
                "need float4 load/store granularity; TILED 4-byte groups stay 8-aligned for any "
                "E>=4 since a*B and lane*E are both multiples of 4");
  constexpr bool IS_NORM_MODE = (MODE == QMODE_PLAIN || MODE == QMODE_ADD);
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n_rows) return;

  // QMODE_SILU_FUSED reads ONE [.,2K]-wide row (ggml's fused GLU layout);
  // every other mode reads row-stride-K tensors.
  const size_t stride0 = (MODE == QMODE_SILU_FUSED) ? (size_t)2 * K : (size_t)K;
  const float* base0 = in0 + (size_t)row * stride0;
  const float* row0 = (MODE == QMODE_SILU_FUSED) ? base0 + (gate_first ? 0 : K) : base0;
  const float* row1 = (MODE == QMODE_SILU_FUSED) ? base0 + (gate_first ? K : 0)
                     : (MODE != QMODE_PLAIN && in1) ? in1 + (size_t)row * (size_t)K : nullptr;
  float* rout = (MODE == QMODE_ADD && residual_out) ? residual_out + (size_t)row * (size_t)K : nullptr;
  float* yout_row = (IS_NORM_MODE && y_out) ? y_out + (size_t)row * (size_t)K : nullptr;

  float v[MAXAW][E];
  float local_max = 0.0f;

  // ---- LOAD (fusion point) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const float* blk0 = row0 + (size_t)a * B + lane * E;
      if constexpr (MODE == QMODE_SILU_SPLIT || MODE == QMODE_SILU_FUSED) {
        const float* blk1 = row1 + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e += 4) {
          const float4 tg = *reinterpret_cast<const float4*>(blk0 + e);
          const float4 tu = *reinterpret_cast<const float4*>(blk1 + e);
          const float gv[4] = {tg.x, tg.y, tg.z, tg.w};
          const float uv[4] = {tu.x, tu.y, tu.z, tu.w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const float sg = gv[q] / (1.0f + expf(-gv[q]));
            v[j][e + q] = sg * uv[q];
          }
        }
      } else {
#pragma unroll
        for (int e = 0; e < E; e += 4) {
          const float4 t0 = *reinterpret_cast<const float4*>(blk0 + e);
          float4 t1 = make_float4(0.f, 0.f, 0.f, 0.f);
          if constexpr (MODE == QMODE_ADD) {
            if (row1) t1 = *reinterpret_cast<const float4*>(row1 + (size_t)a * B + lane * E + e);
          }
          v[j][e] = t0.x + t1.x; v[j][e + 1] = t0.y + t1.y;
          v[j][e + 2] = t0.z + t1.z; v[j][e + 3] = t0.w + t1.w;
        }
        if constexpr (MODE == QMODE_ADD) {
          if (rout) {
            float* ob = rout + (size_t)a * B + lane * E;
#pragma unroll
            for (int e = 0; e < E; e++) ob[e] = v[j][e];
          }
        }
      }
    }
  }

  // ---- optional fused RMSNorm*w (PLAIN/ADD only) ----
  if constexpr (IS_NORM_MODE) {
    if (norm_w != nullptr) {
      float ss = 0.0f;
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
#pragma unroll
          for (int e = 0; e < E; e++) ss += v[j][e] * v[j][e];
        }
      }
      ss = warp_reduce_sum<32>(ss);
      float tot = ss;
      if constexpr (NW > 1) {
        __shared__ float s_ss[NW];
        if (lane == 0) s_ss[wave] = ss;
        __syncthreads();
        tot = 0.0f;
#pragma unroll
        for (int w = 0; w < NW; w++) tot += s_ss[w];
        __syncthreads();
      }
      const float rms = rsqrtf(tot / (float)K + norm_eps);
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          const float* wb = norm_w + (size_t)a * B + lane * E;
#pragma unroll
          for (int e = 0; e < E; e++) v[j][e] = v[j][e] * rms * wb[e];
        }
      }
    }
    if (yout_row) {
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          float* ob = yout_row + (size_t)a * B + lane * E;
#pragma unroll
          for (int e = 0; e < E; e++) ob[e] = v[j][e];
        }
      }
    }
  }

  // ---- per-a within-block Hadamard/FWHT (unchanged from the oracle) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
#pragma unroll
      for (int stride = 1; stride < E; stride <<= 1) {
#pragma unroll
        for (int e = 0; e < E; e++) {
          if ((e & stride) == 0) {
            const float lo = v[j][e];
            const float hi = v[j][e + stride];
            v[j][e] = lo + hi;
            v[j][e + stride] = lo - hi;
          }
        }
      }
#pragma unroll
      for (int ls = 1; ls < 32; ls <<= 1) {
        const bool upper = (lane & ls) != 0;
#pragma unroll
        for (int e = 0; e < E; e++) {
          const float mine = v[j][e];
          const float p = __shfl_xor(mine, ls, 32);
          v[j][e] = upper ? (p - mine) : (mine + p);
        }
      }
      const float inv_sqrt_b = rsqrtf((float)B);
#pragma unroll
      for (int e = 0; e < E; e++) {
        v[j][e] *= inv_sqrt_b;
        if constexpr (!HAS_HA) local_max = fmaxf(local_max, fabsf(v[j][e]));
      }
    }
  }

  // ---- cross-a Kronecker mix via LDS, processed in column STRIPS ---------
  // Replaces the oracle's register-resident `y[i]` cross-wave read (only
  // legal at NW==1). Each lane's ESTRIP-wide slice of every 'a' block is
  // staged through `s_row` (sized a_dim_max*32*ESTRIP floats) before the
  // mix, one `__syncthreads()`-guarded strip at a time; the fmaf
  // accumulation order over `i` (0..a_dim-1) is unchanged per strip, so
  // this is bit-identical to the oracle -- it only changes WHEN each
  // column's mix happens, not the arithmetic.
  //
  // LAYOUT (measured fix, see rocprofv3 trace notes): the strip buffer must
  // be laid out SoA (`s_row[a_slot*32*ESTRIP + ee*32 + lane]`), NOT AoS
  // (`... + lane*ESTRIP + ee`, what an earlier version of this file did).
  // AoS puts each lane's ESTRIP floats contiguously, so the per-`ee`
  // store/load that all 32 lanes issue together strides by ESTRIP words
  // across lanes -- bank = (lane*ESTRIP) mod 32, which for every ESTRIP
  // this file dispatches (8, 16, or 32, all powers of two dividing 32)
  // collapses to gcd(ESTRIP,32) = 8..32 lanes sharing one of only 32/that
  // many banks, i.e. an 8-to-32-way LDS bank conflict on EVERY write and
  // read of the mix (measured: this alone made the strip-split version
  // slower, 68 GB/s, than the very "occupancy-limited" full-buffer version
  // it was meant to replace, 106 GB/s). SoA puts each lane's floats
  // ESTRIP-strided instead: for fixed `ee`, incrementing `lane` by 1 moves
  // the address by exactly 1 word, so all 32 lanes hit 32 distinct banks --
  // zero conflicts, same total buffer size, same values, same order.
  if constexpr (HAS_HA) {
    constexpr int ESTRIP = (E < 8) ? E : 8;
    constexpr int NCHUNK = E / ESTRIP;
    static_assert(E % ESTRIP == 0, "E must be a multiple of ESTRIP");
    constexpr int ROWBUF = MAXAW * NW * 32 * ESTRIP;   // a_dim_max * 32 * ESTRIP floats
    constexpr int HABUF  = (MAXAW * NW) * (MAXAW * NW);
    __shared__ float s_row[ROWBUF];
    __shared__ float s_ha[HABUF];
    for (int idx = threadIdx.x; idx < a_dim * a_dim; idx += blockDim.x) s_ha[idx] = h_a[idx];
    __syncthreads();
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const int e0 = c * ESTRIP;
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          // SoA layout: consecutive lanes hit consecutive LDS words (bank =
          // address mod 32 = lane), not the old lane*ESTRIP AoS stride
          // (bank = (lane*ESTRIP) mod 32, an 8..32-way conflict for every
          // ESTRIP in {8,16,32} -- see the file-level note above). Same
          // ROWBUF size, same values, same per-(a,lane,ee) address space --
          // only which word within the a-slot each (lane,ee) maps to moved,
          // so this is bit-identical.
          float* dst = s_row + a * 32 * ESTRIP + lane;
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) dst[ee * 32] = v[j][e0 + ee];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int k = wave + j * NW;
        if (k < a_dim) {
          float y[ESTRIP];
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) y[ee] = 0.0f;
          for (int i = 0; i < a_dim; i++) {
            const float h = s_ha[i * a_dim + k];
            const float* src = s_row + i * 32 * ESTRIP + lane;
#pragma unroll
            for (int ee = 0; ee < ESTRIP; ee++) y[ee] = fmaf(h, src[ee * 32], y[ee]);
          }
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) {
            v[j][e0 + ee] = y[ee];
            local_max = fmaxf(local_max, fabsf(y[ee]));
          }
        }
      }
      __syncthreads();   // s_row is reused by the next strip -- guard the WAR hazard
    }
  }

  // ---- amax reduce + scale (unchanged) ----
  local_max = warp_reduce_max<32>(local_max);
  float row_max = local_max;
  if constexpr (NW > 1) {
    __shared__ float s_max[NW];
    if (lane == 0) s_max[wave] = local_max;
    __syncthreads();
    row_max = s_max[0];
#pragma unroll
    for (int w = 1; w < NW; w++) row_max = fmaxf(row_max, s_max[w]);
  }
  const float scale = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) a_scale[row] = scale;

  // ---- single coalesced packed fp8 TILED store (unchanged) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const int k0 = a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale,
                                                        v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
        *reinterpret_cast<uint32_t*>(a_tiled + radp_tiled_off(row, k0 + e, K)) = packed;
      }
    }
  }
}

// Variant 5 (gated per-head RMSNorm, GDN out_proj site): row `o` is n_heads
// groups of head_dim columns; each group is independently rms-normalized
// over head_dim, scaled by norm_w[head_dim] (broadcast per head), then
// gated by silu(z) (z has the SAME logical [head_dim,n_heads,M] shape as
// `o` but its own strides z_nb1/z_nb2 -- see radiance_quant.h). The result
// is then rotated + TILED-quantized exactly like the other variants (no
// separate norm_w there -- that stage's norm_w argument is always null,
// per the header comment: "the norm is the per-head one above, NOT a
// full-row norm").
//
// Thread/data mapping: this reuses the same (wave,lane) -> (a,column)
// mapping as every other variant here (a = wave + j*NW owns block
// [a*B, a*B+B), lane owns [lane*E, lane*E+E) within that block). The
// caller-checked precondition (see rdna4_ml8_qrot_gated_norm_tiled) is
// head_dim a multiple of E and head_dim/E a power of two dividing 32, so
// EVERY lane's own E-wide slice lies entirely inside exactly one head --
// no lane straddles a head boundary, so the per-head sum-of-squares only
// ever needs a reduction among the `lanes_per_head` lanes that share a
// head (a plain __shfl_xor over that sub-group), never LDS.
// GATE: 0 = pattern C (per-head rmsnorm*w, gated by silu(z)); 1 = attention output
// gating (no norm, gated by sigmoid(z)) -- ggml_fp8_quant_rot_gated, replaces
// ggml_cont(gate)+ggml_sigmoid+ggml_mul in qwen35's build_layer_attn. Same
// expression order as those ops: o * (1/(1+expf(-z))).
template <int B, int NW, int MAXAW, bool HAS_HA, int GATE = 0>
__launch_bounds__(32 * NW, (MAXAW * (B / 32) <= 32) ? 2 : 1)
__global__ void radp_ml8_qrot_gated_norm_kernel(
    const float* __restrict__ o, const char* __restrict__ z_bytes, size_t z_nb1, size_t z_nb2,
    const float* __restrict__ norm_w, float norm_eps, int head_dim, int lanes_per_head,
    const float* __restrict__ h_a, uint8_t* __restrict__ a_tiled, float* __restrict__ a_scale,
    float* __restrict__ y_out, int K, int a_dim, int n_rows) {
  constexpr int E = B / 32;
  static_assert(E >= 4 && (E & (E - 1)) == 0,
                "need float4 load/store granularity; TILED 4-byte groups stay 8-aligned for any "
                "E>=4 since a*B and lane*E are both multiples of 4");
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n_rows) return;

  const float* orow = o + (size_t)row * (size_t)K;
  const char* zrow = z_bytes + (size_t)row * z_nb2;
  float* yout_row = y_out ? y_out + (size_t)row * (size_t)K : nullptr;

  const int lane_in_head = lane % lanes_per_head;   // position (in E-wide units) within its head
  const int local_head   = lane / lanes_per_head;   // which head within THIS a-block
  const int heads_per_a  = B / head_dim;

  float v[MAXAW][E];
  float local_max = 0.0f;

  // ---- LOAD: read o, per-head RMSNorm*w, gate by silu(z) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const float* blk = orow + (size_t)a * B + lane * E;
      float ss = 0.0f;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const float4 t = *reinterpret_cast<const float4*>(blk + e);
        v[j][e] = t.x; v[j][e + 1] = t.y; v[j][e + 2] = t.z; v[j][e + 3] = t.w;
      }
      float rms = 1.0f;
      if constexpr (GATE == 0) {
#pragma unroll
        for (int e = 0; e < E; e++) ss += v[j][e] * v[j][e];
        for (int off = 1; off < lanes_per_head; off <<= 1) ss += __shfl_xor(ss, off, lanes_per_head);
        const float mean = ss / (float)head_dim;
        rms = rsqrtf(mean + norm_eps);
      }
      const int global_head = a * heads_per_a + local_head;
      const float* wb = (GATE == 0) ? norm_w + lane_in_head * E : nullptr;
      const char* zblk = zrow + (size_t)global_head * z_nb1 + (size_t)(lane_in_head * E) * sizeof(float);
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const float4 zv = *reinterpret_cast<const float4*>(zblk + (size_t)e * sizeof(float));
        const float zg[4] = {zv.x, zv.y, zv.z, zv.w};
#pragma unroll
        for (int q = 0; q < 4; q++) {
          if constexpr (GATE == 0) {
            const float r = v[j][e + q] * rms * wb[e + q];
            const float sg = zg[q] / (1.0f + expf(-zg[q]));
            v[j][e + q] = r * sg;
          } else {
            const float sg = 1.0f / (1.0f + expf(-zg[q]));
            v[j][e + q] = v[j][e + q] * sg;
          }
        }
      }
      if (yout_row) {
        float* ob = yout_row + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e++) ob[e] = v[j][e];
      }
    }
  }

  // ---- per-a within-block Hadamard/FWHT (identical to every other variant) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
#pragma unroll
      for (int stride = 1; stride < E; stride <<= 1) {
#pragma unroll
        for (int e = 0; e < E; e++) {
          if ((e & stride) == 0) {
            const float lo = v[j][e];
            const float hi = v[j][e + stride];
            v[j][e] = lo + hi;
            v[j][e + stride] = lo - hi;
          }
        }
      }
#pragma unroll
      for (int ls = 1; ls < 32; ls <<= 1) {
        const bool upper = (lane & ls) != 0;
#pragma unroll
        for (int e = 0; e < E; e++) {
          const float mine = v[j][e];
          const float p = __shfl_xor(mine, ls, 32);
          v[j][e] = upper ? (p - mine) : (mine + p);
        }
      }
      const float inv_sqrt_b = rsqrtf((float)B);
#pragma unroll
      for (int e = 0; e < E; e++) {
        v[j][e] *= inv_sqrt_b;
        if constexpr (!HAS_HA) local_max = fmaxf(local_max, fabsf(v[j][e]));
      }
    }
  }

  // ---- cross-a Kronecker mix via LDS, in column strips (identical to the
  // other Kronecker variants -- see their shared comment above) ----
  if constexpr (HAS_HA) {
    constexpr int ESTRIP = (E < 8) ? E : 8;
    constexpr int NCHUNK = E / ESTRIP;
    static_assert(E % ESTRIP == 0, "E must be a multiple of ESTRIP");
    constexpr int ROWBUF = MAXAW * NW * 32 * ESTRIP;
    constexpr int HABUF  = (MAXAW * NW) * (MAXAW * NW);
    __shared__ float s_row[ROWBUF];
    __shared__ float s_ha[HABUF];
    for (int idx = threadIdx.x; idx < a_dim * a_dim; idx += blockDim.x) s_ha[idx] = h_a[idx];
    __syncthreads();
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const int e0 = c * ESTRIP;
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          // SoA layout: consecutive lanes hit consecutive LDS words (bank =
          // address mod 32 = lane), not the old lane*ESTRIP AoS stride
          // (bank = (lane*ESTRIP) mod 32, an 8..32-way conflict for every
          // ESTRIP in {8,16,32} -- see the file-level note above). Same
          // ROWBUF size, same values, same per-(a,lane,ee) address space --
          // only which word within the a-slot each (lane,ee) maps to moved,
          // so this is bit-identical.
          float* dst = s_row + a * 32 * ESTRIP + lane;
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) dst[ee * 32] = v[j][e0 + ee];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int k = wave + j * NW;
        if (k < a_dim) {
          float y[ESTRIP];
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) y[ee] = 0.0f;
          for (int i = 0; i < a_dim; i++) {
            const float h = s_ha[i * a_dim + k];
            const float* src = s_row + i * 32 * ESTRIP + lane;
#pragma unroll
            for (int ee = 0; ee < ESTRIP; ee++) y[ee] = fmaf(h, src[ee * 32], y[ee]);
          }
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) {
            v[j][e0 + ee] = y[ee];
            local_max = fmaxf(local_max, fabsf(y[ee]));
          }
        }
      }
      __syncthreads();
    }
  }

  // ---- amax reduce + scale (unchanged) ----
  local_max = warp_reduce_max<32>(local_max);
  float row_max = local_max;
  if constexpr (NW > 1) {
    __shared__ float s_max[NW];
    if (lane == 0) s_max[wave] = local_max;
    __syncthreads();
    row_max = s_max[0];
#pragma unroll
    for (int w = 1; w < NW; w++) row_max = fmaxf(row_max, s_max[w]);
  }
  const float scale = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) a_scale[row] = scale;

  // ---- single coalesced packed fp8 TILED store (unchanged) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const int k0 = a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale,
                                                        v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
        *reinterpret_cast<uint32_t*>(a_tiled + radp_tiled_off(row, k0 + e, K)) = packed;
      }
    }
  }
}

// ---- bf16-input upcast: bits -> fp32, matching how a bf16 value's bits sit
// in the upper 16 of an fp32 (bf16 is just a truncated fp32 mantissa/exp) --
// no rounding, exact for every bf16 value. Takes raw uint16_t bits so this
// file (and radiance_quant.h) never has to name a specific bf16 type
// (__hip_bfloat16 vs nv_bfloat16 vs a plain uint16_t "bits" convention) --
// bit pattern is identical across all of them for this purpose. ----
static __device__ __forceinline__ float radp_bf16_bits_to_f32(uint16_t bits) {
  uint32_t b = (uint32_t)bits << 16;
  float f;
  memcpy(&f, &b, 4);
  return f;
}

// Variant 6 (bf16-INPUT split silu*mul): identical to
// radp_ml8_qrot_kernel<...,QMODE_SILU_SPLIT> except the load, which reads
// bf16 gate/up (8 elements = 16 bytes per lane-step, one uint4 transaction)
// instead of fp32 float4 -- LLAMA_ACT_BF16 prefill's ffn_up/ffn_gate GEMM
// outputs are bf16, so this halves the bytes this kernel is bandwidth-bound
// on (measured 475 GB/s reading fp32 gate+up in production). Every value is
// upcast to fp32 immediately after the load (radp_bf16_bits_to_f32, exact,
// no rounding), so silu*mul -> Hadamard/Kronecker rotate -> amax -> pack is
// the IDENTICAL fp32 math the fp32-input kernel runs -- only the load
// stage differs, verbatim-copied tail below.
template <int B, int NW, int MAXAW, bool HAS_HA>
__launch_bounds__(32 * NW, (MAXAW * (B / 32) <= 32) ? 2 : 1)
__global__ void radp_ml8_qrot_silu_mul_split_bf16_kernel(
    const uint16_t* __restrict__ gate_bf16, const uint16_t* __restrict__ up_bf16,
    const float* __restrict__ h_a, uint8_t* __restrict__ a_tiled, float* __restrict__ a_scale,
    int K, int a_dim, int n_rows) {
  constexpr int E = B / 32;
  static_assert(E >= 4 && (E & (E - 1)) == 0,
                "need at least uint2 (4x bf16) load granularity; a*B and lane*E are both "
                "multiples of 4 for every E>=4 dispatched here");
  // B=1024/512/256 (E=32/16/8): one uint4 (16B = 8 bf16) transaction per
  // step. B=128 (E=4, the block-Hadamard production shape this bf16
  // variant primarily targets, N=17408): a lane owns only 4 elements
  // total, so an 8-wide load would reach into the NEXT lane's columns --
  // uint2 (8B = 4 bf16) instead, one transaction covering the whole E.
  constexpr int LOADSTEP = (E >= 8) ? 8 : E;
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n_rows) return;
  const uint16_t* grow = gate_bf16 + (size_t)row * (size_t)K;
  const uint16_t* urow = up_bf16 + (size_t)row * (size_t)K;

  float v[MAXAW][E];
  float local_max = 0.0f;

#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const uint16_t* blkg = grow + (size_t)a * B + lane * E;
      const uint16_t* blku = urow + (size_t)a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += LOADSTEP) {
        if constexpr (LOADSTEP == 8) {
          const uint4 rg = *reinterpret_cast<const uint4*>(blkg + e);
          const uint4 ru = *reinterpret_cast<const uint4*>(blku + e);
          const uint32_t gw[4] = {rg.x, rg.y, rg.z, rg.w};
          const uint32_t uw[4] = {ru.x, ru.y, ru.z, ru.w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const float g0 = radp_bf16_bits_to_f32((uint16_t)(gw[q] & 0xFFFFu));
            const float g1 = radp_bf16_bits_to_f32((uint16_t)(gw[q] >> 16));
            const float u0 = radp_bf16_bits_to_f32((uint16_t)(uw[q] & 0xFFFFu));
            const float u1 = radp_bf16_bits_to_f32((uint16_t)(uw[q] >> 16));
            v[j][e + 2 * q]     = (g0 / (1.0f + expf(-g0))) * u0;
            v[j][e + 2 * q + 1] = (g1 / (1.0f + expf(-g1))) * u1;
          }
        } else {
          const uint2 rg = *reinterpret_cast<const uint2*>(blkg + e);
          const uint2 ru = *reinterpret_cast<const uint2*>(blku + e);
          const uint32_t gw[2] = {rg.x, rg.y};
          const uint32_t uw[2] = {ru.x, ru.y};
#pragma unroll
          for (int q = 0; q < 2; q++) {
            const float g0 = radp_bf16_bits_to_f32((uint16_t)(gw[q] & 0xFFFFu));
            const float g1 = radp_bf16_bits_to_f32((uint16_t)(gw[q] >> 16));
            const float u0 = radp_bf16_bits_to_f32((uint16_t)(uw[q] & 0xFFFFu));
            const float u1 = radp_bf16_bits_to_f32((uint16_t)(uw[q] >> 16));
            v[j][e + 2 * q]     = (g0 / (1.0f + expf(-g0))) * u0;
            v[j][e + 2 * q + 1] = (g1 / (1.0f + expf(-g1))) * u1;
          }
        }
      }
    }
  }

  // ---- per-a within-block Hadamard/FWHT (identical to every other variant) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
#pragma unroll
      for (int stride = 1; stride < E; stride <<= 1) {
#pragma unroll
        for (int e = 0; e < E; e++) {
          if ((e & stride) == 0) {
            const float lo = v[j][e];
            const float hi = v[j][e + stride];
            v[j][e] = lo + hi;
            v[j][e + stride] = lo - hi;
          }
        }
      }
#pragma unroll
      for (int ls = 1; ls < 32; ls <<= 1) {
        const bool upper = (lane & ls) != 0;
#pragma unroll
        for (int e = 0; e < E; e++) {
          const float mine = v[j][e];
          const float p = __shfl_xor(mine, ls, 32);
          v[j][e] = upper ? (p - mine) : (mine + p);
        }
      }
      const float inv_sqrt_b = rsqrtf((float)B);
#pragma unroll
      for (int e = 0; e < E; e++) {
        v[j][e] *= inv_sqrt_b;
        if constexpr (!HAS_HA) local_max = fmaxf(local_max, fabsf(v[j][e]));
      }
    }
  }

  // ---- cross-a Kronecker mix via LDS, in column strips (identical to the
  // other Kronecker variants -- see radp_ml8_qrot_kernel's shared comment) ----
  if constexpr (HAS_HA) {
    constexpr int ESTRIP = (E < 8) ? E : 8;
    constexpr int NCHUNK = E / ESTRIP;
    static_assert(E % ESTRIP == 0, "E must be a multiple of ESTRIP");
    constexpr int ROWBUF = MAXAW * NW * 32 * ESTRIP;
    constexpr int HABUF  = (MAXAW * NW) * (MAXAW * NW);
    __shared__ float s_row[ROWBUF];
    __shared__ float s_ha[HABUF];
    for (int idx = threadIdx.x; idx < a_dim * a_dim; idx += blockDim.x) s_ha[idx] = h_a[idx];
    __syncthreads();
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const int e0 = c * ESTRIP;
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          // SoA layout: consecutive lanes hit consecutive LDS words (bank =
          // address mod 32 = lane), not the old lane*ESTRIP AoS stride
          // (bank = (lane*ESTRIP) mod 32, an 8..32-way conflict for every
          // ESTRIP in {8,16,32} -- see the file-level note above). Same
          // ROWBUF size, same values, same per-(a,lane,ee) address space --
          // only which word within the a-slot each (lane,ee) maps to moved,
          // so this is bit-identical.
          float* dst = s_row + a * 32 * ESTRIP + lane;
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) dst[ee * 32] = v[j][e0 + ee];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int k = wave + j * NW;
        if (k < a_dim) {
          float y[ESTRIP];
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) y[ee] = 0.0f;
          for (int i = 0; i < a_dim; i++) {
            const float h = s_ha[i * a_dim + k];
            const float* src = s_row + i * 32 * ESTRIP + lane;
#pragma unroll
            for (int ee = 0; ee < ESTRIP; ee++) y[ee] = fmaf(h, src[ee * 32], y[ee]);
          }
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) {
            v[j][e0 + ee] = y[ee];
            local_max = fmaxf(local_max, fabsf(y[ee]));
          }
        }
      }
      __syncthreads();
    }
  }

  // ---- amax reduce + scale (unchanged) ----
  local_max = warp_reduce_max<32>(local_max);
  float row_max = local_max;
  if constexpr (NW > 1) {
    __shared__ float s_max[NW];
    if (lane == 0) s_max[wave] = local_max;
    __syncthreads();
    row_max = s_max[0];
#pragma unroll
    for (int w = 1; w < NW; w++) row_max = fmaxf(row_max, s_max[w]);
  }
  const float scale = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) a_scale[row] = scale;

  // ---- single coalesced packed fp8 TILED store (unchanged) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const int k0 = a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale,
                                                        v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
        *reinterpret_cast<uint32_t*>(a_tiled + radp_tiled_off(row, k0 + e, K)) = packed;
      }
    }
  }
}

// Variant 7 (bf16-INPUT residual add): identical to
// radp_ml8_qrot_kernel<...,QMODE_ADD> except the load, which reads bf16 x
// and residual (8 elements/16 bytes per lane-step) instead of fp32 float4,
// upcasts both to fp32 (exact, radp_bf16_bits_to_f32) and adds in fp32 --
// same reasoning as the silu*mul bf16 variant above. residual_out (if
// requested) is written fp32 (the task's "keep it simple" -- the residual
// stream itself going bf16 end-to-end is a later step, not this kernel's
// concern). norm_w/y_out share the exact same optional-fused-RMSNorm and
// post-norm-capture semantics as the fp32 ADD kernel's tail.
template <int B, int NW, int MAXAW, bool HAS_HA>
__launch_bounds__(32 * NW, (MAXAW * (B / 32) <= 32) ? 2 : 1)
__global__ void radp_ml8_qrot_add_bf16_kernel(
    const uint16_t* __restrict__ x_bf16, const uint16_t* __restrict__ residual_bf16,
    float* __restrict__ residual_out, const float* __restrict__ h_a,
    const float* __restrict__ norm_w, float norm_eps, uint8_t* __restrict__ a_tiled,
    float* __restrict__ a_scale, float* __restrict__ y_out, int K, int a_dim, int n_rows) {
  constexpr int E = B / 32;
  static_assert(E >= 4 && (E & (E - 1)) == 0,
                "need at least uint2 (4x bf16) load granularity; a*B and lane*E are both "
                "multiples of 4 for every E>=4 dispatched here");
  constexpr int LOADSTEP = (E >= 8) ? 8 : E;   // see the silu*mul bf16 kernel's own comment
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n_rows) return;
  const uint16_t* xrow = x_bf16 + (size_t)row * (size_t)K;
  const uint16_t* rrow = residual_bf16 ? residual_bf16 + (size_t)row * (size_t)K : nullptr;
  float* rout = residual_out ? residual_out + (size_t)row * (size_t)K : nullptr;
  float* yout_row = y_out ? y_out + (size_t)row * (size_t)K : nullptr;

  float v[MAXAW][E];
  float local_max = 0.0f;

#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const uint16_t* blkx = xrow + (size_t)a * B + lane * E;
      const uint16_t* blkr = rrow ? rrow + (size_t)a * B + lane * E : nullptr;
#pragma unroll
      for (int e = 0; e < E; e += LOADSTEP) {
        if constexpr (LOADSTEP == 8) {
          const uint4 rx = *reinterpret_cast<const uint4*>(blkx + e);
          uint4 rr = make_uint4(0, 0, 0, 0);
          if (blkr) rr = *reinterpret_cast<const uint4*>(blkr + e);
          const uint32_t xw[4] = {rx.x, rx.y, rx.z, rx.w};
          const uint32_t rw[4] = {rr.x, rr.y, rr.z, rr.w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            const float x0 = radp_bf16_bits_to_f32((uint16_t)(xw[q] & 0xFFFFu));
            const float x1 = radp_bf16_bits_to_f32((uint16_t)(xw[q] >> 16));
            const float r0 = blkr ? radp_bf16_bits_to_f32((uint16_t)(rw[q] & 0xFFFFu)) : 0.0f;
            const float r1 = blkr ? radp_bf16_bits_to_f32((uint16_t)(rw[q] >> 16)) : 0.0f;
            v[j][e + 2 * q]     = x0 + r0;
            v[j][e + 2 * q + 1] = x1 + r1;
          }
        } else {
          const uint2 rx = *reinterpret_cast<const uint2*>(blkx + e);
          uint2 rr = make_uint2(0, 0);
          if (blkr) rr = *reinterpret_cast<const uint2*>(blkr + e);
          const uint32_t xw[2] = {rx.x, rx.y};
          const uint32_t rw[2] = {rr.x, rr.y};
#pragma unroll
          for (int q = 0; q < 2; q++) {
            const float x0 = radp_bf16_bits_to_f32((uint16_t)(xw[q] & 0xFFFFu));
            const float x1 = radp_bf16_bits_to_f32((uint16_t)(xw[q] >> 16));
            const float r0 = blkr ? radp_bf16_bits_to_f32((uint16_t)(rw[q] & 0xFFFFu)) : 0.0f;
            const float r1 = blkr ? radp_bf16_bits_to_f32((uint16_t)(rw[q] >> 16)) : 0.0f;
            v[j][e + 2 * q]     = x0 + r0;
            v[j][e + 2 * q + 1] = x1 + r1;
          }
        }
      }
      if (rout) {
        float* ob = rout + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e++) ob[e] = v[j][e];
      }
    }
  }

  if (norm_w != nullptr) {
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < MAXAW; j++) {
      const int a = wave + j * NW;
      if (a < a_dim) {
#pragma unroll
        for (int e = 0; e < E; e++) ss += v[j][e] * v[j][e];
      }
    }
    ss = warp_reduce_sum<32>(ss);
    float tot = ss;
    if constexpr (NW > 1) {
      __shared__ float s_ss[NW];
      if (lane == 0) s_ss[wave] = ss;
      __syncthreads();
      tot = 0.0f;
#pragma unroll
      for (int w = 0; w < NW; w++) tot += s_ss[w];
      __syncthreads();
    }
    const float rms = rsqrtf(tot / (float)K + norm_eps);
#pragma unroll
    for (int j = 0; j < MAXAW; j++) {
      const int a = wave + j * NW;
      if (a < a_dim) {
        const float* wb = norm_w + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e++) v[j][e] = v[j][e] * rms * wb[e];
      }
    }
  }
  if (yout_row) {
#pragma unroll
    for (int j = 0; j < MAXAW; j++) {
      const int a = wave + j * NW;
      if (a < a_dim) {
        float* ob = yout_row + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e++) ob[e] = v[j][e];
      }
    }
  }

  // ---- per-a within-block Hadamard/FWHT (identical to every other variant) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
#pragma unroll
      for (int stride = 1; stride < E; stride <<= 1) {
#pragma unroll
        for (int e = 0; e < E; e++) {
          if ((e & stride) == 0) {
            const float lo = v[j][e];
            const float hi = v[j][e + stride];
            v[j][e] = lo + hi;
            v[j][e + stride] = lo - hi;
          }
        }
      }
#pragma unroll
      for (int ls = 1; ls < 32; ls <<= 1) {
        const bool upper = (lane & ls) != 0;
#pragma unroll
        for (int e = 0; e < E; e++) {
          const float mine = v[j][e];
          const float p = __shfl_xor(mine, ls, 32);
          v[j][e] = upper ? (p - mine) : (mine + p);
        }
      }
      const float inv_sqrt_b = rsqrtf((float)B);
#pragma unroll
      for (int e = 0; e < E; e++) {
        v[j][e] *= inv_sqrt_b;
        if constexpr (!HAS_HA) local_max = fmaxf(local_max, fabsf(v[j][e]));
      }
    }
  }

  if constexpr (HAS_HA) {
    constexpr int ESTRIP = (E < 8) ? E : 8;
    constexpr int NCHUNK = E / ESTRIP;
    static_assert(E % ESTRIP == 0, "E must be a multiple of ESTRIP");
    constexpr int ROWBUF = MAXAW * NW * 32 * ESTRIP;
    constexpr int HABUF  = (MAXAW * NW) * (MAXAW * NW);
    __shared__ float s_row[ROWBUF];
    __shared__ float s_ha[HABUF];
    for (int idx = threadIdx.x; idx < a_dim * a_dim; idx += blockDim.x) s_ha[idx] = h_a[idx];
    __syncthreads();
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const int e0 = c * ESTRIP;
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          // SoA layout: consecutive lanes hit consecutive LDS words (bank =
          // address mod 32 = lane), not the old lane*ESTRIP AoS stride
          // (bank = (lane*ESTRIP) mod 32, an 8..32-way conflict for every
          // ESTRIP in {8,16,32} -- see the file-level note above). Same
          // ROWBUF size, same values, same per-(a,lane,ee) address space --
          // only which word within the a-slot each (lane,ee) maps to moved,
          // so this is bit-identical.
          float* dst = s_row + a * 32 * ESTRIP + lane;
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) dst[ee * 32] = v[j][e0 + ee];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int k = wave + j * NW;
        if (k < a_dim) {
          float y[ESTRIP];
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) y[ee] = 0.0f;
          for (int i = 0; i < a_dim; i++) {
            const float h = s_ha[i * a_dim + k];
            const float* src = s_row + i * 32 * ESTRIP + lane;
#pragma unroll
            for (int ee = 0; ee < ESTRIP; ee++) y[ee] = fmaf(h, src[ee * 32], y[ee]);
          }
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) {
            v[j][e0 + ee] = y[ee];
            local_max = fmaxf(local_max, fabsf(y[ee]));
          }
        }
      }
      __syncthreads();
    }
  }

  local_max = warp_reduce_max<32>(local_max);
  float row_max = local_max;
  if constexpr (NW > 1) {
    __shared__ float s_max[NW];
    if (lane == 0) s_max[wave] = local_max;
    __syncthreads();
    row_max = s_max[0];
#pragma unroll
    for (int w = 1; w < NW; w++) row_max = fmaxf(row_max, s_max[w]);
  }
  const float scale = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) a_scale[row] = scale;

#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const int k0 = a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale,
                                                        v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
        *reinterpret_cast<uint32_t*>(a_tiled + radp_tiled_off(row, k0 + e, K)) = packed;
      }
    }
  }
}

// Variant 8 (gated per-head RMSNorm, bf16 + libr4d head-permuted `o`): lets
// the GDN gated-norm fusion consume libr4d chunk_scan's OUTPUT directly
// (dropping the adapter's separate "output cast+unpermute" kernel --
// r4d_gdn_cast_o_unpermute_kernel in mt_gdn_r4d.cu, ~179us/T=1024 layer-
// call, ~140ms/16k). libr4d's output is bf16, [T, H, head_dim] contiguous
// (o_nb_head/o_nb_tok let the caller pass its real strides without this
// file assuming that layout), with heads in libr4d's OWN GQA-blocked order
// -- mt_gdn_r4d.cu's own formula (r4d_gdn_perm_head, its header comment
// has the correctness proof): for logical (ggml) head h,
//   hv' = perm(h) = R*(h mod Hg) + h/Hg,   R = H/Hg
// is the libr4d index to read. `head_src[h]` is exactly that lookup,
// precomputed by the caller (nullptr => identity, R=1/Hg=H case). Every
// OTHER input/output (z, norm_w, a_scale, A_tiled, y_out) stays in LOGICAL
// head order -- only the `o` LOAD is redirected through head_src and read
// as bf16 (upcast exactly via radp_bf16_bits_to_f32, same as this file's
// other bf16 variants) instead of packed fp32. Tail (Hadamard/Kronecker
// rotate, amax, e4m3 TILED store) is verbatim-identical to
// radp_ml8_qrot_gated_norm_kernel.
//
// Precondition (checked by the host launcher, same as the plain gated-norm
// entry point): head_dim a multiple of E=B/32 and head_dim/E a power of
// two dividing 32, so every lane's own E-wide slice lies entirely inside
// ONE head -- meaning each thread needs exactly ONE head_src lookup
// (global_head = a*heads_per_a + local_head, the SAME global_head the
// z/norm_w addressing below already computes), not a per-column one.
template <int B, int NW, int MAXAW, bool HAS_HA>
__launch_bounds__(32 * NW, (MAXAW * (B / 32) <= 32) ? 2 : 1)
__global__ void radp_ml8_qrot_gated_norm_r4d_kernel(
    const uint16_t* __restrict__ o_bf16, size_t o_nb_head, size_t o_nb_tok,
    const int32_t* __restrict__ head_src, const char* __restrict__ z_bytes, size_t z_nb1,
    size_t z_nb2, const float* __restrict__ norm_w, float norm_eps, int head_dim,
    int lanes_per_head, const float* __restrict__ h_a, uint8_t* __restrict__ a_tiled,
    float* __restrict__ a_scale, float* __restrict__ y_out, int K, int a_dim, int n_rows) {
  constexpr int E = B / 32;
  static_assert(E >= 4 && (E & (E - 1)) == 0,
                "need at least uint2 (4x bf16) load granularity; a*B and lane*E are both "
                "multiples of 4 for every E>=4 dispatched here");
  constexpr int LOADSTEP = (E >= 8) ? 8 : E;   // see the silu*mul/add bf16 kernels' own comment
  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  const int row = blockIdx.x;
  if (row >= n_rows) return;

  const char* o_row_bytes = (const char*)o_bf16 + (size_t)row * o_nb_tok;
  const char* zrow = z_bytes + (size_t)row * z_nb2;
  float* yout_row = y_out ? y_out + (size_t)row * (size_t)K : nullptr;

  const int lane_in_head = lane % lanes_per_head;   // position (in E-wide units) within its head
  const int local_head   = lane / lanes_per_head;   // which head within THIS a-block
  const int heads_per_a  = B / head_dim;

  float v[MAXAW][E];
  float local_max = 0.0f;

  // ---- LOAD: read o (bf16, through head_src + strides), per-head
  // RMSNorm*w, gate by silu(z) (z stays logical-order/fp32, unchanged) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const int global_head = a * heads_per_a + local_head;
      const int src_head = head_src ? head_src[global_head] : global_head;
      const uint16_t* blk =
          (const uint16_t*)(o_row_bytes + (size_t)src_head * o_nb_head) + lane_in_head * E;
      float ss = 0.0f;
#pragma unroll
      for (int e = 0; e < E; e += LOADSTEP) {
        if constexpr (LOADSTEP == 8) {
          const uint4 rb = *reinterpret_cast<const uint4*>(blk + e);
          const uint32_t bw[4] = {rb.x, rb.y, rb.z, rb.w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            v[j][e + 2 * q]     = radp_bf16_bits_to_f32((uint16_t)(bw[q] & 0xFFFFu));
            v[j][e + 2 * q + 1] = radp_bf16_bits_to_f32((uint16_t)(bw[q] >> 16));
          }
        } else {
          const uint2 rb = *reinterpret_cast<const uint2*>(blk + e);
          const uint32_t bw[2] = {rb.x, rb.y};
#pragma unroll
          for (int q = 0; q < 2; q++) {
            v[j][e + 2 * q]     = radp_bf16_bits_to_f32((uint16_t)(bw[q] & 0xFFFFu));
            v[j][e + 2 * q + 1] = radp_bf16_bits_to_f32((uint16_t)(bw[q] >> 16));
          }
        }
      }
#pragma unroll
      for (int e = 0; e < E; e++) ss += v[j][e] * v[j][e];
      for (int off = 1; off < lanes_per_head; off <<= 1) ss += __shfl_xor(ss, off, lanes_per_head);
      const float mean = ss / (float)head_dim;
      const float rms = rsqrtf(mean + norm_eps);
      const float* wb = norm_w + lane_in_head * E;
      const char* zblk = zrow + (size_t)global_head * z_nb1 + (size_t)(lane_in_head * E) * sizeof(float);
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const float4 zv = *reinterpret_cast<const float4*>(zblk + (size_t)e * sizeof(float));
        const float zg[4] = {zv.x, zv.y, zv.z, zv.w};
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const float r = v[j][e + q] * rms * wb[e + q];
          const float sg = zg[q] / (1.0f + expf(-zg[q]));
          v[j][e + q] = r * sg;
        }
      }
      if (yout_row) {
        // y_out stays LOGICAL head order (per the header comment: "it must
        // equal what the graph's MUL would have produced") -- addressed by
        // `a` (the logical Kronecker/Hadamard block index), NOT src_head.
        float* ob = yout_row + (size_t)a * B + lane * E;
#pragma unroll
        for (int e = 0; e < E; e++) ob[e] = v[j][e];
      }
    }
  }

  // ---- per-a within-block Hadamard/FWHT (identical to every other variant) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
#pragma unroll
      for (int stride = 1; stride < E; stride <<= 1) {
#pragma unroll
        for (int e = 0; e < E; e++) {
          if ((e & stride) == 0) {
            const float lo = v[j][e];
            const float hi = v[j][e + stride];
            v[j][e] = lo + hi;
            v[j][e + stride] = lo - hi;
          }
        }
      }
#pragma unroll
      for (int ls = 1; ls < 32; ls <<= 1) {
        const bool upper = (lane & ls) != 0;
#pragma unroll
        for (int e = 0; e < E; e++) {
          const float mine = v[j][e];
          const float p = __shfl_xor(mine, ls, 32);
          v[j][e] = upper ? (p - mine) : (mine + p);
        }
      }
      const float inv_sqrt_b = rsqrtf((float)B);
#pragma unroll
      for (int e = 0; e < E; e++) {
        v[j][e] *= inv_sqrt_b;
        if constexpr (!HAS_HA) local_max = fmaxf(local_max, fabsf(v[j][e]));
      }
    }
  }

  // ---- cross-a Kronecker mix via LDS, in column strips (identical to the
  // other Kronecker variants -- see radp_ml8_qrot_kernel's shared comment) ----
  if constexpr (HAS_HA) {
    constexpr int ESTRIP = (E < 8) ? E : 8;
    constexpr int NCHUNK = E / ESTRIP;
    static_assert(E % ESTRIP == 0, "E must be a multiple of ESTRIP");
    constexpr int ROWBUF = MAXAW * NW * 32 * ESTRIP;
    constexpr int HABUF  = (MAXAW * NW) * (MAXAW * NW);
    __shared__ float s_row[ROWBUF];
    __shared__ float s_ha[HABUF];
    for (int idx = threadIdx.x; idx < a_dim * a_dim; idx += blockDim.x) s_ha[idx] = h_a[idx];
    __syncthreads();
#pragma unroll
    for (int c = 0; c < NCHUNK; c++) {
      const int e0 = c * ESTRIP;
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
          // SoA layout: consecutive lanes hit consecutive LDS words (bank =
          // address mod 32 = lane), not the old lane*ESTRIP AoS stride
          // (bank = (lane*ESTRIP) mod 32, an 8..32-way conflict for every
          // ESTRIP in {8,16,32} -- see the file-level note above). Same
          // ROWBUF size, same values, same per-(a,lane,ee) address space --
          // only which word within the a-slot each (lane,ee) maps to moved,
          // so this is bit-identical.
          float* dst = s_row + a * 32 * ESTRIP + lane;
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) dst[ee * 32] = v[j][e0 + ee];
        }
      }
      __syncthreads();
#pragma unroll
      for (int j = 0; j < MAXAW; j++) {
        const int k = wave + j * NW;
        if (k < a_dim) {
          float y[ESTRIP];
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) y[ee] = 0.0f;
          for (int i = 0; i < a_dim; i++) {
            const float h = s_ha[i * a_dim + k];
            const float* src = s_row + i * 32 * ESTRIP + lane;
#pragma unroll
            for (int ee = 0; ee < ESTRIP; ee++) y[ee] = fmaf(h, src[ee * 32], y[ee]);
          }
#pragma unroll
          for (int ee = 0; ee < ESTRIP; ee++) {
            v[j][e0 + ee] = y[ee];
            local_max = fmaxf(local_max, fabsf(y[ee]));
          }
        }
      }
      __syncthreads();
    }
  }

  // ---- amax reduce + scale (unchanged) ----
  local_max = warp_reduce_max<32>(local_max);
  float row_max = local_max;
  if constexpr (NW > 1) {
    __shared__ float s_max[NW];
    if (lane == 0) s_max[wave] = local_max;
    __syncthreads();
    row_max = s_max[0];
#pragma unroll
    for (int w = 1; w < NW; w++) row_max = fmaxf(row_max, s_max[w]);
  }
  const float scale = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
  const float inv_scale = 1.0f / scale;
  if (threadIdx.x == 0) a_scale[row] = scale;

  // ---- single coalesced packed fp8 TILED store (unchanged, LOGICAL order) ----
#pragma unroll
  for (int j = 0; j < MAXAW; j++) {
    const int a = wave + j * NW;
    if (a < a_dim) {
      const int k0 = a * B + lane * E;
#pragma unroll
      for (int e = 0; e < E; e += 4) {
        const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale,
                                                        v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
        *reinterpret_cast<uint32_t*>(a_tiled + radp_tiled_off(row, k0 + e, K)) = packed;
      }
    }
  }
}

}  // namespace ml8qrot

// ═════════════════════════════════════════════════════════════════════════
// Shape dispatch. Mirrors ml8_launch_qrot_v3's own coverage in ml8.cu
// (b_dim in {1024,512,256} Kronecker; {128,64,32} block-Hadamard), except
// NW is chosen to make MAXAW==1 wherever the Kronecker mix is now LDS-
// mediated (see this file's header comment) instead of NW==1 always.
namespace {

enum class RadpQrotDisp {
  NONE,
  KRON_B1024_A5, KRON_B1024_A8, KRON_B512_A16, KRON_B256_A16,
  HAD_B128_NW32_A4, HAD_B128_NW32_A12, HAD_B128_NW16_A34,
};

RadpQrotDisp radp_pick_qrot_shape(bool kronecker, int a_dim, int b_dim) {
  if (kronecker) {
    if (b_dim == 1024 && a_dim <= 5) return RadpQrotDisp::KRON_B1024_A5;
    if (b_dim == 1024 && a_dim <= 8) return RadpQrotDisp::KRON_B1024_A8;
    if (b_dim == 512 && a_dim <= 16) return RadpQrotDisp::KRON_B512_A16;
    if (b_dim == 256 && a_dim <= 16) return RadpQrotDisp::KRON_B256_A16;
    return RadpQrotDisp::NONE;
  }
  if (b_dim == 128) {
    if (a_dim <= 32 * 4) return RadpQrotDisp::HAD_B128_NW32_A4;
    if (a_dim <= 32 * 12) return RadpQrotDisp::HAD_B128_NW32_A12;
    if (a_dim <= 16 * 34) return RadpQrotDisp::HAD_B128_NW16_A34;
    return RadpQrotDisp::NONE;
  }
  // b_dim in {64,32} are declined outright: ml8_fp8_qrot_v3_kernel's own
  // vectorized float4 load (`for (e=0;e<E;e+=4){ *(float4*)(blk+e) ... }`,
  // E=B/32) is only well-defined for E>=4, i.e. B>=128 -- E=1 or 2 (B=32 or
  // 64) reads/writes 4 floats per iteration into a v[][E] array only 1 or 2
  // wide, which overruns the array. That is a latent bug in ml8.cu's own
  // dispatch coverage for these two widths (ml8_launch_qrot_v3 does
  // instantiate them), not something introduced here; this file simply
  // does not re-instantiate a known-broken template. Nothing in the task's
  // required shape set (K=5120, 6144, 17408) or its "at least" minimum
  // needs b_dim<128, so declining cleanly (caller falls back) is correct.
  return RadpQrotDisp::NONE;
}

bool radp_qrot_shape_ok(int rot_kind, int a_dim, int b_dim, int K, RadpQrotDisp* disp) {
  if (a_dim <= 0 || K <= 0 || (long)a_dim * (long)b_dim != K) return false;
  bool kronecker;
  if (rot_kind == RDNA4_ML8_QROT_KIND_KRONECKER) kronecker = true;
  else if (rot_kind == RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD) kronecker = false;
  else return false;
  *disp = radp_pick_qrot_shape(kronecker, a_dim, b_dim);
  return *disp != RadpQrotDisp::NONE;
}

// One macro instantiates+launches the right (B,NW,MAXAW,HAS_HA) kernel for
// a given MODE; keeps the switch below from repeating the 9-way shape table
// for each of the 3 launcher entry points.
#define RADP_DISPATCH(MODE_, IN0, IN1, ROUT, HA, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_, STREAM_) \
  switch (disp) { \
    case RadpQrotDisp::KRON_B1024_A5: \
      ml8qrot::radp_ml8_qrot_kernel<1024, 5, 1, true, MODE_><<<M_, 32 * 5, 0, STREAM_>>>( \
          IN0, IN1, ROUT, HA, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    case RadpQrotDisp::KRON_B1024_A8: \
      ml8qrot::radp_ml8_qrot_kernel<1024, 8, 1, true, MODE_><<<M_, 32 * 8, 0, STREAM_>>>( \
          IN0, IN1, ROUT, HA, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    case RadpQrotDisp::KRON_B512_A16: \
      ml8qrot::radp_ml8_qrot_kernel<512, 16, 1, true, MODE_><<<M_, 32 * 16, 0, STREAM_>>>( \
          IN0, IN1, ROUT, HA, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    case RadpQrotDisp::KRON_B256_A16: \
      ml8qrot::radp_ml8_qrot_kernel<256, 16, 1, true, MODE_><<<M_, 32 * 16, 0, STREAM_>>>( \
          IN0, IN1, ROUT, HA, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    case RadpQrotDisp::HAD_B128_NW32_A4: \
      ml8qrot::radp_ml8_qrot_kernel<128, 32, 4, false, MODE_><<<M_, 32 * 32, 0, STREAM_>>>( \
          IN0, IN1, ROUT, nullptr, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    case RadpQrotDisp::HAD_B128_NW32_A12: \
      ml8qrot::radp_ml8_qrot_kernel<128, 32, 12, false, MODE_><<<M_, 32 * 32, 0, STREAM_>>>( \
          IN0, IN1, ROUT, nullptr, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    case RadpQrotDisp::HAD_B128_NW16_A34: \
      ml8qrot::radp_ml8_qrot_kernel<128, 16, 34, false, MODE_><<<M_, 32 * 16, 0, STREAM_>>>( \
          IN0, IN1, ROUT, nullptr, NORMW, NEPS, ATILED, ASCALE, YOUT, GFIRST, K_, ADIM, M_); \
      break; \
    default: return false; \
  } \
  return true;

}  // namespace

bool rdna4_ml8_qrot_tiled(const float* x, const void* h_a, int rot_kind, int a_dim, int b_dim,
                           const float* norm_w, float norm_eps, int M, int K, uint8_t* A_tiled,
                           float* a_scale, hipStream_t stream, float* y_out) {
  if (M <= 0 || K <= 0 || (K & 15)) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;

  RADP_DISPATCH(ml8qrot::QMODE_PLAIN, x, /*in1=*/nullptr, /*residual_out=*/nullptr, h_a_f, norm_w,
                norm_eps, A_tiled, a_scale, y_out, /*gate_first=*/0, K, a_dim, M, stream)
}

bool rdna4_ml8_qrot_add_tiled(const float* x, const float* residual, float* residual_out,
                               const void* h_a, int rot_kind, int a_dim, int b_dim,
                               const float* norm_w, float norm_eps, int M, int K,
                               uint8_t* A_tiled, float* a_scale, hipStream_t stream,
                               float* y_out) {
  if (M <= 0 || K <= 0 || (K & 15)) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;

  RADP_DISPATCH(ml8qrot::QMODE_ADD, x, residual, residual_out, h_a_f, norm_w, norm_eps, A_tiled,
                a_scale, y_out, /*gate_first=*/0, K, a_dim, M, stream)
}

bool rdna4_ml8_qrot_silu_mul_tiled(const float* gate_up, int gate_first, const void* h_a,
                                    int rot_kind, int a_dim, int b_dim, int M, int N,
                                    uint8_t* A_tiled, float* a_scale, hipStream_t stream) {
  if (M <= 0 || N <= 0 || (N & 15) || !gate_up) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, N, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)N;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;

  RADP_DISPATCH(ml8qrot::QMODE_SILU_FUSED, gate_up, /*in1=*/nullptr, /*residual_out=*/nullptr,
                h_a_f, /*norm_w=*/nullptr, /*norm_eps=*/0.0f, A_tiled, a_scale, /*y_out=*/nullptr,
                gate_first, N, a_dim, M, stream)
}

bool rdna4_ml8_qrot_silu_mul_split_tiled(const float* gate, const float* up, const void* h_a,
                                          int rot_kind, int a_dim, int b_dim, int M, int N,
                                          uint8_t* A_tiled, float* a_scale, hipStream_t stream) {
  if (M <= 0 || N <= 0 || (N & 15) || !gate || !up) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, N, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)N;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;

  RADP_DISPATCH(ml8qrot::QMODE_SILU_SPLIT, gate, up, /*residual_out=*/nullptr, h_a_f,
                /*norm_w=*/nullptr, /*norm_eps=*/0.0f, A_tiled, a_scale, /*y_out=*/nullptr,
                /*gate_first=*/0, N, a_dim, M, stream)
}

// bf16-INPUT split silu*mul (LLAMA_ACT_BF16 prefill: ffn_up/ffn_gate GEMM
// outputs are bf16 -- see radp_ml8_qrot_silu_mul_split_bf16_kernel's own
// comment for why this halves the bytes the fp32 variant is bandwidth-
// bound on). Same shape dispatch table as rdna4_ml8_qrot_silu_mul_split_tiled
// (radp_qrot_shape_ok); gate/up are raw bf16 bit patterns (uint16_t, or any
// type whose bits ARE bf16 -- __hip_bfloat16/nv_bfloat16 are bit-identical
// to a plain uint16_t for this purpose, see radp_bf16_bits_to_f32).
bool rdna4_ml8_qrot_silu_mul_split_tiled_bf16(const void* gate_bf16, const void* up_bf16,
                                               const void* h_a, int rot_kind, int a_dim, int b_dim,
                                               int M, int N, uint8_t* A_tiled, float* a_scale,
                                               hipStream_t stream) {
  if (M <= 0 || N <= 0 || (N & 15) || !gate_bf16 || !up_bf16) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, N, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)N;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;
  const uint16_t* g16 = (const uint16_t*)gate_bf16;
  const uint16_t* u16 = (const uint16_t*)up_bf16;

  using namespace ml8qrot;
  switch (disp) {
    case RadpQrotDisp::KRON_B1024_A5:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<1024, 5, 1, true>
          <<<M, 32 * 5, 0, stream>>>(g16, u16, h_a_f, A_tiled, a_scale, N, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B1024_A8:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<1024, 8, 1, true>
          <<<M, 32 * 8, 0, stream>>>(g16, u16, h_a_f, A_tiled, a_scale, N, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B512_A16:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<512, 16, 1, true>
          <<<M, 32 * 16, 0, stream>>>(g16, u16, h_a_f, A_tiled, a_scale, N, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B256_A16:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<256, 16, 1, true>
          <<<M, 32 * 16, 0, stream>>>(g16, u16, h_a_f, A_tiled, a_scale, N, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A4:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<128, 32, 4, false>
          <<<M, 32 * 32, 0, stream>>>(g16, u16, nullptr, A_tiled, a_scale, N, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A12:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<128, 32, 12, false>
          <<<M, 32 * 32, 0, stream>>>(g16, u16, nullptr, A_tiled, a_scale, N, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW16_A34:
      radp_ml8_qrot_silu_mul_split_bf16_kernel<128, 16, 34, false>
          <<<M, 32 * 16, 0, stream>>>(g16, u16, nullptr, A_tiled, a_scale, N, a_dim, M);
      break;
    default: return false;
  }
  return true;
}

// bf16-INPUT residual add (see radp_ml8_qrot_add_bf16_kernel's own comment).
// residual_out, if requested, is still written fp32 -- keeping the residual
// stream itself bf16 end-to-end is a later step, not this kernel's concern.
bool rdna4_ml8_qrot_add_tiled_bf16(const void* x_bf16, const void* residual_bf16,
                                    float* residual_out, const void* h_a, int rot_kind, int a_dim,
                                    int b_dim, const float* norm_w, float norm_eps, int M, int K,
                                    uint8_t* A_tiled, float* a_scale, hipStream_t stream,
                                    float* y_out) {
  if (M <= 0 || K <= 0 || (K & 15) || !x_bf16) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;
  const uint16_t* x16 = (const uint16_t*)x_bf16;
  const uint16_t* r16 = (const uint16_t*)residual_bf16;

  using namespace ml8qrot;
  switch (disp) {
    case RadpQrotDisp::KRON_B1024_A5:
      radp_ml8_qrot_add_bf16_kernel<1024, 5, 1, true><<<M, 32 * 5, 0, stream>>>(
          x16, r16, residual_out, h_a_f, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B1024_A8:
      radp_ml8_qrot_add_bf16_kernel<1024, 8, 1, true><<<M, 32 * 8, 0, stream>>>(
          x16, r16, residual_out, h_a_f, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B512_A16:
      radp_ml8_qrot_add_bf16_kernel<512, 16, 1, true><<<M, 32 * 16, 0, stream>>>(
          x16, r16, residual_out, h_a_f, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B256_A16:
      radp_ml8_qrot_add_bf16_kernel<256, 16, 1, true><<<M, 32 * 16, 0, stream>>>(
          x16, r16, residual_out, h_a_f, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A4:
      radp_ml8_qrot_add_bf16_kernel<128, 32, 4, false><<<M, 32 * 32, 0, stream>>>(
          x16, r16, residual_out, nullptr, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A12:
      radp_ml8_qrot_add_bf16_kernel<128, 32, 12, false><<<M, 32 * 32, 0, stream>>>(
          x16, r16, residual_out, nullptr, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW16_A34:
      radp_ml8_qrot_add_bf16_kernel<128, 16, 34, false><<<M, 32 * 16, 0, stream>>>(
          x16, r16, residual_out, nullptr, norm_w, norm_eps, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    default: return false;
  }
  return true;
}

#undef RADP_DISPATCH

namespace {
// head_dim must divide evenly into E=B/32 groups (every lane's own E-wide
// slice must lie entirely inside one head, never straddle a head
// boundary) and lanes_per_head=head_dim/E must be a power of two dividing
// 32 (so a plain __shfl_xor sub-group reduction covers exactly the lanes
// sharing a head). True for the documented production shape
// (head_dim=128, B=1024 -> E=32, lanes_per_head=4); declined otherwise so
// the caller falls back rather than getting silently wrong per-head norms.
bool radp_gated_lanes_per_head(int head_dim, int B, int* lanes_per_head) {
  const int E = B / 32;
  if (head_dim <= 0 || E <= 0 || head_dim % E != 0) return false;
  const int lph = head_dim / E;
  if (lph <= 0 || (lph & (lph - 1)) != 0 || (32 % lph) != 0) return false;
  *lanes_per_head = lph;
  return true;
}
}  // namespace

bool rdna4_ml8_qrot_gated_norm_tiled(const float* o, const float* z, size_t z_nb1, size_t z_nb2,
                                      const float* norm_w, float norm_eps, int head_dim, int n_heads,
                                      const void* h_a, int rot_kind, int a_dim, int b_dim, int M,
                                      uint8_t* A_tiled, float* a_scale, hipStream_t stream,
                                      float* y_out) {
  if (M <= 0 || head_dim <= 0 || n_heads <= 0 || !o || !z || !norm_w) return false;
  const int K = head_dim * n_heads;
  if (K <= 0 || (K & 15)) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;
  const char* z_bytes = (const char*)z;
  int lph = 0;

  using namespace ml8qrot;
  switch (disp) {
    case RadpQrotDisp::KRON_B1024_A5:
      if (!radp_gated_lanes_per_head(head_dim, 1024, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<1024, 5, 1, true><<<M, 32 * 5, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B1024_A8:
      if (!radp_gated_lanes_per_head(head_dim, 1024, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<1024, 8, 1, true><<<M, 32 * 8, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B512_A16:
      if (!radp_gated_lanes_per_head(head_dim, 512, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<512, 16, 1, true><<<M, 32 * 16, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B256_A16:
      if (!radp_gated_lanes_per_head(head_dim, 256, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<256, 16, 1, true><<<M, 32 * 16, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A4:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<128, 32, 4, false><<<M, 32 * 32, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A12:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<128, 32, 12, false><<<M, 32 * 32, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW16_A34:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<128, 16, 34, false><<<M, 32 * 16, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    default: return false;
  }
  return true;
}

// Attention output gating variant (GATE=1): o * sigmoid(z), no norm. See the kernel comment.
bool rdna4_ml8_qrot_gate_sigmoid_tiled(const float* o, const float* z, size_t z_nb1, size_t z_nb2,
                                        int head_dim, int n_heads,
                                        const void* h_a, int rot_kind, int a_dim, int b_dim, int M,
                                        uint8_t* A_tiled, float* a_scale, hipStream_t stream) {
  const float* norm_w = nullptr; const float norm_eps = 0.0f; float* y_out = nullptr;
  if (M <= 0 || head_dim <= 0 || n_heads <= 0 || !o || !z) return false;
  const int K = head_dim * n_heads;
  if (K <= 0 || (K & 15)) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;
  const char* z_bytes = (const char*)z;
  int lph = 0;

  using namespace ml8qrot;
  switch (disp) {
    case RadpQrotDisp::KRON_B1024_A5:
      if (!radp_gated_lanes_per_head(head_dim, 1024, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<1024, 5, 1, true, 1><<<M, 32 * 5, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B1024_A8:
      if (!radp_gated_lanes_per_head(head_dim, 1024, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<1024, 8, 1, true, 1><<<M, 32 * 8, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B512_A16:
      if (!radp_gated_lanes_per_head(head_dim, 512, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<512, 16, 1, true, 1><<<M, 32 * 16, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B256_A16:
      if (!radp_gated_lanes_per_head(head_dim, 256, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<256, 16, 1, true, 1><<<M, 32 * 16, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A4:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<128, 32, 4, false, 1><<<M, 32 * 32, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A12:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<128, 32, 12, false, 1><<<M, 32 * 32, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW16_A34:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_kernel<128, 16, 34, false, 1><<<M, 32 * 16, 0, stream>>>(
          o, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph, nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    default: return false;
  }
  return true;
}

// bf16 + libr4d-head-permuted `o` variant (see
// radp_ml8_qrot_gated_norm_r4d_kernel's own comment for the full contract
// and mt_gdn_r4d.cu's r4d_gdn_perm_head for the permutation formula this
// exists to consume without an extra unpermute+cast pass). Same shape
// dispatch / head_dim-alignment precondition as rdna4_ml8_qrot_gated_norm_tiled.
bool rdna4_ml8_qrot_gated_norm_tiled_r4d(const void* o_bf16, size_t o_nb_head, size_t o_nb_tok,
                                          const int32_t* head_src, const float* z, size_t z_nb1,
                                          size_t z_nb2, const float* norm_w, float norm_eps,
                                          int head_dim, int n_heads, const void* h_a, int rot_kind,
                                          int a_dim, int b_dim, int M, uint8_t* A_tiled,
                                          float* a_scale, hipStream_t stream, float* y_out) {
  if (M <= 0 || head_dim <= 0 || n_heads <= 0 || !o_bf16 || !z || !norm_w) return false;
  const int K = head_dim * n_heads;
  if (K <= 0 || (K & 15)) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;

  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  (void)hipMemsetAsync(A_tiled, 0, at_bytes, stream);
  const float* h_a_f = (const float*)h_a;
  const uint16_t* o16 = (const uint16_t*)o_bf16;
  const char* z_bytes = (const char*)z;
  int lph = 0;

  using namespace ml8qrot;
  switch (disp) {
    case RadpQrotDisp::KRON_B1024_A5:
      if (!radp_gated_lanes_per_head(head_dim, 1024, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<1024, 5, 1, true><<<M, 32 * 5, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B1024_A8:
      if (!radp_gated_lanes_per_head(head_dim, 1024, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<1024, 8, 1, true><<<M, 32 * 8, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B512_A16:
      if (!radp_gated_lanes_per_head(head_dim, 512, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<512, 16, 1, true><<<M, 32 * 16, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B256_A16:
      if (!radp_gated_lanes_per_head(head_dim, 256, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<256, 16, 1, true><<<M, 32 * 16, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          h_a_f, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A4:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<128, 32, 4, false><<<M, 32 * 32, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A12:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<128, 32, 12, false><<<M, 32 * 32, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW16_A34:
      if (!radp_gated_lanes_per_head(head_dim, 128, &lph)) return false;
      radp_ml8_qrot_gated_norm_r4d_kernel<128, 16, 34, false><<<M, 32 * 16, 0, stream>>>(
          o16, o_nb_head, o_nb_tok, head_src, z_bytes, z_nb1, z_nb2, norm_w, norm_eps, head_dim, lph,
          nullptr, A_tiled, a_scale, y_out, K, a_dim, M);
      break;
    default: return false;
  }
  return true;
}

bool rdna4_ml8_qrot_oracle_rowmajor(const float* x, const void* h_a, int rot_kind, int a_dim,
                                     int b_dim, const float* norm_w, float norm_eps, int M, int K,
                                     uint8_t* a_fp8_rowmajor, float* a_scale, hipStream_t stream) {
  if (M <= 0 || K <= 0) return false;
  RadpQrotDisp disp;
  if (!radp_qrot_shape_ok(rot_kind, a_dim, b_dim, K, &disp)) return false;
  const float* h_a_f = (const float*)h_a;
  using namespace ml8qrot;
  switch (disp) {
    case RadpQrotDisp::KRON_B1024_A5:
      ml8_fp8_qrot_v3_kernel<1024, 1, 5, true>
          <<<M, 32, 0, stream>>>(x, h_a_f, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B1024_A8:
      ml8_fp8_qrot_v3_kernel<1024, 1, 8, true>
          <<<M, 32, 0, stream>>>(x, h_a_f, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B512_A16:
      ml8_fp8_qrot_v3_kernel<512, 1, 16, true>
          <<<M, 32, 0, stream>>>(x, h_a_f, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    case RadpQrotDisp::KRON_B256_A16:
      ml8_fp8_qrot_v3_kernel<256, 1, 16, true>
          <<<M, 32, 0, stream>>>(x, h_a_f, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    // HAD_B128_*: instantiate the EXACT SAME (B,NW,MAXAW) as
    // radp_pick_qrot_shape picked for this a_dim, not just "some" correct
    // instantiation. Reduction results (amax, the FWHT butterfly) are
    // mathematically order-independent, but a DIFFERENT NW/MAXAW template
    // instantiation is different generated code, and this toolchain's
    // default fp-contraction is free to fuse multiply-adds differently
    // across differently-shaped instantiations -- that residual codegen
    // freedom (not the algorithm) was the source of the previously
    // reported ~275/71M-byte, 1-ulp-scale-only mismatch for K=17408.
    // Locking the oracle to the identical instantiation removes that
    // degree of freedom entirely, so plain/add are bit-identical by
    // construction (same compiled kernel body, only the store differs).
    case RadpQrotDisp::HAD_B128_NW32_A4:
      ml8_fp8_qrot_v3_kernel<128, 32, 4, false>
          <<<M, 32 * 32, 0, stream>>>(x, nullptr, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW32_A12:
      ml8_fp8_qrot_v3_kernel<128, 32, 12, false>
          <<<M, 32 * 32, 0, stream>>>(x, nullptr, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    case RadpQrotDisp::HAD_B128_NW16_A34:
      ml8_fp8_qrot_v3_kernel<128, 16, 34, false>
          <<<M, 32 * 16, 0, stream>>>(x, nullptr, norm_w, norm_eps, a_fp8_rowmajor, a_scale, K, a_dim, M);
      break;
    default: return false;
  }
  return true;
}
