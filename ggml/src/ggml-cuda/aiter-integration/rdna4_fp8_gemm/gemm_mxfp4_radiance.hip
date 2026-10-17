// gemm_mxfp4_radiance.hip -- radiance's MXFP4 x FP8 GEMM for ggml MXFP4
// weights. See gemm_mxfp4_radiance.h for layout and numerics.
//
// PORTED from ~/GitHub/ggz14-vllm-mxfp4/radiance_mxfp4_fp8.hip:
//   :45-75      kMag fold table                       -- verbatim
//   :80-86,217-224 tile #defines                      -- verbatim
//   :240-460    radiance_mxfp4_fp8_gemm_folded        -- WPERM=true branch
//               verbatim; the row-order (WPERM=false) staging branch is
//               dropped (our weights are always repacked to fragment order);
//               the epilogue stores through rmx_store<OutT> instead of
//               (__bf16) -- the ONLY change, same fp32 value.
//   :775-1029   radiance_mxfp4_fp8_gemm_decode        -- same treatment
//   :1030-1116  DEC_* constants, decode_bk64, split_k_for -- verbatim
//   :1136-1304  launch_impl's wref branch              -- verbatim policy;
//               env vars renamed RDNA4_MXFP4_RADIANCE_*, decode scratch
//               passed in by the caller instead of set_decode_scratch(),
//               decode band default 64 (serve-mxfp4.sh's setting) instead
//               of 0, counters memset per call (pool memory, not a
//               zeroed-once torch buffer).
// The pb1/pb2/astride partition select is kept verbatim in both kernels;
// the launcher always passes the disabling values, as radiance's own
// launch() does.
//
// New here: rmx_repack_kernel (ggml block_mxfp4 -> WPERM/Ws/Wref, the
// equivalent of radiance_mxfp4.py's weight_scale.T + make_row_ref +
// permute_w, plus ggml's j/j+16 nibble split -> even/odd k) and
// rmx_quant_a_kernel (radiance_add_rms_quant's scale/convert numerics on a
// plain f32 row, no norm).
//
// Optional hi/lo activations (A_lo != null): the producer also writes
// lo = e4m3(x/s - hi) with the same scale s, so x ~= (hi + lo) * s. The GEMM
// then runs two WMMAs per k-step into ONE accumulator, reusing the same
// weight fragment: acc += W*hi, acc += W*lo. Error drops from ~0.026 to
// ~0.0006 rel RMS. The weight path is shared with the single-A kernels.
#include "gemm_mxfp4_radiance.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdlib>

typedef float rmx_floatx8 __attribute__((ext_vector_type(8)));
typedef int rmx_int2 __attribute__((ext_vector_type(2)));
typedef unsigned int rmx_uint4 __attribute__((ext_vector_type(4)));
typedef unsigned int rmx_uint2 __attribute__((ext_vector_type(2)));

// ── radiance_mxfp4_fp8.hip:80-86 / 217-224, verbatim ──────────────────────
#define BK 64
#define PAD 8
#define ASTR (BK + PAD)
#define WSTR (BK + PAD)
#define TM 4
#define WM 4
#define WN 2
#define NWAVE (WM * WN)
#define NTHREADS (NWAVE * 32)
#define BMF (WM * TM * 16)
#define BNF_OF(tn) (WN * (tn) * 16)

// ── radiance_mxfp4_fp8.hip:775-776 / 1030-1032 / 1118, verbatim ───────────
#define DEC_MTILE 16
#define DEC_PAD 8
#define DEC_MAX_N 36864
#define DEC_KS 4
#define DEC_DWN 8
#define DEC_MAX_TM 8

// Output store: f32 stores the value unchanged; bf16 rounds the SAME value
// RNE (radiance's own (__bf16) cast).
template <typename OutT>
static __device__ __forceinline__ void rmx_store(OutT* p, float v) {
    if constexpr (sizeof(OutT) == 4) {
        *p = v;
    } else {
        uint32_t u = __builtin_bit_cast(uint32_t, v);
        if ((u & 0x7fffffffu) > 0x7f800000u) {
            *p = (OutT)((u >> 16) | 0x0040u);
        } else {
            u += 0x7fffu + ((u >> 16) & 1u);
            *p = (OutT)(u >> 16);
        }
    }
}

// Grouped (MUL_MAT_ID) launch descriptor, passed by value to the GROUPED kernel instantiations.
// All pointers are device pointers. padoff[e] is expert e's first row in the expert-grouped activation
// buffer (every segment is padded to a multiple of 16 rows); bounds[e]..bounds[e+1] indexes ids_dst for
// expert e's routed rows (ids_dst = dst row = token*n_used + slot). lst_* is the work list the plan kernel
// built for this kernel family (decode: lst_e only; atiled: lst_e + lst_mt = m-tile within the expert).
struct rmx_moe {
  const int *padoff;
  const int *bounds;
  const int *ids_dst;
  const int *lst_e;
  const int *lst_mt;
  const int *lst_n;
  long wstride;      // bytes between experts' packed matrices
  long ldc;          // dst row length in floats (N for one matrix, n_mats*N for a fused gate;up pair)
  long coff;         // column offset of this matrix's N outputs inside a dst row
  // Row mode (decode kernel only, tiny n_tok): no routing tables at all. Block row li is routed row li
  // (= token*n_used + slot): its expert is ids_rows[li], its activation row is row li of the row-major hi/lo
  // buffer, and its dst row is li. nullptr = the expert-grouped mode above.
  const int *ids_rows;
  int n_as;
  // Weight-pager expert pointers (mul_mat_id expert_ptrs): when non-null, expert e's matrix is at
  // (const uchar*)wptrs[e] + matoff (+ ws_off / wref_off for the Ws / Wref sub-buffers) instead of
  // base + e*wstride, so slots need no uniform stride.
  const void *const *wptrs;
  long matoff;
  long ws_off;
  long wref_off;
};

#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)

// ── radiance_mxfp4_fp8.hip:45-75, verbatim (see the source for why the
// table runs into e4m3 subnormals) ─────────────────────────────────────────
__device__ __constant__ unsigned int kMag[16][2] = {
  {0x3c383000u, 0x4c484440u},
  {0x34302800u, 0x44403c38u},
  {0x2c282000u, 0x3c383430u},
  {0x24201800u, 0x34302c28u},
  {0x1c181000u, 0x2c282420u},
  {0x14100800u, 0x24201c18u},
  {0x0c080400u, 0x1c181410u},
  {0x06040200u, 0x14100c08u},
  {0x03020100u, 0x0c080604u},
  {0x01010000u, 0x06040302u},
  {0x01000000u, 0x03020101u},
  {0x00000000u, 0x01010100u},
  {0x00000000u, 0x01000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
  {0x00000000u, 0x00000000u},
};

// ── radiance_mxfp4_fp8.hip:240-460, WPERM=true ────────────────────────────
template <int TN, bool EPIFAST, bool HILO, typename OutT>
__global__ __launch_bounds__(NTHREADS) void rmx_gemm_folded(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ A_lo,
    const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, OutT *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  __shared__ unsigned char sA[BMF * ASTR];
  __shared__ unsigned char sW[WN * TN * 16 * WSTR];
  __shared__ unsigned char sA_lo[HILO ? BMF * ASTR : 1];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  constexpr int BNF_T = WN * TN * 16;
  const int m0 = blockIdx.y * BMF, n0 = blockIdx.x * BNF_T;
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    if constexpr (HILO) A_lo += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  rmx_floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    const unsigned char *__restrict__ Ab = A + (size_t)m0 * K + k0;
#pragma unroll
    for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
      const int idx = off + tid * 16;
      const int r = idx / BK, c = idx % BK;
      const int rc = r < M - 1 - m0 ? r : M - 1 - m0;              // clamp, never predicate
      *(rmx_uint4 *)(&sA[r * ASTR + c]) = *(const rmx_uint4 *)(Ab + (rc * K + c));
    }
    if constexpr (HILO) {
      const unsigned char *__restrict__ Abl = A_lo + (size_t)m0 * K + k0;
#pragma unroll
      for (int off = 0; off < BMF * BK; off += NTHREADS * 16) {
        const int idx = off + tid * 16;
        const int r = idx / BK, c = idx % BK;
        const int rc = r < M - 1 - m0 ? r : M - 1 - m0;
        *(rmx_uint4 *)(&sA_lo[r * ASTR + c]) = *(const rmx_uint4 *)(Abl + (rc * K + c));
      }
    }
    {
      constexpr int KSTEPS_T = BK / 16, NTILES_T = BNF_T / 16;
      constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
      constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
      const unsigned int *__restrict__ Wp = (const unsigned int *)W;
      const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
      for (int off = 0; off < TOT_SLOTS; off += NTHREADS * WSLOTS) {
        const int sl = off + tid * WSLOTS;
        const int lane_ = sl & 31, rest = sl >> 5;
        const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
        const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
        const int gn = n0 + r;
        const int gc = gn < N - WSLOTS ? gn : N - WSLOTS;
        const int lanec = (lane_ & 16) | (gc & 15);
        const unsigned int *src = &Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec];
        unsigned int wq[WSLOTS];
        if constexpr (WSLOTS == 4) {
          const rmx_uint4 v = *(const rmx_uint4 *)src;
          wq[0] = v[0]; wq[1] = v[1]; wq[2] = v[2]; wq[3] = v[3];
        } else {
          const rmx_uint2 v = *(const rmx_uint2 *)src;
          wq[0] = v[0]; wq[1] = v[1];
        }
        const int blk = (k0 + kst * 16) / 32;
        unsigned int refw, wsw;
        if constexpr (WSLOTS == 4) {
          refw = *(const unsigned int *)(Wref + gc);
          wsw = *(const unsigned int *)(Ws + (size_t)blk * N + gc);
        } else {
          refw = *(const unsigned short *)(Wref + gc);
          wsw = *(const unsigned short *)(Ws + (size_t)blk * N + gc);
        }
#pragma unroll
        for (int q = 0; q < WSLOTS; ++q) {
          int d = (int)((refw >> (8 * q)) & 0xFF) - (int)((wsw >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int ev = wq[q] & 0x0F0F0F0Fu;
          const unsigned int od = (wq[q] >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(rmx_uint2 *)(&sW[(r + q) * WSTR + kloc]) =
              rmx_uint2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                        __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    }
    __syncthreads();

#pragma unroll
    for (int step = 0; step < BK / 16; ++step) {
      const int kk = step * 16 + kb8;
      rmx_int2 af[TM], wf[TN], al[HILO ? TM : 1];
#pragma unroll
      for (int i = 0; i < TM; ++i) {
        const unsigned char *p = &sA[(wm * TM * 16 + i * 16 + col) * ASTR + kk];
        af[i][0] = *(const int *)p; af[i][1] = *(const int *)(p + 4);
      }
      if constexpr (HILO) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
          const unsigned char *p = &sA_lo[(wm * TM * 16 + i * 16 + col) * ASTR + kk];
          al[i][0] = *(const int *)p; al[i][1] = *(const int *)(p + 4);
        }
      }
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * WSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf[j], acc[i][j]);
          if constexpr (HILO)
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(al[i], wf[j], acc[i][j]);
        }
    }
    __syncthreads();
  }

  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  if constexpr (EPIFAST) {
    OutT *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            rmx_store(&Cb[r * N + n], acc[i][j][e] * rf[j] * Asb[r]);
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) rmx_store(&C[(size_t)m * N + n], acc[i][j][e] * rf[j] * As[m]);
      }
}

// ── radiance_mxfp4_fp8.hip:778-1029, WPERM=true ───────────────────────────
// ── radiance_mxfp4_fp8.hip:524-758, radiance_mxfp4_fp8_gemm_atiled, WPERM=true ─
// A-tiled prefill kernel: A fragments are loaded straight from the tiled global
// buffer into registers (no A LDS); only the W tile goes through LDS. HILO loads
// the lo fragments from the tiled A_lo buffer and does a second WMMA with the same
// wf into the same accumulator. Epilogue identical to rmx_gemm_folded.
static __device__ __forceinline__ void rmx_lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int TN, int LBK, bool HILO, bool GROUPED, typename OutT>
__global__ __launch_bounds__(NTHREADS) void rmx_gemm_atiled(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ AT_lo,
    const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, OutT *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride, rmx_moe moe) {
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + PAD;
  constexpr int BNF_T = WN * TN * 16;
  __shared__ unsigned char sW[BNF_T * LWSTR];
  __shared__ unsigned int sMag[32];   // kMag, 128 B: ds_load keeps the lookup off the loadcnt chain

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / WN, wn = wave % WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  int m0 = blockIdx.y * BMF;
  const int n0 = blockIdx.x * BNF_T;
  int e_ = 0;
  if constexpr (GROUPED) {
    // work list entry = (expert, m-tile); blocks past the list exit before any barrier (block-uniform)
    const int li = blockIdx.y;
    if (li >= *moe.lst_n) return;
    e_ = moe.lst_e[li];
    m0 = moe.lst_mt[li] * BMF;
    M = moe.bounds[e_ + 1] - moe.bounds[e_];
    const long po = moe.padoff[e_];
    AT += po * K;
    if constexpr (HILO) AT_lo += po * K;
    As += po;
    if (moe.wptrs != nullptr) {
      const unsigned char *wb = (const unsigned char *)moe.wptrs[e_] + moe.matoff;
      W = wb; Ws = wb + moe.ws_off; Wref = wb + moe.wref_off;
    } else {
      const long wo = (long)e_ * moe.wstride;
      W += wo; Ws += wo; Wref += wo;
    }
  }
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    AT += (size_t)p_ * astride;
    if constexpr (HILO) AT_lo += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  if (tid < 32) sMag[tid] = ((const unsigned int *)kMag)[tid];
  rmx_lds_barrier();

  // A tile bases, one per M-fragment: wave-uniform (SGPR). Tile-granular clamp, never
  // predicate: rows of a clamped tile only reach accumulators the epilogue drops.
  const unsigned char *abase[TM];
  const unsigned char *abase_lo[HILO ? TM : 1];
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    int mt = (m0 >> 4) + wm * TM + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    const unsigned int tb = (unsigned int)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
    abase[i] = AT + tb;
    if constexpr (HILO) abase_lo[i] = AT_lo + tb;
  }
  const unsigned int aoff = lane * 8;

  // W staging geometry, hoisted out of the K loop (WPERM fragment order; see the folded
  // kernel). WSLOTS = 4 (16 B) when the tile has the slots for it, else 2 (8 B).
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= NTHREADS * 4) ? 4 : 2;
  constexpr int NW = TOT_SLOTS / (NTHREADS * WSLOTS);
  static_assert(NW * NTHREADS * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * NTHREADS * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wcol[it] = kst * 16 + (lane_ >> 4) * 8;
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);
    wkst[it] = kst;
  }
  unsigned int wref[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    if constexpr (WSLOTS == 4) wref[it] = *(const unsigned int *)(Wref + wrc[it]);
    else                       wref[it] = *(const unsigned short *)(Wref + wrc[it]);
  }

  rmx_floatx8 acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += LBK) {
    // A fragments straight from global, one SADDR clause, consumed at the WMMAs.
    rmx_int2 af[TM][NS], al[HILO ? TM : 1][HILO ? NS : 1];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const rmx_int2 *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    if constexpr (HILO) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int st = 0; st < NS; ++st)
          al[i][st] = *(const rmx_int2 *)(abase_lo[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));
    }

    // ---- W tile -> LDS, folded to e4m3 (kMag[d] via v_perm) ----
    unsigned int wq[NW][WSLOTS], wsv[NW];
    const unsigned int *__restrict__ Wp = (const unsigned int *)W;
    const int kstep0 = k0 / 16;
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const unsigned int *src = &Wp[((size_t)(wrc[it] >> 4) * ksteps_g + kstep0 + wkst[it]) * 32 + wlanec[it]];
      if constexpr (WSLOTS == 4) {
        const rmx_uint4 v = *(const rmx_uint4 *)src;
        wq[it][0] = v[0]; wq[it][1] = v[1]; wq[it][2] = v[2]; wq[it][3] = v[3];
      } else {
        const rmx_uint2 v = *(const rmx_uint2 *)src;
        wq[it][0] = v[0]; wq[it][1] = v[1];
      }
      const int blk = (k0 + wkst[it] * 16) / 32;
      if constexpr (WSLOTS == 4) wsv[it] = *(const unsigned int *)(Ws + (size_t)blk * N + wrc[it]);
      else                       wsv[it] = *(const unsigned short *)(Ws + (size_t)blk * N + wrc[it]);
    }
#pragma unroll
    for (int it = 0; it < NW; ++it)
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        int d = (int)((wref[it] >> (8 * q)) & 0xFF) - (int)((wsv[it] >> (8 * q)) & 0xFF);
        d = d < 0 ? 0 : (d > 15 ? 15 : d);
        const rmx_uint2 t = *(const rmx_uint2 *)&sMag[d * 2];
        const unsigned int wqq = wq[it][q];
        const unsigned int ev = wqq & 0x0F0F0F0Fu;
        const unsigned int od = (wqq >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t[1], t[0], ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t[1], t[0], od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        *(rmx_uint2 *)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) =
            rmx_uint2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
      }
    rmx_lds_barrier();

#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      rmx_int2 wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
          if constexpr (HILO)
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(al[i][step], wf[j], acc[i][j]);
        }
    }
    rmx_lds_barrier();
  }

  // Epilogue: same as rmx_gemm_folded's EPIFAST path (grouped: always the scatter path below).
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? __int_as_float((int)Wref[n] << 23) : 0.f;
  }
  if constexpr (!GROUPED) {
    OutT *__restrict__ Cb = C + (size_t)(m0 + wm * TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * TM * 16;
    const bool full = (m0 + wm * TM * 16 + (TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            rmx_store(&Cb[r * N + n], acc[i][j][e] * rf[j] * Asb[r]);
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) {
          size_t crow = (size_t)m * N;
          if constexpr (GROUPED) crow = (size_t)moe.ids_dst[moe.bounds[e_] + m] * moe.ldc + moe.coff;
          rmx_store(&C[crow + n], acc[i][j][e] * rf[j] * As[m]);
        }
      }
}

template <bool NT, typename T>
static __device__ __forceinline__ T rmx_ld_w(const T *p) {
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}
static __device__ __forceinline__ unsigned int rmx_wq_dec(rmx_uint2 v, int q) { return q ? v[1] : v[0]; }

template <int DWN, int DBK, int DKS, int DTM, bool NT, bool HILO, bool GROUPED, typename OutT>
__global__ __launch_bounds__(DWN * 32) void rmx_gemm_decode(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ A_lo,
    const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    OutT *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride, rmx_moe moe) {
  constexpr int BND = DWN * 16;
  constexpr int DASTR = DBK + DEC_PAD, DWSTR = DBK + DEC_PAD;
  constexpr int DNTHREADS = DWN * 32;
  __shared__ unsigned char sA[DEC_MTILE * DTM * DASTR];
  __shared__ unsigned char sA_lo[HILO ? DEC_MTILE * DTM * DASTR : 1];
  __shared__ unsigned char sW[BND * DWSTR];
  __shared__ int s_last;

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int n0 = blockIdx.x * BND;
  const int ks = blockIdx.z;
  int e_ = 0;
  int li_ = 0;
  if constexpr (GROUPED) {
    // work list entry = expert; blocks past the list exit before any barrier (block-uniform)
    const int li = blockIdx.y;
    li_ = li;
    long po;
    if (moe.ids_rows != nullptr) {
      // row mode: one routed row per block row, M = 1
      e_ = moe.ids_rows[li];
      if (e_ < 0 || e_ >= moe.n_as) return;
      M = 1;
      po = li;
      P += (size_t)li * DKS * N;
    } else {
      if (li >= *moe.lst_n) return;
      e_ = moe.lst_e[li];
      M = moe.bounds[e_ + 1] - moe.bounds[e_];
      po = moe.padoff[e_];
      // per-list-entry split-K partial region
      P += (size_t)li * DKS * (DEC_MTILE * DTM) * N;
    }
    A += po * K;
    if constexpr (HILO) A_lo += po * K;
    As += po;
    if (moe.wptrs != nullptr) {
      const unsigned char *wb = (const unsigned char *)moe.wptrs[e_] + moe.matoff;
      W = wb; Ws = wb + moe.ws_off; Wref = wb + moe.wref_off;
    } else {
      const long wo = (long)e_ * moe.wstride;
      W += wo; Ws += wo; Wref += wo;
    }
    cnt += (size_t)li * gridDim.x;
  }
  auto crow = [&](int m) -> size_t {
    if constexpr (GROUPED) {
      if (moe.ids_rows != nullptr) return (size_t)li_ * moe.ldc + moe.coff;
      return (size_t)moe.ids_dst[moe.bounds[e_] + m] * moe.ldc + moe.coff;
    } else return (size_t)m * N;
  };
  // 16-row tiles this block actually has rows for (block-uniform). The ungrouped kernel always runs DTM.
  const int ntm = GROUPED ? min(DTM, (M + DEC_MTILE - 1) / DEC_MTILE) : DTM;
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    A += (size_t)p_ * astride;
    if constexpr (HILO) A_lo += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }

  const int slabs = (K + DBK - 1) / DBK;
  const int spb = (slabs + DKS - 1) / DKS;
  const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

  rmx_floatx8 acc[DTM];
#pragma unroll
  for (int i = 0; i < DTM; ++i)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;

  auto stage_a = [&](int k0) {
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * ntm * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;                  // clamp, never predicate
        *(rmx_uint4 *)(&sA[r * DASTR + c]) = *(const rmx_uint4 *)(A + (size_t)rc * K + k0 + c);
      }
    }
  };
  auto stage_a_lo = [&](int k0) {
#pragma unroll
    for (int off = 0; off < DEC_MTILE * DTM * DBK; off += DNTHREADS * 16) {
      const int idx = off + tid * 16;
      if (idx < DEC_MTILE * ntm * DBK) {
        const int r = idx / DBK, c = idx % DBK;
        const int rc = r < M - 1 ? r : M - 1;
        *(rmx_uint4 *)(&sA_lo[r * DASTR + c]) = *(const rmx_uint4 *)(A_lo + (size_t)rc * K + k0 + c);
      }
    }
  };
  auto stage_w_wperm = [&](int k0) {
    constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
    const unsigned int *__restrict__ Wp = (const unsigned int *)W;
    const int ksteps_g = K / 16, kstep0 = k0 / 16;
#pragma unroll
    for (int off = 0; off < NTILES_T * KSTEPS_T * 32; off += DNTHREADS * 2) {
      const int sl = off + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
      const int r = ntl * 16 + (lane_ & 15), kloc = kst * 16 + (lane_ >> 4) * 8;
      const int gn = n0 + r;
      const int gc = gn < N - 2 ? gn : N - 2;
      const int lanec = (lane_ & 16) | (gc & 15);
      const rmx_uint2 wv = rmx_ld_w<NT>(
          (const rmx_uint2 *)&Wp[((size_t)(gc >> 4) * ksteps_g + kstep0 + kst) * 32 + lanec]);
      const int blk = (k0 + kst * 16) / 32;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        int d = (int)Wref[gc + q] - (int)Ws[(size_t)blk * N + gc + q];
        d = d < 0 ? 0 : (d > 15 ? 15 : d);
        const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
        const unsigned int ev = rmx_wq_dec(wv, q) & 0x0F0F0F0Fu;
        const unsigned int od = (rmx_wq_dec(wv, q) >> 4) & 0x0F0F0F0Fu;
        const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                              | ((ev & 0x08080808u) << 4);
        const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                              | ((od & 0x08080808u) << 4);
        *(rmx_uint2 *)(&sW[(r + q) * DWSTR + kloc]) =
            rmx_uint2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                      __builtin_amdgcn_perm(bo, be, 0x07030602u)};
      }
    }
  };
  auto mma_slab = [&]() {
    __syncthreads();
#pragma unroll
    for (int step = 0; step < DBK / 16; ++step) {
      const int kk = step * 16 + kb8;
      rmx_int2 af[DTM], wf, al[HILO ? DTM : 1];
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        if (i < ntm) {
          const unsigned char *pa = &sA[(i * 16 + col) * DASTR + kk];
          af[i][0] = *(const int *)pa; af[i][1] = *(const int *)(pa + 4);
        }
      }
      if constexpr (HILO) {
#pragma unroll
        for (int i = 0; i < DTM; ++i) {
          if (i < ntm) {
            const unsigned char *pl = &sA_lo[(i * 16 + col) * DASTR + kk];
            al[i][0] = *(const int *)pl; al[i][1] = *(const int *)(pl + 4);
          }
        }
      }
      const unsigned char *pw = &sW[(wave * 16 + col) * DWSTR + kk];
      wf[0] = *(const int *)pw; wf[1] = *(const int *)(pw + 4);
#pragma unroll
      for (int i = 0; i < DTM; ++i) {
        if (i < ntm) {
          acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i], wf, acc[i]);
          if constexpr (HILO)
            acc[i] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(al[i], wf, acc[i]);
        }
      }
    }
    __syncthreads();
  };

  if constexpr (GROUPED) {
    // Software-pipelined slab loop (grouped / row mode): the global loads of slab s+1 (W words, Ws bytes, A rows) are
    // issued before slab s's WMMAs, and only converted + written to LDS afterwards, so the K loop no longer pays one
    // full global-load latency per slab. Same numerics as the plain loop below (identical dequant and MMA order).
    constexpr int KSTEPS_T = DBK / 16, NTILES_T = BND / 16;
    constexpr int NW = NTILES_T * KSTEPS_T * 32 / (DNTHREADS * 2);
    static_assert(NW * DNTHREADS * 2 == NTILES_T * KSTEPS_T * 32, "W slot groups must tile");
    constexpr int NA = (DEC_MTILE * DTM * DBK + DNTHREADS * 16 - 1) / (DNTHREADS * 16);
    const unsigned int *__restrict__ Wp = (const unsigned int *)W;
    const int ksteps_g = K / 16;
    int p_gc[NW], p_r[NW], p_kloc[NW], p_kst[NW], p_lanec[NW];
    unsigned short p_ref[NW];
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const int sl = it * DNTHREADS * 2 + tid * 2;
      const int lane_ = sl & 31, rest = sl >> 5;
      p_kst[it] = rest % KSTEPS_T;
      const int ntl = rest / KSTEPS_T;
      p_r[it] = ntl * 16 + (lane_ & 15);
      p_kloc[it] = p_kst[it] * 16 + (lane_ >> 4) * 8;
      const int gn = n0 + p_r[it];
      p_gc[it] = gn < N - 2 ? gn : N - 2;
      p_lanec[it] = (lane_ & 16) | (p_gc[it] & 15);
      p_ref[it] = *(const unsigned short *)(Wref + p_gc[it]);
    }
    rmx_uint2 wvr[NW];
    unsigned short wsr[NW];
    rmx_uint4 ar[NA], arl[HILO ? NA : 1];
    const int a_rows_bytes = DEC_MTILE * ntm * DBK;
    auto ld_slab = [&](int k0) {
      const int kstep0 = k0 / 16;
#pragma unroll
      for (int it = 0; it < NW; ++it) {
        wvr[it] = rmx_ld_w<NT>(
            (const rmx_uint2 *)&Wp[((size_t)(p_gc[it] >> 4) * ksteps_g + kstep0 + p_kst[it]) * 32 + p_lanec[it]]);
        const int blk = (k0 + p_kst[it] * 16) / 32;
        wsr[it] = *(const unsigned short *)(Ws + (size_t)blk * N + p_gc[it]);
      }
#pragma unroll
      for (int ia = 0; ia < NA; ++ia) {
        const int idx = ia * DNTHREADS * 16 + tid * 16;
        if (idx < a_rows_bytes) {
          const int r = idx / DBK, c = idx % DBK;
          const int rc = r < M - 1 ? r : M - 1;                  // clamp, never predicate
          ar[ia] = *(const rmx_uint4 *)(A + (size_t)rc * K + k0 + c);
          if constexpr (HILO) arl[ia] = *(const rmx_uint4 *)(A_lo + (size_t)rc * K + k0 + c);
        }
      }
    };
    auto st_slab = [&]() {
#pragma unroll
      for (int ia = 0; ia < NA; ++ia) {
        const int idx = ia * DNTHREADS * 16 + tid * 16;
        if (idx < a_rows_bytes) {
          const int r = idx / DBK, c = idx % DBK;
          *(rmx_uint4 *)(&sA[r * DASTR + c]) = ar[ia];
          if constexpr (HILO) *(rmx_uint4 *)(&sA_lo[r * DASTR + c]) = arl[ia];
        }
      }
#pragma unroll
      for (int it = 0; it < NW; ++it) {
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          int d = (int)((p_ref[it] >> (8 * q)) & 0xFF) - (int)((wsr[it] >> (8 * q)) & 0xFF);
          d = d < 0 ? 0 : (d > 15 ? 15 : d);
          const unsigned int t0 = kMag[d][0], t1 = kMag[d][1];
          const unsigned int wq = rmx_wq_dec(wvr[it], q);
          const unsigned int ev = wq & 0x0F0F0F0Fu;
          const unsigned int od = (wq >> 4) & 0x0F0F0F0Fu;
          const unsigned int be = __builtin_amdgcn_perm(t1, t0, ev & 0x07070707u)
                                | ((ev & 0x08080808u) << 4);
          const unsigned int bo = __builtin_amdgcn_perm(t1, t0, od & 0x07070707u)
                                | ((od & 0x08080808u) << 4);
          *(rmx_uint2 *)(&sW[(p_r[it] + q) * DWSTR + p_kloc[it]]) =
              rmx_uint2{__builtin_amdgcn_perm(bo, be, 0x05010400u),
                        __builtin_amdgcn_perm(bo, be, 0x07030602u)};
        }
      }
    };
    if (s_lo < s_hi) ld_slab(s_lo * DBK);
    for (int s = s_lo; s < s_hi; ++s) {
      st_slab();
      if (s + 1 < s_hi) ld_slab((s + 1) * DBK);
      mma_slab();
    }
  } else
  for (int s = s_lo; s < s_hi; ++s) {
    const int k0 = s * DBK;
    stage_a(k0);
    if constexpr (HILO) stage_a_lo(k0);
    stage_w_wperm(k0);
    mma_slab();
  }

  const int n = n0 + wave * 16 + col;

  if constexpr (DKS == 1) {
    if (n < N) {
      const float rf = __int_as_float((int)Wref[n] << 23);
#pragma unroll
      for (int i = 0; i < DTM; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int m = i * 16 + kb8 + e;
          if (m < M) rmx_store(&C[crow(m) + n], acc[i][e] * rf * As[m]);
        }
    }
    return;
  }

  if (n < N) {
#pragma unroll
    for (int i = 0; i < DTM; ++i)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = i * 16 + kb8 + e;
        if (m < M) P[((size_t)ks * M + m) * N + n] = acc[i][e];
      }
  }

  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
  }
  __syncthreads();
  if (!s_last) return;
  if (tid == 0) cnt[blockIdx.x] = 0;

  const int nhi = min(n0 + BND, N);
  for (int nn = n0 + tid; nn < nhi; nn += DNTHREADS) {
    const float rf = __int_as_float((int)Wref[nn] << 23);
    for (int m = 0; m < M; ++m) {
      float sum = 0.f;
      for (int k = 0; k < DKS; ++k) sum += P[((size_t)k * M + m) * N + nn];
      rmx_store(&C[crow(m) + nn], sum * rf * As[m]);
    }
  }
}

// ── activation quant: radiance_add_rms_quant's scale + convert, no norm ───
// One 256-thread block per row. HILO also writes the e4m3 residual q_lo.
template <bool HILO>
__global__ __launch_bounds__(256) void rmx_quant_a_kernel(const float *__restrict__ x, int64_t ld,
                                                          int K, unsigned char *__restrict__ q,
                                                          unsigned char *__restrict__ q_lo,
                                                          float *__restrict__ scale) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const float *xr = x + (int64_t)row * ld;
  float amax = 0.f;
  for (int k = tid; k < K; k += 256) amax = fmaxf(amax, fabsf(xr[k]));
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  __shared__ float lds[8];
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < 32) {
    float v = tid < 8 ? lds[tid] : 0.f;
    for (int o = 4; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[row] = sc;
  unsigned char *qr = q + (int64_t)row * K;
  unsigned char *qlr = HILO ? q_lo + (int64_t)row * K : nullptr;
  for (int k = 2 * tid; k < K; k += 512) {           // K % 128 == 0, so pairs never straddle
    const float a = fminf(fmaxf(xr[k] * rs, -448.f), 448.f);
    const float b = fminf(fmaxf(xr[k + 1] * rs, -448.f), 448.f);
    const unsigned int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu;
    *(unsigned short *)(qr + k) = (unsigned short)p;
    if constexpr (HILO) {
      // residual of the hi part, same scale; lo = e4m3(x/s - hi)
      const float ra = a - __builtin_amdgcn_cvt_f32_fp8(p, 0);
      const float rb = b - __builtin_amdgcn_cvt_f32_fp8(p, 1);
      const unsigned int pl = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(ra, -448.f), 448.f),
                                                              fminf(fmaxf(rb, -448.f), 448.f),
                                                              0u, false) & 0xffffu;
      *(unsigned short *)(qlr + k) = (unsigned short)pl;
    }
  }
}

// Tiled variant of rmx_quant_a_kernel (same numerics, A-tiled store layout). One block
// per row of the padded M (ceil(M/16)*16); pad rows (row >= M) write zeros, no scale.
// Byte offset of (row, k) in the tiled buffer: ((row/16)*(K/16) + k/16)*256
// + ((k/8)&1)*128 + (row%16)*8 + k%8.
template <bool HILO>
__global__ __launch_bounds__(256) void rmx_quant_a_tiled_kernel(const float *__restrict__ x, int64_t ld,
                                                                int M, int K,
                                                                unsigned char *__restrict__ q,
                                                                unsigned char *__restrict__ q_lo,
                                                                float *__restrict__ scale) {
  const int row = blockIdx.x, tid = threadIdx.x;
  const bool valid = row < M;
  const float *xr = x + (int64_t)row * ld;
  float amax = 0.f;
  if (valid)
    for (int k = tid; k < K; k += 256) amax = fmaxf(amax, fabsf(xr[k]));
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  __shared__ float lds[8];
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < 32) {
    float v = tid < 8 ? lds[tid] : 0.f;
    for (int o = 4; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (valid && tid == 0) scale[row] = sc;
  const int64_t rbase = (int64_t)(row >> 4) * (K >> 4);
  const int rl = row & 15;
  for (int k = 2 * tid; k < K; k += 512) {           // K % 128 == 0, so pairs never straddle
    float a = 0.f, b = 0.f;
    if (valid) {
      a = fminf(fmaxf(xr[k] * rs, -448.f), 448.f);
      b = fminf(fmaxf(xr[k + 1] * rs, -448.f), 448.f);
    }
    const unsigned int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu;
    const int64_t off = ((rbase + (k >> 4)) << 8) + (((k >> 3) & 1) << 7) + (rl << 3) + (k & 7);
    *(unsigned short *)(q + off) = (unsigned short)p;
    if constexpr (HILO) {
      const float ra = a - __builtin_amdgcn_cvt_f32_fp8(p, 0);
      const float rb = b - __builtin_amdgcn_cvt_f32_fp8(p, 1);
      const unsigned int pl = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(ra, -448.f), 448.f),
                                                              fminf(fmaxf(rb, -448.f), 448.f),
                                                              0u, false) & 0xffffu;
      *(unsigned short *)(q_lo + off) = (unsigned short)pl;
    }
  }
}

// Grouped (MUL_MAT_ID) activation quant: one 256-thread block per row of the expert-grouped, 16-row-padded
// buffer. Padded row r belongs to the expert e with padoff[e] <= r < padoff[e+1] (binary search); j = r -
// padoff[e] is its routed-row index inside the expert, valid iff j < count(e). The source row is read from
// src1 through ids_dst (dst row = token*n_used + slot): offset = (bcast ? 0 : slot)*nb1 + token*nb2 floats.
// Experts with count <= thr (the decode class) get row-major A, the others the tiled A layout; both put
// segment e at padoff[e]*K bytes. Pad rows quantize to zero and get no scale. Same numerics as
// rmx_quant_a_kernel / rmx_quant_a_tiled_kernel (hi/lo always).
__global__ __launch_bounds__(256) void rmx_quant_a_moe_kernel(
    const float *__restrict__ x, long nb1, long nb2, int bcast, int n_used,
    const int *__restrict__ bounds, const int *__restrict__ padoff, const int *__restrict__ ids_dst,
    int n_as, int thr, int K, unsigned char *__restrict__ q, unsigned char *__restrict__ q_lo,
    float *__restrict__ scale) {
  const int r = blockIdx.x, tid = threadIdx.x;
  if (r >= padoff[n_as]) return;
  int lo = 0, hi = n_as;                      // last e with padoff[e] <= r
  while (hi - lo > 1) {
    const int mid = (lo + hi) >> 1;
    if (padoff[mid] <= r) lo = mid; else hi = mid;
  }
  const int e = lo;
  const int j = r - padoff[e];
  const int cnt = bounds[e + 1] - bounds[e];
  const bool valid = j < cnt;
  const bool tiled = cnt > thr;
  const float *xr = x;
  if (valid) {
    const int d = ids_dst[bounds[e] + j];
    const int t = d / n_used;
    const int sl = d - t * n_used;
    xr = x + (bcast ? 0L : (long)sl * nb1) + (long)t * nb2;
  }
  float amax = 0.f;
  if (valid)
    for (int k = tid; k < K; k += 256) amax = fmaxf(amax, fabsf(xr[k]));
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  __shared__ float lds[8];
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < 32) {
    float v = tid < 8 ? lds[tid] : 0.f;
    for (int o = 4; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (valid && tid == 0) scale[r] = sc;
  const long rbase = (long)(r >> 4) * (K >> 4);
  const int rl = r & 15;
  for (int k = 2 * tid; k < K; k += 512) {           // K % 128 == 0, so pairs never straddle
    float a = 0.f, b = 0.f;
    if (valid) {
      a = fminf(fmaxf(xr[k] * rs, -448.f), 448.f);
      b = fminf(fmaxf(xr[k + 1] * rs, -448.f), 448.f);
    }
    const unsigned int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu;
    const float ra = a - __builtin_amdgcn_cvt_f32_fp8(p, 0);
    const float rb = b - __builtin_amdgcn_cvt_f32_fp8(p, 1);
    const unsigned int pl = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(ra, -448.f), 448.f),
                                                            fminf(fmaxf(rb, -448.f), 448.f),
                                                            0u, false) & 0xffffu;
    const long off = tiled ? (((rbase + (k >> 4)) << 8) + (((k >> 3) & 1) << 7) + (rl << 3) + (k & 7))
                           : ((long)r * K + k);
    *(unsigned short *)(q + off) = (unsigned short)p;
    *(unsigned short *)(q_lo + off) = (unsigned short)pl;
  }
}

// Row-mode quant (tiny n_tok): one block per routed row r = token*n_used + slot, row-major hi/lo at r*K, scale[r].
// Block 0 also zeroes the split-K counters (n_cnt ints) so the GEMM needs no memset launch.
__global__ __launch_bounds__(256) void rmx_quant_a_rows_kernel(
    const float *__restrict__ x, long nb1, long nb2, int bcast, int n_used, int K,
    unsigned char *__restrict__ q, unsigned char *__restrict__ q_lo, float *__restrict__ scale,
    int *__restrict__ cnt, int n_cnt) {
  const int r = blockIdx.x, tid = threadIdx.x;
  if (r == 0)
    for (int i = tid; i < n_cnt; i += 256) cnt[i] = 0;
  const int t = r / n_used;
  const int sl = r - t * n_used;
  const float *xr = x + (bcast ? 0L : (long)sl * nb1) + (long)t * nb2;
  float amax = 0.f;
  for (int k = tid; k < K; k += 256) amax = fmaxf(amax, fabsf(xr[k]));
  for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_down(amax, o));
  __shared__ float lds[8];
  if ((tid & 31) == 0) lds[tid >> 5] = amax;
  __syncthreads();
  if (tid < 32) {
    float v = tid < 8 ? lds[tid] : 0.f;
    for (int o = 4; o; o >>= 1) v = fmaxf(v, __shfl_down(v, o));
    if (tid == 0) lds[0] = v;
  }
  __syncthreads();
  const float sc = fmaxf(lds[0] * (1.f / 448.f), 1.f / (448.f * 512.f));
  const float rs = 1.f / sc;
  if (tid == 0) scale[r] = sc;
  for (int k = 2 * tid; k < K; k += 512) {
    const float a = fminf(fmaxf(xr[k] * rs, -448.f), 448.f);
    const float b = fminf(fmaxf(xr[k + 1] * rs, -448.f), 448.f);
    const unsigned int p = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false) & 0xffffu;
    const float ra = a - __builtin_amdgcn_cvt_f32_fp8(p, 0);
    const float rb = b - __builtin_amdgcn_cvt_f32_fp8(p, 1);
    const unsigned int pl = __builtin_amdgcn_cvt_pk_fp8_f32(fminf(fmaxf(ra, -448.f), 448.f),
                                                            fminf(fmaxf(rb, -448.f), 448.f),
                                                            0u, false) & 0xffffu;
    *(unsigned short *)(q + (long)r * K + k) = (unsigned short)p;
    *(unsigned short *)(q_lo + (long)r * K + k) = (unsigned short)pl;
  }
}

#else
// Other-arch device pass: declaration-only stubs so the host launchers below
// still resolve. They are never launched off gfx1201 (the dispatcher gates on
// the device arch), same convention as gemm_ml84_radiance.hip.
template <int TN, bool EPIFAST, bool HILO, typename OutT>
__global__ __launch_bounds__(NTHREADS) void rmx_gemm_folded(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ A_lo,
    const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, OutT *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride);
template <int DWN, int DBK, int DKS, int DTM, bool NT, bool HILO, bool GROUPED, typename OutT>
__global__ __launch_bounds__(DWN * 32) void rmx_gemm_decode(
    const unsigned char *__restrict__ A, const unsigned char *__restrict__ A_lo,
    const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, float *__restrict__ P, int *__restrict__ cnt,
    OutT *__restrict__ C, int M, int N, int K, int pb1, int pb2, long astride, rmx_moe moe);
template <bool HILO>
__global__ __launch_bounds__(256) void rmx_quant_a_kernel(const float *__restrict__ x, int64_t ld,
                                                          int K, unsigned char *__restrict__ q,
                                                          unsigned char *__restrict__ q_lo,
                                                          float *__restrict__ scale);
template <int TN, int LBK, bool HILO, bool GROUPED, typename OutT>
__global__ __launch_bounds__(NTHREADS) void rmx_gemm_atiled(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ AT_lo,
    const unsigned char *__restrict__ W,
    const unsigned char *__restrict__ Ws, const unsigned char *__restrict__ Wref,
    const float *__restrict__ As, OutT *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride, rmx_moe moe);
template <bool HILO>
__global__ __launch_bounds__(256) void rmx_quant_a_tiled_kernel(const float *__restrict__ x, int64_t ld,
                                                                int M, int K,
                                                                unsigned char *__restrict__ q,
                                                                unsigned char *__restrict__ q_lo,
                                                                float *__restrict__ scale);
__global__ __launch_bounds__(256) void rmx_quant_a_moe_kernel(
    const float *__restrict__ x, long nb1, long nb2, int bcast, int n_used,
    const int *__restrict__ bounds, const int *__restrict__ padoff, const int *__restrict__ ids_dst,
    int n_as, int thr, int K, unsigned char *__restrict__ q, unsigned char *__restrict__ q_lo,
    float *__restrict__ scale);
__global__ __launch_bounds__(256) void rmx_quant_a_rows_kernel(
    const float *__restrict__ x, long nb1, long nb2, int bcast, int n_used, int K,
    unsigned char *__restrict__ q, unsigned char *__restrict__ q_lo, float *__restrict__ scale,
    int *__restrict__ cnt, int n_cnt);
#endif   // __gfx1201__ || host pass

// ── repack: ggml block_mxfp4 -> radiance packed (pure byte work, any arch) ─
// One 128-thread block per row n. Each thread takes 32-element blocks of the
// row: writes the block exponent to Ws[blk][n], rebuilds the 16 bytes in
// checkpoint order (byte b = elem 2b | elem 2b+1 << 4; ggml stores
// qs[j] = elem j | elem j+16 << 4) and scatters them to their WPERM slots;
// the row max exponent goes to Wref[n].
//
// WPERM (radiance_mxfp4.py permute_w): checkpoint byte kb of row n, with
// kstep = kb/8, half = (kb/4)&1, bb = kb&3, lands at
//   ((n/16)*(K/16) + kstep)*128 + half*64 + (n%16)*4 + bb.
__global__ __launch_bounds__(128) void rmx_repack_kernel(const unsigned char *__restrict__ src,
                                                         int N, int K,
                                                         unsigned char *__restrict__ W,
                                                         unsigned char *__restrict__ Ws,
                                                         unsigned char *__restrict__ Wref) {
  const int n = blockIdx.x, tid = threadIdx.x;
  const int nb = K / 32, ksteps = K / 16;
  const unsigned char *row = src + (size_t)n * nb * 17;
  unsigned int emax = 0;
  for (int blk = tid; blk < nb; blk += 128) {
    const unsigned char *b = row + (size_t)blk * 17;
    const unsigned int e = b[0];
    emax = e > emax ? e : emax;
    Ws[(size_t)blk * N + n] = (unsigned char)e;
    unsigned char el[32];
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      el[j]      = b[1 + j] & 0x0F;
      el[j + 16] = b[1 + j] >> 4;
    }
#pragma unroll
    for (int bi = 0; bi < 16; ++bi) {
      const int kb = blk * 16 + bi;
      const int kstep = kb >> 3, half = (kb >> 2) & 1, bb = kb & 3;
      W[((size_t)(n >> 4) * ksteps + kstep) * 128 + half * 64 + (n & 15) * 4 + bb] =
          (unsigned char)(el[2 * bi] | (el[2 * bi + 1] << 4));
    }
  }
  for (int o = 16; o; o >>= 1) {
    const unsigned int v = __shfl_down(emax, o);
    emax = v > emax ? v : emax;
  }
  __shared__ unsigned int lds[4];
  if ((tid & 31) == 0) lds[tid >> 5] = emax;
  __syncthreads();
  if (tid == 0) {
    unsigned int m = lds[0];
    for (int i = 1; i < 4; ++i) m = lds[i] > m ? lds[i] : m;
    Wref[n] = (unsigned char)m;
  }
}


// ── inverse of rmx_repack_kernel (packed -> ggml block_mxfp4), exact except for Wref which is dropped ─────
// One 128-thread block per row n; each thread rebuilds whole 32-element blocks. Reads the nibbles back from
// their WPERM slots (byte kb = blk*16+bi: elem 2bi | elem 2bi+1 << 4), un-splits them into ggml's
// qs[j] = elem j | elem j+16 << 4 and takes the E8M0 exponent from the transposed Ws[blk][n].
__global__ __launch_bounds__(128) void rmx_unpack_kernel(const unsigned char *__restrict__ W,
                                                         const unsigned char *__restrict__ Ws,
                                                         int N, int K, unsigned char *__restrict__ dst) {
  const int n = blockIdx.x, tid = threadIdx.x;
  const int nb = K / 32, ksteps = K / 16;
  unsigned char *row = dst + (size_t)n * nb * 17;
  for (int blk = tid; blk < nb; blk += 128) {
    unsigned char *b = row + (size_t)blk * 17;
    b[0] = Ws[(size_t)blk * N + n];
    unsigned char el[32];
#pragma unroll
    for (int bi = 0; bi < 16; ++bi) {
      const int kb = blk * 16 + bi;
      const int kstep = kb >> 3, half = (kb >> 2) & 1, bb = kb & 3;
      const unsigned char v = W[((size_t)(n >> 4) * ksteps + kstep) * 128 + half * 64 + (n & 15) * 4 + bb];
      el[2 * bi] = v & 0x0F;
      el[2 * bi + 1] = v >> 4;
    }
#pragma unroll
    for (int j = 0; j < 16; ++j) b[1 + j] = (unsigned char)(el[j] | (el[j + 16] << 4));
  }
}

// ── grouped launch plan: one block. From expert_bounds it builds the 16-row-padded segment offsets and the two
// work lists (decode class: 0 < count <= thr, one entry per expert; atiled class: count > thr, one entry per
// 256-row m-tile). counts[0] = #decode entries, counts[1] = #atiled entries. List writes are capped (cap) so a
// violated host bound can never write out of range; the host sizes lists with exact upper bounds.
__global__ __launch_bounds__(256) void rmx_moe_plan_kernel(const int *__restrict__ bounds, int n_as, int thr,
                                                           int at_cap, int *__restrict__ padoff,
                                                           int *__restrict__ dec_e, int *__restrict__ at_e,
                                                           int *__restrict__ at_mt, int *__restrict__ counts) {
  __shared__ int sp[3][256];
  const int t = threadIdx.x;
  const int chunk = (n_as + 255) / 256;
  const int lo = min(n_as, t * chunk), hi = min(n_as, lo + chunk);
  int a0 = 0, a1 = 0, a2 = 0;
  for (int e = lo; e < hi; ++e) {
    const int c = bounds[e + 1] - bounds[e];
    a0 += (c + 15) & ~15;
    if (c > 0 && c <= thr) a1 += 1;
    else if (c > thr) a2 += (c + BMF - 1) / BMF;
  }
  sp[0][t] = a0; sp[1][t] = a1; sp[2][t] = a2;
  __syncthreads();
  for (int o = 1; o < 256; o <<= 1) {              // inclusive Hillis-Steele scan
    int v0 = 0, v1 = 0, v2 = 0;
    if (t >= o) { v0 = sp[0][t - o]; v1 = sp[1][t - o]; v2 = sp[2][t - o]; }
    __syncthreads();
    sp[0][t] += v0; sp[1][t] += v1; sp[2][t] += v2;
    __syncthreads();
  }
  int p = sp[0][t] - a0, d = sp[1][t] - a1, a = sp[2][t] - a2;   // exclusive
  for (int e = lo; e < hi; ++e) {
    const int c = bounds[e + 1] - bounds[e];
    padoff[e] = p;
    p += (c + 15) & ~15;
    if (c > 0 && c <= thr) {
      dec_e[d++] = e;
    } else if (c > thr) {
      const int nt = (c + BMF - 1) / BMF;
      for (int j = 0; j < nt; ++j, ++a) {
        if (a < at_cap) { at_e[a] = e; at_mt[a] = j; }
      }
    }
  }
  if (t == 255) {
    padoff[n_as] = sp[0][255];
    counts[0] = sp[1][255];
    counts[1] = min(sp[2][255], at_cap);
  }
}

// ── host ──────────────────────────────────────────────────────────────────
bool rdna4_mxfp4_radiance_shape_ok(int N, int K) {
  return N > 0 && K > 0 && N % 16 == 0 && K % 128 == 0;
}
size_t rdna4_mxfp4_radiance_ws_offset(int N, int K) { return (size_t)N * K / 2; }
size_t rdna4_mxfp4_radiance_wref_offset(int N, int K) {
  return rdna4_mxfp4_radiance_ws_offset(N, K) + (size_t)(K / 32) * N;
}
size_t rdna4_mxfp4_radiance_packed_bytes(int N, int K) {
  return rdna4_mxfp4_radiance_wref_offset(N, K) + (size_t)N;
}

bool rdna4_mxfp4_radiance_repack(const void* src_blocks, int N, int K, void* dst_packed,
                                 hipStream_t stream) {
  if (!rdna4_mxfp4_radiance_shape_ok(N, K)) return false;
  auto *dst = static_cast<unsigned char*>(dst_packed);
  hipLaunchKernelGGL(rmx_repack_kernel, dim3(N), dim3(128), 0, stream,
                     static_cast<const unsigned char*>(src_blocks), N, K, dst,
                     dst + rdna4_mxfp4_radiance_ws_offset(N, K),
                     dst + rdna4_mxfp4_radiance_wref_offset(N, K));
  return hipGetLastError() == hipSuccess;
}

size_t rdna4_mxfp4_radiance_a_bytes(int M, int K, bool tiled) {
  return tiled ? (size_t)((M + 15) & ~15) * (size_t)K : (size_t)M * (size_t)K;
}

bool rdna4_mxfp4_radiance_quant_a(const float* x, int64_t ld, int M, int K, uint8_t* A_fp8,
                                  uint8_t* A_lo, float* a_scale, bool tiled, hipStream_t stream) {
  if (M <= 0 || K <= 0 || K % 128) return false;
  if (tiled) {
    const int Mp = (M + 15) & ~15;
    if (A_lo)
      hipLaunchKernelGGL(rmx_quant_a_tiled_kernel<true>, dim3(Mp), dim3(256), 0, stream, x, ld, M,
                         K, A_fp8, A_lo, a_scale);
    else
      hipLaunchKernelGGL(rmx_quant_a_tiled_kernel<false>, dim3(Mp), dim3(256), 0, stream, x, ld,
                         M, K, A_fp8, A_lo, a_scale);
    return hipGetLastError() == hipSuccess;
  }
  if (A_lo)
    hipLaunchKernelGGL(rmx_quant_a_kernel<true>, dim3(M), dim3(256), 0, stream, x, ld, K, A_fp8,
                       A_lo, a_scale);
  else
    hipLaunchKernelGGL(rmx_quant_a_kernel<false>, dim3(M), dim3(256), 0, stream, x, ld, K, A_fp8,
                       A_lo, a_scale);
  return hipGetLastError() == hipSuccess;
}

static int rmx_decode_max_m() {
  static const int v = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_DECODE_MAX_M");
    return e ? std::atoi(e) : 64;
  }();
  return v;
}

// ── radiance_mxfp4_fp8.hip:1088-1116, verbatim (env renamed) ──────────────
static bool rmx_decode_bk64(int tm, int ks) {
  static const bool pin128 = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_DECODE_BK");
    return e && std::atoi(e) == 128;
  }();
  return !pin128 && tm == 4 && ks == 1;
}
static int rmx_split_k_for(int nblk, int M) {
  static const int forced = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_DECODE_KS");
    return e ? std::atoi(e) : 0;
  }();
  if (forced) return forced;
  const int fill = (M <= 24) ? 110 : 78;
  for (int ks = 1; ks < DEC_KS; ks <<= 1)
    if (nblk * ks >= fill) return ks;
  return DEC_KS;
}

// launch_impl's decode band policy (radiance_mxfp4_fp8.hip:1179-1226), verbatim.
static void rmx_decode_plan(int M, int N, int K, int *dks_out, int *bk_out) {
  const int dec_nblk = (N + DEC_DWN * 16 - 1) / (DEC_DWN * 16);
  int dks, bkx;
  if (M > 64) {
    if (dec_nblk >= 110)      { dks = 1; bkx = 64; }
    else if (dec_nblk >= 48)  { dks = 1; bkx = 128; }
    else if (dec_nblk <= 8)   { dks = DEC_KS; bkx = 128; }
    else if (K >= 6144)       { if (M <= 80) { dks = 2; bkx = 128; }
                                else         { dks = DEC_KS; bkx = 64; } }
    else                      { dks = 1; bkx = 128; }
  } else {
    dks = rmx_split_k_for(dec_nblk, M);
    bkx = 0;
  }
  if (K % 128) {
    bkx = 64;
    if (dks == 2) dks = DEC_KS;
  }
  if (bkx == 0) bkx = rmx_decode_bk64((M + DEC_MTILE - 1) / DEC_MTILE, dks) ? 64 : 128;
  *dks_out = dks;
  *bk_out = bkx;
}

static bool rmx_use_decode(int M, int N) {
  return M > 0 && M <= rmx_decode_max_m() && M <= DEC_MTILE * DEC_MAX_TM && N <= DEC_MAX_N;
}

// split-K for row mode: enough blocks to fill the GPU, env RDNA4_MXFP4_RADIANCE_ROWS_KS forces it
static int rmx_rows_split_k(int nblk_rows) {
  static const int forced = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_ROWS_KS");
    return e ? std::atoi(e) : 0;
  }();
  if (forced) return forced == 1 || forced == 2 ? forced : 4;
  static const int fill = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_ROWS_FILL");
    return e ? std::atoi(e) : 110;
  }();
  for (int ks = 1; ks < DEC_KS; ks <<= 1)
    if (nblk_rows * ks >= fill) return ks;
  return DEC_KS;
}

size_t rdna4_mxfp4_radiance_scratch_bytes(int M, int N, int K) {
  if (!rmx_use_decode(M, N)) return 0;
  int dks, bkx;
  rmx_decode_plan(M, N, K, &dks, &bkx);
  if (dks == 1) return 0;
  const int dec_nblk = (N + DEC_DWN * 16 - 1) / (DEC_DWN * 16);
  const size_t part = (size_t)dks * M * N * sizeof(float);
  return ((part + 255) & ~(size_t)255) + (size_t)dec_nblk * sizeof(int);
}

template <bool HILO, typename OutT>
static bool rmx_launch_impl(int M, const uint8_t* A, const uint8_t* A_lo, const float* As,
                            const void* packed, OutT* C, int N, int K, void* scratch,
                            size_t scratch_bytes, bool a_tiled, hipStream_t stream) {
  if (M <= 0 || !rdna4_mxfp4_radiance_shape_ok(N, K)) return false;
  // The decode kernel reads row-major A only; tiled A is a prefill-class layout.
  if (a_tiled && rmx_use_decode(M, N)) return false;
  const auto *Wb = static_cast<const unsigned char*>(packed);
  const unsigned char *W = Wb;
  const unsigned char *Ws = Wb + rdna4_mxfp4_radiance_ws_offset(N, K);
  const unsigned char *Wref = Wb + rdna4_mxfp4_radiance_wref_offset(N, K);
  static const bool dec_nt = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_DECODE_NT");
    return e && std::atoi(e) != 0;
  }();
  static const int tn4_min_m = [] {
    const char *e = std::getenv("RDNA4_MXFP4_RADIANCE_TN4_MIN_M");
    return e ? std::atoi(e) : 2048;
  }();

  if (rmx_use_decode(M, N)) {
    int dks, bkx;
    rmx_decode_plan(M, N, K, &dks, &bkx);
    float *P = nullptr;
    int *cnt = nullptr;
    if (dks > 1) {
      if (scratch == nullptr || scratch_bytes < rdna4_mxfp4_radiance_scratch_bytes(M, N, K)) return false;
      const size_t part = (size_t)dks * M * N * sizeof(float);
      P = static_cast<float*>(scratch);
      cnt = reinterpret_cast<int*>(static_cast<char*>(scratch) + ((part + 255) & ~(size_t)255));
      const int dec_nblk = (N + DEC_DWN * 16 - 1) / (DEC_DWN * 16);
      if (hipMemsetAsync(cnt, 0, (size_t)dec_nblk * sizeof(int), stream) != hipSuccess) return false;
    }
    constexpr int DWN = DEC_DWN;
    const int dec_nblk = (N + DWN * 16 - 1) / (DWN * 16);
    const dim3 dblock(DWN * 32);
    const int tm = (M + DEC_MTILE - 1) / DEC_MTILE;
#define RMX_DEC_LAUNCH(BK_, KS_, TM_, NT_)                                                      \
    hipLaunchKernelGGL((rmx_gemm_decode<DWN, BK_, KS_, TM_, NT_, HILO, false, OutT>),            \
                       dim3(dec_nblk, 1, KS_), dblock, 0, stream, A, A_lo, W, Ws, Wref, As,    \
                       P, cnt, C, M, N, K, 1 << 30, 1 << 30, 0L, rmx_moe{})
#define RMX_DEC_BY_TM(BK_, KS_, NT_)                                                          \
    if (tm == 1)      RMX_DEC_LAUNCH(BK_, KS_, 1, NT_);                                        \
    else if (tm == 2) RMX_DEC_LAUNCH(BK_, KS_, 2, NT_);                                        \
    else if (tm == 3) RMX_DEC_LAUNCH(BK_, KS_, 3, NT_);                                        \
    else if (tm == 4) RMX_DEC_LAUNCH(BK_, KS_, 4, NT_);                                        \
    else if (tm == 5) RMX_DEC_LAUNCH(BK_, KS_, 5, NT_);                                        \
    else if (tm == 6) RMX_DEC_LAUNCH(BK_, KS_, 6, NT_);                                        \
    else if (tm == 7) RMX_DEC_LAUNCH(BK_, KS_, 7, NT_);                                        \
    else              RMX_DEC_LAUNCH(BK_, KS_, 8, NT_)
#define RMX_DEC_BY_KS(NT_)                                                                    \
    if (dks <= 1)      { if (bkx == 64) { RMX_DEC_BY_TM(64, 1, NT_); }                         \
                         else           { RMX_DEC_BY_TM(128, 1, NT_); } }                      \
    else if (dks == 2) { RMX_DEC_BY_TM(128, 2, NT_); }                                         \
    else               { if (bkx == 64) { RMX_DEC_BY_TM(64, 4, NT_); }                         \
                         else           { RMX_DEC_BY_TM(128, 4, NT_); } }
    if (dec_nt) { RMX_DEC_BY_KS(true); } else { RMX_DEC_BY_KS(false); }
#undef RMX_DEC_BY_KS
#undef RMX_DEC_BY_TM
#undef RMX_DEC_LAUNCH
    return hipGetLastError() == hipSuccess;
  }

  const dim3 fblock(NTHREADS);
  if (a_tiled) {
    // radiance's launch_at_impl: TN=4 (LBK=64) when M >= tn4_min_m and N >= 128, else TN=2 (LBK=128).
    if (M >= tn4_min_m && N >= BNF_OF(4)) {
      constexpr int B_ = BNF_OF(4);
      hipLaunchKernelGGL((rmx_gemm_atiled<4, 64, HILO, false, OutT>),
                         dim3((N + B_ - 1) / B_, (M + BMF - 1) / BMF), fblock, 0, stream, A, A_lo, W,
                         Ws, Wref, As, C, M, N, K, 1 << 30, 1 << 30, 0L, rmx_moe{});
    } else {
      constexpr int B_ = BNF_OF(2);
      hipLaunchKernelGGL((rmx_gemm_atiled<2, 128, HILO, false, OutT>),
                         dim3((N + B_ - 1) / B_, (M + BMF - 1) / BMF), fblock, 0, stream, A, A_lo, W,
                         Ws, Wref, As, C, M, N, K, 1 << 30, 1 << 30, 0L, rmx_moe{});
    }
    return hipGetLastError() == hipSuccess;
  }
  if (M >= tn4_min_m && N >= BNF_OF(4)) {
    constexpr int B_ = BNF_OF(4);
    hipLaunchKernelGGL((rmx_gemm_folded<4, true, HILO, OutT>),
                       dim3((N + B_ - 1) / B_, (M + BMF - 1) / BMF), fblock, 0, stream, A, A_lo, W,
                       Ws, Wref, As, C, M, N, K, 1 << 30, 1 << 30, 0L);
  } else {
    constexpr int B_ = BNF_OF(2);
    hipLaunchKernelGGL((rmx_gemm_folded<2, true, HILO, OutT>),
                       dim3((N + B_ - 1) / B_, (M + BMF - 1) / BMF), fblock, 0, stream, A, A_lo, W,
                       Ws, Wref, As, C, M, N, K, 1 << 30, 1 << 30, 0L);
  }
  return hipGetLastError() == hipSuccess;
}

template <typename OutT>
static bool rmx_launch(int M, const uint8_t* A, const uint8_t* A_lo, const float* As,
                       const void* packed, OutT* C, int N, int K, void* scratch,
                       size_t scratch_bytes, bool a_tiled, hipStream_t stream) {
  return A_lo ? rmx_launch_impl<true, OutT>(M, A, A_lo, As, packed, C, N, K, scratch, scratch_bytes,
                                            a_tiled, stream)
              : rmx_launch_impl<false, OutT>(M, A, A_lo, As, packed, C, N, K, scratch,
                                             scratch_bytes, a_tiled, stream);
}

bool rdna4_gemm_mxfp4_radiance(int M, const uint8_t* A_fp8, const uint8_t* A_lo,
                               const float* a_scale, const void* packed, float* C_f32, int N,
                               int K, void* scratch, size_t scratch_bytes, bool a_tiled,
                               hipStream_t stream) {
  return rmx_launch<float>(M, A_fp8, A_lo, a_scale, packed, C_f32, N, K, scratch, scratch_bytes,
                           a_tiled, stream);
}

// ── unpack: packed -> ggml block_mxfp4 (exact inverse of repack; Wref is dropped) ────────────────────
bool rdna4_mxfp4_radiance_unpack(const void* packed, int N, int K, void* dst_blocks, hipStream_t stream) {
  if (!rdna4_mxfp4_radiance_shape_ok(N, K)) return false;
  const auto *Wb = static_cast<const unsigned char*>(packed);
  hipLaunchKernelGGL(rmx_unpack_kernel, dim3(N), dim3(128), 0, stream, Wb,
                     Wb + rdna4_mxfp4_radiance_ws_offset(N, K), N, K,
                     static_cast<unsigned char*>(dst_blocks));
  return hipGetLastError() == hipSuccess;
}

// ── grouped (MUL_MAT_ID) path ────────────────────────────────────────────────────────────────────────
static size_t rmx_al256(size_t v) { return (v + 255) & ~(size_t)255; }

struct rmx_moe_geom {
  int n_as, n_used, n_tok, N, K;
  long T;            // routed rows, n_tok * n_used
  int dec_ymax;      // upper bound on decode-class experts = min(n_as, T)
  long Pmax;         // upper bound on padded rows (16-row segments)
  int thr;           // decode class: 0 < count <= thr; atiled class: count > thr
  int dtm;           // decode kernel M tiles (0 = no decode launch)
  bool need_at;      // some expert can exceed thr
  int at_cap;        // upper bound on atiled work-list entries
  int dks, bkx, nblk;
  size_t off_padoff, off_dec_e, off_at_e, off_at_mt, off_counts, off_scale, off_ahi, off_alo, off_part, off_cnt, total;
};

static bool rmx_moe_geom_make(rmx_moe_geom &g, int n_as, int n_used, int n_tok, int N, int K) {
  if (n_as <= 0 || n_used <= 0 || n_tok <= 0 || !rdna4_mxfp4_radiance_shape_ok(N, K)) return false;
  g.n_as = n_as; g.n_used = n_used; g.n_tok = n_tok; g.N = N; g.K = K;
  g.T = (long)n_tok * n_used;
  g.dec_ymax = (int)(g.T < n_as ? g.T : n_as);
  g.Pmax = (g.T + 15L * g.dec_ymax + 15) & ~15L;
  const int dec_max = rmx_decode_max_m() < 64 ? rmx_decode_max_m() : 64;
  const int n_dec_rows = dec_max < n_tok ? dec_max : n_tok;
  g.dtm = n_dec_rows > 0 ? (n_dec_rows + 15) / 16 : 0;
  g.thr = g.dtm > 0 ? (dec_max < 16 * g.dtm ? dec_max : 16 * g.dtm) : 0;
  g.need_at = g.T > g.thr;
  const long n_at_exp = g.T / (g.thr + 1) < n_as ? g.T / (g.thr + 1) : n_as;
  g.at_cap = (int)(g.T / BMF + n_at_exp + 2);
  g.nblk = (N + DEC_DWN * 16 - 1) / (DEC_DWN * 16);
  g.dks = 1; g.bkx = 128;
  if (g.dtm > 0) {
    g.dks = rmx_split_k_for(g.nblk * g.dec_ymax, n_dec_rows);
    g.bkx = rmx_decode_bk64(g.dtm, g.dks) ? 64 : 128;
    if (g.bkx == 64 && (g.dtm != 4 || g.dks != 1)) g.bkx = 128;
  }
  if (g.dec_ymax > 65535 || g.at_cap > 65535) return false;
  size_t o = 0;
  g.off_padoff = o; o += rmx_al256((size_t)(n_as + 1) * sizeof(int));
  g.off_dec_e  = o; o += rmx_al256((size_t)n_as * sizeof(int));
  g.off_at_e   = o; o += rmx_al256((size_t)g.at_cap * sizeof(int));
  g.off_at_mt  = o; o += rmx_al256((size_t)g.at_cap * sizeof(int));
  g.off_counts = o; o += 256;
  g.off_scale  = o; o += rmx_al256((size_t)g.Pmax * sizeof(float));
  g.off_ahi    = o; o += rmx_al256((size_t)g.Pmax * K);
  g.off_alo    = o; o += rmx_al256((size_t)g.Pmax * K);
  g.off_part = o;
  if (g.dtm > 0 && g.dks > 1)
    o += rmx_al256((size_t)g.dks * g.dec_ymax * (size_t)(DEC_MTILE * g.dtm) * N * sizeof(float));
  g.off_cnt = o;
  if (g.dtm > 0 && g.dks > 1) o += rmx_al256((size_t)g.dec_ymax * g.nblk * sizeof(int));
  g.total = o;
  return true;
}

size_t rdna4_mxfp4_radiance_moe_ws_bytes(int n_as, int n_used, int n_tok, int N, int K) {
  rmx_moe_geom g;
  return rmx_moe_geom_make(g, n_as, n_used, n_tok, N, K) ? g.total : 0;
}

bool rdna4_gemm_mxfp4_radiance_moe(const float* x, int64_t x_nb1, int64_t x_nb2, bool bcast,
                                   const int* bounds, const int* ids_dst, int n_as, int n_used,
                                   int n_tok, const void* packed, int64_t wstride,
                                   const void* const* wptrs, int n_mats,
                                   int64_t mat_off, float* dst, int N, int K, void* ws,
                                   size_t ws_bytes, hipStream_t stream) {
  if (n_mats < 1 || n_mats > 2) return false;
  rmx_moe_geom g;
  if (!rmx_moe_geom_make(g, n_as, n_used, n_tok, N, K) || ws == nullptr || ws_bytes < g.total) return false;
  auto *wsb = static_cast<unsigned char*>(ws);
  int *padoff = reinterpret_cast<int*>(wsb + g.off_padoff);
  int *dec_e  = reinterpret_cast<int*>(wsb + g.off_dec_e);
  int *at_e   = reinterpret_cast<int*>(wsb + g.off_at_e);
  int *at_mt  = reinterpret_cast<int*>(wsb + g.off_at_mt);
  int *counts = reinterpret_cast<int*>(wsb + g.off_counts);
  float *scale = reinterpret_cast<float*>(wsb + g.off_scale);
  unsigned char *a_hi = wsb + g.off_ahi, *a_lo = wsb + g.off_alo;

  hipLaunchKernelGGL(rmx_moe_plan_kernel, dim3(1), dim3(256), 0, stream, bounds, n_as, g.thr, g.at_cap,
                     padoff, dec_e, at_e, at_mt, counts);
  hipLaunchKernelGGL(rmx_quant_a_moe_kernel, dim3((unsigned)g.Pmax), dim3(256), 0, stream, x,
                     (long)x_nb1, (long)x_nb2, bcast ? 1 : 0, n_used, bounds, padoff, ids_dst, n_as,
                     g.thr, K, a_hi, a_lo, scale);

  for (int mi = 0; mi < n_mats; ++mi) {
  const auto *Wb = static_cast<const unsigned char*>(packed) + (size_t)mi * (size_t)mat_off;
  const unsigned char *Ws = Wb + rdna4_mxfp4_radiance_ws_offset(N, K);
  const unsigned char *Wref = Wb + rdna4_mxfp4_radiance_wref_offset(N, K);
  const long ldc = (long)n_mats * N, coff = (long)mi * N;

  if (g.dtm > 0) {
    float *P = nullptr;
    int *cnt = nullptr;
    if (g.dks > 1) {
      P = reinterpret_cast<float*>(wsb + g.off_part);
      cnt = reinterpret_cast<int*>(wsb + g.off_cnt);
      if (hipMemsetAsync(cnt, 0, (size_t)g.dec_ymax * g.nblk * sizeof(int), stream) != hipSuccess) return false;
    }
    const rmx_moe moe{padoff, bounds, ids_dst, dec_e, nullptr, counts, (long)wstride, ldc, coff, nullptr, n_as,
                     wptrs, (long)mi * (long)mat_off, (long)rdna4_mxfp4_radiance_ws_offset(N, K),
                     (long)rdna4_mxfp4_radiance_wref_offset(N, K)};
#define RMX_MOE_DEC(BK_, KS_, TM_)                                                                    \
    hipLaunchKernelGGL((rmx_gemm_decode<DEC_DWN, BK_, KS_, TM_, false, true, true, float>),         \
                       dim3(g.nblk, g.dec_ymax, KS_), dim3(DEC_DWN * 32), 0, stream, a_hi, a_lo,     \
                       Wb, Ws, Wref, scale, P, cnt, dst, 0, N, K, 1 << 30, 1 << 30, 0L, moe)
#define RMX_MOE_DEC_TM(BK_, KS_)                                                                      \
    if (g.dtm == 1)      RMX_MOE_DEC(BK_, KS_, 1);                                                    \
    else if (g.dtm == 2) RMX_MOE_DEC(BK_, KS_, 2);                                                    \
    else if (g.dtm == 3) RMX_MOE_DEC(BK_, KS_, 3);                                                    \
    else                 RMX_MOE_DEC(BK_, KS_, 4)
    if (g.dks <= 1) {
      if (g.bkx == 64) { RMX_MOE_DEC(64, 1, 4); }
      else             { RMX_MOE_DEC_TM(128, 1); }
    } else if (g.dks == 2) {
      RMX_MOE_DEC_TM(128, 2);
    } else {
      RMX_MOE_DEC_TM(128, 4);
    }
#undef RMX_MOE_DEC_TM
#undef RMX_MOE_DEC
  }
  if (g.need_at) {
    const rmx_moe moe{padoff, bounds, ids_dst, at_e, at_mt, counts + 1, (long)wstride, ldc, coff, nullptr, n_as,
                     wptrs, (long)mi * (long)mat_off, (long)rdna4_mxfp4_radiance_ws_offset(N, K),
                     (long)rdna4_mxfp4_radiance_wref_offset(N, K)};
    constexpr int B_ = BNF_OF(2);
    hipLaunchKernelGGL((rmx_gemm_atiled<2, 128, true, true, float>), dim3((N + B_ - 1) / B_, g.at_cap),
                       dim3(NTHREADS), 0, stream, a_hi, a_lo, Wb, Ws, Wref, scale, dst, 0, N, K,
                       1 << 30, 1 << 30, 0L, moe);
  }
  }
  return hipGetLastError() == hipSuccess;
}

// ── row mode (tiny n_tok): ids -> hi/lo quant -> ONE decode launch per matrix; no routing helper, no plan ──
// workspace: scale[T] | a_hi[T*K] | a_lo[T*K] | split-K partials [T][dks][N] | counters [T][nblk]
struct rmx_rows_geom {
  int T, nblk, dks;
  size_t off_scale, off_ahi, off_alo, off_part, off_cnt, total;
};
static bool rmx_rows_geom_make(rmx_rows_geom &g, int n_used, int n_tok, int N, int K) {
  if (n_used <= 0 || n_tok <= 0 || !rdna4_mxfp4_radiance_shape_ok(N, K) || rmx_decode_max_m() < 1) return false;
  g.T = n_tok * n_used;
  if (g.T > 65535) return false;
  g.nblk = (N + DEC_DWN * 16 - 1) / (DEC_DWN * 16);
  g.dks = rmx_rows_split_k(g.nblk * g.T);
  size_t o = 0;
  g.off_scale = o; o += rmx_al256((size_t)g.T * sizeof(float));
  g.off_ahi = o;   o += rmx_al256((size_t)g.T * K);
  g.off_alo = o;   o += rmx_al256((size_t)g.T * K);
  g.off_part = o;
  if (g.dks > 1) o += rmx_al256((size_t)g.T * g.dks * N * sizeof(float));
  g.off_cnt = o;
  if (g.dks > 1) o += rmx_al256((size_t)g.T * g.nblk * sizeof(int));
  g.total = o;
  return true;
}

size_t rdna4_mxfp4_radiance_moe_rows_ws_bytes(int n_used, int n_tok, int N, int K) {
  rmx_rows_geom g;
  return rmx_rows_geom_make(g, n_used, n_tok, N, K) ? g.total : 0;
}

bool rdna4_gemm_mxfp4_radiance_moe_rows(const float* x, int64_t x_nb1, int64_t x_nb2, bool bcast,
                                        const int* ids, int n_as, int n_used, int n_tok,
                                        const void* packed, int64_t wstride,
                                        const void* const* wptrs, int n_mats,
                                        int64_t mat_off, float* dst, int N, int K, void* ws,
                                        size_t ws_bytes, hipStream_t stream) {
  rmx_rows_geom g;
  if (n_mats < 1 || n_mats > 2 || n_as <= 0 || !rmx_rows_geom_make(g, n_used, n_tok, N, K) || ws == nullptr ||
      ws_bytes < g.total)
    return false;
  auto *wsb = static_cast<unsigned char*>(ws);
  float *scale = reinterpret_cast<float*>(wsb + g.off_scale);
  unsigned char *a_hi = wsb + g.off_ahi, *a_lo = wsb + g.off_alo;
  float *P = g.dks > 1 ? reinterpret_cast<float*>(wsb + g.off_part) : nullptr;
  int *cnt = g.dks > 1 ? reinterpret_cast<int*>(wsb + g.off_cnt) : nullptr;
  hipLaunchKernelGGL(rmx_quant_a_rows_kernel, dim3(g.T), dim3(256), 0, stream, x, (long)x_nb1, (long)x_nb2,
                     bcast ? 1 : 0, n_used, K, a_hi, a_lo, scale, cnt, g.dks > 1 ? g.T * g.nblk : 0);
  for (int mi = 0; mi < n_mats; ++mi) {
    const auto *Wb = static_cast<const unsigned char*>(packed) + (size_t)mi * (size_t)mat_off;
    const unsigned char *Ws = Wb + rdna4_mxfp4_radiance_ws_offset(N, K);
    const unsigned char *Wref = Wb + rdna4_mxfp4_radiance_wref_offset(N, K);
    const rmx_moe moe{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, (long)wstride,
                      (long)n_mats * N, (long)mi * N, ids, n_as,
                      wptrs, (long)mi * (long)mat_off, (long)rdna4_mxfp4_radiance_ws_offset(N, K),
                      (long)rdna4_mxfp4_radiance_wref_offset(N, K)};
#define RMX_ROWS_DEC(KS_)                                                                            \
    hipLaunchKernelGGL((rmx_gemm_decode<DEC_DWN, 128, KS_, 1, false, true, true, float>),           \
                       dim3(g.nblk, g.T, KS_), dim3(DEC_DWN * 32), 0, stream, a_hi, a_lo, Wb, Ws,   \
                       Wref, scale, P, cnt, dst, 0, N, K, 1 << 30, 1 << 30, 0L, moe)
    if (g.dks <= 1)      { RMX_ROWS_DEC(1); }
    else if (g.dks == 2) { RMX_ROWS_DEC(2); }
    else                 { RMX_ROWS_DEC(4); }
#undef RMX_ROWS_DEC
  }
  return hipGetLastError() == hipSuccess;
}
