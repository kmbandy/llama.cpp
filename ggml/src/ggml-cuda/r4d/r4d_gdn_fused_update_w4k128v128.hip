// r4d_gdn_fused_update_w4k128v128.hip -- the decode GDN step as ONE launch: the rolling
// speculative conv update, then the recurrent delta-rule update, separated by a grid-wide
// barrier instead of a kernel boundary.
//
// WHY. The two stages are 7-13 us kernels launched 2 x 48 layers per forward; at the measured
// ~3.3 us graph-dispatch gap the boundary costs as much as the smaller kernel. They cannot be
// block-fused: a recurrent head hv consumes q/k written by DIFFERENT conv units, so the handoff
// has to cross workgroups. The barrier is the decode GEMM's fused split-K pattern (arrive on an
// atomic, last arrival resets it, so the counter is zeroed for the next launch and graph replay
// needs no clearing pass) -- and it is deadlock-free BY CONSTRUCTION, not by luck: the grid is
// capped at FU_MAXWG workgroups of 8 waves, far under residency, and both phases WORK-LOOP over
// their items instead of assuming one item per block.
//
// The math is copied VERBATIM from r4d_gdn_conv_w4_h128_bf16.hip (decode entry) and
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip (ABF16 templated, NM=0 -- the production
// configuration; a caller wanting the fused row norm must use the unfused pair). Outputs are
// bit-identical to the pair: every (n, unit) / (n, head) item is independent, so the work-loop
// remap changes scheduling only. If either source kernel changes, this file has to follow -- the
// duplication is the price of not linking device code across TUs (no -fgpu-rdc in this build).
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_gdn_wmma.h"

#define FU_HD    128
#define FU_W       4
#define FU_ST   (FU_W - 1)
#define FU_DPL     4
#define FU_MAXT    1            // matches CU_MAXT: the x preload band of the decode conv
#define FU_K     128
#define FU_V     128
#define FU_HK     64
#define FU_MAXWG  32            // 8-wave workgroups; 32 is <= 1 per WGP, always resident

__device__ __forceinline__ float fu_silu(float x) { return x / (1.0f + __expf(-x)); }
__device__ __forceinline__ float fu_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}
template <int ABF16>
__device__ __forceinline__ float fu_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

template <int ABF16>
__global__ __launch_bounds__(256)
void r4d_gdn_fused_update_kernel(
    const unsigned short* __restrict__ x, long xpitch,
    const unsigned short* __restrict__ wgt, const unsigned short* __restrict__ bias,
    unsigned short* __restrict__ cstate, long cs_seq, long cs_dim, long cs_tok,
    int state_len_max, const int* __restrict__ cache_idx, long ci_stride,
    const int* __restrict__ naccept, const int* __restrict__ cu,
    int N, int H, int Hg, int maxq,
    unsigned short* __restrict__ q, unsigned short* __restrict__ k,
    unsigned short* __restrict__ v,
    const void* __restrict__ av, const void* __restrict__ bv, long ab_stride,
    const float* __restrict__ A_log, const float* __restrict__ dt_bias,
    float* __restrict__ state, long st_slot, long st_head,
    unsigned short* __restrict__ o,
    const int* __restrict__ sidx, long sidx_stride,
    float scale, float sp_thr, int* __restrict__ bar, int numWG, int o_rows)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int units = 2 * Hg + H;

  // ---- phase 1: the decode conv, one warp per (sequence, unit), work-looped -----------------
  for (int item = blockIdx.x * 8 + warp; item < N * units; item += numWG * 8) {
    const int n = item / units, unit = item - n * units;
    const int bos = cu[n], slen = cu[n + 1] - bos;
    if (slen <= 0) continue;
    const int slot = cache_idx[(size_t)n * ci_stride];
    if (slot <= 0) continue;

    const int off = naccept ? (naccept[n] - 1) : 0;
    const int slen_eff = state_len_max - (maxq - slen);
    const int d0 = unit * FU_HD + lane * FU_DPL;

    float wt[FU_W][FU_DPL], bs[FU_DPL];
#pragma unroll
    for (int c = 0; c < FU_DPL; ++c) {
      const ushort4 w4 = *(const ushort4*)(wgt + (size_t)(d0 + c) * FU_W);
      const unsigned short* ws = (const unsigned short*)&w4;
#pragma unroll
      for (int i = 0; i < FU_W; ++i) wt[i][c] = bf2f(ws[i]);
      bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
    }

    unsigned short* sp = cstate + (size_t)slot * cs_seq + (size_t)d0 * cs_dim;
    float win[FU_W][FU_DPL];
    unsigned short hist[FU_ST][FU_DPL];
#pragma unroll
    for (int i = 0; i < FU_ST; ++i)
#pragma unroll
      for (int c = 0; c < FU_DPL; ++c) {
        const unsigned short raw = sp[(size_t)c * cs_dim + (size_t)(off + i) * cs_tok];
        hist[i][c] = raw;
        win[i][c] = bf2f(raw);
      }

    const int hq = unit, hk2 = unit - Hg, hv2 = unit - 2 * Hg;
    const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

    ushort4 xb[FU_MAXT];
    const bool xpre = (slen <= FU_MAXT);
#pragma unroll
    for (int t = 0; t < FU_MAXT; ++t)
      if (xpre && t < slen) xb[t] = *(const ushort4*)(x + (size_t)(bos + t) * xpitch + d0);

    for (int t = 0; t < slen; ++t) {
      const ushort4 raw = xpre ? xb[t]
                               : *(const ushort4*)(x + (size_t)(bos + t) * xpitch + d0);
      win[FU_W - 1][0] = bf2f(raw.x); win[FU_W - 1][1] = bf2f(raw.y);
      win[FU_W - 1][2] = bf2f(raw.z); win[FU_W - 1][3] = bf2f(raw.w);

      unsigned short out[FU_DPL];
#pragma unroll
      for (int c = 0; c < FU_DPL; ++c) {
        float acc = bs[c];
#pragma unroll
        for (int i = 0; i < FU_W; ++i) acc += wt[i][c] * win[i][c];
        out[c] = f2bf(fu_silu(acc));
      }
      unsigned short* dst = (region == 2)
          ? (v + ((size_t)(bos + t) * H + hv2) * FU_HD + lane * FU_DPL)
          : ((region ? k : q) + ((size_t)(bos + t) * Hg + (region ? hk2 : hq)) * FU_HD
             + lane * FU_DPL);
      *(ushort4*)dst = make_ushort4(out[0], out[1], out[2], out[3]);

#pragma unroll
      for (int i = 0; i < FU_W - 1; ++i)
#pragma unroll
        for (int c = 0; c < FU_DPL; ++c) win[i][c] = win[i + 1][c];
    }

    const int VAL = FU_W - 2;
#pragma unroll
    for (int c = 0; c < FU_DPL; ++c) {
      unsigned short* dst = sp + (size_t)c * cs_dim;
      for (int i = 0; i < slen_eff; ++i) {
        unsigned short val;
        if (i + slen < slen_eff) {
          val = hist[i + 1][c];
        } else {
          const int ti = i - VAL;
          val = (xpre && ti >= 0 && ti < slen)
                    ? ((const unsigned short*)&xb[ti])[c]
                    : x[(size_t)(bos + ti) * xpitch + d0 + c];
        }
        dst[(size_t)i * cs_tok] = val;
      }
    }
  }

  // ---- grid barrier: arrive, last resets -- counter is zero again for the next launch -------
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    if (atomicAdd(bar, 1) == numWG - 1) {
      __threadfence();
      atomicExch(bar, 0);
    } else {
      while (atomicAdd(bar, 0) != 0) { __builtin_amdgcn_s_sleep(8); }
    }
  }
  __syncthreads();
  __threadfence();

  // ---- phase 2: the recurrent update, one workgroup per (sequence, v-head), work-looped -----
  // 256 threads = the VS=1 shape: two threads per v row, a thread owns half the k range.
  const int row = tid >> 1, half = tid & 1, k0 = half * FU_HK;
  for (int item = blockIdx.x; item < N * H; item += numWG) {
    const int n = item / H, hv = item - n * H;
    const int bos = cu[n], T = cu[n + 1] - bos;
    if (T <= 0) continue;
    const int hk = hv / (H / Hg);
    const int it = naccept ? (naccept[n] - 1) : 0;
    const int si = sidx[(size_t)n * sidx_stride + it];
    if (si <= 0) continue;

    float h[FU_HK];
    {
      const float* p = state + (size_t)si * st_slot + (size_t)hv * st_head
                     + (size_t)row * FU_K + k0;
#pragma unroll
      for (int j = 0; j < FU_HK; j += 4) *(float4*)&h[j] = *(const float4*)(p + j);
    }
    const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];

    for (int t = 0; t < T; ++t) {
      const int tok = bos + t;
      const unsigned short* qp = q + ((size_t)tok * Hg + hk) * FU_K + k0;
      const unsigned short* kp = k + ((size_t)tok * Hg + hk) * FU_K + k0;

      float qq[FU_HK], kk[FU_HK];
#pragma unroll
      for (int j = 0; j < FU_HK; j += 8) {
        const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
        const unsigned short* sq2 = (const unsigned short*)&wq;
        const unsigned short* sk2 = (const unsigned short*)&wk;
#pragma unroll
        for (int e = 0; e < 8; ++e) { qq[j + e] = bf2f(sq2[e]); kk[j + e] = bf2f(sk2[e]); }
      }

      float sq = 0.0f, sk = 0.0f;
#pragma unroll
      for (int j = 0; j < FU_HK; ++j) { sq += qq[j] * qq[j]; sk += kk[j] * kk[j]; }
      sq += __shfl_xor(sq, 1, 32);
      sk += __shfl_xor(sk, 1, 32);
      const float qs = __frsqrt_rn(sq + 1e-6f) * scale, ks = __frsqrt_rn(sk + 1e-6f);
#pragma unroll
      for (int j = 0; j < FU_HK; ++j) { qq[j] *= qs; kk[j] *= ks; }

      const float g  = -alog * fu_softplus(
          fu_ab<ABF16>(av, (size_t)tok * ab_stride + hv) + dtb, sp_thr);
      const float bt = 1.0f / (1.0f + __expf(-fu_ab<ABF16>(bv, (size_t)tok * ab_stride + hv)));
      const float eg = __expf(g);

      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < FU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
      dot += __shfl_xor(dot, 1, 32);

      const float u = (bf2f(v[((size_t)tok * H + hv) * FU_V + row]) - dot) * bt;

      float on = 0.0f;
#pragma unroll
      for (int j = 0; j < FU_HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }
      on += __shfl_xor(on, 1, 32);

      if (half == 0) o[((size_t)tok * H + hv) * FU_V + row] = f2bf(on);

      const int so = sidx[(size_t)n * sidx_stride + t];
      if (so > 0) {
        float* p = state + (size_t)so * st_slot + (size_t)hv * st_head
                 + (size_t)row * FU_K + k0;
#pragma unroll
        for (int j = 0; j < FU_HK; j += 4) *(float4*)(p + j) = *(const float4*)&h[j];
      }
    }
  }

  // ---- phase 3: zero the cudagraph pad rows of o ---------------------------------------------
  // The recurrent phase writes o only for rows a sequence covers, [0, cu[N]). core_attn_out is
  // the padded batch (o_rows), and vLLM zero-fills it before this op precisely because a
  // torch.empty there leaks garbage from the pad rows into the layers below (vLLM PR 28182).
  // Zeroing the tail here lets the caller drop that fill: rows [cu[N], o_rows), H*V bf16 each,
  // strided over the workgroups, ushort4 per thread. Disjoint from the rows phase 2 wrote.
  {
    const int first = cu[N];
    const int rowv4 = (H * FU_V) >> 2;
    for (int r = first + blockIdx.x; r < o_rows; r += numWG) {
      ushort4* dst = (ushort4*)(o + (size_t)r * H * FU_V);
      for (int i = tid; i < rowv4; i += 256) dst[i] = make_ushort4(0, 0, 0, 0);
    }
  }
}

extern "C" int r4d_gdn_fused_update_w4k128v128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, const void* cu, int N, int H, int Hg,
    int K, int V, int width, int max_query_len, void* q, void* k, void* v,
    const void* a, const void* b, long ab_stride, int ab_is_bf16,
    const void* A_log, const void* dt_bias, void* state, long st_slot, long st_head,
    void* o, const void* sidx, long sidx_stride, float scale, float softplus_thr,
    void* barrier_cnt, int o_rows, void* stream)
{
  if (K != FU_K || V != FU_V || width != FU_W) return -1;
  if (H % Hg) return -2;
  if (!barrier_cnt) return -3;
  int numWG = N * H;
  if (numWG > FU_MAXWG) numWG = FU_MAXWG;
  if (numWG < 1) numWG = 1;
  dim3 grid(numWG), blk(256);
#define FU_LAUNCH(AB)                                                                         \
  hipLaunchKernelGGL((r4d_gdn_fused_update_kernel<AB>), grid, blk, 0, (hipStream_t)stream,    \
                     (const unsigned short*)x, xpitch, (const unsigned short*)wgt,            \
                     (const unsigned short*)bias, (unsigned short*)cstate, cs_seq, cs_dim,    \
                     cs_tok, state_len_max, (const int*)cache_idx, ci_stride,                 \
                     (const int*)num_accepted, (const int*)cu, N, H, Hg, max_query_len,       \
                     (unsigned short*)q, (unsigned short*)k, (unsigned short*)v,              \
                     a, b, ab_stride, (const float*)A_log, (const float*)dt_bias,             \
                     (float*)state, st_slot, st_head, (unsigned short*)o,                     \
                     (const int*)sidx, sidx_stride, scale, softplus_thr,                      \
                     (int*)barrier_cnt, numWG, o_rows)
  if (ab_is_bf16) FU_LAUNCH(1); else FU_LAUNCH(0);
#undef FU_LAUNCH
  return (int)hipGetLastError();
}
