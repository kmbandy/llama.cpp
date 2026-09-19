// r4d_gdn_conv_w4_h128_bf16.hip -- the gated-delta-net depthwise causal convolution and the work fused around
// it, one entry point per serving phase.
//
// PREFILL (r4d_gdn_conv_prep): everything the layer does between the qkv projection and the
// chunked scan, in one kernel:
//
//     conv(x)        depthwise causal convolution, width 4, silu, varlen, with the conv-state
//                    cache read at a sequence's first chunk and rewritten from its last tokens
//     split          the packed [q | k | v] channel range into three contiguous tensors
//     l2norm         over each 128-dim q and k head
//     gate           g = -exp(A_log) . softplus(a + dt_bias), beta = sigmoid(b)
//     cumsum         g summed along the chunk, which is what the scan actually consumes
//
// It replaces causal_conv1d_fn + fused_post_conv_prep + chunk_local_cumsum. The pair it replaces
// spends most of its time moving the conv output: causal_conv1d_fn writes all conv_dim channels of
// every token to HBM and fused_post_conv_prep reads every one of them straight back, which at this
// model's shape is 34 MB out and 34 MB in per layer per rank for a tensor that is dead two
// microseconds later. Here the conv output only ever exists in registers.
//
// The layout falls out of the l2norm. A workgroup is HPB waves, one wave per 128-dim head, one
// lane per four dims -- so the norm of a head is a wave reduction with no LDS and no barrier, and
// four contiguous bf16 per lane is one 8-byte access on both sides. Channel groups are counted in
// heads rather than in channels so that a group never straddles the q/k/v boundaries (they fall on
// head multiples at every tensor-parallel size).
//
// Tokens are blocked by the chunk, 64, because the cumsum is per chunk: the whole scan is then a
// wave-level prefix over 32 lanes holding two tokens each, and no workgroup needs another's
// partial sums. The conv reads three tokens of halo per block, 4.7% of the traffic, which is
// cheaper than the barrier a shared staging buffer would need.
//
// The conv output is rounded to bf16 BEFORE the norm and the store, which is what the two kernels
// it replaces do (the conv writes a bf16 tensor and the prep reads it back). Keeping the fp32
// accumulator through the norm would be slightly finer and slightly different; matching is worth
// more than the last half ULP here, because it makes this kernel a drop-in that can be A/B'd
// against the pair on real weights.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_wmma.h"

#define CP_HD    128            // dims per head, q k and v alike
#define CP_BT     64            // token block = the scan's chunk, so the cumsum stays wave-local
#define CP_W       4            // conv width
#define CP_ST   (CP_W - 1)      // conv state depth
// CU_MAXT: tokens the DECODE update pre-loads into registers. Slower at 16, 10.6 us against
// 9.4: the sixteen ushort4 cost more in register pressure and predicated-load exec-mask work
// than the round trips they hide, and the compiler was already batching the loop's loads.
// 1 disables it (slen is num_spec+1 > 1 on every speculative step).
#ifndef CU_MAXT
#define CU_MAXT 1
#endif
// WVEC: read a channel's CP_W conv weights as one vector load instead of CP_W scalar ones.
#ifndef WVEC
#define WVEC 1
#endif
#define CP_DPL     4            // dims per lane (CP_HD / 32)

__device__ __forceinline__ float cp_silu(float z) {
  return z / (1.0f + __expf(-z));
}

// softplus, in the two-sided form FLA uses, with the same large-x cutoff
__device__ __forceinline__ float cp_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x)))
                    : __logf(1.0f + __expf(x));
}

// a and b arrive as the layer holds them: a chunk() of the ba projection, so bf16 and strided by
// 2*H, not the fp32 contiguous pair FLA's own kernels are usually handed. Reading them where they
// lie is what keeps this a single kernel -- a .float().contiguous() per tensor per layer would put
// two more launches and 0.6 MB back on the path this exists to shorten.
template <int ABF16>
__device__ __forceinline__ float cp_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

template <int HPB, int ABF16>
__global__ __launch_bounds__(HPB * 32)
void r4d_gdn_conv_prep_kernel(const unsigned short* __restrict__ x, long xpitch,
                              const unsigned short* __restrict__ wgt,
                              const unsigned short* __restrict__ bias,
                              unsigned short* __restrict__ cstate,
                              long cs_seq, long cs_dim, long cs_tok,
                              const int* __restrict__ cache_idx, long ci_stride,
                              const unsigned char* __restrict__ has_init,
                              const void* __restrict__ av, const void* __restrict__ bv,
                              long ab_stride,
                              const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                              unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                              unsigned short* __restrict__ v,
                              float* __restrict__ go, float* __restrict__ beta,
                              const int* __restrict__ cu, int N, int H, int Hg, float sp_thr)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;

  // flat chunk id -> (sequence, chunk); see r4d_gdn_kkt_solve_k128_c64_bf16.hip for why the grid is an upper
  // bound rather than the exact count
  int n = -1, t0 = 0, bos = 0, rows = 0, slen = 0;
  {
    int acc = 0, want = blockIdx.x;
    for (int i = 0; i < N; ++i) {
      const int b = cu[i], e = cu[i + 1];
      const int nc = (e - b + CP_BT - 1) / CP_BT;
      if (want < acc + nc) { n = i; bos = b; slen = e - b; t0 = (want - acc) * CP_BT;
                             rows = min(CP_BT, slen - t0); break; }
      acc += nc;
    }
    if (n < 0) return;
  }
  const bool first = (t0 == 0);

  // ---- the gating half, on one workgroup per chunk ----------------------------------------
  // g and beta come from a and b alone, so this is independent of the convolution; it rides on
  // the y == 0 workgroup rather than costing a launch of its own.
  if (blockIdx.y == 0) {
    for (int hv = warp; hv < H; hv += HPB) {
      const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];
      const int i0 = 2 * lane, i1 = i0 + 1;
      float g0 = 0.0f, g1 = 0.0f;
      if (i0 < rows)
        g0 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i0) * ab_stride + hv)
                                 + dtb, sp_thr);
      if (i1 < rows)
        g1 = -alog * cp_softplus(cp_ab<ABF16>(av, (size_t)(bos + t0 + i1) * ab_stride + hv)
                                 + dtb, sp_thr);
      // inclusive wave scan of the per-lane pair sums, made exclusive by subtracting the pair
      float p = g0 + g1;
#pragma unroll
      for (int d = 1; d < 32; d <<= 1) {
        const float o = __shfl_up(p, d, 32);
        if (lane >= d) p += o;
      }
      const float pre = p - (g0 + g1);            // sum of all lanes before this one
      if (i0 < rows) {
        go[(size_t)(bos + t0 + i0) * H + hv] = pre + g0;
        beta[(size_t)(bos + t0 + i0) * H + hv] = 1.0f / (1.0f + __expf(
            -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i0) * ab_stride + hv)));
      }
      if (i1 < rows) {
        go[(size_t)(bos + t0 + i1) * H + hv] = pre + g0 + g1;
        beta[(size_t)(bos + t0 + i1) * H + hv] = 1.0f / (1.0f + __expf(
            -cp_ab<ABF16>(bv, (size_t)(bos + t0 + i1) * ab_stride + hv)));
      }
    }
  }

  // ---- the convolution half ----------------------------------------------------------------
  const int unit = blockIdx.y * HPB + warp;       // global 128-dim head unit
  const int d0 = unit * CP_HD + lane * CP_DPL;    // this lane's first channel
  const unsigned short* xp = x + (size_t)(bos + t0) * xpitch + d0;

  float wt[CP_W][CP_DPL], bs[CP_DPL];
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
#pragma unroll
    for (int i = 0; i < CP_W; ++i) wt[i][c] = bf2f(wgt[(size_t)(d0 + c) * CP_W + i]);
    bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
  }

  // window[i] holds x[t - (W-1) + i]; seed it from the previous tokens, or from the conv state
  // cache when this is the first chunk of the sequence
  float win[CP_W][CP_DPL];
  {
    const bool init = first && has_init && has_init[n];
    const unsigned short* sp = init ? (cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                                       + (size_t)d0 * cs_dim) : nullptr;
#pragma unroll
    for (int i = 0; i < CP_ST; ++i) {              // i = 0,1,2  ->  x[t0-3+i]
      const int src = t0 - CP_ST + i;
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) {
        float val = 0.0f;
        if (src >= 0) val = bf2f(x[(size_t)(bos + src) * xpitch + d0 + c]);
        else if (init) val = bf2f(sp[(size_t)c * cs_dim + (size_t)(CP_ST + src) * cs_tok]);
        win[i][c] = val;
      }
    }
  }

  // ---- write the new conv state, from x, at the sequence's first chunk ---------------------
  // Ordering: the cache line is updated in place, so the read above and this write must be in the
  // same workgroup for the same channels -- which they are, and this is why the FIRST chunk owns
  // the write even though the tokens it stores belong to the last one.
  if (first && cache_idx) {
    unsigned short* sp = cstate + (size_t)cache_idx[(size_t)n * ci_stride] * cs_seq
                       + (size_t)d0 * cs_dim;
    const bool init = has_init && has_init[n];
    // Channel-major, because the cache is [dim][slot] with the slot contiguous: this lane owns
    // CP_DPL consecutive channels, so its CP_DPL*CP_ST shorts are one contiguous run and the wave
    // covers 32 of those back to back. The slot-major order this replaces made every store a
    // strided 2-byte scalar.
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
#pragma unroll
      for (int i = 0; i < CP_ST; ++i) {
        const int src = slen - CP_ST + i;
        unsigned short val = 0;
        if (src >= 0) val = x[(size_t)(bos + src) * xpitch + d0 + c];
        else if (init) val = sp[(size_t)c * cs_dim + (size_t)(CP_ST + src) * cs_tok];
        sp[(size_t)c * cs_dim + (size_t)i * cs_tok] = val;
      }
    }
  }

  const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
  const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

  for (int t = 0; t < rows; ++t) {
    float o[CP_DPL];
    {
      const unsigned short* xr = xp + (size_t)t * xpitch;
      const ushort4 raw = *(const ushort4*)xr;
      win[CP_W - 1][0] = bf2f(raw.x); win[CP_W - 1][1] = bf2f(raw.y);
      win[CP_W - 1][2] = bf2f(raw.z); win[CP_W - 1][3] = bf2f(raw.w);
    }
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
      float acc = bs[c];
#pragma unroll
      for (int i = 0; i < CP_W; ++i) acc += wt[i][c] * win[i][c];
      // the pair this replaces rounds here, on the way to HBM; keep that
      o[c] = bf2f(f2bf(cp_silu(acc)));
    }

    if (region < 2) {
      float ss = 0.0f;
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) ss += o[c] * o[c];
#pragma unroll
      for (int d = 16; d; d >>= 1) ss += __shfl_xor(ss, d, 32);
      const float inv = 1.0f / sqrtf(ss + 1e-6f);   // L2NORM_EPS, as in the kernel it replaces
      unsigned short* dst = (region ? k : q)
                          + ((size_t)(bos + t0 + t) * Hg + (region ? hk : hq)) * CP_HD
                          + lane * CP_DPL;
      *(ushort4*)dst = make_ushort4(f2bf(o[0] * inv), f2bf(o[1] * inv),
                                    f2bf(o[2] * inv), f2bf(o[3] * inv));
    } else {
      unsigned short* dst = v + ((size_t)(bos + t0 + t) * H + hv) * CP_HD + lane * CP_DPL;
      *(ushort4*)dst = make_ushort4(f2bf(o[0]), f2bf(o[1]), f2bf(o[2]), f2bf(o[3]));
    }

#pragma unroll
    for (int i = 0; i < CP_W - 1; ++i)
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) win[i][c] = win[i + 1][c];
  }
}

extern "C" int r4d_gdn_conv_prep_w4_h128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, const void* cache_idx, long ci_stride,
    const void* has_init,
    const void* a, const void* b, long ab_stride, int ab_is_bf16, const void* A_log,
    const void* dt_bias, void* q, void* k, void* v, void* g, void* beta, const void* cu,
    int N, int T, int H, int Hg, int K, int V, int width, float softplus_thr, void* stream)
{
  if (K != CP_HD || V != CP_HD || width != CP_W) return -1;
  const int units = 2 * Hg + H;
  int hpb = 8;
  while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
  const int nchunk_ub = (T + CP_BT - 1) / CP_BT + N;
  dim3 grid(nchunk_ub, units / hpb), blk(hpb * 32);
#define CP_LAUNCH(P, AB)                                                                      \
  hipLaunchKernelGGL((r4d_gdn_conv_prep_kernel<P, AB>), grid, blk, 0, (hipStream_t)stream,     \
                     (const unsigned short*)x, xpitch, (const unsigned short*)wgt,            \
                     (const unsigned short*)bias, (unsigned short*)cstate,                    \
                     cs_seq, cs_dim, cs_tok, (const int*)cache_idx, ci_stride,                \
                     (const unsigned char*)has_init, a, b, ab_stride,                         \
                     (const float*)A_log, (const float*)dt_bias, (unsigned short*)q,          \
                     (unsigned short*)k, (unsigned short*)v, (float*)g, (float*)beta,         \
                     (const int*)cu, N, H, Hg, softplus_thr)
#define CP_SWITCH(AB)                                                                         \
  switch (hpb) {                                                                              \
    case 8: CP_LAUNCH(8, AB); break;                                                          \
    case 4: CP_LAUNCH(4, AB); break;                                                          \
    case 2: CP_LAUNCH(2, AB); break;                                                          \
    default: CP_LAUNCH(1, AB); break;                                                         \
  }
  if (ab_is_bf16) { CP_SWITCH(1); } else { CP_SWITCH(0); }
#undef CP_SWITCH
#undef CP_LAUNCH
  return (int)hipGetLastError();
}

// ---------------------------------------------------------------------------------------------
// The decode-time convolution. Same depthwise causal filter, but the tokens of a step are a
// speculative window rather than a chunk, and the state cache is a rolling buffer of
// width-1 + num_spec entries rather than width-1: a step reads its window starting at the slot
// the last ACCEPTED token left behind, and rewrites the buffer shifted by one so the next step
// can do the same. That offset is the only thing the caller has to get right, and it is why this
// is a separate entry point rather than a flag on the one above.
//
// It writes q, k and v in their own layouts, which also deletes the torch.cat the layer used to
// need to make the three views contiguous for the recurrent kernel.
// ---------------------------------------------------------------------------------------------
template <int HPB>
__global__ __launch_bounds__(HPB * 32)
void r4d_gdn_conv_update_kernel(const unsigned short* __restrict__ x, long xpitch,
                                const unsigned short* __restrict__ wgt,
                                const unsigned short* __restrict__ bias,
                                unsigned short* __restrict__ cstate,
                                long cs_seq, long cs_dim, long cs_tok, int state_len_max,
                                const int* __restrict__ cache_idx, long ci_stride,
                                const int* __restrict__ naccept,
                                unsigned short* __restrict__ q, unsigned short* __restrict__ k,
                                unsigned short* __restrict__ v,
                                const int* __restrict__ cu, int H, int Hg, int maxq)
{
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int n = blockIdx.x;
  const int bos = cu[n], slen = cu[n + 1] - bos;
  if (slen <= 0) return;
  const int slot = cache_idx[(size_t)n * ci_stride];
  if (slot <= 0) return;                          // NULL_BLOCK_ID / pad slot

  const int off = naccept ? (naccept[n] - 1) : 0;
  // the cache depth this step actually uses; FLA shrinks the caller's max_query_len bound by the
  // tokens this sequence did not bring
  const int slen_eff = state_len_max - (maxq - slen);

  const int unit = blockIdx.y * HPB + warp;
  const int d0 = unit * CP_HD + lane * CP_DPL;

  // The weight of one channel is CP_W contiguous shorts, so a lane's four channels are four
  // vector loads, not sixteen scalar ones.
  float wt[CP_W][CP_DPL], bs[CP_DPL];
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
    if (WVEC && CP_W == 4) {
      const ushort4 w4 = *(const ushort4*)(wgt + (size_t)(d0 + c) * CP_W);
      const unsigned short* ws = (const unsigned short*)&w4;
#pragma unroll
      for (int i = 0; i < CP_W; ++i) wt[i][c] = bf2f(ws[i]);
    } else {
#pragma unroll
      for (int i = 0; i < CP_W; ++i) wt[i][c] = bf2f(wgt[(size_t)(d0 + c) * CP_W + i]);
    }
    bs[c] = bias ? bf2f(bias[d0 + c]) : 0.0f;
  }

  unsigned short* sp = cstate + (size_t)slot * cs_seq + (size_t)d0 * cs_dim;
  float win[CP_W][CP_DPL];
  unsigned short hist[CP_ST][CP_DPL];             // the raw history, for the shifted rewrite
#pragma unroll
  for (int i = 0; i < CP_ST; ++i)
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
      const unsigned short raw = sp[(size_t)c * cs_dim + (size_t)(off + i) * cs_tok];
      hist[i][c] = raw;
      win[i][c] = bf2f(raw);
    }

  const int hq = unit, hk = unit - Hg, hv = unit - 2 * Hg;
  const int region = (unit < Hg) ? 0 : ((unit < 2 * Hg) ? 1 : 2);

  // This step's inputs, all issued before the first is waited on. With the load inside the loop
  // each of the (num_spec + 1) four-tap taps pays an HBM round trip that nothing hides: a decode
  // step launches one wave per channel group, so there is no other wave on the SIMD to run
  // meanwhile. They are also exactly the values the cache rewrite at the bottom re-reads.
  ushort4 xb[CU_MAXT];
  const bool xpre = (slen <= CU_MAXT);
#pragma unroll
  for (int t = 0; t < CU_MAXT; ++t)
    if (xpre && t < slen) xb[t] = *(const ushort4*)(x + (size_t)(bos + t) * xpitch + d0);

  for (int t = 0; t < slen; ++t) {
    const ushort4 raw = xpre ? xb[t]
                             : *(const ushort4*)(x + (size_t)(bos + t) * xpitch + d0);
    win[CP_W - 1][0] = bf2f(raw.x); win[CP_W - 1][1] = bf2f(raw.y);
    win[CP_W - 1][2] = bf2f(raw.z); win[CP_W - 1][3] = bf2f(raw.w);

    unsigned short out[CP_DPL];
#pragma unroll
    for (int c = 0; c < CP_DPL; ++c) {
      float acc = bs[c];
#pragma unroll
      for (int i = 0; i < CP_W; ++i) acc += wt[i][c] * win[i][c];
      out[c] = f2bf(cp_silu(acc));
    }
    unsigned short* dst = (region == 2)
        ? (v + ((size_t)(bos + t) * H + hv) * CP_HD + lane * CP_DPL)
        : ((region ? k : q) + ((size_t)(bos + t) * Hg + (region ? hk : hq)) * CP_HD
           + lane * CP_DPL);
    *(ushort4*)dst = make_ushort4(out[0], out[1], out[2], out[3]);

#pragma unroll
    for (int i = 0; i < CP_W - 1; ++i)
#pragma unroll
      for (int c = 0; c < CP_DPL; ++c) win[i][c] = win[i + 1][c];
  }

  // Rewrite the cache shifted by one: the tail of the old history, then this step's tokens. The
  // first CP_ST-1 entries are exactly the history already in registers, so the rewrite costs no
  // extra reads -- and reading before writing is what makes the in-place update safe.
  // Channel-major for the same reason as the prefill path, and it matters more here: this is
  // width-1 + num_spec slots per channel, so the slot-major order made it 44 strided 2-byte
  // stores per lane where channel-major is one 88-byte run.
  const int VAL = CP_W - 2;
#pragma unroll
  for (int c = 0; c < CP_DPL; ++c) {
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

extern "C" int r4d_gdn_conv_update_w4_h128_bf16(
    const void* x, long xpitch, const void* wgt, const void* bias, void* cstate,
    long cs_seq, long cs_dim, long cs_tok, int state_len_max, const void* cache_idx,
    long ci_stride, const void* num_accepted, void* q, void* k, void* v, const void* cu,
    int N, int H, int Hg, int K, int V, int width, int max_query_len, void* stream)
{
  if (K != CP_HD || V != CP_HD || width != CP_W) return -1;
  const int units = 2 * Hg + H;
  int hpb = 8;
  while (hpb > 1 && (Hg % hpb || H % hpb)) hpb >>= 1;
  // A decode step is a handful of sequences, so the channel grouping is what decides whether the
  // grid covers the device: at 8 sequences and 8 heads per block this launches 40 workgroups for
  // 64 CUs. Narrow the block until there are enough of them.
  while (hpb > 1 && (long)N * (units / hpb) < 256) hpb >>= 1;
  dim3 grid(N, units / hpb), blk(hpb * 32);
#define CU_LAUNCH(P)                                                                          \
  hipLaunchKernelGGL((r4d_gdn_conv_update_kernel<P>), grid, blk, 0, (hipStream_t)stream,      \
                     (const unsigned short*)x, xpitch, (const unsigned short*)wgt,            \
                     (const unsigned short*)bias, (unsigned short*)cstate, cs_seq, cs_dim,    \
                     cs_tok, state_len_max, (const int*)cache_idx, ci_stride,                 \
                     (const int*)num_accepted, (unsigned short*)q, (unsigned short*)k,        \
                     (unsigned short*)v, (const int*)cu, H, Hg, max_query_len)
  switch (hpb) {
    case 8: CU_LAUNCH(8); break;
    case 4: CU_LAUNCH(4); break;
    case 2: CU_LAUNCH(2); break;
    default: CU_LAUNCH(1); break;
  }
#undef CU_LAUNCH
  return (int)hipGetLastError();
}
