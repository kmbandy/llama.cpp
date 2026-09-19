// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip -- the gated-delta-net recurrent state update, which is what
// decode runs where prefill runs the chunked scan. One kernel for the whole step:
//
//     g = -exp(A_log) . softplus(a + dt_bias)      beta = sigmoid(b)      q, k l2-normalised
//     S  *= e^g
//     u   = beta . (v - S k)                       S += u k^T              o = S q
//
// per token, against a PAGED state cache: the initial state is read from the slot the last
// accepted token wrote, and every candidate token writes its own slot, because which of them
// survives verification is not known until after this layer has run. It replaces FLA's
// fused_sigmoid_gating_delta_rule_update, and its non-speculative sibling falls out as the
// width-1 case.
//
// THE STATE TRAFFIC IS THE WHOLE COST. One [V,K] state is 64 KB, and a speculative step writes
// one per candidate token per head: at width 9 that is 113 MB per layer per rank, at the
// bandwidth limit of this part. So this kernel is a port that removes a Triton dependency and
// a kernel launch, not an arithmetic win, and it is written to keep the state resident rather
// than to be clever: one thread owns one v row and half its k range, so
//
//   * S k and S q are in-lane dot products closed by a single lane<->lane^1 exchange, not a
//     tree reduction;
//   * the outer-product update is 64 independent FMAs with no cross-lane traffic at all;
//   * a thread's 64 state elements are contiguous in k, so the load and every store is four
//     16-byte accesses over one 256-byte run.
//
// The state cache is fp32 -- this model's HF config sets mamba_ssm_dtype=float32, which vLLM
// honours over the `auto` mamba dtype, so the temporal state is fp32 while the conv state is
// bf16. That is also what makes the state traffic the cost: one [V,K] state is 64 KB, not 32.
//
// The slot and head strides are ARGUMENTS, not H*V*K: vLLM pads the mamba page so it matches the
// attention page size (here 421888 elements per slot against the 393216 the shape implies), and a
// kernel that derives the stride from the shape reads every slot but the first at the wrong
// offset -- which looks like a correct kernel producing noise.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_wmma.h"

#define RU_K   128              // head_k
#define RU_V   128              // head_v
#define RU_HK   64              // k elements per thread (RU_K / 2)
#define RU_THR (RU_V * 2)       // two threads per v row
// VS: how many workgroups share a head's 128 v rows. The grid is (sequences, heads), which at
// one sequence is 24 workgroups on 64 CUs -- 37% of the machine, and the kernel is not DRAM
// bound there (its whole working set fits the 64 MB last level). Nothing in a v row's update
// crosses to another v row: the two dot products contract over k, and the outer-product update
// is per (v, k). Only the fused RMS norm spans the row, so VS > 1 is compiled without it.
#ifndef VSMAX
#define VSMAX 4
#endif

__device__ __forceinline__ float ru_softplus(float x, float thr) {
  if (x > thr) return x;
  return (x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : __logf(1.0f + __expf(x));
}

// a and b are a chunk() of the ba projection: bf16, strided by 2*H. See r4d_gdn_conv_w4_h128_bf16.hip.
template <int ABF16>
__device__ __forceinline__ float ru_ab(const void* p, size_t i) {
  return ABF16 ? bf2f(((const unsigned short*)p)[i]) : ((const float*)p)[i];
}

// The layer's gated RMS norm, folded into the epilogue. A workgroup here owns one (sequence,
// head) and therefore the WHOLE 128-channel row of the output, which is exactly what an RMS needs
// -- so on this path the norm costs one workgroup reduction per token and no kernel launch, no
// second pass over the output, and no read of it back from HBM. The prefill path cannot do this
// (its workgroup owns half a row) and calls r4d_gdn_gated_rmsnorm_h128_bf16 instead.
__device__ __forceinline__ float ru_act(float z, int kind) {
  const float s = 1.0f / (1.0f + __expf(-z));
  return kind ? s : z * s;
}

template <int ABF16, int NORM, int VS>
__global__ __launch_bounds__(RU_V / VS * 2)
void r4d_gdn_recurrent_update_kernel(const unsigned short* __restrict__ q,
                                     const unsigned short* __restrict__ k,
                                     const unsigned short* __restrict__ v,
                                     const void* __restrict__ av,
                                     const void* __restrict__ bv, long ab_stride,
                                     const float* __restrict__ A_log,
                                     const float* __restrict__ dt_bias,
                                     float* __restrict__ state, long st_slot, long st_head,
                                     unsigned short* __restrict__ o,
                                     const int* __restrict__ cu,
                                     const int* __restrict__ sidx, long sidx_stride,
                                     const int* __restrict__ naccept,
                                     const unsigned short* __restrict__ zg,
                                     const float* __restrict__ nw, float neps, int nact,
                                     int H, int Hg, float scale, float sp_thr)
{
  constexpr int NR = RU_V / VS;         // v rows this workgroup owns
  constexpr int THR = NR * 2;
  static_assert(!(NORM && VS > 1), "the fused row norm needs the whole 128-wide row");
  __shared__ float wsum[THR / 32];
  const int n = blockIdx.x, hv = blockIdx.y;
  const int bos = cu[n], T = cu[n + 1] - bos;
  if (T <= 0) return;

  const int tid = threadIdx.x;
  const int row = (tid >> 1) + (VS > 1 ? blockIdx.z * NR : 0), half = tid & 1, k0 = half * RU_HK;
  const int hk = hv / (H / Hg);

  // the initial state is the one the last accepted token of the previous step wrote
  const int it = naccept ? (naccept[n] - 1) : 0;
  const int si = sidx[(size_t)n * sidx_stride + it];
  if (si <= 0) return;                                  // NULL_BLOCK_ID: nothing scheduled here

  float h[RU_HK];
  {
    const float* p = state + (size_t)si * st_slot + (size_t)hv * st_head
                   + (size_t)row * RU_K + k0;
#pragma unroll
    for (int j = 0; j < RU_HK; j += 4) *(float4*)&h[j] = *(const float4*)(p + j);
  }

  const float alog = __expf(A_log[hv]), dtb = dt_bias[hv];

  for (int t = 0; t < T; ++t) {
    const int tok = bos + t;
    const unsigned short* qp = q + ((size_t)tok * Hg + hk) * RU_K + k0;
    const unsigned short* kp = k + ((size_t)tok * Hg + hk) * RU_K + k0;

    float qq[RU_HK], kk[RU_HK];
#pragma unroll
    for (int j = 0; j < RU_HK; j += 8) {
      const uint4 wq = *(const uint4*)(qp + j), wk = *(const uint4*)(kp + j);
      const unsigned short* sq = (const unsigned short*)&wq;
      const unsigned short* sk = (const unsigned short*)&wk;
#pragma unroll
      for (int e = 0; e < 8; ++e) { qq[j + e] = bf2f(sq[e]); kk[j + e] = bf2f(sk[e]); }
    }

    // l2 norms: 64 in-lane terms, then one exchange with the lane holding the other half
    float sq = 0.0f, sk = 0.0f;
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { sq += qq[j] * qq[j]; sk += kk[j] * kk[j]; }
    sq += __shfl_xor(sq, 1, 32);
    sk += __shfl_xor(sk, 1, 32);
    const float qs = __frsqrt_rn(sq + 1e-6f) * scale, ks = __frsqrt_rn(sk + 1e-6f);
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { qq[j] *= qs; kk[j] *= ks; }

    const float g   = -alog * ru_softplus(
        ru_ab<ABF16>(av, (size_t)tok * ab_stride + hv) + dtb, sp_thr);
    const float bt  = 1.0f / (1.0f + __expf(-ru_ab<ABF16>(bv, (size_t)tok * ab_stride + hv)));
    const float eg  = __expf(g);

    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { h[j] *= eg; dot += h[j] * kk[j]; }
    dot += __shfl_xor(dot, 1, 32);

    const float u = (bf2f(v[((size_t)tok * H + hv) * RU_V + row]) - dot) * bt;

    float on = 0.0f;
#pragma unroll
    for (int j = 0; j < RU_HK; ++j) { h[j] += u * kk[j]; on += h[j] * qq[j]; }
    on += __shfl_xor(on, 1, 32);

    if (NORM) {
      // sum of squares over the row: each pair holds one channel, so one wave covers 16 of them
      float sq = (half == 0) ? on * on : 0.0f;
#pragma unroll
      for (int m = 16; m; m >>= 1) sq += __shfl_xor(sq, m, 32);
      if ((threadIdx.x & 31) == 0) wsum[threadIdx.x >> 5] = sq;
      __syncthreads();
      float tot = 0.0f;
#pragma unroll
      for (int i = 0; i < THR / 32; ++i) tot += wsum[i];
      const float sc = __frsqrt_rn(tot * (1.0f / RU_V) + neps);
      on *= sc * nw[row] * ru_act(bf2f(zg[((size_t)tok * H + hv) * RU_V + row]), nact);
      __syncthreads();                 // wsum is reused by the next token
    }

    if (half == 0) o[((size_t)tok * H + hv) * RU_V + row] = f2bf(on);

    const int so = sidx[(size_t)n * sidx_stride + t];
    if (so > 0) {
      float* p = state + (size_t)so * st_slot + (size_t)hv * st_head
               + (size_t)row * RU_K + k0;
#pragma unroll
      for (int j = 0; j < RU_HK; j += 4) *(float4*)(p + j) = *(const float4*)&h[j];
    }
  }
}

extern "C" int r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
    const void* q, const void* k, const void* v, const void* a, const void* b,
    long ab_stride, int ab_is_bf16, const void* A_log, const void* dt_bias, void* state,
    long state_slot_stride, long state_head_stride, void* o, const void* cu,
    const void* ssm_state_indices, long indices_stride, const void* num_accepted,
    const void* z_gate, const void* norm_weight, float norm_eps, int norm_act,
    int N, int H, int Hg, int K, int V, float scale, float softplus_thr, void* stream)
{
  if (K != RU_K || V != RU_V) return -1;
  if (H % Hg) return -2;
  const int nm0 = (z_gate && norm_weight) ? 1 : 0;
  // Fill the machine: at one sequence the (N, H) grid is 24 workgroups. Split a head's v rows
  // until there are at least 64 workgroups, but never past four (that is 64 threads, two waves).
  int vs = 1;
  if (!nm0) { const int want = (64 + N * H - 1) / (N * H);
              vs = want >= 4 ? 4 : (want >= 2 ? 2 : 1);
              if (vs > VSMAX) vs = VSMAX; }
  dim3 grid(N, H, vs), blk(RU_V / vs * 2);
#define RU_LAUNCH(AB, NM, VSC)                                                                \
  hipLaunchKernelGGL((r4d_gdn_recurrent_update_kernel<AB, NM, VSC>), grid, blk, 0,            \
                     (hipStream_t)stream,                                                     \
                     (const unsigned short*)q, (const unsigned short*)k,                      \
                     (const unsigned short*)v, a, b, ab_stride,                               \
                     (const float*)A_log, (const float*)dt_bias, (float*)state,               \
                     state_slot_stride, state_head_stride,                                    \
                     (unsigned short*)o, (const int*)cu, (const int*)ssm_state_indices,       \
                     indices_stride, (const int*)num_accepted,                                \
                     (const unsigned short*)z_gate, (const float*)norm_weight, norm_eps,      \
                     norm_act, H, Hg, scale, softplus_thr)
#define RU_PICKV(AB)                                                                          \
  do { if (vs == 4)      RU_LAUNCH(AB, 0, 4);                                                 \
       else if (vs == 2) RU_LAUNCH(AB, 0, 2);                                                 \
       else              RU_LAUNCH(AB, 0, 1); } while (0)
  if (ab_is_bf16) { if (nm0) RU_LAUNCH(1, 1, 1); else RU_PICKV(1); }
  else            { if (nm0) RU_LAUNCH(0, 1, 1); else RU_PICKV(0); }
#undef RU_PICKV
#undef RU_LAUNCH
  return (int)hipGetLastError();
}
