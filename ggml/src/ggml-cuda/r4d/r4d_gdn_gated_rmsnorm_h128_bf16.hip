// r4d_gdn_gated_rmsnorm_h128_bf16.hip -- the gated RMS norm a gated-delta-net layer applies to its own
// output before the output projection:
//
//     out[r][d] = x[r][d] . rsqrt(mean_d(x[r][:]^2) + eps) . w[d] . act(z[r][d])
//
// one row per (token, head), 128 wide. It replaces FLA's `rmsnorm_fn` (mamba/ops/layernorm_gated),
// which is the last Triton kernel in the layer.
//
// It exists as its own kernel only because the chunked scan cannot host it: that kernel's
// workgroup owns 64 of the 128 v channels of a row (BV=64, and widening it to 128 leaves 24
// workgroups on 64 CUs), while an RMS needs the whole row. The DECODE kernel does own the whole
// row, so there the same arithmetic is fused into the epilogue and this kernel is not launched at
// all -- see r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip.
//
// One wave per row, four channels per lane: the reduction is five shuffles with no LDS and no
// barrier, and both loads and the store are one 8-byte access per lane.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_wmma.h"

#define GN_D    128             // channels per row
#define GN_DPL    4             // channels per lane
#define GN_NW     8             // rows in flight per workgroup

__device__ __forceinline__ float gn_act(float z, int kind) {
  const float s = 1.0f / (1.0f + __expf(-z));
  return kind ? s : z * s;      // 1 = sigmoid, 0 = silu / swish
}

template <int ACT>
__global__ __launch_bounds__(GN_NW * 32)
void r4d_gdn_gated_rmsnorm_kernel(const unsigned short* __restrict__ x,
                                  const unsigned short* __restrict__ z,
                                  const float* __restrict__ w,
                                  unsigned short* __restrict__ o,
                                  long rows, long xrow, long zrow, long orow, float eps)
{
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const long r = (long)blockIdx.x * GN_NW + warp;
  if (r >= rows) return;
  const int d0 = lane * GN_DPL;

  const ushort4 xv = *(const ushort4*)(x + r * xrow + d0);
  float v[GN_DPL] = {bf2f(xv.x), bf2f(xv.y), bf2f(xv.z), bf2f(xv.w)};

  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < GN_DPL; ++c) ss += v[c] * v[c];
#pragma unroll
  for (int m = 16; m; m >>= 1) ss += __shfl_xor(ss, m, 32);
  const float scale = __frsqrt_rn(ss * (1.0f / GN_D) + eps);

  const ushort4 zv = *(const ushort4*)(z + r * zrow + d0);
  const float zz[GN_DPL] = {bf2f(zv.x), bf2f(zv.y), bf2f(zv.z), bf2f(zv.w)};
  const float4 ww = *(const float4*)(w + d0);
  const float wv[GN_DPL] = {ww.x, ww.y, ww.z, ww.w};

  unsigned short out[GN_DPL];
#pragma unroll
  for (int c = 0; c < GN_DPL; ++c)
    out[c] = f2bf(v[c] * scale * wv[c] * gn_act(zz[c], ACT));
  *(ushort4*)(o + r * orow + d0) = make_ushort4(out[0], out[1], out[2], out[3]);
}

extern "C" int r4d_gdn_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w,
                                               void* o, long rows, long xrow, long zrow,
                                               long orow, int width, float eps, int act,
                                               void* stream)
{
  if (width != GN_D) return -1;
  dim3 grid((unsigned)((rows + GN_NW - 1) / GN_NW)), blk(GN_NW * 32);
  if (act)
    hipLaunchKernelGGL((r4d_gdn_gated_rmsnorm_kernel<1>), grid, blk, 0, (hipStream_t)stream,
                       (const unsigned short*)x, (const unsigned short*)z, (const float*)w,
                       (unsigned short*)o, rows, xrow, zrow, orow, eps);
  else
    hipLaunchKernelGGL((r4d_gdn_gated_rmsnorm_kernel<0>), grid, blk, 0, (hipStream_t)stream,
                       (const unsigned short*)x, (const unsigned short*)z, (const float*)w,
                       (unsigned short*)o, rows, xrow, zrow, orow, eps);
  return (int)hipGetLastError();
}
