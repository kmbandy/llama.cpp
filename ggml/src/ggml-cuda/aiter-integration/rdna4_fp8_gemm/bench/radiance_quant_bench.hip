// bench/radiance_quant_bench.hip — radiance_quant.h's ml8-4 qrot kernels:
// bit-exactness vs the verbatim oracle (ml8_fp8_qrot_v3_kernel, unmodified)
// and timing/GB/s vs the "current sequence" (original-qrot row-major +
// separate elementwise op + a separate retile pass).
//
// Build:
//   hipcc --offload-arch=gfx1201 -O3 -std=c++17 -I/opt/rocm/include -I".." \
//     ../gemm_blockscale.hip ../gemm_trfeed_prod.hip ../gemm_ml84_prod.hip \
//     ../gemm_ml84_radiance.hip ../radiance_quant.hip radiance_quant_bench.hip \
//     -o ../out/radiance_quant_bench
//
// Env:
//   RADQ_ONLY=M,K   run just one (M,K) shape instead of the full sweep
//   RADQ_ITERS=N    timing iterations per kernel (default 20)
//
// This file used to also bench radiance_add_rms_quant / silu_mul_quant /
// gdn_norm_quant (direct ports of radiance's OWN bf16 rmsnorm/silu/gdn
// epilogues). Those are gone from radiance_quant.h (see its header comment:
// they are wrong for ml8-4's rotated activation path, MXFP4 has no rotation
// step, and nothing calls them) -- deleted here too, so this file only
// carries what the fused rot_kind/a_dim/b_dim entry points actually need.
//
// For each of K in {5120 (a=5,b=1024 Kronecker), 6144 (a=6,b=1024
// Kronecker), 17408 (a=136,b=128 block-Hadamard)} at M=4096, this checks:
//   PLAIN     rdna4_ml8_qrot_tiled            vs oracle_rowmajor(x)+retile
//   ADD       rdna4_ml8_qrot_add_tiled        vs oracle_rowmajor(x+res)+retile
//   SILU_MUL  rdna4_ml8_qrot_silu_mul_split_tiled
//                                             vs oracle_rowmajor(silu(g)*u)+retile
// PLAIN/ADD must be BIT-IDENTICAL to the oracle path (same math, only the
// store point and the Kronecker mix's register-vs-LDS data path differ --
// see radiance_quant.hip's header comment). SILU_MUL's own load fusion has
// no oracle equivalent to diff bit-for-bit against (the oracle never reads
// gate/up), so it is checked to fp32-rounding-order tolerance instead,
// against the same silu(gate)*up -> oracle-rotate -> retile pipeline run on
// the CPU-visible naive silu*mul kernel's output.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "../radiance_quant.h"
#include "../gemm_ml84_radiance.h"
#include "ml84_trfeed_layout.h"   // ml84_decode_e4m3 (OCP e4m3)
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <utility>

#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

// ─────────────────────────────────────────────────────────────────────────
// "Current sequence" naive reference kernels -- new, minimal, deliberately
// unfused: these are the per-op baselines the fused kernels are supposed to
// beat (residual add, split-gate/up silu*mul), timed standalone AND as
// input to the oracle+retile pipeline for the numerics check.
__global__ void naive_add_kernel(const float* __restrict__ x, const float* __restrict__ res,
                                  int K, float* __restrict__ out) {
  const int row = blockIdx.x;
  const float* xr = x + (size_t)row * K;
  const float* rr = res + (size_t)row * K;
  float* orow = out + (size_t)row * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) orow[k] = xr[k] + rr[k];
}

__global__ void naive_silu_mul_split_kernel(const float* __restrict__ gate,
                                             const float* __restrict__ up, int N,
                                             float* __restrict__ out) {
  const int row = blockIdx.x;
  const float* g = gate + (size_t)row * N;
  const float* u = up + (size_t)row * N;
  float* orow = out + (size_t)row * N;
  for (int k = threadIdx.x; k < N; k += blockDim.x) {
    const float gv = g[k];
    orow[k] = (gv / (1.f + expf(-gv))) * u[k];
  }
}

// GATED_NORM "current sequence" baseline: 4 separate, unfused launches
// mirroring the actual ggml graph (RMS_NORM -> MUL(norm_w) -> SILU(z) ->
// MUL(gate)) instead of one fused kernel. ggml's own rms_norm_f32
// arithmetic: sum of squares in fp32 over ne00(=head_dim), mean = sum/ne00,
// scale = 1/sqrtf(mean+eps).
__global__ void naive_rmsnorm_perhead_kernel(const float* __restrict__ o, int head_dim,
                                              int n_heads, float eps, float* __restrict__ y) {
  const int row = blockIdx.x;
  const int K = head_dim * n_heads;
  const float* orow = o + (size_t)row * K;
  float* yrow = y + (size_t)row * K;
  for (int h = threadIdx.x; h < n_heads; h += blockDim.x) {
    float ss = 0.f;
    for (int d = 0; d < head_dim; d++) { const float v = orow[h * head_dim + d]; ss += v * v; }
    const float rms = rsqrtf(ss / (float)head_dim + eps);
    for (int d = 0; d < head_dim; d++) yrow[h * head_dim + d] = orow[h * head_dim + d] * rms;
  }
}

__global__ void naive_mul_normw_kernel(float* __restrict__ y, const float* __restrict__ norm_w,
                                        int K, int head_dim) {
  const int row = blockIdx.x;
  float* yrow = y + (size_t)row * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) yrow[k] *= norm_w[k % head_dim];
}

__global__ void naive_silu_strided_kernel(const char* __restrict__ z_bytes, size_t z_nb1,
                                           size_t z_nb2, int head_dim, int n_heads,
                                           float* __restrict__ silu_out) {
  const int row = blockIdx.x;
  const int K = head_dim * n_heads;
  const char* zrow = z_bytes + (size_t)row * z_nb2;
  float* orow = silu_out + (size_t)row * K;
  for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
    const int h = idx / head_dim, d = idx % head_dim;
    const float zv = *reinterpret_cast<const float*>(zrow + (size_t)h * z_nb1 + (size_t)d * sizeof(float));
    orow[idx] = zv / (1.f + expf(-zv));
  }
}

__global__ void naive_mul_gate_kernel(float* __restrict__ y, const float* __restrict__ silu_out, int K) {
  const int row = blockIdx.x;
  float* yrow = y + (size_t)row * K;
  const float* srow = silu_out + (size_t)row * K;
  for (int k = threadIdx.x; k < K; k += blockDim.x) yrow[k] *= srow[k];
}

// CPU read of one element out of the TILED A layout (inverse of
// radiance_quant.hip's radp_tiled_off).
static uint8_t tiled_read(const std::vector<uint8_t>& buf, int row, int k, int K) {
  const int g = k >> 3, kt = g >> 1, half = g & 1;
  const int mt = row >> 4, lrow = row & 15, KT = K >> 4;
  const size_t off = (size_t)(mt * KT + kt) * 256 + (size_t)lrow * 8 + (size_t)half * 128 + (size_t)(k & 7);
  return buf[off];
}

static double now_ms(hipEvent_t s, hipEvent_t e) { float ms = 0; (void)hipEventElapsedTime(&ms, s, e); return (double)ms; }

template <typename F>
static double time_us(hipStream_t stream, int iters, F&& fn) {
  hipEvent_t s, e;
  (void)hipEventCreate(&s); (void)hipEventCreate(&e);
  fn(); HCK(hipStreamSynchronize(stream));   // warmup
  (void)hipEventRecord(s, stream);
  for (int i = 0; i < iters; i++) fn();
  (void)hipEventRecord(e, stream);
  (void)hipEventSynchronize(e);
  return now_ms(s, e) * 1000.0 / iters;
}

static double gbps(double bytes, double us) { return us > 0 ? (bytes / 1e9) / (us / 1e6) : 0.0; }

// Approximate local e4m3 ulp near byte `b`'s decoded value: flip the
// mantissa LSB to the nearest OTHER representable code and measure the
// decoded step. Diagnostic only (per the coordinator's ask: distinguish
// "1-ulp rounding, acceptable" from "something structural" on any residual
// byte mismatches) -- not a claim of the format's true ulp definition.
static double e4m3_ulp_near(uint8_t b) { return std::fabs((double)ml84_decode_e4m3(b) - (double)ml84_decode_e4m3((uint8_t)(b ^ 0x1))); }

// For every byte where A_ours/A_oracle disagree, decode both (with their
// row's own a_scale) and report the worst absolute decoded-value diff plus
// how many of those mismatches exceed ~1 local ulp (in decoded/post-scale
// units) -- i.e. whether the residual bit differences from the bit-exact
// check are genuine 1-ulp rounding noise or something structural.
static void report_byte_diff_detail(const char* label, const std::vector<uint8_t>& A_ours,
                                     const std::vector<uint8_t>& A_oracle,
                                     const std::vector<float>& as_ours,
                                     const std::vector<float>& as_oracle, int M, int K) {
  double max_decoded_diff = 0.0;
  size_t n_over_1ulp = 0, n_checked = 0;
  const int KT = K >> 4;
  for (int row = 0; row < M; row++) {
    const int mt = row >> 4, lrow = row & 15;
    for (int k = 0; k < K; k++) {
      const int g = k >> 3, kt = g >> 1, half = g & 1;
      const size_t off = (size_t)(mt * KT + kt) * 256 + (size_t)lrow * 8 + (size_t)half * 128 + (size_t)(k & 7);
      const uint8_t bo = A_ours[off], br = A_oracle[off];
      if (bo == br) continue;
      n_checked++;
      const double vo = (double)ml84_decode_e4m3(bo) * (double)as_ours[row];
      const double vr = (double)ml84_decode_e4m3(br) * (double)as_oracle[row];
      const double diff = std::fabs(vo - vr);
      max_decoded_diff = std::max(max_decoded_diff, diff);
      const double ulp = e4m3_ulp_near(br) * (double)as_oracle[row];
      if (diff > 1.0001 * std::max(ulp, 1e-30)) n_over_1ulp++;
    }
  }
  if (n_checked == 0) return;
  printf("    %s mismatch detail: %zu bytes differ, max decoded-value diff %.3e, %zu (%.1f%%) exceed ~1 ulp %s\n",
         label, n_checked, max_decoded_diff, n_over_1ulp, 100.0 * n_over_1ulp / n_checked,
         n_over_1ulp == 0 ? "(all rounding-order noise)" : "(SUSPECT: structural)");
}

// ─────────────────────────────────────────────────────────────────────────
static int run_qrot_shape(bool kronecker, int a_dim, int b_dim, int M, int iters) {
  const int K = a_dim * b_dim;
  const int rot_kind = kronecker ? RDNA4_ML8_QROT_KIND_KRONECKER : RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD;
  printf("==== ml8_qrot %s  M=%d a_dim=%d b_dim=%d K=%d ====\n",
         kronecker ? "KRONECKER" : "BLOCK_HADAMARD", M, a_dim, b_dim, K);
  std::mt19937 rng(0x51C0u ^ (uint32_t)(M * 131071 + K));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<float> x((size_t)M * K), res((size_t)M * K), gate((size_t)M * K), up((size_t)M * K),
      norm_w(K), h_a((size_t)a_dim * a_dim);
  for (auto& v : x) v = d(rng);
  for (auto& v : res) v = d(rng);
  for (auto& v : gate) v = d(rng);
  for (auto& v : up) v = d(rng);
  for (auto& v : norm_w) v = 0.5f + 0.5f * d(rng);
  for (auto& v : h_a) v = d(rng);   // random is fine: both sides read the SAME h_a

  float *dx, *dres, *dgate, *dup, *dnormw, *dha, *dsum, *dsilu, *dresidual_out;
  float *da_scale, *da_scale_oracle;
  uint8_t *dA_tiled, *dA_oracle_rm, *dA_oracle_tiled;
  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  HCK(hipMalloc(&dx, x.size() * 4)); HCK(hipMalloc(&dres, res.size() * 4));
  HCK(hipMalloc(&dgate, gate.size() * 4)); HCK(hipMalloc(&dup, up.size() * 4));
  HCK(hipMalloc(&dnormw, norm_w.size() * 4)); HCK(hipMalloc(&dha, h_a.size() * 4));
  HCK(hipMalloc(&dsum, (size_t)M * K * 4)); HCK(hipMalloc(&dsilu, (size_t)M * K * 4));
  HCK(hipMalloc(&dresidual_out, (size_t)M * K * 4));
  HCK(hipMalloc(&da_scale, (size_t)M * 4)); HCK(hipMalloc(&da_scale_oracle, (size_t)M * 4));
  HCK(hipMalloc(&dA_tiled, at_bytes)); HCK(hipMalloc(&dA_oracle_rm, (size_t)M * K));
  HCK(hipMalloc(&dA_oracle_tiled, at_bytes));
  HCK(hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dres, res.data(), res.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dgate, gate.data(), gate.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dup, up.data(), up.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dnormw, norm_w.data(), norm_w.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dha, h_a.data(), h_a.size() * 4, hipMemcpyHostToDevice));

  hipStream_t stream = 0;
  const float* h_a_arg = kronecker ? dha : nullptr;
  const double bytes_in_out_5 = (double)M * K * (4 /*read*/ + 1 /*fp8 write*/);

  // ---- PLAIN: rdna4_ml8_qrot_tiled vs oracle_rowmajor(x)+retile ----
  {
    bool ok = rdna4_ml8_qrot_tiled(dx, h_a_arg, rot_kind, a_dim, b_dim, dnormw, 1e-6f, M, K,
                                    dA_tiled, da_scale, stream);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!ok) {
      printf("  PLAIN: declined this shape (outside the dispatch subset)\n");
    } else {
      bool oracle_ok = rdna4_ml8_qrot_oracle_rowmajor(dx, h_a_arg, rot_kind, a_dim, b_dim, dnormw,
                                                       1e-6f, M, K, dA_oracle_rm, da_scale_oracle, stream);
      bool retile_ok = oracle_ok && rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
      HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
      if (!oracle_ok || !retile_ok) {
        printf("  PLAIN: oracle/retile declined (oracle=%d retile=%d)\n", oracle_ok, retile_ok);
      } else {
        std::vector<uint8_t> A_ours(at_bytes), A_oracle(at_bytes);
        HCK(hipMemcpy(A_ours.data(), dA_tiled, at_bytes, hipMemcpyDeviceToHost));
        HCK(hipMemcpy(A_oracle.data(), dA_oracle_tiled, at_bytes, hipMemcpyDeviceToHost));
        size_t nmiss = 0;
        for (size_t i = 0; i < at_bytes; i++) if (A_ours[i] != A_oracle[i]) nmiss++;
        printf("  PLAIN bit-exact: %zu/%zu bytes differ %s\n", nmiss, at_bytes,
               nmiss == 0 ? "(PASS)" : "(MISMATCH)");
        if (nmiss != 0) {
          std::vector<float> as_ours(M), as_oracle(M);
          HCK(hipMemcpy(as_ours.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
          HCK(hipMemcpy(as_oracle.data(), da_scale_oracle, M * 4, hipMemcpyDeviceToHost));
          report_byte_diff_detail("PLAIN", A_ours, A_oracle, as_ours, as_oracle, M, K);
        }

        const double us_ours = time_us(stream, iters, [&] {
          rdna4_ml8_qrot_tiled(dx, h_a_arg, rot_kind, a_dim, b_dim, dnormw, 1e-6f, M, K, dA_tiled, da_scale, stream);
        });
        const double us_oracle = time_us(stream, iters, [&] {
          rdna4_ml8_qrot_oracle_rowmajor(dx, h_a_arg, rot_kind, a_dim, b_dim, dnormw, 1e-6f, M, K, dA_oracle_rm, da_scale_oracle, stream);
        });
        const double us_retile = time_us(stream, iters, [&] {
          rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
        });
        printf("  PLAIN timing: fused-tiled %.1f us (%.1f GB/s) | original-qrot %.1f us + retile %.1f us = %.1f us | speedup %.2fx\n",
               us_ours, gbps(bytes_in_out_5, us_ours), us_oracle, us_retile, us_oracle + us_retile,
               (us_oracle + us_retile) / us_ours);
      }
    }
  }

  // ---- ADD: rdna4_ml8_qrot_add_tiled vs oracle_rowmajor(x+res)+retile ----
  {
    bool ok = rdna4_ml8_qrot_add_tiled(dx, dres, dresidual_out, h_a_arg, rot_kind, a_dim, b_dim,
                                        dnormw, 1e-6f, M, K, dA_tiled, da_scale, stream);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!ok) {
      printf("  ADD: declined this shape\n");
    } else {
      naive_add_kernel<<<M, 256, 0, stream>>>(dx, dres, K, dsum);
      bool oracle_ok = rdna4_ml8_qrot_oracle_rowmajor(dsum, h_a_arg, rot_kind, a_dim, b_dim, dnormw,
                                                       1e-6f, M, K, dA_oracle_rm, da_scale_oracle, stream);
      bool retile_ok = oracle_ok && rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
      HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
      if (!oracle_ok || !retile_ok) {
        printf("  ADD: oracle/retile declined (oracle=%d retile=%d)\n", oracle_ok, retile_ok);
      } else {
        std::vector<uint8_t> A_ours(at_bytes), A_oracle(at_bytes);
        HCK(hipMemcpy(A_ours.data(), dA_tiled, at_bytes, hipMemcpyDeviceToHost));
        HCK(hipMemcpy(A_oracle.data(), dA_oracle_tiled, at_bytes, hipMemcpyDeviceToHost));
        size_t nmiss = 0;
        for (size_t i = 0; i < at_bytes; i++) if (A_ours[i] != A_oracle[i]) nmiss++;
        std::vector<float> res_out((size_t)M * K), ref_sum((size_t)M * K);
        HCK(hipMemcpy(res_out.data(), dresidual_out, res_out.size() * 4, hipMemcpyDeviceToHost));
        HCK(hipMemcpy(ref_sum.data(), dsum, ref_sum.size() * 4, hipMemcpyDeviceToHost));
        size_t nres_miss = 0;
        for (size_t i = 0; i < res_out.size(); i++) if (res_out[i] != ref_sum[i]) nres_miss++;
        printf("  ADD bit-exact: %zu/%zu A bytes differ, %zu/%zu residual_out differ %s\n",
               nmiss, at_bytes, nres_miss, res_out.size(),
               (nmiss == 0 && nres_miss == 0) ? "(PASS)" : "(MISMATCH)");
        if (nmiss != 0) {
          std::vector<float> as_ours(M), as_oracle(M);
          HCK(hipMemcpy(as_ours.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
          HCK(hipMemcpy(as_oracle.data(), da_scale_oracle, M * 4, hipMemcpyDeviceToHost));
          report_byte_diff_detail("ADD", A_ours, A_oracle, as_ours, as_oracle, M, K);
        }

        // RADQ_ROTATE=N: cycle through N distinct copies of x/res so the working set exceeds the
        // 64 MB infinity cache -- measures HBM-streaming time like production, not cache-resident.
        const int rot_n = std::getenv("RADQ_ROTATE") ? std::max(1, std::atoi(std::getenv("RADQ_ROTATE"))) : 1;
        float *dx_rot = dx, *dres_rot = dres;
        const size_t mk = (size_t)M * K;
        if (rot_n > 1) {
          HCK(hipMalloc(&dx_rot, rot_n * mk * 4)); HCK(hipMalloc(&dres_rot, rot_n * mk * 4));
          for (int r = 0; r < rot_n; r++) {
            HCK(hipMemcpy(dx_rot + r * mk, dx, mk * 4, hipMemcpyDeviceToDevice));
            HCK(hipMemcpy(dres_rot + r * mk, dres, mk * 4, hipMemcpyDeviceToDevice));
          }
        }
        int rot_i = 0;
        const double us_ours = time_us(stream, iters, [&] {
          const int r = (rot_i++) % rot_n;
          rdna4_ml8_qrot_add_tiled(dx_rot + r * mk, dres_rot + r * mk, dresidual_out, h_a_arg, rot_kind, a_dim, b_dim, dnormw, 1e-6f, M, K, dA_tiled, da_scale, stream);
        });
        const double us_add = time_us(stream, iters, [&] { naive_add_kernel<<<M, 256, 0, stream>>>(dx, dres, K, dsum); });
        const double us_oracle = time_us(stream, iters, [&] {
          rdna4_ml8_qrot_oracle_rowmajor(dsum, h_a_arg, rot_kind, a_dim, b_dim, dnormw, 1e-6f, M, K, dA_oracle_rm, da_scale_oracle, stream);
        });
        const double us_retile = time_us(stream, iters, [&] {
          rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
        });
        const double bytes_add = (double)M * K * (4 + 4 + 1 + 4);  // read x,res; write fp8; write residual_out
        printf("  ADD timing: fused-tiled %.1f us (%.1f GB/s) | separate-add %.1f us + original-qrot %.1f us + retile %.1f us = %.1f us | speedup %.2fx\n",
               us_ours, gbps(bytes_add, us_ours), us_add, us_oracle, us_retile,
               us_add + us_oracle + us_retile, (us_add + us_oracle + us_retile) / us_ours);
      }
    }
  }

  // ---- SILU_MUL (split gate/up): rdna4_ml8_qrot_silu_mul_split_tiled vs
  // naive-silu + oracle_rowmajor(silu*up)+retile (fp32-order tolerance --
  // the oracle has no gate/up load of its own to diff bit-for-bit against).
  {
    bool ok = rdna4_ml8_qrot_silu_mul_split_tiled(dgate, dup, h_a_arg, rot_kind, a_dim, b_dim, M, K,
                                                   dA_tiled, da_scale, stream);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!ok) {
      printf("  SILU_MUL: declined this shape\n");
    } else {
      naive_silu_mul_split_kernel<<<M, 256, 0, stream>>>(dgate, dup, K, dsilu);
      bool oracle_ok = rdna4_ml8_qrot_oracle_rowmajor(dsilu, h_a_arg, rot_kind, a_dim, b_dim, nullptr,
                                                       0.0f, M, K, dA_oracle_rm, da_scale_oracle, stream);
      bool retile_ok = oracle_ok && rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
      HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
      if (!oracle_ok || !retile_ok) {
        printf("  SILU_MUL: oracle/retile declined (oracle=%d retile=%d)\n", oracle_ok, retile_ok);
      } else {
        std::vector<uint8_t> A_ours(at_bytes), A_oracle(at_bytes);
        HCK(hipMemcpy(A_ours.data(), dA_tiled, at_bytes, hipMemcpyDeviceToHost));
        HCK(hipMemcpy(A_oracle.data(), dA_oracle_tiled, at_bytes, hipMemcpyDeviceToHost));
        std::vector<float> as_ours(M), as_oracle(M);
        HCK(hipMemcpy(as_ours.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
        HCK(hipMemcpy(as_oracle.data(), da_scale_oracle, M * 4, hipMemcpyDeviceToHost));
        size_t nbyte_mismatch = 0;
        double max_val_err = 0.0;
        const int SM = std::min(M, 32);
        for (int si = 0; si < SM; si++) {
          const int row = (int)((size_t)si * M / SM);
          for (int k = 0; k < K; k++) {
            const uint8_t bo = tiled_read(A_ours, row, k, K), br = tiled_read(A_oracle, row, k, K);
            if (bo != br) nbyte_mismatch++;
            const double vo = (double)ml84_decode_e4m3(bo) * (double)as_ours[row];
            const double vr = (double)ml84_decode_e4m3(br) * (double)as_oracle[row];
            max_val_err = std::max(max_val_err, std::fabs(vo - vr));
          }
        }
        printf("  SILU_MUL check (sampled %d/%d rows): %zu byte diffs, max decoded-value diff %.3e %s\n",
               SM, M, nbyte_mismatch, max_val_err,
               max_val_err < 1.0 ? "(PASS, fp32-order tolerance)" : "(SUSPECT)");

        const double us_ours = time_us(stream, iters, [&] {
          rdna4_ml8_qrot_silu_mul_split_tiled(dgate, dup, h_a_arg, rot_kind, a_dim, b_dim, M, K, dA_tiled, da_scale, stream);
        });
        const double us_silu = time_us(stream, iters, [&] { naive_silu_mul_split_kernel<<<M, 256, 0, stream>>>(dgate, dup, K, dsilu); });
        const double us_oracle = time_us(stream, iters, [&] {
          rdna4_ml8_qrot_oracle_rowmajor(dsilu, h_a_arg, rot_kind, a_dim, b_dim, nullptr, 0.0f, M, K, dA_oracle_rm, da_scale_oracle, stream);
        });
        const double us_retile = time_us(stream, iters, [&] {
          rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
        });
        const double bytes_silu = (double)M * K * (4 + 4 + 1);  // read gate,up; write fp8
        printf("  SILU_MUL timing: fused-tiled %.1f us (%.1f GB/s) | separate-silu*mul %.1f us + original-qrot %.1f us + retile %.1f us = %.1f us | speedup %.2fx\n",
               us_ours, gbps(bytes_silu, us_ours), us_silu, us_oracle, us_retile,
               us_silu + us_oracle + us_retile, (us_silu + us_oracle + us_retile) / us_ours);
      }
    }
  }

  (void)hipFree(dx); (void)hipFree(dres); (void)hipFree(dgate); (void)hipFree(dup);
  (void)hipFree(dnormw); (void)hipFree(dha); (void)hipFree(dsum); (void)hipFree(dsilu);
  (void)hipFree(dresidual_out); (void)hipFree(da_scale); (void)hipFree(da_scale_oracle);
  (void)hipFree(dA_tiled); (void)hipFree(dA_oracle_rm); (void)hipFree(dA_oracle_tiled);
  return 0;
}

// ─────────────────────────────────────────────────────────────────────────
// bf16-input host helpers: truncate an fp32 to bf16 bits (exact -- this is
// building test data, not converting a value that must round-trip
// correctly) and upcast bf16 bits back to fp32 (also exact, same formula
// the kernels use). The "reference" for every bf16 check below is the
// EXISTING fp32 path fed the upcast value -- i.e. the bf16 truncation's
// own precision loss is baked into what both sides compare against, so
// these checks validate the kernel's bf16 LOAD, not bf16 truncation error.
static uint16_t f32_to_bf16_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return (uint16_t)(bits >> 16);
}
static float bf16_bits_to_f32(uint16_t h) {
  uint32_t bits = (uint32_t)h << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// SILU_MUL_SPLIT_BF16: rdna4_ml8_qrot_silu_mul_split_tiled_bf16 vs
// rdna4_ml8_qrot_silu_mul_split_tiled (fp32) fed the bf16-upcast gate/up --
// compared the same way as SILU_MUL (sampled decoded-value diff, fp32-order
// tolerance). GB/s counts the bf16 byte budget (2 bytes/element, not 4) --
// that halving relative to SILU_MUL's own GB/s IS the point of this kernel.
static int run_silu_mul_split_bf16_shape(bool kronecker, int a_dim, int b_dim, int M, int iters) {
  const int K = a_dim * b_dim;
  const int rot_kind = kronecker ? RDNA4_ML8_QROT_KIND_KRONECKER : RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD;
  printf("==== ml8_qrot SILU_MUL_SPLIT_BF16  M=%d a_dim=%d b_dim=%d K=%d ====\n", M, a_dim, b_dim, K);
  std::mt19937 rng(0xB416u ^ (uint32_t)(M * 131071 + K));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<uint16_t> gate_bf(( size_t)M * K), up_bf((size_t)M * K);
  std::vector<float> gate_f32((size_t)M * K), up_f32((size_t)M * K), h_a((size_t)a_dim * a_dim);
  for (size_t i = 0; i < gate_bf.size(); i++) {
    gate_bf[i] = f32_to_bf16_bits(d(rng));
    gate_f32[i] = bf16_bits_to_f32(gate_bf[i]);
  }
  for (size_t i = 0; i < up_bf.size(); i++) {
    up_bf[i] = f32_to_bf16_bits(d(rng));
    up_f32[i] = bf16_bits_to_f32(up_bf[i]);
  }
  for (auto& v : h_a) v = d(rng);

  uint16_t *dgate_bf, *dup_bf;
  float *dgate_f32, *dup_f32, *dha, *da_scale, *da_scale_ref;
  uint8_t *dA_bf16, *dA_ref;
  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  HCK(hipMalloc(&dgate_bf, gate_bf.size() * 2)); HCK(hipMalloc(&dup_bf, up_bf.size() * 2));
  HCK(hipMalloc(&dgate_f32, gate_f32.size() * 4)); HCK(hipMalloc(&dup_f32, up_f32.size() * 4));
  HCK(hipMalloc(&dha, h_a.size() * 4)); HCK(hipMalloc(&da_scale, (size_t)M * 4));
  HCK(hipMalloc(&da_scale_ref, (size_t)M * 4));
  HCK(hipMalloc(&dA_bf16, at_bytes)); HCK(hipMalloc(&dA_ref, at_bytes));
  HCK(hipMemcpy(dgate_bf, gate_bf.data(), gate_bf.size() * 2, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dup_bf, up_bf.data(), up_bf.size() * 2, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dgate_f32, gate_f32.data(), gate_f32.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dup_f32, up_f32.data(), up_f32.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dha, h_a.data(), h_a.size() * 4, hipMemcpyHostToDevice));

  hipStream_t stream = 0;
  const float* h_a_arg = kronecker ? dha : nullptr;

  bool ok = rdna4_ml8_qrot_silu_mul_split_tiled_bf16(dgate_bf, dup_bf, h_a_arg, rot_kind, a_dim,
                                                      b_dim, M, K, dA_bf16, da_scale, stream);
  HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
  if (!ok) {
    printf("  SILU_MUL_SPLIT_BF16: declined this shape\n");
  } else {
    bool ref_ok = rdna4_ml8_qrot_silu_mul_split_tiled(dgate_f32, dup_f32, h_a_arg, rot_kind, a_dim,
                                                       b_dim, M, K, dA_ref, da_scale_ref, stream);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!ref_ok) {
      printf("  SILU_MUL_SPLIT_BF16: fp32 reference declined (unexpected -- same shape just accepted)\n");
    } else {
      std::vector<uint8_t> A_bf16(at_bytes), A_ref(at_bytes);
      HCK(hipMemcpy(A_bf16.data(), dA_bf16, at_bytes, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(A_ref.data(), dA_ref, at_bytes, hipMemcpyDeviceToHost));
      std::vector<float> as_bf16(M), as_ref(M);
      HCK(hipMemcpy(as_bf16.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(as_ref.data(), da_scale_ref, M * 4, hipMemcpyDeviceToHost));
      size_t nbyte_mismatch = 0;
      double max_val_err = 0.0;
      const int SM = std::min(M, 32);
      for (int si = 0; si < SM; si++) {
        const int row = (int)((size_t)si * M / SM);
        for (int k = 0; k < K; k++) {
          const uint8_t bo = tiled_read(A_bf16, row, k, K), br = tiled_read(A_ref, row, k, K);
          if (bo != br) nbyte_mismatch++;
          const double vo = (double)ml84_decode_e4m3(bo) * (double)as_bf16[row];
          const double vr = (double)ml84_decode_e4m3(br) * (double)as_ref[row];
          max_val_err = std::max(max_val_err, std::fabs(vo - vr));
        }
      }
      printf("  SILU_MUL_SPLIT_BF16 check (sampled %d/%d rows): %zu byte diffs, max decoded-value diff %.3e %s\n",
             SM, M, nbyte_mismatch, max_val_err,
             max_val_err < 1.0 ? "(PASS, fp32-order tolerance)" : "(SUSPECT)");

      const double us_bf16 = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_silu_mul_split_tiled_bf16(dgate_bf, dup_bf, h_a_arg, rot_kind, a_dim, b_dim, M, K, dA_bf16, da_scale, stream);
      });
      const double us_fp32 = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_silu_mul_split_tiled(dgate_f32, dup_f32, h_a_arg, rot_kind, a_dim, b_dim, M, K, dA_ref, da_scale_ref, stream);
      });
      const double bytes_bf16 = (double)M * K * (2 + 2 + 1);  // read gate,up (bf16); write fp8
      const double bytes_fp32 = (double)M * K * (4 + 4 + 1);
      printf("  SILU_MUL_SPLIT_BF16 timing: bf16-input %.1f us (%.1f GB/s) | fp32-input %.1f us (%.1f GB/s) | speedup %.2fx\n",
             us_bf16, gbps(bytes_bf16, us_bf16), us_fp32, gbps(bytes_fp32, us_fp32), us_fp32 / us_bf16);
    }
  }

  (void)hipFree(dgate_bf); (void)hipFree(dup_bf); (void)hipFree(dgate_f32); (void)hipFree(dup_f32);
  (void)hipFree(dha); (void)hipFree(da_scale); (void)hipFree(da_scale_ref);
  (void)hipFree(dA_bf16); (void)hipFree(dA_ref);
  return 0;
}

// ADD_BF16: rdna4_ml8_qrot_add_tiled_bf16 vs rdna4_ml8_qrot_add_tiled (fp32)
// fed the bf16-upcast x/residual -- same comparison style as SILU_MUL_SPLIT_BF16.
static int run_add_bf16_shape(bool kronecker, int a_dim, int b_dim, int M, int iters) {
  const int K = a_dim * b_dim;
  const int rot_kind = kronecker ? RDNA4_ML8_QROT_KIND_KRONECKER : RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD;
  printf("==== ml8_qrot ADD_BF16  M=%d a_dim=%d b_dim=%d K=%d ====\n", M, a_dim, b_dim, K);
  std::mt19937 rng(0xADD6u ^ (uint32_t)(M * 131071 + K));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  std::vector<uint16_t> x_bf((size_t)M * K), res_bf((size_t)M * K);
  std::vector<float> x_f32((size_t)M * K), res_f32((size_t)M * K), norm_w(K), h_a((size_t)a_dim * a_dim);
  for (size_t i = 0; i < x_bf.size(); i++) { x_bf[i] = f32_to_bf16_bits(d(rng)); x_f32[i] = bf16_bits_to_f32(x_bf[i]); }
  for (size_t i = 0; i < res_bf.size(); i++) { res_bf[i] = f32_to_bf16_bits(d(rng)); res_f32[i] = bf16_bits_to_f32(res_bf[i]); }
  for (auto& v : norm_w) v = 0.5f + 0.5f * d(rng);
  for (auto& v : h_a) v = d(rng);
  const float eps = 1e-6f;

  uint16_t *dx_bf, *dres_bf;
  float *dx_f32, *dres_f32, *dnormw, *dha, *da_scale, *da_scale_ref, *dresidual_out;
  uint8_t *dA_bf16, *dA_ref;
  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  HCK(hipMalloc(&dx_bf, x_bf.size() * 2)); HCK(hipMalloc(&dres_bf, res_bf.size() * 2));
  HCK(hipMalloc(&dx_f32, x_f32.size() * 4)); HCK(hipMalloc(&dres_f32, res_f32.size() * 4));
  HCK(hipMalloc(&dnormw, norm_w.size() * 4)); HCK(hipMalloc(&dha, h_a.size() * 4));
  HCK(hipMalloc(&da_scale, (size_t)M * 4)); HCK(hipMalloc(&da_scale_ref, (size_t)M * 4));
  HCK(hipMalloc(&dresidual_out, (size_t)M * K * 4));
  HCK(hipMalloc(&dA_bf16, at_bytes)); HCK(hipMalloc(&dA_ref, at_bytes));
  HCK(hipMemcpy(dx_bf, x_bf.data(), x_bf.size() * 2, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dres_bf, res_bf.data(), res_bf.size() * 2, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dx_f32, x_f32.data(), x_f32.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dres_f32, res_f32.data(), res_f32.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dnormw, norm_w.data(), norm_w.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dha, h_a.data(), h_a.size() * 4, hipMemcpyHostToDevice));

  hipStream_t stream = 0;
  const float* h_a_arg = kronecker ? dha : nullptr;

  bool ok = rdna4_ml8_qrot_add_tiled_bf16(dx_bf, dres_bf, dresidual_out, h_a_arg, rot_kind, a_dim,
                                          b_dim, dnormw, eps, M, K, dA_bf16, da_scale, stream);
  HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
  if (!ok) {
    printf("  ADD_BF16: declined this shape\n");
  } else {
    bool ref_ok = rdna4_ml8_qrot_add_tiled(dx_f32, dres_f32, nullptr, h_a_arg, rot_kind, a_dim, b_dim,
                                            dnormw, eps, M, K, dA_ref, da_scale_ref, stream);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!ref_ok) {
      printf("  ADD_BF16: fp32 reference declined (unexpected -- same shape just accepted)\n");
    } else {
      std::vector<uint8_t> A_bf16(at_bytes), A_ref(at_bytes);
      HCK(hipMemcpy(A_bf16.data(), dA_bf16, at_bytes, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(A_ref.data(), dA_ref, at_bytes, hipMemcpyDeviceToHost));
      std::vector<float> as_bf16(M), as_ref(M);
      HCK(hipMemcpy(as_bf16.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(as_ref.data(), da_scale_ref, M * 4, hipMemcpyDeviceToHost));
      size_t nbyte_mismatch = 0;
      double max_val_err = 0.0;
      const int SM = std::min(M, 32);
      for (int si = 0; si < SM; si++) {
        const int row = (int)((size_t)si * M / SM);
        for (int k = 0; k < K; k++) {
          const uint8_t bo = tiled_read(A_bf16, row, k, K), br = tiled_read(A_ref, row, k, K);
          if (bo != br) nbyte_mismatch++;
          const double vo = (double)ml84_decode_e4m3(bo) * (double)as_bf16[row];
          const double vr = (double)ml84_decode_e4m3(br) * (double)as_ref[row];
          max_val_err = std::max(max_val_err, std::fabs(vo - vr));
        }
      }
      printf("  ADD_BF16 check (sampled %d/%d rows): %zu byte diffs, max decoded-value diff %.3e %s\n",
             SM, M, nbyte_mismatch, max_val_err,
             max_val_err < 1.0 ? "(PASS, fp32-order tolerance)" : "(SUSPECT)");

      const double us_bf16 = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_add_tiled_bf16(dx_bf, dres_bf, dresidual_out, h_a_arg, rot_kind, a_dim, b_dim, dnormw, eps, M, K, dA_bf16, da_scale, stream);
      });
      const double us_fp32 = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_add_tiled(dx_f32, dres_f32, nullptr, h_a_arg, rot_kind, a_dim, b_dim, dnormw, eps, M, K, dA_ref, da_scale_ref, stream);
      });
      const double bytes_bf16 = (double)M * K * (2 + 2 + 1 + 4);  // read x,res (bf16); write fp8; write residual_out (fp32)
      const double bytes_fp32 = (double)M * K * (4 + 4 + 1);
      printf("  ADD_BF16 timing: bf16-input %.1f us (%.1f GB/s) | fp32-input %.1f us (%.1f GB/s) | speedup %.2fx\n",
             us_bf16, gbps(bytes_bf16, us_bf16), us_fp32, gbps(bytes_fp32, us_fp32), us_fp32 / us_bf16);
    }
  }

  (void)hipFree(dx_bf); (void)hipFree(dres_bf); (void)hipFree(dx_f32); (void)hipFree(dres_f32);
  (void)hipFree(dnormw); (void)hipFree(dha); (void)hipFree(da_scale); (void)hipFree(da_scale_ref);
  (void)hipFree(dresidual_out); (void)hipFree(dA_bf16); (void)hipFree(dA_ref);
  return 0;
}

// ─────────────────────────────────────────────────────────────────────────
// GATED_NORM: rdna4_ml8_qrot_gated_norm_tiled vs the naive 4-launch
// unfused pipeline (RMS_NORM -> MUL(norm_w) -> SILU(z) -> MUL(gate)) ->
// oracle_rowmajor(norm_w=nullptr) -> retile. z is a genuinely STRIDED
// buffer (z_nb1/z_nb2 carry padding bytes beyond the packed head_dim*4 /
// n_heads*z_nb1 sizes) to prove the kernel honors the caller's strides
// rather than assuming a packed layout.
static int run_gated_norm_shape(int M, int head_dim, int n_heads, bool kronecker, int a_dim,
                                 int b_dim, size_t z_nb1, size_t z_nb2, int iters) {
  const int K = head_dim * n_heads;
  const int rot_kind = kronecker ? RDNA4_ML8_QROT_KIND_KRONECKER : RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD;
  printf("==== ml8_qrot GATED_NORM  M=%d head_dim=%d n_heads=%d K=%d a_dim=%d b_dim=%d "
         "z_nb1=%zu z_nb2=%zu ====\n", M, head_dim, n_heads, K, a_dim, b_dim, z_nb1, z_nb2);
  std::mt19937 rng(0x6A7Eu ^ (uint32_t)(M * 131071 + K));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  const float eps = 1e-6f;

  std::vector<float> o((size_t)M * K), norm_w(head_dim), h_a((size_t)a_dim * a_dim);
  for (auto& v : o) v = d(rng);
  for (auto& v : norm_w) v = 0.5f + 0.5f * d(rng);
  for (auto& v : h_a) v = d(rng);
  // z is a raw, strided byte buffer: z_nb2 bytes per row, z_nb1 bytes per
  // head within a row, head_dim contiguous floats within a head. Bytes
  // outside those head_dim*4 spans are padding this bench never reads back
  // -- their presence (nonzero z_nb1/z_nb2 beyond the packed size) is what
  // proves the kernel is using the caller's strides, not assuming packed.
  std::vector<uint8_t> z_host((size_t)M * z_nb2, 0);
  for (int row = 0; row < M; row++) {
    for (int h = 0; h < n_heads; h++) {
      float* hp = reinterpret_cast<float*>(z_host.data() + (size_t)row * z_nb2 + (size_t)h * z_nb1);
      for (int dd = 0; dd < head_dim; dd++) hp[dd] = d(rng);
    }
  }

  float *dO, *dnormw, *dha, *dy, *dsilu_out, *da_scale, *da_scale_oracle;
  uint8_t *dz, *dA_tiled, *dA_oracle_rm, *dA_oracle_tiled;
  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  HCK(hipMalloc(&dO, o.size() * 4)); HCK(hipMalloc(&dnormw, norm_w.size() * 4));
  HCK(hipMalloc(&dha, h_a.size() * 4)); HCK(hipMalloc(&dz, z_host.size()));
  HCK(hipMalloc(&dy, (size_t)M * K * 4)); HCK(hipMalloc(&dsilu_out, (size_t)M * K * 4));
  HCK(hipMalloc(&da_scale, (size_t)M * 4)); HCK(hipMalloc(&da_scale_oracle, (size_t)M * 4));
  HCK(hipMalloc(&dA_tiled, at_bytes)); HCK(hipMalloc(&dA_oracle_rm, (size_t)M * K));
  HCK(hipMalloc(&dA_oracle_tiled, at_bytes));
  HCK(hipMemcpy(dO, o.data(), o.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dnormw, norm_w.data(), norm_w.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dha, h_a.data(), h_a.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dz, z_host.data(), z_host.size(), hipMemcpyHostToDevice));

  hipStream_t stream = 0;
  const float* h_a_arg = kronecker ? dha : nullptr;
  const float* z_f = (const float*)dz;
  const char* z_bytes = (const char*)dz;

  bool ok = rdna4_ml8_qrot_gated_norm_tiled(dO, z_f, z_nb1, z_nb2, dnormw, eps, head_dim, n_heads,
                                             h_a_arg, rot_kind, a_dim, b_dim, M, dA_tiled, da_scale,
                                             stream, /*y_out=*/nullptr);
  HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
  if (!ok) {
    printf("  GATED_NORM: declined this shape (head_dim/b_dim alignment or dispatch coverage)\n");
  } else {
    naive_rmsnorm_perhead_kernel<<<M, 128, 0, stream>>>(dO, head_dim, n_heads, eps, dy);
    naive_mul_normw_kernel<<<M, 256, 0, stream>>>(dy, dnormw, K, head_dim);
    naive_silu_strided_kernel<<<M, 256, 0, stream>>>(z_bytes, z_nb1, z_nb2, head_dim, n_heads, dsilu_out);
    naive_mul_gate_kernel<<<M, 256, 0, stream>>>(dy, dsilu_out, K);
    bool oracle_ok = rdna4_ml8_qrot_oracle_rowmajor(dy, h_a_arg, rot_kind, a_dim, b_dim, nullptr,
                                                     0.0f, M, K, dA_oracle_rm, da_scale_oracle, stream);
    bool retile_ok = oracle_ok && rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!oracle_ok || !retile_ok) {
      printf("  GATED_NORM: oracle/retile declined (oracle=%d retile=%d)\n", oracle_ok, retile_ok);
    } else {
      std::vector<uint8_t> A_ours(at_bytes), A_oracle(at_bytes);
      HCK(hipMemcpy(A_ours.data(), dA_tiled, at_bytes, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(A_oracle.data(), dA_oracle_tiled, at_bytes, hipMemcpyDeviceToHost));
      std::vector<float> as_ours(M), as_oracle(M);
      HCK(hipMemcpy(as_ours.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(as_oracle.data(), da_scale_oracle, M * 4, hipMemcpyDeviceToHost));
      size_t nbyte_mismatch = 0;
      double max_val_err = 0.0;
      const int SM = std::min(M, 32);
      for (int si = 0; si < SM; si++) {
        const int row = (int)((size_t)si * M / SM);
        for (int k = 0; k < K; k++) {
          const uint8_t bo = tiled_read(A_ours, row, k, K), br = tiled_read(A_oracle, row, k, K);
          if (bo != br) nbyte_mismatch++;
          const double vo = (double)ml84_decode_e4m3(bo) * (double)as_ours[row];
          const double vr = (double)ml84_decode_e4m3(br) * (double)as_oracle[row];
          max_val_err = std::max(max_val_err, std::fabs(vo - vr));
        }
      }
      printf("  GATED_NORM check (sampled %d/%d rows): %zu byte diffs, max decoded-value diff %.3e %s\n",
             SM, M, nbyte_mismatch, max_val_err,
             max_val_err < 1.0 ? "(PASS, fp32-order tolerance)" : "(SUSPECT)");

      const double us_ours = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_gated_norm_tiled(dO, z_f, z_nb1, z_nb2, dnormw, eps, head_dim, n_heads,
                                         h_a_arg, rot_kind, a_dim, b_dim, M, dA_tiled, da_scale, stream, nullptr);
      });
      const double us_rms = time_us(stream, iters, [&] { naive_rmsnorm_perhead_kernel<<<M, 128, 0, stream>>>(dO, head_dim, n_heads, eps, dy); });
      const double us_mulw = time_us(stream, iters, [&] { naive_mul_normw_kernel<<<M, 256, 0, stream>>>(dy, dnormw, K, head_dim); });
      const double us_silu = time_us(stream, iters, [&] { naive_silu_strided_kernel<<<M, 256, 0, stream>>>(z_bytes, z_nb1, z_nb2, head_dim, n_heads, dsilu_out); });
      const double us_mulg = time_us(stream, iters, [&] { naive_mul_gate_kernel<<<M, 256, 0, stream>>>(dy, dsilu_out, K); });
      const double us_oracle = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_oracle_rowmajor(dy, h_a_arg, rot_kind, a_dim, b_dim, nullptr, 0.0f, M, K, dA_oracle_rm, da_scale_oracle, stream);
      });
      const double us_retile = time_us(stream, iters, [&] {
        rdna4_gemm_ml84_radiance_retile_a(M, K, dA_oracle_rm, dA_oracle_tiled, stream);
      });
      const double us_naive_total = us_rms + us_mulw + us_silu + us_mulg + us_oracle + us_retile;
      const double bytes_gn = (double)M * K * (4 /*o*/ + 4 /*z*/ + 1 /*fp8 write*/);
      printf("  GATED_NORM timing: fused-tiled %.1f us (%.1f GB/s) | rms_norm %.1f + mul(w) %.1f + silu(z) %.1f + "
             "mul(gate) %.1f + original-qrot %.1f + retile %.1f = %.1f us | speedup %.2fx\n",
             us_ours, gbps(bytes_gn, us_ours), us_rms, us_mulw, us_silu, us_mulg, us_oracle, us_retile,
             us_naive_total, us_naive_total / us_ours);
    }
  }

  (void)hipFree(dO); (void)hipFree(dnormw); (void)hipFree(dha); (void)hipFree(dz);
  (void)hipFree(dy); (void)hipFree(dsilu_out); (void)hipFree(da_scale); (void)hipFree(da_scale_oracle);
  (void)hipFree(dA_tiled); (void)hipFree(dA_oracle_rm); (void)hipFree(dA_oracle_tiled);
  return 0;
}

// ─────────────────────────────────────────────────────────────────────────
// GATED_NORM_R4D: rdna4_ml8_qrot_gated_norm_tiled_r4d (bf16 `o`, read
// through a head permutation + strides -- libr4d chunk_scan's own output
// layout, see mt_gdn_r4d.cu's r4d_gdn_perm_head) vs the EXISTING fp32
// logical-order kernel (rdna4_ml8_qrot_gated_norm_tiled) fed the
// bf16-upcast logical-order `o` -- same "reference = existing fp32 kernel
// on the bf16-upcast logical data" pattern as the other bf16 checks. The
// permutation here is a random one (not necessarily libr4d's actual R/Hg
// formula) -- this bench validates the kernel honors WHATEVER head_src
// mapping it's given, strides included; the actual r4d caller is
// responsible for computing head_src = r4d_gdn_perm_head(h, Hg, R).
static int run_gated_norm_r4d_shape(int M, int head_dim, int n_heads, bool kronecker, int a_dim,
                                     int b_dim, size_t z_nb1, size_t z_nb2, int iters) {
  const int K = head_dim * n_heads;
  const int rot_kind = kronecker ? RDNA4_ML8_QROT_KIND_KRONECKER : RDNA4_ML8_QROT_KIND_BLOCK_HADAMARD;
  printf("==== ml8_qrot GATED_NORM_R4D  M=%d head_dim=%d n_heads=%d K=%d a_dim=%d b_dim=%d "
         "z_nb1=%zu z_nb2=%zu ====\n", M, head_dim, n_heads, K, a_dim, b_dim, z_nb1, z_nb2);
  std::mt19937 rng(0xA4D0u ^ (uint32_t)(M * 131071 + K));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  const float eps = 1e-6f;

  std::vector<float> o_f32((size_t)M * K), norm_w(head_dim), h_a((size_t)a_dim * a_dim);
  for (auto& v : o_f32) v = d(rng);
  for (auto& v : norm_w) v = 0.5f + 0.5f * d(rng);
  for (auto& v : h_a) v = d(rng);
  std::vector<uint8_t> z_host((size_t)M * z_nb2, 0);
  for (int row = 0; row < M; row++) {
    for (int h = 0; h < n_heads; h++) {
      float* hp = reinterpret_cast<float*>(z_host.data() + (size_t)row * z_nb2 + (size_t)h * z_nb1);
      for (int dd = 0; dd < head_dim; dd++) hp[dd] = d(rng);
    }
  }

  // Random head permutation: head_src[h] = perm[h] = where logical head h's
  // data lives in o_bf16. Build the permuted bf16 buffer from o_f32, AND
  // the logical-order bf16-upcast buffer the fp32 reference kernel reads --
  // both derived from the SAME truncated bits, so this validates the
  // kernel's head_src/stride handling, not bf16 rounding.
  std::vector<int32_t> perm(n_heads);
  std::iota(perm.begin(), perm.end(), 0);
  std::shuffle(perm.begin(), perm.end(), rng);
  std::vector<uint16_t> o_bf16((size_t)M * n_heads * head_dim);
  std::vector<float> o_logical_upcast((size_t)M * K);
  for (int row = 0; row < M; row++) {
    for (int h = 0; h < n_heads; h++) {
      for (int dd = 0; dd < head_dim; dd++) {
        const uint16_t bits = f32_to_bf16_bits(o_f32[(size_t)row * K + (size_t)h * head_dim + dd]);
        o_bf16[(size_t)row * n_heads * head_dim + (size_t)perm[h] * head_dim + dd] = bits;
        o_logical_upcast[(size_t)row * K + (size_t)h * head_dim + dd] = bf16_bits_to_f32(bits);
      }
    }
  }
  const size_t o_nb_head = (size_t)head_dim * 2;
  const size_t o_nb_tok  = (size_t)n_heads * o_nb_head;

  uint16_t* dO_bf16;
  int32_t* dperm;
  float *dO_logical, *dnormw, *dha, *da_scale, *da_scale_ref;
  uint8_t *dz, *dA_r4d, *dA_ref;
  const int Mt = (M + 15) >> 4;
  const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
  HCK(hipMalloc(&dO_bf16, o_bf16.size() * 2)); HCK(hipMalloc(&dperm, perm.size() * 4));
  HCK(hipMalloc(&dO_logical, o_logical_upcast.size() * 4));
  HCK(hipMalloc(&dnormw, norm_w.size() * 4)); HCK(hipMalloc(&dha, h_a.size() * 4));
  HCK(hipMalloc(&dz, z_host.size()));
  HCK(hipMalloc(&da_scale, (size_t)M * 4)); HCK(hipMalloc(&da_scale_ref, (size_t)M * 4));
  HCK(hipMalloc(&dA_r4d, at_bytes)); HCK(hipMalloc(&dA_ref, at_bytes));
  HCK(hipMemcpy(dO_bf16, o_bf16.data(), o_bf16.size() * 2, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dperm, perm.data(), perm.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dO_logical, o_logical_upcast.data(), o_logical_upcast.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dnormw, norm_w.data(), norm_w.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dha, h_a.data(), h_a.size() * 4, hipMemcpyHostToDevice));
  HCK(hipMemcpy(dz, z_host.data(), z_host.size(), hipMemcpyHostToDevice));

  hipStream_t stream = 0;
  const float* h_a_arg = kronecker ? dha : nullptr;
  const float* z_f = (const float*)dz;

  bool ok = rdna4_ml8_qrot_gated_norm_tiled_r4d(dO_bf16, o_nb_head, o_nb_tok, dperm, z_f, z_nb1,
                                                 z_nb2, dnormw, eps, head_dim, n_heads, h_a_arg,
                                                 rot_kind, a_dim, b_dim, M, dA_r4d, da_scale, stream,
                                                 /*y_out=*/nullptr);
  HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
  if (!ok) {
    printf("  GATED_NORM_R4D: declined this shape\n");
  } else {
    bool ref_ok = rdna4_ml8_qrot_gated_norm_tiled(dO_logical, z_f, z_nb1, z_nb2, dnormw, eps,
                                                   head_dim, n_heads, h_a_arg, rot_kind, a_dim, b_dim,
                                                   M, dA_ref, da_scale_ref, stream, /*y_out=*/nullptr);
    HCK(hipGetLastError()); HCK(hipStreamSynchronize(stream));
    if (!ref_ok) {
      printf("  GATED_NORM_R4D: fp32 logical-order reference declined (unexpected -- same shape just accepted)\n");
    } else {
      std::vector<uint8_t> A_r4d(at_bytes), A_ref(at_bytes);
      HCK(hipMemcpy(A_r4d.data(), dA_r4d, at_bytes, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(A_ref.data(), dA_ref, at_bytes, hipMemcpyDeviceToHost));
      std::vector<float> as_r4d(M), as_ref(M);
      HCK(hipMemcpy(as_r4d.data(), da_scale, M * 4, hipMemcpyDeviceToHost));
      HCK(hipMemcpy(as_ref.data(), da_scale_ref, M * 4, hipMemcpyDeviceToHost));
      size_t nbyte_mismatch = 0;
      double max_val_err = 0.0;
      const int SM = std::min(M, 32);
      for (int si = 0; si < SM; si++) {
        const int row = (int)((size_t)si * M / SM);
        for (int k = 0; k < K; k++) {
          const uint8_t bo = tiled_read(A_r4d, row, k, K), br = tiled_read(A_ref, row, k, K);
          if (bo != br) nbyte_mismatch++;
          const double vo = (double)ml84_decode_e4m3(bo) * (double)as_r4d[row];
          const double vr = (double)ml84_decode_e4m3(br) * (double)as_ref[row];
          max_val_err = std::max(max_val_err, std::fabs(vo - vr));
        }
      }
      printf("  GATED_NORM_R4D check (sampled %d/%d rows): %zu byte diffs, max decoded-value diff %.3e %s\n",
             SM, M, nbyte_mismatch, max_val_err,
             max_val_err < 1.0 ? "(PASS, fp32-order tolerance, head permutation honored)" : "(SUSPECT)");

      const double us_r4d = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_gated_norm_tiled_r4d(dO_bf16, o_nb_head, o_nb_tok, dperm, z_f, z_nb1, z_nb2,
                                             dnormw, eps, head_dim, n_heads, h_a_arg, rot_kind, a_dim,
                                             b_dim, M, dA_r4d, da_scale, stream, nullptr);
      });
      const double us_fp32_logical = time_us(stream, iters, [&] {
        rdna4_ml8_qrot_gated_norm_tiled(dO_logical, z_f, z_nb1, z_nb2, dnormw, eps, head_dim, n_heads,
                                         h_a_arg, rot_kind, a_dim, b_dim, M, dA_ref, da_scale_ref, stream, nullptr);
      });
      const double bytes_r4d   = (double)M * K * (2 /*o bf16*/ + 4 /*z*/ + 1 /*fp8 write*/);
      const double bytes_fp32  = (double)M * K * (4 /*o fp32*/ + 4 /*z*/ + 1 /*fp8 write*/);
      printf("  GATED_NORM_R4D timing: r4d-bf16 %.1f us (%.1f GB/s) | fp32-logical %.1f us (%.1f GB/s) | speedup %.2fx "
             "(vs the adapter's separate ~179us/T=1024 output cast+unpermute pass this drops entirely)\n",
             us_r4d, gbps(bytes_r4d, us_r4d), us_fp32_logical, gbps(bytes_fp32, us_fp32_logical),
             us_fp32_logical / us_r4d);
    }
  }

  (void)hipFree(dO_bf16); (void)hipFree(dperm); (void)hipFree(dO_logical);
  (void)hipFree(dnormw); (void)hipFree(dha); (void)hipFree(dz);
  (void)hipFree(da_scale); (void)hipFree(da_scale_ref);
  (void)hipFree(dA_r4d); (void)hipFree(dA_ref);
  return 0;
}

int main() {
  // (M, K) pairs; K determines (a_dim,b_dim,kronecker) below. Default sweep
  // is just M=4096 across the three required K's -- RADQ_ONLY overrides
  // with a single (M,K) if the env var names a K this bench knows how to
  // map to (a_dim,b_dim).
  std::vector<std::pair<int,int>> shapes = {{4096, 5120}, {4096, 6144}, {4096, 17408}};
  if (const char* only = std::getenv("RADQ_ONLY")) {
    int m, k;
    if (sscanf(only, "%d,%d", &m, &k) == 2) shapes = {{m, k}};
  }
  const int iters = std::getenv("RADQ_ITERS") ? std::atoi(std::getenv("RADQ_ITERS")) : 20;

  for (auto [M, K] : shapes) {
    if (K == 5120) run_qrot_shape(/*kronecker=*/true, /*a_dim=*/5, /*b_dim=*/1024, M, iters);
    else if (K == 6144) run_qrot_shape(/*kronecker=*/true, /*a_dim=*/6, /*b_dim=*/1024, M, iters);
    else if (K == 17408) run_qrot_shape(/*kronecker=*/false, /*a_dim=*/136, /*b_dim=*/128, M, iters);
    else printf("skip K=%d: RADQ_ONLY must name one of 5120, 6144, 17408 (this bench's known shapes)\n", K);
  }

  // SILU_MUL_SPLIT_BF16 / ADD_BF16: fixed shapes (LLAMA_ACT_BF16 prefill's
  // actual production sizes -- N=17408 block-Hadamard for the ffn gate/up
  // silu*mul, K=5120 Kronecker for the residual add), regardless of
  // RADQ_ONLY, same reasoning as GATED_NORM below.
  run_silu_mul_split_bf16_shape(/*kronecker=*/false, /*a_dim=*/136, /*b_dim=*/128, /*M=*/4096, iters);
  run_add_bf16_shape(/*kronecker=*/true, /*a_dim=*/5, /*b_dim=*/1024, /*M=*/4096, iters);

  // GATED_NORM: fixed at the production GDN out_proj shape (M=4096,
  // head_dim=128, n_heads=48, K=6144, Kronecker a=6/b=1024) regardless of
  // RADQ_ONLY -- it isn't a (M,K)-only shape (head_dim/n_heads/strides
  // matter too), so it gets its own always-on section rather than folding
  // into the RADQ_ONLY-scoped sweep above.
  {
    const int head_dim = 128, n_heads = 48;
    const size_t z_nb1 = (size_t)head_dim * 4 + 64;                 // packed + 64B padding
    const size_t z_nb2 = (size_t)n_heads * z_nb1 + 256;             // packed + 256B padding
    run_gated_norm_shape(/*M=*/4096, head_dim, n_heads, /*kronecker=*/true,
                          /*a_dim=*/6, /*b_dim=*/1024, z_nb1, z_nb2, iters);
    // Production ssm_out (journal: "qrot: tiled path active (K=6144, a=48, b=128)")
    // is per-head BLOCK-HADAMARD, not Kronecker -- exercise that tier too.
    run_gated_norm_shape(/*M=*/4096, head_dim, n_heads, /*kronecker=*/false,
                          /*a_dim=*/48, /*b_dim=*/128, z_nb1, z_nb2, iters);
    // and a fully packed z (nb1 = head_dim*4, nb2 = n_heads*head_dim*4)
    run_gated_norm_shape(/*M=*/4096, head_dim, n_heads, /*kronecker=*/false,
                          /*a_dim=*/48, /*b_dim=*/128, (size_t)head_dim*4, (size_t)n_heads*head_dim*4, iters);
    // GATED_NORM_R4D: production shape (block-Hadamard a=48/b=128) with a
    // random head permutation standing in for libr4d's own GQA-blocked
    // head order (see run_gated_norm_r4d_shape's own comment).
    run_gated_norm_r4d_shape(/*M=*/4096, head_dim, n_heads, /*kronecker=*/false,
                              /*a_dim=*/48, /*b_dim=*/128, z_nb1, z_nb2, iters);
  }
  return 0;
}
