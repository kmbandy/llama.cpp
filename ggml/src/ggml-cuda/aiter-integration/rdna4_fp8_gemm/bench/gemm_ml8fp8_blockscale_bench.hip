// bench/gemm_ml8fp8_blockscale_bench.hip — MAD-305 Phase 5 shape-parameterised
// bench for BOTH production block-scale GEMM formats:
//   ml8_fp8  : GGML_TYPE_ML8_FP8, G=32  scale groups (rdna4_gemm_ml8fp8_blockscale)
//   fp8_b128 : GGML_TYPE_FP8_B128, G=128 scale groups (rdna4_gemm_fp8b128_blockscale,
//              "round 2" -- RDNA shares the WMMA/VALU issue port, so folding
//              every 32 K paid ~150% VALU overhead; folding every 128 K cuts
//              that to ~1/6th)
//
// Usage:
//   gemm_ml8fp8_blockscale_bench M N K [iters]              # ml8_fp8, single shape
//   gemm_ml8fp8_blockscale_bench --b128 M N K [iters]        # fp8_b128, single shape
//   gemm_ml8fp8_blockscale_bench --all [iters]               # production geometry
//                                                             # list, BOTH formats
//
// Each mode gates correctness at a small oracle shape (CPU reference using
// the SAME quantized bytes the GPU kernel consumes, so this checks the
// kernel's math/layout, not e4m3 rounding noise) before any timed run.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>
#include "../gemm_capi.h"
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

#define HCK(x) do{ hipError_t e_=(x); if(e_!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e_)); exit(3); } }while(0)

static uint8_t enc_e4m3(float x) { __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec_e4m3(uint8_t b) { __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v) = b; return (float) v; }

// Distinct A/B fill patterns (A != B^T) so a transpose/layout bug can't hide,
// values chosen exactly representable in e4m3 (mirrors gemm_trfeed_bench.hip).
static float patA(int i, int k) { return (float) (((i * 7 + k * 3) % 9) - 4) * 0.5f; }
static float patB(int k, int j) { return (float) (((k * 5 + j * 2) % 9) - 4) * 0.5f; }

// Build host-side activation row bytes: [K e4m3][K/G fp32 scale] per row,
// with the scale computed as absmax(row-group)/448 (matching the
// FP8_QUANT_ROT contract) and applied so a_fp8 * a_scale ~= the fp32 value.
// G=32 for ml8_fp8, G=128 for fp8_b128 -- same layout shape, different width.
static void build_activation(std::vector<uint8_t> & a_host, int M, int K, int G) {
    const int n_groups = K / G;
    const int row_bytes = K + n_groups * (int) sizeof(float);
    a_host.assign((size_t) M * row_bytes, 0);
    std::vector<float> row(K);
    for (int m = 0; m < M; ++m) {
        uint8_t * rp = a_host.data() + (size_t) m * row_bytes;
        for (int k = 0; k < K; ++k) row[k] = patA(m, k);
        for (int g = 0; g < n_groups; ++g) {
            float amax = 1e-12f;
            for (int k = g * G; k < g * G + G; ++k) amax = std::max(amax, std::fabs(row[k]));
            float scale = amax / 448.0f;
            if (scale <= 0.0f) scale = 1e-12f;
            for (int k = g * G; k < g * G + G; ++k) rp[k] = enc_e4m3(row[k] / scale);
            memcpy(rp + K + g * (int) sizeof(float), &scale, sizeof(float));
        }
    }
}

static double run_gate_and_time(
    const char * label,
    // oracle args
    const std::vector<uint8_t> & a_host, int oM, int oN, int oK,
    const std::vector<float> & oracle_out,
    // device buffers already populated for the oracle shape
    const uint8_t * d_oa, int o_row_bytes, const uint8_t * d_ob_shuf, const void * d_obscale,
    float * d_oc,
    bool is_b128,
    // perf shape
    int M, int N, int K, int iters,
    const uint8_t * d_a, int row_bytes, const uint8_t * d_b_shuf, const void * d_bscale, float * d_c) {

    // ---- oracle gate ----
    hipError_t rc = is_b128
        ? rdna4_gemm_fp8b128_blockscale(d_oa, o_row_bytes, d_ob_shuf, (const float *) d_obscale, d_oc, oM, oN, oK, 0)
        : rdna4_gemm_ml8fp8_blockscale(d_oa, o_row_bytes, d_ob_shuf, d_obscale, d_oc, oM, oN, oK, 0);
    HCK(rc);
    HCK(hipDeviceSynchronize());

    std::vector<float> got((size_t) oM * oN);
    HCK(hipMemcpy(got.data(), d_oc, got.size() * sizeof(float), hipMemcpyDeviceToHost));
    float max_abs_err = 0.0f, max_ref = 1e-9f;
    for (size_t i = 0; i < got.size(); ++i) {
        max_abs_err = std::max(max_abs_err, std::fabs(got[i] - oracle_out[i]));
        max_ref = std::max(max_ref, std::fabs(oracle_out[i]));
    }
    const float rel = max_abs_err / max_ref;
    const bool pass = rel < 1e-3f;
    printf("[%s] oracle M=%d N=%d K=%d max_abs_err=%.6f max_rel_err=%.6f -> %s\n",
           label, oM, oN, oK, max_abs_err, rel, pass ? "PASS" : "FAIL");
    if (!pass) {
        return -1.0;
    }

    // ---- timed run at the requested (perf) shape ----
    for (int i = 0; i < 5; ++i) {
        HCK(is_b128
            ? rdna4_gemm_fp8b128_blockscale(d_a, row_bytes, d_b_shuf, (const float *) d_bscale, d_c, M, N, K, 0)
            : rdna4_gemm_ml8fp8_blockscale(d_a, row_bytes, d_b_shuf, d_bscale, d_c, M, N, K, 0));
    }
    HCK(hipDeviceSynchronize());

    hipEvent_t s, e; HCK(hipEventCreate(&s)); HCK(hipEventCreate(&e));
    HCK(hipEventRecord(s));
    for (int i = 0; i < iters; ++i) {
        HCK(is_b128
            ? rdna4_gemm_fp8b128_blockscale(d_a, row_bytes, d_b_shuf, (const float *) d_bscale, d_c, M, N, K, 0)
            : rdna4_gemm_ml8fp8_blockscale(d_a, row_bytes, d_b_shuf, d_bscale, d_c, M, N, K, 0));
    }
    HCK(hipEventRecord(e));
    HCK(hipEventSynchronize(e));
    float ms = 0; HCK(hipEventElapsedTime(&ms, s, e));
    ms /= iters;
    hipEventDestroy(s); hipEventDestroy(e);

    const double flops = 2.0 * (double) M * (double) N * (double) K;
    const double tflops = flops / (ms * 1e-3) / 1e12;
    printf("[%s] M=%d N=%d K=%d iters=%d: %.4f ms  %.2f TFLOPS\n", label, M, N, K, iters, ms, tflops);
    return tflops;
}

// ml8_fp8 (G=32): weight scale is fp16 [K/32, N] (g outer, n inner) -- one
// scalar per (group, output column).
static double bench_ml8fp8(int M, int N, int K, int iters) {
    // Oracle shape: deliberately ragged M (not a tile multiple), small N/K.
    const int oM = 33, oN = 32, oK = 64;
    std::vector<uint8_t> oa; build_activation(oa, oM, oK, 32);
    const int o_row_bytes = oK + (oK / 32) * (int) sizeof(float);
    std::vector<uint8_t> ob_kn((size_t) oK * oN);
    for (int k = 0; k < oK; ++k) for (int n = 0; n < oN; ++n) ob_kn[(size_t) k * oN + n] = enc_e4m3(patB(k, n));
    std::vector<uint16_t> ob_scale((size_t) (oK / 32) * oN);
    for (auto & v : ob_scale) { __half h = __float2half(0.03125f); memcpy(&v, &h, sizeof(v)); }

    std::vector<float> oracle_out((size_t) oM * oN, 0.0f);
    for (int m = 0; m < oM; ++m) {
        const uint8_t * rp = oa.data() + (size_t) m * o_row_bytes;
        for (int g = 0; g < oK / 32; ++g) {
            float a_scale; memcpy(&a_scale, rp + oK + g * 4, sizeof(float));
            for (int n = 0; n < oN; ++n) {
                __half bh; memcpy(&bh, &ob_scale[(size_t) g * oN + n], sizeof(__half));
                const float b_scale = __half2float(bh);
                float acc = 0.0f;
                for (int k = g * 32; k < g * 32 + 32; ++k) acc += dec_e4m3(rp[k]) * dec_e4m3(ob_kn[(size_t) k * oN + n]);
                oracle_out[(size_t) m * oN + n] += a_scale * b_scale * acc;
            }
        }
    }

    uint8_t *d_oa, *d_ob_kn, *d_ob_shuf; uint16_t *d_obscale; float *d_oc;
    HCK(hipMalloc(&d_oa, oa.size()));
    HCK(hipMalloc(&d_ob_kn, ob_kn.size()));
    HCK(hipMalloc(&d_ob_shuf, ob_kn.size()));
    HCK(hipMalloc(&d_obscale, ob_scale.size() * sizeof(uint16_t)));
    HCK(hipMalloc(&d_oc, (size_t) oM * oN * sizeof(float)));
    HCK(hipMemcpy(d_oa, oa.data(), oa.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_ob_kn, ob_kn.data(), ob_kn.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_obscale, ob_scale.data(), ob_scale.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HCK(rdna4_preshuffle_b_ml8fp8(d_ob_kn, d_ob_shuf, oK, oN, 0));
    HCK(hipDeviceSynchronize());

    // Round-trip unshuffle check (get_tensor/get_rows fallback path).
    std::vector<uint8_t> ob_kn_rt(ob_kn.size());
    uint8_t * d_ob_rt; HCK(hipMalloc(&d_ob_rt, ob_kn.size()));
    HCK(rdna4_unshuffle_b_ml8fp8(d_ob_shuf, d_ob_rt, oK, oN, 0));
    HCK(hipDeviceSynchronize());
    HCK(hipMemcpy(ob_kn_rt.data(), d_ob_rt, ob_kn.size(), hipMemcpyDeviceToHost));
    printf("[ml8_fp8] preshuffle/unshuffle round-trip -> %s\n", (ob_kn_rt == ob_kn) ? "PASS" : "FAIL");
    hipFree(d_ob_rt);

    // Perf shape.
    const int row_bytes = K + (K / 32) * (int) sizeof(float);
    std::vector<uint8_t> a_host; build_activation(a_host, M, K, 32);
    std::vector<uint8_t> b_kn((size_t) K * N);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) b_kn[(size_t) k * N + n] = enc_e4m3(patB(k, n));
    std::vector<uint16_t> b_scale((size_t) (K / 32) * N);
    for (auto & v : b_scale) { __half h = __float2half(0.03125f); memcpy(&v, &h, sizeof(v)); }

    uint8_t *d_a, *d_b_kn, *d_b_shuf; uint16_t *d_bscale; float *d_c;
    HCK(hipMalloc(&d_a, a_host.size()));
    HCK(hipMalloc(&d_b_kn, b_kn.size()));
    HCK(hipMalloc(&d_b_shuf, b_kn.size()));
    HCK(hipMalloc(&d_bscale, b_scale.size() * sizeof(uint16_t)));
    HCK(hipMalloc(&d_c, (size_t) M * N * sizeof(float)));
    HCK(hipMemcpy(d_a, a_host.data(), a_host.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_b_kn, b_kn.data(), b_kn.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_bscale, b_scale.data(), b_scale.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HCK(rdna4_preshuffle_b_ml8fp8(d_b_kn, d_b_shuf, K, N, 0));
    HCK(hipDeviceSynchronize());

    const double tf = run_gate_and_time("ml8_fp8", oa, oM, oN, oK, oracle_out,
        d_oa, o_row_bytes, d_ob_shuf, d_obscale, d_oc, /*is_b128=*/false,
        M, N, K, iters, d_a, row_bytes, d_b_shuf, d_bscale, d_c);

    hipFree(d_oa); hipFree(d_ob_kn); hipFree(d_ob_shuf); hipFree(d_obscale); hipFree(d_oc);
    hipFree(d_a); hipFree(d_b_kn); hipFree(d_b_shuf); hipFree(d_bscale); hipFree(d_c);
    return tf;
}

// fp8_b128 (G=128): weight scale is fp32 [K/128, N/128] (kb outer, n_tile
// inner) -- one scalar per (128-K-group, 128-wide N tile). N%128==0 required.
static double bench_fp8b128(int M, int N, int K, int iters) {
    const int oM = 33, oN = 128, oK = 256;   // smallest legal (N%128==0, K%128==0), ragged M
    std::vector<uint8_t> oa; build_activation(oa, oM, oK, 128);
    const int o_row_bytes = oK + oK / 32;   // K + K/32 (G=128 => K/128 scales * 4B == K/32 bytes already)
    std::vector<uint8_t> ob_kn((size_t) oK * oN);
    for (int k = 0; k < oK; ++k) for (int n = 0; n < oN; ++n) ob_kn[(size_t) k * oN + n] = enc_e4m3(patB(k, n));
    const int o_kb = oK / 128, o_ntiles = oN / 128;
    std::vector<float> ob_scale((size_t) o_kb * o_ntiles);
    for (size_t i = 0; i < ob_scale.size(); ++i) ob_scale[i] = 0.03125f + 0.001f * (float) i;

    std::vector<float> oracle_out((size_t) oM * oN, 0.0f);
    for (int m = 0; m < oM; ++m) {
        const uint8_t * rp = oa.data() + (size_t) m * o_row_bytes;
        for (int kb = 0; kb < o_kb; ++kb) {
            float a_scale; memcpy(&a_scale, rp + oK + kb * 4, sizeof(float));
            for (int n = 0; n < oN; ++n) {
                const float b_scale = ob_scale[(size_t) kb * o_ntiles + (n / 128)];
                float acc = 0.0f;
                for (int k = kb * 128; k < kb * 128 + 128; ++k) acc += dec_e4m3(rp[k]) * dec_e4m3(ob_kn[(size_t) k * oN + n]);
                oracle_out[(size_t) m * oN + n] += a_scale * b_scale * acc;
            }
        }
    }

    uint8_t *d_oa, *d_ob_kn, *d_ob_shuf; float *d_obscale, *d_oc;
    HCK(hipMalloc(&d_oa, oa.size()));
    HCK(hipMalloc(&d_ob_kn, ob_kn.size()));
    HCK(hipMalloc(&d_ob_shuf, ob_kn.size()));
    HCK(hipMalloc(&d_obscale, ob_scale.size() * sizeof(float)));
    HCK(hipMalloc(&d_oc, (size_t) oM * oN * sizeof(float)));
    HCK(hipMemcpy(d_oa, oa.data(), oa.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_ob_kn, ob_kn.data(), ob_kn.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_obscale, ob_scale.data(), ob_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    HCK(rdna4_preshuffle_b_ml8fp8(d_ob_kn, d_ob_shuf, oK, oN, 0));   // same shuffle, reused for fp8_b128
    HCK(hipDeviceSynchronize());

    std::vector<uint8_t> ob_kn_rt(ob_kn.size());
    uint8_t * d_ob_rt; HCK(hipMalloc(&d_ob_rt, ob_kn.size()));
    HCK(rdna4_unshuffle_b_ml8fp8(d_ob_shuf, d_ob_rt, oK, oN, 0));
    HCK(hipDeviceSynchronize());
    HCK(hipMemcpy(ob_kn_rt.data(), d_ob_rt, ob_kn.size(), hipMemcpyDeviceToHost));
    printf("[fp8_b128] preshuffle/unshuffle round-trip -> %s\n", (ob_kn_rt == ob_kn) ? "PASS" : "FAIL");
    hipFree(d_ob_rt);

    if (N % 128 != 0 || K % 128 != 0) {
        fprintf(stderr, "fp8_b128: N and K must be %%128==0 (got N=%d K=%d)\n", N, K);
        hipFree(d_oa); hipFree(d_ob_kn); hipFree(d_ob_shuf); hipFree(d_obscale); hipFree(d_oc);
        return -1.0;
    }
    const int row_bytes = K + K / 32;   // G=128 => K/128 scales * 4B == K/32 bytes already
    std::vector<uint8_t> a_host; build_activation(a_host, M, K, 128);
    std::vector<uint8_t> b_kn((size_t) K * N);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) b_kn[(size_t) k * N + n] = enc_e4m3(patB(k, n));
    const int kb_n = K / 128, ntiles_n = N / 128;
    std::vector<float> b_scale((size_t) kb_n * ntiles_n);
    for (size_t i = 0; i < b_scale.size(); ++i) b_scale[i] = 0.03125f + 0.0005f * (float) i;

    uint8_t *d_a, *d_b_kn, *d_b_shuf; float *d_bscale, *d_c;
    HCK(hipMalloc(&d_a, a_host.size()));
    HCK(hipMalloc(&d_b_kn, b_kn.size()));
    HCK(hipMalloc(&d_b_shuf, b_kn.size()));
    HCK(hipMalloc(&d_bscale, b_scale.size() * sizeof(float)));
    HCK(hipMalloc(&d_c, (size_t) M * N * sizeof(float)));
    HCK(hipMemcpy(d_a, a_host.data(), a_host.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_b_kn, b_kn.data(), b_kn.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_bscale, b_scale.data(), b_scale.size() * sizeof(float), hipMemcpyHostToDevice));
    HCK(rdna4_preshuffle_b_ml8fp8(d_b_kn, d_b_shuf, K, N, 0));
    HCK(hipDeviceSynchronize());

    const double tf = run_gate_and_time("fp8_b128", oa, oM, oN, oK, oracle_out,
        d_oa, o_row_bytes, d_ob_shuf, d_obscale, d_oc, /*is_b128=*/true,
        M, N, K, iters, d_a, row_bytes, d_b_shuf, d_bscale, d_c);

    hipFree(d_oa); hipFree(d_ob_kn); hipFree(d_ob_shuf); hipFree(d_obscale); hipFree(d_oc);
    hipFree(d_a); hipFree(d_b_kn); hipFree(d_b_shuf); hipFree(d_bscale); hipFree(d_c);
    return tf;
}

// Production geometry sweep (MAD-305 Phase 5 round 2 perf gate). K-split
// shapes (5120,4224,1920,12416) come from the router chain's real weight
// splits; the small-M rows cover decode (M=2) / speculative-verify (M=33)
// and a mid prefill batch (M=512) at the largest N,K pair.
struct shape_t { int M, N, K; };
static const shape_t kProdShapes[] = {
    {2048, 17408, 5120},
    {2048, 5120, 17408},
    {2048, 5120, 12416},
    {2048, 5120, 4224},
    {512,  17408, 5120},
    {33,   17408, 5120},
    {2,    17408, 5120},
};

int main(int argc, char ** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--all") {
        const int iters = argc > 2 ? atoi(argv[2]) : 50;
        bool all_ok = true;
        for (const auto & sh : kProdShapes) {
            const double tf = bench_ml8fp8(sh.M, sh.N, sh.K, iters);
            all_ok &= (tf >= 0.0);
        }
        for (const auto & sh : kProdShapes) {
            // fp8_b128 needs N,K %128==0; the K=4224/1920/12416 K-split shapes
            // and N=17408/5120 all satisfy that (see gemm_capi.h comment).
            if (sh.N % 128 != 0 || sh.K % 128 != 0) {
                fprintf(stderr, "[fp8_b128] skipping M=%d N=%d K=%d (not %%128==0)\n", sh.M, sh.N, sh.K);
                continue;
            }
            const double tf = bench_fp8b128(sh.M, sh.N, sh.K, iters);
            all_ok &= (tf >= 0.0);
        }
        return all_ok ? 0 : 1;
    }

    bool is_b128 = false;
    int argi = 1;
    if (argc >= 2 && std::string(argv[1]) == "--b128") {
        is_b128 = true;
        argi = 2;
    }
    if (argc < argi + 3) {
        fprintf(stderr, "usage: %s [--b128] M N K [iters]\n"
                        "       %s --all [iters]\n", argv[0], argv[0]);
        return 2;
    }
    const int M = atoi(argv[argi]), N = atoi(argv[argi + 1]), K = atoi(argv[argi + 2]);
    const int iters = argc > argi + 3 ? atoi(argv[argi + 3]) : 50;

    const double tf = is_b128 ? bench_fp8b128(M, N, K, iters) : bench_ml8fp8(M, N, K, iters);
    return tf >= 0.0 ? 0 : 1;
}
