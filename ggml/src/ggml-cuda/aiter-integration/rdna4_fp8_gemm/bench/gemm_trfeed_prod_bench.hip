// bench/gemm_trfeed_prod_bench.hip — MAD-305 Phase 5 (round 3) production
// entry-point bench. Exercises rdna4_gemm_fp8_trfeed (gemm_capi.h) -- the
// FROZEN Phase-1 "trfeed" kernel (bench/trfeed_kernels.h's
// gemm_fp8_trfeed<128,2>, byte-identical to what gemm_trfeed_bench.hip
// measures) through the SAME production glue GGML_OP_FP8_MUL_MAT's HIP
// dispatch (ml8.cu) uses: preshuffle B once via rdna4_preshuffle_b_ml8fp8,
// pad M up to a multiple of BM=128 (the frozen kernel's A-tile LDS fill is
// UNGUARDED -- see gemm_capi.h), launch, then read back only the first M
// rows. Oracle-gated per-row a_scale x per-column b_scale, bf16 compare,
// at every FP8_MUL_MAT production shape from the MAD-305 handoff plus the
// M=33/M=2 decode-batch edge cases (both exercise the M_pad path since
// neither is already a multiple of 128).
//
// For every M<=32 (decode/verify tile) shape, also exercises
// rdna4_gemm_fp8_trfeed_splitk (gemm_capi.h) -- the split-K decode variant
// that trades atomic-add fp32 reduction for far more workgroups streaming
// the weight in parallel -- against the same oracle samples, at
// n_splits in {1,2,4,8,16} plus the rdna4_trfeed_splitk_default_splits()
// heuristic, printing ms and effective GB/s (N*K weight bytes / time)
// alongside the non-split-K number above.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "trfeed_common.h"
#include "../gemm_capi.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

static uint8_t enc(float x){ __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec(uint8_t b){ __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v)=b; return (float)v; }

// Same A != B^T pattern as gemm_trfeed_bench.hip (a transpose bug can't hide).
static float patA(int i,int k){ return (float)(((i*7 + k*3) % 9) - 4) * 0.5f; }
static float patB(int k,int j){ return (float)(((k*5 + j*2) % 9) - 4) * 0.5f; }
// Distinct, non-trivial per-row / per-column scales (not all-ones): exercises
// the epilogue's a_scale[m] * b_scale[n] multiply, not just the raw matmul.
static float scaleA(int i){ return 0.6f + 0.05f * (float)(i % 11); }
static float scaleB(int j){ return 0.4f + 0.03f * (float)(j % 13); }

static int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

// Runs one (M,N,K) shape: builds host A/B/scales, preshuffles B on-device,
// pads M up to a multiple of 128 exactly like ggml_cuda_op_fp8_mul_mat does,
// launches rdna4_gemm_fp8_trfeed, checks vs a CPU oracle, then times it.
static int run_shape(int M, int N, int K, int iters) {
    if (K % 32 != 0 || N % 128 != 0) {
        fprintf(stderr, "shape M=%d N=%d K=%d skipped: K%%32 or N%%128 violated\n", M, N, K);
        return 3;
    }
    // Mirrors ggml_cuda_op_fp8_mul_mat's tile choice: M<=32 -> the decode/
    // verify tile (gemm_fp8_trfeed<32,1>, M_pad round_up(M,32)==32 exactly);
    // M>32 -> the prefill tile (gemm_fp8_trfeed<128,2>, M_pad round_up(M,128)).
    // rdna4_gemm_fp8_trfeed infers which tile from M_pad itself.
    const int M_pad = (M <= 32) ? 32 : round_up(M, 128);

    std::vector<uint8_t> Ah((size_t)M * K), Bh((size_t)K * N);
    std::vector<float> as(M), bs(N);
    for (int i = 0; i < M; i++) { as[i] = scaleA(i); for (int k = 0; k < K; k++) Ah[(size_t)i*K+k] = enc(patA(i,k)); }
    for (int j = 0; j < N; j++) bs[j] = scaleB(j);
    for (int k = 0; k < K; k++) for (int j = 0; j < N; j++) Bh[(size_t)k*N+j] = enc(patB(k,j));

    // A full M*N*K CPU oracle is intractable at these shapes (2048x17408x5120
    // is ~1.8e11 MACs); a bit-exact kernel bug shows up at ANY sampled output
    // element just as reliably (the A!=B^T pattern has no symmetry to hide
    // behind), so spot-check a deterministic, shape-spanning sample of
    // (i,j) pairs instead -- O(samples*K), not O(M*N*K).
    struct Sample { int i, j; float ref; };
    std::vector<Sample> samples;
    const int SI = std::min(M, 32), SJ = std::min(N, 32);
    samples.reserve((size_t)SI * SJ);
    for (int si = 0; si < SI; si++) {
        const int i = (int)((size_t)si * M / SI);
        for (int sj = 0; sj < SJ; sj++) {
            const int j = (int)((size_t)sj * N / SJ);
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += dec(Ah[(size_t)i*K+k]) * dec(Bh[(size_t)k*N+j]);
            samples.push_back({i, j, acc * as[i] * bs[j]});
        }
    }

    // ---- device buffers, padded exactly like ggml_cuda_op_fp8_mul_mat ----
    uint8_t *A_pad = nullptr, *B_kn = nullptr, *B_shuf = nullptr;
    float *as_pad = nullptr, *bscale = nullptr;
    __hip_bfloat16 *C_bf16 = nullptr;
    HCK(hipMalloc(&A_pad, (size_t)M_pad * K));
    HCK(hipMalloc(&B_kn, (size_t)K * N));
    HCK(hipMalloc(&B_shuf, (size_t)K * N));
    HCK(hipMalloc(&as_pad, (size_t)M_pad * 4));
    HCK(hipMalloc(&bscale, (size_t)N * 4));
    HCK(hipMalloc(&C_bf16, (size_t)M_pad * N * 2));

    HCK(hipMemset(A_pad, 0, (size_t)M_pad * K));
    HCK(hipMemset(as_pad, 0, (size_t)M_pad * 4));
    HCK(hipMemcpy(A_pad, Ah.data(), (size_t)M * K, hipMemcpyHostToDevice));
    HCK(hipMemcpy(as_pad, as.data(), (size_t)M * 4, hipMemcpyHostToDevice));
    HCK(hipMemcpy(B_kn, Bh.data(), (size_t)K * N, hipMemcpyHostToDevice));
    HCK(hipMemcpy(bscale, bs.data(), (size_t)N * 4, hipMemcpyHostToDevice));

    hipStream_t stream = 0;
    HCK(rdna4_preshuffle_b_ml8fp8(B_kn, B_shuf, K, N, stream));
    HCK(hipStreamSynchronize(stream));

    hipError_t rc = rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, bscale, M_pad, N, K, stream);
    if (rc != hipSuccess) {
        fprintf(stderr, "shape M=%d N=%d K=%d: rdna4_gemm_fp8_trfeed failed: %s\n",
                M, N, K, hipGetErrorString(rc));
        return 3;
    }
    HCK(hipDeviceSynchronize());

    // ---- oracle check: pull back only the sampled (i,j) bf16 values ----
    float mx = 0.f;
    for (const auto & samp : samples) {
        uint16_t hi;
        HCK(hipMemcpy(&hi, (const uint16_t*)C_bf16 + (size_t)samp.i * N + samp.j, 2, hipMemcpyDeviceToHost));
        uint32_t bits = (uint32_t)hi << 16; float gv; memcpy(&gv, &bits, 4);
        float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
        if (e > mx) mx = e;
    }
    const bool pass = mx < 0.03f;
    printf("  M=%-5d N=%-6d K=%-6d M_pad=%-5d  max_rel_err=%.4f  %s",
           M, N, K, M_pad, mx, pass ? "PASS" : "*** FAIL ***");

    // ---- perf: 5 warmup + `iters` timed, grid sized to M_pad (matches prod) ----
    for (int i = 0; i < 5; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, bscale, M_pad, N, K, stream);
    HCK(hipDeviceSynchronize());
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for (int i = 0; i < iters; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, bscale, M_pad, N, K, stream);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e); ms /= iters;
    hipEventDestroy(s); hipEventDestroy(e);
    // FLOPs counted against the REAL M (production work), not M_pad (padding
    // overhead is real cost but not real work) -- matches how the fleet has
    // quoted "TF at production shapes" elsewhere (gemm_blockscale.hip).
    const double gflop = 2.0 * (double)M * N * K / 1e12;
    const double tf = gflop / (ms / 1e3);
    printf("   %8.4f ms  %7.2f TF\n", ms, tf);

    // ---- split-K decode variant (M_pad==32 only -- rdna4_gemm_fp8_trfeed_splitk
    // is not defined/usable for the prefill tile): same oracle samples, same
    // bf16-level relative-error tolerance (split-K accumulates in fp32, so it
    // should be at least as accurate as the frozen kernel's bf16 output), and
    // per-n_splits timing + effective GB/s (weight bytes N*K streamed / time --
    // the whole point of split-K is to raise how much of the 640 GB/s the
    // decode shape's tiny M actually captures).
    int splitk_failures = 0;
    if (M_pad == 32) {
        float* C_f32 = nullptr;
        HCK(hipMalloc(&C_f32, (size_t)M_pad * N * sizeof(float)));

        const int default_splits = rdna4_trfeed_splitk_default_splits(N, K);
        static const int try_splits[] = {1, 2, 4, 8, 16};
        for (int trial = 0; trial < (int)(sizeof(try_splits)/sizeof(try_splits[0])) + 1; trial++) {
            const bool is_default = trial == (int)(sizeof(try_splits)/sizeof(try_splits[0]));
            const int n_splits = is_default ? default_splits : try_splits[trial];

            hipError_t rc2 = rdna4_gemm_fp8_trfeed_splitk(A_pad, B_shuf, C_f32, as_pad, bscale,
                                                          M_pad, N, K, n_splits, stream);
            if (rc2 != hipSuccess) {
                fprintf(stderr, "  splitk M=%d N=%d K=%d n_splits=%d: dispatch failed: %s\n",
                        M, N, K, n_splits, hipGetErrorString(rc2));
                splitk_failures++;
                continue;
            }
            HCK(hipDeviceSynchronize());

            float mx2 = 0.f;
            for (const auto & samp : samples) {
                float gv;
                HCK(hipMemcpy(&gv, C_f32 + (size_t)samp.i * N + samp.j, 4, hipMemcpyDeviceToHost));
                float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
                if (e > mx2) mx2 = e;
            }
            const bool pass2 = mx2 < 0.03f;
            if (!pass2) splitk_failures++;

            for (int i = 0; i < 5; i++)
                rdna4_gemm_fp8_trfeed_splitk(A_pad, B_shuf, C_f32, as_pad, bscale, M_pad, N, K, n_splits, stream);
            HCK(hipDeviceSynchronize());
            hipEvent_t s2, e2; hipEventCreate(&s2); hipEventCreate(&e2);
            hipEventRecord(s2);
            for (int i = 0; i < iters; i++)
                rdna4_gemm_fp8_trfeed_splitk(A_pad, B_shuf, C_f32, as_pad, bscale, M_pad, N, K, n_splits, stream);
            hipEventRecord(e2); hipEventSynchronize(e2);
            float ms2 = 0; hipEventElapsedTime(&ms2, s2, e2); ms2 /= iters;
            hipEventDestroy(s2); hipEventDestroy(e2);

            const double gbytes = (double)N * (double)K / 1e9;
            const double gbps = gbytes / (ms2 / 1e3);
            printf("    splitk n_splits=%-3d%s max_rel_err=%.4f  %s   %8.4f ms  %7.1f GB/s\n",
                   n_splits, is_default ? " (default)" : "          ",
                   mx2, pass2 ? "PASS" : "*** FAIL ***", ms2, gbps);
        }
        hipFree(C_f32);
    }

    hipFree(A_pad); hipFree(B_kn); hipFree(B_shuf); hipFree(as_pad); hipFree(bscale); hipFree(C_bf16);
    return (pass && splitk_failures == 0) ? 0 : 1;
}

int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("device: %s  CUs=%d\n", p.name, p.multiProcessorCount);
    printf("=== rdna4_gemm_fp8_trfeed (frozen Phase-1 kernel) production-shape bench ===\n");

    struct Shape { int M, N, K; };
    // MAD-305 handoff production shapes + decode-batch edge cases. M=2/16
    // exercise the NEW decode/verify tile (gemm_fp8_trfeed<32,1>, M_pad=32);
    // M=33 exercises the prefill tile's M_pad round-up-to-128 path.
    static const Shape shapes[] = {
        {2048, 17408, 5120},
        {2048,  5120,17408},
        {2048,  5120,12416},
        {2048,  5120, 4224},
        { 512, 17408, 5120},
        {  33, 17408, 5120},
        {  16, 17408, 5120},
        {   2, 17408, 5120},
        {   8,  5120,17408},   // ffn_down-shaped decode: only N/128=40 workgroups
    };
    int failures = 0;
    for (const auto & s : shapes) {
        int rc = run_shape(s.M, s.N, s.K, 30);
        if (rc != 0) failures++;
    }
    if (failures) {
        printf("\n%d/%zu SHAPES FAILED THE ORACLE GATE.\n", failures, sizeof(shapes)/sizeof(shapes[0]));
        return 1;
    }
    printf("\nall shapes PASS.\n");
    return 0;
}
