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
    int fail_extra = 0;

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

    // ---- fp32-output variant: rdna4_gemm_fp8_trfeed_f32 (MAD-305, freeze lifted
    // 2026-09-18) -- removes the bf16 round trip + convert_unary<bf16,float> pass
    // the bf16 path above needs. Destination is sized to the CALLER's true M (no
    // padded scratch), M_valid=M -- the realistic ggml call shape. Checked against
    // BOTH the fp32 CPU oracle (samp.ref, same samples as above) and the bf16
    // kernel's own output already read back above (bf16_ref) -- the f32 store
    // should be at least as accurate as the bf16 round trip, never worse. Also
    // timed against the bf16 launch above at the SAME shape/iters so the report
    // can show the extra fp32 store bytes didn't cost real throughput.
    {
        std::vector<float> bf16_ref(samples.size());
        for (size_t si = 0; si < samples.size(); si++) {
            uint16_t hi;
            HCK(hipMemcpy(&hi, (const uint16_t*)C_bf16 + (size_t)samples[si].i * N + samples[si].j, 2, hipMemcpyDeviceToHost));
            uint32_t bits = (uint32_t)hi << 16; float gv; memcpy(&gv, &bits, 4);
            bf16_ref[si] = gv;
        }

        float* C_f32 = nullptr;
        HCK(hipMalloc(&C_f32, (size_t)M * N * sizeof(float)));   // exactly M rows -- no padded scratch
        HCK(hipMemset(C_f32, 0, (size_t)M * N * sizeof(float)));

        hipError_t rcf = rdna4_gemm_fp8_trfeed_f32((const uint8_t*)A_pad, (const uint8_t*)B_shuf, C_f32,
                                                    as_pad, bscale, M_pad, /*M_valid=*/M, N, K, stream);
        if (rcf != hipSuccess) {
            fprintf(stderr, "shape M=%d N=%d K=%d: rdna4_gemm_fp8_trfeed_f32 failed: %s\n",
                    M, N, K, hipGetErrorString(rcf));
            hipFree(C_f32);
            return 3;
        }
        HCK(hipDeviceSynchronize());

        float mxf_ref = 0.f, mxf_bf16 = 0.f;
        for (size_t si = 0; si < samples.size(); si++) {
            float gv;
            HCK(hipMemcpy(&gv, C_f32 + (size_t)samples[si].i * N + samples[si].j, 4, hipMemcpyDeviceToHost));
            float eref  = fabsf(gv - samples[si].ref) / (fabsf(samples[si].ref) + 1e-3f);
            float ebf16 = fabsf(gv - bf16_ref[si]);
            if (eref  > mxf_ref)  mxf_ref  = eref;
            if (ebf16 > mxf_bf16) mxf_bf16 = ebf16;
        }
        const bool passf = mxf_ref < 0.03f;
        if (!passf) fail_extra++;
        printf("    f32     M_valid=%-5d max_rel_err_vs_ref=%.4f max_abs_diff_vs_bf16=%.5f  %s",
               M, mxf_ref, mxf_bf16, passf ? "PASS" : "*** FAIL ***");

        for (int i = 0; i < 5; i++)
            rdna4_gemm_fp8_trfeed_f32((const uint8_t*)A_pad, (const uint8_t*)B_shuf, C_f32, as_pad, bscale, M_pad, M, N, K, stream);
        HCK(hipDeviceSynchronize());
        hipEvent_t sf, ef; hipEventCreate(&sf); hipEventCreate(&ef);
        hipEventRecord(sf);
        for (int i = 0; i < iters; i++)
            rdna4_gemm_fp8_trfeed_f32((const uint8_t*)A_pad, (const uint8_t*)B_shuf, C_f32, as_pad, bscale, M_pad, M, N, K, stream);
        hipEventRecord(ef); hipEventSynchronize(ef);
        float msf = 0; hipEventElapsedTime(&msf, sf, ef); msf /= iters;
        hipEventDestroy(sf); hipEventDestroy(ef);
        const double tff = gflop / (msf / 1e3);
        printf("   %8.4f ms  %7.2f TF  (vs bf16 %+.1f%%)\n", msf, tff, 100.0 * (msf - ms) / ms);

        // ---- ragged M_valid = M_pad - 37 bounds test: proves the epilogue's
        // M_valid guard (not just the M==M_pad happy path above) is correct --
        // destination sized to M_ragged rows (whatever M_valid asks for, not the
        // shape's real M), and rows [M, M_ragged) (present only when M_ragged >
        // M, i.e. the ragged cut falls INSIDE the zero-padded region A_pad/as_pad
        // already carry) have a well-defined reference of exactly 0 (A_pad's row
        // is all-zero there), so this also exercises the zero-fill path, not only
        // the truncation path.
        const int M_ragged = M_pad - 37;
        if (M_ragged > 0 && M_ragged != M) {
            float* C_ragged = nullptr;
            HCK(hipMalloc(&C_ragged, (size_t)M_ragged * N * sizeof(float)));
            HCK(hipMemset(C_ragged, 0, (size_t)M_ragged * N * sizeof(float)));
            hipError_t rcr = rdna4_gemm_fp8_trfeed_f32((const uint8_t*)A_pad, (const uint8_t*)B_shuf, C_ragged,
                                                        as_pad, bscale, M_pad, M_ragged, N, K, stream);
            if (rcr != hipSuccess) {
                fprintf(stderr, "shape M=%d N=%d K=%d: rdna4_gemm_fp8_trfeed_f32 (ragged M_valid=%d) failed: %s\n",
                        M, N, K, M_ragged, hipGetErrorString(rcr));
                hipFree(C_ragged); hipFree(C_f32);
                return 3;
            }
            HCK(hipDeviceSynchronize());
            float mx_ragged = 0.f;
            const int SIr = std::min(M_ragged, 32);
            for (int si = 0; si < SIr; si++) {
                const int i = (int)((size_t)si * M_ragged / SIr);
                for (int sj = 0; sj < SJ; sj++) {
                    const int j = (int)((size_t)sj * N / SJ);
                    float ref = 0.f;
                    if (i < M) {
                        float acc = 0.f;
                        for (int k = 0; k < K; k++) acc += dec(Ah[(size_t)i*K+k]) * dec(Bh[(size_t)k*N+j]);
                        ref = acc * as[i] * bs[j];
                    }
                    float gv;
                    HCK(hipMemcpy(&gv, C_ragged + (size_t)i * N + j, 4, hipMemcpyDeviceToHost));
                    float e = fabsf(gv - ref) / (fabsf(ref) + 1e-3f);
                    if (e > mx_ragged) mx_ragged = e;
                }
            }
            const bool pass_ragged = mx_ragged < 0.03f;
            if (!pass_ragged) fail_extra++;
            printf("    f32-ragged M_valid=%-5d (M_pad-37) max_rel_err=%.4f  %s\n",
                   M_ragged, mx_ragged, pass_ragged ? "PASS" : "*** FAIL ***");
            hipFree(C_ragged);
        }
        hipFree(C_f32);
    }

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
    return (pass && splitk_failures == 0 && fail_extra == 0) ? 0 : 1;
}

// ---------------- SwiGLU epilogue kernel bench (MAD-305 fused-FFN task) ----------------
// Exercises rdna4_gemm_fp8_trfeed_swiglu_f32 directly (gemm_capi.h): builds
// two independent "gate"/"up" e4m3 weight matrices [K, N_half], interleaves
// them host-side into ONE fused B_transposed[K, N_fused] using the SAME
// 64-columns-per-128-tile placement rule rdna4_expand_ml84_pair_to_trfeed
// uses (gemm_ml84_prod.hip): fused column j = 128*block + pos, pos<64 comes
// from gate column block*64+pos, pos>=64 comes from up column
// block*64+(pos-64). Preshuffles that fused matrix with the EXISTING
// rdna4_preshuffle_b_ml8fp8 (this bench deliberately stays on plain e4m3
// weights, not ML8_4 nibbles -- it isolates the SwiGLU KERNEL itself; the
// full ML8_4-weight path, expander included, is covered by
// tests/test-backend-ops.cpp's test_ml8_ffn_gate_up_swiglu instead). Then
// compares ONE fused-kernel launch against TWO separate
// rdna4_gemm_fp8_trfeed_f32 launches (gate, up) plus a HOST-side
// silu(gate)*up reference, for both correctness (host reference) and
// performance (fused vs the sum of the two plain-GEMM launches it replaces
// -- the exact comparison this task asked for).
static int run_swiglu_shape(int M, int N_half, int K, int iters) {
    if (K % 32 != 0 || N_half % 64 != 0) {
        fprintf(stderr, "swiglu shape M=%d N_half=%d K=%d skipped: K%%32 or N_half%%64 violated\n", M, N_half, K);
        return 3;
    }
    const int N_fused = 2 * N_half;
    const int M_pad = (M <= 32) ? 32 : round_up(M, 128);

    std::vector<uint8_t> Ah((size_t) M * K), Bg((size_t) K * N_half), Bu((size_t) K * N_half);
    std::vector<float> as(M), bsg(N_half), bsu(N_half);
    for (int i = 0; i < M; i++) {
        as[i] = scaleA(i);
        for (int k = 0; k < K; k++) Ah[(size_t) i * K + k] = enc(patA(i, k));
    }
    for (int j = 0; j < N_half; j++) { bsg[j] = scaleB(j); bsu[j] = scaleB(j + 7919); }
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N_half; j++) {
            Bg[(size_t) k * N_half + j] = enc(patB(k, j));
            Bu[(size_t) k * N_half + j] = enc(patB(k + 104729, j));   // large odd offset -> a visibly different pattern
        }
    }

    // ROUND 2 interleave (32-column granularity, matches
    // rdna4_expand_ml84_pair_to_trfeed / trfeed_kernels.h's SwiGLU epilogue
    // exactly): within each 128-fused-column block, pos<32 is gate channel
    // [0,32), [32,64) is up channel [0,32) (SAME channel, paired), [64,96)
    // is gate channel [32,64), [96,128) is up channel [32,64). This is what
    // puts a gate/up pair in the SAME GEMM wave's two fragments -- see the
    // 22.50ms-vs-4.73ms round-1 regression this bench originally caught.
    std::vector<uint8_t> Bf((size_t) K * N_fused);
    std::vector<float> bsf(N_fused);
    const int n_blocks = N_half / 64;
    auto pair_col_src = [](int j, bool* is_gate, int* n_src) {
        const int block = j / 128, pos = j % 128;
        const int half = pos / 64, sub = pos % 64;
        *is_gate = sub < 32;
        const int channel = half * 32 + (*is_gate ? sub : sub - 32);
        *n_src = block * 64 + channel;
    };
    for (int j = 0; j < N_fused; j++) {
        bool is_gate; int n_src;
        pair_col_src(j, &is_gate, &n_src);
        bsf[j] = is_gate ? bsg[n_src] : bsu[n_src];
    }
    for (int k = 0; k < K; k++) {
        for (int block = 0; block < n_blocks; block++) {
            for (int pos = 0; pos < 128; pos++) {
                const int j = block * 128 + pos;
                bool is_gate; int n_src;
                pair_col_src(j, &is_gate, &n_src);
                Bf[(size_t) k * N_fused + j] = is_gate ? Bg[(size_t) k * N_half + n_src] : Bu[(size_t) k * N_half + n_src];
            }
        }
    }

    hipStream_t stream = 0;
    uint8_t *A_pad = nullptr, *Bg_d = nullptr, *Bu_d = nullptr, *Bf_d = nullptr;
    uint8_t *Bg_shuf = nullptr, *Bu_shuf = nullptr, *Bf_shuf = nullptr;
    float *as_pad = nullptr, *bsg_d = nullptr, *bsu_d = nullptr, *bsf_d = nullptr;
    HCK(hipMalloc(&A_pad, (size_t) M_pad * K));
    HCK(hipMemset(A_pad, 0, (size_t) M_pad * K));
    HCK(hipMemcpy(A_pad, Ah.data(), (size_t) M * K, hipMemcpyHostToDevice));
    HCK(hipMalloc(&as_pad, (size_t) M_pad * sizeof(float)));
    HCK(hipMemset(as_pad, 0, (size_t) M_pad * sizeof(float)));
    HCK(hipMemcpy(as_pad, as.data(), (size_t) M * sizeof(float), hipMemcpyHostToDevice));

    HCK(hipMalloc(&Bg_d, (size_t) K * N_half));
    HCK(hipMemcpy(Bg_d, Bg.data(), (size_t) K * N_half, hipMemcpyHostToDevice));
    HCK(hipMalloc(&Bu_d, (size_t) K * N_half));
    HCK(hipMemcpy(Bu_d, Bu.data(), (size_t) K * N_half, hipMemcpyHostToDevice));
    HCK(hipMalloc(&Bf_d, (size_t) K * N_fused));
    HCK(hipMemcpy(Bf_d, Bf.data(), (size_t) K * N_fused, hipMemcpyHostToDevice));
    HCK(hipMalloc(&Bg_shuf, (size_t) K * N_half));
    HCK(hipMalloc(&Bu_shuf, (size_t) K * N_half));
    HCK(hipMalloc(&Bf_shuf, (size_t) K * N_fused));
    HCK(hipMalloc(&bsg_d, N_half * sizeof(float)));
    HCK(hipMemcpy(bsg_d, bsg.data(), N_half * sizeof(float), hipMemcpyHostToDevice));
    HCK(hipMalloc(&bsu_d, N_half * sizeof(float)));
    HCK(hipMemcpy(bsu_d, bsu.data(), N_half * sizeof(float), hipMemcpyHostToDevice));
    HCK(hipMalloc(&bsf_d, N_fused * sizeof(float)));
    HCK(hipMemcpy(bsf_d, bsf.data(), N_fused * sizeof(float), hipMemcpyHostToDevice));

    if (rdna4_preshuffle_b_ml8fp8(Bg_d, Bg_shuf, K, N_half, stream) != hipSuccess ||
        rdna4_preshuffle_b_ml8fp8(Bu_d, Bu_shuf, K, N_half, stream) != hipSuccess ||
        rdna4_preshuffle_b_ml8fp8(Bf_d, Bf_shuf, K, N_fused, stream) != hipSuccess) {
        fprintf(stderr, "swiglu shape M=%d N_half=%d K=%d: preshuffle failed\n", M, N_half, K);
        return 3;
    }
    HCK(hipDeviceSynchronize());

    float *Cg = nullptr, *Cu = nullptr, *Cf = nullptr;
    HCK(hipMalloc(&Cg, (size_t) M * N_half * sizeof(float)));
    HCK(hipMalloc(&Cu, (size_t) M * N_half * sizeof(float)));
    HCK(hipMalloc(&Cf, (size_t) M * N_half * sizeof(float)));

    if (rdna4_gemm_fp8_trfeed_f32(A_pad, Bg_shuf, Cg, as_pad, bsg_d, M_pad, M, N_half, K, stream) != hipSuccess ||
        rdna4_gemm_fp8_trfeed_f32(A_pad, Bu_shuf, Cu, as_pad, bsu_d, M_pad, M, N_half, K, stream) != hipSuccess) {
        fprintf(stderr, "swiglu shape M=%d N_half=%d K=%d: plain gemm failed\n", M, N_half, K);
        return 3;
    }
    const hipError_t rcf = rdna4_gemm_fp8_trfeed_swiglu_f32(A_pad, Bf_shuf, Cf, as_pad, bsf_d, M_pad, M, N_fused, K, stream);
    if (rcf != hipSuccess) {
        fprintf(stderr, "swiglu shape M=%d N_half=%d K=%d: rdna4_gemm_fp8_trfeed_swiglu_f32 failed: %s\n",
                M, N_half, K, hipGetErrorString(rcf));
        return 3;
    }
    HCK(hipDeviceSynchronize());

    std::vector<float> hg((size_t) M * N_half), hu((size_t) M * N_half), hf((size_t) M * N_half);
    HCK(hipMemcpy(hg.data(), Cg, hg.size() * sizeof(float), hipMemcpyDeviceToHost));
    HCK(hipMemcpy(hu.data(), Cu, hu.size() * sizeof(float), hipMemcpyDeviceToHost));
    HCK(hipMemcpy(hf.data(), Cf, hf.size() * sizeof(float), hipMemcpyDeviceToHost));

    float mx_err = 0.f;
    for (size_t idx = 0; idx < hg.size(); idx++) {
        const float g = hg[idx], u = hu[idx];
        const float ref = (g / (1.0f + expf(-g))) * u;   // host-side silu(gate)*up reference
        const float got = hf[idx];
        const float e = fabsf(got - ref) / (fabsf(ref) + 1e-3f);
        if (e > mx_err) mx_err = e;
    }
    const bool pass = mx_err < 0.03f;
    printf("  swiglu M=%-5d N_half=%-6d K=%-6d max_rel_err_vs_2gemm+host_silu=%.4f  %s\n",
           M, N_half, K, mx_err, pass ? "PASS" : "*** FAIL ***");

    // Perf: the fused kernel vs the SUM of the two plain-GEMM launches it
    // replaces (this is the comparison the task asked for; the host-side
    // silu(gate)*up above is the correctness reference only, not timed).
    for (int i = 0; i < 5; i++) {
        rdna4_gemm_fp8_trfeed_f32(A_pad, Bg_shuf, Cg, as_pad, bsg_d, M_pad, M, N_half, K, stream);
        rdna4_gemm_fp8_trfeed_f32(A_pad, Bu_shuf, Cu, as_pad, bsu_d, M_pad, M, N_half, K, stream);
        rdna4_gemm_fp8_trfeed_swiglu_f32(A_pad, Bf_shuf, Cf, as_pad, bsf_d, M_pad, M, N_fused, K, stream);
    }
    HCK(hipDeviceSynchronize());
    hipEvent_t s0, e0; hipEventCreate(&s0); hipEventCreate(&e0);
    hipEventRecord(s0);
    for (int i = 0; i < iters; i++) {
        rdna4_gemm_fp8_trfeed_f32(A_pad, Bg_shuf, Cg, as_pad, bsg_d, M_pad, M, N_half, K, stream);
        rdna4_gemm_fp8_trfeed_f32(A_pad, Bu_shuf, Cu, as_pad, bsu_d, M_pad, M, N_half, K, stream);
    }
    hipEventRecord(e0); hipEventSynchronize(e0);
    float ms_unfused = 0; hipEventElapsedTime(&ms_unfused, s0, e0); ms_unfused /= iters;

    hipEvent_t s1, e1; hipEventCreate(&s1); hipEventCreate(&e1);
    hipEventRecord(s1);
    for (int i = 0; i < iters; i++) {
        rdna4_gemm_fp8_trfeed_swiglu_f32(A_pad, Bf_shuf, Cf, as_pad, bsf_d, M_pad, M, N_fused, K, stream);
    }
    hipEventRecord(e1); hipEventSynchronize(e1);
    float ms_fused = 0; hipEventElapsedTime(&ms_fused, s1, e1); ms_fused /= iters;
    hipEventDestroy(s0); hipEventDestroy(e0); hipEventDestroy(s1); hipEventDestroy(e1);

    printf("    2x plain gemm(gate+up): %8.4f ms   fused swiglu gemm: %8.4f ms   (%+.1f%%)\n",
           ms_unfused, ms_fused, 100.0 * (ms_fused - ms_unfused) / ms_unfused);

    hipFree(A_pad); hipFree(as_pad);
    hipFree(Bg_d); hipFree(Bu_d); hipFree(Bf_d);
    hipFree(Bg_shuf); hipFree(Bu_shuf); hipFree(Bf_shuf);
    hipFree(bsg_d); hipFree(bsu_d); hipFree(bsf_d);
    hipFree(Cg); hipFree(Cu); hipFree(Cf);
    return pass ? 0 : 1;
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

    printf("\n=== rdna4_gemm_fp8_trfeed_swiglu_f32 (fused FFN gate/up/swiglu epilogue) bench ===\n");
    static const Shape swiglu_shapes[] = {
        {2048, 17408, 5120},   // the production Qwen3.8-27B ML8_4 prefill shape (task's measured trace)
        { 512, 17408, 5120},
        { 300,   256, 1024},   // small ragged shape (M_pad rounds 300 up to 384)
    };
    for (const auto & s : swiglu_shapes) {
        int rc = run_swiglu_shape(s.M, s.N, s.K, 30);
        if (rc != 0) failures++;
    }

    if (failures) {
        printf("\n%d shapes FAILED THE ORACLE GATE.\n", failures);
        return 1;
    }
    printf("\nall shapes PASS.\n");
    return 0;
}
