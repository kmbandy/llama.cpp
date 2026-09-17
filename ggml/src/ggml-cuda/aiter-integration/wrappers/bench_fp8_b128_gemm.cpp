// bench_fp8_b128_gemm — times the FP8_B128 preshuffle GEMM
// (mt_fp8_b128_gemm) against the existing generic WF=0 ML8_FP8 GEMM
// (mt_ml8_gemm, weight_format=0) at the same (M, K, N) shapes, through the
// same production dispatch path each op uses (JIT compile + cache, then
// hipModuleLaunchKernel). FP8_B128 phase 2 design doc section 4(e).
//
// Usage: bench_fp8_b128_gemm [N] [K] [iters]
// Defaults: N=17408 K=5120, M swept over {16, 64, 512, 2048}.
#include "mt_fp8_b128_gemm.h"
#include "mt_ml8_gemm.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s @%d\n", hipGetErrorString(e), __LINE__); std::exit(1); } } while (0)

static double median_ms(hipStream_t stream, int iters, const std::function<void()> & call) {
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        CHECK(hipEventRecord(e0, stream));
        call();
        CHECK(hipEventRecord(e1, stream));
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipEventElapsedTime(&ms[i], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    CHECK(hipEventDestroy(e0)); CHECK(hipEventDestroy(e1));
    return ms[iters / 2];
}

static void bench_preshuffle(hipStream_t stream, int32_t M, int32_t K, int32_t N, int iters) {
    const int32_t n_groups = K / 128;
    const int32_t a_row    = K + K / 32;   // packed activation row (qs + fp32 scales)
    const int32_t tiles_n  = N / 128;

    void *a, *b, *c, *bs;
    CHECK(hipMalloc(&a,  (size_t) M * a_row));
    CHECK(hipMalloc(&b,  (size_t) N * K));                 // preshuffled, same byte count as [N,K]
    CHECK(hipMalloc(&c,  (size_t) M * N * sizeof(float)));
    CHECK(hipMalloc(&bs, (size_t) n_groups * tiles_n * sizeof(float)));
    CHECK(hipMemset(a, 0x38, (size_t) M * K));              // fp8 1.0 in the qs region
    std::vector<float> ones_scale((size_t) M * (n_groups), 1.0f);
    // Fill the embedded per-row scale region (byte offset K within each row) with 1.0f.
    for (int32_t m = 0; m < M; m++) {
        CHECK(hipMemcpy((char *) a + (size_t) m * a_row + K, ones_scale.data(),
                        (size_t) n_groups * sizeof(float), hipMemcpyHostToDevice));
    }
    std::vector<float> ones_bs((size_t) n_groups * tiles_n, 1.0f);
    CHECK(hipMemcpy(bs, ones_bs.data(), ones_bs.size() * sizeof(float), hipMemcpyHostToDevice));

    mt_fp8_b128_gemm_args_t args{};
    args.N = N; args.K = K; args.M = M;
    args.a_packed = a; args.b_preshuffled = b; args.b_scale = bs; args.c = c;
    args.stride_am = a_row; args.stride_ak = 1;
    args.stride_bn = K * 16; args.stride_bk = 1;
    args.stride_cm = N; args.stride_cn = 1;
    args.stride_ascale_m = a_row / 4; args.stride_ascale_k = 1;
    args.stride_bscale_k = tiles_n; args.stride_bscale_n = 1;

    for (int i = 0; i < 3; i++) CHECK(mt_fp8_b128_gemm(stream, &args));
    CHECK(hipStreamSynchronize(stream));
    const double ms = median_ms(stream, iters, [&]() { CHECK(mt_fp8_b128_gemm(stream, &args)); });
    std::printf("  preshuffle : M=%-5d K=%-6d N=%-6d  median %.1f us\n", M, K, N, ms * 1000.0);

    hipFree(a); hipFree(b); hipFree(c); hipFree(bs);
}

static void bench_generic_wf0(hipStream_t stream, int32_t M_req, int32_t K, int32_t N, int iters) {
    const int32_t group_size = 32;   // QK_ML8_FP8
    const int32_t n_groups_k = K / group_size;
    const mt_ml8_tuned_cfg cfg = ml8_pick_config(M_req, K, N);
    // Production dispatch (ggml_cuda_op_ml8_fp8_mul_mat) pads M to a multiple
    // of the tuned tier's BLOCK_SIZE_M before calling mt_ml8_gemm — replicate
    // that here so the comparison reflects the real per-call cost.
    const int32_t M = ((M_req + cfg.bm - 1) / cfg.bm) * cfg.bm;

    void *a, *b, *c, *as, *bs, *lut;
    CHECK(hipMalloc(&a,  (size_t) M * K));
    CHECK(hipMalloc(&b,  (size_t) K * N));                  // WF=0: raw e4m3 [K, N]
    CHECK(hipMalloc(&c,  (size_t) M * N * 2));               // bf16 output
    CHECK(hipMalloc(&as, (size_t) M * 4));
    CHECK(hipMalloc(&bs, (size_t) n_groups_k * N * 2));      // WF=0: fp16 scale table
    CHECK(hipMalloc(&lut,(size_t) n_groups_k * 16));
    CHECK(hipMemset(a, 0x38, (size_t) M * K));
    CHECK(hipMemset(b, 0x38, (size_t) K * N));
    CHECK(hipMemset(lut, 0x38, (size_t) n_groups_k * 16));
    std::vector<uint16_t> ones_fp16((size_t) n_groups_k * N, 0x3C00 /* fp16 1.0 */);
    CHECK(hipMemcpy(bs, ones_fp16.data(), ones_fp16.size() * 2, hipMemcpyHostToDevice));
    std::vector<float> ones_fp32((size_t) M, 1.0f);
    CHECK(hipMemcpy(as, ones_fp32.data(), (size_t) M * 4, hipMemcpyHostToDevice));

    mt_ml8_gemm_args_t args{};
    args.shape = { N, K, group_size, 16, /*weight_format=*/0 };
    args.a_fp8 = a; args.b_packed = b; args.c = c;
    args.a_scale_fp32 = as; args.b_scale_fp32 = bs; args.centroid_lut_fp8 = lut;
    args.M = M;
    args.stride_am = K; args.stride_ak = 1;
    args.stride_bk = N; args.stride_bn = 1;   // WF=0: B is [K, N]
    args.stride_cm = N; args.stride_cn = 1;
    args.stride_ascale_m = 1;
    args.stride_bscale_k = N; args.stride_bscale_n = 1;
    args.stride_lut_k = 0;

    for (int i = 0; i < 3; i++) CHECK(mt_ml8_gemm(stream, &args));
    CHECK(hipStreamSynchronize(stream));
    const double ms = median_ms(stream, iters, [&]() { CHECK(mt_ml8_gemm(stream, &args)); });
    std::printf("  generic WF0: M=%-5d(padded %-5d) K=%-6d N=%-6d  median %.1f us\n",
                M_req, M, K, N, ms * 1000.0);

    hipFree(a); hipFree(b); hipFree(c); hipFree(as); hipFree(bs); hipFree(lut);
}

int main(int argc, char ** argv) {
    const int32_t N     = argc > 1 ? std::atoi(argv[1]) : 17408;
    const int32_t K     = argc > 2 ? std::atoi(argv[2]) : 5120;
    const int     iters = argc > 3 ? std::atoi(argv[3]) : 20;

    hipStream_t stream; CHECK(hipStreamCreate(&stream));

    const int32_t Ms[] = { 16, 64, 512, 2048 };
    for (int32_t M : Ms) {
        std::printf("M=%d K=%d N=%d:\n", M, K, N);
        bench_preshuffle(stream, M, K, N, iters);
        bench_generic_wf0(stream, M, K, N, iters);
    }
    return 0;
}
