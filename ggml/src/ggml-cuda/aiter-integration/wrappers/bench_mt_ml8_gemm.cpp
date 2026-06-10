// bench_mt_ml8_gemm — isolated timing through the production mt_ml8_gemm
// wrapper (same AOT compile path + dispatch as llama.cpp's ml8 GEMMs).
//
// Usage: bench_mt_ml8_gemm M K N [iters]
// Prints median per-call ms. Used to attribute the in-graph (rocprof) vs
// JIT-µbench timing gap to the AOT binary vs run conditions (#185).
#include "mt_ml8_gemm.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s @%d\n", hipGetErrorString(e), __LINE__); std::exit(1); } } while (0)

int main(int argc, char ** argv) {
    const int32_t M = argc > 1 ? std::atoi(argv[1]) : 512;
    const int32_t K = argc > 2 ? std::atoi(argv[2]) : 2560;
    const int32_t N = argc > 3 ? std::atoi(argv[3]) : 9216;
    const int     iters = argc > 4 ? std::atoi(argv[4]) : 20;
    const int32_t group_size = 64, n_centroids = 16, n_groups_k = K / group_size;

    void *a, *b, *c, *as, *bs, *lut;
    CHECK(hipMalloc(&a,  (size_t) M * K));
    CHECK(hipMalloc(&b,  (size_t) K / 2 * N));
    CHECK(hipMalloc(&c,  (size_t) M * N * 2));
    CHECK(hipMalloc(&as, (size_t) M * 4));
    CHECK(hipMalloc(&bs, (size_t) n_groups_k * N * 4));
    CHECK(hipMalloc(&lut,(size_t) n_groups_k * n_centroids));
    CHECK(hipMemset(a, 0x38, (size_t) M * K));          // fp8 1.0
    CHECK(hipMemset(b, 0x10, (size_t) K / 2 * N));
    CHECK(hipMemset(lut, 0x38, (size_t) n_groups_k * n_centroids));
    std::vector<float> ones((size_t) n_groups_k * N, 1.0f);
    CHECK(hipMemcpy(bs, ones.data(), ones.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(as, ones.data(), (size_t) M * 4, hipMemcpyHostToDevice));

    mt_ml8_gemm_args_t args{};
    args.shape = { N, K, group_size, n_centroids, 1 };
    args.a_fp8 = a; args.b_packed = b; args.c = c;
    args.a_scale_fp32 = as; args.b_scale_fp32 = bs; args.centroid_lut_fp8 = lut;
    args.M = M;
    args.stride_am = K; args.stride_ak = 1;
    args.stride_bk = N; args.stride_bn = 1;
    args.stride_cm = N; args.stride_cn = 1;
    args.stride_ascale_m = 1;
    args.stride_bscale_k = N; args.stride_bscale_n = 1;
    args.stride_lut_k = n_centroids;

    hipStream_t stream; CHECK(hipStreamCreate(&stream));
    for (int i = 0; i < 3; i++) CHECK(mt_ml8_gemm(stream, &args));   // warmup + compile
    CHECK(hipStreamSynchronize(stream));

    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    std::vector<float> ms(iters);
    for (int i = 0; i < iters; i++) {
        CHECK(hipEventRecord(e0, stream));
        CHECK(mt_ml8_gemm(stream, &args));
        CHECK(hipEventRecord(e1, stream));
        CHECK(hipStreamSynchronize(stream));
        CHECK(hipEventElapsedTime(&ms[i], e0, e1));
    }
    std::sort(ms.begin(), ms.end());
    std::printf("M=%d K=%d N=%d  median %.0f us  (min %.0f, max %.0f)\n",
                M, K, N, ms[iters/2] * 1000.0f, ms.front() * 1000.0f, ms.back() * 1000.0f);
    return 0;
}
