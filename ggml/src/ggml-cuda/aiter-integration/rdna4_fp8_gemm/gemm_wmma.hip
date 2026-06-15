#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "gemm_capi.h"
using namespace rocwmma;

// Naive: one wave (32 lanes) computes one 16x16 C tile, looping K in steps of 16.
// Correctness-first; occupancy tiling comes in P1. Epilogue scales via an LDS f32 scratch.
__global__ void gemm_fp8_naive(const float8_t* A, const float8_t* B, __hip_bfloat16* C,
                               const float* a_scale, const float* b_scale,
                               int M, int N, int K) {
    int ntiles_n = N / 16;
    int tm = (blockIdx.x / ntiles_n) * 16;   // C tile row origin
    int tn = (blockIdx.x % ntiles_n) * 16;   // C tile col origin
    fragment<accumulator, 16, 16, 16, float> acc;
    fragment<matrix_a, 16, 16, 16, float8_t, row_major> fa;   // A[M,K] row-major, ld=K
    fragment<matrix_b, 16, 16, 16, float8_t, row_major> fb;   // B[K,N] row-major, ld=N
    fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        load_matrix_sync(fa, A + tm * K + k, K);
        load_matrix_sync(fb, B + k * N + tn, N);
        mma_sync(acc, fa, fb, acc);
    }
    __shared__ float scratch[16 * 16];
    store_matrix_sync(scratch, acc, 16, mem_row_major);
    __syncthreads();
    // 32 lanes cooperatively scale + cast 256 elements -> bf16
    for (int idx = threadIdx.x; idx < 256; idx += 32) {
        int r = tm + idx / 16, c = tn + idx % 16;
        float v = scratch[idx] * a_scale[r] * b_scale[c];
        C[r * N + c] = (__hip_bfloat16)v;
    }
}

extern "C" void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                                       const float* a_scale, const float* b_scale,
                                       int M, int N, int K, hipStream_t stream) {
    dim3 grid((M / 16) * (N / 16));
    gemm_fp8_naive<<<grid, 32, 0, stream>>>(
        (const float8_t*)A, (const float8_t*)B, (__hip_bfloat16*)C, a_scale, b_scale, M, N, K);
}
