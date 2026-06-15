// Raw WMMA throughput microbench for gfx1201 (R9700).
// Back-to-back mma_sync with NACC independent accumulators (ILP), no memory in
// the hot loop -> measures the matrix-core instruction ceiling, isolated from
// the feeding/occupancy bottleneck that caps real GEMMs.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
using namespace rocwmma;

template <typename T, int NACC>
__global__ void wmma_peak(const T* A, const T* B, float* out, int iters) {
    fragment<matrix_a, 16, 16, 16, T, row_major> fa;
    fragment<matrix_b, 16, 16, 16, T, col_major> fb;
    fragment<accumulator, 16, 16, 16, float> acc[NACC];
    load_matrix_sync(fa, A, 16);
    load_matrix_sync(fb, B, 16);
#pragma unroll
    for (int i = 0; i < NACC; i++) fill_fragment(acc[i], 0.0f);
    for (int it = 0; it < iters; ++it) {
#pragma unroll
        for (int i = 0; i < NACC; i++) mma_sync(acc[i], fa, fb, acc[i]);
    }
#pragma unroll
    for (int i = 0; i < NACC; i++)
        store_matrix_sync(out + (blockIdx.x * NACC + i) * 256, acc[i], 16, mem_row_major);
}

template <typename T, int NACC>
double run(const char* name, int grid, int iters) {
    T *A, *B; float* out;
    hipMalloc(&A, 256 * sizeof(T));
    hipMalloc(&B, 256 * sizeof(T));
    hipMalloc(&out, (size_t)grid * NACC * 256 * sizeof(float));
    hipMemset(A, 1, 256 * sizeof(T));
    hipMemset(B, 1, 256 * sizeof(T));
    // warmup
    wmma_peak<T, NACC><<<grid, 32>>>(A, B, out, 256);
    hipDeviceSynchronize();
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    wmma_peak<T, NACC><<<grid, 32>>>(A, B, out, iters);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e);
    double mma = (double)grid * iters * NACC;          // one 16x16x16 mma per wave per inner step
    double flop = mma * 16.0 * 16.0 * 16.0 * 2.0;
    double tf = flop / (ms / 1e3) / 1e12;
    printf("  %-18s grid=%d NACC=%d iters=%d  %.2f ms  %.1f TFLOPS\n", name, grid, NACC, iters, ms, tf);
    hipFree(A); hipFree(B); hipFree(out);
    hipEventDestroy(s); hipEventDestroy(e);
    return tf;
}

int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("device: %s  CUs=%d\n", p.name, p.multiProcessorCount);
    int wgp = p.multiProcessorCount;             // RDNA: this is WGP count (2 CU each)
    printf("== fp8 peak vs grid fill (4 SIMD/WGP x 16 waves = %d resident wave slots) ==\n", wgp * 4 * 16);
    for (int mult : {1, 2, 4, 8}) {
        int grid = wgp * 2 * 16 * mult;
        run<float8_t, 8>("fp8 NACC8", grid, 30000);
    }
    printf("== f16 peak (for the 2x check) ==\n");
    run<float16_t, 8>("f16 NACC8", wgp * 4 * 16, 30000);
    run<float16_t, 16>("f16 NACC16", wgp * 4 * 16, 20000);
    return 0;
}
