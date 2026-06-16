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

// Low-occupancy-allowed twin: __launch_bounds__(32,1) lets the compiler use up to the
// 256-VGPR static cap (occupancy target = 1 block/CU) so high-NACC variants allocate the
// accumulators in registers instead of spilling -> clean sub-9-wave occupancy points.
template <typename T, int NACC>
__global__ void __launch_bounds__(32, 1)
wmma_peak_lo(const T* A, const T* B, float* out, int iters) {
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
double run_lo(const char* name, int grid, int iters) {
    T *A, *B; float* out;
    hipMalloc(&A, 256 * sizeof(T)); hipMalloc(&B, 256 * sizeof(T));
    hipMalloc(&out, (size_t)grid * NACC * 256 * sizeof(float));
    hipMemset(A, 1, 256 * sizeof(T)); hipMemset(B, 1, 256 * sizeof(T));
    wmma_peak_lo<T, NACC><<<grid, 32>>>(A, B, out, 256);
    hipDeviceSynchronize();
    hipEvent_t s, e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    wmma_peak_lo<T, NACC><<<grid, 32>>>(A, B, out, iters);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms = 0; hipEventElapsedTime(&ms, s, e);
    double mma = (double)grid * iters * NACC;
    double tf = mma * 16.0 * 16.0 * 16.0 * 2.0 / (ms / 1e3) / 1e12;
    printf("  %-18s grid=%d NACC=%d iters=%d  %.2f ms  %.1f TFLOPS\n", name, grid, NACC, iters, ms, tf);
    hipFree(A); hipFree(B); hipFree(out); hipEventDestroy(s); hipEventDestroy(e);
    return tf;
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

    // == occupancy-sensitivity of the PURE-WMMA ceiling (MAD-305 Phase-3 proxy) ==
    // Does the issue-bound matrix regime need full occupancy? Raise NACC: acc-VGPR grows
    // (NACC*8 + ~5) -> fewer resident waves/SIMD (16 -> ~6 over NACC 8->28), while ILP only
    // grows (>= neutral). So any TF drop here is attributable to the occupancy fall, isolated
    // from feed (there is no memory in the hot loop). Contrast vs the real acc-tile GEMM, which
    // fell 131->121 TF over the SAME 16->6-wave range: if pure-WMMA HOLDS ~307, the GEMM's
    // occupancy sensitivity is about hiding FEED latency (fat-tile-with-less-feed may tolerate
    // low occ -> dyn-VGPR viable); if pure-WMMA also DROPS, occupancy binds the issue itself
    // (fat tiles are a dead end regardless of feed).
    int sgrid = wgp * 2 * 16 * 8;                 // fixed saturating grid; only NACC (VGPR) varies
    printf("== fp8 PURE-WMMA occupancy sweep @ grid=%d (NACC up -> acc-VGPR up -> occupancy down) ==\n", sgrid);
    run<float8_t, 8 >("fp8 NACC8 ", sgrid, 20000);   // ~69  VGPR -> 16 waves/SIMD (full)
    run<float8_t, 12>("fp8 NACC12", sgrid, 20000);   // ~101 VGPR -> ~13 waves
    run<float8_t, 16>("fp8 NACC16", sgrid, 20000);   // ~133 VGPR -> ~11 waves
    run<float8_t, 20>("fp8 NACC20", sgrid, 20000);   // ~165 VGPR -> ~9  waves
    run<float8_t, 24>("fp8 NACC24", sgrid, 20000);   // ~197 VGPR -> SPILLS at default occ target
    run<float8_t, 28>("fp8 NACC28", sgrid, 20000);   // ~229 VGPR -> SPILLS at default occ target
    // sub-9-wave, NO-SPILL (launch_bounds(32,1) raises the VGPR ceiling to 256) -> overlaps the
    // real acc-tile GEMM's 9->6-wave range. If these hold ~307, WMMA-issue is occupancy-insensitive
    // even at GEMM-like occupancy, so the GEMM's occupancy sensitivity is FEED-latency, not issue.
    printf("== fp8 PURE-WMMA sub-9-wave (launch_bounds(32,1), no spill) ==\n");
    run_lo<float8_t, 20>("fp8lo NACC20", sgrid, 20000);  // ~165 VGPR -> ~9 waves (cross-check vs run<>)
    run_lo<float8_t, 24>("fp8lo NACC24", sgrid, 20000);  // ~197 VGPR -> ~7 waves
    run_lo<float8_t, 28>("fp8lo NACC28", sgrid, 20000);  // ~229 VGPR -> ~6 waves (== acc[6][4] occ)
    run_lo<float8_t, 30>("fp8lo NACC30", sgrid, 15000);  // ~245 VGPR -> ~6 waves (near static cap)
    return 0;
}
