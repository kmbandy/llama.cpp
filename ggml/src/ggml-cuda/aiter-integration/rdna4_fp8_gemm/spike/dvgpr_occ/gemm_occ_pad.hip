// gemm_occ_pad.hip  (dvgpr_occ P3-T5, Approach-2 cross-check -- UNSUPERVISED normal HIP launch)
//
// Validates Prong 1's occupancy->throughput finding on the REAL fp8 GEMM kernel
// (gemm_fp8_tiled, the raw-intrinsic verified core), through the normal HIP launch path
// (no PM4, no hang risk). We cannot raise occupancy above the kernel's natural ~8 with
// static VGPR -- that regime is exactly what dyn-VGPR unlocks (Prongs 1-2) -- but we CAN
// walk occupancy DOWN and read the slope: a steep TFLOPS-vs-occupancy slope at occ ~8
// corroborates that headroom exists above it.
//
// Occupancy knob: extra *dynamic* LDS in the launch config. The kernel ignores it, but it
// reserves LDS and forces fewer concurrent blocks/CU -> lower occupancy. hipOccupancy...
// reports the achieved blocks/CU per point. (Pure runtime; the kernel is reused unchanged.)
//
// Build (RAM-capped): see build comment at bottom. Run: ./gemm_occ_pad
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>
#include "../gemm_wmma_raw_intrinsic_verified.hip"   // gemm_fp8_tiled + BM/BN/BLOCK_THREADS + extern fn

#define HCK(x) do { hipError_t e=(x); if(e!=hipSuccess){ \
    fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } } while(0)

int main(int argc, char** argv) {
    const int M = 4096, N = 4096, K = 4096;   // compute-bound square (occupancy-relevant regime)
    const int iters = 30, warmup = 5;

    size_t aBytes = (size_t)M * K, bBytes = (size_t)K * N, cBytes = (size_t)M * N * 2; // bf16
    void *A, *B, *C; float *as, *bs;
    HCK(hipMalloc(&A, aBytes)); HCK(hipMalloc(&B, bBytes)); HCK(hipMalloc(&C, cBytes));
    HCK(hipMalloc(&as, M * sizeof(float))); HCK(hipMalloc(&bs, N * sizeof(float)));
    HCK(hipMemset(A, 0x38, aBytes));   // 0x38 = 1.0 in e4m3 -> real (non-NaN) WMMA work
    HCK(hipMemset(B, 0x38, bBytes));
    { std::vector<float> ones(N, 1.0f);
      HCK(hipMemcpy(as, ones.data(), M*sizeof(float), hipMemcpyHostToDevice));
      HCK(hipMemcpy(bs, ones.data(), N*sizeof(float), hipMemcpyHostToDevice)); }

    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    int dev = 0; hipDeviceProp_t prop; HCK(hipGetDeviceProperties(&prop, dev));
    double gflop = 2.0 * M * N * K / 1e12;   // TFLOP per GEMM

    printf("=== Approach-2 cross-check: real gemm_fp8_tiled TFLOPS vs occupancy (M=N=K=%d) ===\n", M);
    printf("  device: %s   (occupancy walked DOWN via extra dynamic LDS)\n", prop.name);
    printf("\n  extraLDS_KB  blocks/CU   TFLOPS\n");

    // LDS ladder: 0 = natural occupancy; increasing reserves push concurrent blocks down.
    const int ldsKB[] = {0, 4, 8, 12, 16, 20, 24, 28};
    for (int kb : ldsKB) {
        size_t dynLDS = (size_t)kb * 1024;
        int blocksPerCU = 0;
        hipError_t oe = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocksPerCU, (const void*)gemm_fp8_tiled, BLOCK_THREADS, dynLDS);
        if (oe != hipSuccess) blocksPerCU = -1;

        // warmup
        for (int i = 0; i < warmup; ++i)
            gemm_fp8_tiled<<<grid, BLOCK_THREADS, dynLDS, 0>>>(
                (const float8_t*)A, (const float8_t*)B, (__hip_bfloat16*)C, as, bs, M, N, K);
        HCK(hipDeviceSynchronize());

        hipEvent_t e0, e1; HCK(hipEventCreate(&e0)); HCK(hipEventCreate(&e1));
        HCK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i)
            gemm_fp8_tiled<<<grid, BLOCK_THREADS, dynLDS, 0>>>(
                (const float8_t*)A, (const float8_t*)B, (__hip_bfloat16*)C, as, bs, M, N, K);
        HCK(hipEventRecord(e1)); HCK(hipEventSynchronize(e1));
        float ms = 0; HCK(hipEventElapsedTime(&ms, e0, e1));
        double tf = gflop / (ms / 1e3 / iters);
        printf("  %9d    %7d    %7.1f\n", kb, blocksPerCU, tf);
        hipEventDestroy(e0); hipEventDestroy(e1);
    }
    printf("\n  Read the slope near the top (occ ~8): steep => headroom above 8 (dyn-VGPR's regime).\n");
    hipFree(A); hipFree(B); hipFree(C); hipFree(as); hipFree(bs);
    return 0;
}
// Build (RAM-capped, from this dir):
//   systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
//     /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 -I ../.. gemm_occ_pad.hip -o gemm_occ_pad
//   ./gemm_occ_pad
// Peak-vs-steady VGPR gap (the projected dyn-VGPR win magnitude):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1201 -O3 --save-temps -S -c gemm_occ_pad.hip -o /dev/null
//   grep -E '\.vgpr_count|\.vgpr_spill' gemm_occ_pad-hip-amdgcn-amd-amdhsa-gfx1201.s
