#pragma once
#include <hip/hip_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
// C = (A_fp8[M,K] @ B_fp8[K,N]) * a_scale[M] (per-row) * b_scale[N] (per-col), out bf16[M,N].
// A,B are float8_e4m3 (OCP). All device pointers. Row-major. M,N multiples of 16; K multiple of 16.
void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                            const float* a_scale, const float* b_scale,
                            int M, int N, int K, hipStream_t stream);

// B is ml8: packed 4-bit indices [K/2, N] uint8 (lo-nibble-first) + per-K-group fp8 centroid LUT
// [n_groups_k,16] + per-(group,N) fp32 scale [n_groups_k, N]. group_size = K / n_groups_k.
// A is plain fp8 [M,K] with per-row a_scale[M]. Out bf16[M,N]. C = dequant(B)·A scaled.
void rdna4_gemm_ml8_forward(const void* A, const void* B_idx, void* C,
                            const float* a_scale, const void* centroids_fp8,
                            const float* b_group_scale,
                            int M, int N, int K, int group_size, hipStream_t stream);
#ifdef __cplusplus
}
#endif
