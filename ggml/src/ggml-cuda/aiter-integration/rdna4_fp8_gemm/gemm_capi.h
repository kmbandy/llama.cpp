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
#ifdef __cplusplus
}
#endif
