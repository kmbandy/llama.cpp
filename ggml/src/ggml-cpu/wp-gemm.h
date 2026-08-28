#pragma once

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Multi-column q4_K x q8_K GEMM. Returns false if the shape is unsupported, in
// which case the caller must fall back to the per-element vec_dot path.
//
//   A       : nrc_x rows of block_q4_K, bx bytes apart
//   B       : nrc_y columns of block_q8_K, by bytes apart
//   C       : C[iy * stride_C + ix]
bool wp_gemm_q4K_q8K(int n, int nrc_x, int nrc_y,
                     const void * A, size_t bx,
                     const void * B, size_t by,
                     float * C, size_t stride_C);

// Multi-column q5_1 x q8_1 GEMM. Same contract as wp_gemm_q4K_q8K.
bool wp_gemm_q5_1_q8_1(int n, int nrc_x, int nrc_y,
                       const void * A, size_t bx,
                       const void * B, size_t by,
                       float * C, size_t stride_C);

// Runtime gate: WP_CPU_GEMM=1 enables the multi-column path.
bool wp_gemm_enabled(void);

#ifdef __cplusplus
}
#endif
