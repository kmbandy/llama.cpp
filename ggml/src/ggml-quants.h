#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// TurboQuant KV cache compression (arXiv 2504.19874)
GGML_API void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
// MAD-301C Lever B: native head_dim-64 turbo4 (64-element block). CPU refs exist to
// satisfy ggml type-traits; the hot path is CUDA paged (mt_pagedattn). No-RHT, matching
// the paged scatter convention (dequant = centroid*norm).
GGML_API void quantize_row_turbo4_64_ref(const float * GGML_RESTRICT x, block_turbo4_64 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turbo4_64(const block_turbo4_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// TQ3_1S: WHT-rotated 3-bit weight quantization (8-level Lloyd-Max)
GGML_API void quantize_row_tq3_1s_ref(const float * GGML_RESTRICT x, block_tq3_1s * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq3_1s(const block_tq3_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_tq3_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// TQ4_1S: WHT-rotated 4-bit weight quantization (16-level Lloyd-Max)
GGML_API void quantize_row_tq4_1s_ref(const float * GGML_RESTRICT x, block_tq4_1s * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq4_1s(const block_tq4_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_tq4_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// MAD-214: turbo-FP8 BS=256 KV cache. Decode requires a per-(kv, layer)
// runtime centroid LUT (16 E4M3 bytes) that the standard quantize/dequantize
// signatures do not carry. These stubs exist only to satisfy the ggml_type
// type-trait table; KV cache fill/drain goes through a custom CUDA path
// (mt_pagedattn_turbo_fp8.cuh) that has access to the per-layer LUT.
GGML_API void quantize_row_turbo4_fp8_bs256_ref(const float * GGML_RESTRICT x, block_turbo4_fp8_bs256 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turbo4_fp8_bs256(const block_turbo4_fp8_bs256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// MAD-223 Phase G: ml8-4 weight quantization + fp8 e4m3 centroid storage.
// Phase G.1 ships stubs only — real implementations land in G.2 alongside the
// CPU dequant + vec_dot fallback. ml8_4 specifically cannot be implemented
// via the standard `(block*, float*, n)` signature because dequant requires
// the per-K-group centroid LUT sidecar; the matmul graph node carries it via
// op_params. These stubs exist solely to satisfy the ggml_type type-trait
// table and abort if anything else tries the standard API.
// See aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md §2.1.
GGML_API void quantize_row_ml8_4_ref(const float * GGML_RESTRICT x, block_ml8_4 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_ml8_4(const block_ml8_4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// G.2: extended-signature ml8 dequant that takes the per-K-group centroid LUT.
// The standard `dequantize_row_ml8_4` continues to abort since the standard
// signature has no slot for the LUT. The matmul graph node calls this one
// with the sidecar LUT pointer obtained via op_params (G.3).
//
//   x        : [n_groups_k] block_ml8_4 — one row's worth of indices + scales
//   lut_fp8  : [n_groups_k * 16] uint8 fp8 e4m3 — per-K-group centroid table
//   y        : [k] float output
//   k        : row length, must equal n_groups_k * QK_ML8
GGML_API void dequantize_row_ml8_4_with_lut(const block_ml8_4 * GGML_RESTRICT x,
                                            const uint8_t * GGML_RESTRICT lut_fp8,
                                            float * GGML_RESTRICT y,
                                            int64_t k);

GGML_API void quantize_row_f8_e4m3_ref(const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_f8_e4m3(const uint8_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// MAD Task 9: ml8-fp8 dequant. Each block: fp16 scale + 32 e4m3 bytes → 32 fp32 outputs.
// k must be divisible by QK_ML8_FP8 (32).
GGML_API void dequantize_row_ml8_fp8(const block_ml8_fp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// MAD Task 10: ml8-fp8 quantize (inverse of dequantize_row_ml8_fp8).
// scale = amax(|x|) / 448.0f; qs[i] = f32_to_e4m3fn(x[i]/scale).
// k must be divisible by QK_ML8_FP8 (32).
GGML_API void quantize_row_ml8_fp8_ref(const float * GGML_RESTRICT x, block_ml8_fp8 * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_ml8_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif
