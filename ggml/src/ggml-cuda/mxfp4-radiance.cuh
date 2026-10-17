#pragma once

// MXFP4 MUL_MAT / MUL_MAT_ID on rdna4 radiance packed weights (gfx1201 only).
// Weights are flagged with GGML_TENSOR_FLAG_MXFP4_RADIANCE (ggml.h); see
// gemm_mxfp4_radiance.h for the packed layout and kernel contract.

#include "common.cuh"

// gfx1201 is cc 0x1201 under the AMD offset (see ggml_cuda_parse_id).
#define GGML_CUDA_CC_GFX1201 (GGML_CUDA_CC_OFFSET_AMD + 0x1201)

// true only when the radiance MXFP4 kernels are compiled in and cc is gfx1201.
bool ggml_cuda_mxfp4_radiance_available(int cc);

// supports_op for a MUL_MAT / MUL_MAT_ID whose src0 carries GGML_TENSOR_FLAG_MXFP4_RADIANCE.
bool ggml_cuda_mxfp4_radiance_supports_op(const ggml_tensor * op, int cc);

// Dispatch targets. Both abort (never fall back) if radiance is unavailable on ctx.device.
void ggml_cuda_mul_mat_mxfp4_radiance(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                      const ggml_tensor * src1, ggml_tensor * dst);
void ggml_cuda_mul_mat_id_mxfp4_radiance(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
