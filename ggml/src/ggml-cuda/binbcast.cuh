#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);
void ggml_cuda_op_fused_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// dst = (a * b) + c, one pass, broadcast-indexing every operand.
// mul_first selects the operand order of the add: true => (a*b) + c, false => c + (a*b).
// All of a/b/c/dst must be F32; dst must be contiguous.
void ggml_cuda_op_fused_bcast_mul_add(ggml_backend_cuda_context & ctx,
                                      const ggml_tensor *         a,
                                      const ggml_tensor *         b,
                                      const ggml_tensor *         c,
                                      ggml_tensor *               dst,
                                      bool                        mul_first);
