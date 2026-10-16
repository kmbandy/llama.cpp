#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// CONCAT(conv_state, TRANSPOSE(qkv)) feeding only an SSM_CONV: skip the CONCAT and let SSM_CONV read both sources
bool ggml_cuda_ssm_conv_try_elide_concat(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_ssm_conv_clear_elided();
