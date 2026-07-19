#include "common.cuh"

// Fused Sinkhorn normalization (DS4 hyper-connection).
// Replaces ~139 tiny ggml nodes per call with a single dispatch.
void ggml_cuda_op_sinkhorn_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
