#include "common.cuh"

// GGML_OP_WAIT_ADD: dst = src0 + <host-pinned slot>, optionally spinning on a
// host-published epoch flag first (HIP only; supports_op is false elsewhere).
void ggml_cuda_op_wait_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
