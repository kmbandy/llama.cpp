// ggml-ml8.h
//
// MAD-223 Phase G.3: ml8-4 matmul graph node (CPU-resident via GGML_OP_CUSTOM).
//
// `ggml_ml8_mul_mat` constructs a CUSTOM op that performs a quantized matmul
// `y = w @ x.T` where `w` is `GGML_TYPE_ML8_4` and `centroids` is its per-K-group
// fp8 LUT sidecar. The CPU compute callback dequantizes `w` block-by-block via
// `dequantize_row_ml8_4_with_lut` and then runs a standard fp32 dot product
// against `x`.
//
// Rotation + AWQ are NOT inside this op — those are constructed as separate
// ggml nodes (element-wise multiply + small matmuls) by the model graph builder
// (G.3b). Keeping this op narrow lets us test the matmul in isolation.
//
// Backend support:
//   - CPU: dispatched via GGML_OP_CUSTOM compute callback (in ml8.c).
//   - HIP: NOT yet — G.4 will replace this with a typed `GGML_OP_ML8_MUL_MAT`
//          (or extend `GGML_OP_MUL_MAT`) and call into `mt_ml8_gemm`.
//
// See aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md §2.
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Construct a graph node computing y = w @ x.T using the ml8-4 quantized
// weight `w` and its per-K-group centroid LUT `centroids`.
//
// Tensor shapes (ggml row-major convention):
//   w         : [K, N]      GGML_TYPE_ML8_4   (K = product of K-groups; N = out features)
//   centroids : [16, n_groups_k]  GGML_TYPE_F8_E4M3 sidecar LUT
//   x         : [K, M]      GGML_TYPE_F32     activations
//
// Output:
//   y         : [N, M]      GGML_TYPE_F32     (matches plain ggml_mul_mat layout)
//
// Constraints:
//   - K must be a multiple of QK_ML8 (64)
//   - n_groups_k (centroids ne1) must equal K / QK_ML8
//   - centroids ne0 must equal 16
GGML_API struct ggml_tensor * ggml_ml8_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * x);

// Apply a Kronecker rotation Q = H_a ⊗ H_b to the leading dim of `x`.
//
// Math (matches scripts/calibration/kronecker_rotation.py::KroneckerRotation.forward):
//   Reshape x along its leading dim from d = a*b → (b, a), then per token compute
//   Y = H_a^T @ X @ H_b, reshape back to d. H_b is the Sylvester Hadamard of size
//   b_dim, constructed internally (deterministic, no storage needed).
//
// Tensor shapes:
//   x   : [d, n_tokens]  GGML_TYPE_F32   (d == a_dim * b_dim)
//   h_a : [a_dim, a_dim] GGML_TYPE_F32   from the GGUF rotation_h_a sidecar
//
// Output:
//   y   : [d, n_tokens]  GGML_TYPE_F32
//
// Constraints:
//   - b_dim must be a positive power of 2
//   - a_dim * b_dim must equal x->ne[0]
//   - h_a->ne[0] == h_a->ne[1] == a_dim
//
// CPU-only for now (G.4 will lift this to GPU). When `h_a` is NULL, returns `x`
// unchanged so callers can plumb the optional rotation uniformly.
GGML_API struct ggml_tensor * ggml_ml8_apply_rotation(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * h_a,
        int64_t a_dim,
        int64_t b_dim);

// MAD-223 G.7 — ml8-4 MoE matmul. Mirrors ggml_mul_mat_id's shape contract
// but with per-expert ml8-4 weight stacks and per-expert centroid LUTs.
//
// Tensor shapes (ggml row-major):
//   w         : [K, N, n_experts]            GGML_TYPE_ML8_4   per-expert weights
//   centroids : [16, n_groups_k, n_experts]  GGML_TYPE_F8_E4M3 per-expert LUT stack
//   x         : [K, n_expert_used, n_tokens] GGML_TYPE_F32     gathered activations
//   ids       : [n_expert_used, n_tokens]    GGML_TYPE_I32     expert routing
//
// Output:
//   y         : [N, n_expert_used, n_tokens] GGML_TYPE_F32
//
// Constraints:
//   - K % QK_ML8 == 0
//   - n_groups_k (centroids ne0=16, ne1=n_groups_k) must equal K / QK_ML8
//   - ids ne0 == x ne1, ids ne1 == x ne2
//   - w->ne[2] == centroids->ne[2] (same n_experts)
//
// Rotation + AWQ are NOT part of this op — the graph builder is expected to
// apply them on `x` upstream (same pattern as the dense path).
GGML_API struct ggml_tensor * ggml_ml8_mul_mat_id(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * x,
        struct ggml_tensor  * ids);

#ifdef __cplusplus
}  // extern "C"
#endif
