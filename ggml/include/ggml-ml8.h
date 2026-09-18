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
//   - n_groups_k (centroids ne1) must be >= K / QK_ML8 (may be larger under
//     tensor parallelism, where centroids is mirrored in full but w holds
//     only a K-slice — see lut_group_off below)
//   - centroids ne0 must equal 16
GGML_API struct ggml_tensor * ggml_ml8_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * x);

// Returns the node's lut_group_off (op_params[0]): the first centroid
// K-group this node reads, i.e. the effective LUT pointer is
// `(const uint8_t *) centroids->data + lut_group_off * 16`. Always 0 for a
// freshly-constructed node; the meta backend rewrites it per device for a
// K-split weight whose centroid LUT is mirrored in full.
GGML_API int32_t ggml_ml8_mul_mat_lut_group_off(const struct ggml_tensor * y);

// rotation_meta[3] kind_id values. The GGUF rotation_meta sidecar is
// I32[4] = [a_dim, b_dim, in_features, kind_id]; kind_id selects which of the
// two rotation kinds below `ggml_ml8_apply_rotation` is computing. Mirrored
// in scripts/calibration/kronecker_rotation.py
// (KRONECKER_ORTH_SYLVESTER_KIND_ID / BLOCK_HADAMARD_KIND_ID) and consumed by
// scripts/calibration/ml8_to_gguf.py::_rotation_meta_bytes — update all three
// together if a new kind is ever added.
#define GGML_ML8_ROTATION_KIND_KRONECKER_ORTH_SYLVESTER 1
#define GGML_ML8_ROTATION_KIND_BLOCK_HADAMARD            2

// Apply a rotation to the leading dim of `x`. Two kinds, selected by whether
// `h_a` is non-NULL:
//
//   h_a != NULL — kind_id 1, "kronecker_orth_sylvester": Q = H_a ⊗ H_b.
//     Math (matches scripts/calibration/kronecker_rotation.py::
//     KroneckerRotation.forward): reshape x along its leading dim from
//     d = a*b → (b, a), then per token compute Y = H_a^T @ X @ H_b, reshape
//     back to d. h_a is the GGUF rotation_h_a sidecar; a_dim is limited to
//     16 on the HIP path (fits a register array — see ml8.cu).
//
//   h_a == NULL — kind_id 2, "block_hadamard": Q = I_a ⊗ H_b. Independent
//     normalized Hadamard applied to each contiguous b_dim-sized block of
//     the leading dim, no cross-block mixing (equivalent to the kronecker
//     path with h_a == identity — see
//     scripts/calibration/kronecker_rotation.py::BlockHadamardRotation and
//     test_block_hadamard_matches_kronecker_identity). No a_dim limit; the
//     caller must still pass a_dim = x->ne[0] / b_dim.
//
// In both cases H_b is the Sylvester Hadamard of size b_dim, constructed
// internally (deterministic, no storage needed).
//
// Tensor shapes:
//   x   : [d, n_tokens]  GGML_TYPE_F32   (d == a_dim * b_dim; n_tokens may
//                                         span ne[1]..ne[3] for batched/MoE
//                                         inputs)
//   h_a : [a_dim, a_dim] GGML_TYPE_F32   from the GGUF rotation_h_a sidecar,
//                                        or NULL for block_hadamard
//
// Output:
//   y   : [d, n_tokens]  GGML_TYPE_F32
//
// Constraints:
//   - b_dim must be a positive power of 2
//   - a_dim * b_dim must equal x->ne[0]
//   - when h_a != NULL: h_a->ne[0] == h_a->ne[1] == a_dim
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

// Native ml8-4 row gather (token-embedding lookup). Gathers row `ids[i]` from the
// ml8-4 weight `w` [K, N] and dequantizes it via the per-K-group centroid LUT to
// K fp32 values — kept native 4-bit (no inline bf16 dequant of the table).
//
// Tensor shapes (ggml row-major):
//   w         : [K, N]            GGML_TYPE_ML8_4   (N = vocab rows)
//   centroids : [16, n_groups_k]  GGML_TYPE_F8_E4M3 per-K-group LUT
//   ids       : [n_rows, ...]     GGML_TYPE_I32
//
// Output:
//   y         : [K, n_rows, ...]  GGML_TYPE_F32     (matches ggml_get_rows layout)
//
// Constraints:
//   - K % QK_ML8 == 0
//   - centroids ne0 == 16, ne1 == K / QK_ML8
GGML_API struct ggml_tensor * ggml_ml8_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * ids);

// FP8_B128 phase 2 (see ggml.h GGML_OP_FP8_QUANT_ROT / GGML_OP_FP8_MUL_MAT
// doc comments and scripts/calibration/convert_fp8_rotated.py --format
// fp8_b128 for the full design).
//
// kind values for ggml_fp8_quant_rot's `kind` argument. Kinds 1/2 reuse the
// exact same rotation math as ggml_ml8_apply_rotation (see
// GGML_ML8_ROTATION_KIND_* above); kind 0 is new — a plain pass-through
// (no rotation) fused with the block-128 e4m3 quantize.
#define GGML_FP8_QUANT_ROT_KIND_NONE             0
#define GGML_FP8_QUANT_ROT_KIND_KRONECKER        1
#define GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD   2

// Fused activation rotate + block-G fp8 quantize. Computed once per input
// tensor and shared by every GEMM of the input group (see
// GGML_OP_FP8_QUANT_ROT in ggml.h for the exact packed row layout).
//
// `G` is the scale-group width: 128 for the FP8_B128 weight format, 32 for
// the ML8_FP8 weight format (34-byte blocks), or 0 for PER-ROW (one scale
// per whole row -- the frozen gfx1201 GEMM kernel's contract: A fp8 [M,K]
// contiguous with row stride exactly K, a_scale fp32 [M]). op_params[3] = G;
// it must be passed explicitly -- 0 means per-row, NOT "same as 128"
// (the old 0-aliases-128 behaviour is gone; every existing caller already
// passes 32 or 128 explicitly and is unaffected). See the 6-arg overload
// below for callers that want the historical G=128 default spelled out.
//
// Tensor shapes:
//   x   : [K, n1, n2, n3] GGML_TYPE_F32   (rows contiguous, nb[1] == K*4)
//   h_a : [a_dim, a_dim]  GGML_TYPE_F32   required iff kind == KRONECKER,
//                                         must be NULL otherwise
//
// Output:
//   y   : [K + 4*K/G, n1, n2, n3] GGML_TYPE_I8   (G == 32 or 128, grouped)
//   y   : [K + 4, n1, n2, n3]     GGML_TYPE_I8   (G == 0, per-row; NOT
//         row-contiguous -- see the GGML_OP_FP8_QUANT_ROT doc comment in
//         ggml.h for the exact "all A rows first, then all scales" byte
//         layout this mode uses instead)
//
// Constraints:
//   - G == 32 or 128: K % G == 0. G == 0 (per-row): no constraint on K.
//   - kind == GGML_FP8_QUANT_ROT_KIND_NONE:            h_a must be NULL
//   - kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER:        h_a required,
//     K == a_dim * b_dim, h_a->ne[0] == h_a->ne[1] == a_dim
//   - kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD:   h_a must be NULL,
//     K == a_dim * b_dim (a_dim == K / b_dim)
GGML_API struct ggml_tensor * ggml_fp8_quant_rot(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * h_a,
        int64_t a_dim,
        int64_t b_dim,
        int32_t kind,
        int32_t G);

// Back-compat 6-arg overload (C++ only) — existing callers keep compiling
// unchanged and get G=128 (the historical FP8_B128 behaviour).
#ifdef __cplusplus
static inline struct ggml_tensor * ggml_fp8_quant_rot(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * h_a,
        int64_t a_dim,
        int64_t b_dim,
        int32_t kind) {
    return ggml_fp8_quant_rot(ctx, x, h_a, a_dim, b_dim, kind, 128);
}
#endif

// Block-fp8 weight x packed-fp8 activation matmul. `a` must be the output of
// ggml_fp8_quant_rot (or bit-compatible with it). w may be GGML_TYPE_FP8_B128
// (G=128, 130-byte blocks) or GGML_TYPE_ML8_FP8 (G=32, 34-byte blocks); G is
// implied by w's type, and `a`'s packed row width must match EITHER the
// grouped layout (K + 4*K/G) OR the per-row layout (K + 4, `a` produced with
// G=0) -- the weight side dequantizes the same way regardless (the converter
// writes one scale per row replicated across the row's blocks for the
// per-row activation case, so per-block dequant is still exact).
//
// Tensor shapes:
//   w : [K, N]                             GGML_TYPE_FP8_B128 or GGML_TYPE_ML8_FP8
//   a : [K + 4*K/G, n1, n2, n3] (grouped)   GGML_TYPE_I8
//   a : [K + 4, n1, n2, n3]     (per-row)   GGML_TYPE_I8
//
// Output:
//   y : [N, n1, n2, n3]           GGML_TYPE_F32
//
// Constraints:
//   - K % G == 0 (G = 128 for FP8_B128, 32 for ML8_FP8)
//   - a->ne[0] == w->ne[0] + 4 * w->ne[0] / G, or a->ne[0] == w->ne[0] + 4 (per-row)
GGML_API struct ggml_tensor * ggml_fp8_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * a);

#ifdef __cplusplus
}  // extern "C"
#endif
