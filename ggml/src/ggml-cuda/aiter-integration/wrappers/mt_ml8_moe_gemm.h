// mt_ml8_moe_gemm.h
//
// Stable C API around the ml8-4 MoE GEMM Triton kernel
// (kernels/moe_op_gemm_ml8.py::_moe_gemm_a8w8_blockscale with WEIGHT_FORMAT=1).
//
// Sibling of mt_ml8_gemm.{h,cpp} (the dense path). Same dispatch model:
// kernel handles are JIT-compiled at first call via `aiter::Registry::
// get_or_compile()`, then cached on disk under
// `${AITER_CACHE_DIR:-${HOME}/.cache/llama.cpp/aiter}/<key>/`. Different
// (shape) tuples produce different compiled kernels — one per process per
// distinct shape (asserted-equal on subsequent calls).
//
// Kernel coverage (full feature surface):
//   - Per-expert weight + LUT base (stride_w_e, stride_lut_expert)
//   - Per-row activation blockscale (PER_ROW_X_SCALE constexpr)
//   - Per-(K-group, N) weight blockscale
//   - Optional fused SwiGLU activation (APPLY_SWIGLU constexpr)
//   - Optional residual add (ADD_RESIDUAL constexpr)
//   - Optional gather-indexed activation read (GatherIndx, nullable)
//   - Optional bias / gammas / static scales (nullable runtime ptrs)
//   - Padding-aware experts via ExptOffsSum (nullable)
//
// MAD-223 Phase C.3 / MAD-244.
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <stddef.h>

// ─────────────────────────────────────────────────────────────────────────
// Default tile config. Matches the values used in
// `tests/test_ml8_kernel_moe.py` (validated Phase B.5, max_err = 0).
// Phase F will sweep these per-shape and emit AOT specializations.
// ─────────────────────────────────────────────────────────────────────────
#define MT_ML8_MOE_BLOCK_M       16
#define MT_ML8_MOE_BLOCK_N       16
#define MT_ML8_MOE_GROUP_M        1
#define MT_ML8_MOE_XCD_SWIZZLE    1
#define MT_ML8_MOE_SPLIT_K        1
#define MT_ML8_MOE_NUM_STAGES     1   // gfx1201 NUM_STAGES>=2 UAF (RDNA4 audit §2.2)

// Model-shape parameters. The wrapper builds the Triton signature from these
// at first call. BLOCKSCALE_K is the kernel constraint == group_size (per
// kernel docstring: "BLOCKSCALE_K must equal BLOCK_K"; we set BLOCK_K =
// group_size to satisfy this).
struct mt_ml8_moe_gemm_shape_t {
    int32_t N;                       // out_features per expert
    int32_t K;                       // in_features per expert
    int32_t group_size;              // ml8-4 = 64
    int32_t n_centroids;             // ml8-4 = 16
    int32_t n_experts;               // total experts in the layer
    int32_t n_expts_act;             // experts activated per token (top-k routing)
    // Fused-op constexprs (one constexpr → one compiled kernel; changing these
    // forces a recompile)
    int32_t apply_swiglu;            // 0 / 1
    int32_t activation_reduction_n;  // 1 (no reduction) or 2 (SwiGLU halves N)
    int32_t add_residual;            // 0 / 1
    // Per-row vs 2D activation blockscale
    int32_t per_row_x_scale;         // 0 = 2D, 1 = per-row
    // Workspace-tuning constexprs (rarely changed)
    int32_t even_k;                  // 1 if K % BLOCK_K == 0 (the only supported case in v1)
    int32_t mask_k_limit;            // typically == K
    int32_t upcast_indices;          // 0 (= use int32 indices)
    // LOCAL PATCH #6 "feature-present" constexpr flags (MAD-244). Each gates
    // a runtime `is None` check in the kernel that AOT-mode cannot encode
    // via signature alone. Setting a flag to 1 means: the corresponding
    // pointer in mt_ml8_moe_gemm_args_t will be a real device pointer that
    // the kernel will load + use. Setting it to 0 means: the pointer should
    // be NULL; the kernel skips that feature entirely (dead-code elimination
    // at AOT compile). Changing any of these triggers a recompile.
    int32_t has_bias;                // 0 / 1
    int32_t has_gammas;              // 0 / 1
    int32_t has_x_static_scale;      // 0 / 1
    int32_t has_w_static_scale;      // 0 / 1
    int32_t has_quant_static_scale;  // 0 / 1
};

#ifdef __cplusplus
extern "C" {
#endif

// Argument bundle for mt_ml8_moe_gemm().
//
// Tensor layouts (all device memory):
//   y                    bf16     [M, N']  where N' = N / activation_reduction_n
//   x_fp8                fp8_e4m3 [M, K]
//   w_packed             uint8    [n_experts, K/2, N]
//   x_scale_fp32         fp32     per-row [M] when PER_ROW_X_SCALE=1
//                                  2D     [M_blocks, K_blocks] when PER_ROW_X_SCALE=0
//   w_scale_fp32         fp32     [n_experts, n_groups_k, N]
//   centroid_lut_fp8     fp8_e4m3 [n_experts, n_groups_k, n_centroids]
//   bias                 bf16     [n_experts, N]  (nullable)
//   gammas               fp32     [M]             (nullable; per-row gather scale)
//   x_static_scale       fp32     scalar          (nullable; only when using static-fp8)
//   w_static_scale       fp32     scalar          (nullable)
//   quant_static_scale   fp32     scalar          (nullable)
//   gather_indx          int32    [M]             (nullable; identity routing if NULL)
//   expt_hist            int32    [n_experts]     histogram of tokens per expert
//   expt_offs            int32    [n_experts]     prefix-sum offsets
//   expt_offs_sum        int32    scalar          (nullable; padding-aware sum)
//   expt_data            int32    [grid_m]        packed (block_id << 16) | expt_id
struct mt_ml8_moe_gemm_args_t {
    struct mt_ml8_moe_gemm_shape_t shape;
    // I/O
    void       *y;
    const void *x_fp8;
    const void *w_packed;
    // Scales
    const void *x_scale_fp32;
    const void *w_scale_fp32;
    // LUT
    const void *centroid_lut_fp8;
    // Optional fused-op runtime pointers (set to NULL for "absent")
    const void *bias;
    const void *gammas;
    const void *x_static_scale;
    const void *w_static_scale;
    const void *quant_static_scale;
    // SwiGLU runtime params (only used when apply_swiglu=1; set to 0 otherwise)
    float       alpha;
    float       limit;
    // Routing tensors
    const void *gather_indx;         // NULL = identity (each row routed to its block's expert)
    const void *expt_hist;
    const void *expt_offs;
    const void *expt_offs_sum;       // NULL = no padding-aware optimization
    const void *expt_data;
    // Runtime dims
    int32_t     M;
    int32_t     grid_m;              // = M / BLOCK_M (caller computes)
    int32_t     grid_n;              // = N / BLOCK_N (caller computes)
    // Strides (in elements, NOT bytes)
    int32_t     stride_y_k, stride_y_m, stride_y_n;
    int32_t     stride_x_m, stride_x_k;
    int32_t     stride_x_bs_m, stride_x_bs_k;
    int32_t     stride_w_e, stride_w_k, stride_w_n;
    int32_t     stride_w_bs_e, stride_w_bs_k, stride_w_bs_n;
    int32_t     stride_b_e;
    int32_t     stride_lut_expert, stride_lut_k;
};

// Launch the ml8-4 MoE GEMM kernel on the given stream. Returns hipSuccess
// on success, or the first non-success hipError_t. First call for a given
// shape JIT-compiles via Triton (~2.5s, cached); subsequent calls reuse the
// cached kernel handle.
hipError_t mt_ml8_moe_gemm(hipStream_t stream, const struct mt_ml8_moe_gemm_args_t *args);

// Reset the cached kernel handle. Not thread-safe with concurrent calls.
void mt_ml8_moe_gemm_reset_cache(void);

#ifdef __cplusplus
}  // extern "C"
#endif
