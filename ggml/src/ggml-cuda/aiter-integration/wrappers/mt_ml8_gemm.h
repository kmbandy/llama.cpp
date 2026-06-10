// mt_ml8_gemm.h
//
// Stable C API around the ml8-4 dense GEMM Triton kernel
// (kernels/gemm_ml8.py::_gemm_a8w8_blockscale_kernel with WEIGHT_FORMAT=1).
//
// The kernel is dispatched at runtime via `aiter::Registry::get_or_compile()`
// — same path used by mt_aiter_unified_attn.cpp. First call for a given
// (N, K, group_size, n_centroids) shape JIT-compiles via Triton (~2.5s,
// cached to ${AITER_CACHE_DIR:-${HOME}/.cache/llama.cpp/aiter}/<key>/);
// subsequent calls are direct hipModuleLaunchKernel invocations.
//
// Includable from C, C++, and .cu translation units. Pointers are passed as
// void*/int32_t*/float* and cast to hipDeviceptr_t at the call site so
// callers don't need HIP runtime headers in their public surface.
//
// Currently the wrapper supports ONE shape per process (cached on first
// call, asserted-equal subsequently). Multi-shape support is a future
// extension if a single process ever serves multiple ml8 models.
//
// Scope: dense path only. MoE counterpart (mt_ml8_moe_gemm) is a parallel
// follow-up wrapper; it shares the runtime registry but has a different
// arg bundle for expert-routing dispatch.
//
// MAD-223 Phase C.2.
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <stddef.h>

// ─────────────────────────────────────────────────────────────────────────
// Kernel tuning constants. Shape-independent values stay as #defines;
// shape-dependent block sizes are picked per (M, K, N) by
// ml8_pick_config() — populated from scripts/calibration/tune_gemm_ml8.py's
// sweep (gemm_ml8_tune.json, captured 2026-05-26 MAD-223 G.6.a).
// ─────────────────────────────────────────────────────────────────────────
#define MT_ML8_NUM_KSPLIT       1
#define MT_ML8_NUM_STAGES       1   // gfx1201 NUM_STAGES>=2 UAF per RDNA4 audit §2.2
#define MT_ML8_GROUP_N          1   // per-N b_scale (matches ml8 calibration's
                                    // scale_per_group: fp32[rows=N, n_groups_k])

// ml8.cu pads M to multiples of this. Must equal the smallest BLOCK_SIZE_M
// returned by ml8_pick_config across all tiers (decode tier = M=16 → BM=16).
#define MT_ML8_PAD_M           16

// MT_ML8_BLOCK_SIZE_N is the smallest BLOCK_SIZE_N our tuned configs use
// (BN=32 for down decode). N must be a multiple of this for the validation
// check in mt_ml8_gemm() to accept any layer's N. All tuned configs assume
// N is a multiple of their respective BLOCK_SIZE_N.
#define MT_ML8_BLOCK_SIZE_N    16

struct mt_ml8_tuned_cfg {
    int32_t bm;     // BLOCK_SIZE_M
    int32_t bn;     // BLOCK_SIZE_N
    int32_t gsm;    // GROUP_SIZE_M
    int32_t nw;     // num_warps
};

// Pick a tuned config for (M, K, N). Returns Phase-A defaults if shape is
// unknown. Tier split: M <= 16 = decode, M > 16 = prefill.
//
// Tuned values from gemm_ml8_tune.json (2026-05-26 sweep, MAD-223 G.6.a).
// Validated speedups vs old hardcoded (BM=16, BN=16, GSM=1, NW=4):
//   gate/up M=512: 0.860 ms → 0.370 ms (2.3x)
//   down    M=512: similar improvement expected (sweep best 0.389 ms)
static inline mt_ml8_tuned_cfg ml8_pick_config(int32_t M, int32_t K, int32_t N) {
    const bool prefill = (M > 16);
    // Qwen3.5-4B gate/up: K=2560, N=9216
    if (K == 2560 && N == 9216) {
        return prefill
            ? mt_ml8_tuned_cfg{128, 64, 4, 4}
            : mt_ml8_tuned_cfg{ 16, 64, 1, 4};
    }
    // Qwen3.5-4B down: K=9216, N=2560
    if (K == 9216 && N == 2560) {
        return prefill
            ? mt_ml8_tuned_cfg{128, 64, 1, 4}
            : mt_ml8_tuned_cfg{ 16, 32, 4, 4};
    }
    // Generic shapes (#185): the explicit G.6.a winners above are both
    // BM=128 / BN=64 at prefill, and the BM=16/BN=16 Phase-A defaults are
    // ~5x off at M=512 (measured: qkv 2560x8192 2.34ms vs 0.42ms-class).
    // Apply the winning tile shape whenever divisibility allows, largest
    // BM/BN first. ml8.cu pads M to a multiple of cfg.bm, so any BM here
    // is legal; BN must divide N exactly.
    const int32_t bm = prefill ? 128 : 16;
    for (int32_t bn = 64; bn >= 16; bn >>= 1) {
        if (N % bn == 0) {
            return mt_ml8_tuned_cfg{bm, bn, prefill ? 4 : 1, 4};
        }
    }
    // N not a multiple of 16 — Phase-A fallback.
    return mt_ml8_tuned_cfg{16, 16, 1, 4};
}

// Model-shape parameters — set by the caller per (calibrated) Linear layer.
// The wrapper builds the Triton signature from these at first call.
// BLOCK_SIZE_K is the kernel constraint == group_size (per kernel docstring:
// "For this kernel implementation, GROUP_K must equal BLOCK_K").
struct mt_ml8_gemm_shape_t {
    int32_t N;             // out_features (calibration's "rows")
    int32_t K;             // in_features  (calibration's "in_features")
    int32_t group_size;    // typically 64 for ml8-4 (Cell C recipe), 32 for ml8-fp8
    int32_t n_centroids;   // 16 for ml8-4 (ignored when weight_format == 0)
    // WEIGHT_FORMAT switch into _gemm_a8w8_blockscale_kernel:
    //   1 = ml8-4 LUT path: B is packed 4-bit nibbles [K/2, N] indexing a
    //       per-K-group fp8 centroid LUT (b_packed + centroid_lut_fp8).
    //   0 = scaled-fp8 (ml8-fp8) baseline: B is raw e4m3 bytes [K, N] fed
    //       straight to tl.dot; the centroid LUT branch is dead-code-eliminated.
    //       centroid_lut_fp8 must still point at a non-null device buffer (it is
    //       never dereferenced) and stride_lut_k must be 0.
    // Callers MUST set this explicitly (it is part of the kernel cache key):
    // 1 for ml8-4 weights, 0 for ml8-fp8 weights.
    int32_t weight_format;
};

#ifdef __cplusplus
extern "C" {
#endif

// Argument bundle for mt_ml8_gemm().
//
// Tensor layouts (all device memory):
//   a_fp8            fp8_e4m3  [M, K]              row-major
//   b_packed         uint8     [K/2, N]            row-major (kernel layout;
//                                                  load_ml8_layer transposes
//                                                  from on-disk [N, K/2])
//   c                bf16      [M, N]              row-major
//   a_scale_fp32     fp32      [M]                 per-row activation scale
//   b_scale_fp32     fp32      [n_groups_k, N]     per-(K-group, N), n_groups_k = K/group_size
//   centroid_lut_fp8 fp8_e4m3  [n_groups_k, 16]    per-K-group LUT
//
// Reconstruction formula the kernel computes:
//   W[k, n]   = centroid_lut[k // group_size][unpack_4bit(b_packed[k/2, n], lo_first)] * b_scale[k // group_size, n]
//   out[m, n] = (sum_k(a_fp8[m, k] * W[k, n])) * a_scale[m]
//
// Constraints (Phase C.2 v1; padding/masking is Phase C.3 work):
//   - M       % MT_ML8_BLOCK_SIZE_M  == 0
//   - shape.N % MT_ML8_BLOCK_SIZE_N  == 0
//   - shape.K % shape.group_size     == 0
//   - shape.N must equal calibrated layer's out_features
//   - shape.K must equal calibrated layer's in_features
struct mt_ml8_gemm_args_t {
    // Shape — must be the same across all calls in a process (kernel handle
    // cached after first call; subsequent calls with a different shape error).
    struct mt_ml8_gemm_shape_t shape;
    // I/O
    const void *a_fp8;
    const void *b_packed;
    void       *c;
    // Scales
    const void *a_scale_fp32;
    const void *b_scale_fp32;
    // LUT
    const void *centroid_lut_fp8;
    // Runtime dim
    int32_t     M;
    // Strides (in elements, NOT bytes; matches Triton's stride convention)
    int32_t     stride_am, stride_ak;
    int32_t     stride_bk, stride_bn;     // for [K/2, N] packed-byte layout
    int32_t     stride_cm, stride_cn;
    int32_t     stride_ascale_m;
    int32_t     stride_bscale_k, stride_bscale_n;
    int32_t     stride_lut_k;
};

// Launch the ml8-4 dense GEMM kernel on the given stream.
// Returns hipSuccess on success, or the first non-success hipError_t.
// First call for a given shape JIT-compiles via Triton (~2.5s, cached);
// subsequent calls reuse the cached kernel handle.
hipError_t mt_ml8_gemm(hipStream_t stream, const struct mt_ml8_gemm_args_t *args);

// Reset the cached kernel handle. Useful for tests that exercise multiple
// shapes in one process. Not thread-safe with concurrent mt_ml8_gemm calls.
void mt_ml8_gemm_reset_cache(void);

#ifdef __cplusplus
}  // extern "C"
#endif
