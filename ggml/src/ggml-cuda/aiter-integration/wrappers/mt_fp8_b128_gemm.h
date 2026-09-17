// mt_fp8_b128_gemm.h
//
// Stable C API around the AITER preshuffle blockscale GEMM Triton kernel
// (kernels/gemm_ml8.py::_gemm_a8w8_blockscale_preshuffle_kernel), used by
// GGML_OP_FP8_MUL_MAT (FP8_B128 phase 2 — see the design doc section 4(b)).
//
// Unlike mt_ml8_gemm (the non-preshuffle WEIGHT_FORMAT=0/1 kernel), this
// kernel:
//   - reads B from the AITER shuffle_weight(layout=(16,16)) preshuffled
//     layout (see ml8.cu's fp8_b128_pack_weight_kernel for the exact byte
//     permutation);
//   - masks/wraps M internally (offs_am % M on load, offs_cm < M on store),
//     so the caller passes the TRUE M — no M padding;
//   - reads the activation's per-128-group scale from INSIDE the packed
//     activation row (byte offset K), not a separate buffer;
//   - reads the weight's per-(128-K-group, 128-N-tile) scale from a
//     standalone fp32 [K/128, N/128] table.
//
// Same runtime-JIT-via-aiter::Registry pattern as mt_ml8_gemm.h.
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

// Fixed by the design (radiance _ps_cfg / vllm-radiance recipe).
#define MT_FP8_B128_GROUP_K              128
#define MT_FP8_B128_GROUP_N              128
#define MT_FP8_B128_BLOCK_SIZE_K         128
#define MT_FP8_B128_GROUP_SIZE_M         8
#define MT_FP8_B128_NUM_WARPS            4
#define MT_FP8_B128_NUM_STAGES           1
#define MT_FP8_B128_WAVES_PER_EU         2
#define MT_FP8_B128_MATRIX_INSTR_NONKDIM 16
#define MT_FP8_B128_NUM_KSPLIT           1

struct mt_fp8_b128_tuned_cfg {
    int32_t bm;   // BLOCK_SIZE_M: 16 if M<=32 else 64 (radiance _ps_cfg)
    int32_t bn;   // BLOCK_SIZE_N: 128
};

// Diagnostic overrides for the prefill tier (M > 32) only — A/B sweeps, not config:
//   MT_FP8_BM, MT_FP8_BN (BLOCK_SIZE_M/N), MT_FP8_GSM (GROUP_SIZE_M),
//   MT_FP8_WPE (waves_per_eu), MT_ML8_NUM_WARPS / MT_ML8_NUM_STAGES (shared with ml8).
static inline int32_t mt_fp8_b128_env_int(const char * name, int32_t def) {
    const char * s = getenv(name);
    if (s == NULL) return def;
    const int v = atoi(s);
    return v > 0 ? v : def;
}

static inline mt_fp8_b128_tuned_cfg mt_fp8_b128_pick_config(int32_t M) {
    if (M <= 32) {
        return mt_fp8_b128_tuned_cfg{ 16, MT_FP8_B128_GROUP_N };
    }
    return mt_fp8_b128_tuned_cfg{ mt_fp8_b128_env_int("MT_FP8_BM", 64), mt_fp8_b128_env_int("MT_FP8_BN", MT_FP8_B128_GROUP_N) };
}

#ifdef __cplusplus
extern "C" {
#endif

// Argument bundle for mt_fp8_b128_gemm().
//
// Tensor layouts (all device memory):
//   a_packed        fp8_e4m3 bytes [M, K + K/32]  row-major — the packed
//                    activation row produced by GGML_OP_FP8_QUANT_ROT: first
//                    K bytes are e4m3 qs, then K/32 bytes (K/128 fp32
//                    scales) at byte offset K.
//   b_preshuffled   fp8_e4m3 bytes, AITER shuffle_weight(layout=(16,16))
//                    permutation of the on-disk [N, K] weight, viewed as
//                    [N/16, K*16].
//   b_scale         fp32 [K/128, N/128] row-major (kb outer, tile_n inner).
//   c               fp32 [M, N] row-major.
//
// M is the TRUE row count — the kernel masks/wraps internally, no padding.
struct mt_fp8_b128_gemm_args_t {
    int32_t     N, K, M;

    const void *a_packed;
    const void *b_preshuffled;
    const void *b_scale;
    void       *c;

    // Strides, in elements (Triton convention), NOT bytes.
    int32_t     stride_am, stride_ak;
    int32_t     stride_bn, stride_bk;
    int32_t     stride_cm, stride_cn;
    int32_t     stride_ascale_m, stride_ascale_k;
    int32_t     stride_bscale_k, stride_bscale_n;
};

// Launch the FP8_B128 preshuffle GEMM on the given stream. Returns hipSuccess
// on success, or the first non-success hipError_t. First call for a given
// (N, K, M-tier) JIT-compiles via Triton (cached to
// ${AITER_CACHE_DIR}/<key>/); subsequent calls reuse the cached handle.
hipError_t mt_fp8_b128_gemm(hipStream_t stream, const struct mt_fp8_b128_gemm_args_t *args);

// Reset the cached kernel handle (tests only; not thread-safe with
// concurrent mt_fp8_b128_gemm calls).
void mt_fp8_b128_gemm_reset_cache(void);

#ifdef __cplusplus
}  // extern "C"
#endif
