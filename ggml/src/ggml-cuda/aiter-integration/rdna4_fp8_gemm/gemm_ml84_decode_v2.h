// gemm_ml84_decode_v2.h — MAD-305 ML8_4 (4.5 bpw) decode-band GEMM v2.
//
// Radiance-style decode kernel for real M<=32 (typically 1..8, K7 DFlash
// verify) replacing gemm_ml84_trfeed_splitk<32,1,*>'s fixed 32-row tile.
// Ported from ~/GitHub/ggz14-vllm-mxfp4/radiance_mxfp4_fp8.hip's
// radiance_mxfp4_fp8_gemm_decode STRUCTURE (tile shape, split-K fill law,
// fused last-arriver reduction, persistent scratch) onto ml8-4's own
// nibble/LUT/per-group-scale numerics (unchanged from
// bench/trfeed_ml84_kernels.h — see gemm_ml84_decode_v2.hip for exactly
// which lines were matched). See gemm_ml84_decode_v2.hip for the full
// design writeup, build/bench commands and every numerics assumption.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Decode/verify GEMM, v2. Same B_nib / lut / a_scale / b_scale_g contracts
// as rdna4_gemm_ml84_trfeed_decode_splitk (gemm_capi.h — B_nib: ML84_TRFEED
// packed nibbles; lut: F8_E4M3 [K/64][16]; b_scale_g: fp32 [K/64][N];
// a_scale: fp32[M_pad]; A: fp8 e4m3 [M_pad,K] row-major stride K).
//
// UNLIKE the old launcher, this one needs the REAL row count `M` (not just
// the 32-padded M_pad) to size its DTM tile — computing 32 rows of WMMA
// work when only M=1..8 are real is exactly the waste this kernel removes.
// `M` is added as a new LEADING argument; every other argument (name,
// order, type) is copied verbatim from the rdna4_gemm_ml84_trfeed_decode_
// splitk call site at ml8.cu:3338-3372, so the wiring agent's substitution
// is: add the already-in-scope `M` local as the first argument, forward
// the rest unchanged, and change the call to treat a `false` return as
// "fall back to rdna4_gemm_ml84_trfeed_decode_splitk" (this fn has NO
// hipError_t return — errors after a supported-shape launch are async HIP
// errors exactly like the old launcher's hipGetLastError()-returning
// contract, so the wiring agent should still hipGetLastError()/GGML_ASSERT
// after a `true` return, same as today).
//
// Returns false (no kernel launched, no allocation, no side effects) if
// the shape is unsupported and the caller must fall back to
// rdna4_gemm_ml84_trfeed_decode_splitk:
//   - M <= 0 or M > 32
//   - M_pad < M, M_pad <= 0
//   - N <= 0, N % 16 != 0, or N exceeds the persistent scratch's compiled
//     ceiling (ml84_decode_v2_max_n(), currently 248320 — the largest
//     production shape, the output head)
//   - K <= 0 or K % 64 != 0 (QK_ML8 group width — same floor as the old
//     kernel; K % 128 selects the DBK=64 fallback internally, still
//     supported, just not the default 128-wide slab)
// `n_splits`: 0 (or negative) lets the launcher pick via its own fill-law
// heuristic (mirrors rdna4_ml84_trfeed_splitk_default_splits's role);
// >0 forces that split count, clamped to {1,2,4}. MT_ML8_4_DECODE_V2_KS
// overrides both (env > explicit arg > heuristic).
bool rdna4_gemm_ml84_decode_v2(int M, const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                float* C_f32, const float* a_scale, const float* b_scale_g,
                                int M_pad, int N, int K, int n_splits, hipStream_t stream);

// Compiled ceiling for N that the persistent split-K scratch is sized for
// (248320 — Qwen3.8-27B's output head). rdna4_gemm_ml84_decode_v2 returns
// false for any N above this rather than growing scratch mid-launch.
int ml84_decode_v2_max_n(void);

#ifdef __cplusplus
}
#endif
