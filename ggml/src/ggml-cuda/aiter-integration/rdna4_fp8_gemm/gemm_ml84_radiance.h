// gemm_ml84_radiance.h — MAD-305 ML8_4 (4.5 bpw) PORT of radiance's prefill
// GEMM kernel (radiance_mxfp4_fp8_gemm_atiled, WPERM=1 instantiation) onto
// ml8-4's nibble+LUT+per-group-scale numerics.
//
// THIS IS A PORT, NOT A NEW KERNEL. Per the task directive: take radiance's
// prefill GEMM kernel VERBATIM (same tile shapes, LDS layout, double
// buffering, sched_barrier/lds-barrier placement, fragment loads, epilogue)
// from ~/GitHub/ggz14-vllm-mxfp4/radiance_mxfp4_fp8.hip and change ONLY the
// code that turns a packed 4-bit weight nibble into an fp8 e4m3 byte at LDS
// staging time. See gemm_ml84_radiance.hip's header comment for exact
// source line ranges and the exact staging-step diff.
//
// Why the atiled variant: ggz14's serve-mxfp4.sh defaults
// RADIANCE_MXFP4_A_TILED_MIN_M=513 and RADIANCE_MXFP4_WPERM=1 (unless TP
// padding forces WPERM=0), so the profile that measured 3512 tok/s
// dispatches radiance_mxfp4_fp8_gemm_atiled<TN,WPERM=true> for every
// prefill-class (M>=513) linear, not the row-major "_folded" kernel. See
// the report accompanying this change for the grep trail
// (radiance_mxfp4.py's A_TILED_MIN_M / a_tiled_wanted gate + the launch_at
// dispatch in radiance_mxfp4_fp8.hip).
//
// Because the dispatched kernel is the ATILED variant, its activation
// operand must already be in radiance's fragment-tiled layout (the layout
// radiance_add_rms_quant / radiance_silu_mul_quant emit under
// RADIANCE_MXFP4_A_TILED_MIN_M). This directory has no equivalent fused
// norm+quant producer yet, so a standalone "retile A" entry point is
// provided below (ported from radiance_silu_mul_quant's TILED store
// indexing, gemm_ml84_radiance.hip's header comment gives the exact lines)
// so the bench and a future dispatch can turn a plain row-major fp8 [M,K]
// activation + per-row a_scale into the tiled operand this GEMM expects.
//
// FORMAT MAPPING (ml8-4 -> radiance's staging slot):
//   Radiance:  MXFP4 nibble (E2M1, sign+3-bit magnitude) -> fp8 e4m3, with
//              the group's E8M0 (power-of-two) scale folded into the fp8
//              EXPONENT at staging (kMag[d] table, d = exponent shift).
//   Ours:      ml8-4 nibble is a plain 4-bit INDEX into a 16-entry e4m3
//              centroid LUT, per-(64-wide K-group g, column n) scale
//              b_scale_g[g][n] which is an ARBITRARY fp32 (not a power of
//              two) -- it cannot be folded into the fp8 exponent the way
//              radiance folds E8M0. Instead we precompute, ONCE at
//              weight-load time (rdna4_gemm_ml84_radiance_prep below), a
//              per-(group, column) 16-byte conversion table
//                T[g][n][c] = e4m3_round(centroid[c] * b_scale_g[g][n] / colscale[n])
//              (the SAME double-quantization the production
//              rdna4_expand_ml84_to_trfeed path already performs and ships
//              -- see gemm_ml84_prod.hip's "SCALE CONVENTION" comment) and
//              a per-column colscale[n] computed EXACTLY the way
//              ml84_scale_colmax_kernel computes it (gemm_ml84_prod.hip).
//              The ported kernel's staging step then does a single 16-way
//              LUT gather `byte = T[g][n][nibble]` via v_perm/byte-select,
//              the same technique bench/trfeed_ml84_kernels.h's
//              ml84_lut_gather4 uses for the decode kernel's LUT select.
//              colscale[n] is applied in the epilogue in the exact spot
//              radiance's own per-column weight-scale factor
//              (2^Wref[n]) was applied, alongside the per-token a_scale.
#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─────────────────────────────────────────────────────────────────────────
// Load-time prep: builds the per-(group,column) conversion table T and the
// per-column colscale from the ml8-4 weight's own LUT + b_scale_g -- NOT a
// per-token/per-inference-step cost. Call once after
// rdna4_pack_ml84_trfeed (gemm_ml84_prod.hip) produces B_nib/b_scale_g for
// this weight, before any rdna4_gemm_ml84_radiance call against it.
//
// B_nib is accepted (not read) only for interface symmetry with the other
// ml8-4 entry points in this directory (gemm_capi.h) and to leave room for
// a future variant that validates T against the packed weight; the current
// implementation only reads lut and b_scale_g.
//
// Sizes:
//   lut:            F8_E4M3 [K/64][16]        (K/64 * 16 bytes)
//   b_scale_g:      fp32    [K/64][N]          (K/64 * N floats)
//   T_out:          uint8   [K/64][N][16]      (K/64 * N * 16 bytes)  -- caller-allocated
//   colscale_out:   fp32    [N]                (N floats)            -- caller-allocated
//
// Returns false (nothing launched) if N<=0, K<=0, K % 64 != 0 (QK_ML8) or
// N % 16 != 0. Errors after a `true` return are async HIP errors, same
// convention as every other launcher in this directory (pick up with the
// caller's own hipGetLastError()).
bool rdna4_gemm_ml84_radiance_prep(const uint8_t* B_nib, const uint8_t* lut, const float* b_scale_g,
                                    int N, int K, uint8_t* T_out, float* colscale_out, hipStream_t stream);

// Host (CPU) fallback for rdna4_gemm_ml84_radiance_prep -- bit-identical
// formula, used by the bench oracle so the oracle does not depend on the
// device kernel it is meant to check. Synchronous; no stream.
void rdna4_ml84_radiance_prep_host(const uint8_t* lut, const float* b_scale_g,
                                    int N, int K, uint8_t* T_out, float* colscale_out);

// ─────────────────────────────────────────────────────────────────────────
// Retile a row-major fp8 e4m3 activation A[M,K] (per-row a_scale, applied
// by the caller/epilogue elsewhere -- this only moves bytes) into the
// fragment-tiled layout radiance_mxfp4_fp8_gemm_atiled's AT operand
// expects. Ported from radiance_silu_mul_quant's TILED store indexing
// (radiance_mxfp4_fp8.hip:1495, `(mt*kst+(g>>1))*32+lrow+16*(g&1)`) --
// see gemm_ml84_radiance.hip for the derivation.
//
// AT_out must be sized ceil(M/16)*16*K bytes (rows padded up to a multiple
// of 16; pad rows are zero-filled and never read back into any output
// element -- radiance's own M-tile clamp only ever re-reads a REAL row for
// the last partial tile, per the ORIGINAL kernel's clamp-not-predicate
// design, unchanged here). K must be a multiple of 128 (same requirement
// rdna4_gemm_ml84_radiance enforces, since both consume the same AT
// layout).
//
// Returns false (nothing launched) if M<=0, K<=0 or K % 128 != 0.
bool rdna4_gemm_ml84_radiance_retile_a(int M, int K, const uint8_t* A_rowmajor_fp8,
                                        uint8_t* AT_out, hipStream_t stream);

// Host (CPU) fallback for rdna4_gemm_ml84_radiance_retile_a, used by the
// bench to build ground truth without depending on the device kernel.
void rdna4_ml84_radiance_retile_a_host(int M, int K, const uint8_t* A_rowmajor_fp8, uint8_t* AT_out);

// ─────────────────────────────────────────────────────────────────────────
// The ported GEMM. C[M,N] fp32 = A_fp8[M,K] (fragment-tiled, per-row
// a_scale) . dequant(ml8-4 B)[N,K]^T, dequant via the T/colscale tables
// rdna4_gemm_ml84_radiance_prep built.
//
//   A_fp8:     fragment-tiled fp8 e4m3 operand, EXACTLY the layout
//              rdna4_gemm_ml84_radiance_retile_a produces (radiance's own
//              AT contract -- see that function's comment). NOT row-major.
//   a_scale:   fp32[M] per-row activation scale (row indexes the ORIGINAL,
//              untiled M -- same M this call's own `M` argument names).
//   B_nib:     ML84_TRFEED packed nibbles, N*K/2 bytes (bench/
//              ml84_trfeed_layout.h's b_tile_offset(kt,nt,N/16)/2 tile
//              addressing -- the SAME buffer every other ml8-4 entry point
//              in this directory uses).
//   T:         the per-(group,column) conversion table from
//              rdna4_gemm_ml84_radiance_prep, uint8 [K/64][N][16].
//   colscale:  the per-column scale from rdna4_gemm_ml84_radiance_prep,
//              fp32[N].
//   C_f32:     fp32 [M,N] row-major output (dst-shaped, like every other
//              ml8-4 entry point here -- NOT radiance's own bf16 output).
//
// Returns false (no kernel launched, no allocation, no side effects) if
// the caller must fall back to the existing path
// (rdna4_expand_ml84_to_trfeed + rdna4_gemm_fp8_trfeed_f32):
//   - N <= 0, N % 16 != 0
//   - K <= 0, K % 128 != 0 (radiance's own tiled-A requirement -- see
//     radiance_mxfp4_fp8.hip:1663, "tiled A needs K % 128 == 0") or
//     K % 64 != 0 (QK_ML8 group width)
//   - M <= 0
// This function has NO hipError_t return, same convention as every other
// launcher in this directory (errors after a `true` return are async HIP
// errors the caller picks up with its own hipGetLastError()).
bool rdna4_gemm_ml84_radiance(int M, const void* A_fp8, const float* a_scale, const uint8_t* B_nib,
                               const uint8_t* T, const float* colscale, float* C_f32, int N, int K,
                               hipStream_t stream);

#ifdef __cplusplus
}
#endif
