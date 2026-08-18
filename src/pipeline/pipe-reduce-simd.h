#pragma once

// pipe-reduce-simd: vectorized helpers for the spine-side "unpack" path of
// the expert-dispatch pipeline protocol (pipe-protocol.cpp /
// pipe-expert-dispatcher.cpp).
//
// Measured bottleneck (DS4-Flash sliced-expert spine, 2026-08-17): each
// decode token does one 43-layer cross-machine expert dispatch, and per
// dispatch the spine-side CPU work is: for each of 43 layers x 3 workers,
// deserialize one partial, convert f16 -> f32 if needed, and sum it into an
// accumulator. That convert+sum is ~48% of the token's wall time and today
// runs through ggml's bulk row conversion (scalar loop) plus a separate
// plain elementwise add loop in the dispatcher -- two full passes over the
// buffer with no vectorization on either.
//
// These helpers replace that with one (or, for the fused variant, one)
// vectorized pass: runtime-dispatched AVX-512F / AVX2+F16C kernels on
// x86-64 with an always-correct scalar fallback for any other target or a
// tail that isn't a multiple of the vector width.
//
// NUMERIC CONTRACT -- read before changing any of this:
//   - The cross-worker reduction (summing N worker partials into one
//     accumulator for a given layer) MUST stay in a FIXED, caller-chosen
//     order and MUST be done in f32. f16-summing was tried upstream and
//     reverted; do not reintroduce it. These helpers do not decide that
//     order -- the caller (the dispatcher) invokes pipe_simd_accumulate_f32
//     / pipe_simd_decode_f16_accumulate once per partial, in whatever fixed
//     request order it already uses, and each call only does the
//     elementwise add of *that one partial* into the accumulator. No
//     reduction across elements happens inside these functions.
//   - f16 -> f32 widening is exact (every representable half has an exact
//     f32 value), so the vectorized decode is bit-identical to the scalar
//     reference (ggml_compute_fp16_to_fp32) for every input -- there is no
//     tolerance to document there.
//   - The f32 accumulate itself (acc[i] = acc[i] + add[i]) is a single FP
//     add per element in every implementation (scalar, AVX2, AVX-512); it is
//     not a horizontal reduction, so there is no associativity reordering
//     and the vectorized accumulate is bit-identical to the scalar
//     reference too. (This is different from summing multiple elements of
//     one row into a scalar, which WOULD be reorder-sensitive -- that's not
//     what this does.)

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// acc[i] += add[i] for i in [0, n). Vectorized f32 elementwise accumulate
// with a scalar fallback; safe for any n (including 0) and any alignment.
// Does not reorder anything across calls -- callers control cross-partial
// summation order by the order in which they call this.
void pipe_simd_accumulate_f32(float * acc, const float * add, size_t n);

// out[i] = fp16_to_fp32(half[i]) for i in [0, n). Vectorized f16->f32
// convert-only path (no accumulate), bit-identical to
// ggml_compute_fp16_to_fp32 / ggml_fp16_to_fp32_row for every input.
void pipe_simd_convert_f16_to_f32(float * out, const uint16_t * half, size_t n);

// Fused decode+accumulate: acc[i] += fp16_to_fp32(half[i]) for i in [0, n).
// Avoids materializing a separate f32 scratch buffer between convert and
// add at the dispatcher's per-partial harvest site. Bit-identical to
// calling pipe_simd_convert_f16_to_f32 into a scratch buffer followed by
// pipe_simd_accumulate_f32 (both of which are themselves bit-identical to
// their scalar references).
void pipe_simd_decode_f16_accumulate(float * acc, const uint16_t * half, size_t n);

// Runtime gate for the SIMD unpack path, read once from WP_SIMD_UNPACK
// (unset/0 = off = the legacy ggml bulk-convert + scalar scatter_add; 1 = on).
// Default OFF so the shipping serving path is unchanged until measured; flip
// via env for the A/B, then promote to config-of-record once confirmed. The
// SIMD path is bit-identical to the scalar path, so this gate only affects
// speed, never output.
int pipe_simd_unpack_enabled(void);

#ifdef __cplusplus
}
#endif
