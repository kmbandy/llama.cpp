// gemm_mxfp4_radiance.h -- radiance's MXFP4 x FP8 GEMM (gfx1201 / RDNA4 fp8
// WMMA) for ggml MXFP4 weights. Kernels are radiance_mxfp4_fp8.hip's decode
// and folded kernels (~/GitHub/ggz14-vllm-mxfp4, the copy gemm_ml84_radiance
// was ported from), WPERM=true instantiation only, with the epilogue store
// templated on the output type (f32 for ggml dst, bf16 kept for parity).
//
// Numerics are radiance's W4A8: the E2M1 weight is upconverted to e4m3 with
// the 32-block E8M0 exponent folded in relative to a per-row reference
// exponent Wref[n] (max over the row's blocks), exact while
// d = Wref[n] - e[blk] <= 8 (e4m3 subnormals, honoured by the gfx12 WMMA),
// rounded for d 9..12, flushed beyond. Activations are per-row fp8 e4m3.
//
// Weight layout ("packed"), one contiguous buffer per weight [N rows, K]:
//   W    [N*K/2]   e2m1 nibbles, low nibble = even k, in radiance's WPERM
//                  fragment order (one 32-lane uint32 slot per 16-row n-tile
//                  and 16-wide k-step)
//   Ws   [K/32][N] E8M0 block exponents, transposed
//   Wref [N]       per-row max block exponent
// rdna4_mxfp4_radiance_repack builds it from ggml block_mxfp4 rows
// (17 B per 32 elements: e, then qs[j] = elem j | elem j+16 << 4).
#pragma once
#include <hip/hip_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Shape gate shared by every entry point: N % 16 == 0, K % 128 == 0.
bool rdna4_mxfp4_radiance_shape_ok(int N, int K);

// Packed-buffer size and sub-buffer offsets (bytes) for an [N, K] weight.
size_t rdna4_mxfp4_radiance_packed_bytes(int N, int K);
size_t rdna4_mxfp4_radiance_ws_offset(int N, int K);
size_t rdna4_mxfp4_radiance_wref_offset(int N, int K);

// ggml block_mxfp4 [N][K/32] -> packed (dst must not alias src).
bool rdna4_mxfp4_radiance_repack(const void* src_blocks, int N, int K, void* dst_packed,
                                 hipStream_t stream);

// Bytes for one activation buffer (A_fp8 or A_lo): M*K row-major, or
// ceil(M/16)*16*K tiled (pad rows zero).
size_t rdna4_mxfp4_radiance_a_bytes(int M, int K, bool tiled);

// f32 activations [M, K] (row stride `ld` elements) -> fp8 e4m3 [M, K] + per-row
// fp32 scale, radiance's producer numerics:
// scale = max(amax/448, 1/(448*512)), q = cvt_pk_fp8(clamp(x/scale, +-448)).
// A_lo (nullable): hi/lo split. Also writes lo = e4m3(x/scale - hi) [M, K], same
// scale, so x ~= (hi + lo) * scale. A_lo == null gives today's single-A output.
// tiled == false: row-major [M, K] (byte-identical to the original path).
// tiled == true : radiance's A-tiled layout: 16x16 fragments of 256 contiguous
//   bytes per (mt = m/16, ks = k/16) at base (mt*(K/16) + ks)*256; lane l
//   (0..31) owns bytes 8l..8l+7 = A[mt*16 + l%16][ks*16 + (l/16)*8 .. +7].
//   Pad rows (m >= M, up to a multiple of 16) are zero. Same values and scales.
bool rdna4_mxfp4_radiance_quant_a(const float* x, int64_t ld, int M, int K, uint8_t* A_fp8,
                                  uint8_t* A_lo, float* a_scale, bool tiled, hipStream_t stream);

// Split-K scratch the GEMM may need for this (M, N): partial sums + per-block
// counters. 0 means none. The GEMM zeroes the counters itself (memsetAsync),
// so pool memory is fine.
size_t rdna4_mxfp4_radiance_scratch_bytes(int M, int N, int K);

// C[M, N] (row-major, contiguous) = A_fp8[M, K] . dequant(W)[N, K]^T.
// A_lo (nullable, same layout as A_fp8): when set, C = (A_fp8 + A_lo) . dequant(W)^T,
// with W*hi and W*lo accumulated into one accumulator (same weight fragment).
// a_tiled == false: row-major A. Decode kernel for M <= RDNA4_MXFP4_RADIANCE_DECODE_MAX_M
// (default 64; M in 65..128 measured faster on the atiled hilo path than on hilo-decode,
// 2026-10-17), folded kernel above.
// a_tiled == true : A is in the tiled layout (see quant_a). Atiled kernel; only valid
// above the decode band (decode reads row-major A), returns false otherwise.
// Returns false (nothing launched) on an unsupported shape.
bool rdna4_gemm_mxfp4_radiance(int M, const uint8_t* A_fp8, const uint8_t* A_lo,
                               const float* a_scale, const void* packed, float* C_f32, int N,
                               int K, void* scratch, size_t scratch_bytes, bool a_tiled,
                               hipStream_t stream);

// packed -> ggml block_mxfp4 [N][K/32] (exact inverse of repack: WPERM undone, even/odd nibbles back to
// ggml's j / j+16 split, Ws back into each block's e byte; Wref is dropped). dst must not alias packed.
bool rdna4_mxfp4_radiance_unpack(const void* packed, int N, int K, void* dst_blocks, hipStream_t stream);

// ---- grouped (MUL_MAT_ID) path: device-side routing, no host sync, graph-capturable ----
// Workspace the grouped GEMM needs (0 = unsupported shape). Depends only on host-known sizes.
size_t rdna4_mxfp4_radiance_moe_ws_bytes(int n_as, int n_used, int n_tok, int N, int K);

// dst[ids_dst[r]][N] = x_row(r) . dequant(expert_of(r))^T for every routed row r, one launch per kernel family.
//   x          src1 f32 [K, ne11, n_tok]; row for (token t, slot s) at x + (bcast ? 0 : s)*x_nb1 + t*x_nb2
//              (strides in floats); bcast = (ne11 == 1)
//   bounds     [n_as+1] expert_bounds and ids_dst [n_tok*n_used] from ggml_cuda_launch_mm_ids_helper:
//              compact row c of expert e (bounds[e] <= c < bounds[e+1]) writes dst row ids_dst[c]
//              = token*n_used + slot (dst rows are N contiguous floats; dst nb[2] == n_used*nb[1])
//   packed     expert e's packed matrix at packed + e*wstride (wstride >= packed_bytes(N, K));
//              wptrs (nullable): device array [n_as] of per-expert matrix base pointers (the weight pager's
//              expert_ptrs); when set, `packed`/`wstride` are ignored and expert e's matrix is at wptrs[e].
//              n_mats = 2 is a fused gate;up pair: matrix 1 of the same expert sits mat_off bytes after
//              matrix 0, both have N rows, and dst rows are [mat0 N | mat1 N] (ldc = 2N). The activations
//              are routed + quantized once and shared by both GEMMs.
//   ws         ws_bytes >= rdna4_mxfp4_radiance_moe_ws_bytes(...), any pool memory
// Experts with few rows run the decode kernel (row-major hi/lo A), experts above the decode band the atiled
// kernel (tiled hi/lo A); blocks beyond an expert's rows exit early. Returns false (nothing useful launched)
// on an unsupported shape or short workspace.
bool rdna4_gemm_mxfp4_radiance_moe(const float* x, int64_t x_nb1, int64_t x_nb2, bool bcast,
                                   const int* bounds, const int* ids_dst, int n_as, int n_used,
                                   int n_tok, const void* packed, int64_t wstride,
                                   const void* const* wptrs, int n_mats,
                                   int64_t mat_off, float* dst, int N, int K, void* ws,
                                   size_t ws_bytes, hipStream_t stream);

// ---- row mode for tiny n_tok (decode): no routing helper / plan / padding ----
// Each routed row r = token*n_used + slot is its own M=1 problem: expert ids[r] (ids contiguous [n_tok][n_used]),
// activation row quantized to hi/lo once per row, ONE decode launch per matrix over all rows. Rows with an
// out-of-range expert id are skipped (dst row untouched). Workspace from rdna4_mxfp4_radiance_moe_rows_ws_bytes.
size_t rdna4_mxfp4_radiance_moe_rows_ws_bytes(int n_used, int n_tok, int N, int K);
bool rdna4_gemm_mxfp4_radiance_moe_rows(const float* x, int64_t x_nb1, int64_t x_nb2, bool bcast,
                                        const int* ids, int n_as, int n_used, int n_tok,
                                        const void* packed, int64_t wstride,
                                        const void* const* wptrs, int n_mats,
                                        int64_t mat_off, float* dst, int N, int K, void* ws,
                                        size_t ws_bytes, hipStream_t stream);

#ifdef __cplusplus
}
#endif
