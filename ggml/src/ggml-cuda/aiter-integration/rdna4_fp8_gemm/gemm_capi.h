#pragma once
#include <hip/hip_runtime.h>
#ifdef __cplusplus
extern "C" {
#endif
// C = (A_fp8[M,K] @ B_fp8[K,N]) * a_scale[M] (per-row) * b_scale[N] (per-col), out bf16[M,N].
// A,B are float8_e4m3 (OCP). All device pointers. Row-major. M,N multiples of 16; K multiple of 16.
void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                            const float* a_scale, const float* b_scale,
                            int M, int N, int K, hipStream_t stream);

// B is ml8: packed 4-bit indices [K/2, N] uint8 (lo-nibble-first) + per-K-group fp8 centroid LUT
// [n_groups_k,16] + per-(group,N) fp32 scale [n_groups_k, N]. group_size = K / n_groups_k.
// A is plain fp8 [M,K] with per-row a_scale[M]. Out bf16[M,N]. C = dequant(B)·A scaled.
void rdna4_gemm_ml8_forward(const void* A, const void* B_idx, void* C,
                            const float* a_scale, const void* centroids_fp8,
                            const float* b_group_scale,
                            int M, int N, int K, int group_size, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// MAD-305 Phase 5 — production GGML_TYPE_ML8_FP8 block-scale GEMM (trfeed).
//
// A (activation, GGML_OP_FP8_QUANT_ROT output, GGML_TYPE_I8):
//   row m starts at (const uint8_t*)a_packed + m*stride_am_bytes:
//     bytes [0, K)          raw OCP e4m3 weights for row m
//     bytes [K, K+K/8)      K/32 fp32 per-32-group absmax/448 activation scales
//   stride_am_bytes must equal K + K/8 (== a tensor's ne[0] in bytes).
//
// B (weight, packed once at load time by rdna4_preshuffle_b_ml8fp8 below):
//   b_shuf:      N*K bytes. The logical (transposed) e4m3 matrix B[K,N] (B[k,n]
//                is the weight's raw e4m3 byte at output row n, input col k),
//                pre-shuffled into 16(K)x16(N) `global_load_tr_b64` fragment
//                tiles (tile-major, 256 bytes/tile, KT=K/16 x NT=N/16 grid;
//                see rdna4_fp8_gemm/bench/global_load_tr_contract.md).
//   b_scale_f16: (K/32)*N fp16 values, row-major [K/32, N] (k-group outer, n
//                inner) — the on-disk per-(group,n) weight scale, copied
//                through verbatim (no widen/narrow) and upcast to fp32 only
//                inside the GEMM's per-K-tile scale fold.
//
// C: fp32 [M, N] row-major (row stride N floats) — dst->data, written directly.
//
// Math: out[m,n] = sum_g a_scale[m,g] * b_scale[g,n] * sum_{k in g} A[m,k]*B[k,n],
// fp32 accumulate; BK=32 == one scale group, so the fold happens once per K-tile.
// K % 32 == 0, N % 16 == 0. M arbitrary in [1, 4096] (masked A load / C store,
// no M padding — src rows beyond M-1 are never read).
hipError_t rdna4_gemm_ml8fp8_blockscale(const void* a_packed, int stride_am_bytes,
                                        const void* b_shuf, const void* b_scale_f16,
                                        float* c, int M, int N, int K, hipStream_t stream);

// One-time load-time repack: pre-shuffle the logical (transposed) e4m3 matrix
// b_transposed[K,N] (row-major over K, N inner — the SAME convention as the
// existing ML8_FP8 "triton" WEIGHT_FORMAT=0 b_packed layout) into the
// `global_load_tr_b64` fragment-tile layout `b_shuf` consumed by
// rdna4_gemm_ml8fp8_blockscale above. Device pointers only, launched on
// `stream`, no host round trip. K, N multiples of 16 (K%32==0 is the GEMM's
// stricter requirement; the shuffle itself only needs 16|K, 16|N).
hipError_t rdna4_preshuffle_b_ml8fp8(const void* b_transposed, void* b_shuf,
                                     int K, int N, hipStream_t stream);

// Inverse of rdna4_preshuffle_b_ml8fp8: gather `b_shuf` back into the
// transposed [K,N] row-major e4m3 layout. Used by get_tensor/cpy_tensor and
// the GET_ROWS dequant fallback to invert the packed weight. NOTE: despite
// the "ml8fp8" name this pair is a pure e4m3-byte permutation, independent
// of which on-disk quant format the bytes came from -- ml8.cu's FP8_B128
// rdna4 layout (round 2, G=128 scale groups) reuses these two functions
// unchanged for its own weight-byte packing.
hipError_t rdna4_unshuffle_b_ml8fp8(const void* b_shuf, void* b_transposed,
                                    int K, int N, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// MAD-305 Phase 5 (round 2) — production GGML_TYPE_FP8_B128 block-scale GEMM,
// G=128 scale groups (gemm_blockscale.hip). Same trfeed feed/tile mechanics
// as rdna4_gemm_ml8fp8_blockscale above, but folds the scale once per 128-K
// group instead of per 32-K tile: RDNA shares the WMMA and VALU issue port,
// and 32-wide folding measured only 34-37 TF (vs trfeed's 131 TF) from VALU
// overhead alone, not spills. 128-wide folding (this entry point) is the fix.
//
// A (activation, GGML_OP_FP8_QUANT_ROT's G=128 default output, GGML_TYPE_I8):
//   row m starts at (const uint8_t*)a_packed + m*stride_am_bytes:
//     bytes [0, K)          raw OCP e4m3 weights for row m
//     bytes [K, K+K/32)      K/128 fp32 per-128-group absmax/448 activation scales
//   stride_am_bytes must equal K + K/32 (== a tensor's ne[0] in bytes).
//
// B (weight, GGML_TYPE_FP8_B128, packed once at load time by
// rdna4_preshuffle_b_ml8fp8, FP8_B128_LAYOUT_RDNA4 in ml8.cu):
//   b_shuf:      N*K bytes. The logical (transposed) e4m3 matrix B[K,N],
//                pre-shuffled into the SAME `global_load_tr_b64` fragment-tile
//                layout rdna4_gemm_ml8fp8_blockscale's b_shuf uses (identical
//                permutation -- see rdna4_preshuffle_b_ml8fp8 above).
//   b_scale_f32: (K/128)*(N/128) fp32 values, row-major [K/128, N/128]
//                (k-group outer, n-tile inner) -- the on-disk per-128x128-tile
//                weight scale (FP8_B128's own on-disk granularity, so no
//                widen/narrow round-trip and no group-vs-tile mismatch).
//
// C: fp32 [M, N] row-major (row stride N floats) — dst->data, written directly.
//
// Math: out[m,n] = sum_g a_scale[m,g] * b_scale[g,n/128] * sum_{k in g} A[m,k]*B[k,n],
// fp32 accumulate; the 128-wide N tile a block owns is exactly one FP8_B128
// scale tile, so b_scale is ONE scalar per (block, K-group), not a per-column
// slice.
// K % 128 == 0, N % 128 == 0 (b_scale is indexed by n/128; a ggml N that
// isn't a multiple of 128 has nowhere consistent to source a tile scale from
// -- same CONVERTER INVARIANT the existing FP8_B128 preshuffle/generic
// layouts already require). M arbitrary in [1, 4096+] (masked A load / C
// store, no M padding — src rows beyond M-1 are never read; tile choice is
// purely a function of M at every call, covering M=1..40 decode/verify
// batches and a prompt's ragged last ubatch identically).
hipError_t rdna4_gemm_fp8b128_blockscale(const void* a_packed, int stride_am_bytes,
                                         const void* b_shuf, const float* b_scale_f32,
                                         float* c, int M, int N, int K, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// MAD-305 Phase 5 (round 3, CURRENT PRODUCTION PATH) — the FROZEN Phase-1
// "trfeed" kernel (bench/gemm_trfeed_bench.hip's gemm_fp8_trfeed<128,2>:
// acc[4][4], BM=128/BN=128/BK=32, 4 waves, B fed straight from a
// pre-shuffled global buffer via `global_load_tr_b64`, per-row a_scale x
// per-column b_scale applied ONLY in the epilogue after the full K
// reduction, bf16 output). Measured 131 TF at real production shapes
// (gemm_blockscale.hip's own comment, MAD-305) vs the two in-loop
// block-scale-fold kernels above, which measured 7-35 TF from VALU
// starvation/spills at every scale-group width tried (32 and 128) --
// THOSE ARE ABANDONED; this is the sole entry point GGML_OP_FP8_MUL_MAT's
// HIP dispatch (ml8.cu) uses by default.
//
// THE KERNEL BODY IS FROZEN. rdna4_gemm_fp8_trfeed's implementation
// (gemm_trfeed_prod.hip) `#include`s bench/trfeed_kernels.h -- the SAME
// header bench/gemm_trfeed_bench.hip includes -- so production and bench
// compile byte-identical device code for gemm_fp8_trfeed<128,2>. Do not
// change one instruction of that header's K-loop or epilogue; if a
// different named variant (e.g. gemm_fp8_trfeed_rb, also frozen verbatim in
// the same header) is ever wanted instead, swap ONLY the template
// instantiation in gemm_trfeed_prod.hip's launch call -- never edit the
// kernel bodies themselves.
//
// A: fp8 e4m3 [M_pad, K] row-major, stride exactly K (no interleaved
//    scale bytes -- unlike the two blockscale ABIs above, this kernel's A
//    pointer is pure weight bytes; the caller reads a_scale out of the
//    activation's separate per-row scale segment, see the GGML_OP_FP8_QUANT_ROT
//    "per-row" (G=0) contract: dst I8 [K+4, M], bytes [M*K, M*K+4*M) =
//    fp32 a_scale[M]). Rows >= the caller's true M must still be PRESENT
//    (zeroed) up to M_pad: the frozen kernel's A-tile LDS fill has NO bound
//    check against M (only the C-store epilogue masks on M), so M_pad must
//    be a full multiple of the M tile actually dispatched, not merely of 16.
//    Two tile instantiations of this SAME frozen template share the entry
//    point: M<=32 (decode/verify) dispatches gemm_fp8_trfeed<32,1> with
//    M_pad = round_up(M,32) == 32 exactly; M>32 (prefill) dispatches
//    gemm_fp8_trfeed<128,2> with M_pad = round_up(M,128). A plain
//    round_up(M,16) would leave the grid's last M-tile partially reading
//    uninitialized/OOB pool memory for any M in, e.g., {2, 33, 512+1..127}.
//    ggml_cuda_op_fp8_mul_mat picks the tile/M_pad this way; the entry point
//    below infers WHICH tile from M_pad itself (M_pad==32 -> decode tile,
//    else M_pad must be a multiple of 128 -> prefill tile).
//    a_scale: fp32[M_pad] (tail beyond the caller's true M may be anything;
//    C is only ever stored for rows < the true M via the epilogue mask).
// B_shuf: N*K bytes, the logical (transposed) e4m3 matrix B[K,N] pre-shuffled
//    into 16(K)x16(N) `global_load_tr_b64` fragment tiles exactly as
//    bench/trfeed_common.h's preshuffle_B()/b_tile_offset() define. Produced
//    on-device by rdna4_preshuffle_b_ml8fp8 above (verified byte-for-byte:
//    that function's bs_trperm/bs_tile_offset implement the identical
//    permutation as trfeed_common.h's trperm/b_tile_offset).
// b_scale: fp32[N], one scalar per output column n (FP8_B128's on-disk
//    weight scale is one value per output row n, replicated by the
//    converter into every 128-block of that row -- the packer reads the
//    fp16 d of block (n, kblock 0) and converts; see the
//    FP8_B128_LAYOUT_RDNA4_TRFEED weight-pack comment in ml8.cu).
// C_bf16: bf16 [M_pad, N] row-major, stride N. Caller converts the first M
//    rows to dst->data (fp32 [M, N]) with a small bf16->fp32 kernel; rows
//    [M, M_pad) of C_bf16 are scratch and never read back.
//
// M_pad == 32 (decode/verify tile) OR M_pad % 128 == 0 (prefill tile; exact,
// not merely %16 -- see above), N % 128 == 0, K % 32 == 0. Grid = (N/BN,
// M_pad/BM) for whichever tile, both exact divisions.
hipError_t rdna4_gemm_fp8_trfeed(const void* A, const void* B_shuf, void* C_bf16,
                                 const float* a_scale, const float* b_scale,
                                 int M_pad, int N, int K, hipStream_t stream);
#ifdef __cplusplus
}
#endif
