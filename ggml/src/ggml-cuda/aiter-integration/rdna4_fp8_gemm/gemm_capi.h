#pragma once
#include <hip/hip_runtime.h>
#include <stdint.h>
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

// ─────────────────────────────────────────────────────────────────────────
// fp32-output sibling of rdna4_gemm_fp8_trfeed above (MAD-305, freeze lifted
// 2026-09-18): the frozen Phase-1 "trfeed" kernel body is UNCHANGED (its
// K-loop and WMMA sequence are identical bytes to the bf16 instantiation --
// see bench/trfeed_kernels.h's template comment); only the epilogue's output
// branch differs, selected at compile time via
// gemm_fp8_trfeed<TBM,TWAVES_M,float>. Motivation: every production caller's
// consumer of C is fp32 (ggml_cuda_op_fp8_mul_mat writes dst->data, fp32), so
// the bf16 path costs both a bf16 rounding pass AND a full
// convert_unary<bf16,float> kernel copying M_pad x N elements back out
// (measured 0.385s of a 7.2s 8192-token prefill on an R9700, MAD-305
// 2026-09-18 rocprofv3 trace) -- this entry point removes both.
//
// Same A/B_shuf/a_scale/b_scale contracts and M_pad/tile-selection rules as
// rdna4_gemm_fp8_trfeed (M_pad==32 -> decode/verify tile; else M_pad must be
// an exact multiple of BM=128 -> prefill tile), PLUS:
//   c_f32:    fp32 [M_valid, N] row-major, stride N -- sized to the CALLER's
//             true M, NOT M_pad. No padded scratch is needed or read: the
//             epilogue's float branch (unlike the bf16 branch, which the
//             existing callers make an effectively-unguarded full-tile
//             store by always passing M=M_pad) masks every store against
//             M_valid, so rows in [M_valid, M_pad) -- the padding the
//             frozen kernel's A-tile LDS fill still computes results for --
//             are computed but never written.
//   M_valid:  the caller's true M (1 <= M_valid <= M_pad). Rows >= M_valid
//             within the dispatched tile are silently dropped by the
//             epilogue; this is the ONLY new required parameter versus the
//             bf16 entry point above.
// Returns hipErrorInvalidValue if M_valid is <= 0 or > M_pad, in addition to
// every bf16-path validity check (N % 128 == 0, K % 32 == 0, M_pad tile
// rules).
hipError_t rdna4_gemm_fp8_trfeed_f32(const uint8_t* a_fp8, const uint8_t* b_shuf, float* c_f32,
                                     const float* a_scale, const float* b_scale,
                                     int M_pad, int M_valid, int N, int K, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// bf16-output sibling of rdna4_gemm_fp8_trfeed_f32 above (LLAMA_ACT_BF16,
// 2026-09-18 phase 2): same frozen K-loop, same M_valid-guarded epilogue
// contract as the f32 entry point, but writes bf16 -- for the qwen35
// BF16-activation-stream mode, where the ML8_MUL_MAT/FP8_MUL_MAT consumer
// wants a bf16 dst directly (half the write traffic vs f32, and no
// convert_unary<bf16,float> pass is needed since nothing reads a scratch
// fp32 buffer). Same M_pad/M_valid/N/K contracts as rdna4_gemm_fp8_trfeed_f32
// (c_bf16 is [M_valid, N], NOT [M_pad, N] -- rows >= M_valid are silently
// dropped by the epilogue, see trfeed_kernels.h's M_valid comment on the
// bf16 branch).
hipError_t rdna4_gemm_fp8_trfeed_bf16(const uint8_t* a_fp8, const uint8_t* b_shuf, void* c_bf16,
                                      const float* a_scale, const float* b_scale,
                                      int M_pad, int M_valid, int N, int K, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// SwiGLU-epilogue sibling of rdna4_gemm_fp8_trfeed_f32 above (MAD-305 fused
// FFN task, 2026-09-18): fuses ffn_gate/ffn_up's two ML8_4 GEMMs and the
// swiglu(gate,up) GLU into ONE kernel launch, so the two GEMM output
// stores (2x142 MB at Qwen3.8-27B ML8_4 prefill shapes) and the GLU's own
// read/write pass (~430 MB) never happen -- see ml8.cu's
// ggml_cuda_op_ml8_ffn_gate_up_swiglu, the sole caller.
//
// The K-loop is BYTE-IDENTICAL to rdna4_gemm_fp8_trfeed_f32's (same frozen
// gemm_fp8_trfeed<TBM,TWAVES_M> body, only instantiated with the added
// SwiGLU=true template flag).
//
// ROUND 2 (perf bug, 2026-09-18): the FIRST version of this fusion
// interleaved gate/up at 64-COLUMN granularity within each 128-wide tile
// (tile t's first 64 columns all gate, second 64 all up). Under
// WAVES_N=2/BN=128 that put a tile's gate half in wave_n=0 and its up half
// in a DIFFERENT wavefront (wave_n=1), so the epilogue needed a
// __syncthreads() + cross-wave shared-memory exchange per output tile --
// measured 22.50ms vs 4.73ms for the two plain GEMMs it replaced
// (M=2048/N_half=17408/K=5120, R9700 bench). Fixed by interleaving at
// 32-COLUMN granularity instead (see rdna4_expand_ml84_pair_to_trfeed and
// b_shuf_fused below): gate and its matching up channel then always land
// in the SAME wave's two fragments (ni and ni+TFRAGS_N/2), a pure
// register-local read with no cross-wave traffic and no barrier at all --
// see trfeed_kernels.h's epilogue comment for the full derivation. The
// K-loop itself never changed between the two versions; only this column
// placement rule did.
//
// a_fp8/a_scale: SAME single activation contract as every other trfeed
//   entry point (one shared `x` GEMV operand for both gate and up -- this
//   is what makes the fusion legal in the first place: both ML8_MUL_MATs
//   the caller is fusing must read the identical x).
// b_shuf_fused: N_fused*K bytes, built by rdna4_expand_ml84_pair_to_trfeed
//   (gemm_ml84_prod.hip) from the gate and up ML8_4 weights: N_fused =
//   2*N_half, and each of b_shuf_fused's 128-wide B_shuf tiles interleaves
//   gate/up at 32-column granularity within the (shared) output width
//   N_half: tile-local pos in [0,32) is gate channel [0,32), [32,64) is up
//   channel [0,32) (the SAME channel as the gate slot 32 positions
//   earlier), [64,96) is gate channel [32,64), [96,128) is up channel
//   [32,64).
// b_scale_fused: fp32[N_fused], ALSO produced by rdna4_expand_ml84_pair_to_trfeed,
//   indexed in the SAME interleaved fused-N space as b_shuf_fused -- gate
//   and up have independent per-column scales, applied to their own
//   accumulator BEFORE silu/mul, exactly as the un-fused
//   rdna4_gemm_fp8_trfeed_f32 epilogue applies a_scale[m]*b_scale[n] before
//   ANY other op.
// c_f32: fp32 [M_valid, N_half] row-major (N_half = N_fused/2) -- HALF the
//   fused width, one silu(gate)*up value per output column. No bf16
//   scratch, no separate GLU pass: this is the ONLY store of the fused
//   FFN's ffn_gate/ffn_up/glu subgraph.
// M_pad/M_valid: same rules as rdna4_gemm_fp8_trfeed_f32 (M_pad==32 ->
//   decode/verify tile, but this fusion is NOT wired for M_pad==32 by
//   ml8.cu -- decode stays two separate un-fused split-K GEMMs plus the
//   existing GLU kernel; M_pad>32 -> exact multiple of BM=128, the prefill
//   tile, which IS what ml8.cu fuses).
// N_fused % 128 == 0 (so N_half % 64 == 0, satisfying
// rdna4_expand_ml84_pair_to_trfeed's own N_half%64==0 requirement), K % 32 == 0.
hipError_t rdna4_gemm_fp8_trfeed_swiglu_f32(const uint8_t* a_fp8, const uint8_t* b_shuf_fused,
                                            float* c_f32, const float* a_scale, const float* b_scale_fused,
                                            int M_pad, int M_valid, int N_fused, int K, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// PRODUCTION for workgroup-starved decode shapes only (2026-09-17 bench,
// 9070 XT, M<=32 tile): N=17408/K=5120 (136 WGs) the frozen bf16 path
// 0.176 ms beats every split count (n=1: 0.227 ms, 16: 1.147 ms -- the
// atomic epilogue costs more than the parallelism buys, the kernel is
// already at ~80% of memory bandwidth there); N=5120/K=17408 (40 WGs) the
// frozen path is 0.478 ms and n_splits=4 gives 0.224 ms (2.1x). See
// rdna4_trfeed_splitk_default_splits for the rule (returns 1 = don't split).
//
// MAD-305 decode split-K variant of the FROZEN Phase-1 "trfeed" kernel,
// M_pad==32 (decode/verify tile) ONLY. Same A/B_shuf/a_scale/b_scale
// contracts as rdna4_gemm_fp8_trfeed above (M_pad must be exactly 32 --
// the prefill tile is never split-K'd, see below), but:
//   - grid.z (blockIdx.z) splits K into n_splits slices of
//     k_tiles_per_split = ceil((K/32)/n_splits) BK=32-wide tiles each, so
//     n_splits x more workgroups stream the weight matrix in parallel
//     instead of one workgroup per (N/128) tile serially walking the whole K.
//   - each workgroup's partial a_scale[m]*b_scale[n]-scaled product for its
//     K-slice is combined across splits with `atomicAdd` into C_f32 (fp32),
//     not a plain bf16 store -- so C_f32 must be freshly zeroed before
//     accumulation begins; this function does that itself with
//     hipMemsetAsync on `stream` before launching.
//   - output is fp32 [M_pad, N] directly (no bf16 intermediate), which also
//     removes the bf16->fp32 convert kernel the non-split-K production path
//     (rdna4_gemm_fp8_trfeed above) runs afterward.
// Why M_pad==32 only: the prefill tile (M_pad%128==0) already launches
// N/128 * M_pad/128 workgroups -- plenty of parallelism at M>32 -- so
// split-K there would only add atomic-reduction overhead for no
// occupancy gain. Kernel body: bench/trfeed_splitk_kernels.h's
// gemm_fp8_trfeed_splitk<32,1>, a separate copy of trfeed_kernels.h's
// frozen gemm_fp8_trfeed<32,1> body (main K-loop/WMMA sequence identical;
// only the K-tile range bounds and the epilogue's atomic fp32 store
// differ -- see that header's top comment).
// n_splits must be >= 1; the caller may use
// rdna4_trfeed_splitk_default_splits() below to pick one, or override via
// its own env var (production wiring: MT_FP8_TRFEED_SPLITS, ml8.cu).
hipError_t rdna4_gemm_fp8_trfeed_splitk(const void* A, const void* B_shuf, float* C_f32,
                                        const float* a_scale, const float* b_scale,
                                        int M_pad, int N, int K, int n_splits, hipStream_t stream);

// Heuristic default split count for rdna4_gemm_fp8_trfeed_splitk above:
// aims for >= 256 total workgroups (N/128 * n_splits) while keeping each
// split at least 4 K-tiles (128 K-elements) wide, i.e.
// n_splits = clamp(ceil(256 / (N/128)), 1, (K/32)/4). Returns 1 (no
// split) if N or K is non-positive or too small to split further.
int rdna4_trfeed_splitk_default_splits(int N, int K);

// ─────────────────────────────────────────────────────────────────────────
// MAD-305 ML8_4 (4.5 bpw) decode + prefill entry points (aiter-integration/
// rdna4_fp8_gemm/gemm_ml84_prod.hip). NOT wired into ggml_cuda_op_fp8_mul_mat
// or ml8.cu by this task -- a later task does that wiring. These reuse the
// FROZEN fp8 trfeed kernel bodies (trfeed_kernels.h) unmodified: the decode
// path is a separate-copy kernel with the same tile geometry/epilogue
// structure (bench/trfeed_ml84_kernels.h's gemm_ml84_trfeed<32,1>), and the
// prefill path is the expander below followed by the UNCHANGED
// rdna4_gemm_fp8_trfeed.
//
// Packed "ML84_TRFEED" layout (bench/ml84_trfeed_layout.h):
//   B_nib:      N*K/2 bytes. Same b_tile_offset(kt,nt,NT) 16(K)x16(N) tile
//               addressing as the frozen fp8 B_shuf, but each fp8 byte is
//               replaced by its 4-bit LUT index, two indices/byte: within
//               the 4-byte dword a lane would read (nib_dword_addr =
//               tile_nib_base + lane*4), elements s=0..3 sit in the LOW
//               nibbles of the dword's 4 bytes and elements s=4..7 in the
//               HIGH nibbles -- exactly the split the frozen kernel's
//               tr_load8 v2i32{x,y} return already uses (x=bytes0-3,
//               y=bytes4-7), so a plain load + LUT expand rebuilds the
//               identical WMMA B fragment.
//   b_scale_g:  fp32 [K/64][N] row-major (K-group outer, N inner) -- the
//               SAME layout ml8.cu's ggml_cuda_ml8_repack_blocks already
//               produces for block_ml8_4 weights; reused unchanged.
//   lut:        F8_E4M3 [K/64][16] as stored (16 bytes/group), the model's
//               per-K-group centroid table (ggml-turbo-quant.c's
//               dequantize_row_ml8_4_with_lut). Values are OCP e4m3 with
//               |c| <= 1 by construction of the calibration pipeline.
// ─────────────────────────────────────────────────────────────────────────

// One-time load-time repack: on-disk block_ml8_4 weight blocks, layout
// `w[N][K/64]` row-major per output column n (the SAME src_blocks
// convention ggml_cuda_ml8_repack_blocks/ml8_repack_kernel in ml8.cu
// consume: N groups of n_groups_k=K/64 consecutive 36-byte block_ml8_4
// records), into the ML84_TRFEED B_nib + b_scale_g packed layout above.
// OUT-OF-PLACE ONLY (w_blocks must not alias B_nib/b_scale_g) -- see
// gemm_ml84_prod.hip's ml84_pack_kernel comment for what an in-place
// variant would additionally need (that's the later wiring task's problem).
// K % 64 == 0 (QK_ML8 group width), N % 16 == 0.
hipError_t rdna4_pack_ml84_trfeed(const void* w_blocks, int N, int K,
                                  uint8_t* B_nib, float* b_scale_g, hipStream_t stream);

// Decode/verify GEMM: launches bench/trfeed_ml84_kernels.h's
// gemm_ml84_trfeed_splitk<32,1,ATOMIC=false> with n_splits=1 (one
// workgroup per (n-tile) walks the WHOLE K range and plain-stores once --
// see rdna4_gemm_ml84_trfeed_decode_splitk below, which this just calls).
// ROUND 3: an earlier version launched a separate LDS-staged kernel on the
// theory that exposed per-group global loads explained a measured
// 0.31ms-vs-0.176ms gap against the frozen fp8 kernel; that staging measured
// ZERO improvement (removed -- see trfeed_ml84_kernels.h's HISTORY comment)
// while unstaged split-K at n_splits=1 measured 0.194 ms, already within
// 10% of the frozen fp8 kernel's 0.176 ms.
// This kernel's B feed is a plain load from B_nib + a per-K-group
// (QK_ML8=64) LUT expand instead of `global_load_tr_b64` from a
// pre-shuffled fp8 B_shuf, and its per-(64-K-group, column) scale
// (b_scale_g) is folded into the accumulator once per group (inside the
// kernel) rather than once at the very end (unlike the fp8 path, where a
// single per-column scale is applied after the FULL K reduction) -- ML8_4
// has no single per-column scale, only per-(group,column), so the fold
// must happen per-group.
// A: fp8 e4m3 [M_pad, K] row-major, stride exactly K -- same contract as
//    rdna4_gemm_fp8_trfeed's A (pure weight bytes; caller supplies a_scale
//    separately). M_pad MUST be exactly 32 (this entry point only ever
//    dispatches the (32,1) decode tile; see rdna4_gemm_ml84_trfeed_prefill
//    below for the 128x128 prefill-tile experiment, and
//    rdna4_expand_ml84_to_trfeed + rdna4_gemm_fp8_trfeed for the
//    expander+frozen-fp8 prefill path). a_scale: fp32[32].
// C_f32: fp32 [32, N] row-major, dst->data-shaped -- fp32 output directly
//    (no bf16 intermediate; ML8_4's per-group scale fold already costs more
//    VALU than the fp8 path's single post-K multiply, so there is no
//    argument for adding a bf16 convert pass on top).
// K % 64 == 0, N % 128 == 0.
hipError_t rdna4_gemm_ml84_trfeed_decode(const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                         float* C_f32, const float* a_scale, const float* b_scale_g,
                                         int M_pad, int N, int K, hipStream_t stream);

// Split-K decode: for workgroup-starved N (N/128 workgroups too few to
// saturate the GPU at decode's tiny M), splits each (n-tile)'s K-GROUP
// range (QK_ML8=64-wide groups, not raw K-tiles -- a split boundary
// mid-group would corrupt the per-group scale fold) across blockIdx.z via
// bench/trfeed_ml84_kernels.h's gemm_ml84_trfeed_splitk<32,1,ATOMIC>.
// n_splits==1 uses the ATOMIC=false template path (plain store, no memset,
// grid.z=1 -- what rdna4_gemm_ml84_trfeed_decode calls); n_splits>1 uses
// ATOMIC=true, atomicAdd'ing each split's partial
// a_scale[m]*(sum over its groups of b_scale_g*acc_g) into a caller-zeroed
// fp32 C (this function zeroes it itself in that case, same as
// rdna4_gemm_fp8_trfeed_splitk). M_pad MUST be exactly 32. n_splits must be
// >= 1; rdna4_ml84_trfeed_splitk_default_splits below picks one.
hipError_t rdna4_gemm_ml84_trfeed_decode_splitk(const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                                float* C_f32, const float* a_scale, const float* b_scale_g,
                                                int M_pad, int N, int K, int n_splits, hipStream_t stream);

// Heuristic default split count for rdna4_gemm_ml84_trfeed_decode_splitk
// above -- ROUND 3, re-derived from measured numbers (9070 XT): N=17408
// (136 WGs) n_splits=1 was best (0.194 ms/258 GB/s; 2/4/8 all worse);
// N=5120/K=17408 (40 WGs) n_splits=4 was best (0.177 ms/283 GB/s; 1/2 worse,
// 8 WORSE than 4). Rule: no split once N/128 >= 128 (comfortably above a
// 9070 XT's CU count already); otherwise round(160/(N/128)) (160/40=4,
// exactly the measured-best n=4), capped at n_groups_k/2 (a split needs
// >=2 groups to amortize its own loop overhead) and at a hard 8 (n=8
// measured worse than n=4 at the only workgroup-starved shape tested).
int rdna4_ml84_trfeed_splitk_default_splits(int N, int K);

// Prefill EXPERIMENT: gemm_ml84_trfeed_splitk instantiated at the frozen
// prefill tile geometry (TBM=128, TWAVES_M=2, ATOMIC=false,
// groups_per_split=n_groups_k, grid.z=1 -- same "n_splits=1" trick as
// rdna4_gemm_ml84_trfeed_decode, just at the bigger tile) instead of
// expander+frozen-fp8. In-kernel LUT expansion is VALU work sharing the
// issue port with WMMA (ml84_lut_gather4's __builtin_amdgcn_perm calls), so
// this is expected to lose to expander+rdna4_gemm_fp8_trfeed at large M
// (where the expander's fixed ~0.4-0.6ms cost is amortized away) but may win
// at smaller M where that fixed cost dominates -- gemm_ml84_bench.hip
// measures both at M in {128,512,2048} rather than asserting an answer here.
// M_pad MUST be a multiple of 128 (the frozen prefill tile's BM). Same
// A/a_scale/b_scale_g/B_nib/lut contracts as rdna4_gemm_ml84_trfeed_decode,
// generalized to M_pad rows; C_f32: fp32 [M_pad, N] row-major.
hipError_t rdna4_gemm_ml84_trfeed_prefill(const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                          float* C_f32, const float* a_scale, const float* b_scale_g,
                                          int M_pad, int N, int K, hipStream_t stream);

// Prefill path, step 1: re-expand ML84_TRFEED (B_nib/lut/b_scale_g) into the
// frozen fp8 kernel's B_shuf (bench/trfeed_common.h's preshuffle_B tile
// layout -- SAME permutation, provably: both this expander and B_nib derive
// (lane L, slot s) from tile-local (kl,nl) via the identical formula, see
// ml84_trfeed_layout.h's ml84_lane_slot) plus a single per-column fp32
// scale, matching the frozen path's `amax/448` convention exactly:
//   b_scale_out[n] = (max over K-groups g of b_scale_g[g,n]) / 448
//   fp8 byte(k,n)  = e4m3_round(centroid(g,idx) * b_scale_g[g,n] / b_scale_out[n])
// (valid because LUT entries have |c| <= 1, so |w| <= b_scale_g[g,n] <=
// max_g(...) for every element -- the byte never exceeds e4m3's range).
// Step 2 is simply the UNCHANGED rdna4_gemm_fp8_trfeed(A, B_shuf_out, ...,
// b_scale_out, ...) -- no new prefill GEMM kernel exists or is needed.
// `lut` may already be offset by the caller for a K-slice (tensor-parallel
// use): this function indexes it as lut[g*16+idx] with g LOCAL to the
// (possibly-sliced) K range passed in, exactly like the decode path.
// K % 64 == 0, N % 16 == 0, K % 16 == 0.
hipError_t rdna4_expand_ml84_to_trfeed(const uint8_t* B_nib, const uint8_t* lut,
                                       const float* b_scale_g, int N, int K,
                                       uint8_t* B_shuf_out, float* b_scale_out, hipStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// Dual-source expander (MAD-305 fused FFN task, 2026-09-18; interleave
// granularity changed to 32 columns 2026-09-18, see below): the SAME
// re-expansion rdna4_expand_ml84_to_trfeed does for one ML8_4 weight, but
// for a (gate, up) PAIR sharing the same [N_half, K] shape, producing ONE
// fused B_shuf of width N_fused = 2*N_half whose 128-column tiles
// interleave the two sources at 32-COLUMN granularity: within each
// 128-fused-column block (fused column j = 128*block + pos, pos in
// [0,128)), pos in [0,32) is gate channel [0,32), [32,64) is up channel
// [0,32) (the SAME channel, paired -- silu(gate[c])*up[c] needs matching
// c), [64,96) is gate channel [32,64), [96,128) is up channel [32,64).
//
// ROUND 2 (perf bug): the FIRST version interleaved at 64-column
// granularity (tile t's first 64 columns all gate, second 64 all up).
// Under the GEMM's WAVES_N=2/BN=128 split that put a tile's gate half in
// wave_n=0 and its up half in a DIFFERENT physical wavefront (wave_n=1),
// so the SwiGLU epilogue needed a __syncthreads() + cross-wave
// shared-memory exchange per output tile -- measured a 4-5x SLOWDOWN vs
// the two plain GEMMs it replaced (bench/gemm_trfeed_prod_bench.hip,
// R9700: 22.50ms vs 4.73ms at M=2048/N_half=17408/K=5120). The 32-column
// scheme instead puts gate channel c and its matching up channel c in the
// SAME wave's two fragments (ni and ni+TFRAGS_N/2 under TFRAGS_N=4) for
// every 128-wide N-tile the GEMM processes -- a pure register-local
// pairing at epilogue time, no cross-wave traffic, no barrier -- see
// trfeed_kernels.h's SwiGLU epilogue comment for the full derivation. The
// GEMM K-loop is unaffected either way; only this column placement rule
// changed.
//
// Each source keeps its OWN LUT and per-(K-group,column) scale (the LUT is
// per-K-group PER WEIGHT, not shared -- gate and up are trained/calibrated
// independently), selected by which 32-column slot of the fused tile a
// given output column falls in; this function's dispatch has no notion of
// "combine the two LUTs", it just picks gate's or up's verbatim per
// element, exactly like rdna4_expand_ml84_to_trfeed picks its single
// source's LUT per element.
//
// Fused per-column b_scale is emitted the SAME "amax/448" way as the
// single-source expander (b_scale_out[j] = (max over K-groups of that
// source's own b_scale_g for j's source column) / 448), just computed
// against whichever source j's 32-column slot selects -- so a downstream
// GEMM epilogue can index b_scale_out[j] directly with no extra
// indirection to figure out which source's scale table j belongs to.
//
// gate_packed/gate_lut/gate_scale_g, up_packed/up_lut/up_scale_g: each is
// exactly one weight's ML84_TRFEED B_nib/lut/b_scale_g triple (see this
// header's ML84_TRFEED section above), both shaped [N_half, K] (same
// N_half, K for both -- ggml's FFN gate/up projections always share both
// dims; the caller, ml8.cu's fusion match, checks this before calling).
// b_shuf_out: N_fused*K bytes (N_fused = 2*N_half), fused B_shuf as
//   described above -- feeds gemm_fp8_trfeed_swiglu_f32.
// b_scale_out: fp32[N_fused], fused per-column scale as described above.
// N_half % 64 == 0 (so N_fused % 128 == 0, the GEMM's own tiling
// requirement -- N_half's OWN N%128==0 requirement from the single-source
// path does NOT apply here: N_half only needs to be a multiple of 64, two
// 32-column gate/up pairs per 128-wide GEMM tile), K % 64 == 0 (QK_ML8),
// K % 16 == 0.
hipError_t rdna4_expand_ml84_pair_to_trfeed(const uint8_t* gate_packed, const uint8_t* gate_lut,
                                            const float* gate_scale_g,
                                            const uint8_t* up_packed, const uint8_t* up_lut,
                                            const float* up_scale_g,
                                            int N_half, int K,
                                            uint8_t* b_shuf_out, float* b_scale_out, hipStream_t stream);
#ifdef __cplusplus
}
#endif
