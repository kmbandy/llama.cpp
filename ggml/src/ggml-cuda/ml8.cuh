// ml8.cuh — GGML_TYPE_ML8_4 on-device repack for the HIP backend (MAD-223 G.4.d).
//
// GGUF stores ml8-4 weight rows as a stream of `block_ml8_4` structures
// (fp32 scale + 32 packed nibbles per QK_ML8=64 K-block), interleaved per
// row. The mt_ml8_gemm Triton kernel — which the HIP backend dispatches
// GGML_OP_ML8_MUL_MAT to — instead expects two separated row-major
// device tensors:
//
//   b_packed [K/2, N]    uint8   — nibbles only
//   b_scale  [n_groups_k, N] fp32 — scales only
//
// This module owns the one-time repack from the on-device block_ml8_4
// layout into the separated layout, plus a process-static cache keyed on
// the weight's device pointer so each weight tensor is repacked at most
// once per process.
//
// Two layers, separated for testability:
//   1. ggml_cuda_ml8_repack_blocks(...): pure kernel-launch helper. Takes
//      device pointers in/out. No ggml dependency. Unit-testable.
//   2. ggml_cuda_ml8_get_or_repack(...): cache-keyed lookup over (1).
//      Allocates device side buffers on first call for a weight, stores
//      pointers in a static map, returns the same pointers on subsequent
//      calls.
//
// MAD-223 Phase G.4.d.

#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_tensor;

struct ml8_weight_repack_t {
    void *  b_packed;     // device: uint8 [K/2, N] row-major
    // device: [n_groups_k, N] row-major per-(group, col) scale. Dtype is a
    // property of which weight type populated this entry, NOT a fixed C
    // type: ML8_4 (LUT path) scales are fp32 (float*); ML8_FP8 (no-LUT,
    // in-place) scales are fp16 (__half*), copied through verbatim from the
    // on-disk fp16 scale to keep the packed layout at 8.5 bpw (see
    // ggml_cuda_ml8_inplace_alloc_size). Callers must know which producer
    // filled this struct and cast accordingly — see ggml_cuda_ml8_get_or_repack
    // (fp32) vs ggml_cuda_ml8_fp8_get_or_repack / the in-place ML8_FP8 path
    // (fp16) in ml8.cu.
    void *  b_scale;
    int32_t N;
    int32_t K;
    int32_t n_groups_k;
    int32_t group_size;   // currently always QK_ML8 = 64
    // FP8_B128 only (design 4(a)/(b) + generic-layout follow-up): which
    // packed byte layout b_packed/b_scale are in — 0 = FP8_B128_LAYOUT_GENERIC
    // (b_packed is e4m3 [K, N] row-major, b_scale is fp32 [K/128, N/128]),
    // 1 = FP8_B128_LAYOUT_PRESHUFFLE (b_packed is the AITER shuffle_weight
    // (16,16) permutation, b_scale is the same fp32 [K/128, N/128] table).
    // Set once at pack time from MT_FP8_B128_LAYOUT; unused (left 0) by
    // ML8_4/ML8_FP8 producers. See the FP8_B128_LAYOUT_* constants in ml8.cu.
    int32_t layout;
};

// Pure repack helper. All pointers are device (HIP) pointers. Caller owns
// allocations; this function only launches the repack kernel on `stream`.
//
//   src_blocks  device, byte-shape (N, n_groups_k * sizeof(block_ml8_4))
//               row-major. Each row of N is n_groups_k contiguous blocks
//               of 36 bytes (4-byte fp32 scale + 32 bytes packed nibbles).
//   dst_b_packed device, uint8 (K/2, N) row-major. Must be at least
//               (K/2) * N bytes.
//   dst_b_scale device, fp32 (n_groups_k, N) row-major. Must be at least
//               n_groups_k * N * sizeof(float) bytes.
//
// K must be a positive multiple of group_size; group_size must equal
// QK_ML8 (64) for now.
void ggml_cuda_ml8_repack_blocks(
    cudaStream_t stream,
    const void * src_blocks,
    void *       dst_b_packed,
    float *      dst_b_scale,
    int32_t      N,
    int32_t      K,
    int32_t      group_size);

// Cache-keyed lookup. On first call for a given weight tensor, allocates
// device buffers and repacks. On subsequent calls, returns the cached
// pointers. Cache key is `w->data` (the device pointer of the weight
// blocks); cache is process-static, mutex-protected, and survives until
// ggml_cuda_ml8_clear_cache() is called explicitly (or process exit).
//
// `w` must be a GGML_TYPE_ML8_4 tensor with ne[0]=K (multiple of QK_ML8)
// and ne[1]=N. Returns nullptr on shape/type validation failure.
const ml8_weight_repack_t * ggml_cuda_ml8_get_or_repack(
    cudaStream_t        stream,
    const ggml_tensor * w);

// Free every cached repack entry's device allocations and clear the
// cache. Intended for tests and explicit shutdown; not called from the
// normal backend teardown path (the OS reclaims VRAM at process exit).
void ggml_cuda_ml8_clear_cache(void);

// ─────────────────────────────────────────────────────────────────────
// ML8_FP8 in-place repack (load-time).
//
// The WF=0 Triton GEMM reads B as raw e4m3 [K, N] plus fp16 scales
// [K/32, N] (copied through verbatim from the on-disk value, upcast to
// fp32 only in the epilogue multiply); the GGUF stores [N, K] rows of
// 34-byte {fp16 scale, 32 e4m3} blocks. The cache above builds the kernel
// layout as a SECOND device copy on first use, which doubles the weight
// footprint of a full model. For a plain 2D ML8_FP8 weight the HIP buffer
// instead allocates the kernel layout directly (8.5 bpw, matching the
// on-disk size exactly) and set_tensor transposes the host blocks into it
// once at load; the tensor's ->data then IS the repack and the GEMM uses
// it with no cache entry. get_tensor reverses the transform.
// ─────────────────────────────────────────────────────────────────────

// True when `t` is a contiguous 2D ML8_FP8 (K % 32 == 0) or ML8_4 (K % 64 == 0)
// tensor that the HIP buffer stores in the kernel layout (ML8_4: nibbles
// [K/2,N] + fp32 scales, 4.5 bpw, no growth). Must give the same answer at
// get_alloc_size and at set_tensor time. Env WP_ML8_INPLACE=0 disables
// (falls back to the cached second copy; diagnostic only).
bool ggml_cuda_ml8_inplace_eligible(const ggml_tensor * t);

// Bytes of the kernel layout: for ML8_FP8, K*N e4m3 + (K/32)*N fp16 scales
// (== ggml_nbytes(t), 8.5 bpw); for ML8_4, (K/2)*N nibbles + (K/64)*N fp32
// scales (4.5 bpw, unchanged).
size_t ggml_cuda_ml8_inplace_alloc_size(const ggml_tensor * t);

// Host -> device write of on-disk block bytes into an eligible tensor.
// Accepts the same (offset, size, n_copies, stride_tensor, stride_data)
// shape as set_tensor_2d (n_copies == 1 for a plain set_tensor); partial
// writes are staged on the device and the repack runs once the whole
// tensor has arrived. Synchronous on return.
void ggml_cuda_ml8_inplace_set(
    cudaStream_t  stream,
    ggml_tensor * t,
    const void *  data,
    size_t        offset,
    size_t        size,
    size_t        n_copies,
    size_t        stride_tensor,
    size_t        stride_data);

// Device -> host read of on-disk block bytes [offset, offset+size) from an
// eligible, fully-written tensor. Synchronous on return.
void ggml_cuda_ml8_inplace_get(
    cudaStream_t        stream,
    const ggml_tensor * t,
    void *              data,
    size_t              offset,
    size_t              size);

// True when `data` is the ->data of a fully-repacked in-place tensor.
bool ggml_cuda_ml8_inplace_is_packed(const void * data);

// GGML_OP_GET_ROWS on an in-place packed ML8_FP8 tensor (token_embd): gathers
// rows out of the kernel layout. Returns false (does nothing) when src0 is
// not a packed in-place tensor so the caller falls through to the generic
// block-layout path.
bool ggml_cuda_ml8_inplace_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Register `dst_data` as a packed copy of `src_data` (device-to-device
// buffer copy of the whole allocation). No-op if src is not packed.
void ggml_cuda_ml8_inplace_alias(const void * src_data, const void * dst_data);

// Drop every in-place registration and staging entry whose pointer lies in
// [base, base + size): called when a HIP buffer is freed.
void ggml_cuda_ml8_inplace_forget_range(const void * base, size_t size);

// Quantize a row-major fp32 activation tensor (src, [M, K]) into the
// (a_fp8[M, K] uint8 e4m3, a_scale[M] fp32) layout that mt_ml8_gemm
// consumes. Per-row absmax scaling: a_scale[m] = max(|x[m]|) / 448 +
// epsilon; a_fp8[m, k] = round_to_e4m3(x[m, k] / a_scale[m]). The
// mt_ml8_gemm formula multiplies a_scale back at the end, so the round-
// trip is `x ≈ a_fp8 × a_scale` up to fp8 quant noise.
//
// All pointers are device. Caller owns allocations. dst_a_fp8 must be
// at least M*K bytes; dst_a_scale must be at least M*sizeof(float).
//
// M_valid ≤ M: rows in [M_valid, M) are emitted as zero fp8 + epsilon scale
// WITHOUT reading src_fp32 — this folds the GEMM M-padding into the quantize
// kernel so callers don't need a zero-padded fp32 staging copy of x
// (src_fp32 only needs M_valid rows). Pass M_valid == M when src has all rows.
void ggml_cuda_ml8_quantize_activations(
    cudaStream_t stream,
    const float * src_fp32,    // device, fp32 [M_valid, K] row-major
    void *        dst_a_fp8,   // device, uint8 [M, K] row-major
    float *       dst_a_scale, // device, fp32 [M]
    int32_t       M,
    int32_t       K,
    int32_t       M_valid);

struct ggml_backend_cuda_context;

// Execute GGML_OP_ML8_MUL_MAT on the HIP backend. `dst` is fp32 [N, M]
// (ne[0]=N, ne[1]=M) with sources:
//   dst->src[0]: w         — GGML_TYPE_ML8_4,   ne[0]=K, ne[1]=N
//   dst->src[1]: centroids — GGML_TYPE_F8_E4M3, ne[0]=16, ne[1]=n_groups_k
//   dst->src[2]: x         — GGML_TYPE_F32,     ne[0]=K, ne[1]=M
//
// Pipeline (all on `ctx.stream()`):
//   1. Cache-lookup or build repacked weights: w → (b_packed[K/2, N],
//      b_scale[n_groups_k, N]) (load-time work, one-shot per weight).
//   2. Pad M up to a multiple of MT_ML8_BLOCK_SIZE_M (16); copy x into
//      a temp fp32 buffer with zero-padding for the extra rows.
//   3. Quantize: temp_fp32 → (a_fp8[M_pad, K], a_scale[M_pad]).
//   4. Launch mt_ml8_gemm → bf16 [M_pad, N] temp output.
//   5. Convert bf16 → fp32 for the first M*N elements, written into
//      dst->data.
//
// Padding rows are zero-input and the kernel just produces zero rows
// the dst slice never copies — no correctness impact, only a small
// compute overhead at small M.
void ggml_cuda_op_ml8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// GGML_OP_ML8_GET_ROWS dispatch — native 4-bit token-embedding gather.
// Unlike ggml_cuda_op_ml8_mul_mat this needs NO AITER GEMM: it gathers row
// ids[i] from the ml8-4 weight and dequantizes via the per-K-group centroid
// LUT directly on device. Available on any CUDA/HIP build.
//   dst:        fp32  [K, ids->ne0, ids->ne1, ids->ne2]
//   dst->src[0]: w    — GGML_TYPE_ML8_4,   ne[0]=K (mult. of QK_ML8=64), ne[1]=N(vocab)
//   dst->src[1]: cent — GGML_TYPE_F8_E4M3, ne[0]=16, ne[1]=K/QK_ML8 (shared LUT)
//   dst->src[2]: ids  — GGML_TYPE_I32
void ggml_cuda_op_ml8_get_rows(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// Execute a plain GGML_OP_MUL_MAT whose src[0] is a GGML_TYPE_ML8_FP8
// (scaled-fp8) weight, via the no-LUT FP8-WMMA path (WEIGHT_FORMAT=0).
// Unlike GGML_OP_ML8_MUL_MAT there is no centroid sidecar:
//   dst:        fp32 [N, M]
//   dst->src[0]: w  — GGML_TYPE_ML8_FP8, ne[0]=K (mult. of QK_ML8_FP8=32), ne[1]=N
//   dst->src[1]: x  — GGML_TYPE_F32,     ne[0]=K, ne[1]=M
// Routed from ggml_cuda_mul_mat (NOT op-swapped at load time).
void ggml_cuda_op_ml8_fp8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// MAD-223 G.7 — per-expert MoE repack. Sibling of ml8_weight_repack_t.
//   b_packed [n_experts, K/2, N]    uint8
//   b_scale  [n_experts, n_groups_k, N] fp32
struct ml8_weight_repack_moe_t {
    void *  b_packed;
    float * b_scale;
    int32_t N;
    int32_t K;
    int32_t n_groups_k;
    int32_t group_size;   // currently QK_ML8 = 64
    int32_t n_experts;
};

// Pure per-expert repack helper. src_blocks layout matches the on-device
// stack-of-experts ML8_4 tensor (n_experts × N × n_groups_k × 36 bytes).
// Calls the dense repack kernel n_experts times under the hood.
void ggml_cuda_ml8_repack_blocks_moe(
    cudaStream_t stream,
    const void * src_blocks,
    void *       dst_b_packed,
    float *      dst_b_scale,
    int32_t      N,
    int32_t      K,
    int32_t      group_size,
    int32_t      n_experts);

// Cache-keyed MoE repack. Key is `w->data` (the per-tensor device pointer);
// `w` must be a GGML_TYPE_ML8_4 tensor with ne[0]=K, ne[1]=N, ne[2]=n_experts.
const ml8_weight_repack_moe_t * ggml_cuda_ml8_get_or_repack_moe(
    cudaStream_t        stream,
    const ggml_tensor * w);

// Execute GGML_OP_ML8_MUL_MAT_ID on the HIP backend.
//   dst:        fp32 [N, n_used, n_tokens]
//   src[0]: w         GGML_TYPE_ML8_4    [K, N, n_experts]
//   src[1]: centroids GGML_TYPE_F8_E4M3  [16, n_groups_k, n_experts]
//   src[2]: x         GGML_TYPE_F32      [K, n_used, n_tokens]
//   src[3]: ids       GGML_TYPE_I32      [n_used, n_tokens]
//
// Pipeline (all on ctx.stream()):
//   1. Cache-lookup or build per-expert repacked weights stack.
//   2. Read `ids` to host; bin (s, t) pairs by expert; build routing
//      tensors (ExptHist, ExptOffs, GatherIndx, ExptData, InvGather) and
//      upload to device. Pad each expert's chunk to MT_ML8_MOE_BLOCK_M.
//   3. Quantize x[K, n_used*n_tokens] → fp8 + per-row scale (same kernel
//      as the dense path; GatherIndx routes inside the gemm).
//   4. Launch mt_ml8_moe_gemm → bf16 [M_padded, N] in sorted order.
//   5. Scatter sorted bf16 output back to dst[N, n_used, n_tokens] fp32
//      via InvGather.
void ggml_cuda_op_ml8_mul_mat_id(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// Execute GGML_OP_ML8_APPLY_ROTATION on the HIP backend.
//   dst:       fp32 [d, n_tokens]
//   src[0]: x  fp32 [d, n_tokens]   (d = a_dim * b_dim; n_tokens spans
//                                    ne[1]..ne[3] for batched/MoE inputs)
//   src[1]: h_a fp32 [a_dim, a_dim] OR NULL
//   op_params[0] = a_dim, op_params[1] = b_dim (power of 2, 16..1024)
//
// Math, kronecker_orth_sylvester (h_a != NULL): Y[:, t] reshapes X[:, t] to
// (a, b), then H_a^T @ X @ H_b (per token). a_dim <= 16 (register-array
// bound on the H_a leg — see ml8_h_a_left_multiply_kernel in ml8.cu).
//
// Math, block_hadamard (h_a == NULL, MAD-266): Y[:, t] = X[:, t] @ H_b only
// (Q = I_a ⊗ H_b) — no H_a leg, no a_dim limit. Used for ML8_FP8 weights
// under tensor-parallel K-split, where each device only holds a slice of x
// and there is no cross-block a-leg to mix.
//
// H_b is the Sylvester Hadamard, applied via the row-wise FWHT kernel
// (turbo_fp8_hadamard.cuh), normalized identically in both kinds.
void ggml_cuda_op_ml8_apply_rotation(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// ─────────────────────────────────────────────────────────────────────
// G.6.d — fused {ML8_APPLY_ROTATION → ML8_MUL_MAT} dispatch.
//
// The rotation's FWHT (H_b) + small H_a^T left-multiply are absorbed into
// the GEMM's activation-quantize prologue as ONE kernel, eliding the
// rotation node's output tensor and its memcpy/fwht/h_a_left launches.
// Per rotated GEMM the chain shrinks from
//   copy → fwht → h_a_left → quantize → gemm → bf16→fp32
// to
//   fused_rot_quant → gemm → bf16→fp32.
// Bitwise-equivalent math to the unfused chain (same butterfly schedule,
// normalize, accumulation and quantize ordering).
//
// can_fuse gate (cheap, called from ggml_cuda_try_fuse):
//   mm->src[2] == rot, fp32 contiguous input, a_dim ≤ 16, b_dim pow2 in
//   [16, 1024], K = a_dim*b_dim fits LDS, M > 1 (M == 1 keeps the GEMV
//   fast path unfused), ML8_NO_FUSE unset, ML8_DUMP debug harness off.
bool ggml_cuda_ml8_can_fuse_rot_mm(
    const ggml_tensor * rot,
    const ggml_tensor * mm);

// `rot` is the ML8_APPLY_ROTATION node, `dst` the ML8_MUL_MAT node.
// Reads x from rot->src[0] and h_a from rot->src[1]; rot->data is never
// touched (the tensor is elided by the fusion).
void ggml_cuda_op_ml8_mul_mat_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor *         rot,
    ggml_tensor *               dst);

// ─────────────────────────────────────────────────────────────────────
// FP8_B128 phase 2 (design doc section 4). GGML_TYPE_FP8_B128 rides the
// SAME in-place registry as ML8_FP8/ML8_4 above (ggml_cuda_ml8_inplace_*):
// eligible()/alloc_size()/set()/get() all branch on t->type internally, so
// none of the ggml-cuda.cu set_tensor/get_tensor/cpy_tensor/get_alloc_size
// hooks need to change. The packed layout differs from ML8_FP8 though: it
// is the AITER preshuffled-(16,16) weight layout + a [K/128, N/128] fp32
// scale table (design 4(a)), not a straight transpose.
// ─────────────────────────────────────────────────────────────────────

// Execute GGML_OP_FP8_QUANT_ROT on the HIP backend (design 4(c)). Reuses the
// same FWHT (mt_turbo_fp8_fwht) and H_a^T left-multiply
// (ml8_h_a_left_multiply_kernel) primitives ggml_cuda_op_ml8_apply_rotation
// already uses, so the rotation math is byte-for-byte the same kernel code
// as the ML8_FP8/ML8_4 rotation path; only the final per-128-group e4m3
// quantize + packed-row layout is new.
void ggml_cuda_op_fp8_quant_rot(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// Execute GGML_OP_FP8_MUL_MAT on the HIP backend (design 4(b)): looks up the
// packed weight in the in-place registry (or a cache-keyed second copy when
// WP_ML8_INPLACE=0 or the weight isn't in-place eligible, e.g. N not a
// multiple of 128) and launches the GEMM matching how the weight was
// packed (ml8_weight_repack_t::layout, decided once at load time by
// MT_FP8_B128_LAYOUT):
//   generic (default):  _gemm_a8w8_blockscale_kernel (WEIGHT_FORMAT=0) via
//                        mt_fp8_b128_gemm_generic — measured faster on
//                        gfx1201.
//   preshuffle:          _gemm_a8w8_blockscale_preshuffle_kernel via
//                        mt_fp8_b128_gemm — kept selectable for A/B.
void ggml_cuda_op_fp8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);

// Unpack a packed in-place (or aliased) GGML_TYPE_FP8_B128 tensor into a
// freshly cudaMalloc'd device buffer holding the on-disk block_fp8_b128
// bytes (ggml_nbytes(t) long). Used by the generic GET_ROWS dequant
// fallback (getrows.cu) so it never reads packed preshuffled bytes as
// blocks. Caller owns the returned pointer and must cudaFree it. Synchronous
// on `stream` before returning (the caller reads back through it
// immediately). Returns nullptr if `t->data` is not a packed FP8_B128 entry.
void * ggml_cuda_ml8_inplace_fp8_b128_unpack_to_device(
    cudaStream_t         stream,
    const ggml_tensor  * t);

// MAD-305 Phase 5 -- ML8_FP8 sibling of the above, needed once ML8_FP8 gained
// a second packed layout (RDNA4 trfeed, alongside the original TRITON [K,N]
// transpose): unpacks a packed in-place GGML_TYPE_ML8_FP8 entry into a
// freshly cudaMalloc'd device buffer holding the on-disk block_ml8_fp8 bytes
// (ggml_nbytes(t) long), regardless of which layout it was packed in. Used by
// the GET_ROWS dequant fallback (getrows.cu) when the packed-layout fast path
// (ggml_cuda_ml8_inplace_get_rows) declines an RDNA4-layout weight. Caller
// owns the returned pointer and must cudaFree it. Returns nullptr if
// `t->data` is not a fully-packed ML8_FP8 entry.
void * ggml_cuda_ml8_inplace_ml8fp8_unpack_to_device(
    cudaStream_t         stream,
    const ggml_tensor  * t);

// True when the FP8_B128 packed layout selected at load (MT_FP8_B128_LAYOUT,
// default rdna4 = frozen trfeed kernel) consumes the per-row (G=0) activation
// packing; false for the Triton layouts (block-128 activation packing).
bool ggml_cuda_fp8_b128_layout_is_per_row(void);
