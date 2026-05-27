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
    float * b_scale;      // device: fp32 [n_groups_k, N] row-major
    int32_t N;
    int32_t K;
    int32_t n_groups_k;
    int32_t group_size;   // currently always QK_ML8 = 64
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

// Quantize a row-major fp32 activation tensor (src, [M, K]) into the
// (a_fp8[M, K] uint8 e4m3, a_scale[M] fp32) layout that mt_ml8_gemm
// consumes. Per-row absmax scaling: a_scale[m] = max(|x[m]|) / 448 +
// epsilon; a_fp8[m, k] = round_to_e4m3(x[m, k] / a_scale[m]). The
// mt_ml8_gemm formula multiplies a_scale back at the end, so the round-
// trip is `x ≈ a_fp8 × a_scale` up to fp8 quant noise.
//
// All pointers are device. Caller owns allocations. dst_a_fp8 must be
// at least M*K bytes; dst_a_scale must be at least M*sizeof(float).
void ggml_cuda_ml8_quantize_activations(
    cudaStream_t stream,
    const float * src_fp32,    // device, fp32 [M, K] row-major
    void *        dst_a_fp8,   // device, uint8 [M, K] row-major
    float *       dst_a_scale, // device, fp32 [M]
    int32_t       M,
    int32_t       K);

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
//   src[0]: x  fp32 [d, n_tokens]   (d = a_dim * b_dim)
//   src[1]: h_a fp32 [a_dim, a_dim]
//   op_params[0] = a_dim, op_params[1] = b_dim (power of 2, ≤ 1024)
//
// Math: Y[:, t] reshapes X[:, t] to (a, b), then H_a^T @ X @ H_b (per token).
// H_b is the Sylvester Hadamard, built once per b_dim and cached in device
// memory. One CUDA block per token, blockDim.x = b_dim, shared memory holds
// the intermediate (a*b) fp32 buffer (≤ 36KB at a=9, b=1024 — fits AMD LDS).
void ggml_cuda_op_ml8_apply_rotation(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst);
