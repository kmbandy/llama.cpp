#pragma once

// mt_pagedattn_wmma_fp8 — hand-written HIP WMMA flash-attention prefill
// kernel for the (head_dim=256, GQA=6, TURBO4_FP8_BS256 KV cache) shape
// that AITER's Triton kernel_unified_attention_2d currently serves at
// ~27 TF on gfx1201/RDNA4. See mt_pagedattn_wmma_fp8.cu for the full
// design writeup, VGPR/LDS budget, and known limitations.
//
// Gated by env MT_PAGED_ATTN_WMMA=1 (default OFF). Only ever selected for
// prefill-sized M (per-seq max q_len >= 64); decode stays on AITER. Only
// covers the non-Hadamard TURBO4_FP8_BS256 path (MT_TURBO_FP8_HADAMARD
// unset/0) — see the .cu file header comment for why.

#include "common.cuh"

struct ggml_tensor;
namespace ggml_backend_cuda_context_fwd {}  // (fwd decl not needed; see common.cuh)
class ggml_backend_cuda_context;

namespace mt {

// True iff MT_PAGED_ATTN_WMMA=1 is set. Read once, cached for process lifetime.
bool paged_attn_wmma_fp8_env_enabled();

// Runtime shape gate — call BEFORE aiter_backend_enabled()'s early-return in
// ggml_cuda_op_paged_attn_mt. All of the following must hold:
//   head_size == 256, cache_type == GGML_TYPE_TURBO4_FP8_BS256,
//   n_heads / n_kv_heads == 6 (GQA-6), max_q_len >= 64 (prefill-sized M),
//   amd_wmma_available(cc), and Hadamard mode NOT active
//   (mt_turbo_fp8::hadamard_required() == false — v1 does not rotate Q).
bool paged_attn_wmma_fp8_shape_ok(int cc, int head_size, ggml_type cache_type,
                                   int n_heads, int n_kv_heads, int max_q_len);

// Dispatch entry. Same dst->src[] layout as ggml_cuda_op_paged_attn_mt_aiter
// (q, k_cache, v_cache, block_tables, context_lens, q_lens, k_cur, v_cur,
// slot_mapping) and the same op_params (scale, block_size, max_bps,
// n_kv_heads, max_q_len, max_ctx_len). Does NOT perform the KV scatter —
// caller must have already written k_cur/v_cur into the paged cache via the
// normal AITER scatter path before calling this (see mt_pagedattn.cu wiring:
// we piggy-back on ggml_cuda_op_paged_attn_mt_aiter's own scatter call by
// letting AITER's scatter run first, then overriding just the attention
// kernel choice). Asserts the shape gate itself; caller should still check
// paged_attn_wmma_fp8_shape_ok() first to decide whether to route here at all.
void ggml_cuda_op_paged_attn_mt_wmma_fp8(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

}  // namespace mt
