#pragma once

// mt_sparse_attn_dsv4 — AITER-backed native sparse DSv4 attention op
// (GGML_OP_SPARSE_ATTN_DSV4). Active only when ggml-hip is built with
// GGML_HIP_AITER=ON and the runtime knob WP_DSV41_SPARSE_ATTN=1 is set.
//
// See ggml/src/ggml-cuda/aiter-integration/wrappers/mt_sparse_attn_dsv4.h
// for the kernel-launch contract, and src/models/deepseek41.cpp's
// build_attention_v41 for the graph-side index construction (window +
// indexer top-k -> a fixed-stride [n_idx, n_tokens] kv_indices tensor) that
// feeds this op.

#include "common.cuh"

namespace mt {

#ifdef GGML_HIP_AITER

// Runtime gate. Reads env var WP_DSV41_SPARSE_ATTN once and caches the
// result. Compiled out entirely when GGML_HIP_AITER is undefined.
bool sparse_attn_dsv4_enabled();

// Debug/equivalence-check gate: WP_DSV41_SPARSE_ATTN_CHECK=1 additionally
// compares, for the layers/ubatches it fires on, the sparse op's constructed
// index set against the dense kq_mask's admitted columns (logged from the
// graph-build side, see deepseek41.cpp) and the sparse op's own output
// against a freshly-computed dense attention result for the same inputs.
bool sparse_attn_dsv4_check_enabled();

#endif // GGML_HIP_AITER

// Shape gate for ggml_backend_cuda_supports_op — must be cheap and callable
// unconditionally (returns false when GGML_HIP_AITER is undefined or the
// AITER kernel isn't available).
bool ggml_cuda_sparse_attn_dsv4_supported(const ggml_tensor * op);

// Dispatch entry, called from ggml-cuda.cu's compute-forward switch.
void ggml_cuda_op_sparse_attn_dsv4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

} // namespace mt
