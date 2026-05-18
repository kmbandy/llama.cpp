#pragma once

// mt_pagedattn_aiter — AITER-backed path for the mt:: paged attention op.
//
// Active only when ggml-hip is built with GGML_HIP_AITER=ON and the runtime
// flag is set (env var `MAD_USE_AITER=1` for now; will become a CLI option).
//
// The AITER path uses a different KV cache layout than the existing tile/
// scalar/decode kernels:
//
//   AITER  : [num_blocks, block_size, n_kv_heads, head_size]
//   existing: K = [num_blocks, n_kv_heads, head_size/x, block_size, x]
//             V = [num_blocks, n_kv_heads, head_size, block_size]
//
// The cache buffer's BYTE size is identical for both layouts (total elements
// `num_blocks * block_size * n_kv_heads * head_size` either way) — only the
// scatter and attention kernels' interpretation differs. So the allocator
// doesn't need to change: one layout per run, chosen at startup. Mixing is
// not supported (mid-run swap would require a re-shuffle).
//
// MAD-188.

#include "common.cuh"

namespace mt {

#ifdef GGML_HIP_AITER

// Runtime gate. Reads env var `MAD_USE_AITER` once and caches the result.
// Returns true iff the AITER path should handle paged attention for this
// process. Compiled out entirely when GGML_HIP_AITER is undefined.
bool aiter_backend_enabled();

// AITER-path dispatch entry. Same signature as
// ggml_cuda_op_paged_attn_mt — performs scatter (AITER layout) and attention
// (unified_attention 3D + reduce_segments). The caller must have verified
// aiter_backend_enabled() returns true before calling this.
void ggml_cuda_op_paged_attn_mt_aiter(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

#else  // GGML_HIP_AITER undefined — stub out

inline bool aiter_backend_enabled() { return false; }
inline void ggml_cuda_op_paged_attn_mt_aiter(ggml_backend_cuda_context &, ggml_tensor *) {
    // Unreachable: caller must check aiter_backend_enabled() first.
}

#endif

}  // namespace mt
