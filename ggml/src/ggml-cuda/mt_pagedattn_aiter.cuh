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

// Non-static wrapper around this file's internal layer-index parser (reads a
// tensor name like "cache_k_l<N>" / "cache_v_l<N>"). Exported so the R4D
// adapter (mt_pagedattn_r4d.cu) can bind the turbo4_fp8 centroid LUTs to the
// right layer for its own turbo4kv path without duplicating the parse logic.
// Returns -1 if the pattern doesn't match.
int mt_aiter_parse_layer_from_kv_cache_name(const char * name);

// Exported turbo4_fp8 KV scatter launch (MAD-214 Phase 1G-G kernel), reused
// by the R4D adapter's turbo4kv path so both adapters scatter into the exact
// same on-disk record layout. k_cache/v_cache are the SEPARATE turbo4_fp8
// K/V cache buffers (162-byte records per (block, slot, kv head)); k_cur/
// v_cur are F16 (num_tokens, n_kv_heads, head_size). Looks up the
// per-(layer, K|V) centroid LUTs and the registry's Hadamard-rotation flag
// itself, exactly as the AITER scatter call site does. Only (head_size=256,
// block_size=16) is wired; asserts otherwise.
void mt_aiter_scatter_kv_turbo4_fp8_launch(
        void * k_cache, void * v_cache,
        const __half * k_cur, const __half * v_cur,
        const int32_t * slot_mapping,
        int n_tokens, int n_kv_heads, int head_size, int block_size,
        int layer, cudaStream_t stream);

#else  // GGML_HIP_AITER undefined — stub out

inline bool aiter_backend_enabled() { return false; }
inline void ggml_cuda_op_paged_attn_mt_aiter(ggml_backend_cuda_context &, ggml_tensor *) {
    // Unreachable: caller must check aiter_backend_enabled() first.
}
inline int mt_aiter_parse_layer_from_kv_cache_name(const char *) { return -1; }
inline void mt_aiter_scatter_kv_turbo4_fp8_launch(
        void *, void *, const __half *, const __half *, const int32_t *,
        int, int, int, int, int, cudaStream_t) {
    // Unreachable: caller must check r4d_turbo4_enabled() / GGML_HIP_AITER first.
}

#endif

}  // namespace mt
