// mt_sparse_attn_dsv4.h
//
// Stable C API around AITER's native sparse DSv4 attention kernels
// (kernels/sparse_attention_dsv4.py's _sparse_attn_prefill_kernel and
// kernels/pa_decode_sparse.py's _pa_decode_sparse, KV_SPLITS=1 -- see
// scratchpad/sparse-attn/bench.py and test_aot_ctypes.py for the Stage 1/2
// benchmark + standalone-launch validation this wrapper's configs came from).
//
// Bound to ONE model shape: head_size=512, num_heads=64, K==V latent MQA
// (1 kv "head"). The caller (ggml-cuda's ggml_cuda_op_sparse_attn_dsv4) must
// verify this against the op's tensors before calling in -- see
// ggml_cuda_sparse_attn_dsv4_supported() in mt_sparse_attn_dsv4.cuh, and
// build_attention_v41's own build-time check against DS4.1's hparams.
//
// Dispatch: prefill kernel for n_tokens >= MT_SPARSE_ATTN_DSV4_PREFILL_MIN_T,
// otherwise the decode (KV_SPLITS=1, single-CTA) kernel -- both were
// benchmarked correct and fast at the shapes on either side of that
// threshold (Stage 1: prefill T=256/4096, decode T=1/2/8).
//
// WP_DSV41_SPARSE_ATTN.
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>

#define MT_SPARSE_ATTN_DSV4_HEAD_DIM        512
#define MT_SPARSE_ATTN_DSV4_NUM_HEADS       64
#define MT_SPARSE_ATTN_DSV4_PREFILL_MIN_T   32

#ifdef __cplusplus
extern "C" {
#endif

// Tensor layouts (all device memory, contiguous, matching ggml_sparse_attn_dsv4's
// contract in ggml.h):
//   q           fp16  [n_tokens, num_heads, head_dim]     (ggml ne order reversed: head_dim fastest)
//   k_all       fp16  [n_kv,     head_dim]                (K==V; raw window rows first, then compressed)
//   kv_indices  i32   [n_tokens, n_idx]                   (-1 = pad/skip; ggml ne[0]=n_idx fastest)
//   attn_sink   fp32  [num_heads]
//   out         fp16  [n_tokens, num_heads, head_dim]
struct mt_sparse_attn_dsv4_args_t {
    const void    * q;
    const void    * k_all;
    const int32_t * kv_indices;
    const int32_t * kv_indptr; // [n_tokens+1] row offsets, built in-graph (graph-capture safe)
    const float   * attn_sink;
    void          * out;
    int32_t         n_tokens;
    int32_t         n_kv;      // k_all's row count (raw + compressed), for the sparse-index bounds check
    int32_t         n_idx;     // kv_indices row width (DS4.1: window + top_k, e.g. 640)
    float           scale;
};

// Launch attention. Returns the first non-success hipError_t, or hipSuccess.
// Aborts (GGML_ASSERT-style, via stderr + std::abort) if the compiled-in
// shape assumptions (head_dim==512, num_heads==64) don't match -- the caller
// is expected to have already checked this via
// ggml_cuda_sparse_attn_dsv4_supported() and never call in otherwise.
hipError_t mt_sparse_attn_dsv4(hipStream_t stream,
                                const struct mt_sparse_attn_dsv4_args_t * args);

#ifdef __cplusplus
}  // extern "C"
#endif
