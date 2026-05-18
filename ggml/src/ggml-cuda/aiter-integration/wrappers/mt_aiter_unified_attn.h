// mt_aiter_unified_attn.h
//
// Stable C API around AITER's `kernel_unified_attention_3d` + `reduce_segments`
// AOT-compiled launchers. The launcher symbols themselves are spec-hash-named
// (e.g. uattn_3d_7de434cb_0d3d4d5d) and change when the Triton signature or
// source drifts; this wrapper is the single update site when that happens.
//
// Includable from C, C++, and .cu translation units. Pointers are passed as
// void*/int32_t*/float* and cast to hipDeviceptr_t at the call site so callers
// don't need to drag the HIP runtime headers into their public surface.
//
// Currently bound to one specialization (Qwen3.6 decode shape — head_size=128,
// 16 q-heads, 2 kv-heads, block_size=16, 32 split-K segments, ALL_DECODE=1).
// Other shapes require their own AOT spec block in CMakeLists.txt + a parallel
// wrapper instantiation.
//
// MAD-188.
#pragma once

#include <hip/hip_runtime_api.h>
#include <stdint.h>
#include <stddef.h>

// ─────────────────────────────────────────────────────────────────────────
// Spec constants — must match the SIGNATURE block in CMakeLists.txt for
// uattn_3d / uattn_reduce. Changing these here without re-AOT'ing will
// silently produce wrong output.
// ─────────────────────────────────────────────────────────────────────────
#define MT_AITER_UATTN_NUM_Q_HEADS          16
#define MT_AITER_UATTN_NUM_KV_HEADS         2
#define MT_AITER_UATTN_HEAD_SIZE            128
#define MT_AITER_UATTN_BLOCK_SIZE           16
#define MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ 32
#define MT_AITER_UATTN_BLOCK_Q              2

#ifdef __cplusplus
extern "C" {
#endif

// Argument bundle for mt_aiter_unified_attn().
//
// Tensor layouts (all device memory):
//   q           fp16    [num_q_tokens, NUM_Q_HEADS, HEAD_SIZE]
//   k_cache     fp16    [num_blocks,   BLOCK_SIZE,  NUM_KV_HEADS, HEAD_SIZE]
//   v_cache     fp16    [num_blocks,   BLOCK_SIZE,  NUM_KV_HEADS, HEAD_SIZE]
//   out         fp16    [num_q_tokens, NUM_Q_HEADS, HEAD_SIZE]
//   segm_output fp32    [num_q_tokens, NUM_Q_HEADS, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE]
//   segm_max    fp32    [num_q_tokens, NUM_Q_HEADS, NUM_SEGMENTS_PER_SEQ]
//   segm_expsum fp32    [num_q_tokens, NUM_Q_HEADS, NUM_SEGMENTS_PER_SEQ]
//   block_tables int32  [num_seqs, block_table_stride]   (paged table per seq)
//   seq_lens     int32  [num_seqs]                       (full seq len incl. ctx)
//   query_start_len int32 [num_seqs + 1]                 (cumulative q-token offsets)
//   q/k/v_descale fp32  [1]                              (scalar; pass ones for fp16 path)
//   out_scale    fp32   [1] or NULL                       (NULL → no output rescale)
struct mt_aiter_uattn_args_t {
    // I/O
    const void    *q;
    const void    *k_cache;
    const void    *v_cache;
    void          *out;
    // Workspace (caller-owned, sized via mt_aiter_uattn_*_bytes() helpers below)
    void          *segm_output;
    void          *segm_max;
    void          *segm_expsum;
    // Indexes
    const int32_t *block_tables;
    const int32_t *seq_lens;
    const int32_t *query_start_len;
    // Descale pointers — pass ones-buffer for unquantized fp16 path
    const float   *q_descale;
    const float   *k_descale;
    const float   *v_descale;
    const float   *out_scale;     // may be NULL
    // Scalars
    float          scale;          // attention softmax scale (typically 1/sqrt(head_size))
    int32_t        num_seqs;
    int64_t        block_table_stride;
    // Strides
    int64_t        q_stride_0;     // bytes per row in q = NUM_Q_HEADS * HEAD_SIZE
    int64_t        output_stride_0;
    int64_t        k_stride_0;     // block stride: BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE
    int64_t        k_stride_1;     // token stride: NUM_KV_HEADS * HEAD_SIZE
    int64_t        k_stride_2;     // head stride: HEAD_SIZE
    int64_t        v_stride_0;
    int64_t        v_stride_1;
    int64_t        v_stride_2;
};

// Launch attention: 3D split-K + reduce_segments, in stream order.
// Returns the first non-success hipError_t, or hipSuccess.
//
// The kernels do NOT pre-initialize segm_* workspace — the 3D kernel
// initializes its slice on first write. Caller is responsible only for
// allocating large enough buffers.
hipError_t mt_aiter_unified_attn(hipStream_t stream,
                                  const struct mt_aiter_uattn_args_t *args);

// Workspace sizing helpers (in bytes). All use fp32 internally.
size_t mt_aiter_uattn_segm_output_bytes(int num_q_tokens);
size_t mt_aiter_uattn_segm_max_bytes(int num_q_tokens);
size_t mt_aiter_uattn_segm_expsum_bytes(int num_q_tokens);

#ifdef __cplusplus
}  // extern "C"
#endif
