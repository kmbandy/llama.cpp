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
// Tuning constants — ported from AITER upstream host dispatcher
// (kernels/unified_attention_host_reference.py — select_2d_config /
//  select_3d_config / use_2d_kernel). These mirror upstream's values for
// RDNA + num_queries_per_kv<=16 (the Qwen3 family case).
//
// We compile THREE kernel variants and dispatch at runtime in
// mt_aiter_unified_attn():
//
//   3D split-K decode       — BLOCK_M=16, BLOCK_Q=2, NUM_SEGMENTS=32
//                             (matches upstream's base 3D config; NUM_SEGMENTS
//                             is hardcoded here, upstream computes it from
//                             cu_count — see MAD-203 phase 2)
//   2D base prefill         — BLOCK_M=16, BLOCK_Q=2, TILE_SIZE=32
//                             (used for short prefills where use_2d_kernel
//                             returns true but max_seqlen_q < 256)
//   2D large prefill        — BLOCK_M=64, BLOCK_Q=8, TILE_SIZE=32
//                             (max_seqlen_q >= 256; 4× LDS reuse vs base;
//                             this is the meat of MAD-203)
//
// BLOCK_Q in each spec must equal BLOCK_M / num_queries_per_kv. The values
// here assume num_queries_per_kv == 8 (Qwen3.5/3.6: 16 q-heads, 2 kv-heads).
// For other GQA ratios these need to be regenerated — guard added below.
//
// Dispatch (mirrors upstream use_2d_kernel; see MAD-203 phase 2 for the
// full program-count-driven heuristic):
//   if avg_q_len >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD → 2D large
//   else if avg_q_len >= MT_AITER_UATTN_BLOCK_Q            → 2D base
//   else                                                   → 3D split-K
// ─────────────────────────────────────────────────────────────────────────
#define MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ      32
#define MT_AITER_UATTN_TILE_SIZE                 32

// 3D + 2D-base spec (decode + short prefill)
#define MT_AITER_UATTN_BLOCK_Q                   2
#define MT_AITER_UATTN_BLOCK_M                   16

// 2D-large spec (max_seqlen_q >= 256). MAD-203.
#define MT_AITER_UATTN_BLOCK_M_LARGE             64
#define MT_AITER_UATTN_BLOCK_Q_LARGE             8
#define MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD   256

// KV cache element format. Selects which AOT spec / runtime-compile path
// gets dispatched. F16 keeps the upstream `*fp16:16` pointer signature;
// turbo3/turbo4 switch the K/V cache pointers to `*i8:16` byte pointers
// and bake CACHE_TYPE=1/2 as the kernel constexpr (see MAD-199 chunk A in
// kernels/unified_attention.py).
enum mt_aiter_cache_type {
    MT_AITER_CACHE_F16    = 0,

    // Legacy turbo3/turbo4 (MAD-199 v2). Dequant to FP16, FP16 attention path.
    MT_AITER_CACHE_TURBO3 = 1,
    MT_AITER_CACHE_TURBO4 = 2,

    // MAD-214: turbo-FP8 family — FP8 matrix core path via centroid LUT decode.
    // Numbering: 10..14 = turbo3_fp8 × BS{16,32,64,128,256}
    //            20..24 = turbo4_fp8 × BS{16,32,64,128,256}
    //            30..34 = turbo5_fp8 × BS{16,32,64,128,256}
    // Only the BS=256 variants are wired at MAD-214 Phase 1 ship; the others
    // are accepted by the wrapper API but throw "kernel not implemented for
    // this BLOCK_SIZE yet — calibration data + AOT spec needed" until
    // MAD-215 wires them.
    MT_AITER_CACHE_TURBO3_FP8_BS16  = 10,
    MT_AITER_CACHE_TURBO3_FP8_BS32  = 11,
    MT_AITER_CACHE_TURBO3_FP8_BS64  = 12,
    MT_AITER_CACHE_TURBO3_FP8_BS128 = 13,
    MT_AITER_CACHE_TURBO3_FP8_BS256 = 14,

    MT_AITER_CACHE_TURBO4_FP8_BS16  = 20,
    MT_AITER_CACHE_TURBO4_FP8_BS32  = 21,
    MT_AITER_CACHE_TURBO4_FP8_BS64  = 22,
    MT_AITER_CACHE_TURBO4_FP8_BS128 = 23,
    MT_AITER_CACHE_TURBO4_FP8_BS256 = 24,

    MT_AITER_CACHE_TURBO5_FP8_BS16  = 30,
    MT_AITER_CACHE_TURBO5_FP8_BS32  = 31,
    MT_AITER_CACHE_TURBO5_FP8_BS64  = 32,
    MT_AITER_CACHE_TURBO5_FP8_BS128 = 33,
    MT_AITER_CACHE_TURBO5_FP8_BS256 = 34,

    // Production aliases — the unsuffixed name resolves to the production
    // BS=256 variant (kernel = single tl.dot per Q·K^T, like vanilla FP8 attn).
    MT_AITER_CACHE_TURBO3_FP8 = MT_AITER_CACHE_TURBO3_FP8_BS256,
    MT_AITER_CACHE_TURBO4_FP8 = MT_AITER_CACHE_TURBO4_FP8_BS256,
    MT_AITER_CACHE_TURBO5_FP8 = MT_AITER_CACHE_TURBO5_FP8_BS256,
};

// Helper: returns nonzero if the cache type is part of the turbo-FP8 family.
// Cheap to inline at C call sites; the kernel-internal dispatch uses the
// numeric ranges directly.
static inline int mt_aiter_cache_is_turbo_fp8(int t) {
    return (t >= 10 && t <= 14) || (t >= 20 && t <= 24) || (t >= 30 && t <= 34);
}

// Model-shape parameters — set by the caller per model. The wrapper builds
// the Triton signature from these at first call, so the runtime registry
// can compile a kernel matched to the current shape.
struct mt_aiter_uattn_shape_t {
    int32_t head_size;       // per-head embed dim (e.g. 128)
    int32_t num_q_heads;     // attention head count
    int32_t num_kv_heads;    // GQA group count (head_count / queries_per_kv)
    int32_t block_size;      // paged-cache block size in tokens (e.g. 16)
    int32_t cache_type;      // mt_aiter_cache_type value; drives kernel selection (MAD-199)
};

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
    // Shape — must be the same across all calls in a process (kernel handles
    // are cached after first call; subsequent calls with a different shape
    // abort).
    struct mt_aiter_uattn_shape_t shape;
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
    // MAD-214: per-(kv head, current layer) centroid LUT pointers for
    // turbo-FP8 cache types. Each LUT is N_CENTROIDS uint8_t bytes
    // (positive E4M3) — caller indexes by attention-layer position before
    // each attention call. NULL for non-FP8 cache_type (F16, TURBO3/4).
    const uint8_t *centroids_k;
    const uint8_t *centroids_v;
    // Scalars
    float          scale;          // attention softmax scale (typically 1/sqrt(head_size))
    int32_t        num_seqs;
    int32_t        num_q_tokens;   // total q tokens across all seqs (= sum of q_lens). For pure decode == num_seqs.
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

// Workspace sizing helpers (in bytes). All use fp32 internally. Take the
// shape because per-token workspace = num_q_heads * NUM_SEGMENTS * head_size.
size_t mt_aiter_uattn_segm_output_bytes(const struct mt_aiter_uattn_shape_t *shape, int num_q_tokens);
size_t mt_aiter_uattn_segm_max_bytes(const struct mt_aiter_uattn_shape_t *shape, int num_q_tokens);
size_t mt_aiter_uattn_segm_expsum_bytes(const struct mt_aiter_uattn_shape_t *shape, int num_q_tokens);

#ifdef __cplusplus
}  // extern "C"
#endif
