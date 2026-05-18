// mt_aiter_unified_attn.cpp — see header for API contract.
//
// Spec-hash bindings: when CMake re-AOTs the kernels (signature change,
// AITER source bump, Triton version bump), the hash suffix changes and
// THESE INCLUDES + CALLS must be updated. Validated at 2026-05-18 with:
//   uattn_3d_7de434cb_0d3d4d5d   (kernel_unified_attention_3d, Qwen3.6 spec)
//   uattn_reduce_de7d05ae_0d1d   (reduce_segments,             Qwen3.6 spec)

#include "mt_aiter_unified_attn.h"
#include <hip/hip_runtime.h>

// Forward-declare the AOT launchers with C linkage. We deliberately do NOT
// `#include` the generated headers — they `#include <hip/hip_runtime.h>` at
// global scope (transitively pulling in C++ STL), which breaks under an
// `extern "C" { #include ... }` wrap. Manual decls sidestep that and keep
// the symbol contract in one place.
//
// When the AOT spec is bumped, update the two function names below AND the
// includes referenced in CMakeLists.txt SPECS blocks.
extern "C" {
hipError_t uattn_3d_7de434cb_0d3d4d5d(
    hipStream_t stream,
    hipDeviceptr_t segm_output_ptr, hipDeviceptr_t segm_max_ptr, hipDeviceptr_t segm_expsum_ptr,
    hipDeviceptr_t query_ptr, hipDeviceptr_t key_cache_ptr, hipDeviceptr_t value_cache_ptr,
    hipDeviceptr_t sink_ptr,
    hipDeviceptr_t block_tables_ptr, hipDeviceptr_t seq_lens_ptr,
    hipDeviceptr_t alibi_slopes_ptr, hipDeviceptr_t qq_bias_ptr,
    float scale,
    hipDeviceptr_t q_descale_ptr, hipDeviceptr_t k_descale_ptr, hipDeviceptr_t v_descale_ptr,
    float softcap,
    int64_t block_table_stride,
    int64_t query_stride_0,
    int64_t qq_bias_stride_0,
    int64_t stride_k_cache_0, int64_t stride_k_cache_1, int64_t stride_k_cache_2,
    int64_t stride_v_cache_0, int64_t stride_v_cache_1, int64_t stride_v_cache_2,
    hipDeviceptr_t query_start_len_ptr,
    int32_t num_seqs);

hipError_t uattn_reduce_de7d05ae_0d1d(
    hipStream_t stream,
    hipDeviceptr_t output_ptr,
    hipDeviceptr_t segm_output_ptr, hipDeviceptr_t segm_max_ptr, hipDeviceptr_t segm_expsum_ptr,
    hipDeviceptr_t seq_lens_ptr,
    int32_t num_seqs,
    hipDeviceptr_t out_scale_ptr,
    int64_t output_stride_0,
    int64_t block_table_stride,
    hipDeviceptr_t query_start_len_ptr);
}  // extern "C"

#define HIP_TRY(expr)                                                          \
    do {                                                                       \
        hipError_t _err = (expr);                                              \
        if (_err != hipSuccess) return _err;                                   \
    } while (0)

hipError_t mt_aiter_unified_attn(hipStream_t stream,
                                  const mt_aiter_uattn_args_t *a) {
    // qq_bias is disabled (USE_QQ_BIAS=0 constexpr in the AOT spec), so its
    // stride is ignored by the kernel. Pass 0 explicitly.
    const int64_t qq_bias_stride_0 = 0;

    HIP_TRY(uattn_3d_7de434cb_0d3d4d5d(
        stream,
        (hipDeviceptr_t)a->segm_output,
        (hipDeviceptr_t)a->segm_max,
        (hipDeviceptr_t)a->segm_expsum,
        (hipDeviceptr_t)a->q,
        (hipDeviceptr_t)a->k_cache,
        (hipDeviceptr_t)a->v_cache,
        /* sink_ptr     */ (hipDeviceptr_t)nullptr,
        (hipDeviceptr_t)a->block_tables,
        (hipDeviceptr_t)a->seq_lens,
        /* alibi_slopes */ (hipDeviceptr_t)nullptr,
        /* qq_bias      */ (hipDeviceptr_t)nullptr,
        a->scale,
        (hipDeviceptr_t)a->q_descale,
        (hipDeviceptr_t)a->k_descale,
        (hipDeviceptr_t)a->v_descale,
        /* softcap      */ 0.0f,
        a->block_table_stride,
        a->q_stride_0,
        qq_bias_stride_0,
        a->k_stride_0, a->k_stride_1, a->k_stride_2,
        a->v_stride_0, a->v_stride_1, a->v_stride_2,
        (hipDeviceptr_t)a->query_start_len,
        a->num_seqs));

    HIP_TRY(uattn_reduce_de7d05ae_0d1d(
        stream,
        (hipDeviceptr_t)a->out,
        (hipDeviceptr_t)a->segm_output,
        (hipDeviceptr_t)a->segm_max,
        (hipDeviceptr_t)a->segm_expsum,
        (hipDeviceptr_t)a->seq_lens,
        a->num_seqs,
        (hipDeviceptr_t)a->out_scale,
        a->output_stride_0,
        a->block_table_stride,
        (hipDeviceptr_t)a->query_start_len));

    return hipSuccess;
}

size_t mt_aiter_uattn_segm_output_bytes(int num_q_tokens) {
    return (size_t)num_q_tokens
         * MT_AITER_UATTN_NUM_Q_HEADS
         * MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ
         * MT_AITER_UATTN_HEAD_SIZE
         * sizeof(float);
}

size_t mt_aiter_uattn_segm_max_bytes(int num_q_tokens) {
    return (size_t)num_q_tokens
         * MT_AITER_UATTN_NUM_Q_HEADS
         * MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ
         * sizeof(float);
}

size_t mt_aiter_uattn_segm_expsum_bytes(int num_q_tokens) {
    return mt_aiter_uattn_segm_max_bytes(num_q_tokens);
}
