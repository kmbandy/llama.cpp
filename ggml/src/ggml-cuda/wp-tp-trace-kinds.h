#pragma once

// Row kinds for WP_TP_TRACE_FILE (see wp-tp-trace.cuh for the full contract
// and summarize.py for the reader). Deliberately dependency-free -- no
// ggml.h, no vendor/CUDA types -- so both the CUDA-side implementation
// (wp-tp-trace.cu, allreduce.cu, ggml-cuda.cu) and callers that only reach it
// through a resolved proc-address and never link CUDA at all
// (ggml-backend-meta.cpp) can share the same numeric kind without either
// side pulling in the other's headers.
enum wp_tp_trace_kind {
    WP_TPT_UBATCH_BEGIN = 0,
    WP_TPT_UBATCH_END,
    WP_TPT_COMPUTE_SUBMIT_BEGIN,
    WP_TPT_COMPUTE_SUBMIT_END,
    WP_TPT_COMPUTE_GPU_START,
    WP_TPT_COMPUTE_GPU_END,
    WP_TPT_AR_BEGIN_HOST,
    WP_TPT_AR_END_HOST,
    WP_TPT_AR_PACK_DONE,
    WP_TPT_AR_SENT,
    WP_TPT_AR_RECVD,
    WP_TPT_AR_UNPACK_DONE,
    WP_TPT_FENCE_WAIT,
    WP_TPT_HEADER, // internal only: per-device base-event row, emitted by wp-tp-trace.cu itself
};
