#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the CUDA device IDs in rank order.
// Returns nullptr on allocation failure.
ggml_cuda_ar_pipeline * ggml_cuda_ar_pipeline_init(
    const int * devices, size_t n_devices);

// Release all resources owned by the pipeline.
void ggml_cuda_ar_pipeline_free(ggml_cuda_ar_pipeline * pipeline);

// Split AllReduce: begin() enqueues the wire transfer (side streams + copy
// engines) and returns immediately; end() makes the compute streams wait for
// the peer data and runs the add kernel.  ggml_cuda_ar_allreduce() below is
// begin()+end().  A caller may hold at most two begun-but-not-ended ops
// (double-buffered receive/staging), which is what the meta backend's
// two-ubatch overlap needs.  When the pipeline's transport cannot run a
// given call asynchronously (legacy copy-engine transport, chunked-kernel
// size class, oversize payload) begin() performs the whole reduce and marks
// the op not pending; end() is then a no-op.
struct ggml_cuda_ar_op {
    bool      pending   = false;
    int       slot      = -1;
    int64_t   ne        = 0;
    ggml_type dst_type  = GGML_TYPE_F32;   // tensor / accumulator type
    ggml_type wire_type = GGML_TYPE_F32;   // on-wire type (BF16 for F32 inputs by default)
    void *    dst[GGML_CUDA_MAX_DEVICES] = {};
};

bool ggml_cuda_ar_allreduce_begin(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors,
    ggml_cuda_ar_op       * op);

bool ggml_cuda_ar_allreduce_end(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_cuda_ar_op       * op);

// Execute an in-place AllReduce (sum) across tensors[0..n_devices-1].
// tensors[i] must live on the device managed by backends[i] and be
// contiguous F32, F16, or BF16.
// Preconditions are checked by the CUDA comm dispatcher before calling this.
// Returns true once the reduction work has been enqueued successfully.
bool ggml_cuda_ar_allreduce(
    ggml_cuda_ar_pipeline * pipeline,
    ggml_backend_t        * backends,
    ggml_tensor           ** tensors);

