#pragma once

#include "common.cuh"
#include "ggml-backend-impl.h"

#include <cstddef>
#include <cstdint>

// Diagnostic: cudaStreamWaitEvent + record into the AR watchdog's wait ring
// (dumped on a stall). Use for any cross-stream/cross-device wait whose
// satisfaction matters for deadlock analysis.
void ggml_cuda_ar_wait_logged(cudaStream_t stream, cudaEvent_t event, const char * tag, uint64_t op, int dev);

// Opaque pipeline context -- owns all pinned buffers, streams, and events.
struct ggml_cuda_ar_pipeline;

// Allocate a pipeline for n_devices GPUs.
// devices[] holds the GPU device IDs in rank order.
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
    int       hist      = -1;   // index into the per-op event history ring
    uint64_t  op_id     = 0;    // dx_call number of this op
    int64_t   ne        = 0;
    ggml_type dst_type  = GGML_TYPE_F32;   // tensor / accumulator type
    ggml_type wire_type = GGML_TYPE_F32;   // on-wire type (BF16 for F32 inputs by default)
    void *    dst[GGML_CUDA_MAX_DEVICES] = {};

    // WP_TP_TRACE_FILE only: the rolling-loop i_slot this op belongs to (== i_op
    // in ggml_backend_cuda_comm_allreduce_begin/_end, which sets this before
    // calling ggml_cuda_ar_allreduce_begin/_end below) -- NOT the same thing
    // as `slot` above (that's the pinned-staging double-buffer slot). -1 when
    // unset (tracing disabled, or a caller that never set it); begin()/end()
    // treat that as "unknown" and omit slot/ubatch/subgraph correlation on
    // their AR_* trace rows.
    int trace_slot = -1;
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

// Read-only accessor into a live AllReduce pipeline's existing per-device
// stream (searches the pipeline registry used by ggml_cuda_ar_dump_state).
// `device` is a CUDA device id (as passed to ggml_cuda_set_device), not a TP
// rank. Returns nullptr if no internal-AllReduce pipeline is live for that
// device (single GPU, NCCL transport, or comm not yet initialized).
//
// Intended for MT_ML8_4_EXPAND_ON_AR_STREAM (ml8.cu): on a 2-hardware-queue
// device, adding a third stream for the ML8_4 expander's lookahead causes
// queue time-slicing (measured -160 pp), so the expander lookahead instead
// shares the AR stream already used for allreduce's H2D/D2H legs
// (GGML_CUDA_AR_SINGLE_STREAM). The returned stream must only be used to
// enqueue work that is fenced (via caller-managed events) against both the
// AR pipeline's own use of the stream and the compute stream's reads of
// whatever buffer the enqueued work writes -- this accessor does not
// serialize anything on its own.
cudaStream_t ggml_cuda_ar_stream_for_device(int device);

// PRINT-ONLY diagnostic: dump a host-side heartbeat of every live pipeline's
// state (call_count, pool slot/token, spin-watchdog fields, etc.) to stderr.
// Safe to call from a SIGABRT handler (see the .cu definition) -- uses only
// plain/atomic loads and snprintf+write(2), no locks, no allocation.
void ggml_cuda_ar_dump_state(const char * reason);

// mad-lab: MAD_META_GPUTIME -- per-device compute/AllReduce busy-vs-idle probe
// for the meta (tensor-parallel step-loop) backend. See the .cu definition
// for the full contract; summary here is just enough for callers.
//
// One function serves every call site (COMPUTE brackets from
// ggml-backend-meta.cpp's compute() via the "mad_meta_gputime_mark"
// proc-address, and AR_PACK/AR_XFER/AR_UNPACK brackets called directly from
// this TU around ggml_cuda_ar_allreduce_begin/_end's per-device enqueues).
//
// `backend` non-null: GPU-timed span -- an event is recorded on
// static_cast<ggml_backend_cuda_context*>(backend->context)->stream() (the
// SAME stream the bracketed work was just enqueued on), never a host wait.
// `backend` null: host-only span (HOSTWAIT) -- `device_hint` tags the row
// (-1 for a cross-device/global wait); only steady_clock is used.
//
// `phase` 0 = begin (row_id_in ignored; returns a new row id, 0 if the probe
// is disabled or the record table is full) and 1 = end (row_id_in is the id
// `phase==0` returned; return value is unused). Kinds mirror
// mad_meta_gputime::Kind in allreduce.cu: 0=COMPUTE, 1=AR_PACK, 2=AR_XFER,
// 3=AR_UNPACK, 4=HOSTWAIT.
uint64_t mad_meta_gputime_mark(ggml_backend_t backend, long long device_hint, int kind, int phase,
                                long long slot, long long subgraph, long long step, uint64_t row_id_in);

