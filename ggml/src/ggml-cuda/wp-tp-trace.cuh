#pragma once

// wp-tp-trace: WP_TP_TRACE_FILE=<path> fine-grained timeline of the rolling
// tensor-parallel prefill loop -- HOST and GPU timestamps for every
// COMPUTE_SUBMIT (subgraph dispatch), AR phase (pack/sent/recvd/unpack),
// fence wait, and ubatch boundary. Built to locate the ~1.5-2s of
// neither-compute-nor-wire time in a 7.0s / <3.5s-compute / ~4.2s-wire
// prefill that rocprofv3 cannot see (attaching it changes the AR pipeline's
// path). See summarize.py (ds4-runs/tp27b/ladder-9070-0916/tptrace) for the
// reader.
//
// Zero cost when WP_TP_TRACE_FILE is unset: wp_tp_trace_enabled() is a
// single cached bool check (see also the mirrored, equally cheap
// ggml_backend_meta_wp_tp_trace_enabled() in ggml-backend-meta.cpp, which
// gates every call site in that backend-agnostic file before it even reaches
// a proc-address call here), and every entry point below early-returns
// before doing anything else -- no allocation, no event record, no lock.
//
// Rows are appended to an in-memory buffer and flushed to the CSV in bulk --
// at WP_TP_TRACE_FLUSH_N rows (default 200000; env override), at process
// exit (atexit), or via an explicit wp_tp_trace_flush() call -- never per
// row.
//
// GPU timestamps are per-device monotonic clocks: the first time a device is
// seen, a base timing event is recorded on whatever stream is at hand and
// the host time at that moment is captured too (filed as a HEADER row).
// Every later GPU-timestamped row's gpu_t_us is resolved at flush time as
// base_host_us[device] + cudaEventElapsedTime(base_event[device], row_event) * 1000,
// which puts every device's GPU timestamps on the same (approximate) host
// time axis so summarize.py can compare across devices.
//
// AR_PACK_DONE/AR_SENT/AR_RECVD/AR_UNPACK_DONE reuse the *timing* of
// allreduce.cu's existing dx_ev events (app/sent/recvd/freed) but NOT the
// events themselves: those are created with cudaEventDisableTiming (see
// allreduce.cu's dx_ev pool), so this module records its own parallel
// timing-enabled event on the same stream at the same call site instead.

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "common.cuh"
#include "wp-tp-trace-kinds.h"

#include <cstdint>
#include <cstddef>

bool wp_tp_trace_enabled();

// Host-only row. Pass -1 for any of ubatch_idx/slot/subgraph_idx that don't
// apply to `kind`, and -1 for `device` when the row isn't device-specific
// (e.g. UBATCH_BEGIN/END). `extra` is a short free-form string (may be
// nullptr); truncated to the row's fixed extra buffer.
void wp_tp_trace_log_host(int kind, long long ubatch_idx, long long slot, long long subgraph_idx,
                           int device, const char * extra);

// GPU-timestamped row: records a NEW timing-enabled event on `stream` right
// now for `device`, then files a row whose gpu_t_us is resolved lazily at
// flush time (see the header comment above). Falls back to a host-only row
// if event creation fails or `device` is out of range.
void wp_tp_trace_log_gpu(int kind, long long ubatch_idx, long long slot, long long subgraph_idx,
                          int device, cudaStream_t stream, const char * extra);

// Cross-referencing for AR call sites: allreduce.cu/ggml-cuda.cu's AllReduce
// path knows its transport-buffer `slot` (== i_op == the rolling-loop
// i_slot in this codebase -- see ggml_backend_cuda_comm_allreduce_begin)
// but not which ubatch_idx/subgraph_idx it belongs to. The meta backend
// side (ggml-backend-meta.cpp, via wp_tp_trace_mark/_global below) notes
// those here as it logs UBATCH_BEGIN / COMPUTE_SUBMIT_BEGIN; AR call sites
// read them back with the _current_ getters. Safe to call/query even when
// tracing is disabled (both become no-ops / return -1).
void      wp_tp_trace_note_ubatch(long long slot, long long ubatch_idx);
void      wp_tp_trace_note_subgraph(long long slot, long long subgraph_idx);
long long wp_tp_trace_current_ubatch(long long slot);
long long wp_tp_trace_current_subgraph(long long slot);

// Proc-address bridges for callers that only hold a ggml_backend_t (the CUDA
// backend for one device) and never link CUDA at all -- ggml-backend-meta.cpp
// reaches these exactly the way it already reaches
// ggml_backend_cuda_set_stream_no / ggml_backend_comm_allreduce_begin: via
// ggml_backend_reg_get_proc_address(reg, "wp_tp_trace_mark" / "_global" /
// "_gpu_mark"), cast through a locally-declared function-pointer typedef
// (see ggml-backend-meta.cpp). wp_tp_trace_mark derives `device` from
// backend->context (must be a CUDA backend); wp_tp_trace_gpu_mark
// additionally derives the stream (cuda_ctx->stream(), i.e. whichever slot
// stream WP_META_SLOT_STREAMS last selected via set_stream_no) and records a
// GPU event on it. wp_tp_trace_mark_global needs no backend at all (device
// is forced to -1) -- used for the UBATCH_BEGIN/END rows, which aren't
// per-device.
void wp_tp_trace_mark(ggml_backend_t backend, int kind, long long ubatch_idx, long long slot,
                       long long subgraph_idx, const char * extra);
void wp_tp_trace_mark_global(int kind, long long ubatch_idx, long long slot, long long subgraph_idx,
                              const char * extra);
void wp_tp_trace_gpu_mark(ggml_backend_t backend, int kind, long long ubatch_idx, long long slot,
                           long long subgraph_idx, const char * extra);

// Force a flush now (resolves any pending GPU events, appends to
// WP_TP_TRACE_FILE, clears the in-memory buffer). No-op when tracing is
// disabled. Also registered via atexit() the first time tracing is used, and
// invoked automatically once the in-memory buffer hits WP_TP_TRACE_FLUSH_N
// rows.
void wp_tp_trace_flush();
