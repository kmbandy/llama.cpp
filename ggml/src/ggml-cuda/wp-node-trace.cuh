#pragma once

// wp-node-trace: WP_NODE_TRACE=1 diagnostic ring of the last N graph nodes
// launched per device, so ggml_cuda_ar_watchdog_dump() (allreduce.cu) can
// report which node a stuck device was last executing.
//
// Zero cost when the env var is unset: wp_node_trace_enabled() is a single
// cached bool check, and both entry points early-return before touching any
// per-device state.
//
// Thread-safety: wp_node_trace_record() is called from the (single) compute
// thread for a given device; wp_node_trace_dump() is called concurrently
// from the watchdog thread. The ring uses a std::atomic<uint32_t> head index
// per device and no locks -- entries read by the dumper are best-effort and
// may be torn/stale, which is fine for a diagnostic.

#include "ggml.h"
// cudaEvent_t/cudaStream_t here are whatever common.cuh's vendor header
// (vendors/hip.h on a HIP build, vendors/cuda.h otherwise) maps them to --
// do NOT include <cuda_runtime.h> directly, it does not exist on a HIP-only
// toolchain and would define the wrong (non-hipified) types anyway.
#include "common.cuh"

// Cached getenv("WP_NODE_TRACE") == "1", checked once.
bool wp_node_trace_enabled();

// Record that `node` (index `node_idx` within its cgraph) was just launched
// on `stream` for `device`. Must be called right after the node's kernel(s)
// are launched, on the same stream they were launched on. `in_capture` must
// be true iff this launch is happening while `stream` is being captured into
// a CUDA/HIP graph (i.e. use_cuda_graph && cuda_graph_update_required in
// ggml_cuda_graph_evaluate_and_capture()) -- an event recorded on a stream
// mid-capture would itself be captured into the graph and fire on every
// future replay, which is not what we want, so no event is recorded for
// in-capture launches; the ring entry is still pushed, marked "graph-replay".
//
// No-op when wp_node_trace_enabled() is false.
void wp_node_trace_record(int device, cudaStream_t stream, const ggml_tensor * node, int node_idx, bool in_capture);

// Print one "wp ar-watchdog: node-trace ..." line to stderr for `device`,
// summarizing the newest/oldest-still-pending entries in its ring. Uses
// cudaEventQuery only -- never CUDA_CHECK, since this runs from the watchdog
// thread and must never abort the process while diagnosing a stall. Caller
// must have already done ggml_cuda_set_device(device) (or does not care
// which device is current -- this function does not change it).
//
// No-op when wp_node_trace_enabled() is false.
void wp_node_trace_dump(int device);
