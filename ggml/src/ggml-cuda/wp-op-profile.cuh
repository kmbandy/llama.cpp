#pragma once

// wp-op-profile: WP_OP_PROFILE=1 per-op GPU time accounting for eagerly
// launched graph nodes. Each launched node gets a (start,end) event pair on
// its stream; pairs are drained lazily (never synchronously) at the next
// graph_compute entry and accumulated per (device, token-bucket, op-key).
// A summary is printed to stderr every WP_OP_PROFILE_S seconds (default 5)
// and the accumulators reset, so a summary covers roughly one prefill or
// one stretch of decode. Nodes launched inside a HIP-graph capture are not
// timed (an event recorded mid-capture is captured too); replayed graphs
// are accounted as one "graph-replay" entry per launch instead.
//
// Zero cost when the env var is unset: every entry point early-returns on
// a cached bool.

#include "ggml.h"
#include "common.cuh"

bool wp_op_profile_enabled();

// Call once per ggml_backend_cuda_graph_compute() before the node loop.
// Drains finished pairs, picks the token bucket for this graph, prints the
// periodic summary.
void wp_op_profile_begin_graph(int device, const ggml_cgraph * cgraph);

// Wrap one eager node launch (or one fused group, `node` being its head).
// begin() records the start event on `stream`; end() records the end event
// and files the pair. No-ops when in_capture.
void wp_op_profile_begin_node(int device, cudaStream_t stream, bool in_capture);
void wp_op_profile_end_node(int device, cudaStream_t stream, const ggml_tensor * node, int n_fused, bool in_capture);

// Wrap a cudaGraphLaunch replay of the whole cgraph.
void wp_op_profile_begin_replay(int device, cudaStream_t stream);
void wp_op_profile_end_replay(int device, cudaStream_t stream);

// Generic spans (e.g. AllReduce phases): begin records an event on the stream,
// end records another and accounts the elapsed time under `key` (bucketed like
// ops). Spans do not participate in GAP accounting.
void wp_op_profile_span_begin(int device, cudaStream_t stream);
void wp_op_profile_span_end(int device, cudaStream_t stream, const char * key);
