#pragma once

// Inner access protocol — how llama_memory_tiered talks to whatever
// inner cache it wraps (llama_kv_cache, llama_kv_cache_iswa,
// llama_memory_recurrent, llama_memory_hybrid_iswa) to read and
// modify per-layer K/V tensors and per-sequence recurrent state.
//
// Replaces the legacy `dynamic_cast` chain in
// tools/server/server-tiered-cache.cpp:201-209. Each subclass of
// llama_memory_i overrides `make_tier_view()` (added in
// src/llama-memory.h during Phase 2 integration) to return an
// InnerView populated for its own backing storage. Subclasses that
// can't be tiered (or shouldn't) leave the default empty
// implementation, which the tiered wrapper treats as "not tierable"
// (logs a warning at init and falls back to passthrough).
//
// This is a snapshot, not a live accessor. The pointers must remain
// stable for the lifetime of the inner cache or until the next
// memory_breakdown / structural change. For the standard llama_kv_cache
// implementations that holds: K/V tensors are allocated once at init.
//
// All public types live in namespace `mt` to avoid colliding with
// llama_* names elsewhere.

#include "llama.h"        // llama_seq_id, llama_pos
#include <cstddef>
#include <cstdint>
#include <vector>

struct ggml_tensor;

namespace mt {

// One attention layer's K/V tensors plus the metadata the mover needs
// to compute per-token byte offsets correctly across quantized types.
//
// `k_row_bytes` and `v_row_bytes` MUST be `tensor->nb[1]` (the row
// stride from ggml's metadata), NOT `ne[0] * element_size`. The latter
// returns block size for quantized types like turbo4 / Q4_K and gives
// wildly wrong values — this was bug B-K1 in the legacy pager.
struct AttentionLayerView {
    ggml_tensor * k          = nullptr;
    ggml_tensor * v          = nullptr;
    bool          v_trans    = false;  // true: V stored transposed (default for HIP/CUDA)
    int64_t       kv_size    = 0;      // max tokens (n_ctx_max for this slot ring)
    size_t        k_row_bytes = 0;
    size_t        v_row_bytes = 0;
};

// One sequence's recurrent state. Per the architecture, recurrent layers
// have two state tensors:
//   - r: convolutional state (delta-net's conv kernel hidden state)
//   - s: SSM state (delta-net's recurrent matrix)
//
// Some hybrid architectures only have one of these; absent ones are
// represented by null pointers in the corresponding entry.
//
// Storage model: r and s tensors are 2D `[n_embd_X, n_seqs]` (or 3D
// equivalent). One sequence's slice starts at `tensor->data + seq_slot
// * nb[1]` and is `nb[1]` bytes long. The view carries the slot index
// and per-seq byte size so the mover can compute offsets without
// re-walking ggml metadata.
//
// Migration unit is the WHOLE sequence's r+s across all recurrent
// layers. Per-layer pointers are exposed so the mover can copy without
// extra abstraction overhead.
struct RecurrentStateView {
    llama_seq_id seq_id    = 0;
    int          seq_slot  = -1;  // index into the [n_embd_X, n_seqs] tensors
    struct Layer {
        ggml_tensor * r              = nullptr;
        size_t        r_bytes_per_seq = 0;  // == r->nb[1] for a 2D [n_embd_r, n_seqs] tensor
        ggml_tensor * s              = nullptr;
        size_t        s_bytes_per_seq = 0;
    };
    std::vector<Layer> layers;
    // Total bytes-per-sequence for r and s respectively, summed across
    // layers. The mover uses these to size warm/cold buffers without
    // re-walking the layer list.
    size_t r_bytes_total = 0;
    size_t s_bytes_total = 0;
};

// Composite snapshot. Either or both fields may be empty:
//   - attn_layers empty + recur_seqs populated  -> pure-recurrent model
//     (e.g. mamba-only)
//   - attn_layers populated + recur_seqs empty  -> standard attention
//     model (e.g. llama, gemma, qwen2)
//   - both populated                            -> hybrid (qwen3.5,
//     qwen3-next, falcon-h1, nemotron-h)
struct InnerView {
    std::vector<AttentionLayerView> attn_layers;
    std::vector<RecurrentStateView> recur_seqs;

    bool empty() const { return attn_layers.empty() && recur_seqs.empty(); }
};

}  // namespace mt
