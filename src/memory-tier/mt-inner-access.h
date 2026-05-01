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
//
// Layer view is *just the K/V tensors*. Slot occupancy (which slot
// holds which position) is shared across every layer in the same
// physical cache and lives on the parent AttentionCacheSnapshot.
struct AttentionLayerView {
    ggml_tensor * k          = nullptr;
    ggml_tensor * v          = nullptr;
    bool          v_trans    = false;  // true: V stored transposed (default for HIP/CUDA)
    size_t        k_row_bytes = 0;
    size_t        v_row_bytes = 0;

    // kv_size is duplicated from the parent AttentionCacheSnapshot. Kept
    // here because AttentionMover takes a layer view by reference and
    // needs the slot count for bounds checks and (for transposed V) the
    // pitch in hipMemcpy2D. All layers in one cache share the same value.
    int64_t       kv_size    = 0;
};

// Per-cell occupancy snapshot. Captured once per make_tier_view call;
// reflects the cache state at that moment. Index in the parent
// AttentionCacheSnapshot::cells vector IS the slot index.
//
// pos == -1 means the slot is empty. seq_id == -1 means either empty
// or "multi-seq cell" (the underlying cell ring can hold cells shared
// by multiple seqs; the wrapper picks the primary seq for tier
// purposes). Tier eviction only acts on cells with pos >= 0 AND a
// single owner seq — cells shared across seqs are conservatively left
// alone.
struct CellSnapshot {
    llama_pos    pos    = -1;
    llama_seq_id seq_id = -1;
};

// One physical attention cache. For a non-SWA model this is the only
// AttentionCacheSnapshot in InnerView::attn_caches. For an iswa model
// there are two: one with is_swa=false (the base cache, full-attention
// layers, restorable) and one with is_swa=true (the SWA cache, layers
// that attend only within a rolling window, NOT restorable because
// the SWA mask hides positions older than n_swa back from the head).
//
// Tier eviction must consult is_swa: for backup it can copy from base
// (the SWA layers are a subset of those positions) and for restore it
// must only fill base cache slots. Restoring into an SWA cache silently
// fails — the cache holds the data but the mask makes it invisible.
struct AttentionCacheSnapshot {
    std::vector<AttentionLayerView> layers;        // K/V per layer in this cache
    std::vector<CellSnapshot>       cells;         // index = slot, value = (pos, seq_id)
    int64_t                         kv_size = 0;   // == cells.size()
    bool                            is_swa  = false;
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
//   - attn_caches empty + recur_seqs populated   -> pure-recurrent
//     model (e.g. mamba-only)
//   - attn_caches populated + recur_seqs empty   -> standard attention
//     model. Single cache for non-SWA (e.g. llama, qwen2). Two caches
//     (base + swa) for iswa (e.g. qwen3.6 SWA).
//   - both populated                             -> hybrid (qwen3.5,
//     qwen3-next, falcon-h1, nemotron-h).
struct InnerView {
    std::vector<AttentionCacheSnapshot> attn_caches;
    std::vector<RecurrentStateView>     recur_seqs;

    bool empty() const { return attn_caches.empty() && recur_seqs.empty(); }

    // Total attention layer count across all caches. Used by the
    // wrapper for warm staging sizing and progress logging.
    size_t attn_layer_count() const {
        size_t n = 0;
        for (const auto & c : attn_caches) n += c.layers.size();
        return n;
    }
};

}  // namespace mt
