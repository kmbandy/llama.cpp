// llama-ml8-registry.h
//
// MAD-223 Phase G.12: ml8 sidecar registry + build_ml8_or_mul_mat helper.
//
// Provides a thin registry mapping `const ggml_tensor* weight` to its
// calibration sidecars (centroids, optional rotation, optional AWQ scale),
// and a helper that selects the correct ggml graph op based on weight type
// and sidecar availability.
//
// Usage:
//   ml8_registry reg;
//   reg.register_weight(w, {centroids, rotation_h_a, awq_scale});
//   ggml_tensor * y = build_ml8_or_mul_mat(ctx, reg, w, x);
//
// See aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md §2.
#pragma once

#include "ggml.h"
#include "ggml-ml8.h"

#include <unordered_map>

// ─── sidecar struct ──────────────────────────────────────────────────────────

// Per-weight calibration tensors. All fields are nullable; the caller is
// responsible for setting them to nullptr when a particular sidecar is absent.
//
// Note: rotation_meta is intentionally excluded — the dense Kronecker rotation
// derives `b` from x and h_a shapes at graph-build time and needs no stored
// metadata tensor.
struct ml8_sidecars {
    // [16, n_groups_k] GGML_TYPE_F8_E4M3 — per-K-group centroid LUT.
    // Required for ML8_4 matmul; nullptr means the ml8 path is unavailable.
    struct ggml_tensor * centroids    = nullptr;

    // [a_dim, a_dim] GGML_TYPE_F32 — Kronecker rotation factor H_a.
    // Optional. When non-null, the rotation is applied to x before matmul.
    struct ggml_tensor * rotation_h_a = nullptr;

    // Elementwise AWQ scale tensor (broadcastable over x's leading dim).
    // Optional. When non-null, applied to x before the rotation.
    struct ggml_tensor * awq_scale    = nullptr;
};

// ─── registry ────────────────────────────────────────────────────────────────

// Thin, non-owning mapping from weight tensor pointer to calibration sidecars.
// No global instance — callers (T13 model graphs) own one and pass it in.
struct ml8_registry {
    // Register sidecars for a weight. Overwrites any previous entry.
    void register_weight(const struct ggml_tensor * w, ml8_sidecars sc) {
        entries[w] = sc;
    }

    // Look up sidecars for a weight. Returns nullptr on miss.
    const ml8_sidecars * find(const struct ggml_tensor * w) const {
        auto it = entries.find(w);
        if (it == entries.end()) return nullptr;
        return &it->second;
    }

private:
    std::unordered_map<const struct ggml_tensor *, ml8_sidecars> entries;
};

// ─── helper ──────────────────────────────────────────────────────────────────

// Build a matmul graph node, dispatching to the ml8 path when appropriate.
//
// Dispatch logic:
//   - GGML_TYPE_ML8_4 + registry entry with non-null centroids:
//       Apply optional AWQ scale then optional Kronecker rotation to x, then
//       return ggml_ml8_mul_mat(ctx, weight, centroids, x_transformed).
//   - GGML_TYPE_ML8_4 but sidecars/centroids are absent:
//       GGML_ASSERT — an ML8_4 weight cannot be dispatched via plain mul_mat.
//   - GGML_TYPE_ML8_FP8:
//       return ggml_mul_mat(ctx, weight, x)   [backend auto-dispatches FP8]
//   - Any other type:
//       return ggml_mul_mat(ctx, weight, x)
//
// This is a pure function over the registry — no global state.
struct ggml_tensor * build_ml8_or_mul_mat(
        struct ggml_context  * ctx,
        const ml8_registry   & reg,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x);
