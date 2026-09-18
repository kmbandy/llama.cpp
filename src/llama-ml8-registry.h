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

#include <cstdint>
#include <functional>
#include <unordered_map>

// ─── sidecar struct ──────────────────────────────────────────────────────────

// Per-weight calibration tensors. All ggml_tensor* fields are nullable; the
// caller is responsible for setting them to nullptr when a particular
// sidecar is absent.
struct ml8_sidecars {
    // [16, n_groups_k] GGML_TYPE_F8_E4M3 — per-K-group centroid LUT.
    // Required for ML8_4 matmul; nullptr means the ml8 path is unavailable.
    // Always nullptr for ML8_FP8 and FP8_B128 (neither GEMM has a LUT — see
    // ml8.cuh's ggml_cuda_op_ml8_fp8_mul_mat and the FP8_B128 phase 2 design).
    struct ggml_tensor * centroids    = nullptr;

    // [a_dim, a_dim] GGML_TYPE_F32 — kronecker_orth_sylvester rotation factor
    // H_a. Optional. When non-null, ggml_ml8_apply_rotation(..., h_a, a, b)
    // (ML8_FP8) or ggml_fp8_quant_rot(..., h_a, a, b, KIND_KRONECKER)
    // (FP8_B128) is applied to x before matmul, with a = rotation_h_a->ne[0].
    struct ggml_tensor * rotation_h_a = nullptr;

    // Elementwise AWQ scale tensor (broadcastable over x's leading dim).
    // Optional. When non-null, applied to x before the rotation.
    struct ggml_tensor * awq_scale    = nullptr;

    // Host-side copy of rotation_meta[1] (b_dim), read ONCE at model load
    // time (see qwen35.cpp's read_rotation_meta) — never read from the
    // sidecar tensor's device data at graph-build time. 0 means "no
    // rotation_meta sidecar": for the kronecker path this means derive
    // b = x->ne[0] / a_dim (legacy ml8-4 GGUFs, backward compatible);
    // block_hadamard requires this to be set (there is no h_a to derive
    // a_dim from).
    int64_t rotation_b_dim = 0;

    // True when rotation_meta's kind_id was GGML_ML8_ROTATION_KIND_
    // BLOCK_HADAMARD (rotation_h_a is null in this case — Q = I_a ⊗ H_b,
    // no H_a leg). Mutually exclusive with rotation_h_a being non-null.
    bool rotation_block_hadamard = false;
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

    // Mutable lookup, for the post-load pass that canonicalizes a group's
    // rotation_h_a pointer (llama_model_validate_fp8_rotation_groups): the
    // quant_rot memo key is the h_a POINTER, so group members that carry
    // byte-identical but separately loaded h_a tensors must be pointed at
    // one of them or the shared input gets rotated+quantized once per member.
    ml8_sidecars * find_mut(const struct ggml_tensor * w) {
        auto it = entries.find(w);
        if (it == entries.end()) return nullptr;
        return &it->second;
    }

private:
    std::unordered_map<const struct ggml_tensor *, ml8_sidecars> entries;
};

// ─── FP8_B128 quant-rot memo ─────────────────────────────────────────────────

// FP8_B128 phase 2: every weight in an "input group" (e.g. {attn_qkv,
// attn_gate}, {ffn_gate, ffn_up}) consumes the SAME activation tensor and,
// per the converter invariant, was rotated with the SAME rotation — so the
// fused rotate+quantize (GGML_OP_FP8_QUANT_ROT) only needs to be computed
// once per graph build and shared by every mul_mat of the group. This key
// identifies "the same quant_rot call" — two calls with an identical key
// produce byte-identical output, so the second is served from the memo
// instead of emitting a duplicate graph node.
//
// The memo itself (an instance of fp8_qrot_memo below) is owned by the
// caller (llm_graph_context — see llama-graph.h) and MUST be reset once per
// graph build: llama_model::build_graph constructs a fresh
// unique_ptr<llm_graph_context> for every build, so a plain non-static
// member there resets itself automatically. Do not make this map static or
// hang it off the (long-lived, per-model) ml8_registry — that would leak
// stale tensor pointers across graph builds.
// G (the activation scale-group width, 32 for ML8_FP8 weights / 128 for
// FP8_B128 weights) is part of the key: the two weight formats can in
// principle share the same raw input tensor `x` (e.g. during an A/B
// migration) but must never share a quant_rot node across them, since their
// packed row layouts differ.
struct fp8_qrot_key {
    const struct ggml_tensor * x   = nullptr; // the raw input tensor (pre-AWQ)
    const struct ggml_tensor * h_a = nullptr; // nullptr for NONE/BLOCK_HADAMARD
    int64_t a_dim = 0;
    int64_t b_dim = 0;
    int32_t kind  = 0;
    int32_t G     = 128;

    bool operator==(const fp8_qrot_key & o) const {
        return x == o.x && h_a == o.h_a && a_dim == o.a_dim && b_dim == o.b_dim && kind == o.kind && G == o.G;
    }
};

struct fp8_qrot_key_hash {
    size_t operator()(const fp8_qrot_key & k) const noexcept {
        size_t h = std::hash<const void *>()(k.x);
        h = h * 1000003u ^ std::hash<const void *>()(k.h_a);
        h = h * 1000003u ^ std::hash<int64_t>()(k.a_dim);
        h = h * 1000003u ^ std::hash<int64_t>()(k.b_dim);
        h = h * 1000003u ^ std::hash<int32_t>()(k.kind);
        h = h * 1000003u ^ std::hash<int32_t>()(k.G);
        return h;
    }
};

using fp8_qrot_memo = std::unordered_map<fp8_qrot_key, struct ggml_tensor *, fp8_qrot_key_hash>;

// ─── helper ──────────────────────────────────────────────────────────────────

// Build a matmul graph node, dispatching to the ml8 path when appropriate.
//
// Dispatch logic:
//   - GGML_TYPE_ML8_4 + registry entry with non-null centroids (default,
//     MT_ML8_4_ACT unset/not "legacy", read once via getenv):
//       Same memoized quant_rot + ML8_MUL_MAT(prequantized) shape as the
//       FP8_B128/ML8_FP8 quant_rot path below, but producing an
//       GGML_OP_ML8_MUL_MAT node instead of GGML_OP_FP8_MUL_MAT: apply the
//       optional AWQ scale, then build/reuse (via `qrot_memo`, keyed by
//       fp8_qrot_key on the ORIGINAL `x` with G=0/per-row) a single
//       ggml_fp8_quant_rot(..., G=0) node per input group (kind KRONECKER if
//       rotation_h_a is set, BLOCK_HADAMARD if rotation_block_hadamard is
//       set, else NONE), and return
//       ggml_ml8_mul_mat(ctx, weight, centroids, qrot) — the pre-quantized
//       I8 activation contract (see ggml-ml8.h). This collapses the
//       activation pipeline for the input group to ONE launch (the fused
//       rotate+quantize kernel) shared by every ML8_4/ML8_FP8/FP8_B128
//       matmul of that group, vs. the legacy path's separate
//       ML8_APPLY_ROTATION (f32->f32, its own launch(es) + D2D copy) followed
//       by ML8_MUL_MAT's own internal quantize pass.
//   - GGML_TYPE_ML8_4 (MT_ML8_4_ACT=legacy):
//       The pre-existing behavior, kept for A/B comparison against the
//       quant_rot path above: apply optional AWQ scale then optional
//       rotation (kronecker if rotation_h_a is set, else block_hadamard if
//       rotation_block_hadamard is set) to x via apply_ml8_input_xform
//       (ggml_ml8_apply_rotation, not quant_rot), then return
//       ggml_ml8_mul_mat(ctx, weight, centroids, x_transformed) with a raw
//       F32 activation (the GEMM quantizes internally).
//   - GGML_TYPE_ML8_4 but sidecars/centroids are absent:
//       GGML_ASSERT — an ML8_4 weight cannot be dispatched via plain mul_mat.
//   - GGML_TYPE_ML8_FP8 (default, WP_ML8_FP8_LEGACY unset/0):
//       Same memoized quant_rot + fp8_mul_mat path as FP8_B128 below, but
//       with G=32 (ML8_FP8's 34-byte, 32-wide-K-group blocks) — i.e. apply
//       the optional AWQ scale, then build/reuse (via `qrot_memo`, keyed by
//       fp8_qrot_key on the ORIGINAL `x` with G=32) a single
//       ggml_fp8_quant_rot(..., G=32) node per input group, and return
//       ggml_fp8_mul_mat(ctx, weight, qrot). A registry miss uses
//       GGML_FP8_QUANT_ROT_KIND_NONE, same as FP8_B128.
//   - GGML_TYPE_ML8_FP8 (WP_ML8_FP8_LEGACY=1, read once via getenv):
//       The pre-existing behavior, kept for A/B comparison against the
//       quant_rot path above: apply the optional AWQ+rotation transform via
//       apply_ml8_input_xform (ggml_ml8_apply_rotation, not quant_rot), then
//       return ggml_mul_mat(ctx, weight, x_xf). Registry miss or a registry
//       entry with no rotation info is a plain ggml_mul_mat(ctx, weight, x).
//   - GGML_TYPE_FP8_B128:
//       Apply the optional AWQ scale (as today), then build/reuse (via
//       `qrot_memo`, keyed by fp8_qrot_key on the ORIGINAL `x` — i.e. before
//       AWQ — with G picked by env `MT_FP8_B128_LAYOUT`, read once: "rdna4"
//       (default) -> G=0, per-row, matching the frozen gfx1201 GEMM kernel's
//       fixed contract; "generic"/"preshuffle" -> G=128, the historical
//       grouped layout) a single ggml_fp8_quant_rot node per input group,
//       named "<weight>.qrot" the first time it's created, and return
//       ggml_fp8_mul_mat(ctx, weight, qrot). A registry miss (no rotation
//       sidecars) uses GGML_FP8_QUANT_ROT_KIND_NONE. `qrot_memo == nullptr`
//       disables sharing (every call builds its own node) — used by the
//       single-shot output-projection path in llama-context.cpp where there
//       is only one weight and nothing to share with. The memo key includes
//       G, so switching MT_FP8_B128_LAYOUT never collides quant_rot nodes
//       across the two layouts.
//   - Any other type:
//       return ggml_mul_mat(ctx, weight, x)
//
// This is a pure function over the registry (and, for FP8_B128, the caller-
// supplied per-build memo) — no global state.
// out_type (LLAMA_ACT_BF16, 2026-09-18 phase 2): GGML_TYPE_F32 (default,
// byte-identical to every existing caller) or GGML_TYPE_BF16 to request a
// bf16 dst from the ML8_4/FP8_B128/ML8_FP8 GEMM (via ggml_ml8_mul_mat_bf16 /
// ggml_fp8_mul_mat_bf16 -- see ggml-ml8.h). Only honored for those three
// weight types; the "any other type" plain-ggml_mul_mat fallback always
// returns f32 regardless of out_type (asserted below), since plain mul_mat
// has no bf16-dst variant here. Callers requesting bf16 must independently
// confirm the CUDA backend's supports_op will actually accept it for the
// resulting op's shape (ggml_cuda_ml8_4_mul_mat_supports_bf16_out et al. in
// ggml-cuda.cu) -- this function does not check device support itself.
struct ggml_tensor * build_ml8_or_mul_mat(
        struct ggml_context  * ctx,
        const ml8_registry   & reg,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x,
        fp8_qrot_memo        * qrot_memo = nullptr,
        enum ggml_type         out_type = GGML_TYPE_F32);
