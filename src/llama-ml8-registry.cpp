// llama-ml8-registry.cpp
//
// MAD-223 Phase G.12: ml8 sidecar registry + build_ml8_or_mul_mat helper.
//
// See llama-ml8-registry.h for the public contract.

#include "llama-ml8-registry.h"

#include "ggml.h"
#include "ggml-ml8.h"

#include <cstdlib>
#include <string>

// WP_ML8_FP8_LEGACY=1 keeps ML8_FP8 weights on the pre-quant_rot dispatch
// path (ggml_ml8_apply_rotation + plain ggml_mul_mat) for A/B comparison
// against the new memoized GGML_OP_FP8_QUANT_ROT(G=32) + GGML_OP_FP8_MUL_MAT
// path that FP8_B128 already uses. Read once (env vars don't change mid-run).
static bool ml8_fp8_legacy_enabled() {
    static const bool legacy = [] {
        const char * v = std::getenv("WP_ML8_FP8_LEGACY");
        return v != nullptr && v[0] == '1';
    }();
    return legacy;
}

// MT_FP8_B128_LAYOUT selects the activation quant_rot scale-group width used
// for FP8_B128 weights: "rdna4" (default) -> G=0 (per-row, the frozen gfx1201
// GEMM kernel's fixed contract: A fp8 [M,K] contiguous, a_scale fp32 [M]);
// "generic" or "preshuffle" -> G=128 (the historical grouped layout, for
// backends without the RDNA4 kernel). Read once (env vars don't change
// mid-run). Unrecognized values fall back to the rdna4 (G=0) default.
static int32_t fp8_b128_layout_G() {
    static const int32_t G = [] {
        const char * v = std::getenv("MT_FP8_B128_LAYOUT");
        if (v != nullptr && (std::string(v) == "generic" || std::string(v) == "preshuffle")) {
            return int32_t(128);
        }
        return int32_t(0); // "rdna4" (default) and anything unrecognized
    }();
    return G;
}

// MT_ML8_4_ACT=legacy keeps ML8_4 weights on the pre-quant_rot dispatch path
// (apply_ml8_input_xform + ggml_ml8_mul_mat with a raw F32 activation, the
// GEMM quantizing internally) for A/B comparison against the default
// GGML_OP_FP8_QUANT_ROT(G=0) + GGML_OP_ML8_MUL_MAT(prequantized) path. Read
// once (env vars don't change mid-run).
static bool ml8_4_act_legacy_enabled() {
    static const bool legacy = [] {
        const char * v = std::getenv("MT_ML8_4_ACT");
        return v != nullptr && std::string(v) == "legacy";
    }();
    return legacy;
}

// Apply the optional AWQ scale then the optional rotation to `x`, shared by
// both the ML8_4 and ML8_FP8 dispatch branches below. Kind is picked by which
// of rotation_h_a / rotation_block_hadamard is set (they're mutually
// exclusive — see ml8_sidecars in llama-ml8-registry.h); neither set means
// no rotation. `b` prefers the rotation_meta-derived rotation_b_dim (when
// present) over deriving b = x->ne[0] / a_dim, which keeps behavior
// unchanged for ml8-4 GGUFs predating rotation_meta.
static struct ggml_tensor * apply_ml8_input_xform(
        struct ggml_context  * ctx,
        struct ggml_tensor   * x,
        const ml8_sidecars   & sc) {
    struct ggml_tensor * x_xf = x;

    if (sc.awq_scale) {
        x_xf = ggml_mul(ctx, x_xf, sc.awq_scale);
    }

    if (sc.rotation_h_a) {
        const int64_t a = sc.rotation_h_a->ne[0];
        const int64_t b = sc.rotation_b_dim > 0 ? sc.rotation_b_dim : x_xf->ne[0] / a;
        x_xf = ggml_ml8_apply_rotation(ctx, x_xf, sc.rotation_h_a, a, b);
    } else if (sc.rotation_block_hadamard) {
        GGML_ASSERT(sc.rotation_b_dim > 0 && "block_hadamard rotation missing b_dim (rotation_meta)");
        const int64_t b = sc.rotation_b_dim;
        const int64_t a = x_xf->ne[0] / b;
        x_xf = ggml_ml8_apply_rotation(ctx, x_xf, nullptr, a, b);
    }

    return x_xf;
}

// Shared by build_fp8_quant_rot_mul_mat (below) and build_ml8_quant_rot_mul_mat
// (ML8_4's default activation path): apply the optional AWQ scale, derive the
// FP8_QUANT_ROT kind/a_dim/b_dim from `sc` the same way for every weight
// format, then build (or reuse, via `qrot_memo`) the ggml_fp8_quant_rot node
// for this input group. Returns the qrot I8 tensor; callers wrap it in
// whichever *_mul_mat op matches their weight's dequant contract.
//
// Memo key uses the ORIGINAL `x` (pre-AWQ) and includes G, matching the
// design: group members share one activation and (per the load-time
// assertion in qwen35.cpp) an identical rotation, so any member reaching
// this function first builds the shared node; G keeps FP8_B128, ML8_FP8 and
// ML8_4 quant_rot nodes from ever colliding even if they somehow shared the
// same raw `x` (ML8_4 always uses G=0/per-row — see build_ml8_quant_rot_mul_mat).
static struct ggml_tensor * get_or_build_fp8_quant_rot(
        struct ggml_context  * ctx,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x,
        const ml8_sidecars   * sc,
        fp8_qrot_memo        * qrot_memo,
        int32_t                G) {
    // AWQ acts on the raw activation, same as apply_ml8_input_xform — apply
    // it BEFORE building/looking up the quant_rot node. Not emitted by the
    // fp8_b128/ml8_4 converters today (see the design docs), so this is dead
    // code for the shipping Qwen3.8-27B recipe, but kept for parity with
    // apply_ml8_input_xform's contract.
    struct ggml_tensor * x_xf = (sc && sc->awq_scale) ? ggml_mul(ctx, x, sc->awq_scale) : x;

    int32_t kind  = GGML_FP8_QUANT_ROT_KIND_NONE;
    struct ggml_tensor * h_a = nullptr;
    int64_t a_dim = 0;
    int64_t b_dim = 0;

    if (sc && sc->rotation_h_a) {
        h_a   = sc->rotation_h_a;
        a_dim = h_a->ne[0];
        b_dim = sc->rotation_b_dim > 0 ? sc->rotation_b_dim : x_xf->ne[0] / a_dim;
        kind  = GGML_FP8_QUANT_ROT_KIND_KRONECKER;
    } else if (sc && sc->rotation_block_hadamard) {
        GGML_ASSERT(sc->rotation_b_dim > 0 && "block_hadamard rotation missing b_dim (rotation_meta)");
        b_dim = sc->rotation_b_dim;
        a_dim = x_xf->ne[0] / b_dim;
        kind  = GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD;
    }

    const fp8_qrot_key key{ x, h_a, a_dim, b_dim, kind, G };

    struct ggml_tensor * qrot = nullptr;
    if (qrot_memo) {
        auto it = qrot_memo->find(key);
        if (it != qrot_memo->end()) {
            qrot = it->second;
        }
    }
    if (qrot == nullptr) {
        qrot = ggml_fp8_quant_rot(ctx, x_xf, h_a, a_dim, b_dim, kind, G);
        const std::string qrot_name = std::string(weight->name) + ".qrot";
        ggml_set_name(qrot, qrot_name.c_str());
        if (qrot_memo) {
            (*qrot_memo)[key] = qrot;
        }
    }

    return qrot;
}

// Shared quant_rot + fp8_mul_mat dispatch for both FP8_B128 (G=128) and the
// non-legacy ML8_FP8 path (G=32). See build_ml8_or_mul_mat's doc comment in
// llama-ml8-registry.h for the full contract.
static struct ggml_tensor * build_fp8_quant_rot_mul_mat(
        struct ggml_context  * ctx,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x,
        const ml8_sidecars   * sc,
        fp8_qrot_memo        * qrot_memo,
        int32_t                G) {
    struct ggml_tensor * qrot = get_or_build_fp8_quant_rot(ctx, weight, x, sc, qrot_memo, G);
    return ggml_fp8_mul_mat(ctx, weight, qrot);
}

// ML8_4's default activation path (MAD-3xx): same memoized quant_rot shape
// as build_fp8_quant_rot_mul_mat, always G=0 (per-row — the only activation
// scale layout GGML_OP_ML8_MUL_MAT's pre-quantized contract accepts, see
// ggml-ml8.h), wrapped in ggml_ml8_mul_mat instead of ggml_fp8_mul_mat since
// ML8_4's GEMM dequantizes the WEIGHT side via the centroid LUT rather than
// a per-block fp8 scale pair. See build_ml8_or_mul_mat's doc comment in
// llama-ml8-registry.h for the full contract.
static struct ggml_tensor * build_ml8_quant_rot_mul_mat(
        struct ggml_context  * ctx,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x,
        const ml8_sidecars   * sc,
        fp8_qrot_memo        * qrot_memo) {
    struct ggml_tensor * qrot = get_or_build_fp8_quant_rot(ctx, weight, x, sc, qrot_memo, /*G=*/0);
    return ggml_ml8_mul_mat(ctx, weight, sc->centroids, qrot);
}

struct ggml_tensor * build_ml8_or_mul_mat(
        struct ggml_context  * ctx,
        const ml8_registry   & reg,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x,
        fp8_qrot_memo        * qrot_memo) {

    const ml8_sidecars * sc = reg.find(weight);

    if (weight->type == GGML_TYPE_FP8_B128) {
        return build_fp8_quant_rot_mul_mat(ctx, weight, x, sc, qrot_memo, fp8_b128_layout_G());
    }

    if (weight->type == GGML_TYPE_ML8_4) {
        // ML8_4 weights MUST have a centroids sidecar — plain mul_mat cannot
        // handle them. Assert loudly rather than falling through to a cryptic
        // backend abort. This mirrors the assertion style in qwen35.cpp:540-544.
        GGML_ASSERT(sc           && "ML8_4 weight has no registry entry — missing centroids sidecar");
        GGML_ASSERT(sc->centroids && "ML8_4 weight registry entry has null centroids");

        if (ml8_4_act_legacy_enabled()) {
            // Pre-existing behavior, kept for A/B comparison (MT_ML8_4_ACT=legacy).
            struct ggml_tensor * x_xf = apply_ml8_input_xform(ctx, x, *sc);
            return ggml_ml8_mul_mat(ctx, weight, sc->centroids, x_xf);
        }
        // Default: fused GGML_OP_FP8_QUANT_ROT(G=0) + GGML_OP_ML8_MUL_MAT
        // (prequantized) path — one launch for the activation pipeline,
        // shared with every other weight of this input group.
        return build_ml8_quant_rot_mul_mat(ctx, weight, x, sc, qrot_memo);
    }

    if (weight->type == GGML_TYPE_ML8_FP8) {
        if (ml8_fp8_legacy_enabled()) {
            // Pre-existing behavior, kept for A/B comparison (WP_ML8_FP8_LEGACY=1).
            // The CUDA backend inspects src0->type in ggml_cuda_mul_mat and
            // auto-dispatches to the no-LUT FP8 path; the CPU backend has
            // ML8_FP8 vec_dot traits. No centroids sidecar exists for FP8 (the
            // GEMM has no LUT) — only the optional AWQ+rotation input transform
            // applies. A registry miss (sc == nullptr) is the pre-existing
            // behavior: plain ggml_mul_mat on the untransformed x.
            struct ggml_tensor * x_xf = sc ? apply_ml8_input_xform(ctx, x, *sc) : x;
            return ggml_mul_mat(ctx, weight, x_xf);
        }
        // Default: same memoized quant_rot + fp8_mul_mat path as FP8_B128,
        // with G=32 (ML8_FP8's 34-byte, 32-wide-K-group blocks).
        return build_fp8_quant_rot_mul_mat(ctx, weight, x, sc, qrot_memo, /*G=*/32);
    }

    // All other weight types (F32, BF16, Q4_0, …) — plain mul_mat.
    return ggml_mul_mat(ctx, weight, x);
}
