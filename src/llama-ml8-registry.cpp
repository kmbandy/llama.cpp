// llama-ml8-registry.cpp
//
// MAD-223 Phase G.12: ml8 sidecar registry + build_ml8_or_mul_mat helper.
//
// See llama-ml8-registry.h for the public contract.

#include "llama-ml8-registry.h"

#include "ggml.h"
#include "ggml-ml8.h"

#include <string>

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

struct ggml_tensor * build_ml8_or_mul_mat(
        struct ggml_context  * ctx,
        const ml8_registry   & reg,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x,
        fp8_qrot_memo        * qrot_memo) {

    const ml8_sidecars * sc = reg.find(weight);

    if (weight->type == GGML_TYPE_FP8_B128) {
        // AWQ acts on the raw activation, same as the ML8_4/ML8_FP8 paths
        // above — apply it BEFORE building/looking up the quant_rot node.
        // Not emitted by the fp8_b128 converter today (see the design doc),
        // so this is dead code for the shipping Qwen3.8-27B recipe, but kept
        // for parity with apply_ml8_input_xform's contract.
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

        // Memo key uses the ORIGINAL `x` (pre-AWQ), matching the design:
        // group members share one activation and (per the load-time
        // assertion in qwen35.cpp) an identical rotation, so any member
        // reaching this function first builds the shared node.
        const fp8_qrot_key key{ x, h_a, a_dim, b_dim, kind };

        struct ggml_tensor * qrot = nullptr;
        if (qrot_memo) {
            auto it = qrot_memo->find(key);
            if (it != qrot_memo->end()) {
                qrot = it->second;
            }
        }
        if (qrot == nullptr) {
            qrot = ggml_fp8_quant_rot(ctx, x_xf, h_a, a_dim, b_dim, kind);
            const std::string qrot_name = std::string(weight->name) + ".qrot";
            ggml_set_name(qrot, qrot_name.c_str());
            if (qrot_memo) {
                (*qrot_memo)[key] = qrot;
            }
        }

        return ggml_fp8_mul_mat(ctx, weight, qrot);
    }

    if (weight->type == GGML_TYPE_ML8_4) {
        // ML8_4 weights MUST have a centroids sidecar — plain mul_mat cannot
        // handle them. Assert loudly rather than falling through to a cryptic
        // backend abort. This mirrors the assertion style in qwen35.cpp:540-544.
        GGML_ASSERT(sc           && "ML8_4 weight has no registry entry — missing centroids sidecar");
        GGML_ASSERT(sc->centroids && "ML8_4 weight registry entry has null centroids");

        struct ggml_tensor * x_xf = apply_ml8_input_xform(ctx, x, *sc);
        return ggml_ml8_mul_mat(ctx, weight, sc->centroids, x_xf);
    }

    if (weight->type == GGML_TYPE_ML8_FP8) {
        // The CUDA backend inspects src0->type in ggml_cuda_mul_mat and
        // auto-dispatches to the no-LUT FP8 path; the CPU backend has
        // ML8_FP8 vec_dot traits. No centroids sidecar exists for FP8 (the
        // GEMM has no LUT) — only the optional AWQ+rotation input transform
        // applies. A registry miss (sc == nullptr) is the pre-existing
        // behavior: plain ggml_mul_mat on the untransformed x.
        struct ggml_tensor * x_xf = sc ? apply_ml8_input_xform(ctx, x, *sc) : x;
        return ggml_mul_mat(ctx, weight, x_xf);
    }

    // All other weight types (F32, BF16, Q4_0, …) — plain mul_mat.
    return ggml_mul_mat(ctx, weight, x);
}
