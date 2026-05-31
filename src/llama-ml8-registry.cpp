// llama-ml8-registry.cpp
//
// MAD-223 Phase G.12: ml8 sidecar registry + build_ml8_or_mul_mat helper.
//
// See llama-ml8-registry.h for the public contract.

#include "llama-ml8-registry.h"

#include "ggml.h"
#include "ggml-ml8.h"

struct ggml_tensor * build_ml8_or_mul_mat(
        struct ggml_context  * ctx,
        const ml8_registry   & reg,
        struct ggml_tensor   * weight,
        struct ggml_tensor   * x) {

    const ml8_sidecars * sc = reg.find(weight);

    if (weight->type == GGML_TYPE_ML8_4) {
        // ML8_4 weights MUST have a centroids sidecar — plain mul_mat cannot
        // handle them. Assert loudly rather than falling through to a cryptic
        // backend abort. This mirrors the assertion style in qwen35.cpp:540-544.
        GGML_ASSERT(sc           && "ML8_4 weight has no registry entry — missing centroids sidecar");
        GGML_ASSERT(sc->centroids && "ML8_4 weight registry entry has null centroids");

        // Apply optional AWQ scale (elementwise) then optional Kronecker rotation.
        // This is the exact transform from qwen35.cpp:546-558.
        struct ggml_tensor * x_xf = x;

        if (sc->awq_scale) {
            x_xf = ggml_mul(ctx, x_xf, sc->awq_scale);
        }

        if (sc->rotation_h_a) {
            const int64_t a = sc->rotation_h_a->ne[0];
            const int64_t b = x_xf->ne[0] / a;
            x_xf = ggml_ml8_apply_rotation(ctx, x_xf, sc->rotation_h_a, a, b);
        }

        return ggml_ml8_mul_mat(ctx, weight, sc->centroids, x_xf);
    }

    if (weight->type == GGML_TYPE_ML8_FP8) {
        // The CUDA backend inspects src0->type in ggml_cuda_mul_mat and
        // auto-dispatches to the no-LUT FP8 path; the CPU backend has
        // ML8_FP8 vec_dot traits. Plain ggml_mul_mat is correct here.
        return ggml_mul_mat(ctx, weight, x);
    }

    // All other weight types (F32, BF16, Q4_0, …) — plain mul_mat.
    return ggml_mul_mat(ctx, weight, x);
}
