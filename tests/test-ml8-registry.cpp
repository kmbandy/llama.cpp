// test-ml8-registry — MAD-223 Phase G.12: ml8 sidecar registry + path-selection contract.
//
// Proves the zero-impact fallback: the registry dispatches to the correct
// graph op depending on weight type and whether sidecars are present.
//
//   Case A: plain F32 weight + empty registry → GGML_OP_MUL_MAT
//   Case B: ML8_4 weight + registry with centroids → GGML_OP_ML8_MUL_MAT
//   Case C: ML8_FP8 weight + empty registry → GGML_OP_MUL_MAT
//
// This test does NOT compute values — it inspects the graph node op field,
// which is sufficient to verify the dispatch contract without needing a
// full backend compute pass.

#include "ggml.h"
#include "ggml-ml8.h"
#include "llama-ml8-registry.h"

#include <cassert>
#include <cstdio>

// ─── helpers ────────────────────────────────────────────────────────────────

static struct ggml_context * make_ctx() {
    struct ggml_init_params p {};
    p.mem_size   = 4 * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = true;   // graph-wiring only — no data needed
    struct ggml_context * ctx = ggml_init(p);
    assert(ctx && "ggml_init failed");
    return ctx;
}

// ─── main ───────────────────────────────────────────────────────────────────

int main(void) {
    std::printf("# test-ml8-registry (MAD-223 Phase G.12)\n");

    // K must be a multiple of 64 (QK_ML8). Use 128 for two K-groups.
    constexpr int64_t K = 128;
    constexpr int64_t N = 16;
    constexpr int64_t M = 4;
    constexpr int64_t n_groups_k = K / 64;  // = 2

    // ─── Case A: F32 weight, empty registry → plain mul_mat ─────────────
    {
        struct ggml_context * ctx = make_ctx();

        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

        ml8_registry reg;   // empty

        struct ggml_tensor * result = build_ml8_or_mul_mat(ctx, reg, w, x);
        assert(result != nullptr && "Case A: result must not be null");
        assert(result->op == GGML_OP_MUL_MAT &&
               "Case A: F32 weight, empty registry must produce GGML_OP_MUL_MAT");
        (void)result;

        ggml_free(ctx);
        std::printf("  [PASS] Case A: F32 weight + empty registry → GGML_OP_MUL_MAT\n");
    }

    // ─── Case B: ML8_4 weight + registry with centroids → ml8 op ────────
    {
        struct ggml_context * ctx = make_ctx();

        struct ggml_tensor * w          = ggml_new_tensor_2d(ctx, GGML_TYPE_ML8_4,   K, N);
        struct ggml_tensor * centroids  = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 16, n_groups_k);
        struct ggml_tensor * x          = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,     K, M);

        ml8_registry reg;
        ml8_sidecars sc {};
        sc.centroids    = centroids;
        sc.rotation_h_a = nullptr;
        sc.awq_scale    = nullptr;
        reg.register_weight(w, sc);

        struct ggml_tensor * result = build_ml8_or_mul_mat(ctx, reg, w, x);
        assert(result != nullptr && "Case B: result must not be null");
        // ggml_ml8_mul_mat emits GGML_OP_ML8_MUL_MAT (see ggml/src/ggml-ml8.c:279)
        assert(result->op == GGML_OP_ML8_MUL_MAT &&
               "Case B: ML8_4 weight + centroids must produce GGML_OP_ML8_MUL_MAT");
        // Also confirm it is NOT the plain path
        assert(result->op != GGML_OP_MUL_MAT &&
               "Case B: ML8_4 with sidecars must NOT produce GGML_OP_MUL_MAT");
        (void)result;

        ggml_free(ctx);
        std::printf("  [PASS] Case B: ML8_4 weight + centroids → GGML_OP_ML8_MUL_MAT\n");
    }

    // ─── Case C: ML8_FP8 weight, empty registry → plain mul_mat ─────────
    {
        struct ggml_context * ctx = make_ctx();

        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_ML8_FP8, K, N);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,     K, M);

        ml8_registry reg;   // empty

        struct ggml_tensor * result = build_ml8_or_mul_mat(ctx, reg, w, x);
        assert(result != nullptr && "Case C: result must not be null");
        assert(result->op == GGML_OP_MUL_MAT &&
               "Case C: ML8_FP8 weight must produce GGML_OP_MUL_MAT");
        (void)result;

        ggml_free(ctx);
        std::printf("  [PASS] Case C: ML8_FP8 weight + empty registry → GGML_OP_MUL_MAT\n");
    }

    std::printf("\n=== PASS: ml8 registry path-selection contract verified ===\n");
    return 0;
}
