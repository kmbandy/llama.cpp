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
#include <cstdlib>

// This test is built in Release (-DNDEBUG), where assert() is a no-op. Use an
// always-on check so the dispatch/regression contract is actually enforced.
#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL: %s\n  at %s:%d\n", (msg), __FILE__, __LINE__); \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// ─── helpers ────────────────────────────────────────────────────────────────

static struct ggml_context * make_ctx() {
    struct ggml_init_params p {};
    p.mem_size   = 4 * 1024 * 1024;
    p.mem_buffer = nullptr;
    p.no_alloc   = true;   // graph-wiring only — no data needed
    struct ggml_context * ctx = ggml_init(p);
    CHECK(ctx, "ggml_init failed");
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
        CHECK(result != nullptr, "Case A: result must not be null");
        CHECK(result->op == GGML_OP_MUL_MAT, "Case A: F32 weight, empty registry must produce GGML_OP_MUL_MAT");
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
        CHECK(result != nullptr, "Case B: result must not be null");
        // ggml_ml8_mul_mat emits GGML_OP_ML8_MUL_MAT (see ggml/src/ggml-ml8.c:279)
        CHECK(result->op == GGML_OP_ML8_MUL_MAT, "Case B: ML8_4 weight + centroids must produce GGML_OP_ML8_MUL_MAT");
        // Also confirm it is NOT the plain path
        CHECK(result->op != GGML_OP_MUL_MAT, "Case B: ML8_4 with sidecars must NOT produce GGML_OP_MUL_MAT");
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
        CHECK(result != nullptr, "Case C: result must not be null");
        CHECK(result->op == GGML_OP_MUL_MAT, "Case C: ML8_FP8 weight must produce GGML_OP_MUL_MAT");
        (void)result;

        ggml_free(ctx);
        std::printf("  [PASS] Case C: ML8_FP8 weight + empty registry → GGML_OP_MUL_MAT\n");
    }

    // ─── Case D: regression guard — fallback is byte-identical to mul_mat ─
    // T13 routes build_lora_mm's base matmul through build_ml8_or_mul_mat. For
    // every non-ml8 weight (the registry misses), the helper must be a pure
    // pass-through to ggml_mul_mat(ctx, w, x): same op, same src0 (weight),
    // same src1 (x). This proves the shared-graph change is zero-impact for
    // bf16/quantized weights and all non-ml8 models.
    {
        struct ggml_context * ctx = make_ctx();

        struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, N);
        struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, M);

        ml8_registry reg;   // empty → guaranteed miss

        struct ggml_tensor * direct = ggml_mul_mat(ctx, w, x);
        struct ggml_tensor * helper = build_ml8_or_mul_mat(ctx, reg, w, x);

        CHECK(helper != nullptr, "Case D: helper result must not be null");
        CHECK(helper->op     == direct->op, "Case D: op must match plain mul_mat");
        CHECK(helper->src[0] == direct->src[0], "Case D: src[0] (weight) must match");
        CHECK(helper->src[1] == direct->src[1], "Case D: src[1] (x) must match");
        CHECK(helper->src[0] == w, "Case D: src[0] must be the weight tensor");
        CHECK(helper->src[1] == x, "Case D: src[1] must be the input tensor");
        // Result shape must equal a plain mul_mat result.
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            CHECK(helper->ne[i] == direct->ne[i], "Case D: result shape must match");
        }

        ggml_free(ctx);
        std::printf("  [PASS] Case D: empty-registry fallback is byte-identical to ggml_mul_mat\n");
    }

    // ─── Case E: register_weight + find round-trip ──────────────────────
    // The registry must store and return the exact sidecar set for a weight,
    // and miss (nullptr) for an unregistered weight.
    {
        struct ggml_context * ctx = make_ctx();

        struct ggml_tensor * w         = ggml_new_tensor_2d(ctx, GGML_TYPE_ML8_4,   K, N);
        struct ggml_tensor * w_other   = ggml_new_tensor_2d(ctx, GGML_TYPE_ML8_4,   K, N);
        struct ggml_tensor * centroids = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 16, n_groups_k);
        struct ggml_tensor * rot_h_a   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,     8, 8);
        struct ggml_tensor * awq       = ggml_new_tensor_1d(ctx, GGML_TYPE_F32,     K);

        ml8_registry reg;
        CHECK(reg.find(w) == nullptr, "Case E: unregistered weight must miss");

        ml8_sidecars sc {};
        sc.centroids    = centroids;
        sc.rotation_h_a = rot_h_a;
        sc.awq_scale    = awq;
        reg.register_weight(w, sc);

        const ml8_sidecars * got = reg.find(w);
        CHECK(got != nullptr, "Case E: registered weight must be found");
        CHECK(got->centroids    == centroids, "Case E: centroids round-trip");
        CHECK(got->rotation_h_a == rot_h_a, "Case E: rotation_h_a round-trip");
        CHECK(got->awq_scale    == awq, "Case E: awq_scale round-trip");
        CHECK(reg.find(w_other) == nullptr, "Case E: distinct weight must still miss");

        ggml_free(ctx);
        std::printf("  [PASS] Case E: register_weight + find round-trips sidecars\n");
    }

    std::printf("\n=== PASS: ml8 registry path-selection contract verified ===\n");
    return 0;
}
