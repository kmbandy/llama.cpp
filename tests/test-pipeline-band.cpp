// Unit tests for the cross-machine pipeline band helpers (src/llama-pipeline.*).
//
// These predicates decide which tensors a stage loads (create_tensor) and
// which tensors wp-stage-split writes. A disagreement -- or a band that is
// empty, discontinuous, or role-inconsistent -- produces a model that runs
// and emits garbage, so the validation paths are tested hard.
//
// Standalone build:
//   g++ -std=c++17 -I include -I ggml/include -I src -I . \
//       tests/test-pipeline-band.cpp src/llama-pipeline.cpp -o /tmp/t && /tmp/t

#include "llama-pipeline.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

#define CHECK_THROWS(expr)                                                      \
    do {                                                                        \
        bool threw = false;                                                     \
        try { expr; } catch (const std::runtime_error &) { threw = true; }      \
        if (!threw) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: expected throw: %s\n",            \
                         __FILE__, __LINE__, #expr);                            \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

static void test_band_enabled() {
    CHECK(!llama_pipeline_band_enabled(-1, -1));
    CHECK( llama_pipeline_band_enabled( 0, 56));
    CHECK( llama_pipeline_band_enabled(57, 77));
    // one-sided is "enabled" so that resolve_band gets to reject it loudly
    CHECK( llama_pipeline_band_enabled( 0, -1));
    CHECK( llama_pipeline_band_enabled(-1, 77));
}

static void test_resolve_band() {
    // unset resolves to the full range -- the legacy behaviour
    {
        const llama_pipeline_stage b = llama_pipeline_resolve_band(-1, -1, 78);
        CHECK(b.first == 0 && b.last == 77);
    }
    // valid bands pass through unchanged
    {
        const llama_pipeline_stage b = llama_pipeline_resolve_band(0, 56, 78);
        CHECK(b.first == 0 && b.last == 56);
    }
    {
        const llama_pipeline_stage b = llama_pipeline_resolve_band(57, 77, 78);
        CHECK(b.first == 57 && b.last == 77);
    }
    // single-layer band is legal (a middle stage can be one layer)
    {
        const llama_pipeline_stage b = llama_pipeline_resolve_band(40, 40, 78);
        CHECK(b.first == 40 && b.last == 40);
    }

    // one-sided: refuse loudly
    CHECK_THROWS(llama_pipeline_resolve_band( 0, -1, 78));
    CHECK_THROWS(llama_pipeline_resolve_band(-1, 77, 78));

    // empty band: refuse
    CHECK_THROWS(llama_pipeline_resolve_band(57, 56, 78));

    // out of range: refuse
    CHECK_THROWS(llama_pipeline_resolve_band(-2, 56, 78));
    CHECK_THROWS(llama_pipeline_resolve_band( 0, 78, 78));
    CHECK_THROWS(llama_pipeline_resolve_band(77, 78, 78));

    // degenerate model
    CHECK_THROWS(llama_pipeline_resolve_band(0, 0, 0));
}

static void test_block_index() {
    CHECK(llama_pipeline_tensor_block_index("blk.0.attn_q.weight") == 0);
    CHECK(llama_pipeline_tensor_block_index("blk.57.ffn_up_exps.weight") == 57);
    CHECK(llama_pipeline_tensor_block_index("blk.77.output.weight") == 77);
    CHECK(llama_pipeline_tensor_block_index("enc.blk.3.attn_q.weight") == 3);
    CHECK(llama_pipeline_tensor_block_index("dec.blk.12.ffn_norm.weight") == 12);

    CHECK(llama_pipeline_tensor_block_index("token_embd.weight") == -1);
    CHECK(llama_pipeline_tensor_block_index("output_norm.weight") == -1);
    CHECK(llama_pipeline_tensor_block_index("output.weight") == -1);
    CHECK(llama_pipeline_tensor_block_index("blkx.3.attn_q.weight") == -1);
    CHECK(llama_pipeline_tensor_block_index("blk.attn_q.weight") == -1);
    CHECK(llama_pipeline_tensor_block_index("blk.3x.attn_q.weight") == -1);
    CHECK(llama_pipeline_tensor_block_index(nullptr) == -1);
}

static void test_owns_tensor() {
    const int32_t n_layer = 78;
    auto owns = [n_layer](
            int32_t first, int32_t last, int32_t n_layer_nextn,
            const char * name, bool duplicated_embd) {
        return llama_pipeline_owns_tensor(
            first, last, n_layer, n_layer_nextn, name, duplicated_embd);
    };

    // full range owns everything
    for (const char * n : {"blk.0.attn_q.weight", "blk.77.ffn_down_exps.weight",
                           "token_embd.weight", "output_norm.weight", "output.weight"}) {
        CHECK(owns(0, 77, 0, n, false));
    }

    // no-MTP head stage [0, 56]: owns its layers + token_embd, not output tensors
    CHECK( owns(0, 56, 0, "blk.0.attn_norm.weight",    false));
    CHECK( owns(0, 56, 0, "blk.56.ffn_up_exps.weight", false));
    CHECK(!owns(0, 56, 0, "blk.57.ffn_up_exps.weight", false));
    CHECK(!owns(0, 56, 0, "blk.77.attn_out.weight",    false));
    CHECK( owns(0, 56, 0, "token_embd.weight",         false));
    CHECK(!owns(0, 56, 0, "output_norm.weight",        false));
    CHECK(!owns(0, 56, 0, "output.weight",             false));

    // tail stage [57, 77]: owns its layers + output tensors, not token_embd
    CHECK(!owns(57, 77, 0, "blk.56.ffn_up_exps.weight", false));
    CHECK( owns(57, 77, 0, "blk.57.ffn_up_exps.weight", false));
    CHECK( owns(57, 77, 0, "blk.77.ffn_norm.weight",    false));
    CHECK(!owns(57, 77, 0, "token_embd.weight",         false));
    CHECK( owns(57, 77, 0, "output_norm.weight",        false));
    CHECK( owns(57, 77, 0, "output.weight",             false));

    // ... unless the tail has tied embeddings (duplicated_embd): then it does
    // load token_embd as its lm_head
    CHECK(owns(57, 77, 0, "token_embd.weight", true));

    // ... but a middle or head-adjacent stage must NOT claim token_embd
    // through the duplicated-output fallback
    CHECK(!owns(20, 40, 0, "token_embd.weight", true));
    CHECK( owns( 0, 40, 0, "token_embd.weight", true)); // head owns it anyway

    // middle stage [20, 40]: owns only its layers
    CHECK( owns(20, 40, 0, "blk.20.attn_q.weight", false));
    CHECK( owns(20, 40, 0, "blk.40.attn_q.weight", false));
    CHECK(!owns(20, 40, 0, "blk.19.attn_q.weight", false));
    CHECK(!owns(20, 40, 0, "blk.41.attn_q.weight", false));
    CHECK(!owns(20, 40, 0, "token_embd.weight",    false));
    CHECK(!owns(20, 40, 0, "output_norm.weight",   false));
    CHECK(!owns(20, 40, 0, "output.weight",        false));

    // small global tensors are owned by every stage
    CHECK(owns(57, 77, 0, "some_global_bias.weight", false));
    CHECK(owns( 0, 56, 0, "some_global_bias.weight", false));

    // Without MTP metadata, preserve the old rule for blocks past n_layer.
    CHECK( owns(57, 77, 0, "blk.78.nextn_eh_proj.weight", false));
    CHECK(!owns( 0, 56, 0, "blk.78.nextn_eh_proj.weight", false));

    // With MTP metadata, NextN and output tensors also belong to the head.
    CHECK(!owns(57, 77, 1, "blk.78.nextn_eh_proj.weight", false));
    CHECK( owns( 0, 56, 1, "blk.78.nextn_eh_proj.weight", false));
    CHECK(!owns(20, 40, 1, "blk.78.nextn_eh_proj.weight", false));
    CHECK( owns( 0, 56, 1, "output_norm.weight", false));
    CHECK( owns( 0, 56, 1, "output.weight",      false));
    CHECK( owns(57, 77, 1, "output_norm.weight", false));
    CHECK( owns(57, 77, 1, "output.weight",      false));

    // "output" prefix matching must not confuse output_norm with output
    CHECK(!owns(0, 56, 0, "output_norm.weight", false));
    CHECK(!owns(0, 56, 0, "output.weight",      false));

    CHECK(!owns(0, 77, 0, nullptr, false));
}

static void test_validate_stages() {
    const int32_t n_layer = 78;

    // the GLM-5.2 split: head [0,56] + tail [57,77]
    llama_pipeline_validate_stages({{0, 56}, {57, 77}}, n_layer);
    // single stage owning everything
    llama_pipeline_validate_stages({{0, 77}}, n_layer);
    // three stages
    llama_pipeline_validate_stages({{0, 20}, {21, 56}, {57, 77}}, n_layer);
    // order in the vector does not have to be sorted? -- it does: stages are
    // validated in pipeline order, so unsorted input is a gap/overlap error
    CHECK_THROWS(llama_pipeline_validate_stages({{57, 77}, {0, 56}}, n_layer));

    // gap: nobody owns layers 57..60 -> refuse
    CHECK_THROWS(llama_pipeline_validate_stages({{0, 56}, {61, 77}}, n_layer));

    // overlap: two stages own layer 40 -> refuse
    CHECK_THROWS(llama_pipeline_validate_stages({{0, 40}, {40, 77}}, n_layer));

    // missing head: token_embd unowned -> refuse
    CHECK_THROWS(llama_pipeline_validate_stages({{10, 77}}, n_layer));

    // missing tail: output_norm/output unowned -> refuse
    CHECK_THROWS(llama_pipeline_validate_stages({{0, 60}}, n_layer));

    // no stages at all -> refuse
    CHECK_THROWS(llama_pipeline_validate_stages({}, n_layer));

    // a stage with an invalid band is refused even in an otherwise fine set
    CHECK_THROWS(llama_pipeline_validate_stages({{0, 56}, {57, 99}}, n_layer));
    CHECK_THROWS(llama_pipeline_validate_stages({{0, 56}, {-1, 77}}, n_layer));
}

int main() {
    test_band_enabled();
    test_resolve_band();
    test_block_index();
    test_owns_tensor();
    test_validate_stages();

    if (g_failed == 0) {
        std::printf("OK: all pipeline band tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d check(s) failed\n", g_failed);
    return 1;
}
