// Unit tests for the hash-layer expert oracle (src/pipeline/pipe-hash-oracle.*).
//
// No model, no ggml: the tid2eid table is handed in as plain host memory, which
// is the whole point of the class -- the lookup is separable from the pager and
// from the model.
//
// Standalone build:
//   g++ -std=c++17 -I src -I src/pipeline
//       tests/test-pipe-hash-oracle.cpp src/pipeline/pipe-hash-oracle.cpp -o /tmp/t

#include "pipe-hash-oracle.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

using pipe_expert_dispatcher::hash_oracle;

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
        try { expr; } catch (const std::invalid_argument &) { threw = true; }   \
        if (!threw) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s did not throw\n",              \
                         __FILE__, __LINE__, #expr);                            \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

// A 4-token vocabulary selecting 2 of 8 experts. Row-major by token id, matching
// the {n_expert_used, n_vocab} tensor shape where ne[0] is the row stride.
//   token 0 -> {3, 1}
//   token 1 -> {1, 5}
//   token 2 -> {7, 0}
//   token 3 -> {3, -1}   (-1 = unused slot padding)
static const int32_t kTable[] = {
    3, 1,
    1, 5,
    7, 0,
    3, -1,
};
static constexpr int32_t kUsed  = 2;
static constexpr int32_t kVocab = 4;
static constexpr int32_t kExp   = 8;

static hash_oracle make_oracle() {
    hash_oracle oracle;
    oracle.register_layer(0, kUsed, kVocab, kExp, kTable);
    oracle.register_layer(2, kUsed, kVocab, kExp, kTable);
    return oracle;
}

static void test_empty() {
    hash_oracle oracle;
    CHECK(oracle.empty());
    CHECK(oracle.layers().empty());

    std::vector<int32_t> out;
    const int32_t token = 0;
    CHECK(!oracle.experts_for(0, &token, 1, out));
    CHECK(out.empty());
}

static void test_registration() {
    const hash_oracle oracle = make_oracle();
    CHECK(!oracle.empty());
    CHECK(oracle.layers() == std::vector<int32_t>({ 0, 2 }));
}

static void test_lookup_single_token() {
    const hash_oracle oracle = make_oracle();
    std::vector<int32_t> out;

    const int32_t t0 = 0;
    CHECK(oracle.experts_for(0, &t0, 1, out));
    CHECK(out == std::vector<int32_t>({ 1, 3 }));   // ASCENDING, not table order

    const int32_t t2 = 2;
    CHECK(oracle.experts_for(0, &t2, 1, out));
    CHECK(out == std::vector<int32_t>({ 0, 7 }));
}

static void test_lookup_union_is_deduped_and_sorted() {
    const hash_oracle oracle = make_oracle();
    std::vector<int32_t> out;

    // tokens 0,1 select {3,1} and {1,5}: expert 1 appears twice.
    const int32_t tokens[] = { 0, 1 };
    CHECK(oracle.experts_for(0, tokens, 2, out));
    CHECK(out == std::vector<int32_t>({ 1, 3, 5 }));

    // Whole vocabulary. Token 3's -1 padding contributes nothing.
    const int32_t all[] = { 0, 1, 2, 3 };
    CHECK(oracle.experts_for(2, all, 4, out));
    CHECK(out == std::vector<int32_t>({ 0, 1, 3, 5, 7 }));
}

static void test_unknown_layer_is_false_not_throw() {
    const hash_oracle oracle = make_oracle();
    std::vector<int32_t> out{ 99 };   // must be cleared even on the false path

    const int32_t token = 0;
    CHECK(!oracle.experts_for(1, &token, 1, out));   // layer 1 has no table
    CHECK(out.empty());
    CHECK(!oracle.experts_for(40, &token, 1, out));  // past the hash block
    CHECK(out.empty());
}

static void test_out_of_range_tokens_are_skipped() {
    const hash_oracle oracle = make_oracle();
    std::vector<int32_t> out;

    // A draft model can propose an id this table does not cover. A speculative
    // hint has no business throwing over it -- it must simply contribute nothing.
    const int32_t tokens[] = { -1, 4, 9999, 2 };
    CHECK(oracle.experts_for(0, tokens, 4, out));
    CHECK(out == std::vector<int32_t>({ 0, 7 }));   // only token 2 contributed

    const int32_t bad_only[] = { -1, 4 };
    CHECK(oracle.experts_for(0, bad_only, 2, out));
    CHECK(out.empty());   // true (layer exists) with nothing to prefetch

    CHECK(oracle.experts_for(0, nullptr, 0, out));
    CHECK(out.empty());
}

static void test_saturation_early_exit() {
    // Every expert selected: the sweep must still return each exactly once and
    // in order after the early exit fires.
    static const int32_t dense[] = {
        0, 1,
        2, 3,
        4, 5,
        6, 7,
    };
    hash_oracle oracle;
    oracle.register_layer(0, 2, 4, 8, dense);

    std::vector<int32_t> out;
    const int32_t tokens[] = { 0, 1, 2, 3, 3, 2, 1, 0 };
    CHECK(oracle.experts_for(0, tokens, 8, out));
    CHECK(out == std::vector<int32_t>({ 0, 1, 2, 3, 4, 5, 6, 7 }));
}

static void test_bad_registration_throws() {
    hash_oracle oracle;
    CHECK_THROWS(oracle.register_layer(-1, kUsed, kVocab, kExp, kTable));
    CHECK_THROWS(oracle.register_layer(0, 0, kVocab, kExp, kTable));
    CHECK_THROWS(oracle.register_layer(0, kUsed, 0, kExp, kTable));
    CHECK_THROWS(oracle.register_layer(0, kUsed, kVocab, 0, kTable));
    CHECK_THROWS(oracle.register_layer(0, kUsed, kVocab, kExp, nullptr));
    // A row wider than the model's expert count means the two disagree.
    CHECK_THROWS(oracle.register_layer(0, kUsed, kVocab, /*n_expert=*/1, kTable));
    CHECK(oracle.empty());   // nothing partially registered

    // An id the model cannot have. Clamping it would send a worker after the
    // wrong expert and make "prefetch did not help" indistinguishable from a
    // real negative result, so it must be loud.
    static const int32_t over[] = { 0, 9 };
    CHECK_THROWS(oracle.register_layer(0, 2, 1, /*n_expert=*/8, over));

    oracle.register_layer(0, kUsed, kVocab, kExp, kTable);
    CHECK_THROWS(oracle.register_layer(0, kUsed, kVocab, kExp, kTable));
    CHECK(oracle.layers() == std::vector<int32_t>({ 0 }));
}

// ---- experts_ranked: the confidence behind each expert ---------------------

static float conf_of(const std::vector<hash_oracle::ranked_expert> & v, int32_t id) {
    for (const auto & e : v) {
        if (e.expert_id == id) {
            return e.conf;
        }
    }
    return -1.0f;
}

static bool near(float a, float b) { return std::fabs(a - b) < 1e-5f; }

static void test_ranked_matches_flat_union_without_weights() {
    const hash_oracle oracle = make_oracle();

    // tokens 0,1 -> {3,1} U {1,5} = {1,3,5}. No weights means every token is
    // certain, which must reproduce experts_for() exactly, ids and all.
    const int32_t tokens[] = { 0, 1 };
    std::vector<int32_t> flat;
    CHECK(oracle.experts_for(0, tokens, 2, flat));

    std::vector<hash_oracle::ranked_expert> ranked;
    CHECK(oracle.experts_ranked(0, tokens, 2, nullptr, ranked));
    CHECK(ranked.size() == flat.size());
    for (size_t i = 0; i < ranked.size() && i < flat.size(); ++i) {
        CHECK(ranked[i].expert_id == flat[i]);   // ascending by id, as the wire needs
        CHECK(near(ranked[i].conf, 1.0f));
    }
}

static void test_ranked_agreement_beats_a_single_token() {
    const hash_oracle oracle = make_oracle();

    // token 0 -> {3,1}, token 1 -> {1,5}. Expert 1 is wanted by BOTH, expert 3
    // and 5 by one each. With both tokens at 0.5, expert 1 must outrank them:
    // 1 - 0.5*0.5 = 0.75 vs 0.5. This is the property max() would lose and the
    // whole reason the frame can be ranked at all.
    const int32_t tokens[] = { 0, 1 };
    const float   w[]      = { 0.5f, 0.5f };
    std::vector<hash_oracle::ranked_expert> ranked;
    CHECK(oracle.experts_ranked(0, tokens, 2, w, ranked));
    CHECK(ranked.size() == 3);
    CHECK(near(conf_of(ranked, 1), 0.75f));
    CHECK(near(conf_of(ranked, 3), 0.5f));
    CHECK(near(conf_of(ranked, 5), 0.5f));
}

static void test_ranked_zero_weight_token_contributes_nothing() {
    const hash_oracle oracle = make_oracle();

    // A token the drafter is sure is wrong must not put its experts in the set
    // at all -- not at conf 0, which would still occupy a top-M slot.
    const int32_t tokens[] = { 0, 2 };
    const float   w[]      = { 1.0f, 0.0f };
    std::vector<hash_oracle::ranked_expert> ranked;
    CHECK(oracle.experts_ranked(0, tokens, 2, w, ranked));
    CHECK(ranked.size() == 2);              // {1,3} from token 0 only
    CHECK(conf_of(ranked, 7) < 0.0f);       // token 2's experts absent
    CHECK(conf_of(ranked, 0) < 0.0f);
    CHECK(near(conf_of(ranked, 1), 1.0f));
    CHECK(near(conf_of(ranked, 3), 1.0f));
}

static void test_ranked_skips_padding_and_bad_tokens() {
    const hash_oracle oracle = make_oracle();

    // token 3 -> {3, -1}: the padding slot must not become expert -1, and an
    // out-of-vocab id is skipped rather than throwing (a draft model may
    // propose one).
    const int32_t tokens[] = { 3, 99, -4 };
    const float   w[]      = { 0.25f, 1.0f, 1.0f };
    std::vector<hash_oracle::ranked_expert> ranked;
    CHECK(oracle.experts_ranked(0, tokens, 3, w, ranked));
    CHECK(ranked.size() == 1);
    CHECK(ranked[0].expert_id == 3);
    CHECK(near(ranked[0].conf, 0.25f));
}

static void test_ranked_unknown_layer_is_false_not_throw() {
    const hash_oracle oracle = make_oracle();
    const int32_t token = 0;
    std::vector<hash_oracle::ranked_expert> ranked;
    CHECK(!oracle.experts_ranked(1, &token, 1, nullptr, ranked));
    CHECK(ranked.empty());
}

int main() {
    test_empty();
    test_registration();
    test_lookup_single_token();
    test_lookup_union_is_deduped_and_sorted();
    test_unknown_layer_is_false_not_throw();
    test_out_of_range_tokens_are_skipped();
    test_saturation_early_exit();
    test_bad_registration_throws();
    test_ranked_matches_flat_union_without_weights();
    test_ranked_agreement_beats_a_single_token();
    test_ranked_zero_weight_token_contributes_nothing();
    test_ranked_skips_padding_and_bad_tokens();
    test_ranked_unknown_layer_is_false_not_throw();

    if (g_failed == 0) {
        std::printf("test-pipe-hash-oracle: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-pipe-hash-oracle: %d check(s) failed\n", g_failed);
    return 1;
}
