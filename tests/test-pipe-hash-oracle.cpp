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

int main() {
    test_empty();
    test_registration();
    test_lookup_single_token();
    test_lookup_union_is_deduped_and_sorted();
    test_unknown_layer_is_false_not_throw();
    test_out_of_range_tokens_are_skipped();
    test_saturation_early_exit();
    test_bad_registration_throws();

    if (g_failed == 0) {
        std::printf("test-pipe-hash-oracle: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-pipe-hash-oracle: %d check(s) failed\n", g_failed);
    return 1;
}
