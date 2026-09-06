// [MAD-445] CPU-only, no-model unit test for the token-prefix matching logic that the
// cross-request prefix cache (server-context.cpp: try_adopt_live_prefix) and the existing
// --cache-ram whole-prompt cache (server_prompt_cache::load) both key their decisions on:
// server_tokens::get_common_prefix(). This does not spin up a llama_context or load a model
// -- it only exercises the pure token-list comparison, which is exactly the piece that
// decides whether a "shared prefix" is a real, token-exact match and how long it is.
//
// Run on the CPU build, no GPU / no inference:
//   ctest -R test-prefix-cache-match

#include "server-common.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int g_failures = 0;

#define CHECK_EQ(actual, expected, what)                                                     \
    do {                                                                                     \
        const auto a_ = (actual);                                                            \
        const auto e_ = (expected);                                                          \
        if (a_ != e_) {                                                                      \
            fprintf(stderr, "FAIL %s: got %zu, expected %zu (%s:%d)\n",                      \
                    what, (size_t) a_, (size_t) e_, __FILE__, __LINE__);                     \
            g_failures++;                                                                    \
        } else {                                                                             \
            fprintf(stdout, "OK   %s\n", what);                                              \
        }                                                                                    \
    } while (0)

static server_tokens make_tokens(const std::vector<llama_token> & v) {
    return server_tokens(v, /* has_mtmd */ false);
}

int main() {
    // 1) Identical prompts: the "6k shared prefix, 0 unique tail" degenerate case.
    {
        std::vector<llama_token> v(6000);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (llama_token) i;

        server_tokens a = make_tokens(v);
        server_tokens b = make_tokens(v);

        CHECK_EQ(a.get_common_prefix(b), v.size(), "identical 6000-token prompts share the full length");
    }

    // 2) 6k shared prefix + 200-token unique tail on each side (the acceptance-test shape).
    {
        std::vector<llama_token> shared(6000);
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = (llama_token) (i * 7 + 1);

        std::vector<llama_token> va = shared;
        std::vector<llama_token> vb = shared;
        for (int i = 0; i < 200; ++i) { va.push_back(100000 + i); vb.push_back(200000 + i); }

        server_tokens a = make_tokens(va);
        server_tokens b = make_tokens(vb);

        CHECK_EQ(a.get_common_prefix(b), (size_t) 6000,
                 "6000-token shared prefix with divergent 200-token tails matches exactly at the boundary");
    }

    // 3) A single differing token anywhere inside the "shared" region must cap the match --
    //    this is the property that makes prefix-cache adoption safe: it is a token-exact
    //    prefix hash/compare, not a fuzzy or length-based heuristic. A single injected-prompt
    //    or reordered-tool-schema token must not be silently treated as still-shared.
    {
        std::vector<llama_token> va(1000), vb(1000);
        for (size_t i = 0; i < va.size(); ++i) { va[i] = (llama_token) i; vb[i] = (llama_token) i; }
        vb[500] = 999999; // single divergence in the middle

        server_tokens a = make_tokens(va);
        server_tokens b = make_tokens(vb);

        CHECK_EQ(a.get_common_prefix(b), (size_t) 500, "a single mismatched token caps the match at that index");
    }

    // 4) Empty candidate vs. non-empty: this is the "fresh/idle slot" case that
    //    try_adopt_live_prefix and server_prompt_cache::load both special-case (an empty
    //    slot should be willing to adopt any sufficiently long cached/live prefix).
    {
        std::vector<llama_token> v(300);
        for (size_t i = 0; i < v.size(); ++i) v[i] = (llama_token) i;

        server_tokens empty = make_tokens({});
        server_tokens full  = make_tokens(v);

        CHECK_EQ(empty.get_common_prefix(full), (size_t) 0, "an empty token list has zero common prefix length");
        CHECK_EQ(empty.size(), (size_t) 0, "sanity: empty() token list really is empty");
    }

    // 5) Shorter-than-tail candidate: the donor's whole prompt is itself a prefix of the
    //    incoming request (e.g. donor only ever saw the shared system prompt so far).
    {
        std::vector<llama_token> shared(256);
        for (size_t i = 0; i < shared.size(); ++i) shared[i] = (llama_token) i;

        std::vector<llama_token> vb = shared;
        for (int i = 0; i < 50; ++i) vb.push_back(1000 + i);

        server_tokens donor = make_tokens(shared);
        server_tokens task  = make_tokens(vb);

        CHECK_EQ(donor.get_common_prefix(task), shared.size(),
                 "donor prompt that is itself a strict prefix of the task matches its full length");
    }

    if (g_failures > 0) {
        fprintf(stderr, "\n%d check(s) FAILED\n", g_failures);
        return 1;
    }

    fprintf(stdout, "\nall prefix-cache matching checks passed\n");
    return 0;
}
