// MAD-137: unit tests for mt::BlockSemanticIndex — the per-(seq,lblock)
// fingerprint store + cosine-similarity scoring used for paged-block
// prefetch hints. Bare main()/assert style; no real GPU.

#include "../src/memory-tier/mt-semantic.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

using mt::BlockSemanticIndex;
using mt::SemanticIndex;
using Tier = SemanticIndex::Tier;

namespace {

// Build an L2-normalized embedding from a free-form direction vector.
// Cosine similarity of the result with itself is 1.0 exactly.
std::vector<float> normed(std::vector<float> v) {
    double sq = 0.0;
    for (float x : v) sq += (double) x * x;
    const double n = std::sqrt(sq);
    if (n > 0.0) {
        for (float & x : v) x = (float) ((double) x / n);
    }
    return v;
}

// Unique tmp path so parallel test runs don't collide. Honors TMPDIR
// so sandboxed environments (and BSDs) get a writable directory.
std::string tmp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    if (dir == nullptr || dir[0] == '\0') dir = "/tmp";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/test-mt-bsi-%s-%d.bin", dir, tag, (int) ::getpid());
    return std::string(buf);
}

}  // namespace

int main() {
    // ─── Empty state — every accessor returns the safe zero ───
    {
        BlockSemanticIndex idx;
        assert(idx.size() == 0);
        assert(idx.size(0) == 0);
        assert(!idx.has_fingerprint(0, 0));
        const auto hints = idx.score(0, normed({1, 0, 0}), /* top_k */ 5, /* threshold */ 0.0f);
        assert(hints.empty());
        printf("test-mt-block-semantic-index: empty state ok\n");
    }

    // ─── Basic add + has_fingerprint + size ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(/* seq */ 0, /* lblock */ 5,
                            normed({1, 0, 0}), Tier::Hot);
        assert(idx.has_fingerprint(0, 5));
        assert(!idx.has_fingerprint(0, 6));   // different lblock
        assert(!idx.has_fingerprint(1, 5));   // different seq
        assert(idx.size() == 1);
        assert(idx.size(0) == 1);
        assert(idx.size(1) == 0);
        printf("test-mt-block-semantic-index: add + has_fingerprint + size ok\n");
    }

    // ─── Overwrite at the same (seq, lblock) ───
    // Used after a partial-range edit re-fingerprints a block.
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 1, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(0, 1, normed({0, 1, 0}), Tier::Cold);
        assert(idx.size() == 1);  // not 2 — same key

        // Score should reflect the second embedding (now the "Y" direction)
        // and the second tier (Cold).
        const auto hints = idx.score(0, normed({0, 1, 0}), 5, 0.0f);
        assert(hints.size() == 1);
        assert(hints[0].lblock == 1);
        assert(hints[0].tier == Tier::Cold);
        assert(hints[0].score > 0.99f);  // ~1.0 against same direction
        printf("test-mt-block-semantic-index: overwrite at same key ok\n");
    }

    // ─── Per-seq scoping — score(seq) only sees that seq's blocks ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(1, 0, normed({1, 0, 0}), Tier::Hot);   // same direction, different seq
        idx.add_fingerprint(2, 0, normed({1, 0, 0}), Tier::Hot);

        const auto h0 = idx.score(0, normed({1, 0, 0}), 10, 0.0f);
        assert(h0.size() == 1);
        assert(h0[0].seq_id == 0);

        const auto h1 = idx.score(1, normed({1, 0, 0}), 10, 0.0f);
        assert(h1.size() == 1);
        assert(h1[0].seq_id == 1);
        printf("test-mt-block-semantic-index: per-seq scoping ok\n");
    }

    // ─── Score: deterministic descending order ───
    {
        BlockSemanticIndex idx;
        // Three blocks at known angles to the query (1,0,0):
        //   block 0: (1,0,0)        cos = 1.0
        //   block 1: (cos45, sin45) cos ≈ 0.707
        //   block 2: (0,1,0)        cos = 0.0
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(0, 1, normed({1, 1, 0}), Tier::Warm);
        idx.add_fingerprint(0, 2, normed({0, 1, 0}), Tier::Cold);

        const auto hints = idx.score(0, normed({1, 0, 0}), 10, /* threshold */ -1.0f);
        assert(hints.size() == 3);
        assert(hints[0].lblock == 0);
        assert(hints[1].lblock == 1);
        assert(hints[2].lblock == 2);
        // Scores strictly descending.
        assert(hints[0].score > hints[1].score);
        assert(hints[1].score > hints[2].score);
        printf("test-mt-block-semantic-index: descending-score ordering ok\n");
    }

    // ─── Threshold filter — drops scores strictly below ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(0, 1, normed({1, 1, 0}), Tier::Warm);  // ~0.707
        idx.add_fingerprint(0, 2, normed({0, 1, 0}), Tier::Cold);  // ~0.0

        // Threshold 0.5 keeps the first two only.
        const auto hits = idx.score(0, normed({1, 0, 0}), 10, /* threshold */ 0.5f);
        assert(hits.size() == 2);
        assert(hits[0].lblock == 0);
        assert(hits[1].lblock == 1);
        printf("test-mt-block-semantic-index: threshold filter ok\n");
    }

    // ─── top_k cap ───
    {
        BlockSemanticIndex idx;
        for (uint32_t b = 0; b < 5; ++b) {
            idx.add_fingerprint(0, b, normed({1, 0, 0}), Tier::Hot);
        }
        const auto hits = idx.score(0, normed({1, 0, 0}), /* top_k */ 3, -1.0f);
        assert(hits.size() == 3);
        printf("test-mt-block-semantic-index: top_k cap ok\n");
    }

    // ─── Edge cases for score() — empty query, top_k=0, unknown seq ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        assert(idx.score(0, /* empty query */ {}, 5, 0.0f).empty());
        assert(idx.score(0, normed({1, 0, 0}), /* top_k */ 0, 0.0f).empty());
        assert(idx.score(0, normed({1, 0, 0}), -1, 0.0f).empty());
        assert(idx.score(/* unknown seq */ 99, normed({1, 0, 0}), 5, 0.0f).empty());
        printf("test-mt-block-semantic-index: score edge cases ok\n");
    }

    // ─── update_tier — changes tier annotation; no-op on unknown key ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.update_tier(0, 0, Tier::Cold);
        const auto hits = idx.score(0, normed({1, 0, 0}), 1, -1.0f);
        assert(hits.size() == 1);
        assert(hits[0].tier == Tier::Cold);

        // Unknown (seq, lblock): no crash, no state change.
        idx.update_tier(99, 99, Tier::Hot);
        idx.update_tier(0, 99, Tier::Hot);
        assert(idx.size() == 1);
        printf("test-mt-block-semantic-index: update_tier ok\n");
    }

    // ─── remove_block — drops one entry; seq stays if other entries remain ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(0, 1, normed({0, 1, 0}), Tier::Warm);
        assert(idx.size() == 2);

        idx.remove_block(0, 0);
        assert(!idx.has_fingerprint(0, 0));
        assert(idx.has_fingerprint(0, 1));
        assert(idx.size(0) == 1);
        assert(idx.size() == 1);

        // Drop the last entry — seq map should be removed (size(seq)==0).
        idx.remove_block(0, 1);
        assert(idx.size() == 0);
        assert(idx.size(0) == 0);

        // No-op on unknown.
        idx.remove_block(99, 0);
        idx.remove_block(0, 99);
        assert(idx.size() == 0);
        printf("test-mt-block-semantic-index: remove_block ok\n");
    }

    // ─── remove_seq — drops every entry for that seq, leaves others ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(0, 1, normed({0, 1, 0}), Tier::Hot);
        idx.add_fingerprint(1, 0, normed({1, 0, 0}), Tier::Hot);
        assert(idx.size() == 3);

        idx.remove_seq(0);
        assert(idx.size() == 1);
        assert(idx.size(0) == 0);
        assert(idx.size(1) == 1);

        // No-op on unknown.
        idx.remove_seq(42);
        assert(idx.size() == 1);
        printf("test-mt-block-semantic-index: remove_seq ok\n");
    }

    // ─── clear — wipes everything ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint(1, 0, normed({0, 1, 0}), Tier::Warm);
        assert(idx.size() == 2);

        idx.clear();
        assert(idx.size() == 0);
        assert(idx.size(0) == 0);
        assert(idx.size(1) == 0);
        // Index is reusable after clear.
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        assert(idx.size() == 1);
        printf("test-mt-block-semantic-index: clear ok\n");
    }

    // ─── save_to_disk + load_from_disk round-trip (PSFI v1) ───
    {
        const std::string path = tmp_path("rt");

        BlockSemanticIndex original;
        // Three seqs × multiple blocks × different tiers.
        original.add_fingerprint(0, 0, normed({1, 0, 0, 0}), Tier::Hot);
        original.add_fingerprint(0, 1, normed({0, 1, 0, 0}), Tier::Warm);
        original.add_fingerprint(0, 2, normed({0, 0, 1, 0}), Tier::Cold);
        original.add_fingerprint(7, 0, normed({0.5f, 0.5f, 0.5f, 0.5f}), Tier::Warm);
        original.add_fingerprint(7, 4, normed({1, 1, 0, 0}), Tier::Hot);

        const bool save_ok = original.save_to_disk(path);
        assert(save_ok);

        BlockSemanticIndex restored;
        // Pre-load, restored should be empty.
        assert(restored.size() == 0);
        // Add a stray entry first to verify load REPLACES (not merges).
        restored.add_fingerprint(99, 99, normed({1, 0, 0, 0}), Tier::Hot);
        assert(restored.size() == 1);

        const bool load_ok = restored.load_from_disk(path);
        assert(load_ok);
        // The stray entry should be gone — load replaces in-memory state.
        assert(!restored.has_fingerprint(99, 99));
        assert(restored.size() == 5);
        assert(restored.size(0) == 3);
        assert(restored.size(7) == 2);

        // Verify a specific (seq, lblock) round-trips its tier and embedding.
        const auto hits = restored.score(0, normed({0, 1, 0, 0}), /* top_k */ 1, -1.0f);
        assert(hits.size() == 1);
        assert(hits[0].lblock == 1);
        assert(hits[0].tier == Tier::Warm);
        assert(hits[0].score > 0.99f);  // exact direction match

        ::unlink(path.c_str());
        printf("test-mt-block-semantic-index: save/load round-trip ok\n");
    }

    // ─── load_from_disk on missing file returns false, leaves index alone ───
    {
        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        const bool load_ok = idx.load_from_disk("/tmp/this-file-definitely-does-not-exist-mad137");
        assert(!load_ok);
        // In-memory state should be untouched.
        assert(idx.size() == 1);
        assert(idx.has_fingerprint(0, 0));
        printf("test-mt-block-semantic-index: load missing file safe ok\n");
    }

    // ─── load_from_disk on wrong magic returns false ───
    {
        const std::string path = tmp_path("badmagic");
        // Write a file that has the wrong magic.
        FILE * f = std::fopen(path.c_str(), "wb");
        assert(f);
        const uint32_t wrong_magic = 0xDEADBEEF;
        const uint32_t version     = 1;
        std::fwrite(&wrong_magic, sizeof(wrong_magic), 1, f);
        std::fwrite(&version,     sizeof(version),     1, f);
        std::fclose(f);

        BlockSemanticIndex idx;
        idx.add_fingerprint(0, 0, normed({1, 0, 0}), Tier::Hot);
        const bool load_ok = idx.load_from_disk(path);
        assert(!load_ok);
        // In-memory state untouched.
        assert(idx.size() == 1);
        assert(idx.has_fingerprint(0, 0));

        ::unlink(path.c_str());
        printf("test-mt-block-semantic-index: load bad magic safe ok\n");
    }

    printf("test-mt-block-semantic-index: ALL PASS\n");
    return 0;
}
