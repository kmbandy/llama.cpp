// MAD-137: unit tests for mt::SemanticIndex — the chunk-level
// fingerprint store owned by mt::llama_memory_tiered.
//
// Notes on scope:
//
//   The MAD-137 ticket asks us to "verify the wrapper is genuinely thin
//   (no paged scaffolding)" and to round-trip embed_text +
//   record_chunk_fingerprint + recurrent backup/restore. Those last two
//   require a real model + a real inner cache and so live at the
//   integration tier (test-paged-lifecycle / stress-paged-multi-seq).
//
//   What's testable in isolation here is the chunk-level SemanticIndex
//   that the wrapper composes — it's the slice that survived the MAD-127
//   thinning, so exercising it end-to-end is the right unit-level proof
//   that the wrapper retained its public semantic surface after the
//   paged scaffolding was carved out.
//
//   The absence of paged-block surface is enforced by code review (no
//   BlockPool / BlockTable / llama_kv_cache_paged references in
//   mt-tiered.{h,cpp}); a unit test asserting on the absence of named
//   members would just be brittle without catching anything semantic.

#include "../src/memory-tier/mt-semantic.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

using mt::SemanticIndex;
using Tier = SemanticIndex::Tier;

namespace {

std::vector<float> normed(std::vector<float> v) {
    double sq = 0.0;
    for (float x : v) sq += (double) x * x;
    const double n = std::sqrt(sq);
    if (n > 0.0) {
        for (float & x : v) x = (float) ((double) x / n);
    }
    return v;
}

std::string tmp_path(const char * tag) {
    const char * dir = std::getenv("TMPDIR");
    if (dir == nullptr || dir[0] == '\0') dir = "/tmp";
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s/test-mt-tiered-thin-%s-%d.bin", dir, tag, (int) ::getpid());
    return std::string(buf);
}

}  // namespace

int main() {
    // ─── Empty state ───
    {
        SemanticIndex idx;
        assert(idx.size() == 0);
        assert(idx.score(normed({1, 0, 0}), 5, 0.0f).empty());
        printf("test-mt-tiered-thin: empty state ok\n");
    }

    // ─── add + size — every add increments size until the FIFO cap ───
    {
        SemanticIndex idx;
        idx.add_fingerprint({0, 1, 2}, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint({3, 4, 5}, normed({0, 1, 0}), Tier::Warm);
        assert(idx.size() == 2);
        printf("test-mt-tiered-thin: add + size ok\n");
    }

    // ─── score — descending order, returns positions + tier ───
    {
        SemanticIndex idx;
        // Three chunks at known angles to query (1,0,0):
        //   chunk A: (1,0,0)        cos = 1.0
        //   chunk B: (1,1,0) / √2   cos ≈ 0.707
        //   chunk C: (0,1,0)        cos = 0.0
        idx.add_fingerprint({100}, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint({200}, normed({1, 1, 0}), Tier::Warm);
        idx.add_fingerprint({300}, normed({0, 1, 0}), Tier::Cold);

        const auto hits = idx.score(normed({1, 0, 0}), /* top_k */ 10, /* threshold */ -1.0f);
        assert(hits.size() == 3);
        assert(hits[0].positions.size() == 1 && hits[0].positions[0] == 100);
        assert(hits[0].tier == Tier::Hot);
        assert(hits[1].positions.size() == 1 && hits[1].positions[0] == 200);
        assert(hits[1].tier == Tier::Warm);
        assert(hits[2].positions.size() == 1 && hits[2].positions[0] == 300);
        assert(hits[2].tier == Tier::Cold);
        // Strictly descending.
        assert(hits[0].score > hits[1].score);
        assert(hits[1].score > hits[2].score);
        printf("test-mt-tiered-thin: score returns positions + tier in descending order ok\n");
    }

    // ─── threshold filter ───
    {
        SemanticIndex idx;
        idx.add_fingerprint({1}, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint({2}, normed({1, 1, 0}), Tier::Warm);   // ~0.707
        idx.add_fingerprint({3}, normed({0, 1, 0}), Tier::Cold);   // ~0.0

        const auto hits = idx.score(normed({1, 0, 0}), 10, /* threshold */ 0.5f);
        assert(hits.size() == 2);
        assert(hits[0].positions[0] == 1);
        assert(hits[1].positions[0] == 2);
        printf("test-mt-tiered-thin: threshold filter ok\n");
    }

    // ─── top_k cap + edge cases ───
    {
        SemanticIndex idx;
        for (int i = 0; i < 5; ++i) {
            idx.add_fingerprint({(llama_pos) i}, normed({1, 0, 0}), Tier::Hot);
        }
        assert(idx.score(normed({1, 0, 0}), /* top_k */ 3, -1.0f).size() == 3);
        assert(idx.score(normed({1, 0, 0}), /* top_k */ 0, -1.0f).empty());
        assert(idx.score(normed({1, 0, 0}), -1, -1.0f).empty());
        assert(idx.score(/* empty query */ {}, 5, -1.0f).empty());
        printf("test-mt-tiered-thin: top_k cap + edge cases ok\n");
    }

    // ─── FIFO eviction at kMaxFingerprints ───
    // Fill past the cap; verify size plateaus and the first-inserted
    // entry is the one that gets evicted (lowest turn).
    {
        SemanticIndex idx;
        const size_t cap = SemanticIndex::kMaxFingerprints;

        // Insert exactly the cap. The first one carries position 0 and
        // a unique direction we can probe for later.
        idx.add_fingerprint({/* pos */ 0}, normed({1, 0, 0, 0}), Tier::Hot);
        for (size_t i = 1; i < cap; ++i) {
            idx.add_fingerprint({(llama_pos) i}, normed({0, 1, 0, 0}), Tier::Warm);
        }
        assert(idx.size() == cap);

        // The oldest entry is still findable by its unique direction.
        {
            const auto hits = idx.score(normed({1, 0, 0, 0}), 1, /* threshold */ 0.99f);
            assert(hits.size() == 1);
            assert(hits[0].positions[0] == 0);
        }

        // Insert one more — total should stay at cap, oldest must be gone.
        idx.add_fingerprint({(llama_pos) cap}, normed({0, 0, 1, 0}), Tier::Cold);
        assert(idx.size() == cap);
        {
            const auto hits = idx.score(normed({1, 0, 0, 0}), 1, 0.99f);
            assert(hits.empty());  // the (1,0,0,0) entry was evicted
        }
        // The newest is findable.
        {
            const auto hits = idx.score(normed({0, 0, 1, 0}), 1, 0.99f);
            assert(hits.size() == 1);
            assert(hits[0].positions[0] == (llama_pos) cap);
        }
        printf("test-mt-tiered-thin: FIFO eviction at kMaxFingerprints ok\n");
    }

    // ─── clear + reuse — clear() resets size and the turn counter ───
    {
        SemanticIndex idx;
        idx.add_fingerprint({0}, normed({1, 0, 0}), Tier::Hot);
        idx.add_fingerprint({1}, normed({0, 1, 0}), Tier::Warm);
        assert(idx.size() == 2);

        idx.clear();
        assert(idx.size() == 0);
        assert(idx.score(normed({1, 0, 0}), 5, -1.0f).empty());

        // Index is reusable.
        idx.add_fingerprint({0}, normed({1, 0, 0}), Tier::Hot);
        const auto hits = idx.score(normed({1, 0, 0}), 1, 0.99f);
        assert(hits.size() == 1);
        printf("test-mt-tiered-thin: clear + reuse ok\n");
    }

    // ─── save_to_disk + load_from_disk round-trip (MTFI v1) ───
    {
        const std::string path = tmp_path("rt");

        SemanticIndex original;
        original.add_fingerprint({10, 11, 12}, normed({1, 0, 0, 0}), Tier::Hot);
        original.add_fingerprint({20, 21},     normed({0, 1, 0, 0}), Tier::Warm);
        original.add_fingerprint({30},         normed({0, 0, 1, 0}), Tier::Cold);

        const bool save_ok = original.save_to_disk(path);
        assert(save_ok);

        SemanticIndex restored;
        // Pre-load: stray entry to verify load REPLACES (not merges).
        restored.add_fingerprint({999}, normed({1, 0, 0, 0}), Tier::Hot);
        assert(restored.size() == 1);

        const bool load_ok = restored.load_from_disk(path);
        assert(load_ok);
        assert(restored.size() == 3);  // not 4 — replaced

        // Probe each direction; verify positions + tier survived.
        {
            const auto hits = restored.score(normed({1, 0, 0, 0}), 1, 0.99f);
            assert(hits.size() == 1);
            assert(hits[0].positions == std::vector<llama_pos>({10, 11, 12}));
            assert(hits[0].tier == Tier::Hot);
        }
        {
            const auto hits = restored.score(normed({0, 1, 0, 0}), 1, 0.99f);
            assert(hits.size() == 1);
            assert(hits[0].positions == std::vector<llama_pos>({20, 21}));
            assert(hits[0].tier == Tier::Warm);
        }
        {
            const auto hits = restored.score(normed({0, 0, 1, 0}), 1, 0.99f);
            assert(hits.size() == 1);
            assert(hits[0].positions == std::vector<llama_pos>({30}));
            assert(hits[0].tier == Tier::Cold);
        }

        ::unlink(path.c_str());
        printf("test-mt-tiered-thin: save/load round-trip ok\n");
    }

    // ─── load_from_disk on missing file returns false ───
    {
        SemanticIndex idx;
        idx.add_fingerprint({1}, normed({1, 0, 0}), Tier::Hot);
        const bool load_ok = idx.load_from_disk("/tmp/this-file-does-not-exist-mad137-tiered");
        assert(!load_ok);
        // In-memory state untouched.
        assert(idx.size() == 1);
        printf("test-mt-tiered-thin: load missing file safe ok\n");
    }

    // ─── load_from_disk on wrong magic returns false ───
    {
        const std::string path = tmp_path("badmagic");
        FILE * f = std::fopen(path.c_str(), "wb");
        assert(f);
        const uint32_t wrong_magic = 0xCAFEBABE;
        const uint32_t version     = 1;
        std::fwrite(&wrong_magic, sizeof(wrong_magic), 1, f);
        std::fwrite(&version,     sizeof(version),     1, f);
        std::fclose(f);

        SemanticIndex idx;
        idx.add_fingerprint({1}, normed({1, 0, 0}), Tier::Hot);
        const bool load_ok = idx.load_from_disk(path);
        assert(!load_ok);
        assert(idx.size() == 1);

        ::unlink(path.c_str());
        printf("test-mt-tiered-thin: load bad magic safe ok\n");
    }

    printf("test-mt-tiered-thin: ALL PASS\n");
    return 0;
}
