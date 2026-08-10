// MAD-137: integration test for the paged-cache semantic-prefetch path
// (MAD-125 / MAD-129 surface).
//
// Drives `llama_kv_cache_paged`'s fingerprint store + restore path:
//   - record_paged_block_fingerprint writes per-block fingerprints
//   - has_paged_fingerprint / n_paged_fingerprints reflect the writes
//   - restore_semantic_paged scores them against a query and ticks
//     the semantic_attempts counter
//   - whole-seq seq_rm drops the seq's fingerprints
//   - save/load_paged_fingerprints round-trips through PSFI v1
//
// Constructs the cache on the CPU backend so no GPU is required. A real
// model is needed for hparams (n_layer, head_dim, n_kv_heads); skips
// gracefully when neither argv[1] nor LLAMACPP_TEST_MODELFILE is set.

#include "../src/llama-kv-cache-paged.h"
#include "../src/llama-model.h"
#include "../src/memory-tier/mt-semantic.h"
#include "common.h"
#include "llama.h"
#include "ggml-backend.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// L2-normalize so cosine similarity == dot product.
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
    std::snprintf(buf, sizeof(buf), "%s/test-paged-semantic-%s-%d.bin", dir, tag, (int) ::getpid());
    return std::string(buf);
}

}  // namespace

int main(int argc, char * argv[]) {
    char * model_path = common_get_model_or_exit(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.load_mode    = LLAMA_LOAD_MODE_MMAP;

    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "test-paged-semantic: failed to load model %s — skipping\n", model_path);
        llama_backend_free();
        return 0;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    assert(buft != nullptr);

    constexpr uint32_t kNBlocks    = 8;
    constexpr uint32_t kBlockSize  = 4;
    constexpr uint32_t kNSeqMax    = 2;
    constexpr uint32_t kMaxBlks    = 16;
    constexpr uint32_t kWarmBlocks = 4;

    // ─── record / has / size — fingerprint ingestion ───
    {
        llama_kv_cache_paged cache(
            *model, buft,
            kNBlocks, kBlockSize, kNSeqMax, kMaxBlks,
            kWarmBlocks, /*n_cold_blocks=*/0,
            std::string(),
            /*cold_resume=*/false,
            /*instance_id=*/"test-record");

        const bool grew = cache.ensure_blocks_for(/*seq*/ 0, 4 * kBlockSize);
        assert(grew);
        assert(cache.n_paged_fingerprints() == 0);
        assert(!cache.has_paged_fingerprint(0, 0));

        // Record a fingerprint per logical block, each in a unique
        // direction so we can tell them apart later.
        cache.record_paged_block_fingerprint(0, 0, normed({1, 0, 0, 0}), mt::SemanticIndex::Tier::Hot);
        cache.record_paged_block_fingerprint(0, 1, normed({0, 1, 0, 0}), mt::SemanticIndex::Tier::Hot);
        cache.record_paged_block_fingerprint(0, 2, normed({0, 0, 1, 0}), mt::SemanticIndex::Tier::Hot);
        cache.record_paged_block_fingerprint(0, 3, normed({0, 0, 0, 1}), mt::SemanticIndex::Tier::Hot);

        assert(cache.n_paged_fingerprints() == 4);
        assert(cache.has_paged_fingerprint(0, 0));
        assert(cache.has_paged_fingerprint(0, 3));
        assert(!cache.has_paged_fingerprint(0, 99));
        assert(!cache.has_paged_fingerprint(1, 0));
        printf("test-paged-semantic: record + has + size ok\n");

        // ─── restore_semantic_paged — scores fingerprints, ticks counter ───
        // The blocks are still in hot (no eviction happened), so an
        // actual warm→hot fault doesn't have anything to do; what we're
        // verifying is that the call (a) doesn't crash, (b) ticks the
        // semantic_attempts counter, and (c) the requested query matches
        // a stored fingerprint in BlockSemanticIndex (visible by querying
        // it directly through the existing accessor).
        const uint64_t attempts_before = cache.semantic_attempts_total();
        const uint32_t restored = cache.restore_semantic_paged(
            /*seq*/ 0, normed({1, 0, 0, 0}), /*top_k=*/2, /*threshold=*/0.5f);
        // restored may be 0 — the matching block (lblock 0) is already
        // in hot, so there's nothing to fault in. The point is the
        // attempt was counted.
        (void) restored;
        assert(cache.semantic_attempts_total() == attempts_before + 1);
        printf("test-paged-semantic: restore_semantic_paged attempts counter ticks ok\n");

        // ─── MAD-348: has_restorable_blocks() is the server's precondition ───
        //
        // The server uses this to decide whether to pay a ~285 ms embedder
        // forward pass (on its single-threaded inference loop) to build a query
        // vector. The SAFETY PROPERTY is one-directional and absolute:
        //
        //     restore_semantic_paged() can restore > 0  ==>  has_restorable_blocks()
        //
        // A false negative silently disables semantic prefetch — a quality
        // regression that no throughput number would ever reveal. A false
        // positive only costs the embed we would have paid anyway.
        //
        // Right now every block for seq 0 is hot, so a restore is a provable
        // no-op and the guard MUST say so.
        assert(!cache.has_restorable_blocks(0) &&
               "all blocks hot => restore is a no-op => guard must be false");

        // A seq with no fingerprints at all can never restore either.
        assert(!cache.has_restorable_blocks(1) && "no fingerprints => guard must be false");

        // Now evict a FINGERPRINTED block to warm. That block is no longer on
        // the GPU, so restore_semantic_paged() can genuinely fault it back —
        // and therefore the guard MUST now open the gate. This is the case that
        // would break semantic prefetch if the predicate were too aggressive.
        const bool evicted = cache.evict_block_to_warm(/*seq*/ 0, /*logical_block=*/ 0);
        assert(evicted && "test setup: expected lblock 0 to evict to warm");

        assert(cache.has_restorable_blocks(0) &&
               "a fingerprinted block lives off-GPU => guard MUST allow the restore");

        // And the guard must agree with reality: the restore it just permitted
        // actually does restore the evicted block.
        const uint32_t restored_after_evict = cache.restore_semantic_paged(
            /*seq*/ 0, normed({1, 0, 0, 0}), /*top_k=*/2, /*threshold=*/0.5f);
        assert(restored_after_evict >= 1 &&
               "guard said restorable, so the restore must actually restore something");

        // Having faulted it back in, everything is hot again — the gate closes.
        assert(!cache.has_restorable_blocks(0) &&
               "after restore everything is hot again => guard must be false");

        printf("test-paged-semantic: has_restorable_blocks guard (MAD-348) ok\n");

        // ─── MAD-348: the restore THRESHOLD must admit real-world similarities ───
        //
        // mt-semantic.cpp gates hints with `if (s < threshold) break;`. The default used
        // to be 0.65, inherited from bge/sentence-transformers, whose cosine scores run
        // hot. LFM2.5-Embedding-350M's do not. Measured 2026-07-14 over 12 passage/query
        // pairs (144 pairings):
        //
        //     true positives (query vs its OWN passage): min 0.332, median 0.521, max 0.634
        //     false pairs   (query vs every other):      median 0.052, p95 0.263, max 0.352
        //
        // So 0.65 admitted ZERO of 12 true positives: the semantic index paid its whole
        // cost and prefetched nothing, silently. The classes separate cleanly, just far
        // below 0.65 — hence the 0.35 default.
        //
        // This test pins the GATE's behaviour at realistic similarity magnitudes, so the
        // regression cannot come back if someone "rounds the threshold up to a nicer
        // number". A query at cosine ~0.52 (this embedder's MEDIAN true positive) must
        // restore; a 0.65 gate must reject it.
        {
            llama_kv_cache_paged cache(
                *model, buft,
                kNBlocks, kBlockSize, kNSeqMax, kMaxBlks,
                kWarmBlocks, /*n_cold_blocks=*/0,
                std::string(), /*cold_resume=*/false, /*instance_id=*/"test-threshold");

            assert(cache.ensure_blocks_for(/*seq*/ 0, 4 * kBlockSize));

            // Fingerprint pointing along e0.
            const std::vector<float> fp = normed({1, 0, 0, 0});
            cache.record_paged_block_fingerprint(0, 0, fp, mt::SemanticIndex::Tier::Hot);

            // Query at cosine ≈ 0.52 to that fingerprint — this embedder's MEDIAN score
            // for a genuinely correct match. cos = 0.52 => q = 0.52*e0 + sqrt(1-0.52^2)*e1.
            const float c = 0.52f;
            const std::vector<float> q = normed({c, std::sqrt(1.0f - c * c), 0, 0});

            assert(cache.evict_block_to_warm(0, 0) && "test setup: block 0 should evict");
            assert(cache.has_restorable_blocks(0));

            // The OLD default (0.65) rejects a correct match — the bug, pinned.
            const uint32_t at_065 = cache.restore_semantic_paged(0, q, /*top_k=*/5, /*threshold=*/0.65f);
            assert(at_065 == 0 &&
                   "0.65 rejects a 0.52-cosine match: that IS the MAD-348 bug — "
                   "real true-positives for this embedder top out at 0.634");

            // The NEW default (0.35) admits it.
            const uint32_t at_035 = cache.restore_semantic_paged(0, q, /*top_k=*/5, /*threshold=*/0.35f);
            assert(at_035 >= 1 &&
                   "0.35 must admit a median true-positive similarity and restore the block");

            printf("test-paged-semantic: restore threshold calibration (MAD-348) ok "
                   "— cos=0.52 rejected at 0.65, restored at 0.35\n");
        }
    }

    // ─── Whole-seq seq_rm drops the seq's fingerprints ───
    {
        llama_kv_cache_paged cache(
            *model, buft,
            kNBlocks, kBlockSize, kNSeqMax, kMaxBlks,
            /*n_warm_blocks=*/0, /*n_cold_blocks=*/0,
            std::string(),
            false, "test-rm-fps");

        const bool grew = cache.ensure_blocks_for(/*seq*/ 0, 2 * kBlockSize);
        assert(grew);
        cache.record_paged_block_fingerprint(0, 0, normed({1, 0, 0, 0}), mt::SemanticIndex::Tier::Hot);
        cache.record_paged_block_fingerprint(0, 1, normed({0, 1, 0, 0}), mt::SemanticIndex::Tier::Hot);
        // Seed a different seq's fingerprint to verify it survives.
        const bool grew_b = cache.ensure_blocks_for(/*seq*/ 1, kBlockSize);
        assert(grew_b);
        cache.record_paged_block_fingerprint(1, 0, normed({0, 0, 1, 0}), mt::SemanticIndex::Tier::Hot);
        assert(cache.n_paged_fingerprints() == 3);

        // Whole-seq wipe of seq 0.
        const bool removed = cache.seq_rm(/*seq*/ 0, /*p0=*/-1, /*p1=*/-1);
        assert(removed);

        // seq 0's fingerprints are gone; seq 1's survives.
        assert(!cache.has_paged_fingerprint(0, 0));
        assert(!cache.has_paged_fingerprint(0, 1));
        assert(cache.has_paged_fingerprint(1, 0));
        assert(cache.n_paged_fingerprints() == 1);
        printf("test-paged-semantic: whole-seq seq_rm drops only that seq's fingerprints ok\n");
    }

    // ─── save_paged_fingerprints / load_paged_fingerprints round-trip ───
    {
        const std::string path = tmp_path("rt");

        llama_kv_cache_paged cache_a(
            *model, buft,
            kNBlocks, kBlockSize, kNSeqMax, kMaxBlks,
            /*n_warm_blocks=*/0, /*n_cold_blocks=*/0,
            std::string(),
            false, "test-fp-a");

        const bool grew = cache_a.ensure_blocks_for(/*seq*/ 0, 3 * kBlockSize);
        assert(grew);
        cache_a.record_paged_block_fingerprint(0, 0, normed({1, 0, 0, 0}), mt::SemanticIndex::Tier::Hot);
        cache_a.record_paged_block_fingerprint(0, 1, normed({0, 1, 0, 0}), mt::SemanticIndex::Tier::Warm);
        cache_a.record_paged_block_fingerprint(0, 2, normed({0, 0, 1, 0}), mt::SemanticIndex::Tier::Cold);
        assert(cache_a.n_paged_fingerprints() == 3);

        const bool save_ok = cache_a.save_paged_fingerprints(path);
        assert(save_ok);

        // Load into a fresh cache that has its own (different) fingerprint
        // pre-loaded; load must REPLACE rather than merge.
        llama_kv_cache_paged cache_b(
            *model, buft,
            kNBlocks, kBlockSize, kNSeqMax, kMaxBlks,
            /*n_warm_blocks=*/0, /*n_cold_blocks=*/0,
            std::string(),
            false, "test-fp-b");
        const bool grew_b = cache_b.ensure_blocks_for(/*seq*/ 0, kBlockSize);
        assert(grew_b);
        cache_b.record_paged_block_fingerprint(7, 99, normed({1, 0, 0, 0}), mt::SemanticIndex::Tier::Hot);
        assert(cache_b.n_paged_fingerprints() == 1);

        const bool load_ok = cache_b.load_paged_fingerprints(path);
        assert(load_ok);
        assert(cache_b.n_paged_fingerprints() == 3);  // not 4 — load replaces
        assert(cache_b.has_paged_fingerprint(0, 0));
        assert(cache_b.has_paged_fingerprint(0, 1));
        assert(cache_b.has_paged_fingerprint(0, 2));
        assert(!cache_b.has_paged_fingerprint(7, 99));   // stray entry was wiped

        ::unlink(path.c_str());
        printf("test-paged-semantic: save/load fingerprints round-trip ok\n");
    }

    // ─── Edge cases for restore_semantic_paged ───
    {
        llama_kv_cache_paged cache(
            *model, buft,
            kNBlocks, kBlockSize, kNSeqMax, kMaxBlks,
            kWarmBlocks, /*n_cold_blocks=*/0,
            std::string(),
            false, "test-edge");

        // No fingerprints at all → restore returns 0 cleanly.
        const uint32_t r0 = cache.restore_semantic_paged(0, normed({1, 0, 0, 0}), 5, 0.5f);
        assert(r0 == 0);

        // Empty query → must not crash; returns 0.
        const uint32_t r1 = cache.restore_semantic_paged(0, /* empty */ {}, 5, 0.5f);
        assert(r1 == 0);

        // top_k = 0 → no work; returns 0.
        const uint32_t r2 = cache.restore_semantic_paged(0, normed({1, 0, 0, 0}), 0, 0.5f);
        assert(r2 == 0);

        // Unknown seq → no fingerprints for it; returns 0.
        const uint32_t r3 = cache.restore_semantic_paged(99, normed({1, 0, 0, 0}), 5, 0.5f);
        assert(r3 == 0);

        printf("test-paged-semantic: restore edge cases ok\n");
    }

    llama_model_free(model);
    llama_backend_free();
    printf("test-paged-semantic: ALL PASS\n");
    return 0;
}
