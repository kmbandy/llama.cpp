#pragma once

// SemanticIndex — optional fingerprint store + cosine-similarity
// scoring for tier-prefetch hints.
//
// When the user supplies --kv-tier-semantic-index PATH, the integration
// code loads a CPU-side embedding model (e.g. bge-small) separately and
// passes already-computed L2-normalized embeddings into this index.
// This keeps the model-loading concerns out of mt-semantic and lets
// the scoring layer stay simple.
//
// Workflow:
//   1. On eviction of a token range, integration code:
//        - extracts the original text for those positions,
//        - embeds it via the CPU embedding model,
//        - calls add_fingerprint(positions, embedding, tier).
//   2. On a new query, integration code embeds the query text and calls
//      score(query_embedding, top_k, threshold). Returned hints carry
//      positions whose stored embeddings are most similar to the query.
//   3. Integration code uses the hints to issue prefetch warm/cold ->
//      hot moves.
//
// All methods threadsafe via internal mutex. The fingerprint vector is
// capped at kMaxFingerprints to bound memory.

#include "llama.h"  // llama_pos

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace mt {

class SemanticIndex {
public:
    static constexpr size_t kMaxFingerprints = 1024;  // FIFO cap

    enum class Tier : uint8_t { Hot = 0, Warm = 1, Cold = 2 };

    struct Fingerprint {
        std::vector<llama_pos> positions;
        std::vector<float>     embedding;  // expected L2-normalized
        Tier                   tier   = Tier::Cold;
        uint64_t               turn   = 0;     // monotonic insertion counter
    };

    struct Hint {
        std::vector<llama_pos> positions;
        float                  score = 0.0f;  // cosine similarity in [-1, 1]
        Tier                   tier  = Tier::Cold;
    };

    SemanticIndex() = default;

    // Add a fingerprint. The caller-supplied embedding SHOULD be
    // L2-normalized; if it isn't, scoring degrades to dot-product
    // rather than cosine.
    //
    // When the index reaches kMaxFingerprints, the oldest entry
    // (lowest `turn`) is evicted before the new one is inserted.
    void add_fingerprint(std::vector<llama_pos> positions,
                          std::vector<float>     embedding,
                          Tier                   tier);

    // Save / load fingerprints to disk. Format is a simple binary blob;
    // the file is rewritten every save_to_disk call. Returns false on
    // I/O error.
    bool save_to_disk(const std::string & path) const;
    bool load_from_disk(const std::string & path);

    // Score the index against a query embedding. Returns up to `top_k`
    // hints whose cosine similarity >= `threshold`, sorted by descending
    // score.
    std::vector<Hint> score(const std::vector<float> & query_embedding,
                             int                        top_k,
                             float                      threshold) const;

    // Drop everything. Useful for tests.
    void   clear();
    size_t size() const;

private:
    mutable std::mutex          mu_;
    std::vector<Fingerprint>    fingerprints_;
    uint64_t                     next_turn_ = 0;
};

}  // namespace mt
