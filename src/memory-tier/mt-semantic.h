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
#include <unordered_map>
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

// BlockSemanticIndex — paged-block-keyed fingerprint store for MAD-122.
//
// Parallel to SemanticIndex but keys fingerprints by (seq_id, logical_block_idx)
// so a query can find which paged blocks of a given seq are most semantically
// similar. The paged path stores at most one fingerprint per block (the 16-tok
// block size is small enough that one embedding per block is reasonable),
// matched 1:1 with BlockTable's lifecycle: blocks come and go with the seq;
// fingerprints are dropped on whole-seq wipe (mt::seq_rm with sentinel range).
//
// No FIFO cap: lifecycle is tied to BlockTable, so memory grows with the
// active context size rather than indefinitely. For the army goal (4 seqs ×
// 8k blocks/seq × 384-dim fp32) the worst-case footprint is ~48 MiB —
// acceptable for a CPU-side index and well below the warm-tier staging cost.
class BlockSemanticIndex {
public:
    struct BlockHint {
        llama_seq_id        seq_id = -1;
        uint32_t            lblock = 0;
        float               score  = 0.0f;
        SemanticIndex::Tier tier   = SemanticIndex::Tier::Warm;
    };

    BlockSemanticIndex() = default;

    // Store the fingerprint for (seq_id, lblock). Overwrites any prior
    // fingerprint at the same key — useful if a block is re-fingerprinted
    // after a partial-range edit. embedding SHOULD be L2-normalized
    // (caller's responsibility); scoring degrades to dot-product if it
    // isn't.
    void add_fingerprint(llama_seq_id        seq_id,
                          uint32_t            lblock,
                          std::vector<float>  embedding,
                          SemanticIndex::Tier tier);

    // Update only the tier annotation (e.g. when a block migrates
    // hot→warm→cold). No-op if the (seq, lblock) isn't tracked.
    void update_tier(llama_seq_id seq_id, uint32_t lblock, SemanticIndex::Tier tier);

    // Drop a single (seq, lblock) entry. No-op if not tracked.
    void remove_block(llama_seq_id seq_id, uint32_t lblock);

    // MAD-129: O(1) check whether a fingerprint already exists for
    // (seq_id, lblock). Used by the server's prefill-time write trigger
    // to skip blocks that have already been fingerprinted (typical when
    // a turn's prompt re-processes a prior turn's accumulated context).
    bool has_fingerprint(llama_seq_id seq_id, uint32_t lblock) const;

    // Drop every fingerprint for `seq_id`. Called on whole-seq wipe.
    void remove_seq(llama_seq_id seq_id);

    // Drop everything.
    void clear();

    // Score `seq_id`'s blocks against `query_embedding`. Returns up to
    // `top_k` blocks with cosine similarity >= `threshold`, sorted by
    // descending score. Blocks from other seqs are not considered —
    // semantic prefetch is per-seq because cross-seq attention isn't a
    // thing in the paged-attention model.
    std::vector<BlockHint> score(llama_seq_id              seq_id,
                                  const std::vector<float> & query_embedding,
                                  int                        top_k,
                                  float                      threshold) const;

    // Diagnostics.
    size_t size() const;
    size_t size(llama_seq_id seq_id) const;

private:
    struct Entry {
        std::vector<float>  embedding;
        SemanticIndex::Tier tier = SemanticIndex::Tier::Warm;
    };

    mutable std::mutex                                                          mu_;
    std::unordered_map<llama_seq_id, std::unordered_map<uint32_t, Entry>>       fps_;
};

}  // namespace mt
