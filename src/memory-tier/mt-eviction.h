#pragma once

// Eviction-policy scoring for the tiered KV cache.
//
// Per-token bookkeeping (last access, access count, attention score)
// plus a single get_eviction_candidates(policy, n) entry point that
// returns the n tokens scoring highest under the chosen policy.
//
// Policy semantics:
//   LRU       — tokens not recently accessed evict first
//   LFU       — tokens with fewest accesses evict first
//   Attention — tokens with lowest attention scores evict first
//   Hybrid    — weighted combination of the above (default)
//
// Threadsafe via internal mutex. The numeric eviction-policy mapping
// (0..3) matches the long-standing CLI flag values.

#include "mt-config.h"
#include "llama.h"          // llama_pos

#include <cstdint>
#include <ctime>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace mt {

// Per-token state used by every policy. Layout is fixed-size so the
// store's hashmap stays cache-friendly.
struct TokenMeta {
    llama_pos   position         = 0;
    float       attention_score  = 0.0f;
    std::time_t last_access      = 0;
    uint32_t    access_count     = 0;
};

// Default weight blend for Hybrid policy. Same numbers as the legacy
// llama_eviction_score so users see the same scoring on the same
// workload.
struct HybridWeights {
    float attention = 0.5f;
    float recency   = 0.3f;
    float frequency = 0.2f;
};

class TokenMetadataStore {
public:
    TokenMetadataStore() = default;

    // Add a token (no-op if already tracked).
    void add(llama_pos pos, float initial_attention = 0.0f);

    // Bump access count and last_access. Adds the token if not tracked.
    void record_access(llama_pos pos);

    // Update attention via exponential moving average (alpha = 0.3).
    // No-op if pos isn't tracked.
    void update_attention(llama_pos pos, float score);

    // Apply attention updates in batch. positions and scores must have
    // equal sizes; mismatched calls are dropped.
    void batch_update_attention(const std::vector<llama_pos> & positions,
                                 const std::vector<float>   & scores);

    // Drop a token (e.g. context shift erased it).
    void remove(llama_pos pos);

    // Diagnostic: how many tokens are tracked.
    size_t size() const;

    // Reset all metadata.
    void clear();

    // Return up to `count` positions ranked highest-to-lowest under the
    // policy's eviction score (highest score = first to evict).
    //
    // Hybrid uses the supplied weights; pass HybridWeights{} for the
    // default blend. Other policies ignore the weights argument.
    std::vector<llama_pos> get_eviction_candidates(
        EvictionPolicy        policy,
        uint32_t              count,
        const HybridWeights & weights = HybridWeights{}) const;

private:
    mutable std::mutex                            mu_;
    std::unordered_map<llama_pos, TokenMeta>      meta_;
};

}  // namespace mt
