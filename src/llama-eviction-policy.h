#pragma once

#include "llama.h"
#include <chrono>
#include <vector>
#include <unordered_map>
#include <ctime>
#include <algorithm>

// Eviction policy types
enum llama_eviction_policy {
    LRU,           // Least Recently Used
    LFU,           // Least Frequently Used
    ATTENTION,     // Attention-based eviction
    HYBRID         // Attention + recency + frequency (default)
};

// Eviction score components
struct llama_eviction_score {
    float attention_weight;    // 0.0-1.0, lower = less important
    float recency_weight;      // Based on time since last access
    float frequency_weight;    // Based on access frequency

    // Default weights for hybrid policy
    static constexpr float DEFAULT_ATTENTION_WEIGHT = 0.5f;
    static constexpr float DEFAULT_RECENCY_WEIGHT = 0.3f;
    static constexpr float DEFAULT_FREQUENCY_WEIGHT = 0.2f;

    // Configurable weights for hybrid policy
    float att_w = DEFAULT_ATTENTION_WEIGHT;
    float rec_w = DEFAULT_RECENCY_WEIGHT;
    float freq_w = DEFAULT_FREQUENCY_WEIGHT;

    // Calculate combined eviction score (higher = more likely to evict)
    float calculate() const {
        return att_w * attention_weight +
               rec_w * recency_weight +
               freq_w * frequency_weight;
    }

    // Calculate with custom weights
    float calculate(float att_w, float rec_w, float freq_w) const {
        return att_w * attention_weight +
               rec_w * recency_weight +
               freq_w * frequency_weight;
    }
};

// Token metadata for eviction decisions
struct llama_token_metadata {
    llama_pos position;
    float attention_score;
    std::time_t last_access;
    uint32_t access_count;

    llama_token_metadata() : position(0), attention_score(0.0f),
                             last_access(std::time(nullptr)), access_count(0) {}

    llama_token_metadata(llama_pos pos, float att_score = 0.0f)
        : position(pos), attention_score(att_score),
          last_access(std::time(nullptr)), access_count(0) {}

    void record_access() {
        access_count++;
        last_access = std::time(nullptr);
    }

    void update_attention_score(float score) {
        // Exponential moving average for attention score
        attention_score = 0.7f * attention_score + 0.3f * score;
    }
};

// Token metadata store for eviction tracking
class llama_token_metadata_store {
public:
    using pos_to_metadata = std::unordered_map<llama_pos, llama_token_metadata>;

    void record_access(llama_pos pos) {
        auto it = metadata.find(pos);
        if (it != metadata.end()) {
            it->second.record_access();
        } else {
            metadata.emplace(pos, llama_token_metadata(pos));
        }
    }

    void update_attention(llama_pos pos, float attention_score) {
        auto it = metadata.find(pos);
        if (it != metadata.end()) {
            it->second.update_attention_score(attention_score);
        }
    }

    void add_token(llama_pos pos, float initial_attention = 0.0f) {
        if (metadata.find(pos) == metadata.end()) {
            metadata.emplace(pos, llama_token_metadata(pos, initial_attention));
        }
    }

    void remove_token(llama_pos pos) {
        metadata.erase(pos);
    }

    const llama_token_metadata* get_metadata(llama_pos pos) const {
        auto it = metadata.find(pos);
        return it != metadata.end() ? &it->second : nullptr;
    }

    // Batch attention update for efficiency
    void batch_update_attention(const std::vector<llama_pos>& positions,
                                 const std::vector<float>& attention_scores) {
        if (positions.size() != attention_scores.size()) {
            return;
        }
        for (size_t i = 0; i < positions.size(); i++) {
            update_attention(positions[i], attention_scores[i]);
        }
    }

    // Get tokens below attention threshold (candidates for eviction)
    std::vector<llama_pos> get_low_attention_tokens(float threshold) const {
        std::vector<llama_pos> result;
        for (const auto& kv : metadata) {
            if (kv.second.attention_score < threshold) {
                result.push_back(kv.first);
            }
        }
        return result;
    }

    // Normalize attention scores across all tokens
    void normalize_attention_scores() {
        if (metadata.empty()) return;

        // Find max attention score
        float max_score = 0.0f;
        for (const auto& kv : metadata) {
            max_score = std::max(max_score, kv.second.attention_score);
        }

        if (max_score > 0.0f) {
            for (auto& kv : metadata) {
                kv.second.attention_score /= max_score;
            }
        }
    }

    // Clear all metadata
    void clear_all() {
        metadata.clear();
    }

    // Get number of tracked tokens
    size_t get_token_count() const {
        return metadata.size();
    }

    // Get tokens sorted by eviction score (highest first)
    std::vector<llama_pos> get_eviction_candidates(
        llama_eviction_policy policy,
        uint32_t count,
        float attention_weight = llama_eviction_score::DEFAULT_ATTENTION_WEIGHT,
        float recency_weight = llama_eviction_score::DEFAULT_RECENCY_WEIGHT,
        float frequency_weight = llama_eviction_score::DEFAULT_FREQUENCY_WEIGHT) const {

        std::vector<std::pair<llama_pos, float>> scored_tokens;
        scored_tokens.reserve(metadata.size());

        const auto now = std::time(nullptr);

        for (const auto& kv : metadata) {
            const auto& pos = kv.first;
            const auto& meta = kv.second;
            float score = 0.0f;

            switch (policy) {
                case LRU:
                    // Score based on recency (older = higher score)
                    score = float(now - meta.last_access);
                    break;

                case LFU:
                    // Score based on frequency (less frequent = higher score)
                    score = meta.access_count > 0 ? 1.0f / float(meta.access_count) : 1.0f;
                    break;

                case ATTENTION:
                    // Score based on attention (lower attention = higher score)
                    score = 1.0f - meta.attention_score;
                    break;

                case HYBRID:
                default:
                    // Combined score
                    float att_score = 1.0f - meta.attention_score;
                    float rec_score = float(now - meta.last_access) / 60.0f; // Normalize to minutes
                    float freq_score = meta.access_count > 0 ? 1.0f / float(meta.access_count) : 1.0f;
                    score = att_score * attention_weight +
                            rec_score * recency_weight +
                            freq_score * frequency_weight;
                    break;
            }

            scored_tokens.emplace_back(pos, score);
        }

        // Sort by score (highest first)
        std::sort(scored_tokens.begin(), scored_tokens.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // Return top 'count' positions
        std::vector<llama_pos> result;
        size_t n = std::min(count, uint32_t(scored_tokens.size()));
        result.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            result.push_back(scored_tokens[i].first);
        }

        return result;
    }

    size_t size() const { return metadata.size(); }
    void clear() { metadata.clear(); }

private:
    pos_to_metadata metadata;
};

// Layer importance for per-layer eviction decisions
struct llama_layer_importance {
    float importance;  // 0.0-1.0, higher = more important to keep

    static constexpr float DEFAULT_IMPORTANCE = 1.0f;
};
