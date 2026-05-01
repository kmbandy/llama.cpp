#include "mt-eviction.h"

#include <algorithm>

namespace mt {

namespace {
constexpr float kAttentionEmaAlpha = 0.3f;  // 0.7 * old + 0.3 * new
}

void TokenMetadataStore::add(llama_pos pos, float initial_attention) {
    std::lock_guard<std::mutex> lk(mu_);
    if (meta_.find(pos) != meta_.end()) return;
    TokenMeta m;
    m.position        = pos;
    m.attention_score = initial_attention;
    m.last_access     = std::time(nullptr);
    m.access_count    = 0;
    meta_.emplace(pos, m);
}

void TokenMetadataStore::record_access(llama_pos pos) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = meta_.find(pos);
    if (it == meta_.end()) {
        TokenMeta m;
        m.position     = pos;
        m.last_access  = std::time(nullptr);
        m.access_count = 1;
        meta_.emplace(pos, m);
    } else {
        ++it->second.access_count;
        it->second.last_access = std::time(nullptr);
    }
}

void TokenMetadataStore::update_attention(llama_pos pos, float score) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = meta_.find(pos);
    if (it == meta_.end()) return;
    it->second.attention_score =
        (1.0f - kAttentionEmaAlpha) * it->second.attention_score +
        kAttentionEmaAlpha * score;
}

void TokenMetadataStore::batch_update_attention(
    const std::vector<llama_pos> & positions,
    const std::vector<float>   & scores) {
    if (positions.size() != scores.size()) return;
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t i = 0; i < positions.size(); ++i) {
        auto it = meta_.find(positions[i]);
        if (it == meta_.end()) continue;
        it->second.attention_score =
            (1.0f - kAttentionEmaAlpha) * it->second.attention_score +
            kAttentionEmaAlpha * scores[i];
    }
}

void TokenMetadataStore::remove(llama_pos pos) {
    std::lock_guard<std::mutex> lk(mu_);
    meta_.erase(pos);
}

size_t TokenMetadataStore::size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return meta_.size();
}

void TokenMetadataStore::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    meta_.clear();
}

std::vector<llama_pos> TokenMetadataStore::get_eviction_candidates(
    EvictionPolicy        policy,
    uint32_t              count,
    const HybridWeights & w) const {
    std::lock_guard<std::mutex> lk(mu_);

    if (meta_.empty() || count == 0) return {};

    std::vector<std::pair<llama_pos, float>> scored;
    scored.reserve(meta_.size());

    const std::time_t now = std::time(nullptr);

    for (const auto & kv : meta_) {
        const TokenMeta & m = kv.second;
        float score = 0.0f;

        switch (policy) {
            case EvictionPolicy::LRU:
                score = (float)(now - m.last_access);
                break;
            case EvictionPolicy::LFU:
                // 1 / (count + 1): zero-access tokens score higher than
                // one-access tokens, so a never-accessed token always
                // evicts before a once-accessed one.
                score = 1.0f / (float)(m.access_count + 1);
                break;
            case EvictionPolicy::Attention:
                score = 1.0f - m.attention_score;
                break;
            case EvictionPolicy::Hybrid: {
                const float att_score  = 1.0f - m.attention_score;
                const float rec_score  = (float)(now - m.last_access) / 60.0f;  // minutes
                const float freq_score = 1.0f / (float)(m.access_count + 1);
                score = w.attention * att_score
                      + w.recency   * rec_score
                      + w.frequency * freq_score;
                break;
            }
        }

        scored.emplace_back(kv.first, score);
    }

    // Highest score first = first to evict.
    std::sort(scored.begin(), scored.end(),
              [](const auto & a, const auto & b) { return a.second > b.second; });

    std::vector<llama_pos> out;
    const size_t n = std::min<size_t>(count, scored.size());
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(scored[i].first);
    }
    return out;
}

}  // namespace mt
