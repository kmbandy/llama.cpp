#include "pipe-prefetch-hints.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace pipe_expert_dispatcher {
namespace {

static constexpr char     NGRAM_MAGIC[8] = { 'W', 'P', 'N', 'G', 'R', 'A', 'M', '\0' };
static constexpr uint32_t NGRAM_VERSION  = 1;

void read_exact(std::ifstream & input, void * data, size_t size) {
    input.read(static_cast<char *>(data), (std::streamsize) size);
    if (!input) {
        throw std::runtime_error("truncated n-gram hint table");
    }
}

uint16_t read_u16(std::ifstream & input) {
    uint8_t data[2];
    read_exact(input, data, sizeof(data));
    return (uint16_t) data[0] | ((uint16_t) data[1] << 8);
}

uint32_t read_u32(std::ifstream & input) {
    uint8_t data[4];
    read_exact(input, data, sizeof(data));
    return (uint32_t) data[0] | ((uint32_t) data[1] << 8) | ((uint32_t) data[2] << 16) | ((uint32_t) data[3] << 24);
}

uint64_t read_u64(std::ifstream & input) {
    uint64_t result = 0;
    for (int shift = 0; shift < 64; shift += 8) {
        uint8_t byte = 0;
        read_exact(input, &byte, 1);
        result |= (uint64_t) byte << shift;
    }
    return result;
}

std::vector<int32_t> rank_top(const std::vector<double> & scores, int32_t top_m) {
    top_m = std::min<int32_t>(PREFETCH_HINT_MAX_EXPERTS,
                              std::max<int32_t>(0, std::min<int32_t>(top_m, (int32_t) scores.size())));
    std::vector<int32_t> ranked(scores.size());
    std::iota(ranked.begin(), ranked.end(), 0);
    std::partial_sort(ranked.begin(), ranked.begin() + top_m, ranked.end(), [&scores](int32_t a, int32_t b) {
        if (scores[(size_t) a] != scores[(size_t) b]) {
            return scores[(size_t) a] > scores[(size_t) b];
        }
        return a < b;
    });
    ranked.resize((size_t) top_m);
    std::sort(ranked.begin(), ranked.end());
    return ranked;
}

// rank_top plus a softmax probability floor. Ported from the whole-expert
// pager's RouterPredictor::predict (wp-router-predictor.cpp), which is the
// version this mechanism was proven in.
//
// The softmax runs over ALL n_expert pooled scores, so the denominator is the
// layer's whole routing mass and the resulting p is comparable across layers --
// that is what makes ONE threshold meaningful for every layer. Scores are
// shifted by the max before exp() for the usual overflow reason.
//
// Emission stops at the FIRST expert below the floor rather than skipping it:
// the candidates are in descending score order, so every later one is lower too.
std::vector<int32_t> rank_top_gated(const std::vector<double> & scores,
                                    const std::vector<double> & logits,
                                    int32_t top_m, float min_conf) {
    if (scores.empty() || logits.size() != scores.size()) {
        return {};
    }
    // RANK on `scores` (the model's own selection rule), GATE on `logits`.
    const double max_score = *std::max_element(logits.begin(), logits.end());
    double denom = 0.0;
    for (const double s : logits) {
        denom += std::exp(s - max_score);
    }
    if (!(denom > 0.0)) {
        denom = 1.0;
    }

    // Rank first, then gate: rank_top already resolves ties deterministically
    // (score desc, then expert id asc), and the hint dedup downstream depends
    // on the surviving set being a pure function of the activations.
    std::vector<int32_t> ranked = rank_top(scores, top_m);
    // rank_top returns ASCENDING ids, so re-order by score to apply the floor.
    std::sort(ranked.begin(), ranked.end(), [&scores](int32_t a, int32_t b) {
        if (scores[(size_t) a] != scores[(size_t) b]) {
            return scores[(size_t) a] > scores[(size_t) b];
        }
        return a < b;
    });

    std::vector<int32_t> kept;
    kept.reserve(ranked.size());
    for (const int32_t expert : ranked) {
        const double p = std::exp(logits[(size_t) expert] - max_score) / denom;
        if (p < (double) min_conf) {
            break;
        }
        kept.push_back(expert);
    }
    std::sort(kept.begin(), kept.end());   // back to the wire's ascending order
    return kept;
}

}  // namespace

std::vector<int32_t> router2_top_experts(const float * weights,
                                         const float * bias,
                                         const float * activations,
                                         int64_t       n_tokens,
                                         int32_t       n_expert,
                                         int32_t       n_embd,
                                         int32_t       top_m,
                                         float         min_conf) {
    if (weights == nullptr || bias == nullptr || activations == nullptr || n_tokens <= 0 || n_expert <= 0 ||
        n_embd <= 0 || top_m <= 0) {
        return {};
    }

    // TWO pooled quantities, and they are NOT interchangeable.
    //
    // `pooled` is DS4's actual selection score, sqrt(softplus(logit)) + bias --
    // that is what the model ranks by, so it is what the hint ranks by.
    //
    // `pooled_logit` is the RAW logit, and it is what the confidence gate needs.
    // Softmaxing the selection score does not work: sqrt(softplus(.)) compresses
    // logits of +-5 into roughly 0.08..2.24, so across 256 experts the largest
    // achievable share is ~0.05 and ANY floor at or above that rejects every
    // expert on every layer. Measured 2026-08-19 before this fix: floors of
    // 0.10, 0.5 and 0.8 all produced identical hint volume, because router2 was
    // emitting nothing at all and the traffic seen was the unrelated hash-layer
    // hints. Raw logits keep their exponential spread, which is what makes a
    // softmax share meaningful and comparable across layers -- and it is the
    // quantity the whole-expert pager gates on (wp-router-predictor.cpp).
    std::vector<double> pooled((size_t) n_expert, -std::numeric_limits<double>::infinity());
    std::vector<double> pooled_logit((size_t) n_expert, -std::numeric_limits<double>::infinity());
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * h = activations + (size_t) token * (size_t) n_embd;
        for (int32_t expert = 0; expert < n_expert; ++expert) {
            const float * row    = weights + (size_t) expert * (size_t) n_embd;
            float         logits = 0.0f;
            for (int32_t i = 0; i < n_embd; ++i) {
                logits += h[i] * row[i];
            }
            const float  softplus   = std::max(logits, 0.0f) + std::log1p(std::exp(-std::fabs(logits)));
            const double score      = (double) std::sqrt(softplus) + (double) bias[expert];
            pooled[(size_t) expert] = std::max(pooled[(size_t) expert], score);
            pooled_logit[(size_t) expert] =
                std::max(pooled_logit[(size_t) expert], (double) logits);
        }
    }
    if (min_conf > 0.0f) {
        return rank_top_gated(pooled, pooled_logit, top_m, min_conf);
    }
    return rank_top(pooled, top_m);
}

uint64_t ngram_hint_table::key(int32_t token, int32_t layer) {
    return ((uint64_t) (uint32_t) token << 32) | (uint32_t) layer;
}

ngram_hint_table::ngram_hint_table(const std::string & path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open n-gram hint table: " + path);
    }

    char magic[sizeof(NGRAM_MAGIC)];
    read_exact(input, magic, sizeof(magic));
    if (!std::equal(std::begin(magic), std::end(magic), std::begin(NGRAM_MAGIC))) {
        throw std::runtime_error("bad n-gram hint table magic");
    }
    const uint32_t version   = read_u32(input);
    const uint32_t n_layers  = read_u32(input);
    const uint32_t n_experts = read_u32(input);
    const uint32_t row_width = read_u32(input);
    const uint64_t n_rows    = read_u64(input);
    if (version != NGRAM_VERSION || n_layers == 0 || n_layers > UINT16_MAX || n_experts == 0 ||
        n_experts > UINT16_MAX || row_width == 0 || row_width > PREFETCH_HINT_MAX_EXPERTS || n_rows > SIZE_MAX) {
        throw std::runtime_error("unsupported n-gram hint table header");
    }
    n_layers_  = (int32_t) n_layers;
    n_experts_ = (int32_t) n_experts;
    row_width_ = (int32_t) row_width;

    const auto read_row = [this, &input]() {
        row result;
        result.total             = read_u32(input);
        const uint16_t n_entries = read_u16(input);
        const uint16_t reserved  = read_u16(input);
        if (result.total == 0 || n_entries == 0 || n_entries > (uint16_t) row_width_ || reserved != 0) {
            throw std::runtime_error("invalid n-gram hint row header");
        }
        result.entries.reserve(n_entries);
        uint64_t stored_total = 0;
        for (uint16_t i = 0; i < n_entries; ++i) {
            entry value;
            value.expert = read_u16(input);
            value.count  = read_u32(input);
            if (value.expert >= (uint16_t) n_experts_ || value.count == 0) {
                throw std::runtime_error("invalid n-gram hint row entry");
            }
            for (const entry & previous : result.entries) {
                if (previous.expert == value.expert) {
                    throw std::runtime_error("duplicate expert in n-gram hint row");
                }
            }
            stored_total += value.count;
            result.entries.push_back(value);
        }
        if (stored_total > result.total) {
            throw std::runtime_error("n-gram hint row counts exceed total");
        }
        return result;
    };

    popularity_.reserve(n_layers_);
    for (int32_t layer = 0; layer < n_layers_; ++layer) {
        popularity_.push_back(read_row());
    }

    rows_.reserve((size_t) n_rows);
    for (uint64_t i = 0; i < n_rows; ++i) {
        const uint32_t token_u32 = read_u32(input);
        const uint16_t layer     = read_u16(input);
        const uint16_t reserved  = read_u16(input);
        if (token_u32 > INT32_MAX || layer >= (uint16_t) n_layers_ || reserved != 0) {
            throw std::runtime_error("invalid n-gram hint key");
        }
        const uint64_t row_key = key((int32_t) token_u32, (int32_t) layer);
        if (!rows_.emplace(row_key, read_row()).second) {
            throw std::runtime_error("duplicate n-gram hint key");
        }
    }

    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("trailing data in n-gram hint table");
    }
    if (!input.eof()) {
        throw std::runtime_error("failed reading n-gram hint table");
    }
}

std::vector<int32_t> ngram_hint_table::top_experts(const int32_t * tokens,
                                                   size_t          n_tokens,
                                                   int32_t         layer,
                                                   int32_t         top_m) const {
    if (tokens == nullptr || n_tokens == 0 || layer < 0 || layer >= n_layers_ || top_m <= 0) {
        return {};
    }
    std::vector<double> scores((size_t) n_experts_, 0.0);
    const row &         pop = popularity_[(size_t) layer];
    for (const entry & value : pop.entries) {
        scores[value.expert] += 1.0e-3 * (double) value.count / (double) pop.total;
    }
    for (size_t i = 0; i < n_tokens; ++i) {
        const auto found = rows_.find(key(tokens[i], layer));
        if (found == rows_.end()) {
            continue;
        }
        const row & token_row = found->second;
        for (const entry & value : token_row.entries) {
            scores[value.expert] += (double) value.count / (double) token_row.total;
        }
    }
    return rank_top(scores, top_m);
}

}  // namespace pipe_expert_dispatcher
