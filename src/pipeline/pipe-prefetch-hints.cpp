#include "pipe-prefetch-hints.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
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

// One token's router pass: raw logits, DS4 selection scores (sqrt(softplus)
// + bias), `order` reset to 0..n_expert-1, and the best softmax probability
// over the RAW logits (what the confidence gate reads).
double router2_score_token(const float * h, const float * weights, const float * bias,
                           int32_t n_expert, int32_t n_embd,
                           std::vector<double> & logits, std::vector<double> & scores,
                           std::vector<int32_t> & order) {
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        const float * row = weights + (size_t) expert * (size_t) n_embd;
        float         d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        int32_t       i  = 0;
        for (; i + 3 < n_embd; i += 4) {
            d0 += h[i]     * row[i];
            d1 += h[i + 1] * row[i + 1];
            d2 += h[i + 2] * row[i + 2];
            d3 += h[i + 3] * row[i + 3];
        }
        float dot = d0 + d1 + d2 + d3;
        for (; i < n_embd; ++i) {
            dot += h[i] * row[i];
        }
        logits[(size_t) expert] = (double) dot;
        const float softplus =
            std::max(dot, 0.0f) + std::log1p(std::exp(-std::fabs(dot)));
        scores[(size_t) expert] = (double) std::sqrt(softplus) + (double) bias[expert];
        order[(size_t) expert]  = expert;
        max_logit = std::max(max_logit, logits[(size_t) expert]);
    }
    double denom = 0.0;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        denom += std::exp(logits[(size_t) expert] - max_logit);
    }
    if (!(denom > 0.0)) {
        denom = 1.0;
    }
    double best_p = 0.0;
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        best_p = std::max(best_p, std::exp(logits[(size_t) expert] - max_logit) / denom);
    }
    return best_p;
}

}  // namespace

float router2_margin() {
    static const float value = [] {
        const char * e = std::getenv("WP_HINT_ROUTER2_MARGIN");
        if (e == nullptr || e[0] == '\0') {
            return 0.15f;   // was 0.0 (gate off) in-tree; 0.15 sits in the 0.1-0.2 band the
                            // offline scoring picked (63% useful at 0.2, 26% with no gate)
        }
        const float f = std::strtof(e, nullptr);
        return f > 0.0f ? f : 0.0f;   // 0 = margin gate off (old default)
    }();
    return value;
}

float router2_margin_late() {
    static const float value = [] {
        const char * e = std::getenv("WP_HINT_ROUTER2_MARGIN_LATE");
        if (e == nullptr || e[0] == '\0') {
            return router2_margin();   // unset: rows >= 2 use the same margin
        }
        const float f = std::strtof(e, nullptr);
        return f > 0.0f ? f : 0.0f;
    }();
    return value;
}

std::vector<int32_t> router2_top_experts(const float * weights,
                                         const float * bias,
                                         const float * activations,
                                         int64_t       n_tokens,
                                         int32_t       n_expert,
                                         int32_t       n_embd,
                                         int32_t       top_m,
                                         float         min_conf) {
    router2_scratch scratch;
    return router2_top_experts(weights, bias, activations, n_tokens, n_expert, n_embd,
                               top_m, min_conf, scratch);
}

std::vector<int32_t> router2_top_experts(const float *      weights,
                                         const float *      bias,
                                         const float *      activations,
                                         int64_t            n_tokens,
                                         int32_t            n_expert,
                                         int32_t            n_embd,
                                         int32_t            top_m,
                                         float              min_conf,
                                         router2_scratch &  scratch) {
    if (weights == nullptr || bias == nullptr || activations == nullptr || n_tokens <= 0 || n_expert <= 0 ||
        n_embd <= 0 || top_m <= 0) {
        return {};
    }

    top_m = std::min(top_m, n_expert);
    // Reuse the caller's scratch: resize only grows the underlying buffer
    // (never shrinks capacity), and every slot is overwritten below before
    // it is read, so no explicit clear is needed for logits/scores/order.
    std::vector<int> &     hits   = scratch.hits;
    std::vector<double> &  logits = scratch.logits;
    std::vector<double> &  scores = scratch.scores;
    std::vector<int32_t> & order  = scratch.order;
    hits.assign((size_t) n_expert, 0);
    logits.resize((size_t) n_expert);
    scores.resize((size_t) n_expert);
    order.resize((size_t) n_expert);
    double best_p = 0.0;
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * h = activations + (size_t) token * (size_t) n_embd;
        best_p = std::max(best_p, router2_score_token(h, weights, bias, n_expert, n_embd,
                                                      logits, scores, order));
        // WP_HINT_ROUTER2_MARGIN=x (default 0.15; 0 = off) keeps only experts whose score beats this
        // token's (top_m+1)-th best by at least x. Offline on DS4.1 decode
        // (~/ds4-runs/dsv41/pred/score3.py, L+2 top-6): no gate reads 26%
        // useful at 44% miss recall, 0.2 reads 63% useful at 20% recall.
        const float margin = router2_margin();
        const int32_t n_sort = margin > 0.0f ? std::min(top_m + 1, n_expert) : top_m;
        std::partial_sort(order.begin(), order.begin() + n_sort, order.end(),
                          [&scores](int32_t a, int32_t b) {
                              if (scores[(size_t) a] != scores[(size_t) b]) {
                                  return scores[(size_t) a] > scores[(size_t) b];
                              }
                              return a < b;
                          });
        const double floor_score = margin > 0.0f && n_sort > top_m
            ? scores[(size_t) order[(size_t) top_m]] + (double) margin
            : -std::numeric_limits<double>::infinity();
        for (int32_t i = 0; i < top_m; ++i) {
            if (scores[(size_t) order[(size_t) i]] >= floor_score) {
                ++hits[(size_t) order[(size_t) i]];
            }
        }
    }
    if (min_conf > 0.0f && best_p < (double) min_conf) {
        return {};
    }
    std::vector<int32_t> & kept = scratch.kept;
    kept.clear();
    kept.reserve((size_t) n_expert);
    for (int32_t expert = 0; expert < n_expert; ++expert) {
        if (hits[(size_t) expert] > 0) {
            kept.push_back(expert);
        }
    }
    if (kept.size() > (size_t) PREFETCH_HINT_MAX_EXPERTS) {
        std::nth_element(kept.begin(),
                         kept.begin() + PREFETCH_HINT_MAX_EXPERTS, kept.end(),
                         [&hits](int32_t a, int32_t b) {
                             if (hits[(size_t) a] != hits[(size_t) b]) {
                                 return hits[(size_t) a] > hits[(size_t) b];
                             }
                             return a < b;
                         });
        kept.resize((size_t) PREFETCH_HINT_MAX_EXPERTS);
        std::sort(kept.begin(), kept.end());
    }
    // Caller receives its own copy -- scratch.kept is overwritten next call.
    return std::vector<int32_t>(kept.begin(), kept.end());
}

std::vector<std::vector<int32_t>> router2_row_tiers(const float *     weights,
                                                    const float *     bias,
                                                    const float *     activations,
                                                    int64_t           n_tokens,
                                                    int32_t           n_expert,
                                                    int32_t           n_embd,
                                                    int32_t           top_m,
                                                    int32_t           row_cap,
                                                    float             min_conf,
                                                    float             margin,
                                                    float             margin_late,
                                                    size_t            total_cap,
                                                    router2_scratch & scratch) {
    std::vector<std::vector<int32_t>> tiers;
    if (weights == nullptr || bias == nullptr || activations == nullptr || n_tokens <= 0 || n_expert <= 0 ||
        n_embd <= 0 || top_m <= 0) {
        return tiers;
    }
    top_m = std::min(top_m, n_expert);
    const int32_t cap = row_cap > 0 ? std::min(row_cap, top_m) : top_m;

    std::vector<int> &     seen   = scratch.hits;   // dedup across rows
    std::vector<double> &  logits = scratch.logits;
    std::vector<double> &  scores = scratch.scores;
    std::vector<int32_t> & order  = scratch.order;
    seen.assign((size_t) n_expert, 0);
    logits.resize((size_t) n_expert);
    scores.resize((size_t) n_expert);
    order.resize((size_t) n_expert);

    // Score every row first: the confidence gate is all-or-nothing on the
    // layer (best p over ALL rows), so a layer that fails it must emit
    // nothing, not just row 0's share.
    double best_p = 0.0;
    std::vector<std::vector<int32_t>> ranked((size_t) n_tokens);
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * h = activations + (size_t) token * (size_t) n_embd;
        best_p = std::max(best_p, router2_score_token(h, weights, bias, n_expert, n_embd,
                                                      logits, scores, order));
        const float   m      = token >= 2 ? margin_late : margin;
        const int32_t n_sort = m > 0.0f ? std::min(cap + 1, n_expert) : cap;
        std::partial_sort(order.begin(), order.begin() + n_sort, order.end(),
                          [&scores](int32_t a, int32_t b) {
                              if (scores[(size_t) a] != scores[(size_t) b]) {
                                  return scores[(size_t) a] > scores[(size_t) b];
                              }
                              return a < b;
                          });
        const double floor_score = m > 0.0f && n_sort > cap
            ? scores[(size_t) order[(size_t) cap]] + (double) m
            : -std::numeric_limits<double>::infinity();
        // Best-ranked first: a total_cap cut below drops the weakest picks.
        for (int32_t i = 0; i < cap; ++i) {
            if (scores[(size_t) order[(size_t) i]] >= floor_score) {
                ranked[(size_t) token].push_back(order[(size_t) i]);
            }
        }
    }
    if (min_conf > 0.0f && best_p < (double) min_conf) {
        return tiers;
    }
    // Row 0 first, then each later row's experts not already claimed. Every
    // row keeps its own quota (nothing here ranks rows against each other),
    // which is the point: a vote-ranked global cap starves late rows.
    size_t total = 0;
    for (int64_t token = 0; token < n_tokens; ++token) {
        std::vector<int32_t> tier;
        for (const int32_t expert : ranked[(size_t) token]) {
            if (total_cap != 0 && total >= total_cap) {
                break;
            }
            if (seen[(size_t) expert] == 0) {
                seen[(size_t) expert] = 1;
                tier.push_back(expert);
                ++total;
            }
        }
        if (!tier.empty()) {
            std::sort(tier.begin(), tier.end());   // the wire's ascending order
            tiers.push_back(std::move(tier));
        }
    }
    return tiers;
}

void router2_trace_scores(const float *     weights,
                          const float *     bias,
                          const float *     activations,
                          int64_t           n_tokens,
                          int32_t           n_expert,
                          int32_t           n_embd,
                          int32_t           top_n,
                          router2_scratch & scratch,
                          uint16_t *        ids_out,
                          float *           score_out,
                          float *           prob_out) {
    if (weights == nullptr || bias == nullptr || activations == nullptr || n_tokens <= 0 || n_expert <= 0 ||
        n_embd <= 0 || top_n <= 0) {
        return;
    }
    std::vector<double> &  logits = scratch.logits;
    std::vector<double> &  scores = scratch.scores;
    std::vector<int32_t> & order  = scratch.order;
    logits.resize((size_t) n_expert);
    scores.resize((size_t) n_expert);
    order.resize((size_t) n_expert);
    const int32_t n_sort = std::min(top_n, n_expert);
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * h = activations + (size_t) token * (size_t) n_embd;
        router2_score_token(h, weights, bias, n_expert, n_embd, logits, scores, order);
        std::partial_sort(order.begin(), order.begin() + n_sort, order.end(),
                          [&scores](int32_t a, int32_t b) {
                              if (scores[(size_t) a] != scores[(size_t) b]) {
                                  return scores[(size_t) a] > scores[(size_t) b];
                              }
                              return a < b;
                          });
        const double max_logit = *std::max_element(logits.begin(), logits.end());
        double       denom     = 0.0;
        for (const double l : logits) {
            denom += std::exp(l - max_logit);
        }
        if (!(denom > 0.0)) {
            denom = 1.0;
        }
        const size_t base = (size_t) token * (size_t) top_n;
        for (int32_t i = 0; i < top_n; ++i) {
            if (i < n_sort) {
                const int32_t e = order[(size_t) i];
                ids_out[base + (size_t) i]   = (uint16_t) e;
                score_out[base + (size_t) i] = (float) scores[(size_t) e];
                prob_out[base + (size_t) i]  = (float) (std::exp(logits[(size_t) e] - max_logit) / denom);
            } else {
                ids_out[base + (size_t) i]   = 0xFFFF;
                score_out[base + (size_t) i] = -std::numeric_limits<float>::infinity();
                prob_out[base + (size_t) i]  = 0.0f;
            }
        }
    }
}

bool parse_pscore_file(const std::string & path, pscore_model & out, std::string * err) {
    static const char * const names[PSF_COUNT] = { "min_d", "rank", "margin", "prob", "n_dist", "layer", "gap", "row", "age" };
    const auto fail = [&](const std::string & m) {
        if (err != nullptr) {
            *err = m;
        }
        return false;
    };
    std::ifstream in(path);
    if (!in) {
        return fail("cannot open " + path);
    }
    pscore_model m;
    std::string  line;
    size_t       lineno = 0;
    while (std::getline(in, line)) {
        ++lineno;
        const size_t a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos || line[a] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string        tok;
        ss >> tok;
        const std::string where = path + ":" + std::to_string(lineno) + ": ";
        if (tok == "bias") {
            if (!(ss >> m.bias)) {
                return fail(where + "bad bias");
            }
            continue;
        }
        int factor = -1;
        for (int i = 0; i < PSF_COUNT; ++i) {
            if (tok == names[i]) {
                factor = i;
            }
        }
        pscore_model::bucket b{};
        b.factor = factor;
        if (factor < 0) {
            return fail(where + "unknown factor '" + tok + "'");
        }
        if (!(ss >> b.lo >> b.hi >> b.w) || !(b.lo < b.hi)) {
            return fail(where + "expected '<factor> <lo> <hi> <weight>' with lo < hi");
        }
        m.buckets.push_back(b);
    }
    if (m.buckets.empty()) {
        return fail(path + ": no buckets");
    }
    out = std::move(m);
    return true;
}

double pscore_eval(const pscore_model & model, const pscore_features & f) {
    double z = model.bias;
    for (const pscore_model::bucket & b : model.buckets) {
        const float x = f.v[b.factor];
        if (b.lo <= x && x < b.hi) {
            z += b.w;
        }
    }
    return 1.0 / (1.0 + std::exp(-z));
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
