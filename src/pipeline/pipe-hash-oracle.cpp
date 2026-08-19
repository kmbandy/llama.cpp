#include "pipe-hash-oracle.h"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace pipe_expert_dispatcher {

void hash_oracle::register_layer(int32_t         layer,
                                 int32_t         n_expert_used,
                                 int32_t         n_vocab,
                                 int32_t         n_expert,
                                 const int32_t * data) {
    if (layer < 0 || n_expert_used <= 0 || n_vocab <= 0 || n_expert <= 0 ||
        data == nullptr) {
        throw std::invalid_argument(
            "hash_oracle: invalid tid2eid table shape for layer " +
            std::to_string(layer));
    }
    if (n_expert_used > n_expert) {
        throw std::invalid_argument(
            "hash_oracle: tid2eid row is wider than the model's expert count on layer " +
            std::to_string(layer));
    }
    if (find(layer) != nullptr) {
        throw std::invalid_argument(
            "hash_oracle: layer " + std::to_string(layer) + " registered twice");
    }

    table t;
    t.layer         = layer;
    t.n_expert_used = n_expert_used;
    t.n_vocab       = n_vocab;
    t.n_expert      = n_expert;

    const size_t n = (size_t) n_expert_used * (size_t) n_vocab;
    t.data.assign(data, data + n);

    // Validate ONCE at load rather than on every lookup. experts_for() runs per
    // layer per ubatch on the dispatch path; an id check there would be pure
    // per-token cost for a property the table either has or does not have.
    for (size_t i = 0; i < n; ++i) {
        if (t.data[i] >= n_expert) {
            throw std::invalid_argument(
                "hash_oracle: tid2eid on layer " + std::to_string(layer) +
                " selects expert " + std::to_string(t.data[i]) +
                " but the model has " + std::to_string(n_expert));
        }
    }

    tables_.push_back(std::move(t));
    layers_.push_back(layer);
    std::sort(layers_.begin(), layers_.end());
}

const hash_oracle::table * hash_oracle::find(int32_t layer) const {
    for (const table & t : tables_) {
        if (t.layer == layer) {
            return &t;
        }
    }
    return nullptr;
}

bool hash_oracle::experts_for(int32_t                layer,
                              const int32_t *        tokens,
                              size_t                 n_tokens,
                              std::vector<int32_t> & out) const {
    out.clear();

    const table * t = find(layer);
    if (t == nullptr) {
        return false;
    }
    if (tokens == nullptr || n_tokens == 0) {
        return true;
    }

    // Mark-and-sweep over the expert space, not sort-then-unique over the
    // selections. The output must be ascending for the wire, and sweeping
    // 0..n_expert delivers that for free; it is also cheaper the way this is
    // actually called -- a 2048-token prefill ubatch makes 16384 selections into
    // an expert space of 256, so the selections vastly outnumber the experts.
    std::vector<uint8_t> seen((size_t) t->n_expert, 0);

    const size_t stride = (size_t) t->n_expert_used;
    size_t       n_seen = 0;
    for (size_t i = 0; i < n_tokens; ++i) {
        const int32_t token = tokens[i];
        if (token < 0 || token >= t->n_vocab) {
            continue;
        }
        const size_t base = (size_t) token * stride;
        for (size_t k = 0; k < stride; ++k) {
            const int32_t expert_id = t->data[base + k];
            if (expert_id < 0) {
                continue;   // padding for a row that selects fewer than n_expert_used
            }
            if (seen[(size_t) expert_id] == 0) {
                seen[(size_t) expert_id] = 1;
                ++n_seen;
            }
        }
        if (n_seen == (size_t) t->n_expert) {
            break;   // saturated: no later token can add anything
        }
    }

    out.reserve(n_seen);
    for (int32_t expert_id = 0; expert_id < t->n_expert; ++expert_id) {
        if (seen[(size_t) expert_id] != 0) {
            out.push_back(expert_id);
        }
    }
    return true;
}

bool hash_oracle::experts_ranked(int32_t                      layer,
                                 const int32_t *              tokens,
                                 size_t                       n_tokens,
                                 const float *                weights,
                                 std::vector<ranked_expert> & out) const {
    out.clear();

    const table * t = find(layer);
    if (t == nullptr) {
        return false;
    }
    if (tokens == nullptr || n_tokens == 0) {
        return true;
    }

    // miss[e] = prod(1 - w_t) over the tokens t seen so far that select e, i.e.
    // the probability NOTHING that wants this expert survives. Accumulating the
    // complement keeps this a multiply per selection with no branch, and the
    // 1.0 sentinel doubles as "not selected yet" -- an expert no token selects
    // has miss == 1 and drops out of the sweep below for free.
    //
    // No early exit on saturation: unlike the flat union, a later token can
    // still RAISE an expert's confidence after every expert has been touched.
    std::vector<float> miss((size_t) t->n_expert, 1.0f);

    const size_t stride = (size_t) t->n_expert_used;
    for (size_t i = 0; i < n_tokens; ++i) {
        const int32_t token = tokens[i];
        if (token < 0 || token >= t->n_vocab) {
            continue;
        }
        // Clamped, not validated: a confidence is advisory and an out-of-range
        // one must degrade to "certain"/"ignored", never poison the arithmetic.
        float w = weights != nullptr ? weights[i] : 1.0f;
        if (!(w > 0.0f)) {   // also catches NaN
            continue;
        }
        if (w > 1.0f) {
            w = 1.0f;
        }
        const float keep = 1.0f - w;
        const size_t base = (size_t) token * stride;
        for (size_t k = 0; k < stride; ++k) {
            const int32_t expert_id = t->data[base + k];
            if (expert_id < 0) {
                continue;   // padding for a row that selects fewer than n_expert_used
            }
            miss[(size_t) expert_id] *= keep;
        }
    }

    out.reserve(16);
    for (int32_t expert_id = 0; expert_id < t->n_expert; ++expert_id) {
        const float m = miss[(size_t) expert_id];
        if (m >= 1.0f) {
            continue;   // never selected by a token with any weight
        }
        out.push_back({ expert_id, 1.0f - m });
    }
    return true;
}

}  // namespace pipe_expert_dispatcher
