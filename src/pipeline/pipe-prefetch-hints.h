#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipe_expert_dispatcher {

static constexpr int32_t PREFETCH_HINT_MAX_EXPERTS = 16;
static constexpr size_t  PREFETCH_HINT_MAX_TOKENS  = 16;

// Score a target layer's experts by applying ITS router to the activations of
// an earlier layer, max-pooled over token positions, and return the top-M
// ASCENDING (the wire's dedup invariant).
//
// min_conf is a SOFTMAX PROBABILITY FLOOR over all n_expert pooled scores: an
// expert is emitted only if its share of the routing mass clears it. 0 = no
// gate, which is what this function did before and is kept for the A/B.
//
// WHY THE GATE IS NOT OPTIONAL IN PRACTICE. The whole-expert pager shipped this
// exact predictor without a confidence floor and it lost: taking a flat top-M
// means that on a layer where the router is UNDECIDED you still fetch M experts,
// and widening M only reaches deeper into low-probability ones. Measured there
// (2026-07-22 and -27): +12-14% NVMe bytes for a 2-3.6% hit rate, with M=4
// scoring WORSE than M=2. With the floor, a peaked layer emits its few real
// candidates and an undecided layer emits NOTHING -- which is the only way a
// speculative read is affordable on a concurrency-bound drive.
std::vector<int32_t> router2_top_experts(const float * weights,
                                         const float * bias,
                                         const float * activations,
                                         int64_t       n_tokens,
                                         int32_t       n_expert,
                                         int32_t       n_embd,
                                         int32_t       top_m,
                                         float         min_conf = 0.0f);

// WPNGRAM v1 is little-endian: header(version, dimensions, row count), one
// popularity row per layer, then keyed token rows. Each row stores its full
// count total plus up to 16 (u16 expert, u32 count) entries.
class ngram_hint_table {
  public:
    explicit ngram_hint_table(const std::string & path);

    int32_t n_layers() const { return n_layers_; }

    int32_t n_experts() const { return n_experts_; }

    int32_t row_width() const { return row_width_; }

    size_t row_count() const { return rows_.size(); }

    std::vector<int32_t> top_experts(const int32_t * tokens, size_t n_tokens, int32_t layer, int32_t top_m) const;

  private:
    struct entry {
        uint16_t expert = 0;
        uint32_t count  = 0;
    };

    struct row {
        uint32_t           total = 0;
        std::vector<entry> entries;
    };

    static uint64_t key(int32_t token, int32_t layer);

    int32_t                           n_layers_  = 0;
    int32_t                           n_experts_ = 0;
    int32_t                           row_width_ = 0;
    std::vector<row>                  popularity_;
    std::unordered_map<uint64_t, row> rows_;
};

}  // namespace pipe_expert_dispatcher
