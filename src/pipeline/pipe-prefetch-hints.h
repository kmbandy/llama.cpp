#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace pipe_expert_dispatcher {

static constexpr int32_t PREFETCH_HINT_MAX_EXPERTS = 16;
static constexpr size_t  PREFETCH_HINT_MAX_TOKENS  = 16;

std::vector<int32_t> router2_top_experts(const float * weights,
                                         const float * bias,
                                         const float * activations,
                                         int64_t       n_tokens,
                                         int32_t       n_expert,
                                         int32_t       n_embd,
                                         int32_t       top_m);

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
