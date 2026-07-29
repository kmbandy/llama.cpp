#pragma once

#include "weight-pager/wp-page-catalog.h"

#include <cstdint>
#include <string>
#include <vector>

namespace wp_repack {

struct ExpertMember {
    uint8_t     role_mask   = 0;
    uint16_t    file_idx    = 0;
    uint64_t    file_offset = 0;
    uint64_t    size        = 0;
    std::string catalog_name;
    std::string source_tensor_name;
};

struct ExpertGroup {
    int                       block_idx  = -1;
    int                       expert_idx = -1;
    uint64_t                  size       = 0;
    std::vector<ExpertMember> members;
};

struct LayerRange {
    int first = -1;
    int last  = -1;
};

struct ShardPlan {
    int                 layer_first = -1;
    int                 layer_last  = -1;
    uint64_t            size        = 0;
    std::vector<size_t> group_indices;
};

std::vector<ExpertGroup> build_expert_groups(const wp::PageCatalog & catalog);

std::vector<LayerRange> parse_layer_ranges(const std::string & text);

std::vector<ShardPlan> plan_shards_by_layer(const std::vector<ExpertGroup> & groups);

std::vector<ShardPlan> plan_shards_max_bytes(const std::vector<ExpertGroup> & groups, uint64_t max_shard_bytes);

// Explicit layer ranges. Any expert group whose layer falls outside `ranges` is
// NOT written to the shard set. That is occasionally wanted (repacking a subset
// for a test) but is a silent-data-loss footgun for the primary use case, where
// the ranges describe a machine split and a mistyped boundary would drop a whole
// layer while still reporting success -- and --verify cannot catch it, because it
// validates what the index claims, not what the model contains.
//
// So incomplete coverage THROWS unless allow_partial is set. The message names the
// uncovered layers.
std::vector<ShardPlan> plan_shards_for_ranges(const std::vector<ExpertGroup> & groups,
                                              const std::vector<LayerRange> &  ranges,
                                              bool                             allow_partial = false);

// Layers present in `groups` that no range covers. Empty == full coverage.
// Exposed separately so callers can report coverage without triggering the throw.
std::vector<int> uncovered_layers(const std::vector<ExpertGroup> & groups,
                                  const std::vector<LayerRange> &  ranges);

}  // namespace wp_repack
