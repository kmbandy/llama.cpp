#include "wp-repack-lib.h"

#include <algorithm>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace wp_repack {

namespace {

struct LayerGroups {
    int                 block_idx = -1;
    uint64_t            size      = 0;
    std::vector<size_t> group_indices;
};

std::vector<LayerGroups> collect_layers(const std::vector<ExpertGroup> & groups) {
    std::vector<LayerGroups> layers;
    for (size_t i = 0; i < groups.size(); ++i) {
        const ExpertGroup & group = groups[i];
        if (layers.empty() || layers.back().block_idx != group.block_idx) {
            layers.push_back({ group.block_idx, 0, {} });
        }
        LayerGroups & layer = layers.back();
        if (group.size > std::numeric_limits<uint64_t>::max() - layer.size) {
            throw std::overflow_error("expert bytes overflow layer total");
        }
        layer.size += group.size;
        layer.group_indices.push_back(i);
    }
    return layers;
}

void append_layer(ShardPlan & shard, const LayerGroups & layer) {
    if (shard.group_indices.empty()) {
        shard.layer_first = layer.block_idx;
    }
    shard.layer_last = layer.block_idx;
    if (layer.size > std::numeric_limits<uint64_t>::max() - shard.size) {
        throw std::overflow_error("expert bytes overflow shard total");
    }
    shard.size += layer.size;
    shard.group_indices.insert(shard.group_indices.end(), layer.group_indices.begin(), layer.group_indices.end());
}

}  // namespace

std::vector<ExpertGroup> build_expert_groups(const wp::PageCatalog & catalog) {
    std::map<std::pair<int, int>, ExpertGroup> grouped;

    for (int i = 0; i < catalog.size(); ++i) {
        const wp::PageMeta & page = catalog.at(i);
        if (!page.is_expert || page.is_consolidated || page.is_pinned) {
            continue;
        }
        if (page.block_idx < 0 || page.expert_idx < 0) {
            throw std::runtime_error("slottable expert page has no valid (block_idx, expert_idx): " + page.tensor_name);
        }

        const std::pair<int, int> key(page.block_idx, page.expert_idx);
        ExpertGroup &             group = grouped[key];
        group.block_idx                 = page.block_idx;
        group.expert_idx                = page.expert_idx;

        ExpertMember member;
        member.role_mask          = page.expert_role_mask;
        member.file_idx           = page.file_idx;
        member.file_offset        = page.file_offset;
        member.size               = page.size;
        member.catalog_name       = page.tensor_name;
        member.source_tensor_name = page.tensor_name;

        if (page.is_sub_expert) {
            if (page.parent_page_idx < 0 || page.parent_page_idx >= catalog.size()) {
                throw std::runtime_error("sub-expert has invalid parent: " + page.tensor_name);
            }
            member.source_tensor_name = catalog.at(page.parent_page_idx).tensor_name;
        }

        if (member.size > std::numeric_limits<uint64_t>::max() - group.size) {
            throw std::overflow_error("expert bytes overflow group total");
        }
        group.size += member.size;
        group.members.push_back(std::move(member));
    }

    std::vector<ExpertGroup> result;
    result.reserve(grouped.size());
    for (auto & item : grouped) {
        ExpertGroup & group = item.second;
        std::sort(group.members.begin(), group.members.end(), [](const ExpertMember & a, const ExpertMember & b) {
            if (a.role_mask != b.role_mask) {
                return a.role_mask < b.role_mask;
            }
            return a.catalog_name < b.catalog_name;
        });
        result.push_back(std::move(group));
    }
    return result;
}

std::vector<LayerRange> parse_layer_ranges(const std::string & text) {
    if (text.empty()) {
        throw std::invalid_argument("layer ranges cannot be empty");
    }

    std::vector<LayerRange> ranges;
    size_t                  start = 0;
    while (start < text.size()) {
        const size_t      comma = text.find(',', start);
        const std::string item  = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        const size_t      dash  = item.find('-');
        if (dash == std::string::npos || dash == 0 || dash + 1 == item.size() ||
            item.find('-', dash + 1) != std::string::npos) {
            throw std::invalid_argument("invalid layer range: " + item);
        }

        size_t          first_used = 0;
        size_t          last_used  = 0;
        const long long first_ll   = std::stoll(item.substr(0, dash), &first_used, 10);
        const long long last_ll    = std::stoll(item.substr(dash + 1), &last_used, 10);
        if (first_used != dash || last_used != item.size() - dash - 1 || first_ll < 0 || last_ll < first_ll ||
            last_ll > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("invalid layer range: " + item);
        }

        LayerRange range{ static_cast<int>(first_ll), static_cast<int>(last_ll) };
        if (!ranges.empty() && range.first <= ranges.back().last) {
            throw std::invalid_argument("layer ranges must be ordered and non-overlapping");
        }
        ranges.push_back(range);

        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
        if (start == text.size()) {
            throw std::invalid_argument("layer ranges cannot end with a comma");
        }
    }
    return ranges;
}

std::vector<ShardPlan> plan_shards_by_layer(const std::vector<ExpertGroup> & groups) {
    std::vector<ShardPlan> result;
    for (const LayerGroups & layer : collect_layers(groups)) {
        ShardPlan shard;
        append_layer(shard, layer);
        result.push_back(std::move(shard));
    }
    return result;
}

std::vector<ShardPlan> plan_shards_max_bytes(const std::vector<ExpertGroup> & groups, uint64_t max_shard_bytes) {
    if (max_shard_bytes == 0) {
        throw std::invalid_argument("max shard bytes must be positive");
    }

    std::vector<ShardPlan> result;
    ShardPlan              current;
    for (const LayerGroups & layer : collect_layers(groups)) {
        const bool would_exceed = current.size > max_shard_bytes || layer.size > max_shard_bytes - current.size;
        if (!current.group_indices.empty() && would_exceed) {
            result.push_back(std::move(current));
            current = ShardPlan();
        }
        append_layer(current, layer);
    }
    if (!current.group_indices.empty()) {
        result.push_back(std::move(current));
    }
    return result;
}

std::vector<int> uncovered_layers(const std::vector<ExpertGroup> & groups,
                                  const std::vector<LayerRange> &  ranges) {
    std::vector<int> missing;
    for (const ExpertGroup & group : groups) {
        bool covered = false;
        for (const LayerRange & range : ranges) {
            if (group.block_idx >= range.first && group.block_idx <= range.last) {
                covered = true;
                break;
            }
        }
        if (!covered && (missing.empty() || missing.back() != group.block_idx)) {
            missing.push_back(group.block_idx);
        }
    }
    std::sort(missing.begin(), missing.end());
    missing.erase(std::unique(missing.begin(), missing.end()), missing.end());
    return missing;
}

std::vector<ShardPlan> plan_shards_for_ranges(const std::vector<ExpertGroup> & groups,
                                              const std::vector<LayerRange> &  ranges,
                                              bool                             allow_partial) {
    if (ranges.empty()) {
        throw std::invalid_argument("at least one layer range is required");
    }

    // Silent partial output is the failure mode this guards: the ranges normally
    // describe a machine split, so a dropped layer produces a shard set that looks
    // complete and fails later at page-in. Refuse unless the caller says it meant it.
    if (!allow_partial) {
        const std::vector<int> missing = uncovered_layers(groups, ranges);
        if (!missing.empty()) {
            std::string msg = "layer ranges do not cover every expert layer; uncovered layers:";
            for (size_t i = 0; i < missing.size(); ++i) {
                msg += (i == 0 ? " " : ", ") + std::to_string(missing[i]);
            }
            msg += " (pass --allow-partial to repack a subset deliberately)";
            throw std::invalid_argument(msg);
        }
    }

    std::vector<ShardPlan> result;
    result.reserve(ranges.size());
    size_t group_index = 0;

    for (const LayerRange & range : ranges) {
        ShardPlan shard;
        shard.layer_first = range.first;
        shard.layer_last  = range.last;

        while (group_index < groups.size() && groups[group_index].block_idx < range.first) {
            ++group_index;
        }
        size_t i = group_index;
        while (i < groups.size() && groups[i].block_idx <= range.last) {
            const ExpertGroup & group = groups[i];
            if (group.size > std::numeric_limits<uint64_t>::max() - shard.size) {
                throw std::overflow_error("expert bytes overflow shard total");
            }
            shard.size += group.size;
            shard.group_indices.push_back(i);
            ++i;
        }
        group_index = i;
        result.push_back(std::move(shard));
    }
    return result;
}

}  // namespace wp_repack
