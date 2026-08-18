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

namespace {

int64_t parse_positive_i64(const std::string & item, const char * what) {
    if (item.empty()) {
        throw std::invalid_argument(std::string("empty ") + what + " in expert slice spec");
    }
    size_t    used  = 0;
    long long value = 0;
    try {
        value = std::stoll(item, &used, 10);
    } catch (const std::exception &) {
        throw std::invalid_argument(std::string("invalid ") + what + " in expert slice spec: " + item);
    }
    if (used != item.size() || value <= 0) {
        throw std::invalid_argument(std::string("invalid ") + what + " in expert slice spec: " + item);
    }
    return static_cast<int64_t>(value);
}

std::vector<std::string> split_on(const std::string & text, char sep) {
    std::vector<std::string> items;
    size_t                   start = 0;
    while (true) {
        const size_t pos = text.find(sep, start);
        if (pos == std::string::npos) {
            items.push_back(text.substr(start));
            return items;
        }
        items.push_back(text.substr(start, pos - start));
        start = pos + 1;
    }
}

}  // namespace

SliceSpec parse_slice_spec(const std::string & text) {
    if (text.empty()) {
        throw std::invalid_argument("expert slice spec cannot be empty");
    }

    const bool has_colon = text.find(':') != std::string::npos;
    const bool has_comma = text.find(',') != std::string::npos;
    if (has_colon && has_comma) {
        throw std::invalid_argument("expert slice spec mixes ':' (ratios) and ',' (explicit widths): " + text);
    }

    SliceSpec spec;
    spec.text = text;

    const std::vector<std::string> items = split_on(text, has_colon ? ':' : ',');
    if (items.size() < 2) {
        // A one-way split is a v1 repack with extra steps, and almost certainly a typo.
        throw std::invalid_argument("expert slice spec needs at least two slices: " + text);
    }

    if (has_colon) {
        spec.from_ratios = true;
        for (const std::string & item : items) {
            spec.ratios.push_back(parse_positive_i64(item, "ratio"));
        }
    } else {
        for (const std::string & item : items) {
            spec.widths.push_back(parse_positive_i64(item, "width"));
        }
    }
    return spec;
}

void resolve_slice_widths(SliceSpec & spec, int64_t n_ff, int64_t blck) {
    if (blck <= 0) {
        throw std::invalid_argument("expert tensor quant block size must be positive");
    }
    if (n_ff <= 0 || n_ff % blck != 0) {
        // Nothing downstream can fix this: no slicing of this tensor lands on
        // block boundaries, so refuse rather than emit re-cut blocks.
        throw std::invalid_argument("FFN intermediate size " + std::to_string(n_ff) +
                                    " is not a multiple of the quant block size " + std::to_string(blck));
    }

    const int64_t total_units = n_ff / blck;

    if (spec.from_ratios) {
        const size_t n = spec.ratios.size();
        if (static_cast<int64_t>(n) > total_units) {
            throw std::invalid_argument("expert slice spec asks for " + std::to_string(n) + " slices but only " +
                                        std::to_string(total_units) + " quant block(s) are available to split");
        }

        int64_t ratio_sum = 0;
        for (const int64_t r : spec.ratios) {
            if (r > std::numeric_limits<int64_t>::max() - ratio_sum) {
                throw std::overflow_error("expert slice ratios overflow");
            }
            ratio_sum += r;
        }

        // Largest-remainder apportionment in units of one quant block, with a
        // floor of one block per slice. Deterministic, and the assigned units
        // always sum to total_units exactly, so the widths always cover n_ff.
        std::vector<int64_t> units(n, 0);
        std::vector<int64_t> remainder(n, 0);
        int64_t              assigned = 0;
        for (size_t i = 0; i < n; ++i) {
            const int64_t scaled = total_units * spec.ratios[i];
            units[i]             = std::max<int64_t>(1, scaled / ratio_sum);
            remainder[i]         = scaled % ratio_sum;
            assigned += units[i];
        }
        if (assigned > total_units) {
            throw std::invalid_argument("expert slice ratios cannot be met: the one-block floor for " +
                                        std::to_string(n) + " slices already exceeds " + std::to_string(total_units) +
                                        " block(s)");
        }

        std::vector<size_t> order(n);
        for (size_t i = 0; i < n; ++i) {
            order[i] = i;
        }
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return remainder[a] > remainder[b]; });
        for (size_t k = 0; assigned < total_units; ++k) {
            units[order[k % n]] += 1;
            ++assigned;
        }

        spec.widths.assign(n, 0);
        for (size_t i = 0; i < n; ++i) {
            spec.widths[i] = units[i] * blck;
        }
    }

    if (spec.widths.size() < 2) {
        throw std::invalid_argument("expert slice spec needs at least two slices");
    }

    int64_t sum = 0;
    for (size_t i = 0; i < spec.widths.size(); ++i) {
        const int64_t w = spec.widths[i];
        if (w <= 0) {
            throw std::invalid_argument("expert slice width must be positive (slice " + std::to_string(i) + ")");
        }
        if (w % blck != 0) {
            throw std::invalid_argument("expert slice width " + std::to_string(w) + " (slice " + std::to_string(i) +
                                        ") is not a multiple of the quant block size " + std::to_string(blck) +
                                        "; a slice boundary inside a quant block would cut the block");
        }
        if (w > n_ff - sum) {
            throw std::invalid_argument("expert slice widths exceed the FFN intermediate size " +
                                        std::to_string(n_ff));
        }
        sum += w;
    }
    if (sum != n_ff) {
        throw std::invalid_argument("expert slice widths sum to " + std::to_string(sum) + " but the FFN intermediate "
                                    "size is " + std::to_string(n_ff) + "; every element must be covered exactly once");
    }
}

std::vector<SliceRange> slice_ranges(const std::vector<int64_t> & widths) {
    std::vector<SliceRange> ranges;
    ranges.reserve(widths.size());
    int64_t cursor = 0;
    for (size_t i = 0; i < widths.size(); ++i) {
        SliceRange range;
        range.index = static_cast<int>(i);
        range.first = cursor;
        range.last  = cursor + widths[i];
        cursor      = range.last;
        ranges.push_back(range);
    }
    return ranges;
}

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
