#include "wp-repack-lib.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int failures = 0;

#define EXPECT_TRUE(condition, message)                                      \
    do {                                                                     \
        if (!(condition)) {                                                  \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            ++failures;                                                      \
        }                                                                    \
    } while (0)

#define EXPECT_EQ(actual, expected, message)                                               \
    do {                                                                                   \
        const auto actual_value   = (actual);                                              \
        const auto expected_value = (expected);                                            \
        if (actual_value != expected_value) {                                              \
            std::fprintf(stderr, "FAIL: %s: got %llu, expected %llu (line %d)\n", message, \
                         static_cast<unsigned long long>(actual_value),                    \
                         static_cast<unsigned long long>(expected_value), __LINE__);       \
            ++failures;                                                                    \
        }                                                                                  \
    } while (0)

static void test_grouping() {
    wp::PageCatalog catalog;

    catalog.add_consolidated_experts("blk.1.ffn_gate_exps.weight", 0, 1000, 4 * 120, 4);
    catalog.add_consolidated_experts("blk.1.ffn_up_exps.weight", 1, 2000, 4 * 80, 4);
    catalog.add_consolidated_experts("blk.1.ffn_down_exps.weight", 2, 3000, 4 * 160, 4);

    catalog.add("blk.2.ffn_up.7.weight", 0, 4000, 91);
    catalog.add("blk.2.ffn_down.7.weight", 1, 5000, 137);

    catalog.add("blk.2.ffn_gate_inp.weight", 0, 6000, 32);
    catalog.add("blk.2.ffn_up_shexp.weight", 0, 7000, 64);
    catalog.add("blk.2.ffn_down_shexp.weight", 0, 8000, 64);

    const std::vector<wp_repack::ExpertGroup> groups = wp_repack::build_expert_groups(catalog);

    EXPECT_EQ(groups.size(), 5u, "four consolidated experts plus one native expert");

    for (size_t expert = 0; expert < 4 && expert < groups.size(); ++expert) {
        const wp_repack::ExpertGroup & group = groups[expert];
        EXPECT_EQ(group.block_idx, 1, "consolidated group block");
        EXPECT_EQ(group.expert_idx, static_cast<int>(expert), "consolidated group expert");
        EXPECT_EQ(group.members.size(), 3u, "consolidated group has three roles");
        EXPECT_EQ(group.size, 360u, "different role sizes sum without padding");
        if (group.members.size() == 3) {
            EXPECT_EQ(group.members[0].role_mask, wp::ROLE_UP, "canonical up role first");
            EXPECT_EQ(group.members[1].role_mask, wp::ROLE_GATE, "canonical gate role second");
            EXPECT_EQ(group.members[2].role_mask, wp::ROLE_DOWN, "canonical down role third");
            EXPECT_EQ(group.members[0].size, 80u, "up role size retained");
            EXPECT_EQ(group.members[1].size, 120u, "gate role size retained");
            EXPECT_EQ(group.members[2].size, 160u, "down role size retained");
            EXPECT_EQ(group.members[0].file_idx, 1u, "split shard index retained for up");
            EXPECT_EQ(group.members[1].file_idx, 0u, "split shard index retained for gate");
            EXPECT_EQ(group.members[2].file_idx, 2u, "split shard index retained for down");
            EXPECT_TRUE(group.members[0].source_tensor_name == "blk.1.ffn_up_exps.weight",
                        "sub-expert records source parent tensor name");
        }
    }

    if (groups.size() >= 5) {
        const wp_repack::ExpertGroup & two_role = groups[4];
        EXPECT_EQ(two_role.block_idx, 2, "native group block");
        EXPECT_EQ(two_role.expert_idx, 7, "native group expert");
        EXPECT_EQ(two_role.members.size(), 2u, "two-role group is not padded");
        EXPECT_EQ(two_role.size, 228u, "two-role sizes sum exactly");
        if (two_role.members.size() == 2) {
            EXPECT_EQ(two_role.members[0].role_mask, wp::ROLE_UP, "two-role up first");
            EXPECT_EQ(two_role.members[1].role_mask, wp::ROLE_DOWN, "two-role down second");
            EXPECT_EQ(two_role.members[0].file_idx, 0u, "native up source shard");
            EXPECT_EQ(two_role.members[1].file_idx, 1u, "native down source shard");
        }
    }

    for (const wp_repack::ExpertGroup & group : groups) {
        for (const wp_repack::ExpertMember & member : group.members) {
            EXPECT_TRUE(member.catalog_name.find("gate_inp") == std::string::npos, "router is excluded");
            EXPECT_TRUE(member.catalog_name.find("_shexp") == std::string::npos, "shared expert is excluded");
        }
    }
}

static void test_sharding() {
    wp::PageCatalog catalog;
    catalog.add("blk.0.ffn_up.0.weight", 0, 0, 10);
    catalog.add("blk.0.ffn_down.0.weight", 0, 10, 20);
    catalog.add("blk.0.ffn_up.1.weight", 0, 30, 30);
    catalog.add("blk.0.ffn_down.1.weight", 0, 60, 40);
    catalog.add("blk.1.ffn_up.0.weight", 0, 100, 50);
    catalog.add("blk.1.ffn_down.0.weight", 0, 150, 60);
    catalog.add("blk.2.ffn_up.0.weight", 0, 210, 70);
    catalog.add("blk.2.ffn_down.0.weight", 0, 280, 80);

    const std::vector<wp_repack::ExpertGroup> groups = wp_repack::build_expert_groups(catalog);
    EXPECT_EQ(groups.size(), 4u, "synthetic sharding group count");

    const std::vector<wp_repack::ShardPlan> by_layer = wp_repack::plan_shards_by_layer(groups);
    EXPECT_EQ(by_layer.size(), 3u, "one shard per layer");
    if (by_layer.size() == 3) {
        EXPECT_EQ(by_layer[0].layer_first, 0, "first shard starts at layer zero");
        EXPECT_EQ(by_layer[0].layer_last, 0, "first shard ends at layer zero");
        EXPECT_EQ(by_layer[0].group_indices.size(), 2u, "whole layer keeps both groups");
        EXPECT_EQ(by_layer[0].size, 100u, "whole layer byte size");
        EXPECT_EQ(by_layer[1].group_indices.size(), 1u, "layer one group remains whole");
        EXPECT_EQ(by_layer[2].group_indices.size(), 1u, "layer two group remains whole");
    }

    const std::vector<wp_repack::ShardPlan> coalesced = wp_repack::plan_shards_max_bytes(groups, 220);
    EXPECT_EQ(coalesced.size(), 2u, "adjacent layers coalesce up to limit");
    if (coalesced.size() == 2) {
        EXPECT_EQ(coalesced[0].layer_first, 0, "coalesced shard first layer");
        EXPECT_EQ(coalesced[0].layer_last, 1, "coalesced shard last layer");
        EXPECT_EQ(coalesced[0].group_indices.size(), 3u, "coalescing keeps whole groups");
        EXPECT_EQ(coalesced[0].size, 210u, "coalesced bytes");
        EXPECT_EQ(coalesced[1].layer_first, 2, "second shard starts on layer boundary");
        EXPECT_EQ(coalesced[1].layer_last, 2, "second shard ends on layer boundary");
        EXPECT_EQ(coalesced[1].size, 150u, "second shard bytes");
    }

    const std::vector<wp_repack::ShardPlan> undersized_limit = wp_repack::plan_shards_max_bytes(groups, 90);
    EXPECT_EQ(undersized_limit.size(), 3u, "oversize layers remain whole");
    if (undersized_limit.size() == 3) {
        EXPECT_EQ(undersized_limit[0].group_indices.size(), 2u, "oversize layer does not split groups");
        EXPECT_EQ(undersized_limit[0].layer_first, 0, "oversize shard starts on layer boundary");
        EXPECT_EQ(undersized_limit[0].layer_last, 0, "oversize shard ends on layer boundary");
    }

    const std::vector<wp_repack::LayerRange> ranges          = wp_repack::parse_layer_ranges("0-1,2-2");
    const std::vector<wp_repack::ShardPlan>  explicit_shards = wp_repack::plan_shards_for_ranges(groups, ranges);
    EXPECT_EQ(explicit_shards.size(), 2u, "explicit ranges define two shards");
    if (explicit_shards.size() == 2) {
        EXPECT_EQ(explicit_shards[0].group_indices.size(), 3u, "first explicit range has whole layers");
        EXPECT_EQ(explicit_shards[1].group_indices.size(), 1u, "second explicit range has whole layer");
    }
}

// Explicit --layer-ranges that omit a layer must FAIL rather than silently emitting a
// short shard set. The ranges normally describe a machine split, so a mistyped boundary
// would drop a whole layer while still reporting success, and --verify cannot catch it:
// verify validates what the index claims, not what the model contains.
static void test_range_coverage() {
    wp::PageCatalog catalog;
    catalog.add("blk.0.ffn_up.0.weight", 0, 0, 10);
    catalog.add("blk.0.ffn_down.0.weight", 0, 10, 20);
    catalog.add("blk.1.ffn_up.0.weight", 0, 30, 30);
    catalog.add("blk.1.ffn_down.0.weight", 0, 60, 40);
    catalog.add("blk.2.ffn_up.0.weight", 0, 100, 50);
    catalog.add("blk.2.ffn_down.0.weight", 0, 150, 60);

    const std::vector<wp_repack::ExpertGroup> groups = wp_repack::build_expert_groups(catalog);
    EXPECT_EQ(groups.size(), 3u, "one group per layer");

    // Full coverage: no throw, every group placed.
    const std::vector<wp_repack::LayerRange> full = wp_repack::parse_layer_ranges("0-1,2-2");
    EXPECT_EQ(wp_repack::uncovered_layers(groups, full).size(), 0u, "full coverage reports nothing missing");
    bool threw_on_full = false;
    try {
        const std::vector<wp_repack::ShardPlan> shards = wp_repack::plan_shards_for_ranges(groups, full);
        size_t placed = 0;
        for (const wp_repack::ShardPlan & s : shards) placed += s.group_indices.size();
        EXPECT_EQ(placed, groups.size(), "full coverage places every group");
    } catch (const std::exception &) {
        threw_on_full = true;
    }
    EXPECT_TRUE(!threw_on_full, "full coverage must not throw");

    // Trailing layer omitted (the mistyped-boundary case).
    const std::vector<wp_repack::LayerRange> drops_last = wp_repack::parse_layer_ranges("0-1");
    const std::vector<int> missing_last = wp_repack::uncovered_layers(groups, drops_last);
    EXPECT_EQ(missing_last.size(), 1u, "trailing uncovered layer detected");
    if (missing_last.size() == 1) {
        EXPECT_EQ(missing_last[0], 2, "uncovered layer is layer two");
    }
    bool threw_on_trailing = false;
    try {
        (void) wp_repack::plan_shards_for_ranges(groups, drops_last);
    } catch (const std::invalid_argument &) {
        threw_on_trailing = true;
    }
    EXPECT_TRUE(threw_on_trailing, "omitting a trailing layer must throw");

    // Leading layer omitted -- the skip loop silently advanced past these before the fix.
    const std::vector<wp_repack::LayerRange> drops_first = wp_repack::parse_layer_ranges("1-2");
    const std::vector<int> missing_first = wp_repack::uncovered_layers(groups, drops_first);
    EXPECT_EQ(missing_first.size(), 1u, "leading uncovered layer detected");
    if (missing_first.size() == 1) {
        EXPECT_EQ(missing_first[0], 0, "uncovered layer is layer zero");
    }
    bool threw_on_leading = false;
    try {
        (void) wp_repack::plan_shards_for_ranges(groups, drops_first);
    } catch (const std::invalid_argument &) {
        threw_on_leading = true;
    }
    EXPECT_TRUE(threw_on_leading, "omitting a leading layer must throw");

    // A middle gap must also be caught, not just the ends.
    const std::vector<wp_repack::LayerRange> gap = wp_repack::parse_layer_ranges("0-0,2-2");
    const std::vector<int> missing_gap = wp_repack::uncovered_layers(groups, gap);
    EXPECT_EQ(missing_gap.size(), 1u, "interior uncovered layer detected");
    if (missing_gap.size() == 1) {
        EXPECT_EQ(missing_gap[0], 1, "uncovered layer is layer one");
    }

    // Deliberate subsets stay possible via allow_partial, and place only covered groups.
    bool threw_on_allowed = false;
    try {
        const std::vector<wp_repack::ShardPlan> shards =
            wp_repack::plan_shards_for_ranges(groups, drops_last, /*allow_partial=*/true);
        size_t placed = 0;
        for (const wp_repack::ShardPlan & s : shards) placed += s.group_indices.size();
        EXPECT_EQ(placed, 2u, "allow_partial places only the covered groups");
    } catch (const std::exception &) {
        threw_on_allowed = true;
    }
    EXPECT_TRUE(!threw_on_allowed, "allow_partial must not throw");
}

int main() {
    test_grouping();
    test_sharding();
    test_range_coverage();

    if (failures != 0) {
        std::fprintf(stderr, "%d wp-repack test(s) failed\n", failures);
        return 1;
    }
    std::puts("all wp-repack tests passed");
    return 0;
}
