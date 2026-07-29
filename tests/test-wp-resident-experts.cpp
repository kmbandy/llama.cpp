// Unit tests for the resident-expert planner (wp::ResidentExpertPlan).
//
// The planner decides which transformer blocks keep their routed experts
// RESIDENT on the FFN-island device instead of paging them. Three call sites
// consult the resulting plan -- the buft override, is_paged_weight() and the
// pager-catalog filter -- and any disagreement between them produces silently
// wrong weights rather than a crash. So the predicate itself is tested hard.

#include "weight-pager/wp-router.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

// A block of `n_experts` experts across three role tensors of equal size, which
// is the common shape. max_page_bytes is one (role, expert) sub-page.
static wp::LayerExpertBytes block(int idx, uint64_t bytes, uint64_t max_page_bytes) {
    wp::LayerExpertBytes b;
    b.block_idx      = idx;
    b.bytes          = bytes;
    b.max_page_bytes = max_page_bytes;
    return b;
}

static std::vector<wp::LayerExpertBytes> uniform_layers(int n, uint64_t bytes) {
    std::vector<wp::LayerExpertBytes> out;
    for (int i = 0; i < n; ++i) {
        out.push_back(block(i, bytes, bytes / 3));
    }
    return out;
}

static void test_name_predicates() {
    CHECK(wp::is_routed_expert_name("blk.3.ffn_up_exps.weight"));
    CHECK(wp::is_routed_expert_name("blk.3.ffn_gate_exps.weight"));
    CHECK(wp::is_routed_expert_name("blk.3.ffn_down_exps.weight"));

    // Shared experts and the router are NOT routed experts -- they are placed
    // by separate rules and must never be swept into a resident-expert plan.
    CHECK(!wp::is_routed_expert_name("blk.3.ffn_up_shexp.weight"));
    CHECK(!wp::is_routed_expert_name("blk.3.ffn_gate_inp.weight"));
    CHECK(!wp::is_routed_expert_name("blk.3.ffn_norm.weight"));
    CHECK(!wp::is_routed_expert_name("blk.3.attn_q.weight"));
    CHECK(!wp::is_routed_expert_name("token_embd.weight"));
    CHECK(!wp::is_routed_expert_name(nullptr));

    int b = -1;
    CHECK(wp::parse_block_index("blk.0.ffn_up_exps.weight", b) && b == 0);
    CHECK(wp::parse_block_index("blk.47.ffn_up_exps.weight", b) && b == 47);
    CHECK(wp::parse_block_index("blk.123.attn_q.weight", b) && b == 123);
    CHECK(!wp::parse_block_index("token_embd.weight", b));
    CHECK(!wp::parse_block_index("blk..ffn_up_exps.weight", b));
    CHECK(!wp::parse_block_index("blkx.3.ffn_up_exps.weight", b));
    CHECK(!wp::parse_block_index(nullptr, b));
}

static void test_empty_plan_is_inert() {
    wp::ResidentExpertPlan plan;
    CHECK(plan.empty());
    CHECK(plan.bytes() == 0);
    CHECK(plan.pattern().empty());
    CHECK(plan.describe() == "none");
    // An empty plan must claim nothing, or the default configuration changes.
    CHECK(!plan.covers_tensor("blk.0.ffn_up_exps.weight"));
    CHECK(!plan.covers_block(0));
}

static void test_pattern_is_index_anchored() {
    // The regression this guards: an unanchored alternation lets block 1 match
    // "blk.17.", quietly making unplanned blocks resident and unallocated.
    std::vector<wp::LayerExpertBytes> layers = { block(1, 10, 3), block(17, 10, 3) };
    auto plan = wp::ResidentExpertPlan::from_blocks(layers, { 1 });

    CHECK(plan.covers_tensor("blk.1.ffn_up_exps.weight"));
    CHECK(!plan.covers_tensor("blk.17.ffn_up_exps.weight"));
    CHECK(!plan.covers_tensor("blk.11.ffn_up_exps.weight"));

    // The emitted regex must agree with covers_tensor(); it is what the
    // allocator actually uses.
    CHECK(plan.pattern() == "^blk\\.(1)\\.ffn_(up|gate|down)_exps\\.");
}

static void test_covers_tensor_scope() {
    std::vector<wp::LayerExpertBytes> layers = { block(4, 10, 3), block(5, 10, 3) };
    auto plan = wp::ResidentExpertPlan::from_blocks(layers, { 4 });

    // All three roles of a covered block, and nothing else in that block.
    CHECK(plan.covers_tensor("blk.4.ffn_up_exps.weight"));
    CHECK(plan.covers_tensor("blk.4.ffn_gate_exps.weight"));
    CHECK(plan.covers_tensor("blk.4.ffn_down_exps.weight"));
    CHECK(!plan.covers_tensor("blk.4.ffn_up_shexp.weight"));
    CHECK(!plan.covers_tensor("blk.4.ffn_gate_inp.weight"));
    CHECK(!plan.covers_tensor("blk.4.attn_q.weight"));
    CHECK(!plan.covers_tensor("blk.5.ffn_up_exps.weight"));
}

static void test_from_blocks_ignores_unknown() {
    std::vector<wp::LayerExpertBytes> layers = { block(0, 10, 3), block(1, 10, 3) };
    // Block 9 has no routed experts (e.g. a dense block). from_blocks drops it;
    // the caller is responsible for treating the size mismatch as a typo.
    auto plan = wp::ResidentExpertPlan::from_blocks(layers, { 0, 9 });
    CHECK(plan.layers().size() == 1);
    CHECK(plan.covers_block(0));
    CHECK(!plan.covers_block(9));

    // Duplicates collapse and must not double-count bytes.
    auto dup = wp::ResidentExpertPlan::from_blocks(layers, { 1, 1, 1 });
    CHECK(dup.layers().size() == 1);
    CHECK(dup.bytes() == 10);
}

static void test_describe_collapses_ranges() {
    std::vector<wp::LayerExpertBytes> layers = uniform_layers(30, 10);
    auto plan = wp::ResidentExpertPlan::from_blocks(layers, { 0, 1, 2, 3, 7, 20, 21 });
    CHECK(plan.describe() == "0-3,7,20-21");
}

static void test_parse_request() {
    // Off. NOTE "0" is NOT here: a bare number is a block index, and letting
    // "0" mean off silently disabled the feature for block 0 while looking configured.
    for (const char * off : { "", "off", "none" }) {
        auto r = wp::parse_resident_expert_request(off);
        CHECK(!r.enabled);
    }
    {
        auto z = wp::parse_resident_expert_request("0");
        CHECK(z.enabled);
        CHECK(z.blocks.size() == 1 && z.blocks[0] == 0);
    }
    // Trailing garbage in a range item must throw, not be silently reinterpreted.
    for (const char * bad : { "5x", "1-2-3", "3-4y" }) {
        bool threw = false;
        try { wp::parse_resident_expert_request(bad); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw);
    }
    CHECK(!wp::parse_resident_expert_request(nullptr).enabled);

    // Auto -> budget from free VRAM, decided by the caller.

    // "auto" and byte sizes are REJECTED, not quietly reinterpreted.
    for (const char * gone : { "auto", "12GiB", "8G", "512MB" }) {
        bool threw = false;
        try { wp::parse_resident_expert_request(gone); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw);
    }

    // Block lists.
    auto b = wp::parse_resident_expert_request("0-6,20-22");
    CHECK(b.enabled);
    CHECK(b.blocks.size() == 10);
    CHECK(b.blocks.front() == 0 && b.blocks.back() == 22);

    auto single = wp::parse_resident_expert_request("5");
    CHECK(single.enabled);
    CHECK(single.blocks.size() == 1 && single.blocks[0] == 5);

    // A bare number is a BLOCK, never a byte count -- "5" meaning five bytes
    // would silently disable the feature while looking configured.

    // Malformed input throws rather than silently doing nothing.
    bool threw = false;
    try { wp::parse_resident_expert_request("6-2"); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);

    threw = false;
    try { wp::parse_resident_expert_request("3,,4"); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);

    threw = false;
    try { wp::parse_resident_expert_request("12ZB"); } catch (const std::invalid_argument &) { threw = true; }
    CHECK(threw);
}

int main() {
    test_name_predicates();
    test_empty_plan_is_inert();
    test_pattern_is_index_anchored();
    test_covers_tensor_scope();
    test_from_blocks_ignores_unknown();
    test_describe_collapses_ranges();
    test_parse_request();

    if (g_failed != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test-wp-resident-experts: all checks passed\n");
    return 0;
}
