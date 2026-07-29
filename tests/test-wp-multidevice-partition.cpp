#include "weight-pager/wp-partition.h"
#include "weight-pager/wp-router.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static int g_failed = 0;

#define CHECK(cond) do {                                                       \
    if (!(cond)) {                                                             \
        std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
        ++g_failed;                                                            \
    }                                                                          \
} while (0)

static void test_single_device_is_identity() {
    int dev0 = 0;
    std::vector<wp::PagePartitionInput> pages = {
        {"blk.0.ffn_up_exps.weight", &dev0, true},
        {"blk.0.ffn_up_exps.weight#expert.0", &dev0, true},
        {"blk.1.ffn_up_exps.weight", &dev0, true},
    };
    const auto result = wp::partition_pages_by_device(pages);
    CHECK(result.partitions.size() == 1);
    CHECK(result.partitions[0].source_indices ==
          std::vector<size_t>({0, 1, 2}));
    CHECK(result.n_paged == pages.size());
    for (size_t i = 0; i < pages.size(); ++i) {
        const auto route = result.routes.at(pages[i].name);
        CHECK(route.partition_idx == 0);
        CHECK(route.page_idx == (int) i);
    }
}

static void test_partition_and_routing() {
    int dev0 = 0;
    int dev1 = 1;
    std::vector<wp::PagePartitionInput> pages = {
        {"blk.0.ffn_up_exps.weight", &dev0, true},
        {"blk.0.ffn_up_exps.weight#expert.0", &dev0, true},
        {"blk.40.ffn_up_exps.weight", &dev1, true},
        {"blk.40.ffn_up_exps.weight#expert.0", &dev1, true},
        {"blk.1.ffn_up_exps.weight", &dev0, true},
    };
    const auto result = wp::partition_pages_by_device(pages);
    CHECK(result.partitions.size() == 2);
    CHECK(result.partitions[0].device == &dev0);
    CHECK(result.partitions[1].device == &dev1);
    CHECK(result.partitions[0].source_indices ==
          std::vector<size_t>({0, 1, 4}));
    CHECK(result.partitions[1].source_indices ==
          std::vector<size_t>({2, 3}));
    CHECK(result.routes.at(pages[4].name).partition_idx == 0);
    CHECK(result.routes.at(pages[4].name).page_idx == 2);
    CHECK(result.routes.at(pages[3].name).partition_idx == 1);
    CHECK(result.routes.at(pages[3].name).page_idx == 1);
    CHECK(result.n_paged == pages.size());
}

static void test_resident_pages_are_excluded() {
    int dev0 = 0;
    int dev1 = 1;
    std::vector<wp::PagePartitionInput> pages = {
        {"blk.0.ffn_up_exps.weight", &dev0, true},
        {"blk.1.ffn_up_exps.weight", &dev1, false},
        {"blk.2.ffn_up_exps.weight", &dev1, true},
    };
    const auto result = wp::partition_pages_by_device(pages);
    CHECK(result.n_paged == 2);
    CHECK(result.routes.count(pages[1].name) == 0);
    size_t assigned = 0;
    for (const auto & partition : result.partitions) {
        assigned += partition.source_indices.size();
    }
    CHECK(assigned == result.n_paged);
}

static void test_invalid_partition_input() {
    bool threw = false;
    try {
        (void) wp::partition_pages_by_device({{"page", nullptr, true}});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);

    int dev0 = 0;
    threw = false;
    try {
        (void) wp::partition_pages_by_device({
            {"page", &dev0, true}, {"page", &dev0, true}});
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

static void test_explicit_device_layer_parser() {
    const auto request =
        wp::parse_device_layer_request("ROCm0:0-37;ROCm1:38-74");
    CHECK(request.size() == 2);
    CHECK(request[0].device == "ROCm0");
    CHECK(request[0].blocks.front() == 0);
    CHECK(request[0].blocks.back() == 37);
    CHECK(request[1].device == "ROCm1");
    CHECK(request[1].blocks.front() == 38);
    CHECK(request[1].blocks.back() == 74);

    bool threw = false;
    try {
        (void) wp::parse_device_layer_request("auto");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);

    threw = false;
    try {
        (void) wp::parse_device_layer_request("ROCm0:0-5;ROCm1:");
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

static void test_device_layer_plan_routing() {
    auto buft0 = reinterpret_cast<ggml_backend_buffer_type_t>(0x1000);
    auto buft1 = reinterpret_cast<ggml_backend_buffer_type_t>(0x2000);
    wp::DeviceLayerPlan plan;
    plan.add("ROCm1", {38, 39, 40}, buft1);

    CHECK(plan.buft_for_tensor(
        "blk.39.ffn_up_exps.weight", buft0) == buft1);
    CHECK(plan.buft_for_tensor(
        "blk.37.ffn_up_exps.weight", buft0) == buft0);
    CHECK(plan.buft_for_tensor(
        "blk.39.ffn_up_shexp.weight", buft0) == buft0);

    const auto overrides = wp::build_router_overrides(
        buft0, buft1, nullptr, nullptr, true, nullptr, nullptr, &plan);
    CHECK(std::string(overrides[0].pattern) ==
          "blk\\.(38|39|40)\\.ffn_(up|gate|down)_exps\\.");
    CHECK(overrides[0].buft == buft1);
    CHECK(std::string(overrides[1].pattern) == wp::ROUTER_EXPERT_PATTERN);
    CHECK(overrides[1].buft == buft0);

    bool threw = false;
    try {
        plan.add("ROCm0", {40, 41}, buft0);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    CHECK(threw);
}

int main() {
    test_single_device_is_identity();
    test_partition_and_routing();
    test_resident_pages_are_excluded();
    test_invalid_partition_input();
    test_explicit_device_layer_parser();
    test_device_layer_plan_routing();

    if (g_failed != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failed);
        return 1;
    }
    std::printf("test-wp-multidevice-partition: all checks passed\n");
    return 0;
}
