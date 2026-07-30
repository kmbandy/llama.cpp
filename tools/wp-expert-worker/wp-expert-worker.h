#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace wp_expert_worker {

struct ResourcePage {
    int      layer = -1;
    uint64_t size  = 0;
};

struct SlotClass {
    uint64_t size      = 0;
    int      slots     = 0;
    int      pin_floor = 0;
    int      pages     = 0;
};

struct ResourcePlan {
    int                    requested_slots       = 0;
    int                    slot_count            = 0;
    uint64_t               device_budget_bytes  = 0;
    uint64_t               device_bytes         = 0;
    uint64_t               host_budget_bytes    = 0;
    uint64_t               staging_buffer_bytes = 0;
    uint64_t               staging_bytes        = 0;
    int                    staging_buffers      = 0;
    bool                   size_classes         = false;
    std::vector<SlotClass> slot_classes;
};

struct TestHooks {
    std::function<void(int, int)>      read_started;
    std::function<void(int, int)>      read_finished;
    std::function<void()>              staging_borrowed;
    std::function<void(int, int, int)> slot_reserved;
};

struct Options {
    std::filesystem::path shard_manifest;
    std::filesystem::path descriptor;
    std::string           device;
    std::string           listen_host;
    int                   listen_port = 0;
    int                   slots       = 0;
    uint64_t              host_budget_bytes = 0;
    TestHooks *           test_hooks = nullptr;
    bool                  once        = false;
};

// Derive the device size classes and bounded staging arena from page metadata.
// requested_slots sets a device budget in max-page equivalents; slot_count is
// the actual number carved. A zero host budget selects up to 16 staging buffers.
ResourcePlan plan_resources(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes = 0);

// Construct the same backend and resource pools used by run(), then report
// their own allocation accounting. Intended for diagnostics and CPU tests.
ResourcePlan inspect_resources(const Options & options);

// Serve expert dispatch connections. With once=true, accept one connection
// and return after it closes; this is used by the CPU integration test.
int run(const Options & options);

} // namespace wp_expert_worker
