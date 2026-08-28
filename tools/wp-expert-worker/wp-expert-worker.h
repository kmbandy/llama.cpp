#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace wp_expert_worker {

struct ResourcePage {
    int      layer = -1;
    uint64_t size  = 0;
    bool     pinned = false;
    uint64_t staging_size = 0;
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
    uint64_t               slot_budget_bytes    = 0;
    uint64_t               reserved_bytes      = 0;
    uint64_t               requested_reserved_bytes = 0;
    uint64_t               named_reservable_bytes = 0;
    int                    reserved_slot_count = 0;
    int                    general_slot_count  = 0;
    std::vector<int>       reserved_slot_indices;
    uint64_t               pinned_bytes         = 0;
    uint64_t               device_bytes         = 0;
    uint64_t               host_budget_bytes    = 0;
    uint64_t               staging_buffer_bytes = 0;
    uint64_t               staging_bytes        = 0;
    int                    staging_buffers      = 0;
    bool                   size_classes         = false;
    std::vector<SlotClass> slot_classes;
};

struct DeviceMemberLayout {
    uint64_t offset = 0;
    uint64_t size   = 0;
};

// Place independently allocated tensor members in one slot. `size` is the
// buffer type allocation size, including any backend-specific tensor padding.
std::vector<DeviceMemberLayout> plan_device_member_layout(
        const std::vector<uint64_t> & sizes, uint64_t alignment);

struct TestHooks {
    std::function<void(int, int)>      read_started;
    std::function<void(int, int)>      read_finished;
    std::function<void()>              staging_borrowed;
    std::function<void(int, int, int)> slot_reserved;
    // Fires once per ExpertSlotPool::stripe_plan() call with
    // (page_size, n_pageins, n_stripes_chosen). read_started/read_finished
    // fire once per PAGE regardless of stripe count, so this is the only way
    // a test can observe whether a given page size actually got split -- in
    // particular, whether the sliced-rig min-part fix (WP_EXPERT_STRIPE_MIN_PART)
    // restores striping for small width-slice pages that the old 1 MiB floor
    // silently collapsed to a single whole-page read.
    std::function<void(uint64_t page_size, size_t n_pageins, size_t n_stripes)> stripe_planned;
};

struct Options {
    std::filesystem::path shard_manifest;
    std::filesystem::path descriptor;
    std::string           device;
    std::vector<std::string> devices;
    std::string           listen_host;
    int                   listen_port = 0;
    int                   slots       = 0;
    std::vector<int>      device_slots;
    uint64_t              host_budget_bytes = 0;
    uint64_t              host_victim_bytes = 0;
    std::vector<int>      resident_expert_blocks;
    bool                  resident_expert_blocks_set = false;
    std::vector<int>      expert_reserve_blocks;
    bool                  expert_reserve_blocks_set = false;
    uint64_t              expert_reserve_bytes = 0;
    TestHooks *           test_hooks = nullptr;
    bool                  once        = false;
};

// Derive the device size classes and bounded staging arena from page metadata.
// requested_slots sets the total device budget in max-page equivalents;
// pinned_bytes is reserved first, and slot_count is the actual pageable pool
// count. A zero host budget selects up to 16 staging buffers.
ResourcePlan plan_resources(
        const std::vector<ResourcePage> & pages,
        int requested_slots,
        uint64_t host_budget_bytes = 0,
        uint64_t pinned_bytes = 0,
        const std::vector<int> & reserve_blocks = {},
        uint64_t reserve_bytes = 0);

// Construct the same backend and resource pools used by run(), then report
// their own allocation accounting. Intended for diagnostics and CPU tests.
ResourcePlan inspect_resources(const Options & options);

// Serve expert dispatch connections. With once=true, accept one connection
// and return after it closes; this is used by the CPU integration test.
int run(const Options & options);

// Register an in-process factory with the expert dispatcher. When
// WP_TRUNK_INPROC=1 and WP_TRUNK_INPROC_SHARDS lists a port, that worker is
// constructed inside the spine process instead of over TCP. llama-server
// must call this before llama_init so the dispatcher constructor sees it.
// No-op if the env flag is off; TCP workers are unchanged.
void install_inproc_factory();

// Decode/prefill compute-profile policy. Pure functions of the env string so
// tests can pin defaults without latching process-lifetime getenv statics.
//
// Gather is a prefill optimisation: at n_tokens==1 density is 100% and
// get_rows + scatter add nodes for nothing. Default min tokens is 2.
int  parse_gather_min_tokens(const char * env);
bool parse_env_default_on(const char * env);
bool parse_env_default_off(const char * env);
bool use_expert_gather(uint32_t n_tokens, bool force_dense, int min_tokens, bool gather_enabled);

// Compact a router-weight row to the tokens that actually route here.
// Empty (all-zero) rows keep a single dummy index 0 / weight 0 so the
// gather graph still has one row — same contract as compute_batch.
struct CompactRouting {
    std::vector<int32_t> idx;
    std::vector<float>   weights;
};
CompactRouting compact_routing_rows(const std::vector<float> & wv);

// Scatter compacted [n_embd, n_sel] rows onto a zeroed [n_embd, n_tokens]
// dest. idx is I32 [n_sel] and MUST be unique (ggml_set_rows overwrites;
// colliding dest rows are undefined). Rows not named in idx stay 0 — the
// same as ggml_get_rows_back when idx has no repeats, without the
// O(ncols * n_tokens * n_sel) dest scan.
ggml_tensor * scatter_compact_rows(
        struct ggml_context * ctx,
        struct ggml_tensor * compact,
        struct ggml_tensor * idx,
        struct ggml_tensor * full_shape);

// dest is [n_embd, n_tokens] already allocated (the io result). Adds compact
// into dest[idx] via get_rows + add + set_rows. No full-ubatch zero tensor.
// idx unique per call; the same dest row may be hit by later experts.
ggml_tensor * scatter_add_compact_rows(
        struct ggml_context * ctx,
        struct ggml_tensor * dest,
        struct ggml_tensor * compact,
        struct ggml_tensor * idx);

} // namespace wp_expert_worker
