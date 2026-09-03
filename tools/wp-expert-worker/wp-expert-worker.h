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
    // Role type bytes that the arena stride must represent exactly.
    std::vector<uint64_t> role_type_sizes;
};

struct SlotClass {
    uint64_t size      = 0;
    // Allocated bytes per slot, including backend and role-type alignment.
    uint64_t stride    = 0;
    int      slots     = 0;
    int      pin_floor = 0;
    int      pages     = 0;
    // Slots of this class the pool ACTUALLY carved, i.e. `slots` minus the pad
    // tail of every arena of the class (and minus anything an exhausted arena
    // run could not carve). 0 until ExpertSlotPool's constructor fills it in;
    // plan_resources() leaves it 0 because pads are only known once the arenas
    // exist. This -- not `slots` -- is how many pages of this class a device
    // can hold resident, so it is what the ownership policy budgets against.
    int      usable_slots = 0;
    // PAD slots actually reserved per arena of THIS class, filled in by
    // ExpertSlotPool's constructor after allocate_slot_arenas() runs (0 until
    // then). May differ class-to-class from ResourcePlan::pad_slots_per_arena
    // in principle, though in practice every class gets the full reservation
    // -- see the class_pad_plan() note on wp-expert-worker.cpp.
    int      pad_slots = 0;
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
    // PAD SLOTS. Every arena reserves this many slots at its END. They exist
    // in the arena buffer at the normal stride -- so a strided mul_mat_id view
    // over an arena can address them -- are zero-filled once at allocation and
    // are never bound to a page, never handed out, never evicted and never a
    // DMA target. The grouped prefill path uses them as the weight-0 filler
    // ids that make every arena group carry the SAME canonical route width;
    // the CUDA/HIP scatter-quantize path requires the ids of one token row to
    // be distinct, so the filler must be a real, distinct slot of that arena.
    // Set from the model's n_expert_used by DeviceWorker. Taken OUT of each
    // arena's usable slot count, so the slot budget is unchanged.
    int                    pad_slots_per_arena  = 0;
    // Slots the plan asked for, BEFORE pad reservation. slot_count is what the
    // pool actually carved (and what HELLO reports).
    int                    planned_slot_count   = 0;
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

// Test-only observability for the grouped arena prefill path
// (WP_EXPERT_ARENA_PREFILL=1). Cumulative across every device tier of every
// worker in this process; the worker runs in a thread inside the CPU
// integration test, so a counter is the only way a test can tell a grouped
// hit from a silent fall-back to the per-expert gather path. Both counters
// are process-wide atomics; reset before a run and read after it.
void     test_reset_arena_prefill_counters();
uint64_t test_arena_prefill_hits();
uint64_t test_arena_prefill_fallbacks();
// Grouped-prefill graph BUILDS (cache misses). Rises once per distinct
// (n_tokens, route width, group shape) bucket; a stable cache stops it rising.
uint64_t test_arena_prefill_builds();
// Fingerprint of the pool slots the last grouped prefill's assignments landed
// in. Only meaningful as "same" vs "different" between two runs.
uint64_t test_arena_prefill_placement();
// Pad-slot bookkeeping of the most recently constructed ExpertSlotPool:
// slots reserved per arena, arenas allocated, planned slots before the
// reservation, and the slots actually carved (== HELLO n_slots).
uint64_t test_pool_pad_slots_per_arena();
uint64_t test_pool_arena_count();
uint64_t test_pool_planned_slots();
uint64_t test_pool_usable_slots();
// Times the grouped prefill saw a bound page whose arena-local slot index fell
// inside its arena's PAD region. Must stay 0: pads have no pool slot index.
uint64_t test_arena_prefill_pad_bound();

// Decode/prefill compute-profile policy. Pure functions of the env string so
// tests can pin defaults without latching process-lifetime getenv statics.
//
// Gather is a prefill optimisation: at n_tokens==1 density is 100% and
// get_rows + scatter add nodes for nothing. Default min tokens is 2.
int  parse_gather_min_tokens(const char * env);
bool parse_env_default_on(const char * env);
bool parse_env_default_off(const char * env);
bool use_expert_gather(uint32_t n_tokens, bool force_dense, int min_tokens, bool gather_enabled);

// WP_EXPERT_MM_PIN policy. One flag has to serve two phases with opposite
// needs: the GGML_HINT_MUL_MAT_PIN hint forces MMQ on CUDA/HIP (and the
// mat-vec path on Vulkan) regardless of shape, which MEASURED 2026-09-02 costs
// ~16% of per-128-token-chunk kernel wall in PREFILL but buys ~3-4 tok/s in
// DECODE. So the value is a three-way policy, not a boolean:
//   unset / "" / "0"  -> off
//   "decode"          -> pin ONLY requests with n_tokens <= max_tokens
//                        (WP_EXPERT_MM_PIN_MAX_TOKENS, default 8: decode is
//                        n_tokens==1, spec-verify blocks are <= 8, and stream4
//                        prefill chunks are 128 with 74-token tails, so the
//                        wide chunks stay unpinned)
//   anything else     -> on, the legacy wide-request pin
// The legacy "on" mode pins only gather-path requests wider than a verify
// block (n_tokens > 8) and at least WP_EXPERT_MM_PIN_MIN_TOKENS. "decode" must
// NOT require gather: gather is bypassed at n_tokens==1 (see
// WP_EXPERT_GATHER_MIN_TOKENS above), and the dense path shares the same
// mul_mat lambda, so requiring it would make the decode pin a no-op.
enum class mm_pin_mode { off, on, decode };
mm_pin_mode parse_mm_pin_mode(const char * env);
int  parse_mm_pin_min_tokens(const char * env);   // default 9
int  parse_mm_pin_max_tokens(const char * env);   // default 8
bool use_mm_pin(uint32_t n_tokens, bool use_gather, mm_pin_mode mode,
                int min_tokens, int max_tokens);

// WP_EXPERT_ARENA_PREFILL policy, per device: unset/"" and "0" are off, "1"
// is on for every device, a comma-separated device-name list ("ROCm0,CUDA0")
// is on only for those devices (exact match), and the same list prefixed
// with "!" ("!Vulkan0") is on for every device except those named. Whitespace
// around the whole value and around each comma-separated name is ignored.
bool parse_arena_prefill_enabled(const char * env, const std::string & device_name);

// ---------------------------------------------------------------------------
// WP_EXPERT_OWNER_POLICY -- which device OWNS (and therefore pages in) an
// expert page on a multi-device worker.
//
// Whatever the policy, the map MUST be static for the life of the process and
// a pure function of (pin file, device list, slot counts). Measured 2026-09-02:
// gfx1030 and gfx1201 are each self-deterministic but never bit-identical to
// each other, so a page that moves from one to the other between runs changes
// the output md5. No timing, no hash order, no live counters may reach it.
enum class owner_policy {
    // Today's behaviour, and the default: the expert-id range is cut into
    // contiguous bands proportional to each device's slot count, in
    // device-list order. Every device then takes the same miss rate per owned
    // expert -- which is why the slow-link device (ROCm1 behind a 2.78 GB/s
    // Thunderbolt x4 link, measured 2026-09-02) paged 61.7 GB over one run and
    // became the rig's straggler.
    proportional,
    // Hot-set first. Pages are ranked by their WP_EXPERT_PIN_FILE warm-start
    // counts and packed, hottest first, into the highest-priority device up to
    // its USABLE slot capacity in that page's size class, then the next
    // device, and so on. The top-priority device therefore owns exactly a hot
    // set it can hold FULLY RESIDENT, and stops paging over its slow link.
    hot,
};

// unset / "" / "proportional" -> proportional. "hot" -> hot. Anything else
// warns on stderr and falls back to proportional. Leading/trailing whitespace
// is ignored.
owner_policy parse_owner_policy(const char * env);

// WP_EXPERT_OWNER_PRIORITY: comma-separated device names, HIGHEST priority
// first. Returns indices into `device_names`. Names that are not in
// `device_names` are ignored (with a warning); devices the list does not name
// are appended in device-list order. The result is always a permutation of
// [0, device_names.size()), so an unset/empty env yields the device-list order
// itself -- the documented default.
std::vector<size_t> parse_owner_priority(
        const char * env, const std::vector<std::string> & device_names);

// The proportional split, factored out of the worker so a test can pin it.
// Byte-for-byte the pre-WP_EXPERT_OWNER_POLICY behaviour.
size_t proportional_owner_for_expert(
        int expert, int expert_first, int expert_last,
        const std::vector<int> & device_slots);

// Sentinel page class for a page the placement policy does not manage (a
// resident-layer page). Such pages always keep their proportional owner.
constexpr size_t HOT_OWNER_NO_CLASS = (size_t) -1;

struct HotOwnerInput {
    size_t n_devices = 0;
    // Per page id. page_class[i] is the placement size class of page i, or
    // HOT_OWNER_NO_CLASS.
    std::vector<size_t> page_class;
    // Per page id: the proportional owner, used as the last-resort fallback.
    std::vector<size_t> page_static_owner;
    // capacity[class][device] -- USABLE owner slots (SlotClass::usable_slots,
    // so pad slots are already excluded).
    std::vector<std::vector<size_t>> capacity;
    // Page ids, hottest first. Duplicates and out-of-range ids are skipped.
    std::vector<size_t> ranked;
    // Device indices, highest priority first.
    std::vector<size_t> priority;
};

struct HotOwnerPlan {
    std::vector<size_t> owner;        // per page id
    std::vector<char>   from_ranked;  // per page id: came off `ranked`, not the fallback
};

// Pure core of WP_EXPERT_OWNER_POLICY=hot. Deterministic: it reads only its
// argument, and every loop is over an index range or an explicitly sorted
// vector.
//
//  1. Walk `ranked` in order. Assign each page to the FIRST device in
//     `priority` order that still has free capacity in that page's class,
//     decrementing that capacity. A page whose class has no free capacity on
//     any device is left for the fallback (so is every page `ranked` never
//     mentions).
//  2. FALLBACK, per size class, over the still-unassigned pages of that class
//     sorted by ascending page id (K of them): spread them proportionally over
//     the devices that still have capacity left in the class, weighted by that
//     REMAINING capacity, using the same integer-band idiom as the
//     proportional policy (page k of K goes to the device whose cumulative
//     weight band contains k*W/K). The spread does not consume capacity: K is
//     normally far larger than W, and every one of these pages is a cold page
//     that will be paged in on demand wherever it lands.
//     If no device has remaining capacity in the class, the weights fall back
//     to the class's TOTAL capacity, so pages still avoid a device that cannot
//     physically hold their size class. If that is empty too, the page keeps
//     its proportional owner.
//  3. HOT_OWNER_NO_CLASS pages keep their proportional owner.
HotOwnerPlan plan_hot_owner_map(const HotOwnerInput & in);

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
