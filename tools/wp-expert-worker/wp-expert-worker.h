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

// Why WP_EXPERT_FUSE_GATE_UP did not fire for one request. Checked in this
// order: clamp, type, shape, adjacency, gather.
enum class FuseGateUpReason {
    Ok = 0,
    Clamp,
    Gather,
    Type,
    Shape,
    Adjacency,
};

struct FuseGateUpCheck {
    float    swiglu_clamp = 0.0f;
    bool     use_gather = false;
    bool     gather_allowed = false;
    int      gate_type = 0;
    int      up_type = 0;
    int64_t  gate_ne0 = 0;
    int64_t  gate_ne1 = 0;
    int64_t  up_ne0 = 0;
    int64_t  up_ne1 = 0;
    uint64_t gate_device_offset = 0;
    uint64_t up_device_offset = 0;
};

struct FuseGateUpDiag {
    FuseGateUpReason reason = FuseGateUpReason::Ok;
    uint64_t         go = 0;
    uint64_t         uo = 0;
    uint64_t         gate_bytes = 0;
};

FuseGateUpDiag classify_fuse_gate_up(const FuseGateUpCheck & check);
const char *   fuse_gate_up_reason_name(FuseGateUpReason reason);
std::string    format_fuse_gate_up_reason(const FuseGateUpDiag & diag);
// "gate" then "up" then the other names in `names` order. Unchanged if either
// role is missing. Used by WP_EXPERT_FUSE_GATE_UP_LAYOUT=1.
std::vector<std::string> fuse_gate_up_layout_names(
        const std::vector<std::string> & names);

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

// Test-only snapshot of the multi-device placement/ownership tables the last
// Worker in this process built. Written once, from the Worker constructor, by
// initialize_placement_policy() (the capacity half) and, under
// WP_EXPERT_OWNER_POLICY=hot, by apply_hot_owner_policy() (the ownership
// half). Empty on a single-device worker, which builds no tables at all.
//
// These are what the startup "owner_policy=hot ... class[c]bytes=... owned=...
// usable=..." lines print, so a test can assert on the same numbers an
// operator reads out of the log without standing up a socket.
struct PlacementReport {
    std::vector<std::string> devices;
    // One entry per placement size class, in the same order as the log's
    // class[c] index. `class_bytes` is the catalog page size of the class.
    std::vector<uint64_t> class_bytes;
    // [class][device]. `planned` is SlotClass::slots, `usable` is what the
    // pool actually carved (SlotClass::usable_slots, pads removed) -- the
    // number WP_EXPERT_OWNER_POLICY=hot budgets ownership against.
    std::vector<std::vector<uint64_t>> planned;
    std::vector<std::vector<uint64_t>> usable;
    // [class][device] pages OWNED after the policy ran, and per device whether
    // every class fits (owned <= usable). Only filled under `hot`.
    std::vector<std::vector<uint64_t>> owned;
    std::vector<char>                  fully_resident;
    // Device indices, highest priority first. Only filled under `hot`.
    std::vector<size_t> priority;
    bool hot = false;
};
void                    test_reset_placement_report();
const PlacementReport & test_placement_report();

// Test-only snapshot of the multi-device WP_EXPERT_PIN_FILE / PIN_CLASS_PCT
// cap accounting the last Worker::load_pin_file() run built -- the per-class
// {bytes, slots, cap, pinned, skipped} table the startup "pin_class device=...
// bytes=... cap=... pinned=... skipped=..." lines print. [device][class],
// same class order as each device's own ResourcePlan::slot_classes (NOT
// necessarily the same order or count as PlacementReport::class_bytes, which
// is keyed on placement classes rather than per-device slot classes). Empty
// on a single-device worker, which pins through DeviceWorker's own path
// instead of Worker::load_pin_file().
struct PinClassReport {
    std::vector<std::string> devices;
    std::vector<std::vector<uint64_t>> class_bytes;
    std::vector<std::vector<uint64_t>> class_slots;
    std::vector<std::vector<uint64_t>> class_cap;
    std::vector<std::vector<uint64_t>> class_pinned;
    std::vector<std::vector<uint64_t>> class_skipped;
};
void                   test_reset_pin_class_report();
const PinClassReport & test_pin_class_report();

// Test-only: exercises the exact index-bucketing algorithm
// Worker::assignment_groups uses internally (in wp-expert-worker.cpp) to turn
// a request's assignments into per-device sub-dispatch groups -- one group
// per distinct owning device, ordered by device id, each group's assignment
// indices kept in their original encounter order -- without needing a live
// multi-device Worker. `owner_for_index[i]` is the device that owns
// assignment i (what owning_device_for_page would have returned); pass a
// synthetic owner sequence to test the bucketing shape directly.
struct AssignmentGroupsTestReport {
    std::vector<size_t>              devices;
    std::vector<std::vector<size_t>> indices;
};
AssignmentGroupsTestReport test_bucket_assignment_groups(
        const std::vector<size_t> & owner_for_index);

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
// WP_EXPERT_FOLD_LAST uses the same parser (CUDA "1" kills {MUL,ADD} fusion).
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

// WP_EXPERT_OWNER_OVERFLOW -- which devices absorb the pages that do NOT fit
// anywhere once every device's usable capacity in their size class is gone.
//
// Measured 2026-09-02 in production: 24576 pages against ~16600 usable slots,
// so ~8000 pages MUST overflow. Spreading that overflow proportionally to
// total capacity handed ROCm1 -- the 2.78 GB/s Thunderbolt device the `hot`
// policy exists to keep fully resident -- 3197 pages it can never hold, which
// is the exact page-in traffic the policy was supposed to remove. Dumping all
// of it on the lowest-priority device instead would bury the 1200-slot CPU
// tier, so the split is a policy, not a rule of thumb.
//
// Value: comma-separated device names, each with an optional ":<weight>"
// non-negative integer, e.g. "ROCm0:4,CPU:1". A device with no weight (or a
// weight of 0, or an unparseable one) is weighted by its USABLE CAPACITY in
// the size class being spread. Unknown names warn and are ignored; duplicates
// keep the first spelling. Whitespace around the value, each name and each
// weight is ignored.
//
// Unset/empty (`from_env == false`) means the documented default: every
// device EXCEPT the first-priority one, weighted by usable capacity in the
// class. Either way the first-priority device never receives overflow unless
// it is the ONLY device on the list -- that is the whole point of the policy.
struct HotOwnerOverflow {
    // Device indices, in the order the list named them (or device-list order
    // for the default). Parallel to `weights`.
    std::vector<size_t>   devices;
    // 0 means "use this device's usable capacity in the class as the weight".
    std::vector<uint64_t> weights;
    // True when WP_EXPERT_OWNER_OVERFLOW named at least one KNOWN device. A
    // value that names only unknown devices is not a list, it is a typo, and
    // falls back to the default rather than silently owning nothing.
    bool from_env = false;
};

HotOwnerOverflow parse_owner_overflow(
        const char * env, const std::vector<std::string> & device_names);

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
    // Which devices absorb pages that fit nowhere. Default-constructed (no
    // `from_env`) means the documented default: everyone but priority[0].
    HotOwnerOverflow overflow;
};

struct HotOwnerPlan {
    std::vector<size_t> owner;        // per page id
    std::vector<char>   from_ranked;  // per page id: came off `ranked` into real capacity
    // Per page id: placed by the OVERFLOW spread, i.e. its size class had no
    // free capacity left on any device. Distinct from a page that is merely
    // absent from `ranked` -- one of those still lands in real, free capacity
    // (from_ranked == 0 && from_overflow == 0), and the two categories are
    // logged separately because only the overflow count is page-in traffic.
    std::vector<char>   from_overflow;
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
//  2. RESIDUAL FILL, per size class, over the still-unassigned pages of that
//     class sorted by ascending page id (K of them), WHEN capacity is left in
//     the class: spread them proportionally over the devices that still have
//     capacity, weighted by that REMAINING capacity, using the integer-band
//     idiom of the proportional policy (page k of K goes to the device whose
//     cumulative weight band contains k*W/K). The spread does not consume
//     capacity: K is normally far larger than W, and every one of these pages
//     is a cold page that will be paged in on demand wherever it lands. These
//     pages are `from_ranked == 0 && from_overflow == 0`.
//  3. OVERFLOW, per size class, when NO device has capacity left in the class:
//     the same ascending-page-id integer-band spread, but only over the
//     devices `overflow` allows (default: every device except priority[0]),
//     weighted by `overflow.weights` -- with a 0 weight meaning "this
//     device's usable capacity in this class". priority[0] is dropped from
//     the list unless it is the only device on it. These pages are
//     `from_overflow == 1`.
//     If NO allowed device has usable capacity in this class -- even an
//     explicitly weighted one, since a weight must not park a page on a
//     device with no slot of its size -- the weights fall back to the class's
//     TOTAL capacity across every device EXCEPT
//     priority[0]; if that is empty too, to the class's total capacity across
//     ALL devices (so a page still never lands on a device that cannot
//     physically hold its size class); and if THAT is empty the page keeps
//     its proportional owner.
//  4. HOT_OWNER_NO_CLASS pages keep their proportional owner.
//
// INVARIANT (warned about at runtime, asserted in the unit test): the
// first-priority device's owned count in a class is never more than its usable
// capacity in that class, so `expected_fully_resident` is `yes` for it
// whenever `ranked` holds at least that many pages of the class.
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
