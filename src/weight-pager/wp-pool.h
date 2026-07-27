#pragma once

// PoolAllocator - a fixed-size VRAM slot ring with LRU eviction.
//
// One pool per ggml_backend_buffer_type_t. Phase 1 is single-device by
// design (the WeightPager rejects multi-device configs at init), but the
// API takes a buffer-type so per-device pools are a drop-in extension.
//
// Allocation strategy:
//   - All n_slots slots are co-allocated in a single ggml_backend_buffer
//     (fixes B-P4 in docs/dev/memory-tier-bug-catalog.md: tensors paged in
//     must have a valid `tensor->buffer` matching the device's buffer-type
//     so ggml_cuda_mul_mat's assertion passes).
//   - When alloc_slot() is called and all slots are in use, the LRU slot
//     is evicted: the pool invokes the caller-supplied eviction callback
//     (so the caller can clear the page metadata that owned the slot)
//     and then hands the slot to the new owner.
//   - The pool itself does not know what a "page" is. It only knows
//     "slot i was last used at tick T."

#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

namespace wp {

class PoolAllocator {
public:
    // Called once, with the slot index that the pool is about to overwrite.
    // The callee MUST clear any external state pointing at that slot (page
    // table entries etc.) before returning.
    using EvictionCallback = std::function<void(int slot_idx)>;

    PoolAllocator() = default;
    ~PoolAllocator();

    PoolAllocator(const PoolAllocator &)             = delete;
    PoolAllocator & operator=(const PoolAllocator &) = delete;

    // Allocate the underlying ggml backend buffer. Must be called once
    // before any slot operations. Returns false on failure (allocation
    // failed; pool unusable).
    //
    // After success: vram_buf() and slot_ptr(i) for 0 <= i < n_slots() are
    // valid. The buffer is freed when the PoolAllocator is destroyed.
    //
    // device_idx (optional, default -1) enables the MAD-234 UMA safety check:
    // on integrated GPUs (APUs like Strix Halo gfx1151, Phoenix gfx1103,
    // Renoir/Cezanne gfx90c) the buffer-type allocator pulls from system RAM
    // — an oversized pool there silently swaps out the rest of the system,
    // grinding the box to a halt with no clean error. With device_idx >= 0
    // we look up the device, detect UMA, and refuse early with an actionable
    // error if total > MemAvailable - 2 GiB. Pass -1 to skip the check (tests,
    // discrete-only setups).
    // `extra_alignment` forces every slot offset to be a multiple of it, on top
    // of the buffer type's own alignment. Vulkan needs this: its quantized
    // matmul indexes the weight buffer as an array of quant blocks, so an
    // expert's base must be an exact multiple of the block byte size (210 for
    // Q6_K) or it cannot be expressed as a block index at all. The buffer-type
    // alignment alone (256 on Vulkan) is not a multiple of that. Harmless
    // elsewhere: CUDA/HIP pass raw byte pointers and pass 1 here.
    bool init(ggml_backend_buffer_type_t buft,
              int                        n_slots,
              size_t                     slot_size,
              int                        device_idx      = -1,
              size_t                     extra_alignment = 1);

    // Register the eviction callback. Optional; default is a no-op.
    void set_eviction_callback(EvictionCallback cb) { on_evict_ = std::move(cb); }

    // Acquire a slot. If all slots are in use, the LRU slot is evicted via
    // the registered eviction callback and reused. The returned slot index
    // is marked as used and its LRU tick is bumped.
    //
    // Pinned slots (pin_count > 0) are NEVER evicted — they are skipped in
    // the LRU walk. If every slot is pinned, alloc_slot returns -1 and logs
    // a warning. Callers that get -1 should treat it as transient pressure
    // (eviction will succeed once an op completes and unpins its slots).
    //
    // Returns -1 if the pool is uninitialised, n_slots == 0, or all slots
    // are pinned.
    int alloc_slot(size_t requested_size = 0);

    // Free (unused + unpinned) slots only — never LRU-evicts. For sample
    // oracle / speculative prefetch that must not thrash the MoE working set.
    // Returns -1 if none free.
    int alloc_slot_no_evict(size_t requested_size = 0);

    // Count of slots that are free and unpinned (no eviction needed to use).
    int n_free_unpinned() const;

    // --- Speculative eviction tier (cross-layer prefetch) ------------------
    // A slot flagged speculative holds a prefetched-but-not-yet-demanded page.
    // alloc_slot() evicts the LRU speculative slot BEFORE touching the pinned/
    // hot working set, so speculation never evicts live pages. mark_used()
    // (a demand hit) promotes a speculative slot to non-speculative.
    void set_speculative(int slot_idx, bool spec);
    bool is_speculative(int slot_idx) const;
    int  n_speculative() const;

    // Bump the LRU tick of an already-allocated slot, signalling a cache
    // hit. Caller must ensure the slot was previously returned by
    // alloc_slot() and has not been evicted.
    void mark_used(int slot_idx);

    // Bump the LRU tick WITHOUT promoting a speculative slot. Use this when a
    // prefetch's read lands (harvest): the page is now resident but DEMAND has
    // not asked for it yet, so it must stay in the speculative tier and remain
    // the first thing evicted. mark_used() is for genuine demand hits only --
    // calling it on harvest clears speculative_ and the tier never accumulates,
    // which collapses alloc_slot's speculative-first eviction onto the demand
    // working set.
    void touch_lru(int slot_idx);

    // Pin / unpin a slot to protect it from eviction. Refcounted: a slot
    // pinned twice must be unpinned twice before becoming evictable again.
    // Used by the eval-callback to keep slots referenced by an in-flight
    // op safe from prefetch-triggered eviction (MAD-231).
    //
    // pin_slot: increments pin_count_[slot]. No-op (with warn) on OOB or
    // overflow (>= 65535).
    // unpin_slot: decrements pin_count_[slot]. No-op (with warn) on OOB or
    // underflow (called more times than pinned).
    void pin_slot(int slot_idx);
    void unpin_slot(int slot_idx);

    // Inspection helpers (tests + instrumentation).
    bool   is_pinned(int slot_idx) const;
    int    pin_count(int slot_idx) const;
    int    n_pinned() const;

    // Diagnostic: how many times alloc_slot's LRU walk has skipped past a
    // pinned slot. A growing number under normal load is expected (every
    // op that needs eviction will skip pin'd slots from the previous op);
    // a number trending toward n_slots * n_evictions suggests the pin set
    // is dominating the pool, i.e. n_slots is too small for the working set.
    uint64_t lru_walk_pinned_skips() const { return lru_walk_pinned_skips_; }

    // ---------------------------------------------------------------------
    // MAD-237 — per-slot popularity counter + CLOCK-style hot protection.
    //
    // Purpose: keep frequently-touched slots resident across LRU pressure.
    // For MoE workloads where the routing entropy is high enough that pure
    // LRU approaches random replacement, protecting a small "hot" set turns
    // worst-case sync-pread misses into hits.
    //
    // Default: threshold == 0 ⇒ pure LRU, byte-identical to MAD-231-only
    // behavior. Operators enable via WP_HOT_HIT_THRESHOLD env (wired in
    // WeightPager::init). Hipfire's own analysis cautions this may be
    // marginal vs flat LRU on A3B-class entropy; ship the counter for
    // telemetry value regardless.
    //
    // Algorithm: mark_used increments hit_count_[slot]. alloc_slot's LRU
    // walk does TWO passes when threshold > 0:
    //   Pass A: pick LRU among (unpinned AND hit_count <= threshold)
    //   Pass B: if A is empty (all unpinned slots are hot), pick LRU among
    //           unpinned regardless of hit count. Always evictable.
    // Pinned slots are never evicted in either pass (per MAD-231).
    //
    // Decay: every kDecayEvery evictions, halve every hit count. Keeps
    // counters from monotonically growing and lets cold-becomes-hot pages
    // get demoted over time. kDecayEvery is a compile-time constant; can
    // be made configurable if needed.
    // ---------------------------------------------------------------------

    void     set_hot_hit_threshold(uint32_t t) { hot_hit_threshold_ = t; }
    uint32_t hot_hit_threshold() const         { return hot_hit_threshold_; }
    uint32_t hit_count(int slot_idx) const;

    // Diagnostic: how many times alloc_slot's LRU walk has skipped past a
    // HOT slot (Pass A). Distinct from pinned skips. If this approaches
    // n_evictions * n_slots, the hot set is dominating the pool — operator
    // should lower WP_HOT_HIT_THRESHOLD or grow n_slots.
    uint64_t lru_walk_hot_skips() const { return lru_walk_hot_skips_; }

    // Diagnostic: how many decay passes have run.
    uint64_t n_decays() const { return n_decays_; }

    // Explicit free. Rarely needed — LRU eviction is the primary path.
    // Useful for tests and for the pager's shutdown flow.
    void release_slot(int slot_idx);

    // Pointer into the slot's memory region. Valid for the lifetime of the
    // pool. Returns nullptr if slot_idx is out of range.
    void * slot_ptr(int slot_idx) const;

    // Stable slot base for graph capture. Alias for slot_ptr(), named for
    // MAD-P1 callers that need the fixed pool-lifetime capture surface.
    void * slot_base_for_capture(int slot_idx) const { return slot_ptr(slot_idx); }

    // Backing buffer; pass to tensor->buffer when paging a tensor in.
    ggml_backend_buffer_t vram_buf() const { return buf_; }

    int    n_slots()   const { return n_slots_;   }
    size_t slot_size() const { return slot_size_; }
    size_t slot_size(int slot_idx) const;
    size_t pool_size() const { return arena_size_; }
    void * pool_base() const { return base_; }
    bool   size_class_slots_enabled() const { return size_class_slots_; }

    // Inspect LRU state for tests / metrics.
    int lru_slot() const;

private:
    ggml_backend_buffer_t buf_       = nullptr;
    void *                base_      = nullptr;
    int                   n_slots_   = 0;
    size_t                slot_size_ = 0;
    size_t                arena_size_ = 0;
    size_t                slot_alignment_ = 1;
    bool                  size_class_slots_ = false;
    size_t                high_water_ = 0;
    uint64_t              tick_      = 0;

    std::vector<bool>     used_;
    std::vector<uint64_t> last_used_;
    std::vector<size_t>   slot_offset_;
    std::vector<size_t>   slot_bytes_;
    std::vector<size_t>   slot_class_;
    std::unordered_map<size_t, std::vector<int>> free_by_class_;
    // Per-slot pin refcount (MAD-231). Slots with pin_count_>0 are skipped
    // by alloc_slot's LRU walk. uint16_t allows 65k simultaneous pins per
    // slot — orders of magnitude more than the conservative
    // pin-during-op-then-unpin lifecycle needs; the cap protects against
    // refcount bugs by failing loudly rather than overflowing silently.
    std::vector<uint16_t> pin_count_;
    // Per-slot speculative flag (cross-layer prefetch). 1 = prefetched, not yet
    // demanded; evicted before the working set and cleared on demand-hit/reuse.
    std::vector<char>     speculative_;
    EvictionCallback      on_evict_;
    // Telemetry: count of LRU-walk iterations that skipped a pinned slot.
    uint64_t              lru_walk_pinned_skips_ = 0;
    // MAD-237 — popularity counter + CLOCK hot-protection.
    std::vector<uint32_t> hit_count_;
    uint32_t              hot_hit_threshold_     = 0;   // 0 = disabled
    uint64_t              lru_walk_hot_skips_    = 0;
    uint64_t              n_evictions_since_decay_ = 0;
    uint64_t              n_decays_              = 0;
    static constexpr uint64_t kDecayEvery = 1024;       // halve counts every N evictions

    int    alloc_slot_fixed_();
    int    alloc_slot_size_class_(size_t requested_size);
    int    take_free_size_class_slot_(size_t requested_class);
    int    pick_size_class_victim_(size_t requested_class,
                                   int & n_pinned_skipped,
                                   int & n_hot_skipped) const;
    size_t size_class_for_(size_t requested_size) const;
    void   decay_after_eviction_();
};

// ---------------------------------------------------------------------------
// MAD-234 — UMA / APU safety helpers
//
// Integrated GPUs (Strix Halo gfx1151, Phoenix gfx1103, Renoir/Cezanne
// gfx90c, etc.) share physical memory between VRAM and host RAM. The HIP
// allocator silently succeeds for IOMMU-mapped buffers that exceed actual
// MemAvailable, then page-faults each touched page through swap — which on
// a 16 GiB UMA box produces multi-second per-step stalls and ultimately a
// soft-locked desktop. Detect UMA up front and refuse oversized requests
// with an actionable error instead of taking the box down silently.
// ---------------------------------------------------------------------------

// True if `device_idx` is an integrated / UMA GPU. Uses
// hipDeviceGetAttribute(hipDeviceAttributeIntegrated) as the primary signal,
// AND a fallback gfx-string prefix match (gfx115x / gfx1103 / gfx90c) for
// older HIP runtimes that don't report Integrated reliably. Returns false on
// non-HIP builds and for any error path.
bool is_uma_device(int device_idx);

// Lower-level helper: returns true iff `arch_name` starts with one of the
// known UMA gfx-string prefixes. Exposed at namespace scope so unit tests
// can validate the prefix table without a real HIP device.
bool is_uma_archname(const char * arch_name);

// Read MemAvailable from /proc/meminfo (Linux only), in bytes. Returns 0
// on non-Linux or any parse failure. MemAvailable is the kernel's estimate
// of bytes that can be allocated without swapping — more accurate than
// MemFree for the UMA pre-flight check.
size_t read_mem_available_bytes();

}  // namespace wp
