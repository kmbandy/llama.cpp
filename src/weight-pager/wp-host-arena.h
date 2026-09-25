#pragma once

// HostArena - fixed-stride pinned host RAM tier for weight pages.
//
// Replaces wp-host-tier's variable-size free-list arena with a fixed-stride
// slot allocator: every entry is exactly `entry_bytes` (the largest catalog
// page size), so there is no fragmentation and no free-list bucketing --
// slot N's address is always base + N * entry_bytes within its chunk. This
// is pure C++ bookkeeping; the caller supplies the allocator/deallocator so
// the slab/LRU/state-machine behavior is unit-testable without a GPU (the
// pinned host allocation itself is a later task's concern).
//
// State machine per entry: Free -> Reading -> Resident -> Free. A page is
// reserved (Free -> Reading) by begin_read(), which hands back a pointer to
// write into; finish_read() lands it (-> Resident) or discards it (-> Free)
// depending on whether the read succeeded. Resident entries are read via
// borrow()/release(), which do NOT change state -- they just refcount so an
// outstanding borrow blocks eviction.
//
// Two LRU lists partition Resident, unpinned entries: `spec_lru_` holds
// speculative (prefetched-but-unused) entries, `lru_` holds demand entries.
// Eviction always drains spec_lru_ first so a misprediction never displaces
// a page the caller actually demanded. Pinned entries live in neither list
// (pin() removes the entry from its LRU) so LRU eviction never considers
// them. Reading entries are also in neither list.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <condition_variable>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace wp {

class HostArena {
public:
    using Handle = uint64_t;                       // entry generation, never reused
    static constexpr Handle kInvalidHandle = 0;
    enum class State : uint8_t { Free, Reading, Resident };

    struct Config {
        size_t budget_bytes    = 0;    // total arena bytes (rounded DOWN to whole entries)
        size_t entry_bytes     = 0;    // fixed stride = largest catalog page size
        size_t chunk_bytes     = (size_t) 1 << 30;   // allocation granularity
        int    spec_frac_pct   = 25;   // max % of entries that may be speculative
        int    pinned_cap_pct  = 90;   // max % of entries that may be pinned
        int    read_inflight_max = 16; // max entries in Reading at once
        // Retention cap, separate from budget_bytes: a release()/finish_read()
        // that leaves a Resident, unborrowed, unpinned entry pool over this
        // many bytes trims LRU (speculative first) down to it, in the same
        // locked call. 0 (default) frees an entry the instant nothing holds
        // it -- the pre-arena StagingPool behaviour, byte-for-byte. Pinned
        // bytes never count against this cap.
        size_t tier_bytes      = 0;
        // Frequency-gated admission (WP_HOST_TIER_POLICY=freq_admit in the
        // worker). Default false: byte-for-byte the plain-LRU behaviour
        // above. See the block comment above admit_landed_locked_() in the
        // .cpp for the policy itself and why plain LRU gets ~0 hit rate on
        // a cyclic sweep bigger than the tier.
        bool   freq_admission  = false;
        // Count-min sketch shape backing freq_admission. sketch_width_ is
        // per-row counters (rows fixed at kSketchRows); ~0 memory either way
        // (4 * 65536 bytes at the default) but exposed for the unit tests
        // to exercise aging on a small sketch without a slow test.
        size_t sketch_width    = 65536;
    };
    // alloc(bytes) returns 4096-aligned memory or nullptr; free(ptr, bytes).
    using Allocator   = std::function<void *(size_t)>;
    using Deallocator = std::function<void(void *, size_t)>;

    HostArena() = default;
    ~HostArena();
    HostArena(const HostArena &) = delete;
    HostArena & operator=(const HostArena &) = delete;

    // Allocates ceil(budget/chunk) chunks; a failed chunk stops growth (shrink,
    // never fallback to a different allocator). Returns false only if zero
    // entries fit. Idempotent: second call returns is_initialized().
    bool init(const Config & cfg, Allocator alloc, Deallocator dealloc);
    void shutdown();
    bool is_initialized() const;

    // --- read path (miss) ---
    // Reserve an entry for `page_idx` and mark it Reading. Evicts LRU if no
    // Free entry (never a Reading, borrowed, or pinned entry; speculative
    // entries first). Fails if: page already Reading/Resident (caller must
    // borrow()), read_inflight_max reached, speculative && spec cap full and
    // no speculative victim, or nothing evictable.
    bool begin_read(int page_idx, bool speculative, void ** data_out, Handle * handle_out);
    // Blocking variant: waits (up to timeout_ms total) instead of failing
    // immediately when begin_read() would refuse for capacity reasons
    // (inflight cap, or nothing evictable right now). Every release() and
    // finish_read() call wakes waiters, so this succeeds as soon as ANY
    // reader thread -- of this batch or another -- frees an entry, which is
    // what lets one batch's page count exceed read_inflight_max: readers
    // claim, read, and release concurrently rather than every page of a
    // batch being reserved up front. Returns false only after timeout_ms
    // elapses with the request still refused.
    bool begin_read_wait(int page_idx, bool speculative, void ** data_out,
                         Handle * handle_out, uint64_t timeout_ms);
    // Tri-state variant for a DEMAND caller that will borrow() on a hit:
    //   Reserved -- the entry is Reading, owned by the caller (as begin_read).
    //   Present  -- the page is Resident (already, or after waiting for a
    //               concurrent Reading of it -- another connection's demand
    //               read or a speculative landing -- to finish_read()): the
    //               caller must borrow() it. NEVER read the page again.
    //   Timeout  -- capacity refusal persisted for the whole timeout_ms, or
    //               the concurrent read never landed in time. timeout_ms is an
    //               absolute deadline from the call, not per retry.
    // A concurrent read that fails (finish_read ok=false) frees the entry, and
    // this call then reserves it for the caller (Reserved) instead.
    enum class Reserve : uint8_t { Reserved, Present, Timeout };
    Reserve reserve_wait(int page_idx, bool speculative, void ** data_out,
                         Handle * handle_out, uint64_t timeout_ms);
    // ok=true: Reading -> Resident. ok=false: Reading -> Free (bytes discarded),
    // regardless of keep_borrowed. keep_borrowed=true (ok=true only) hands the
    // caller ONE outstanding borrow atomically with the landing -- `handle` is
    // still the borrow token, ever_borrowed is set, and the entry cannot be
    // evicted until a matching release(page_idx, handle) drops it back to
    // zero. This is what lets a reader thread land a page and an in-flight
    // async H2D read from it survive a concurrent eviction elsewhere in the
    // arena: the caller holds the entry for the copy's whole lifetime instead
    // of racing a lazily-fenced buffer-reuse check. Applies the tier_bytes
    // trim afterward either way (an entry landing over the cap while NOT
    // held still trims other entries down to it). Stale handle: no-op.
    // prefill_hint: true iff this page-in belongs to a prefill request
    // (n_tokens > 1 at the worker -- see PageIn::prefill in
    // wp-expert-worker.cpp). Only consulted when cfg_.freq_admission is set
    // and the entry is a non-speculative demand landing; see
    // admit_landed_locked_() in the .cpp for why decode (prefill_hint=false)
    // is deliberately never gated -- gating it regressed decode's own hit
    // rate relative to plain LRU in the docs/dev/sim-host-tier.py trials
    // (freq_admit vs freq_admit_phase), because decode's legitimate,
    // already-LRU-friendly reuse doesn't need or benefit from an admission
    // filter, and gating it only slows how fast the cache tracks it.
    void finish_read(int page_idx, Handle handle, bool ok, bool keep_borrowed = false,
                     bool prefill_hint = false);

    // --- hit path ---
    // Resident only. Increments borrow count, touches LRU, clears
    // `speculative` (a demand hit confirms a prediction) unless
    // demand=false. Returns false on miss or if the entry is Reading.
    bool borrow(int page_idx, const void ** src_out, Handle * handle_out, bool demand = true);
    void release(int page_idx, Handle handle);

    // --- pinning (coding hot set) ---
    // Marks a Resident entry pinned (LRU skips it). Fails if not Resident or
    // pinned cap reached. The cap is pinned_cap_pct of the TIER entries
    // (tier_bytes / entry_bytes) only -- the read_inflight_max in-flight
    // entries are never pinnable, so a pin set cannot starve readers.
    bool pin(int page_idx);
    void unpin(int page_idx);

    // --- introspection ---
    State  state_of(int page_idx) const;   // Free if unknown
    bool   is_resident(int page_idx) const;
    size_t entry_bytes()     const;
    size_t tier_bytes()      const;   // Config::tier_bytes (retention cap)
    size_t entry_count()     const;
    size_t resident_count()  const;
    size_t resident_bytes()  const;
    size_t pinned_bytes()    const;
    size_t spec_bytes()      const;
    size_t reading_count()   const;
    size_t chunk_count()     const;
    uint64_t evictions()             const;
    uint64_t spec_evicted_unused()   const;
    uint64_t spec_promotions()       const;
    uint64_t begin_read_refusals()   const;  // inflight cap / nothing evictable
    // freq_admission only: demand landings placed at the cold (immediately
    // evictable) end instead of MRU because the incoming page's sketch
    // estimate lost to the current LRU victim's. Always 0 when the policy
    // is off.
    uint64_t admission_cold_landed() const;
    // Proof-of-lookup counters, unconditional (not freq_admission-gated):
    // every borrow() call is one cache lookup attempt (the only lookup path
    // -- the demand pagein path is currently the only caller). Added
    // 2026-09-25 after a live run measured n_host_hit=0 with no way to tell
    // whether the lookup was ever reached at all; see the block comment
    // above begin_read_locked_'s speculative-eviction branch in the .cpp for
    // what was actually wrong (a separate bug, not this counter's fault, but
    // this pair is what would have made it visible immediately instead of
    // needing a live-vs-simulator diff).
    uint64_t lookups()      const;   // every borrow() call, hit or miss
    uint64_t lookup_hits()  const;   // borrow() calls that found the page Resident


private:
    // Which LRU list (if any) currently holds this entry. Pinned and Reading
    // entries are in neither (None); a Resident entry is in exactly one of
    // spec_lru_ (speculative) or lru_ (demand) unless it is pinned.
    enum class ListLoc : uint8_t { None, Lru, SpecLru };

    struct Entry {
        int      page_idx     = -1;
        State    state        = State::Free;
        int      borrows      = 0;
        bool     speculative  = false;
        bool     pinned       = false;
        bool     ever_borrowed = false;   // set by any borrow() (demand or peek), reset
                                           // when the entry becomes Reading again; an
                                           // eviction while still speculative only counts
                                           // as "unused" (spec_evicted_unused_) when false
        Handle   gen          = kInvalidHandle;
        uint8_t * data        = nullptr;
        ListLoc  loc          = ListLoc::None;
        std::list<size_t>::iterator lru_pos;   // valid iff loc != None
    };

    struct Chunk {
        uint8_t * base  = nullptr;
        size_t    bytes = 0;
    };

    // Which side of the partition an eviction may take from. Any scans
    // spec_lru_ first, falling back to lru_; SpecOnly restricts the victim to
    // spec_lru_ (used when a new speculative read needs a victim and the
    // spec cap is already full -- it must not steal room from a demand page
    // just to seat a guess).
    //
    // Eviction is a scan: walk the applicable list from the front, skipping
    // a borrowed entry IN PLACE (it is neither removed nor reordered, so a
    // later attempt sees it in the same spot), and evict the first unborrowed
    // entry found. Refuse only when the scan reaches the end of both lists
    // without finding one. This keeps eviction order exactly LRU order among
    // the entries that are actually evictable at the moment.
    enum class EvictScope { Any, SpecOnly };

    bool     evict_one_locked_(EvictScope scope);
    void     trim_to_tier_cap_locked_();
    // Caller holds mu_. Shared body of begin_read()/begin_read_wait().
    bool     begin_read_locked_(int page_idx, bool speculative, void ** data_out, Handle * handle_out);
    void     remove_from_list_locked_(size_t idx);
    void     insert_mru_locked_(size_t idx, bool speculative);
    void     touch_locked_(size_t idx);
    size_t   spec_cap_entries_() const;
    size_t   pinned_cap_entries_() const;

    // --- frequency sketch (freq_admission only) -----------------------------
    // Count-min sketch, kSketchRows independent hashes over sketch_width_
    // counters each. Persists across eviction (unlike Entry::borrows, which
    // resets to 0 the instant an entry frees) -- that persistence is the
    // whole point: it is what lets a page that was evicted many sweeps ago
    // still outbid a first-time page at the next admission decision. See
    // admit_landed_locked_() in the .cpp for how it is used.
    static constexpr int kSketchRows = 4;
    void     sketch_record_locked_(int page_idx);
    uint32_t sketch_estimate_locked_(int page_idx) const;
    size_t   sketch_lane_locked_(int page_idx, int row) const;
    // Called from finish_read()'s ok==true branch instead of a bare
    // insert_mru_locked_ call when cfg_.freq_admission is set.
    void     admit_landed_locked_(size_t idx, bool prefill_hint);

    mutable std::mutex mu_;
    // Notified by every release() and finish_read() (both outcomes) -- the
    // only two calls that can turn a not-evictable entry into an evictable
    // or free one. Backs begin_read_wait().
    std::condition_variable cv_;

    Config   cfg_;
    Allocator   alloc_;
    Deallocator dealloc_;
    bool     initialized_ = false;

    std::vector<Chunk>  chunks_;
    std::vector<Entry>  entries_;
    std::vector<size_t> free_;                       // indices of Free entries
    std::unordered_map<int, size_t> by_page_;         // page_idx -> entry index (Reading|Resident)

    // LRU order: front = least recently used, back = most recently used.
    std::list<size_t> lru_;
    std::list<size_t> spec_lru_;

    Handle next_gen_ = kInvalidHandle + 1;

    size_t   resident_count_ = 0;
    size_t   resident_bytes_ = 0;
    size_t   pinned_bytes_   = 0;
    size_t   spec_bytes_     = 0;
    size_t   reading_count_  = 0;

    uint64_t evictions_           = 0;
    uint64_t spec_evicted_unused_ = 0;
    uint64_t spec_promotions_     = 0;
    uint64_t begin_read_refusals_ = 0;
    uint64_t admission_cold_landed_ = 0;
    uint64_t lookups_     = 0;
    uint64_t lookup_hits_ = 0;

    // Sketch storage: kSketchRows * cfg_.sketch_width counters, row r's
    // lane at r * cfg_.sketch_width + (hash % cfg_.sketch_width). Empty
    // (never allocated) when cfg_.freq_admission is false.
    std::vector<uint8_t> sketch_;
    uint64_t             sketch_ops_    = 0;
    uint64_t             sketch_period_ = 0;   // halve all counters every this many records
};

}  // namespace wp
