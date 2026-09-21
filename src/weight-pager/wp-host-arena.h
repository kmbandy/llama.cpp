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
    void finish_read(int page_idx, Handle handle, bool ok, bool keep_borrowed = false);

    // --- hit path ---
    // Resident only. Increments borrow count, touches LRU, clears
    // `speculative` (a demand hit confirms a prediction) unless
    // demand=false. Returns false on miss or if the entry is Reading.
    bool borrow(int page_idx, const void ** src_out, Handle * handle_out, bool demand = true);
    void release(int page_idx, Handle handle);

    // --- pinning (coding hot set) ---
    // Marks a Resident entry pinned (LRU skips it). Fails if not Resident or
    // pinned cap reached.
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
};

}  // namespace wp
