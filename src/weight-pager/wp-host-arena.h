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
    // ok=true: Reading -> Resident. ok=false: Reading -> Free (bytes discarded).
    // Stale handle: no-op.
    void finish_read(int page_idx, Handle handle, bool ok);

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
        int      page_idx    = -1;
        State    state       = State::Free;
        int      borrows     = 0;
        bool     speculative = false;
        bool     pinned      = false;
        Handle   gen         = kInvalidHandle;
        uint8_t * data       = nullptr;
        ListLoc  loc         = ListLoc::None;
        std::list<size_t>::iterator lru_pos;   // valid iff loc != None
    };

    struct Chunk {
        uint8_t * base  = nullptr;
        size_t    bytes = 0;
    };

    // Which side of the partition an eviction may take from. Any tries
    // spec_lru_'s front first, falling back to lru_'s front; SpecOnly
    // restricts the victim to spec_lru_ (used when a new speculative read
    // needs a victim and the spec cap is already full -- it must not steal
    // room from a demand page just to seat a guess).
    //
    // Eviction only ever looks at the single LRU-least (front) candidate of
    // the applicable list(s) -- it does not scan past a borrowed front entry
    // to find a later unborrowed one. A borrowed front entry blocks that
    // eviction attempt entirely (the entry is left "in place": neither
    // removed nor reordered), so the caller sees a refusal and can retry
    // once the borrow is released. This keeps eviction O(1) and matches the
    // observable contract: eviction order is exactly LRU order, never
    // reordered to skip around an in-use entry.
    enum class EvictScope { Any, SpecOnly };

    bool     evict_one_locked_(EvictScope scope);
    void     remove_from_list_locked_(size_t idx);
    void     insert_mru_locked_(size_t idx, bool speculative);
    void     touch_locked_(size_t idx);
    size_t   spec_cap_entries_() const;
    size_t   pinned_cap_entries_() const;

    mutable std::mutex mu_;

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
