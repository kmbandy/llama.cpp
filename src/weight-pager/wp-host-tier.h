#pragma once

// HostTier - optional pinned/pageable host RAM cache for weight pages.
//
// Enabled only when WP_HOST_BUDGET_BYTES > 0. The allocator bookkeeping is
// plain C++: exact-size free lists, a bump high-water cursor, and an LRU over
// resident page indices. Backend-specific allocation is confined to init() /
// shutdown() so the slab/LRU behavior is unit-testable without a GPU.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace wp {

class HostTier {
public:
    HostTier() = default;
    ~HostTier();

    HostTier(const HostTier &)             = delete;
    HostTier & operator=(const HostTier &) = delete;

    bool init(size_t budget_bytes, int device_idx);
    void shutdown();

    // Opaque identifier for one specific entry generation. Assigned when the
    // entry is created (store() / store_from_device()) and never reused, so
    // it distinguishes the exact instance a borrow() call saw from any later
    // entry that happens to reuse the same page_idx (e.g. after an
    // erase()+re-store() while the original was still borrowed). release()
    // uses it to decrement the exact generation the caller borrowed, never a
    // different one that happens to share the page_idx.
    using BorrowHandle = uint64_t;
    static constexpr BorrowHandle kInvalidBorrowHandle = 0;

    bool store(int page_idx, const void * src_bytes, size_t n);
    // Copy a resident page into caller-owned storage. The copy is completed
    // while holding mu_, so a concurrent store/erase cannot reclaim the arena
    // slot before the caller consumes the bytes. `n` must equal the page size.
    bool lookup(int page_idx, void * dst_bytes, size_t n);

    // Zero-copy borrow: hand back a pointer directly into the arena instead
    // of copying. `n` must equal the page size, exactly as lookup() requires.
    // On a hit, increments the entry's borrow refcount (so eviction skips it
    // until every borrow is released), touches the LRU (same as lookup()),
    // and returns that entry's generation handle in `handle_out`. Returns
    // false on any miss (absent page or size mismatch); `src_out` and
    // `handle_out` are left untouched in that case and the caller must fall
    // through to a fresh read exactly as a lookup() miss does today.
    //
    // Every successful borrow() MUST be paired with a release() of the
    // SAME (page_idx, handle) pair -- the pointer handed back is only valid
    // for the arena's lifetime while that specific borrow is outstanding.
    // See release() below and the borrow/release design at
    // docs/superpowers/specs/2026-07-25-hosttier-zerocopy-promotion-design.md.
    bool borrow(int page_idx, const void ** src_out, size_t n, BorrowHandle * handle_out);

    // Release a borrow taken by borrow(). `handle` must be the exact value
    // borrow() returned; it disambiguates the entry to decrement from any
    // OTHER entry that may now occupy the same page_idx (a re-store() while
    // the original was still borrowed creates a new, distinguishable
    // generation -- see erase_resident_()/pending_ below). Decrements that
    // entry's borrow refcount; if it reaches zero and the entry was retired
    // while borrowed, completes the deferred slot reclamation now. A stale or
    // already-fully-released handle is a no-op.
    void release(int page_idx, BorrowHandle handle);

    // Store a page whose bytes live in DEVICE memory (D2H copy into the arena).
    // The exclusive victim path: a VRAM slot being evicted is moved to RAM.
    // Returns false if the tier is disabled or the copy fails.
    bool store_from_device(int page_idx, const void * device_bytes, size_t n);

    // How to copy device bytes into the arena. MUST be set by the owner when
    // the pool is anything other than a raw-addressable HIP/CUDA allocation:
    // a Vulkan pool "pointer" is a sentinel base plus an offset, so the
    // built-in hipMemcpy fallback would issue a device copy from an address
    // that does not exist. With no reader set and no HIP/CUDA build,
    // store_from_device refuses rather than guessing.
    using DeviceReader = std::function<bool(void * dst_host, const void * src_device, size_t n)>;
    void set_device_reader(DeviceReader reader) { device_reader_ = std::move(reader); }

    // Remove a page from the tier (used when it is promoted back to VRAM, so a
    // page never lives in both tiers).
    void erase(int page_idx);

    bool   is_initialized() const;
    size_t budget_bytes()   const;
    size_t used_bytes()     const;
    size_t high_water()     const;
    size_t resident_count() const;
    bool   backend_pinned() const;
    bool   mlocked()        const;
    bool   contains(int page_idx) const;

private:
    struct Resident {
        size_t       offset       = 0;
        size_t       bytes        = 0;
        int          borrow_count = 0;  // outstanding borrow()s; blocks eviction while > 0
        BorrowHandle gen          = kInvalidBorrowHandle;  // this entry's generation id
    };

    bool acquire_slot_(int page_idx, size_t n, size_t & offset_out);
    bool evict_one_lru_();
    void erase_resident_(int page_idx);
    void touch_lru_(int page_idx);
    void reclaim_(const Resident & r);

    uint8_t * arena_       = nullptr;
    size_t    budget_bytes_ = 0;
    size_t    used_bytes_   = 0;
    size_t    high_water_   = 0;
    bool      backend_pinned_ = false;
    DeviceReader device_reader_;
    bool      mlocked_        = false;

    // Monotonically increasing, never reused. Assigned to a new Resident's
    // `gen` at the moment it is created (store()/store_from_device()), so a
    // re-stored page's entry is a distinguishably different generation from
    // whatever this page_idx's previous entry was.
    BorrowHandle next_gen_ = kInvalidBorrowHandle + 1;

    std::unordered_map<int, Resident> resident_;
    std::unordered_map<size_t, std::vector<size_t>> free_lists_;
    // LRU order: front = least recently used, back = most recently used.
    // lru_pos_ gives O(1) access to a page's node so touch/erase never scan.
    std::list<int> lru_;
    std::unordered_map<int, std::list<int>::iterator> lru_pos_;
    // Entries retired (erase()'d, or displaced by a re-store of the same
    // page_idx) while still borrowed. Removed from resident_/lru_ immediately
    // -- so contains() goes false and a re-store gets a fresh slot with no
    // aliasing -- but their arena slot is withheld from free_lists_ until
    // every outstanding borrow_count on them is released(). Keyed by the
    // entry's OWN generation handle, not by page_idx: page_idx can collide
    // with whatever new entry now occupies resident_[page_idx], but a
    // generation handle never does, which is exactly what lets release()
    // find the exact entry a borrow() call saw with no ambiguity.
    std::unordered_map<BorrowHandle, Resident> pending_;
    mutable std::mutex mu_;
};

}  // namespace wp
