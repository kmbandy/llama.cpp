#pragma once

// HostTier - optional pinned/pageable host RAM cache for weight pages.
//
// Enabled only when WP_HOST_BUDGET_BYTES > 0. The allocator bookkeeping is
// plain C++: exact-size free lists, a bump high-water cursor, and an LRU over
// resident page indices. Backend-specific allocation is confined to init() /
// shutdown() so the slab/LRU behavior is unit-testable without a GPU.

#include <cstddef>
#include <cstdint>
#include <deque>
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

    bool store(int page_idx, const void * src_bytes, size_t n);
    // Copy a resident page into caller-owned storage. The copy is completed
    // while holding mu_, so a concurrent store/erase cannot reclaim the arena
    // slot before the caller consumes the bytes. `n` must equal the page size.
    bool lookup(int page_idx, void * dst_bytes, size_t n);

    // Zero-copy borrow: hand back a pointer directly into the arena instead
    // of copying. `n` must equal the page size, exactly as lookup() requires.
    // On a hit, increments the entry's borrow refcount (so eviction skips it
    // until every borrow is released) and touches the LRU, same as lookup().
    // Returns false on any miss (absent page or size mismatch); `src_out` is
    // left untouched in that case and the caller must fall through to a
    // fresh read exactly as a lookup() miss does today.
    //
    // Every successful borrow() MUST be paired with a release(); the pointer
    // handed back is only valid for the arena's lifetime while the borrow is
    // outstanding. See release() below and the borrow/release design at
    // docs/superpowers/specs/2026-07-25-hosttier-zerocopy-promotion-design.md.
    bool borrow(int page_idx, const void ** src_out, size_t n);

    // Release a borrow taken by borrow(). Decrements the entry's borrow
    // refcount; if it reaches zero and the page was erased (or re-stored)
    // while still borrowed, completes the deferred slot reclamation now.
    void release(int page_idx);

    // Store a page whose bytes live in DEVICE memory (D2H copy into the arena).
    // The exclusive victim path: a VRAM slot being evicted is moved to RAM.
    // Returns false if the tier is disabled or the copy fails.
    bool store_from_device(int page_idx, const void * device_bytes, size_t n);

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
        size_t offset       = 0;
        size_t bytes        = 0;
        int    borrow_count = 0;  // outstanding borrow()s; blocks eviction while > 0
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
    bool      mlocked_        = false;

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
    // page_idx they were retired under (release() has no other handle);
    // a deque per key because a page could in principle be retired more than
    // once while still-older borrows drain, oldest retirement first.
    std::unordered_map<int, std::deque<Resident>> pending_;
    mutable std::mutex mu_;
};

}  // namespace wp
