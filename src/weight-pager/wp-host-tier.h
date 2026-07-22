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
        size_t offset = 0;
        size_t bytes  = 0;
    };

    bool acquire_slot_(int page_idx, size_t n, size_t & offset_out);
    bool evict_one_lru_();
    void erase_resident_(int page_idx);
    void touch_lru_(int page_idx);

    uint8_t * arena_       = nullptr;
    size_t    budget_bytes_ = 0;
    size_t    used_bytes_   = 0;
    size_t    high_water_   = 0;
    bool      backend_pinned_ = false;
    bool      mlocked_        = false;

    std::unordered_map<int, Resident> resident_;
    std::unordered_map<size_t, std::vector<size_t>> free_lists_;
    std::deque<int> lru_;
    mutable std::mutex mu_;
};

}  // namespace wp
