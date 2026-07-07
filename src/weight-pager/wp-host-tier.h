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
    const void * lookup(int page_idx);

    bool   is_initialized() const { return arena_ != nullptr && budget_bytes_ > 0; }
    size_t budget_bytes()   const { return budget_bytes_; }
    size_t used_bytes()     const { return used_bytes_; }
    size_t high_water()     const { return high_water_; }
    size_t resident_count() const { return resident_.size(); }
    bool   backend_pinned() const { return backend_pinned_; }
    bool   mlocked()        const { return mlocked_; }
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
};

}  // namespace wp
