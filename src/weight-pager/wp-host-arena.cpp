#include "wp-host-arena.h"

#include <algorithm>
#include <cassert>

namespace wp {

HostArena::~HostArena() { shutdown(); }

bool HostArena::init(const Config & cfg, Allocator alloc, Deallocator dealloc) {
    std::lock_guard<std::mutex> lock(mu_);
    if (initialized_) return true;     // idempotent
    if (cfg.entry_bytes == 0) return false;

    cfg_     = cfg;
    alloc_   = std::move(alloc);
    dealloc_ = std::move(dealloc);

    const size_t entries_per_chunk = std::max<size_t>(1, cfg_.chunk_bytes / cfg_.entry_bytes);
    size_t want = cfg_.budget_bytes / cfg_.entry_bytes;   // rounds DOWN to whole entries

    while (want > 0) {
        const size_t n     = std::min(want, entries_per_chunk);
        const size_t bytes = n * cfg_.entry_bytes;
        void * p = alloc_(bytes);
        if (p == nullptr) break;   // shrink: stop growing, never fall back to another allocator

        chunks_.push_back({ (uint8_t *) p, bytes });
        for (size_t i = 0; i < n; ++i) {
            Entry e;
            e.data = (uint8_t *) p + i * cfg_.entry_bytes;
            entries_.push_back(e);
            free_.push_back(entries_.size() - 1);
        }
        want -= n;
    }

    initialized_ = !entries_.empty();
    return initialized_;
}

void HostArena::shutdown() {
    std::lock_guard<std::mutex> lock(mu_);
    if (chunks_.empty() && entries_.empty() && !initialized_) return;   // nothing to do

    for (auto & c : chunks_) {
        if (dealloc_) dealloc_(c.base, c.bytes);
    }

    chunks_.clear();
    entries_.clear();
    free_.clear();
    by_page_.clear();
    lru_.clear();
    spec_lru_.clear();

    resident_count_       = 0;
    resident_bytes_       = 0;
    pinned_bytes_         = 0;
    spec_bytes_           = 0;
    reading_count_        = 0;
    evictions_            = 0;
    spec_evicted_unused_  = 0;
    spec_promotions_      = 0;
    begin_read_refusals_  = 0;
    next_gen_             = kInvalidHandle + 1;
    initialized_          = false;
}

bool HostArena::is_initialized() const {
    std::lock_guard<std::mutex> lock(mu_);
    return initialized_;
}

// --- LRU list helpers (caller holds mu_) -----------------------------------

void HostArena::remove_from_list_locked_(size_t idx) {
    Entry & e = entries_[idx];
    if (e.loc == ListLoc::Lru)      lru_.erase(e.lru_pos);
    else if (e.loc == ListLoc::SpecLru) spec_lru_.erase(e.lru_pos);
    e.loc = ListLoc::None;
}

void HostArena::insert_mru_locked_(size_t idx, bool speculative) {
    Entry & e = entries_[idx];
    if (speculative) {
        spec_lru_.push_back(idx);
        e.lru_pos = std::prev(spec_lru_.end());
        e.loc     = ListLoc::SpecLru;
    } else {
        lru_.push_back(idx);
        e.lru_pos = std::prev(lru_.end());
        e.loc     = ListLoc::Lru;
    }
}

void HostArena::touch_locked_(size_t idx) {
    Entry & e = entries_[idx];
    if (e.loc == ListLoc::None) return;   // pinned (or not resident) -- nothing to touch
    const bool speculative = e.loc == ListLoc::SpecLru;
    remove_from_list_locked_(idx);
    insert_mru_locked_(idx, speculative);
}

size_t HostArena::spec_cap_entries_() const {
    return entries_.size() * (size_t) cfg_.spec_frac_pct / 100;
}

size_t HostArena::pinned_cap_entries_() const {
    return entries_.size() * (size_t) cfg_.pinned_cap_pct / 100;
}

// Evict a single entry, preferring a speculative victim (a misprediction
// must never displace a page the caller actually demanded). Only the front
// (least-recently-used) entry of the applicable list is ever considered: if
// it is borrowed, this attempt is refused rather than scanning past it for a
// later unborrowed entry, so an in-flight borrow can only ever delay -- never
// reorder -- eviction. SpecOnly refuses outright if spec_lru_'s front is
// unusable rather than falling back to a demand victim, because a full spec
// budget must be relieved from the spec side, never the demand side.
bool HostArena::evict_one_locked_(EvictScope scope) {
    size_t idx    = 0;
    bool   found  = false;
    bool   from_spec = false;

    if (!spec_lru_.empty()) {
        size_t cand = spec_lru_.front();
        if (entries_[cand].borrows == 0) {
            idx = cand; found = true; from_spec = true;
        }
    }
    if (!found) {
        if (scope == EvictScope::SpecOnly) return false;
        if (!lru_.empty()) {
            size_t cand = lru_.front();
            if (entries_[cand].borrows == 0) {
                idx = cand; found = true; from_spec = false;
            }
        }
    }
    if (!found) return false;

    Entry & e = entries_[idx];
    const int page_idx = e.page_idx;

    if (from_spec) {
        spec_lru_.pop_front();
        spec_bytes_ -= cfg_.entry_bytes;
        ++spec_evicted_unused_;   // still speculative when evicted => never used
    } else {
        lru_.pop_front();
    }
    e.loc = ListLoc::None;

    resident_count_ -= 1;
    resident_bytes_ -= cfg_.entry_bytes;
    by_page_.erase(page_idx);

    e.state       = State::Free;
    e.page_idx    = -1;
    e.pinned      = false;
    e.borrows     = 0;
    e.speculative = false;
    e.gen         = kInvalidHandle;   // any outstanding handle to this slot is now stale
    free_.push_back(idx);

    ++evictions_;
    return true;
}

// --- read path ---------------------------------------------------------

bool HostArena::begin_read(int page_idx, bool speculative, void ** data_out, Handle * handle_out) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!initialized_) return false;

    auto it = by_page_.find(page_idx);
    if (it != by_page_.end()) {
        // Already Reading or Resident -- caller must borrow() instead. A
        // Resident hit here still means genuine demand for this page, so
        // touch its recency even though we refuse to start a new read.
        if (entries_[it->second].state == State::Resident) {
            touch_locked_(it->second);
        }
        return false;
    }

    if (reading_count_ >= (size_t) cfg_.read_inflight_max) {
        ++begin_read_refusals_;
        return false;
    }

    size_t idx;
    if (speculative) {
        const size_t spec_cap_bytes = spec_cap_entries_() * cfg_.entry_bytes;
        if (spec_bytes_ + cfg_.entry_bytes > spec_cap_bytes) {
            // Spec budget already full: the victim MUST come from the
            // speculative side, regardless of whether a Free entry exists
            // elsewhere -- otherwise speculative occupancy could creep past
            // its cap by borrowing space intended for demand pages.
            if (!evict_one_locked_(EvictScope::SpecOnly)) {
                ++begin_read_refusals_;
                return false;
            }
            idx = free_.back();
            free_.pop_back();
        } else if (!free_.empty()) {
            idx = free_.back();
            free_.pop_back();
        } else if (evict_one_locked_(EvictScope::Any)) {
            idx = free_.back();
            free_.pop_back();
        } else {
            ++begin_read_refusals_;
            return false;
        }
    } else {
        if (!free_.empty()) {
            idx = free_.back();
            free_.pop_back();
        } else if (evict_one_locked_(EvictScope::Any)) {
            idx = free_.back();
            free_.pop_back();
        } else {
            ++begin_read_refusals_;
            return false;
        }
    }

    Entry & e    = entries_[idx];
    e.page_idx    = page_idx;
    e.state       = State::Reading;
    e.borrows     = 0;
    e.speculative = speculative;
    e.pinned      = false;
    e.gen         = next_gen_++;
    e.loc         = ListLoc::None;

    by_page_[page_idx] = idx;
    ++reading_count_;

    *data_out   = e.data;
    *handle_out = e.gen;
    return true;
}

void HostArena::finish_read(int page_idx, Handle handle, bool ok) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return;

    size_t idx = it->second;
    Entry & e  = entries_[idx];
    if (e.gen != handle || e.state != State::Reading) return;   // stale handle: no-op

    --reading_count_;

    if (ok) {
        e.state = State::Resident;
        ++resident_count_;
        resident_bytes_ += cfg_.entry_bytes;
        if (e.speculative) spec_bytes_ += cfg_.entry_bytes;
        insert_mru_locked_(idx, e.speculative);
    } else {
        // Reading -> Free: discard the bytes, the page never landed.
        by_page_.erase(it);
        e.state    = State::Free;
        e.page_idx = -1;
        e.gen      = kInvalidHandle;
        free_.push_back(idx);
    }
}

// --- hit path ---------------------------------------------------------

bool HostArena::borrow(int page_idx, const void ** src_out, Handle * handle_out, bool demand) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return false;

    size_t idx = it->second;
    Entry & e  = entries_[idx];
    if (e.state != State::Resident) return false;   // miss or Reading

    ++e.borrows;

    if (!e.pinned) {
        if (demand && e.speculative) {
            // A demand hit confirms a prediction: promote out of the
            // speculative side into the demand side.
            remove_from_list_locked_(idx);
            e.speculative = false;
            spec_bytes_ -= cfg_.entry_bytes;
            ++spec_promotions_;
            insert_mru_locked_(idx, false);
        } else {
            touch_locked_(idx);
        }
    }

    *src_out    = e.data;
    *handle_out = e.gen;
    return true;
}

void HostArena::release(int page_idx, Handle handle) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return;   // stale: page no longer tracked

    Entry & e = entries_[it->second];
    if (e.gen != handle) return;        // stale: a different generation now owns this page_idx
    if (e.borrows > 0) --e.borrows;
}

// --- pinning ------------------------------------------------------------

bool HostArena::pin(int page_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return false;

    size_t idx = it->second;
    Entry & e  = entries_[idx];
    if (e.state != State::Resident) return false;
    if (e.pinned) return true;   // already pinned

    const size_t cap_bytes = pinned_cap_entries_() * cfg_.entry_bytes;
    if (pinned_bytes_ + cfg_.entry_bytes > cap_bytes) return false;

    remove_from_list_locked_(idx);   // LRU skips pinned entries
    e.pinned = true;
    pinned_bytes_ += cfg_.entry_bytes;
    return true;
}

void HostArena::unpin(int page_idx) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return;

    size_t idx = it->second;
    Entry & e  = entries_[idx];
    if (!e.pinned) return;

    e.pinned = false;
    pinned_bytes_ -= cfg_.entry_bytes;
    if (e.state == State::Resident) {
        insert_mru_locked_(idx, e.speculative);
    }
}

// --- introspection --------------------------------------------------------

HostArena::State HostArena::state_of(int page_idx) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return State::Free;
    return entries_[it->second].state;
}

bool HostArena::is_resident(int page_idx) const {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = by_page_.find(page_idx);
    return it != by_page_.end() && entries_[it->second].state == State::Resident;
}

size_t HostArena::entry_bytes()    const { std::lock_guard<std::mutex> lock(mu_); return cfg_.entry_bytes; }
size_t HostArena::entry_count()    const { std::lock_guard<std::mutex> lock(mu_); return entries_.size(); }
size_t HostArena::resident_count() const { std::lock_guard<std::mutex> lock(mu_); return resident_count_; }
size_t HostArena::resident_bytes() const { std::lock_guard<std::mutex> lock(mu_); return resident_bytes_; }
size_t HostArena::pinned_bytes()   const { std::lock_guard<std::mutex> lock(mu_); return pinned_bytes_; }
size_t HostArena::spec_bytes()     const { std::lock_guard<std::mutex> lock(mu_); return spec_bytes_; }
size_t HostArena::reading_count()  const { std::lock_guard<std::mutex> lock(mu_); return reading_count_; }
size_t HostArena::chunk_count()    const { std::lock_guard<std::mutex> lock(mu_); return chunks_.size(); }

uint64_t HostArena::evictions()           const { std::lock_guard<std::mutex> lock(mu_); return evictions_; }
uint64_t HostArena::spec_evicted_unused() const { std::lock_guard<std::mutex> lock(mu_); return spec_evicted_unused_; }
uint64_t HostArena::spec_promotions()     const { std::lock_guard<std::mutex> lock(mu_); return spec_promotions_; }
uint64_t HostArena::begin_read_refusals() const { std::lock_guard<std::mutex> lock(mu_); return begin_read_refusals_; }

}  // namespace wp
