#include "wp-host-arena.h"

#include <algorithm>
#include <cassert>
#include <chrono>

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

    if (cfg_.freq_admission && initialized_) {
        sketch_.assign((size_t) kSketchRows * std::max<size_t>(cfg_.sketch_width, 1), 0);
        // Aging period: halve every counter after this many record() calls,
        // same shape as the WP_EXPERT_LFU_HALFLIFE / doorkeeper_lru sim
        // knob -- scaled off entries_.size() (the tier's own turnover rate)
        // rather than sketch width, so a bigger sketch doesn't make aging
        // slower. Floor of 4096 keeps a tiny test arena from aging on
        // nearly every record.
        sketch_period_ = std::max<uint64_t>(8 * entries_.size(), 4096);
    }

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
    admission_cold_landed_ = 0;
    lookups_       = 0;
    lookup_hits_   = 0;
    sketch_.clear();
    sketch_ops_    = 0;
    sketch_period_ = 0;
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
    // Tier entries only: the read_inflight_max in-flight entries are never
    // pinnable (see pin() in the header).
    const size_t tier_entries = cfg_.entry_bytes == 0 ? 0 : cfg_.tier_bytes / cfg_.entry_bytes;
    return std::min(tier_entries, entries_.size()) * (size_t) cfg_.pinned_cap_pct / 100;
}

// --- frequency sketch (freq_admission only) -------------------------------

namespace {
// splitmix64 finalizer -- deterministic, no per-process seed (unlike
// std::hash<std::string>), so a captured trace replays identically run to
// run. page_idx is small (a catalog page index) so it is folded into the
// state rather than used as a seed on its own.
inline uint64_t mix64_(uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}
}  // namespace

size_t HostArena::sketch_lane_locked_(int page_idx, int row) const {
    const uint64_t h = mix64_((uint64_t) (uint32_t) page_idx * 0x9E3779B97F4A7C15ULL +
                               (uint64_t) row * 0xBF58476D1CE4E5B9ULL);
    return (size_t) row * cfg_.sketch_width + (h % cfg_.sketch_width);
}

void HostArena::sketch_record_locked_(int page_idx) {
    if (sketch_.empty()) return;   // freq_admission off, or init() never sized it
    if (++sketch_ops_ % sketch_period_ == 0) {
        // Aging: halve every counter so a page popular many sweeps ago
        // cannot out-bid current traffic forever. O(kSketchRows *
        // sketch_width) but amortized over sketch_period_ real records, so
        // this stays a small constant per record on average.
        for (uint8_t & c : sketch_) c >>= 1;
    }
    for (int r = 0; r < kSketchRows; ++r) {
        uint8_t & c = sketch_[sketch_lane_locked_(page_idx, r)];
        if (c < 255) ++c;
    }
}

uint32_t HostArena::sketch_estimate_locked_(int page_idx) const {
    if (sketch_.empty()) return 0;
    uint32_t est = 255;
    for (int r = 0; r < kSketchRows; ++r) {
        est = std::min<uint32_t>(est, sketch_[sketch_lane_locked_(page_idx, r)]);
    }
    return est;
}

// *** WHY PLAIN LRU GETS ~0 HITS HERE, AND WHAT THIS BUYS. ***
// A cyclic prefill sweep touches every non-resident page exactly once per
// pass, in the same order every pass. When the sweep is bigger than the
// tier, LRU (and FIFO, and any "admit everyone, evict oldest" policy) evicts
// every page before its next reference: hit rate 0, byte for byte the
// symptom this exists to fix. The fix needs memory that OUTLIVES eviction --
// an evicted Entry forgets everything (state resets to Free) -- which is
// exactly what the sketch above is: counts keyed by page_idx, independent of
// whether the page currently holds an arena slot.
//
// *** WHY THE GATE ONLY APPLIES TO PREFILL LANDINGS (prefill_hint). ***
// An earlier version of this gated every demand landing, prefill or decode.
// docs/dev/sim-host-tier.py's freq_admit() vs freq_admit_phase() trials
// measured that ungating decode regressed decode's OWN hit rate 40-55%
// relative to plain LRU at realistic tier sizes -- decode's per-token
// traffic already has real, LRU-friendly short-term reuse, and a frequency
// gate can only ever slow down how fast the cache admits and tracks that
// (a decode page starts at estimate 0 same as any other newcomer, so gating
// it costs real hits for no scan-resistance benefit: decode was never the
// phase producing one-shot scans). Prefill is: every prefill page-in is, by
// construction, part of a sweep that touches each page exactly once, so
// there is nothing to lose by gating it, and gating it is what stops it
// from trashing whatever decode -- or an earlier prefill pass -- already
// proved is worth keeping.
//
// admit_landed_locked_ is the one place that memory gets used: a freshly
// landed DEMAND page (never a speculative one -- those keep the pre-existing
// prefetch-protection ordering untouched) is placed at the cold (LRU-front,
// next-to-evict) end instead of the usual MRU end when (a) landing it will
// require a trim, (b) that trim's victim would come from the demand list
// (spec_lru_ has nothing left to give up -- if it does, the incoming page
// isn't really competing with anything, so let it land hot as before), and
// (c) the incoming page's sketch estimate is NOT STRICTLY GREATER than the
// victim's -- a tie goes to the INCUMBENT.
//
// Deliberately NOT recorded here: landing a page does not, by itself, bump
// its own sketch count (only borrow() -- an actual repeat reference -- does,
// see below). A fresh miss is exactly as uninformative about future demand
// as the LRU victim it is displacing; recording on landing would let a page
// win purely for having been read a SECOND calendar time (once per sweep
// pass, same as literally every other swept page), which defeats the
// tie-break below and degenerates back to plain LRU. With landing
// unrecorded, a page's estimate is 0 until it earns a real hit, so on a
// pure cyclic sweep (every page touched exactly once per pass, nothing ever
// resident long enough to be hit twice) every comparison is a 0-vs-0 tie
// forever -- and tie-favors-incumbent is what makes that converge: whichever
// ~tier_size pages happen to be resident when the tier first fills keep
// winning every subsequent tie (a challenger can only unseat them by
// scoring STRICTLY higher, which requires a real hit, which a page that
// self-evicts before anyone can reference it again can never get). Hit rate
// settles at tier_size / sweep_size once locked in, instead of the 0% plain
// LRU gets on the same trace, and it locks in even without genuine
// popularity differences.
//
// The starvation this implies -- a challenger that loses its first tie
// self-evicts before it can ever earn the hit that would let it win next
// time, so a locked-in resident set can in principle hold forever even past
// its actual popularity -- is now confined to PREFILL-vs-PREFILL contention
// only (decode never lands through the gate at all, see below), which is
// the scan-resistance behavior this exists to provide, not a bug: a prefill
// sweep page losing forever to whatever else earned real hits (from decode,
// or from surviving an earlier prefill pass) is the intended outcome. A
// full W-TinyLFU windowed admit (the W segment gives every newcomer a
// bounded number of real chances before facing the gate) would still be a
// strictly more general fix and is a reasonable next step; not built here
// because the phase split above already closes the one case
// (docs/dev/sim-host-tier.py's freq_admit vs freq_admit_phase measurements)
// where the simpler gate actually regressed something.
void HostArena::admit_landed_locked_(size_t idx, bool prefill_hint) {
    Entry & e = entries_[idx];
    bool land_cold = false;
    if (cfg_.freq_admission && !e.speculative && prefill_hint) {
        const bool would_trim = resident_bytes_ - pinned_bytes_ > cfg_.tier_bytes;
        if (would_trim && spec_lru_.empty() && !lru_.empty()) {
            const size_t victim_idx = lru_.front();
            const uint32_t cand_f  = sketch_estimate_locked_(e.page_idx);
            const uint32_t vict_f  = sketch_estimate_locked_(entries_[victim_idx].page_idx);
            land_cold = cand_f <= vict_f;
        }
    }
    insert_mru_locked_(idx, e.speculative);
    if (land_cold) {
        remove_from_list_locked_(idx);
        lru_.push_front(idx);
        entries_[idx].lru_pos = lru_.begin();
        entries_[idx].loc     = ListLoc::Lru;
        ++admission_cold_landed_;
    }
}

// Evict a single entry, preferring a speculative victim (a misprediction
// must never displace a page the caller actually demanded). Scans the
// applicable list from the front, skipping a borrowed entry IN PLACE
// (neither removed nor reordered -- a later attempt sees it in the same
// spot), and evicts the first unborrowed entry found. SpecOnly refuses
// outright if spec_lru_ has no unborrowed entry rather than falling back to
// a demand victim, because a full spec budget must be relieved from the
// spec side, never the demand side.
bool HostArena::evict_one_locked_(EvictScope scope) {
    size_t idx    = 0;
    bool   found  = false;
    bool   from_spec = false;

    for (size_t cand : spec_lru_) {
        if (entries_[cand].borrows == 0) {
            idx = cand; found = true; from_spec = true; break;
        }
    }
    if (!found) {
        if (scope == EvictScope::SpecOnly) return false;
        for (size_t cand : lru_) {
            if (entries_[cand].borrows == 0) {
                idx = cand; found = true; from_spec = false; break;
            }
        }
    }
    if (!found) return false;

    Entry & e = entries_[idx];
    const int page_idx = e.page_idx;

    if (from_spec) {
        spec_lru_.erase(e.lru_pos);
        spec_bytes_ -= cfg_.entry_bytes;
        // Still speculative when evicted: only "unused" if a demand or
        // peek borrow() never touched it (see ever_borrowed).
        if (!e.ever_borrowed) ++spec_evicted_unused_;
    } else {
        lru_.erase(e.lru_pos);
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
    return begin_read_locked_(page_idx, speculative, data_out, handle_out);
}

bool HostArena::begin_read_wait(int page_idx, bool speculative, void ** data_out,
                                Handle * handle_out, uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    if (begin_read_locked_(page_idx, speculative, data_out, handle_out)) {
        return true;
    }
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            // One last try right at the deadline -- a release() that fired
            // just before the wait timed out must not be wasted.
            return begin_read_locked_(page_idx, speculative, data_out, handle_out);
        }
        if (begin_read_locked_(page_idx, speculative, data_out, handle_out)) {
            return true;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
}

HostArena::Reserve HostArena::reserve_wait(int page_idx, bool speculative, void ** data_out,
                                           Handle * handle_out, uint64_t timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (true) {
        auto it = by_page_.find(page_idx);
        if (it != by_page_.end()) {
            if (entries_[it->second].state == State::Resident) {
                touch_locked_(it->second);
                return Reserve::Present;
            }
            // Reading by someone else: wait for its finish_read (either
            // outcome notifies cv_), then re-classify.
        } else if (begin_read_locked_(page_idx, speculative, data_out, handle_out)) {
            return Reserve::Reserved;
        }
        if (std::chrono::steady_clock::now() >= deadline ||
                cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            // One last look right at the deadline.
            it = by_page_.find(page_idx);
            if (it != by_page_.end() && entries_[it->second].state == State::Resident) {
                touch_locked_(it->second);
                return Reserve::Present;
            }
            if (it == by_page_.end() &&
                    begin_read_locked_(page_idx, speculative, data_out, handle_out)) {
                return Reserve::Reserved;
            }
            return Reserve::Timeout;
        }
    }
}

bool HostArena::begin_read_locked_(int page_idx, bool speculative, void ** data_out, Handle * handle_out) {
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
        } else if (evict_one_locked_(
                       cfg_.freq_admission ? EvictScope::SpecOnly : EvictScope::Any)) {
            // *** THE BUG THE LIVE 2026-09-25 RUN FOUND: n_host_hit=0 EVEN
            // WITH A TIER SIZED TO HOLD THE WHOLE WORKING SET. ***
            // admit_landed_locked_ only gates WHERE a demand page LANDS
            // (finish_read). It says nothing about THIS call -- a
            // SPECULATIVE (layer-ahead/WP_EXPERT_SPEC_PAGEIN) begin_read
            // reserving a fresh entry. Before this fix, when spec_lru_ was
            // under its cap but free_ was empty (arena at its budget, which
            // it reaches fast once demand fills the tier), this fell back to
            // EvictScope::Any -- and Any evicts from lru_ (the DEMAND list)
            // whenever spec_lru_ happens to have nothing evictable at that
            // exact instant. Under continuous prefetch
            // (WP_PREFILL_LAYER_AHEAD=1 + WP_EXPERT_SPEC_PAGEIN=1, the live
            // production config, NOT exercised by any test in this file
            // before this one) spec_lru_ churns constantly and routinely
            // empties for a moment, so an UNCONFIRMED guess kept evicting a
            // CONFIRMED, possibly-about-to-be-reused demand page --
            // completely bypassing the frequency gate, which only ever sees
            // finish_read's landing decision. That is exactly consistent
            // with the live symptom: ram_evictions far exceeding the
            // distinct page count (speculative churn, not genuine demand
            // turnover) and n_host_hit staying at 0 regardless of tier size.
            // Fix: under freq_admission, a speculative reservation must
            // relieve ONLY the speculative side (SpecOnly) -- if spec has
            // nothing evictable either, refuse (the existing
            // begin_read_refusals_ / "advisory, never fails the worker"
            // contract every other speculative-path failure already uses)
            // rather than stealing a demand page to seat a guess. Gated on
            // cfg_.freq_admission so default (policy unset) behaviour is
            // byte-for-byte unchanged.
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

    Entry & e     = entries_[idx];
    e.page_idx    = page_idx;
    e.state       = State::Reading;
    e.borrows     = 0;
    e.speculative = speculative;
    e.pinned      = false;
    e.ever_borrowed = false;
    e.gen         = next_gen_++;
    e.loc         = ListLoc::None;

    by_page_[page_idx] = idx;
    ++reading_count_;

    *data_out   = e.data;
    *handle_out = e.gen;
    return true;
}

void HostArena::finish_read(int page_idx, Handle handle, bool ok, bool keep_borrowed,
                            bool prefill_hint) {
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
        admit_landed_locked_(idx, prefill_hint);
        if (keep_borrowed) {
            e.borrows       = 1;
            e.ever_borrowed = true;
        }
        // Trim regardless of keep_borrowed: a held entry cannot itself be
        // evicted (evict_one_locked_ skips borrowed entries), but landing it
        // may have pushed the OTHER unborrowed/unpinned resident bytes over
        // the cap, and those must still come down in this same locked call.
        trim_to_tier_cap_locked_();
    } else {
        // Reading -> Free: discard the bytes, the page never landed.
        by_page_.erase(it);
        e.state    = State::Free;
        e.page_idx = -1;
        e.gen      = kInvalidHandle;
        free_.push_back(idx);
    }
    // Either branch can free capacity (a new Free entry, or trim_to_tier_cap_
    // above evicting something else) that begin_read_wait() is blocked on.
    cv_.notify_all();
}

// --- hit path ---------------------------------------------------------

bool HostArena::borrow(int page_idx, const void ** src_out, Handle * handle_out, bool demand) {
    std::lock_guard<std::mutex> lock(mu_);
    ++lookups_;   // unconditional: proves the lookup path is even reached (see .h comment)
    auto it = by_page_.find(page_idx);
    if (it == by_page_.end()) return false;

    size_t idx = it->second;
    Entry & e  = entries_[idx];
    if (e.state != State::Resident) return false;   // miss or Reading
    ++lookup_hits_;

    ++e.borrows;
    e.ever_borrowed = true;   // any borrow, demand or peek, counts as "used"

    if (cfg_.freq_admission && demand) {
        // Every real demand hit grows the page's persistent frequency --
        // this (not the per-Entry LRU touch, which is forgotten on
        // eviction) is what lets a hot decode expert keep winning admission
        // ties after being evicted and re-read. Speculative peeks
        // (demand=false) do not count: an unconfirmed prefetch guess is not
        // yet real popularity.
        sketch_record_locked_(page_idx);
    }

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
    trim_to_tier_cap_locked_();
    cv_.notify_all();
}

// --- retention cap -------------------------------------------------------

// Caller holds mu_. Evicts LRU (speculative side first, via evict_one_locked_'s
// own scan order) until Resident-unpinned bytes are at or under tier_bytes, or
// nothing more can be evicted (everything left is borrowed or pinned). A
// borrowed entry that is itself over the cap is left alone -- it comes down
// on its own next release()/finish_read() call, once it is actually
// evictable.
void HostArena::trim_to_tier_cap_locked_() {
    while (resident_bytes_ - pinned_bytes_ > cfg_.tier_bytes) {
        if (!evict_one_locked_(EvictScope::Any)) break;
    }
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
    if (e.speculative) {
        // A pinned page is a demand page by definition: pinning it is a
        // promotion out of the speculative side, same accounting as a
        // demand borrow() hit.
        e.speculative = false;
        spec_bytes_ -= cfg_.entry_bytes;
        ++spec_promotions_;
    }
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
    // The entry just became evictable: a reserve_wait()/begin_read_wait()
    // parked on "nothing evictable" must re-check.
    cv_.notify_all();
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
size_t HostArena::tier_bytes()     const { std::lock_guard<std::mutex> lock(mu_); return cfg_.tier_bytes; }
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
uint64_t HostArena::admission_cold_landed() const { std::lock_guard<std::mutex> lock(mu_); return admission_cold_landed_; }
uint64_t HostArena::lookups()     const { std::lock_guard<std::mutex> lock(mu_); return lookups_; }
uint64_t HostArena::lookup_hits() const { std::lock_guard<std::mutex> lock(mu_); return lookup_hits_; }

}  // namespace wp
