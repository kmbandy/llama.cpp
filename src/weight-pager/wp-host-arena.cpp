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
    reject_lru_.clear();

    resident_count_       = 0;
    resident_bytes_       = 0;
    pinned_bytes_         = 0;
    spec_bytes_           = 0;
    reading_count_        = 0;
    evictions_            = 0;
    evictions_spec_       = 0;
    evictions_reject_     = 0;
    evictions_lru_        = 0;
    spec_evicted_unused_  = 0;
    spec_promotions_      = 0;
    reject_promotions_    = 0;
    spec_promotions_rejected_ = 0;
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
    if (e.loc == ListLoc::Lru)           lru_.erase(e.lru_pos);
    else if (e.loc == ListLoc::SpecLru)  spec_lru_.erase(e.lru_pos);
    else if (e.loc == ListLoc::RejectLru) reject_lru_.erase(e.lru_pos);
    e.loc = ListLoc::None;
}

void HostArena::insert_mru_locked_(size_t idx, ListLoc loc) {
    Entry & e = entries_[idx];
    std::list<size_t> * list = loc == ListLoc::SpecLru   ? &spec_lru_
                              : loc == ListLoc::RejectLru ? &reject_lru_
                                                           : &lru_;
    list->push_back(idx);
    e.lru_pos = std::prev(list->end());
    e.loc     = loc;
}

void HostArena::touch_locked_(size_t idx) {
    Entry & e = entries_[idx];
    if (e.loc == ListLoc::None) return;   // pinned (or not resident) -- nothing to touch
    const ListLoc loc = e.loc;   // stays in the same list -- touch is a recency bump, not a promotion
    remove_from_list_locked_(idx);
    insert_mru_locked_(idx, loc);
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
// prefetch-protection ordering untouched) is placed in reject_lru_ instead
// of lru_ when (a) landing it will require a trim, (b) spec_lru_ has
// nothing left to give up (if it does, the incoming page isn't really
// competing with anything that matters, so let it land in lru_ as normal;
// spec_lru_ absorbs the trim instead), and (c) the incoming page's sketch
// estimate is NOT STRICTLY GREATER than the WEAKEST currently resident
// page's -- reject_lru_'s own front if it has anything, else lru_'s front
// -- a tie goes to the INCUMBENT.
//
// *** 2026-09-25 REVISION: record on EVERY demand access, hit or miss. ***
// The original version of this policy (commit 2ccf93034) deliberately did
// NOT bump a page's sketch count on landing -- only borrow() (an actual
// repeat reference) did. That was analyzed to converge to tier_size /
// sweep_size hit rate on a cyclic sweep via tie-favors-incumbent. A live run
// (2026-09-25, two back-to-back 24.5k-token prefills, main tier 24G/~1300
// pages of a ~10k-page sweep) instead measured ram_evictions almost exactly
// equal to ram_lookups -- i.e. a real cache-resident page was being evicted
// on nearly every single page-in, not just during the cyclic churn among
// losing candidates. The cause was NOT the sketch's recording rule: it was
// that begin_read_locked_'s reservation (the call that actually vacates a
// slot for the incoming read) had no visibility into admission at all --
// admit_landed_locked_ only ever decided WHERE a page that had ALREADY been
// read landed, after evict_one_locked_(Any) had already evicted whatever sat
// at lru_.front() to make room for it. A losing candidate landing "cold" at
// lru_.front() (the old design) was consequently always the very next
// eviction victim, so the front of lru_ was, in steady state, permanently
// occupied by the MOST RECENTLY rejected page -- meaning every subsequent
// miss's reservation evicted a page that had ITSELF just displaced a real
// resident one page-in ago, not a stable incumbent. reject_lru_ (this
// revision) fixes the mechanism, not just the bookkeeping: a rejected
// landing now goes to its OWN list, which evict_one_locked_ drains ahead of
// lru_ (see the EvictScope::Any ordering in the .h), so the reservation for
// the NEXT page-in takes its victim from reject_lru_ instead of lru_ as long
// as reject_lru_ has anything in it -- and it always will, in steady state,
// because one rejection happens on almost every non-admitted landing. Real
// lru_ residents stop being touched by this churn entirely once the first
// rejection has landed.
//
// Recording on every access (not just hits) is what makes the comparison
// meaningful under that fix: with reservation-time eviction no longer
// clobbering lru_, a losing candidate's estimate must still be able to grow
// via its own repeated MISSES (once per sweep pass) so that it can be
// distinguished from a page that is winning purely by sweep order rather
// than genuine reuse. On a pure cyclic sweep with no decode traffic, a
// resident incumbent's estimate grows once per pass from decode/demand HITS
// and a challenger's grows once per pass from its own MISSES -- the same
// rate -- so ties still persist and tie-favors-incumbent still converges to
// tier_size / sweep_size, exactly as before, but now with lru_evictions()
// approaching 0 instead of tracking ram_lookups 1:1. Where this recording
// change actually matters is decode: a page borrow()'d many times across a
// session accrues a real, strictly-greater estimate than a same-pass-once
// prefill sweep page, so it keeps winning admission ties even after being
// evicted and re-read -- see HostArena::sketch_record_locked_'s doc comment.
// Compare against the WEAKEST currently resident page: reject_lru_'s own
// front if it has anything (reject_lru_ entries are, by construction, no
// stronger than any lru_ entry -- comparing against them is still a real
// admission test, not a free pass), else lru_'s front; false (never loses)
// if both are empty -- nothing to compare against, so nothing to lose to.
// Earlier revision of this bug (admit_landed_locked_ only): requiring
// reject_lru_ to be EMPTY before comparing at all meant the gate stopped
// firing the moment the first page was ever rejected (which happens almost
// immediately and then stays true forever in one-in-one-out steady state --
// see evict_one_locked_'s reject-first preference), so every later
// candidate landed hot unconditionally: a simulated cyclic sweep measured
// 0% hit rate with that bug, not the tier_size/sweep_size convergence the
// design intends. Comparing against reject_lru_'s own front instead of
// skipping the comparison keeps the gate live regardless of reject_lru_'s
// occupancy.
bool HostArena::loses_admission_locked_(int page_idx) const {
    if (reject_lru_.empty() && lru_.empty()) return false;
    const size_t victim_idx = !reject_lru_.empty() ? reject_lru_.front() : lru_.front();
    const uint32_t cand_f  = sketch_estimate_locked_(page_idx);
    const uint32_t vict_f  = sketch_estimate_locked_(entries_[victim_idx].page_idx);
    return cand_f <= vict_f;
}

void HostArena::admit_landed_locked_(size_t idx, bool prefill_hint) {
    Entry & e = entries_[idx];
    bool land_cold = false;
    if (cfg_.freq_admission && !e.speculative && prefill_hint) {
        const bool would_trim = resident_bytes_ - pinned_bytes_ > cfg_.tier_bytes;
        if (would_trim && spec_lru_.empty()) {
            land_cold = loses_admission_locked_(e.page_idx);
        }
    }
    if (land_cold) {
        insert_mru_locked_(idx, ListLoc::RejectLru);
        ++admission_cold_landed_;
    } else {
        insert_mru_locked_(idx, e.speculative ? ListLoc::SpecLru : ListLoc::Lru);
    }
}

// Evict a single entry, preferring the cheapest victim first: spec_lru_
// (a misprediction must never displace a page the caller actually
// demanded), then reject_lru_ (a page admission already judged disposable
// -- see admit_landed_locked_ -- is a strictly better victim than a page
// that won admission), and lru_ only as the last resort. Scans the
// applicable list(s) from the front, skipping a borrowed entry IN PLACE
// (neither removed nor reordered -- a later attempt sees it in the same
// spot), and evicts the first unborrowed entry found in the
// highest-priority list that has one. SpecOnly/SpecOrReject refuse outright
// rather than falling back to lru_, because a speculative read must never
// steal room from a confirmed demand page just to seat a guess (see the
// .h EvictScope comment).
bool HostArena::evict_one_locked_(EvictScope scope) {
    size_t idx   = 0;
    bool   found = false;
    ListLoc from = ListLoc::None;

    for (size_t cand : spec_lru_) {
        if (entries_[cand].borrows == 0) { idx = cand; found = true; from = ListLoc::SpecLru; break; }
    }
    if (!found && scope != EvictScope::SpecOnly) {
        for (size_t cand : reject_lru_) {
            if (entries_[cand].borrows == 0) { idx = cand; found = true; from = ListLoc::RejectLru; break; }
        }
    }
    if (!found && scope == EvictScope::Any) {
        for (size_t cand : lru_) {
            if (entries_[cand].borrows == 0) { idx = cand; found = true; from = ListLoc::Lru; break; }
        }
    }
    if (!found) return false;

    Entry & e = entries_[idx];
    const int page_idx = e.page_idx;

    if (from == ListLoc::SpecLru) {
        spec_lru_.erase(e.lru_pos);
        spec_bytes_ -= cfg_.entry_bytes;
        // Still speculative when evicted: only "unused" if a demand or
        // peek borrow() never touched it (see ever_borrowed).
        if (!e.ever_borrowed) ++spec_evicted_unused_;
        ++evictions_spec_;
    } else if (from == ListLoc::RejectLru) {
        reject_lru_.erase(e.lru_pos);
        ++evictions_reject_;
    } else {
        lru_.erase(e.lru_pos);
        ++evictions_lru_;
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
                       cfg_.freq_admission ? EvictScope::SpecOrReject : EvictScope::Any)) {
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
            // relieve ONLY the speculative or reject side (SpecOrReject,
            // widened from the original SpecOnly once reject_lru_ existed --
            // a disposable reject_lru_ entry is no worse a victim for a
            // guess than another speculative one, and letting spec draw on
            // it instead of refusing outright means fewer begin_read_refusals_
            // under heavy layer-ahead prefetch) -- if BOTH have nothing
            // evictable, refuse (the existing begin_read_refusals_ /
            // "advisory, never fails the worker" contract every other
            // speculative-path failure already uses) rather than stealing a
            // demand page to seat a guess. Gated on cfg_.freq_admission so
            // default (policy unset) behaviour is byte-for-byte unchanged.
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

    // Record this demand access into the frequency sketch NOW, on the one
    // definitive success path (not per retry attempt -- reserve_wait/
    // begin_read_wait may call begin_read_locked_ several times for the
    // SAME logical page-in while capacity is refused, and recording on
    // every attempt would inflate a congested page's estimate for reasons
    // unrelated to genuine popularity). Paired with borrow()'s own
    // recording on every demand HIT, this is what makes "every demand
    // access, hit or miss" (see the 2026-09-25 revision comment above
    // admit_landed_locked_) actually every access exactly once: a page
    // either hits (borrow() records) or misses and reserves here (this
    // records) -- never both for the same access.
    if (cfg_.freq_admission && !speculative) {
        sketch_record_locked_(page_idx);
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

bool HostArena::borrow(int page_idx, const void ** src_out, Handle * handle_out, bool demand,
                       bool prefill_hint) {
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
            // speculative side into the demand side. 2026-09-25: THIS is the
            // promotion path a live run found unguarded -- under
            // freq_admission, a prefill-hinted promotion is gated exactly
            // like a fresh demand landing (admit_landed_locked_), using the
            // SAME loses_admission_locked_ comparison, because on live
            // traffic dominated by layer-ahead speculative reads this path
            // (not admit_landed_locked_'s finish_read call, which a
            // speculative landing never even passes prefill_hint=true to)
            // is where nearly every page actually enters demand standing.
            // Leaving it ungated made the whole policy a no-op: every
            // promoted page landed hot regardless of frequency, degenerating
            // to plain LRU. No would_trim precondition here (unlike
            // admit_landed_locked_): promoting doesn't add resident bytes
            // (the page is already Resident, just relabeled), so there is no
            // "would this landing need a trim" question -- only "does this
            // page deserve genuine lru_ standing", which is always worth
            // asking once the tier is at its cap. Deliberately >= here, NOT
            // > like admit_landed_locked_'s would_trim: that check runs
            // AFTER resident_bytes_ already counts the new landing, so ==
            // there means "fit with room to spare, nothing to gate". Here
            // resident_bytes_ does NOT change (the page is already
            // Resident), and trim_to_tier_cap_locked_ keeps the tier at
            // EXACTLY tier_bytes in steady state between separate calls (it
            // always restores <=, never leaves capacity transiently unspent)
            // -- so a strict > would almost never be true when borrow() is
            // later called on live traffic, silently reproducing the same
            // "gate looks right but never fires" failure this whole promotion
            // path exists to fix. >= is what actually engages at the
            // steady-state cap.
            const bool tier_full = resident_bytes_ - pinned_bytes_ >= cfg_.tier_bytes;
            const bool reject_promotion =
                cfg_.freq_admission && prefill_hint && tier_full &&
                loses_admission_locked_(page_idx);
            remove_from_list_locked_(idx);
            e.speculative = false;
            spec_bytes_ -= cfg_.entry_bytes;
            ++spec_promotions_;
            if (reject_promotion) {
                insert_mru_locked_(idx, ListLoc::RejectLru);
                ++spec_promotions_rejected_;
            } else {
                insert_mru_locked_(idx, ListLoc::Lru);
            }
        } else if (demand && e.loc == ListLoc::RejectLru) {
            // A demand hit on a page admission had judged disposable: it
            // just proved itself worth keeping after all, same promotion
            // shape as the speculative case above.
            remove_from_list_locked_(idx);
            ++reject_promotions_;
            insert_mru_locked_(idx, ListLoc::Lru);
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
        // Never back to reject_lru_: unpinning is an explicit decision that
        // this page still matters, the same standing a promoted page gets.
        insert_mru_locked_(idx, e.speculative ? ListLoc::SpecLru : ListLoc::Lru);
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
uint64_t HostArena::evictions_spec()   const { std::lock_guard<std::mutex> lock(mu_); return evictions_spec_; }
uint64_t HostArena::evictions_reject() const { std::lock_guard<std::mutex> lock(mu_); return evictions_reject_; }
uint64_t HostArena::evictions_lru()    const { std::lock_guard<std::mutex> lock(mu_); return evictions_lru_; }
uint64_t HostArena::reject_promotions() const { std::lock_guard<std::mutex> lock(mu_); return reject_promotions_; }
uint64_t HostArena::spec_promotions_rejected() const { std::lock_guard<std::mutex> lock(mu_); return spec_promotions_rejected_; }
uint64_t HostArena::lookups()     const { std::lock_guard<std::mutex> lock(mu_); return lookups_; }
uint64_t HostArena::lookup_hits() const { std::lock_guard<std::mutex> lock(mu_); return lookup_hits_; }

}  // namespace wp
