// DSWS control-law reference model (SPEC_DSWS_CONTROLLER.md). Pure host functions that define the
// EXACT semantics the Phase-3 gfx1201 asm transcribes. Kept header-light (included directly by the
// test and, later, mirrored by hand-asm) so there is one source of truth for the protocol.
//
//   watermark_decision  -> the boundary band check (sensing -> action sign)
//   epoch_of            -> the per-WG decision clock E = segments_processed >> EPOCH_SHIFT
//   gate_try_win        -> the lock-free single-winner ticket  (asm: ds_cmpst_b32 on gate[dir])
//   reserve_grow        -> the sum-envelope reservation         (asm: atomic_add/sub on vgpr_reserved)
#pragma once
#include <atomic>
#include <cstdint>

// Sensing -> action sign for ring X. Bands are STRICT (edges are dead-zone) so LOW==HIGH degenerates
// to "act only when strictly past the edge", and LOW<value<HIGH is always the no-op dead-zone.
//   occ < low  -> +1  (ring draining empty: the consumer is STARVED for X -> wants more feed-X)
//   occ > high -> -1  (ring backing up full: feed-X is OVER-SERVING -> a feed-X wave can leave)
//   else       ->  0  (dead-zone: hold)
static inline int watermark_decision(uint32_t occ, uint32_t low, uint32_t high) {
    if (occ < low)  return +1;
    if (occ > high) return -1;
    return 0;
}

// Per-workgroup decision clock. Ticks every 2^shift segments of progress; no one "advances" it, it is
// purely a function of throughput. shift is the cadence knob (small=reactive, large=damped).
static inline uint32_t epoch_of(uint32_t segments_processed, uint32_t shift) {
    return segments_processed >> shift;
}

// Lock-free single-winner ticket for one conversion direction at epoch E. gate holds the last epoch in
// which this direction fired. Among many waves racing the same (g < E), exactly one CAS succeeds; the
// rest observe g advanced and back off. Guarantees <=1 conversion per direction per epoch, per WG.
//   asm: g = ds_read gate[dir]; if g>=E -> lose; else old = ds_cmpst_b32(gate[dir], g, E); win iff old==g
static inline bool gate_try_win(std::atomic<uint32_t>& gate, uint32_t E) {
    uint32_t g = gate.load(std::memory_order_relaxed);
    if (g >= E) return false;                         // direction already fired this (or a later) epoch
    // compare_exchange_strong updates `g` to the observed value on failure -> mirrors ds_cmpst's
    // returned-old semantics; a single retry loop is unnecessary because a failed CAS means someone
    // else won this epoch (g becomes >=E), so we simply lose.
    return gate.compare_exchange_strong(g, E, std::memory_order_acq_rel, std::memory_order_relaxed);
}

// Sum-envelope reservation for a feed->compute GROW. Reserve first (atomic_add), then validate against
// budget; if the reservation would blow the envelope, undo it (atomic_sub) and reject. The atomic
// serializes concurrent grows: the second to validate sees the first's reservation and backs off.
//   asm: r = atomic_add(vgpr_reserved, delta); if (r+delta) > budget -> atomic_sub(delta), abort
static inline bool reserve_grow(std::atomic<uint32_t>& resv, uint32_t delta, uint32_t budget) {
    uint32_t prev = resv.fetch_add(delta, std::memory_order_acq_rel);
    if (prev + delta > budget) {
        resv.fetch_sub(delta, std::memory_order_acq_rel);  // over-budget: cleanly undo, stay current role
        return false;
    }
    return true;
}

// Compute-burst reserve with spin-retry (models .Lcompute_reserve): reserve +delta against the
// sum-envelope; on over-budget, reserve_grow has already undone its add, so back off and retry.
// `spins` accumulates the backoff count (permit-starvation depth). Bounded when >=1 peak fits.
static inline void reserve_spin(std::atomic<uint32_t>& resv, uint32_t delta,
                                uint32_t budget, uint64_t& spins) {
    while (!reserve_grow(resv, delta, budget)) ++spins;
}
// Release a booked burst (models the post-shrink lds_fetch_add VRESV_OFF, -delta). Never fails.
static inline void reserve_release(std::atomic<uint32_t>& resv, uint32_t delta) {
    resv.fetch_sub(delta, std::memory_order_acq_rel);
}

struct WgSnap { uint32_t nC, nA, nB; };

static inline WgSnap snapshot_counts(uint32_t nC, uint32_t nA, uint32_t nB) {
    return WgSnap{nC, nA, nB};
}

// Sentinels = work-threshold + snapshot role-count terminal bails (Phase A arithmetic,
// with compile-time constants replaced by the per-epoch snapshot).
static inline bool quiesce_ready(uint32_t rowblk_next, uint32_t bfrag_next,
                                 uint32_t arow_next, const WgSnap& s,
                                 uint32_t G, uint32_t FN) {
    return rowblk_next >= (G  + s.nC)
        && bfrag_next  >= (FN + s.nB)
        && arow_next   >= (G  + s.nA);
}

// Role-agnostic safety net: fixed N waves, wid0 claimer never bails -> exactly N-1 bails.
static inline bool quiesce_ready_nm1(uint32_t quiesce_cnt, uint32_t N) {
    return quiesce_cnt >= (N - 1);
}

// ---- Task 1: dispatch, cooldown, pool invariants ----

enum Role { COMPUTE, AFEED, BFEED };

inline Role role_dispatch(uint32_t slot_id) {
    return slot_id == 24 ? COMPUTE : (slot_id == 28 ? AFEED : BFEED);
}

inline uint32_t cooldown_step(uint32_t cd) { return cd ? cd - 1 : 0; }

inline bool in_cooldown(uint32_t cd) { return cd > 0; }

inline bool pool_fits_lean(uint32_t n_pool, uint32_t vlean, uint32_t budget) {
    return (uint64_t)n_pool * vlean <= budget;
}

inline bool quiesce_ready_pool(uint32_t quiesce_cnt, uint32_t n_pool) {
    return quiesce_cnt >= n_pool - 1;
}

// ---- Pool-T7 root cause: first-entry contract (dispatch vs re-dispatch) ----
//
// A wave's FIRST-time entry and its RE-dispatch (after a per-super-tile bail) are NOT
// interchangeable. First entry must run, in order, the per-role _alloc and _init blocks:
//   (a) ran_alloc      -- s_alloc_vgpr 32, the DYNVGPR per-wave lean allocator handshake,
//   (b) waited_initflag -- spin until the claimer publishes INITFLAG == 0xACED (LDS ready),
//   (c) seeded_epoch   -- s35 = 0, the local last-seen-epoch baseline,
// THEN falls into _follow. Re-dispatch legitimately skips (a)/(b)/(c): the wave already ran
// them once and conv_apply already sized its footprint -- so the scalar-only .Ldispatch
// trampoline lands straight on _follow. The bug: the seed arms pointed FIRST entry at
// .Ldispatch too, so first-time followers skip _alloc/_init and desync from the epoch clock.
struct WaveEntry { bool ran_alloc; bool waited_initflag; bool seeded_epoch; };

// Where a wave's first-time entry lands. LAND_FOLLOW == seed arms branch to .Ldispatch (the
// buggy routing); LAND_ROLE_ENTRY == seed arms branch to .Lbfeed/.Lafeed/.Lcompute (the fix).
enum EntryLanding { LAND_FOLLOW, LAND_ROLE_ENTRY };

inline WaveEntry simulate_first_entry(EntryLanding land) {
    // Only the role entry labels chain _alloc -> _init -> _follow; landing on _follow skips both.
    bool full = (land == LAND_ROLE_ENTRY);
    return WaveEntry{full, full, full};
}

// A wave is correctly initialized for the follow/quiesce protocol iff it ran all three.
inline bool entry_safe(const WaveEntry& w) {
    return w.ran_alloc && w.waited_initflag && w.seeded_epoch;
}

// End-to-end handshake: does the claimer's per-super-tile QUIESCE_CNT >= WAVES-1 gate ever
// close? Every non-claimer follower (wid0 is the claimer and never bails) must reach its
// _quiesce bail once per super-tile; only a follower that entered safely is synced to the
// epoch clock and reliably does. An unsafe entry is a deterministic code-path defect (garbage
// s35 + skipped INITFLAG in identically-compiled waves), so ALL seed-entered followers desync
// -> zero reliable bumps -> the claimer spins forever in .Lclaimer_wait_done.
inline bool claimer_quiesce_converges(EntryLanding land, uint32_t waves) {
    uint32_t n_followers = waves - 1;
    uint32_t n_safe = entry_safe(simulate_first_entry(land)) ? n_followers : 0;
    return quiesce_ready_pool(n_safe, waves);
}
