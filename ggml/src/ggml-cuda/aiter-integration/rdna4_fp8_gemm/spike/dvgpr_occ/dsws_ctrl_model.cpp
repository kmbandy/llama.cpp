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
