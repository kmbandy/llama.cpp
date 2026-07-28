# Phase 1B adversarial review — Codex (task-mrz2kqmz), 2026-07-24

**Verdict: NEEDS A FIX FIRST. Three BLOCKERs.** (Codex ran read-only; it could not write this file itself —
transcribed from its task-complete message. BLOCKER 1 independently verified at source by claude__main.)

## BLOCKER 1 — Claim with stale STI (VERIFIED by claude__main)
`SL_RBNEXT=0` is written **before** `SL_STI`/`SL_GEN` in the grow-fail STAMP (occ_kernel_dsws_flow.s:4717
vs SL_STI ~:4740, SL_GEN ~:4750). The claim path (:3577-3599) is POISON-UNTIL-STAGED: it gates only on
`RB_PENDING` (SL_RBNEXT), **never validates `SL_GEN`** (:3577), and its ABA-safety comment (:3582-3583)
explicitly relies on the invariant "reuse re-writes RB_PENDING FIRST ⟹ RB_PENDING-clear means fully staged
+ STI valid." Writing `SL_RBNEXT=0` (claimable) before `SL_STI` breaks that invariant: a compute wave can
observe advanced `STAGE_HEAD` + a claimable counter and read the PRIOR occupant's STI → self-load wrong
(mblk,tcol,ksi) operands → silent wrong C. Same missing-cross-wave-ordering class as the INITBAR
dropped-group race (DSWS_TESTING_LOG.md:1297).
FIX DIRECTION (builder decides): make claimability the release fence — write `SL_RBNEXT=0` (claimable)
LAST, after `SL_STI`/`SL_GEN`, so "claimable ⟹ everything published"; OR add `SL_GEN==expected`
validation to the claim path before trusting SL_STI.

## BLOCKER 2 — Premature drain on slot reuse
After slot reuse a wave can observe the NEW `SL_GEN` but a STALE prior-generation `SL_RBDONE=ACC_N`;
`drain_advance` (:1303) then skips an uncomputed item. The sentinel walkers cited as precedent
(.Lflow_da_cf_sentinel_stage_walk / .Lflow_da_ss_stage_walk) do NOT publish claimable payloads for another
wave, so they do not prove this protocol — the grow-fail item is real work (SL_RBDONE=0), not a
pre-completed sentinel.

## BLOCKER 3 — No ring consumer after RCONV (liveness)
Repeated coasts can convert EVERY compute mailbox permanently to `ROLE_AFEED` (:5167). Under DSWS2_OVERLAP,
feed roles jump straight to assignment and never claim the ring (:4135, the neutered `.Lflow_feed`). The
first later grow-fail publishes a ring item that no wave can consume → `DRAIN_HEAD` permanently blocked.
(This refutes the "priority ladder drains the backlog" liveness argument: if no wave remains in the
ring-claiming role, the backlog has no consumer.)

## NOT falsifiable (confirmed correct)
Self-load address algebra: B addressing matches the primary path; A uses `mblk*G + group*ACC_N +
local_rowblk`; the `GROUPS==1` `s41` hazard is correctly avoided (:3677). This part is sound.

## Path forward
Hand these three to the builder to fix (Phase 1B rev 2), then re-review. Do NOT proceed to Phase 2 on the
current Phase 1B. grow-fail=0 means none of this is silicon-testable until the full fix binds the budget —
so the static review IS the gate here, and it failed.
