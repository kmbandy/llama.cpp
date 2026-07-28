# ADVERSARIAL REVIEW — DSWS2 CF0 stack (2026-07-24, round 2)

Two INDEPENDENT reviewers, denied the design docs / builder progress docs / all claude__main conclusions.
Lenses: **R1 = liveness/work-conservation**, **R2 = cross-wave ordering/races** (R2 assembled AND
disassembled the exact profile — its claims are checked against emitted code).
Build: A1 profile + `CFASSIGN=0 DSWS2_OVERLAP=1 DSWS2_ROLEFLOW=1 DSWS2_PREFETCH=1 DSWS2_RCONV=1`.

> ## ⚠ PROFILE CORRECTION — 2026-07-25 (affects two findings below)
>
> The bin that ACTUALLY RAN ON SILICON is sha `85954d3c`, reproduced bit-for-bit on 2026-07-25 by exhaustive
> defsym sweep. Its complete profile is:
> `WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1 BANKZERO=1 STAGGER=1`
> `SELFSERVE=1 SSWIN=32 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1 CFASSIGN=0 DSWS2_OVERLAP=1`
> `DSWS2_ROLEFLOW=1 DSWS2_ROLEFLOW_BACK_N=2 DSWS2_PREFETCH=1 DSWS2_RCONV=1`  (.text 34068B, LDS 13824B)
>
> Two deltas from what the reviewers were given:
> 1. **`SSWIN=32`, not 8.** The `SLOT_N == SSWIN` slot-aliasing invariant recorded below was reasoned at
>    window 8. **It has NOT been audited at 32.** Re-confirm before relying on it.
> 2. **`DSWS2_ROLEFLOW_BACK_N=2`, not 16.** R2-C3's "the profile passed a stale 16" applies to the
>    REVIEWER'S build, not the silicon bin. Real reversion return cost is 64 passes, not 1024 — so the
>    "returns cost 16x more" sub-claim in R2-C3 does not describe what ran.
>
> Everything else in this review stands. The irony is noted: this is the same class of error as the round-1
> `FM=1` phantom blocker, in the opposite direction — an INCOMPLETE profile in the brief, again.

## VERDICT: no BLOCKERs. Correctness is sound. But one SAFETY REGRESSION must be reverted, and all three mechanisms are measurably ineffective.

---

## SAFETY REGRESSION — fix before any dispatch (R2-C1)

`deadman_progress` at the **grow-fail** site (`.Lflow_da_cf0_growfail`) **disarms the anti-brick watchdog
for the exact failure mode it exists to catch.** `deadman_check` re-stamps and clears `DM_PROG` whenever
it is set, so a wave reaching that site once per `DEADMAN_EVERY`(64) iterations can **never** be
force-retired, however long the WG is stalled.

Failing scenario: wave A wins a reservation, grows to NFV, stalls inside the burst. Waves B..Z loop
peek → `s_alloc_vgpr` fails (budget held by A) → mark progress → sleep → repeat. Zero forward progress,
and **no wave ever reaches `DEADMAN_TICKS`** → no clean retire → presents as a HUNG QUEUE instead of the
clean force-retire `CLAUDE.md` rule 3 depends on.

The **CAS-loss** site is legitimate and stays: that wave observed `s47 != s44`, i.e. it directly witnessed
the strictly-monotone `ASSIGN_HEAD` advance — real global progress. The grow-fail site observed *nothing*
advance; it only learned it could not grow. claude__main's brief conflated the two.

FIX: remove `deadman_progress` from the grow-fail path, or condition it on `ASSIGN_HEAD`/`DRAIN_HEAD`
actually differing from the previous tick's latched values.

## CONVERGENT: all three mechanisms are ineffective (both reviewers)

1. **Prefetch warms ~1.5% of its target** (R2-C4, R1-F5). The real B footprint for one `(tcol,ksi)` is
   `KSEG_STEPS(16) x FN(4) x 256B = 16KB`. The prefetch issues only the `ks=0, ni=0` line of four
   different `ksi` = 4x256B = 1KB, and the four are `s10` apart so they warm nothing adjacent. Worse, its
   mandatory `s_wait_loadcnt 0x0` blocks the coasting wave on full L2/HBM latency **immediately before**
   `.Lflow_feed_empty` — it trades a reserve attempt for a memory round trip. It also keeps firing through
   the entire terminal drain (the `FLOWTERM` test is downstream of the burst).
2. **The role economy collapses to all-AFEED within ~64 iterations, regardless of load** (R2-C3, R1-F1/F2).
   `.Lflow_havestage`'s `s_mov_b32 s50, 0` is now **structurally unreachable** (`.Lflow_compute`
   unconditionally branches to `.Lflow_coast`), so RCONV's coast counter free-runs and increments even on
   PRODUCTIVE passes. 27 of 30 waves start `ROLE_COMPUTE` and all convert within ~64 passes. Return costs
   16x more (`BACK_N=16` against a 1-in-64 throttled probe = 1024 passes back vs 64 out).
   => `CONVCNT` / `CNT_COAST` / `CNT_CNOSTG` are NO LONGER LOAD SIGNALS in this build. Do not read them as such.
   ALSO: the profile used `DSWS2_ROLEFLOW_BACK_N=16`, but the rebuild's own documented default is **2**
   (the probe became throttled). 16 is a stale value — claude__main's brief passed it.
3. **Role does not gate work at all** (R1-F1). Under OVERLAP, `.Lflow_feed`, `.Lflow_compute` and
   `.Lflow_coast` all funnel to `.Lflow_feed_empty`, and the reserve path never reads `ROLE[wid]`. An
   AFEED wave and a COMPUTE wave run byte-identical reservation logic. The reversion mechanism therefore
   buys nothing while costing 3 extra `lds_get` on the hottest shared words — the shape this file
   measured at **16x (97.3 -> 5.9 TF)**.
4. **Grow-before-CAS relocates the grow-fail manufacturing rather than removing it** (R2-C2, R1-F3). At the
   budget ceiling `(1536 - 30*32)/(80-32) = 12`, twelve waves can all pass the peek, all grow, all CAS —
   **eleven fat waves consume the entire SIMD dyn-VGPR budget to lose a CAS**, then each pays a
   `s_wait_storecnt` + shrink-spin round trip. A 13th wave peeking in that window gets grow-fail. Under
   the old commit-then-grow order only the CAS winner ever grew. The peek budget is 8, so ONE visit can
   execute up to 8 grow/shrink round-trips, and `s_alloc_vgpr` does `WaitIdleExceptStoreCnt()` — it drains
   the wave's whole pipeline **before it can even refuse**. Correctness fine; the throughput rationale in
   the header comment is not supported. (Also: the fat window is 16 emitted instructions, not the 6 claimed.)

## HOST BUG (R1-F4) — ✅ FIXED 2026-07-24

`entered = occ[96] - occ[73]` in `occ_dispatch.cpp` was structurally wrong and **could go negative**:
post-CF0 the two counters count disjoint events (occ[96] only after a won CAS AND a successful grow;
occ[73] only on a path that never emitted). The documented reading ("entered==shrunk on a clean run",
"entered==0 => never engaged") was inverted. The genuine work-exactness gate (`computed == G*TOTAL_super`,
occ[71]) was UNAFFECTED throughout.

FIX APPLIED (both the success-path print and the timeout-forensics mirror): the host cannot read a build
defsym, so it no longer derives ONE build-order-dependent number. It now prints the raw counters plus
**both** readings, labelled — `grow-first (CF0): entered = occ[96]` and `commit-first (legacy): entered =
occ[96] - occ[73]` — and emits an explicit NOTE when `occ[96] < occ[73]`, which is impossible under the
legacy order and therefore positively identifies a grow-first bin. Print-only change; host compiles clean
(23 pre-existing warnings, 0 errors). Rationale recorded in-line at the call site: printing one silently
wrong number is how a first run gets misread.

## LOAD-BEARING UNDOCUMENTED INVARIANTS (R2 — record these)

- **`SLOT_N == SSWIN`** is load-bearing against slot aliasing during the publish window (the sentinel writes
  `SL_RBDONE=ACC_N` while `SL_GEN` still names occupant `r-32`). Enforced only by a `.set`, not a `.error`.
- **`.Lflow_da_rollback` and `.Lflow_da_termslot` are DEAD CODE** (verified: no inbound branch target in the
  emitted binary). They are the only decrementing CAS sites; `ASSIGN_HEAD` is therefore strictly monotone,
  which is what makes the widened peek->CAS window ABA-free. **If either is ever revived, the whole window
  becomes ABA-exposed.**
- **The ring-compute claim's safety rests on UNREACHABILITY, not on publish order.** It is the one reader
  that gates on the pending bit without re-validating `SL_GEN`. If `DSWS2_ROLEFLOW=0` is ever built with
  `DSWS2_OVERLAP=1`, that path becomes reachable and must be re-audited against the BLOCKER-1 argument.

## CONFIRMED CLEAN (invariants cited, verified in emitted code)

No silent wrong C; no lost or duplicated work (single commit point = the `ASSIGN_HEAD` CAS; the grow is
wave-private; the grow-fail STAMP block is compiled out — `RB_PENDING` is materialized exactly 32 times,
all in cold-start slot init). No deadlock (every "cannot proceed" path is an unconditional branch to
`.Lflow_feedmt_sleep`; all fat windows bounded). No cross-wave ordering violation (every `lds_put` carries
its own `s_wait_dscnt 0x0` per store; the historical `RINGINIT` hole is covered by the real barrier).
ROLE CAS correct on both win and loss. Prefetch cannot go OOB (`t` clamped to `TOTAL-1`, `ksi` to
`[0,n_kseg-1]`, address a strict subset of the real access set). No register collisions.

## WHERE THIS LEAVES THE STACK

**Safe to run** (after the C1 revert) but **not expected to help**: the prefetch warms 1.5%, the role
economy is inert and mislabels every wave, and grow-before-CAS costs pipeline drains proportional to
contention. The honest position is that the CF0 stack is correctness-complete and mechanism-incomplete.

## PROCESS

claude__main's own review of this code found no blockers, twice. Both rounds of independent review found
real defects — including, this round, a safety regression claude__main itself introduced via the brief.
INDEPENDENCE (denying the reviewer all prior conclusions) is the active ingredient, not model choice.
Give the reviewer the COMPLETE build profile — last round a missing `FM=1` produced a phantom blocker.
