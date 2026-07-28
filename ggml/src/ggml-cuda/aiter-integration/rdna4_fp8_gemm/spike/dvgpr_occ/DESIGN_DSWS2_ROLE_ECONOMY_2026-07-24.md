# DESIGN — DSWS2 Bidirectional Role Economy (feed = prefetch), 2026-07-24

**Author:** claude__main, from a kmbandy round-table (2026-07-24). **Status:** design for phased build.
**Kernel:** `occ_kernel_dsws_flow.s` + host `occ_dispatch.cpp`. **Config:** A1 canonical + `DSWS2_RCONV=1`.
**Builder:** Sonnet subagent writes the assembly. **PROSE + mechanism + invariants ONLY — no code.**
**Supersedes** `DESIGN_OVERLAP_2WGCU_2026-07-24.md` and the ad-hoc Phase-1B control-only ring (its 3 Codex
BLOCKERs are *dissolved* by this design, not patched — see §5).

> Concurrency-critical (role transitions + dual-generation frontier state + a launch change). This kernel's
> handoff races have repeatedly beaten static reasoning (the 2026-07-20 dropped-group race; the Phase-1B
> stale-STI race Codex just caught). So: paper-design first, oracle-CLEAN + WORK-EXACT gated, independent
> adversarial review MANDATORY on every handoff change, no throughput claim without a supervised steady run.
> Everything defsym-gated; OFF byte-identical to `cac3ff7c`.

---

## 1. The vision (kmbandy, settled 2026-07-24)

DSWS is an **adaptive wave-role economy**: waves rebalance between roles toward the live bottleneck, using
dyn-VGPR to make the per-SIMD VGPR budget fungible across roles. The current kernel implements only *half*
of that — RCONV converts compute→feed but **never back** (one-way ratchet, `:5173`; `ROLE_COMPUTE` written
only at init `:3181`). A one-way conversion is not adaptive. The fix is a **bidirectional** economy, and —
because self-serve already eliminated LDS operand staging — **"feed" is repurposed from "stage LDS operands"
to "prefetch the next super-tile."**

## 2. The measured problem it answers

Assign-bound: the frontier exposes one field = `n_kseg = K/SEGK = 10` split-K slices of one super-tile;
with 30 waves (soon 16/WG × 2 WG/CU) only ~10 have work, the rest coast. ADVPROBE: the advance mechanism
is ~10% of the ~2600-tick interval; ~90% is the dead gap where nothing new is exposed. "Assign-bound,"
"100% empty-frontier," "20 idle waves" are one fact: **too few work-units exposed for the wave count.**

## 3. The architecture

**Roles (bidirectional).** Every wave is one of {compute-current, feed-next}. It flows between them by the
live bottleneck — toward feed when the frontier is nearly drained / a prefetch is due, back to compute when
a field opens / work is available. No one-way ratchet; no permanent role; no privileged wave. The flow is
adaptive (driven by frontier/completion state), never a static designation.

**Feed = prefetch the next super-tile.** With operands L2-self-loaded (§4), a feed wave does NOT stage LDS
operands. It: (a) issues the next super-tile's global loads to **warm L2** so incoming compute waves'
self-loads hit cache, and (b) helps **advance the frontier** (the overlap trigger / reservation) so the
next field is open the instant the current drains. The ~(waves − n_kseg) "extra" waves — e.g. 16 waves vs
10 slices — are exactly the prefetch capacity. This is the same mechanism as the two-generation overlap:
the slack waves prefetch generation N+1 while the busy waves finish generation N.

**Two-generation overlap.** A second ACC_N accumulator generation (funded by the operand-pool reclaim).
The frontier ping-pongs even/odd super-tile between generations; the trigger is **real completion**
(`GSTORED`/`TILEDONE`), never `DRAIN` (which is a reservation artifact under SELFSERVE — see §5). Exactly
1-deep. Per-generation `TILEDONE`/`GSTORED`/`zero_banks` so a completer never stores half-summed banks
(the POOL_N>1 `bad=96/116` race class).

**2 WG/CU (16 waves).** Split 30 waves → 2×16 so each funnel arbitrates 10 slices among 16 waves (10/16,
not 10/30 — halves the herd) and each WG runs the proven single-generation-per-WG frontier. Host guard
(`occ_dispatch.cpp:2010-2024`) requires per-WG LDS ≤ 32,768B and WAVES ≤ 16 (32 wave-slots/CU). Operands
L2-only + 2 ACC generations ≈ 26KB/WG fits (2×26 = 52KB < 64KB); the prior "2 WG/CU garbage" verdict was a
clamp artifact (`:1995`), so this is untested, not disproven.

**Operands L2-only.** The self-serve burst already self-loads A+B from L2 (`:4755`/`:4764`); commit ALL
paths to it and reclaim the 40,960B `OP_BASE + POOL_N*OPSTRIDE` pool. B-reuse stays L2-warm-cache for all
paths (kmbandy: full-L2, 2026-07-24). The operand-bandwidth question is deferred, not tested — it can't
bind while assign-bound, and a NOBLOAD run reads falsely flat until the wall moves.

## 4. Why grow-fail can't be tested offline (keep the config)

grow-fail=0 on every measured shape **because we are assign-bound** — starved waves never contend the VGPR
budget, so it never binds. The fix (more exposed work → more concurrent fat waves) is *what makes grow-fail
bind*. So grow-fail validates as a **consequence of the fix working**, at the final supervised run — NOT by
forcing it with a config change (which would test the wrong thing). Keep the config; do not chase grow-fail
separately.

## 5. How the three Codex BLOCKERs dissolve (not patched)

Phase-1B built an ad-hoc control-only ring for the grow-fail handoff; Codex found 3 real blockers
(`CODEX_OVERLAP_P1B_REVIEW.md`). This design removes that machinery:
- **BLOCKER 3 (one-way RCONV → no ring consumer → DRAIN deadlock):** dissolved by bidirectional flow — a
  grow-failed wave *flows to feed/prefetch*, it does not shove a half-published item into a ring for a
  stranger to claim; and roles flow back to compute so the consumer population never bleeds to zero.
- **BLOCKERs 1 & 2 (grow-fail STAMP stale-STI claim `:4717`; premature drain on reuse):** these live in the
  grow-fail→ring publish. With no ring handoff, there is no cross-wave claim of a grow-fail-published slot,
  so the stale-STI and premature-drain races have no surface. A grow-failed wave keeps/【cleanly releases】
  its own reservation and retries or flows to feed — it never publishes a claimable payload for another
  wave.
- **The `ASSIGN−DRAIN` trigger was wrong** (DRAIN advances on the pre-completed sentinel `:4664-4684`
  before compute) — the overlap trigger reads `GSTORED`/`TILEDONE` instead.

## 6. Phased build plan (each phase oracle-CLEAN + WORK-EXACT before the next; adversarial review on each)

1. **P1 — operands L2-only + delete the grow-fail ring.** Keep Phase-1B's clean L2-reclaim; REPLACE its
   control-only-ring grow-fail handoff with a **clean grow-fail path that publishes nothing claimable**: a
   grow-failed wave drops cleanly to coast/retry (keeps correctness; grow-fail is rare/zero in this regime,
   so correct-if-it-fires is enough — the *real* grow-fail-flows-to-feed is P2). This removes all 3 Codex
   blockers by removing the ring. Per-WG LDS ~14KB (one generation). **THIS HANDOFF.**
2. **P2 — bidirectional role economy + feed-as-prefetch.** feed→compute reversion (kill the one-way
   ratchet); redefine feed as L2-prefetch + frontier-advance of the next super-tile; grow-fail flows to
   feed. Adaptive flow driven by frontier/completion state, anti-thrash hysteresis.
3. **P3 — two-generation overlap.** second ACC generation, `GSTORED`/`TILEDONE`-relative trigger, per-gen
   completion protection, exactly 1-deep.
4. **P4 — 2 WG/CU.** build WAVES=16, dispatch `ML8_POOL=128`; host guard passes; supervised measure.

Do NOT fuse phases. Each is independently oracle-gatable — the only way a concurrency-critical change on
this kernel earns trust.

## 7. Invariants (STOP and report if any cannot hold)

- **No claimable cross-wave payload without a full release fence.** The stale-STI class (Codex BLOCKER 1):
  any state that makes a slot claimable by another wave must be published LAST, after all payload
  (`SL_STI`) and the `SL_GEN` fence — or the claim path must validate `SL_GEN`. P1 sidesteps this by
  publishing nothing claimable on grow-fail.
- **Per-generation completion protection** (P3): gen A's C never zeroed/stored while gen A still
  accumulates, independent of gen B. The `bad=96/116` race class.
- **Bidirectional, adaptive, no ratchet, no designator** (P2): roles flow both ways by live state; no wave
  is permanently any role; whoever is slack feeds, whoever has work computes.
- **Trigger on real completion** (`GSTORED`/`TILEDONE`), never `DRAIN`.
- **POOL_N=1, SEGK=256, JDEPTH, VBUDGET, duty-cycle/moat unchanged.** Operands L2-only; only the
  accumulator doubles (P3). Do NOT raise VBUDGET to dodge grow-fail.
- **Byte-identical off.** `DSWS2_OVERLAP=0` (and the P2+ defsyms=0) byte-identical to `cac3ff7c` at A1.
- **Moat is measured, not assumed.** grow-fail binding is the goal, not a failure; measure, don't pre-judge.

## 8. Gates (builder = OFFLINE ONLY; never dispatches)

Per phase: `=0` byte-identical to `cac3ff7c`; ON build assembles + links 0-spill (RGA); host compiles +
occupancy guard/`static_assert` hold; written self-audit vs §7 + a static argument for any path that
can't be exercised offline (grow-fail). Oracle-CLEAN + WORK-EXACT + steady-state TF is a **supervised
dispatch claude__main runs** with a greenlight. **Every handoff-touching phase gets an independent
adversarial (Codex) review before its supervised run** — the Phase-1B experience is why.

## 9. What NOT to do

No ring handoff that publishes a claimable payload (P1). No second operand slot / POOL_N change. No
SEGK/JDEPTH/duty-cycle change. No `DRAIN`-based trigger. No forcing grow-fail with a config change. No
fusing phases. No assembly in this doc. Builder never dispatches to the GPU. Do not run/gate a B-bandwidth
(NOBLOAD) test now.
