# DSWS Emergent Wave-Economy — Design (Step 1 of the "feed + kill-the-8-wave-limit" work)

**Date:** 2026-07-05
**Status:** design, pre-plan. Follows `DSWS_STAGGER_DESIGN.md` / `DSWS_STAGGER_PLAN.md` (write-once + stagger, both oracle-clean to 14 waves/WG).
**Council:** Fable + Codex consulted; verdict + corrections baked into §5.

---

## 1. Goal

Stop specifying the wave mix at all. The VGPR-budget accounting we just built should *decide* the
wave economy at runtime — how many waves are fat compute vs lean feed emerges from the physical VGPR
file, moment to moment. **No artificial wave-count limit.** Owner's words: *"we built this elegant
accounting… so why are we even specifying waves? Let the accounting decide; set no artificial limit."*

This is **Step 1 of two**. It removes the baked `NCOMP/NAFEED/NBFEED` mix and lets the economy emerge.
It is **not** the operand-overlap feed fix (`POOL_N=2` via `SEGK=16`) — that is **Step 2**, a separate,
separately-measured experiment (§7). The two are orthogonal and must not be conflated (§5, Risk B).

---

## 2. The mechanical truth (why this is a subtraction, not a rewrite)

Grounded in the current `occ_kernel_dsws_flow.s`:

- **Concurrent-fat already emerges from hardware.** The per-burst grow (`.Lflow_compute`) is a raw
  `s_alloc_vgpr NFV` → `s_cbranch_scc0 .Lflow_growfail`. SCC=0 means the *physical VGPR file* could not
  grant the grow. That is the moat and the true cap. The `BUDGET` / `reserve_try` LDS ledger
  (`VRESV_OFF=52`) is used **only** by `conv_apply` (the old role-conversion envelope), which is
  **dormant** in the flow build. So we do **not** need to build a budget mechanism — the hardware is it.
- **No runtime role emergence exists today.** The coordinator's sense/nudge is deferred
  (`.Lflow_coord_period`: *"static launch mix for the first flow build"*). A wave reads a **static**
  `ROLE[wid]` each loop and simply *is* that role; the coordinator only ever rewrites `ROLE[wid]` to
  `ROLE_RETIRE` at the end.
- **The only fungibility is the coast.** A COMPUTE wave with nothing staged, or whose grow fails,
  runs feed code (`.Lflow_coast` → helps whichever of A/B is behind) and returns to `.Lflow_loop`.
  Every path (`.Lflow_feed`, `.Lflow_feed_empty` = `s_sleep;branch`, `.Lflow_stage_adv`, bank-done,
  coast) loops back to the head, which re-reads `ROLE[wid]` and re-dispatches. So a **compute-seeded
  wave already oscillates compute↔feed** by (staged-availability × hardware-grow-success) each
  iteration. This flow is **asymmetric**: compute→feed only. A feed-seeded wave never tries compute.

**Consequence:** the "baked mix" problem is entirely in (a) the *seed* (feed-seeded waves are stuck
feeding) and (b) the assemble-time coupling of launch count / guards to `NCOMP/NAFEED/NBFEED`. Fix the
seed to be compute-biased with a minimal feed floor, decouple the launch count, and the economy emerges
using machinery that already exists and is already oracle-clean.

---

## 3. Design

### 3.1 Compute-biased seed with a static liveness floor
Replace the mix-proportional seed (`wid<NBFEED→BFEED`, `wid<NBFEED+NAFEED→AFEED`, else COMPUTE) with:

```
wid 0            -> coordinator (runs lean B-feed between ASSIGN duties, as today)
wid 1            -> ROLE_AFEED     (static A-feed floor)
wid 2            -> ROLE_BFEED     (static B-feed floor)     [wid0 already B-class, so this is the 2nd B]
wid 3 .. W-1     -> ROLE_COMPUTE   (everything else)
```

- The floor (`≥1` dedicated A-feed + `≥1` dedicated B-feed, plus the coordinator's B-class feeding) is
  a **liveness invariant**, not a mix (§5, Risk A). At `POOL_N=1` the pipeline is depth-1 ping-pong;
  liveness *requires* someone always able to fill the operand pool regardless of the compute cohort's
  phase. This guarantees it deterministically, cheaply, with no spin.
- Everything above the floor is COMPUTE-seeded. Those waves self-distribute: when work is staged and a
  grow succeeds they burst; when the budget is full (`growfail`) or nothing is staged they coast to
  feed and fill the pool; next iteration they try compute again. **Concurrent-fat = whatever the
  physical VGPR file admits** (the `growfail` boundary). No number is imposed.
- The floor is fixed by `wid`, independent of any `NCOMP/NAFEED/NBFEED`. Those three symbols are
  **deleted** from the kernel.

### 3.2 Decouple the launch count from the mix
- Introduce a single wave-count defsym (e.g. `WAVES` becomes a standalone `-defsym`, default the
  swept-safe max). It no longer equals `NCOMP+NAFEED+NBFEED`. Everything sized off `WAVES` (the
  32-wave mailbox region, the retire `.rept WAVES`, guards) keeps working — it already tolerates up
  to 30 (see below).
- `G` (rowblk-group size = LDS bank count) is **structural**, currently written `.set G, 6 // = NCOMP_MAX`.
  Make `G` its own defsym (default 6), *not* tied to the launch mix. `ACC_N` (banks) stays coupled to `G`.

### 3.3 `BUDGET` becomes a physical constant, decoupled from the mix
- `BUDGET` today = `NCOMP*NFV + (NAFEED+NBFEED)*VLEAN` (mix-derived). With the mix gone, redefine it as
  a **physical VGPR budget** defsym `VBUDGET` (the VGPR-file credit ceiling), independent of wave count.
- This does **not** change concurrent-fat behavior (the hardware `s_alloc_vgpr` is the real cap; the
  ledger is dormant). Its only jobs now are (i) keeping the assemble-time sanity guards meaningful and
  (ii) being available if we ever re-arm `conv_apply`. Keep the guards as physical sanity checks:
  - `WAVES*VLEAN ≤ VBUDGET`  (all waves fit lean)
  - `WAVES*VLEAN + (NFV-VLEAN) ≤ VBUDGET`  (≥1 wave can grow — liveness for compute progress)

### 3.4 Host auto-derives `W_launch` (Option 3 — no wave number anywhere)
The host picks the launch count as the **min of all real caps**, expecting VGPR to almost never be the
winner:

```
W_launch = min( occupancy_cap,          // swept empirically (resident waves/SIMD, ~16 wave32)
                coordinator_cap = 30,    // mailbox-tail squat (§3.5); hard structural ceiling
                lean_fit = (VBUDGET - (NFV-VLEAN)) / VLEAN )   // all-lean + one grow of headroom
```

- Emergence belongs on the **fat/lean time profile** (the ledger/hardware already deliver it) and on
  the **role mix above the floor** — **not** on resident wave count, which is an
  occupancy-and-coordinator decision the ledger never measured.
- The host stops reading `DSWS_NCOMP/NAFEED/NBFEED` for the flow path and stops naming the bin per-mix.
  One flow bin, built at `WAVES=W_launch`; the host launches `W_launch*32` threads/WG.
- `occupancy_cap` is the one unknown → sweep it (§7). Until swept, gate `W_launch` at a conservative
  value and raise it by measurement.

### 3.5 Mailbox / coordinator safety at larger counts
- The mailbox already reserves a full 32-wave region: `ROLE_BASE=20 … SLOTC_BASE=148`. The coordinator
  state squats at `COORD_KSI_OFF=140` / `COORD_T_OFF=144` = `ROLE[30]` / `ROLE[31]`. Hence **`WAVES ≤ 30`
  is safe today**; `WAVES` of 31–32 would corrupt coord state **silently**.
- `W_launch` is therefore hard-capped at **30** in §3.4. This is comfortably above the expected
  occupancy cap (~16 wave32), so we do **not** need to relocate coord state for Step 1. (If a future
  step wants 31–32, move `COORD_KSI/T` past the 32-wave region — out of scope here.)

---

## 4. What we deliberately do NOT build

- **Option 2 (fully fungible, one uniform wave body): rejected.** Council-unanimous. Dyn-VGPR already
  decouples launch footprint from peak, so fusion doesn't save VGPR; its real cost is that feed would
  become a *residual of compute contention* (you feed only when a grow fails), coupling the bottleneck
  resource to the noisy grow-failure rate the stagger is built to damp. Keeping distinct role bodies
  keeps feed a **controlled variable with a floor**. Also: it's a rewrite of a just-clean kernel.
- **Re-arming the coordinator sense/nudge / `conv_apply` conversion:** not needed. The compute-biased
  seed + coast + hardware grow-fail deliver emergence without it. `conv_apply` stays dormant.
- **The `POOL_N=2` / `SEGK=16` feed-overlap fix:** that is Step 2 (§7), not this change.

---

## 5. Risks & mitigations (from the council)

- **Risk A — liveness / stampede (both agents).** All-lean-unassigned could stampede all-compute
  (nobody feeds → compute stalls on an empty pool) or all-feed (nothing computes). **Mitigation:** the
  static feed floor in §3.1 is a *correctness invariant*, not tuning. At `POOL_N=1` the single operand
  slot also self-serializes, and `.Lflow_feed_empty` yields safely; deadman covers any stall.
- **Risk B — this is ONE change or TWO? (both agents): TWO.** Emergent mix governs *who* works;
  `POOL_N` governs *whether feed and compute overlap at all*. Orthogonal. Killing the wave cap alone can
  even **regress** (more waves eligible to grab compute → more pressure on the one slot). The feed win
  is gated on `POOL_N=2` (`SEGK=16`), which is not free: halving SEGK doubles K-segments → doubles
  `ds_add` + `RBDONE` churn **and halves the WMMA burst length**, attacking the stagger's grow/shrink
  amortization. **Mitigation:** ship Step 1, re-confirm oracle + stagger self-maintenance, measure;
  *then* do Step 2 with its own before/after.
- **Risk C — the VGPR file is NOT the binding cap (Fable; the one I underweighted).** `VBUDGET`/VLEAN
  math lands ~45 waves, but occupancy (~16 wave32) and the coordinator ≤30 bind first. **Mitigation:**
  §3.4 derives `W_launch = min(occupancy, 30, lean_fit)` — VGPR is a sanity ceiling, not the driver.
- **Risk D — coordinator correctness under dynamic churn (Codex).** Oracle-clean at 14 waves does not
  prove stability near the physical boundary where grow-fails, coasting, pool state, and completer
  election interact. **Mitigation:** strongest oracle + repeats at each new wave count; watch for the
  first sign of a completer/mailbox race and root-cause immediately (do not defer).

---

## 6. Acceptance criteria (Step 1 done = all hold)

1. **Correctness:** oracle `bad=0`, `max_rel=0`, `occ[0]=0` at the new (larger) `W_launch`, with repeats
   (Risk D). Re-confirm at the old 4c2a2b-equivalent count too.
2. **No mix anywhere:** kernel has no `NCOMP/NAFEED/NBFEED`; host does not read `DSWS_NCOMP/AFEED/BFEED`
   for the flow path and does not name the bin per-mix; `W_launch` is host-derived.
3. **The stagger engages:** `grow-fail` (STAGINSTR `occ[73]`) flips from **0 → >0** — the proof the
   physical VGPR file finally binds and the stagger's coast-repulsion is doing real work. Peak resident
   waves (`occ[1]`) rises meaningfully above the old 8.
4. **Stagger self-maintains:** no wedge/brick; deadman `occ[0]` stays 0; coast fraction sane (not ~100%
   feed-starve *and* not ~0% — some genuine grow-fail repulsion).

Measurement uses the STAGINSTR counters already in place: `occ[70]=coast`, `[71]=computed`, `[72]=feed`,
`[73]=growfail`, plus `occ[1]` resident peak and `occ[58]` peak concurrent fat.

---

## 7. Step 2 (sketch — separate experiment, not part of this plan)

`POOL_N=2` double-buffer so feed and compute overlap instead of serializing on one operand slot.
Needs `SEGK=16` to halve operand footprint and fit two slots + `ACC_N=6` banks in 64 KB. Costs: 2× K-seg
count (2× `ds_add`/`RBDONE` churn), halved WMMA burst (stagger amortization pressure), and the "ksi=0
write vs add" first-segment handling must become an explicit bank-zero (two ksi in flight otherwise race
the write/add). Land and measure on its own after Step 1 is clean.

---

## 8. Open calibration items (resolved during the plan, not before)

- `VBUDGET` exact value (physical VGPR-file credit for R9700 wave32) — sweep/confirm; default generous.
- `occupancy_cap` — sweep resident waves/SIMD to find the real ceiling below 30.
- Whether the coordinator (wid0) B-feed duty + `wid1` A-feed + `wid2` B-feed is the right floor, or
  `1×A + 1×B` (drop the redundant 2nd B) suffices — decide by the liveness sweep.
