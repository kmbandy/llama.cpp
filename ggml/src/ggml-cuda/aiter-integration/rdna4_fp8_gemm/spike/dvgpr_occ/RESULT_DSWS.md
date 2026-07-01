# DSWS Adaptive Wave-Role Controller — Results

Substrate: `occ_kernel_coop.s` (dyn-VGPR + split-K + POOLTERM). Plan: `PLAN_DSWS_CONTROLLER.md`.
Spec: `SPEC_DSWS_CONTROLLER.md`. This doc is appended through Phases 1–4.

## Phase 1 — static 3-role substrate

### Offline gates (T1.1–T1.3)

- **T1.1 config + refuse-path** (`occ_dispatch.cpp --dsws`): host compile clean; refusal smoke GREEN
  (`DSWS_NCOMP=0` → `REFUSE nComp>=1` rc=4; valid-but-unbuilt → `bin NOT BUILT` rc=4; `LOW>HIGH` → refuse).
- **T1.2 A-feed role + A-ring** (`occ_kernel_coop.s`, behind `.if DSWS`): 
  - `DSWS=0` **byte-identical** to the proven coop d0 (1716B) — the substrate is untouched.
  - `DSWS=1` assembles for all 3 plan cells × {dyn,static}: 4c2a2b, 6c1a1b, 2c3a3b.
  - Design: **frag-partitioned B** (B-feed wave owns `{ni : ni%NBFEED==b_id}`, bumps `prod_b[b_id]`,
    compute gates on `min(prod_b[*])`); **band-partitioned A** (A-feed wave owns `{bnd : bnd%NAFEED==a_id}`,
    band has one producer `prod_a[bnd]` + one consumer `cons_a[bnd]`); lead B-feed (wid 0) is the sole
    tile-claimer/broadcaster; A-feed + non-lead B-feed follow the ti/epoch broadcast. Zero inter-feed
    rendezvous → barrier-free invariant preserved (KG `4ce31886`).

### RGA static-analysis constants (gfx1201, 2×4 tile, dyn) — needed for the Phase-3 reservation envelope

| metric | value | source |
|---|---|---|
| **V_fat** (compute peak-live, grown wave) | **82** VGPR | RGA livereg "Maximum # VGPR used" |
| **NFV** (compute grow alloc target) | **112** VGPR | `((FB+2*FN+15)&~15)`; HW rounds 108→120 |
| **V_lean** (feed/A-feed lean alloc) | **32** VGPR | `s_alloc_vgpr 32` lean footprint |
| VGPR spills / SGPR spills / scratch | **0 / 0 / 0** | RGA stats (all 3 cells) |
| per-grow delta (V_fat_alloc − V_lean) | **80** VGPR | NFV(112) − 32 |
| BUDGET (per-SIMD VGPR pool) | *TBD T1.4/Phase 3* | ISA / occupancy math |

### T2.2 control-law CPU model (TDD, offline)

`dsws_ctrl_model.cpp` + `test_dsws_ctrl_model.cpp`: `watermark_decision` / `epoch_of` /
`gate_try_win` (single-winner CAS) / `reserve_grow` (envelope) — **ALL PASS**, stable across repeats
under real multi-thread races. Locks the control semantics the Phase-3 asm must match 1:1.

### T1.3 dispatch wiring — DONE (2026-06-28)

`run_mbcoop` extended with two defaulted params (`totalWaves`, `ldsBytesOverride`) — proven coop callers
byte-identical; `--dsws` calls it with `P=NCOMP`, `totalWaves=N`, `ldsBytesOverride=LDS_TOTAL_DSWS`.
`build_dsws.sh` builds all 6 bins (`occ_dsws_{4c2a2b,6c1a1b,2c3a3b}_r2[_dyn]_gd.bin`). **Offline gate GREEN:**
harness links clean; all 6 cells assemble; RGA 0-spill (livereg 82); both refuse-paths intact
(invalid → validation refuse; valid+unbuilt → bin-guard refuse; valid+built → would dispatch = T1.4).

<details><summary>(superseded) T1.3 in-progress notes</summary>

Approach: extend `run_mbcoop` with two **defaulted** params so the proven 2-role coop callers stay
byte-identical, then `--dsws` calls it with `P=NCOMP`, `totalWaves=N`, and the DSWS LDS size.

- **DONE:** `run_mbcoop` signature + `WAVES_LAUNCH = totalWaves>0 ? totalWaves : (1+P)` (occ_dispatch.cpp).
  Harness recompiles clean (niced build, 2026-06-27 23:08).
- **REMAINING (3 items, all offline):**
  1. `ldsBytes` override line in `run_mbcoop` (`ldsBytesOverride>0 ? ldsBytesOverride : coop-formula`).
  2. Replace the T1.1 `--dsws` stub (`[T1.1] ... dispatch wiring lands in T1.3`) with the real call:
     compute `N`, `ldsDsws = BRING + 4*nComp+16 + 4*(nBfeed-1) + RINGD*nComp*FM*256 + 8*nComp`
     (FM=2,FN=4 fixed v1 tile), loop `{down, down_pf}`, small tile-multiple oracle first,
     `run_mbcoop(dswsBin, dyn, pool, Mo,No,Ko, 2,4, nComp, ringd, /*fullCheck*/true, /*GENDIV*/true,
     1,0.0, /*totalWaves*/N, /*ldsOverride*/ldsDsws)`, gate on `bad==0`.
  3. `build_dsws.sh` — emits `occ_dsws_<c>c<a>a<b>b_r<RINGD>[_dyn]_gd.bin` (DSWS=1 FM=2 FN=4 POOLTERM=1)
     for cells {4c2a2b, 6c1a1b, 2c3a3b} × {dyn,static}; + RGA gate.
- **Gate:** `./build_dsws.sh static` (all cells assemble + RGA 0-spill) + harness links. No GPU.
</details>

### Gate 1 dispatch #1 — HUNG (not bricked), 2026-06-29

First GPU dispatch of the campaign fired (user-greenlit). `4c2a2b` static, `down_pf`, single-tile, but at the
harness **default pool=64** (I left `ML8_POOL` unset). Result: **clean software deadlock**, NOT a brick —
`timeout 30` caught it, teardown deliberately did not destroy the queue, dmesg silent (zero ring/fence/fault/MES),
both GPUs responsive. Wedge frame (direct atomics, reliable): `maxlive=64 claim=64 live=1` → 63 WGs correctly
retired on `ti>=TOTAL`, the **one WG that won `ti=0` deadlocked** in produce/consume. All fine markers 0 — but
that was instrumentation blindness (see below), not proof of no progress.

**Root cause of the blindness (2026-06-29):** `DIAG` defaults to 0 and `build_dsws.sh` never passed it → every
`mark` compiled to nothing. NOT the 2026-06-24 "v2 clobber" hypothesis (v2 is set once at kernel:276 and never
rewritten; the v2==0-gated `lds_put` provably works since the 63 WGs retired via LDS broadcast). Additionally the
live DSWS feed bodies (`.Lbfeed_role`/`.Lafeed_role`) had ZERO marks — the feed marks lived in the dead
`.Lfeed_role`. FIX (all offline, verified): `build_dsws.sh` now passes `DIAG=${DIAG:-1}`; added 10 marks to the
DSWS feed bodies (B: feedPhase/ti + slotok[18]/loadtr[19]/dsstore[20]/publish[21]; A: reached[28]/Aload[29]/
dsstore[30]/publish[31] on fresh occ slots 28-31); harness zeros+prints occ[28..31] + a DIAGINIT line (occ23-27).
Gates: 6 DIAG=1 bins clean, 55 mark atomics in disasm, RGA livereg 82 / 0-spill, DSWS=0 still 1716B byte-identical.

**Reflight plan (pool=1, awaiting greenlight):** matches the kernel's POOL=1 DIAGNOSTIC TERMINAL design, removes
the 64-WG claim-contention variable, and with DIAG=1 will localize the exact wedge instruction:
`ML8_COOP_STREAM=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_POOL=1 DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2`
`DSWS_RINGD=2 DSWS_DYN=0 DSWS_ONLY=down_pf DSWS_ORACLE_MTL=1 DSWS_ORACLE_NTL=1 timeout 30 ./occ_dispatch --dsws`

### Gate 1 dispatch #2+#3 — root cause LOCALIZED + CONFIRMED, first DSWS green (2026-06-29)

**#2 (pool=1, 4c2a2b):** HUNG again, but DIAG-instrumented this time. Wedge frame:
`INIT adm1 tmr1 lds1 rdv1 | feedPh2 compPh4 cons3 tiles0 | feed:tr2 pub3 comp:dsB2 wm3`. Decode: init/rendezvous
FULLY PASSED (kills the old init-deadlock fear for static); the protocol RAN and made real progress (B published 3
steps, compute did 3 WMMA) then froze at step ~3 of 32. → a MID-K-LOOP producer/consumer stall, not an init wedge.

**Root cause (LOCALIZED by evidence + code, CONFIRMED by #3):** the A-feed loop is **band-outer / K-inner**
(`occ_kernel_coop.s` `.Lafeed_role`: `.rept NCOMP` wraps the K-loop), so when an A-feed wave owns >1 band it
produces band b's ENTIRE 32-step K-loop before touching band b+NAFEED. The compute waves consuming the not-yet-fed
bands starve → never release their consumer counters → `min_cons` stays low → the shared B-ring (depth RINGD=2)
can't recycle slots → B and all compute jam → WG wedges at ~step RINGD. B-feed does it RIGHT (K-outer / frag-inner).

**#3 (pool=1, 2c3a3b) — CONFIRMING PROBE: `oracle CLEAN ok=16 bad=0`.** With NAFEED=3 ≥ NCOMP=2 every band has
its own A-feed wave → zero band-sequential starvation → GREEN. Airtight: the only changed variable is bands-per-
A-feed-wave, and it flips green↔hang exactly as predicted. **FIRST oracle-green of the DSWS 3-role kernel — the
cooperative A+B-feed→compute→store protocol is NUMERICALLY CORRECT.** The hang was a pure feed-scheduling bug.

All three dispatches HUNG-or-completed CLEANLY (no brick; dmesg silent, R9700 responsive each time).

**THE FIX (applied + verified 2026-06-29):** restructured `.Lafeed_role` to **K-outer / band-inner** (mirror
B-feed): every K-step produces one step for ALL owned bands, so no compute band starves. `astep==k` fell out for
free (use the loop counter directly; recompute per-band saddr from trow each iter — lean feed has SGPR headroom).
Offline gates: 6 cells assemble, RGA livereg 82 / 0-spill, DSWS=0 still 1716B byte-identical.

### Gate 1 dispatch #4 — FIX VALIDATED: 4c2a2b oracle CLEAN ok=32 bad=0 (2026-06-29)

The headline cell that deadlocked at step 3 in #2 now runs the full 32-step tile and matches the fp8 oracle
exactly (32/32 frags), clean exit, no hang, no brick.

### Gate 1 dispatch #5 — extreme case: 6c1a1b oracle CLEAN ok=48 bad=0 (2026-06-29)

One A-feed wave cycling all 6 bands per K-step (192x64 tile) — the worst case for the old band-outer bug —
runs CLEAN, 48/48 frags. **Both cells that deadlocked (4c2a2b, 6c1a1b) now pass; 2c3a3b passed pre-fix.**
The K-outer A-feed fix is validated across the full band-count range. Pending: a 2c3a3b post-fix regression
re-check (low risk — NAFEED>=NCOMP means each wave owns <=1 band, the simplest K-outer case), then Gate 1 (T1.4)
is fully GREEN and the DSWS static 3-role substrate (Phase 1) is DONE. 5 GPU dispatches total, ZERO bricks.

### ✅ GATE 1 (T1.4) PASSED — 2026-06-29

All three role-mix cells oracle-CLEAN at pool=1 single-tile down_pf, static (DYNVGPR=0):
**2c3a3b ok=16 / 4c2a2b ok=32 / 6c1a1b ok=48, all bad=0.** The DSWS static 3-role substrate (Phase 1) is DONE —
the cooperative fp8 GEMM with separate fat-compute / lean-A-feed / lean-B-feed waves is numerically correct across
the full role-mix range. 6 supervised GPU dispatches total today, ZERO bricks (every one hung-cleanly or completed;
dmesg silent each time). Net of fixes that got here: SAFEPROBE feed ti-clamps + SAFEPROBE=1 build; DIAG=1 + feed-body
marks (instrumentation was DIAG=0/uncompiled, not the old "v2 clobber"); and THE bug — A-feed band-outer/K-inner
loop starved un-fed compute bands → fixed to K-outer/band-inner.

NEXT: Phase 2 (sensing + role slots + reservation counter), then Phase 3 (conversion/epoch-ticket + dyn-VGPR
DYNVGPR=1), then Phase 4 (adaptivity proof + tuning). NOTE the pool=64 path still hung in dispatch #1 — pool>1
cross-WG behavior is a separate open item to revisit before any multi-WG perf run (Phase 1 gate is pool=1 by design).

<details><summary>(historical) Gate 1 pre-dispatch notes</summary>

First display-GPU dispatch of the whole campaign. As of 2026-06-28: ZERO GPU dispatches run — all offline.

**HARDENING PASS — DONE 2026-06-29 (all offline, zero GPU):**
1. ✅ **SAFEPROBE `ti`-clamps added to BOTH A-feed and B-feed decodes** (`occ_kernel_coop.s`), mirroring
   compute lines 471-472; gated `.if SAFEPROBE` inside `.if DSWS`. The `s_min` clamp is the safety mechanism,
   independent of the (possibly-flaky) diagnostic `mark`. On inspection the feeds were ALREADY covered on the
   *vector* vaddr (the v8/v9/v10 clamps live in the common prologue, run by all waves) AND already retire on
   `ti>=TOTAL` (terminal test) — the new clamp closes the one remaining asymmetry on the *scalar* saddr.
2. ✅ **Rebuilt all 6 bins with `SAFEPROBE=1`** — build_dsws.sh was passing it ZERO times → bins were
   SAFEPROBE=0 = NO clamps compiled at all; THIS was the real lever. RGA re-gated: livereg **82**,
   SCRATCH/SGPR_SPILLS/VGPR_SPILLS all **0**. DSWS=0 still byte-identical to `.clean_bins` (**1716B**).
3. ⏳ **First contact = single tile** (`DSWS_ORACLE_MTL=1 DSWS_ORACLE_NTL=1` → TOTAL=1) — dispatch-time env,
   set in the gated command below. Awaiting user greenlight.

**Brick-risk assessment (honest, 2026-06-28):** LOW-to-MODERATE. Static (`DYNVGPR=0`) **eliminates the
dyn-VGPR brick class** (#1 historical vector — no `s_alloc_vgpr`). Residual: (a) protocol-hang → teardown
wedge (200 lines of new hand-asm, never executed; `timeout 30` + 0.75s chunk-abort are the net, but
recovery-from-hang is less proven than clean completion); (b) OOB page-fault from a bad `ti` — mitigated
to ~proven-coop level by the hardening pass above. NOT claiming "can't brick."

**Dispatch discipline:** set `ML8_COOP_CHUNK` (compositor-safe, opt-in; `ML8_COOP_CHUNK_MAXS=0.75`) +
`ML8_COOP_STREAM=1` + `timeout 30`. ONE gated dispatch at a time. Hang/brick = full STOP + report, never
auto-fire the next. Proposed first command (post-hardening, static, single-tile, down_pf):
`ML8_COOP_STREAM=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2`
`DSWS_RINGD=2 DSWS_DYN=0 DSWS_ONLY=down_pf DSWS_ORACLE_MTL=1 DSWS_ORACLE_NTL=1 timeout 30 ./occ_dispatch --dsws`

</details>

---

## Phase 2 — Sensing + role slots + reservation (offline, no GPU)  — 2026-06-30

**Status:** Tasks 2.1, 2.2, 2.3 DONE + gated offline. Task 2.4 (supervised GPU sensor-sanity) is the next STOP-for-go dispatch.

### T2.1 — controller LDS state (`occ_kernel_coop.s`, behind `.if DSWS`)
Appended after `CONS_A_OFF` so all proven offsets stay byte-identical:
`NCOMP_SLOT / NAFEED_SLOT / NBFEED_SLOT` (role slots), `GATE_OFF[4]` (one epoch-gate per conversion
direction: 0=comp→Afeed 1=comp→Bfeed 2=Afeed→comp 3=Bfeed→comp), `VRESV_OFF` (vgpr_reserved
envelope), `SEGCNT_OFF` (per-WG segments_processed). `LDS_TOTAL_DSWS` rolled forward; existing
`>65536 .error` now covers them. **Gate:** 6 bins assemble; RGA 0-spill, `USED_LDS=32768`, livereg
**V_fat=82**; DSWS=0 byte-identical (1716B).

### T2.2 — CPU control-law model (already banked, re-verified)
`dsws_ctrl_model.cpp` + `test_dsws_ctrl_model.cpp` → `ALL PASS` (watermark_decision / epoch_of /
gate_try_win single-winner / reserve_grow envelope). This is the 1:1 semantic spec the Phase-3 asm transcribes.

### T2.3 — leader state-init + read-only sensing + snapshot instrumentation
- **Reservation unit clarified:** the envelope uses the hardware **alloc** footprint, not live-peak.
  Fat = `NFV` (=112 this tile, the actual `s_alloc_vgpr` target); lean = `VLEAN`=32 (the feeds'
  `s_alloc_vgpr 32`). `vgpr_reserved` init = `NCOMP*NFV + (NAFEED+NBFEED)*VLEAN`. (RESULT table's
  "V_fat=82" is live-peak, a different measurement; the grow-delta accounting `NFV−32=80` confirms
  the envelope is in alloc units.)
- **Leader init** (extends the barrier-free INITFLAG publish, all lean-32 there): role slots ← launch
  mix, gates ← 0, `vgpr_reserved` ← launch envelope, `segcnt` ← 0. Constants written by each wave's
  lane-0 to WG-shared LDS (idempotent — identical values).
- **Read-only sensing (NO actuation):**
  - New `mark_set off,val` macro = lane-0 **plain global_store** to `occ[off]` (vs `mark`'s atomic-MAX),
    so the harness's 200 ms poll sees a sensor *oscillate* instead of latching its max.
  - Compute **segment boundary** (once per tile): `occ_b = min(prod_b)−cons` → occ[32]; `occ_a =
    prod_a[cid]−a_step` → occ[33] (both consumer-observed ring backlog, range [0,RINGD]/[0,RINGD_A]).
  - Per-WG `segcnt` bump by lead compute (cid==0) on each segment (POOL=1: one tile = one WG segment).
  - Role slots streamed once per compute wave (read LDS → occ[34/35/36]) — proves the controller-state
    init round-trips through cross-wave LDS.
- **`occ_dispatch.cpp`:** DIAGFINE zero-loop extended to 14..36; new fields added to the 200 ms live
  stream line (`SENS occ_b.. occ_a.. roles[../../..]`) and the TIMEOUT dump (`DSWS sensors` line).

**DEVIATION from plan (single-writer sensor, deliberate):** the plan also names a "feed inter-frag"
sensing point. I made the **compute segment boundary the sole writer** of occ_a/occ_b. Having both the
producer (feed) and consumer (compute) write the same occ slot would race and corrupt the very
oscillation T2.4 is meant to validate. Feed-side pressure reads belong in Phase 3, where the feed
actually *actuates* on them (conversion trigger). occ_a/occ_b here are the clean consumer-side signal.

**Offline gates (all green):** 6 cells assemble; RGA 0-spill, `USED_LDS=32768`, livereg **82**;
DSWS=0 still **1716B byte-identical**; `occ_dispatch` rebuilds clean (rc=0, 0 errors). Everything
UNCOMMITTED (commit-on-ask). **NEXT = T2.4 supervised dispatch (STOP for user go).**

---

## BUG (found at T2.4, fixed offline) — multi-tile A-feed starvation — 2026-06-30

**Surfaced by:** the first MULTI-tile DSWS dispatch ever (T2.4 at 32 tiles, `DSWS_ORACLE_MTL=4 NTL=8`).
Gate 1 and the first T2.4 run were all **single-tile** (`MTL=NTL=1`), so the per-tile claim/feed loop
was never exercised. Single-tile = CLEAN; 32-tile = **COOP TIMEOUT** (`claim=2`, compute wedged into tile 1).

**DIAGFINE evidence (the localization):** feed published B to step 35 (into tile 1 fine); **A-feed
`publish[31]=32`** — stuck at exactly one tile's worth (KT=32); compute `prodwait=32 consRel=33`, wedged
mid tile-1 K-loop. `roles[4/2/2]` streamed correctly (init round-trips); `occ_b/occ_a=0` only because
no compute segment ever *completed*. → compute is **A-starved**.

**Root cause (confirmed by code, not assumption):** the A-feed published `prod_a[bnd] = k+1` using the
**per-tile** K counter `k` (s26), which **resets to 0 each tile**. The compute consumer's `a_step`
(s54) is **global/cumulative across tiles** (like the B-ring step). Tile 0: both 0..31 → match → CLEAN.
Tile ≥1: compute's `a_step` is 32→63 and waits `prod_a>32`, but the A-feed re-publishes `prod_a=1..32`
→ never exceeds 32 → permanent A-starvation deadlock. The B-feed was correct (global continuous step);
the A-feed was the lone asymmetry. **Pre-existing** in the Phase-1 K-outer A-feed rewrite — NOT caused
by the Phase-2 sensing (compute wedges before reaching the boundary sensor).

**Fix:** introduce a **global cumulative A-step `s60` (gk)** in the A-feed, used for the ring protocol
(publish count, ring slot index, slot-free gate vs `cons_a`) — mirroring the B-feed and matching
compute's global `s54`. The per-tile `k` (s26) is kept for **A-matrix addressing only** (`k*16` within
the tile's K). `s60` inits to 0 once before `.Lafeed_loop`, increments per K-step, never resets.
(Note: first picked `s49` — caught in review that it's the `lds_get/lds_put` exec-save scratch; moved
to `s60`, 0 refs kernel-wide.)

**Offline gates (green):** 6 bins assemble; RGA 0-spill, LDS 32768, livereg 82; DSWS=0 byte-identical
(1716B). **Needs a supervised MULTI-tile re-dispatch to verify the deadlock is gone** (this is also the
real T2.4 sensor-oscillation run: 32 segments → occ_a/occ_b vary across the 200ms stream). Uncommitted.

---

### ✅ GATE T2.4 PASSED — sensors report sane + multi-tile clean — 2026-06-30

Four supervised dispatches (static DYN=0, pool=1, compositor-safe-chunked, ZERO bricks):
1. single-tile 4c2a2b → `oracle CLEAN ok=32 bad=0` (sensing code correct/safe; too fast to read sensors).
2. **32-tile → COOP TIMEOUT** → surfaced + root-caused the multi-tile A-feed starvation bug (above).
3. 32-tile, A-feed fix → `oracle CLEAN ok=1024 bad=0` (deadlock GONE) but occ read 0 (boundary-drained).
4. 32-tile, sensors relocated to consume points → **`oracle CLEAN ok=1024 bad=0`, `occ_b=1 occ_a=1`,
   `roles[4/2/2]`** — sensors read real in-range backlog, NOT pinned. **PASS.**

**Sensor-placement refinement (banked design note for Phase 3):** occ_a/occ_b are sampled at the per-K
CONSUME points (where the gate `prod>cons` guarantees occ ∈ [1,RINGD]), NOT at the segment boundary
(where the ring has drained → occ≈0, which would make the controller read "always starved"). The
segment boundary remains only the `segcnt` decision-clock tick. The eventual watermark LOW/HIGH must be
read against this consume-point occupancy, and likely wants a per-segment max/avg, not a single sample.

**Phase 2 status: COMPLETE** (T2.1 LDS state, T2.2 control model, T2.3 init+sensing, T2.4 supervised
gate). Plus an out-of-band WIN: the **multi-tile substrate is now proven** (ok=1024) — Gate 1 had only
ever tested single-tile, so this de-risks all of Phase 3 (real GEMMs are multi-tile). Everything static
(DYN=0), UNCOMMITTED. Open items unchanged: pool>1 cross-WG (separate), dyn-VGPR arm (Phase 3).

---

## Phase 3 — Conversion + epoch/ticket + role-agnostic terminal (offline)

### T3.1 — epoch + lock-free gate-CAS ticket (no actuation) — 2026-06-30
`try_gate dir, swin` macro transcribes dsws_ctrl_model.cpp `epoch_of` + `gate_try_win` EXACTLY:
`E = segcnt >> EPOCH_SHIFT`; read `g = gate[dir]`; lose if `g>=E`; else lane0 does an LDS compare-swap
`ds_cmpstore_rtn_b32 v6,v5,v6(=g),v7(=E)` and WINS iff returned-old == g. Single winner per (dir,epoch)
across all racing waves (the LDS atomic serializes). New defsyms `EPOCH_SHIFT=3`, `CTRL_LOW=1`,
`CTRL_HIGH=RINGD-1` (mirror the harness DSWS_* env). New `mark_inc off` macro = lane0 atomic-add counter.
**Unit call site** (compute boundary, ALL compute waves, NO actuation): race `try_gate 0` (compute→Afeed)
each segment, atomic-inc would-win counter `occ[39]`. occ_dispatch decodes `gateWin[39]`.

**CAS operand-order note:** RDNA4 `ds_cmpstore_rtn_b32` is `(vaddr, vCMP, vDATA)` → `MEM=(MEM==vCMP)?vDATA:MEM`
(GCN's order was flipped). Assumed cmp=g, data=E. **Validated by:** the T3.1 micro-check (would-win must be
≤1 per epoch per dir — a wrong order gives double-winners or none) and the Gate-2 conversion-storm oracle.

**Offline gates (green):** 6 bins assemble; RGA 0-spill, LDS 32768, livereg 83; DSWS=0 byte-identical
(1716B); occ_dispatch rebuilds rc=0. UNCOMMITTED. **Optional next:** T3.1 Step-3 supervised micro-check
(1 streamed dispatch; confirms `gateWin[39] ≈ #epochs`, NOT NCOMP×#epochs) — validates the gate-CAS before
3.2/3.3 build conversions on it. Else defer to Gate 2 (T3.5).

### ✅ T3.1 micro-check PASSED (caught + fixed a gate-CAS operand-order bug) — 2026-06-30
First micro-check run: `gateWin[39]=3` — ANOMALOUS (last chunk reaches only epoch E=1 once, so a correct
single-winner CAS gives exactly 1). Diagnosis: my `ds_cmpstore_rtn_b32` operand order was BACKWARDS.
Verified authoritatively OFFLINE via LLVM `cmpxchg` lowering (`clang -S` of an IR `cmpxchg ptr,%cmp,%new`):
`ds_cmpstore_rtn_b32 vdst,vaddr,vsrc0,vsrc1` is `MEM=(MEM==vsrc1)?vsrc0:MEM` -> **vsrc0=NEW, vsrc1=CMP**
(the GCN order, NOT flipped as I'd assumed). The swapped form compared against E and stored g, so gate
stayed 0 and `old==g` held for every racer -> all ~NCOMP waves "won" each epoch. FIX: swap the two source
operands (`... v7(=E,new), v6(=g,cmp)`). Re-dispatch (same config, kernel-only change): **`gateWin 3 -> 1`**,
oracle CLEAN ok=1024 bad=0, no brick. Single-winner ticket PROVEN. The gate-CAS is correct for 3.2/3.3.

---

## SUBSTRATE v2 — claim-based work decomposition + split-K (2026-06-30)

**Why v2:** Phase-3 review (3/3 consensus: kmbandy + Claude + Codex) found the proven coop substrate binds
the matrix WORK decomposition to compile-time role counts + wave identity (`rowblk=trow*P+cid`, A-band 1:1
`cid` pairing, B-frag `owner=ni%NBFEED`). Naive role conversion therefore BRICKS (hang fires before any
wrong-output). Fix = decouple work from identity: make it CLAIMABLE by whoever holds the role. Split-K
folded in (makes B resident-per-segment → replay free → rowblk-count decouples from live nComp). New design
in `SPEC_DSWS_SUBSTRATE_V2.md`; plan in `PLAN_DSWS_SUBSTRATE_V2.md`. New kernel `occ_kernel_dsws.s` (coop
kernel NEVER touched). Config: G=6 SEGK=64 FM=2 FN=4, 8-wave WG, LDS 16640B.

**Model/workflow:** Sonnet 5 implements; Opus + Codex review; kmbandy greenlights every GPU dispatch.

**Phase A offline — DONE, oracle/RGA-gated (no GPU yet):**
- A1 scaffold + v2 LDS layout + harness `--dsws2` dry-print + `mk2` build.
- A2 tiered oracle `oracle_compare` (fp8_oracle.cpp) — TIGHT{5e-3,1e-2} for n_kseg=1, LOOSE{3e-2,2e-2}
  for n_kseg>1 (the CPU wmma_ref chain is not bit-identical to GPU WMMA, so "exact" = the proven tight
  tolerance, not bit-match). Self-test passes.
- A3–A7 datapath (claimer + resident B/A feeds + compute w/ `global_atomic_add_f32` partial-flush +
  completion handshake + role-agnostic sentinel terminal). `global_atomic_add_f32` ENCODES on gfx1201.
- Host launch `run_dsws2` + tiered oracle wired.

**Round-table caught 5 brick/correctness bugs OFFLINE (all fixed + re-gated: ASSEMBLE_OK, RGA
SGPR/VGPR_SPILLS=0, mk2 bin 4840B, harness compiles, dry-print clean):**
1. [Codex] Reset/quiesce straggler race — a compute wave increments ROWBLK_DONE then loops to claim; if
   descheduled, the claimer resets ROWBLK_NEXT and the straggler claims row 0 of the next super-tile with
   stale state. FIX: claimer advance-gate also drains the CLAIM counters (ROWBLK_NEXT≥G+NCOMP,
   BFRAG_NEXT≥FN+NBFEED, AROW_NEXT≥G+NAFEED). [BRICK]
2. [Codex] n_kseg=1 magic-div overflow (ceil(2^32/1) truncates to 0 → t=0 all sti). FIX: shift/mask decode
   (`ksi=sti&mask, t=sti>>shift`, shift=ctz(n_kseg)) — handles n_kseg=1 for free. [BRICK]
3. [Sonnet flagged→Opus→Codex corrected] kernargs s16/s17 undeliverable (PM4 only defines
   COMPUTE_USER_DATA_0..15; all proven paths use 15) AND s16 doubly-used as the per-chunk terminal. Opus's
   first fix was INCOMPLETE (dropped the chunk terminal) — Codex caught it. FINAL: 15 kernargs (s0..s14,
   USER_SGPR=15); n_kseg derived in-kernel (KT>>NKSEG_SHIFT); chunk terminal memory-carried in occ[24];
   claimer publishes SENTINEL 0xFFFFFFFF at terminal, followers retire on it. [BRICK]
4. [Opus + Codex] C body never memset=0 (only the canary) → atomic-adds onto garbage. FIX: memset(C,0)
   once before the chunk loop (not per-chunk, so split-K accumulates across chunks). [wrong-oracle]
5. [Codex] DSWS2 ignored positional mix arg + no role-floor/32-bit-overflow validation. FIXED. [correctness]

**Phase-B carry-forward:** the quiesce sentinels use compile-time NCOMP/NAFEED/NBFEED (correct for STATIC
roles only); Phase-B conversion must switch to live role counts / epoch-snapshot drained counters. (In-code
note at `.Lclaimer_wait_done`.)

**NEXT = A8 [SUPERVISED GPU — kmbandy greenlights], morning 2026-07-01.** First cell (n_kseg=1 TIGHT/exact,
compositor-safe): `ML8_POOL=1 ML8_COOP_CHUNK=8 ML8_COOP_CHUNK_MAXS=0.75 ML8_COOP_STREAM=1 DSWS2_NKSEG=1
timeout 30 ./occ_dispatch --dsws2 4c2a2b` → expect ok=32 bad=0, occ[0]=0 clean, fence FIRED, no brick
(TOTAL_super=32 → 4 chunks of 8). Then n_kseg=8 LOOSE + mixes 6c1a1b/2c3a3b. Gate: static substrate
oracle-green both tiers before Phase B (conversion). All uncommitted. KG: b8c689cc (A8-ready contract),
48625333 (kernel round-table), 86e33108 (the blocker + 3/3 consensus).
