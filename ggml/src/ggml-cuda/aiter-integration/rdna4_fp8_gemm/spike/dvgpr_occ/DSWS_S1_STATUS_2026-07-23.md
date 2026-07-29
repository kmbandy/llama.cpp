# DSWS S1 — Status (2026-07-23 EOD)

**Project:** MAD-305 DSWS (Dynamic Staggered Wave-Spec) fp8 GEMM, gfx1201/RDNA4, R9700.
**Kernel:** `occ_kernel_dsws_flow.s`. **Config of record:** A1 (see `HARNESS.md`).
**This doc = where we are.** Morning pickup plan is in `DSWS_BRIEF_2026-07-24_AM.md`.

---

## TL;DR

The frontier-advance wall is now **measured end-to-end and understood**: the ~2600-tick/advance
interval is **~10% serial advance-mechanism work + ~90% idle-waiting on the ASSIGN publish rate**.
The advance *mechanism* is cheap (~264 ticks). The real wall is **upstream starvation** — the 30 lean
waves drain the ASSIGN frontier faster than it is republished. This **definitively closes** the entire
boundary-election line (herd → funnel → read-spin), all of which only touch that 10%.

**Next:** understand *why* ASSIGN can't keep 30 waves fed (offline source read of the ASSIGN-publish
path), before touching any lever. Do **not** jump to SEGK yet.

---

## The measurement chain (how we got here, 2026-07-23)

1. **RCONV shipped — the one real win.** Runtime role conversion (`DSWS2_RCONV=1`): a starved compute
   wave rewrites its own `ROLE[wid]=AFEED` after 64 consecutive coasts. On silicon this collapsed
   **RING_WAIT 56% → 0.3%** and doubled **WMMA 19% → 45%**. Correct + adaptive (river-ethos aligned).

2. **The wall moved to SS_WAIT (50.7%).** BNDSPLIT (`occ[127-130]`) localized it: **93.1% of boundary
   entries LOSE the ZLOCK election CAS** (thundering herd), DRAINGATE_BAIL 0%, CSTOREGATE_BAIL 6.8%,
   ADVANCE 0.1%. ~930 waves storm the single election per advance.

3. **The boundary-election line was fully explored and CLOSED as a dead end:**
   - **FUNNEL** (`DSWS2_FUNNEL`): a read-only readiness pre-gate before the election (no-bail by
     construction; CSTOREGATE_BAIL → 0 on GPU). **Correct but ~1.6× SLOWER** than the herd —
     structurally, because it removes the herd's incidental ZLOCK serialization.
   - **Read-spin** (`DSWS2_FUNNEL_SPIN_N`): bounded poising to keep waves responsive. **FALSIFIED** the
     per-advance-latency theory — poising made it *worse* (285M vs flow-off 270M vs herd 166M ticks/rep).
   - Both herd and funnel are **100% ASSIGN-bound** (door1 NOTHING-STAGED = 100%). Election tuning
     cannot touch the wall.

4. **ADVPROBE (`DSWS2_ADVPROBE`) — the decisive instrument.** Throttled-RTC timer of the ZLOCK critical
   section (election win → `DA_ZDONE` release write). Built by Codex terra, offline-verified by me
   (0-spill, byte-identity, `s71` 1/64 throttle, win→advance pairing). **Result on the full ml8 shape:**

   | Component | Ticks | Share |
   |---|---|---|
   | Serial advance mechanism (win → drain-gate → zero_banks → rebase → `DA_ZDONE` release) | ~264 | ~10% |
   | Idle-waiting (boundary claimed, nothing ready to advance) | ~2336 | ~90% |

   `ticks/advance = 264.5` (occ[131]=15076 / occ[132]=57, n=57 throttled samples). Full-population
   corroboration (un-throttled, trustworthy):
   - **STARVATION = 211,955,610** feed-path iters with an **empty ASSIGN frontier** (100% → ASSIGN-BOUND)
   - **door1 NOTHING-STAGED = 100.0% of coast**; **coast-frac = 95.1%**
   - **door4 GROW-FAIL = 0%** — the dyn-VGPR moat NEVER engages; VGPR headroom left, no work to spend it on
   - **CARRIER STALL = 0** — carriers are fed; the stall is not there
   - oracle **CLEAN** (ok=76032 bad=0, all 3168 tiles); **WORK-EXACT** (computed=190080); emissions occ[96]=63360

---

## Proven / Dead / Open

**PROVEN**
- RCONV is the shipped win (RING_WAIT 56→0.3, WMMA 19→45), correct + adaptive.
- The frontier advance MECHANISM is cheap (~264 ticks, ~10% of the interval) — **not** the wall.
- The wall is **ASSIGN-publish-rate starvation** (~90% idle, 100% empty-frontier feed iters).
- The dyn-VGPR moat has headroom to spare (GROW-FAIL 0%) — the kernel is nowhere near VGPR-bound.

**DEAD (do not revisit)**
- The whole boundary-election line: herd tuning, the FUNNEL readiness pre-gate, and read-spin poising.
  All only touch the 10% mechanism. FUNNEL is structurally 1.6× slower; read-spin falsified the latency
  theory. `DESIGN_BOUNDARY_FUNNEL_2026-07-23.md` documents the funnel design (now closed).

**OPEN (morning)**
- *Why* can't the ASSIGN frontier keep 30 lean waves fed? Read the ASSIGN-publish path
  (CFASSIGN/DECENTASN advance of `ASSIGN_HEAD`): what it's serialized behind, and whether the 10 split-K
  reduction boundaries per super-tile are the natural cadence limit or an artificial one.
- SEGK is the brief-flagged lever (at SEGK=256, K=2560 → n_kseg=10 reduction boundaries/super-tile), but
  **understand-before-acting**: do not build a SEGK change until the publish path is read.

---

## Instruments (all defsym-gated; `DSWS2_x=0` byte-identical to baseline `cac3ff7c`)

| defsym | occ slots | what it measures | state |
|---|---|---|---|
| `DSWS2_RCONV` (COAST_N=64) | occ[48] | runtime role conversion (SHIPPED WIN) | on in prod runs |
| `BNDSPLIT` | occ[127-130] | 4-way boundary split (herd/draingate/cstore/advance) | localized the herd |
| `DSWS2_FUNNEL` (+`_SPIN_N`) | — | readiness pre-gate + read-spin | DEAD END |
| `DSWS2_ADVPROBE` | occ[131-132] | ZLOCK critical-section ticks/advance (s71-throttled RTC) | **result banked** |

Bin shas: baseline `cac3ff7c` (all off) / ADVPROBE-on `3280ef6d`. Disk currently holds **baseline**.

---

## Tree / harness state

- **Tree:** clean (baseline bin restored). My working changes (`build_flow.sh`, `occ_dispatch.cpp`,
  `occ_kernel_dsws_flow.s`, deleted stale scripts) are **unstaged** — the git tree is SHARED with a live
  weight-pager session (`arg.cpp` / `router-fleet-main.ini` / `server-models.cpp` are theirs). **Stage nothing.**
- **Harness:** `build_flow.sh` builds → `gpu_run.sh` dispatches. `dsws_realshape_bench.py live` = all-shape
  sweep. See `HARNESS.md` §34 for the canonical **dispatch** env (distinct from build defsyms — see the
  gotcha below).
- **Card:** free. **Latch:** clear.

## KEY HARNESS GOTCHA (cost a full-stop today)

The `gpu_run.sh` **dispatch** env var names are DIFFERENT from the `build_flow.sh` **build** defsym names.
Passing build-names as runtime env silently falls through to the host default small oracle → geometry
mismatch → pre-live timeout. Canonical A1 flow dispatch (host `getenv` names):
`DSWS2_FLOW=1` (REQUIRED — selects the flow path), `FLOW_WAVES=30`, `DSWS2_FM=1`, `DSWS2_G=6`,
`DSWS2_ACC_N=3`, `FLOW_POOL_N=1`, `DSWS2_SEGK=256`, `DSWS2_K=2560`, `DSWS2_ORACLE_MTL=22`,
`DSWS2_ORACLE_NTL=144` (MTL/NTL set the full ml8 shape 2112×9216×2560), `DSWS2_TARGET_SECS=<s>`,
`ML8_COOP_CHUNK=96`. RCONV/ADVPROBE are BUILD defsyms baked into the bin — NOT passed at dispatch.
