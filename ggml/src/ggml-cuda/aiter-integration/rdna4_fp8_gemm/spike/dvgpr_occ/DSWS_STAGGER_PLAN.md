# DSWS Stagger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (inline, checkpoint-gated) to implement this plan. Steps use checkbox (`- [ ]`) syntax. **Every GPU run is a hard checkpoint requiring kmbandy's individual greenlight — a subagent MUST NOT dispatch `./occ_dispatch` on its own.**

**Goal:** Build the staggered traveling-peak occupancy layer on the write-once-C foundation, inside `occ_kernel_dsws_flow.s`, converting the split-K atomic-flush compute path into a WG-local per-rowblk LDS reduction with self-maintaining stagger.

**Architecture:** Three incremental stages, each ending in an oracle-verified GPU run. Stage 1 changes the *write path* (segment atomic-flush → per-rowblk `ds_add_f32` LDS accumulator → single `global_store`) with no stagger. Stage 2 adds *per-burst* grow/shrink + coast-on-grow-fail + a launch-offset warm-up + instrumentation. Stage 3 tunes the equilibrium (burst length, coast granularity).

**Tech Stack:** GCN assembly (gfx1201, wave32), raw PM4/KFD dispatch via `occ_dispatch.cpp`, `/opt/rocm/llvm/bin` toolchain, oracle-gated correctness.

**Spec:** `DSWS_STAGGER_DESIGN.md` (same directory). Read it before starting.

## Global Constraints

- Target `amdgcn-amd-amdhsa` `-mcpu=gfx1201`, wave32. Build via `build_flow.sh` (defsym passthrough).
- **0-spill is mandatory** — any `scratch`/spill reference is a build failure.
- **LDS ≤ 65536 B/WG.** The kernel already asserts `LDS_TOTAL_FLOW > 65536 → .error`; keep it.
- **Oracle gate every run:** `bad=0`, and for write-once the tight tier (`rel 5e-3 / abs 1e-2`). A run with `bad>0` fails the task.
- **Safety:** `DEADMAN=1` on every run; SAFEPROBE + bounds gate on; never `--gl2c`; logs to real disk `/home/kmbandy/dsws_gpu_logs/`, never tmpfs. Single-shot before sustained.
- **GPU runs are individually greenlit by kmbandy.** Offline assemble/spill/LDS checks are free; dispatches are not.
- **Commit only when kmbandy explicitly asks.** Do not `git commit` as a plan step unless told.
- **Hold tile size fixed at FM=2 FN=4** — tile size is the *next* foundational step, out of scope here.
- Baseline to beat: **write-once grind = 1.5 TF @ pool 64, 576×512×2048** (occupancy 0.5 waves/SIMD). Split-K baseline ~0.7 TF.

---

## File Structure

- `occ_kernel_dsws_flow.s` — the kernel. All mechanism changes land here.
- `occ_dispatch.cpp` — host. Only the flow-path LDS-byte computation (~line 1803 / ~6004) and, if added, instrumentation readout in `run_dsws2`.
- `build_flow.sh` — add any new defsyms (`ACC_N`, `BURST`, `COAST_GRAN`, `STAGGER_WARMUP`, `STAGINSTR`).
- `DSWS_STAGGER_DESIGN.md` — reference; update its "build status" as stages land.

---

## Stage 1 — Write-once path into the economy (no stagger yet)

### Task 1: LDS accumulator pool

**Files:**
- Modify: `occ_kernel_dsws_flow.s` (LDS layout block, ~lines 318–338)
- Modify: `occ_dispatch.cpp` (flow LDS-byte calc, ~1803 and ~6004)
- Modify: `build_flow.sh` (add `ACC_N` defsym)

**Interfaces:**
- Produces: `ACC_BASE` (LDS byte offset), `ACC_STRIDE = 8192` (FM*FN*1024), `ACC_N` (defsym, default 2), macro `acc_of slot → acc_slot` mapping a live rowblk to one of `ACC_N` accumulator banks, and `ACC_TOTAL = ACC_N*8192`.

- [ ] **Step 1:** Add defsym `ACC_N` (default 2) in `build_flow.sh` and a matching `.ifndef ACC_N / .set ACC_N, 2` in the kernel near the other pool defsyms.
- [ ] **Step 2:** In the LDS layout block, place the accumulator pool after the operand pool: `.set ACC_BASE, (OP_BASE + POOL_N*OPSTRIDE)` and `.set ACC_STRIDE, (FM*FN*1024)` (=8192) and `.set LDS_TOTAL_FLOW, (ACC_BASE + ACC_N*ACC_STRIDE)`. Keep the existing `.if LDS_TOTAL_FLOW > 65536 → .error`.
- [ ] **Step 3:** Set `POOL_N=2, ACC_N=2` as the default co-budget (`256 + 2*16384 + 2*8192 = 49408 ≤ 65536`). Leave `POOL_N=3, ACC_N=1` (`57600`) as a documented alternate.
- [ ] **Step 4:** In `occ_dispatch.cpp`, change the flow-path `ldsBytesRaw` to `256 + poolSlots*operandBytes + accN*8192` so the host requests the matching group-segment size. Read `accN` from env `DSWS2_ACC_N` (default 2). Both call sites (~1803, ~6004).
- [ ] **Step 5 (offline gate):** `POOL_N=2 ACC_N=2 ./build_flow.sh 4 2 2` → assemble clean, **0 spill** (`llvm-objdump -d … | grep -c scratch` = 0), no LDS `.error`. Record the `.text` byte size.

_No GPU run this task — layout only._

### Task 2: Per-rowblk LDS reduction (segment atomic-flush → `ds_add_f32` + single store)

**Files:**
- Modify: `occ_kernel_dsws_flow.s` — coordinator ASSIGN (work-unit granularity), slot fields, `DECODE_STI`, `.Lflow_compute` body (~1441–1537).

**Interfaces:**
- Consumes: `ACC_BASE/ACC_STRIDE/ACC_N/acc_of` (Task 1).
- Produces: a compute path where all `n_kseg` K-segments of a rowblk `(mblk,tcol,r)` reduce into one LDS accumulator bank and the completer writes C once.

**Design (this is a work-model change, implement to these invariants — not a copy-paste):**

The current work unit is a super-tile `(mblk,tcol,ksi)` whose G rowblks each `global_atomic_add_f32`-flush. Change the unit so a **rowblk's `n_kseg` segments are WG-local-reduced**:

- **Pin:** the coordinator's ASSIGN hands a slot a rowblk-group for a fixed `(mblk,tcol)` and the slot walks `ksi = 0..n_kseg-1` as its segment work-list (the existing `SL_STI` STAMP carries `(mblk,tcol)`; `ksi` becomes an internal per-slot counter, not part of the claimed unit).
- **Accumulator bind:** on a slot beginning a rowblk `r`, bind an accumulator bank `acc_of(slot)` and **zero** its `ACC_STRIDE` bytes (or mark first-segment-writes-instead-of-adds).
- **Per segment (the burst):** compute wave does `KSEG_STEPS` WMMA into VGPR acc (unchanged 1454–1495), then instead of the `global_atomic_add_f32` block (1504–1518), `ds_add_f32` each of the `FM*FN*8` accumulator elements into `ACC_BASE + acc_slot*ACC_STRIDE + frag*1024 + e*4`. Then `s_wait_dscnt 0x0` before advancing the segment done-counter.
- **Completer:** when a rowblk's segment done-counter reaches `n_kseg`, exactly one wave `global_store`s the `FM*FN` fragments from the LDS accumulator bank to `C` at the address the current code already computes (1496–1503), then the bank is recycled. **Ordering invariant (Codex):** the completer must observe completion only after all prior segments `s_wait_dscnt 0` — do not let the done-increment race the DS adds.
- **Invariant carried from spec §4:** rowblk/segment claims still happen in the compute body (after any grow), so this task does not touch grow ordering.

- [ ] **Step 1:** Introduce a per-slot `SL_KSI` counter and a per-slot/per-rowblk `SL_SEGDONE` counter in the slot control block (extend `SLOTC_STRIDE` if needed; keep `SLOTC_BASE + POOL_N*SLOTC_STRIDE ≤ OP_BASE`, the existing assert).
- [ ] **Step 2:** Add the accumulator-zero on rowblk start (a `ds_store` loop over `ACC_STRIDE`, or a first-segment write path).
- [ ] **Step 3:** Replace the `.if NOCFLUSH==0 … global_atomic_add_f32 …` flush (1504–1518) with the `ds_add_f32` reduction into `acc_of(slot)`, followed by `s_wait_dscnt 0x0`.
- [ ] **Step 4:** Add the completer branch: on `SL_SEGDONE == n_kseg`, single `global_store` of the accumulator bank to C, then reset the bank/counters for reuse and advance the DRAIN frontier.
- [ ] **Step 5:** Update the coordinator ASSIGN + `DECODE_STI` so `ksi` is derived per-slot internally rather than from the claimed unit.
- [ ] **Step 6 (offline gate):** `POOL_N=2 ACC_N=2 ./build_flow.sh 4 2 2` → clean, 0-spill, no LDS error.

### Task 3: Oracle-verify write-once inside the economy (CHECKPOINT — greenlit)

**Files:** none (run only).

- [ ] **Step 1:** Confirm host build current (`ls -la occ_dispatch`), flow bin built from Task 2.
- [ ] **Step 2 (GREENLIT GPU RUN):** `DSWS2_FLOW=1 ML8_POOL=64 <shape 576×512×2048> ./occ_dispatch --dsws2 4c2a2b 2>&1 | tee /home/kmbandy/dsws_gpu_logs/dsws2_flow_wo_s1.log`
- [ ] **Step 3 (gate):** oracle `bad=0`, `max_rel` within tight tier, `occ[0]=0` (clean drain), no wedge. TF should land near the grind's write-once (~1.5 TF) — it will not yet stagger, so do not expect more. **A regression below split-K's ~0.7 or any `bad>0` fails the task.**
- [ ] **Step 4:** Record TF + oracle in the design doc's build-status. If green, Stage 1 done: write-once now lives in the multi-wave economy.

---

## Stage 2 — Per-burst grow/shrink + coast-on-fail + instrumentation

### Task 4: Instrumentation (land it BEFORE the stagger so we can see it work)

**Files:**
- Modify: `occ_kernel_dsws_flow.s` (new `STAGINSTR`-gated atomics + occ offsets), `occ_dispatch.cpp` (`run_dsws2` readout), `build_flow.sh` (`STAGINSTR` defsym, default 0).

**Interfaces:**
- Produces occ counters: `FATNOW` (live concurrent-fat), `FATPEAK` (`atomic_max`), `FATACC` (time-integral or sample sum), `GROWFAIL` (SCC0 count), `COASTCYC` (cycles coasted before a successful grow), `DUTYFAT`/`DUTYTOT` (fat vs total cycles). All `.if STAGINSTR` gated → zero bytes when off.

- [ ] **Step 1:** Reserve occ slots for the six counters (past the existing `PH_*`/`ALLLIVE` block); document offsets.
- [ ] **Step 2:** On a successful grow: `global_atomic_add_u32 FATNOW,+1`; `global_atomic_max_u32 FATPEAK,FATNOW`. On shrink: `FATNOW,-1`. All `.if STAGINSTR`.
- [ ] **Step 3:** On grow SCC0: `GROWFAIL,+1`; accumulate coasted RTC into `COASTCYC` (reuse the deadman RTC read pattern).
- [ ] **Step 4:** Duty: stamp RTC at grow and shrink; add the fat span to `DUTYFAT`, total lap span to `DUTYTOT`.
- [ ] **Step 5:** `occ_dispatch.cpp` `run_dsws2`: print `FATPEAK`, `FATACC/laps`, `GROWFAIL`, mean `COASTCYC`, `DUTYFAT/DUTYTOT` when `STAGINSTR`.
- [ ] **Step 6 (offline gate):** build with `STAGINSTR=0` → byte-identical to pre-Task-4 bin (proves gating is clean); build `STAGINSTR=1` → 0-spill.

### Task 5: Per-burst grow/shrink + coast-on-grow-fail

**Files:** Modify `occ_kernel_dsws_flow.s` — the grow macro (`conv_apply` / `.Lca_alloc` ~1039–1044, `.Lflow_grow` ~1417) and the compute burst loop.

**Interfaces:**
- Consumes: Task 4 counters.
- Produces: a compute wave that grows per burst, and on grow SCC0 **coasts** (branch to `.Lflow_coast`) instead of spinning.

**Design (spec §4 — the load-bearing invariant is "commit nothing until grow succeeds"):**

- [ ] **Step 1:** Move grow/shrink from role-epoch to **per-burst**: the compute wave, at the top of each segment burst, is lean; it `try-grow`s, does the segment, `ds_add`s, then `shrink`s back to lean. `BURST` defsym (default 1 = one K-segment per grow) sets how many segments per grow before shrink.
- [ ] **Step 2:** Replace the grow spin `s_cbranch_scc0 .Lca_alloc` with **coast**: `s_cbranch_scc0 .Lflow_coast` (SCC0 = budget full → coast lean). Ensure the rowblk/segment **claim happens only after** SCC1 (grow success) — audit that no `lds_fetch_add`/counter write precedes the grow on the burst path.
- [ ] **Step 3:** Confirm coast returns to the loop head (`.Lflow_loop`) where `deadman_check` runs — this is what closes the grow-spin brick gap. Verify no path reaches `s_alloc_vgpr` retry without a loop-head pass.
- [ ] **Step 4 (offline gate):** `BURST=1 STAGINSTR=1 POOL_N=2 ACC_N=2 ./build_flow.sh 4 2 2` → clean, 0-spill.

### Task 6: Launch-offset warm-up

**Files:** Modify `occ_kernel_dsws_flow.s` — entry, after `.Lflow_alloc`.

- [ ] **Step 1:** Add `STAGGER_WARMUP` defsym (default 1). When set, compute-class wave `wid` delays its first grow by `wid * WARMUP_STEP` cycles via `s_sleep`/RTC-spin (spread initial phases). Warm-up only — steady state is maintained by the gate.
- [ ] **Step 2 (offline gate):** build clean, 0-spill.

### Task 7: Stage-2 verify (CHECKPOINT — greenlit, single-shot then sustained)

- [ ] **Step 1 (GREENLIT single-shot):** `DSWS2_FLOW=1 STAGINSTR=1 ML8_POOL=64 576×512×2048 ./occ_dispatch --dsws2 4c2a2b | tee /home/kmbandy/dsws_gpu_logs/dsws2_flow_stag_s2.log`
- [ ] **Step 2 (gate — correctness + safety):** `bad=0`, `occ[0]=0`, no wedge. Instrumentation must show `FATPEAK > 1` and a real duty cycle (`DUTYFAT/DUTYTOT` between ~0.3 and ~0.9) — i.e., the stagger is actually forming, not lockstep (peak-then-zero) or square-wave (duty ≈ 1).
- [ ] **Step 3 (GREENLIT sustained):** re-run with `DSWS2_TARGET_SECS=2` → deadman-clean, no wedge across the sweep (proves the brick gap is closed). Record `GROWFAIL` rate + mean `COASTCYC` (the handoff gap).
- [ ] **Step 4:** Record all counters + TF in the design doc build-status. Gate to Stage 3: no wedge, `FATPEAK>1`, healthy duty.

---

## Stage 3 — Tune the equilibrium

### Task 8: Burst-length × coast-granularity sweep (CHECKPOINT — greenlit)

**Files:** Modify `occ_kernel_dsws_flow.s` (add `COAST_GRAN` defsym = feed units per grow-retry), `build_flow.sh`.

- [ ] **Step 1:** Add `COAST_GRAN` defsym (default 1 = one feed unit per retry). Wire the coast fallback to do `COAST_GRAN` feed units then retry grow.
- [ ] **Step 2 (offline):** build the sweep matrix: `BURST ∈ {1,2,4}` × `COAST_GRAN ∈ {1,2}` × `POOL_N/ACC_N ∈ {2/2, 3/1}`, all `STAGINSTR=1`, 0-spill each.
- [ ] **Step 3 (GREENLIT sweep):** run each at 576×512×2048, ML8_POOL swept {32,64,128}, logs to `/home/kmbandy/dsws_gpu_logs/`. Read TF + `FATPEAK`/`FATACC` (concurrent-fat vs budget) + `GROWFAIL` + `COASTCYC` + duty.
- [ ] **Step 4 (gate — SUCCESS CRITERION):** find the config where TF **beats 1.5 TF** with concurrent-fat meaningfully **> 0.5 waves/SIMD** and low thrash (`GROWFAIL` not dominating), oracle-clean. Record the winning `(BURST, COAST_GRAN, POOL_N, ACC_N)`.
- [ ] **Step 5:** If nothing beats 1.5 TF: consult the instrumentation to decide which risk fired — feed-capped (compute waves fat but starved on operands → the lever is the feed, not occupancy), or bad equilibrium (thrash / under-fill → burst/coast dials). Record the diagnosis; do not thrash on fixes — bring the finding back for a decision.

---

## Self-Review (spec coverage)

- **Spec §3 (substrate: split-K bursts → LDS `ds_add_f32` → write-once):** Tasks 1–3. ✓
- **Spec §4 (coast-on-grow-fail, commit-after-grow, closes brick gap):** Task 5. ✓
- **Spec §5 (self-maintaining stagger + launch warm-up + burst knob):** Tasks 5, 6, 8. ✓
- **Spec §6 (accounting consistency under coast):** enforced by Task 5 Step 2 (claim-after-grow) + Task 2 completer ordering. ✓
- **Spec §7 (handoff gap, coast granularity dial, measured):** Task 4 (`COASTCYC`), Task 8 (`COAST_GRAN`). ✓
- **Spec §8 (instrumentation: fat count/peak, grow-fail, coast, duty, confirm B):** Task 4. (Per-SIMD budget B: `FATPEAK` at saturation × peak footprint gives an empirical read — note it in Task 7.) ✓
- **Spec §9 (oracle, deadman, greenlit runs):** every CHECKPOINT task. ✓
- **Spec §10 (3-stage incremental):** stage structure. ✓
- **Spec §11 (risks: feed cap, equilibrium quality, per-SIMD placement, ds_add_f32 hw):** Task 3 (ds_add_f32 hw proof via oracle), Task 8 Step 5 (feed cap / equilibrium diagnosis). ✓
