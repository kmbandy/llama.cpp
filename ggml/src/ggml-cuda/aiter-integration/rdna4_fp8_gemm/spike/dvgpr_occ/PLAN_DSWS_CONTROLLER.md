# DSWS Adaptive Wave-Role Controller — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Domain note:** this is hand-written gfx1201 (RDNA4 wave32) assembly + a raw-PM4 KFD harness + a CPU fp8 oracle — NOT a pytest codebase. "Tests" are: the CPU fp8 e4m3 **oracle gate** (bit/tolerance match of stored C), the offline **RGA gate** (0 spills / 0 scratch + livereg peak), and **supervised GPU** oracle/perf runs. Assembly tasks are **gate-defined**: the step states the precise structural change and the exact gate command + expected output; the gate is the test. Host/logic tasks (oracle harness, CAS single-winner model, LDS sizing, watermark logic) get real code + commands.

**Goal:** A single fp8 GEMM kernel on gfx1201 that senses its in-kernel bottleneck (ring pressure) and rebalances its mix of {compute / A-feed / B-feed} waves at runtime via `s_alloc_vgpr`, beating the 165 TF static winner on ml8 shapes while staying oracle-correct.

**Architecture:** Build on `occ_kernel_coop.s` (dyn-VGPR + split-K + POOLTERM, raw-PM4 dispatch). Add a 3rd wave role (A-feed) to make a static 3-role substrate, then layer a lock-free, barrier-free controller: per-WG LDS role-count slots + ring-occupancy sensing + watermark hysteresis + an epoch-gated single-winner CAS ticket that serializes conversions (≤1 per direction per `2^EPOCH_SHIFT` segments), with sum-envelope reservation enforced at grow-time.

**Tech Stack:** gfx1201 hand-asm (LLVM `clang` assembler), `occ_dispatch.cpp` (libhsakmt KFD PM4 harness), `fp8_oracle.cpp` CPU reference, `rga_check.sh` (Radeon GPU Analyzer), `rocprofv3 --att/--pmc`. Build via `build.sh` / `build_coop.sh` defsym matrices.

## Global Constraints

- **Target:** gfx1201 (AMD R9700 / Navi48, RDNA4, **wave32**). One WMMA = 16×16×16 = 8192 FLOP. fp8 e4m3 inputs, fp32 accumulate. fp8 WMMA ceiling = 307 TF.
- **Dispatch:** raw PM4 on a KFD compute queue only (dyn-VGPR armed via `COMPUTE_PGM_RSRC2` bit 6). Never HIP for the kernel. **Never pass `--gl2c`** (MES-crash landmine). SAFEPROBE clamp + bounds gate + padding stay ON. Every run streams to disk (`ML8_COOP_STREAM=1`).
- **GPU runs are SUPERVISED.** R9700 drives the displays — a hang resets monitors. THE USER greenlights EACH dispatch individually. A hang/timeout = full STOP + report, never auto-fire the next variant. Only sub-second bounded (compositor-safe chunked) dispatches. Do max work OFFLINE first (disasm, RGA, CPU oracle).
- **A GPU brick is a BUG, never a tax.** Freeze dyn dispatch on any build known to leave the queue non-idle.
- **Commits:** per the user's standing rule, **commit only when the user asks.** The `git` steps below are checkpoints — batch them and run when greenlit. Co-author/session trailers per repo convention.
- **Correctness invariant:** DSWS is a pure *performance* transform — the adaptive mix MUST NOT change the math. Every gate is "stored C matches the fp8 oracle for any (shape, mix, conversion schedule)."
- **ml8 bench shapes (always bench these, not throwaway squares):** `down` = M=2048 K=9216 N=2560; `down_pf` = M=512 K=9216 N=2560.
- **Baselines to beat:** static 3-role baseline (Phase 1) and the 165.7 TF 8×2 static winner.

## File Structure

- `occ_kernel_coop.s` — **modify.** The substrate. Add A-feed role + A-ring LDS; add controller state (role slots, gates, reservation, epoch counter); add sensing + conversion at boundaries; make the POOLTERM terminal role-agnostic. All new behavior behind defsyms so `DSWS=0` stays byte-identical to today's proven `coop` d0.
- `occ_dispatch.cpp` — **modify.** New `--dsws` mode: launch N waves, arm dyn-VGPR, program the launch mix, the per-chunk hang-abort, oracle gate, role-count/conversion snapshot decode.
- `fp8_oracle.cpp` / `test_fp8_oracle.cpp` — **reuse**, extend harness only (role-agnostic math).
- `dsws_ctrl_model.cpp` / `test_dsws_ctrl_model.cpp` — **create.** A pure-CPU model of the control-law logic (epoch/gate single-winner, watermark band decision, reservation envelope) so the *logic* is unit-tested offline before it goes into asm.
- `build_dsws.sh` — **create.** Defsym build matrix (NCOMP/NAFEED/NBFEED, RINGD, LOW/HIGH, EPOCH_SHIFT, DSWS, STORE) + oracle + RGA gates.
- `rga_check.sh` — **reuse** (re-point KSRC).
- `RESULT_DSWS.md` — **create** (Phase 4 outcome).

---

## Phase 1 — Static 3-role substrate (Gate 1)

### Task 1.1: Oracle harness accepts a 3-role fixed-mix config

**Files:**
- Modify: `occ_dispatch.cpp` (the `--dsws` arg parse + config struct)
- Reuse: `fp8_oracle.cpp` (math unchanged — roles don't change C = A·B)

**Interfaces:**
- Produces: a `DswsCfg{ uint32_t nComp, nAfeed, nBfeed, ringd, low, high, epochShift; bool dyn; }` parsed from env (`DSWS_NCOMP`, `DSWS_NAFEED`, `DSWS_NBFEED`, `DSWS_RINGD`, `DSWS_LOW`, `DSWS_HIGH`, `DSWS_EPOCHSHIFT`, `DSWS_DYN`) with `N = nComp+nAfeed+nBfeed`.

- [ ] **Step 1: Add the config struct + env parse + a validation refuse-path** (mirror the existing WAVESPEC handler's bin-presence guard at `occ_dispatch.cpp` ~2811). Validate `nComp≥1, nAfeed≥1, nBfeed≥1` and that the requested `_gd` bin for `(N, ringd, dyn)` exists; if not, print refusal + `rc=4`, do NOT dispatch.
- [ ] **Step 2: Gate (host compile + refusal smoke, no GPU)**

Run: `cd spike/dvgpr_occ && clang++ -std=c++17 -O2 -Wall -Wno-unused -I "$PM4/vendor/compat" -I "$PM4/vendor" -I "$PM4" -I /opt/rocm/include occ_dispatch.cpp fp8_oracle.cpp "$PM4/vendor/PM4Packet.cpp" "$PM4/vendor/BasePacket.cpp" /opt/rocm/lib/libhsakmt.a -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch` (PM4=../dvgpr_pm4)
Then: `DSWS_NCOMP=0 ./occ_dispatch --dsws` → Expected: prints refusal (`nComp≥1`), `rc=4`, no dispatch.
- [ ] **Step 3: Commit (when greenlit):** `git add occ_dispatch.cpp; git commit -m "dsws: 3-role fixed-mix config + validation refuse-path"`

### Task 1.2: A-feed role + A-ring in the kernel (static mix)

**Files:**
- Modify: `occ_kernel_coop.s` (LDS layout: add `ARING_OFF`, `PROD_A_OFF`, `CONS_A_OFF`; add the A-feed wave loop ported from `occ_kernel_wggemm2.s` A-LDS-share; switch compute's A source from direct-load to A-ring `ds_load`)

**Interfaces:**
- Produces: defsyms `NCOMP`, `NAFEED`, `NBFEED` (static role counts); `RINGD_A` (A-ring depth, default = `RINGD`); `DSWS` (0 = byte-identical to current coop d0, 1 = 3-role). LDS symbols `ARING_OFF`, `PROD_A_OFF`, `CONS_A_OFF`, `min_cons_a` macro (mirror of B-side `PROD_OFF`/`CONS_OFF`/`min_cons`).

- [ ] **Step 1: Add the A-ring LDS layout + the A-feed wave loop** behind `.if DSWS`. Port the proven cooperative A-fill/share addressing from `occ_kernel_wggemm2.s` (the KWIN A-LDS pattern); A-feed waves `global_load` A → `ds_store` into the A-ring → bump `prod_a`. Compute waves `ds_load` A from the A-ring (gated on `prod_a`) instead of `global_load`-direct. Keep `DSWS=0` path = the existing direct-A code, byte-identical.
- [ ] **Step 2: RGA gate (offline, no GPU)**

Run: `KSRC=occ_kernel_coop.s DEFS="-defsym DSWS=1 -defsym NCOMP=4 -defsym NAFEED=2 -defsym NBFEED=2 -defsym RINGD=2" ./rga_check.sh`
Expected: assembles; **0 VGPR spills / 0 SGPR spills / 0 scratch**; prints livereg peak per role (record `V_fat` = compute peak-live, `V_lean` = feed peak-live — needed in Phase 3).
- [ ] **Step 3: Byte-identity gate for DSWS=0**

Run: build the `DSWS=0` d0 bin and `cmp` against the committed `.clean_bins` coop d0.
Expected: **byte-identical** (the 1716B reference) — proves the A-feed addition is fully gated and the proven substrate is untouched.
- [ ] **Step 4: Commit (when greenlit):** `git add occ_kernel_coop.s; git commit -m "dsws: static A-feed role + A-ring LDS (DSWS=0 byte-identical)"`

### Task 1.3: `--dsws` static dispatch + oracle gate wiring

**Files:**
- Modify: `occ_dispatch.cpp` (the `--dsws` dispatch: WG = `N*32` threads, program role counts into `userdata`, load the `_gd` bin, STORE=1 oracle path)
- Create: `build_dsws.sh` (defsym matrix + oracle + RGA)

**Interfaces:**
- Consumes: `DswsCfg` (1.1), kernel defsyms (1.2).
- Produces: `build_dsws.sh` emitting `occ_dsws_<NCOMP>c<NAFEED>a<NBFEED>b_r<RINGD>[_dyn][_st].bin`; dispatch grid = `pool * N * 32`.

- [ ] **Step 1: Wire the `--dsws` dispatch** (clone the WAVESPEC/coop dispatch path): set `COMPUTE_START_X` dims for `N*32` threads/WG, write `NCOMP/NAFEED/NBFEED` into `userdata` slots the kernel reads at init, load the matching `_gd` bin, run STORE=1 then bit-check vs `fp8_oracle`.
- [ ] **Step 2: `build_dsws.sh` static matrix** — cells `{4c2a2b, 6c1a1b, 2c3a3b}` × `RINGD={2}` × `{st(STORE=1), perf(STORE=0)}`, each oracle-gated and RGA-gated offline.
- [ ] **Step 3: Gate (offline build + RGA, no GPU)**

Run: `./build_dsws.sh static`
Expected: all cells assemble, RGA 0-spill, harness links. No GPU yet.
- [ ] **Step 4: Commit (when greenlit):** `git add occ_dispatch.cpp build_dsws.sh; git commit -m "dsws: static 3-role dispatch + build matrix + oracle wiring"`

### Task 1.4 [SUPERVISED GPU]: Gate 1 — static 3-role oracle-green

**Files:** none (run only).

- [ ] **Step 1: Prepare ONE gated oracle dispatch.** Config `4c2a2b r2 STORE=1`, compositor-safe chunked, `ML8_COOP_STREAM=1`. Present the exact command to the user; **STOP for go.**
- [ ] **Step 2: On greenlight, run Gate 1** across `{down, down_pf, square}` × `{4c2a2b, 6c1a1b, 2c3a3b}`, one dispatch at a time.

Run (example): `ML8_COOP_STREAM=1 DSWS_NCOMP=4 DSWS_NAFEED=2 DSWS_NBFEED=2 DSWS_RINGD=2 WG_M=2048 WG_K=9216 WG_N=2560 timeout 30 ./occ_dispatch --dsws`
Expected: `oracle CLEAN ok=<N> bad=0`, fence FIRED, clean teardown, no dmesg faults, **user-confirmed no brick**.
- [ ] **Step 3:** If any cell bricks/hangs → STOP, capture last stream snapshot, report. Do not proceed to Phase 2 until all Gate-1 cells are clean.
- [ ] **Step 4: Commit the green substrate (when greenlit)** + note RGA `V_fat`/`V_lean`/`BUDGET` numbers in `RESULT_DSWS.md` (created here, appended through Phase 4).

---

## Phase 2 — Sensing + role slots + reservation (no conversions yet)

### Task 2.1: Controller LDS state + sizing assertion

**Files:**
- Modify: `occ_kernel_coop.s` (LDS layout additions behind `.if DSWS`)

**Interfaces:**
- Produces LDS symbols: `NCOMP_SLOT`, `NAFEED_SLOT`, `NBFEED_SLOT` (u32 each), `GATE_OFF` (4×u32, one per direction), `VRESV_OFF` (u32 `vgpr_reserved`), `SEGCNT_OFF` (u32 per-WG segments_processed). `LDS_TOTAL_DSWS` recomputed.

- [ ] **Step 1: Add the state symbols + recompute `LDS_TOTAL`.** Add a build-time `.if (LDS_TOTAL_DSWS > 65536) .error` assertion (gfx1201 = 64 KB LDS/WG).
- [ ] **Step 2: Gate (RGA + LDS-size assert, offline)**

Run: `KSRC=occ_kernel_coop.s DEFS="-defsym DSWS=1 -defsym NCOMP=4 -defsym NAFEED=2 -defsym NBFEED=2 -defsym RINGD=2" ./rga_check.sh`
Expected: assembles (LDS assertion passes for the swept tiles), 0-spill. If a large tile trips the 64 KB assert → record the max RINGD/tile that fits (feeds the "LDS sizing" open detail).
- [ ] **Step 3: Commit (when greenlit):** `git add occ_kernel_coop.s; git commit -m "dsws: controller LDS state (role slots, gates, reservation, segcnt) + 64KB assert"`

### Task 2.2: CPU model of the control-law logic (TDD — this is genuinely unit-testable)

**Files:**
- Create: `dsws_ctrl_model.cpp`, `test_dsws_ctrl_model.cpp`

**Interfaces:**
- Produces (pure functions, mirrored 1:1 by the asm in Phase 3):
  - `int watermark_decision(uint32_t occ, uint32_t low, uint32_t high)` → `-1` (over-served, occ>high), `+1` (starved, occ<low), `0` (dead-zone).
  - `uint32_t epoch_of(uint32_t segs, uint32_t shift)` → `segs >> shift`.
  - `bool gate_try_win(std::atomic<uint32_t>& gate, uint32_t E)` → CAS single-winner (read g; if g≥E false; else CAS(g→E)).
  - `bool reserve_grow(std::atomic<uint32_t>& resv, uint32_t delta, uint32_t budget)` → atomic_add; if over budget, atomic_sub back + false; else true.

- [ ] **Step 1: Write failing tests**

```cpp
// test_dsws_ctrl_model.cpp
#include "dsws_ctrl_model.cpp"
#include <atomic>#include <thread>#include <vector>#include <cassert>#include <cstdio>
int main(){
  // watermark bands
  assert(watermark_decision(0,2,6)==+1);   // empty -> starved
  assert(watermark_decision(7,2,6)==-1);   // full  -> over-served
  assert(watermark_decision(4,2,6)==0);    // dead-zone
  // epoch
  assert(epoch_of(0,3)==0 && epoch_of(8,3)==1 && epoch_of(15,3)==1 && epoch_of(16,3)==2);
  // gate: exactly ONE winner per epoch among many racers
  for(uint32_t E=1;E<50;++E){ std::atomic<uint32_t> g{E-1}; std::atomic<int> wins{0};
    std::vector<std::thread> ts; for(int i=0;i<64;++i) ts.emplace_back([&]{ if(gate_try_win(g,E)) wins++; });
    for(auto&t:ts)t.join(); assert(wins.load()==1); }
  // reservation: concurrent grows never exceed budget
  { std::atomic<uint32_t> r{0}; std::atomic<int> ok{0};
    std::vector<std::thread> ts; for(int i=0;i<10;++i) ts.emplace_back([&]{ if(reserve_grow(r,30,100)) ok++; });
    for(auto&t:ts)t.join(); assert(r.load()<=100 && ok.load()==3); } // 3*30=90<=100, 4th would be 120>100
  printf("dsws_ctrl_model: ALL PASS\n"); return 0; }
```
- [ ] **Step 2: Run to verify it fails**

Run: `clang++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o t_ctrl && ./t_ctrl`
Expected: FAIL to compile (`dsws_ctrl_model.cpp` not written).
- [ ] **Step 3: Write `dsws_ctrl_model.cpp`** — the four pure functions exactly as specced in Interfaces.
- [ ] **Step 4: Run to verify pass**

Run: `clang++ -std=c++17 -O2 -pthread test_dsws_ctrl_model.cpp -o t_ctrl && ./t_ctrl`
Expected: `dsws_ctrl_model: ALL PASS`. (This locks the control *logic* before it's transcribed to asm — the asm must match these semantics.)
- [ ] **Step 5: Commit (when greenlit):** `git add dsws_ctrl_model.cpp test_dsws_ctrl_model.cpp; git commit -m "dsws: CPU model + tests for watermark/epoch/gate-CAS/reservation"`

### Task 2.3: Sensing reads + state init in the kernel (read-only; mix stays static)

**Files:**
- Modify: `occ_kernel_coop.s` (leader zeroes/initializes role slots to the launch mix, `vgpr_reserved` to the launch footprint, gates to 0, segcnt to 0; add `occ_a`/`occ_b` reads + `watermark_decision`-equivalent at the boundaries — but **do not act** on them yet; stream `occ_a/occ_b` + role slots into the snapshot)
- Modify: `occ_dispatch.cpp` (decode the new snapshot fields)

**Interfaces:**
- Consumes: LDS state (2.1), watermark semantics (2.2).

- [ ] **Step 1: Add leader init** (extend the existing barrier-free INITFLAG publish): zero gates/segcnt, set role slots to `NCOMP/NAFEED/NBFEED`, set `vgpr_reserved = NCOMP*V_fat + (NAFEED+NBFEED)*V_lean`.
- [ ] **Step 2: Add read-only sensing** at the compute segment boundary and the feed inter-frag check: compute `occ_a`/`occ_b`, evaluate the band (no action), and the per-WG `segcnt` bump on segment completion. Stream `occ_a, occ_b, nComp, nAfeed, nBfeed` in the occ snapshot.
- [ ] **Step 3: Gate (RGA offline + oracle unchanged)**

Run: `./build_dsws.sh static && KSRC=... ./rga_check.sh`
Expected: assembles, 0-spill; oracle STILL green (no behavior change — sensing is read-only). `DSWS=0` still byte-identical.
- [ ] **Step 4: Commit (when greenlit):** `git add occ_kernel_coop.s occ_dispatch.cpp; git commit -m "dsws: state init + read-only ring sensing + snapshot instrumentation"`

### Task 2.4 [SUPERVISED GPU]: Sensors-report-sane gate

**Files:** none (run only).

- [ ] **Step 1: Prepare ONE gated dispatch** (`4c2a2b r2 STORE=1`, streamed). **STOP for go.**
- [ ] **Step 2: On greenlight, run** and inspect the streamed snapshots.

Expected: oracle CLEAN; `occ_b` (and `occ_a`) visibly oscillate in `[0, RINGD]` as the rings fill/drain; role slots remain the static launch mix; no brick. This confirms the sensors read real pressure before we let them actuate anything.
- [ ] **Step 3:** Record a representative occ trace in `RESULT_DSWS.md`. If sensors look wrong (pinned at 0 or RINGD always) → STOP, the sensing addressing is off; fix offline before Phase 3.

---

## Phase 3 — Conversion + epoch/ticket + role-agnostic terminal (Gate 2)

### Task 3.1: Epoch + gate-CAS in asm (transcribe the verified model)

**Files:**
- Modify: `occ_kernel_coop.s` (`epoch = segcnt >> EPOCH_SHIFT`; `gate_try_win` via `ds_cmpst_b32` (LDS compare-swap) on `GATE_OFF + dir*4`)

**Interfaces:**
- Consumes: `dsws_ctrl_model.cpp` semantics (2.2) — the asm MUST match `gate_try_win`/`epoch_of` exactly. `EPOCH_SHIFT` defsym.

- [ ] **Step 1: Implement `epoch_of` + `gate_try_win`** as an asm macro `try_gate dir` returning SCC/flag. Use `ds_cmpst_b32` for the LDS CAS (read g; branch if g≥E; else CAS(g,E); success iff returned-old==g).
- [ ] **Step 2: Gate (RGA offline)** — assembles, 0-spill. **Still no conversion wired** (macro defined + a unit call site that logs "would-win" into a snapshot slot, no actuation).
- [ ] **Step 3 [SUPERVISED GPU, optional micro-check]:** if cheap, one streamed run confirming the "would-win" counter increments ≤1 per epoch per direction. Else defer to 3.5.
- [ ] **Step 4: Commit (when greenlit):** `git add occ_kernel_coop.s; git commit -m "dsws: epoch + lock-free gate-CAS ticket (ds_cmpst), no actuation yet"`

### Task 3.2: compute→feed conversion (shrink path)

**Files:**
- Modify: `occ_kernel_coop.s` (at the compute segment boundary: if `occ_X<LOW` & `try_gate(compute→feedX)` & `nComp>1` → CAS-dec `nComp`, inc `nFeedX`, `atomic_sub vgpr_reserved`, `s_alloc_vgpr LEANREG`, jump into the feed loop)

**Interfaces:**
- Consumes: gate macro (3.1), role slots + reservation (2.1), `V_fat`/`V_lean` (1.2 RGA).

- [ ] **Step 1: Implement the shrink-conversion** at the segment boundary, floor-guarded (`nComp>1` via `ds_cmpst`/atomic dec-if-greater), epoch-gated. Shrink always succeeds.
- [ ] **Step 2: Gate (offline RGA + build)** — assembles, 0-spill.
- [ ] **Step 3: Oracle gate (offline-buildable, GPU-run deferred to 3.5)** — build STORE=1 bin.
- [ ] **Step 4: Commit (when greenlit):** `git add occ_kernel_coop.s; git commit -m "dsws: compute->feed conversion (shrink, floor+epoch gated)"`

### Task 3.3: feed→compute conversion (grow path + envelope abort)

**Files:**
- Modify: `occ_kernel_coop.s` (at the feed inter-frag check: if `occ_X>HIGH` & `try_gate(feedX→compute)` & `nFeedX>1` → `reserve_grow`; on success: CAS-dec `nFeedX`, inc `nComp`, `s_alloc_vgpr NFV`, enter compute loop; on reserve fail: abort, leave slots/reservation untouched, stay feed)

**Interfaces:**
- Consumes: `reserve_grow` semantics (2.2), gate (3.1), `BUDGET` (per-SIMD VGPR budget, from RGA/ISA in 1.4).

- [ ] **Step 1: Implement the grow-conversion** with the reservation-then-validate-then-maybe-undo envelope (transcribe `reserve_grow`). Order: reserve first (atomic_add), validate `≤BUDGET`, then commit slot move + `s_alloc_vgpr` grow; on over-budget atomic_sub + abort (NO slot change).
- [ ] **Step 2: Gate (offline RGA + build)** — assembles, 0-spill; `DSWS=0` still byte-identical.
- [ ] **Step 3: Commit (when greenlit):** `git add occ_kernel_coop.s; git commit -m "dsws: feed->compute conversion (grow + sum-envelope reservation/abort)"`

### Task 3.4: Role-agnostic POOLTERM terminal

**Files:**
- Modify: `occ_kernel_coop.s` (every role's loop checks the `ti≥TOTAL` terminal broadcast at the SAME boundary it checks for conversion; a converted wave re-checks immediately)

**Interfaces:**
- Consumes: the existing POOLTERM `ti≥TOTAL` feed broadcast (KG `dac0bb8c`).

- [ ] **Step 1: Hoist the terminal check** so it is evaluated by compute AND both feed roles at their decision boundaries, on the RAW ti (before SAFEPROBE clamp). A wave that just converted falls through to the terminal check before doing role work.
- [ ] **Step 2: Gate (offline)** — assembles; `DSWS=0` byte-identical.
- [ ] **Step 3: Commit (when greenlit):** `git add occ_kernel_coop.s; git commit -m "dsws: role-agnostic POOLTERM terminal (no wave stranded past drain)"`

### Task 3.5 [SUPERVISED GPU]: Gate 2 — dynamic oracle-green + conversion-storm

**Files:** none (run only).

- [ ] **Step 1: Prepare ONE gated dispatch** — `DSWS_DYN=1 4c2a2b r2 STORE=1`, normal watermarks, streamed. **STOP for go.**
- [ ] **Step 2: On greenlight, dynamic oracle** across `{down, down_pf, square}`. Expected: `ok=N bad=0`, conversions visible in the snapshot (role slots move), fence FIRED, clean teardown, no brick.
- [ ] **Step 3: Conversion-storm stress** — `DSWS_LOW`/`DSWS_HIGH` tight + `DSWS_EPOCHSHIFT=0` (max conversion rate) + many repeats (oracle STORE=1, x10). Expected: STILL `bad=0` under the storm — proves the lock-free protocol has no race (the strong-oracle+repeats discipline that caught 136/552 before). Any `bad>0` → STOP, a conversion race exists; bisect offline.
- [ ] **Step 4:** Record Gate-2 results in `RESULT_DSWS.md`. Do not proceed to tuning until dynamic + storm are both clean.

---

## Phase 4 — Adaptivity proof + tuning

### Task 4.1: Role-count + conversion-counter instrumentation

**Files:**
- Modify: `occ_kernel_coop.s` (per-direction conversion counters into snapshot), `occ_dispatch.cpp` (decode + print a compact mix-over-time table)

- [ ] **Step 1: Add 4 conversion counters** (one per direction) + emit role slots each snapshot; harness prints `t, nComp, nAfeed, nBfeed, convs[4]`.
- [ ] **Step 2: Gate (offline build + RGA).**
- [ ] **Step 3: Commit (when greenlit):** `git add occ_kernel_coop.s occ_dispatch.cpp; git commit -m "dsws: mix-over-time + conversion-counter instrumentation"`

### Task 4.2 [SUPERVISED GPU]: Converge-from-wrong-start proof

**Files:** none (run only).

- [ ] **Step 1: Prepare ONE gated dispatch** — feed-bound shape (`down`, N=2560), deliberately wrong launch mix (e.g. `6c1a1b`, compute-heavy). **STOP for go.**
- [ ] **Step 2: On greenlight, run** and watch the mix table + TF. Expected (the money shot): the controller **shifts toward more feed** (nAfeed/nBfeed climb, nComp falls) and **TF climbs** as it converges, settling near the static-optimal mix. Compare adaptive-from-wrong-start vs static-wrong-start (should beat it) and vs static-optimal (should approach it).
- [ ] **Step 3:** Record the convergence trace + TF curve in `RESULT_DSWS.md`. This is the "it actually adapts" evidence.

### Task 4.3 [SUPERVISED GPU]: Tuning sweep + issue-mix verification

**Files:** none (run only); record in `RESULT_DSWS.md`.

- [ ] **Step 1:** Sweep `{LOW, HIGH, RINGD, EPOCH_SHIFT}` on `down` and `down_pf`, **one gated dispatch at a time** (no auto-sweep on the display GPU — each cell is a separate greenlit run, or a compositor-safe chunked batch with the per-chunk hang-abort). Oracle (STORE=1) before perf (STORE=0) for any new geometry.
- [ ] **Step 2: `--att` issue-mix** on the winning config. Expected: the compute waves issue **near-pure WMMA** — measurably fewer non-WMMA issues per WMMA than the 31:32 static baseline (the wall we set out to break).
- [ ] **Step 3: Success-metric gate:** the adaptive kernel (a) oracle-correct, (b) beats the static 3-role baseline AND the 165.7 TF winner on `down`/`down_pf`, (c) demonstrably adapts across the two shapes (different settled mixes). Record TF, the settled mixes, and the `--att` deltas.

### Task 4.4: Bank the outcome

**Files:**
- Modify: `RESULT_DSWS.md` (final), `MAD305_DSWS_MASTER.md` (§2 numbers + §9 next steps)

- [ ] **Step 1:** Finalize `RESULT_DSWS.md` (numbers, settled mixes, issue-mix deltas, honest verdict incl. any null result).
- [ ] **Step 2:** Update the master doc; `mneme_write` a session_summary banking the result + the controller design.
- [ ] **Step 3: Commit (when greenlit).** Jira MAD-305 update.

---

## Self-Review

**1. Spec coverage** (every spec section → a task):
- 3-role economy / role slots → T1.2, T2.1. ✓
- Data-path fixed, feed floor=1, compute floor≥1 → enforced in T3.2/T3.3 floor guards; no dual-path code (compute always A-ring) per T1.2. ✓
- Sensing / watermark bands → T2.2 (logic), T2.3 (asm read-only), T2.4 (sane gate). ✓
- Conversion both directions → T3.2 (shrink), T3.3 (grow). ✓
- Epoch + gate-CAS single-winner → T2.2 (model), T3.1 (asm). ✓
- Sum-envelope reservation → T2.2 (`reserve_grow`), T3.3 (asm). ✓
- Role-agnostic POOLTERM terminal → T3.4. ✓
- No barrier / lock-free → inherited (no new rendezvous); verified by storm T3.5. ✓
- Two-gate sequencing → Gate 1 = T1.4, Gate 2 = T3.5. ✓
- Conversion-storm race stress → T3.5 step 3. ✓
- Adaptivity proof (converge-from-wrong-start) → T4.2. ✓
- Success metric (beats baselines + --att issue cut) → T4.3. ✓
- Supervised GPU discipline / brick safety → every `[SUPERVISED GPU]` task + Global Constraints. ✓
- Open details (oracle determinism, segcnt source, LDS sizing, V_fat/V_lean/BUDGET, feed check cadence) → resolved in T1.4 (RGA constants), T2.1 (LDS assert), T2.3 (segcnt + cadence), oracle tolerance carried from coop in T1.3. ✓

**2. Placeholder scan:** No "TBD/implement later." Assembly tasks are gate-defined (precise structural change + exact gate command), which is the honest unit for hand-asm — not a placeholder. ✓

**3. Type consistency:** `DswsCfg` fields, defsym names (`NCOMP/NAFEED/NBFEED/RINGD/RINGD_A/EPOCH_SHIFT/DSWS/STORE`), LDS symbols (`ARING_OFF/PROD_A_OFF/CONS_A_OFF/NCOMP_SLOT/NAFEED_SLOT/NBFEED_SLOT/GATE_OFF/VRESV_OFF/SEGCNT_OFF`), and model functions (`watermark_decision/epoch_of/gate_try_win/reserve_grow`) are used consistently across tasks. ✓

No gaps found.
