# DSWS Emergent Wave-Economy — Implementation Plan (Step 1)

> **For agentic workers:** implement task-by-task. Steps use checkbox (`- [ ]`) syntax. Spec:
> `DSWS_EMERGENT_ECONOMY_DESIGN.md`. This is Step 1 (kill the baked mix); Step 2 (POOL_N=2 feed
> overlap) is a separate later plan.

**Goal:** Delete the baked `NCOMP/NAFEED/NBFEED` wave mix. Launch one generous lean pool seeded
compute-biased with a static feed floor; the wave economy (who is fat compute vs lean feed) emerges at
runtime from the hardware `s_alloc_vgpr` grow-fail + the existing coast. Launch count is host-derived,
capped at 30.

**Architecture:** A *subtraction* on `occ_kernel_dsws_flow.s` (concurrent-fat already emerges from
hardware; the only fungibility is the coast; roles are static-from-seed). Change the seed to
compute-biased+floor, decouple `WAVES` from the deleted mix, make `BUDGET` a physical constant
(`VBUDGET`), and have the host derive `W_launch` and load a single per-count flow bin.

**Tech Stack:** amdgcn-amd-amdhsa assembly (gfx1201 wave32), C++17 host (raw PM4/KFD), bash build.

## Global Constraints

- **NO GPU dispatch without kmbandy's individual greenlight.** Tasks 1-3 verify OFFLINE (assemble /
  compile / dry-run only). Task 4 is the single greenlit GPU checkpoint.
- GPU data logs → `/home/kmbandy/dsws_gpu_logs/` (real disk), never tmpfs. DEADMAN on. Never `--gl2c`.
  Keep SAFEPROBE + bounds gate. Commit to git ONLY when kmbandy asks.
- `WAVES ≤ 30` is a hard structural ceiling (coordinator state squats at `ROLE[30]/[31]`).
- Do not re-arm `conv_apply` / coordinator sense-nudge. Do not touch `POOL_N`/`SEGK` (that's Step 2).
- Run config for all builds/runs stays: `SEGK=32 POOL_N=1 ACC_N=6 G=6` (+ `TFPROBE=1 STAGINSTR=1`).

---

### Task 1: Kernel — delete the baked mix, physical BUDGET, compute-biased seed

**Files:**
- Modify: `occ_kernel_dsws_flow.s` (5 edits below)

**Interfaces:**
- Consumes: `-defsym,WAVES=<N>` and `-defsym,VBUDGET=<credits>` from the build (Task 2).
- Produces: a bin whose per-WG role economy emerges from a compute-biased seed; no `NCOMP/NAFEED/NBFEED`.

- [ ] **Step 1a: Cut the dangling `STAGGER_PERIOD → NCOMP` reference (line ~202-204).**

Replace:
```
.ifndef STAGGER_PERIOD
  .set STAGGER_PERIOD, NCOMP   // phase slots in the stagger ring (R3 sweep). Used iff STAGGER=1.
.endif
```
with:
```
.ifndef STAGGER_PERIOD
  .set STAGGER_PERIOD, 4       // phase slots in the stagger ring (R3 sweep). Used iff STAGGER=1 (inert here).
.endif
```

- [ ] **Step 1b: Fix the `G` comment (line ~131-133).** `G` is already `.ifndef`-guarded and
  defsym-driven — only decouple the misleading comment:
```
.ifndef G
  .set G, 6            // cooperative M-extent (rowblks per super-tile) = LDS accumulator-bank count (ACC_N)
.endif
```

- [ ] **Step 1c: Replace the role-count block with a standalone `WAVES` + mailbox guard (line ~362-374).**

Replace:
```
.if DSWS2
  // ---- role counts (lifted from coop's `.ifndef NCOMP` etc., gated under DSWS2) ----
  .ifndef NCOMP
    .set NCOMP, 4                            // compute waves (fat, dyn-grow). Compute floor >= 1.
  .endif
  .ifndef NAFEED
    .set NAFEED, 2                           // A-feed waves (lean). Feed floor >= 1.
  .endif
  .ifndef NBFEED
    .set NBFEED, 2                           // B-feed waves (lean). Feed floor >= 1.
  .endif
  .set WAVES, (NCOMP + NAFEED + NBFEED)      // total waves launched per WG (harness dims must match)
.endif
```
with:
```
.if DSWS2
  // ---- launch wave count (EMERGENT economy: NO baked compute/feed mix; roles emerge at runtime) ----
  .ifndef WAVES
    .set WAVES, 16                           // waves/WG launched; host launches the SAME count.
  .endif
  .if WAVES > 30
    .error "WAVES>30 collides with COORD_KSI/T at ROLE[30]/ROLE[31] -- relocate coord state first"
  .endif
.endif
```

- [ ] **Step 1d: Make `BUDGET` physical (`VBUDGET`) + add always-on sanity guards (line ~881-900).**

Replace the whole `.ifndef BUDGET … .endif` block AND the two gated guard blocks:
```
.ifndef BUDGET
.if DSWS2_ENVELOPE
  .set BUDGET, (WAVES*VLEAN + PEAK_CONC*(NFV-VLEAN))   // rolling: lean floor + concurrent-peak headroom
.else
  .set BUDGET, (NCOMP*NFV + (NAFEED+NBFEED)*VLEAN)     // = VRESV_OFF init (conservation ceiling)
.endif
.endif

.if DSWS2_CONV
// compile-time no-parking invariant: every launched wave must fit lean at once
.if (WAVES * VLEAN) > BUDGET
  .error "WAVES*VLEAN exceeds BUDGET — pool cannot stay all-lean (parking is out of scope)"
.endif
.endif
.if DSWS2_ENVELOPE
// forward-progress: the budget must admit at least one concurrent peak or a claimed wave can never grow
.if (WAVES*VLEAN + (NFV-VLEAN)) > BUDGET
  .error "ENVELOPE: BUDGET admits < 1 concurrent peak — forward progress impossible"
.endif
.endif
```
with:
```
.ifndef VBUDGET
  .set VBUDGET, 1536        // physical VGPR-file credit ceiling (R9700 wave32, per SIMD). Calibrate.
.endif                       //   Sanity ceiling only: the hardware s_alloc_vgpr is the real concurrent-fat cap.
.ifndef BUDGET
.if DSWS2_ENVELOPE
  .set BUDGET, (WAVES*VLEAN + PEAK_CONC*(NFV-VLEAN))   // rolling: lean floor + concurrent-peak headroom
.else
  .set BUDGET, VBUDGET      // EMERGENT: budget is PHYSICAL, not mix-derived (ledger is dormant; conv-only).
.endif
.endif

// emergent-economy PHYSICAL sanity (always on): all waves fit lean, and >=1 can grow.
.if (WAVES * VLEAN) > BUDGET
  .error "WAVES*VLEAN exceeds VBUDGET -- pool cannot stay all-lean"
.endif
.if (WAVES*VLEAN + (NFV-VLEAN)) > BUDGET
  .error "VBUDGET admits < 1 concurrent grow -- compute can never make progress"
.endif
```

- [ ] **Step 1e: Replace the seed with compute-biased + static floor (line ~1345-1357).**

Replace:
```
    // seed per-wave ROLE mailbox with the launch mix: wid<NBFEED -> BFEED (wid0=coordinator, B-class),
    //   wid<NBFEED+NAFEED -> AFEED, else COMPUTE
    .set w, 0
    .rept WAVES
      .if w < NBFEED
        lds_put (ROLE_BASE + w*4), ROLE_BFEED
      .elseif w < (NBFEED + NAFEED)
        lds_put (ROLE_BASE + w*4), ROLE_AFEED
      .else
        lds_put (ROLE_BASE + w*4), ROLE_COMPUTE
      .endif
      .set w, w+1
    .endr
```
with:
```
    // EMERGENT economy seed: minimal liveness FLOOR + everything else COMPUTE. wid0=coordinator (runs
    //   lean B-feed between ASSIGN duties), wid1=dedicated A-feed, wid2=dedicated B-feed; wid>=3=COMPUTE.
    //   Excess compute waves self-distribute to feed via .Lflow_coast; concurrent-fat emerges from the
    //   hardware s_alloc_vgpr grow-fail. NO baked NCOMP/NAFEED/NBFEED.
    .set w, 0
    .rept WAVES
      .if w == 0
        lds_put (ROLE_BASE + w*4), ROLE_BFEED
      .elseif w == 1
        lds_put (ROLE_BASE + w*4), ROLE_AFEED
      .elseif w == 2
        lds_put (ROLE_BASE + w*4), ROLE_BFEED
      .else
        lds_put (ROLE_BASE + w*4), ROLE_COMPUTE
      .endif
      .set w, w+1
    .endr
```

- [ ] **Step 1f: Verify no live `NCOMP/NAFEED/NBFEED` references remain (comments OK).**

Run: `grep -n "NCOMP\|NAFEED\|NBFEED" occ_kernel_dsws_flow.s`
Expected: only comment lines (e.g. the `conv_apply`/`conv_dec_floor` doc comments ~912/995). No `.set`,
no expression use. If any live use remains, resolve before building.

- [ ] **Step 1g: Assemble clean at WAVES=16 (offline, safe).**

Run: `WAVES=16 VBUDGET=1536 SEGK=32 POOL_N=1 ACC_N=6 G=6 TFPROBE=1 STAGINSTR=1 ./build_flow.sh`
(requires Task 2 first — build_flow.sh must pass the new defsyms). Do Task 2 before 1g.
Expected: `OK   occ_dsws2_w16_flow_gd.bin (<size>B .text)`. Then check no register spill:
`grep -i "spill" /tmp/flow_build.err || echo "0-spill"` → `0-spill`.

- [ ] **Step 1h: Assemble at WAVES=8 too (regression sanity).**

Run: `WAVES=8 VBUDGET=1536 SEGK=32 POOL_N=1 ACC_N=6 G=6 TFPROBE=1 STAGINSTR=1 ./build_flow.sh`
Expected: `OK   occ_dsws2_w8_flow_gd.bin`, 0-spill. (Seed differs from the old 4c2a2b — this is the
intended behavior change, not a byte-identity target.)

---

### Task 2: Build — `build_flow.sh` passes `WAVES`/`VBUDGET`, drops the mix

**Files:**
- Modify: `build_flow.sh` (the `mkflow` defsym line + tag + the `c/a/b` arg plumbing)

**Interfaces:**
- Consumes: env `WAVES`, `VBUDGET` (plus existing `SEGK/ACC_N/G/POOL_N/...`).
- Produces: `occ_dsws2_w<WAVES>_flow_gd.bin` (name no longer carries the mix).

- [ ] **Step 2a: Rewrite `mkflow` to defsym `WAVES`/`VBUDGET`, drop `NCOMP/NAFEED/NBFEED`, rename tag.**

In `mkflow`, change the tag line to:
```
  local tag="occ_dsws2_w${WAVES:-16}_flow_gd"
```
In the `clang` invocation, replace the mix defsyms
`-Wa,-defsym,NCOMP=$1 -Wa,-defsym,NAFEED=$2 -Wa,-defsym,NBFEED=$3` with:
```
     -Wa,-defsym,WAVES=${WAVES:-16} -Wa,-defsym,VBUDGET=${VBUDGET:-1536} \
```
Also add `-Wa,-defsym,G=${G:-6}` to the defsym list if not already present (Task 1b relies on `G` being
a defsym; the current script passes `G=6` literally — keep it or make it `${G:-6}`).

- [ ] **Step 2b: Drop the positional `c/a/b` handling at the call site.**

Replace the bottom block:
```
c=${1:-4}; a=${2:-2}; b=${3:-2}
echo "== FIX 1 flow bin (occ_kernel_dsws_flow.s; G=6 SEGK=64 FM=2 FN=4 POOL_N=${POOL_N:-3}) =="
mkflow "$c" "$a" "$b"
```
with:
```
echo "== flow bin (occ_kernel_dsws_flow.s; EMERGENT mix; WAVES=${WAVES:-16} G=${G:-6} SEGK=${SEGK:-64} POOL_N=${POOL_N:-3} VBUDGET=${VBUDGET:-1536}) =="
mkflow
```
And change the `mkflow()` header comment/params — it no longer takes `$1 $2 $3`.

- [ ] **Step 2c: Build succeeds (this is also Task 1g/1h).**

Run: `WAVES=16 VBUDGET=1536 SEGK=32 POOL_N=1 ACC_N=6 G=6 TFPROBE=1 STAGINSTR=1 ./build_flow.sh`
Expected: `OK   occ_dsws2_w16_flow_gd.bin (<size>B .text)  [POOL_N=1 ...]`, exit 0.

---

### Task 3: Host — derive `W_launch`, single per-count flow bin, skip mix for flow

**Files:**
- Modify: `occ_dispatch.cpp` — the DSWS2 dispatch block (~6040-6127) and the flow bin-name (~6113).

**Interfaces:**
- Consumes: env `FLOW_WAVES` (default 16), `DSWS2_FLOW`. Physical constants `NFV=112`,`VLEAN=32` for the
  sanity check (or reuse existing helpers if present).
- Produces: launches `W_launch*32` threads/WG; loads `occ_dsws2_w<W_launch>_flow_gd.bin`; `run_dsws2`
  receives `(W_launch, 0, 0)` so `WAVES_LAUNCH == W_launch` (C-sizing uses `Gv`, unaffected).

- [ ] **Step 3a: Derive `W_launch` and override the mix for the flow path (just before the bin block, ~6104).**

Insert, right after `char dswsBin[160];` and before the `if (getenv("DSWS2_FLOW"))` name switch:
```cpp
            // EMERGENT economy (flow path): no baked mix. Launch a host-derived pool, capped at 30
            //   (coordinator mailbox squat) and sanity-checked against the lean-fit budget.
            if (getenv("DSWS2_FLOW")) {
                uint32_t Wlaunch = getenv("FLOW_WAVES") ? (uint32_t)atoi(getenv("FLOW_WAVES")) : 16u;
                if (Wlaunch < 4)  Wlaunch = 4;            // floor(3) + >=1 compute
                if (Wlaunch > 30) { printf("  [flow] FLOW_WAVES=%u > 30 (coord cap) -> clamping to 30\n", Wlaunch); Wlaunch = 30; }
                const uint32_t VB = getenv("FLOW_VBUDGET") ? (uint32_t)atoi(getenv("FLOW_VBUDGET")) : 1536u;
                const uint32_t leanFit = (VB - (112u - 32u)) / 32u;   // (VBUDGET-(NFV-VLEAN))/VLEAN
                if (Wlaunch > leanFit)
                    printf("  [flow] WARNING FLOW_WAVES=%u exceeds lean-fit=%u for VBUDGET=%u (bin's .error will catch a real overflow)\n", Wlaunch, leanFit, VB);
                c.nComp = Wlaunch; c.nAfeed = 0; c.nBfeed = 0;   // WAVES_LAUNCH = Wlaunch inside run_dsws2
            }
```

- [ ] **Step 3b: Name the flow bin by wave count (~6113).**

Change the flow branch:
```cpp
            if (getenv("DSWS2_FLOW"))
                snprintf(dswsBin, sizeof dswsBin, "occ_dsws2_%uc%ua%ub_flow_gd.bin", c.nComp, c.nAfeed, c.nBfeed);
```
to:
```cpp
            if (getenv("DSWS2_FLOW"))
                snprintf(dswsBin, sizeof dswsBin, "occ_dsws2_w%u_flow_gd.bin", c.nComp);   // c.nComp == W_launch (3a)
```

- [ ] **Step 3c: Skip the positional-mix REFUSE for the flow path (~6055-6076).**

Wrap the `if (g_posMixArg) { sscanf … REFUSE … }` block so it is skipped when `DSWS2_FLOW` is set:
```cpp
            if (!getenv("DSWS2_FLOW") && g_posMixArg) {
                // ... existing positional-mix parse + mismatch REFUSE ...
            }
```
(The `nComp<1` / `N()` sum checks below stay; with `(Wlaunch,0,0)` they pass: `nComp>=4`.)

- [ ] **Step 3d: Compile the host (offline, safe).**

Run:
```
clang++ -std=c++17 -O2 -I ../dvgpr_pm4/vendor/compat -I ../dvgpr_pm4/vendor -I ../dvgpr_pm4 -I /opt/rocm/include \
  occ_dispatch.cpp fp8_oracle.cpp ../dvgpr_pm4/vendor/PM4Packet.cpp ../dvgpr_pm4/vendor/BasePacket.cpp \
  /opt/rocm/lib/libhsakmt.a -ldrm_amdgpu -ldrm -lnuma -lpthread -ldl -lrt -o occ_dispatch
```
Expected: builds to `occ_dispatch` (hsakmt linter warnings are pre-existing noise). No new errors.

- [ ] **Step 3e: Dry-run prints the derived launch (offline, safe — DSWS2_DRYRUN, no GPU).**

Run:
```
DSWS2_FLOW=1 DSWS2_DRYRUN=1 DSWS2_SEGK=32 DSWS2_ACC_N=6 FLOW_POOL_N=1 FLOW_WAVES=16 \
  DSWS2_ORACLE_MTL=3 DSWS2_ORACLE_NTL=8 DSWS2_NKSEG=64 ./occ_dispatch --dsws2
```
Expected: prints `waves/WG=16(=16c0a0b)` and the bin name `occ_dsws2_w16_flow_gd.bin`, returns before any
GPU submit. If `DSWS2_DRYRUN` isn't wired for the flow path, verify by inspection that the derivation +
bin-name print are correct and proceed (do NOT submit to GPU here).

---

### Task 4: GPU checkpoint — oracle-clean at the new pool + emergence proof (GREENLIT ONLY)

**Files:** none (verification only).

**Interfaces:** consumes the Task-1/2/3 artifacts (`occ_dsws2_w16_flow_gd.bin`, `occ_dspatch`).

> **STOP — do not run any command in this task without kmbandy's explicit per-dispatch greenlight.**

- [ ] **Step 4a: Oracle at W=16 (greenlit).** Log to `/home/kmbandy/dsws_gpu_logs/`.

Run (after greenlight):
```
DSWS2_FLOW=1 DSWS2_SEGK=32 DSWS2_ACC_N=6 FLOW_POOL_N=1 FLOW_WAVES=16 \
  DSWS2_ORACLE_MTL=3 DSWS2_ORACLE_NTL=8 DSWS2_NKSEG=64 ML8_POOL=24 \
  ./occ_dispatch --dsws2 2>&1 | tee /home/kmbandy/dsws_gpu_logs/emergent_w16_$(date +%s).log
```
Expected (acceptance §6): `bad=0`, `max_rel=0`, `occ[0]=0`. Repeat 2-3× for the race check (Risk D).

- [ ] **Step 4b: Read the emergence proof from STAGINSTR.**

From the same run's readout: `occ[73]` (grow-fail) should be **> 0** (was 0 at 8 waves — proof the
physical VGPR file finally binds and the stagger's coast-repulsion engages), and `occ[1]` (peak resident
waves) should rise meaningfully above 8. Also sanity-check the coast fraction (`occ[70]`) is neither ~100%
(pure feed-starve) nor ~0%.

- [ ] **Step 4c: Re-confirm at W=8 (greenlit).** Build `WAVES=8 …` bin (Task 1h) and run with
  `FLOW_WAVES=8`; expect `bad=0` — confirms the seed change didn't regress the small pool.

- [ ] **Step 4d: Record results** in a short note (and memory) — TF, grow-fail, occ[1] at W=8/16, and
  whether emergence engaged. Do NOT commit to git unless kmbandy asks.

---

## Self-Review

- **Spec coverage:** §3.1 seed → Task 1e; §3.2 decouple WAVES/G → 1b/1c; §3.3 physical BUDGET+guards →
  1d; §3.4 host W_launch derive + single bin → Task 3; §3.5 ≤30 cap → 1c guard + 3a clamp; §6 acceptance
  → Task 4. Step 2 (POOL_N=2) intentionally excluded.
- **Placeholder scan:** `VBUDGET=1536` and `FLOW_WAVES=16` are explicit calibratable defaults, not TBDs;
  Task 4 pins the empirical acceptance numbers.
- **Type/name consistency:** bin name `occ_dsws2_w<N>_flow_gd.bin` matches between Task 2a (tag) and Task
  3b (host). `WAVES` defsym name matches kernel (1c) ↔ build (2a). `(W_launch,0,0)` (3a) makes
  `WAVES_LAUNCH` (run_dsws2) == the built `WAVES`.
