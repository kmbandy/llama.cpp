# DSWS / Adaptive Dynamic Staggered Wave Specialization — Kernel Review

**Date:** 2026-07-08  
**Author:** Grok (review of night-end state after Claude session)  
**Kernel tree:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/spike/dvgpr_occ/`  
**Primary sources:** `MORNING_2026-07-07.md`, commit `03225f6bb`, run logs under `/home/kmbandy/dsws_gpu_logs/`, `occ_kernel_dsws_flow.s`, flow/emergent/G-resident design docs  
**Scope:** Detailed rundown of the latest kernel architecture, silicon results, room for improvement, and recommended architectural changes. Decoupled from weight-paging and ml8 quant.

---

## 1. Executive summary

DSWS is a serious, hardware-true campaign: raw-PM4 dyn-VGPR on RDNA4, wave specialization (compute / A-feed / B-feed), staggered traveling-peak VGPR footprints, and (designed, not yet active) runtime adaptive role rebalancing.

**Night-end truth (2026-07-06/07):**

| Finding | Evidence |
|---|---|
| Dyn-VGPR arming works | PM4 RSRC2 bit6; `DYN_VGPR_EN=1` |
| Flow path can be oracle-correct | Run 3: w16 whole-tile small shape, `ok=1152 bad=0` |
| Flow path is not yet a performance kernel | Run 3: **coast-frac ~98.2%**, TF printed ~0 |
| Dyn-VGPR pool does **not** bind | Run 5: **fatPeak=263** (~2 fat/SIMD), **grow-fail=0**, need ~13–14/SIMD |
| Group-split (ACC_N=3) deadlocks | Runs 6–7: `live=0 claim=26 fence=--` from +0.2s on a shape whole-tile cleans |
| Run 8 (frontier localization) not yet fired | Greenlight pending |

**Strategic takeaway:** the active experiment (group-split for occupancy/bind) is stacked on a pipeline that is already **coordination/coast-bound**, not issue-port or VGPR-bound. Fix pipeline health (STAGE lead, pool depth) before more LDS gymnastics. The adaptive moat (runtime multi-role VGPR economy) is **architecturally right** but **not running** on the current w16 flow bin.

---

## 2. What DSWS is (one paragraph)

**Dynamic-Split Wave-Specialization** (also described as Adaptive Dynamic Staggered Wave Specialization): one persistent-wave fp8 GEMM kernel on gfx1201 (RDNA4 / R9700, wave32) that rebalances a mix of **fat compute**, **lean A-feed**, and **lean B-feed** waves toward the runtime bottleneck. Dynamic VGPR (`s_alloc_vgpr`), armed only via raw PM4 on a KFD compute queue (HIP/MES strip the enable), makes per-SIMD VGPR **fungible** across roles. Peaks are **staggered** (lean → peak → lean) so the sum of instantaneous allocations stays under a budget. The adaptive piece senses ring / pipeline pressure and migrates roles at safe boundaries. North star: **250–300 TF** fp8 on R9700 (WMMA ceiling ~307 TF); prior static KWINBPF lineage peaked ~165 TF (~52%).

---

## 3. Layer map (where the work actually sits)

| Layer | Intent | Status |
|---|---|---|
| **L0 PM4 dyn-VGPR** | Arm `DYN_VGPR_EN`; prove `s_alloc_vgpr` grow/shrink | **Proven** (`dvgpr_pm4/`) |
| **L1 Static 3-role wave-spec** | Fixed compute / A-feed / B-feed; oracle | **Proven earlier** (4c2a2b / 6c1a1b / 2c3a3b on coop substrate) |
| **L2 Flow economy** | Non-blocking 3-frontier pipeline + coast | **Oracle-clean on small whole-tile; ~98% coast** |
| **L3 Occupancy / bind** | Shrink LDS / raise duty-cycle so VGPR pool binds | **FATMAX proves under-bound; group-split hangs** |
| **L4 Adaptive conversion** | Runtime role rebalance (mailbox nudge / conv) | **Dormant on flow path** |

Latest commit `03225f6bb` is **L3** (group-split + FATMAX gauge + throttled frontier instrumentation) on an L2 substrate that is correctness-first, not throughput-ready.

---

## 4. Kernel architecture (current: `occ_kernel_dsws_flow.s`)

### 4.1 Dispatch and moat

- Launch: raw PM4 on KFD compute queue (`occ_dispatch.cpp`), not HIP.
- Dyn-VGPR: `COMPUTE_PGM_RSRC2` bit 6; silicon `SQ_DYN_VGPR` already enabled.
- Display GPU constraint: R9700 drives monitors → sub-second compositor-safe chunks, `ML8_COOP_CHUNK_MAXS`, yield between chunks. Bricks are MES/queue class as often as OOB.

### 4.2 Flow economy (L2)

Three monotone LDS frontiers (invariant `DRAIN ≤ STAGE ≤ ASSIGN ≤ DRAIN + POOL_N`):

| Frontier | Writer | Meaning |
|---|---|---|
| **ASSIGN_HEAD** | Coordinator wid0 only | Next local index gets a global super-tile STAMP |
| **STAGE_HEAD** | Feeds (CAS advance) | Oldest fully staged super-tile |
| **DRAIN_HEAD** | Compute completer (CAS advance) | Oldest fully reduced (and C-stored if last ksi) super-tile |

Per-slot operand pool (ARES + BRES) + write-once fp32 C banks in LDS.

**Hot loop idea (design):** each wave reads `ROLE[wid]` (stale = last role = free coast), `try_grab` is one atomic returning work-or-empty, never spin on a producer. Empty → complementary work (coast) or short sleep.

**Coast (implemented):** compute with nothing staged (or grow-fail) runs feed code **without** `s_alloc_vgpr` resize (fat wave can execute lean feed touching ≤v31). This is the high-value free case.

**Role resize:** intended only on deliberate mailbox role change; per-burst grow/shrink for compute trapezoid is separate.

### 4.3 Emergent seed (Step 1 economy)

```
wid 0  -> B-feed (coordinator + assign duty)
wid 1  -> A-feed floor
wid 2  -> B-feed floor
wid 3..W-1 -> COMPUTE
```

No baked `NCOMP/NAFEED/NBFEED` in the kernel. Concurrent-fat is meant to emerge from hardware `s_alloc_vgpr` failure (`grow-fail`). Coordinator sense/nudge is still deferred (`// sense/nudge deferred` in `.Lflow_coord_period`).

**Implication:** the system is **not yet adaptive**. It is compute-biased + coast + hardware grow-fail. That is a liveness/economy substrate, not the runtime control loop that is the long-term IP.

### 4.4 Write-once C

- Per-rowblk fp32 partials live in LDS banks (`ACC_N` banks × 8KB).
- `ksi==0`: `ds_store` init bank; later ksi: `ds_add`.
- Completer (unique wave that pushes `RBDONE` to target) stores banks to global C on last ksi, then advances DRAIN.
- Avoids global atomic flush tax (historically 26–51% class).

Cost: **48KB banks at G=6** → ~1 WG/CU → few concurrent fat waves → dyn-VGPR never binds.

### 4.5 Group-split (L3, latest)

**Goal:** reduce LDS banks without losing B reuse.

```
G = 6 rowblks/tile
ACC_N = banks per group (3 → GROUPS=2; 2 → GROUPS=3)
GROUPS = G / ACC_N sequential passes
STAMP = (group << 28) | sti
B re-scanned per group as L2 hit (whole B column in L2)
POOL_N=1 keeps groups sequential (no bank race)
ACC_N=G => GROUPS=1 => byte-identical whole-tile path
```

| ACC_N | GROUPS | LDS/WG | WG/CU (target) | Silicon |
|---|---|---|---|---|
| 6 | 1 | 57600 | 1 | Oracle-clean small shape; coast-bound |
| 3 | 2 | 33024 | 1 (step toward bind) | **Deadlocks** |
| 2 | 3 | 24832 | **2** (occupancy win) | Not reached |

### 4.6 What one file owns (complexity density)

`occ_kernel_dsws_flow.s` (~2k lines) currently carries:

- dyn-VGPR grow/shrink + fat gauge  
- 3-role bodies + coast  
- 3-frontier pipeline  
- write-once reduce + completer store  
- tile-claim / split-K / group cursor  
- deadman + RETBARRIER  
- frontier snapshot  
- dormant conv_apply / conversion envelope  

This density is now a first-class project risk.

---

## 5. Silicon results (detailed)

Logs: `/home/kmbandy/dsws_gpu_logs/`. Safe whole-tile bin: `occ_dsws2_w16_flow_gd.bin` sha `77b9365e…` (16392B).

### 5.1 Run 3 — whole-tile w16, small shape (throttle path)

- Shape: `576×512×2048`, ACC_N=6, WAVES=16, n_kseg=64  
- **Oracle: ok=1152 bad=0**  
- coast≈496085, computed=9216, feed-stages=0, grow-fail=0  
- **coast-frac ≈ 98.2%**  
- TF printed ≈ 0  

**Read:** correctness works; almost all wave-time is coast. `feed-stages=0` is ambiguous (counter may only hit dedicated feed path; coast-feed does the real staging).

### 5.2 Run 4 / Run 5 — big shape whole-tile + FATMAX

Run 5 config: STAGINSTR=1, ACC_N=6, shape `3072×1024×2048`, pool=128, chunk=64.

| Signal | Value | Meaning |
|---|---|---|
| fatPeak occ[58] | **263** | Gauge valid |
| fatResidual occ[57] | **0** | fat_inc/dec balanced |
| Concurrent fat | 263/128 ≈ **2.05 / SIMD** | Need ~13–14 to bind |
| grow-fail | **0** | Pool never binds |
| coast vs computed | 858k vs 7.2k | Same coast pathology |
| Timeout | claim=64, live=0, fence=-- | Incomplete / aborted |

**Headline science result of the night:** traveling-peak dyn-VGPR is **not engaged**. Occupancy/bind work is justified; the system is ~**7× under** bind.

### 5.3 Run 6 — group-split ACC_N=3, oracle attempt

- Build `.text` 11624B  
- Small shape that whole-tile cleans  
- **Timeout:** `occ0(live)=0 occ20(claim)=26 fence=--`  
- STAGINSTR off → blind hang  
- claim=26 with hi=2 and ML8_POOL=24 matches **2 real tile claims + 24 terminal overclaims** → claimer ran; completion died  

### 5.4 Run 7 — group-split + unthrottled diag

- Same freeze from **+0.2s**  
- Unthrottled global snapshot spam → memory engine hot → MES REMOVE_QUEUE wedge → **MODE1 brick**  
- Lesson: high-rate global stores on a hung spin convert a quiet hang into a brick (same class as deadman message-bus spam)

### 5.5 Run 8 — not fired

Throttled STAGINSTR=1 + 200ms streamed frontier (`ASSIGN/STAGE/DRAIN/RB/BF/AR/barrier`). Correct next probe. Script: `fire_run8_groupsplit_frontier.sh`.

### 5.6 Earlier clean baselines (context)

- Whole-tile write-once at **w8** and some w16 paths: oracle CLEAN on small shape  
- Throughput numbers remain tiny vs 165 TF KWINBPF lineage — flow is a different vehicle  

**Correction to “safe bin” language:** safe means **oracle + no brick on known configs**, not “performance substrate ready for bind experiments.”

---

## 6. Group-split diagnosis (pre-Run-8)

### 6.1 Hang signature

```
+0.2s: live=0, claim=26, fence=--
```

- Not “stuck forever in WMMA”  
- All waves eventually decremented live  
- EOP fence never observed  
- Group-split-specific (whole-tile ACC_N=6 cleans same shape)

### 6.2 Implementation mismatches (prime suspects)

**A. STAGE completion still keys off `G`, not `ACC_N`**

In `occ_kernel_dsws_flow.s`:

- ASTAGE_R claims until `r >= G`  
- STAGE advances when `ARDONE >= G`  
- Compute claims only `local r < ACC_N`  
- Completer advances when `RBDONE >= ACC_N`  

Under `GROUPS>1`, every group super-tile still stages **all G A-rowblks** while compute drains **ACC_N**. Over-stage can still complete, but:

- Cost model of the lever is wrong (A-stage work not cut)  
- Any future “stage only ACC_N” edit without dual-side rewrite will hang or corrupt  

True group-split invariant should be:

```
per super-tile (tile, group, ksi):
  A stages: ACC_N (group-local)
  B stages: FN
  compute drains: ACC_N
  STAGE ready: ARDONE >= ACC_N && BFDONE >= FN
  DRAIN complete: RBDONE >= ACC_N
```

**B. Dual-keyed A addressing**

- Feed stores absolute `mblk*G + r` into ARES slots 0..G-1  
- Compute maps `actual = group*ACC_N + local` then loads ARES[actual]  
- Works only while every group pass re-stages the full 0..G-1 set  

**C. Completer / bank lifetime**

- Group-aware C base offset is present  
- Bank init still `ksi==0` store vs later add — safe only with strict sequential groups (`POOL_N=1`)  
- `POOL_N>1` without `(tile,group)` bank ownership reopens init/add races (Fable H1/H5)

**D. Completion / EOP class**

Fast `live=0` without fence suggests drain/retire/EOP protocol under group-split, not only math. Candidate classes:

| Class | Fits because |
|---|---|
| DRAIN never reaches ASSIGN; deadman/retire races | live→0 without full work |
| RETBARRIER / quiesce with many immediate-terminal WGs | 22/24 WGs terminal; 2 workers must finish |
| Coordinator FLOWTERM / cursor under `GROUPS*n_kseg` | premature RETIRE |
| Instrumentation brick path | Run 7 only (separate from hang) |

### 6.3 How to read Run 8 frontier

Stream line fields (occ[74..80]):

| Field | Stall meaning |
|---|---|
| `DRAIN < STAGE < ASSIGN` | Stalled between drain and stage |
| `RB < ACC_N` | Never finished compute for group |
| `BF < FN` or `AR < G` (or AR < ACC_N if fixed) | Staging never finished |
| `DRAIN < ASSIGN` with full slot counters | Completer / DRAIN-advance bug |
| `barrier < WAVES` | Exit-barrier never closed |
| `comp` rising vs 0 | Did any compute happen |

---

## 7. Deeper performance diagnosis

Even when oracle CLEAN:

```
coast-frac ≈ 98%
grow-fail = 0
fatPeak/SIMD ≈ 2
TF ≈ 0 (printed)
```

1. **Not issue-port bound right now.** The 165 TF / ~31 non-WMMA per 32 WMMA story was KWINBPF. Flow+write-once+emergent w16 is **pipeline starvation / coordination tax**.

2. **Dyn-VGPR cannot help until concurrent fat rises.** FATMAX is the proof.

3. **Emergent mix without STAGE lead is self-defeating.**  
   Seed ~13 compute + 3 feed floor → compute coasts immediately → everyone feeds → tiny compute → repeat. That is depth-1 ping-pong, not adaptive intelligence.

4. **`POOL_N=1` is the structural enemy of overlap.** Emergent design itself called `POOL_N=2` / shorter SEGK “Step 2” and warned Step 1 alone can regress. Group-split tried to fix occupancy without fixing pipeline depth — wrong order for TF.

5. **Write-once 48KB banks vs bind** is the real trilemma (VGPR whole-K vs LDS banks vs global atomic). Escapes: group-split (count), J-burst duty-cycle (GRESIDENT v2 Path A), or rejected WOFLUSH. Need one green escape before stacking another.

---

## 8. What is strong (keep)

1. **Hardware-honest stack** — forced dyn-VGPR past HIP/MES; real RDNA4 path.  
2. **Flow + coast + write-once** — right *kind* of architecture for a non-blocking economy.  
3. **Measurement culture** — oracle, FATMAX residual, brick taxonomy, byte-identical feature-off builds.  
4. **Ops discipline** — compositor chunking, deadman, safe-bin restore, real-disk logs, per-dispatch greenlight.  
5. **Novelty framing** — static warp-spec is known; **runtime multi-role rebalancing + mid-flight VGPR fungibility under a sum-envelope** is the real IP. RDNA4 is first substrate, not the whole product.  
6. **Night-end hygiene** — morning doc, restored safe bin, commit `03225f6bb`, fire scripts for Run 8.

---

## 9. Room for improvement

### 9.1 Process / program structure

- Too many concurrent state machines in one iteration.  
- L3 bind experiments on a failed M1 (coast) substrate.  
- “Adaptive” language ahead of the mailbox controller actually running.  
- Printed TF≈0 with CLEAN oracle is under-used as a hard gate.

### 9.2 Protocol / code

- `G` vs `ACC_N` predicate mismatch under group-split.  
- Single coast counter (need reason codes: no-assign / no-stage / grow-fail).  
- Dedicated feed-stage counter misleading when coast does staging.  
- Coordinator nudge still stubbed.  
- Group-split peppered as `.if GROUPS>1` instead of a clean protocol fork.

### 9.3 Performance path

- No demonstrated STAGE-lead.  
- No demonstrated grow-fail>0 on flow.  
- No competition yet with 165 TF static lineage.  
- Occupancy lever pursued before pipeline-depth lever.

### 9.4 Complexity / maintainability

- ~2k-line hand-asm single file + 6k-line dispatch harness.  
- IP portability story requires extracting an **abstract economy** from RDNA4 PM4 details; not started as a clean layer.

---

## 10. Recommended architectural changes

### 10.1 Milestone fence (do not cross-contaminate)

| Milestone | Goal | Success metric | Forbidden |
|---|---|---|---|
| **M0** | Flow correctness | Oracle CLEAN, fence fires, live=0 clean | Occupancy experiments |
| **M1** | Pipeline not coast-bound | **coast-frac < 30%**, STAGE usually ahead of DRAIN | Group-split, adaptive conversion |
| **M2** | Dyn-VGPR binds | **grow-fail > 0**, fatPeak/SIMD **≥ ~8** | New roles |
| **M3** | Adaptive mix | TF ≥ best static mix across shapes | New LDS layouts |
| **M4** | Portability | Second ISA backend of the *economy* | RDNA4-only PM4 in control law |

Hard gate proposal:

> No bind/occupancy experiment until coast-frac < 50% on the small shape with the safe whole-tile bin.

### 10.2 Order of levers

1. **`POOL_N=2` (or 3) whole-tile**, ACC_N=G — measure coast-frac and STAGE-lead.  
2. If unbound: **duty-cycle J** (burst owns several ksi) on whole-tile — GRESIDENT Path A, no group boundary hazards.  
3. Only if still LDS-bound: **group-split** with full ACC_N-scoped protocol rewrite.  
4. Only after bind: **mailbox adaptive nudge** (true DSWS moat).

### 10.3 Group-split as protocol fork (if kept)

```
super-tile = (tile, group, ksi)
staging    = ACC_N A + FN B
compute    = ACC_N
C store    = group slice on last ksi of group
banks      = zero-once per (tile,group), then always ds_add
```

Delete every remaining compare-to-`G` under `GROUPS>1` (STAGE, ASTAGE claim, docs saying ARDONE==G).

### 10.4 Separate fat emergence from role emergence

| Concept | Mechanism |
|---|---|
| Fat emergence | Hardware `s_alloc_vgpr` + sum-envelope / stagger |
| Role emergence | Who feeds vs computes (mailbox + floors) |

Hardware caps fat; it cannot invent staged operands. Ship static best mix on flow until M1/M2, or implement coordinator nudge before calling the system adaptive.

### 10.5 Instrumentation

Keep: throttled snapshots, 200ms host stream, fatPeak residual, safe-bin restore.

Add:

- Coast **reason codes** (no-assign / no-stage / grow-fail)  
- STAGE-lead histogram (`STAGE - DRAIN`)  
- Per-role time (compute / A / B / coast)  
- Never high-rate global stores on hang spin paths  

### 10.6 Performance truthfulness

Treat coast-frac and STAGE-lead as **primary KPIs** for the next phase; fatPeak/SIMD and grow-fail as KPIs of the phase after. The moat is real only after both gates.

---

## 11. Concrete backlog (priority)

1. **Fire Run 8** (throttled frontier) — localize group-split freeze; do not guess past the frame.  
2. If STAGE starved: fix A/B completion predicates for `GROUPS>1` (`ACC_N` vs `G`).  
3. If DRAIN starved with full stage: completer / RBDONE / group C-store.  
4. If `barrier < WAVES`: retire/EOP protocol, not GEMM math.  
5. **Park group-split** once characterized; ACC_N=6 only green path.  
6. **M1 sprint:** `POOL_N=2` whole-tile; measure coast-frac / STAGE-lead / TF.  
7. **M2 sprint:** J-burst or reduced banks only after M1.  
8. **Then** adaptive mailbox controller.  
9. Keep static 3-role / KWINBPF-class path as **performance baseline to beat**.

---

## 12. IP / moat framing (for product, not marketing)

**Strong claim (defensible):**

> A portable adaptive wave economy for FP8 GEMM: specialize roles, treat registers as a shared budget, sense starvation from pipeline/ring pressure, rebalance at safe boundaries — so shape becomes runtime state, not a new kernel binary. RDNA4 is the first substrate; backends are translations of the economy.

**Weak claim (avoid until measured):**

> One kernel, any shape, any FP8 GPU, no tuning, already works.

**Portable abstract machine (crown jewels):**

1. Role contracts (compute / A-feed / B-feed)  
2. Sum-envelope VGPR economics + traveling peaks  
3. Sensing (ring or frontier pressure) + hysteresis  
4. Convert-at-safe-boundary discipline  
5. Oracle + KPI harness  

**Not portable as-is:** PM4 arming, WMMA frag layout, LDS sizes, `global_load_tr`, MES/compositor constraints.

Industry already has static warp-spec and launch-time register budgets (CUTLASS / `setmaxnreg`). Differentiation is **mid-kernel multi-role rebalancing with fungible VGPR under a live envelope**. That is not running on the current flow bin until M3.

---

## 13. Bottom line

| Question | Answer |
|---|---|
| Is the architecture direction right? | **Yes** — especially the adaptive economy as long-term IP |
| Is the current w16 flow bin that economy? | **No** — coast substrate + dormant controller |
| Did the night produce useful science? | **Yes** — FATMAX under-bind + group-split hang class + brick lessons |
| What is the binding problem *now*? | **Pipeline coast / STAGE lead**, then LDS occupancy, then adaptation |
| Next silicon action | **Run 8 frontier**, then M1 `POOL_N` depth, not more group-split variants |

Night-end state is well saved (commit, morning doc, safe bin restore, real-disk logs). The work is at a healthy “measure, then sequence” point — not a dead end, and not yet a shippable adaptive kernel.

---

## 14. Key file index

| Path | Role |
|---|---|
| `spike/dvgpr_occ/occ_kernel_dsws_flow.s` | Current flow + group-split kernel |
| `spike/dvgpr_occ/occ_dispatch.cpp` | Raw PM4 harness, chunking, oracle, stream |
| `spike/dvgpr_occ/MORNING_2026-07-07.md` | Night-end pickup |
| `spike/dvgpr_occ/FLOW_ECONOMY_DESIGN.md` | Non-blocking flow design |
| `spike/dvgpr_occ/DSWS_EMERGENT_ECONOMY_DESIGN.md` | Emergent seed design |
| `spike/dvgpr_occ/DSWS_GRESIDENT_DESIGN.md` | Duty-cycle / bank occupancy design |
| `spike/dvgpr_occ/MAD305_DSWS_MASTER.md` | Campaign master |
| `spike/dvgpr_pm4/` | Dyn-VGPR PM4 unlock |
| `/home/kmbandy/dsws_gpu_logs/` | Run logs + fire scripts (Run 5–8) |

---

## 15. Suggested next sessions

1. Run 8 only: fire, capture frontier, one-page root-cause note.  
2. M1 design: `POOL_N=2` whole-tile plan with coast-frac KPI and brick budget.  
3. Optional: extract “abstract DSWS economy” doc for IP (no asm), separate from RDNA4 backend.

---

*End of review.*
