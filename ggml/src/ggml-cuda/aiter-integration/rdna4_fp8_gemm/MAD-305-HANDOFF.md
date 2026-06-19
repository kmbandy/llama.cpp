# MAD-305 — gfx1201 (RDNA4 / R9700) fp8 WMMA GEMM toward 250–300 TF
## State of Play & Multi-Model Handoff Brief

**Date:** 2026-06-17 · **HW:** AMD R9700 / gfx1201 (RDNA4), wave32, ROCm 7.2.3, KFD node 1 ·
**Branch:** `sync/upstream-2026-06-09` (all session code UNCOMMITTED) ·
**Repo root for this work:** `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`

> **Purpose of this document.** We are *not* abandoning the 250–300 TF goal; we are bringing in
> additional models to attack it with fresh eyes. This brief is written so a model with **no prior
> context** can become productive without digging through the codebase or the knowledge graph. It
> contains the goal, the hardware ceiling, the two kernel "vehicles," every wall we hit (with the
> measured numbers), what's settled vs still open, exact code locations + build/run commands, the
> operational gotchas (including how we wedged the GPU twice), and a prioritized list of where fresh
> eyes can help. The authoritative KG note index is at the end.

---

## 0. TL;DR (read this first)

- **Goal:** a hand-tuned gfx1201 **fp8 (e4m3) WMMA GEMM** that beats hipBLASLt (143 TF) and reaches
  toward a **250–300 TF** "pitch bar" for an AMD partnership. Target workload: **training, specifically
  wgrad-shaped GEMMs** (big-K). Stretch unit of merit: % of the **307 TF** raw-WMMA issue ceiling.
- **Where we are:** **155 TF** is the best *real fed* uniform GEMM (a conventional LDS-cooperative HIP
  kernel) — already **1.08× hipBLASLt**. Everything above 155 is blocked by a **silicon wall** we have
  now measured **from both possible vehicles**.
- **The wall (the central result):** On RDNA, **WMMA executes on the SIMD's VALU issue port**, which it
  shares with the operand-feed instructions (loads + address math). So a real fed GEMM is **issue-bound
  at ~54% (≈155 TF)** even with perfect latency hiding. The only lever to raise it is a **bigger
  register-blocked accumulator tile** (more WMMAs per load → fewer feed instructions per WMMA), which
  needs **>256 VGPR → dynamic VGPR**. dyn-VGPR is reachable on gfx1201 **only via raw PM4 + hand-asm**,
  on a **single-wave persistent micro-batch** kernel — and that vehicle is **occupancy-dead** (caps at
  ~3.5 waves/SIMD, can't hide the ~400-cycle feed latency → ~0.3 TF). **The occupancy↔fat-tile tension
  is the wall**, hit from both sides:
  - **HIP vehicle:** high occupancy, latency hidden → *issue-bound at 155*; fat tile would help but
    can't fit in the 256-VGPR static cap.
  - **PM4 micro-batch vehicle:** fat tile fits via dyn-VGPR → but *latency-bound at 0.3 TF*; occupancy
    capped at ~896 waves (~3.5/SIMD) can't cover the feed.
  **Conclusion (measured, not extrapolated): 250–300 TF on a *uniform* fp8 GEMM is not reachable on
  gfx1201 with either known vehicle.**
- **What is NOT a wall (proven):** the matrix unit itself does **305 TF = 99.5%** of peak when fed
  feed-free; the feed is the entire 305↔155 gap. `NOFEED` (operands reused from registers) on the
  *real* kernel hits **94%**. So the silicon is willing; the *feed instruction density on the shared
  issue port* is the limiter.
- **Forward (where friends help):** (1) build the **HIP LDS-cooperative "stream-K" merge** → safely
  lands ~155 (no wedge risk) = the shippable training kernel; (2) attack the **occupancy↔fat-tile
  tension** — the one genuinely open question is whether a *multi-wave-cooperative* micro-batch (share
  the A-tile in LDS across a wave-group) can keep occupancy **and** raise effective reuse; (3) the
  cheap-but-untested **feed-instruction-reduction levers** in HIP (`b128` wide loads, address-gen
  hoist, B-in-LDS); (4) the **MoE variable-tile** case where dyn-VGPR's load-balancing actually pays;
  (5) the **silicon dossier** for AMD (what they'd have to change).

---

## 1. The Goal & Business Context

- Build a gfx1201 fp8 GEMM that demonstrates we can drive AMD's matrix unit hard — an asset in an AMD
  partnership conversation. **Bar = 250–300 TF.** Reference points on this HW: **hipBLASLt = 143 TF**;
  our conventional kernel = **155 TF**; the raw fp8-WMMA issue ceiling = **307 TF** (= 15.9 WMMA/cycle).
- **Target workload = training, wgrad-shaped GEMMs.** wgrad = `gradᵀ @ input`, contracting over
  `batch×seq` tokens ⇒ shapes `M=hidden, N=intermediate, K=tokens` with **huge K**. Big-K is the
  regime where prologue/epilogue overhead vanishes and you'd hope to approach peak. (Earlier we briefly
  considered MoE; the user redirected to training because that's where 250–300 TF buys the most.)

---

## 2. Hardware & The Raw Ceiling

- **gfx1201 / R9700**, RDNA4, **wave32**, ROCm 7.2.3. `rocm-smi`: this is **node 1** (GUID 39578); the
  box also has a second smaller AMD GPU (node 2 / GPU 0) we don't use. `FCompute=128`.
- **307 TF fp8 WMMA ceiling = 15.9 WMMA/cycle.** This is the *matrix-unit issue ceiling*: back-to-back
  `v_wmma_f32_16x16x16_fp8_fp8` with operands already in registers, no feed. Measured in
  `bench/wmma_peak.hip`. Occupancy is NOT the matrix constraint — pure-WMMA holds 307 from occ 16 down
  to occ 6.
- **Why WMMA shares the issue port:** on RDNA (unlike CDNA/MI300, which has a *separate* matrix pipe),
  `v_wmma_*` is a VALU instruction issued on the SIMD's single VALU port. Every load (`ds_read`,
  `global_load_tr`), address calc, and loop-control instruction competes with WMMA for that one port.
  **This is the root physical fact behind the 155 wall.**

---

## 3. The Two Vehicles

There are exactly two ways we know to run this on gfx1201, and they have **complementary, mutually
exclusive** strengths. Understanding this is the whole story.

### Vehicle A — the HIP kernel (`bench/gemm_fp8_levers.hip`)
- One configurable static fp8 GEMM, every lever a template parameter
  (`TBM,TBN,TWM,TWN,TBK,AFEED,DBUF`). Compiled with `hipcc -O3 --offload-arch=gfx1201`. Runs under the
  normal HIP runtime — **cannot wedge the box.**
- **Winner config = `gemm_fp8_lever<128,128,2,2,32,AFEED=0,DBUF=1>` = 155 TF** @ 4096³ (≈1.08×
  hipBLASLt). This IS the "LDS-cooperative" method: `AFEED=0` stages the A-tile in **LDS**, shared by
  `TWM*TWN = 4` cooperating waves; B fed direct from L2 via `global_load_tr_b64` from a pre-shuffled
  tile-major buffer; A double-buffered (`DBUF=1`).
- **Codegen is excellent** (verified by reading the device ISA — see §5): 16 independent accumulator
  chains issued back-to-back, fine-grained `s_wait_loadcnt 0x6→0x0` release ladder, address math
  hoisted to pointer increments. hipcc has essentially nothing left to give at 4×4.
- **Hard limitation: 256-VGPR static cap.** The 4×4 accumulator already uses 183 VGPR. A bigger tile
  (6×4 = ~222 fits; **8×8 = ~512 does NOT**) — and the big tiles are exactly what would cut feed
  density. Bigger-than-256 needs dyn-VGPR, which HIP/LLVM will not emit for an `amdgpu_kernel` on
  gfx1201 (caps 256, spills).

### Vehicle B — the PM4 micro-batch (`spike/dvgpr_occ/occ_kernel_mbgemm.s` + `occ_dispatch.cpp`)
- Hand-written gfx1201 assembly, dispatched via **raw PM4 over libhsakmt (KFD)**, bypassing the HSA
  runtime. A **persistent pool of single-wave workgroups** each pull output tiles from a **global
  atomic work-queue** (stream-K style), and per tile use **`s_alloc_vgpr` to GROW** to a fat `FM×FN`
  accumulator, compute, ship, and **SHRINK** back to lean.
- **This is the ONLY way to get dyn-VGPR (a fat >256-VGPR tile) on gfx1201.** dyn-VGPR is armed by
  setting `COMPUTE_PGM_RSRC2` bit 6 in the PM4 dispatch (the MAD-304 lift).
- **Fully `-Wa,-defsym`-configurable levers:** `FM, FN` (tile), `DYNVGPR`, `BATCH` (tiles per atomic
  grab), `NOFEED` (isolation probe), `PROFILE` (in-kernel phase timers), and (added this session) a
  fine `WMMABUF_WAIT` release ladder + a `STAGGER` lever (lockstep-break, see §6).
- **Bit-exact** vs a chained-`wmma_ref` CPU oracle at every config.
- **Two hard problems (the reason it's not the throughput vehicle):**
  1. **Occupancy-dead.** Single-wave-per-tile caps at **~896 concurrent waves (~3.5/SIMD)** at 192 VGPR;
     bumping the pool past that admits nothing more. 3.5 waves/SIMD cannot hide the ~400-cycle feed
     latency (you need ~8). → **fed GEMM tops out at ~0.3 TF.**
  2. **Codegen quality is hard to match by hand** (a feed-free hand-asm chain managed only ~64 TF early
     on vs hipcc's 307 — though that was later shown to be partly under-feeding).

### The synthesis we wanted (and why it doesn't close)
"The merge" = put Vehicle A's codegen + LDS-cooperative feed-hiding **inside** Vehicle B's dyn-VGPR
fat-tile shell. But: **the fat tile (high VGPR) destroys occupancy, and occupancy is what hides the
feed.** You cannot have both a fat tile AND ~8 waves/SIMD on this silicon. That tension is the wall.

---

## 4. The Walls — Measured From Both Sides (the crux, with the data)

### 4a. HIP vehicle: issue-bound at 155 (the "ISSUE_PROBE", 2026-06-17 AM)
We built a bit-faithful clone of the 155 winner (`gemm_fp8_occ` in `gemm_fp8_levers.hip`, run with
`ISSUE_PROBE=1`) and ran it on **wgrad shapes** (`4096²×K16384`, `4096×14336×K16384`):
- **Occupancy sweep** (`__launch_bounds__` min-waves, fixed inner loop): **flat at 8.3 WMMA/cyc / ~160
  TF** across minWaves 4→12, all at ~20 waves/SIMD (~62% occ). Already latency-hidden; more waves do
  nothing.
- **Deep software pipeline** (`gemm_fp8_pipe`, prefetch A *and* B a tile ahead): **64–75 TF = HALF of
  the lever.** Hiding more latency made it *worse* — adding instructions to cover latency backfires
  because the bottleneck is the *issue port*, not latency.
- **`NOFEED`** (same kernel, same occupancy, per-step `ds_read`/`global_load_tr` removed, operands
  reused from registers): **284–289 TF = 92–94% = ~15 WMMA/cyc.** Removing the feed *instructions*
  (latency already hidden) nearly doubles throughput.
- **Verdict:** the 155 / 54% wall is **issue-port contention on feed instructions**, not latency. Per
  inner step/wave: **8 feed instructions (4 `ds_read` A + 4 `global_load_tr` B + per-`kk`
  `b_tile_offset` address math) feed 16 WMMAs.** The only way down is fewer feed-instructions/WMMA = a
  bigger reuse tile = dyn-VGPR.

### 4b. PM4 micro-batch vehicle: latency-bound at 0.3 TF, occupancy-capped (2026-06-17 PM)
We applied the **lockstep-stagger fix** (phase-offset persistent waves by `TGID_X×STAGGER` so their
synchronized feed stalls interleave — the highest-leverage lever predicted by KG `50147c07`) and ran
the **full `pool × stagger` grid** on `4096²×K8192`, BATCH=1:

| pool | maxlive | STAGGER=0 | STAGGER=16 | STAGGER=64 |
|------|---------|-----------|------------|------------|
| 768  | 768 | 0.3 TF | 0.3 TF | 0.3 TF |
| 1152 | **896** | 0.3 TF | 0.3 TF | 0.3 TF |
| 1536 | **896** | 0.3 TF | 0.4 TF | 0.3 TF |

- **Dead flat — 0.3–0.4 TF / 0.02 WMMA/cyc across all 9 cells.** Neither more occupancy (768→896) nor
  any stagger amount moved the needle.
- **Concurrent occupancy hard-caps at `maxlive=896` (~3.5 waves/SIMD).** Requesting pool 1152 or 1536
  both land at 896; the HW won't admit more single-wave workgroups at this VGPR footprint.
- **Verdict (pre-registered decision rule):** flat ⇒ the **single-wave micro-batch is occupancy-dead
  for feed-hiding.** 3.5 waves/SIMD cannot bury a ~400-cycle feed stall no matter how phased. The
  stagger fix is **measured dead** (full grid, not one lever).

### 4c. The combined conclusion
- **305 TF** (micro-batch feed-free) and **94%** (HIP `NOFEED`) prove the matrix + the silicon can do
  it. **155 TF** (HIP fed) is the issue-port wall. **0.3 TF** (PM4 fed) is the occupancy wall.
- ~~The fat tile is the only lever past 155; it requires dyn-VGPR; dyn-VGPR requires the occupancy-dead
  single-wave vehicle. **No third vehicle exists.** ⇒ 250–300 on a uniform fp8 GEMM is not reachable
  on gfx1201.~~
  **SUPERSEDED (2026-06-19).** A *third vehicle* now exists and is the live winner: the **wave-group
  cooperative GEMM** (`spike/dvgpr_occ/occ_kernel_wggemm2.s`) — neither the HIP kernel nor the
  single-wave micro-batch. It is at **162 TF (8×2 reuse tile @ TWN=4), first config to beat HIP 161**,
  with levers still open (KWIN=2 occupancy, big-reuse tiles). It runs on the **same raw-PM4 harness
  that arms dyn-VGPR**, so the fat-tile lever is reachable on it — dyn-VGPR is NOT tied to the dead
  single-wave vehicle. The "250–300 not reachable" conclusion is retracted; the climb is in progress.

---

## 5. The hipcc Inner-Loop ISA (the codegen reference)
Extracted via `hipcc -save-temps=obj` → the device `.s`. The 155 winner's hot loop:
```
.LBB0_9:
  s_wait_loadcnt_dscnt 0x700
  v_wmma_f32_16x16x16_fp8_fp8 v[121:128], v[149:150], v[158:159], v[121:128]
  s_wait_loadcnt 0x6  ; v_wmma ... (B frag 1)
  s_wait_loadcnt 0x5  ; v_wmma ... (B frag 2)
  s_wait_loadcnt 0x4  ; v_wmma ... (B frag 3)
  ... 12 more WMMAs, NO waits ...           ; 16 distinct accumulators v[1:8]..v[121:128]
  v_add_nc_u32_e32 v181, 32, v181           ; the ONLY address math in the loop (hoisted)
  s_cbranch ...
```
Key facts for anyone trying to hand-write or transplant this:
- **16 independent accumulator chains** issued back-to-back = the ILP that covers WMMA latency.
- **A is in LDS** (`ds_read`, tracked on `dscnt`), so `loadcnt` is **B-only** → a clean `0x6→0x0`
  release ladder. (Feeding A direct-from-L2 muddies the ladder because A then lands on `loadcnt` too.)
- **All address math hoisted** to two pointer-increments per 32 WMMAs.
- Aggregate over the whole file: **1104 WMMAs vs ~5000 address-arithmetic instructions** — even hipcc's
  *best* drowns WMMAs in feed/address ops. That IS the 54% issue wall in the instruction stream.

---

## 6. dyn-VGPR Status — SETTLED, do NOT re-derive (we burned time re-deriving this 3×)
- **dyn-VGPR is ARMABLE on gfx1201 compute via raw PM4** (set `COMPUTE_PGM_RSRC2` bit 6; MAD-304,
  commit `133f9d151`). It is **NOT "AMD-locked."** It is the basis of the whole micro-batch vehicle.
- There is **no clean HIP/compiler path**: the LLVM `amdgpu-dynamic-vgpr-block-size` attr is inert,
  caps 256 + spills; the descriptor enable bit is gfx1250-only; dyn-VGPR codegen is bound to
  `cs_chain`. `s_alloc_vgpr` reaches the ISA **only via hand-asm / inline-asm.**
- dyn-VGPR's throughput value as an **occupancy** lever is **weak (+1–13%)** — confirmed repeatedly.
  Its *real* value is enabling **big reuse tiles (>256 VGPR)** that cut feed density, and **variable
  expert tiles** (MoE). It is NOT an occupancy play.
- **The fat-file deadlock (operational, important):** if `occ × per-wave-VGPR` approaches the SIMD
  register file, the persistent waves lockstep-deadlock and **wedge the GPU (reboot required)**. This
  session that manifested as **`pool=2048 × 192 VGPR = 100% of the VGPR file, zero slack → hang.`** Keep
  the persistent pool well under the file (≤ ~896 waves admit at 192 VGPR here anyway).

---

## 7. Code Locations & How to Build/Run

All paths relative to `ggml/src/ggml-cuda/aiter-integration/rdna4_fp8_gemm/`.

| File | Role |
|------|------|
| `bench/wmma_peak.hip` | The 307-TF raw-WMMA ceiling microbench + occupancy sweep. |
| `bench/gemm_fp8_levers.hip` | **Vehicle A.** The configurable HIP GEMM + the `pipe` deep-pipeline kernel + the `ISSUE_PROBE` clone (`gemm_fp8_occ`) + a K-depth sweep + `PROFILE_WINNER` rocprof path. |
| `bench/trfeed_common.h` | `preshuffle_B` (tile-major trperm) + `b_tile_offset` for `global_load_tr`. |
| `spike/dvgpr_occ/occ_kernel_mbgemm.s` | **Vehicle B kernel.** Fed fat-tile micro-batch; levers `FM,FN,DYNVGPR,BATCH,NOFEED,PROFILE,STAGGER`; `LOADBUF`/`WMMABUF`/`WMMABUF_WAIT` macros. |
| `spike/dvgpr_occ/occ_dispatch.cpp` | **Vehicle B host harness** (KFD PM4). Modes `--microbatch --mbgemm --mbprof --merge` (+ older prongs). `run_mbgemm()` = alloc/preshuffle/dispatch/oracle/time. Crash-survivable `prog()` disk logger. |
| `spike/dvgpr_occ/build.sh` | RAM-capped (`MemoryMax=4G`) `-defsym` build matrix `[1a]…[1e]` + CPU oracle + harness link. |
| `spike/dvgpr_occ/RESULT.md`, `RESULT_P3.md`, `RESULT_MBGEMM.md` | Phase result dossiers. |
| `docs/superpowers/specs/2026-06-17-dynvgpr-fat-tile-fp8-gemm-merge-design.md` | The merge design spec. |
| `docs/superpowers/plans/2026-06-17-dynvgpr-fat-tile-fp8-gemm-merge.md` | The merge implementation plan (T1–T9). |

**Build & run the HIP vehicle (safe, no PM4):**
```bash
cd bench
hipcc -O3 --offload-arch=gfx1201 gemm_fp8_levers.hip -o gemm_fp8_levers
./gemm_fp8_levers                 # full lever sweep + 4096^3 vs hipBLASLt + K-sweep
ISSUE_PROBE=1 ./gemm_fp8_levers   # the issue-vs-latency probe (occupancy sweep, NOFEED, pipe)
```

**Build & run the PM4 vehicle (SUPERVISED / wedge-capable — read §8 first):**
```bash
cd spike/dvgpr_occ
MEMMAX=4G ./build.sh              # assembles all -defsym bins, CPU oracle, links occ_dispatch (no GPU)
# Then, headless + backgrounded + crash-logged:
MERGE_BIG=1 timeout 60 ./occ_dispatch --merge   # pool x stagger sweep; progress -> /tmp/occ_merge_progress.log
./occ_dispatch --mbgemm           # tile/reuse sweep + NOFEED probe
./occ_dispatch --mbprof           # in-kernel REALTIME phase breakdown
```

---

## 8. Operational Gotchas (READ before touching the PM4 vehicle)
- **Raw PM4 can WEDGE the GPU (reboot).** We did it twice this session. The cause both times traced to
  **VGPR-file oversubscription** (`pool × per-wave-VGPR` at/over 100% of the file). Once wedged, a
  hung kernel makes `hsaKmtDestroyQueue` block, so the process can't exit and an outer `timeout` may
  fail to reap it. **Mitigations we now use:** keep `pool` well under capacity (≤768–896 here); **run
  backgrounded** (`run_in_background`) so a hang can't kill the controlling terminal; wrap in
  `timeout`; and the GPU recovers when the process is finally killed (confirmed via `rocm-smi`).
- **Crash-survivable logging is in place.** `occ_dispatch.cpp` has a `prog()` helper that writes each
  step to `/tmp/occ_merge_progress.log` and **`fflush`+`fsync`s it to disk BEFORE every GPU dispatch**.
  After a hard hang/reboot, the last `STARTING …` line names exactly which config wedged. This turned
  blind reboots into precise root-causing — **keep using it for any new PM4 sweep.**
- **rocprof CANNOT see the PM4 kernels** (they run under the HSA layer rocprof hooks). All PM4 timing
  uses an **in-kernel `s_sendmsg` REALTIME counter** validated at exactly **100 MHz** (`--timercheck`).
  rocprof CAN see the HIP kernels (that's how we got the 8.6 WMMA/cyc on the 155 kernel).
- **Power-of-2 tile-grid decode:** the micro-batch needs `N/(16*FN)` a power of two (no ISA divide:
  `ti → row=ti>>LOG2, col=ti&MASK`). So `4096²` works (`4096/64=64`), but realistic wgrad `N=14336`
  (`/64 = 224`, non-pow2) does **not** without a general-divide kernel change.
- **K must be a multiple of 32** (KT even; unroll-by-2; tail pair peeled).
- **`SQ_INSTS_VALU` reads 0 on gfx1201** (counter gap); use `GRBM_GUI_ACTIVE` for busy% on HIP kernels.
- **The work is SUPERVISED** (user-present for PM4), and currently **headless/remote** — favor
  backgrounded + disk-logged runs.

---

## 9. The Levers — Pulled (with results) and Remaining

**Pulled & measured (don't re-run blind):**
- Wide feed (preshuffle + `global_load_tr`, no LDS for B): 68→140 TF. **THE Phase-1 win.**
- A in LDS + double-buffer (the 155 winner). +12% over Phase-1.
- Bigger static tiles (6×4=222 VGPR, 4×8/8×4=256 spill): **lose** (occupancy / spill).
- Explicit deep A+B software pipeline (`gemm_fp8_pipe`): **regressed to 64–75 TF** (issue pressure).
- micro-batch BATCH-grab (amortize the atomic): real 2–3× (the single device-scope atomic was a
  serializer) but not dominant.
- micro-batch reuse / fat tile (FM×FN): ~2× per shape step but VGPR-capped (~2.4 reuse even at 256).
- micro-batch K-depth sweep: **flat** for a fed GEMM (every K-step re-pays the feed; K is a non-lever).
- micro-batch **lockstep-stagger** (this session): **flat / dead** across the full pool×stagger grid.
- Occupancy as a lever (both vehicles): weak/dead (HIP already saturated; PM4 capped at 896).

**Genuinely open / untested (where fresh eyes help — see §10):**
- **`b128` wide loads** + **address-gen hoist** + **B-in-LDS reuse** in the HIP kernel — partial
  feed-instruction-count reduction *without* a bigger tile. Untested. Could move 155 some amount.
- **6×4 / 5×4 static tiles in HIP** at the wgrad shape specifically (only swept at square shapes).
- **Multi-wave-cooperative micro-batch** (the real open question): a persistent *wave-group* (not
  single wave) that shares the A-tile in LDS — combining occupancy (many waves) with effective reuse
  (LDS sharing) — i.e., rebuild the 155 kernel's trick *inside* the stream-K queue. This is the only
  idea that might break the occupancy↔fat-tile tension.

---

## 10. Where Fresh Eyes Can Help (prioritized, concrete)

1. **Build the HIP LDS-cooperative "stream-K" merge (SAFE, shippable, ~155 TF).** Take
   `gemm_fp8_levers.hip`'s 155 winner (`AFEED=0, DBUF=1`), swap its grid scheduler for a **persistent
   atomic work-queue** (the micro-batch's load-balancer), keep the LDS-cooperative A path. Runs under
   HIP (no wedge risk). On a uniform GEMM it lands ~155; its *distinct* value shows up on **ragged /
   MoE shapes** where the queue avoids the tail bubble a tiled kernel leaves. This is the concrete
   product deliverable.
2. **Attack the occupancy↔fat-tile tension (the only path past 155).** Can a **2- or 4-wave-cooperative
   micro-batch** keep ~8–20 waves/SIMD AND raise effective reuse by sharing A in LDS across the
   wave-group? If yes, the feed density drops without per-wave VGPR exploding. This is the open
   research question; everything else is engineering.
3. **The cheap HIP feed-reduction levers** (`b128` loads, hoisted addressing, B-in-LDS) — quantify how
   far 155 can move *without* dyn-VGPR. Low risk (HIP), low effort, never measured.
4. **The MoE variable-tile case** — the 35B-A3B MoE fp8 GEMM, where dyn-VGPR's variable-K tiles +
   stream-K load-balancing actually stack and beat a tiled kernel. dyn-VGPR's real product home.
5. **The silicon dossier for AMD** — write up the measured wall: fp8 WMMA caps at 155 on gfx1201
   because WMMA shares the VALU issue port (no separate matrix pipe like CDNA), and the fat-tile escape
   is blocked by the lack of a compiler dyn-VGPR path that preserves occupancy. This is the partnership
   ask, now backed by data from both vehicles.

---

## 11. Open Questions for Fresh Eyes
- Is `maxlive=896` (~3.5 waves/SIMD at 192 VGPR) a VGPR limit, a workgroup-slot limit, or something
  else? If we could get to 8 waves/SIMD *without* the full-file deadlock, the stagger might finally pay
  off. (We only have the empirical cap; the mechanism isn't pinned.)
- Does a multi-wave-cooperative micro-batch (LDS-shared A) actually break the tension, or does the LDS
  capacity / barrier cost reintroduce the same ceiling?
- Is there a fp8 WMMA variant on gfx1201 with **larger K per instruction** (16×16×32/64) that would
  change the feed:WMMA instruction ratio? (We've only used `16x16x16`.)
- Can `global_load_tr_b128` (vs `_b64`) halve B-load instruction count cleanly on this ISA?

---

## 12. KG Note Index (the authoritative trail — query these for full detail)

Read via the `mneme_*` tools on the `mad-lab-memory` MCP (scope=self). **Read-first:**
- `2601d691` — **ISSUE_PROBE result** (HIP issue-bound at 155, NOFEED=94%, pipe=½, fat tile = the lever). Corrected; supersedes the deleted `949ed7c5`.
- `cd407a9b` — **the crux**: compute:load ratio = reuse; single-wave VGPR-capped; LDS-cooperative is the conventional kernel's trick. (Now empirically confirmed by this session's stagger grid.)
- `324cefb7` — 305-TF feed-free ceiling (corrected 276→305); 155 = the LDS-cooperative kernel; `--mbprof` 69% feed-stall breakdown.
- `c709589e` — the **merge plan** (stream-K + shared-memory tiling); MoE is its real win.
- `50147c07` — the lockstep diagnosis + the **stagger lever** (now measured DEAD this session — its prediction did not hold because occupancy hard-caps at 3.5/SIMD).
- `e4e1c1ef` — the issue-port-wall physics (right physics).
- `7e7f6e94` — per-tile framework overhead (NOFEED isolation method; small-K regime).
- `7ea1acbe` — the fed fat-tile build + first fed numbers + the double-buffer fix.
- `19c981fb` / `a43cbce3` / `2ff53769` / `f50237e2` — breakthrough(276), resume pointers, the WMMA/cyc convergence thesis.
- `40cd2823` — the **fat-file deadlock** (the wedge mechanism).
- `0dbcb65f` — **dyn-VGPR armable via PM4** (check-first; do not regress to "AMD-locked").
- `59e9b8eb` — feedback: measure, don't extrapolate; the user decides when to stop.

**This document supersedes the "next lever = stagger" framing.** The next moves are §10 items 1–5.

---

## 13. Disposition (the honest bottom line for incoming models)
- **155 TF is a real, shippable win** (1.08× hipBLASLt) and is the practical ceiling for a *uniform*
  fp8 GEMM on gfx1201. Build it safely in HIP.
- **250–300 on a uniform GEMM is blocked by silicon** (issue-port sharing) + tooling (no occupancy-
  preserving dyn-VGPR path). We measured this from both vehicles; it is not an unexamined wall.
- **The remaining upside is real but specific:** (a) the multi-wave-cooperative idea (§10.2) is the
  only uniform-GEMM lever left untried; (b) MoE/variable-tile is where dyn-VGPR genuinely pays; (c) the
  measured wall is itself the AMD pitch. Bring friends to attack (a), (b), and the cheap HIP levers in
  parallel.
