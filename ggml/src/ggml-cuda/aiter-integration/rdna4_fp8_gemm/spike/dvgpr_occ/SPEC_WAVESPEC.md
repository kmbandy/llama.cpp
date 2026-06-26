# Spec: Lean Wave-Specialized fp8 GEMM with dyn-VGPR (MAD-305 #323)

Date: 2026-06-21. Companion: [[WAVESPEC_RESEARCH.md]] (NVIDIA setmaxnreg → AMD
s_alloc_vgpr, primary-source verified).

## Goal

Build, from scratch, a persistent wave-specialized fp8 (e4m3) WMMA GEMM for
gfx1201/wave32 that launches LEAN (small per-wave VGPR footprint) and uses
dyn-VGPR (`s_alloc_vgpr`, armed via raw PM4 RSRC2 bit6) to keep dedicated LOADER
waves at 32 VGPR while only COMPUTE waves carry accumulators. The bet: the freed
register budget raises resident-wave occupancy enough to hide the measured 26%
unhidden B-feed wall on the feed-latency-bound substrate, beating the 165 TF
static winner.

Clean-room build (not a fork of `occ_kernel_wggemm2.s`) for an uncontaminated
attribution of the lean+wave-spec+dyn-VGPR lever. A/B/barrier/oracle machinery is
COPY-AND-ADAPT from existing kernels, not greenfield.

## Why this is the untested regime (not a re-run of closed results)

- Big-tile + dyn-VGPR thesis is CLOSED (2026-06-16): dyn ≤ static at a fat tile;
  reuse/occupancy tension fights itself; fat tile blows the dyn cap. We do NOT
  pursue fatter tiles.
- dyn-VGPR measured 0% overhead / ~0% benefit — but ONLY on the compute-bound
  micro-batch vehicle (never feed-latency-bound). "Occupancy HURTS" was measured
  feed-FREE.
- The winner (8x2 KWINBPF, 165 TF) IS feed-latency-bound (ATT: s_wait_loadcnt ≈
  99% of stall; 26% unhidden B-feed). That is the ONE regime where more resident
  waves can convert. This spec puts lean+wave-spec+dyn there for the first time.

## Architecture

- New kernel: `occ_kernel_wavespec.s` (gfx1201, wave32, hand-asm).
- New dispatch: `WAVESPEC` mode in `occ_dispatch.cpp` — clone of
  `run_wggemm_perf`, but arms dyn-VGPR: `BuildPgmRsrc2(true)` (RSRC2 bit6) + the
  umr `SQ_DYN_VGPR.BLOCK_SIZE=1` cap-lift to 256 (line ~530 pattern).
- New `build.sh` target with `-defsym` knobs + oracle-gate + RGA.
- Reused verbatim: CPU e4m3 oracle (`fp8_oracle.cpp`), `rga_check.sh` (KSRC
  re-point), rocprofv3 `--pmc`/`--att` flow.

## Dispatch model: persistent atomic-claim queue (the shippable shape)

Fixed pool of persistent workgroups sized to fill the GPU. Per WG:

1. Leader wave `atomic_add`-claims the next output tile index `ti` from a global
   work-counter (the elastic queue).
2. Leader `ds_store ti` to LDS broadcast slot → `s_barrier` → all waves read `ti`.
3. Loaders + compute (below) cooperatively process tile `ti`, full-K.
4. Ship C for `ti`.
5. Loop to (1) until the counter exceeds total tiles (drained). Self-leveling
   load balance, no static partition.

Claim-broadcast is the proven `occ_kernel_wggemm2.s` pattern (#295 resolved the
atomic-claim wall). Claim is at WG granularity (not per-wave) so loaders and
compute agree on which tile's B to stream.

## Kernel: two wave roles in one workgroup

Workgroup launches `NLOAD + NCOMP` waves.

### Loader waves (`wid < NLOAD`)
- At entry: `s_alloc_vgpr 32` (LEANREG) — stay lean for the kernel's life.
- Per claimed tile, loop over K: `global_load_tr` a B-slice from HBM →
  `ds_store` into the B-ring in LDS → participate in the K-window barrier. Run
  one K-window AHEAD of compute (double-buffered ring).

### Compute waves (`wid >= NLOAD`)
- At entry: `s_alloc_vgpr <TILE_VGPR>` GROW to the tile's accumulator footprint.
  Check SCC; on fail (must not happen at lean tiles) branch to a safe stall.
- Per claimed tile, loop over K: barrier-wait for the published B-slot →
  `ds_load` A (from A-LDS) + `ds_load` B (from B-ring) → `FM*FN` WMMAs → advance.
- After full-K: ship C. If `COMPSHRINK`, `s_alloc_vgpr LEANREG` before reclaiming
  the next tile, then re-GROW after claim (default off).

The asymmetry is the lever: a STATIC kernel allocates every wave at the fat
compute size (the max any wave needs); dyn-VGPR lets loaders sit at 32 → more
waves resident → occupancy to hide the feed.

## Data flow / LDS layout

- A tile in LDS (cooperative fill, reused across the K-window — KWIN pattern).
- B-ring in LDS, `RINGDEPTH` slots (2 = double-buffer): loaders publish slot
  (t+1) while compute drains slot (t).
- `ti` broadcast slot.

## Synchronization

No mbarrier on RDNA. Producer→consumer handoff = `s_barrier` across the WG, one
per K-window, with the double-buffered B-ring giving loader/compute overlap. This
is the proven KWINBPF prefetch-one-ahead cadence, lifted.

## dyn-VGPR no-fail-grow guarantee

Lean tiles are far under the cap: 2x2 ≈ 54 live, 4x4 ≈ 166 live, both ≪ 256
(BLOCK_SIZE=1 via umr). Loaders hold only 32 each. The compute GROW is therefore
satisfiable by construction. We still test SCC after every `s_alloc_vgpr` and
branch to a safe stall on failure (defensive; should never fire at swept tiles).

## Knobs (`-defsym`, all sweepable)

- `FM`, `FN` — lean tile. FIRST BUILD: 2x2 (FM=2, FN=2, ~54 live).
- `NLOAD` — loader waves (sweep 1/2/4).
- `NCOMP` — compute waves.
- `RINGDEPTH` — B-ring slots (2).
- `LEANREG` — loader lean target (32).
- `DYNVGPR` (0/1) — 0 = role-split with STATIC fat allocation (the WSPEC-only
  number); 1 = lean loaders + grow compute. Gives dyn-VGPR attribution from one
  extra build.
- `COMPSHRINK` (0/1) — compute shrink-while-claiming (default 0).
- `STORE` (0/1) — perf (no C store) vs correctness (store C for oracle).

## Instrumentation (EVERY config, before deciding next — standing rule)

- RGA: `rga_check.sh` (KSRC=occ_kernel_wavespec.s) — livereg peak-live + ISA.
- rocprofv3 `--pmc` — occupancy / issue counters; `--att` — stall breakdown
  (confirm feed-wait % drops vs the 165 winner).
- In-kernel `PROFILE` timer (lifted from mbgemm) — compute / feed-wait / barrier
  / claim phase split.

## Correctness gate (HARD, before any TF number counts)

- CPU e4m3 oracle (`fp8_oracle.cpp` / `test_fp8_oracle.cpp` pattern): stored C
  must match the oracle at EVERY tile config.
- Build green: assembles for all swept FM/FN/NLOAD.

## Success criteria

- PRIMARY: does `DYNVGPR=1` lean wave-spec beat BOTH the `DYNVGPR=0` static-role
  baseline AND the 165 TF static winner, in the feed-bound regime?
- SECONDARY: rocprof shows higher occupancy + lower feed-wait % at DYNVGPR=1.
- GATE: oracle-correct C at every config.
- HONEST PRIOR (per 0dbcb65f): dyn-VGPR occupancy is historically a WEAK lever
  (+3–13%) for this GEMM. A null result here (lean+wave-spec doesn't convert in
  the feed-bound regime either) is a publishable closing of the lever space, not
  a failure. We report the measured number either way.

## Build order (implementation plan — NOT building yet)

1. CPU oracle + frag layout header (TDD, CPU-green).
2. `occ_kernel_wavespec.s`: persistent claim + role-split + B-ring, NO dyn
   (static alloc). Oracle-gate at 2x2 static (STORE=1).
3. Add `DYNVGPR` grow/lean + SCC guard; arm dyn in the WAVESPEC dispatch
   (RSRC2 bit6 + umr cap-lift).
4. `build.sh` matrix (defsym sweep) + CPU-green + RGA gate.
5. [SUPERVISED — STOP for go] first raw-PM4 KFD GPU dispatch; baseline smoke +
   oracle.
6. [SUPERVISED] tile/NLOAD sweep with rocprof+RGA on every cell.
7. RESULT_WAVESPEC.md + KG bank + Jira.

## Safety

- First GPU run = raw-PM4 KFD dispatch → SUPERVISED (stop for go).
- NEVER pass `--gl2c` (dormant MES-crash landmine).
- gfx1201 = R9700 = display GPU; a hang resets monitors.

---

# EXTENSION (2026-06-25): Rolling dyn-VGPR allocation — the "traveling peak" / ping-pong scheme

> Added after the multi-wave dyn-VGPR cooperative kernel was proven RUNNING + ORACLE-CORRECT
> at P=1 (KG `8a9ce97f`; root cause of the prior deadlock = pre-grow out-of-range VGPRs > v15,
> `17f209af`). The static wave-spec above (NLOAD/NCOMP fixed roles) was the substrate. THIS is
> the architecture it was always walking toward — and it is the **third vehicle** the 2026-06-17
> "250–300 TF NOT reachable; there is no third vehicle" conclusion (KG, MAD-305-HANDOFF) said
> did not exist. It did not exist *then* because multi-wave dyn-VGPR deadlocked. It exists now.

## The idea (CDNA "ping-pong", expressed through dyn-VGPR allocation)

Instead of fixed fat compute waves, run a **chain of compute waves whose fat allocation is a
traveling peak**. Each wave's VGPR footprint over its lifetime is a trapezoid:

```
lean(32) → s_alloc_vgpr GROW → ░peak(112) / WMMA burst░ → s_alloc_vgpr SHRINK → lean(32)
```

Phase-offset the waves so the peaks never coincide:

```
wave A:  lean → GROW → ░peak░ → SHRINK → lean
wave B:         lean →  GROW → ░peak░ → SHRINK → lean
wave C:                lean →  GROW → ░peak░ → SHRINK ...
                               ↑ B grows into the registers A just freed
```

The per-wave grow→burst→shrink machinery **already exists** in `occ_kernel_coop.s` today
(`s_alloc_vgpr NFV` grow ~L325, `s_alloc_vgpr 32` shrink ~L431, per tile). The missing piece is
the **cross-wave phase handshake** that staggers P≥2 compute waves so their grow/shrink windows
interleave. That is scheduling on proven plumbing, not new plumbing.

## Why one rule does everything: the sum-envelope constraint

Let `B` = per-SIMD VGPR budget, `V_peak` = a compute wave's peak (112 here), `V_lean` = 32.
The design rule (kmbandy, 2026-06-25): **the SUM of all resident waves' *instantaneous*
allocations must stay below `B` at every instant — so your concurrent peak budget is the
*average* footprint, not the max.** Worked example (his numbers): budget 1000, peak 500 →
profile `0 < 250 < 500 < 250 < 0` across phase-offset neighbors, each averaging 250 → **four**
waves sum to 1000 with the running sum pinned at budget and never over. Four waves in a budget
that holds only two at peak.

This single rule does **three jobs at once**:

1. **Occupancy ≈ doubles.** A symmetric trapezoid has time-average ≈ `V_peak/2`, so resident
   waves ≈ `B / (V_peak/2)` = **2×** the all-at-peak count `B / V_peak`. This is the lever that
   breaks the occupancy↔fat-tile tension that was declared *the* wall: fatness becomes TEMPORAL,
   not simultaneous — full-size peak tile AND high average occupancy.
2. **No grow ever fails (brick-avoidance).** If the instantaneous sum stays below `B` by ≥ one
   grow-step, every `s_alloc_vgpr` succeeds. This is the exact inverse of the measured
   `pool=2048` (=100% VGPR file, zero slack) deadlock. The reserved slack IS the forward-progress
   guarantee.
3. **ISA §3.3.3.2 unreachable by construction.** "Multiple waves simultaneously blocked needing
   max allocation" cannot occur when the envelope guarantees ~one peak at a time. The
   shrink-before-next-grow is the "software scheme for forward progress" the ISA invites.

## The master knob: burst length (duty cycle fights itself)

The duty cycle is the tuning fulcrum and it is self-opposing:

- **Long WMMA burst** (wave holds peak many K-windows): great matrix efficiency, grow/shrink cost
  amortized — but higher duty cycle → average creeps toward peak → loses the 2× occupancy.
- **Short burst**: low duty cycle → max occupancy gain — but needs MORE concurrent waves to keep
  the matrix unit saturated and pays grow/shrink overhead more often.

**Optimum = the shortest burst that still keeps the WMMA pipe full.** Burst length (K-windows
held at peak before shrink) is the primary sweep parameter for the rolling kernel.

## Prerequisite measurement (before building the chain)

The duty-cycle headroom only converts to throughput **if VGPR is the binding occupancy
constraint.** The 2026-06-16/162 TF finding hinted one occupancy cap was *structural (tile
geometry), not VGPR*. So FIRST, off the working P=1 base, measure:

- (a) the actual per-SIMD VGPR budget `B` on gfx1201 (RGA / ISA / occupancy probe);
- (b) what currently binds occupancy on the cooperative kernel — VGPR, LDS, or WG-slots
  (rocprofv3 `--pmc` resident-wave + the maxlive counter).

If VGPR-bound → the rolling scheme is the unlock and there is a clear runway to 250–300. If
structural → fix geometry first; the rolling design still stands, it just pulls a different lever.

## Build order (extension — NOT building yet)

R0. Measure occupancy binding-constraint on the P=1 base (above). Decision gate.
R1. P≥2 compute waves with the **cross-wave phase-stagger handshake** (LDS phase tokens; reuse
    the lock-free prod/cons counter style — NO s_barrier). Each wave grows on its phase slot,
    bursts, shrinks; the envelope (Σ < B − slack) enforced by the stagger period.
R2. Oracle-gate the staggered chain (correctness FIRST, every config).
R3. Sweep burst-length × stagger-period × NCOMP; rocprof+RGA every cell; watch maxlive and
    feed-wait %. Verify the Σ-envelope never hits zero-slack (brick guard).
R4. RESULT + KG bank; verdict vs 162 TF and vs the static wave-spec baseline.

## What this reopens

The 2026-06-17 pre-registered conclusion was honest for its evidence: two vehicles (HIP
issue-bound at 155; single-wave PM4 micro-batch occupancy-capped at ~3.5 waves/SIMD, 0.3 TF),
no third. The rolling-dyn-VGPR cooperative chain is a genuinely distinct third vehicle that
attacks the *exact* wall (occupancy↔fat-tile) those two could not. NOT a claim of success — a
reopened, measurable runway. The old "stagger DEAD" result (50147c07) was a DIFFERENT mechanism
(phase-offset *independent single* waves to interleave *feed* stalls), so it does not pre-refute
this; but it is a standing caution that occupancy is the variable to watch like a hawk.

---

## SESSION STATUS — 2026-06-26 (resume pointer; SUPERSEDES the 2026-06-25 status below it)

> **2026-06-26 correction of the record.** The 2026-06-25 status (retained verbatim further down for
> history) named TWO wrong causes for the pool≥2 brick. Both were disproven on 2026-06-26. Do not act on
> them. The confirmed cause + fix are here.

**Proven (banked, still solid):**
- The dyn-VGPR cooperative kernel (`occ_kernel_coop.s`) RUNS + is ORACLE-CORRECT. The 2-session
  "deadlock" was pre-grow out-of-range VGPRs (>v15 unbacked at lean-16 dyn launch); fixed by gating
  all pre-grow LDS/atomic temps to v11/v14 (KG `17f209af`, `8a9ce97f`).
- **pool=1**: `oracle CLEAN ok=256 bad=0`, fence FIRED, clean retire, clean teardown. Genuinely good.
- **pool=2**: `oracle CLEAN ok=256 bad=0` BOTH shapes — i.e. the per-tile cooperative B-handoff is
  numerically correct with two workgroups co-resident. BUT the run does NOT terminate cleanly (see below).

**WRONG theories, both disproven 2026-06-26 (kept so we don't re-chase them):**
1. ~~"terminal C store doesn't drain → EOP won't fire" → fix with `STOREWAIT` (`s_wait_storecnt 0x0`
   before `s_endpgm`)~~. REFUTED on silicon: `coop_dyn_pool2_storewait.log` — fence still `--`, same late
   brick. The compute wave never REACHES `s_endpgm`, so there is no store to drain.
2. ~~"ISA §3.3.3.2 multi-grower contention — two growers on one SIMD, the loser spins at the grow-retry"~~.
   RETRACTED: two separate WGs can land on different SIMDs where each is the sole grower, yet the fence
   never fires regardless. Contention can't explain a placement-independent failure.

**CONFIRMED ROOT CAUSE (read it in the kernel's own comment at `.Lcompute_loop`):** the compute
terminal is a **pool=1-only diagnostic stub**. Compute counts tiles it processes (`s57`) and exits at
`s57 == TOTAL` — valid ONLY when ONE workgroup owns all TOTAL tiles. At pool≥2 the WGs' feeds SPLIT the
TOTAL tiles via the shared global atomic claim (`offset:20`), so each compute receives only its share
(<TOTAL); the count NEVER reaches TOTAL; after its last real tile the compute loops to `.Lwait_epoch`
and waits forever for a tile that went to the sibling WG → that WG never retires → the dispatch never
reaches end-of-pipe → EOP `RELEASE_MEM` never fires → the queue is never IDLE → ANY reclaim
(`hsaKmtDestroyQueue` OR process-exit) wedges the GPU. This explains everything at once: pool=1 clean /
pool=2 brick (terminal is pool=1-only), oracle CLEAN at pool=2 (the ~16+16 tiles that DID run are
correct), STOREWAIT a no-op (never reaches the store), and the `compPh=7`-not-`8` signal in every
pool=2 stream (stuck pre-terminal). The author (me, prior session) flagged it: *"General pool>1
terminal needs an lds_barrier-class rendezvous; TBD."*

**✅ FIX CONFIRMED ON SILICON (2026-06-26, `coop_dyn_pool2_poolterm.log`):** pool=2, both shapes →
`compPh=8` (was stuck 7), **`fence=FIRED`** (was `--`), clean teardown (no WARN), `oracle CLEAN
ok=256 bad=0`, exit 0, dmesg clean, **user-confirmed NO brick**. First-ever clean pool=2 dyn-VGPR
cooperative run — correct AND terminates AND GPU survives. The pool≥2 teardown brick is CLOSED.
KG `0a2cea44` (confirm), `dac0bb8c` (root cause). The live `_d1_gd.bin` slot now holds the POOLTERM bin.

**FIX (gated behind `POOLTERM` defsym):** the feed
already broadcasts a per-WG terminal signal — on its terminal claim it writes `ti≥TOTAL` into the
per-WG LDS `TI_OFF` and bumps epoch BEFORE `.Lfeed_exit`. So compute now exits when it observes a
broadcast `ti≥TOTAL` (checked on the RAW ti, before the SAFEPROBE clamp pins it to TOTAL-1). This is
the per-WG feed→compute "lds_barrier-class rendezvous" — correct for ANY WG count; the old count
terminal is kept as the pool=1 fast path. `POOLTERM=0` (default) → static d0 **byte-identical** to
`.clean_bins` (verified 1716B exact); `DYNVGPR=1 POOLTERM=1` assembles (1784B) and disasm shows the
`s_cmp_ge_u32 s17,s11 → .Lcompute_exit` at the compute-loop top. STOREWAIT is refuted — leave it off.

**NEXT (in order):**
1. **ONE gated pool=2 run** with a `DYNVGPR=1 POOLTERM=1` bin (build fresh; do NOT reuse the STOREWAIT
   bin). Predict: `compPh=8`, **`fence=FIRED`**, clean destroy, **no brick**. If fence still `--` →
   the terminal handshake has a hole (e.g. compute parked at `.Lwait_prod` mid-tile when the feed
   exits) → back to offline trace. ONE run, gated, streamed — NOT a sweep.
2. If clean → **R0 occupancy attribution** (VGPR vs LDS vs WG-slots; RGA: compute peak-live ~81 /
   HW-allocated 120 → ~40 VGPR over-allocated → trim NFV toward ~96 = free occupancy). Step pool ONE
   at a time, abort on any non-clean exit.
3. Then **`ML8_P=2`** (two compute waves, one feed, one WG) — the real reuse/throughput lever
   (`reuse: P=2→2.0, P=3→2.4, P=4→2.67`). Note: P>1 puts two growers in one WG, the genuine §3.3.3.2
   territory the rolling-stagger (R1–R4) is designed for. P=2 may also need its own terminal review
   (1 feed → P consumers; `min_cons` already loops over P).

**Standing safety — CORRECTED 2026-06-26: a GPU brick is a BUG, never an "accepted tax."** Recoverable
via MODE1/Hyprland-safe-mode ≠ acceptable. FREEZE dyn GPU dispatch whenever the current build is
known to leave the queue non-idle; only a config we have offline reason to believe retires cleanly
earns a single gated run. Every run streams to disk; never `--gl2c`; bounds gate + SAFEPROBE + padding
stay on; ONE dispatch at a time, NO sweeps. Kernel-correct ≠ teardown-safe ≠ GPU-survives.

---

## SESSION STATUS — end of 2026-06-25 (resume pointer)  [SUPERSEDED — see 2026-06-26 above]

**Proven this session (banked, solid):**
- The dyn-VGPR cooperative kernel (`occ_kernel_coop.s`) RUNS + is ORACLE-CORRECT — root cause of the
  2-session deadlock was pre-grow out-of-range VGPRs (>v15 unbacked at lean-16 dyn launch); fixed by
  gating all pre-grow LDS/atomic temps to v11/v14 (KG `17f209af`, `8a9ce97f`).
- **pool=1**: `oracle CLEAN ok=256 bad=0`, fence FIRED, clean exit.
- **pool=2 (multi-WG co-residency)**: `oracle CLEAN ok=256 bad=0` on BOTH shapes (down + down_pf),
  maxlive=2, both WGs claim+process all tiles, retire. **[WRONG — see 2026-06-26: the WGs SPLIT the
  tiles and one never retires; §3.3.3.2 claim retracted.]**

**BLOCKER (open) — the teardown/EOP brick:** at pool≥2 the EOP `RELEASE_MEM` fence NEVER fires
(confirmed: not late — never within 5s), because the compute wave's terminal C store does not drain
before `s_endpgm` → the queue is never IDLE → tearing it down wedges the GPU. **[WRONG diagnosis — see
2026-06-26: cause is the pool=1-only count terminal; STOREWAIT refuted.]** The harness teardown
guard (`occ_dispatch.cpp` run_mbcoop: wait ≤5s for fence, else skip destroy + leak) makes the harness
exit CLEAN (we now get the oracle verdict) BUT only DEFERS the brick to process-exit. (KG `3027411a`, `b9dff173`.)

---

# mad-lab-DSWS — Dynamic Staggered Wave-Spec (2026-06-26, coined by kmbandy). THE target architecture.

This is the completion of the Rolling dyn-VGPR / traveling-peak idea above — it names the **missing
precondition** without which that idea cannot work on GEMM: **split-K to shrink the peaks.**

## Why the rolling/stagger idea doesn't work on the CURRENT (full-K) kernel
The stagger's whole benefit (avg footprint ≈ V_peak/2 → ~2× occupancy; sum-of-instantaneous < budget →
no `s_alloc` failure → §3.3.3.2 unreachable) assumes **brief, trapezoidal peaks**. A register-blocked
GEMM with long K (ml8 `down` K=9216) holds its fp32 accumulators (~64 of 112 VGPR) at peak for the
**entire K-loop** → ~95% duty cycle → a high-duty **square wave**, not a trapezoid → avg ≈ peak (NOT
peak/2). So:
- staggering buys **no extra occupancy** (you can't time-multiplex a plateau), and
- high pool **oversubscribes** the per-SIMD VGPR → the pool=64 global deadlock (compute can't re-grow →
  unbounded `s_alloc` spin → feed-drain cascade → freeze).
This is why dyn ≈ static was measured before (2026-06-20, 0% benefit): the kernel was full-K.

## The fix IS the design: split-K
Work item becomes **(output tile, K-SEGMENT)** instead of (output tile). Each compute wave:

    grab (tile, seg) → s_alloc GROW(112) → accumulate ONE short K-segment
                     → atomic-flush partial sum to C → s_alloc SHRINK(lean) → grab next

Now the peak lasts **one segment** (brief) → duty cycle drops → the profile becomes the trapezoid the
stagger needs → the staggered grow/shrink handshake keeps `sum(instantaneous alloc) < budget` at every
instant → **2× occupancy + no `s_alloc` failure + §3.3.3.2 deadlock unreachable by construction.** This
is the first version where **dyn-VGPR earns its keep on GEMM**: temporal register multiplexing needs
duty < 100%, and split-K is what creates it. "Shrink the peaks" = split-K = THE design, not a side quest.

## What the build must PROVE (don't assume a win)
- Split-K isn't free: each segment **flushes a 64-VGPR partial** (atomic-add fp32 → C) + re-zeros = `S×`
  more C-write traffic + atomic contention. The bet: brief-peak staggering occupancy gain **>** split-K
  reduction overhead. Measure it.
- The cooperative **B-feed/sharing** structure changes when waves split K (P compute waves currently
  share one B-column-tile; split-K changes what's shared) — resolve in design.
- The full-K kernel is **feed-bound** (FED 156 vs NOFEED 287 TF). DSWS must beat that, and the lever may
  be the feed, not occupancy. Get the full-K baseline TF (via the compositor-safe sweep) as the number
  to beat.

## Enablers already built (2026-06-26) — make building+measuring DSWS safe on the single display GPU
- **POOLTERM** — multi-WG cooperative retire, no teardown brick (KG `dac0bb8c` / `0a2cea44`).
- **Compositor-safe chunking** — bounded dispatches + yield → real-shape perf measurable imperceptibly
  on the R9700 that also drives the desktop (KG `21827908`). `ML8_COOP_CHUNK` / `ML8_COOP_CHUNK_MAXS`.
- Still needed before a high-pool sweep: a **fast per-chunk hang-abort** (current per-chunk poll timeout
  is 25s — a deadlocked chunk would freeze the display that long; cap it to ~1.5s). The *proper* fix is
  the stagger's **coordinated grow** (wait-for-budget instead of spin-fail) = graceful degradation, no
  deadlock by construction.

KG: `dc9faf0d` (this decision), `5fedf098` (original rolling idea, now superseded/completed by DSWS).

---

# DSWS v2 — Adaptive Wave-Role Economy (2026-06-26, kmbandy). SUPERSEDES the v1 framing above.

> The v1 DSWS section above frames it as "split-K to shrink peaks → stagger → 2× occupancy." That's
> **wrong about the wall.** Measured data (RESULT_WGGEMM.md) says occupancy is NOT the bottleneck —
> `minWaves 4→8` is FLAT at ~160 TF. The real wall is the **VALU issue port**. v2 corrects this.

## The measured wall (hard numbers, from RESULT_WGGEMM.md + RGA stats — SQUARE shapes)
- HW ceiling: **307 TF = 15.9 WMMA/cyc** (fp8); feed-free ceiling **272 TF**.
- 165 TF static winner (HIP 4×4-dbuf lever; PM4 8×2 clones it): **161 TF = 52.5%**, **8.34 WMMA/cyc**,
  VGPR **183** (PM4 8×2 = 207), per-wave 4×4 = 16 frags = **128 acc VGPR**, resident **5 blk / 20 wv**.
- **Occupancy is FLAT 4→8 waves** → NOT the wall.
- **The wall = the shared VALU issue port:** ~**31 non-WMMA issues per 32 WMMA** (8 B-load + ~8 A-load
  + 9 waitcnt + ~6 ptr/ctrl). Half the issue slots are feed, not math → 52% of ceiling. **Lever = cut
  non-WMMA issues per WMMA**, NOT add waves.
- ⚠️ CAVEAT (kmbandy): these are **SQUARE** shapes (4096², 4096×14336, K=16384), NOT ml8 shapes. The
  HW ceiling + the issue-port *mechanism* transfer; the 161 number and the exact 31:32 ratio do NOT.
  We still need the raw-fp8 baseline TF + `--att` issue-mix measured on ml8 `down` (M=2048 K=9216
  N=2560) — now safe via the compositor-safe chunking.

## DSWS v2: the design
One kernel, three **fungible** wave roles sharing the per-SIMD/WGP VGPR budget:
**compute (fat) · A-feed (lean) · B-feed (lean).** dyn-VGPR makes the budget tradeable across roles,
**asymmetrically**: shrink/retire 1 fat compute wave (~112–256 VGPR) → fund **3–7 lean feed waves**
(32 each). The kernel **senses the bottleneck at runtime** (the prod/cons **ring counters already in
`occ_kernel_coop.s`**: B-ring draining empty = starved for B = make B-feed; rings full = compute-bound
= make compute) and **rebalances the role mix** to it — per shape, per moment. Split-K's job is NOT
occupancy; it **creates the headroom** (brief compute peaks → budget frequently freeable → feed waves
grow on demand instead of compute camping the SIMD all K). This directly attacks the issue-port wall:
wave-spec puts feed instructions on **separate** waves/SIMDs so compute SIMDs issue **near-pure WMMA**,
and the **optimal feed:compute split is shape-dependent**, so a self-balancing kernel beats a static
one across diverse ml8 shapes.

## Prior art (researched-from-memory; VERIFY via deep-research before building)
- **Warp specialization + lean-producer/fat-consumer via dynamic register realloc is ESTABLISHED on
  NVIDIA Hopper (CUTLASS 3.x):** producer warps `setmaxnreg.dec`, consumer warps `setmaxnreg.inc`
  (`setmaxnreg` = NVIDIA's `s_alloc_vgpr`). The "ping-pong"/"cooperative" CUTLASS schedulers are this.
  → The lean/fat wave-spec MECHANISM is **not** our moat; NVIDIA ships it.
- **Our narrower novelty claims:** (1) doing it on **RDNA4 via raw PM4** (AMD toolchain won't emit
  `s_alloc_vgpr` for compute GEMM — RT-only in Mesa); (2) **RUNTIME-ADAPTIVE rebalancing** — CUTLASS's
  producer:consumer split is **static** (compile-time, one-time realloc at launch). A kernel that
  **senses + shifts the role mix mid-flight** is the candidate-genuinely-novel piece. **The adaptivity
  is the moat, if unclaimed.**

## Immediate plan (post-compact)
1. **Deep-research pass** (NVIDIA / AMD / arXiv): does anyone do runtime-adaptive warp/wave-role
   rebalancing (vs CUTLASS static warp-spec)? AND mine CUTLASS warp-spec `setmaxnreg` mechanics to
   copy, not reinvent (mbarrier handoff, realloc timing, ping-pong cadence). Happy to borrow if it
   exists; if not, we're first on the adaptive bit.
2. **ml8-shape ground truth** — add a fast per-chunk hang-abort (current 25s → ~1.5s) so the pool
   sweep can't re-brick at the oversubscription point; then measure on `down`: raw-fp8 baseline TF +
   `--att` issue-mix (is it issue-port-bound, at what feed:WMMA ratio?). N=2560 < 4096 → less B-reuse
   → plausibly MORE feed-bound → wave-spec even more valuable.
3. **DSWS v2 brainstorm → spec → plan → build** on the cooperative substrate (POOLTERM + chunking).

KG: `63583120` (DSWS v2), `dc9faf0d` (v1, superseded), `21827908` (chunking), `dac0bb8c`/`0a2cea44`
(POOLTERM). Note: this raw-fp8 substrate is separate from the ml8 LUT-dequant front-end, which adds
gather/select instructions → worsens the issue mix → the raw-fp8 ml8 number is an UPPER bound.
