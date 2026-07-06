# DSWS v2 — In-Kernel Phase Profile (compute-wave time breakdown)

**What this is:** measured, in-kernel timing of where the DSWS v2 split-K compute wave actually
spends its cycles — *not* inferred from lever-pulls. Each compute wave stamps `GET_REALTIME` (RTC,
100 MHz) at every phase boundary, accumulates per-phase ticks in registers, and emits them once at
retire. Host sums across all compute waves and prints ticks + % per phase.

This doc is the running record: every profiling run's **args, shape, and breakdown** goes in the
table below. Add rows; don't overwrite.

---

## The six phases (compute wave critical path)

| phase | what it measures |
|---|---|
| `FOLLOW_WAIT` | idle — spinning until the claimer publishes the next super-tile (coordination latency) |
| `STAGE_WAIT`  | idle — spinning until the A/B feed waves have staged this super-tile's operands |
| `GROW`        | rowblk claim + `s_alloc_vgpr` dyn-VGPR grow 32→112 |
| `WMMA`        | LDS frag loads + `v_wmma_f32_16x16x16_fp8` compute over the SEGK segment |
| `FLUSH`       | split-K C reduction — `global_atomic_add_f32` per (frag,elem), drained |
| `SHRINK`      | `s_alloc_vgpr` dyn-VGPR shrink 112→32 |

## Methodology / how to reproduce

- **Kernel:** `occ_kernel_dsws.s`, gated on `-Wa,-defsym,PHASEPROBE=1`. Accumulators in SGPRs
  `s78..s83` (last-stamp RTC in `s77`); single `phase_flush` atomic-add to `occ[64..69]` (bytes
  256–276, above the per-chunk memset) at compute retire. **PHASEPROBE=0 → `.text` byte-identical to
  the safe production bin `c62568f6`.**
- **Build (per run):** `DSWS2=1 FM=2 FN=4 G=6 SEGK=64 SAFEPROBE=1 DIAG=0 TFPROBE=1 PHASEPROBE=1
  NOCFLUSH=<0|1> NCOMP=<c> NAFEED=<a> NBFEED=<b> DSWS2_CONV=0`
- **Run (per run):** `ML8_POOL=16 ML8_COOP_CHUNK=0 DSWS2_SEGK=64 DSWS2_NKSEG=32 DSWS2_ORACLE_MTL=3
  DSWS2_ORACLE_NTL=8 ./occ_dispatch --dsws2 <mix>` — **single pass** (no `DSWS2_TARGET_SECS`).
- **Shape:** `576×512×2048`, super-tile `192×64` (G=6 FM=2 FN=4), `n_kseg=32`, `TOTAL_super=768`,
  `POOL=16` WGs. (M=576 = the proven-brick-safe regime; the distribution is shape-robust.)
- Logs on real disk: `~/dsws_gpu_logs/phase_<label>.log`.

### Measurement caveats (read before trusting a number)
1. **Single-pass distribution.** Accumulators are u32; a sustained run would overflow. The **%
   distribution** is the deliverable, not absolute ticks. It's a ratio → robust to run length.
2. **Uniform RTC-read bias.** Each `phase_stamp` costs ~one `s_sendmsg GET_REALTIME` round-trip,
   charged ~uniformly across phases. It slightly *over*-weights the tiny phases (WMMA, GROW, SHRINK),
   so their true share is **≤** what's shown. Big phases (FLUSH, STAGE_WAIT, FOLLOW_WAIT) are robust.
3. **Artifact fixed 2026-07-04:** the first version did a global atomic *per stamp*; the kernel's
   `s_wait_storecnt` drained those instrumentation writes into the FLUSH phase (bogus FLUSH=25% even
   with NOCFLUSH=1, and 33× WMMA swings). Now accumulate-in-reg + single-flush → NOCFLUSH correctly
   reads FLUSH≈1%. **All rows below are post-fix.** Pre-fix numbers are discarded.

---

## Results (576×512×2048, CONV=0, DYNVGPR=1, single pass)

| wave-start (c/a/b) | probe | FOLLOW | STAGE | GROW | WMMA | FLUSH | SHRINK | log |
|---|---|--:|--:|--:|--:|--:|--:|---|
| **4c2a2b** (4/2/2) | —        | 16.6% | 35.5% | 1.0% | 2.5% | **44.1%** | 0.3% | `phase_4c2a2b.log` |
| **6c1a1b** (6/1/1) | —        | 22.1% | **49.2%** | 0.6% | 1.9% | 25.9% | 0.2% | `phase_6c1a1b.log` |
| **2c3a3b** (2/3/3) | —        | 15.3% | 26.1% | 2.1% | 4.4% | **51.4%** | 0.7% | `phase_2c3a3b.log` |
| **4c2a2b** (4/2/2) | NOCFLUSH | 41.7% | 44.6% | 3.3% | 8.0% | 1.2% | 1.3% | `phase_4c2a2b_nocflush.log` |
| **4c2a2b RING** (D=2) | FIX 1a | **10.1%** | **15.1%** | 1.0% | 2.4% | 71.0% | 0.3% | `ring_phase_4c2a2b.log` |

### Stacked view (each bar = 50 chars = 100% of that config's compute-wave time)
```
              FOLLOW▓  STAGE▒  GROW·  WMMA:  FLUSH█  SHRINK·
4c2a2b   ▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒·:████████████████████████·
6c1a1b   ▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒·:█████████████·
2c3a3b   ▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒·::███████████████████████████·
4c2a2b   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒·::█·   (NOCFLUSH)
NOFLUSH
ring2    ▓▓▓▓▓▒▒▒▒▒▒▒·:███████████████████████████████████·   (FIX 1a D=2)
```

### FIX 1a result (2026-07-04): the two waits cut in half, oracle bit-exact
The D=2 ring-of-slots pipeline (`occ_kernel_dsws_ring.s`, run with `DSWS2_RING=1`) double-buffers
super-tiles so compute drains slot A while feeds stage slot B. Measured at the same `576x512x2048`
brick-safe shape, `4c2a2b`, single pass — **and it verified bit-exact (oracle CLEAN, max_rel=0,
occ[0]==0, no hang) on the first GPU run of the barrier-free protocol.**

| metric | single-slot (lockstep) | RING (D=2) | Δ |
|---|--:|--:|--:|
| FOLLOW_WAIT | 16.6% | **10.1%** | −6.5 pts |
| STAGE_WAIT  | 35.5% | **15.1%** | −20.4 pts (>halved) |
| **FOLLOW+STAGE (the target)** | **52.1%** | **25.2%** | **−26.9 pts (halved)** |
| FLUSH (split-K C-writes) | 44.1% | **71.0%** | +26.9 pts — same abs work, leaner timeline |
| WMMA / GROW / SHRINK | ~3.8% | ~3.7% | unchanged (dyn-VGPR still ~1%) |

**Reading:** the coordination + feed-stall waits collapse (as designed — next slot pre-published AND
pre-staged), so compute waves now spend ~74% of their time on *real* work vs ~48% before. The residual
STAGE_WAIT (15%) is the feed:compute ratio (fix #3 — only NBFEED-1 B-stagers at this mix). What's left
standing at **71% is split-K's 32× C-write amplification** — now unambiguously the wall, exactly as the
"3 fixes then split-K" plan predicted. **Next quantification:** a TFPROBE/sustained-reps wall-clock run
to confirm the fraction-shrink converts to a real speedup (phase % is a distribution, not wall-time).

---

## Findings (measured — this replaces every earlier *inferred* bottleneck claim)

1. **C-write reduction (`FLUSH`) is the #1 cost: 26–51% of compute-wave time.** This is split-K's
   32× write amplification (`global_atomic_add_f32`, each C cell hit `n_kseg=32` times). Confirmed by
   NOCFLUSH: removing it drops FLUSH to ~1% and frees ~44 points. **The single highest-leverage fix.**

2. **Feed staging (`STAGE_WAIT`) is the #2 cost: 26–49%.** Compute waves spend a *third to a half* of
   their time blocked waiting for the A/B feeds to stage the current super-tile. This **overturns the
   earlier "feeds keep up" inference** (which was based on claimer-sampled ring occupancy, not the
   compute wave's own stall). Worst at 6c1a1b (1 feed each can't keep 6 compute fed → 49%).

3. **Coordination (`FOLLOW_WAIT`): 15–22%.** The per-super-tile claimer handshake latency.

4. **The dyn-VGPR "moat" (`GROW`+`SHRINK`) is NEARLY FREE: 0.8–2.8%.** The `s_alloc_vgpr` grow/shrink
   every rowblk costs almost nothing at runtime. **It is NOT the bottleneck.** (Matches the older
   "measured GROW/SHRINK ≈ 0%" result.) Whatever the dyn deadlock-at-scale problem is, runtime cost
   is not why DSWS is slow.

5. **Actual compute (`WMMA`) is 2–4%.** The kernel spends ~96% of compute-wave time on
   writes + waiting, ~2–4% computing. The ceiling is entirely a memory/coordination problem.

### Wave-start effects
- **More compute (6c1a1b):** STAGE_WAIT ↑ to 49% — feeds starve; per-wave flush work ↓ (25.9%).
- **More feeds (2c3a3b):** STAGE_WAIT ↓ to 26% — but only 2 compute waves each do more rowblks →
  FLUSH ↑ to 51%.
- **Balanced (4c2a2b):** STAGE 36% / FLUSH 44% — the split the substrate ships with.

### Where this points
- Kill the C-write amplification (WG-local reduce-then-flush-once) → attacks the 26–51% FLUSH.
- Deepen the feed pipeline so compute never blocks on STAGE → attacks the 26–49% STAGE_WAIT.
- Neither touches dyn-VGPR — leave the moat, it's free.

---

## Run log (append new profiles here)
| date | shape | mix | probe | notes | log |
|---|---|---|---|---|---|
| 2026-07-04 | 576×512×2048 | 4c2a2b | — | baseline balanced | `phase_4c2a2b.log` |
| 2026-07-04 | 576×512×2048 | 6c1a1b | — | compute-heavy → feed-starved | `phase_6c1a1b.log` |
| 2026-07-04 | 576×512×2048 | 2c3a3b | — | feed-heavy → flush-bound | `phase_2c3a3b.log` |
| 2026-07-04 | 576×512×2048 | 4c2a2b | NOCFLUSH | isolates C-write cost (oracle intentionally fails) | `phase_4c2a2b_nocflush.log` |
| 2026-07-04 | 576×512×2048 | 4c2a2b RING | FIX 1a | D=2 ring; FOLLOW+STAGE 52%→25%, oracle bit-exact | `ring_phase_4c2a2b.log` |

---

## Next: finish the architecture — the 3 core fixes (do these BEFORE judging the arch)

**Framing (the thing we keep forgetting): the architecture has never actually run the way it's
meant to.** The accounting/conversion economy was wired wrong the whole time — frozen at 4/2/2, then
runaway to 1/6/1, then the sensor + zero-slack-budget bugs. So *every* performance verdict so far —
including "split-K is the wall" — was drawn on an unfinished substrate. The path is to **finish these
three, re-measure, and only then judge.** Split-K, if it's still a problem after, is addressed *after*.

### 1. Fix the waits — background accounting ("paddle"), not per-wave wait→calc→adjust
Today every wave, every super-tile, blocks on the synchronous quiesce handshake (measured
**FOLLOW_WAIT 15–22%**) and compute blocks on feeds (**STAGE_WAIT 26–49%**). Replace it: the
accounting runs **in the background** and only *intervenes* when a lane needs more. Waves always
flow; a shared **paddle** (target role split) biases which role a *freed* wave pulls next — the
floodgate opens a little wider toward the starved lane, no strict stop-and-recalculate.
- Discrete-wave caveat: the paddle sets a target waves **pull** toward; it does not push fractional flow.
- Hard constraint: respect the ISA dyn-VGPR stagger rule — never multiple waves growing to max VGPR
  at once on a shared pool (that is the M=1920 deadlock). The paddle model staggers naturally; we
  gate the grow explicitly anyway.

### 2. Fix the hardcoded 8 waves
`WAVES=8` is *our* arbitrary choice, not a hardware limit. It should be **derived from a
max-VGPR-by-wave-type budget** — fat compute (112 VGPR) + lean feeds (32) against the per-SIMD/WGP
VGPR pool — and be **settable**, not baked. The wave count falls out of the budget, so we fill the
hardware instead of capping at 8.

### 3. Fix the 1-compute ⇄ 1-feed coupling
The economy converts **1:1** (one compute ⇄ one feed), but a fat compute wave needs **≥2 lean feed
waves** to stay fed. The profile proves it: **6c1a1b** (1 A + 1 B feeding 6 compute) → **STAGE_WAIT
49%**, starved. Targets/conversions must hold the correct **feed:compute ratio**, not a 1:1 swap.

**Sequence:** (1) → (2) → (3), re-running the phase profile after each. These three are the core of
the architecture; the accounting was simply wired incorrectly. Only after all three do we revisit
split-K / the C-write amplification.
