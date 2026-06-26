# RDNA4 FP8 GEMM: New Experiment Ideas

Date: 2026-06-19

This note intentionally avoids restating the existing `RESULT_WGGEMM.md` ladder. It is a list of new or under-tested mechanisms to try after the current 8x2/KWIN work.

## Current Constraint Model

The current best path is the wave-group PM4 kernel, with the 8x2 reuse tile reaching about 162 TFLOPS. The important model correction is asymmetric feed cost:

- B feed per MAC is roughly `1/FM`, and B is the binding side because it comes from `global_load_tr_b64`.
- A feed per MAC is roughly `1/FN`, but A is LDS-staged/shared and has not been the binding side.
- The 8x2 tile improved throughput by reducing B feed even though total feed/MAC increased.
- After 8x2, the wall moved: `FED < FEEDONLY`, so the next gains likely require better WMMA/feed overlap, register scheduling, residency, or a more radical B-feed reduction.

## 1. Lane-Major B Layout With Plain Loads

Replace `global_load_tr_b64` with ordinary `global_load_b64` from a prepacked lane-major B-fragment layout.

Current B layout is designed so `global_load_tr_b64` produces the WMMA B fragment. Instead, make the stored B buffer already match the exact per-lane WMMA source bytes:

```text
Bfrag[kt][ntile][frag_n][kk][lane][8 bytes]
```

Then lane `L` loads:

```asm
global_load_b64 v[fb:fb+1], vaddr_lane8, s[Bfrag_base] offset:...
```

Why this is interesting:

- It bypasses the special transpose-load path entirely.
- It may allow `global_load_b128` to load two adjacent B fragments per lane, giving a real 2x cut in B load instructions without requiring `global_load_tr_b128`.
- B is already repacked, so changing the repack contract is acceptable.

First experiment:

1. Add a tiny host repacker for one 16x16 B fragment into per-lane bytes using the proven `frag_layout.h` map.
2. Write a single-WMMA oracle kernel using `global_load_b64` for B.
3. If correct, add a `BPLAIN=1` variant to `occ_kernel_wggemm2.s`.
4. Then test `global_load_b128` as two adjacent B frags from the same lane-major layout.

Pass signal:

- Single-tile oracle passes bit-exact.
- 8x2 `BPLAIN_B128` reduces B issue count and moves FED toward or above FEEDONLY.

## 2. VGPR/Register Coloring Sweep

The 8x2 path likely has operand-collector, VGPR bank, or scheduling effects. The current register layout is tightly packed:

```text
ACC -> FA -> FB
```

Try padding the bases without changing the logical algorithm.

Suggested knobs:

```asm
ACC_STRIDE = 8, 9, 10, 12, 16
FA_PAD     = 0, 4, 8, 16
FB_PAD     = 0, 4, 8, 16
```

The simplest first version is to keep `ACC` contiguous and only move `FA` and `FB`:

```asm
FA = ACC + FM*FN*8 + FA_PAD
FB = FA + 2*FM*2 + FB_PAD
```

Why this is interesting:

- WMMA source and accumulator bank conflicts can look like "compute latency" or "scheduler exposure".
- A few unused VGPRs may be cheaper than repeatedly colliding with the same register banks.

Pass signal:

- No change in correctness.
- A small but repeatable gain in 8x2 `FED/FEEDONLY`, especially when `FED < FEEDONLY`.

## 3. B-Stationary WMMA Ordering

Try changing the WMMA issue order for 8x2 so one B fragment feeds all M fragments before switching B.

Current ordering appears inherited from the 4x4 winner. For 8x2, B reuse is the central lever, so try:

```text
for kk:
  for ni:
    for mi:
      wmma acc[mi][ni] += A[mi] * B[ni]
```

instead of an order that switches B more often.

Why this is interesting:

- It may reduce operand collector churn for B.
- It better matches the 8x2 reason for winning: maximize reuse of the binding B operand.
- It may hurt accumulator reuse spacing, so it needs a direct A/B.

Pass signal:

- If `FED` rises while `FEEDONLY` is flat, the issue order improved overlap/scheduling.
- If both fall, accumulator latency spacing was more important than B stationarity.

## 4. B Broadcast/Dedup Across More M Waves

The existing B-in-LDS idea halves global B feed when two M waves share B. Push this harder with more `TWM`.

Candidate geometry:

```text
TWM=4, TWN=2, FM=8, FN=2, KWIN=2
```

One `wave_m==0` wave per N group loads B from global, stores it to LDS, and four M waves consume it.

Why this is interesting:

- It attacks the binding term directly: B global feed can be reduced by up to 4x across M waves.
- The cost is extra LDS traffic and synchronization, but prior work suggests A/LDS-side traffic is cheaper than B global transpose feed.
- `KWIN=2` may keep LDS below the occupancy cliff.

Pass signal:

- FEEDONLY rises significantly relative to 8x2.
- FED follows if WMMA overlap/residency holds.

Risk:

- B LDS ring footprint and barriers may eat the savings.
- Needs full-fragment oracle because cross-wave B sharing is easy to mis-address.

## 5. A-Fragment Register Prefetch From LDS

Prior attempts overlapped global A publish and B feed. A different target is the consume-side LDS-to-register dependency:

```text
ds_load A[u+1] into FA_next while WMMA consumes A[u] from FA_cur
```

This is not global-load prefetch. It is A-fragment double buffering after A is already in LDS.

Why this is interesting:

- The 8x2 wall is now partly WMMA/feed overlap. Consume-side `ds_load -> dscnt -> WMMA` may still serialize.
- It may pair well with `ALD2` wide LDS reads.

First experiment:

- Try a lean shape first where extra FA registers fit.
- Then try 8x2 with dyn-VGPR or reduced KWIN.

Pass signal:

- FED approaches FEEDONLY without changing B feed.

## 6. Normal HSA/HIP Dispatch for Static Hand-Asm Kernels

The current best static kernel does not require dynamic VGPR. Therefore raw PM4 is not strictly needed for that specific path.

Try packaging the hand-written assembly as an amdhsa code object and launching it through normal HSA/HIP, while keeping PM4 only for dyn-VGPR experiments.

Why this is interesting:

- It may restore real workgroup IDs and remove the atomic-claim workaround.
- It should make profiling and production integration easier.
- It narrows which costs are PM4 vehicle artifacts.

Pass signal:

- Same correctness and comparable throughput to PM4.
- Reduced or eliminated tile-claim overhead.
- Better profiler visibility.

## 7. Producer-Side Fragment-Major Layout

For training, the most valuable change may be upstream of GEMM. Instead of treating A/B repacking as a GEMM-local prepass, make the producer write activations/gradients directly in fragment-major or lane-major layout.

Why this is interesting:

- Wgrad/dgrad tensors are produced immediately before GEMM. If the producer writes the next consumer's preferred layout, GEMM feed cost falls without adding a copy.
- This enables plain-load B/A fragment feeds and may remove A LDS publication for some paths.

Candidate:

- Add a training-path experiment where the previous op writes K-major tiles in the exact lane-major format consumed by the GEMM.
- Benchmark end-to-end step time, not just GEMM time.

Pass signal:

- GEMM gets simpler feed.
- End-to-end training throughput improves even if standalone GEMM uses more specialized layout.

## 8. Dyn-VGPR Fat M-Reuse Tile

Use dynamic VGPR for what it is best at: fatter live reuse, not ordinary occupancy.

Candidate concept:

- Hold two 8x2 M panels' accumulators live for the same B tile.
- Load B once, apply to twice as much M.
- Amortize `s_alloc_vgpr` over a long K window or a full output tile.

Why this is interesting:

- It attacks B-feed/MAC directly, which is the proven binding side.
- `S_ALLOC_VGPR` has an idle cost, so only a long fat phase is worth testing.

Pass signal:

- FEEDONLY rises above the 8x2 182 TF level.
- FED follows after residency/overlap tuning.

Risk:

- Dyn allocation deadlock or residency collapse.
- Needs strict pool sizing and a single-shot gated protocol.

## 9. Split-K or Two-Stage Accumulation to Reduce Wave Cooperation

Consider intentionally duplicating A work to eliminate some barriers, then reduce partial C in a second pass.

For example:

- 1-wave or 2-wave workgroups compute smaller independent C tiles with B reuse optimized.
- Accumulate partial K chunks into fp32 scratch.
- Final reduction combines partials.

Why this is interesting:

- The real FED path pays for cooperative A-share barriers.
- If barrier/LDS structure is the major tax, duplicating some operand loads may be cheaper than synchronizing waves.
- Training big-K shapes can amortize a second reduction if the primary GEMM gets much closer to peak.

Pass signal:

- Primary GEMM reaches much higher WMMA issue density.
- End-to-end including reduction beats the cooperative wave-group kernel.

## 10. Explicit Wave Specialization

Within a multi-wave workgroup, assign roles:

- loader waves pull future B/A fragments,
- compute waves run dense WMMA,
- roles rotate every K window.

Why this is interesting:

- RDNA4 has a shared VALU issue port, so mixed feed/WMMA waves compete. Role specialization might improve temporal locality of issue: some waves are mostly feed, others mostly WMMA.
- This is conceptually similar to a software pipeline across waves rather than within one wave.

Pass signal:

- Higher FED/FEEDONLY ratio at the same B-feed/MAC.

Risk:

- More barriers and LDS handoff may erase the gain.
- Needs very small prototypes first.

## 11. Wide-K FP8 Load Pairing

AMD's RDNA4 WMMA guide describes a specific low-precision feed workaround: combine two K=16 WMMA operations into one logical K=32 operation so each lane can load 16 FP8/INT8 elements with a 128-bit load, then feed the low and high halves to two WMMA instructions.

This is related to the lane-major B idea, but it should be tested separately:

- `BPLAIN_B128` asks whether B can escape `global_load_tr_b64`.
- `WIDEK_B128` asks whether both A and B can use one 128-bit load for two K slices.

First experiment:

1. Make a single-wave 16x16x32 oracle with a prepacked K-major lane layout.
2. Issue one `global_load_b128` for A and one for B per K=32 fragment.
3. Split each 16-byte register group into low/high 8-byte fragments.
4. Run two `v_wmma_f32_16x16x16_fp8_fp8` instructions into the same accumulator.
5. Compare bit-for-bit against two ordinary K=16 steps.

Pass signal:

- The oracle is correct.
- FEEDONLY rises above the current 8x2 feed-only ceiling.
- FED follows once issue order is tuned.

## Suggested Test Order

1. Wide-K FP8 load pairing oracle.
2. Lane-major B with plain `global_load_b64`, then `global_load_b128`.
3. VGPR/register coloring sweep on 8x2.
4. B-stationary WMMA ordering for 8x2.
5. B broadcast/dedup with `TWM=4`.
6. A-frag register prefetch from LDS.
7. Dyn-VGPR fat M-reuse only after the static mechanisms are exhausted.
