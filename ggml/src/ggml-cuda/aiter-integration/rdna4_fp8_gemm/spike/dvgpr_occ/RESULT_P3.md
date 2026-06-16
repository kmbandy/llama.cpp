# RESULT (Phase 3 / MAD-305): dyn-VGPR GEMM occupancy de-risk — **UNRESOLVED** (not a no-go)

**Date:** 2026-06-16. **Hardware:** AMD R9700 / gfx1201 (RDNA4), KFD node 1, wave32, ROCm 7.2.3.
Builds on the Phase-2 occupancy proof (`spike/dvgpr_occ/`) and MAD-300 campaign findings.

## Bottom line (honest, recalibrated)

The question — *does the dyn-VGPR occupancy lever convert to TFLOPS for a real fp8 GEMM?* — is
**not answered**. An initial "NO-GO" call was **over-hasty and partially wrong** (caught in review).
After investigation, the dyn-VGPR thesis is **neither proven nor disproven**. One real constraint
was found (a per-wave VGPR cap); the scarier "synchronized-fat corruption" claim was **refuted**.

## What was built

A host-timed WMMA-throughput harness on the Phase-2 raw-PM4 vehicle (`occ_kernel.s` +
`occ_dispatch.cpp`). Getting trustworthy numbers required fixing two harness bugs and one
timing-methodology problem (below). The dyn kernel is correct at low/normal occupancy
(`--probe`/default cap-probe mode reproduces all of this).

## Findings

1. **The 32× dispatch bug (root of all the "impossible TFLOPS").** `PM4 DISPATCH_DIRECT`'s
   `DIM_X` is in **threads, not workgroups** — passing `nWG` launched only `nWG/32` waves, so we
   over-counted work 32× and every timer read physically-impossible throughput (>1000 TF). Fixed
   by dispatching `nWG*32` threads + a launched-wave counter (`occ[4]`) used for the work count.
   *(Retroactively explains Phase-2's "grid 2048 → maxlive 64".)*

2. **In-kernel GPU-clock timer, validated.** Host `submit→fence` timing does not bracket the
   kernel on raw PM4 (fence fires early); switched to `wall_clock64` (`s_sendmsg_rtn
   MSG_RTN_GET_REALTIME`, lifted vs `llvm-objdump`) with a global min-start/max-end span +
   `live==0`/end-stable completion gating. **Single-wave probe validates it:** span is linear in
   KDEPTH, ~**6.7 ns/WMMA** (≈17 core cycles @ 2.5 GHz) — physical.

3. **Prong 1 (occupancy → throughput, static): flat ~33 TF across occ 7→16** — BUT this is only
   **11% of the 307 TF ceiling**, i.e. the synthetic is **latency/issue-bound, not matrix-bound**
   (a clean WMMA microbench hits 307 at the same NACC=8). A non-matrix-bound kernel **cannot
   validly answer** "does occupancy help a matrix-bound GEMM." **Prong 1 is inconclusive.**

4. **dyn-VGPR per-wave cap ∈ (128, 160].** `s_alloc_vgpr 160` corrupts **0/6 even at 4 waves**
   (no oversubscription); `s_alloc 96` and Phase-2's `s_alloc 128` are fine. So tiles up to
   ~128 VGPR (= **16 fp8 accumulator fragments**, a real tile) work under dyn-VGPR; only the very
   biggest (160+) are blocked. *(Open: is this cap a hard wire or a configurable dyn-VGPR
   register? If configurable, even big tiles are back.)*

5. **The "synchronized-fat corruption law" is REFUTED.** At grid 2560 and 4096 the kernel runs at
   **full occ 16 with `s_alloc 96` and is correct 6/6.** So `occ × footprint ≈ file` is *not* a
   correctness limit. The only failure is a razor-thin edge case: **grid == resident == 2048**
   (one perfectly-synchronized full fill) corrupts **deterministically 0/6**; 1920→OK, 2176→5/6,
   2560/4096→6/6. A real GEMM has grid ≫ occupancy (the OK regime), so it **does not hit this.**

| nWaves | occ/SIMD | OK / 6 |
|---|---|---|
| 1536 | 12 | 6/6 |
| 1920 | 15 | 6/6 |
| **2048** | **16** | **0/6** (exact one-fill edge case) |
| 2176 | 16 | 5/6 |
| 2560 | 16 | 6/6 |
| 4096 | 16 | 6/6 |

## What is NOT proven (the real open question)

A **valid dyn-vs-static throughput comparison** on a **matrix-bound (~150–300 TF), ≤128-VGPR**
kernel at high occupancy. Prong 1's vehicle was non-matrix-bound; Prong 2's heavy kernel was over
the cap. **That number was never legitimately produced.** Until it is, dyn-VGPR is open.

## Candidate levers to 250–300 TF (dyn-VGPR is still on the list)

- **dyn-VGPR, tested properly** — viable for ≤128-VGPR tiles at occ 16; needs a matrix-bound
  synthetic to measure if occupancy actually converts to throughput.
- **Feed-width** — the campaign's #1 *un-exploited* lever: `global_load_tr_b64` to load A/B
  fragments transposed straight into registers, killing the 64× `ds_load_u8` byte-gather that
  walls the real rocWMMA kernel at 90 TF. Independent of dyn-VGPR.
- **ILP / inner-loop scheduling** — the synthetic at 33 TF vs the microbench's 307 at the *same*
  NACC=8 shows large throughput left on the table from unrolling + independent WMMA chains.
- These **stack** (feed-width × occupancy × ILP).

## AMD-facing value (bonus)

Phase-2's occupancy proof + the per-wave cap + the exact-one-fill **corruption-instead-of-stall**
behavior form a clean silicon-characterization / feature-request package. The "oversubscribed
(or exact-fill) `s_alloc` corrupts rather than stalls" behavior is plausibly a **HW/firmware edge
bug worth reporting**.

## Reproduce

```
./build.sh
timeout 90  ./occ_dispatch            # dyn-VGPR correctness vs occupancy investigation (6x/grid) + cap probe
timeout 60  ./occ_dispatch --probe    # single-wave per-WMMA cost (timer validation, ~6.7 ns/WMMA)
timeout 120 ./occ_dispatch --prong1   # occupancy->throughput (static; inconclusive: synthetic 11% of ceiling)
```
