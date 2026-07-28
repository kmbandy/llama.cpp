# L0 RESVPROBE — PRE-REGISTRATION (written BEFORE the dispatch, 2026-07-27 ~08:45)

Bin: `2ca16ea0d9cace67…`  `.text`=30,824 B  LDS=17,920 B  (RESVPROBE=1 is the ONLY delta vs
`a581c7b8…`/30,812 B; +12 B = 3 added instructions)
Shape: `ml8_dense_ffn_down` M=2048 N=2560 K=9216, super-tile 128x64, MTLsuper=16 NTL=40 n_kseg=36

## Derived from the HOST's own formulas (not from memory)

| quantity | formula | value |
|---|---|---|
| TOTAL (tiles) | MTLsuper * NTL | 640 |
| TOTAL_super | TOTAL * n_kseg | 23,040 |
| GROUPS | G/ACC_N = 4/2 | 2 |
| `computed` | G * TOTAL_super * reps | **92,160 x reps** (=460,800 @ reps=5) |
| `occ[96]` | GROUPS * TOTAL_super * reps | **46,080 x reps** (=230,400 @ reps=5) |

`reps` is chosen at runtime from `DSWS2_TARGET_SECS=1.5`; last night this shape gave reps=5.
**DO NOT hardcode reps.** Read the host's printed expectation and check delta == 0.
(07-26 trap: I pre-registered 92,160 from memory when truth was 184,320 — exactly half, so a 50%
work loss would have read as SUCCESS.)

## Hard gates (any failure = STOP, do not continue the campaign)
- `oracle bad = 0`
- WORK-EXACT: `computed == G*TOTAL_super*reps`, delta 0
- `occ[96]` delta +0 against the host's own printed expectation
- LDS: bin publishes 17,920 B and host reconstruction AGREES

## THE PREDICTION (the actual point of the run)

**PRIMARY: `WINDOW FULL -> STAGE-BOUND`, i.e. pfFrac > 50%.**

Reasoning: only `ACC_N=2` waves per group may compute concurrently (`SL_RBNEXT` hands out
`0..ACC_N-1`), coast measured 96.9%, and 2,048 resident waves contend for an `SSWIN=32` window.
A wave tests the window (`r-DRAIN >= SSWIN`) and bails BEFORE attempting the `ASSIGN_HEAD` CAS,
so window-full should absorb most bails and starve the CAS-loss counter.

**Secondary: `contention < 1.0`** (occ[87] well below occ[96]).

Baseline to beat, same shape, non-probe bin, last night:
`occ[86]`=22,821,888 empty-frontier bails · `occ[96]`=230,400 · grow-fail `occ[73]`=3,412,194
i.e. ~99 bails per successful reserve.

### What would falsify me
- `contention > 1.0` -> CURSOR-CONTENDED -> shard `ASSIGN_HEAD`; my "waves bail before the CAS"
  reasoning is wrong.
- `NEITHER` -> the 22.8M empties are ZLOCK boundary serialization; both my levers (G and sharding)
  are aimed at the wrong thing.

### Instrument sanity check to perform on the output
`occ[87]` and `occ[89]` should be plausible SUBSETS of the `occ[86]` bail population. If
`occ[87]+occ[89] > occ[86]` the accounting is broken and the split is not trustworthy.

## Known-bad output on this build — IGNORE, DO NOT QUOTE
1. `"carriers are fed (stall is not the wall)"` — RETRACTED 07-26. `.Lflow_jwait` does not exist at
   JDEPTH=1 (kernel :2974); occ[88]=0 is STRUCTURAL. The line is still in occ_dispatch.cpp
   (2 occurrences) — my brief claimed I removed it; **that claim was false.**
2. `"=> ASSIGN-BOUND"` — RETRACTED 07-26. Still in the source (2 occurrences).
3. `oracle ... LDS=<n>B` on the oracle line — computed with a DEFAULT G=6, not the actual G.
   Last night printed 65,792 B while running G=4 (65,792 is the G=6 operand-inclusive total).
   Display-only (occ_dispatch.cpp:7308/7320); actual alloc trusts the `.lds` sidecar.
4. `RESVPROBE` host comment says `occ[96] == TOTAL_super` (:2655) — stale label; occ[96] is
   `GROUPS*TOTAL_super*reps`. The contention/pfFrac RATIOS are unaffected (run-total / run-total).

## Do NOT quote TF from this run
RESVPROBE is a probe build. Probes lie (PHASEPROBE 44x, FORENSICS 62%). This build is cheap
(+12 B) so distortion should be small, but the standing rule is: no TF verdict from a probe build.
