# DSWS2_SETTLE test — PRE-REGISTRATION (written BEFORE the dispatch, 2026-07-27, queued #1)

Bin: **UNCHANGED** `4f567be6…` / 31,144 B / LDS 17,920 (the funnel-on bin from the 12:30 run).
**`DSWS2_SETTLE=0.02` is the ONLY changed variable** (default 0.30). No rebuild — same binary,
same shape (`ml8_dense_ffn_down` M2048 N2560 K9216), same geometry.

## The claim under test

`occ_dispatch.cpp:2184` — `settle` defaults to **0.30 s**. The completion gate at `:2207`:

```c
if (admitted && occW[0]==0 && (ff || (end != 0 && (now - lastEndChange) > settle))) done = true;
```

The EOP fence (`ff`) is armed **only on the final chunk** (`:2173-2174`), deliberately — a stalled
per-chunk fence used to block the next chunk from launching. So every **non-final** chunk must fall
through to the `settle` branch and burn a flat 0.30 s after the last wave stamps its exit.

**Therefore ~91% of chunk wall is a host-side constant, not GPU time.**

## ★ THE DECISIVE TEST IS *WITHIN* THIS SINGLE RUN ★

This is the important design property, and it is why this test is sound where the funnel A/B was not.
Each rep runs TWO chunks with DIFFERENT completion paths:

| chunk | tiles | last? | completion path | 12:30 measured | PREDICTED @ settle=0.02 |
|---|---:|---|---|---:|---|
| base=0 | 512 | no | **settle** | 0.317 s | **~0.03-0.05 s** |
| base=512 | 128 | yes | **EOP fence** | 0.013 s | **~0.013 s (UNCHANGED)** |

Chunk 2 is the control: it never pays settle, so it MUST NOT move. If both chunks move, or neither
does, the model is wrong. **No cross-run comparison is required for the mechanism claim** — which
matters because mlambaformer has held the card in between, and cross-session absolute numbers on a
shared box are not comparable (mlambaformer measurement-discipline lesson, 2026-07-24).

## Hard gates — any failure is a FULL STOP
- **`oracle bad = 0`** ← the real gate. Failure mode is FAIL-LOUD by design (`:2179`: a stale C read
  "fails the oracle, never a false CLEAN"), so a too-short settle CANNOT silently corrupt a result.
- `computed == 92,160 × reps`, WORK-EXACT
- `occ[96]` delta +0 against the host's printed expectation
- `occ[0] = 0`, canary clean

## Secondary predictions
- Total chunk wall: **1.650 s → ~0.25 s**
- **Span-TF must NOT change (~3.2).** `settle` is host-side idle time after the waves have exited;
  it cannot touch the GPU busy span. **If span-TF moves materially, my model is WRONG** and settle is
  somehow affecting execution rather than just the host's completion detection.
  (Caveat: this one IS a cross-run comparison and carries the drift risk above. The within-run
  chunk1-vs-chunk2 test does not.)
- End-to-end TF: 0.29 → ~1.9 (4.832e11 FLOP / total wall). Harness cost only — **this does not
  improve the kernel**, which remains ~3.2 TF span-measured against hipBLASLt.
**[CORRECTED 2026-07-27 EVE: the hipBLASLt band on real DENSE shapes is 123-189 TF, not 12.6-70.6. That band was the ml8 MoE M=512 subset only. Mean ratio ~80x. See RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md and the CORRECTION box in DSWS_BRIEF_2026-07-27_AM.md.]**

## Prior art
Swept 2026-07-21 at **FM=1 G=6**: settle 0.05 / 0.02 / 0.01 gave mean chunk wall 0.050 / 0.025 /
0.016 s with `oracle bad=0` at every value. The default was never lowered because **span-TF did not
move (2.0 / 1.9 / 2.0)** and the sweep therefore read as a null result — our headline metric is
structurally blind to this cost. This run re-validates at **FM=2**, which has never been tested.

## Discipline
Same bin, so this is NOT a new-kernel bring-up — but still **ONE dispatch, then STOP** (rule 1).
`DEADMAN=1`, ticks unchanged. Dispatch only via `gpu_run.sh`. Do not chain the sweep onto this.
