# DSWS2_FUNNEL bring-up — PRE-REGISTRATION (written BEFORE the dispatch, 2026-07-27, queued #2)

Bin `4f567be6…` · `.text` 31,144 B · LDS 17,920 B
Build: `RESVPROBE=1 BNDSPLIT=1 DSWS2_FUNNEL=1 DSWS2_FUNNEL_SPIN_N=1 FM=2 G=4 ACC_N=2 ./build_flow.sh`
Shape: `ml8_dense_ffn_down` M2048 N2560 K9216 — identical to L0/L0b, so this is a clean A/B.

## The single changed variable
`DSWS2_FUNNEL=1` (+128 B over the L0b bin `61ffe8b2…`/31,016 B). Everything else byte-for-byte the
same build profile. `SPIN_N=1` is documented-inert (SCC polarity, see testing log §5) — this tests a
**check-once read-only pre-gate**, NOT a 1-iteration spin.

## Hard gates — any failure is a FULL STOP
- `oracle bad = 0`
- `computed == G*TOTAL_super*reps` (92,160 × reps), WORK-EXACT
- `occ[96]` delta **+0** against the host's own printed expectation (never from memory)
- bin publishes LDS 17,920 B and host reconstruction AGREES
- `occ[0] = 0`, canary clean

## THE PREDICTION

**PRIMARY — the funnel thins the herd:**
`ZLOCK_LOST` falls **well below 76.5%** and `ADVANCE` rises **above 1.6%**.

Mechanism: 76.5% of boundary entries lose an election they were always going to lose, and a further
21.8% take the lock only to be refused by the GSTORED C-store gate. The funnel pre-checks that same
gate read-only, before the CAS, so both populations should stop contending.

**SECONDARY — the frontier should be less starved:**
`UNACCOUNTED` (empty-frontier) falls below **96.1%** of bails, and iterations-per-reserve falls below
**129.9**.

### Baseline to beat — L0b, same shape, same geometry
```
feed-path bails occ[86] = 29,928,830   (129.9 per successful reserve)
  CAS-loss    occ[87] =    488,061   1.6%
  window-full occ[89] =    164,203   0.5%
  boundary    occ[97] =    513,443   1.7%
  UNACCOUNTED         = 28,763,123  96.1%
BNDSPLIT (33,653 sampled entries):
  ZLOCK_LOST 76.5% | DRAINGATE_BAIL 0.0% | CSTOREGATE_BAIL 21.8% | ADVANCE 1.6%
grow-fail occ[73] = 6,574,885 ; coast-frac 97.2% ; TF 3.3 (probe build, not quotable)
```

### A TF REGRESSION IS AN ACCEPTABLE OUTCOME, NOT A FAILURE
The funnel adds ~**8.6M extra serialized LDS loads** (4 words per boundary entry × ~2.15M entries;
`lds_get` = `ds_load_b32` + `s_wait_dscnt 0`). This kernel has a documented **16x** regression from
ONE extra LDS read in the peek path (kernel :5950-5954). So the honest hypothesis is:
**the funnel trades LDS read traffic against avoided CAS contention, and the net is unknown.**
Report both halves. Do not quote TF from a probe build in either direction.

### What would falsify the mechanism
- `ZLOCK_LOST` unchanged → the herd is not gated by the conditions the funnel checks; the pre-gate
  reads stale values and waves pile onto the CAS anyway.
- `ADVANCE` unchanged while `ZLOCK_LOST` drops → contention was never what limited advancement; the
  C-store gate is a hard latency floor and thinning the herd cannot help. **This would redirect the
  whole effort at `GSTORED` / C-store retirement, not at contention.**
- `UNACCOUNTED` unchanged → frontier starvation has a source upstream of the boundary entirely.

## Known-bad output to IGNORE on this build
- `occ[88]=0` "carrier stall" — STRUCTURAL at JDEPTH=1 (`.Lflow_jwait` does not exist, kernel :2974).
- `occ[98]=0` BATON — NOT WIRED, no writer exists.
- COAST DECOMP door2/door3 — now correctly labelled RESVPROBE aliases and excluded from the sum;
  they are window-full and CAS-loss, not lead-gate and fat-peak-full.
- Do not read the door-sum mismatch as a defect: occ[73] merges 3 grow-fail sites, only :7176 coasts.

## Discipline
Changed bin → **rule 2: ONE dispatch, then STOP and report.** Do not chain the 30-shape sweep onto
this. `DEADMAN=1`, ticks unchanged at 0.5 s. Dispatch only via `gpu_run.sh`.
