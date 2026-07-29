# DSWS S1 — MORNING BRIEF, 2026-07-26

**Read `DSWS_FINDINGS_2026-07-25.md` first — it has every number and every retraction.**
This brief is only: where we stopped, what to fix first, and what not to redo.

---

# ✅ CLOSED 2026-07-26 — THE POLLSTAGE DECOMPOSITION IS COMPLETE

**The "count discrepancy" that headlined this brief was MY error, not a kernel defect.** I compared
`PS_N` from the POLLSTAGE build against `occ[86]` from a **different build's log**. Within-run the ratio
is **1.015**, exactly as it should be. Stage 6 settles it: its `n = 29,906,543` equals its own run's
`occ[86]` **to the unit**. No code fix was needed. Nothing was wrong.
=> **NEW METHOD RULE 9: NEVER compare a counter across runs.** Probe cost swings the pass count 5x;
`occ[86]` legitimately ranges 30M–154M across these runs at the same kernel and shape.
The timeout was just my 10-minute command ceiling. Latch was clear; stages 3–6 all ran clean.

**Full result, all six stages, cross-validated — see `DSWS_FINDINGS_2026-07-25.md` §7.**

| stage | what | ms/wave | share |
|---|---|---|---|
| **5** | **`da_peek` reservation attempt (ends in a park)** | **0.566** | **34.1%** |
| **6** | **park + `s_sleep`** | **0.491** | **29.6%** |
| 1 | loop head + `deadman_check` | 0.296 | 17.8% |
| 2 | snapshot / FLOWTERM / body-gate | 0.282 | 17.0% |
| 3 | role select + dispatch | 0.020 | 1.2% |
| 4 | feed → `da_peek` gate | 0.004 | 0.3% |
| | **SUM** | **1.659** | |

**IT CLOSES: 1.659 ms/wave vs 1.77 ms measured independently by the GAP probe = 94%.** Two instruments
sharing no code agree to 6%. All seven runs work-exact (`computed=190080`), `oracle bad=0`, exit 0.

**=> 63.7% of every poll pass is the failed reservation peek + the park that follows it.**
The role economy and feed dispatch (stages 3+4) are **1.5%**. Read the two cautions in §7 before
quoting stage 3/4 reach — it measures top-of-region entry, not gate survival.

---

# 1. WHERE THE TIME IS (all in ms, 100k ticks = 1 ms)

```
kernel                       275.0 ms   (33 chunks -> 8.33 ms/chunk)
per wave:
  HEAD  live -> first work      0.588 ms   <- ~1,470 poll iterations   [NOT PINNED, see below]
  GAP   between bursts          0.169 ms   [SOLID]
  TAIL  last work -> exit       1.016 ms   [SOLID]
  ------------------------------------
  wave lifetime                 1.77 ms    of an 8.33 ms chunk
```
TAIL and GAP agree across two builds with very different probe costs => trust them.
HEAD disagrees 6x between builds => that spread IS probe cost; the SHAPE (HEAD > TAIL >> GAP) is
consistent everywhere.

**THE HEADLINE COUNT: `NOBURST` = 22,153–25,035 of 63,360 => ~35% OF WAVES NEVER RUN A SINGLE BURST.**
A pure count, immune to probe distortion, stable across two runs. This is the strongest signal we have.

---

# 2. WHAT IS DEAD — DO NOT RE-INVESTIGATE ANY OF THESE

| candidate | verdict | how |
|---|---|---|
| compute path (WMMA/reduction/C-store/B-fetch) | **~2%** | 4 ablations; deleting the math makes it SLOWER |
| launch / admission | **DEAD** | peak concurrent = 1920 = nominal, entries = 63,360, live_end = 0 |
| coordination as a whole | **~7.6%** | 3-arm SEGK/POOL_N decomposition |
| pipelining (`POOL_N>1`) | **NEUTRAL** | first ever POOL_N=2 run: works, does nothing |
| double-buffering the k-step loop | **REFUTED** | KDBUF built + verified in emitted code: **0.32% SLOWER** |
| boundary election | <1% | BNDTIME |
| `s_sleep` / SLEEPN | ~1% | offline arithmetic |
| reservation CAS | ~0.01% | PASSTIME T2 + counters |
| carrier stalls / VGPR budget / drain | 0 | every run |
| poll-loop throughput | irrelevant | doubled it, runtime moved 1.3% |
| "coordination costs ~600x the work" (KG 07-13) | **STALE, wrong by 2 orders** | see findings §2.5 |

---

# 3. THE OPEN QUESTION

A wave lives 1.77 ms of an 8.33 ms chunk; 35% never get work; occupancy is full; the work itself is free.
**Why does the frontier expose so little work that a third of a fully-resident fleet never receives any?**

**Structural fact already measured:** `ML8_COOP_CHUNK=96` across 64 WGs means only **32 WGs per chunk
have a second tile**, capping next-tile latches at `33 x 32 = 1056 = 3168/3` exactly. **The kernel cannot
invent work the dispatch did not give it.**
⚠ `ML8_COOP_CHUNK` is the documented **compositor-safety** knob (96 tiles/dispatch, 5 ms yield between
chunks). Raising it trades desktop safety — **step it, do not jump, and this is a rule-7 conversation.**

---

# 4. TWO STRUCTURAL WINS BANKED YESTERDAY (both replicated)

**Prefetch rebuilt: +8.6% vs having NO prefetch.** It had been aiming at an already-consumed block of the
CURRENT tile because the two-generation frontier was never built. **Deleting it measured as a 1.3% win
and would have locked in a permanent 8.6% loss.**

**SELFSERVE dead-staging removed:** `POOL_N` is now genuinely inert (POOL_N=1/2/3 byte-identical), LDS
54,784 -> 13,824 at SEGK=256 **without needing `DSWS2_OVERLAP`**, ~4.9 KB of dead `.text` gone,
`feed-stages` structurally 0. `DSWS2_OVERLAP=1` is now byte-identical to SELFSERVE alone — the whole
subsystem existed to reclaim something that should never have been taken.

---

# 5. METHOD RULES EARNED YESTERDAY (the expensive part)

1. **ABLATE, DON'T PROBE.** Every reliable number came from an ablation; all three elaborate in-kernel
   probes (PASSTIME/BURSTCNT/WTBUDGET) produced **zero** defensible numbers. An ablation measures the
   thing; a probe measures whatever it sampled and then you must prove what that was.
2. **"Checkpoint 2 − checkpoint 1. That's ALL YOU NEED."** The two-stamp GAP probe worked FIRST TRY
   after three elaborate ones failed. If a measurement needs more machinery, it is the wrong measurement.
3. **Dividing a wall-clock span by a count of CONCURRENT work gives a RATE, not a DURATION.** This one
   error drove two wrong hypotheses.
4. **Any per-wave latency argument must first answer "why isn't this hidden by the other 14 waves on the
   SIMD?"** Two hypotheses died of exactly this.
5. **Probe cost scales with times EXECUTED.** A global atomic on one address from 1920 waves cost **44%**.
   An unthrottled per-pass counter cost **26%**.
6. **A checker that flags code which passed silicon correctness is suspect before the code is.** My own
   audit script cried violation three times in a row and was wrong every time.
7. **A magnitude match is not a mechanism.** Two different bugs predicted the same 2.03e9.
8. **Pre-register the prediction AND its interpretation** before a decisive run.

---

# 6. TREE STATE

All uncommitted, **nothing staged**, shared tree with a live weight-pager session — **stage nothing**.
Every defsym defaults 0 and is verified byte-identical when off:
`DSWS2_PASSTIME` `DSWS2_BURSTCNT` `DSWS2_WTBUDGET` `DSWS2_GAP` `DSWS2_KDBUF` `DSWS2_POLLSTAGE`
`DSWS2_PREFETCH` `DSWS2_OVERLAP` `DSWS2_ROLEFLOW` `DSWS2_RCONV`.
Baseline on disk: **`815f9894`** (SEGK=256) / `d7221d80` (SEGK=64). Host compiles 0 errors, 23
pre-existing `-Wformat` warnings.
Card **released**. Latch **clear**.

**PRE-FLIGHT, EVERY TIME:** `ls -la occ_dispatch occ_dispatch.cpp` — `gpu_run.sh` guards a stale KERNEL
bin but **NOT** a stale host. That cost a dispatch yesterday, twice.
