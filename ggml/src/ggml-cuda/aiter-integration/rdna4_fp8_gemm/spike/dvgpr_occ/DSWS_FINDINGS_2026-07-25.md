# DSWS S1 (MAD-305) — FINDINGS, 2026-07-25

Companion to `DSWS_TESTING_LOG.md` (2026-07-25 entries) and `DSWS_BRIEF_2026-07-26_AM.md`.
Everything here is measured unless explicitly marked INFERRED or RETRACTED.

---

## 0. ONE-LINE STATE

The wall is **not** in the compute path (~2%), **not** in coordination (~7.6%), **not** in launch
(peak concurrency = 1920 = nominal). **~35% of waves never run a single burst**, and a wave lives
**1.77 ms inside an 8.33 ms chunk**. The kernel does 0.4 TF against a 307 TF peak. The wall is
**waves waiting for work that is not there**, and the remaining question is *why the frontier
exposes so little*.

---

## 1. THE MEASUREMENT LEDGER (bins + spans, clean profile, probes OFF)

Clean profile (this is the measurement profile of record):
```
WAVES=30 G=6 FM=1 SEGK=256 POOL_N=1 ACC_N=3 JDEPTH=1 KMAJOR=0 DECENTASN=1 BANKZERO=1
STAGGER=1 SELFSERVE=1 SSWIN=32 FORENSICS=0 STAGINSTR=1 TFPROBE=1 DEADMAN=1 CFASSIGN=0 BATONGATE=1
DSWS2_OVERLAP=0 DSWS2_ROLEFLOW=0 DSWS2_RCONV=0 DSWS2_PASSTIME=0 DSWS2_PREFETCH=0
DSWS2_BURSTCNT=0 DSWS2_KDBUF=0 DSWS2_WTBUDGET=0 DSWS2_GAP=0 DSWS2_POLLSTAGE=0
```
Shape ml8 `2112x9216x2560`, super-tile 96x64, n_kseg=10, TOTAL=3168, TOTAL_super=31680,
`ML8_COOP_CHUNK=96` => **33 chunks**, 64 WGs x 30 waves = **1920 resident**, 63,360 wave-instances.

| bin | what | span (ticks) | ms |
|---|---|---|---|
| `815f9894` | **BASELINE** (SEGK=256) | **24,060,216** | **240.6** |
| `d7221d80` | baseline SEGK=64 | — | — |
| `0fe26484` | NOWMMA (all math deleted) | 24,243,168 | 242.4 |
| `25ebf127` | NODSADD (LDS reduction deleted) | 24,153,200 | 241.5 |
| `e35037c1` | NOCFLUSH (C-store deleted) | 24,392,756 | 243.9 |
| `9f9bdf8e` | NOBLOAD (all B fetches deleted) | 23,538,772 | 235.4 |
| `38d7ddd6` | GAP probe | 24,468,892 | 244.7 |
| `53013156` | GAP + HEAD/TAIL/NOBURST | 27,501,600 | 275.0 |
| `1d7efe14` | + peak concurrency | 39,726,372 | 397.3 |
| `c5138582` | KDBUF double-buffer | 24,137,328 | 241.4 |
| `f036dd15`..`10bba694` | POLLSTAGE 1..6 | (1,2 only) | — |

**Noise floor ~1.2%** (established by replicating P2 twice at 0.91% and P3 twice at 1.19%).
1 tick = 10 ns @ 100 MHz. **100,000 ticks = 1 ms.**

---

## 2. WHAT IS SETTLED (measured, not inferred)

### 2.1 The compute path is ~2% of runtime — ABLATION, four arms
Delete the math: **slower**. Delete the reduction: **slower**. Delete the C-store: **slower**.
Delete every B fetch: **-2.2%**, barely outside noise. All arms `bad=76032` (the oracle failing IS the
proof the ablation bit) and `computed=190080` (work-exact held). All four bins verified DISTINCT before
running, so no arm was inert.
**=> WMMA, LDS reduction and C-store are free. Only the B fetch registers at all, and only just.**

### 2.2 Launch/admission is NOT the problem
`PEAK concurrent = 1920` (= nominal) · `ENTRIES = 63,360` (= 33 x 1920) · `live_end = 0`.
Every wave launches, they all coexist, they all retire cleanly.

### 2.3 ~35% of waves never do any work
`NOBURST = 22,153` and `25,035` across two runs with very different probe costs (of 63,360).
**A pure count — immune to probe distortion.** This is the strongest single signal of the day.

### 2.4 Wave lifetime, in ms (cheaper build `53013156`)
```
kernel                  275.0 ms   (33 chunks -> 8.33 ms/chunk)
per wave:
  HEAD  live -> first work   0.588 ms   (~1,470 poll iterations)
  GAP   between bursts       0.169 ms   (2.4 gaps x 0.071)
  TAIL  last work -> exit    1.016 ms
  -------------------------------------
  lifetime                   1.77 ms    of an 8.33 ms chunk
```
Heavier build `1d7efe14` (44% probe tax): HEAD 3.746 / GAP 0.201 / TAIL 0.950, lifetime 4.90 ms.
**TAIL and GAP agree across builds (1.02 vs 0.95; 0.17 vs 0.20) => SOLID.**
**HEAD disagrees 6x => NOT PINNED.** All builds agree on the SHAPE: HEAD > TAIL >> GAP.

### 2.5 Coordination is ~7.6% of the system rate (three-arm SEGK/POOL_N experiment)
```
A POOL_N=1 SEGK=256  190,080 items  24,535,292
C POOL_N=1 SEGK=64   760,320 items  30,111,052
B POOL_N=2 SEGK=64   760,320 items  30,335,700
```
4x the work items cost only 23% more time. Solving A vs C as work + fixed-coordination gives
~29.8 ticks work per SEGK=64 unit + ~9.8 ticks fixed => coordination ~7.6% at SEGK=256.
**POOL_N=2 (first ever execution) is NEUTRAL.**

### 2.6 Eliminated earlier, all by direct measurement
boundary election <1% · `s_sleep` ~1% · reservation CAS ~0.01% · carrier stalls 0 ·
grow-fail 0 even at 2048 waves · drain ~0 · poll-loop throughput (doubled it, runtime moved 1.3%).

---

## 3. STRUCTURAL FIXES THAT LANDED (real, replicated)

### 3.1 The prefetch was BUILT WRONG — rebuilt, +8.6% vs having none
It aimed at an **already-consumed** 256B block of the CURRENT tile, four times over, because the
two-generation frontier holding a next-tile identity was **"still NOT built"** (:451/:464).
At n_kseg=10 (non-power-of-two) the `s_min_u32` clamp collapsed all four guesses onto slice 9.
Rebuilt in four phases (t_next frontier -> dedicated lean prefetch wave -> counters -> coverage):
```
41,102,028  broken prefetch
40,575,032  NO prefetch (ablation)
36,904,204 / 37,242,000  correct prefetch  <- 8.6% FASTER THAN NO PREFETCH
```
**Deleting it measured as a 1.3% win and would have locked in a permanent 8.6% loss.**
Second time in this fleet (after the weight-pager's `LOOKAHEAD_K=1` sweep) that a technique was nearly
retired on the strength of a broken implementation.

### 3.2 SELFSERVE left dead machinery live — POOL_N is now inert
kmbandy called this repeatedly before it was found. Under SELFSERVE waves self-serve from GLOBAL and
never read the staged LDS pool, but `ACC_BASE = OP_BASE + POOL_N*OPSTRIDE` still ALLOCATED it and
`ASTAGE_R`/`BSTAGE_R` still EXECUTED (65,918 stage events at SEGK=64).
| | before | after |
|---|---|---|
| POOL_N=1/2/3 under SELFSERVE | different bins | **BYTE-IDENTICAL** |
| LDS @ SEGK=256 | 54,784 | **13,824** (no OVERLAP needed) |
| `.text` @ SEGK=64 | 28,428 | **23,564** (~4.9KB dead code) |
| `feed-stages` | 1,454 / 65,918 | **0, structurally** |
**`DSWS2_OVERLAP=1` is now byte-identical to SELFSERVE alone** — the whole subsystem existed to reclaim
something that should never have been taken, and its POOL_N=1-only restriction is what refused pipelining.
A second defect of the same class: the dead coordinator still assembled `s_cmp_ge_u32 s46, POOL_N`.

---

## 4. HYPOTHESES I GOT WRONG (all refuted by measurement)

| # | hypothesis | how it died |
|---|---|---|
| 1 | LDS-CAS contention on shared cursor words | reservation CAS = ~0.01% of pass time |
| 2 | `SLEEPN` sleep latency | ~1% of wave-time, killed offline with arithmetic |
| 3 | "the compute burst is ~92% of runtime" | rate-vs-duration error (see 5.1); ablations say ~2% |
| 4 | "16 exposed L2 latencies / not double-buffered" | built KDBUF, pipeline verified in emitted code, **0.32% SLOWER** |
| 5 | "coordination costs ~600x the work" (KG, 07-13) | ~7.6%; two orders of magnitude off |
| 6 | "POOL_N=1 is why" | POOL_N=2 neutral — blocked-for-nothing |
| 7 | "84% is per-dispatch overhead" | span EXCLUDES host gaps; didn't follow |
| 8 | "T is an assignment overwritten per chunk" | actually u32 overflow of the global sum (same magnitude, different mechanism) |

**#4 is the important one:** the build was CORRECT (`s_wait_loadcnt` census 35x`0x0` -> 20x`0x0` +
15x`0x5`, `s_alloc_vgpr` 80 -> 96, k+1 loads genuinely issuing before the wait). The HYPOTHESIS was
wrong: at **15 waves/SIMD the latency was already hidden by multithreading**.

---

## 5. METHOD — THE PART THAT MATTERS MOST

### 5.1 THE ARITHMETIC ERROR THAT DROVE TWO WRONG HYPOTHESES
I computed `span / items = 129` and called it **"ticks per item"**, treating a **system-wide completion
rate** as the **duration of one work item**. With 1920 concurrent waves the real figure is
`span x waves / items` = ~243,000 wave-ticks. **Three orders of magnitude out.**
**DIVIDING A WALL-CLOCK SPAN BY A COUNT OF CONCURRENT WORK GIVES A RATE, NOT A DURATION.**

### 5.2 ANY PER-WAVE LATENCY ARGUMENT MUST FIRST ANSWER: "WHY ISN'T THIS HIDDEN BY THE OTHER 14 WAVES?"
Hypotheses 1 and 4 both died of this. I had even written *"plain LDS latency should be hidden by 15-way
SIMD multithreading"* in my own notes hours before failing to apply it to the k-step loads.
**A latency that multithreading can absorb is not a wall.**

### 5.3 *** ABLATIONS WORK. PROBES DID NOT. ***
- **Every reliable number today came from an ablation** — prefetch rebuild, POOL_N, KDBUF, SELFSERVE
  fix, the four-arm compute sweep. All clean on the FIRST run.
- **Every unreliable one came from an in-kernel probe** — PASSTIME (poisoned by a shared `s71` trigger
  with the prefetch), BURSTCNT (26% schedule tax), WTBUDGET (u32 overflow, negative residual).
  Between the three: **zero numbers worth defending.**
**An ablation measures the thing itself. A probe measures whatever it happens to sample — and then you
must prove what that was.** Every layer added to a probe (T0 calibration, armed flags, u64 sums,
external budget assertions, s71 co-tenant masks, five buckets, sampling, derivation, residuals) existed
to fix a problem the previous layer created.
**kmbandy: "keep it simple stupid" / "literally just checkpoint 2 - checkpoint 1, that's ALL YOU NEED."**
He was right, and the two-stamp GAP probe worked FIRST TRY after three elaborate ones failed.

### 5.4 PROBE COST IS REAL AND SCALES WITH TIMES EXECUTED, NOT CALL SITES
| probe | cost |
|---|---|
| unthrottled idle counter (BURSTCNT) | **26%** |
| `atomic_max` on one address from 1920 waves | **44%** |
| PASSTIME (throttled) | 22–58% |
| GAP (2 RTC reads/burst, ~99/wave) | ~2% |
**A global atomic on ONE address hit by 1920 waves is the single most expensive thing you can add.**

### 5.5 MY OWN CHECKERS WERE WRONG THREE TIMES IN ONE AUDIT
On the BURSTCNT ACC-live audit I reported a violation three times before getting it right: first
flagging the kernel's own functional operand loads, then the retire emit (my script had no notion of
LOCATION), then the `v_wmma` instructions themselves. The build was correct throughout.
**RULE: a checker that reports a violation in code that has passed silicon correctness is suspect
BEFORE the code is.**

### 5.6 A MAGNITUDE MATCH IS NOT A MECHANISM
I "confirmed" that `T` was an assignment overwritten per chunk because `1920 x span/33 = 2.016e9`
matched the measured 2.03e9 to 0.6%. The real cause was a u32 overflow of the global sum
(`6.65e10 mod 2^32 = 2.1e9`) — **both predict the same magnitude.**

### 5.7 PRE-REGISTER THE PREDICTION *AND ITS INTERPRETATION*
Before KDBUF I recorded: *"multi-x if the diagnosis is right; a few percent means the diagnosis is
INCOMPLETE and I want to know that rather than bank a small win."* It returned 0.32% slower and read
immediately as a **refutation** instead of a null to explain away.

---

## 6. THE OPEN QUESTION, STATED PRECISELY

A wave lives **1.77 ms** inside an **8.33 ms** chunk. **35% never get work at all.** Peak concurrency
is full (1920). The compute path is free. Coordination is 7.6%.
**So: why does the frontier expose so little work that a third of a fully-resident wave fleet never
receives any?**
Structural fact already on record (measured, `tiles_latched`): `ML8_COOP_CHUNK=96` across 64 WGs means
only **32 WGs per chunk have a second tile**, capping next-tile prefetch latches at `33 x 32 = 1056 =
3168/3` exactly. **The kernel cannot invent work that the dispatch did not give it.**
`ML8_COOP_CHUNK` is the documented compositor-safety knob (96 tiles/dispatch, 5 ms yield between), so
raising it trades desktop safety and is NOT a free lever.

---

## 7. THE POLLSTAGE DECOMPOSITION — ✅ COMPLETE 2026-07-26, AND IT CLOSES

> ### ❌ RETRACTED: "the counts look wrong" was MY error, not the kernel's.
> The 2026-07-25 draft of this section refused to quote stage 1 because
> `n = 30.7M` implied 485 passes/wave while "`occ[86]` parks = 169.7M" implied 2,679.
> **I compared `PS_N` from the POLLSTAGE build against `occ[86]` from a DIFFERENT build's log.**
> The stage-1 run's OWN `occ[86]` is **30,289,464**, not 169.7M. Within-run:
> `n/occ[86]` = **1.015** (stage 1) and **1.014** (stage 2) — exactly right, since loop heads are a
> superset of parks. The stamp fires on every iteration, as the source says it does.
> **Confirmed decisively by stage 6, whose `n = 29,906,543` equals its run's `occ[86]` to the unit** —
> the park counter and the stage-6 counter are the same event, and they agree perfectly.
> ⇒ **METHOD RULE 9: NEVER compare a counter across runs. Probe cost changes the pass count by up to 5x.**
> `occ[86]` legitimately ranges 30M–154M across these seven runs *for the same kernel and shape*.
> The timeout was also just my 10-minute command ceiling; the latch was clear and stages 3–6 all ran clean.

Six single-stage builds (`DSWS2_POLLSTAGE=1..6`), 2 RTC reads/pass, SGPR accumulate, emit at retire.
The six stages are **contiguous and non-overlapping** (`:4704`→`:7118`), so this is a true partition
of one poll pass. All seven runs: `computed=190080` work-exact, `oracle ok=76032 bad=0`, exit 0, no reset.

| stage | bin | what it brackets | mean | n | span |
|---|---|---|---|---|---|
| 1 | `f036dd15` | loop head + `deadman_check` | 0.000618 ms | 30,736,951 | 23,226,408 |
| 2 | `44288b5c` | snapshot / FLOWTERM / body-gate | 0.000590 ms | 31,516,673 | 23,431,028 |
| 3 | `04efe002` | role select + dispatch | 0.000487 ms | 4,567,862 | 22,966,308 |
| 3′ | `04efe002` | (replicate) | 0.000477 ms | 4,873,280 | 22,757,568 |
| 4 | `6c6a7888` | feed → `da_peek` gate | 0.000097 ms | 14,713,995 | 24,580,112 |
| 5 | `277e17e7` | `da_peek` reservation attempt | 0.001206 ms | 32,544,886 | 23,423,040 |
| 6 | `10bba694` | park + `s_sleep` | 0.001040 ms | 29,906,543 | 23,108,972 |

**Normalization (run-invariant).** Pass counts differ up to 5x between runs, so absolute `total` is NOT
comparable across stages. Use `ns per loop-head pass = mean x reach`, where
`reach = n / (occ[86] x 1.01477)` and the constant is `n1/occ[86]1` from the stage-1 run
(the one run where `n` IS the loop-head count).

### ⇒ MS PER STAGE, PER WAVE (479 passes/wave, from the stage-6 run where n == occ[86] exactly)

| stage | what | ms/wave | share |
|---|---|---|---|
| **5** | **`da_peek` reservation attempt (ends in a park)** | **0.566** | **34.1%** |
| **6** | **park + `s_sleep`** | **0.491** | **29.6%** |
| 1 | loop head + `deadman_check` | 0.296 | 17.8% |
| 2 | snapshot / FLOWTERM / body-gate | 0.282 | 17.0% |
| 3 | role select + dispatch | 0.020 | 1.2% |
| 4 | feed → `da_peek` gate | 0.004 | 0.3% |
| | **SUM** | **1.659** | 100% |

**IT CLOSES.** 1.659 ms/wave vs the **1.77 ms** wave lifetime measured independently by the GAP probe
(two-stamp HEAD/TAIL, a completely different instrument) = **94% closure**. Two instruments that share
no code agree to 6%. This is the first decomposition in the project that reconciles against an
independent measurement instead of against an assumption.

**THE ANSWER TO "WHERE IS OUR TIME":** **63.7% of every poll pass is stage 5 + stage 6 — the reservation
peek that fails, and the park that follows it.** Another 34.8% is stages 1+2, which is loop-head
watchdog and gate re-evaluation. Stages 3+4 — the actual role economy and feed dispatch — are **1.5%**.

### Two cautions on reading this table

1. **Stage 3/4 `reach` (~9%) does NOT mean "only 9% of passes get past the body gate."** Stage 5 sits
   downstream of stage 4 yet reaches **98%**. That is only possible because `pollstage_leave` is
   ARMED-gated: ~89% of passes **branch into the feed region below `pollstage_enter 4` (`:5710`)**,
   flow through the region, and are never counted by 3 or 4. Stages 3/4 measure *top-of-region entry*,
   not gate survival. The per-pass ns figures are still correct (reach and mean are consistent).
2. **These are WAVE-TIME durations on a 15-way-multithreaded SIMD, not instruction latencies.** A wave's
   wall-clock inside a stage includes time it sat descheduled while other waves ran (method rule 4).
   So "stage 6 = 0.491 ms" is *not* a claim that `s_sleep` executes for 0.491 ms — `SLEEPN=2` is ~128
   clocks. It is a claim about where wave-time is *attributed*, which is what we asked for.
