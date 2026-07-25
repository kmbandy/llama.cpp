# ⛔ DSWS CONTINUATION BRIEF — 2026-07-21 END OF DAY ⛔
## READ §0 AND §1 BEFORE RUNNING OR CITING ANYTHING.
### This supersedes `DSWS_BRIEF_2026-07-21.md` (the midday one). Authoritative results file:
### **`RESULTS_DSWS_BASELINE_2026-07-21.md`** — not the older `RESULTS_DSWS_vs_hipBLASLt_*.md`.

---

# §0 — FOUR RETRACTIONS TODAY, ONE ROOT CAUSE

**I published four values that had never been checked against their source. kmbandy spent hours
unable to trust any result. That cost is the headline of this day, not the throughput numbers.**

| # | what I published | truth | how it was caught |
|---|---|---|---|
| 1 | 4 wins over hipBLASLt; flatness CV 0.700 | **0 wins. All 46 sweep rows misparsed.** The column was the `spread N%` jitter field, never throughput. | kmbandy asked why lm_head's log said `TF=0.6` when the table said 0.20 |
| 2 | "bin sha `397bfbe1cb010c6e`" in 3 documents | Matches **no hash of any artifact**. I never ran it. | tried to verify before a dispatch |
| 3 | `TF=0.0` reported as a measured zero | `printf("%.1f")` floor. True value **0.035**. | kmbandy: "does that seem logical?" |
| 4 | "G=4 is 1.7x–12.6x FASTER than G=6" | A **chunking artifact** — morning G=4 ran 1 unbounded chunk, afternoon G=6 ran 512-tile chunks | kmbandy: "was it 1 for 1? no 'well…'" |

**THE RULE, now enforced in code rather than discipline:** derive values from full-precision inputs,
cross-check against the renderer, and **refuse rather than report** on mismatch. See §5.

**A SECOND, SUBTLER FAILURE MODE, three times today:** stating a *cause* before isolating it.
I said the compositor cap explained 7.2→1.2 while `SSWIN` was still uncontrolled; I said the
300ms settle was the cap penalty when it is host wall *outside* the TF span; I said TFPROBE was the
majority of the fixed cost when it is ~⅓. **Say "not isolated" out loud when two variables moved.**

---

# §1 — ⭐ THE ESTABLISHED BASELINE ⭐

**Config A1:** `G=6 ACC_N=3 CFASSIGN=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 VBUDGET=1536
JDEPTH=1 STAGGER=1 DECENTASN=1 SELFSERVE=1 BANKZERO=1 RBU=1 INITBAR=1 TERMFIX=1 BATCH=1`
→ bin `cac3ff7c2338e73f`. **Identify builds by COMMIT + DEFSYMS. A remembered sha is not a sha.**

| condition | mean | median | max | best shape |
|---|---:|---:|---:|---|
| 512-tile cap (old default) | 1.028 | 0.725 | 4.543 | ffn_down M2048 |
| **1 chunk (`ML8_COOP_CHUNK=32768`)** | **2.012** | 1.133 | **9.146** | lm_head |
| 1 chunk + `FLOW_WAVES=5` | — | — | **14.2** | ffn_gate_up M2048 (single shape) |

hipBLASLt (mean-of-30, methodology-matched): **66.89 mean**. **WINS: 0 / 25. Best ratio 0.057x.**

`ml8_dense_ffn_gate_up M2048` across today, same shape, all WORK-EXACT + oracle clean:
**1.51 (fabricated) → 1.5 (512-cap) → 7.96 (1 chunk) → 14.2 (1 chunk + W=5)**.

---

# §2 — ⭐ THE REAL FINDING: ~3.0 µs PER LAUNCHED WAVE ⭐

Measured on an 8-tile shape (`96x512x2048`) whose actual work is ~12 µs, so the span is ~all overhead.
`span = atomic-MAX(exit tick) − atomic-MIN(entry tick)`, so it covers first-wave-in to last-wave-out.

**TWO ORTHOGONAL SWEEPS, SAME CONSTANT:**
- waves/WG at 64 WGs: 30→5.86ms, 10→1.91ms, 5→1.02ms  (3.05 / 2.98 / 3.19 µs per total wave)
- WG count at 30 waves: 64→5.99ms, 32→2.97ms, 16→1.63ms, 8→0.97ms  (3.12 / 3.10 / 3.40 / 4.05 µs)

**Fit: `cost ≈ 0.25 ms + 3.0 µs × total_waves`.** At the default 64 WGs × 30 waves = 1920 waves =
**~5.9 ms per dispatch**, which is **~52% of GPU time even at ONE CHUNK**. Driver is TOTAL wave count,
not WGs and not waves-per-WG independently.

**THIS EXPLAINS THE MORNING'S UNEXPLAINED 4.3x.** `WAVES 30→5` was worth 4.3x and I burned the
afternoon on five wrong theories (`n_kseg`, `SEGK`, `BATCH`, the `ASSIGN` cursor, `SSWIN`) — all
refuted. Wave count *is* the cost, linearly, and always was.

### ~⅓ of it is our own instrumentation
`tfspan` (`occ_kernel_dsws_flow.s:2171`), run by EVERY wave at entry (`:2880`) and exit (`:4868`):
```
s_sendmsg_rtn_b64 ... MSG_RTN_GET_REALTIME ; s_wait_kmcnt 0x0   <- blocking message-bus round trip
global_atomic_min/max_u32 ... scope:SCOPE_DEV                    <- device-scope, ONE shared address
```
1920 waves × 2 sites = 3840 message-bus round trips + 3840 contended device-scope atomics.
**`CLAUDE.md` rule 5 says `s_sendmsg_rtn` spam BRICKS and the deadman throttles it 1-in-64.
TFPROBE does it unthrottled, twice per wave.**

MEASURED (8-tile, W=30, 1 chunk, host wall via `ML8_CHUNK_DIAG`): **TFPROBE=1 → 6 ms, TFPROBE=0 → 4 ms.**
So **~2 ms of ~6 ms (~33%) is the instrument; ~4 ms is genuine.** My prediction that it was the
majority is **FALSIFIED**. Fourth instance of this project's instrument-perturbs-the-measurement
pattern (after PHASEPROBE 44x, `flow_gauge`, PHIST 220–294%).

### Leading un-tested suspect for the remaining ~4 ms
`cnt_flush` (`:4857`) emits **up to 15 device-scope atomics PER WAVE at retire** (COAST, COMP, FEED,
GROWFAIL, BWRITE, BADD, FEEDMT, JWAIT, CLEAD, CNOSTG, DMFAT, TOKLEAK, + SELFSERVE + DECENTASN).
~29,000 contended atomics at 1920 waves. It sits at `:4857`, **11 lines before the exit stamp at
`:4868`, so it is INSIDE the measured span.** The hot path is clean (`cnt_inc` is one SGPR add) —
"single-emit-at-retire" became fifteen.

---

# §3 — ⚠ WAVE COUNT IS A TRICK LEVER. DO NOT SHIP `WAVES=5` AS A CONCLUSION. ⚠

kmbandy caught this and he is right. At `WAVES=5`:
- **`coast-frac` is still 45.2%** — the machine is half-idle even at the "optimum"
- **`door3 FAT-PEAK-FULL = 0` and `door4 GROW-FAIL = 0`** at every wave count → with 5 waves there
  is no VGPR contention, `s_alloc_vgpr` never fails, the stagger never gates. **We are getting
  14.2 TF by switching off the dyn-VGPR moat that DSWS exists to exploit.**
- **`computed` RISES as waves fall** (13.1M → 16.9M): fewer waves do strictly MORE total work in the
  same wall time. Waves actively INTERFERE; they are not merely idling.
- hipBLASLt is 176 TF on this shape. A 12x gap cannot be closed by 5 waves doing more each — that
  needs parallelism, which needs waves.

**=> USE THE WAVE-COUNT CURVE'S SLOPE AS THE INSTRUMENT, NOT AS A KNOB.** It is currently NEGATIVE
(every added wave hurts). **A genuine fix to the per-wave cost should FLIP IT POSITIVE.** That is a
sharper falsifier than any TF number: it distinguishes "removed an overhead" from "found a lever".

**Both of today's big gains are SUBTRACTION** (removed chunks, removed waves). Real, worth banking,
but they are ceilings, not directions — they run out when there is nothing left to remove.

---

# §4 — THE COMPOSITOR CAP: 512 WAS A SCAR, NOT A THRESHOLD

`ML8_COOP_CHUNK=512` was set after a 2.46s chunk killed Hyprland this morning — **but that was a
PHIST probe build (220–294% instrumentation overhead)**. We took a number from an instrumented run
and applied it to every production dispatch.

**MEASURED: worst per-chunk wall across ALL 28 real shapes at ONE chunk = 0.0275 s — 27x under the
0.75 s abort.** Worst case is lm_head at 32,000 tiles. A shape would need ~430,000 tiles to approach
the threshold; the largest real shape is 32,000.

Cost of the cap: **mean 2.23x, median 1.08x, max 17.69x** (lm_head, 63 chunks). It is a FIXED
per-dispatch cost, so many-tile shapes pay it repeatedly and sub-512-tile shapes pay nothing.

**REAL CONSTRAINT, and it is not tile count:** the 0.75s abort is evaluated BETWEEN chunks, so at
`nChunks==1` it can never fire. Per-tile cost also varies with K (K=9216 tiles cost ~12x K=768 tiles),
so **a tile-count budget is the wrong unit — a WALL-TIME budget is the right one**, and we now have
the model to predict it (`6.24 ms + marginal × tiles`).

`DSWS2_SETTLE` (host, default 0.30s, `occ_dispatch.cpp:2082`) is a SEPARATE thing: non-final chunks
wait for store quiescence because a per-chunk EOP fence stalls and blocks the next dispatch
(`:2069`). Swept 0.30→0.01: wall 0.308→0.018 s per chunk, correctness clean throughout, **TF
UNCHANGED** — it is host wall, OUTSIDE the tick span. Worth ~2 s per multi-chunk call in real
inference latency; worth nothing in TF.

---

# §5 — THE NEW HARNESS (this is the thing that must not be lost)

`dsws_realshape_bench.py` + `test_dsws_realshape_bench.py` (7 tests). Built by Codex; I added
`--g/--acc-n/--sswin/--waves/--tag/--chunk` and the `n_kseg>=2` legality screen.

- **DERIVES** TF from full-precision ticks/geometry/reps. **Never scrapes a rendered field.**
  `reps` derived from the WORK-EXACT count, not read off a line.
- **CROSS-CHECKS** the derived value against the kernel's own rendered `TF=` AND its `% of peak`,
  and **REFUSES the row on mismatch** (`SELF_VALIDATION_MISMATCH`).
- **No throughput at all** for a run that is not WORK-EXACT with oracle `bad=0`.
- **Every row names its source log.**
- **GUARD PROVEN NON-VACUOUS:** halve a real log's tick count, leave its rendered TF intact → REJECTED.
  Verified by me on my own fixture, not just theirs.
- Acceptance over the 114 archived logs: **91 pass, 23 refused** (11 invalid-run markers, 8 no
  WORK-EXACT gate, 3 non-real geometry, 1 no oracle).

`bench_hipblaslt_matched.py` — vendor bar, **mean-of-30** (was min-of-N), **shapes IMPORTED from
`dsws_realshape_bench.SHAPES`** (same Python object — the two sides cannot drift).
Cross-check: new min-of-30 vs old min-of-N agrees to **median 1.023**, so the old vendor numbers were
never fabricated, only best-case. **min-of-N overstated the vendor by 17.0%** (its own per-call
spread is median 36%, max 79%). Old `hipblaslt_ml8_baseline.json` → `RETIRED_min-of-N_*`.

**REMAINING UN-EQUALIZABLE ASYMMETRY:** vendor runs unchunked, we run capped. State it with every ratio.

---

# §6 — CFASSIGN (counter-free assign) IS ADOPTED

Built by Codex. Removes the shared `ASSIGN_HEAD` CAS; each wave derives its unit from its wave id in
the current cohort. One atomic per TILE (`occ[20]`, unchanged) instead of one CAS per unit.

**+13.5% G=6 / +13.9% G=4 at the 512-cap; +12.2% / +12.8% at 1 chunk.** Survived two independent
conditions and two geometries. Correct across `n_kseg` ∈ {2,3,6,8,10,16,36}; bring-up used a **dense
stride=1 oracle (864/864 tiles)** because the count gates are structurally blind to DUPLICATED work.
**Byte-identical at `CFASSIGN=0`.**

**ASSEMBLER GUARDS — DO NOT REMOVE:** needs `DECENTASN=1`, `SELFSERVE=1`, `BATCH=1`, and
**`WAVES <= SSWIN`** — a flat wid→unit map with more waves than control slots aliases two waves onto
one `SL_GEN`/`SL_RBDONE` ⇒ **WRONG C WITH EXACT COUNTS**. That is why the baseline runs `SSWIN=32`;
CFASSIGN **cannot build** at `WAVES=30, SSWIN=8`.

---

# §7 — RULED OUT TODAY. DO NOT RE-DIAGNOSE.

- **G=6 vs G=4**: at 1 chunk it is a DEAD HEAT (A1 2.012 vs B1 2.030). The earlier "G=6 beats G=4"
  was itself a cap artifact. **G is not a meaningful lever.**
- **STAGGER**: 6.01 vs 5.78 ms (3.8%). Not the per-wave cost. `door3`=0 always.
- **SSWIN**: 8 vs 32 vs 64 → 1.3 / 1.2 / 1.5 TF on the same shape. Not a factor. It caps
  `ASSIGN − DRAIN`, but SELFSERVE publishes a pre-completed sentinel and calls `drain_advance`
  BEFORE compute, so the window never gates concurrent work.
- **`DSWS2_SETTLE`**: host wall, outside the TF span (§4).
- **STAGINSTR hot path**: `cnt_inc` is one SGPR add, no memory. (The RETIRE flush is still suspect — §2.)
- **`min(WAVES, n_kseg)` as a throughput model**: dead. It is true about the reservable set but does
  not predict TF.
- **`door1 NOTHING-STAGED = 100% of coast`**: NOT a starvation signal. Under SELFSERVE it is the
  VESTIGIAL RING door and reads 100% regardless. I over-read it twice.
- Older, still standing: admission/concurrency, NOWMMA (−0.1%), NOBLOAD (−2.0%), POOL_N, MSSCAN.
  `PLAN_UNPIN_COMPUTE.md` REJECTED on source.

---

# §8 — NEXT WORK, IN ORDER

1. **FIX THE GATES THAT CANNOT EVALUATE.** `TFPROBE=0` and `STAGINSTR=0` both zero `occ[71]`, so the
   WORK-EXACT gate reads `computed=0`, reports **INEXACT, and LATCHES** — a false positive that makes
   the instrumentation untestable. **A gate that cannot evaluate must say so, not claim failure.**
   Same class as the reps-aware bug fixed this morning. This blocks item 2.
2. **ABLATE `cnt_flush`** (~15 device-scope atomics/wave at retire, inside the span) — the leading
   suspect for the remaining ~4 ms. Needs item 1 first. Rely on the dense oracle for correctness.
3. **THROTTLE OR MOVE `tfspan`.** It is ~⅓ of the fixed cost and violates the documented
   `s_sendmsg_rtn` rule. Options: throttle like the deadman (1-in-64), stamp per-WG instead of
   per-wave, or derive the span host-side. **Note it cannot simply be switched off — the WORK-EXACT
   gate depends on it (see item 1).**
4. **RAISE THE CHUNK DEFAULT.** 512 → wall-time-budget sizing (or ≥32768 tiles, which puts every real
   shape in one chunk with 27x measured margin). One-line host change, worth 2.2x mean / 17.7x on
   lm_head. Keep the predicted-wall guard because the abort cannot fire at `nChunks==1`.
5. **RE-SWEEP the full table at the new default** with the fixed harness once 1–4 land.
6. **`WAVES=4` is UNBUILDABLE**: `NCOMPUTE = WAVES−3 = 1` → `BATON_MAGIC = 2^32`, not 32-bit. The
   `.if NCOMPUTE < 1` guard (`:780`) catches 0 but not 1. Fix it (or `STAGGER=0`, measured inert) if
   probing below 5 — the wave curve had not turned over at 5.
7. **FREE WIN, STILL UNCLAIMED SINCE 2026-07-13:** `mlmf_mamba_in_proj` N=4200 is REJECTED by
   `torch._scaled_mm` (N%16) and falls back to bf16 at 67.1 TF; the padded N=4208 variant runs fp8
   at **105.7 TF = 1.58x** for a one-line weight pad.

---

# §9 — STATE OF THE TREE

- **HEAD `0bad7c1af`** (docs) on `652053c69` (code). Both on master, **UNPUSHED**.
- **UNCOMMITTED:** `occ_kernel_dsws_flow.s` (CFASSIGN), `build_flow.sh` (CFASSIGN defsym),
  `dsws_realshape_bench.py` + tests, `bench_hipblaslt_matched.py`,
  `RESULTS_DSWS_BASELINE_2026-07-21.md`, `DSWS_REALSHAPE_HARNESS_REPORT.md`, acceptance artifacts,
  this brief. **`sweep_dsws_realshapes.sh` DELETED** (the parser-bug harness; recoverable from
  `652053c69` only to see what not to do).
- **NEVER stage `docs/examples/router-fleet-main.ini`** — kmbandy's unrelated WIP.
- **Latch CLEAR. No board claim held. Card free.**
- Raw data: `~/dsws_gpu_logs/` — `matrix_2026-07-21/` (512-cap 4-arm), `matrix1chunk_2026-07-21/`
  (1-chunk 4-arm), `cap1_*` (1-chunk replication), `capsweep_c*` (chunk curve), `settle_*`,
  `fixedcost_w*`, `wgcount_p*`, `stagger*`, `tfprobe*`, `w1c_*` (wave curve at 1 chunk),
  `hipblaslt_matched_2026-07-21.{json,txt}`.
- **A Codex analysis of the per-wave cost was commissioned and NEVER DELIVERED A FILE** (~1h, two
  dispatch attempts). My briefing didn't name an output artifact the first time. §2's findings are
  mine, from source + measurement.

---

# §10 — PROCESS, EARNED TODAY

- **NEVER PUBLISH A NUMBER YOU HAVE NOT CHECKED AGAINST ITS SOURCE.** Four failures, one cause. All
  four survived because they were GOOD NEWS. Excitement is the tell to re-derive, not to celebrate.
- **NEVER STATE A CAUSE BEFORE ISOLATING IT.** Three failures. If two variables moved, say so.
- **KEEP THE RAW PER-RUN LOGS.** `~/dsws_gpu_logs/rs_*.log` survived the bad harness and let the
  entire corrected table be rebuilt with ZERO GPU time.
- **AN INSTRUMENT CAN BE THE COST.** Fourth instance on this project. Ablate instrumentation before
  trusting what it reports about overhead — and make sure the ablation is *possible* (item 1).
- **SUBTRACTION IS NOT A DIRECTION.** Removing chunks and waves gave 2x and 1.7x. Both real, both
  ceilings. The ceiling only moves when the per-wave cost goes.
- **BRIEF SUBAGENTS WITH AN OUTPUT ARTIFACT AND THE LEGALITY RULES.** Two Codex handoffs today were
  degraded by my briefing: one missed `n_kseg>=2` and halted a 4-arm matrix after 26 shapes; one
  never wrote a file. The agents did what I asked; I asked incompletely.
