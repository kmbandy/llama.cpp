# ⭐ DSWS ESTABLISHED BASELINE — 2026-07-21 ⭐
## G=6 / G=4 × CFASSIGN off/on, all real ml8 + mlambaformer shapes

**This file replaces `RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md`, whose DSWS column was a parser bug
(see §6). Every number here is tick-derived and self-validated. Do not cite the old file.**

---

# §1 — HOW THESE NUMBERS WERE PRODUCED (read before citing any of them)

Harness: `dsws_realshape_bench.py` (+ `test_dsws_realshape_bench.py`). It does **not** scrape a rendered
TF value. It derives throughput from full-precision integers emitted by the dispatcher:

```
padded_TF = 2 * padded_M * N * K * reps * tick_hz / summed_ticks / 1e12
real_TF   = padded_TF * real_M / padded_M          <- padding counts AGAINST us, never for us
reps      = WORK_EXACT_count / (G * TOTAL_super)   <- derived, not read off a rendered field
```

**Every row is refused unless all of the following hold**, and a refused row carries no TF at all:
- exactly one of each required emission, and header/dispatch/timing geometries agree;
- `occ[0]=0`, oracle `ok>0 bad=0`, no abort / timeout / INCOMPLETE / WORK-INEXACT / dirty canary;
- derived reps agree with the `SUSTAINED` line; timed chunks == chunks/rep × reps;
- **the tick-derived TF agrees with the kernel's own rendered `TF=` to within one-decimal rounding**,
  and independently with its `% of peak` field.

That last check is the one that would have caught this morning's bug. It is proven non-vacuous: a log
with its tick count halved and its rendered TF left intact is **rejected**
(`SELF_VALIDATION_MISMATCH`), as is one with `bad=1` under a "CLEAN" banner. Acceptance over the 114
archived logs: **91 pass, 23 refused** (11 invalid-run markers, 8 with no WORK-EXACT gate, 3 non-real
geometry, 1 no oracle).

Every row names its source log. Raw: `~/dsws_gpu_logs/mx{A0,A1,B0,B1}_*.log`,
tables/JSON in `~/dsws_gpu_logs/matrix_2026-07-21/`.

### ⚠ THE MEASUREMENT CONDITION THAT DOMINATES EVERYTHING — STATE IT WITH EVERY NUMBER

**This baseline is measured at the compositor-safe cap of 512 tiles/dispatch.** That cap is a *safety
requirement* (rule 7: a kernel saturating HBM starves the compositor and kills the desktop with NO GPU
reset), but it is also **the single largest performance factor found on 2026-07-21** — larger than
CFASSIGN, wave count, or G. See §5. A DSWS number quoted without its chunk count is meaningless.

---

# §2 — ⭐ THE BASELINE MATRIX ⭐

All four arms: `WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 VBUDGET=1536 JDEPTH=1 STAGGER=1
DECENTASN=1 SELFSERVE=1 BANKZERO=1 RBU=1 INITBAR=1 TERMFIX=1 BATCH=1`, oracle stride 8,
`DSWS2_TARGET_SECS=1.5`, **512-tile cap**. 27 shapes dispatched per arm, 6 UNSUPPORTED, **0 failures**.
All WORK-EXACT, all oracle `bad=0`.

| arm | G | ACC_N | GROUPS | super-tile M | CFASSIGN | bin sha |
|---|---:|---:|---:|---:|---|---|
| A0 | 6 | 3 | 2 | 96 | off | `128500f7314cafce` |
| **A1** | 6 | 3 | 2 | 96 | **on** | `cac3ff7c2338e73f` |
| B0 | 4 | 4 | 1 | 64 | off | `31d5ce7cae0f647e` |
| B1 | 4 | 4 | 1 | 64 | on | `f6f315af34de4709` |

TF is **real-FLOP corrected** (M padded up to the super-tile, then scaled back by `real_M/padded_M`).

| shape | M | N | K | n_kseg | A0 G6 | **A1 G6+CF** | B0 G4 | B1 G4+CF | CF gain | hipBLASLt | best/hipBLASLt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ml8_dense_ffn_down | 2048 | 2560 | 9216 | 36 | 4.462 | **4.543** | 3.332 | 3.320 | +2% | 189.3 | 0.0240x |
| ml8_dense_attn_o | 2048 | 2560 | 4096 | 16 | 2.270 | **2.658** | 1.600 | 1.866 | +17% | 159.2 | 0.0167x |
| ml8_dense_ffn_down | 512 | 2560 | 9216 | 36 | 2.451 | 2.528 | 2.565 | **2.638** | +3% | 123.3 | 0.0214x |
| ml8_dense_attn_q | 2048 | 4096 | 2560 | 10 | 1.559 | **1.925** | 1.255 | 1.534 | +23% | 159.6 | 0.0121x |
| ml8_dense_ffn_gate_up | 2048 | 9216 | 2560 | 10 | 1.536 | **1.822** | 1.234 | 1.516 | +19% | 186.7 | 0.0098x |
| ml8_dense_ffn_gate_up | 512 | 9216 | 2560 | 10 | 1.340 | **1.673** | 0.970 | 1.196 | +25% | 135.1 | 0.0124x |
| ml8_dense_attn_q | 512 | 4096 | 2560 | 10 | 1.208 | 1.503 | 1.242 | **1.553** | +24% | 84.8 | 0.0183x |
| ml8_dense_attn_kv | 2048 | 1024 | 2560 | 10 | 1.220 | 1.498 | 1.241 | **1.548** | +23% | 97.3 | 0.0159x |
| ml8_dense_attn_o | 512 | 2560 | 4096 | 16 | 1.223 | 1.371 | 1.245 | **1.462** | +12% | 79.4 | 0.0184x |
| ml8_moe_attn_q | 512 | 4096 | 2048 | 8 | 0.989 | 1.224 | 1.014 | **1.262** | +24% | 70.0 | 0.0180x |
| ml8_moe_attn_o | 512 | 2048 | 4096 | 16 | 0.999 | 1.160 | 1.020 | **1.186** | +16% | 72.2 | 0.0164x |
| mlmf_mamba_out_proj | 4096 | 768 | 1536 | 6 | 0.663 | **0.767** | 0.614 | 0.748 | +16% | 68.9 | 0.0111x |
| mlmf_lm_head | 4096 | 32000 | 768 | 3 | 0.564 | **0.725** | 0.413 | 0.501 | +28% | 167.9 | 0.0043x |
| ml8_dense_attn_kv | 512 | 1024 | 2560 | 10 | 0.362 | 0.395 | 0.373 | **0.416** | +9% | 37.5 | 0.0111x |
| mlmf_attn_o_proj | 4096 | 768 | 768 | 3 | 0.333 | 0.380 | 0.332 | **0.382** | +14% | 45.7 | 0.0084x |
| mlmf_router_down_proj | 4096 | 256 | 768 | 3 | 0.226 | 0.252 | 0.235 | **0.258** | +11% | 15.7 | 0.0165x |
| mlmf_MoE_expert_fc1 | 512 | 1536 | 768 | 3 | 0.170 | 0.192 | 0.181 | **0.199** | +13% | 14.2 | 0.0140x |
| mlmf_attn_linear_k | 4096 | 192 | 768 | 3 | 0.176 | 0.191 | 0.181 | **0.198** | +9% | 15.0 | 0.0132x |
| ml8_moe_attn_kv | 512 | 512 | 2048 | 8 | 0.169 | **0.177** | 0.160 | 0.165 | +5% | 15.2 | 0.0117x |
| mlmf_MoE_expert_fc2 | 512 | 768 | 1536 | 6 | **0.190** | 0.176 | 0.180 | 0.190 | -8% | 17.3 | 0.0110x |
| ml8_moe_ffn_down | 512 | 2048 | 512 | 2 | 0.159 | 0.175 | 0.160 | **0.176** | +10% | 12.4 | 0.0143x |
| ml8_moe_attn_o | 64 | 2048 | 4096 | 16 | 0.164 | 0.169 | 0.171 | **0.176** | +3% | 9.6 | 0.0183x |
| ml8_moe_attn_q | 64 | 4096 | 2048 | 8 | 0.162 | 0.155 | 0.160 | **0.169** | -4% | 16.9 | 0.0100x |
| ml8_moe_ffn_down | 64 | 2048 | 512 | 2 | 0.023 | 0.024 | 0.023 | **0.024** | +4% | 1.6 | 0.0147x |
| ml8_moe_attn_kv | 64 | 512 | 2048 | 8 | 0.024 | 0.023 | 0.024 | 0.023 | -2% | 1.7 | 0.0139x |

| arm | n | mean | median | max | CV |
|---|---:|---:|---:|---:|---:|
| A0  G=6 ACC_N=3 | 25 | 0.906 | 0.564 | 4.462 | 1.100 |
| **A1  G=6 ACC_N=3 +CFASSIGN** | 25 | **1.028** | **0.725** | **4.543** | 1.037 |
| B0  G=4 ACC_N=4 | 25 | 0.797 | 0.413 | 3.332 | 1.009 |
| B1  G=4 ACC_N=4 +CFASSIGN | 25 | 0.908 | 0.501 | 3.320 | 0.942 |

**⭐ A1 (G=6, ACC_N=3, CFASSIGN=1) IS THE BASELINE CONFIG. ⭐**

**WINS vs hipBLASLt: 0 / 25.** Best ratio on the table is 0.024x. We do not beat the vendor anywhere.

---

# §3 — CFASSIGN (counter-free assign) IS ADOPTED

**+13.5% mean at G=6 (0.906 -> 1.028), +13.9% at G=4 (0.797 -> 0.908).** Near-identical gain in two
independent tilings over 25 shapes each — that reproducibility across geometries is why it is adopted
rather than treated as noise. Gains concentrate on mid-size dense shapes (+12% to +28%); ~0 on the
chunk-overhead-bound tiny MoE shapes and on `n_kseg=36`, where the field (64 wide) already exceeds 30
waves so there was never a shortage of units to contend over. Two shapes regress slightly (-8%, -4%,
-2%) — all in the sub-0.2 TF noise floor.

**What it does:** removes the shared `ASSIGN_HEAD` CAS. Each wave derives its unit from its wave id
within the current cohort; one atomic per TILE (`occ[20]`, unchanged) instead of one CAS per unit.
Built by Codex; gated behind `CFASSIGN`, **byte-identical at `CFASSIGN=0`** (verified: same source
builds to the HEAD sha with the flag off).

**Correct across the whole shape space:** WORK-EXACT + oracle `bad=0` on all 27 shapes at
`n_kseg` ∈ {2,3,6,8,10,16,36} — every cohort/field width the mapping must handle. Bring-up used a
**dense (stride=1) oracle, 864/864 tiles**, because the count gates (TILEDONE/DRAIN/GSTORED) are
structurally blind to *duplicated* work.

**Guards (assembler-enforced, do not remove):** `CFASSIGN` requires `DECENTASN=1`, `SELFSERVE=1`,
`BATCH=1`, and **`WAVES <= SSWIN`** — a flat wid→unit map with more waves than control slots aliases
two waves onto one `SL_GEN`/`SL_RBDONE` and produces **wrong C with exact counts**. This is why the
baseline runs `SSWIN=32`: `CFASSIGN` cannot build at `WAVES=30, SSWIN=8`.

---

# §4 — G=6/ACC_N=3 BEATS G=4/ACC_N=4

G=4 loses on the large shapes (0.70–0.80x) and ties on the small. `GROUPS = G/ACC_N`, so G=4/ACC_N=4
gives GROUPS=1 (no group serialization) and a 64-row super-tile that divides every real M exactly — both
of which *should* favour it. They do not, at the 512-tile cap. **Not explained.**

⛔ **A claim was made and retracted on 2026-07-21 that G=4 was 1.7x–12.6x FASTER.** That compared
morning G=4 runs (1 unbounded chunk) against afternoon G=6 runs (512-tile cap). It was a chunking
artifact, not a property of G. The G comparison above is cap-matched and stands; **the G question is
NOT settled at 1 chunk** — no G=6 data exists in that condition.

---

# §5 — ⚠ THE COMPOSITOR CAP IS THE LARGEST SINGLE FACTOR ⚠

Measured directly, same shape, same binary, same config (`G=4 ACC_N=4 SSWIN=8 CFASSIGN=0 WAVES=30`),
**one variable — `ML8_COOP_CHUNK`:**

| shape | M | 1 chunk | 512-tile cap | cost |
|---|---:|---:|---:|---:|
| mlmf_lm_head | 4096 | **7.314** | 0.413 | **17.69x** |
| ml8_dense_ffn_gate_up | 2048 | **7.158** | 1.234 | 5.80x |
| ml8_dense_attn_q | 2048 | **4.110** | 1.255 | 3.28x |
| ml8_dense_attn_o | 2048 | **4.284** | 1.600 | 2.68x |
| ml8_dense_ffn_gate_up | 512 | **2.584** | 0.970 | 2.66x |
| ml8_dense_ffn_down | 2048 | **7.516** | 3.332 | 2.26x |

**mean 2.23x, median 1.08x, max 17.69x.** Uneven by construction: it is a fixed per-dispatch cost, so
shapes with many tiles pay it repeatedly (`lm_head` = 32000 tiles = 63 chunks) while shapes already
under 512 tiles pay nothing. `SSWIN` is NOT a factor — 8 vs 32 measured 1.3 vs 1.2 on the same shape.

**The full 1-chunk sweep reproduces 2026-07-21 morning shape-for-shape within ~1%** (`ffn_down M2048`
7.463 -> 7.516; `lm_head` 7.205 -> 7.314; `ffn_gate_up M2048` 7.236 -> 7.158). Those morning numbers
were REAL; only the sweep script's *table* was fabricated (§6). Raw: `~/dsws_gpu_logs/cap1_*.log`,
`matrix_2026-07-21/CAP1_G4_SSWIN8.json`.

**BEST HONEST DSWS NUMBER TO DATE: 7.5 TF** (`ffn_down M2048`, 1 chunk) — against hipBLASLt's 189.3,
i.e. **~25x behind**, not the ~80x the capped numbers imply.

**Open and worth real work:** the 512 default was set after a 2.46s single chunk killed Hyprland. It was
never derived. `ffn_gate_up M2048` runs a 4608-tile chunk in **13ms** — 58x under the 0.75s abort
threshold. The safe ceiling is likely far above 512 and is worth finding: it is free throughput.
Note the abort guard fires only BETWEEN chunks, so a single chunk is never interruptible regardless of
size — that is the real constraint, not the tile count.

---

# §5a — THE VENDOR BAR, RE-MEASURED METHODOLOGY-MATCHED (2026-07-21)

The old vendor harness (`~/dsws_gpu_logs/bench_hipblaslt_ml8.py`) was never wrong arithmetically — it
derives TF from a measured wall time, no string parsing, full precision in its JSON. But the
**comparison** was inconsistent: it reported `min`-of-20/50 (the vendor's luckiest run) while DSWS
reports a mean. Re-measured with `bench_hipblaslt_matched.py`:

- **primary statistic = mean-of-30**, matching DSWS (min/median/max/spread all retained)
- **shapes IMPORTED from `dsws_realshape_bench.SHAPES`** — the same Python object, so the two sides
  cannot drift; there is no second shape list to maintain
- run under a held board claim

**HARNESS CROSS-CHECK PASSED:** new min-of-30 vs old min-of-N agrees to **median 1.023**. The two
harnesses measure the same thing; only the statistic differed. The old numbers were not fabricated.

**⚠ min-of-N OVERSTATED THE VENDOR BY 17.0% ON AVERAGE** (mean/min-time = 0.855 mean, 0.862 median).
Cause is visible in the data: hipBLASLt's own **per-call spread is median 36%, max 79%**, so taking
the minimum of 20–50 draws reliably selected a lucky one.

**⭐ THE HEAD-TO-HEAD, MEAN vs MEAN ⭐** (`~/dsws_gpu_logs/hipblaslt_matched_2026-07-21.{json,txt}`)

| | DSWS A1 (512-cap) | DSWS 1-chunk | hipBLASLt (mean-of-30) |
|---|---:|---:|---:|
| mean over 25 shared shapes | **1.028** | **1.808** | **66.89** |
| ratio to vendor | **0.0154x** | **0.0270x** | — |
| **WINS** | **0 / 25** | **0 / 25** | — |

Selected shapes (DSWS real-FLOP corrected; vendor mean, with the retired min-of-N for reference):

| shape | M | A1 (512-cap) | 1-chunk | **hipBLASLt mean** | ~~min-of-N~~ | vendor spread |
|---|---:|---:|---:|---:|---:|---:|
| ml8_dense_ffn_down | 2048 | 4.543 | **7.516** | **180.4** | ~~189.3~~ | 26% |
| ml8_dense_ffn_gate_up | 2048 | 1.822 | **7.158** | **176.1** | ~~186.7~~ | 27% |
| mlmf_lm_head | 4096 | 0.725 | **7.314** | **159.9** | ~~167.9~~ | 13% |
| ml8_dense_attn_o | 2048 | 2.658 | **4.284** | **155.3** | ~~159.2~~ | 26% |
| ml8_dense_attn_q | 2048 | 1.925 | **4.110** | **147.9** | ~~159.6~~ | 19% |
| ml8_dense_attn_kv | 512 | 0.395 | 0.401 | **29.1** | ~~37.5~~ | 36% |
| ml8_moe_ffn_gate_up | 512 | 0.177 | 0.181 | **12.4** | ~~15.4~~ | 42% |

**BEST HONEST STANDING: ~24x behind at our best shape** (7.516 vs 180.4), not the ~80x the capped
numbers against min-of-N implied. Still zero wins anywhere, on any shape, in any configuration.

**THE ONE ASYMMETRY THAT REMAINS AND CANNOT BE EQUALIZED:** hipBLASLt runs ONE unchunked call; DSWS
runs under the compositor cap. That is a constraint we accept and the vendor does not (§5). Quote the
DSWS chunk condition with every ratio; never absorb it silently.

### Two free wins visible in the vendor data, still unclaimed
- `mlmf_mamba_in_proj` N=4200 is **REJECTED** by `torch._scaled_mm` (N%16) and falls back to bf16 at
  **67.1 TF**, while the padded `mlmf_in_proj_ML8PAD` N=4208 runs fp8 at **105.7 TF** — a **1.58x**
  free win for a one-line weight-tensor pad. (Known since 2026-07-13; still not taken.)
- `mlmf_router_out` N=8 also rejects and runs bf16 at 0.244 TF. Both models contain shapes the
  vendor's own fp8 path refuses.

### RETIRED
`~/dsws_gpu_logs/hipblaslt_ml8_baseline.json` (min-of-N, own shape list) is **superseded** by
`hipblaslt_matched_2026-07-21.json`. Do not cite the old file for head-to-head ratios; its numbers are
best-case and its shape list is a second copy that can drift from ours.

---

# §6 — WHAT THIS FILE REPLACES

The prior results file claimed **4 wins over hipBLASLt** (`moe attn_kv M64` 10.87 = "6.39x") and a
flatness result (CV 0.700 vs 0.922). **All fabricated by an extraction bug.** The old harness matched
`'<num> TF'`; the kernel prints `TF=<num>`, so the pattern never matched and every row fell through to
a fallback that took the LAST decimal on the line — the `spread N%` jitter field, or `% of peak`.

**All 46 rows were wrong. None was correct.** Errors span 0.3x to 601x. The four "wins" were simply the
rows where the wrong quantity was largest: tiny MoE shapes are 0.1 GFLOP problems that jitter 10–16%
between reps, so their *instability* was published as their throughput. `moe attn_kv M64` published as
10.87 TF actually reads `TF=0.0`, and re-measured under the same config today reads **0.0253**.

The bug **inverted the table** — it made the least reliable shapes look best and understated the
genuinely fast dense shapes, whose spread is small. It then survived because it said what the project
wanted to hear, and nobody opened a raw log.

Also retracted the same day: an unverifiable "bin sha `397bfbe1cb010c6e`" that matched no artifact, and
reporting `TF=0.0` as a measured zero when it is a one-decimal `printf` floor (true value ~0.035).

**The rule that follows:** derive values from full-precision inputs, cross-check against the renderer,
and refuse rather than report on mismatch. That is now enforced in the harness, not left to discipline.
