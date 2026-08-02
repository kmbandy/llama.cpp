# DSWS S1 (MAD-305) — MORNING BRIEF, 2026-08-01

**SUPERSEDES `DSWS_BRIEF_2026-07-30_AM.md`.** Its §0 (bring-up) is DONE, and its §2 cut-list is
DE-PRIORITISED by measurement — see §2 below before spending a day on instruction economy.

Detail: `DSWS_TESTING_LOG.md` §78–86 (4,895 lines). Data: `DSWS_STATIC_MATRIX_2026-07-31.csv`,
`DSWS_DYNAMIC_MATRIX_2026-07-31.csv`. KG newest first: `84089f78` (the campaign) · `340f6965`
(cross-session TF is invalid) · `baf67773` (instruction cuts) · `8e201972` (**shapes are inputs, not levers**).

---

# 0. STATE — NOTHING IS IN FLIGHT

Tree clean of staged changes; **NOTHING STAGED** (shared with a live weight-pager session).
Latch CLEAR. No board claim held. No GPU work pending.
Source `57ab3100c9450ad6` · bin `58e965a46f3e162d` · `.text` 28,852 · LDS 34,304.
**A bare `./build_flow.sh` now produces exactly that bin** (§86b — it did not, until yesterday).

**There is no queued dispatch. Nothing needs the GPU first thing.** Yesterday's campaign is complete and
logged. Start from §3 (what to measure next), not from a run.

---

# 1. ★ THE RESULT THAT MATTERS ★

Twelve kernel variations, measured statically (RGA) and dynamically (silicon), joined 1:1 on the **same
twelve binaries**. Per kmbandy's directive: *"measure and only measure… the commonalities across the
variations are going to tell us structurally where the bottlenecks are and which levers are bottlenecks
themselves."*

### 1a. DYNAMIC — a two-term cost per coordination event (§85)

> ### `ns_per_event = 20.80 + 37.36 × MFLOP_per_event`
> **Fit on TWO cells (base, fm1). Predicts the two held out: `fm4fn2` −0.5%, `fn2` +0.9%.**

**At the config of record that is 20.8 ns of 60.1 ns — 35% OF RUNTIME — as fixed per-event cost carrying
no work.** Consequences, all measured:
- Halving work-per-event (`fm1`, `fn2`) costs **25% throughput**: same overhead, half the payload.
- **The frag grid and feed ratio are IRRELEVANT at matched work-per-event.** `base` vs `fm4fn2`: 60.07 vs
  60.28 ns — **0.3% apart** across a transposed 2×4→4×2 grid and superM 256→512.

### 1b. STATIC — 90% of the kernel is unreachable by every lever we have (§82–83)
COMPUTE varies **+106%** across cells; REST varies **+8%** and is **87.6–93.5% of the kernel in every
cell**. `SGPR=72` and **zero spills** in all twelve; `livereg` tracks only the frag grid.
**What REST is:** `v_cmp_eq_u32` = 360 and `s_cbranch_execz` = 360, *exactly equal* → the `lds_*` lane-0
accessor family. **`lds_put` is 9 instructions to write ONE 32-bit LDS word** (4 exec-guard, 2
scalar→vector marshal, 1 `ds_store`, 1 wait, 1 restore). 267 source sites → **360 guarded blocks ×
5 pure-bookkeeping instructions = 1,800 = 34% of the kernel, moving no data.**

### 1c. ⚠ THE CORRESPONDENCE — NOT YET A FINDING, AND THIS IS THE MAIN OPEN QUESTION
**35% of runtime (dynamic) vs 34% of instructions (static) may or may not be the same thing.** Two matching
percentages are not evidence of a causal link. **Do not cite this as established.** §3 is how to settle it.

---

# 2. WHY THE 07-30 CUT-LIST IS NOT THE PRIORITY ANY MORE

That brief's §2 ordered ~17 instrs/rowblk of address arithmetic, blocked on 3 SGPRs. **Measurement now
says that work would land flat**: the five cuts (389 instructions, −17% of the slice) measured **+0.44%,
inside noise** (§79), and §82 explains why — every geometry lever reaches only ~10% of the code, and the
cuts were inside that 10%. Unblocking three SGPRs to remove ~17 more instructions from the same 10% is
not where the time is. **Do not spend a day there.** The `global_load_tr_b128` item (halves B-loads) is
unaffected by this reasoning but remains a design conversation touching the wait pipeline.

---

# 3. ★ WHAT TO DO NEXT — SETTLE §1c BY DESIGNED ABLATION, NOT BY REDESIGN ★

The question is exactly one thing: **is the 20.8 ns fixed per-event cost the `lds_*` exec-guard idiom, or
something else that merely happens to be a similar fraction?**

**The falsifier: make the guard cheaper on a subset of `lds_*` sites and re-fit `ns_per_event`.** If the
fixed term falls roughly in proportion to the guard instructions removed, the link is causal. If the fixed
term does not move, the correspondence is a coincidence and the real cost is elsewhere — and that is a
*result*, not a failure.

Candidate cheapenings to evaluate **offline first** (rule 6), cheapest and least invasive first:
1. `s_mov_b32 exec_lo, 1` in place of `v_cmp_eq_u32` + `s_and_b32` where the target is unconditionally
   lane 0 — saves 2 of the 5 guard instructions per block. **Verify the exec-restore still pairs.**
2. Hoisting the exec save/restore across *adjacent* `lds_*` calls (many sites appear in runs), amortising
   2 instructions over N accesses instead of paying them per access.
3. The scalar→vector marshalling `v_mov`s — check whether `ds_store` can take the address from an already
   resident lane-0 vreg rather than re-materialising it each call.

**None of these is a commitment to ship.** They are ablation arms whose only job is to move the fixed
term. Build them, RGA them, confirm the instruction delta, THEN measure. **Correctness is non-negotiable:
these touch coordination-state writes — a wrong LDS write is a silent wrong C, and the oracle is the only
thing that has ever caught this class.**

### Also open, in rough value order
- **SEGK's extra cost is unexplained.** `segk128` +27% and `segk64` +53% ABOVE the per-event prediction.
  Its own 3-point fit is `time ≈ 1.63 ms + 0.108 ms × n_kseg`, slopes agreeing to **1% over a 4× range**.
  SEGK is the strongest measured axis (17.5 / 10.2 / 5.6 TF) and the per-event model does not cover it.
- **The G-varying cells miss too** (`g4` +21%, `g6accn2` +20%). GROUPS costs ~10% per step at fixed
  coordination count, measured in both directions — but it is not in the model.
- **`waves16` is a real −14%** (clock-normalised) with `coast/computed` = **13.023 vs base's 1.246**.
  It does NOT reproduce §45's 10.2 TF at matched geometry.
- The 3 `--segk 128` shapes; the 30-shape gate at the cuts config; tasks #43/#45/#46 still `in_progress`.
- **`build_flow.sh`'s WG/CU + pool figures are still a hardcoded string** (§86b), correct only at 34,304 B.
  Deriving them from the published `.lds` sidecar is the durable fix.

---

# 4. ⛔ MEASUREMENT VALIDITY — READ BEFORE QUOTING ANY NUMBER

### 4a. TF IS NOT COMPARABLE ACROSS SESSIONS (§80)
The **identical binary** (`beb031c195df`, hash-verified) measured **15.4 TF on 07-29 evening and 17.5 TF
on 07-31 morning — +12.5% on byte-identical code**, same shape, same env, same host. Yesterday's four
repeats were tight (±1.3%), so this is a **day-level shift, not noise**.
**NOT THERMAL, measured:** 55 °C/51 W before vs 54 °C/49 W after; a 0.69 s run at 5.7% of peak does not
move the die. **Cause still unidentified.**
> **Only SAME-SESSION A/B against a rebuilt control bin is valid. Any baseline carried across a session
> boundary must be re-measured in-session before it is compared to anything.**

Within-session sweeps are unaffected — all twelve cells ran in one session, and `base` reproduced
5.5365 ms/rep against the morning's 5.5137 (**+0.41%**), confirming the drift did not recur intraday.

### 4b. CLOCKS VARY 28% ACROSS CELLS — THE INSTRUMENT IS NOW MANDATORY (§81, §85)
`gpu_run.sh` records busy-band sclk/power/temp to `$LOG.telemetry` on every run. **First real use caught a
1806–2307 MHz spread across the twelve cells**, which silently rewrites every sub-15% comparison:
- **`waves8`'s apparent −6.3% is ENTIRELY CLOCK.** Normalised 17.8 vs base 17.5 — WAVES 6→8 costs nothing.
  Without the instrument this would have been logged as a wave-axis result.
- `accn2` is **worse** than raw TF showed: −10% normalised, not −7%.
**Normalise before quoting any effect under ~15%.**
`$LOG.journal` is a `journalctl` **error** grep and is correctly empty on a clean run — it is brick
forensics, NOT telemetry. Never read its emptiness as missing data.

### 4c. PHASEPROBE IS A BRICK RISK — DO NOT RUN IT
`phase_stamp` issues **`s_sendmsg_rtn_b64 MSG_RTN_GET_REALTIME` + `s_wait_kmcnt 0`** — an unthrottled
message-bus RTC read at 27 sites. That is the exact 2026-07-14 brick vector rule 5 names and that
`DUTYPROBE` is hard-disabled for. 44× overhead, and its cost lands *per transition*, biasing the very
distribution it would measure. **The free `STAGINSTR` counters gave the whole dynamic attribution at zero
overhead and zero risk.** Use those.

---

# 5. THE REFERENCE NUMBER — RE-READ IT, DO NOT RE-QUOTE IT

`RESULTS_DSWS_vs_hipBLASLt_2026-07-21.md` is authoritative and **already contains its own retraction**.
**hipBLASLt on `ml8_dense_ffn_down` M2048 N2560 K9216 = 189.3 TF** (62% of the 307 TF roofline).
We were at 4.36 (2.3% of vendor) on 07-21; at 17.5 we are **~9%** — a real 4× since then, ~11× still to go
on that shape. Their span across these shapes is 1.6 → 189.3 TF.
**I asserted yesterday that we had never measured this. That was wrong and the file was in the tree.**
A figure in a brief is a POINTER, not a measurement — re-read the source before reasoning from it.

---

# 6. RETRACTIONS FROM YESTERDAY — ALL SELF-CAUGHT, LISTED SO THEY DO NOT RECUR

1. **The `TOTAL_super` grouping** (§85a). I built a "two coordination levels" story across three cells;
   the dispatcher's own `computed` column refutes it — all non-SEGK cells run **identical 92,160
   events/rep**. I reasoned from a formula never checked against run output. *The pairs are real; my
   reason for them was wrong.*
2. **"The empty journals are a bug"** — no; see §4b.
3. **"WAVES=4 fails from signed overflow at 2³¹"** — refuted in ten seconds by assembling the literal.
   Real cause: `NCOMPUTE=1` → `BATON_MAGIC = 2³²` (§84).
4. **"`lds_put_r` is a duplicate definition"** — withdrawn; it is `.if !(DSWS2_CONV || DSWS2_ENVELOPE)`
   guarded, with a comment saying why.
5. **"We have never measured hipBLASLt on our shapes"** — wrong; see §5.
6. Standing from 07-30: **"we are 3.7× worse than hipBLASLt on instruction economy" remains WITHDRAWN.**
   That window spanned 60 labels including the entire retire/drain/C-store path. We still do not have a
   window that is provably only the k-step loop.

---

# 7. METHOD THAT EARNED ITS KEEP — KEEP DOING THESE

- **Pre-register before each cell.** Predicting `g4 ≈ 13` and getting 14.4 is what exposed GROUPS as a
  second variable, which then **confirmed on a held-out pair in both directions** (~10%/step).
- **Fit on a subset, predict the rest.** The per-event model was fit on 2 cells and predicts 2 others to
  <1%. A curve through all four would have proved nothing — twelve points accept almost any model with
  enough terms, which is how the feed-ratio model survived as long as it did. **No third parameter was
  fitted to rescue the SEGK/G cells; they are logged as unexplained on purpose.**
- **Reconstruct-and-hash-validate.** An uncommitted prior revision was recovered by scripted inversion of
  documented edits and **proven exact by binary hash before it ever ran** (§79).
- **Verify the analysis object matches the shipped one.** RGA re-assembles with only the defsyms you hand
  it; the harness extracts each cell's **exact** clang line from `bash -x` and appends `RGADESC=1`.
  Related trap: RGA's `USED_LDS_BYTES`/`USED_VGPRs` are artifacts of that analysis descriptor (it reports
  13,824 B for a 34,304 B kernel) — take only livereg/SGPR/spills/ISA-size from RGA.
- **`/usr/bin/rga` on this box is ripgrep-all.** The real one is `~/Downloads/rdts/…/rga` (v2.14.2.8).
- **Guard conditions that must agree should be ONE symbol.** `A_MI1_HOISTED` (§77) and now the
  `NCOMPUTE < 2` pair (§86a): splitting either yields code that assembles and is silently wrong.

---

# 8. TREE STATE AT LOGOFF
Modified under `spike/dvgpr_occ/`: `occ_kernel_dsws_flow.s` (guard fix), `build_flow.sh` (telemetry-era
defaults + corrected header), `gpu_run.sh` (telemetry sampler), `DSWS_TESTING_LOG.md` (§78–86),
`occ_dispatch.cpp`, `dsws_realshape_bench.py`, `DSWS_BRIEF_2026-07-28_AM.md`, 3 `.lds` sidecars.
New: `DSWS_STATIC_MATRIX_2026-07-31.csv`, `DSWS_DYNAMIC_MATRIX_2026-07-31.csv`, this brief.
**NOTHING STAGED.** Latch CLEAR. No claim held. Bin `58e965a46f3e162d`.
