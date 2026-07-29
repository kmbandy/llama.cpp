# DSWS S1 (MAD-305) — EOD BRIEF 2026-07-23 (post-compact continuation)

Read this + `HARNESS.md` before touching anything. Every claim below is marked measured vs hypothesis.
This session: RCONV shipped its one real win; the entire boundary-election line was explored and
**closed as a dead end**; the real wall (frontier advance rate) is now isolated and an instrument
(ADVPROBE) is mid-build to measure it.

---

## 0. ONE-LINE STATE

RCONV works (RING_WAIT 56%→0.3%, the one win). The boundary FUNNEL is a **dead end** (structurally
~1.6× slower than the herd; read-spin ablation *falsified* the responsiveness theory). Both herd and
funnel are **100% ASSIGN-bound** — the real wall is the **frontier advance rate (~2,600 ticks/advance)**,
which no election tuning touches. ADVPROBE (throttled RTC timing the advance) is being built by terra to
split that 2,600 ticks into serial-work vs idle-waiting. **Run ADVPROBE first thing post-compact.**

## 1. CONFIG-OF-RECORD + HARNESS (the harness confusion cost hours — pinned in `HARNESS.md`)

**THE canonical A1 build → sha `cac3ff7c2338e73f`** (this is the byte-identity anchor for every gate):
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 \
  SEGK=256 POOL_N=1 G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 ./build_flow.sh
```
- **A plain `./build_flow.sh` at "A1" does NOT reproduce `cac3ff7c`** — its defaults are `STAGGER=0
  TFPROBE=0 STAGINSTR=0`, giving `55a6983d` (a *different* kernel). Always pass the full profile above.
- **Harness: `gpu_run.sh` is the ONLY dispatch path** (`./gpu_run.sh <name> -- <ENV...> ./occ_dispatch
  --dsws2`). `dsws_realshape_bench.py live` is the all-shapes sweep (wraps gpu_run). `build_flow.sh`
  builds the kernel; `build.sh` builds the host `occ_dispatch`. Deleted 4 stale run scripts this session
  (`dsws.sh` had the WRONG old config ACC_N=6/SEGK=64/POOL_N=2 and caused the confusion).
- **Standard single-shape dispatch** (ffn_gate_up M2048 = 2112×9216×2560): `FLOW_WAVES=30 DSWS2_FLOW=1
  DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 FLOW_POOL_N=1 DSWS2_SEGK=256 DSWS2_K=2560 DSWS2_ORACLE_MTL=22
  DSWS2_ORACLE_NTL=144 DSWS2_ORACLE_STRIDE=8 DSWS2_REPS=<n> STAGINSTR=1 FORENSICS=0 TFPROBE=1`. The host
  GEOM must match the bin's build defsyms. Expected: `computed = ACC_N×TOTAL_super = 190080/rep`,
  `emissions occ[96] = 63360/rep`, oracle bad=0.
- **PHASEPROBE/PHIST/BNDSPLIT/heavy-probe builds use `ML8_COOP_CHUNK=96`** (small chunk) so a chunk
  stays under the 0.75s compositor-abort cap. Non-probe fed runs can use the default 512.

## 2. THE MEASUREMENT CHAIN (what happened, in order, all measured)

1. **RCONV bring-up (DSWS2_RCONV=1):** first-ever runtime role conversion on silicon. `convCount>0`,
   WORK-EXACT, oracle-clean, no brick. Then the PHASEPROBE dynamic run: **RING_WAIT collapsed 56.0%→0.3%,
   WMMA 19.3%→44.9%.** THE ONE REAL WIN. But the wall moved: **SS_WAIT (self-serve reservation) 21.3%→50.7%.**
2. **SS_WAIT diagnosed via BNDSPLIT** (new occ[127-130] probe): the boundary election is a **93% thundering
   herd** — of every boundary entry, ZLOCK_LOST=93.1%, DRAINGATE_BAIL=0.0%, CSTOREGATE_BAIL=6.8%,
   ADVANCE=0.1%. ~930 waves storm the single ZLOCK election per advance.
3. **The FUNNEL** (readiness gate, DSWS2_FUNNEL): move the `GSTORED`-readiness check UPSTREAM of the
   election → waves only contest the ZLOCK when the advance is genuinely ready → no bail. Built by terra,
   read-only pre-gate at `:4274-4290`. On silicon: **CSTOREGATE_BAIL→0 (correct!)** but **~1.6× SLOWER.**
4. **Read-spin ablation** (DSWS2_FUNNEL_SPIN_N, bounded spin keeping waves poised instead of flowing off):
   **FALSIFIED the responsiveness theory.** 285M ticks/rep vs flow-off funnel 270M vs herd 166M — poising
   made it *slightly worse*. So the 1.6× is NOT flow-off latency; it's the **pre-gate itself** (checking
   readiness before the election, common to both variants).
5. **Conclusion:** the boundary-election line (herd/RCONV/funnel/read-spin) is CLOSED. The frontier advance
   rate (~2,600 ticks/advance) is the real ASSIGN-bound wall. **ADVPROBE** (in-build) measures it.

## 3. PROVEN / DEAD / OPEN

**PROVEN (measured):**
- RCONV = the one win. RING_WAIT 56→0.3, WMMA doubled. Correct + adaptive on silicon.
- The funnel readiness gate is *correct* (no-bail by construction, CSTOREGATE_BAIL→0 on GPU) but
  **structurally ~1.6× slower** than the herd, unfixable by poising (read-spin falsified it).
- Both herd and funnel are **100% ASSIGN-bound** (`door1 NOTHING-STAGED = 100%` of coast in both).
- Real work is identical across herd/funnel: **emissions occ[96] = 63,360/rep** in both. This is the
  trustworthy advance signal. (The BNDSPLIT `ADVANCE`/PCSTORE counter is NOT — see §6.)

**DEAD (do not revisit):**
- The boundary FUNNEL as a throughput play. It can't beat the herd. Code is gated off, harmless in-tree.
- "Responsiveness / per-advance latency" as the funnel's 1.6× cause — falsified by the read-spin.
- A designated advancer (off-ethos, kmbandy rejected). A fixed-K token bucket (dam). A time-gate funnel
  (`s_memtime` unsupported on gfx1201; `s_sendmsg_rtn` bricks).

**OPEN (the real work, post-compact):**
- **What sets the ~2,600 ticks/advance frontier rate?** Hypothesis (NOT yet measured): mostly idle-waiting
  (~94%), likely the C-store (`GSTORED`) completion rate that gates each advance — connects to the old
  "FLUSH/split-K reduction is the wall" finding and the **SEGK lever** (bigger SEGK = fewer flushes).
  `zero_banks` is estimated ~150 ticks (~6%), NOT the wall, and is NOT cleanly ablatable (its TILEDONE
  reset at `:1232-1237` is structural — a naive skip would wedge). ADVPROBE settles serial-vs-idle.

## 4. INSTRUMENTS BUILT THIS SESSION (all defsym-gated, DSWS2_x=0 byte-identical to cac3ff7c)

| defsym | what | key sites | on-sha (A1+RCONV) |
|---|---|---|---|
| `DSWS2_RCONV` (0) | runtime role conversion: starved compute wave writes ROLE[wid]=ROLE_AFEED at coast, threshold `DSWS2_RCONV_COAST_N`=64, counter s50, convCount occ[48] | note-drop `:4888`, reset `:3339` | `53a309f7` |
| `BNDSPLIT` (0) | boundary-interlock waterfall: 4 throttled(s71 1/64) counters occ[127-130] → herd/drain/cstore/advance split by subtraction. s57 exec-save | pre-gate reads at `.Lflow_da_boundary`, tail bumps | — |
| `DSWS2_FUNNEL` (0) | readiness pre-gate (reads-only): flow on unless (DRAIN>=ASSIGN AND GSTORED>=z>>shift). s54/s55 | `:4274-4290`, flow-off `.Lflow_feedmt_sleep` | `79ff525b` |
| `DSWS2_FUNNEL_SPIN_N` (1024) | bounded read-spin: not-ready re-reads up to N then flows off. Counter s56. `.Lflow_da_funnel_ready`/`_notready` (out-of-line at `:4916`) | `:4278-4290`, `:4916` | `390056a8` |
| `DSWS2_ADVPROBE` (0) | **IN-BUILD (terra task-mry0n3aw)**: throttled(s71) RTC delta ZLOCK-win(`:4296`)→DA_ZDONE-write(`:4326`+tile), accumulate ticks + advance-count into free occ slots, host prints ticks/advance | TBD | TBD |

- `build_flow.sh` passes all of these through (env-gated defaults preserve byte-identity). RCONV requires
  CFASSIGN=1 (`.error` guard `:997`); BNDSPLIT/FUNNEL require DEADMAN=1 for the s71 throttle.
- **RGA note:** terra's sandbox blocks `~/.rga` (read-only), so terra reports the spill gate BLOCKED;
  **claude__main runs RGA itself** — all builds so far: 0-spill, USED_SGPRs unchanged at 72.

## 5. ADVPROBE — verify + run first post-compact

Terra is building it (task `task-mry0n3aw-3ai59p`). When it lands (watch `CODEX_ADVPROBE_PROGRESS.md`):
1. **Verify:** RGA 0-spill (run it — sandbox blocks terra); byte-identity `DSWS2_ADVPROBE=0`==`cac3ff7c`;
   confirm the RTC reads are throttled on s71 (unthrottled RTC on the hot advance path BRICKS — this is
   the one hard constraint); confirm the win→DA_ZDONE stamp pairing.
2. **Run** (greenlit-per-dispatch): `DSWS2_ADVPROBE=1 DSWS2_RCONV=1` at A1, `ML8_COOP_CHUNK=96`, reps=2.
   Read `ticks/advance in the critical section`. Compare to the ~2,600-tick total interval
   (166M ticks/rep ÷ 63,360 advances = herd; measure the actual for this build).
3. **Interpret:** critical-section ≪ 2,600 → the advance is **idle-waiting-bound** (next: is it the
   C-store `GSTORED` rate? probe GSTORED bumps vs advances) → attack the split-K reduction rate (SEGK).
   critical-section ≈ 2,600 → the serial advance work IS the wall (unexpected given the ~150-tick estimate).

## 6. CORRECTIONS / GOTCHAS FROM THIS SESSION (I was wrong several times; these are pinned so I don't repeat)

- **The "90× slower" funnel claim was WRONG** — I compared the funnel (steady state) to the RCONV
  *bring-up* (57.6ms, spin-up, which I'd even flagged as not-steady-state). Real, steady-state-to-steady:
  **~1.6-1.7×.** Always compare steady-state to steady-state; never against a spin-up run.
- **The BNDSPLIT `ADVANCE`/PCSTORE counter is MISLEADING.** It counts waves that reach the post-election
  C-store gate, which the funnel legitimately cuts by flowing waves off *before* the election — NOT real
  advances. It showed the funnel doing "20× fewer advances" while completing identical work; that's
  impossible. **The trustworthy work/advance signal is `occ[96]` emissions = 63,360/rep** (identical in
  herd and funnel). Grep the actual DA_ZDONE writes, not a proxy counter.
- **Responsiveness was FALSIFIED, not confirmed** — the read-spin ablation (keeping waves poised) made it
  *worse*, not better. Good outcome: we KNOW now. "Ablate, don't theorize" paid off.
- **PHASEPROBE has a blind spot for the funnel:** it only stamps the compute path; the funnel's
  flow-off/idle-coast time isn't stamped and mis-bills to the next phase. The funnel's compute-path split
  looked *fine* (SS_WAIT 47.2 vs 50.7) precisely because the extra 1.6× lives where PHASEPROBE can't see.
- **`zero_banks` is ~150 ticks (~6%), not the wall,** and not cleanly ablatable (TILEDONE reset is structural).
- **INCOMPLETE ≠ brick.** The funnel bring-up returned INCOMPLETE because the first 512-chunk ran 0.76s >
  the 0.75s compositor cap (the funnel is slower) → gpu_run aborted cleanly. No reset, card healthy. Rule 3
  full-stop still applied; kmbandy cleared the latch. Use `ML8_COOP_CHUNK=96` for slow/probe builds.

## 7. TREE STATE

- `occ_kernel_dsws_flow.s`, `occ_dispatch.cpp`, `build_flow.sh` — modified (RCONV + BNDSPLIT + FUNNEL +
  read-spin + host prints + passthroughs; ADVPROBE incoming). **All defsym-gated; DSWS2_x=0 byte-identical
  to `cac3ff7c`.** Nothing staged. Baseline bin on disk = `cac3ff7c`. Latch clear. Card free.
- New docs (untracked): `HARNESS.md` (authoritative harness/config), this brief, `DESIGN_BOUNDARY_FUNNEL_
  2026-07-23.md` (rev 2, the funnel design — now a dead end), `CODEX_*_PROGRESS.md` (terra records),
  `HANDOFF_*_2026-07-23.md`.
- Shared git tree with a live weight-pager session — **stage nothing, flag before any `git diff`, never
  touch `occ_kernel_coop.s`**, only the spike dir.
- Codex terra: write mode works via `--write`; it runs as a background codex task (`task-*`) that the
  codex-rescue launcher can't poll — **watch its output FILES** (CODEX_*_PROGRESS.md + kernel mtime) to
  detect completion. `/codex:status` is user-invocable only (I can't call it).

## 8. NEXT ACTIONS (post-compact, in order)

1. **Verify + run ADVPROBE** (§5). This is the entry point — it tells us serial-vs-idle for the advance.
2. If idle-waiting-bound (expected): **probe whether the advance is C-store(`GSTORED`)-rate-limited** —
   count GSTORED bumps vs advances; if ~1:1 and co-paced, the frontier is C-store-limited.
3. If C-store-limited: **attack the split-K reduction rate** — the **SEGK lever** (bigger SEGK = fewer
   flushes; SEGK/POOL_N trade at identical LDS per decision 9ccbf559) is the standing candidate, and it
   connects to the old "FLUSH is the wall" profile. This is the ASSIGN-bound attack neither the herd nor
   the funnel touches.
4. Consider stripping the dead funnel/read-spin code once the frontier direction is settled (optional; it's
   gated off and harmless).

## 9. THE ETHOS (kmbandy, do not re-litigate)

River rule: waves never stop flowing; accounting is **gates that bias the flow, never a dam**. **Adaptive,
never a static cap. A limiter, never a designator. A bail means the mechanism is broken.** The funnel
honored all of these and still lost — because it was attacking a *symptom* (the boundary storm) downstream
of the real wall (the frontier advance rate). The next attack must be on the frontier rate itself.
