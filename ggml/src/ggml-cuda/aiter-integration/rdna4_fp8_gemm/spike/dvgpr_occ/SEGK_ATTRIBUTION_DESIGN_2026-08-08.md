# SEGK PER-SEGMENT COST ATTRIBUTION — DESIGN + PREREGISTRATION (2026-08-08)

**The question.** The per-event model (§85: `ns_per_event = 20.8 + 37.4×MF_per_event`) misses the SEGK
cells by +27% (segk128) and +53% (segk64). SEGK's own 3-point fit is
`time_per_rep ≈ 1.63 ms + 0.108 ms × n_kseg` (slopes agreeing to 1% over 4×). At the config of record
(n_kseg=36) the per-segment term is **~3.9 ms of the 5.5 ms rep — ~70% of runtime**. §87 established the
per-event fixed cost is LDS round-trip latency, not guard ALU. **What is the 0.108 ms/n_kseg made of?**
This is the #1 open item in `DSWS_BRIEF_2026-08-01_AM.md` §3 and it prices the exact quantity the
architectural fork (round-trip reduction vs register residency) hinges on.

**The method: slope decomposition by one-variable ablation, NOT timing probes.**
For each candidate per-segment component, an ablation arm removes (or renders inert) ONLY that component.
Each arm is measured at the three sanctioned SEGK points {256, 128, 64} (n_kseg 36/72/144), fixed reps,
base geometry otherwise at the config of record. Re-fit `time = a + b×n_kseg` per arm: **the drop in the
slope `b` vs baseline attributes that component's per-segment share.** Intercept moves are reported but
not the target. No RTC probes anywhere — PHASEPROBE is banned (brick vector, §85), and the §"NIGHT"
probe-scaling rule applies: no instrument whose emission rate scales with the population under test.

## 1. Candidate components (each needs mechanism confirmed IN SOURCE before an arm is built)

Codex: read the source and confirm each mechanism + per-segment multiplicity before implementing.
Do not trust this list's arithmetic — verify against `occ_kernel_dsws_flow.s` at the config-of-record
defsyms and correct the spec's counts in your report where they are wrong.

- **C1 — dyn-VGPR grow/shrink round-trip.** Under the duty-cycle design a wave grows to fat, computes ONE
  segment burst, flushes, shrinks — an `s_alloc_vgpr` pair per segment burst. The 07-21 note measured the
  grow/shrink round-trip at ~40% of wave time on the OLD geometry; never re-measured at the modern config.
  Candidate switch: a build arm that pre-grows once and stays fat for the whole chunk (duty violation AS AN
  INSTRUMENT ONLY — see §3 constraints), or an existing static-VGPR build path if one exists (the
  2026-06-20 "dyn==static at full-K" measurement used one; find it).
- **C2 — bank flush (`ds_add` reduce) per segment.** Each segment's partial goes to the LDS bank:
  ds_add_rtn/ds_add + wait per ACC frag per segment. Existing switch: `NODSADD=1` (verify it bites on the
  banked path — NOCFLUSH was inert on it once, §2026-07-20 lesson). ORACLE-INVALID arm (wrong C by
  construction): must still be WORK-EXACT and is used ONLY for the slope, never for a TF claim.
- **C3 — operand staging quantum.** A/B staging work that repeats per segment (BSTAGE/ASTAGE ds stores +
  waits + `s_wait` watermarks). Existing switch: `NOBLOAD=1` (B loads become no-ops; −2.0% at base §N3 —
  but its SEGK slope has never been measured; a flat-slope result here is itself attribution evidence).
- **C4 — reservation/claim per segment.** One reserve (peek ~5 `lds_get` + CAS) per rowblk-segment.
  §88 measured the CFASSIGN alternative slower at WAVES=6, but that swaps the mechanism, not the count.
  If no clean removal switch exists, this component is estimated by residual: slope − (C1+C2+C3 shares).
  Say so explicitly rather than inventing a risky switch.
- **C5 — boundary/group transitions.** DA_ZDONE field width scales with n_kseg (more transitions per
  group). BNDPROBE (occ[104..126]) already counts transitions EXACTLY and is unthrottled/cheap — use its
  counts to normalize, no new instrument needed.

## 2. Deliverables (offline only — Codex does NOT dispatch)

1. **The switch inventory**: for C1–C5, the existing defsym/env that isolates it (file:line), or the new
   defsym-gated arm you built (default 0, byte-identical off — same discipline as the guard ablation),
   or a written justification why that component cannot be isolated and must be residual.
2. **New arm(s)** in `occ_kernel_dsws_flow.s` + `build_flow.sh` plumbing where needed (C1 is the likely
   build; keep it minimal — one symbol, §86a coupling rule).
3. **Free-SGPR audit refresh** if any new counter is needed: disassemble the CURRENT config-of-record bin
   (`llvm-objdump -d --mcpu=gfx1201`), collect emitted sNN/s[a:b]; do NOT reuse the 07-21 free list
   (s54/s55 went live since, per the 07-30 brief §1).
4. **`SEGK_ATTRIBUTION_REPORT_2026-08-08.md`**: per-arm — mechanism confirmed at file:line, what the arm
   removes, predicted slope effect (pre-registered BEFORE silicon), byte-identity/assembly proofs (shas,
   0 spills, LDS unchanged unless the arm inherently changes it — then say so), and the exact 9-cell run
   matrix (3 arms you deem highest-value × 3 SEGK points, plus baseline 3) for the silicon session with
   env lines ready to paste (SSWIN=32 host env, DSWS_ALLOW_NONSTD naming, fixed DSWS2_REPS per §87 note).
5. **`segk_fit.py`**: tiny analysis script — ingest the gpu_run logs, extract span/computed per §88's
   derivation rules (never scrape rendered TF; refuse non-WORK-EXACT rows), fit per-arm slopes, emit the
   decomposition table.

## 3. Constraints (non-negotiable)

- **The duty-cycle invariant stands** (assembler DUTYGUARD, J*SEGK ≤ 256). C1's stay-fat arm, if built,
  is an ablation INSTRUMENT that deliberately violates the duty model to price the grow/shrink pair —
  it must be its own defsym, documented as measurement-only, never a config candidate, and it must NOT
  touch JDEPTH/SEGK or the DUTYGUARD itself. If it cannot be built without weakening DUTYGUARD, stop and
  report; kmbandy's sign-off gate on that guard is the design, not an obstacle.
- SEGK values only from the sanctioned {64,128,256}; geometry otherwise pinned to the config of record.
- No GPU dispatch, no gpu_run.sh, no occ_dispatch execution (DSWS2_DRYRUN config checks are allowed —
  contractually GPU-free, occ_dispatch.cpp:7430). Claude runs the silicon session under a board claim.
- Touch only `spike/dvgpr_occ/` files; no commits; don't append to DSWS_TESTING_LOG.md.
- Wrong-C arms (NODSADD-class): label ORACLE-INVALID in every table; WORK-EXACT gate still applies.
- Verify every ablation switch BITES (instruction census delta at the defsym) before listing it — an
  inert switch burned a run once (§2026-07-20).

## 4. Pre-registered outcome space (write predictions in the report before silicon)

- If C1 (grow/shrink) owns the slope: the fork discussion changes shape — the per-segment cost is the
  price of the moat's brief-peak discipline itself, and the trade becomes explicit and quantified.
- If C2/C3 (flush/staging LDS traffic) own it: round-trip reduction WITHIN the invariant is the lever
  (batching waits, wider stores), and register-residency arguments gain no new ammunition here.
- If nothing moves the slope: the per-segment cost is in the un-ablatable coordination fabric — that is
  the counter-free-fabric datum for the fork, stated as such.
