# DSWS — Targets until Fable usage resets (written 2026-08-09, post-§90)

**Where we are:** SEGK=512 measured **3.59 ms/rep, 26.9 TF (+55%)**, model-exact (predicted 3.61).
The per-segment flush owns ~95% of the slope (§89); larger J×SEGK product = fewer flushes = the lever.
Deep-J is SHELVED (sound POOL_N=1 impossibility, see DEEPJ_SELFSERVE_DESIGN §"Design gate"); the SEGK
ladder is the road. Config of record is still SEGK=256 — every product>256 cell runs under kmbandy's
DUTY_OVERRIDE authorization, which stands for **measurement cells only**.

**Read order for a fresh session:** DSWS_TESTING_LOG.md §89–§90 → this file → DEEPJ_SELFSERVE_DESIGN_2026-08-09.md.

---

## A. Standing gates (non-negotiable, every session — Opus: read this box twice)

- board_check (READ cpu.load AND memory.percent, not just VRAM) → board_claim before ANY GPU work; a
  probationary grant is not a claim; release with result.
- `gpu_run.sh` is the ONLY dispatch path. Never DEADMAN=0, never raise DEADMAN_TICKS, never PHASEPROBE,
  never `--gl2c`. A hang latches = full stop, human clears.
- Build before every row; bare `./build_flow.sh` must produce `58e965a46f3e162d` byte-identical — verify
  after ANY kernel edit before believing anything else.
- Every new/changed bin: ONE full-stride bring-up (chunk=64, REPS=1, ORACLE_STRIDE=1, 320/320 bad=0)
  before any perf number. WORK-EXACT is necessary, never sufficient — the oracle is the gate.
- New mechanism arms additionally need **proof of engagement**: disassembly evidence (addresses) that the
  LIVE path reaches the new code. §90's placebo lesson.
- Perf rows: fixed DSWS2_REPS (never TARGET_SECS), SSWIN=32 in host env, stride-8 oracle, same-session
  A/B only, span-ticks/computed for fits (never rendered TF). Env contract template: §89 / the §87 rows.
- Codex handoffs: coding only, NEVER invokes occ_dispatch (any form, DRYRUN included), never GPU. All
  dispatches + DRYRUN checks belong to the interactive Claude session. DS4 overnight: same rule, NO GPU.
- Commit NOTHING unless kmbandy asks. The tree carries other sessions' WIP.

---

## B. Silicon targets (day sessions, in value order)

1. **SEGK=1024 (KSEG_STEPS=64).** The big one. Extend both kernel shift-ladders (+ `==64 → shift 6`) +
   occ_dispatch whitelist + host rebuild (mirror the 512 pattern, 2-line diff each). Prediction:
   rep ≈ 1.644 + 9×0.1091 ≈ **2.63 ms ≈ 37 TF** — the measured flush-free floor. Look for: byte-identity,
   full-stride oracle, whether the linear model still holds at n_kseg=9, and WHAT breaks if it breaks
   (feed watermark? operand staging quantum? boundary handling at few-segment fields?). If clean, try
   SEGK=2048 (KSEG_STEPS=128, n_kseg=4.5 — NON-INTEGER: expect a structural refusal; K=9216 needs
   n_kseg integral. 2048 gives 4.5 → likely .error; SEGK=1152? not power-of-2×16 — the ladder ends at
   1024 for K=9216. Note this in the log when confirmed.)
2. **Repeat + widen SEGK=512/1024 evidence before any config-of-record talk:** second same-session
   repro of the 512 cell; the fm1 control cell at 512 (refit per-event model: does b0 hold ~20 ns?);
   M=8192 (HEAD amortization: base showed +6% there); 2-3 real ml8/mlambaformer shapes — does +55%
   transfer to the shapes that matter? Look for: shape-dependent regressions, oracle cleanliness at
   every geometry, clock behavior (fixed reps sized for ~0.7 s busy).
3. **The duty ledger (the other half of the invariant, still unmeasured).** kmbandy's standing question:
   what does the longer peak actually cost? Test at 2 WG/CU (FLOW_WAVES=12/ML8_POOL=128-class cells)
   where VGPR pressure is real: SEGK 256 vs 512 at matched work. Look for: occupancy loss, grow-fail>0
   (first ever?), throughput crossover. This is the data the promote-512 decision needs.
4. **WAVES right-size sweep (3–8) at SEGK=512.** The 20 ns/event fixed term survived guard-removal (§87)
   and wait-fusion (§89 BATCHLDS null) — polling cadence is the standing hypothesis. Clock-normalized
   (§81 instrument; raw TF meaningless across WAVES cells). Look for: b0 movement with poller count;
   coast/computed ratio; the §85 waves16 −14% reproducing at the new geometry.
5. **HEAD cost decomposition (19.9% of wave time, contents unknown).** TRACE=1 per-super-tile claimer
   timeline at config of record + M=64 MoE shapes (0.09–0.61 TF class). Look for: ramp vs drain vs
   steady split; what amortizes with M and what doesn't. Feeds the adaptivity story.
6. **BATCH cursor sweep (2, 4)** — amortizes the claim CAS. Post-§89 the claim residual is only ~5% of
   the slope, so expect small; cheap to run alongside 4. Look for: any interaction with SEGK=512.
7. **Board hygiene items needing kmbandy sign-off:** update gpu_run.sh CONFIG-OF-RECORD block
   (still enforces retired 16/128 → forces DSWS_ALLOW_NONSTD on every run); the DUTY_KMAX=512-or-1024
   promotion decision once (2) and (3) are in; refresh DSWS_STATE.md (stale since 07-17).

## C. DS4 overnight targets (offline ONLY: analysis, design docs, kernel arms default-0 — no GPU, no occ_dispatch)

1. **KSEG_STEPS=64 extension patch** (kernel ladders + whitelist + host rebuild) ready for the morning
   silicon session, byte-identity proven offline (sha table like the 512 round). Same for the n_kseg
   divisibility analysis at K=9216 (which SEGK values are legal above 1024, if any).
2. **Feed-path readiness at large SEGK:** with 18→9 segments the per-segment operand burst doubles in K;
   audit KDBUF_LPT watermark, bcnt waits, L2 reuse at KSEG_STEPS=32/64. Design (don't build)
   `global_load_tr_b128` (124→62 B-loads, proven on occ_kernel_btr128.s) as the follow-on if feed
   becomes binding. Look for: static evidence the feed pipeline stalls grow with segment depth.
3. **Flush restructure design study:** can the banked split-K reduce become wide stores + a final reduce
   (exclusive bank ownership per wave/segment-range) instead of 128 ds_add_f32? Price it against the
   measured 0.1034 ms/segment; state LDS budget implications. Paper design + pre-registered predictions.
4. **Frontier-protocol redesign sketch (deep-J revival, contingency only):** per-window ownership record
   or POOL_N>1 slot ring — what would stage/drain completion rules look like? Only worth building if the
   SEGK ladder tops out above the flush-free floor. Deliverable: design doc with the §"Design gate"
   constraints as the spec, no code.
5. **NODSADD intercept anomaly (§89):** fit intercept rose 1.64→2.44 ms when the flush was removed.
   Waves that don't wait on flush change coast dynamics how? Analyze from existing logs + counters;
   propose the counter that would settle it.
6. **Duty-ledger experiment design for B.3:** exact cell list, occupancy instrument, pre-registered
   predictions for what the moat buys at 2 WG/CU. DS4 designs, Claude flies.
7. **Consolidation docs:** a fresh DSWS_BRIEF morning doc superseding 08-01 (§85–§90 arc, current lever
   board, config-of-record status + the DUTY_OVERRIDE authorization scope); DSWS_STATE.md refresh;
   a losers' ledger (LEANGUARD/GUARDHOIST/LEANMARSH/CFASSIGN/BATCHLDS arms: keep-as-instrument vs
   delete candidates — recommend, kmbandy decides).
8. **Integration path study:** what would it take to mount the DSWS kernel behind llama.cpp's mul_mat
   for the router/serve path (the aiter-integration target)? Dispatch interface, shape coverage gaps
   (the M=64 MoE floor is the blocker §B.5 informs), fallback strategy. Analysis only.
9. **Fleet housekeeping if cycles remain:** mneme-code indexer blank-row bug (flagged 08-08); the
   handoff_complete MCP tool unavailable in Codex sessions (every handoff notes it).

## D. What NOT to do (standing decisions, do not relitigate)

- No register-resident/hipBLASLt-inversion pivot (kmbandy, 08-08: "if it kills the whole point of dsws,
  we're not doing it").
- CFASSIGN stays 0 (§88, −4.5% at modern geometry). Instruction-economy cuts in coordination code are
  dead (four strikes). WIDESLOT not pursued (§89 rule-gated).
- No deep-J building without the C.4 design first — the POOL_N=1 gate is accepted architecture now.
- DUTY_KMAX default stays 256 in-tree; override is per-measurement-cell, explicitly named in run logs,
  until kmbandy promotes a new config of record.
