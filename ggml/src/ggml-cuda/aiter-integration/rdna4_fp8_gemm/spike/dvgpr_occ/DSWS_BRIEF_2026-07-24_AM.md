# DSWS S1 — Morning Pickup Brief (2026-07-24 AM)

**Read `DSWS_S1_STATUS_2026-07-23.md` first** (full status). This is the short "start here" brief.
Supersedes `DSWS_BRIEF_2026-07-23_EOD.md` for the next-action list (that brief's §8 dispatch command
was WRONG — it used build defsyms as dispatch env; see the gotcha below).

## Where we stopped

ADVPROBE is done and the frontier wall is understood: **~10% serial advance-mechanism (~264 ticks) +
~90% idle-waiting on the ASSIGN publish rate.** The advance mechanism is NOT the wall; the pipeline is
**starved upstream of the boundary**. This closes the boundary-election line for good (herd/funnel/read-spin
only touch the 10%). Card free, latch clear, tree clean (baseline `cac3ff7c` on disk), nothing staged.

## The next task (agreed with kmbandy — understand-before-acting, adaptivity-first)

**Read the ASSIGN-publish path, offline, before touching any lever.** The question:

> Why can't the ASSIGN frontier keep 30 lean waves fed? (100% of feed iters find an empty frontier.)

Concretely, in `occ_kernel_dsws_flow.s`:
1. Find where `ASSIGN_HEAD` advances (the CFASSIGN / DECENTASN publish site — search `ASSIGN_HEAD_OFF`
   stores, and the coordinator/completer path that bumps it). What must complete before it moves?
2. Is the publish serialized behind the 10 split-K reduction boundaries per super-tile (n_kseg=10 at
   SEGK=256, K=2560)? I.e. does ASSIGN only advance once a full K-reduction + C-store lands?
3. Is that cadence a **natural** limit (real dependency: can't assign the next group until this one's
   banks free) or an **artificial** one (a lock / single-publisher / conservative gate we can relax
   while staying river-safe)?

Deliverable: a short written map of the publish path + a judgment on natural-vs-artificial. Only THEN
design a change (and per ethos it must be a limiter/gate that biases flow, adaptive, never a dam/designator).

## The SEGK lever (candidate, do NOT build yet)

Brief-flagged: SEGK governs republish cadence. Bigger SEGK = fewer/larger reduction segments = frontier
advances in bigger steps, less often; smaller = more often, more overhead. Connects to the old
"FLUSH is the wall" profile and decision `9ccbf559` (SEGK/POOL_N trade at identical LDS). This is the
ASSIGN-bound attack neither herd nor funnel touched — but confirm the publish-path read first.

## How to run (canonical A1 flow dispatch — the gotcha that cost a full-stop)

`gpu_run.sh` **dispatch** env names ≠ `build_flow.sh` **build** defsym names. Use HOST names or the run
falls through to the default small oracle (768×512×512, FM=2, 8 waves) and wedges pre-live.

Build the bin (defsyms):
```
STAGGER=1 TFPROBE=1 STAGINSTR=1 BATONGATE=1 WAVES=30 SSWIN=32 FM=1 FN=4 SEGK=256 POOL_N=1 \
  G=6 ACC_N=3 CFASSIGN=1 DECENTASN=1 SELFSERVE=1 BATCH=1 [DSWS2_RCONV=1] [DSWS2_ADVPROBE=1] ./build_flow.sh
```
Dispatch the FULL ml8 gate_up shape (host names):
```
./gpu_run.sh <name> -- FLOW_WAVES=30 DSWS2_FLOW=1 DSWS2_FM=1 DSWS2_G=6 DSWS2_ACC_N=3 \
  FLOW_POOL_N=1 DSWS2_SEGK=256 DSWS2_K=2560 DSWS2_ORACLE_MTL=22 DSWS2_ORACLE_NTL=144 \
  DSWS2_TARGET_SECS=2 ML8_COOP_CHUNK=96 STAGINSTR=1 TFPROBE=1 ./occ_dispatch --dsws2
```
- `DSWS2_FLOW=1` REQUIRED (selects flow path). `MTL=22 NTL=144` → 2112×9216. `DSWS2_K=2560` → n_kseg=10.
- RCONV/ADVPROBE are baked into the bin, NOT dispatch env.
- Host GEOM (`FLOW_WAVES`/`DSWS2_FM`/`DSWS2_G`/`DSWS2_ACC_N`/`FLOW_POOL_N`/`DSWS2_SEGK`) MUST match the
  bin's build defsyms or C corrupts / it wedges.
- Full shape runs ~330 chunks; it will exceed a 180s foreground window — watch
  `~/dsws_gpu_logs/<name>_*.log`, not the terminal (tee flushes at close).

## Non-negotiables (unchanged)

- **GPU dispatch rules** (`CLAUDE.md` in this dir): ONE dispatch per greenlight, never a batch; a changed
  kernel gets ONE bring-up then STOP; hang/timeout/INCOMPLETE = FULL STOP, latch cleared only by a human;
  never raise `DEADMAN_TICKS`; nothing new in the hot path that touches the message bus / emits unthrottled
  stores; bandwidth-risky knobs get a small chunk first. `board_check` immediately before `board_claim`;
  release by claim_id. Never touch the GPU without asking + waiting. Dispatch only via `./gpu_run.sh`.
- **Shared tree:** stage NOTHING (`arg.cpp` / `router-fleet-main.ini` / `server-models.cpp` belong to the
  live weight-pager session). Only touch the spike dir. `DSWS2_*=0` MUST stay byte-identical to `cac3ff7c`.
- **No assembly in plans/specs** — Codex terra writes the assembly; terra must NEVER dispatch to the GPU.

## Pointers

- `DSWS_S1_STATUS_2026-07-23.md` — full status (proven/dead/open, instrument table).
- `HARNESS.md` §34 — canonical dispatch; §42 — GEOM-must-match rule.
- `DESIGN_BOUNDARY_FUNNEL_2026-07-23.md` — funnel design (closed dead end).
- KG facts (self scope): ADVPROBE result `0b752e8b`; dispatch-env gotcha `7ee10b03`.
- Green reference log: `~/dsws_gpu_logs/advprobe_big_180208.log` (full-shape ADVPROBE run).
