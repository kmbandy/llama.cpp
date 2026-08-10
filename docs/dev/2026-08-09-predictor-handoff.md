# §4 Predictor — continuation orders (for Opus, until Fable usage resets)

Written 2026-08-09 ~09:50 by Fable at kmbandy's request. Read alongside
`docs/dev/2026-08-09-runs.txt` (day-4 ledger, **including the RETRACTION
section — the closure verdict was retracted, read it first**).

> **STATUS UPDATE (10:30, Fable's last act):** step 1 below is DONE — the
> capture writer is fixed and verified. The serve shortcut `ds4-serve`
> (fish function, `~/.config/fish/functions/ds4-serve.fish`) launches the
> config of record with captures armed into `~/ds4-runs/<arm>/`; the 08-08
> dataset moved to `~/ds4-runs/2026-08-08-serve/` (predcap GOOD 47 GB,
> draftcap BROKEN — kmbandy may delete draftcap.bin; derived npz tables
> preserved in `derived/`). Start at step 1's "remaining work", then step 2.

## Where this stands, in four sentences

The pre-registered §4 falsifier (beat router-k=2's 28.3% first-non-resident
recall on a simulated miss stream at M∈{8,16}/layer, or close the direction)
is **built, debugged end-to-end, and has NOT yet tested the predictor**,
because the draft-embedding capture turned out to be corrupted — effective
rank 2 across the whole 2.24 GB file. Every instrument was exonerated by
canary: the solver reproduces the n-gram through the identical code path
(0.500 vs 0.485 rare-recall), the pool sim replicates the 08-07 baseline
out-of-sample (29.6–30.8% @ M=8), and the labels/join carry real structure
(same-token Jaccard 0.253 vs 0.058 random). A codex/luna handoff
(**task 34620fd2**) is fixing the capture writer. The one genuinely new
result that survives: **token identity alone recalls 45.6–49.7% of the miss
stream at M=16** (n-gram, task-split, in-domain DSWS content) — better than
router-k=2 on its own turf.

## Standing rules that bind you (do not relitigate)

- Safety floor: never run inference / touch a GPU without asking kmbandy or
  holding a board claim (board_check IMMEDIATELY before every board_claim).
  Never restart live services; never `systemctl --user daemon-reload`.
  Router :8090 on main and everything on 2026 ports 8082/8093/18800/18810
  are untouchable.
- Campaign NOs for DS4: no expert re-shard, no top-k reduction, no pruning,
  no KV/weight quant changes.
- Don't write C++ yourself — hand fixes to codex luna, then YOU build both
  `build-cpu` and `build-hip` and run the tests (build-hip is the deployment
  target; luna validating only on build-cpu is structurally blind to
  weak-symbol/backend traps).
- Repo (`~/GitHub/llama.cpp`, branch `featire-wp-improvements`) is dirty
  with the whole campaign. No commits unless kmbandy asks; stage by explicit
  path only if he does.
- Metric discipline: the falsifier metric is miss-stream recall at fixed
  M/layer. Never headline rank-1 precision or union recall (produced one
  retracted verdict on 08-07 already).

## NEW pre-registered gate (added after the rank-2 burn — mandatory)

**Input-sanity gate, run BEFORE any training on any capture:**
1. Stale-duplicate rate: <5% of blocks sharing identical leading 512 floats
   (measured: 67% on the broken capture).
2. Effective rank (99% variance, centered, ≥4k random rows) > 100
   (measured: 2).
3. No truncation signature: cos(row0,row1) for multi-token blocks must not
   sit at 0.50±0.01 systematically.
The diagnostic one-liners are in the ledger's retraction section and the
scripts below. A capture that fails any of the three is not evidence about
anything except the writer.

## The pipeline (all working, ~25 min end-to-end on main, CPU only)

Scripts: `docs/dev/falsifier-2026-08-09/` — run in this order with paths
edited at the top of each: `scan_captures.py` (inventory + format check) →
`index_captures.py` (metadata/token index → capture_index.npz) →
`prepare_dataset.py` (step/task tables + draft↔step join → falsifier_dataset.npz)
→ `train_probes.py` (ridge probes, torch) → `falsify.py` (pool sim + verdict
table; `falsify2.py` variant in the session scratchpad shows the live-slice
form). Session scratchpad with the .npz files and canary scripts:
`/tmp/claude-1000/-home-kmbandy-GitHub-bg2-rice/50850f84-92f6-4f9d-ae53-f552f68f2303/scratchpad/`
(may not survive a reboot; everything regenerates from the captures).

Hard-won environment facts:
- **Handoff completion notifications require the session launch flag**
  `claude --dangerously-load-development-channels server:mad-lab-handoff`
  — without it the mad-lab-handoff channel (codex results, board alerts,
  queue turns) never surfaces in-session and you must poll
  `handoff_check(task_id)` instead. Ask kmbandy to launch your session
  with that flag. Also: confirm every codex_handoff reached
  `in_progress` with a non-null agent_session right after dispatch (a
  returned task_id only means the row was created), and never pass
  `agent_session` to resume — send fresh self-contained handoffs.
- System numpy on main links REFERENCE cblas (5 Gflops). All linalg goes
  through **torch CPU** (753 Gflops) with
  `HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' CUDA_VISIBLE_DEVICES=''`
  set BEFORE import (keeps the ROCm build off the GPUs; the
  "register fat binary failed" spam is harmless with them set).
- Bash-tool 10-min timeout applies even after backgrounding: `nohup ... &`
  for anything long, monitor the log.
- Old captures preserved: `~/ds4-runs/2026-08-08-serve/` (predcap 50.4 GB GOOD,
  draftcap 2.24 GB BROKEN — keep as the negative control for the sanity gate).
- 2026 is reachable via wg `100.124.155.84`, NOT via its Tailscale addr.
- pkill: always bracket-trick patterns (`'llama-wp-expert-worke[r]'`).

## Ordered next steps

1. ~~Collect the luna capture-fix result~~ **DONE 10:20 08-09, VERIFIED.**
   Luna's fix landed and Fable independently reproduced the sanity numbers
   on its smoke capture (`/var/tmp/wp-draft-34620q.bin`): dup-rate 0.000%,
   effective rank 146, no truncation signature, median row cos 0.919.
   Real root cause: `llama_get_embeddings_nextn` for DSpark returns a
   SCALAR CONFIDENCE broadcast across 4096 columns — never was an
   embedding; the fix taps the pre-normalization collapsed DSpark hidden
   state, **4096-dim, header updated**. Both builds green; the only test
   failure is the documented pre-existing speed-split expectation in
   test-wp-expert-dispatcher (2:1 vs 0.5) — not luna's.
   **YOUR REMAINING WORK ON THIS ITEM:**
   - Update `N_EMBD_D = 16384` → read-from-header (or 4096) in
     `prepare_dataset.py` / `train_probes.py` / `falsify.py` before the
     next run. New captures are ~16 KB/token — 4× smaller.
   - **Scope review before any serve banks numbers on this binary**
     (2026-08-01 lesson): luna also touched `src/models/dflash.cpp`,
     `deepseek4.cpp`, `llama-context.{cpp,h}`, `llama-ext.h` (the
     hidden-state tap — expected) plus several tests +
     `wp-expert-dispatcher/main.cpp` ("stale dirty-tree interface" fixes).
     Read those diffs (mtimes 09:25–10:20 on 08-09), confirm the model-side
     tap is capture-gated (zero cost when env unset), and ideally rerun one
     code700 bench pair against the previous binary's 3.83/16 numbers to
     prove no regression before a long serve.
2. **New capture ride-along** (needs kmbandy: serve = board claims + his
   content). Ask him for TWO content domains if possible — another DSWS-like
   code session AND something different (prose/chat/other repo). Capture
   costs ~0.35 t/s decode; it is not a banking run. Both env vars:
   `WP_DRAFT_CAPTURE` + `WP_PREDICT_CAPTURE` (the predcap side is verified
   good — DO NOT let anyone "fix" pipe-expert-dispatch-graph.cpp).
3. **Run the input-sanity gate** on the new draftcap. Fail → back to luna,
   do not train.
4. **Rerun the falsifier verbatim** (task-split, M∈{8,16}, same pools).
   Read the verdict against BOTH bars: the pre-registered 28.3%, and the
   n-gram's number on the same table — the probe must beat the **n-gram**
   to justify existing, since the n-gram is free (no GEMM, no capture
   dependency, no scorer thread). Probe ≥ n-gram + 5pts at both M → §4
   earns integration design. Probe ≤ n-gram → close §4 for real and write
   the closure into the KG citing both this doc and the ledger.
5. **Cross-domain n-gram test (independent of luna/draftcap!)** — the
   n-gram needs only predcap + WTB1 token ids, which are trustworthy. With
   any new capture from a DIFFERENT content domain: build the table on the
   old 72 DSWS tasks, evaluate miss-stream recall on the new domain (and
   vice versa). **This is the single highest-information/lowest-cost test
   available right now.** If cross-domain M=16 recall holds ≥~40%, propose
   exactly ONE live arm to kmbandy: table-lookup hints at batch-submit
   (PREDICTED provenance, SPEC_HOST=1 landing, volume-gated M≤16, prefill
   gated off — the queue/dedup/landing infra all exists from 08-07). If it
   collapses (<15%), note it and don't integrate.
6. **If both survive**: probe-vs-ngram-vs-hybrid economics, THEN one live
   arm max, kmbandy approves first. The 08-07 lesson stands: offline recall
   converted to zero live throughput because router confidence ≈ LRU
   residency; the n-gram's edge (if real) is that it ranks the TAIL, which
   is exactly what LRU can't hold. Watch landed/promoted/wasted counters,
   not just decode t/s.

## Context that will save you a day

- predcap structure: `T<w>` (WTB1 batch token ids) then `W0..W42` per
  verify step; `W43..45` = DSpark draft layers (feed the R9700 pool in the
  sim, never scored; their CPU worker is always-resident). Verify width w =
  n_drafted+1; position 0 = committed token (no draft embedding exists for
  it — n-gram covers it, probe can't).
- Pool sim: R9700 2200 slots (layer 0..45 × expert 85..255), 2026 merged
  1100 (layer 0..42 × expert 0..84), LRU, warmup 300 steps excluded,
  ~80% residency, ~106 misses/step.
- Splits: task%5==3 val, ==4 test, rest train; plus last-15-tasks tail as
  the second split. 72 tasks in the old capture.
- The "8th negative-guard recurrence": kmbandy's "are you sure?" unwound a
  shipped closure within the hour. Before closing ANYTHING, prove the
  instrument engaged (sanity gates, canaries through the identical code
  path). It is now a pattern, not an anecdote.

## KG anchors

- Retraction decision (supersedes the closure): written 2026-08-09 after
  4fe2b132 was invalidated — search `mneme_search("§4 falsifier retraction
  rank-2")`.
- Day-3 brief: 256e4c0c. Ledgers: 2026-08-08-runs.txt, 2026-08-09-runs.txt.
