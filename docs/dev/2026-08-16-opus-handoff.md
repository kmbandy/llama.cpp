# Opus handoff — 2026-08-16 evening (Fable usage exhausted)

Two campaigns in flight in this tree (branch `sync/upstream-2026-08-10`, ~4 days
of DELIBERATELY UNCOMMITTED work). Fable wrote this; Opus picks up until usage
resets. KG entries referenced by id are in claude__main's mneme.

## Safety floor (non-negotiable, all learned the hard way)
- `board_claim` every GPU before touching it (mad-lab-memory MCP); release + TEAR
  DOWN serves when gates finish — **no idle serves, ever** (kmbandy, twice; KG 30987aad).
- ONE MODEL LOAD PER BOX at a time, fleet-wide (host RAM during load OOM-killed
  kmbandy's desktop 08-16 15:29).
- NO git state-changing commands in this tree (checkout/restore/reset/stash/clean).
  NO commits unless kmbandy asks. Undo = reverse Edit only.
- 2026 builds: rsync changed sources to 2026's OWN tree first (it does not share
  main's checkout — an unsynced "build" exits 0 in seconds having compiled
  nothing), then capped `systemd-run --user --scope -p MemoryMax=8000M -j2`.
  Verify binary timestamps.
- One element under test per run; print gate values; teacher-forced NLL not
  output-text for quality gates on DS4 (see docs/dev/2026-07-31-design-expert-deferral.md §2.1).
- Do NOT re-propose: expert-index re-sharding of the OLD layout (standing no),
  host victim tier, deferral-on-DS4, MTP-as-prefetch. Read the KG before
  re-running any disabled feature.

## Campaign A — DS4 sliced experts ("divided experts", 4:2:1:1)

kmbandy killed the old whole-expert method. The sliced layout is the only DS4
path going forward. Everything below is DONE unless marked TODO.

### Done today
- Residency sim (real routing trace) — gate PASSED: NVMe critical path
  143→88 ms/tok (−38%), p99 250→167. Sim + numbers: `~/ds4-runs/eslice-sim/residency_sim.py`, KG 1a833292.
- Data plane fully staged, byte-exact verified (KG 89be060f):
  - full v2 sliced set: `~/models/ds4-eslice/` (46 shards, 157.4 GB, `--verify` PASS);
  - per-slice worker sets + descriptors: slice0 74G (R9700) + slice1 37G
    (6900XT) on main `~/models/ds4-eslice/slice{0,1}/`; slice2+3 37G
    (1070/RX480) on 2026 `/mnt/nvme/ds4-eslice/slice{2,3}/`.
    Descriptor files: `<base>-experts-manifest.expert-descriptor.json`.
- Consumer chain implemented by terra (codex handoffs b7129b9a + a514a5bc):
  `wp-expert-shard --slice N` extraction; slice-mode worker catalog + sliced
  FFN graphs (`tools/wp-expert-worker/wp-expert-worker.cpp`); dispatcher
  auto-detects slice rigs (workers advertise full expert range + `slice:`
  shard-identity marker) and broadcasts + sums partials
  (`src/pipeline/pipe-expert-dispatcher.cpp`). Unit tests green both boxes.
- CUDA slot-padding crash FIXED and rebuilt on both boxes: CUDA pads quantized
  tensors with ne0 % 512 != 0 (the width-256 `down` slice `[256,4096]`); slots
  were sized at exact bytes → memset past slot (crash) / into neighbor page
  (latent corruption). Sizing now derives from `ggml_backend_buft_get_alloc_size`
  with offset validation before every init_tensor (wp-expert-worker.cpp:1323,
  :1378, :3217, :1965, :3811; test at tests/test-wp-expert-worker.cpp:419).
- First live sliced boot SUCCEEDED post-fix: 4 workers + spine up in ~25 s.
  Launcher (reuse it): `~/ds4-runs/eslice-run1/launch_sliced.sh`
  (slots 3200/1200/2800/2800; spine 6900XT ROCm1; spec OFF for run 1).
  Gate script: `~/ds4-runs/eslice-run1/gate_sliced.py`.
- Unsliced reference oracle captured (spec-off, temp 0, 3 prompts × 128 tok,
  3.38–4.26 t/s): `~/ds4-runs/eslice-run1/ref-unsliced.json`.
- `ds4-stackd.service` (main) is STOPPED — it supervises the OLD layout and was
  respawning old workers over ssh, fighting the new rig for ports. Do not
  re-enable until its HARNESS config points at the sliced layout.

### Gate results (the two open problems)
1. **Throughput: sliced = 1.1–1.2 t/s vs unsliced 3.4–4.3 — 3× SLOWER, and it
   is NOT cold-start** (warm rerun: cap 1.130, code 1.196 t/s). The sim only
   modeled NVMe; the live loss is almost certainly the dispatch path: every
   layer now broadcasts to ALL 4 workers (4× request count), pays wait-on-
   slowest of 4 legs including two Tailscale hops, × 46 layers × per-request
   protocol cost (~69 ms/tok of protocol was already measured on the OLD rig
   with 3 workers, 2026-08-02). TODO #1 below.
2. **Parity gate as written is unreachable and needs replacing.** Outputs
   DIVERGED@0/1/15 but are fully coherent (arguably better than ref). Bitwise
   temp-0 parity cannot survive this layout: each expert output is now a sum of
   four partials computed on FOUR different backends (HIP/HIP/CUDA/Vulkan).
   Note the OLD rig was itself non-deterministic run-to-run (choose_worker
   timing → ~35% assignment variance → FP order; 2026-08-02 KG). Correct
   quality gates: (a) logit-gap probe at flip points (near-tie ⇒ numerics —
   method: replay generated prefix via /completion ending BEFORE the leading
   space of the next token, n_probs; see KG 8c579742 for the tokenization
   gotcha), (b) teacher-forced decode NLL sliced vs unsliced (the established
   DS4 instrument; NOT llama-perplexity — it is gated n_tokens==1 and measures
   an inactive feature; NOT output-text comparison), (c) terra's exact
   partial-sum unit test (already green).

### DS4 TODO list, in order
1. **Diagnose the 3× slowdown.** Bring the rig up (launch_sliced.sh), run a
   fixed prompt, read per-worker stats lines (`wp expert worker stats`, ns_wait/
   ns_prep/ns_submit/submit_hist) + spine `WP_DISPATCH_STATS` and decompose:
   protocol/request-count vs wait-on-slowest vs compute. Candidate fixes, in
   likely-value order: coalesce the per-layer broadcast into ONE request per
   worker per dispatch (if not already), concurrent issue/await across all 4
   sockets (nonblocking send + harvest-as-ready — Kimi K3's rec, numerics-
   preserving, attacks the measured ~69 ms/tok protocol floor), batch layers,
   pipeline activation upload with compute. The 6900XT leg shares the card with
   the spine — check contention. Do NOT propose moving expert ownership between
   machines (standing no); slice ownership is fixed by the built sets.
2. **Quality gate**: logit-gap probe + teacher-forced NLL (unsliced NLL needs
   one old-rig boot — harness `SPEC= ARM=nll` — or reuse an existing banked NLL
   if comparable). Bank the verdict before any tuning.
3. **DSpark arm** (after 1+2): SPEC=1 SPEC_CONF=0 on the sliced rig — this IS
   the gate-off DS4 re-baseline (old banked acceptance 1.76–2.44 was gate-
   censored; expect ungated ≥ that; KG 9103713b/8e22e94c for background).
4. 6900XT slice pool sizing: sim says every GB pays (3/6/9 GB → −19/−38/−48%
   NVMe cp); run-1 used a conservative 1200 slots (~4 GB). Raise after the
   slowdown is fixed and VRAM headroom is confirmed with spine+KV loaded.
5. Housekeeping once gates pass: delete the redundant full v2 set
   (`~/models/ds4-eslice/*.wpb`, 157 GB, main disk at 91%) — slice sets are the
   serving copies, everything re-derivable; repoint + re-enable ds4-stackd.

## Campaign B — Qwen3.8-27B BF16 TP stack: road to 50 t/s

Current record: DSpark n6 conf0 chat = 15.67 t/s mean / 22.2 peak on the 6-task
battery (MTP n3 = 15.94 mean, more robust; kmbandy DECIDED DSpark n6 is the
spec-type of record — do not re-litigate, KG 11c2c4a7). Stack is DOWN (no idle
serves). Launch = the radix-serve2 command in KG 8c579742 / ~/ds4-runs/qwen38-bf16/.
PIPE v12 is shipped + gated (wire halved, dspark ships zero nextn bytes; KG
7395aca8); its perf A/B is NOT yet banked (gates ran while a 157 GB repack
saturated the disk) — take a clean A/B first time the stack is up.

Levers, ranked by measured headroom:
1. **head_project_us ~5.9 ms per 4-tok block** (LM-head projection on the TP
   pair). Opus-agent diagnosis (KG 7395aca8): (a) FIRST check ggml-cuda.cu:2467
   — if the pair lacks BF16 MMA it F32-promotes and converts the ENTIRE 2.54 GB
   LM head per call, which would dominate everything; (b) the HBM floor for
   2.54 GB BF16 @ ~1.27 GB/device is 2.5–5 ms/CALL — so batch more tokens per
   projection (nearly free per call); (c) zero-risk first step: split the timer
   at src/llama-context.cpp:1395-1415 into build/compute/readback; (d) safe
   micro-fix: blocking strided D2H at :1414 → ggml_backend_tensor_get_async +
   one sched sync (meta backend implements get_tensor_async, ggml-backend-meta.cpp:1746);
   (e) sched churn: per-call sched_reset+alloc, never reserve; fresh split uid
   per call defeats meta graph reuse (ggml-backend.cpp:1653).
2. **Draft cost**: DSpark draft_us ≈ 16.2 ms/block steady-state (radix logs) —
   at 15.7 t/s that is ~25% of wall. Services round-trip top-K wire ~21 ms/call
   was flagged earlier; profile the draft step (WP_DSPARK_DEBUG exists).
3. **n8+ cliff**: decode flat ~5.2 t/s at n-max ≥ 8, suspiciously equal to
   --segment-rs-snapshots 8. Uninvestigated. If it is the snapshot ring, deeper
   verify becomes free and DSpark's flat-tail head gets more room.
4. **Prefill** is 1.8× worse per token than decode — dominant for real
   workloads, untouched.
5. Adaptive spec-type / calibrated conf gate (SGLang STS, arXiv:2607.05147) —
   parked, kmbandy has not approved; DSpark n6 fixed is the default.
6. Upstream PR backlog (not perf): mmvf rows-per-block, meta handle_view guard,
   M-RoPE OOB, 480 BAR allocator guard (ggml-vulkan.cpp:3512), Vulkan K-quant
   A-load hoist.

## Loose ends
- PIPE v12 residual: no live proof the nextn alias flag fires (unit tests prove
  the byte saving; acceptance identity proves correctness). A one-line tail-side
  log when nextn_aliased=1 closes it — trivial, do during any v12 touch.
- llama-graph.h:965 comment is stale (t_h_nextn is post-output_norm on qwen35).
- Board claims from tonight are RELEASED; re-claim before any GPU work.
- codex_handoff RESUME is still broken (exit 2) — always send fresh tasks.
