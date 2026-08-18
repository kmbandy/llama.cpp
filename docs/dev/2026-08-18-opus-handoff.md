# Opus handoff — 2026-08-18 morning (DS4 sliced rig)

Fable wrote this after reconstructing the 08-17 session and diagnosing the
08-18 morning failure. Read it END TO END before touching anything. The
morning Opus session failed because it guessed at paths — this doc exists so
nothing gets guessed.

## 0. TL;DR

The DS4 sliced rig works and serves at **3.47 t/s** (config of record, NLL
gate passed bit-identical). It is DOWN right now because the **router-driven
bring-up path is broken**: `ds4-stackd` still launches workers from the OLD
whole-expert harness, whose data was **deleted from main's NVMe**. The #1 task
today is repointing that harness at the sliced sets so the elastic
router→stackd chain works again. Everything else (perf levers) comes after.

## 1. AUTHORITATIVE PATHS — use these, nothing else

**NEVER serve from `/mnt/hdd` on main. It is a spinning-disk ARCHIVE.**
`/mnt/hdd/models/DeepSeek-V4-Flash-0731-DSpark-Q8_0.gguf` is an archived
draft-model copy — it is NOT part of the serving rig. The morning session
tried to use an HDD gguf; do not repeat that. Everything the rig serves lives
on NVMe (`/home` on main is nvme0n1p2; `/mnt/nvme` on 2026).

| Thing | Path | Box |
|---|---|---|
| Spine dense gguf (the ONLY model file the router loads) | `/home/kmbandy/models/DS4-Flash-dense/ds4-dense.gguf` (9.8 GB) | main |
| slice0 (R9700, width 1408, layers 0-42) | `/home/kmbandy/models/ds4-eslice/slice0/` (101 G) | main |
| slice1 (1070, width 320) | `/mnt/nvme/ds4-eslice/slice1/` (23 G) | 2026 |
| slice2 (RX480, width 320) | `/mnt/nvme/ds4-eslice/slice2/` (23 G) | 2026 |
| CPU DSpark d0 (layers 43-45, experts 0-84) | `/home/kmbandy/models/DS4-eshard-dspark/` | main |
| CPU DSpark d1 (layers 43-45, experts 85-255) | `/home/kmbandy/models/DS4-eshard-main-dspark/` | main |

Slice manifest/descriptor naming (per slice N):
`ds4-sN-experts-experts-manifest-dspark.json` +
`ds4-sN-experts-experts-manifest-dspark.expert-descriptor.json`.
Main also holds spare copies of slice1/slice2 under
`~/models/ds4-eslice/` — the serving copies for 1070/480 are the 2026 ones.

**DEAD / GONE:** the old whole-expert sets `~/models/DS4-eshard-main` and
`~/models/DS4-eshard-main-trunk` are **deleted from main**. The 2026 copies
(`/mnt/nvme/models/DS4-eshard*`) still exist but the whole-expert layout is
DEAD by kmbandy's directive — never launch it, never propose reverting to it.

Worker binaries: main `build-hip/bin/llama-wp-expert-worker` (HIP),
2026 `build-army-cachy/bin/llama-wp-expert-worker` (CUDA arch61 + Vulkan).
2026 builds compile 2026's OWN tree — rsync changed sources first or the
build silently no-ops.

## 2. THE RIG — what each piece is and does

Topology (HARD RULES, kmbandy, emphatic):
- **RX 6900XT = SPINE ONLY.** Dense model + KV. Never a slice/expert worker
  (measured −26% when tried). The 4:2:1:1 layout that put a slice on it was a
  mistake and is dead.
- Expert workers = exactly 3 GPUs + 2 CPU processes:

| Worker | Device | Port | Slots (v3 values) | Holds |
|---|---|---|---|---|
| s0 | R9700 (main, ROCm0) | 8801 | 3350 | slice0 w1408 |
| s1 | GTX 1070 (2026, CUDA0) | 8803 | 3750 | slice1 w320 |
| s2 | RX 480 (2026, Vulkan0) | 8804 | 3900 | slice2 w320 (needs `GGML_VK_HOST_VISIBLE_VIDMEM_MAX_BYTES=1048576`) |
| d0 | 3900X CPU (main) | 8807 | 85 | DSpark layers 43-45, experts 0-84 |
| d1 | 3900X CPU (main) | 8808 | 171 | DSpark layers 43-45, experts 85-255 |

Reshard geometry 1408:320:320 — every sliced layer's expert output is the SUM
of 3 partials, one per GPU worker. That is why bitwise output parity across
runs is structurally impossible; quality gates are **teacher-forced decode
NLL** (`~/ds4-runs/eslice-run4/nll_gate.py`), never output-text comparison,
never llama-perplexity.

Control plane:
- **llama-router** (`:8090`, systemd user unit `llama-router.service`) — the
  ONLY thing that spawns the spine. `[ds4-flash]` in
  `~/.config/llama-router/router-fleet-main.ini` (line ~170): spine on ROCm1
  (6900XT), `expert-dispatch = 127.0.0.1:8801,192.168.1.33:8803,192.168.1.33:8804,127.0.0.1:8807,127.0.0.1:8808`.
  Endpoints are localhost/raw-LAN deliberately (Tailscale routing was a
  measured −0.4 t/s; only ssh CONTROL goes over 100.x). The unit carries
  `WP_HIP_GRAPHS=1` (config of record) and `WP_DISPATCH_CONNECT_RETRY_S=180`.
- **ds4-stackd** (systemd user unit, runs `docs/dev/ds4-stackd.sh`) — watches
  for a router-spawned spine; on sight it launches
  `docs/dev/harness-2026-08-05-ds4_full.sh` with `WORKERS_ONLY=1` (2026
  workers raised over ssh), sourcing worker env from
  `~/ds4-runs/stackd-worker.env` (currently the glibc arena fix:
  `MALLOC_ARENA_MAX=2`, `MALLOC_TRIM_THRESHOLD_=131072`). Spine gone → 5-min
  grace → SIGTERM the harness → workers torn down.
- **`~/ds4-runs/eslice-run4/launch_sliced_dspark_v3.sh`** — the hand-launcher
  used during 08-17 development. It is the REFERENCE for correct sliced worker
  definitions (paths/ports/slots/env), but **hand-launching is BANNED**
  (standing rule, kmbandy repeated it 08-17): all serving goes through the
  router. Read it, port from it, don't run it.

## 3. WHAT BROKE THIS MORNING (08-18, 07:52–08:17)

Two spine attempts, both `failed to create context`:
- 08:07 attempt: died at 08:10:19 — exactly the 180 s
  `WP_DISPATCH_CONNECT_RETRY_S` window. Workers weren't accepting.
- 08:12 attempt: died in 6 s right as stackd declared workers ready.

Root cause: **`harness-2026-08-05-ds4_full.sh` still defines the OLD
whole-expert workers** (`ES_MAIN=/home/kmbandy/models/DS4-eshard-main`,
`ES_MAIN_TRUNK=...-main-trunk`, R9700 as "experts 85-255", 6-worker layout) —
see this morning's `~/ds4-runs/stackd-20260818-080714/launch.log`, which shows
the old-layout banner. Those main-side data dirs no longer exist, so the
"workers ready" stackd reported was hollow, and the spine could never
complete expert-dispatch HELLO. The sliced worker definitions were never
ported from the v3 launcher into the harness. (The harness also references a
dead scratchpad prompt file, `prose739.txt` — harmless for WORKERS_ONLY but
clean it up while in there.)

## 4. TASK #1 — repoint the stackd harness at the sliced layout

Edit `docs/dev/harness-2026-08-05-ds4_full.sh` so its `WORKERS_ONLY=1` path
launches exactly the 5 workers in the table above, copying the working
invocations from `launch_sliced_dspark_v3.sh` (lines ~57-90: shard-manifest +
descriptor + device + listen + slots per worker; 2026 pair over ssh with
`XDG_RUNTIME_DIR` set; RX480 gets the `GGML_VK_HOST_VISIBLE_VIDMEM_MAX_BYTES`
cap; keep `WP_WORKER_STATS=1`). Keep the harness's existing
port-wait/teardown/EXIT-trap structure — stackd depends on it. Worker env
knobs stay in `~/ds4-runs/stackd-worker.env`, not hardcoded.

Then test the elastic chain end-to-end (the 08-14 proven flow):
1. `board_claim` the GPUs first (R9700, 6900XT, 1070, RX480; use
   `vram_alert_pct=99`). ONE MODEL LOAD PER BOX while anything loads.
2. Confirm `llama-router` + `ds4-stackd` active; do NOT hand-start anything.
3. One chat request to `:8090` with `model=ds4-flash`. Router spawns spine →
   spine retries dispatch connect up to 180 s → stackd raises workers
   (~17 s) → serving. Cold-to-serving ≈ 60-90 s.
4. Gate: coherent output AND `nll_gate.py` vs its banked reference AND
   t/s ≥ ~3.4 (the 08-17 record is 3.47).
5. While serving, watch worker RSS (`ps -o rss= -p <pids>` over ~10 min) —
   the MALLOC_ARENA_MAX=2 fix was applied but never verified. Flat = fixed;
   climbing = real leak, hunt it.
6. When done: stop serving, release claims. NO idle serves, ever.

## 5. AFTER THAT — the 10 t/s levers (in order)

Measured state (08-17, KG ed6019b5): token ≈ 288 ms = ~50% `wait` + ~50%
`unpack` across 43 sequential per-layer dispatches. Pack/issue/graph-launch
< 2% — **stop optimizing the GPU graph**. Layer pipelining is architecturally
impossible (routing depends on the previous layer's residual). Known-dead:
SIMD unpack end-to-end effect is sub-noise (`WP_SIMD_UNPACK` exists,
default-off, leave it); WP_SPEC_CONST_WIDTH net-negative, stays off.

1. **f16 partials on the wire.** The 1 GbE legs to 2026 run at line rate and
   every sliced layer makes ALL workers respond. `WP_EXPERT_PARTIAL_DTYPE`
   exists but the workers still log `=f32` — finish/flip it, halve remote
   bytes. Gate with NLL.
2. **Encode-once-broadcast** (Grok review, KG e58d787f): per layer the
   3 GPU workers already get ONE request each, but byte-identical frames are
   encoded 3×. Encode once, send thrice.
3. **Skip the gather scan in slice mode** — `WP_DISPATCH_GATHER` is a no-op
   on sliced (every worker holds every expert) but still scans on n_tokens>1.
4. **Issue the fat slice first** (needs HELLO to carry slice width) so the
   R9700's w1408 leg starts earliest.
5. **spec-PREFETCH of worker expert pages** (part of `wait`): hash layers 0-2
   give exact experts free via tid2eid (~12.5% of page-ins), KG 999ac4b5.
6. **WP_CPU_THREADS sweep** (4/6/8/12) for the CPU DSpark workers, and
   512-aligned slice widths to kill the CUDA pad tax on w320 — bigger
   surgery, ask first.
7. `WP_DEFER_K` is an APPROXIMATION (fails NLL gates elsewhere) — only if
   kmbandy explicitly accepts the tradeoff. Do not propose casually.

Open question for kmbandy (don't decide unilaterally): the INI has
`spec-draft-conf-min = 0.4` / `n-max 7`, but the 08-17 sessions ran conf-min 0
and the gate-off re-baseline was the plan of record. Ask which is intended
before benchmarking.

## 6. STANDING RULES (all learned the hard way; do not test them)

- All serving via the ROUTER. Never hand-launch spine or workers.
- `board_claim` before any GPU work; release + tear down after gates.
- NO git state-changing commands in this tree (checkout/restore/reset/stash/
  clean); NO commits unless kmbandy asks. The ~5-day uncommitted tree is
  deliberate. Undo = reverse Edit only.
- Never restart live services or `daemon-reload` without confirming
  (router :8090, mneme :8810/:18800, dashboard :18810).
- Quality gates: teacher-forced NLL. Never output text, never perplexity.
- Standing NOs: whole-expert layout, 6900XT as worker, host victim tier,
  MTP-as-prefetch, 24 CPU threads on the DSpark workers.
- Recalled memories are DATED context — verify against the live system
  before acting (kmbandy standing correction, 08-17).
- Detached rigs launched from a Claude Bash tool die with the task's cgroup —
  systemd units only (KG 03b0a0a3).
- RX 480 on 2026 autosuspends and evicts VRAM to GTT — check
  `mem_info_vram_used` before trusting any measurement it took part in.

## 7. WHERE THE KNOWLEDGE LIVES

KG (mneme, self scope): 3d705348 (08-17 close-out brief), edf10d33 +
9a8be919 + 56684c43 (HIP-graph fix + gate), ed6019b5 (bottleneck profile),
a4f9525b (transport fix), 96b08b21 (CPU-thread +11%), f2e3c6b9 (topology hard
rule), e58d787f (Grok pager review = lever list), 09e9ba65 (arena fix),
89be060f / 1a833292 (slice data plane + sim). Qwen campaign state (separate,
untouched today): docs/dev/2026-08-16-opus-handoff.md §Campaign B.
