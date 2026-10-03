# Multi-machine llama-router — design

Date: 2026-10-03 · Status: approved design, pre-plan · Owner: router/handoff session
Related: KG decisions aef2e7d8 (single multi-machine router), 582b48a6 (full-eviction swaps);
DS4.1 session draft 73a8a1ee; worker branch `wp/worker-stop-snapshot`.

## 1. Goal

Replace the two independent per-box routers (mad-lab-main :8090, mad-lab-2026 :8093) with **one
router** whose ledger, placement and eviction span both machines. The router is **mechanism
only**: it loads and unloads models on request, by priority. *When* to swap models (DS4.1
architect ↔ Qwen 3.8 27B builders) is decided by an outside orchestrator (spec 2, async
handoff), which this design must serve.

Problems this fixes:

- Evicting a weight-paging spine leaves its expert workers running (VRAM + RAM held, no owner).
- 2026's router uses a whole-card VRAM probe capped at 8192 MB and has no host-RAM accounting.
- The router ignores the coordination board: it can load onto a GPU a Claude session has claimed.
- Two routers cannot place one model group across both boxes.

Non-goals: the handoff orchestrator itself (spec 2); expert-worker internals (DS4.1 session);
router failover when the leader is down.

## 2. Decisions

| Topic | Decision |
|---|---|
| Topology | Leader router on mad-lab-main :8090 + `llama-server --router-node` daemon per remote box (2026 :8094). Main's node runs in-process. Rejected: ssh spawning (an ssh blip EOFs child stdin and kills it); Python control plane over two routers (keeps two eviction loops). |
| Machine registry | `~/.config/mad-lab-agents/machines.json` (existing, used by mcp/dash/console/tui). New optional key per machine: `router_node` (URL of that box's node daemon). No second registry. |
| Placement | Pinned aliases (`machine`, `gpu`); pools (`placement = any`, `replicas = N`); per-request `machine=` override. |
| Weight-paging eviction | **Full**: spine and all workers stop, VRAM + RAM tiers freed. Parking is an opt-in preset mode, never the default swap. |
| Warm resume | Spine: `--slot-autosave` (KV restore, no re-prefill). Workers: expert map written on SIGTERM, reseeded at start (VRAM hot set + LFU heat, then RAM tier). |
| Priority | Per-alias default, per-request override: `highest` / `middle` (default) / `lowest` (§6). |
| Swap timing | Outside the router. Orchestrator protects a model with a hold lease (§7). |
| Leader down | Everything offline. No node passthrough fallback. |

## 3. Components

### 3.1 Leader (`server-models.cpp`, extended)

- **Machines:** read `machines.json`; `local: true` → in-process node; else `router_node` URL.
  Remove the hard-coded child address `127.0.0.1`; a child's address is its node's host.
- **GPU slots** are named `<machine>/<dev>` (e.g. `mad-lab-main/ROCm0`). The `gpus=` global
  key gains the machine prefix.
- **Ledger v2** (§4), **model groups** (§5), **admission + priority** (§6), **holds** (§7),
  **board client** (§8), **API** (§9).

### 3.2 Node (`tools/server/server-node.cpp`, new mode of the same binary)

- `POST /node/spawn {name, gen, args, env}` → `{pid, port}`; `POST /node/stop {name, timeout_s}`;
  `POST /node/signal {name, sig}`.
- `GET /node/state` → children (name, gen, pid, port, status), per-PID VRAM per device,
  per-PID RssAnon + RssShmem, MemAvailable, per-device sysfs used.
- `GET /node/events` → SSE: child state changes (relayed from the existing child state lines)
  + heartbeat.
- The node owns its children (they exit when the node exits). On node start, any process
  tagged with a stale router generation is killed or re-adopted by (name, gen).
- Bind: LAN/Tailscale interface only. Shared bearer token from a file readable by kmbandy only.

## 4. Ledger and admission inputs

**VRAM per slot:** `free = min(total − router_reservations − foreign_usage, total − sysfs_used)`.

- `foreign_usage`: per-PID from amdgpu `/proc/<pid>/fdinfo` `drm-memory-vram` (per pdev)
  for every PID not owned by the router. CUDA slots (2026 GTX 1070): NVML per-process, else
  whole-card.
- Group workers count their declared `vram-mb` while loading and their measured fdinfo once running.
- Drop the 8192 MB cap on 2026.

**RAM per machine:** `free = MemAvailable − headroom` (default 4096 MB, configurable).
Each model declares `ram-mb` (KV host tier + worker RAM tier). A load must fit **both**
VRAM on every slot it touches and RAM on every machine it touches.

## 5. Model groups (weight-paging models)

### 5.1 Preset shape

```
[dsv41]          machine=mad-lab-main gpu=ROCm1 depends=dsv41-w-main,dsv41-w-2026
                 stop-timeout=120 slot-autosave=<path>
[dsv41-w-main]   kind=external machine=mad-lab-main gpu=ROCm0 launch=<cmd>
                 vram-mb=31300 ram-mb=38000 park-file=<path>
[dsv41-w-2026]   kind=external machine=mad-lab-2026 gpu=ROCm1 launch=<cmd>
                 vram-mb=14200 ram-mb=21000 park-file=<path>
```

- `kind=external` sections are spawned by the node of their machine; they are not
  requestable on their own and exist only as members of a group.
- The router injects `WP_EXPERT_PARK_FILE=<park-file>` and `WP_EXPERT_SEED_FROM_PARK=1`.
- Optional `park-mode=opt-in` keeps the SIGUSR1/SIGUSR2 park path available, never used by
  default eviction.

### 5.2 Load sequence

1. Admission for the whole group footprint (all slots, all machines).
2. Start workers in parallel; they reseed from their expert map.
3. Start the spine once every worker port accepts connections; spine restores its slot autosave
   while workers keep seeding.
4. Group is `ready` when the spine is ready.

### 5.3 Unload sequence

1. Drain in-flight requests (configurable timeout; at `highest` priority, cut them).
2. Stop spine: it writes its autosave; force-kill after `stop-timeout` (120 s for dsv41;
   the 10 s default kills the save mid-write).
3. SIGTERM each worker: it quiesces (`WP_EXPERT_PARK_QUIESCE_MS`, default 30000), writes its
   map, exits 0. Bound: quiesce + 10 s, then SIGKILL. Expected log lines:
   `stop snapshot written: <path> (<N> rows)`, `stop snapshot FAILED: <reason>`,
   `stop snapshot: quiesce timeout after <ms> ms, snapshotting anyway`.
4. Confirm VRAM (fdinfo) and RAM (MemAvailable) are back.
5. Release board claims.

### 5.4 Failure handling

- **Worker start failure** = process exit, a `Hip error` line, or startup timeout (do not wait
  only for `listening`). Tear down anything already started; load fails.
- **Worker dies mid-serve** → stop the spine, group status `failed`; next request reloads.
- **Child environment** comes only from the preset and router-level env, never from request
  parameters. The node refuses to spawn if `TEMP`, `TMP` or `TMPDIR` is set to a non-directory
  (hipblaslt then fails at init with a misleading "out of memory").
- No orphans on any path (§3.2 stale-generation sweep).

## 6. Admission and priority

Input: alias, priority, optional machine override.

1. Resolve candidate slots: one for a pinned alias; every fitting slot (respecting `machine=`)
   for a pool.
2. If a needed slot is held by an active board claim not owned by the router:
   - `highest`: claim at head of queue (behind the current holder), `notify` the holder with an
     urgent yield request. **Never kills another session's process.**
   - `middle`: queue, `notify` the holder ("router waiting on gpu:X to load <alias>").
   - `lowest`: queue silently.
3. Choose victims among router residents. Idle first; busy residents only at `highest`.
   Pinned models and held models (§7) are never victims. Evicting a group member evicts the
   whole group.
4. Take board claims, evict, wait for fdinfo to confirm VRAM freed, load, keep claims until unload.

Fold-in fix: `choose_gpu_evictions` ignores `req_count` and can evict a busy child — fix in step 3.

## 7. Holds

`POST /models/hold {model, ttl_s, owner}` → lease id; `POST /models/release {lease}`;
holds are renewable. A held model is exempt from eviction and from the idle timeout. Used by the
orchestrator to keep DS4.1 resident while it plans between turns.

## 8. Board integration

### 8.1 Board-side changes (mad-lab-mcp + mneme)

- REST face on :18800: `POST /api/board/claim|release|renew|announce|notify`,
  `GET /api/board/state`. Thin wrappers over the same `mneme_store` calls; MCP tools unchanged.
- `priority` field on claims; `highest` sorts to head of the queue (still behind the holder).
- `renew` verb (claims currently cannot be updated); router claims use a short TTL (~10 min)
  and renew on a heartbeat, so a dead router's claims lapse on their own.
- `notify`: push a message to the holder's session through the existing VRAM-alert delivery path.
- Fix known defects (2026-08-04): a `machine` claim must collide with per-resource claims on that
  machine; `board_check` and `board_claim` must agree.

### 8.2 Router behavior

- Holder `llama-router`. One claim per GPU it loads onto (`gpu:<name>`) plus `ram` per machine
  touched; the note names the alias.
- Poll `/api/board/state` every few seconds. When anyone queues behind a GPU the router holds,
  unload idle residents there immediately. Pinned or held models keep the GPU until released.

## 9. Router API (consumed by spec 2)

- `POST /models/load {model, priority?, machine?}` → `202 {state: queued|loading|ready,
  queue_pos, blocked_by}`.
- `/v1/*` requests accept `priority` and `machine`.
- `POST /models/hold`, `POST /models/release` (§7).
- `GET /models/sse` adds events: `queued`, `blocked`, `loading`, `ready`, `evicting`, `unloaded`.
- `/models` ledger shows machine, slot, group, priority, holds, claim ids.

## 10. Phases

| # | Phase | Ships |
|---|---|---|
| 1 | Ledger v2 (local) | fdinfo/NVML per-PID VRAM, RAM accounting, busy-child eviction fix, env scrub. Deployed on both current routers. |
| 2 | Model groups (local) | `kind=external`, `depends`, launch/stop contract, failure detection, orphan sweep. dsv41 with main worker only. |
| 3 | Board (mcp + mneme) | REST face, priority, renew, notify, two defect fixes. Parallel to 1–2. |
| 4 | Router board client | Claims, queue+notify, yielding, priority, holds, new SSE events. |
| 5 | Node mode | `--router-node`, `router_node` in machines.json, prefixed slots, remote proxy, heartbeat. dsv41 as full cross-box group. |
| 6 | Pools + migration | `placement=any`, replicas, `machine=` override; 2026 presets under the leader; clients to :8090 (mad-lab-dash, madlab-console `settings.rs`, mneme `board_probe.py`, `ds41-serve`); per-arm override path for the DS4.1 bench harness (no whole-file ini overwrite); retire :8093. |

## 11. Testing

- C++ unit tests for ledger and admission with injected fdinfo / MemAvailable; admission is a
  pure decision function tested by table (priority × claims × holds × pins × busy).
- pytest for board changes.
- One live check per phase, under a board claim. Phase 2 is the first router-driven dsv41
  evict/reload: worker stop-snapshot lines, `seed-from-park` and RAM-phase lines, slot-autosave
  restore, decode ms/step during vs after seeding. Phase 6: full DS4.1 → 2× Qwen → DS4.1 cycle,
  timed against today's 3–5 min cold swap.

## 12. Coordination constraints

- Router code lives on branch `router/multi-machine` in worktree `~/GitHub/llama.cpp-router`.
- The DS4.1 bench harness overwrites the whole `router-fleet-main.ini`; no ini edits while the
  DS4.1 session reports a bench window open.
- Cross-box sync is git only (2026 pulls main via its `mainbox` remote).
- Never restart or daemon-reload live router services without asking kmbandy.
