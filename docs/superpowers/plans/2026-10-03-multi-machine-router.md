# Multi-machine llama-router Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One llama-router (leader on mad-lab-main :8090 + node daemon on mad-lab-2026 :8094) that places, loads and evicts models — including weight-paging groups (spine + expert workers) — across both machines, by priority, while honoring the coordination board.

**Architecture:** Extend the existing fork router (`tools/server/server-models.cpp`) with a per-process VRAM + host-RAM ledger, model groups, a pure admission function, holds, and a board client; add a `--router-node` mode of the same binary that spawns/stops/reports children on remote boxes. The board (mneme + mad-lab-mcp on mad-lab-2026) gains priority, yield notifications and router-facing REST routes.

**Tech Stack:** C++17 (llama.cpp server, cpp-httplib, nlohmann::json), CMake/ctest; Python 3 (mneme daemon FastAPI, mad-lab-mcp FastMCP custom routes), pytest.

**Spec:** `docs/superpowers/specs/2026-10-03-multi-machine-router-design.md` (read it before any task — section numbers below refer to it).

## Plan rules (kmbandy, overrides the writing-plans default)

- **This plan contains no code.** Tasks give direction, file pointers, one-line interfaces, behavior rules, what the tests must assert, and a definition of done. The implementer writes all code and tests.
- **Implementers do not build or run live services.** They write code + tests and may run pure unit tests (pytest; C++ tests only if a build dir already exists and they're told so). The controller session builds (`build-hip` on main, the equivalent on 2026), runs ctest, and does every live check under a board claim.
- **Stop and report** instead of improvising when a pointer is wrong, a contract in this plan contradicts the code, or a test can't be written as described.
- Commit per task on the stated branch; message `router: <what>` / `board: <what>`; end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Global Constraints

- Router branch `router/multi-machine`, worktree `~/GitHub/llama.cpp-router` (mad-lab-main). Never touch `~/GitHub/llama.cpp` (DS4.1 session's tree, branch `wp/dspark-spec-sampling`).
- Board repos are canonical on **mad-lab-2026**: `~/GitHub/mneme` (`src/mneme/board.py`, `src/mneme/service/app.py`, `src/mneme/service/client.py`) and `~/GitHub/mad-lab-mcp` (`server.py`). The mad-lab-main checkouts are stale — do not edit them. Work in a branch `board/router-priority` on 2026.
- Machine registry = `~/.config/mad-lab-agents/machines.json` (`{name: {local?, ssh?, models_dir?, router_node?}}`). `router_node` is the only new key. No other registry.
- Do not edit `~/.config/llama-router/router-fleet-main.ini` while the DS4.1 session has a bench window open; new preset examples go in test fixtures, not the live ini, until Phase 6.
- Never restart/daemon-reload `llama-router.service`, `mad-lab-mcp`, or the mneme daemon without kmbandy's OK.
- Priority values exactly: `highest`, `middle` (default), `lowest`.
- Router board holder name exactly: `llama-router`.
- Worker stop contract (DS4.1 session, branch `wp/worker-stop-snapshot`): SIGTERM → quiesce (`WP_EXPERT_PARK_QUIESCE_MS`, default 30000) → write `WP_EXPERT_PARK_FILE` → exit 0. Log lines: `wp expert worker: stop snapshot written: <path> (<N> rows)`, `wp expert worker: stop snapshot FAILED: <reason>`, `wp expert worker: stop snapshot: quiesce timeout after <ms> ms, snapshotting anyway`. Router bound: quiesce + 10 s, then SIGKILL. Router always injects `WP_EXPERT_PARK_FILE=<park-file>` and `WP_EXPERT_SEED_FROM_PARK=1`.
- Child env comes only from preset + router env; never from request params. Refuse to spawn if `TEMP`/`TMP`/`TMPDIR` is set to a non-directory.

## File structure

New C++ units (each with one job, unit-testable without a GPU) in `tools/server/`:

| File | Responsibility |
|---|---|
| `server-router-probe.{h,cpp}` | Read per-PID VRAM (amdgpu fdinfo `drm-memory-vram`, per pdev), per-PID RssAnon/RssShmem, MemAvailable, sysfs used. All reads take an injectable root path (default `/`) so tests use a fake `/proc` + `/sys` tree. NVML per-PID for CUDA slots behind a runtime dlopen; absent → whole-card. |
| `server-router-ledger.{h,cpp}` | Pure accounting: given slots, router reservations, foreign usage, sysfs used, MemAvailable, headroom → free VRAM per slot and free RAM per machine (spec §4). |
| `server-router-admission.{h,cpp}` | Pure decision function (spec §6): request + ledger + residents + board claims + holds + pins → `admit` / `queue` / `evict-then-admit` with victim list, board actions, notify messages. No I/O. |
| `server-router-board.{h,cpp}` | HTTP client to the board REST routes (claim, release, renew, queue, notify) + a poller thread caching board state. |
| `server-router-machines.{h,cpp}` | Parse `machines.json`; resolve local machine, node URLs, node token path. |
| `server-node.{h,cpp}` | `--router-node` HTTP API (spec §3.2) and the child-process table it owns; also used in-process by the leader for the local machine. |

Tests: new `tests/test-router-*.cpp`, registered in `tests/CMakeLists.txt` next to `test-server-models-estimate-key.cpp` (line ~166, same `llama_build_and_test` + `llama-server-impl` link pattern). Fixtures under `tests/router-fixtures/`. Live-ish router tests extend `tools/server/tests/unit/test_router.py` only where they need no GPU.

---

## Phase 1 — Ledger v2 (local only)

### Task 1: Per-process probe

**Files:** Create `tools/server/server-router-probe.{h,cpp}`, `tests/test-router-probe.cpp`, fixture tree `tests/router-fixtures/proc-sys-*`. Modify `tools/server/CMakeLists.txt` (add sources to `llama-server-impl`), `tests/CMakeLists.txt`.

**Interfaces — Produces:**
- `struct proc_vram { int pid; std::string pdev; int64_t vram_bytes; }`
- `std::vector<proc_vram> probe_fdinfo_vram(const std::string & root)` — all PIDs, summed per (pid, pdev), deduplicated by drm-client-id.
- `struct proc_mem { int64_t rss_anon; int64_t rss_shmem; }`; `std::optional<proc_mem> probe_proc_mem(const std::string & root, int pid)`
- `int64_t probe_mem_available(const std::string & root)`
- `int64_t probe_sysfs_vram_used(const std::string & root, const std::string & pdev)`
- `std::string pdev_for_device(...)` mapping a router device name (`ROCm0`) to a PCI address — reuse whatever mapping `read_physical_free_bytes` (`server-models.cpp:1080`) already uses.

**Behavior rules:** unreadable `/proc/<pid>` (exited, permission) is skipped, not an error. Several fds of one DRM client must not be double-counted (dedupe on `drm-client-id`). Units in fdinfo are KiB/MiB-suffixed — parse both.

**Tests must assert:** a fixture with two PIDs on two pdevs, one PID with two fds sharing a client id → correct per-pdev sums with no double count; vanished PID skipped; MemAvailable and RssShmem parse; sysfs used parse.

**DoD:** tests written and committed; controller builds and `ctest -R test-router-probe` passes.

- [ ] Write failing tests + fixtures
- [ ] Implement
- [ ] Controller: build + ctest
- [ ] Commit `router: per-process VRAM/RAM probe`

### Task 2: Ledger v2

**Files:** Create `tools/server/server-router-ledger.{h,cpp}`, `tests/test-router-ledger.cpp`. Modify `tools/server/server-models.cpp` — `read_physical_free_bytes` (:1080), `effective_free_bytes_locked` (:1112), `reserve/credit/reconcile_gpu_reservation_locked` (:1142–1206), `load_gpu_config` (:849); `server-models.h` slot/meta structs.

**Interfaces — Consumes:** Task 1 probe functions. **Produces:**
- `struct ledger_slot { std::string id /* "<machine>/<dev>" */; std::string pdev; int64_t total; int64_t router_reserved; }`
- `int64_t ledger_free_vram(const ledger_slot &, int64_t foreign_bytes, int64_t sysfs_used)` = spec §4 formula.
- `int64_t ledger_free_ram(int64_t mem_available, int64_t headroom)`
- New preset keys parsed in `parse_model_placement` (:1002): `ram-mb` (per model). New router CLI flag `--models-ram-headroom-mb` (default 4096).

**Behavior rules:**
- "Foreign" = any PID that is not a router child (and not a declared group worker once Phase 2 lands). The router's own children count via their reservation, never twice.
- Remove the 8192 MB budget cap path that 2026 uses; whole-card total comes from the probe/sysfs.
- Admission (today's `ensure_gpu_placement`, :1354) must now also require RAM fit on the model's machine; RAM shortfall evicts idle residents on that machine like VRAM shortfall does.
- `/models` (:3345) and `gpu_slots_json` (:1118) expose per-slot `foreign_mb`, `free_mb`, and per-machine `ram_free_mb`.
- Slot ids stay unprefixed in this phase (local only) but go through one helper so Phase 5 can prefix them.

**Tests must assert:** formula table (foreign usage reduces free; sysfs stricter than ledger wins; router child not double-counted); RAM fit refusal; headroom respected.

**DoD:** tests pass under controller build; controller live check on mad-lab-main (board claim): `/models` shows foreign usage of a running non-router process matching `/proc/<pid>/fdinfo` within 64 MB.

- [ ] Write failing tests
- [ ] Implement
- [ ] Controller: build + ctest + live check
- [ ] Commit `router: per-process VRAM + host RAM ledger`

### Task 3: Busy-eviction fix + child env scrub

**Files:** Modify `tools/server/server-models.cpp` — `choose_gpu_evictions_locked` (:1325), child env assembly (`get_environment` :534, `parse_model_env` :970, spawn in `load` :2054). Test: `tests/test-router-ledger.cpp` (or a new `test-router-evict.cpp` if eviction selection needs extracting into a testable helper — extract it).

**Behavior rules:** eviction candidates exclude any model with `req_count > 0` (until Phase 4 adds `highest`). Child env = inherited router env + preset `env` key only. If the final env has `TEMP`/`TMP`/`TMPDIR` pointing at a non-directory, fail the load with an error naming the variable.

**Tests must assert:** busy model never chosen as victim; idle one is; env scrub rejects `TMPDIR=/etc/passwd`, accepts a real dir, accepts unset.

**DoD:** tests pass; commit `router: never evict busy children; scrub temp-dir env`.

---

## Phase 2 — Model groups (local only)

### Task 4: Preset keys for groups

**Files:** Modify `common/arg.cpp` (router preset-only keys near :5408–5459, follow `stop-timeout`/`pinned`/`exclusive`), `common/preset.cpp` (unknown-key rejection ~:318 must accept the new keys), `tools/server/server-models.{h,cpp}` (`parse_model_placement` :1002, meta struct). Test: `tests/test-router-groups.cpp` with fixture ini.

**Interfaces — Produces:** meta fields `kind` (`model` | `external`), `depends` (list of section names), `launch` (command line string), `park_file`, `vram_mb`, `ram_mb`, `slot_autosave` (passes through to `LLAMA_ARG_SLOT_AUTOSAVE`), `park_mode` (`none` | `opt-in`), `startup_timeout_s` (default 300).

**Behavior rules:** `kind=external` sections are not requestable (requests return 404-style "not a model"), not listed as loadable in `/v1/models`, but appear in `/models` with their group. `depends` must reference existing `kind=external` sections; a dangling reference fails preset load with a clear error. A worker may belong to only one group.

**Tests must assert:** fixture with spine + 2 externals parses into one group; dangling depends rejected; external not requestable; worker in two groups rejected.

**DoD:** tests pass; commit `router: preset keys for model groups`.

### Task 5: Group lifecycle (load/unload/failure)

**Files:** Modify `tools/server/server-models.cpp` — `load` (:2054), `unload` (:2301), `on_child_exit` (:2279), `server_monitor` (`stop` :102, `on_line` :199), `ensure_gpu_placement` (:1354). Test: `tests/test-router-groups.cpp` using a fake worker (a tiny script/binary in fixtures that listens on a port, prints the stop-snapshot line on TERM, or dies with `Hip error` on demand) and `tools/server/tests/unit/test_router.py` for an end-to-end non-GPU run.

**Behavior rules (spec §5.2–5.4):**
- Admission covers the whole group footprint.
- Load: start all externals in parallel with injected `WP_EXPERT_PARK_FILE` + `WP_EXPERT_SEED_FROM_PARK=1`; spine starts only when every worker port accepts TCP; group ready = spine ready.
- Worker start failure = process exit, any line matching `Hip error`, or `startup_timeout_s` elapsed → tear down started members, load fails with the reason.
- Unload order: drain spine requests → stop spine with its `stop-timeout` → SIGTERM workers in parallel → wait for exit up to quiesce+10 s → SIGKILL → mark unloaded. Record which stop-snapshot line each worker printed (written / FAILED / timeout / none) in the model's status JSON.
- Worker exits while group ready → stop spine, group status `failed`, next request reloads the group.
- Router startup: kill any process whose cmdline/env marks it as a child of a previous router generation (define the marker: an env var `LLAMA_ROUTER_GEN=<uuid>` injected into every child).

**Tests must assert:** load order (workers before spine); Hip-error worker fails the load and leaves no process; TERM path records `written`; worker death mid-serve stops spine; stale-generation child killed on startup.

**DoD:** tests pass under controller build; controller live check (board claim on mad-lab-main GPUs, DS4.1 session told in advance — this doubles as their worker validation): load `dsv41` with main worker only via a test alias, evict it, confirm no worker process remains, VRAM/RAM back, worker log shows stop-snapshot line; reload and capture `seed-from-park` + RAM-phase lines, slot-autosave restore, decode ms/step during vs after seed. Report numbers to kmbandy.

- [ ] Write failing tests + fake worker
- [ ] Implement
- [ ] Controller: build + ctest + live check
- [ ] Commit `router: model groups with full-eviction lifecycle`

---

## Phase 3 — Board (mad-lab-2026, parallel with Phases 1–2)

### Task 6: Priority, yield notify, router routes

**Files (on mad-lab-2026, branch `board/router-priority`):** `mneme/src/mneme/board.py` (`claim` :226, `_enqueue` :76, `queue_list` :128, `pop_queue` :434, `update_claim` :304), `mneme/src/mneme/service/app.py` (board routes ~:1094–1140), `mneme/src/mneme/service/client.py`, `mad-lab-mcp/server.py` (custom routes `/board/*` ~:515–650; MCP `board_claim` tool), tests `mneme/tests/test_board.py`, `test_board_queue.py`, `mad-lab-mcp/test_board_tools.py`.

**Behavior rules:**
- Claims and queue entries gain `priority` (`highest`/`middle`/`lowest`, default `middle`). Queue order: priority, then join time. `highest` never displaces the current holder.
- New write routes on mad-lab-mcp (proxied to the daemon, same style as the existing GET proxies): `POST /board/claims`, `DELETE /board/claims/{id}`, `PATCH /board/claims/{id}` (renew = set `ttl_hours`, re-anchors), `GET /board/queue`, `POST /board/queue/leave`, `POST /board/notify`.
- `POST /board/notify {claim_id, kind: wait|yield, content}` delivers to the claim holder's session through the existing delivery path (`enqueue_board_alert` style); `yield` renders as an urgent request, `wait` as informational. Idempotent per (claim_id, kind) while undelivered.
- Auth: write routes require the same bearer token as the router node (token file path configurable); GET routes unchanged.
- MCP `board_claim` tool accepts `priority`.

**Tests must assert:** queue ordering by priority then time; `highest` does not evict holder; renew re-anchors TTL; notify idempotent; write route rejects missing token.
Regression tests for the 2026-08-04 defects (expected to already pass — if either fails, fix it): a `machine` claim conflicts with an existing `gpu:X` claim on that machine and vice versa; `check(machine, "gpu:X")` reports busy exactly when `claim(machine, "gpu:X")` would queue.

**DoD:** pytest green in both repos; controller deploys only after kmbandy OKs a service restart.

- [ ] Write failing tests
- [ ] Implement
- [ ] Controller: pytest both repos
- [ ] Commit `board: priority, yield notify, router write routes`

---

## Phase 4 — Router board client, priority, holds

### Task 7: Admission as a pure function

**Files:** Create `tools/server/server-router-admission.{h,cpp}`, `tests/test-router-admission.cpp`. Modify `server-models.cpp` `ensure_gpu_placement` (:1354) and `choose_gpu_evictions_locked` (:1325) to delegate to it.

**Interfaces — Produces:** `admission_result decide_admission(const admission_input &)` where input = {alias, priority, machine override, group footprint per slot + per machine RAM, candidate slots, ledger free values, residents (name, slots, busy, pinned, held, group), board claims per resource (holder, is_router, priority)}; result = {verdict: admit | evict_then_admit | queue, chosen slots, victims, board_claims_to_take, notify list (claim_id, kind), blocked_by}.

**Behavior rules:** spec §6 exactly. Pins and holds never victims. Busy residents are victims only at `highest`. Foreign claim on a needed resource → queue (+ `wait` notify at `middle`, `yield` notify at `highest`, nothing at `lowest`). Pools pick the slot with the fewest/cheapest victims, then most free VRAM.

**Tests must assert:** a table covering priority × {free, idle resident, busy resident, pinned, held, foreign claim} × {pinned alias, pool, pool + machine override}; group footprint across two slots; RAM-only shortfall.

**DoD:** table test passes; existing router tests still pass; commit `router: pure admission decision`.

### Task 8: Board client + holds + priority plumbing + SSE

**Files:** Create `tools/server/server-router-board.{h,cpp}`, `tests/test-router-board.cpp` (against a tiny in-process fake board HTTP server). Modify `server-models.cpp` — request entry (`router_validate_model` :3086, `is_autoload` :3118, `proxy_request` :2623), `notify_sse` (:1563), `idle_sweeper_loop` (:2705), `/models` handlers (~:3345); `server.cpp` route registration.

**Behavior rules:**
- Router CLI: `--board-url`, `--board-token-file`; no board configured → board features off, router behaves as today (log a warning).
- On load: take one claim per GPU (`gpu:<board name>`) and `ram` per machine, holder `llama-router`, note = alias, ttl ~10 min, renewed every ~3 min while resident; released on unload. Queue result → model state `queued` with `queue_pos`, `blocked_by`.
- Poll board state every ~3 s; when anyone queues behind a GPU the router holds, unload idle residents there now (pins/holds excepted).
- `priority` and `machine` accepted on `POST /models/load` and on `/v1/*` (JSON body field or `X-Priority` / `X-Machine` header); per-alias default via preset key `priority`.
- `POST /models/load` returns `202 {state, queue_pos, blocked_by}` immediately when not ready; `/v1/*` at `lowest` blocks until ready, at `middle`/`highest` returns `503` + `Retry-After` + queue info while queued.
- Holds: `POST /models/hold {model, ttl_s, owner}` → `{lease}`; `POST /models/release {lease}`; renew by re-POSTing hold with the lease. Held = no eviction, no idle unload. Expired leases drop silently.
- SSE events added: `queued`, `blocked`, `loading`, `ready`, `evicting`, `unloaded` (payload includes model, machine, slots, reason).

**Tests must assert:** claim taken on load and released on unload against fake board; renew happens; queued state surfaced; yield-on-queue unloads idle resident but not held/pinned; hold blocks eviction and idle unload; lease expiry; `/v1` 503 vs block by priority.

**DoD:** tests pass under controller build; controller live check on mad-lab-main: a Claude session holding `gpu:R9700` → router load of an R9700 alias queues and the session receives the `wait` notice; release → router loads.

- [ ] Write failing tests + fake board
- [ ] Implement
- [ ] Controller: build + ctest + live check
- [ ] Commit `router: board client, priority, holds, SSE states`

---

## Phase 5 — Node mode (multi-machine)

### Task 9: Machines registry + node daemon

**Files:** Create `tools/server/server-router-machines.{h,cpp}`, `tools/server/server-node.{h,cpp}`, `tests/test-router-node.cpp`. Modify `tools/server/server.cpp` / `main.cpp` (new `--router-node` mode, `--node-token-file`, `--node-bind`), `common/arg.cpp` for the flags.

**Interfaces — Produces:** node HTTP API exactly as spec §3.2 (`/node/spawn`, `/node/stop`, `/node/signal`, `/node/state`, `/node/events`); `machines_registry load_machines(path)` with `local_machine()`, `node_url(name)`.

**Behavior rules:** node owns children (stdin pipe as today so children die with the node); children tagged `LLAMA_ROUTER_GEN`; on node start, kill or re-adopt (by name+gen, when leader asks) leftover tagged processes; `/node/state` uses the Task 1 probe; heartbeat on `/node/events` every 2 s; bearer token required on every route; temp-dir env refusal applies here too.

**Tests must assert:** spawn/stop/signal of a fake child; state reports it with its PID; missing token → 401; orphan sweep on restart.

**DoD:** tests pass; commit `router: --router-node daemon + machines.json`.

### Task 10: Leader drives remote nodes

**Files:** Modify `server-models.cpp` — replace `CHILD_ADDR` (:74, uses ~:563, :2655, :3600, :3646, :3684) with per-child host; `load`/`unload`/monitor paths go through a node interface (in-process for local, HTTP for remote); slot ids become `<machine>/<dev>`; `load_gpu_config` (:849) accepts machine-prefixed `gpus=`.

**Behavior rules:** leader reads `machines.json`; remote node heartbeat lost (>10 s) → that machine's slots offline, its models `unavailable`, requests get `503` + `Retry-After`, pools route elsewhere; heartbeat back → leader re-adopts children by name+gen or stops them. Ledger uses node-reported probe data for remote slots.

**Tests must assert:** with a fake remote node: spawn routed to it, proxy uses its host, heartbeat loss marks slots offline, re-adoption.

**DoD:** tests pass under controller build; controller deploys a node on 2026 (kmbandy OK required) and live-checks: `dsv41` as a full cross-box group (main + 2026 workers) loads and fully evicts with no processes left on either box.

- [ ] Write failing tests
- [ ] Implement
- [ ] Controller: build both boxes + ctest + live check
- [ ] Commit `router: leader drives remote nodes`

---

## Phase 6 — Pools and migration

### Task 11: Pools, replicas, machine override

**Files:** `server-models.cpp` placement parse + `decide_admission` callers; `common/arg.cpp` keys `placement` (`pinned`|`any`), `replicas`, `pool-gpus` (optional allow-list); tests in `test-router-admission.cpp` + `test-router-groups.cpp`.

**Behavior rules:** a pool alias may run up to `replicas` instances, each a router child on a different slot; requests to the alias load-balance across ready replicas (fewest in-flight); additional replicas load on demand when all are busy and admission allows at the request's priority; `machine=` restricts candidates.

**Tests must assert:** two replicas placed on two slots; third request with no free slot queues rather than evicting at `middle` when only busy residents exist; machine override honored.

**DoD:** tests pass; commit `router: pools and replicas`.

### Task 12: Migration (controller-led, with kmbandy)

Not a subagent task — the controller executes it with kmbandy's OK at each step:
- Preset: add `[*global*] gpus` with machine prefixes, move 2026's presets into the leader ini, add dsv41 worker sections (`launch` = today's per-worker command from `~/ds4-runs/dsv41/workers-la.sh` with `LA=0`, 8G/8G tiers), `stop-timeout=120`.
- Per-arm override path for the DS4.1 bench harness (agree the mechanism with the DS4.1 session; no whole-file ini overwrite).
- Repoint clients to :8090: mad-lab-dash engine endpoints, madlab-console `crates/madlab-core/src/settings.rs` router_url, mneme `board_probe.py` router probe, `ds41-serve` / `qwen38-serve` fish functions → `/models/load` wrappers.
- Retire :8093 (stop unit only after kmbandy OK).
- Final live check: DS4.1 → 2× Qwen 3.8 27B (parallel) → DS4.1 full cycle timed vs today's 3–5 min cold swap; report to kmbandy.
