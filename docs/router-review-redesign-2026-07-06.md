# llama.cpp Model-Router: Review + Redesign for Per-Machine Multi-GPU Routing

**Date:** 2026-07-06
**Scope:** `tools/server/server-models.{h,cpp}`, `tools/server/server-cors-proxy.h`, router sections of `tools/server/server.cpp` in the fork at `~/GitHub/llama.cpp` (branch `sync/upstream-2026-07-06`, HEAD `06a3da0e6`).
**Method:** read-only code review + live inspection of both machines. No code was modified, no services touched, no inference run.

---

## 0. Verified Topology (measured, not assumed)

### mad-lab-2026 (this box, `hostname` confirmed)

| Device | VRAM | Backend reality | ggml device name |
|---|---|---|---|
| NVIDIA GTX 1070 | 8 GiB (7.4 GiB in use at inspection) | CUDA sm_61 | `CUDA0` (also enumerated as `Vulkan1`) |
| AMD RX 480 (gfx803, Polaris) | 8 GiB | **No usable ROCm/HIP** — driven via **Vulkan/RADV** | `Vulkan0` |

- `build-army` is a **single unified binary with BOTH backends compiled in**: `GGML_CUDA:BOOL=ON` + `GGML_VULKAN:BOOL=ON` (`build-army/CMakeCache.txt`), `CMAKE_CUDA_ARCHITECTURES=61`. So the user's "one build drives 2 GPUs" **holds here**, with two caveats:
  1. The 1070 is enumerated **twice** (`CUDA0` and `Vulkan1`); `llama-start-vulkan.sh` explicitly pins `--device Vulkan0` for the RX 480 "so the unified multi-backend build doesn't auto-split across both".
  2. gfx803 cannot be driven by HIP; Vulkan is the only path (plus the documented RADV runtime-PM/GTT-spill quirk worked around with `RADV_PERFTEST=nogttspill` in `llama-start-vulkan.sh`).
- Current serving = **static per-model systemd units** (`llama-server-{embedding,lfm25-8b,ornith-9b,...}.service`), pinned per GPU via `Environment=CUDA_VISIBLE_DEVICES=0` (1070 units) or `--device Vulkan0` (RX 480 units). **Router mode (`--models-dir`) is not deployed anywhere** (no unit or script references it on either machine).

### mad-lab-main (remote, ssh read-only; default remote shell is fish)

| Device | VRAM (rocm-smi) | ISA | ggml device name |
|---|---|---|---|
| AMD Radeon AI PRO R9700 | 32 GiB (card1, 31.4 GiB in use — musacoder-27B was resident) | gfx1201 (RDNA4) | `ROCm0` |
| AMD RX 6900 XT | 16 GiB (card0, ~0.3 GiB in use) | gfx1030 (RDNA2) | `ROCm1` |

- **There is no `build-hop`.** The build actually referenced by every unit is **`build-hip`** (`ExecStart=.../build-hip/bin/llama-server`). Presumably "build-hop" was a mis-remembering.
- `build-hip/CMakeCache.txt`: `GGML_HIP:BOOL=ON`, `AMDGPU_TARGETS=gfx1201;gfx1030` — **both ISAs are compiled into the one HIP build**, so a single process CAN tensor-split one model across R9700 + 6900 XT. Same-backend split (layer mode) is fully supported.
- Units pin placement with `--device ROCm0` (R9700: `qwen36-27b-swatm-r9700`) or `--device ROCm1` (6900 XT: `gemma4-12b`, `lfm2-24b`, `lfm25-8b-swarm-6900xt`). Units with **no `--device`** (`gpt-oss`, `musacoder`, `ornith-35b`, `qwen36-27b`, `qwen36-35b`, `qwen3-coder`) default to *all* HIP devices — this is how "a large model spans both GPUs" happens today (implicitly, via default split-mode=layer + the default `fit_params` machinery), not via an explicit `--tensor-split`. **No unit currently passes `--tensor-split`.**
- Several units share ports 8090/8091/8092 → they are mutually exclusive and are started/stopped by hand. **This manual swap dance is exactly the problem the router should absorb.**

### Tension with the user's description
1. "one llama.cpp build per machine, each build drives 2 GPUs" — TRUE on both machines, but on mad-lab-2026 only because build-army is a dual-backend (CUDA+Vulkan) binary; a HIP+CUDA combination would be impossible (GGML_HIP reuses the ggml-cuda source tree; the two are mutually exclusive at build time). The 1070's double enumeration (CUDA0/Vulkan1) must be handled by any router policy.
2. `build-hop` does not exist; it is `build-hip`.
3. The "large model spanning both GPUs" on mad-lab-main currently happens by *omitting* `--device`, not by explicit `--tensor-split`.

---

## 1. GOAL 1 — Upstream Maintenance Status

### Caveat: the local "upstream mirror" is unusable
`~/GitHub/llama.cpp-upstream` is a **shallow clone (depth 1)** frozen at `873c825` (2026-04-13) — `rev-list --count HEAD` = 1, `.git/shallow` present. It cannot answer history questions. All upstream measurements below therefore use the fork's own `upstream` remote (`https://github.com/ggml-org/llama.cpp.git`), fetched 2026-07-06 (`upstream/master` = `20a04b220`, 2026-07-06). **Recommend either unshallowing or deleting the mirror.**

### Verdict: actively maintained, fast-moving
`git log --follow -- tools/server/server-models.cpp` (in the fork, which contains full upstream history): **53 commits** on this file since its introduction (~Dec 2025). Sample of the last ~6 weeks:

```
799fcc04a 2026-06-30 common,server: handle bracketed IPv6 literals in URL authority
1a87dcdc4 2026-06-26 server + ui: SSE Replay Buffer (#23226)
721354fbd 2026-06-22 server: (router) move model downloading to dedicated process (#24834)
d6d899580 2026-06-21 server: real-time model load progress tracking via /models/sse (#24828)
2b686a912 2026-06-20 server: refactor child --> router communication (#24821)
fe7c8b241 2026-06-18 server: (router) fix stopping_thread potentially hang (#24728)
968c43891 2026-06-18 server: fix router args not being forwarded to child instances (#24760)
4b4d13ae7 2026-06-17 server: (router) add model management API (#23976)
```

Primary maintainers: **Xuan-Son Nguyen (ngxson)** and Pascal. Churn is high (comms protocol refactored 2026-06-20, download moved to a dedicated child 2026-06-22, management API added 2026-06-17) — the subsystem is under active construction, and `server.cpp:432` still prints "router mode is experimental".

### Fork divergence: zero on the router
- `git diff upstream/master...HEAD -- tools/server/server-models.{cpp,h} server-cors-proxy.h server.cpp` → **empty**. The fork carries **no local router patches** and is **not behind** upstream on any router fix (branch merged `upstream/master` on 2026-07-06).
- Fork-local changes live elsewhere in `tools/server/` (MTP/speculative work, `--kv-tiered`/turbo4 KV features visible in `server-context.cpp` and the systemd units) — the router passes these through to children untouched because children run the same binary.

### What would need review/porting if we patch the router
Since the fork will diverge once we add placement logic (Goal 4), the specific hot areas to re-review on every upstream sync are: (a) `server_models::load()`/`unload_lru()` (our insertion point, and upstream's known TOCTOU-fix area, cf. `c1b911654` "fix router mode deadlock on child crash and TOCTOU race in models_max"); (b) the child↔router state protocol (`CMD_CHILD_TO_ROUTER_STATE`, refactored a month ago and likely to change again — upstream TODO at `server-models.cpp:910` about splitting stdout/stderr); (c) the preset system (`common/preset.cpp`), which upstream reworked twice in June (`60bc8866b`, `75ad0b23e`). Also worth tracking: the commented-out upstream stub for pin-in-memory at `common/arg.cpp:4538` ("in server router mode, do not unload this model if models_max is exceeded") — upstream intends a pinning flag, which our design also needs.

---

## 2. GOAL 2 — How the Router Operates Today

### Process model: **one router process proxying to spawned child `llama-server` processes**
This is definitive, not one-process-many-models:
- Router mode is entered when llama-server is started with **no model** (`server.cpp:110-111`: `is_router_server = params.model.path.empty() && params.model.hf_repo.empty()`). The router **never touches the GPU** — device enumeration is explicitly skipped so no CUDA/HIP primary context is created (`server.cpp:113-114`, upstream `64b38b561`).
- `server_models::load()` **spawns a child subprocess of the same binary** via `subprocess_create_ex` (`server-models.cpp:914`), passing the rendered per-model args and a copy of the router's environment plus `LLAMA_SERVER_ROUTER_PORT=<router port>` (`server-models.cpp:891-893`). A process detects it is a child by that env var (`server-models.cpp:1377-1380`).
- Each child binds `127.0.0.1:<random free port>` (`CHILD_ADDR`, `server-models.cpp:55`; `get_free_port()` binds port 0 and reads back the assignment, `server-models.cpp:713-771`).
- Requests are **HTTP-proxied** to the owning child (`server_models::proxy_request`, `server-models.cpp:1272-1303`, using `server_http_proxy`, `server-models.cpp:2147-2310` — a thread + unbounded in-memory `pipe_t` queue streaming chunks back).

### Registration / discovery (`server_models::load_models()`, `server-models.cpp:342-668`)
Three preset sources, merged with defined precedence (`:345-390`):
1. **cache** — models previously downloaded from HF (`ctx_preset.load_from_cache()`);
2. **`--models-dir`** — local GGUF directory scan (`common/arg.cpp:3459`);
3. **`--models-preset <ini>`** — an INI file; each `[section]` is a model whose keys are any llama-server CLI option (mapped via arg env names, e.g. `LLAMA_ARG_DEVICE`), plus a `[*global*]` section cascaded onto all (`:354-364`).

Local dir beats cache; custom preset merges on top; finally the **router's own CLI args are overlaid onto every model** (`:388-390`) so e.g. a router-level `--temp 0` reaches all children. Per model, `--alias a,b` and `--tags t1,t2` are parsed into `aliases`/`tags` sets (`:283-303`); name/alias conflicts throw. Reload (`/models?reload=1`, or `need_reload` after a download) diffs the new preset list against running instances: changed/removed running models are unloaded, presets of stopped models updated, new ones added (`:491-668`).

### State machine (`server-models.h:19-38`)
```
UNLOADED -> LOADING -> LOADED <-> SLEEPING     (+ DOWNLOADING -> DOWNLOADED for fetches)
    ^-failed-/            \-> UNLOADED (exit)
```
`is_running()` = LOADED | LOADING | SLEEPING (`server-models.h:93-95`) — note **SLEEPING counts against `--models-max`**.

### Load path (`server_models::load()`, `server-models.cpp:838-1039`)
1. `unload_lru()` first (`:843`), then under the mutex: block while a reload is in flight (`:849`), re-check capacity (`:861-871`, throws `"model limit reached, try again later"` on race loss).
2. Pick a free port, render args — `update_args()` injects `--host 127.0.0.1 --port N --alias name` into the preset and re-injects the `LLAMA_APP_CMD` subcommand for the unified binary (`server-models.cpp:202-218`).
3. Spawn child; start a **monitoring thread** per instance (`:922-1018`) that (a) tails the child's combined stdout/stderr, forwarding to the router log and intercepting control lines prefixed `cmd_child_to_router:state:` (`:938-943`, handled by `handle_child_state()` `:1305-1371`); (b) runs a `stopping_thread` that on stop request writes `cmd_router_to_child:exit` to the child's stdin and force-kills after `stop_timeout` (default 10 s, `:48`; per-model `stop-timeout` preset key).
4. Child side: loads the model normally, reports LOADING progress / READY(+model info) / SLEEPING over stdout (`notify_to_router`, `:1493-1504`; wired in `server.cpp:388-392,449-452`).

### Unload path (`:1041-1066`)
`unload()` inserts the name into `stopping_models` and wakes the stopping thread → graceful stdin exit command → child process exits → monitoring thread reaps it (`subprocess_join`, `:1005`) and sets status UNLOADED. A model still LOADING is force-killed (`:1055-1059`). **VRAM lifecycle is therefore process-granular: all VRAM for a model is freed by child process exit — nothing finer.**

### What triggers a swap
- **Request routing by model name only.** `proxy_post` parses the JSON body and reads `"model"` (`server-models.cpp:1667-1683`); GET uses `?model=` (`:1656-1665`). Alias → canonical name resolution in `router_validate_model` (`:1557-1578`).
- If the target is UNLOADED and `--models-autoload` (default **on**, `common/common.h:691`; per-request override `?autoload=`, `:1580-1587`), the request path calls `ensure_model_ready()` (`:1238-1270`) which loads and **blocks until READY**.
- **LRU eviction only at load time**: `unload_lru()` (`:801-832`) evicts exactly one least-recently-used running model when `count_active >= models_max` (default 4, `common/arg.cpp:3471`). `last_used` is bumped only by proxied POSTs (`:1280-1283`). **There is no router-side idle timer.**
- **Idle sleep is child-side**: `--sleep-idle-seconds` (`common/arg.cpp:3611-3617`, default -1 = off). The child's queue loop flips to SLEEPING after N idle seconds (`server-queue.cpp:125-187`); `handle_sleeping_state()` then calls `destroy()` — **frees model, contexts, mtmd, i.e. all VRAM — while the process stays alive** (`server-context.cpp:1052-1082`). Any new request wakes it by a **full model reload from disk** (`load_model(params_base)`, `:1076-1078`). The router just mirrors the state (`:1246-1248`: sleeping counts as "running"; the proxied request itself wakes the child).
- Conversation stickiness: `X-Conversation-Id` → owning model map for resumable streams (`server-models.h:136-168`, routes `:1874-1997`) — best-effort, not placement.

### Relevant flags & env (verified)
| Flag / env | Where | Role |
|---|---|---|
| `--models-dir`, `--models-preset`, `--models-max` (4), `--models-autoload` (on) | `common/arg.cpp:3459-3481` | router sources & policy |
| `--sleep-idle-seconds` (-1) | `arg.cpp:3611`, `common/common.h:665` | child idle sleep |
| `stop-timeout`, `load-on-startup` | preset-only keys (`server-models.cpp:417-430,473-489`) | per-model lifecycle |
| `--device` / `LLAMA_ARG_DEVICE` | `arg.cpp:2638-2644` → `parse_device_list()` `:924-943` → `ggml_backend_dev_by_name()` | filters ggml device list per process; names are backend+index: `CUDA0`, `ROCm0/ROCm1` (HIP build renames CUDA→ROCm, `ggml-cuda/common.cuh:1456`), `Vulkan0/Vulkan1` |
| `--split-mode` (`layer`/`row`/`none`), `--tensor-split`, `--main-gpu` | `arg.cpp:2757,2784,2794` | distribution across the filtered device list |
| `--fit-params` family | `common/common.h:476-481`, `common/fit.h` | default-on auto-shrink of ctx/offload to fit free VRAM, 1 GiB/device margin |
| `CUDA_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` / `ROCR_VISIBLE_DEVICES` / `GGML_VK_VISIBLE_DEVICES` (`ggml-vulkan.cpp:7047`) | driver/backend | gate device *visibility*; children inherit the router's env verbatim (`server-models.cpp:892` copies `base_env`) — **no per-child env control today** |
| `LLAMA_SERVER_ROUTER_PORT`, `LLAMA_SERVER_CHILD_MODE=download`, `LLAMA_APP_CMD` | `server-models.cpp:893,897,214` | child plumbing |

### server-cors-proxy.h — not part of model routing
It is the WebUI-MCP CORS proxy: `/cors-proxy?url=...` forwards a request to an **arbitrary client-supplied URL** (`server-cors-proxy.h:22-75`), gated only by `--ui-mcp-proxy` (`server.cpp:287-297`) and a scheme check. It shares the `server_http_proxy` machinery, nothing else. Security note: when enabled it is an SSRF primitive into the LAN (upstream itself warns "do not expose"); keep it off on the fleet (it is off by default).

---

## 3. GOAL 3 — Improvement Opportunities (ranked)

**P1. VRAM-aware placement, per-GPU (fork-local, the Goal 4 redesign).** Today the only capacity control is a *count* (`models_max`); the router knows nothing about devices or bytes. On a 32+16 GiB box a count is meaningless. See §4. *(Upstream-portable in spirit — ngxson's stub at `arg.cpp:4538` shows appetite for richer policy — but ship fork-first.)*

**P2. Router-side idle eviction policy (upstream-portable, small).** There is no idle unload; the only idle mechanism is child sleep, and a SLEEPING child still consumes a `models_max` slot (`server-models.h:93`). Add a router timer thread: `if (now - last_used > idle_unload_seconds) unload(name)`, per-model preset key `idle-unload-seconds`, plus a `pinned` preset key exempting models (matches upstream's commented intent, `arg.cpp:4538`). Also make `unload_lru()` skip SLEEPING models when choosing what to evict *for capacity* — evicting a sleeping child frees ~nothing (its VRAM is already released) while killing warm page cache.

**P3. Swap speed (mix of fork-local ops changes and upstream-portable code).**
- *Ops (no code):* the mad-lab-main units all pass `--no-mmap`, which forces a full read + copy on every load and makes sleep/wake (a full reload, `server-context.cpp:1076`) maximally slow. With mmap, a swap-in of a recently used model comes largely from page cache. Recommendation: drop `--no-mmap` for router-managed models on mad-lab-main (64 GiB host RAM class) unless a specific allocator issue forced it; keep `--no-warmup` choice per model.
- *Warm standby:* `--sleep-idle-seconds` + generous `models_max` is the cheap "keep resident" tier: process alive, VRAM free, wake = mmap reload from page cache. Combine with P2 so sleeping models don't block slots.
- *Code:* (a) partial sleep — free KV/compute buffers but keep weights resident (needs a new path in `server_context_impl::destroy()`, `server-context.cpp:1057-1068`; biggest single win for wake latency, weights are the bulk of load time); (b) predictive prefetch — on `load(A)`, background-`readahead()` the GGUF of the most-likely-next model; (c) parallel load — `load()` currently serializes spawn under the global mutex; the spawn is cheap but `ensure_model_ready` waits are per-request threads, fine — the real serialization is GPU allocation in children, unavoidable.
- *Fleet synergy:* the NVMe→VRAM direct-load work (SAM/ReBAR, mad-lab-main) would slot in below mmap as a faster cold path; the router design below is agnostic to it.

**P4. Correctness/robustness gaps (all upstream-portable):**
1. **Unbounded proxy buffering** — `pipe_t::write()` never blocks or caps (`server-models.cpp:2042-2050`); a slow client + fast child buffers the whole response in router RAM. Cap the queue and apply backpressure.
2. **`proxy_post` fully parses every request body** just to read `"model"` (`:1669`) — for multimodal payloads with base64 images this is a large, per-request JSON parse done *twice* (again in the child). A streaming scan for the top-level `"model"` key, or honoring a `X-Model` header, removes it.
3. **Capacity race behavior**: a load that loses the `models_max` re-check throws "try again later" (`:869`) → a client request 500s instead of queueing. Should enqueue behind the eviction it just triggered.
4. **`get_free_port()` TOCTOU** (`:713-771`): port can be stolen between close and child bind. Rare, but a bind-retry loop in the child (or passing an inherited bound socket) is cheap insurance.
5. **Control protocol over combined stdout/stderr** (`:912`, TODO at `:910`): a child log line ≥128 KiB splits `fgets` and can shear a `cmd_child_to_router:state:` frame. Move control to a dedicated fd/pipe (upstream is drifting this way already).
6. **`unload_lru()` selection race** (`:801-832`): scan and unload are under separate lock acquisitions; concurrent loads can double-evict. Also `cv.wait` uses `mapping[name]` (`:828`) which default-constructs an entry if the model got erased mid-wait.
7. **Router `/health` is unconditionally OK** — it never reflects child health; a wedged child is invisible to fleet monitoring except via per-request failures. Add per-child liveness probing (the monitoring thread already knows process state; expose it).

**P5. Routing quality (upstream-portable, nice-to-have).** Only exact name/alias match exists; the `tags` field is explicitly "not used for routing" (`server-models.h:77`). Tag-based selection ("smallest loaded model tagged `code`"), or fallback-to-loaded-model on 404, would let clients name capabilities instead of files. Low urgency for a 2-box fleet with pinned clients.

---

## 4. GOAL 4 — Redesign: Machine-Level Multi-GPU Routing

### 4.0 Grounding in llama.cpp's device model (verified)
- ggml registers one backend per compiled backend lib; each exposes devices named `<Backend><idx>`: on mad-lab-main's HIP build `ROCm0` = R9700 (gfx1201), `ROCm1` = 6900 XT (gfx1030) (agent order matches `rocminfo`; confirmed by unit descriptions). On mad-lab-2026's dual build: `CUDA0` = 1070, `Vulkan0` = RX 480, `Vulkan1` = 1070-again.
- `--device a,b` filters that list per process (`arg.cpp:924-943`); `--tensor-split f0,f1` splits layers (mode `layer`) or rows (mode `row`, CUDA/HIP only, needs fast peer access — not recommended over PCIe here) across the *filtered* list; `--main-gpu` indexes into the filtered list and owns non-repeating tensors/small buffers. With no `--tensor-split`, `llama.cpp` splits proportionally to free memory at load time, and default-on `--fit-params` (`common/common.h:476`) auto-shrinks ctx/offload to fit with a 1 GiB/device margin.
- Free/total VRAM per device is available in-process via `ggml_backend_dev_memory()` (used by `common/fit.cpp:108-132`), and out-of-process via sysfs (`/sys/class/drm/cardN/device/mem_info_vram_{total,used}`, AMD) / NVML (NVIDIA). The router must use the out-of-process path because it deliberately never initializes GPU backends (`server.cpp:113-114`).
- Visibility env (`HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `CUDA_VISIBLE_DEVICES`, `GGML_VK_VISIBLE_DEVICES`) is a *sledgehammer*: it renumbers devices per process. Since `--device` gives exact, stable, name-based placement **within one build**, the design below uses `--device` as the primary mechanism and env-gating only as an optional belt-and-braces layer (which requires a small code change: children currently inherit `base_env` verbatim, `server-models.cpp:892`).

### 4.1 Concepts and data structures (code change in `server-models.{h,cpp}`)

```cpp
// router-side, no GPU context ever created
struct gpu_slot {
    std::string  dev_name;     // "ROCm0" — the exact string passed to --device
    std::string  vram_probe;   // sysfs path or "nvml:<uuid>" for physical readings
    int64_t      total_bytes;
    int64_t      reserved_bytes;   // sum of ledger reservations (logical)
    // physical_free() polled from probe; effective_free = min(physical, total - reserved)
    std::string  exclusive_holder; // model name holding an exclusive (spanning) reservation, or ""
};

struct placement {                 // per model, stored in server_model_meta
    std::vector<std::string> devs; // 1 dev = pinned; >1 = spanning; empty = "any"
    std::vector<float> split;      // tensor-split fractions when devs.size() > 1
    int64_t need_bytes_per_dev[GGML_MAX_DEVICES]; // estimate, see 4.3
    bool exclusive;                // spanning models take whole devices
    bool pinned;                   // never auto-evicted
};

struct gpu_ledger {                // one per router, guarded by server_models::mutex
    std::vector<gpu_slot> slots;   // built from config at startup (see 4.2)
    // reservations keyed by model name; credit back on UNLOADED (and see sleep note, 4.6)
};
```

Preset-only INI keys (same mechanism as `load-on-startup` / `stop-timeout`, parsed in `load_models()` around `server-models.cpp:417-430`):

```ini
[qwen36-27b]
model = /home/kmbandy/models/Qwen3.6-27B-Q6_K.gguf
gpu = ROCm0            ; pin | "any" (default) | comma list = span
vram-mb = 24500        ; optional override of the estimator
pinned = true          ; optional: exempt from eviction

[ornith-35b]
gpu = ROCm0,ROCm1      ; SPANNING model: reserves BOTH devices exclusively
tensor-split = 2,1     ; optional; default = proportional to slot totals (32:16)
```

Because per-model INI keys are just CLI args, `gpu=` maps onto injecting `LLAMA_ARG_DEVICE` (and `LLAMA_ARG_TENSOR_SPLIT`) into the child preset exactly where `update_args()` already injects HOST/PORT/ALIAS (`server-models.cpp:205-207`) — a ~5-line rendering change; the new logic is all in the ledger/policy.

### 4.2 Per-machine device config (handles the heterogeneity)
The router cannot enumerate GPUs itself, and on mad-lab-2026 raw enumeration would be wrong anyway (1070 appears twice). So the slot table is **declared once per machine** (router INI `[*global*]` or a small `--gpus` config):

- **mad-lab-main:** `ROCm0` (34.2 GB total, probe `/sys/class/drm/card1/device/mem_info_vram_used`), `ROCm1` (17.2 GB, card0). Homogeneous backend → spanning allowed between them.
- **mad-lab-2026:** `CUDA0` (8 GB, NVML), `Vulkan0` (8 GB, card1 sysfs). `Vulkan1` is *not listed* → can never be scheduled, which structurally fixes the double-enumeration hazard. Spanning across CUDA0+Vulkan0 is technically possible (mixed-backend layer split is supported by the device-list model) but pointless on 8+8 GB Pascal/Polaris over PCIe — the config simply omits a span permission here. Every model on this box is a single-GPU pin, mirroring today's units (1070 = CUDA models, RX 480 = Vulkan models).
- A one-shot startup validation spawns `llama-server --list-devices` (a child, so the router still never opens a GPU context) and asserts every configured `dev_name` exists — catching driver renumbering after kernel/ROCm upgrades.

### 4.3 Footprint estimation (reuses `common/fit`)
`common_get_device_memory_data()` (`common/fit.h:48-56`) already loads a model with `no_alloc` and returns per-device `{model, context, compute}` byte requirements. Add a third child mode next to DOWNLOAD (`server_child_mode` at `server-models.h:46-49`): `SERVER_CHILD_MODE_ESTIMATE` — spawn the same binary with the model's full arg set plus an estimate flag; it prints the per-device byte breakdown via the existing `notify_to_router` stdout channel and exits. Cache the result on disk keyed by `(gguf path+mtime, n_ctx, kv cache types, n_parallel)`. `vram-mb` in the preset overrides. This runs off the hot path (first registration / reload), so estimate latency is irrelevant. Note the fork's `--kv-tiered` moves KV partially off-VRAM — the estimator runs the *same fork binary with the same flags*, so tiering is reflected automatically.

### 4.4 Placement + eviction algorithm (replaces `unload_lru()`, `server-models.cpp:801`)
All under the existing `server_models::mutex` (single placement decision at a time — this also serializes competing spanning loads, avoiding eviction deadlocks):

```
place(model M):
 1. candidates = M.devs if pinned/spanning else all slots (backend-compatible)
 2. for spanning M (devs = D1..Dn):
      need exclusive hold on every Di:
        evict-set = every non-pinned resident model on any Di   (refuse if any pinned)
 3. for single-GPU M:
      fits(dev) := effective_free(dev) >= need_bytes + margin(1 GiB, fit_params_target)
      if any candidate fits -> pick max effective_free (best-fit on the emptiest GPU,
                               keeps the other GPU free for the next big model)
      else -> per candidate dev, compute cheapest evict-set:
              non-pinned residents sorted by last_used asc (LRU), skipping SLEEPING
              models *only if* their reservation was already credited back (see 4.6);
              cost = bytes_still_warm evicted; pick dev minimizing (#evictions, recency)
 4. execute: unload(evict-set members), wait UNLOADED (existing cv machinery,
             server-models.cpp:826-830), debit ledger, mark exclusive_holder for spans,
             inject --device/--tensor-split/--main-gpu into the preset, spawn child.
 5. on child READY: reconcile ledger against physical probe (estimate error -> log + adjust).
    on child exit/fail: credit ledger, clear exclusive_holder.
Any request routed to a dev whose exclusive_holder != "" and != M -> treated as "does not fit"
-> triggers eviction of the spanning model iff it is LRU and not pinned, else 503 with Retry-After.
```

**The mad-lab-main flagship case** falls out directly: `ornith-35b` (or any big model) declares `gpu = ROCm0,ROCm1`. Loading it evicts *everything* non-pinned from both GPUs, takes both slots exclusively, and the child gets `--device ROCm0,ROCm1 --split-mode layer --tensor-split 2,1 --main-gpu 0` (main on the R9700; ratio defaults to total-VRAM proportion, tunable per model). While it holds the machine, a request for `gemma4-12b` either 503s or (policy `autoload`) evicts the spanner if it is LRU — an explicit, ledger-visible decision instead of today's silent OOM-or-fit lottery. Conversely, loading the spanner while `qwen36-27b` sits on ROCm0 evicts qwen from ROCm0 *and* whatever is on ROCm1 in one placement transaction.

### 4.5 What is a code change vs orchestration wrapper
- **Must be in `server-models.cpp` (fork patch):** the ledger, placement/eviction algorithm, preset-only `gpu/vram-mb/pinned` keys, `--device/--tensor-split` injection, ESTIMATE child mode, ledger↔SSE events. Reason: with `--models-autoload` the load decision happens *on the request path inside the router* (`router_validate_model` → `ensure_model_ready`, `server-models.cpp:1570`); an external wrapper cannot intercept it, and the `/models/load` API takes no placement overrides. Estimated size: ~500-700 lines, entirely additive around `load()`/`unload_lru()` — good upstream-PR shape (it generalizes ngxson's pinning TODO).
- **Can be an orchestration wrapper (interim, zero-patch):** a static regime that already beats today's manual unit-swapping: run the stock router with `--models-preset fleet.ini` where every model carries a hand-written `device = ...` (and the spanner its `tensor-split`), `--models-max 2` on mad-lab-main. The stock LRU then handles swaps; what you lose is byte-accuracy (two 6900XT-pinned models can be co-resident only if the *count* allows, regardless of bytes) and exclusive spanning semantics (a spanner load will LRU-evict only *one* model). This interim mode is worth deploying first to shake out the child-arg plumbing (the fork's `--kv-tiered` flags flowing through presets) before the patch lands.
- **Stays outside llama.cpp either way:** systemd unit for the router itself, the per-machine GPU slot config, monitoring (scrape the ledger via an extended `/models` response), and the RX 480 runtime-PM root fix (`power/control=on`).

### 4.6 Sleep interaction (design decision required)
When a child sleeps, `destroy()` frees its VRAM (`server-context.cpp:1070-1075`) but the router-side reservation would still hold. Two coherent options:
- **(a) Keep the reservation (safe, recommended initially):** sleeping = "parked but guaranteed to wake". No placement churn; wake can never fail for lack of VRAM. Cost: idle VRAM is not reusable.
- **(b) Credit back on SLEEPING:** maximizes utilization, but wake happens *inside the child* on the next proxied request with no placement check — the GPU may have been refilled. Requires routing wakes through `place()` (intercept in `ensure_model_ready`, `server-models.cpp:1246-1248`: instead of proxying straight to a sleeping child, first re-reserve; if the dev is now full, evict per policy or unload+reload the sleeper elsewhere). This is the "models move in/out of individual GPUs" behavior of requirement (a) taken to its conclusion — a model can *wake on a different GPU* than it slept on, since placement is just `--device` at (re)spawn... but only via full unload/respawn, because `--device` is fixed at process start. Phase 2.

### 4.7 Risks & open questions
1. **Estimate drift:** long-context units (1M+ ctx, `--kv-tiered`, `--ctx-checkpoints`) have workload-dependent VRAM curves; the ledger must reconcile against the physical probe at READY and periodically (mitigation in 4.4 step 5). The 1 GiB `fit_params_target` margin is the backstop.
2. **RX 480 probe lies:** amdgpu runtime-PM evicts VRAM→GTT when idle (documented in `llama-start-vulkan.sh`); `mem_info_vram_used` can read ~0 for a model that will re-migrate on wake. Mitigation: on mad-lab-2026 trust the ledger (logical) over the probe for Vulkan0, and apply the root fix (`power/control=on`).
3. **Device order stability:** `ROCm0/1` follow HIP enumeration; a PCIe re-seat or ROCm upgrade can swap them. The `--list-devices` startup assertion (4.2) plus PCI-bus-ID pinning in the slot config (match probe path by PCI address, not cardN) de-risks this.
4. **Upstream churn:** the router is being actively refactored (comms protocol, presets). Keeping the patch additive and behind preset keys minimizes rebase pain, but every sync must re-review `load()`/`unload_lru()`/preset code (§1).
5. **Fork feature flags in presets:** `--kv-tiered` etc. must round-trip through `common_preset` (they do if registered in `arg.cpp` with env names — verify the fork registered `LLAMA_ARG_*` envs for its custom flags before relying on INI presets; unverified in this review).
6. **gfx1030 + gfx1201 mixed kernels in one process:** compiled targets cover both (CMakeCache verified), but FA/rocWMMA paths differ per arch; a spanning model runs at the speed of the slower card for split layers. Benchmark the 2:1 split before committing to spanning as the default for 30B-class models that *could* fit on the R9700 alone (musacoder-27B currently does, 31.4/34.2 GiB).
7. **Open:** should the router expose the ledger on `/models` (yes, add `"devices"` array) — needed for fleet dashboards; and should cross-machine routing (2026 ↔ main) live in this layer or stay in the ollama-proxy/client config? This review deliberately scopes to per-machine.

---

## Appendix: evidence index
- Fork state: branch `sync/upstream-2026-07-06`, HEAD `06a3da0e6`; `upstream` remote = ggml-org, fetched 2026-07-06 (`20a04b220`). Router-file diff vs `upstream/master`: empty.
- `~/GitHub/llama.cpp-upstream`: shallow depth-1 at `873c825` (2026-04-13) — stale, not used as baseline.
- mad-lab-2026: `hostname` = mad-lab-2026; `nvidia-smi` GTX 1070 8 GiB; `rocminfo` gfx803 RX 480; `vulkaninfo` GPU0=RX480(RADV), GPU1=GTX1070; `build-army/CMakeCache.txt` GGML_CUDA=ON + GGML_VULKAN=ON.
- mad-lab-main (ssh, read-only): `rocminfo` gfx1201 R9700 + gfx1030 6900 XT; `build-hip/CMakeCache.txt` `AMDGPU_TARGETS=gfx1201;gfx1030`; `rocm-smi` card0 17.2 GB / card1 34.2 GB; 10 static `llama-server-*.service` units on build-hip, devices pinned via `--device ROCm0/ROCm1` or unpinned (default spans); no `--tensor-split` anywhere; no router deployment on either machine.
