# MoE-Aware Resident/Paged Split + Multi-Device Placement + Expert-Locality Cache — Design

**Date:** 2026-07-07
**Branch:** `feat/router-multigpu`
**Status:** Design approved; ready for implementation planning (writing-plans → Codex per-task with review).

---

## Goal

Run frontier fine-grained MoE models (DeepSeek V4 Flash, Qwen3.5-397B-A17B, GLM-5.2, MiniMax-M3) on a single-box, dual-GPU rig by keeping the small **dense** weight set VRAM-resident and streaming only the sparse **routed experts** from NVMe. Lift measured paged decode of DeepSeek V4 Flash Q8 (~151 GB) on the R9700 from **0.02 t/s** toward **0.5–3 t/s**.

## Rig

- **ROCm0 = R9700**, 32 GB, gfx1201, native PCIe (fast), the NVMe-facing paging + compute card.
- **ROCm1 = 6900XT**, 16 GB, gfx1030, **Thunderbolt 3 eGPU** (~2.75 GB/s PCIe-tunnel), the dense-resident card.
- ~15 GB system RAM. NVMe SSD holds the model shards.

---

## Confirmed root cause (measured, this session)

The pager init reports `33987 pages, 384 slots × 64 MB (24 GiB), nothing resident`. **Every one of the 33,987 tensors — dense and expert — demand-pages through the same 384-slot pool.** Evidence it is structural, not I/O-bound:

- **P2P and host transports produced byte-identical stats**: `page_ins=9121, evictions=5895, prefetch_hit_rate≈5%, sync_fallbacks=1696, io_effective_gb_s=0.101`. Two different I/O mechanisms, identical throughput ⇒ the transport is not the limiter.
- Arch math: DeepSeek V4 Flash = 43 layers × 6 routed experts/token ≈ 258 experts ≈ 6.7 GB that legitimately must stream. But ~1013 page-ins/token were measured ⇒ **~750/token are dense weights** (MLA attention, shared expert, norms, embeddings) that are needed every token yet get evicted and re-read. `lru_walk_pinned_skips=132634` confirms the pool thrashing against transient ensure_batch pins.

`io_effective_gb_s=0.101` is partly a measurement artifact (batched pages are each charged `seconds_since(io_t0)` from a single pre-loop `io_t0` at `wp-pager.cpp:958-970`, inflating `io_seconds` ~N×). It should not be read as literal device bandwidth. The structural counts are the real signal.

### Why the split is favorable (measured + HF configs)

Dense is a small, fixed fraction of every modern fine-grained MoE, and it shrinks as models grow:

| model | total | dense (resident) | dense % | active/token | dense Q4 | dense Q8 |
|---|---|---|---|---|---|---|
| DeepSeek V4 Flash | 284B | 13.7 GiB (measured Q8) | ~9% | 6/256 | ~7.1 GB | 13.7 GB |
| Qwen3.5-397B-A17B | 403B | 16.9B | 4.2% | 10/512 | ~9.3 GB | ~17.9 GB |
| GLM-5.2 | 753B | 28.5B | 3.8% | 8/256 | ~15.7 GB | ~30.2 GB |
| MiniMax-M3 | 427B | 13.9B | 3.3% | 4/128 | ~7.6 GB | ~14.7 GB |
| Kimi-K2.7-Code | 1.06T | 43.9B | 4.1% | 8/384 | ~24.1 GB | (non-goal) |

All four target models' dense sets fit the rig; **Kimi-K2.7 is an explicit non-goal** (24 GB dense at Q4 leaves no expert-cache headroom).

---

## Architecture

Split the model by **tensor class** and place each class on the card suited to it.

- **Dense class** (`n_experts == 0`: MLA attention, shared expert, embeddings, output, norms) → **normal resident VRAM tensors** (not paged). Compute in place on the resident device.
- **Expert class** (`_exps` tensors) → registered with the pager, demand-streamed NVMe→VRAM into an expert-only cache pool on the paging device.

**Per-token data flow (multi-device):** input → resident device (6900XT) computes dense attention → hidden-state activation copied across TB3 (scheduler-inserted) → paging device (R9700) pages + computes routed experts → activation copied back for the next layer's attention → … → output. Only KB-scale activations cross TB3 (~0.7–1.5 ms/token measured); ~1–4 s/token of expert paging on the R9700 overlaps the 6900XT's attention compute. **The win is I/O-hiding + reclaimed cache, not compute parallelism** — layers are a sequential chain; the two cards do not compute the same token in parallel.

### Feasibility (verified in code this session)

1. **Cross-device scheduling is free.** `ggml_backend_sched` places each op on its weight tensor's device (`ggml/src/ggml-backend.cpp:909-930`, "operations with weights are preferably run on the same backend as the weights") and **auto-inserts cross-device activation copies** at split boundaries (`:1331-1360`). Same mechanism `--tensor-split` uses. No custom scheduler code.
2. **Per-tensor placement hook already exists.** `tensor_buft_overrides` (the `-ot`/`--override-tensor` feature, `src/llama-model-loader.cpp:1226-1251`) assigns any tensor an explicit device buft by name regex, bypassing the layer-range default (`src/llama-model.cpp:1329-1339`). This is the dense→B / expert→A router.
3. **The only real lift is the pager's single-device assumption.** `WeightPager::init` hard-rejects >1 device (`src/weight-pager/wp-pager.cpp:182-188`); one pool on one buft (`src/weight-pager/wp-pool.h:5-6`); the eval callback patches every paged src to that one `pool_buf` (`src/weight-pager/wp-eval-cb.cpp:1096, 1134-1144`). The pool header already notes "per-device pools are a drop-in extension."

---

## Components (independently testable)

### C1 — Tensor classifier
`is_dense(tensor_name, n_experts) -> bool`. Pure function: expert iff the tensor is a routed-expert `_exps` tensor (equivalently `n_experts > 0` on its page info). Dense = everything else. Unit-tested against a **real GGUF tensor list** (dumped via `gguf`), never invented names.

### C2 — Resident/paged loader split
At init, dense tensors load as normal resident VRAM buffers; only expert tensors enter `ml.weight_page_infos` (the paged catalog). Today `init_weight_pager` (`src/llama.cpp:169-171`) adds **every** tensor as a page — the change is to add **only expert** tensors and let dense load via the normal tensor path. Pager slot budget shrinks to `VRAM − dense_resident − KV − compute_reserve`.
- **Fail-loud guard:** assert every dense tensor landed resident and every expert tensor landed paged; abort with the exact byte/VRAM numbers on any misclassification or resident-device OOM. (No silent fall-through.)

### C3 — Device router (multi-device, Phase 2)
CLI `--weight-paging-resident-device <dev|auto>` emits `tensor_buft_overrides` entries: dense tensor names → resident-device buft, expert paged tensors → paging-device buft. `auto` = the non-paging GPU if present, else the paging card itself (collapses to Phase-1 single-card). Reuses the existing override mechanism at `src/llama-model-loader.cpp:1226-1251`.

### C4 — Per-device pager pools (multi-device, Phase 2)
Lift `PoolAllocator`/`WeightPager` from one pool to **per-device pools keyed by `ggml_backend_buffer_type_t`**; relax the `devices_used.size() > 1` guard (`wp-pager.cpp:182-188`); `pool_buf()`/`ensure()` select the pool matching the paged tensor's assigned device. The pool API already takes a buft (`wp-pool.h:6-7`).

### C5 — Expert-locality cache (Phase 3)
The pool becomes expert-only (dense no longer competes). Retention policy:
- Keep existing **within-layer sister prefetch** (MAD-88, `wp-eval-cb.cpp:792-824`) and **cross-layer speculative reuse prefetch** (MAD-233, `wp-eval-cb.cpp:826-910`, `WP_NEXT_LAYER_PREFETCH_K`).
- **New:** frequency/hotness-biased eviction so experts hot across a *generation* survive, layered on the existing LRU + hot-count (`wp-pool.*`). Instrument hit-rate.
- **Explicit non-goal:** true cross-layer routing *prediction*. Expert selection is data-dependent on each layer's own hidden state (router gate matmul consumes that layer's residual, `src/llama-graph.cpp:1908`); L+1's experts are unknowable until L completes. Only speculation (reuse locality) is possible.

---

## Implementation phases

- **Phase 1 — dense-resident split, single card (R9700).** C1 + C2. No cross-device work. **Gate:** page-ins/token ~1013 → ~258; decode ≥ ~0.4 t/s; coherent output; 0 GPU faults. Proves the thesis on the model already loaded.
- **Phase 2 — multi-device via TB3.** C3 + C4 + `--weight-paging-resident-device`. **Gate:** dense resident on 6900XT, full 32 GB R9700 as expert cache; coherent output with activations crossing TB3; then perf ≥ Phase-1.
- **Phase 3 — expert-locality cache tuning.** C5. **Gate:** expert-cache hit-rate rises materially above the ~5% baseline; decode climbs toward 1–3 t/s (model-dependent per the per-token GB table).

---

## CLI / config

- `--weight-paging-resident-device <dev|auto>` — where the dense set pins (Phase 2). Default `auto`.
- Existing `--device`, `--weight-paging`, `--weight-paging-slots`, `--weight-paging-prefetch`, `LLAMA_WP_TRANSPORT`, `WP_*` env gates unchanged.
- **Gate:** the dense-resident split is enabled by `WP_RESIDENT_DENSE=1` (default **off** initially, consistent with the other `WP_*` features; flipped to default-on after the Phase-1 gate passes and the resident/paged classification is validated on all four target models). With the gate off, the pager keeps today's all-paged behavior exactly. With it on and a single device, Phase 1 applies; with it on plus `--weight-paging-resident-device`, Phase 2 applies.

## Error handling

- Resident-device OOM → abort with `dense_size` vs free-VRAM numbers (fail loud).
- Resident device absent / TB3 disconnected at init → fall back to single-card resident split on the paging device (Phase 1 semantics), warn.
- Any dense tensor failing to load resident → abort; never silently page it.
- Misclassification (dense landed paged, or expert landed resident) → abort at the C2 guard.

## Testing / validation

- **C1:** classifier unit tests over a real GGUF tensor-name fixture (dense vs `_exps`).
- **C2:** init-time assertion that placement matches classification (fail-loud); Phase-1 perf gate (page-ins/token, decode t/s, faults).
- **C3/C4:** multi-device correctness (coherent output, activation copies observed crossing the device boundary), per-device pool accounting; then perf.
- **C5:** hit-rate instrumentation; A/B of retention policy vs plain LRU on a fixed prompt.
- Existing `tests/test-weight-pager.cpp` suite must stay green throughout (37/37 today).

## Non-goals

- Kimi-K2.7-Code (dense too large for the rig).
- True predictive (non-speculative) cross-layer expert prefetch (physically impossible — data-dependent routing).
- Changing the quant of the loaded GGUF (dense size follows whatever quant the GGUF ships).

## Key file references

- Scheduler placement + auto-copy: `ggml/src/ggml-backend.cpp:845-933, 1014-1360`.
- Per-tensor buft override: `src/llama-model-loader.cpp:1102-1258` (override at `:1226-1251`); layer-range default `src/llama-model.cpp:1329-1352`.
- Pager init / single-device guard / pool: `src/weight-pager/wp-pager.cpp:166-232` (guard `:182-188`), `src/weight-pager/wp-pool.h:5-14`.
- Pager eval callback / paged-src patching / prefetch: `src/weight-pager/wp-eval-cb.cpp:453, 498, 569-596, 696-719, 787-789, 792-824, 826-910, 1096, 1134-1148`.
- Pager registration: `src/llama-context.cpp:1364-1368`.
- init_weight_pager (add_page loop): `src/llama.cpp:169-171`.
- MoE graph (router → top-k → expert matmuls): `src/llama-graph.cpp:1876, 1908, 1992, 2054/2073/2086/2175`.
- io_effective accounting artifact: `src/weight-pager/wp-pager.cpp:958-970`.
