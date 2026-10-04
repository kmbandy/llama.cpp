# Cross-Device Flash Attention via Attention-Island (Home-Device Inversion) — Design

**Date:** 2026-07-07
**Status:** Approved for planning
**Parent design:** `docs/superpowers/specs/2026-07-07-moe-resident-split-multidevice-design.md` (this supersedes that doc's **Phase 2** section)
**Branch:** `feat/wp-attention-island` (off `feat/dsws-phaseb-conversion`)

## Goal

Run the MoE-aware resident/paged split across **two GPUs** with Flash Attention **enabled**, so that dense attention lives on the small TB3 eGPU (ROCm1 / 6900XT / 16 GB) and routed experts demand-page on the big card (ROCm0 / R9700 / 32 GB). This frees ROCm0 VRAM for a larger expert page-cache — the lever against the expert-cache thrash identified as the Phase-1 bottleneck — **without** triggering the non-Flash-Attention path.

## Background & Root Cause

The original Phase 2 (`feat/dsws-phaseb-conversion`) placed each layer's *home* device on the **paging** card (ROCm0) and used a greedy `.*` tensor-buft override to drag dense weights *over* to the resident card (ROCm1). Result on GPU (`wp_logs/dsv4-p2host.log`):

```
sched_reserve: layer 0 is assigned to device ROCm0 but the Flash Attention tensor
               is assigned to device ROCm1 (usually due to missing support)
sched_reserve: Flash Attention was auto, set to disabled
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 40288.31 MiB on device 1: out of memory
```

Failure chain, confirmed by code read:

1. The `.*` override moves attention **weights** to ROCm1, so the ggml scheduler's weight-affinity rule (`ggml/src/ggml-backend.cpp:908-929`) pulls the **FA node** onto ROCm1.
2. But **KV cache follows `dev_layer(il)`** (`src/llama-kv-cache.cpp:261-262`: `dev = model.dev_layer(il)`), which is still ROCm0.
3. `llama_context::sched_reserve()` (`src/llama-context.cpp:521-523`) disables auto-FA whenever the FA node's device ≠ `dev_layer(il)`. Here ROCm1 ≠ ROCm0 → **FA disabled**.
4. The non-FA attention path (`src/llama-graph.cpp:2707-2752`) materializes the full KQ score matrix → a ~40 GB compute buffer → OOM/GPU-fault on the 16 GB card.

Genuine cross-device Flash Attention (an FA kernel reading K/V from a peer GPU) is **not** the solution: the HIP/CUDA backend rejects FA inputs whose source buffers live on another device (`ggml/src/ggml-cuda/ggml-cuda.cu:5625-5633`, `:6138-6140`). That path (parent-doc "Option C") is high-risk and out of scope.

**Key insight — the design was inverted.** If instead each layer's *home* device is the **resident/attention** card (ROCm1), and only the routed experts are overridden *out* to the **paging** card (ROCm0), then attention weights, KV cache, and the FA node all sit on ROCm1. `device_fa == dev_layer(il)` becomes true → FA stays enabled → the 40 GB buffer is never built. No `sched_reserve` rewrite and no explicit `ggml_backend_sched_set_tensor_backend()` surgery are required. The two attention/expert "islands" form naturally from weight-affinity; the only cross-device traffic is the residual/hidden vector at the attn↔MoE boundary, which the scheduler already copies at split boundaries (`ggml/src/ggml-backend.cpp:1352-1370`).

This is the parent doc's "Option A" **outcome** (intra-device attention island) reached by a strictly simpler **mechanism** than the parent doc anticipated.

## Feasibility (already demonstrated)

In the failing run the model **loaded successfully** — 1199 tensors on ROCm1 (attention, `token_embd` ≈ 1.0 GB, `output` ≈ 1.0 GB, norms, shared expert) fit within the 16 GB card. The *only* allocation that failed was the non-FA 40 GB compute buffer. So this is a "keep FA on" problem, not a "make it fit" problem. With FA enabled the compute buffer is small.

DeepSeek V4 Flash uses **MLA** (compressed `attn_kv_a`/`attn_kv_b` latent), so the KV cache placed on ROCm1 is small. The same holds for the other MLA/GLA arches under consideration.

## Architecture

Two device islands, per offloaded layer, ×43:

- **ROCm1 (resident / attention home):** attention + MLA weights, KV cache, FA node, `attn_norm`/`ffn_norm`, shared expert (`ffn_*_shexp`), router (`ffn_gate_inp`), `token_embd`, `output`, `output_norm`, `output_hc_*`.
- **ROCm0 (paging):** routed experts only — `blk.N.ffn_(up|gate|down)_exps.weight` — demand-paged through the single WeightPager pool.

Decode data flow (per layer):

```
input ─▶ [ROCm1] attn: Qkv proj → RoPE → KV write (ROCm1 cache) → FLASH_ATTN_EXT → attn_out
      ─▶ residual  ── copy ROCm1→ROCm0 ─▶  [ROCm0] router → expert MUL_MAT_ID (paged) → expert_out
      ── copy ROCm0→ROCm1 ─▶  next layer
```

The residual crossing is ~`hidden_size` × dtype per token (≈ 8 KB/token at hidden=4096, Q8-adjacent activations in F16) — negligible bandwidth on TB3; the concern is per-layer round-trip latency, measured in S3.

## Components (change set)

### C1 — Home-device inversion (the one load-bearing change)
When the WP device-router is active (`WP_RESIDENT_DENSE` + a distinct resident device selected), `dev_layer(il)` must return the **resident** device for every offloaded (non-CPU) layer, instead of the paging device.

- Mechanism: in `src/llama-model.cpp` `load_tensors()` where `get_layer_buft_list()` / `pimpl->dev_layer[]` are assigned (~`:1436-1459`), when `wp_device_router_enabled`, pin the offloaded-layer home device to the resident device (`wp_select_resident_device_index`) rather than the tensor-split result.
- Validation shortcut (no code): forcing `--tensor-split` so all layers land on the resident device reproduces the same `dev_layer` outcome, enabling an S1 smoke test before C1 is written.

### C2 — Override simplification
In the router-override synthesis (`src/llama-model.cpp:1359-1384`):
- **Remove** the greedy dense override `{".*", wp_resident_buft}`.
- **Keep only** the expert override `{"ffn_(up|gate|down)_exps\\.", wp_paging_buft}`.

With C1 making the home device ROCm1, attention/dense weights default to the resident buft with no override, and only experts need redirecting. This also resolves the parent review's Medium finding "WP overrides shadow user overrides" (the `.*` was the culprit) and the Medium "page_buft imprecise" concern is unaffected because paged tensors remain single-device.

### C3 — Pager stays single-device
Paging device = ROCm0; all paged tensors (experts) live on ROCm0 → one pool keyed by the ROCm0 buft. No per-device pools. The existing `init_weight_pager` single-paged-device guard (`src/llama.cpp:144-155`) is **satisfied**, not relaxed. Update the stale guard comment/log to state the single-paged-device contract explicitly (parent review High #1 → downgraded to a doc/comment fix under this design).

### C4 — Fail-loud safety net
After `sched_reserve()` resolves auto-FA, if the WP device-router is active across ≥2 devices **and** FA resolved to *disabled*, raise a hard error (do **not** silently fall through to the non-FA 40 GB path). Message must name the offending layer and the FA-device vs KV-device mismatch, and point at C1. This guarantees we never re-enter the brick-the-GPU failure mode silently.

### C5 — Correctness prerequisites (separate branch `feat/wp-md-correctness`)
Two device-placement bugs from the parent review, latent on single-card, fatal on two-card:
- **C5a (High):** routed-expert pointer table (`s_dev_expert_ptrs`, `src/weight-pager/wp-eval-cb.cpp:506-514`) is `hipMalloc`'d on the ambient HIP device with no `hipSetDevice()`. Must be allocated/filled on the device that executes `GGML_OP_MUL_MAT_ID` (ROCm0). No-op on single-card.
- **C5b (Medium):** slot auto-size (`src/llama.cpp:225-249`) calls `hipMemGetInfo()` on the ambient device; must query the parsed paging device. No-op on single-card.

C5 must land and merge into `feat/wp-attention-island` **before any two-card GPU run, including S1** — S1 flips the layer home to ROCm1 while experts execute on ROCm0, which is exactly the two-device condition C5a guards.

## Validation Strategy (staged, GPU-fault-cautious)

The rig has faulted twice this cycle; every GPU stage starts small and checks logs before scaling. All inference runs use `--no-mmap` and require explicit user go-ahead per standing rule.

- **S0 — CPU unit tests** stay green (37+ existing) throughout.
- **S1 — zero-code hypothesis smoke test:** with C5 landed, force all layers to ROCm1 via `--tensor-split`, **short context (e.g. 512) and a handful of tokens**. Pass = log shows `Flash Attention ... set to enabled`, **no** 40 GB allocation, coherent tokens, 0 GPU faults. Fail here ⇒ the mechanism is wrong; stop before writing C1.
- **S2 — feature run:** implement C1+C2+C4, re-run the S1 configuration via `--weight-paging-resident-device` (no `--tensor-split` hack). Same pass criteria.
- **S3 — performance run:** full context, size the expert pool to the freed ROCm0 VRAM (~2× Phase 1's 8.5 GB pool), measure decode t/s + page-in/eviction/prefetch-hit stats. Compare against Phase 1 baseline (1.038 t/s, P2P depth-4). Success target: FA enabled, 0 faults, coherent, and decode ≥ Phase-1 baseline; **open question** whether the larger pool breaks the thrash (this run is what answers it).

## Error Handling

- **FA silently disabled** → C4 hard-errors (never build the 40 GB buffer).
- **Home-inversion assumption break** (something expects home == paging device) → surfaced by the S1 smoke test before feature code; if the pager or eval-cb asserts, that assertion is the signal to investigate, not to suppress.
- **Resident device absent / single-GPU** → router collapses to Phase-1 single-card behavior (home == paging == the one card); no inversion, `.*`-free override set is a no-op superset. Existing Phase-1 path unchanged.

## Testing

- **Unit (CPU, `tests/test-weight-pager.cpp` + model-loader coverage where feasible):**
  - Router-active override synthesis produces the **expert-only** override set (no `.*` entry), and preserves any user `tensor_buft_overrides` after it.
  - Device-selection helpers (`wp_select_resident_device_index` / `wp_select_paging_device_index`) return distinct indices for a 2-GPU list and collapse to the single index for a 1-GPU list.
- **GPU:** staged S1–S3 above (manual, user-gated).

## Risks & Open Questions

1. **Residual round-trip latency** across TB3 ×43 layers ×2 crossings/layer — bandwidth is negligible, latency is the unknown; quantified in S3.
2. **Does the freed VRAM actually break the thrash?** Phase-1's "bigger cache didn't help" was confounded by dense+pool+KV competing for one 32 GB card; S3 removes that confound. Genuinely open until measured.
3. **`dev_layer` inversion side effects** — any code assuming the layer home is the compute/paging device. Primary mitigation: S1 smoke test surfaces it cheaply before feature code.

## Out of Scope

- Genuine peer-device FA (parent "Option C").
- N-device (>2) generalization / per-device paging pools (parent High #1 kept as single-device contract).
- Expert-locality cache policy changes (parent Phase 3).
