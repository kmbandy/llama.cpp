# MoE Resident/Paged Split + Multi-Device + Expert Cache — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Execution model for this plan: Codex implements each task; Claude reviews each task before it is marked complete.**

**Goal:** Keep the small dense weight set VRAM-resident and stream only the sparse routed experts, so frontier fine-grained MoE models run fast on a dual-GPU rig — lifting DeepSeek V4 Flash Q8 paged decode from 0.02 t/s toward 0.5–3 t/s.

**Architecture:** Split model tensors by class — dense (`n_experts==1` at load / `!is_expert` in the catalog: MLA attention, shared expert, embeddings, output, norms) load as normal resident VRAM buffers; only routed-expert (`_exps`) tensors register with the pager and demand-stream from NVMe. Phase 1 does this single-card; Phase 2 places dense on a second GPU via per-tensor buft overrides + per-device pager pools; Phase 3 adds frequency-biased expert-cache retention.

**Tech Stack:** C++17, ggml/ggml-hip (ROCm), llama.cpp weight pager (`src/weight-pager/`), `tests/test-weight-pager.cpp` (bespoke no-framework unit harness). Build+run on mad-lab-main `~/llama-wp/build-hip`.

## Global Constraints

- Feature gate: `WP_RESIDENT_DENSE=1` enables the dense-resident split; **default off**. Gate off ⇒ byte-for-byte today's all-paged behavior.
- Always pass `--no-mmap` to llama.cpp tools. Never run inference without the user's go-ahead.
- Remote GPU box shell is fish: wrap as `ssh mad-lab-main bash -s <<'EOF' … EOF`.
- Do NOT touch pre-existing WIP: `examples/pagedattn-*`, `src/llama-graph.cpp` (unrelated edits), `examples/CMakeLists.txt`. Do NOT clobber mad-lab-main's `feat/dsws-phaseb-conversion` — use the `~/llama-wp` worktree.
- Commit only the files a task names. Co-author trailer + Claude-Session trailer on every commit (see existing history).
- Expert classification is name-based and already computed at `src/llama-model.cpp:1774-1798` (`is_consolidated` from `ffn_up_exps.`/`ffn_gate_exps.`/`ffn_down_exps.`). Reuse it; do NOT invent new name patterns. Ground every name/pattern table in a real GGUF dump.
- The pager catalog already exposes classification: `PageMeta.is_expert`, `PageCatalog::has_experts()`, `n_expert_pages()` (`src/weight-pager/wp-page-catalog.h:39,118,121`). `add_pinned` (MAD-236, `:101`) is the existing resident-page API.
- Unit suite must stay green (37/37 today): `./build-hip/bin/test-weight-pager`.

---

## Phase 1 — Single-card dense-resident split

Foundational and independently valuable. Proves the thesis on the model already loaded, no cross-device work.

### File structure (Phase 1)

- `src/llama-model.cpp:1762-1804` — the per-tensor `weight_page_info` push. **Filter point:** under the gate, page only expert tensors; dense tensors skip paging and load via the normal allocator.
- `src/llama-model-loader.h:33-44` — `llama_weight_page_info`. Add an `is_expert` bool (derived from the existing `is_consolidated` detection) so the filter reads one flag, not a re-parse.
- `src/llama.cpp:96-211` — `init_weight_pager`: slot-budget accounting must reserve dense-resident VRAM; add the fail-loud placement guard + a resident/paged telemetry line.
- `tests/test-weight-pager.cpp` — unit tests for the classifier flag + a catalog-level "only experts paged" assertion.
- `src/weight-pager/wp-eval-cb.cpp` — no change expected in Phase 1 (dense tensors are simply absent from the catalog; the callback already no-ops on non-catalog tensors). Verify, don't edit.

### Task 1: Add `is_expert` to `llama_weight_page_info` and set it at load

**Files:**
- Modify: `src/llama-model-loader.h:33-44`
- Modify: `src/llama-model.cpp:1762-1799`

**Interfaces:**
- Produces: `llama_weight_page_info::is_expert` (bool, default false) — true iff the tensor is a routed-expert consolidated tensor. Consumed by Task 2 (filter) and Task 3 (budget/guard).

- [ ] **Step 1: Add the field.** In `src/llama-model-loader.h` inside `struct llama_weight_page_info` (after `int n_experts = 1;`):

```cpp
    // True iff this is a routed-expert (consolidated ffn_*_exps) tensor.
    // Set at load from the same name detection that computes n_experts.
    // Dense tensors (attention, shared expert, embeddings, norms) are false.
    bool is_expert = false;
```

- [ ] **Step 2: Set it where `is_consolidated` is already computed.** In `src/llama-model.cpp`, the block at `:1774-1797` already computes `bool is_consolidated`. Immediately before `ml.weight_page_infos.push_back(info);` (`:1799`), add:

```cpp
                    info.is_expert = is_consolidated;
```

Note: `is_consolidated` is declared inside the inner `{ … }` scope at `:1777`. Move the `info.is_expert = is_consolidated;` assignment to the last line **inside** that inner block (right after the `if (n_exp > 1) { info.n_experts = n_exp; }`), so it is in scope.

- [ ] **Step 3: Build the loader TU only** (fast compile check), on mad-lab-main:

```
ssh mad-lab-main bash -s <<'EOF'
cd ~/llama-wp/build-hip && cmake --build . --target llama -j4 2>&1 | tail -3
EOF
```
Expected: `Built target llama`, no errors.

- [ ] **Step 4: Commit.**

```
git add src/llama-model-loader.h src/llama-model.cpp
git commit -m "feat(wp): tag routed-expert tensors with is_expert at load"
```

### Task 2: Gate BOTH filters on `WP_RESIDENT_DENSE` (revised after code trace)

> **Correction (implemented as `07874dba3`):** The original single-filter plan
> below was WRONG and would have produced garbage. Weight-paging has **two
> independent filters** that must agree: `is_paged_weight()` (`llama-model.cpp:1608-1616`,
> the load-bearing one — it drives the manual per-tensor resident allocator at
> `:1624-1663`) and the catalog population (`:1800`). Touching only the catalog
> leaves dense tensors `is_paged_weight==true` ⇒ no buffer ⇒ `load_all_data`
> skips them (`llama-model-loader.cpp:1607` skips `data==NULL`) ⇒
> never-allocated-never-loaded. The stale comment at `:1822-1833` ("we page
> everything") is false — the manual allocator already keeps token_embd/output/
> norms/tiny resident; this feature just widens that set. **Actual change:** a
> shared `wp_is_routed_expert(name)` predicate + `wp_resident_dense_enabled()`,
> added to BOTH `is_paged_weight()` (dense → return false → resident) AND the
> catalog push (`page_this = !gate || info.is_expert`). `init_weight_pager` runs
> AFTER `load_tensors` (`llama.cpp:604-613`), so the pager pool auto-sizes to the
> VRAM remaining after dense is resident.

**Original (superseded) single-filter plan:**

**Files:**
- Modify: `src/llama-model.cpp:1799-1803` (the push into `weight_page_infos` + `weight_tensor_ptrs`)

**Interfaces:**
- Consumes: `info.is_expert` (Task 1).
- Produces: when `WP_RESIDENT_DENSE=1`, `ml.weight_page_infos` and `weight_pager->weight_tensor_ptrs` contain **only** expert tensors; dense tensors are left to the normal allocator.

- [ ] **Step 1: Add a gate helper near the top of `src/llama-model.cpp`** (next to other `getenv` helpers; if none, add a file-local static):

```cpp
static bool wp_resident_dense_enabled() {
    const char * v = std::getenv("WP_RESIDENT_DENSE");
    return v != nullptr && v[0] == '1';
}
```

- [ ] **Step 2: Filter the push.** Replace `:1799-1803`:

```cpp
                ml.weight_page_infos.push_back(info);
                // Collect the actual model tensor pointer for the weight pager
                if (weight_pager) {
                    weight_pager->weight_tensor_ptrs.push_back(t);
                }
```

with:

```cpp
                // WP_RESIDENT_DENSE: page ONLY routed-expert tensors; dense
                // tensors (attention, shared expert, embeddings, norms) fall
                // through to the normal buffer allocator and stay VRAM-resident.
                const bool page_this = !wp_resident_dense_enabled() || info.is_expert;
                if (page_this) {
                    ml.weight_page_infos.push_back(info);
                    if (weight_pager) {
                        weight_pager->weight_tensor_ptrs.push_back(t);
                    }
                }
```

- [ ] **Step 3: Build `llama` target** (as Task 1 Step 3). Expected: clean.

- [ ] **Step 4: Sanity-log check (no run yet).** Grep confirms the gate compiles into the catalog-population path — verify by reading back the diff; expected: dense tensors excluded when gate on.

- [ ] **Step 5: Commit.**

```
git add src/llama-model.cpp
git commit -m "feat(wp): WP_RESIDENT_DENSE gates paging to expert tensors only"
```

### Task 3: Fail-loud placement guard + resident/paged telemetry in init_weight_pager

**Files:**
- Modify: `src/llama.cpp:96-211` (`init_weight_pager`)

**Interfaces:**
- Consumes: `ml.weight_page_infos` (expert-only when gated), the catalog's `n_expert_pages()`/`size()`.
- Produces: an init log line `resident_dense=<on|off> paged_pages=<N> (experts=<E>) dense_resident=<skipped>`, and a hard abort if the gate is on but any non-expert page slipped into the catalog.

- [ ] **Step 1: After the catalog is populated (`src/llama.cpp:169-173`), add the guard + telemetry:**

```cpp
    if (wp_resident_dense_enabled_llama()) {
        // Under the resident-dense split the catalog must contain ONLY expert
        // pages. A dense page here means the filter (llama-model.cpp) missed a
        // tensor — fail loud rather than silently thrash it.
        const int n_pages   = model.wp_pager->n_pages();
        const int n_experts = model.wp_pager->catalog_n_expert_pages();
        if (n_pages != n_experts) {
            throw std::runtime_error(format(
                "weight pager: WP_RESIDENT_DENSE on but catalog has %d non-expert "
                "pages (n_pages=%d, expert_pages=%d) — dense filter missed tensors",
                n_pages - n_experts, n_pages, n_experts));
        }
        LLAMA_LOG_WARN("%s: resident_dense=ON  paged_pages=%d (all experts)  "
                       "dense tensors left resident to normal allocator\n",
                       __func__, n_pages);
    }
```

- [ ] **Step 2: Add the two accessors this needs.** `wp_resident_dense_enabled_llama()` — a file-local `getenv("WP_RESIDENT_DENSE")=="1"` in `src/llama.cpp` (mirror of Task 2's helper; DRY note: both are one-liners over the same env, acceptable per file locality). `WeightPager::catalog_n_expert_pages()` — add to `src/weight-pager/wp-pager.h` a `int catalog_n_expert_pages() const { return catalog_.n_expert_pages(); }` inline accessor (the catalog method already exists at `wp-page-catalog.h:121`).

- [ ] **Step 3: Build `llama` target.** Expected: clean.

- [ ] **Step 4: Commit.**

```
git add src/llama.cpp src/weight-pager/wp-pager.h
git commit -m "feat(wp): fail-loud guard + telemetry for resident-dense split"
```

### Task 4: Unit test — only experts paged when gated

**Files:**
- Modify: `tests/test-weight-pager.cpp` (add one test + register in `main`)

**Interfaces:**
- Consumes: `wp::PageCatalog` (`add`, `add_consolidated_experts`, `has_experts`, `n_expert_pages`, `at`).

- [ ] **Step 1: Write the failing test.** Mirror the existing catalog tests (e.g. `test_page_catalog_moe_classification`). The test asserts the classifier invariant the loader filter relies on: a consolidated expert tensor is `is_expert`, a dense tensor is not.

```cpp
static int test_catalog_is_expert_classification() {
    int fails = 0;
    wp::PageCatalog cat;
    // Dense tensors — must NOT be experts.
    int p_attn = cat.add("blk.0.attn_q.weight",      0, 0,   4096);
    int p_emb  = cat.add("token_embd.weight",        0, 0, 100000);
    int p_shex = cat.add("blk.0.ffn_down_shexp.weight", 0, 0, 8192);
    // Consolidated routed experts — MUST be experts.
    int first_sub = cat.add_consolidated_experts("blk.0.ffn_gate_exps.weight",
                                                 0, 0, 256*8192, 256);
    EXPECT(!cat.at(p_attn).is_expert, "attn_q is dense");
    EXPECT(!cat.at(p_emb).is_expert,  "token_embd is dense");
    EXPECT(!cat.at(p_shex).is_expert, "ffn_down_shexp is dense (shared expert)");
    EXPECT(cat.at(first_sub).is_expert, "ffn_gate_exps sub-page is expert");
    EXPECT(cat.has_experts(), "catalog reports experts present");
    return fails;
}
```

Note: confirm `ffn_down_shexp` (shared expert) is classified dense by the loader detection at `llama-model.cpp:1780-1782` — it matches `ffn_down_` but NOT `ffn_down_exps.`, so `is_consolidated` is false ⇒ dense. This test locks that in. If the catalog's own `add()` name-parse disagrees with the loader detection, **that discrepancy is a real bug to surface**, not to paper over.

- [ ] **Step 2: Register in `main`** (`tests/test-weight-pager.cpp` `named_test tests[]`):

```cpp
        { "catalog_is_expert_classification", test_catalog_is_expert_classification },
```

- [ ] **Step 3: Run — expect FAIL if `is_expert` isn't set on sub-pages yet, else PASS.**

```
ssh mad-lab-main bash -s <<'EOF'
cd ~/llama-wp/build-hip && cmake --build . --target test-weight-pager -j4 >/dev/null 2>&1 && ./bin/test-weight-pager 2>&1 | grep -E "is_expert_classification|total failures"
EOF
```
Expected: `PASS test_catalog_is_expert_classification`.

- [ ] **Step 4: Commit.**

```
git add tests/test-weight-pager.cpp
git commit -m "test(wp): catalog is_expert classification (dense vs routed)"
```

### Task 5: VRAM budget — VERIFY ordering (likely no code, per Task-2 trace)

> **Simplified after the Task-2 code trace.** `init_weight_pager` (which sizes
> the pager pool from free VRAM) runs AFTER `load_tensors` allocates the manual
> resident buffer that now includes all dense tensors. So `hipMemGetInfo` at
> pool-sizing time already reports VRAM net of dense-resident — the pool
> auto-sizes to what's left. This task reduces to: confirm via the Phase-1 gate
> run that free_vram at pool sizing excludes dense (log line), and that
> dense+pool+KV fit. Add the subtraction below ONLY if the gate shows the pool
> over-allocating. Do not add speculative code.

**Contingency (only if the gate shows over-allocation):**

**Files:**
- Modify: `src/llama.cpp:184-211` (the auto slot-budget block)

**Interfaces:**
- Consumes: the gate; free-VRAM query already present at `:195-196`.

- [ ] **Step 1: The problem.** With the gate on, dense tensors load to VRAM via the normal allocator. If `init_weight_pager` runs its `hipMemGetInfo` **after** dense is allocated, `free_vram` already excludes dense — good. If it runs **before**, the pool would over-allocate. Verify ordering by reading the call site of `init_weight_pager` relative to `load_tensors`. **First implementation step is to confirm ordering (add a one-line `LLAMA_LOG_INFO("%s: free_vram=%zu MiB at pool sizing\n", __func__, free_vram/1048576)` and read it in the Phase-1 gate run).**

- [ ] **Step 2: If pool sizing runs before dense load** (free_vram too high): subtract the known dense-resident bytes. The dense byte total = sum of `ggml_nbytes` over non-expert weight tensors; expose it from the loader as `ml.dense_resident_bytes` (populate in the same loop at `llama-model.cpp:1799` when `!page_this`). Then in `src/llama.cpp` before `n_slots_fit`:

```cpp
        size_t dense_reserve = wp_resident_dense_enabled_llama() ? ml.dense_resident_bytes : 0;
        size_t usable2 = (usable > dense_reserve) ? (usable - dense_reserve) : 0;
        const int n_slots_fit = (max_page_size > 0) ? (int)(usable2 / max_page_size) : 0;
```

- [ ] **Step 3: If ordering already excludes dense** (free_vram already net): no subtraction needed; document it with a comment and skip the reserve. **Pick exactly one path based on Step 1's evidence; do not guess.**

- [ ] **Step 4: Build + commit.**

```
git add src/llama.cpp src/llama-model.cpp src/llama-model-loader.h
git commit -m "feat(wp): reserve dense-resident VRAM before sizing expert pool"
```

### Task 6: Phase-1 integration gate (measured on mad-lab-main)

**This task is a measurement + acceptance gate, not code. Requires the user's go-ahead to run inference.**

- [ ] **Step 1: Build the server.** `cmake --build ~/llama-wp/build-hip --target llama-server -j4`.
- [ ] **Step 2: Baseline (gate OFF).** Run the existing recipe (systemd-run, DeepSeek V4 Flash, `WP_ENSURE_BATCH=1`, depth 8), capture pager stats at clean SIGINT. Expect ~1013 page-ins/token, ~0.02 t/s (confirms no regression to baseline).
- [ ] **Step 3: Gate ON.** Same recipe + `--setenv=WP_RESIDENT_DENSE=1`. Capture stats.
- [ ] **Step 4: Acceptance gate.** PASS requires: page-ins/token drops to ≈ the routed-expert count (≈258 for DeepSeek), `sync_fallbacks` falls sharply, coherent output, 0 GPU faults, decode ≥ ~0.4 t/s. Record numbers in `docs/dev/weight-paging-batch-eval-continuation.md`.
- [ ] **Step 5: Commit the doc update.** (No code.)

---

## Phase 2 — Multi-device via Thunderbolt 3 (scoped; detailed after Phase 1)

> These tasks are scoped with interfaces + anchors + gates. They are expanded into bite-sized TDD steps **after Phase 1 lands**, because the exact code depends on Phase 1's resident-load path. Listed here so the whole design is visible.

- **Task 7 — `--weight-paging-resident-device <dev|auto>` CLI flag.** Add to `common/arg.cpp` + `llama_model_params`. `auto` = the non-paging GPU if present else the paging card. Anchor: existing `--device`/`--weight-paging*` flags. **Gate:** flag parses, threads to `init_weight_pager`.
- **Task 8 — Emit `tensor_buft_overrides` for dense→resident-device, expert→paging-device.** Under gate + resident-device set, synthesize override entries (dense tensor-name regex → resident buft; expert `_exps` → paging buft) reusing `src/llama-model-loader.cpp:1226-1251`. **Gate:** dense tensors land on the resident device buffer, experts on the paging device (assert at load); scheduler auto-inserts activation copies (`ggml-backend.cpp:1331-1360`) — verified by a coherent multi-device forward.
- **Task 9 — Lift pager to per-device pools.** `PoolAllocator`/`WeightPager` keyed by `ggml_backend_buffer_type_t`; relax the `devices_used.size() > 1` guard (`wp-pager.cpp:182-188`); `pool_buf()`/`ensure()` select the pool for the paged tensor's device. Anchor: `wp-pool.h:5-14` ("per-device pools are a drop-in extension"). **Gate:** unit test for two-pool allocation; multi-device paged run coherent.
- **Task 10 — Phase-2 integration gate.** Dense on 6900XT (ROCm1), full 32 GB R9700 (ROCm0) as expert cache; coherent output; decode ≥ Phase-1.

## Phase 3 — Expert-locality cache (scoped; detailed after Phase 2)

- **Task 11 — Hit-rate instrumentation** for the expert-only pool (per-generation reuse counter). Anchor: `wp-pool.*` LRU + hot-count, existing `prefetch_hit_rate` telemetry.
- **Task 12 — Frequency/hotness-biased retention.** Bias eviction so generation-hot experts survive, layered on the existing LRU + hot-count. Keep MAD-88 sister prefetch (`wp-eval-cb.cpp:792-824`) and MAD-233 speculative reuse (`:826-910`) as-is. **Gate:** A/B hit-rate vs plain LRU on a fixed prompt shows improvement.
- **Task 13 — Phase-3 integration gate.** Expert-cache hit-rate materially above ~5% baseline; decode climbs toward the model's per-token-bytes ceiling (1–3 t/s for the small-per-token models).

## Non-goals (from spec)

Kimi-K2.7-Code (dense too large); true cross-layer routing prediction (data-dependent, impossible); changing the GGUF quant.
