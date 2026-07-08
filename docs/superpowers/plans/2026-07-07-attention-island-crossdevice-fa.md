# Attention-Island (Cross-Device Flash Attention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the MoE resident/paged split across two GPUs with Flash Attention enabled, by making each offloaded layer's *home* device the resident/attention card (ROCm1) and overriding only routed experts out to the paging card (ROCm0).

**Architecture:** Invert the current Phase-2 placement. Instead of layer-home = paging card + a greedy `.*` override dragging dense weights to the resident card, set layer-home = resident card and override *only* `ffn_(up|gate|down)_exps` to the paging card. Attention weights, KV cache, and the FA node then all live on the resident card, so `sched_reserve`'s `device_fa == dev_layer(il)` check passes and FA stays enabled — no scheduler surgery, no 40 GB non-FA buffer. The residual stream crosses TB3 at the attn↔MoE boundary via the scheduler's existing split-boundary auto-copy.

**Tech Stack:** C++17, llama.cpp fork, ggml/ggml-backend scheduler, HIP/ROCm (gfx1201 R9700 + gfx1030 6900XT), io_uring weight pager. CPU unit tests via the bespoke `EXPECT`/`EXPECT_EQ_INT` harness in `tests/test-weight-pager.cpp`.

## Global Constraints

- Branch: `feat/wp-attention-island`, based on `feat/dsws-phaseb-conversion`, on host `mad-lab-main` at `/home/kmbandy/GitHub/llama.cpp`.
- Spec: `docs/superpowers/specs/2026-07-07-attention-island-crossdevice-fa-design.md`.
- Routed-expert tensor pattern (the ONLY paged/overridden pattern): `ffn_(up|gate|down)_exps\.` (regex; matches `blk.N.ffn_up_exps.weight` etc.). Must stay identical to the paging catalog / `is_paged_weight` filter.
- No LLM inference in implementation tasks. CPU unit tests only: `cmake -S . -B build-cpu -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release` then `cmake --build build-cpu --target test-weight-pager -j"$(nproc)"`. All existing tests stay green (37+ today).
- GPU validation (S1–S3) is manual, user-gated, and always uses `--no-mmap`. The build for GPU runs is the single dual-arch `build-hip` (gfx1201;gfx1030) — never single-arch.
- Commit on the branch when a task compiles + unit tests pass. Do NOT push.
- C5 prerequisites (below) must merge into this branch before any two-card GPU run (S1 included).

---

## Prerequisites (separate branch — not tasks in this plan)

**C5 device-placement correctness fixes** are being implemented by Codex on branch `feat/wp-md-correctness` (off `feat/dsws-phaseb-conversion`):
- **C5a (High):** routed-expert pointer table (`s_dev_expert_ptrs`, `src/weight-pager/wp-eval-cb.cpp:506-514`) must be `hipMalloc`'d + filled on the device that executes `GGML_OP_MUL_MAT_ID`, not the ambient HIP device.
- **C5b (Medium):** slot auto-size (`src/llama.cpp:225-249`) must `hipSetDevice(paging_device_idx)` around `hipMemGetInfo()`.

**Merge gate:** before the first two-card GPU run (S1), merge `feat/wp-md-correctness` into `feat/wp-attention-island` (`git merge feat/wp-md-correctness`). S1 flips layer-home to ROCm1 while experts execute on ROCm0 — exactly the two-device condition C5a guards.

---

## Task 1: Extract testable router-override builder

**Files:**
- Create: `src/weight-pager/wp-router.h`
- Create: `src/weight-pager/wp-router.cpp` (auto-compiled via `file(GLOB weight-pager/*.cpp)` in `src/CMakeLists.txt:10` — no CMake edit needed; a fresh `cmake` configure picks it up)
- Test: `tests/test-weight-pager.cpp` (add two tests + register them)

**Interfaces:**
- Produces:
  - `extern const char * const wp::ROUTER_EXPERT_PATTERN;` (value `"ffn_(up|gate|down)_exps\\."`)
  - `std::vector<llama_model_tensor_buft_override> wp::build_router_overrides(ggml_backend_buffer_type_t paging_buft, const llama_model_tensor_buft_override * user_overrides);`
  - Returns: `[ {ROUTER_EXPERT_PATTERN, paging_buft}, <user overrides...>, {nullptr,nullptr} ]`. Only routed experts are overridden; everything else defaults to its layer home. `.pattern` of the expert entry is a static string literal (stable lifetime); user-override patterns are borrowed from `user_overrides` storage.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test-weight-pager.cpp` near the other `static int test_*` functions (e.g. after `test_routing_boundary_prepass`). Add `#include "weight-pager/wp-router.h"` with the other includes at the top of the file.

```cpp
static int test_router_overrides_expert_only() {
    // sentinel non-null buft pointers (builder never dereferences them)
    auto paging = (ggml_backend_buffer_type_t) 0x1;
    auto ov = wp::build_router_overrides(paging, nullptr);
    EXPECT_EQ_INT((int) ov.size(), 2, "expert entry + terminator only");
    EXPECT(std::string(ov[0].pattern) == std::string(wp::ROUTER_EXPERT_PATTERN),
           "first override is the expert pattern");
    EXPECT(ov[0].buft == paging, "expert routed to paging buft");
    EXPECT(ov[1].pattern == nullptr, "list is terminated");
    for (const auto & o : ov) {
        if (o.pattern) {
            EXPECT(std::string(o.pattern) != std::string(".*"),
                   "no greedy .* dense override present");
        }
    }
    return 0;
}

static int test_router_overrides_preserve_user() {
    auto paging   = (ggml_backend_buffer_type_t) 0x1;
    auto userbuft = (ggml_backend_buffer_type_t) 0x2;
    llama_model_tensor_buft_override user[] = {
        { "attn_q\\.", userbuft },
        { nullptr, nullptr },
    };
    auto ov = wp::build_router_overrides(paging, user);
    EXPECT_EQ_INT((int) ov.size(), 3, "expert + 1 user override + terminator");
    EXPECT(std::string(ov[0].pattern) == std::string(wp::ROUTER_EXPERT_PATTERN),
           "expert override comes first");
    EXPECT(std::string(ov[1].pattern) == std::string("attn_q\\."),
           "user override preserved after expert");
    EXPECT(ov[1].buft == userbuft, "user override buft preserved");
    EXPECT(ov[2].pattern == nullptr, "list is terminated");
    return 0;
}
```

Register both in the `named_test tests[]` array in `main()`:

```cpp
        { "router_overrides_expert_only",   test_router_overrides_expert_only   },
        { "router_overrides_preserve_user", test_router_overrides_preserve_user },
```

- [ ] **Step 2: Run tests to verify they fail (link error / undefined symbol)**

Run: `cmake -S . -B build-cpu -DGGML_HIP=OFF -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpu --target test-weight-pager -j"$(nproc)"`
Expected: FAIL — compile/link error (`wp-router.h` not found or `wp::build_router_overrides` undefined).

- [ ] **Step 3: Create the header**

`src/weight-pager/wp-router.h`:

```cpp
#pragma once

#include "llama.h"          // llama_model_tensor_buft_override
#include "ggml-backend.h"   // ggml_backend_buffer_type_t

#include <vector>

namespace wp {

// Routed-expert tensor-name regex (consolidated MoE experts). MUST stay
// identical to the paging catalog / is_paged_weight filter.
extern const char * const ROUTER_EXPERT_PATTERN;

// Build the tensor_buft_override list for the resident-dense device router.
// Routes ONLY routed-expert tensors to `paging_buft`; every other tensor is
// left with no override so it defaults to its layer-home device. Any
// caller-supplied user overrides (nullptr-terminated array, may be null) are
// appended AFTER the expert entry, then a {nullptr,nullptr} terminator.
// The expert entry's .pattern is a static string literal (stable); user
// patterns are borrowed from `user_overrides` and must outlive the result.
std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        const llama_model_tensor_buft_override * user_overrides);

} // namespace wp
```

- [ ] **Step 4: Create the implementation**

`src/weight-pager/wp-router.cpp`:

```cpp
#include "weight-pager/wp-router.h"

namespace wp {

const char * const ROUTER_EXPERT_PATTERN = "ffn_(up|gate|down)_exps\\.";

std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        const llama_model_tensor_buft_override * user_overrides) {
    std::vector<llama_model_tensor_buft_override> out;
    out.push_back({ ROUTER_EXPERT_PATTERN, paging_buft });
    if (user_overrides != nullptr) {
        for (const auto * o = user_overrides; o->pattern != nullptr; ++o) {
            out.push_back(*o);
        }
    }
    out.push_back({ nullptr, nullptr });
    return out;
}

} // namespace wp
```

- [ ] **Step 5: Re-run the CPU tests, verify green**

Run: `cmake --build build-cpu --target test-weight-pager -j"$(nproc)" && ./build-cpu/bin/test-weight-pager`
Expected: PASS — all existing tests + the two new `router_overrides_*` tests green.

- [ ] **Step 6: Commit**

```bash
git add src/weight-pager/wp-router.h src/weight-pager/wp-router.cpp tests/test-weight-pager.cpp
git commit -m "feat(wp): extract testable build_router_overrides (expert-only, no .* dense override)"
```

---

## Task 2: Home-device inversion (C1)

Make each offloaded (GPU) layer's home device the resident device when the WP router is active, so KV cache, attention weights, and the FA node all land on the resident card.

**Files:**
- Modify: `src/llama-model.cpp` — hoist the resident device out of the router block (~1362-1394) and use it in `get_layer_buft_list` (~1436-1459).

**Interfaces:**
- Consumes: `wp_device_router_enabled` (bool, set in the router block ~1385), `devices` (`std::vector<llama_device>`), `wp_select_resident_device_index()` (existing, ~1298).
- Produces: `pimpl->dev_layer[il].dev == <resident device>` for all offloaded layers when the router is active.

- [ ] **Step 1: Hoist the resident device to the outer scope**

In `src/llama-model.cpp`, find the router-block declarations (~1362):

```cpp
    ggml_backend_buffer_type_t wp_paging_buft = nullptr;
    ggml_backend_buffer_type_t wp_resident_buft = nullptr;
    bool wp_device_router_enabled = false;
```

Add one line:

```cpp
    ggml_backend_buffer_type_t wp_paging_buft = nullptr;
    ggml_backend_buffer_type_t wp_resident_buft = nullptr;
    ggml_backend_dev_t         wp_resident_dev = nullptr;   // C1: layer-home device when router active
    bool wp_device_router_enabled = false;
```

- [ ] **Step 2: Record the resident device when the router resolves**

Inside the router block, where `wp_device_router_enabled = true;` is set (~1385, just after the buft push and before the `LLAMA_LOG_INFO`), add:

```cpp
            wp_device_router_enabled = true;
            wp_resident_dev = devices[resident_idx].dev;   // C1: home device for offloaded layers
```

(`resident_idx` is in scope here — it is declared at ~1368.)

- [ ] **Step 3: Force offloaded-layer home to the resident device**

In `get_layer_buft_list` (~1436-1459), the GPU branch currently returns the split-selected device:

```cpp
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
```

Insert the router override just before the `LLAMA_LOG_DEBUG`:

```cpp
        const int layer_gpu = std::upper_bound(splits.begin(), splits.begin() + n_devices(), float(il - i_gpu_start)/act_gpu_layers) - splits.begin();
        auto * dev = devices.at(layer_gpu).dev;
        if (wp_device_router_enabled && wp_resident_dev != nullptr) {
            // C1: pin every offloaded layer's home to the resident/attention
            // device so KV cache + attention weights + FA node co-locate there
            // and Flash Attention stays intra-device. Only routed experts are
            // moved off (via tensor_buft_overrides), not the layer home.
            dev = wp_resident_dev;
        }
        LLAMA_LOG_DEBUG("load_tensors: layer %3d assigned to device %s, is_swa = %d\n", il, ggml_backend_dev_name(dev), is_swa);
        return {dev, &pimpl->gpu_buft_list.at(dev)};
```

- [ ] **Step 4: Add an explicit INFO log for S1 verification**

Immediately after the `dev_output` assignment (~1459, after the `for` loop over `dev_layer`), add:

```cpp
    if (wp_device_router_enabled && wp_resident_dev != nullptr) {
        LLAMA_LOG_INFO("%s: WP router: layer-home pinned to resident device %s "
                       "(experts overridden to paging device)\n",
                       __func__, ggml_backend_dev_name(wp_resident_dev));
    }
```

- [ ] **Step 5: Build the CPU unit tests, verify still green**

Run: `cmake --build build-cpu --target test-weight-pager -j"$(nproc)" && ./build-cpu/bin/test-weight-pager`
Expected: PASS (this change is on the model-load path; no unit test exercises it — the check is that nothing regressed and it compiles). GPU verification is S1.

- [ ] **Step 6: Commit**

```bash
git add src/llama-model.cpp
git commit -m "feat(wp): C1 invert layer-home to resident device for the WP router

With layer-home = resident/attention card, KV cache (dev_layer) + attention
weights + FA node co-locate there, so sched_reserve's device_fa==dev_layer
check passes and Flash Attention stays enabled. Only routed experts move to
the paging card via tensor_buft_overrides. The greedy .* dense override is now
redundant (removed in the next task)."
```

---

## Task 3: Wire build_router_overrides and drop the `.*` override (C2)

**Files:**
- Modify: `src/llama-model.cpp` — router block override synthesis (~1359-1384).

**Interfaces:**
- Consumes: `wp::build_router_overrides()` (Task 1), `wp_paging_buft`, `params.tensor_buft_overrides`.
- Depends on Task 2 (C1): removing the `.*` override is only safe once layer-home is the resident device.

- [ ] **Step 1: Add the include**

At the top of `src/llama-model.cpp` with the other `#include`s, add:

```cpp
#include "weight-pager/wp-router.h"
```

- [ ] **Step 2: Replace the inline override synthesis**

Find the current synthesis (~1376-1387):

```cpp
        if (wp_paging_buft != nullptr && wp_resident_buft != nullptr) {
            wp_expert_override_pattern = "ffn_(up|gate|down)_exps\\.";
            wp_dense_override_pattern = ".*";
            wp_tensor_buft_overrides.push_back({ wp_expert_override_pattern.c_str(), wp_paging_buft });
            wp_tensor_buft_overrides.push_back({ wp_dense_override_pattern.c_str(),  wp_resident_buft });
            if (params.tensor_buft_overrides != nullptr) {
                for (const auto * o = params.tensor_buft_overrides; o->pattern != nullptr; ++o) {
                    wp_tensor_buft_overrides.push_back(*o);
                }
            }
            wp_tensor_buft_overrides.push_back({ nullptr, nullptr });
            ml.tensor_buft_overrides = wp_tensor_buft_overrides.data();
            wp_device_router_enabled = true;
            wp_resident_dev = devices[resident_idx].dev;
            LLAMA_LOG_INFO(...);
        } else {
```

Replace the body (keeping the `wp_device_router_enabled`/`wp_resident_dev`/`LLAMA_LOG_INFO` lines) with:

```cpp
        if (wp_paging_buft != nullptr && wp_resident_buft != nullptr) {
            // C2: override ONLY routed experts out to the paging device. Dense/
            // attention tensors default to their layer home (the resident
            // device, pinned in C1) — no greedy ".*" override.
            wp_tensor_buft_overrides = wp::build_router_overrides(wp_paging_buft, params.tensor_buft_overrides);
            ml.tensor_buft_overrides = wp_tensor_buft_overrides.data();
            wp_device_router_enabled = true;
            wp_resident_dev = devices[resident_idx].dev;
            LLAMA_LOG_INFO("%s: WP_RESIDENT_DENSE router: paging=%s (%s), resident=%s (%s)\n",
                           __func__,
                           ggml_backend_dev_name(devices[paging_idx].dev), ggml_backend_buft_name(wp_paging_buft),
                           ggml_backend_dev_name(devices[resident_idx].dev), ggml_backend_buft_name(wp_resident_buft));
        } else {
```

- [ ] **Step 3: Remove the now-unused pattern locals**

Delete the now-dead declarations (~1359-1360):

```cpp
    std::string wp_expert_override_pattern;
    std::string wp_dense_override_pattern;
```

Keep `std::vector<llama_model_tensor_buft_override> wp_tensor_buft_overrides;` (still used, now assigned from the helper). Its lifetime spans the `create_tensor()` calls in the same `load_tensors()` scope, unchanged.

- [ ] **Step 4: Build the CPU unit tests, verify green**

Run: `cmake --build build-cpu --target test-weight-pager -j"$(nproc)" && ./build-cpu/bin/test-weight-pager`
Expected: PASS. Compilation confirms the wiring; the `router_overrides_*` unit tests (Task 1) already lock the expert-only behavior.

- [ ] **Step 5: Commit**

```bash
git add src/llama-model.cpp
git commit -m "feat(wp): C2 drop greedy .* dense override, route only experts to paging card

Dense/attention now defaults to the resident layer-home (C1); only
ffn_*_exps is overridden to the paging device. Also removes the parent
review's 'WP overrides shadow user overrides' finding (the .* was the cause)."
```

---

## Task 4: Fail-loud when FA would silently disable (C4)

**Files:**
- Modify: `src/llama-context.cpp` — after the `fa_device_mismatch` resolution (~525-543).

**Interfaces:**
- Consumes: `fa_device_mismatch` (local bool), `model.wp_pager` (unique_ptr member, non-null when weight paging active), `model.n_devices()`.

- [ ] **Step 1: Add the fail-loud guard**

Find the mismatch resolution block (~534-543):

```cpp
        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }
```

Insert a hard error before `cparams.flash_attn = false;` when weight paging spans >1 device:

```cpp
        if (fa_device_mismatch) {
            if (model.wp_pager != nullptr && model.n_devices() > 1) {
                // C4: under the WP two-card resident-dense router, a disabled FA
                // means the non-FA path will try to allocate the full ~40 GB KQ
                // matrix on a GPU and OOM / fault. Refuse loudly instead of
                // silently falling through. See the attention-island design:
                // layer-home + KV + FA must co-locate on the resident device.
                throw std::runtime_error(
                    "weight-paging cross-device: Flash Attention resolved to DISABLED "
                    "(FA node device != layer/KV device). The non-FA attention path "
                    "would OOM a full attention matrix. Ensure the resident device "
                    "hosts attention+KV+FA (home-device inversion / --weight-paging-resident-device).");
            }
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }
```

- [ ] **Step 2: Build the CPU unit tests, verify green**

Run: `cmake --build build-cpu --target test-weight-pager -j"$(nproc)" && ./build-cpu/bin/test-weight-pager`
Expected: PASS (no test exercises this GPU path; compilation + no-regression is the check).

- [ ] **Step 3: Commit**

```bash
git add src/llama-context.cpp
git commit -m "feat(wp): C4 fail loud if cross-device WP disables Flash Attention

Prevents silently entering the non-FA path that allocates a ~40 GB attention
matrix and faults the GPU. Fires only when weight paging spans >1 device."
```

---

## Task 5: Documentation / stale-comment fixes (C3)

**Files:**
- Modify: `src/weight-pager/wp-pager.h` (~116-120, "per-device pools" / "ALL weights paged" comment).
- Modify: `src/llama-model.cpp` (~1960-1971, catalog comment claiming "ALL weights are paged").
- Modify: `src/llama.cpp` (~144-155, the single-GPU-buft guard comment).

**Interfaces:** none (comments only).

- [ ] **Step 1: Correct the pager header comment**

In `src/weight-pager/wp-pager.h` (~116-120), update the comment describing paging scope to state the single-paged-device contract explicitly. Replace whatever claims all weights page / per-device pools are needed with:

```cpp
    // Paging is single-device by contract: under WP_RESIDENT_DENSE only routed
    // experts page, and they all live on the paging device, so one pool keyed
    // by that device's buffer type is sufficient. Dense/attention weights are
    // resident on the (possibly different) attention device and never page.
```

- [ ] **Step 2: Correct the catalog comment in the model loader**

In `src/llama-model.cpp` (~1960-1971), replace any "ALL weights are paged" phrasing with a note that under resident-dense only routed experts are cataloged/paged; dense tensors use the normal resident allocator.

- [ ] **Step 3: Correct the single-GPU-buft guard comment**

In `src/llama.cpp` (~144-155), update the comment above the `gpu_bufts.size() > 1` throw to state that this is the intended single-paged-device contract (paged tensors are experts-only and single-device), not a temporary limitation.

- [ ] **Step 4: Build the CPU unit tests, verify green**

Run: `cmake --build build-cpu --target test-weight-pager -j"$(nproc)" && ./build-cpu/bin/test-weight-pager`
Expected: PASS (comments only).

- [ ] **Step 5: Commit**

```bash
git add src/weight-pager/wp-pager.h src/llama-model.cpp src/llama.cpp
git commit -m "docs(wp): C3 state single-paged-device contract, drop stale 'all weights paged' comments"
```

---

## GPU Validation (manual, user-gated — NOT auto-run)

All runs on `mad-lab-main` with the dual-arch `build-hip`, `--no-mmap`, and explicit user go-ahead per standing rule. Merge `feat/wp-md-correctness` (C5) into this branch and rebuild `build-hip` before S1.

### S1 — zero-code hypothesis smoke test (before Task 2 lands, or with Task 2)
Force all layers to the resident card via `--tensor-split` (equivalent to C1's outcome), short context, few tokens.
- Command shape: `llama-server ... --no-mmap --weight-paging --tensor-split 0,1 -c 512 -ngl <all>` with `WP_RESIDENT_DENSE=1`, `--weight-paging-slots 2000`.
- **Pass:** log shows `sched_reserve: Flash Attention ... set to enabled`, NO `allocating 40288 MiB` line, coherent tokens, 0 GPU faults.
- **Fail here ⇒ mechanism wrong; stop before/after Task 2 and re-diagnose.**

### S2 — feature run (after Tasks 1–4)
Same as S1 but via the flag instead of the split hack: `--weight-paging-resident-device ROCm1` (or `auto`), no `--tensor-split`.
- **Pass:** same criteria as S1, plus the Task 2 INFO log `WP router: layer-home pinned to resident device ROCm1`.

### S3 — performance run (after S2 passes)
Full context; size the expert pool to the freed ROCm0 VRAM (~2× Phase-1's 8.5 GB, e.g. `--weight-paging-slots ~4000`). Log to `/home/kmbandy/wp_logs/` (real disk).
- Measure decode t/s + page-in / eviction / prefetch-hit stats; compare to Phase-1 baseline (1.038 t/s, P2P depth-4).
- **Success:** FA enabled, 0 faults, coherent, decode ≥ Phase-1 baseline. **Open question (the point of S3):** does the larger pool break the expert-cache thrash?

---

## Self-Review Notes

- **Spec coverage:** C1→Task 2, C2→Tasks 1+3, C3→Task 5, C4→Task 4, C5→Prerequisites (separate branch), S1–S3→Validation. All spec components mapped.
- **Task ordering:** Task 2 (C1) precedes Task 3 (drop `.*`) — dropping the dense override is only safe once layer-home is resident. Task 1 (extract+test) is standalone and precedes Task 3 (which wires it).
- **Type consistency:** `wp::build_router_overrides` / `wp::ROUTER_EXPERT_PATTERN` used identically in Task 1 (def) and Task 3 (call). `wp_resident_dev` declared (Task 2 Step 1) before use (Steps 2–4).
