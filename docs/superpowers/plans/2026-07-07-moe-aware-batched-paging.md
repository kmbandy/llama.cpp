# MoE-Aware Batched Paging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the weight-pager eval-callback batching engage on MoE models under real paging pressure (partial residency + evictions) so large-MoE decode removes the per-op GPU-sync tax without the near-null routing fault.

**Architecture:** A one-time pre-pass marks each `MUL_MAT_ID` and its ids-producer as range-break points; the eval-cb breaks the batch range there so every routing op runs isolated (fixing the read-before-produce H3 and TLS take-steal H4 faults); a per-range pin lifecycle (release-after-sync, with reactive auto-break on pin exhaustion) replaces the per-op unpin, letting `batch_safe()` drop its `evictions==0`/`!has_experts()` requirement. All new behavior is behind a default-off `WP_PAGED_BATCH` flag.

**Tech Stack:** C++17, ggml/llama.cpp, ROCm/HIP (gfx1201 R9700), `test-weight-pager` (lightweight no-framework unit suite), llama-perplexity + llama-server for GPU validation.

## Global Constraints

- Branch: `feat/wp-vnext`. Edit/build-check on **mad-lab-2026**; all GPU validation on **mad-lab-main** (R9700 = ROCm0, gfx1201, 32GB). Remote shell is fish → `ssh mad-lab-main bash -s <<'EOF'`.
- Every new behavior is behind `WP_PAGED_BATCH` (default OFF). Flag OFF ⇒ byte-identical to current shipped behavior (dense resident-batching via `WP_BATCH_EVAL_CB`; MoE and any eviction ⇒ per-op). Do not flip default until both validation gates pass.
- Always build multi-arch: `cmake --build build-hip -j2 --target <targets>` with `gfx1201;gfx1030`, `GGML_HIP_AITER=ON` (reuse Triton AOT stamps — never `rm` them). Use the capped unit: `systemd-run --user --unit=<u> --collect -p MemoryMax=13000M -p MemoryHigh=11000M -p CPUQuota=600% ...`.
- Never run inference without the standing `--no-mmap` flag. Print VRAM pre-flight before any 20B+ paged run.
- `test-weight-pager` pattern: each `static int test_x()` returns failure count; register it in `main()`'s sum; use `ScopedEnv` to set/restore env vars; HIP-only assertions gate on `GGML_USE_HIP` at runtime.
- DeepSeek V4 support is present (`deepseek4` arch, `src/models/deepseek4.cpp`, upstream #24162, compiled in build-hip). Validate via llama-perplexity (V4 Flash tripped a llama-server `--direct-io` fit-hang before; avoid that path).

---

## File Structure

- `src/weight-pager/wp-pager.h` — add `mark_routing_boundaries` / `is_routing_break` declarations + `routing_break_tensors_` member + topology-signature member.
- `src/weight-pager/wp-pager.cpp` — implement the pre-pass + the `batch_safe()` flag-gated change.
- `src/weight-pager/wp-eval-cb.cpp` — `WP_PAGED_BATCH` flag reader; break-at-boundary in `eval_cb_op_return`; per-range pin lifecycle; reactive auto-break.
- `src/llama-context.cpp` — call `mark_routing_boundaries(gf)` before `ggml_backend_sched_graph_compute_async`.
- `tests/test-weight-pager.cpp` — unit tests for the pre-pass, the flag reader, and the `batch_safe` predicate.

---

## Task 1: Routing-boundary pre-pass (Unit A)

Pure graph-walk logic; fully unit-testable. Produces the marker set the eval-cb consumes.

**Files:**
- Modify: `src/weight-pager/wp-pager.h` (class `WeightPager`, public section near line 143; private members near line 300)
- Modify: `src/weight-pager/wp-pager.cpp` (add two methods near `batch_safe`, line ~506)
- Test: `tests/test-weight-pager.cpp`

**Interfaces:**
- Produces:
  - `void WeightPager::mark_routing_boundaries(const struct ggml_cgraph * gf);`
  - `bool WeightPager::is_routing_break(const struct ggml_tensor * t) const;`

- [ ] **Step 1: Write the failing test**

Add to `tests/test-weight-pager.cpp` (include `"ggml.h"` if not already present):

```cpp
static int test_routing_boundary_prepass() {
    int fails = 0;
    // Build a tiny graph: producer -> view(as src[2]) -> MUL_MAT_ID.
    // We only need op types + src wiring; no compute.
    struct ggml_init_params ip = { /*.mem_size=*/ 16*1024*1024, /*.mem_buffer=*/ nullptr, /*.no_alloc=*/ true };
    struct ggml_context * ctx = ggml_init(ip);

    struct ggml_tensor * ids_producer = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 8);
    ggml_set_name(ids_producer, "ids_producer");
    struct ggml_tensor * ids_view = ggml_view_1d(ctx, ids_producer, 8, 0);   // src[2] reaches producer via view
    struct ggml_tensor * as = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 4, 4, 2);
    struct ggml_tensor * b  = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 8);
    struct ggml_tensor * mmid = ggml_mul_mat_id(ctx, as, b, ids_view);
    ggml_set_name(mmid, "mmid");

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, mmid);

    wp::WeightPager pager;                     // default-constructed; pre-pass needs no VRAM
    pager.mark_routing_boundaries(gf);

    if (!pager.is_routing_break(mmid))         { fprintf(stderr, "FAIL: mmid not marked\n"); fails++; }
    if (!pager.is_routing_break(ids_producer)) { fprintf(stderr, "FAIL: ids producer (view root) not marked\n"); fails++; }
    // a non-boundary tensor must NOT be marked
    if (pager.is_routing_break(b))             { fprintf(stderr, "FAIL: unrelated tensor marked\n"); fails++; }

    ggml_free(ctx);
    return fails;
}
```

Register it in `main()` alongside the other `total += test_*();` lines.

- [ ] **Step 2: Run test to verify it fails**

Build + run on mad-lab-main:
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target test-weight-pager 2>&1 | tail -5
EOF
```
Expected: compile error — `mark_routing_boundaries` / `is_routing_break` not declared.

- [ ] **Step 3: Add declarations + members (wp-pager.h)**

In the `public:` section (near line 143):
```cpp
    // Range-break markers for WP_PAGED_BATCH: every GGML_OP_MUL_MAT_ID node and
    // the view-root of its src[2] (the router-ids producer). Populated once per
    // graph by mark_routing_boundaries(); consumed by the eval-cb to break the
    // batch range so each routing op runs isolated.
    void mark_routing_boundaries(const struct ggml_cgraph * gf);
    bool is_routing_break(const struct ggml_tensor * t) const {
        return routing_break_tensors_.count(t) != 0;
    }
```
In the `private:` members (near line 300):
```cpp
    std::unordered_set<const struct ggml_tensor *> routing_break_tensors_;
    // Cheap topology signature to skip the walk when the graph is unchanged.
    struct { int n_nodes = -1; const void * first = nullptr; const void * last = nullptr; } routing_sig_;
```
Ensure `#include <unordered_set>` is present in wp-pager.h.

- [ ] **Step 4: Implement the pre-pass (wp-pager.cpp, near line 506)**

```cpp
void WeightPager::mark_routing_boundaries(const struct ggml_cgraph * gf) {
    if (gf == nullptr || gf->n_nodes <= 0) { return; }
    // Skip the walk if topology is unchanged (decode graphs are stable).
    const void * first = gf->nodes[0];
    const void * last  = gf->nodes[gf->n_nodes - 1];
    if (routing_sig_.n_nodes == gf->n_nodes && routing_sig_.first == first && routing_sig_.last == last) {
        return;
    }
    routing_break_tensors_.clear();
    for (int i = 0; i < gf->n_nodes; ++i) {
        struct ggml_tensor * node = gf->nodes[i];
        if (node->op != GGML_OP_MUL_MAT_ID) { continue; }
        routing_break_tensors_.insert(node);                 // isolate the routing op
        struct ggml_tensor * ids = node->src[2];             // router-selected expert ids
        while (ids != nullptr && ids->view_src != nullptr) { // resolve view chain to producer root
            ids = ids->view_src;
        }
        if (ids != nullptr) { routing_break_tensors_.insert(ids); }  // break AFTER the producer
    }
    routing_sig_.n_nodes = gf->n_nodes;
    routing_sig_.first = first;
    routing_sig_.last  = last;
}
```
Add `#include "ggml.h"` to wp-pager.cpp if not already included (for `ggml_cgraph`/`ggml_tensor`/`GGML_OP_MUL_MAT_ID`).

- [ ] **Step 5: Run the test to verify it passes**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target test-weight-pager 2>&1 | tail -3 && ./build-hip/bin/test-weight-pager 2>&1 | tail -5
EOF
```
Expected: build clean, `test-weight-pager` prints 0 failures (exit 0).

- [ ] **Step 6: Commit**
```bash
git add src/weight-pager/wp-pager.h src/weight-pager/wp-pager.cpp tests/test-weight-pager.cpp
git commit -m "feat(wp): routing-boundary pre-pass (mark_routing_boundaries/is_routing_break)"
```

---

## Task 2: Wire the pre-pass into graph_compute (Unit A call site)

**Files:**
- Modify: `src/llama-context.cpp` (`llama_context::graph_compute`, near line 2473-2492; pager handle is `model.wp_pager`, cf. eval-cb registration at ~1365)

**Interfaces:**
- Consumes: `WeightPager::mark_routing_boundaries(const ggml_cgraph*)` (Task 1)

- [ ] **Step 1: Add the call before sched compute**

In `llama_context::graph_compute`, immediately before `ggml_backend_sched_graph_compute_async(sched.get(), gf);` (line ~2492):
```cpp
    if (model.wp_pager) {
        model.wp_pager->mark_routing_boundaries(gf);
    }
```
Match the existing guard style used for the eval-cb registration (`if (model.wp_pager)` / `model.wp_pager.get()`), confirming the exact member accessor by reading lines 1360-1370.

- [ ] **Step 2: Build to verify it compiles**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target llama 2>&1 | tail -3
EOF
```
Expected: build clean (this is a wiring-only change; behavior is inert until the eval-cb consumes the markers in Task 4).

- [ ] **Step 3: Commit**
```bash
git add src/llama-context.cpp
git commit -m "feat(wp): call mark_routing_boundaries before sched compute"
```

---

## Task 3: WP_PAGED_BATCH flag + batch_safe change (Unit E + C predicate)

**Files:**
- Modify: `src/weight-pager/wp-eval-cb.cpp` (flag reader near `wp_batch_eval_cb_enabled`, line ~67)
- Modify: `src/weight-pager/wp-pager.cpp` (`batch_safe()`, line ~506)
- Modify: `src/weight-pager/wp-pager.h` (declare `paged_batch()` accessor if needed)
- Test: `tests/test-weight-pager.cpp`

**Interfaces:**
- Produces: `bool wp_paged_batch_enabled();` (in `wp::` namespace, declared in wp-eval-cb.h)
- Produces: `batch_safe()` returns true under the paged-batch regime without requiring `evictions==0` / `!has_experts()`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test-weight-pager.cpp`:
```cpp
static int test_wp_paged_batch_flag_default_off() {
    int fails = 0;
    ScopedEnv guard("WP_PAGED_BATCH");
    unsetenv("WP_PAGED_BATCH");
    if (wp::wp_paged_batch_enabled()) { fprintf(stderr, "FAIL: WP_PAGED_BATCH must default OFF\n"); fails++; }
    setenv("WP_PAGED_BATCH", "1", 1);
    // NOTE: the flag is read once via function-local static, so this asserts the
    // parse rule, not live re-read. Validate the "1" branch in a fresh process.
    return fails;
}
```
(Flag readers in this file cache via `static const bool`; the test asserts the default-off parse. The `=1` branch is exercised by the GPU validation runs, not a re-reading unit test.)

- [ ] **Step 2: Run to verify it fails**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target test-weight-pager 2>&1 | tail -3
EOF
```
Expected: compile error — `wp_paged_batch_enabled` undeclared.

- [ ] **Step 3: Add the flag reader (wp-eval-cb.cpp near line 67, mirroring wp_batch_eval_cb_enabled)**
```cpp
bool wp_paged_batch_enabled() {
    // Default OFF. Enables MoE-aware batched paging: routing-boundary breaks +
    // per-range pin lifecycle, and lets batch_safe() drop evictions==0/!has_experts.
    static const bool enabled = []() {
        const char * v = std::getenv("WP_PAGED_BATCH");
        return v != nullptr && std::strcmp(v, "1") == 0;
    }();
    return enabled;
}
```
Declare it in `src/weight-pager/wp-eval-cb.h` next to the other `wp_*_enabled()` declarations.

- [ ] **Step 4: Change batch_safe (wp-pager.cpp:506)**

Replace:
```cpp
bool WeightPager::batch_safe() const {
    return stats_.evictions == 0 && pool_.size_class_slots_enabled() && !catalog_.has_experts();
}
```
with:
```cpp
bool WeightPager::batch_safe() const {
    // WP_PAGED_BATCH: batching is governed by live pinnability + routing-boundary
    // breaks (see wp-eval-cb.cpp), not by a static eviction count, so MoE and
    // eviction pressure are allowed. Legacy path keeps the conservative gate.
    if (wp_paged_batch_enabled()) {
        return pool_.size_class_slots_enabled();
    }
    return stats_.evictions == 0 && pool_.size_class_slots_enabled() && !catalog_.has_experts();
}
```
Ensure wp-pager.cpp can see `wp_paged_batch_enabled()` (include `wp-eval-cb.h`).

- [ ] **Step 5: Run tests to verify pass**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target test-weight-pager 2>&1 | tail -3 && ./build-hip/bin/test-weight-pager 2>&1 | tail -5
EOF
```
Expected: build clean, 0 failures.

- [ ] **Step 6: Commit**
```bash
git add src/weight-pager/wp-eval-cb.cpp src/weight-pager/wp-eval-cb.h src/weight-pager/wp-pager.cpp src/weight-pager/wp-pager.h tests/test-weight-pager.cpp
git commit -m "feat(wp): WP_PAGED_BATCH flag + batch_safe pinnability-governed under it"
```

---

## Task 4: Break the batch range at routing boundaries (Unit B enable — the H3/H4 fix)

**Files:**
- Modify: `src/weight-pager/wp-eval-cb.cpp` (`eval_cb_op_return` lambda, lines ~236-244)

**Interfaces:**
- Consumes: `WeightPager::is_routing_break` (Task 1), `wp_paged_batch_enabled` (Task 3)

- [ ] **Step 1: Add the boundary break to eval_cb_op_return**

The lambda currently returns `false` (batch) when `batch_safe() && !routing_tls_set && sync_fallback unchanged`. Add a boundary break: when `WP_PAGED_BATCH` is on and `t` is a marked routing boundary, return `true` to end the range here. Insert at the top of the lambda body (keep the existing conditions):
```cpp
    auto eval_cb_op_return = [&]() -> bool {
        if (batch_eval_cb && wp_paged_batch_enabled() && pager->is_routing_break(t)) {
            return true;   // break the range: ids-producer syncs before, MUL_MAT_ID runs isolated
        }
        if (batch_eval_cb &&
            pager->batch_safe() &&
            !routing_tls_set &&
            pager->sync_fallback_count() == sync_fallbacks_before) {
            return false;
        }
        return true;
    };
```
(`t` is the callback's tensor argument, in scope in the lambda.)

- [ ] **Step 2: Build**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target llama llama-perplexity 2>&1 | tail -3
EOF
```
Expected: clean build.

- [ ] **Step 3: GPU smoke — MoE no longer faults with paged-batch on (LFM, resident)**

This is the correctness proof for the boundary break in isolation (before the pin-lifecycle change). At this point `batch_safe()` for MoE requires `WP_PAGED_BATCH` on; the pin lifecycle is still per-op (Task 5 not done), so keep `evictions==0` (fully resident, slots=750). Expect NO fault and PPL == 27.09.
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
systemd-run --user --unit=t4-lfm --collect -p MemoryMax=24000M --working-directory=/home/kmbandy/GitHub/llama.cpp \
  --setenv=WP_SIZE_CLASS_SLOTS=1 --setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1 \
  bash -c "./build-hip/bin/llama-perplexity -m ~/models/LFM2.5-8B-A1B-Q6_K.gguf -f wikitext-2-raw/wiki.test.raw --chunks 4 -c 512 --no-mmap --weight-paging --weight-paging-slots 750 --weight-paging-prefetch -ngl 99 --device ROCm0 > /tmp/t4-lfm.log 2>&1"
while systemctl --user is-active --quiet t4-lfm.service; do sleep 5; done
grep -Ei 'Final estimate|Memory access fault' /tmp/t4-lfm.log
grep -Ei 'evictions|routing_ptrs' /tmp/t4-lfm.log | head
EOF
```
Expected: `Final estimate: PPL = 27.09xx`, no "Memory access fault", `routing_ptrs_discarded_unconsumed` low/0.

- [ ] **Step 4: Commit**
```bash
git add src/weight-pager/wp-eval-cb.cpp
git commit -m "feat(wp): break batch range at routing boundaries under WP_PAGED_BATCH (fixes H3/H4)"
```

---

## Task 5: Batched per-range pin lifecycle + reactive auto-break (Unit C core)

The load-bearing change. Replaces per-op unpin with release-after-range-sync, and auto-breaks a range when a pin can't be satisfied. GPU-integration validated; a small bookkeeping helper is unit-tested.

**Files:**
- Modify: `src/weight-pager/wp-eval-cb.cpp` (MAD-231 unpin block ~337-343; pin-recording sites; `eval_cb_op_return`)
- Modify: `src/weight-pager/wp-pool.h`/`.cpp` — add `bool can_pin_without_evicting_pinned(size_t requested_size) const;` (pure predicate: is there a free or unpinned-evictable slot of a fitting class?)
- Test: `tests/test-weight-pager.cpp` (test the pool predicate)

**Interfaces:**
- Consumes: `WeightPager::is_routing_break`, `wp_paged_batch_enabled`, `pool_.alloc_slot`/`is_pinned`.
- Produces: `bool PoolAllocator::can_pin_without_evicting_pinned(size_t) const;`

- [ ] **Step 1: Write the failing test for the pool predicate**

Model on `test_pool_alloc_returns_neg1_when_all_pinned` (line ~1018). Pin every slot, assert `can_pin_without_evicting_pinned` is false; unpin one, assert true:
```cpp
static int test_pool_can_pin_predicate() {
    int fails = 0;
    wp::PoolAllocator pool;
    pool.init(/*n_slots=*/2, /*slot_bytes=*/1024, /*vram_buf=*/nullptr);   // match existing test init signature
    int s0 = pool.alloc_slot(1024); pool.pin_slot(s0);
    int s1 = pool.alloc_slot(1024); pool.pin_slot(s1);
    if (pool.can_pin_without_evicting_pinned(1024)) { fprintf(stderr, "FAIL: all pinned but predicate true\n"); fails++; }
    pool.unpin_slot(s1);
    if (!pool.can_pin_without_evicting_pinned(1024)) { fprintf(stderr, "FAIL: unpinned slot exists but predicate false\n"); fails++; }
    return fails;
}
```
Confirm the exact `init`/`pin_slot`/`unpin_slot` names/signatures against the existing pool tests (lines ~491, 902-1060) and adjust the test to match before writing it.

- [ ] **Step 2: Run to verify it fails**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target test-weight-pager 2>&1 | tail -3
EOF
```
Expected: compile error — `can_pin_without_evicting_pinned` undeclared.

- [ ] **Step 3: Implement the pool predicate**

In `wp-pool.h`/`.cpp`, add (mirroring the LRU walk in `alloc_slot`, but read-only — return true if any slot is free, or unpinned and of a fitting size class; false if every fitting slot is pinned):
```cpp
bool PoolAllocator::can_pin_without_evicting_pinned(size_t requested_size) const {
    for (int s = 0; s < n_slots_; ++s) {
        if (!slot_in_use_[s]) { return true; }                 // free slot
        if (!is_pinned(s) && slot_fits_(s, requested_size)) { return true; } // reclaimable
    }
    return false;
}
```
Use the actual member names for the in-use flag and size-fit check found in `alloc_slot`; keep it read-only (no state mutation).

- [ ] **Step 4: Run the predicate test to pass**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target test-weight-pager 2>&1 | tail -3 && ./build-hip/bin/test-weight-pager 2>&1 | tail -5
EOF
```
Expected: 0 failures.

- [ ] **Step 5: Convert per-op unpin to per-range release (wp-eval-cb.cpp)**

Replace the MAD-231 immediate unpin (lines ~337-343, the non-async branch that unpins `s_pinned_pages_prev_op` at callback top) with per-range accumulation:
- Add file-scope state: `std::vector<int> s_range_pins;` (pages pinned in the range currently being built) and `std::vector<int> s_range_pins_to_release;` (pages from the range that just computed+synced).
- At the top of the `ask=true` callback, when `wp_paged_batch_enabled()`: if a release is pending (set when the previous callback ended a range), move `s_range_pins_to_release` → unpin all → clear. Do NOT unpin `s_range_pins` (still in flight).
- Where the current code records a pin into `s_pinned_pages_prev_op`, also (under paged-batch) append to `s_range_pins`.
- When `eval_cb_op_return()` returns `true` (range boundary), move `s_range_pins` → `s_range_pins_to_release` and set the pending-release flag (released at next callback / ask=false).
- In the `ask=false` path (currently `if (!ask) return true;` at line ~227): under paged-batch, if a release is pending, unpin `s_range_pins_to_release` now (this fires right after the scheduler's post-range sync) and clear the flag.
- In `weight_pager_eval_cb_reset`: flush both `s_range_pins` and `s_range_pins_to_release` (unpin all) so teardown leaks nothing.

Keep all of this gated on `wp_paged_batch_enabled()`; when off, the existing per-op `s_pinned_pages_prev_op` path is untouched.

- [ ] **Step 6: Add reactive auto-break on pin exhaustion (eval_cb_op_return)**

Where the callback ensures/pins this op's pages, if (under paged-batch) `!pager->can_pin_without_evicting_pinned(page_size)` for a required page, set a local `bool pin_exhausted = true;`. Then in `eval_cb_op_return`, return `true` when `pin_exhausted` (end the range, forcing compute+sync + release before continuing). Add this as the first condition after the routing-boundary break.

- [ ] **Step 7: Build**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target llama llama-perplexity 2>&1 | tail -3
EOF
```
Expected: clean build.

- [ ] **Step 8: Commit**
```bash
git add src/weight-pager/wp-eval-cb.cpp src/weight-pager/wp-pool.h src/weight-pager/wp-pool.cpp tests/test-weight-pager.cpp
git commit -m "feat(wp): per-range pin lifecycle + reactive auto-break under WP_PAGED_BATCH"
```

---

## Task 6: Compose per-range release with WP_ASYNC_ENSURE (Unit D)

The async-ensure path (wp-eval-cb.cpp:276-331) already defers unpin via `s_pending_async_ops`. Ensure the per-range release awaits async completion before unpinning.

**Files:**
- Modify: `src/weight-pager/wp-eval-cb.cpp` (async branch + the range-release added in Task 5)

- [ ] **Step 1: Make range-release async-aware**

In the per-range release step (Task 5, Step 5), when `pager->async_ensure_enabled()`, route the range's pages through the existing `s_pending_async_ops` completion mechanism (record an event, release on `hipEventQuery` success) instead of an immediate `unpin_page`. Reuse the existing `PendingAsyncOp` machinery (lines 295-317) rather than duplicating it.

- [ ] **Step 2: Build + resident smoke with async on**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
cmake --build build-hip -j2 --target llama llama-perplexity 2>&1 | tail -3
systemd-run --user --unit=t6-lfm --collect -p MemoryMax=24000M --working-directory=/home/kmbandy/GitHub/llama.cpp \
  --setenv=WP_SIZE_CLASS_SLOTS=1 --setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1 --setenv=WP_ASYNC_ENSURE=1 \
  bash -c "./build-hip/bin/llama-perplexity -m ~/models/LFM2.5-8B-A1B-Q6_K.gguf -f wikitext-2-raw/wiki.test.raw --chunks 4 -c 512 --no-mmap --weight-paging --weight-paging-slots 750 --weight-paging-prefetch -ngl 99 --device ROCm0 > /tmp/t6-lfm.log 2>&1"
while systemctl --user is-active --quiet t6-lfm.service; do sleep 5; done
grep -Ei 'Final estimate|Memory access fault' /tmp/t6-lfm.log
EOF
```
Expected: PPL 27.09, no fault.

- [ ] **Step 3: Commit**
```bash
git add src/weight-pager/wp-eval-cb.cpp
git commit -m "feat(wp): compose per-range pin release with async ensure completion"
```

---

## Task 7: Resident validation gate

**Files:** none (validation only). Uses the built `build-hip` binaries.

- [ ] **Step 1: LFM resident correctness + determinism (evictions==0)**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
for i in 1 2; do
systemd-run --user --unit=t7-lfm$i --collect -p MemoryMax=24000M --working-directory=/home/kmbandy/GitHub/llama.cpp \
  --setenv=WP_SIZE_CLASS_SLOTS=1 --setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1 \
  bash -c "./build-hip/bin/llama-perplexity -m ~/models/LFM2.5-8B-A1B-Q6_K.gguf -f wikitext-2-raw/wiki.test.raw --chunks 4 -c 512 --no-mmap --weight-paging --weight-paging-slots 750 --weight-paging-prefetch -ngl 99 --device ROCm0 > /tmp/t7-lfm$i.log 2>&1"
while systemctl --user is-active --quiet t7-lfm$i.service; do sleep 5; done
echo "run $i: $(grep -E 'Final estimate' /tmp/t7-lfm$i.log)"; grep -qi 'Memory access fault' /tmp/t7-lfm$i.log && echo "  FAULT" || echo "  no fault"
done
EOF
```
Expected: both runs `PPL = 27.09xx`, no fault (deterministic).

- [ ] **Step 2: 27B dense unaffected (regression guard)**
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
systemd-run --user --unit=t7-27b --collect -p MemoryMax=28000M --working-directory=/home/kmbandy/GitHub/llama.cpp \
  --setenv=WP_SIZE_CLASS_SLOTS=1 --setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1 \
  bash -c "./build-hip/bin/llama-perplexity -m ~/models/Qwen3.6-27B-Q6_K.gguf -f wikitext-2-raw/wiki.test.raw --chunks 4 -c 512 --no-mmap --weight-paging --weight-paging-slots 345 --weight-paging-prefetch -ngl 99 --device ROCm0 > /tmp/t7-27b.log 2>&1"
while systemctl --user is-active --quiet t7-27b.service; do sleep 5; done
grep -E 'Final estimate' /tmp/t7-27b.log
EOF
```
Expected: `PPL = 5.4623` (dense unchanged).

- [ ] **Step 3: Decode t/s via llama-server (LFM)** — confirm t/s ≥ per-op baseline. Launch llama-server with the same env, hit `/completion`, read `predicted_per_second`. Record the number in the plan's results and compare to a `WP_PAGED_BATCH=0` run.

- [ ] **Step 4: Record results** in `docs/dev/weight-paging-vnext-validation.md` (append a "MoE-aware batched paging — resident gate" row) and commit.

---

## Task 8: Paged validation gate (DeepSeek V4 Flash — the production target)

**Files:** none (validation only). Requires the DeepSeek V4 Flash Q8 GGUF (~162GB) on mad-lab-main.

- [ ] **Step 1: VRAM/geometry pre-flight**

Confirm the GGUF path, then probe pager geometry with a tiny slots load (read the `wp::WeightPager: N pages, S slots x B budget` line) to size slots. Target: shared/dense weights resident, experts streaming (evictions>0 by construction since 162GB ≫ 32GB). Print VRAM math before launch (standing rule). Do NOT use `--direct-io` (prior fit-hang).

- [ ] **Step 2: Native (no-paging) reference is infeasible at 162GB** — instead establish the baseline as **per-op paged PPL** (`WP_PAGED_BATCH=0`, MoE per-op path, known-correct) and require batched paged PPL to match it within reduction-order noise.
```
ssh mad-lab-main bash -s <<'EOF'
cd /home/kmbandy/GitHub/llama.cpp
# baseline: per-op paged (correct path)
systemd-run --user --unit=t8-perop --collect -p MemoryMax=28000M --working-directory=/home/kmbandy/GitHub/llama.cpp \
  --setenv=WP_SIZE_CLASS_SLOTS=1 \
  bash -c "./build-hip/bin/llama-perplexity -m <DSV4_Q8_PATH> -f wikitext-2-raw/wiki.test.raw --chunks 4 -c 512 --no-mmap --weight-paging --weight-paging-slots <N> --weight-paging-prefetch -ngl 99 --device ROCm0 > /tmp/t8-perop.log 2>&1"
while systemctl --user is-active --quiet t8-perop.service; do sleep 10; done
grep -E 'Final estimate|evictions' /tmp/t8-perop.log
EOF
```
Expected: a finite PPL and `evictions > 0` (confirms genuine paging).

- [ ] **Step 3: Batched paged run (`WP_PAGED_BATCH=1`)** — same command with `--setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1`. Expected: PPL matches Step 2 within noise, **0 GPU faults**, `routing_ptrs_discarded_unconsumed` sane, and decode t/s (via llama-server) up vs the per-op baseline. Record prefetch hit-rate + eviction counts from the teardown summary.

- [ ] **Step 4: Decision** — if both gates pass (resident + paged), propose flipping `WP_PAGED_BATCH` default-on in a follow-up commit (its own review). Record all numbers in `docs/dev/weight-paging-vnext-validation.md` and the continuation doc; commit.

---

## Notes for the implementer

- Tasks 1 and 3 are pure-logic and fully unit-tested. Tasks 4-6 are GPU-integration; their "tests" are the hardware gates (exact commands + expected numbers above) because `test-weight-pager` has no GPU eval-cb harness — this matches the existing suite's `GGML_USE_HIP`-gated approach.
- Keep every change gated on `wp_paged_batch_enabled()`; a flag-OFF run must remain byte-identical to today (verify with the 27B `WP_PAGED_BATCH=0` PPL == 5.4623 at any checkpoint if in doubt).
- The `s_dev_expert_ptrs` static device buffer and the per-op routing path (Unit B) are unchanged — the fix is entirely about *when* the routing op runs (isolated), not *how* it builds pointers.
- If Task 5's pin lifecycle proves subtle on hardware (fault or PPL drift), that is the point to bring in a codex+Fable consult with the exact repro, per the session's established pattern.
