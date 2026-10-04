# Cross-Layer Prefetch-Overlap Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Predict layer L+k's routed experts from layer L's live residual (host-side router GEMV) and prefetch them during layer L's compute, so the NVMe queue stays saturated and effective paging bandwidth rises from ~2.6 toward the proven ~5.8 GB/s ceiling.

**Architecture:** A host-side `RouterPredictor` (CPU GEMV of resident `ffn_gate_inp` against the captured residual, plain top-M) feeds the existing `PrefetchScheduler` via a new speculative eviction tier in `PoolAllocator`. Zero GPU-graph changes. All new behavior gated behind `WP_PREFETCH_XLAYER=1`; default path stays byte-identical.

**Tech Stack:** C++17, ggml/HIP (ROCm), llama.cpp weight-pager subsystem, bespoke `EXPECT` test harness in `tests/test-weight-pager.cpp`.

## Global Constraints
- `--no-mmap` always; no further quantizing; single-stream (`--parallel 1`).
- `WP_PREFETCH_XLAYER=0` (default) MUST be byte-identical to current behavior.
- Speculation MUST NEVER evict a pinned/working-set page (the footgun).
- Edit all files ON mad-lab-main via ssh; GPU runs are manual + user-gated; commit only when asked.
- Model is NOCOW at `/home/kmbandy/models/ds4-nocow/`; nocow buffered p2p baseline = 1.81 t/s / 2.6 GB/s.
- Build: `cmake --build build-hip --target llama-server test-weight-pager -j$(nproc --ignore=2)` (clang++ from /opt/rocm/llvm/bin).

## File Structure
- **Create** `src/weight-pager/wp-router-predictor.h` / `.cpp` — RouterPredictor: host `ffn_gate_inp` copies + `predict()`. One responsibility: residual → predicted (layer,expert) refs.
- **Modify** `src/weight-pager/wp-pool.h` / `.cpp` — speculative eviction tier (flag + evict-speculative-first + promote + cap query).
- **Modify** `src/weight-pager/wp-pager.h` / `.cpp` — own the RouterPredictor + `(block,expert)→pages` reverse index + config + stats.
- **Modify** `src/weight-pager/wp-eval-cb.cpp` — capture `ffn_gate_inp` lazily; at gate op, predict + submit prefetch (speculative); DFlash union.
- **Modify** `tests/test-weight-pager.cpp` — unit tests for predictor math, reverse index, speculative eviction.

---

### Task 1: RouterPredictor (host GEMV + top-M)

**Files:**
- Create: `src/weight-pager/wp-router-predictor.h`, `src/weight-pager/wp-router-predictor.cpp`
- Test: `tests/test-weight-pager.cpp` (new `test_router_predictor`)

**Interfaces:**
- Produces: `struct ExpertRef { int layer; int expert; };`
  `class RouterPredictor { void set_router(int layer, const float* W_row_major_256x4096, int n_expert, int n_embd); bool has_router(int layer) const; void predict(const float* h, int from_layer, int K, int M, int n_layer, std::vector<ExpertRef>& out) const; };`
- `set_router` stores a per-layer f32 copy of `ffn_gate_inp` (`[n_expert, n_embd]`, row-major so row e is expert e's weight vector). `predict` computes, for d in 1..K with T=from_layer+d < n_layer and `has_router(T)`, logits[e] = dot(W[T][e], h) over n_embd, then appends the top-M experts (partial-sort by logit desc) as `{T, e}` to `out`. No allocation of W inside predict.

- [ ] **Step 1: Write the failing test** (append to `tests/test-weight-pager.cpp`, register in `main`)
```cpp
static void test_router_predictor() {
    using namespace wp;
    RouterPredictor rp;
    const int n_expert = 4, n_embd = 3;
    // layer 1 router: expert 2 aligns with h=(1,0,0); expert 0 second.
    float W1[n_expert*n_embd] = {
        0.5f,0,0,   // e0
        0,1,0,      // e1
        1,0,0,      // e2 (max dot with h)
        0,0,1 };    // e3
    rp.set_router(/*layer=*/1, W1, n_expert, n_embd);
    EXPECT(rp.has_router(1), "router present after set");
    EXPECT(!rp.has_router(2), "router absent for unset layer");
    float h[n_embd] = {1.0f, 0.0f, 0.0f};
    std::vector<ExpertRef> out;
    rp.predict(h, /*from_layer=*/0, /*K=*/1, /*M=*/2, /*n_layer=*/43, out);
    EXPECT_EQ_INT((int)out.size(), 2, "K=1,M=2 -> 2 refs");
    EXPECT_EQ_INT(out[0].layer, 1, "predicted target layer");
    EXPECT_EQ_INT(out[0].expert, 2, "top-1 expert is e2");
    EXPECT_EQ_INT(out[1].expert, 0, "top-2 expert is e0");
    // K beyond n_layer or unset router -> no refs
    out.clear();
    rp.predict(h, /*from_layer=*/1, /*K=*/1, /*M=*/2, /*n_layer=*/43, out); // target 2 unset
    EXPECT_EQ_INT((int)out.size(), 0, "unset target router -> no refs");
}
```
- [ ] **Step 2: Run to verify it fails** — `cmake --build build-hip --target test-weight-pager -j$(nproc --ignore=2)` → FAIL (undefined `RouterPredictor`).
- [ ] **Step 3: Implement `wp-router-predictor.h`**
```cpp
#pragma once
#include <vector>
#include <cstdint>
namespace wp {
struct ExpertRef { int layer; int expert; };
class RouterPredictor {
public:
    void set_router(int layer, const float* W, int n_expert, int n_embd);
    bool has_router(int layer) const;
    // Append top-M experts for each target layer from_layer+1..from_layer+K
    // (that has a router and is < n_layer) to out. Plain top-M of W[T].h.
    void predict(const float* h, int from_layer, int K, int M,
                 int n_layer, std::vector<ExpertRef>& out) const;
    int n_expert() const { return n_expert_; }
    int n_embd()   const { return n_embd_; }
private:
    struct Router { std::vector<float> W; }; // [n_expert*n_embd] row-major, empty=unset
    std::vector<Router> routers_;            // indexed by layer
    int n_expert_ = 0, n_embd_ = 0;
};
} // namespace wp
```
- [ ] **Step 4: Implement `wp-router-predictor.cpp`**
```cpp
#include "wp-router-predictor.h"
#include <algorithm>
namespace wp {
void RouterPredictor::set_router(int layer, const float* W, int n_expert, int n_embd) {
    if (layer < 0 || W == nullptr || n_expert <= 0 || n_embd <= 0) return;
    if ((int) routers_.size() <= layer) routers_.resize(layer + 1);
    n_expert_ = n_expert; n_embd_ = n_embd;
    routers_[layer].W.assign(W, W + (size_t) n_expert * n_embd);
}
bool RouterPredictor::has_router(int layer) const {
    return layer >= 0 && layer < (int) routers_.size() && !routers_[layer].W.empty();
}
void RouterPredictor::predict(const float* h, int from_layer, int K, int M,
                              int n_layer, std::vector<ExpertRef>& out) const {
    if (h == nullptr || K <= 0 || M <= 0) return;
    std::vector<std::pair<float,int>> logits((size_t) n_expert_);
    for (int d = 1; d <= K; ++d) {
        const int T = from_layer + d;
        if (T >= n_layer || !has_router(T)) continue;
        const float* W = routers_[T].W.data();
        for (int e = 0; e < n_expert_; ++e) {
            const float* w = W + (size_t) e * n_embd_;
            float s = 0.0f;
            for (int j = 0; j < n_embd_; ++j) s += w[j] * h[j];
            logits[(size_t) e] = { s, e };
        }
        const int m = std::min(M, n_expert_);
        std::partial_sort(logits.begin(), logits.begin() + m, logits.end(),
                          [](const std::pair<float,int>& a, const std::pair<float,int>& b){ return a.first > b.first; });
        for (int i = 0; i < m; ++i) out.push_back(ExpertRef{ T, logits[(size_t) i].second });
    }
}
} // namespace wp
```
- [ ] **Step 5: Add `wp-router-predictor.cpp` to the weight-pager build** — in `src/weight-pager/CMakeLists.txt` (or wherever `wp-prefetch.cpp` is listed), add `wp-router-predictor.cpp`. Verify with `grep -rn wp-prefetch.cpp src/weight-pager/CMakeLists.txt ggml/**/CMakeLists.txt` first and mirror that line.
- [ ] **Step 6: Run to verify it passes** — build `test-weight-pager` → `PASS test_router_predictor`; full suite still green (37+).
- [ ] **Step 7: Commit** (only when the user asks): `git add src/weight-pager/wp-router-predictor.* src/weight-pager/CMakeLists.txt tests/test-weight-pager.cpp && git commit -m "feat(wp): RouterPredictor host-side top-M cross-layer expert prediction"`

---

### Task 2: (block,expert) → sister-pages reverse index

**Files:**
- Modify: `src/weight-pager/wp-pager.h` (declare), `src/weight-pager/wp-pager.cpp` (build at init from catalog)
- Test: `tests/test-weight-pager.cpp` (new `test_expert_page_index`)

**Interfaces:**
- Produces on `WeightPager`: `void expert_sister_pages(int block_idx, int expert_idx, std::vector<int>& out) const;` — appends the gate/up/down page indices for (block,expert). Built from the catalog: for each page `p`, if `meta(p).expert_idx >= 0`, insert into `expert_pages_[{block_idx,expert_idx}].push_back(p)`.

- [ ] **Step 1: Write the failing test** — build a small `PageCatalog` with consolidated expert pages (mirror existing `test_catalog_*` insert style: names like `blk.5.ffn_gate_exps.weight` etc. with parsed `block_idx`/`expert_idx`/`role_mask`), construct the reverse index helper (extract it as a free function `wp::build_expert_page_index(const PageCatalog&, std::map<std::pair<int,int>,std::vector<int>>&)` so it is unit-testable without a full pager), assert `(block=5,expert=3)` returns the 3 sister page indices and `(block=99,expert=0)` returns empty.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** the free function in `wp-pager.cpp` (declared in `wp-pager.h`): iterate `cat.size()`, read `cat.at(i)`, key `{m.block_idx, m.expert_idx}` when `m.expert_idx >= 0`, push `i`. `WeightPager::init` calls it into member `expert_page_index_`; `expert_sister_pages` looks up and appends.
- [ ] **Step 4: Run to verify it passes.**
- [ ] **Step 5: Commit** (on request): `feat(wp): (block,expert)->sister page reverse index`

---

### Task 3: Speculative eviction tier in PoolAllocator

**Files:**
- Modify: `src/weight-pager/wp-pool.h`, `src/weight-pager/wp-pool.cpp`
- Test: `tests/test-weight-pager.cpp` (new `test_pool_speculative`)

**Interfaces:**
- Produces on `PoolAllocator`: `void set_speculative(int slot, bool spec);` `bool is_speculative(int slot) const;` `int n_speculative() const;`
- Behavior change in `alloc_slot`: victim selection gains a **pass 0** that evicts an unpinned **speculative** slot (LRU among speculative) before the existing cold/hot LRU passes. A slot allocated normally is non-speculative (`speculative_[slot]=false`). `mark_used(slot)` clears `speculative_[slot]` (promotion on demand-hit). `set_speculative(slot,true)` is called by the pager right after a prefetch claims a slot.

- [ ] **Step 1: Write the failing test**
```cpp
static void test_pool_speculative() {
    using namespace wp;
    PoolAllocator pool;
    // init a tiny CPU-backed pool of 3 slots (mirror existing test_pool_allocator init;
    // if init requires a real ggml buffer, use the same buft the other pool tests use).
    // ... init pool with n_slots=3 ...
    int a = pool.alloc_slot(); int b = pool.alloc_slot(); int c = pool.alloc_slot();
    pool.set_speculative(b, true);                 // b is a speculative prefetch
    pool.pin_slot(a); pool.pin_slot(c);            // a,c are working set (pinned)
    int d = pool.alloc_slot();                     // must evict b (speculative), not a/c
    EXPECT_EQ_INT(d, b, "alloc evicts the speculative slot first");
    EXPECT(!pool.is_speculative(d), "reused slot is non-speculative");
    // promotion: a speculative slot that gets mark_used is no longer speculative
    pool.set_speculative(d, true);
    pool.mark_used(d);
    EXPECT(!pool.is_speculative(d), "mark_used promotes (clears speculative)");
}
```
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** — add `std::vector<char> speculative_;` sized with the pool; `set_speculative`/`is_speculative`/`n_speculative`; in `alloc_slot`, before the cold/hot LRU walk, scan for an unpinned slot with `speculative_[s]` (pick LRU by existing recency ordering) and evict it if found; `mark_used` sets `speculative_[slot]=false`; fresh `alloc_slot` returns a slot with `speculative_[slot]=false`. Keep pinned-never-evicted invariant in all passes.
- [ ] **Step 4: Run to verify it passes** — `PASS test_pool_speculative`; existing `test_pool_allocator` still green (non-speculative path unchanged).
- [ ] **Step 5: Commit** (on request): `feat(wp): pool speculative eviction tier (evict-first + promote-on-use)`

---

### Task 4: eval_cb invocation + pager wiring (integration; gated)

**Files:**
- Modify: `src/weight-pager/wp-pager.h`/`.cpp` (own `RouterPredictor predictor_;`, config fields, submit helper), `src/weight-pager/wp-eval-cb.cpp` (capture router weights + drive prediction)

**Interfaces:**
- Consumes: `RouterPredictor` (Task 1), `expert_sister_pages` (Task 2), `PoolAllocator::set_speculative` (Task 3), existing `PrefetchScheduler::submit_batch`, existing `mark_cross_layer_prefetch_candidates`.
- Produces on `WeightPager`: `void note_router_weight(int block_idx, const float* W, int n_expert, int n_embd);` (forwards to `predictor_.set_router`) and `void submit_xlayer_prefetch(const float* h, int from_layer);` (predict → expand to sister pages via `expert_sister_pages` → dedupe vs resident/in-flight → `alloc_slot_no_evict`-or-`alloc_slot` a slot per page, `set_speculative(slot,true)`, build `PrefetchBatchRequest` from `catalog_.at(page)` (file_idx→fd, file_offset, size) + `slot_ptr`, `prefetch_.submit_batch(reqs)`, and `mark_cross_layer_prefetch_candidates(pages)`).

- [ ] **Step 1: Lazy router capture in eval_cb.** In `weight_pager_eval_cb`, where ops are inspected (near the existing gate handling ~line 546/714), add: when `t->op == GGML_OP_MUL_MAT` and `ggml_get_name(t->src[0])` matches `blk.<L>.ffn_gate_inp.weight` and `!pager->predictor_has_router(L)`, D2H-copy `t->src[0]->data` ([n_expert, n_embd] BF16) into an f32 temp and call `pager->note_router_weight(L, tmp, n_expert, n_embd)`. Guard the whole block behind `WP_PREFETCH_XLAYER` (parse once, static). (BF16→f32: `(uint32(bits)<<16)` reinterpret.)
- [ ] **Step 2: Drive prediction at the gate op.** At the existing routing-break gate op where `h_L = t->src[1]` is already D2H'd for capture (~line 714-740): when `WP_PREFETCH_XLAYER` is on, convert `h_L` to f32 (handle F16/F32 by `ri->type`), and for each token row (n_tok, DFlash union), call `pager->submit_xlayer_prefetch(h_row, meta.block_idx)`. Union across draft rows by collecting predicted pages into a `std::vector` + a `seen` set before submit.
- [ ] **Step 3: Implement `submit_xlayer_prefetch` + `note_router_weight`** in `wp-pager.cpp` per the Interfaces block. Skip pages already `page_loaded_` or `page_to_slot_>=0`. For each fresh page, `int slot = pool_.alloc_slot(size)` (may evict a speculative/cold slot — NEVER a pinned one, guaranteed by Task 3), `pool_.set_speculative(slot, true)`, set `page_to_slot_`/`slot_to_page_` maps, append a `PrefetchBatchRequest`. Cap total in-flight speculative pages at `WP_PREFETCH_MAX_SLOTS` (skip submit once `pool_.n_speculative() >= cap`). Bump `stats_.cross_layer_prefetch_submitted`.
- [ ] **Step 4: Verify default path byte-identical.** Build `llama-server`; run `bash ~/wp_logs/accounting/buffered-nocow-validate.sh` (WP_PREFETCH_XLAYER unset) → t/s within noise of 1.81 and identical decoded text vs a saved baseline completion. Expected: no behavior change when disabled.
- [ ] **Step 5: Commit** (on request): `feat(wp): cross-layer router prefetch wiring (gated WP_PREFETCH_XLAYER)`

---

### Task 5: Config knobs + stats

**Files:** Modify `src/weight-pager/wp-pager.cpp` (env parse + stats block), `src/weight-pager/wp-pager.h` (stat fields).

- [ ] **Step 1:** Parse once at pager init (static-local getenv): `WP_PREFETCH_XLAYER` (bool, default 0), `WP_PREFETCH_LOOKAHEAD_K` (int, default 2), `WP_PREFETCH_TOPK` (int, default 16), `WP_PREFETCH_MAX_SLOTS` (int, default = n_slots/4). Store on the pager; pass K/M into `submit_xlayer_prefetch`.
- [ ] **Step 2:** Add stats to the existing print block (mirror `cross_layer_prefetch_submitted` at wp-pager.cpp ~995/1124): `xlayer_recall_top6`, `xlayer_recall_topM` (running: predicted∩actual / actual, updated in `ensure_batch` by comparing demanded pages to the speculative bitmap), `speculative_evicted_unused` (increment in the pool eviction callback when an evicted slot was speculative and never `mark_used`).
- [ ] **Step 3:** Log the config line at init: `LLAMA_LOG_INFO("wp::xlayer prefetch: on K=%d M=%d cap=%d\n", ...)` only when enabled.
- [ ] **Step 4: Commit** (on request): `feat(wp): xlayer prefetch config knobs + recall/waste stats`

---

### Task 6: GPU integration + sweep (user-gated)

**Files:** Create `~/wp_logs/accounting/xlayer-sweep.sh` (mirror `buffered-nocow-validate.sh` + env matrix).

- [ ] **Step 1:** Script a matrix over `WP_PREFETCH_XLAYER=1` with `K in {1,2,3}`, `TOPK in {6,16,32}`, `MAX_SLOTS in {512,1024,2048}`, each a 128-tok deterministic decode on `ds4-nocow`, logging `predicted_per_second`, `ensure_batch_gb_s`, `ensure_batch_wait_ms`, `cross_layer_hit_in_ensure`, `xlayer_recall_*`, `speculative_evicted_unused`.
- [ ] **Step 2:** VRAM note in the script header: pool stays 6500 slots (28.9 GB) — speculation is a redistribution WITHIN it, no new allocation.
- [ ] **Step 3 (user-gated, ask before running):** run the sweep; read off (a) does recall match offline 0.64/0.82, (b) does `ensure_batch_gb_s` rise above 2.6, (c) best t/s vs 1.81, (d) the MAX_SLOTS that maximizes t/s (the VRAM-split answer).
- [ ] **Step 4:** Record results + chosen defaults back into `docs/dev/2026-07-09-ds4flash-strategy-compute-reopening.md`.

---

## Self-Review
- **Spec coverage:** RouterPredictor (T1) ✓; eval-cb invocation + DFlash union (T4) ✓; speculative eviction tier + cap (T3) ✓; config knobs (T5) ✓; stats (T5) ✓; unit + integration tests (T1/T2/T3 unit, T6 GPU) ✓; (block,expert)→pages (T2, implied by spec's "expand to 3 sister pages") ✓.
- **Type consistency:** `ExpertRef{int layer;int expert;}`, `RouterPredictor::predict(const float*,int,int,int,int,std::vector<ExpertRef>&)`, `PoolAllocator::set_speculative(int,bool)`, `WeightPager::expert_sister_pages(int,int,std::vector<int>&)`, `submit_xlayer_prefetch(const float*,int)` used consistently across T1/T2/T3/T4.
- **Gaps/notes:** router weight dtype is BF16 in-model → f32 host copy (T4 step 1); h_L dtype F16/F32 handled by `ri->type` (T4 step 2); grouped-topk exact routing intentionally omitted (spec Decision ①, non-goal).
