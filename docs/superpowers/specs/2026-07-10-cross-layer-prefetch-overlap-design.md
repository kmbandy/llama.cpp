# Cross-Layer Prefetch-Overlap Engine — Design (2026-07-10)

Branch: `feat/wp-dflash-ds4` (mad-lab-main). Companion to `docs/dev/2026-07-09-ds4flash-strategy-compute-reopening.md`.

## Goal
Keep the NVMe read queue saturated during compute so effective expert-paging bandwidth rises from ~2.6 GB/s (demand-only, buffered) toward the proven ~5.8 GB/s ceiling, and the ~40% SSD duty cycle climbs. Mechanism: predict layer L+k's routed experts from layer L's live residual and prefetch them during layer L's compute. Must run with DFlash spec-decode enabled.

## Background / measured ground truth
- Routed-expert pool = 11,008 `(layer,expert)` slots x 13.37 MB = 147 GB. R9700 holds 6500 slots ≈ 28.9 GB ≈ 20% of the pool.
- Demand-only paging: ~2.6 GB/s active, ~40% duty (SSD idle during compute), 1.81 t/s (nocow buffered p2p).
- Standalone O_DIRECT proves the SSD sustains ~5.8 GB/s at QD4 — but the pager reads in ~6-page bursts with a barrier per `ensure_batch`, draining the queue between batches. Sustaining QD requires reads issued AHEAD of demand.
- Cross-layer routing predictability (measured, NO training): applying layer L+k's `ffn_gate_inp` router to layer L's router-input residual predicts L+k's experts at recall 0.64@top6 / 0.82@top16 (k=1). This is the prediction engine.
- Cross-TOKEN prediction has ~0 locality (prior prefetch 0% hit) — NOT used. DFlash-as-predictor is shelved (adapter unbuilt, projects at chance); DFlash stays a co-resident spec-decode lever on the 6900XT.

## Architecture
Approach 1: a **host-side router predictor** feeding the **existing** `PrefetchScheduler`. Zero GPU-graph changes. The predictor GEMV is trivially cheap and deterministic (unit-testable). Reuses the built async two-stage prefetch pipeline, the `cross_layer_prefetch_candidate_` bitmap, and the `cross_layer_hit_in_ensure` stat.

## Components

### 1. RouterPredictor — new `src/weight-pager/wp-router-predictor.{h,cpp}`
- Responsibility: given a router-input residual `h` (n_embd=4096) captured at layer L, predict top-M experts per target layer in [L+1, L+K].
- State: host-resident copies of `ffn_gate_inp` per layer (`W[layer] = [256, 4096]`), loaded once at init (~90 MB BF16). Converted to the compute dtype the CPU GEMV uses.
- API: `void predict(const float* h, int from_layer, int K, int M, std::vector<ExpertRef>& out)` where `struct ExpertRef { int layer; int expert; };`. Computes `W[L+d] . h` (256x4096 GEMV) for d in 1..K, takes top-M per target layer.
- DECISION (1): **plain top-M** of `W.h` (matches the measured 0.64/0.82 recall), NOT DeepSeek's grouped-topk+sigmoid+bias. Prefetch wants recall, not exact routing; grouped-topk is a later recall-tuning option, not required.
- Testable: deterministic. Unit test feeds `~/wp_logs/accounting/routing_capture.bin` residuals + real `W`, asserts recall matches offline `analyze-routing.py` (0.64@top6 / 0.82@top16 at k=1).

### 2. Predictor invocation — modify `src/weight-pager/wp-eval-cb.cpp`
- At each gate op (layer L): reuse the existing `h_L` D2H (promote it from `WP_CAPTURE_ROUTING`-only to always-on when prefetch is enabled). Call `predict(h_L, L, K, M)`.
- Expand each `(layer,expert)` to its 3 sister pages (gate/up/down) via the catalog; dedupe vs resident/in-flight pages; submit via `PrefetchScheduler.submit_batch()` into the speculative tier; mark `cross_layer_prefetch_candidate_` for hit accounting.
- DFlash batch>1: when the pass has `n_draft` tokens, `h_L` is `[n_draft, 4096]`. Predict per draft token and UNION the predicted experts (capped by `WP_PREFETCH_TOPK` * a union factor) so draft-verification passes prefetch the right union.

### 3. Pool speculative eviction tier — modify `src/weight-pager/wp-pool.{h,cpp}` (+ wp-pager wiring)
- The footgun fix. Prefetched pages are flagged **speculative**. Eviction victim selection order: (1) speculative pages (LRU within them), (2) normal LRU working-set pages, (3) NEVER a pinned page. So speculation can never evict the hot working set; wrong predictions are the first to go.
- On an `ensure_batch` HIT of a speculative page (prediction was right → page actually demanded), PROMOTE it: clear the speculative flag → it becomes a normal working-set LRU page (and is pinned for the op as usual).
- Cap: `WP_PREFETCH_MAX_SLOTS` bounds how many pool slots speculation may hold at once. This is the VRAM-split knob — a REDISTRIBUTION of the existing 6500-slot pool (no new allocation, stays within the proven 28.9 GB).
- DECISION (2): priority-tier + cap, NOT a hard reserved partition — prefetch expands into idle capacity but the working set reclaims speculative slots automatically when it grows.

### 4. Config knobs (the sweep surface)
- `WP_PREFETCH_XLAYER=1` — enable the cross-layer router prefetch (default off; default path stays byte-identical).
- `WP_PREFETCH_LOOKAHEAD_K` — predictor depth in layers ahead (default 2).
- `WP_PREFETCH_TOPK` — M experts per target layer (default 16).
- `WP_PREFETCH_MAX_SLOTS` — speculative-slot cap = the VRAM-split knob.
- The old recency-sourced cross-layer path (`WP_NEXT_LAYER_PREFETCH_K`) is disabled/superseded (it was the 0%-hit source).

### 5. Stats — extend pager stats block
- Online predictor recall (predicted ∩ actual / actual, per layer-ahead).
- `cross_layer_prefetch_submitted`, `cross_layer_hit_in_ensure` (exist).
- `speculative_evicted_unused` (mispredictions evicted without ever being demanded).
- Existing `ensure_batch_gb_s`, `ensure_batch_wait_ms`, page_ins, duty proxy.
- These drive the K/M/cap and VRAM-split sweep.

### 6. Testing
- Unit (CPU, in `tests/test-weight-pager.cpp`, bespoke EXPECT harness): RouterPredictor recall vs offline numbers; pool eviction-priority (speculative evicted before working-set; demand-hit promotes; pinned never evicted); DFlash union path.
- Integration (GPU, user-gated): end-to-end decode on ds4-nocow — recall + `ensure_batch_gb_s` + t/s vs the 1.81 baseline; then sweep K in {1,2,3}, M in {6,16,32}, cap in {512,1024,2048}.

## Success criteria
- Predictor online recall matches offline (≈0.6+@top6, ≈0.8+@top16).
- `ensure_batch_gb_s` rises above 2.6 toward ~5; SSD duty cycle up; `cross_layer_hit_in_ensure` materially > 0.
- End-to-end t/s > 1.81 (nocow buffered baseline).
- `WP_PREFETCH_XLAYER=0` (default) is byte-identical to current behavior.
- Working set is never evicted by speculation (unit-proven).

## Non-goals
- DFlash→routing adapter / cross-token prediction (parked).
- Grouped-topk exact routing (option, later).
- The pager O_DIRECT sustained-QD path (separate; buffered p2p is the substrate here).

## VRAM safety
Prefetch consumes slots WITHIN the existing 6500-slot pool (redistribution, not new allocation). No new VRAM math vs the proven config.

## Global constraints
`--no-mmap` always; no further quantizing; single-stream (`--parallel 1`); GPU runs are manual + user-gated; commit only when asked; model is NOCOW at `/home/kmbandy/models/ds4-nocow/`.

## Files
- Create: `src/weight-pager/wp-router-predictor.{h,cpp}`
- Modify: `src/weight-pager/wp-eval-cb.cpp`, `src/weight-pager/wp-pool.{h,cpp}`, `src/weight-pager/wp-pager.{h,cpp}`, `tests/test-weight-pager.cpp`
