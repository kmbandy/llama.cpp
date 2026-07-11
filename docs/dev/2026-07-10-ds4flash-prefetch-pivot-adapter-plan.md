# DS4 Flash Decode — Cross-Layer Prefetch Result, the Pivot to DFlash-Driven Streaming, and the Adapter Plan

**Date:** 2026-07-10 (evening). **Branch:** `feat/wp-dflash-ds4` (mad-lab-main).
**Model:** DeepSeek-V4-Flash Q8 (~151 GB, 284B/13B-active, 256 experts, 43 layers), NVMe-weight-paged, `/home/kmbandy/models/ds4-nocow/`.
**One-line status:** Built + measured the cross-layer prefetch engine → it REGRESSES (no lead time, capacity-bound). Pivoted to the DFlash-driven predictive-streaming idea, reusing the same prediction substrate. Next action is a GPU-free-to-prep, morning-gated data-capture run to train + gate a DFlash→routing **adapter**.

---

## 1. What we built today (committed)

Executed the approved cross-layer prefetch plan (`docs/superpowers/plans/2026-07-10-cross-layer-prefetch-overlap.md`) inline, TDD, unit-tested. Commits on `feat/wp-dflash-ds4`:

- **`620cb8da5`** — checkpoint: banked prior DS4 bandwidth WIP + **io_uring `WeightPager` layout ABI fix** + **T1 RouterPredictor**.
- **`f61ec4f85`** — **T2** `(block,expert)→sister-pages` reverse index + **T3** `PoolAllocator` speculative-eviction tier.
- **`d71223efa`** — **T4/T5** cross-layer eval-cb wiring + config knobs + stats (all gated `WP_PREFETCH_XLAYER`, default off = byte-identical).

**The ABI-fix (important, general):** `WeightPager`'s three `io_uring` members were `#if defined(LLAMA_HAVE_IO_URING)`-guarded in `wp-pager.h`, but that macro is `PRIVATE` to the `llama` CMake target. So `test-weight-pager` (which lacks the macro) computed a smaller `sizeof(WeightPager)` than `libllama.so`, and libllama's ctor wrote past the test's stack object → `free(): invalid pointer` in `test_routing_boundary_prepass`. A stale test binary had hidden it. **Fix:** members are now unconditional (layout can't depend on a per-TU macro); `#if` guards remain only in the `.cpp`. This is a general hazard — a shared-header struct layout must never vary by a per-TU macro.

**The prefetch engine (T1–T5), as built:**
- `RouterPredictor` (`src/weight-pager/wp-router-predictor.{h,cpp}`): host copies of each layer's `ffn_gate_inp`; `predict(h, from_layer, K, M)` = plain top-M of `W[T]·h` for target layers `from_layer+1..+K`.
- `build_expert_page_index` / `WeightPager::expert_sister_pages` (`wp-pager.cpp`): O(1) `(block,expert)→pages`.
- `PoolAllocator` speculative tier (`wp-pool.cpp`): `set/is/n_speculative`, a "Pass 0" that evicts LRU speculative slots before the working set (both fixed + size-class paths), `mark_used` promotes.
- eval-cb hook (`wp-eval-cb.cpp`, at the `ffn_gate_inp` `MUL_MAT`): capture router weight once, D2H the residual, `submit_xlayer_prefetch` per token.
- `submit_xlayer_prefetch` (`wp-pager.cpp`): predict → sister pages → dedup vs resident → speculative cap → `prefetch_pages_batch(speculative=true)`.
- Config: `WP_PREFETCH_XLAYER / _LOOKAHEAD_K(2) / _TOPK(16) / _MAX_SLOTS(n_slots/4)`.

Full unit suite: **41/41 green.**

**Uncommitted diagnostic instrumentation** (in `wp-pager.{h,cpp}`, `wp-eval-cb.cpp`): `[xlayer]` counters (`predict_calls / pred_pages / resident_skips / blocked_budget / blocked_free_queue`, logged at WARN), `speculative_evicted_unused`, and a `WP_PREFETCH_XLAYER_NOSYNC` flag. Compile-verified, 41/41 green. Not committed pending the pivot decision.

---

## 2. The measured result — a regression, and WHY

User-gated GPU smoke + diagnostic runs (single-stream, `ds4-nocow`, 128-tok deterministic decode):

| config | t/s | vs base | gb/s | ensure_wait_ms | xlayer submitted | hit | text==base |
|---|---|---|---|---|---|---|---|
| baseline (off) | **1.629** | — | 2.663 | 44.1k | 0 | 0 | — |
| xlayer **sync** | 1.420 | **−12.8%** | 2.743 | 44.9k | 23 | 7 | YES |
| xlayer **nosync** | 1.467 | **−9.9%** | 2.747 | 44.9k | 23 | 7 | YES |

**Correctness is solid** (both reproduce baseline text exactly). But it's a net **~10–13% regression** with **negligible prefetch**. Two stacked causes:

1. **No lead time (the fatal flaw).** The engine predicts/prefetches *within the single token being decoded*, 1–2 layers ahead. Within one autoregressive token the layers run back-to-back and per-layer decode compute is tiny (13B active), so the prefetch has a sub-10 ms horizon — not enough to hide even one cold ~13 MB / ~5 ms NVMe expert read. `sync→nosync` shows the D2H stream-sync is only ~3% of the cost; the other **~10% is the per-layer host work itself** (D2H + GEMV + bookkeeping), unrecovered.
2. **Capacity-bound.** Only **23** pages were ever fresh-to-prefetch over 128 tokens × 43 layers (7 useful), and `ensure_wait_ms` was unchanged. The predicted experts are already resident; the cold-tail misses that cause the 44 s wait are capacity evictions prediction can't prevent. This re-confirms the earlier finding: **binding constraint = VRAM capacity, not predictability**, and the bandwidth-idle window can't be filled by a zero-lead predictor.

(Note: the `[xlayer]` counters were initially invisible because they were logged at `LLAMA_LOG_INFO` while the server filters to `WARN` — fixed to WARN, uncommitted. Not a bug in the engine.)

---

## 3. The pivot (user's direction): DFlash IS the lead time

The whole problem is horizon. The lever with a real horizon is **DFlash speculative decode**: it drafts ~4–8 tokens (block_size 8, ~4 reliably accepted ≈ **~1 s lead**). That's the window to stream the *next tokens'* experts in during the current compute — pushing the SSD from ~40% duty toward ~100% (the ceiling analysis: perfect overlap → ~2.9–3.7 t/s; see `docs/dev/2026-07-09-dflash-predictive-streaming-plan.md`).

**Design (agreed):** *normal* DFlash (unchanged decoding) → read its drafted-token hidden `inp_g` (already captured read-only via `WP_CAPTURE_DFLASH`) → a small learned **adapter** maps `inp_g` into the target's **router-input space** → run the **real target routers = the RouterPredictor we already built** on the drafted future tokens → prefetch their experts with real lead. **We reuse T1/T2/T3 as the whole back half; only the *feed* changes** (from within-token `ffn_gate_inp` → DFlash-draft-via-adapter). For DS4's **hash layers** (early), the expert is a deterministic `tid2eid(token)`, so the drafted token ID gives exact experts with no adapter (machinery partly exists); the adapter is only needed for the **softmax** (mid/late) layers.

**Why the adapter is unavoidable:** DFlash's `inp_g` fed *directly* into the target routers is random — measured **recall@6 = 0.017** (chance 0.023). `inp_g` lives in DFlash's own EAGLE3-fused space. The adapter is the small learned bridge.

Reuse table:

| piece | fate |
|---|---|
| RouterPredictor (T1) | reused (runs the real routers) |
| expert→pages index (T2) | reused |
| pool speculative tier (T3) | reused (draft experts ARE speculative) |
| eval-cb feed (T4) | replaced by the DFlash-draft feed via adapter |
| RAM victim tier (plan Task 2) | new, parallel, low-risk (catches mispredicts at ~28 GB/s) |

---

## 4. The adapter — how we get it, and the gate

**Training pairs = one decoded token each** (no labeling): `(DFlash g[pos], target routing[pos+1])`.
- input `g[pos]` (4096-d) from `dflash_capture.bin`.
- label from `routing_capture.bin`, which stores **both** the selected experts (`sel`) **and** the router-input residual per layer (dry-run confirmed residual dim = **4096**, so path-b is viable).

**Two adapter formulations (train both, keep whichever clears the gate):**
- **path_a (fused):** `A_L: g → 256 logits`, experts = topP(A_L·g). Directly optimizes recall; tiny capture (only `sel` needed).
- **path_b (reuse routers):** `B_L: g → router-input(4096)`, experts = topP(W_L·(B_L·g)). Keeps the RouterPredictor intact; needs the bulky residual capture.

Per-tap-layer (or per-layer), start **linear/low-rank** (closed-form ridge, numpy, no GPU); PCA-reduce `g` (e.g. 128 comps) for conditioning + speed. **GATE: val recall@6 ≥ 0.60**, train/val split **by prompt** (no leakage). If it clears, the prefetch finally has a horizon; if not, fall back to transport-only (~2.9 ceiling) and prediction-prefetch is dead.

**Data volume:** current `dflash_capture.bin` is only ~135 positions from one prompt (overfits). Need **a few thousand pairs across diverse prompts** = one ~40–80 min gated inference run generating ~3k tokens over ~12 varied prompts.

**Dry-run validation (offline, done today):** `adapter_train.py` reproduced the known baseline exactly (`recall@6 = 0.017`), confirmed residual dim 4096, and the full train path runs; on the overfit single-prompt data a trained PCA-ridge adapter lifted in-sample recall@6 to **0.275 (path_a) / 0.175 (path_b)** — meaningless as a gate (one prompt, position split) but proves the signal is *learnable*.

---

## 5. NEXT ACTION — the morning run (prepped, single launch)

All artifacts in `/home/kmbandy/wp_logs/accounting/adapter/`:
- `capture-pairs.sh` — 2 GPU phases: (A) non-spec `WP_CAPTURE_ROUTING=1` → clean routing; (B) DFlash spec `WP_CAPTURE_DFLASH=1` → drafted hidden. 12 diverse prompts, both phases, lossless-greedy → aligned; writes `manifest_{routing,dflash}.txt`. Backs up existing captures to `.pre-adapter.bak`.
- `adapter_train.py` — numpy ridge trainer + gate (validated).

**Launch (fires on go-ahead):**
```
DRAFT_DEV=ROCm0 bash /home/kmbandy/wp_logs/accounting/adapter/capture-pairs.sh
```
Then offline: `python3 /home/kmbandy/wp_logs/accounting/adapter/adapter_train.py --pca 128`

**VRAM math (per the ASK-FIRST rule):** pool = **27,625 MiB** on ROCm1 (R9700, 32 GB) both phases.
- Phase A (no draft): ~27.6 GB pool + ~1.5 GB KV/act ≈ 29 GB → ~3 GB headroom (= smoke-run footprint, proven).
- Phase B (spec): draft (1.9 GB) defaulted to **ROCm0** (6900 XT, ~1 GB resident) so ROCm1 stays ~29 GB / ~3 GB headroom; ROCm0 ~3 GB of 16 GB. Both safe. `DRAFT_DEV=ROCm1` = original proven-but-tighter (~1 GB headroom).
- Timeline: 2 model loads (~10 min each) + ~30 min decode each ≈ **~80 min**. Knobs: `NPRED` (default 256), prompt count.

**GPU RULE (do not forget):** never launch any GPU work — including this — without asking kmbandy and WAITING; multiple sessions share the box. (Violated once today by auto-launching a diag run off a menu selection; corrected.)

---

## 6. After the gate

- **If recall@6 ≥ 0.60:** wire the adapter as the new prefetch feed — DFlash `inp_g` (read at the existing `speculative.cpp` hook) → adapter → RouterPredictor → sister pages → speculative prefetch, submitted during the draft/verify window. Add `tid2eid` hash-layer prefetch (exact, full block lead). Then measure toward ~3 t/s with isolated A/Bs (no combined-lever runs). Add the RAM victim tier (plan Task 2) for mispredicts.
- **If it fails / is marginal:** try low-rank/tiny-MLP + tap-layer pooling + more data; if still short, prediction-prefetch is dead → transport-only ceiling (~2.9) and shift to depth-aware residency / more pinned VRAM (capacity is the true lever).

---

## 7. Artifact & state inventory

- Commits: `620cb8da5`, `f61ec4f85`, `d71223efa` on `feat/wp-dflash-ds4`.
- Uncommitted (repo): diagnostic counters + `WP_PREFETCH_XLAYER_NOSYNC` + WARN log fix in `wp-pager.{h,cpp}`, `wp-eval-cb.cpp`. Also present: another session's edits (aiter `occ_kernel_dsws_flow.s`, dsws review .md) — NOT ours.
- Adapter work (outside repo): `/home/kmbandy/wp_logs/accounting/adapter/{capture-pairs.sh, adapter_train.py, dryrun*.out}`; captures `dflash_capture.bin` (135 pos), `routing_capture.bin`; `dflash-align.py` (original direct-projection test).
- Sweep scripts (superseded but kept): `/home/kmbandy/wp_logs/accounting/{xlayer-sweep.sh, xlayer-diag.sh}` + result logs.
- Related docs: `2026-07-09-dflash-predictive-streaming-plan.md` (the 3-tier + ceiling), `2026-07-08-wp-hetero-dflash-oracle-plan.md` (tid2eid oracle), `2026-07-09-ds4flash-strategy-compute-reopening.md` (session log), specs/plans for the cross-layer engine.

## 8. Standing constraints (persist)
`--no-mmap` always; single-stream `--parallel 1`; NO more quantizing; GPU logs → `/home/kmbandy/wp_logs`; edit repo ON mad-lab-main via ssh; commit only when asked; **ASK before any GPU run and WAIT**; when building, build completely and flag failed/incomplete work explicitly.
