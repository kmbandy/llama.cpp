# Weight-paging decode-speed work — continuation (morning pickup)

**Branch:** `feat/wp-vnext` (edit/build-check on **mad-lab-2026** at `/home/kmbandy/GitHub/llama.cpp`).
Pushed to `origin`; **mad-lab-main** is on `feat/dsws-phaseb-conversion` with the wp-vnext commits
cherry-picked. All GPU validation is on mad-lab-main's **R9700 (ROCm0, gfx1201, 32 GB)**; the 6900XT
is ROCm1. Remote shell is **fish** → always `ssh mad-lab-main bash -s <<'EOF'`.

---
---

# ★★★ 2026-07-07 FULL DAY — READ THIS FIRST ★★★
## WP_PAGED_BATCH shipped + DeepSeek V4 Flash (284B/162GB) RUNS on 32 GB. Next front: decode-speed I/O parallelism.

### TL;DR of the whole day
1. **Shipped the `WP_PAGED_BATCH` feature** (MoE-aware batched paging) — 9 commits, validated across
   **dense/MoE × resident/paged** (all 4 quadrants: correct PPL, zero faults). Fixes the batched-MoE
   near-null fault (H3 read-before-produce + H4 TLS take-steal) via routing-boundary breaks, and makes
   batching safe under **eviction** via a per-range pin lifecycle + reactive auto-break.
2. **Fixed 3 DeepSeek-V4 × weight-paging load bugs** (fit-dry-run crash, tiny-weights-paged,
   view-only-leaf buffer_id). Root-caused the last two via **codex+Fable consults** converging on
   `ggml_backend_sched_split_graph` pass-4.
3. **THE WIN: DeepSeek V4 Flash (284B total / 13B active, 162 GB Q8) runs paged on the 32 GB R9700.**
   Wikitext **PPL = 4.1524**, **0 GPU faults**, **62,233 evictions/run**, `routing_ptrs_discarded=0`
   (clean routing). A model **5× the size of VRAM**, computing correctly.
4. **NEXT FRONT (start here after compact): decode-speed I/O parallelism.** DeepSeek decode is
   **~0.04 t/s**, gated by **~0.5 GB/s effective I/O = ~1/10th of the SN850's ~5-7 GB/s**. It is
   **QD≈1-serialized** in the expert page-in (`page_in_sync_` one-page-at-a-time), so P2P dma_buf,
   io_uring depth 16, and async ensure **did not help** (async made it worse). The fix is code, not
   flags: **issue a MoE op's N active-expert reads concurrently to fill the io_uring queue**, then wait.

### Commit stack today (all on `feat/wp-vnext`, pushed; tip `d384d93f4`)
```
d384d93f4 fix(wp): anchor paged weights to pool_buf so view-only leaves get a backend   (DeepSeek fix #3)
3faad72a0 fix(wp): keep tiny per-layer weights resident (don't page < 4KB)               (DeepSeek fix #2)
978a720bf fix(wp): disable weight paging in the memory-fit dry-run                        (DeepSeek fix #1)
f25563762 feat(wp): reactive auto-break on pin-budget under WP_PAGED_BATCH (5b)
2854c4f97 feat(wp): per-range pin lifecycle under WP_PAGED_BATCH (release after range sync)  (5a)
0c9a6e159 feat(wp): break batch range at routing boundaries under WP_PAGED_BATCH (H3/H4)
e5cb0582e feat(wp): call mark_routing_boundaries before sched compute
cb15e252a feat(wp): WP_PAGED_BATCH flag + batch_safe pinnability-governed under it
f64ccfdc8 feat(wp): routing-boundary pre-pass (mark_routing_boundaries/is_routing_break)
```
Design spec `docs/superpowers/specs/2026-07-07-moe-aware-batched-paging-design.md`; plan
`docs/superpowers/plans/2026-07-07-moe-aware-batched-paging.md`.

### The `WP_PAGED_BATCH` feature — architecture (5 units)
Enable with `WP_PAGED_BATCH=1` (default OFF; requires `WP_BATCH_EVAL_CB=1` + `WP_SIZE_CLASS_SLOTS=1`).
- **A. Routing-boundary pre-pass** — `WeightPager::mark_routing_boundaries(gf)` (wp-pager.cpp), called
  from `llama_context::graph_compute` before sched compute (llama-context.cpp ~2492). Marks every
  `MUL_MAT_ID` node + the view-root of its `src[2]` (ids-producer) into `routing_break_tensors_`.
  `is_routing_break(t)` is an O(1) lookup. Cached via a topology signature.
- **B. Break at routing boundaries** — `eval_cb_op_return()` (wp-eval-cb.cpp ~249) returns `true`
  (end range) when `is_routing_break(t)`. This isolates each `MUL_MAT_ID`, so the router+ids-producer
  compute+sync first (ids valid = fixes **H3**), and no other op shares its range to `take()`-steal
  the routed-expert TLS (fixes **H4**; note `ggml_cuda_mul_mat_q` is called for regular mul_mat too,
  ggml-cuda.cu:2889).
- **C. Per-range pin lifecycle (5a)** — under paged_batch, pins accumulate in `s_range_pins`, move to
  `s_range_pins_pending` when a range ends, and release at the **top of the next callback** (post-sync,
  guaranteed). Replaces the per-op MAD-231 unpin. `reset()` flushes at teardown.
- **D. Reactive auto-break (5b)** — track `s_range_pinned_bytes` at pin sites; `eval_cb_op_return()`
  ends the range when it crosses **70% of `pool_arena_bytes()`**. Bounds a range's pinned working set
  so a dense stretch (no routing boundary) can't overflow pins under eviction (`alloc_slot -1`). This
  is what lets `batch_safe()` drop the `evictions==0` requirement — batching under real eviction.
- **E. Flag** — `wp_paged_batch_enabled()` (wp-eval-cb.cpp); `batch_safe()` returns
  `pool_.size_class_slots_enabled()` when the flag is on (drops `evictions==0 && !has_experts`).

**Validation matrix (all committed, all GPU-validated, PPL exact + 0 faults):**
| | resident (evictions==0) | paged (evictions>0) |
|---|---|---|
| **dense (Qwen3.6-27B)** | PPL 5.4623 @ slots=345 | PPL 5.4623 @ slots=200, **2174 evict** |
| **MoE (LFM2.5-8B-A1B)** | PPL 27.0938 @ slots=750 | PPL 27.0938 @ slots=300, **7784 evict** |
| **MoE @ scale (DeepSeek V4)** | — | **PPL 4.1524 @ slots=384, 62233 evict** |

Unit tests: `test_routing_boundary_prepass`, `test_wp_paged_batch_flag_default_off` in
tests/test-weight-pager.cpp (builds/runs on mad-lab-2026 `build-army`, or mad-lab-main `build-hip`).

### The 3 DeepSeek-V4 × weight-paging load fixes (root causes)
1. **Fit-dry-run crash** (`978a720bf`, common/fit.cpp:~60): `common_get_device_memory_data`'s
   `no_alloc` dry-run cleared mmap/mlock/direct_io but NOT `weight_paging_enabled`, so paged tensors got
   null buffers and `ggml_gallocr` asserted `buffer_id>=0` during the dry-run reserve. **Fix:**
   `mparams_copy.weight_paging_enabled = false` in the dry-run. (Any paged model was un-loadable via the
   auto-fit path before this.)
2. **Tiny weights paged** (`3faad72a0`, src/llama-model.cpp): DeepSeek V4's tiny per-layer weights
   (hyper-connection `hc_attn_scale` {3}, `hc_*_base` {hc_mix_dim}, sinks, expert-prob bias) were paged;
   a paged non-matmul leaf gets no backend → assert. **Fix:** `WP_MIN_PAGED_BYTES = 4096` floor
   (file-scope const) in both `is_paged_weight` and the `weight_page_infos` registration loop. Standard
   {n_embd} norms stay paged (27B/LFM unchanged).
3. **View-only-leaf buffer_id** (`d384d93f4`, src/llama.cpp:287) — **the systemic one.** A paged weight
   consumed **only via a reshape view** (DeepSeek `wo_a = ggml_reshape_3d(...)->mul_mat`, deepseek4.cpp:1068)
   inherits `buffer_id -1` from `ggml_backend_sched_split_graph` **pass 4** (ggml-backend.cpp:1217-1241):
   the view's id comes from its `view_src` (paged leaf, `buffer==NULL` → -1), then the src-loop assigns
   the leaf = the view's still-`-1` id **before** the view's own fallback runs. Directly-consumed weights
   are rescued at line 1233 by their pass-3-assigned matmul; view-consumed ones aren't. **Fix (1 line):**
   set paged weights' `t->buffer = pool_buf` (instead of `nullptr`) at init, giving the scheduler a
   backend to name. Known-safe: the eval_cb **already** sets `src->buffer = pool_buf` on every patch
   (wp-eval-cb.cpp:1085/1094), so this just brings load-time state to the post-first-decode steady state;
   gallocr still skips the leaf (`is_allocated` checks `data!=NULL` first). **Eval-time correctness of
   reshaped/viewed paged weights is already handled** — the eval_cb resolves the page via `view_src`
   (wp-eval-cb.cpp:919-921) and patches the VIEW's data `= vram + view_offs` (1089-1097, the "B-P1"
   path). Root-caused by codex + Fable consults converging on the pass-4 ordering.

### DeepSeek V4 Flash — run recipe + geometry
- **Model:** `~/Downloads/DeepSeek-V4-Flash-UD-Q8_K_XL-0000{1..5}-of-00005.gguf` (162 GB; load via the
  `-00001-of-00005` file, llama.cpp auto-pulls the rest). Part 1 is a 5 MB **metadata-only shard**
  (that is normal, not truncated). Arch = `deepseek4` (upstream #24162, has the lightning-indexer / NSA
  sparse attention). Converter is absent from this tree — use a pre-quantized GGUF.
- **Geometry:** 33,987 paged pages (after the tiny-weights fix; was 34,349), **max_page = 64 MiB**.
  slots=384 → 24 GB arena. Total VRAM ~28.6 GB (arena + resident embeds/output + KV/compute). Fits 32 GB.
- **PPL command (works):**
  ```
  ssh mad-lab-main bash -s <<'E'
  cd /home/kmbandy/GitHub/llama.cpp
  systemd-run --user --unit=dsv4 --collect -p MemoryMax=30000M --working-directory=/home/kmbandy/GitHub/llama.cpp \
    --setenv=WP_SIZE_CLASS_SLOTS=1 --setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1 --setenv=WP_DENSE_PREFETCH_N=8 \
    bash -c "./build-hip/bin/llama-perplexity -m ~/Downloads/DeepSeek-V4-Flash-UD-Q8_K_XL-00001-of-00005.gguf \
      -f wikitext-2-raw/wiki.test.raw --chunks 2 -c 512 --no-mmap --weight-paging --weight-paging-slots 384 \
      --weight-paging-prefetch -ngl 99 --device ROCm0 > /tmp/dsv4.log 2>&1"
  E
  ```
  Expect PPL 4.1524, evictions ~62k, no fault. **~6.4 min / 2 chunks (I/O-bound).**

### ★ NEXT FRONT — decode-speed I/O parallelism (start here) ★
**The measured problem:** DeepSeek decode ≈ **0.04 t/s**; per token ~11 GB streams from NVMe at
**~0.5 GB/s effective ≈ 1/10th of the SN850's ~5-7 GB/s**. Not bandwidth-bound, not compute-bound, not
sync-bound (batching is irrelevant at this residency). It is **QD≈1 serialized**.

**What was tried and did NOT help (so don't rehash):**
- P2P dma_buf transport (`LLAMA_WP_TRANSPORT=p2p LLAMA_WP_TRANSPORT_FORCE=1`) — confirmed active
  ("P2P enabled — pool dma_buf exported+mmap'd", `page_in_sync_ EXIT p2p`); **no change** (still 0.04).
- `WP_PREFETCH_DEPTH=16 WP_IOURING_DEPTH=16` — **no change** (nothing to parallelize if reads are serial).
- `WP_ASYNC_ENSURE=1` — **worse** (0.02 t/s); adds overhead on the same serial path (and its compose
  with the 5a/5b per-range pins is unvalidated — see Task 6).

**Diagnosis:** the expert page-in goes through `page_in_sync_` **one page at a time, waiting on each**,
so the deep queue is never filled. For a MoE op the N active experts are all known at once (post-routing)
— they should be issued **concurrently** (fill io_uring SQ to depth N) then waited together, turning
QD≈1 into QD≈N and approaching NVMe bandwidth (~10× headroom → ~0.4 t/s ceiling at 15% residency).

**Where to work (code):**
- eval_cb **Step 2 ensure loop** (wp-eval-cb.cpp ~1051): `for (j...) pager->ensure(page_idx)` — serial,
  each `ensure()` blocks. This is the serialization to break.
- The **"Pass 1: fire async prefetch for every active expert"** comment (~wp-eval-cb.cpp:531 in the MoE
  routing branch) — intended concurrency; verify whether it actually issues parallel reads or is inert.
- `WeightPager::ensure` / `page_in_sync_` (wp-pager.cpp) and the io_uring file-io layer (wp-file-io) —
  the sync path vs a batched-submit path.
**Measurement to do first:** `iostat -x /dev/nvme0n1 3` during a decode (was cut off last run — redo),
plus the pager's `io_effective_gb_s` counter, to confirm QD/throughput and quantify the gap before
coding. Likely a consult candidate (subtle async/io_uring path).

### Other open items (lower priority than the I/O front)
- **Task 6 (async compose):** `WP_ASYNC_ENSURE=1` + `WP_PAGED_BATCH` per-range pins is **unvalidated**
  (5a/5b were validated async-OFF). The async path operates on `s_pinned_pages_prev_op` which paged_batch
  bypasses (pins go to `s_range_pins`), so it may already be inert-but-safe — needs a correctness check.
- **Flip `WP_PAGED_BATCH` default-on** — only after Task 6 + a decision; it's self-gated (size-class +
  batch_eval_cb) so low-risk, but hold until the I/O work settles.
- **gpt-oss-20b** produces NaN under paging (pre-existing, gpt-oss/MXFP4-specific, `discarded:6`) — NOT a
  blocker; DeepSeek/LFM route cleanly (`discarded:0`).

### State at compact
- All 9 commits pushed; `feat/wp-vnext` tip `d384d93f4` == origin. Working tree: only pre-existing WIP
  (`examples/pagedattn-*`, `src/llama-graph.cpp`, `examples/CMakeLists.txt`) — do NOT touch.
- mad-lab-main: wp-vnext commits cherry-picked; `src/llama.cpp` + `src/llama-model.cpp` present as
  staged-matching-origin (applied via `git checkout origin/feat/wp-vnext -- <file>`); the throwaway
  `ggml-alloc.c` WP-DIAG diagnostic was **reverted** (clean). GPU **clear** (VRAM ~1.3 GB baseline).
- Build targets on mad-lab-main: `cmake --build build-hip -j2 --target llama llama-perplexity llama-server`
  in the capped `systemd-run --user` unit (MemoryMax=13000M CPUQuota=600%).

---
---

## TL;DR of the session

Took **paged-resident dense-27B decode from 7.0 → 21.0 t/s = native speed**, with **PPL exactly
matching native** (5.4623). Three gated, default-off, committed+pushed commits. Then found that
the last/biggest fix (`WP_BATCH_EVAL_CB`) **faults on MoE** — that's the open item for the morning.

## UPDATE — morning session 2026-07-07 (Step 1 DONE; Step 2 scoped)

**Step 1 complete — dense win locked in and shipped default-on:**
- `2a221b9ef` **dense-only guard**: `batch_safe()` now also requires `!catalog_.has_experts()`. On MoE
  → per-op path (no batching), so no fault.
- `aa1130789` **flip `WP_BATCH_EVAL_CB` default ON** (off only when `WP_BATCH_EVAL_CB=0`). Self-gated:
  only engages under `--weight-paging` + size-class + fully-resident + dense.
- Both pushed to `origin/feat/wp-vnext`; cherry-picked + built + validated on mad-lab-main
  (`5919648e2`, `656ae5a14`).
- **Validated:** dense 27B PPL = **5.4623** with explicit `=1` AND with the flag **unset** (default takes
  effect). MoE guard proof: guarded `=1` is **byte-identical** to `=0` on gpt-oss (same NaN, same stats,
  no fault) — guard makes `=1` a no-op on MoE.

**Step 2 evidence gathered (LFM2.5-8B-A1B, a small 8B/1B-active MoE, valid finite PPL — a far better
test case than gpt-oss):**
- LFM native (no paging) PPL = **27.2266**.
- LFM paged **per-op** (BATCH=0, slots=750, fully resident) = **27.0938** — matches native within
  reduction-order noise (same 0.13-ish gap dense shows: 5.4543 vs 5.4623). **Per-op MoE paging is CORRECT.**
- `routing_ptrs_discarded_unconsumed: 6` appears on BOTH LFM (correct output) and gpt-oss (NaN) →
  **that counter is a benign accounting artifact, NOT the corruption source.** The gpt-oss NaN is
  gpt-oss/MXFP4-specific, off the critical path.
- **Batching fault is GENERAL:** temporarily removed the guard, rebuilt, ran LFM with batching →
  `Memory access fault … address 0xdc000 … Page not present` (same near-null class as gpt-oss's
  0x11f000). So the batching-on-MoE fault reproduces on a clean valid-PPL MoE. GPU recovered cleanly.
  (Experiment reverted; mad-lab-main back to shipped guarded state.)

**Net for Step 2:** the ONLY MoE bug is the batching range/pin lifetime. Per-op is a correct baseline.
We now have a **clean repro + a valid PPL target (27.09)** to validate the redesign against.
**Do the redesign on LFM2.5-8B-A1B** (fast, fully resident at slots=750, valid PPL), not gpt-oss.

## The result (dense Qwen3.6-27B-Q6_K, fully resident, R9700)

| config | decode t/s | PPL (wiki, -c512 --chunks 4) |
|---|---|---|
| native (no paging) | ~native | **5.4623** |
| paged, all flags OFF (per-op sync) | 7.0 | 5.4543 |
| paged + size-class + resident-fadvise | 9.4 | — |
| **paged + WP_BATCH_EVAL_CB=1** | **20.97** | **5.4623** ✅ matches native |

**The 5.4623 vs old 5.4543:** confirmed it's not a regression — native (no pager) is *also* 5.4623.
The old per-op-sync 5.4543 was the anomaly (serialized execution → different GPU reduction order).
Batching makes the paged path converge to native in both speed AND numerics. Determinism proven:
batch-off → 5.4543 (x2), batch-on → 5.4623 (x2), fully reversible.

## Commit stack this session (on `feat/wp-vnext`, all pushed)

- `390a7e18d` feat(wp): **size-class VRAM slots** `WP_SIZE_CLASS_SLOTS` (default off) — packs the
  whole model resident in a size-class arena (vs fixed max-size slots). The enabler: on 32 GB the
  27B (22.8 GB) can't be resident with fixed slots (each = 70 MB), but size-class packs all 863
  pages. Known risk documented: no coalescing → a large *required* page can `alloc_slot → -1`
  (never triggered on the 27B at slots=345).
- `ec48c2316` perf(wp): env-tunable `WP_PREFETCH_DEPTH` / `WP_IOURING_DEPTH` (from before this session).
- `5ee99edcd` diag(wp): **`WP_PROFILE_EVAL`** — RAII host-time profiler inside `weight_pager_eval_cb`.
- `74a5c5aa2` perf(wp): `find_page(const char*)` alloc-free overload + gate the LOW_ADDR diagnostic
  loop. (Measured ~0 effect — the string alloc was NOT the cost; kept as cheap hygiene.)
- `c1bb508e1` diag(wp): split `WP_PROFILE_EVAL` into pre-Step1 / step1-resolve / ensure buckets.
- `b7a33e849` perf(wp): **resident-aware fadvise** — `advise_layer_lookahead` skips the
  `posix_fadvise(WILLNEED)` NVMe readahead when the lookahead window is already resident
  (`page_loaded_`). Was ~35% of decode when resident (readahead warms nothing). **+34% (7.0→9.4).**
- `78358b158` perf(wp): **`WP_BATCH_EVAL_CB`** (default off) — the 3× (9.4→21). See below.

## The big one: WP_BATCH_EVAL_CB (root cause + fix)

**Root cause (verified in `ggml/src/ggml-backend.cpp:1700-1729`, confirmed by two independent
consults — codex + Fable):** when an eval callback is registered, the scheduler abandons the
whole-split async path (line 1701) and runs **per-node**, issuing a full
`ggml_backend_synchronize(split_backend)` after each node-*range* (line 1729). The range only
extends **while the callback returns `false`** (line 1716). `weight_pager_eval_cb` returned `true`
at every op → every one of ~3700 nodes/token became a singleton compute + GPU sync → decode fully
serialized (submit→sync→submit→sync). Native (no callback) submits the whole split in one async
compute. **This per-op sync was the entire ~2-3× gap** — not `ensure()`, not I/O, not the string
work (all measured near-zero when resident). The profiler saw it as ~100 µs/weight-op of "host time"
(the CPU blocking in `ggml_backend_synchronize`, off-book from the callback's own timer).

**The fix:** the return value doesn't gate patching (the scheduler calls `ask=true` per node during
range-building, so `src->data` patching still happens before compute); it *only* controls whether
the scheduler syncs after the node. The pager does nothing on the `ask=false` post-callback, so no
node needs observing. So return `false` when safe → the scheduler batches → sync per-split like
native. Implemented as `eval_cb_op_return()` at the two op-level exits, returning `false` only when:
- `WP_BATCH_EVAL_CB=1` (default 0 → returns `true` → byte-identical old behavior), AND
- `pager->batch_safe()` = `stats_.evictions == 0 && pool_.size_class_slots_enabled()` (no eviction
  ⇒ no slot reuse ⇒ unpinning the prev op's slots before the batched compute can't recycle an
  in-flight slot — MAD-231 pin-lifecycle safe), AND
- `!routing_tls_set` (set inside the `GGML_OP_MUL_MAT_ID` routing branch when it calls
  `ggml_cuda_set_routed_expert_ptrs`), AND
- `sync_fallback_count()` didn't increase this op.
The MAD-230 `ggml_cuda_discard_routed_expert_ptrs()` still runs at the top of every ask=true call.

## OPEN / NEXT MORNING — WP_BATCH_EVAL_CB faults on MoE
> **SUPERSEDED by the 2026-07-07 UPDATE above.** Step 1 (guard + default-on) is DONE and shipped.
> The MoE fault is now confirmed GENERAL (reproduces on clean LFM2.5-8B-A1B), and per-op MoE paging
> is confirmed CORRECT (valid baseline 27.09). What remains below is the still-accurate root-cause
> analysis and the redesign spec (path 2). Ignore the "decide first thing" framing — path 1 is done.

**Validated on gpt-oss-20b-MXFP4 (11.3 GB, resident, slots=1024 → 12.24 GB arena, ~15 GB total VRAM):**
- native PPL = **427** (garbage — gpt-oss is NOT a raw-text LM; wikitext PPL is meaningless for it,
  so it's a poor PPL test model. Pick a real MoE with valid base-LM PPL next.)
- batch0 (no batching): **completes**, but `routing_ptrs_discarded_unconsumed: 6` — a pre-existing
  MAD-230-class routing leak on gpt-oss *independent of batching*.
- **batch1 (batching on): HARD GPU FAULT** — `Memory access fault … addr 0x11f000 … kernel
  mul_mat_q … void const* const*` (the routed-expert pointer array), preceded by `[1]nan,[2]nan`.
  Classic MAD-230 near-null expert-pointer fault; batching triggered it. GPU recovered cleanly
  (no wedge, device enum OK, VRAM back to baseline — a contained fault).

**Why the routing guard was insufficient (root cause of the MoE fault):** the scheduler's
`while(!need)` loop takes the *first* node that returns `true` as the **last** node of the batched
range (`j1` at line 1716-1721), NOT a standalone. So a routing op returning `true` still computes
**inside** a batched range together with the preceding non-routing ops — the expert-pointer TLS /
pinned-expert-slot lifetime is NOT isolated. `routing_tls_set → return true` does not break the
range *before* the routing op.

**Two paths for the morning (decide first thing):**

1. **Lock the dense win safely (small, do first regardless):** add `!catalog_.has_experts()` to the
   batch gate (in `batch_safe()` or alongside it) so `WP_BATCH_EVAL_CB` only ever engages for
   **dense** models. Then it's safe to flip **default-on** for dense (see below) with zero MoE risk;
   MoE always keeps the working per-op path. `catalog_.has_experts()` already exists
   (`wp-page-catalog.h`). This is the minimum needed before any default-on.

2. **MoE batching redesign (the real MoE prize — its own focused effort, likely codex+Fable):** make
   a routing op **break the range *before* it** (so it never computes inside a batched range), and
   keep the active-expert slots pinned across the batched range. This is a genuine correctness
   redesign of the routing/batching interaction, not a quick patch. gpt-oss is a fragile test case
   (pre-existing leak, junk PPL) — validate the redesign on a **real MoE with valid PPL** (candidate:
   `ornith-1.0-35b-Q5_K_M.gguf` = qwen35moe, 23 GB — but that's tight on 32 GB resident: 23 GB
   arena + overhead ≈ 26-28 GB, right at the ceiling, do the VRAM math carefully; or find a smaller
   valid-PPL MoE). Confirm PPL matches native AND `routing_ptrs_discarded_unconsumed == 0` AND no
   fault.

## Default-on decision (pending step 1)

`WP_BATCH_EVAL_CB` is inherently scoped: the eval callback it controls is **only registered when
`--weight-paging` is active** (non-paged inference never hits this code), and it self-gates on
`batch_safe()` (only fires when fully resident + size-class, reverts to per-op the instant anything
evicts). After step 1 (dense-only guard), flipping the default to ON (i.e. on unless
`WP_BATCH_EVAL_CB=0`) is safe and gives the dense win with no flag. Hold the flip until the
dense-only guard is in.

## Validation recipes (copy-paste)

Common env: `LLAMA_WP_TRANSPORT=p2p LLAMA_WP_TRANSPORT_FORCE=1 WP_DENSE_PREFETCH_N=8 WP_SIZE_CLASS_SLOTS=1`
Common args: `--no-mmap --weight-paging --weight-paging-slots <N> --weight-paging-prefetch -ngl 99 --device ROCm0 -c 512`

- **Dense 27B** (`~/models/Qwen3.6-27B-Q6_K.gguf`, `wikitext-2-raw/wiki.test.raw`): slots=**345**
  (~24 GB arena). PPL gate = **5.4623** (== native) with `WP_BATCH_EVAL_CB=1`. Decode via
  llama-server `/completion` `predicted_per_second`.
- **VRAM safety:** size-class arena = `n_slots × max_page_size` allocated up front — read the pager
  init line first (`wp::WeightPager: N pages, S slots x B budget …`) on a tiny slots=64 load to get
  `max_page_size`, then size slots for arena ≥ model, total ≤ 28 GB. gpt-oss max_page = 11.95 MB /
  2760 pages; 27B max_page = 70 MB / 863 pages.
- **Tools:** llama-cli is UNUSABLE (parks interactive, ignores -no-cnv); llama-bench has no
  `--weight-paging`; use **llama-perplexity** (PPL) + **llama-server** (`/completion` decode t/s).
- **llama-server build:** must be rebuilt when `common_params` changes — a stale `libllama-server-impl.so`
  vs fresh `libllama-common` = ABI-skew segfault in the arg parser (hit + fixed this session). Always
  include `llama-server` in the build target set.

## Build commands (mad-lab-main, capped)

```
systemd-run --user --unit=wp-build -p MemoryMax=13000M -p MemoryHigh=11000M -p CPUQuota=600% \
  --working-directory=/home/kmbandy/GitHub/llama.cpp \
  bash -c "cmake --build build-hip -j2 --target llama llama-server llama-perplexity > /tmp/x.log 2>&1"
```
build-hip = multi-arch gfx1201;gfx1030, GGML_HIP_AITER=ON (Triton AOT — reuse stamps, don't rm).

## Env flags added this session (all default-off / behavior-preserving)

`WP_SIZE_CLASS_SLOTS`, `WP_PREFETCH_DEPTH`, `WP_IOURING_DEPTH`, `WP_PROFILE_EVAL` (diag),
`WP_BATCH_EVAL_CB`. Existing knobs still relevant: `WP_DENSE_PREFETCH_N`, `WP_FADVISE_LOOKAHEAD`
(now resident-aware), `LLAMA_WP_TRANSPORT=p2p`, `LLAMA_WP_TRANSPORT_FORCE`, `WP_ASYNC_ENSURE`.

## Cleanup state at logoff

No GPU processes running; R9700 VRAM at ~1.4 GB (desktop baseline). All work committed + pushed;
`feat/wp-vnext` == `origin` == mad-lab-main tip `78358b158`. Only uncommitted = pre-existing WIP
(`examples/pagedattn-*`, `src/llama-graph.cpp`, `examples/CMakeLists.txt`) — do NOT touch.
