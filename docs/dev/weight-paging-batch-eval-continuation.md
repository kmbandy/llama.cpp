# Weight-paging decode-speed work — continuation (morning pickup)

**Branch (CURRENT):** `feat/router-multigpu` = `feat/wp-vnext` (37 commits) **merged with** the router
branch, + the `ensure_batch` change. Edit/build-check on **mad-lab-2026** at
`/home/kmbandy/GitHub/llama.cpp`. Pushed to `origin`. **mad-lab-main** GPU box (R9700 / ROCm0 / gfx1201
/ 32 GB; 6900XT = ROCm1) has this branch checked out in a **git worktree at `~/llama-wp`** (its main
`~/GitHub/llama.cpp` checkout is on `feat/dsws-phaseb-conversion` with uncommitted MAD-305 DSWS work —
**do NOT touch/checkout that**; always use the `~/llama-wp` worktree for weight-paging builds/runs).
Remote shell is **fish** → always `ssh mad-lab-main bash -s <<'EOF'`.

---
---

# ★★★ 2026-07-07 EVENING — READ THIS FIRST (supersedes the "NEXT FRONT" below) ★★★
## `ensure_batch` (Colibri QD=N page-in) is BUILT + CORRECT but a **2× DECODE REGRESSION**. Root cause found. Fix = **completion demux**. Do that next.

### What happened this session (chronological, so we don't rehash)
1. **Inspiration:** the GLM-5.2 CPU engine **Colibri** (`github.com/JustVugg/colibri`, file `c/glm.c`)
   does exactly our thesis on CPU — streams routed experts from NVMe. Its decode loop (`glm.c:1016`)
   is the pattern we want: **batch the misses, issue ALL their reads concurrently (`omp parallel`,
   each a coalesced single read), then wait** — QD1→QDN. User: apply that to our weight paging.
2. **Merged** `feat/wp-vnext` into `feat/router-multigpu` (0 conflicts; pre-existing WIP
   `examples/pagedattn-*`, `src/llama-graph.cpp`, `examples/CMakeLists.txt` stashed/popped — **do NOT
   touch those**). Merge commit on the branch; then implemented `ensure_batch`.
3. **Implemented `ensure_batch`** — commit **`e5089c241`** (pushed). Three edits:
   - `wp-pager.h/.cpp`: new `WeightPager::ensure_batch(pages, out_ptrs, out_pinned)` — reserve **and
     pin** every cold-miss slot up front (alloc_slot skips pinned, so no sibling read can evict it),
     then on the P2P/direct-to-device path submit all misses in **one `io_uring` batch (QD=N)** into
     the VRAM slots, wait for all, harvest; on read failure, sync-fallback into the **same pinned
     slot**. Returns `out_pinned` for the caller to unpin next callback.
   - `wp-pager.cpp`: `page_in_sync_(page, reuse_slot=-1)` refactor — reads into a caller-owned pinned
     slot without releasing it on error (the 5 error paths guard `if (owns_slot) release_slot`).
   - `wp-eval-cb.cpp` (~line 598): the MoE active-expert page-in is gated by **`WP_ENSURE_BATCH=1`**
     (default OFF; the existing prefetch+ensure path stays in the `else` for A/B + rollback). New
     branch calls `ensure_batch`, records the returned pins into `s_pinned_pages_prev_op`/`s_range_pins`.
4. **Built clean** on mad-lab-main worktree `~/llama-wp/build-hip` (gfx1201, **AITER OFF**, ROCWMMA-FA
   ON). `ensure_batch` compiles with zero errors (only pre-existing `nodiscard hipError_t` warnings).
5. **Measured on R9700, DeepSeek V4 Flash paged, P2P, `WP_ENSURE_BATCH=1`, ring depth 4:**
   **decode = 0.023 t/s (vs 0.04 baseline) — a 2× REGRESSION.** Output **fully coherent**
   (`"...the founding of the city of Rome in 753 BC..."`) → correctness OK, pin-lifecycle sound,
   **no expert corruption**. So the problem is PERF, not correctness.
   - (0.04 baseline IS the valid apples-to-apples number: `WP_ENSURE_BATCH=0` on THIS merged branch ==
     the old wp-vnext path. Do not re-excuse it as "different branch.")
6. **Tried depth 8/16 first (WP_IOURING_DEPTH=16 WP_PREFETCH_DEPTH=16) — LOAD HANGS.** Confirmed via
   `/proc/<load-thread>/wchan = io_cqring_wait` (waiting on an io_uring completion that never arrives).
   Fell back to the known-good depth-4 ring for the run above.

### ROOT CAUSE — one bug explains BOTH the regression and the depth-8/16 hang: **shared io_uring ring CROSS-DRAIN**
The sync path (`page_in_sync_`, sentinel `req_id=(uint64)-1`) / `ensure_batch`'s wait loop AND the
`PrefetchScheduler` all drain the **same** io_uring ring. The io layer's `wait_any()`
(`wp-file-io.cpp` IoUringAsyncFileIO, ~line 255-306) **unconditionally `io_uring_cqe_seen()`s** every
CQE it returns. Both the sync/ensure_batch waiters and `PrefetchScheduler::tick()` **DROP any completion
whose `req_id` isn't theirs** — but the CQE is already consumed. `PrefetchScheduler::process_io_`
(`wp-prefetch.cpp:302`) frees a queue slot only on **its own** completions (via `req_to_slot_`); a
completion cannibalized by the sync/ensure path **never frees that scheduler slot** → `free_slots_`
leaks empty → **speculative cross-layer prefetch stalls out**.
- **DECODE (WP_ENSURE_BATCH=1):** `ensure_batch`'s `while(reaped<n) wait_any(-1)` eats the prefetch
  scheduler's completions → prefetch pipeline dies → we lose the **compute↔I/O overlap** the old path
  relied on → every MoE op becomes a cold, un-overlapped read → **2× slower** (net LOSS; "more
  concurrency" but no overlap, and capped at depth 4 anyway).
- **LOAD at depth 16:** heavier prefetch traffic → `tick()` (`wait_any(0)` drain) eats
  `page_in_sync_`'s completion → `page_in_sync_` blocks forever in `io_cqring_wait`. At depth 4 the
  collision is rare enough to mostly work.

### THE FIX (do this next — it is the real unlock, NOT "just avoid it")
**Completion demux in the io layer** (`wp-file-io.cpp`, IoUringAsyncFileIO): drain the ring **once**
into a shared `std::unordered_map<uint64_t req_id, IoResult>` (`io_uring_cqe_seen` exactly once per
CQE). Every waiter — `page_in_sync_`, `ensure_batch`, `PrefetchScheduler` — first checks that map for
ITS `req_id`(s); only if absent does it pull a fresh CQE (moving any non-matching completions into the
map, never discarding). No completion is ever lost to another consumer. This single change is expected
to fix ALL THREE: (a) the 2× decode regression (prefetch pipeline survives `ensure_batch`), (b) the
depth-8/16 load hang (`page_in_sync_` never loses its CQE), (c) `ensure_batch` finally gets **true
QD=8** (raise `WP_IOURING_DEPTH`/`WP_PREFETCH_DEPTH` to ≥8 once the demux lands).
- **Watch:** the load/decode paths are single-threaded per op (`PrefetchScheduler` +
  `page_in_sync_`/`ensure_batch` run on the same inference thread during a forward pass), so the map
  needs no lock — **verify** that assumption before relying on it.
- Files: `wp-file-io.cpp` (the demux + a `wait_for(req_id)` that consults it), `wp-prefetch.cpp`
  (`tick()`/`process_io_` pull from the map), `wp-pager.cpp` (`page_in_sync_`, `ensure_batch` waits).

### VALIDATION GATES after the demux (this is the proof the cross-drain was the cause)
1. **Decode ≥ 0.04** with `WP_ENSURE_BATCH=1` at depth 4 (regression gone: prefetch pipeline restored).
2. **Depth-8 load does NOT hang** (`WP_IOURING_DEPTH=8 WP_PREFETCH_DEPTH=8`).
3. **Decode at QD=8** climbs toward the **~0.4 t/s** ceiling (the ~10× headroom the diagnosis predicts).
4. Output stays coherent; `routing_ptrs_discarded=0`; 0 faults.

### EXACT REBUILD + RUN RECIPE (mad-lab-main, `~/llama-wp` worktree)
```
# BUILD (fresh worktree build; ~25 min at -j2, RAM-capped for the 15 GB host; CPU-only, safe):
ssh mad-lab-main bash -s <<'E'
cd ~/llama-wp
ROCM_PATH=/opt/rocm LD_LIBRARY_PATH=/opt/rocm/lib HIP_PATH=/opt/rocm cmake -S . -B build-hip \
  -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1201 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_HIP_AITER=OFF -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build build-hip -j2 --target llama-perplexity llama-server
E
# RUN a decode measurement (server on :8081, R9700):
ssh mad-lab-main bash -s <<'E'
systemd-run --user --unit=dsv4-decode --collect -p MemoryMax=9000M --working-directory=/home/kmbandy/llama-wp \
  --setenv=WP_SIZE_CLASS_SLOTS=1 --setenv=WP_BATCH_EVAL_CB=1 --setenv=WP_PAGED_BATCH=1 \
  --setenv=WP_DENSE_PREFETCH_N=8 --setenv=WP_ENSURE_BATCH=1 \
  --setenv=LLAMA_WP_TRANSPORT=p2p --setenv=LLAMA_WP_TRANSPORT_FORCE=1 \
  bash -c './build-hip/bin/llama-server -m /home/kmbandy/Downloads/DeepSeek-V4-Flash-UD-Q8_K_XL-00001-of-00005.gguf \
    --no-mmap --weight-paging --weight-paging-slots 384 --weight-paging-prefetch \
    -ngl 99 --device ROCm0 --host 127.0.0.1 --port 8081 --ctx-size 4096 --parallel 1 > /tmp/dsv4-decode.log 2>&1'
# wait for /health==200 (paged load ~3-5 min), then:
curl -s http://127.0.0.1:8081/completion -H 'Content-Type: application/json' \
  -d '{"prompt":"The history of the Roman Empire begins with","n_predict":16,"temperature":0,"cache_prompt":false}' \
  | python3 -c "import sys,json;t=json.load(sys.stdin)['timings'];print('t/s',t['predicted_per_second'])"
E
```
- **DO NOT** set `WP_IOURING_DEPTH`/`WP_PREFETCH_DEPTH` > 4 until the demux lands (hangs the load).
- To get clean pager stats (`prefetch_hit_rate`, `sync_fallbacks`, `io_effective_gb_s`) the server MUST
  shut down GRACEFULLY — `systemctl stop` hard-kills before `log_stats_summary()` runs. Send SIGTERM
  and let it drain, or add a signal handler, if you need those numbers.
- R9700 shares the box with the murmur **captain (:8090, ROCm0)** and the **6900XT LFM2.5 (:8092,
  ROCm1)**. When no murmur is active the captain holds ~0 VRAM so DeepSeek fits the 32 GB fine; if a
  murmur is running, the captain will contend for ROCm0 — check `rocm-smi --showmeminfo vram` first.

### Related (separate KG handoff notes written this session)
- **KG `3b83ca79`** — this ensure_batch result + cross-drain root cause + demux fix (full detail).
- **KG `5e2b8089`** — the recurrent-state mover HIP-only stub bug + 6900xt LFM2.5 concurrent GPU fault
  (a DIFFERENT subsystem; another session is picking up murmur; not on the critical path here).

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

---

## ★★★ 2026-07-07 LATE — PHASE 1 RESIDENT-DENSE SPLIT VALIDATED (47× win) ★★★

**WP_RESIDENT_DENSE=1 shipped (commits d4344dfbc, 07874dba3, 0b2ae033b, 16d486dbd).**
Dense weights (n_experts==0: MLA attention, shared expert, embeddings, norms —
~13.7 GiB, ~9% of DeepSeek V4 Flash Q8) load VRAM-resident via the existing
manual per-tensor allocator; ONLY routed-expert (_exps) tensors page. Two
filters made to agree via `wp_is_routed_expert()`: `is_paged_weight()`
(llama-model.cpp:1608, the load-bearing one) + catalog push. Fail-loud guard
in init_weight_pager. Unit test catalog_is_expert_classification (38/38 green).

**MEASURED on R9700 (ROCm0), DeepSeek V4 Flash Q8, host transport, --weight-paging-slots 2000:**
- Baseline (gate OFF, all-paged): **0.02 t/s**, io_effective 0.101 GB/s, ~9121 page-ins, dense thrashing.
- Gate ON, minimal (no batching, depth 4): **0.63 t/s**, coherent, io 1.384 GB/s, sync_fallbacks≈page_ins (serial).
- Gate ON, batching (ensure_batch+paged_batch+prefetch, depth 4): **0.93 t/s** (~47×), coherent
  ("' the founding of the city of Rome in'"), **sync_fallbacks=0** (QD=N working), io 1.399 GB/s.
- Pool auto-sizes to VRAM after dense resident (Task 5 ordering confirmed). Omit --weight-paging-slots
  or it OOMs (dense 13.7 + 384×64MB pool > 32GB); slot size drops to ~4.45MB (max expert sub-page).

**KNOWN LIMITATION — depth>4 HANGS under resident-dense (next lever):**
host+depth-4 works; host+depth-8 AND host+depth-16 HANG (main thread stuck in
io_cqring_wait, GPU idle, no I/O progress). Reproduces on HOST transport, so it
is NOT P2P. depth-8 worked WITHOUT resident-dense earlier this session, so it is
an INTERACTION: the resident-dense page set (only experts, ~4.45MB slots, 2000
slots) stresses the prefetch/ensure_batch concurrency on the shared io_uring
ring differently than the all-paged 64MB-slot layout the demux fix was validated
against. Root-cause needed (ptrace_scope=1 blocks gdb without sudo). This is the
lever to raise QD past 4 → past 0.93 t/s toward 1-3.

**NOT YET TESTED:** P2P transport + resident-dense at depth 4 (may beat host's
1.4 GB/s via direct-to-VRAM). Phase 2 (multi-device TB3) + Phase 3 (expert cache)
still ahead.

**DEPTH-4 SWEEP (best stable config found):**
- host, slots 2000: 0.93 t/s
- host, slots 3000: 0.65 t/s (more slots did NOT help — 0% prefetch hit, low cross-token expert locality)
- **P2P, slots 2000: 1.038 t/s** ← BEST, coherent, ~52× over 0.02 baseline, INTO the 1-3 t/s target.
  (io_effective "186 GB/s" is the batched-timing artifact for direct-to-VRAM; not literal.)

**Bottleneck now:** page_ins ~11k, evictions ~9k, prefetch_hit_rate 0% — the expert cache barely
hits (routing locality low at this cache size). Levers to go higher: (1) fix the depth>4 hang → QD>4;
(2) Phase 2 (dense on 6900XT eGPU frees full 32GB R9700 for expert cache); (3) Phase 3 frequency-biased
retention + prefetch coverage. Slot-count alone is a weak lever. Phase 1 gate: PASSED (coherent, >>0.4).

---

## 2026-07-07 — QD>4 HANG FIXED (Codex, validated) + parallel Codex work

**QD>4 hang FIXED** (commit 945f9074a, merged): io_uring backends counted `pending_` when SQEs
were PREPARED but ignored `io_uring_submit()`'s return. At depth 8/16 under resident-dense churn the
CQ ring fills → `io_uring_submit` returns -EBUSY (SQEs NOT submitted) → a blocking wait_cqe waits for
a req_id never kernel-owned → hang. Fix: track prepared-but-unsubmitted req_ids (pending_submit_
deque), drain the CQ into the demux ready_ buffer on -EBUSY and retry; synthesize ErrorNoSubmit if
truly stuck so waiters never hang. Applied to host + P2P io_uring. Regression test
test_file_io_submit_batch_depth_one_targeted_waits. GPU-VALIDATED: depth-8 P2P resident-dense (was
hanging) now health@40s, decode 0.9872 t/s, coherent ("...Rome in 753 BC."), sync_fallbacks=0.

**KEY FINDING: QD is NOT the bottleneck.** depth-8 (0.99) ≈ depth-4 (1.04). page_ins 11348,
evictions 9359, prefetch_hit_rate 0% — the expert cache thrashes (2000 slots ≪ working set, low
cross-token routing locality). The path to 1-3 t/s is Phase 2 (dense on 6900XT eGPU → full 32GB
R9700 for expert cache) + Phase 3 (frequency-biased retention + prefetch coverage), NOT queue depth.

**Phase 2 committed on feat/wp-phase2 (Codex, d8196031a) — NOT yet GPU-validated.**
--weight-paging-resident-device flag, dense→resident/expert→paging tensor_buft_overrides, per-device
pager pools + relaxed >1-device guard, two-pool unit test. 38/38 unit tests. Needs multi-device
forward validation on the R9700+6900XT(TB3) rig (cross-device activation copies are code-trace-only).
