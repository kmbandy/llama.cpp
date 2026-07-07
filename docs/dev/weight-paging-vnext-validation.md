# Weight Paging v_next — Branch `feat/wp-vnext` validation checklist

Six stacked commits (MAD-P0..P5) adding instrumentation + perf scaffolding to the
weight pager. Built on top of `feat/router-multigpu` so it fast-forwards cleanly onto
master once the router work lands there.

**Everything except P0 is behind a default-OFF flag, so the branch changes NOTHING about
default paging behavior until a flag is set.** Local verification on the dev box was
CUDA build + HIP `-fsyntax-only` (no gfx1201 GPU / no liburing here) + the
`test-weight-pager` unit suite. The GPU decode, the real io_uring/dma_buf compile, and all
flag-ON correctness/perf work happen here on mad-lab-main (R9700 gfx1201 / 6900XT gfx1030,
ROCm 6.4+, liburing present).

## Build on mad-lab-main

Multi-arch, always both (per standing rule):
```
cmake -B build-hip -DGGML_HIP=ON -DAMDGPU_TARGETS="gfx1201;gfx1030" -DLLAMA_BUILD_TESTS=ON ...
cmake --build build-hip -j...   # first real compile of the io_uring + dma_buf paths
./build-hip/bin/test-weight-pager   # expect 0 failures
```
This is the first time the P3 dma_buf/io_uring real path and the P2 async HIP path are
actually compiled with liburing — treat compile errors there as the first thing to fix.

## The commits

| Commit | Story | Flag (default) | What to validate |
|---|---|---|---|
| P0 | instrumentation + `/metrics` + teardown summary | none (always on) | **Do this first.** Run a paged decode and read the summary: prefetch hit rate, effective GB/s, sync_fallbacks, cross-layer hit rate, routing counters. This is the baseline every other story is judged against. Cross-check GB/s vs `iostat`. |
| P1 | HIP graphs under paging | `WP_HIP_GRAPHS=1` | Partial scaffolding (coarse recapture + `cudaGraphExecUpdate`, not a per-node arg patcher). Prove PPL identical + token-identical + 0 GPU faults vs flag-off. Over-pinning is now bounded (`WP_GRAPH_PIN_MAX`, keeps a floor of evictable slots, degrades to non-graph for overflow — see fix commit `3ca8f71f8`); full lifetime-correct pinning still needs the arg-patcher. |
| P2 | async ensure / compute-transfer overlap | `WP_ASYNC_ENSURE=1` | The shutdown UAF is FIXED (`weight_pager_eval_cb_reset` drained from `shutdown()`, commit `3ca8f71f8`). Validate token-identical + sync_fallback drop + decode t/s. |
| P3 | dma_buf NVMe→VRAM P2P | `LLAMA_WP_TRANSPORT=p2p` | Run `wp-dmabuf-probe` first to confirm the box supports `hsa_amd_portable_export_dmabuf` (records ROCm version; validated on 7.2.2). Then prove bit-exact output vs the host-staged default, and measure GB/s delta. Fallback ladder (p2p→io_uring-host→sync-pread) should log the active tier. |
| P4 | host-tier pinned slab cache | `WP_HOST_BUDGET_BYTES=<bytes>` (+`WP_PIN_HOST=1`) | **Conditional — only enable if P0's baseline shows a high cold-miss rate that P3 P2P doesn't already solve.** Slab/LRU logic is unit-tested; validate the RAM→VRAM hit path lowers cold-miss + raises t/s on a working-set-exceeds-VRAM model. `WP_PIN_HOST` must gracefully fall back on memlock-ulimit and skip on UMA. |
| P5 | routing hardening | `WP_ROUTING_GUARD=1` | Counters are always-on: confirm `routing_ptrs_discarded_unconsumed` reads **0** across a clean decode (nonzero = MAD-230 leak-class recurrence). Run one decode with `WP_ROUTING_GUARD=1` and confirm no abort (invariant holds). |

## Post-P0..P5 speed levers (Fable gates #4 / #5)

Two additional default-off commits landed after the P0..P5 stack, targeting decode
throughput once correctness is established:

| Commit | Lever | Flag(s) (default) | What to validate |
|---|---|---|---|
| gate #4 | env-tunable prefetch + io_uring queue depth | `WP_PREFETCH_DEPTH=<N>` (4), `WP_IOURING_DEPTH=<N>` (derives from prefetch_depth, clamped ≥ prefetch_depth) | Defaults byte-identical. Sweep `WP_PREFETCH_DEPTH` (e.g. 4/8/16/32) with `--weight-paging-prefetch` + `WP_DENSE_PREFETCH_N` and watch `io_effective_gb_s` climb off the QD≈1 ceiling (~1.6 GB/s measured). SQ depth is coupled to prefetch_depth by default; `WP_IOURING_DEPTH` raises it independently. |
| gate #5 | size-class VRAM slots | `WP_SIZE_CLASS_SLOTS=1` | **The structural residency unblock.** Fixed slots = max_page_size each, so the 27B holds only ~64 pages resident and paged decode is I/O-bound (re-pages every token). Size-class packs pages by actual size → far more resident → higher prefetch hit-rate, and shrinks the padding memset to near-zero. Prove PPL identical to fixed mode first, then measure resident-page count + prefetch hit-rate + decode t/s. **This is the enabler for MoE decode approaching active-params speed and for P1 graphs to matter (decode must be launch-bound, not I/O-bound).** |

**KNOWN RISK on `WP_SIZE_CLASS_SLOTS=1` — watch for this on hardware:** size-class
eviction can only reclaim a slot of class **≥** the requested page size (no coalescing of
several small unpinned slots into one large slot). A large **required** page can therefore
get `alloc_slot → -1` in `page_in_sync_` even when smaller slots are evictable — fixed mode
never fails this way. Symptom to watch: `wp::PoolAllocator::alloc_slot: no unpinned
size-class slot can fit …` WARN followed by a page-in failure / degraded output. If it bites,
fix direction is to reserve N max-size slots at the arena top, or add slot compaction. It is
gated off, so the branch default is unaffected.

## Decode-speed work (2026-07-06 session) — dense at native speed

Beyond the residency levers above, the session that took paged-resident **dense** decode to native:

| Commit | Lever | Flag (default) | Result |
|---|---|---|---|
| `b7a33e849` | resident-aware fadvise | `WP_FADVISE_LOOKAHEAD` now skips when resident | +34% (7.0→9.4 t/s); the readahead warms nothing when the model is in VRAM |
| `78358b158` | `WP_BATCH_EVAL_CB` — stop the callback de-batching the graph | `WP_BATCH_EVAL_CB=1` (default 0) | **9.4→21.0 t/s = native**, PPL 5.4623 == native |

`WP_PROFILE_EVAL=1` (diag, `5ee99edcd`/`c1bb508e1`) prints a host-time breakdown of `weight_pager_eval_cb`
at teardown — used to root-cause the above.

**Root cause of the ~2-3× paged-decode gap** (verified in `ggml/src/ggml-backend.cpp:1700-1729`):
a registered eval callback makes the scheduler compute node-by-node and `ggml_backend_synchronize`
after each; the range only batches while the callback returns `false`. The pager returned `true`
everywhere → ~3700 GPU syncs/token. `WP_BATCH_EVAL_CB=1` returns `false` when
`batch_safe()` (evictions==0 && size-class) && not a routing op && no sync-fallback → batches like
native. Patching still happens (ask=true per node before compute); only the sync is removed.

**OPEN — `WP_BATCH_EVAL_CB` faults on MoE.** On gpt-oss-20b it caused a near-null routed-expert GPU
fault (`mul_mat_q … void const* const*`), because the scheduler includes a `true`-returning routing
op as the *last* node of a batched range (not standalone), so the routing-TLS/expert-pin lifetime
isn't isolated. **Before any default-on:** add `!catalog_.has_experts()` to the gate (dense-only).
The proper MoE fix (routing op must break the range *before* it) is a separate effort. Full detail
+ morning pickup: `docs/dev/weight-paging-batch-eval-continuation.md`.

## Recommended validation order

1. **P0 baseline** — measure current paged decode (dense first: `Qwen3.5-4B-UD-IQ3_XXS`,
   the known-safe repro; PPL baseline 10.8649). Print the VRAM pre-flight before any
   20B+ paged run (two prior OOM/hang restarts — don't skip it).
2. **P5 counters** — cheap, confirms `discarded_unconsumed == 0` (routing health) before
   trusting anything else.
3. **P1** then **P2** (the P2 UAF is already fixed, commit `3ca8f71f8`) — the launch-overhead
   + overlap wins Fable ranked highest for decode t/s. Prove correctness (token-identical)
   before trusting perf.
4. **P3** — the bandwidth lever; probe → bit-exact → measure.
5. **P4** — only if P0 data says the cold-miss rate justifies it.

## Local verification already done (dev box, no RDNA GPU / no liburing)

- build-army (CUDA + Vulkan) `llama`/`llama-server`/`test-weight-pager`: clean.
- HIP `-fsyntax-only` (ROCm 6.2.4, non-io_uring paths) of every touched + new wp file: clean.
- `test-weight-pager`: 0 failures, incl. new MAD-231 pin, MAD-234 UMA, MAD-236 catalog,
  MAD-237 hot-set, and MAD-P4 HostTier slab/LRU tests.
- NOT compiled locally: the real dma_buf+io_uring path (P3) and the flag-on HIP async path
  (needs liburing + dma_buf-capable ROCm) — first compiled here.
