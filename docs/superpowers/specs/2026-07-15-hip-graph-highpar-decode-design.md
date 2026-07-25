# HIP Graph Capture for High-Concurrency Decode — Design Spec

**Date:** 2026-07-15
**Status:** Draft (pre-implementation; written for the post-compact dive)
**Epic:** MAD-160 (synth-fleet aggregate throughput) / MAD-348 (paged decode perf)
**Author:** claude__main + kmbandy

---

## 1. Problem

Aggregate decode throughput on the fleet goes **flat with concurrency**. On the
6900XT (gfx1030), features-on paged decode caps at ~300–380 tok/s across
par 24→160, and even feature-off (plain KV) tops out ~596 and stops scaling —
instead of climbing with slot count as it should for a memory-bandwidth-bound
decode with more independent streams to amortize weight loads.

This defeats the platform thesis: **one agent, one config, features always on,
and it just scales.** Users pick a model + make an agent; Mneme handles the rest.
Nobody should tune llama.cpp, and nobody should run two servers for short vs long
context. So "features-on must scale with concurrency like features-off" is a
hard product requirement, not a nice-to-have.

## 2. Root cause (MEASURED, not theorized)

Method: gdb stack-sampling the server's inference thread during par128 decode
(after enabling `ptrace_scope=0`), plus rocprofv3 kernel traces and code read.

- The main inference thread spends **~83% of samples in
  `ggml_backend_cuda_synchronize`** — waiting on the GPU. **Zero** paged-specific
  host frames were ever sampled (no `prepare_batch_tensors`, scatter, block-table,
  tier). So the CPU is not the bottleneck; it waits on a slow, gappy GPU graph.
- Decode-phase GPU occupancy is **62%** (38% idle) — the idle is **gaps between
  kernels**, i.e. **kernel-launch latency**, not compute. A single MoE decode step
  is ~1500 tiny dispatches (per-expert matmuls + norms + attention + paged extras).
- **HIP/CUDA graph capture is OFF for this workload**, which is exactly what turns
  those ~1500 launches from one batched graph replay into 1500 individual host
  launches with latency between each → the 38% idle → flat aggregate.

Gates on graph capture (`ggml/src/ggml-cuda/ggml-cuda.cu`), re-verified 2026-07-15:

1. **Arch gate — NOT the problem on AMD.** `ggml_cuda_graph_set_enabled` (~5581)
   disables graphs if `cc < GGML_CUDA_CC_AMPERE`. That is the *NVIDIA* Ampere
   threshold (800); AMD devices carry a large `GGML_CUDA_CC_OFFSET_AMD` cc, so the
   check is false on gfx1030 → **graphs are NOT arch-disabled on the 6900XT.**
   (An earlier note "graphs effectively off on HIP" predates this / referred to a
   different tree.)

2. **THE core blocker — MoE `MUL_MAT_ID` batch>8 disable** (`ggml_cuda_graph_check_compability`,
   ~4261): graphs disabled for any `GGML_OP_MUL_MAT_ID` node whose token batch
   `ne[2] > get_mmvq_mmid_max_batch(type, cc)`. gfx1030 + Q8_0 → threshold
   `MMVQ_MAX_BATCH_SIZE = 8` (mmvq.cuh:3; Q8_0 hits the default arm of
   `get_mmvq_mmid_max_batch_rdna1_rdna2`). LFM2.5-8B-A1B is a MoE (a `MUL_MAT_ID`
   per FFN layer), so **any decode batch > 8 slots disables graphs.** Dispatch
   (`ggml_cuda_mul_mat_id`, ~3483): batch≤8 → mmvq (graph-safe), batch>8 → MMQ
   (`ggml_cuda_mul_mat_q`). Comment says the >8 path "needs to synchronize the
   stream" (upstream ggml PR #18958). **Open question:** does the MMQ path really
   sync, or is the blanket disable over-conservative (the sync note ~3552 describes
   the dequant/cuBLAS fall-through, not MMQ)?

3. **Paged capture-safety — the MAD-288 fix is NOT in this tree.** MAD-288 (fixed
   on mad-lab-2026 `~/GitHub/llama-gpu`, gpu-portability worktree — **not ported to
   mad-lab-main `~/GitHub/llama.cpp`**) found the paged flash-decode `partials`
   reduce-scratch is allocated per-call via `ggml_cuda_pool_alloc(ctx.pool(), …)`;
   that pooled pointer is not a graph-tracked tensor, so a captured graph records
   the address and REPLAY reads a stale/recycled one → silent token-salad. Verified
   this tree still uses the per-call pool alloc (`mt_pagedattn.cu`, in
   `launch_paged_attn_decode`). So **capturing the paged path here would corrupt on
   replay** until the MAD-288 persistent-scratch fix is ported. Fix design (from
   MAD-288): a static per-device `paged_decode_partials_scratch` grown only while
   the stream is NOT capturing (`cudaStreamIsCapturing` guard), reaching high-water
   during uncaptured warmup → stable address valid across capture+replay.

4. **`WP_HIP_GRAPHS` is NOT the on/off lever (corrected).** `ggml_cuda_wp_hip_graphs_enabled`
   (4235, used 5638) only gates the *reuse-on-property-change* subcase, and is unset
   on this box. MEASURED: `WP_HIP_GRAPHS=1` vs default gave identical par8 plain
   decode (265.6 vs 258.3 tok/s) — no effect. It is a red herring for the main
   capture path; do not scope around it.

**RESOLVED (Phase 0, MEASURED 2026-07-15).** Added env-gated diag `GGML_GRAPH_DIAG=1`
to `ggml-cuda.cu` (counts capture/replay/direct + the disabling clause). NB: ggml
`GGML_LOG_INFO` is *suppressed* in llama-server (only llama INFO + ggml WARN print),
so the diag emits at `GGML_LOG_WARN`. Cliff sweep on the KV-paged features-on config
(depth 500, prewarmed, separate instances per par):

| par | evals | capture | replay | direct | %direct | disabling ne2 |
|-----|-------|---------|--------|--------|---------|---------------|
| 1   | 192   | 4       | **178**| 10     | 5%      | 512 (prefill) |
| 8   | 576   | 28      | 5      | 543    | 94%     | 15            |
| 9   | 448   | 5       | 5      | 438    | 98%     | 9             |
| 128 | 2048  | 4       | 5      | 2039   | 99.6%   | 25            |

Findings:
- **Graphs capture+replay perfectly at true batch-1 decode (par1 = 93% replay), output coherent.** Graphs are NOT broken on gfx1030.
- The disabling clause is **always** `op=30` (`GGML_OP_MUL_MAT_ID`) with `ne[2] > mmvq_mmid_max(=8)` — i.e. gate #2 below, and *only* that gate. Never ml8, never split-buffer.
- **There is no par≤8 escape hatch.** Continuous batching aggregates prefill+decode tokens across slots, so the routed MUL_MAT_ID batch exceeds 8 nearly every step at any real concurrency (par8 already shows ne2=15). The batch>8 gate is therefore the **sole and total** cap on high-par decode.

RULED OUT by measurement (not just code-read): gate #1 arch (graphs ran at par1),
gate #3 ml8/weight-pager (`wp_pager` is **inactive** for `--kv-tier-paged-blocks` —
it only arms for `--weight-paging`/`params.weight_paging_enabled`; the model emits
plain `MUL_MAT_ID`, not `ML8_MUL_MAT_ID`), gate #4 `WP_HIP_GRAPHS` (only meaningful
when the pager is armed — which is why the earlier plain-KV no-op was the wrong path).

**MAD-288 nuance:** par1 replayed the *paged flash-decode* path 178× with coherent
output, so the per-call pool-alloc'd `partials` pointer is stable at steady state.
MAD-288 corruption is churn/pressure-triggered (tier movement, eviction, high-par
block turnover), NOT a blocker for basic capture — but the persistent-scratch port
is still required before we capture at par>8 under load.

**Scope:** this caps the WHOLE high-par decode (plain AND paged); paged is worse
because it adds kernels (scatter + 2-pass paged-attn + empty chunk-blocks →
336 vs plain 596 @ par128). Fixing graphs helps everything; the paged-specific
kernel reduction is a smaller follow-on.

## 3. Goal & success criteria

Enable HIP graph capture for high-concurrency (batch > 8) MoE decode, so that:

- **G1 — Scaling:** paged (features-on) aggregate decode **scales with
  concurrency** and lands within ~10% of feature-off at matched par; the flat cap
  is gone (target: par128 features-on ≥ ~550 tok/s, up from ~336).
- **G2 — Correctness:** bit-for-bit / PPL parity with the non-graph path
  (greedy determinism + a perplexity gate); no garbage under graph replay.
- **G3 — Stability:** no HIP stream-capture crashes on the full production path
  (paged + tiered 75/25 + semantic + eviction), sustained, at par up to 160.
- **G4 — Zero user config:** on by default in the standard server/router launch;
  no per-workload flags.

## 4. Approaches (to be finalized in Phase 0)

Three levers, likely combined:

- **A. Kill the MoE batch>8 stream sync (the core blocker).** Make the MoE
  token→expert binning happen **on-device** (a sort/scan/segmented layout kernel)
  instead of any host-side ids download, so `MUL_MAT_ID` at batch>8 is
  capture-safe. Then narrow/remove the batch>8 clause in
  `ggml_cuda_graph_check_compability`. This is the upstream-hard piece (PR #18958)
  but the real unlock. Sub-question: is MMQ already sync-free and only the
  *compat check* is over-conservative? If so, A collapses to "prove MMQ is
  capture-safe + narrow the check."
- **B. Make the paged/tiered path capture-safe.** (i) **Port the MAD-288
  persistent partials-scratch fix** from `llama-gpu` (mad-lab-2026) — mandatory, or
  paged capture corrupts on replay. (ii) Audit every other stream op the fused
  scatter+attn, tiered movers (`mt-mover-attn.cpp` hipMemcpy), and weight-pager
  eval-cb perform during a decode step for capture-legality (no blocking syncs, no
  pool-alloc'd pointers recorded into a graph). Understand precisely what broke,
  don't just flip flags. NOTE: `WP_HIP_GRAPHS` is a red herring (measured no-op);
  do not scope around it.
- **C. Graph reuse across decode steps.** Decode steps share graph structure and
  differ only in data pointers / a few scalars; confirm `properties_src_data_ptrs_only`
  holds each step so the executable graph is *updated*, not re-captured, every
  step (re-capture every step would erase the win). May require hoisting the few
  changing scalars (positions, context_lens) into capture-stable device buffers.

## 5. Risks

- **HIP graph stability on RDNA2** — graphs are historically flaky on ROCm; the
  reason WP_HIP_GRAPHS is gated. G3 (sustained stability under the full feature
  set) is the highest risk; budget real soak testing.
- **Correctness under replay** — pointer/scalar staleness across steps → silent
  wrong output. Greedy-determinism + PPL gate is mandatory (this session already
  burned two wrong theories; measure everything).
- **Upstream divergence** — the MoE on-device binning may be a large, partly
  upstream change; scope A carefully and prefer the smallest change that makes
  MMQ capture-safe.
- **Per-step re-capture regression** — if graph structure changes each step, we
  pay capture cost every step and go slower. C must be verified early.

## 6. Out of scope (follow-ons)

- Paged-specific kernel reduction (fuse scatter+attn+reduce; size decode
  `num_chunks` by ACTUAL context length not ALLOCATED ctx to kill empty
  chunk-blocks). Closes the residual paged-vs-plain gap AFTER graphs are on.
- The three decode fixes already landed this session (fanout gate, ctx-floor env,
  block-table skip) — valid and kept; not part of this epic.

## 7. Validation plan

- **Confirming test (Phase 0):** force graphs off at par≤8 (or on where legal) and
  show low-par decode drops to the flat high-par level — locks graphs as the lever
  before building.
- **Throughput:** the features-on synth sweep (`scratchpad/synth_sweep_featureson.sh
  {full|nosem|plain}`) par 24→160; success = features-on scales and ≥ ~90% of plain.
- **Correctness:** greedy determinism (graph vs no-graph, same prompt) + a
  perplexity parity gate on a fixed corpus.
- **Stability:** sustained par160 full-feature soak (paged+tiered+semantic+eviction),
  watch for stream-capture faults / amdgpu ring timeouts.

## 8. Key references (code)

- `ggml/src/ggml-cuda/ggml-cuda.cu`: `ggml_cuda_graph_check_compability` (~4240),
  the MUL_MAT_ID batch>8 clause (~4261), `ggml_cuda_wp_hip_graphs_enabled` (~4235,
  used ~5638), `ggml_cuda_mul_mat_id` dispatch (~3483).
- `ggml/src/ggml-cuda/mmvq.cu` / `mmvq.cuh`: `MMVQ_MAX_BATCH_SIZE=8`,
  `get_mmvq_mmid_max_batch_rdna1_rdna2` (~187).
- Paged path stream ops: `ggml/src/ggml-cuda/mt_pagedattn*.cu`,
  `src/memory-tier/mt-mover-attn.cpp`.
- Measurement harnesses (scratchpad): `synth_sweep_featureson.sh`, `prof_nosem.sh`,
  `gdb_sample.sh`, `decode_stream.py`, `parse_rocpd.py`.
