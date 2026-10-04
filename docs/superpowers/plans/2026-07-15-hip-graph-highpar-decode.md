# HIP Graph Capture for High-Par Decode — Implementation Plan

> **⛔ SUPERSEDED 2026-07-15 (post-dive measurement). The premise was wrong.**
> HIP graphs give ~0 throughput on this workload (par1 99.5=99.5 graphs on/off;
> par128 735 vs 714; and graphs can't stay resident under continuous batching's
> per-step property churn). The "38% idle → launch latency → need graphs" reading
> was a misdiagnosis: plain KV is *more* idle (53%) yet faster — it's kernel WORK,
> not launch latency.
>
> **REAL root cause (rocprofv3-proven):** the features-on scaling deficit is the
> PAGED PREFILL ATTENTION using the slow scalar `mt_paged_attention_kernel`
> (`ggml/src/ggml-cuda/mt_pagedattn.cu`, launch ~1602). It is **47% of all paged
> GPU time** (5943ms/372 calls @ par64) vs plain KV's `flash_attn_ext_vec` at
> 1137ms for the same attention — ~5× the work. The flash-decode kernel is fast
> but gated to q_len≤8 (`decode_gate_on` ~1823); prefill q_len (hundreds) falls to
> the scalar path. Fixing paged prefill attention (q-tiled flash / partitioned
> scalar / new flash-prefill kernel) is the actual MAD-348 lever. New spec/plan to
> follow. Phase 1 below (narrow the MUL_MAT_ID graph gate) is a correct,
> upstreamable cleanup but NOT a throughput lever here — keep or drop.
>
> Decomposition (naive metric, par128): plain decode 518 / prefill 14418; paged+tier
> 345 / 7008; +semantic 295 / 5812. Tiering is NOT the cost (all-hot 100/0/0 ≈ 75/25).

**Spec:** docs/superpowers/specs/2026-07-15-hip-graph-highpar-decode-design.md
**Date:** 2026-07-15 · **Status:** SUPERSEDED — see banner above
**Rule:** MEASURE before/after every change. Three wrong theories killed by
measurement this session (copyBuffer, block-table upload, HIP graphs) — the rocprof
kernel breakdown is the ground truth; every fix gets a measured gate.

Build: `cmake --build build-hip --target llama-server -j 6` (gfx1201+gfx1030,
~2 TUs + relink for a .cu change, few min). Always `--no-mmap`. 6900XT only
(`--device ROCm1`); NEVER the R9700 (ROCm0, DSWS) or the RX480/Vulkan. Ask
before touching any GPU beyond the 6900XT.

---

## Phase 0 — Lock the premise + resolve the open questions

**T0.1 — Confirming test: graphs ARE the lever. ✅ DONE (2026-07-15).** Built an
env-gated diag (`GGML_GRAPH_DIAG=1`, at `GGML_LOG_WARN` since ggml INFO is muted in
llama-server) counting capture/replay/direct per graph_compute. Result: par1 = 93%
graph **replay** (108 tok/s, coherent); par8/9/128 = 94–99.6% **direct** launches.
Graphs are the lever, and they collapse exactly when the batch crosses 8.

**T0.3 — Graph baseline at par≤8. ✅ DONE (2026-07-15).** Definitive, not inferred:
graphs capture+replay at true batch-1; there is **no par≤8 escape hatch** because
continuous batching aggregates the routed MUL_MAT_ID batch past 8 nearly every step
at any real concurrency (par8 already ne2=15). Disabling clause is **always**
`GGML_OP_MUL_MAT_ID` batch>8 (op=30) — ml8/pager/arch/WP all ruled out by measurement.
MAD-288: par1 replayed the paged decode path 178× coherently (steady-state pool ptr
stable) — corruption is churn-triggered, so the persistent-scratch port is required
before capturing par>8 under load, but is NOT blocking basic capture.

**T0.2 — Does batch>8 MMQ `mul_mat_id` actually sync? ← IMMEDIATE NEXT GATE.** This
now solely decides Phase 1's size. Read `ggml_cuda_mul_mat_q`'s mul_mat_id path
end-to-end (dispatch at `ggml_cuda_mul_mat_id` ~3483, batch>8 → MMQ); determine
whether it does any host-side ids download / `hipStreamSynchronize` (like the ml8
path) or bins experts on-device. If needed, probe with a capture-illegal-op assert
under `GGML_GRAPH_DIAG`.
*Gate:* definitive yes/no + exact sync site — or "MMQ is capture-safe; the compat
check is merely over-conservative" (→ Phase 1 collapses to narrowing the check).
*Owner:* codex `--model gpt-5.6-terra` (per user directive; it's a code-read + maybe a probe).

**T0.4 — Clean plain-vs-paged par128 differential** (the earlier sed-failed run).
rocprofv3 both, diff decode-phase kernel counts + occupancy, to size how much is
graphs-general vs paged-specific.
*Gate:* attribution numbers feeding Phase 4 sizing.

## Phase 1 — Narrow the over-conservative `MUL_MAT_ID` graph gate (the core unlock)

T0.2 resolved: **MMQ is sync-free** (device-side ids binning via `mmid.cu`). So this
is the SMALL branch — narrow `ggml_cuda_graph_check_compability` (`ggml-cuda.cu:4261`)
to mirror the `ggml_cuda_mul_mat_id` dispatch (`3524-3566`) exactly:

- Graph-**SAFE** iff the op would take a device-side path:
  1. MMVQ: `ggml_is_quantized(src0->type) && !is_tq && ne[2] <= get_mmvq_mmid_max_batch(type,cc)`, OR
  2. MMQ: `ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], src0->ne[2])`, OR
  3. MMF: `ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)`.
- Disable ONLY if none match (→ the D2H+sync dequant fallback at `3595-3618`), plus
  keep the existing `is_tq`, non-quant-with-no-MMF, `ML8_MUL_MAT_ID`, and split-buffer
  exclusions.
- `routing_active` forces MMQ but also implies the weight-pager set `DISABLE_GRAPHS`,
  so graphs are globally off then — moot for this gate; the predicate can ignore it.

**Correctness rule:** the compat predicate MUST match the dispatch exactly, or we
capture a graph containing the sync and corrupt/crash. Prefer a **shared helper**
(e.g. `ggml_cuda_mul_mat_id_graph_safe(node)`) called by BOTH the dispatch's assert
path and the compat check — single source of truth — over duplicating the branch.

**Validate on PLAIN KV first (isolation):** the MoE still emits `MUL_MAT_ID` under
plain KV, but there's no paged flash-decode `partials` → no MAD-288 exposure. This
tests the check-narrowing alone before the paged path is layered on.

*Gate (T1, plain KV par128):* `GRAPH_DIAG` flips from ~99% DIRECT to mostly REPLAY;
greedy-token parity vs graphs-off (identical tokens, fixed prompt); coherent output;
decode throughput climbs toward plain-with-graphs. Only then proceed to Phase 2.

## Phase 2 — Make the paged/tiered path capture-safe

**T2.1 — Port the MAD-288 partials-scratch fix (mandatory first).** Bring the
persistent per-device `paged_decode_partials_scratch` (cudaStreamIsCapturing-guarded
grow) from `llama-gpu` (mad-lab-2026, gpu-portability worktree) into
`ggml/src/ggml-cuda/mt_pagedattn.cu` here, replacing the per-call
`ggml_cuda_pool_alloc(ctx.pool(), partials_n)`. Without this, paged capture corrupts
on replay. Gate: paged par8 with graphs forced on stays coherent + greedy-parity.

**T2.2 — Audit remaining capture-illegal ops** in the decode step: fused paged
scatter+attn ordering, tiered movers (`mt-mover-attn.cpp` hipMemcpy — should be
inactive at hot-tier depth but verify), weight-pager eval-cb pointer array. Make
graph reuse (Phase-C / `properties_src_data_ptrs_only`) hold each decode step (hoist
changing scalars — positions, context_lens — into capture-stable device buffers).
NOTE: `WP_HIP_GRAPHS` is a measured no-op — the goal is graphs running on the DEFAULT
launch, not flipping that flag.

*Gate (T2):* sustained par160 full-feature (paged+tiered+semantic+eviction) decode,
graphs on, no capture faults, coherent output.

## Phase 3 — Validation gauntlet

- **Correctness:** greedy determinism (graph vs no-graph, identical tokens on a
  fixed prompt at par1 and par128) + perplexity parity on a fixed corpus (tol from
  the mad-lab ~±0.05 PPL noise floor).
- **Throughput:** `synth_sweep_featureson.sh {full|nosem|plain}` par 24→160;
  success = features-on SCALES and ≥ ~90% of plain (G1).
- **Regression:** deep-context par8 (murmur, depth 6110) unchanged; single-stream
  decode unchanged.
- **Stability soak:** long par160 full-feature run; watch amdgpu rings / compositor
  (note: compute on a display GPU can starve the compositor — see the yield rule).

*Gate (T3):* G1–G4 all green with numbers in the plan ledger.

## Phase 4 — (Follow-on, separate) paged-specific kernel reduction

After graphs are on, close the residual paged-vs-plain gap: fuse paged
scatter+attn+reduce where possible; size decode `num_chunks` by ACTUAL max
context_len not ALLOCATED ctx (kill empty chunk-block launches). Re-measure.

---

## State carried into the dive (uncommitted)

Three decode fixes landed this session, all in the paged decode path, all
validated, NONE addressing the graph cap (keep them; commit after review + the
router-ini alias cleanup):
- **Fanout gate** — `mt_pagedattn.cu` gate on true per-seq max q_len via
  `op_params[4]` (set in `llama-graph.cpp`); fixed decode-at-depth (par8 219≈222).
- **Ctx-floor env** — `get_paged_decode_min_ctx()` default 512 (was hard 8192),
  `GGML_PAGED_DECODE_MIN_CTX`; flash-decode now fires at short ctx.
- **Block-table skip** — `prepare_batch_tensors` content-compare shadow
  (`h_block_table_gpu_` in `llama-kv-cache-paged.h`); skips unchanged full-table
  uploads (hygiene; helps deep-ctx murmur, not the synth cap).

Also owed before any commit: strip the ~30 ablation aliases from both
`docs/examples/router-fleet-*.ini`; fix/delete the stale anti-scaling section in
the RESULTS.md investigation doc.

Baseline numbers to beat (6900XT, par128 short-ctx, features-on paged):
~336 tok/s decode; plain (features-off) ~596; target ≥ ~550 features-on.
