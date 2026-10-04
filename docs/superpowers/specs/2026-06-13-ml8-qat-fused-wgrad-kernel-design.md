# Fused ml8-QAT Backward Weight-Gradient Kernel — Design

**Date:** 2026-06-13
**Task:** #220 (MAD-281) — rewrite the fp8 QAT backward off the dense-scatter substrate
**Status:** design approved, ready for implementation plan

## Problem

`Ml8Fp8Fn.backward` (`scripts/calibration/fp8_qat.py:93-125`) is the per-layer
weight-gradient pass of the fp8 QAT trainer. On the 4B it ran at ~16 s/micro
(~4 hr for a 3-arm verdict), and the Phase-E verdict (frozen vs gptq vs
gptq-interleave) is still unanswered because the trainer is too slow.

The prior framing (carried in the task and KG) was "the backward is
dispatch-bound / off-substrate dense PyTorch; rewrite it as a fused aiter/Triton
GEMM kernel + graph capture." **A micro-benchmark overturned that framing.**

### Measurement (R9700, single-layer backward, synthetic saved tensors)

Representative 4B linears, group_size=128. Per-component CUDA time and a
torch.profiler pass:

| component | mlp_down M=1024 K=9728 N=2560 | note |
|---|---|---|
| dy_quant | 0.19 ms | |
| W reconstruct (gather) | 1.69 ms | FLOP/bandwidth-bound, fine |
| dx GEMM | 4.93 ms | near torch FLOP ceiling |
| dW_raw GEMM | 2.93 ms | near torch FLOP ceiling |
| **dcent** (`index_put_` atomic) | **12.87 ms** | atomic-scatter into G×16 bins |
| **dscales** (`index_add_` atomic) | **18.52 ms** | atomic-scatter into N×G bins |
| **total** | **41.2 ms** | |

Profiler (attn shape, 10 iters): **GPU-busy fraction 186 %** (self CUDA
24.5 ms/layer > self CPU 13.2 ms/layer); 125 launches/layer; `aten::index_add_`
alone = 56 ms of 245 ms total CUDA — the single largest kernel.

### Conclusions (what the numbers actually say)

1. **The backward is GPU-compute-bound, not dispatch-bound.** CPU (launch)
   time is half the GPU time. **HIP-graph capture would buy ~nothing here.**
   The "30-55 % GPU / dispatch-bound" observation was the *inference* path
   (per-token, tiny M) and/or the full training step (teacher + student + gptq),
   not the backward in isolation.
2. **The GEMMs do not need fusing.** dx / dW_raw / W-reconstruct are already at
   the torch FLOP ceiling. A fused dgrad LUT kernel is **out of scope** — it
   would chase work that is not the bottleneck.
3. **The entire cost is atomic-scatter contention into tiny codebook bins**
   (`dcent`, `dscales`).
4. **`dscales` never needed a scatter at all.** `gidx = arange(K)//gsz` means
   groups are contiguous K-blocks, so
   `index_add_(1, gidx, contrib)  ≡  contrib.view(N, G, gsz).sum(2)`.
   Verified exact (max|Δ| = 1.07e-4) and **95× faster** (16.84 ms → 0.18 ms).
5. **`dcent` is the only place a custom kernel earns its keep.** Pure-torch
   alternatives lose: a reshaped `scatter_add` measured **17× slower** than the
   current `index_put_` (192 ms vs 11 ms). It needs an LDS-histogram kernel.

## Goal

Replace the two atomic-scatter blocks in `Ml8Fp8Fn.backward` with a single fused
Triton kernel that computes **both** `dcent` and `dscales` directly from
`dW_raw` + `indices`, with no dense `W` reconstruction for the weight grads and
no atomic-scatter. Target ~41 → ~12 ms/layer (~3.5×), per-micro backward floor
~8.2 s → ~2.4 s. Everything is gated against the current backward as oracle and
re-timed before "done."

## Architecture

Both weight gradients consume the same two `[N,K]` tensors — `dW_raw` and
`indices` — so they fuse into one kernel that reads each once:

```
dscales[n,g] = Σ_{k∈group g} dW_raw[n,k] · cent[g, idx[n,k]]
dcent[g,c]   = Σ_{n, k∈group g, idx[n,k]=c} dW_raw[n,k] · scales[n,g]
```

- **dscales** is a per-`(n,g)` reduction over the gsz columns of a group →
  each output element is owned by exactly one program → **zero atomics**.
- **dcent** is a 16-bin weighted histogram per group → 16 fp32 register
  accumulators per program → **atomic-add only across N-tiles** into the G×16
  output (negligible contention).

**Kernel layout.** Grid `(G, num_n_tiles)`. Program `(g, nt)` owns row-block
`nt` (BLOCK_N rows) and the contiguous K-slab `[g·gsz : (g+1)·gsz]`. It loads
`dW_raw[BLOCK_N, gsz]` and `idx[BLOCK_N, gsz]`, loads `scales[BLOCK_N, g]` and
the group's `cent[g, 0:16]`, then:
- accumulates `dscales[rows, g]` per row over the gsz columns (LUT-gathered
  centroid value × dW_raw), writing each `(n,g)` once (no atomic);
- loops `c ∈ 0..15`, `partial[c] += sum(where(idx==c, dW_raw, 0))` over the
  tile, then `atomic_add` the 16-vector × `scales` into `dcent[g, :]`.

This mirrors the forward kernel's per-K-group LUT structure (`gemm_ml8.py`
WEIGHT_FORMAT=1) on the gradient side. `num_stages=1` (gfx1201 RDNA4 audit:
num_stages≥2 triggers a UAF — same constraint the forward kernel documents).

## What stays unchanged (deliberate non-goals)

- `dy_quant`, `W` reconstruction, `dx` GEMM, `dW_raw` GEMM — all FLOP-bound and
  near ceiling; left as torch. **No dgrad LUT kernel, no `dx` fusion, no graph
  capture** — the measurement says none of these is the bottleneck.
- The `capture_dLdW` (pv / Axis-B) side channel still receives `dW_raw` and
  `h = E[x²]`; `dW_raw` is still materialized by its GEMM, so this path is
  untouched.

## Components / files

- **New: `scripts/calibration/ml8_backward_kernels.py`**
  - `@triton.jit _ml8_wgrad_kernel(...)` — the fused kernel above.
  - `ml8_wgrad(dW_raw, indices, centroids, scales, gsz) -> (dcent, dscales)` —
    Python wrapper: shape/stride checks, grid calc, output allocation.
  - Lives with the trainer (training-only), **not** in
    `ggml/src/ggml-cuda/aiter-integration/` (that is the deployed inference
    path; this kernel never ships in a GGUF).
- **Modify: `scripts/calibration/fp8_qat.py:93-125`** — replace the `dcent`
  `index_put_` block and the `dscales` `index_add_` block with one
  `ml8_wgrad(...)` call. `dy_quant`, `W`, `dx`, `dW_raw`, and the
  `capture_dLdW` stash stay byte-identical.

## Correctness, fallback, performance (all measure-gated)

- **Oracle = the current backward.** TDD: `ml8_wgrad` output matches the
  `index_add_` / `index_put_` results within fp tolerance across attn / mlp_up /
  mlp_down shapes **and** an odd-`N` (non-tile-multiple) masking case.
- **`dscales` reshape** is independently proven exact and lands even if the
  Triton kernel is descoped.
- **Auto-fallback.** A one-time probe at first call compares `ml8_wgrad` against
  the pure-torch path (`dscales` reshape + `index_put_` dcent) on the live
  device; if the kernel does not win, or raises on gfx1201, fall back to the
  pure-torch path. The reshape half-win is banked regardless. A
  `ML8_WGRAD_BACKEND={auto,triton,torch}` env overrides the probe for testing.
- **End-to-end gate.** A full-backward equality test (`Ml8Fp8Fn.backward` with
  the kernel vs a frozen reference) over a small real-shaped layer, plus a
  re-timed per-layer number (< current 41 ms) recorded before the design is
  called done.

## Expected payoff

Backward ~41 → ~12 ms/layer (~3.5×). Per-micro backward floor ~8.2 s → ~2.4 s.
Unblocks the 4B Axis-B verdict (frozen vs gptq vs gptq-interleave), then Phase F
(re-emit + PPL gate). The separate streaming-memory model (init_idx NVMe re-read,
weight paging) is a follow-up spec — deferred because the four committed memory
fixes already put the 4B at 9.68 GB peak (< 10 GB of 32 GB).
