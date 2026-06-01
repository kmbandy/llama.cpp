# ml8 Quantization — Architecture & Structural Overview

> **Status:** structural/architecture reference. Metrics, PPL numbers, and
> bench results are intentionally left as `TODO(metrics)` placeholders and will
> be filled in once the current calibration/validation runs complete. This doc
> describes *what ml8 is and how it works*, not how well it scores (yet).

---

## 1. What ml8 is, in one paragraph

**ml8 is mad-lab's quantization family.** It is built on a single core idea:
represent values with a small **learned codebook** whose entries are snapped to
the **E4M3 FP8 lattice**, so that at inference the dequantized values land
directly on the format the GPU's **FP8 matrix cores (WMMA)** consume — letting
the dequant→matmul path run on native FP8 hardware instead of a software
dequant loop. The codebooks are not fixed; they are **calibrated** to the actual
value distribution of a specific model. The family has two arms that share this
DNA:

- **The weight arm — `ml8-4`** (and sibling `ml8-N` precisions): codebook-indexed
  **model weights** at ~4 bits/value, calibrated with a GPTQ-style
  Hessian-aware loop.
- **The KV-cache arm — ml8 KV tiers** (`ml8-3` / `ml8-4` / `ml8-5`):
  codebook-quantized **K/V cache** at 3–7 bits/value, calibrated to the
  attention key/value distribution.

Both arms are the same machine (Lloyd-Max codebook + E4M3 snap + calibration +
FP8-WMMA-friendly LUT dequant) pointed at two different tensors (weights vs KV).

> **Naming note.** ml8 is the brand for **both** arms — weights *and* KV cache.
> The tier number tracks bits-per-value, so the same suffix can appear on both
> sides (e.g. weight `ml8-4` vs KV `ml8-4`); this doc disambiguates with
> "weight arm" / "KV arm" wherever it matters. (`turbo` is reserved for a
> different feature and is **not** an ml8 name.)

---

## 2. Background concepts (so the rest reads cleanly)

- **bpv — bits per value.** The amortized storage cost of one quantized scalar,
  including index bits plus amortized codebook/scale overhead. `ml8-4` ≈
  4.1–4.25 bpv; the ml8 KV tiers ≈ 3.125 (`ml8-3`) / 4 (`ml8-4`) / near-lossless
  (`ml8-5`) bpv.

- **Codebook / LUT quantization.** Instead of storing each value, store a small
  table (codebook) of `n_centroids` representative values, and per value store
  only a **k-bit index** into that table. `n_centroids = 16` → 4-bit index. At
  read time, the value is recovered by a **LUT lookup**.

- **E4M3.** An 8-bit floating-point format: 1 sign / 4 exponent / 3 mantissa.
  It is the lattice RDNA4 / gfx1201 FP8 matrix cores operate on. **ml8 snaps its
  codebook centroids onto this lattice** so the recovered values are already
  FP8-representable.

- **WMMA (Wave Matrix Multiply-Accumulate).** The GPU matrix-core instruction.
  gfx1201 exposes an FP8 variant (`v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12`) that
  multiplies two FP8 operands and accumulates in FP32. Because ml8 centroids are
  E4M3, the dequantized operand feeds this instruction directly.

- **Lloyd-Max.** An iterative algorithm that places codebook centroids to
  minimize quantization error (MSE, optionally magnitude-weighted) over a value
  distribution. It is how ml8 *chooses* its centroids.

- **GPTQ / Hessian-aware quantization.** A weight-quantization method that
  quantizes column-by-column and **propagates the residual error** of each
  column into the not-yet-quantized columns, weighted by the layer's Hessian
  `H = XXᵀ` (X = calibration activations). It deliberately accepts larger
  *element-wise* weight error to minimize *output* error.

- **Y_SNR vs W_SNR.** Two ways to score a quantized weight matrix:
  - **W_SNR** — element-wise weight error (`‖Q − W‖²`). Auxiliary only.
  - **Y_SNR** — Hessian-weighted **output**-space error
    (`trace((Q−W) H (Q−W)ᵀ) / trace(W H Wᵀ)`). **This is the headline metric**
    for GPTQ-class algorithms, because GPTQ optimizes output error, not
    element-wise error. A widening W_SNR↔Y_SNR gap means GPTQ is doing *more*
    useful work, not less.

---

## 3. The shared core mechanism

Every ml8 tensor (weight or KV) goes through the same conceptual pipeline:

```
  values ──▶ [optional rotation] ──▶ [group] ──▶ [Lloyd-Max fit] ──▶ [E4M3 snap] ──▶ codebook + indices
                                                                                          │
  inference:  indices ──▶ LUT lookup (E4M3 codebook in LDS) ──▶ FP8-WMMA matmul ──▶ FP32 accumulate
```

Three properties make this fast and accurate:

1. **Calibrated, not fixed.** The codebook is fit to the real distribution
   (per layer, per kind, per group as configured). A codebook tuned to the data
   beats a generic one.

2. **E4M3 centroids = free WMMA.** Snapping centroids to the FP8 lattice costs a
   few % in fit error but means the dequant output is *already* the matmul input
   format — no FP16 round-trip, and the LUT (16 FP8 bytes per group) is tiny
   enough to live in **LDS**, loaded once per K-tile at a group boundary.

3. **Scale absorbed into centroids.** Calibration folds each group's scale into
   the centroid values themselves. So the inference inner loop has **no separate
   `load(scale) + fmul`** — it is strictly *lighter* than a standard int4 path
   (which carries a per-group scale multiply). Perf target is therefore
   "match or beat int4," not "tie int4 plus LUT cost."

---

## 4. The weight arm — `ml8-4`

### 4.1 What it stores

For each weight matrix (or MoE expert), weights are split into **groups**
(e.g. `group_size = 64`). Each weight becomes a **4-bit index** (`n_centroids =
16`) into a codebook of E4M3 centroids, plus small per-group metadata (the group
scale, folded into the centroids for inference). Net ≈ **4.1–4.25 bpv** — the
4-bit index dominates; the fractional remainder is per-group overhead amortized
over `group_size`.

### 4.2 How it's calibrated (the pipeline)

`scripts/calibration/` implements a custom GPTQ loop (rolled in-house because
auto-gptq / gptqmodel don't install/run cleanly on this stack). For each linear:

1. **(Optional) rotation.** Apply an orthogonal transform to spread outliers
   before quantization (see §6). Makes the distribution flatter → easier to fit.
2. **Hessian.** Accumulate `H = XXᵀ` from calibration activations (the inputs
   that actually flow into this linear), with a small `percdamp` ridge for
   Cholesky stability.
3. **GPTQ column sweep.** Cholesky-of-`H⁻¹`; quantize each column to the codebook
   (per-column snap), then propagate the residual error into the remaining
   columns. This is what trades element-wise error for output fidelity.
4. **Lloyd-Max centroids.** Place the 16 centroids to minimize (mag-weighted)
   MSE over the group's value distribution.
5. **E4M3 snap.** Snap centroids onto the FP8 lattice (this is the step that
   makes it *actually ml8* rather than a generic 4-bit codebook quant).
6. **Score with Y_SNR** per kind (gate/up/down for FFN, etc.); W_SNR auxiliary.

Default recipe knobs (current): `fit_loss = mse`, `group_size = 64`,
`n_centroids = 16`, `percdamp = 0.05`, `snap_centroids = e4m3`, calibration set
of `n_samples × seq_len` tokens.

### 4.2.1 The calibration pipeline at a glance

The full weight-arm flow, end to end. Stage 0 is a one-time offline rewrite;
Stages 1–2 are what `calibrate_ml8_paged.py` runs; the heavy/act_order block is
the current near-lossless frontier (bit-free, validated bit-equivalent in dev,
PPL validation in progress).

```
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ STAGE 0 · (optional, offline, one-time)   rotate_model_quarot.py           │
 │   bf16 GGUF ──▶ QuaRot-R1 residual-stream Hadamard ──▶ rotated bf16 GGUF    │
 │   • equivalence-preserving (absorbed into residual-reading/writing          │
 │     linears + RMSNorm γ); calibration then runs --rotation none             │
 │   • BUILT but NOT used in the current 35B recipe — per-channel rotation     │
 │     has little headroom on this model (per-token outliers, wrong axis;      │
 │     barely moves down_proj). Shelved, see §6.1 caveat.                      │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1 · Hessian collection (forward passes)        [NVMe-paging-bound]   │
 │   calib corpus (n_samples × seq_len tokens)                                │
 │        │                                                                    │
 │        ▼   paged forward (experts streamed from disk via weight-pager)      │
 │   accumulate  H = XᵀX / n_tok   per (layer, kind)                           │
 │        │                                                                    │
 │        ▼                                                                    │
 │   cache ──▶ hessians.pt   key=(model/gguf/n_samples/seq_len/max_layers/     │
 │                                strategy)   ← reused → Stage 1 SKIPPED       │
 │   (dual-GPU data-parallel shard+merge: design; n_tok-weighted merge)        │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2 · GPTQ quantization        task queue, 1 worker/GPU (MANDATORY)     │
 │                                                                            │
 │   tasks = (layer × kind × expert-chunk)                                    │
 │      ├─────────────▶ [cuda:0  R9700]  ┐                                     │
 │      └─────────────▶ [cuda:1  6900XT] ┘  ~3:1 throughput, both compute      │
 │                                                                            │
 │   per task:                                                                │
 │     0. (optional) rotation — Kronecker input-side, applied in-calibration  │
 │          (--rotation kronecker) ; THIS is the rotation the 35B run uses     │
 │     1. GPTQ column sweep (straight order)                                  │
 │          Cholesky(H⁻¹) ; quantize col → propagate residual error           │
 │     2. Lloyd-Max centroid fit  (16 centroids, mag-weighted optional)        │
 │     3. E4M3 snap                                                           │
 │     ┌───────────────── heavy / act_order block (bit-free) ──────────────┐  │
 │     │ 4. act_order reassign:  perm = argsort(diag(H), desc)             │  │
 │     │       re-sweep in Hessian-importance order, reassign indices       │  │
 │     │       (groups stay in ORIGINAL space; only sweep order permuted)   │  │
 │     │ 5. heavy tune loop  × heavy_rounds:                                │  │
 │     │       Adam-tune centroids+scales on  tr((W−Wq) H (W−Wq)ᵀ)          │  │
 │     │         ↓ snap centroids to E4M3                                    │  │
 │     │         ↓ act_order reassign indices  (frozen → re-fit → repeat)    │  │
 │     └─────────────────────────────────────────────────────────────────┘  │
 │     6. score Y_SNR (output-space, headline) ; W_SNR auxiliary              │
 │     7. write blob ; append to manifest.json (resume-safe)                  │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ ASSEMBLE                                                                   │
 │   ml8_to_gguf.py: pack codebooks + 4-bit indices → ml8_4_soa GGUF (SoA)    │
 │   (optional) requantize_nonexpert.py: bf16 non-expert path → Q8_0          │
 │              (size win; experts left byte-identical)                       │
 └──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
 ┌──────────────────────────────────────────────────────────────────────────┐
 │ INFERENCE  (per K-tile)                                                    │
 │   4-bit index ──▶ LUT (16 E4M3 bytes in LDS) ──▶ FP8 operand               │
 │              ──▶ tl.dot(fp8, fp8, fp32) ──▶ FP32 accumulate                 │
 │   no scale multiply (folded into centroids)                                │
 └──────────────────────────────────────────────────────────────────────────┘
```

> **On act_order + the heavy loop (steps 4–5).** Both are bit-free (they change
> *which* centroid each weight maps to and *what* the 16 centroids are, never the
> index width). act_order is a pure reordering of the GPTQ sweep by Hessian
> importance; the heavy loop is AQLM/PV-tuning-style alternation between
> gradient-tuning the centroids/scales against the output-error loss and
> re-running act_order index assignment. Observed behaviour on Qwen3.6-35B-A3B:
> act_order helps `gate/up` most at early layers (washing out with depth), while
> the heavy loop gives a smaller but **uniform** lift on `down_proj` across all
> layers — consistent with `down` being the structural floor (see §6.1 caveat).
> `TODO(metrics)`: per-kind Y_SNR deltas and end-to-end PPL once the validation
> run lands.

### 4.3 MoE handling

For Mixture-of-Experts FFNs, each expert's `gate/up/down` projections are
quantized **independently** (each expert has its own learned distribution). The
calibration operates over the **consolidated expert stacks**
(`blk.L.ffn_{gate,up,down}_exps.weight`, shape `[n_experts, …]`) in batched
chunks across experts. Shared experts (`ffn_*_shexp`) and the router
(`ffn_gate_inp`) are quantized as ordinary 2D linears.

> **Memory-footprint note (gfx-relevant).** On a 32 GB R9700, an `ml8-4` 35B-A3B
> MoE (~24 GB) fits **fully resident in VRAM**. The disk weight-pager is the
> *calibration-time* mechanism and the fallback for models that don't fit — not
> the production decode path for models that do.

### 4.4 Inference kernel

The FP8-WMMA dequant→gemm kernel is specified in
`ggml/src/ggml-cuda/aiter-integration/ML8_WMMA_KERNEL_DESIGN.md`. Key points:

- The codebook is loaded into **LDS** once per K-tile at the group boundary
  (16 E4M3 bytes — trivial footprint).
- Inner loop: read 4-bit index → LUT → FP8 operand → `tl.dot(fp8, fp8, fp32)` →
  FP32 accumulate. No scale multiply (absorbed).
- Built on the AITER Triton GEMM kernels, vendored with a `WEIGHT_FORMAT`
  constexpr branch (same fork pattern as the ml8 KV-cache kernels).
- **RDNA4 gotcha:** on gfx12 WMMA, `lane % 16 = column` (not row, unlike gfx11).
  Wrong handling silently transposes output and *passes uniform-input smoke
  tests* — non-symmetric test inputs are required to catch it.

---

## 5. The KV-cache arm — ml8 KV tiers

### 5.1 What it does

KV-cache quantization shrinks the per-token K and V tensors so longer contexts
fit in VRAM and the attention matmul can run on FP8 cores. The ml8 KV tiers
trade bpv against fidelity:

- **`ml8-3` (KV)** — ~3.125 bpv, Lloyd-Max codebook. Most aggressive.
- **`ml8-4` (KV)** — ~4 bpv. Middle ground; strong "drop-in upgrade for q4_0 KV."
- **`ml8-5` (KV)** — ~near-lossless, modest savings vs raw FP8.

### 5.2 How it works there

Same core machine, pointed at the KV cache instead of weights:

- **Per-(layer, kv-direction) codebooks**, Lloyd-Max-fit to the K/V activation
  distribution, snapped to E4M3 so the attention `Q·Kᵀ` / `scores·V` matmuls run
  on FP8-WMMA.
- The **E4M3 lattice constraint costs only ~4–5% MSE** vs an unconstrained
  FP16-scale codebook — i.e. FP8-WMMA throughput is essentially free on quality.
- **Hadamard on K** spreads K outliers before fitting (helps; magnitude
  `TODO(metrics)` — observed modest on Qwen, outliers are more per-token than
  per-channel there).
- Quantization is calibrated, with a planned **two-tier UX**: a one-time
  ~30 s **auto-quick-fit** on first load (built-in corpus), plus an optional
  `--quick/--standard/--thorough` **fine-tune CLI** that writes per-model LUTs to
  a cache keyed by model fingerprint.

### 5.3 Inference & AMD fast-path constraints

- **Inline dequant inside the attention shader** is the preferred path (reading
  FP8/INT codebook entries in-kernel), rather than pre-dequantizing to an FP16
  scratch buffer — the offline-scratch approach adds measurable per-prompt
  overhead at the R9700's bandwidth.
- **Symmetric K/V quant type is required for the fused Flash-Attention fast path
  on AMD HIP.** Asymmetric (`-ctk X -ctv Y` with different types) silently falls
  back to a slow non-fused path. Keep K and V on the same ml8 KV type.
- Architecture note: only a fraction of layers in the hybrid Qwen3.5/3.6 family
  are full-attention (the rest are delta-net/SSM), so whole-model KV savings
  scale with the attention-layer fraction, not the layer count.

---

## 6. Shared infrastructure

### 6.1 Rotation (outlier spreading)

Quantization error is dominated by outliers — a few large-magnitude channels
force a wide codebook range that wastes resolution on the dense center. An
**orthogonal rotation** applied before quantization spreads that energy across
dimensions, flattening the distribution, while being **mathematically
equivalence-preserving** (the model computes the same function because the
rotation is absorbed into adjacent weights / the norm scale).

Lineage of the weight-side rotation:
- **Kronecker rotation** `Q = H_a ⊗ H_b` — factored Hadamard on the **input (K)**
  dimension, stored ~100,000× smaller than a dense `Q`.
- **QuaRot-R1 (residual-stream Hadamard)** — rotates the **residual stream**
  basis (`R = D ⊙ H_sylvester`), absorbed into every residual-reading linear
  (input side) and residual-writing linear (output side), with RMSNorm γ folded
  in. This reaches the **output (N)** dimension that the Kronecker input-rotation
  doesn't, giving the quantizer a different orthogonal basis to work in. The
  offline rotation pass (`scripts/calibration/rotate_model_quarot.py`) rewrites a
  bf16 GGUF in place so calibration can run `--rotation none` on already-rotated
  weights.

On the KV side, the analogous lever is **Hadamard on K** before codebook fitting.

> **Structural caveat (down_proj / SwiGLU).** Residual-stream rotation strongly
> helps the residual-*reading* projections (gate/up) but barely moves
> `down_proj`, whose quantization difficulty is **structural** — the SwiGLU
> gating gives `down_proj` a near-diagonal Hessian, so its hardness is the
> Hessian shape, not input outliers, and an input-side rotation has nothing to
> grab. Improving `down_proj` needs a different lever (e.g. an output-side
> transform or a different `group_size` for that kind). `TODO(metrics)`: per-kind
> Y_SNR deltas.

### 6.2 Dual output format

ml8 ships in **two first-class formats simultaneously**, neither a derivative of
the other:

1. **Native `.ml8`** — the canonical, lean mad-lab format for the mad-lab
   inference platform.
2. **GGUF-wrapped** (`model-ml8_4.gguf` convention) — tooling-compatible, drops
   into existing `llama.cpp` workflows for the OSS community.

### 6.3 Calibration tooling

`scripts/calibration/` holds the pipeline: the GPTQ driver, the centroid
quantizer (Lloyd-Max + optional mag-weighted + Hessian-aware), the rotation
passes, GGUF (de)serialization, the paged weight loader for calibrating models
larger than VRAM, and reporting (per-kind Y_SNR distributions, worst layers).
A vectorized batched Lloyd-Max kernel removed the dominant HIP-dispatch overhead
in the MoE calibration path.

---

## 7. Why it's built this way (design rationale)

- **AMD-first.** The whole stack targets RDNA4 (gfx1201 R9700) and siblings, with
  native FP8/INT4 WMMA as the throughput substrate. E4M3 centroids exist to land
  on those matrix cores.
- **Calibrated quality.** A learned, per-model codebook is the differentiator vs
  fixed-codebook quants — and the calibration is exposed as a user-facing feature
  (auto-quick-fit + optional fine-tune), not just an internal step.
- **Quality metric honesty.** Y_SNR (output-space) is the headline because the
  algorithm optimizes output error; element-wise W_SNR is reported as auxiliary
  only.
- **Mission framing.** Enterprise revenue funds the platform; the actual target
  is OSS home-inference users — hence the dual `.ml8` + GGUF output and the
  drop-in-on-consumer-AMD posture.

---

## 8. Metrics & results — `TODO(metrics)`

To be filled in after the current validation completes:

- [ ] `ml8-4` weight PPL vs Q4_K_XL and vs bf16 baseline (Qwen3.6-35B-A3B,
      wikitext-2, ctx=4096).
- [ ] Per-kind Y_SNR (gate / up / down), unrotated vs QuaRot-R1.
- [ ] `ml8-3/4/5` KV: MSE vs q4_0 / raw-FP8, PPL + NIAH across context lengths.
- [ ] Inference throughput: ml8-4 (weights) dequant-gemm vs int4 / f16; ml8 KV
      attention prefill/decode vs f16, on R9700.
- [ ] bpv exact bit-budget tables per format.

---

## 9. Related design docs

- `ggml/src/ggml-cuda/aiter-integration/ML8_WMMA_KERNEL_DESIGN.md` — weight
  inference kernel (FP8 WMMA).
- `ggml/src/ggml-cuda/aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md` — GGUF
  wrapping.
- `ggml/src/ggml-cuda/aiter-integration/TURBO_FP8_KERNEL_DESIGN.md` /
  `TURBO_FP8_CALIBRATION_DESIGN.md` — ml8 KV-cache arm (filenames + CLI flags
  predate the turbo→ml8 KV rename; the content is the ml8 KV design).
- `docs/aiter-integration/2026-05-28-ml8-hadamard-scatter-design.md` /
  `…-plan.md` — QuaRot-R1 weight rotation.
- `docs/aiter-integration/2026-05-28-ml8-moe-soa-design.md` — MoE structure-of-
  arrays repack.
