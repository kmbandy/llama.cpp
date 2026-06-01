# ml8-4 Weight Calibration

MAD-223 Phase B. End-to-end tooling to quantize a HuggingFace model's MLP weights
to **ml8-4** format: 16 signed centroids per group, ~4.125 bpv.

For KV-cache calibration (MAD-214), see the existing `README.md` in this directory.

## Pipeline overview

```
HF model (fp16) ──┐
                  │   calibrate_ml8.py
wikitext-2 ───────┤  ────────────────────►  per-layer .pt blobs + manifest.json
                  │   (Lloyd-Max + GPTQ)        │
                  │                             │
                  │                             ▼  reconstruct_model.py
                  │                         ml8-overlaid HF model + PPL re-eval
                  │
                  └─►  diagnose_calibration.py  (compare quant variants)
```

## Scripts

| Script | Purpose |
|---|---|
| `centroid_quantizer.py` | `CentroidQuantizer` — drop-in for auto_gptq.Quantizer (signed-16 LUT, MSE Lloyd-Max, optional mag_weighted / Hessian-aware) |
| `calibrate_ml8.py` | Main driver: HF model + wikitext-2 → per-layer GPTQ-quantized .pt blobs + manifest |
| `diagnose_calibration.py` | Runs 5 quantization variants on the same matrix (uniform INT4, naive ± loss, GPTQ ± loss) for debugging |
| `ml8_io.py` | `load_ml8_layer`, `reconstruct_weight`, `bits_per_value` — disk format helpers |
| `reconstruct_model.py` | Load HF model + overlay all saved .pt blobs → re-eval PPL on rehydrated model |
| `calibration_report.py` | Pretty-print `manifest.json` — Y_SNR distribution, per-kind, size, PPL, worst layers |

## Quick start

```bash
# 1. Calibrate (≈45 min for Qwen3.5-4B on R9700)
python3 calibrate_ml8.py \
    --model Qwen/Qwen3.5-4B \
    --output-dir /tmp/ml8-qwen3-4b \
    --n-samples 32 --seq-len 1024 --group-size 128 \
    --eval-ppl --ppl-max-tokens 100000

# 2. Summary
python3 calibration_report.py /tmp/ml8-qwen3-4b

# 3. Re-eval PPL from saved artifacts (verifies disk format)
python3 reconstruct_model.py \
    --model Qwen/Qwen3.5-4B \
    --calibration-dir /tmp/ml8-qwen3-4b \
    --eval-ppl --also-eval-baseline --ppl-max-tokens 100000
```

## Defaults (and why)

| Knob | Default | Rationale |
|---|---|---|
| `--fit-loss` | `mse` | Lloyd-Max with uniform sample weighting. **Do NOT use `mag_weighted p=5`** — it was the MAD-214 KV winner because activations have heavy outliers; weights are Gaussian-near-zero and mag_p5 starves the dense center (measured: 10 dB regression on Qwen3.5-4B). |
| `--n-centroids` | 16 | 4-bit indices (ml8-**4**). |
| `--group-size` | 128 | Matches auto-gptq convention. Smaller = more LUT overhead but better fit; 128 is the sweet spot. |
| `--percdamp` | 0.01 | Hessian diagonal damping for Cholesky stability. |
| `--n-samples` | 64 | Calibration corpus size in wikitext rows. 32 also works; 128 is overkill. |
| `--seq-len` | 2048 | Per-row tokenization length. Combined with `--n-samples`, target ≈30-130K total calibration tokens. |

## Metrics

Two SNR metrics are reported per linear:

- **W_SNR** — element-wise weight reconstruction (what naive snap optimizes)
- **Y_SNR** — output-space reconstruction weighted by H (what GPTQ optimizes) ← **the meaningful one**

GPTQ deliberately makes individual weight elements DIFFER MORE from the original to make the OUTPUT activations MATCH MORE closely. Expect GPTQ to LOSE on W_SNR and WIN big on Y_SNR. Typical Qwen-class MLP linears: Y_SNR 26–30 dB with MSE+GPTQ.

## File format (interim, before MAD-223 Phase C)

Each layer's `.pt` blob:

```python
{
    "name": str,                                # e.g. "model.layers.0.mlp.up_proj"
    "shape": [rows, in_features],
    "group_size": int,                          # 128
    "n_centroids": int,                         # 16
    "indices": int8 [rows, in_features],        # values 0..15
    "centroids_per_group": fp32 [n_groups, 16],
    "scale_per_group": fp32 [rows, n_groups],
    "mse": float, "w_snr_db": float, "y_snr_db": float, "rel_err": float,
}
```

Reconstruction: `W[r, c] = centroids_per_group[c // group_size][indices[r, c]] * scale_per_group[r, c // group_size]`

Phase C will define the native `.ml8` binary format + GGUF-wrapped variant (per the
decision to ship both formats — see saved KG decision 2026-05-22).

## Checkpoint / resume

`calibrate_ml8_paged.py` resumes a killed or crashed calibration from the per-linear
`.pt` blobs on disk. Resume is **on by default**; pass `--no-resume` for a clean run.
It works for both strategies, but the two are not symmetric:

- **MoE** precomputes every Hessian upfront from the original model, then quantizes
  experts independently — so resume simply skips per-expert blobs that already exist.
- **Dense** is interleaved: each layer's Hessian is computed against the running,
  partially **quantized** model (GPTQ cross-layer error propagation). So on resume the
  dense path reloads the **contiguous completed prefix** of blobs back into the model
  (dequant → inverse rotation → absorb AWQ, via `reconstruct_weight_from_blob`) before
  continuing. Only a contiguous prefix is trusted: a blob that exists *after* a gap was
  computed against a different upstream state and is stale, so the scan stops at the
  first missing blob and resumes there. See `dense_completed_prefix` and
  `load_dense_prefix_into_model`.

Because each layer's quantization is deterministic, a resumed run reproduces the same
blobs it would have produced uninterrupted.

## Acceptance gate (MAD-223)

`Δ_PPL = quantized − baseline` measured on wikitext-2 test split.

- **`Δ_PPL < 0.08`** → ship as-is (Phase B.5 AWQ scaling NOT needed)
- **`Δ_PPL ≥ 0.08`** → enable Phase B.5 AWQ activation-scaling preprocess

`calibrate_ml8.py --eval-ppl` and `calibration_report.py` both check this automatically.
