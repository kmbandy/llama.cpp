# MAD-214 Calibration Tooling

Offline calibration for the **turbo-FP8 KV cache** design (MAD-214). Produces FP8-constrained Lloyd-Max centroids and a go/no-go MSE report before any kernel work begins.

## Workflow

```bash
# 1. Extract KV samples from a HuggingFace model (Qwen3.5-4B prefill on wikitext + NIAH)
python3 extract_kv.py \
    --model Qwen/Qwen3.5-4B \
    --output-dir /tmp/mad214_kv \
    --num-wikitext 32 \
    --num-niah 32 \
    --wikitext-chunk-tokens 512 \
    --niah-ctx-tokens 8192

# 2. Fit FP8-constrained Lloyd-Max centroids + produce go/no-go MSE report
python3 fit_centroids.py \
    --input-dir /tmp/mad214_kv \
    --output-report /tmp/mad214_fit_report.json \
    --block-size 32 \
    --subsample-tokens 4096
```

## Go/no-go gate

`fit_centroids.py` prints a verdict at the end:

- **PROCEED** — turbo4_fp8 MSE within 1.5× of fp8_raw → go to Phase 1 (kernel implementation)
- **MARGINAL** — within 1.5×–3× → revisit design before committing to kernel. Knobs: per-head centroids, K/V split, Hadamard preprocessing for K outliers
- **STOP** — >3× → fundamental codebook fit is bad, redesign needed

## BPV math (corrected from MAD-214 ticket)

Per-block scale is stored as **FP32** (multiplied post-WMMA per hipfire production trick — folding into FP8 pre-WMMA risks E4M3 saturation). This makes BPV overhead 32/block_size bits per value, not 8/block_size.

With block_size=32:

| Variant | Index bits | Sign | Scale (FP32/32) | Total BPV | Saving vs FP8 |
|---|---|---|---|---|---|
| turbo3-FP8 | 3 | 1 | 1.0 | 5.0 | 37.5% |
| turbo4-FP8 | 4 | 1 | 1.0 | 6.0 | 25% |
| turbo5-FP8 | 5 | 1 | 1.0 | 7.0 | 12.5% |
| fp8_raw | — | — | — | 8.0 | baseline |

## Limitations of Phase 0 calibration

- **Global centroids only.** A single centroid table fits all layers + K + V combined. Per-layer / per-head / K-V-split granularity will improve MSE substantially. If the global Phase 0 fit is borderline, that's still a `PROCEED` once you account for granularity headroom.
- **HF transformers, not llama.cpp.** Distribution-equivalent (same weights) but doesn't exercise the actual deployment KV cache layout. Production recalibration belongs in Phase 1+.
- **MSE as PPL proxy.** Strong empirical correlation in quantization literature (`MSE within 1.5× → PPL degradation <0.05`), but not a substitute for actual PPL+NIAH. That's Phase 2.

## Files

- `extract_kv.py` — HF transformers KV cache extraction
- `fit_centroids.py` — Lloyd-Max fitting + E4M3 lattice snap + MSE report
- `README.md` — this file

## Reference

- MAD-214 ticket: turbo-FP8 KV cache design
- `ggml/src/ggml-cuda/aiter-integration/RDNA4_AUDIT_2026-05-20.md` Round 2 — prior art research
