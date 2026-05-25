"""AWQ — Activation-aware Weight Quantization helpers for ml8.

Per-input-channel rescaling so salient channels (those receiving big activations)
get protected at quantization time. In grouped quantization, the group's scale
is set by the max magnitude in the group — without AWQ, one outlier channel
wastes the group's dynamic range. AWQ shrinks salient channels (W_awq = W / s
per column), making within-group magnitudes more uniform, so the per-group
quant scale captures more useful resolution for every channel.

Math identity (AWQ is exact in floating point):
    y = x @ W.T == (x * s) @ (W / s).T   for any positive s

At calibration: compute s_i = mean(|x_i|)^alpha over a calibration corpus,
rescale W = W / s per column, then quantize.

At reconstruction (HF-eval / inference): multiply each column of the dequantized
weight back by s. Math becomes:
    W_inference = dequant(W / s) * s
                = (W/s + quant_noise) * s
                = W + quant_noise * s

The quant noise is bounded by the quant scale of W/s. For salient channels,
W/s magnitudes are smaller, so the per-group scale is smaller, so noise is
smaller absolute — making AWQ + grouped quant a real quality win.

Compose with rotation (Hadamard) in pipeline order:
    1. AWQ on weight (W → W / s_diag)
    2. Rotation (W_awq → W_awq @ Q)
    3. Quantize the rotated AWQ weight
    4. At reconstruction: dequant, then absorb rotation (multiply by Q.T),
       then absorb AWQ scale (multiply columns by s).

Order in code: see calibrate_ml8.py main loop.
"""

import torch


def compute_awq_scale(x: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """Per-input-channel AWQ scale: s_i = mean(|x_i|)^alpha.

    x has shape (..., in_features) — typically (batch * seq, in_features) after
    flattening the batch/sequence dims. Returns shape (in_features,).

    alpha controls how aggressively we shift mass to salient channels:
      - alpha = 0   → s == 1 everywhere (no AWQ, identity)
      - alpha = 0.5 → AWQ paper default, square-root weighting
      - alpha = 1.0 → linear weighting, strongest AWQ effect
    """
    if alpha == 0.0:
        return torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    x_flat = x.reshape(-1, x.shape[-1])  # (N, in_features)
    mean_abs = x_flat.abs().mean(dim=0)   # (in_features,)
    # Clamp to avoid zero scales on dead channels (division blows up later otherwise).
    mean_abs = mean_abs.clamp_min(1e-8)
    return mean_abs.pow(alpha)


def apply_awq_to_weight(W: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Rescale weight by dividing each input-channel column by s_i.

    W shape (out_features, in_features), s shape (in_features,).
    Returns W_awq = W / s per column.
    """
    if W.shape[-1] != s.shape[0]:
        raise ValueError(f"W last dim {W.shape[-1]} != s shape {s.shape[0]}")
    return W / s.unsqueeze(0)  # broadcast across rows


def absorb_awq_in_reconstruction(W_awq: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Multiply each input-channel column by s_i to recover the inference-equivalent weight.

    Inverse of apply_awq_to_weight in floating point (exact when W_awq is the
    pre-quant rescaling; approximate when W_awq is the dequantized form).
    """
    if W_awq.shape[-1] != s.shape[0]:
        raise ValueError(f"W_awq last dim {W_awq.shape[-1]} != s shape {s.shape[0]}")
    return W_awq * s.unsqueeze(0)
