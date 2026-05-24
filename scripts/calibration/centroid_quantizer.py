"""
MAD-223 ml8-4 weight quant — CentroidQuantizer.

Drop-in replacement for auto_gptq.quantization.Quantizer that snaps to a
shared per-group LUT of 16 signed centroids instead of a uniform INT4 grid.

Design (locked 2026-05-22):
  - Per (layer, in-channel-group, out-row): one scale (FP16).
  - Per (layer, in-channel-group): one shared LUT of 16 signed centroids.
  - Per element: 4-bit centroid index. Total ≈ 4.625 bpv at group_size=128.

Lloyd-Max:
  - SIGNED variant. Fits centroids directly on raw weight values (not on
    magnitudes — that was the KV path's sign-bit+magnitude factorization).
  - Optional weighted MSE:
      * fit_loss="mag_weighted": w_i = |x_i|^p. Empirically dominant lever in
        the MAD-214 KV calibration sweep; transfers to weights as the AWQ-style
        magnitude weighting.
      * Hessian-aware: when self.hessian_diag is set externally, samples in
        column j are weighted by H[j,j]. This composes with mag_weighted —
        effective weight = |x_i|^p * H[col_of_i, col_of_i].

Integration with auto-gptq:
  GPTQ(layer).quantizer = CentroidQuantizer(...)
  # before fasterquant():
  gptq.quantizer.hessian_diag = torch.diag(gptq.H)  # optional Hessian-aware fit
  scale, _, g_idx = gptq.fasterquant(group_size=128, actorder=True)
  # gptq.quantizer.collected_indices has per-group [rows, group_cols] int8 indices
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CentroidQuantizer(nn.Module):
    """auto-gptq Quantizer drop-in using a signed-N centroid LUT (default N=16)."""

    def __init__(self, n_centroids: int = 16, n_iter: int = 25):
        super().__init__()
        # Configured via configure() to match auto-gptq's lifecycle.
        self.n_centroids = n_centroids
        self.n_iter = n_iter
        # Filled by configure():
        self.fit_loss = "mag_weighted"
        self.mag_weight_p = 5.0
        self.sym = True
        # Filled by find_params():
        self.centroids: torch.Tensor | None = None  # [n_centroids], normalized signed
        self.scale: torch.Tensor | None = None      # [rows, 1], per-row
        # Hooks for GPTQ-aware fit (set externally before fasterquant):
        self.hessian_diag: torch.Tensor | None = None  # [in_features], one-shot
        self._group_col_offset = 0  # tracks which slice of hessian_diag applies
        # Per-quantize-call output capture (the snap step writes here):
        self.last_indices: torch.Tensor | None = None  # [rows, 1] int8
        # Accumulator: every quantize() call appends its [rows, 1] indices.
        # fasterquant calls quantize() once per column in order, so the list
        # is column-ordered; driver concats and un-permutes (if actorder).
        self.per_column_indices: list[torch.Tensor] = []
        # Per-group centroid LUTs (one per find_params() call):
        self.collected_centroids: list[torch.Tensor] = []
        # For auto-gptq compatibility — it checks .maxq > 0 in enabled() and
        # all(scale != 0) in ready().
        self.register_buffer("maxq", torch.tensor(self.n_centroids - 1))

    # ─── auto-gptq Quantizer interface ───────────────────────────────────────

    def configure(self, bits: int = 4, perchannel: bool = True, sym: bool = True,
                  mse: bool = False, fit_loss: str = "mag_weighted",
                  mag_weight_p: float = 5.0, **_ignored):
        """Called by auto-gptq pipelines. bits/perchannel/sym kept for parity."""
        assert (1 << bits) == self.n_centroids, (
            f"bits={bits} doesn't match n_centroids={self.n_centroids}")
        self.sym = sym
        self.fit_loss = fit_loss
        self.mag_weight_p = mag_weight_p
        # auto-gptq's gptq.py reads self.maxq to know group_size compatibility;
        # we expose it as N-1 so its sanity checks pass.
        self.maxq = torch.tensor(self.n_centroids - 1)

    def find_params(self, x: torch.Tensor, weight: bool = True):
        """Fit centroids + scale on a column-group slice of weights.

        Called by GPTQ.fasterquant once per group with x = W[:, group_start:group_end].
        x shape: [rows, group_cols]. We:
          1. Compute per-row scale from max(|x|) per row.
          2. Normalize: x_norm = x / scale  → values in [-1, 1] (roughly).
          3. Fit n_centroids signed centroids on x_norm via Lloyd-Max.
        """
        assert weight, "CentroidQuantizer is weight-only"
        assert x.dim() == 2, f"expected [rows, group_cols], got shape {tuple(x.shape)}"

        dev = x.device
        x_f = x.float()
        rows, group_cols = x_f.shape

        # Per-row scale (max abs in this group). Floor to avoid div-by-0.
        scale = x_f.abs().max(dim=1, keepdim=True).values.clamp_(min=1e-8)  # [rows, 1]
        x_norm = x_f / scale  # [rows, group_cols], roughly in [-1, 1]

        # Per-column Hessian weight, if available.
        # self._group_col_offset is set by the wrapper before fasterquant calls
        # find_params; tracks where in H we are.
        col_hw = None
        if self.hessian_diag is not None:
            end = self._group_col_offset + group_cols
            assert end <= self.hessian_diag.numel(), (
                f"hessian_diag size {self.hessian_diag.numel()} < group end {end}")
            col_hw = self.hessian_diag[self._group_col_offset:end].to(dev).float()  # [group_cols]

        centroids = _lloyd_max_signed(
            x_norm.flatten(),
            sample_col_idx=torch.arange(group_cols, device=dev).repeat(rows) if col_hw is not None else None,
            col_weights=col_hw,
            n_levels=self.n_centroids,
            n_iter=self.n_iter,
            fit_loss=self.fit_loss,
            mag_weight_p=self.mag_weight_p,
        )

        self.centroids = centroids.to(dev)  # [n_centroids]
        self.scale = scale.to(dev)          # [rows, 1]
        # auto-gptq's fasterquant appends self.zero alongside self.scale per
        # group. Centroid quant has no zero-point offset; expose a zero-filled
        # tensor of matching shape so the collection logic doesn't choke.
        self.zero = torch.zeros_like(self.scale)
        # Stash this group's LUT for export.
        self.collected_centroids.append(self.centroids.detach().clone())

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Snap each value in x to nearest centroid*scale. Stores indices.

        Called by GPTQ.fasterquant once per column with x = w.unsqueeze(1).
        x shape: [rows, 1]. Returns dequantized values, same shape.
        """
        if not self.ready():
            return x

        # x: [rows, 1], scale: [rows, 1], centroids: [n_centroids]
        x_norm = x / self.scale  # [rows, 1]
        # Distance to each centroid: [rows, 1, n_centroids]
        dist = (x_norm.unsqueeze(-1) - self.centroids).abs()
        idx = dist.argmin(dim=-1)  # [rows, 1]
        self.last_indices = idx.to(torch.int8)
        self.per_column_indices.append(self.last_indices.detach().clone())
        return self.centroids[idx] * self.scale  # [rows, 1]

    def ready(self) -> bool:
        return self.centroids is not None and self.scale is not None

    def enabled(self) -> bool:
        return self.maxq > 0

    # ─── Helpers for the calibrate_ml8.py driver ────────────────────────────

    def set_group_offset(self, col_offset: int):
        """Set before each fasterquant call so find_params can slice hessian_diag."""
        self._group_col_offset = col_offset

    def export(self) -> dict:
        """Return everything needed to write the ml8-4 weight format for this
        layer. Call AFTER fasterquant() completes.

          - 'indices' : [rows, in_features] int8, in original column order
                        (driver must un-permute if actorder=True was used).
          - 'centroids_per_group' : [n_groups, n_centroids] float
          - 'scale_per_group' : [rows, n_groups] float (matches auto-gptq's
                                returned scale tensor)
        """
        if not self.per_column_indices:
            raise RuntimeError("no quantize() calls captured; did fasterquant run?")
        indices = torch.cat(self.per_column_indices, dim=1)  # [rows, in_features]
        centroids = torch.stack(self.collected_centroids, dim=0)  # [n_groups, n_centroids]
        return {
            "indices": indices,
            "centroids_per_group": centroids,
        }

    def reset_capture(self):
        """Clear per-call accumulators so the same quantizer can be reused on
        the next layer."""
        self.per_column_indices.clear()
        self.collected_centroids.clear()
        self.last_indices = None
        self.centroids = None
        self.scale = None


# ───────────────────────────── Lloyd-Max kernels ─────────────────────────────

def _lloyd_max_signed(
    samples: torch.Tensor,
    *,
    sample_col_idx: torch.Tensor | None,
    col_weights: torch.Tensor | None,
    n_levels: int,
    n_iter: int,
    fit_loss: str,
    mag_weight_p: float,
) -> torch.Tensor:
    """Fit n_levels SIGNED centroids on `samples` (1-D, real-valued).

    Differences from fit_centroids_from_dump.lloyd_max_unsigned:
      * Operates on raw signed values, not magnitudes. Centroids span the
        full signed range.
      * Init by quantile of raw samples (not abs).
      * Returns sorted signed centroids.

    Weighting:
      * fit_loss="mag_weighted": per-sample w_i = |x_i|^p.
      * If col_weights given (Hessian-aware): per-sample weight is multiplied
        by col_weights[sample_col_idx[i]].
    """
    s = samples.float()
    s = s[torch.isfinite(s)]
    if s.numel() == 0:
        return torch.linspace(-1.0, 1.0, n_levels)

    # Initialize from quantiles of the signed distribution.
    q = torch.linspace(0.0, 1.0, n_levels, device=s.device)
    centroids = torch.quantile(s, q)

    # Build per-sample weights once.
    if fit_loss == "mag_weighted":
        w_mag = s.abs().pow(mag_weight_p)
    else:  # "mse"
        w_mag = torch.ones_like(s)
    if col_weights is not None:
        # Broadcast Hessian weight by column. sample_col_idx aligned with s
        # IFF samples were not deduped by isfinite-mask. To keep it correct,
        # we'd need to also filter sample_col_idx; for the skeleton we punt
        # and only apply Hessian weighting when caller guarantees no NaN.
        # (Linear weights from a healthy model: no NaNs in practice.)
        if sample_col_idx is not None and sample_col_idx.numel() == w_mag.numel():
            w_mag = w_mag * col_weights[sample_col_idx]
    weights = w_mag

    for _ in range(n_iter):
        # Edges between adjacent centroids → assign each sample to nearest.
        edges = (centroids[:-1] + centroids[1:]) / 2.0
        bins = torch.bucketize(s, edges)  # [N], values in [0, n_levels-1]

        new = centroids.clone()
        for k in range(n_levels):
            mask = bins == k
            if not mask.any():
                continue
            wk = weights[mask]
            sk = s[mask]
            denom = wk.sum().clamp_min(1e-30)
            new[k] = (wk * sk).sum() / denom

        if torch.allclose(new, centroids, rtol=1e-6):
            centroids = new
            break
        centroids = new

    return torch.sort(centroids).values


__all__ = ["CentroidQuantizer"]
