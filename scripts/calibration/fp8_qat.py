# fp8_qat.py
import torch

FP8_E4M3_MAX = 448.0
FP8_E5M2_MAX = 57344.0
_FMT = {"e4m3": (torch.float8_e4m3fn, FP8_E4M3_MAX),
        "e5m2": (torch.float8_e5m2, FP8_E5M2_MAX)}

def fp8_quant(x: torch.Tensor, fmt: str = "e4m3"):
    """Per-row (last-dim) amax scaling → fp8. Returns (x_fp8, scale[*,1])."""
    dt, fmax = _FMT[fmt]
    amax = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = (amax / fmax).clamp_min(torch.finfo(torch.float32).tiny)
    scale = torch.where(amax > 0, scale, torch.ones_like(scale))
    x_fp8 = (x / scale).to(dt)
    return x_fp8, scale

def pad_to_multiple(x: torch.Tensor, m: int, dim: int = 0):
    """Zero-pad `x` along `dim` up to a multiple of `m`. Returns (padded, n_pad)."""
    n = x.shape[dim]
    n_pad = (-n) % m
    if n_pad == 0:
        return x, 0
    shape = list(x.shape); shape[dim] = n_pad
    pad = x.new_zeros(shape)
    return torch.cat([x, pad], dim=dim), n_pad

import ml8_runtime
from centroid_quantizer import snap_to_e4m3


class Ml8Fp8Fn(torch.autograd.Function):
    """Forward pass for ml8 weight-format-1 GEMM via the deployed LUT kernel.

    Inputs:
        x          -- fp32/bf16 activations [..., K]
        centroids  -- fp32 Parameter [G, 16]
        scales     -- fp32 Parameter [N, G]
        indices    -- uint8 Buffer   [N, K]
        gidx       -- long  Buffer   [K]

    Returns bf16 output [..., N].  Backward raises NotImplementedError (Task A.5).

    The STE boundary: forward computes with e4m3-snapped centroids
    (identical to the deployed kernel's LUT), while the gradient flows
    through the fp32 master centroids via the straight-through identity.
    """

    @staticmethod
    def forward(ctx, x, centroids, scales, indices, gidx):
        # STE: snap centroids to e4m3 lattice in forward, identity grad to fp32 master.
        cent_e4m3 = centroids + (snap_to_e4m3(centroids) - centroids).detach()

        # Build kernel-ready Ml8Layer from live trainer tensors.
        layer = ml8_runtime.layer_from_components(
            centroids=cent_e4m3, scales=scales, indices=indices, gidx=gidx,
            device=x.device)

        K = x.shape[-1]
        xf = x.reshape(-1, K).to(torch.float32)   # flatten leading dims, ensure fp32
        M = xf.shape[0]

        # Per-row fp8 quantization of activations.
        x8, sx = fp8_quant(xf, "e4m3")            # x8:[M,K] fp8, sx:[M,1] fp32

        # Pad M to multiple of 16 (kernel block_size_m requirement).
        x8p, n_pad = pad_to_multiple(x8, 16, dim=0)   # [M', K]
        sxp, _ = pad_to_multiple(sx, 16, dim=0)       # [M', 1]

        # a_scale must be [M'] contiguous float32 (matches Ml8Linear.forward contract).
        a_scale = sxp.squeeze(-1).contiguous()         # [M']

        y = ml8_runtime.ml8_gemm(x8p, layer, a_scale=a_scale)   # [M', N] bf16

        # Strip padding rows.
        if n_pad:
            y = y[: y.shape[0] - n_pad]               # [M, N]

        ctx.save_for_backward(x8, sx, cent_e4m3, scales, indices, gidx)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        raise NotImplementedError("Task A.5")


def ml8_ref_linear(x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """Reference fp8 linear y = x @ W.T via torch._scaled_mm (test oracle)."""
    x8, sx = fp8_quant(x, "e4m3")                       # [M,K], [M,1]
    w8, sw = fp8_quant(W, "e4m3")                       # [N,K], [N,1]
    # _scaled_mm(A[M,K], B[K,N]) needs B to be column-major.
    # w8 is [N,K] row-major; w8.contiguous().t() is [K,N] column-major.
    w8_col = w8.contiguous().t()                         # [K,N], column-major
    out = torch._scaled_mm(x8, w8_col,
                           scale_a=sx, scale_b=sw.t(),
                           out_dtype=torch.bfloat16)
    return out
