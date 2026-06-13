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
from ml8_backward_kernels import ml8_wgrad


class Ml8Fp8Fn(torch.autograd.Function):
    """Forward pass for ml8 weight-format-1 GEMM via the deployed LUT kernel.

    Inputs:
        x          -- fp32/bf16 activations [..., K]
        centroids  -- fp32 Parameter [G, 16]
        scales     -- fp32 Parameter [N, G]
        indices    -- uint8 Buffer   [N, K]
        gidx       -- long  Buffer   [K]

    Returns bf16 output [..., N].

    The STE boundary: forward computes with e4m3-snapped centroids
    (identical to the deployed kernel's LUT), while the gradient flows
    through the fp32 master centroids via the straight-through identity.
    """

    loss_scale = 1.0           # set by the trainer
    last_dLdW = {}             # side-channel: id(indices) -> dL/dW_raw (for Axis B)
    last_h = {}                # side-channel: id(indices) -> diag curvature h_k=E[x_k^2]
    # Each stashed dL/dW is a DENSE [N,K] fp32 tensor; populated every backward across
    # all targets it hoards ~the whole model in fp32 (~14GB on the 4B) in module state
    # autograd can't free — it OOM'd the 4B. Only the pv reassign path (and the pv
    # diagnostic) consume it, so capture is OFF by default; the trainer sets it True
    # only for a pv run. frozen/gptq/mse never read it.
    capture_dLdW = False

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

        ctx.indices_id = id(indices)
        ctx.save_for_backward(x8, sx, cent_e4m3, scales, indices, gidx)
        return y.reshape(*x.shape[:-1], y.shape[-1])

    @staticmethod
    def backward(ctx, dy):
        x8, sx, cent_e4m3, scales, indices, gidx = ctx.saved_tensors
        N, K = indices.shape
        dyf = dy.reshape(-1, dy.shape[-1]) / Ml8Fp8Fn.loss_scale      # [M,N]
        dy8, sdy = fp8_quant(dyf, "e5m2")
        # reconstruct raw e4m3 weight W[N,K] = cent_e4m3[gidx, indices] * scales[:,gidx]
        cent_per_col = cent_e4m3[gidx]                                # [K,16]
        W = cent_per_col.unsqueeze(0).expand(N, -1, -1).gather(
            2, indices.long().unsqueeze(-1)).squeeze(-1) * scales[:, gidx]   # [N,K]
        x = (x8.float() * sx)                                         # dequant acts [M,K]
        dx = (dy8.float() * sdy) @ W                                  # [M,K]
        dW_raw = (dy8.float() * sdy).t() @ x                          # [N,K]
        # Axis-B (pv) side channel — see capture_dLdW note above. OFF by default so
        # these dense [N,K] fp32 tensors are NOT retained in module state across layers
        # (that hoard is what OOM'd the 4B). dW_raw itself is still computed (transient)
        # for the dcent/dscales gradient below and freed when this layer's backward ends.
        if Ml8Fp8Fn.capture_dLdW:
            Ml8Fp8Fn.last_dLdW[ctx.indices_id] = dW_raw               # Axis B taps this
            # Diagonal GPTQ curvature h_k = E[x_k^2] (mean over the M batch rows, matching
            # the token-mean loss convention of dW_raw). pv v2 uses this to make the flip
            # criterion quadratic so argmin lands near the Newton point, not an extreme.
            Ml8Fp8Fn.last_h[ctx.indices_id] = (x * x).mean(dim=0)     # [K]
        # Fused wgrad: dcent[G,16] + dscales[N,G] from dW_raw + indices, no dense
        # W-scatter. gsz = K // G; groups are contiguous (gidx == arange(K)//gsz).
        gsz = K // scales.shape[1]
        dcent, dscales = ml8_wgrad(dW_raw, indices, cent_e4m3, scales, gsz)
        return (dx.reshape(dy.shape[:-1] + (K,)), dcent, dscales, None, None)


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
