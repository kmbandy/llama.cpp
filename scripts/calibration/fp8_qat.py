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
