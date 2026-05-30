import torch
from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8

def kurtosis(x: torch.Tensor) -> float:
    x = x.flatten().float(); x = x - x.mean()
    return (x.pow(4).mean() / x.pow(2).mean().pow(2).clamp_min(1e-12)).item()

def fp8_sensitivity_db(w: torch.Tensor, group_size: int = 32) -> float:
    wq = dequantize_scaled_fp8(quantize_scaled_fp8(w, group_size))
    return 10*torch.log10((w.pow(2).sum()/(w-wq).pow(2).sum()).clamp_min(1e-12)).item()

def report(name: str, w: torch.Tensor, group_size: int = 32) -> dict:
    """One-line record per SSM tensor for mlambaformer SSM characterization."""
    return {"name": name,
            "per_channel_kurtosis": kurtosis(w.float().abs().amax(0)) if w.ndim > 1 else None,
            "per_token_kurtosis": kurtosis(w),
            "fp8_snr_db": fp8_sensitivity_db(w, group_size)}
