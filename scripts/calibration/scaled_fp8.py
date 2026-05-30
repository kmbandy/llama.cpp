import torch
from centroid_quantizer import snap_to_e4m3   # existing E4M3 lattice round

E4M3_MAX = 448.0

def quantize_scaled_fp8(w: torch.Tensor, group_size: int = 32) -> dict:
    """Per-group (along K, the last dim) scale + e4m3 cast. w: [N, K]."""
    N, K = w.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    g = K // group_size
    wg = w.reshape(N, g, group_size)
    scale = wg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / E4M3_MAX   # [N, g, 1]
    e4m3 = snap_to_e4m3(wg / scale).reshape(N, K)
    return {"e4m3": e4m3.to(torch.float32), "scale": scale.reshape(N, g).to(torch.float16),
            "group_size": group_size, "shape": (N, K)}

def dequantize_scaled_fp8(packed: dict) -> torch.Tensor:
    N, K = packed["shape"]; gs = packed["group_size"]; g = K // gs
    e4m3 = packed["e4m3"].reshape(N, g, gs)
    scale = packed["scale"].to(torch.float32).reshape(N, g, 1)
    return (e4m3 * scale).reshape(N, K)
