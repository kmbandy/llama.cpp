import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from scaled_fp8 import quantize_scaled_fp8, dequantize_scaled_fp8

def _snr_db(w, wq):
    return 10*torch.log10((w.pow(2).sum()/(w-wq).pow(2).sum()).clamp_min(1e-12)).item()

def test_roundtrip_snr():
    torch.manual_seed(0)
    w = torch.randn(256, 128)                       # [N, K]
    packed = quantize_scaled_fp8(w, group_size=32)  # per-group along K
    wq = dequantize_scaled_fp8(packed)
    assert wq.shape == w.shape
    assert _snr_db(w, wq) > 30.0, f"SNR too low: {_snr_db(w, wq):.1f} dB"

def test_scale_grouping_shape():
    w = torch.randn(64, 256)
    packed = quantize_scaled_fp8(w, group_size=32)
    assert packed["scale"].shape == (64, 256 // 32)
    assert packed["e4m3"].shape == (64, 256)

def test_zero_group_no_nan():
    w = torch.zeros(8, 32); w[0, :] = 1.0
    wq = dequantize_scaled_fp8(quantize_scaled_fp8(w, group_size=32))
    assert not torch.isnan(wq).any()

if __name__ == "__main__":
    test_roundtrip_snr(); test_scale_grouping_shape(); test_zero_group_no_nan()
    print("ALL SCALED-FP8 TESTS PASSED")
