# scripts/calibration/tests/test_faithful_forward.py
import sys
from pathlib import Path
import torch
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
from faithful_forward import build_rotations   # noqa: E402

def test_build_rotations_matches_inline_formula():
    # mirrors calibrate_ml8_paged.py:1390-1397 seed math
    dims = {("L0", "ffn_gate"): 2560}
    seeds = {("L0", "ffn_gate"): 5 + 0 * 7 + 0}
    rots = build_rotations(dims, seeds, max_b=1024)
    from kronecker_rotation import KroneckerRotation, random_orthogonal, factor_for_dim
    a, b = factor_for_dim(2560, max_b=1024)
    ref = KroneckerRotation(h_a=random_orthogonal(a, seed=5), b_dim=b)
    x = torch.randn(3, 2560)
    assert torch.allclose(rots[("L0", "ffn_gate")].forward(x), ref.forward(x), atol=1e-6)
