# scripts/calibration/tests/test_faithful_wiring_smoke.py
import sys
from pathlib import Path
import torch, torch.nn as nn
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
import faithful_forward as ff

def test_persistent_hooks_propagate_then_hessian_targets_one():
    torch.manual_seed(0)
    K = 256
    l1, l2 = nn.Linear(K, K, bias=False), nn.Linear(K, 4, bias=False)
    model = nn.Sequential(l1, l2)
    rots = {l1: ff.build_rotations({"a": K}, {"a": 1})["a"],
            l2: ff.build_rotations({"b": K}, {"b": 2})["b"]}
    hooks = {m: ff.FaithfulActHook(r, enabled=True) for m, r in rots.items()}
    handles = [m.register_forward_pre_hook(h) for m, h in hooks.items()]
    hooks[l2].set_hessian_target(True)
    x = torch.randn(6, K)
    model(x)
    for h in handles: h.remove()
    assert hooks[l2].H is not None and hooks[l1].H is None   # only l2 targeted
    assert hooks[l2].H.shape == (K, K)
