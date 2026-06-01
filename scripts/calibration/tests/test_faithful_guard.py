# scripts/calibration/tests/test_faithful_guard.py
import sys
from pathlib import Path
CALIB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CALIB))
import faithful_forward as ff

def test_assert_no_double_rotation_helper():
    # When faithful-acts is on, rotate_hessian must not be applied to H again.
    assert hasattr(ff, "assert_not_double_rotated")
    ff.assert_not_double_rotated(faithful_acts=True, rotate_hessian_called=False)  # ok
    try:
        ff.assert_not_double_rotated(faithful_acts=True, rotate_hessian_called=True)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
