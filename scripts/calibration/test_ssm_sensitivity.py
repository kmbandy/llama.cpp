import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import torch
from ssm_sensitivity import kurtosis, fp8_sensitivity_db

def test_kurtosis_gaussian_near_3():
    torch.manual_seed(0)
    k = kurtosis(torch.randn(100000))
    assert 2.5 < k < 3.5, f"gaussian kurtosis {k}"

def test_fp8_sensitivity_positive_db():
    torch.manual_seed(1)
    w = torch.randn(64, 128)
    db = fp8_sensitivity_db(w, group_size=32)
    assert db > 25.0

if __name__ == "__main__":
    test_kurtosis_gaussian_near_3(); test_fp8_sensitivity_positive_db()
    print("ALL SSM-SENSITIVITY TESTS PASSED")
