#!/usr/bin/env python3
"""Test 'soft' determinism: seed + pinned GEMM workspace + tf32 off, but WITHOUT the strict
torch.use_deterministic_algorithms(True) hammer. Hypothesis: the hammer is what perturbs
cholesky_inverse into a non-PD H^-1 (breaking GPTQ's second Cholesky on 130/151 tensors), while
softer determinism still gives bit-reproducible calibration. Disables the driver's built-in
(hard) determinism block via ML8_NONDETERMINISTIC=1, then sets the soft subset here."""
import os
import sys

os.environ["ML8_NONDETERMINISTIC"] = "1"                  # skip the driver's hard-determinism block
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HIPBLASLT_DETERMINISTIC", "1")

import torch  # noqa: E402

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
for _obj, _attr, _val in (
    (torch.backends.cudnn, "deterministic", True),
    (torch.backends.cudnn, "benchmark", False),
    (torch.backends.cudnn, "allow_tf32", False),
    (torch.backends.cuda.matmul, "allow_tf32", False),
    (torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False),
    (torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False),
):
    try:
        setattr(_obj, _attr, _val)
    except AttributeError:
        pass
print("[soft-determinism] seed + pinned GEMM + tf32 off, NO use_deterministic_algorithms",
      file=sys.stderr, flush=True)

_here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(_here, "calibrate_ml8_paged.py")] + sys.argv[1:]
import runpy  # noqa: E402
runpy.run_path(sys.argv[0], run_name="__main__")
