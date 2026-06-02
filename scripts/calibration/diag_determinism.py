#!/usr/bin/env python3
"""Determinism diagnostic for the ml8 GPTQ calibration path.

Enables torch.use_deterministic_algorithms(True, warn_only=True) + seeding + GEMM-determinism
env BEFORE running the real calibrate_ml8_paged driver, so PyTorch logs every op in our path
that has no deterministic implementation. Pass-through args go to the driver (use a tiny
--max-layers / --n-samples for a fast run). Run with PYTHONWARNINGS=always to see every warning.

Read the result by grepping stderr for:  "does not have a deterministic implementation"
"""
import os
import sys
import warnings

# GEMM / reduction determinism env (must be set before the first CUDA/HIP context).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("HIPBLASLT_DETERMINISTIC", "1")   # hipBLASLt analog; harmless if ignored
os.environ.setdefault("MIOPEN_DEBUG_CONV_IMPLICIT_GEMM", "0")

import torch  # noqa: E402

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
except Exception as e:
    print(f"[diag] backend flag note: {e}", file=sys.stderr)

# warn_only=True ⇒ nondeterministic ops WARN instead of raising, so we enumerate them all.
torch.use_deterministic_algorithms(True, warn_only=True)
warnings.simplefilter("always")
print("[diag] determinism enabled (warn_only=True); seed=0; running calibrate driver...",
      file=sys.stderr, flush=True)

# Hand off to the real driver in THIS process (the flags above persist on the torch singleton).
_here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(_here, "calibrate_ml8_paged.py")] + sys.argv[1:]
import runpy  # noqa: E402
runpy.run_path(sys.argv[0], run_name="__main__")
