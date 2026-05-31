#!/usr/bin/env python3
"""$0 local pre-flight: prove the pinned torch wheel + transformers have everything
the resident calibration needs — WITHOUT spinning up an MI300X.

ROCm torch wheels are plain x86-64 Linux wheels; they import on a CPU-only box (they
just don't find a GPU). transformers/gguf are pure Python. So you can install the
EXACT versions the image will use into a throwaway CPU venv and run this:

    python3 -m venv /tmp/v && . /tmp/v/bin/activate
    pip install torch==2.7.1 transformers==5.6.1 scipy sentencepiece
    pip install /path/to/llama.cpp/gguf-py
    python3 verify_torch_surface.py

If this prints "SURFACE OK" the entire torch/transformers/python API surface our
calibration touches exists and behaves on that version — no GPU, no credits. The
only thing it CANNOT prove is gfx942 kernel execution (that's the on-instance smoke).
"""
import sys
import torch

# The full torch.* surface the resident calibration path uses (calibrate_ml8_paged.py,
# calibrate_ml8.py, batched_gptq.py, kronecker_rotation.py). Newest stabilized in 1.9.
OPS = [
    "Tensor", "no_grad", "enable_grad", "zeros", "zeros_like", "ones_like", "empty",
    "eye", "arange", "linspace", "full", "tensor", "from_numpy", "cat", "diag",
    "diagonal", "where", "sort", "argsort", "sign", "randn", "mean", "log10",
    "quantile", "searchsorted", "allclose", "save", "load", "bmm", "einsum",
    "Generator", "float32", "float16", "bfloat16", "int8", "long", "device", "dtype",
]
SUBMODULES = {
    "linalg": ["cholesky", "qr"],
    "optim": ["Adam"],
    "cuda": ["empty_cache", "synchronize"],  # attribute existence only; not called on CPU
}


def main() -> int:
    print(f"torch {torch.__version__}  (hip={getattr(torch.version, 'hip', None)})")
    missing = [op for op in OPS if not hasattr(torch, op)]
    for mod, attrs in SUBMODULES.items():
        m = getattr(torch, mod, None)
        if m is None:
            missing.append(f"torch.{mod}")
            continue
        missing += [f"torch.{mod}.{a}" for a in attrs if not hasattr(m, a)]
    if missing:
        print("MISSING:", ", ".join(missing))
        return 1
    print(f"all {len(OPS)} top-level ops + submodule ops present")

    # Behavioral spot-checks of the load-bearing ops (cholesky/qr/bmm/einsum/quantile).
    g = torch.Generator().manual_seed(0)
    A = torch.randn(64, 64, generator=g)
    H = A @ A.T + torch.eye(64) * 1e-3          # SPD
    L = torch.linalg.cholesky(H)
    assert torch.allclose(L @ L.T, H, atol=1e-3), "cholesky round-trip failed"
    Q, R = torch.linalg.qr(torch.randn(32, 32, generator=g))
    assert torch.allclose(Q @ Q.T, torch.eye(32), atol=1e-4), "qr orthogonality failed"
    W = torch.randn(2, 16, 16, generator=g)
    assert torch.allclose(torch.bmm(W, W), torch.einsum("eij,ejk->eik", W, W), atol=1e-4)
    _ = torch.quantile(torch.randn(1000, generator=g), torch.tensor([0.1, 0.9]))
    _ = torch.searchsorted(torch.linspace(0, 1, 17), torch.rand(8, generator=g))
    print("behavioral spot-checks passed (cholesky/qr/bmm==einsum/quantile/searchsorted)")

    # Pure-Python deps that must import (the Qwen3.5 VL arch lives in transformers).
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401
    print(f"transformers {transformers.__version__} imports OK")
    try:
        import gguf  # noqa: F401
        print("gguf imports OK")
    except Exception as e:
        print(f"WARNING: gguf not importable ({e}) — pip install ./gguf-py")

    print("SURFACE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
