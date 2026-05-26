#!/usr/bin/env python3
"""MAD-223 Phase A vendor smoke — `tests/test_ml8_vendor_smoke.py`.

Verifies the vendored ml8 kernels parse and Triton's @jit decoration applies:

  - `kernels/gemm_ml8.py`         (vendored from AITER gemm_a8w8_blockscale.py)
  - `kernels/moe_op_gemm_ml8.py`  (vendored from AITER moe_op_gemm_a8w8_blockscale.py)

This catches:
  - Broken inlined helpers (LOCAL PATCH #1 in each file)
  - Stale stub for get_gemm_config
  - Accidental re-introduction of `from aiter.ops...` imports (which would
    trigger AITER's package __init__.py JIT build of module_aiter_core
    and fail on systems without psutil / ninja / etc.)
  - Triton API drift breaking the @triton.heuristics / @triton.jit decorators

This does NOT verify:
  - Kernel correctness (Phase B)
  - GPU launch / autotune (Phase B/C)
  - Bit-equivalence to direct AITER call (Phase A bit-identical smoke, deferred
    — AITER's package init is too heavy to import alongside our vendored copies)

Usage:
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \\
    /home/kmbandy/venvs/agents/bin/python3 tests/test_ml8_vendor_smoke.py

Re-run after any of:
  - Re-vendoring (`git pull` AITER + re-copy the source files)
  - Triton upgrade
  - Edits to LOCAL PATCH #1 (the inlined helper block)

See: ggml/src/ggml-cuda/aiter-integration/ML8_WMMA_KERNEL_DESIGN.md
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNELS_DIR = REPO_ROOT / "ggml/src/ggml-cuda/aiter-integration/kernels"

if not KERNELS_DIR.is_dir():
    print(f"ERROR: kernels dir not found: {KERNELS_DIR}", file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, str(KERNELS_DIR))


def main() -> int:
    print("# importing vendored gemm_ml8 (dense baseline)...")
    import gemm_ml8
    print(f"  ✓ module loaded: {gemm_ml8.__file__}")
    print(f"  ✓ kernel:            {gemm_ml8._gemm_a8w8_blockscale_kernel}")
    print(f"  ✓ preshuffle kernel: {gemm_ml8._gemm_a8w8_blockscale_preshuffle_kernel}")

    # Sanity-check the get_gemm_config stub exists and returns a dict.
    cfg, is_tuned = gemm_ml8.get_gemm_config("GEMM-A8W8-BLOCKSCALE", 16, 64, 128)
    assert isinstance(cfg, dict) and "BLOCK_SIZE_M" in cfg, "get_gemm_config stub broken"
    assert is_tuned is False, "stub should return is_tuned=False"
    assert cfg["NUM_STAGES"] == 1, "RDNA4 audit §2.2 requires NUM_STAGES=1"

    print()
    print("# importing vendored moe_op_gemm_ml8 (MoE baseline)...")
    import moe_op_gemm_ml8
    print(f"  ✓ module loaded: {moe_op_gemm_ml8.__file__}")
    print(f"  ✓ kernel:        {moe_op_gemm_ml8._moe_gemm_a8w8_blockscale}")

    # Sanity-check inlined helpers are present in MoE file.
    for helper in ("pid_grid", "_compute_static_fp8_quant", "_swiglu", "clip"):
        assert hasattr(moe_op_gemm_ml8, helper), f"inlined helper missing: {helper}"

    # Guard against accidental re-introduction of `from aiter.ops...` imports
    for path in (KERNELS_DIR / "gemm_ml8.py", KERNELS_DIR / "moe_op_gemm_ml8.py"):
        text = path.read_text()
        bad_imports = [
            line for line in text.splitlines()
            if line.startswith("from aiter.") or line.startswith("import aiter")
        ]
        assert not bad_imports, (
            f"{path.name} re-introduced AITER package import (would trigger "
            f"module_aiter_core JIT build on every load): {bad_imports}"
        )

    print()
    print("=== PASS: vendored kernels parse, decorators apply, no AITER imports ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
