#!/usr/bin/env bash
# Reproducible pinned Triton build for the R9700 (gfx1201) dev box.
# Idempotent: re-running is safe. Fails fast (set -e) on any step.
set -euo pipefail

TRITON_DIR="${TRITON_DIR:-$HOME/GitHub/triton}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$(awk 'NR==1{print $1}' "$HERE/TRITON_PIN.txt")"
PATCH="$HERE/../cmake/patch_triton_driverless_aot.py"

echo "== [1/5] checkout $PIN in $TRITON_DIR =="
git -C "$TRITON_DIR" fetch --quiet origin
git -C "$TRITON_DIR" checkout --quiet "$PIN"

echo "== [2/5] pre-flight: driverless AOT patch anchors present? =="
python3 "$PATCH" --check "$TRITON_DIR/python"

echo "== [3/5] apply driverless AOT patch (idempotent) =="
python3 "$PATCH" "$TRITON_DIR/python"

echo "== [4/5] rebuild editable install =="
pip install -e "$TRITON_DIR/python" --no-build-isolation

echo "== [5/5] post-build smoke: import + compile a gfx1201 kernel =="
python3 - <<'PY'
import triton, triton.language as tl, torch
print("triton", triton.__version__)
@triton.jit
def _k(xp, op, n, BLOCK: tl.constexpr):
    off = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK); m = off < n
    tl.store(op+off, tl.load(xp+off, mask=m).to(tl.float8e4nv).to(tl.float32), mask=m)
x = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
_k[(triton.cdiv(x.numel(),1024),)](x, o, x.numel(), BLOCK=1024)
torch.cuda.synchronize(); print("gfx1201 fp8 kernel OK")
PY
echo "== DONE: pinned Triton $PIN built, patched, smoke-passed =="
