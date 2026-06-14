#!/usr/bin/env bash
# Reproducible pinned Triton build for the R9700 (gfx1201) dev box.
#
# Hardened 2026-06-14 (MAD-294) after a string of editable-skew + RAM failures
# bumping 4768da5e -> 007ef1530. Each step below encodes a bug that actually bit:
#   [1] install from the REPO ROOT, not python/  — newer Triton moved setup.py /
#       pyproject.toml to the root; `pip install -e $DIR/python` errors with
#       "neither setup.py nor pyproject.toml found".
#   [2] --break-system-packages — this box's system python is PEP-668
#       externally-managed (Arch); bare `pip install -e` is refused.
#   [3] wipe build/ before rebuilding — a stale CMakeCache from the prior SHA
#       pinned TRITON_GSAN_CLANGXX at a no-longer-downloaded LLVM hash, so the
#       build died with `clang++: No such file` (exit 127). A clean reconfigure
#       re-resolves every tool path against the LLVM the new SHA actually fetched.
#   [4] RAM CAP — 15 GB host. Triton's build defaults to `cmake --build -j<all>`
#       (saw -j48), which OOM-killed the user session once (2026-05-31). We pin
#       MAX_JOBS and run the compile inside a systemd cgroup scope with a hard
#       MemoryMax so it can never take the desktop down, watched or not.
#   [5] smoke from a real .py file — Triton @jit introspects its source file and
#       rejects kernels defined on stdin/`python - <<HEREDOC`.
#
# Idempotent. Fails fast (set -e). Override knobs via env: TRITON_DIR, MAX_JOBS,
# TRITON_BUILD_MEM_MAX, TRITON_BUILD_MEM_HIGH.
set -euo pipefail

TRITON_DIR="${TRITON_DIR:-$HOME/GitHub/triton}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIN="$(awk 'NR==1{print $1}' "$HERE/TRITON_PIN.txt")"
PATCH="$HERE/../cmake/patch_triton_driverless_aot.py"

# RAM safety (15 GB host — see the 2026-05-31 OOM incident). Conservative defaults.
MAX_JOBS="${MAX_JOBS:-2}"
MEM_MAX="${TRITON_BUILD_MEM_MAX:-8G}"
MEM_HIGH="${TRITON_BUILD_MEM_HIGH:-6G}"

echo "== host RAM before build =="
free -h | awk 'NR<=2'
avail_mb="$(free -m | awk '/^Mem:/{print $7}')"
if [ "${avail_mb:-0}" -lt 4000 ]; then
  echo "ABORT: only ${avail_mb} MB available; need >= 4 GB headroom for the capped build." >&2
  echo "Close other RAM users (calibrations, browsers) and retry." >&2
  exit 1
fi
echo "(${avail_mb} MB available; build cgroup capped at MemoryMax=$MEM_MAX, MAX_JOBS=$MAX_JOBS)"

echo "== [1/6] checkout $PIN in $TRITON_DIR =="
git -C "$TRITON_DIR" fetch --quiet origin
git -C "$TRITON_DIR" checkout --quiet "$PIN"

echo "== [2/6] pre-flight: driverless AOT patch anchors present? =="
python3 "$PATCH" --check "$TRITON_DIR/python"

echo "== [3/6] apply driverless AOT patch (idempotent) =="
python3 "$PATCH" "$TRITON_DIR/python"

echo "== [4/6] wipe stale build/ (clean CMake reconfigure -> correct LLVM paths) =="
rm -rf "$TRITON_DIR/build"

echo "== [5/6] rebuild editable install (cgroup-capped) =="
# REPO ROOT (not python/); --break-system-packages for PEP-668; RAM-capped scope.
systemd-run --user --scope \
  -p MemoryHigh="$MEM_HIGH" -p MemoryMax="$MEM_MAX" \
  env MAX_JOBS="$MAX_JOBS" \
  pip install -e "$TRITON_DIR" --no-build-isolation --break-system-packages

echo "== [6/6] post-build smoke: import + compile a gfx1201 fp8 kernel (from a real file) =="
SMOKE="$(mktemp --suffix=.py)"
trap 'rm -f "$SMOKE"' EXIT
cat > "$SMOKE" <<'PY'
import triton, triton.language as tl, torch
print("triton", triton.__version__)
@triton.jit
def _k(xp, op, n, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = off < n
    tl.store(op + off, tl.load(xp + off, mask=m).to(tl.float8e4nv).to(tl.float32), mask=m)
x = torch.randn(4096, device="cuda"); o = torch.empty_like(x)
_k[(triton.cdiv(x.numel(), 1024),)](x, o, x.numel(), BLOCK=1024)
torch.cuda.synchronize(); print("gfx1201 fp8 kernel OK")
PY
python3 "$SMOKE"
echo "== DONE: pinned Triton $PIN built (RAM-capped), patched, smoke-passed =="
