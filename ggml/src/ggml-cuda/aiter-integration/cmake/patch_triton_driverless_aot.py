#!/usr/bin/env python3
"""Make Triton's AOT compiler (tools/compile.py) DRIVERLESS.

Why this exists
---------------
Triton 3.7.0 (the pinned 4768da5e the MI300X image builds) regressed ahead-of-time
compilation: `triton.tools.compile.compile_kernel()` touches `driver.active` in two
spots that require a *live* GPU, even when an explicit `--target hip:gfx942:64` is
passed. A rootless `podman build` container has no GPU, so the AITER AOT step dies
with `RuntimeError: 0 active drivers ([])`. AOT cross-compilation must NOT need the
target (or any) GPU — that's the entire point of compiling ahead of time.

Both accesses are gratuitous:
  1. `kernel.create_binder()` -> jit.py `driver.active.get_current_target()`. The
     binder is a *runtime-launch* helper; AOT only needs `ASTSource`. Older Triton
     called `triton.compiler.ASTSource(...)` directly, with no binder.
  2. `ty_to_cpp = driver.active.map_python_to_cpp_type`. That method body is just
     `return ty_to_cpp(ty)` (the instance `self` is unused) — a thin shim over the
     module-level `ty_to_cpp` in the AMD backend driver. Import it directly.

After patching, `python3 -m triton.tools.compile ... --target hip:gfx942:64` emits
real gfx942 HIP AOT artifacts with no GPU present (verified: emits the .c/.h pair).

This patch is AMD/HIP-specific (the image only ever AOT-compiles `hip:gfx942`), and
idempotent. If an anchor is missing AND the patched form is also absent, it exits
non-zero so a Triton version bump can't silently reintroduce the GPU dependency.

Usage:  patch_triton_driverless_aot.py [/opt/triton]
"""
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/triton")
target = root / "python" / "triton" / "tools" / "compile.py"
if not target.is_file():
    sys.exit(f"[patch_triton_driverless_aot] not found: {target}")

src = target.read_text()

# (anchor, replacement). An empty replacement means "delete the anchor line".
EDITS = [
    # 1a. drop the runtime-only binder (needs the live driver's current target)
    ("    kernel.create_binder()\n", ""),
    # 1b. ...and source the AST directly instead of via the binder's class attr
    ("src = kernel.ASTSource(", "src = triton.compiler.ASTSource("),
    # 2. pull the python->c++ type map from the module, not the live driver instance
    ("ty_to_cpp = triton.runtime.driver.active.map_python_to_cpp_type",
     "from triton.backends.amd.driver import ty_to_cpp"),
]

for anchor, repl in EDITS:
    if anchor in src:
        src = src.replace(anchor, repl)
        continue
    # idempotent: already patched?  removal -> anchor simply absent; replace -> repl present
    if repl == "" or repl in src:
        continue
    sys.exit(
        f"[patch_triton_driverless_aot] ANCHOR NOT FOUND: {anchor!r}\n"
        f"  Triton's AOT source changed — re-verify the driverless fix before trusting it."
    )

target.write_text(src)
print(f"[patch_triton_driverless_aot] patched {target} — AOT is now driverless (gfx942, no GPU)")
