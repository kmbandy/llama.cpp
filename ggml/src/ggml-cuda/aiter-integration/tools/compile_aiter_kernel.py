#!/usr/bin/env python3
"""
compile_aiter_kernel.py — runtime-AOT helper invoked by the C++ AITER
registry. Compiles one Triton kernel specialization and emits a self-contained
artifact pair the C++ side can consume directly:

    <out_dir>/kernel.hsaco    — raw HIP code-object (ELF binary). Loaded via
                                hipModuleLoadData.
    <out_dir>/meta.json       — kernel symbol name (passed to
                                hipModuleGetFunction), threads/block, shared
                                memory bytes, compile timing.

We intentionally DO NOT use the generated .c launcher — it has the
fp32-scalar-truncation bug and is just a convenience wrapper around the same
HIP API calls that C++ can make directly. Computing the launch grid in C++ also
lets us skip the C launcher's hardcoded grid expression.

Invocation (all values come from the C++ side via argv):
    compile_aiter_kernel.py \\
        --source unified_attention.py \\
        --kernel-name kernel_unified_attention_3d \\
        --target hip:gfx1201:32 \\
        --signature "<triton signature string>" \\
        --num-warps 4 \\
        --num-stages 1 \\
        --out-dir /path/to/cache/<cache-key>/

On success: exits 0 after writing kernel.hsaco + meta.json. On failure:
exits non-zero and writes the error to stderr.

This script can be invoked manually for debugging — see --help.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Match the byte-array declaration emitted by triton.tools.compile.
# Example: `unsigned char HSACO_NAME[21800] = { 0x7f, 0x45, 0x4c, 0x46, ... };`
_HSACO_DECL_RE = re.compile(
    r"unsigned\s+char\s+\w+\s*\[\s*(\d+)\s*\]\s*=\s*\{(.*?)\}\s*;",
    re.DOTALL,
)
# Match the kernel symbol name passed to hipModuleGetFunction.
# Example: `hipModuleGetFunction(&...., ..., "kernel_unified_attention_3d")`
_KERNEL_SYM_RE = re.compile(r'hipModuleGetFunction\([^,]+,\s*[^,]+,\s*"([^"]+)"\s*\)')
# Threads/block: hipModuleLaunchKernel(func, gX, gY, gZ, BX, BY, BZ, smem, ...)
# Triton emits BX = num_warps * warp_size, BY = 1, BZ = 1.
_LAUNCH_RE = re.compile(
    r"hipModuleLaunchKernel\([^,]+,\s*\w+,\s*\w+,\s*\w+,"
    r"\s*([^,]+),\s*(\d+),\s*(\d+),\s*(\d+)\s*,"
)


def parse_generated_c(c_text: str) -> dict:
    """Extract HSACO bytes, kernel symbol, and launch params from Triton's
    generated C launcher."""
    m_blob = _HSACO_DECL_RE.search(c_text)
    if not m_blob:
        raise RuntimeError("compile_aiter_kernel: no HSACO byte array found in generated .c")
    declared_len = int(m_blob.group(1))
    # The body is a comma-separated list of `0xHH`. Extract them as ints.
    hex_tokens = re.findall(r"0x([0-9a-fA-F]+)", m_blob.group(2))
    blob = bytes(int(t, 16) for t in hex_tokens)
    if len(blob) != declared_len:
        raise RuntimeError(
            f"compile_aiter_kernel: HSACO size mismatch (declared {declared_len}, "
            f"got {len(blob)} bytes)"
        )

    m_sym = _KERNEL_SYM_RE.search(c_text)
    if not m_sym:
        raise RuntimeError("compile_aiter_kernel: no kernel symbol found in generated .c")
    kernel_symbol = m_sym.group(1)

    m_launch = _LAUNCH_RE.search(c_text)
    if not m_launch:
        raise RuntimeError("compile_aiter_kernel: no hipModuleLaunchKernel call found")
    # block_x is `4 * 32`-style expression; eval it. block_y/z are literals.
    block_x = eval(m_launch.group(1), {"__builtins__": {}})  # noqa: S307 — trusted source
    block_y = int(m_launch.group(2))
    block_z = int(m_launch.group(3))
    smem    = int(m_launch.group(4))

    return {
        "kernel_symbol": kernel_symbol,
        "hsaco":         blob,
        "block_x":       block_x,
        "block_y":       block_y,
        "block_z":       block_z,
        "shared_mem_bytes": smem,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source",      required=True, type=Path, help="Triton .py source file")
    ap.add_argument("--kernel-name", required=True,            help="@triton.jit function name in source")
    ap.add_argument("--target",      required=True,            help="e.g. hip:gfx1201:32")
    ap.add_argument("--signature",   required=True,            help="Triton signature string")
    ap.add_argument("--num-warps",   type=int, default=4)
    ap.add_argument("--num-stages",  type=int, default=1)
    ap.add_argument("--out-dir",     required=True, type=Path, help="Output directory (will be created)")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"compile_aiter_kernel: source not found: {args.source}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="aiter-aot-") as tmp:
        tmp_path = Path(tmp)
        # The grid expression doesn't affect the HSACO content — it's only used
        # inside the generated launcher (which we discard). Pass a trivial
        # placeholder; the real grid is computed by the C++ caller at launch.
        cmd = [
            sys.executable, "-m", "triton.tools.compile",
            str(args.source),
            "--kernel-name", args.kernel_name,
            "--target",      args.target,
            "--signature",   args.signature,
            "--grid",        "1, 1, 1",
            "--num-warps",   str(args.num_warps),
            "--num-stages",  str(args.num_stages),
            "--out-name",    "kern",
            "--out-path",    str(tmp_path / "kern"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"compile_aiter_kernel: triton.tools.compile failed (rc={proc.returncode}):", file=sys.stderr)
            print(proc.stderr, file=sys.stderr)
            return 1

        # Find the generated .c file (Triton names it kern.<hash>.c)
        c_files = list(tmp_path.glob("kern.*.c"))
        if len(c_files) != 1:
            print(f"compile_aiter_kernel: expected 1 generated .c, got {len(c_files)}", file=sys.stderr)
            return 1
        parsed = parse_generated_c(c_files[0].read_text())

    compile_secs = time.monotonic() - t0

    # Write outputs atomically: stage to .tmp then rename.
    hsaco_tmp = args.out_dir / "kernel.hsaco.tmp"
    meta_tmp  = args.out_dir / "meta.json.tmp"
    hsaco_tmp.write_bytes(parsed["hsaco"])
    meta = {
        "kernel_symbol":    parsed["kernel_symbol"],
        "block_x":          parsed["block_x"],
        "block_y":          parsed["block_y"],
        "block_z":          parsed["block_z"],
        "shared_mem_bytes": parsed["shared_mem_bytes"],
        "hsaco_bytes":      len(parsed["hsaco"]),
        "compile_seconds":  round(compile_secs, 3),
        # Round-trip the spec so the C++ side can sanity-check what it loaded.
        "spec": {
            "kernel_name": args.kernel_name,
            "target":      args.target,
            "signature":   args.signature,
            "num_warps":   args.num_warps,
            "num_stages":  args.num_stages,
        },
    }
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n")
    hsaco_tmp.rename(args.out_dir / "kernel.hsaco")
    meta_tmp.rename(args.out_dir / "meta.json")

    print(f"compile_aiter_kernel: {args.kernel_name} → {args.out_dir} "
          f"({parsed['kernel_symbol']}, {len(parsed['hsaco']):,} bytes, "
          f"{compile_secs:.2f}s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
