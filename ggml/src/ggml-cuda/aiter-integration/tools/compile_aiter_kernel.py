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

FP8_B128 phase 2: this used to shell out to `python -m triton.tools.compile`
and scrape its generated .c launcher for the HSACO bytes + kernel symbol +
launch params. triton.tools.compile (in Triton 3.8) has no --waves-per-eu /
--matrix-instr-nonkdim flags, which the radiance preshuffle GEMM config
needs (waves_per_eu=2, matrix_instr_nonkdim=16) — and it has no reasonable
way to add them short of forking the script. So this now calls
triton.compile(ASTSource, target, options={...}) directly, replicating
compile.py's compile_kernel() logic (same signature/hint parsing, same
backend.parse_options() call) but reading metadata off the CompiledKernel
object instead of round-tripping through a generated C file. The .hsaco /
meta.json OUTPUT CONTRACT is unchanged — the C++ side (aiter_runtime_compiler.cpp)
does not need to change.

Invocation (all values come from the C++ side via argv):
    compile_aiter_kernel.py \\
        --source unified_attention.py \\
        --kernel-name kernel_unified_attention_3d \\
        --target hip:gfx1201:32 \\
        --signature "<triton signature string>" \\
        --num-warps 4 \\
        --num-stages 1 \\
        --waves-per-eu 0 \\
        --matrix-instr-nonkdim 0 \\
        --out-dir /path/to/cache/<cache-key>/

On success: exits 0 after writing kernel.hsaco + meta.json. On failure:
exits non-zero and writes the error to stderr.

This script can be invoked manually for debugging — see --help.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path


def _constexpr(s: str):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return None


def compile_kernel(args) -> dict:
    """Compile one Triton kernel specialization via the public
    triton.compile(ASTSource, ...) API (same code path
    triton/tools/compile.py's compile_kernel() uses internally, minus the
    generated-C-launcher step we don't need)."""
    import triton
    import triton.backends

    arg_path = args.source
    sys.path.insert(0, str(arg_path.parent))
    spec = importlib.util.spec_from_file_location(arg_path.stem, arg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kernel = getattr(mod, args.kernel_name)

    signature = [s.strip(" ") for s in args.signature.split(",")]

    hints = {(i,): _constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {kernel.arg_names[i]: _constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    for key, value in hints.items():
        if value == 1:
            constants[kernel.arg_names[key[0]]] = value
    sig_map = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}
    for key in constants:
        sig_map[key] = "constexpr"

    for h in hints.values():
        if h not in (1, 16):
            raise RuntimeError(f"compile_aiter_kernel: only divisibility hints 1/16 are supported, got {h}")
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}

    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=sig_map, attrs=attrs)

    target_parts = args.target.split(":")
    if len(target_parts) != 3:
        raise RuntimeError(f"compile_aiter_kernel: --target must be '<backend>:<arch>:<warp-size>', got {args.target!r}")
    target = triton.backends.compiler.GPUTarget(target_parts[0], target_parts[1], int(target_parts[2]))
    backend = triton.compiler.make_backend(target)

    opt_kwargs = {"num_warps": args.num_warps, "num_stages": args.num_stages}
    if args.waves_per_eu:
        opt_kwargs["waves_per_eu"] = args.waves_per_eu
    if args.matrix_instr_nonkdim:
        opt_kwargs["matrix_instr_nonkdim"] = args.matrix_instr_nonkdim
    options = backend.parse_options(opt_kwargs)

    ccinfo = triton.compile(src, target=target, options=options.__dict__)

    if getattr(ccinfo.metadata, "global_scratch_size", 0) > 0:
        raise RuntimeError("compile_aiter_kernel: kernels with global scratch requirements are not supported")
    if getattr(ccinfo.metadata, "profile_scratch_size", 0) > 0:
        raise RuntimeError("compile_aiter_kernel: kernels with profile scratch requirements are not supported")

    hsaco = ccinfo.asm[backend.binary_ext]
    # metadata.name is the ACTUAL symbol embedded in the ELF (driven by the
    # kernel's `repr=` if any); our C++ side looks this symbol up via
    # hipModuleGetFunction, so use it directly rather than assuming it
    # equals args.kernel_name (only true when repr's config_keys are empty
    # — see gemm_ml8.py LOCAL PATCH #5/#6 for why that's kept true here).
    kernel_symbol = getattr(ccinfo.metadata, "name", args.kernel_name)

    return {
        "kernel_symbol":    kernel_symbol,
        "hsaco":            hsaco,
        "block_x":          args.num_warps * target.warp_size,
        "block_y":          1,
        "block_z":          1,
        "shared_mem_bytes": int(ccinfo.metadata.shared),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source",      required=True, type=Path, help="Triton .py source file")
    ap.add_argument("--kernel-name", required=True,            help="@triton.jit function name in source")
    ap.add_argument("--target",      required=True,            help="e.g. hip:gfx1201:32")
    ap.add_argument("--signature",   required=True,            help="Triton signature string")
    ap.add_argument("--num-warps",   type=int, default=4)
    ap.add_argument("--num-stages",  type=int, default=1)
    ap.add_argument("--waves-per-eu",          type=int, default=0, help="AMDGPU waves-per-EU hint (0 = Triton default)")
    ap.add_argument("--matrix-instr-nonkdim",  type=int, default=0, help="MFMA/WMMA K-dim override (0 = Triton default)")
    ap.add_argument("--out-dir",     required=True, type=Path, help="Output directory (will be created)")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"compile_aiter_kernel: source not found: {args.source}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        parsed = compile_kernel(args)
    except Exception as exc:  # noqa: BLE001 — surface any compile failure to the C++ caller
        print(f"compile_aiter_kernel: compile failed for {args.kernel_name}: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1
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
            "kernel_name":           args.kernel_name,
            "target":                args.target,
            "signature":             args.signature,
            "num_warps":             args.num_warps,
            "num_stages":            args.num_stages,
            "waves_per_eu":          args.waves_per_eu,
            "matrix_instr_nonkdim":  args.matrix_instr_nonkdim,
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
