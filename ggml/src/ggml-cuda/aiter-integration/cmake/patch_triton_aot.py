#!/usr/bin/env python3
"""
patch_triton_aot.py — post-process Triton AOT-generated C launchers
to work around upstream Triton bugs we've reported.

Currently handles one bug:

    Triton AOT fp32 scalar truncation (surfaced 2026-05-17, MAD-188).
    Upstream `triton.tools.compile` emits the C launcher with `double NAME`
    parameters for kernel args declared `fp32` in the Triton signature.
    The launcher then packs `&NAME` into args[] for hipModuleLaunchKernel.
    The kernel reads 4 bytes (its declared fp32 size) from an 8-byte double's
    IEEE-754 representation, getting a denormal ≈ 0. Silently wrong output.

    Fix applied here: rewrite `double NAME` → `float NAME` in the launcher's
    .c and .h, for the explicit list of names passed via --fp32-scalars.
    With `float NAME`, `&NAME` is a 4-byte pointer — correctly sized for the
    kernel's fp32 read.

    NOT a generic "all double → float" rewrite: Triton signatures CAN
    legitimately contain fp64 scalars, so we patch only names the spec
    declares as fp32.

Invoked from TritonAOT.cmake's add_custom_command, after triton.tools.compile.
"""
import argparse
import re
import sys
from pathlib import Path


def patch_fp32_scalars(text: str, names: list[str]) -> tuple[str, int]:
    """Rewrite `double NAME` → `float NAME` for each name in `names`.

    Returns (patched_text, num_substitutions). num_substitutions > 0 confirms
    the patch landed on something — useful for catching name typos or upstream
    Triton changes that obviate the patch.
    """
    total = 0
    for name in names:
        # Match `double NAME` as whole-word, anywhere it appears. The Triton
        # generator emits this only in two places: the function signature in
        # the .h and the function definition in the .c.
        pattern = re.compile(rf"\bdouble\s+{re.escape(name)}\b")
        text, n = pattern.subn(f"float {name}", text)
        total += n
    return text, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="Directory containing the generated .c and .h files")
    ap.add_argument("--kernel-name", required=True,
                    help="Logical kernel name used in the file basenames")
    ap.add_argument("--fp32-scalars", default="",
                    help="Comma-separated names of fp32 scalar args to patch")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if --fp32-scalars is non-empty but zero "
                         "substitutions land — likely indicates a typo or "
                         "upstream Triton change")
    args = ap.parse_args()

    names = [n.strip() for n in args.fp32_scalars.split(",") if n.strip()]
    if not names:
        # Nothing to patch — exit cleanly so spec blocks that don't need any
        # patching don't have to omit this step.
        return 0

    targets = list(args.out_dir.glob(f"{args.kernel_name}.*.c")) + \
              list(args.out_dir.glob(f"{args.kernel_name}.*.h"))
    if not targets:
        print(f"patch_triton_aot: no {args.kernel_name}.*.{{c,h}} found in "
              f"{args.out_dir}", file=sys.stderr)
        return 1

    grand_total = 0
    for path in targets:
        original = path.read_text()
        patched, n = patch_fp32_scalars(original, names)
        if n > 0:
            path.write_text(patched)
            print(f"patch_triton_aot: {path.name} — patched {n} fp32 scalar(s): "
                  f"{', '.join(names)}")
        grand_total += n

    if args.strict and grand_total == 0:
        print(f"patch_triton_aot: strict mode — no substitutions for names "
              f"{names} in {args.out_dir}. Either the names are wrong or "
              f"upstream Triton no longer emits `double` for fp32 scalars.",
              file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
