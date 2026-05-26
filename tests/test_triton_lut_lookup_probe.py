#!/usr/bin/env python3
"""MAD-223 Phase B.2 — Triton LUT-lookup codegen probe for RDNA4 (gfx1201).

Critical risk-check before building the ml8 inner-loop LUT path: does Triton's
natural `tl.load(lut_ptr + idx)` pattern emit a fast LDS / cached-load
instruction on gfx1201, or does it fall through to slow `flat_load_ubyte`
from global memory?

If it emits `ds_read_u8` / `global_load_ubyte` with caching → use native pattern
  in Phase B.1, no inline asm needed.
If it emits `flat_load_ubyte` (uncached global) → need inline-asm fallback
  via `tl.inline_asm_elementwise` with `__builtin_amdgcn_ds_read_u8` or
  similar. Mirror Phase 0 FP8 WMMA contingency.

Test:
  1. JIT compile a minimal Triton kernel that does per-lane LUT lookup:
     out[i] = lut[idx[i]]  for a 16-entry fp8 LUT, per-lane uint8 idx.
  2. Run with known LUT (bytes 0..15) and identity indices → expected = LUT.
  3. Verify numerical correctness.
  4. Inspect the cached AMDGCN assembly for the actual load instruction emitted.

Usage:
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \\
    /home/kmbandy/venvs/agents/bin/python3 tests/test_triton_lut_lookup_probe.py
"""

import os
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


N_PER_BLOCK = 64  # per-block lane count for the gather
N_CENTROIDS = 16  # ml8-4 LUT size


@triton.jit
def lut_lookup_kernel(
    lut_ptr,        # *fp8_e4m3 — 16 entries
    idx_ptr,        # *uint8    — N indices, values in [0, 15]
    out_ptr,        # *fp8_e4m3 — N output bytes
    N: tl.constexpr,
):
    """Per-lane LUT lookup: out[i] = lut[idx[i]]. This is the EXACT pattern
    the ml8 kernel inner loop will use for centroid lookup."""
    offs = tl.arange(0, N)
    idx = tl.load(idx_ptr + offs)               # uint8 → broadcast int promotion
    idx_i32 = idx.to(tl.int32)
    val = tl.load(lut_ptr + idx_i32)            # fp8 LUT gather — THIS is what we're probing
    tl.store(out_ptr + offs, val)


def fp32_to_e4m3(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.float8_e4m3fn)


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    device = torch.device("cuda")
    print(f"# device: {torch.cuda.get_device_name(0)}", flush=True)

    # Build LUT: 16 distinct fp8 values that all round cleanly.
    # Use {1, 2, 3, ..., 16} — all E4M3-exact, all distinct after cast.
    lut_fp32 = torch.arange(1.0, float(N_CENTROIDS + 1), device=device, dtype=torch.float32)
    lut_fp8 = fp32_to_e4m3(lut_fp32)

    # Indices: cycle through 0..15 to cover the full LUT.
    idx = torch.arange(N_PER_BLOCK, device=device, dtype=torch.uint8) % N_CENTROIDS
    out = torch.zeros(N_PER_BLOCK, device=device, dtype=torch.float8_e4m3fn)

    # Clear / snapshot Triton cache
    cache_dir = Path(os.environ.get(
        "TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")
    ))
    if cache_dir.exists():
        pre_caches = set(p.name for p in cache_dir.iterdir())
    else:
        pre_caches = set()

    print(f"# launching lut_lookup_kernel N={N_PER_BLOCK} …", flush=True)
    grid = (1,)
    lut_lookup_kernel[grid](lut_fp8, idx, out, N=N_PER_BLOCK)
    torch.cuda.synchronize()

    # Correctness: out[i] should equal lut[idx[i]] = idx[i]+1.0 in fp32 terms
    expected = lut_fp32[idx.to(torch.int64)]
    actual = out.to(torch.float32)
    diff = (actual - expected).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    n_mismatch = (diff > 1e-3).sum().item()
    print(
        f"# correctness: max_err={max_err:.4g}, rms_err={rms_err:.4g}, "
        f"mismatches (>1e-3)={n_mismatch} / {N_PER_BLOCK}",
        flush=True,
    )

    # Inspect cached AMDGCN
    if cache_dir.exists():
        post_caches = list(cache_dir.iterdir())
        new_entries = [p for p in post_caches if p.name not in pre_caches]
        scan_dirs = new_entries or post_caches[-10:]
        print(f"# inspecting {len(scan_dirs)} cache dir(s) for amdgcn assembly", flush=True)

        # Instructions we're looking for, in order of "what we want"
        wanted_instr = {
            "ds_read_u8":          "LDS byte read (best — what JohnTDI's pattern hits)",
            "ds_read_b8":          "LDS byte read variant",
            "buffer_load_ubyte":   "buffer-mode global load (cached path)",
            "global_load_ubyte":   "global cached load (acceptable)",
            "flat_load_ubyte":     "uncached global load (SLOW — needs inline-asm fallback)",
            "v_perm_b32":          "byte permute (if LUT got placed in regs)",
        }
        found = {k: [] for k in wanted_instr}
        any_load = False

        for d in scan_dirs:
            if not d.is_dir():
                continue
            for f in d.rglob("*.amdgcn"):
                content = f.read_text(errors="ignore")
                if "lut_lookup_kernel" not in content:
                    continue  # not our kernel
                for instr in wanted_instr:
                    for line in content.splitlines():
                        if instr in line:
                            found[instr].append(line.strip())
                            any_load = True

        print()
        print("# AMDGCN load instructions found in lut_lookup_kernel:")
        if not any_load:
            print("  (no recognized byte-load instructions found in scanned cache dirs)")
        else:
            for instr, lines in found.items():
                if lines:
                    desc = wanted_instr[instr]
                    print(f"  • {instr} ({desc}): {len(lines)} occurrence(s)")
                    for ln in lines[:3]:
                        print(f"      {ln}")

    # Verdict
    correctness_ok = (max_err < 1e-3)
    print()
    print("=== VERDICT ===")
    print(f"  correctness:        {'PASS' if correctness_ok else 'FAIL'}")
    if not correctness_ok:
        print("  => Numerical failure. Triton LUT lookup is BROKEN — debug before kernel work.")
        return 1

    # Decision logic for instr emission
    has_ds = bool(found.get("ds_read_u8", [])) or bool(found.get("ds_read_b8", []))
    has_global_cached = bool(found.get("global_load_ubyte", [])) or bool(
        found.get("buffer_load_ubyte", [])
    )
    has_flat_uncached = bool(found.get("flat_load_ubyte", []))
    has_perm = bool(found.get("v_perm_b32", []))

    if has_ds:
        print("  => Native `tl.load` lowered to ds_read_u8 (LDS path). BEST.")
        print("     Phase B.1 uses native pattern, no inline asm needed.")
        verdict = "native_lds"
    elif has_perm and not has_flat_uncached:
        print("  => LUT placed in registers, lookup via v_perm_b32. EXCELLENT.")
        print("     Phase B.1 uses native pattern; even faster than LDS.")
        verdict = "native_reg"
    elif has_global_cached:
        print("  => Cached global load (buffer_load_ubyte / global_load_ubyte).")
        print("     Acceptable for Phase B.1; consider LDS hint for tighter inner loop.")
        verdict = "native_global_cached"
    elif has_flat_uncached:
        print("  => Uncached flat_load_ubyte. SLOW.")
        print("     Phase B.1 should add inline-asm helper for ds_read_u8.")
        verdict = "needs_inline_asm"
    else:
        print("  => No recognized byte load. May be scalarized differently.")
        print("     Inspect cached .amdgcn manually before deciding.")
        verdict = "unknown"

    print(f"  verdict_tag:        {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
