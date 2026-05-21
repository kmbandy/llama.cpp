#!/usr/bin/env python3
"""
MAD-214 Phase 1D step 0b: Triton FP8 dot codegen probe for RDNA4 (gfx1201).

Critical risk-check before building the full turbo-FP8 attention kernel in
unified_attention.py: does Triton's `tl.dot(fp8, fp8, fp32_acc)` actually
emit `v_wmma_f32_16x16x16_fp8_fp8_w32_gfx12` on our chip?

If YES → proceed with standard `tl.dot` in the production kernel.
If NO  → fall back to the builtin intrinsic via inline-asm helpers; the
         intrinsic is verified to work on R9700 by tests/wmma_rdna4_fp8_probe.cu.

Test:
  1. JIT compile a minimal Triton kernel that does a 16×16 FP8 matmul
     into an FP32 accumulator.
  2. Run with A = identity, B = test pattern → expected output = B.
  3. Verify numerical correctness.
  4. Inspect the cached AMDGCN assembly for `v_wmma_f32_16x16x16_fp8_fp8`
     to confirm the WMMA instruction was emitted (not lowered to scalar
     dot4-style ops).

Usage:
  python3 tests/test_triton_fp8_dot_probe.py
"""

import os
import shutil
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def fp8_dot_kernel(
    a_ptr,       # *fp8_e4m3 — [M, K] row-major
    b_ptr,       # *fp8_e4m3 — [K, N] row-major
    c_ptr,       # *fp32     — [M, N] row-major
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    # Single program, single tile: load full M×K and K×N, compute M×N.
    # Constraint: M=N=K=16 (one WMMA tile).
    offs_m = tl.arange(0, M)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)

    a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])  # (M, K) fp8
    b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])  # (K, N) fp8

    # FP32 accumulator; tl.dot should pick the FP8 WMMA path given fp8 inputs
    acc = tl.zeros((M, N), dtype=tl.float32)
    acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

    tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], acc)


def fp32_to_e4m3(t: torch.Tensor) -> torch.Tensor:
    """Cast fp32 → e4m3fn (round-to-nearest-even, saturating)."""
    return t.to(torch.float8_e4m3fn)


def main():
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    device = torch.device("cuda")
    print(f"# device: {torch.cuda.get_device_name(0)}", flush=True)

    M = N = K = 16

    # A = identity, B = increasing pattern (B[i][j] = i + j*16 truncated to fp8 range)
    a_fp32 = torch.eye(M, K, dtype=torch.float32, device=device)
    b_fp32 = torch.zeros(K, N, dtype=torch.float32, device=device)
    for i in range(K):
        for j in range(N):
            # Pick values that round cleanly in E4M3 (powers of 2 / small ints work best).
            # i in [0, 16) representable; j in [0, 16) representable; both stay under E4M3 max=448.
            b_fp32[i, j] = float(i + 1)  # values 1..16, all E4M3-exact

    a_fp8 = fp32_to_e4m3(a_fp32)
    b_fp8 = fp32_to_e4m3(b_fp32)
    c_out = torch.zeros(M, N, dtype=torch.float32, device=device)

    # Clear Triton cache so we get a fresh compile (so the dumped IR is from THIS run)
    cache_dir = Path(os.environ.get("TRITON_CACHE_DIR", str(Path.home() / ".triton" / "cache")))
    if cache_dir.exists():
        # Don't nuke cache entries unrelated to this test — just remember the size delta later
        pre_caches = set(p.name for p in cache_dir.iterdir())
    else:
        pre_caches = set()

    print(f"# launching fp8_dot_kernel M=N=K={M} …", flush=True)
    grid = (1,)
    fp8_dot_kernel[grid](a_fp8, b_fp8, c_out, M=M, N=N, K=K)
    torch.cuda.synchronize()

    # Correctness check: A=I, B=row-pattern → C = B
    expected = b_fp8.to(torch.float32)
    diff = (c_out - expected).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    n_mismatch = (diff > 1e-3).sum().item()
    print(
        f"# correctness: max_err={max_err:.4g}, rms_err={rms_err:.4g}, "
        f"mismatches (>1e-3)={n_mismatch} / {M*N}",
        flush=True,
    )

    # Find the freshly-compiled cache entry
    if cache_dir.exists():
        post_caches = list(cache_dir.iterdir())
        new_entries = [p for p in post_caches if p.name not in pre_caches]
        # If no new entry, the kernel might have hit an existing cache; just scan all
        scan_dirs = new_entries or post_caches[-10:]
        print(f"# inspecting {len(scan_dirs)} cache dir(s) for amdgcn assembly", flush=True)
        found_wmma_fp8 = False
        found_wmma_any = False
        for d in scan_dirs:
            if not d.is_dir():
                continue
            for f in d.rglob("*.amdgcn"):
                content = f.read_text(errors="ignore")
                if "v_wmma_f32_16x16x16_fp8_fp8" in content:
                    found_wmma_fp8 = True
                    print(f"# FOUND v_wmma_f32_16x16x16_fp8_fp8 in {f}", flush=True)
                if "v_wmma" in content:
                    found_wmma_any = True
                    matching = [line.strip() for line in content.splitlines() if "v_wmma" in line]
                    for m in matching[:5]:
                        print(f"#   wmma op: {m}", flush=True)
        if not found_wmma_any:
            print("# WARNING: no v_wmma instructions found in any cached amdgcn — tl.dot likely scalarized.", flush=True)
    else:
        print(f"# Triton cache dir {cache_dir} does not exist; cannot inspect assembly", flush=True)
        found_wmma_fp8 = False

    # Verdict
    correctness_ok = (max_err < 1e-3)
    print("")
    print("=== VERDICT ===")
    print(f"  correctness:        {'PASS' if correctness_ok else 'FAIL'}")
    print(f"  v_wmma_*_fp8_fp8:   {'EMITTED' if found_wmma_fp8 else 'NOT FOUND'}")
    if correctness_ok and found_wmma_fp8:
        print("  => Triton tl.dot emits FP8 WMMA on gfx1201. Proceed with standard tl.dot.")
        return 0
    if correctness_ok and not found_wmma_fp8:
        print("  => Output correct but no FP8 WMMA in cached assembly.")
        print("     Either tl.dot lowered to scalar/dp4a, or cache scan missed the file.")
        print("     Fallback path needed: builtin intrinsic via inline asm.")
        return 2
    print("  => Numerical failure. Investigate before kernel work.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
