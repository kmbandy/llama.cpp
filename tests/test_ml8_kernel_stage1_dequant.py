#!/usr/bin/env python3
"""MAD-223 Phase B.3 — Stage 1 dequant + GEMM unit test for the ml8 kernel.

Validates the WEIGHT_FORMAT=1 branch of `_gemm_a8w8_blockscale_kernel`
(in `ggml/src/ggml-cuda/aiter-integration/kernels/gemm_ml8.py`) against
a PyTorch fp32 reference implementation of the same formula.

What this test covers (Phase B.3 scope):
  - Kernel compiles end-to-end with WEIGHT_FORMAT=1 (Triton JIT lowers
    the LUT branch, gather, and B-pointer dual-setup successfully)
  - Nibble unpack produces correct indices (lo-first convention)
  - LUT lookup (`tl.load(centroid_lut_ptr + k * stride_lut_k + b_idx)`)
    gathers correct fp8 centroid values
  - Combined dequant + tl.dot + scale multiply matches the reference
    formula: C[m, n] = sum_k(A[m, k] * centroids[g][indices[k, n]] * b_scale[g, n])

What this test does NOT yet exercise (later stages):
  - Multi-tile cross-boundary correctness (Stage 3, B.4)
  - lane%16=column fragment-write trap on real WMMA (Stage 2 / B.4)
  - Real Cell C calibration artifact (deferred until Phase D)
  - MoE expert routing (B.5)

Tolerance: ~1e-2 absolute, ~5e-2 relative. fp8 e4m3 ≈ 2-bit mantissa so
quantization noise is non-trivial; matmul accumulation in fp32 then cast
back to bf16 output.

Usage:
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \\
    /home/kmbandy/venvs/agents/bin/python3 \\
    tests/test_ml8_kernel_stage1_dequant.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ggml/src/ggml-cuda/aiter-integration/kernels"))
sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))

import gemm_ml8  # noqa: E402
from ml8_to_packed import pack_indices  # noqa: E402


def reference_dequant_gemm(
    a_fp32: torch.Tensor,         # [M, K]
    indices: torch.Tensor,        # [K, N] int8 in [0, 15]
    centroids: torch.Tensor,      # [n_groups_k, N_CENTROIDS] fp32
    b_scale: torch.Tensor,        # [n_groups_k, N] fp32
    a_scale: torch.Tensor,        # [M] fp32
    group_size: int,
) -> torch.Tensor:
    """Reference computation matching the kernel's formula exactly.

    For each (m, n):
        W[k, n]   = centroids[k // group_size][indices[k, n]] * b_scale[k // group_size, n]
        C[m, n]   = sum_k(A[m, k] * W[k, n]) * a_scale[m]
    """
    M, K = a_fp32.shape
    _, N = indices.shape
    n_groups_k = K // group_size
    assert tuple(centroids.shape) == (n_groups_k, 16)
    assert tuple(b_scale.shape) == (n_groups_k, N)
    assert tuple(a_scale.shape) == (M,)

    # Dequantize B: W[k, n] = centroids[g][indices[k, n]] * b_scale[g, n]
    k_to_g = torch.arange(K, device=indices.device) // group_size            # [K]
    group_idx = k_to_g.unsqueeze(1).expand(K, N)                              # [K, N]
    cent_lookup = centroids[group_idx, indices.long()]                        # [K, N]
    scale_lookup = b_scale.gather(0, group_idx)                               # [K, N]
    W = cent_lookup * scale_lookup                                            # [K, N]

    # GEMM in fp32 + a_scale post-mul
    C = a_fp32 @ W                                                            # [M, N]
    C = C * a_scale.unsqueeze(1)
    return C


def run_ml8_kernel(
    a_fp8: torch.Tensor,           # [M, K] fp8 e4m3
    b_packed: torch.Tensor,        # [K // 2, N] uint8 (lo-first nibble packed)
    centroids_fp8: torch.Tensor,   # [n_groups_k, N_CENTROIDS] fp8 e4m3
    b_scale: torch.Tensor,         # [n_groups_k, N] fp32
    a_scale: torch.Tensor,         # [M] fp32
    group_size: int,
    n_centroids: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Launch the ml8 kernel with WEIGHT_FORMAT=1 and return the C tensor."""
    M, K = a_fp8.shape
    K_packed, N = b_packed.shape
    assert K_packed == K // 2
    n_groups_k = K // group_size
    device = a_fp8.device

    c = torch.empty(M, N, dtype=out_dtype, device=device)

    # Block / split-K configuration
    # NOTE: kernel constraint GROUP_K == BLOCK_SIZE_K, so BLOCK_SIZE_K = group_size
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = group_size   # = 64 for ml8-4
    GROUP_K = group_size
    GROUP_N = 1                 # per-N b_scale (matches ml8 calibration's scale_per_group: fp32[rows=N, n_groups_k])
    GROUP_SIZE_M = 1
    NUM_KSPLIT = 1
    SPLITK_BLOCK_SIZE = K
    NUM_STAGES = 1              # gfx1201 NUM_STAGES>=2 UAF per RDNA4 audit

    # Strides (in elements, not bytes — Triton handles element-size scaling)
    stride_am, stride_ak = a_fp8.stride()
    stride_bk, stride_bn = b_packed.stride()
    stride_cm, stride_cn = c.stride()
    stride_ck = 0  # NUM_KSPLIT==1
    stride_ascale_m = 1
    stride_ascale_k = 0  # single group, can be anything
    stride_bscale_k, stride_bscale_n = b_scale.stride()
    stride_lut_k = centroids_fp8.stride(0)

    grid_mn = (M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N)
    grid = (grid_mn * NUM_KSPLIT,)
    even_k = (K % BLOCK_SIZE_K == 0)

    # Direct kernel call. EVEN_K/GRID_MN now explicit (heuristics decorator
    # removed per LOCAL PATCH #3 in gemm_ml8.py for AOT compatibility).
    gemm_ml8._gemm_a8w8_blockscale_kernel[grid](
        # Pointers
        a_fp8, b_packed, c, a_scale, b_scale,
        # Dimensions
        M, N, K,
        # Strides
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_ck, stride_cm, stride_cn,
        stride_ascale_m, stride_ascale_k,
        stride_bscale_k, stride_bscale_n,
        # Meta
        GROUP_K=GROUP_K,
        GROUP_N=GROUP_N,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_KSPLIT=NUM_KSPLIT,
        SPLITK_BLOCK_SIZE=SPLITK_BLOCK_SIZE,
        EVEN_K=even_k,
        GRID_MN=grid_mn,
        num_stages=NUM_STAGES,
        # ml8 LUT branch
        WEIGHT_FORMAT=1,
        N_CENTROIDS=n_centroids,
        centroid_lut_ptr=centroids_fp8,
        stride_lut_k=stride_lut_k,
    )
    torch.cuda.synchronize()
    return c


# --- Tests ---------------------------------------------------------------

def test_single_tile_random_inputs():
    """Single-tile ml8 GEMM with random A, random indices, fp8-snapped centroids."""
    device = torch.device("cuda")
    M, N, K = 16, 16, 64
    n_centroids = 16
    group_size = 64
    n_groups_k = K // group_size  # = 1

    torch.manual_seed(42)

    # A: random fp32 → fp8 (forces representable values)
    a_fp32 = (torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    a_fp32_actual = a_fp8.to(torch.float32)  # round-trip for reference

    # Centroids: random → fp8 lattice (pre-snapped)
    cent_fp32 = (torch.randn(n_groups_k, n_centroids, device=device) * 0.5)
    centroids_fp8 = cent_fp32.to(torch.float8_e4m3fn)
    centroids_fp32_actual = centroids_fp8.to(torch.float32)

    # Indices: random in [0, 15]
    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)

    # b_scale: random positive fp32
    b_scale = (torch.randn(n_groups_k, N, device=device).abs() * 0.1 + 0.01)
    # a_scale: random positive fp32
    a_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    # Reference (fp32 throughout)
    C_ref = reference_dequant_gemm(
        a_fp32_actual, indices, centroids_fp32_actual, b_scale, a_scale, group_size
    )

    # Kernel input: pack indices to 4-bit nibbles (lo-first)
    # ml8_to_packed.pack_indices wants [rows, n_cols] with n_cols == K dim;
    # but the kernel's B is shape [K, N], so the "rows" of pack_indices's input
    # need to be the K-axis. We do indices.T → [N, K] → pack → reshape.
    # Actually pack_indices is generic — pack along last axis (n_cols).
    # We want packed B shape [K // 2, N], so we transpose, pack, transpose back.
    indices_t = indices.T.cpu().contiguous()  # [N, K] int8
    packed_bytes = pack_indices(indices_t, nibble_lo_first=True)
    # Result is [N, K // 2] uint8; transpose to [K // 2, N]
    import numpy as np
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    b_packed = torch.from_numpy(packed_np.T.copy()).contiguous().to(device)

    # Launch kernel
    C_kernel = run_ml8_kernel(
        a_fp8, b_packed, centroids_fp8, b_scale, a_scale,
        group_size=group_size, n_centroids=n_centroids,
    )

    # Compare
    C_ref_bf16 = C_ref.to(torch.bfloat16).to(torch.float32)
    C_kernel_fp32 = C_kernel.to(torch.float32)
    diff = (C_kernel_fp32 - C_ref_bf16).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    rel_err = (diff / C_ref_bf16.abs().clamp(min=1e-6)).max().item()

    print(f"  M={M}, N={N}, K={K}, group_size={group_size}")
    print(f"  max_err = {max_err:.4g}")
    print(f"  rms_err = {rms_err:.4g}")
    print(f"  rel_err = {rel_err:.4g}")
    print(f"  C_ref    [0, :4] = {C_ref[0, :4].cpu().tolist()}")
    print(f"  C_kernel [0, :4] = {C_kernel_fp32[0, :4].cpu().tolist()}")

    # Tolerance: bf16 rounding + fp32 → bf16 cast on output, plus fp8 quant noise on
    # A and centroids. ~1e-2 absolute is reasonable for a small tile.
    assert max_err < 5e-2, f"max_err {max_err:.4g} exceeds 5e-2 — likely a real bug"
    print("  ✓ kernel output matches reference within tolerance")


def test_multi_tile_cross_kgroup():
    """Multi-tile GEMM exercising cross-K-group boundaries.

    Dimensions chosen so we have MULTIPLE tiles in M, N, and K simultaneously:
      M=32 = 2 × BLOCK_M
      N=32 = 2 × BLOCK_N
      K=256 = 4 × group_size (4 K-groups, 4 K-iterations)

    Catches bugs in:
      - Cross-K-tile accumulator advance (a_ptrs += BLOCK_SIZE_K * stride_ak)
      - Cross-K-tile b_ml8_ptrs advance (BLOCK_SIZE_K // 2 stride)
      - Cross-K-group LUT pointer advance (k * stride_lut_k)
      - Cross-K-group b_scale advance (offs_ks_step * stride_bscale_k)
      - Multi-block M/N tile dispatch
    """
    device = torch.device("cuda")
    M, N, K = 32, 32, 256
    n_centroids = 16
    group_size = 64
    n_groups_k = K // group_size  # = 4

    torch.manual_seed(123)

    a_fp32 = (torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)
    a_fp32_actual = a_fp8.to(torch.float32)

    cent_fp32 = (torch.randn(n_groups_k, n_centroids, device=device) * 0.5)
    centroids_fp8 = cent_fp32.to(torch.float8_e4m3fn)
    centroids_fp32_actual = centroids_fp8.to(torch.float32)

    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)
    b_scale = (torch.randn(n_groups_k, N, device=device).abs() * 0.1 + 0.01)
    a_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    C_ref = reference_dequant_gemm(
        a_fp32_actual, indices, centroids_fp32_actual, b_scale, a_scale, group_size
    )

    # Pack indices: [K, N] → transpose to [N, K] → pack along K → [N, K//2] → transpose back
    indices_t = indices.T.cpu().contiguous()
    packed_bytes = pack_indices(indices_t, nibble_lo_first=True)
    import numpy as np
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    b_packed = torch.from_numpy(packed_np.T.copy()).contiguous().to(device)

    C_kernel = run_ml8_kernel(
        a_fp8, b_packed, centroids_fp8, b_scale, a_scale,
        group_size=group_size, n_centroids=n_centroids,
    )

    C_ref_bf16 = C_ref.to(torch.bfloat16).to(torch.float32)
    C_kernel_fp32 = C_kernel.to(torch.float32)
    diff = (C_kernel_fp32 - C_ref_bf16).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()
    rel_err = (diff / C_ref_bf16.abs().clamp(min=1e-6)).max().item()

    print(f"  M={M}, N={N}, K={K}, group_size={group_size}, n_groups_k={n_groups_k}")
    print(f"  M tiles = {M // 16}, N tiles = {N // 16}, K iters = {K // group_size}")
    print(f"  max_err = {max_err:.4g}")
    print(f"  rms_err = {rms_err:.4g}")
    print(f"  rel_err = {rel_err:.4g}")
    print(f"  C_ref    [0, :4] = {C_ref[0, :4].cpu().tolist()}")
    print(f"  C_kernel [0, :4] = {C_kernel_fp32[0, :4].cpu().tolist()}")
    print(f"  C_ref    [31, -4:] = {C_ref[31, -4:].cpu().tolist()}  (last tile)")
    print(f"  C_kernel [31, -4:] = {C_kernel_fp32[31, -4:].cpu().tolist()}  (last tile)")

    # Slightly looser tolerance — 4 K-tiles of fp32 accumulation amplifies the
    # bf16 output-cast rounding when sums get larger. Still should be small.
    assert max_err < 1e-2, f"max_err {max_err:.4g} exceeds 1e-2 — likely a real bug"
    print("  ✓ multi-tile cross-K-group kernel output matches reference")


def test_asymmetric_shape():
    """Wide/tall shape (M ≠ N) to catch any shape-symmetric bugs.

    Dimensions: M=16, N=48, K=64 (skewed, single K-iter, multiple N tiles).
    """
    device = torch.device("cuda")
    M, N, K = 16, 48, 64
    n_centroids = 16
    group_size = 64

    torch.manual_seed(7)
    a_fp32 = (torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)
    a_fp8 = a_fp32.to(torch.float8_e4m3fn)

    cent_fp32 = (torch.randn(1, n_centroids, device=device) * 0.5)
    centroids_fp8 = cent_fp32.to(torch.float8_e4m3fn)

    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)
    b_scale = (torch.randn(1, N, device=device).abs() * 0.1 + 0.01)
    a_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    C_ref = reference_dequant_gemm(
        a_fp8.to(torch.float32), indices,
        centroids_fp8.to(torch.float32), b_scale, a_scale, group_size,
    )

    indices_t = indices.T.cpu().contiguous()
    packed_bytes = pack_indices(indices_t, nibble_lo_first=True)
    import numpy as np
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    b_packed = torch.from_numpy(packed_np.T.copy()).contiguous().to(device)

    C_kernel = run_ml8_kernel(
        a_fp8, b_packed, centroids_fp8, b_scale, a_scale,
        group_size=group_size, n_centroids=n_centroids,
    )

    diff = (C_kernel.to(torch.float32) - C_ref.to(torch.bfloat16).to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  M={M}, N={N}, K={K}")
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 1e-2, f"max_err {max_err:.4g} exceeds 1e-2"
    print("  ✓ asymmetric shape (M≠N) kernel output matches reference")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print()

    print("# test_single_tile_random_inputs (Stage 1: single-tile dequant + GEMM)")
    test_single_tile_random_inputs()
    print()

    print("# test_multi_tile_cross_kgroup (Stage 2/3: multi-tile + cross-K-group)")
    test_multi_tile_cross_kgroup()
    print()

    print("# test_asymmetric_shape (M ≠ N, multi-N-tile)")
    test_asymmetric_shape()

    print()
    print("=== PASS: ml8 kernel verified on single-tile, multi-tile, and asymmetric shapes ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
