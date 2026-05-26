#!/usr/bin/env python3
"""MAD-223 Phase B.5 — MoE kernel ml8 LUT branch unit test.

Validates the WEIGHT_FORMAT=1 branch of `_moe_gemm_a8w8_blockscale`
(in `ggml/src/ggml-cuda/aiter-integration/kernels/moe_op_gemm_ml8.py`)
against the same PyTorch fp32 reference used for the dense kernel.

Scope: minimal 1-expert test. Exercises:
  - Per-expert W offset (`W += expt_id * stride_w_e`)
  - Per-expert LUT base (`centroid_lut_ptr + expt_id * stride_lut_expert`)
  - Expert-routing args (GatherIndx=None identity case, ExptHist/Offs/Data)
  - WEIGHT_FORMAT=1 dequant + tl.dot + scale post-multiply

Does NOT yet exercise:
  - Multi-expert dispatch (tokens routed to different experts) — needs
    real model integration to test meaningfully; deferred to Phase E.
  - GatherIndx-based reordering — same.

Usage:
  PYTHONPATH=/home/kmbandy/GitHub/triton/python \\
    /home/kmbandy/venvs/agents/bin/python3 \\
    tests/test_ml8_kernel_moe.py
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ggml/src/ggml-cuda/aiter-integration/kernels"))
sys.path.insert(0, str(REPO_ROOT / "scripts/calibration"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

import moe_op_gemm_ml8  # noqa: E402
from ml8_to_packed import pack_indices  # noqa: E402
from test_ml8_kernel_stage1_dequant import reference_dequant_gemm  # noqa: E402


def run_moe_ml8_kernel_single_expert(
    x_fp8: torch.Tensor,           # [M, K]
    w_packed: torch.Tensor,        # [1 expert, K // 2, N]
    centroids_fp8: torch.Tensor,   # [1 expert, n_groups_k, N_CENTROIDS]
    w_scale: torch.Tensor,         # [1 expert, n_groups_k, N]
    x_scale: torch.Tensor,         # [M] per-row activation scale
    group_size: int,
    n_centroids: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Launch the MoE kernel with 1 expert + WEIGHT_FORMAT=1."""
    M, K = x_fp8.shape
    n_experts_w, K_packed, N = w_packed.shape
    assert n_experts_w == 1
    assert K_packed == K // 2
    n_groups_k = K // group_size
    device = x_fp8.device

    y = torch.empty(M, N, dtype=out_dtype, device=device)

    # MoE configuration
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_K = group_size           # == GROUP_K constraint
    GROUP_M = 1
    BLOCKSCALE_M = M               # unused (PER_ROW_X_SCALE=True)
    BLOCKSCALE_N = 1               # per-N w_scale (matches ml8 calibration)
    BLOCKSCALE_K = group_size
    XCD_SWIZZLE = 1
    SPLIT_K = 1
    N_EXPTS_ACT = 1

    # Grid: 1 expert, M_tile=1 (16/16), N_tile=1, SPLIT_K=1
    grid_m = M // BLOCK_M          # = 1
    grid_n = N // BLOCK_N          # = 1
    total_tiles = grid_m * grid_n * SPLIT_K

    # MoE routing tensors for the 1-expert identity case
    # ExptData[pid_m] = (block_id << 16) | expt_id
    expt_data = torch.zeros(grid_m, dtype=torch.int32, device=device)
    # block 0, expert 0 → 0
    ExptHist = torch.tensor([M], dtype=torch.int32, device=device)
    ExptOffs = torch.tensor([0], dtype=torch.int32, device=device)
    # ExptOffsSum=None skips the padding-aware optimization

    # Strides (in elements)
    stride_y_k = 0                 # SPLIT_K=1 → unused
    stride_y_m, stride_y_n = y.stride()
    stride_x_m, stride_x_k = x_fp8.stride()
    stride_w_e, stride_w_k, stride_w_n = w_packed.stride()  # for [E, K//2, N] uint8
    # x_scale is per-row [M] — pass as a 1D tensor via XBlockScale ptr
    stride_x_bs_m = 1
    stride_x_bs_k = 0              # single K-group, advance by 0
    # w_scale: [1 expert, n_groups_k, N]
    stride_w_bs_e, stride_w_bs_k, stride_w_bs_n = w_scale.stride()
    # centroid LUT: [1 expert, n_groups_k, 16] fp8
    stride_lut_expert = n_groups_k * n_centroids
    stride_lut_k = centroids_fp8.stride(1)

    moe_op_gemm_ml8._moe_gemm_a8w8_blockscale[(total_tiles,)](
        # Y + strides
        y, stride_y_k, stride_y_m, stride_y_n,
        # X + strides
        x_fp8, stride_x_m, stride_x_k,
        # XBlockScale (per-row x_scale here)
        x_scale, stride_x_bs_m, stride_x_bs_k,
        # W + strides
        w_packed, stride_w_e, stride_w_k, stride_w_n,
        # WBlockScale
        w_scale, stride_w_bs_e, stride_w_bs_k, stride_w_bs_n,
        # static scales (None — we use blockscale)
        None, None, None,
        # Bias (None)
        None, 0,
        # Gammas (None)
        None,
        # shapes
        N, K,
        # expert data
        None,                      # GatherIndx (identity routing)
        ExptHist, ExptOffs,
        None,                      # ExptOffsSum
        expt_data,                 # ExptData
        # grid_m, grid_n
        grid_m, grid_n,
        # activation fn
        APPLY_SWIGLU=False, alpha=0.0, limit=0.0,
        ACTIVATION_REDUCTION_N=1,
        ADD_RESIDUAL=False,
        # MoE
        N_EXPTS_ACT=N_EXPTS_ACT,
        # block sizes
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_M=GROUP_M,
        BLOCKSCALE_M=BLOCKSCALE_M, BLOCKSCALE_N=BLOCKSCALE_N,
        BLOCKSCALE_K=BLOCKSCALE_K,
        XCD_SWIZZLE=XCD_SWIZZLE,
        EVEN_K=True,
        MASK_K_LIMIT=K,
        SPLIT_K=SPLIT_K,
        # LOCAL PATCH #4 (MAD-244): W_CACHE_MODIFIER constexpr removed from
        # kernel — string-constexpr breaks Triton AOT signature parser.
        UPCAST_INDICES=False,
        PER_ROW_X_SCALE=True,
        # ml8 LUT branch
        WEIGHT_FORMAT=1,
        N_CENTROIDS=n_centroids,
        centroid_lut_ptr=centroids_fp8,
        stride_lut_expert=stride_lut_expert,
        stride_lut_k=stride_lut_k,
    )
    torch.cuda.synchronize()
    return y


def test_moe_single_expert_random_inputs():
    """1-expert MoE GEMM with WEIGHT_FORMAT=1 matches dense reference."""
    device = torch.device("cuda")
    M, N, K = 16, 16, 64
    n_centroids = 16
    group_size = 64
    n_groups_k = K // group_size  # = 1

    torch.manual_seed(42)

    x_fp32 = (torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)
    x_fp32_actual = x_fp8.to(torch.float32)

    # Single expert: centroids shape [1, n_groups_k, n_centroids]
    cent_fp32 = (torch.randn(1, n_groups_k, n_centroids, device=device) * 0.5)
    centroids_fp8 = cent_fp32.to(torch.float8_e4m3fn)
    centroids_fp32_actual = centroids_fp8.to(torch.float32)

    # Indices: [K, N] in [0, 15]
    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)
    # w_scale: [1 expert, n_groups_k, N]
    w_scale = (torch.randn(1, n_groups_k, N, device=device).abs() * 0.1 + 0.01)
    x_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    # Reference using the dense reference helper
    # (1-expert MoE collapses to dense for the one expert's tokens)
    Y_ref = reference_dequant_gemm(
        x_fp32_actual,
        indices,
        centroids_fp32_actual[0],   # [n_groups_k, n_centroids]
        w_scale[0],                  # [n_groups_k, N]
        x_scale,
        group_size,
    )

    # Pack indices: [K, N] → [N, K//2] via transpose → pack → transpose back
    indices_t = indices.T.cpu().contiguous()   # [N, K]
    packed_bytes = pack_indices(indices_t, nibble_lo_first=True)
    import numpy as np
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    w_packed_2d = torch.from_numpy(packed_np.T.copy()).contiguous().to(device)
    # Add expert dim: [1, K//2, N]
    w_packed = w_packed_2d.unsqueeze(0)

    Y_kernel = run_moe_ml8_kernel_single_expert(
        x_fp8, w_packed, centroids_fp8, w_scale, x_scale,
        group_size=group_size, n_centroids=n_centroids,
    )

    Y_ref_bf16 = Y_ref.to(torch.bfloat16).to(torch.float32)
    Y_kernel_fp32 = Y_kernel.to(torch.float32)
    diff = (Y_kernel_fp32 - Y_ref_bf16).abs()
    max_err = diff.max().item()
    rms_err = diff.pow(2).mean().sqrt().item()

    print(f"  M={M}, N={N}, K={K}, n_experts=1")
    print(f"  max_err = {max_err:.4g}")
    print(f"  rms_err = {rms_err:.4g}")
    print(f"  Y_ref    [0, :4] = {Y_ref[0, :4].cpu().tolist()}")
    print(f"  Y_kernel [0, :4] = {Y_kernel_fp32[0, :4].cpu().tolist()}")
    assert max_err < 1e-2, f"max_err {max_err:.4g} exceeds 1e-2"
    print("  ✓ 1-expert MoE kernel output matches dense reference")


def test_moe_single_expert_multi_kgroup():
    """1-expert MoE with multiple K-groups — exercises per-K-group LUT advance in MoE path."""
    device = torch.device("cuda")
    M, N, K = 16, 16, 128
    n_centroids = 16
    group_size = 64
    n_groups_k = K // group_size  # = 2

    torch.manual_seed(99)

    x_fp32 = (torch.randn(M, K, device=device) * 0.3).clamp(-1.5, 1.5)
    x_fp8 = x_fp32.to(torch.float8_e4m3fn)

    cent_fp32 = (torch.randn(1, n_groups_k, n_centroids, device=device) * 0.5)
    centroids_fp8 = cent_fp32.to(torch.float8_e4m3fn)

    indices = torch.randint(0, n_centroids, (K, N), dtype=torch.int8, device=device)
    w_scale = (torch.randn(1, n_groups_k, N, device=device).abs() * 0.1 + 0.01)
    x_scale = (torch.randn(M, device=device).abs() * 0.1 + 0.01)

    Y_ref = reference_dequant_gemm(
        x_fp8.to(torch.float32),
        indices,
        centroids_fp8[0].to(torch.float32),
        w_scale[0],
        x_scale,
        group_size,
    )

    indices_t = indices.T.cpu().contiguous()
    packed_bytes = pack_indices(indices_t, nibble_lo_first=True)
    import numpy as np
    packed_np = np.frombuffer(packed_bytes, dtype=np.uint8).reshape(N, K // 2)
    w_packed = torch.from_numpy(packed_np.T.copy()).contiguous().unsqueeze(0).to(device)

    Y_kernel = run_moe_ml8_kernel_single_expert(
        x_fp8, w_packed, centroids_fp8, w_scale, x_scale,
        group_size=group_size, n_centroids=n_centroids,
    )

    diff = (Y_kernel.to(torch.float32) - Y_ref.to(torch.bfloat16).to(torch.float32)).abs()
    max_err = diff.max().item()
    print(f"  M={M}, N={N}, K={K}, n_groups_k={n_groups_k}, K-iters={K // group_size}")
    print(f"  max_err = {max_err:.4g}")
    assert max_err < 1e-2, f"max_err {max_err:.4g} exceeds 1e-2"
    print("  ✓ multi-K-group MoE kernel output matches reference")


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    print(f"# device: {torch.cuda.get_device_name(0)}")
    print()

    print("# test_moe_single_expert_random_inputs (single K-group)")
    test_moe_single_expert_random_inputs()
    print()

    print("# test_moe_single_expert_multi_kgroup (cross-K-group advance)")
    test_moe_single_expert_multi_kgroup()

    print()
    print("=== PASS: MoE ml8 kernel verified on 1-expert single + multi K-group ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
