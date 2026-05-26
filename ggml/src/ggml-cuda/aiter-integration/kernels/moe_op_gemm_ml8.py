# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM ROCm/aiter
#
#   Source:   aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py
#   Upstream: https://github.com/ROCm/aiter
#   Repo HEAD at vendoring time: 9c79a5b59 (2026-05-25)
#   Last commit touching this file: 24a53c6d8 (#3293 [TRITON] Moe gfx1250 optimizations)
#
# Vendored 2026-05-25 for MAD-223 Phase A — see ML8_WMMA_KERNEL_DESIGN.md.
# MoE-shape companion to gemm_ml8.py (the dense baseline). Inner-loop
# dequant pattern is identical; this file adds AITER's standard MoE
# expert-routing dispatch.
#
# License: MIT (matches upstream aiter LICENSE).
#
# ─────────────────────────────────────────────────────────────────────────
# LOCAL PATCHES APPLIED:
#
#   #1 (Phase A, 2026-05-25): Inlined AITER helpers (pid_grid,
#      _compute_static_fp8_quant, _swiglu, clip). Replaces three
#      `from aiter.ops.triton...` imports that would otherwise trigger
#      AITER's package __init__.py JIT build (module_aiter_core) on
#      import. Inlined block lives at the top of the file just after
#      the standard imports.
#
#   #2 (Phase B, planned): WEIGHT_FORMAT: tl.constexpr branch following
#      the design doc spec.
#
# ─────────────────────────────────────────────────────────────────────────

# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────
# LOCAL PATCH #1: inlined AITER helpers (replaces three aiter.ops.triton...
# imports that would otherwise trigger AITER's heavy package __init__.py
# which JIT-builds module_aiter_core on import).
#
# Sources (all MIT, copyright AMD):
#   pid_grid                  ← aiter/ops/triton/utils/_triton/pid_preprocessing.py
#   _compute_static_fp8_quant ← aiter/ops/triton/_triton_kernels/moe/quant_moe.py
#   _swiglu, clip             ← aiter/ops/triton/_triton_kernels/moe/activations.py
# ─────────────────────────────────────────────────────────────────────────


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """Maps 1D pid to 2D grid coords (pid_m, pid_n)."""
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        tl.assume(group_size_m >= 0)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _compute_static_fp8_quant(tensor, scale):
    """Quantize a tensor to FP8 E4M3 using a scalar scale."""
    tensor = tensor.to(tl.float32)
    tensor = tensor / scale
    tensor = tensor.to(tl.float8e4nv)
    return tensor


@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
    res = tl.minimum(x, limit)
    if clip_lower:
        res = tl.maximum(-limit, res)
    return res


@triton.jit
def _swiglu(input, alpha, limit, ADD_RESIDUAL: tl.constexpr):
    """SwiGLU activation: silu(gelu) * linear (+ residual if ADD_RESIDUAL).
    If alpha=1.0, equivalent to SiLU activation."""
    gelu, linear = tl.split(tl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(tl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(tl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + tl.exp2(-1.44269504089 * alpha * gelu))
    if ADD_RESIDUAL:
        return tl.fma(s, linear, s)
    else:
        return s * linear


def matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    M, N, K = None, args["N"], args["K"]
    Y, X, W = args["Y"], args["X"], args["W"]
    hist = args["ExptHist"]
    if hist is not None:
        n_rows = int(hist.float().mean())
        n_tokens = float(hist.sum())
        n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()

    def repr(s, x):
        return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"

    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} [{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    gindx = args.get("GatherIndx", None)
    if gindx is not None:
        gindx = gindx.to(torch.int32)
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"
    if args["Quant_static_scale"] is not None:
        ret["name"] += "_quant"

    fM = n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_expts_act = args["N_EXPTS_ACT"]

        if gindx is not None:
            # recreate inverse GatherIndx.
            dst = torch.full_like(gindx, -1)
            idx = torch.arange(len(gindx), device=gindx.device, dtype=torch.int32)
            mask = gindx != -1
            dst[gindx[mask]] = idx[mask]
            n_read_rows = (dst.view((-1, n_expts_act)) != -1).any(dim=1).sum()
        else:
            n_read_rows = n_tokens
        n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret


# TODO: using aiter swizzle instead can lead to perf degradation in rare cases
@triton.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: tl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@triton.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a8w8_blockscale(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XBlockScale,  # [M, K_blocks] or [M_blocks, K_blocks]
    stride_x_bs_m,
    stride_x_bs_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WBlockScale,  # [K_blocks, N_blocks]
    stride_w_bs_e,
    stride_w_bs_k,
    stride_w_bs_n,
    X_static_scale,
    W_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: tl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: tl.constexpr,
    ADD_RESIDUAL: tl.constexpr,
    # MoE config
    N_EXPTS_ACT: tl.constexpr,
    # optimization config
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    BLOCKSCALE_M: tl.constexpr,
    BLOCKSCALE_N: tl.constexpr,
    BLOCKSCALE_K: tl.constexpr,
    XCD_SWIZZLE: tl.constexpr,
    EVEN_K: tl.constexpr,
    MASK_K_LIMIT: tl.constexpr,
    SPLIT_K: tl.constexpr,
    # LOCAL PATCH #4 (MAD-244): removed `W_CACHE_MODIFIER: tl.constexpr` —
    # Triton AOT signature parser rejects string-constexpr args (KeyError on
    # the empty-string literal). All `tl.load` calls below drop the
    # `cache_modifier=W_CACHE_MODIFIER` kwarg as a consequence. Same patch
    # as gemm_ml8.py's Patch #4; required for the C++ wrapper's AOT path.
    UPCAST_INDICES: tl.constexpr = False,
    # Use per-row or 2D blockscale on X
    PER_ROW_X_SCALE: tl.constexpr = False,
    # ─── LOCAL PATCH #6 (MAD-244): explicit constexpr "feature present" flags ─
    # Triton's AOT signature parser cannot encode `None` in the signature
    # string (compile.py:105 only accepts int/float literals as constexprs).
    # Without these flags, every `if X is not None:` runtime check would
    # evaluate True at AOT time (because the signature dtype is a pointer,
    # not None), forcing the kernel to always execute the optional-feature
    # branch — which would corrupt the smoke test for Quant_static_scale and
    # waste cycles for the others. JIT path: pass each flag explicitly to
    # match the args you provide (HAS_BIAS=False ↔ B=None, etc.). AOT path:
    # wrapper bakes 0/1 into the signature.
    HAS_BIAS: tl.constexpr = False,
    HAS_GAMMAS: tl.constexpr = False,
    HAS_X_STATIC_SCALE: tl.constexpr = False,
    HAS_W_STATIC_SCALE: tl.constexpr = False,
    HAS_QUANT_STATIC_SCALE: tl.constexpr = False,
    # ─── LOCAL PATCH #2 (MAD-223 Phase B.5): ml8 LUT branch additions ─────
    # See ../ML8_WMMA_KERNEL_DESIGN.md §"The ml8 modification" — same
    # structure as gemm_ml8.py's Patch #2, applied here for the MoE shape.
    # When WEIGHT_FORMAT == 0: byte-identical to upstream MoE blockscale.
    # When WEIGHT_FORMAT == 1: W is packed 4-bit indices per expert;
    #                          centroid LUT is per-(expert, group_k).
    WEIGHT_FORMAT: tl.constexpr = 0,            # 0 = blockscale (upstream), 1 = ml8_lut
    N_CENTROIDS: tl.constexpr = 16,             # 16 for ml8-4
    centroid_lut_ptr=None,                      # *fp8_e4m3, [n_experts, n_groups_k, N_CENTROIDS]
    stride_lut_expert=0,                        # stride between experts in LUT buffer
    stride_lut_k=0,                             # stride between K-groups in LUT buffer
):
    """
    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - E: Matrix E with shape (E, K, N).
    - Y: Matrix C with shape (E, M, N).
    - x_scale: Scale tensor for A with shape (M // blockscale_m, K // blockscale_k) or (M, K // blockscale_k)
    - w_scale: Scale tensor for B with shape (K // blockscale_k, N // blockscale_n)
    - PER_ROW_X_SCALE: Determines whether we use per-row or 2D blockscale on X

    For this kernel implementation, BLOCKSCALE_K must equal BLOCK_K.
    """

    tl.assume(stride_y_k >= 0)
    tl.assume(stride_y_m >= 0)
    tl.assume(stride_y_n >= 0)
    tl.assume(stride_x_m >= 0)
    tl.assume(stride_x_k >= 0)
    tl.assume(stride_w_e >= 0)
    tl.assume(stride_w_k >= 0)
    tl.assume(stride_w_n >= 0)
    if stride_x_bs_m is not None:
        tl.assume(stride_x_bs_m >= 0)
    if stride_x_bs_k is not None:
        tl.assume(stride_x_bs_k >= 0)
    if stride_w_bs_e is not None:
        tl.assume(stride_w_bs_e >= 0)
    if stride_w_bs_k is not None:
        tl.assume(stride_w_bs_k >= 0)
    if stride_w_bs_n is not None:
        tl.assume(stride_w_bs_n >= 0)
    # LOCAL PATCH #6: was `if B is not None:`. See top-of-kernel comment.
    if HAS_BIAS:
        tl.assume(stride_b_e >= 0)
    tl.assume(grid_m >= 0)
    tl.assume(grid_n >= 0)
    tl.static_assert(
        BLOCKSCALE_K == BLOCK_K, "This kernel assumes one K-block per tile"
    )

    is_x_blockscale: tl.constexpr = XBlockScale is not None
    is_w_blockscale: tl.constexpr = WBlockScale is not None

    OUT_BLOCK_N: tl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = tl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - tl.load(ExptOffsSum)
    else:
        padding_m: tl.constexpr = 0

    index_type: tl.constexpr = tl.int64 if UPCAST_INDICES else tl.int32

    unpadded_m = grid_m - padding_m
    tl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = tl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = tl.load(ExptHist + expt_id)
    start_m = tl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # A pointers
    splitk_block_size = tl.cdiv(K, SPLIT_K)
    offs_k_scale = (pid_k * splitk_block_size) // BLOCKSCALE_K
    offs_k = tl.arange(0, BLOCK_K)
    offs_k_split = pid_k * splitk_block_size + offs_k

    offs_x_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_x_m = tl.max_contiguous(tl.multiple_of(offs_x_m % M, BLOCK_M), BLOCK_M)
    if GatherIndx is None:
        offs_x_m = start_m + offs_x_m
    else:
        GatherIndx += start_m
        # no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = tl.load(GatherIndx + offs_x_m) // N_EXPTS_ACT
    XPtrs = (
        X
        + offs_x_m.to(index_type)[:, None] * stride_x_m
        + offs_k_split.to(index_type)[None, :] * stride_x_k
    )

    # B pointers
    offs_w_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_w_n = tl.max_contiguous(
        tl.multiple_of(offs_w_n % N, BLOCK_N),
        BLOCK_N,
    )
    W += expt_id * stride_w_e
    WPtrs = W + (
        offs_k_split.to(index_type)[:, None] * stride_w_k
        + offs_w_n.to(index_type)[None, :] * stride_w_n
    )

    # ─── LOCAL PATCH #2: ml8 packed-4-bit W pointers + per-expert LUT base ─
    # Per-byte stride is K/2 (2 nibbles per byte). Triton DCEs when
    # WEIGHT_FORMAT==0. Per-expert LUT offset baked in here so the inner
    # loop only does k * stride_lut_k indexing.
    if WEIGHT_FORMAT == tl.constexpr(1):
        offs_k_packed = tl.arange(0, BLOCK_K // 2)
        offs_k_split_packed = pid_k * (splitk_block_size // 2) + offs_k_packed
        # W already advanced by expt_id * stride_w_e above, so the base is per-expert.
        W_ml8_ptrs = W + (
            offs_k_split_packed.to(index_type)[:, None] * stride_w_k
            + offs_w_n.to(index_type)[None, :] * stride_w_n
        )
        centroid_lut_ptr_expt = centroid_lut_ptr + expt_id * stride_lut_expert

    if is_x_blockscale:
        if PER_ROW_X_SCALE:
            # XScale: [M, K_blocks]
            XScalePtrs = (
                XBlockScale
                + offs_x_m.to(index_type) * stride_x_bs_m
                + offs_k_scale * stride_x_bs_k
            )
        else:
            # XScale: [M_blocks, K_blocks]
            offs_x_scale_m = offs_x_m // BLOCKSCALE_M
            XScalePtrs = (
                XBlockScale
                + offs_x_scale_m.to(index_type) * stride_x_bs_m
                + offs_k_scale * stride_x_bs_k
            )

    if is_w_blockscale:
        WBlockScale += expt_id * stride_w_bs_e
        offs_w_scale_n = offs_w_n // BLOCKSCALE_N
        # WBlockScale: [K_blocks, N_blocks]
        WScalePtrs = (
            WBlockScale + offs_k_scale * stride_w_bs_k + offs_w_scale_n * stride_w_bs_n
        )

    offs_ks_step = BLOCK_K // BLOCKSCALE_K
    num_k_iter = tl.cdiv(splitk_block_size, BLOCK_K)
    # compute output
    x_scale = tl.full((BLOCK_M,), 1.0, dtype=tl.float32)
    w_scale = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
        # X-load and scale-loads are SHARED between both WEIGHT_FORMAT paths.
        if EVEN_K:
            x = tl.load(XPtrs)
        else:
            x = tl.load(XPtrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)

        if is_x_blockscale:
            x_scale = tl.load(XScalePtrs)
            XScalePtrs += offs_ks_step * stride_x_bs_k

        if is_w_blockscale:
            w_scale = tl.load(WScalePtrs)
            WScalePtrs += offs_ks_step * stride_w_bs_k

        scale_matrix = x_scale[:, None] * w_scale[None, :]

        # ─── LOCAL PATCH #2: WEIGHT_FORMAT-branched W-load + dequant ───────
        if WEIGHT_FORMAT == tl.constexpr(0):
            # Blockscale baseline path — byte-identical to upstream.
            if EVEN_K:
                w = tl.load(WPtrs)
            else:
                w = tl.load(
                    WPtrs,
                    mask=offs_k[:, None] < K - k * BLOCK_K,
                    other=0.0,
                )
            acc += tl.dot(x, w) * scale_matrix
        else:
            # ml8 LUT path (decisions B + C + D) — same shape as dense kernel
            # in gemm_ml8.py, just with MoE naming + per-expert LUT base.
            if EVEN_K:
                w_packed = tl.load(W_ml8_ptrs)
            else:
                w_packed = tl.load(
                    W_ml8_ptrs,
                    mask=offs_k_packed[:, None] < (K - k * BLOCK_K) // 2,
                    other=0,
                )
            # Fused unpack-and-extract (decision B-Option 4):
            out_k = tl.arange(0, BLOCK_K)
            byte_row = out_k // 2
            shift = (out_k % 2) * 4
            byte_row_2d = byte_row[:, None] + tl.zeros(
                (1, BLOCK_N), dtype=tl.int32
            )
            w_byte = tl.gather(w_packed, byte_row_2d, axis=0)
            w_idx = ((w_byte >> shift[:, None]) & 0x0F).to(tl.int32)
            # LUT lookup — native cached buffer_load_u8 per Phase B.2 probe.
            # Per-expert base is centroid_lut_ptr_expt (set up before the loop).
            w_fp8 = tl.load(centroid_lut_ptr_expt + k * stride_lut_k + w_idx)
            acc += tl.dot(x, w_fp8) * scale_matrix

        XPtrs += BLOCK_K * stride_x_k
        # Path-specific W-pointer advance (Triton DCEs the unused branch).
        if WEIGHT_FORMAT == tl.constexpr(0):
            WPtrs += BLOCK_K * stride_w_k
        else:
            W_ml8_ptrs += (BLOCK_K // 2) * stride_w_k

    # scalar fp8 scale
    # LOCAL PATCH #6: was `is not None`. See top-of-kernel comment.
    if HAS_X_STATIC_SCALE:
        acc = acc * tl.load(X_static_scale)
    if HAS_W_STATIC_SCALE:
        acc = acc * tl.load(W_static_scale)
    # bias
    offs_m = BLOCK_M * block_id + tl.arange(0, BLOCK_M)
    offs_y_n = BLOCK_N * pid_n + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_y_n < N
    # LOCAL PATCH #6: was `if B is not None:`. See top-of-kernel comment.
    if HAS_BIAS:
        BPtrs = B + expt_id * stride_b_e + offs_y_n
        if pid_k == 0:
            bias = tl.load(BPtrs, mask=mask_n, other=0)
        else:
            bias = tl.full([BLOCK_N], 0, dtype=tl.float32)
        acc = acc + bias[None, :]
    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
        tl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        offs_y_n = OUT_BLOCK_N * pid_n + tl.arange(0, OUT_BLOCK_N)
        mask_n = offs_y_n < yN
    else:
        tl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc
    # LOCAL PATCH #6: was `if Gammas is not None:`. See top-of-kernel comment.
    if HAS_GAMMAS:
        gammas = tl.load(Gammas + start_m + offs_m, mask=mask_m, other=0.0)
        out *= gammas[:, None]
    # quant
    # LOCAL PATCH #6: was `if Quant_static_scale is not None:`. See top-of-kernel comment.
    if HAS_QUANT_STATIC_SCALE:
        out = _compute_static_fp8_quant(out, tl.load(Quant_static_scale))
    # write-back
    Y += start_m * stride_y_m
    offs_y_m = offs_m
    YPtrs = (
        Y
        + offs_y_m.to(index_type)[:, None] * stride_y_m
        + offs_y_n.to(index_type)[None, :] * stride_y_n
    )
    mask = mask_m[:, None] & mask_n[None, :]
    tl.store(YPtrs, out, mask=mask)
