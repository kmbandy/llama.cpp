# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM ROCm/aiter
#
#   Source:   aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py
#   Upstream: https://github.com/ROCm/aiter
#   Repo HEAD at vendoring time: 9c79a5b59 (2026-05-25)
#   Last commit touching this file: 8ad75337e (#3230 [TRITON] Make splitk reduce common)
#
# Vendored 2026-05-25 for MAD-223 Phase A — see ML8_WMMA_KERNEL_DESIGN.md.
# This is the explicit-dequant + plain-tl.dot baseline for ml8-4 weight
# inference. ml8 LUT branch will be added in Phase B as a tl.constexpr
# WEIGHT_FORMAT switch following the design doc spec.
#
# License: MIT (matches upstream aiter LICENSE — see
#          https://github.com/ROCm/aiter/blob/main/LICENSE).
#
# ─────────────────────────────────────────────────────────────────────────
# LOCAL PATCHES APPLIED:
#
#   #1 (Phase A, 2026-05-25): Inlined AITER helpers (pid_grid, remap_xcd,
#      make_kernel_repr) and stubbed get_gemm_config. Replaces three
#      `from aiter.ops.triton.utils...` imports that would otherwise
#      trigger AITER's package __init__.py JIT build (module_aiter_core)
#      on import. Inlined block lives at the top of the file just after
#      the standard imports. Re-vendor: copy upstream kernel body in,
#      re-apply this inlined block, keep get_gemm_config stub until
#      Phase F autotune work emits real gfx1201 configs.
#
#   #2 (Phase B, planned): WEIGHT_FORMAT: tl.constexpr branch around
#      the B-load + post-tl.dot dequant block, per
#      ML8_WMMA_KERNEL_DESIGN.md §"The ml8 modification".
#
# ─────────────────────────────────────────────────────────────────────────

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import os

import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────
# LOCAL PATCH #1: inlined AITER helpers (replaces three aiter.ops.triton.utils
# imports that would otherwise trigger AITER's heavy package __init__.py
# which JIT-builds module_aiter_core on import).
#
# Sources (all MIT, copyright AMD):
#   pid_grid, remap_xcd  ← aiter/ops/triton/utils/_triton/pid_preprocessing.py
#   make_kernel_repr     ← aiter/ops/triton/utils/_triton/kernel_repr.py
#   get_gemm_config      ← STUBBED — Phase A permissive defaults; real
#                          gfx1201 tuning configs deferred to Phase F.
# ─────────────────────────────────────────────────────────────────────────


@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    """XCD swizzle for CDNA multi-die scheduling. No-op-ish on RDNA4 (one die)
    but kept faithful to upstream so autotune configs that set NUM_XCDS still
    behave identically."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


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


def _sanitize_constexpr_value(value):
    if value is None:
        return "NONE"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, (list, tuple, set)):
        items = sorted(value, key=str) if isinstance(value, set) else value
        sanitized_items = [_sanitize_constexpr_value(item) for item in items]
        joined = "_".join(sanitized_items)
        return joined if joined else "NONE"
    if isinstance(value, str):
        cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")
        return cleaned_value.upper() if cleaned_value else "NONE"
    cleaned_value = "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")
    return cleaned_value.upper() if cleaned_value else "NONE"


def make_kernel_repr(base_name, config_keys, name_key=None):
    """Build a kernel-name string from a specialization's constexpr values
    (used by Triton for telemetry / cache key dedup)."""
    def _repr(specialization):
        constants = specialization.constants
        name = base_name
        if name_key is not None:
            override = constants.get(name_key, None)
            if override:
                cleaned = "".join(
                    ch if ch.isalnum() or ch == "_" else "_" for ch in str(override)
                )
                if cleaned:
                    name = cleaned
        name_parts = []
        for key in config_keys:
            value = constants.get(key, None)
            symbol = _sanitize_constexpr_value(value)
            name_parts.append(f"{key}_{symbol}")
        if not name_parts:
            return name
        suffix = "_".join(name_parts)
        return f"{name}_{suffix}"
    return _repr


# MAD-223 G.6.b: tuned configs from gemm_ml8_tune.json (sweep emitted by
# scripts/calibration/tune_gemm_ml8.py). Keyed by (K, N, M_tier) where
# M_tier == "decode" if requested M <= 16 else "prefill". Fall back to
# Phase-A defaults on miss.
_TUNE_JSON_PATH = os.path.join(os.path.dirname(__file__), "gemm_ml8_tune.json")
_TUNED_LOOKUP: dict = {}
if os.path.exists(_TUNE_JSON_PATH):
    with open(_TUNE_JSON_PATH) as _f:
        _raw = json.load(_f)
    for _entry in _raw.values():
        _shape = _entry["shape"]
        _M = _shape["M"]
        _tier = "decode" if _M <= 16 else "prefill"
        _key = (_shape["K"], _shape["N"], _tier)
        _TUNED_LOOKUP[_key] = _entry["best"]

_PHASE_A_DEFAULTS = {
    "BLOCK_SIZE_M": 32,
    "BLOCK_SIZE_N": 64,
    "BLOCK_SIZE_K": 128,
    "GROUP_SIZE_M": 8,
    "NUM_KSPLIT": 1,
    "WAVES_PER_EU": 0,
    "MATRIX_INSTR_NONKDIM": 16,
    "KPACK": 1,
    "NUM_WARPS": 4,
    "NUM_STAGES": 1,  # gfx1201 num_stages>=2 UAF per RDNA4 audit §2.2
    "cache_modifier": "",
}


def get_gemm_config(kernel_name: str, M: int, N: int, K: int):
    """Look up tuned config in _TUNED_LOOKUP; fall back to Phase-A defaults.

    Returns:
        (config_dict, is_tuned)
    """
    tier = "decode" if M <= 16 else "prefill"
    tuned = _TUNED_LOOKUP.get((K, N, tier))
    if tuned is None:
        return dict(_PHASE_A_DEFAULTS), False
    cfg = dict(_PHASE_A_DEFAULTS)
    # Overlay the swept knobs onto the defaults so untouched constants
    # (BLOCK_SIZE_K, NUM_KSPLIT, MATRIX_INSTR_NONKDIM, NUM_STAGES, etc.)
    # stay at the safe Phase-A values.
    for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "GROUP_SIZE_M", "NUM_WARPS", "KPACK"):
        if k in tuned:
            cfg[k] = tuned[k]
    return cfg, True

# LOCAL PATCH #5 (MAD-223 Phase C.2, 2026-05-26): config_keys emptied.
#
# Reason: Triton's generated .c file references the kernel via the base name
# (`_gemm_a8w8_blockscale_kernel`) in its `hipModuleGetFunction(...)` call,
# but the HSACO it produces has the FULL repr-expanded symbol when
# `@triton.jit(repr=...)` is set. compile_aiter_kernel.py parses the base
# name from the .c — so the lookup mismatches the actual binary symbol.
# Empty config_keys → repr returns just the base name → HSACO symbol matches
# what's emitted in the .c file. Triton's name disambiguation across
# specializations is replaced by the runtime registry's full cache_key.
_gemm_a8w8_blockscale_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_kernel",
    [],  # was: [GROUP_K, GROUP_N, BLOCK_SIZE_M, ...] — pre-Patch #5 list
)


# LOCAL PATCH #3 (MAD-223 Phase C.2, 2026-05-26): @triton.heuristics removed.
# Reason: triton.tools.compile (AOT path) cannot access create_binder on a
# Heuristics-wrapped function. Computing EVEN_K and GRID_MN at every call
# site is trivial — they're derivable from M/N/K/BLOCK_*. ml8_runtime.ml8_gemm
# and tests now pass them as explicit constexpr kwargs.
@triton.jit(repr=_gemm_a8w8_blockscale_repr)
def _gemm_a8w8_blockscale_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    # LOCAL PATCH #4 (MAD-223 Phase C.2, 2026-05-26): `cache_modifier` arg
    # removed — Triton's AOT signature parser rejects string constexpr literals
    # ("KeyError: '\"\"'"). All call sites previously passed "" (the tl.load
    # default), so removing the arg + replacing usage with bare tl.load is
    # functionally equivalent. If non-default cache hints are wanted in the
    # future, refactor to an int-mode constexpr (0=default, 1=cs, 2=ca) and
    # branch in the kernel body. Same patch applied at b_ptrs / b_ml8_ptrs
    # tl.load call sites below.
    num_stages: tl.constexpr,
    # ─── LOCAL PATCH #2 (MAD-223 Phase B.1): ml8 LUT branch additions ──────
    # See ../ML8_WMMA_KERNEL_DESIGN.md §"The ml8 modification".
    # When WEIGHT_FORMAT == 0: byte-identical to upstream blockscale path.
    # When WEIGHT_FORMAT == 1: B is packed 4-bit indices; centroid LUT lookup
    #                          replaces 8-bit B load. b_scale handling shared.
    WEIGHT_FORMAT: tl.constexpr,         # 0 = blockscale (upstream), 1 = ml8_lut
    N_CENTROIDS: tl.constexpr,           # 16 for ml8-4 (ignored when WEIGHT_FORMAT==0)
    centroid_lut_ptr,                    # *fp8_e4m3, shape [n_groups_k, N_CENTROIDS]
    stride_lut_k,                        # stride (in elements) between adjacent K-groups in LUT
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.
    With WEIGHT_FORMAT=1, computes the ml8-4 matmul: B is packed 4-bit indices into
    a per-K-group fp8 centroid LUT; per-group scale multiplies the dot output exactly
    as in the blockscale baseline (NOT absorbed into centroids).

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B (WEIGHT_FORMAT=0): Matrix B with shape (K, N), 8-bit (fp8 or int8).
    - B (WEIGHT_FORMAT=1): Packed B with shape (K/2, N), uint8 nibbles (lo-first).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n). SHARED across paths.
    - centroid_lut: (WEIGHT_FORMAT=1 only) per-K-group fp8 LUT, shape (*scale_k, N_CENTROIDS).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N

    For this kernel implementation, GROUP_K must equal BLOCK_K.
    For WEIGHT_FORMAT=1, calibration constraint requires group_size == BLOCK_SIZE_K
    (one LUT per K-tile iter).
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        # SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        # ^ Number of K blocks within our split-K partition

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_k_split[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        )

        # ─── LOCAL PATCH #2: ml8 packed-4-bit B pointers (decision A) ──────
        # Per-byte stride is K/2 (2 nibbles per byte). Triton DCE removes
        # this when WEIGHT_FORMAT==0. See gemm_a8wfp4.py:164-170 for the
        # packed-byte stride reference pattern.
        if WEIGHT_FORMAT == tl.constexpr(1):
            offs_k_packed = tl.arange(0, BLOCK_SIZE_K // 2)
            offs_k_split_packed = pid_k * (SPLITK_BLOCK_SIZE // 2) + offs_k_packed
            b_ml8_ptrs = b_ptr + (
                offs_k_split_packed[:, None] * stride_bk
                + offs_bn[None, :] * stride_bn
            )

        # Create pointers for the scales
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_bn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in tl.range(
            pid_k * num_k_iter, (pid_k + 1) * num_k_iter, num_stages=num_stages
        ):
            # Load A and scales — SHARED between both paths.
            if EVEN_K:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # ─── LOCAL PATCH #2: WEIGHT_FORMAT-branched B-load + dequant ──
            if WEIGHT_FORMAT == tl.constexpr(0):
                # Blockscale baseline path — byte-identical to upstream.
                if EVEN_K:
                    b = tl.load(b_ptrs)  # LOCAL PATCH #4: cache_modifier removed (AOT compat)
                else:
                    b = tl.load(
                        b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                    )
                # Perform dot operation and apply scale
                accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
            else:
                # ml8 LUT path (decisions B + C + D).
                # B is packed 4-bit nibbles (lo-first); centroid LUT is fp8.
                if EVEN_K:
                    b_packed = tl.load(b_ml8_ptrs)  # LOCAL PATCH #4: cache_modifier removed
                else:
                    b_packed = tl.load(
                        b_ml8_ptrs,
                        mask=offs_k_packed[:, None] < (K - k * BLOCK_SIZE_K) // 2,
                        other=0,
                    )
                # Fused unpack-and-extract (decision B-Option 4):
                #   byte_row[i] = i // 2   (which packed byte holds K-position i)
                #   shift[i]    = (i % 2) * 4  (low nibble for even K, high for odd)
                out_k = tl.arange(0, BLOCK_SIZE_K)
                byte_row = out_k // 2
                shift = (out_k % 2) * 4
                # Expand b_packed by byte_row across N. The +tl.zeros broadcast
                # makes byte_row_2d match output shape, as required by tl.gather.
                byte_row_2d = byte_row[:, None] + tl.zeros(
                    (1, BLOCK_SIZE_N), dtype=tl.int32
                )
                b_byte = tl.gather(b_packed, byte_row_2d, axis=0)
                b_idx = ((b_byte >> shift[:, None]) & 0x0F).to(tl.int32)
                # LUT lookup — native cached buffer_load_u8 per Phase B.2 probe.
                # Per-K-group LUT (decision D); k iteration index == group_k index
                # (BLOCK_SIZE_K == GROUP_K per kernel constraint).
                b_fp8 = tl.load(
                    centroid_lut_ptr + k * stride_lut_k + b_idx
                )
                # Same scale post-multiply shape as blockscale (decision A.4 corrected).
                accumulator += tl.dot(a, b_fp8) * a_scale[:, None] * b_scale[None, :]

            # Advance the ptrs to the next K block (SHARED for A + scales).
            a_ptrs += BLOCK_SIZE_K * stride_ak
            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

            # Path-specific B-pointer advance (Triton DCEs the unused branch).
            if WEIGHT_FORMAT == tl.constexpr(0):
                b_ptrs += BLOCK_SIZE_K * stride_bk
            else:
                b_ml8_ptrs += (BLOCK_SIZE_K // 2) * stride_bk

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


_gemm_a8w8_blockscale_preshuffle_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_preshuffle_kernel",
    [
        "GROUP_K",
        "GROUP_N",
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "NUM_KSPLIT",
        "SPLITK_BLOCK_SIZE",
        "EVEN_K",
        "GRID_MN",
        "cache_modifier",
    ],
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit(repr=_gemm_a8w8_blockscale_preshuffle_repr)
def _gemm_a8w8_blockscale_preshuffle_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_KSPLIT: tl.constexpr,
    SPLITK_BLOCK_SIZE: tl.constexpr,
    EVEN_K: tl.constexpr,
    GRID_MN: tl.constexpr,
    cache_modifier: tl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N

    For this kernel implementation, GROUP_K must equal BLOCK_K.
    """

    tl.assume(stride_am > 0)
    tl.assume(stride_ak > 0)
    tl.assume(stride_bk > 0)
    tl.assume(stride_bn > 0)
    tl.assume(stride_ck > 0)
    tl.assume(stride_cm > 0)
    tl.assume(stride_cn > 0)
    tl.assume(stride_ascale_m > 0)
    tl.assume(stride_ascale_k > 0)
    tl.assume(stride_bscale_k > 0)
    tl.assume(stride_bscale_n > 0)

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid_unified = tl.program_id(axis=0)
    pid_k = pid_unified % NUM_KSPLIT
    pid = pid_unified // NUM_KSPLIT
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        remap_xcd(pid, GRID_MN)

        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    tl.assume(pid_m >= 0)
    tl.assume(pid_n >= 0)
    tl.assume(pid_k >= 0)

    if (pid_k * SPLITK_BLOCK_SIZE) < K:

        # SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        num_k_iter = tl.cdiv(SPLITK_BLOCK_SIZE, BLOCK_SIZE_K)
        # ^ Number of K blocks within our split-K partition

        # Create pointers for first block of A and B input matrices
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        offs_k_shuffle_arr = tl.arange(0, BLOCK_SIZE_K * 16)
        offs_k_split = pid_k * SPLITK_BLOCK_SIZE + offs_k
        offs_k_shuffle = pid_k * SPLITK_BLOCK_SIZE * 16 + offs_k_shuffle_arr

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * (BLOCK_SIZE_N // 16) + tl.arange(0, BLOCK_SIZE_N // 16)) % (
            N // 16
        )
        a_ptrs = a_ptr + (
            offs_am[:, None] * stride_am + offs_k_split[None, :] * stride_ak
        )
        b_ptrs = b_ptr + (
            offs_bn[:, None] * stride_bn + offs_k_shuffle[None, :] * stride_bk
        )

        # Create pointers for the scales
        offs_bsn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k_scale = (pid_k * SPLITK_BLOCK_SIZE) // GROUP_K
        a_scale_ptrs = (
            a_scale_ptr + offs_am * stride_ascale_m + offs_k_scale * stride_ascale_k
        )
        offs_b_scale_n = offs_bsn // GROUP_N
        b_scale_ptrs = (
            b_scale_ptr
            + offs_k_scale * stride_bscale_k
            + offs_b_scale_n * stride_bscale_n
        )
        offs_ks_step = BLOCK_SIZE_K // GROUP_K

        acc_dtype = tl.float32 if c_ptr.type.element_ty != tl.int8 else tl.int32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype)

        for k in range(pid_k * num_k_iter, (pid_k + 1) * num_k_iter):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            if EVEN_K:
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, cache_modifier=cache_modifier)
            else:
                a = tl.load(
                    a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0
                )
                b = tl.load(
                    b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0
                )

            b = (
                b.reshape(
                    1,
                    BLOCK_SIZE_N // 16,
                    BLOCK_SIZE_K // 32,
                    2,
                    16,
                    16,
                )
                .permute(0, 1, 4, 2, 3, 5)
                .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
                .trans(1, 0)
            )

            a_scale = tl.load(a_scale_ptrs)
            b_scale = tl.load(b_scale_ptrs)

            # Perform dot operation and apply scale
            accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]

            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * 16 * stride_bk

            a_scale_ptrs += offs_ks_step * stride_ascale_k
            b_scale_ptrs += offs_ks_step * stride_bscale_k

        c = accumulator.to(c_ptr.type.element_ty)

        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
        c_ptrs = (
            c_ptr
            + stride_cm * offs_cm[:, None]
            + stride_cn * offs_cn[None, :]
            + pid_k * stride_ck
        )
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)


def _get_config(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
):
    shuffle_suffix = "_PRESHUFFLED" if shuffle else ""
    config_name = f"GEMM-A8W8_BLOCKSCALE{shuffle_suffix}"

    return get_gemm_config(config_name, M, N, K)


# LOCAL PATCH #3 (Phase C.2, 2026-05-26): the @triton.heuristics decorator was
# removed from _gemm_a8w8_blockscale_kernel above (see comment there). The
# previous `_aot` alias that tried to expose .fn for AOT-compile is no longer
# needed — Python JIT and C++ AOT now both target the same unwrapped JITFunction.
