# ─────────────────────────────────────────────────────────────────────────
# VENDORED FROM ROCm/aiter
#
#   Source:   aiter/ops/triton/_triton_kernels/attention/unified_attention.py
#   Upstream: https://github.com/ROCm/aiter
#   Repo HEAD at vendoring time: 163e6a02517fa9b86abf2f648c97fe6ae74b6d7c (2026-05-20)
#   Last commit touching this file: 7f613da837208886c1d9a958550f1ad50bd0ef7e (2026-05-20)
#
# Re-vendor 2026-05-20: bumped to AITER HEAD `163e6a02`. Sole upstream change
# since previous SHA `32e1e6d` (2026-05-18) that touches this kernel is PR #3231
# (commit 7f613da8): `acc += tl.dot(P.to(V.dtype), V)` →
# `acc = tl.dot(P.to(V.dtype), V, acc=acc)` at the V·scores accumulate sites
# in kernel_unified_attention_2d (line 754) and kernel_unified_attention_3d
# (line 1099). Patch applied. See ../RDNA4_AUDIT_2026-05-20.md for the
# investigation that motivated the re-vendor (MAD-202).
#
# License: MIT (matches upstream aiter LICENSE — see
#          https://github.com/ROCm/aiter/blob/main/LICENSE).
#
# AITER itself is "adapted from vLLM" per the original module header
# preserved below — see https://github.com/vllm-project/vllm/blob/main/
# vllm/attention/ops/triton_unified_attention.py.
#
# ─────────────────────────────────────────────────────────────────────────
# LOCAL PATCHES APPLIED (deviations from upstream):
#
# 1. e4m3_dtype import (line 6 upstream):
#    Replaced  `from aiter.ops.triton.utils.types import e4m3_dtype`
#    With      hardcoded `e4m3_dtype = torch.float8_e4m3fn`
#
#    Why: the upstream import pulls in the entire aiter package just to
#    resolve a single dtype. For our AOT-compile use case (we ship the
#    .py and feed it to `triton.tools.compile`, never import aiter at
#    runtime), the package dep is dead weight. Upstream resolves e4m3 to
#    `torch.float8_e4m3fn` on gfx1201/gfx1100/Hopper+ targets and to
#    `torch.float8_e4m3fnuz` on gfx942/MI300; this vendor hardcodes the
#    former. If/when we target MI300, branch on arch here.
#
# 2. CACHE_TYPE constexpr + turbo3/turbo4 inline dequant (MAD-199):
#    Adds  CACHE_TYPE: tl.constexpr  to kernel_unified_attention_{2d,3d}
#    plus turbo3_centroid / turbo4_centroid LUT helpers and
#    load_{turbo3,turbo4}_kv_tile @triton.jit dequant helpers.
#    K/V load sites branch on CACHE_TYPE:
#       0 = F16   (upstream behavior, unchanged)
#       1 = TURBO3 (3.125 bpv, our local quant scheme)
#       2 = TURBO4 (4.125 bpv, our local quant scheme)
#
#    Why: avoid F16 round-trip scratch buffer when AITER serves a
#    turbo-quantized KV cache. Keeps the kernel body close to upstream —
#    branching only at the load sites, decode/softmax/V·scores all share.
#    Per mt_pagedattn.cu turbo4 scatter step-4: RHT is intentionally
#    SKIPPED for paged-tile (AITER mirrors this), so dequant is just
#    `centroid_LUT[idx] * norm` — no rotation math.
#
#    AOT spec implications: turbo cache pointers are *i8:16 (byte ptrs),
#    F16 cache pointers remain *fp16:16. CACHE_TYPE is baked per-spec.
#
# Validated: AITER 2D + 3D + reduce_segments AOT-compile cleanly for
#            gfx1201 (R9700) and gfx1030 (6900XT) from this vendor.
#            See docs/aiter-integration/ARCHITECTURE.md §7 + MAD-188.
# ─────────────────────────────────────────────────────────────────────────

# The kernels in this file are adapted from vLLM:
# https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/triton_unified_attention.py
import triton
import triton.language as tl
import torch

# LOCAL PATCH (see vendoring header above): hardcoded for gfx1201/gfx1100/Hopper+.
e4m3_dtype = torch.float8_e4m3fn

float8_info = torch.finfo(e4m3_dtype)


@triton.jit
def fast_exp(x):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    return tl.math.exp2(x * RCP_LN2)


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def apply_softcap(S, x):
    Sdiv = S / x
    p1 = tl.math.exp2(Sdiv)
    p2 = tl.math.exp2(-Sdiv)
    return x * (p1 - p2) / (p1 + p2)


@triton.jit
def find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    left: tl.int32 = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val

        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid

    return left - 1


# ─────────────────────────────────────────────────────────────────────────
# LOCAL ADDITION (MAD-199): turbo3 / turbo4 inline KV dequant.
#
# CACHE_TYPE constants (kept as plain Python ints so they're cheap to use
# in `tl.constexpr` comparisons inside @triton.jit functions):
#   0 = F16    — upstream f16 KV cache, no dequant
#   1 = TURBO3 — 3.125 bpv (3-bit centroid + 1-bit hi-bit split)
#   2 = TURBO4 — 4.125 bpv (4-bit nibble-packed centroid)
#
# Block-byte layouts (matches ggml-common.h block_turbo{3,4}_0; sizeof is
# 2 + QK_TURBO3/4 + QK_TURBO3/8 = 50 and 2 + 2 + QK_TURBO4/2 = 68 with
# QK_TURBO3 = QK_TURBO4 = 128 — the inline "// N bytes" comments in
# ggml-common.h are stale from a pre-128 QK era, the static_assert is
# authoritative):
#   block_turbo3_0 = 50 bytes per 128-element block:
#     [0..2)    norm  (fp16)
#     [2..34)   qs    (32 bytes, 2 low-bits-of-3-bit-idx × 4 per byte)
#     [34..50)  signs (16 bytes, 1 hi-bit-of-3-bit-idx × 8 per byte)
#   block_turbo4_0 = 68 bytes per 128-element block:
#     [0..2)   norm  (fp16)
#     [2..4)   rnorm (fp16, reserved/unused in 4-bit mode)
#     [4..68)  qs    (64 bytes, 4-bit nibble packed, 2 indices per byte)
#
# Centroid LUT values: Lloyd-Max optimal for N(0, 1/d) — match the C++
# TURBO_CENTROIDS_{3,4}BIT constants in ggml-cuda/turbo-quant.cuh exactly.
# Both backends MUST use the same LUTs (cache contents are interchangeable
# in principle, even though Option-B layouts differ).
# ─────────────────────────────────────────────────────────────────────────

# Python-side constants — for callers that want named values (not used in kernels)
CACHE_TYPE_F16    = 0
CACHE_TYPE_TURBO3 = 1
CACHE_TYPE_TURBO4 = 2

# MAD-214: turbo-FP8 family — FP8 matrix-core path via centroid LUT decode.
# Numeric scheme matches mt_aiter_cache_type enum in
# wrappers/mt_aiter_unified_attn.h:
#   10..14 = turbo3_fp8 × BS{16,32,64,128,256}
#   20..24 = turbo4_fp8 × BS{16,32,64,128,256}
#   30..34 = turbo5_fp8 × BS{16,32,64,128,256}
# Only BS=256 (production) variants are wired at MAD-214 Phase 1 ship;
# others reserved for MAD-215 follow-up.
CACHE_TYPE_TURBO3_FP8_BS16  = 10
CACHE_TYPE_TURBO3_FP8_BS32  = 11
CACHE_TYPE_TURBO3_FP8_BS64  = 12
CACHE_TYPE_TURBO3_FP8_BS128 = 13
CACHE_TYPE_TURBO3_FP8_BS256 = 14

CACHE_TYPE_TURBO4_FP8_BS16  = 20
CACHE_TYPE_TURBO4_FP8_BS32  = 21
CACHE_TYPE_TURBO4_FP8_BS64  = 22
CACHE_TYPE_TURBO4_FP8_BS128 = 23
CACHE_TYPE_TURBO4_FP8_BS256 = 24

CACHE_TYPE_TURBO5_FP8_BS16  = 30
CACHE_TYPE_TURBO5_FP8_BS32  = 31
CACHE_TYPE_TURBO5_FP8_BS64  = 32
CACHE_TYPE_TURBO5_FP8_BS128 = 33
CACHE_TYPE_TURBO5_FP8_BS256 = 34

# Production aliases — unsuffixed names resolve to the BS=256 variant.
CACHE_TYPE_TURBO3_FP8 = CACHE_TYPE_TURBO3_FP8_BS256
CACHE_TYPE_TURBO4_FP8 = CACHE_TYPE_TURBO4_FP8_BS256
CACHE_TYPE_TURBO5_FP8 = CACHE_TYPE_TURBO5_FP8_BS256

# Triton-side constexprs — wrapped in tl.constexpr because Triton rejects plain
# Python globals inside @triton.jit functions per Triton docs (NameError on use,
# even though triton.language.constexpr(...) is the canonical workaround).
BYTES_PER_TURBO3_BLOCK = tl.constexpr(50)
BYTES_PER_TURBO4_BLOCK = tl.constexpr(68)

# turbo-FP8 block sizes — must match ggml-common.h
#   block_turbo{3,4,5}_fp8_bs{N} = 2 (FP16 scale) + N*K/8 (idx) + N/8 (sign)
# where K = 3 / 4 / 5 for turbo3 / turbo4 / turbo5 respectively.
BYTES_PER_TURBO3_FP8_BS16_BLOCK  = tl.constexpr(10)
BYTES_PER_TURBO3_FP8_BS32_BLOCK  = tl.constexpr(18)
BYTES_PER_TURBO3_FP8_BS64_BLOCK  = tl.constexpr(34)
BYTES_PER_TURBO3_FP8_BS128_BLOCK = tl.constexpr(66)
BYTES_PER_TURBO3_FP8_BS256_BLOCK = tl.constexpr(130)

BYTES_PER_TURBO4_FP8_BS16_BLOCK  = tl.constexpr(12)
BYTES_PER_TURBO4_FP8_BS32_BLOCK  = tl.constexpr(22)
BYTES_PER_TURBO4_FP8_BS64_BLOCK  = tl.constexpr(42)
BYTES_PER_TURBO4_FP8_BS128_BLOCK = tl.constexpr(82)
BYTES_PER_TURBO4_FP8_BS256_BLOCK = tl.constexpr(162)

BYTES_PER_TURBO5_FP8_BS16_BLOCK  = tl.constexpr(14)
BYTES_PER_TURBO5_FP8_BS32_BLOCK  = tl.constexpr(26)
BYTES_PER_TURBO5_FP8_BS64_BLOCK  = tl.constexpr(50)
BYTES_PER_TURBO5_FP8_BS128_BLOCK = tl.constexpr(98)
BYTES_PER_TURBO5_FP8_BS256_BLOCK = tl.constexpr(194)


@triton.jit
def turbo3_centroid(idx):
    # 3-bit Lloyd-Max centroids (8 entries). idx in [0, 8).
    #
    # The centroids are antisymmetric around the midpoint:
    #   centroid[i] = -centroid[7-i] for all i.
    # Exploit that: peel off the sign (bit 2), then 2-level binary-tree
    # select over the 4 magnitudes (vs 7-deep sequential tl.where chain).
    # Critical path: 2 levels of select + 1 select for sign + 1 negate.
    is_pos    = idx >= 4                                # bit 2 == 1
    mag_idx   = tl.where(is_pos, idx - 4, 3 - idx)      # 0..3, mag in increasing order
    m01       = tl.where(mag_idx == 0, 0.021460, 0.065717)  # |c| for 0, 1
    m23       = tl.where(mag_idx == 2, 0.117832, 0.190685)  # |c| for 2, 3
    mag       = tl.where(mag_idx < 2,  m01, m23)
    return tl.where(is_pos, mag, -mag)


@triton.jit
def turbo4_centroid(idx):
    # 4-bit Lloyd-Max centroids (16 entries). idx in [0, 16).
    #
    # Antisymmetric: centroid[i] = -centroid[15-i]. Same trick as turbo3:
    # peel sign (bit 3), 3-level binary tree over 8 magnitudes.
    # Critical path: 3 levels of select + 1 select for sign + 1 negate
    # (down from 15 sequential).
    is_pos    = idx >= 8
    mag_idx   = tl.where(is_pos, idx - 8, 7 - idx)      # 0..7
    m01       = tl.where(mag_idx == 0, 0.006938, 0.020989)
    m23       = tl.where(mag_idx == 2, 0.035597, 0.051262)
    m45       = tl.where(mag_idx == 4, 0.068756, 0.089527)
    m67       = tl.where(mag_idx == 6, 0.117195, 0.173926)
    m0123     = tl.where(mag_idx < 2,  m01, m23)
    m4567     = tl.where(mag_idx < 6,  m45, m67)
    mag       = tl.where(mag_idx < 4,  m0123, m4567)
    return tl.where(is_pos, mag, -mag)


# ─────────────────────────────────────────────────────────────────────────
# MAD-214: turbo-FP8 decode helpers (production-shape, output FP8 bytes for
# direct tl.dot consumption). Parameterized by N_CENTROIDS / IDX_BITS so the
# same body handles turbo3 (8 cent, 3 bits) / turbo4 (16, 4) / turbo5 (32, 5).
#
# Unlike turbo3_centroid / turbo4_centroid above (which return fp32 magnitudes
# from a hardcoded table), turbo-FP8 uses PER-(kv, layer) centroid LUTs loaded
# at runtime as positive E4M3 bytes — see scripts/calibration/export_centroids.py
# generated headers. Sign bit is XOR'd into the byte at decode time, producing
# a signed FP8 byte that's reinterpreted as tl.float8e4nv for tl.dot.
#
# Validated bit-exact against the scalar reference in
# tests/test-turbo-fp8-reference.cpp (see also tests/test_triton_turbo_fp8_decode.py).
# ─────────────────────────────────────────────────────────────────────────

@triton.jit
def turbo_fp8_extract_idx(qs_word, bit_off, IDX_BITS: tl.constexpr):
    """Extract IDX_BITS index from a 16-bit window at bit_off."""
    mask = (1 << IDX_BITS) - 1
    return (qs_word >> bit_off) & mask


@triton.jit
def turbo_fp8_decode_byte(
    qs_lo,            # uint8 — low byte of the bit window for this element
    qs_hi,            # uint8 — high byte (next byte; ignored if element is byte-aligned)
    bit_off,          # int   — bit offset within the low byte
    sign_byte,        # uint8 — sign-bits byte for this element's group
    sign_bit_off,     # int   — bit offset within the sign byte
    lut_ptr,          # *uint8 — N_CENTROIDS E4M3 bytes for this (kv, layer)
    IDX_BITS: tl.constexpr,
):
    """
    Decode one packed turbo-FP8 element to a signed FP8 byte.

      idx       = ((qs_hi << 8) | qs_lo) >> bit_off  & ((1 << IDX_BITS) - 1)
      sign      = (sign_byte >> sign_bit_off) & 1
      cent_byte = lut[idx]
      out_byte  = cent_byte ^ (sign << 7)   # XOR into the FP8 sign bit

    Returns an int8 holding the signed E4M3 byte. Reinterpret as
    tl.float8e4nv via .to(tl.float8e4nv, bitcast=True) at the tl.dot site.
    """
    word          = qs_lo.to(tl.int32) | (qs_hi.to(tl.int32) << 8)
    idx           = turbo_fp8_extract_idx(word, bit_off, IDX_BITS)
    sign_bit      = (sign_byte.to(tl.int32) >> sign_bit_off) & 1
    centroid_byte = tl.load(lut_ptr + idx).to(tl.int32)
    return (centroid_byte | (sign_bit << 7)).to(tl.int8)


@triton.jit
def _broadcast_norms_by_qb(
    cache_byte_ptr,          # *i8 — byte pointer to cache
    block_byte_base,         # (TILE_SIZE,) — per-token byte base for qb_idx=0
    qb_idx_per_d,            # (HEAD_SIZE_PADDED,) — qb_idx of each head_dim
    tile_mask,               # (TILE_SIZE,) — valid-token mask
    QB_PER_TOK: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
):
    # Returns (HEAD_SIZE_PADDED, TILE_SIZE) fp32: per-(head_dim,token) norm value
    # obtained by ONE fp16 load per (token, qb_idx) then broadcast.
    #
    # The naive version (one tl.load per (head_dim, token)) is correctness-
    # equivalent but does QB_PER_TOK*128× more loads since each turbo block
    # holds 128 head_dim elements that all share the block's norm.
    if QB_PER_TOK == 1:
        # Single norm per token. Load once, broadcast across head_dim.
        norm_ptr = (cache_byte_ptr + block_byte_base).to(tl.pointer_type(tl.float16))
        norms_t  = tl.load(norm_ptr, mask=tile_mask, other=0.0).to(tl.float32)  # (TILE_SIZE,)
        return norms_t[None, :] + tl.zeros((1, 1), dtype=tl.float32)  # broadcast hint
    else:
        # QB_PER_TOK == 2 (head_size=256). Load both norms per token, pick by qb_idx.
        ptr0  = (cache_byte_ptr + block_byte_base).to(tl.pointer_type(tl.float16))
        ptr1  = (cache_byte_ptr + block_byte_base + BYTES_PER_BLOCK).to(tl.pointer_type(tl.float16))
        n0    = tl.load(ptr0, mask=tile_mask, other=0.0).to(tl.float32)  # (TILE_SIZE,)
        n1    = tl.load(ptr1, mask=tile_mask, other=0.0).to(tl.float32)
        # Select per head_dim
        return tl.where(qb_idx_per_d[:, None] == 0, n0[None, :], n1[None, :])


@triton.jit
def load_turbo3_kv_tile_K(
    cache_byte_ptr,          # *i8 — byte pointer to start of K cache
    physical_block_idx,      # tensor of shape (TILE_SIZE,) — paged block id per token
    token_in_block,          # tensor of shape (TILE_SIZE,) — slot within paged block
    offs_d,                  # tensor of shape (HEAD_SIZE_PADDED,) — head_dim offsets
    kv_head_idx,             # scalar int — kv head index
    n_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    dim_mask,                # tensor of shape (HEAD_SIZE_PADDED,) — valid head_dim mask
    tile_mask,               # tensor of shape (TILE_SIZE,) — valid token mask
):
    # Returns: tensor of shape (HEAD_SIZE_PADDED, TILE_SIZE), dtype fp32.
    # Matches the f16 K_load shape convention so the matmul site is unchanged.
    QK_TURBO: tl.constexpr = 128
    QB_PER_TOK: tl.constexpr = HEAD_SIZE // QK_TURBO

    # Token's first-block byte base (qb_idx = 0):
    block_byte_base = (
        physical_block_idx * (BLOCK_SIZE * n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO3_BLOCK)
        + token_in_block   * (n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO3_BLOCK)
        + kv_head_idx      * (QB_PER_TOK * BYTES_PER_TURBO3_BLOCK)
    )  # shape (TILE_SIZE,) int64

    qb_idx_per_d = (offs_d // QK_TURBO).to(tl.int64)              # (HEAD_SIZE_PADDED,)
    j_per_d      = offs_d % QK_TURBO                              # (HEAD_SIZE_PADDED,)
    elem_block_off = qb_idx_per_d[:, None] * BYTES_PER_TURBO3_BLOCK  # (HEAD_SIZE_PADDED, 1)

    load_mask = dim_mask[:, None] & tile_mask[None, :]

    # Norm: one load per (token, qb_idx) then broadcast across head_dim within
    # the qblock (MAD-199 chunk D1: fixes a 64-128× redundant-norm-load bug).
    norms = _broadcast_norms_by_qb(
        cache_byte_ptr, block_byte_base, qb_idx_per_d, tile_mask,
        QB_PER_TOK, BYTES_PER_TURBO3_BLOCK,
    )

    # qs starts at byte 2 within the block; signs at byte 34 (= 2 + 32, after qs).
    qs_byte_off    = block_byte_base[None, :] + elem_block_off +  2 + (j_per_d[:, None] // 4)
    signs_byte_off = block_byte_base[None, :] + elem_block_off + 34 + (j_per_d[:, None] // 8)

    qs_bytes    = tl.load(cache_byte_ptr + qs_byte_off,    mask=load_mask, other=0)
    signs_bytes = tl.load(cache_byte_ptr + signs_byte_off, mask=load_mask, other=0)

    lo2 = (qs_bytes    >> ((j_per_d[:, None] % 4) * 2).to(tl.uint8)) & 0x3
    hi1 = (signs_bytes >> ( j_per_d[:, None] % 8     ).to(tl.uint8)) & 0x1
    idx = lo2 | (hi1 << 2)

    return turbo3_centroid(idx) * norms


@triton.jit
def load_turbo4_kv_tile_K(
    cache_byte_ptr,
    physical_block_idx,
    token_in_block,
    offs_d,
    kv_head_idx,
    n_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    dim_mask,
    tile_mask,
):
    # See load_turbo3_kv_tile_K for layout notes.
    QK_TURBO: tl.constexpr = 128
    QB_PER_TOK: tl.constexpr = HEAD_SIZE // QK_TURBO

    block_byte_base = (
        physical_block_idx * (BLOCK_SIZE * n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO4_BLOCK)
        + token_in_block   * (n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO4_BLOCK)
        + kv_head_idx      * (QB_PER_TOK * BYTES_PER_TURBO4_BLOCK)
    )

    qb_idx_per_d   = (offs_d // QK_TURBO).to(tl.int64)
    j_per_d        = offs_d % QK_TURBO
    elem_block_off = qb_idx_per_d[:, None] * BYTES_PER_TURBO4_BLOCK

    load_mask = dim_mask[:, None] & tile_mask[None, :]

    norms = _broadcast_norms_by_qb(
        cache_byte_ptr, block_byte_base, qb_idx_per_d, tile_mask,
        QB_PER_TOK, BYTES_PER_TURBO4_BLOCK,
    )

    # qs starts at byte 4 (skip norm + rnorm), 1 byte holds 2 nibbles.
    qs_byte_off = block_byte_base[None, :] + elem_block_off + 4 + (j_per_d[:, None] // 2)
    qs_bytes    = tl.load(cache_byte_ptr + qs_byte_off, mask=load_mask, other=0)
    idx         = (qs_bytes >> ((j_per_d[:, None] % 2) * 4).to(tl.uint8)) & 0xF

    return turbo4_centroid(idx) * norms


@triton.jit
def _broadcast_norms_by_qb_V(
    cache_byte_ptr,
    block_byte_base,         # (TILE_SIZE,) per-token byte base for qb_idx=0
    qb_idx_per_d,            # (HEAD_SIZE_PADDED,)
    tile_mask,               # (TILE_SIZE,)
    QB_PER_TOK: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
):
    # V version: returns (TILE_SIZE, HEAD_SIZE_PADDED) fp32 — transposed shape vs K helper.
    if QB_PER_TOK == 1:
        norm_ptr = (cache_byte_ptr + block_byte_base).to(tl.pointer_type(tl.float16))
        norms_t  = tl.load(norm_ptr, mask=tile_mask, other=0.0).to(tl.float32)
        return norms_t[:, None] + tl.zeros((1, 1), dtype=tl.float32)
    else:
        ptr0 = (cache_byte_ptr + block_byte_base).to(tl.pointer_type(tl.float16))
        ptr1 = (cache_byte_ptr + block_byte_base + BYTES_PER_BLOCK).to(tl.pointer_type(tl.float16))
        n0   = tl.load(ptr0, mask=tile_mask, other=0.0).to(tl.float32)
        n1   = tl.load(ptr1, mask=tile_mask, other=0.0).to(tl.float32)
        return tl.where(qb_idx_per_d[None, :] == 0, n0[:, None], n1[:, None])


@triton.jit
def load_turbo3_kv_tile_V(
    cache_byte_ptr,
    physical_block_idx,      # shape (TILE_SIZE,)
    token_in_block,          # shape (TILE_SIZE,)
    offs_d,                  # shape (HEAD_SIZE_PADDED,)
    kv_head_idx,
    n_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    dim_mask,
    tile_mask,
):
    # V tile is (TILE_SIZE, HEAD_SIZE_PADDED) — token-major, transpose of K.
    QK_TURBO: tl.constexpr = 128
    QB_PER_TOK: tl.constexpr = HEAD_SIZE // QK_TURBO

    block_byte_base = (
        physical_block_idx * (BLOCK_SIZE * n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO3_BLOCK)
        + token_in_block   * (n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO3_BLOCK)
        + kv_head_idx      * (QB_PER_TOK * BYTES_PER_TURBO3_BLOCK)
    )

    qb_idx_per_d   = (offs_d // QK_TURBO).to(tl.int64)
    j_per_d        = offs_d % QK_TURBO
    elem_block_off = qb_idx_per_d[None, :] * BYTES_PER_TURBO3_BLOCK  # (1, HEAD_SIZE_PADDED)

    load_mask = dim_mask[None, :] & tile_mask[:, None]

    norms = _broadcast_norms_by_qb_V(
        cache_byte_ptr, block_byte_base, qb_idx_per_d, tile_mask,
        QB_PER_TOK, BYTES_PER_TURBO3_BLOCK,
    )

    qs_byte_off    = block_byte_base[:, None] + elem_block_off +  2 + (j_per_d[None, :] // 4)
    signs_byte_off = block_byte_base[:, None] + elem_block_off + 34 + (j_per_d[None, :] // 8)

    qs_bytes    = tl.load(cache_byte_ptr + qs_byte_off,    mask=load_mask, other=0)
    signs_bytes = tl.load(cache_byte_ptr + signs_byte_off, mask=load_mask, other=0)

    lo2 = (qs_bytes    >> ((j_per_d[None, :] % 4) * 2).to(tl.uint8)) & 0x3
    hi1 = (signs_bytes >> ( j_per_d[None, :] % 8     ).to(tl.uint8)) & 0x1
    idx = lo2 | (hi1 << 2)

    return turbo3_centroid(idx) * norms


@triton.jit
def load_turbo4_kv_tile_V(
    cache_byte_ptr,
    physical_block_idx,
    token_in_block,
    offs_d,
    kv_head_idx,
    n_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    dim_mask,
    tile_mask,
):
    QK_TURBO: tl.constexpr = 128
    QB_PER_TOK: tl.constexpr = HEAD_SIZE // QK_TURBO

    block_byte_base = (
        physical_block_idx * (BLOCK_SIZE * n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO4_BLOCK)
        + token_in_block   * (n_kv_heads * QB_PER_TOK * BYTES_PER_TURBO4_BLOCK)
        + kv_head_idx      * (QB_PER_TOK * BYTES_PER_TURBO4_BLOCK)
    )

    qb_idx_per_d   = (offs_d // QK_TURBO).to(tl.int64)
    j_per_d        = offs_d % QK_TURBO
    elem_block_off = qb_idx_per_d[None, :] * BYTES_PER_TURBO4_BLOCK

    load_mask = dim_mask[None, :] & tile_mask[:, None]

    norms = _broadcast_norms_by_qb_V(
        cache_byte_ptr, block_byte_base, qb_idx_per_d, tile_mask,
        QB_PER_TOK, BYTES_PER_TURBO4_BLOCK,
    )

    qs_byte_off = block_byte_base[:, None] + elem_block_off + 4 + (j_per_d[None, :] // 2)
    qs_bytes    = tl.load(cache_byte_ptr + qs_byte_off, mask=load_mask, other=0)
    idx         = (qs_bytes >> ((j_per_d[None, :] % 2) * 4).to(tl.uint8)) & 0xF

    return turbo4_centroid(idx) * norms


# ─────────────────────────────────────────────────────────────────────────
# MAD-214: turbo-FP8 KV tile loaders.
#
# Unlike load_turbo{3,4}_kv_tile_K/V (which return dequant'd FP32 with scale
# baked in), the turbo-FP8 loaders return:
#   (fp8_tile_bytes, scales_per_token)
#
# fp8_tile_bytes  shape (HEAD_SIZE_PADDED, TILE_SIZE)  dtype tl.int8 (signed
#                                                     E4M3 bit pattern,
#                                                     bitcast at tl.dot site)
# scales          shape (TILE_SIZE,)                  dtype fp32 (per-token
#                                                     K-row scale; multiplied
#                                                     into the FP32 accumulator
#                                                     POST-WMMA)
#
# Production assumes BLOCK_SIZE=256 (one block per K-row), so QB_PER_TOK=1 and
# there's no per-qb broadcast. BS<256 variants (per MAD-215) require a
# different load function with per-block accumulation in the caller.
# ─────────────────────────────────────────────────────────────────────────

@triton.jit
def load_turbo_fp8_kv_tile_K_bs256(
    cache_byte_ptr,          # *i8 — base of K cache
    physical_block_idx,      # (TILE_SIZE,) — paged block id per token
    token_in_block,          # (TILE_SIZE,) — slot within paged block
    offs_d,                  # (HEAD_SIZE_PADDED,) — head_dim offsets
    kv_head_idx,             # scalar int — KV head index
    lut_ptr,                 # *u8 — per-(kv, layer) centroid LUT (N_CENTROIDS bytes)
    n_kv_heads:        tl.constexpr,
    BLOCK_SIZE:        tl.constexpr,
    HEAD_SIZE:         tl.constexpr,
    IDX_BITS:          tl.constexpr,  # 3 (turbo3) / 4 (turbo4) / 5 (turbo5)
    BYTES_PER_BLOCK:   tl.constexpr,  # 130 / 162 / 194 for BS=256
    dim_mask,                # (HEAD_SIZE_PADDED,) — valid head_dim mask
    tile_mask,               # (TILE_SIZE,) — valid token mask
):
    # BS=256 simplification: one block per token row (assuming HEAD_SIZE=256).
    # If HEAD_SIZE != BLOCK_SIZE=256 we'd need the per-block accumulation path
    # (MAD-215 — currently flagged as not-implemented at the wrapper layer).
    tl.static_assert(HEAD_SIZE == 256, "turbo-FP8 BS=256 kernel requires HEAD_SIZE=256 (Qwen3.5 family)")

    # Byte base of each token's block in the paged cache.
    block_byte_base = (
        physical_block_idx * (BLOCK_SIZE * n_kv_heads * BYTES_PER_BLOCK)
        + token_in_block   * (n_kv_heads * BYTES_PER_BLOCK)
        + kv_head_idx      * BYTES_PER_BLOCK
    )  # shape (TILE_SIZE,) int64

    # Per-token FP16 scale (load once, no broadcast needed across head_dim
    # since the block IS the row).
    scale_ptr = (cache_byte_ptr + block_byte_base).to(tl.pointer_type(tl.float16))
    scales    = tl.load(scale_ptr, mask=tile_mask, other=0.0).to(tl.float32)  # (TILE_SIZE,)

    # Within a block, layout is:
    #   bytes [0..2)                            : FP16 scale
    #   bytes [2..2+256*IDX_BITS/8)             : packed indices (IDX_BITS per element)
    #   bytes [2+256*IDX_BITS/8 .. +32)         : sign bits (1 per element, 32 bytes total)
    QS_BYTES:     tl.constexpr = 256 * IDX_BITS // 8
    SIGNS_OFFSET: tl.constexpr = 2 + QS_BYTES

    # Per-element bit position in the packed qs[] stream.
    bit_pos     = offs_d * IDX_BITS                    # (HEAD_SIZE_PADDED,)
    byte_lo_off = (bit_pos // 8).to(tl.int64)          # which byte in qs
    bit_off     = (bit_pos % 8).to(tl.int32)           # bit offset within that byte

    # Broadcast offsets to (HEAD_SIZE_PADDED, TILE_SIZE) and add token-base.
    qs_lo_addr = block_byte_base[None, :] + 2 + byte_lo_off[:, None]
    qs_hi_addr = qs_lo_addr + 1
    sign_addr  = block_byte_base[None, :] + SIGNS_OFFSET + (offs_d[:, None] // 8).to(tl.int64)

    load_mask = dim_mask[:, None] & tile_mask[None, :]
    qs_lo      = tl.load(cache_byte_ptr + qs_lo_addr, mask=load_mask, other=0)
    qs_hi      = tl.load(cache_byte_ptr + qs_hi_addr, mask=load_mask, other=0)
    sign_bytes = tl.load(cache_byte_ptr + sign_addr,  mask=load_mask, other=0)
    sign_bit_off = (offs_d % 8).to(tl.int32)            # (HEAD_SIZE_PADDED,)

    # Extract IDX_BITS index, look up centroid byte, XOR sign → signed E4M3 byte.
    word          = qs_lo.to(tl.int32) | (qs_hi.to(tl.int32) << 8)
    mask          = (1 << IDX_BITS) - 1
    idx           = (word >> bit_off[:, None]) & mask
    centroid_byte = tl.load(lut_ptr + idx).to(tl.int32)
    sign_bit      = (sign_bytes.to(tl.int32) >> sign_bit_off[:, None]) & 1
    fp8_bytes     = (centroid_byte | (sign_bit << 7)).to(tl.int8)

    return fp8_bytes, scales


@triton.jit
def load_turbo_fp8_kv_tile_V_bs256(
    cache_byte_ptr,
    physical_block_idx,
    token_in_block,
    offs_d,
    kv_head_idx,
    lut_ptr,
    n_kv_heads:        tl.constexpr,
    BLOCK_SIZE:        tl.constexpr,
    HEAD_SIZE:         tl.constexpr,
    IDX_BITS:          tl.constexpr,
    BYTES_PER_BLOCK:   tl.constexpr,
    dim_mask,
    tile_mask,
):
    """V loader — transposes the result shape vs K (matches existing
    load_turbo4_kv_tile_V convention)."""
    tl.static_assert(HEAD_SIZE == 256, "turbo-FP8 BS=256 kernel requires HEAD_SIZE=256")

    block_byte_base = (
        physical_block_idx * (BLOCK_SIZE * n_kv_heads * BYTES_PER_BLOCK)
        + token_in_block   * (n_kv_heads * BYTES_PER_BLOCK)
        + kv_head_idx      * BYTES_PER_BLOCK
    )

    scale_ptr = (cache_byte_ptr + block_byte_base).to(tl.pointer_type(tl.float16))
    scales    = tl.load(scale_ptr, mask=tile_mask, other=0.0).to(tl.float32)  # (TILE_SIZE,)

    QS_BYTES:     tl.constexpr = 256 * IDX_BITS // 8
    SIGNS_OFFSET: tl.constexpr = 2 + QS_BYTES

    bit_pos     = offs_d * IDX_BITS
    byte_lo_off = (bit_pos // 8).to(tl.int64)
    bit_off     = (bit_pos % 8).to(tl.int32)

    # V tile shape is (TILE_SIZE, HEAD_SIZE_PADDED) — token-major.
    qs_lo_addr = block_byte_base[:, None] + 2 + byte_lo_off[None, :]
    qs_hi_addr = qs_lo_addr + 1
    sign_addr  = block_byte_base[:, None] + SIGNS_OFFSET + (offs_d[None, :] // 8).to(tl.int64)

    load_mask = tile_mask[:, None] & dim_mask[None, :]
    qs_lo      = tl.load(cache_byte_ptr + qs_lo_addr, mask=load_mask, other=0)
    qs_hi      = tl.load(cache_byte_ptr + qs_hi_addr, mask=load_mask, other=0)
    sign_bytes = tl.load(cache_byte_ptr + sign_addr,  mask=load_mask, other=0)
    sign_bit_off = (offs_d % 8).to(tl.int32)

    word          = qs_lo.to(tl.int32) | (qs_hi.to(tl.int32) << 8)
    mask          = (1 << IDX_BITS) - 1
    idx           = (word >> bit_off[None, :]) & mask
    centroid_byte = tl.load(lut_ptr + idx).to(tl.int32)
    sign_bit      = (sign_bytes.to(tl.int32) >> sign_bit_off[None, :]) & 1
    fp8_bytes     = (centroid_byte | (sign_bit << 7)).to(tl.int8)

    return fp8_bytes, scales


@triton.jit
def kernel_unified_attention_2d(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale: tl.constexpr,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    out_scale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    ALL_DECODE: tl.constexpr = False,  # bool
    CACHE_TYPE: tl.constexpr = 0,       # 0=F16 (default), 1=TURBO3, 2=TURBO4 (MAD-199),
                                        # 10..14 = TURBO3_FP8_BS{16..256} (MAD-214),
                                        # 20..24 = TURBO4_FP8_BS{16..256},
                                        # 30..34 = TURBO5_FP8_BS{16..256}
    # MAD-214: per-(kv, layer) E4M3 centroid LUTs for turbo-FP8 paths. Caller
    # passes the LUT slice already indexed to the current layer's K/V head.
    # Safely None when CACHE_TYPE is not in the turbo-FP8 family.
    centroids_k_ptr = None,             # *u8 — N_CENTROIDS bytes; None for non-FP8
    centroids_v_ptr = None,             # *u8 — N_CENTROIDS bytes; None for non-FP8
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    if ALL_DECODE or BLOCK_M >= num_query_heads:
        Q_cache_modifier: tl.constexpr = ".cg"
    else:
        Q_cache_modifier: tl.constexpr = ""
    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
        cache_modifier=Q_cache_modifier,
    )

    block_table_offset = seq_idx * block_table_stride

    if not USE_SINKS:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        # Prescale with RCP_LN2, needed for exp2
        M = (
            tl.load(
                sink_ptr + query_offset_1,
                mask=query_mask_1,
                other=float("-inf"),
            ).to(dtype=tl.float32)
            * RCP_LN2
        )

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    # ---- Sliding-window tile pruning --------------------
    # Default: keep previous global behavior
    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        # Query rows covered by this Q-block
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(
            qpos_lo + (BLOCK_M - 1) // num_queries_per_kv,
            cur_batch_query_len - 1,
        )
        # For sliding window, each query position q can only attend to
        # keys in the range [q_abs - SLIDING_WINDOW + 1, q_abs]
        # where q_abs = context_len + q
        # The union of allowed key positions for this Q-block is:
        # [context_len + qpos_lo - SLIDING_WINDOW + 1, context_len + qpos_hi]
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        # Convert to tile indices and clamp
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        k_descale = None
        v_descale = None
    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""

    # MAD-214: Q → FP8 quantization for the turbo-FP8 path. Computed ONCE per
    # Q-block (Q is loaded once before the tile loop and reused). Per-row max
    # gives FP8 dynamic range; the Q_scale_fp8 vector is multiplied into the
    # FP32 accumulator AFTER the FP8 tl.dot via qk_scale broadcast.
    #
    # This branch is constexpr-dead for non-FP8 CACHE_TYPEs.
    IS_TURBO_FP8: tl.constexpr = (CACHE_TYPE >= 10) and (CACHE_TYPE <= 34)
    if IS_TURBO_FP8:
        Q_fp32        = Q.to(tl.float32)
        Q_abs_max     = tl.max(tl.abs(Q_fp32), axis=1)          # (BLOCK_M,)
        Q_scale_fp8   = tl.where(Q_abs_max > 0, Q_abs_max, 1.0) # guard zero rows
        Q_normalized  = Q_fp32 / Q_scale_fp8[:, None]           # in [-1, 1]
        Q_fp8_tensor  = Q_normalized.to(tl.float8e4nv)          # (BLOCK_M, HEAD_SIZE_PADDED)

    # iterate through tiles (now limited to the sliding window range)
    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        # to reduce the masking effect when not needed
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        token_in_block = seq_offset % BLOCK_SIZE

        # MAD-199: branch K/V loads on cache type. Branches are tl.constexpr
        # if/elif so only one path lowers per AOT spec; the f16 branch is
        # byte-for-byte the upstream code.
        if CACHE_TYPE == 0:  # F16
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + token_in_block[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + token_in_block[None, :] * stride_k_cache_1
            )
            # K : (HEAD_SIZE, TILE_SIZE)
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
                cache_modifier=KV_cache_modifier,
            )
            # V : (TILE_SIZE, HEAD_SIZE)
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
                cache_modifier=KV_cache_modifier,
            )
            K = K_load.to(Q.dtype)
            V = V_load.to(Q.dtype)
        elif CACHE_TYPE == 1:  # TURBO3
            # n_kv_heads is constexpr-derivable from num_query_heads + GQA factor.
            N_KV_HEADS: tl.constexpr = num_query_heads // num_queries_per_kv
            K = load_turbo3_kv_tile_K(
                key_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
            V = load_turbo3_kv_tile_V(
                value_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
        elif CACHE_TYPE == 2:  # TURBO4
            N_KV_HEADS: tl.constexpr = num_query_heads // num_queries_per_kv
            K = load_turbo4_kv_tile_K(
                key_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
            V = load_turbo4_kv_tile_V(
                value_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
        else:  # MAD-214: turbo-FP8 family. CACHE_TYPE in {10..14, 20..24, 30..34}.
            # Derive constexpr metadata from CACHE_TYPE:
            #   IDX_BITS:        3 for turbo3-FP8 (10..14)
            #                    4 for turbo4-FP8 (20..24)
            #                    5 for turbo5-FP8 (30..34)
            #   BYTES_PER_BLOCK: BS=256 layout = 2 (FP16 scale) + 256*IDX_BITS/8 + 32
            # Only the BS=256 variants (CACHE_TYPE ∈ {14, 24, 34}) are wired
            # at MAD-214 Phase 1 ship; BS<256 (MAD-215) requires the per-block
            # accumulation kernel path not implemented in this kernel.
            tl.static_assert(
                (CACHE_TYPE == 14) or (CACHE_TYPE == 24) or (CACHE_TYPE == 34),
                "turbo-FP8: only BS=256 variants are wired in this kernel (MAD-215 covers BS<256)",
            )
            IDX_BITS: tl.constexpr = 3 if CACHE_TYPE < 20 else (4 if CACHE_TYPE < 30 else 5)
            BYTES_PER_FP8_BLOCK: tl.constexpr = 2 + 256 * IDX_BITS // 8 + 32  # 130 / 162 / 194
            N_KV_HEADS: tl.constexpr = num_query_heads // num_queries_per_kv

            K_fp8, K_scales = load_turbo_fp8_kv_tile_K_bs256(
                key_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                centroids_k_ptr,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                IDX_BITS, BYTES_PER_FP8_BLOCK,
                dim_mask, tile_mask,
            )
            V_fp8, V_scales = load_turbo_fp8_kv_tile_V_bs256(
                value_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                centroids_v_ptr,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                IDX_BITS, BYTES_PER_FP8_BLOCK,
                dim_mask, tile_mask,
            )
            # NOTE (MAD-214 Phase 1F-B, next commit): the dot sites below still
            # reference K and V (fp16) — for the FP8 path we'll branch the dot
            # calls to use (K_fp8, K_scales) and (V_fp8, V_scales). Until that
            # commit lands, this branch produces unused variables. The
            # tl.static_assert above prevents callers from accidentally hitting
            # this path until the dot-site wiring is in place.

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2
        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    # epilogue
    # This helps the compiler do Newton Raphson on l_i vs on acc which is much larger.
    if v_descale is not None:
        one_over_L = v_descale / L[:, None]
    else:
        one_over_L = 1.0 / L[:, None]
    acc = acc * one_over_L
    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    output_offset = (
        query_offset_0[:, None] * output_stride_0
        + query_offset_1[:, None] * output_stride_1
        + offs_d[None, :]
    )

    tl.store(
        output_ptr + output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )


@triton.jit
def kernel_unified_attention_3d(
    segm_output_ptr,
    # [num_tokens, num_query_heads, num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, num_segments]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blks, blk_size, num_kv_heads, head_size]
    sink_ptr,  # [num_query_heads]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    alibi_slopes_ptr,  # [num_query_heads]
    qq_bias_ptr,  # [num_query_tokens, num_query_tokens]
    scale,  # float32
    q_descale_ptr,  # float32
    k_descale_ptr,  # float32
    v_descale_ptr,  # float32
    softcap,  # float32
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    ALL_DECODE: tl.constexpr = False,  # bool
    CACHE_TYPE: tl.constexpr = 0,       # 0=F16 (default), 1=TURBO3, 2=TURBO4 (MAD-199),
                                        # 10..14/20..24/30..34 = TURBO{3,4,5}_FP8_BS{16..256} (MAD-214)
    centroids_k_ptr = None,             # MAD-214: per-(kv, layer) K centroid LUT (*u8)
    centroids_v_ptr = None,             # MAD-214: per-(kv, layer) V centroid LUT (*u8)
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    # needed to use exp2 (exp2 -> exp conversion)
    RCP_LN2 = 1.4426950408889634
    qk_scale = scale * RCP_LN2

    seq_idx = find_seq_idx(
        query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True
    )

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx

    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)

    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        if segm_idx == 0:
            # Prescale with RCP_LN2, needed for exp2
            M = (
                tl.load(
                    sink_ptr + query_offset_1,
                    mask=query_mask_1,
                    other=float("-inf"),
                ).to(dtype=tl.float32)
                * RCP_LN2
            )
        else:
            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    else:
        M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    # context length for this particular sequences
    context_len = seq_len - cur_batch_query_len

    # alibi slope for this head
    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    # query-query attention bias
    if USE_QQ_BIAS:
        qq_bias_row_ptrs = (
            qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0
        )  # shape: [BLOCK_M]

    # compute the length of the longest sequence prefix spanned by any
    # query token in the current q_block (q_block_local_idx)
    max_seq_prefix_len = (
        context_len
        + q_block_local_idx * BLOCK_Q
        + (BLOCK_M - 1) // num_queries_per_kv
        + 1
    )

    # adjust for potential padding in the last q_block by considering the
    # actual sequence length
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)

    # calculate the number of tiles that need to be processed to
    # cover the longest sequence prefix (due to causal masking, tiles beyond
    # this prefix can be skipped)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    KV_cache_modifier: tl.constexpr = ".cg" if ALL_DECODE else ""
    if q_descale_ptr is not None:
        q_descale = tl.load(q_descale_ptr)
        qk_scale = qk_scale * q_descale
    else:
        q_descale = None
    if k_descale_ptr is not None and v_descale_ptr is not None:
        k_descale = tl.load(k_descale_ptr)
        v_descale = tl.load(v_descale_ptr)
        qk_scale = qk_scale * k_descale
    else:
        k_descale = None
        v_descale = None

    # iterate through tiles within current segment
    for j in range(
        segm_idx * tiles_per_segment,
        min((segm_idx + 1) * tiles_per_segment, num_tiles),
    ):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        token_in_block = seq_offset % BLOCK_SIZE

        # MAD-199: branch K/V loads on cache type — see 2D kernel for notes.
        if CACHE_TYPE == 0:  # F16
            v_offset = (
                physical_block_idx[:, None] * stride_v_cache_0
                + kv_head_idx * stride_v_cache_2
                + offs_d[None, :] * stride_v_cache_3
                + token_in_block[:, None] * stride_v_cache_1
            )
            k_offset = (
                physical_block_idx[None, :] * stride_k_cache_0
                + kv_head_idx * stride_k_cache_2
                + offs_d[:, None] * stride_k_cache_3
                + token_in_block[None, :] * stride_k_cache_1
            )
            K_load = tl.load(
                key_cache_ptr + k_offset,
                mask=dim_mask[:, None] & tile_mask[None, :],
                other=0.0,
                cache_modifier=KV_cache_modifier,
            )
            V_load = tl.load(
                value_cache_ptr + v_offset,
                mask=dim_mask[None, :] & tile_mask[:, None],
                other=0.0,
                cache_modifier=KV_cache_modifier,
            )
            K = K_load.to(Q.dtype)
            V = V_load.to(Q.dtype)
        elif CACHE_TYPE == 1:  # TURBO3
            N_KV_HEADS: tl.constexpr = num_query_heads // num_queries_per_kv
            K = load_turbo3_kv_tile_K(
                key_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
            V = load_turbo3_kv_tile_V(
                value_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
        else:  # CACHE_TYPE == 2, TURBO4
            N_KV_HEADS: tl.constexpr = num_query_heads // num_queries_per_kv
            K = load_turbo4_kv_tile_K(
                key_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)
            V = load_turbo4_kv_tile_V(
                value_cache_ptr, physical_block_idx, token_in_block,
                offs_d, kv_head_idx,
                N_KV_HEADS, BLOCK_SIZE, HEAD_SIZE,
                dim_mask, tile_mask,
            ).to(Q.dtype)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        # S : (BLOCK_M, TILE_SIZE)
        # qk_scale = scale * RCP_LN2 (log_2 e) so that we can use exp2 later
        S = qk_scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            # softcap here uses exp2 and consumes RCP_LN2 conversion.
            # multiply by RCP_LN2 again to be used in later exp2
            S = apply_softcap(S, softcap) * RCP_LN2

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if SLIDING_WINDOW > 0:
            S = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                S,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            # prescale w. RCP_LN2 for later exp2
            S += alibi_slope[:, None] * (seq_offset - context_len) * RCP_LN2

        if USE_QQ_BIAS:
            # compute key positions relative to query section
            key_rel_pos = seq_offset - context_len  # shape: [BLOCK_SIZE]
            # load bias only for keys that correspond to queries
            is_query_key = key_rel_pos >= 0 and key_rel_pos < qq_bias_stride_0
            qq_bias = tl.load(
                qq_bias_row_ptrs + key_rel_pos[None, :],
                mask=is_query_key[None, :],  # avoid OOB for context keys
                other=0.0,
            )
            # prescale w. RCP_LN2 for later exp2
            S += qq_bias * RCP_LN2

        # compute running maximum
        # m_j : (BLOCK_M,)
        m_j = tl.maximum(M, tl.max(S, axis=1))

        # For sliding window there's a chance the max is -inf due to masking of
        # the entire row. In this case we need to set m_j 0 to avoid NaN
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        # P : (BLOCK_M, TILE_SIZE,)
        P = tl.math.exp2(S - m_j[:, None])

        # l_j : (BLOCK_M,)
        l_j = tl.sum(P, axis=1)

        # alpha : (BLOCK_M, )
        alpha = tl.math.exp2(M - m_j)

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = acc * alpha[:, None]

        # update constants
        L = L * alpha + l_j
        M = m_j

        # acc : (BLOCK_M, HEAD_SIZE_PADDED)
        acc = tl.dot(P.to(V.dtype), V, acc=acc)

    if v_descale is not None:
        acc = acc * v_descale

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )
    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_ptr,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    num_segments = NUM_SEGMENTS_PER_SEQ
    tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)

    # create masks for subsequent loads
    act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)
    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )

    if HEAD_SIZE_PADDED != HEAD_SIZE:
        offs_d = tl.arange(0, HEAD_SIZE_PADDED)
        dim_mask = offs_d < HEAD_SIZE
    else:
        dim_mask = tl.full((1,), 1, dtype=tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.math.exp2(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.math.exp2(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if out_scale_ptr is not None:
        acc = acc / tl.load(out_scale_ptr)

    if output_ptr.type.element_ty.is_fp8():
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)
