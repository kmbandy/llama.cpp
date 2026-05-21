#!/usr/bin/env python3
"""
MAD-214 Phase 1D step 1: Triton turbo-FP8 decode helper test.

Standalone Triton kernel that takes a packed block_turbo{3,4,5}_fp8 block
(scale + packed indices + sign bits) plus a per-(kv, layer) E4M3 centroid
LUT, and emits the dequantized FP8 bytes ready for tl.dot consumption.

Validates the decoder in isolation against the scalar reference implementation
(tests/test-turbo-fp8-reference.cpp's quantize_block / dequantize_block) before
integrating into ggml/src/ggml-cuda/aiter-integration/kernels/unified_attention.py.

The decoder is parameterized by N_CENTROIDS and IDX_BITS as tl.constexpr so the
same kernel body covers turbo3-FP8 (N=8, bits=3), turbo4 (N=16, bits=4), and
turbo5 (N=32, bits=5).

Usage:
  python3 tests/test_triton_turbo_fp8_decode.py
"""

import os
import sys

import numpy as np
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Scalar reference helpers (mirror of tests/test-turbo-fp8-reference.cpp)
# ---------------------------------------------------------------------------

QK_TURBO_FP8 = 32


def enumerate_e4m3_positive_with_bytes():
    pairs = []
    for byte in range(0, 0x80):
        e = (byte >> 3) & 0xF
        m = byte & 0x7
        if e == 15 and m == 7:
            continue
        if e == 0:
            val = (2 ** -6) * (m / 8.0)
        else:
            val = (2 ** (e - 7)) * (1 + m / 8.0)
        pairs.append((byte, val))
    return sorted(pairs, key=lambda p: p[1])


def e4m3_byte_to_fp32(byte: int) -> float:
    sign = (byte >> 7) & 1
    e = (byte >> 3) & 0xF
    m = byte & 0x7
    if e == 15 and m == 7:
        return float('nan')
    if e == 0:
        v = (2 ** -6) * (m / 8.0)
    else:
        v = (2 ** (e - 7)) * (1 + m / 8.0)
    return -v if sign else v


def quantize_block_ref(values_fp32, centroids_e4m3_bytes, idx_bits):
    """Quantize 32 fp32 values to a packed turbo-FP8 block.

    Returns (scale_fp16_bits, qs_bytes, sign_bytes) tuple.
    Mirror of quantize_block() in tests/test-turbo-fp8-reference.cpp.
    """
    n_centroids = len(centroids_e4m3_bytes)
    qs_n_bytes = (QK_TURBO_FP8 * idx_bits + 7) // 8

    # Per-block scale (max abs), rounded through fp16
    scale_fp32 = float(np.max(np.abs(values_fp32))) if values_fp32.size > 0 else 0.0
    scale_fp16 = np.float16(scale_fp32 if scale_fp32 > 0 else 1.0)
    scale_eff  = float(scale_fp16) if float(scale_fp16) > 0 else scale_fp32
    if scale_eff <= 0:
        scale_eff = 1.0

    centroid_vals = [e4m3_byte_to_fp32(b) for b in centroids_e4m3_bytes]

    qs    = bytearray(qs_n_bytes)
    signs = bytearray(QK_TURBO_FP8 // 8)

    for i, v in enumerate(values_fp32):
        s = 1 if v < 0 else 0
        m = abs(v) / scale_eff
        # Nearest centroid by magnitude
        best_k = 0
        best_err = abs(m - centroid_vals[0])
        for k in range(1, n_centroids):
            e = abs(m - centroid_vals[k])
            if e < best_err:
                best_k, best_err = k, e
        # Pack idx_bits into qs at position i
        bit_pos  = i * idx_bits
        byte_lo  = bit_pos // 8
        bit_off  = bit_pos % 8
        word     = qs[byte_lo] | (qs[byte_lo + 1] << 8 if byte_lo + 1 < qs_n_bytes else 0)
        mask     = ((1 << idx_bits) - 1) << bit_off
        word     = (word & ~mask) | ((best_k << bit_off) & mask)
        qs[byte_lo]     = word & 0xFF
        if byte_lo + 1 < qs_n_bytes:
            qs[byte_lo + 1] = (word >> 8) & 0xFF
        # Sign bit
        sb, bo = i // 8, i % 8
        signs[sb] = (signs[sb] & ~(1 << bo)) | ((s & 1) << bo)

    # FP16 raw bits as uint16
    scale_bits = np.float16(scale_eff).view(np.uint16).item()
    return scale_bits, bytes(qs), bytes(signs)


def dequantize_block_ref(scale_bits, qs_bytes, sign_bytes, centroids_e4m3_bytes, idx_bits):
    """Scalar dequant matching the reference. Returns 32 fp32 values."""
    qs_n_bytes = (QK_TURBO_FP8 * idx_bits + 7) // 8
    scale = float(np.array([scale_bits], dtype=np.uint16).view(np.float16)[0])
    out = np.zeros(QK_TURBO_FP8, dtype=np.float32)
    for i in range(QK_TURBO_FP8):
        bit_pos = i * idx_bits
        byte_lo = bit_pos // 8
        bit_off = bit_pos % 8
        word    = qs_bytes[byte_lo] | (qs_bytes[byte_lo + 1] << 8 if byte_lo + 1 < qs_n_bytes else 0)
        idx     = (word >> bit_off) & ((1 << idx_bits) - 1)
        sb, bo  = i // 8, i % 8
        s       = (sign_bytes[sb] >> bo) & 1
        mag     = e4m3_byte_to_fp32(centroids_e4m3_bytes[idx])
        out[i]  = (-mag if s else mag) * scale
    return out


# ---------------------------------------------------------------------------
# Triton decode kernel
# ---------------------------------------------------------------------------

@triton.jit
def turbo_fp8_decode_kernel(
    scale_ptr,         # *uint16  — block scale (FP16 bits)
    qs_ptr,            # *uint8   — packed indices
    signs_ptr,         # *uint8   — sign bits (4 bytes for 32 elements)
    lut_ptr,           # *uint8   — N_CENTROIDS E4M3 bytes
    out_ptr,           # *fp32    — dequantized 32 values
    N_CENTROIDS: tl.constexpr,
    IDX_BITS:    tl.constexpr,
):
    # Single program, processes ONE block of QK_TURBO_FP8 = 32 elements.
    BLOCK_SIZE: tl.constexpr = 32

    elem_idx = tl.arange(0, BLOCK_SIZE)               # (32,) element index in block

    # ----- Extract N-bit index per element from packed qs -----
    bit_pos = elem_idx * IDX_BITS                     # (32,) bit position
    byte_lo = bit_pos // 8                            # (32,) low byte index
    bit_off = bit_pos % 8                             # (32,) bit offset within low byte
    # Load low + high bytes (some elements straddle a byte boundary)
    qs_lo = tl.load(qs_ptr + byte_lo).to(tl.int32)
    qs_hi = tl.load(qs_ptr + byte_lo + 1).to(tl.int32)
    word  = qs_lo | (qs_hi << 8)                      # (32,) up to 16-bit window
    mask  = (1 << IDX_BITS) - 1
    idx   = (word >> bit_off) & mask                  # (32,) centroid index per element

    # ----- Extract sign bit per element -----
    sb = elem_idx // 8
    bo = elem_idx % 8
    sign_byte = tl.load(signs_ptr + sb).to(tl.int32)
    sign_bit  = (sign_byte >> bo) & 1                 # (32,) 0 or 1

    # ----- Look up centroid byte from LUT (small gather) -----
    centroid_byte = tl.load(lut_ptr + idx).to(tl.int32)  # (32,) positive E4M3 byte

    # ----- Apply sign by XOR with high bit -----
    signed_byte = centroid_byte | (sign_bit << 7)        # (32,) signed E4M3 byte

    # ----- Convert E4M3 byte → fp32 magnitude (inline decode) -----
    # E4M3: 1 sign + 4 exp + 3 mant, bias 7. (Implements the same logic as
    # e4m3_byte_to_fp32() in the scalar reference.)
    s = (signed_byte >> 7) & 1
    e = (signed_byte >> 3) & 0xF
    m = signed_byte & 0x7
    # Subnormal: e == 0  → 2^-6 * (m / 8)
    # Normal:    e > 0   → 2^(e-7) * (1 + m/8)
    sub_val = (m.to(tl.float32) / 8.0) * (1.0 / 64.0)
    # 2^(e-7) using ldexp via cast trick: e in [1, 15] → exp in [-6, 8]
    # tl.math.exp2 takes a fp value
    exp_part = tl.math.exp2((e - 7).to(tl.float32))
    norm_val = exp_part * (1.0 + m.to(tl.float32) / 8.0)
    mag = tl.where(e == 0, sub_val, norm_val)
    fp32_val = tl.where(s == 1, -mag, mag)

    # ----- Multiply by per-block scale (load FP16, cast to FP32) -----
    scale_bits = tl.load(scale_ptr).to(tl.uint16)
    # Reinterpret uint16 as fp16 → fp32. Triton doesn't have direct bit-cast for
    # uint16→fp16; use the dedicated cast.
    scale = scale_bits.to(tl.float16, bitcast=True).to(tl.float32)

    out = fp32_val * scale
    tl.store(out_ptr + elem_idx, out)


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------

def run_variant(name, n_centroids, idx_bits, device):
    print(f"\n--- variant: {name} (N_CENTROIDS={n_centroids}, IDX_BITS={idx_bits}) ---", flush=True)

    # Build a centroid LUT: linearly spaced over [0, 1] snapped to E4M3 (matches the
    # synthetic test in test-turbo-fp8-reference.cpp).
    pairs = enumerate_e4m3_positive_with_bytes()

    def snap_pos(v):
        best = min(pairs, key=lambda p: abs(p[1] - v))
        return best[0]
    centroids = [snap_pos(k / max(1, n_centroids - 1)) for k in range(n_centroids)]
    print(f"  centroid bytes: {[hex(b) for b in centroids]}", flush=True)

    # Generate 32 fp32 values with realistic distribution
    rng = np.random.default_rng(42)
    values = rng.standard_normal(QK_TURBO_FP8).astype(np.float32) * 0.5

    # Scalar quantize → packed block
    scale_bits, qs_bytes, sign_bytes = quantize_block_ref(values, centroids, idx_bits)
    # Scalar dequantize → reference output
    ref_out = dequantize_block_ref(scale_bits, qs_bytes, sign_bytes, centroids, idx_bits)

    # Triton dequantize → kernel output
    qs_n_bytes = (QK_TURBO_FP8 * idx_bits + 7) // 8
    # Pad qs by 1 byte so the kernel's byte_lo+1 load doesn't OOB
    scale_t = torch.tensor([scale_bits], dtype=torch.int16, device=device).view(torch.uint16) \
        if hasattr(torch, "uint16") else torch.tensor([scale_bits], dtype=torch.int16, device=device)
    # Triton uses int8 / uint8 — pack as uint8 tensor
    qs_t    = torch.tensor(list(qs_bytes)  + [0] * 4, dtype=torch.uint8, device=device)  # +pad
    signs_t = torch.tensor(list(sign_bytes), dtype=torch.uint8, device=device)
    lut_t   = torch.tensor(centroids, dtype=torch.uint8, device=device)
    out_t   = torch.zeros(QK_TURBO_FP8, dtype=torch.float32, device=device)
    scale_t = torch.tensor([scale_bits], dtype=torch.int16, device=device)

    turbo_fp8_decode_kernel[(1,)](
        scale_t, qs_t, signs_t, lut_t, out_t,
        N_CENTROIDS=n_centroids, IDX_BITS=idx_bits,
    )
    torch.cuda.synchronize()
    triton_out = out_t.cpu().numpy()

    # Compare
    max_err = float(np.max(np.abs(triton_out - ref_out)))
    rms_err = float(np.sqrt(np.mean((triton_out - ref_out) ** 2)))
    n_mismatch = int(np.sum(np.abs(triton_out - ref_out) > 1e-4))
    print(f"  max_err={max_err:.6g}  rms_err={rms_err:.6g}  mismatches(>1e-4)={n_mismatch} / {QK_TURBO_FP8}", flush=True)
    if n_mismatch:
        print("  --- first 10 mismatches ---", flush=True)
        for i, (r, t) in enumerate(zip(ref_out, triton_out)):
            if abs(r - t) > 1e-4:
                print(f"    [{i:2d}] ref={r:.6g}  triton={t:.6g}  diff={t-r:+.6g}", flush=True)
                if i >= 10:
                    break
    return n_mismatch == 0


def main():
    if not torch.cuda.is_available():
        print("CUDA / HIP device not available.")
        return 1
    device = torch.device("cuda")
    print(f"# device: {torch.cuda.get_device_name(0)}", flush=True)

    ok = True
    ok &= run_variant("turbo3_fp8", 8,  3, device)
    ok &= run_variant("turbo4_fp8", 16, 4, device)
    ok &= run_variant("turbo5_fp8", 32, 5, device)

    print("")
    if ok:
        print("=== ALL VARIANTS PASS ===")
        return 0
    print("=== SOME VARIANTS FAILED ===")
    return 1


if __name__ == "__main__":
    sys.exit(main())
