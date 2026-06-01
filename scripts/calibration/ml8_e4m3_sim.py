# scripts/calibration/ml8_e4m3_sim.py
"""Bit-exact Python mirror of the ml8 activation e4m3 path.

Ground truth = the CUDA kernel ml8_fp32_to_e4m3 (ggml/src/ggml-cuda/ml8.cu:440),
NOT the C ref quantize_row_f8_e4m3_ref (which carries the old e_out>=15 bug).
"""
import struct
import torch

E4M3_MAX = 448.0
ACT_SCALE_EPS = 1e-12   # matches ML8_ACT_SCALE_EPS

def fp32_to_e4m3_bits(xv: float) -> int:
    """Return the uint8 e4m3 code for one fp32, RNE, saturating, e4m3fn."""
    bits = struct.unpack("<I", struct.pack("<f", xv))[0]
    sign  = (bits >> 31) & 1
    exp_b = (bits >> 23) & 0xFF
    mant  = bits & 0x7FFFFF
    if exp_b == 0xFF:           # NaN/Inf -> e4m3 NaN
        return (sign << 7) | 0x7F
    if exp_b == 0:              # zero / fp32 subnormal -> zero
        return sign << 7
    e_un = exp_b - 127
    if e_un >= 9 or (e_un == 8 and mant >= 0x600000):   # saturate ±448
        return (sign << 7) | (0xF << 3) | 0x6
    if e_un >= -6:              # normal e4m3
        guard  = (mant >> 19) & 1
        sticky = 1 if (mant & ((1 << 19) - 1)) else 0
        lsb    = (mant >> 20) & 1
        m_e4m3 = (mant >> 20) & 0x7
        if guard and (sticky or lsb):
            m_e4m3 += 1
        e_out = e_un + 7
        if m_e4m3 == 8:
            m_e4m3 = 0
            e_out += 1
            if e_out > 15:
                return (sign << 7) | (0xF << 3) | 0x6
        if e_out == 15 and m_e4m3 == 7:
            m_e4m3 = 6
        return (sign << 7) | (e_out << 3) | m_e4m3
    shift = 23 - (e_un + 9)     # subnormal e4m3
    if shift > 31:
        return sign << 7
    implicit = (1 << 23) | mant
    guard  = (implicit >> (shift - 1)) & 1
    sticky = 1 if (implicit & ((1 << (shift - 1)) - 1)) else 0
    m_e4m3 = implicit >> shift
    lsb    = m_e4m3 & 1
    if guard and (sticky or lsb):
        m_e4m3 += 1
    if m_e4m3 >= 8:
        return (sign << 7) | (1 << 3)
    return (sign << 7) | m_e4m3

def e4m3_bits_to_fp32(code: int) -> float:
    """Decode a uint8 e4m3fn code to fp32 (NaN slot -> nan)."""
    sign = -1.0 if (code & 0x80) else 1.0
    e = (code >> 3) & 0xF
    m = code & 0x7
    if e == 0:
        return sign * (m / 8.0) * (2.0 ** -6)     # subnormal
    if e == 15 and m == 7:
        return float("nan")
    return sign * (1.0 + m / 8.0) * (2.0 ** (e - 7))
