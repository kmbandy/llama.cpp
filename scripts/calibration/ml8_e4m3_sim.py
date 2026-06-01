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

# append to ml8_e4m3_sim.py
@torch.no_grad()
def e4m3_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """Vectorized fp32 -> e4m3 -> fp32, bit-identical to the scalar path."""
    orig_dtype = x.dtype
    xf = x.to(torch.float32).contiguous()
    bits = xf.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign  = (bits >> 31) & 1
    exp_b = (bits >> 23) & 0xFF
    mant  = bits & 0x7FFFFF
    e_un  = exp_b - 127

    out_code = torch.zeros_like(bits)
    is_nan_inf = exp_b == 0xFF
    is_zero    = exp_b == 0
    is_sat     = (e_un >= 9) | ((e_un == 8) & (mant >= 0x600000))
    is_normal  = (~is_nan_inf) & (~is_zero) & (~is_sat) & (e_un >= -6)
    is_sub     = (~is_nan_inf) & (~is_zero) & (~is_sat) & (e_un < -6)

    sat_code = (sign << 7) | (0xF << 3) | 0x6
    # normal
    guard  = (mant >> 19) & 1
    sticky = ((mant & ((1 << 19) - 1)) != 0).to(torch.int64)
    lsb    = (mant >> 20) & 1
    m_n    = (mant >> 20) & 0x7
    m_n    = m_n + (guard & (sticky | lsb))
    e_n    = e_un + 7
    carry  = (m_n == 8)
    m_n    = torch.where(carry, torch.zeros_like(m_n), m_n)
    e_n    = torch.where(carry, e_n + 1, e_n)
    normal_overflow = carry & (e_n > 15)
    nan_fix = (e_n == 15) & (m_n == 7)
    m_n = torch.where(nan_fix, torch.full_like(m_n, 6), m_n)
    normal_code = (sign << 7) | (e_n << 3) | m_n
    normal_code = torch.where(normal_overflow, sat_code, normal_code)
    # subnormal
    shift = (23 - (e_un + 9)).clamp(min=0)
    too_small = (23 - (e_un + 9)) > 31
    implicit = (1 << 23) | mant
    sh1 = (shift - 1).clamp(min=0)
    g_s = (implicit >> sh1) & 1
    st_s = ((implicit & ((1 << sh1) - 1)) != 0).to(torch.int64)
    m_s = implicit >> shift
    lsb_s = m_s & 1
    m_s = m_s + (g_s & (st_s | lsb_s))
    sub_overflow = m_s >= 8
    sub_code = (sign << 7) | m_s
    sub_code = torch.where(sub_overflow, (sign << 7) | (1 << 3), sub_code)
    sub_code = torch.where(too_small, sign << 7, sub_code)

    out_code = torch.where(is_nan_inf, (sign << 7) | 0x7F, out_code)
    out_code = torch.where(is_zero, sign << 7, out_code)
    out_code = torch.where(is_sat, sat_code, out_code)
    out_code = torch.where(is_normal, normal_code, out_code)
    out_code = torch.where(is_sub, sub_code, out_code)

    # decode
    c = out_code
    s = torch.where((c & 0x80) != 0, torch.full_like(xf, -1.0), torch.ones_like(xf))
    e = ((c >> 3) & 0xF).to(torch.float32)
    m = (c & 0x7).to(torch.float32)
    sub_val = s * (m / 8.0) * (2.0 ** -6)
    nan_slot = (e == 15) & ((c & 0x7) == 7)
    norm_val = s * (1.0 + m / 8.0) * torch.pow(torch.tensor(2.0), e - 7)
    val = torch.where(e == 0, sub_val, norm_val)
    val = torch.where(nan_slot, torch.full_like(val, float("nan")), val)
    return val.to(orig_dtype)

@torch.no_grad()
def quantize_act_per_row(x: torch.Tensor) -> torch.Tensor:
    """Per-row (per-token) e4m3 activation quant, kernel-faithful.

    x: [..., K]; the last dim is K. scale = row_absmax / 448 (eps-floored);
    returns dequantized fp32 a_fp8*scale, same shape & dtype as x.
    """
    orig_dtype = x.dtype
    xf = x.to(torch.float32)
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(ACT_SCALE_EPS)
    scale = absmax / E4M3_MAX
    q = e4m3_roundtrip(xf / scale) * scale
    return q.to(orig_dtype)
