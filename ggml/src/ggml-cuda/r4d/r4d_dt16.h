// 16-bit operand traits. The QK and PV matmuls both take a 16-bit A/B pair; which 16-bit format is
// a free choice because BOTH sources are exactly representable in either:
//   * an fp8 e4m3 K/V has 3 mantissa bits, so widening to bf16 (7) or f16 (10) is exact;
//   * a bf16 query has 7, so bf16->f16 is exact for |q| in [6.1e-5, 65504].
// The formats are NOT equal in cost. gfx1201 has NO bf16 convert instruction, so every f32->bf16
// is ~6 VALU of software RTNE, while v_cvt_pkrtz_f16_f32 does TWO f32->f16 in ONE. On the softmax
// path that is 24 VALU per 8-element P fragment against 4. f16 also carries 3 more mantissa bits.
// What f16 gives up is exponent range, and that is exactly what GROW below prices.
#pragma once
#include "r4d_common.h"

typedef __bf16   v8bf __attribute__((ext_vector_type(8)));
typedef __fp16   v8hf __attribute__((ext_vector_type(8)));  // what the f16 WMMA builtin takes
typedef __fp16   v2hf __attribute__((ext_vector_type(2)));  // what cvt_pkrtz returns

// GROW = how many octaves the running row max may exceed the REFERENCE max before the
// accumulator has to be rescaled. p = 2^(s - m_ref) must stay inside the format, so GROW is
// log2(format max) with margin. This is the whole point of the lazy-rescale scheme: the eager
// online-softmax form pays 128 multiplies EVERY m-tile; with a reference max it pays them only
// when the max moves more than GROW, which at long context is almost never.
template<int F16> struct DT16;

template<> struct DT16<0> {                       // bf16: exponent range of f32
    typedef v8bf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 60.0f;         // 2^60 * |v| * n stays far under f32 max
    __device__ __forceinline__ static frag mk(uint2 lo, uint2 hi) {
        union { uint32_t w[4]; frag f; } u;       // union, never a pointer cast
        u.w[0] = lo.x; u.w[1] = lo.y; u.w[2] = hi.x; u.w[3] = hi.y; return u.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
    }
    __device__ __forceinline__ static uint32_t pk(float x, float y) {
        return (uint32_t)f32_to_bf16(x) | ((uint32_t)f32_to_bf16(y) << 16);
    }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {   // what cvt_pk_f32_fp8 returns
        union { v2f f; uint32_t u[2]; } q; q.f = x;
        return __builtin_amdgcn_perm(q.u[1], q.u[0], 0x07060302u);
    }
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) { return w; }  // identity
};

template<> struct DT16<1> {                       // f16: 3 more mantissa bits, 1 convert instead of 6
    typedef v8hf frag;
    static constexpr float SHIFT = 0.0f;
    static constexpr float GROW  = 14.0f;         // p <= 2^14 = 16384 < 65504
    __device__ __forceinline__ static frag mk(uint2 lo, uint2 hi) {
        union { uint32_t w[4]; frag f; } u;
        u.w[0] = lo.x; u.w[1] = lo.y; u.w[2] = hi.x; u.w[3] = hi.y; return u.f;
    }
    __device__ __forceinline__ static v8f wmma(frag a, frag b, v8f c) {
        return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    }
    __device__ __forceinline__ static uint32_t pk(float x, float y) {
        union { v2hf h; uint32_t u; } q; q.h = __builtin_amdgcn_cvt_pkrtz(x, y); return q.u;
    }
    __device__ __forceinline__ static uint32_t pkv(v2f x) {
        union { v2f f; float e[2]; } s; s.f = x;
        union { v2hf h; uint32_t u; } q; q.h = __builtin_amdgcn_cvt_pkrtz(s.e[0], s.e[1]); return q.u;
    }
    // two bf16 in a dword -> two f16 in a dword. bf16->f32 is a shift, so this is 3 VALU.
    __device__ __forceinline__ static uint32_t from_bf16w(uint32_t w) {
        union { v2hf h; uint32_t u; } q;
        q.h = __builtin_amdgcn_cvt_pkrtz(__builtin_bit_cast(float, w << 16),
                                         __builtin_bit_cast(float, w & 0xffff0000u));
        return q.u;
    }
};

// Widen 8 fp8 (2 dwords) to 8 16-bit values (4 dwords) in element order.
template<class D>
__device__ __forceinline__ void fp8x8_to_16x4w(uint2 raw, uint32_t* w4) {
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int wv = (q == 0) ? (int)raw.x : (int)raw.y;
        w4[q * 2 + 0] = D::pkv(__builtin_amdgcn_cvt_pk_f32_fp8(wv, false));
        w4[q * 2 + 1] = D::pkv(__builtin_amdgcn_cvt_pk_f32_fp8(wv, true));
    }
}

__device__ __forceinline__ int wave_any(bool c) {
    return (int)__builtin_amdgcn_ballot_w32(c);
}
