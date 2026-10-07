// CPU ml8_4 wire pack, bit-identical to the GPU kernel ml8_4_quantize()
// (ggml/src/ggml-cuda/allreduce-ml8.cuh). Header-only so the exactness test can
// include it without linking libllama.
//
// Numerics that matter (verified byte-for-byte by tests/test-ml8-4-pack-cpu-vs-gpu):
//  - scale = fp16 RNE of block absmax (NaN ignored by max);  d = float(scale)
//  - inv = d > 0 ? 1/d (IEEE divide) : 0
//  - the GPU compiles fabsf(v*inv - c) with FMA contraction, i.e. |fma(v, inv, -c)|,
//    so the CPU must use a fused multiply-add too (single rounding).
//  - nearest centroid: strict '<' scan over the 16 centroids, first index wins ties;
//    NaN compares false -> index 0.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include <immintrin.h>
#define PIPE_ML8_PACK_AVX2 1
#endif

namespace pipe_ml8 {

static const float k_cent4[16] = {
    -0.93750000f, -0.75000000f, -0.56250000f, -0.43750000f, -0.31250000f, -0.21875000f, -0.12500000f, -0.03906250f,
    +0.04296875f, +0.12500000f, +0.21875000f, +0.31250000f, +0.43750000f, +0.56250000f, +0.75000000f, +0.93750000f,
};

// Scalar reference (also the non-AVX2 fallback). fp16 conversion done in software, RNE.
static inline uint16_t f32_to_f16_rne(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7fffffffu;
    if (ax >= 0x7f800000u) {                       // inf / nan
        return (uint16_t) (sign | (ax > 0x7f800000u ? 0x7e00u : 0x7c00u));
    }
    if (ax >= 0x477ff000u) {                       // rounds to >= 65520 -> inf
        return (uint16_t) (sign | 0x7c00u);
    }
    if (ax < 0x38800000u) {                        // fp16 subnormal / zero
        if (ax < 0x33000000u) { return (uint16_t) sign; }
        const int e = (int) (ax >> 23);            // 102..112
        const uint32_t m = (ax & 0x7fffffu) | 0x800000u;
        const int sh = 126 - e;                    // 14..24
        uint32_t r = m >> sh;
        const uint32_t rem = m & ((1u << sh) - 1u);
        const uint32_t half = 1u << (sh - 1);
        if (rem > half || (rem == half && (r & 1u))) { r++; }
        return (uint16_t) (sign | r);
    }
    uint32_t r = ((ax - 0x38000000u) >> 13);       // rebias exponent, truncate
    const uint32_t rem = ax & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) { r++; }
    return (uint16_t) (sign | r);
}

static inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ffu, out;
    if (e == 0) {
        if (m == 0) { out = sign; }
        else {
            int s = 0; while (!(m & 0x400u)) { m <<= 1; s++; }
            m &= 0x3ffu;
            out = sign | ((uint32_t) (113 - s) << 23) | (m << 13);
        }
    } else if (e == 31) {
        out = sign | 0x7f800000u | (m << 13);
    } else {
        out = sign | ((e + 112) << 23) | (m << 13);
    }
    float f; std::memcpy(&f, &out, 4); return f;
}

static inline void pack_block_scalar(const float * v, uint8_t * blk) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float a = std::fabs(v[j]);
        amax = a > amax ? a : amax;                // NaN -> keeps amax (fmaxf semantics)
    }
    const uint16_t h = f32_to_f16_rne(amax);
    const float d = f16_to_f32(h);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    std::memcpy(blk, &h, 2);
    uint8_t * qs = blk + 2;
    for (int j = 0; j < 32; j += 2) {
        int idx[2];
        for (int k = 0; k < 2; ++k) {
            int best = 0;
            float bd = std::fabs(std::fmaf(v[j + k], inv, -k_cent4[0]));
            for (int i = 1; i < 16; ++i) {
                const float dd = std::fabs(std::fmaf(v[j + k], inv, -k_cent4[i]));
                if (dd < bd) { bd = dd; best = i; }
            }
            idx[k] = best;
        }
        qs[j / 2] = (uint8_t) (idx[0] | (idx[1] << 4));
    }
}

#ifdef PIPE_ML8_PACK_AVX2
static inline void pack_block_avx2(const float * v, uint8_t * blk, const __m256 * cneg) {
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 x0 = _mm256_loadu_ps(v), x1 = _mm256_loadu_ps(v + 8),
                 x2 = _mm256_loadu_ps(v + 16), x3 = _mm256_loadu_ps(v + 24);
    // max_ps returns its 2nd operand if either is NaN -> keep acc second.
    __m256 m = _mm256_max_ps(_mm256_and_ps(x0, absmask), _mm256_setzero_ps());
    m = _mm256_max_ps(_mm256_and_ps(x1, absmask), m);
    m = _mm256_max_ps(_mm256_and_ps(x2, absmask), m);
    m = _mm256_max_ps(_mm256_and_ps(x3, absmask), m);
    __m128 m4 = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
    m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
    const float amax = _mm_cvtss_f32(m4);
    const uint16_t h = (uint16_t) _cvtss_sh(amax, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const float d = _cvtsh_ss(h);
    const float inv = d > 0.0f ? 1.0f / d : 0.0f;
    std::memcpy(blk, &h, 2);
    const __m256 vinv = _mm256_set1_ps(inv);
    const __m256 xs[4] = { x0, x1, x2, x3 };
    alignas(32) int32_t idx[32];
    for (int q = 0; q < 4; ++q) {
        __m256 bd = _mm256_and_ps(_mm256_fmadd_ps(xs[q], vinv, cneg[0]), absmask);
        __m256i best = _mm256_setzero_si256();
        for (int i = 1; i < 16; ++i) {
            const __m256 dd = _mm256_and_ps(_mm256_fmadd_ps(xs[q], vinv, cneg[i]), absmask);
            const __m256 lt = _mm256_cmp_ps(dd, bd, _CMP_LT_OQ);
            bd = _mm256_blendv_ps(bd, dd, lt);
            best = _mm256_blendv_epi8(best, _mm256_set1_epi32(i), _mm256_castps_si256(lt));
        }
        _mm256_store_si256((__m256i *) (idx + 8 * q), best);
    }
    uint8_t * qs = blk + 2;
    for (int j = 0; j < 16; ++j) {
        qs[j] = (uint8_t) (idx[2 * j] | (idx[2 * j + 1] << 4));
    }
}
#endif

// n must be a multiple of 32. dst gets (n/32)*18 bytes.
static inline void pack_ml8_4(uint8_t * dst, const float * src, size_t n) {
    const size_t nb = n / 32;
#ifdef PIPE_ML8_PACK_AVX2
    __m256 cneg[16];
    for (int i = 0; i < 16; ++i) { cneg[i] = _mm256_set1_ps(-k_cent4[i]); }
    for (size_t b = 0; b < nb; ++b) {
        pack_block_avx2(src + b * 32, dst + b * 18, cneg);
    }
#else
    for (size_t b = 0; b < nb; ++b) {
        pack_block_scalar(src + b * 32, dst + b * 18);
    }
#endif
}

} // namespace pipe_ml8
