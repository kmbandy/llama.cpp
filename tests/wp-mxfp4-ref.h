// Shared by wp-mxfp4-gemm-test.cpp / wp-mxfp4-gemm-bench.cpp.
// ref_vec_dot_mxfp4_q8_0 is the mainline ggml_vec_dot_mxfp4_q8_0 AVX2 path
// (ggml/src/ggml-cpu/arch/x86/quants.c) copied verbatim, plus its scalar tail.
#pragma once
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "simd-mappings.h"
#include <immintrin.h>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
alignas(16) static const int8_t kvalues_fp4[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
float ggml_table_f32_e8m0_half[1 << 8];
float ggml_table_f32_f16[1 << 16];
static void ref_init_tables() {
    for (int i = 0; i < (1 << 16); ++i) { union { uint16_t u; ggml_fp16_t h; } u = {(uint16_t) i}; ggml_table_f32_f16[i] = _cvtsh_ss(u.u); }
    for (int i = 0; i < 256; ++i) ggml_table_f32_e8m0_half[i] = GGML_E8M0_TO_FP32_HALF(i);
}
extern "C" bool wp_gemm_mxfp4_q8_0(int, int, int, const void *, size_t, const void *, size_t, float *, size_t);

static inline float ref_hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}
static inline __m256i ref_mul_add_epi8(const __m256i x, const __m256i y) {
    const __m256i ax = _mm256_sign_epi8(x, x);
    const __m256i sy = _mm256_sign_epi8(y, x);
    return _mm256_maddubs_epi16(ax, sy);
}

static void ref_vec_dot_mxfp4_q8_0(int n, float * s, const void * vx, const void * vy) {
    const block_mxfp4 * x = (const block_mxfp4 *) vx;
    const block_q8_0 * y = (const block_q8_0 *) vy;
    const int nb = n / QK_MXFP4;
    int ib = 0;
    float sumf = 0;
    const __m128i values128 = _mm_loadu_si128((const __m128i*)kvalues_fp4);
    const __m128i m4b  = _mm_set1_epi8(0x0f);
    const __m256i mone = _mm256_set1_epi16(1);
    __m256 accum1 = _mm256_setzero_ps();
    __m256 accum2 = _mm256_setzero_ps();
    for (; ib + 1 < nb; ib += 2) {
        const __m128i q4bits_1 = _mm_loadu_si128((const __m128i*)x[ib + 0].qs);
        const __m128i q4bits_2 = _mm_loadu_si128((const __m128i*)x[ib + 1].qs);
        const __m256i q8b_1 = _mm256_loadu_si256((const __m256i *)y[ib + 0].qs);
        const __m256i q8b_2 = _mm256_loadu_si256((const __m256i *)y[ib + 1].qs);
        const __m256i q4b_1 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_1, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_1, m4b)));
        const __m256i q4b_2 = MM256_SET_M128I(_mm_shuffle_epi8(values128, _mm_and_si128(_mm_srli_epi16(q4bits_2, 4), m4b)),
                                              _mm_shuffle_epi8(values128, _mm_and_si128(q4bits_2, m4b)));
        const __m256i p16_1 = ref_mul_add_epi8(q4b_1, q8b_1);
        const __m256i p16_2 = ref_mul_add_epi8(q4b_2, q8b_2);
        const __m256i p_1 = _mm256_madd_epi16(p16_1, mone);
        const __m256i p_2 = _mm256_madd_epi16(p16_2, mone);
        const __m256 scale0 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 0].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 0].e));
        const __m256 scale1 = _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(y[ib + 1].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib + 1].e));
        accum1 = _mm256_fmadd_ps(scale0, _mm256_cvtepi32_ps(p_1), accum1);
        accum2 = _mm256_fmadd_ps(scale1, _mm256_cvtepi32_ps(p_2), accum2);
    }
    sumf = ref_hsum_float_8(_mm256_add_ps(accum1, accum2));
    for (; ib < nb; ++ib) {
        const float d = GGML_CPU_FP16_TO_FP32(y[ib].d)*GGML_CPU_E8M0_TO_FP32_HALF(x[ib].e);
        int sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < QK_MXFP4/2; ++j) {
            sumi1 += y[ib].qs[j +          0] * kvalues_fp4[x[ib].qs[j] & 0xf];
            sumi2 += y[ib].qs[j + QK_MXFP4/2] * kvalues_fp4[x[ib].qs[j] >>  4];
        }
        sumf += d * (sumi1 + sumi2);
    }
    *s = sumf;
}
