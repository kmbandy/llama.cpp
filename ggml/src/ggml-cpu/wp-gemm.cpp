// Multi-column AVX2 GEMM for the CPU expert tier.
//
// WHY THIS EXISTS. Mainline's ggml_vec_dot_q4_K_q8_K computes ONE output
// element per call: it decodes the super-block scales and unpacks the 4-bit
// weight nibbles, uses them against a single q8_K column, and throws the
// unpacked bits away. For a matmul with T columns it therefore repeats the
// row-only work T times. Measured on a Ryzen 9 3900X at the qwen38-next expert
// geometry (gate/up q4_K [2560,448], down q5_1 [448,2560]), single-threaded
// cost per token was FLAT in T -- 40.5 us/token at T=1 and at T=256, a 1.04x
// "speedup" from 256x more work. There is no weight reuse at all. The same
// control showed q5_1 at 66.8 us/token and 1.00x: literally zero amortization.
//
// This file hoists the row-only work (scale decode + nibble unpack) out of the
// column loop, exactly the technique ik_llama.cpp's iqk kernels use. The
// arithmetic per output element is UNCHANGED and the summation order within a
// dot product is unchanged, so results are bit-identical to the scalar path,
// not merely close -- see wp_gemm_selfcheck in the bench.
//
// AVX2 has 16 YMM registers. NY is capped at 4 deliberately: at NY=8 the
// accumulators (acc[8] + sumi[8]) alone exceed the register file and the
// compiler spills, which measured SLOWER than NY=4. ik uses 8 only on AVX512
// where there are 32 registers. Do not raise NY without re-measuring.
//
// The mins/bsums term is accumulated as a scalar float rather than a __m128
// per column: it is touched once per super-block (not once per 64-element
// group), so it is not worth the registers.

#include "wp-gemm.h"

#include <cstdlib>

#include <cstdint>
#include <cstring>

#if defined(__AVX2__) && defined(__FMA__)

#include <immintrin.h>

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

namespace {

inline float hsum_f32_8(__m256 x) {
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    r = _mm_add_ps(r, _mm_movehl_ps(r, r));
    r = _mm_add_ss(r, _mm_movehdup_ps(r));
    return _mm_cvtss_f32(r);
}

// Broadcast super-block scale i (a uint16 lane) across all 16 int16 lanes.
// Identical table to mainline's get_scale_shuffle_k4; shuffle_epi8 is per
// 128-bit lane, so the caller must duplicate the scale half into both lanes.
inline __m256i scale_shuffle(int i) {
    static const uint8_t k[256] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
         2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
         6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,
        14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15
    };
    return _mm256_loadu_si256((const __m256i *) k + i);
}

// One row of A against NY columns of B. The j-loop body loads and unpacks the
// weight nibbles ONCE and then consumes them for every column -- that hoist is
// the entire point of this file.
template <int NY>
void gemm_q4K_q8K_row(int nb,
                      const block_q4_K * __restrict a,
                      const block_q8_K * const __restrict b[NY],
                      float * __restrict out, size_t stride_C) {
    static const uint32_t kmask1 = 0x3f3f3f3f;
    static const uint32_t kmask2 = 0x0f0f0f0f;
    static const uint32_t kmask3 = 0x03030303;

    const __m256i m4 = _mm256_set1_epi8(0xF);

    __m256 acc[NY];
    __m128 accm[NY];
    for (int iy = 0; iy < NY; ++iy) { acc[iy] = _mm256_setzero_ps(); accm[iy] = _mm_setzero_ps(); }

    uint32_t utmp[4];

    for (int i = 0; i < nb; ++i) {
        // ---- row-only work: decode the 6-bit packed scales and mins ----
        std::memcpy(utmp, a[i].scales, 12);
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        const uint32_t uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        const __m256i mins_and_scales =
            _mm256_cvtepu8_epi16(_mm_set_epi32(utmp[3], utmp[2], utmp[1], utmp[0]));
        const __m128i mins128 = _mm256_extracti128_si256(mins_and_scales, 1);
        const __m128i sc128   = _mm256_extracti128_si256(mins_and_scales, 0);
        const __m256i scales  = _mm256_set_m128i(sc128, sc128);

        const float xd    = GGML_CPU_FP16_TO_FP32(a[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        const float xdmin = GGML_CPU_FP16_TO_FP32(a[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);

        // The mins term touches bsums only: once per super-block, per column.
        for (int iy = 0; iy < NY; ++iy) {
            const __m256i q8sums = _mm256_loadu_si256((const __m256i *) b[iy][i].bsums);
            const __m128i q8s    = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0),
                                                  _mm256_extracti128_si256(q8sums, 1));
            const __m128i prod   = _mm_madd_epi16(mins128, q8s);
            // Accumulate the four lanes as floats and fold only at the end --
            // this is mainline's exact order. Summing the lanes as integers
            // first and multiplying once is more accurate but NOT bit-identical,
            // and it showed up as a ~1e-6 drift through the graph.
            accm[iy] = _mm_fmadd_ps(_mm_set1_ps(-b[iy][i].d * xdmin),
                                    _mm_cvtepi32_ps(prod), accm[iy]);
        }

        __m256i sumi[NY];
        for (int iy = 0; iy < NY; ++iy) sumi[iy] = _mm256_setzero_si256();

        const uint8_t * __restrict q4 = a[i].qs;

        for (int j = 0; j < QK_K / 64; ++j) {
            // ---- row-only work, hoisted out of the column loop ----
            const __m256i scale_l = _mm256_shuffle_epi8(scales, scale_shuffle(2 * j + 0));
            const __m256i scale_h = _mm256_shuffle_epi8(scales, scale_shuffle(2 * j + 1));
            const __m256i q4bits  = _mm256_loadu_si256((const __m256i *) q4); q4 += 32;
            const __m256i q4l     = _mm256_and_si256(q4bits, m4);
            const __m256i q4h     = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

            // ---- per-column work, reusing the unpack above ----
            for (int iy = 0; iy < NY; ++iy) {
                const int8_t * __restrict q8 = b[iy][i].qs + 64 * j;
                __m256i p16l = _mm256_maddubs_epi16(q4l, _mm256_loadu_si256((const __m256i *) q8));
                p16l = _mm256_madd_epi16(scale_l, p16l);
                __m256i p16h = _mm256_maddubs_epi16(q4h, _mm256_loadu_si256((const __m256i *) (q8 + 32)));
                p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p16l, p16h));
            }
        }

        for (int iy = 0; iy < NY; ++iy) {
            acc[iy] = _mm256_fmadd_ps(_mm256_set1_ps(b[iy][i].d * xd),
                                      _mm256_cvtepi32_ps(sumi[iy]), acc[iy]);
        }
    }

    for (int iy = 0; iy < NY; ++iy) {
        __m128 m = accm[iy];
        m = _mm_add_ps(m, _mm_movehl_ps(m, m));
        m = _mm_add_ss(m, _mm_movehdup_ps(m));
        out[iy * stride_C] = hsum_f32_8(acc[iy]) + _mm_cvtss_f32(m);
    }
}


// Verbatim from mainline's arch/x86/quants.c so results stay bit-identical.
inline __m256i bytes_from_bits_32(const uint8_t * x) {
    uint32_t x32;
    std::memcpy(&x32, x, sizeof(uint32_t));
    const __m256i shuf_mask = _mm256_set_epi64x(
            0x0303030303030303, 0x0202020202020202,
            0x0101010101010101, 0x0000000000000000);
    __m256i bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(x32), shuf_mask);
    const __m256i bit_mask = _mm256_set1_epi64x(0x7fbfdfeff7fbfdfe);
    bytes = _mm256_or_si256(bytes, bit_mask);
    return _mm256_cmpeq_epi8(bytes, _mm256_set1_epi64x(-1));
}

inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp    = _mm_loadu_si128((const __m128i *) rsi);
    const __m256i bytes  = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
    const __m256i dot  = _mm256_maddubs_epi16(ax, sy);
    const __m256i ones = _mm256_set1_epi16(1);
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(ones, dot));
}

// q5_1 carries per-32-element scales, so unlike q4_K there is no scales array
// and no sumi array to keep live -- register pressure is low enough for 8
// columns. The hoisted work is larger here too: the nibble unpack AND the
// 32-bit-to-32-byte expansion of qh are both row-only, which is why the
// baseline amortized at exactly 1.00x.
template <int NY>
void gemm_q5_1_q8_1_row(int nb,
                        const block_q5_1 * __restrict a,
                        const block_q8_1 * const __restrict b[NY],
                        float * __restrict out, size_t stride_C) {
    __m256 acc[NY];
    float  summs[NY];
    for (int iy = 0; iy < NY; ++iy) { acc[iy] = _mm256_setzero_ps(); summs[iy] = 0.0f; }

    for (int ib = 0; ib < nb; ++ib) {
        // ---- row-only work, hoisted out of the column loop ----
        const float xd = GGML_CPU_FP16_TO_FP32(a[ib].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        const float xm = GGML_CPU_FP16_TO_FP32(a[ib].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        __m256i qx   = bytes_from_nibbles_32(a[ib].qs);
        __m256i bxhi = bytes_from_bits_32(a[ib].qh);
        bxhi = _mm256_and_si256(bxhi, _mm256_set1_epi8(0x10));
        qx   = _mm256_or_si256(qx, bxhi);

        // ---- per-column work, reusing the unpack above ----
        for (int iy = 0; iy < NY; ++iy) {
            const block_q8_1 & yb = b[iy][ib];
            summs[iy] += xm * GGML_CPU_FP16_TO_FP32(yb.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.s);
            const float dy = GGML_CPU_FP16_TO_FP32(yb.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
            const __m256i qy = _mm256_loadu_si256((const __m256i *) yb.qs);
            const __m256  q  = mul_sum_us8_pairs_float(qx, qy);
            acc[iy] = _mm256_fmadd_ps(q, _mm256_set1_ps(xd * dy), acc[iy]);
        }
    }

    for (int iy = 0; iy < NY; ++iy) {
        out[iy * stride_C] = hsum_f32_8(acc[iy]) + summs[iy];
    }
}

} // namespace

bool wp_gemm_q4K_q8K(int n, int nrc_x, int nrc_y,
                     const void * A, size_t bx,
                     const void * B, size_t by,
                     float * C, size_t stride_C) {
    if (n % QK_K != 0) {
        return false;
    }
    const int nb = n / QK_K;

    // COLUMN-BLOCK OUTER, ROW INNER. The other order is the obvious one and it
    // is wrong: with rows outer, the whole B panel (nrc_y q8_K columns, ~2.6 KB
    // each) is re-read once per row -- 298 MB of traffic at nrc_y=256, which
    // measured as a speedup DECAYING from 1.22x at 4 columns to 1.06x at 256.
    // With columns outer, one 4-column block is ~10 KB and stays L1-resident
    // across every row while A streams from L3 once per block.
    for (int iy = 0; iy + 4 <= nrc_y; iy += 4) {
        const block_q8_K * cols[4];
        for (int k = 0; k < 4; ++k) {
            cols[k] = (const block_q8_K *) ((const char *) B + (size_t) (iy + k) * by);
        }
        for (int ix = 0; ix < nrc_x; ++ix) {
            const block_q4_K * a = (const block_q4_K *) ((const char *) A + (size_t) ix * bx);
            gemm_q4K_q8K_row<4>(nb, a, cols, C + (size_t) iy * stride_C + ix, stride_C);
        }
    }
    for (int iy = nrc_y & ~3; iy < nrc_y; ++iy) {
        const block_q8_K * cols[1] = {
            (const block_q8_K *) ((const char *) B + (size_t) iy * by)
        };
        for (int ix = 0; ix < nrc_x; ++ix) {
            const block_q4_K * a = (const block_q4_K *) ((const char *) A + (size_t) ix * bx);
            gemm_q4K_q8K_row<1>(nb, a, cols, C + (size_t) iy * stride_C + ix, stride_C);
        }
    }
    return true;
}


bool wp_gemm_q5_1_q8_1(int n, int nrc_x, int nrc_y,
                       const void * A, size_t bx,
                       const void * B, size_t by,
                       float * C, size_t stride_C) {
    if (n % QK5_1 != 0) {
        return false;
    }
    const int nb = n / QK5_1;

    int iy = 0;
    // Column-block outer, row inner -- same cache argument as q4_K above.
    #define WP_Q5_1_BLOCK(W)                                                          \
        while (iy + (W) <= nrc_y) {                                                   \
            const block_q8_1 * cols[W];                                               \
            for (int k = 0; k < (W); ++k) {                                           \
                cols[k] = (const block_q8_1 *) ((const char *) B + (size_t) (iy + k) * by); \
            }                                                                         \
            for (int ix = 0; ix < nrc_x; ++ix) {                                      \
                const block_q5_1 * a =                                                \
                    (const block_q5_1 *) ((const char *) A + (size_t) ix * bx);        \
                gemm_q5_1_q8_1_row<W>(nb, a, cols, C + (size_t) iy * stride_C + ix, stride_C); \
            }                                                                         \
            iy += (W);                                                                \
        }
    WP_Q5_1_BLOCK(8)
    WP_Q5_1_BLOCK(4)
    WP_Q5_1_BLOCK(2)
    WP_Q5_1_BLOCK(1)
    #undef WP_Q5_1_BLOCK
    return true;
}

#else  // no AVX2

bool wp_gemm_q4K_q8K(int, int, int, const void *, size_t,
                     const void *, size_t, float *, size_t) {
    return false;
}

bool wp_gemm_q5_1_q8_1(int, int, int, const void *, size_t,
                       const void *, size_t, float *, size_t) {
    return false;
}

#endif

// DEFAULT OFF, deliberately. The kernels below are bit-exact and measurably
// faster in isolation (q4_K 1.23x, q5_1 1.99x on a 3900X), but they have not
// been A/B'd end-to-end on the rig yet, and an unmeasured-in-situ lever that is
// on by default stops a bare run from being the config of record.
bool wp_gemm_enabled(void) {
    static const bool enabled = [] {
        const char * e = std::getenv("WP_CPU_GEMM");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}
