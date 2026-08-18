#include "pipe-reduce-simd.h"

#include <cstring>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)
#define PIPE_SIMD_X86_64 1
#endif

#if defined(PIPE_SIMD_X86_64) && (defined(__GNUC__) || defined(__clang__))
#define PIPE_SIMD_HAVE_DISPATCH 1
#include <immintrin.h>
#endif

namespace {

// -----------------------------------------------------------------------
// Scalar reference path. Bit-for-bit the same widening algorithm as
// ggml_compute_fp16_to_fp32 (ggml/src/ggml-impl.h) -- IEEE-754 half -> float
// is an exact (lossless) conversion, so any correct implementation, scalar
// or vectorized, software or hardware (F16C), produces identical bits. This
// copy keeps pipe-reduce-simd self-contained (no dependency on ggml's
// private headers) rather than because the math differs.
inline float fp16_to_fp32_scalar(uint16_t h) {
    const uint32_t w    = (uint32_t) h << 16;
    const uint32_t sign = w & 0x80000000u;
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = 0xE0u << 23;
    const float exp_scale = 0x1.0p-112f;
    float normalized_value;
    {
        uint32_t bits = (two_w >> 4) + exp_offset;
        std::memcpy(&normalized_value, &bits, sizeof(bits));
    }
    normalized_value *= exp_scale;

    const uint32_t magic_mask = 126u << 23;
    const float magic_bias = 0.5f;
    float denormalized_value;
    {
        uint32_t bits = (two_w >> 17) | magic_mask;
        std::memcpy(&denormalized_value, &bits, sizeof(bits));
    }
    denormalized_value -= magic_bias;

    const uint32_t denormalized_cutoff = 1u << 27;
    uint32_t normalized_bits, denormalized_bits;
    std::memcpy(&normalized_bits, &normalized_value, sizeof(normalized_bits));
    std::memcpy(&denormalized_bits, &denormalized_value, sizeof(denormalized_bits));
    const uint32_t result = sign | (two_w < denormalized_cutoff ? denormalized_bits : normalized_bits);

    float out;
    std::memcpy(&out, &result, sizeof(result));
    return out;
}

void accumulate_f32_scalar(float * acc, const float * add, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        acc[i] += add[i];
    }
}

void convert_f16_to_f32_scalar(float * out, const uint16_t * half, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        out[i] = fp16_to_fp32_scalar(half[i]);
    }
}

void decode_f16_accumulate_scalar(float * acc, const uint16_t * half, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        acc[i] += fp16_to_fp32_scalar(half[i]);
    }
}

#if defined(PIPE_SIMD_HAVE_DISPATCH)

// -----------------------------------------------------------------------
// AVX2 + F16C. Processes 8 floats / 8 halfs per iteration; scalar tail for
// n % 8 != 0. _mm256_cvtph_ps is an exact hardware half->float widening
// (IEEE-754), so this is bit-identical to fp16_to_fp32_scalar for every
// input -- no rounding-mode dependency (widening never rounds). These are
// compiled in unconditionally via the GCC/Clang `target()` attribute
// (function multiversioning), independent of the translation unit's
// baseline -march -- runtime dispatch below decides whether they're ever
// called.
__attribute__((target("avx2,f16c,fma")))
static void accumulate_f32_avx2(float * acc, const float * add, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(acc + i);
        __m256 b = _mm256_loadu_ps(add + i);
        _mm256_storeu_ps(acc + i, _mm256_add_ps(a, b));
    }
    for (; i < n; ++i) {
        acc[i] += add[i];
    }
}

__attribute__((target("avx2,f16c,fma")))
static void convert_f16_to_f32_avx2(float * out, const uint16_t * half, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *) (half + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(out + i, f);
    }
    for (; i < n; ++i) {
        out[i] = fp16_to_fp32_scalar(half[i]);
    }
}

__attribute__((target("avx2,f16c,fma")))
static void decode_f16_accumulate_avx2(float * acc, const uint16_t * half, size_t n) {
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i h = _mm_loadu_si128((const __m128i *) (half + i));
        __m256 f = _mm256_cvtph_ps(h);
        __m256 a = _mm256_loadu_ps(acc + i);
        _mm256_storeu_ps(acc + i, _mm256_add_ps(a, f));
    }
    for (; i < n; ++i) {
        acc[i] += fp16_to_fp32_scalar(half[i]);
    }
}

// AVX-512F path: 16 floats / 16 halfs per iteration. Only ever reached if
// detect_isa() below saw __builtin_cpu_supports("avx512f") at runtime.
__attribute__((target("avx512f,f16c,fma")))
static void accumulate_f32_avx512(float * acc, const float * add, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 a = _mm512_loadu_ps(acc + i);
        __m512 b = _mm512_loadu_ps(add + i);
        _mm512_storeu_ps(acc + i, _mm512_add_ps(a, b));
    }
    for (; i < n; ++i) {
        acc[i] += add[i];
    }
}

__attribute__((target("avx512f,f16c,fma")))
static void convert_f16_to_f32_avx512(float * out, const uint16_t * half, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *) (half + i));
        __m512 f = _mm512_cvtph_ps(h);
        _mm512_storeu_ps(out + i, f);
    }
    for (; i < n; ++i) {
        out[i] = fp16_to_fp32_scalar(half[i]);
    }
}

__attribute__((target("avx512f,f16c,fma")))
static void decode_f16_accumulate_avx512(float * acc, const uint16_t * half, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m256i h = _mm256_loadu_si256((const __m256i *) (half + i));
        __m512 f = _mm512_cvtph_ps(h);
        __m512 a = _mm512_loadu_ps(acc + i);
        _mm512_storeu_ps(acc + i, _mm512_add_ps(a, f));
    }
    for (; i < n; ++i) {
        acc[i] += fp16_to_fp32_scalar(half[i]);
    }
}

// -----------------------------------------------------------------------
// Runtime dispatch. Resolved once (function-local statics are
// thread-safe-initialized in C++11+ -- "magic statics") and cached, so the
// hot per-partial call site pays only one indirect-ish switch, not a cpuid
// probe.

enum class pipe_simd_isa { scalar, avx2, avx512 };

pipe_simd_isa detect_isa() {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("f16c")) {
        return pipe_simd_isa::avx512;
    }
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        return pipe_simd_isa::avx2;
    }
    return pipe_simd_isa::scalar;
}

pipe_simd_isa isa() {
    static const pipe_simd_isa cached = detect_isa();
    return cached;
}

#endif // PIPE_SIMD_HAVE_DISPATCH

} // namespace

extern "C" void pipe_simd_accumulate_f32(float * acc, const float * add, size_t n) {
    if (n == 0) {
        return;
    }
#if defined(PIPE_SIMD_HAVE_DISPATCH)
    switch (isa()) {
        case pipe_simd_isa::avx512:
            accumulate_f32_avx512(acc, add, n);
            return;
        case pipe_simd_isa::avx2:
            accumulate_f32_avx2(acc, add, n);
            return;
        case pipe_simd_isa::scalar:
            break;
    }
#endif
    accumulate_f32_scalar(acc, add, n);
}

extern "C" void pipe_simd_convert_f16_to_f32(float * out, const uint16_t * half, size_t n) {
    if (n == 0) {
        return;
    }
#if defined(PIPE_SIMD_HAVE_DISPATCH)
    switch (isa()) {
        case pipe_simd_isa::avx512:
            convert_f16_to_f32_avx512(out, half, n);
            return;
        case pipe_simd_isa::avx2:
            convert_f16_to_f32_avx2(out, half, n);
            return;
        case pipe_simd_isa::scalar:
            break;
    }
#endif
    convert_f16_to_f32_scalar(out, half, n);
}

extern "C" void pipe_simd_decode_f16_accumulate(float * acc, const uint16_t * half, size_t n) {
    if (n == 0) {
        return;
    }
#if defined(PIPE_SIMD_HAVE_DISPATCH)
    switch (isa()) {
        case pipe_simd_isa::avx512:
            decode_f16_accumulate_avx512(acc, half, n);
            return;
        case pipe_simd_isa::avx2:
            decode_f16_accumulate_avx2(acc, half, n);
            return;
        case pipe_simd_isa::scalar:
            break;
    }
#endif
    decode_f16_accumulate_scalar(acc, half, n);
}

extern "C" int pipe_simd_unpack_enabled(void) {
    static const int enabled = []() {
        const char * e = getenv("WP_SIMD_UNPACK");
        return e && atoi(e) != 0 ? 1 : 0;
    }();
    return enabled;
}
