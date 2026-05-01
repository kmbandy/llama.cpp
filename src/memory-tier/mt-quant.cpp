#include "mt-quant.h"

#include <algorithm>
#include <cmath>

namespace mt {

namespace {

// 4-bit signed range [-8, +7]. Encoded with a +8 bias so the byte's
// low/high nibble holds an unsigned 4-bit value in [0, 15].
constexpr int  kInt4Min      = -8;
constexpr int  kInt4Max      =  7;
constexpr int  kInt4Bias     =  8;
constexpr float kInt4Scale   = 7.0f;   // map [-1, +1] -> [-7, +7] (avoids saturating to -8)

// 8-bit signed range [-128, +127]. Encoded with a +128 bias.
constexpr int   kInt8Min   = -128;
constexpr int   kInt8Max   =  127;
constexpr int   kInt8Bias  =  128;
constexpr float kInt8Scale = 127.0f;

inline int clamp_int(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline uint8_t encode_int4(float x) {
    int q = (int) std::round(x * kInt4Scale);
    q = clamp_int(q, kInt4Min, kInt4Max);
    return (uint8_t)((q + kInt4Bias) & 0x0F);
}

inline float decode_int4(uint8_t nibble) {
    int q = (int)(nibble & 0x0F) - kInt4Bias;
    return (float) q / kInt4Scale;
}

inline uint8_t encode_int8(float x) {
    int q = (int) std::round(x * kInt8Scale);
    q = clamp_int(q, kInt8Min, kInt8Max);
    return (uint8_t)((q + kInt8Bias) & 0xFF);
}

inline float decode_int8(uint8_t v) {
    int q = (int) v - kInt8Bias;
    return (float) q / kInt8Scale;
}

}  // namespace

std::vector<uint8_t> quantize_int4(const float * src, size_t n) {
    std::vector<uint8_t> out;
    if (n == 0 || src == nullptr) return out;
    out.resize((n + 1) / 2);

    for (size_t i = 0; i < n; i += 2) {
        const uint8_t lo = encode_int4(src[i]);
        const uint8_t hi = (i + 1 < n) ? encode_int4(src[i + 1]) : 0u;
        out[i / 2] = (uint8_t)(lo | (hi << 4));
    }
    return out;
}

bool dequantize_int4(const uint8_t * src, float * dst, size_t n) {
    if (n == 0) return true;
    if (src == nullptr || dst == nullptr) return false;

    for (size_t i = 0; i < n; i += 2) {
        const uint8_t byte = src[i / 2];
        dst[i] = decode_int4((uint8_t)(byte & 0x0F));
        if (i + 1 < n) {
            dst[i + 1] = decode_int4((uint8_t)((byte >> 4) & 0x0F));
        }
    }
    return true;
}

std::vector<uint8_t> quantize_int8(const float * src, size_t n) {
    std::vector<uint8_t> out;
    if (n == 0 || src == nullptr) return out;
    out.resize(n);
    for (size_t i = 0; i < n; ++i) {
        out[i] = encode_int8(src[i]);
    }
    return out;
}

bool dequantize_int8(const uint8_t * src, float * dst, size_t n) {
    if (n == 0) return true;
    if (src == nullptr || dst == nullptr) return false;
    for (size_t i = 0; i < n; ++i) {
        dst[i] = decode_int8(src[i]);
    }
    return true;
}

}  // namespace mt
