// MAD-135: round-trip tests for mt::quantize_int4 / int8 +
// the per-block scaled int4 helpers used by the cold-tier
// compression in llama_kv_cache_paged.

#include "../src/memory-tier/mt-quant.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

double max_abs_err(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        const double e = std::fabs((double) a[i] - (double) b[i]);
        if (e > m) m = e;
    }
    return m;
}

double cosine_sim(const std::vector<float> & a, const std::vector<float> & b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        dot += (double) a[i] * b[i];
        na  += (double) a[i] * a[i];
        nb  += (double) b[i] * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 1.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// Generate Gaussian-distributed test data with given std, simulating
// real K/V magnitudes (~|x| up to 5-10× the std).
std::vector<float> gauss(size_t n, float stddev, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, stddev);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = dist(rng);
    return v;
}

}  // anon

int main() {
    // ─── int4 (no scale) — input must be in [-1, +1] ───
    {
        const std::vector<float> src = {-1.0f, -0.5f, -0.125f, 0.0f, 0.125f, 0.5f, 1.0f};
        auto packed = mt::quantize_int4(src.data(), src.size());
        std::vector<float> back(src.size());
        bool ok = mt::dequantize_int4(packed.data(), back.data(), src.size());
        assert(ok);
        const double err = max_abs_err(src, back);
        // 4 bits, 7 levels per side → ~1/7 ≈ 14% step; max-abs err <
        // half-step ≈ 0.072. We give margin: 0.10.
        printf("test-mt-quant: int4 simple max_abs_err=%.4f\n", err);
        assert(err < 0.10);
    }

    // ─── int8 (no scale) — input must be in [-1, +1] ───
    {
        auto src = gauss(1024, 0.3f, 42);
        for (auto & v : src) v = std::max(-1.0f, std::min(1.0f, v));
        auto packed = mt::quantize_int8(src.data(), src.size());
        std::vector<float> back(src.size());
        bool ok = mt::dequantize_int8(packed.data(), back.data(), src.size());
        assert(ok);
        const double err = max_abs_err(src, back);
        printf("test-mt-quant: int8 max_abs_err=%.5f\n", err);
        // 8 bits → step ≈ 1/127 ≈ 0.008; half-step ≈ 0.004. Margin 0.01.
        assert(err < 0.01);
    }

    // ─── int4 with per-block scale — handles arbitrary range ───
    {
        // Realistic K/V magnitude: stddev=2.0 → range ~[-8, +8] typical.
        auto src = gauss(1024, 2.0f, 123);
        std::vector<uint8_t> packed((src.size() + 1) / 2);
        float scale = 0.0f;
        bool ok = mt::quantize_block_int4_with_scale(
            src.data(), src.size(), &scale, packed.data());
        assert(ok);
        assert(scale > 0.0f);

        std::vector<float> back(src.size());
        ok = mt::dequantize_block_int4_with_scale(
            packed.data(), scale, back.data(), src.size());
        assert(ok);

        const double err  = max_abs_err(src, back);
        const double cs   = cosine_sim(src, back);
        printf("test-mt-quant: int4-with-scale n=%zu scale=%.3f max_abs_err=%.4f cosine_sim=%.5f\n",
               src.size(), (double) scale, err, cs);
        fflush(stdout);
        // Step = 2*scale/14; half-step = scale/14 ≈ 0.07 * scale.
        // For scale~6 (max-abs of stddev=2 Gaussian over 1024 samples),
        // expected max err ≈ 0.43. Margin: 1.0.
        assert(err < (double) scale * 0.15);
        // Cosine similarity for int4 with per-block scale on Gaussian
        // data: SNR ≈ 29 → cs ≈ 0.98. Threshold 0.97 leaves margin.
        assert(cs > 0.97);
    }

    // ─── Edge cases ───
    {
        // All zeros: scale=1 by convention; round-trip is exact.
        std::vector<float> src(64, 0.0f);
        std::vector<uint8_t> packed(32);
        float scale = 0.0f;
        mt::quantize_block_int4_with_scale(src.data(), src.size(), &scale, packed.data());
        std::vector<float> back(src.size());
        mt::dequantize_block_int4_with_scale(packed.data(), scale, back.data(), src.size());
        for (auto v : back) assert(v == 0.0f);
        printf("test-mt-quant: all-zero round-trip ok\n");
    }
    {
        // Single non-zero element: scale = |x|; that element decodes
        // back to ±scale.
        std::vector<float> src(16, 0.0f);
        src[7] = -3.7f;
        std::vector<uint8_t> packed(8);
        float scale = 0.0f;
        mt::quantize_block_int4_with_scale(src.data(), src.size(), &scale, packed.data());
        std::vector<float> back(src.size());
        mt::dequantize_block_int4_with_scale(packed.data(), scale, back.data(), src.size());
        // src[7] = -3.7 → encoded as -1.0 (fully saturating) → decoded as -scale = -3.7.
        // (encode_int4 maps -1.0 to -7, decode_int4 maps -7 to -1.0; * 3.7 = -3.7.)
        printf("test-mt-quant: single-value round-trip src[7]=%.3f back[7]=%.3f scale=%.3f\n",
               (double) src[7], (double) back[7], (double) scale);
        assert(std::fabs(back[7] - src[7]) < 0.01);
        for (size_t i = 0; i < src.size(); ++i) {
            if (i == 7) continue;
            assert(std::fabs(back[i]) < scale * 0.10);  // others stayed near zero
        }
    }

    printf("test-mt-quant: ALL PASS\n");
    return 0;
}
