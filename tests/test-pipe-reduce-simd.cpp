// Correctness + microbench for src/pipeline/pipe-reduce-simd.{h,cpp}.
//
// This is the SIMD unpack helper for the DS4-Flash sliced-expert spine's
// per-dispatch "unpack" phase (see pipe-reduce-simd.h for the numeric
// contract). It proves:
//   1. pipe_simd_accumulate_f32          == scalar acc[i] += add[i]
//   2. pipe_simd_convert_f16_to_f32      == scalar fp16->fp32 widening
//   3. pipe_simd_decode_f16_accumulate   == scalar acc[i] += fp16->fp32(half[i])
// over randomized inputs across many sizes, including non-multiple-of-
// vector-width tails (vector widths in play here are 8 (AVX2) and 16
// (AVX-512)), plus edge-case f16 bit patterns (zero, -0, denormals, inf,
// NaN, max normal). It then microbenches the vectorized path against the
// scalar reference and prints ns/call and GB/s.
//
// Self-contained: no dependency on ggml or any other project header. The
// scalar reference conversion below is an independent implementation of
// IEEE-754 half->float widening (same well-known bit-trick algorithm ggml
// uses internally), which is fine because half->float widening is exact --
// there is exactly one correct f32 value per f16 input, so any correct
// implementation must produce it.

#include "../src/pipeline/pipe-reduce-simd.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------
// Independent scalar reference (not the implementation under test).

static float ref_fp16_to_fp32(uint16_t h) {
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
    uint32_t nb, db;
    std::memcpy(&nb, &normalized_value, sizeof(nb));
    std::memcpy(&db, &denormalized_value, sizeof(db));
    const uint32_t result = sign | (two_w < denormalized_cutoff ? db : nb);

    float out;
    std::memcpy(&out, &result, sizeof(result));
    return out;
}

// ---------------------------------------------------------------------
// Helpers

static int g_failures = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        g_failures++;
    }
}

static uint32_t bits_of(float f) {
    uint32_t b;
    std::memcpy(&b, &f, sizeof(b));
    return b;
}

// ---------------------------------------------------------------------
// Correctness: pipe_simd_accumulate_f32

static void test_accumulate(std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1e4f, 1e4f);
    double max_abs_err = 0.0;
    uint64_t mismatches = 0, total = 0;

    for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(7), size_t(8), size_t(9),
                      size_t(15), size_t(16), size_t(17), size_t(31), size_t(32),
                      size_t(100), size_t(257), size_t(4096), size_t(8192 + 5)}) {
        std::vector<float> acc_ref(n), acc_simd(n), add(n);
        for (size_t i = 0; i < n; ++i) {
            acc_ref[i] = acc_simd[i] = dist(rng);
            add[i]     = dist(rng);
        }
        for (size_t i = 0; i < n; ++i) {
            acc_ref[i] += add[i];
        }
        pipe_simd_accumulate_f32(acc_simd.data(), add.data(), n);

        for (size_t i = 0; i < n; ++i) {
            total++;
            if (bits_of(acc_ref[i]) != bits_of(acc_simd[i])) {
                mismatches++;
                max_abs_err = std::max(max_abs_err, (double) std::fabs(acc_ref[i] - acc_simd[i]));
            }
        }
    }
    check(mismatches == 0, "pipe_simd_accumulate_f32 bit-identical to scalar acc[i]+=add[i]");
    std::printf("accumulate_f32: %llu elements checked, %llu mismatches, max_abs_err=%g\n",
                (unsigned long long) total, (unsigned long long) mismatches, max_abs_err);
}

// ---------------------------------------------------------------------
// Correctness: pipe_simd_convert_f16_to_f32 and pipe_simd_decode_f16_accumulate

static void test_decode(std::mt19937 & rng) {
    std::uniform_int_distribution<int> byte_dist(0, 0xFFFF);

    // Edge-case half bit patterns: +0, -0, smallest denormal, largest
    // denormal, smallest normal, largest normal (finite), +inf, -inf, qNaN,
    // sNaN-ish pattern, 1.0, -1.0.
    std::vector<uint16_t> edge_cases = {
        0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF,
        0x7C00, 0xFC00, 0x7E00, 0x7C01, 0x3C00, 0xBC00,
    };

    double max_abs_err = 0.0, max_rel_err = 0.0;
    uint64_t convert_mismatches = 0, decode_mismatches = 0, total = 0;

    for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(7), size_t(8), size_t(9),
                      size_t(15), size_t(16), size_t(17), size_t(31), size_t(32),
                      size_t(100), size_t(257), size_t(4096), size_t(8192 + 5)}) {
        std::vector<uint16_t> half(n);
        for (size_t i = 0; i < n; ++i) {
            // Mix random bit patterns with edge cases so every size class
            // (including tails) sees at least a few of them.
            if (i < edge_cases.size()) {
                half[i] = edge_cases[i];
            } else {
                half[i] = (uint16_t) byte_dist(rng);
            }
        }

        // convert-only
        std::vector<float> out_ref(n), out_simd(n);
        for (size_t i = 0; i < n; ++i) {
            out_ref[i] = ref_fp16_to_fp32(half[i]);
        }
        pipe_simd_convert_f16_to_f32(out_simd.data(), half.data(), n);
        for (size_t i = 0; i < n; ++i) {
            total++;
            bool ref_nan  = std::isnan(out_ref[i]);
            bool simd_nan = std::isnan(out_simd[i]);
            if (ref_nan || simd_nan) {
                // NaN payload bit patterns are not required to match --
                // just that both sides agree it's a NaN.
                if (ref_nan != simd_nan) {
                    convert_mismatches++;
                }
                continue;
            }
            if (bits_of(out_ref[i]) != bits_of(out_simd[i])) {
                convert_mismatches++;
                max_abs_err = std::max(max_abs_err, (double) std::fabs(out_ref[i] - out_simd[i]));
            }
        }

        // fused decode+accumulate
        std::uniform_real_distribution<float> accdist(-1e3f, 1e3f);
        std::vector<float> acc_ref(n), acc_simd(n);
        for (size_t i = 0; i < n; ++i) {
            acc_ref[i] = acc_simd[i] = accdist(rng);
        }
        for (size_t i = 0; i < n; ++i) {
            acc_ref[i] += ref_fp16_to_fp32(half[i]);
        }
        pipe_simd_decode_f16_accumulate(acc_simd.data(), half.data(), n);
        for (size_t i = 0; i < n; ++i) {
            bool ref_nan  = std::isnan(acc_ref[i]);
            bool simd_nan = std::isnan(acc_simd[i]);
            if (ref_nan || simd_nan) {
                if (ref_nan != simd_nan) {
                    decode_mismatches++;
                }
                continue;
            }
            if (bits_of(acc_ref[i]) != bits_of(acc_simd[i])) {
                decode_mismatches++;
                double err = std::fabs(acc_ref[i] - acc_simd[i]);
                max_abs_err = std::max(max_abs_err, err);
                if (acc_ref[i] != 0.0f) {
                    max_rel_err = std::max(max_rel_err, err / std::fabs((double) acc_ref[i]));
                }
            }
        }
    }

    check(convert_mismatches == 0, "pipe_simd_convert_f16_to_f32 matches scalar reference");
    check(decode_mismatches == 0, "pipe_simd_decode_f16_accumulate matches scalar reference");
    std::printf("decode: %llu elements checked, convert_mismatches=%llu decode_mismatches=%llu "
                "max_abs_err=%g max_rel_err=%g\n",
                (unsigned long long) total, (unsigned long long) convert_mismatches,
                (unsigned long long) decode_mismatches, max_abs_err, max_rel_err);
}

// ---------------------------------------------------------------------
// Microbench

template <typename F>
static double bench_ns_per_call(F && fn, int iters) {
    // warm up
    for (int i = 0; i < 5; ++i) fn();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) fn();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

static void bench(std::mt19937 & rng) {
    // Representative size: one layer's partial for a modest hidden dim /
    // token batch (the DS4-Flash spine sums 43 layers x 3 workers of
    // roughly this size per token).
    const size_t n = 8192;
    const int iters = 20000;

    std::uniform_real_distribution<float> dist(-1e3f, 1e3f);
    std::vector<float> acc(n), add(n), out(n);
    std::vector<uint16_t> half(n);
    for (size_t i = 0; i < n; ++i) {
        acc[i]  = dist(rng);
        add[i]  = dist(rng);
        half[i] = (uint16_t) (rng() & 0xFFFF);
    }

    auto scalar_accumulate = [&]() {
        for (size_t i = 0; i < n; ++i) acc[i] += add[i];
    };
    auto scalar_decode_accumulate = [&]() {
        for (size_t i = 0; i < n; ++i) acc[i] += ref_fp16_to_fp32(half[i]);
    };

    double ns_scalar_acc    = bench_ns_per_call(scalar_accumulate, iters);
    double ns_simd_acc      = bench_ns_per_call([&]() { pipe_simd_accumulate_f32(acc.data(), add.data(), n); }, iters);
    double ns_scalar_decode = bench_ns_per_call(scalar_decode_accumulate, iters);
    double ns_simd_decode   = bench_ns_per_call([&]() { pipe_simd_decode_f16_accumulate(acc.data(), half.data(), n); }, iters);
    double ns_simd_convert  = bench_ns_per_call([&]() { pipe_simd_convert_f16_to_f32(out.data(), half.data(), n); }, iters);

    auto gbps = [&](double ns, size_t bytes_touched) {
        return (double) bytes_touched / ns; // bytes/ns == GB/s
    };

    // accumulate touches 2 reads + 1 write of n floats = 3*4*n bytes.
    const size_t acc_bytes = 3ull * 4 * n;
    // decode touches n halfs read + n floats read (acc) + n floats write = 2*n*4 + n*2
    const size_t decode_bytes = 2ull * n * 4 + n * 2;
    const size_t convert_bytes = n * 2 + n * 4;

    std::printf("\n-- microbench (n=%zu elements, %d iters) --\n", n, iters);
    std::printf("accumulate_f32        scalar: %8.1f ns  (%.2f GB/s)\n", ns_scalar_acc, gbps(ns_scalar_acc, acc_bytes));
    std::printf("accumulate_f32        simd:   %8.1f ns  (%.2f GB/s)  speedup=%.2fx\n",
                ns_simd_acc, gbps(ns_simd_acc, acc_bytes), ns_scalar_acc / ns_simd_acc);
    std::printf("decode_f16_accumulate scalar: %8.1f ns  (%.2f GB/s)\n", ns_scalar_decode, gbps(ns_scalar_decode, decode_bytes));
    std::printf("decode_f16_accumulate simd:   %8.1f ns  (%.2f GB/s)  speedup=%.2fx\n",
                ns_simd_decode, gbps(ns_simd_decode, decode_bytes), ns_scalar_decode / ns_simd_decode);
    std::printf("convert_f16_to_f32    simd:   %8.1f ns  (%.2f GB/s)\n", ns_simd_convert, gbps(ns_simd_convert, convert_bytes));
    std::printf("fused decode+accumulate vs convert-then-accumulate (simd+simd = %.1f ns): fused is %.2fx\n",
                ns_simd_convert + ns_simd_acc, (ns_simd_convert + ns_simd_acc) / ns_simd_decode);
}

int main() {
    std::mt19937 rng(0xC0FFEEu);

    test_accumulate(rng);
    test_decode(rng);
    bench(rng);

    if (g_failures > 0) {
        std::fprintf(stderr, "\n%d correctness check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\nAll correctness checks passed.\n");
    return 0;
}
