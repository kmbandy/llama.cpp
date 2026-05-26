// test-ml8-dequant — MAD-223 Phase G.2 CPU dequant round-trip tests.
//
// Validates:
//   - fp8_e4m3 byte → fp32 conversion against known reference points
//   - fp8_e4m3 round-trip (quantize → dequantize) preserves representable values
//   - dequantize_row_ml8_4_with_lut produces the expected fp32 output for a
//     synthetic block (centroid LUT + indices + scale → known result)
//
// Pure C++ test, no GPU required. Runs in milliseconds.

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-common.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int n_failures = 0;

static void check_close(const char *label, float got, float expected, float tol) {
    const float diff = std::fabs(got - expected);
    if (diff > tol || std::isnan(got) != std::isnan(expected)) {
        std::printf("  [FAIL] %s: got=%.6g expected=%.6g diff=%.6g (tol=%.6g)\n",
                    label, got, expected, diff, tol);
        n_failures++;
    } else {
        std::printf("  [ ok ] %s: got=%.6g expected=%.6g\n", label, got, expected);
    }
}

static void test_f8_e4m3_known_values(void) {
    std::printf("\n# test_f8_e4m3_known_values\n");
    // 256-element LUT exercised; spot-check the well-defined points.
    std::vector<uint8_t> bytes = { 0x00, 0x80, 0x38, 0xB8, 0x40, 0xC0, 0x7E, 0xFE };
    std::vector<float>   refs  = { 0.0f, -0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 448.0f, -448.0f };
    std::vector<float>   y(bytes.size());
    dequantize_row_f8_e4m3(bytes.data(), y.data(), (int64_t)bytes.size());
    for (size_t i = 0; i < bytes.size(); i++) {
        char label[32];
        std::snprintf(label, sizeof(label), "0x%02X", bytes[i]);
        check_close(label, y[i], refs[i], 1e-6f);
    }
    // NaN encoding (S.1111.111 = 0x7F / 0xFF)
    uint8_t nan_bytes[2] = { 0x7F, 0xFF };
    float   nan_out[2]   = { 0.0f, 0.0f };
    dequantize_row_f8_e4m3(nan_bytes, nan_out, 2);
    if (!std::isnan(nan_out[0]) || !std::isnan(nan_out[1])) {
        std::printf("  [FAIL] NaN encoding 0x7F/0xFF did not produce NaN\n");
        n_failures++;
    } else {
        std::printf("  [ ok ] 0x7F → NaN, 0xFF → NaN\n");
    }
}

static void test_f8_e4m3_roundtrip(void) {
    std::printf("\n# test_f8_e4m3_roundtrip (representable values are exact)\n");
    // Values that lie exactly on the e4m3 lattice — should round-trip exactly.
    std::vector<float> vals = {
        0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f,
        128.0f, 256.0f, 448.0f, -448.0f, 1.5f, 1.25f, 1.125f, -1.5f,
    };
    std::vector<uint8_t> bytes(vals.size());
    std::vector<float>   back(vals.size());
    quantize_row_f8_e4m3_ref(vals.data(), bytes.data(), (int64_t)vals.size());
    dequantize_row_f8_e4m3(bytes.data(), back.data(), (int64_t)vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
        char label[32];
        std::snprintf(label, sizeof(label), "v=%.4g", vals[i]);
        check_close(label, back[i], vals[i], 1e-6f);
    }
}

static void test_f8_e4m3_saturation(void) {
    std::printf("\n# test_f8_e4m3_saturation (|x| > 448 saturates to ±448)\n");
    std::vector<float>   vals  = { 500.0f, -500.0f, 1e6f, -1e6f, 449.0f, -449.0f };
    std::vector<float>   refs  = { 448.0f, -448.0f, 448.0f, -448.0f, 448.0f, -448.0f };
    std::vector<uint8_t> bytes(vals.size());
    std::vector<float>   back(vals.size());
    quantize_row_f8_e4m3_ref(vals.data(), bytes.data(), (int64_t)vals.size());
    dequantize_row_f8_e4m3(bytes.data(), back.data(), (int64_t)vals.size());
    for (size_t i = 0; i < vals.size(); i++) {
        char label[32];
        std::snprintf(label, sizeof(label), "v=%.4g", vals[i]);
        check_close(label, back[i], refs[i], 1e-6f);
    }
}

static void test_ml8_4_dequant_single_block(void) {
    std::printf("\n# test_ml8_4_dequant_single_block (one block, known LUT/indices/scale)\n");
    // Synthetic block: scale = 1.5, even positions index 0 (centroid=1.0),
    // odd positions index 1 (centroid=2.0). LUT has 16 entries; only [0]
    // and [1] are used.
    block_ml8_4 blk;
    blk.scale = 1.5f;
    // Each byte holds two nibbles, lo-first. lo-nibble=0, hi-nibble=1 → 0x10.
    for (int i = 0; i < QK_ML8 / 2; i++) blk.qs[i] = 0x10;
    uint8_t lut[16] = { 0 };
    lut[0] = 0x38;  // fp8 1.0
    lut[1] = 0x40;  // fp8 2.0
    // Remaining 14 entries are 0x00 (= +0.0), unused for this test.
    std::vector<float> y(QK_ML8, 0.0f);
    dequantize_row_ml8_4_with_lut(&blk, lut, y.data(), QK_ML8);
    int ok = 0, fail = 0;
    for (int i = 0; i < QK_ML8; i++) {
        const float expected = (i % 2 == 0) ? (1.0f * 1.5f) : (2.0f * 1.5f);
        if (std::fabs(y[i] - expected) > 1e-6f) {
            if (fail < 3) std::printf("  [FAIL] y[%d] = %.6g, expected %.6g\n", i, y[i], expected);
            fail++;
        } else {
            ok++;
        }
    }
    std::printf("  %d/%d correct (even=1.5, odd=3.0)\n", ok, QK_ML8);
    if (fail) n_failures++;
}

static void test_ml8_4_dequant_multi_block(void) {
    std::printf("\n# test_ml8_4_dequant_multi_block (3 K-groups, per-group LUT + scale)\n");
    constexpr int N_GROUPS = 3;
    std::vector<block_ml8_4> blocks(N_GROUPS);
    std::vector<uint8_t>     lut(N_GROUPS * 16, 0);
    // Group g: scale = (g+1), all indices = 0, LUT[g][0] = fp8(g+1)
    // → output[g*64..g*64+63] = (g+1) * (g+1)
    blocks[0].scale = 1.0f;
    blocks[1].scale = 2.0f;
    blocks[2].scale = 4.0f;
    for (int g = 0; g < N_GROUPS; g++) for (int i = 0; i < QK_ML8 / 2; i++) blocks[g].qs[i] = 0;
    lut[0 * 16 + 0] = 0x38;  // fp8 1.0  → group 0: 1.0 * 1.0 = 1.0
    lut[1 * 16 + 0] = 0x40;  // fp8 2.0  → group 1: 2.0 * 2.0 = 4.0
    lut[2 * 16 + 0] = 0x48;  // fp8 4.0  → group 2: 4.0 * 4.0 = 16.0
    std::vector<float> y(N_GROUPS * QK_ML8, 0.0f);
    dequantize_row_ml8_4_with_lut(blocks.data(), lut.data(), y.data(), N_GROUPS * QK_ML8);
    const float expected_per_group[N_GROUPS] = { 1.0f, 4.0f, 16.0f };
    int fail = 0;
    for (int g = 0; g < N_GROUPS; g++) {
        for (int i = 0; i < QK_ML8; i++) {
            const float v = y[g * QK_ML8 + i];
            if (std::fabs(v - expected_per_group[g]) > 1e-6f) {
                if (fail < 3) std::printf("  [FAIL] g=%d i=%d: y=%.6g expected=%.6g\n", g, i, v, expected_per_group[g]);
                fail++;
            }
        }
    }
    std::printf("  group expected = {1.0, 4.0, 16.0}, mismatches = %d / %d\n", fail, N_GROUPS * QK_ML8);
    if (fail) n_failures++;
}

int main(void) {
    std::printf("# ml8-4 CPU dequant + f8_e4m3 round-trip tests (MAD-223 Phase G.2)\n");
    test_f8_e4m3_known_values();
    test_f8_e4m3_roundtrip();
    test_f8_e4m3_saturation();
    test_ml8_4_dequant_single_block();
    test_ml8_4_dequant_multi_block();
    if (n_failures == 0) {
        std::printf("\n=== PASS ===\n");
        return 0;
    } else {
        std::printf("\n=== FAIL: %d ===\n", n_failures);
        return 1;
    }
}
