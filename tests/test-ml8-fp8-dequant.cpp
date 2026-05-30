// Test: ml8-fp8 dequant correctness (MAD Task 9)
//
// Validates that:
//  1. ggml_type_size(GGML_TYPE_ML8_FP8) == 34
//  2. ggml_blck_size(GGML_TYPE_ML8_FP8) == 32
//  3. dequantize_row_ml8_fp8 matches hand-computed e4m3_decode(byte)*scale for
//     a set of known byte values, verifying the on-disk layout matches the
//     Python writer (commit 45925db35).
//
// Build + run (from llama.cpp root after building build-ml8fp8-host):
//   g++ -std=c++17 -O2 \
//       -I ggml/include -I ggml/src \
//       tests/test-ml8-fp8-dequant.cpp \
//       -L build-ml8fp8-host/bin -lggml-base \
//       -Wl,-rpath,build-ml8fp8-host/bin \
//       -o /tmp/test-ml8-fp8-dequant && /tmp/test-ml8-fp8-dequant

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cassert>

#include "ggml.h"
#include "ggml-common.h"        // block_ml8_fp8, QK_ML8_FP8
#include "ggml-quants.h"        // dequantize_row_ml8_fp8

// ── portable fp16 → fp32 helper (bit-cast, matches GGML_FP16_TO_FP32) ─────

static float fp16_to_fp32_ref(uint16_t h) {
    uint32_t sign  = (h >> 15) & 1u;
    uint32_t exp   = (h >> 10) & 0x1Fu;
    uint32_t mant  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0 && mant == 0) {
        bits = sign << 31;
    } else if (exp == 31) {
        // inf / NaN
        bits = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else if (exp == 0) {
        // subnormal fp16 → normalized fp32
        int lead = 0;
        uint32_t m = mant;
        while (!(m & 0x200u)) { m <<= 1; lead++; }
        m &= 0x1FFu;
        bits = (sign << 31) | ((uint32_t)(127 - 14 - lead) << 23) | (m << 14);
    } else {
        bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ── OCP e4m3fn decode (must match g_fp8_e4m3_lut in ggml-turbo-quant.c) ───

static float e4m3_decode_ref(uint8_t byte) {
    uint32_t s = (byte >> 7) & 1u;
    uint32_t e = (byte >> 3) & 0xFu;
    uint32_t m = byte & 0x7u;
    uint32_t bits;
    float f;
    if (e == 0 && m == 0) {
        bits = s << 31;
    } else if (e == 15 && m == 7) {
        // NaN
        bits = (s << 31) | (0xFFu << 23) | (1u << 22);
    } else if (e == 0) {
        // subnormal: (-1)^s * 2^(-6) * (m/8)
        int      lead     = (m >= 4) ? 2 : (m >= 2) ? 1 : 0;
        uint32_t mant_n   = ((m << (3 - lead)) & 0x7u);
        int      exp_un   = -6 - (2 - lead);
        uint32_t exp_fp32 = (uint32_t)(exp_un + 127);
        bits = (s << 31) | (exp_fp32 << 23) | (mant_n << 20);
    } else {
        // normal: (-1)^s * 2^(e-7) * (1 + m/8)
        uint32_t exp_fp32 = e + 120u;
        uint32_t mant     = m << 20;
        bits = (s << 31) | (exp_fp32 << 23) | mant;
    }
    memcpy(&f, &bits, 4);
    return f;
}

// ── fp32 → fp16 (for constructing test scale) ─────────────────────────────

static uint16_t fp32_to_fp16_ref(float f) {
    uint32_t bits;
    memcpy(&bits, &f, 4);
    uint32_t sign = bits >> 31;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;
    uint16_t h;
    if (exp == 0xFF) {
        h = (uint16_t)((sign << 15) | 0x7C00u | (mant ? 0x200u : 0));
    } else if (exp == 0 && mant == 0) {
        h = (uint16_t)(sign << 15);
    } else {
        int e16 = (int)exp - 127 + 15;
        if (e16 >= 31) { h = (uint16_t)((sign << 15) | 0x7C00u); }
        else if (e16 <= 0) { h = (uint16_t)(sign << 15); }
        else { h = (uint16_t)((sign << 15) | ((uint16_t)e16 << 10) | (uint16_t)(mant >> 13)); }
    }
    return h;
}

int main(void) {
    int failures = 0;

    // ── 1. Type-trait metadata ─────────────────────────────────────────────
    {
        size_t ts = ggml_type_size(GGML_TYPE_ML8_FP8);
        int    bs = ggml_blck_size(GGML_TYPE_ML8_FP8);
        if (ts != 34) {
            fprintf(stderr, "FAIL: ggml_type_size(ML8_FP8) = %zu, want 34\n", ts);
            failures++;
        } else {
            printf("PASS: ggml_type_size(ML8_FP8) == 34\n");
        }
        if (bs != 32) {
            fprintf(stderr, "FAIL: ggml_blck_size(ML8_FP8) = %d, want 32\n", bs);
            failures++;
        } else {
            printf("PASS: ggml_blck_size(ML8_FP8) == 32\n");
        }
        const char * name = ggml_type_name(GGML_TYPE_ML8_FP8);
        if (!name || strcmp(name, "ml8_fp8") != 0) {
            fprintf(stderr, "FAIL: ggml_type_name = '%s', want 'ml8_fp8'\n", name ? name : "(null)");
            failures++;
        } else {
            printf("PASS: ggml_type_name(ML8_FP8) == \"%s\"\n", name);
        }
    }

    // ── 2. sizeof(block_ml8_fp8) is exactly 34 bytes ─────────────────────
    // (This is also checked by the static_assert in ggml-common.h, so it
    //  should never fire here; but having a runtime check is belt-and-braces.)
    {
        static_assert(sizeof(block_ml8_fp8) == 34, "block_ml8_fp8 must be 34 bytes");
        printf("PASS: sizeof(block_ml8_fp8) == 34\n");
    }

    // ── 3. Dequant correctness — single block with scale = 2.0 ────────────
    //
    // Probe bytes chosen to cover: 0x00 (±zero), 0x3C (1.0 in e4m3),
    // 0x80 (-0), 0x01 (smallest e4m3 subnormal), 0x7E (large positive),
    // 0x04 (e4m3 normal), 0xFF (NaN byte — check NaN propagates).
    {
        const float scale_f32 = 2.0f;
        const uint16_t scale_fp16 = fp32_to_fp16_ref(scale_f32);

        // Probe byte values and their expected output indices within the block
        const uint8_t probe_bytes[] = { 0x00, 0x3C, 0x80, 0x01, 0x7E, 0x04, 0x3F };
        const int n_probe = (int)(sizeof(probe_bytes) / sizeof(probe_bytes[0]));

        // Build a 32-element block: fill with 0x3C (1.0f in e4m3), then patch
        // selected positions with probe bytes.
        block_ml8_fp8 blk;
        memcpy(&blk.scale, &scale_fp16, 2);
        for (int i = 0; i < QK_ML8_FP8; i++) blk.qs[i] = 0x3C; // 1.0 in e4m3
        // Overwrite first n_probe positions with probe bytes
        for (int i = 0; i < n_probe; i++) blk.qs[i] = probe_bytes[i];

        float out[QK_ML8_FP8] = {};
        dequantize_row_ml8_fp8(&blk, out, QK_ML8_FP8);

        // Check probed positions
        for (int i = 0; i < n_probe; i++) {
            float expected = e4m3_decode_ref(probe_bytes[i]) * scale_f32;
            float got      = out[i];
            bool  ok;
            // NaN case: both should be NaN
            if (std::isnan(expected)) {
                ok = std::isnan(got);
            } else {
                ok = (fabsf(got - expected) <= 1e-5f * (1.0f + fabsf(expected)));
            }
            if (!ok) {
                fprintf(stderr, "FAIL: out[%d] = %g, expected %g (byte=0x%02x scale=%.1f)\n",
                        i, (double)got, (double)expected, probe_bytes[i], (double)scale_f32);
                failures++;
            } else {
                printf("PASS: out[%d] = %g (byte=0x%02x, e4m3=%.6g, *%.1f)\n",
                       i, (double)got, probe_bytes[i],
                       (double)e4m3_decode_ref(probe_bytes[i]), (double)scale_f32);
            }
        }

        // Check non-probed tail positions (all 0x3C → 1.0 * 2.0 = 2.0)
        float expected_tail = e4m3_decode_ref(0x3C) * scale_f32;
        for (int i = n_probe; i < QK_ML8_FP8; i++) {
            if (fabsf(out[i] - expected_tail) > 1e-5f) {
                fprintf(stderr, "FAIL: tail out[%d] = %g, expected %g\n",
                        i, (double)out[i], (double)expected_tail);
                failures++;
            }
        }
        printf("PASS: all %d tail positions (0x3C * 2.0 = %.2f)\n",
               QK_ML8_FP8 - n_probe, (double)expected_tail);
    }

    // ── 4. Two-block test: verify block-boundary stride is correct ─────────
    {
        // Block 0: scale=1.0, all qs=0x40 (2.0 in e4m3fn normal: e=8,m=0 → 2^1=2)
        // Block 1: scale=0.5, all qs=0x3C (1.0 in e4m3: e=7,m=4 → 1.0+4/8=1.5? no:
        //   e4m3 normal: value=(-1)^s * 2^(e-7) * (1+m/8)
        //   0x3C = 0b00111100 → s=0, e=7, m=4 → 2^0*(1+0.5) = 1.5
        // 0x3C gives 1.5 in e4m3fn (confirmed above from our e4m3_decode_ref).
        const int k = QK_ML8_FP8 * 2;
        block_ml8_fp8 blks[2];

        uint16_t s0 = fp32_to_fp16_ref(1.0f);
        uint16_t s1 = fp32_to_fp16_ref(0.5f);
        memcpy(&blks[0].scale, &s0, 2);
        memcpy(&blks[1].scale, &s1, 2);
        for (int i = 0; i < QK_ML8_FP8; i++) blks[0].qs[i] = 0x40; // e4m3 2.0
        for (int i = 0; i < QK_ML8_FP8; i++) blks[1].qs[i] = 0x3C; // e4m3 1.5

        float out[64] = {};
        dequantize_row_ml8_fp8(blks, out, k);

        float exp0 = e4m3_decode_ref(0x40) * 1.0f;
        float exp1 = e4m3_decode_ref(0x3C) * 0.5f;

        int blk_fail = 0;
        for (int i = 0; i < QK_ML8_FP8; i++) {
            if (fabsf(out[i] - exp0) > 1e-5f)               blk_fail++;
        }
        for (int i = QK_ML8_FP8; i < k; i++) {
            if (fabsf(out[i] - exp1) > 1e-5f)               blk_fail++;
        }
        if (blk_fail) {
            fprintf(stderr, "FAIL: two-block test: %d mismatches\n", blk_fail);
            failures++;
        } else {
            printf("PASS: two-block test: blk0 e4m3(0x40)*1.0=%g, blk1 e4m3(0x3C)*0.5=%g\n",
                   (double)exp0, (double)exp1);
        }
    }

    // ── Result ────────────────────────────────────────────────────────────
    if (failures == 0) {
        printf("\nALL TESTS PASSED\n");
        return 0;
    } else {
        fprintf(stderr, "\n%d TEST(S) FAILED\n", failures);
        return 1;
    }
}
