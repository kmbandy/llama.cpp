// scripts/calibration/tools/ml8_e4m3_golden.c
// Host copy of ggml/src/ggml-cuda/ml8.cu:ml8_fp32_to_e4m3 (the FIXED kernel,
// e_out > 15). Emits: for each fp32 in the battery, one uint8 e4m3 code.
// Output file format: int32 count, then `count` float32 inputs, then `count` uint8 codes.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

static uint8_t ml8_fp32_to_e4m3(float xv) {
    uint32_t bits; memcpy(&bits, &xv, 4);
    const uint32_t sign  = (bits >> 31) & 1u;
    const uint32_t exp_b = (bits >> 23) & 0xFFu;
    const uint32_t mant  = bits & 0x7FFFFFu;
    if (exp_b == 0xFFu) return (uint8_t)((sign << 7) | 0x7Fu);
    if (exp_b == 0)     return (uint8_t)(sign << 7);
    const int32_t e_un = (int32_t) exp_b - 127;
    if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u))
        return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
    if (e_un >= -6) {
        const uint32_t e_e4m3 = (uint32_t)(e_un + 7);
        const uint32_t guard  = (mant >> 19) & 1u;
        const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
        const uint32_t lsb    = (mant >> 20) & 1u;
        uint32_t       m_e4m3 = (mant >> 20) & 0x7u;
        if (guard && (sticky || lsb)) m_e4m3 += 1;
        uint32_t e_out = e_e4m3;
        if (m_e4m3 == 8) { m_e4m3 = 0; e_out += 1;
            if (e_out > 15) return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u); }
        if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
        return (uint8_t)((sign << 7) | (e_out << 3) | m_e4m3);
    }
    const int32_t shift = 23 - (e_un + 9);
    if (shift > 31) return (uint8_t)(sign << 7);
    const uint32_t implicit = (1u << 23) | mant;
    const uint32_t guard    = (implicit >> (shift - 1)) & 1u;
    const uint32_t sticky   = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
    uint32_t       m_e4m3   = implicit >> shift;
    const uint32_t lsb      = m_e4m3 & 1u;
    if (guard && (sticky || lsb)) m_e4m3 += 1;
    if (m_e4m3 >= 8) return (uint8_t)((sign << 7) | (1u << 3));
    return (uint8_t)((sign << 7) | m_e4m3);
}

int main(void) {
    // Battery: dense low range, every normal lattice boundary, the 256..448
    // band (the e=15 fix), subnormals < 2^-6, ties, saturation, sign symmetry.
    float xs[100000]; int n = 0;
    for (float v = -512.0f; v <= 512.0f; v += 0.013f) xs[n++] = v;       // dense sweep
    float edges[] = {448.0f, 449.0f, 256.0f, 288.0f, 320.0f, 480.0f,
                     0.015625f, 0.0078125f, 0.001953125f,                // 2^-6,2^-7,2^-9
                     1e-30f, 1e30f, -0.0f};
    for (unsigned i = 0; i < sizeof(edges)/sizeof(float); i++) { xs[n++]=edges[i]; xs[n++]=-edges[i]; }
    float inf = INFINITY, nan = NAN; xs[n++]=inf; xs[n++]=-inf; xs[n++]=nan;

    FILE *f = fopen("/tmp/ml8_e4m3_golden.bin", "wb");
    fwrite(&n, 4, 1, f);
    fwrite(xs, 4, n, f);
    for (int i = 0; i < n; i++) { uint8_t c = ml8_fp32_to_e4m3(xs[i]); fwrite(&c, 1, 1, f); }
    fclose(f);
    printf("wrote %d cases to /tmp/ml8_e4m3_golden.bin\n", n);
    return 0;
}
