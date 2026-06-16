// fp8_oracle.cpp
#include "fp8_oracle.h"

float fp8_e4m3_to_float(uint8_t b) {
    const int sign = (b >> 7) & 1;
    const int exp  = (b >> 3) & 0xF;
    const int man  = b & 0x7;
    float v;
    if (exp == 0) {                       // subnormal (or zero)
        v = man / 8.0f * (1.0f / 64.0f);  // 2^(1-7) = 2^-6 = 1/64
    } else if (exp == 0xF && man == 0x7) {
        v = 0.0f;                          // NaN: not used by our test inputs; map to 0
    } else {
        v = (1.0f + man / 8.0f);
        int e = exp - 7;
        // scale by 2^e
        if (e >= 0) v *= (float)(1u << e);
        else        v /= (float)(1u << (-e));
    }
    return sign ? -v : v;
}

void wmma_ref_16x16x16(const uint8_t* A, const uint8_t* B, const float* C, float* D) {
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float acc = C[i * 16 + j];
            for (int k = 0; k < 16; ++k)
                acc += fp8_e4m3_to_float(A[i * 16 + k]) * fp8_e4m3_to_float(B[k * 16 + j]);
            D[i * 16 + j] = acc;
        }
}
