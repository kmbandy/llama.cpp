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

#include <cmath>
// Tiered oracle comparison. bad = #elements exceeding rel*|ref|+abs_; max_rel = worst |got-ref|/|ref|.
OracleCmp oracle_compare(const float* got, const float* ref, long n, float rel, float abs_) {
    OracleCmp r{true, 0, 0.0};
    for (long i = 0; i < n; ++i) {
        float d   = std::fabs(got[i] - ref[i]);
        float thr = rel * std::fabs(ref[i]) + abs_;
        double rl = (double)d / ((double)std::fabs(ref[i]) + 1e-30);
        if (rl > r.max_rel) r.max_rel = rl;
        if (d > thr) { r.ok = false; ++r.bad; }
    }
    return r;
}

#ifdef ORACLE_SELFTEST
#include <cassert>
#include <cstdio>
#include <vector>
int main() {
    const long n = 256;
    // ref ~ O(100) so the abs term doesn't dominate the rel term in the test.
    std::vector<float> ref(n), id(n), p01(n), p1(n), p5(n);
    for (long i = 0; i < n; ++i) {
        float v = 100.0f + 50.0f * (float)(i % 7);
        ref[i] = v; id[i] = v;
        p01[i] = v * 1.001f;   // 0.1%
        p1[i]  = v * 1.01f;    // 1%
        p5[i]  = v * 1.05f;    // 5%
    }
    assert( oracle_compare(id.data(),  ref.data(), n, 5e-3f, 1e-2f).ok);   // identical -> tight ok
    assert( oracle_compare(p01.data(), ref.data(), n, 5e-3f, 1e-2f).ok);   // 0.1% -> tight ok
    assert(!oracle_compare(p1.data(),  ref.data(), n, 5e-3f, 1e-2f).ok);   // 1%   -> tight REJECTS
    assert( oracle_compare(p1.data(),  ref.data(), n, 3e-2f, 2e-2f).ok);   // 1%   -> loose ok
    assert(!oracle_compare(p5.data(),  ref.data(), n, 3e-2f, 2e-2f).ok);   // 5%   -> loose REJECTS
    printf("ORACLE_SELFTEST all pass\n");
    return 0;
}
#endif
