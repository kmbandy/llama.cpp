// test_fp8_oracle.cpp
#include "fp8_oracle.h"
#include <cstdio>
#include <cmath>
#include <cstdlib>

static int fails = 0;
#define CHECK(cond) do { if(!(cond)){ printf("FAIL %s:%d %s\n",__FILE__,__LINE__,#cond); fails++; } } while(0)

int main() {
    // e4m3: 0x00=0.0, 0x38=1.0, 0x40=2.0, 0xB8=-1.0, 0x3C=1.5
    CHECK(fp8_e4m3_to_float(0x00) == 0.0f);
    CHECK(fp8_e4m3_to_float(0x38) == 1.0f);
    CHECK(fp8_e4m3_to_float(0x40) == 2.0f);
    CHECK(fp8_e4m3_to_float(0xB8) == -1.0f);
    CHECK(fp8_e4m3_to_float(0x3C) == 1.5f);

    // 16x16x16 reference: A = identity-ish (all 1.0 in row 0), B all 1.0, C=0
    // D[i][j] = sum_k A[i][k]*B[k][j]. A all-1, B all-1 => D[i][j] = 16.0
    uint8_t A[256], B[256]; float C[256] = {0}, D[256];
    for (int i = 0; i < 256; ++i) { A[i] = 0x38; B[i] = 0x38; }  // all 1.0
    wmma_ref_16x16x16(A, B, C, D);
    CHECK(std::fabs(D[0] - 16.0f) < 1e-6);
    CHECK(std::fabs(D[16*16-1] - 16.0f) < 1e-6);

    printf(fails ? "FAILED (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
