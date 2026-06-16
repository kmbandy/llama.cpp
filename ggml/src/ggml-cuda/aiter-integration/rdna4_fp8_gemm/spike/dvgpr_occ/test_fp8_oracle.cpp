// test_fp8_oracle.cpp
#include "fp8_oracle.h"
#include "frag_layout.h"
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

    // ---- round-trip the §7.12 pack maps: prove pack_A/pack_B preserve (row,k)/(k,col) ----
    {
        uint8_t A2in[256], B2in[256]; float C2[256] = {0}, Dref[256];
        for (int i = 0; i < 256; ++i) { A2in[i] = (uint8_t)(0x38 + (i % 3)); B2in[i] = (uint8_t)(0x38 + (i % 2)); }
        wmma_ref_16x16x16(A2in, B2in, C2, Dref);
        uint32_t fa[64], fb[64];
        pack_A(A2in, fa); pack_B(B2in, fb);
        // Emulate the WMMA on the packed fragments by decoding them back and re-doing the matmul,
        // proving pack_A/pack_B preserve the (row,k)/(k,col) values the hardware will see.
        uint8_t A2[256] = {0}, B2[256] = {0};
        for (int L = 0; L < 32; ++L) {
            int row = L & 0xF, colhi = (L>>4)&1, cb = colhi*8;
            for (int p = 0; p < 4; ++p) { A2[row*16+cb+p]   = (fa[L*2]   >> (p*8)) & 0xFF;
                                          A2[row*16+cb+4+p] = (fa[L*2+1] >> (p*8)) & 0xFF; }
            int col = L & 0xF, rowhi = (L>>4)&1, rb = rowhi*8;
            for (int p = 0; p < 4; ++p) { B2[(rb+p)*16+col]   = (fb[L*2]   >> (p*8)) & 0xFF;
                                          B2[(rb+4+p)*16+col] = (fb[L*2+1] >> (p*8)) & 0xFF; }
        }
        float Drt[256]; wmma_ref_16x16x16(A2, B2, C2, Drt);
        for (int i = 0; i < 256; ++i) CHECK(A2[i] == A2in[i] && B2[i] == B2in[i]);
        for (int i = 0; i < 256; ++i) CHECK(std::fabs(Drt[i] - Dref[i]) < 1e-6f);
    }

    printf(fails ? "FAILED (%d)\n" : "PASS\n", fails);
    return fails ? 1 : 0;
}
