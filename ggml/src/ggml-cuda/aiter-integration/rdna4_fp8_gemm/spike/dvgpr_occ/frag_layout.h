// frag_layout.h — the PROVEN gfx12 fp8 WMMA lane maps, from
// spike/gemm_wmma_raw_intrinsic_verified.hip:111-182. Used CPU-side by the harness to pack
// inputs into per-lane fragments and unpack the per-lane v8f32 result into a 16x16 D matrix.
// The KERNEL never does layout math: lane L loads in_A[L*2..], in_B[L*2..], stores out[L*8..].
//
// Confirmed instruction (Task 2 seed disasm, gfx1201):
//   v_wmma_f32_16x16x16_fp8_fp8  vDst[0:7], vA[0:1], vB[0:1], vC[0:7]
//   A = 2xi32/lane (8 fp8 bytes), B = 2xi32/lane (8 fp8 bytes), C/D = v8f32/lane.
#pragma once
#include <cstdint>

// A (row-major 16x16 e4m3): lane L holds row=L&0xF, the 8 contiguous K-bytes [colhi*8..+7]
//   where colhi=(L>>4)&1. Pack into 32 lanes x 2 int32 (= 8 bytes) each.
static inline void pack_A(const uint8_t* A /*256*/, uint32_t* fragA /*64 = 32*2*/) {
    for (int L = 0; L < 32; ++L) {
        int row = L & 0xF, colhi = (L >> 4) & 1, kbase = colhi * 8;
        uint32_t lo = 0, hi = 0;
        for (int p = 0; p < 4; ++p) {
            lo |= (uint32_t)A[row * 16 + kbase + p]     << (p * 8);
            hi |= (uint32_t)A[row * 16 + kbase + 4 + p] << (p * 8);
        }
        fragA[L * 2 + 0] = lo; fragA[L * 2 + 1] = hi;
    }
}
// B (row-major 16x16 e4m3): lane L holds col=L&0xF, K-bytes [rowhi*8..+7], rowhi=(L>>4)&1.
static inline void pack_B(const uint8_t* B /*256*/, uint32_t* fragB /*64*/) {
    for (int L = 0; L < 32; ++L) {
        int col = L & 0xF, rowhi = (L >> 4) & 1, kbase = rowhi * 8;
        uint32_t lo = 0, hi = 0;
        for (int p = 0; p < 4; ++p) {
            lo |= (uint32_t)B[(kbase + p)     * 16 + col] << (p * 8);
            hi |= (uint32_t)B[(kbase + 4 + p) * 16 + col] << (p * 8);
        }
        fragB[L * 2 + 0] = lo; fragB[L * 2 + 1] = hi;
    }
}
// D/C (v8f32 per lane): lane L holds col=L&0xF, rows ((L>>4)&1)*8 + slot, slot in 0..7.
static inline void unpack_D(const float* fragD /*256 = 32*8*/, float* D /*256*/) {
    for (int L = 0; L < 32; ++L) {
        int col = L & 0xF, rowbase = ((L >> 4) & 1) * 8;
        for (int s = 0; s < 8; ++s)
            D[(rowbase + s) * 16 + col] = fragD[L * 8 + s];
    }
}
