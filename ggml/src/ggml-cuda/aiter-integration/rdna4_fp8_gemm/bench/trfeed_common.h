// bench/trfeed_common.h — Phase-1 shared preshuffle + transpose-load addressing.
// Verified contract: bench/global_load_tr_contract.md (Phase 0, bit-exact).
#pragma once
#include <cstdint>
#include <cstddef>

// Closed-form byte permutation of global_load_tr_b64 (lane L passes Bshuf + L*8;
// output (lane L, slot s) receives Bshuf[trperm(L,s)]).  L in 0..31, s in 0..7.
__host__ __device__ inline int trperm(int L, int s) {
    int base = (L & 7) + ((L >> 3) & 1) * 32 + ((L >> 4) & 1) * 128;
    return base + (s & 3) * 8 + ((s >> 2) & 1) * 64;
}

// Byte offset of the 16x16 (K x N) tile (kt, nt) within the tile-major Bshuf buffer.
__host__ __device__ inline size_t b_tile_offset(int kt, int nt, int NT) {
    return (size_t)(kt * NT + nt) * 256;  // 256 bytes per 16x16 fp8 tile
}

// Pre-shuffle row-major fp8 B[K][N] into tile-major Bshuf (16x16 tiles, trperm order).
// K, N multiples of 16. One-time repack (B is static weights -> free at runtime).
inline void preshuffle_B(const uint8_t* B, uint8_t* Bshuf, int K, int N) {
    int KT = K / 16, NT = N / 16;
    for (int kt = 0; kt < KT; ++kt)
        for (int nt = 0; nt < NT; ++nt) {
            uint8_t* tile = Bshuf + b_tile_offset(kt, nt, NT);
            for (int L = 0; L < 32; ++L)
                for (int s = 0; s < 8; ++s) {
                    int kl = ((L >> 4) & 1) * 8 + s, nl = L & 15;
                    tile[trperm(L, s)] = B[(size_t)(kt * 16 + kl) * N + (nt * 16 + nl)];
                }
        }
}
