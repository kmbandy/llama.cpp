// bench/trfeed_common_test.cpp — preshuffle is a within-tile bijection (no byte lost/dup).
#include "trfeed_common.h"
#include <vector>
#include <cstdio>
int main() {
    // trperm must be a bijection over 0..255 across (L,s).
    std::vector<int> seen(256, 0);
    for (int L = 0; L < 32; ++L) for (int s = 0; s < 8; ++s) seen[trperm(L, s)]++;
    int bad = 0; for (int i = 0; i < 256; ++i) if (seen[i] != 1) bad++;
    if (bad) { printf("trperm NOT a bijection: %d offsets wrong\n", bad); return 1; }

    // preshuffle then read-back-by-contract reconstructs B exactly.
    const int K = 32, N = 48;              // multi-tile, not square
    std::vector<uint8_t> B(K * N), Bshuf(K * N, 0), Brec(K * N, 0);
    for (int i = 0; i < K * N; ++i) B[i] = (uint8_t)(i * 31 + 7);
    preshuffle_B(B.data(), Bshuf.data(), K, N);
    int NT = N / 16;
    for (int kt = 0; kt < K / 16; ++kt) for (int nt = 0; nt < NT; ++nt) {
        const uint8_t* tile = Bshuf.data() + b_tile_offset(kt, nt, NT);
        for (int L = 0; L < 32; ++L) for (int s = 0; s < 8; ++s) {
            int kl = ((L >> 4) & 1) * 8 + s, nl = L & 15;
            Brec[(size_t)(kt * 16 + kl) * N + (nt * 16 + nl)] = tile[trperm(L, s)];
        }
    }
    for (int i = 0; i < K * N; ++i) if (B[i] != Brec[i]) { printf("recon mismatch @%d\n", i); return 1; }
    printf("trfeed_common: PASS (bijection + multi-tile round-trip)\n");
    return 0;
}
