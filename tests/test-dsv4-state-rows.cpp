// DSV4 state save row counts. CPU only, no model.
//
// A save keeps (pos_max + 1)/ratio rows of each compressed cache, where ratio is the
// row ratio the cache is sized with. V4 uses 4 (CSA, lid) and 128 (HCA). V4.1 uses
// ratio_kv = min of its two ratios for CSA and lid, so a file with ratio 1 layers must
// keep one row per position.

#include "llama-kv-cache-dsv4.h"

#include <cstdint>
#include <cstdio>

static int n_fail = 0;

static void check(const char * name, uint32_t got, uint32_t want) {
    if (got != want) {
        std::fprintf(stderr, "FAIL %s: got %u, want %u\n", name, got, want);
        n_fail++;
    }
}

int main() {
    // no tokens saved
    check("empty seq", dsv4_state_n_used_k_rows(-1, 4, 1024), 0);

    // V4 CSA ratio 4
    check("v4 csa pos 0",  dsv4_state_n_used_k_rows(0,  4, 1024), 0);
    check("v4 csa pos 3",  dsv4_state_n_used_k_rows(3,  4, 1024), 1);
    check("v4 csa pos 7",  dsv4_state_n_used_k_rows(7,  4, 1024), 2);

    // V4 HCA ratio 128
    check("v4 hca pos 127", dsv4_state_n_used_k_rows(127, 128, 1024), 1);
    check("v4 hca pos 255", dsv4_state_n_used_k_rows(255, 128, 1024), 2);

    // V4.1 with ratio_kv 1: every position owns a row
    check("v41 r1 pos 0",   dsv4_state_n_used_k_rows(0,   1, 1024), 1);
    check("v41 r1 pos 99",  dsv4_state_n_used_k_rows(99,  1, 1024), 100);
    check("v41 r1 clamped", dsv4_state_n_used_k_rows(99,  1, 50),   50);

    // V4.1 with ratio_kv 2
    check("v41 r2 pos 99",  dsv4_state_n_used_k_rows(99,  2, 1024), 50);
    check("v41 r2 pos 98",  dsv4_state_n_used_k_rows(98,  2, 1024), 49);

    if (n_fail != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", n_fail);
        return 1;
    }

    std::printf("test-dsv4-state-rows: OK\n");
    return 0;
}
