// Bit-identity test: wp_gemm_mxfp4_q8_0 vs ggml_vec_dot_mxfp4_q8_0 (verbatim copy).
// Single-threaded, a few seconds. Thread splits are emulated by calling the
// kernel on contiguous row sub-ranges exactly like mul_mat_one_chunk would.
#include "wp-mxfp4-ref.h"
#include <cstdio>
#include <vector>
#include <random>

int main() {
    ref_init_tables();
    std::mt19937_64 rng(1234);
    struct Shape { int K, rows; } shapes[] = { {5120, 2304}, {2304, 5120} };
    const int Ts[] = {1,2,3,4,5,7,8};
    const int splits[] = {1, 3, 8, 12};
    long long total = 0, bad = 0;
    for (auto sh : shapes) {
        const int nb = sh.K / 32;
        const int rows = sh.rows;
        std::vector<block_mxfp4> A((size_t) rows * nb);
        for (auto & b : A) { b.e = (uint8_t)(110 + rng() % 30); for (auto & q : b.qs) q = (uint8_t) rng(); }
        // a few extreme e8m0 values and full-range q8
        A[0].e = 0; A[1].e = 254; A[2].e = 255;
        for (int T : Ts) {
            std::vector<block_q8_0> B((size_t) T * nb);
            for (auto & b : B) {
                float d = (float)((rng() % 2000) + 1) / 100000.f;
                b.d = GGML_FP32_TO_FP16(d);
                for (auto & q : b.qs) q = (int8_t)(int)(rng() % 256 - 128);
            }
            std::vector<float> ref((size_t) T * rows);
            for (int t = 0; t < T; ++t)
                for (int r = 0; r < rows; ++r)
                    ref_vec_dot_mxfp4_q8_0(sh.K, &ref[(size_t) t * rows + r], &A[(size_t) r * nb], &B[(size_t) t * nb]);
            for (int nth : splits) {
                std::vector<float> out((size_t) T * rows, -1.f);
                for (int th = 0; th < nth; ++th) {
                    // uneven split incl. odd sizes / offsets
                    int r0 = (int)((long long) rows * th / nth) + (th % 3 == 1 ? 1 : 0);
                    int r1 = (int)((long long) rows * (th + 1) / nth) + ((th + 1) % 3 == 1 ? 1 : 0);
                    if (th == nth - 1) r1 = rows;
                    if (r1 > rows) r1 = rows;
                    if (r0 >= r1) continue;
                    // also split the column range for odd T on some threads
                    int c0 = 0, c1 = T;
                    if (T >= 3 && th % 2 == 1) { c1 = T / 2; }
                    for (int pass = 0; pass < (c1 < T ? 2 : 1); ++pass) {
                        int ca = pass ? c1 : c0, cb = pass ? T : c1;
                        bool ok = wp_gemm_mxfp4_q8_0(sh.K, r1 - r0, cb - ca,
                            &A[(size_t) r0 * nb], nb * sizeof(block_mxfp4),
                            &B[(size_t) ca * nb], nb * sizeof(block_q8_0),
                            &out[(size_t) ca * rows + r0], rows);
                        if (!ok) { printf("kernel declined\n"); return 2; }
                    }
                }
                for (size_t i = 0; i < out.size(); ++i) {
                    ++total;
                    if (memcmp(&out[i], &ref[i], 4) != 0) { if (bad++ < 5) printf("MISMATCH K=%d T=%d nth=%d i=%zu %a vs %a\n", sh.K, T, nth, i, out[i], ref[i]); }
                }
            }
        }
    }
    printf("compared %lld elements, %lld mismatches (bitwise): %s\n", total, bad, bad ? "FAIL" : "PASS");
    return bad != 0;
}
