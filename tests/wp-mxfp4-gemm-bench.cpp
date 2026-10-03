// MXFP4 x Q8_0 microbench: mainline vec_dot vs wp_gemm_mxfp4_q8_0.
// usage: wp-mxfp4-gemm-bench [threads=8] [T=1] [K=5120] [rows=2304] [experts=48] [iters=3]
// Streams distinct experts from a weight buffer >> L3. Rows are split across
// threads (contiguous ranges, like mul_mat_one_chunk). Reports us/expert (one
// matrix per "expert" here) and GB/s of weight traffic.
#include "wp-mxfp4-ref.h"
#include <cstdio>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <random>

int main(int argc, char ** argv) {
    ref_init_tables();
    int nth = argc > 1 ? atoi(argv[1]) : 8, T = argc > 2 ? atoi(argv[2]) : 1;
    int K = argc > 3 ? atoi(argv[3]) : 5120, rows = argc > 4 ? atoi(argv[4]) : 2304;
    int NE = argc > 5 ? atoi(argv[5]) : 48, iters = argc > 6 ? atoi(argv[6]) : 3;
    const int nb = K / 32;
    const size_t mat = (size_t) rows * nb * sizeof(block_mxfp4);
    std::vector<uint8_t> W(mat * NE);
    std::mt19937_64 rng(7);
    for (auto & b : W) b = (uint8_t) rng();
    std::vector<block_q8_0> B((size_t) T * nb);
    for (auto & b : B) { b.d = GGML_FP32_TO_FP16(0.01f); for (auto & q : b.qs) q = (int8_t)(rng() % 256 - 128); }
    std::vector<float> out_ref((size_t) T * rows), out_new((size_t) T * rows);
    printf("K=%d rows=%d T=%d threads=%d experts=%d (%.1f MB total, %.2f MB/expert)\n", K, rows, T, nth, NE, W.size() / 1e6, mat / 1e6);

    for (int mode = 0; mode < 2; ++mode) {
        std::vector<float> & out = mode ? out_new : out_ref;
        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            std::atomic<int> go{0}; std::atomic<int> ready{0};
            auto t0 = std::chrono::steady_clock::time_point();
            std::vector<std::thread> th;
            for (int t = 0; t < nth; ++t) th.emplace_back([&, t] {
                int r0 = (int)((long long) rows * t / nth), r1 = (int)((long long) rows * (t + 1) / nth);
                ready++; while (!go.load()) {}
                for (int e = 0; e < NE; ++e) {
                    const uint8_t * A = W.data() + mat * e;
                    // barrier-free: each thread streams its own slice of every expert
                    if (mode == 0) {
                        for (int c = 0; c < T; ++c)
                            for (int r = r0; r < r1; ++r)
                                ref_vec_dot_mxfp4_q8_0(K, &out[(size_t) c * rows + r], A + (size_t) r * nb * sizeof(block_mxfp4), &B[(size_t) c * nb]);
                    } else {
                        wp_gemm_mxfp4_q8_0(K, r1 - r0, T, A + (size_t) r0 * nb * sizeof(block_mxfp4), nb * sizeof(block_mxfp4),
                                           B.data(), nb * sizeof(block_q8_0), &out[r0], rows);
                    }
                }
            });
            while (ready.load() < nth) {}
            auto s = std::chrono::steady_clock::now();
            go = 1;
            for (auto & x : th) x.join();
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - s).count();
            if (sec < best) best = sec;
            (void) t0;
        }
        printf("%-12s %8.1f us/expert-matrix  %6.1f GB/s\n", mode ? "wp_gemm" : "vec_dot", best / NE * 1e6, mat * (double) NE / best / 1e9);
    }
    printf("outputs %s\n", memcmp(out_ref.data(), out_new.data(), out_ref.size() * 4) == 0 ? "bit-identical" : "DIFFER");
    return 0;
}
