// Bit-exactness: CPU ml8_4 pack (pipe-ml8-pack-cpu.h) vs ggml_cuda_expert_wire_pack_ml8_4.
// Standalone; build one-off against libggml-hip (see commit message). Uses GPU 0 briefly.
#include "../src/pipeline/pipe-ml8-pack-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

// Weak: --cpu-only links without any GPU library (the GPU reference is absent).
extern "C" __attribute__((weak)) bool ggml_cuda_expert_wire_pack_ml8_4(const float * src, void * dst, int64_t ne);

static size_t g_floats = 0, g_blocks = 0, g_bad_blocks = 0, g_bad_scalar = 0;

static void check(const std::vector<float> & x, const char * what) {
    const size_t n = x.size() / 32 * 32;
    if (!n) { return; }
    std::vector<uint8_t> gpu(n / 32 * 18), cpu(gpu.size()), sc(gpu.size());
    if (!ggml_cuda_expert_wire_pack_ml8_4) { std::printf("no GPU reference linked (use --cpu-only)\n"); std::exit(2); }
    if (!ggml_cuda_expert_wire_pack_ml8_4(x.data(), gpu.data(), (int64_t) n)) { std::printf("GPU pack failed\n"); std::exit(2); }
    pipe_ml8::pack_ml8_4(cpu.data(), x.data(), n);
    for (size_t b = 0; b < n / 32; ++b) pipe_ml8::pack_block_scalar(x.data() + b * 32, sc.data() + b * 18);
    size_t shown = 0;
    for (size_t b = 0; b < n / 32; ++b) {
        if (std::memcmp(&gpu[b * 18], &cpu[b * 18], 18) != 0) {
            g_bad_blocks++;
            if (shown++ < 3) {
                std::printf("MISMATCH [%s] block %zu\n  in:", what, b);
                for (int j = 0; j < 32; ++j) std::printf(" %a", x[b * 32 + j]);
                std::printf("\n  gpu:"); for (int j = 0; j < 18; ++j) std::printf(" %02x", gpu[b * 18 + j]);
                std::printf("\n  cpu:"); for (int j = 0; j < 18; ++j) std::printf(" %02x", cpu[b * 18 + j]);
                std::printf("\n");
            }
        }
        if (std::memcmp(&gpu[b * 18], &sc[b * 18], 18) != 0) g_bad_scalar++;
    }
    g_floats += n; g_blocks += n / 32;
}

// CPU-only (no GPU use): `test-ml8-4-pack-cpu-vs-gpu --cpu-only`. Covers the
// spine's call shape (pipe_expert_wire_pack_matrix with WP_SPINE_WIRE_CPU_PACK=1:
// n_tokens <= 16, n_embd % 32 == 0) -- the whole [rows x n_embd] matrix packed in
// one call must equal row-by-row packing and the per-block scalar reference.
// The GPU-equality part of the claim is only checked by the default mode above.
static int cpu_only_spine_shape() {
    std::mt19937_64 rng(777);
    std::normal_distribution<float> nd(0.f, 1.f);
    size_t bad = 0, cases = 0;
    for (int rows : {1, 2, 3, 6, 8, 16}) {
        for (int n_embd : {32, 2048, 4096}) {
            for (int mode = 0; mode < 3; ++mode) {
                std::vector<float> x((size_t) rows * n_embd);
                for (auto & v : x) v = mode == 0 ? nd(rng) : mode == 1 ? nd(rng) * 1e-4f : (nd(rng) > 0 ? 1.f : -1.f) * 0.9375f;
                const size_t rb = (size_t) n_embd / 32 * 18;
                std::vector<uint8_t> whole(rows * rb), byrow(rows * rb), sc(rows * rb);
                pipe_ml8::pack_ml8_4(whole.data(), x.data(), x.size());
                for (int r = 0; r < rows; ++r) {
                    pipe_ml8::pack_ml8_4(byrow.data() + r * rb, x.data() + (size_t) r * n_embd, n_embd);
                    for (int b = 0; b < n_embd / 32; ++b) {
                        pipe_ml8::pack_block_scalar(x.data() + (size_t) r * n_embd + b * 32, sc.data() + r * rb + b * 18);
                    }
                }
                cases++;
                if (std::memcmp(whole.data(), byrow.data(), whole.size()) != 0) bad++;
                if (std::memcmp(whole.data(), sc.data(), whole.size()) != 0) bad++;
            }
        }
    }
    std::printf("cpu-only spine shape: cases=%zu mismatches=%zu\n", cases, bad);
    return bad ? 1 : 0;
}

// AVX2 kernel vs scalar reference, bitwise, over spine shapes and adversarial inputs.
// Also checks the runtime-dispatched pack_ml8_4 equals the scalar path.
static int cpu_only_avx2_vs_scalar() {
#ifndef PIPE_ML8_PACK_X86
    std::printf("avx2 vs scalar: skipped (not x86-64 GCC/Clang build)\n");
    return 0;
#else
    if (!pipe_ml8::ml8_pack_avx2_supported()) {
        std::printf("avx2 vs scalar: skipped (CPU lacks avx2/fma/f16c)\n");
        return 0;
    }
    std::mt19937_64 rng(4242);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    const float sp[] = {0.f, -0.f, 1e-45f, -1e-45f, 1e-40f, 6e-5f, 6.1e-5f, 65504.f, -65504.f, 65519.f, 65520.f, 65536.f,
                        3.4e38f, -3.4e38f, inf, -inf, nan, -nan, 1.f, -1.f, 0.9375f, -0.9375f, 2.9802322e-8f, 5.96e-8f, 3e-8f};
    const int n_sp = (int) (sizeof(sp) / sizeof(sp[0]));
    size_t cases = 0, bad_blocks = 0, bad_dispatch = 0, blocks = 0;
    std::vector<uint8_t> a, s;
    auto cmp = [&](const std::vector<float> & x, const char * what) {
        const size_t n = x.size() / 32 * 32;
        if (!n) { return; }
        a.assign(n / 32 * 18, 0xAA); s.assign(a.size(), 0x55);
        std::vector<uint8_t> d(a.size(), 0x33);
        pipe_ml8::pack_ml8_4_avx2(a.data(), x.data(), n);
        pipe_ml8::pack_ml8_4_scalar(s.data(), x.data(), n);
        pipe_ml8::pack_ml8_4(d.data(), x.data(), n);
        cases++; blocks += n / 32;
        for (size_t b = 0; b < n / 32; ++b) {
            if (std::memcmp(&a[b * 18], &s[b * 18], 18) != 0) {
                if (bad_blocks++ < 3) {
                    std::printf("MISMATCH avx2 vs scalar [%s] block %zu\n  in:", what, b);
                    for (int j = 0; j < 32; ++j) std::printf(" %a", x[b * 32 + j]);
                    std::printf("\n  avx2:"); for (int j = 0; j < 18; ++j) std::printf(" %02x", a[b * 18 + j]);
                    std::printf("\n  scal:"); for (int j = 0; j < 18; ++j) std::printf(" %02x", s[b * 18 + j]);
                    std::printf("\n");
                }
            }
        }
        if (std::memcmp(d.data(), s.data(), d.size()) != 0) bad_dispatch++;
    };
    // 1) spine shapes with the same distributions as the GPU test
    for (int rows = 1; rows <= 16; ++rows) {
        for (int n_embd : {32, 2048, 4096}) {
            for (int mode = 0; mode < 6; ++mode) {
                std::vector<float> x((size_t) rows * n_embd);
                for (auto & v : x) {
                    switch (mode) {
                        case 0: v = nd(rng); break;
                        case 1: v = nd(rng) * 1e-4f; break;
                        case 2: { float t = nd(rng); v = t * t * t * t * (ud(rng) < 0 ? -1.f : 1.f); break; }
                        case 3: v = ud(rng); break;
                        case 4: v = (nd(rng) > 0 ? 1.f : -1.f) * 0.9375f; break;   // exact centroids
                        default: v = (ud(rng) < 0 ? -1.f : 1.f) * std::ldexp(1.f, (int) (rng() % 60) - 30) * (1.f + ud(rng) * 0.5f); break;
                    }
                }
                cmp(x, "spine-shape");
            }
        }
    }
    // 2) exact ties: amax = 1 (fp16-exact, inv = 1 exact), entries exactly at midpoints
    //    between adjacent centroids, +-1 ulp around them, and at the centroids.
    {
        std::vector<float> x;
        for (int rep = 0; rep < 2000; ++rep) {
            for (int i = 0; i < 15; ++i) {
                const float mid = 0.5f * (pipe_ml8::k_cent4[i] + pipe_ml8::k_cent4[i + 1]);
                float blk[32];
                for (auto & b : blk) b = ud(rng);
                blk[0] = (rep & 1) ? 1.f : -1.f;
                blk[1] = mid; blk[2] = -mid; blk[3] = std::nextafterf(mid, 1.f); blk[4] = std::nextafterf(mid, -1.f);
                blk[5] = pipe_ml8::k_cent4[i]; blk[6] = pipe_ml8::k_cent4[i + 1]; blk[7] = 0.f; blk[8] = -0.f;
                for (int j = 0; j < 32; ++j) x.push_back(blk[j] * std::ldexp(1.f, (rep % 17) - 8));
            }
        }
        cmp(x, "ties-pow2");
        // ties at non-pow2 scale: amax = d (fp16-exact) so inv = 1/d rounded; midpoints scaled by d
        x.clear();
        for (float d : {0.5f, 0.75f, 3.f, 1234.f, 60000.f}) {
            for (int i = 0; i < 15; ++i) {
                const float mid = 0.5f * (pipe_ml8::k_cent4[i] + pipe_ml8::k_cent4[i + 1]);
                float blk[32] = {};
                blk[0] = d;
                blk[1] = mid * d; blk[2] = -mid * d; blk[3] = pipe_ml8::k_cent4[i] * d; blk[4] = 0.f;
                for (int j = 5; j < 32; ++j) blk[j] = (ud(rng)) * d;
                for (float v : blk) x.push_back(v);
            }
        }
        cmp(x, "ties-scaled");
        // dense sweep around each midpoint (1-ulp steps)
        x.clear();
        for (int i = 0; i < 15; ++i) {
            const float mid = 0.5f * (pipe_ml8::k_cent4[i] + pipe_ml8::k_cent4[i + 1]);
            float v = mid; for (int st = 0; st < 2000; ++st) v = std::nextafterf(v, -2.f);
            for (int st = 0; st < 4000; ++st) {
                x.push_back(st % 32 == 0 ? 1.f : v);
                v = std::nextafterf(v, 2.f);
            }
        }
        while (x.size() % 32) x.push_back(0.f);
        cmp(x, "midpoint-sweep");
    }
    // 3) zeros, denormals, +-max, specials (NaN/Inf propagate exactly as the scalar path)
    {
        std::vector<float> z(32 * 64, 0.f); cmp(z, "zeros");
        std::vector<float> nz(32 * 64, -0.f); cmp(nz, "neg-zeros");
        std::vector<float> dn;
        for (int b = 0; b < 20000; ++b) { const float sc = std::ldexp(1.f, -149 + (b % 150)); for (int j = 0; j < 32; ++j) dn.push_back(nd(rng) * sc); }
        cmp(dn, "denormal");
        std::vector<float> mx;
        for (int b = 0; b < 20000; ++b) { for (int j = 0; j < 32; ++j) mx.push_back(((rng() & 1) ? 1.f : -1.f) * (rng() % 3 == 0 ? 3.4028235e38f : rng() % 2 ? 65504.f : nd(rng))); }
        cmp(mx, "max");
        std::vector<float> spx;
        std::uniform_int_distribution<int> pick(0, n_sp - 1);
        for (int b = 0; b < 200000; ++b) {
            for (int j = 0; j < 32; ++j) {
                const int mode = b % 3;
                spx.push_back(mode == 0 ? sp[pick(rng)] : mode == 1 ? (j < 3 ? sp[pick(rng)] : nd(rng)) : sp[pick(rng)] * (ud(rng) * 0.5f + 0.5f));
            }
        }
        cmp(spx, "special");
        std::vector<float> rb;
        std::uniform_int_distribution<uint32_t> bits;
        for (size_t i = 0; i < 32 * 200000; ++i) { uint32_t u = bits(rng); float f; std::memcpy(&f, &u, 4); rb.push_back(f); }
        cmp(rb, "randbits");
    }
    std::printf("avx2 vs scalar: cases=%zu blocks=%zu block_mismatches=%zu dispatch_mismatches=%zu\n",
                cases, blocks, bad_blocks, bad_dispatch);
    return (bad_blocks || bad_dispatch) ? 1 : 0;
#endif
}

// Micro-benchmark, median of 1000 runs per configuration (6 rows x 4096 floats).
static void cpu_only_bench() {
    std::mt19937_64 rng(99);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x(6 * 4096); for (auto & v : x) v = nd(rng);
    std::vector<uint8_t> o(x.size() / 32 * 18);
    auto median_us = [&](auto && fn) {
        for (int w = 0; w < 50; ++w) fn();
        std::vector<double> t(1000);
        for (auto & ti : t) {
            auto t0 = std::chrono::steady_clock::now();
            fn();
            ti = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        }
        std::nth_element(t.begin(), t.begin() + 500, t.end());
        return t[500];
    };
    const double sc = median_us([&] { pipe_ml8::pack_ml8_4_scalar(o.data(), x.data(), x.size()); });
    std::printf("bench 6x4096 scalar: median %.2f us\n", sc);
#ifdef PIPE_ML8_PACK_X86
    if (pipe_ml8::ml8_pack_avx2_supported()) {
        const double av = median_us([&] { pipe_ml8::pack_ml8_4_avx2(o.data(), x.data(), x.size()); });
        std::printf("bench 6x4096 avx2:   median %.2f us (speedup %.2fx)\n", av, sc / av);
    }
#endif
}

int main(int argc, char ** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--cpu-only") == 0) {
        int rc = cpu_only_spine_shape();
        rc |= cpu_only_avx2_vs_scalar();
        cpu_only_bench();
        return rc ? 1 : 0;
    }
    std::mt19937_64 rng(12345);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(-1.f, 1.f);
    const float scales[] = {1e-6f, 1e-3f, 0.05f, 1.f, 30.f, 1000.f, 6e4f};
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    for (int rep = 0; rep < 6; ++rep)
    for (int nmul = 1; nmul <= 8; ++nmul) {
        for (int tail : {0, 5, 31, 17}) {
            const size_t n = 5120 * nmul + tail;
            std::vector<float> x(n);
            for (float s : scales) {
                for (auto & v : x) v = nd(rng) * s;                 check(x, "normal");
                for (auto & v : x) { float t = nd(rng); v = t * t * t * t * s * (ud(rng) < 0 ? -1.f : 1.f); } check(x, "heavy");
                for (auto & v : x) v = ud(rng) * s;                 check(x, "uniform");
            }
        }
    }
    // exact ties / centroid hits / midpoints: values = (c_i + c_j)/2 * amax with amax chosen so inv exact
    {
        std::vector<float> x;
        for (int i = 0; i < 15; ++i) for (int k = 0; k < 64; ++k) {
            const float mid = 0.5f * (pipe_ml8::k_cent4[i] + pipe_ml8::k_cent4[i + 1]);
            float blk[32]; for (auto & b : blk) b = (ud(rng) < 0 ? -1.f : 1.f) * (0.1f + 0.9f * (ud(rng) * 0.5f + 0.5f));
            blk[0] = 1.0f;  // amax = 1 (or 2^k scaled) -> inv exact
            blk[1] = mid; blk[2] = std::nextafterf(mid, 1.f); blk[3] = std::nextafterf(mid, -1.f);
            blk[4] = pipe_ml8::k_cent4[i]; blk[5] = -mid; blk[6] = std::nextafterf(-mid, 1.f);
            const float sc = std::ldexp(1.f, (k % 21) - 10);
            for (auto b : blk) x.push_back(b * sc);
        }
        check(x, "ties");
        // dense sweep across [-1,1] around every midpoint, several scales incl non-pow2
        for (float sc : {1.f, 3.f, 0.7f, 1234.5f}) {
            x.clear();
            for (int i = 0; i < 15; ++i) {
                const float mid = 0.5f * (pipe_ml8::k_cent4[i] + pipe_ml8::k_cent4[i + 1]);
                float v = mid; for (int s = 0; s < 8; ++s) v = std::nextafterf(v, -2.f);
                for (int s = 0; s < 31 * 16; ++s) { x.push_back((s % 32 == 0 ? 1.f : v) * sc); if (s % 32) v = std::nextafterf(v, 2.f); }
                while (x.size() % 32) x.push_back(sc);
            }
            check(x, "midpoint-sweep");
        }
    }
    // special values
    {
        std::vector<float> x;
        const float sp[] = {0.f, -0.f, 1e-45f, -1e-45f, 1e-40f, 6e-5f, 6.1e-5f, 65504.f, 65519.f, 65520.f, 65536.f, 1e30f, -1e30f,
                            3.4e38f, inf, -inf, nan, -nan, 1.f, -1.f, 2.9802322e-8f, 5.96e-8f, 3e-8f, 1e-8f};
        std::uniform_int_distribution<int> pick(0, (int) (sizeof(sp) / sizeof(sp[0])) - 1);
        for (int b = 0; b < 200000; ++b) {
            const int mode = b % 4;
            for (int j = 0; j < 32; ++j) {
                float v = mode == 0 ? sp[pick(rng)] : mode == 1 ? (j < 4 ? sp[pick(rng)] : nd(rng)) : mode == 2 ? (j == 0 ? sp[pick(rng)] : nd(rng) * 1e-6f) : sp[pick(rng)] * (ud(rng) * 0.5f + 0.5f);
                x.push_back(v);
            }
        }
        check(x, "special");
        std::vector<float> z(32 * 64, 0.f); check(z, "zeros");
        // denormal-only and fp16-subnormal-scale blocks
        x.clear();
        for (int b = 0; b < 100000; ++b) { const float s = std::ldexp(1.f, -149 + (b % 40)); for (int j = 0; j < 32; ++j) x.push_back(nd(rng) * s); }
        check(x, "denormal");
        // random bit patterns
        x.clear();
        std::uniform_int_distribution<uint32_t> bits;
        for (size_t i = 0; i < 32 * 200000; ++i) { uint32_t u = bits(rng); float f; std::memcpy(&f, &u, 4); x.push_back(f); }
        check(x, "randbits");
        // random bits with sane exponent
        x.clear();
        for (size_t i = 0; i < 32 * 200000; ++i) { uint32_t u = bits(rng); u = (u & 0x807fffffu) | ((uint32_t) (100 + (bits(rng) % 40)) << 23); float f; std::memcpy(&f, &u, 4); x.push_back(f); }
        check(x, "randexp");
    }
    std::printf("floats=%zu blocks=%zu mismatching_blocks(avx2 vs gpu)=%zu mismatching_blocks(scalar vs gpu)=%zu\n",
                g_floats, g_blocks, g_bad_blocks, g_bad_scalar);
    // timing
    std::vector<float> x(32768); for (auto & v : x) v = nd(rng);
    std::vector<uint8_t> o(32768 / 32 * 18);
    for (int n : {5120, 32768}) {
        for (int w = 0; w < 50; ++w) pipe_ml8::pack_ml8_4(o.data(), x.data(), n);
        const int it = 2000; auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < it; ++i) pipe_ml8::pack_ml8_4(o.data(), x.data(), n);
        const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / it;
        std::printf("cpu pack n=%d: %.2f us/call\n", n, us);
    }
    return (g_bad_blocks || g_bad_scalar) ? 1 : 0;
}
