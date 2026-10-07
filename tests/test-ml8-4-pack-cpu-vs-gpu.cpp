// Bit-exactness: CPU ml8_4 pack (pipe-ml8-pack-cpu.h) vs ggml_cuda_expert_wire_pack_ml8_4.
// Standalone; build one-off against libggml-hip (see commit message). Uses GPU 0 briefly.
#include "../src/pipeline/pipe-ml8-pack-cpu.h"
#include <chrono>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

extern "C" bool ggml_cuda_expert_wire_pack_ml8_4(const float * src, void * dst, int64_t ne);

static size_t g_floats = 0, g_blocks = 0, g_bad_blocks = 0, g_bad_scalar = 0;

static void check(const std::vector<float> & x, const char * what) {
    const size_t n = x.size() / 32 * 32;
    if (!n) { return; }
    std::vector<uint8_t> gpu(n / 32 * 18), cpu(gpu.size()), sc(gpu.size());
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

int main() {
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
