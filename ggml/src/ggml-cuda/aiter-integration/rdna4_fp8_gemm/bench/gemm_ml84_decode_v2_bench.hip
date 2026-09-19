// bench/gemm_ml84_decode_v2_bench.hip — MAD-305 ML8_4 decode-v2 bench.
//
// Standalone (built directly with hipcc, like bench/gemm_ml84_bench.hip --
// see ../build.sh's "gemm_ml84_bench" section and this file's own build
// command in the header comment below). Compares the existing production
// decode launcher (rdna4_gemm_ml84_trfeed_decode_splitk, gemm_ml84_prod.hip)
// against the new v2 launcher (rdna4_gemm_ml84_decode_v2,
// gemm_ml84_decode_v2.hip) at every M in {1,4,8,16,32} x (N,K) in
// {(5120,5120),(34816,5120),(5120,17408),(248320,5120)} (gate_up's fused N,
// K=5120 -- so (34816,5120), and (5120,17408) for the "N=5120,K=17408"
// workgroup-starved shape referenced by the existing decode split heuristic's
// own comments, per the task's N/K axis list).
//
// Oracle: same discipline as gemm_ml84_bench.hip -- dequantize a SAMPLE of
// (m,n) entries from the ground-truth block_ml8_4 weight + LUT (not the
// packed B_nib, so this is an independent check of the whole pipeline) and
// compare against both kernels' full output at those same entries. Both
// kernels are also compared directly against EACH OTHER over the full
// output (cheap: same device buffer sizes, no re-dequant needed) --
// that catches a mismatch between the two kernels even if the oracle
// sampling happens to miss it.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "ml84_trfeed_layout.h"
#include "trfeed_ml84_kernels.h"
#include "../gemm_capi.h"
#include "../gemm_ml84_decode_v2.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

struct ShapeStat {
    int M, N, K;
    bool old_ok = false, v2_ok = false;
    double old_us = 0, v2_us = 0;
    double old_gbps = 0, v2_gbps = 0;
    double max_abs_diff_old_vs_v2 = 0, max_rel_diff_old_vs_v2 = 0;
    double max_rel_diff_v2_vs_oracle = 0, max_rel_diff_old_vs_oracle = 0;
};

// bytes = N*K/2 (B_nib) + n_groups_k*16 (lut) + n_groups_k*N*4 (b_scale_g)
// + M_pad*K (A fp8) + M_pad*4 (a_scale) + M*N*4 (C f32) -- per the task
// brief's byte-accounting formula, generalized with the lut/b_scale_g terms
// (the old formula "N*K/2 + LUT + scales + A + C" spelled out).
static double bytes_per_launch(int M, int M_pad, int N, int K) {
    const int n_groups_k = K / QK_ML8;
    double b = 0.0;
    b += (double) N * K / 2.0;                 // B_nib
    b += (double) n_groups_k * 16.0;            // lut
    b += (double) n_groups_k * N * 4.0;         // b_scale_g
    b += (double) M_pad * K;                    // A (fp8)
    b += (double) M_pad * 4.0;                  // a_scale
    b += (double) M * N * 4.0;                  // C (fp32)
    return b;
}

// ─────────────────────────────────────────────────────────────────────────
// ML84_V2_SELFCHECK=1: host-side dry run of the device kernel's OWN
// addressing formulas (not a re-derivation -- literally the same
// expressions as gemm_ml84_decode_v2.hip's ml84_v2_group /
// ml84_decode_v2_kernel, including calling the real ml84_tile_nib_base via
// this file's #include of ml84_trfeed_layout.h), for lanes {0,1,31} of
// waves {0,7} at the first and last block ((blockIdx.x,blockIdx.z) =
// (0,0) and (nblk-1,ks-1)), printing every computed byte offset and
// asserting it against the actual allocation size -- BEFORE any kernel
// launch, so a bad address shows up as a printed assertion on the host
// instead of a GPU memory fault. Added in response to the coordinator's
// 2026-09-19 report (a confirmed sign-extension bug in a since-removed
// __builtin_amdgcn_readfirstlane hi/lo pointer reconstruction -- see
// gemm_ml84_decode_v2.hip's ml84_v2_group comment).
//
// `dtm`/`dbk`/`ks` here mirror rdna4_gemm_ml84_decode_v2's OWN internal
// shape decision (gemm_ml84_decode_v2.hip: dtm = M<=16?1:2, dbk =
// K%128==0?128:64, ks = MT_ML8_4_DECODE_V2_KS env override else the
// split_k_for fill law) -- duplicated here rather than queried from the
// launcher because none of those are exported; if that heuristic ever
// changes, this copy must move with it (both are tiny and adjacent to
// their own call sites, so the duplication is easy to keep honest by eye).
static int ml84_v2_split_k_for_mirror(int nblk, int M) {
    const int fill = (M <= 24) ? 110 : 78;
    for (int ks = 1; ks < 4; ks <<= 1) if (nblk * ks >= fill) return ks;
    return 4;
}

static bool ml84_v2_selfcheck(int M, int N, int K,
                              size_t B_nib_bytes, size_t lut_bytes, size_t A_bytes, size_t P_bytes) {
    if (!std::getenv("ML84_V2_SELFCHECK")) return true;
    constexpr int V2_BN = 128;   // must match gemm_ml84_decode_v2.hip's V2_DWN(8)*16
    const int dtm = (M <= 16) ? 1 : 2;
    const int dbk = (K % 128 == 0) ? 128 : 64;
    const int nblk = (N + V2_BN - 1) / V2_BN;
    const char* env_ks = std::getenv("MT_ML8_4_DECODE_V2_KS");
    const int ks = env_ks ? std::max(1, std::atoi(env_ks)) : ml84_v2_split_k_for_mirror(nblk, M);
    const int groups_per_slab = dbk / QK_ML8;
    printf("--- ML84_V2_SELFCHECK M=%d N=%d K=%d dtm=%d dbk=%d ks=%d nblk=%d ---\n", M, N, K, dtm, dbk, ks, nblk);
    printf("    B_nib_bytes=%zu lut_bytes=%zu A_bytes=%zu P_bytes=%zu\n", B_nib_bytes, lut_bytes, A_bytes, P_bytes);

    bool ok = true;
    const int block_x[2] = {0, std::max(0, nblk - 1)};
    const int block_z[2] = {0, std::max(0, ks - 1)};
    const int waves[2]   = {0, 7};
    const int lanes[3]   = {0, 1, 31};

    for (int bx : block_x) {
        for (int bz : block_z) {
            const int n0 = bx * V2_BN;
            const int slabs = K / dbk;
            const int spb = (slabs + ks - 1) / ks;
            const int s_lo = bz * spb, s_hi = std::min(slabs, s_lo + spb);
            const int slab_probe[2] = {s_lo, std::max(s_lo, s_hi - 1)};
            for (int s : slab_probe) {
                if (s < s_lo || s >= s_hi) continue;
                const int k0 = s * dbk;

                // A: rows 0..dtm*16-1, this slab's full DBK-wide int4 fill --
                // check the LAST 16-byte vector read in the slab (largest offset).
                for (int r : {0, dtm * 16 - 1}) {
                    const size_t a_off = (size_t) r * K + k0 + (dbk - 16);
                    const size_t a_need = a_off + 16;
                    printf("    A  bx=%d bz=%d s=%d r=%d off=%zu need<=%zu\n", bx, bz, s, r, a_off, a_need);
                    if (a_need > A_bytes) { printf("      *** A OOB ***\n"); ok = false; }
                }

                for (int g_local = 0; g_local < groups_per_slab; ++g_local) {
                    const int k_abs0 = k0 + g_local * QK_ML8;
                    const int g = k_abs0 / QK_ML8;

                    const size_t lut_off = (size_t) g * 16;
                    const size_t lut_need = lut_off + 16;
                    printf("    LUT bx=%d bz=%d s=%d g=%d off=%zu need<=%zu\n", bx, bz, s, g, lut_off, lut_need);
                    if (lut_need > lut_bytes) { printf("      *** LUT OOB ***\n"); ok = false; }

                    for (int step : {0, 3}) {
                        const int kt = (k_abs0 + step * 16) / 16;
                        for (int w : waves) {
                            const int nt = n0 / 16 + w;
                            const size_t nib_base = ml84_tile_nib_base(kt, nt, N / 16);
                            for (int ln : lanes) {
                                const size_t b_off = nib_base + (size_t) ln * 4;
                                const size_t b_need = b_off + 4;
                                printf("    B_nib bx=%d bz=%d s=%d g=%d step=%d wave=%d lane=%d kt=%d nt=%d off=%zu need<=%zu\n",
                                       bx, bz, s, g, step, w, ln, kt, nt, b_off, b_need);
                                if (b_need > B_nib_bytes) { printf("      *** B_nib OOB ***\n"); ok = false; }
                            }
                        }
                    }
                }
            }

            if (ks > 1) {
                const int nhi = std::min(n0 + V2_BN, N);
                for (int m : {0, std::max(0, M - 1)}) {
                    for (int nn : {n0, std::max(n0, nhi - 1)}) {
                        const size_t p_off = ((size_t) bz * 32 + m) * N + nn;
                        const size_t p_need = (p_off + 1) * sizeof(float);
                        printf("    P  bx=%d bz=%d m=%d n=%d off=%zu need<=%zu\n", bx, bz, m, nn, p_off, p_need);
                        if (p_need > P_bytes) { printf("      *** P OOB ***\n"); ok = false; }
                    }
                }
            }
        }
    }
    printf("--- ML84_V2_SELFCHECK %s ---\n", ok ? "PASS" : "*** FAIL ***");
    return ok;
}

static int run_shape(int M, int N, int K, int iters, ShapeStat* out) {
    out->M = M; out->N = N; out->K = K;
    if (K % QK_ML8 != 0 || N % 16 != 0) {
        fprintf(stderr, "shape M=%d N=%d K=%d skipped: K%%64 or N%%16 violated\n", M, N, K);
        return 3;
    }
    const int n_groups_k = K / QK_ML8;
    const int M_pad = 32;   // the OLD launcher's fixed decode tile requires exactly 32

    std::mt19937 rng(0x4D4C5632u ^ (uint32_t) (M * 1000003 + N * 7919 + K));
    std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);
    std::uniform_int_distribution<int> nib_dist(0, 15);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);

    // ---- ground-truth ml8_4 weight blocks: w[N][n_groups_k] ----
    std::vector<block_ml8_4> w((size_t) N * n_groups_k);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < n_groups_k; ++g) {
            block_ml8_4& blk = w[(size_t) n * n_groups_k + g];
            blk.scale = scale_dist(rng);
            for (int i = 0; i < QK_ML8 / 2; ++i) blk.qs[i] = (uint8_t) (nib_dist(rng) | (nib_dist(rng) << 4));
        }
    }
    // ---- per-group LUT: 16 sorted signed e4m3 values in [-1,1] ----
    std::vector<uint8_t> lut((size_t) n_groups_k * 16);
    for (int g = 0; g < n_groups_k; ++g) {
        std::vector<float> centroids(16);
        for (int i = 0; i < 16; ++i) centroids[i] = -1.0f + 2.0f * (float) i / 15.0f;
        for (int i = 0; i < 16; ++i) centroids[i] += 0.02f * a_dist(rng);
        std::sort(centroids.begin(), centroids.end());
        for (int i = 0; i < 16; ++i) lut[(size_t) g * 16 + i] = ml84_encode_e4m3(std::max(-1.0f, std::min(1.0f, centroids[i])));
    }
    auto dequant_w = [&](int n, int k) -> float {
        const int g = k / QK_ML8;
        const block_ml8_4& blk = w[(size_t) n * n_groups_k + g];
        const int kk = k % QK_ML8;
        const uint8_t packed = blk.qs[kk / 2];
        const uint8_t idx = (kk & 1) ? ((packed >> 4) & 0x0Fu) : (packed & 0x0Fu);
        return ml84_decode_e4m3(lut[(size_t) g * 16 + idx]) * blk.scale;
    };

    // ---- fp32 A rows + per-row a_scale = amax/448, e4m3-encoded ----
    std::vector<float> Af((size_t) M_pad * K, 0.0f);
    std::vector<float> a_scale(M_pad, 0.0f);
    std::vector<uint8_t> Ah((size_t) M_pad * K, 0);
    for (int m = 0; m < M; ++m) {
        float amax = 1e-6f;
        for (int k = 0; k < K; ++k) { Af[(size_t) m * K + k] = a_dist(rng); amax = std::max(amax, std::fabs(Af[(size_t) m * K + k])); }
        a_scale[m] = amax / 448.0f;
        for (int k = 0; k < K; ++k) Ah[(size_t) m * K + k] = ml84_encode_e4m3(Af[(size_t) m * K + k] / a_scale[m]);
    }

    // ---- oracle samples ----
    struct Sample { int i, j; float ref; };
    std::vector<Sample> samples;
    const int SI = std::min(M, 8), SJ = std::min(N, 32);
    for (int si = 0; si < SI; si++) {
        const int i = (int) ((size_t) si * M / SI);
        for (int sj = 0; sj < SJ; sj++) {
            const int j = (int) ((size_t) sj * N / SJ);
            float acc = 0.f;
            for (int k = 0; k < K; k++) acc += ml84_decode_e4m3(Ah[(size_t) i * K + k]) * dequant_w(j, k);
            samples.push_back({i, j, acc * a_scale[i]});
        }
    }

    // ---- pack B_nib + b_scale_g ----
    std::vector<uint8_t> B_nib_h((size_t) N * K / 2);
    std::vector<float> b_scale_g_h((size_t) n_groups_k * N);
    ml84_trfeed_pack_host(w.data(), N, K, B_nib_h.data(), b_scale_g_h.data());

    uint8_t *d_lut = nullptr, *d_B_nib = nullptr, *d_A = nullptr;
    float *d_b_scale_g = nullptr, *d_a_scale = nullptr, *d_C_old = nullptr, *d_C_v2 = nullptr;
    HCK(hipMalloc(&d_lut, lut.size()));
    HCK(hipMalloc(&d_B_nib, B_nib_h.size()));
    HCK(hipMalloc(&d_b_scale_g, b_scale_g_h.size() * sizeof(float)));
    HCK(hipMalloc(&d_A, Ah.size()));
    HCK(hipMalloc(&d_a_scale, a_scale.size() * sizeof(float)));
    HCK(hipMalloc(&d_C_old, (size_t) M * N * sizeof(float)));
    HCK(hipMalloc(&d_C_v2, (size_t) M * N * sizeof(float)));
    HCK(hipMemcpy(d_lut, lut.data(), lut.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_B_nib, B_nib_h.data(), B_nib_h.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_b_scale_g, b_scale_g_h.data(), b_scale_g_h.size() * sizeof(float), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_A, Ah.data(), Ah.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_a_scale, a_scale.data(), a_scale.size() * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream = 0;
    const int n_splits_old = rdna4_ml84_trfeed_splitk_default_splits(N, K);

    // ---- correctness: one launch each, then compare ----
    HCK(hipMemset(d_C_old, 0, (size_t) M * N * sizeof(float)));
    HCK(hipMemset(d_C_v2, 0, (size_t) M * N * sizeof(float)));
    // Old launcher always computes/writes the full M_pad=32 rows into a
    // 32-row scratch, then the real wiring copies only the first M rows out
    // (ml8.cu:~3360); reproduce that here with a 32-row scratch of our own.
    float* d_C_old_pad = nullptr;
    HCK(hipMalloc(&d_C_old_pad, (size_t) M_pad * N * sizeof(float)));
    const hipError_t rc_old = rdna4_gemm_ml84_trfeed_decode_splitk(
        d_A, d_B_nib, d_lut, d_C_old_pad, d_a_scale, d_b_scale_g, M_pad, N, K, n_splits_old, stream);
    out->old_ok = (rc_old == hipSuccess);
    if (out->old_ok) {
        HCK(hipMemcpy(d_C_old, d_C_old_pad, (size_t) M * N * sizeof(float), hipMemcpyDeviceToDevice));
        HCK(hipStreamSynchronize(stream));
    }
    hipFree(d_C_old_pad);

    // Host-side dry run of the SAME addressing arithmetic the v2 kernel
    // will use, before it ever touches the GPU (env ML84_V2_SELFCHECK=1;
    // no-op otherwise). The P scratch's assertion size is the LAUNCHER'S
    // compiled ceiling (ml84_decode_v2_max_n() * 32 rows * 4 splits *
    // 4 bytes), not this shape's own N -- that is what the persistent
    // allocator actually sized, per gemm_ml84_decode_v2.hip's
    // ml84_v2_scratch_get.
    const size_t v2_p_scratch_bytes = (size_t) 4 * 32 * (size_t) ml84_decode_v2_max_n() * sizeof(float);
    if (!ml84_v2_selfcheck(M, N, K, B_nib_h.size(), lut.size(), Ah.size(), v2_p_scratch_bytes)) {
        fprintf(stderr, "  ML84_V2_SELFCHECK failed at M=%d N=%d K=%d -- refusing to launch v2\n", M, N, K);
        hipFree(d_lut); hipFree(d_B_nib); hipFree(d_b_scale_g); hipFree(d_A); hipFree(d_a_scale);
        hipFree(d_C_old); hipFree(d_C_v2);
        return 3;
    }

    out->v2_ok = rdna4_gemm_ml84_decode_v2(M, d_A, d_B_nib, d_lut, d_C_v2, d_a_scale, d_b_scale_g,
                                           M_pad, N, K, /*n_splits=*/0, stream);
    if (out->v2_ok) {
        HCK(hipGetLastError());
        HCK(hipStreamSynchronize(stream));
    }

    std::vector<float> C_old_h, C_v2_h;
    if (out->old_ok) { C_old_h.resize((size_t) M * N); HCK(hipMemcpy(C_old_h.data(), d_C_old, C_old_h.size() * sizeof(float), hipMemcpyDeviceToHost)); }
    if (out->v2_ok)  { C_v2_h.resize((size_t) M * N);  HCK(hipMemcpy(C_v2_h.data(), d_C_v2, C_v2_h.size() * sizeof(float), hipMemcpyDeviceToHost)); }

    if (out->old_ok && out->v2_ok) {
        double max_abs = 0.0, max_rel = 0.0;
        for (size_t i = 0; i < C_old_h.size(); ++i) {
            const double d = std::fabs((double) C_old_h[i] - (double) C_v2_h[i]);
            const double r = d / (std::fabs((double) C_old_h[i]) + 1e-6);
            max_abs = std::max(max_abs, d);
            max_rel = std::max(max_rel, r);
        }
        out->max_abs_diff_old_vs_v2 = max_abs;
        out->max_rel_diff_old_vs_v2 = max_rel;
    }
    for (const auto& s : samples) {
        const size_t idx = (size_t) s.i * N + s.j;
        if (out->v2_ok && idx < C_v2_h.size()) {
            const double r = std::fabs((double) C_v2_h[idx] - s.ref) / (std::fabs((double) s.ref) + 1e-3);
            out->max_rel_diff_v2_vs_oracle = std::max(out->max_rel_diff_v2_vs_oracle, r);
        }
        if (out->old_ok && idx < C_old_h.size()) {
            const double r = std::fabs((double) C_old_h[idx] - s.ref) / (std::fabs((double) s.ref) + 1e-3);
            out->max_rel_diff_old_vs_oracle = std::max(out->max_rel_diff_old_vs_oracle, r);
        }
    }

    // ---- timing: 50 launches each, hipEvents ----
    const double bpl = bytes_per_launch(M, M_pad, N, K);
    if (out->old_ok) {
        float* d_C_pad = nullptr;
        HCK(hipMalloc(&d_C_pad, (size_t) M_pad * N * sizeof(float)));
        for (int i = 0; i < 5; i++) rdna4_gemm_ml84_trfeed_decode_splitk(d_A, d_B_nib, d_lut, d_C_pad, d_a_scale, d_b_scale_g, M_pad, N, K, n_splits_old, stream);
        HCK(hipDeviceSynchronize());
        hipEvent_t s0, e0; hipEventCreate(&s0); hipEventCreate(&e0);
        hipEventRecord(s0, stream);
        for (int i = 0; i < iters; i++) rdna4_gemm_ml84_trfeed_decode_splitk(d_A, d_B_nib, d_lut, d_C_pad, d_a_scale, d_b_scale_g, M_pad, N, K, n_splits_old, stream);
        hipEventRecord(e0, stream); hipEventSynchronize(e0);
        float ms = 0; hipEventElapsedTime(&ms, s0, e0);
        hipEventDestroy(s0); hipEventDestroy(e0);
        out->old_us = (double) ms * 1000.0 / iters;
        out->old_gbps = bpl / (out->old_us * 1e-6) / 1e9;
        hipFree(d_C_pad);
    }
    if (out->v2_ok) {
        for (int i = 0; i < 5; i++) rdna4_gemm_ml84_decode_v2(M, d_A, d_B_nib, d_lut, d_C_v2, d_a_scale, d_b_scale_g, M_pad, N, K, 0, stream);
        HCK(hipDeviceSynchronize());
        hipEvent_t s1, e1; hipEventCreate(&s1); hipEventCreate(&e1);
        hipEventRecord(s1, stream);
        for (int i = 0; i < iters; i++) rdna4_gemm_ml84_decode_v2(M, d_A, d_B_nib, d_lut, d_C_v2, d_a_scale, d_b_scale_g, M_pad, N, K, 0, stream);
        hipEventRecord(e1, stream); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, s1, e1);
        hipEventDestroy(s1); hipEventDestroy(e1);
        out->v2_us = (double) ms * 1000.0 / iters;
        out->v2_gbps = bpl / (out->v2_us * 1e-6) / 1e9;
    }

    hipFree(d_lut); hipFree(d_B_nib); hipFree(d_b_scale_g); hipFree(d_A); hipFree(d_a_scale);
    hipFree(d_C_old); hipFree(d_C_v2);
    return 0;
}

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 50;
    const int Ms[] = {1, 4, 8, 16, 32};
    struct NK { int N, K; };
    const NK nks[] = {
        {5120, 5120},     // representative decode GEMM
        {34816, 5120},    // fused gate_up
        {5120, 17408},    // workgroup-starved (N/128=40) shape from the old splitk heuristic's own tuning notes
        {248320, 5120},   // output head
    };

    printf("%-6s %-8s %-8s | %-8s %-8s %-9s | %-8s %-8s %-9s | %-10s %-10s | %-10s %-10s\n",
           "M", "N", "K", "old_us", "old_GBps", "old_ok", "v2_us", "v2_GBps", "v2_ok",
           "old_vs_v2", "v2_orcl", "old_orcl", "");
    bool any_fail = false;
    for (const auto& nk : nks) {
        for (int M : Ms) {
            ShapeStat st;
            const int rc = run_shape(M, nk.N, nk.K, iters, &st);
            if (rc != 0) { any_fail = true; continue; }
            printf("%-6d %-8d %-8d | %-8.1f %-8.1f %-9s | %-8.1f %-8.1f %-9s | %-10.4g %-10.4g | %-10.4g\n",
                   st.M, st.N, st.K,
                   st.old_us, st.old_gbps, st.old_ok ? "yes" : "NO",
                   st.v2_us, st.v2_gbps, st.v2_ok ? "yes" : "NO",
                   st.max_rel_diff_old_vs_v2, st.max_rel_diff_v2_vs_oracle, st.max_rel_diff_old_vs_oracle);
            if (st.v2_ok && st.max_rel_diff_v2_vs_oracle > 0.05) {
                fprintf(stderr, "  *** v2 vs oracle relative error high at M=%d N=%d K=%d: %.4g\n",
                        st.M, nk.N, nk.K, st.max_rel_diff_v2_vs_oracle);
                any_fail = true;
            }
            if (st.old_ok && st.v2_ok && st.max_rel_diff_old_vs_v2 > 0.05) {
                fprintf(stderr, "  *** old vs v2 relative diff high at M=%d N=%d K=%d: %.4g\n",
                        st.M, nk.N, nk.K, st.max_rel_diff_old_vs_v2);
                any_fail = true;
            }
        }
    }
    printf("%s\n", any_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return any_fail ? 1 : 0;
}
