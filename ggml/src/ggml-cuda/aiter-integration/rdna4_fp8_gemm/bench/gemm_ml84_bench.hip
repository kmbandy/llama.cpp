// bench/gemm_ml84_bench.hip — MAD-305 ML8_4 (4.5 bpw) decode/prefill bench.
//
// Exercises the three gemm_ml84_prod.hip entry points (gemm_capi.h):
//   - rdna4_pack_ml84_trfeed          (host-side ml84_trfeed_pack_host is
//     used to build the ground-truth packed weight directly, matching how
//     gemm_trfeed_prod_bench.hip preshuffles B on-device rather than
//     round-tripping through the packer for its own oracle setup; the
//     device packer path is separately exercised in ONE shape below to
//     confirm it agrees byte-for-byte with the host packer.)
//   - rdna4_gemm_ml84_trfeed_decode   (M<=32 native decode kernel)
//   - rdna4_expand_ml84_to_trfeed + the UNCHANGED rdna4_gemm_fp8_trfeed
//     (prefill path: expand once, run the frozen kernel)
//
// Oracle: dequantize_row_ml8_4_with_lut-style fp32 dequant (w = centroid *
// scale, centroid = e4m3_decode(lut[g*16+idx])) then fp32 A@B on a SAMPLE of
// output entries (full M*N*K is intractable at the M=2048 shapes -- same
// sampling discipline as gemm_trfeed_prod_bench.hip's `samples`).
//
// Tolerances: 0.03 relative error for the native decode path (same as the
// existing fp8 trfeed bench -- fp8 activation rounding dominates); looser
// for the expander+frozen-prefill path since e4m3-requantizing an already
// 4-bit-quantized weight adds a second rounding step -- we PRINT the actual
// measured error rather than asserting a specific bound, per the task brief
// ("we want to know it").
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "ml84_trfeed_layout.h"
#include "trfeed_ml84_kernels.h"
#include "../gemm_capi.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>

#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

static int round_up(int x, int m) { return ((x + m - 1) / m) * m; }

// ─────────────────────────────────────────────────────────────────────────
// ml84_trfeed_nk_to_pos / ml84_trfeed_pos_to_nk round-trip self-check, on a
// small shape (independent of the GPU -- pure host arithmetic). Written as
// the task brief requests: loop every position, assert the round trip.
// ─────────────────────────────────────────────────────────────────────────
static bool check_pos_roundtrip() {
    const int N = 64, K = 128;   // 4 N-tiles x 8 K-tiles worth of positions
    bool ok = true;
    for (int n = 0; n < N && ok; ++n) {
        for (int k = 0; k < K && ok; ++k) {
            const size_t pos = ml84_trfeed_nk_to_pos(n, k, N, K);
            int n2, k2;
            ml84_trfeed_pos_to_nk(pos, N, K, &n2, &k2);
            if (n2 != n || k2 != k) {
                fprintf(stderr, "  pos roundtrip FAIL: (n=%d,k=%d) -> pos=%zu -> (n=%d,k=%d)\n",
                        n, k, pos, n2, k2);
                ok = false;
            }
        }
    }
    // also confirm pos spans exactly [0, N*K) bijectively (no collisions):
    // every (n,k) pair maps to a distinct pos in [0,N*K).
    std::vector<uint8_t> seen((size_t) N * K, 0);
    for (int n = 0; n < N && ok; ++n) {
        for (int k = 0; k < K && ok; ++k) {
            const size_t pos = ml84_trfeed_nk_to_pos(n, k, N, K);
            if (pos >= (size_t) N * K || seen[pos]) { ok = false; break; }
            seen[pos] = 1;
        }
    }
    printf("ml84_trfeed_pos_to_nk/nk_to_pos round-trip + bijection check (N=%d,K=%d): %s\n",
           N, K, ok ? "PASS" : "*** FAIL ***");
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────
// Expander self-check (coordinator-requested diagnostic): dequantize a
// sample of (n,k) positions DIRECTLY from the just-computed B_shuf_out +
// b_scale_out -- using the EXACT convention the frozen kernel's epilogue
// applies (byte * b_scale_out[n]) -- and compare against the SAME dequant
// used for the main oracle (`dequant_w`, reading B_nib/lut/b_scale_g
// straight). This is independent of rdna4_gemm_fp8_trfeed entirely, so it
// pins a residual error to the EXPANDER's own output vs. how the frozen GEMM
// consumes it. Tolerance is not asserted here (only ~6% e4m3 requantization
// noise is expected); the caller prints/uses the returned max relative diff.
// ─────────────────────────────────────────────────────────────────────────
static bool check_expand_values(const uint8_t* d_B_shuf, const float* d_b_scale_out, int N, int K,
                                const std::function<float(int, int)>& dequant_w, float* out_max_rel) {
    const int NT = N / 16;
    std::vector<float> b_scale_out_h(N);
    if (hipMemcpy(b_scale_out_h.data(), d_b_scale_out, (size_t) N * sizeof(float), hipMemcpyDeviceToHost) != hipSuccess) {
        return false;
    }
    float mx = 0.f;
    const int SN = std::min(N, 16), SK = std::min(K, 16);
    for (int si = 0; si < SN; si++) {
        const int n = (int) ((size_t) si * N / SN);
        for (int sj = 0; sj < SK; sj++) {
            const int k = (int) ((size_t) sj * K / SK);
            const int kt = k / 16, nt = n / 16, kl = k % 16, nl = n % 16;
            int L, s;
            ml84_lane_slot(kl, nl, &L, &s);
            const size_t addr = b_tile_offset(kt, nt, NT) + (size_t) trperm(L, s);
            uint8_t byte;
            if (hipMemcpy(&byte, d_B_shuf + addr, 1, hipMemcpyDeviceToHost) != hipSuccess) {
                return false;
            }
            const float gv = ml84_decode_e4m3(byte) * b_scale_out_h[n];
            const float ref = dequant_w(n, k);
            const float e = fabsf(gv - ref) / (fabsf(ref) + 1e-3f);
            if (e > mx) mx = e;
        }
    }
    *out_max_rel = mx;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// fp8 frozen-kernel reference (task item 4): times rdna4_gemm_fp8_trfeed
// (and, for the decode tile, rdna4_gemm_fp8_trfeed_splitk at its own
// default split count) on a RANDOM fp8 B_shuf of the SAME (N,K) -- not the
// ML8_4 weight at all, just a same-shape timing reference so every table
// row shows "ML8_4 vs fp8" side by side. Correctness is not re-checked here
// (the frozen kernel is separately, extensively validated by
// gemm_trfeed_prod_bench.hip); this only measures ms.
// ─────────────────────────────────────────────────────────────────────────
static bool bench_fp8_frozen(int M, int N, int K, int iters, double* out_ms) {
    if (K % 32 != 0 || N % 128 != 0) return false;   // frozen kernel's own BK/BN divisibility
    const int M_pad = (M <= 32) ? 32 : round_up(M, 128);

    std::vector<uint8_t> Bh((size_t) K * N);
    std::vector<float> bs(N);
    std::mt19937 rng(0xF8F8F8F8u ^ (uint32_t) (M * 1000003 + N * 7919 + K));
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
    for (auto& b : Bh) b = ml84_encode_e4m3(a_dist(rng));
    for (auto& s : bs) s = 0.4f + 0.03f * fabsf(a_dist(rng));

    uint8_t *A_pad = nullptr, *B_kn = nullptr, *B_shuf = nullptr;
    float *as_pad = nullptr, *bscale = nullptr;
    __hip_bfloat16* C_bf16 = nullptr;
    if (hipMalloc(&A_pad, (size_t) M_pad * K) != hipSuccess) return false;
    if (hipMalloc(&B_kn, (size_t) K * N) != hipSuccess) return false;
    if (hipMalloc(&B_shuf, (size_t) K * N) != hipSuccess) return false;
    if (hipMalloc(&as_pad, (size_t) M_pad * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&bscale, (size_t) N * sizeof(float)) != hipSuccess) return false;
    if (hipMalloc(&C_bf16, (size_t) M_pad * N * 2) != hipSuccess) return false;
    hipMemset(A_pad, 0, (size_t) M_pad * K);
    hipMemset(as_pad, 0, (size_t) M_pad * sizeof(float));
    std::vector<uint8_t> Ah((size_t) M * K);
    std::vector<float> asc(M, 0.5f);
    for (auto& a : Ah) a = ml84_encode_e4m3(a_dist(rng));
    hipMemcpy(A_pad, Ah.data(), (size_t) M * K, hipMemcpyHostToDevice);
    hipMemcpy(as_pad, asc.data(), (size_t) M * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(B_kn, Bh.data(), (size_t) K * N, hipMemcpyHostToDevice);
    hipMemcpy(bscale, bs.data(), (size_t) N * sizeof(float), hipMemcpyHostToDevice);

    hipStream_t stream = 0;
    bool ok = rdna4_preshuffle_b_ml8fp8(B_kn, B_shuf, K, N, stream) == hipSuccess;
    if (ok) ok = hipStreamSynchronize(stream) == hipSuccess;

    double best_ms = 1e18;
    if (ok) {
        for (int i = 0; i < 5; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, bscale, M_pad, N, K, stream);
        hipDeviceSynchronize();
        hipEvent_t s0, e0; hipEventCreate(&s0); hipEventCreate(&e0);
        hipEventRecord(s0);
        for (int i = 0; i < iters; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, bscale, M_pad, N, K, stream);
        hipEventRecord(e0); hipEventSynchronize(e0);
        float ms = 0; hipEventElapsedTime(&ms, s0, e0); ms /= iters;
        hipEventDestroy(s0); hipEventDestroy(e0);
        best_ms = ms;

        if (M_pad == 32) {   // also try the frozen kernel's own split-K at its default split count
            const int n_splits = rdna4_trfeed_splitk_default_splits(N, K);
            float* C_f32 = nullptr;
            if (hipMalloc(&C_f32, (size_t) M_pad * N * sizeof(float)) == hipSuccess) {
                for (int i = 0; i < 5; i++)
                    rdna4_gemm_fp8_trfeed_splitk(A_pad, B_shuf, C_f32, as_pad, bscale, M_pad, N, K, n_splits, stream);
                hipDeviceSynchronize();
                hipEvent_t s1, e1; hipEventCreate(&s1); hipEventCreate(&e1);
                hipEventRecord(s1);
                for (int i = 0; i < iters; i++)
                    rdna4_gemm_fp8_trfeed_splitk(A_pad, B_shuf, C_f32, as_pad, bscale, M_pad, N, K, n_splits, stream);
                hipEventRecord(e1); hipEventSynchronize(e1);
                float ms2 = 0; hipEventElapsedTime(&ms2, s1, e1); ms2 /= iters;
                hipEventDestroy(s1); hipEventDestroy(e1);
                if (ms2 < best_ms) best_ms = ms2;
                hipFree(C_f32);
            }
        }
    }

    hipFree(A_pad); hipFree(B_kn); hipFree(B_shuf); hipFree(as_pad); hipFree(bscale); hipFree(C_bf16);
    if (!ok) return false;
    *out_ms = best_ms;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Shape run.
// ─────────────────────────────────────────────────────────────────────────
struct ShapeResult {
    int M, N, K;
    bool decode_pass = false, native_ok = false;
    double decode_ms = 0, decode_gbps = 0, decode_err = 0;
    bool did_expand_compare = false;
    double expand_native_expand_ms = 0, expand_native_gemm_ms = 0, expand_native_err = 0;
    bool did_prefill = false;
    double expand_ms = 0, gemm_ms = 0, tf = 0, prefill_err = 0;
    bool did_expand_selfcheck = false;
    float expand_selfcheck_err = 0;
    // fp8 frozen-kernel reference column (task 4): same shape, random fp8 B,
    // for side-by-side comparison against the ML8_4 numbers above.
    bool did_fp8_ref = false;
    double fp8_ref_ms = 0, fp8_ref_tf_or_gbps = 0;
};

static int run_shape(int M, int N, int K, int iters, bool run_device_packer_check, bool bench_splitk,
                     ShapeResult* out) {
    out->M = M; out->N = N; out->K = K;
    if (K % QK_ML8 != 0 || N % 16 != 0) {
        fprintf(stderr, "shape M=%d N=%d K=%d skipped: K%%64 or N%%16 violated\n", M, N, K);
        return 3;
    }
    const int n_groups_k = K / QK_ML8;
    std::mt19937 rng(0x4D4C3834u ^ (uint32_t) (M * 1000003 + N * 7919 + K));
    std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);
    std::uniform_int_distribution<int> nib_dist(0, 15);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);

    // ---- ground-truth ml8_4 weight blocks: w[N][n_groups_k], on-disk layout ----
    std::vector<block_ml8_4> w((size_t) N * n_groups_k);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < n_groups_k; ++g) {
            block_ml8_4& blk = w[(size_t) n * n_groups_k + g];
            blk.scale = scale_dist(rng);
            for (int i = 0; i < QK_ML8 / 2; ++i) {
                blk.qs[i] = (uint8_t) (nib_dist(rng) | (nib_dist(rng) << 4));
            }
        }
    }
    // ---- per-group LUT: 16 sorted signed e4m3 values in [-1,1] (Lloyd-Max-like) ----
    std::vector<uint8_t> lut((size_t) n_groups_k * 16);
    for (int g = 0; g < n_groups_k; ++g) {
        std::vector<float> centroids(16);
        for (int i = 0; i < 16; ++i) centroids[i] = -1.0f + 2.0f * (float) i / 15.0f;
        // small per-group jitter, then re-sort so it stays a valid ordered LUT
        for (int i = 0; i < 16; ++i) centroids[i] += 0.02f * a_dist(rng);
        std::sort(centroids.begin(), centroids.end());
        for (int i = 0; i < 16; ++i) {
            float c = std::max(-1.0f, std::min(1.0f, centroids[i]));
            lut[(size_t) g * 16 + i] = ml84_encode_e4m3(c);
        }
    }

    // ---- CPU dequant of the full weight (for the oracle sample loop) ----
    auto dequant_w = [&](int n, int k) -> float {
        const int g = k / QK_ML8;
        const block_ml8_4& blk = w[(size_t) n * n_groups_k + g];
        const int kk = k % QK_ML8;
        const uint8_t packed = blk.qs[kk / 2];
        const uint8_t idx = (kk & 1) ? ((packed >> 4) & 0x0Fu) : (packed & 0x0Fu);
        return ml84_decode_e4m3(lut[(size_t) g * 16 + idx]) * blk.scale;
    };

    // ---- random fp32 A rows + per-row a_scale = amax/448 (matches ml8.cu's
    // documented convention), then e4m3-encode ----
    std::vector<float> Af((size_t) M * K);
    std::vector<float> a_scale(M);
    std::vector<uint8_t> Ah((size_t) M * K);
    for (int m = 0; m < M; ++m) {
        float amax = 1e-6f;
        for (int k = 0; k < K; ++k) { Af[(size_t) m * K + k] = a_dist(rng); amax = std::max(amax, std::fabs(Af[(size_t) m * K + k])); }
        a_scale[m] = amax / 448.0f;
        for (int k = 0; k < K; ++k) Ah[(size_t) m * K + k] = ml84_encode_e4m3(Af[(size_t) m * K + k] / a_scale[m]);
    }

    // ---- oracle samples: dequantized-A x dequantized-W, sampled (i,j) ----
    struct Sample { int i, j; float ref; };
    std::vector<Sample> samples;
    const int SI = std::min(M, 32), SJ = std::min(N, 32);
    samples.reserve((size_t) SI * SJ);
    for (int si = 0; si < SI; si++) {
        const int i = (int) ((size_t) si * M / SI);
        for (int sj = 0; sj < SJ; sj++) {
            const int j = (int) ((size_t) sj * N / SJ);
            float acc = 0.f;
            // use the fp8-ROUNDED activation (matches what the kernel actually
            // consumes) times the dequantized ml8_4 weight, scaled by a_scale[i].
            for (int k = 0; k < K; k++) {
                acc += ml84_decode_e4m3(Ah[(size_t) i * K + k]) * dequant_w(j, k);
            }
            samples.push_back({i, j, acc * a_scale[i]});
        }
    }

    // ---- pack B_nib + b_scale_g (host reference packer) ----
    std::vector<uint8_t> B_nib_h((size_t) N * K / 2);
    std::vector<float> b_scale_g_h((size_t) n_groups_k * N);
    ml84_trfeed_pack_host(w.data(), N, K, B_nib_h.data(), b_scale_g_h.data());

    // ---- device buffers ----
    uint8_t *d_w = nullptr, *d_lut = nullptr, *d_B_nib = nullptr;
    float *d_b_scale_g = nullptr;
    HCK(hipMalloc(&d_w, w.size() * sizeof(block_ml8_4)));
    HCK(hipMalloc(&d_lut, lut.size()));
    HCK(hipMalloc(&d_B_nib, B_nib_h.size()));
    HCK(hipMalloc(&d_b_scale_g, b_scale_g_h.size() * sizeof(float)));
    HCK(hipMemcpy(d_w, w.data(), w.size() * sizeof(block_ml8_4), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_lut, lut.data(), lut.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_B_nib, B_nib_h.data(), B_nib_h.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_b_scale_g, b_scale_g_h.data(), b_scale_g_h.size() * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream = 0;

    if (run_device_packer_check) {
        // Confirm the DEVICE packer agrees byte-for-byte with the host
        // reference packer (ml84_trfeed_pack_host), on this one shape.
        std::vector<uint8_t> B_nib_dev(B_nib_h.size());
        std::vector<float> b_scale_g_dev(b_scale_g_h.size());
        uint8_t* d_B_nib2 = nullptr; float* d_b_scale_g2 = nullptr;
        HCK(hipMalloc(&d_B_nib2, B_nib_h.size()));
        HCK(hipMalloc(&d_b_scale_g2, b_scale_g_h.size() * sizeof(float)));
        HCK(rdna4_pack_ml84_trfeed(d_w, N, K, d_B_nib2, d_b_scale_g2, stream));
        HCK(hipStreamSynchronize(stream));
        HCK(hipMemcpy(B_nib_dev.data(), d_B_nib2, B_nib_h.size(), hipMemcpyDeviceToHost));
        HCK(hipMemcpy(b_scale_g_dev.data(), d_b_scale_g2, b_scale_g_h.size() * sizeof(float), hipMemcpyDeviceToHost));
        const bool nib_match = std::memcmp(B_nib_dev.data(), B_nib_h.data(), B_nib_h.size()) == 0;
        const bool scale_match = std::memcmp(b_scale_g_dev.data(), b_scale_g_h.data(), b_scale_g_h.size() * sizeof(float)) == 0;
        printf("  device packer vs host packer (M=%d N=%d K=%d): B_nib %s, b_scale_g %s\n",
               M, N, K, nib_match ? "MATCH" : "*** MISMATCH ***", scale_match ? "MATCH" : "*** MISMATCH ***");
        hipFree(d_B_nib2); hipFree(d_b_scale_g2);
    }


    // =========================================================================
    // (a) native decode kernel, M<=32 only (M_pad must be exactly 32)
    // =========================================================================
    if (M <= 32) {
        const int M_pad = 32;
        uint8_t* A_pad = nullptr;
        float* as_pad = nullptr;
        float* C_f32 = nullptr;
        HCK(hipMalloc(&A_pad, (size_t) M_pad * K));
        HCK(hipMalloc(&as_pad, (size_t) M_pad * sizeof(float)));
        HCK(hipMalloc(&C_f32, (size_t) M_pad * N * sizeof(float)));
        HCK(hipMemset(A_pad, 0, (size_t) M_pad * K));
        HCK(hipMemset(as_pad, 0, (size_t) M_pad * sizeof(float)));
        HCK(hipMemcpy(A_pad, Ah.data(), (size_t) M * K, hipMemcpyHostToDevice));
        HCK(hipMemcpy(as_pad, a_scale.data(), (size_t) M * sizeof(float), hipMemcpyHostToDevice));

        hipError_t rc = rdna4_gemm_ml84_trfeed_decode(A_pad, d_B_nib, d_lut, C_f32, as_pad, d_b_scale_g,
                                                       M_pad, N, K, stream);
        if (rc != hipSuccess) {
            fprintf(stderr, "  decode M=%d N=%d K=%d: dispatch failed: %s\n", M, N, K, hipGetErrorString(rc));
        } else {
            HCK(hipDeviceSynchronize());
            float mx = 0.f;
            // per-row max error, printed for M=32 (task diagnostic #5): if
            // rows 16-31 (the second M-fragment, mi=1 in the <32,1> tile's
            // TFRAGS_M=2) are systematically worse than rows 0-15, that
            // points at the a_scale/row mapping of the second fragment in
            // the epilogue or the scale fold; if uniform, it's just fp8
            // activation rounding of the random A (expected, same as the
            // frozen fp8 bench's own tolerance).
            std::vector<float> row_max(M, 0.f);
            for (const auto& samp : samples) {
                float gv;
                HCK(hipMemcpy(&gv, C_f32 + (size_t) samp.i * N + samp.j, 4, hipMemcpyDeviceToHost));
                float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
                if (e > mx) mx = e;
                if (e > row_max[samp.i]) row_max[samp.i] = e;
            }
            out->decode_err = mx;
            out->native_ok = mx < 0.03f;
            if (M == 32) {
                printf("  decode per-row max_rel_err (M=32,N=%d,K=%d):", N, K);
                for (int r = 0; r < M; r++) printf(" %.4f", row_max[r]);
                printf("\n");
            }

            for (int i = 0; i < 5; i++) rdna4_gemm_ml84_trfeed_decode(A_pad, d_B_nib, d_lut, C_f32, as_pad, d_b_scale_g, M_pad, N, K, stream);
            HCK(hipDeviceSynchronize());
            hipEvent_t s, e2; hipEventCreate(&s); hipEventCreate(&e2);
            hipEventRecord(s);
            for (int i = 0; i < iters; i++) rdna4_gemm_ml84_trfeed_decode(A_pad, d_B_nib, d_lut, C_f32, as_pad, d_b_scale_g, M_pad, N, K, stream);
            hipEventRecord(e2); hipEventSynchronize(e2);
            float ms = 0; hipEventElapsedTime(&ms, s, e2); ms /= iters;
            hipEventDestroy(s); hipEventDestroy(e2);
            out->decode_ms = ms;
            const double packed_bytes = (double) N * K / 2.0 + (double) n_groups_k * N * sizeof(float);
            out->decode_gbps = (packed_bytes / 1e9) / (ms / 1e3);
            out->decode_pass = true;

            // ---- split-K decode table (task item 4), for the caller-flagged
            // workgroup-starved shapes only ----
            if (bench_splitk) {
                float* C_f32_sk = nullptr;
                HCK(hipMalloc(&C_f32_sk, (size_t) M_pad * N * sizeof(float)));
                const int default_splits = rdna4_ml84_trfeed_splitk_default_splits(N, K);
                static const int try_splits[] = {1, 2, 4, 8};
                printf("  splitk decode (M=%d N=%d K=%d), default_splits=%d:\n", M, N, K, default_splits);
                for (int trial = 0; trial < (int) (sizeof(try_splits) / sizeof(try_splits[0])); ++trial) {
                    const int n_splits = try_splits[trial];
                    hipError_t rcs = rdna4_gemm_ml84_trfeed_decode_splitk(
                        A_pad, d_B_nib, d_lut, C_f32_sk, as_pad, d_b_scale_g, M_pad, N, K, n_splits, stream);
                    if (rcs != hipSuccess) {
                        printf("    n_splits=%-3d dispatch failed: %s\n", n_splits, hipGetErrorString(rcs));
                        continue;
                    }
                    HCK(hipDeviceSynchronize());
                    float mxs = 0.f;
                    for (const auto& samp : samples) {
                        float gv;
                        HCK(hipMemcpy(&gv, C_f32_sk + (size_t) samp.i * N + samp.j, 4, hipMemcpyDeviceToHost));
                        float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
                        if (e > mxs) mxs = e;
                    }
                    for (int i = 0; i < 5; i++)
                        rdna4_gemm_ml84_trfeed_decode_splitk(A_pad, d_B_nib, d_lut, C_f32_sk, as_pad, d_b_scale_g, M_pad, N, K, n_splits, stream);
                    HCK(hipDeviceSynchronize());
                    hipEvent_t ss, se; hipEventCreate(&ss); hipEventCreate(&se);
                    hipEventRecord(ss);
                    for (int i = 0; i < iters; i++)
                        rdna4_gemm_ml84_trfeed_decode_splitk(A_pad, d_B_nib, d_lut, C_f32_sk, as_pad, d_b_scale_g, M_pad, N, K, n_splits, stream);
                    hipEventRecord(se); hipEventSynchronize(se);
                    float sms = 0; hipEventElapsedTime(&sms, ss, se); sms /= iters;
                    hipEventDestroy(ss); hipEventDestroy(se);
                    const double gbps = (packed_bytes / 1e9) / (sms / 1e3);
                    printf("    n_splits=%-3d%s max_rel_err=%.4f  %8.4f ms  %7.1f GB/s\n",
                           n_splits, n_splits == default_splits ? " (default)" : "          ",
                           mxs, sms, gbps);
                }
                hipFree(C_f32_sk);
            }
        }

        // ---- comparison: expander + frozen <32,1> for this same M<=32 shape ----
        uint8_t* B_shuf = nullptr; float* b_scale_out = nullptr; __hip_bfloat16* C_bf16 = nullptr;
        HCK(hipMalloc(&B_shuf, (size_t) K * N));
        HCK(hipMalloc(&b_scale_out, (size_t) N * sizeof(float)));
        HCK(hipMalloc(&C_bf16, (size_t) M_pad * N * 2));
        hipError_t rc2 = rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
        if (rc2 == hipSuccess) {
            rc2 = hipStreamSynchronize(stream);
        }
        if (rc2 == hipSuccess) {
            // self-check (task item 1): compare the expander's OWN output
            // against the direct B_nib/lut/b_scale_g dequant, independent of
            // rdna4_gemm_fp8_trfeed entirely.
            float sc_err = 0.f;
            if (check_expand_values(B_shuf, b_scale_out, N, K, dequant_w, &sc_err)) {
                out->did_expand_selfcheck = true;
                out->expand_selfcheck_err = sc_err;
                printf("  expander self-check (M=%d N=%d K=%d): max_rel_diff=%.4f (vs B_nib/lut/b_scale_g direct)\n",
                       M, N, K, sc_err);
            }
        }
        if (rc2 == hipSuccess) {
            rc2 = rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
        }
        if (rc2 != hipSuccess) {
            fprintf(stderr, "  expand+frozen<32,1> M=%d N=%d K=%d: dispatch failed: %s\n", M, N, K, hipGetErrorString(rc2));
        } else {
            HCK(hipDeviceSynchronize());
            float mx = 0.f;
            int printed = 0;
            for (const auto& samp : samples) {
                uint16_t hi;
                HCK(hipMemcpy(&hi, (const uint16_t*) C_bf16 + (size_t) samp.i * N + samp.j, 2, hipMemcpyDeviceToHost));
                uint32_t bits = (uint32_t) hi << 16; float gv; std::memcpy(&gv, &bits, 4);
                float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
                if (e > mx) mx = e;
                // Diagnostic (task item 2): the coordinator's round-2 report
                // found expNat/prefill errors BIT-IDENTICAL across a full
                // expander rewrite -- i.e. this comparison itself looked
                // broken independent of the expander. Print concrete
                // (i,j,ref,gv) so a stale-buffer/wrong-arg bug (if still
                // present) is visible directly instead of only as a max_err
                // number.
                if (printed < 3) {
                    printf("    expNat sample i=%d j=%d ref=%.6f gv=%.6f rel_err=%.4f\n",
                           samp.i, samp.j, samp.ref, gv, e);
                    printed++;
                }
            }
            out->expand_native_err = mx;

            // ---- timing: expand-alone and GEMM-alone, isolated (task item 2) ----
            for (int i = 0; i < 5; i++) rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
            HCK(hipDeviceSynchronize());
            hipEvent_t se0, se1; hipEventCreate(&se0); hipEventCreate(&se1);
            hipEventRecord(se0);
            for (int i = 0; i < iters; i++) rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
            hipEventRecord(se1); hipEventSynchronize(se1);
            float ems = 0; hipEventElapsedTime(&ems, se0, se1); ems /= iters;
            hipEventDestroy(se0); hipEventDestroy(se1);

            for (int i = 0; i < 5; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
            HCK(hipDeviceSynchronize());
            hipEvent_t sg0, sg1; hipEventCreate(&sg0); hipEventCreate(&sg1);
            hipEventRecord(sg0);
            for (int i = 0; i < iters; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
            hipEventRecord(sg1); hipEventSynchronize(sg1);
            float gms = 0; hipEventElapsedTime(&gms, sg0, sg1); gms /= iters;
            hipEventDestroy(sg0); hipEventDestroy(sg1);

            out->expand_native_expand_ms = ems;
            out->expand_native_gemm_ms = gms;
            out->did_expand_compare = true;
        }
        hipFree(B_shuf); hipFree(b_scale_out); hipFree(C_bf16);
        hipFree(A_pad); hipFree(as_pad); hipFree(C_f32);
    } else {
        // =====================================================================
        // (b) prefill: expander + UNCHANGED frozen gemm_fp8_trfeed<128,2>
        // =====================================================================
        const int M_pad = round_up(M, 128);
        uint8_t* A_pad = nullptr; float* as_pad = nullptr;
        uint8_t* B_shuf = nullptr; float* b_scale_out = nullptr; __hip_bfloat16* C_bf16 = nullptr;
        HCK(hipMalloc(&A_pad, (size_t) M_pad * K));
        HCK(hipMalloc(&as_pad, (size_t) M_pad * sizeof(float)));
        HCK(hipMalloc(&B_shuf, (size_t) K * N));
        HCK(hipMalloc(&b_scale_out, (size_t) N * sizeof(float)));
        HCK(hipMalloc(&C_bf16, (size_t) M_pad * N * 2));
        HCK(hipMemset(A_pad, 0, (size_t) M_pad * K));
        HCK(hipMemset(as_pad, 0, (size_t) M_pad * sizeof(float)));
        HCK(hipMemcpy(A_pad, Ah.data(), (size_t) M * K, hipMemcpyHostToDevice));
        HCK(hipMemcpy(as_pad, a_scale.data(), (size_t) M * sizeof(float), hipMemcpyHostToDevice));

        hipError_t rc = rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
        if (rc == hipSuccess) rc = hipStreamSynchronize(stream);
        if (rc != hipSuccess) {
            fprintf(stderr, "  prefill M=%d N=%d K=%d: expand failed: %s\n", M, N, K, hipGetErrorString(rc));
        } else {
            // self-check (task item 1), same as the M<=32 expNat path above.
            float sc_err = 0.f;
            if (check_expand_values(B_shuf, b_scale_out, N, K, dequant_w, &sc_err)) {
                out->did_expand_selfcheck = true;
                out->expand_selfcheck_err = sc_err;
                printf("  expander self-check (M=%d N=%d K=%d): max_rel_diff=%.4f (vs B_nib/lut/b_scale_g direct)\n",
                       M, N, K, sc_err);
            }
            rc = rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
        }
        if (rc != hipSuccess) {
            fprintf(stderr, "  prefill M=%d N=%d K=%d: gemm failed: %s\n", M, N, K, hipGetErrorString(rc));
        } else {
            HCK(hipDeviceSynchronize());
            float mx = 0.f;
            int printed = 0;
            for (const auto& samp : samples) {
                uint16_t hi;
                HCK(hipMemcpy(&hi, (const uint16_t*) C_bf16 + (size_t) samp.i * N + samp.j, 2, hipMemcpyDeviceToHost));
                uint32_t bits = (uint32_t) hi << 16; float gv; std::memcpy(&gv, &bits, 4);
                float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
                if (e > mx) mx = e;
                if (printed < 3) {   // diagnostic, see the expNat block's comment above
                    printf("    prefill sample i=%d j=%d ref=%.6f gv=%.6f rel_err=%.4f\n",
                           samp.i, samp.j, samp.ref, gv, e);
                    printed++;
                }
            }
            out->prefill_err = mx;

            hipEvent_t s0, e0, s1, e1;
            hipEventCreate(&s0); hipEventCreate(&e0); hipEventCreate(&s1); hipEventCreate(&e1);
            for (int i = 0; i < 5; i++) rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
            HCK(hipDeviceSynchronize());
            hipEventRecord(s0);
            for (int i = 0; i < iters; i++) rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
            hipEventRecord(e0); hipEventSynchronize(e0);
            float ems = 0; hipEventElapsedTime(&ems, s0, e0); ems /= iters;

            for (int i = 0; i < 5; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
            HCK(hipDeviceSynchronize());
            hipEventRecord(s1);
            for (int i = 0; i < iters; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
            hipEventRecord(e1); hipEventSynchronize(e1);
            float gms = 0; hipEventElapsedTime(&gms, s1, e1); gms /= iters;
            hipEventDestroy(s0); hipEventDestroy(e0); hipEventDestroy(s1); hipEventDestroy(e1);

            out->expand_ms = ems;
            out->gemm_ms = gms;
            const double gflop = 2.0 * (double) M * N * K / 1e12;
            out->tf = gflop / ((ems + gms) / 1e3);
            out->did_prefill = true;
        }
        hipFree(A_pad); hipFree(as_pad); hipFree(B_shuf); hipFree(b_scale_out); hipFree(C_bf16);
    }

    hipFree(d_w); hipFree(d_lut); hipFree(d_B_nib); hipFree(d_b_scale_g);

    // ---- fp8 frozen-kernel reference column (task item 4) ----
    double fp8_ms = 0;
    if (bench_fp8_frozen(M, N, K, iters, &fp8_ms)) {
        out->did_fp8_ref = true;
        out->fp8_ref_ms = fp8_ms;
        if (M <= 32) {
            const double packed_bytes = (double) N * K + (double) N * sizeof(float);   // fp8 B is N*K bytes + b_scale
            out->fp8_ref_tf_or_gbps = (packed_bytes / 1e9) / (fp8_ms / 1e3);
        } else {
            const double gflop = 2.0 * (double) M * N * K / 1e12;
            out->fp8_ref_tf_or_gbps = gflop / (fp8_ms / 1e3);
        }
    }
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────
// Task item 3 EXPERIMENT: rdna4_gemm_ml84_trfeed_prefill (in-kernel LUT
// expand at the frozen prefill tile geometry) vs expander+frozen
// (rdna4_expand_ml84_to_trfeed + rdna4_gemm_fp8_trfeed), at M in
// {128,512,2048}, for one (N,K) shape. Hypothesis: in-kernel LUT expansion
// is VALU work sharing the issue port with WMMA, so it should lose at
// M=2048 (expander's fixed cost amortized away) but may win at M=128-512
// (fixed cost dominates) -- measured here, not asserted.
// ─────────────────────────────────────────────────────────────────────────
static void bench_prefill_experiment(int N, int K, int iters) {
    printf("\n-- prefill experiment (N=%d K=%d): native in-kernel-LUT-expand vs expander+frozen --\n", N, K);
    const int n_groups_k = K / QK_ML8;
    std::mt19937 rng(0x50524546u ^ (uint32_t) (N * 7919 + K));
    std::uniform_real_distribution<float> scale_dist(0.5f, 1.5f);
    std::uniform_int_distribution<int> nib_dist(0, 15);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);

    std::vector<block_ml8_4> w((size_t) N * n_groups_k);
    for (int n = 0; n < N; ++n)
        for (int g = 0; g < n_groups_k; ++g) {
            block_ml8_4& blk = w[(size_t) n * n_groups_k + g];
            blk.scale = scale_dist(rng);
            for (int i = 0; i < QK_ML8 / 2; ++i) blk.qs[i] = (uint8_t) (nib_dist(rng) | (nib_dist(rng) << 4));
        }
    std::vector<uint8_t> lut((size_t) n_groups_k * 16);
    for (int g = 0; g < n_groups_k; ++g) {
        std::vector<float> c(16);
        for (int i = 0; i < 16; ++i) c[i] = -1.0f + 2.0f * (float) i / 15.0f + 0.02f * a_dist(rng);
        std::sort(c.begin(), c.end());
        for (int i = 0; i < 16; ++i) lut[(size_t) g * 16 + i] = ml84_encode_e4m3(std::max(-1.0f, std::min(1.0f, c[i])));
    }
    auto dequant_w = [&](int n, int k) -> float {
        const int g = k / QK_ML8;
        const block_ml8_4& blk = w[(size_t) n * n_groups_k + g];
        const int kk = k % QK_ML8;
        const uint8_t packed = blk.qs[kk / 2];
        const uint8_t idx = (kk & 1) ? ((packed >> 4) & 0x0Fu) : (packed & 0x0Fu);
        return ml84_decode_e4m3(lut[(size_t) g * 16 + idx]) * blk.scale;
    };

    std::vector<uint8_t> B_nib_h((size_t) N * K / 2);
    std::vector<float> b_scale_g_h((size_t) n_groups_k * N);
    ml84_trfeed_pack_host(w.data(), N, K, B_nib_h.data(), b_scale_g_h.data());

    uint8_t *d_lut = nullptr, *d_B_nib = nullptr;
    float* d_b_scale_g = nullptr;
    hipMalloc(&d_lut, lut.size());
    hipMalloc(&d_B_nib, B_nib_h.size());
    hipMalloc(&d_b_scale_g, b_scale_g_h.size() * sizeof(float));
    hipMemcpy(d_lut, lut.data(), lut.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_B_nib, B_nib_h.data(), B_nib_h.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_b_scale_g, b_scale_g_h.data(), b_scale_g_h.size() * sizeof(float), hipMemcpyHostToDevice);
    hipStream_t stream = 0;

    printf("  %-6s | %-9s %-8s %-7s | %-8s %-8s %-7s\n",
           "M", "nativeMs", "nativeTF", "natErr", "expandMs", "gemmMs", "expTF");
    static const int Ms[] = {128, 512, 2048};
    for (int M : Ms) {
        const int M_pad = round_up(M, 128);
        std::vector<float> Af((size_t) M * K), a_scale(M);
        std::vector<uint8_t> Ah((size_t) M * K);
        for (int m = 0; m < M; ++m) {
            float amax = 1e-6f;
            for (int k = 0; k < K; ++k) { Af[(size_t) m * K + k] = a_dist(rng); amax = std::max(amax, std::fabs(Af[(size_t) m * K + k])); }
            a_scale[m] = amax / 448.0f;
            for (int k = 0; k < K; ++k) Ah[(size_t) m * K + k] = ml84_encode_e4m3(Af[(size_t) m * K + k] / a_scale[m]);
        }
        struct Sample { int i, j; float ref; };
        std::vector<Sample> samples;
        const int SI = std::min(M, 16), SJ = std::min(N, 16);
        for (int si = 0; si < SI; si++) {
            const int i = (int) ((size_t) si * M / SI);
            for (int sj = 0; sj < SJ; sj++) {
                const int j = (int) ((size_t) sj * N / SJ);
                float acc = 0.f;
                for (int k = 0; k < K; k++) acc += ml84_decode_e4m3(Ah[(size_t) i * K + k]) * dequant_w(j, k);
                samples.push_back({i, j, acc * a_scale[i]});
            }
        }

        uint8_t* A_pad = nullptr; float* as_pad = nullptr;
        hipMalloc(&A_pad, (size_t) M_pad * K);
        hipMalloc(&as_pad, (size_t) M_pad * sizeof(float));
        hipMemset(A_pad, 0, (size_t) M_pad * K);
        hipMemset(as_pad, 0, (size_t) M_pad * sizeof(float));
        hipMemcpy(A_pad, Ah.data(), (size_t) M * K, hipMemcpyHostToDevice);
        hipMemcpy(as_pad, a_scale.data(), (size_t) M * sizeof(float), hipMemcpyHostToDevice);

        char nat_ms_col[64] = "-", nat_tf_col[64] = "-", nat_err_col[64] = "-";
        char exp_ms_col[64] = "-", gemm_ms_col[64] = "-", exp_tf_col[64] = "-";

        // ---- native in-kernel LUT-expand prefill ----
        {
            float* C_f32 = nullptr;
            if (hipMalloc(&C_f32, (size_t) M_pad * N * sizeof(float)) == hipSuccess &&
                rdna4_gemm_ml84_trfeed_prefill(A_pad, d_B_nib, d_lut, C_f32, as_pad, d_b_scale_g, M_pad, N, K, stream) == hipSuccess &&
                hipDeviceSynchronize() == hipSuccess) {
                float mx = 0.f;
                for (const auto& samp : samples) {
                    float gv = 0.f;
                    hipMemcpy(&gv, C_f32 + (size_t) samp.i * N + samp.j, 4, hipMemcpyDeviceToHost);
                    float e = fabsf(gv - samp.ref) / (fabsf(samp.ref) + 1e-3f);
                    if (e > mx) mx = e;
                }
                snprintf(nat_err_col, sizeof nat_err_col, "%.4f", mx);
                for (int i = 0; i < 5; i++) rdna4_gemm_ml84_trfeed_prefill(A_pad, d_B_nib, d_lut, C_f32, as_pad, d_b_scale_g, M_pad, N, K, stream);
                hipDeviceSynchronize();
                hipEvent_t s0, e0; hipEventCreate(&s0); hipEventCreate(&e0);
                hipEventRecord(s0);
                for (int i = 0; i < iters; i++) rdna4_gemm_ml84_trfeed_prefill(A_pad, d_B_nib, d_lut, C_f32, as_pad, d_b_scale_g, M_pad, N, K, stream);
                hipEventRecord(e0); hipEventSynchronize(e0);
                float ms = 0; hipEventElapsedTime(&ms, s0, e0); ms /= iters;
                hipEventDestroy(s0); hipEventDestroy(e0);
                snprintf(nat_ms_col, sizeof nat_ms_col, "%.4f", ms);
                snprintf(nat_tf_col, sizeof nat_tf_col, "%.2f", 2.0 * (double) M * N * K / 1e12 / (ms / 1e3));
            }
            hipFree(C_f32);
        }

        // ---- expander + frozen fp8 prefill (same math as run_shape's (b) branch) ----
        {
            uint8_t* B_shuf = nullptr; float* b_scale_out = nullptr; __hip_bfloat16* C_bf16 = nullptr;
            hipMalloc(&B_shuf, (size_t) K * N);
            hipMalloc(&b_scale_out, (size_t) N * sizeof(float));
            hipMalloc(&C_bf16, (size_t) M_pad * N * 2);
            if (rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream) == hipSuccess &&
                hipStreamSynchronize(stream) == hipSuccess) {
                for (int i = 0; i < 5; i++) rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
                hipDeviceSynchronize();
                hipEvent_t s0, e0; hipEventCreate(&s0); hipEventCreate(&e0);
                hipEventRecord(s0);
                for (int i = 0; i < iters; i++) rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, B_shuf, b_scale_out, stream);
                hipEventRecord(e0); hipEventSynchronize(e0);
                float ems = 0; hipEventElapsedTime(&ems, s0, e0); ems /= iters;
                hipEventDestroy(s0); hipEventDestroy(e0);

                for (int i = 0; i < 5; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
                hipDeviceSynchronize();
                hipEvent_t s1, e1; hipEventCreate(&s1); hipEventCreate(&e1);
                hipEventRecord(s1);
                for (int i = 0; i < iters; i++) rdna4_gemm_fp8_trfeed(A_pad, B_shuf, C_bf16, as_pad, b_scale_out, M_pad, N, K, stream);
                hipEventRecord(e1); hipEventSynchronize(e1);
                float gms = 0; hipEventElapsedTime(&gms, s1, e1); gms /= iters;
                hipEventDestroy(s1); hipEventDestroy(e1);

                snprintf(exp_ms_col, sizeof exp_ms_col, "%.4f", ems);
                snprintf(gemm_ms_col, sizeof gemm_ms_col, "%.4f", gms);
                snprintf(exp_tf_col, sizeof exp_tf_col, "%.2f", 2.0 * (double) M * N * K / 1e12 / ((ems + gms) / 1e3));
            }
            hipFree(B_shuf); hipFree(b_scale_out); hipFree(C_bf16);
        }

        printf("  %-6d | %-9s %-8s %-7s | %-8s %-8s %-7s\n",
               M, nat_ms_col, nat_tf_col, nat_err_col, exp_ms_col, gemm_ms_col, exp_tf_col);

        hipFree(A_pad); hipFree(as_pad);
    }

    hipFree(d_lut); hipFree(d_B_nib); hipFree(d_b_scale_g);
}

int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("device: %s  CUs=%d\n", p.name, p.multiProcessorCount);
    printf("=== gemm_ml84_bench: ML8_4 (4.5 bpw) decode + prefill ===\n\n");

    const bool pos_ok = check_pos_roundtrip();
    printf("\n");

    struct Shape { int M, N, K; };
    static const Shape shapes[] = {
        {1,    17408, 5120},
        {2,    17408, 5120},
        {8,    17408, 5120},
        {16,   17408, 5120},
        {32,   17408, 5120},
        {8,    5120, 17408},
        {2048, 17408, 5120},
        {2048, 5120, 17408},
    };

    std::vector<ShapeResult> results;
    int failures = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); ++i) {
        const auto& sh = shapes[i];
        ShapeResult res;
        const bool packer_check = (i == 0);   // one shape is enough to validate the device packer
        // split-K decode table (task item 4) for the two shapes the
        // coordinator asked about: the "normal" N=17408 decode shape (136
        // workgroups at n_splits=1, i.e. NOT workgroup-starved -- a useful
        // contrast) and the workgroup-starved (8,5120,17408) shape (N/128=40
        // workgroups, the one measured at only 51 GB/s).
        const bool bench_splitk = (sh.M == 2 && sh.N == 17408 && sh.K == 5120) ||
                                   (sh.M == 8 && sh.N == 5120 && sh.K == 17408);
        int rc = run_shape(sh.M, sh.N, sh.K, 20, packer_check, bench_splitk, &res);
        if (rc != 0) { failures++; continue; }
        results.push_back(res);
    }

    printf("\n%-6s %-7s %-7s | %-9s %-8s %-9s | %-8s %-8s %-9s | %-8s %-8s %-7s %-9s | %-8s %-8s\n",
           "M", "N", "K", "decodeErr", "decodeMs", "decodeGBps",
           "expNatEx", "expNatGm", "expNatErr", "expandMs", "gemmMs", "TF", "prefillErr",
           "fp8RefMs", "fp8Ref*");
    for (const auto& r : results) {
        char decode_col[64] = "-", decode_ms_col[64] = "-", decode_gbps_col[64] = "-";
        char expnat_ex_col[64] = "-", expnat_gm_col[64] = "-", expnat_err_col[64] = "-";
        char expand_ms_col[64] = "-", gemm_ms_col[64] = "-", tf_col[64] = "-", prefill_err_col[64] = "-";
        char fp8_ms_col[64] = "-", fp8_metric_col[64] = "-";
        if (r.decode_pass) {
            snprintf(decode_col, sizeof decode_col, "%.4f%s", r.decode_err, r.native_ok ? "" : "*");
            snprintf(decode_ms_col, sizeof decode_ms_col, "%.4f", r.decode_ms);
            snprintf(decode_gbps_col, sizeof decode_gbps_col, "%.1f", r.decode_gbps);
        }
        if (r.did_expand_compare) {
            snprintf(expnat_ex_col, sizeof expnat_ex_col, "%.4f", r.expand_native_expand_ms);
            snprintf(expnat_gm_col, sizeof expnat_gm_col, "%.4f", r.expand_native_gemm_ms);
            snprintf(expnat_err_col, sizeof expnat_err_col, "%.4f", r.expand_native_err);
        }
        if (r.did_prefill) {
            snprintf(expand_ms_col, sizeof expand_ms_col, "%.4f", r.expand_ms);
            snprintf(gemm_ms_col, sizeof gemm_ms_col, "%.4f", r.gemm_ms);
            snprintf(tf_col, sizeof tf_col, "%.2f", r.tf);
            snprintf(prefill_err_col, sizeof prefill_err_col, "%.4f", r.prefill_err);
        }
        if (r.did_fp8_ref) {
            snprintf(fp8_ms_col, sizeof fp8_ms_col, "%.4f", r.fp8_ref_ms);
            snprintf(fp8_metric_col, sizeof fp8_metric_col, "%.2f", r.fp8_ref_tf_or_gbps);
        }
        printf("%-6d %-7d %-7d | %-9s %-8s %-9s | %-8s %-8s %-9s | %-8s %-8s %-7s %-9s | %-8s %-8s\n",
               r.M, r.N, r.K, decode_col, decode_ms_col, decode_gbps_col,
               expnat_ex_col, expnat_gm_col, expnat_err_col, expand_ms_col, gemm_ms_col, tf_col, prefill_err_col,
               fp8_ms_col, fp8_metric_col);
    }
    printf("(fp8Ref* = GB/s for M<=32 rows [best of plain/split-K], TF for prefill rows)\n");

    // ---- task item 3: native prefill-tile in-kernel LUT expand vs expander+frozen ----
    bench_prefill_experiment(17408, 5120, 20);
    bench_prefill_experiment(5120, 17408, 20);

    if (!pos_ok || failures) {
        printf("\n%d/%zu SHAPES FAILED TO RUN; pos round-trip %s.\n",
               failures, sizeof(shapes) / sizeof(shapes[0]), pos_ok ? "PASS" : "FAIL");
        return 1;
    }
    printf("\nall shapes ran.\n");
    return 0;
}
