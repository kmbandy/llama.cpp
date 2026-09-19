// bench/gemm_ml84_radiance_bench.hip — MAD-305 ML8_4 radiance-port bench.
//
// Harness SHAPE (RNG setup, ShapeStat struct, gflops helper, SELFCHECK
// discipline, the two-oracle old-path/pure comparison) copied from
// bench/gemm_ml84_prefill_v2_bench.hip -- that file's own oracle
// (dequant_w / dequant_w_old, the old-path colmax+expand+frozen-trfeed
// chain) is REUSED VERBATIM below (old_orcl is small, per the task
// directive), only the "v2" kernel calls are replaced by this port's
// rdna4_gemm_ml84_radiance_prep / _retile_a / _radiance calls, and a
// SEPARATE isolated timing of the ported GEMM alone (excluding prep/
// retile, both load-time-only costs) is added.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O3 -I/opt/rocm/include -I"$HERE" \
//     gemm_blockscale.hip gemm_trfeed_prod.hip gemm_ml84_prod.hip \
//     gemm_ml84_radiance.hip bench/gemm_ml84_radiance_bench.hip \
//     -o out/gemm_ml84_radiance_bench
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "../bench/ml84_trfeed_layout.h"
#include "../bench/trfeed_ml84_kernels.h"
#include "../gemm_capi.h"
#include "../gemm_ml84_radiance.h"
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
    bool old_ok = false, rad_ok = false;
    double old_us = 0, rad_us = 0, rad_kernel_only_us = 0;
    double old_gflops = 0, rad_gflops = 0, rad_kernel_only_gflops = 0;
    double max_rel_diff_old_vs_rad = 0;
    double max_rel_diff_rad_vs_oracle = 0, max_rel_diff_old_vs_oracle = 0;
};

static double gflops_per_launch(int M, int N, int K, double us) {
    const double flops = 2.0 * (double) M * (double) N * (double) K;
    return flops / (us * 1e-6) / 1e9;
}

// ─────────────────────────────────────────────────────────────────────────
// ML84_RADIANCE_STAGECHECK=1: pure-CPU replica of the ported kernel's OWN
// staging arithmetic (gemm_ml84_radiance.hip's radp_gemm_ml84_atiled W-tile
// staging loop, kt/nt/L/s addressing + T-table gather), run against the
// REAL packed B_nib buffer and REAL T/colscale tables (via
// rdna4_ml84_radiance_prep_host), and compared element-by-element against
// `dequant_w_old` (the SAME double-quantized reference the old path's own
// oracle uses -- see the comment above `samples` in run_shape for why that,
// not the pure dequant, is the correct target here). No GPU involved: this
// exists to catch a nibble-order / tile-order / table-order bug
// DETERMINISTICALLY, independent of anything that could go wrong on the
// device side (WMMA operand layout, perm ISA semantics, etc).
//
// For every (kt, nt, lane L) this is EXACTLY the kernel's own read:
//   nib_base = ml84_tile_nib_base(kt, nt, NT)          (bench/ml84_trfeed_layout.h)
//   wq       = B_nib[nib_base + L*4 .. +4)             (one dword, lane L's dword)
//   g        = (kt*16) / QK_ML8                        (kt is a GLOBAL 16-wide k-tile)
//   for s in 0..7: nibble = (s<4) low-nibble of byte s of wq
//                            : high-nibble of byte (s-4) of wq
//                  n = nt*16 + (L & 15), k = kt*16 + ((L>>4)&1)*8 + s
//                  byte = T[g][n][nibble]               (radp_table_kernel's own layout)
//                  val  = decode_e4m3(byte) * colscale[n]
// and `val` must equal dequant_w_old(n, k) (the port's T/colscale double-
// quantization is bit-for-bit the same formula the old path's own
// B_shuf/b_scale_out double-quantization uses -- see run_shape).
// ─────────────────────────────────────────────────────────────────────────
static bool ml84_radiance_stagecheck(const std::vector<uint8_t>& B_nib_h, const std::vector<uint8_t>& T_h,
                                      const std::vector<float>& colscale_h, int N, int K,
                                      const std::function<float(int, int)>& dequant_old_ref,
                                      int M, int N_full) {
    (void) M; (void) N_full;
    if (!std::getenv("ML84_RADIANCE_STAGECHECK")) return true;
    const int NT = N / 16, KT = K / 16;
    // Sample every kt (K-tiles are cheap to walk in full) but only a few
    // representative nt tiles (first, middle, last) -- covers every group
    // g and every lane/s combination without an O(N*K) sweep on huge shapes.
    const int nts[] = {0, std::max(0, NT / 2), std::max(0, NT - 1)};
    int nchecked = 0, nmismatch = 0;
    int first_n = -1, first_k = -1, first_kt = -1, first_L = -1;
    uint8_t first_byte = 0;
    float first_got = 0.f, first_want = 0.f;
    for (int kt = 0; kt < KT; ++kt) {
        const int g = (kt * 16) / QK_ML8;
        for (int nt : nts) {
            if (nt >= NT) continue;
            const size_t nib_base = ml84_tile_nib_base(kt, nt, NT);
            for (int L = 0; L < 32; ++L) {
                if (nib_base + (size_t) L * 4 + 4 > B_nib_h.size()) continue;
                uint32_t wq;
                std::memcpy(&wq, B_nib_h.data() + nib_base + (size_t) L * 4, 4);
                const int nl = L & 15, half = (L >> 4) & 1;
                const int n = nt * 16 + nl;
                for (int s = 0; s < 8; ++s) {
                    const int nibble = (s < 4) ? (int) ((wq >> (s * 8)) & 0xF)
                                                : (int) ((wq >> ((s - 4) * 8 + 4)) & 0xF);
                    const int k = kt * 16 + half * 8 + s;
                    if (n >= N || k >= K) continue;
                    const size_t tidx = ((size_t) g * N + n) * 16 + nibble;
                    if (tidx >= T_h.size()) continue;
                    const uint8_t byte = T_h[tidx];
                    const float got = ml84_decode_e4m3(byte) * colscale_h[n];
                    const float want = dequant_old_ref(n, k);
                    ++nchecked;
                    const float rel = std::fabs(got - want) / (std::fabs(want) + 1e-3f);
                    if (rel > 1e-2f) {
                        ++nmismatch;
                        if (first_n < 0) {
                            first_n = n; first_k = k; first_kt = kt; first_L = L;
                            first_byte = byte; first_got = got; first_want = want;
                        }
                    }
                }
            }
        }
    }
    printf("--- ML84_RADIANCE_STAGECHECK N=%d K=%d checked=%d mismatches=%d ---\n", N, K, nchecked, nmismatch);
    if (nmismatch) {
        printf("    first mismatch: n=%d k=%d kt=%d lane=%d T-byte=0x%02x got=%.6g want=%.6g\n",
               first_n, first_k, first_kt, first_L, first_byte, first_got, first_want);
    }
    return nmismatch == 0;
}

// ─────────────────────────────────────────────────────────────────────────
// ML84_RADIANCE_STAGECHECK=1, second dimension: WMMA k-PAIRING check.
//
// The check above verifies VALUES (does the byte at a given LDS position
// decode to the right dequantized number) assuming the position <-> k
// mapping is whatever the nibble index `s` implies -- it does NOT verify
// that mapping against what the (unchanged, verbatim) WMMA-consuming read
// actually assumes. That gap is exactly what hid the real bug: radiance's
// staging store applies a BYTE INTERLEAVE (`perm(bo,be,0x05010400u)` /
// `perm(bo,be,0x07030602u)`, radiance_mxfp4_fp8.hip:675-677) that undoes
// ITS OWN checkpoint's even/odd-nibble packing -- applying that same
// interleave to ml84-4's ALREADY-sequential nibble order (B_nib) silently
// permutes which k each LDS byte position holds, so a lane's fp8 byte
// ends up paired with the WRONG k of the SAME magnitude range (hence
// right total magnitude, ~1/4 of terms correct by chance at the
// boundaries, exactly the 0.2515 correlation the orchestrator measured).
//
// Since a real 4-bit nibble cannot carry a unique k marker for K > 16,
// this check bypasses the nibble/T layer entirely and stages synthetic,
// uniquely-identifying LOCAL-k byte markers (0..LBK-1, one byte each --
// LBK <= 128 always fits in a byte) through the EXACT SAME staging
// write expression the kernel uses (radp_gemm_ml84_atiled's `sW[...] =
// uint2_radp{be, bo}` after the fix -- copy this by hand if that
// expression ever changes again), then reads them back through the EXACT
// SAME consuming read expression (`kk = step*16+kb8`, 8 bytes/j-tile),
// and asserts the recovered marker equals the k the read ADDRESS itself
// implies (the ISA-fixed A/B pairing: k = k0 + step*16 + (lane>>4)*8 + e,
// which the A-fragment side (af, unchanged) also assumes for the SAME
// lane -- see gemm_ml84_radiance.hip's "A tile bases" comment). Reports
// the first mismatch as (lane, byte, kA, kB): kA is the address-implied k
// (what the WMMA pairing requires), kB is the k whose marker actually
// landed there (what the staging bug -- if any -- put there instead).
// ─────────────────────────────────────────────────────────────────────────
static bool ml84_radiance_wmma_pairing_check(int N, int K) {
    if (!std::getenv("ML84_RADIANCE_STAGECHECK")) return true;
    constexpr int RADP_WN = 2, RADP_WM = 4, TN = 4, LBK = 64;
    constexpr int RADP_NTHREADS = 256;
    constexpr int BNF_T = RADP_WN * TN * 16;      // 128
    constexpr int KSTEPS_T = LBK / 16;            // 4
    constexpr int NTILES_T = BNF_T / 16;          // 8
    constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
    constexpr int WSLOTS = 4;
    constexpr int NW = TOT_SLOTS / (RADP_NTHREADS * WSLOTS);
    constexpr int LWSTR = LBK + 8;
    if (N < BNF_T || K < LBK) { printf("--- ML84_RADIANCE_WMMA_PAIRING_CHECK skipped (shape too small) ---\n"); return true; }
    const int n0 = 0, k0 = 0;

    // ---- staging: identical control flow to radp_gemm_ml84_atiled's W
    // staging loop, with `be`/`bo` replaced by synthetic LOCAL-k markers
    // (byte value = half*8+0..3 for be, half*8+4..7 for bo) instead of a
    // real T-table gather -- isolates the write/read byte-order
    // machinery from the nibble/LUT semantics already covered above. ----
    std::vector<uint8_t> sW((size_t) BNF_T * LWSTR, 0xFF);
    for (int tid = 0; tid < RADP_NTHREADS; ++tid) {
        for (int it = 0; it < NW; ++it) {
            const int sl = it * RADP_NTHREADS * WSLOTS + tid * WSLOTS;
            const int lane_ = sl & 31, rest = sl >> 5;
            const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
            const int wrow = ntl * 16 + (lane_ & 15);
            const int wcol = kst * 16 + (lane_ >> 4) * 8;
            const int gn = n0 + wrow;
            const int wrc = gn < N - WSLOTS ? gn : N - WSLOTS;
            for (int q = 0; q < WSLOTS; ++q) {
                (void) wrc;
                // Marker = wcol + b, NOT just half*8+b: it must be unique
                // across every k-step (kst) within the LDS row, matching
                // wcol's own value (kst*16 + half*8), or two different
                // steps would share marker 0..15 and a real cross-step
                // mispairing bug could hide behind a false PASS.
                uint8_t be[4], bo[4];
                for (int b = 0; b < 4; ++b) { be[b] = (uint8_t) (wcol + b); bo[b] = (uint8_t) (wcol + 4 + b); }
                uint32_t be32, bo32;
                std::memcpy(&be32, be, 4); std::memcpy(&bo32, bo, 4);
                // EXACT kernel write expression (post-fix): uint2_radp{be, bo}.
                std::memcpy(&sW[(size_t) (wrow + q) * LWSTR + wcol], &be32, 4);
                std::memcpy(&sW[(size_t) (wrow + q) * LWSTR + wcol + 4], &bo32, 4);
            }
        }
    }

    // ---- consumption: identical control flow to the WMMA-read loop. ----
    for (int wm = 0; wm < RADP_WM; ++wm)
        for (int wn = 0; wn < RADP_WN; ++wn)
            for (int lane = 0; lane < 32; ++lane) {
                const int col = lane & 15, kb8 = (lane >> 4) * 8;
                for (int j = 0; j < TN; ++j) {
                    const int row = wn * TN * 16 + j * 16 + col;
                    for (int step = 0; step < KSTEPS_T; ++step) {
                        const int kk = step * 16 + kb8;
                        for (int e = 0; e < 8; ++e) {
                            const uint8_t local_k_got = sW[(size_t) row * LWSTR + kk + e];
                            const int kA = step * 16 + kb8 + e;   // address-implied local k (0..LBK-1)
                            const int kB = (int) local_k_got;      // k whose marker actually landed here
                            if (kA != kB) {
                                printf("--- ML84_RADIANCE_WMMA_PAIRING_CHECK FAIL ---\n");
                                printf("    first mismatch: lane=%d byte=%d kA=%d kB=%d (wm=%d wn=%d j=%d step=%d)\n",
                                       lane, e, kA, kB, wm, wn, j, step);
                                return false;
                            }
                        }
                    }
                }
            }
    printf("--- ML84_RADIANCE_WMMA_PAIRING_CHECK PASS (all lanes x bytes x TN agree on k) ---\n");
    return true;
}

static int run_shape(int M, int N, int K, int iters, ShapeStat* out) {
    out->M = M; out->N = N; out->K = K;
    if (K % QK_ML8 != 0 || N % 16 != 0 || M % 128 != 0) {
        fprintf(stderr, "shape M=%d N=%d K=%d skipped: needs K%%64==0, N%%16==0, M%%128==0\n", M, N, K);
        return 3;
    }
    if (K % 128 != 0) {
        fprintf(stderr, "shape M=%d N=%d K=%d skipped: radiance port needs K%%128==0 (tiled-A requirement)\n", M, N, K);
        return 3;
    }
    const int n_groups_k = K / QK_ML8;

    std::mt19937 rng(0x52414450u ^ (uint32_t) (M * 1000003 + N * 7919 + K));   // 'RADP'
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

    // ---- fp32 A rows + per-row a_scale = amax/448, e4m3-encoded. Row-major,
    // exactly M rows. ----
    std::vector<float> Af((size_t) M * K, 0.0f);
    std::vector<float> a_scale(M, 0.0f);
    std::vector<uint8_t> Ah((size_t) M * K, 0);
    for (int m = 0; m < M; ++m) {
        float amax = 1e-6f;
        for (int k = 0; k < K; ++k) { Af[(size_t) m * K + k] = a_dist(rng); amax = std::max(amax, std::fabs(Af[(size_t) m * K + k])); }
        a_scale[m] = amax / 448.0f;
        for (int k = 0; k < K; ++k) Ah[(size_t) m * K + k] = ml84_encode_e4m3(Af[(size_t) m * K + k] / a_scale[m]);
    }

    // ---- OLD-PATH oracle (verbatim from gemm_ml84_prefill_v2_bench.hip):
    // the old path re-quantizes ml8-4 -> fp8 a SECOND time
    // (rdna4_expand_ml84_to_trfeed), so its own reference must replicate
    // that requantization, not the pure one-step dequant. ----
    std::vector<float> lut_amax_g(n_groups_k, 0.f);
    for (int g = 0; g < n_groups_k; ++g) {
        float mx = 0.f;
        for (int i = 0; i < 16; ++i) mx = std::max(mx, std::fabs(ml84_decode_e4m3(lut[(size_t) g * 16 + i])));
        lut_amax_g[g] = mx;
    }
    std::vector<float> b_scale_out_host(N, 0.f);
    for (int n = 0; n < N; ++n) {
        float mx = 0.f;
        for (int g = 0; g < n_groups_k; ++g) mx = std::max(mx, w[(size_t) n * n_groups_k + g].scale * lut_amax_g[g]);
        b_scale_out_host[n] = std::max(mx / 448.0f, 1e-12f);
    }
    auto dequant_w_old = [&](int n, int k) -> float {
        const float w_val = dequant_w(n, k);
        const float denom = b_scale_out_host[n];
        return ml84_decode_e4m3(ml84_encode_e4m3(w_val / denom)) * denom;
    };

    // ---- oracle samples: BOTH targets are computed, but unlike
    // bench/gemm_ml84_prefill_v2_bench.hip's v2 kernel (which reads ml8-4
    // in ONE step, no second requantization), the RADIANCE PORT explicitly
    // performs a SECOND fp8 requantization at load time -- T[g][n][c] =
    // e4m3_round(centroid[c]*b_scale_g[g][n]/colscale[n]) is exactly the
    // SAME double-quantization ml84_expand_kernel performs for the OLD
    // path (gemm_ml84_prod.hip), and colscale[n] is computed by the exact
    // b_scale_out formula (`max_g(b_scale_g*lut_amax)/448`, floored
    // 1e-12) that produces `b_scale_out_host` above -- i.e. colscale IS
    // b_scale_out_host, same numbers. So the value the port's WMMA
    // actually sees for element (n,k) is
    // `decode(encode(dequant_w(n,k)/b_scale_out_host[n])) * b_scale_out_host[n]`
    // == dequant_w_old(n,k), NOT the pure dequant_w(n,k). `ref` (pure) is
    // therefore the WRONG oracle for this port -- comparing against it
    // reproduces exactly the "near-cancellation blows up a relative-error
    // check against the wrong reference" failure mode the old-path comment
    // above already describes, and is very likely why an earlier version
    // of this bench reported rad_orcl in the tens-to-thousands while
    // old_orcl (correctly compared against ref_old) stayed at 1e-4..6e-3.
    // `ref_old` is the correct target for rad and is what it is compared
    // against below; `ref` is kept only for documentation/inspection. ----
    struct Sample { int i, j; float ref, ref_old; };
    std::vector<Sample> samples;
    const int SI = std::min(M, 8), SJ = std::min(N, 32);
    for (int si = 0; si < SI; si++) {
        const int i = (int) ((size_t) si * M / SI);
        for (int sj = 0; sj < SJ; sj++) {
            const int j = (int) ((size_t) sj * N / SJ);
            float acc = 0.f, acc_old = 0.f;
            for (int k = 0; k < K; k++) {
                const float av = ml84_decode_e4m3(Ah[(size_t) i * K + k]);
                acc += av * dequant_w(j, k);
                acc_old += av * dequant_w_old(j, k);
            }
            samples.push_back({i, j, acc * a_scale[i], acc_old * a_scale[i]});
        }
    }

    // ---- pack B_nib + b_scale_g ----
    std::vector<uint8_t> B_nib_h((size_t) N * K / 2);
    std::vector<float> b_scale_g_h((size_t) n_groups_k * N);
    ml84_trfeed_pack_host(w.data(), N, K, B_nib_h.data(), b_scale_g_h.data());

    // ---- ML84_RADIANCE_STAGECHECK=1: pure-CPU staging check, before any
    // device launch. Builds T/colscale via the host prep fallback so this
    // needs no GPU at all. ----
    if (std::getenv("ML84_RADIANCE_STAGECHECK")) {
        std::vector<uint8_t> T_h((size_t) n_groups_k * N * 16);
        std::vector<float> colscale_h(N);
        rdna4_ml84_radiance_prep_host(lut.data(), b_scale_g_h.data(), N, K, T_h.data(), colscale_h.data());
        const bool stage_ok = ml84_radiance_stagecheck(B_nib_h, T_h, colscale_h, N, K, dequant_w_old, M, N);
        const bool pairing_ok = ml84_radiance_wmma_pairing_check(N, K);
        if (!stage_ok || !pairing_ok) {
            fprintf(stderr, "  ML84_RADIANCE_STAGECHECK found mismatches at M=%d N=%d K=%d -- refusing to launch\n", M, N, K);
            return 3;
        }
    }

    uint8_t *d_lut = nullptr, *d_B_nib = nullptr, *d_A = nullptr, *d_B_shuf = nullptr, *d_AT = nullptr, *d_T = nullptr;
    float *d_b_scale_g = nullptr, *d_a_scale = nullptr, *d_C_old = nullptr, *d_C_rad = nullptr;
    float *d_b_scale_out = nullptr, *d_colscale = nullptr;
    HCK(hipMalloc(&d_lut, lut.size()));
    HCK(hipMalloc(&d_B_nib, B_nib_h.size()));
    HCK(hipMalloc(&d_b_scale_g, b_scale_g_h.size() * sizeof(float)));
    HCK(hipMalloc(&d_A, Ah.size()));
    HCK(hipMalloc(&d_a_scale, a_scale.size() * sizeof(float)));
    HCK(hipMalloc(&d_C_old, (size_t) M * N * sizeof(float)));
    HCK(hipMalloc(&d_C_rad, (size_t) M * N * sizeof(float)));
    HCK(hipMalloc(&d_B_shuf, (size_t) N * K));
    HCK(hipMalloc(&d_b_scale_out, (size_t) N * sizeof(float)));
    const size_t Mt = (size_t) (M + 15) / 16;
    const size_t at_bytes = Mt * 16 * (size_t) K;
    HCK(hipMalloc(&d_AT, at_bytes));
    HCK(hipMalloc(&d_T, (size_t) n_groups_k * N * 16));
    HCK(hipMalloc(&d_colscale, (size_t) N * sizeof(float)));
    HCK(hipMemcpy(d_lut, lut.data(), lut.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_B_nib, B_nib_h.data(), B_nib_h.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_b_scale_g, b_scale_g_h.data(), b_scale_g_h.size() * sizeof(float), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_A, Ah.data(), Ah.size(), hipMemcpyHostToDevice));
    HCK(hipMemcpy(d_a_scale, a_scale.data(), a_scale.size() * sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream = 0;

    // ---- correctness: one launch each, then compare ----
    HCK(hipMemset(d_C_old, 0, (size_t) M * N * sizeof(float)));
    HCK(hipMemset(d_C_rad, 0, (size_t) M * N * sizeof(float)));

    // OLD path: colmax + expand (rdna4_expand_ml84_to_trfeed) into B_shuf,
    // then the FROZEN rdna4_gemm_fp8_trfeed_f32.
    hipError_t rc_old = rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, d_B_shuf, d_b_scale_out, stream);
    if (rc_old == hipSuccess) {
        rc_old = rdna4_gemm_fp8_trfeed_f32(d_A, d_B_shuf, d_C_old, d_a_scale, d_b_scale_out, M, M, N, K, stream);
    }
    out->old_ok = (rc_old == hipSuccess);
    if (out->old_ok) HCK(hipStreamSynchronize(stream));

    // RADIANCE-PORT path: load-time prep (colscale + T), retile A, then GEMM.
    out->rad_ok = rdna4_gemm_ml84_radiance_prep(d_B_nib, d_lut, d_b_scale_g, N, K, d_T, d_colscale, stream);
    if (out->rad_ok) out->rad_ok = rdna4_gemm_ml84_radiance_retile_a(M, K, d_A, d_AT, stream);
    if (out->rad_ok) {
        HCK(hipGetLastError());
        HCK(hipStreamSynchronize(stream));
        out->rad_ok = rdna4_gemm_ml84_radiance(M, d_AT, d_a_scale, d_B_nib, d_T, d_colscale, d_C_rad, N, K, stream);
    }
    if (out->rad_ok) {
        HCK(hipGetLastError());
        HCK(hipStreamSynchronize(stream));
    }

    // ---- SELFCHECK: host-side replica of prep + retile, compared against
    // the device outputs, before trusting the GEMM's own correctness
    // check below (same "don't trust a launch that returned true"
    // discipline as gemm_ml84_decode_v2_bench.hip's ml84_v2_selfcheck). ----
    if (out->rad_ok && std::getenv("ML84_RADIANCE_SELFCHECK")) {
        std::vector<float> colscale_h(N), colscale_dev(N);
        std::vector<uint8_t> T_h((size_t) n_groups_k * N * 16), T_dev((size_t) n_groups_k * N * 16);
        std::vector<uint8_t> AT_h(at_bytes), AT_dev(at_bytes);
        rdna4_ml84_radiance_prep_host(lut.data(), b_scale_g_h.data(), N, K, T_h.data(), colscale_h.data());
        rdna4_ml84_radiance_retile_a_host(M, K, Ah.data(), AT_h.data());
        HCK(hipMemcpy(colscale_dev.data(), d_colscale, N * sizeof(float), hipMemcpyDeviceToHost));
        HCK(hipMemcpy(T_dev.data(), d_T, T_h.size(), hipMemcpyDeviceToHost));
        HCK(hipMemcpy(AT_dev.data(), d_AT, at_bytes, hipMemcpyDeviceToHost));
        bool ok = true;
        for (int n = 0; n < N; ++n)
            if (std::fabs(colscale_h[n] - colscale_dev[n]) > 1e-6f * std::max(1.0f, std::fabs(colscale_h[n]))) { ok = false; break; }
        if (ok && std::memcmp(T_h.data(), T_dev.data(), T_h.size()) != 0) ok = false;
        if (ok && std::memcmp(AT_h.data(), AT_dev.data(), at_bytes) != 0) ok = false;
        printf("--- ML84_RADIANCE_SELFCHECK M=%d N=%d K=%d %s ---\n", M, N, K, ok ? "PASS" : "*** FAIL ***");
        if (!ok) { out->rad_ok = false; }
    }

    std::vector<float> C_old_h, C_rad_h;
    if (out->old_ok) { C_old_h.resize((size_t) M * N); HCK(hipMemcpy(C_old_h.data(), d_C_old, C_old_h.size() * sizeof(float), hipMemcpyDeviceToHost)); }
    if (out->rad_ok) { C_rad_h.resize((size_t) M * N); HCK(hipMemcpy(C_rad_h.data(), d_C_rad, C_rad_h.size() * sizeof(float), hipMemcpyDeviceToHost)); }
    // ML84_RADIANCE_DUMP=<prefix>: write C_old (the production path, matches its oracle) and
    // C_rad as raw fp32 [M,N] so the error structure can be inspected offline.
    if (const char* dp = std::getenv("ML84_RADIANCE_DUMP")) {
        char fn[512];
        snprintf(fn, sizeof fn, "%s_old_%dx%dx%d.f32", dp, M, N, K); if (FILE* f = fopen(fn, "wb")) { fwrite(C_old_h.data(), 4, C_old_h.size(), f); fclose(f); }
        snprintf(fn, sizeof fn, "%s_rad_%dx%dx%d.f32", dp, M, N, K); if (FILE* f = fopen(fn, "wb")) { fwrite(C_rad_h.data(), 4, C_rad_h.size(), f); fclose(f); }
    }

    if (out->old_ok && out->rad_ok) {
        double max_rel = 0.0;
        for (size_t i = 0; i < C_old_h.size(); ++i) {
            const double d = std::fabs((double) C_old_h[i] - (double) C_rad_h[i]);
            const double r = d / (std::fabs((double) C_old_h[i]) + 1e-6);
            max_rel = std::max(max_rel, r);
        }
        out->max_rel_diff_old_vs_rad = max_rel;
    }
    for (const auto& s : samples) {
        const size_t idx = (size_t) s.i * N + s.j;
        if (out->rad_ok && idx < C_rad_h.size()) {
            // ref_old, not ref -- see the comment above `samples`: the port
            // double-quantizes through T/colscale exactly like the old path
            // double-quantizes through B_shuf/b_scale_out, so ref_old is its
            // correct oracle.
            const double r = std::fabs((double) C_rad_h[idx] - s.ref_old) / (std::fabs((double) s.ref_old) + 1e-3);
            out->max_rel_diff_rad_vs_oracle = std::max(out->max_rel_diff_rad_vs_oracle, r);
        }
        if (out->old_ok && idx < C_old_h.size()) {
            const double r = std::fabs((double) C_old_h[idx] - s.ref_old) / (std::fabs((double) s.ref_old) + 1e-3);
            out->max_rel_diff_old_vs_oracle = std::max(out->max_rel_diff_old_vs_oracle, r);
        }
    }

    // ---- timing: OLD is the full expand+GEMM chain. RAD is timed TWO
    // ways: `rad_us` = the full prep+retile+GEMM chain (apples-to-apples
    // against old's own "includes its own weight-side prep" accounting),
    // and `rad_kernel_only_us` = the ported GEMM kernel ALONE, prep/retile
    // excluded, since both are load-time-only costs in production (prep
    // runs once per weight load, retile is a per-call cost this bench
    // still counts in `rad_us` because production would fuse it into an
    // existing quant kernel, not because it belongs in the port's own
    // per-call budget -- see gemm_ml84_radiance.h's retile comment). ----
    if (out->old_ok) {
        for (int i = 0; i < 5; i++) {
            rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, d_B_shuf, d_b_scale_out, stream);
            rdna4_gemm_fp8_trfeed_f32(d_A, d_B_shuf, d_C_old, d_a_scale, d_b_scale_out, M, M, N, K, stream);
        }
        HCK(hipDeviceSynchronize());
        hipEvent_t s0, e0; hipEventCreate(&s0); hipEventCreate(&e0);
        hipEventRecord(s0, stream);
        for (int i = 0; i < iters; i++) {
            rdna4_expand_ml84_to_trfeed(d_B_nib, d_lut, d_b_scale_g, N, K, d_B_shuf, d_b_scale_out, stream);
            rdna4_gemm_fp8_trfeed_f32(d_A, d_B_shuf, d_C_old, d_a_scale, d_b_scale_out, M, M, N, K, stream);
        }
        hipEventRecord(e0, stream); hipEventSynchronize(e0);
        float ms = 0; hipEventElapsedTime(&ms, s0, e0);
        hipEventDestroy(s0); hipEventDestroy(e0);
        out->old_us = (double) ms * 1000.0 / iters;
        out->old_gflops = gflops_per_launch(M, N, K, out->old_us);
    }
    if (out->rad_ok) {
        for (int i = 0; i < 5; i++) {
            rdna4_gemm_ml84_radiance_prep(d_B_nib, d_lut, d_b_scale_g, N, K, d_T, d_colscale, stream);
            rdna4_gemm_ml84_radiance_retile_a(M, K, d_A, d_AT, stream);
            rdna4_gemm_ml84_radiance(M, d_AT, d_a_scale, d_B_nib, d_T, d_colscale, d_C_rad, N, K, stream);
        }
        HCK(hipDeviceSynchronize());
        hipEvent_t s1, e1; hipEventCreate(&s1); hipEventCreate(&e1);
        hipEventRecord(s1, stream);
        for (int i = 0; i < iters; i++) {
            rdna4_gemm_ml84_radiance_prep(d_B_nib, d_lut, d_b_scale_g, N, K, d_T, d_colscale, stream);
            rdna4_gemm_ml84_radiance_retile_a(M, K, d_A, d_AT, stream);
            rdna4_gemm_ml84_radiance(M, d_AT, d_a_scale, d_B_nib, d_T, d_colscale, d_C_rad, N, K, stream);
        }
        hipEventRecord(e1, stream); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, s1, e1);
        hipEventDestroy(s1); hipEventDestroy(e1);
        out->rad_us = (double) ms * 1000.0 / iters;
        out->rad_gflops = gflops_per_launch(M, N, K, out->rad_us);

        // ISOLATED ported-kernel-only timing: prep/retile run ONCE up
        // front (load-time), the timed loop launches only
        // rdna4_gemm_ml84_radiance.
        rdna4_gemm_ml84_radiance_prep(d_B_nib, d_lut, d_b_scale_g, N, K, d_T, d_colscale, stream);
        rdna4_gemm_ml84_radiance_retile_a(M, K, d_A, d_AT, stream);
        HCK(hipDeviceSynchronize());
        for (int i = 0; i < 5; i++) rdna4_gemm_ml84_radiance(M, d_AT, d_a_scale, d_B_nib, d_T, d_colscale, d_C_rad, N, K, stream);
        HCK(hipDeviceSynchronize());
        hipEvent_t s2, e2; hipEventCreate(&s2); hipEventCreate(&e2);
        hipEventRecord(s2, stream);
        for (int i = 0; i < iters; i++) rdna4_gemm_ml84_radiance(M, d_AT, d_a_scale, d_B_nib, d_T, d_colscale, d_C_rad, N, K, stream);
        hipEventRecord(e2, stream); hipEventSynchronize(e2);
        float ms2 = 0; hipEventElapsedTime(&ms2, s2, e2);
        hipEventDestroy(s2); hipEventDestroy(e2);
        out->rad_kernel_only_us = (double) ms2 * 1000.0 / iters;
        out->rad_kernel_only_gflops = gflops_per_launch(M, N, K, out->rad_kernel_only_us);
    }

    hipFree(d_lut); hipFree(d_B_nib); hipFree(d_b_scale_g); hipFree(d_A); hipFree(d_a_scale);
    hipFree(d_C_old); hipFree(d_C_rad); hipFree(d_B_shuf); hipFree(d_b_scale_out);
    hipFree(d_AT); hipFree(d_T); hipFree(d_colscale);
    return 0;
}

int main(int argc, char** argv) {
    const int iters = (argc > 1) ? std::atoi(argv[1]) : 20;
    const int Ms[] = {1024, 2048, 4096, 8192};
    struct NK { int N, K; };
    const NK nks[] = {
        {5120, 5120}, {17408, 5120}, {34816, 5120},
        {5120, 17408}, {17408, 17408}, {34816, 17408},
    };

    printf("%-6s %-8s %-8s | %-9s %-9s %-9s | %-9s %-9s %-9s | %-11s %-9s | %-10s %-10s %-10s\n",
           "M", "N", "K", "old_us", "old_TF", "old_ok", "rad_us", "rad_TF", "rad_ok",
           "rad_kern_us", "rad_kTF", "old_vs_rad", "rad_orcl", "old_orcl");
    bool any_fail = false;
    // ML84_RADIANCE_ONLY="M,N,K" restricts the run to one shape (keeps host RSS small for
    // the stage-check and lets a single shape be iterated on without the full sweep).
    int only_M = 0, only_N = 0, only_K = 0;
    if (const char* o = std::getenv("ML84_RADIANCE_ONLY")) sscanf(o, "%d,%d,%d", &only_M, &only_N, &only_K);
    for (const auto& nk : nks) {
        for (int M : Ms) {
            if (only_M && (M != only_M || nk.N != only_N || nk.K != only_K)) continue;
            ShapeStat st;
            const int rc = run_shape(M, nk.N, nk.K, iters, &st);
            if (rc != 0) { any_fail = true; continue; }
            printf("%-6d %-8d %-8d | %-9.1f %-9.2f %-9s | %-9.1f %-9.2f %-9s | %-11.1f %-9.2f | %-10.4g %-10.4g %-10.4g\n",
                   st.M, st.N, st.K,
                   st.old_us, st.old_gflops / 1000.0, st.old_ok ? "yes" : "NO",
                   st.rad_us, st.rad_gflops / 1000.0, st.rad_ok ? "yes" : "NO",
                   st.rad_kernel_only_us, st.rad_kernel_only_gflops / 1000.0,
                   st.max_rel_diff_old_vs_rad, st.max_rel_diff_rad_vs_oracle, st.max_rel_diff_old_vs_oracle);
            if (st.rad_ok && st.max_rel_diff_rad_vs_oracle > 0.05) {
                fprintf(stderr, "  *** radiance-port vs oracle relative error high at M=%d N=%d K=%d: %.4g\n",
                        st.M, nk.N, nk.K, st.max_rel_diff_rad_vs_oracle);
                any_fail = true;
            }
            if (st.old_ok && st.rad_ok && st.max_rel_diff_old_vs_rad > 0.05) {
                fprintf(stderr, "  *** old vs radiance-port relative diff high at M=%d N=%d K=%d: %.4g\n",
                        st.M, nk.N, nk.K, st.max_rel_diff_old_vs_rad);
                any_fail = true;
            }
            if (st.old_ok && st.rad_ok) {
                fprintf(stderr, "  speedup (full chain) at M=%d N=%d K=%d: %.2fx | kernel-only: %.2fx\n",
                        st.M, nk.N, nk.K, st.old_us / st.rad_us, st.old_us / st.rad_kernel_only_us);
            }
        }
    }
    printf("%s\n", any_fail ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return any_fail ? 1 : 0;
}
