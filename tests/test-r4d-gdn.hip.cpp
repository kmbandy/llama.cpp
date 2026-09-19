// Standalone correctness + timing check for the vendored R4D Gated DeltaNet entry points
// (ggml/src/ggml-cuda/r4d/, see r4d/README-ORIGIN.txt for provenance -- upstream radiance-libr4d
// commit b9e42ab-rx6). Exercises, directly against libr4d's own C ABI (NOT through the ggml
// adapter in ggml/src/ggml-cuda/mt_gdn_r4d.cu -- this is a check of the vendored kernels
// themselves, PLUS the value-head permutation the adapter applies around them):
//
//   r4d_gdn_kkt_solve_k128_c64_bf16                       (the chunk KKT preamble)
//   r4d_gdn_chunk_scan_k128_v128_c64_bf16                 (the chunked WY-transform scan -- the
//                                                          "prefill" path, but see below: it is
//                                                          exact for ANY chunk length, including
//                                                          a single token, so this is also what
//                                                          the ggml adapter uses for "decode")
//   r4d_gdn_recurrent_update_k128_v128_bf16_fp32state     (the fused recurrent decode step --
//                                                          tested here on its OWN terms, since its
//                                                          C ABI takes raw pre-activation
//                                                          alpha/beta + A_log/dt_bias and does its
//                                                          own q/k L2-norm + scale internally; the
//                                                          ggml adapter does not call this kernel
//                                                          -- see mt_gdn_r4d.cu's header comment)
//
// against a straightforward fp32 CPU reference of the gated delta rule's exact per-token
// recurrence, written in GGML'S OWN GQA convention (value head h pairs with key/query head
// h mod Hg -- gated_delta_net.cu's `iq1 = h_idx % neqk1`, confirmed by reading that kernel
// directly):
//
//     kh    = h mod Hg                     (ggml's key/query-head selection for value head h)
//     g_t   = exp(gl_t)                    (gl_t: raw per-token log-decay, NOT cumsum'd)
//     kv_t  = S_{t-1}^T k_t
//     d_t   = beta_t * (v_t - g_t * kv_t)
//     S_t   = g_t * S_{t-1} + k_t d_t^T
//     o_t   = scale * (S_t^T q_t)
//
// Geometry (fixed by the vendored kernels): head_k=128, head_v=128, chunk=64 (r4d_gdn_dims).
// Two GQA configs are tested, both against the SAME ggml-convention CPU reference:
//   - Hg=H=3   (R=1): the identity case for the head permutation mt_gdn_r4d.cu applies.
//   - Hg=16,H=48 (R=3): Qwen3.8-27B's actual linear-attention geometry (config.json under
//     /home/kmbandy/GitHub/vllm-radiance/models/Qwen3.8-27B-*/) -- exercises the permutation
//     perm(h) = R*(h mod Hg) + (h/Hg) this test applies by hand before/after the raw libr4d calls,
//     mirroring exactly what mt_gdn_r4d.cu does. A wrong or missing permutation here would pair
//     value heads with the wrong key/query heads whenever R>1 and fail against the reference.
// T=100 tokens per sequence (2 full 64-token chunks plus a 36-token remainder) for both configs.
//
// A second, reference-free check documents chunk_scan's state-carry contract directly (also
// catches state-permutation bugs, independent of whether the OUTPUT permutation happens to be
// right): the same inputs are run once as a single T-token chunk_scan call, and once as two
// back-to-back calls split at some T0, where the second call's h0 is the first call's ht. Split at
// 64 (chunk-aligned) is asserted to reproduce the whole-call output and final state bit-for-bit
// (tolerance 1e-3) and GATES the overall PASS/FAIL. Split at 50 (NOT chunk-aligned) is asserted to
// DIVERGE instead -- this is the production adapter's (mt_gdn_r4d.cu) own eligibility contract
// made explicit: chunk_scan's h0/ht carry is exact only at 64-token chunk boundaries, which is
// exactly why that file only ever calls it with T a multiple of 64 and declines everything else
// (including every single-token decode call) back to ggml's own exact per-token kernel. The
// misaligned split is reported but does NOT gate all_pass -- a small divergence there is what the
// contract predicts, not a bug in anything this test or the adapter controls.
//
// This file is deliberately NOT wired into tests/CMakeLists.txt, matching the existing convention
// for every other test in this tree that calls ggml-hip's internal (non-public-API) symbols --
// see tests/test-r4d-attn.hip.cpp, which this file is modeled on directly.
//
// Build (requires a build-hip configured with GGML_HIP_R4D, default ON since AMDGPU_TARGETS
// already contains gfx1201 in this tree's build-hip):
//
//   hipcc --offload-arch=gfx1201 -O2 -std=c++17 \
//       -I ggml/src/ggml-cuda -DGGML_HIP_R4D \
//       tests/test-r4d-gdn.hip.cpp \
//       -L build-hip/bin -lggml-hip -Wl,-rpath,$(pwd)/build-hip/bin \
//       -o /tmp/test-r4d-gdn
//   /tmp/test-r4d-gdn
//
// Expected output: one PASS/FAIL line per (config x check) -- chunk_scan output, chunk_scan final
// state, split@64 agreement (output and state, gates PASS), split@50 divergence (reported, does
// NOT gate PASS) -- for each of the two GQA configs; then the recurrent-decode-kernel check; then
// one us/launch line per path plus a dedicated N=1/T=1024 (a full, chunk-aligned production
// ubatch) timing-only run at the real Qwen3.8-27B GQA geometry.

#include "r4d/r4d.h"
#include "r4d/ggml-r4d.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1); } } while (0)

// ============================================================================================
// bf16 <-> f32 (software RTNE, matching r4d_common.h's f32_to_bf16)
// ============================================================================================
static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}
static inline float bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f; memcpy(&f, &u, 4);
    return f;
}
static std::vector<uint16_t> to_bf16(const std::vector<float> & f) {
    std::vector<uint16_t> out(f.size());
    for (size_t i = 0; i < f.size(); ++i) out[i] = f32_to_bf16(f[i]);
    return out;
}
static std::vector<float> widen_bf16(const std::vector<uint16_t> & q) {
    std::vector<float> out(q.size());
    for (size_t i = 0; i < q.size(); ++i) out[i] = bf16_to_f32(q[i]);
    return out;
}

// ============================================================================================
// Geometry (K/V dims and chunk size are fixed by the vendored kernels; N, H, Hg, T vary per test).
// ============================================================================================
static constexpr int K_DIM = 128;
static constexpr int V_DIM = 128;
static constexpr int CHUNK = 64;
static constexpr int N_SEQS = 2;

// value-head permutation mt_gdn_r4d.cu applies: ggml's (h mod Hg) key/query-head convention <->
// libr4d's (hv / R) one. Self-inverse under "apply forward both ways" (see that file's header
// comment for the proof); R = H/Hg.
static inline int perm_head(int h, int Hg, int R) { return R * (h % Hg) + (h / Hg); }

// L2-normalize q/k, per (n,t,head) row over the K_DIM axis, eps=1e-6, NO scale folded in --
// exactly what r4d_gdn_conv_w4_h128_bf16.hip's conv_prep kernel does before q/k ever reach
// kkt_solve/chunk_scan ("const float inv = 1.0f / sqrtf(ss + 1e-6f); // L2NORM_EPS, as in the
// kernel it replaces", region < 2 branch -- q and k, NOT v), and what ggml's own graph does via
// build_gdn_l2_norm() before GGML_OP_GATED_DELTA_NET (qwen35.cpp). Skipping this in synthetic
// test inputs left q/k at ~unit-per-element magnitude (128-dim uniform[-1,1], norm ~6.5) instead
// of unit norm, which is exactly what pushed the KKT triangular solve (I + strict_lower(diag(beta)
// K K^T e^dg))^-1 into a catastrophically ill-conditioned regime -- the naive per-token CPU
// reference has no such matrix inversion and stayed numerically sane, which is why the divergence
// looked like a contract bug rather than an input-domain one.
static void l2_normalize_rows(std::vector<float> & buf, size_t num_rows, int dim) {
    for (size_t r = 0; r < num_rows; ++r) {
        float * row = &buf[r * (size_t) dim];
        float ss = 0.0f;
        for (int i = 0; i < dim; ++i) ss += row[i] * row[i];
        const float inv = 1.0f / sqrtf(ss + 1e-6f);
        for (int i = 0; i < dim; ++i) row[i] *= inv;
    }
}

struct ErrStats { float max_abs, max_rel, mean_abs; };
static ErrStats compare(const std::vector<float> & got, const std::vector<float> & ref) {
    ErrStats e{0.0f, 0.0f, 0.0f};
    double sum_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float ad = fabsf(got[i] - ref[i]);
        e.max_abs = std::max(e.max_abs, ad);
        sum_abs += ad;
        if (fabsf(ref[i]) > 5e-2f) e.max_rel = std::max(e.max_rel, ad / fabsf(ref[i]));
    }
    e.mean_abs = (float)(sum_abs / std::max<size_t>(1, ref.size()));
    return e;
}
static void report(const char * name, const ErrStats & e, float abs_tol, float rel_tol, bool & all_pass) {
    const bool ok = (e.max_abs < abs_tol) && (e.max_rel < rel_tol);
    all_pass &= ok;
    printf("  %-34s max_abs=%.6f max_rel=%.6f mean_abs=%.6f -- %s\n",
           name, e.max_abs, e.max_rel, e.mean_abs, ok ? "PASS" : "FAIL");
}

// ============================================================================================
// CPU reference: exact per-token gated delta rule recurrence, in GGML'S OWN GQA convention
// (value head h pairs with key/query head h mod Hg -- see file header). Operates on ALREADY
// bf16-rounded q/k/v (so error against the GPU kernel is pure algorithmic/accumulation drift, not
// input quantization noise) plus fp32 g (raw, per-token, NOT cumsum'd), beta, h0.
//
// Layouts (all row-major, fastest axis last in the C loop nesting below, matching r4d.h /
// ggml's own contiguous tensor layout):
//   q,k : [N,T,Hg,K]   v : [N,T,H,V]   g,beta : [N,T,H]   h0,ht : [N,H,V,K]  (K fastest)   o : [N,T,H,V]
// ============================================================================================
static void cpu_gdn_reference(
    int N, int T, int H, int Hg,
    const std::vector<float> & q, const std::vector<float> & k, const std::vector<float> & v,
    const std::vector<float> & g_raw, const std::vector<float> & beta, const std::vector<float> & h0,
    float scale, std::vector<float> & o, std::vector<float> & ht)
{
    o.assign((size_t) N * T * H * V_DIM, 0.0f);
    ht = h0;

    // h0 is [N,H,V,K] (K fastest, gated_delta_net.cu's "S[i][col]" convention: i=K-axis,
    // col=V-axis) -- so S[k][v] (key axis fastest in our own scratch, to match the update loops
    // below) reads h0 as h0[(((n*H+h)*V+v)*K)+k].
    std::vector<double> S(K_DIM * V_DIM);
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            const int kh = h % Hg;   // ggml's key/query-head for value head h
            for (int kk = 0; kk < K_DIM; ++kk)
                for (int vv = 0; vv < V_DIM; ++vv)
                    S[kk * V_DIM + vv] = h0[(((size_t) n * H + h) * V_DIM + vv) * K_DIM + kk];

            for (int t = 0; t < T; ++t) {
                const size_t qkbase = (((size_t) n * T + t) * Hg + kh) * K_DIM;
                const size_t vbase  = (((size_t) n * T + t) * H  + h)  * V_DIM;
                const size_t gbase  = (size_t) n * T * H + (size_t) t * H + h;
                const float gt = expf(g_raw[gbase]);
                const float bt = beta[gbase];

                std::vector<double> kv(V_DIM, 0.0);
                for (int kk = 0; kk < K_DIM; ++kk) {
                    const double kval = k[qkbase + kk];
                    for (int vv = 0; vv < V_DIM; ++vv) kv[vv] += S[kk * V_DIM + vv] * kval;
                }
                std::vector<double> delta(V_DIM);
                for (int vv = 0; vv < V_DIM; ++vv)
                    delta[vv] = ((double) v[vbase + vv] - (double) gt * kv[vv]) * (double) bt;

                for (int kk = 0; kk < K_DIM; ++kk) {
                    const double kval = k[qkbase + kk];
                    for (int vv = 0; vv < V_DIM; ++vv)
                        S[kk * V_DIM + vv] = (double) gt * S[kk * V_DIM + vv] + kval * delta[vv];
                }
                for (int vv = 0; vv < V_DIM; ++vv) {
                    double acc = 0.0;
                    for (int kk = 0; kk < K_DIM; ++kk) acc += S[kk * V_DIM + vv] * (double) q[qkbase + kk];
                    o[vbase + vv] = (float) (acc * scale);
                }
            }
            for (int kk = 0; kk < K_DIM; ++kk)
                for (int vv = 0; vv < V_DIM; ++vv)
                    ht[(((size_t) n * H + h) * V_DIM + vv) * K_DIM + kk] = (float) S[kk * V_DIM + vv];
        }
    }
}

// Per-chunk (64-token, reset per sequence) inclusive prefix sum of raw g -> what kkt_solve /
// chunk_scan expect ("g already summed along the chunk", r4d.h). Operates in whatever H-axis
// order the caller already has g_raw in (this is applied AFTER permutation in the harness below,
// same order mt_gdn_r4d.cu computes it in).
static std::vector<float> chunk_cumsum(const std::vector<float> & g_raw, int N, int T, int H, int chunk_start_offset) {
    std::vector<float> out(g_raw.size());
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            float acc = 0.0f;
            for (int t = 0; t < T; ++t) {
                if ((t + chunk_start_offset) % CHUNK == 0) acc = 0.0f;
                const size_t idx = (size_t) n * T * H + (size_t) t * H + h;
                acc += g_raw[idx];
                out[idx] = acc;
            }
        }
    }
    return out;
}

// Permute the H axis of a [N,T,H,X] (X=1 for g/beta) or apply to a [N,H,V,K] state buffer.
static std::vector<float> permute_gbeta(const std::vector<float> & src, int N, int T, int H, int Hg, int R) {
    std::vector<float> out(src.size());
    for (int n = 0; n < N; ++n)
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h)
                out[(size_t) n * T * H + (size_t) t * H + perm_head(h, Hg, R)]
                    = src[(size_t) n * T * H + (size_t) t * H + h];
    return out;
}
static std::vector<float> permute_v(const std::vector<float> & src, int N, int T, int H, int Hg, int R, int V) {
    std::vector<float> out(src.size());
    for (int n = 0; n < N; ++n)
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h) {
                const int hp = perm_head(h, Hg, R);
                std::memcpy(&out[((size_t) n * T + t) * H * V + (size_t) hp * V],
                            &src[((size_t) n * T + t) * H * V + (size_t) h  * V], V * sizeof(float));
            }
    return out;
}
static std::vector<float> permute_state(const std::vector<float> & src, int N, int H, int Hg, int R, int VK) {
    std::vector<float> out(src.size());
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            const int hp = perm_head(h, Hg, R);
            std::memcpy(&out[((size_t) n * H + hp) * VK], &src[((size_t) n * H + h) * VK], VK * sizeof(float));
        }
    return out;
}
// Inverse of permute_v/permute_state: read at perm(h), write at h (same perm() formula --
// see mt_gdn_r4d.cu's header proof for why one formula serves both directions).
static std::vector<float> unpermute_v(const std::vector<float> & src, int N, int T, int H, int Hg, int R, int V) {
    std::vector<float> out(src.size());
    for (int n = 0; n < N; ++n)
        for (int t = 0; t < T; ++t)
            for (int h = 0; h < H; ++h) {
                const int hp = perm_head(h, Hg, R);
                std::memcpy(&out[((size_t) n * T + t) * H * V + (size_t) h  * V],
                            &src[((size_t) n * T + t) * H * V + (size_t) hp * V], V * sizeof(float));
            }
    return out;
}
static std::vector<float> unpermute_state(const std::vector<float> & src, int N, int H, int Hg, int R, int VK) {
    std::vector<float> out(src.size());
    for (int n = 0; n < N; ++n)
        for (int h = 0; h < H; ++h) {
            const int hp = perm_head(h, Hg, R);
            std::memcpy(&out[((size_t) n * H + h) * VK], &src[((size_t) n * H + hp) * VK], VK * sizeof(float));
        }
    return out;
}

// ============================================================================================
// One kkt_solve+chunk_scan call, in libr4d's own (permuted) head order. g_raw_perm is RAW
// (not yet cumsum'd) and already in libr4d order; chunk_start_offset lets a caller run a
// second half-length call whose chunk boundaries continue from where the first one left off
// (needed for the split-vs-whole agreement check).
// ============================================================================================
struct ChunkScanDeviceBufs {
    uint16_t *q_d=nullptr,*k_d=nullptr,*v_d=nullptr,*A_d=nullptr,*o_d=nullptr;
    float *g_d=nullptr,*beta_d=nullptr,*h0_d=nullptr,*ht_d=nullptr;
    int32_t *cu_d=nullptr;
};
static ChunkScanDeviceBufs alloc_chunk_scan_bufs(int N, int T, int H, int Hg) {
    ChunkScanDeviceBufs b;
    const size_t qk_n = (size_t) N * T * Hg * K_DIM;
    const size_t v_n  = (size_t) N * T * H * V_DIM;
    const size_t gb_n = (size_t) N * T * H;
    const size_t st_n = (size_t) N * H * V_DIM * K_DIM;
    HIP_CHECK(hipMalloc(&b.q_d, qk_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b.k_d, qk_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b.v_d, v_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b.o_d, v_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b.A_d, (size_t) N*T*H*CHUNK*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&b.g_d, gb_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&b.beta_d, gb_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&b.h0_d, st_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&b.ht_d, st_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&b.cu_d, (N+1)*sizeof(int32_t)));
    return b;
}
static void free_chunk_scan_bufs(ChunkScanDeviceBufs & b) {
    HIP_CHECK(hipFree(b.q_d)); HIP_CHECK(hipFree(b.k_d)); HIP_CHECK(hipFree(b.v_d));
    HIP_CHECK(hipFree(b.o_d)); HIP_CHECK(hipFree(b.A_d)); HIP_CHECK(hipFree(b.g_d));
    HIP_CHECK(hipFree(b.beta_d)); HIP_CHECK(hipFree(b.h0_d)); HIP_CHECK(hipFree(b.ht_d));
    HIP_CHECK(hipFree(b.cu_d));
}
// Runs kkt_solve+chunk_scan for one call; q/k/v/g_raw/beta/h0 are already in LIBR4D (permuted)
// head order and un-cumsum'd g. Returns (kkt_rc, scan_rc); fills o_perm/ht_perm (libr4d order).
static std::pair<int,int> run_chunk_scan_device(
    int N, int T, int H, int Hg, int chunk_start_offset,
    const std::vector<uint16_t> & q_bf, const std::vector<uint16_t> & k_bf, const std::vector<uint16_t> & v_bf,
    const std::vector<float> & g_raw_perm, const std::vector<float> & beta_perm, const std::vector<float> & h0_perm,
    hipStream_t stream, std::vector<uint16_t> & o_perm_bf, std::vector<float> & ht_perm)
{
    const std::vector<float> g_cumsum = chunk_cumsum(g_raw_perm, N, T, H, chunk_start_offset);
    std::vector<int32_t> cu(N + 1);
    for (int i = 0; i <= N; ++i) cu[i] = i * T;

    ChunkScanDeviceBufs b = alloc_chunk_scan_bufs(N, T, H, Hg);
    const size_t qk_n = (size_t) N*T*Hg*K_DIM, v_n = (size_t) N*T*H*V_DIM, gb_n = (size_t) N*T*H;
    const size_t st_n = (size_t) N*H*V_DIM*K_DIM;
    HIP_CHECK(hipMemcpy(b.q_d, q_bf.data(), qk_n*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.k_d, k_bf.data(), qk_n*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.v_d, v_bf.data(), v_n*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.g_d, g_cumsum.data(), gb_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.beta_d, beta_perm.data(), gb_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.h0_d, h0_perm.data(), st_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b.cu_d, cu.data(), (N+1)*sizeof(int32_t), hipMemcpyHostToDevice));

    const float scale = 1.0f / sqrtf((float) V_DIM);
    const int T_total = N * T;
    const int rc1 = r4d_gdn_kkt_solve_k128_c64_bf16(b.k_d, b.beta_d, b.g_d, b.A_d, b.cu_d,
                                                     N, T_total, H, Hg, K_DIM, CHUNK, stream);
    const int rc2 = r4d_gdn_chunk_scan_k128_v128_c64_bf16(b.q_d, b.k_d, b.v_d, b.A_d, b.g_d, b.beta_d,
                                                           b.h0_d, b.o_d, b.ht_d, b.cu_d,
                                                           N, H, Hg, K_DIM, V_DIM, CHUNK, scale, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    o_perm_bf.resize(v_n);
    ht_perm.resize(st_n);
    if (rc1 == 0 && rc2 == 0) {
        HIP_CHECK(hipMemcpy(o_perm_bf.data(), b.o_d, v_n*sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ht_perm.data(), b.ht_d, st_n*sizeof(float), hipMemcpyDeviceToHost));
    }
    free_chunk_scan_bufs(b);
    return {rc1, rc2};
}

// ============================================================================================
// Path 1 (per GQA config): kkt_solve + chunk_scan, WITH the value-head permutation applied by
// hand (mirroring mt_gdn_r4d.cu), checked against the ggml-convention CPU reference. Also runs
// the split-vs-whole state-carry agreement check.
// ============================================================================================
static void run_chunk_scan_path(const char * cfg_name, int H, int Hg, int T, std::mt19937 & rng, bool & all_pass) {
    const int R = H / Hg;
    printf("== chunked scan: %s (N=%d H=%d Hg=%d R=%d T=%d) ==\n", cfg_name, N_SEQS, H, Hg, R, T);

    std::uniform_real_distribution<float> qkv_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> g_dist(-0.30f, -0.01f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);
    std::uniform_real_distribution<float> h0_dist(-0.2f, 0.2f);

    const size_t qk_n = (size_t) N_SEQS * T * Hg * K_DIM;
    const size_t v_n  = (size_t) N_SEQS * T * H * V_DIM;
    const size_t gb_n = (size_t) N_SEQS * T * H;
    const size_t st_n = (size_t) N_SEQS * H * V_DIM * K_DIM;
    const int VK = V_DIM * K_DIM;

    std::vector<float> q_f(qk_n), k_f(qk_n), v_f(v_n), g_raw(gb_n), beta_f(gb_n), h0_f(st_n);
    for (auto & x : q_f) x = qkv_dist(rng);
    for (auto & x : k_f) x = qkv_dist(rng);
    for (auto & x : v_f) x = qkv_dist(rng);
    for (auto & x : g_raw) x = g_dist(rng);
    for (auto & x : beta_f) x = beta_dist(rng);
    for (auto & x : h0_f) x = h0_dist(rng);

    // q/k reach kkt_solve/chunk_scan already L2-normalized in both the real pipeline (conv_prep,
    // or ggml's build_gdn_l2_norm) -- normalize here too, see l2_normalize_rows's comment.
    l2_normalize_rows(q_f, (size_t) N_SEQS * T * Hg, K_DIM);
    l2_normalize_rows(k_f, (size_t) N_SEQS * T * Hg, K_DIM);

    const std::vector<float> q_bf_f = widen_bf16(to_bf16(q_f));
    const std::vector<float> k_bf_f = widen_bf16(to_bf16(k_f));
    const std::vector<float> v_bf_f = widen_bf16(to_bf16(v_f));

    const float scale = 1.0f / sqrtf((float) V_DIM);
    std::vector<float> ref_o, ref_ht;
    cpu_gdn_reference(N_SEQS, T, H, Hg, q_bf_f, k_bf_f, v_bf_f, g_raw, beta_f, h0_f, scale, ref_o, ref_ht);

    // ---- stage everything in libr4d (permuted) order, exactly as mt_gdn_r4d.cu does ----
    const std::vector<uint16_t> q_bf = to_bf16(q_f), k_bf = to_bf16(k_f);
    const std::vector<float> v_perm  = permute_v(v_f, N_SEQS, T, H, Hg, R, V_DIM);
    const std::vector<uint16_t> v_bf = to_bf16(v_perm);
    const std::vector<float> g_perm    = permute_gbeta(g_raw, N_SEQS, T, H, Hg, R);
    const std::vector<float> beta_perm = permute_gbeta(beta_f, N_SEQS, T, H, Hg, R);
    const std::vector<float> h0_perm   = permute_state(h0_f, N_SEQS, H, Hg, R, VK);

    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));

    // ---- whole-call path ----
    std::vector<uint16_t> o_perm_bf; std::vector<float> ht_perm;
    const auto rc = run_chunk_scan_device(N_SEQS, T, H, Hg, /*chunk_start_offset=*/0,
                                           q_bf, k_bf, v_bf, g_perm, beta_perm, h0_perm,
                                           stream, o_perm_bf, ht_perm);
    if (rc.first != 0 || rc.second != 0) {
        printf("  launch rejected shape (kkt_rc=%d scan_rc=%d) -- FAIL\n", rc.first, rc.second);
        all_pass = false;
    } else {
        const std::vector<float> o_got  = unpermute_v(widen_bf16(o_perm_bf), N_SEQS, T, H, Hg, R, V_DIM);
        const std::vector<float> ht_got = unpermute_state(ht_perm, N_SEQS, H, Hg, R, VK);
        report("chunk_scan output", compare(o_got, ref_o), 0.08f, 0.08f, all_pass);
        report("chunk_scan final state", compare(ht_got, ref_ht), 0.08f, 0.08f, all_pass);
    }

    // ---- split-vs-whole state-carry agreement ----------------------------------------------
    // mt_gdn_r4d.cu (production adapter) now only ever calls chunk_scan with T a multiple of 64
    // -- one call's h0/ht carry into the next is bit-exact ONLY at that boundary; this is a
    // property of the chunked algorithm, not a marshalling bug (see that file's header comment,
    // "Eligibility: T must be a whole multiple of the 64-token chunk"). Checked here two ways
    // against the SAME whole-call reference: splitting at 64 (chunk-aligned) must reproduce it
    // bit-for-bit and GATES all_pass; splitting at 50 (not chunk-aligned) is expected to DIVERGE
    // -- checked as a negative assertion (documents the contract; does not gate all_pass, since a
    // small divergence there is exactly what the contract predicts, not a bug).
    if (T >= 2 && rc.first == 0 && rc.second == 0) {
        auto slice_qk = [&](const std::vector<uint16_t> & buf, int t0, int len) {
            std::vector<uint16_t> out((size_t) N_SEQS * len * Hg * K_DIM);
            for (int n = 0; n < N_SEQS; ++n)
                std::memcpy(&out[(size_t) n*len*Hg*K_DIM], &buf[((size_t) n*T + t0)*Hg*K_DIM],
                            (size_t) len*Hg*K_DIM*sizeof(uint16_t));
            return out;
        };
        auto slice_v = [&](const std::vector<uint16_t> & buf, int t0, int len) {
            std::vector<uint16_t> out((size_t) N_SEQS * len * H * V_DIM);
            for (int n = 0; n < N_SEQS; ++n)
                std::memcpy(&out[(size_t) n*len*H*V_DIM], &buf[((size_t) n*T + t0)*H*V_DIM],
                            (size_t) len*H*V_DIM*sizeof(uint16_t));
            return out;
        };
        auto slice_gb = [&](const std::vector<float> & buf, int t0, int len) {
            std::vector<float> out((size_t) N_SEQS * len * H);
            for (int n = 0; n < N_SEQS; ++n)
                std::memcpy(&out[(size_t) n*len*H], &buf[((size_t) n*T + t0)*H], (size_t) len*H*sizeof(float));
            return out;
        };
        const std::vector<float> o_whole = widen_bf16(o_perm_bf);

        // gate_pass: whether this split's result feeds all_pass. require_exact selects PASS
        // criterion: tight tolerance (aligned) vs. must-diverge (misaligned, documents the
        // contract rather than testing correctness of anything this file controls).
        auto run_split_check = [&](int T0, const char * label, bool require_exact, bool gate_pass) {
            if (T0 <= 0 || T0 >= T) return;
            const int T1 = T - T0;
            std::vector<uint16_t> o0_bf, o1_bf; std::vector<float> ht0, ht1;
            const auto rcs0 = run_chunk_scan_device(N_SEQS, T0, H, Hg, 0,
                slice_qk(q_bf,0,T0), slice_qk(k_bf,0,T0), slice_v(v_bf,0,T0),
                slice_gb(g_perm,0,T0), slice_gb(beta_perm,0,T0), h0_perm, stream, o0_bf, ht0);
            const auto rcs1 = run_chunk_scan_device(N_SEQS, T1, H, Hg, T0,
                slice_qk(q_bf,T0,T1), slice_qk(k_bf,T0,T1), slice_v(v_bf,T0,T1),
                slice_gb(g_perm,T0,T1), slice_gb(beta_perm,T0,T1), ht0, stream, o1_bf, ht1);
            if (rcs0.first || rcs0.second || rcs1.first || rcs1.second) {
                printf("  %-34s launch rejected shape -- FAIL\n", label);
                if (gate_pass) all_pass = false;
                return;
            }
            std::vector<float> o_split(v_n);
            const std::vector<float> o0 = widen_bf16(o0_bf), o1 = widen_bf16(o1_bf);
            for (int n = 0; n < N_SEQS; ++n) {
                std::memcpy(&o_split[(size_t) n*T*H*V_DIM], &o0[(size_t) n*T0*H*V_DIM], (size_t) T0*H*V_DIM*sizeof(float));
                std::memcpy(&o_split[(size_t) n*T*H*V_DIM + (size_t) T0*H*V_DIM], &o1[(size_t) n*T1*H*V_DIM],
                            (size_t) T1*H*V_DIM*sizeof(float));
            }
            const ErrStats eo = compare(o_split, o_whole);
            const ErrStats es = compare(ht1, ht_perm);
            if (require_exact) {
                bool ok = true;
                report(label, eo, 1e-3f, 1e-3f, ok);
                char state_label[64]; snprintf(state_label, sizeof state_label, "%s (state)", label);
                report(state_label, es, 1e-3f, 1e-3f, ok);
                if (gate_pass) all_pass &= ok;
            } else {
                const bool diverges = eo.max_abs > 1e-3f || es.max_abs > 1e-3f;
                printf("  %-34s max_abs=%.6f/%.6f (out/state) -- %s\n", label, eo.max_abs, es.max_abs,
                       diverges ? "PASS (diverges as the contract predicts)" : "FAIL (unexpectedly exact)");
                if (gate_pass) all_pass &= diverges;
            }
        };

        run_split_check(64, "split@64 (chunk-aligned)", /*require_exact=*/true,  /*gate_pass=*/true);
        run_split_check(50, "split@50 (NOT aligned)",   /*require_exact=*/false, /*gate_pass=*/false);
    }

    // ---- timing: whole-call path, 20 launches ----
    hipEvent_t ev0, ev1; HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    std::vector<uint16_t> tmp_o; std::vector<float> tmp_ht;
    for (int i = 0; i < 3; ++i) run_chunk_scan_device(N_SEQS,T,H,Hg,0,q_bf,k_bf,v_bf,g_perm,beta_perm,h0_perm,stream,tmp_o,tmp_ht);
    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < 20; ++i) run_chunk_scan_device(N_SEQS,T,H,Hg,0,q_bf,k_bf,v_bf,g_perm,beta_perm,h0_perm,stream,tmp_o,tmp_ht);
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    printf("  %.2f us/launch (kkt_solve+chunk_scan pair, 20 launches, host-side permute+memcpy included)\n", ms * 1000.0f / 20.0f);

    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipStreamDestroy(stream));
}

// ============================================================================================
// Timing only: N=1, T=1024 (a full production-sized ubatch: 1024 = 16*64, chunk-aligned), the
// Qwen3.8-27B GQA geometry (H=48, Hg=16) -- gives a per-1024-token-slice kkt_solve+chunk_scan cost
// to compare against the existing ggml gated_delta_net kernel's own per-1024-token cost later. No
// correctness check here (already covered by run_chunk_scan_path's T=100, R=3 case above).
// ============================================================================================
static void run_1024_timing(std::mt19937 & rng) {
    const int H = 48, Hg = 16, R = H / Hg, N = 1, T = 1024;
    printf("== timing only: N=%d T=%d H=%d Hg=%d (production ubatch size) ==\n", N, T, H, Hg);

    std::uniform_real_distribution<float> qkv_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> g_dist(-0.30f, -0.01f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);
    std::uniform_real_distribution<float> h0_dist(-0.2f, 0.2f);

    const size_t qk_n = (size_t) N * T * Hg * K_DIM;
    const size_t v_n  = (size_t) N * T * H * V_DIM;
    const size_t gb_n = (size_t) N * T * H;
    const size_t st_n = (size_t) N * H * V_DIM * K_DIM;
    const int VK = V_DIM * K_DIM;

    std::vector<float> q_f(qk_n), k_f(qk_n), v_f(v_n), g_raw(gb_n), beta_f(gb_n), h0_f(st_n);
    for (auto & x : q_f) x = qkv_dist(rng);
    for (auto & x : k_f) x = qkv_dist(rng);
    for (auto & x : v_f) x = qkv_dist(rng);
    for (auto & x : g_raw) x = g_dist(rng);
    for (auto & x : beta_f) x = beta_dist(rng);
    for (auto & x : h0_f) x = h0_dist(rng);
    l2_normalize_rows(q_f, (size_t) N * T * Hg, K_DIM);
    l2_normalize_rows(k_f, (size_t) N * T * Hg, K_DIM);

    const std::vector<uint16_t> q_bf = to_bf16(q_f), k_bf = to_bf16(k_f);
    const std::vector<float> v_perm  = permute_v(v_f, N, T, H, Hg, R, V_DIM);
    const std::vector<uint16_t> v_bf = to_bf16(v_perm);
    const std::vector<float> g_perm    = permute_gbeta(g_raw, N, T, H, Hg, R);
    const std::vector<float> beta_perm = permute_gbeta(beta_f, N, T, H, Hg, R);
    const std::vector<float> h0_perm   = permute_state(h0_f, N, H, Hg, R, VK);

    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));
    std::vector<uint16_t> tmp_o; std::vector<float> tmp_ht;
    for (int i = 0; i < 3; ++i) run_chunk_scan_device(N,T,H,Hg,0,q_bf,k_bf,v_bf,g_perm,beta_perm,h0_perm,stream,tmp_o,tmp_ht);
    hipEvent_t ev0, ev1; HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < 20; ++i) run_chunk_scan_device(N,T,H,Hg,0,q_bf,k_bf,v_bf,g_perm,beta_perm,h0_perm,stream,tmp_o,tmp_ht);
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    printf("  %.2f us/launch (kkt_solve+chunk_scan pair, T=1024, 20 launches, host-side permute+memcpy included)\n",
           ms * 1000.0f / 20.0f);

    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipStreamDestroy(stream));
}

// ============================================================================================
// Path 2: r4d_gdn_recurrent_update_k128_v128_bf16_fp32state, tested on ITS OWN terms (raw
// alpha/beta + A_log/dt_bias, internal q/k L2-norm+scale) -- see mt_gdn_r4d.cu's header for why
// the ggml adapter does not call this kernel. One token, N_SEQS sequences, Hg=H (this kernel's
// own GQA parameter is exercised elsewhere in its own right; the adapter never drives it, so this
// test keeps Hg=H here and does not layer the permutation on top of it), non-speculative
// (num_accepted = nullptr), non-paged (state slot n+1 for sequence n; slot 0 reserved as
// NULL_BLOCK_ID).
// ============================================================================================
static void run_recurrent_update_path(std::mt19937 & rng, bool & all_pass) {
    printf("== recurrent decode (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state) ==\n");
    const int H = 3, Hg = 3;

    std::uniform_real_distribution<float> qkv_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> ab_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> alog_dist(-2.0f, 0.5f);
    std::uniform_real_distribution<float> dtb_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> h0_dist(-0.2f, 0.2f);

    const size_t qk_n = (size_t) N_SEQS * H * K_DIM;    // T=1, Hg==H here
    const size_t v_n  = (size_t) N_SEQS * H * V_DIM;
    const size_t ab_n = (size_t) N_SEQS * H;
    const size_t st_n = (size_t) (N_SEQS + 1) * H * V_DIM * K_DIM;

    std::vector<float> q_raw(qk_n), k_raw(qk_n), v_f(v_n), a_f(ab_n), b_f(ab_n);
    std::vector<float> A_log(H), dt_bias(H), h0_f(st_n, 0.0f);
    for (auto & x : q_raw) x = qkv_dist(rng);
    for (auto & x : k_raw) x = qkv_dist(rng);
    for (auto & x : v_f)   x = qkv_dist(rng);
    for (auto & x : a_f)   x = ab_dist(rng);
    for (auto & x : b_f)   x = ab_dist(rng);
    for (auto & x : A_log) x = alog_dist(rng);
    for (auto & x : dt_bias) x = dtb_dist(rng);
    for (size_t i = H * (size_t) V_DIM * K_DIM; i < st_n; ++i) h0_f[i] = h0_dist(rng);

    const float scale = 1.0f / sqrtf((float) V_DIM);
    const float sp_thr = 20.0f;

    std::vector<float> q_bf_f(qk_n), k_bf_f(qk_n), g_raw(ab_n), beta_f(ab_n);
    {
        const std::vector<float> q_rounded = widen_bf16(to_bf16(q_raw));
        const std::vector<float> k_rounded = widen_bf16(to_bf16(k_raw));
        for (int n = 0; n < N_SEQS; ++n) {
            for (int h = 0; h < H; ++h) {
                double sq = 0.0, sk = 0.0;
                const size_t base = ((size_t) n * H + h) * K_DIM;
                for (int i = 0; i < K_DIM; ++i) { sq += (double) q_rounded[base+i]*q_rounded[base+i];
                                                   sk += (double) k_rounded[base+i]*k_rounded[base+i]; }
                const float qs = 1.0f / sqrtf((float) sq + 1e-6f) * scale;
                const float ks = 1.0f / sqrtf((float) sk + 1e-6f);
                for (int i = 0; i < K_DIM; ++i) { q_bf_f[base+i] = q_rounded[base+i]*qs;
                                                   k_bf_f[base+i] = k_rounded[base+i]*ks; }
                const size_t gb = (size_t) n * H + h;
                const float alog = expf(A_log[h]);
                const float sp_in = a_f[gb] + dt_bias[h];
                const float sp = (sp_in > sp_thr) ? sp_in
                                 : (sp_in > 0.0f ? sp_in + logf(1.0f+expf(-sp_in)) : logf(1.0f+expf(sp_in)));
                g_raw[gb]  = -alog * sp;
                beta_f[gb] = 1.0f / (1.0f + expf(-b_f[gb]));
            }
        }
    }
    std::vector<float> v_bf_f = widen_bf16(to_bf16(v_f));
    std::vector<float> ref_h0(N_SEQS * (size_t) H * V_DIM * K_DIM);
    for (size_t i = 0; i < ref_h0.size(); ++i)
        ref_h0[i] = h0_f[H * (size_t) V_DIM * K_DIM + i];

    // one-step recurrence, inlined (T=1 case of the same math cpu_gdn_reference implements).
    std::vector<float> ref_o(v_n, 0.0f), ref_ht = ref_h0;
    for (int n = 0; n < N_SEQS; ++n) {
        for (int h = 0; h < H; ++h) {
            std::vector<double> S(K_DIM * V_DIM);
            for (int kk = 0; kk < K_DIM; ++kk)
                for (int vv = 0; vv < V_DIM; ++vv)
                    S[kk*V_DIM+vv] = ref_h0[(((size_t)n*H+h)*V_DIM+vv)*K_DIM+kk];
            const size_t qkbase = ((size_t) n * H + h) * K_DIM;
            const size_t vbase  = ((size_t) n * H + h) * V_DIM;
            const size_t gb     = (size_t) n * H + h;
            const float gt = expf(g_raw[gb]), bt = beta_f[gb];
            std::vector<double> kv(V_DIM, 0.0);
            for (int kk = 0; kk < K_DIM; ++kk) {
                const double kval = k_bf_f[qkbase+kk];
                for (int vv = 0; vv < V_DIM; ++vv) kv[vv] += S[kk*V_DIM+vv]*kval;
            }
            std::vector<double> delta(V_DIM);
            for (int vv = 0; vv < V_DIM; ++vv) delta[vv] = ((double) v_bf_f[vbase+vv] - (double) gt*kv[vv]) * (double) bt;
            for (int kk = 0; kk < K_DIM; ++kk) {
                const double kval = k_bf_f[qkbase+kk];
                for (int vv = 0; vv < V_DIM; ++vv) S[kk*V_DIM+vv] = (double) gt*S[kk*V_DIM+vv] + kval*delta[vv];
            }
            for (int vv = 0; vv < V_DIM; ++vv) {
                double acc = 0.0;
                for (int kk = 0; kk < K_DIM; ++kk) acc += S[kk*V_DIM+vv] * (double) q_bf_f[qkbase+kk];
                ref_o[vbase+vv] = (float) acc;   // scale already folded into q_bf_f
            }
            for (int kk = 0; kk < K_DIM; ++kk)
                for (int vv = 0; vv < V_DIM; ++vv)
                    ref_ht[(((size_t)n*H+h)*V_DIM+vv)*K_DIM+kk] = (float) S[kk*V_DIM+vv];
        }
    }

    const std::vector<uint16_t> q_bf = to_bf16(q_raw), k_bf = to_bf16(k_raw), v_bf = to_bf16(v_f);
    std::vector<int32_t> cu(N_SEQS + 1);
    for (int i = 0; i <= N_SEQS; ++i) cu[i] = i;
    std::vector<int32_t> sidx(N_SEQS);
    for (int n = 0; n < N_SEQS; ++n) sidx[n] = n + 1;

    uint16_t *q_d=nullptr,*k_d=nullptr,*v_d=nullptr,*o_d=nullptr;
    float *a_d=nullptr,*b_d=nullptr,*Alog_d=nullptr,*dtb_d=nullptr,*state_d=nullptr;
    int32_t *cu_d=nullptr,*sidx_d=nullptr;
    HIP_CHECK(hipMalloc(&q_d, qk_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&k_d, qk_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&v_d, v_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&o_d, v_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&a_d, ab_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&b_d, ab_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&Alog_d, H*sizeof(float)));
    HIP_CHECK(hipMalloc(&dtb_d, H*sizeof(float)));
    HIP_CHECK(hipMalloc(&state_d, st_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&cu_d, (N_SEQS+1)*sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&sidx_d, N_SEQS*sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(q_d, q_bf.data(), qk_n*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_d, k_bf.data(), qk_n*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_d, v_bf.data(), v_n*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(a_d, a_f.data(), ab_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_d, b_f.data(), ab_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Alog_d, A_log.data(), H*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dtb_d, dt_bias.data(), H*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cu_d, cu.data(), (N_SEQS+1)*sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sidx_d, sidx.data(), N_SEQS*sizeof(int32_t), hipMemcpyHostToDevice));

    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));
    const long state_slot_stride = (long) H * V_DIM * K_DIM;
    const long state_head_stride = (long) V_DIM * K_DIM;

    auto launch_once = [&]() -> int {
        HIP_CHECK(hipMemcpy(state_d, h0_f.data(), st_n*sizeof(float), hipMemcpyHostToDevice));
        return r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
            q_d, k_d, v_d, a_d, b_d, /*ab_stride=*/H, /*ab_is_bf16=*/0,
            Alog_d, dtb_d, state_d, state_slot_stride, state_head_stride, o_d, cu_d,
            sidx_d, /*indices_stride=*/1, /*num_accepted=*/nullptr,
            /*z_gate=*/nullptr, /*norm_weight=*/nullptr, /*norm_eps=*/0.0f, /*norm_act=*/0,
            N_SEQS, H, Hg, K_DIM, V_DIM, scale, sp_thr, stream);
    };

    const int rc = launch_once();
    HIP_CHECK(hipStreamSynchronize(stream));

    if (rc != 0) {
        printf("  launch rejected shape (rc=%d) -- FAIL\n", rc);
        all_pass = false;
    } else {
        std::vector<uint16_t> o_bf(v_n);
        std::vector<float> state_got(st_n);
        HIP_CHECK(hipMemcpy(o_bf.data(), o_d, v_n*sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(state_got.data(), state_d, st_n*sizeof(float), hipMemcpyDeviceToHost));
        const std::vector<float> o_got = widen_bf16(o_bf);
        std::vector<float> ht_got(N_SEQS * (size_t) H * V_DIM * K_DIM);
        for (size_t i = 0; i < ht_got.size(); ++i) ht_got[i] = state_got[H*(size_t)V_DIM*K_DIM + i];
        report("recurrent_update output", compare(o_got, ref_o), 0.08f, 0.08f, all_pass);
        report("recurrent_update final state", compare(ht_got, ref_ht), 0.08f, 0.08f, all_pass);
    }

    hipEvent_t ev0, ev1; HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    for (int i = 0; i < 3; ++i) launch_once();
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < 20; ++i) launch_once();
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    printf("  %.2f us/launch (recurrent_update, 20 launches, state re-uploaded each iter)\n", ms * 1000.0f / 20.0f);

    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(q_d)); HIP_CHECK(hipFree(k_d)); HIP_CHECK(hipFree(v_d)); HIP_CHECK(hipFree(o_d));
    HIP_CHECK(hipFree(a_d)); HIP_CHECK(hipFree(b_d)); HIP_CHECK(hipFree(Alog_d)); HIP_CHECK(hipFree(dtb_d));
    HIP_CHECK(hipFree(state_d)); HIP_CHECK(hipFree(cu_d)); HIP_CHECK(hipFree(sidx_d));
}

int main() {
    if (!ggml_cuda_r4d_available()) {
        fprintf(stderr,
            "ggml_cuda_r4d_available() returned false -- either this build was not compiled with "
            "GGML_HIP_R4D, or the current device is not gfx1201. Aborting rather than launching "
            "kernels the current device cannot run.\n");
        return 1;
    }
    int hk, hv, chunk;
    r4d_gdn_dims(&hk, &hv, &chunk);
    if (hk != K_DIM || hv != V_DIM || chunk != CHUNK) {
        fprintf(stderr, "r4d_gdn_dims() = (%d,%d,%d), test assumes (%d,%d,%d)\n",
                hk, hv, chunk, K_DIM, V_DIM, CHUNK);
        return 1;
    }

    printf("R4D Gated DeltaNet correctness/timing check\n");
    printf("geometry: head_k=%d head_v=%d chunk=%d N=%d T=100\n\n", K_DIM, V_DIM, CHUNK, N_SEQS);

    std::mt19937 rng(20260919);
    bool all_pass = true;
    run_chunk_scan_path("Hg==H identity (R=1)", /*H=*/3,  /*Hg=*/3,  /*T=*/100, rng, all_pass);
    run_chunk_scan_path("Qwen3.8-27B GQA (R=3)", /*H=*/48, /*Hg=*/16, /*T=*/100, rng, all_pass);
    run_recurrent_update_path(rng, all_pass);
    run_1024_timing(rng);

    printf("\n%s\n", all_pass ? "OVERALL: PASS" : "OVERALL: FAIL");
    return all_pass ? 0 : 1;
}
