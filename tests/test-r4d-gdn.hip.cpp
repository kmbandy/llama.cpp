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
#include <functional>
#include <utility>

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1); } } while (0)

// ============================================================================================
// bf16 <-> f32 (software RTNE, matching r4d_common.h's f32_to_bf16)
// ============================================================================================
__host__ __device__ static inline uint16_t f32_to_bf16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t rounded = u + 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}
__host__ __device__ static inline float bf16_to_f32(uint16_t h) {
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
__host__ __device__ static inline int perm_head(int h, int Hg, int R) { return R * (h % Hg) + (h / Hg); }

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

// Diagnostics for a "tiny mean, huge max" error signature (a handful of elements wrong, the rest
// right): prints the (token, head, dim) of the 5 worst elements, a per-token-bucket max-abs
// histogram (t=0..3 -- the conv_width-1=3 tokens whose conv taps read cstate, plus one past that
// boundary as a control; t=4..63 -- the rest of the first 64-token chunk; t=64.. -- everything
// after), and a per-head max-abs. `got`/`ref` are flat [N,T,heads,dim] (dim fastest, matching
// every q/k/v layout in this file). Keeps only a running top-5 (insertion into a tiny fixed array)
// rather than materializing every element's error, so this is safe to call at T=1024/H=48/dim=128.
static void print_worst_elements(const char * name, const std::vector<float> & got,
                                  const std::vector<float> & ref, int N, int T, int heads, int dim) {
    struct Worst { float err = -1.0f; int n = 0, t = 0, h = 0, d = 0; };
    Worst top[5];
    std::vector<float> per_token_max((size_t) T, 0.0f);
    std::vector<float> per_head_max((size_t) heads, 0.0f);
    for (int n = 0; n < N; ++n) {
        for (int t = 0; t < T; ++t) {
            for (int h = 0; h < heads; ++h) {
                const size_t base = (((size_t) n * T + t) * heads + h) * dim;
                for (int d = 0; d < dim; ++d) {
                    const float err = fabsf(got[base + d] - ref[base + d]);
                    if (err > per_token_max[t]) per_token_max[t] = err;
                    if (err > per_head_max[h]) per_head_max[h] = err;
                    if (err > top[4].err) {
                        top[4] = Worst{err, n, t, h, d};
                        std::sort(top, top + 5, [](const Worst & a, const Worst & b) { return a.err > b.err; });
                    }
                }
            }
        }
    }
    printf("  %s: 5 worst elements (err, n, t, h, d):\n", name);
    for (const auto & w : top) {
        if (w.err < 0.0f) continue;
        printf("    err=%.6f n=%d t=%d h=%d d=%d\n", w.err, w.n, w.t, w.h, w.d);
    }
    float bucket_head = 0.0f, bucket_early = 0.0f, bucket_rest = 0.0f; // t=0..3, t=4..63, t=64..
    for (int t = 0; t < T; ++t) {
        if (t < 4)       bucket_head  = std::max(bucket_head,  per_token_max[t]);
        else if (t < 64) bucket_early = std::max(bucket_early, per_token_max[t]);
        else             bucket_rest  = std::max(bucket_rest,  per_token_max[t]);
    }
    printf("  %s: per-token max-abs -- t=0..3: %.6f  t=4..63: %.6f  t=64..: %.6f\n",
           name, bucket_head, bucket_early, bucket_rest);
    printf("  %s: per-head max-abs:", name);
    for (int h = 0; h < heads; ++h) {
        printf(" h%d=%.4f", h, per_head_max[h]);
    }
    printf("\n");
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
// Adapter-overhead breakdown (MAD-406 op-mapping task): unlike run_1024_timing above (which times
// ONLY the raw kkt_solve+chunk_scan pair, permuting on the HOST), this exercises the actual GPU-
// side cast/permute/cumsum kernels mt_gdn_r4d.cu's production entry (ggml_cuda_gdn_r4d_prefix)
// runs, at that entry's real production shape (N=1, T=1024, H=48, Hg=16), and reports per-stage
// us/launch so the orchestrator can see where the standalone 3.05 ms/1024-token-slice measurement
// goes. Kernels are reproduced here (not linked from mt_gdn_r4d.cu, which is not built as a
// library target this test can pull in) -- the strided kernel bodies are copy-identical to that
// file's; any divergence between the two would be a bug, which is exactly what comparing this
// path's output against cpu_gdn_reference below catches.
//
// (b) the conv_w4-prepared path (r4d_gdn_conv_prep_w4_h128_bf16 feeding chunk_scan directly,
// skipping this file's own cast/permute/cumsum) is exercised separately, in
// run_conv_prep_path() further down -- not here, since it needs a/b/A_log/dt_bias inputs this
// function's harness doesn't build.
// ============================================================================================
__global__ void test_cast_qk_two_launch_kernel(
        const float * __restrict__ src, uint16_t * __restrict__ dst,
        int64_t n_seqs, int64_t P, int Hg, int K_dim, int64_t s1, int64_t s2, int64_t s3) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_seqs * P * Hg * K_dim;
    if (i >= total) return;
    const int    k   = (int) (i % K_dim);
    size_t       tmp = i / K_dim;
    const int    h   = (int) (tmp % Hg);
    tmp /= Hg;
    const int     t = (int) (tmp % P);
    const int64_t n = tmp / P;
    const float v = src[(size_t) n * s3 + (size_t) t * s2 + (size_t) h * s1 + k];
    dst[i] = f32_to_bf16(v);
}
// Combined q+k cast, ONE launch -- mirrors mt_gdn_r4d.cu's r4d_gdn_cast_qk_combined_strided_kernel
// exactly (kept as a separate copy here rather than an #include, matching this test's existing
// "reproduce, don't link the adapter" convention -- see file header).
__global__ void test_cast_qk_combined_kernel(
        const float * __restrict__ q_src, const float * __restrict__ k_src,
        uint16_t * __restrict__ q_dst, uint16_t * __restrict__ k_dst,
        int64_t n_seqs, int64_t P, int Hg, int K_dim, int64_t s1, int64_t s2, int64_t s3) {
    const size_t per = (size_t) n_seqs * P * Hg * K_dim;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * per) return;
    const bool   is_k = i >= per;
    const size_t idx  = is_k ? (i - per) : i;
    const int    k    = (int) (idx % K_dim);
    size_t       tmp  = idx / K_dim;
    const int    h    = (int) (tmp % Hg);
    tmp /= Hg;
    const int     t = (int) (tmp % P);
    const int64_t n = tmp / P;
    const float * s = is_k ? k_src : q_src;
    uint16_t *    d = is_k ? k_dst : q_dst;
    d[idx] = f32_to_bf16(s[(size_t) n * s3 + (size_t) t * s2 + (size_t) h * s1 + k]);
}
__global__ void test_cast_v_permute_kernel(
        const float * __restrict__ src, uint16_t * __restrict__ dst,
        int64_t n_seqs, int64_t P, int H, int Hg, int R, int V, int64_t s1, int64_t s2, int64_t s3) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_seqs * P * H * V;
    if (i >= total) return;
    const int    v_  = (int) (i % V);
    size_t       tmp = i / V;
    const int    h   = (int) (tmp % H);
    tmp /= H;
    const int     t  = (int) (tmp % P);
    const int64_t n  = tmp / P;
    const int hp = perm_head(h, Hg, R);
    const float val = src[(size_t) n * s3 + (size_t) t * s2 + (size_t) h * s1 + v_];
    dst[((size_t) n * P + t) * H * V + (size_t) hp * V + v_] = f32_to_bf16(val);
}
__global__ void test_gbeta_permute_cumsum_kernel(
        const float * __restrict__ g_in, const float * __restrict__ beta_in,
        float * __restrict__ g_out, float * __restrict__ beta_out,
        int64_t n_seqs, int64_t P, int H, int Hg, int R, int chunk, int64_t s1, int64_t s2, int64_t s3) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = blockIdx.y;
    if (h >= H || n >= n_seqs) return;
    const int hp = perm_head(h, Hg, R);
    const size_t base_in = (size_t) n * s3 + (size_t) h * s1;
    float * gcol_out = g_out    + ((size_t) n * P) * H + hp;
    float * bcol_out = beta_out + ((size_t) n * P) * H + hp;
    float acc = 0.0f;
    for (int64_t t = 0; t < P; ++t) {
        if (t % chunk == 0) acc = 0.0f;
        acc += g_in[base_in + (size_t) t * s2];
        gcol_out[(size_t) t * H] = acc;
        bcol_out[(size_t) t * H] = beta_in[base_in + (size_t) t * s2];
    }
}
__global__ void test_cast_o_unpermute_kernel(
        const uint16_t * __restrict__ src, float * __restrict__ dst,
        int64_t n_seqs, int64_t P, int H, int Hg, int R, int V) {
    const size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_seqs * P * H * V;
    if (j >= total) return;
    const int    v_  = (int) (j % V);
    size_t       tmp = j / V;
    const int    h   = (int) (tmp % H);
    tmp /= H;
    const int     t = (int) (tmp % P);
    const int64_t n = tmp / P;
    const int hp = perm_head(h, Hg, R);
    dst[j] = bf16_to_f32(src[((size_t) n * P + t) * H * V + (size_t) hp * V + v_]);
}

static float time_launches(hipStream_t stream, int n, const std::function<void()> & fn) {
    for (int i = 0; i < 3; ++i) fn();
    hipEvent_t ev0, ev1; HIP_CHECK(hipEventCreate(&ev0)); HIP_CHECK(hipEventCreate(&ev1));
    HIP_CHECK(hipEventRecord(ev0, stream));
    for (int i = 0; i < n; ++i) fn();
    HIP_CHECK(hipEventRecord(ev1, stream));
    HIP_CHECK(hipEventSynchronize(ev1));
    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, ev0, ev1));
    HIP_CHECK(hipEventDestroy(ev0)); HIP_CHECK(hipEventDestroy(ev1));
    return ms * 1000.0f / n;   // us/launch
}

static void run_adapter_overhead_timing(std::mt19937 & rng, bool & all_pass) {
    const int H = 48, Hg = 16, R = H / Hg, N = 1, T = 1024;
    printf("== adapter overhead breakdown: N=%d T=%d H=%d Hg=%d (production ubatch shape) ==\n", N, T, H, Hg);

    // A canonical, fully-contiguous q/k/v/g/beta layout (a standalone GDN op call, not a fused
    // qkv-projection view) is enough to exercise every stage's kernel body identically to
    // production -- ggml_cuda_gdn_r4d_prefix's strided addressing degenerates to this when the
    // strides are the product-of-dims values, which is exactly what canonical contiguity gives.
    const int64_t sq1 = K_DIM, sq2 = (int64_t) K_DIM * Hg, sq3 = sq2 * T;
    const int64_t sv1 = V_DIM, sv2 = (int64_t) V_DIM * H,  sv3 = sv2 * T;
    const int64_t sb1 = 1,     sb2 = (int64_t) H,           sb3 = sb2 * T;

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

    const std::vector<float> q_bf_f = widen_bf16(to_bf16(q_f));
    const std::vector<float> k_bf_f = widen_bf16(to_bf16(k_f));
    const std::vector<float> v_bf_f = widen_bf16(to_bf16(v_f));
    const float scale = 1.0f / sqrtf((float) V_DIM);
    std::vector<float> ref_o, ref_ht;
    cpu_gdn_reference(N, T, H, Hg, q_bf_f, k_bf_f, v_bf_f, g_raw, beta_f, h0_f, scale, ref_o, ref_ht);

    float *q_d=nullptr,*k_d=nullptr,*v_d=nullptr,*g_d=nullptr,*beta_d=nullptr,*h0_d=nullptr,*o_d=nullptr;
    uint16_t *q_bf16=nullptr,*k_bf16=nullptr,*v_bf16=nullptr,*o_bf16=nullptr;
    float *g_cumsum=nullptr,*beta_perm=nullptr,*h0_perm=nullptr,*ht_perm=nullptr;
    uint16_t *A_scratch=nullptr; int32_t *cu_d=nullptr;
    HIP_CHECK(hipMalloc(&q_d, qk_n*sizeof(float)));    HIP_CHECK(hipMalloc(&k_d, qk_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&v_d, v_n*sizeof(float)));     HIP_CHECK(hipMalloc(&g_d, gb_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&beta_d, gb_n*sizeof(float))); HIP_CHECK(hipMalloc(&h0_d, st_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&o_d, v_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&q_bf16, qk_n*sizeof(uint16_t))); HIP_CHECK(hipMalloc(&k_bf16, qk_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&v_bf16, v_n*sizeof(uint16_t)));  HIP_CHECK(hipMalloc(&o_bf16, v_n*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&g_cumsum, gb_n*sizeof(float)));  HIP_CHECK(hipMalloc(&beta_perm, gb_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&h0_perm, st_n*sizeof(float)));   HIP_CHECK(hipMalloc(&ht_perm, st_n*sizeof(float)));
    HIP_CHECK(hipMalloc(&A_scratch, (size_t) N*T*H*CHUNK*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&cu_d, (N+1)*sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(q_d, q_f.data(), qk_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_d, k_f.data(), qk_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_d, v_f.data(), v_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_d, g_raw.data(), gb_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(beta_d, beta_f.data(), gb_n*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(h0_d, h0_f.data(), st_n*sizeof(float), hipMemcpyHostToDevice));
    {
        std::vector<int32_t> cu = {0, (int32_t) T};
        HIP_CHECK(hipMemcpy(cu_d, cu.data(), (N+1)*sizeof(int32_t), hipMemcpyHostToDevice));
    }
    // h0 -> h0_perm is the same permuting fp32 copy mt_gdn_r4d.cu's kernel E does; done here with
    // a host round trip since this test does not reproduce that (unchanged, non-strided) kernel.
    HIP_CHECK(hipMemcpy(h0_perm, permute_state(h0_f, N, H, Hg, R, VK).data(), st_n*sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));
    constexpr int THREADS = 256;
    const size_t qk_elems = qk_n, v_elems = v_n;

    auto run_full_pipeline = [&](bool combined_qk) {
        if (combined_qk) {
            const size_t blocks = (2*qk_elems + THREADS - 1) / THREADS;
            test_cast_qk_combined_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(
                q_d, k_d, q_bf16, k_bf16, N, T, Hg, K_DIM, sq1, sq2, sq3);
        } else {
            const size_t blocks = (qk_elems + THREADS - 1) / THREADS;
            test_cast_qk_two_launch_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(q_d, q_bf16, N, T, Hg, K_DIM, sq1, sq2, sq3);
            test_cast_qk_two_launch_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(k_d, k_bf16, N, T, Hg, K_DIM, sq1, sq2, sq3);
        }
        {
            const size_t blocks = (v_elems + THREADS - 1) / THREADS;
            test_cast_v_permute_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(v_d, v_bf16, N, T, H, Hg, R, V_DIM, sv1, sv2, sv3);
        }
        {
            const dim3 grid((unsigned) ((H + 63) / 64), (unsigned) N);
            test_gbeta_permute_cumsum_kernel<<<grid, 64, 0, stream>>>(g_d, beta_d, g_cumsum, beta_perm, N, T, H, Hg, R, CHUNK, sb1, sb2, sb3);
        }
        const int rc1 = r4d_gdn_kkt_solve_k128_c64_bf16(k_bf16, beta_perm, g_cumsum, A_scratch, cu_d, N, T, H, Hg, K_DIM, CHUNK, stream);
        const int rc2 = r4d_gdn_chunk_scan_k128_v128_c64_bf16(q_bf16, k_bf16, v_bf16, A_scratch, g_cumsum, beta_perm,
                                                               h0_perm, o_bf16, ht_perm, cu_d, N, H, Hg, K_DIM, V_DIM, CHUNK, scale, stream);
        {
            const size_t blocks = (v_elems + THREADS - 1) / THREADS;
            test_cast_o_unpermute_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(o_bf16, o_d, N, T, H, Hg, R, V_DIM);
        }
        return std::make_pair(rc1, rc2);
    };

    // ---- correctness: combined-qk pipeline vs CPU reference ----
    const auto rc = run_full_pipeline(/*combined_qk=*/true);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (rc.first != 0 || rc.second != 0) {
        printf("  launch rejected shape (kkt_rc=%d scan_rc=%d) -- FAIL\n", rc.first, rc.second);
        all_pass = false;
    } else {
        std::vector<float> o_got(v_n);
        HIP_CHECK(hipMemcpy(o_got.data(), o_d, v_n*sizeof(float), hipMemcpyDeviceToHost));
        report("adapter path (combined qk) output", compare(o_got, ref_o), 0.08f, 0.08f, all_pass);
    }

    // ---- per-stage timing: old two-launch qk cast vs combined ----
    const float us_qk_two      = time_launches(stream, 20, [&]{
        const size_t blocks = (qk_elems + THREADS - 1) / THREADS;
        test_cast_qk_two_launch_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(q_d, q_bf16, N, T, Hg, K_DIM, sq1, sq2, sq3);
        test_cast_qk_two_launch_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(k_d, k_bf16, N, T, Hg, K_DIM, sq1, sq2, sq3);
    });
    const float us_qk_combined = time_launches(stream, 20, [&]{
        const size_t blocks = (2*qk_elems + THREADS - 1) / THREADS;
        test_cast_qk_combined_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(q_d, k_d, q_bf16, k_bf16, N, T, Hg, K_DIM, sq1, sq2, sq3);
    });
    const float us_v = time_launches(stream, 20, [&]{
        const size_t blocks = (v_elems + THREADS - 1) / THREADS;
        test_cast_v_permute_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(v_d, v_bf16, N, T, H, Hg, R, V_DIM, sv1, sv2, sv3);
    });
    const float us_gbeta = time_launches(stream, 20, [&]{
        const dim3 grid((unsigned) ((H + 63) / 64), (unsigned) N);
        test_gbeta_permute_cumsum_kernel<<<grid, 64, 0, stream>>>(g_d, beta_d, g_cumsum, beta_perm, N, T, H, Hg, R, CHUNK, sb1, sb2, sb3);
    });
    const float us_kkt = time_launches(stream, 20, [&]{
        r4d_gdn_kkt_solve_k128_c64_bf16(k_bf16, beta_perm, g_cumsum, A_scratch, cu_d, N, T, H, Hg, K_DIM, CHUNK, stream);
    });
    const float us_scan = time_launches(stream, 20, [&]{
        r4d_gdn_chunk_scan_k128_v128_c64_bf16(q_bf16, k_bf16, v_bf16, A_scratch, g_cumsum, beta_perm,
                                               h0_perm, o_bf16, ht_perm, cu_d, N, H, Hg, K_DIM, V_DIM, CHUNK, scale, stream);
    });
    const float us_out = time_launches(stream, 20, [&]{
        const size_t blocks = (v_elems + THREADS - 1) / THREADS;
        test_cast_o_unpermute_kernel<<<(unsigned) blocks, THREADS, 0, stream>>>(o_bf16, o_d, N, T, H, Hg, R, V_DIM);
    });
    const float us_full_combined = time_launches(stream, 20, [&]{ run_full_pipeline(/*combined_qk=*/true); });
    const float us_full_two      = time_launches(stream, 20, [&]{ run_full_pipeline(/*combined_qk=*/false); });

    printf("  %-38s %8.2f us/launch\n", "q/k cast, OLD (2 launches)", us_qk_two);
    printf("  %-38s %8.2f us/launch\n", "q/k cast, combined (1 launch)", us_qk_combined);
    printf("  %-38s %8.2f us/launch\n", "v cast+permute", us_v);
    printf("  %-38s %8.2f us/launch\n", "g/beta cumsum+permute", us_gbeta);
    printf("  %-38s %8.2f us/launch\n", "kkt_solve", us_kkt);
    printf("  %-38s %8.2f us/launch\n", "chunk_scan", us_scan);
    printf("  %-38s %8.2f us/launch\n", "output cast+unpermute", us_out);
    printf("  %-38s %8.2f us/launch  (%.3f ms)\n", "FULL pipeline, combined qk", us_full_combined, us_full_combined/1000.0f);
    printf("  %-38s %8.2f us/launch  (%.3f ms)\n", "FULL pipeline, old 2-launch qk", us_full_two, us_full_two/1000.0f);
    // (b) the conv_w4-prepared path (r4d_gdn_conv_prep_w4_h128_bf16 feeding chunk_scan directly,
    // skipping the cast/permute/cumsum kernels timed above) IS now exercised -- see
    // run_conv_prep_path() below, which times conv_prep itself against this same breakdown's
    // q/k+v+g/beta stages (MAD-406 follow-up: the a/b/A_log/dt_bias reachability gap this used to
    // block on is closed by ssm_conv's src[2] sidecar, mt_gdn_r4d.cu/qwen35.cpp).

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(q_d)); HIP_CHECK(hipFree(k_d)); HIP_CHECK(hipFree(v_d)); HIP_CHECK(hipFree(g_d));
    HIP_CHECK(hipFree(beta_d)); HIP_CHECK(hipFree(h0_d)); HIP_CHECK(hipFree(o_d));
    HIP_CHECK(hipFree(q_bf16)); HIP_CHECK(hipFree(k_bf16)); HIP_CHECK(hipFree(v_bf16)); HIP_CHECK(hipFree(o_bf16));
    HIP_CHECK(hipFree(g_cumsum)); HIP_CHECK(hipFree(beta_perm)); HIP_CHECK(hipFree(h0_perm)); HIP_CHECK(hipFree(ht_perm));
    HIP_CHECK(hipFree(A_scratch)); HIP_CHECK(hipFree(cu_d));
}

// ============================================================================================
// chain-218 three-stage composition (2026-09-19 follow-up): gated_delta_net.cu's use_prefill_
// chunked branch, at the EXACT shape rocprof chain 218 logged ("use_prefill_chunked P=4032
// n_tokens=4096 K=8 r4d_prefix=yes") -- r4d prefix chunk_scan over [0,P), then the [P,T0)
// remainder (56 tokens, P=4032, T0=n_tokens-K=4088), then the K=8 autoregressive snapshot tail
// over [T0,n_tokens). gated_delta_net.cu's own fix (this session) replaces the remainder's
// kernel with gated_delta_net_chunked_cuda (the short-block UT-transform kernel, ~52 us/launch
// in the chain-222 trace) whenever n_seqs==1 and the remainder fits GGML_CUDA_GDN_CHUNK_MAX (16)
// -- 56 > 16 here, so THIS specific shape still falls back to the autoregressive kernel (see
// gated_delta_net.cu's use_prefill_chunked branch comment); the composition is still checked
// end-to-end here because P/T0/remainder/K interact (state carried across all three segments)
// and because other (ubatch size, K) combinations DO land with remainder <= 16.
//
// gated_delta_net_chunked_cuda and the plain autoregressive kernel are static, non-exported
// symbols in gated_delta_net.cu -- this standalone test (built directly against libr4d, not
// ggml-hip's internals) cannot link either one. It instead uses r4d_gdn_chunk_scan_k128_v128_
// c64_bf16 as an exact stand-in for BOTH (this file's own header comment, and the split@64/
// split@50 checks above, already establish it is exact at any chunk length, including T=1) to
// verify the STATE CARRIES CORRECTLY across all three segments against the SAME whole-span CPU
// reference used elsewhere in this file, and separately times the qualitative effect the real
// fix trades on: one parallel launch over a short span vs that many serial 1-token launches
// chained by hand (which is what the autoregressive kernel's O(T) serial recurrence pays,
// structurally, even though this is not literally gated_delta_net_cuda's own launch overhead).
// ============================================================================================
static void run_chain218_composition(std::mt19937 & rng, bool & all_pass) {
    const int H = 48, Hg = 16, R = H / Hg, N = 1;
    const int n_tokens = 4096, K = 8;
    const int T0        = n_tokens - K;      // 4088
    const int P         = (T0 / 64) * 64;    // 4032
    const int remainder = T0 - P;            // 56 -- > GGML_CUDA_GDN_CHUNK_MAX(16) in THIS shape
    printf("== chain-218 composition: N=%d H=%d Hg=%d n_tokens=%d K=%d (P=%d remainder=%d tail=%d) ==\n",
           N, H, Hg, n_tokens, K, P, remainder, K);

    std::uniform_real_distribution<float> qkv_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> g_dist(-0.30f, -0.01f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);
    std::uniform_real_distribution<float> h0_dist(-0.2f, 0.2f);

    const size_t qk_n = (size_t) N * n_tokens * Hg * K_DIM;
    const size_t v_n  = (size_t) N * n_tokens * H * V_DIM;
    const size_t gb_n = (size_t) N * n_tokens * H;
    const size_t st_n = (size_t) N * H * V_DIM * K_DIM;
    const int VK = V_DIM * K_DIM;

    std::vector<float> q_f(qk_n), k_f(qk_n), v_f(v_n), g_raw(gb_n), beta_f(gb_n), h0_f(st_n);
    for (auto & x : q_f) x = qkv_dist(rng);
    for (auto & x : k_f) x = qkv_dist(rng);
    for (auto & x : v_f) x = qkv_dist(rng);
    for (auto & x : g_raw) x = g_dist(rng);
    for (auto & x : beta_f) x = beta_dist(rng);
    for (auto & x : h0_f) x = h0_dist(rng);
    l2_normalize_rows(q_f, (size_t) N * n_tokens * Hg, K_DIM);
    l2_normalize_rows(k_f, (size_t) N * n_tokens * Hg, K_DIM);

    const std::vector<float> q_bf_f = widen_bf16(to_bf16(q_f));
    const std::vector<float> k_bf_f = widen_bf16(to_bf16(k_f));
    const std::vector<float> v_bf_f = widen_bf16(to_bf16(v_f));
    const float scale = 1.0f / sqrtf((float) V_DIM);
    std::vector<float> ref_o, ref_ht;
    cpu_gdn_reference(N, n_tokens, H, Hg, q_bf_f, k_bf_f, v_bf_f, g_raw, beta_f, h0_f, scale, ref_o, ref_ht);

    const std::vector<uint16_t> q_bf = to_bf16(q_f), k_bf = to_bf16(k_f);
    const std::vector<float> v_perm    = permute_v(v_f, N, n_tokens, H, Hg, R, V_DIM);
    const std::vector<uint16_t> v_bf   = to_bf16(v_perm);
    const std::vector<float> g_perm    = permute_gbeta(g_raw, N, n_tokens, H, Hg, R);
    const std::vector<float> beta_perm = permute_gbeta(beta_f, N, n_tokens, H, Hg, R);
    const std::vector<float> h0_perm   = permute_state(h0_f, N, H, Hg, R, VK);

    auto slice_qk = [&](const std::vector<uint16_t> & buf, int t0, int len) {
        std::vector<uint16_t> out((size_t) N * len * Hg * K_DIM);
        std::memcpy(out.data(), &buf[(size_t) t0 * Hg * K_DIM], out.size() * sizeof(uint16_t));
        return out;
    };
    auto slice_v = [&](const std::vector<uint16_t> & buf, int t0, int len) {
        std::vector<uint16_t> out((size_t) N * len * H * V_DIM);
        std::memcpy(out.data(), &buf[(size_t) t0 * H * V_DIM], out.size() * sizeof(uint16_t));
        return out;
    };
    auto slice_gb = [&](const std::vector<float> & buf, int t0, int len) {
        std::vector<float> out((size_t) N * len * H);
        std::memcpy(out.data(), &buf[(size_t) t0 * H], out.size() * sizeof(float));
        return out;
    };

    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));

    // ---- stage 1: [0, P) ----
    std::vector<uint16_t> o1_bf; std::vector<float> ht1;
    const auto rc1 = run_chunk_scan_device(N, P, H, Hg, /*chunk_start_offset=*/0,
        slice_qk(q_bf,0,P), slice_qk(k_bf,0,P), slice_v(v_bf,0,P),
        slice_gb(g_perm,0,P), slice_gb(beta_perm,0,P), h0_perm, stream, o1_bf, ht1);
    // ---- stage 2: [P, T0) -- the "remainder", 56 tokens here ----
    std::vector<uint16_t> o2_bf; std::vector<float> ht2;
    const auto rc2 = run_chunk_scan_device(N, remainder, H, Hg, /*chunk_start_offset=*/P,
        slice_qk(q_bf,P,remainder), slice_qk(k_bf,P,remainder), slice_v(v_bf,P,remainder),
        slice_gb(g_perm,P,remainder), slice_gb(beta_perm,P,remainder), ht1, stream, o2_bf, ht2);
    // ---- stage 3: [T0, n_tokens) -- the K=8 snapshot tail ----
    std::vector<uint16_t> o3_bf; std::vector<float> ht3;
    const auto rc3 = run_chunk_scan_device(N, K, H, Hg, /*chunk_start_offset=*/T0,
        slice_qk(q_bf,T0,K), slice_qk(k_bf,T0,K), slice_v(v_bf,T0,K),
        slice_gb(g_perm,T0,K), slice_gb(beta_perm,T0,K), ht2, stream, o3_bf, ht3);

    if (rc1.first || rc1.second || rc2.first || rc2.second || rc3.first || rc3.second) {
        printf("  launch rejected shape (rc1=%d/%d rc2=%d/%d rc3=%d/%d) -- FAIL\n",
               rc1.first, rc1.second, rc2.first, rc2.second, rc3.first, rc3.second);
        all_pass = false;
    } else {
        std::vector<float> o_composed(v_n);
        const std::vector<float> o1 = widen_bf16(o1_bf), o2 = widen_bf16(o2_bf), o3 = widen_bf16(o3_bf);
        std::memcpy(&o_composed[0],                         o1.data(), (size_t) P * H * V_DIM * sizeof(float));
        std::memcpy(&o_composed[(size_t) P * H * V_DIM],     o2.data(), (size_t) remainder * H * V_DIM * sizeof(float));
        std::memcpy(&o_composed[(size_t) T0 * H * V_DIM],    o3.data(), (size_t) K * H * V_DIM * sizeof(float));
        const std::vector<float> o_got  = unpermute_v(o_composed, N, n_tokens, H, Hg, R, V_DIM);
        const std::vector<float> ht_got = unpermute_state(ht3, N, H, Hg, R, VK);
        report("3-stage composed output (vs whole-span CPU ref)", compare(o_got, ref_o), 0.08f, 0.08f, all_pass);
        report("3-stage composed final state (slot 0)",           compare(ht_got, ref_ht), 0.08f, 0.08f, all_pass);
    }

    // ---- timing: remainder as ONE call (T=remainder) vs `remainder` SERIAL T=1 calls chained --
    // Proxy for gated_delta_net_chunked_cuda (one launch) vs the autoregressive kernel's O(T)
    // serial recurrence (structurally: many small dependent launches) -- see header comment.
    const std::vector<uint16_t> q_rem = slice_qk(q_bf,P,remainder), k_rem = slice_qk(k_bf,P,remainder),
                                 v_rem = slice_v(v_bf,P,remainder);
    const std::vector<float> g_rem = slice_gb(g_perm,P,remainder), b_rem = slice_gb(beta_perm,P,remainder);

    const float us_one_call = time_launches(stream, 10, [&]{
        std::vector<uint16_t> tmp_o; std::vector<float> tmp_ht;
        run_chunk_scan_device(N, remainder, H, Hg, P, q_rem, k_rem, v_rem, g_rem, b_rem, ht1, stream, tmp_o, tmp_ht);
    });
    const float us_serial = time_launches(stream, 10, [&]{
        std::vector<float> state = ht1;
        for (int t = 0; t < remainder; ++t) {
            std::vector<uint16_t> tmp_o; std::vector<float> tmp_ht;
            run_chunk_scan_device(N, 1, H, Hg, P + t,
                slice_qk(q_rem,t,1), slice_qk(k_rem,t,1), slice_v(v_rem,t,1),
                slice_gb(g_rem,t,1), slice_gb(b_rem,t,1), state, stream, tmp_o, tmp_ht);
            state = tmp_ht;
        }
    });
    printf("  %-46s %9.2f us/launch\n", "remainder as ONE chunk_scan call (T=56)", us_one_call);
    printf("  %-46s %9.2f us/launch  (%d serial calls)\n", "remainder as 56 chained T=1 calls", us_serial, remainder);
    printf("  ratio (serial / one-call): %.2fx\n", us_serial / std::max(us_one_call, 1e-6f));

    HIP_CHECK(hipStreamDestroy(stream));
}

// ============================================================================================
// MAD-406 follow-up (chain 251): the NEW primed composition -- gated_delta_net.cu's
// use_prefill_chunked branch when conv_prep already primed buffers covering the WHOLE call.
// Unlike run_chain218_composition above (3 stages: r4d prefix [0,P) + a chunked/autoregressive
// remainder [P,T0) + an autoregressive K-tail [T0,n_tokens), all reading strided ggml tensors),
// this is 2 stages, BOTH reading straight out of what would be conv_prep's primed scratch (a
// flat, per-sequence-length-T0/-K buffer here, since this standalone test does not build a real
// cgraph or run conv_prep itself):
//
//   stage 1: r4d_gdn_chunk_scan_k128_v128_c64_bf16 over the FULL [0, T0) in ONE call (T0=4088,
//            NOT a multiple of 64 -- exercising exactly the "single un-split call at a non-64
//            multiple T is exact" fact this redesign relies on, already established by this
//            file's own T=100 section).
//   stage 2: r4d_gdn_recurrent_update_k128_v128_bf16_fp32state, ONE call, N=1 item, T=K=8
//            tokens, mirroring ggml_cuda_gdn_r4d_tail's exact protocol: sidx[t]=K-1-t (kernel T's
//            mapping), slot K-1 pre-seeded with stage 1's final state (the value
//            ggml_cuda_gdn_r4d_tail copies into state_d[K-1] via cudaMemcpyAsync before this
//            launch), q/k/v sliced from the SAME already-L2-normalized/permuted buffers stage 1
//            read (recurrent_update re-normalizes q/k internally -- see
//            ggml_cuda_gdn_r4d_tail's header comment on why an already-unit vector re-normalized
//            is a bounded, near-zero perturbation, not a structural error), and RAW a/b (permuted
//            to libr4d head order, NOT the already-gated g/beta stage 1 used) + A_log/dt_bias.
//
// Both stages' output/final-state are checked against cpu_gdn_reference's WHOLE-SPAN (all
// n_tokens) result -- the same reference chain-218's composition test above uses.
// ============================================================================================
static void run_primed_tail_composition(std::mt19937 & rng, bool & all_pass) {
    const int H = 48, Hg = 16, R = H / Hg, N = 1;
    const int n_tokens = 4096, K = 8;
    const int T0 = n_tokens - K; // 4088, NOT a multiple of 64
    printf("== primed tail composition (chain 251): N=%d H=%d Hg=%d n_tokens=%d K=%d T0=%d ==\n",
           N, H, Hg, n_tokens, K, T0);

    std::uniform_real_distribution<float> qkv_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> ab_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> alog_dist(-3.0f, 0.5f);
    std::uniform_real_distribution<float> dtb_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> h0_dist(-0.2f, 0.2f);

    const size_t qk_n = (size_t) N * n_tokens * Hg * K_DIM;
    const size_t v_n  = (size_t) N * n_tokens * H * V_DIM;
    const size_t gb_n = (size_t) N * n_tokens * H;
    const size_t st_n = (size_t) N * H * V_DIM * K_DIM;
    const int VK = V_DIM * K_DIM;

    std::vector<float> q_f(qk_n), k_f(qk_n), v_f(v_n), a_f(gb_n), b_f(gb_n), h0_f(st_n);
    std::vector<float> A_log(H), dt_bias(H);
    for (auto & x : q_f) x = qkv_dist(rng);
    for (auto & x : k_f) x = qkv_dist(rng);
    for (auto & x : v_f) x = qkv_dist(rng);
    for (auto & x : a_f) x = ab_dist(rng);
    for (auto & x : b_f) x = ab_dist(rng);
    for (auto & x : A_log)   x = alog_dist(rng);
    for (auto & x : dt_bias) x = dtb_dist(rng);
    for (auto & x : h0_f) x = h0_dist(rng);
    l2_normalize_rows(q_f, (size_t) N * n_tokens * Hg, K_DIM);
    l2_normalize_rows(k_f, (size_t) N * n_tokens * Hg, K_DIM);

    // g_raw/beta -- what cpu_gdn_reference and stage 1 (chunk_scan) want: g already the per-token
    // gate value (NOT cumsum'd -- cpu_gdn_reference exponentiates per token itself; chunk_scan's
    // own cu_seqlens-driven cumsum, via kkt_solve, is fed via permute_gbeta+chunk_cumsum below,
    // same as every other chunk_scan path in this file), beta = sigmoid(b).
    std::vector<float> g_raw(gb_n), beta_f(gb_n);
    const float sp_thr = 20.0f;
    for (int t = 0; t < n_tokens; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t gb = (size_t) t * H + h;
            const float alog = expf(A_log[h]);
            const float x  = a_f[gb] + dt_bias[h];
            const float sp = (x > sp_thr) ? x : ((x > 0.0f) ? x + logf(1.0f + expf(-x)) : logf(1.0f + expf(x)));
            g_raw[gb]  = -alog * sp;
            beta_f[gb] = 1.0f / (1.0f + expf(-b_f[gb]));
        }
    }

    const std::vector<float> q_bf_f = widen_bf16(to_bf16(q_f));
    const std::vector<float> k_bf_f = widen_bf16(to_bf16(k_f));
    const std::vector<float> v_bf_f = widen_bf16(to_bf16(v_f));
    const float scale = 1.0f / sqrtf((float) V_DIM);
    std::vector<float> ref_o, ref_ht;
    cpu_gdn_reference(N, n_tokens, H, Hg, q_bf_f, k_bf_f, v_bf_f, g_raw, beta_f, h0_f, scale, ref_o, ref_ht);

    const std::vector<uint16_t> q_bf = to_bf16(q_f), k_bf = to_bf16(k_f);
    const std::vector<float> v_perm    = permute_v(v_f, N, n_tokens, H, Hg, R, V_DIM);
    const std::vector<uint16_t> v_bf   = to_bf16(v_perm);
    const std::vector<float> g_perm    = permute_gbeta(g_raw, N, n_tokens, H, Hg, R);
    const std::vector<float> beta_perm = permute_gbeta(beta_f, N, n_tokens, H, Hg, R);
    const std::vector<float> h0_perm   = permute_state(h0_f, N, H, Hg, R, VK);

    auto slice_qk = [&](const std::vector<uint16_t> & buf, int t0, int len) {
        std::vector<uint16_t> out((size_t) N * len * Hg * K_DIM);
        std::memcpy(out.data(), &buf[(size_t) t0 * Hg * K_DIM], out.size() * sizeof(uint16_t));
        return out;
    };
    auto slice_v = [&](const std::vector<uint16_t> & buf, int t0, int len) {
        std::vector<uint16_t> out((size_t) N * len * H * V_DIM);
        std::memcpy(out.data(), &buf[(size_t) t0 * H * V_DIM], out.size() * sizeof(uint16_t));
        return out;
    };
    auto slice_gb = [&](const std::vector<float> & buf, int t0, int len) {
        std::vector<float> out((size_t) N * len * H);
        std::memcpy(out.data(), &buf[(size_t) t0 * H], out.size() * sizeof(float));
        return out;
    };

    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));

    // A_log/dt_bias are needed by every recurrent_update call below (stage (b)'s single step AND
    // stage (c)'s K-step tail) -- allocated once, up front, rather than per-stage.
    //
    // MAD-406 follow-up (2026-09-20): r4d_gdn_recurrent_update_k128_v128_bf16_fp32state indexes
    // A_log[hv]/dt_bias[hv] by hv (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip:115) --
    // the SAME libr4d-order head index it uses for v/output (both already permuted below), NOT
    // the GGML order A_log/dt_bias are declared in here. Uploading them unpermuted was the actual
    // root cause of this test's final-state mismatch: perm_head has only 2 fixed points out of 48
    // at this geometry (R=3), so almost every head's decay constant was paired with the WRONG
    // head's q/k/v/state -- which corrupts the state update per-head (visible as "content wrong,
    // mapping right": most elements only mildly off, a few heads badly off, depending on how far
    // A_log[h] happens to differ from A_log[perm_head(h)]) while the single-step OUTPUT (computed
    // from the same corrupted math) can still land inside a loose tolerance by chance on a given
    // random seed -- exactly the "stage (b) output PASS, state FAIL" split that was observed.
    std::vector<float> A_log_perm(H), dt_bias_perm(H);
    for (int h = 0; h < H; ++h) {
        const int hp = perm_head(h, Hg, R);
        A_log_perm[hp]   = A_log[h];
        dt_bias_perm[hp] = dt_bias[h];
    }
    float * Alog_d = nullptr, * dtb_d = nullptr;
    HIP_CHECK(hipMalloc(&Alog_d, A_log_perm.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dtb_d, dt_bias_perm.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(Alog_d, A_log_perm.data(), A_log_perm.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dtb_d, dt_bias_perm.data(), dt_bias_perm.size() * sizeof(float), hipMemcpyHostToDevice));

    // ---- stage 1: chunk_scan over the FULL [0, T0) in ONE call (T0=4088, not a 64-multiple) ---
    std::vector<uint16_t> o1_bf; std::vector<float> ht1;
    const auto rc1 = run_chunk_scan_device(N, T0, H, Hg, /*chunk_start_offset=*/0,
        slice_qk(q_bf,0,T0), slice_qk(k_bf,0,T0), slice_v(v_bf,0,T0),
        slice_gb(g_perm,0,T0), slice_gb(beta_perm,0,T0), h0_perm, stream, o1_bf, ht1);
    if (rc1.first != 0 || rc1.second != 0) {
        printf("  stage 1 (chunk_scan, T0=%d) rejected shape (rc=%d/%d) -- FAIL\n", T0, rc1.first, rc1.second);
        all_pass = false;
        HIP_CHECK(hipStreamDestroy(stream));
        return;
    }

    // ---- coordinator-requested staged breakdown (2026-09-20 follow-up): isolate WHERE the tail
    // composition's final-state mismatch actually comes from, one step at a time, instead of only
    // checking the end-to-end composed result. cpu_ref_prefix(len) is cpu_gdn_reference run over
    // just the first `len` tokens from the SAME h0_f -- since the recurrence is a pure sequential
    // fold with no dependence on how far it is eventually carried, its own returned `ht` after
    // `len` tokens is exactly the intermediate state a full n_tokens-length reference run would
    // have had at token len-1, computed independently rather than sliced out of one big run. -----
    auto cpu_ref_prefix = [&](int len, std::vector<float> & o_out, std::vector<float> & ht_out) {
        std::vector<float> q_pre(q_bf_f.begin(), q_bf_f.begin() + (size_t) len * Hg * K_DIM);
        std::vector<float> k_pre(k_bf_f.begin(), k_bf_f.begin() + (size_t) len * Hg * K_DIM);
        std::vector<float> v_pre(v_bf_f.begin(), v_bf_f.begin() + (size_t) len * H  * V_DIM);
        std::vector<float> g_pre(g_raw.begin(),   g_raw.begin()   + (size_t) len * H);
        std::vector<float> b_pre(beta_f.begin(),  beta_f.begin()  + (size_t) len * H);
        cpu_gdn_reference(N, len, H, Hg, q_pre, k_pre, v_pre, g_pre, b_pre, h0_f, scale, o_out, ht_out);
    };

    // ---- stage (a): does chunk_scan over [0,T0) ALONE already diverge from the CPU reference's
    // own state after T0 tokens? If this fails, the bug is upstream of the tail entirely (the
    // T0=4088 un-split call itself), and nothing below can be trusted regardless of its own
    // result. -----------------------------------------------------------------------------------
    std::vector<float> ref_o_T0, ref_ht_T0;
    cpu_ref_prefix(T0, ref_o_T0, ref_ht_T0);
    const std::vector<float> ht1_unperm = unpermute_state(ht1, N, H, Hg, R, VK);
    report("stage (a): chunk_scan([0,T0)) state vs CPU state-after-T0", compare(ht1_unperm, ref_ht_T0), 0.08f, 0.08f, all_pass);

    // ---- stage (b): ONE recurrent_update step (token T0) from stage (a)'s state, checked against
    // the CPU reference continued one more token. Isolates the recurrent kernel's OWN per-step
    // math (gate/l2norm/state-update) from the multi-step (T=K) launch used in stage (c) below --
    // a bug here would also show up in stage (c), but not vice versa (a bug that only appears
    // with T>1 -- e.g. in the register-resident state carry across the kernel's own token loop --
    // would pass stage (b) and fail only at stage (c)). ------------------------------------------
    {
        std::vector<float> ref_o_T0p1, ref_ht_T0p1;
        cpu_ref_prefix(T0 + 1, ref_o_T0p1, ref_ht_T0p1);
        std::vector<float> ref_o_step(H * (size_t) V_DIM);
        std::memcpy(ref_o_step.data(), &ref_o_T0p1[(size_t) T0 * H * V_DIM], ref_o_step.size() * sizeof(float));

        std::vector<float> a1_perm(H), b1_perm(H);
        for (int h = 0; h < H; ++h) {
            const int hp = perm_head(h, Hg, R);
            a1_perm[hp] = a_f[(size_t) T0 * H + h];
            b1_perm[hp] = b_f[(size_t) T0 * H + h];
        }
        const std::vector<uint16_t> q1 = slice_qk(q_bf, T0, 1), k1 = slice_qk(k_bf, T0, 1), v1 = slice_v(v_bf, T0, 1);
        // ONE state slot: sidx must still avoid index 0 (NULL_BLOCK_ID -- see stage (c)'s own
        // comment below for the full derivation), so this uses index 1 with the same base-pointer-
        // shifted-by-one-slot trick, on a buffer that is otherwise exactly one slot.
        std::vector<float> st1_host(H * (size_t) V_DIM * K_DIM);
        std::memcpy(st1_host.data(), ht1.data(), st1_host.size() * sizeof(float));
        std::vector<int32_t> sidx1_host = {1};
        std::vector<int32_t> cu1_host   = {0, 1};

        uint16_t *q1d=nullptr,*k1d=nullptr,*v1d=nullptr,*o1d=nullptr;
        float *a1d=nullptr,*b1d=nullptr,*st1d=nullptr;
        int32_t *cu1d=nullptr,*sidx1d=nullptr;
        HIP_CHECK(hipMalloc(&q1d, q1.size()*sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&k1d, k1.size()*sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&v1d, v1.size()*sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&o1d, v1.size()*sizeof(uint16_t)));
        HIP_CHECK(hipMalloc(&a1d, a1_perm.size()*sizeof(float)));
        HIP_CHECK(hipMalloc(&b1d, b1_perm.size()*sizeof(float)));
        HIP_CHECK(hipMalloc(&st1d, st1_host.size()*sizeof(float)));
        HIP_CHECK(hipMalloc(&cu1d, cu1_host.size()*sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&sidx1d, sidx1_host.size()*sizeof(int32_t)));
        HIP_CHECK(hipMemcpy(q1d, q1.data(), q1.size()*sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(k1d, k1.data(), k1.size()*sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(v1d, v1.data(), v1.size()*sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(a1d, a1_perm.data(), a1_perm.size()*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(b1d, b1_perm.data(), b1_perm.size()*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(st1d, st1_host.data(), st1_host.size()*sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cu1d, cu1_host.data(), cu1_host.size()*sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(sidx1d, sidx1_host.data(), sidx1_host.size()*sizeof(int32_t), hipMemcpyHostToDevice));

        const long st1_slot_stride = (long) H * V_DIM * K_DIM;
        const long st1_head_stride = (long) V_DIM * K_DIM;
        float * st1_base = st1d - st1_slot_stride;
        const int rc_b = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
            q1d, k1d, v1d, a1d, b1d, /*ab_stride=*/H, /*ab_is_bf16=*/0,
            Alog_d, dtb_d, st1_base, st1_slot_stride, st1_head_stride, o1d, cu1d,
            sidx1d, /*indices_stride=*/1, /*num_accepted=*/nullptr,
            /*z_gate=*/nullptr, /*norm_weight=*/nullptr, /*norm_eps=*/0.0f, /*norm_act=*/0,
            /*N=*/1, H, Hg, K_DIM, V_DIM, scale, sp_thr, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        if (rc_b != 0) {
            printf("  stage (b) (recurrent_update, single step) rejected shape (rc=%d) -- FAIL\n", rc_b);
            all_pass = false;
        } else {
            std::vector<uint16_t> o1_step_bf(v1.size());
            std::vector<float> st1_got(st1_host.size());
            HIP_CHECK(hipMemcpy(o1_step_bf.data(), o1d, o1_step_bf.size()*sizeof(uint16_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(st1_got.data(), st1d, st1_got.size()*sizeof(float), hipMemcpyDeviceToHost));
            const std::vector<float> o1_step = widen_bf16(o1_step_bf);
            const std::vector<float> o1_step_unperm = unpermute_v(o1_step, N, 1, H, Hg, R, V_DIM);
            const std::vector<float> ht_step_unperm = unpermute_state(st1_got, N, H, Hg, R, VK);

            // Coordinator-requested candidates (2026-09-20 follow-up), checked before the
            // pass/fail verdict below so they print regardless of outcome:
            {
                // (1) pre-update state: does the kernel write back the state it READ (i.e. drop
                // token T0's update entirely)? Compare against ref_ht_T0 (stage (a)'s state,
                // == the state BEFORE token T0).
                const ErrStats e1 = compare(ht_step_unperm, ref_ht_T0);
                printf("    candidate 1 (wrote pre-update / state-before-T0 instead): max_abs=%.6f max_rel=%.6f mean_abs=%.6f\n",
                       e1.max_abs, e1.max_rel, e1.mean_abs);
                // (2) transposed [K,V] vs [V,K]: swap the two S_v axes of the CPU reference and
                // compare against that instead.
                std::vector<float> ref_ht_T0p1_T(H * (size_t) V_DIM * K_DIM);
                for (int h = 0; h < H; ++h) {
                    for (int vv = 0; vv < V_DIM; ++vv) {
                        for (int kk = 0; kk < K_DIM; ++kk) {
                            ref_ht_T0p1_T[((size_t) h * V_DIM + vv) * K_DIM + kk] =
                                ref_ht_T0p1[((size_t) h * V_DIM + kk) * K_DIM + vv]; // note kk/vv swapped on the RHS
                        }
                    }
                }
                const ErrStats e2 = compare(ht_step_unperm, ref_ht_T0p1_T);
                printf("    candidate 2 (state stored [K,V] transposed): max_abs=%.6f max_rel=%.6f mean_abs=%.6f\n",
                       e2.max_abs, e2.max_rel, e2.mean_abs);
                // (3) gate mismatch: print the CPU reference's own per-token gate for (token T0,
                // head 0) alongside what the kernel's formula would compute from the SAME
                // (now-permuted, post-fix) A_log/dt_bias/a/b this call actually uploaded --
                // g_raw[] was computed with the UNPERMUTED (ggml-order) A_log/dt_bias/a/b, so a
                // real mismatch would show up as different values here.
                {
                    const float alog0 = expf(A_log[0]);
                    const float x0 = a_f[(size_t) T0 * H + 0] + dt_bias[0];
                    const float sp0 = (x0 > sp_thr) ? x0 : ((x0 > 0.0f) ? x0 + logf(1.0f+expf(-x0)) : logf(1.0f+expf(x0)));
                    const float g0_cpu = -alog0 * sp0;
                    printf("    candidate 3: CPU g[T0,head0]=%.6f (uses A_log[0]=%.6f, dt_bias[0]=%.6f, a=%.6f)\n",
                           g0_cpu, A_log[0], dt_bias[0], a_f[(size_t) T0 * H + 0]);
                }
                // (4) layout/dtype sanity: state_head_stride and the buffer are both explicit
                // S_v*S_v-fp32-per-head values here (r4d.h's "fp32state" contract, matching this
                // test's own float state_host/state_got vectors) -- print the actual stride/size
                // used so a real ~2x or dtype-halved mismatch would be visible directly.
                printf("    candidate 4: st1_slot_stride=%ld st1_head_stride=%ld sizeof(state elem)=%zu bytes\n",
                       st1_slot_stride, st1_head_stride, sizeof(float));
            }

            report("stage (b): single recurrent step output vs CPU (token T0)", compare(o1_step_unperm, ref_o_step), 0.08f, 0.08f, all_pass);
            report("stage (b): single recurrent step state vs CPU state-after-(T0+1)", compare(ht_step_unperm, ref_ht_T0p1), 0.08f, 0.08f, all_pass);
        }
        HIP_CHECK(hipFree(q1d)); HIP_CHECK(hipFree(k1d)); HIP_CHECK(hipFree(v1d)); HIP_CHECK(hipFree(o1d));
        HIP_CHECK(hipFree(a1d)); HIP_CHECK(hipFree(b1d)); HIP_CHECK(hipFree(st1d));
        HIP_CHECK(hipFree(cu1d)); HIP_CHECK(hipFree(sidx1d));
    }

    // ---- stage (c): the K-token tail, r4d_gdn_recurrent_update_k128_v128_bf16_fp32state, ONE
    // call (N=1, T=K), mirroring ggml_cuda_gdn_r4d_tail's exact protocol. ------------------------
    std::vector<float> a_perm_f((size_t) K * H), b_perm_f((size_t) K * H);
    for (int t = 0; t < K; ++t) {
        for (int h = 0; h < H; ++h) {
            const int hp = perm_head(h, Hg, R);
            a_perm_f[(size_t) t * H + hp] = a_f[(size_t) (T0 + t) * H + h];
            b_perm_f[(size_t) t * H + hp] = b_f[(size_t) (T0 + t) * H + h];
        }
    }
    const std::vector<uint16_t> q_tail = slice_qk(q_bf, T0, K);
    const std::vector<uint16_t> k_tail = slice_qk(k_bf, T0, K);
    const std::vector<uint16_t> v_tail = slice_v(v_bf, T0, K);

    // state buffer: K slots (0..K-1), channel-major [S_v,S_v,H] per slot -- slot K-1 pre-seeded
    // with stage 1's final state (ht1), matching ggml_cuda_gdn_r4d_tail's cudaMemcpyAsync.
    std::vector<float> state_host((size_t) K * H * V_DIM * K_DIM, 0.0f);
    std::memcpy(&state_host[(size_t) (K - 1) * H * V_DIM * K_DIM], ht1.data(), ht1.size() * sizeof(float));
    // sidx is 1-based (kernel T, mt_gdn_r4d.cu): r4d_gdn_recurrent_update_k128_v128_bf16_
    // fp32state treats index/slot 0 as NULL_BLOCK_ID (`if (si<=0) return;` on the read, `if (so>0)`
    // guarding every write -- read directly from the .hip kernel body) and silently DROPS a write
    // whose index is <=0. Using ggml's raw 0-based target_slot=K-1-t here would therefore never
    // write target_slot 0 (the FINAL state) -- exactly the bug this test caught (max_rel==1.0,
    // slot 0 came back all-zero). Fix: sidx[t] = K-t (values in [1,K], never 0), paired with a
    // state BASE POINTER shifted back by one slot-stride at the call site below, so index i lands
    // on physical slot (i-1) == target_slot(t) exactly.
    std::vector<int32_t> sidx_host(K);
    for (int t = 0; t < K; ++t) sidx_host[t] = K - t; // kernel T's mapping
    std::vector<int32_t> cu_host = {0, K};

    uint16_t *q_d=nullptr,*k_d=nullptr,*v_d=nullptr,*o_d=nullptr;
    float *a_d=nullptr,*b_d=nullptr,*state_d=nullptr;
    int32_t *cu_d=nullptr,*sidx_d=nullptr;
    HIP_CHECK(hipMalloc(&q_d, q_tail.size()*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&k_d, k_tail.size()*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&v_d, v_tail.size()*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&o_d, v_tail.size()*sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&a_d, a_perm_f.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&b_d, b_perm_f.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&state_d, state_host.size()*sizeof(float)));
    HIP_CHECK(hipMalloc(&cu_d, cu_host.size()*sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&sidx_d, sidx_host.size()*sizeof(int32_t)));

    HIP_CHECK(hipMemcpy(q_d, q_tail.data(), q_tail.size()*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_d, k_tail.data(), k_tail.size()*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_d, v_tail.data(), v_tail.size()*sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(a_d, a_perm_f.data(), a_perm_f.size()*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_d, b_perm_f.data(), b_perm_f.size()*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(state_d, state_host.data(), state_host.size()*sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cu_d, cu_host.data(), cu_host.size()*sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(sidx_d, sidx_host.data(), sidx_host.size()*sizeof(int32_t), hipMemcpyHostToDevice));

    const long state_slot_stride = (long) H * V_DIM * K_DIM;
    const long state_head_stride = (long) V_DIM * K_DIM;
    // See sidx_host's comment above: the kernel's own indices are 1-based, so its `state` base
    // pointer must be shifted back by one slot-stride for index i to land on physical slot (i-1).
    // Never dereferenced directly -- every read/write inside the kernel adds so*state_slot_stride
    // with so in [1,K], so the lowest address ever touched is exactly state_d.
    float * state_base = state_d - state_slot_stride;
    const int rc2 = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
        q_d, k_d, v_d, a_d, b_d, /*ab_stride=*/H, /*ab_is_bf16=*/0,
        Alog_d, dtb_d, state_base, state_slot_stride, state_head_stride, o_d, cu_d,
        sidx_d, /*indices_stride=*/K, /*num_accepted=*/nullptr,
        /*z_gate=*/nullptr, /*norm_weight=*/nullptr, /*norm_eps=*/0.0f, /*norm_act=*/0,
        /*N=*/1, H, Hg, K_DIM, V_DIM, scale, sp_thr, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    if (rc2 != 0) {
        printf("  stage 2 (recurrent_update tail, K=%d) rejected shape (rc=%d) -- FAIL\n", K, rc2);
        all_pass = false;
    } else {
        std::vector<uint16_t> o2_bf(v_tail.size());
        std::vector<float> state_got(state_host.size());
        HIP_CHECK(hipMemcpy(o2_bf.data(), o_d, o2_bf.size()*sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(state_got.data(), state_d, state_got.size()*sizeof(float), hipMemcpyDeviceToHost));

        std::vector<float> o_composed(v_n);
        const std::vector<float> o1 = widen_bf16(o1_bf), o2 = widen_bf16(o2_bf);
        std::memcpy(&o_composed[0],                  o1.data(), (size_t) T0 * H * V_DIM * sizeof(float));
        std::memcpy(&o_composed[(size_t) T0*H*V_DIM], o2.data(), (size_t) K  * H * V_DIM * sizeof(float));
        const std::vector<float> o_got = unpermute_v(o_composed, N, n_tokens, H, Hg, R, V_DIM);

        // Diagnostic (kept permanently, not just for this bug): compare ref_ht against EVERY
        // physical slot 0..K-1 of state_got and report the best match -- a wrong slot/off-by-one
        // mapping then shows up directly as "best match is slot S != 0" instead of a bare FAIL.
        {
            const size_t slot_sz = H * (size_t) V_DIM * K_DIM;
            int best_slot = -1; float best_max_abs = 1e30f;
            for (int s = 0; s < K; ++s) {
                std::vector<float> cand(slot_sz);
                std::memcpy(cand.data(), &state_got[(size_t) s * slot_sz], slot_sz * sizeof(float));
                const std::vector<float> cand_unperm = unpermute_state(cand, N, H, Hg, R, VK);
                const ErrStats e = compare(cand_unperm, ref_ht);
                printf("    slot %d vs whole-span final state: max_abs=%.6f max_rel=%.6f mean_abs=%.6f\n",
                       s, e.max_abs, e.max_rel, e.mean_abs);
                if (e.max_abs < best_max_abs) { best_max_abs = e.max_abs; best_slot = s; }
            }
            printf("  best-matching slot: %d (max_abs=%.6f) -- expected slot 0\n", best_slot, best_max_abs);
        }

        // final state = slot 0 (sidx maps t=K-1, the LAST tail token, to slot 0 -- "most recent").
        std::vector<float> ht_perm_got(H * (size_t) V_DIM * K_DIM);
        std::memcpy(ht_perm_got.data(), &state_got[0], ht_perm_got.size() * sizeof(float));
        const std::vector<float> ht_got = unpermute_state(ht_perm_got, N, H, Hg, R, VK);

        // Looser tolerance than chain-218's composition test: recurrent_update re-normalizes
        // conv_prep's already-unit-norm q/k (see this function's header comment) on top of the
        // usual bf16 + chunked-vs-serial accumulation-order drift chain-218 already tolerates at
        // 0.08 -- still a small, bounded effect, not a correctness break.
        report("primed composition output (vs whole-span CPU ref)", compare(o_got, ref_o), 0.12f, 0.12f, all_pass);
        report("primed composition final state (slot 0)",           compare(ht_got, ref_ht), 0.12f, 0.12f, all_pass);

        // Coordinator-requested check: the composed-output PASS above covers all n_tokens rows
        // (T0 stage-1 rows are the overwhelming majority, 4088 of 4096), so a per-element max_abs
        // could stay under tolerance even if the K TAIL rows specifically are off -- check them in
        // isolation, and per-token, against the CPU reference's own last-K rows.
        {
            const std::vector<float> o2_unperm = unpermute_v(o2, N, K, H, Hg, R, V_DIM);
            std::vector<float> ref_o_tail(K * H * (size_t) V_DIM);
            std::memcpy(ref_o_tail.data(), &ref_o[(size_t) T0 * H * V_DIM], ref_o_tail.size() * sizeof(float));
            report("stage (c): tail-only output (all K rows) vs CPU", compare(o2_unperm, ref_o_tail), 0.12f, 0.12f, all_pass);
            for (int t = 0; t < K; ++t) {
                std::vector<float> got_t(H * (size_t) V_DIM), ref_t(H * (size_t) V_DIM);
                std::memcpy(got_t.data(), &o2_unperm[(size_t) t * H * V_DIM], got_t.size() * sizeof(float));
                std::memcpy(ref_t.data(), &ref_o_tail[(size_t) t * H * V_DIM], ref_t.size() * sizeof(float));
                const ErrStats e = compare(got_t, ref_t);
                printf("    tail token %d (global %d) output: max_abs=%.6f max_rel=%.6f mean_abs=%.6f\n",
                       t, T0 + t, e.max_abs, e.max_rel, e.mean_abs);
            }
        }
    }

    // ---- stage (d) (2026-09-20 follow-up, state head-order bug): reproduces the ACTUAL
    // production data flow, which stage (c) above does not. ggml_cuda_gdn_r4d_prefix (mt_gdn_r4d.
    // cu) unpermutes chunk_scan's `ht` back to GGML head order before handing it out as
    // `prefill_state_out` -- that function's own header comment says so explicitly ("in the same
    // contiguous [S_v,S_v,H,n_seqs] layout ggml's own state tensors use ... unpermuted"), and it
    // is exactly ht1_unperm (stage (a) above), NOT ht1, that ggml_cuda_gdn_r4d_tail is really
    // handed as its seed. Stage (c) above fed ht1 (still libr4d order) directly, which is why it
    // passed even while production was broken: r4d_gdn_recurrent_update_k128_v128_bf16_fp32state
    // indexes its `state` argument by hv, the SAME libr4d-blocked head index it uses for
    // q/k/v/A_log/dt_bias -- so a GGML-order seed pairs almost every head's [V,K] state with the
    // wrong head's q/k/v/gate (perm_head has only 2 fixed points out of 48 at R=3), exactly the
    // bug class the A_log/dt_bias permute already fixes, but for the state tensor.
    //
    // "seed_bad" reproduces ggml_cuda_gdn_r4d_tail's PRE-FIX behavior (a flat copy of the
    // GGML-order prefill_state_out straight into the kernel's libr4d-order state slot -- no
    // permutation) and documents that it diverges, the same way split@50 documents an expected
    // divergence: reported, not gated, because failing to diverge here would mean this test
    // stopped being able to tell the two head orders apart. "seed_fixed" reproduces the FIX (this
    // task's change to mt_gdn_r4d.cu: permute the GGML-order seed into libr4d order, same
    // perm_head() relabeling as A_log/dt_bias, before it ever reaches the kernel) and gates
    // all_pass -- this is the check that FAILS on the pre-fix adapter's data flow and PASSES on
    // the fixed one.
    {
        auto run_seeded = [&](const std::vector<float> & seed_perm, const char * label,
                               bool gate_pass, bool expect_match) {
            std::vector<float> state_host_d((size_t) K * H * V_DIM * K_DIM, 0.0f);
            std::memcpy(&state_host_d[(size_t) (K - 1) * H * V_DIM * K_DIM], seed_perm.data(),
                        seed_perm.size() * sizeof(float));
            float * state_dd = nullptr;
            HIP_CHECK(hipMalloc(&state_dd, state_host_d.size() * sizeof(float)));
            HIP_CHECK(hipMemcpy(state_dd, state_host_d.data(), state_host_d.size() * sizeof(float),
                                 hipMemcpyHostToDevice));
            uint16_t * o_dd = nullptr;
            HIP_CHECK(hipMalloc(&o_dd, v_tail.size() * sizeof(uint16_t)));
            float * state_base_d = state_dd - state_slot_stride;
            const int rc = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
                q_d, k_d, v_d, a_d, b_d, /*ab_stride=*/H, /*ab_is_bf16=*/0,
                Alog_d, dtb_d, state_base_d, state_slot_stride, state_head_stride, o_dd, cu_d,
                sidx_d, /*indices_stride=*/K, /*num_accepted=*/nullptr,
                /*z_gate=*/nullptr, /*norm_weight=*/nullptr, /*norm_eps=*/0.0f, /*norm_act=*/0,
                /*N=*/1, H, Hg, K_DIM, V_DIM, scale, sp_thr, stream);
            HIP_CHECK(hipStreamSynchronize(stream));
            if (rc != 0) {
                printf("  %-42s launch rejected shape -- FAIL\n", label);
                if (gate_pass) all_pass = false;
                HIP_CHECK(hipFree(state_dd)); HIP_CHECK(hipFree(o_dd));
                return;
            }
            std::vector<float> got(state_host_d.size());
            HIP_CHECK(hipMemcpy(got.data(), state_dd, got.size() * sizeof(float), hipMemcpyDeviceToHost));
            std::vector<float> ht_perm_got(H * (size_t) V_DIM * K_DIM);
            std::memcpy(ht_perm_got.data(), &got[0], ht_perm_got.size() * sizeof(float));
            const std::vector<float> ht_got = unpermute_state(ht_perm_got, N, H, Hg, R, VK);
            const ErrStats e = compare(ht_got, ref_ht);
            if (expect_match) {
                report(label, e, 0.12f, 0.12f, all_pass);
            } else {
                const bool diverges = e.max_abs > 0.12f || e.max_rel > 0.12f;
                printf("  %-42s max_abs=%.6f max_rel=%.6f -- %s\n", label, e.max_abs, e.max_rel,
                       diverges ? "PASS (diverges, as the GGML-vs-libr4d head-order bug predicts)"
                                : "FAIL (unexpectedly matched -- test no longer distinguishes head order)");
                if (gate_pass) all_pass &= diverges;
            }
            HIP_CHECK(hipFree(state_dd)); HIP_CHECK(hipFree(o_dd));
        };

        // Pre-fix production behavior: prefill_state_out (GGML order) copied straight in, no
        // permutation -- documents the bug (non-gating, like split@50 above).
        run_seeded(ht1_unperm, "stage (d): seed=GGML-order (pre-fix bug)", /*gate_pass=*/false,
                   /*expect_match=*/false);
        // Fixed behavior: prefill_state_out re-permuted to libr4d order before seeding (mirrors
        // this task's mt_gdn_r4d.cu fix -- r4d_gdn_state_permute_kernel, src_is_ggml_order=true).
        // This is the check that fails against the OLD adapter's data flow and passes against the
        // new one -- it gates all_pass.
        const std::vector<float> ht1_reperm = permute_state(ht1_unperm, N, H, Hg, R, VK);
        run_seeded(ht1_reperm, "stage (d): seed=libr4d-order (fixed)", /*gate_pass=*/true,
                   /*expect_match=*/true);
    }

    HIP_CHECK(hipFree(q_d)); HIP_CHECK(hipFree(k_d)); HIP_CHECK(hipFree(v_d)); HIP_CHECK(hipFree(o_d));
    HIP_CHECK(hipFree(a_d)); HIP_CHECK(hipFree(b_d));
    HIP_CHECK(hipFree(state_d)); HIP_CHECK(hipFree(cu_d)); HIP_CHECK(hipFree(sidx_d));
    HIP_CHECK(hipFree(Alog_d)); HIP_CHECK(hipFree(dtb_d));
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

// ============================================================================================
// MAD-406 (R4D GDN conv_prep, task 5): r4d_gdn_conv_prep_w4_h128_bf16 (the causal conv + SiLU +
// q/k/v split + q/k L2-norm + gate prep, ONE kernel -- see r4d_gdn_conv_w4_h128_bf16.hip's own
// header comment) driving kkt_solve + chunk_scan directly, at production geometry (N=1, T=1024,
// H=48, Hg=16, K=V=128, conv width 4 -- Qwen3.8-27B's actual linear-attention shape, matching
// run_adapter_overhead_timing's geometry above), checked against a from-scratch CPU reference of
// (causal conv + SiLU + split + q/k L2-norm + gate math + delta rule) and timed against this
// file's own reproduction of mt_gdn_r4d.cu's cast/permute/cumsum path (test_cast_qk_combined_
// kernel / test_cast_v_permute_kernel / test_gbeta_permute_cumsum_kernel above), which is what a
// real GDN call pays when ggml_cuda_try_gdn_conv_prep_fusion (mt_gdn_r4d.cu) declines.
//
// conv_prep's own inputs, mirrored here exactly (see r4d_gdn_conv_w4_h128_bf16.hip and this
// test's ggml_cuda_try_gdn_conv_prep_fusion counterpart in mt_gdn_r4d.cu for the ggml-side
// derivation of each):
//   x        [T, C]      bf16, channel-fastest, THIS call's new tokens only
//   cstate   [N, C, 3]   bf16, channel-major, tok-fastest -- the causal history (task 2: in
//                        production this is always freshly staged from ggml's own SSM_CONV
//                        src[0] history rows; here it is simply the 3 rows immediately preceding
//                        x in a synthetic longer sequence, exercised with has_init=true so the
//                        kernel actually reads it -- the same call shape production makes)
//   wgt      [C, 4]      bf16, channel-major, tap-fastest (matches ggml's own conv1d weight layout)
//   a, b     [T, H]      fp32, raw pre-activation alpha/beta (ab_is_bf16=0)
//   A_log, dt_bias [H]   fp32, per-head
// producing q[T,Hg,K]/k[T,Hg,K]/v[T,H,V] bf16 and g[T,H] (already chunk-cumsum'd)/beta[T,H] fp32,
// all in GGML's own head order -- which this test then permutes by hand (permute_v/permute_gbeta,
// same as every other path in this file) before kkt_solve/chunk_scan, exactly what
// ggml_cuda_try_gdn_conv_prep_fusion does in production.
// ============================================================================================
static void run_conv_prep_path(std::mt19937 & rng, bool & all_pass) {
    const int H = 48, Hg = 16, R = H / Hg, N = 1, T = 1024, W = 3; // conv width 4 -> W = 3 history rows
    const int64_t C = (int64_t) 2 * Hg * K_DIM + (int64_t) H * V_DIM; // 10240, the real production width
    printf("== conv_prep-driven path: N=%d T=%d H=%d Hg=%d C=%lld (production geometry) ==\n",
           N, T, H, Hg, (long long) C);

    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> w_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> ab_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> alog_dist(-3.0f, 0.5f);   // A_log itself, any real value
    std::uniform_real_distribution<float> dtb_dist(-1.0f, 1.0f);

    // x_stage, host side: [N][T][C], channel-fastest -- exactly r4d_gdn_conv_prep_w4_h128_bf16's
    // own `x` layout (xpitch=C), so it needs no transpose before upload.
    //
    // hist_f (the cstate seed) is laid out [N][C][W], tok-FASTEST (W stride 1, channel stride W)
    // -- NOT [N][W][C] -- because that is what cs_seq=(long)C*W / cs_dim=W / cs_tok=1 (passed to
    // r4d_gdn_conv_prep_w4_h128_bf16 below) actually address: element (n, channel, i) sits at
    // n*cs_seq + channel*cs_dim + i*cs_tok = n*C*W + channel*W + i. This mirrors
    // r4d_gdn_conv_prep_stage_kernel's own write in mt_gdn_r4d.cu exactly
    // (`cstate[((size_t) n * C + c) * W + i] = ...`) -- getting this transposed (as an earlier
    // version of this test did, generating hist_f as [N][W][C] while still passing cs_dim=W/
    // cs_tok=1) reads a channel-permuted history for every one of the first CONV_WIDTH-1 tokens
    // of the sequence: q/k/v come out right everywhere EXCEPT tokens 0..2, which is exactly the
    // "tiny mean, huge max" error signature this bug produced (see run_conv_prep_path's per-token
    // histogram below).
    std::vector<float> hist_f((size_t) N * C * W), x_f((size_t) N * T * C);
    for (auto & v : hist_f) v = x_dist(rng);
    for (auto & v : x_f)    v = x_dist(rng);

    std::vector<float> wgt_f((size_t) C * 4);
    for (auto & v : wgt_f) v = w_dist(rng);

    std::vector<float> a_f((size_t) N * T * H), b_f((size_t) N * T * H);
    for (auto & v : a_f) v = ab_dist(rng);
    for (auto & v : b_f) v = ab_dist(rng);
    std::vector<float> A_log_f(H), dt_bias_f(H);
    for (auto & v : A_log_f)   v = alog_dist(rng);
    for (auto & v : dt_bias_f) v = dtb_dist(rng);

    // ── CPU reference: causal conv (width 4, using hist_f as the 3-row history) + SiLU, rounded
    // to bf16 exactly where the kernel rounds (r4d_gdn_conv_w4_h128_bf16.hip: "rounded to bf16
    // BEFORE the norm"), then split, q/k L2-norm (eps 1e-6, on the ROUNDED values, matching the
    // kernel), gate g_raw[t,h] = -exp(A_log[h])*softplus(a+dt_bias[h]) (NOT cumsum'd -- fed to
    // cpu_gdn_reference below, which exponentiates per token itself), beta = sigmoid(b). ────────
    const int64_t qk_c = (int64_t) Hg * K_DIM; // channel range width for q and for k
    std::vector<float> q_ref((size_t) N * T * Hg * K_DIM), k_ref((size_t) N * T * Hg * K_DIM);
    std::vector<float> v_ref((size_t) N * T * H * V_DIM);
    std::vector<float> g_raw_ref((size_t) N * T * H), beta_ref((size_t) N * T * H);

    auto conv_col = [&](int n, int t, int64_t c) -> float {
        // causal 4-tap conv at (n, t, c): taps are x[t-3..t] (t indexed into the W+T history+new
        // stream), matching r4d_gdn_conv_w4_h128_bf16.hip's `win[i] holds x[t-(W-1)+i]`.
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            const int src = t - 3 + i; // position in the T-length new-token stream
            float xv;
            if (src >= 0) {
                xv = x_f[((size_t) n * T + src) * C + c];
            } else {
                const int hsrc = W + src; // position in the W-length history
                xv = (hsrc >= 0) ? hist_f[((size_t) n * C + c) * W + hsrc] : 0.0f;
            }
            acc += wgt_f[(size_t) c * 4 + i] * xv;
        }
        const float silu = acc / (1.0f + expf(-acc));
        return bf16_to_f32(f32_to_bf16(silu)); // match the kernel's bf16 rounding point exactly
    };
    for (int t = 0; t < T; ++t) {
        // q heads: channels [0, qk_c)
        for (int hg = 0; hg < Hg; ++hg) {
            float row[K_DIM];
            float ss = 0.0f;
            for (int d = 0; d < K_DIM; ++d) {
                row[d] = conv_col(0, t, (int64_t) hg * K_DIM + d);
                ss += row[d] * row[d];
            }
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            for (int d = 0; d < K_DIM; ++d) {
                const float normed = bf16_to_f32(f32_to_bf16(row[d] * inv));
                q_ref[((size_t) t * Hg + hg) * K_DIM + d] = normed;
            }
        }
        // k heads: channels [qk_c, 2*qk_c)
        for (int hg = 0; hg < Hg; ++hg) {
            float row[K_DIM];
            float ss = 0.0f;
            for (int d = 0; d < K_DIM; ++d) {
                row[d] = conv_col(0, t, qk_c + (int64_t) hg * K_DIM + d);
                ss += row[d] * row[d];
            }
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            for (int d = 0; d < K_DIM; ++d) {
                const float normed = bf16_to_f32(f32_to_bf16(row[d] * inv));
                k_ref[((size_t) t * Hg + hg) * K_DIM + d] = normed;
            }
        }
        // v heads: channels [2*qk_c, 2*qk_c + H*V_DIM) -- no norm.
        for (int h = 0; h < H; ++h) {
            for (int d = 0; d < V_DIM; ++d) {
                v_ref[((size_t) t * H + h) * V_DIM + d] = conv_col(0, t, 2 * qk_c + (int64_t) h * V_DIM + d);
            }
        }
        // gate/beta: independent of the conv, straight off a/b/A_log/dt_bias.
        for (int h = 0; h < H; ++h) {
            const float alog = expf(A_log_f[h]);
            float x = a_f[(size_t) t * H + h] + dt_bias_f[h];
            const float sp = (x > 20.0f) ? x : ((x > 0.0f) ? x + logf(1.0f + expf(-x)) : logf(1.0f + expf(x)));
            g_raw_ref[(size_t) t * H + h]  = -alog * sp;
            beta_ref[(size_t) t * H + h]   = 1.0f / (1.0f + expf(-b_f[(size_t) t * H + h]));
        }
    }
    std::vector<float> h0_ref((size_t) N * H * V_DIM * K_DIM, 0.0f); // fresh state -- this test is
                                                                      // about conv_prep, not the
                                                                      // SSM state carry (already
                                                                      // covered by run_chunk_scan_path)
    std::vector<float> o_ref, ht_ref_unused;
    const float scale = 1.0f / sqrtf((float) V_DIM);
    cpu_gdn_reference(N, T, H, Hg, q_ref, k_ref, v_ref, g_raw_ref, beta_ref, h0_ref, scale, o_ref, ht_ref_unused);

    // ── GPU path: conv_prep -> permute v/g/beta into libr4d order -> kkt_solve -> chunk_scan. ───
    hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));

    const std::vector<uint16_t> hist_bf = to_bf16(hist_f);
    const std::vector<uint16_t> x_bf    = to_bf16(x_f);
    const std::vector<uint16_t> wgt_bf  = to_bf16(wgt_f);
    std::vector<int32_t> cache_idx(N);
    for (int i = 0; i < N; ++i) cache_idx[i] = i;
    std::vector<uint8_t> has_init(N, 1);
    std::vector<int32_t> cu(N + 1);
    for (int i = 0; i <= N; ++i) cu[i] = i * T;

    uint16_t *x_d, *cstate_d, *wgt_d, *q_d, *k_d, *v_d;
    float *a_d, *b_d, *Alog_d, *dtb_d, *g_d, *beta_d;
    int32_t *cache_idx_d, *cu_d;
    uint8_t *has_init_d;
    HIP_CHECK(hipMalloc(&x_d, x_bf.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&cstate_d, hist_bf.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&wgt_d, wgt_bf.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&a_d, a_f.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&b_d, b_f.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&Alog_d, A_log_f.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dtb_d, dt_bias_f.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&cache_idx_d, cache_idx.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&has_init_d, has_init.size() * sizeof(uint8_t)));
    HIP_CHECK(hipMalloc(&cu_d, cu.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&q_d, (size_t) N * T * Hg * K_DIM * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&k_d, (size_t) N * T * Hg * K_DIM * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&v_d, (size_t) N * T * H * V_DIM * sizeof(uint16_t)));
    HIP_CHECK(hipMalloc(&g_d, (size_t) N * T * H * sizeof(float)));
    HIP_CHECK(hipMalloc(&beta_d, (size_t) N * T * H * sizeof(float)));

    HIP_CHECK(hipMemcpy(x_d, x_bf.data(), x_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cstate_d, hist_bf.data(), hist_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(wgt_d, wgt_bf.data(), wgt_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(a_d, a_f.data(), a_f.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(b_d, b_f.data(), b_f.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(Alog_d, A_log_f.data(), A_log_f.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dtb_d, dt_bias_f.data(), dt_bias_f.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cache_idx_d, cache_idx.data(), cache_idx.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(has_init_d, has_init.data(), has_init.size() * sizeof(uint8_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(cu_d, cu.data(), cu.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    const int cp_rc = r4d_gdn_conv_prep_w4_h128_bf16(
        x_d, /*xpitch=*/C, wgt_d, /*bias=*/nullptr, cstate_d,
        /*cs_seq=*/(long) C * W, /*cs_dim=*/W, /*cs_tok=*/1,
        cache_idx_d, /*ci_stride=*/1, has_init_d,
        a_d, b_d, /*ab_stride=*/H, /*ab_is_bf16=*/0,
        Alog_d, dtb_d, q_d, k_d, v_d, g_d, beta_d, cu_d,
        N, T, H, Hg, K_DIM, V_DIM, /*width=*/4, /*softplus_thr=*/20.0f, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (cp_rc != 0) {
        printf("  r4d_gdn_conv_prep_w4_h128_bf16 rejected this shape (rc=%d) -- FAIL\n", cp_rc);
        all_pass = false;
    } else {
        std::vector<uint16_t> q_ggml_bf((size_t) N * T * Hg * K_DIM), k_ggml_bf((size_t) N * T * Hg * K_DIM);
        std::vector<uint16_t> v_ggml_bf((size_t) N * T * H * V_DIM);
        std::vector<float> g_ggml((size_t) N * T * H), beta_ggml((size_t) N * T * H);
        HIP_CHECK(hipMemcpy(q_ggml_bf.data(), q_d, q_ggml_bf.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(k_ggml_bf.data(), k_d, k_ggml_bf.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(v_ggml_bf.data(), v_d, v_ggml_bf.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(g_ggml.data(), g_d, g_ggml.size() * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(beta_ggml.data(), beta_d, beta_ggml.size() * sizeof(float), hipMemcpyDeviceToHost));

        // q/k: no permutation ever needed (indexed by Hg) -- compare straight against the CPU
        // reference's own bf16-rounded q/k. Diagnostics (worst elements / per-token / per-head
        // breakdown) always printed, not just on failure -- a "tiny mean, huge max" signature is
        // exactly the case a plain PASS/FAIL line hides.
        const std::vector<float> q_got = widen_bf16(q_ggml_bf);
        const std::vector<float> k_got = widen_bf16(k_ggml_bf);
        report("conv_prep q", compare(q_got, q_ref), 1e-2f, 5e-2f, all_pass);
        print_worst_elements("conv_prep q", q_got, q_ref, N, T, Hg, K_DIM);
        report("conv_prep k", compare(k_got, k_ref), 1e-2f, 5e-2f, all_pass);
        print_worst_elements("conv_prep k", k_got, k_ref, N, T, Hg, K_DIM);
        // v: conv_prep's output is in GGML head order (same as v_ref) -- straight comparison, no
        // permutation needed here (that happens below, before kkt_solve/chunk_scan).
        const std::vector<float> v_got = widen_bf16(v_ggml_bf);
        report("conv_prep v", compare(v_got, v_ref), 1e-2f, 5e-2f, all_pass);
        print_worst_elements("conv_prep v", v_got, v_ref, N, T, H, V_DIM);
        // beta: no cumsum, straight compare against beta_ref (both GGML order).
        report("conv_prep beta", compare(beta_ggml, beta_ref), 1e-4f, 1e-3f, all_pass);
        // g: conv_prep's output is ALREADY the 64-token chunk-cumsum of g_raw_ref -- cumsum the
        // CPU reference the same way before comparing.
        const std::vector<float> g_ref_cumsum = chunk_cumsum(g_raw_ref, N, T, H, /*chunk_start_offset=*/0);
        report("conv_prep g (cumsum)", compare(g_ggml, g_ref_cumsum), 5e-3f, 1e-2f, all_pass);

        // ── feed conv_prep's own output through kkt_solve + chunk_scan (permuted into libr4d
        // order, exactly like ggml_cuda_try_gdn_conv_prep_fusion), and check the FINAL output
        // against o_ref -- the end-to-end check this task asked for. ─────────────────────────────
        const std::vector<float> v_perm    = permute_v(v_got, N, T, H, Hg, R, V_DIM);
        const std::vector<float> g_perm    = permute_gbeta(g_ggml, N, T, H, Hg, R);   // already cumsum'd
        const std::vector<float> beta_perm = permute_gbeta(beta_ggml, N, T, H, Hg, R);
        const std::vector<uint16_t> v_perm_bf = to_bf16(v_perm);
        std::vector<float> h0_perm(h0_ref.size(), 0.0f); // R=3 permutation of an all-zero state is itself

        ChunkScanDeviceBufs cb = alloc_chunk_scan_bufs(N, T, H, Hg);
        HIP_CHECK(hipMemcpy(cb.q_d, q_ggml_bf.data(), q_ggml_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cb.k_d, k_ggml_bf.data(), k_ggml_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cb.v_d, v_perm_bf.data(), v_perm_bf.size() * sizeof(uint16_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cb.g_d, g_perm.data(), g_perm.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cb.beta_d, beta_perm.data(), beta_perm.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cb.h0_d, h0_perm.data(), h0_perm.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(cb.cu_d, cu.data(), cu.size() * sizeof(int32_t), hipMemcpyHostToDevice));

        const int kkt_rc = r4d_gdn_kkt_solve_k128_c64_bf16(cb.k_d, cb.beta_d, cb.g_d, cb.A_d, cb.cu_d,
                                                            N, T, H, Hg, K_DIM, CHUNK, stream);
        const int scan_rc = r4d_gdn_chunk_scan_k128_v128_c64_bf16(cb.q_d, cb.k_d, cb.v_d, cb.A_d, cb.g_d,
                                                                   cb.beta_d, cb.h0_d, cb.o_d, cb.ht_d, cb.cu_d,
                                                                   N, H, Hg, K_DIM, V_DIM, CHUNK, scale, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
        if (kkt_rc != 0 || scan_rc != 0) {
            printf("  kkt_solve/chunk_scan rejected the conv_prep output shape (rc=%d,%d) -- FAIL\n", kkt_rc, scan_rc);
            all_pass = false;
        } else {
            std::vector<uint16_t> o_perm_bf((size_t) N * T * H * V_DIM);
            HIP_CHECK(hipMemcpy(o_perm_bf.data(), cb.o_d, o_perm_bf.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
            const std::vector<float> o_got = unpermute_v(widen_bf16(o_perm_bf), N, T, H, Hg, R, V_DIM);
            report("conv_prep end-to-end o", compare(o_got, o_ref), 5e-2f, 1e-1f, all_pass);
        }
        free_chunk_scan_bufs(cb);
    }

    // ── timing: conv_prep alone vs. this file's own reproduction of the cast/permute/cumsum path
    // conv_prep replaces (test_cast_qk_combined_kernel + test_cast_v_permute_kernel +
    // test_gbeta_permute_cumsum_kernel, all launched from CONTIGUOUS f32 sources the way
    // ggml_cuda_op_gated_delta_net_r4d's non-strided entry would see them, i.e. the closest
    // apples-to-apples comparison this standalone test can make without linking ggml itself). ───
    float *q_src_d, *k_src_d, *v_src_d, *g_src_d, *beta_src_d;
    HIP_CHECK(hipMalloc(&q_src_d, q_ref.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&k_src_d, k_ref.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&v_src_d, v_ref.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&g_src_d, g_raw_ref.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&beta_src_d, beta_ref.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(q_src_d, q_ref.data(), q_ref.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k_src_d, k_ref.data(), k_ref.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v_src_d, v_ref.data(), v_ref.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g_src_d, g_raw_ref.data(), g_raw_ref.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(beta_src_d, beta_ref.data(), beta_ref.size() * sizeof(float), hipMemcpyHostToDevice));

    const float us_conv_prep = time_launches(stream, 20, [&]() {
        r4d_gdn_conv_prep_w4_h128_bf16(x_d, C, wgt_d, nullptr, cstate_d, (long) C * W, W, 1,
                                       cache_idx_d, 1, has_init_d, a_d, b_d, H, 0, Alog_d, dtb_d,
                                       q_d, k_d, v_d, g_d, beta_d, cu_d, N, T, H, Hg, K_DIM, V_DIM,
                                       4, 20.0f, stream);
    });
    const float us_cast_path = time_launches(stream, 20, [&]() {
        constexpr int TH = 256;
        const size_t n_qk = (size_t) N * T * Hg * K_DIM;
        test_cast_qk_combined_kernel<<<(unsigned) ((2 * n_qk + TH - 1) / TH), TH, 0, stream>>>(
            q_src_d, k_src_d, q_d, k_d, N, T, Hg, K_DIM, K_DIM, (int64_t) Hg * K_DIM, (int64_t) T * Hg * K_DIM);
        const size_t n_v = (size_t) N * T * H * V_DIM;
        test_cast_v_permute_kernel<<<(unsigned) ((n_v + TH - 1) / TH), TH, 0, stream>>>(
            v_src_d, v_d, N, T, H, Hg, R, V_DIM, V_DIM, (int64_t) H * V_DIM, (int64_t) T * H * V_DIM);
        const dim3 grid((unsigned) ((H + 63) / 64), (unsigned) N);
        test_gbeta_permute_cumsum_kernel<<<grid, 64, 0, stream>>>(
            g_src_d, beta_src_d, g_d, beta_d, N, T, H, Hg, R, CHUNK, 1, (int64_t) H, (int64_t) T * H);
    });
    printf("  %.2f us/launch (conv_prep, one kernel: conv+silu+split+l2norm+gate+cumsum)\n", us_conv_prep);
    printf("  %.2f us/launch (cast/permute/cumsum path conv_prep replaces upstream of it: "
           "q+k cast + v permute-cast + g/beta permute-cumsum, 3 launches; does NOT include the "
           "conv/silu/split/l2norm/gate-prep work itself, which in the non-conv_prep path runs as "
           "separate ggml ops entirely outside this adapter)\n", us_cast_path);

    HIP_CHECK(hipFree(x_d)); HIP_CHECK(hipFree(cstate_d)); HIP_CHECK(hipFree(wgt_d));
    HIP_CHECK(hipFree(a_d)); HIP_CHECK(hipFree(b_d)); HIP_CHECK(hipFree(Alog_d)); HIP_CHECK(hipFree(dtb_d));
    HIP_CHECK(hipFree(cache_idx_d)); HIP_CHECK(hipFree(has_init_d)); HIP_CHECK(hipFree(cu_d));
    HIP_CHECK(hipFree(q_d)); HIP_CHECK(hipFree(k_d)); HIP_CHECK(hipFree(v_d));
    HIP_CHECK(hipFree(g_d)); HIP_CHECK(hipFree(beta_d));
    HIP_CHECK(hipFree(q_src_d)); HIP_CHECK(hipFree(k_src_d)); HIP_CHECK(hipFree(v_src_d));
    HIP_CHECK(hipFree(g_src_d)); HIP_CHECK(hipFree(beta_src_d));
    HIP_CHECK(hipStreamDestroy(stream));
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
    run_adapter_overhead_timing(rng, all_pass);
    run_chain218_composition(rng, all_pass);
    run_primed_tail_composition(rng, all_pass);
    run_conv_prep_path(rng, all_pass);

    printf("\n%s\n", all_pass ? "OVERALL: PASS" : "OVERALL: FAIL");
    return all_pass ? 0 : 1;
}
