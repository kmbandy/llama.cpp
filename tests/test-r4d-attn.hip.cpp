// Standalone correctness + timing check for the vendored R4D fp8-KV / bf16-KV paged attention
// entry points (ggml/src/ggml-cuda/r4d/, see r4d/README-ORIGIN.txt for provenance -- upstream
// radiance-libr4d commit b9e42ab-rx6). Exercises:
//
//   r4d_attn_prefill_h256_gqa6_fp8kv / _bf16kv
//   r4d_attn_decode_h256_gqa6_fp8kv  / _bf16kv   (q_len=1 and q_len=8, the speculative-verify band)
//   r4d_attn_decode_h256_gqa6_scratch_bytes
//
// against a straightforward fp32 CPU reference: causal softmax attention, KV dequantized with
// descale 1.0 (e4m3 dequant via the OCP e4m3fn table below for the fp8kv variants; a plain bf16
// widen for the bf16kv variants), scale = 1/sqrt(256).
//
// Geometry (fixed by the vendored kernels): head_dim=256, gqa=6, paged block_size=16.
// Test config: kv_heads=4 (so q_heads=24), num_seqs=3, contexts {37, 512, 2000} (the first is
// deliberately shorter than the prefill q_len below, so some of that seq's queries have no causal
// keys at all and are expected to come back exactly zero -- a degenerate but well-defined case
// both the kernel and the CPU reference handle identically). Prefill q_len=64 (queries are the
// LAST 64 tokens of each context, per r4d's causal convention: klimit = ctx - q_len + qpos).
//
// This file is deliberately NOT wired into tests/CMakeLists.txt, matching the existing convention
// for every other test in this tree that calls the raw HIP runtime directly against ggml-hip's
// internal (non-public-API) symbols -- see tests/test-ar-codec-q8-wide-grid.hip.cpp,
// tests/test_aiter_uattn_verify_bench.cpp, tests/test_aiter_turbo_fp8_perf.cpp, etc., none of
// which appear in tests/CMakeLists.txt either. Those link -lggml-hip directly from the build
// output dir; this test does the same, since r4d_attn_* are extern "C" symbols compiled into
// libggml-hip.so (via ggml/src/ggml-hip/CMakeLists.txt's GGML_HIP_R4D block) but are not part of
// ggml's public C API.
//
// Build (requires a build-hip configured with GGML_HIP_R4D, default ON since AMDGPU_TARGETS
// already contains gfx1201 in this tree's build-hip):
//
//   hipcc --offload-arch=gfx1201 -O2 -std=c++17 \
//       -I ggml/src/ggml-cuda -DGGML_HIP_R4D \
//       tests/test-r4d-attn.hip.cpp \
//       -L build-hip/bin -lggml-hip -Wl,-rpath,$(pwd)/build-hip/bin \
//       -o /tmp/test-r4d-attn
//   /tmp/test-r4d-attn
//
// Expected output: one PASS/FAIL line per kernel variant with max-abs / max-rel error against the
// CPU reference, then one µs/launch line per variant (20 launches, hipEvent-timed).

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

// ============================================================================================
// OCP e4m3fn (1 sign, 4 exp bias-7, 3 mantissa; S.1111.111 = NaN; max finite 448). This is what
// gfx1201's __builtin_amdgcn_cvt_pk_f32_fp8 / cvt_pk_fp8_f32 implement, and what the vendored
// kernels' KV dequant path relies on (r4d_dt16.h's fp8x8_to_16x4w).
// ============================================================================================
static float e4m3_decode(uint8_t byte) {
    const int sign = (byte >> 7) & 1;
    const int exp  = (byte >> 3) & 0xF;
    const int mant = byte & 0x7;
    float val;
    if (exp == 0) {
        val = ldexpf((float)mant, -9);              // subnormal: mant/8 * 2^-6
    } else if (exp == 15 && mant == 7) {
        val = std::numeric_limits<float>::quiet_NaN();
    } else {
        val = ldexpf(1.0f + mant / 8.0f, exp - 7);
    }
    return sign ? -val : val;
}
static float g_e4m3_pos_table[127];   // decode(byte) for byte in [0,126], ascending
static void e4m3_init_table() {
    for (int b = 0; b <= 126; ++b) g_e4m3_pos_table[b] = e4m3_decode((uint8_t)b);
}
static uint8_t e4m3_encode(float x) {
    const uint8_t sign = x < 0.0f ? 0x80 : 0x00;
    const float ax = fabsf(x);
    int lo = 0, hi = 126;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (g_e4m3_pos_table[mid] < ax) lo = mid + 1; else hi = mid;
    }
    int best = lo; float bestd = fabsf(g_e4m3_pos_table[lo] - ax);
    if (lo > 0 && fabsf(g_e4m3_pos_table[lo - 1] - ax) < bestd) best = lo - 1;
    return sign | (uint8_t)best;
}

// ============================================================================================
// Geometry
// ============================================================================================
static constexpr int HEAD_DIM  = 256;
static constexpr int GQA        = 6;
static constexpr int BLOCK_SIZE = 16;
static constexpr int KV_HEADS   = 4;
static constexpr int Q_HEADS    = KV_HEADS * GQA;   // 24
static constexpr int NUM_SEQS   = 3;
static const int CTX[NUM_SEQS] = {37, 512, 2000};

// ============================================================================================
// CPU reference: causal softmax attention over a paged KV cache already dequantized to fp32.
// KVdeq layout: [num_blocks, kv_heads, block_size, 2, head_dim] (K at [...,0,:], V at [...,1,:]),
// matching R4D's "K then V per slot" convention (r4d.h's R4DArgs.kv doc comment).
// ============================================================================================
static void cpu_attention_ref(
    const std::vector<float> & Qf,          // [num_seqs*q_len, q_heads, head_dim]
    const std::vector<float> & KVdeq,       // [num_blocks, kv_heads, block_size, 2, head_dim]
    const std::vector<int>   & block_table, // [num_seqs, max_blocks]
    int max_blocks, int q_len, float scale,
    std::vector<float> & out)               // [num_seqs*q_len, q_heads, head_dim]
{
    out.assign((size_t)NUM_SEQS * q_len * Q_HEADS * HEAD_DIM, 0.0f);
    std::vector<float> scores;
    for (int seq = 0; seq < NUM_SEQS; ++seq) {
        const int ctx = CTX[seq];
        for (int qpos = 0; qpos < q_len; ++qpos) {
            const int klimit = ctx - q_len + qpos;   // inclusive index of the last visible key
            const int nkeys  = std::min(klimit + 1, ctx);
            for (int kvh = 0; kvh < KV_HEADS; ++kvh) {
                for (int hi = 0; hi < GQA; ++hi) {
                    const int qhead = kvh * GQA + hi;
                    const float * qvec = &Qf[((size_t)(seq * q_len + qpos) * Q_HEADS + qhead) * HEAD_DIM];
                    float * ovec = &out[((size_t)(seq * q_len + qpos) * Q_HEADS + qhead) * HEAD_DIM];
                    if (nkeys <= 0) continue;   // all-masked query: leave the zero fill
                    scores.resize(nkeys);
                    float m = -std::numeric_limits<float>::infinity();
                    for (int kk = 0; kk < nkeys; ++kk) {
                        const int blk = block_table[(size_t)seq * max_blocks + kk / BLOCK_SIZE];
                        const float * kvec = &KVdeq[(((size_t)blk * KV_HEADS + kvh) * BLOCK_SIZE
                                                   + (kk % BLOCK_SIZE)) * 2 * HEAD_DIM + 0];
                        float s = 0.0f;
                        for (int d = 0; d < HEAD_DIM; ++d) s += qvec[d] * kvec[d];
                        s *= scale;
                        scores[kk] = s;
                        m = std::max(m, s);
                    }
                    float l = 0.0f;
                    std::vector<float> acc(HEAD_DIM, 0.0f);
                    for (int kk = 0; kk < nkeys; ++kk) {
                        const float p = expf(scores[kk] - m);
                        l += p;
                        const int blk = block_table[(size_t)seq * max_blocks + kk / BLOCK_SIZE];
                        const float * vvec = &KVdeq[(((size_t)blk * KV_HEADS + kvh) * BLOCK_SIZE
                                                   + (kk % BLOCK_SIZE)) * 2 * HEAD_DIM + HEAD_DIM];
                        for (int d = 0; d < HEAD_DIM; ++d) acc[d] += p * vvec[d];
                    }
                    if (l > 0.0f) {
                        for (int d = 0; d < HEAD_DIM; ++d) ovec[d] = acc[d] / l;
                    }
                }
            }
        }
    }
}

struct ErrStats { float max_abs, max_rel, mean_abs; };

static ErrStats compare_bf16(const std::vector<uint16_t> & gpu_bf16, const std::vector<float> & ref) {
    ErrStats e{0.0f, 0.0f, 0.0f};
    double sum_abs = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float g = bf16_to_f32(gpu_bf16[i]);
        const float r = ref[i];
        const float ad = fabsf(g - r);
        e.max_abs = std::max(e.max_abs, ad);
        sum_abs += ad;
        if (fabsf(r) > 5e-2f) e.max_rel = std::max(e.max_rel, ad / fabsf(r));
    }
    e.mean_abs = (float)(sum_abs / std::max<size_t>(1, ref.size()));
    return e;
}

// ============================================================================================
// Synthetic paged KV cache: shared ground truth in fp32, quantized once to e4m3 bytes and once
// exactly-widened to bf16, so both KV-dtype variants are checked against data that is bit-for-bit
// what the kernel will actually dequantize (no separate "true" value floating around).
// ============================================================================================
struct KvCache {
    int num_blocks, max_blocks;
    std::vector<int>      block_table;   // [num_seqs, max_blocks]
    std::vector<float>    truth;         // [num_blocks, kv_heads, block_size, 2, head_dim]
    std::vector<uint8_t>  fp8_bytes;     // same shape, e4m3
    std::vector<float>    fp8_dequant;   // decode(fp8_bytes) -- CPU reference ground truth for fp8kv
    std::vector<uint16_t> bf16_words;    // same shape, bf16
    std::vector<float>    bf16_dequant;  // widen(bf16_words) -- CPU reference ground truth for bf16kv
};

static KvCache build_kv_cache(std::mt19937 & rng) {
    KvCache kv;
    int needed[NUM_SEQS], offset[NUM_SEQS];
    kv.max_blocks = 0;
    for (int s = 0; s < NUM_SEQS; ++s) {
        needed[s] = (CTX[s] + BLOCK_SIZE - 1) / BLOCK_SIZE;
        kv.max_blocks = std::max(kv.max_blocks, needed[s]);
    }
    int running = 0;
    for (int s = 0; s < NUM_SEQS; ++s) { offset[s] = running; running += needed[s]; }
    kv.num_blocks = running;

    kv.block_table.assign((size_t)NUM_SEQS * kv.max_blocks, 0);
    for (int s = 0; s < NUM_SEQS; ++s) {
        for (int i = 0; i < kv.max_blocks; ++i) {
            kv.block_table[(size_t)s * kv.max_blocks + i] = offset[s] + std::min(i, needed[s] - 1);
        }
    }

    const size_t n = (size_t)kv.num_blocks * KV_HEADS * BLOCK_SIZE * 2 * HEAD_DIM;
    kv.truth.resize(n);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    for (size_t i = 0; i < n; ++i) kv.truth[i] = dist(rng);

    kv.fp8_bytes.resize(n);
    kv.fp8_dequant.resize(n);
    for (size_t i = 0; i < n; ++i) {
        kv.fp8_bytes[i]   = e4m3_encode(kv.truth[i]);
        kv.fp8_dequant[i] = e4m3_decode(kv.fp8_bytes[i]);
    }
    kv.bf16_words.resize(n);
    kv.bf16_dequant.resize(n);
    for (size_t i = 0; i < n; ++i) {
        kv.bf16_words[i]   = f32_to_bf16(kv.truth[i]);
        kv.bf16_dequant[i] = bf16_to_f32(kv.bf16_words[i]);
    }
    return kv;
}

static std::vector<uint16_t> random_bf16_q(std::mt19937 & rng, int q_len) {
    std::vector<uint16_t> q((size_t)NUM_SEQS * q_len * Q_HEADS * HEAD_DIM);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto & v : q) v = f32_to_bf16(dist(rng));
    return q;
}
static std::vector<float> widen_bf16(const std::vector<uint16_t> & q) {
    std::vector<float> out(q.size());
    for (size_t i = 0; i < q.size(); ++i) out[i] = bf16_to_f32(q[i]);
    return out;
}

// ============================================================================================
// One prefill or decode run: allocate device buffers, launch, copy back, report PASS/FAIL.
// ============================================================================================
struct RunResult { bool ok; ErrStats err; float us_per_launch; };

static RunResult run_variant(
    const char * name, bool is_prefill, bool is_fp8kv,
    const KvCache & kv, const std::vector<uint16_t> & q_host, int q_len,
    const std::vector<float> & ref_out)
{
    printf("== %s ==\n", name);

    const size_t kv_elems = (size_t)kv.num_blocks * KV_HEADS * BLOCK_SIZE * 2 * HEAD_DIM;
    void * d_kv = nullptr;
    if (is_fp8kv) {
        HIP_CHECK(hipMalloc(&d_kv, kv_elems * sizeof(uint8_t)));
        HIP_CHECK(hipMemcpy(d_kv, kv.fp8_bytes.data(), kv_elems * sizeof(uint8_t), hipMemcpyHostToDevice));
    } else {
        HIP_CHECK(hipMalloc(&d_kv, kv_elems * sizeof(uint16_t)));
        HIP_CHECK(hipMemcpy(d_kv, kv.bf16_words.data(), kv_elems * sizeof(uint16_t), hipMemcpyHostToDevice));
    }

    int * d_block_table = nullptr;
    HIP_CHECK(hipMalloc(&d_block_table, kv.block_table.size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_block_table, kv.block_table.data(), kv.block_table.size() * sizeof(int), hipMemcpyHostToDevice));

    int seqused_h[NUM_SEQS];
    for (int s = 0; s < NUM_SEQS; ++s) seqused_h[s] = CTX[s];
    int * d_seqused = nullptr;
    HIP_CHECK(hipMalloc(&d_seqused, sizeof(seqused_h)));
    HIP_CHECK(hipMemcpy(d_seqused, seqused_h, sizeof(seqused_h), hipMemcpyHostToDevice));

    void * d_q = nullptr;
    HIP_CHECK(hipMalloc(&d_q, q_host.size() * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(d_q, q_host.data(), q_host.size() * sizeof(uint16_t), hipMemcpyHostToDevice));

    const size_t out_elems = (size_t)NUM_SEQS * q_len * Q_HEADS * HEAD_DIM;
    void * d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, out_elems * sizeof(uint16_t)));

    std::vector<float> descale(NUM_SEQS * KV_HEADS, 1.0f);
    float * d_kdesc = nullptr, * d_vdesc = nullptr;
    HIP_CHECK(hipMalloc(&d_kdesc, descale.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_vdesc, descale.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_kdesc, descale.data(), descale.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vdesc, descale.data(), descale.size() * sizeof(float), hipMemcpyHostToDevice));

    R4DArgs args{};
    args.q            = d_q;
    args.kv           = d_kv;
    args.block_table  = d_block_table;
    args.seqused_k    = d_seqused;
    args.out          = d_out;
    args.k_descale    = d_kdesc;
    args.v_descale    = d_vdesc;
    args.q_descale    = nullptr;
    args.scratch      = nullptr;
    args.num_seqs     = NUM_SEQS;
    args.q_len        = q_len;
    args.q_heads      = Q_HEADS;
    args.kv_heads     = KV_HEADS;
    args.head_dim     = HEAD_DIM;
    args.block_size   = BLOCK_SIZE;
    args.max_blocks   = kv.max_blocks;
    args.kv_block_stride = (long)KV_HEADS * BLOCK_SIZE * 2 * HEAD_DIM;
    args.kv_head_stride  = (long)BLOCK_SIZE * 2 * HEAD_DIM;
    args.scale        = 1.0f / sqrtf((float)HEAD_DIM);
    args.splits       = 0;
    args.max_ctx      = *std::max_element(CTX, CTX + NUM_SEQS);

    void * d_scratch = nullptr;
    if (!is_prefill) {
        const long scratch_bytes = r4d_attn_decode_h256_gqa6_scratch_bytes(&args);
        HIP_CHECK(hipMalloc(&d_scratch, (size_t)scratch_bytes));
        args.scratch = d_scratch;
    }

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    auto launch_once = [&]() -> int {
        if (is_prefill) {
            return is_fp8kv ? r4d_attn_prefill_h256_gqa6_fp8kv(&args, stream)
                             : r4d_attn_prefill_h256_gqa6_bf16kv(&args, stream);
        } else {
            return is_fp8kv ? r4d_attn_decode_h256_gqa6_fp8kv(&args, stream)
                             : r4d_attn_decode_h256_gqa6_bf16kv(&args, stream);
        }
    };

    const int rc = launch_once();
    HIP_CHECK(hipStreamSynchronize(stream));

    RunResult result{};
    if (rc != 0) {
        printf("  launch returned %d (rejected shape) -- FAIL\n", rc);
        result.ok = false;
    } else {
        std::vector<uint16_t> gpu_out(out_elems);
        HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, out_elems * sizeof(uint16_t), hipMemcpyDeviceToHost));
        result.err = compare_bf16(gpu_out, ref_out);
        // bf16 output: ~7-bit mantissa (~4e-3 relative eps) plus fp32-vs-tree-order softmax drift;
        // budget generously since this is a numerics smoke test, not a tight ULP bound.
        result.ok = (result.err.max_abs < 0.05f) && (result.err.max_rel < 0.05f);
        printf("  max_abs_err=%.6f  max_rel_err=%.6f  mean_abs_err=%.6f  -- %s\n",
               result.err.max_abs, result.err.max_rel, result.err.mean_abs, result.ok ? "PASS" : "FAIL");
    }

    // Timing: 20 launches, hipEvent-timed on the same inputs/outputs (correctness already checked).
    hipEvent_t ev_start, ev_stop;
    HIP_CHECK(hipEventCreate(&ev_start));
    HIP_CHECK(hipEventCreate(&ev_stop));
    for (int i = 0; i < 3; ++i) launch_once();          // warm-up
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipEventRecord(ev_start, stream));
    for (int i = 0; i < 20; ++i) launch_once();
    HIP_CHECK(hipEventRecord(ev_stop, stream));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
    result.us_per_launch = ms * 1000.0f / 20.0f;
    printf("  %.2f us/launch (20 launches)\n", result.us_per_launch);

    HIP_CHECK(hipEventDestroy(ev_start));
    HIP_CHECK(hipEventDestroy(ev_stop));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_kv));
    HIP_CHECK(hipFree(d_block_table));
    HIP_CHECK(hipFree(d_seqused));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipFree(d_kdesc));
    HIP_CHECK(hipFree(d_vdesc));
    if (d_scratch) HIP_CHECK(hipFree(d_scratch));
    return result;
}

int main() {
    if (!ggml_cuda_r4d_available()) {
        fprintf(stderr,
            "ggml_cuda_r4d_available() returned false -- either this build was not compiled with "
            "GGML_HIP_R4D, or the current device is not gfx1201. Aborting rather than launching "
            "kernels the current device cannot run.\n");
        return 1;
    }

    e4m3_init_table();
    std::mt19937 rng(12345);
    const KvCache kv = build_kv_cache(rng);

    printf("R4D attention correctness/timing check\n");
    printf("geometry: head_dim=%d gqa=%d block_size=%d kv_heads=%d q_heads=%d\n",
           HEAD_DIM, GQA, BLOCK_SIZE, KV_HEADS, Q_HEADS);
    printf("num_seqs=%d contexts={%d,%d,%d} num_blocks=%d max_blocks=%d\n\n",
           NUM_SEQS, CTX[0], CTX[1], CTX[2], kv.num_blocks, kv.max_blocks);

    bool all_pass = true;

    // ---- prefill, q_len=64 -----------------------------------------------------------------
    {
        const int q_len = 64;
        const std::vector<uint16_t> q_host = random_bf16_q(rng, q_len);
        const std::vector<float>    q_f32  = widen_bf16(q_host);
        std::vector<float> ref_fp8, ref_bf16;
        cpu_attention_ref(q_f32, kv.fp8_dequant,  kv.block_table, kv.max_blocks, q_len, 1.0f / sqrtf((float)HEAD_DIM), ref_fp8);
        cpu_attention_ref(q_f32, kv.bf16_dequant, kv.block_table, kv.max_blocks, q_len, 1.0f / sqrtf((float)HEAD_DIM), ref_bf16);

        all_pass &= run_variant("prefill fp8kv  (q_len=64)",  true,  true,  kv, q_host, q_len, ref_fp8).ok;
        all_pass &= run_variant("prefill bf16kv (q_len=64)",  true,  false, kv, q_host, q_len, ref_bf16).ok;
    }

    // ---- decode, q_len=1 and q_len=8 (speculative verify band) ------------------------------
    for (int q_len : {1, 8}) {
        const std::vector<uint16_t> q_host = random_bf16_q(rng, q_len);
        const std::vector<float>    q_f32  = widen_bf16(q_host);
        std::vector<float> ref_fp8, ref_bf16;
        cpu_attention_ref(q_f32, kv.fp8_dequant,  kv.block_table, kv.max_blocks, q_len, 1.0f / sqrtf((float)HEAD_DIM), ref_fp8);
        cpu_attention_ref(q_f32, kv.bf16_dequant, kv.block_table, kv.max_blocks, q_len, 1.0f / sqrtf((float)HEAD_DIM), ref_bf16);

        char nfp8[64], nbf16[64];
        snprintf(nfp8,  sizeof nfp8,  "decode fp8kv  (q_len=%d)", q_len);
        snprintf(nbf16, sizeof nbf16, "decode bf16kv (q_len=%d)", q_len);
        all_pass &= run_variant(nfp8,  false, true,  kv, q_host, q_len, ref_fp8).ok;
        all_pass &= run_variant(nbf16, false, false, kv, q_host, q_len, ref_bf16).ok;
    }

    printf("\n%s\n", all_pass ? "OVERALL: PASS" : "OVERALL: FAIL");
    return all_pass ? 0 : 1;
}
