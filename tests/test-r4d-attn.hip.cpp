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
#include <hip/hip_fp16.h>
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

// ============================================================================================
// turbo4_fp8_bs256 host packer -- mirrors mt_scatter_kv_turbo4_fp8_aiter_kernel exactly
// (mt_pagedattn_aiter.cu, mt_scatter_kv_turbo4_fp8_aiter_kernel, stages 2-5): a 162-byte record
// per (block, slot, kv head) -- [0..1] fp16 per-vector scale, [2..129] 128 bytes of 4-bit LUT
// indices (2 per byte, low nibble = even element), [130..161] 32 bytes of sign bits (bit j%8 of
// byte j/8, 1 = negative). value_j = (neg?-1:1) * lut_f[idx_j] * scale.
// ============================================================================================
static constexpr int TURBO4_RECORD_BYTES = 162;
static constexpr int TURBO4_N_CENT       = 16;

// 16 increasing e4m3 magnitude centroids in [0,1] -- a plausible LUT for a synthetic test (the
// real ones come from calibration; this is only about exercising the dequant path faithfully).
// lut_f is the e4m3 DECODE of each byte, since that -- not the target -- is what both the
// quantizer's nearest-centroid search and the kernel's fp8-hardware read of the LUT actually use.
static void turbo4_build_lut(uint8_t lut_bytes[TURBO4_N_CENT], float lut_f[TURBO4_N_CENT]) {
    static const float targets[TURBO4_N_CENT] = {
        0.00f, 0.03f, 0.06f, 0.10f, 0.14f, 0.19f, 0.24f, 0.30f,
        0.36f, 0.43f, 0.51f, 0.60f, 0.70f, 0.80f, 0.90f, 1.00f
    };
    for (int i = 0; i < TURBO4_N_CENT; ++i) {
        lut_bytes[i] = e4m3_encode(targets[i]);
        lut_f[i]     = e4m3_decode(lut_bytes[i]);
    }
}

// Quantize one head_dim-length vector into a 162-byte turbo4 record, mirroring the scatter
// kernel's stages 2-5 exactly: scale = max_j|x_j| cast to fp16 (the fp16-rounded value is used for
// everything after), mag = |x_j|/scale, idx = argmin_c |mag - lut_f[c]|, sign = x_j<0. Also writes
// the DEQUANTIZED vector (sign*lut_f[idx]*scale, as float) into dequant_out so a CPU oracle can be
// built from exactly what the kernel will read back -- isolating it from quantization error.
static void turbo4_quantize_vec(const float * x, int head_dim, const float * lut_f,
                                 uint8_t * rec /* TURBO4_RECORD_BYTES */, float * dequant_out) {
    float maxabs = 0.0f;
    for (int d = 0; d < head_dim; ++d) maxabs = std::max(maxabs, fabsf(x[d]));
    const __half scale_h   = __float2half(maxabs);
    const float  scale_eff = __half2float(scale_h);
    const float  inv_scale = (scale_eff > 0.0f) ? (1.0f / scale_eff) : 0.0f;

    memset(rec, 0, TURBO4_RECORD_BYTES);
    memcpy(rec, &scale_h, 2);

    for (int d = 0; d < head_dim; ++d) {
        const float v   = x[d];
        const int   sgn = v < 0.0f ? 1 : 0;
        const float mag = fabsf(v) * inv_scale;
        int best = 0; float bestd = fabsf(mag - lut_f[0]);
        for (int c = 1; c < TURBO4_N_CENT; ++c) {
            const float e = fabsf(mag - lut_f[c]);
            if (e < bestd) { best = c; bestd = e; }
        }
        const uint8_t nib = (uint8_t)best;
        if (d & 1) rec[2 + d / 2] |= (uint8_t)(nib << 4);
        else       rec[2 + d / 2] |= nib;
        if (sgn) rec[130 + d / 8] |= (uint8_t)(1u << (d % 8));
        dequant_out[d] = (sgn ? -1.0f : 1.0f) * lut_f[best] * scale_eff;
    }
}

// Single-sequence causal softmax attention oracle for an arbitrary (kv_heads, gqa, head_dim)
// geometry, with K/V given directly in LOGICAL key order (no block-table indirection needed --
// the turbo4 correctness gate below already resolves physical block ids while it quantizes).
static void cpu_attention_ref_single_seq(
    const std::vector<float> & Qf,    // [q_len, q_heads, head_dim]
    const std::vector<float> & Kdeq,  // [ctx, kv_heads, head_dim]
    const std::vector<float> & Vdeq,  // [ctx, kv_heads, head_dim]
    int q_len, int ctx, int kv_heads, int gqa, int head_dim, float scale,
    std::vector<float> & out)         // [q_len, q_heads, head_dim]
{
    const int q_heads = kv_heads * gqa;
    out.assign((size_t)q_len * q_heads * head_dim, 0.0f);
    std::vector<float> scores;
    for (int qpos = 0; qpos < q_len; ++qpos) {
        const int klimit = ctx - q_len + qpos;
        const int nkeys  = std::min(klimit + 1, ctx);
        if (nkeys <= 0) continue;
        for (int kvh = 0; kvh < kv_heads; ++kvh) {
            for (int hi = 0; hi < gqa; ++hi) {
                const int qhead = kvh * gqa + hi;
                const float * qvec = &Qf[((size_t)qpos * q_heads + qhead) * head_dim];
                float * ovec = &out[((size_t)qpos * q_heads + qhead) * head_dim];
                scores.resize(nkeys);
                float m = -std::numeric_limits<float>::infinity();
                for (int kk = 0; kk < nkeys; ++kk) {
                    const float * kvec = &Kdeq[((size_t)kk * kv_heads + kvh) * head_dim];
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; ++d) s += qvec[d] * kvec[d];
                    s *= scale;
                    scores[kk] = s;
                    m = std::max(m, s);
                }
                float l = 0.0f;
                std::vector<float> acc(head_dim, 0.0f);
                for (int kk = 0; kk < nkeys; ++kk) {
                    const float p = expf(scores[kk] - m);
                    l += p;
                    const float * vvec = &Vdeq[((size_t)kk * kv_heads + kvh) * head_dim];
                    for (int d = 0; d < head_dim; ++d) acc[d] += p * vvec[d];
                }
                if (l > 0.0f) for (int d = 0; d < head_dim; ++d) ovec[d] = acc[d] / l;
            }
        }
    }
}

// Row relRMSE: per (token, head) output vector, sqrt(mean((gpu-ref)^2)) / sqrt(mean(ref^2)); the
// max over rows. A per-vector quality metric distinct from compare_bf16's flat max_abs/max_rel --
// the same quantity r4d_attn_prefill_h256_gqa6.hip's dev comments cite for the R4D_ATTN_FP8 legs
// (e.g. "1.7e-3 -> ... row relRMSE"), reported here for the turbo4 gate but not itself a pass gate.
static float row_rel_rmse(const std::vector<uint16_t> & gpu_bf16, const std::vector<float> & ref,
                           int rows, int row_len) {
    float worst = 0.0f;
    for (int r = 0; r < rows; ++r) {
        double se = 0.0, re = 0.0;
        for (int d = 0; d < row_len; ++d) {
            const float g    = bf16_to_f32(gpu_bf16[(size_t)r * row_len + d]);
            const float v    = ref[(size_t)r * row_len + d];
            const float diff = g - v;
            se += (double)diff * diff;
            re += (double)v * v;
        }
        const float denom = sqrtf((float)(re / row_len));
        const float num   = sqrtf((float)(se / row_len));
        const float rel   = denom > 1e-6f ? num / denom : num;
        worst = std::max(worst, rel);
    }
    return worst;
}

// ============================================================================================
// Production-shape prefill timing bench (MAD radiance-vs-ours 2x investigation, see
// mt_pagedattn_r4d.cu's header comment for the R4DArgs field-by-field diff this backs).
//
// Field-by-field comparison against radiance_r4d_attn.py's R4DAttentionImpl.forward() found every
// controllable R4DArgs field for the prefill launch already matching between the two callers
// (kv_block_stride/kv_head_stride formula, k/v descale null in the non-fp8-scale case, splits=0,
// max_ctx -- which the prefill kernel does not even read; only decode_splits()/scratch sizing do).
// The one remaining, VERIFIABLE difference in what each caller passes as `num_seqs` (R4DArgs.num_seqs,
// which becomes the kernel's grid.z and indexes block_table/seqused_k, one slot per z):
//
//   radiance (radiance_r4d_attn.py forward(), R4DAttentionMetadataBuilder._plan()): num_seqs is the
//   ACTUAL live-sequence count in this particular launch's run (a single-request 16k-prefill bench
//   has exactly one live sequence => num_seqs=1 => grid.z=1).
//
//   ours (mt_pagedattn_r4d.cu): num_seqs = block_tables->ne[1], the paged cache's STATIC n_seq_max
//   -- ALWAYS, even when only one sequence is actually being prefilled -- because R4D's block_table/
//   seqused_k/slot-indexed Q are SEQ-ID-INDEXED, NOT compacted (a live sequence can sit at any slot;
//   see the identical, explicit precedent in mt_pagedattn_aiter.cu's num_seqs_dispatch comment,
//   ~:1021-1034: "the correct (and only safe) bound for anything that INDEXES into those arrays
//   (grid dims, ...)"). This is NOT a bug in the adapter -- shrinking grid.z to the live count while
//   still addressing block_table/seqused_k by absolute slot index would read the WRONG sequences'
//   context whenever the live sequence(s) are not packed at slot 0..num_active-1, which llama.cpp's
//   paged cache does not guarantee. So this file cannot "fix" it without either (a) a host-visible
//   guarantee that live slots are always slot-packed starting at 0 for this call (not established
//   here), or (b) compacting block_table/seqused_k on device first (a real change, reported rather
//   than made per the task's instructions on scope).
//
// What THIS bench does instead: measure whether that one remaining difference -- launching the
// SAME compiled kernel (verified byte-identical against the vendored
// r4d_attn_prefill_h256_gqa6.hip against .../radiance-libr4d/b9e42ab-rx9's copy) with grid.z padded
// to a static n_seq_max of slots that are mostly idle (seqused_k<=0, early-return) -- actually costs
// the reported ~2x at production shape (q_len=4096, ctx=16384, 48 q-heads / 8 kv-heads, fp8 KV).
// "ours-before" = num_seqs_static=8 (a plausible --parallel/n_seq_max server config) with only slot
// 0 live; "radiance-matched" = num_seqs_static=1, matching radiance's exact grid.z for this same
// single-request prefill. If the two report the same us/launch, grid.z padding is NOT the
// explanation and the gap must come from something outside the launch arguments this file can see
// (GPU/measurement conditions, server flags, etc.) -- report that back rather than re-guessing here.
// ============================================================================================
static bool run_production_prefill_bench(int num_seqs_static, const char * label) {
    constexpr int P_HEAD_DIM  = 256;
    constexpr int P_GQA        = 6;
    constexpr int P_BLOCK_SIZE = 16;
    constexpr int P_KV_HEADS   = 8;
    constexpr int P_Q_HEADS    = P_KV_HEADS * P_GQA;   // 48
    // RA_QLEN / RA_CTX env override (orchestrator sweep: ours = 4096-query ubatches at ctx
    // 4096..16384; radiance = ~2731-query chunks) -- defaults keep the original shape.
    const int P_Q_LEN = getenv("RA_QLEN") ? atoi(getenv("RA_QLEN")) : 4096;
    const int P_CTX   = getenv("RA_CTX")  ? atoi(getenv("RA_CTX"))  : 16384;

    // ---- bench-vs-production reconciliation knobs (2026-09-20 follow-up) -----------------------
    // Every knob defaults to the ORIGINAL bench behavior (mode 0 / extra 0), so an unset
    // environment reproduces the exact numbers already measured.
    //
    // RA_BT_MODE=0 (default): block ids for the live slot are 0..max_blocks-1, sequential --
    //   i.e. contiguous physical addresses in a freshly hipMalloc'd buffer, the BEST-case
    //   locality a paged allocator could ever produce. RA_BT_MODE=1: the live slot's logical
    //   blocks 0..max_blocks-1 are assigned a RANDOM PERMUTATION of ids drawn from a
    //   RA_BT_POOL_MULT (default 4) times larger physical pool -- simulating a KV cache that has
    //   already cycled through other sequences/evictions so the live chain is scattered across a
    //   wider physical range, rather than the bench's current best-case-contiguous default.
    const int  bt_mode      = getenv("RA_BT_MODE") ? atoi(getenv("RA_BT_MODE")) : 0;
    const int  bt_pool_mult = getenv("RA_BT_POOL_MULT") ? atoi(getenv("RA_BT_POOL_MULT")) : 4;

    // RA_KV_MODE=0 (default): raw random bytes in [0x00, 0x7F] -- this DOES include 0x78..0x7F
    // (exponent==1111), and byte 0x7F specifically is the e4m3 NaN encoding (S.1111.111), so ~1/128
    // of every KV element the default mode ever quantizes is silently NaN today. RA_KV_MODE=1:
    // e4m3_encode() of a realistic (bounded, non-saturating) value distribution -- e4m3_encode's
    // binary search only ever returns index 0..126 (see its definition above), so this mode can
    // never emit the NaN byte pattern.
    // RA_KV_MODE=2: turbo4_fp8_bs256 (KVP=2) -- host-quantize a real fp32 K/V into the 162-byte
    // per-(block,slot,kv head) turbo4 record (turbo4_quantize_vec above, mirroring
    // mt_scatter_kv_turbo4_fp8_aiter_kernel in mt_pagedattn_aiter.cu) and gate
    // r4d_attn_prefill_h256_gqa6_turbo4kv's output against a CPU oracle built from the
    // DEQUANTIZED K/V, so the compare isolates the kernel from turbo4's own quantization error.
    // Unlike modes 0/1 this DOES affect all_pass below -- it is the correctness gate this knob
    // exists for, not a timing-only probe. A decode-shaped (q_len*gqa<=64) request now runs
    // r4d_attn_decode_h256_gqa6_turbo4kv (split-KV, scratch/splits/max_ctx set exactly like
    // run_variant's decode path) against the same dequantized-K/V oracle; larger shapes still run
    // the turbo4 prefill kernel. NOTE: the CPU oracle is O(q_len*ctx*q_heads*head_dim) -- run this
    // mode with small RA_QLEN/RA_CTX (e.g. 64/512, or RA_QLEN=1/RA_QLEN=8 for the decode band), not
    // the 4096/16384 production default, or the reference computation will not finish in a
    // reasonable time.
    const int  kv_mode = getenv("RA_KV_MODE") ? atoi(getenv("RA_KV_MODE")) : 0;

    // RA_DESCALE_MODE=0 (default): k_descale/v_descale both null (r4d.h: NULL => 1.0, matches
    // radiance's fold when scale==1.0). RA_DESCALE_MODE=1: allocate real (num_seqs_static *
    // kv_heads) fp32 buffers filled with 1.0 and pass THOSE pointers instead, so the kernel takes
    // its "a.k_descale ? a.k_descale[...] : 1.0f" branch the other way (one extra global load of
    // a uniform value per workgroup at kernel entry -- see r4d_attn_prefill_kernel's `kdesc`/`vdesc`).
    const int  descale_mode = getenv("RA_DESCALE_MODE") ? atoi(getenv("RA_DESCALE_MODE")) : 0;

    // RA_CTX_EXTRA=0 (default): seqused_k for the live slot is exactly P_CTX, matching
    // production's context_lens[seq] = pos_max+1 convention exactly (mt_pagedattn.cu:1357,
    // llama-kv-cache-paged.cpp:1582). A nonzero value directly measures what an off-by-N context
    // accounting bug elsewhere in the paged-attn pipeline would cost HERE, in isolation: it both
    // grows seqused_k[0] by N and grows max_blocks (block_table row length) enough to cover it, so
    // the kernel's own `ntiles = ceil(min(klimit_cta+1, ctx) / TILE)` sees exactly N extra tokens
    // of real (non-early-return) causal work per query row -- not a knob production is known to
    // need, purely a calibration probe for (e).
    const int  ctx_extra = getenv("RA_CTX_EXTRA") ? atoi(getenv("RA_CTX_EXTRA")) : 0;
    const int  ctx_effective = P_CTX + ctx_extra;

    printf("== production prefill bench: %s (num_seqs_static=%d, q_len=%d, ctx=%d"
           "%s%d%s, bt_mode=%d bt_pool_mult=%d kv_mode=%d descale_mode=%d) ==\n",
           label, num_seqs_static, P_Q_LEN, P_CTX,
           ctx_extra ? " +" : "", ctx_extra, ctx_extra ? " ctx_extra" : "",
           bt_mode, bt_pool_mult, kv_mode, descale_mode);

    const bool is_decode = (P_Q_LEN * P_GQA <= 64);   // turbo4 (kv_mode==2) decode band, gqa=6 => q_len<=10

    const int max_blocks     = (ctx_effective + P_BLOCK_SIZE - 1) / P_BLOCK_SIZE;   // block_table row length
    const int num_blocks_pool = (bt_mode == 1) ? max_blocks * std::max(1, bt_pool_mult) : max_blocks;

    // ---- diagnostics for (c)/(e): tile counts the kernel will actually run, computed with the
    // SAME formulas as r4d_attn_prefill_kernel (r4d_attn_prefill_h256_gqa6.hip) so a mismatch here
    // would mean the bench's (q_len, ctx) pairing does NOT reproduce production's causal window. --
    {
        constexpr int P_NWARPS = 24, P_TILE = 48;                 // must match r4d_attn_paged_h256_gqa6.hip
        const int BQ = (P_NWARPS * 16) / P_GQA;                   // 64: queries per CTA (grid.x unit)
        const int grid_x = (P_Q_LEN + BQ - 1) / BQ;
        auto ntiles_for_cta = [&](int qb) {
            const int qmax_cta   = std::min(qb * BQ + BQ - 1, P_Q_LEN - 1);
            const int klimit_cta = ctx_effective - P_Q_LEN + qmax_cta;
            return (std::min(klimit_cta + 1, ctx_effective) + P_TILE - 1) / P_TILE;
        };
        const int ntiles_first = ntiles_for_cta(0);
        const int ntiles_last  = ntiles_for_cta(grid_x - 1);
        const int pad_rows     = grid_x * BQ - P_Q_LEN;           // dead rows: computed in full, just not written
        printf("  diag: BQ=%d grid.x=%d (q_len %% BQ == %d, %d padding rows computed-but-discarded), "
               "ntiles[cta0]=%d ntiles[ctaLast]=%d\n",
               BQ, grid_x, P_Q_LEN % BQ, pad_rows, ntiles_first, ntiles_last);
    }

    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::normal_distribution<float>       ndist(0.0f, 0.5f);   // RA_KV_MODE=1: bounded, non-saturating

    std::vector<int> block_table((size_t)num_seqs_static * max_blocks, 0);
    if (bt_mode == 1) {
        std::vector<int> pool(num_blocks_pool);
        for (int i = 0; i < num_blocks_pool; ++i) pool[i] = i;
        std::shuffle(pool.begin(), pool.end(), rng);
        for (int i = 0; i < max_blocks; ++i) block_table[i] = pool[i];   // scattered across the wider pool
    } else {
        for (int i = 0; i < max_blocks; ++i) block_table[i] = i;        // slot 0: sequential real chain
    }
    // other slots (rows 1..num_seqs_static-1): left at 0 -- unused (idle, seqused=0 below)

    std::vector<int> seqused_h(num_seqs_static, 0);
    seqused_h[0] = ctx_effective;   // only slot 0 is live -- every other slot returns immediately in-kernel

    const size_t q_elems = (size_t)num_seqs_static * P_Q_LEN * P_Q_HEADS * P_HEAD_DIM;
    std::vector<uint16_t> q_host(q_elems, 0);
    // RA_QMUL (default 1): scale Q so the scores swing wide (real attention is peaky) and the
    // kernel's lazy max-rescale lets P grow towards its PGROW budget before the V-scale fold.
    const float qmul = getenv("RA_QMUL") ? (float) atof(getenv("RA_QMUL")) : 1.0f;
    for (size_t i = 0; i < (size_t)P_Q_LEN * P_Q_HEADS * P_HEAD_DIM; ++i)   // fill slot 0's rows only
        q_host[i] = f32_to_bf16(dist(rng) * qmul);

    int * d_block_table = nullptr;
    HIP_CHECK(hipMalloc(&d_block_table, block_table.size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_block_table, block_table.data(), block_table.size() * sizeof(int), hipMemcpyHostToDevice));

    int * d_seqused = nullptr;
    HIP_CHECK(hipMalloc(&d_seqused, seqused_h.size() * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_seqused, seqused_h.data(), seqused_h.size() * sizeof(int), hipMemcpyHostToDevice));

    void * d_q = nullptr;
    HIP_CHECK(hipMalloc(&d_q, q_elems * sizeof(uint16_t)));
    HIP_CHECK(hipMemcpy(d_q, q_host.data(), q_elems * sizeof(uint16_t), hipMemcpyHostToDevice));

    void * d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_out, q_elems * sizeof(uint16_t)));

    R4DArgs args{};
    args.q               = d_q;
    args.block_table     = d_block_table;
    args.seqused_k       = d_seqused;
    args.out             = d_out;
    args.q_descale       = nullptr;
    args.scratch         = nullptr;   // prefill does not read scratch; turbo4 decode below overwrites this
    args.num_seqs        = num_seqs_static;
    args.q_len           = P_Q_LEN;
    args.q_heads         = P_Q_HEADS;
    args.kv_heads        = P_KV_HEADS;
    args.head_dim        = P_HEAD_DIM;
    args.block_size      = P_BLOCK_SIZE;
    args.max_blocks      = max_blocks;
    args.scale           = 1.0f / sqrtf((float)P_HEAD_DIM);
    args.splits          = 0;             // radiance: "let the kernel's split law choose" (unused by prefill anyway)
    args.max_ctx         = ctx_effective; // radiance: common_attn_metadata.max_seq_len (unused by prefill anyway)

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    bool ok = true;
    void * d_kv = nullptr, * d_vcache = nullptr;
    float * d_kdesc = nullptr, * d_vdesc = nullptr;
    unsigned char * d_klut = nullptr, * d_vlut = nullptr;
    void * d_scratch = nullptr;   // turbo4 decode only (split-KV partials)

    if (kv_mode == 2) {
        // ---- turbo4_fp8_bs256: host-quantize real K/V, gate against a dequantized-K/V oracle ----
        uint8_t k_lut_bytes[TURBO4_N_CENT], v_lut_bytes[TURBO4_N_CENT];
        float   k_lut_f[TURBO4_N_CENT],     v_lut_f[TURBO4_N_CENT];
        turbo4_build_lut(k_lut_bytes, k_lut_f);
        turbo4_build_lut(v_lut_bytes, v_lut_f);   // same centroids for K and V -- one plausible LUT

        args.kv_block_stride = (long)P_KV_HEADS * P_BLOCK_SIZE * TURBO4_RECORD_BYTES;  // BYTES
        args.kv_slot_stride  = (long)P_KV_HEADS * TURBO4_RECORD_BYTES;                 // BYTES
        args.kv_head_stride  = (long)TURBO4_RECORD_BYTES;                              // BYTES
        args.k_descale       = nullptr;
        args.v_descale       = nullptr;

        const size_t rec_elems = (size_t)num_blocks_pool * P_KV_HEADS * P_BLOCK_SIZE * TURBO4_RECORD_BYTES;
        std::vector<uint8_t> k_bytes(rec_elems, 0), v_bytes(rec_elems, 0);

        // Oracle ground truth in LOGICAL key order: [ctx, kv_heads, head_dim]. Physical block ids
        // (block_table row 0, possibly shuffled by RA_BT_MODE) are resolved right here, so the
        // oracle itself never needs to walk block_table.
        std::vector<float> Kdeq((size_t)ctx_effective * P_KV_HEADS * P_HEAD_DIM);
        std::vector<float> Vdeq((size_t)ctx_effective * P_KV_HEADS * P_HEAD_DIM);
        std::vector<float> vecbuf(P_HEAD_DIM);
        float scratch_dequant[P_HEAD_DIM];
        const float vmul = getenv("RA_VMUL") ? (float) atof(getenv("RA_VMUL")) : 1.0f;

        for (int i = 0; i < max_blocks; ++i) {
            const int blk = block_table[i];   // row 0 -- the only live sequence
            for (int slot = 0; slot < P_BLOCK_SIZE; ++slot) {
                const int kk = i * P_BLOCK_SIZE + slot;
                const bool in_ctx = kk < ctx_effective;
                for (int kvh = 0; kvh < P_KV_HEADS; ++kvh) {
                    uint8_t * rec_k = k_bytes.data()
                        + (size_t)blk * args.kv_block_stride + (size_t)slot * args.kv_slot_stride
                        + (size_t)kvh * args.kv_head_stride;
                    uint8_t * rec_v = v_bytes.data()
                        + (size_t)blk * args.kv_block_stride + (size_t)slot * args.kv_slot_stride
                        + (size_t)kvh * args.kv_head_stride;
                    float * kdeq_dst = in_ctx ? &Kdeq[((size_t)kk * P_KV_HEADS + kvh) * P_HEAD_DIM] : scratch_dequant;
                    float * vdeq_dst = in_ctx ? &Vdeq[((size_t)kk * P_KV_HEADS + kvh) * P_HEAD_DIM] : scratch_dequant;

                    for (int d = 0; d < P_HEAD_DIM; ++d) vecbuf[d] = ndist(rng);
                    turbo4_quantize_vec(vecbuf.data(), P_HEAD_DIM, k_lut_f, rec_k, kdeq_dst);

                    // RA_VMUL (default 1): scale the V vectors so the per-key fp16 V scale is large,
                    // like real V rows (max|v| of tens+) -- the fold of that scale into the fp8/f16 P
                    // fragment is where the kernel can overflow (found in production as all-NaN output).
                    for (int d = 0; d < P_HEAD_DIM; ++d) vecbuf[d] = ndist(rng) * vmul;
                    turbo4_quantize_vec(vecbuf.data(), P_HEAD_DIM, v_lut_f, rec_v, vdeq_dst);
                }
            }
        }

        HIP_CHECK(hipMalloc(&d_kv, rec_elems));
        HIP_CHECK(hipMemcpy(d_kv, k_bytes.data(), rec_elems, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc(&d_vcache, rec_elems));
        HIP_CHECK(hipMemcpy(d_vcache, v_bytes.data(), rec_elems, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc((void**)&d_klut, TURBO4_N_CENT));
        HIP_CHECK(hipMemcpy(d_klut, k_lut_bytes, TURBO4_N_CENT, hipMemcpyHostToDevice));
        HIP_CHECK(hipMalloc((void**)&d_vlut, TURBO4_N_CENT));
        HIP_CHECK(hipMemcpy(d_vlut, v_lut_bytes, TURBO4_N_CENT, hipMemcpyHostToDevice));

        args.kv      = d_kv;
        args.v_cache = d_vcache;
        args.k_lut   = d_klut;
        args.v_lut   = d_vlut;

        if (is_decode) {
            // Mirror run_variant's decode setup exactly: args.splits=0 / args.max_ctx=ctx_effective
            // are already set above the same way for prefill and decode; scratch is decode-only.
            const long scratch_bytes = r4d_attn_decode_h256_gqa6_scratch_bytes(&args);
            HIP_CHECK(hipMalloc(&d_scratch, (size_t)scratch_bytes));
            args.scratch = d_scratch;
        }

        auto launch_once = [&]() -> int {
            return is_decode ? r4d_attn_decode_h256_gqa6_turbo4kv(&args, stream)
                              : r4d_attn_prefill_h256_gqa6_turbo4kv(&args, stream);
        };

        const int rc = launch_once();
        HIP_CHECK(hipStreamSynchronize(stream));
        if (rc != 0) {
            printf("  launch returned %d (rejected shape) -- FAIL\n", rc);
            ok = false;
        } else {
            const std::vector<uint16_t> q_row0(q_host.begin(),
                q_host.begin() + (size_t)P_Q_LEN * P_Q_HEADS * P_HEAD_DIM);
            const std::vector<float> q_f32 = widen_bf16(q_row0);
            std::vector<float> ref_out;
            cpu_attention_ref_single_seq(q_f32, Kdeq, Vdeq, P_Q_LEN, ctx_effective,
                                          P_KV_HEADS, P_GQA, P_HEAD_DIM, args.scale, ref_out);

            const size_t out_elems = (size_t)P_Q_LEN * P_Q_HEADS * P_HEAD_DIM;
            std::vector<uint16_t> gpu_out(out_elems);
            HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, out_elems * sizeof(uint16_t), hipMemcpyDeviceToHost));
            const ErrStats err   = compare_bf16(gpu_out, ref_out);
            const float    rrmse = row_rel_rmse(gpu_out, ref_out, P_Q_LEN * P_Q_HEADS, P_HEAD_DIM);
            // Same tolerance as the fp8kv prefill/decode gate in run_variant (bf16 output: ~4e-3
            // relative eps plus fp32-vs-tree-order softmax drift, budgeted generously as a numerics
            // smoke test rather than a tight ULP bound): max_abs<0.05 and max_rel<0.05. Fair to hold
            // turbo4 to the SAME bar here because the oracle is built from the DEQUANTIZED K/V --
            // this isolates the kernel's own numerics from turbo4's quantization error, exactly like
            // the fp8kv gate isolates the fp8 kernel from e4m3's.
            ok = (err.max_abs < 0.05f) && (err.max_rel < 0.05f);
            printf("  max_abs_err=%.6f  max_rel_err=%.6f  mean_abs_err=%.6f  row_relRMSE=%.6f  -- %s\n",
                   err.max_abs, err.max_rel, err.mean_abs, rrmse, ok ? "PASS" : "FAIL");
        }

        hipEvent_t ev_start, ev_stop;
        HIP_CHECK(hipEventCreate(&ev_start));
        HIP_CHECK(hipEventCreate(&ev_stop));
        const int n_warm = getenv("RA_WARM") ? atoi(getenv("RA_WARM")) : 60;
        for (int i = 0; i < n_warm; ++i) launch_once();
        HIP_CHECK(hipStreamSynchronize(stream));
        HIP_CHECK(hipEventRecord(ev_start, stream));
        const int N_LAUNCHES = getenv("RA_ITERS") ? atoi(getenv("RA_ITERS")) : 10;
        for (int i = 0; i < N_LAUNCHES; ++i) launch_once();
        HIP_CHECK(hipEventRecord(ev_stop, stream));
        HIP_CHECK(hipEventSynchronize(ev_stop));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
        printf("  %.3f ms/launch (%d launches)\n", ms / N_LAUNCHES, N_LAUNCHES);
        HIP_CHECK(hipEventDestroy(ev_start));
        HIP_CHECK(hipEventDestroy(ev_stop));
    } else {
        // ---- original fp8/bf16-random-byte KV path (kv_mode 0/1), UNCHANGED behavior ------------
        const size_t kv_elems = (size_t)num_blocks_pool * P_KV_HEADS * P_BLOCK_SIZE * 2 * P_HEAD_DIM;
        std::vector<uint8_t> kv_fp8(kv_elems);
        if (kv_mode == 1) {
            for (auto & b : kv_fp8) b = e4m3_encode(ndist(rng));       // never the NaN byte (0x7F)
        } else {
            for (auto & b : kv_fp8) b = (uint8_t)(rng() & 0x7Fu);      // original: can hit 0x7F (NaN) ~1/128
        }

        HIP_CHECK(hipMalloc(&d_kv, kv_elems * sizeof(uint8_t)));
        HIP_CHECK(hipMemcpy(d_kv, kv_fp8.data(), kv_elems * sizeof(uint8_t), hipMemcpyHostToDevice));

        if (descale_mode == 1) {
            const size_t n_desc = (size_t)num_seqs_static * P_KV_HEADS;
            std::vector<float> ones(n_desc, 1.0f);
            HIP_CHECK(hipMalloc(&d_kdesc, n_desc * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_vdesc, n_desc * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_kdesc, ones.data(), n_desc * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_vdesc, ones.data(), n_desc * sizeof(float), hipMemcpyHostToDevice));
        }

        args.kv_block_stride = (long)P_KV_HEADS * P_BLOCK_SIZE * 2 * P_HEAD_DIM;
        args.kv_head_stride  = (long)P_BLOCK_SIZE * 2 * P_HEAD_DIM;
        args.kv              = d_kv;
        args.k_descale       = d_kdesc;   // null unless RA_DESCALE_MODE=1
        args.v_descale       = d_vdesc;

        auto launch_once = [&]() -> int {
            return r4d_attn_prefill_h256_gqa6_fp8kv(&args, stream);
        };

        const int rc = launch_once();
        HIP_CHECK(hipStreamSynchronize(stream));
        if (rc != 0) {
            printf("  launch returned %d (rejected shape) -- SKIPPED\n", rc);
        } else {
            hipEvent_t ev_start, ev_stop;
            HIP_CHECK(hipEventCreate(&ev_start));
            HIP_CHECK(hipEventCreate(&ev_stop));
            // warm-up long enough to lift the card out of its idle DPM state (~0.5 s):
            // RA_WARM launches (default 60 -- ~1.5 s at the 4096/16384 shape)
            const int n_warm = getenv("RA_WARM") ? atoi(getenv("RA_WARM")) : 60;
            for (int i = 0; i < n_warm; ++i) launch_once();
            HIP_CHECK(hipStreamSynchronize(stream));
            HIP_CHECK(hipEventRecord(ev_start, stream));
            const int N_LAUNCHES = getenv("RA_ITERS") ? atoi(getenv("RA_ITERS")) : 10;
            for (int i = 0; i < N_LAUNCHES; ++i) launch_once();
            HIP_CHECK(hipEventRecord(ev_stop, stream));
            HIP_CHECK(hipEventSynchronize(ev_stop));
            float ms = 0.0f;
            HIP_CHECK(hipEventElapsedTime(&ms, ev_start, ev_stop));
            printf("  %.3f ms/launch (%d launches)\n", ms / N_LAUNCHES, N_LAUNCHES);
            HIP_CHECK(hipEventDestroy(ev_start));
            HIP_CHECK(hipEventDestroy(ev_stop));
        }
    }

    HIP_CHECK(hipStreamDestroy(stream));
    if (d_kv)      HIP_CHECK(hipFree(d_kv));
    if (d_vcache)  HIP_CHECK(hipFree(d_vcache));
    if (d_klut)    HIP_CHECK(hipFree(d_klut));
    if (d_vlut)    HIP_CHECK(hipFree(d_vlut));
    if (d_scratch) HIP_CHECK(hipFree(d_scratch));
    HIP_CHECK(hipFree(d_block_table));
    HIP_CHECK(hipFree(d_seqused));
    HIP_CHECK(hipFree(d_q));
    HIP_CHECK(hipFree(d_out));
    if (d_kdesc) HIP_CHECK(hipFree(d_kdesc));
    if (d_vdesc) HIP_CHECK(hipFree(d_vdesc));
    return ok;
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

    // ---- production-shape prefill bench: ours-before (padded num_seqs) vs radiance-matched -----
    // At RA_KV_MODE 0/1 (default) this does not affect all_pass -- it's a timing-only probe for the
    // radiance-vs-ours 2x investigation (see run_production_prefill_bench's header comment), not a
    // correctness check. At RA_KV_MODE=2 it IS a correctness gate (turbo4_fp8_bs256 vs its
    // dequantized-K/V oracle) and its result folds into all_pass like every other variant above.
    printf("\n");
    bool turbo4_ok = true;
    turbo4_ok &= run_production_prefill_bench(8, "ours-before (num_seqs_static=8, static n_seq_max padding)");
    turbo4_ok &= run_production_prefill_bench(1, "radiance-matched (num_seqs_static=1, exact live count)");
    all_pass &= turbo4_ok;

    printf("\n%s\n", all_pass ? "OVERALL: PASS" : "OVERALL: FAIL");
    return all_pass ? 0 : 1;
}
