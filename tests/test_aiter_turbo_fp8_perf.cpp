// MAD-214: F16 vs turbo4-FP8 kernel performance comparison.
//
// Same shape as the correctness test (NUM_KV_TOKENS=512, GQA 16/4, head=256)
// but skips the scalar reference and instead times N iterations of each
// cache_type via hipEvent. Reports per-call latency and FP8/F16 speedup.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 -I ggml/src/ggml-cuda/aiter-integration/wrappers \
//       tests/test_aiter_turbo_fp8_perf.cpp -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin -o /tmp/test_aiter_turbo_fp8_perf

#include "mt_aiter_unified_attn.h"
#include "../ggml/src/ggml-cuda/aiter-integration/turbo_fp8_data/qwen35_4b_bs256_centroids.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
        return 1; \
    } \
} while(0)

constexpr int HEAD_SIZE       = 256;
constexpr int NUM_Q_HEADS     = 16;
constexpr int NUM_KV_HEADS    = 4;
constexpr int BLOCK_SIZE      = 16;
constexpr int NUM_KV_TOKENS   = 512;
constexpr int NUM_SEQS        = 1;
constexpr int Q_LEN           = 1;
constexpr int NUM_BLOCKS      = NUM_KV_TOKENS / BLOCK_SIZE;
constexpr int N_CENTROIDS_T4  = 16;
constexpr int BYTES_PER_FP8_BLOCK = 162;

// FP8 byte packer (matches the smoke test — see test_aiter_turbo_fp8_smoke.cpp
// for the format spec).
static float e4m3_byte_to_fp32(uint8_t b) {
    int sign = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    float v = (e == 0) ? (1.0f / 64.0f) * (m / 8.0f) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return sign ? -v : v;
}
static void quantize_block_bs256(const float *in, uint8_t *out, const uint8_t *centroids) {
    float scale = 0.0f;
    for (int i = 0; i < 256; ++i) scale = std::max(scale, std::fabs(in[i]));
    if (scale == 0.0f) { std::memset(out, 0, BYTES_PER_FP8_BLOCK); return; }
    _Float16 s16 = (_Float16) scale;
    float seff = (float) s16;
    if (seff == 0.0f) seff = scale;
    std::memcpy(out, &s16, 2);
    uint8_t *qs = out + 2, *signs = out + 130;
    std::memset(qs, 0, 128); std::memset(signs, 0, 32);
    float cv[N_CENTROIDS_T4];
    for (int k = 0; k < N_CENTROIDS_T4; ++k) cv[k] = e4m3_byte_to_fp32(centroids[k]);
    for (int i = 0; i < 256; ++i) {
        float v = in[i]; int s = v < 0 ? 1 : 0; float m = std::fabs(v) / seff;
        int best = 0; float be = std::fabs(m - cv[0]);
        for (int k = 1; k < N_CENTROIDS_T4; ++k) { float e = std::fabs(m - cv[k]); if (e < be) { best = k; be = e; } }
        if ((i & 1) == 0) qs[i / 2] = best; else qs[i / 2] |= (best << 4);
        signs[i / 8] |= (s & 1) << (i & 7);
    }
}

struct BenchResult {
    const char *name;
    double   mean_us;
    double   min_us;
    double   max_us;
    int      n_iters;
};

static int bench_cache_type(int cache_type, const char *name, int n_iters, BenchResult *out) {
    // ── K/V cache buffers — size depends on cache_type ────────────────────
    const bool is_fp8 = (cache_type == MT_AITER_CACHE_TURBO4_FP8);
    const size_t kv_bytes = is_fp8
        ? (size_t) NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK
        : (size_t) NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * sizeof(_Float16);

    // Random fp32 data, used to populate either FP8 packed or F16 buffers.
    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> k_fp32(NUM_KV_TOKENS * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> v_fp32(NUM_KV_TOKENS * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> q_fp32(NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE);
    for (auto &v : k_fp32) v = dist(rng);
    for (auto &v : v_fp32) v = dist(rng);
    for (auto &v : q_fp32) v = dist(rng);

    std::vector<uint8_t> kv_host_k(kv_bytes, 0), kv_host_v(kv_bytes, 0);
    if (is_fp8) {
        const uint8_t *ck = mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3;
        const uint8_t *cv = mt_turbo4_fp8_centroids_qwen35_4b_bs256_v_L3;
        for (int t = 0; t < NUM_KV_TOKENS; ++t) {
            for (int h = 0; h < NUM_KV_HEADS; ++h) {
                int blk = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
                size_t off = ((size_t) blk * BLOCK_SIZE * NUM_KV_HEADS + tok * NUM_KV_HEADS + h) * BYTES_PER_FP8_BLOCK;
                quantize_block_bs256(&k_fp32[(t * NUM_KV_HEADS + h) * HEAD_SIZE], &kv_host_k[off], ck);
                quantize_block_bs256(&v_fp32[(t * NUM_KV_HEADS + h) * HEAD_SIZE], &kv_host_v[off], cv);
            }
        }
    } else {
        // F16 path: AITER layout [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE] fp16
        _Float16 *kp = reinterpret_cast<_Float16 *>(kv_host_k.data());
        _Float16 *vp = reinterpret_cast<_Float16 *>(kv_host_v.data());
        for (int t = 0; t < NUM_KV_TOKENS; ++t) {
            int blk = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
            for (int h = 0; h < NUM_KV_HEADS; ++h) {
                size_t off = (((size_t) blk * BLOCK_SIZE + tok) * NUM_KV_HEADS + h) * HEAD_SIZE;
                for (int d = 0; d < HEAD_SIZE; ++d) {
                    kp[off + d] = (_Float16) k_fp32[(t * NUM_KV_HEADS + h) * HEAD_SIZE + d];
                    vp[off + d] = (_Float16) v_fp32[(t * NUM_KV_HEADS + h) * HEAD_SIZE + d];
                }
            }
        }
    }

    // GPU upload
    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_q,   q_fp32.size() * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_k,   kv_bytes));
    HIP_CHECK(hipMalloc(&d_v,   kv_bytes));
    HIP_CHECK(hipMalloc(&d_out, q_fp32.size() * sizeof(_Float16)));

    std::vector<_Float16> q_fp16(q_fp32.size());
    for (size_t i = 0; i < q_fp32.size(); ++i) q_fp16[i] = (_Float16) q_fp32[i];
    HIP_CHECK(hipMemcpy(d_q, q_fp16.data(), q_fp16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, kv_host_k.data(), kv_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, kv_host_v.data(), kv_bytes, hipMemcpyHostToDevice));

    constexpr int NUM_SEGMENTS = 128;
    size_t SEGM_OUT_BYTES = (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * NUM_SEGMENTS * HEAD_SIZE * sizeof(float);
    size_t SEGM_MAX_BYTES = (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * NUM_SEGMENTS * sizeof(float);
    void *d_segm_out = nullptr, *d_segm_max = nullptr, *d_segm_expsum = nullptr;
    HIP_CHECK(hipMalloc(&d_segm_out, SEGM_OUT_BYTES));
    HIP_CHECK(hipMalloc(&d_segm_max, SEGM_MAX_BYTES));
    HIP_CHECK(hipMalloc(&d_segm_expsum, SEGM_MAX_BYTES));

    std::vector<int32_t> h_block_tables(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) h_block_tables[i] = i;
    std::vector<int32_t> h_seq_lens = {NUM_KV_TOKENS};
    std::vector<int32_t> h_query_start_len = {0, Q_LEN};
    int32_t *d_block_tables, *d_seq_lens, *d_query_start_len;
    HIP_CHECK(hipMalloc(&d_block_tables, h_block_tables.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_seq_lens, sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_query_start_len, 2 * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_block_tables, h_block_tables.data(), h_block_tables.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq_lens, h_seq_lens.data(), sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_query_start_len, h_query_start_len.data(), 2 * sizeof(int32_t), hipMemcpyHostToDevice));

    float h_one = 1.0f;
    float *d_ones; HIP_CHECK(hipMalloc(&d_ones, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_ones, &h_one, sizeof(float), hipMemcpyHostToDevice));

    uint8_t *d_ck = nullptr, *d_cv = nullptr;
    if (is_fp8) {
        HIP_CHECK(hipMalloc(&d_ck, N_CENTROIDS_T4));
        HIP_CHECK(hipMalloc(&d_cv, N_CENTROIDS_T4));
        HIP_CHECK(hipMemcpy(d_ck, mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3, N_CENTROIDS_T4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_cv, mt_turbo4_fp8_centroids_qwen35_4b_bs256_v_L3, N_CENTROIDS_T4, hipMemcpyHostToDevice));
    }

    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = HEAD_SIZE;
    shape.num_q_heads  = NUM_Q_HEADS;
    shape.num_kv_heads = NUM_KV_HEADS;
    shape.block_size   = BLOCK_SIZE;
    shape.cache_type   = cache_type;

    mt_aiter_uattn_args_t args {};
    args.shape           = shape;
    args.q               = d_q;
    args.k_cache         = d_k;
    args.v_cache         = d_v;
    args.out             = d_out;
    args.segm_output     = d_segm_out;
    args.segm_max        = d_segm_max;
    args.segm_expsum     = d_segm_expsum;
    args.block_tables    = d_block_tables;
    args.seq_lens        = d_seq_lens;
    args.query_start_len = d_query_start_len;
    args.q_descale       = d_ones;
    args.k_descale       = d_ones;
    args.v_descale       = d_ones;
    args.out_scale       = d_ones;
    args.centroids_k     = d_ck;
    args.centroids_v     = d_cv;
    args.scale           = 1.0f / std::sqrt((float) HEAD_SIZE);
    args.num_seqs        = NUM_SEQS;
    args.num_q_tokens    = NUM_SEQS * Q_LEN;
    args.block_table_stride = NUM_BLOCKS;
    args.q_stride_0      = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    args.output_stride_0 = args.q_stride_0;
    args.k_stride_0      = (int64_t) BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_1      = (int64_t) NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_2      = (int64_t) HEAD_SIZE;
    args.v_stride_0      = args.k_stride_0;
    args.v_stride_1      = args.k_stride_1;
    args.v_stride_2      = args.k_stride_2;

    // Warmup (compiles + populates cache)
    for (int i = 0; i < 5; ++i) {
        hipError_t e = mt_aiter_unified_attn(0, &args);
        if (e != hipSuccess) { fprintf(stderr, "warmup launch failed (%s): %s\n", name, hipGetErrorString(e)); return 1; }
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Per-iteration timing — each iteration its own event pair so we can
    // capture min/max/mean instead of just the bulk total.
    double sum_us = 0.0, min_us = 1e30, max_us = 0.0;
    for (int i = 0; i < n_iters; ++i) {
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0));
        HIP_CHECK(hipEventCreate(&e1));
        HIP_CHECK(hipEventRecord(e0));
        hipError_t e = mt_aiter_unified_attn(0, &args);
        if (e != hipSuccess) { fprintf(stderr, "bench launch failed: %s\n", hipGetErrorString(e)); return 1; }
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        double us = ms * 1000.0;
        sum_us += us;
        if (us < min_us) min_us = us;
        if (us > max_us) max_us = us;
        HIP_CHECK(hipEventDestroy(e0));
        HIP_CHECK(hipEventDestroy(e1));
    }
    out->name    = name;
    out->mean_us = sum_us / n_iters;
    out->min_us  = min_us;
    out->max_us  = max_us;
    out->n_iters = n_iters;

    // Cleanup (suppress nodiscard warnings)
    (void) hipFree(d_q); (void) hipFree(d_k); (void) hipFree(d_v); (void) hipFree(d_out);
    (void) hipFree(d_segm_out); (void) hipFree(d_segm_max); (void) hipFree(d_segm_expsum);
    (void) hipFree(d_block_tables); (void) hipFree(d_seq_lens); (void) hipFree(d_query_start_len);
    (void) hipFree(d_ones);
    if (d_ck) (void) hipFree(d_ck);
    if (d_cv) (void) hipFree(d_cv);
    return 0;
}

int main(int argc, char **argv) {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop {};
    (void) hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);
    fprintf(stderr, "# workload: decode, NUM_KV_TOKENS=%d, GQA=%d/%d, HEAD=%d, Q_LEN=%d\n",
            NUM_KV_TOKENS, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, Q_LEN);
    fprintf(stderr, "# path: 3D split-K decode (avg_q_len=%d < BLOCK_Q=%d)\n",
            Q_LEN, MT_AITER_UATTN_BLOCK_Q);

    // The AITER handle cache pins ONE shape per process (cache_type included),
    // so this binary benches one cache type per invocation. Driver script runs
    // it twice and compares the outputs.
    const std::string mode = (argc > 1) ? argv[1] : "f16";
    const int N_ITERS = (argc > 2) ? atoi(argv[2]) : 100;

    BenchResult r {};
    if (mode == "f16") {
        if (bench_cache_type(MT_AITER_CACHE_F16, "F16", N_ITERS, &r) != 0) return 1;
    } else if (mode == "fp8") {
        if (bench_cache_type(MT_AITER_CACHE_TURBO4_FP8, "TURBO4_FP8", N_ITERS, &r) != 0) return 1;
    } else {
        fprintf(stderr, "usage: %s [f16|fp8] [n_iters]\n", argv[0]);
        return 2;
    }

    // KV cache footprint (informational — same regardless of mode)
    const size_t f16_kv_bytes = (size_t) NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * sizeof(_Float16) * 2;
    const size_t fp8_kv_bytes = (size_t) NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK * 2;

    fprintf(stderr, "\n=== %s LATENCY (%d iters) ===\n", r.name, r.n_iters);
    fprintf(stderr, "  mean=%.2f us   min=%.2f us   max=%.2f us\n",
            r.mean_us, r.min_us, r.max_us);
    fprintf(stderr, "  KV footprint (K+V): F16=%zu, FP8=%zu (%.2fx smaller)\n",
            f16_kv_bytes, fp8_kv_bytes, (double) f16_kv_bytes / (double) fp8_kv_bytes);

    // Machine-readable line for driver script to grep
    printf("MODE=%s MEAN_US=%.3f MIN_US=%.3f MAX_US=%.3f\n",
           r.name, r.mean_us, r.min_us, r.max_us);

    return 0;
}
