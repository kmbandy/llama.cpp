// MAD-2026-09-11: PREFILL-shape microbenchmark of kernel_unified_attention_2d
// through the real mt_aiter_unified_attn() wrapper, at the production TP
// shape (Qwen3.8-27B, tsa 1,1): num_q_heads=12, num_kv_heads=2 (GQA=6),
// head_size=256, paged BLOCK_SIZE=16.
//
// NOTE on "BS=256": the turbo4_fp8 cache-type suffix "_BS256" names the FP8
// QUANTIZATION group size (= head_size — quantize_block_bs256() below packs
// one full 256-element head row per K/V token per (token, kv_head)), not the
// paged KV-cache block_size. The paged block_size is a separate shape field
// (mt_aiter_uattn_shape_t::block_size) and production wires ONLY block_size=16
// for both F16 and TURBO4_FP8 at head_size=256 — see the
// "GGML_ABORT(...block_size=%d) instantiation" guards in
// ggml/src/ggml-cuda/mt_pagedattn_aiter.cu (F16: head_size==256 && block_size
// ==16; TURBO4_FP8 scatter: only head_size==256 && block_size==16 wired).
// This bench therefore uses BLOCK_SIZE=16 (paged) for BOTH cache types, not
// 256 — using paged block_size=256 would exercise a shape production never
// runs and that isn't even wired for turbo4_fp8's scatter kernel.
//
// Shape is chosen so avg_q_len (=Q_LEN, one sequence) is >= the 2D
// large-prefill threshold (256), so ensure_initialized() picks the "2D large
// prefill" spec: BLOCK_M = next_pow2(BLOCK_Q_LARGE * GQA) = next_pow2(8*6) =
// 64, BLOCK_Q = BLOCK_M / GQA = 64/6 = 10 (floor; see
// mt_aiter_uattn_block_m_large / _block_q_large in mt_aiter_unified_attn.h).
// This is the SAME spec the wrapper's own comments describe as "the kernel
// that carries prefill under TP with the 6900XT holding a fraction of the KV
// heads" — i.e. this bench IS the production prefill dispatch path, not an
// approximation of it.
//
// Centroid LUT: no Qwen3.8-27B calibration header exists in-tree yet (only
// ggml/src/ggml-cuda/aiter-integration/turbo_fp8_data/qwen35_4b_bs256_centroids.h,
// a 4B-model calibration). Timing/counters do not depend on centroid values
// (they only affect the FP8 dequant LUT contents, not control flow or memory
// traffic pattern), so this bench reuses the same qwen35_4b L3 K/V centroid
// tables the existing turbo_fp8 smoke/perf tests use. If/when a 27B
// calibration header lands, swap the #include and the two LUT symbol names
// below.
//
// Context modeling: NUM_KV_TOKENS is the FULL causal context length already
// resident in the paged KV cache (scatter of the current Q_LEN chunk's own
// K/V happens before this attention call, exactly as in production —
// mirrors the existing turbo_fp8 smoke/perf tests' seq_lens=[NUM_KV_TOKENS],
// query_start_len=[0, Q_LEN] convention: the Q_LEN queries are the LAST
// Q_LEN positions of the NUM_KV_TOKENS-token sequence).
//
// Respects (does not set) MT_AITER_FP8_LOADER_V2, MT_AITER_NUM_WARPS,
// MT_AITER_NUM_STAGES, MT_AITER_GFX1201_NUM_WARPS8 — set those in the
// environment before running this binary to A/B them.
//
// Build:
//   hipcc --offload-arch=gfx1201 --offload-arch=gfx1030 -O2 \
//       -I ggml/src/ggml-cuda/aiter-integration/wrappers \
//       tests/test_aiter_uattn_prefill_bench.cpp -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin -o /tmp/test_aiter_uattn_prefill_bench
//
// Usage:
//   test_aiter_uattn_prefill_bench [f16|fp8] [Q_LEN] [NUM_KV_TOKENS] [iters]
//   defaults:                        f16      1024    16384          20

#include "mt_aiter_unified_attn.h"
#include "../ggml/src/ggml-cuda/aiter-integration/turbo_fp8_data/qwen35_4b_bs256_centroids.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

// ── Production TP-shape constants (Qwen3.8-27B, tsa 1,1) ───────────────────
constexpr int HEAD_SIZE       = 256;
constexpr int NUM_Q_HEADS     = 12;
constexpr int NUM_KV_HEADS    = 2;             // GQA = 6
constexpr int BLOCK_SIZE      = 16;            // paged KV block size (tokens) — see header comment
constexpr int NUM_SEQS        = 1;
constexpr int N_CENTROIDS_T4  = 16;
constexpr int BYTES_PER_FP8_BLOCK = 162;        // 2 (fp16 scale) + 128 (4-bit idx) + 32 (signs)

// ── FP8 byte packer — mirrors tests/test_aiter_turbo_fp8_smoke.cpp exactly.
// Values are random and don't need to be numerically meaningful for a
// timing/counters bench; only the packed layout (scale + 4-bit indices +
// sign bits) needs to match what the kernel expects to read.
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

static int bench_cache_type(int cache_type, const char *name, int q_len, int num_kv_tokens,
                             int n_iters, BenchResult *out) {
    const bool is_fp8 = (cache_type == MT_AITER_CACHE_TURBO4_FP8);
    const int  num_blocks = (num_kv_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int  num_kv_tokens_padded = num_blocks * BLOCK_SIZE;

    const size_t kv_bytes = is_fp8
        ? (size_t) num_blocks * BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK
        : (size_t) num_blocks * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * sizeof(_Float16);

    // Random fp32 source data — values are irrelevant for timing/counters.
    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> k_fp32((size_t) num_kv_tokens_padded * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> v_fp32((size_t) num_kv_tokens_padded * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> q_fp32((size_t) NUM_SEQS * q_len * NUM_Q_HEADS * HEAD_SIZE);
    for (auto &v : k_fp32) v = dist(rng);
    for (auto &v : v_fp32) v = dist(rng);
    for (auto &v : q_fp32) v = dist(rng);

    std::vector<uint8_t> kv_host_k(kv_bytes, 0), kv_host_v(kv_bytes, 0);
    if (is_fp8) {
        const uint8_t *ck = mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3;
        const uint8_t *cv = mt_turbo4_fp8_centroids_qwen35_4b_bs256_v_L3;
        for (int t = 0; t < num_kv_tokens_padded; ++t) {
            for (int h = 0; h < NUM_KV_HEADS; ++h) {
                int blk = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
                size_t off = ((size_t) blk * BLOCK_SIZE * NUM_KV_HEADS + tok * NUM_KV_HEADS + h) * BYTES_PER_FP8_BLOCK;
                quantize_block_bs256(&k_fp32[((size_t) t * NUM_KV_HEADS + h) * HEAD_SIZE], &kv_host_k[off], ck);
                quantize_block_bs256(&v_fp32[((size_t) t * NUM_KV_HEADS + h) * HEAD_SIZE], &kv_host_v[off], cv);
            }
        }
    } else {
        // F16 path: AITER layout [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE] fp16
        _Float16 *kp = reinterpret_cast<_Float16 *>(kv_host_k.data());
        _Float16 *vp = reinterpret_cast<_Float16 *>(kv_host_v.data());
        for (int t = 0; t < num_kv_tokens_padded; ++t) {
            int blk = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
            for (int h = 0; h < NUM_KV_HEADS; ++h) {
                size_t off = (((size_t) blk * BLOCK_SIZE + tok) * NUM_KV_HEADS + h) * HEAD_SIZE;
                for (int d = 0; d < HEAD_SIZE; ++d) {
                    kp[off + d] = (_Float16) k_fp32[((size_t) t * NUM_KV_HEADS + h) * HEAD_SIZE + d];
                    vp[off + d] = (_Float16) v_fp32[((size_t) t * NUM_KV_HEADS + h) * HEAD_SIZE + d];
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

    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = HEAD_SIZE;
    shape.num_q_heads  = NUM_Q_HEADS;
    shape.num_kv_heads = NUM_KV_HEADS;
    shape.block_size   = BLOCK_SIZE;
    shape.cache_type   = cache_type;

    const int num_q_tokens = NUM_SEQS * q_len;

    // Workspace — sized via the wrapper's own helpers so this bench never
    // under-allocates even though the 2D-prefill launch path doesn't
    // actually touch segm_*; kept for parity with the args struct contract.
    void *d_segm_out = nullptr, *d_segm_max = nullptr, *d_segm_expsum = nullptr;
    HIP_CHECK(hipMalloc(&d_segm_out,    mt_aiter_uattn_segm_output_bytes(&shape, num_q_tokens)));
    HIP_CHECK(hipMalloc(&d_segm_max,    mt_aiter_uattn_segm_max_bytes(&shape, num_q_tokens)));
    HIP_CHECK(hipMalloc(&d_segm_expsum, mt_aiter_uattn_segm_expsum_bytes(&shape, num_q_tokens)));

    std::vector<int32_t> h_block_tables(num_blocks);
    for (int i = 0; i < num_blocks; ++i) h_block_tables[i] = i;
    // seq_lens = full causal context length (KV already resident, INCLUDING
    // this chunk's own K/V — scatter-before-attend, as in production).
    std::vector<int32_t> h_seq_lens = { num_kv_tokens };
    // query_start_len: one sequence, q_len queries — these are the LAST
    // q_len positions of the num_kv_tokens-token causal sequence.
    std::vector<int32_t> h_query_start_len = { 0, q_len };
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
    args.num_q_tokens    = num_q_tokens;
    args.block_table_stride = num_blocks;
    // MAD-2026-09-11 fp8-predequant: lets this existing fp8-vs-f16 bench
    // exercise the new gfx1030 2D-large pre-dequant path (see
    // mt_aiter_unified_attn.cpp). 0 for the f16 run — harmless, the wrapper
    // only reads num_blocks for cache_type == TURBO4_FP8_BS256.
    // MAD-2026-09-12 predequant-scratch: single seq, fully populated table
    // (block_table_stride == num_blocks, h_block_tables[i]=i) — the
    // compacted table is the identity, so reusing d_block_tables directly
    // reproduces the pre-fix scratch layout exactly.
    args.scratch_block_tables = is_fp8 ? d_block_tables : nullptr;
    args.num_scratch_blocks   = is_fp8 ? num_blocks : 0;
    args.q_stride_0      = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    args.output_stride_0 = args.q_stride_0;
    args.k_stride_0      = (int64_t) BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_1      = (int64_t) NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_2      = (int64_t) HEAD_SIZE;
    args.v_stride_0      = args.k_stride_0;
    args.v_stride_1      = args.k_stride_1;
    args.v_stride_2      = args.k_stride_2;

    // Warmup (compiles + populates the runtime-compiler kernel cache)
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

    const std::string mode   = (argc > 1) ? argv[1] : "f16";
    const int q_len           = (argc > 2) ? atoi(argv[2]) : 1024;
    const int num_kv_tokens   = (argc > 3) ? atoi(argv[3]) : 16384;
    const int n_iters         = (argc > 4) ? atoi(argv[4]) : 20;

    fprintf(stderr, "# workload: prefill, Q_LEN=%d, NUM_KV_TOKENS=%d, GQA=%d/%d, HEAD=%d, BLOCK_SIZE=%d (paged), causal\n",
            q_len, num_kv_tokens, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE);
    fprintf(stderr, "# expected path: 2D %s prefill (avg_q_len=%d vs BLOCK_Q=%d / LARGE_PREFILL_THRESHOLD=%d)\n",
            q_len >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD ? "LARGE" : "base",
            q_len, MT_AITER_UATTN_BLOCK_Q, MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD);

    BenchResult r {};
    if (mode == "f16") {
        if (bench_cache_type(MT_AITER_CACHE_F16, "F16", q_len, num_kv_tokens, n_iters, &r) != 0) return 1;
    } else if (mode == "fp8") {
        if (bench_cache_type(MT_AITER_CACHE_TURBO4_FP8, "TURBO4_FP8", q_len, num_kv_tokens, n_iters, &r) != 0) return 1;
    } else {
        fprintf(stderr, "usage: %s [f16|fp8] [Q_LEN] [NUM_KV_TOKENS] [iters]\n", argv[0]);
        return 2;
    }

    fprintf(stderr, "\n=== %s PREFILL LATENCY (%d iters) ===\n", r.name, r.n_iters);
    fprintf(stderr, "  mean=%.2f us   min=%.2f us   max=%.2f us\n",
            r.mean_us, r.min_us, r.max_us);

    // Machine-readable line for driver/counter scripts to grep.
    printf("MODE=%s Q_LEN=%d KV=%d MEAN_US=%.3f MIN_US=%.3f MAX_US=%.3f\n",
           r.name, q_len, num_kv_tokens, r.mean_us, r.min_us, r.max_us);

    return 0;
}
