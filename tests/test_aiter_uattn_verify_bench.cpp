// MAD-2026-09-12 dispatch-fix: real-shape verify-geometry bench + correctness
// test for kernel_unified_attention_3d vs the (old, pre-fix-default) 2D base
// kernel at the production MTP/DFlash "verify" decode shapes described in
// ~/ds4-runs/tp27b/tp-spec-ladder-0911/decode-depth-scaling-0912.txt and
// dispatch-fix-0912.txt.
//
// Modeled on tests/test_aiter_uattn_prefill_bench.cpp (shape constants, FP8
// packer, timing loop) and tests/test_aiter_fp8_predequant.cpp (the
// spawn-a-child-process-per-config pattern, needed because both
// MT_AITER_UATTN_FORCE_2D and MT_AITER_FP8_LOADER_V2/etc. are read ONCE per
// process — see mt_aiter_uattn_should_use_2d() in
// ggml/src/ggml-cuda/aiter-integration/wrappers/mt_aiter_unified_attn.cpp).
//
// Per combination of (num_seqs, num_q_tokens_per_seq, seq_len, cache_type)
// this spawns TWO children — one with MT_AITER_UATTN_FORCE_2D=0 (forces the
// new/kept 3D split-K path; for num_q_tokens_per_seq>1 that is the new
// ALL_DECODE=0 handle added by this patch), one with
// MT_AITER_UATTN_FORCE_2D=1 (forces the pre-fix 2D base kernel, i.e. what
// production always ran for these shapes before this patch) — then
// byte-diffs their output tensors (expect only fp32-reduction-order-level
// numerical drift, not a real correctness difference) and prints each side's
// mean/min/max per-launch latency.
//
// Does NOT run automatically as part of this task (per the "do not build,
// do not run" constraint on the agent that wrote it) — the operator builds
// and runs it. See the two hipcc command lines at the bottom of this header
// comment / in the report this patch's task asked for
// (~/ds4-runs/tp27b/tp-spec-ladder-0911/dispatch-fix-0912.txt).
//
// Build (same recipe as the other aiter-integration test binaries):
//   hipcc --offload-arch=gfx1201 --offload-arch=gfx1030 -O2 \
//       -I ggml/src/ggml-cuda/aiter-integration/wrappers \
//       tests/test_aiter_uattn_verify_bench.cpp -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin -o /tmp/test_aiter_uattn_verify_bench
//
// Usage:
//   test_aiter_uattn_verify_bench                       # full production matrix, f16 + fp8
//   test_aiter_uattn_verify_bench <f16|fp8> <num_seqs> <q_per_seq> <seq_len> [iters]
//   test_aiter_uattn_verify_bench --child <f16|fp8> <num_seqs> <q_per_seq> <seq_len> <iters> <out.bin> <timing.txt>

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
#include <unistd.h>

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
        return 1; \
    } \
} while(0)

// ── Production TP-shape constants (Qwen3.8-27B, tsa 1,1) — same as
//    tests/test_aiter_uattn_prefill_bench.cpp / test_aiter_fp8_predequant.cpp.
constexpr int HEAD_SIZE            = 256;
constexpr int NUM_Q_HEADS          = 12;
constexpr int NUM_KV_HEADS         = 2;   // GQA = 6
constexpr int BLOCK_SIZE           = 16;  // paged KV block size (tokens) — see prefill bench header comment
constexpr int N_CENTROIDS_T4       = 16;
constexpr int BYTES_PER_FP8_BLOCK  = 162; // 2 (fp16 scale) + 128 (4-bit idx) + 32 (signs)

// ── FP8 byte packer — copied verbatim from the other two aiter-integration
// benches/tests (same layout contract, deterministic seed). ────────────────
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

// Runs N_ITERS mt_aiter_unified_attn() calls at the given verify geometry,
// dumps the LAST call's raw f16 output to out_path, and appends a
// machine-readable timing line to timing_path. Reads MT_AITER_UATTN_FORCE_2D
// from the environment (set by the parent before exec) — does not set it
// itself, since the predicate caches it once per process (see
// mt_aiter_uattn_should_use_2d() in mt_aiter_unified_attn.cpp).
static int run_and_dump(const std::string &mode, int num_seqs, int q_per_seq, int seq_len,
                         int n_iters, const char *out_path, const char *timing_path) {
    const bool is_fp8 = (mode == "fp8");
    const int  num_blocks_per_seq = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int  seq_len_padded     = num_blocks_per_seq * BLOCK_SIZE;
    const int  num_blocks_total   = num_blocks_per_seq * num_seqs;   // distinct block range per seq
    const int  num_q_tokens       = num_seqs * q_per_seq;

    const size_t kv_bytes = is_fp8
        ? (size_t) num_blocks_total * BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK
        : (size_t) num_blocks_total * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * sizeof(_Float16);

    std::mt19937 rng(2026);  // fixed seed: both children (3D / forced-2D) must see identical inputs
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> k_fp32((size_t) seq_len_padded * num_seqs * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> v_fp32(k_fp32.size());
    std::vector<float> q_fp32((size_t) num_q_tokens * NUM_Q_HEADS * HEAD_SIZE);
    for (auto &v : k_fp32) v = dist(rng);
    for (auto &v : v_fp32) v = dist(rng);
    for (auto &v : q_fp32) v = dist(rng);

    std::vector<uint8_t> kv_host_k(kv_bytes, 0), kv_host_v(kv_bytes, 0);
    const uint8_t *ck = mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3;
    const uint8_t *cv = mt_turbo4_fp8_centroids_qwen35_4b_bs256_v_L3;

    // Layout: block_tables row s owns physical blocks
    // [s*num_blocks_per_seq, (s+1)*num_blocks_per_seq) — independent KV ranges
    // per sequence, matching how distinct live sequences never share blocks
    // in production.
    for (int s = 0; s < num_seqs; ++s) {
        for (int t = 0; t < seq_len_padded; ++t) {
            const int blk_local = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
            const int blk_global = s * num_blocks_per_seq + blk_local;
            const size_t src = ((size_t) s * seq_len_padded + t) * NUM_KV_HEADS * HEAD_SIZE;
            for (int h = 0; h < NUM_KV_HEADS; ++h) {
                if (is_fp8) {
                    size_t off = ((size_t) blk_global * BLOCK_SIZE * NUM_KV_HEADS + tok * NUM_KV_HEADS + h) * BYTES_PER_FP8_BLOCK;
                    quantize_block_bs256(&k_fp32[src + (size_t) h * HEAD_SIZE], &kv_host_k[off], ck);
                    quantize_block_bs256(&v_fp32[src + (size_t) h * HEAD_SIZE], &kv_host_v[off], cv);
                } else {
                    _Float16 *kp = reinterpret_cast<_Float16 *>(kv_host_k.data());
                    _Float16 *vp = reinterpret_cast<_Float16 *>(kv_host_v.data());
                    size_t off = ((size_t) blk_global * BLOCK_SIZE + tok) * NUM_KV_HEADS * HEAD_SIZE + (size_t) h * HEAD_SIZE;
                    for (int d = 0; d < HEAD_SIZE; ++d) {
                        kp[off + d] = (_Float16) k_fp32[src + (size_t) h * HEAD_SIZE + d];
                        vp[off + d] = (_Float16) v_fp32[src + (size_t) h * HEAD_SIZE + d];
                    }
                }
            }
        }
    }

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
    shape.cache_type   = is_fp8 ? MT_AITER_CACHE_TURBO4_FP8 : MT_AITER_CACHE_F16;

    void *d_segm_out = nullptr, *d_segm_max = nullptr, *d_segm_expsum = nullptr;
    HIP_CHECK(hipMalloc(&d_segm_out,    mt_aiter_uattn_segm_output_bytes(&shape, num_q_tokens)));
    HIP_CHECK(hipMalloc(&d_segm_max,    mt_aiter_uattn_segm_max_bytes(&shape, num_q_tokens)));
    HIP_CHECK(hipMalloc(&d_segm_expsum, mt_aiter_uattn_segm_expsum_bytes(&shape, num_q_tokens)));

    std::vector<int32_t> h_block_tables((size_t) num_seqs * num_blocks_per_seq);
    for (int s = 0; s < num_seqs; ++s)
        for (int b = 0; b < num_blocks_per_seq; ++b)
            h_block_tables[(size_t) s * num_blocks_per_seq + b] = s * num_blocks_per_seq + b;

    std::vector<int32_t> h_seq_lens(num_seqs, seq_len);
    std::vector<int32_t> h_query_start_len(num_seqs + 1);
    for (int s = 0; s <= num_seqs; ++s) h_query_start_len[s] = s * q_per_seq;

    int32_t *d_block_tables, *d_seq_lens, *d_query_start_len;
    HIP_CHECK(hipMalloc(&d_block_tables, h_block_tables.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_seq_lens, h_seq_lens.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_query_start_len, h_query_start_len.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_block_tables, h_block_tables.data(), h_block_tables.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq_lens, h_seq_lens.data(), h_seq_lens.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_query_start_len, h_query_start_len.data(), h_query_start_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    float h_one = 1.0f;
    float *d_ones; HIP_CHECK(hipMalloc(&d_ones, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_ones, &h_one, sizeof(float), hipMemcpyHostToDevice));

    uint8_t *d_ck = nullptr, *d_cv = nullptr;
    if (is_fp8) {
        HIP_CHECK(hipMalloc(&d_ck, N_CENTROIDS_T4));
        HIP_CHECK(hipMalloc(&d_cv, N_CENTROIDS_T4));
        HIP_CHECK(hipMemcpy(d_ck, ck, N_CENTROIDS_T4, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_cv, cv, N_CENTROIDS_T4, hipMemcpyHostToDevice));
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
    args.num_seqs        = num_seqs;
    args.num_q_tokens    = num_q_tokens;
    args.block_table_stride = num_blocks_per_seq;
    // MAD-2026-09-12 predequant-scratch: each seq row s owns physical blocks
    // [s*num_blocks_per_seq, (s+1)*num_blocks_per_seq) with no padding
    // (block_table_stride == num_blocks_per_seq, h_block_tables[s][b] =
    // s*num_blocks_per_seq+b) — the compacted mapping (prefix[s] =
    // s*num_blocks_per_seq) is again the identity, so reusing d_block_tables
    // directly reproduces the pre-fix scratch layout exactly.
    args.scratch_block_tables = is_fp8 ? d_block_tables : nullptr;
    args.num_scratch_blocks   = is_fp8 ? num_blocks_total : 0;
    args.q_stride_0      = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    args.output_stride_0 = args.q_stride_0;
    args.k_stride_0      = (int64_t) BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_1      = (int64_t) NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_2      = (int64_t) HEAD_SIZE;
    args.v_stride_0      = args.k_stride_0;
    args.v_stride_1      = args.k_stride_1;
    args.v_stride_2      = args.k_stride_2;

    const char * f2d = std::getenv("MT_AITER_UATTN_FORCE_2D");
    fprintf(stderr, "[child pid=%d] MT_AITER_UATTN_FORCE_2D=%s mode=%s num_seqs=%d q_per_seq=%d seq_len=%d launching...\n",
            (int) getpid(), f2d ? f2d : "(unset)", mode.c_str(), num_seqs, q_per_seq, seq_len);

    // Warmup (compiles + populates the runtime-compiler kernel cache).
    for (int i = 0; i < 3; ++i) {
        hipError_t e = mt_aiter_unified_attn(0, &args);
        if (e != hipSuccess) { fprintf(stderr, "warmup launch failed: %s\n", hipGetErrorString(e)); return 1; }
    }
    HIP_CHECK(hipDeviceSynchronize());

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

    std::vector<_Float16> h_out(q_fp32.size());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    FILE *f = std::fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "could not open %s for writing\n", out_path); return 1; }
    std::fwrite(h_out.data(), sizeof(_Float16), h_out.size(), f);
    std::fclose(f);

    FILE *tf = std::fopen(timing_path, "w");
    if (!tf) { fprintf(stderr, "could not open %s for writing\n", timing_path); return 1; }
    std::fprintf(tf, "MEAN_US=%.3f MIN_US=%.3f MAX_US=%.3f N_ITERS=%d\n", sum_us / n_iters, min_us, max_us, n_iters);
    std::fclose(tf);

    (void) hipFree(d_q); (void) hipFree(d_k); (void) hipFree(d_v); (void) hipFree(d_out);
    (void) hipFree(d_segm_out); (void) hipFree(d_segm_max); (void) hipFree(d_segm_expsum);
    (void) hipFree(d_block_tables); (void) hipFree(d_seq_lens); (void) hipFree(d_query_start_len);
    (void) hipFree(d_ones);
    if (d_ck) (void) hipFree(d_ck);
    if (d_cv) (void) hipFree(d_cv);
    return 0;
}

static bool read_file(const char *path, std::vector<_Float16> *out) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out->resize((size_t) sz / sizeof(_Float16));
    size_t got = std::fread(out->data(), sizeof(_Float16), out->size(), f);
    std::fclose(f);
    return got == out->size();
}

static double read_mean_us(const char *timing_path) {
    FILE *f = std::fopen(timing_path, "r");
    if (!f) return -1.0;
    double mean = -1.0;
    if (std::fscanf(f, "MEAN_US=%lf", &mean) != 1) mean = -1.0;
    std::fclose(f);
    return mean;
}

// Runs one (mode, num_seqs, q_per_seq, seq_len) combination: spawns the
// forced-3D and forced-2D children, compares outputs, reports timing.
// Returns 0 on pass (outputs agree within fp32-reduction-order tolerance),
// 1 on any failure (launch error, size mismatch, or diff over tolerance).
static int run_combo(const std::string &argv0, const std::string &mode,
                      int num_seqs, int q_per_seq, int seq_len, int n_iters) {
    char out3d[256], out2d[256], t3d[256], t2d[256];
    std::snprintf(out3d, sizeof(out3d), "/tmp/uattn_verify_3d_%s_%d_%d_%d.bin", mode.c_str(), num_seqs, q_per_seq, seq_len);
    std::snprintf(out2d, sizeof(out2d), "/tmp/uattn_verify_2d_%s_%d_%d_%d.bin", mode.c_str(), num_seqs, q_per_seq, seq_len);
    std::snprintf(t3d, sizeof(t3d), "/tmp/uattn_verify_3d_%s_%d_%d_%d.timing", mode.c_str(), num_seqs, q_per_seq, seq_len);
    std::snprintf(t2d, sizeof(t2d), "/tmp/uattn_verify_2d_%s_%d_%d_%d.timing", mode.c_str(), num_seqs, q_per_seq, seq_len);

    char cmd[1024];
    std::snprintf(cmd, sizeof(cmd),
        "MT_AITER_UATTN_FORCE_2D=0 \"%s\" --child %s %d %d %d %d \"%s\" \"%s\"",
        argv0.c_str(), mode.c_str(), num_seqs, q_per_seq, seq_len, n_iters, out3d, t3d);
    int rc3d = std::system(cmd);
    std::snprintf(cmd, sizeof(cmd),
        "MT_AITER_UATTN_FORCE_2D=1 \"%s\" --child %s %d %d %d %d \"%s\" \"%s\"",
        argv0.c_str(), mode.c_str(), num_seqs, q_per_seq, seq_len, n_iters, out2d, t2d);
    int rc2d = std::system(cmd);

    fprintf(stderr, "\n=== %s num_seqs=%d q_per_seq=%d seq_len=%d ===\n", mode.c_str(), num_seqs, q_per_seq, seq_len);
    if (rc3d != 0 || rc2d != 0) {
        fprintf(stderr, "FAIL: child process error (3d rc=%d, 2d rc=%d)\n", rc3d, rc2d);
        printf("FAIL mode=%s num_seqs=%d q_per_seq=%d seq_len=%d child_error\n", mode.c_str(), num_seqs, q_per_seq, seq_len);
        return 1;
    }

    std::vector<_Float16> h3d, h2d;
    if (!read_file(out3d, &h3d) || !read_file(out2d, &h2d)) {
        fprintf(stderr, "FAIL: could not read child output files\n");
        return 1;
    }
    if (h3d.size() != h2d.size()) {
        fprintf(stderr, "FAIL: output size mismatch (%zu vs %zu)\n", h3d.size(), h2d.size());
        return 1;
    }

    double max_abs_diff = 0.0, max_rel_diff = 0.0;
    for (size_t i = 0; i < h3d.size(); ++i) {
        double fa = (double) h3d[i], fb = (double) h2d[i];
        double ad = std::fabs(fa - fb);
        max_abs_diff = std::max(max_abs_diff, ad);
        double denom = std::max(std::fabs(fa), 1e-6);
        max_rel_diff = std::max(max_rel_diff, ad / denom);
    }

    // Tolerance: this must be fp32-reduction-order noise only (different
    // segment counts / accumulation order between the 3D split-K path and
    // the 2D single-pass path), never a real correctness difference. F16
    // storage (~3 decimal digits) plus multi-segment online-softmax
    // re-association gives a looser but still tight bound than pure
    // same-algorithm rounding; FP8 additionally carries the (identical,
    // deterministic) centroid-LUT quantization noise on both sides, so its
    // tolerance is not tightened further here.
    const double tol_abs = (mode == "fp8") ? 5e-2 : 8e-3;
    const bool pass = max_abs_diff <= tol_abs;

    double mean3d = read_mean_us(t3d), mean2d = read_mean_us(t2d);
    fprintf(stderr, "  elements=%zu max_abs_diff=%.6g max_rel_diff=%.6g tol_abs=%.6g -> %s\n",
            h3d.size(), max_abs_diff, max_rel_diff, tol_abs, pass ? "PASS" : "FAIL");
    fprintf(stderr, "  3D (forced)     mean=%.2f us\n", mean3d);
    fprintf(stderr, "  2D-base (forced) mean=%.2f us   speedup(2D/3D)=%.2fx\n",
            mean2d, (mean3d > 0.0) ? mean2d / mean3d : -1.0);

    printf("%s mode=%s num_seqs=%d q_per_seq=%d seq_len=%d max_abs_diff=%.6g "
           "3D_MEAN_US=%.3f 2D_MEAN_US=%.3f\n",
           pass ? "PASS" : "FAIL", mode.c_str(), num_seqs, q_per_seq, seq_len,
           max_abs_diff, mean3d, mean2d);

    return pass ? 0 : 1;
}

int main(int argc, char **argv) {
    if (argc >= 9 && std::string(argv[1]) == "--child") {
        HIP_CHECK(hipSetDevice(0));
        const std::string mode = argv[2];
        const int num_seqs  = std::atoi(argv[3]);
        const int q_per_seq = std::atoi(argv[4]);
        const int seq_len   = std::atoi(argv[5]);
        const int n_iters   = std::atoi(argv[6]);
        return run_and_dump(mode, num_seqs, q_per_seq, seq_len, n_iters, argv[7], argv[8]);
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop {};
    (void) hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s, CUs=%d)\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);
    fprintf(stderr, "# shape: GQA=%d/%d HEAD=%d BLOCK_SIZE=%d (paged)\n", NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE);

    // Single-combo mode: test_aiter_uattn_verify_bench <f16|fp8> <num_seqs> <q_per_seq> <seq_len> [iters]
    if (argc >= 5) {
        const std::string mode = argv[1];
        const int num_seqs  = std::atoi(argv[2]);
        const int q_per_seq = std::atoi(argv[3]);
        const int seq_len   = std::atoi(argv[4]);
        const int n_iters   = (argc > 5) ? std::atoi(argv[5]) : 20;
        return run_combo(argv[0], mode, num_seqs, q_per_seq, seq_len, n_iters);
    }

    // Default: full production verify matrix (decode-depth-scaling-0912.txt /
    // this patch's task spec) — num_seqs in {1,4}, num_q_tokens per seq in
    // {1 (plain decode), 5 (MTP verify n4), 8 (DFlash verify)}, seq_len in
    // {8192, 32768, 65536}, both cache types.
    const int    num_seqs_list[]  = { 1, 4 };
    const int    q_per_seq_list[] = { 1, 5, 8 };
    const int    seq_len_list[]   = { 8192, 32768, 65536 };
    const char * modes[]          = { "f16", "fp8" };
    const int    n_iters          = 20;

    int n_fail = 0, n_total = 0;
    for (const char * mode : modes) {
        for (int num_seqs : num_seqs_list) {
            for (int q_per_seq : q_per_seq_list) {
                for (int seq_len : seq_len_list) {
                    ++n_total;
                    if (run_combo(argv[0], mode, num_seqs, q_per_seq, seq_len, n_iters) != 0) {
                        ++n_fail;
                    }
                }
            }
        }
    }

    fprintf(stderr, "\n=== SUMMARY: %d/%d combinations passed ===\n", n_total - n_fail, n_total);
    return n_fail == 0 ? 0 : 1;
}
