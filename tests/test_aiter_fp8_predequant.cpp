// MAD-2026-09-11 fp8-predequant correctness test.
//
// Verifies that the gfx1030 2D-large-prefill fp8-predequant path
// (mt_aiter_unified_attn.cpp: dequant_turbo4_fp8_bs256_to_f16_2d + the F16
// shadow 2D-large launch) is BIT-IDENTICAL to the existing in-kernel fp8
// dequant path (kernel_unified_attention_2d's CACHE_TYPE=24/USE_FP8_WMMA=0
// branch) at the production TP shape: GQA 12/2, head_size=256, paged
// BLOCK_SIZE=16, Q_LEN=1024, NUM_KV_TOKENS=16384 — copied from
// tests/test_aiter_uattn_prefill_bench.cpp (same random fp8 packer, same
// shape constants).
//
// MT_AITER_FP8_PREDEQUANT is read ONCE per process (ensure_initialized()
// caches env-derived state in the per-device CachedHandles the first time a
// given device is used), so a single process cannot A/B the flag by calling
// setenv() between two mt_aiter_unified_attn() calls on the same device.
// Per the task's own guidance this test instead spawns itself twice via
// argv, once per MT_AITER_FP8_PREDEQUANT setting, each child dumping its
// raw f16 output tensor to a file; the parent then byte-compares the two
// files and reports PASS/FAIL + max abs diff (computed in f16->f32 space,
// even though a PASS is expected to be exact byte-for-byte).
//
// Build:
//   hipcc --offload-arch=gfx1030 -O2 \
//       -I ggml/src/ggml-cuda/aiter-integration/wrappers \
//       tests/test_aiter_fp8_predequant.cpp -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin -o /tmp/test_aiter_fp8_predequant
//
// Usage:
//   test_aiter_fp8_predequant                     # parent: runs both modes, compares
//   test_aiter_fp8_predequant --child <out.bin>    # child: single run, dumps output

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
//    tests/test_aiter_uattn_prefill_bench.cpp. ────────────────────────────
constexpr int HEAD_SIZE       = 256;
constexpr int NUM_Q_HEADS     = 12;
constexpr int NUM_KV_HEADS    = 2;             // GQA = 6
constexpr int BLOCK_SIZE      = 16;            // paged KV block size (tokens)
constexpr int NUM_SEQS        = 1;
constexpr int N_CENTROIDS_T4  = 16;
constexpr int BYTES_PER_FP8_BLOCK = 162;        // 2 (fp16 scale) + 128 (4-bit idx) + 32 (signs)
constexpr int Q_LEN           = 1024;
constexpr int NUM_KV_TOKENS   = 16384;

// ── FP8 byte packer — copied verbatim from test_aiter_uattn_prefill_bench.cpp
// (== test_aiter_turbo_fp8_smoke.cpp's packer). Deterministic per-run seed so
// both children (predequant=0, predequant=1) quantize the identical cache. ──
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

// Runs ONE mt_aiter_unified_attn(cache_type=TURBO4_FP8) call at the
// production shape and writes the raw f16 output tensor to `out_path`.
// Reads MT_AITER_FP8_PREDEQUANT from the environment (already set by the
// parent process before exec, or by the caller for a same-process smoke
// check) — does NOT set it itself, since the wrapper only reads env once
// per process/device and this function must not race that caching.
static int run_and_dump(const char *out_path) {
    const int num_blocks = (NUM_KV_TOKENS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int num_kv_tokens_padded = num_blocks * BLOCK_SIZE;
    const size_t kv_bytes = (size_t) num_blocks * BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK;

    std::mt19937 rng(2026);  // fixed seed: both children must quantize identically
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> k_fp32((size_t) num_kv_tokens_padded * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> v_fp32((size_t) num_kv_tokens_padded * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> q_fp32((size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE);
    for (auto &v : k_fp32) v = dist(rng);
    for (auto &v : v_fp32) v = dist(rng);
    for (auto &v : q_fp32) v = dist(rng);

    std::vector<uint8_t> kv_host_k(kv_bytes, 0), kv_host_v(kv_bytes, 0);
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
    shape.cache_type   = MT_AITER_CACHE_TURBO4_FP8;

    const int num_q_tokens = NUM_SEQS * Q_LEN;

    void *d_segm_out = nullptr, *d_segm_max = nullptr, *d_segm_expsum = nullptr;
    HIP_CHECK(hipMalloc(&d_segm_out,    mt_aiter_uattn_segm_output_bytes(&shape, num_q_tokens)));
    HIP_CHECK(hipMalloc(&d_segm_max,    mt_aiter_uattn_segm_max_bytes(&shape, num_q_tokens)));
    HIP_CHECK(hipMalloc(&d_segm_expsum, mt_aiter_uattn_segm_expsum_bytes(&shape, num_q_tokens)));

    std::vector<int32_t> h_block_tables(num_blocks);
    for (int i = 0; i < num_blocks; ++i) h_block_tables[i] = i;
    std::vector<int32_t> h_seq_lens = { NUM_KV_TOKENS };
    std::vector<int32_t> h_query_start_len = { 0, Q_LEN };
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
    HIP_CHECK(hipMalloc(&d_ck, N_CENTROIDS_T4));
    HIP_CHECK(hipMalloc(&d_cv, N_CENTROIDS_T4));
    HIP_CHECK(hipMemcpy(d_ck, ck, N_CENTROIDS_T4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cv, cv, N_CENTROIDS_T4, hipMemcpyHostToDevice));

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
    args.num_blocks      = num_blocks;  // MAD-2026-09-11: the field under test
    args.q_stride_0      = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    args.output_stride_0 = args.q_stride_0;
    args.k_stride_0      = (int64_t) BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_1      = (int64_t) NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_2      = (int64_t) HEAD_SIZE;
    args.v_stride_0      = args.k_stride_0;
    args.v_stride_1      = args.k_stride_1;
    args.v_stride_2      = args.k_stride_2;

    const char * pd_env = std::getenv("MT_AITER_FP8_PREDEQUANT");
    fprintf(stderr, "[child pid=%d] MT_AITER_FP8_PREDEQUANT=%s launching...\n",
            (int) getpid(), pd_env ? pd_env : "(unset)");

    hipError_t e = mt_aiter_unified_attn(0, &args);
    if (e != hipSuccess) { fprintf(stderr, "launch failed: %s\n", hipGetErrorString(e)); return 1; }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<_Float16> h_out(q_fp32.size());
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    FILE *f = std::fopen(out_path, "wb");
    if (!f) { fprintf(stderr, "could not open %s for writing\n", out_path); return 1; }
    std::fwrite(h_out.data(), sizeof(_Float16), h_out.size(), f);
    std::fclose(f);

    (void) hipFree(d_q); (void) hipFree(d_k); (void) hipFree(d_v); (void) hipFree(d_out);
    (void) hipFree(d_segm_out); (void) hipFree(d_segm_max); (void) hipFree(d_segm_expsum);
    (void) hipFree(d_block_tables); (void) hipFree(d_seq_lens); (void) hipFree(d_query_start_len);
    (void) hipFree(d_ones); (void) hipFree(d_ck); (void) hipFree(d_cv);
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

int main(int argc, char **argv) {
    if (argc >= 3 && std::string(argv[1]) == "--child") {
        HIP_CHECK(hipSetDevice(0));
        return run_and_dump(argv[2]);
    }

    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop {};
    (void) hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);
    fprintf(stderr, "# shape: GQA=%d/%d HEAD=%d BLOCK_SIZE=%d Q_LEN=%d NUM_KV_TOKENS=%d\n",
            NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE, BLOCK_SIZE, Q_LEN, NUM_KV_TOKENS);

    const std::string out0 = "/tmp/test_aiter_fp8_predequant_off.bin";
    const std::string out1 = "/tmp/test_aiter_fp8_predequant_on.bin";

    // MT_AITER_FP8_PREDEQUANT is read once per process (cached in the
    // per-device CachedHandles at first ensure_initialized() call), so each
    // mode gets its own child process rather than two calls in this process.
    std::string cmd0 = std::string("MT_AITER_FP8_PREDEQUANT=0 \"") + argv[0] + "\" --child \"" + out0 + "\"";
    std::string cmd1 = std::string("MT_AITER_FP8_PREDEQUANT=1 \"") + argv[0] + "\" --child \"" + out1 + "\"";

    fprintf(stderr, "# running predequant=0 (in-kernel fp8 dequant, today's path)...\n");
    int rc0 = std::system(cmd0.c_str());
    fprintf(stderr, "# running predequant=1 (dequant pre-pass + F16 2D-large shadow)...\n");
    int rc1 = std::system(cmd1.c_str());

    if (rc0 != 0 || rc1 != 0) {
        fprintf(stderr, "FAIL: child process error (rc0=%d rc1=%d)\n", rc0, rc1);
        return 1;
    }

    std::vector<_Float16> h0, h1;
    if (!read_file(out0.c_str(), &h0) || !read_file(out1.c_str(), &h1)) {
        fprintf(stderr, "FAIL: could not read child output files\n");
        return 1;
    }
    if (h0.size() != h1.size()) {
        fprintf(stderr, "FAIL: output size mismatch (%zu vs %zu)\n", h0.size(), h1.size());
        return 1;
    }

    size_t n_diff = 0;
    double max_abs_diff = 0.0;
    for (size_t i = 0; i < h0.size(); ++i) {
        uint16_t a, b;
        std::memcpy(&a, &h0[i], 2);
        std::memcpy(&b, &h1[i], 2);
        if (a != b) {
            ++n_diff;
            double fa = (double) h0[i], fb = (double) h1[i];
            max_abs_diff = std::max(max_abs_diff, std::fabs(fa - fb));
        }
    }

    fprintf(stderr, "# elements=%zu differing=%zu max_abs_diff=%.6g\n",
            h0.size(), n_diff, max_abs_diff);
    if (n_diff == 0) {
        fprintf(stderr, "PASS: predequant path is byte-identical to the in-kernel fp8 dequant path.\n");
        printf("PASS elements=%zu max_abs_diff=%.6g\n", h0.size(), max_abs_diff);
        return 0;
    } else {
        fprintf(stderr, "FAIL: %zu / %zu elements differ.\n", n_diff, h0.size());
        printf("FAIL elements=%zu differing=%zu max_abs_diff=%.6g\n", h0.size(), n_diff, max_abs_diff);
        return 1;
    }
}
