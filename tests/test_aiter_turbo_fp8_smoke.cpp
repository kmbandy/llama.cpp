// MAD-214 Phase 1H step 2: numerical correctness validation for the
// turbo-FP8 AITER kernel.
//
// Generates random fp32 K, V, Q. CPU-quantizes K and V into the AITER paged
// cache layout (turbo4_fp8 BS=256 packed bytes). Runs the kernel via
// mt_aiter_unified_attn. Computes scalar reference attention on the SAME
// packed K/V using the same centroid LUT. Compares element-wise.
//
// Pass criterion: max abs err < 5% of output range, mean abs err < 1%.
// FP8 quantization + WMMA accumulation noise dominates; we're checking the
// kernel math is structurally correct, not bit-exact.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 -I ggml/src/ggml-cuda/aiter-integration/wrappers \
//       tests/test_aiter_turbo_fp8_smoke.cpp -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin -o /tmp/test_aiter_turbo_fp8_smoke

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
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n", __FILE__, __LINE__, hipGetErrorString(_e), _e); \
        return 1; \
    } \
} while(0)

// ─────────────────────────────────────────────────────────────────────────
// Scalar reference helpers — minimal turbo4-FP8 BS=256 quantize / dequant /
// attention (mirrors tests/test-turbo-fp8-reference.cpp but specialized to
// the AITER paged-cache layout).
// ─────────────────────────────────────────────────────────────────────────

constexpr int BS = 256;               // turbo-FP8 block size (= HEAD_SIZE)
constexpr int IDX_BITS = 4;           // turbo4
constexpr int N_CENTROIDS_T4 = 16;
constexpr int BYTES_PER_FP8_BLOCK = 162;  // 2 + 256*4/8 + 256/8

static float e4m3_byte_to_fp32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e    = (b >> 3) & 0xF;
    int m    = b & 0x7;
    if (e == 15 && m == 7) return NAN;
    float val;
    if (e == 0) val = (1.0f / 64.0f) * (m / 8.0f);
    else        val = std::ldexp(1.0f + m / 8.0f, e - 7);
    return sign ? -val : val;
}

// Quantize 256 fp32 values into one packed turbo4_fp8 BS=256 block (162 bytes).
// Layout: [d:fp16 scale | qs:128 bytes 4-bit indices | signs:32 bytes].
static void quantize_block_bs256(const float *in, uint8_t *out, const uint8_t *centroids) {
    // Per-block scale = max(|values|), stored as fp16
    float scale = 0.0f;
    for (int i = 0; i < BS; ++i) scale = std::max(scale, std::fabs(in[i]));
    if (scale == 0.0f) { std::memset(out, 0, BYTES_PER_FP8_BLOCK); return; }
    _Float16 scale_fp16 = (_Float16) scale;
    float scale_eff = (float) scale_fp16;
    if (scale_eff == 0.0f) scale_eff = scale;
    std::memcpy(out, &scale_fp16, 2);

    uint8_t *qs    = out + 2;
    uint8_t *signs = out + 2 + (BS * IDX_BITS / 8);  // = out + 130
    std::memset(qs, 0, BS * IDX_BITS / 8);
    std::memset(signs, 0, BS / 8);

    float cv[N_CENTROIDS_T4];
    for (int k = 0; k < N_CENTROIDS_T4; ++k) cv[k] = e4m3_byte_to_fp32(centroids[k]);

    for (int i = 0; i < BS; ++i) {
        float v = in[i];
        int   s = v < 0.0f ? 1 : 0;
        float m = std::fabs(v) / scale_eff;
        int best = 0; float best_err = std::fabs(m - cv[0]);
        for (int k = 1; k < N_CENTROIDS_T4; ++k) {
            float e = std::fabs(m - cv[k]);
            if (e < best_err) { best = k; best_err = e; }
        }
        // 4-bit pack: 2 indices per byte
        if ((i & 1) == 0) qs[i / 2] = (uint8_t) best;
        else              qs[i / 2] |= (uint8_t)(best << 4);
        signs[i / 8] |= (uint8_t)((s & 1) << (i & 7));
    }
}

// Inverse — dequantize one block to 256 fp32 values.
static void dequantize_block_bs256(const uint8_t *in, float *out, const uint8_t *centroids) {
    _Float16 scale_fp16; std::memcpy(&scale_fp16, in, 2);
    float scale = (float) scale_fp16;
    const uint8_t *qs    = in + 2;
    const uint8_t *signs = in + 2 + (BS * IDX_BITS / 8);
    for (int i = 0; i < BS; ++i) {
        int idx;
        if ((i & 1) == 0) idx = qs[i / 2] & 0xF;
        else              idx = (qs[i / 2] >> 4) & 0xF;
        int s = (signs[i / 8] >> (i & 7)) & 1;
        float mag = e4m3_byte_to_fp32(centroids[idx]);
        out[i] = (s ? -mag : mag) * scale;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AITER paged cache layout: K/V cache [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_size_packed]
//   head_size_packed = 162 bytes per (token, kv_head) for turbo4_fp8 BS=256
//   (one block covers the full head_dim=256 row).
// Byte offset of (block_idx, token_in_block, kv_head)'s 162-byte block:
//   (block_idx * BLOCK_SIZE * NUM_KV_HEADS + token_in_block * NUM_KV_HEADS + kv_head) * 162
// ─────────────────────────────────────────────────────────────────────────

int main() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop {};
    (void) hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    constexpr int HEAD_SIZE     = 256;
    constexpr int NUM_Q_HEADS   = 16;
    constexpr int NUM_KV_HEADS  = 4;
    constexpr int Q_HEADS_PER_KV = NUM_Q_HEADS / NUM_KV_HEADS;
    constexpr int BLOCK_SIZE    = 16;
    // NUM_KV_TOKENS=512 forces multi-segment work in the 3D split-K decode
    // kernel. With NUM_SEGMENTS_PER_SEQ=32 and TILE_SIZE=32, segments 0..15
    // each process one tile (32 tokens); 16+ are empty. This exercises both
    // the per-segment partial softmax AND the reduce_segments cross-segment
    // max/expsum reduction — a true multi-segment test, not the degenerate
    // single-segment case where 3D and 2D paths happen to be bit-identical.
    constexpr int NUM_KV_TOKENS = 512;
    constexpr int NUM_SEQS      = 1;
    constexpr int Q_LEN         = 1;
    constexpr int NUM_BLOCKS    = NUM_KV_TOKENS / BLOCK_SIZE;  // = 32

    constexpr size_t KV_TOTAL_BYTES_FP8 =
        (size_t) NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK;
    constexpr size_t KV_TOTAL_BYTES_F16 =
        (size_t) NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE * sizeof(_Float16);
    constexpr size_t KV_TOTAL_BYTES =
        KV_TOTAL_BYTES_F16 > KV_TOTAL_BYTES_FP8 ? KV_TOTAL_BYTES_F16 : KV_TOTAL_BYTES_FP8;

    fprintf(stderr, "# KV cache allocation: %zu bytes (F16=%zu, FP8=%zu)\n",
            KV_TOTAL_BYTES, KV_TOTAL_BYTES_F16, KV_TOTAL_BYTES_FP8);

    // ── Generate random fp32 K, V, Q ──────────────────────────────────────
    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 0.3f);

    std::vector<float> k_fp32(NUM_KV_TOKENS * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> v_fp32(NUM_KV_TOKENS * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> q_fp32(NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE);
    for (auto &v : k_fp32) v = dist(rng);
    for (auto &v : v_fp32) v = dist(rng);
    for (auto &v : q_fp32) v = dist(rng);

    // Pick centroids — use the production qwen35_4b_bs256 LUT for layer 3
    // (first attention layer). Both K and V use the same LUT for the smoke
    // test, even though in production they'd differ.
    const uint8_t *centroids_k_host = mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3;
    const uint8_t *centroids_v_host = mt_turbo4_fp8_centroids_qwen35_4b_bs256_v_L3;

    // ── CPU-quantize K and V into the AITER cache layout ──────────────────
    std::vector<uint8_t> k_packed(KV_TOTAL_BYTES, 0);
    std::vector<uint8_t> v_packed(KV_TOTAL_BYTES, 0);
    for (int t = 0; t < NUM_KV_TOKENS; ++t) {
        for (int h = 0; h < NUM_KV_HEADS; ++h) {
            const float *k_src = &k_fp32[(t * NUM_KV_HEADS + h) * HEAD_SIZE];
            const float *v_src = &v_fp32[(t * NUM_KV_HEADS + h) * HEAD_SIZE];
            int block_idx = t / BLOCK_SIZE;
            int tok_in_blk = t % BLOCK_SIZE;
            size_t off = ((size_t) block_idx * BLOCK_SIZE * NUM_KV_HEADS
                          + tok_in_blk * NUM_KV_HEADS + h) * BYTES_PER_FP8_BLOCK;
            quantize_block_bs256(k_src, &k_packed[off], centroids_k_host);
            quantize_block_bs256(v_src, &v_packed[off], centroids_v_host);
        }
    }

    // ── CPU scalar reference: dequant K, V, compute full fp32 attention ────
    // This is the ground truth the kernel output should match (up to FP8
    // arithmetic noise).
    std::vector<float> k_dq(NUM_KV_TOKENS * NUM_KV_HEADS * HEAD_SIZE);
    std::vector<float> v_dq(NUM_KV_TOKENS * NUM_KV_HEADS * HEAD_SIZE);
    for (int t = 0; t < NUM_KV_TOKENS; ++t) {
        for (int h = 0; h < NUM_KV_HEADS; ++h) {
            int block_idx = t / BLOCK_SIZE;
            int tok_in_blk = t % BLOCK_SIZE;
            size_t off = ((size_t) block_idx * BLOCK_SIZE * NUM_KV_HEADS
                          + tok_in_blk * NUM_KV_HEADS + h) * BYTES_PER_FP8_BLOCK;
            dequantize_block_bs256(&k_packed[off], &k_dq[(t * NUM_KV_HEADS + h) * HEAD_SIZE], centroids_k_host);
            dequantize_block_bs256(&v_packed[off], &v_dq[(t * NUM_KV_HEADS + h) * HEAD_SIZE], centroids_v_host);
        }
    }

    std::vector<float> ref_out(NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE, 0.0f);
    const float scale_attn = 1.0f / std::sqrt((float) HEAD_SIZE);
    for (int qi = 0; qi < Q_LEN; ++qi) {
        for (int qh = 0; qh < NUM_Q_HEADS; ++qh) {
            int kvh = qh / Q_HEADS_PER_KV;
            std::vector<float> scores(NUM_KV_TOKENS);
            for (int t = 0; t < NUM_KV_TOKENS; ++t) {
                float s = 0.0f;
                for (int d = 0; d < HEAD_SIZE; ++d) {
                    s += q_fp32[(qi * NUM_Q_HEADS + qh) * HEAD_SIZE + d] *
                         k_dq[(t * NUM_KV_HEADS + kvh) * HEAD_SIZE + d];
                }
                scores[t] = s * scale_attn;
            }
            float max_s = scores[0];
            for (int t = 1; t < NUM_KV_TOKENS; ++t) if (scores[t] > max_s) max_s = scores[t];
            float sum_exp = 0.0f;
            for (int t = 0; t < NUM_KV_TOKENS; ++t) { scores[t] = std::exp(scores[t] - max_s); sum_exp += scores[t]; }
            for (int t = 0; t < NUM_KV_TOKENS; ++t) scores[t] /= sum_exp;
            for (int d = 0; d < HEAD_SIZE; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < NUM_KV_TOKENS; ++t)
                    acc += scores[t] * v_dq[(t * NUM_KV_HEADS + kvh) * HEAD_SIZE + d];
                ref_out[(qi * NUM_Q_HEADS + qh) * HEAD_SIZE + d] = acc;
            }
        }
    }

    // ── Upload data to GPU ────────────────────────────────────────────────
    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_q,   q_fp32.size() * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_k,   KV_TOTAL_BYTES));
    HIP_CHECK(hipMalloc(&d_v,   KV_TOTAL_BYTES));
    HIP_CHECK(hipMalloc(&d_out, q_fp32.size() * sizeof(_Float16)));

    // Q: convert fp32 → fp16 and upload
    std::vector<_Float16> q_fp16(q_fp32.size());
    for (size_t i = 0; i < q_fp32.size(); ++i) q_fp16[i] = (_Float16) q_fp32[i];
    HIP_CHECK(hipMemcpy(d_q, q_fp16.data(), q_fp16.size() * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k, k_packed.data(), KV_TOTAL_BYTES_FP8, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v, v_packed.data(), KV_TOTAL_BYTES_FP8, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, q_fp16.size() * sizeof(_Float16)));

    // segm workspace
    constexpr int NUM_SEGMENTS = 128;
    size_t SEGM_OUT_BYTES    = (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * NUM_SEGMENTS * HEAD_SIZE * sizeof(float);
    size_t SEGM_MAX_BYTES    = (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * NUM_SEGMENTS * sizeof(float);
    void *d_segm_out = nullptr, *d_segm_max = nullptr, *d_segm_expsum = nullptr;
    HIP_CHECK(hipMalloc(&d_segm_out,    SEGM_OUT_BYTES));
    HIP_CHECK(hipMalloc(&d_segm_max,    SEGM_MAX_BYTES));
    HIP_CHECK(hipMalloc(&d_segm_expsum, SEGM_MAX_BYTES));

    std::vector<int32_t> h_block_tables(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) h_block_tables[i] = i;  // identity mapping
    std::vector<int32_t> h_seq_lens       = {NUM_KV_TOKENS};
    std::vector<int32_t> h_query_start_len = {0, Q_LEN};
    int32_t *d_block_tables = nullptr, *d_seq_lens = nullptr, *d_query_start_len = nullptr;
    HIP_CHECK(hipMalloc(&d_block_tables,    h_block_tables.size()    * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_seq_lens,        h_seq_lens.size()        * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_query_start_len, h_query_start_len.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_block_tables,    h_block_tables.data(),    h_block_tables.size() * sizeof(int32_t),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq_lens,        h_seq_lens.data(),        h_seq_lens.size() * sizeof(int32_t),        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_query_start_len, h_query_start_len.data(), h_query_start_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    std::vector<float> h_ones = {1.0f};
    float *d_ones = nullptr;
    HIP_CHECK(hipMalloc(&d_ones, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_ones, h_ones.data(), sizeof(float), hipMemcpyHostToDevice));

    uint8_t *d_centroids_k = nullptr, *d_centroids_v = nullptr;
    HIP_CHECK(hipMalloc(&d_centroids_k, N_CENTROIDS_T4));
    HIP_CHECK(hipMalloc(&d_centroids_v, N_CENTROIDS_T4));
    HIP_CHECK(hipMemcpy(d_centroids_k, centroids_k_host, N_CENTROIDS_T4, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_centroids_v, centroids_v_host, N_CENTROIDS_T4, hipMemcpyHostToDevice));

    // ── Build shape + args ────────────────────────────────────────────────
    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = HEAD_SIZE;
    shape.num_q_heads  = NUM_Q_HEADS;
    shape.num_kv_heads = NUM_KV_HEADS;
    shape.block_size   = BLOCK_SIZE;
    shape.cache_type   = MT_AITER_CACHE_TURBO4_FP8;

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
    args.centroids_k     = d_centroids_k;
    args.centroids_v     = d_centroids_v;
    args.scale           = scale_attn;
    args.num_seqs        = NUM_SEQS;
    args.num_q_tokens    = NUM_SEQS * Q_LEN;
    args.block_table_stride = NUM_BLOCKS;  // per-sequence stride into block_tables
    args.q_stride_0      = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    args.output_stride_0 = args.q_stride_0;
    args.k_stride_0      = (int64_t) BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_1      = (int64_t) NUM_KV_HEADS * HEAD_SIZE;
    args.k_stride_2      = (int64_t) HEAD_SIZE;
    args.v_stride_0      = args.k_stride_0;
    args.v_stride_1      = args.k_stride_1;
    args.v_stride_2      = args.k_stride_2;

    fprintf(stderr, "# calling mt_aiter_unified_attn (cache_type=%d)...\n", shape.cache_type);
    hipError_t err = mt_aiter_unified_attn(0 /*stream*/, &args);
    if (err != hipSuccess) {
        fprintf(stderr, "FAIL: kernel launch returned %s\n", hipGetErrorString(err));
        return 1;
    }
    HIP_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "# kernel returned hipSuccess, sync OK\n");

    // ── Read back kernel output, compare to scalar reference ──────────────
    std::vector<_Float16> kernel_out_fp16(q_fp16.size());
    HIP_CHECK(hipMemcpy(kernel_out_fp16.data(), d_out, kernel_out_fp16.size() * sizeof(_Float16), hipMemcpyDeviceToHost));

    float max_abs_err = 0.0f, sum_abs_err = 0.0f, sum_sq_err = 0.0f;
    float max_ref = 0.0f;
    int n_nan = 0;
    for (size_t i = 0; i < ref_out.size(); ++i) {
        float k = (float) kernel_out_fp16[i];
        float r = ref_out[i];
        if (k != k) { ++n_nan; continue; }
        float d = std::fabs(k - r);
        max_abs_err = std::max(max_abs_err, d);
        sum_abs_err += d;
        sum_sq_err  += d * d;
        max_ref = std::max(max_ref, std::fabs(r));
    }
    float mean_abs_err = sum_abs_err / (float) ref_out.size();
    float rms_err = std::sqrt(sum_sq_err / (float) ref_out.size());
    float max_rel = max_ref > 0 ? max_abs_err / max_ref : 0.0f;

    fprintf(stderr, "\n=== CORRECTNESS COMPARISON ===\n");
    fprintf(stderr, "  ref range: |max| = %.6g\n", max_ref);
    fprintf(stderr, "  max_abs_err  = %.6g  (%.2f%% of ref max)\n", max_abs_err, 100.0 * max_rel);
    fprintf(stderr, "  mean_abs_err = %.6g\n", mean_abs_err);
    fprintf(stderr, "  rms_err      = %.6g\n", rms_err);
    fprintf(stderr, "  NaN count    = %d / %zu\n", n_nan, ref_out.size());

    // Print first 8 (kernel, ref, diff) for inspection
    fprintf(stderr, "  first 8: (kernel, ref, diff)\n");
    for (int i = 0; i < 8; ++i) {
        fprintf(stderr, "    [%d] %+.4f  %+.4f  %+.4g\n",
                i, (float) kernel_out_fp16[i], ref_out[i],
                (float) kernel_out_fp16[i] - ref_out[i]);
    }

    // Pass criterion: max abs err < 5% of ref max, no NaNs
    bool pass = (max_rel < 0.05f) && (n_nan == 0);
    fprintf(stderr, "\n=== %s ===\n", pass ? "PASS — kernel output matches reference within FP8 noise"
                                            : "FAIL — output differs from reference beyond FP8 noise bound");

    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
    hipFree(d_segm_out); hipFree(d_segm_max); hipFree(d_segm_expsum);
    hipFree(d_block_tables); hipFree(d_seq_lens); hipFree(d_query_start_len);
    hipFree(d_ones);
    hipFree(d_centroids_k); hipFree(d_centroids_v);
    return pass ? 0 : 1;
}
