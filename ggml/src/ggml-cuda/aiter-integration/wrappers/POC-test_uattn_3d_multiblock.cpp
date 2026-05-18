// Stress test of AITER's 3D + reduce path with:
//   - ctx=64 (4 paged blocks @ block_size=16) — exercises multi-block walk
//   - Uniform K (all 1.0) → uniform softmax over 64 tokens (1/64 weight each)
//   - V varies BY TOKEN: V[blk, tok, kvh, d] = (kvh+1) * (d/128) * (gtok/ctx)
//     where gtok = blk*block_size + tok is the global token index.
//
// Expected attention output:
//   out[h, d] = sum_t softmax_t * V[t, kvh(h), d]
//             = (1/64) * sum_t (kvh+1) * (d/128) * (t/64)
//             = (kvh+1) * (d/128) * mean_t(t/64)
//             = (kvh+1) * (d/128) * (63 / 128)        (sum 0..63 = 2016 → mean = 31.5 → /64 = 0.4921875)
//             = (kvh+1) * (d/128) * 0.4921875
//
// This exercises:
//   ✓ Multi-block paged walk (block_tables = [0, 1, 2, 3])
//   ✓ Causal mask (q_pos=63, all 64 tokens visible)
//   ✓ Softmax over 64 tokens
//   ✓ V values varying in both head_dim and token axis
//   ✓ GQA fanout

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" {
#include "uattn/uattn_3d.11808415_0d3d4d5d.h"
#include "uattn/uattn_reduce.ecdee0c8_0d1d.h"
}

#define CHECK(call) do { hipError_t err = (call); \
    if (err != hipSuccess) { fprintf(stderr, "HIP error at %s:%d  %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(err)); return 1; } \
} while (0)

typedef uint16_t fp16_t;

static fp16_t float_to_fp16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0) return (fp16_t)(sign << 15);
    if (exp >= 31) return (fp16_t)((sign << 15) | (0x1f << 10));
    return (fp16_t)((sign << 15) | (exp << 10) | mant);
}
static float fp16_to_float(fp16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0)      out = (sign << 31);
    else if (exp == 31) out = (sign << 31) | (0xff << 23) | (mant << 13);
    else               out = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

int main() {
    constexpr int NUM_SEQS         = 1;
    constexpr int CTX_LEN          = 64;     // 4 paged blocks
    constexpr int Q_LEN            = 1;
    constexpr int TOTAL_Q_TOKENS   = Q_LEN * NUM_SEQS;
    constexpr int NUM_Q_HEADS      = 16;
    constexpr int NUM_KV_HEADS     = 2;
    constexpr int HEAD_SIZE        = 128;
    constexpr int BLOCK_SIZE       = 16;
    constexpr int NUM_BLOCKS       = CTX_LEN / BLOCK_SIZE;     // 4
    constexpr int MAX_BLOCKS_PER_SEQ = NUM_BLOCKS;             // 4
    constexpr int NUM_SEGMENTS_PER_SEQ = 32;
    constexpr double SCALE         = 0.0883883;

    constexpr size_t Q_ELEMS        = (size_t)TOTAL_Q_TOKENS * NUM_Q_HEADS * HEAD_SIZE;
    constexpr size_t KV_CACHE_ELEMS = (size_t)NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    constexpr size_t SEGM_OUT_ELEMS = (size_t)TOTAL_Q_TOKENS * NUM_Q_HEADS * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE;
    constexpr size_t SEGM_MS_ELEMS  = (size_t)TOTAL_Q_TOKENS * NUM_Q_HEADS * NUM_SEGMENTS_PER_SEQ;

    // Q = all 1.0; K = all 1.0
    std::vector<fp16_t> q_h(Q_ELEMS, float_to_fp16(1.0f));
    std::vector<fp16_t> k_h(KV_CACHE_ELEMS, float_to_fp16(1.0f));
    std::vector<fp16_t> v_h(KV_CACHE_ELEMS);
    std::vector<fp16_t> out_h(Q_ELEMS, float_to_fp16(-99.0f));

    // V varies by global token position
    for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
        for (int t = 0; t < BLOCK_SIZE; ++t) {
            int gtok = blk * BLOCK_SIZE + t;
            float gtok_norm = (float)gtok / (float)CTX_LEN;
            for (int kvh = 0; kvh < NUM_KV_HEADS; ++kvh) {
                for (int d = 0; d < HEAD_SIZE; ++d) {
                    float val = (kvh + 1.0f) * ((float)d / (float)HEAD_SIZE) * gtok_norm;
                    size_t off = ((size_t)blk * BLOCK_SIZE + t) * NUM_KV_HEADS * HEAD_SIZE
                                 + (size_t)kvh * HEAD_SIZE + d;
                    v_h[off] = float_to_fp16(val);
                }
            }
        }
    }

    // Block tables: logical 0..3 → physical 0..3
    std::vector<int32_t> block_tables_h = {0, 1, 2, 3};
    std::vector<int32_t> seq_lens_h = {CTX_LEN};
    std::vector<int32_t> cu_seqlens_q_h = {0, Q_LEN};
    const float one_f = 1.0f;

    // Device buffers
    void *q_d, *k_d, *v_d, *out_d;
    void *bt_d, *sl_d, *cu_d;
    void *q_descale_d, *k_descale_d, *v_descale_d, *out_scale_d;
    void *segm_out_d, *segm_max_d, *segm_expsum_d;
    CHECK(hipMalloc(&q_d,        Q_ELEMS        * sizeof(fp16_t)));
    CHECK(hipMalloc(&k_d,        KV_CACHE_ELEMS * sizeof(fp16_t)));
    CHECK(hipMalloc(&v_d,        KV_CACHE_ELEMS * sizeof(fp16_t)));
    CHECK(hipMalloc(&out_d,      Q_ELEMS        * sizeof(fp16_t)));
    CHECK(hipMalloc(&bt_d, block_tables_h.size() * sizeof(int32_t)));
    CHECK(hipMalloc(&sl_d, seq_lens_h.size()     * sizeof(int32_t)));
    CHECK(hipMalloc(&cu_d, cu_seqlens_q_h.size() * sizeof(int32_t)));
    CHECK(hipMalloc(&q_descale_d, sizeof(float)));
    CHECK(hipMalloc(&k_descale_d, sizeof(float)));
    CHECK(hipMalloc(&v_descale_d, sizeof(float)));
    CHECK(hipMalloc(&out_scale_d, sizeof(float)));
    CHECK(hipMalloc(&segm_out_d,    SEGM_OUT_ELEMS * sizeof(float)));
    CHECK(hipMalloc(&segm_max_d,    SEGM_MS_ELEMS  * sizeof(float)));
    CHECK(hipMalloc(&segm_expsum_d, SEGM_MS_ELEMS  * sizeof(float)));

    CHECK(hipMemcpy(q_d, q_h.data(),     q_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k_d, k_h.data(),     k_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v_d, v_h.data(),     v_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(out_d, out_h.data(), out_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(bt_d, block_tables_h.data(), block_tables_h.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(sl_d, seq_lens_h.data(),     seq_lens_h.size()     * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(cu_d, cu_seqlens_q_h.data(), cu_seqlens_q_h.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(q_descale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k_descale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v_descale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(out_scale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));

    const int64_t query_stride_0     = (int64_t)NUM_Q_HEADS * HEAD_SIZE;
    const int64_t output_stride_0    = (int64_t)NUM_Q_HEADS * HEAD_SIZE;
    const int64_t block_table_stride = MAX_BLOCKS_PER_SEQ;
    const int64_t stride_k_0 = (int64_t)BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;
    const int64_t stride_k_1 = (int64_t)NUM_KV_HEADS * HEAD_SIZE;
    const int64_t stride_k_2 = HEAD_SIZE;
    const int64_t stride_v_0 = stride_k_0;
    const int64_t stride_v_1 = stride_k_1;
    const int64_t stride_v_2 = stride_k_2;
    const int64_t qq_bias_stride_0 = 0;

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    printf("Stress test: ctx=%d (%d blocks), q_len=%d, varying V values\n",
           CTX_LEN, NUM_BLOCKS, Q_LEN);

    hipError_t err = uattn_3d_11808415_0d3d4d5d(
        stream,
        (hipDeviceptr_t) segm_out_d, (hipDeviceptr_t) segm_max_d, (hipDeviceptr_t) segm_expsum_d,
        (hipDeviceptr_t) q_d, (hipDeviceptr_t) k_d, (hipDeviceptr_t) v_d,
        (hipDeviceptr_t) nullptr,
        (hipDeviceptr_t) bt_d, (hipDeviceptr_t) sl_d,
        (hipDeviceptr_t) nullptr, (hipDeviceptr_t) nullptr,
        SCALE,
        (hipDeviceptr_t) q_descale_d, (hipDeviceptr_t) k_descale_d, (hipDeviceptr_t) v_descale_d,
        0.0,
        block_table_stride, query_stride_0, qq_bias_stride_0,
        stride_k_0, stride_k_1, stride_k_2,
        stride_v_0, stride_v_1, stride_v_2,
        (hipDeviceptr_t) cu_d, (int32_t) NUM_SEQS);
    if (err != hipSuccess) { fprintf(stderr, "3D failed: %s\n", hipGetErrorString(err)); return 1; }
    CHECK(hipStreamSynchronize(stream));
    printf("  ✓ 3D kernel returned hipSuccess (multi-block walk over %d blocks)\n", NUM_BLOCKS);

    err = uattn_reduce_ecdee0c8_0d1d(
        stream,
        (hipDeviceptr_t) out_d,
        (hipDeviceptr_t) segm_out_d, (hipDeviceptr_t) segm_max_d, (hipDeviceptr_t) segm_expsum_d,
        (hipDeviceptr_t) sl_d, (int32_t) NUM_SEQS,
        (hipDeviceptr_t) out_scale_d,
        output_stride_0, block_table_stride,
        (hipDeviceptr_t) cu_d);
    if (err != hipSuccess) { fprintf(stderr, "reduce failed: %s\n", hipGetErrorString(err)); return 1; }
    CHECK(hipStreamSynchronize(stream));
    printf("  ✓ reduce_segments returned hipSuccess\n");

    CHECK(hipMemcpy(out_h.data(), out_d, out_h.size() * sizeof(fp16_t), hipMemcpyDeviceToHost));

    // Hand-derived expected mean factor over tokens (0..ctx_len-1):
    // sum(t/ctx_len for t in 0..ctx_len-1) / ctx_len  =  sum(t) / (ctx_len^2)
    // = (ctx_len*(ctx_len-1)/2) / ctx_len^2
    // = (ctx_len-1) / (2*ctx_len)
    const float mean_tok_factor = (float)(CTX_LEN - 1) / (float)(2 * CTX_LEN);  // 0.4921875 for ctx=64
    printf("  hand-computed mean(tok/ctx) over 0..%d = %.7f\n", CTX_LEN-1, mean_tok_factor);

    int nan_count = 0, mismatch_count = 0, sentinel_count = 0;
    const fp16_t SENTINEL = float_to_fp16(-99.0f);
    float max_err = 0.0f;
    float worst_rel = 0.0f;
    int print_sample = 0;

    for (int h = 0; h < NUM_Q_HEADS; ++h) {
        int kvh = h / (NUM_Q_HEADS / NUM_KV_HEADS);
        for (int d = 0; d < HEAD_SIZE; ++d) {
            size_t idx = (size_t)h * HEAD_SIZE + d;
            float actual = fp16_to_float(out_h[idx]);
            float expected = (kvh + 1.0f) * ((float)d / (float)HEAD_SIZE) * mean_tok_factor;
            if (std::isnan(actual) || std::isinf(actual)) nan_count++;
            if (out_h[idx] == SENTINEL)                   sentinel_count++;
            float e = std::fabs(actual - expected);
            if (e > max_err) max_err = e;
            float rel = (std::fabs(expected) > 1e-4f) ? e / std::fabs(expected) : 0.0f;
            if (rel > worst_rel) worst_rel = rel;
            // fp16 tol: ~1e-2 relative or 1e-3 absolute
            float tol = std::max(1e-3f, 1e-2f * std::fabs(expected));
            if (e > tol) {
                mismatch_count++;
                if (print_sample < 5) {
                    printf("  mismatch h=%2d (kvh=%d) d=%3d  actual=%.5f  expected=%.5f  err=%.5f  rel=%.4f\n",
                           h, kvh, d, actual, expected, e, rel);
                    print_sample++;
                }
            }
        }
    }

    printf("\nresult: nan/inf=%d  sentinel-unchanged=%d  mismatch=%d/%zu  max_err=%.5f  worst_rel=%.4f%%\n",
           nan_count, sentinel_count, mismatch_count, Q_ELEMS, max_err, worst_rel * 100);

    bool pass = (nan_count == 0) && (sentinel_count == 0) && (mismatch_count == 0);
    printf("\n%s\n", pass ? "✓ AITER multi-block 3D+reduce test PASSED" : "✗ AITER multi-block 3D+reduce test FAILED");

    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(q_d));   CHECK(hipFree(k_d));   CHECK(hipFree(v_d));   CHECK(hipFree(out_d));
    CHECK(hipFree(bt_d));  CHECK(hipFree(sl_d));  CHECK(hipFree(cu_d));
    CHECK(hipFree(q_descale_d));   CHECK(hipFree(k_descale_d));
    CHECK(hipFree(v_descale_d));   CHECK(hipFree(out_scale_d));
    CHECK(hipFree(segm_out_d));    CHECK(hipFree(segm_max_d));
    CHECK(hipFree(segm_expsum_d));
    unload_uattn_3d_11808415_0d3d4d5d();
    unload_uattn_reduce_ecdee0c8_0d1d();

    return pass ? 0 : 2;
}
