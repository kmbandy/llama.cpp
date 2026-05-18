// test_mt_aiter_unified_attn — exercises the mt_aiter_unified_attn wrapper
// with the validated long-ctx 2-class-K setup (was POC-test_uattn_3d_longctx).
//
// Setup:
//   ctx=1024 (64 paged blocks @ block_size=16), 1 seq, decode (q_len=1).
//   Q=1.0, K is 2-class: tokens [0..ctx/2)=0.0, [ctx/2..ctx)=1.0.
//   V[blk,t,kvh,d] = (kvh+1) * (d/128) * (gtok/ctx).
//
// Expected:
//   class B softmax weight dominates (~exp(11.31) vs exp(0))
//   out[h,d] ≈ (kvh+1) * (d/128) * mean_{t in B}(t/ctx) = ... * 0.74948
//
// Validates: scale path (non-uniform QK), paged walk across all 64 blocks,
// all 32 split-K segments, reduce_segments correctness, and the wrapper's
// arg-bundling matches the raw launcher's arg order.
//
// Reference (POC, pre-wrapper) on R9700: 0/2048 mismatches, max_err 0.00048.

#include "mt_aiter_unified_attn.h"

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK(call) do {                                                       \
    hipError_t err = (call);                                                   \
    if (err != hipSuccess) {                                                   \
        fprintf(stderr, "HIP error at %s:%d  %s: %s\n",                        \
                __FILE__, __LINE__, #call, hipGetErrorString(err));            \
        return 1;                                                              \
    }                                                                          \
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
    if (exp == 0)       out = (sign << 31);
    else if (exp == 31) out = (sign << 31) | (0xff << 23) | (mant << 13);
    else                out = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    float f; std::memcpy(&f, &out, 4); return f;
}

// Run the long-ctx 2-class-K correctness test for a given model shape. The
// runtime registry compiles a kernel for whatever shape we pass — `head=128
// nq=16 nkv=2` is the historical POC shape; the test now also runs with a
// real-Qwen-like shape (`head=128 nq=28 nkv=4`) to exercise the
// shape-derivation path end-to-end.
static int run_long_ctx_test(const mt_aiter_uattn_shape_t & shape) {
    constexpr int    NUM_SEQS        = 1;
    constexpr int    CTX_LEN         = 1024;       // 64 paged blocks at BS=16
    constexpr int    Q_LEN           = 1;
    constexpr int    TOTAL_Q_TOKENS  = Q_LEN * NUM_SEQS;
    const int        NUM_BLOCKS      = CTX_LEN / shape.block_size;
    const int        MAX_BLKS_PER_SQ = NUM_BLOCKS;
    const float      SCALE           = 1.0f / std::sqrt((float)shape.head_size);

    const size_t Q_ELEMS        = (size_t)TOTAL_Q_TOKENS
                                * shape.num_q_heads
                                * shape.head_size;
    const size_t KV_CACHE_ELEMS = (size_t)NUM_BLOCKS
                                * shape.block_size
                                * shape.num_kv_heads
                                * shape.head_size;

    printf("\n──── shape: head_size=%d num_q_heads=%d num_kv_heads=%d block_size=%d (GQA=%d) ────\n",
           shape.head_size, shape.num_q_heads, shape.num_kv_heads, shape.block_size,
           shape.num_q_heads / shape.num_kv_heads);

    // ── Host inputs ─────────────────────────────────────────────────────────
    std::vector<fp16_t>  q_h(Q_ELEMS, float_to_fp16(1.0f));
    std::vector<fp16_t>  k_h(KV_CACHE_ELEMS);
    std::vector<fp16_t>  v_h(KV_CACHE_ELEMS);
    for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
        for (int t = 0; t < shape.block_size; ++t) {
            int gtok = blk * shape.block_size + t;
            float kval     = (gtok < CTX_LEN / 2) ? 0.0f : 1.0f;
            float gtok_n   = (float)gtok / (float)CTX_LEN;
            for (int kvh = 0; kvh < shape.num_kv_heads; ++kvh) {
                for (int d = 0; d < shape.head_size; ++d) {
                    size_t off = ((size_t)blk * shape.block_size + t)
                               * shape.num_kv_heads * shape.head_size
                               + (size_t)kvh * shape.head_size + d;
                    k_h[off] = float_to_fp16(kval);
                    v_h[off] = float_to_fp16((kvh + 1.0f)
                              * ((float)d / (float)shape.head_size)
                              * gtok_n);
                }
            }
        }
    }
    std::vector<fp16_t>  out_h(Q_ELEMS, float_to_fp16(-99.0f));
    std::vector<int32_t> block_tables_h(NUM_BLOCKS);
    for (int i = 0; i < NUM_BLOCKS; ++i) block_tables_h[i] = i;
    std::vector<int32_t> seq_lens_h     = {CTX_LEN};
    std::vector<int32_t> query_start_h  = {0, Q_LEN};
    const float          one_f          = 1.0f;

    // ── Device allocations ──────────────────────────────────────────────────
    void *q_d, *k_d, *v_d, *out_d;
    void *bt_d, *sl_d, *cu_d;
    void *q_descale_d, *k_descale_d, *v_descale_d, *out_scale_d;
    void *segm_out_d, *segm_max_d, *segm_expsum_d;
    CHECK(hipMalloc(&q_d,          Q_ELEMS        * sizeof(fp16_t)));
    CHECK(hipMalloc(&k_d,          KV_CACHE_ELEMS * sizeof(fp16_t)));
    CHECK(hipMalloc(&v_d,          KV_CACHE_ELEMS * sizeof(fp16_t)));
    CHECK(hipMalloc(&out_d,        Q_ELEMS        * sizeof(fp16_t)));
    CHECK(hipMalloc(&bt_d,         block_tables_h.size() * sizeof(int32_t)));
    CHECK(hipMalloc(&sl_d,         seq_lens_h.size()     * sizeof(int32_t)));
    CHECK(hipMalloc(&cu_d,         query_start_h.size()  * sizeof(int32_t)));
    CHECK(hipMalloc(&q_descale_d,  sizeof(float)));
    CHECK(hipMalloc(&k_descale_d,  sizeof(float)));
    CHECK(hipMalloc(&v_descale_d,  sizeof(float)));
    CHECK(hipMalloc(&out_scale_d,  sizeof(float)));
    CHECK(hipMalloc(&segm_out_d,   mt_aiter_uattn_segm_output_bytes(&shape, TOTAL_Q_TOKENS)));
    CHECK(hipMalloc(&segm_max_d,   mt_aiter_uattn_segm_max_bytes(&shape, TOTAL_Q_TOKENS)));
    CHECK(hipMalloc(&segm_expsum_d,mt_aiter_uattn_segm_expsum_bytes(&shape, TOTAL_Q_TOKENS)));

    CHECK(hipMemcpy(q_d,           q_h.data(),            q_h.size()           * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k_d,           k_h.data(),            k_h.size()           * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v_d,           v_h.data(),            v_h.size()           * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(out_d,         out_h.data(),          out_h.size()         * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(bt_d,          block_tables_h.data(), block_tables_h.size()* sizeof(int32_t),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(sl_d,          seq_lens_h.data(),     seq_lens_h.size()    * sizeof(int32_t),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(cu_d,          query_start_h.data(),  query_start_h.size() * sizeof(int32_t),hipMemcpyHostToDevice));
    CHECK(hipMemcpy(q_descale_d,   &one_f,                sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k_descale_d,   &one_f,                sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v_descale_d,   &one_f,                sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(out_scale_d,   &one_f,                sizeof(float), hipMemcpyHostToDevice));

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    // ── Build wrapper arg bundle ────────────────────────────────────────────
    mt_aiter_uattn_args_t args = {};
    args.shape         = shape;
    args.q             = q_d;
    args.k_cache       = k_d;
    args.v_cache       = v_d;
    args.out           = out_d;
    args.segm_output   = segm_out_d;
    args.segm_max      = segm_max_d;
    args.segm_expsum   = segm_expsum_d;
    args.block_tables  = (const int32_t*)bt_d;
    args.seq_lens      = (const int32_t*)sl_d;
    args.query_start_len = (const int32_t*)cu_d;
    args.q_descale     = (const float*)q_descale_d;
    args.k_descale     = (const float*)k_descale_d;
    args.v_descale     = (const float*)v_descale_d;
    args.out_scale     = (const float*)out_scale_d;
    args.scale              = SCALE;
    args.num_seqs           = NUM_SEQS;
    args.num_q_tokens       = TOTAL_Q_TOKENS;
    args.block_table_stride = MAX_BLKS_PER_SQ;
    args.q_stride_0         = (int64_t)shape.num_q_heads * shape.head_size;
    args.output_stride_0    = args.q_stride_0;
    args.k_stride_0         = (int64_t)shape.block_size * shape.num_kv_heads * shape.head_size;
    args.k_stride_1         = (int64_t)shape.num_kv_heads * shape.head_size;
    args.k_stride_2         = shape.head_size;
    args.v_stride_0         = args.k_stride_0;
    args.v_stride_1         = args.k_stride_1;
    args.v_stride_2         = args.k_stride_2;

    printf("  ctx=%d (%d blocks, %d segments)  scale=%.4f  QK class-B score=%.4f\n",
           CTX_LEN, NUM_BLOCKS, MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ,
           SCALE, SCALE * shape.head_size);

    hipError_t rc = mt_aiter_unified_attn(stream, &args);
    if (rc != hipSuccess) {
        fprintf(stderr, "mt_aiter_unified_attn failed: %s\n", hipGetErrorString(rc));
        return 1;
    }
    CHECK(hipStreamSynchronize(stream));
    printf("  ✓ mt_aiter_unified_attn returned hipSuccess\n");

    CHECK(hipMemcpy(out_h.data(), out_d, out_h.size() * sizeof(fp16_t), hipMemcpyDeviceToHost));

    // ── Reference: full softmax weight on class B (class A weight is negligible) ──
    double ref_mean = 0.0;
    int    n_class_b = 0;
    for (int t = CTX_LEN / 2; t < CTX_LEN; ++t) {
        ref_mean += (double)t / (double)CTX_LEN;
        n_class_b++;
    }
    ref_mean /= (double)n_class_b;

    int   nan_count = 0, mismatch_count = 0, sentinel_count = 0;
    const fp16_t SENTINEL = float_to_fp16(-99.0f);
    float max_err = 0.0f, worst_rel = 0.0f;
    int   print_sample = 0;

    for (int h = 0; h < shape.num_q_heads; ++h) {
        int kvh = h / (shape.num_q_heads / shape.num_kv_heads);
        for (int d = 0; d < shape.head_size; ++d) {
            size_t idx = (size_t)h * shape.head_size + d;
            float actual   = fp16_to_float(out_h[idx]);
            float expected = (kvh + 1.0f) * ((float)d / (float)shape.head_size)
                           * (float)ref_mean;
            if (std::isnan(actual) || std::isinf(actual)) nan_count++;
            if (out_h[idx] == SENTINEL) sentinel_count++;
            float e = std::fabs(actual - expected);
            if (e > max_err) max_err = e;
            float rel = (std::fabs(expected) > 1e-4f) ? e / std::fabs(expected) : 0.0f;
            if (rel > worst_rel) worst_rel = rel;
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

    printf("  result: nan/inf=%d  sentinel=%d  mismatch=%d/%zu  max_err=%.5f  worst_rel=%.4f%%\n",
           nan_count, sentinel_count, mismatch_count, Q_ELEMS, max_err, worst_rel * 100);

    bool pass = (nan_count == 0) && (sentinel_count == 0) && (mismatch_count == 0);
    printf("  %s\n", pass ? "✓ PASSED" : "✗ FAILED");

    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(q_d));   CHECK(hipFree(k_d));   CHECK(hipFree(v_d));   CHECK(hipFree(out_d));
    CHECK(hipFree(bt_d));  CHECK(hipFree(sl_d));  CHECK(hipFree(cu_d));
    CHECK(hipFree(q_descale_d));   CHECK(hipFree(k_descale_d));
    CHECK(hipFree(v_descale_d));   CHECK(hipFree(out_scale_d));
    CHECK(hipFree(segm_out_d));    CHECK(hipFree(segm_max_d));
    CHECK(hipFree(segm_expsum_d));

    return pass ? 0 : 2;
}

int main() {
    // Single-process AITER cache supports one shape, so we can only test one
    // shape per run. Default: POC shape (h=128/nq=16/nkv=2/bs=16). Override
    // with AITER_TEST_SHAPE="h,nq,nkv,bs" — useful for the Qwen-shape probe.
    mt_aiter_uattn_shape_t shape = {128, 16, 2, 16};
    const char * env = std::getenv("AITER_TEST_SHAPE");
    if (env && *env) {
        int h, nq, nkv, bs;
        if (std::sscanf(env, "%d,%d,%d,%d", &h, &nq, &nkv, &bs) == 4) {
            shape = {h, nq, nkv, bs};
        } else {
            fprintf(stderr, "AITER_TEST_SHAPE must be \"head,nq,nkv,bs\" (got %s)\n", env);
            return 1;
        }
    }
    return run_long_ctx_test(shape);
}
