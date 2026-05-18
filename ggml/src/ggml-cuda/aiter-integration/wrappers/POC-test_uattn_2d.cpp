// End-to-end test of AITER's kernel_unified_attention_2d via Triton AOT.
//
// Setup: tiny but real attention — num_seqs=1, q_len=1 (decode), ctx=16,
// num_query_heads=16, num_kv_heads=2 (GQA ratio 8), head_size=128. Same
// constexpr values baked into the AOT binary (see signature in the .py
// invocation at MAD-188-PROGRESS.md).
//
// Test data is chosen so the expected output is computable by hand:
//   Q[h, d] = 1.0           K[t, kvh, d] = 1.0
//   V[t, kvh, d] = (kvh+1) * (d/128)
//
// → All QK scores = 128 (sum over head_dim of 1*1)
// → uniform softmax (1/16 per token)
// → V@p reduces to V[kvh, d] since V is constant across tokens
// → output[h, d] = ((h/8) + 1) * (d/128)   (GQA: heads 0-7 use kvh 0, 8-15 use kvh 1)

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// Generated AOT launcher header — extern "C" wrap because Triton's emitted
// header lacks the guards (drafted upstream issue 02).
extern "C" {
#include "uattn/uattn_2d.7902ab7d_0d1d2d3d.h"
}

#define CHECK(call)                                                    \
    do {                                                               \
        hipError_t err = (call);                                       \
        if (err != hipSuccess) {                                       \
            fprintf(stderr, "HIP error at %s:%d  %s: %s\n",            \
                    __FILE__, __LINE__, #call, hipGetErrorString(err));\
            return 1;                                                  \
        }                                                              \
    } while (0)

// fp16 conversion helpers (without pulling in any heavy dep)
typedef uint16_t fp16_t;

static fp16_t float_to_fp16(float f) {
    // IEEE 754 binary16 cast — use HIP's __float2half via a tiny inline.
    // Simple approach: use HIP's __half conversion symbol at link time.
    // But we want a host-side helper. Use a bitwise fast cast.
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3ff;
    if (exp <= 0) {
        return (fp16_t)(sign << 15);  // flush subnormals to zero (good enough for our test)
    }
    if (exp >= 31) {
        return (fp16_t)((sign << 15) | (0x1f << 10));  // inf
    }
    return (fp16_t)((sign << 15) | (exp << 10) | mant);
}

static float fp16_to_float(fp16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t out;
    if (exp == 0) {
        out = (sign << 31);  // close enough; subnormals→0
    } else if (exp == 31) {
        out = (sign << 31) | (0xff << 23) | (mant << 13);
    } else {
        out = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

int main() {
    // ── shape / size constants (must match the AOT constexpr values) ──
    constexpr int NUM_SEQS         = 1;
    constexpr int CTX_LEN          = 16;     // tokens in seq's KV cache
    constexpr int Q_LEN            = 1;      // decode: one query token
    constexpr int TOTAL_Q_TOKENS   = Q_LEN * NUM_SEQS;
    constexpr int NUM_Q_HEADS      = 16;
    constexpr int NUM_KV_HEADS     = 2;      // GQA ratio = 16/2 = 8
    constexpr int HEAD_SIZE        = 128;
    constexpr int BLOCK_SIZE       = 16;
    constexpr int NUM_BLOCKS       = 1;
    constexpr int MAX_BLOCKS_PER_SEQ = 1;

    constexpr size_t Q_ELEMS       = (size_t)TOTAL_Q_TOKENS * NUM_Q_HEADS * HEAD_SIZE;
    constexpr size_t KV_CACHE_ELEMS= (size_t)NUM_BLOCKS * BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE;

    // ── host buffers ──
    std::vector<fp16_t> q_h(Q_ELEMS);
    std::vector<fp16_t> k_h(KV_CACHE_ELEMS);
    std::vector<fp16_t> v_h(KV_CACHE_ELEMS);
    std::vector<fp16_t> out_h(Q_ELEMS, float_to_fp16(-99.0f));  // sentinel

    // Q: all 1.0
    for (size_t i = 0; i < Q_ELEMS; ++i) q_h[i] = float_to_fp16(1.0f);

    // K cache layout: [num_blocks, block_size, num_kv_heads, head_size]
    // All K values = 1.0 → every QK dot = head_size = 128
    for (size_t i = 0; i < KV_CACHE_ELEMS; ++i) k_h[i] = float_to_fp16(1.0f);

    // V cache: V[block, token, kvh, d] = (kvh+1) * (d/HEAD_SIZE)
    for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
        for (int t = 0; t < BLOCK_SIZE; ++t) {
            for (int kvh = 0; kvh < NUM_KV_HEADS; ++kvh) {
                for (int d = 0; d < HEAD_SIZE; ++d) {
                    float val = (kvh + 1.0f) * ((float)d / (float)HEAD_SIZE);
                    size_t off = ((size_t)blk * BLOCK_SIZE + t) * NUM_KV_HEADS * HEAD_SIZE
                                 + (size_t)kvh * HEAD_SIZE + d;
                    v_h[off] = float_to_fp16(val);
                }
            }
        }
    }

    // Block tables: seq 0 uses logical block 0 → physical block 0
    std::vector<int32_t> block_tables_h = {0};
    std::vector<int32_t> seq_lens_h = {CTX_LEN};
    std::vector<int32_t> cu_seqlens_q_h = {0, Q_LEN};  // [0, 1]

    // Dummy descale buffers: kernel always loads q/k/v_descale because we
    // passed *fp32 (typed pointer) in the AOT signature. The Triton
    // `if x_descale_ptr is not None` check is True for any non-None
    // typed pointer at runtime — only literal-None in the AOT signature
    // would short-circuit it. Setting to 1.0 makes the multiplication
    // a no-op. Same logic for out_scale.
    const float one_f = 1.0f;
    std::vector<float> scale_h = {one_f};

    // ── device buffers ──
    void *q_d, *k_d, *v_d, *out_d;
    void *bt_d, *sl_d, *cu_d;
    void *q_descale_d, *k_descale_d, *v_descale_d, *out_scale_d;
    CHECK(hipMalloc(&q_d,  q_h.size()   * sizeof(fp16_t)));
    CHECK(hipMalloc(&k_d,  k_h.size()   * sizeof(fp16_t)));
    CHECK(hipMalloc(&v_d,  v_h.size()   * sizeof(fp16_t)));
    CHECK(hipMalloc(&out_d, out_h.size() * sizeof(fp16_t)));
    CHECK(hipMalloc(&bt_d, block_tables_h.size() * sizeof(int32_t)));
    CHECK(hipMalloc(&sl_d, seq_lens_h.size()     * sizeof(int32_t)));
    CHECK(hipMalloc(&cu_d, cu_seqlens_q_h.size() * sizeof(int32_t)));
    CHECK(hipMalloc(&q_descale_d, sizeof(float)));
    CHECK(hipMalloc(&k_descale_d, sizeof(float)));
    CHECK(hipMalloc(&v_descale_d, sizeof(float)));
    CHECK(hipMalloc(&out_scale_d, sizeof(float)));

    CHECK(hipMemcpy(q_d, q_h.data(), q_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k_d, k_h.data(), k_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v_d, v_h.data(), v_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(out_d, out_h.data(), out_h.size() * sizeof(fp16_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(bt_d, block_tables_h.data(), block_tables_h.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(sl_d, seq_lens_h.data(),     seq_lens_h.size()     * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(cu_d, cu_seqlens_q_h.data(), cu_seqlens_q_h.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(q_descale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(k_descale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(v_descale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(out_scale_d, &one_f, sizeof(float), hipMemcpyHostToDevice));

    // ── strides (in elements, Triton convention) ──
    const int64_t query_stride_0      = NUM_Q_HEADS * HEAD_SIZE;            // 16 * 128 = 2048
    const int64_t output_stride_0     = NUM_Q_HEADS * HEAD_SIZE;            // 2048
    const int64_t block_table_stride  = MAX_BLOCKS_PER_SEQ;                 // 1
    const int64_t stride_k_cache_0    = BLOCK_SIZE * NUM_KV_HEADS * HEAD_SIZE; // 4096
    const int64_t stride_k_cache_1    = NUM_KV_HEADS * HEAD_SIZE;           // 256
    const int64_t stride_k_cache_2    = HEAD_SIZE;                          // 128
    // stride_k_cache_3 = 1 (baked in as constexpr)
    const int64_t stride_v_cache_0    = stride_k_cache_0;
    const int64_t stride_v_cache_1    = stride_k_cache_1;
    const int64_t stride_v_cache_2    = stride_k_cache_2;
    const int64_t qq_bias_stride_0    = 0;  // unused, USE_QQ_BIAS=0

    hipStream_t stream;
    CHECK(hipStreamCreate(&stream));

    // ── launch ──
    printf("Calling AITER kernel_unified_attention_2d on R9700...\n");
    printf("  shape: 1 seq, ctx=%d, q_len=%d, n_q_heads=%d, n_kv_heads=%d, head=%d\n",
           CTX_LEN, Q_LEN, NUM_Q_HEADS, NUM_KV_HEADS, HEAD_SIZE);

    hipError_t err = uattn_2d_7902ab7d_0d1d2d3d(
        stream,
        (hipDeviceptr_t) out_d,
        (hipDeviceptr_t) q_d,
        (hipDeviceptr_t) k_d,
        (hipDeviceptr_t) v_d,
        (hipDeviceptr_t) nullptr,     // sink_ptr (USE_SINKS=0)
        (hipDeviceptr_t) bt_d,
        (hipDeviceptr_t) sl_d,
        (hipDeviceptr_t) nullptr,     // alibi_slopes_ptr
        (hipDeviceptr_t) nullptr,     // qq_bias_ptr
        (hipDeviceptr_t) q_descale_d, // 1.0 — no-op
        (hipDeviceptr_t) k_descale_d, // 1.0 — no-op
        (hipDeviceptr_t) v_descale_d, // 1.0 — no-op
        (hipDeviceptr_t) out_scale_d, // 1.0 — no-op
        0.0,                          // softcap (USE_SOFTCAP=0, value unused)
        block_table_stride,
        query_stride_0,
        output_stride_0,
        qq_bias_stride_0,
        stride_k_cache_0, stride_k_cache_1, stride_k_cache_2,
        stride_v_cache_0, stride_v_cache_1, stride_v_cache_2,
        (hipDeviceptr_t) cu_d,
        (int32_t) NUM_SEQS);

    if (err != hipSuccess) {
        fprintf(stderr, "launcher returned: %s\n", hipGetErrorString(err));
        return 1;
    }

    CHECK(hipStreamSynchronize(stream));
    CHECK(hipMemcpy(out_h.data(), out_d, out_h.size() * sizeof(fp16_t), hipMemcpyDeviceToHost));

    // ── verify ──
    int nan_count = 0;
    int mismatch_count = 0;
    int sentinel_count = 0;
    constexpr fp16_t SENTINEL = 0;  // we set -99 sentinel; sentinel_count is for unchanged outputs
    const fp16_t SENTINEL_VAL = float_to_fp16(-99.0f);

    float max_err = 0.0f;
    int print_sample = 0;
    for (int h = 0; h < NUM_Q_HEADS; ++h) {
        int kvh = h / (NUM_Q_HEADS / NUM_KV_HEADS);  // GQA mapping: head/8
        for (int d = 0; d < HEAD_SIZE; ++d) {
            size_t idx = (size_t)h * HEAD_SIZE + d;  // token=0
            float actual = fp16_to_float(out_h[idx]);
            float expected = (kvh + 1.0f) * ((float)d / (float)HEAD_SIZE);

            if (std::isnan(actual) || std::isinf(actual)) nan_count++;
            if (out_h[idx] == SENTINEL_VAL)               sentinel_count++;

            float err_abs = std::fabs(actual - expected);
            if (err_abs > max_err) max_err = err_abs;

            // fp16 tolerance: relative ~1e-2 for non-tiny, absolute ~1e-3 for near-zero
            float tol = std::max(1e-3f, 1e-2f * std::fabs(expected));
            if (err_abs > tol) {
                mismatch_count++;
                if (print_sample < 5) {
                    printf("  mismatch  h=%2d (kvh=%d) d=%3d  actual=%.4f  expected=%.4f  err=%.4f\n",
                           h, kvh, d, actual, expected, err_abs);
                    print_sample++;
                }
            }
        }
    }

    printf("\nresult: nan/inf=%d  sentinel-unchanged=%d  mismatch=%d/%zu  max_err=%.4f\n",
           nan_count, sentinel_count, mismatch_count, Q_ELEMS, max_err);

    bool pass = (nan_count == 0) && (sentinel_count == 0) && (mismatch_count == 0);
    printf("\n%s\n", pass ? "✓ AITER 2D test PASSED" : "✗ AITER 2D test FAILED");

    // cleanup
    CHECK(hipStreamDestroy(stream));
    CHECK(hipFree(q_d));   CHECK(hipFree(k_d));   CHECK(hipFree(v_d));   CHECK(hipFree(out_d));
    CHECK(hipFree(bt_d));  CHECK(hipFree(sl_d));  CHECK(hipFree(cu_d));
    CHECK(hipFree(q_descale_d));  CHECK(hipFree(k_descale_d));
    CHECK(hipFree(v_descale_d));  CHECK(hipFree(out_scale_d));
    unload_uattn_2d_7902ab7d_0d1d2d3d();

    return pass ? 0 : 2;
}
