// MAD-214 Phase 1H: smoke test for the turbo-FP8 AITER kernel path.
//
// Goal: prove that calling mt_aiter_unified_attn() with
// cache_type = MT_AITER_CACHE_TURBO4_FP8 actually launches the kernel on
// gfx1201 and returns hipSuccess. Output correctness is a follow-up; this
// test just exercises the linkage + Triton JIT + kernel launch chain.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 \
//       -I ggml/src/ggml-cuda/aiter-integration/wrappers \
//       tests/test_aiter_turbo_fp8_smoke.cpp \
//       -L build-hip/bin -lggml-hip \
//       -Wl,-rpath,$(pwd)/build-hip/bin \
//       -o /tmp/test_aiter_turbo_fp8_smoke

#include "mt_aiter_unified_attn.h"
#include "../ggml/src/ggml-cuda/aiter-integration/turbo_fp8_data/qwen35_4b_bs256_centroids.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s (%d)\n", __FILE__, __LINE__, hipGetErrorString(_e), _e); \
        return 1; \
    } \
} while(0)

int main() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop {};
    (void) hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    // Shape: minimal viable Qwen3.5-4B-like configuration.
    // BS=256 → head_size MUST be 256 (per the kernel's tl.static_assert).
    constexpr int HEAD_SIZE     = 256;
    constexpr int NUM_Q_HEADS   = 16;
    constexpr int NUM_KV_HEADS  = 4;
    constexpr int BLOCK_SIZE    = 16;       // paged-cache block size in tokens
    constexpr int NUM_KV_TOKENS = 16;       // one paged block worth
    constexpr int NUM_SEQS      = 1;
    constexpr int Q_LEN         = 1;        // pure decode (simplest)

    // Block sizes for turbo4_fp8_bs256 (from ggml-common.h)
    constexpr int BYTES_PER_FP8_BLOCK = 162;  // 2 scale + 128 idx + 32 sign

    // K/V cache layout: [num_blocks, BLOCK_SIZE, NUM_KV_HEADS, head_size_packed]
    //   where head_size_packed = head_size / QK_TURBO_FP8 × BYTES_PER_FP8_BLOCK
    //   With BS=256 and head_size=256, that's 1 × 162 bytes per (token, kv_head).
    constexpr int NUM_BLOCKS    = NUM_KV_TOKENS / BLOCK_SIZE;  // = 1
    constexpr size_t KV_BLOCK_BYTES = (size_t) BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK;
    constexpr size_t KV_TOTAL_BYTES = (size_t) NUM_BLOCKS * KV_BLOCK_BYTES;

    fprintf(stderr, "# KV cache: %zu bytes per block × %d blocks = %zu bytes total\n",
            KV_BLOCK_BYTES, NUM_BLOCKS, KV_TOTAL_BYTES);

    // ── Allocate device buffers ────────────────────────────────────────────
    void *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_q,   (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_k,   KV_TOTAL_BYTES));
    HIP_CHECK(hipMalloc(&d_v,   KV_TOTAL_BYTES));
    HIP_CHECK(hipMalloc(&d_out, (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE * sizeof(_Float16)));
    HIP_CHECK(hipMemset(d_q,   0, (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE * sizeof(_Float16)));
    HIP_CHECK(hipMemset(d_k,   0, KV_TOTAL_BYTES));
    HIP_CHECK(hipMemset(d_v,   0, KV_TOTAL_BYTES));
    HIP_CHECK(hipMemset(d_out, 0, (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE * sizeof(_Float16)));

    // Split-K workspace (segm_*)
    constexpr int NUM_SEGMENTS = 128;  // matches MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ
    constexpr size_t SEGM_OUT_BYTES    = (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * NUM_SEGMENTS * HEAD_SIZE * sizeof(float);
    constexpr size_t SEGM_MAX_BYTES    = (size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * NUM_SEGMENTS * sizeof(float);
    constexpr size_t SEGM_EXPSUM_BYTES = SEGM_MAX_BYTES;
    void *d_segm_out = nullptr, *d_segm_max = nullptr, *d_segm_expsum = nullptr;
    HIP_CHECK(hipMalloc(&d_segm_out,    SEGM_OUT_BYTES));
    HIP_CHECK(hipMalloc(&d_segm_max,    SEGM_MAX_BYTES));
    HIP_CHECK(hipMalloc(&d_segm_expsum, SEGM_EXPSUM_BYTES));

    // Paged tables / seq lens
    std::vector<int32_t> h_block_tables   = {0};                  // one seq → block index 0
    std::vector<int32_t> h_seq_lens       = {NUM_KV_TOKENS};
    std::vector<int32_t> h_query_start_len = {0, Q_LEN};
    int32_t *d_block_tables = nullptr, *d_seq_lens = nullptr, *d_query_start_len = nullptr;
    HIP_CHECK(hipMalloc(&d_block_tables,    h_block_tables.size()    * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_seq_lens,        h_seq_lens.size()        * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_query_start_len, h_query_start_len.size() * sizeof(int32_t)));
    HIP_CHECK(hipMemcpy(d_block_tables,    h_block_tables.data(),    h_block_tables.size() * sizeof(int32_t),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_seq_lens,        h_seq_lens.data(),        h_seq_lens.size() * sizeof(int32_t),        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_query_start_len, h_query_start_len.data(), h_query_start_len.size() * sizeof(int32_t), hipMemcpyHostToDevice));

    // Descales: ones (FP16 path passes scales of 1.0; FP8 path multiplies in its own per-block scale)
    std::vector<float> h_descale = {1.0f};
    float *d_q_descale = nullptr, *d_k_descale = nullptr, *d_v_descale = nullptr;
    HIP_CHECK(hipMalloc(&d_q_descale, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_k_descale, sizeof(float)));
    HIP_CHECK(hipMalloc(&d_v_descale, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q_descale, h_descale.data(), sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_k_descale, h_descale.data(), sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_v_descale, h_descale.data(), sizeof(float), hipMemcpyHostToDevice));

    // Upload centroid LUTs for layer 0 (= attention layer 3 = first traditional attention).
    // The header gives us per-(kv head, attention-layer-position) tables; for the smoke
    // test we just use the L=3 table for kv=k and kv=v.
    constexpr int N_CENTROIDS = 16;  // turbo4_fp8
    uint8_t *d_centroids_k = nullptr, *d_centroids_v = nullptr;
    HIP_CHECK(hipMalloc(&d_centroids_k, N_CENTROIDS));
    HIP_CHECK(hipMalloc(&d_centroids_v, N_CENTROIDS));
    HIP_CHECK(hipMemcpy(d_centroids_k, mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3, N_CENTROIDS, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_centroids_v, mt_turbo4_fp8_centroids_qwen35_4b_bs256_v_L3, N_CENTROIDS, hipMemcpyHostToDevice));

    // ── Build shape + args ─────────────────────────────────────────────────
    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = HEAD_SIZE;
    shape.num_q_heads  = NUM_Q_HEADS;
    shape.num_kv_heads = NUM_KV_HEADS;
    shape.block_size   = BLOCK_SIZE;
    shape.cache_type   = MT_AITER_CACHE_TURBO4_FP8;  // = 24 (BS=256 production)

    mt_aiter_uattn_args_t args {};
    args.shape              = shape;
    args.q                  = d_q;
    args.k_cache            = d_k;
    args.v_cache            = d_v;
    args.out                = d_out;
    args.segm_output        = d_segm_out;
    args.segm_max           = d_segm_max;
    args.segm_expsum        = d_segm_expsum;
    args.block_tables       = d_block_tables;
    args.seq_lens           = d_seq_lens;
    args.query_start_len    = d_query_start_len;
    args.q_descale          = d_q_descale;
    args.k_descale          = d_k_descale;
    args.v_descale          = d_v_descale;
    args.out_scale          = nullptr;
    args.centroids_k        = d_centroids_k;
    args.centroids_v        = d_centroids_v;
    args.scale              = 1.0f / 16.0f;   // 1/sqrt(head_size)
    args.num_seqs           = NUM_SEQS;
    args.num_q_tokens       = NUM_SEQS * Q_LEN;
    args.block_table_stride = (int64_t) h_block_tables.size();
    args.q_stride_0         = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    args.output_stride_0    = (int64_t) NUM_Q_HEADS * HEAD_SIZE;
    // K/V strides (for turbo-FP8 these are not used by the loader — it computes
    // byte offsets internally — but the launch signature requires them.)
    args.k_stride_0         = (int64_t) BLOCK_SIZE * NUM_KV_HEADS * BYTES_PER_FP8_BLOCK;
    args.k_stride_1         = (int64_t) NUM_KV_HEADS * BYTES_PER_FP8_BLOCK;
    args.k_stride_2         = (int64_t) BYTES_PER_FP8_BLOCK;
    args.v_stride_0         = args.k_stride_0;
    args.v_stride_1         = args.k_stride_1;
    args.v_stride_2         = args.k_stride_2;

    // ── Launch ─────────────────────────────────────────────────────────────
    fprintf(stderr, "# calling mt_aiter_unified_attn (cache_type=%d, JIT compile may take ~5s on first call)...\n",
            shape.cache_type);
    hipStream_t stream = 0;
    hipError_t err = mt_aiter_unified_attn(stream, &args);
    fprintf(stderr, "# mt_aiter_unified_attn returned: %s (%d)\n", hipGetErrorString(err), err);
    if (err != hipSuccess) {
        fprintf(stderr, "FAIL: kernel launch did not succeed\n");
        return 1;
    }

    HIP_CHECK(hipDeviceSynchronize());
    fprintf(stderr, "# hipDeviceSynchronize OK\n");

    // ── Sanity check the output: not all zeros, not NaN ────────────────────
    std::vector<_Float16> h_out((size_t) NUM_SEQS * Q_LEN * NUM_Q_HEADS * HEAD_SIZE);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(_Float16), hipMemcpyDeviceToHost));
    int nz_count = 0, nan_count = 0;
    for (auto v : h_out) {
        float fv = (float) v;
        if (fv != 0.0f) ++nz_count;
        if (fv != fv) ++nan_count;
    }
    fprintf(stderr, "# output: %d non-zero / %zu, %d NaN\n", nz_count, h_out.size(), nan_count);

    // Cleanup
    hipFree(d_q); hipFree(d_k); hipFree(d_v); hipFree(d_out);
    hipFree(d_segm_out); hipFree(d_segm_max); hipFree(d_segm_expsum);
    hipFree(d_block_tables); hipFree(d_seq_lens); hipFree(d_query_start_len);
    hipFree(d_q_descale); hipFree(d_k_descale); hipFree(d_v_descale);
    hipFree(d_centroids_k); hipFree(d_centroids_v);

    fprintf(stderr, "\n=== PASS — kernel launched and returned hipSuccess ===\n");
    if (nan_count == 0) {
        fprintf(stderr, "===        output finite (no NaN)                  ===\n");
    } else {
        fprintf(stderr, "=== WARNING: %d NaN values in output               ===\n", nan_count);
    }
    return 0;
}
