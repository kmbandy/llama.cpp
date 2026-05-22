// MAD-214 Phase 1G step 1: standalone test for the paged-tile turbo-FP8
// BS=256 cooperative decoder. Validates the GPU decode helper against the
// scalar CPU packer/unpacker. No paged-tile kernel surgery yet — this is
// the foundation we'll integrate in step 2.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 -x hip \
//       -I ggml/include -I ggml/src \
//       tests/test_turbo_fp8_paged_tile_decode.cu \
//       -o /tmp/test_turbo_fp8_paged_tile_decode

#include "../ggml/src/ggml-cuda/mt_pagedattn_turbo_fp8.cuh"
#include "../ggml/src/ggml-cuda/aiter-integration/turbo_fp8_data/qwen35_4b_bs256_centroids.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#define HIP_CHECK(call) do { \
    hipError_t _e = (call); \
    if (_e != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); \
        return 1; \
    } \
} while(0)

constexpr int HEAD_SIZE          = 256;
constexpr int BLOCK_SIZE         = 16;
constexpr int K_TILE_N           = 16;
constexpr int N_WARPS            = 4;
constexpr int N_KV_HEADS         = 4;
constexpr int KV_HEAD_IDX        = 1;  // pick a non-zero head to test indexing
constexpr int N_BLOCKS           = 1;
constexpr int BYTES_PER_FP8_BLOCK = 162;

// CPU packer — same as the smoke test (kept inline to avoid cross-file deps).
static float e4m3_byte_to_fp32_host(uint8_t b) {
    int sign = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 0x7;
    float v = (e == 0) ? (1.0f / 64.0f) * (m / 8.0f) : std::ldexp(1.0f + m / 8.0f, e - 7);
    return sign ? -v : v;
}
static void quantize_block_bs256_host(const float *in, uint8_t *out, const uint8_t *centroids) {
    float scale = 0.0f;
    for (int i = 0; i < 256; ++i) scale = std::max(scale, std::fabs(in[i]));
    if (scale == 0.0f) { std::memset(out, 0, BYTES_PER_FP8_BLOCK); return; }
    _Float16 s16 = (_Float16) scale;
    float seff = (float) s16;
    if (seff == 0.0f) seff = scale;
    std::memcpy(out, &s16, 2);
    uint8_t *qs = out + 2, *signs = out + 130;
    std::memset(qs, 0, 128); std::memset(signs, 0, 32);
    float cv[16];
    for (int k = 0; k < 16; ++k) cv[k] = e4m3_byte_to_fp32_host(centroids[k]);
    for (int i = 0; i < 256; ++i) {
        float v = in[i]; int s = v < 0 ? 1 : 0; float m = std::fabs(v) / seff;
        int best = 0; float be = std::fabs(m - cv[0]);
        for (int k = 1; k < 16; ++k) { float e = std::fabs(m - cv[k]); if (e < be) { best = k; be = e; } }
        if ((i & 1) == 0) qs[i / 2] = best; else qs[i / 2] |= (best << 4);
        signs[i / 8] |= (s & 1) << (i & 7);
    }
}
static void dequantize_block_bs256_host(const uint8_t *in, float *out, const uint8_t *centroids) {
    _Float16 s16; std::memcpy(&s16, in, 2);
    float scale = (float) s16;
    const uint8_t *qs = in + 2, *signs = in + 130;
    for (int i = 0; i < 256; ++i) {
        int idx = ((i & 1) == 0) ? (qs[i / 2] & 0xF) : ((qs[i / 2] >> 4) & 0xF);
        int s = (signs[i / 8] >> (i & 7)) & 1;
        float mag = e4m3_byte_to_fp32_host(centroids[idx]);
        out[i] = (s ? -mag : mag) * scale;
    }
}

// Test kernel that just invokes the cooperative decoder and writes the
// decoded tile to global memory. One block per kv_head, K_TILE_N rows
// × HEAD_SIZE cols, N_WARPS warps × 32 lanes.
__global__ void test_decode_kernel(
        __half *out, const void *cache, const uint8_t *centroids,
        const int *block_table, int block_valid_ctx, int kv_head_idx, int n_kv_heads) {
    __shared__ __half smem[K_TILE_N * HEAD_SIZE];

    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    mt_turbo_fp8::coop_stage_turbo4_fp8_bs256_tile<HEAD_SIZE, BLOCK_SIZE, N_WARPS, K_TILE_N>(
        smem, cache, centroids, block_table,
        /*k_tile_start=*/0, block_valid_ctx, kv_head_idx, n_kv_heads,
        warp_id, lane_id);

    __syncthreads();

    // Cooperative store smem → global
    const int tid = warp_id * 32 + lane_id;
    const int n_threads = N_WARPS * 32;
    for (int i = tid; i < K_TILE_N * HEAD_SIZE; i += n_threads) {
        out[i] = smem[i];
    }
}

int main() {
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop {};
    (void) hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    // Random fp32 KV-cache-shaped data
    std::mt19937 rng(2026);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> kv_fp32(K_TILE_N * N_KV_HEADS * HEAD_SIZE);
    for (auto &v : kv_fp32) v = dist(rng);

    const uint8_t *centroids_host = mt_turbo4_fp8_centroids_qwen35_4b_bs256_k_L3;

    // Pack into AITER paged-cache layout: [N_BLOCKS, BLOCK_SIZE, N_KV_HEADS, 162]
    const size_t kv_bytes = (size_t) N_BLOCKS * BLOCK_SIZE * N_KV_HEADS * BYTES_PER_FP8_BLOCK;
    std::vector<uint8_t> kv_packed(kv_bytes, 0);
    for (int t = 0; t < K_TILE_N; ++t) {
        for (int h = 0; h < N_KV_HEADS; ++h) {
            int blk = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
            size_t off = ((size_t) blk * BLOCK_SIZE * N_KV_HEADS + tok * N_KV_HEADS + h) * BYTES_PER_FP8_BLOCK;
            quantize_block_bs256_host(&kv_fp32[(t * N_KV_HEADS + h) * HEAD_SIZE], &kv_packed[off], centroids_host);
        }
    }

    // CPU dequant of the slice [kv_head=KV_HEAD_IDX, all 16 tokens] —
    // this is what the GPU helper should produce.
    std::vector<float> ref_dq(K_TILE_N * HEAD_SIZE);
    for (int t = 0; t < K_TILE_N; ++t) {
        int blk = t / BLOCK_SIZE, tok = t % BLOCK_SIZE;
        size_t off = ((size_t) blk * BLOCK_SIZE * N_KV_HEADS + tok * N_KV_HEADS + KV_HEAD_IDX) * BYTES_PER_FP8_BLOCK;
        dequantize_block_bs256_host(&kv_packed[off], &ref_dq[t * HEAD_SIZE], centroids_host);
    }

    // GPU upload
    void *d_cache; uint8_t *d_centroids; int *d_block_table; __half *d_out;
    HIP_CHECK(hipMalloc(&d_cache, kv_bytes));
    HIP_CHECK(hipMalloc(&d_centroids, 16));
    HIP_CHECK(hipMalloc(&d_block_table, sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, K_TILE_N * HEAD_SIZE * sizeof(__half)));
    HIP_CHECK(hipMemcpy(d_cache, kv_packed.data(), kv_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_centroids, centroids_host, 16, hipMemcpyHostToDevice));
    const int h_block_table = 0;
    HIP_CHECK(hipMemcpy(d_block_table, &h_block_table, sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_out, 0, K_TILE_N * HEAD_SIZE * sizeof(__half)));

    dim3 block(32, N_WARPS, 1);
    dim3 grid(1, 1, 1);
    test_decode_kernel<<<grid, block>>>(d_out, d_cache, d_centroids, d_block_table,
                                         K_TILE_N, KV_HEAD_IDX, N_KV_HEADS);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<__half> gpu_out(K_TILE_N * HEAD_SIZE);
    HIP_CHECK(hipMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(__half), hipMemcpyDeviceToHost));

    // Compare — the GPU decode should be bit-exact w.r.t. the CPU dequant,
    // modulo fp16 rounding on the final store. Allow <1e-3 abs error.
    double max_err = 0, sum_err = 0;
    int n_bad = 0;
    for (size_t i = 0; i < ref_dq.size(); ++i) {
        float g = __half2float(gpu_out[i]);
        float r = ref_dq[i];
        // Also round CPU ref through fp16 to match storage precision
        _Float16 r16 = (_Float16) r;
        float r_fp16 = (float) r16;
        float err = std::fabs(g - r_fp16);
        max_err = std::max(max_err, (double) err);
        sum_err += err;
        if (err > 1e-3) ++n_bad;
    }
    double mean_err = sum_err / ref_dq.size();
    fprintf(stderr, "max_err = %.6g, mean_err = %.6g, n_bad(>1e-3) = %d/%zu\n",
            max_err, mean_err, n_bad, ref_dq.size());

    if (max_err < 1e-3) {
        fprintf(stderr, "=== PASS — GPU decoder matches CPU reference ===\n");
        return 0;
    } else {
        // Print a few mismatches for diagnosis
        for (size_t i = 0; i < ref_dq.size() && n_bad > 0; ++i) {
            float g = __half2float(gpu_out[i]);
            float r = (float)(_Float16) ref_dq[i];
            if (std::fabs(g - r) > 1e-3) {
                fprintf(stderr, "  [%zu] gpu=%.4f ref=%.4f diff=%.4g\n", i, g, r, g - r);
                if (--n_bad <= 0) break;
            }
        }
        fprintf(stderr, "=== FAIL ===\n");
        return 1;
    }
}
