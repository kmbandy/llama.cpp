// test-ml8-mul-mat — MAD-223 Phase G.3 ggml-graph-level ml8 matmul round-trip.
//
// Drives `ggml_ml8_mul_mat` through the standard ggml graph compute path and
// verifies output matches the reference formula:
//   y[n, m] = sum_k dequant(W[n, k]) * X[k, m]
//
// Setup mirrors the Phase C.2/C.3 standalone kernel tests:
//   K=64, N=16, M=4 (tiny shape, exercises 1 K-group)
//   W ml8: each block has 32 packed bytes = 0x10 → lo-nibble 0, hi-nibble 1
//     → dequant pattern: 1.0, 2.0, 1.0, 2.0, ... (K-direction)
//   LUT: centroid[0] = fp8(1.0) = 0x38, centroid[1] = fp8(2.0) = 0x40, rest 0
//   W block scale = 1.0
//   X: all ones
//   Expected y[n, m] = sum_k W[n, k] * X[k, m] = K * 1.5 = 96.0 for all (n, m)
//
// This is the same numerical test that mt_ml8_gemm passes on the GPU side,
// just via the ggml graph compute path on CPU.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-ml8.h"

// ggml-common.h is multi-target — caller must declare the language variant
// before including (mirrors how ggml.c / ggml-quants.c do it internally).
#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main(void) {
    std::printf("# ml8 graph-level matmul test (MAD-223 Phase G.3)\n");

    constexpr int64_t K = 64;
    constexpr int64_t N = 16;
    constexpr int64_t M = 4;
    constexpr int64_t n_groups_k = K / QK_ML8;  // = 1

    // ─── ggml context (CPU, owned-alloc) ─────────────────────────────────
    struct ggml_init_params init_params {};
    init_params.mem_size   = 16 * 1024 * 1024;  // 16 MB, plenty for this test
    init_params.mem_buffer = nullptr;
    init_params.no_alloc   = false;
    struct ggml_context * ctx = ggml_init(init_params);
    if (!ctx) { std::printf("[FAIL] ggml_init returned NULL\n"); return 1; }

    // Allocate tensors. ggml allocates blocks of `type_size` × `nelements / blck_size`.
    // For W ml8 (blck_size=64, type_size=36): ne[0]=K, ne[1]=N → total bytes = N*K/64 * 36
    struct ggml_tensor * w   = ggml_new_tensor_2d(ctx, GGML_TYPE_ML8_4,   K, N);
    struct ggml_tensor * lut = ggml_new_tensor_2d(ctx, GGML_TYPE_F8_E4M3, 16, n_groups_k);
    struct ggml_tensor * x   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,     K, M);
    if (!w || !lut || !x) {
        std::printf("[FAIL] ggml_new_tensor_2d returned NULL\n");
        ggml_free(ctx);
        return 1;
    }

    // ─── Populate W (ml8) ───────────────────────────────────────────────
    // N rows × n_groups_k blocks per row; each block = scale + 32 packed bytes
    block_ml8_4 * w_blocks = (block_ml8_4 *) w->data;
    for (int64_t n = 0; n < N; n++) {
        for (int64_t g = 0; g < n_groups_k; g++) {
            block_ml8_4 * blk = &w_blocks[n * n_groups_k + g];
            blk->scale = 1.0f;
            for (int i = 0; i < QK_ML8 / 2; i++) {
                // lo-nibble = 0 (centroid 0 = 1.0), hi-nibble = 1 (centroid 1 = 2.0)
                blk->qs[i] = 0x10;
            }
        }
    }

    // ─── Populate LUT ───────────────────────────────────────────────────
    uint8_t * lut_bytes = (uint8_t *) lut->data;
    std::memset(lut_bytes, 0, 16 * n_groups_k);
    for (int64_t g = 0; g < n_groups_k; g++) {
        lut_bytes[g * 16 + 0] = 0x38;  // fp8 1.0
        lut_bytes[g * 16 + 1] = 0x40;  // fp8 2.0
    }

    // ─── Populate X ────────────────────────────────────────────────────
    float * x_data = (float *) x->data;
    for (int64_t i = 0; i < K * M; i++) x_data[i] = 1.0f;

    // ─── Build graph ───────────────────────────────────────────────────
    struct ggml_tensor * y = ggml_ml8_mul_mat(ctx, w, lut, x);
    if (!y) { std::printf("[FAIL] ggml_ml8_mul_mat returned NULL\n"); ggml_free(ctx); return 1; }
    if (y->ne[0] != N || y->ne[1] != M) {
        std::printf("[FAIL] y shape mismatch: got [%lld, %lld], expected [%lld, %lld]\n",
                    (long long) y->ne[0], (long long) y->ne[1], (long long) N, (long long) M);
        ggml_free(ctx);
        return 1;
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    // ─── Compute (CPU) ─────────────────────────────────────────────────
    const int n_threads = 4;
    ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        std::printf("[FAIL] ggml_graph_compute_with_ctx status=%d\n", (int) st);
        ggml_free(ctx);
        return 1;
    }

    // ─── Verify ────────────────────────────────────────────────────────
    const float expected = (float) K * 1.5f;  // = 96.0
    const float * y_data = (const float *) y->data;
    int n_fail = 0;
    float max_err = 0.0f;
    for (int64_t m = 0; m < M; m++) {
        for (int64_t n = 0; n < N; n++) {
            const float v = y_data[m * N + n];
            const float err = std::fabs(v - expected);
            if (err > max_err) max_err = err;
            if (err > 1e-4f) {
                if (n_fail < 3) {
                    std::printf("  [FAIL] y[%lld, %lld] = %.6g (expected %.6g)\n",
                                (long long) m, (long long) n, v, expected);
                }
                n_fail++;
            }
        }
    }
    std::printf("  expected = %.4f for all elements; got y[0,0] = %.4f, y[%lld,%lld] = %.4f\n",
                expected, y_data[0], (long long)(M-1), (long long)(N-1), y_data[(M-1)*N + (N-1)]);
    std::printf("  max_err = %.6g, mismatches = %d / %lld\n", max_err, n_fail, (long long)(M * N));

    ggml_free(ctx);

    if (n_fail == 0) {
        std::printf("\n=== PASS: ggml_ml8_mul_mat matches reference ===\n");
        return 0;
    } else {
        std::printf("\n=== FAIL: %d mismatches\n", n_fail);
        return 1;
    }
}
