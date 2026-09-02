#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int64_t K = 2560;
constexpr int64_t M = 448;
constexpr int64_t N_MAX = 512;

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> compute(
        ggml_backend_t backend,
        ggml_type type,
        int64_t n,
        const std::vector<uint8_t> & quantized,
        const std::vector<float> & activations) {
    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(4, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create MUL_MAT pin context");

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, K, M);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n);
    ggml_tensor * output = ggml_mul_mat(ctx, weight, input);
    if (std::getenv("WP_PIN_TEST_NOHINT") == nullptr) {
        ggml_mul_mat_set_hint(output, GGML_HINT_MUL_MAT_PIN);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate MUL_MAT pin tensors");
    // async on the backend stream: a plain tensor_set from pageable memory can return
    // with the CUDA copy still in flight, and synchronize() only waits on the compute stream
    ggml_backend_tensor_set_async(backend, weight, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set_async(backend, input, activations.data(), 0, (size_t) (K * n) * sizeof(float));
    ggml_backend_synchronize(backend);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "pinned MUL_MAT graph compute failed");

    ggml_backend_synchronize(backend);
    // WP_PIN_TEST_BENCH=<reps>: time the same graph <reps> more times and print ms/iter,
    // so the pinned vs unpinned (WP_PIN_TEST_NOHINT=1) kernel cost can be attributed per backend
    static const int bench_reps = [] { const char * e = std::getenv("WP_PIN_TEST_BENCH"); return e ? std::atoi(e) : 0; }();
    if (bench_reps > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < bench_reps; ++r) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "bench compute failed");
        }
        ggml_backend_synchronize(backend);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / bench_reps;
        std::printf("test-wp-mul-mat-pin: bench %s %s n=%3d hint=%d %8.3f ms/iter\n", ggml_backend_name(backend), ggml_type_name(type), (int) n,
                    std::getenv("WP_PIN_TEST_NOHINT") == nullptr ? 1 : 0, ms);
    }
    std::vector<float> result((size_t) (M * n));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

int failures = 0;

ggml_backend_t g_oracle = nullptr;

// MUL_MAT_ID case: `as` holds N_SLOTS expert matrices, ids [N_USED, n] selects a slot per
// (k, token). The reference is the N_MAX-token product with the same ids on the shared token
// prefix, so row (t, k) must be bit-identical for every n once the op is pinned.
constexpr int64_t N_SLOTS = 6;
constexpr int64_t N_USED  = 4;

int32_t slot_for(int64_t k, int64_t t) {
    return (int32_t) ((t * 7 + k * 5 + (t / 3)) % N_SLOTS);
}

std::vector<float> compute_id(
        ggml_backend_t backend,
        ggml_type type,
        int64_t n,
        const std::vector<uint8_t> & quantized_slots,
        const std::vector<float> & activations) {
    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(4, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create MUL_MAT_ID pin context");

    ggml_tensor * as    = ggml_new_tensor_3d(ctx, type, K, M, N_SLOTS);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, 1, n);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, n);
    ggml_tensor * output = ggml_mul_mat_id(ctx, as, input, ids);
    if (std::getenv("WP_PIN_TEST_NOHINT") == nullptr) {
        ggml_mul_mat_id_set_hint(output, GGML_HINT_MUL_MAT_PIN);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate MUL_MAT_ID pin tensors");
    std::vector<int32_t> ids_host((size_t) (N_USED * n));
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t k = 0; k < N_USED; ++k) {
            ids_host[(size_t) (t * N_USED + k)] = slot_for(k, t);
        }
    }
    ggml_backend_tensor_set_async(backend, as, quantized_slots.data(), 0, quantized_slots.size());
    ggml_backend_tensor_set_async(backend, input, activations.data(), 0, (size_t) (K * n) * sizeof(float));
    ggml_backend_tensor_set_async(backend, ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
    ggml_backend_synchronize(backend);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "pinned MUL_MAT_ID graph compute failed");
    ggml_backend_synchronize(backend);
    std::vector<float> result((size_t) (M * N_USED * n));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

void test_type_id(ggml_backend_t backend, ggml_type type) {
    std::vector<float> weights((size_t) (K * M * N_SLOTS));
    std::vector<float> activations((size_t) (K * N_MAX));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = ((int) ((i * 31 + i / K * 5) % 241) - 120) * 0.0007f;
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = ((int) ((i * 13 + i / K * 3) % 131) - 65) * 0.003f;
    }
    std::vector<uint8_t> quantized(ggml_row_size(type, K) * M * N_SLOTS);
    std::vector<float> imatrix((size_t) K, 1.0f);
    const float * imatrix_data = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;
    require(ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, M * N_SLOTS, K, imatrix_data) == quantized.size(),
            std::string("failed to quantize expert slots for ") + ggml_type_name(type));

    const std::vector<float> reference = compute_id(backend, type, N_MAX, quantized, activations);
    bool any_nonzero = false;
    for (float value : reference) {
        require(std::isfinite(value), std::string("non-finite MUL_MAT_ID reference for ") + ggml_type_name(type));
        any_nonzero = any_nonzero || value != 0.0f;
    }
    require(any_nonzero, std::string("all-zero MUL_MAT_ID reference for ") + ggml_type_name(type));
    std::vector<float> oracle;
    if (g_oracle != nullptr) {
        oracle = compute_id(g_oracle, type, N_MAX, quantized, activations);
    }
    for (const int64_t n : std::array<int64_t, 7>{ 1, 3, 8, 9, 64, 128, 512 }) {
        const std::vector<float> actual = compute_id(backend, type, n, quantized, activations);
        size_t n_diff = 0;
        float max_diff = 0.0f;
        int64_t first_row = -1;
        size_t bad_rows = 0;
        float worst_rel = 0.0f;
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t k = 0; k < N_USED; ++k) {
                float row_rel = 0.0f;
                for (int64_t r = 0; r < M; ++r) {
                    const size_t i = ((size_t) t * N_USED + (size_t) k) * (size_t) M + (size_t) r;
                    if (std::memcmp(&actual[i], &reference[i], sizeof(float)) != 0) {
                        if (n_diff == 0) { first_row = t * N_USED + k; }
                        ++n_diff;
                        max_diff = std::max(max_diff, std::fabs(actual[i] - reference[i]));
                    }
                    if (!oracle.empty()) {
                        row_rel = std::max(row_rel, std::fabs(actual[i] - oracle[i]) / (std::fabs(oracle[i]) + 1.0f));
                    }
                }
                worst_rel = std::max(worst_rel, row_rel);
                if (row_rel > 2e-2f) { ++bad_rows; }
            }
        }
        if (!oracle.empty()) {
            std::printf("test-wp-mul-mat-pin: oracle %s mul_mat_id %s n_tokens=%lld: worst rel err %.3e, bad rows %zu/%lld\n",
                        ggml_backend_name(backend), ggml_type_name(type), (long long) n, worst_rel, bad_rows, (long long) (n * N_USED));
            if (bad_rows > 0) { ++failures; }
        }
        if (n_diff > 0) {
            ++failures;
            std::printf("test-wp-mul-mat-pin: MISMATCH %s mul_mat_id %s n_tokens=%lld vs %lld: %zu values differ, max |diff| %.3e, first row %lld\n",
                        ggml_backend_name(backend), ggml_type_name(type), (long long) n, (long long) N_MAX,
                        n_diff, max_diff, (long long) first_row);
        }
    }
}

void oracle_check(ggml_backend_t backend, ggml_type type, int64_t n,
                  const std::vector<uint8_t> & quantized, const std::vector<float> & activations,
                  const std::vector<float> & actual) {
    if (g_oracle == nullptr) {
        return;
    }
    const std::vector<float> expect = compute(g_oracle, type, n, quantized, activations);
    size_t bad_cols = 0;
    float worst = 0.0f;
    int64_t first_bad = -1;
    for (int64_t c = 0; c < n; ++c) {
        float col_worst = 0.0f;
        for (int64_t r = 0; r < M; ++r) {
            const size_t i = (size_t) c * M + r;
            const float d = std::fabs(actual[i] - expect[i]) / (std::fabs(expect[i]) + 1.0f);
            col_worst = std::max(col_worst, d);
        }
        worst = std::max(worst, col_worst);
        if (col_worst > 2e-2f) {
            if (bad_cols == 0) { first_bad = c; }
            ++bad_cols;
        }
    }
    std::printf("test-wp-mul-mat-pin: oracle %s %s ne11=%lld: worst rel err %.3e, bad columns %zu/%lld%s\n",
                ggml_backend_name(backend), ggml_type_name(type), (long long) n, worst, bad_cols, (long long) n,
                first_bad >= 0 ? (" first bad " + std::to_string(first_bad)).c_str() : "");
    GGML_UNUSED(backend);
}

void test_type(ggml_backend_t backend, ggml_type type) {
    std::vector<float> weights((size_t) (K * M));
    std::vector<float> activations((size_t) (K * N_MAX));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = ((int) ((i * 29 + i / K * 7) % 251) - 125) * 0.0007f;
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = ((int) ((i * 17 + i / K * 11) % 127) - 63) * 0.003f;
    }

    std::vector<uint8_t> quantized(ggml_row_size(type, K) * M);
    std::vector<float> imatrix((size_t) K, 1.0f);
    const float * imatrix_data = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;
    require(ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, M, K, imatrix_data) == quantized.size(),
            std::string("failed to quantize ") + ggml_type_name(type));

    const std::vector<float> reference = compute(
        backend, type, N_MAX, quantized, activations);
    bool any_nonzero = false;
    for (float value : reference) {
        require(std::isfinite(value), std::string("non-finite reference result for ") + ggml_type_name(type));
        any_nonzero = any_nonzero || value != 0.0f;
    }
    require(any_nonzero, std::string("all-zero reference result for ") + ggml_type_name(type));
    oracle_check(backend, type, N_MAX, quantized, activations, reference);
    for (const int64_t n : std::array<int64_t, 7>{ 1, 3, 8, 9, 64, 128, 512 }) {
        const std::vector<float> actual = compute(
            backend, type, n, quantized, activations);
        oracle_check(backend, type, n, quantized, activations, actual);
        size_t n_diff = 0;
        float max_diff = 0.0f;
        int64_t first_col = -1;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &reference[i], sizeof(float)) != 0) {
                if (n_diff == 0) {
                    first_col = (int64_t) (i / (size_t) M);
                }
                ++n_diff;
                max_diff = std::max(max_diff, std::fabs(actual[i] - reference[i]));
            }
        }
        if (n_diff != 0) {
            std::printf("test-wp-mul-mat-pin: %s ne11=%lld differs: %zu/%zu values, max |diff| %.3e, first column %lld\n",
                        ggml_type_name(type), (long long) n, n_diff, actual.size(), max_diff, (long long) first_col);
            failures = failures + 1;
        }
    }
}

} // namespace

int main() {
    try {
        require(setenv("WP_CPU_GEMM", "1", 1) == 0, "failed to enable WP_CPU_GEMM test arm");
        // WP_PIN_TEST_BACKEND selects a backend by name (CPU, ROCm0, CUDA0, Vulkan0)
        const char * backend_name = std::getenv("WP_PIN_TEST_BACKEND");
        ggml_backend_t backend = nullptr;
        if (backend_name == nullptr || std::strcmp(backend_name, "CPU") == 0) {
            backend = ggml_backend_cpu_init();
            require(backend != nullptr, "failed to initialize CPU backend");
            ggml_backend_cpu_set_n_threads(backend, 4);
        } else {
            ggml_backend_load_all();
            backend = ggml_backend_init_by_name(backend_name, nullptr);
            require(backend != nullptr, std::string("failed to initialize backend ") + backend_name);
        }
        std::printf("test-wp-mul-mat-pin: backend %s\n", ggml_backend_name(backend));
        // WP_PIN_TEST_ORACLE=1: also compare every shape against the CPU backend
        if (std::getenv("WP_PIN_TEST_ORACLE") != nullptr && backend_name != nullptr && std::strcmp(backend_name, "CPU") != 0) {
            g_oracle = ggml_backend_cpu_init();
            require(g_oracle != nullptr, "failed to initialize CPU oracle backend");
            ggml_backend_cpu_set_n_threads(g_oracle, 4);
        }

        for (const ggml_type type : {
                 GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                 GGML_TYPE_Q4_K, GGML_TYPE_Q5_K }) {
            test_type(backend, type);
            test_type_id(backend, type);
        }

        ggml_backend_free(backend);
        require(failures == 0, std::to_string(failures) + " pinned MUL_MAT shape(s) differ");
        std::puts("test-wp-mul-mat-pin: all tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-wp-mul-mat-pin: %s\n", error.what());
        return 1;
    }
}
