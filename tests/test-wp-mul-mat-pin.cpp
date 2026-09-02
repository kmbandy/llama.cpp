#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
    ggml_mul_mat_set_hint(output, GGML_HINT_MUL_MAT_PIN);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate MUL_MAT pin tensors");
    ggml_backend_tensor_set(weight, quantized.data(), 0, quantized.size());
    ggml_backend_tensor_set(input, activations.data(), 0, (size_t) (K * n) * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "pinned MUL_MAT graph compute failed");

    std::vector<float> result((size_t) (M * n));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
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
    for (const int64_t n : std::array<int64_t, 7>{ 1, 3, 8, 9, 64, 128, 512 }) {
        const std::vector<float> actual = compute(
            backend, type, n, quantized, activations);
        const size_t bytes = actual.size() * sizeof(float);
        require(std::memcmp(actual.data(), reference.data(), bytes) == 0,
                std::string("pinned ") + ggml_type_name(type) +
                    " MUL_MAT differs at ne11=" + std::to_string(n));
    }
}

} // namespace

int main() {
    try {
        require(setenv("WP_CPU_GEMM", "1", 1) == 0, "failed to enable WP_CPU_GEMM test arm");
        ggml_backend_t backend = ggml_backend_cpu_init();
        require(backend != nullptr, "failed to initialize CPU backend");
        ggml_backend_cpu_set_n_threads(backend, 4);

        for (const ggml_type type : {
                 GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                 GGML_TYPE_Q4_K, GGML_TYPE_Q5_K }) {
            test_type(backend, type);
        }

        ggml_backend_free(backend);
        std::puts("test-wp-mul-mat-pin: all tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-wp-mul-mat-pin: %s\n", error.what());
        return 1;
    }
}
