#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#include "wp-gemm.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int N_EMBD = 2560;
constexpr int N_FF = 192;
constexpr int N_WEIGHT_COPIES = 128;

struct BenchType {
    const char * name;
    ggml_type gate_up;
    ggml_type down;
};

constexpr BenchType BENCH_TYPES[] = {
    { "q4k_q5_1", GGML_TYPE_Q4_K, GGML_TYPE_Q5_1 },
    { "q4k_q8_0", GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 },
    { "q5k_q8_0", GGML_TYPE_Q5_K, GGML_TYPE_Q8_0 },
};

struct BufferHolder {
    ggml_backend_buffer_t ptr = nullptr;

    BufferHolder() = default;

    ~BufferHolder() {
        if (ptr != nullptr) {
            ggml_backend_buffer_free(ptr);
        }
    }

    BufferHolder(const BufferHolder &) = delete;
    BufferHolder & operator=(const BufferHolder &) = delete;
};

struct ContextHolder {
    ggml_context * ptr = nullptr;

    ContextHolder() = default;

    ~ContextHolder() {
        if (ptr != nullptr) {
            ggml_free(ptr);
        }
    }

    ContextHolder(const ContextHolder &) = delete;
    ContextHolder & operator=(const ContextHolder &) = delete;
};

struct AlignedStorage {
    std::vector<uint8_t> storage;
    size_t bytes;
    uint8_t * ptr;

    explicit AlignedStorage(size_t size) : storage(size + 63), bytes(size) {
        const uintptr_t raw = reinterpret_cast<uintptr_t>(storage.data());
        ptr = reinterpret_cast<uint8_t *>((raw + 63) & ~static_cast<uintptr_t>(63));
    }

    void * data() {
        return ptr;
    }

    const void * data() const {
        return ptr;
    }

    size_t size() const {
        return bytes;
    }
};

struct GraphHolder {
    ggml_gallocr_t galloc = nullptr;
    ContextHolder ctx;
    BufferHolder weights_buffer;
    BufferHolder input_buffer;
    BufferHolder route_buffer;
    ggml_tensor * gate = nullptr;
    ggml_tensor * up = nullptr;
    ggml_tensor * down = nullptr;
    ggml_tensor * result = nullptr;
    ggml_tensor * input = nullptr;
    ggml_cgraph * cgraph = nullptr;

    GraphHolder() = default;

    ~GraphHolder() {
        if (galloc != nullptr) {
            ggml_gallocr_free(galloc);
        }
    }

    GraphHolder(const GraphHolder &) = delete;
    GraphHolder & operator=(const GraphHolder &) = delete;
};

size_t row_bytes(ggml_type type, int n) {
    return ggml_row_size(type, n);
}

void attach(ggml_tensor * tensor, ggml_backend_buffer_t buffer, void * data) {
    tensor->buffer = buffer;
    tensor->data = data;
}

void fill_random(std::vector<float> & values, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    for (float & value : values) {
        value = dist(rng);
    }
}

void quantize_rows(ggml_type type, const std::vector<float> & source,
                   void * destination, int nrows, int n_per_row) {
    if (ggml_quantize_chunk(type, source.data(), destination, 0, nrows, n_per_row, nullptr) == 0) {
        throw std::runtime_error("quantization failed");
    }
}

struct WeightSet {
    size_t gate_bytes;
    size_t up_bytes;
    size_t down_bytes;
    size_t expert_bytes;
    AlignedStorage data;

    explicit WeightSet(const BenchType & type) :
        gate_bytes(row_bytes(type.gate_up, N_EMBD) * N_FF),
        up_bytes(gate_bytes),
        down_bytes(row_bytes(type.down, N_FF) * N_EMBD),
        expert_bytes(gate_bytes + up_bytes + down_bytes),
        data(expert_bytes * N_WEIGHT_COPIES) {
        std::vector<float> source;
        source.resize(static_cast<size_t>(N_EMBD) * N_FF);
        for (int expert = 0; expert < N_WEIGHT_COPIES; ++expert) {
            uint8_t * base = static_cast<uint8_t *>(data.data()) + static_cast<size_t>(expert) * expert_bytes;

            fill_random(source, 0x1000u + static_cast<uint32_t>(expert));
            quantize_rows(type.gate_up, source, base, N_FF, N_EMBD);

            fill_random(source, 0x2000u + static_cast<uint32_t>(expert));
            quantize_rows(type.gate_up, source, base + gate_bytes, N_FF, N_EMBD);

            source.resize(static_cast<size_t>(N_FF) * N_EMBD);
            fill_random(source, 0x3000u + static_cast<uint32_t>(expert));
            quantize_rows(type.down, source, base + gate_bytes + up_bytes, N_EMBD, N_FF);
            source.resize(static_cast<size_t>(N_EMBD) * N_FF);
        }
    }
};

std::unique_ptr<GraphHolder> build_graph(ggml_backend_t backend, const BenchType & type,
                                         int n_tokens, WeightSet & weights,
                                         float * input_data, float * route_data) {
    auto graph = std::make_unique<GraphHolder>();
    graph->weights_buffer.ptr = ggml_backend_cpu_buffer_from_ptr(
        weights.data.data(), weights.data.size());
    graph->input_buffer.ptr = ggml_backend_cpu_buffer_from_ptr(
        input_data, static_cast<size_t>(N_EMBD) * n_tokens * sizeof(float));
    graph->route_buffer.ptr = ggml_backend_cpu_buffer_from_ptr(
        route_data, static_cast<size_t>(n_tokens) * sizeof(float));
    if (graph->weights_buffer.ptr == nullptr || graph->input_buffer.ptr == nullptr ||
            graph->route_buffer.ptr == nullptr) {
        throw std::runtime_error("failed to wrap benchmark buffers");
    }

    const ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead() * 20 + ggml_graph_overhead_custom(16, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    graph->ctx.ptr = ggml_init(params);
    if (graph->ctx.ptr == nullptr) {
        throw std::runtime_error("failed to allocate benchmark context");
    }

    graph->input = ggml_new_tensor_2d(graph->ctx.ptr, GGML_TYPE_F32, N_EMBD, n_tokens);
    graph->gate = ggml_new_tensor_2d(graph->ctx.ptr, type.gate_up, N_EMBD, N_FF);
    graph->up = ggml_new_tensor_2d(graph->ctx.ptr, type.gate_up, N_EMBD, N_FF);
    graph->down = ggml_new_tensor_2d(graph->ctx.ptr, type.down, N_FF, N_EMBD);
    ggml_tensor * route = ggml_new_tensor_2d(graph->ctx.ptr, GGML_TYPE_F32, 1, n_tokens);
    if (graph->input == nullptr || graph->gate == nullptr || graph->up == nullptr ||
            graph->down == nullptr || route == nullptr) {
        throw std::runtime_error("failed to allocate benchmark tensors");
    }

    attach(graph->input, graph->input_buffer.ptr, input_data);
    uint8_t * weight_base = static_cast<uint8_t *>(weights.data.data());
    attach(graph->gate, graph->weights_buffer.ptr, weight_base);
    attach(graph->up, graph->weights_buffer.ptr, weight_base + weights.gate_bytes);
    attach(graph->down, graph->weights_buffer.ptr,
           weight_base + weights.gate_bytes + weights.up_bytes);
    attach(route, graph->route_buffer.ptr, route_data);

    ggml_tensor * gate_x = ggml_mul_mat(graph->ctx.ptr, graph->gate, graph->input);
    ggml_tensor * up_x = ggml_mul_mat(graph->ctx.ptr, graph->up, graph->input);
    ggml_tensor * hidden = ggml_swiglu_split(graph->ctx.ptr, gate_x, up_x);
    ggml_tensor * output = ggml_mul_mat(graph->ctx.ptr, graph->down, hidden);
    ggml_tensor * weighted = ggml_mul(graph->ctx.ptr, output, route);
    graph->result = ggml_new_tensor_2d(graph->ctx.ptr, GGML_TYPE_F32, N_EMBD, n_tokens);
    ggml_tensor * copy = ggml_cpy(graph->ctx.ptr, weighted, graph->result);
    if (gate_x == nullptr || up_x == nullptr || hidden == nullptr || output == nullptr ||
            weighted == nullptr || graph->result == nullptr || copy == nullptr) {
        throw std::runtime_error("failed to build benchmark graph");
    }

    graph->cgraph = ggml_new_graph_custom(graph->ctx.ptr, 16, false);
    ggml_build_forward_expand(graph->cgraph, copy);
    graph->galloc = ggml_gallocr_new(ggml_backend_cpu_buffer_type());
    if (graph->galloc == nullptr || !ggml_gallocr_alloc_graph(graph->galloc, graph->cgraph)) {
        throw std::runtime_error("failed to allocate benchmark graph");
    }
    if (std::getenv("WP_BENCH_PRINT_GRAPH") != nullptr) {
        ggml_graph_print(graph->cgraph);
    }

    (void) backend;
    return graph;
}

void bind_expert(GraphHolder & graph, WeightSet & weights, int expert) {
    uint8_t * base = static_cast<uint8_t *>(weights.data.data()) +
                     static_cast<size_t>(expert) * weights.expert_bytes;
    graph.gate->data = base;
    graph.up->data = base + weights.gate_bytes;
    graph.down->data = base + weights.gate_bytes + weights.up_bytes;
}

bool compare_dot(ggml_type type, int n, int nrc_x, int nrc_y,
                 const void * weights, size_t weight_stride,
                 const std::vector<uint8_t> & activations, size_t activation_stride) {
    const ggml_type vec_type = ggml_get_type_traits_cpu(type)->vec_dot_type;
    const ggml_vec_dot_t vec_dot = ggml_get_type_traits_cpu(type)->vec_dot;
    const size_t output_stride = static_cast<size_t>(nrc_x);
    std::vector<float> fast(static_cast<size_t>(nrc_x) * nrc_y);
    std::vector<float> reference(fast.size());

    bool dispatched = false;
    if (type == GGML_TYPE_Q4_K && vec_type == GGML_TYPE_Q8_K) {
        dispatched = wp_gemm_q4K_q8K(
            n, nrc_x, nrc_y, weights, weight_stride, activations.data(), activation_stride,
            fast.data(), output_stride);
    } else if (type == GGML_TYPE_Q5_1 && vec_type == GGML_TYPE_Q8_1) {
        dispatched = wp_gemm_q5_1_q8_1(
            n, nrc_x, nrc_y, weights, weight_stride, activations.data(), activation_stride,
            fast.data(), output_stride);
    }
    if (!dispatched) {
        return true;
    }

    for (int iy = 0; iy < nrc_y; ++iy) {
        for (int ix = 0; ix < nrc_x; ++ix) {
            vec_dot(n, &reference[static_cast<size_t>(iy) * output_stride + ix], 0,
                    static_cast<const uint8_t *>(weights) + static_cast<size_t>(ix) * weight_stride,
                    0, activations.data() + static_cast<size_t>(iy) * activation_stride,
                    0, 1);
        }
    }
    return std::memcmp(fast.data(), reference.data(), fast.size() * sizeof(float)) == 0;
}

bool selfcheck(const BenchType & type, int n_tokens, WeightSet & weights,
               const float * input_data) {
    const ggml_type gate_vec_type = ggml_get_type_traits_cpu(type.gate_up)->vec_dot_type;
    const ggml_from_float_t gate_quantize = ggml_get_type_traits_cpu(gate_vec_type)->from_float;
    if (gate_quantize == nullptr) {
        throw std::runtime_error("missing activation quantizer");
    }
    const size_t gate_activation_stride = row_bytes(gate_vec_type, N_EMBD);
    std::vector<uint8_t> gate_activations(gate_activation_stride * n_tokens);
    for (int token = 0; token < n_tokens; ++token) {
        gate_quantize(input_data + static_cast<size_t>(token) * N_EMBD,
                      gate_activations.data() + static_cast<size_t>(token) * gate_activation_stride,
                      N_EMBD);
    }

    const uint8_t * base = static_cast<const uint8_t *>(weights.data.data());
    if (!compare_dot(type.gate_up, N_EMBD, N_FF, n_tokens, base,
                     row_bytes(type.gate_up, N_EMBD), gate_activations,
                     gate_activation_stride)) {
        return false;
    }

    if (type.down == GGML_TYPE_Q5_1) {
        const ggml_type down_vec_type = ggml_get_type_traits_cpu(type.down)->vec_dot_type;
        const ggml_from_float_t down_quantize = ggml_get_type_traits_cpu(down_vec_type)->from_float;
        const size_t down_activation_stride = row_bytes(down_vec_type, N_FF);
        std::vector<float> hidden(static_cast<size_t>(N_FF) * n_tokens);
        std::vector<uint8_t> down_activations(down_activation_stride * n_tokens);
        fill_random(hidden, 0xfeedu + static_cast<uint32_t>(n_tokens));
        for (int token = 0; token < n_tokens; ++token) {
            down_quantize(hidden.data() + static_cast<size_t>(token) * N_FF,
                          down_activations.data() + static_cast<size_t>(token) * down_activation_stride,
                          N_FF);
        }
        if (!compare_dot(type.down, N_FF, N_EMBD, n_tokens,
                         base + weights.gate_bytes + weights.up_bytes,
                         row_bytes(type.down, N_FF), down_activations,
                         down_activation_stride)) {
            return false;
        }
    }
    return true;
}

double run_case(ggml_backend_t backend, const BenchType & type, int n_tokens,
                int repetitions, bool & check_ok, double & checksum) {
    WeightSet weights(type);
    AlignedStorage input_storage(static_cast<size_t>(N_EMBD) * n_tokens * sizeof(float));
    AlignedStorage route_storage(static_cast<size_t>(n_tokens) * sizeof(float));
    float * input = static_cast<float *>(input_storage.data());
    float * route = static_cast<float *>(route_storage.data());
    std::vector<float> input_values(static_cast<size_t>(N_EMBD) * n_tokens);
    fill_random(input_values, 0xabc000u + static_cast<uint32_t>(n_tokens));
    std::memcpy(input, input_values.data(), input_values.size() * sizeof(float));
    std::fill(route, route + n_tokens, 1.0f);

    check_ok = selfcheck(type, n_tokens, weights, input);
    std::unique_ptr<GraphHolder> graph = build_graph(backend, type, n_tokens, weights, input, route);

    for (int i = 0; i < 3; ++i) {
        bind_expert(*graph, weights, i);
        if (ggml_backend_graph_compute(backend, graph->cgraph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("warmup graph compute failed");
        }
    }

    checksum = 0.0;
    const auto started = std::chrono::steady_clock::now();
    for (int i = 0; i < repetitions; ++i) {
        bind_expert(*graph, weights, (i * 17) % N_WEIGHT_COPIES);
        if (ggml_backend_graph_compute(backend, graph->cgraph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("graph compute failed");
        }
        checksum += static_cast<double>(static_cast<const float *>(graph->result->data)[i % (N_EMBD * n_tokens)]);
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    const double us = elapsed * 1.0e6 / repetitions;
    const double gb_s = static_cast<double>(weights.expert_bytes) / (us * 1000.0);
    const double floor_fast = static_cast<double>(weights.expert_bytes) / 15000.0;
    const double floor_slow = static_cast<double>(weights.expert_bytes) / 12000.0;
    std::cout << std::left << std::setw(10) << type.name
              << std::right << std::setw(4) << n_tokens
              << std::setw(12) << std::fixed << std::setprecision(2) << us
              << std::setw(12) << gb_s
              << std::setw(9) << floor_fast << "-" << std::setw(8) << floor_slow
              << std::setw(17) << us / floor_slow << "-" << us / floor_fast
              << "  selfcheck=" << (check_ok ? "ok" : "FAIL") << "\n";
    return us;
}

} // namespace

int main() {
    if (!ggml_cpu_has_avx2() || !ggml_cpu_has_fma()) {
        std::cerr << "bench requires AVX2 and FMA\n";
        return 2;
    }

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (backend == nullptr) {
        std::cerr << "failed to initialize CPU backend\n";
        return 2;
    }
    ggml_backend_cpu_set_n_threads(backend, 1);

    std::cout << "cpu=" << ggml_backend_name(backend)
              << " avx2=" << ggml_cpu_has_avx2()
              << " fma=" << ggml_cpu_has_fma()
              << " wp_gemm=" << (wp_gemm_enabled() ? "on" : "off") << "\n";
    std::cout << "variant   tok   us/expert      GB/s       floor us (15-12)   vs floor (12-15)\n";

    const char * variant_filter = std::getenv("WP_BENCH_VARIANT");
    const int token_filter = std::getenv("WP_BENCH_TOKENS") ? std::atoi(std::getenv("WP_BENCH_TOKENS")) : 0;
    const int repetition_override = std::getenv("WP_BENCH_REPETITIONS") ? std::atoi(std::getenv("WP_BENCH_REPETITIONS")) : 0;
    bool all_ok = true;
    double checksum = 0.0;
    for (const BenchType & type : BENCH_TYPES) {
        for (const int n_tokens : { 1, 2, 4, 8 }) {
            if ((variant_filter != nullptr && std::strcmp(variant_filter, type.name) != 0) ||
                    (token_filter != 0 && token_filter != n_tokens)) {
                continue;
            }
            bool check_ok = false;
            const int repetitions = repetition_override > 0 ? repetition_override : (n_tokens == 1 ? 160 : 80);
            run_case(backend, type, n_tokens, repetitions, check_ok, checksum);
            all_ok = all_ok && check_ok;
        }
    }

    ggml_backend_free(backend);
    std::cerr << "checksum=" << std::setprecision(12) << checksum << "\n";
    return all_ok ? 0 : 1;
}
