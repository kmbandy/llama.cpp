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
        const std::vector<float> & activations,
        int64_t k = K,
        int64_t m = M,
        int64_t pad_n = 0) {
    const int64_t src1_n = pad_n > n ? pad_n : n;
    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(4, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create MUL_MAT pin context");

    ggml_tensor * weight = ggml_new_tensor_2d(ctx, type, k, m);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, src1_n);
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
    if (src1_n == n) {
        ggml_backend_tensor_set_async(backend, input, activations.data(), 0, (size_t) (k * n) * sizeof(float));
    } else {
        std::vector<float> padded((size_t) (k * src1_n), 0.0f);
        std::copy(activations.begin(), activations.begin() + (size_t) (k * n), padded.begin());
        ggml_backend_tensor_set_async(backend, input, padded.data(), 0, padded.size() * sizeof(float));
    }
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
        std::printf("test-wp-mul-mat-pin: bench %s %s n=%3d pad=%3d [%lld,%lld] hint=%d %8.3f ms/iter\n",
                    ggml_backend_name(backend), ggml_type_name(type), (int) n, (int) src1_n,
                    (long long) k, (long long) m,
                    std::getenv("WP_PIN_TEST_NOHINT") == nullptr ? 1 : 0, ms);
    }
    std::vector<float> full((size_t) (m * src1_n));
    ggml_backend_tensor_get(output, full.data(), 0, full.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    if (src1_n == n) {
        return full;
    }
    std::vector<float> result((size_t) (m * n));
    std::memcpy(result.data(), full.data(), result.size() * sizeof(float));
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
    // WP_PIN_TEST_ID_DUP=1: every token lists each of two slots twice ([a, a, b, b]), the shape the
    // worker's zero-weight padding can produce when the pad slot is also a real route
    static const bool dup = std::getenv("WP_PIN_TEST_ID_DUP") != nullptr;
    if (dup) {
        return (int32_t) ((t * 7 + (k / 2) * 5 + (t / 3)) % N_SLOTS);
    }
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

    // WP_PIN_TEST_ID_EXPAND=1: give every (k, token) its own activation row ([K, N_USED, n])
    // instead of the broadcast [K, 1, n] shape, which selects the launcher's dedup/scatter path
    static const bool expand = std::getenv("WP_PIN_TEST_ID_EXPAND") != nullptr;
    ggml_tensor * as    = ggml_new_tensor_3d(ctx, type, K, M, N_SLOTS);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, expand ? N_USED : 1, n);
    ggml_tensor * ids   = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, n);
    ggml_tensor * output = ggml_mul_mat_id(ctx, as, input, ids);
    if (std::getenv("WP_PIN_TEST_NOHINT") == nullptr) {
        ggml_mul_mat_id_set_hint(output, GGML_HINT_MUL_MAT_PIN);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, output);

    // WP_PIN_TEST_ID_TIGHT=1: put the expert slab in its own buffer of exactly its byte size, so
    // any kernel read past the last expert's last row lands outside the allocation (the worker's
    // arena ends exactly at its last slot; 2026-09-02 R9700 page fault in the grouped prefill)
    static const bool tight = std::getenv("WP_PIN_TEST_ID_TIGHT") != nullptr;
    ggml_backend_buffer_t as_buffer = nullptr;
    if (tight) {
        // 64 MiB is a multiple of every allocator granule seen here (2 MiB on HIP/CUDA VMM), so
        // a slab placed at the END of it ends exactly where the mapping ends
        const size_t tight_bytes = (size_t) 64 << 20;
        require(ggml_nbytes(as) <= tight_bytes, "expert slab larger than the tight buffer");
        as_buffer = ggml_backend_alloc_buffer(backend, tight_bytes);
        require(as_buffer != nullptr, "failed to allocate the tight expert slab buffer");
        const size_t align = ggml_backend_buffer_get_alignment(as_buffer);
        const size_t tail = ((tight_bytes - ggml_nbytes(as)) / align) * align;
        require(ggml_backend_tensor_alloc(as_buffer, as, (char *) ggml_backend_buffer_get_base(as_buffer) + tail) == GGML_STATUS_SUCCESS,
                "failed to place the expert slab at the end of its tight buffer");
    }
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate MUL_MAT_ID pin tensors");
    std::vector<int32_t> ids_host((size_t) (N_USED * n));
    for (int64_t t = 0; t < n; ++t) {
        for (int64_t k = 0; k < N_USED; ++k) {
            ids_host[(size_t) (t * N_USED + k)] = slot_for(k, t);
        }
    }
    ggml_backend_tensor_set_async(backend, as, quantized_slots.data(), 0, quantized_slots.size());
    if (expand) {
        std::vector<float> expanded((size_t) (K * N_USED * n));
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t k = 0; k < N_USED; ++k) {
                std::copy(activations.begin() + t * K, activations.begin() + (t + 1) * K,
                          expanded.begin() + (t * N_USED + k) * K);
            }
        }
        ggml_backend_tensor_set_async(backend, input, expanded.data(), 0, expanded.size() * sizeof(float));
        ggml_backend_synchronize(backend);
    } else {
        ggml_backend_tensor_set_async(backend, input, activations.data(), 0, (size_t) (K * n) * sizeof(float));
    }
    ggml_backend_tensor_set_async(backend, ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
    ggml_backend_synchronize(backend);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "pinned MUL_MAT_ID graph compute failed");
    ggml_backend_synchronize(backend);
    std::vector<float> result((size_t) (M * N_USED * n));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    if (as_buffer != nullptr) {
        ggml_backend_buffer_free(as_buffer);
    }
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

// Padded mmvq pin path: src1 has 8 columns (tail zero) so GGML_MUL_MAT_PIN_KERNEL=mmvq
// skips per-mul_mat memset/memcpy. ncols 1..8 must match the unpadded n=8 prefix.
void test_padded(ggml_backend_t backend, ggml_type type, int64_t k, int64_t m) {
    constexpr int64_t pad = 8;
    std::vector<float> weights((size_t) (k * m));
    std::vector<float> activations((size_t) (k * pad));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = ((int) ((i * 29 + i / (size_t) k * 7) % 251) - 125) * 0.0007f;
    }
    for (size_t i = 0; i < activations.size(); ++i) {
        activations[i] = ((int) ((i * 17 + i / (size_t) k * 11) % 127) - 63) * 0.003f;
    }
    std::vector<uint8_t> quantized(ggml_row_size(type, k) * (size_t) m);
    std::vector<float> imatrix((size_t) k, 1.0f);
    const float * imatrix_data = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;
    require(ggml_quantize_chunk(type, weights.data(), quantized.data(), 0, m, k, imatrix_data) == quantized.size(),
            std::string("failed to quantize padded ") + ggml_type_name(type));

    const std::vector<float> reference = compute(
        backend, type, pad, quantized, activations, k, m, 0);
    for (int64_t n = 1; n <= pad; ++n) {
        const std::vector<float> actual = compute(
            backend, type, n, quantized, activations, k, m, pad);
        size_t n_diff = 0;
        float max_diff = 0.0f;
        int64_t first_col = -1;
        for (size_t i = 0; i < actual.size(); ++i) {
            if (std::memcmp(&actual[i], &reference[i], sizeof(float)) != 0) {
                if (n_diff == 0) {
                    first_col = (int64_t) (i / (size_t) m);
                }
                ++n_diff;
                max_diff = std::max(max_diff, std::fabs(actual[i] - reference[i]));
            }
        }
        if (n_diff != 0) {
            std::printf("test-wp-mul-mat-pin: padded %s [%lld,%lld] ne11=%lld vs 8: %zu/%zu values, max |diff| %.3e, first column %lld\n",
                        ggml_type_name(type), (long long) k, (long long) m, (long long) n,
                        n_diff, actual.size(), max_diff, (long long) first_col);
            failures = failures + 1;
        }
    }
}

// ---------------------------------------------------------------------------
// WP_PIN_TEST_ID_BENCH=<reps>: production-shaped MUL_MAT_ID bench.
//
// The MoE expert worker prefills a 128-token chunk per layer either as ONE
// ggml_mul_mat_id over a strided slab of ~499 expert slots (ids [10, 128]), or
// as one pinned ggml_mul_mat per routed expert over that expert's gathered token
// rows. This mode builds both graphs over the SAME slab and the SAME routing and
// times them, so `mul_mat_id vs the equivalent gather sequence` can be compared
// per backend and per quant type.
//
// Overrides: WP_PIN_TEST_ID_BENCH_SLOTS (default 499), _TOKENS (128), _USED (10),
//            _ROUTED (180), _ROWS (expert rows M, default 640).
// ---------------------------------------------------------------------------

int env_int(const char * name, int fallback) {
    const char * e = std::getenv(name);
    if (e == nullptr) {
        return fallback;
    }
    const int v = std::atoi(e);
    return v > 0 ? v : fallback;
}

struct id_bench_config {
    int64_t n_slots;
    int64_t n_tokens;
    int64_t n_used;
    int64_t n_routed;
    int64_t rows;   // M: rows of one expert matrix
};

id_bench_config id_bench_cfg() {
    id_bench_config c;
    c.n_slots  = env_int("WP_PIN_TEST_ID_BENCH_SLOTS",  499);
    c.n_tokens = env_int("WP_PIN_TEST_ID_BENCH_TOKENS", 128);
    c.n_used   = env_int("WP_PIN_TEST_ID_BENCH_USED",    10);
    c.n_routed = env_int("WP_PIN_TEST_ID_BENCH_ROUTED", 180);
    c.rows     = env_int("WP_PIN_TEST_ID_BENCH_ROWS",   640);
    c.n_routed = std::min(c.n_routed, c.n_slots);
    c.n_used   = std::min(c.n_used,   c.n_routed);
    return c;
}

void bench_id(ggml_backend_t backend, ggml_type type) {
    static const int reps = env_int("WP_PIN_TEST_ID_BENCH", 0);
    if (reps <= 0) {
        return;
    }
    const id_bench_config cfg = id_bench_cfg();
    const int64_t BM = cfg.rows;
    const bool    hint = std::getenv("WP_PIN_TEST_NOHINT") == nullptr;

    // routing: 10 DISTINCT slots per token, drawn from a pool of cfg.n_routed
    // "routed" slots spread across the slab (the worker's arena is strided, so the
    // touched slots are not the first n_routed of the slab)
    auto pool_slot = [&](int64_t j) { return (int32_t) ((j * cfg.n_slots) / cfg.n_routed); };
    std::vector<int32_t> ids_host((size_t) (cfg.n_used * cfg.n_tokens));
    std::vector<std::vector<int64_t>> rows_of_pool((size_t) cfg.n_routed);   // pool -> gathered row idx
    std::vector<int64_t> pool_of_row;                                        // gathered row idx -> pool
    std::vector<int64_t> pair_of_row;                                        // gathered row idx -> t*n_used + k
    {
        std::vector<int64_t> taken((size_t) cfg.n_routed, -1);
        std::vector<std::vector<int64_t>> pairs_of_pool((size_t) cfg.n_routed);
        for (int64_t t = 0; t < cfg.n_tokens; ++t) {
            for (int64_t k = 0; k < cfg.n_used; ++k) {
                uint32_t h = (uint32_t) (t * 2654435761u + k * 40503u + (t >> 3) * 97u);
                h ^= h >> 13;
                int64_t j = (int64_t) (h % (uint32_t) cfg.n_routed);
                while (taken[(size_t) j] == t) {   // de-dup within the token
                    j = (j + 1) % cfg.n_routed;
                }
                taken[(size_t) j] = t;
                ids_host[(size_t) (t * cfg.n_used + k)] = pool_slot(j);
                pairs_of_pool[(size_t) j].push_back(t * cfg.n_used + k);
            }
        }
        for (int64_t j = 0; j < cfg.n_routed; ++j) {
            for (const int64_t pair : pairs_of_pool[(size_t) j]) {
                rows_of_pool[(size_t) j].push_back((int64_t) pool_of_row.size());
                pool_of_row.push_back(j);
                pair_of_row.push_back(pair);
            }
        }
    }
    const int64_t n_gathered = (int64_t) pool_of_row.size();   // == n_tokens * n_used

    // slab: quantize slot by slot from one [K, BM] f32 buffer -- the full f32 slab
    // would be 3.3 GB at the default shape
    const size_t slot_bytes = ggml_row_size(type, K) * (size_t) BM;
    std::vector<uint8_t> slab((size_t) cfg.n_slots * slot_bytes, 0);
    {
        std::vector<float> slot((size_t) (K * BM));
        std::vector<float> imatrix((size_t) K, 1.0f);
        const float * imatrix_data = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;
        for (int64_t j = 0; j < cfg.n_routed; ++j) {   // only routed slots are ever read
            const int64_t s = pool_slot(j);
            for (size_t i = 0; i < slot.size(); ++i) {
                slot[i] = ((int) ((i * 31 + i / K * 5 + (size_t) s * 17) % 241) - 120) * 0.0007f;
            }
            require(ggml_quantize_chunk(type, slot.data(), slab.data() + (size_t) s * slot_bytes,
                                        0, BM, K, imatrix_data) == slot_bytes,
                    std::string("id-bench: failed to quantize expert slot for ") + ggml_type_name(type));
        }
    }
    std::vector<float> act((size_t) (K * cfg.n_tokens));
    for (size_t i = 0; i < act.size(); ++i) {
        act[i] = ((int) ((i * 13 + i / K * 3) % 131) - 65) * 0.003f;
    }

    const size_t n_nodes_gather = (size_t) (3 * cfg.n_routed);
    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * (3 * n_nodes_gather + 32)
                          + ggml_graph_overhead_custom(8, false)
                          + ggml_graph_overhead_custom(n_nodes_gather + 8, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create id-bench context");

    // GROUPED: one mul_mat_id over the whole slab, expanded [K, n_used, n_tokens] input
    ggml_tensor * as     = ggml_new_tensor_3d(ctx, type, K, BM, cfg.n_slots);
    ggml_tensor * input  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, cfg.n_used, cfg.n_tokens);
    ggml_tensor * ids    = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, cfg.n_used, cfg.n_tokens);
    ggml_tensor * out_id = ggml_mul_mat_id(ctx, as, input, ids);
    if (hint) {
        ggml_mul_mat_id_set_hint(out_id, GGML_HINT_MUL_MAT_PIN);
    }
    ggml_cgraph * graph_id = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph_id, out_id);

    // GATHER: same routing, one pinned mul_mat per routed expert against that
    // expert's rows inside one [K, n_gathered] gathered activation tensor
    ggml_tensor * gin = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, n_gathered);
    ggml_cgraph * graph_gather = ggml_new_graph_custom(ctx, n_nodes_gather + 8, false);
    std::vector<ggml_tensor *> gather_out((size_t) cfg.n_routed, nullptr);
    int gather_nodes = 0;
    for (int64_t j = 0; j < cfg.n_routed; ++j) {
        const int64_t n_rows_e = (int64_t) rows_of_pool[(size_t) j].size();
        if (n_rows_e == 0) {
            continue;
        }
        const int64_t s = pool_slot(j);
        ggml_tensor * w = ggml_view_2d(ctx, as, K, BM, as->nb[1], (size_t) s * as->nb[2]);
        ggml_tensor * x = ggml_view_2d(ctx, gin, K, n_rows_e, gin->nb[1],
                                       (size_t) rows_of_pool[(size_t) j][0] * gin->nb[1]);
        ggml_tensor * y = ggml_mul_mat(ctx, w, x);
        if (hint) {
            ggml_mul_mat_set_hint(y, GGML_HINT_MUL_MAT_PIN);
        }
        ggml_build_forward_expand(graph_gather, y);
        gather_out[(size_t) j] = y;
        ++gather_nodes;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate id-bench tensors");

    ggml_backend_tensor_set_async(backend, as, slab.data(), 0, slab.size());
    ggml_backend_tensor_set_async(backend, ids, ids_host.data(), 0, ids_host.size() * sizeof(int32_t));
    {
        std::vector<float> expanded((size_t) (K * cfg.n_used * cfg.n_tokens));
        for (int64_t t = 0; t < cfg.n_tokens; ++t) {
            for (int64_t k = 0; k < cfg.n_used; ++k) {
                std::copy(act.begin() + t * K, act.begin() + (t + 1) * K,
                          expanded.begin() + (t * cfg.n_used + k) * K);
            }
        }
        ggml_backend_tensor_set_async(backend, input, expanded.data(), 0, expanded.size() * sizeof(float));
        std::vector<float> gathered((size_t) (K * n_gathered));
        for (int64_t g = 0; g < n_gathered; ++g) {
            const int64_t t = pair_of_row[(size_t) g] / cfg.n_used;
            std::copy(act.begin() + t * K, act.begin() + (t + 1) * K, gathered.begin() + g * K);
        }
        ggml_backend_tensor_set_async(backend, gin, gathered.data(), 0, gathered.size() * sizeof(float));
        ggml_backend_synchronize(backend);
    }

    // one warm-up + correctness pass each
    require(ggml_backend_graph_compute(backend, graph_id) == GGML_STATUS_SUCCESS, "id-bench grouped compute failed");
    require(ggml_backend_graph_compute(backend, graph_gather) == GGML_STATUS_SUCCESS, "id-bench gather compute failed");
    ggml_backend_synchronize(backend);

    std::vector<float> grouped((size_t) (BM * cfg.n_used * cfg.n_tokens));
    ggml_backend_tensor_get(out_id, grouped.data(), 0, grouped.size() * sizeof(float));
    bool grouped_nonzero = false;
    for (const float v : grouped) {
        require(std::isfinite(v), std::string("id-bench: non-finite grouped result for ") + ggml_type_name(type));
        grouped_nonzero = grouped_nonzero || v != 0.0f;
    }
    require(grouped_nonzero, std::string("id-bench: all-zero grouped result for ") + ggml_type_name(type));
    size_t bad_rows = 0;
    float  worst_rel = 0.0f;
    std::vector<float> row((size_t) BM);
    for (int64_t j = 0; j < cfg.n_routed; ++j) {
        if (gather_out[(size_t) j] == nullptr) {
            continue;
        }
        const int64_t n_rows_e = (int64_t) rows_of_pool[(size_t) j].size();
        std::vector<float> got((size_t) (BM * n_rows_e));
        ggml_backend_tensor_get(gather_out[(size_t) j], got.data(), 0, got.size() * sizeof(float));
        for (int64_t i = 0; i < n_rows_e; ++i) {
            const int64_t pair = pair_of_row[(size_t) rows_of_pool[(size_t) j][(size_t) i]];
            float row_rel = 0.0f;
            for (int64_t r = 0; r < BM; ++r) {
                const float a = got[(size_t) (i * BM + r)];
                const float b = grouped[(size_t) (pair * BM + r)];
                row_rel = std::max(row_rel, std::fabs(a - b) / (std::fabs(b) + 1.0f));
            }
            worst_rel = std::max(worst_rel, row_rel);
            if (row_rel > 2e-2f) {
                ++bad_rows;
            }
        }
    }
    if (bad_rows > 0) {
        ++failures;
        std::printf("test-wp-mul-mat-pin: MISMATCH id-bench %s %s grouped vs gather: %zu/%lld rows differ, worst rel err %.3e\n",
                    ggml_backend_name(backend), ggml_type_name(type), bad_rows, (long long) n_gathered, worst_rel);
    }

    auto time_graph = [&](ggml_cgraph * g) {
        ggml_backend_synchronize(backend);
        const auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < reps; ++r) {
            require(ggml_backend_graph_compute(backend, g) == GGML_STATUS_SUCCESS, "id-bench timed compute failed");
            ggml_backend_synchronize(backend);
        }
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    const double grouped_ms = time_graph(graph_id);
    const double gather_ms  = time_graph(graph_gather);

    std::printf("id-bench backend=%s type=%s tokens=%lld used=%lld slots=%lld routed=%lld grouped_ms=%.3f gather_ms=%.3f gather_nodes=%d hint=%d worst_rel=%.3e\n",
                ggml_backend_name(backend), ggml_type_name(type),
                (long long) cfg.n_tokens, (long long) cfg.n_used, (long long) cfg.n_slots, (long long) cfg.n_routed,
                grouped_ms, gather_ms, gather_nodes, hint ? 1 : 0, worst_rel);
    std::fflush(stdout);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
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
            bench_id(backend, type);
        }
        // Production FFN shapes: gate/up [2560,448] q4_K and down [448,2560] q5_1,
        // padded path at ncols 1..8 (WP_PIN_TEST_BENCH times these too).
        test_padded(backend, GGML_TYPE_Q4_K, 2560, 448);
        test_padded(backend, GGML_TYPE_Q5_1, 448, 2560);

        ggml_backend_free(backend);
        require(failures == 0, std::to_string(failures) + " pinned MUL_MAT shape(s) differ");
        std::puts("test-wp-mul-mat-pin: all tests passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "test-wp-mul-mat-pin: %s\n", error.what());
        return 1;
    }
}
