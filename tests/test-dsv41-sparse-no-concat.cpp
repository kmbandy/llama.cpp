// WP_DSV41_SPARSE_NO_CONCAT correctness test.
//
// build_attention_v41 (src/models/deepseek41.cpp) used to build the sparse
// op's K source by unconditionally concatenating the FULL raw-window and
// compressed KV caches every call (ggml_concat(raw_k, comp_k, 2)), even
// though the op only ever reads the small set of rows kv_indices names
// (window + top-k picks, ~640 rows regardless of context length). This test
// exercises dsv41_sparse_attn_gather_k -- the real production function that
// replaces that concat with a targeted gather -- directly, on synthetic CPU
// tensors, since GGML_OP_SPARSE_ATTN_DSV4 itself is HIP-only and has no CPU
// implementation to run end to end.
//
// What this proves on CPU:
//   - k_sel's rows are byte-identical to the rows the OLD path would have
//     read out of ggml_concat(raw_k, comp_k, 2) at the ORIGINAL kv_indices
//     positions, for every valid (non-padding) slot.
//   - the remapped kv_indices point at exactly those rows within k_sel.
//   - padding slots (-1 in the original kv_indices) remain exactly -1 in
//     the remapped indices, regardless of what garbage row the (clamped)
//     gather fetched for that slot -- i.e. the op would still skip them.
//   - this holds for both a decode-shaped call (nt == 1) and a small
//     multi-token call (nt == 3), including window/comp picks that repeat
//     across tokens and picks that differ per token.
//
// What this does NOT prove (GPU-only): that ggml_sparse_attn_dsv4 itself
// (aiter-integration/kernels/sparse_attention_dsv4.py) produces the same
// attention output when fed k_sel + the remapped indices instead of k_all +
// the original indices, and the actual decode-time perf win (measured via
// WP_OP_PROFILE on real hardware). Those need the GPU run mentioned in the
// task's report.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include "../src/models/models.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void require(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

// One test case: raw_k_len raw rows, n_comp compressed rows, k_win window
// slots, kv_indices values (row-major [n_idx, nt], -1 = pad) supplied by the
// caller so both "all valid" and "some padding" shapes get covered.
void run_case(
        const char * name,
        int64_t d,
        int64_t raw_k_len,
        int64_t n_comp,
        int64_t k_win,
        int64_t nt,
        const std::vector<int32_t> & kv_indices_data) {
    const int64_t n_idx = (int64_t) kv_indices_data.size() / nt;
    require(k_win <= n_idx, "k_win <= n_idx");

    ggml_backend_t backend = ggml_backend_cpu_init();
    require(backend != nullptr, "ggml_backend_cpu_init");

    const ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(64, false),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml_init");

    // raw_k[d_i, r] = 1000 + r*10 + d_i (host-side reference formula, kept
    // out of ggml so the check below is independent of any bug shared
    // between production code and the test).
    ggml_tensor * raw_k  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, raw_k_len, 1);
    // comp_k[d_i, r] = 9000 + r*10 + d_i -- disjoint value range from raw_k
    // so a bug that reads the wrong source (raw vs comp) is caught, not
    // just a bug that reads the wrong row.
    ggml_tensor * comp_k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, n_comp, 1);
    ggml_tensor * kv_indices = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_idx, nt);

    ggml_tensor * kv_indices_new = nullptr;
    ggml_tensor * k_sel = dsv41_sparse_attn_gather_k(
            ctx, raw_k, comp_k, kv_indices, k_win, raw_k_len, n_comp, &kv_indices_new);
    require(k_sel != nullptr && kv_indices_new != nullptr, "gather_k returned tensors");

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, k_sel);
    ggml_build_forward_expand(graph, kv_indices_new);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "ggml_backend_alloc_ctx_tensors");

    std::vector<float> raw_data((size_t) (d * raw_k_len));
    for (int64_t r = 0; r < raw_k_len; ++r) {
        for (int64_t di = 0; di < d; ++di) {
            raw_data[(size_t) (r * d + di)] = 1000.0f + (float) (r * 10 + di);
        }
    }
    std::vector<float> comp_data((size_t) (d * n_comp));
    for (int64_t r = 0; r < n_comp; ++r) {
        for (int64_t di = 0; di < d; ++di) {
            comp_data[(size_t) (r * d + di)] = 9000.0f + (float) (r * 10 + di);
        }
    }
    ggml_backend_tensor_set(raw_k,  raw_data.data(),  0, raw_data.size()  * sizeof(float));
    ggml_backend_tensor_set(comp_k, comp_data.data(), 0, comp_data.size() * sizeof(float));
    ggml_backend_tensor_set(kv_indices, kv_indices_data.data(), 0, kv_indices_data.size() * sizeof(int32_t));

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");

    std::vector<int32_t> idx_new((size_t) (n_idx * nt));
    ggml_backend_tensor_get(kv_indices_new, idx_new.data(), 0, idx_new.size() * sizeof(int32_t));

    const int64_t n_sel = k_sel->ne[2]; // [d, n_head_kv=1, n_sel, 1]
    std::vector<float> sel_data((size_t) (d * n_sel));
    ggml_backend_tensor_get(k_sel, sel_data.data(), 0, sel_data.size() * sizeof(float));

    // Host-side reference for "row v of the would-be k_all = concat(raw_k,
    // comp_k, 2)" -- exactly what the OLD code path would have read at
    // kv_indices[j,t] via the sparse op, without ever materializing k_all.
    auto k_all_row = [&](int64_t v, float * out) {
        if (v < raw_k_len) {
            std::memcpy(out, &raw_data[(size_t) (v * d)], (size_t) d * sizeof(float));
        } else {
            const int64_t r = v - raw_k_len;
            require(r >= 0 && r < n_comp, "comp index in range");
            std::memcpy(out, &comp_data[(size_t) (r * d)], (size_t) d * sizeof(float));
        }
    };

    int64_t n_checked_valid = 0, n_checked_pad = 0;
    std::vector<float> expect(d), got(d);
    for (int64_t t = 0; t < nt; ++t) {
        for (int64_t j = 0; j < n_idx; ++j) {
            const int64_t slot = t * n_idx + j;
            const int32_t orig = kv_indices_data[(size_t) slot];
            const int32_t remapped = idx_new[(size_t) slot];
            if (orig < 0) {
                char msg[128];
                std::snprintf(msg, sizeof(msg), "%s: pad slot t=%lld j=%lld stays -1", name, (long long) t, (long long) j);
                require(remapped == -1, msg);
                n_checked_pad++;
                continue;
            }
            char msg[160];
            std::snprintf(msg, sizeof(msg), "%s: valid slot t=%lld j=%lld remapped index in [0,n_sel)", name, (long long) t, (long long) j);
            require(remapped >= 0 && remapped < n_sel, msg);

            k_all_row(orig, expect.data());
            std::memcpy(got.data(), &sel_data[(size_t) (remapped * d)], (size_t) d * sizeof(float));
            for (int64_t di = 0; di < d; ++di) {
                if (expect[di] != got[di]) {
                    std::fprintf(stderr,
                            "FAIL: %s: t=%lld j=%lld orig_idx=%d remapped=%d d=%lld expect=%f got=%f\n",
                            name, (long long) t, (long long) j, orig, remapped, (long long) di,
                            (double) expect[di], (double) got[di]);
                    std::exit(1);
                }
            }
            n_checked_valid++;
        }
    }

    std::printf("ok: %s (checked %lld valid, %lld pad slots, n_sel=%lld)\n",
            name, (long long) n_checked_valid, (long long) n_checked_pad, (long long) n_sel);

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
}

} // namespace

int main() {
    ggml_backend_load_all();

    constexpr int64_t d = 8; // small synthetic head_dim, not DS4.1's real 512

    // Case 1: decode-shaped (nt=1), every window/comp slot valid, values
    // deliberately spread across the raw/comp ranges (not a contiguous
    // top_k prefix) to catch an off-by-offset bug in the remap.
    {
        constexpr int64_t raw_k_len = 20;
        constexpr int64_t n_comp    = 50;
        constexpr int64_t k_win     = 4;
        constexpr int64_t k_top     = 3; (void) k_top;
        const std::vector<int32_t> kv_indices_data = {
            /*window, all valid:*/ 0, 5, 12, 19,
            /*comp, all valid (offset by raw_k_len):*/ raw_k_len + 0, raw_k_len + 25, raw_k_len + 49,
        };
        static_assert(k_win + k_top == 7, "index width");
        run_case("decode nt=1 all valid", d, raw_k_len, n_comp, k_win, /*nt=*/1, kv_indices_data);
    }

    // Case 2: decode-shaped (nt=1), with the causal/short-window padding
    // real decode hits early in a sequence (a few window slots not yet
    // admitted -> -1) and no top_k cap for this row -> some comp slots -1.
    {
        constexpr int64_t raw_k_len = 20;
        constexpr int64_t n_comp    = 50;
        constexpr int64_t k_win     = 4;
        constexpr int64_t k_top     = 3; (void) k_top;
        const std::vector<int32_t> kv_indices_data = {
            /*window, 2 valid + 2 padding:*/ 0, 1, -1, -1,
            /*comp, 1 valid + 2 padding:*/ raw_k_len + 10, -1, -1,
        };
        run_case("decode nt=1 with padding", d, raw_k_len, n_comp, k_win, /*nt=*/1, kv_indices_data);
    }

    // Case 3: multi-token (nt=3, still under the no-concat threshold),
    // picks differing per token including repeats across tokens (the same
    // physical row picked by two different query tokens must gather
    // correctly for BOTH, each into its own compact slot) and a mix of
    // padding positions per token.
    {
        constexpr int64_t raw_k_len = 12;
        constexpr int64_t n_comp    = 30;
        constexpr int64_t k_win     = 3;
        constexpr int64_t k_top     = 2; (void) k_top;
        const std::vector<int32_t> kv_indices_data = {
            // t=0
            0, 1, 2,               raw_k_len + 5, raw_k_len + 6,
            // t=1: window repeats row 1 (attended by both t=0 and t=1),
            // one comp slot padded (no second top_k pick admitted yet)
            1, 2, -1,               raw_k_len + 5, -1,
            // t=2: fully different window rows, both comp slots valid
            9, 10, 11,              raw_k_len + 0, raw_k_len + 29,
        };
        run_case("multi-token nt=3", d, raw_k_len, n_comp, k_win, /*nt=*/3, kv_indices_data);
    }

    std::printf("ok: WP_DSV41_SPARSE_NO_CONCAT gather/remap matches the k_all-concat reference\n");
    return 0;
}
