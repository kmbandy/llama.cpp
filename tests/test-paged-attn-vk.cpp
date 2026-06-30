// Dual-backend equivalence + scatter-oracle harness for GGML_OP_PAGED_ATTN_MT.
// Compares Vulkan0 (RX480) against CUDA0 (GTX1070) — the numeric oracle.
// turbo4_0 paged path is RHT-free: dequant = centroid*norm (un-rotated).
//
// RED baseline (Tasks 3-5 turn green): Vulkan does not yet implement the op;
// harness reports EXPECTED-FAIL and exits 0.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <random>
#include <string>

// Find a backend device whose registry name contains reg_substr and init it.
static ggml_backend_t init_backend(const char * reg_substr) {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (std::string(ggml_backend_reg_name(reg)).find(reg_substr) != std::string::npos) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// Deterministic fill: index-seeded, reproducible across processes/backends.
static void fill_f16(ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int64_t n = ggml_nelements(t);
    std::vector<ggml_fp16_t> buf(n);
    for (int64_t i = 0; i < n; ++i) buf[i] = ggml_fp32_to_fp16(dist(rng));
    ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
}

static void fill_i32(ggml_tensor * t, const std::vector<int32_t> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size() * sizeof(int32_t));
}

struct paged_case {
    int head_dim, n_heads, n_kv_heads, block_size, q_len, ctx_len, n_seq;
    ggml_type cache_type;   // GGML_TYPE_F16 or GGML_TYPE_TURBO4_0
};

struct built_graph {
    ggml_context        * ctx;
    ggml_cgraph         * gf;
    ggml_tensor         * out;
    ggml_tensor         * k_cache;
    ggml_tensor         * v_cache;
    ggml_backend_buffer_t buf;
};

static void free_graph(built_graph & g) {
    if (g.buf) ggml_backend_buffer_free(g.buf);
    if (g.ctx) ggml_free(g.ctx);
    g.buf = nullptr;
    g.ctx = nullptr;
}

// Build a paged_attn_mt graph in a no-alloc context (for op-support querying only).
// Returns the op tensor; caller must ggml_free the ctx.
static ggml_tensor * build_op_noalloc(const paged_case & c, ggml_context ** ctx_out) {
    const int HD            = c.head_dim;
    const int total_tokens  = c.q_len * c.n_seq;
    const int max_blocks    = (c.ctx_len + c.block_size - 1) / c.block_size;
    const int n_blocks_total= max_blocks * c.n_seq;
    const int64_t cache_elts= (int64_t)n_blocks_total * c.block_size * c.n_kv_heads * HD;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx  = ggml_init(ip);

    ggml_tensor * q            = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,   HD, c.n_heads,    total_tokens);
    ggml_tensor * k_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,   HD, c.n_kv_heads, total_tokens);
    ggml_tensor * v_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,   HD, c.n_kv_heads, total_tokens);
    ggml_tensor * k_cache      = ggml_new_tensor_1d(ctx, c.cache_type,    cache_elts);
    ggml_tensor * v_cache      = ggml_new_tensor_1d(ctx, c.cache_type,    cache_elts);
    ggml_tensor * block_tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,   max_blocks, c.n_seq);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,   c.n_seq);
    ggml_tensor * q_lens       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,   c.n_seq);
    ggml_tensor * slot_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,   total_tokens);

    const float scale = 1.0f / sqrtf((float)HD);
    ggml_tensor * out = ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables,
                                           context_lens, q_lens, k_cur, v_cur, slot_mapping,
                                           c.block_size, c.n_kv_heads, scale);
    *ctx_out = ctx;
    return out;
}

static built_graph build_case(const paged_case & c, ggml_backend_t backend) {
    const int HD            = c.head_dim;
    const int total_tokens  = c.q_len * c.n_seq;
    const int max_blocks    = (c.ctx_len + c.block_size - 1) / c.block_size;
    const int n_blocks_total= max_blocks * c.n_seq;
    const int64_t cache_elts= (int64_t)n_blocks_total * c.block_size * c.n_kv_heads * HD;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx  = ggml_init(ip);

    ggml_tensor * q            = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,   HD, c.n_heads,    total_tokens);
    ggml_tensor * k_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,   HD, c.n_kv_heads, total_tokens);
    ggml_tensor * v_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16,   HD, c.n_kv_heads, total_tokens);
    ggml_tensor * k_cache      = ggml_new_tensor_1d(ctx, c.cache_type,    cache_elts);
    ggml_tensor * v_cache      = ggml_new_tensor_1d(ctx, c.cache_type,    cache_elts);
    ggml_tensor * block_tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32,   max_blocks, c.n_seq);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,   c.n_seq);
    ggml_tensor * q_lens       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,   c.n_seq);
    ggml_tensor * slot_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32,   total_tokens);

    const float scale = 1.0f / sqrtf((float)HD);
    ggml_tensor * out = ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables,
                                           context_lens, q_lens, k_cur, v_cur, slot_mapping,
                                           c.block_size, c.n_kv_heads, scale);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Deterministic inputs — identical seeds on both backends.
    fill_f16(q, 1); fill_f16(k_cur, 2); fill_f16(v_cur, 3);

    // Cache starts zeroed; scatter writes only the touched slots.
    {
        std::vector<uint8_t> zeros(ggml_nbytes(k_cache), 0);
        ggml_backend_tensor_set(k_cache, zeros.data(), 0, ggml_nbytes(k_cache));
        ggml_backend_tensor_set(v_cache, zeros.data(), 0, ggml_nbytes(v_cache));
    }

    // Single-seq contiguous layout: block_tables[s][b] = s*max_blocks + b.
    std::vector<int32_t> bt(max_blocks * c.n_seq);
    for (int s = 0; s < c.n_seq; ++s)
        for (int b = 0; b < max_blocks; ++b)
            bt[s * max_blocks + b] = s * max_blocks + b;
    fill_i32(block_tables, bt);
    fill_i32(context_lens, std::vector<int32_t>(c.n_seq, c.ctx_len));
    fill_i32(q_lens,       std::vector<int32_t>(c.n_seq, c.q_len));

    // slot_mapping[i] = i  (physical slot == logical position for one seq).
    std::vector<int32_t> slots(total_tokens);
    for (int i = 0; i < total_tokens; ++i) slots[i] = i;
    fill_i32(slot_mapping, slots);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    return { ctx, gf, out, k_cache, v_cache, buf };
}

struct cb_state { double max_err = 0.0; bool any = false; };

// Callback signature must match ggml_backend_eval_callback exactly
// (non-const ggml_tensor * as per ggml-backend.h:420).
static bool cmp_cb(int /*node_index*/, ggml_tensor * t1, ggml_tensor * t2, void * ud) {
    auto * st = (cb_state *) ud;
    const int64_t n = ggml_nelements(t1);

    // Convert to float for comparison regardless of storage type.
    std::vector<float> a(n), b(n);
    if (t1->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> ra(n), rb(n);
        ggml_backend_tensor_get(t1, ra.data(), 0, ggml_nbytes(t1));
        ggml_backend_tensor_get(t2, rb.data(), 0, ggml_nbytes(t2));
        for (int64_t i = 0; i < n; ++i) {
            a[i] = ggml_fp16_to_fp32(ra[i]);
            b[i] = ggml_fp16_to_fp32(rb[i]);
        }
    } else {
        // Assume F32 output (e.g., when accumulation upcasts).
        ggml_backend_tensor_get(t1, a.data(), 0, ggml_nbytes(t1));
        ggml_backend_tensor_get(t2, b.data(), 0, ggml_nbytes(t2));
    }

    for (int64_t i = 0; i < n; ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > st->max_err) st->max_err = e;
    }
    st->any = true;
    return true;
}

int main() {
    ggml_backend_t vk   = init_backend("Vulkan");
    ggml_backend_t cuda = init_backend("CUDA");

    if (!vk) {
        printf("SKIP: no Vulkan backend\n");
        if (cuda) ggml_backend_free(cuda);
        return 0;
    }
    if (!cuda) {
        printf("SKIP: no CUDA backend (build with WITH_CUDA=1)\n");
        ggml_backend_free(vk);
        return 0;
    }

    // Case: 1 seq, head_dim=128, n_heads=8, n_kv_heads=2 (GQA 4:1),
    //       block_size=16, prefill q_len=32, context_len=32, TURBO4_0 cache.
    const paged_case c { 128, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_0 };

    // Build the op in a no-alloc context to query backend support —
    // avoids allocating TURBO4_0 buffers on a backend that may not support them.
    {
        ggml_context * tmp_ctx = nullptr;
        ggml_tensor  * op      = build_op_noalloc(c, &tmp_ctx);
        bool supported         = ggml_backend_supports_op(vk, op);
        ggml_free(tmp_ctx);

        if (!supported) {
            printf("EXPECTED-FAIL: PAGED_ATTN_MT not yet supported on Vulkan\n");
            ggml_backend_free(vk);
            ggml_backend_free(cuda);
            return 0;   // RED baseline — Tasks 3-5 turn this green
        }
    }

    // Vulkan supports the op: run the real dual-backend comparison.
    built_graph gvk  = build_case(c, vk);
    built_graph gcuda = build_case(c, cuda);

    cb_state st;
    std::vector<const ggml_tensor *> nodes = { gvk.out };
    bool ok = ggml_backend_compare_graph_backend(
        vk, cuda, gvk.gf, cmp_cb, &st,
        nodes.data(), nodes.size());

    const double tol = 5e-2;   // turbo4-class 4-bit centroid quant; tighten once measured
    printf("paged turbo4_0 prefill: max_err=%.6f tol=%.6f %s\n",
           st.max_err, tol,
           (ok && st.max_err <= tol) ? "PASS" : "FAIL");

    free_graph(gvk);
    free_graph(gcuda);
    ggml_backend_free(vk);
    ggml_backend_free(cuda);

    return (ok && st.max_err <= tol) ? 0 : 1;
}
