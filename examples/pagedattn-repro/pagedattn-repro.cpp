// pagedattn-repro: standalone Vulkan-vs-CUDA-vs-f64-reference repro for the
// SP2.5 LFM2.5 paged-attn divergence investigation. NOT for merge.
//
// Loads the REAL Q/K/V tensors captured by pagedattn-dump at LFM2.5's first
// attention layer (paged_q_cast_l2.bin, paged_k_cast_l2.bin, paged_v_cast_l2.bin
// under /tmp/pagedattn_dump_cuda0_v2/), replays them through the exact same
// GGML_OP_PAGED_ATTN_MT op (head_dim=64, n_heads=32, n_kv_heads=8, turbo4_64
// cache, single-seq 16-token causal prefill) on both Vulkan0 and CUDA0, and
// ALSO computes a float64 causal-softmax reference directly from the loaded
// (unquantized) Q/K/V. Reports max_abs_diff for Vulkan-vs-CUDA,
// Vulkan-vs-f64ref, and CUDA-vs-f64ref, to determine which GPU backend
// deviates further from ground truth.

#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

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

static bool load_f16(const std::string & path, std::vector<ggml_fp16_t> & out, size_t n_expected) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "failed to open %s\n", path.c_str()); return false; }
    out.resize(n_expected);
    f.read((char *) out.data(), (std::streamsize)(n_expected * sizeof(ggml_fp16_t)));
    if ((size_t) f.gcount() != n_expected * sizeof(ggml_fp16_t)) {
        fprintf(stderr, "%s: short read (got %zd, want %zu bytes)\n", path.c_str(),
                (ssize_t) f.gcount(), n_expected * sizeof(ggml_fp16_t));
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    const std::string dir = (argc > 1) ? argv[1] : "/tmp/pagedattn_dump_cuda0_v2";

    // Real-model shape at LFM2.5's first attention layer (model layer 2):
    // head_dim=64, n_heads=32, n_kv_heads=8, q_len=ctx_len=16 (single prefill), n_seq=1.
    const int HD = 64, N_HEADS = 32, N_KV_HEADS = 8, Q_LEN = 16, CTX_LEN = 16, N_SEQ = 1;
    const int BLOCK_SIZE = 16; // matches the only CUDA head_size=64 template instantiation available
    const int total_tokens = Q_LEN * N_SEQ;
    const int max_blocks   = (CTX_LEN + BLOCK_SIZE - 1) / BLOCK_SIZE;

    std::vector<ggml_fp16_t> h_q, h_k, h_v;
    if (!load_f16(dir + "/paged_q_cast_l2.bin", h_q, (size_t) HD * N_HEADS    * total_tokens)) return 1;
    if (!load_f16(dir + "/paged_k_cast_l2.bin", h_k, (size_t) HD * N_KV_HEADS * total_tokens)) return 1;
    if (!load_f16(dir + "/paged_v_cast_l2.bin", h_v, (size_t) HD * N_KV_HEADS * total_tokens)) return 1;

    ggml_backend_t vk   = init_backend("Vulkan");
    ggml_backend_t cuda = init_backend("CUDA");
    if (!vk)   { fprintf(stderr, "no Vulkan backend found\n"); return 1; }
    if (!cuda) { fprintf(stderr, "no CUDA backend found\n"); return 1; }

    auto build_and_run = [&](ggml_backend_t backend, std::vector<float> & out_f32) -> bool {
        const int64_t cache_elts = (int64_t) max_blocks * N_SEQ * BLOCK_SIZE * N_KV_HEADS * HD;

        ggml_init_params ip = { ggml_tensor_overhead() * 64 + ggml_graph_overhead(), nullptr, true };
        ggml_context * ctx  = ggml_init(ip);

        ggml_tensor * q            = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, N_HEADS,    total_tokens);
        ggml_tensor * k_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, N_KV_HEADS, total_tokens);
        ggml_tensor * v_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, N_KV_HEADS, total_tokens);
        ggml_tensor * k_cache      = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO4_64, cache_elts);
        ggml_tensor * v_cache      = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO4_64, cache_elts);
        ggml_tensor * block_tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, N_SEQ);
        ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_SEQ);
        ggml_tensor * q_lens       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N_SEQ);
        ggml_tensor * slot_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_tokens);

        const float scale = 1.0f / sqrtf((float) HD);
        ggml_tensor * out = ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables,
                                                context_lens, q_lens, k_cur, v_cur, slot_mapping,
                                                BLOCK_SIZE, N_KV_HEADS, scale);

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { fprintf(stderr, "alloc failed\n"); ggml_free(ctx); return false; }

        ggml_backend_tensor_set(q,     h_q.data(), 0, h_q.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(k_cur, h_k.data(), 0, h_k.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(v_cur, h_v.data(), 0, h_v.size() * sizeof(ggml_fp16_t));

        std::vector<int32_t> bt(max_blocks * N_SEQ);
        for (int s = 0; s < N_SEQ; ++s)
            for (int b = 0; b < max_blocks; ++b)
                bt[s * max_blocks + b] = s * max_blocks + b;
        ggml_backend_tensor_set(block_tables, bt.data(), 0, bt.size() * sizeof(int32_t));
        std::vector<int32_t> cl(N_SEQ, CTX_LEN);
        ggml_backend_tensor_set(context_lens, cl.data(), 0, cl.size() * sizeof(int32_t));
        std::vector<int32_t> ql(N_SEQ, Q_LEN);
        ggml_backend_tensor_set(q_lens, ql.data(), 0, ql.size() * sizeof(int32_t));
        std::vector<int32_t> slots(total_tokens);
        for (int i = 0; i < total_tokens; ++i) slots[i] = i;
        ggml_backend_tensor_set(slot_mapping, slots.data(), 0, slots.size() * sizeof(int32_t));

        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);

        const ggml_status st = ggml_backend_graph_compute(backend, gf);
        if (st != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "graph_compute failed: %d\n", (int) st);
            ggml_backend_buffer_free(buf);
            ggml_free(ctx);
            return false;
        }

        const int64_t n = ggml_nelements(out);
        std::vector<ggml_fp16_t> raw(n);
        ggml_backend_tensor_get(out, raw.data(), 0, ggml_nbytes(out));
        out_f32.resize(n);
        for (int64_t i = 0; i < n; ++i) out_f32[i] = ggml_fp16_to_fp32(raw[i]);

        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return true;
    };

    std::vector<float> out_vk, out_cuda;
    if (!build_and_run(vk,   out_vk))   { fprintf(stderr, "Vulkan run failed\n"); return 1; }
    if (!build_and_run(cuda, out_cuda)) { fprintf(stderr, "CUDA run failed\n"); return 1; }

    // ── float64 causal-softmax reference, computed directly from the loaded
    //    (unquantized) real Q/K/V — ground truth, independent of either
    //    backend's kernel or the turbo4_64 quantizer. ──
    std::vector<double> out_ref((size_t) HD * N_HEADS * total_tokens, 0.0);
    const double scale_d = 1.0 / std::sqrt((double) HD);
    const int gqa_ratio = N_HEADS / N_KV_HEADS;
    for (int head = 0; head < N_HEADS; ++head) {
        const int kv_head = head / gqa_ratio;
        for (int qi = 0; qi < Q_LEN; ++qi) {
            std::vector<double> scores(qi + 1);
            double max_score = -1e300;
            for (int kv_pos = 0; kv_pos <= qi; ++kv_pos) {
                double dot = 0.0;
                for (int d = 0; d < HD; ++d) {
                    const double qv = ggml_fp16_to_fp32(h_q[((size_t) qi * N_HEADS + head) * HD + d]);
                    const double kv = ggml_fp16_to_fp32(h_k[((size_t) kv_pos * N_KV_HEADS + kv_head) * HD + d]);
                    dot += qv * kv;
                }
                scores[kv_pos] = dot * scale_d;
                if (scores[kv_pos] > max_score) max_score = scores[kv_pos];
            }
            double denom = 0.0;
            std::vector<double> w(qi + 1);
            for (int kv_pos = 0; kv_pos <= qi; ++kv_pos) {
                w[kv_pos] = std::exp(scores[kv_pos] - max_score);
                denom += w[kv_pos];
            }
            for (int d = 0; d < HD; ++d) {
                double acc = 0.0;
                for (int kv_pos = 0; kv_pos <= qi; ++kv_pos) {
                    const double vv = ggml_fp16_to_fp32(h_v[((size_t) kv_pos * N_KV_HEADS + kv_head) * HD + d]);
                    acc += (w[kv_pos] / denom) * vv;
                }
                out_ref[((size_t) qi * N_HEADS + head) * HD + d] = acc;
            }
        }
    }

    auto compare = [&](const char * label, const std::vector<float> & a, const std::vector<double> & b) {
        double max_abs = 0.0, sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double diff = (double) a[i] - b[i];
            max_abs = std::max(max_abs, std::fabs(diff));
            sum_sq_diff += diff * diff;
            sum_sq_ref  += b[i] * b[i];
        }
        const double rel_l2 = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-12);
        printf("%-24s max_abs_diff=%.6e  rel_L2=%.6e\n", label, max_abs, rel_l2);
    };
    auto compare_f = [&](const char * label, const std::vector<float> & a, const std::vector<float> & b) {
        double max_abs = 0.0, sum_sq_diff = 0.0, sum_sq_ref = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double diff = (double) a[i] - (double) b[i];
            max_abs = std::max(max_abs, std::fabs(diff));
            sum_sq_diff += diff * diff;
            sum_sq_ref  += (double) b[i] * (double) b[i];
        }
        const double rel_l2 = std::sqrt(sum_sq_diff) / (std::sqrt(sum_sq_ref) + 1e-12);
        printf("%-24s max_abs_diff=%.6e  rel_L2=%.6e\n", label, max_abs, rel_l2);
    };

    printf("\n=== pagedattn-repro: real LFM2.5 layer-2 Q/K/V, turbo4_64, hd=64/n_kv=8/n_heads=32 ===\n");
    compare_f("Vulkan vs CUDA",     out_vk,   out_cuda);
    compare  ("Vulkan vs f64-ref",  out_vk,   out_ref);
    compare  ("CUDA vs f64-ref",    out_cuda, out_ref);

    return 0;
}
