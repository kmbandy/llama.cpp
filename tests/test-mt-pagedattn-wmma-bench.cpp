// MAD (WMMA prefill kernel): standalone bench+check for GGML_OP_PAGED_ATTN_MT
// at the Qwen3.8-27B prefill shape (head_dim=256, 24 q-heads/4 kv-heads =
// GQA-6, paged KV cache type GGML_TYPE_TURBO4_FP8_BS256, causal,
// n_tokens=2048, kv=8192), comparing the AITER Triton path (oracle) against
// the hand-written WMMA kernel in mt_pagedattn_wmma_fp8.cu.
//
// Both paths are selected by a PROCESS-LIFETIME env var
// (MT_PAGED_ATTN_WMMA, cached on first read by both
// mt::paged_attn_wmma_fp8_env_enabled() and mt::aiter_backend_enabled()),
// so this binary computes ONE path per invocation and dumps:
//   - the F16 output tensor, raw, to the file given as argv[1]
//   - "elapsed_ms <value>" on stdout (device-side timed, warmed up)
//
// Run both paths and diff with the Python snippet at the bottom of this
// file (or any numpy script) — see the comment block at EOF for the exact
// two invocations and the diff command.
//
// This constructs the paged KV cache via the REAL scatter kernel (fused
// into GGML_OP_PAGED_ATTN_MT itself — see ggml_paged_attn_mt()) rather than
// hand-packing the turbo4_fp8_bs256 byte layout, so both runs exercise the
// identical scatter->attend pipeline the model uses, and only the
// attention kernel choice differs between the two invocations.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include "mt_turbo_fp8_lut_registry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <chrono>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <output.f16.bin> [n_tokens=2048] [kv_len=8192]\n", argv[0]);
        return 1;
    }
    const char * out_path = argv[1];
    const int    n_tokens = argc > 2 ? std::atoi(argv[2]) : 2048;
    const int    kv_len   = argc > 3 ? std::atoi(argv[3]) : 8192;

    const int head_dim    = 256;
    const int n_heads     = 24;
    const int n_kv_heads  = 4;
    const int block_size  = 16;               // TURBO4_FP8_BS256 physical page size (see
                                               // mt_pagedattn_wmma_fp8.cu's header note —
                                               // NOT 256; that's the quant-block width).
    const int n_blocks    = (kv_len + block_size - 1) / block_size + 1;  // +1 headroom
    const float scale     = 1.0f / std::sqrt((float) head_dim);

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::fprintf(stderr, "failed to init CUDA backend\n");
        return 1;
    }

    // ── registry init (real graph builders do this at KV-cache construction
    //    time — see llama-kv-cache-paged.cpp) ──
    mt_turbo_fp8::model_fingerprint fp{};
    fp.arch       = "qwen3moe";
    fp.n_layer    = 1;
    fp.n_embd     = head_dim * n_heads;
    fp.head_dim   = head_dim;
    fp.n_kv_heads = n_kv_heads;
    if (!mt_turbo_fp8::init(fp, /*auto_calibrate_if_missing=*/true)) {
        std::fprintf(stderr, "mt_turbo_fp8::init failed\n");
        return 1;
    }

    ggml_init_params iparams{
        /*.mem_size   =*/ ggml_tensor_overhead() * 32 + ggml_graph_overhead() * 4 + 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, head_dim, n_heads, n_tokens, 1);
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_tokens);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, head_dim, n_kv_heads, n_tokens);

    const int64_t k_elts_per_layer = (int64_t) n_blocks * block_size * n_kv_heads * head_dim;
    ggml_tensor * k_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO4_FP8_BS256, k_elts_per_layer);
    ggml_tensor * v_cache = ggml_new_tensor_1d(ctx, GGML_TYPE_TURBO4_FP8_BS256, k_elts_per_layer);
    ggml_set_name(k_cache, "paged_k_l0");
    ggml_set_name(v_cache, "paged_v_l0");

    ggml_tensor * block_tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_blocks, 1);   // [max_bps, num_seqs=1]
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * q_lens       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
    ggml_tensor * slot_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);

    ggml_tensor * dst = ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables, context_lens,
                                            q_lens, k_cur, v_cur, slot_mapping,
                                            block_size, n_kv_heads, scale);
    // ggml_paged_attn_mt() only fills op_params[0..3]; the graph builder
    // separately sets [4]=max_q_len, [5]=max_ctx_len (MAD-348) — poke them
    // directly here since we're not going through llama-graph.cpp.
    {
        int32_t * p32 = (int32_t *) dst->op_params;
        p32[4] = n_tokens;  // max_q_len: single seq, all tokens are "prefill"
        p32[5] = kv_len;    // max_ctx_len
    }

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "graph allocation failed\n");
        return 1;
    }

    // ── fill inputs ──
    std::mt19937 rng(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    auto fill_f16 = [&](ggml_tensor * t, int64_t n) {
        std::vector<ggml_fp16_t> buf(n);
        for (int64_t i = 0; i < n; ++i) buf[i] = ggml_fp32_to_fp16(dist(rng));
        ggml_backend_tensor_set(t, buf.data(), 0, n * sizeof(ggml_fp16_t));
    };
    fill_f16(q, (int64_t) head_dim * n_heads * n_tokens);
    fill_f16(k_cur, (int64_t) head_dim * n_kv_heads * n_tokens);
    fill_f16(v_cur, (int64_t) head_dim * n_kv_heads * n_tokens);

    {
        // slot_mapping[t] = t (contiguous slots 0..n_tokens-1 — one growing seq)
        std::vector<int32_t> slots(n_tokens);
        for (int t = 0; t < n_tokens; ++t) slots[t] = t;
        ggml_backend_tensor_set(slot_mapping, slots.data(), 0, n_tokens * sizeof(int32_t));

        // block_tables: identity mapping, logical block i -> physical block i
        std::vector<int32_t> bt(n_blocks);
        for (int i = 0; i < n_blocks; ++i) bt[i] = i;
        ggml_backend_tensor_set(block_tables, bt.data(), 0, n_blocks * sizeof(int32_t));

        const int32_t ctxlen = n_tokens;  // single growing seq, no prior context
        const int32_t qlen   = n_tokens;
        ggml_backend_tensor_set(context_lens, &ctxlen, 0, sizeof(int32_t));
        ggml_backend_tensor_set(q_lens, &qlen, 0, sizeof(int32_t));
    }

    // ── run once (warmup + correctness-affecting scatter), then time N reps ──
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    const int reps = 20;
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const auto t1 = std::chrono::steady_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

    std::vector<ggml_fp16_t> out_buf((size_t) head_dim * n_heads * n_tokens);
    ggml_backend_tensor_get(dst, out_buf.data(), 0, out_buf.size() * sizeof(ggml_fp16_t));

    FILE * f = std::fopen(out_path, "wb");
    if (!f) { std::fprintf(stderr, "failed to open %s\n", out_path); return 1; }
    std::fwrite(out_buf.data(), sizeof(ggml_fp16_t), out_buf.size(), f);
    std::fclose(f);

    std::printf("elapsed_ms %.4f\n", elapsed_ms);
    std::printf("shape head_dim=%d n_heads=%d n_tokens=%d kv_len=%d n_kv_heads=%d\n",
                head_dim, n_heads, n_tokens, kv_len, n_kv_heads);

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return 0;
}

// ── how to run the A/B comparison ───────────────────────────────────────
//
//   MT_PAGED_ATTN_WMMA=0 ./test-mt-pagedattn-wmma-bench /tmp/out_aiter.f16 2048 8192
//   MT_PAGED_ATTN_WMMA=1 ./test-mt-pagedattn-wmma-bench /tmp/out_wmma.f16  2048 8192
//
// Then diff (max abs / rel error, NMSE) with:
//
//   python3 - <<'EOF'
//   import numpy as np
//   a = np.fromfile("/tmp/out_aiter.f16", dtype=np.float16).astype(np.float32)
//   b = np.fromfile("/tmp/out_wmma.f16",  dtype=np.float16).astype(np.float32)
//   abs_err = np.abs(a - b)
//   rel_err = abs_err / np.maximum(np.abs(a), 1e-6)
//   nmse = np.sum((a - b) ** 2) / np.sum(a ** 2)
//   print("max_abs_err", abs_err.max())
//   print("max_rel_err", rel_err.max())
//   print("nmse", nmse)
//   EOF
//
// Timing is printed by each invocation on stdout ("elapsed_ms ...").
