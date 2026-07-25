// Numeric oracle for the paged-attention DECODE kernels (MAD-392).
//
// WHY THIS EXISTS
// Two separate changes on 2026-07-19 produced kernels that were FAST BECAUSE THEY
// WERE WRONG, and both passed an end-to-end coherence probe (4/4 verifiable answers):
//   1. DECODE_MAX_Q 16->32 let total_q reach 30 while the WMMA decode kernel's row
//      index is `lane % 16` — query heads 4 and 5 were never computed.
//   2. A register-pressure refactor shrank acc[] to one element while loops still
//      indexed acc[0..15].
// Neither is detectable from generated text. This harness compares the fast decode
// paths against the SCALAR path elementwise, which catches both immediately.
//
// THE ORACLE
// The paged op selects its kernel from cached env vars, so each arm must be a
// SEPARATE PROCESS. This binary computes one case and dumps raw f32 output; the
// driver script runs it under three env settings and diffs:
//
//   oracle : GGML_PAGED_TILE=0 GGML_PAGED_DECODE=0        (scalar everything)
//   flash  : GGML_PAGED_DECODE_WMMA=0                     (flash-decode, scalar variant)
//   wmma   : (default)                                    (flash-decode, WMMA variant)
//
// Shapes deliberately cover the production regime AND the band that broke:
// HEAD_SIZE=256 with GQA-6 (24 heads / 4 kv heads) is qwen36 and ornith. q_len is
// swept 1..8 so total_q = 6*q_len spans 6..48, crossing the 16-row WMMA tile
// boundary at q_len=3 — exactly where a row-indexing bug starts dropping heads.
//
// usage: test-paged-decode-oracle <case_idx> <out.bin>
//        exit 0 = computed and dumped; 2 = case skipped (no CUDA backend)

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

struct paged_case {
    const char * name;
    int head_dim;
    int n_heads;
    int n_kv_heads;
    int q_len;
    int n_seq;
    int ctx_len;
    int block_size;
    ggml_type cache_type;
};

// GQA-6 at head_dim 256 == qwen36-27B and ornith-35B.
// q_len 1..8 -> total_q 6..48, crossing the WMMA 16-row tile at q_len 3.
static const paged_case g_cases[] = {
    { "hd256_gqa6_q1",  256, 24, 4, 1, 1, 512, 16, GGML_TYPE_F16 },
    { "hd256_gqa6_q2",  256, 24, 4, 2, 1, 512, 16, GGML_TYPE_F16 },
    { "hd256_gqa6_q3",  256, 24, 4, 3, 1, 512, 16, GGML_TYPE_F16 },
    { "hd256_gqa6_q5",  256, 24, 4, 5, 1, 512, 16, GGML_TYPE_F16 },
    { "hd256_gqa6_q8",  256, 24, 4, 8, 1, 512, 16, GGML_TYPE_F16 },
    { "hd128_gqa4_q5",  128, 16, 4, 5, 1, 512, 16, GGML_TYPE_F16 },
    { "hd256_gqa6_q5_multiseq", 256, 24, 4, 5, 3, 512, 16, GGML_TYPE_F16 },
};
static const int g_num_cases = (int)(sizeof(g_cases)/sizeof(g_cases[0]));

static ggml_backend_t init_cuda_backend() {
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        std::string n = ggml_backend_reg_name(reg);
        if (n.find("CUDA") != std::string::npos || n.find("ROCm") != std::string::npos ||
            n.find("HIP")  != std::string::npos) {
            return ggml_backend_dev_init(dev, nullptr);
        }
    }
    return nullptr;
}

// Deterministic fills — identical across arms so any diff is the kernel, not the data.
static void fill_f16(ggml_tensor * t, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const size_t n = ggml_nelements(t);
    std::vector<ggml_fp16_t> h(n);
    for (size_t i = 0; i < n; ++i) h[i] = ggml_fp32_to_fp16(d(rng));
    ggml_backend_tensor_set(t, h.data(), 0, n*sizeof(ggml_fp16_t));
}
static void fill_i32(ggml_tensor * t, const std::vector<int32_t> & v) {
    ggml_backend_tensor_set(t, v.data(), 0, v.size()*sizeof(int32_t));
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <case_idx> <out.bin>\n", argv[0]); return 1; }
    const int ci = atoi(argv[1]);
    if (ci < 0 || ci >= g_num_cases) { fprintf(stderr, "case idx out of range\n"); return 1; }
    const paged_case & c = g_cases[ci];

    ggml_backend_t backend = init_cuda_backend();
    if (!backend) { fprintf(stderr, "no CUDA/ROCm backend — skipping\n"); return 2; }

    const int HD             = c.head_dim;
    const int total_tokens   = c.q_len * c.n_seq;
    const int max_blocks     = (c.ctx_len + c.block_size - 1) / c.block_size;
    const int n_blocks_total = max_blocks * c.n_seq;
    const int64_t cache_elts = (int64_t)n_blocks_total * c.block_size * c.n_kv_heads * HD;

    ggml_init_params ip = { ggml_tensor_overhead()*64 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q            = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, c.n_heads,    total_tokens);
    ggml_tensor * k_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, c.n_kv_heads, total_tokens);
    ggml_tensor * v_cur        = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, HD, c.n_kv_heads, total_tokens);
    ggml_tensor * k_cache      = ggml_new_tensor_1d(ctx, c.cache_type,  cache_elts);
    ggml_tensor * v_cache      = ggml_new_tensor_1d(ctx, c.cache_type,  cache_elts);
    ggml_tensor * block_tables = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks, c.n_seq);
    ggml_tensor * context_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seq);
    ggml_tensor * q_lens       = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, c.n_seq);
    ggml_tensor * slot_mapping = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, total_tokens);

    const float scale = 1.0f / sqrtf((float)HD);
    ggml_tensor * out = ggml_paged_attn_mt(ctx, q, k_cache, v_cache, block_tables,
                                           context_lens, q_lens, k_cur, v_cur, slot_mapping,
                                           c.block_size, c.n_kv_heads, scale);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }

    fill_f16(q, 1); fill_f16(k_cur, 2); fill_f16(v_cur, 3);
    fill_f16(k_cache, 11); fill_f16(v_cache, 12);

    // Block tables: seq s owns blocks [s*max_blocks, (s+1)*max_blocks).
    std::vector<int32_t> bt(max_blocks * c.n_seq);
    for (int s = 0; s < c.n_seq; ++s)
        for (int b = 0; b < max_blocks; ++b) bt[s*max_blocks + b] = s*max_blocks + b;
    fill_i32(block_tables, bt);

    // Deep context so the decode path (not prefill) is exercised.
    std::vector<int32_t> cl(c.n_seq, c.ctx_len);
    std::vector<int32_t> ql(c.n_seq, c.q_len);
    fill_i32(context_lens, cl);
    fill_i32(q_lens, ql);

    // New tokens land at the tail of each seq's context.
    std::vector<int32_t> sm(total_tokens);
    for (int s = 0; s < c.n_seq; ++s)
        for (int t = 0; t < c.q_len; ++t) {
            const int pos = c.ctx_len - c.q_len + t;
            sm[s*c.q_len + t] = (s*max_blocks + pos/c.block_size)*c.block_size + (pos % c.block_size);
        }
    fill_i32(slot_mapping, sm);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n"); return 1;
    }

    const size_t n = ggml_nelements(out);
    std::vector<float> host(n);
    if (out->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(out, host.data(), 0, n*sizeof(float));
    } else {
        std::vector<ggml_fp16_t> h16(n);
        ggml_backend_tensor_get(out, h16.data(), 0, n*sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; ++i) host[i] = ggml_fp16_to_fp32(h16[i]);
    }

    FILE * f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    fwrite(host.data(), sizeof(float), n, f);
    fclose(f);

    fprintf(stderr, "case %-26s n_out=%zu total_q=%d -> %s\n",
            c.name, n, (c.n_heads/c.n_kv_heads)*c.q_len, argv[2]);
    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    return 0;
}
