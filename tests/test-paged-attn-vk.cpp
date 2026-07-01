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
    ggml_tensor         * k_cur;
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

static void fill_turbo4(ggml_tensor * t, uint32_t seed);    // defined below (after host turbo4 tables)
static void fill_turbo4_64(ggml_tensor * t, uint32_t seed); // defined below (after host turbo4 tables)

static built_graph build_case(const paged_case & c, ggml_backend_t backend, bool fill_cache = false) {
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

    // Cache contents. PREFILL cases (fill_cache=false): zeroed; the 32-token
    // scatter populates the real keys. DECODE cases (fill_cache=true): the op
    // scatters only the single current token into slot 0, so without a pre-fill
    // the entire gathered range [1, ctx_len) would be ZERO and every decode case
    // would collapse to out = softmax[0]*V0 (a degenerate single-term result that
    // never exercises the split-K cross-chunk reduce). Pre-populate every cache
    // slot with deterministic VALID non-zero data so the multi-chunk merge runs
    // over real keys. The cache is sized exactly to ctx_len slots and
    // block_tables[s][b]=b + slot_mapping[i]=i make physical slot == context
    // position, so filling the whole buffer fills exactly the gathered range.
    // Both backends get byte-identical bytes via the leaf-input deep-copy.
    if (!fill_cache) {
        std::vector<uint8_t> zeros(ggml_nbytes(k_cache), 0);
        ggml_backend_tensor_set(k_cache, zeros.data(), 0, ggml_nbytes(k_cache));
        ggml_backend_tensor_set(v_cache, zeros.data(), 0, ggml_nbytes(v_cache));
    } else if (c.cache_type == GGML_TYPE_F16) {
        fill_f16(k_cache, 11); fill_f16(v_cache, 12);
    } else if (c.cache_type == GGML_TYPE_TURBO4_64) {
        fill_turbo4_64(k_cache, 11); fill_turbo4_64(v_cache, 12);
    } else {
        fill_turbo4(k_cache, 11); fill_turbo4(v_cache, 12);
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
    return { ctx, gf, out, k_cache, v_cache, k_cur, buf };
}

// ── turbo4_0 no-RHT host quantizer tables (verbatim from turbo_centroids.glsl) ──
static const float HOST_TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};
static const float HOST_TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};
static uint8_t host_turbo_nearest_4bit(float v) {
    for (int i = 0; i < 15; ++i) {
        if (v < HOST_TURBO_MID_4BIT[i]) return (uint8_t) i;
    }
    return 15;
}

// Quantize one 128-element f32 vector into a valid 68-byte block_turbo4_0
// (norm fp16 + rnorm fp16[reserved=0] + 64 nibble-packed centroid indices),
// matching the no-RHT host quantizer used by the scatter-readback oracle. Used
// to PRE-POPULATE the decode KV cache with deterministic, VALID (no NaN/Inf in
// the norm field) turbo4 blocks so the split-K reduce sees real multi-term data.
static void host_turbo4_quantize_block(const float * x, uint8_t * out /*68 bytes*/) {
    float red[128];
    for (int j = 0; j < 128; ++j) red[j] = x[j] * x[j];
    for (int s = 64; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) red[j] += red[j + s];
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;
    uint8_t idxs[128];
    float   rred[128];
    for (int j = 0; j < 128; ++j) {
        const float nv = x[j] * inv_norm;
        idxs[j] = host_turbo_nearest_4bit(nv);
        const float cv = HOST_TURBO_CENTROIDS_4BIT[idxs[j]];
        rred[j] = cv * cv;
    }
    for (int s = 64; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) rred[j] += rred[j + s];
    const float recon_norm     = sqrtf(rred[0]);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    const ggml_fp16_t norm_h  = ggml_fp32_to_fp16(corrected_norm);
    const ggml_fp16_t rnorm_h = ggml_fp32_to_fp16(0.0f); // reserved/unused in 4-bit mode
    memset(out, 0, 68);
    memcpy(out + 0, &norm_h,  sizeof(ggml_fp16_t));
    memcpy(out + 2, &rnorm_h, sizeof(ggml_fp16_t));
    for (int j = 0; j < 128; ++j)
        out[4 + (j >> 1)] |= (uint8_t)(idxs[j] << ((j & 1) * 4));
}

// Pre-populate a turbo4_0 cache tensor with deterministic VALID blocks (seeded,
// reproducible across processes/backends). Every 68-byte block carries a finite
// norm + valid nibbles, so the dual-backend deep-copy sees byte-identical, NaN-
// free contents on both Vulkan and CUDA.
static void fill_turbo4(ggml_tensor * t, uint32_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const int64_t n_blk = (int64_t)(nbytes / 68);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> buf(nbytes, 0);
    for (int64_t b = 0; b < n_blk; ++b) {
        float x[128];
        for (int j = 0; j < 128; ++j) x[j] = dist(rng);
        host_turbo4_quantize_block(x, &buf[(size_t)b * 68]);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

// Quantize one 64-element f32 vector into a valid 34-byte block_turbo4_64
// (norm fp16 + 32 nibble-packed centroid indices, NO rnorm field). Mirrors
// host_turbo4_quantize_block (128-element/68-byte) exactly, just over 64
// elements / 32 packed bytes, reusing the no-RHT scheme + recon-norm
// correction, but with the N=64-calibrated centroid table (NOT
// HOST_TURBO_CENTROIDS_4BIT, which is turbo4_0's N=128 table) — see
// TURBO_CENTROIDS_4BIT_N64 comment in ggml-cuda/turbo-quant.cuh.
static const float HOST_TURBO_CENTROIDS_4BIT_N64[16] = {
    -0.489086f, -0.332636f, -0.244498f, -0.182456f,
    -0.132429f, -0.089625f, -0.051251f, -0.016052f,
     0.016052f,  0.051251f,  0.089625f,  0.132429f,
     0.182456f,  0.244498f,  0.332636f,  0.489086f
};
static const float HOST_TURBO_MID_4BIT_N64[15] = {
    -0.410861f, -0.288567f, -0.213477f, -0.157443f,
    -0.111027f, -0.070438f, -0.033652f,  0.000000f,
     0.033652f,  0.070438f,  0.111027f,  0.157443f,
     0.213477f,  0.288567f,  0.410861f
};
static uint8_t host_turbo_nearest_4bit_n64(float v) {
    for (int i = 0; i < 15; ++i) {
        if (v < HOST_TURBO_MID_4BIT_N64[i]) return (uint8_t) i;
    }
    return 15;
}
static void host_turbo4_64_quantize_block(const float * x /*64*/, uint8_t * out /*34 bytes*/) {
    float red[64];
    for (int j = 0; j < 64; ++j) red[j] = x[j] * x[j];
    for (int s = 32; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) red[j] += red[j + s];
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;
    uint8_t idxs[64];
    float   rred[64];
    for (int j = 0; j < 64; ++j) {
        const float nv = x[j] * inv_norm;
        idxs[j] = host_turbo_nearest_4bit_n64(nv);
        const float cv = HOST_TURBO_CENTROIDS_4BIT_N64[idxs[j]];
        rred[j] = cv * cv;
    }
    for (int s = 32; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) rred[j] += rred[j + s];
    const float recon_norm     = sqrtf(rred[0]);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
    const ggml_fp16_t norm_h = ggml_fp32_to_fp16(corrected_norm);
    memset(out, 0, 34);
    memcpy(out + 0, &norm_h, sizeof(ggml_fp16_t));
    for (int j = 0; j < 64; ++j)
        out[2 + (j >> 1)] |= (uint8_t)(idxs[j] << ((j & 1) * 4));
}

// Pre-populate a turbo4_64 cache tensor with deterministic VALID blocks
// (seeded, reproducible across processes/backends), mirroring fill_turbo4.
static void fill_turbo4_64(ggml_tensor * t, uint32_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const int64_t n_blk = (int64_t)(nbytes / 34);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> buf(nbytes, 0);
    for (int64_t b = 0; b < n_blk; ++b) {
        float x[64];
        for (int j = 0; j < 64; ++j) x[j] = dist(rng);
        host_turbo4_64_quantize_block(x, &buf[(size_t)b * 34]);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

// Deterministic, CUDA-free scatter-readback oracle (480-only). Runs the op on
// Vulkan, reads back k_cache, and checks each turbo4 block against a host
// re-implementation of the no-RHT quantizer (L2-norm -> normalize -> nearest
// centroid -> nibble-pack -> recon-norm-correct) applied to the same k_cur at
// the expected element_block_index. Asserts per-block norm within 1e-3 and all
// nibbles exactly equal. Returns true on PASS.
static bool scatter_turbo4_readback(const paged_case & c, ggml_backend_t vk) {
    const bool is_t64 = (c.cache_type == GGML_TYPE_TURBO4_64);
    const char * label = is_t64 ? "scatter turbo4_64 readback" : "scatter turbo4_0 readback";

    built_graph g = build_case(c, vk);
    if (ggml_backend_graph_compute(vk, g.gf) != GGML_STATUS_SUCCESS) {
        printf("%s: FAIL (graph compute error)\n", label);
        free_graph(g);
        return false;
    }

    const int HD           = c.head_dim;
    const int total_tokens = c.q_len * c.n_seq;
    const int BS           = c.block_size;
    const int QK            = is_t64 ? 64 : 128;
    const int N_QBLK        = HD / QK;
    const int n_kv_heads    = c.n_kv_heads;
    // turbo4_0: fp16 norm + fp16 rnorm + uint8 qs[64] = 68 B.
    // turbo4_64: fp16 norm (NO rnorm) + uint8 qs[32] = 34 B.
    const size_t BLOCK_BYTES = is_t64 ? 34 : 68;
    const size_t QS_OFFSET   = is_t64 ? 2 : 4;

    std::vector<ggml_fp16_t> kcur(ggml_nelements(g.k_cur));
    ggml_backend_tensor_get(g.k_cur, kcur.data(), 0, ggml_nbytes(g.k_cur));
    std::vector<uint8_t> kc(ggml_nbytes(g.k_cache));
    ggml_backend_tensor_get(g.k_cache, kc.data(), 0, ggml_nbytes(g.k_cache));

    double max_norm_err   = 0.0;
    int    nibble_mismatch = 0;

    for (int t = 0; t < total_tokens; ++t) {
        const int slot          = t;            // slot_mapping[i] = i
        const int paged_block   = slot / BS;
        const int slot_in_block = slot % BS;
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int qb = 0; qb < N_QBLK; ++qb) {
                // Gather the QK-element turbo4 block from k_cur.
                float x[128];
                for (int j = 0; j < QK; ++j) {
                    const int d = qb * QK + j;
                    const size_t src = (size_t) t * n_kv_heads * HD + (size_t) h * HD + d;
                    x[j] = ggml_fp16_to_fp32(kcur[src]);
                }
                // L2 norm — same pairwise tree order as the shader.
                float red[128];
                for (int j = 0; j < QK; ++j) red[j] = x[j] * x[j];
                for (int s = QK / 2; s > 0; s >>= 1)
                    for (int j = 0; j < s; ++j) red[j] += red[j + s];
                const float grp_norm = sqrtf(red[0]);
                const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;
                // Normalize -> (NO Hadamard) -> nearest centroid -> recon norm.
                uint8_t idxs[128];
                float   rred[128];
                for (int j = 0; j < QK; ++j) {
                    const float nv = x[j] * inv_norm;
                    idxs[j] = is_t64 ? host_turbo_nearest_4bit_n64(nv) : host_turbo_nearest_4bit(nv);
                    const float cv = is_t64 ? HOST_TURBO_CENTROIDS_4BIT_N64[idxs[j]] : HOST_TURBO_CENTROIDS_4BIT[idxs[j]];
                    rred[j] = cv * cv;
                }
                for (int s = QK / 2; s > 0; s >>= 1)
                    for (int j = 0; j < s; ++j) rred[j] += rred[j + s];
                const float recon_norm     = sqrtf(rred[0]);
                const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;
                const ggml_fp16_t exp_norm_h = ggml_fp32_to_fp16(corrected_norm);

                const int64_t block_ib = ((int64_t) paged_block * n_kv_heads + h) * BS * N_QBLK
                                       + (int64_t) slot_in_block * N_QBLK + qb;
                const size_t base = (size_t) block_ib * BLOCK_BYTES;

                ggml_fp16_t act_norm_h;
                memcpy(&act_norm_h, &kc[base], sizeof(ggml_fp16_t));
                const double ne = std::fabs((double) ggml_fp16_to_fp32(act_norm_h)
                                          - (double) ggml_fp16_to_fp32(exp_norm_h));
                if (ne > max_norm_err) max_norm_err = ne;

                for (int j = 0; j < QK; ++j) {
                    const uint8_t byte = kc[base + QS_OFFSET + (j >> 1)];
                    const uint8_t act  = (byte >> ((j & 1) * 4)) & 0xF;
                    if (act != idxs[j]) nibble_mismatch++;
                }
            }
        }
    }

    const bool pass = (max_norm_err <= 1e-3) && (nibble_mismatch == 0);
    printf("%s: %s (max_norm_err=%.6f nibble_mismatch=%d)\n",
           label, pass ? "PASS" : "FAIL", max_norm_err, nibble_mismatch);
    free_graph(g);
    return pass;
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

// Run one Vulkan-vs-CUDA equivalence case on the op output node. Returns true
// on PASS. Used for the Task 5 decode (q_len==1) cases — CUDA internally routes
// q_len==1 to its own decode kernel, so this is a decode-vs-decode compare.
static bool compare_paged_case(const char * label, const paged_case & c,
                               ggml_backend_t vk, ggml_backend_t cuda, double tol) {
    ggml_context * tmp_ctx = nullptr;
    ggml_tensor  * op      = build_op_noalloc(c, &tmp_ctx);
    bool supported         = ggml_backend_supports_op(vk, op);
    ggml_free(tmp_ctx);
    if (!supported) {
        printf("%s: UNSUPPORTED on Vulkan FAIL\n", label);
        return false;
    }

    // Decode cases: pre-populate the cache so the split-K reduce runs over
    // genuine multi-term data (see build_case fill_cache rationale).
    built_graph gvk   = build_case(c, vk,   /*fill_cache=*/true);
    built_graph gcuda = build_case(c, cuda, /*fill_cache=*/true);

    cb_state st;
    std::vector<const ggml_tensor *> nodes = { gvk.out };
    bool ok = ggml_backend_compare_graph_backend(
        vk, cuda, gvk.gf, cmp_cb, &st, nodes.data(), nodes.size());

    bool pass = ok && st.max_err <= tol;
    printf("%s: max_err=%.6f tol=%.6f %s\n", label, st.max_err, tol, pass ? "PASS" : "FAIL");
    free_graph(gvk);
    free_graph(gcuda);
    return pass;
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

    bool all_ok = true;

    // ── Task 3 (F16): full paged plumbing with the trivial F16 cache type.
    // Same shape as the turbo4 case but F16 cache. Vulkan must support and
    // match the CUDA reference at F16 tolerance (no quant).
    {
        const paged_case cf { 128, 8, 2, 16, 32, 32, 1, GGML_TYPE_F16 };

        ggml_context * tmp_ctx = nullptr;
        ggml_tensor  * op      = build_op_noalloc(cf, &tmp_ctx);
        bool supported         = ggml_backend_supports_op(vk, op);
        ggml_free(tmp_ctx);

        if (!supported) {
            printf("paged f16 prefill: UNSUPPORTED on Vulkan (Task 3 incomplete) FAIL\n");
            all_ok = false;
        } else {
            // The comparator runs the SAME logical graph on CUDA via
            // ggml_backend_graph_copy (CUDA-resident input copies), so the
            // reference is a genuine CUDA computation, not Vulkan-vs-Vulkan.
            built_graph gvk   = build_case(cf, vk);
            built_graph gcuda = build_case(cf, cuda);   // hook (built-but-unused; copy handles residency)

            cb_state st;
            std::vector<const ggml_tensor *> nodes = { gvk.out };
            bool ok = ggml_backend_compare_graph_backend(
                vk, cuda, gvk.gf, cmp_cb, &st,
                nodes.data(), nodes.size());

            const double tol = 2e-3;   // F16, no quant
            bool pass = ok && st.max_err <= tol;
            printf("paged f16 prefill: max_err=%.6f tol=%.6f %s\n",
                   st.max_err, tol, pass ? "PASS" : "FAIL");
            all_ok = all_ok && pass;

            free_graph(gvk);
            free_graph(gcuda);
        }
    }

    // ── turbo4_0: 1 seq, head_dim=128, n_heads=8, n_kv_heads=2 (GQA 4:1),
    //    block_size=16, prefill q_len=32, context_len=32, TURBO4_0 cache.
    //    Still EXPECTED-FAIL on Vulkan until Task 4 adds the turbo4_0 path.
    {
        const paged_case c { 128, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_0 };

        ggml_context * tmp_ctx = nullptr;
        ggml_tensor  * op      = build_op_noalloc(c, &tmp_ctx);
        bool supported         = ggml_backend_supports_op(vk, op);
        ggml_free(tmp_ctx);

        if (!supported) {
            printf("EXPECTED-FAIL: PAGED_ATTN_MT turbo4_0 not yet supported on Vulkan\n");
        } else {
            // Deterministic 480-only scatter-readback oracle (no CUDA needed).
            all_ok = all_ok && scatter_turbo4_readback(c, vk);

            built_graph gvk   = build_case(c, vk);
            built_graph gcuda = build_case(c, cuda);

            cb_state st;
            std::vector<const ggml_tensor *> nodes = { gvk.out };
            bool ok = ggml_backend_compare_graph_backend(
                vk, cuda, gvk.gf, cmp_cb, &st,
                nodes.data(), nodes.size());

            const double tol = 5e-2;   // turbo4-class 4-bit centroid quant
            bool pass = ok && st.max_err <= tol;
            printf("paged turbo4_0 prefill: max_err=%.6f tol=%.6f %s\n",
                   st.max_err, tol, pass ? "PASS" : "FAIL");
            all_ok = all_ok && pass;

            free_graph(gvk);
            free_graph(gcuda);
        }
    }

    // ── Task 5: split-K DECODE (q_len==1) equivalence across context lengths
    //    that span chunk boundaries (CHUNK_KV=128):
    //      ctx=32,128  → single chunk
    //      ctx=200     → partial second chunk (72 valid tokens) — exercises the
    //                    reduce's bounded loop + log-sum-exp merge of 2 chunks
    //      ctx=512     → 4 full chunks — exercises a ≥4-way reduce
    //    Same tolerances as prefill (F16 2e-3, turbo4_0 5e-2). CUDA routes
    //    q_len==1 to its decode kernel, so this is decode-vs-decode.
    {
        const int ctxs[] = { 32, 128, 200, 512 };
        for (int ci = 0; ci < 4; ++ci) {
            const int ctx = ctxs[ci];
            char label[64];

            const paged_case cf { 128, 8, 2, 16, 1, ctx, 1, GGML_TYPE_F16 };
            snprintf(label, sizeof(label), "paged f16 decode ctx=%d", ctx);
            all_ok = compare_paged_case(label, cf, vk, cuda, 2e-3) && all_ok;

            const paged_case ct { 128, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_0 };
            snprintf(label, sizeof(label), "paged turbo4_0 decode ctx=%d", ctx);
            all_ok = compare_paged_case(label, ct, vk, cuda, 5e-2) && all_ok;
        }
    }

    // ── head_dim 256 (N_QBLK=2): turbo4_0 + F16, prefill + decode ───────────────
    {
        const paged_case p256t { 256, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_0 };
        all_ok = compare_paged_case("paged turbo4_0 hd256 prefill", p256t, vk, cuda, 5e-2) && all_ok;
        all_ok = scatter_turbo4_readback(p256t, vk) && all_ok;          // exercises N_QBLK=2 scatter
        const paged_case p256f { 256, 8, 2, 16, 32, 32, 1, GGML_TYPE_F16 };
        all_ok = compare_paged_case("paged f16 hd256 prefill", p256f, vk, cuda, 2e-3) && all_ok;
        for (int ctx : { 128, 512 }) {                                  // decode, multi-chunk reduce
            char lt[64], lf[64];
            snprintf(lt, sizeof lt, "paged turbo4_0 hd256 decode ctx=%d", ctx);
            snprintf(lf, sizeof lf, "paged f16 hd256 decode ctx=%d", ctx);
            const paged_case dt { 256, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_0 };
            const paged_case df { 256, 8, 2, 16, 1, ctx, 1, GGML_TYPE_F16 };
            all_ok = compare_paged_case(lt, dt, vk, cuda, 5e-2) && all_ok;
            all_ok = compare_paged_case(lf, df, vk, cuda, 2e-3) && all_ok;
        }
    }

    // ── head_dim 64 turbo4_64: prefill + decode (read path; cache host-prefilled) ──
    {
        const paged_case t64 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64 };
        all_ok = compare_paged_case("paged turbo4_64 hd64 prefill", t64, vk, cuda, 5e-2) && all_ok;
        for (int ctx : { 128, 512 }) {
            char l[64]; snprintf(l, sizeof l, "paged turbo4_64 hd64 decode ctx=%d", ctx);
            const paged_case d64 { 64, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_64 };
            all_ok = compare_paged_case(l, d64, vk, cuda, 5e-2) && all_ok;
        }
    }

    // ── turbo4_64 cooperative scatter quantizer (Task 4): device scatter vs
    //    host quantizer, bit-exact. Cache is NOT pre-filled, so the device
    //    scatter is what populates it — exercises real correctness.
    {
        const paged_case t64 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64 };
        all_ok = scatter_turbo4_readback(t64, vk) && all_ok;
    }

    ggml_backend_free(vk);
    ggml_backend_free(cuda);

    return all_ok ? 0 : 1;
}
