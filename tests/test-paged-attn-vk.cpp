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
#include <algorithm>

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
static void fill_turbo4_64_ol(ggml_tensor * t, uint32_t seed); // defined below (after host turbo4_64_ol quantizer)
static void fill_turbo4_64_ol8(ggml_tensor * t, uint32_t seed);  // outlier-matrix sweep (2026-07-01)
static void fill_turbo4_64_ol12(ggml_tensor * t, uint32_t seed); // outlier-matrix sweep (2026-07-01)
static void fill_q8_0(ggml_tensor * t, uint32_t seed);      // defined below (after host q8_0 quantizer)

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
    } else if (c.cache_type == GGML_TYPE_TURBO4_64_OL) {
        fill_turbo4_64_ol(k_cache, 11); fill_turbo4_64_ol(v_cache, 12);
    } else if (c.cache_type == GGML_TYPE_TURBO4_64_OL8) {
        fill_turbo4_64_ol8(k_cache, 11); fill_turbo4_64_ol8(v_cache, 12);
    } else if (c.cache_type == GGML_TYPE_TURBO4_64_OL12) {
        fill_turbo4_64_ol12(k_cache, 11); fill_turbo4_64_ol12(v_cache, 12);
    } else if (c.cache_type == GGML_TYPE_Q8_0) {
        fill_q8_0(k_cache, 11); fill_q8_0(v_cache, 12);
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

// ── turbo4_64_ol (SP2.5, 2026-07-01) host quantizer ──
//
// turbo4_64 with 4 fixed-position "massive activation" outlier channels
// {53,49,52,20} extracted verbatim at f16 and excluded from the group-norm
// and centroid quantization of the remaining 60 elements. Uses the SHARED
// HOST_TURBO_CENTROIDS_4BIT/HOST_TURBO_MID_4BIT table (turbo4_0's N=128
// table, above), NOT HOST_TURBO_CENTROIDS_4BIT_N64 — removing the outliers
// from the norm makes the remaining 60 "typical" values close enough to the
// N=128 assumption for that table to be appropriate again (see task brief).
// MUST stay byte-for-byte identical to TURBO4_64_OUTLIER_CHANNELS
// (ggml-common.h) / TURBO4_64_OL_CHANNELS (turbo-quant.cuh) /
// PA_TURBO4_64_OL_CHANNELS (paged_cache_ops.glsl).
static const int HOST_TURBO4_64_OL_CHANNELS[4] = { 53, 49, 52, 20 };

static bool host_turbo64_ol_is_outlier(int d, int * outlier_slot) {
    for (int o = 0; o < 4; ++o) {
        if (HOST_TURBO4_64_OL_CHANNELS[o] == d) { *outlier_slot = o; return true; }
    }
    return false;
}

// Quantize one 64-element f32 vector into a valid 40-byte block_turbo4_64_ol
// (norm fp16 + outliers[4] fp16 + 30 nibble-packed centroid indices for the
// 60 non-outlier elements). Mirrors host_turbo4_64_quantize_block, but
// excludes the 4 fixed outlier channels from the norm/quant and stores them
// verbatim instead.
static void host_turbo4_64_ol_quantize_block(const float * x /*64*/, uint8_t * out /*40 bytes*/) {
    bool is_outlier[64];
    int  outlier_slot[64];
    for (int j = 0; j < 64; ++j) {
        outlier_slot[j] = -1;
        is_outlier[j] = host_turbo64_ol_is_outlier(j, &outlier_slot[j]);
    }

    float red[64];
    for (int j = 0; j < 64; ++j) red[j] = is_outlier[j] ? 0.0f : (x[j] * x[j]);
    for (int s = 32; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) red[j] += red[j + s];
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    uint8_t idxs[64];
    float   rred[64];
    for (int j = 0; j < 64; ++j) {
        if (is_outlier[j]) { idxs[j] = 0; rred[j] = 0.0f; continue; }
        const float nv = x[j] * inv_norm;
        idxs[j] = host_turbo_nearest_4bit(nv);
        const float cv = HOST_TURBO_CENTROIDS_4BIT[idxs[j]];
        rred[j] = cv * cv;
    }
    for (int s = 32; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) rred[j] += rred[j + s];
    const float recon_norm     = sqrtf(rred[0]);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    memset(out, 0, 40);
    const ggml_fp16_t norm_h = ggml_fp32_to_fp16(corrected_norm);
    memcpy(out + 0, &norm_h, sizeof(ggml_fp16_t));
    for (int o = 0; o < 4; ++o) {
        const ggml_fp16_t ov = ggml_fp32_to_fp16(x[HOST_TURBO4_64_OL_CHANNELS[o]]);
        memcpy(out + 2 + o * (int)sizeof(ggml_fp16_t), &ov, sizeof(ggml_fp16_t));
    }
    int nib = 0;
    for (int j = 0; j < 64; ++j) {
        if (is_outlier[j]) continue;
        out[10 + (nib >> 1)] |= (uint8_t)(idxs[j] << ((nib & 1) * 4));
        nib++;
    }
}

// Pre-populate a turbo4_64_ol cache tensor with deterministic VALID blocks
// (seeded, reproducible across processes/backends), mirroring fill_turbo4_64.
static void fill_turbo4_64_ol(ggml_tensor * t, uint32_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const int64_t n_blk = (int64_t)(nbytes / 40);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> buf(nbytes, 0);
    for (int64_t b = 0; b < n_blk; ++b) {
        float x[64];
        for (int j = 0; j < 64; ++j) x[j] = dist(rng);
        host_turbo4_64_ol_quantize_block(x, &buf[(size_t)b * 40]);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

// ── turbo4_64_ol8 / turbo4_64_ol12 (outlier-matrix sweep, 2026-07-01) host
// quantizers ── mechanical mirror of host_turbo4_64_ol_quantize_block above,
// generalized over N via a small template so both variants share one body.
// MUST stay byte-for-byte identical to TURBO4_64_OL8_OUTLIER_CHANNELS /
// TURBO4_64_OL12_OUTLIER_CHANNELS (ggml-common.h) and the CUDA/Vulkan device
// channel lists.
static const int HOST_TURBO4_64_OL8_CHANNELS[8]   = { 53, 49, 52, 20, 21, 54, 14, 15 };
static const int HOST_TURBO4_64_OL12_CHANNELS[12] = { 53, 49, 52, 20, 21, 54, 14, 15, 51, 26, 24, 23 };

template <int N>
static bool host_turbo64_olN_is_outlier(const int (&channels)[N], int d, int * outlier_slot) {
    for (int o = 0; o < N; ++o) {
        if (channels[o] == d) { *outlier_slot = o; return true; }
    }
    return false;
}

// Quantize one 64-element f32 vector into a valid (2 + 2*N + (64-N)/2)-byte
// block_turbo4_64_olN (norm fp16 + outliers[N] fp16 + nibble-packed centroid
// indices for the 64-N non-outlier elements).
template <int N>
static void host_turbo4_64_olN_quantize_block(const int (&channels)[N], const float * x /*64*/, uint8_t * out) {
    bool is_outlier[64];
    int  outlier_slot[64];
    for (int j = 0; j < 64; ++j) {
        outlier_slot[j] = -1;
        is_outlier[j] = host_turbo64_olN_is_outlier<N>(channels, j, &outlier_slot[j]);
    }

    float red[64];
    for (int j = 0; j < 64; ++j) red[j] = is_outlier[j] ? 0.0f : (x[j] * x[j]);
    for (int s = 32; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) red[j] += red[j + s];
    const float grp_norm = sqrtf(red[0]);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    uint8_t idxs[64];
    float   rred[64];
    for (int j = 0; j < 64; ++j) {
        if (is_outlier[j]) { idxs[j] = 0; rred[j] = 0.0f; continue; }
        const float nv = x[j] * inv_norm;
        idxs[j] = host_turbo_nearest_4bit(nv);
        const float cv = HOST_TURBO_CENTROIDS_4BIT[idxs[j]];
        rred[j] = cv * cv;
    }
    for (int s = 32; s > 0; s >>= 1)
        for (int j = 0; j < s; ++j) rred[j] += rred[j + s];
    const float recon_norm     = sqrtf(rred[0]);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    const size_t block_bytes = 2 + 2 * (size_t)N + (64 - (size_t)N) / 2;
    memset(out, 0, block_bytes);
    const ggml_fp16_t norm_h = ggml_fp32_to_fp16(corrected_norm);
    memcpy(out + 0, &norm_h, sizeof(ggml_fp16_t));
    for (int o = 0; o < N; ++o) {
        const ggml_fp16_t ov = ggml_fp32_to_fp16(x[channels[o]]);
        memcpy(out + 2 + o * (int)sizeof(ggml_fp16_t), &ov, sizeof(ggml_fp16_t));
    }
    int nib = 0;
    const size_t qs_off = 2 + 2 * (size_t)N;
    for (int j = 0; j < 64; ++j) {
        if (is_outlier[j]) continue;
        out[qs_off + (nib >> 1)] |= (uint8_t)(idxs[j] << ((nib & 1) * 4));
        nib++;
    }
}

static void host_turbo4_64_ol8_quantize_block(const float * x /*64*/, uint8_t * out /*46 bytes*/) {
    host_turbo4_64_olN_quantize_block<8>(HOST_TURBO4_64_OL8_CHANNELS, x, out);
}
static void host_turbo4_64_ol12_quantize_block(const float * x /*64*/, uint8_t * out /*52 bytes*/) {
    host_turbo4_64_olN_quantize_block<12>(HOST_TURBO4_64_OL12_CHANNELS, x, out);
}

static void fill_turbo4_64_ol8(ggml_tensor * t, uint32_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const int64_t n_blk = (int64_t)(nbytes / 46);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> buf(nbytes, 0);
    for (int64_t b = 0; b < n_blk; ++b) {
        float x[64];
        for (int j = 0; j < 64; ++j) x[j] = dist(rng);
        host_turbo4_64_ol8_quantize_block(x, &buf[(size_t)b * 46]);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

static void fill_turbo4_64_ol12(ggml_tensor * t, uint32_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const int64_t n_blk = (int64_t)(nbytes / 52);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> buf(nbytes, 0);
    for (int64_t b = 0; b < n_blk; ++b) {
        float x[64];
        for (int j = 0; j < 64; ++j) x[j] = dist(rng);
        host_turbo4_64_ol12_quantize_block(x, &buf[(size_t)b * 52]);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, nbytes);
}

// Quantize one 32-element f32 vector into a valid 34-byte block_q8_0 (fp16
// scale + int8 qs[32]) — standard symmetric per-block quant, much simpler
// than the turbo4 no-RHT scheme: amax = max(|x|); scale = amax/127;
// qs[j] = round(x[j]/scale). Mirrors host_turbo4_64_quantize_block's role
// (pre-populate the decode KV cache with deterministic, VALID blocks).
static void host_q8_0_quantize_block(const float * x /*32*/, uint8_t * out /*34 bytes*/) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
    const float scale     = amax / 127.0f;
    const float inv_scale = (scale > 1e-10f) ? (1.0f / scale) : 0.0f;
    const ggml_fp16_t d_h = ggml_fp32_to_fp16(scale);
    memset(out, 0, 34);
    memcpy(out + 0, &d_h, sizeof(ggml_fp16_t));
    for (int j = 0; j < 32; ++j) {
        // Round-half-to-even (matches CUDA's __float2int_rn and GLSL's round()
        // on the tested drivers), NOT lroundf's round-half-away-from-zero —
        // using lroundf here caused spurious ties-only mismatches against both
        // real backends despite them agreeing with each other.
        out[2 + j] = (uint8_t) (int8_t) std::nearbyintf(x[j] * inv_scale);
    }
}

// Pre-populate a q8_0 cache tensor with deterministic VALID blocks (seeded,
// reproducible across processes/backends), mirroring fill_turbo4_64.
static void fill_q8_0(ggml_tensor * t, uint32_t seed) {
    const size_t nbytes = ggml_nbytes(t);
    const int64_t n_blk = (int64_t)(nbytes / 34);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint8_t> buf(nbytes, 0);
    for (int64_t b = 0; b < n_blk; ++b) {
        float x[32];
        for (int j = 0; j < 32; ++j) x[j] = dist(rng);
        host_q8_0_quantize_block(x, &buf[(size_t)b * 34]);
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

// Deterministic, CUDA-free scatter-readback oracle for turbo4_64_ol (SP2.5,
// 2026-07-01), mirroring scatter_turbo4_readback. Runs the op on Vulkan,
// reads back k_cache, and checks each 40-byte block against
// host_turbo4_64_ol_quantize_block applied to the same k_cur at the
// expected element_block_index: outlier values within a small fp16-rounding
// tolerance (verbatim storage, no centroid quant slop) and the 60 packed
// nibbles exactly equal. Asserts per-block norm within 1e-3.
static bool scatter_turbo4_64_ol_readback(const paged_case & c, ggml_backend_t vk) {
    const char * label = "scatter turbo4_64_ol readback";

    built_graph g = build_case(c, vk);
    if (ggml_backend_graph_compute(vk, g.gf) != GGML_STATUS_SUCCESS) {
        printf("%s: FAIL (graph compute error)\n", label);
        free_graph(g);
        return false;
    }

    const int HD           = c.head_dim;
    const int total_tokens = c.q_len * c.n_seq;
    const int BS           = c.block_size;
    const int QK           = 64;
    const int N_QBLK       = HD / QK;
    const int n_kv_heads   = c.n_kv_heads;
    const size_t BLOCK_BYTES = 40;  // fp16 norm + outliers[4] fp16 + qs[30]

    std::vector<ggml_fp16_t> kcur(ggml_nelements(g.k_cur));
    ggml_backend_tensor_get(g.k_cur, kcur.data(), 0, ggml_nbytes(g.k_cur));
    std::vector<uint8_t> kc(ggml_nbytes(g.k_cache));
    ggml_backend_tensor_get(g.k_cache, kc.data(), 0, ggml_nbytes(g.k_cache));

    double max_norm_err    = 0.0;
    double max_outlier_err = 0.0;
    int    nibble_mismatch = 0;

    for (int t = 0; t < total_tokens; ++t) {
        const int slot          = t;            // slot_mapping[i] = i
        const int paged_block   = slot / BS;
        const int slot_in_block = slot % BS;
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int qb = 0; qb < N_QBLK; ++qb) {
                float x[64];
                for (int j = 0; j < QK; ++j) {
                    const int d = qb * QK + j;
                    const size_t src = (size_t) t * n_kv_heads * HD + (size_t) h * HD + d;
                    x[j] = ggml_fp16_to_fp32(kcur[src]);
                }
                uint8_t exp_block[40];
                host_turbo4_64_ol_quantize_block(x, exp_block);

                const int64_t block_ib = ((int64_t) paged_block * n_kv_heads + h) * BS * N_QBLK
                                       + (int64_t) slot_in_block * N_QBLK + qb;
                const size_t base = (size_t) block_ib * BLOCK_BYTES;

                ggml_fp16_t act_norm_h, exp_norm_h;
                memcpy(&act_norm_h, &kc[base], sizeof(ggml_fp16_t));
                memcpy(&exp_norm_h, exp_block, sizeof(ggml_fp16_t));
                const double ne = std::fabs((double) ggml_fp16_to_fp32(act_norm_h)
                                          - (double) ggml_fp16_to_fp32(exp_norm_h));
                if (ne > max_norm_err) max_norm_err = ne;

                for (int o = 0; o < 4; ++o) {
                    ggml_fp16_t act_ol_h, exp_ol_h;
                    memcpy(&act_ol_h, &kc[base + 2 + o * sizeof(ggml_fp16_t)], sizeof(ggml_fp16_t));
                    memcpy(&exp_ol_h, exp_block + 2 + o * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                    const double oe = std::fabs((double) ggml_fp16_to_fp32(act_ol_h)
                                              - (double) ggml_fp16_to_fp32(exp_ol_h));
                    if (oe > max_outlier_err) max_outlier_err = oe;
                }

                int nib = 0;
                for (int j = 0; j < QK; ++j) {
                    int outlier_slot;
                    if (host_turbo64_ol_is_outlier(j, &outlier_slot)) continue;
                    const uint8_t byte = kc[base + 10 + (nib >> 1)];
                    const uint8_t act  = (byte >> ((nib & 1) * 4)) & 0xF;
                    const uint8_t exp_byte = exp_block[10 + (nib >> 1)];
                    const uint8_t exp  = (exp_byte >> ((nib & 1) * 4)) & 0xF;
                    if (act != exp) nibble_mismatch++;
                    nib++;
                }
            }
        }
    }

    const bool pass = (max_norm_err <= 1e-3) && (max_outlier_err <= 1e-3) && (nibble_mismatch == 0);
    printf("%s: %s (max_norm_err=%.6f max_outlier_err=%.6f nibble_mismatch=%d)\n",
           label, pass ? "PASS" : "FAIL", max_norm_err, max_outlier_err, nibble_mismatch);
    free_graph(g);
    return pass;
}

// Deterministic, CUDA-free scatter-readback oracle for turbo4_64_ol8 /
// turbo4_64_ol12 (outlier-matrix sweep, 2026-07-01), mechanical mirror of
// scatter_turbo4_64_ol_readback, generalized over N.
template <int N>
static bool scatter_turbo4_64_olN_readback(const char * label, const int (&channels)[N],
                                            void (*quantize_block)(const float *, uint8_t *),
                                            const paged_case & c, ggml_backend_t vk) {
    built_graph g = build_case(c, vk);
    if (ggml_backend_graph_compute(vk, g.gf) != GGML_STATUS_SUCCESS) {
        printf("%s: FAIL (graph compute error)\n", label);
        free_graph(g);
        return false;
    }

    const int HD           = c.head_dim;
    const int total_tokens = c.q_len * c.n_seq;
    const int BS           = c.block_size;
    const int QK           = 64;
    const int N_QBLK       = HD / QK;
    const int n_kv_heads   = c.n_kv_heads;
    const size_t BLOCK_BYTES = 2 + 2 * (size_t)N + (64 - (size_t)N) / 2;
    const size_t qs_off      = 2 + 2 * (size_t)N;

    std::vector<ggml_fp16_t> kcur(ggml_nelements(g.k_cur));
    ggml_backend_tensor_get(g.k_cur, kcur.data(), 0, ggml_nbytes(g.k_cur));
    std::vector<uint8_t> kc(ggml_nbytes(g.k_cache));
    ggml_backend_tensor_get(g.k_cache, kc.data(), 0, ggml_nbytes(g.k_cache));

    double max_norm_err    = 0.0;
    double max_outlier_err = 0.0;
    int    nibble_mismatch = 0;

    std::vector<uint8_t> exp_block(BLOCK_BYTES);

    for (int t = 0; t < total_tokens; ++t) {
        const int slot          = t;
        const int paged_block   = slot / BS;
        const int slot_in_block = slot % BS;
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int qb = 0; qb < N_QBLK; ++qb) {
                float x[64];
                for (int j = 0; j < QK; ++j) {
                    const int d = qb * QK + j;
                    const size_t src = (size_t) t * n_kv_heads * HD + (size_t) h * HD + d;
                    x[j] = ggml_fp16_to_fp32(kcur[src]);
                }
                quantize_block(x, exp_block.data());

                const int64_t block_ib = ((int64_t) paged_block * n_kv_heads + h) * BS * N_QBLK
                                       + (int64_t) slot_in_block * N_QBLK + qb;
                const size_t base = (size_t) block_ib * BLOCK_BYTES;

                ggml_fp16_t act_norm_h, exp_norm_h;
                memcpy(&act_norm_h, &kc[base], sizeof(ggml_fp16_t));
                memcpy(&exp_norm_h, exp_block.data(), sizeof(ggml_fp16_t));
                const double ne = std::fabs((double) ggml_fp16_to_fp32(act_norm_h)
                                          - (double) ggml_fp16_to_fp32(exp_norm_h));
                if (ne > max_norm_err) max_norm_err = ne;

                for (int o = 0; o < N; ++o) {
                    ggml_fp16_t act_ol_h, exp_ol_h;
                    memcpy(&act_ol_h, &kc[base + 2 + o * sizeof(ggml_fp16_t)], sizeof(ggml_fp16_t));
                    memcpy(&exp_ol_h, exp_block.data() + 2 + o * sizeof(ggml_fp16_t), sizeof(ggml_fp16_t));
                    const double oe = std::fabs((double) ggml_fp16_to_fp32(act_ol_h)
                                              - (double) ggml_fp16_to_fp32(exp_ol_h));
                    if (oe > max_outlier_err) max_outlier_err = oe;
                }

                int nib = 0;
                for (int j = 0; j < QK; ++j) {
                    int outlier_slot;
                    if (host_turbo64_olN_is_outlier<N>(channels, j, &outlier_slot)) continue;
                    const uint8_t byte = kc[base + qs_off + (nib >> 1)];
                    const uint8_t act  = (byte >> ((nib & 1) * 4)) & 0xF;
                    const uint8_t exp_byte = exp_block[qs_off + (nib >> 1)];
                    const uint8_t exp  = (exp_byte >> ((nib & 1) * 4)) & 0xF;
                    if (act != exp) nibble_mismatch++;
                    nib++;
                }
            }
        }
    }

    const bool pass = (max_norm_err <= 1e-3) && (max_outlier_err <= 1e-3) && (nibble_mismatch == 0);
    printf("%s: %s (max_norm_err=%.6f max_outlier_err=%.6f nibble_mismatch=%d)\n",
           label, pass ? "PASS" : "FAIL", max_norm_err, max_outlier_err, nibble_mismatch);
    free_graph(g);
    return pass;
}

// Deterministic, CUDA-free scatter-readback oracle for Q8_0 (mirrors
// scatter_turbo4_readback). Runs the op on Vulkan, reads back k_cache, and
// checks each q8_0 block against host_q8_0_quantize_block applied to the
// same k_cur at the expected element_block_index. Asserts per-block scale
// within a small fp16-rounding tolerance and the int8 qs BYTE-EXACT (no
// centroid quant slop — q8_0 is a plain round-to-nearest affine quant).
static bool scatter_q8_0_readback(const paged_case & c, ggml_backend_t vk) {
    const char * label = "scatter q8_0 readback";

    built_graph g = build_case(c, vk);
    if (ggml_backend_graph_compute(vk, g.gf) != GGML_STATUS_SUCCESS) {
        printf("%s: FAIL (graph compute error)\n", label);
        free_graph(g);
        return false;
    }

    const int HD           = c.head_dim;
    const int total_tokens = c.q_len * c.n_seq;
    const int BS           = c.block_size;
    const int QK           = 32;
    const int N_QBLK       = HD / QK;
    const int n_kv_heads   = c.n_kv_heads;
    const size_t BLOCK_BYTES = 34;   // fp16 d + int8 qs[32]
    const size_t QS_OFFSET   = 2;

    std::vector<ggml_fp16_t> kcur(ggml_nelements(g.k_cur));
    ggml_backend_tensor_get(g.k_cur, kcur.data(), 0, ggml_nbytes(g.k_cur));
    std::vector<uint8_t> kc(ggml_nbytes(g.k_cache));
    ggml_backend_tensor_get(g.k_cache, kc.data(), 0, ggml_nbytes(g.k_cache));

    double max_scale_err  = 0.0;
    int    qs_mismatch    = 0;
    int    qs_total       = 0;
    bool   qs_off_by_more = false;   // any mismatch with |act-exp| > 1 is a real bug, not a tie

    for (int t = 0; t < total_tokens; ++t) {
        const int slot          = t;            // slot_mapping[i] = i
        const int paged_block   = slot / BS;
        const int slot_in_block = slot % BS;
        for (int h = 0; h < n_kv_heads; ++h) {
            for (int qb = 0; qb < N_QBLK; ++qb) {
                float x[32];
                for (int j = 0; j < QK; ++j) {
                    const int d = qb * QK + j;
                    const size_t src = (size_t) t * n_kv_heads * HD + (size_t) h * HD + d;
                    x[j] = ggml_fp16_to_fp32(kcur[src]);
                }
                uint8_t exp_block[34];
                host_q8_0_quantize_block(x, exp_block);
                ggml_fp16_t exp_d_h;
                memcpy(&exp_d_h, exp_block, sizeof(ggml_fp16_t));

                const int64_t block_ib = ((int64_t) paged_block * n_kv_heads + h) * BS * N_QBLK
                                       + (int64_t) slot_in_block * N_QBLK + qb;
                const size_t base = (size_t) block_ib * BLOCK_BYTES;

                ggml_fp16_t act_d_h;
                memcpy(&act_d_h, &kc[base], sizeof(ggml_fp16_t));
                const double se = std::fabs((double) ggml_fp16_to_fp32(act_d_h)
                                           - (double) ggml_fp16_to_fp32(exp_d_h));
                if (se > max_scale_err) max_scale_err = se;

                for (int j = 0; j < QK; ++j) {
                    const int8_t act = (int8_t) kc[base + QS_OFFSET + j];
                    const int8_t exp = (int8_t) exp_block[QS_OFFSET + j];
                    qs_total++;
                    if (act != exp) {
                        qs_mismatch++;
                        if (std::abs((int) act - (int) exp) > 1) qs_off_by_more = true;
                    }
                }
            }
        }
    }

    // Q8_0's round-to-nearest-int8 decision has a genuine exact-tie case (the
    // true mathematical value lands within a few ULP of x.5) that GPU vs CPU
    // can round to either neighboring integer — diagnosed directly: observed
    // mismatches have |x[j]*inv_scale - round(...)| within ~4e-6 of 0.5,
    // i.e. real ties, not an implementation bug (unlike turbo4's centroid
    // lookup, Q8_0 has no norm-correction feedback loop that could amplify
    // this). Require every mismatch be off-by-exactly-1 (never more, which
    // would indicate a real quantization bug) and bounded to a small
    // fraction of all tested elements (observed: <=6 out of thousands,
    // 4/32768 = 0.012% on the worst case here).
    const bool tie_rate_ok = qs_total > 0 && ((double) qs_mismatch / qs_total) <= 0.01;
    const bool pass = (max_scale_err <= 1e-3) && !qs_off_by_more && tie_rate_ok;
    printf("%s: %s (max_scale_err=%.6f qs_mismatch=%d/%d)\n",
           label, pass ? "PASS" : "FAIL", max_scale_err, qs_mismatch, qs_total);
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
            all_ok = compare_paged_case(l, d64, vk, cuda, 1e-2) && all_ok;
        }
    }

    // ── turbo4_64 cooperative scatter quantizer (Task 4): device scatter vs
    //    host quantizer, bit-exact. Cache is NOT pre-filled, so the device
    //    scatter is what populates it — exercises real correctness.
    {
        const paged_case t64 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64 };
        all_ok = scatter_turbo4_readback(t64, vk) && all_ok;
    }

    // ── turbo4_64_ol (SP2.5, 2026-07-01): turbo4_64 with 4 fixed-position
    //    outlier channels {53,49,52,20} extracted verbatim and excluded from
    //    the group-norm/centroid quant of the remaining 60 elements. Same
    //    5e-2 tolerance as turbo4_64 (still 4-bit-dominant). head_dim==64 only.
    {
        const paged_case t64ol { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64_OL };
        all_ok = compare_paged_case("paged turbo4_64_ol hd64 prefill", t64ol, vk, cuda, 5e-2) && all_ok;
        for (int ctx : { 128, 512 }) {
            char l[64]; snprintf(l, sizeof l, "paged turbo4_64_ol hd64 decode ctx=%d", ctx);
            const paged_case d64ol { 64, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_64_OL };
            all_ok = compare_paged_case(l, d64ol, vk, cuda, 5e-2) && all_ok;
        }
        // Cooperative scatter quantizer: device scatter vs host quantizer,
        // bit-exact (nibbles) / near-exact (fp16 outliers + norm). Cache is
        // NOT pre-filled, so the device scatter is what populates it.
        all_ok = scatter_turbo4_64_ol_readback(t64ol, vk) && all_ok;
    }

    // ── turbo4_64_ol8 / turbo4_64_ol12 (outlier-matrix sweep, 2026-07-01):
    //    turbo4_64_ol with 8 / 12 fixed outlier channels instead of 4. Same
    //    5e-2 tolerance, head_dim==64 only. Mirrors the turbo4_64_ol block above.
    {
        const paged_case t64ol8 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64_OL8 };
        all_ok = compare_paged_case("paged turbo4_64_ol8 hd64 prefill", t64ol8, vk, cuda, 5e-2) && all_ok;
        for (int ctx : { 128, 512 }) {
            char l[64]; snprintf(l, sizeof l, "paged turbo4_64_ol8 hd64 decode ctx=%d", ctx);
            const paged_case d64ol8 { 64, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_64_OL8 };
            all_ok = compare_paged_case(l, d64ol8, vk, cuda, 5e-2) && all_ok;
        }
        all_ok = scatter_turbo4_64_olN_readback<8>("scatter turbo4_64_ol8 readback",
                     HOST_TURBO4_64_OL8_CHANNELS, host_turbo4_64_ol8_quantize_block, t64ol8, vk) && all_ok;
    }
    {
        const paged_case t64ol12 { 64, 8, 2, 16, 32, 32, 1, GGML_TYPE_TURBO4_64_OL12 };
        all_ok = compare_paged_case("paged turbo4_64_ol12 hd64 prefill", t64ol12, vk, cuda, 5e-2) && all_ok;
        for (int ctx : { 128, 512 }) {
            char l[64]; snprintf(l, sizeof l, "paged turbo4_64_ol12 hd64 decode ctx=%d", ctx);
            const paged_case d64ol12 { 64, 8, 2, 16, 1, ctx, 1, GGML_TYPE_TURBO4_64_OL12 };
            all_ok = compare_paged_case(l, d64ol12, vk, cuda, 5e-2) && all_ok;
        }
        all_ok = scatter_turbo4_64_olN_readback<12>("scatter turbo4_64_ol12 readback",
                     HOST_TURBO4_64_OL12_CHANNELS, host_turbo4_64_ol12_quantize_block, t64ol12, vk) && all_ok;
    }

    // ── Q8_0 (this task): standard 8-bit symmetric per-32-element-block quant.
    //    Much higher precision than turbo4's 4-bit centroid quant, so it gets
    //    an F16-class tolerance (2e-3) rather than turbo4's 5e-2. Covers all
    //    three head_dim brackets this branch supports: 64, 128, 256.
    {
        const int head_dims[] = { 64, 128, 256 };
        for (int hd : head_dims) {
            char label[80];

            const paged_case pre { hd, 8, 2, 16, 32, 32, 1, GGML_TYPE_Q8_0 };
            snprintf(label, sizeof label, "paged q8_0 hd%d prefill", hd);
            all_ok = compare_paged_case(label, pre, vk, cuda, 2e-3) && all_ok;

            snprintf(label, sizeof label, "scatter q8_0 hd%d readback", hd);
            all_ok = scatter_q8_0_readback(pre, vk) && all_ok;

            for (int ctx : { 128, 512 }) {
                const paged_case dec { hd, 8, 2, 16, 1, ctx, 1, GGML_TYPE_Q8_0 };
                snprintf(label, sizeof label, "paged q8_0 hd%d decode ctx=%d", hd, ctx);
                all_ok = compare_paged_case(label, dec, vk, cuda, 2e-3) && all_ok;
            }
        }
    }

    ggml_backend_free(vk);
    ggml_backend_free(cuda);

    return all_ok ? 0 : 1;
}
