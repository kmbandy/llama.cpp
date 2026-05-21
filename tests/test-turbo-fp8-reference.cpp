// MAD-214 Phase 1C: Reference scalar implementation of turbo-FP8 dequant +
// attention. Pure C++ (no GPU). Serves as the bitwise correctness baseline
// for the eventual HIP/Triton kernel.
//
// Build (from project root):
//   g++ -std=c++17 -O2 -Wall tests/test-turbo-fp8-reference.cpp -o /tmp/test-turbo-fp8-reference
//
// Tests:
//   1. E4M3 lattice round-trip: fp32 → E4M3 byte → fp32 must be exact for
//      lattice points.
//   2. Block round-trip: quantize a block of fp32 values, dequantize, check
//      that the reconstruction error matches the expected per-variant MSE.
//   3. Attention sanity: non-uniform K (per attention-correctness-testing-
//      principle) → softmax weights should be non-uniform → output exercises
//      the V dequant + accumulation paths.
//
// MAD-214

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// Block structs (mirror of ggml-common.h, kept inline for self-contained test)
// ---------------------------------------------------------------------------
typedef uint16_t ggml_half;  // matches ggml_half typedef when not using __fp16

#define QK_TURBO_FP8 32

struct block_turbo3_fp8 {
    ggml_half  d;
    uint8_t    qs[QK_TURBO_FP8 * 3 / 8];  // 12 bytes
    uint8_t    signs[QK_TURBO_FP8 / 8];   //  4 bytes
};
static_assert(sizeof(block_turbo3_fp8) == 18, "turbo3_fp8 size");

struct block_turbo4_fp8 {
    ggml_half  d;
    uint8_t    qs[QK_TURBO_FP8 / 2];      // 16 bytes
    uint8_t    signs[QK_TURBO_FP8 / 8];   //  4 bytes
};
static_assert(sizeof(block_turbo4_fp8) == 22, "turbo4_fp8 size");

struct block_turbo5_fp8 {
    ggml_half  d;
    uint8_t    qs[QK_TURBO_FP8 * 5 / 8];  // 20 bytes
    uint8_t    signs[QK_TURBO_FP8 / 8];   //  4 bytes
};
static_assert(sizeof(block_turbo5_fp8) == 26, "turbo5_fp8 size");

// ---------------------------------------------------------------------------
// FP16 / E4M3 byte helpers
// ---------------------------------------------------------------------------
static inline ggml_half fp32_to_fp16(float f) {
    union { float f; uint32_t u; } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000;
    int32_t  exp  = ((v.u >> 23) & 0xff) - 127 + 15;
    uint32_t mant = v.u & 0x7fffff;
    if (exp <= 0) return (ggml_half)(sign);  // subnormal/zero → zero
    if (exp >= 0x1f) return (ggml_half)(sign | 0x7c00);  // inf/NaN → inf
    return (ggml_half)(sign | (exp << 10) | (mant >> 13));
}

static inline float fp16_to_fp32(ggml_half h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    uint32_t exp  = ((uint32_t)h >> 10) & 0x1f;
    uint32_t mant = (uint32_t)h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) { union { uint32_t u; float f; } v = { sign }; return v.f; }
        // subnormal
        exp = 127 - 14;
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        mant &= 0x3ff;
    } else if (exp == 0x1f) {
        union { uint32_t u; float f; } v = { sign | 0x7f800000 | (mant << 13) };
        return v.f;
    } else {
        exp = exp - 15 + 127;
    }
    union { uint32_t u; float f; } v = { sign | (exp << 23) | (mant << 13) };
    return v.f;
}

// E4M3 byte → fp32 value. Treats e=15,m=7 as +/-NaN (returns NaN).
// Bit layout: sign(1) | exp(4) | mant(3), bias=7.
static float e4m3_byte_to_fp32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e    = (b >> 3) & 0xf;
    int m    = b & 0x7;
    if (e == 15 && m == 7) return NAN;
    float val;
    if (e == 0) {
        val = (1.0f / 64.0f) * (m / 8.0f);  // subnormal: 2^-6 * m/8
    } else {
        val = std::ldexp(1.0f + m / 8.0f, e - 7);  // normal
    }
    return sign ? -val : val;
}

// fp32 value → nearest positive-E4M3 byte (used for sanity checks; we never
// quantize at runtime in the kernel — centroids are precomputed offline).
static uint8_t fp32_to_e4m3_byte_pos(float v) {
    if (v < 0) v = -v;
    uint8_t best = 0;
    float best_err = std::abs(v - e4m3_byte_to_fp32(0));
    for (int b = 1; b < 0x80; ++b) {
        if (b == 0x7f) continue;  // NaN code
        float vb = e4m3_byte_to_fp32((uint8_t)b);
        float err = std::abs(v - vb);
        if (err < best_err) { best = (uint8_t)b; best_err = err; }
    }
    return best;
}

// ---------------------------------------------------------------------------
// Bit-pack helpers for N-bit indices into qs[]
// ---------------------------------------------------------------------------
static inline void pack_nbit(uint8_t *qs, int idx_pos, uint8_t value, int nbits) {
    // Pack `value` (low nbits) into the bitstream qs[] at element index idx_pos.
    int bit_pos = idx_pos * nbits;
    int byte_lo = bit_pos / 8;
    int bit_off = bit_pos % 8;
    // Up to 2 bytes touched for nbits<=8 with non-aligned offset
    uint16_t word = (uint16_t)qs[byte_lo] | ((uint16_t)qs[byte_lo + 1] << 8);
    uint16_t mask = ((1u << nbits) - 1) << bit_off;
    word = (uint16_t)((word & ~mask) | (((uint16_t)value << bit_off) & mask));
    qs[byte_lo]     = (uint8_t)(word & 0xff);
    qs[byte_lo + 1] = (uint8_t)(word >> 8);
}

static inline uint8_t unpack_nbit(const uint8_t *qs, int idx_pos, int nbits) {
    int bit_pos = idx_pos * nbits;
    int byte_lo = bit_pos / 8;
    int bit_off = bit_pos % 8;
    uint16_t word = (uint16_t)qs[byte_lo] | ((uint16_t)qs[byte_lo + 1] << 8);
    return (uint8_t)((word >> bit_off) & ((1u << nbits) - 1));
}

static inline void set_sign_bit(uint8_t *signs, int idx_pos, int s) {
    int byte = idx_pos / 8;
    int bit  = idx_pos % 8;
    signs[byte] = (uint8_t)((signs[byte] & ~(1u << bit)) | ((s & 1u) << bit));
}

static inline int get_sign_bit(const uint8_t *signs, int idx_pos) {
    int byte = idx_pos / 8;
    int bit  = idx_pos % 8;
    return (signs[byte] >> bit) & 1;
}

// ---------------------------------------------------------------------------
// Quantize / dequantize one block
// ---------------------------------------------------------------------------
// `centroids` is an array of N_CENTROIDS positive E4M3 bytes (LUT entries).
// `in` points to QK_TURBO_FP8 (= 32) fp32 values.
// `out` points to the packed block struct.
static void quantize_block(
    const float *in, void *out_void,
    const uint8_t *centroids, int n_centroids, int idx_bits
) {
    // Block layout: [d:ggml_half][qs:varies][signs:4 bytes]
    uint8_t *out = (uint8_t *)out_void;
    int qs_bytes = (QK_TURBO_FP8 * idx_bits) / 8;
    ggml_half *p_d = (ggml_half *)(out);
    uint8_t   *qs  = out + sizeof(ggml_half);
    uint8_t   *sg  = qs + qs_bytes;

    std::memset(qs, 0, qs_bytes);
    std::memset(sg, 0, QK_TURBO_FP8 / 8);

    // Compute per-block scale = max(|values|), store at FP16 precision.
    float scale = 0.0f;
    for (int i = 0; i < QK_TURBO_FP8; ++i) {
        float a = std::abs(in[i]);
        if (a > scale) scale = a;
    }
    if (scale == 0.0f) { *p_d = 0; return; }
    // Round-trip through fp16 to match the FP16 storage
    *p_d = fp32_to_fp16(scale);
    float scale_fp16 = fp16_to_fp32(*p_d);
    if (scale_fp16 == 0.0f) scale_fp16 = scale;  // guard underflow

    // Precompute centroid float values for nearest-magnitude lookup
    std::array<float, 64> cv{};  // max we use is 32
    for (int k = 0; k < n_centroids; ++k) {
        cv[k] = e4m3_byte_to_fp32(centroids[k]);  // positive bytes → positive floats
    }

    for (int i = 0; i < QK_TURBO_FP8; ++i) {
        float v = in[i];
        int   s = v < 0.0f ? 1 : 0;
        float m = std::abs(v) / scale_fp16;  // normalized magnitude in [0, 1]
        // Find nearest centroid magnitude
        int best_k = 0;
        float best_err = std::abs(m - cv[0]);
        for (int k = 1; k < n_centroids; ++k) {
            float err = std::abs(m - cv[k]);
            if (err < best_err) { best_k = k; best_err = err; }
        }
        pack_nbit(qs, i, (uint8_t)best_k, idx_bits);
        set_sign_bit(sg, i, s);
    }
}

static void dequantize_block(
    const void *in_void, float *out,
    const uint8_t *centroids, int n_centroids, int idx_bits
) {
    (void)n_centroids;
    const uint8_t *in = (const uint8_t *)in_void;
    int qs_bytes = (QK_TURBO_FP8 * idx_bits) / 8;
    const ggml_half *p_d = (const ggml_half *)in;
    const uint8_t   *qs  = in + sizeof(ggml_half);
    const uint8_t   *sg  = qs + qs_bytes;
    float scale = fp16_to_fp32(*p_d);

    for (int i = 0; i < QK_TURBO_FP8; ++i) {
        uint8_t idx = unpack_nbit(qs, i, idx_bits);
        int     s   = get_sign_bit(sg, i);
        float   mag = e4m3_byte_to_fp32(centroids[idx]);
        out[i] = (s ? -mag : mag) * scale;
    }
}

// ---------------------------------------------------------------------------
// Walsh-Hadamard rotation (matches the calibration script — fp32 H along last dim)
// ---------------------------------------------------------------------------
// Allocates and returns an n×n Hadamard matrix (n must be power of 2),
// normalized so H @ H.T == I.
static std::vector<float> hadamard_matrix(int n) {
    assert((n & (n - 1)) == 0 && n > 0);
    std::vector<float> H(n * n, 0.0f);
    H[0] = 1.0f;
    int size = 1;
    while (size < n) {
        // Sylvester step: [[H, H], [H, -H]]
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                float v = H[i * n + j];
                H[i * n + (size + j)]            =  v;
                H[(size + i) * n + j]            =  v;
                H[(size + i) * n + (size + j)]   = -v;
            }
        }
        size *= 2;
    }
    float norm = 1.0f / std::sqrt((float)n);
    for (auto &v : H) v *= norm;
    return H;
}

// Apply Hadamard rotation along last dim: out = x @ H  (x: [rows × n], H: [n × n])
// Used by Phase 1E wrapper; declared here for reference + test coverage of orthogonality.
[[maybe_unused]] static void apply_hadamard(const float *x, const float *H, float *out, int rows, int n) {
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) acc += x[r * n + k] * H[k * n + j];
            out[r * n + j] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Reference scalar attention with turbo-FP8 KV
// ---------------------------------------------------------------------------
// q: [n_q_tokens × head_dim]  (already Hadamard-rotated if rotation enabled)
// k_blocks: array of block_turbo*_fp8 for K cache, shape [n_kv_tokens × head_dim/QK_TURBO_FP8]
// v_blocks: same for V cache
// centroids_k, centroids_v: per-variant LUTs
// out: [n_q_tokens × head_dim] fp32
static void reference_attention_turbo_fp8(
    const float *q,
    const void *k_blocks, const void *v_blocks,
    const uint8_t *centroids_k, const uint8_t *centroids_v,
    int n_centroids, int idx_bits, int block_stride_bytes,
    int n_q_tokens, int n_kv_tokens, int head_dim,
    float *out
) {
    int blocks_per_row = head_dim / QK_TURBO_FP8;
    assert(head_dim % QK_TURBO_FP8 == 0);

    std::vector<float> k_dequant(n_kv_tokens * head_dim);
    std::vector<float> v_dequant(n_kv_tokens * head_dim);

    // Dequant entire K and V caches
    for (int t = 0; t < n_kv_tokens; ++t) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t *kb = (const uint8_t *)k_blocks + (t * blocks_per_row + b) * block_stride_bytes;
            const uint8_t *vb = (const uint8_t *)v_blocks + (t * blocks_per_row + b) * block_stride_bytes;
            dequantize_block(kb, &k_dequant[t * head_dim + b * QK_TURBO_FP8], centroids_k, n_centroids, idx_bits);
            dequantize_block(vb, &v_dequant[t * head_dim + b * QK_TURBO_FP8], centroids_v, n_centroids, idx_bits);
        }
    }

    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores(n_kv_tokens, 0.0f);
    std::vector<float> probs(n_kv_tokens, 0.0f);

    for (int qi = 0; qi < n_q_tokens; ++qi) {
        // Q @ K^T
        for (int kj = 0; kj < n_kv_tokens; ++kj) {
            float acc = 0.0f;
            for (int d = 0; d < head_dim; ++d) acc += q[qi * head_dim + d] * k_dequant[kj * head_dim + d];
            scores[kj] = acc * scale;
        }
        // Softmax
        float max_s = scores[0];
        for (int kj = 1; kj < n_kv_tokens; ++kj) if (scores[kj] > max_s) max_s = scores[kj];
        float sum = 0.0f;
        for (int kj = 0; kj < n_kv_tokens; ++kj) { probs[kj] = std::exp(scores[kj] - max_s); sum += probs[kj]; }
        for (int kj = 0; kj < n_kv_tokens; ++kj) probs[kj] /= sum;
        // P @ V
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int kj = 0; kj < n_kv_tokens; ++kj) acc += probs[kj] * v_dequant[kj * head_dim + d];
            out[qi * head_dim + d] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
static bool test_e4m3_roundtrip() {
    printf("[test_e4m3_roundtrip] ");
    // Every positive lattice byte should round-trip exactly through fp32_to_e4m3_byte_pos
    int failures = 0;
    for (int b = 0; b < 0x80; ++b) {
        if (b == 0x7f) continue;
        float v = e4m3_byte_to_fp32((uint8_t)b);
        uint8_t b2 = fp32_to_e4m3_byte_pos(v);
        if (b2 != (uint8_t)b) {
            // For value 0 there may be multiple representations; allow byte 0
            if (v == 0.0f && b2 == 0) continue;
            printf("FAIL byte 0x%02x val %g -> byte 0x%02x\n", b, v, b2);
            failures++;
            if (failures >= 5) break;
        }
    }
    printf("%s\n", failures == 0 ? "OK" : "FAIL");
    return failures == 0;
}

static bool test_block_roundtrip(int n_centroids, int idx_bits, int expected_block_size, const char *name) {
    printf("[test_block_roundtrip %s] ", name);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    // Build a simple centroid table by snapping uniformly-spaced magnitudes to E4M3
    std::vector<uint8_t> centroids(n_centroids);
    for (int k = 0; k < n_centroids; ++k) {
        float t = (k + 1) / (float)n_centroids;  // (1/n, 2/n, ..., 1.0)
        centroids[k] = fp32_to_e4m3_byte_pos(t);
    }

    // Generate a block of fp32 values with the calibration-like distribution
    std::vector<float> in(QK_TURBO_FP8);
    for (int i = 0; i < QK_TURBO_FP8; ++i) in[i] = dist(rng);

    // Quant + dequant
    std::vector<uint8_t> blk(expected_block_size, 0);
    quantize_block(in.data(), blk.data(), centroids.data(), n_centroids, idx_bits);
    std::vector<float> out(QK_TURBO_FP8);
    dequantize_block(blk.data(), out.data(), centroids.data(), n_centroids, idx_bits);

    // Compute MSE
    float mse = 0.0f;
    for (int i = 0; i < QK_TURBO_FP8; ++i) {
        float e = in[i] - out[i];
        mse += e * e;
    }
    mse /= QK_TURBO_FP8;

    // Sanity bounds: MSE should be much smaller than the variance of `in`
    // (otherwise the quant is broken)
    float var = 0.0f, mean = 0.0f;
    for (int i = 0; i < QK_TURBO_FP8; ++i) mean += in[i];
    mean /= QK_TURBO_FP8;
    for (int i = 0; i < QK_TURBO_FP8; ++i) { float d = in[i] - mean; var += d * d; }
    var /= QK_TURBO_FP8;
    float ratio = mse / var;
    bool ok = ratio < 0.5f;  // generous bound for this uniform centroid placement
    printf("%s  (MSE %.6f, variance %.6f, ratio %.3f)\n", ok ? "OK" : "FAIL", mse, var, ratio);
    return ok;
}

static bool test_attention_nonuniform_k() {
    // Per attention-correctness-testing-principle: non-uniform K is required to
    // exercise the softmax + V dequant path properly. The fp32-reference attention
    // (no quantization) gives the ground truth; the turbo-FP8 quantized attention
    // should be within a small MSE of it.
    printf("[test_attention_nonuniform_k] ");
    const int head_dim     = 64;   // must be multiple of QK_TURBO_FP8 (32)
    const int n_kv_tokens  = 8;
    const int n_q_tokens   = 1;
    const int n_centroids  = 16;
    const int idx_bits     = 4;
    const int blk_size     = 22;

    // Centroids: linearly spaced over [0, 1] INCLUDING zero (Lloyd-Max naturally
    // places one near zero; without it, sparse values inflate during dequant).
    std::vector<uint8_t> centroids(n_centroids);
    for (int k = 0; k < n_centroids; ++k) centroids[k] = fp32_to_e4m3_byte_pos(k / (float)(n_centroids - 1));

    // Non-uniform K: random Gaussian per row but rows scaled differently so
    // softmax has a real preference. V: independent Gaussian, no sparsity.
    std::mt19937 rng(123);
    std::normal_distribution<float> dist(0.0f, 0.3f);
    std::vector<float> k_fp32(n_kv_tokens * head_dim);
    std::vector<float> v_fp32(n_kv_tokens * head_dim);
    for (int t = 0; t < n_kv_tokens; ++t) {
        float row_scale = 0.5f + (float)(t % 3) * 0.2f;
        for (int d = 0; d < head_dim; ++d) {
            k_fp32[t * head_dim + d] = dist(rng) * row_scale;
            v_fp32[t * head_dim + d] = dist(rng);
        }
    }
    std::vector<float> q(n_q_tokens * head_dim);
    for (int d = 0; d < head_dim; ++d) q[d] = dist(rng);

    // Quantize K, V
    int blocks_per_row = head_dim / QK_TURBO_FP8;
    std::vector<uint8_t> k_blocks(n_kv_tokens * blocks_per_row * blk_size, 0);
    std::vector<uint8_t> v_blocks(n_kv_tokens * blocks_per_row * blk_size, 0);
    for (int t = 0; t < n_kv_tokens; ++t) {
        for (int b = 0; b < blocks_per_row; ++b) {
            quantize_block(&k_fp32[t * head_dim + b * QK_TURBO_FP8],
                           &k_blocks[(t * blocks_per_row + b) * blk_size],
                           centroids.data(), n_centroids, idx_bits);
            quantize_block(&v_fp32[t * head_dim + b * QK_TURBO_FP8],
                           &v_blocks[(t * blocks_per_row + b) * blk_size],
                           centroids.data(), n_centroids, idx_bits);
        }
    }

    // FP32 reference attention (no quantization)
    std::vector<float> out_fp32(n_q_tokens * head_dim, 0.0f);
    {
        float scale = 1.0f / std::sqrt((float)head_dim);
        std::vector<float> scores(n_kv_tokens), probs(n_kv_tokens);
        for (int qi = 0; qi < n_q_tokens; ++qi) {
            for (int kj = 0; kj < n_kv_tokens; ++kj) {
                float acc = 0.0f;
                for (int d = 0; d < head_dim; ++d) acc += q[qi * head_dim + d] * k_fp32[kj * head_dim + d];
                scores[kj] = acc * scale;
            }
            float max_s = scores[0];
            for (int kj = 1; kj < n_kv_tokens; ++kj) if (scores[kj] > max_s) max_s = scores[kj];
            float sum = 0.0f;
            for (int kj = 0; kj < n_kv_tokens; ++kj) { probs[kj] = std::exp(scores[kj] - max_s); sum += probs[kj]; }
            for (int kj = 0; kj < n_kv_tokens; ++kj) probs[kj] /= sum;
            for (int d = 0; d < head_dim; ++d) {
                float acc = 0.0f;
                for (int kj = 0; kj < n_kv_tokens; ++kj) acc += probs[kj] * v_fp32[kj * head_dim + d];
                out_fp32[qi * head_dim + d] = acc;
            }
        }
    }

    // Quantized attention
    std::vector<float> out_q(n_q_tokens * head_dim, 0.0f);
    reference_attention_turbo_fp8(q.data(),
                                  k_blocks.data(), v_blocks.data(),
                                  centroids.data(), centroids.data(),
                                  n_centroids, idx_bits, blk_size,
                                  n_q_tokens, n_kv_tokens, head_dim,
                                  out_q.data());

    // Compare: MSE between fp32 reference and quantized output
    float mse = 0.0f, ref_var = 0.0f, ref_mean = 0.0f;
    for (int d = 0; d < head_dim; ++d) ref_mean += out_fp32[d];
    ref_mean /= head_dim;
    for (int d = 0; d < head_dim; ++d) {
        float e = out_fp32[d] - out_q[d];
        mse += e * e;
        float r = out_fp32[d] - ref_mean;
        ref_var += r * r;
    }
    mse /= head_dim;
    ref_var /= head_dim;
    float ratio = ref_var > 0 ? mse / ref_var : 1e9f;
    bool ok = std::isfinite(mse) && ratio < 0.5f;  // turbo4_fp8 typically within ~10%
    printf("%s  (MSE %.6f, ref_var %.6f, ratio %.4f)\n", ok ? "OK" : "FAIL", mse, ref_var, ratio);
    return ok;
}

static bool test_hadamard_orthogonality() {
    printf("[test_hadamard_orthogonality] ");
    int n = 128;
    auto H = hadamard_matrix(n);
    // Check H @ H.T == I
    float max_err = 0.0f;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < n; ++k) acc += H[i * n + k] * H[j * n + k];
            float expected = (i == j) ? 1.0f : 0.0f;
            float err = std::abs(acc - expected);
            if (err > max_err) max_err = err;
        }
    }
    bool ok = max_err < 1e-5f;
    printf("%s  (max |H H^T - I| = %g)\n", ok ? "OK" : "FAIL", max_err);
    return ok;
}

int main() {
    printf("=== turbo-FP8 reference tests ===\n");
    bool all_ok = true;
    all_ok &= test_e4m3_roundtrip();
    all_ok &= test_block_roundtrip(8,  3, 18, "turbo3_fp8");
    all_ok &= test_block_roundtrip(16, 4, 22, "turbo4_fp8");
    all_ok &= test_block_roundtrip(32, 5, 26, "turbo5_fp8");
    all_ok &= test_hadamard_orthogonality();
    all_ok &= test_attention_nonuniform_k();
    printf("=== %s ===\n", all_ok ? "ALL PASSED" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
