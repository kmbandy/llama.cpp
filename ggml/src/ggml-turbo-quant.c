/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBO2_0 (2-bit), GGML_TYPE_TURBO3_0 (3-bit) and
 * GGML_TYPE_TURBO4_0 (4-bit) for use as --cache-type-k turboN in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
GGML_API int turbo3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- TURBO3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turbo3_0_ref(const float * GGML_RESTRICT x, block_turbo3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward WHT rotation
        turbo_cpu_fwht(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO3;

            memset(blk->qs, 0, QK_TURBO3 / 4);
            memset(blk->signs, 0, QK_TURBO3 / 8);

            for (int j = 0; j < QK_TURBO3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo3_0(const block_turbo3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBO3 == 0);
    const int nb = k / QK_TURBO3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBO3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO3 == 0);

    size_t row_size = (n_per_row / QK_TURBO3) * sizeof(block_turbo3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo3_0_ref(
            src + row * n_per_row,
            (block_turbo3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO2_0: 2-bit PolarQuant (no QJL) ---------- */

void quantize_row_turbo2_0_ref(const float * GGML_RESTRICT x, block_turbo2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);

    extern int turbo3_cpu_wht_group_size;
    int group_size = turbo3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBO2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turbo2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turbo2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBO2;

            memset(blk->qs, 0, QK_TURBO2 / 4);

            for (int j = 0; j < QK_TURBO2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turbo2_0(const block_turbo2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBO2 == 0);
    const int nb = k / QK_TURBO2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBO2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBO2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turbo2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO2 == 0);

    size_t row_size = (n_per_row / QK_TURBO2) * sizeof(block_turbo2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo2_0_ref(
            src + row * n_per_row,
            (block_turbo2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBO4_0: 3-bit PolarQuant + 1-bit QJL ---------- */

void quantize_row_turbo4_0_ref(const float * GGML_RESTRICT x, block_turbo4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated, d);

#if TURBO4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBO4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBO4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void dequantize_row_turbo4_0(const block_turbo4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBO4 == 0);
    const int nb = k / QK_TURBO4;
    const int d  = QK_TURBO4;

#if TURBO4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    /* TODO: add proper 4-bit centroid table to C code (currently only in Metal) */
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
        /* No inverse WHT, dequant stays in the rotated domain.
        * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores.
        * The inverse WHT is applied to the attention output via GGML_OP_TURBO_WHT (direction=1) in the graph. 
        */
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

/* MAD-301C Lever B: native head_dim-64 turbo4 (64-element block, no RHT).
 * Mirrors the CUDA paged scatter convention (mt_scatter_kv_turbo4_64_kernel):
 * L2-norm -> normalize -> 4-bit centroid quant (NO WHT) -> recon-norm correction;
 * dequant returns centroid*norm. These CPU refs exist only to satisfy ggml
 * type-traits for GGML_TYPE_TURBO4_64; the hot path is CUDA paged attention. */
void quantize_row_turbo4_64_ref(const float * GGML_RESTRICT x, block_turbo4_64 * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        const float norm = sqrtf(norm_sq);
        float normalized[QK_TURBO4_64];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }
        uint8_t indices[QK_TURBO4_64];
        for (int i = 0; i < d; i++) indices[i] = (uint8_t)nearest_centroid_4bit(normalized[i]);
        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) recon_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        const float recon_norm     = sqrtf(recon_sq);
        const float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
    }
}

void dequantize_row_turbo4_64(const block_turbo4_64 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
    }
}

/* turbo4_64_ol: turbo4_64 with 4 fixed-position "massive activation" outlier
 * channels {53,49,52,20} extracted verbatim at f16, excluded from the
 * group-norm and centroid quantization of the remaining 60 elements. Uses
 * the SAME shared CENTROIDS_4BIT table as turbo4_0/turbo4_64's CPU stub
 * above (NOT the N=64-calibrated table used by the CUDA/Vulkan paged hot
 * path) — removing outliers from the norm makes the remaining 60 "typical"
 * values close enough to the N=128 assumption for this table to be
 * appropriate again. These CPU refs exist only to satisfy ggml type-traits
 * for GGML_TYPE_TURBO4_64_OL; the hot path is CUDA/Vulkan paged attention. */
void quantize_row_turbo4_64_ol_ref(const float * GGML_RESTRICT x, block_turbo4_64_ol * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    static const int outlier_ch[TURBO4_64_OL_N_OUTLIERS] = TURBO4_64_OUTLIER_CHANNELS;
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        bool is_outlier[QK_TURBO4_64];
        memset(is_outlier, 0, sizeof(is_outlier));
        for (int o = 0; o < TURBO4_64_OL_N_OUTLIERS; o++) {
            is_outlier[outlier_ch[o]] = true;
            y[block].outliers[o] = GGML_FP32_TO_FP16(src[outlier_ch[o]]);
        }

        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            if (!is_outlier[i]) norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv  = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

        uint8_t indices[QK_TURBO4_64];
        memset(indices, 0, sizeof(indices));
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            indices[i] = (uint8_t) nearest_centroid_4bit(src[i] * inv);
        }

        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            recon_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        const float recon_norm     = sqrtf(recon_sq);
        const float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);

        memset(y[block].qs, 0, sizeof(y[block].qs));
        int nib = 0;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            y[block].qs[nib / 2] |= (uint8_t)((indices[i] & 0xF) << ((nib % 2) * 4));
            nib++;
        }
    }
}

void dequantize_row_turbo4_64_ol(const block_turbo4_64_ol * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    static const int outlier_ch[TURBO4_64_OL_N_OUTLIERS] = TURBO4_64_OUTLIER_CHANNELS;
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;

        bool is_outlier[QK_TURBO4_64];
        memset(is_outlier, 0, sizeof(is_outlier));
        for (int o = 0; o < TURBO4_64_OL_N_OUTLIERS; o++) {
            is_outlier[outlier_ch[o]] = true;
            dst[outlier_ch[o]] = GGML_FP16_TO_FP32(x[block].outliers[o]);
        }

        int nib = 0;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            uint8_t idx = (x[block].qs[nib / 2] >> ((nib % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
            nib++;
        }
    }
}

/* turbo4_64_ol8 / turbo4_64_ol12: outlier-matrix sweep (2026-07-01) variants
 * of turbo4_64_ol with 8 / 12 fixed outlier channels instead of 4. Mechanical
 * mirror of quantize_row_turbo4_64_ol_ref / dequantize_row_turbo4_64_ol above
 * — same shared CENTROIDS_4BIT table, just a different N_OUTLIERS/channel
 * list and block struct. CPU refs exist only to satisfy ggml type-traits;
 * the hot path is CUDA/Vulkan paged attention. */
void quantize_row_turbo4_64_ol8_ref(const float * GGML_RESTRICT x, block_turbo4_64_ol8 * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    static const int outlier_ch[TURBO4_64_OL8_N_OUTLIERS] = TURBO4_64_OL8_OUTLIER_CHANNELS;
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        bool is_outlier[QK_TURBO4_64];
        memset(is_outlier, 0, sizeof(is_outlier));
        for (int o = 0; o < TURBO4_64_OL8_N_OUTLIERS; o++) {
            is_outlier[outlier_ch[o]] = true;
            y[block].outliers[o] = GGML_FP32_TO_FP16(src[outlier_ch[o]]);
        }

        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            if (!is_outlier[i]) norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv  = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

        uint8_t indices[QK_TURBO4_64];
        memset(indices, 0, sizeof(indices));
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            indices[i] = (uint8_t) nearest_centroid_4bit(src[i] * inv);
        }

        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            recon_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        const float recon_norm     = sqrtf(recon_sq);
        const float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);

        memset(y[block].qs, 0, sizeof(y[block].qs));
        int nib = 0;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            y[block].qs[nib / 2] |= (uint8_t)((indices[i] & 0xF) << ((nib % 2) * 4));
            nib++;
        }
    }
}

void dequantize_row_turbo4_64_ol8(const block_turbo4_64_ol8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    static const int outlier_ch[TURBO4_64_OL8_N_OUTLIERS] = TURBO4_64_OL8_OUTLIER_CHANNELS;
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;

        bool is_outlier[QK_TURBO4_64];
        memset(is_outlier, 0, sizeof(is_outlier));
        for (int o = 0; o < TURBO4_64_OL8_N_OUTLIERS; o++) {
            is_outlier[outlier_ch[o]] = true;
            dst[outlier_ch[o]] = GGML_FP16_TO_FP32(x[block].outliers[o]);
        }

        int nib = 0;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            uint8_t idx = (x[block].qs[nib / 2] >> ((nib % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
            nib++;
        }
    }
}

void quantize_row_turbo4_64_ol12_ref(const float * GGML_RESTRICT x, block_turbo4_64_ol12 * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    static const int outlier_ch[TURBO4_64_OL12_N_OUTLIERS] = TURBO4_64_OL12_OUTLIER_CHANNELS;
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        bool is_outlier[QK_TURBO4_64];
        memset(is_outlier, 0, sizeof(is_outlier));
        for (int o = 0; o < TURBO4_64_OL12_N_OUTLIERS; o++) {
            is_outlier[outlier_ch[o]] = true;
            y[block].outliers[o] = GGML_FP32_TO_FP16(src[outlier_ch[o]]);
        }

        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            if (!is_outlier[i]) norm_sq += src[i] * src[i];
        }
        const float norm = sqrtf(norm_sq);
        const float inv  = (norm > 1e-10f) ? (1.0f / norm) : 0.0f;

        uint8_t indices[QK_TURBO4_64];
        memset(indices, 0, sizeof(indices));
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            indices[i] = (uint8_t) nearest_centroid_4bit(src[i] * inv);
        }

        float recon_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            recon_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        const float recon_norm     = sqrtf(recon_sq);
        const float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);

        memset(y[block].qs, 0, sizeof(y[block].qs));
        int nib = 0;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            y[block].qs[nib / 2] |= (uint8_t)((indices[i] & 0xF) << ((nib % 2) * 4));
            nib++;
        }
    }
}

void dequantize_row_turbo4_64_ol12(const block_turbo4_64_ol12 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    static const int outlier_ch[TURBO4_64_OL12_N_OUTLIERS] = TURBO4_64_OL12_OUTLIER_CHANNELS;
    assert(k % QK_TURBO4_64 == 0);
    const int nb = k / QK_TURBO4_64;
    const int d  = QK_TURBO4_64;
    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;

        bool is_outlier[QK_TURBO4_64];
        memset(is_outlier, 0, sizeof(is_outlier));
        for (int o = 0; o < TURBO4_64_OL12_N_OUTLIERS; o++) {
            is_outlier[outlier_ch[o]] = true;
            dst[outlier_ch[o]] = GGML_FP16_TO_FP32(x[block].outliers[o]);
        }

        int nib = 0;
        for (int i = 0; i < d; i++) {
            if (is_outlier[i]) continue;
            uint8_t idx = (x[block].qs[nib / 2] >> ((nib % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
            nib++;
        }
    }
}

size_t quantize_turbo4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBO4 == 0);

    size_t row_size = (n_per_row / QK_TURBO4) * sizeof(block_turbo4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turbo4_0_ref(
            src + row * n_per_row,
            (block_turbo4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ================================================================== */
/* TQ3_1S / TQ4_1S: WHT-rotated weight quantization                  */
/* ================================================================== */

/* Lloyd-Max centroids for N(0,1) — shared with Metal shaders */
static const float TQ3_0_CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

static const float TQ4_0_CENTROIDS[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

/* WHT sign pattern (golden ratio hash, 32-element blocks) — shared by TQ3 and TQ4 */
static const float TQ3_0_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

#define TQ_BLOCK_SIZE 32
#define TQ_INV_SQRT32 0.17677669529663688f  /* 1/sqrt(32) */

/* Forward RHT: sign flips -> WHT butterfly -> normalize */
static void tq3_0_rht_forward(float * buf) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ3_0_SIGNS[i];
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32;
}

/* Inverse RHT: WHT butterfly -> normalize + unsign */
static void tq3_0_rht_inverse(float * buf) {
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32 * TQ3_0_SIGNS[i];
}

/* Nearest centroid for TQ3 (8 centroids) */
static int tq3_0_choose_index(float val) {
    /* Binary search on midpoints of TQ3_0_CENTROIDS */
    if (val < -1.644041f) return 0;
    if (val < -1.015870f) return 1;
    if (val < -0.493925f) return 2;
    if (val < -0.008701f) return 3;
    if (val <  0.477664f) return 4;
    if (val <  1.001363f) return 5;
    if (val <  1.633223f) return 6;
    return 7;
}

/* Nearest centroid for TQ4 (16 centroids) */
static int tq4_0_choose_index(float val) {
    /* Binary search on midpoints of TQ4_0_CENTROIDS */
    if (val < -2.400804f) return 0;
    if (val < -1.843532f) return 1;
    if (val < -1.437139f) return 2;
    if (val < -1.099286f) return 3;
    if (val < -0.799550f) return 4;
    if (val < -0.522404f) return 5;
    if (val < -0.258222f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.258222f) return 8;
    if (val <  0.522404f) return 9;
    if (val <  0.799550f) return 10;
    if (val <  1.099286f) return 11;
    if (val <  1.437139f) return 12;
    if (val <  1.843532f) return 13;
    if (val <  2.400804f) return 14;
    return 15;
}

/* ---------- TQ3_1S quantization ---------- */

void quantize_row_tq3_1s_ref(const float * GGML_RESTRICT x, block_tq3_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ3_0;
        block_tq3_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv0);
                float c = TQ3_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq3_0_choose_index(buf[j] * inv1);
                float c = TQ3_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ3_0 * 3 / 8);

        /* TQ3 packing: 4 groups of 8 indices packed into 3 bytes each */
        for (int g = 0; g < 4; g++) {
            uint8_t indices[8];
            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float inv = (j < 16) ? inv0 : inv1;
                indices[i] = (uint8_t)tq3_0_choose_index(buf[j] * inv);
            }
            uint8_t * qp = blk->qs + g * 3;
            qp[0] = (indices[0] & 7) | ((indices[1] & 7) << 3) | ((indices[2] & 3) << 6);
            qp[1] = ((indices[2] >> 2) & 1) | ((indices[3] & 7) << 1) | ((indices[4] & 7) << 4) | ((indices[5] & 1) << 7);
            qp[2] = ((indices[5] >> 1) & 3) | ((indices[6] & 7) << 2) | ((indices[7] & 7) << 5);
        }
    }
}

void dequantize_row_tq3_1s(const block_tq3_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        /* Unpack 3-bit indices */
        float buf[32];
        for (int g = 0; g < 4; g++) {
            const uint8_t * qp = x[blk_i].qs + g * 3;
            uint8_t idx[8];
            idx[0] =  qp[0]       & 7;
            idx[1] = (qp[0] >> 3) & 7;
            idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
            idx[3] = (qp[1] >> 1) & 7;
            idx[4] = (qp[1] >> 4) & 7;
            idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
            idx[6] = (qp[2] >> 2) & 7;
            idx[7] = (qp[2] >> 5) & 7;

            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float d = (j < 16) ? d0 : d1;
                buf[j] = TQ3_0_CENTROIDS[idx[i]] * d;
            }
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ3_0, buf, QK_TQ3_0 * sizeof(float));
    }
}

size_t quantize_tq3_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ3_0 == 0);

    size_t row_size = (n_per_row / QK_TQ3_0) * sizeof(block_tq3_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq3_1s_ref(
            src + row * n_per_row,
            (block_tq3_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TQ4_1S quantization ---------- */

void quantize_row_tq4_1s_ref(const float * GGML_RESTRICT x, block_tq4_1s * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int block = 0; block < nb; block++) {
        const float * src_blk = x + block * QK_TQ4_1S;
        block_tq4_1s * blk = &y[block];

        /* 1. Forward RHT */
        float buf[TQ_BLOCK_SIZE];
        memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
        tq3_0_rht_forward(buf);

        /* 2. Split into two halves, compute RMS per half */
        float rms0 = 0.0f, rms1 = 0.0f;
        for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
        for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
        rms0 = sqrtf(rms0 / 16.0f);
        rms1 = sqrtf(rms1 / 16.0f);

        /* 3. Scale search (9 points) */
        static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
        float best_d0 = rms0, best_d1 = rms1;
        float best_err = 1e30f;

        for (int si = 0; si < 9; si++) {
            float d0 = rms0 * scales[si];
            float d1 = rms1 * scales[si];
            float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
            float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

            float err = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d0;
                err += diff * diff;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d1;
                err += diff * diff;
            }
            if (err < best_err) {
                best_err = err;
                best_d0 = d0;
                best_d1 = d1;
            }
        }

        /* 4. Iterative refinement (6 iterations) */
        for (int iter = 0; iter < 6; iter++) {
            float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
            float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

            float num0 = 0.0f, den0 = 0.0f;
            float num1 = 0.0f, den1 = 0.0f;
            for (int j = 0; j < 16; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv0);
                float c = TQ4_0_CENTROIDS[idx];
                num0 += buf[j] * c;
                den0 += c * c;
            }
            for (int j = 16; j < 32; j++) {
                int idx = tq4_0_choose_index(buf[j] * inv1);
                float c = TQ4_0_CENTROIDS[idx];
                num1 += buf[j] * c;
                den1 += c * c;
            }
            if (den0 > 1e-10f) best_d0 = num0 / den0;
            if (den1 > 1e-10f) best_d1 = num1 / den1;
        }

        /* 5. Final quantize + pack (nibble packing) */
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        blk->d0 = GGML_FP32_TO_FP16(best_d0);
        blk->d1 = GGML_FP32_TO_FP16(best_d1);
        memset(blk->qs, 0, QK_TQ4_1S / 2);

        for (int j = 0; j < QK_TQ4_1S; j++) {
            float inv = (j < 16) ? inv0 : inv1;
            int idx = tq4_0_choose_index(buf[j] * inv);
            blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j & 1) * 4));
        }
    }
}

void dequantize_row_tq4_1s(const block_tq4_1s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ4_1S == 0);
    const int nb = k / QK_TQ4_1S;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        float buf[32];
        for (int j = 0; j < 32; j++) {
            uint8_t idx = (x[blk_i].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
            float d = (j < 16) ? d0 : d1;
            buf[j] = TQ4_0_CENTROIDS[idx] * d;
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ4_1S, buf, QK_TQ4_1S * sizeof(float));
    }
}

size_t quantize_tq4_1s(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TQ4_1S == 0);

    size_t row_size = (n_per_row / QK_TQ4_1S) * sizeof(block_tq4_1s);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_tq4_1s_ref(
            src + row * n_per_row,
            (block_tq4_1s *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

// MAD-214: turbo-FP8 BS=256 KV cache stubs. Decode requires a per-(kv, layer)
// runtime centroid LUT that this generic API cannot carry; the production
// fill/drain path is custom CUDA (mt_pagedattn_turbo_fp8.cuh + paged-tile
// scatter). These stubs abort if invoked so misuse is loud.
void quantize_row_turbo4_fp8_bs256_ref(const float * GGML_RESTRICT x, block_turbo4_fp8_bs256 * GGML_RESTRICT y, int64_t k) {
    (void) x; (void) y; (void) k;
    GGML_ABORT("turbo4_fp8_bs256: generic quantize_row not supported (needs per-(kv,layer) runtime LUT); use the custom KV cache scatter path");
}

void dequantize_row_turbo4_fp8_bs256(const block_turbo4_fp8_bs256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    (void) x; (void) y; (void) k;
    GGML_ABORT("turbo4_fp8_bs256: generic dequantize_row not supported (needs per-(kv,layer) runtime LUT); use the custom KV cache decode path");
}

// ──────────────────────────────────────────────────────────────────────────
// MAD-223 Phase G.2: ml8-4 weight quant + fp8 e4m3 centroid storage.
//
// The standard ggml dequant signature `(const block*, float*, n)` has no
// slot for the per-K-group centroid LUT that ml8 requires. We keep the
// standard `dequantize_row_ml8_4` as an abort stub (so misuse is loud) and
// provide `dequantize_row_ml8_4_with_lut` as the actually-useful entry
// point. The matmul graph node (G.3) obtains the LUT from the sidecar
// tensor `<weight>.centroids` via op_params and calls the `_with_lut`
// variant.
//
// fp8 e4m3 conversions follow the PyTorch `float8_e4m3fn` (OCP-style)
// encoding — sign bit + 4-bit exponent (bias 7) + 3-bit mantissa, only
// S.1111.111 = NaN (no inf, no other NaNs), max normal magnitude = 448.
//
// See aiter-integration/ML8_GGUF_INTEGRATION_DESIGN.md §2.
// ──────────────────────────────────────────────────────────────────────────

// e4m3 → fp32 lookup table — precomputed, 256 entries, 1 KB total. Cheaper
// than the bit-twiddling for hot paths and avoids subnormal-shift logic at
// call sites. Initialized lazily on first use; deterministic content.
static float       g_fp8_e4m3_lut[256];
static int         g_fp8_e4m3_lut_init = 0;
static inline void ml8_init_fp8_e4m3_lut(void) {
    if (g_fp8_e4m3_lut_init) return;
    for (int i = 0; i < 256; i++) {
        uint32_t x = (uint32_t) i;
        uint32_t s = (x >> 7) & 1u;
        uint32_t e = (x >> 3) & 0xFu;
        uint32_t m = x & 0x7u;
        float    f;
        if (e == 0 && m == 0) {
            // ±0
            uint32_t bits = s << 31;
            memcpy(&f, &bits, 4);
        } else if (e == 15 && m == 7) {
            // NaN (e4m3fn convention: only S.1111.111 is NaN; no infinities)
            uint32_t bits = (s << 31) | (0xFFu << 23) | (1u << 22);
            memcpy(&f, &bits, 4);
        } else if (e == 0) {
            // Subnormal: value = (-1)^s * 2^(-6) * (m/8)
            // Renormalize for fp32: find leading bit position in m (3 bits)
            int       lead = (m >= 4) ? 2 : (m >= 2) ? 1 : 0;
            uint32_t  mant_norm = ((m << (3 - lead)) & 0x7u);  // strip leading 1
            int       exp_un    = -6 - (2 - lead);
            uint32_t  exp_fp32  = (uint32_t)(exp_un + 127);
            uint32_t  bits      = (s << 31) | (exp_fp32 << 23) | (mant_norm << 20);
            memcpy(&f, &bits, 4);
        } else {
            // Normal: value = (-1)^s * 2^(e-7) * (1 + m/8)
            // fp32 exp = (e-7) + 127 = e + 120
            uint32_t exp_fp32 = e + 120u;
            uint32_t mant     = m << 20;  // 3 bits → top of 23-bit fp32 mantissa
            uint32_t bits     = (s << 31) | (exp_fp32 << 23) | mant;
            memcpy(&f, &bits, 4);
        }
        g_fp8_e4m3_lut[i] = f;
    }
    g_fp8_e4m3_lut_init = 1;
}

// ── ml8_4 ─────────────────────────────────────────────────────────────────

void quantize_row_ml8_4_ref(const float * GGML_RESTRICT x, block_ml8_4 * GGML_RESTRICT y, int64_t k) {
    (void) x; (void) y; (void) k;
    GGML_ABORT("ml8_4: quantize_row_ref not supported — calibration is offline (scripts/calibration/calibrate_ml8.py)");
}

void dequantize_row_ml8_4(const block_ml8_4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    (void) x; (void) y; (void) k;
    GGML_ABORT("ml8_4: generic dequantize_row not supported — use dequantize_row_ml8_4_with_lut (requires per-K-group centroid LUT sidecar)");
}

void dequantize_row_ml8_4_with_lut(const block_ml8_4 * GGML_RESTRICT x,
                                   const uint8_t * GGML_RESTRICT lut_fp8,
                                   float * GGML_RESTRICT y,
                                   int64_t k) {
    GGML_ASSERT(k % QK_ML8 == 0);
    ml8_init_fp8_e4m3_lut();
    const int64_t n_groups = k / QK_ML8;
    for (int64_t g = 0; g < n_groups; g++) {
        const block_ml8_4 *blk    = &x[g];
        const uint8_t     *lut_g  = &lut_fp8[g * 16];
        const float        scale  = blk->scale;
        float             *out    = &y[g * QK_ML8];
        // Per-block: 32 packed bytes → 64 indices → 64 fp32 outputs.
        for (int i = 0; i < QK_ML8 / 2; i++) {
            const uint8_t packed = blk->qs[i];
            const uint8_t lo_idx = packed & 0x0Fu;          // K-position 2i   (lo-nibble first)
            const uint8_t hi_idx = (packed >> 4) & 0x0Fu;   // K-position 2i+1
            out[2 * i]     = g_fp8_e4m3_lut[lut_g[lo_idx]] * scale;
            out[2 * i + 1] = g_fp8_e4m3_lut[lut_g[hi_idx]] * scale;
        }
    }
}

// ── f8_e4m3 ───────────────────────────────────────────────────────────────

void dequantize_row_f8_e4m3(const uint8_t * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    ml8_init_fp8_e4m3_lut();
    for (int64_t i = 0; i < k; i++) {
        y[i] = g_fp8_e4m3_lut[x[i]];
    }
}

// ── ml8_fp8 ───────────────────────────────────────────────────────────────
//
// MAD Task 9: dequantize a row of ml8-fp8 blocks.
//
// Block layout (34 bytes, tightly packed, matching Python writer commit 45925db35):
//   [scale : fp16 LE (2 bytes)] [qs : 32 × uint8 OCP e4m3fn]
//
// output[b*32 + i] = e4m3_decode(qs[i]) * fp16_to_fp32(scale)
//
// Reuses g_fp8_e4m3_lut / ml8_init_fp8_e4m3_lut which are the canonical
// OCP e4m3fn decode shared by all ml8 types.  k must be divisible by 32.

void dequantize_row_ml8_fp8(const block_ml8_fp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_ML8_FP8 == 0);
    ml8_init_fp8_e4m3_lut();
    const int64_t n_blocks = k / QK_ML8_FP8;
    for (int64_t b = 0; b < n_blocks; b++) {
        const float   scale = GGML_FP16_TO_FP32(x[b].scale);
        float       * out   = &y[b * QK_ML8_FP8];
        for (int i = 0; i < QK_ML8_FP8; i++) {
            out[i] = g_fp8_e4m3_lut[x[b].qs[i]] * scale;
        }
    }
}

// ── ml8_fp8 quantize ─────────────────────────────────────────────────────────
//
// MAD Task 10: quantize a row of fp32 → ml8_fp8 blocks.
//
// Block layout (34 bytes): [scale : fp16 LE] [qs : 32 × uint8 OCP e4m3fn]
// scale = amax(|x[i]|) / 448.0  (E4M3_MAX; guard scale==0 → 1.0f)
// qs[i] = f32_to_e4m3(x[i] / scale)
//
// This is the exact inverse of dequantize_row_ml8_fp8.  k must be divisible
// by QK_ML8_FP8 (32).

void quantize_row_ml8_fp8_ref(const float * GGML_RESTRICT x, block_ml8_fp8 * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k % QK_ML8_FP8 == 0);
    const int64_t n_blocks = k / QK_ML8_FP8;

    // Temp buffer for scaled values (one block at a time)
    float scaled[QK_ML8_FP8];

    for (int64_t b = 0; b < n_blocks; b++) {
        const float * blk = &x[b * QK_ML8_FP8];

        // Compute amax
        float amax = 0.0f;
        for (int i = 0; i < QK_ML8_FP8; i++) {
            float av = blk[i] < 0.0f ? -blk[i] : blk[i];
            if (av > amax) amax = av;
        }

        // scale = amax / E4M3_MAX (448.0f); guard zero
        float scale = (amax > 0.0f) ? (amax / 448.0f) : 1.0f;
        y[b].scale = GGML_FP32_TO_FP16(scale);

        // Divide by scale, then encode each element as e4m3fn byte
        float inv_scale = 1.0f / scale;
        for (int i = 0; i < QK_ML8_FP8; i++) {
            scaled[i] = blk[i] * inv_scale;
        }
        quantize_row_f8_e4m3_ref(scaled, y[b].qs, QK_ML8_FP8);
    }
}

size_t quantize_ml8_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    GGML_ASSERT(n_per_row % QK_ML8_FP8 == 0);

    const size_t row_size = (n_per_row / QK_ML8_FP8) * sizeof(block_ml8_fp8);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_ml8_fp8_ref(
            src + row * n_per_row,
            (block_ml8_fp8 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

void quantize_row_f8_e4m3_ref(const float * GGML_RESTRICT x, uint8_t * GGML_RESTRICT y, int64_t k) {
    // Round-to-nearest-even fp32 → fp8 e4m3fn. Saturates at ±448 (no inf in
    // this variant). NaN inputs map to S.1111.111. Used at calibration-time
    // export; not on the hot path.
    for (int64_t i = 0; i < k; i++) {
        const float    xv = x[i];
        uint32_t       bits;
        memcpy(&bits, &xv, 4);
        const uint32_t sign  = (bits >> 31) & 1u;
        const uint32_t exp_b = (bits >> 23) & 0xFFu;
        const uint32_t mant  = bits & 0x7FFFFFu;

        // NaN or Inf input → e4m3 NaN (S.1111.111)
        if (exp_b == 0xFFu) {
            y[i] = (uint8_t)((sign << 7) | 0x7Fu);
            continue;
        }
        // Zero (handles both fp32 zero and fp32 subnormal which maps to e4m3 zero)
        if (exp_b == 0) {
            y[i] = (uint8_t)(sign << 7);
            continue;
        }

        const int32_t e_un = (int32_t) exp_b - 127;

        // Saturate: e4m3 max = S.1111.110 = ±448 = 2^8 * 1.75
        if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u)) {
            y[i] = (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
            continue;
        }

        if (e_un >= -6) {
            // Normal e4m3: e ∈ {1..14}, m ∈ {0..7}
            const uint32_t e_e4m3 = (uint32_t)(e_un + 7);
            // RNE on bottom 20 bits of fp32 mantissa (keep top 3 of 23)
            const uint32_t guard  = (mant >> 19) & 1u;
            const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
            const uint32_t lsb    = (mant >> 20) & 1u;
            uint32_t       m_e4m3 = (mant >> 20) & 0x7u;
            if (guard && (sticky || lsb)) m_e4m3 += 1;
            uint32_t e_out = e_e4m3;
            if (m_e4m3 == 8) {
                m_e4m3 = 0;
                e_out += 1;
                if (e_out >= 15) {
                    y[i] = (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
                    continue;
                }
            }
            // Defensive: don't accidentally synthesize NaN (e=15, m=7)
            if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
            y[i] = (uint8_t)((sign << 7) | (e_out << 3) | m_e4m3);
            continue;
        }

        // Subnormal e4m3: |x| < 2^-6
        // Express target |x| ≈ 2^-6 * (m/8) where m ∈ {0..7}; m = round(|x| * 2^9).
        // |x| * 2^9 = (2^23 + mant) >> (23 - (e_un + 9))   (with RNE on the dropped bits)
        const int32_t shift = 23 - (e_un + 9);  // shift ≥ 21
        if (shift > 31) {
            y[i] = (uint8_t)(sign << 7);
            continue;
        }
        const uint32_t implicit = (1u << 23) | mant;
        const uint32_t guard    = (implicit >> (shift - 1)) & 1u;
        const uint32_t sticky   = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
        uint32_t       m_e4m3   = implicit >> shift;
        const uint32_t lsb      = m_e4m3 & 1u;
        if (guard && (sticky || lsb)) m_e4m3 += 1;

        if (m_e4m3 >= 8) {
            // Rounded into smallest normal e4m3 (e=1, m=0)
            y[i] = (uint8_t)((sign << 7) | (1u << 3));
        } else {
            // e=0 subnormal (m=0 means ±0)
            y[i] = (uint8_t)((sign << 7) | m_e4m3);
        }
    }
}
