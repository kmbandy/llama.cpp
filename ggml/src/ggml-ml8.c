// ggml-ml8.c
//
// MAD-223 Phase G.3: CPU compute for ml8-4 matmul nodes.
// See ggml-ml8.h for the API contract.
//
// Implementation: dispatched via `ggml_custom_4d` (GGML_OP_CUSTOM). The
// callback dequantizes `w` block-by-block via `dequantize_row_ml8_4_with_lut`
// into a thread-local fp32 buffer, then computes the standard matmul against
// `x` row-by-row. Multi-threaded via the (ith, nth) split parameters.
//
// Performance is "correctness reference" — see ML8_GGUF_INTEGRATION_DESIGN.md
// Phase G.8 for vectorized fallback. G.4 lands the HIP backend path.

#include "ggml-ml8.h"
#include "ggml-quants.h"
#include "ggml-common.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ─────────────────────────────────────────────────────────────────────────
// Per-op userdata. Stored on heap and freed via custom_op cleanup convention
// — except ggml_custom_4d's API has no cleanup hook, so we leak this for now
// (tiny, one per graph build). G.3b will move the LUT pointer-passing into
// op_params if leak proves problematic at scale.
// ─────────────────────────────────────────────────────────────────────────
struct ml8_custom_args {
    int64_t K;
    int64_t N;
    int64_t M;
    int64_t n_groups_k;
};

// Compute callback. Output dst has shape [N, M] fp32 row-major.
//   dst->src[0] = w (ML8_4)
//   dst->src[1] = centroids (F8_E4M3, [16, n_groups_k])
//   dst->src[2] = x (F32, [K, M])
static void ml8_mul_mat_compute(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ml8_custom_args * a = (const struct ml8_custom_args *) userdata;
    const struct ggml_tensor * w   = dst->src[0];
    const struct ggml_tensor * lut = dst->src[1];
    const struct ggml_tensor * x   = dst->src[2];

    const int64_t K = a->K;
    const int64_t N = a->N;
    const int64_t M = a->M;
    const int64_t n_groups_k = a->n_groups_k;
    GGML_ASSERT(K % QK_ML8 == 0);
    GGML_ASSERT(n_groups_k == K / QK_ML8);

    const block_ml8_4 * w_blocks      = (const block_ml8_4 *) w->data;
    const uint8_t     * lut_fp8       = (const uint8_t     *) lut->data;
    const float       * x_data        = (const float       *) x->data;
    float             * dst_data      = (float             *) dst->data;

    // Per-row strides (in element counts). For contiguous tensors ggml_row_size
    // gives bytes; we use the canonical layout assumption (verified by caller).
    const int64_t w_stride_blocks_per_row = n_groups_k;  // one row of W is n_groups_k blocks
    (void) w_stride_blocks_per_row;

    // Partition N rows across threads.
    const int64_t n_per_thread = (N + nth - 1) / nth;
    const int64_t n_start      = (int64_t) ith * n_per_thread;
    const int64_t n_end        = (n_start + n_per_thread < N) ? (n_start + n_per_thread) : N;

    // Per-thread fp32 scratch for one row of dequantized W (K floats).
    float * w_row_fp32 = (float *) malloc((size_t) K * sizeof(float));
    if (!w_row_fp32) {
        // Out of memory in compute path — log and zero-fill to avoid undefined behavior.
        fprintf(stderr, "ml8_mul_mat_compute: malloc(%zu) failed\n", (size_t)K * sizeof(float));
        for (int64_t n = n_start; n < n_end; n++) {
            for (int64_t m = 0; m < M; m++) {
                dst_data[m * N + n] = 0.0f;
            }
        }
        return;
    }

    for (int64_t n = n_start; n < n_end; n++) {
        // Dequantize row n of W (n_groups_k blocks) → w_row_fp32 [K]
        const block_ml8_4 * w_row = &w_blocks[n * n_groups_k];
        dequantize_row_ml8_4_with_lut(w_row, lut_fp8, w_row_fp32, K);
        // Dot product with each column m of X to produce dst[m][n]
        for (int64_t m = 0; m < M; m++) {
            const float * x_col = &x_data[m * K];
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                sum += w_row_fp32[k] * x_col[k];
            }
            dst_data[m * N + n] = sum;
        }
    }

    free(w_row_fp32);
}

// ─── Kronecker rotation (MAD-223 inference-time activation rotation) ─────
//
// Per scripts/calibration/kronecker_rotation.py: H_b is the Sylvester
// Hadamard of size b_dim, deterministic & orthogonal. We construct it once
// per callback invocation into a thread-shared scratch buffer (b² fp32) and
// then per token compute Y = H_a^T @ X @ H_b where X is x reshaped to (a, b).
//
// Memory cost: O(b²) scratch for H_b + O(d) per-token scratch — both small at
// production b ≤ 1024 and shared activation dims.

struct ml8_rotation_args {
    int64_t a_dim;
    int64_t b_dim;
    int64_t d_dim;       // = a_dim * b_dim, must equal x->ne[0]
    int64_t n_tokens;
};

// Build the b×b Sylvester Hadamard, normalized so H @ H.T = I.
// H_1 = [[1]]; H_{2k} = [[H_k, H_k], [H_k, -H_k]] / sqrt(2).
static void ml8_build_sylvester(float * H, int64_t b) {
    H[0] = 1.0f;
    for (int64_t n = 1; n < b; n *= 2) {
        const float inv_sqrt2 = 0.70710678118654752440f;
        // expand n×n top-left block to 2n×2n, in-place, blocks written stride = 2n
        const int64_t stride_old = b;
        const int64_t stride_new = b;
        // copy current top-left n×n into the other three quadrants
        for (int64_t i = n - 1; i >= 0; i--) {
            for (int64_t j = n - 1; j >= 0; j--) {
                const float v = H[i * stride_old + j] * inv_sqrt2;
                H[(i    ) * stride_new + (j    )] =  v;
                H[(i    ) * stride_new + (j + n)] =  v;
                H[(i + n) * stride_new + (j    )] =  v;
                H[(i + n) * stride_new + (j + n)] = -v;
            }
        }
    }
}

static void ml8_apply_rotation_compute(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    const struct ml8_rotation_args * a = (const struct ml8_rotation_args *) userdata;
    const struct ggml_tensor * x    = dst->src[0];
    const struct ggml_tensor * h_a  = dst->src[1];

    const int64_t a_dim    = a->a_dim;
    const int64_t b_dim    = a->b_dim;
    const int64_t d_dim    = a->d_dim;
    const int64_t n_tokens = a->n_tokens;
    GGML_ASSERT(d_dim == a_dim * b_dim);
    GGML_ASSERT(x->ne[0] == d_dim);

    const float * x_data   = (const float *) x->data;
    const float * h_a_data = (const float *) h_a->data;
    float       * y_data   = (float *) dst->data;

    // Thread 0 fills the shared H_b scratch; all threads wait by partition
    // (no real barrier here — instead, each thread builds its own copy,
    // cheap at b ≤ 1024 and keeps the callback stateless).
    float * h_b = (float *) malloc((size_t) b_dim * b_dim * sizeof(float));
    if (!h_b) {
        fprintf(stderr, "ml8_apply_rotation_compute: malloc h_b failed\n");
        return;
    }
    ml8_build_sylvester(h_b, b_dim);

    // Per-token scratch: an (a, b) buffer for the intermediate X' = H_a^T @ X.
    float * xp = (float *) malloc((size_t) d_dim * sizeof(float));
    if (!xp) {
        fprintf(stderr, "ml8_apply_rotation_compute: malloc xp failed\n");
        free(h_b);
        return;
    }

    // Partition tokens across threads.
    const int64_t per_thread = (n_tokens + nth - 1) / nth;
    const int64_t t_start    = (int64_t) ith * per_thread;
    const int64_t t_end      = (t_start + per_thread < n_tokens) ? (t_start + per_thread) : n_tokens;

    for (int64_t t = t_start; t < t_end; t++) {
        const float * xt = x_data + t * d_dim;   // x[:, t] linearized [d_dim]
        float       * yt = y_data + t * d_dim;

        // Reshape xt as X[k, l] where k ∈ [0, a), l ∈ [0, b), index = k*b + l.
        // Step 1: xp[k, l] = sum_i H_a[i, k] * X[i, l]   (H_a^T @ X over the a axis)
        // Step 2: yt[k, l] = sum_j xp[k, j] * H_b[j, l]  (X' @ H_b over the b axis)
        // h_a is stored row-major in ggml ([a, a] with ne0=a, ne1=a means
        // h_a_data[row*a + col] = H_a[col, row] under PyTorch convention,
        // since ggml ne[0] is the contiguous dim). Verify with calibration:
        // when called from build_layer_ffn we pass the F32 tensor straight
        // from the GGUF, so the on-disk layout matches whatever
        // ml8_to_gguf.py wrote. ml8_to_gguf.py uses numpy default (C-order),
        // GGUF stores as ne[0]=ncols, so h_a_data[i*a+j] = H_a[i, j].
        // i.e. H_a[i, j] = h_a_data[i*a + j].

        // Step 1: xp[k, l] = sum_i H_a[i, k] * X[i, l]
        for (int64_t k = 0; k < a_dim; k++) {
            for (int64_t l = 0; l < b_dim; l++) {
                float s = 0.0f;
                for (int64_t i = 0; i < a_dim; i++) {
                    s += h_a_data[i * a_dim + k] * xt[i * b_dim + l];
                }
                xp[k * b_dim + l] = s;
            }
        }

        // Step 2: yt[k, l] = sum_j xp[k, j] * H_b[j, l]
        for (int64_t k = 0; k < a_dim; k++) {
            for (int64_t l = 0; l < b_dim; l++) {
                float s = 0.0f;
                for (int64_t j = 0; j < b_dim; j++) {
                    s += xp[k * b_dim + j] * h_b[j * b_dim + l];
                }
                yt[k * b_dim + l] = s;
            }
        }
    }

    free(xp);
    free(h_b);
}

struct ggml_tensor * ggml_ml8_apply_rotation(
        struct ggml_context * ctx,
        struct ggml_tensor  * x,
        struct ggml_tensor  * h_a,
        int64_t a_dim,
        int64_t b_dim) {
    GGML_ASSERT(x != NULL);
    if (h_a == NULL) {
        // No rotation configured — return input unchanged.
        return x;
    }
    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(h_a->type == GGML_TYPE_F32);
    GGML_ASSERT(a_dim > 0);
    GGML_ASSERT(b_dim > 0 && (b_dim & (b_dim - 1)) == 0 && "b_dim must be a positive power of 2");
    GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);

    const int64_t d_dim = a_dim * b_dim;
    GGML_ASSERT(x->ne[0] == d_dim);

    struct ggml_tensor * y = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                                 x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
    y->op     = GGML_OP_ML8_APPLY_ROTATION;
    y->src[0] = x;
    y->src[1] = h_a;
    // Pack (a_dim, b_dim) into op_params as int32. Both fit easily —
    // a_dim is typically 5..9, b_dim is a power of 2 up to ~1024.
    int32_t * params = (int32_t *) y->op_params;
    params[0] = (int32_t) a_dim;
    params[1] = (int32_t) b_dim;
    return y;
}

struct ggml_tensor * ggml_ml8_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * x) {
    GGML_ASSERT(w         != NULL);
    GGML_ASSERT(centroids != NULL);
    GGML_ASSERT(x         != NULL);
    GGML_ASSERT(w->type         == GGML_TYPE_ML8_4);
    GGML_ASSERT(centroids->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(x->type         == GGML_TYPE_F32);

    // Shape: w [K, N], x [K, M] → y [N, M]   (ggml row-major)
    const int64_t K = w->ne[0];
    const int64_t N = w->ne[1];
    GGML_ASSERT(x->ne[0] == K && "x and w must share leading K dim");
    GGML_ASSERT(K % QK_ML8 == 0 && "K must be a multiple of QK_ML8=64");

    const int64_t n_groups_k = K / QK_ML8;
    GGML_ASSERT(centroids->ne[0] == 16);
    GGML_ASSERT(centroids->ne[1] == n_groups_k);

    // MAD-223 G.4.c — proper GGML_OP_ML8_MUL_MAT op (replaces previous
    // ggml_custom_4d wiring). Backends (cpu / hip) implement this op directly.
    const int64_t ne[4] = { N, x->ne[1], x->ne[2], x->ne[3] };
    struct ggml_tensor * y = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    y->op     = GGML_OP_ML8_MUL_MAT;
    y->src[0] = w;
    y->src[1] = centroids;
    y->src[2] = x;
    return y;
}

struct ggml_tensor * ggml_ml8_mul_mat_id(
        struct ggml_context * ctx,
        struct ggml_tensor  * w,
        struct ggml_tensor  * centroids,
        struct ggml_tensor  * x,
        struct ggml_tensor  * ids) {
    GGML_ASSERT(w         != NULL);
    GGML_ASSERT(centroids != NULL);
    GGML_ASSERT(x         != NULL);
    GGML_ASSERT(ids       != NULL);
    GGML_ASSERT(w->type == GGML_TYPE_ML8_4 || w->type == GGML_TYPE_ML8_4_SOA);
    GGML_ASSERT(centroids->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(x->type         == GGML_TYPE_F32);
    GGML_ASSERT(ids->type       == GGML_TYPE_I32);

    // Shape contract mirrors ggml_mul_mat_id:
    //   w         [K, N, n_experts]
    //   centroids [16, n_groups_k, n_experts]
    //   x         [K, n_expert_used, n_tokens]
    //   ids       [n_expert_used, n_tokens]
    //   y         [N, n_expert_used, n_tokens]
    const int64_t K         = w->ne[0];
    const int64_t N         = w->ne[1];
    const int64_t n_experts = w->ne[2];
    GGML_ASSERT(K % QK_ML8 == 0 && "K must be a multiple of QK_ML8=64");

    const int64_t n_groups_k = K / QK_ML8;
    GGML_ASSERT(centroids->ne[0] == 16);
    GGML_ASSERT(centroids->ne[1] == n_groups_k);
    GGML_ASSERT(centroids->ne[2] == n_experts);

    GGML_ASSERT(x->ne[0] == K);
    GGML_ASSERT(ids->ne[0] == x->ne[1] && "ids and x must agree on n_expert_used");
    GGML_ASSERT(ids->ne[1] == x->ne[2] && "ids and x must agree on n_tokens");

    const int64_t ne[4] = { N, x->ne[1], x->ne[2], 1 };
    struct ggml_tensor * y = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne);
    y->op     = GGML_OP_ML8_MUL_MAT_ID;
    y->src[0] = w;
    y->src[1] = centroids;
    y->src[2] = x;
    y->src[3] = ids;
    return y;
}
