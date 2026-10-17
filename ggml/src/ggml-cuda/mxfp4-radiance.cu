// mxfp4-radiance.cu -- MXFP4 MUL_MAT / MUL_MAT_ID on rdna4 radiance packed weights.
//
// The weight bytes are the radiance layout (gemm_mxfp4_radiance.h), not block_mxfp4,
// so this file is the only path that may read a flagged tensor. Activations are
// quantized to fp8 hi/lo (always hi/lo: it is the accuracy fix), then the radiance
// GEMM runs on ctx.stream(). A GEMM or quant decline aborts rather than falling back.
//
// Decode vs tiled activations: the GEMM's decode kernel reads row-major A, the atiled
// kernel reads the tiled layout. We pick tiled exactly when M is above the decode band
// (RDNA4_MXFP4_RADIANCE_DECODE_MAX_M, default 64, same env the kernel reads).
//
// MUL_MAT_ID is fully device-side (no host sync, HIP-graph capturable): ggml-cuda's mmq/mmf routing helper
// (ggml_cuda_launch_mm_ids_helper) produces per-expert bounds and the dst-row index list on the device, and
// rdna4_gemm_mxfp4_radiance_moe groups the routed src1 rows by expert (16-row padded segments), quantizes them
// to fp8 hi/lo and runs the radiance GEMM over all experts in one launch per kernel family; results are
// scattered straight into dst by the GEMM epilogue.

#include "mxfp4-radiance.cuh"
#include "ggml-cuda.h"
#include "ggml-backend-impl.h"
#include "mmid.cuh"
#include "mmq.cuh"

#include <climits>
#include <cstdlib>

#if __has_include("aiter-integration/rdna4_fp8_gemm/gemm_mxfp4_radiance.h")
#include "aiter-integration/rdna4_fp8_gemm/gemm_mxfp4_radiance.h"
#define MXFP4_RADIANCE_AVAILABLE 1
#else
#define MXFP4_RADIANCE_AVAILABLE 0
static inline bool rdna4_mxfp4_radiance_shape_ok(int, int) { return false; }
static inline size_t rdna4_mxfp4_radiance_packed_bytes(int, int) { return 0; }
static inline bool rdna4_mxfp4_radiance_repack(const void *, int, int, void *, cudaStream_t) { return false; }
static inline size_t rdna4_mxfp4_radiance_a_bytes(int, int, bool) { return 0; }
static inline bool rdna4_mxfp4_radiance_quant_a(const float *, int64_t, int, int, uint8_t *, uint8_t *,
                                                float *, bool, cudaStream_t) { return false; }
static inline size_t rdna4_mxfp4_radiance_scratch_bytes(int, int, int) { return 0; }
static inline bool rdna4_gemm_mxfp4_radiance(int, const uint8_t *, const uint8_t *, const float *, const void *,
                                             float *, int, int, void *, size_t, bool, cudaStream_t) { return false; }
static inline bool rdna4_mxfp4_radiance_unpack(const void *, int, int, void *, cudaStream_t) { return false; }
static inline size_t rdna4_mxfp4_radiance_moe_ws_bytes(int, int, int, int, int) { return 0; }
static inline size_t rdna4_mxfp4_radiance_moe_rows_ws_bytes(int, int, int, int) { return 0; }
static inline bool rdna4_gemm_mxfp4_radiance_moe_rows(const float *, int64_t, int64_t, bool, const int *, int, int,
                                                      int, const void *, int64_t, const void * const *, int, int64_t,
                                                      float *, int, int, void *, size_t, cudaStream_t) { return false; }
static inline bool rdna4_gemm_mxfp4_radiance_moe(const float *, int64_t, int64_t, bool, const int *, const int *, int,
                                                 int, int, const void *, int64_t, const void * const *, int, int64_t,
                                                 float *, int, int, void *, size_t, cudaStream_t) { return false; }
#endif

// Decode band the GEMM uses for row-major A (same env and default as gemm_mxfp4_radiance.hip).
static int mxfp4_radiance_decode_max_m() {
    static const int v = [] {
        const char * e = std::getenv("RDNA4_MXFP4_RADIANCE_DECODE_MAX_M");
        return e ? std::atoi(e) : 64;
    }();
    return v;
}

// Largest n_tok served by the routing-free row mode (decode / small verify): every routed row is its own M=1 problem.
// Above it the expert-grouped path batches rows per expert. Default 1: with n_tok>1 several rows usually share an expert (and the worker's dense mmid ids repeat every selected expert for every token), where the grouped path reads each expert once.
static int mxfp4_radiance_row_max_tok() {
    static const int v = [] {
        const char * e = std::getenv("RDNA4_MXFP4_RADIANCE_ROW_MAX_TOK");
        return e ? std::atoi(e) : 1;
    }();
    return v;
}

bool ggml_cuda_mxfp4_radiance_available(int cc) {
#if MXFP4_RADIANCE_AVAILABLE
    return cc == GGML_CUDA_CC_GFX1201;
#else
    GGML_UNUSED(cc);
    return false;
#endif
}

// Weight geometry of a flagged src0: N = rows of ONE packed matrix (a fused gate;up pair holds two of them).
struct rad_weight {
    int64_t N;
    int64_t K;
    int     n_mats;
    int64_t mat_off;   // bytes between the pair's matrices
};

static rad_weight rad_weight_of(const ggml_tensor * a) {
    rad_weight w;
    w.K      = a->ne[0];
    w.n_mats = (a->flags & GGML_TENSOR_FLAG_MXFP4_RADIANCE_PAIR) ? 2 : 1;
    w.N      = a->ne[1] / w.n_mats;
    w.mat_off = w.n_mats == 2 ? (int64_t) a->op_params[0] : 0;
    return w;
}

bool ggml_cuda_mxfp4_radiance_supports_op(const ggml_tensor * op, int cc) {
    if (!ggml_cuda_mxfp4_radiance_available(cc)) {
        return false;
    }
    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];
    if (a->type != GGML_TYPE_MXFP4 || !(a->flags & GGML_TENSOR_FLAG_MXFP4_RADIANCE)) {
        return false;
    }
    if (b->type != GGML_TYPE_F32 || op->type != GGML_TYPE_F32) {
        return false;
    }
    const bool pair = (a->flags & GGML_TENSOR_FLAG_MXFP4_RADIANCE_PAIR) != 0;
    if (pair && (a->ne[1] % 2 != 0 || a->op_params[0] <= 0)) {
        return false;
    }
    const rad_weight w = rad_weight_of(a);
    if (!rdna4_mxfp4_radiance_shape_ok((int) w.N, (int) w.K) || a->ne[3] != 1) {
        return false;
    }
    if (pair && w.mat_off < (int64_t) rdna4_mxfp4_radiance_packed_bytes((int) w.N, (int) w.K)) {
        return false;
    }
    // f32 operands with contiguous K rows (row stride may be > K), contiguous dst rows.
    if (b->nb[0] != sizeof(float) || b->nb[1] % sizeof(float) != 0 || b->ne[0] != a->ne[0]) {
        return false;
    }
    if (op->nb[0] != sizeof(float) || op->nb[1] != op->ne[0] * (int64_t) sizeof(float)) {
        return false;
    }

    if (op->op == GGML_OP_MUL_MAT) {
        return a->ne[2] == 1;
    }

    // MUL_MAT_ID
    const ggml_tensor * ids = op->src[2];
    if (ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
        ids->nb[1] != ids->ne[0] * (int64_t) sizeof(int32_t) || ids->ne[2] != 1 || ids->ne[3] != 1) {
        return false;
    }
    if (b->ne[2] != ids->ne[1] || (b->ne[1] != 1 && b->ne[1] != ids->ne[0]) || b->ne[3] != 1) {
        return false;
    }
    // grouped GEMM addresses src1 rows by (slot, token) strides in floats and scatters dst rows by token*n_used+slot
    if (b->nb[2] % sizeof(float) != 0 || op->nb[2] != ids->ne[0] * op->nb[1] ||
        a->nb[2] < (size_t) (w.n_mats == 2 ? w.mat_off + rdna4_mxfp4_radiance_packed_bytes((int) w.N, (int) w.K)
                                            : (int64_t) rdna4_mxfp4_radiance_packed_bytes((int) w.N, (int) w.K))) {
        return false;
    }
    return a->ne[2] >= 1 && ids->ne[1] <= 16384 && ids->ne[0] < (1 << 10);
}

// y[M, N*n_mats] (row-major, contiguous) = x[M, K] (row stride ld floats) . dequant(w_packed matrices)^T on
// ctx.stream(). n_mats = 2 (fused gate;up pair): the activations are quantized to fp8 hi/lo ONCE, one GEMM per matrix
// (the second at w_packed + mat_off), and each result is placed at its column block of the [M, 2N] output.
static void mxfp4_radiance_gemm(ggml_backend_cuda_context & ctx, const void * w_packed, int64_t N, int64_t K,
                                int n_mats, int64_t mat_off, const float * x, int64_t ld, int64_t M, float * y) {
    const bool tiled = M > mxfp4_radiance_decode_max_m();
    const size_t a_bytes   = rdna4_mxfp4_radiance_a_bytes((int) M, (int) K, tiled);
    const size_t scr_bytes = rdna4_mxfp4_radiance_scratch_bytes((int) M, (int) N, (int) K);

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<uint8_t> a_hi(pool, a_bytes);
    ggml_cuda_pool_alloc<uint8_t> a_lo(pool, a_bytes);
    ggml_cuda_pool_alloc<float>   a_scale(pool, (size_t) M);
    ggml_cuda_pool_alloc<uint8_t> scratch(pool, scr_bytes > 0 ? scr_bytes : 1);
    ggml_cuda_pool_alloc<float>   y_tmp(pool, n_mats > 1 ? (size_t) (M * N) : 0);

    cudaStream_t st = ctx.stream();
    if (!rdna4_mxfp4_radiance_quant_a(x, ld, (int) M, (int) K, a_hi.get(), a_lo.get(), a_scale.get(), tiled, st)) {
        GGML_ABORT("mxfp4 radiance: activation quant declined (M=%lld K=%lld)", (long long) M, (long long) K);
    }
    for (int mi = 0; mi < n_mats; ++mi) {
        const void * w = (const char *) w_packed + (size_t) mi * (size_t) mat_off;
        float * out = n_mats > 1 ? y_tmp.get() : y;
        if (!rdna4_gemm_mxfp4_radiance((int) M, a_hi.get(), a_lo.get(), a_scale.get(), w, out, (int) N, (int) K,
                                       scr_bytes > 0 ? scratch.get() : nullptr, scr_bytes, tiled, st)) {
            GGML_ABORT("mxfp4 radiance: GEMM declined (M=%lld N=%lld K=%lld tiled=%d, decode band %d; "
                       "a flagged tensor cannot fall back to stock MXFP4)",
                       (long long) M, (long long) N, (long long) K, (int) tiled, mxfp4_radiance_decode_max_m());
        }
        if (n_mats > 1) {
            CUDA_CHECK(cudaMemcpy2DAsync(y + (size_t) mi * N, (size_t) (n_mats * N) * sizeof(float), out,
                                         (size_t) N * sizeof(float), (size_t) N * sizeof(float), (size_t) M,
                                         cudaMemcpyDeviceToDevice, st));
        }
    }
}

void ggml_cuda_mul_mat_mxfp4_radiance(ggml_backend_cuda_context & ctx, const ggml_tensor * src0,
                                      const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_mxfp4_radiance_available(cc)) {
        GGML_ABORT("mxfp4 radiance: flagged weight %s on a device without radiance MXFP4 support", src0->name);
    }
    GGML_ASSERT(src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const rad_weight w = rad_weight_of(src0);
    const int64_t K = w.K;
    const int64_t N = w.N;
    const int64_t M = src1->ne[1];
    if (!rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        GGML_ABORT("mxfp4 radiance: unsupported shape N=%lld K=%lld", (long long) N, (long long) K);
    }
    GGML_ASSERT(src1->ne[0] == K && src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(dst->ne[0] == N * w.n_mats && dst->ne[1] == M && dst->nb[0] == sizeof(float) &&
                dst->nb[1] == N * w.n_mats * (int64_t) sizeof(float));
    GGML_ASSERT(dst->ne[2] == src1->ne[2] && dst->ne[3] == src1->ne[3]);

    const int64_t ld = src1->nb[1] / (int64_t) sizeof(float);
    for (int64_t i13 = 0; i13 < src1->ne[3]; ++i13) {
        for (int64_t i12 = 0; i12 < src1->ne[2]; ++i12) {
            const char * x = (const char *) src1->data + i12 * src1->nb[2] + i13 * src1->nb[3];
            char *       y = (char *) dst->data + i12 * dst->nb[2] + i13 * dst->nb[3];
            mxfp4_radiance_gemm(ctx, src0->data, N, K, w.n_mats, w.mat_off, (const float *) x, ld, M, (float *) y);
        }
    }
}

void ggml_cuda_mul_mat_id_mxfp4_radiance(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (!ggml_cuda_mxfp4_radiance_available(cc)) {
        GGML_ABORT("mxfp4 radiance: flagged MUL_MAT_ID weight %s on a device without radiance MXFP4 support", src0->name);
    }
    GGML_ASSERT(src0->type == GGML_TYPE_MXFP4 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && ids->nb[0] == sizeof(int32_t));

    const rad_weight w  = rad_weight_of(src0);
    const int64_t K     = w.K;
    const int64_t N     = w.N;   // rows of one packed matrix; dst rows are N*n_mats wide
    const int64_t n_as  = src0->ne[2];
    const int64_t n_ids = ids->ne[0];
    const int64_t n_tok = ids->ne[1];
    if (!rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        GGML_ABORT("mxfp4 radiance: unsupported MUL_MAT_ID shape N=%lld K=%lld", (long long) N, (long long) K);
    }
    GGML_ASSERT(src1->ne[0] == K && src1->ne[2] == n_tok && (src1->ne[1] == 1 || src1->ne[1] == n_ids));
    GGML_ASSERT(src1->nb[0] == sizeof(float) && src1->nb[1] % sizeof(float) == 0);
    GGML_ASSERT(dst->ne[0] == N * w.n_mats && dst->ne[1] == n_ids && dst->ne[2] == n_tok);
    GGML_ASSERT(dst->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[1] == n_ids * (int64_t) sizeof(int32_t));  // ids rows contiguous
    // Weight-pager expert pointers (batch-mmid path): the per-expert matrix base comes from the pointer table, so the
    // slots need no uniform stride. Consumed exactly once, like the stock mmq / mmvq paths do.
    const void * const * wptrs = ggml_cuda_resolve_mul_mat_id_expert_ptrs(dst);
    // without a table, each expert's packed matrix (pair) needs its own stride: nb[2] must hold the packed matrices
    GGML_ASSERT(wptrs != nullptr || (size_t) src0->nb[2] >= (size_t) ((w.n_mats - 1) * w.mat_off) +
                                             rdna4_mxfp4_radiance_packed_bytes((int) N, (int) K));

    GGML_ASSERT(dst->nb[2] == n_ids * dst->nb[1] && dst->nb[1] == N * w.n_mats * (int64_t) sizeof(float));
    GGML_ASSERT(src1->nb[2] % sizeof(float) == 0 && n_tok < (1 << 22) && n_ids < (1 << 10));
    GGML_ASSERT(n_as <= INT_MAX && n_tok * n_ids <= INT_MAX);

    cudaStream_t st = ctx.stream();
    ggml_cuda_pool & pool = ctx.pool();

    if (n_tok <= mxfp4_radiance_row_max_tok() && mxfp4_radiance_decode_max_m() >= 1) {
        // row mode: no routing helper / plan; 2 kernels per matrix pair (quant + GEMM)
        const size_t rows_ws = rdna4_mxfp4_radiance_moe_rows_ws_bytes((int) n_ids, (int) n_tok, (int) N, (int) K);
        if (rows_ws != 0) {
            ggml_cuda_pool_alloc<uint8_t> ws_rows(pool, rows_ws);
            if (!rdna4_gemm_mxfp4_radiance_moe_rows((const float *) src1->data, src1->nb[1] / (int64_t) sizeof(float),
                                                    src1->nb[2] / (int64_t) sizeof(float), src1->ne[1] == 1,
                                                    (const int32_t *) ids->data, (int) n_as, (int) n_ids, (int) n_tok,
                                                    src0->data, (int64_t) src0->nb[2], wptrs, w.n_mats, w.mat_off,
                                                    (float *) dst->data, (int) N, (int) K, ws_rows.get(), rows_ws, st)) {
                GGML_ABORT("mxfp4 radiance: row-mode MUL_MAT_ID GEMM declined (n_as=%lld n_used=%lld n_tok=%lld N=%lld K=%lld)",
                           (long long) n_as, (long long) n_ids, (long long) n_tok, (long long) N, (long long) K);
            }
            return;
        }
    }

    // device-side routing: expert_bounds[n_as+1] and ids_dst[n_tok*n_ids] (dst row = token*n_ids + slot)
    ggml_cuda_pool_alloc<int32_t> ids_src1(pool, (size_t) (n_ids * n_tok));
    ggml_cuda_pool_alloc<int32_t> ids_dst(pool, (size_t) (n_ids * n_tok));
    ggml_cuda_pool_alloc<int32_t> bounds(pool, (size_t) n_as + 1);
    ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), bounds.get(),
                                   (int) n_as, (int) n_tok, (int) n_ids, (int) src1->ne[1],
                                   (int) (ids->nb[1] / sizeof(int32_t)), 1, false, st);

    const size_t ws_bytes = rdna4_mxfp4_radiance_moe_ws_bytes((int) n_as, (int) n_ids, (int) n_tok, (int) N, (int) K);
    if (ws_bytes == 0) {
        GGML_ABORT("mxfp4 radiance: no grouped MUL_MAT_ID plan (n_as=%lld n_used=%lld n_tok=%lld N=%lld K=%lld)",
                   (long long) n_as, (long long) n_ids, (long long) n_tok, (long long) N, (long long) K);
    }
    ggml_cuda_pool_alloc<uint8_t> ws(pool, ws_bytes);
    if (!rdna4_gemm_mxfp4_radiance_moe((const float *) src1->data, src1->nb[1] / (int64_t) sizeof(float),
                                       src1->nb[2] / (int64_t) sizeof(float), src1->ne[1] == 1, bounds.get(),
                                       ids_dst.get(), (int) n_as, (int) n_ids, (int) n_tok, src0->data,
                                       (int64_t) src0->nb[2], wptrs, w.n_mats, w.mat_off, (float *) dst->data, (int) N,
                                       (int) K, ws.get(), ws_bytes, st)) {
        GGML_ABORT("mxfp4 radiance: grouped MUL_MAT_ID GEMM declined (n_as=%lld n_used=%lld n_tok=%lld N=%lld K=%lld)",
                   (long long) n_as, (long long) n_ids, (long long) n_tok, (long long) N, (long long) K);
    }
}

// ---- exported helpers (ggml-cuda.h) ----

bool ggml_backend_cuda_mxfp4_radiance_repack(ggml_backend_t backend, const void * src_blocks_dev,
                                             void * dst_packed_dev, int64_t N, int64_t K) {
    if (!backend || !ggml_backend_is_cuda(backend)) {
        return false;
    }
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    if (!ggml_cuda_mxfp4_radiance_available(ggml_cuda_info().devices[cctx->device].cc)) {
        return false;
    }
    if (N <= 0 || K <= 0 || N > INT_MAX || K > INT_MAX || !rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        return false;
    }
    return rdna4_mxfp4_radiance_repack(src_blocks_dev, (int) N, (int) K, dst_packed_dev, cctx->stream());
}

bool ggml_backend_cuda_mxfp4_radiance_unpack(ggml_backend_t backend, const void * packed_dev, void * dst_blocks_dev,
                                             int64_t N, int64_t K) {
    if (!backend || !ggml_backend_is_cuda(backend) || packed_dev == nullptr || dst_blocks_dev == nullptr) {
        return false;
    }
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    if (!ggml_cuda_mxfp4_radiance_available(ggml_cuda_info().devices[cctx->device].cc)) {
        return false;
    }
    if (N <= 0 || K <= 0 || N > INT_MAX || K > INT_MAX || !rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        return false;
    }
    return rdna4_mxfp4_radiance_unpack(packed_dev, (int) N, (int) K, dst_blocks_dev, cctx->stream());
}

size_t ggml_backend_cuda_mxfp4_radiance_packed_bytes(int64_t N, int64_t K) {
    if (N <= 0 || K <= 0 || N > INT_MAX || K > INT_MAX || !rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        return 0;
    }
    return rdna4_mxfp4_radiance_packed_bytes((int) N, (int) K);
}

bool ggml_backend_cuda_mxfp4_radiance_available(ggml_backend_t backend) {
    if (!backend || !ggml_backend_is_cuda(backend)) {
        return false;
    }
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    return ggml_cuda_mxfp4_radiance_available(ggml_cuda_info().devices[cctx->device].cc);
}

// Worker slot repack: role_dev holds stock block_mxfp4 bytes (stock_bytes of them) at its base; on return it holds
// the radiance packed layout (packed_bytes(N,K) bytes). scratch_dev must hold at least stock_bytes and must not
// overlap role_dev. The D2D copy and the repack are enqueued on the backend's WP copy stream when it is enabled
// (so they stay ordered with the worker's H2D and its copy event), otherwise on the compute stream. host_wait
// makes this call return only after the work has completed. Returns false (nothing enqueued) on unsupported
// device/shape or on a failed enqueue; the caller must treat that as fatal for a flagged tensor.
bool ggml_backend_cuda_wp_mxfp4_radiance_stage(ggml_backend_t backend, void * role_dev, void * scratch_dev,
                                               size_t stock_bytes, int64_t N, int64_t K, bool host_wait) {
    if (!ggml_backend_cuda_mxfp4_radiance_available(backend) || role_dev == nullptr || scratch_dev == nullptr ||
        role_dev == scratch_dev || stock_bytes == 0) {
        return false;
    }
    if (N <= 0 || K <= 0 || N > INT_MAX || K > INT_MAX || !rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        return false;
    }
    if ((size_t) N * ggml_row_size(GGML_TYPE_MXFP4, K) != stock_bytes) {
        return false;
    }
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cctx->device);
    const bool use_copy = ggml_backend_cuda_wp_copy_stream_enabled(backend);
    cudaStream_t st = use_copy ? cctx->wp_copy_stream : cctx->stream();
    if (st == nullptr) {
        return false;
    }
    if (cudaMemcpyAsync(scratch_dev, role_dev, stock_bytes, cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
        return false;
    }
    if (!rdna4_mxfp4_radiance_repack(scratch_dev, (int) N, (int) K, role_dev, st)) {
        return false;
    }
    if (use_copy) {
        if (cudaEventRecord(cctx->wp_copy_latest_event, st) != cudaSuccess) {
            return false;
        }
        cctx->wp_copy_pending = true;
    }
    // Without the copy stream nothing orders the compute stream after this work for the caller, so always wait.
    if (host_wait || !use_copy) {
        return cudaStreamSynchronize(st) == cudaSuccess;
    }
    return true;
}

// Per-thread non-blocking stream (one per device) for worker helpers that are called from threads other than the compute
// thread (reader / demote threads). Using the backend's compute stream from there would enqueue onto a stream that
// another thread may be capturing into a HIP graph, and the shared WP copy stream is the drain thread's. All work on
// this stream is host-synchronized before the call returns, so nothing needs to be ordered after it by event.
static cudaStream_t rad_thread_stream(int device) {
    thread_local cudaStream_t streams[GGML_CUDA_MAX_DEVICES] = {};
    cudaStream_t & st = streams[device];
    if (st == nullptr && cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) {
        st = nullptr;
    }
    return st;
}

// Per-thread device scratch for rad_thread_stream work (grown on demand, freed at thread exit).
struct rad_thread_scratch {
    void * ptr[GGML_CUDA_MAX_DEVICES] = {};
    size_t bytes[GGML_CUDA_MAX_DEVICES] = {};
    ~rad_thread_scratch() {
        for (int d = 0; d < GGML_CUDA_MAX_DEVICES; ++d) {
            if (ptr[d] != nullptr) {
                cudaFree(ptr[d]);   // ignore errors: the process may be tearing down
            }
        }
    }
    void * get(int device, size_t n) {
        if (bytes[device] < n) {
            if (ptr[device] != nullptr) {
                cudaFree(ptr[device]);
                ptr[device] = nullptr;
                bytes[device] = 0;
            }
            if (cudaMalloc(&ptr[device], n) != cudaSuccess) {
                ptr[device] = nullptr;
                return nullptr;
            }
            bytes[device] = n;
        }
        return ptr[device];
    }
};

static bool rad_role_args_ok(ggml_backend_t backend, const void * a, size_t stock_bytes, int64_t N, int64_t K) {
    if (!ggml_backend_cuda_mxfp4_radiance_available(backend) || a == nullptr || stock_bytes == 0) {
        return false;
    }
    if (N <= 0 || K <= 0 || N > INT_MAX || K > INT_MAX || !rdna4_mxfp4_radiance_shape_ok((int) N, (int) K)) {
        return false;
    }
    return (size_t) N * ggml_row_size(GGML_TYPE_MXFP4, K) == stock_bytes;
}

// Worker slot unpack (inverse of the stage above) for D2H of a packed role: role_dev holds the radiance packed layout;
// on return scratch_dev holds the stock block_mxfp4 bytes (stock_bytes of them), ready for D2H. Runs on a per-thread
// stream and is always host-synchronized; safe from any thread. scratch_dev must not overlap role_dev.
bool ggml_backend_cuda_wp_mxfp4_radiance_unstage(ggml_backend_t backend, const void * role_dev, void * scratch_dev,
                                                 size_t stock_bytes, int64_t N, int64_t K, bool host_wait) {
    GGML_UNUSED(host_wait);
    if (!rad_role_args_ok(backend, role_dev, stock_bytes, N, K) || scratch_dev == nullptr || role_dev == scratch_dev) {
        return false;
    }
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cctx->device);
    cudaStream_t st = rad_thread_stream(cctx->device);
    // The slot's repack may still be queued on the WP copy stream (page-ins no longer host-wait for it): drain that
    // stream first so the unpack reads finished packed bytes. Idle in the common case.
    if (cctx->wp_copy_enabled && cctx->wp_copy_stream != nullptr &&
        cudaStreamSynchronize(cctx->wp_copy_stream) != cudaSuccess) {
        return false;
    }
    if (st == nullptr || !rdna4_mxfp4_radiance_unpack(role_dev, (int) N, (int) K, scratch_dev, st)) {
        return false;
    }
    return cudaStreamSynchronize(st) == cudaSuccess;
}

// Reader-thread repack: role_dev holds stock block_mxfp4 bytes (the reader thread's host-synchronized H2D has retired);
// on return it holds the packed layout. Uses a per-thread stream and a per-thread device scratch (several reader threads
// repack concurrently), and is host-synchronized, so the page may be published as soon as this returns.
bool ggml_backend_cuda_wp_mxfp4_radiance_stage_reader(ggml_backend_t backend, void * role_dev, size_t stock_bytes,
                                                      int64_t N, int64_t K) {
    if (!rad_role_args_ok(backend, role_dev, stock_bytes, N, K)) {
        return false;
    }
    ggml_backend_cuda_context * cctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cctx->device);
    thread_local rad_thread_scratch scratch;
    void * scr = scratch.get(cctx->device, stock_bytes);
    cudaStream_t st = rad_thread_stream(cctx->device);
    if (scr == nullptr || st == nullptr) {
        return false;
    }
    if (cudaMemcpyAsync(scr, role_dev, stock_bytes, cudaMemcpyDeviceToDevice, st) != cudaSuccess ||
        !rdna4_mxfp4_radiance_repack(scr, (int) N, (int) K, role_dev, st)) {
        return false;
    }
    return cudaStreamSynchronize(st) == cudaSuccess;
}
