// mt_sparse_attn_dsv4.cu — see .cuh for design notes.
//
// Only compiled when ggml-hip is built with -DGGML_HIP_AITER=ON. The
// aiter_triton_aot static library (libaiter_triton_aot.a), linked into
// ggml-hip when GGML_HIP_AITER=ON, provides mt_sparse_attn_dsv4().

#include "mt_sparse_attn_dsv4.cuh"

#ifdef GGML_HIP_AITER
#include "mt_sparse_attn_dsv4.h"
#endif

#include <cstdlib>
#include <cstdio>

namespace mt {

#ifdef GGML_HIP_AITER

bool sparse_attn_dsv4_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("WP_DSV41_SPARSE_ATTN");
        return v && v[0] == '1';
    }();
    return enabled;
}

bool sparse_attn_dsv4_check_enabled() {
    static const bool enabled = [] {
        const char * v = std::getenv("WP_DSV41_SPARSE_ATTN_CHECK");
        return v && v[0] == '1';
    }();
    return enabled;
}

#endif // GGML_HIP_AITER

bool ggml_cuda_sparse_attn_dsv4_supported(const ggml_tensor * op) {
#ifndef GGML_HIP_AITER
    (void) op;
    return false;
#else
    if (!sparse_attn_dsv4_enabled()) {
        return false;
    }
    const ggml_tensor * q          = op->src[0];
    const ggml_tensor * k_all      = op->src[1];
    const ggml_tensor * kv_indices = op->src[2];
    const ggml_tensor * attn_sink  = op->src[3];
    const ggml_tensor * kv_indptr  = op->src[4];
    if (!q || !k_all || !kv_indices || !attn_sink || !kv_indptr) {
        return false;
    }
    if (kv_indptr->type != GGML_TYPE_I32 || !ggml_is_contiguous(kv_indptr) || kv_indptr->ne[0] != q->ne[2] + 1) {
        return false;
    }
    // Baked AOT-compile-time shape assumptions (see mt_sparse_attn_dsv4.h) --
    // must match exactly, or the compiled kernels' constexpr-baked strides
    // and masks are silently wrong for this call. DS4.1's own hparams are
    // fixed at head_dim=512/n_heads=64, so this should always pass for the
    // op this graph builder actually emits; a mismatch here means the graph
    // builder's own check (build_attention_v41) has a bug, or something else
    // is trying to reuse this op outside its one intended call site.
    if (q->ne[0] != MT_SPARSE_ATTN_DSV4_HEAD_DIM || q->ne[1] != MT_SPARSE_ATTN_DSV4_NUM_HEADS) {
        return false;
    }
    if (op->type       != GGML_TYPE_F16 ||
        q->type         != GGML_TYPE_F16 ||
        k_all->type     != GGML_TYPE_F16 ||
        kv_indices->type != GGML_TYPE_I32 ||
        attn_sink->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(q) || !ggml_is_contiguous(k_all) ||
        !ggml_is_contiguous(kv_indices) || !ggml_is_contiguous(attn_sink)) {
        return false;
    }
    return true;
#endif
}

void ggml_cuda_op_sparse_attn_dsv4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(ctx);
    GGML_ABORT("ggml_cuda_op_sparse_attn_dsv4: called without GGML_HIP_AITER");
#else
    const ggml_tensor * q          = dst->src[0];
    const ggml_tensor * k_all      = dst->src[1];
    const ggml_tensor * kv_indices = dst->src[2];
    const ggml_tensor * attn_sink  = dst->src[3];

    // Defensive -- supports_op should have already refused otherwise.
    GGML_ASSERT(q->ne[0] == MT_SPARSE_ATTN_DSV4_HEAD_DIM);
    GGML_ASSERT(q->ne[1] == MT_SPARSE_ATTN_DSV4_NUM_HEADS);

    float   scale;
    int32_t n_idx_param;
    memcpy(&scale,       (const char *) dst->op_params + 0 * sizeof(int32_t), sizeof(scale));
    memcpy(&n_idx_param, (const char *) dst->op_params + 1 * sizeof(int32_t), sizeof(n_idx_param));
    GGML_ASSERT(n_idx_param == (int32_t) kv_indices->ne[0]);

    mt_sparse_attn_dsv4_args_t args {};
    args.q          = q->data;
    args.k_all      = k_all->data;
    args.kv_indices = (const int32_t *) kv_indices->data;
    args.kv_indptr  = (const int32_t *) dst->src[4]->data;
    args.attn_sink  = (const float *) attn_sink->data;
    args.out        = dst->data;
    args.n_tokens   = (int32_t) q->ne[2];
    args.n_kv       = (int32_t) k_all->ne[2];
    args.n_idx      = n_idx_param;
    args.scale      = scale;

    hipStream_t stream = ctx.stream();
    hipError_t err = mt_sparse_attn_dsv4(stream, &args);
    if (err != hipSuccess) {
        GGML_ABORT("mt_sparse_attn_dsv4 launch failed: %s", hipGetErrorString(err));
    }
#endif
}

} // namespace mt
