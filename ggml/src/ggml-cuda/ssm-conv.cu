#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"
#include "mt_gdn_r4d.cuh"  // GGML_HIP_R4D-gated R4D conv-side hook (no-op stub otherwise; MAD-406)

// BF16 activation coverage (2026-09-18): src0 (conv_x), the conv weight (src1) and dst
// are each independently templated (f32 or bf16); bias stays f32. Internal math is fp32.
template <bool apply_silu, size_t split_d_inner, size_t d_conv, typename T = float, typename Tout = T, typename W = float>
static __global__ void ssm_conv_f32(const T * src0_ptr, const W * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    Tout * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const T     * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const W     * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    Tout        * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const T     * x_block = (const T *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const W     * w_block = (const W *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    Tout        * y_block = (Tout *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(T);
    const int stride_w = src1_nb1 / sizeof(W);
    const int stride_y = dst_nb1 / sizeof(Tout);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = (float) w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = (float) x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = (float) x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = (Tout) (apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf);
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t, typename T = float, typename Tout = T, typename W = float>
static __global__ void ssm_conv_long_token_f32(const T * __restrict__ src0, const W * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, Tout * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const T     * x_block = (const T *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const W     * w_block = (const W *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    Tout        * y_block =
        (Tout *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(T);
    const int stride_w = src1_nb1 / sizeof(W);
    const int stride_y = dst_nb1 / sizeof(Tout);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = (float) x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = (float) w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = (Tout) (apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf);
    }
}

template <bool apply_silu, typename T = float, typename Tout = T, typename W = float>
static void ssm_conv_f32_cuda(const T * src0, const W * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, Tout * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC, T, Tout, W>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t, T, Tout, W><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    // MAD-406: R4D conv-prep hook (mt_gdn_r4d.cu), gated by MAD_USE_R4D_GDN_CONV=1 -- mirrors the
    // mt_pagedattn.cu r4d gate pattern. Always declines today (op-boundary miss: a/b/A_log/dt_bias
    // are not reachable from this op's srcs; see mt_gdn_r4d.cu's header comment), so this falls
    // straight through to the existing body below unless something re-wires the gate inputs.
    if (r4d_gdn_conv_enabled() && ggml_cuda_op_ssm_conv_r4d(ctx, dst, bias_add_node, silu_dst)) {
        return;
    }
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    // BF16 activation coverage (2026-09-18): src0 (conv_x), src1 (conv1d.weight) and
    // out are each independently f32 or bf16; bias stays f32. `out`'s type should
    // match src0's (bf16 in -> bf16 out) once ggml_ssm_conv() (ggml.c) is updated to
    // stop hard-coding a GGML_TYPE_F32 result; until then also accept out==F32 with a
    // bf16 src0 so the existing graph-builder can still exercise the bf16 input path.
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_BF16);
    GGML_ASSERT(out->type == src0->type || (src0->type == GGML_TYPE_BF16 && out->type == GGML_TYPE_F32));

    const size_t ts0 = ggml_type_size(src0->type);
    const size_t ts1 = ggml_type_size(src1->type);
    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == ts0);
    GGML_ASSERT(src1->nb[0] == ts1);
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * ts0);

    const void  * src0_d = src0->data;
    const void  * src1_d = src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    void        * dst_d  = out->data;
    cudaStream_t  stream = ctx.stream();

    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

#define SSM_CONV_DISPATCH(T_IN, T_W, T_OUT)                                                                          \
    do {                                                                                                             \
        if (fuse_silu) {                                                                                             \
            ssm_conv_f32_cuda<true, T_IN, T_OUT, T_W>((const T_IN *) src0_d, (const T_W *) src1_d, bias_d,           \
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], (T_OUT *) dst_d, out->nb[0], out->nb[1],         \
                out->nb[2], nc, nr, n_t, n_s, stream);                                                               \
        } else {                                                                                                     \
            ssm_conv_f32_cuda<false, T_IN, T_OUT, T_W>((const T_IN *) src0_d, (const T_W *) src1_d, bias_d,          \
                src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], (T_OUT *) dst_d, out->nb[0], out->nb[1],         \
                out->nb[2], nc, nr, n_t, n_s, stream);                                                               \
        }                                                                                                            \
        return;                                                                                                      \
    } while (0)

    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && out->type == GGML_TYPE_BF16) {
        SSM_CONV_DISPATCH(nv_bfloat16, nv_bfloat16, nv_bfloat16);
    }
    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && out->type == GGML_TYPE_F32) {
        SSM_CONV_DISPATCH(nv_bfloat16, nv_bfloat16, float);
    }
    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && out->type == GGML_TYPE_BF16) {
        SSM_CONV_DISPATCH(nv_bfloat16, float, nv_bfloat16);
    }
    if (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32) {
        SSM_CONV_DISPATCH(nv_bfloat16, float, float);
    }
    GGML_ASSERT(src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && out->type == GGML_TYPE_F32);
    SSM_CONV_DISPATCH(float, float, float);
#undef SSM_CONV_DISPATCH
}
