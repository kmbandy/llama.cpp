#include "ggml-cuda/common.cuh"
#include "ggml.h"
#include "topk-moe.cuh"
#include "reduce_rows.cuh"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

// Kernel config struct - passed by value to CUDA kernel
struct topk_moe_config {
    bool use_sigmoid;
    bool use_sqrt_softplus;
    bool with_norm;
    bool delayed_softmax;
};

// Warp-local softmax used for both the pre-top-k logits and the post-top-k delayed path.
template <int experts_per_thread, bool use_limit>
__device__ void softmax_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
    float max_val = -INFINITY;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            max_val = max(max_val, vals[i]);
        }
    }

    max_val = warp_reduce_max(max_val);

    float sum = 0.f;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            const float val = expf(vals[i] - max_val);
            vals[i]         = val;
            sum += val;
        } else {
            vals[i] = 0.f;
        }
    }

    sum = warp_reduce_sum(sum);

    const float inv_sum = 1.0f / sum;

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        if (active) {
            vals[i] *= inv_sum;
        }
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sigmoid_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? 1.f / (1.f + expf(-vals[i])) : -INFINITY;
    }
}

template <int experts_per_thread, bool use_limit>
__device__ void sqrt_softplus_warp_inplace(float (&vals)[experts_per_thread], const int limit, const int lane) {
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int  idx    = lane + i * WARP_SIZE;
        const bool active = !use_limit || (idx < limit);
        vals[i]           = active ? sqrtf(vals[i] > 20.0f ? vals[i] : logf(1.0f + expf(vals[i]))) : -INFINITY;
    }
}

/*
    This kernel does the following:
    1. optionally softmax over the logits per token [n_experts, n_tokens]
    2. argmax reduce over the top-k (n_experts_used) logits
    3. write weights + ids to global memory
    4. optionally normalize the weights or apply softmax over the selected logits

    It is intended as fusion of softmax->top-k->get_rows pipeline for MoE models
*/
template <int n_experts, bool has_bias>
__launch_bounds__(TOPK_MOE_ROWS_PER_BLOCK * WARP_SIZE, 1)
__global__ void topk_moe_cuda(const float *         logits,
                              float *               weights,
                              int32_t *             ids,
                              float *               bias,
                              const int             n_rows,
                              const int             n_expert_used,
                              const float           clamp_val,
                              const float           scale_val,
                              const topk_moe_config config) {
#if defined(GGML_USE_MUSA)
    // MUSA: every warp of a partially filled block must reach the barrier below.
    const int row = MIN(blockIdx.x * blockDim.y + threadIdx.y, n_rows - 1);
#else
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
#endif // defined(GGML_USE_MUSA)
    if (row >= n_rows) {
        return;
    }

    logits += n_experts * row;
    weights += n_expert_used * row;
    ids += n_experts * row;

    constexpr int experts_per_thread = (n_experts > WARP_SIZE) ? n_experts / WARP_SIZE : 1;

    float wt[experts_per_thread];

    // Initialize all slots to -INFINITY
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        wt[i] = -INFINITY;
    }

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int i = 0; i < n_experts; i += WARP_SIZE) {
        const int expert  = i + threadIdx.x;
        wt[i / WARP_SIZE] = (n_experts % WARP_SIZE == 0 || expert < n_experts) ? logits[expert] : -INFINITY;
    }

    // Weights and IDs can alias logits, so wait until every row in the block reads its logits.
    __syncthreads();

    if (!config.delayed_softmax) {
        if (config.use_sigmoid) {
           sigmoid_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else if (config.use_sqrt_softplus) {
           sqrt_softplus_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        } else {
           softmax_warp_inplace<experts_per_thread, false>(wt, n_experts, threadIdx.x);
        }
    }

    // Sanitize NaN to -FLT_MAX so the iterative argmax produces unique expert IDs.
    // NaN comparisons always return false, which would cause the same expert to be
    // selected repeatedly. -FLT_MAX compares normally and is still excluded by the
    // -INFINITY sentinel used after each selection round.
    // More relevant for the cuBLAS path. See https://github.com/ggml-org/llama.cpp/issues/19659
#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        if (__isnanf(wt[i])) {
            wt[i] = -FLT_MAX;
        }
    }

    // selection_wt is only needed when bias is present (selection uses wt + bias)
    // when no bias, we use wt directly for both selection and weight values
    [[maybe_unused]] float selection_wt[has_bias ? experts_per_thread : 1];

    if constexpr (has_bias) {
#pragma unroll
        for (int i = 0; i < experts_per_thread; i++) {
            selection_wt[i] = -INFINITY;
        }
#pragma unroll
        for (int i = 0; i < n_experts; i += WARP_SIZE) {
            const int expert = i + threadIdx.x;
            selection_wt[i / WARP_SIZE] =
                (n_experts % WARP_SIZE == 0 || expert < n_experts) ? wt[i / WARP_SIZE] + bias[expert] : -INFINITY;
        }
    }

    //at this point, each thread holds either a portion of the softmax distribution
    //or the raw logits. We do the argmax reduce over n_expert_used, each time marking
    //the expert weight as -inf to exclude from the next iteration

    float wt_sum = 0.f;

    float output_weights[experts_per_thread];

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        output_weights[i] = 0.f;
    }

    ggml_cuda_pdl_lc();
    for (int k = 0; k < n_expert_used; k++) {
        float max_val    = wt[0];
        int   max_expert = threadIdx.x;

        if constexpr (has_bias) {
            float max_val_s = selection_wt[0];

#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && selection_wt[i] > max_val_s) {
                    max_val    = wt[i];
                    max_val_s  = selection_wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const float val_s  = __shfl_xor_sync(0xFFFFFFFF, max_val_s, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val_s > max_val_s || (val_s == max_val_s && expert < max_expert)) {
                    max_val    = val;
                    max_val_s  = val_s;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                selection_wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        } else {
#pragma unroll
            for (int i = 1; i < experts_per_thread; i++) {
                const int expert = threadIdx.x + i * WARP_SIZE;
                if ((n_experts % WARP_SIZE == 0 || expert < n_experts) && wt[i] > max_val) {
                    max_val    = wt[i];
                    max_expert = expert;
                }
            }

#pragma unroll
            for (int mask = WARP_SIZE / 2; mask > 0; mask /= 2) {
                const float val    = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, WARP_SIZE);
                const int   expert = __shfl_xor_sync(0xFFFFFFFF, max_expert, mask, WARP_SIZE);
                if (val > max_val || (val == max_val && expert < max_expert)) {
                    max_val    = val;
                    max_expert = expert;
                }
            }

            if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
                wt[max_expert / WARP_SIZE] = -INFINITY;
            }
        }

        if ((k & (WARP_SIZE - 1)) == threadIdx.x) {
            output_weights[k / WARP_SIZE] = max_val;
        }

        if ((max_expert & (WARP_SIZE - 1)) == threadIdx.x) {
            ids[k] = max_expert;
            if (config.with_norm) {
                wt_sum += max_val;
            }
        }
    }

    if (config.with_norm) {
        wt_sum              = warp_reduce_sum(wt_sum);
        wt_sum              = max(wt_sum, clamp_val);
        const float inv_sum = 1.0f / wt_sum;

        for (int i = 0; i < experts_per_thread; i++) {
            output_weights[i] *= inv_sum;
        }
    }

    if (config.delayed_softmax) {
        softmax_warp_inplace<experts_per_thread, true>(output_weights, n_expert_used, threadIdx.x);
    }

#pragma unroll
    for (int i = 0; i < experts_per_thread; i++) {
        const int idx = i * WARP_SIZE + threadIdx.x;
        if (idx < n_expert_used) {
            weights[idx] = output_weights[i] * scale_val;
        }
    }
}

template<bool has_bias>
static void launch_topk_moe_cuda(ggml_backend_cuda_context & ctx,
                                 const float *               logits,
                                 float *                     weights,
                                 int32_t *                   ids,
                                 float *                     bias,
                                 const int                   n_rows,
                                 const int                   n_expert,
                                 const int                   n_expert_used,
                                 const float                 clamp_val,
                                 const float                 scale_val,
                                 const topk_moe_config       config) {
    GGML_ASSERT(!(config.with_norm && config.delayed_softmax) &&
                "delayed softmax is not supported with weight normalization");
    const int    rows_per_block = TOPK_MOE_ROWS_PER_BLOCK;
    dim3         grid_dims((n_rows + rows_per_block - 1) / rows_per_block, 1, 1);
    dim3         block_dims(WARP_SIZE, rows_per_block, 1);
    cudaStream_t stream = ctx.stream();
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n_expert) {
        case 1:
            ggml_cuda_kernel_launch(topk_moe_cuda<1, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 2:
            ggml_cuda_kernel_launch(topk_moe_cuda<2, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 4:
            ggml_cuda_kernel_launch(topk_moe_cuda<4, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 8:
            ggml_cuda_kernel_launch(topk_moe_cuda<8, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 16:
            ggml_cuda_kernel_launch(topk_moe_cuda<16, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 32:
            ggml_cuda_kernel_launch(topk_moe_cuda<32, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 64:
            ggml_cuda_kernel_launch(topk_moe_cuda<64, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 128:
            ggml_cuda_kernel_launch(topk_moe_cuda<128, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 256:
            ggml_cuda_kernel_launch(topk_moe_cuda<256, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 288: // StepFun 3.7
            ggml_cuda_kernel_launch(topk_moe_cuda<288, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 512:
            ggml_cuda_kernel_launch(topk_moe_cuda<512, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        case 576:
            ggml_cuda_kernel_launch(topk_moe_cuda<576, has_bias>, launch_params,
                logits, weights, ids, bias, n_rows, n_expert_used, clamp_val, scale_val, config);
            break;
        default:
            GGML_ASSERT(false && "fatal error");
    }
}

void ggml_cuda_op_topk_moe(ggml_backend_cuda_context &     ctx,
                           const ggml_tensor *             logits,
                           ggml_tensor *                   weights,
                           ggml_tensor *                   ids,
                           const ggml_tensor *             clamp,
                           const ggml_tensor *             scale,
                           const ggml_tensor *             bias,
                           const ggml_cuda_topk_moe_args & args) {
    GGML_ASSERT(logits->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(ids->type == GGML_TYPE_I32);

    const int n_experts = logits->ne[0];
    const int n_rows    = logits->ne[1];

    const float * logits_d  = (const float *) logits->data;
    float *       weights_d = (float *) weights->data;
    int32_t *     ids_d     = (int32_t *) ids->data;
    float *       bias_d    = bias ? (float *) bias->data : nullptr;

    float scale_val = scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    GGML_ASSERT(ids->nb[1] / ggml_type_size(ids->type) == (size_t) n_experts);

    const int n_expert_used = weights->ne[1];

    const bool with_norm = clamp != nullptr;

    float clamp_val = -INFINITY;
    if (clamp) {
        clamp_val = ggml_get_op_params_f32(clamp, 0);
    }

    topk_moe_config config;
    config.use_sigmoid       = args.sigmoid;
    config.use_sqrt_softplus = args.sqrt_softplus;
    config.with_norm         = with_norm;
    config.delayed_softmax   = args.delayed_softmax;

    if (bias) {
        launch_topk_moe_cuda<true>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    } else {
        launch_topk_moe_cuda<false>(ctx, logits_d, weights_d, ids_d, bias_d, n_rows, n_experts, n_expert_used, clamp_val,
                             scale_val, config);
    }
}

bool ggml_cuda_should_use_topk_moe(const ggml_tensor * gating_op,
                                   const ggml_tensor * weights,
                                   const ggml_tensor * logits,
                                   const ggml_tensor * ids) {
    // must match an instantiation of launch_topk_moe_cuda: a power of 2 up to 512,
    // or one of the non-power-of-2 expert counts of supported models
    const int n_expert = ids->nb[1] / ids->nb[0];
    if (((n_expert & (n_expert - 1)) != 0 || n_expert > 512) && n_expert != 288 && n_expert != 576) {
        return false;
    }

    if (!ggml_is_contiguous(weights) || !ggml_is_contiguous(logits)) {
        return false;
    }

    if (gating_op->op == GGML_OP_SOFT_MAX) {
        const ggml_tensor * softmax  = gating_op;
        float               scale    = 1.0f;
        float               max_bias = 0.0f;

        memcpy(&scale, (const float *) softmax->op_params + 0, sizeof(float));
        memcpy(&max_bias, (const float *) softmax->op_params + 1, sizeof(float));

        if (!ggml_is_contiguous(softmax->src[0])) {
            return false;
        }

        if (scale != 1.0f || max_bias != 0.0f) {
            return false;
        }

        // don't fuse when masks or sinks are present
        if (softmax->src[1] || softmax->src[2]) {
            return false;
        }
    } else if (gating_op->op == GGML_OP_UNARY) {
        ggml_unary_op op = ggml_get_unary_op(gating_op);

        if (op != GGML_UNARY_OP_SIGMOID && op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// WP_DSV4_ROUTER_FUSE: bit-exact fusion of the DeepSeek-V4.1 MoE router tail.
//
// Replaces, per token row, the unfused chain built by build_moe_ffn for
// sqrtsoftplus gating with exp_probs_b and expert weight normalization:
//   p   = sqrt(softplus(logits))            (UNARY softplus, SQRT)
//   key = p + exp_probs_b                   (ADD, selection only)
//   ids = argsort_desc(key)[:n_used]        (ARGSORT, bitonic path, VIEW)
//   w   = get_rows(p, ids)                  (GET_ROWS, unbiased values)
//   s   = clamp(sum_rows(w), min, +inf)     (SUM_ROWS, CLAMP)
//   w   = w / s                             (DIV)
//   w   = scale * w + 0                     (SCALE, only when w_scale != 1)
// Every arithmetic step reuses the exact expression of the unfused CUDA kernel
// (op_softplus, sqrtf, a+b, a/b, fminf(fmaxf()), scale*x+bias). The argsort
// network is the same bitonic network as argsort.cu (ncols <= 1024 path), so ties
// resolve identically. The row sum calls the same reduce_row_f32/block_reduce
// helpers with the same block size that sum_rows_f32_cuda picks for this shape.
// ---------------------------------------------------------------------------

bool ggml_cuda_dsv4_router_fuse_enabled() {
    static const bool v = [] {
        const char * e = std::getenv("WP_DSV4_ROUTER_FUSE");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v;
}

bool ggml_cuda_dsv4_router_fuse_diff_enabled() {
    static const bool v = [] {
        const char * e = std::getenv("WP_DSV4_ROUTER_FUSE_DIFF");
        return e != nullptr && std::strcmp(e, "1") == 0;
    }();
    return v && ggml_cuda_dsv4_router_fuse_enabled();
}

static __device__ __forceinline__ void dsv4_argsort_desc_step(const float * x_row, int * dst_row, const int ncols,
                                                             const int col, const int k, const int j) {
    // identical to bitonic_step<GGML_SORT_ORDER_DESC> in argsort.cu
    const int ixj = col ^ j;
    if (ixj <= col) {
        return;
    }
    if ((col & k) == 0) {
        if (dst_row[col] >= ncols ||
            (dst_row[ixj] < ncols && x_row[dst_row[col]] < x_row[dst_row[ixj]])) {
            const int tmp = dst_row[col];
            dst_row[col]  = dst_row[ixj];
            dst_row[ixj]  = tmp;
        }
    } else {
        if (dst_row[ixj] >= ncols ||
            (dst_row[col] < ncols && x_row[dst_row[col]] > x_row[dst_row[ixj]])) {
            const int tmp = dst_row[col];
            dst_row[col]  = dst_row[ixj];
            dst_row[ixj]  = tmp;
        }
    }
}

// One block per token row; blockDim.x must equal the block size sum_rows picks.
static __global__ void dsv4_router_exact_kernel(const float * __restrict__ logits, const float * __restrict__ bias,
                                                int32_t * __restrict__ ids, const int64_t ids_stride,
                                                float * __restrict__ weights, const int64_t logits_stride,
                                                const int n_expert, const int n_used, const int ncols_pad,
                                                const float clamp_min, const bool do_scale, const float scale_val) {
    extern __shared__ unsigned char dsv4_smem[];
    float * probs_s = (float *) dsv4_smem;      // [n_expert] unbiased sqrt(softplus)
    float * keys_s  = probs_s + n_expert;       // [n_expert] probs + bias (selection keys)
    float * wsh     = keys_s + n_expert;        // [n_used]   gathered weights
    int   * dst_row = (int *) (wsh + n_used);   // [ncols_pad] argsort indices

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float * x_row = logits + (int64_t) row * logits_stride;

    for (int e = tid; e < n_expert; e += blockDim.x) {
        const float x  = x_row[e];
        const float sp = (x > 20.0f) ? x : logf(1.0f + expf(x));   // op_softplus
        const float p  = sqrtf(sp);                                  // op_sqrt
        probs_s[e] = p;
        keys_s[e]  = p + bias[e];                                    // op_add(probs, exp_probs_b)
    }
    for (int col = tid; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }
    __syncthreads();

    // bitonic argsort, descending, same stage structure as k_argsort_f32_i32
    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = tid; col < ncols_pad; col += blockDim.x) {
                dsv4_argsort_desc_step(keys_s, dst_row, n_expert, col, k, j);
            }
            __syncthreads();
        }
    }

    for (int k = tid; k < n_used; k += blockDim.x) {
        const int id = dst_row[k];
        ids[(int64_t) row * ids_stride + k] = id;
        wsh[k] = (id >= 0 && id < n_expert) ? probs_s[id] : 0.0f;   // get_rows
    }
    __syncthreads();

    // same reduction as reduce_rows_f32 (sum_rows) for ncols = n_used
    const float sum   = reduce_row_f32(wsh, n_used);
    const float sum_c = fminf(fmaxf(sum, clamp_min), INFINITY);     // op_clamp(min, +inf)

    for (int k = tid; k < n_used; k += blockDim.x) {
        float v = wsh[k] / sum_c;                                    // op_div
        if (do_scale) {
            v = scale_val * v + 0.0f;                                // op_scale, bias = 0
        }
        weights[(int64_t) row * n_used + k] = v;
    }
}

// block size sum_rows_f32_cuda uses for a contiguous [ncols, nrows] source
static int dsv4_sum_rows_block_size(const ggml_cuda_device_info & info, int device, int ncols, int64_t nrows) {
    const int nsm = info.devices[device].nsm;
    if ((nrows / nsm) < 2) {
        return 512;
    }
    return ncols < 1024 ? 32 : 128;
}

static void dsv4_router_exact_run(ggml_backend_cuda_context & ctx, const float * logits, const float * bias,
                                  int32_t * ids, const int64_t ids_stride, float * weights,
                                  const int64_t logits_stride, const int n_tokens, const int n_expert,
                                  const int n_used, const float clamp_min, const bool do_scale,
                                  const float scale_val) {
    int ncols_pad = 1;
    while (ncols_pad < n_expert) {
        ncols_pad *= 2;
    }
    const int block = dsv4_sum_rows_block_size(ggml_cuda_info(), ctx.device, n_used, n_tokens);
    const size_t smem = sizeof(float) * (2 * (size_t) n_expert + (size_t) n_used) + sizeof(int) * (size_t) ncols_pad;
    GGML_ASSERT(smem <= ggml_cuda_info().devices[ctx.device].smpb);
    GGML_ASSERT(n_expert <= 1024 && n_used <= n_expert);

    dsv4_router_exact_kernel<<<n_tokens, block, smem, ctx.stream()>>>(
        logits, bias, ids, ids_stride, weights, logits_stride, n_expert, n_used, ncols_pad,
        clamp_min, do_scale, scale_val);
}

// Compares the fused result (scratch, contiguous) with the unfused result (real outputs).
static __global__ void dsv4_router_diff_kernel(const int32_t * __restrict__ ids_f, const float * __restrict__ w_f,
                                               const int32_t * __restrict__ ids_u, const int64_t ids_stride,
                                               const float * __restrict__ w_u, const int n_used, const int64_t n,
                                               unsigned long long * counts, unsigned long long * first) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    const int64_t t = idx / n_used;
    const int     k = (int) (idx % n_used);
    const int32_t a_id = ids_f[idx];
    const int32_t b_id = ids_u[t * ids_stride + k];
    const uint32_t a_w = __float_as_uint(w_f[idx]);
    const uint32_t b_w = __float_as_uint(w_u[idx]);
    if (a_id != b_id || a_w != b_w) {
        atomicAdd(counts, 1ull);
        atomicMin(first, (unsigned long long) idx);
    }
}

// Debug mode: the caller has already computed the unfused chain into the real outputs.
void ggml_cuda_op_topk_moe_dsv4_exact_diff(ggml_backend_cuda_context & ctx, const ggml_tensor * logits,
                                           ggml_tensor * weights, ggml_tensor * ids, const ggml_tensor * clamp,
                                           const ggml_tensor * scale, const ggml_tensor * bias) {
    static unsigned long long total_mismatch = 0;
    static unsigned long long total_compared = 0;
    static int                logged         = 0;

    const int n_expert = (int) logits->ne[0];
    const int n_tokens = (int) logits->ne[1];
    const int n_used   = (int) weights->ne[1];
    const int64_t n    = (int64_t) n_used * n_tokens;
    const float clamp_min = ggml_get_op_params_f32(clamp, 0);
    const bool  do_scale  = scale != nullptr;
    const float scale_val = do_scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    ggml_cuda_pool & pool = ctx.pool();
    ggml_cuda_pool_alloc<int32_t> ids_f_alloc(pool, n);
    ggml_cuda_pool_alloc<float>   w_f_alloc(pool, n);
    ggml_cuda_pool_alloc<unsigned long long> cnt_alloc(pool, 2);

    const int64_t ids_stride = ids->nb[1] / sizeof(int32_t);
    dsv4_router_exact_run(ctx, (const float *) logits->data, (const float *) bias->data, ids_f_alloc.get(),
                          n_used, w_f_alloc.get(), logits->nb[1] / sizeof(float), n_tokens, n_expert, n_used,
                          clamp_min, do_scale, scale_val);

    unsigned long long * cnt   = cnt_alloc.get();
    unsigned long long * first = cnt + 1;
    const unsigned long long init[2] = { 0ull, ~0ull };
    CUDA_CHECK(cudaMemcpyAsync(cnt, init, sizeof(init), cudaMemcpyHostToDevice, ctx.stream()));
    dsv4_router_diff_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, ctx.stream()>>>(
        ids_f_alloc.get(), w_f_alloc.get(), (const int32_t *) ids->data, ids_stride,
        (const float *) weights->data, n_used, n, cnt, first);

    unsigned long long host[2] = { 0ull, 0ull };
    CUDA_CHECK(cudaMemcpyAsync(host, cnt, sizeof(host), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));

    total_compared += (unsigned long long) n;
    total_mismatch += host[0];
    if (host[0] != 0 && logged < 32) {
        logged++;
        const unsigned long long idx = host[1];
        const int t = (int) (idx / n_used);
        const int k = (int) (idx % n_used);
        int32_t ids_fv = 0, ids_uv = 0;
        float   w_fv = 0.0f, w_uv = 0.0f;
        CUDA_CHECK(cudaMemcpy(&ids_fv, ids_f_alloc.get() + idx, sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&w_fv, w_f_alloc.get() + idx, sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&ids_uv, (const int32_t *) ids->data + (int64_t) t * ids_stride + k, sizeof(int32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&w_uv, (const float *) weights->data + idx, sizeof(float), cudaMemcpyDeviceToHost));
        uint32_t w_fv_bits = 0, w_uv_bits = 0;
        memcpy(&w_fv_bits, &w_fv, sizeof(w_fv_bits));
        memcpy(&w_uv_bits, &w_uv, sizeof(w_uv_bits));
        GGML_LOG_WARN("%s: WP_DSV4_ROUTER_FUSE_DIFF mismatch layer=%s row=%d slot=%d "
                      "ids fused=%d unfused=%d w fused=%.9g (0x%08x) unfused=%.9g (0x%08x) "
                      "mismatches_this_batch=%llu mismatches_total=%llu compared_total=%llu\n",
                      __func__, logits->name, t, k, ids_fv, ids_uv, w_fv, w_fv_bits,
                      w_uv, w_uv_bits, host[0], total_mismatch, total_compared);
    } else if (logged < 32) {
        logged++;
        GGML_LOG_INFO("%s: WP_DSV4_ROUTER_FUSE_DIFF ok layer=%s rows=%d mismatches_total=%llu compared_total=%llu\n",
                      __func__, logits->name, n_tokens, total_mismatch, total_compared);
    }
}

// Production path: writes the ids view and the final weights tensor directly.
void ggml_cuda_op_topk_moe_dsv4_exact(ggml_backend_cuda_context & ctx, const ggml_tensor * logits,
                                      ggml_tensor * weights, ggml_tensor * ids, const ggml_tensor * clamp,
                                      const ggml_tensor * scale, const ggml_tensor * bias) {
    const int n_expert = (int) logits->ne[0];
    const int n_tokens = (int) logits->ne[1];
    const int n_used   = (int) weights->ne[1];
    const float clamp_min = ggml_get_op_params_f32(clamp, 0);
    const bool  do_scale  = scale != nullptr;
    const float scale_val = do_scale ? ggml_get_op_params_f32(scale, 0) : 1.0f;

    static bool logged_once = false;
    if (!logged_once) {
        logged_once = true;
        GGML_LOG_INFO("%s: WP_DSV4_ROUTER_FUSE active: first fused router tail at layer=%s n_expert=%d n_used=%d\n",
                      __func__, logits->name, n_expert, n_used);
    }

    dsv4_router_exact_run(ctx, (const float *) logits->data, (const float *) bias->data,
                          (int32_t *) ids->data, ids->nb[1] / sizeof(int32_t), (float *) weights->data,
                          logits->nb[1] / sizeof(float), n_tokens, n_expert, n_used, clamp_min, do_scale, scale_val);
}

// Structural checks for the exact path (beyond the generic matcher in ggml-cuda.cu).
bool ggml_cuda_dsv4_router_exact_match(const ggml_tensor * logits, const ggml_tensor * weights,
                                       const ggml_tensor * ids, const ggml_tensor * clamp,
                                       const ggml_tensor * scale, const ggml_tensor * bias) {
    (void) scale;
    if (logits->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
        bias == nullptr || bias->type != GGML_TYPE_F32 || clamp == nullptr || clamp->type != GGML_TYPE_F32) {
        return false;
    }
    if (logits->nb[0] != sizeof(float) || !ggml_is_contiguous(weights) || ids->nb[0] != sizeof(int32_t)) {
        return false;
    }
    if (weights->ne[0] != 1 || ids->ne[0] != weights->ne[1] || ids->ne[1] != logits->ne[1] ||
        weights->ne[2] != logits->ne[1]) {
        return false;
    }
    const int64_t n_expert = logits->ne[0];
    const int64_t n_used   = weights->ne[1];
    if (n_expert > 1024 || n_used > n_expert || n_used < 1 || bias->ne[0] != n_expert || !ggml_is_contiguous(bias)) {
        return false;
    }
    // ggml_clamp(x, min, +inf): the kernel hard-codes the +inf upper bound
    float clamp_max = 0.0f;
    memcpy(&clamp_max, (const float *) clamp->op_params + 1, sizeof(float));
    if (!(clamp_max == INFINITY)) {
        return false;
    }
    return true;
}
