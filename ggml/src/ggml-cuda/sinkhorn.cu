#include "sinkhorn.cuh"

// Fused Sinkhorn normalization for the DeepSeek-V4 hyper-connection mixer.
//
// WHY THIS EXISTS. The graph form in src/models/deepseek4.cpp emitted, per call,
// 39 sum_rows + 40 add + 39 div + 20 cont ~= 139 ggml nodes -- to normalize a
// [n, n, n_tokens] tensor that is SIXTEEN FLOATS per token at decode (n = 4),
// about 1300 FLOPs of real arithmetic. Measured on DS4-Flash: that cost 10.09s
// of 30.20s of total GPU kernel time and 2.2M of 4.3M dispatches over a
// 192-token decode, because a 4-6us launch latency was being paid per node.
// One thread per token does the whole thing in registers.
//
// NUMERICS MUST MATCH THE GRAPH FORM EXACTLY. DS4 expert routing is downstream
// of this result, and routing drives which experts get paged in. A subtly wrong
// normalization does not crash -- it silently changes expert selection and
// doubles physical reads (measured: 76.68 -> 150.37 GB when the iteration count
// was perturbed). So the operation order below mirrors the original op sequence
// literally rather than being re-derived or algebraically simplified:
//
//   A = softmax over ne0 (the dst index) for each src column
//   A += eps                                  <- eps added to the MATRIX, once
//   norm_cols()                               <- one unconditional pass
//   repeat (iters - 1) times: norm_rows(); norm_cols()
//
// where norm_cols divides each dst row by (sum over src + eps) and norm_rows
// divides each src column by (sum over dst + eps). NOTE eps is added to EVERY
// running sum as well as once to the matrix; both are load-bearing.
//
// Layout: src is contiguous [n, n, nt], element (dst=i, src=j, token=t) at
// t*n*n + j*n + i.

#define CUDA_SINKHORN_MAX_N 8

template <int n>
static __global__ void sinkhorn_norm_f32(
        const float * __restrict__ x,
        float       * __restrict__ y,
        const int64_t nt,
        const float   eps,
        const int     iters) {

    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) {
        return;
    }

    const float * xs = x + t*n*n;
    float       * ys = y + t*n*n;

    float m[n*n];

    // softmax along ne0, per src column j (max-subtracted, as ggml_soft_max does)
#pragma unroll
    for (int j = 0; j < n; ++j) {
        float mx = -INFINITY;
        for (int i = 0; i < n; ++i) {
            mx = fmaxf(mx, xs[j*n + i]);
        }
        float s = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float e = expf(xs[j*n + i] - mx);
            m[j*n + i] = e;
            s += e;
        }
        for (int i = 0; i < n; ++i) {
            m[j*n + i] /= s;
        }
    }

#pragma unroll
    for (int k = 0; k < n*n; ++k) {
        m[k] += eps;
    }

    // norm_cols: each dst row i divided by (sum over src j) + eps
#pragma unroll
    for (int i = 0; i < n; ++i) {
        float s = 0.0f;
        for (int j = 0; j < n; ++j) {
            s += m[j*n + i];
        }
        s += eps;
        for (int j = 0; j < n; ++j) {
            m[j*n + i] /= s;
        }
    }

    for (int it = 1; it < iters; ++it) {
        // norm_rows: each src column j divided by (sum over dst i) + eps
#pragma unroll
        for (int j = 0; j < n; ++j) {
            float s = 0.0f;
            for (int i = 0; i < n; ++i) {
                s += m[j*n + i];
            }
            s += eps;
            for (int i = 0; i < n; ++i) {
                m[j*n + i] /= s;
            }
        }
#pragma unroll
        for (int i = 0; i < n; ++i) {
            float s = 0.0f;
            for (int j = 0; j < n; ++j) {
                s += m[j*n + i];
            }
            s += eps;
            for (int j = 0; j < n; ++j) {
                m[j*n + i] /= s;
            }
        }
    }

#pragma unroll
    for (int k = 0; k < n*n; ++k) {
        ys[k] = m[k];
    }
}

static void sinkhorn_norm_f32_cuda(
        const float * x, float * y, const int64_t n, const int64_t nt,
        const float eps, const int iters, cudaStream_t stream) {

    const int64_t block = 128;
    const int64_t grid  = (nt + block - 1) / block;

    switch (n) {
        case 2: sinkhorn_norm_f32<2><<<grid, block, 0, stream>>>(x, y, nt, eps, iters); break;
        case 4: sinkhorn_norm_f32<4><<<grid, block, 0, stream>>>(x, y, nt, eps, iters); break;
        case 8: sinkhorn_norm_f32<8><<<grid, block, 0, stream>>>(x, y, nt, eps, iters); break;
        default: GGML_ABORT("sinkhorn_norm: unsupported n=%ld (expect 2, 4 or 8)", (long) n);
    }
}

void ggml_cuda_op_sinkhorn_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));
    GGML_ASSERT(src0->ne[0] == src0->ne[1]);
    GGML_ASSERT(src0->ne[0] <= CUDA_SINKHORN_MAX_N);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    const int iters = ggml_get_op_params_i32(dst, 1);
    GGML_ASSERT(eps >= 0.0f);
    GGML_ASSERT(iters >= 1);

    const int64_t n  = src0->ne[0];
    const int64_t nt = src0->ne[2] * src0->ne[3];

    sinkhorn_norm_f32_cuda((const float *) src0->data, (float *) dst->data,
                           n, nt, eps, iters, ctx.stream());
}
