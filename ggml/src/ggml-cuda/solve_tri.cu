#include "common.cuh"
#include "ggml.h"
#include "solve_tri.cuh"

#define MAX_N_FAST 64
#define MAX_K_FAST 32

static __global__ void get_batch_pointers(const float *  A,
                                          float *        X,
                                          const float ** A_ptrs,
                                          float **       X_ptrs,
                                          int64_t        ne02,
                                          int64_t        total_batches,
                                          size_t         s02,
                                          size_t         s03,
                                          size_t         s2,
                                          size_t         s3) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_batches) {
        return;
    }

    const int64_t i3 = idx / ne02;
    const int64_t i2 = idx % ne02;

    A_ptrs[idx] = A + i3 * s03 + i2 * s02;
    X_ptrs[idx] = X + i3 * s3 + i2 * s2;
}

static void solve_tri_f32_cublas(ggml_backend_cuda_context & ctx,
                                 const float *               A,
                                 const float *               B,
                                 float *                     X,
                                 int                         n,
                                 int                         k,
                                 int64_t                     ne02,
                                 int64_t                     ne03,
                                 size_t                      s02,
                                 size_t                      s03,
                                 size_t                      s12,
                                 size_t                      s13,
                                 size_t                      s2,
                                 size_t                      s3,
                                 cudaStream_t                stream) {
    const float   alpha         = 1.0f;
    const int64_t total_batches = ne02 * ne03;
    if (total_batches == 0) {
        return;
    }

    // Bulk copy B -> X (contiguous tensors)
    if (X != B) {
        const int64_t total_elements_BX = n * k * total_batches;
        CUDA_CHECK(cudaMemcpyAsync(X, B, total_elements_BX * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    }

    const int id = ggml_cuda_get_device();

    ggml_cuda_pool_alloc<const float *> A_ptrs_alloc(ctx.pool(id), total_batches);
    ggml_cuda_pool_alloc<float *>       X_ptrs_alloc(ctx.pool(id), total_batches);

    const float ** A_ptrs_dev = A_ptrs_alloc.get();
    float **       X_ptrs_dev = X_ptrs_alloc.get();

    get_batch_pointers<<<(total_batches + 255) / 256, 256, 0, stream>>>(A, X, A_ptrs_dev, X_ptrs_dev, ne02,
                                                                        total_batches, s02, s03, s2, s3);

    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));

    // Yes, this is necessary, without this we get RMSE errors
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_DEFAULT_MATH));
    CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(id), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                    CUBLAS_DIAG_NON_UNIT, k, n, &alpha, A_ptrs_dev, n, X_ptrs_dev, k, total_batches));

    // revert to standard mode from common.cuh
    CUBLAS_CHECK(cublasSetMathMode(ctx.cublas_handle(id), CUBLAS_TF32_TENSOR_OP_MATH));

    GGML_UNUSED_VARS(s12, s13);
}

// ======================
// Fast Kernel (n <= 64, ANY k) - Warp-based parallel reduction
// The k independent right-hand sides (x_i = b_i * A^-1) carry no cross-row
// dependency, so k is tiled across blockIdx.y in MAX_K_FAST-sized chunks.
// This keeps blockDim.x*blockDim.y within the 1024-thread limit, which was the
// ONLY reason k was previously capped at MAX_K_FAST.
// ======================
// When ncols_template == 0 the bounds for the loops in this function are not
// known and can't be unrolled. As we want to keep pragma unroll for all other
// cases we suppress the clang transformation warning here.
#ifdef __clang__
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wpass-failed"
#endif  // __clang__
template <int n_template, int k_template>
static __global__ void solve_tri_f32_fast(const float * __restrict__ A,
                                          const float * __restrict__ B,
                                          float * __restrict__ X,
                                          const uint3  ne02,
                                          const size_t nb02,
                                          const size_t nb03,
                                          const size_t nb12,
                                          const size_t nb13,
                                          const size_t nb2,
                                          const size_t nb3,
                                          const int    n_arg,
                                          const int    k_arg) {
    const int n = n_template == 0 ? n_arg : n_template;
    const int k = k_template == 0 ? k_arg : k_template;

    const int batch_idx = blockIdx.x;
    const int lane      = threadIdx.x;
    // k is tiled over blockIdx.y; the (col_idx >= k) guard below retires the ragged tail
    const int col_idx   = blockIdx.y * blockDim.y + threadIdx.y;

    // NOTE: do NOT early-return here. The k tile can be ragged (e.g. k=200 -> 7
    // tiles of 32 = 224), and every thread must still take part in the cooperative
    // sA load below and reach its __syncthreads(). Returning early left sA partially
    // uninitialised and diverged at the barrier, which produced inf results.
    const bool active = col_idx < k;

    const uint2   i02_i03 = fast_div_modulo(batch_idx, ne02);
    const int64_t i02     = i02_i03.y;
    const int64_t i03     = i02_i03.x;

    const float * const A_batch = (const float *) (A + i02 * nb02 + i03 * nb03);
    const float * const B_batch = (const float *) (B + i02 * nb12 + i03 * nb13);
    float *             X_batch = (float *) (X + i02 * nb2 + i03 * nb3);

    __shared__ float sA[MAX_N_FAST * MAX_N_FAST];

    const int offset = threadIdx.x + threadIdx.y * blockDim.x;

    // Number of columns handled by THIS block (== k when k <= MAX_K_FAST). Using the
    // template constant where available keeps the loop unrollable.
    const int ktile = (k_template == 0) ? (int) blockDim.y : k_template;

#pragma unroll
    for (int i = 0; i < n * n; i += ktile * WARP_SIZE) {
        const int i0 = i + offset;
        if (i0 < n * n) {
            sA[i0] = A_batch[i0];
        }
    }

    __syncthreads();

    float x_low  = (active && lane < n) ? B_batch[lane * k + col_idx] : 0.0f;
    float x_high = (active && WARP_SIZE + lane < n) ? B_batch[(WARP_SIZE + lane) * k + col_idx] : 0.0f;

    const int half      = WARP_SIZE;
    const int nrows_low = (n < half) ? n : half;

#pragma unroll
    for (int row = 0; row < nrows_low; ++row) {
        float sum = 0.0f;
        if (lane < row) {
            sum += sA[row * n + lane] * x_low;
        }
        sum = warp_reduce_sum(sum);

        if (lane == row) {
            x_low = (x_low - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int row = half; row < n; ++row) {
        float     sum = sA[row * n + lane] * x_low;
        const int j   = half + lane;
        if (j < row) {
            sum += sA[row * n + j] * x_high;
        }
        sum = warp_reduce_sum(sum);

        if (lane == row - half) {
            x_high = (x_high - sum) / sA[row * n + row];
        }
    }

#pragma unroll
    for (int rr = 0; rr < 2; ++rr) {
        const int row = rr * WARP_SIZE + lane;
        if (active && row < n) {
            const float val            = (row < half) ? x_low : x_high;
            X_batch[row * k + col_idx] = val;
        }
    }
}
#ifdef __clang__
#    pragma clang diagnostic pop
#endif  // __clang__

static void solve_tri_f32_cuda(const float * A,
                               const float * B,
                               float *       X,
                               int           n,
                               int           k,
                               int64_t       ne02,
                               int64_t       ne03,
                               size_t        nb02,
                               size_t        nb03,
                               size_t        nb12,
                               size_t        nb13,
                               size_t        nb2,
                               size_t        nb3,
                               cudaStream_t  stream) {
    const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
    // Tile k so blockDim.x*blockDim.y stays <= 1024; grid.y covers the remaining columns.
    const int   k_tile  = k < MAX_K_FAST ? k : MAX_K_FAST;
    const int   k_blocks = (k + k_tile - 1) / k_tile;
    dim3        threads(WARP_SIZE, k_tile);
    dim3        grid(ne02 * ne03, k_blocks);
    if (n == 64) {
        switch (k) {
            case 32:
                solve_tri_f32_fast<64, 32>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 16:
                solve_tri_f32_fast<64, 16>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 14:
                solve_tri_f32_fast<64, 14>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 12:
                solve_tri_f32_fast<64, 12>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 10:
                solve_tri_f32_fast<64, 10>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 8:
                solve_tri_f32_fast<64, 8>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 6:
                solve_tri_f32_fast<64, 6>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 4:
                solve_tri_f32_fast<64, 4>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 2:
                solve_tri_f32_fast<64, 2>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            case 1:
                solve_tri_f32_fast<64, 1>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, 0, 0);
                break;
            default:
                solve_tri_f32_fast<0, 0>
                    <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
        }
    } else {  // run general case
        solve_tri_f32_fast<0, 0>
            <<<grid, threads, 0, stream>>>(A, B, X, ne02_fd, nb02, nb03, nb12, nb13, nb2, nb3, n, k);
    }
}

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // A (n×n, lower triangular)
    const ggml_tensor * src1 = dst->src[1];  // B (n×k)

    ggml_is_contiguous(src0);
    ggml_is_contiguous(src1);

    const int64_t n    = src0->ne[0];
    const int64_t k    = src1->ne[0];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    // NOTE: only n is limited now -- k is tiled inside solve_tri_f32_cuda. This also
    // keeps us off cuBLAS/hipBLAS Strsm, whose rocBLAS batched path segfaults on
    // gfx1030 (RX 6900 XT) for the n=k=64 shape every Gated DeltaNet layer produces.
    if (n <= MAX_N_FAST) {
        solve_tri_f32_cuda((const float *) src0->data, (const float *) src1->data, (float *) dst->data, n, k,
                           src0->ne[2], src0->ne[3], src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                           src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), dst->nb[2] / sizeof(float),
                           dst->nb[3] / sizeof(float), ctx.stream());
    } else {
        solve_tri_f32_cublas(ctx, (const float *) src0->data, (const float *) src1->data, (float *) dst->data, n, k,
                             ne02, ne03, src0->nb[2] / sizeof(float), src0->nb[3] / sizeof(float),
                             src1->nb[2] / sizeof(float), src1->nb[3] / sizeof(float), dst->nb[2] / sizeof(float),
                             dst->nb[3] / sizeof(float), ctx.stream());
    }
}
