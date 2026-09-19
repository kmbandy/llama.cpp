#include "norm.cuh"
#include <cstdint>

template <int block_size>
static __global__ void norm_f32(
        const float * x, float * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float2 mean_var = make_float2(0.0f, 0.0f);

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float2 s_sum2[];
    mean_var = block_reduce<block_reduce_method::SUM, block_size>(mean_var, s_sum2);

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (x[col] - mean) * inv_std;
    }
}

template <int block_size>
static __global__ void group_norm_f32(const float * x, float * dst, const int group_size, const int ne_elements, const float eps) {
    // blockIdx.x: num_groups idx
    // threadIdx.x: block_size idx
    const int start =     blockIdx.x*group_size + threadIdx.x;
    const int end   = min(blockIdx.x*group_size + group_size,  ne_elements);

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        const float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum + 32);

    const float variance = tmp / group_size;
    const float scale = rsqrtf(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// BF16 activation coverage (2026-09-18)
// src_dst_t is the type of x/dst (and, when mul_add_t == src_dst_t, of mul/add too);
// mul_add_t is the type of the mul/add operands, which may independently be f32 or bf16
// (e.g. an f32 RMS-norm weight applied to a bf16 activation stream). Internal math is fp32.
template <int block_size, typename src_dst_t, typename mul_add_t = src_dst_t, bool do_multiply = false, bool do_add = false>
static __global__ void rms_norm_f32(const src_dst_t * x,
                                    src_dst_t *       dst,
                                    const int     ncols,
                                    const int64_t stride_row,
                                    const int64_t stride_channel,
                                    const int64_t stride_sample,
                                    const float   eps,
                                    const mul_add_t * mul               = nullptr,
                                    const int64_t mul_stride_row       = 0,
                                    const int64_t mul_stride_channel   = 0,
                                    const int64_t mul_stride_sample    = 0,
                                    const uint3   mul_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   mul_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   mul_nsamples_packed  = make_uint3(0, 0, 0),
                                    const mul_add_t * add               = nullptr,
                                    const int64_t add_stride_row       = 0,
                                    const int64_t add_stride_channel   = 0,
                                    const int64_t add_stride_sample    = 0,
                                    const uint3   add_ncols_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nrows_packed     = make_uint3(0, 0, 0),
                                    const uint3   add_nchannels_packed = make_uint3(0, 0, 0),
                                    const uint3   add_nsamples_packed  = make_uint3(0, 0, 0),
                                    const bool    wide_sumsq            = false) {
    ggml_cuda_pdl_lc();
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const uint32_t mul_row     = fastmodulo(row, mul_nrows_packed);
        const uint32_t mul_channel = fastmodulo(channel, mul_nchannels_packed);
        const uint32_t mul_sample  = fastmodulo(sample, mul_nsamples_packed);
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        const int add_row     = fastmodulo(row, add_nrows_packed);
        const int add_channel = fastmodulo(channel, add_nchannels_packed);
        const int add_sample  = fastmodulo(sample, add_nsamples_packed);
        add += add_sample * add_stride_sample + add_channel * add_stride_channel + add_row * add_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();

    // MT_WIDE_KERNELS: at the production shape (ncols=5120, block_size=1024)
    // the scalar loop below is a 5-iteration SEQUENTIAL grid-stride chain per
    // thread (load -> accumulate -> load -> ...), exactly the pattern the
    // gfx1201 microbench found ~3.5x slower per-wave than the same work done
    // with <=2 independent loads per thread. Fixed-order reduction over
    // ncols is unavoidable (one block per row), but each "iteration" can
    // carry VEC elements via one 128-bit load instead of 1: VEC=16/sizeof(T)
    // (4 for f32, 8 for f16/bf16) cuts iterations to
    // ceil(ncols/(block_size*VEC)) = ceil(5120/4096) = 2 for f32 at this
    // shape. The 16B load is issued once per iteration and its VEC scalars
    // are then summed independently (no re-load between them) -- "issued
    // before use". Requires ncols and the row stride both VEC-aligned (true
    // for the contiguous hidden-dim rows this fires on); falls back to the
    // scalar loop otherwise. Changes the thread-to-column assignment (and
    // hence the FP summation order) vs. the scalar path -- within rms_norm's
    // existing test tolerance, not required to be bit-exact.
    constexpr int vec_elems = 16 / (int) sizeof(src_dst_t);
    if (wide_sumsq && vec_elems > 1 && (ncols % vec_elems) == 0 &&
        (stride_row % vec_elems) == 0 && (((uintptr_t) x) % 16) == 0) {
        const int ncols_vec = ncols / vec_elems;
        for (int cv = tid; cv < ncols_vec; cv += block_size) {
            const uint4 raw = *reinterpret_cast<const uint4 *>(x + (size_t) cv * vec_elems);
            const src_dst_t * xs = reinterpret_cast<const src_dst_t *>(&raw);
            #pragma unroll
            for (int j = 0; j < vec_elems; ++j) {
                const float xi = (float) xs[j];
                tmp += xi * xi;
            }
        }
    } else {
        for (int col = tid; col < ncols; col += block_size) {
            const float xi = (float) x[col];
            tmp += xi * xi;
        }
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply && do_add) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            const int add_col = fastmodulo(col, add_ncols_packed);
            dst[col]          = (src_dst_t) (scale * (float) x[col] * (float) mul[mul_col] + (float) add[add_col]);
        } else if constexpr (do_multiply) {
            const int mul_col = fastmodulo(col, mul_ncols_packed);
            dst[col]          = (src_dst_t) (scale * (float) x[col] * (float) mul[mul_col]);
        } else {
            dst[col] = (src_dst_t) (scale * (float) x[col]);
        }
    }
}

// ── MT_WIDE_KERNELS two-pass wide rms_norm (2026-09-18) ─────────────────────
//
// The wide_sumsq vectorized-load path above measured NO effect (105 -> 103 ms
// over 1161 launches at the production 1024x5120 shape): it cuts iterations
// per thread (~5 -> ~2, via 128-bit loads) but does NOT change the grid --
// still exactly one block per row (1024 blocks at this shape). The gfx1201
// microbench behind this task is unambiguous that block COUNT, not just
// per-thread iteration count, is what gates throughput here: a kernel
// reading 20 MB with ~8 sequential iterations/thread at 1024 blocks x 256
// threads runs 2.3-2.8x slower on the RX 9070 XT than the R9700, while the
// same bytes at 8192 blocks x 256 threads x ~1 iteration/thread run at
// parity; read-only reproduces the slowdown, write-only does not. So only
// the sum-of-squares READ needs more blocks; the scale+write pass (which the
// microbench says is not the culprit) can stay a plain flat vectorized copy.
//
// Two passes:
//   pass 1 (rms_norm_f32_wide_pass1_sumsq): grid (nrows, RMS_NORM_WIDE_CHUNKS)
//     -- RMS_NORM_WIDE_CHUNKS=8 blocks per row, 256 threads/block, each doing
//     <=2 128-bit loads (chosen so nrows=1024 lands exactly on the
//     microbench's 8192-block/~1-iteration-per-thread sweet spot). Writes one
//     partial sum-of-squares per (row, chunk) into a small ctx-pool scratch
//     buffer (nrows*RMS_NORM_WIDE_CHUNKS floats).
//   pass 2 (rms_norm_f32_wide_pass2_scale): a single flat 1-D grid over every
//     output float4 (grid = total_float4s / 256) -- each thread locates its
//     row, sums that row's RMS_NORM_WIDE_CHUNKS partials (a handful of floats,
//     cheap relative to the 128-bit x load it's about to do anyway),
//     computes rsqrt(mean+eps), and does one 128-bit load + optional
//     column-broadcast mul + one 128-bit store.
//
// Only wired for do_add==false (mul is optional, add is not) -- the report's
// regression is specifically in the fused-mul instantiation
// (rms_norm_f32<1024, ..., /*do_multiply=*/true, /*do_add=*/false>); the
// fused mul+add path is untouched and keeps the original single-pass kernel.
// Dispatch gating (rms_norm_wide_two_pass_eligible, below) requires
// MT_WIDE_KERNELS=1, nchannels==nsamples==1 (keeps pass 2's row/dst indexing
// exact -- see its "dst is always contiguous" note), ncols*nrows >= 1<<20
// (so decode's 1-row shape keeps the old kernel: nothing to widen when a
// single row already gets its own dedicated block), and the same 128-bit
// alignment/divisibility preconditions the existing wide_sumsq flag already
// required (ncols % vec_elems == 0, stride_row % vec_elems == 0, x 16B-aligned).
constexpr int RMS_NORM_WIDE_CHUNKS = 8;

template <typename src_dst_t>
static __global__ void rms_norm_f32_wide_pass1_sumsq(
        const src_dst_t * __restrict__ x, float * __restrict__ partial,
        const int ncols, const int64_t stride_row, const int chunks_per_row) {
    constexpr int block_size = 256;
    constexpr int vec_elems  = 16 / (int) sizeof(src_dst_t);
    const int row   = blockIdx.x;
    const int chunk = blockIdx.y;
    const int tid    = threadIdx.x;

    const src_dst_t * xr = x + (size_t) row * stride_row;
    const int ncols_vec      = ncols / vec_elems;
    const int chunk_vec_size = (ncols_vec + chunks_per_row - 1) / chunks_per_row;
    const int vec_begin      = chunk * chunk_vec_size;
    const int vec_end        = min(vec_begin + chunk_vec_size, ncols_vec);

    float tmp = 0.0f;
    for (int cv = vec_begin + tid; cv < vec_end; cv += block_size) {
        const uint4 raw = *reinterpret_cast<const uint4 *>(xr + (size_t) cv * vec_elems);
        const src_dst_t * xs = reinterpret_cast<const src_dst_t *>(&raw);
        #pragma unroll
        for (int j = 0; j < vec_elems; ++j) {
            const float xi = (float) xs[j];
            tmp += xi * xi;
        }
    }

    extern __shared__ float s_sum_wide[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum_wide);
    if (tid == 0) {
        partial[(size_t) row * chunks_per_row + chunk] = tmp;
    }
}

template <typename src_dst_t, typename mul_add_t, bool do_multiply, bool do_add = false>
static __global__ void rms_norm_f32_wide_pass2_scale(
        const src_dst_t * __restrict__ x, src_dst_t * __restrict__ dst,
        const mul_add_t * __restrict__ mul, const mul_add_t * __restrict__ add,
        const float * __restrict__ partial,
        const int ncols, const int nrows, const int64_t stride_row, const int chunks_per_row,
        const float eps, const int64_t mul_stride_row,
        const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const int64_t add_stride_row, const uint3 add_ncols_packed, const uint3 add_nrows_packed) {
    constexpr int vec_elems = 16 / (int) sizeof(src_dst_t);
    const int    ncols_vec  = ncols / vec_elems;
    const size_t total_vec  = (size_t) nrows * (size_t) ncols_vec;

    for (size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x; i < total_vec;
         i += (size_t) gridDim.x * blockDim.x) {
        const int row = (int) (i / ncols_vec);
        const int cv  = (int) (i % ncols_vec);

        // A handful of floats (RMS_NORM_WIDE_CHUNKS, typically 8) -- cheap
        // next to the 128-bit x load below, and L2-resident after pass 1.
        float sumsq = 0.0f;
        const float * prow = partial + (size_t) row * chunks_per_row;
        #pragma unroll 8
        for (int c = 0; c < chunks_per_row; ++c) {
            sumsq += prow[c];
        }
        const float mean  = sumsq / ncols;
        const float scale = rsqrtf(mean + eps);

        const src_dst_t * xr  = x + (size_t) row * stride_row;
        const uint4       raw = *reinterpret_cast<const uint4 *>(xr + (size_t) cv * vec_elems);
        const src_dst_t * xs  = reinterpret_cast<const src_dst_t *>(&raw);

        uint4       out;
        src_dst_t * outs = reinterpret_cast<src_dst_t *>(&out);

        const mul_add_t * mul_row_ptr = nullptr;
        if constexpr (do_multiply) {
            const uint32_t mul_row = fastmodulo((uint32_t) row, mul_nrows_packed);
            mul_row_ptr = mul + (size_t) mul_row * mul_stride_row;
        }
        const mul_add_t * add_row_ptr = nullptr;
        if constexpr (do_add) {
            const uint32_t add_row = fastmodulo((uint32_t) row, add_nrows_packed);
            add_row_ptr = add + (size_t) add_row * add_stride_row;
        }

        #pragma unroll
        for (int j = 0; j < vec_elems; ++j) {
            float v = scale * (float) xs[j];
            const int col = cv * vec_elems + j;
            if constexpr (do_multiply) {
                const uint32_t mul_col = fastmodulo((uint32_t) col, mul_ncols_packed);
                v *= (float) mul_row_ptr[mul_col];
            }
            if constexpr (do_add) {
                const uint32_t add_col = fastmodulo((uint32_t) col, add_ncols_packed);
                v += (float) add_row_ptr[add_col];
            }
            outs[j] = (src_dst_t) v;
        }

        // dst is always laid out contiguous-per-row here (see the callers'
        // "((sample*nchannels + channel)*nrows + row)*ncols" offset in the
        // single-pass kernel above, which -- with the nchannels==nsamples==1
        // gate this path requires -- reduces to exactly row*ncols).
        src_dst_t * dstr = dst + (size_t) row * ncols;
        *reinterpret_cast<uint4 *>(dstr + (size_t) cv * vec_elems) = out;
    }
}

// Gate for the two-pass path: everything that must hold for pass 1/2's
// addressing and alignment assumptions above to be valid.
template <typename src_dst_t>
static bool rms_norm_wide_two_pass_eligible(
        const int ncols, const int64_t nrows, const int64_t nchannels, const int64_t nsamples,
        const int64_t stride_row, const void * x_ptr) {
    if (!ggml_cuda_mt_wide_kernels_enabled()) {
        return false;
    }
    if (nchannels != 1 || nsamples != 1) {
        return false;
    }
    // Decode (nrows 1-8) keeps the single-pass kernel. Prefill at SPLIT=2 is
    // 1024 rows; the 1M-element gate used to exclude 1024x256 head RMS
    // (262k elements) which is exactly the 9070 1024-block occupancy floor.
    if (nrows < 256 || ncols < 128) {
        return false;
    }
    constexpr int vec_elems = 16 / (int) sizeof(src_dst_t);
    if (vec_elems <= 1) {
        return false;
    }
    if ((ncols % vec_elems) != 0 || (stride_row % vec_elems) != 0) {
        return false;
    }
    if ((((uintptr_t) x_ptr) % 16) != 0) {
        return false;
    }
    return true;
}

template <typename src_dst_t, typename mul_add_t, bool do_multiply, bool do_add = false>
static void rms_norm_f32_wide_dispatch(
        ggml_backend_cuda_context & ctx,
        const src_dst_t * x, src_dst_t * dst, const mul_add_t * mul, const mul_add_t * add,
        const int ncols, const int nrows, const int64_t stride_row,
        const int64_t mul_stride_row, const uint3 mul_ncols_packed, const uint3 mul_nrows_packed,
        const int64_t add_stride_row, const uint3 add_ncols_packed, const uint3 add_nrows_packed,
        const float eps, cudaStream_t stream) {
    static_assert(!do_add || do_multiply, "fusing add is not supported without multiplying");
    constexpr int chunks_per_row = RMS_NORM_WIDE_CHUNKS;
    constexpr int pass1_block    = 256;
    constexpr int pass2_block    = 256;
    constexpr int vec_elems      = 16 / (int) sizeof(src_dst_t);

    ggml_cuda_pool_alloc<float> partial(ctx.pool(), (size_t) nrows * (size_t) chunks_per_row);

    const dim3 grid1(nrows, chunks_per_row, 1);
    rms_norm_f32_wide_pass1_sumsq<src_dst_t><<<grid1, pass1_block, 32 * sizeof(float), stream>>>(
        x, partial.get(), ncols, stride_row, chunks_per_row);

    const size_t total_vec = (size_t) nrows * (size_t) (ncols / vec_elems);
    const size_t grid2_sz  = (total_vec + pass2_block - 1) / pass2_block;
    // 9070 quiet-link: <~1024 blocks sit on an ~80 us floor; 4096 reaches
    // R9700 parity on the same bytes (mall.hip). Idle extra blocks are cheap
    // here (no fat LDS).
    int grid2 = (int) (grid2_sz > (size_t) INT32_MAX ? (size_t) INT32_MAX : grid2_sz);
    if (nrows >= 256 && grid2 < 4096) {
        grid2 = 4096;
    }

    rms_norm_f32_wide_pass2_scale<src_dst_t, mul_add_t, do_multiply, do_add><<<grid2, pass2_block, 0, stream>>>(
        x, dst, mul, add, partial.get(), ncols, nrows, stride_row, chunks_per_row, eps,
        mul_stride_row, mul_ncols_packed, mul_nrows_packed,
        add_stride_row, add_ncols_packed, add_nrows_packed);
}

template <int block_size>
static __global__ void rms_norm_back_f32(
        const float * grad, const float * xf, float * dst, const int ncols, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    grad += int64_t(row)*ncols;
    xf   += int64_t(row)*ncols;
    dst  += int64_t(row)*ncols;

    float sum_xx = 0.0f; // sum for squares of x, equivalent to forward pass
    float sum_xg = 0.0f; // sum for x * gradient, needed because RMS norm mixes inputs

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xfi = xf[col];
        sum_xx += xfi * xfi;
        sum_xg += xfi * grad[col];
    }

    // sum up partial sums
    sum_xx = warp_reduce_sum(sum_xx);
    sum_xg = warp_reduce_sum(sum_xg);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum_xx[32];
        __shared__ float s_sum_xg[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum_xx[warp_id] = sum_xx;
            s_sum_xg[warp_id] = sum_xg;
        }
        __syncthreads();

        sum_xx = s_sum_xx[lane_id];
        sum_xx = warp_reduce_sum(sum_xx);

        sum_xg = s_sum_xg[lane_id];
        sum_xg = warp_reduce_sum(sum_xg);
    }

    const float mean_eps = sum_xx / ncols + eps;
    const float sum_eps  = sum_xx + ncols*eps;

    const float scale_grad = rsqrtf(mean_eps);
    const float scale_x    = -scale_grad * sum_xg/sum_eps;

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = scale_grad*grad[col] + scale_x*xf[col];
    }
}

// template <int block_size>
// static __global__ void l2_norm_f32(const float * x, float * dst, const int ncols, const float eps) {
//     const int row = blockIdx.x*blockDim.y + threadIdx.y;
//     const int tid = threadIdx.x;

//     float tmp = 0.0f; // partial sum for thread in warp

//     for (int col = tid; col < ncols; col += block_size) {
//         const float xi = x[row*ncols + col];
//         tmp += xi * xi;
//     }

//     // sum up partial sums
//     tmp = warp_reduce_sum(tmp);
//     if (block_size > WARP_SIZE) {
//         __shared__ float s_sum[32];
//         int warp_id = threadIdx.x / WARP_SIZE;
//         int lane_id = threadIdx.x % WARP_SIZE;
//         if (lane_id == 0) {
//             s_sum[warp_id] = tmp;
//         }
//         __syncthreads();
//         tmp = s_sum[lane_id];
//         tmp = warp_reduce_sum(tmp);
//     }

//     // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
//     const float scale = rsqrtf(fmaxf(tmp, eps * eps));

//     for (int col = tid; col < ncols; col += block_size) {
//         dst[row*ncols + col] = scale * x[row*ncols + col];
//     }
// }

// BF16 activation coverage (2026-09-18)
template <int block_size, typename src_dst_t>
static __global__ void l2_norm_f32(
        const src_dst_t * x, src_dst_t * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    float tmp = 0.0f; // partial sum for thread in warp

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = (float) x[col];
        tmp += xi * xi;
    }

    // sum up partial sums
    extern __shared__ float s_sum[];
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    // from https://pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html
    const float scale = rsqrtf(fmaxf(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col] = (src_dst_t) (scale * (float) x[col]);
    }
}

static void norm_f32_cuda(
        const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        norm_f32<WARP_SIZE><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        norm_f32<1024><<<blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float2): 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void group_norm_f32_cuda(
        const float * x, float * dst, const int num_groups, const float eps, const int group_size, const int ne_elements, cudaStream_t stream) {
    if (group_size < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        group_norm_f32<WARP_SIZE><<<num_groups, block_dims, 0, stream>>>(x, dst, group_size, ne_elements, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        group_norm_f32<1024><<<num_groups, block_dims, block_dims.x > WARP_SIZE ? 2 * 32 * sizeof(float): 0, stream>>>(x, dst, group_size, ne_elements, eps);
    }
}

// BF16 activation coverage (2026-09-18)
template <typename src_dst_t>
static void rms_norm_f32_cuda(
        ggml_backend_cuda_context & ctx,
        const src_dst_t * x, src_dst_t * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (rms_norm_wide_two_pass_eligible<src_dst_t>(ncols, nrows, nchannels, nsamples, stride_row, x)) {
        rms_norm_f32_wide_dispatch<src_dst_t, src_dst_t, false>(
            ctx, x, dst, /*mul=*/(const src_dst_t *) nullptr, /*add=*/(const src_dst_t *) nullptr,
            ncols, nrows, stride_row,
            /*mul_stride_row=*/0, make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            /*add_stride_row=*/0, make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            eps, stream);
    } else if (ncols < 1024) {
        const dim3 block_dims(256, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = {blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<256, src_dst_t, src_dst_t, false>, launch_params,
            x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        (const src_dst_t *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        (const src_dst_t *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        ggml_cuda_mt_wide_kernels_enabled());
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(rms_norm_f32<1024, src_dst_t, src_dst_t, false>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps,
        // underlying cudaLaunchKernelEx does not support default params
        (const src_dst_t *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        (const src_dst_t *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
        ggml_cuda_mt_wide_kernels_enabled());
    }
}

// BF16 activation coverage (2026-09-18)
template <typename src_dst_t, typename mul_add_t>
static void rms_norm_mul_f32_cuda(ggml_backend_cuda_context & ctx,
                                  const src_dst_t *  x,
                                  const mul_add_t *  mul,
                                  const mul_add_t *  add,
                                  src_dst_t *        dst,
                                  const int      ncols,
                                  const int      nrows,
                                  const int      nchannels,
                                  const int      nsamples,
                                  const int64_t  stride_row,
                                  const int64_t  stride_channel,
                                  const int64_t  stride_sample,
                                  const int64_t  mul_stride_row,
                                  const int64_t  mul_stride_channel,
                                  const int64_t  mul_stride_sample,
                                  const uint32_t mul_ncols,
                                  const uint32_t mul_nrows,
                                  const uint32_t mul_nchannels,
                                  const uint32_t mul_nsamples,
                                  const int64_t  add_stride_row,
                                  const int64_t  add_stride_channel,
                                  const int64_t  add_stride_sample,
                                  const uint32_t add_ncols,
                                  const uint32_t add_nrows,
                                  const uint32_t add_nchannels,
                                  const uint32_t add_nsamples,
                                  const float    eps,
                                  cudaStream_t   stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f32_cuda<src_dst_t>(ctx, x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (add == nullptr) {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);
        if (mul_stride_channel == 0 && mul_stride_sample == 0 &&
                   rms_norm_wide_two_pass_eligible<src_dst_t>(ncols, nrows, nchannels, nsamples, stride_row, x)) {
            rms_norm_f32_wide_dispatch<src_dst_t, mul_add_t, true>(
                ctx, x, dst, mul, /*add=*/(const mul_add_t *) nullptr, ncols, nrows, stride_row,
                mul_stride_row, mul_ncols_packed, mul_nrows_packed,
                /*add_stride_row=*/0, make_uint3(0, 0, 0), make_uint3(0, 0, 0),
                eps, stream);
        } else if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, src_dst_t, mul_add_t, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            (const mul_add_t *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            ggml_cuda_mt_wide_kernels_enabled());
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, src_dst_t, mul_add_t, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed,
                // underlying cudaLaunchKernelEx does not support default params
            (const mul_add_t *) nullptr, 0, 0, 0, make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0), make_uint3(0, 0, 0),
            ggml_cuda_mt_wide_kernels_enabled());
        }
    } else {
        const uint3 mul_ncols_packed     = init_fastdiv_values(mul_ncols);
        const uint3 mul_nrows_packed     = init_fastdiv_values(mul_nrows);
        const uint3 mul_nchannels_packed = init_fastdiv_values(mul_nchannels);
        const uint3 mul_nsamples_packed  = init_fastdiv_values(mul_nsamples);

        const uint3 add_ncols_packed     = init_fastdiv_values(add_ncols);
        const uint3 add_nrows_packed     = init_fastdiv_values(add_nrows);
        const uint3 add_nchannels_packed = init_fastdiv_values(add_nchannels);
        const uint3 add_nsamples_packed  = init_fastdiv_values(add_nsamples);
        if (mul_stride_channel == 0 && mul_stride_sample == 0 &&
            add_stride_channel == 0 && add_stride_sample == 0 &&
            rms_norm_wide_two_pass_eligible<src_dst_t>(ncols, nrows, nchannels, nsamples, stride_row, x)) {
            rms_norm_f32_wide_dispatch<src_dst_t, mul_add_t, true, true>(
                ctx, x, dst, mul, add, ncols, nrows, stride_row,
                mul_stride_row, mul_ncols_packed, mul_nrows_packed,
                add_stride_row, add_ncols_packed, add_nrows_packed,
                eps, stream);
        } else if (ncols < 1024) {
            const dim3 block_dims(256, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims,block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<256, src_dst_t, mul_add_t, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, ggml_cuda_mt_wide_kernels_enabled());
        } else {
            const dim3 block_dims(1024, 1, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
            ggml_cuda_kernel_launch(rms_norm_f32<1024, src_dst_t, mul_add_t, true, true>, launch_params,
                x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel,
                mul_stride_sample, mul_ncols_packed, mul_nrows_packed, mul_nchannels_packed, mul_nsamples_packed, add,
                add_stride_row, add_stride_channel, add_stride_sample, add_ncols_packed, add_nrows_packed,
                add_nchannels_packed, add_nsamples_packed, ggml_cuda_mt_wide_kernels_enabled());
        }
    }
}

static void rms_norm_back_f32_cuda(const float * grad, const float * xf, float * dst, const int ncols, const int nrows, const float eps, cudaStream_t stream) {
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_back_f32<WARP_SIZE><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_back_f32<1024><<<nrows, block_dims, 0, stream>>>(grad, xf, dst, ncols, eps);
    }
}

// BF16 activation coverage (2026-09-18)
template <typename src_dst_t>
static void l2_norm_f32_cuda(
        const src_dst_t * x, src_dst_t * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<WARP_SIZE, src_dst_t>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{blocks_num, block_dims, block_dims.x > WARP_SIZE ? 32 * sizeof(float): 0, stream};
        ggml_cuda_kernel_launch(l2_norm_f32<1024, src_dst_t>, launch_params, x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    norm_f32_cuda(src0_d, dst_d, ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
}

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    int group_size = src0->ne[0] * src0->ne[1] * ((src0->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_cuda(src0_d, dst_d, num_groups * src0->ne[3], eps, group_size, ggml_nelements(src0), stream);
}

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // BF16 activation coverage (2026-09-18)
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    if (src0->type == GGML_TYPE_BF16) {
        rms_norm_f32_cuda<nv_bfloat16>(ctx, (const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data,
            ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        rms_norm_f32_cuda<float>(ctx, (const float *) src0->data, (float *) dst->data,
            ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float eps = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const void * src0_d = rms_norm_src->data;
    const void * mul_d = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d = mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if(mul_tensor->src[1] == dst) {
        mul_d = mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    void * dst_d = mul_tensor->data;
    cudaStream_t stream = ctx.stream();

    // BF16 activation coverage (2026-09-18): src/dst may be f32 or bf16; the mul weight may
    // independently be f32 or bf16.
    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32 || rms_norm_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == rms_norm_src->type);
    GGML_ASSERT(mul_tensor->type == rms_norm_src->type);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32 || mul_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    if (rms_norm_src->type == GGML_TYPE_BF16 && mul_src->type == GGML_TYPE_BF16) {
        rms_norm_mul_f32_cuda<nv_bfloat16, nv_bfloat16>(ctx, (const nv_bfloat16 *) src0_d, (const nv_bfloat16 *) mul_d, nullptr, (nv_bfloat16 *) dst_d,
            ne00, ne01, ne02, ne03, s01, s02, s03, mul_s01, mul_s02, mul_s03,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples, 0, 0, 0, 0, 0, 0, 0, eps, stream);
    } else if (rms_norm_src->type == GGML_TYPE_BF16 && mul_src->type == GGML_TYPE_F32) {
        rms_norm_mul_f32_cuda<nv_bfloat16, float>(ctx, (const nv_bfloat16 *) src0_d, (const float *) mul_d, nullptr, (nv_bfloat16 *) dst_d,
            ne00, ne01, ne02, ne03, s01, s02, s03, mul_s01, mul_s02, mul_s03,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples, 0, 0, 0, 0, 0, 0, 0, eps, stream);
    } else {
        rms_norm_mul_f32_cuda<float, float>(ctx, (const float *) src0_d, (const float *) mul_d, nullptr, (float *) dst_d,
            ne00, ne01, ne02, ne03, s01, s02, s03, mul_s01, mul_s02, mul_s03,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples, 0, 0, 0, 0, 0, 0, 0, eps, stream);
    }
}

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor) {
    const ggml_tensor * rms_norm_src = (ggml_tensor *) dst->src[0];
    float               eps          = 0.0f;

    memcpy(&eps, dst->op_params, sizeof(float));

    const void *        src0_d  = rms_norm_src->data;
    const void *        mul_d   = nullptr;
    const ggml_tensor * mul_src = nullptr;

    if (mul_tensor->src[0] == dst) {
        mul_d   = mul_tensor->src[1]->data;
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_d   = mul_tensor->src[0]->data;
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    const void *        add_d   = nullptr;
    const ggml_tensor * add_src = nullptr;

    if (add_tensor->src[0] == mul_tensor) {
        add_d   = add_tensor->src[1]->data;
        add_src = add_tensor->src[1];
    } else if (add_tensor->src[1] == mul_tensor) {
        add_d   = add_tensor->src[0]->data;
        add_src = add_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }

    void *       dst_d  = add_tensor->data;
    cudaStream_t stream = ctx.stream();

    // BF16 activation coverage (2026-09-18): src/dst may be f32 or bf16; mul and add
    // operands may independently be f32 or bf16 (but must match each other, since the
    // kernel takes a single mul_add_t type parameter).
    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32 || rms_norm_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == rms_norm_src->type);
    GGML_ASSERT(mul_tensor->type == rms_norm_src->type);
    GGML_ASSERT(add_tensor->type == rms_norm_src->type);
    GGML_ASSERT(mul_src->type == GGML_TYPE_F32 || mul_src->type == GGML_TYPE_BF16);
    GGML_ASSERT(add_src->type == mul_src->type);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;

    const int mul_ncols     = mul_src->ne[0];
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    const size_t ts_add = ggml_type_size(add_src->type);
    GGML_ASSERT(add_src->nb[0] == ts_add);
    const int64_t add_s01 = add_src->nb[1] / ts_add;
    const int64_t add_s02 = add_src->nb[2] / ts_add;
    const int64_t add_s03 = add_src->nb[3] / ts_add;

    const int add_ncols     = add_src->ne[0];
    const int add_nrows     = add_src->ne[1];
    const int add_nchannels = add_src->ne[2];
    const int add_nsamples  = add_src->ne[3];

    if (rms_norm_src->type == GGML_TYPE_BF16 && mul_src->type == GGML_TYPE_BF16) {
        rms_norm_mul_f32_cuda<nv_bfloat16, nv_bfloat16>(ctx, (const nv_bfloat16 *) src0_d, (const nv_bfloat16 *) mul_d, (const nv_bfloat16 *) add_d, (nv_bfloat16 *) dst_d,
            ne00, ne01, ne02, ne03, s01, s02, s03, mul_s01, mul_s02, mul_s03,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
            add_s01, add_s02, add_s03, add_ncols, add_nrows, add_nchannels, add_nsamples, eps, stream);
    } else if (rms_norm_src->type == GGML_TYPE_BF16 && mul_src->type == GGML_TYPE_F32) {
        rms_norm_mul_f32_cuda<nv_bfloat16, float>(ctx, (const nv_bfloat16 *) src0_d, (const float *) mul_d, (const float *) add_d, (nv_bfloat16 *) dst_d,
            ne00, ne01, ne02, ne03, s01, s02, s03, mul_s01, mul_s02, mul_s03,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
            add_s01, add_s02, add_s03, add_ncols, add_nrows, add_nchannels, add_nsamples, eps, stream);
    } else {
        rms_norm_mul_f32_cuda<float, float>(ctx, (const float *) src0_d, (const float *) mul_d, (const float *) add_d, (float *) dst_d,
            ne00, ne01, ne02, ne03, s01, s02, s03, mul_s01, mul_s02, mul_s03,
            mul_ncols, mul_nrows, mul_nchannels, mul_nsamples,
            add_s01, add_s02, add_s03, add_ncols, add_nrows, add_nchannels, add_nsamples, eps, stream);
    }
}

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * grad  = dst->src[0]; // gradients
    const ggml_tensor * src0f = dst->src[1]; // src0 from forward pass

    const float * grad_d  = (const float *) grad->data;
    const float * src0f_d = (const float *) src0f->data;
    float       * dst_d   = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous(grad));

    GGML_ASSERT( grad->type == GGML_TYPE_F32);
    GGML_ASSERT(src0f->type == GGML_TYPE_F32);
    GGML_ASSERT(  dst->type == GGML_TYPE_F32);

    const int64_t ne00 = src0f->ne[0];
    const int64_t nrows = ggml_nrows(src0f);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    rms_norm_back_f32_cuda(grad_d, src0f_d, dst_d, ne00, nrows, eps, stream);
}

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // BF16 activation coverage (2026-09-18)
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == ts0);
    const int64_t s01 = nb01 / ts0;
    const int64_t s02 = nb02 / ts0;
    const int64_t s03 = nb03 / ts0;

    if (src0->type == GGML_TYPE_BF16) {
        l2_norm_f32_cuda<nv_bfloat16>((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data,
            ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    } else {
        l2_norm_f32_cuda<float>((const float *) src0->data, (float *) dst->data,
            ne00, ne01, ne02, ne03, s01, s02, s03, eps, stream);
    }
}
