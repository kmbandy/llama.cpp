#include "scale.cuh"

#define MAX_GRIDDIM_X 0x7FFFFFFF

// BF16 activation coverage (2026-09-18)
template <typename T>
static __global__ void scale_f32(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    ggml_cuda_pdl_lc();
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    ggml_cuda_pdl_sync();
    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = (T) (scale * (float) x[i] + bias);
    }
}

// MT_WIDE_KERNELS: scale_f32's grid-stride loop already covers most shapes
// in a single iteration once nelements/CUDA_SCALE_BLOCK_SIZE exceeds a few
// thousand blocks, but small/medium tensors on the TP critical path (e.g. a
// per-tensor scale applied to something narrower than the 5120-wide hidden
// dim) can land with only a few hundred blocks -- not enough resident waves
// to hide latency on gfx1201's 64 CUs. This path forces a wide grid
// (>=4096 blocks where the element count allows) and reads/writes 128-bit
// (VEC-element) chunks per iteration instead of 1 scalar per thread, so a
// grid-stride loop that used to need several iterations for a modest
// nelements now needs <=2. Arithmetic (scale*x+bias, same cast order) is
// bit-for-bit identical to the scalar kernel.
template <typename T>
static __global__ void scale_f32_vec16(const T * __restrict__ x, T * __restrict__ dst,
                                       const float scale, const float bias, const int64_t nelements) {
    constexpr int VEC = 16 / (int) sizeof(T);
    const int64_t ne_vec = nelements / VEC;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    for (int64_t iv = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; iv < ne_vec; iv += stride) {
        const uint4 xv = *reinterpret_cast<const uint4 *>(x + (size_t) iv * VEC);
        const T * xs = reinterpret_cast<const T *>(&xv);
        T out[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            out[j] = (T) (scale * (float) xs[j] + bias);
        }
        *reinterpret_cast<uint4 *>(dst + (size_t) iv * VEC) = *reinterpret_cast<const uint4 *>(out);
    }
    for (int64_t i = ne_vec * VEC + (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < nelements; i += stride) {
        dst[i] = (T) (scale * (float) x[i] + bias);
    }
}

template <typename T>
static void scale_f32_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    constexpr int VEC = 16 / (int) sizeof(T);
    if (ggml_cuda_mt_wide_kernels_enabled() && VEC > 1 && nelements >= VEC && (nelements % VEC) == 0 &&
        (((uintptr_t) x) % 16) == 0 && (((uintptr_t) dst) % 16) == 0) {
        const int64_t ne_vec = nelements / VEC;
        constexpr int block = 256;
        const int64_t needed = (ne_vec + block - 1) / block;
        // Wide grid target (>=4096 blocks) capped so we never launch more
        // blocks than there is vectorized work for, and never past the
        // gridDim.x limit.
        const int64_t grid = std::max<int64_t>(1, std::min<int64_t>(std::max<int64_t>(needed, 4096), std::min<int64_t>(ne_vec, MAX_GRIDDIM_X)));
        scale_f32_vec16<T><<<(unsigned) grid, block, 0, stream>>>(x, dst, scale, bias, nelements);
        return;
    }
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream);
    ggml_cuda_kernel_launch(scale_f32<T>, launch_params, x, dst, scale, bias, nelements);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    // BF16 activation coverage (2026-09-18)
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == src0->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_BF16) {
        scale_f32_cuda<nv_bfloat16>((const nv_bfloat16 *) src0->data, (nv_bfloat16 *) dst->data, scale, bias, ggml_nelements(src0), stream);
    } else {
        scale_f32_cuda<float>((const float *) src0->data, (float *) dst->data, scale, bias, ggml_nelements(src0), stream);
    }
}
