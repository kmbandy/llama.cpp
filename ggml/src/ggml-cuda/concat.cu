#include "concat.cuh"
#include <cstdlib>

#include <stdint.h>

// contiguous kernels
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE) concat_cont(const T * x,
                                                                             const T * y,
                                                                             T *       dst,
                                                                             int64_t   ne00,
                                                                             int64_t   ne01,
                                                                             int64_t   ne02,
                                                                             int64_t   ne0,
                                                                             int64_t   ne1,
                                                                             int64_t   ne2) {
    static_assert(dim >= 0 && dim <= 2, "dim must be in [0, 2]");

    const int64_t n = ne0 * ne1 * ne2;

    ggml_cuda_pdl_sync();
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        if constexpr (dim == 0) {
            const int64_t row = i / ne0;
            const int64_t i0  = i - row * ne0;

            if (i0 < ne00) {
                dst[i] = x[row * ne00 + i0];
            } else {
                dst[i] = y[row * (ne0 - ne00) + (i0 - ne00)];
            }
        } else if constexpr (dim == 1) {
            const int64_t dst_plane  = ne0 * ne1;
            const int64_t src0_plane = ne0 * ne01;
            const int64_t src1_plane = dst_plane - src0_plane;
            const int64_t i2         = i / dst_plane;
            const int64_t i01        = i - i2 * dst_plane;

            if (i01 < src0_plane) {
                dst[i] = x[i2 * src0_plane + i01];
            } else {
                dst[i] = y[i2 * src1_plane + (i01 - src0_plane)];
            }
        } else {
            const int64_t src0_size = ne0 * ne1 * ne02;

            if (i < src0_size) {
                dst[i] = x[i];
            } else {
                dst[i] = y[i - src0_size];
            }
        }
    }
}

template <typename T>
static void concat_cont_cuda(const T * x,
                             const T * y,
                             T *       dst,
                             int64_t   ne00,
                             int64_t   ne01,
                             int64_t   ne02,
                             int64_t   ne0,
                             int64_t   ne1,
                             int64_t   ne2,
                             int       dim,
                             cudaStream_t stream) {
    const int64_t n          = ne0 * ne1 * ne2;
    const int     num_blocks = (n + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;

    if (dim == 0) {
        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream);
        ggml_cuda_kernel_launch(concat_cont<T, 0>, launch_params, x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    if (dim == 1) {
        concat_cont<T, 1><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
        return;
    }
    concat_cont<T, 2><<<num_blocks, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne00, ne01, ne02, ne0, ne1, ne2);
}

// Fast path for dim-0 concat where src1 is exactly ggml_transpose() of a
// contiguous 2-D [C, T] tensor (e.g. the GDN conv-state concat in
// llm_build_delta_net_base::build_conv_state), which is what makes the
// generic concat_non_cont kernel read src1 with fully uncoalesced,
// element-strided loads (stride C*elem between consecutive "row" elements).
//
// A first attempt at this (see git blame) used a 32x32-thread-tall block
// (1024 threads = 32 wavefronts on gfx1201's wave32) for the LDS-tiled
// transpose and measured SLOWER than the generic kernel despite already
// having coalesced reads/writes and bank-conflict-free padding (tile[32][33])
// -- i.e. neither of the two usual suspects. The actual problem was
// occupancy: a 32-wavefront block can fill an entire CU by itself, so there
// is no sibling block left to hide the __syncthreads()/LDS round-trip
// latency or the load/store phases' memory latency -- every block serializes
// against itself with nothing else in flight on that CU. This version tiles
// the same 32x32 elements but with a 32x8 thread block (256 threads = 8
// wavefronts, matching CUDA_CONCAT_BLOCK_SIZE used by the fallback kernel),
// each thread walking 4 sub-rows of the tile, so multiple blocks can be
// resident per CU and overlap each other's stalls.
//   - src0 -> dst[0:ne00) is a small strided 2-D copy, contiguous on both ends.
//   - src1 -> dst[ne00:ne0) is the 32x32-tile/32x8-thread transpose: reads
//     are coalesced along src1's contiguous axis (dim 1, "C"), writes are
//     coalesced along dst's contiguous axis (dim 0, "W+T").
// Both kernels take a flattened blockIdx.z/blockIdx.y "iseq" index over
// ne[2]*ne[3] so n_seqs (and any ne[3] batching) > 1 is handled in one
// launch rather than requiring the ne[2]==ne[3]==1 restriction the first
// attempt had. Restricted to 4-byte, non-quantized element types (matches
// the byte-size dispatch in concat_cuda); falls back to the generic kernel
// for every other case (non-transpose src1, quantized types, oversized grid).
#define CONCAT_DIM0_TILE      32
#define CONCAT_DIM0_TILE_ROWS 8

// x: [ne00, ne01] contiguous -> dst: [ne0, ne01] contiguous, at column offset 0,
// batched over iseq = blockIdx.y in [0, ne02*ne03).
static __global__ void __launch_bounds__(CONCAT_DIM0_TILE * CONCAT_DIM0_TILE_ROWS)
    concat_dim0_copy_small_src0(const char * __restrict__ x_base, char * __restrict__ dst_base,
                                 int64_t ne00, int64_t ne01, int64_t ne0,
                                 uint64_t nb02, uint64_t nb2) {
    const int64_t iseq = blockIdx.y;
    const uint32_t * __restrict__ x   = (const uint32_t *) (x_base   + iseq * nb02);
    uint32_t *       __restrict__ dst = (uint32_t *)       (dst_base + iseq * nb2);

    const int64_t n = ne00 * ne01;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t) blockDim.x * gridDim.x) {
        const int64_t row = i / ne00;
        const int64_t i0  = i - row * ne00;
        dst[row * ne0 + i0] = x[i];
    }
}

// y: transposed view with ne = [T, C], nb0 = C*elem, nb1 = elem (i.e. y[t, c]
// lives at element offset t*C + c in the underlying contiguous buffer, so
// consecutive c for fixed t is the coalesced-read axis).
// dst: [ne0, C] contiguous, dst[col_off + t, c] written at element offset
// c*ne0 + (col_off + t) (consecutive t for fixed c is the coalesced-write
// axis). Batched over iseq = blockIdx.z in [0, ne02*ne03).
static __global__ void __launch_bounds__(CONCAT_DIM0_TILE * CONCAT_DIM0_TILE_ROWS)
    concat_dim0_transpose_tiled(const char * __restrict__ y_base, char * __restrict__ dst_base,
                                 int64_t T, int64_t C, int64_t ne0, int64_t col_off,
                                 uint64_t nb12, uint64_t nb2) {
    __shared__ uint32_t tile[CONCAT_DIM0_TILE][CONCAT_DIM0_TILE + 1];

    const int64_t iseq = blockIdx.z;
    const uint32_t * __restrict__ y   = (const uint32_t *) (y_base   + iseq * nb12);
    uint32_t *       __restrict__ dst = (uint32_t *)       (dst_base + iseq * nb2);

    const int64_t t_block = (int64_t) blockIdx.x * CONCAT_DIM0_TILE;
    const int64_t c_block = (int64_t) blockIdx.y * CONCAT_DIM0_TILE;

    // load: threadIdx.x indexes c (contiguous read axis in y); 4 sub-rows of
    // t per thread (CONCAT_DIM0_TILE / CONCAT_DIM0_TILE_ROWS == 4).
#pragma unroll
    for (int j = 0; j < CONCAT_DIM0_TILE; j += CONCAT_DIM0_TILE_ROWS) {
        const int64_t c = c_block + threadIdx.x;
        const int64_t t = t_block + threadIdx.y + j;
        if (t < T && c < C) {
            tile[threadIdx.y + j][threadIdx.x] = y[t * C + c];
        }
    }

    __syncthreads();

    // store: threadIdx.x indexes t (contiguous write axis in dst).
#pragma unroll
    for (int j = 0; j < CONCAT_DIM0_TILE; j += CONCAT_DIM0_TILE_ROWS) {
        const int64_t t = t_block + threadIdx.x;
        const int64_t c = c_block + threadIdx.y + j;
        if (t < T && c < C) {
            dst[c * ne0 + (col_off + t)] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Detects the shape/stride class handled above: dim-0 concat, 4-byte
// non-quantized elements, src0 fully contiguous, src1 a genuine transpose
// (ggml_is_transposed) of a contiguous 2-D-per-batch tensor, dst contiguous,
// with matching batch dims (ne[2], ne[3]) across all three tensors. Anything
// else (non-transposed strided src1, quantized/other element widths, a grid
// too large for blockIdx.y/z) falls through to the generic concat_non_cont
// kernel.
static bool ggml_cuda_concat_dim0_transpose_fastpath_supported(const ggml_tensor * src0,
                                                                 const ggml_tensor * src1,
                                                                 const ggml_tensor * dst) {
    if (ggml_is_quantized(src0->type) || ggml_is_quantized(src1->type) || ggml_is_quantized(dst->type)) {
        return false;
    }
    if (ggml_type_size(src0->type) != 4 || ggml_type_size(src1->type) != 4 || ggml_type_size(dst->type) != 4) {
        return false;
    }
    if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(dst)) {
        return false;
    }
    if (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3] ||
        dst->ne[2]  != src0->ne[2] || dst->ne[3]  != src0->ne[3]) {
        return false;
    }
    if (src0->ne[1] != src1->ne[1] || dst->ne[1] != src0->ne[1]) {
        return false;
    }
    if (dst->ne[0] != src0->ne[0] + src1->ne[0]) {
        return false;
    }

    // src1 must be exactly ggml_transpose() of a contiguous [C, T] tensor:
    // nb[1] == elem size (dim1/"C" is the contiguous axis) and
    // nb[0] == ne[1]*elem size (no gaps, i.e. a true transpose, not a
    // general strided view).
    const size_t elem = ggml_type_size(src1->type);
    if (!ggml_is_transposed(src1)) {
        return false;
    }
    if (src1->nb[1] != elem || src1->nb[0] != (size_t) src1->ne[1] * elem) {
        return false;
    }

    // blockIdx.y/z are limited to 65535 on both CUDA and HIP.
    const int64_t n_iseq   = src0->ne[2] * src0->ne[3];
    const int64_t c_tiles  = (src1->ne[1] + CONCAT_DIM0_TILE - 1) / CONCAT_DIM0_TILE;
    if (n_iseq > 65535 || c_tiles > 65535) {
        return false;
    }

    return true;
}

static void concat_dim0_transpose_fastpath_cuda(const ggml_tensor * src0,
                                                  const ggml_tensor * src1,
                                                  ggml_tensor *       dst,
                                                  cudaStream_t        stream) {
    const int64_t ne00   = src0->ne[0]; // W
    const int64_t ne01   = src0->ne[1]; // C
    const int64_t T      = src1->ne[0]; // n_tokens
    const int64_t C      = src1->ne[1]; // C
    const int64_t ne0    = dst->ne[0];  // W + T
    const int64_t n_iseq = src0->ne[2] * src0->ne[3]; // n_seqs (and any ne[3] batching)

    const char * src0_d = (const char *) src0->data;
    const char * src1_d = (const char *) src1->data;
    char *       dst_d  = (char *) dst->data;

    {
        const int64_t n          = ne00 * ne01;
        const int     num_blocks = (int) ((n + CONCAT_DIM0_TILE * CONCAT_DIM0_TILE_ROWS - 1) /
                                           (CONCAT_DIM0_TILE * CONCAT_DIM0_TILE_ROWS));
        dim3 grid(num_blocks, (unsigned) n_iseq);
        concat_dim0_copy_small_src0<<<grid, CONCAT_DIM0_TILE * CONCAT_DIM0_TILE_ROWS, 0, stream>>>(
                src0_d, dst_d, ne00, ne01, ne0, src0->nb[2], dst->nb[2]);
    }

    {
        dim3 block(CONCAT_DIM0_TILE, CONCAT_DIM0_TILE_ROWS);
        dim3 grid((unsigned int) ((T + CONCAT_DIM0_TILE - 1) / CONCAT_DIM0_TILE),
                  (unsigned int) ((C + CONCAT_DIM0_TILE - 1) / CONCAT_DIM0_TILE),
                  (unsigned int) n_iseq);
        concat_dim0_transpose_tiled<<<grid, block, 0, stream>>>(
                src1_d, dst_d, T, C, ne0, ne00, src1->nb[2], dst->nb[2]);
    }
}

// Row-oriented, vectorized dim-0 non-contiguous concat.
//
// concat_non_cont<T,0> below launches one block per (i1,i2,i3) output row and
// re-tests `i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03` for every i0 even
// though for a dim-0 concat ne01==ne1, ne02==ne2, ne03==ne3 always hold, so
// i1/i2/i3 are in-range for the *whole* row regardless of i0 -- the row splits
// cleanly into a [0, ne00) sub-run copied from src0 and a [ne00, ne0) sub-run
// copied from src1, with no per-element branch needed. This kernel copies
// each of those two sub-runs directly, and vectorizes with 16-byte
// (uint4) loads/stores whenever that sub-run's source is itself contiguous
// (its dim-0 stride == sizeof(T)) and the resulting pointers are 16B-aligned
// -- which is exactly the GDN conv-state concat's src0 (a small contiguous
// [d_conv-1, conv_channels] block) and, for the fully-contiguous case, src1
// too. When a source's dim-0 stride isn't sizeof(T) (e.g. src1 is a transpose,
// as in the GDN case), that sub-run falls back to a strided per-element copy
// -- identical bytes to the generic kernel, just without the redundant
// bounds re-check on every element. Bit-identical output to concat_non_cont
// for dim==0; every other dim keeps using the generic kernel unchanged.
template <typename T>
static __device__ __forceinline__ void concat_row_copy_run(const char * __restrict__ src_base,
                                                             uint64_t                 src_step,
                                                             T *       __restrict__   dst,
                                                             int64_t                  n) {
    if (src_step == sizeof(T)) {
        const T * __restrict__ src = (const T *) src_base;
        constexpr int64_t vec_elems = 16 / sizeof(T);
        const bool aligned16 = vec_elems > 1 &&
            ((reinterpret_cast<uintptr_t>(src) & 15) == 0) &&
            ((reinterpret_cast<uintptr_t>(dst) & 15) == 0);
        if (aligned16 && n >= vec_elems) {
            const int64_t   n_vec = n / vec_elems;
            const uint4 * __restrict__ src4 = reinterpret_cast<const uint4 *>(src);
            uint4 *       __restrict__ dst4 = reinterpret_cast<uint4 *>(dst);
            for (int64_t i = threadIdx.x; i < n_vec; i += blockDim.x) {
                dst4[i] = src4[i];
            }
            for (int64_t i = n_vec * vec_elems + threadIdx.x; i < n; i += blockDim.x) {
                dst[i] = src[i];
            }
            return;
        }
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            dst[i] = src[i];
        }
    } else {
        // strided source (e.g. a transpose): no contiguous run to vectorize,
        // copy element-by-element using the real stride.
        for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
            dst[i] = *(const T *) (src_base + i * src_step);
        }
    }
}

template <typename T>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont_dim0_row(const char * __restrict__ src0,
                              const char * __restrict__ src1,
                                    char * __restrict__ dst,
                              int64_t  ne00,
                              uint64_t nb00, uint64_t nb01, uint64_t nb02, uint64_t nb03,
                              uint64_t nb10, uint64_t nb11, uint64_t nb12, uint64_t nb13,
                              int64_t  ne0,
                              uint64_t nb0,  uint64_t nb1,  uint64_t nb2,  uint64_t nb3) {
    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    T * dst_row = (T *) (dst + i3 * nb3 + i2 * nb2 + i1 * nb1);

    const char * src0_row = src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    concat_row_copy_run<T>(src0_row, nb00, dst_row, ne00);

    const char * src1_row = src1 + i3 * nb13 + i2 * nb12 + i1 * nb11;
    concat_row_copy_run<T>(src1_row, nb10, dst_row + ne00, ne0 - ne00);
}

// non-contiguous kernel (slow)
template <typename T, int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3) {
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const T * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const T *)(src0 + i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + i1*nb11 + (i0 - ne00)*nb10);
            } else if constexpr (dim == 1) {
                x = (const T *)(src1 + i3*nb13 + i2*nb12 + (i1 - ne01)*nb11 + i0*nb10);
            } else if constexpr (dim == 2) {
                x = (const T *)(src1 + i3*nb13 + (i2 - ne02)*nb12 + i1*nb11 + i0*nb10);
            } else if constexpr (dim == 3) {
                x = (const T *)(src1 + (i3 - ne03)*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
            }
        }

        T * y = (T *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}

template <typename T>
static void concat_cuda(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, int dim, cudaStream_t stream) {
    if (dim != 3 && ggml_is_contiguous_to_3(src0) && ggml_is_contiguous_to_3(src1)) {
        const T * src0_d = (const T *) src0->data;
        const T * src1_d = (const T *) src1->data;
        T *       dst_d  = (T *) dst->data;

        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            concat_cont_cuda(
                    src0_d + i3*(src0->nb[3] / sizeof(T)),
                    src1_d + i3*(src1->nb[3] / sizeof(T)),
                    dst_d  + i3*( dst->nb[3] / sizeof(T)),
                    ggml_row_size(src0->type, src0->ne[0])/sizeof(T), src0->ne[1], src0->ne[2],
                    ggml_row_size(dst->type, dst->ne[0])/sizeof(T),  dst->ne[1],  dst->ne[2], dim, stream);
        }
    } else if (dim == 3 && ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t size0 = ggml_nbytes(src0);
        const size_t size1 = ggml_nbytes(src1);

        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data,         src0->data, size0, cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync((char *) dst->data + size0, src1->data, size1, cudaMemcpyDeviceToDevice, stream));
    } else if (dim == 0 &&
               dst->nb[0] == sizeof(T) &&
               dst->ne[1] <= 65535 && dst->ne[2] <= 65535 && dst->ne[3] <= 65535) {
        // Row-oriented vectorized path: valid whenever dim-0 concat writes a
        // contiguous destination row (always true for concat's freshly
        // allocated output) -- grid/block layout matches the generic kernel
        // below exactly, just replacing the generic dim==0 instantiation.
        GGML_ASSERT(!ggml_is_quantized(src0->type));
        dim3 grid_dim((unsigned) dst->ne[1], (unsigned) dst->ne[2], (unsigned) dst->ne[3]);
        concat_non_cont_dim0_row<T><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
            (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
            src0->ne[0],
            src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
            src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
            dst->ne[0],
            dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    } else {
        GGML_ASSERT(!ggml_is_quantized(src0->type));

        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_non_cont<T, dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}

void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT(dst->type  == src0->type);

    // Preferred path for the measured GDN conv-state concat shape (dim-0
    // concat where src1 is a transpose view): see the fastpath comment above
    // for tile geometry and why the earlier 32x32-thread-block attempt
    // (previously gated behind MAD_CONCAT_TRANSPOSE_FAST, off by default)
    // measured slower. No longer gated -- it's a strict shape/type match, so
    // anything it doesn't handle falls through to the generic kernel below.
    if (dim == 0 && ggml_cuda_concat_dim0_transpose_fastpath_supported(src0, src1, dst)) {
        concat_dim0_transpose_fastpath_cuda(src0, src1, dst, stream);
        return;
    }

    if (ggml_is_quantized(src0->type)) {
        if (dim == 3) {
            GGML_ASSERT(ggml_is_contiguous(src0));
            GGML_ASSERT(ggml_is_contiguous(src1));
        } else {
            GGML_ASSERT(ggml_is_contiguous_to_3(src0));
            GGML_ASSERT(ggml_is_contiguous_to_3(src1));
        }
        GGML_ASSERT(src0->ne[0] % ggml_blck_size(src0->type) == 0);
        GGML_ASSERT(src1->ne[0] % ggml_blck_size(src1->type) == 0);

        // if first 3 dimensions are contiguous and ne[0] is multiple of the block size we can concat both tensors as byte tensors
        concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
    } else {
        GGML_ASSERT(ggml_blck_size(src0->type) == 1);

        switch (ggml_type_size(src0->type)) {
            case 1:
                concat_cuda<uint8_t>(src0, src1, dst, dim, stream);
                break;
            case 2:
                concat_cuda<uint16_t>(src0, src1, dst, dim, stream);
                break;
            case 4:
                concat_cuda<uint32_t>(src0, src1, dst, dim, stream);
                break;
            case 8:
                concat_cuda<uint64_t>(src0, src1, dst, dim, stream);
                break;
            default:
                GGML_ABORT("Unsupported type size: %zu", ggml_type_size(src0->type));
                break;
        }
    }
}
