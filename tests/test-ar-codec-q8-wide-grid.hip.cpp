// Standalone bit-exactness check for the MT_WIDE_KERNELS retuning of the AR
// codec q8_0 pack/unpack kernels in ggml/src/ggml-cuda/allreduce.cu
// (ggml_cuda_ar_codec_pack_q8_0_kernel / ggml_cuda_ar_codec_unpack_q8_0_kernel).
//
// The retune (see ggml_cuda_ar_codec_lane_grid) only changes how many blocks
// the kernel is launched with -- the kernel body's grid-stride loop is
// untouched, so a narrow grid (many sequential per-thread iterations, the
// old default) and a wide grid (<=2 iterations/thread, MT_WIDE_KERNELS=1)
// must produce bit-identical output for the same input. This test
// reproduces the two kernels verbatim (they are file-local statics in
// allreduce.cu, not exported) and diffs pack+unpack output at a production-
// sized shape (ne = 1024 tokens * 5120 hidden = 5,242,880, the 8k-prompt
// per-sub-batch AR reduce shape) across grid=512 (old default cap) and a
// wide grid (10240, matching ggml_cuda_ar_codec_lane_grid's MT_WIDE_KERNELS
// formula ceil(ne/(2*256)) at this ne).
//
// This file is deliberately NOT wired into tests/CMakeLists.txt -- per the
// task, it is reported but not built/run here. Build with (adjust
// --offload-arch for the target GPU; gfx1201 = RDNA4 RX 9070 XT / R9700):
//
//   hipcc -O2 --offload-arch=gfx1201 -std=c++17 \
//       tests/test-ar-codec-q8-wide-grid.hip.cpp -o /tmp/test-ar-codec-q8-wide-grid
//   /tmp/test-ar-codec-q8-wide-grid
//
// Expected output: "PACK OK (bit-exact)" and "UNPACK OK (bit-exact)".

#include <hip/hip_runtime.h>
#include <cstring>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define QK8_0 32

struct block_q8_0 {
    _Float16 d;
    int8_t   qs[QK8_0];
};

#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \
    fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_e)); exit(1); } } while (0)

__device__ __forceinline__ float warp_reduce_max32(float x) {
    for (int m = 16; m > 0; m >>= 1) {
        x = fmaxf(x, __shfl_xor(x, m, 32));
    }
    return x;
}

// Verbatim copy of ggml_cuda_ar_codec_pack_q8_0_kernel<float> (allreduce.cu).
__global__ void pack_q8_0_kernel(const float * src, block_q8_0 * dst, int n_blocks) {
    const int lane      = threadIdx.x % QK8_0;
    const int ib0       = (blockIdx.x * blockDim.x + threadIdx.x) / QK8_0;
    const int nb_stride = (gridDim.x * blockDim.x) / QK8_0;
    for (int ib = ib0; ib < n_blocks; ib += nb_stride) {
        const float x    = src[ib * QK8_0 + lane];
        const float amax = warp_reduce_max32(fabsf(x));
        const float d    = amax / ((1 << 7) - 1);
        const float id   = d ? 1.0f / d : 0.0f;
        dst[ib].qs[lane] = (int8_t) roundf(x * id);
        if (lane == 0) {
            dst[ib].d = (_Float16) d;
        }
    }
}

// Verbatim copy of ggml_cuda_ar_codec_unpack_q8_0_kernel<float> (allreduce.cu).
__global__ void unpack_q8_0_kernel(float * dst, const block_q8_0 * src, int n_blocks) {
    const int lane      = threadIdx.x % QK8_0;
    const int ib0       = (blockIdx.x * blockDim.x + threadIdx.x) / QK8_0;
    const int nb_stride = (gridDim.x * blockDim.x) / QK8_0;
    for (int ib = ib0; ib < n_blocks; ib += nb_stride) {
        const int   idx = ib * QK8_0 + lane;
        const float x   = dst[idx];
        const float amax = warp_reduce_max32(fabsf(x));
        const float d   = amax / ((1 << 7) - 1);
        const float id  = d ? 1.0f / d : 0.0f;
        const float q   = roundf(x * id);
        const float dh  = (float) (_Float16) d;
        const float local = q * dh;
        const float peer  = (float) src[ib].qs[lane] * (float) (float) src[ib].d;
        dst[idx] = local + peer;
    }
}

int main() {
    const int64_t ne       = 1024LL * 5120LL;   // production 8k-prompt sub-batch shape
    const int     n_blocks = (int) (ne / QK8_0);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

    std::vector<float> h_src(ne), h_dst_old(ne), h_dst_new(ne);
    for (auto & v : h_src)     v = dist(rng);
    for (auto & v : h_dst_old) v = dist(rng);
    h_dst_new = h_dst_old;

    float *d_src, *d_dst_old, *d_dst_new;
    block_q8_0 *d_pack_old, *d_pack_new, *d_peer;
    HIP_CHECK(hipMalloc(&d_src, ne * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst_old, ne * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_dst_new, ne * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_pack_old, n_blocks * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_pack_new, n_blocks * sizeof(block_q8_0)));
    HIP_CHECK(hipMalloc(&d_peer, n_blocks * sizeof(block_q8_0)));

    HIP_CHECK(hipMemcpy(d_src, h_src.data(), ne * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst_old, h_dst_old.data(), ne * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dst_new, h_dst_new.data(), ne * sizeof(float), hipMemcpyHostToDevice));

    // "peer" q8 block used by unpack, identical for both runs.
    pack_q8_0_kernel<<<512, 256>>>(d_src, d_peer, n_blocks);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    const int grid_old  = 512;                                    // old GGML_CUDA_AR_CODEC_GRID cap
    const int grid_wide = (int) ((ne + (2 * 256) - 1) / (2 * 256)); // MT_WIDE_KERNELS formula, <=2 iters

    pack_q8_0_kernel<<<grid_old, 256>>>(d_src, d_pack_old, n_blocks);
    pack_q8_0_kernel<<<grid_wide, 256>>>(d_src, d_pack_new, n_blocks);
    unpack_q8_0_kernel<<<grid_old, 256>>>(d_dst_old, d_peer, n_blocks);
    unpack_q8_0_kernel<<<grid_wide, 256>>>(d_dst_new, d_peer, n_blocks);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<block_q8_0> h_pack_old(n_blocks), h_pack_new(n_blocks);
    std::vector<float> h_unpack_old(ne), h_unpack_new(ne);
    HIP_CHECK(hipMemcpy(h_pack_old.data(), d_pack_old, n_blocks * sizeof(block_q8_0), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_pack_new.data(), d_pack_new, n_blocks * sizeof(block_q8_0), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_unpack_old.data(), d_dst_old, ne * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_unpack_new.data(), d_dst_new, ne * sizeof(float), hipMemcpyDeviceToHost));

    bool pack_ok = (memcmp(h_pack_old.data(), h_pack_new.data(), n_blocks * sizeof(block_q8_0)) == 0);
    bool unpack_ok = (memcmp(h_unpack_old.data(), h_unpack_new.data(), ne * sizeof(float)) == 0);

    printf("grid_old=%d grid_wide=%d n_blocks=%d\n", grid_old, grid_wide, n_blocks);
    printf(pack_ok   ? "PACK OK (bit-exact)\n"   : "PACK MISMATCH\n");
    printf(unpack_ok ? "UNPACK OK (bit-exact)\n" : "UNPACK MISMATCH\n");

    hipFree(d_src); hipFree(d_dst_old); hipFree(d_dst_new);
    hipFree(d_pack_old); hipFree(d_pack_new); hipFree(d_peer);

    return (pack_ok && unpack_ok) ? 0 : 1;
}
