// RDNA4 WMMA C-output lane-mapping probe.
//
// Goal: empirically derive (lane, slot) -> (row, col) for
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12
//
// Method: A = I, B[i][j] = i*16 + j (fits exact in fp16, max 255).
// Then C = A @ B = B, so each acc slot holds row*16 + col — the integer
// value directly encodes its (row, col) destination.
//
// Build:  hipcc --offload-arch=gfx1201 -O2 -o wmma_rdna4_probe tests/wmma_rdna4_probe.cu
// Run:    ./wmma_rdna4_probe | sort -k2 -n

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

using halfx8_t  = __attribute__((ext_vector_type(8))) _Float16;
using floatx8_t = __attribute__((ext_vector_type(8))) float;

// Input load layout assumption for A & B fragments on RDNA4 wave32
// (matches mma.cuh I-major: lane L, slot l -> matrix[L%16][8*(L/16)+l]).
// If the *input* assumption is wrong the C decode will look garbled too,
// but the contiguous I-major load formula is well-established for RDNA4
// inputs (only the C-output layout is the open question).
__global__ void probe_kernel() {
    const int lane = threadIdx.x;       // wave32, one warp per block

    halfx8_t  a_frag;
    halfx8_t  b_frag;
    floatx8_t c_frag = {0,0,0,0,0,0,0,0};

    const int row = lane % 16;
    const int col_base = 8 * (lane / 16);

    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int col = col_base + l;
        // A = identity
        a_frag[l] = (_Float16)((row == col) ? 1.0f : 0.0f);
        // B[i][j] = i*16 + j  (max 255, exact in fp16)
        b_frag[l] = (_Float16)((float)(row * 16 + col));
    }

    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);

    // Emit per-lane printf with a stable sortable prefix.
    // Each slot: value, then decoded (row, col) assuming value == row*16+col.
    for (int l = 0; l < 8; ++l) {
        const float v = c_frag[l];
        const int   iv = (int)v;
        const int   r  = iv / 16;
        const int   c  = iv % 16;
        const bool  exact = (v == (float)iv) && (iv >= 0) && (iv < 256);
        printf("LANE %02d SLOT %d VAL %7.2f  -> (row=%2d, col=%2d) %s\n",
               lane, l, v, r, c, exact ? "OK" : "??");
    }
}

int main() {
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    hipLaunchKernelGGL(probe_kernel, dim3(1), dim3(32), 0, 0);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "kernel launch failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    return 0;
}
