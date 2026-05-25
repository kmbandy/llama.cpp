// RDNA4 WMMA probe v2 — disambiguate input layout using non-symmetric A.
//
// Probe v1 used A=identity, which is transpose-invariant. So a "transposed A
// interpretation" by the HW would produce identical results — v1 cannot
// distinguish A_HW = A_mine vs A_HW = A_mine^T.
//
// v2 fixes this: A[i][j] = i*100 + j (non-symmetric), B = identity.
// C = A · B = A.
//   - If HW reads A in I-major as I write it: C[i][j] = i*100+j.
//   - If HW reads A transposed: C[i][j] = j*100+i.
//
// Output values use 100 (not 16) as the base so each "i" digit is decoded
// trivially: e.g. value 1208 = i=12, j=08 → row 12, col 8 in I-frame; row 8,
// col 12 in T-frame.
//
// We ALSO test B transposed by repeating with B[i][j]=i*100+j, A=identity.
//
// Build: hipcc --offload-arch=gfx1201 -O2 -o wmma_rdna4_probe_v2 tests/wmma_rdna4_probe_v2.cu

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

using halfx8_t  = __attribute__((ext_vector_type(8))) _Float16;
using floatx8_t = __attribute__((ext_vector_type(8))) float;

// Load helper: lane L, slot l writes value computed from (row=L%16, col=8*(L/16)+l).
template <int Mode>
__global__ void probe_kernel() {
    const int lane = threadIdx.x;
    const int row = lane % 16;
    const int col_base = 8 * (lane / 16);

    halfx8_t  a_frag;
    halfx8_t  b_frag;
    floatx8_t c_frag = {0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int l = 0; l < 8; ++l) {
        const int col = col_base + l;
        if (Mode == 0) {
            // A non-symmetric, B = identity
            a_frag[l] = (_Float16)((float)(row * 100 + col));
            b_frag[l] = (_Float16)((row == col) ? 1.0f : 0.0f);
        } else {
            // A = identity, B non-symmetric
            a_frag[l] = (_Float16)((row == col) ? 1.0f : 0.0f);
            b_frag[l] = (_Float16)((float)(row * 100 + col));
        }
    }

    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);

    for (int l = 0; l < 8; ++l) {
        const float v   = c_frag[l];
        const int   iv  = (int)v;
        // Decode under I-major output assumption: lane L slot l → C[L%16][8*(L/16)+l]
        const int   r_o = row;
        const int   c_o = col_base + l;
        // Expected if HW frame == my frame:    iv == r_o*100 + c_o
        // Expected if HW frame is transposed:  iv == c_o*100 + r_o
        const int exp_direct = r_o * 100 + c_o;
        const int exp_trans  = c_o * 100 + r_o;
        const char * verdict = (iv == exp_direct) ? "DIRECT"
                              : (iv == exp_trans)  ? "TRANSPOSED"
                              : "OTHER";
        printf("MODE %d LANE %02d SLOT %d  VAL=%6.1f  out_pos(r=%2d,c=%2d) exp_dir=%4d exp_T=%4d -> %s\n",
               Mode, lane, l, v, r_o, c_o, exp_direct, exp_trans, verdict);
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

    fprintf(stderr, "## Mode 0: A non-symmetric, B=I → C should reveal A's layout\n");
    hipLaunchKernelGGL((probe_kernel<0>), dim3(1), dim3(32), 0, 0);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { fprintf(stderr, "kernel mode 0 failed: %s\n", hipGetErrorString(err)); return 1; }

    fprintf(stderr, "## Mode 1: A=I, B non-symmetric → C should reveal B's layout\n");
    hipLaunchKernelGGL((probe_kernel<1>), dim3(1), dim3(32), 0, 0);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { fprintf(stderr, "kernel mode 1 failed: %s\n", hipGetErrorString(err)); return 1; }

    return 0;
}
