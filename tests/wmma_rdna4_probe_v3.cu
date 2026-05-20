// RDNA4 WMMA probe v3 — ground-truth per-element layout via ISA spec.
//
// Per RDNA4 ISA §7.12.2 (Matrix Element Storage in VGPRs), wave32, fp16/fp32:
//
//   A (16x16 fp16): lane = {col[2], row[3:0]}, vgpr = {col[3], col[1]}, startPosn = col[0]
//   B (16x16 fp16): lane = {row[2], col[3:0]}, vgpr = {row[3], row[1]}, startPosn = row[0]
//   D (16x16 fp32): lane = {row[3], col[3:0]}, vgpr = row[2:0],         startPosn = 0
//
// Each lane stores 8 fp16 across 4 dwords (halfx8). Slot s in halfx8 ↔ (vgpr=s/2, startPosn=s%2).
//
// Probe: write A[i][j] = i*100+j into halfx8 per-lane using the inverse ISA map.
// Write B as true HW identity (1 where i==j else 0).
// Compute C = A·B = A. Read C via the ISA's D-matrix map and verify.
//
// If output matches per-(i,j) expected, ISA understanding is correct → use as fix foundation.
//
// Build: hipcc --offload-arch=gfx1201 -O2 -o wmma_rdna4_probe_v3 tests/wmma_rdna4_probe_v3.cu

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

using halfx8_t  = __attribute__((ext_vector_type(8))) _Float16;
using floatx8_t = __attribute__((ext_vector_type(8))) float;

// For lane L and slot s, return (row, col) of A's element at that position.
__device__ inline void a_pos(int L, int s, int & row, int & col) {
    row = L & 0xF;
    const int col2 = (L >> 4) & 1;   // bit 2 of col
    const int col3 = (s >> 2) & 1;   // bit 3 of col (slot's high bit pair)
    const int col1 = (s >> 1) & 1;   // bit 1 of col
    const int col0 = s & 1;          // bit 0 of col = startPosn
    col = (col3 << 3) | (col2 << 2) | (col1 << 1) | col0;
}

// For lane L and slot s, return (row, col) of B's element at that position.
__device__ inline void b_pos(int L, int s, int & row, int & col) {
    col = L & 0xF;
    const int row2 = (L >> 4) & 1;
    const int row3 = (s >> 2) & 1;
    const int row1 = (s >> 1) & 1;
    const int row0 = s & 1;
    row = (row3 << 3) | (row2 << 2) | (row1 << 1) | row0;
}

// For lane L and slot s, return (row, col) of D's element at that position.
__device__ inline void d_pos(int L, int s, int & row, int & col) {
    col = L & 0xF;
    const int row3 = (L >> 4) & 1;
    row = (row3 << 3) | s;   // s ∈ [0,8) is row[2:0]
}

__global__ void probe_v3_kernel() {
    const int L = threadIdx.x;

    halfx8_t  a_frag;
    halfx8_t  b_frag;
    floatx8_t c_frag = {0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        int ai, aj, bi, bj;
        a_pos(L, s, ai, aj);
        b_pos(L, s, bi, bj);
        a_frag[s] = (_Float16)((float)(ai * 100 + aj));     // A[i][j] = i*100+j
        b_frag[s] = (_Float16)((bi == bj) ? 1.0f : 0.0f);   // B = HW identity
    }

    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a_frag, b_frag, c_frag);

    // Read D and verify D[i][j] == A[i][j] == i*100+j.
    int n_ok = 0, n_bad = 0;
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        int di, dj;
        d_pos(L, s, di, dj);
        const int v        = (int)c_frag[s];
        const int expected = di * 100 + dj;
        const bool ok      = (v == expected);
        if (ok) ++n_ok; else ++n_bad;
        printf("LANE %02d SLOT %d  D(r=%2d,c=%2d) val=%5d  exp=%5d  %s\n",
               L, s, di, dj, v, expected, ok ? "OK" : "MISMATCH");
    }
    if (L == 0) {
        printf("# (per-lane counts not aggregated; grep \"MISMATCH\" to check.)\n");
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

    hipLaunchKernelGGL(probe_v3_kernel, dim3(1), dim3(32), 0, 0);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) { fprintf(stderr, "kernel failed: %s\n", hipGetErrorString(err)); return 1; }
    return 0;
}
