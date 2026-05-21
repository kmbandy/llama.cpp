// MAD-214 Phase 1D step 0: FP8 WMMA probe for RDNA4 (gfx1201).
//
// Goal: empirically verify the per-lane × per-slot layout for
// v_wmma_f32_16x16x16_fp8_fp8 on R9700 / gfx1201, paralleling the FP16 probe
// (tests/wmma_rdna4_probe_v3.cu) that grounded MAD-180's WMMA fix.
//
// Why probe first: per RDNA4 ISA §7.12.2 the A/B/D matrix VGPR layout should
// be structurally identical between fp16 and fp8 since both are 16x16x16
// configs with 8 elements per lane in each 32-lane wave. But "should be" is
// not "is" — yesterday's FP16 fix proved that assuming the ISA without
// probing eats days. FP8 docs are sparser; probe is even more justified.
//
// Test: A = HW identity, B = HW identity (both 16x16 E4M3, encoded via the
// ISA-stated positional map). Compute C = A·B = identity (FP32 16x16). Read
// C via the ISA's D-matrix map and verify C[i][j] == (i==j ? 1 : 0).
//
// If all 256 elements check OK -> FP8 layout matches FP16 layout; we can
// reuse the same fragment-pack/unpack code for the kernel.
// If anything mismatches -> capture which (lane, slot) produces wrong output
// and dig deeper.
//
// Build:
//   hipcc --offload-arch=gfx1201 -O2 -o /tmp/wmma_rdna4_fp8_probe \
//       tests/wmma_rdna4_fp8_probe.cu

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdint>

using fp8x8_t   = __attribute__((ext_vector_type(8))) int8_t;
using floatx8_t = __attribute__((ext_vector_type(8))) float;

// E4M3 byte for the value 1.0: sign=0, exp=7 (bias 7 -> 2^0), mantissa=0
//   byte = (7 << 3) | 0 = 0x38
constexpr int8_t E4M3_ONE  = (int8_t)0x38;
constexpr int8_t E4M3_ZERO = (int8_t)0x00;

// ---------------------------------------------------------------------------
// Position maps per RDNA4 ISA §7.12.2 (mirror of tests/wmma_rdna4_probe_v3.cu)
// ---------------------------------------------------------------------------

// For lane L and slot s, return (row, col) of A's element at that position.
__device__ inline void a_pos(int L, int s, int & row, int & col) {
    row = L & 0xF;
    const int col2 = (L >> 4) & 1;   // bit 2 of col
    const int col3 = (s >> 2) & 1;   // bit 3 of col
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
    row = (row3 << 3) | s;
}

// ---------------------------------------------------------------------------
// Probe kernel — writes full 32x8 result matrix to device memory (avoids
// HIP printf buffer truncation that hides high-slot output).
// ---------------------------------------------------------------------------
struct slot_result {
    int   lane;
    int   slot;
    int   di;
    int   dj;
    float value;
    float expected;
};

__global__ void probe_fp8_kernel(slot_result *out) {
    const int L = threadIdx.x;

    fp8x8_t   a_frag = {0,0,0,0,0,0,0,0};
    fp8x8_t   b_frag = {0,0,0,0,0,0,0,0};
    floatx8_t c_frag = {0,0,0,0,0,0,0,0};

    // Encode A = identity (1 on i==j, 0 else) into this lane's 8 slots
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        int ai, aj;
        a_pos(L, s, ai, aj);
        a_frag[s] = (ai == aj) ? E4M3_ONE : E4M3_ZERO;
    }

    // Encode B = identity similarly
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        int bi, bj;
        b_pos(L, s, bi, bj);
        b_frag[s] = (bi == bj) ? E4M3_ONE : E4M3_ZERO;
    }

    // FP8 WMMA: C = A · B in FP32 accumulator.
    c_frag = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
        a_frag, b_frag, c_frag);

    // Write all 8 slots to device memory at the per-lane stripe.
    #pragma unroll
    for (int s = 0; s < 8; ++s) {
        int di, dj;
        d_pos(L, s, di, dj);
        slot_result &r = out[L * 8 + s];
        r.lane     = L;
        r.slot     = s;
        r.di       = di;
        r.dj       = dj;
        r.value    = c_frag[s];
        r.expected = (di == dj) ? 1.0f : 0.0f;
    }
}

int main() {
    hipError_t err = hipSetDevice(0);
    if (err != hipSuccess) {
        fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(err));
        return 1;
    }
    hipDeviceProp_t prop;
    (void)hipGetDeviceProperties(&prop, 0);
    fprintf(stderr, "# device: %s (gcnArch=%s)\n", prop.name, prop.gcnArchName);

    const int N = 32 * 8;  // lanes × slots
    slot_result *d_out = nullptr;
    err = hipMalloc(&d_out, N * sizeof(slot_result));
    if (err != hipSuccess) { fprintf(stderr, "hipMalloc failed: %s\n", hipGetErrorString(err)); return 1; }

    hipLaunchKernelGGL(probe_fp8_kernel, dim3(1), dim3(32), 0, 0, d_out);
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "kernel failed: %s\n", hipGetErrorString(err));
        hipFree(d_out);
        return 1;
    }

    slot_result h_out[N];
    err = hipMemcpy(h_out, d_out, N * sizeof(slot_result), hipMemcpyDeviceToHost);
    hipFree(d_out);
    if (err != hipSuccess) { fprintf(stderr, "hipMemcpy failed: %s\n", hipGetErrorString(err)); return 1; }

    int n_ok = 0, n_bad = 0;
    for (int i = 0; i < N; ++i) {
        const slot_result &r = h_out[i];
        bool ok = (r.value == r.expected);
        if (ok) ++n_ok; else ++n_bad;
        if (!ok || (i < 16)) {  // print first 16 + every mismatch
            printf("LANE %02d SLOT %d  D(r=%2d,c=%2d) val=%g  exp=%g  %s\n",
                   r.lane, r.slot, r.di, r.dj, r.value, r.expected, ok ? "OK" : "MISMATCH");
        }
    }
    printf("# total: %d OK, %d MISMATCH out of %d  =>  %s\n",
           n_ok, n_bad, N, n_bad == 0 ? "PASS — FP8 layout matches FP16" : "FAIL");
    return n_bad == 0 ? 0 : 1;
}
