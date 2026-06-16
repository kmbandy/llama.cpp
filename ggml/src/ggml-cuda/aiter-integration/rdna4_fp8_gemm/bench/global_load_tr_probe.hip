// bench/global_load_tr_probe.hip — Phase 0: discover + verify gfx1201 global_load_tr_b64 fp8 layout.
// Unsupervised normal-HIP launch (no PM4). Mirrors ck/utility/amd_transpose_load.hpp fp8 path.
//   (default)   : dump the per-lane byte transpose table (discovery).
//   --validate  : prove the derived B-preshuffle contract via a single 16x16x16 fp8 WMMA vs oracle.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <string>

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

// gfx12 global transpose-load, 8-bit element / b64 form (one wave32 lane gets 8 bytes).
__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto gptr = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                    reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(gptr);
}

// ---- discovery probe: lane L passes Bsrc + L*8; record what each lane receives ----
extern "C" __global__ void probe(const uint8_t* __restrict__ Bsrc, uint8_t* __restrict__ out) {
    int lane = threadIdx.x;                 // 0..31
    v2i32 r = tr_load8(Bsrc + lane * 8);
    uint32_t lo = (uint32_t)r.x, hi = (uint32_t)r.y;
    #pragma unroll
    for (int s = 0; s < 4; s++) out[lane * 8 + s]     = (lo >> (s * 8)) & 0xFF;
    #pragma unroll
    for (int s = 0; s < 4; s++) out[lane * 8 + 4 + s] = (hi >> (s * 8)) & 0xFF;
}

static int run_probe() {
    const int NB = 256;
    uint8_t hsrc[NB]; for (int i = 0; i < NB; i++) hsrc[i] = (uint8_t)i;   // byte[k][n] == k*16+n
    uint8_t hout[256] = {0};
    uint8_t *dsrc, *dout;
    if (hipMalloc(&dsrc, NB) || hipMalloc(&dout, 256)) { printf("hipMalloc fail\n"); return 3; }
    hipMemcpy(dsrc, hsrc, NB, hipMemcpyHostToDevice);
    hipMemset(dout, 0, 256);
    probe<<<1, 32>>>(dsrc, dout);
    if (hipDeviceSynchronize() != hipSuccess) { printf("launch fail\n"); return 3; }
    hipMemcpy(hout, dout, 256, hipMemcpyDeviceToHost);
    printf("# byte value v at [k][n] == k*16+n, so v -> (k=v/16, n=v%%16)\n");
    printf("lane :  s0  s1  s2  s3  s4  s5  s6  s7   ->  (k,n) per slot\n");
    for (int L = 0; L < 32; L++) {
        printf("%4d :", L);
        for (int s = 0; s < 8; s++) printf(" %3d", hout[L * 8 + s]);
        printf("   ->");
        for (int s = 0; s < 8; s++) printf(" (%d,%d)", hout[L * 8 + s] / 16, hout[L * 8 + s] % 16);
        printf("\n");
    }
    hipFree(dsrc); hipFree(dout);
    return 0;
}

// ---- validation: the derived transpose permutation + single-tile WMMA vs CPU e4m3 oracle ----
static uint8_t enc(float x){ __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec(uint8_t b){ __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v)=b; return (float)v; }
static float patA(int i,int k){ return (float)(((i*7 + k*3) % 9) - 4) * 0.5f; }
static float patB(int k,int n){ return (float)(((k*5 + n*2) % 9) - 4) * 0.5f; }

// Closed-form transpose permutation derived (+ hand-verified) from the probe table:
// tr_load8(Bshuf + L*8) delivers, to output lane L slot s, the byte Bshuf[trperm(L,s)].
__host__ __device__ inline int trperm(int L, int s) {
    int base = (L & 7) + ((L >> 3) & 1) * 32 + ((L >> 4) & 1) * 128;
    return base + (s & 3) * 8 + ((s >> 2) & 1) * 64;
}

__global__ void onetile(const uint8_t* A, const uint8_t* Bshuf, float* C) {
    int lane = threadIdx.x;
    // A-map (§7.12): lane->row=lane&0xF, colhi=(lane>>4)&1; 8 contiguous K-bytes of that row.
    int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
    v2i32 fa = *reinterpret_cast<const v2i32*>(A + row_a * 16 + colhi * 8);
    v2i32 fb = tr_load8(Bshuf + lane * 8);
    v8f32 acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa, fb, v8f32{0,0,0,0,0,0,0,0});
    // C/D map: lane holds col=lane&0xF, rows ((lane>>4)&1)*8 + s
    int col = lane & 0xF, row0 = ((lane >> 4) & 1) * 8;
    #pragma unroll
    for (int s = 0; s < 8; s++) C[(row0 + s) * 16 + col] = acc[s];
}

static int validate() {
    std::vector<uint8_t> A(256), Blog(256), Bshuf(256, 0);
    for (int i=0;i<16;i++) for (int k=0;k<16;k++) A[i*16+k]   = enc(patA(i,k));
    for (int k=0;k<16;k++) for (int n=0;n<16;n++) Blog[k*16+n]= enc(patB(k,n));
    // preshuffle: place logical B[k][n] at the global offset the transpose pulls into (lane,slot).
    for (int L=0;L<32;L++) for (int s=0;s<8;s++) {
        int k = ((L >> 4) & 1) * 8 + s, n = L & 15;
        Bshuf[trperm(L, s)] = Blog[k * 16 + n];
    }
    std::vector<float> ref(256, 0.f);
    for (int i=0;i<16;i++) for (int n=0;n<16;n++) { float a=0;
        for (int k=0;k<16;k++) a += dec(A[i*16+k]) * dec(Blog[k*16+n]); ref[i*16+n]=a; }
    uint8_t *dA,*dB; float* dC;
    hipMalloc(&dA,256); hipMalloc(&dB,256); hipMalloc(&dC,256*4);
    hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice);
    hipMemcpy(dB, Bshuf.data(), 256, hipMemcpyHostToDevice);
    onetile<<<1,32>>>(dA, dB, dC);
    if (hipDeviceSynchronize() != hipSuccess) { printf("launch fail\n"); return 3; }
    std::vector<float> C(256); hipMemcpy(C.data(), dC, 256*4, hipMemcpyDeviceToHost);
    float mx=0; for (int t=0;t<256;t++){ float e=fabsf(C[t]-ref[t])/(fabsf(ref[t])+1e-3f); if(e>mx)mx=e; }
    printf("onetile (global_load_tr + preshuffle) max_rel_err=%.4f  %s\n", mx, mx<0.03f?"PASS":"*** FAIL ***");
    hipFree(dA); hipFree(dB); hipFree(dC);
    return mx < 0.03f ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--validate") return validate();
    return run_probe();
}
