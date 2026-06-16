// wmma_seed.hip — emit exactly one fp8 WMMA so we can read its encoding.
#include <hip/hip_runtime.h>
typedef int   v2i32  __attribute__((ext_vector_type(2)));
typedef float v8f32  __attribute__((ext_vector_type(8)));
extern "C" __global__ void seed(const v2i32* a, const v2i32* b, v8f32* d) {
    int l = threadIdx.x;
    v8f32 acc = {0,0,0,0,0,0,0,0};
    acc = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[l], b[l], acc);
    d[l] = acc;
}
