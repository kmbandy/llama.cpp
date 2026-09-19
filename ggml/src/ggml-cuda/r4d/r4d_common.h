// Shared device-side machinery for R4D. Everything here is layout algebra verified on hardware
// by targeted layout probes. Do not "fix" any of it from a vendor doc.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

#ifndef R4D_USE_PERMLANE
#define R4D_USE_PERMLANE 0   // an optimisation; __shfl_xor is the correctness baseline
#endif

typedef short    v8s __attribute__((ext_vector_type(8)));
typedef float    v8f __attribute__((ext_vector_type(8)));
typedef float    v2f __attribute__((ext_vector_type(2)));   // what cvt_pk_f32_fp8 actually returns

// exp2 with the hardware transcendental.

__device__ __forceinline__ float ex2(float x) {
#if __has_builtin(__builtin_amdgcn_exp2f)
    return __builtin_amdgcn_exp2f(x);
#else
    return exp2f(x);
#endif
}

// f32 -> bf16, round-to-nearest-even. gfx1201 has NO bf16 convert instruction, so this is the
// software form (verified absent: v_cvt_pk_bf16_f32).
__device__ __forceinline__ uint16_t f32_to_bf16(float f) {
    uint32_t u = __builtin_bit_cast(uint32_t, f);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

// Exchange a value between lane L and lane L^16 (the two halves a wave32 WMMA fragment splits a
// row across). Used for the ONLY cross-lane step the transposed formulation needs.
// v_permlanex16_b32 needs IDENTITY selectors to act as a swap: each 4-bit field names the source
// lane in the other half. sel0 covers lanes 0-7, sel1 lanes 8-15. Zero selectors would broadcast
// lane 0, which is silently wrong rather than obviously wrong.
__device__ __forceinline__ float swap16(float x) {
#if R4D_USE_PERMLANE
    return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(
        __builtin_bit_cast(int, x), __builtin_bit_cast(int, x),
        0x76543210, (int)0xFEDCBA98, /*fi*/false, /*bound_ctrl*/false));
#else
    return __shfl_xor(x, 16, 32);
#endif
}

// 8x8 transpose of 16-bit elements across each group of 8 lanes. Lane j receives element j from
// each of the 8 lanes in its group; the 8 addresses may be arbitrarily strided.
__device__ __forceinline__ v8s load_tr_b128(const void* p) {
    return __builtin_amdgcn_global_load_tr_b128_v8i16((v8s*)const_cast<void*>(p));
}

// Gather the low (sel=0) or high (sel=1) byte of each of four 16-bit lanes of {w0,w1} into one
// dword. Written so the compiler can contract each to a single v_perm_b32.
__device__ __forceinline__ uint32_t byte_gather(uint32_t w0, uint32_t w1, int sel) {
    uint32_t s = (uint32_t)sel * 8u;
    return ((w0 >> s) & 0xffu)
         | (((w0 >> (16 + s)) & 0xffu) << 8)
         | (((w1 >> s) & 0xffu) << 16)
         | (((w1 >> (16 + s)) & 0xffu) << 24);
}

__device__ __forceinline__ void sched_barrier() { __builtin_amdgcn_sched_barrier(0); }

// __syncthreads() is a workgroup-scope acquire-release fence over ALL address spaces, so on gfx12
// it emits `global_inv scope:SCOPE_SE` on both sides of the barrier -- a vector-cache invalidate.
// Every one of those throws away the L0/L1 lines the next tile's KV loads would have hit. These
// barriers only ever order LDS traffic, so scoping the fence to the local address space makes
// the invalidate disappear.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Two-dword form of swap16.
__device__ __forceinline__ uint2 swap16_u2p(uint2 v) {
    return make_uint2(__builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, v.x))),
                      __builtin_bit_cast(uint32_t, swap16(__builtin_bit_cast(float, v.y))));
}
