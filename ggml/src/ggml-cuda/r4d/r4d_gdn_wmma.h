// R4D: gfx1201 wave32 bf16 WMMA fragment helpers, shared by the GDN kernels.
// Layout (probed on this hardware, not documented by AMD):
//   16-bit A/B : idx = lane%16 , k = 8*(e>>2) + 4*(lane>>4) + (e&3)   ("two runs of four")
//   f32   C/D  : n   = lane%16 , m = 8*(lane>>4) + e
// So A wants [M][K] and B wants [N][K] -- both K-contiguous -- and a D written to an
// [N][M] buffer is one contiguous 8-element store while [M][N] costs eight scalar ones.
#pragma once
#include <hip/hip_runtime.h>

typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
typedef float  v8f  __attribute__((ext_vector_type(8)));

// gfx1201 has no bf16 convert instruction, so this is software. Round-half-away
// (u + 0x8000) is one add where RTNE needs bfe + add3 + shift; the two differ only
// when the discarded mantissa is exactly 0x8000, i.e. ~2^-16 of values.
__device__ __forceinline__ unsigned short f2bf(float f) {
  return (unsigned short)((__builtin_bit_cast(unsigned, f) + 0x8000u) >> 16);
}
// Pack two floats into one dword of bf16 in 3 instructions (2 adds + v_perm_b32)
// instead of 2 converts plus a shift/or.
__device__ __forceinline__ unsigned f2bf2(float a, float b) {
  unsigned ua = __builtin_bit_cast(unsigned, a) + 0x8000u;
  unsigned ub = __builtin_bit_cast(unsigned, b) + 0x8000u;
  return __builtin_amdgcn_perm(ub, ua, 0x07060302u);   // {b_hi16, a_hi16}
}
__device__ __forceinline__ uint2 f2bf4(float a, float b, float c, float d) {
  return make_uint2(f2bf2(a, b), f2bf2(c, d));
}
// Truncating pack: ONE v_perm_b32 per pair. The +0x8000 rounding adds cannot be
// dual-issued (VOPD rejects the literal), so each is a full VALU slot -- 96 per chunk.
// Truncation costs up to 1 ULP of bf16 instead of 0.5.
__device__ __forceinline__ unsigned f2bf2t(float a, float b) {
  return __builtin_amdgcn_perm(__builtin_bit_cast(unsigned, b),
                               __builtin_bit_cast(unsigned, a), 0x07060302u);
}
__device__ __forceinline__ uint2 f2bf4t(float a, float b, float c, float d) {
  return make_uint2(f2bf2t(a, b), f2bf2t(c, d));
}
__device__ __forceinline__ float bf2f(unsigned short h) {
  return __builtin_bit_cast(float, (unsigned)h << 16);
}

// K-contiguous fragment from a row pointer already advanced to this lane's row and
// half-K. The four dwords are indexed off ONE base so the assembler folds them into
// two ds_read2_b32 with immediate offsets; writing `p + 8` instead makes LLVM
// materialise a second address register per fragment (262 v_add_nc_u32 in the loop).
// The four dwords are two 8-BYTE groups 16 bytes apart, so reading them as uint2 gets
// one ds_load_2addr_b64 where reading them as four unsigned gets two ds_load_2addr_b32:
// LLVM will only pair dwords into a b64 access if the pointer proves 8-byte alignment,
// which every fragment address has (all row pitches are multiples of 4 shorts and the
// lane's k offset is 4*(lane>>4)). Halves the fragment-load instruction count.
__device__ __forceinline__ v8bf fragP(const unsigned short* p) {
  const uint2* pw = (const uint2*)p;
  union { uint2 w[2]; v8bf f; } u;
  u.w[0] = pw[0]; u.w[1] = pw[2];
  return u.f;
}
// Row pointer for the fragment above: LDS source, `pitch` in elements.
__device__ __forceinline__ const unsigned short* rowP(const unsigned short* src,
                                                      int pitch, int base, int lane) {
  return src + (base + (lane & 15)) * pitch + 4 * (lane >> 4);
}
// Row pointer for a global source; `lim` clamps so a partial last chunk cannot read
// past the tensor (those rows are discarded downstream).
__device__ __forceinline__ const unsigned short* rowG(const unsigned short* base,
                                                      size_t rowstride, int row0,
                                                      int lim, int lane) {
  int r = row0 + (lane & 15); r = r < lim ? r : lim;
  return base + (size_t)r * rowstride + 4 * (lane >> 4);
}
// 32-BIT element offset for a global fragment source, split from a UNIFORM base.
// The pointer form above compiles to ~13 instructions per fragment at this kernel's
// register pressure: the `size_t` row stride forces v_mad_co_i64_i32 + v_lshlrev_b64 +
// a 64-bit add, the ternary becomes exec-mask manipulation, LLVM will not keep the
// address live across the k loop so it remats all of it per load, and a generic
// pointer emits flat_load (which bumps BOTH loadcnt and dscnt on gfx12, so an LDS
// drain also waits on HBM). Keeping the offset in one 32-bit register instead lets
// the whole k loop be global_load_b64 vdst, voff, s[base] with a 13-bit immediate.
__device__ __forceinline__ unsigned rowGo(unsigned rowstride, int row0, int lim,
                                          int lane) {
  int r = row0 + (lane & 15); r = r < lim ? r : lim;
  return (unsigned)r * rowstride + 4u * (unsigned)(lane >> 4);
}
__device__ __forceinline__ v8bf fragGo(const unsigned short* base, unsigned off) {
  const uint2* pw = (const uint2*)(base + off);      // 8-byte aligned: see fragP
  union { uint2 w[2]; v8bf f; } u;
  u.w[0] = pw[0]; u.w[1] = pw[2];
  return u.f;
}
// Fragment from a 32-BIT LDS BYTE OFFSET off the shared-block base. LDS row pointers
// into different buffers of a large __shared__ block are further apart than
// ds_read2_b32's immediate reach (8-bit fields in dwords = 1020 B), so LLVM will not
// keep one base register per buffer live -- it rematerialises
// `v_add_nc_u32 vaddr, <buffer_const>, vbase` in front of EVERY fragment, and then
// splits the fragment's two dword pairs across two address registers as well.
// Holding the offset in one opaque register instead makes all 8 k steps immediates
// (8 dwords apart, 61 max -- well inside the field).
__device__ __forceinline__ v8bf fragO(const char* sb, unsigned off) {
  const uint2* pw = (const uint2*)(sb + off);
  union { uint2 w[2]; v8bf f; } u;
  u.w[0] = pw[0]; u.w[1] = pw[2];
  return u.f;
}
#define LDSOPAQUE(x) asm("" : "+v"(x))

// K-contiguous fragment: src[idx][k], row pitch `pitch` (in elements)
__device__ __forceinline__ v8bf frag(const unsigned short* src, int pitch,
                                     int base, int kbase, int lane) {
  return fragP(rowP(src, pitch, base, lane) + kbase);
}

// Same fragment but the source is stored transposed: src[k][idx]. Costs 8 scalar reads.
__device__ __forceinline__ v8bf fragT(const unsigned short* src, int pitch,
                                      int base, int kbase, int lane) {
  int idx = base + (lane & 15);
  int k0 = kbase + 4 * (lane >> 4);
  union { unsigned short h[8]; v8bf f; } u;
#pragma unroll
  for (int e = 0; e < 4; ++e) {
    u.h[e]     = src[(size_t)(k0 + e) * pitch + idx];
    u.h[4 + e] = src[(size_t)(k0 + 8 + e) * pitch + idx];
  }
  return u.f;
}

__device__ __forceinline__ v8f mma(v8bf a, v8bf b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// C[mt*16 .. ][nt*16 .. ] += A[M][KA] @ B[N][KA]^T over KSTEP 16-wide steps
template <int KSTEP>
__device__ __forceinline__ v8f gemm(const unsigned short* A, int Ap, int mt,
                                    const unsigned short* B, int Bp, int nt,
                                    int lane, v8f acc) {
#pragma unroll
  for (int s = 0; s < KSTEP; ++s)
    acc = mma(frag(A, Ap, mt * 16, s * 16, lane), frag(B, Bp, nt * 16, s * 16, lane), acc);
  return acc;
}

// Fragment straight from global memory (no LDS staging). `lim` clamps the row so a
// partial last chunk cannot read past the tensor; those rows are discarded downstream.
__device__ __forceinline__ v8bf fragG(const unsigned short* base, size_t rowstride,
                                      int row0, int lim, int kbase, int lane) {
  int r = row0 + (lane & 15); r = r < lim ? r : lim;
  const unsigned short* p = base + (size_t)r * rowstride + kbase + 4 * (lane >> 4);
  union { unsigned w[4]; v8bf f; } u;
  u.w[0] = ((const unsigned*)p)[0];       u.w[1] = ((const unsigned*)p)[1];
  u.w[2] = ((const unsigned*)(p + 8))[0]; u.w[3] = ((const unsigned*)(p + 8))[1];
  return u.f;
}
