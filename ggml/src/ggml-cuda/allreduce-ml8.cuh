// ml8-k wire codecs for the tensor-parallel AllReduce.
//
// k-bit index into a signed E4M3 codebook, plus one fp16 absmax scale per
// 32-element block -- the same block size as the ggml q-family, so bit budgets
// compare directly (k + 0.5 bits/element).
//
// Why E4M3 centroids rather than integer levels: gfx1201 (RDNA4) has native fp8,
// and the wider ml8 programme uses one E4M3 lattice across weights, KV and now
// the wire. Measured on real captured partials at matched block size and matched
// bits, ml8 and the q-family are within 2-8% relative RMSE of each other, so the
// format choice is decided by the hardware datapath, not by accuracy.
//
// The codebooks are BAKED constants fitted offline from real partials, so both
// ranks encode identically by construction: no runtime LUT, no calibration
// warmup, and no cross-rank codebook-identity assertion to get wrong.
#pragma once

#include "allreduce-ml8-centroids.cuh"

#define QK_ML8_WIRE 32

typedef struct {
    half    d;                          //  2 bytes: fp16 absmax scale
    uint8_t qs[QK_ML8_WIRE / 2];        // 16 bytes: 4-bit indices
} block_ml8_4_wire;                     // 18 bytes = 4.50 bits/element
static_assert(sizeof(block_ml8_4_wire) == 18, "ml8_4 wire block must be 18 bytes");

typedef struct {
    half    d;                          //  2 bytes: fp16 absmax scale
    uint8_t qh[QK_ML8_WIRE / 8];        //  4 bytes: bit 4 of each index
    uint8_t qs[QK_ML8_WIRE / 2];        // 16 bytes: low nibbles
} block_ml8_5_wire;                     // 22 bytes = 5.50 bits/element
static_assert(sizeof(block_ml8_5_wire) == 22, "ml8_5 wire block must be 22 bytes");

// ml8-6 needs 2 high bits per index (bits 4-5), not the 1 high bit ml8-5's
// qh bitmask carries -- so qh here is a packed 2-bit-per-element plane
// (4 indices/byte) rather than ml8-5's 1-bit-per-element OR'd bitmask
// (8 indices/byte). This does NOT generalize from ml8-5's qh layout by
// just widening the array; see ml8_6_quantize/dequantize below for the
// (j%4)*2 packing this requires instead of ml8-5's (j%8) bit-OR.
typedef struct {
    half    d;                          //  2 bytes: fp16 absmax scale
    uint8_t qh[QK_ML8_WIRE / 4];        //  8 bytes: bits 4-5 of each index, 2 bits/elem
    uint8_t qs[QK_ML8_WIRE / 2];        // 16 bytes: low nibbles (bits 0-3)
} block_ml8_6_wire;                     // 26 bytes = 6.50 bits/element
static_assert(sizeof(block_ml8_6_wire) == 26, "ml8_6 wire block must be 26 bytes");

static __device__ __forceinline__ int ml8_nearest(float v, const float * cb, int n) {
    int   best = 0;
    float bd   = fabsf(v - cb[0]);
    for (int i = 1; i < n; ++i) {
        const float d = fabsf(v - cb[i]);
        if (d < bd) { bd = d; best = i; }
    }
    return best;
}

static __device__ __forceinline__ float ml8_block_scale(const float * v) {
    float amax = 0.0f;
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        amax = fmaxf(amax, fabsf(v[j]));
    }
    return amax;
}

// ---- ml8-4 -----------------------------------------------------------------
static __device__ __forceinline__ void ml8_4_quantize(const float * v, block_ml8_4_wire * b) {
    const float amax = ml8_block_scale(v);
    const half  dh   = __float2half(amax);
    const float d    = __half2float(dh);
    const float inv  = d > 0.0f ? 1.0f / d : 0.0f;
    b->d = dh;
    for (int j = 0; j < QK_ML8_WIRE; j += 2) {
        const int i0 = ml8_nearest(v[j + 0] * inv, ML8_4_CENTROIDS, ML8_4_N_CENT);
        const int i1 = ml8_nearest(v[j + 1] * inv, ML8_4_CENTROIDS, ML8_4_N_CENT);
        b->qs[j / 2] = (uint8_t) (i0 | (i1 << 4));
    }
}

static __device__ __forceinline__ void ml8_4_dequantize(const block_ml8_4_wire * b, float * v) {
    const float d = __half2float(b->d);
    for (int j = 0; j < QK_ML8_WIRE; j += 2) {
        const uint8_t p = b->qs[j / 2];
        v[j + 0] = ML8_4_CENTROIDS[p & 0xF] * d;
        v[j + 1] = ML8_4_CENTROIDS[p >>  4] * d;
    }
}

// ---- ml8-5 -----------------------------------------------------------------
static __device__ __forceinline__ void ml8_5_quantize(const float * v, block_ml8_5_wire * b) {
    const float amax = ml8_block_scale(v);
    const half  dh   = __float2half(amax);
    const float d    = __half2float(dh);
    const float inv  = d > 0.0f ? 1.0f / d : 0.0f;
    b->d = dh;
    for (int j = 0; j < QK_ML8_WIRE / 8; ++j) {
        b->qh[j] = 0;
    }
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        const int idx = ml8_nearest(v[j] * inv, ML8_5_CENTROIDS, ML8_5_N_CENT);
        if (j % 2 == 0) {
            b->qs[j / 2] = (uint8_t) (idx & 0xF);
        } else {
            b->qs[j / 2] |= (uint8_t) ((idx & 0xF) << 4);
        }
        if (idx & 0x10) {
            b->qh[j / 8] |= (uint8_t) (1u << (j % 8));
        }
    }
}

static __device__ __forceinline__ void ml8_5_dequantize(const block_ml8_5_wire * b, float * v) {
    const float d = __half2float(b->d);
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        const int lo = (j % 2 == 0) ? (b->qs[j / 2] & 0xF) : (b->qs[j / 2] >> 4);
        const int hi = (b->qh[j / 8] >> (j % 8)) & 1;
        v[j] = ML8_5_CENTROIDS[lo | (hi << 4)] * d;
    }
}

// ---- ml8-6 -------------------------------------------------------------------
// Same block/scale scheme as ml8-4/ml8-5. 6-bit index = 4 low bits (nibble,
// packed exactly like ml8-4/ml8-5's qs) + 2 high bits. ml8-5's high-bit plane
// is a 1-bit-per-element bitmask (qh[QK/8], OR'd in with `1u << (j % 8)`) --
// that scheme is SPECIALISED to exactly 1 high bit and does not generalize to
// 2. Instead qh here packs 2 bits/element, 4 elements per byte, at bit offset
// (j % 4) * 2 -- so it's an independent 2-bit field, not a widened bitmask.
static __device__ __forceinline__ void ml8_6_quantize(const float * v, block_ml8_6_wire * b) {
    const float amax = ml8_block_scale(v);
    const half  dh   = __float2half(amax);
    const float d    = __half2float(dh);
    const float inv  = d > 0.0f ? 1.0f / d : 0.0f;
    b->d = dh;
    for (int j = 0; j < QK_ML8_WIRE / 4; ++j) {
        b->qh[j] = 0;
    }
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        const int idx = ml8_nearest(v[j] * inv, ML8_6_CENTROIDS, ML8_6_N_CENT);
        if (j % 2 == 0) {
            b->qs[j / 2] = (uint8_t) (idx & 0xF);
        } else {
            b->qs[j / 2] |= (uint8_t) ((idx & 0xF) << 4);
        }
        const int hi = (idx >> 4) & 0x3;
        b->qh[j / 4] |= (uint8_t) (hi << ((j % 4) * 2));
    }
}

static __device__ __forceinline__ void ml8_6_dequantize(const block_ml8_6_wire * b, float * v) {
    const float d = __half2float(b->d);
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        const int lo = (j % 2 == 0) ? (b->qs[j / 2] & 0xF) : (b->qs[j / 2] >> 4);
        const int hi = (b->qh[j / 4] >> ((j % 4) * 2)) & 0x3;
        v[j] = ML8_6_CENTROIDS[lo | (hi << 4)] * d;
    }
}

// ---- ml8-8: raw E4M3 per element + fp16 block scale -------------------------
// The ladder's top rung and the "near-lossless" anchor: no codebook at all, the
// full E4M3 lattice per element. 34 bytes / 32 elements = 8.50 bits, i.e. EXACTLY
// q8_0's budget, which makes this the clean bits-vs-format comparison.
typedef struct {
    half    d;                          //  2 bytes: fp16 absmax scale
    uint8_t qs[QK_ML8_WIRE];            // 32 bytes: one E4M3 byte per element
} block_ml8_8_wire;                     // 34 bytes = 8.50 bits/element
static_assert(sizeof(block_ml8_8_wire) == 34, "ml8_8 wire block must be 34 bytes");

static __device__ __forceinline__ uint8_t ml8_f32_to_e4m3(float x) {
    const uint32_t sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (!(a > 0.0f))      return (uint8_t) sign;              // zero / NaN-free input
    if (a >= 448.0f)      return (uint8_t) (sign | 0x7Eu);    // saturate to max finite
    if (a < 0.015625f) {                                       // subnormal: 2^-6 threshold
        const int m = (int) rintf(a / 0.001953125f);           // 2^-9 step
        return (uint8_t) (sign | (uint32_t) min(m, 7));
    }
    int e = (int) floorf(log2f(a));
    e = max(-6, min(e, 8));
    float step = exp2f((float) (e - 3));
    int   m    = (int) rintf(a / step) - 8;                    // implicit leading 1
    if (m > 7) { m = 0; e += 1; }                              // mantissa carry
    if (e > 8) return (uint8_t) (sign | 0x7Eu);
    return (uint8_t) (sign | ((uint32_t) (e + 7) << 3) | (uint32_t) (m & 7));
}

static __device__ __forceinline__ float ml8_e4m3_to_f32(uint8_t b) {
    const int s = (b >> 7) & 1;
    const int e = (b >> 3) & 0xF;
    const int m = b & 0x7;
    const float v = (e == 0) ? (1.0f / 64.0f) * ((float) m / 8.0f)
                             : exp2f((float) (e - 7)) * (1.0f + (float) m / 8.0f);
    return s ? -v : v;
}

static __device__ __forceinline__ void ml8_8_quantize(const float * v, block_ml8_8_wire * b) {
    const float amax = ml8_block_scale(v);
    const half  dh   = __float2half(amax);
    const float d    = __half2float(dh);
    const float inv  = d > 0.0f ? 1.0f / d : 0.0f;
    b->d = dh;
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        b->qs[j] = ml8_f32_to_e4m3(v[j] * inv);
    }
}

static __device__ __forceinline__ void ml8_8_dequantize(const block_ml8_8_wire * b, float * v) {
    const float d = __half2float(b->d);
    for (int j = 0; j < QK_ML8_WIRE; ++j) {
        v[j] = ml8_e4m3_to_f32(b->qs[j]) * d;
    }
}

// ---- ml8-8r: RAW E4M3, no block, no scale ----------------------------------
// One byte per element, 8.00 bits, and crucially NO per-block absmax reduction.
// That reduction is the only expensive part of q8_0 / ml8-8 packing, and it is
// the part a fused matmul epilogue cannot do for free. So this codec measures
// the accuracy a ZERO-COST fused fp8 emission would deliver: if it holds up,
// fusing e4m3 into the matmul epilogue on gfx1201 is worth building; if it does
// not, the whole fusion idea is dead before anyone touches a matmul kernel.
// e4m3 carries its own exponent, so unscaled activations stay in range
// (~2^-9 .. 448) at a flat ~6% relative precision.
static __device__ __forceinline__ void ml8_8r_quantize_elem(float v, uint8_t * out) {
    *out = ml8_f32_to_e4m3(v);
}
static __device__ __forceinline__ float ml8_8r_dequantize_elem(uint8_t b) {
    return ml8_e4m3_to_f32(b);
}
