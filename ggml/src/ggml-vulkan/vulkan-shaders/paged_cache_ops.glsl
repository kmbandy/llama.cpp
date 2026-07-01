// paged_cache_ops.glsl — GLSL analog of mt_pagedattn_ops.cuh. Cache-type load/store.
// Selected by DATA_A_F16 / DATA_A_TURBO4_0 (Task 4 adds the turbo4_0 block).
//
// The helpers reference the storage buffers `data_k` / `data_v`, which MUST be
// declared by the including shader (paged_attn_scatter.comp / paged_attn.comp)
// with matching binding points. F16 store/load is element-granular (identity);
// the scatter for F16 is a pure permutation copy — no Hadamard/RHT (that is a
// Task 4 turbo4_0 concern).

#ifndef PAGED_CACHE_OPS_GLSL
#define PAGED_CACHE_OPS_GLSL

#ifdef DATA_A_F16
#define PA_KX 8u                                  // 16 / sizeof(f16)
// K: [HEAD_SIZE/KX, BLOCK_SIZE, KX]; off = base + (d/KX)*BS*KX + tok*KX + (d%KX)
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return ((paged_block*n_kv_heads + kv_head) * (HS/PA_KX) * BS * PA_KX)
         + (d/PA_KX)*BS*PA_KX + tok*PA_KX + (d%PA_KX);
}
// V: [HEAD_SIZE, BLOCK_SIZE]; off = base + d*BS + tok
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return ((paged_block*n_kv_heads + kv_head) * HS * BS) + d*BS + tok;
}
float pa_k_load(uint off) { return float(data_k[off]); }   // data_k = f16 cache buffer
float pa_v_load(uint off) { return float(data_v[off]); }
void  pa_k_store(uint off, float val) { data_k[off] = float16_t(val); }
void  pa_v_store(uint off, float val) { data_v[off] = float16_t(val); }
#endif // DATA_A_F16

#if defined(DATA_A_TURBO4_0) || defined(DATA_A_TURBO4_64)
// Shared between turbo4_0 (128-element blocks) and turbo4_64 (64-element
// blocks): centroid table + nearest-centroid helper.
#include "turbo_centroids.glsl"

// Nearest 4-bit centroid (linear midpoint ladder; matches
// turbo_nearest_centroid_4bit in turbo-quant.cuh).
uint pa_turbo_nearest_4bit(float v) {
    if      (v < TURBO_MID_4BIT[ 0]) return  0u;
    else if (v < TURBO_MID_4BIT[ 1]) return  1u;
    else if (v < TURBO_MID_4BIT[ 2]) return  2u;
    else if (v < TURBO_MID_4BIT[ 3]) return  3u;
    else if (v < TURBO_MID_4BIT[ 4]) return  4u;
    else if (v < TURBO_MID_4BIT[ 5]) return  5u;
    else if (v < TURBO_MID_4BIT[ 6]) return  6u;
    else if (v < TURBO_MID_4BIT[ 7]) return  7u;
    else if (v < TURBO_MID_4BIT[ 8]) return  8u;
    else if (v < TURBO_MID_4BIT[ 9]) return  9u;
    else if (v < TURBO_MID_4BIT[10]) return 10u;
    else if (v < TURBO_MID_4BIT[11]) return 11u;
    else if (v < TURBO_MID_4BIT[12]) return 12u;
    else if (v < TURBO_MID_4BIT[13]) return 13u;
    else if (v < TURBO_MID_4BIT[14]) return 14u;
    else                             return 15u;
}
#endif // DATA_A_TURBO4_0 || DATA_A_TURBO4_64

#ifdef DATA_A_TURBO4_0
// turbo4_0: 4-bit PolarQuant, 128-element blocks. Mirrors
// paged_cache_ops<GGML_TYPE_TURBO4_0> (mt_pagedattn_ops.cuh:123-153).
// RHT-FREE: dequant = TURBO_CENTROIDS_4BIT[idx] * norm (un-rotated K) — the
// paged path stores/reads K in the un-rotated domain, so <K,Q> is exact.
#define PA_QK 128u

// turbo4 block index: [(paged_block*n_kv_heads + kv_head)*BS*N_QBLK + tok*N_QBLK + qb].
uint pa_turbo_block_index(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint qb, uint HS, uint BS) {
    const uint N_QBLK = HS / PA_QK;
    return ((paged_block*n_kv_heads + kv_head) * BS * N_QBLK) + tok*N_QBLK + qb;
}

// Encode the load location as a single element offset = block_ib*128 + iqs, so
// the type-generic call site (paged_attn.comp: pa_k_load(pa_k_off(...))) is
// unchanged. K and V share the same turbo4 layout.
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    const uint block_ib = pa_turbo_block_index(paged_block, kv_head, n_kv_heads, tok, d/PA_QK, HS, BS);
    return block_ib * PA_QK + (d % PA_QK);
}
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return pa_k_off(paged_block, kv_head, n_kv_heads, tok, d, HS, BS);
}
float pa_k_load(uint off) {
    const uint ib  = off / PA_QK;
    const uint iqs = off % PA_QK;
    const uint idx = (uint(data_k[ib].qs[iqs >> 1u]) >> ((iqs & 1u) * 4u)) & 0xFu;
    return TURBO_CENTROIDS_4BIT[idx] * float(data_k[ib].norm);
}
float pa_v_load(uint off) {
    const uint ib  = off / PA_QK;
    const uint iqs = off % PA_QK;
    const uint idx = (uint(data_v[ib].qs[iqs >> 1u]) >> ((iqs & 1u) * 4u)) & 0xFu;
    return TURBO_CENTROIDS_4BIT[idx] * float(data_v[ib].norm);
}
#endif // DATA_A_TURBO4_0

#ifdef DATA_A_TURBO4_64
// turbo4_64: 4-bit PolarQuant, 64-element block (34 B: norm + qs[32], NO rnorm).
// RHT-FREE: dequant = TURBO_CENTROIDS_4BIT_N64[idx] * norm (N=64-calibrated
// table, NOT the shared turbo4_0 one — see turbo_centroids.glsl comment).
#define PA_QK64 64u

// Nearest 4-bit centroid using the N=64-calibrated table/midpoints.
uint pa_turbo64_nearest_4bit(float v) {
    if      (v < TURBO_MID_4BIT_N64[ 0]) return  0u;
    else if (v < TURBO_MID_4BIT_N64[ 1]) return  1u;
    else if (v < TURBO_MID_4BIT_N64[ 2]) return  2u;
    else if (v < TURBO_MID_4BIT_N64[ 3]) return  3u;
    else if (v < TURBO_MID_4BIT_N64[ 4]) return  4u;
    else if (v < TURBO_MID_4BIT_N64[ 5]) return  5u;
    else if (v < TURBO_MID_4BIT_N64[ 6]) return  6u;
    else if (v < TURBO_MID_4BIT_N64[ 7]) return  7u;
    else if (v < TURBO_MID_4BIT_N64[ 8]) return  8u;
    else if (v < TURBO_MID_4BIT_N64[ 9]) return  9u;
    else if (v < TURBO_MID_4BIT_N64[10]) return 10u;
    else if (v < TURBO_MID_4BIT_N64[11]) return 11u;
    else if (v < TURBO_MID_4BIT_N64[12]) return 12u;
    else if (v < TURBO_MID_4BIT_N64[13]) return 13u;
    else if (v < TURBO_MID_4BIT_N64[14]) return 14u;
    else                                  return 15u;
}

uint pa_turbo64_block_index(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint qb, uint HS, uint BS) {
    const uint N_QBLK = HS / PA_QK64;
    return ((paged_block*n_kv_heads + kv_head) * BS * N_QBLK) + tok*N_QBLK + qb;
}
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    const uint block_ib = pa_turbo64_block_index(paged_block, kv_head, n_kv_heads, tok, d/PA_QK64, HS, BS);
    return block_ib * PA_QK64 + (d % PA_QK64);
}
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return pa_k_off(paged_block, kv_head, n_kv_heads, tok, d, HS, BS);
}
float pa_k_load(uint off) {
    const uint ib = off / PA_QK64; const uint iqs = off % PA_QK64;
    const uint idx = (uint(data_k[ib].qs[iqs >> 1u]) >> ((iqs & 1u) * 4u)) & 0xFu;
    return TURBO_CENTROIDS_4BIT_N64[idx] * float(data_k[ib].norm);
}
float pa_v_load(uint off) {
    const uint ib = off / PA_QK64; const uint iqs = off % PA_QK64;
    const uint idx = (uint(data_v[ib].qs[iqs >> 1u]) >> ((iqs & 1u) * 4u)) & 0xFu;
    return TURBO_CENTROIDS_4BIT_N64[idx] * float(data_v[ib].norm);
}
#endif // DATA_A_TURBO4_64

#ifdef DATA_A_Q8_0
// Q8_0: standard 8-bit symmetric per-32-element-block quantization. Mirrors
// paged_cache_ops<GGML_TYPE_Q8_0> (mt_pagedattn_ops.cuh:89-121). Plain
// per-block scale (no zero-point, no centroid table, no norm-correction) —
// much simpler than turbo4_0/turbo4_64.
#define PA_QK8_0 32u

uint pa_q80_block_index(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint qb, uint HS, uint BS) {
    const uint N_QBLOCKS_PER_TOKEN = HS / PA_QK8_0;
    return ((paged_block*n_kv_heads + kv_head) * BS * N_QBLOCKS_PER_TOKEN) + tok*N_QBLOCKS_PER_TOKEN + qb;
}
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    const uint block_ib = pa_q80_block_index(paged_block, kv_head, n_kv_heads, tok, d/PA_QK8_0, HS, BS);
    return block_ib * PA_QK8_0 + (d % PA_QK8_0);
}
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return pa_k_off(paged_block, kv_head, n_kv_heads, tok, d, HS, BS);
}
float pa_k_load(uint off) {
    const uint ib  = off / PA_QK8_0;
    const uint iqs = off % PA_QK8_0;
    return float(data_k[ib].qs[iqs]) * float(data_k[ib].d);
}
float pa_v_load(uint off) {
    const uint ib  = off / PA_QK8_0;
    const uint iqs = off % PA_QK8_0;
    return float(data_v[ib].qs[iqs]) * float(data_v[ib].d);
}
#endif // DATA_A_Q8_0

#endif // PAGED_CACHE_OPS_GLSL
