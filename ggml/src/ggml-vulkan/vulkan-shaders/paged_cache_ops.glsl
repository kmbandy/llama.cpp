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

#if defined(DATA_A_TURBO4_0) || defined(DATA_A_TURBO4_64) || defined(DATA_A_TURBO4_64_OL)
// Shared between turbo4_0 (128-element blocks), turbo4_64 (64-element
// blocks), and turbo4_64_ol (64-element blocks, fixed outlier channels):
// centroid table + nearest-centroid helper.
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
#endif // DATA_A_TURBO4_0 || DATA_A_TURBO4_64 || DATA_A_TURBO4_64_OL

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

#ifdef DATA_A_TURBO4_64_OL
// turbo4_64_ol family (SP2.5, 2026-07-01; OL8/OL12 added 2026-07-01 for the
// outlier-matrix sweep): turbo4_64 with PA_TURBO4_64_OL_N fixed-position
// "massive activation" outlier channels extracted verbatim at f16 and
// excluded from the group-norm/centroid quant of the remaining
// (64 - PA_TURBO4_64_OL_N) elements. Mirrors
// paged_cache_ops<GGML_TYPE_TURBO4_64_OL/_OL8/_OL12> (mt_pagedattn_ops.cuh).
// Uses the SHARED TURBO_CENTROIDS_4BIT table (pa_turbo_nearest_4bit, above)
// by default — NOT the N=64-calibrated one — since removing the outliers
// from the norm makes the remaining "typical" values close enough to the
// N=128 assumption for that table to be appropriate again.
// GGML_TURBO4_64_OL_TABLE=n64 (read once on the host, forwarded as the
// p.use_n64_table push constant; see ggml_vk_turbo4_64_ol_use_n64_table in
// ggml-vulkan.cpp / ggml_cuda_turbo4_64_ol_use_n64_table in turbo-quant.cuh)
// can select the N=64-calibrated table instead — investigative-only,
// applies uniformly to N=4/8/12.
//
// This single shader source is instantiated three times (ol/ol8/ol12, see
// vulkan-shaders-gen.cpp) via PA_TURBO4_64_OL_N / PA_TURBO4_64_OL_CHANNELS_INIT
// preprocessor defines; the defaults below reproduce the original N=4
// behavior so the existing turbo4_64_ol registration doesn't need to pass
// them explicitly.
#define PA_QK64_OL 64u
#ifndef PA_TURBO4_64_OL_N
#define PA_TURBO4_64_OL_N 4u
#endif
#define PA_QK64_OL_N_OUTLIERS PA_TURBO4_64_OL_N
#ifndef PA_TURBO4_64_OL_CHANNELS_INIT
#define PA_TURBO4_64_OL_CHANNELS_INIT 53u,49u,52u,20u
#endif
// Fixed outlier channel positions — MUST stay byte-for-byte identical to
// TURBO4_64_OUTLIER_CHANNELS/TURBO4_64_OL8_OUTLIER_CHANNELS/
// TURBO4_64_OL12_OUTLIER_CHANNELS in ggml-common.h / TURBO4_64_OL_CHANNELS
// in turbo-quant.cuh.
const uint PA_TURBO4_64_OL_CHANNELS[PA_TURBO4_64_OL_N] = uint[PA_TURBO4_64_OL_N](PA_TURBO4_64_OL_CHANNELS_INIT);

// ---- runtime centroid-table toggle helpers (investigative-only) ----
// Local copy of the N=64-calibrated nearest-centroid search (duplicated
// rather than shared with the DATA_A_TURBO4_64 block above, since that
// block is compiled only for the plain turbo4_64 cache type and must stay
// untouched). TURBO_CENTROIDS_4BIT_N64 / TURBO_MID_4BIT_N64 come from
// turbo_centroids.glsl, included unconditionally above.
uint pa_turbo_nearest_4bit_n64(float v) {
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
uint pa_turbo_nearest_4bit_sel(float v, bool use_n64) {
    return use_n64 ? pa_turbo_nearest_4bit_n64(v) : pa_turbo_nearest_4bit(v);
}
float pa_turbo_centroid_4bit_sel(uint idx, bool use_n64) {
    return use_n64 ? TURBO_CENTROIDS_4BIT_N64[idx] : TURBO_CENTROIDS_4BIT[idx];
}

// Returns true (and sets outlier_slot) if channel iqs (0..63) is one of the
// PA_TURBO4_64_OL_N fixed outlier positions; otherwise returns false and
// sets nib to the packed-nibble index (0..(64-PA_TURBO4_64_OL_N-1)) among
// non-outlier channels in ascending-d order (must match the scatter
// shader's packing order).
bool pa_turbo64_ol_classify(uint iqs, out uint outlier_slot, out uint nib) {
    uint n_lt = 0u;
    for (uint o = 0u; o < PA_QK64_OL_N_OUTLIERS; o++) {
        if (PA_TURBO4_64_OL_CHANNELS[o] == iqs) {
            outlier_slot = o;
            return true;
        }
        if (PA_TURBO4_64_OL_CHANNELS[o] < iqs) n_lt++;
    }
    nib = iqs - n_lt;
    return false;
}

uint pa_turbo64_ol_block_index(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint qb, uint HS, uint BS) {
    const uint N_QBLK = HS / PA_QK64_OL;
    return ((paged_block*n_kv_heads + kv_head) * BS * N_QBLK) + tok*N_QBLK + qb;
}
// Encode the load location as block_ib*64 + iqs (same convention as the
// other turbo4 variants) so the type-generic call site is unchanged.
uint pa_k_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    const uint block_ib = pa_turbo64_ol_block_index(paged_block, kv_head, n_kv_heads, tok, d/PA_QK64_OL, HS, BS);
    return block_ib * PA_QK64_OL + (d % PA_QK64_OL);
}
uint pa_v_off(uint paged_block, uint kv_head, uint n_kv_heads, uint tok, uint d, uint HS, uint BS) {
    return pa_k_off(paged_block, kv_head, n_kv_heads, tok, d, HS, BS);
}
float pa_k_load(uint off) {
    const uint ib  = off / PA_QK64_OL;
    const uint iqs = off % PA_QK64_OL;
    uint outlier_slot, nib;
    if (pa_turbo64_ol_classify(iqs, outlier_slot, nib)) {
        return float(data_k[ib].outliers[outlier_slot]);
    }
    const uint idx = (uint(data_k[ib].qs[nib >> 1u]) >> ((nib & 1u) * 4u)) & 0xFu;
    return pa_turbo_centroid_4bit_sel(idx, p.use_n64_table != 0u) * float(data_k[ib].norm);
}
float pa_v_load(uint off) {
    const uint ib  = off / PA_QK64_OL;
    const uint iqs = off % PA_QK64_OL;
    uint outlier_slot, nib;
    if (pa_turbo64_ol_classify(iqs, outlier_slot, nib)) {
        return float(data_v[ib].outliers[outlier_slot]);
    }
    const uint idx = (uint(data_v[ib].qs[nib >> 1u]) >> ((nib & 1u) * 4u)) & 0xFu;
    return pa_turbo_centroid_4bit_sel(idx, p.use_n64_table != 0u) * float(data_v[ib].norm);
}
#endif // DATA_A_TURBO4_64_OL

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
