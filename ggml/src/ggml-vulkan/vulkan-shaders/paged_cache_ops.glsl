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

#endif // PAGED_CACHE_OPS_GLSL
