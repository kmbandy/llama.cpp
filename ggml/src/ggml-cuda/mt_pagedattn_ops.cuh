// mt_pagedattn_ops — per-cache-type layout + load/store helpers for the
// paged K/V cache. Shared by mt_pagedattn.cu (existing scalar kernel) and
// mt_pagedattn_tile.cu (WMMA tile kernel).
//
// Encapsulates offset math, dequant on read, and quantize on write for
// the paged K/V cache. The kernels call these statically; actual cache
// layout is specialization-private. Kernel math stays cache-type agnostic
// — adding a new quant type means adding a specialization here, not
// touching the hot kernel body.
//
// Layouts per (paged_block, kv_head):
//
//   F16 K:       [HEAD_SIZE/K_X, BLOCK_SIZE, K_X]   K_X = 16 / sizeof(__half) = 8
//                (X-stride coalesces F16 reads; original mt:: layout)
//   F16 V:       [HEAD_SIZE, BLOCK_SIZE]
//                (head_dim-major; 16 contiguous tokens per fixed d)
//   Q8_0 K/V:    [BLOCK_SIZE, HEAD_SIZE/QK8_0]      of block_q8_0
//                (token-first; each token holds HEAD_SIZE/32 q8_0 blocks
//                 contiguously — natural for per-element dequant)
//   Turbo4 K/V:  [BLOCK_SIZE, HEAD_SIZE/QK_TURBO4]  of block_turbo4_0
//                (same shape as Q8_0; HEAD_SIZE/128 = 2 turbo4 blocks per token)

#pragma once

#include "common.cuh"
#include "turbo-quant.cuh"

namespace mt {

// Primary template — instantiations fail to compile if a needed type
// isn't specialized (lookups force linker error rather than silent fallback).
template <ggml_type T, int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops;

// F16 specialization: existing layout, identity load/store via __half.
template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops<GGML_TYPE_F16, HEAD_SIZE, BLOCK_SIZE> {
    static constexpr int K_X = 16 / sizeof(__half);  // 8 — F16 16-byte coalesce stride
    static_assert(HEAD_SIZE % K_X == 0, "HEAD_SIZE must be divisible by K_X for F16 layout");
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");

    __device__ __forceinline__ static float k_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const __half * k = (const __half *) buf;
        const int dim_outer = d / K_X;
        const int dim_inner = d % K_X;
        const size_t off = ((size_t) paged_block * n_kv_heads + kv_head) * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                         + (size_t) dim_outer * BLOCK_SIZE * K_X
                         + (size_t) token_in_block * K_X
                         + (size_t) dim_inner;
        return (float) k[off];
    }

    __device__ __forceinline__ static float v_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const __half * v = (const __half *) buf;
        const size_t off = ((size_t) paged_block * n_kv_heads + kv_head) * HEAD_SIZE * BLOCK_SIZE
                         + (size_t) d * BLOCK_SIZE
                         + (size_t) token_in_block;
        return (float) v[off];
    }

    // Scatter: store one element of K and one of V. Element-granularity store
    // is correct for F16 (no per-block scale). Quant specializations override
    // this to do per-quant-block cooperative quantization.
    __device__ __forceinline__ static void kv_store(
            void * k_buf, void * v_buf,
            int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d,
            float k_val, float v_val) {
        __half * k = (__half *) k_buf;
        __half * v = (__half *) v_buf;
        const int dim_outer = d / K_X;
        const int dim_inner = d % K_X;
        const size_t k_off = ((size_t) paged_block * n_kv_heads + kv_head) * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                           + (size_t) dim_outer * BLOCK_SIZE * K_X
                           + (size_t) token_in_block * K_X
                           + (size_t) dim_inner;
        const size_t v_off = ((size_t) paged_block * n_kv_heads + kv_head) * HEAD_SIZE * BLOCK_SIZE
                           + (size_t) d * BLOCK_SIZE
                           + (size_t) token_in_block;
        k[k_off] = (__half) k_val;
        v[v_off] = (__half) v_val;
    }
};

// Q8_0 specialization: 8-bit symmetric quant, 32-element blocks.
// kv_store intentionally omitted — Q8_0 quantization needs cooperative
// per-block scale, so scatter for Q8_0 is handled by a dedicated kernel
// (mt_scatter_kv_q8_0_kernel) rather than a per-element store.
template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops<GGML_TYPE_Q8_0, HEAD_SIZE, BLOCK_SIZE> {
    static constexpr int Q_BLOCK = QK8_0;  // 32
    static constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK8_0");

    __device__ __forceinline__ static int64_t element_block_index(
            int paged_block, int kv_head, int n_kv_heads, int token_in_block, int d) {
        return ((int64_t) paged_block * n_kv_heads + kv_head) * BLOCK_SIZE * N_QBLOCKS_PER_TOKEN
             + (int64_t) token_in_block * N_QBLOCKS_PER_TOKEN
             + (int64_t) (d / Q_BLOCK);
    }

    __device__ __forceinline__ static float k_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const block_q8_0 * blocks = (const block_q8_0 *) buf;
        const int64_t ib = element_block_index(paged_block, kv_head, n_kv_heads, token_in_block, d);
        const int     iqs = d % Q_BLOCK;
        const float   d_scale = (float) blocks[ib].d;
        return (float) blocks[ib].qs[iqs] * d_scale;
    }

    __device__ __forceinline__ static float v_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        return k_load(buf, paged_block, kv_head, n_kv_heads, token_in_block, d);
    }
};

// Turbo4_0 specialization: 4-bit PolarQuant with WHT rotation, 128-element blocks.
// kv_store omitted — handled by a dedicated cooperative scatter kernel.
template <int HEAD_SIZE, int BLOCK_SIZE>
struct paged_cache_ops<GGML_TYPE_TURBO4_0, HEAD_SIZE, BLOCK_SIZE> {
    static constexpr int Q_BLOCK = QK_TURBO4;  // 128
    static constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO4");

    __device__ __forceinline__ static int64_t element_block_index(
            int paged_block, int kv_head, int n_kv_heads, int token_in_block, int d) {
        return ((int64_t) paged_block * n_kv_heads + kv_head) * BLOCK_SIZE * N_QBLOCKS_PER_TOKEN
             + (int64_t) token_in_block * N_QBLOCKS_PER_TOKEN
             + (int64_t) (d / Q_BLOCK);
    }

    __device__ __forceinline__ static float k_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        const block_turbo4_0 * blocks = (const block_turbo4_0 *) buf;
        const int64_t ib  = element_block_index(paged_block, kv_head, n_kv_heads, token_in_block, d);
        const int     iqs = d % Q_BLOCK;
        const float   norm = __half2float(blocks[ib].norm);
        return turbo4_dequant_element(&blocks[ib], iqs, norm);
    }

    __device__ __forceinline__ static float v_load(
            const void * buf, int paged_block, int kv_head, int n_kv_heads,
            int token_in_block, int d) {
        return k_load(buf, paged_block, kv_head, n_kv_heads, token_in_block, d);
    }
};

} // namespace mt
