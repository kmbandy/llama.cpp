// MAD-214 Phase 1G step 1: cooperative turbo-FP8 BS=256 decode helpers for
// the paged-tile attention path (the non-AITER CUDA/HIP attention codepath).
//
// Mirrors the role of coop_stage_turbo4_tile in mt_pagedattn_tile.cu but
// adapts to the turbo-FP8 BS=256 packed layout:
//
//   block_turbo4_fp8_bs256 (162 bytes per (token, kv_head)):
//     bytes [0..1]    : fp16 per-block scale
//     bytes [2..129]  : 128 bytes = 256 × 4-bit centroid indices
//     bytes [130..161]: 32 bytes  = 256 × 1-bit sign bits
//
//   Per (token, kv_head) there is ONE block covering the full HEAD_SIZE=256
//   row — so K_TILE_N=16 tokens means 16 qblocks per tile (not 16×2 like
//   turbo4_0 with Q_BLOCK=128).
//
// Critical difference from the existing turbo4 helper: the centroid LUT is
// per-(kv, layer) calibrated at training time and passed in as a runtime
// pointer (16 E4M3 bytes), not the compile-time global TURBO_CENTROIDS_4BIT.
// This decoder accepts the LUT pointer as an argument and decodes E4M3
// inline per-thread (16 centroids per lane is small enough to avoid an
// smem stage).

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

namespace mt_turbo_fp8 {

constexpr int    HEAD_SIZE_BS256        = 256;
constexpr int    N_CENTROIDS_TURBO4_FP8 = 16;
constexpr size_t BYTES_PER_BS256_BLOCK  = 162;

// E4M3 (no NaN) byte → fp32 — same mapping as the CPU packer in
// tests/test_aiter_turbo_fp8_smoke.cpp. Lane-local, no LDS.
static __device__ __forceinline__ float e4m3_to_fp32(uint8_t b) {
    int sign = (b >> 7) & 1;
    int e    = (b >> 3) & 0xF;
    int m    = b & 0x7;
    float v;
    if (e == 0) v = (1.0f / 64.0f) * (m / 8.0f);
    else        v = __builtin_amdgcn_ldexp(1.0f + m / 8.0f, e - 7);
    return sign ? -v : v;
}

// Cooperative decode of K_TILE_N tokens × HEAD_SIZE columns into smem.
//
// Layout assumed in `cache`:
//   [num_blocks, BLOCK_SIZE, num_kv_heads, BYTES_PER_BS256_BLOCK] (byte-strided)
// which matches the AITER paged-cache layout from the FP8 perf/correctness
// tests, so the same on-disk format works for both attention codepaths.
//
// Each warp processes ~K_TILE_N / N_WARPS qblocks. Per qblock (one (token,
// kv_head) row of 256 elements): 32 lanes × 8 elements = 256 covered.
//
// Per lane:
//   - Reads 4 bytes of qs (= 8 4-bit indices = 8 elements)
//   - Reads 1 byte of signs (8 sign bits — this lane covers the right
//     8-element window starting at lane_id * 8)
//   - Lane 0 reads the 2-byte fp16 scale and broadcasts via __shfl_sync
//
// Centroid LUT is shared across all qblocks for this (K vs V, layer); the
// caller passes either the K or V LUT pointer. 16 bytes is small enough
// that each thread re-fetches inline without a coordinated smem stage —
// the cost is dominated by the per-element decode.
template <int HEAD_SIZE, int BLOCK_SIZE, int N_WARPS, int K_TILE_N>
static __device__ __forceinline__ void coop_stage_turbo4_fp8_bs256_tile(
        __half        * __restrict__ smem_dst,        // [K_TILE_N, HEAD_SIZE]
        const void    * __restrict__ cache,            // packed bytes, AITER layout
        const uint8_t * __restrict__ centroids,        // 16 E4M3 bytes
        const int     * __restrict__ seq_block_table,
        int            k_tile_start,
        int            block_valid_ctx,
        int            kv_head_idx,
        int            n_kv_heads,
        int            warp_id,
        int            lane_id) {
    static_assert(HEAD_SIZE == 256, "turbo-FP8 BS=256 helper requires HEAD_SIZE=256");

    constexpr int Q_BLOCK            = HEAD_SIZE_BS256;     // one qblock per token
    constexpr int QBLOCKS_PER_TOKEN  = HEAD_SIZE / Q_BLOCK; // = 1
    constexpr int N_QBLOCKS_PER_TILE = K_TILE_N * QBLOCKS_PER_TOKEN;
    constexpr int ELEMS_PER_LANE     = Q_BLOCK / 32;        // 8 elements/lane

    const uint8_t * cache_bytes = (const uint8_t *) cache;

    // Each lane decodes its own copy of the LUT (16 fp32 values, 64 bytes
    // in registers). Cheaper than coordinating an smem stage for 16 bytes.
    float lut[N_CENTROIDS_TURBO4_FP8];
    #pragma unroll
    for (int k = 0; k < N_CENTROIDS_TURBO4_FP8; ++k) {
        lut[k] = e4m3_to_fp32(centroids[k]);
    }

    #pragma unroll
    for (int qb = warp_id; qb < N_QBLOCKS_PER_TILE; qb += N_WARPS) {
        const int row    = qb;  // QBLOCKS_PER_TOKEN == 1 ⇒ qb is the row idx
        const int token  = k_tile_start + row;

        const uint8_t * blk_bytes = nullptr;
        float scale_f = 0.0f;

        if (token < block_valid_ctx) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            // Matches the existing turbo4 convention: -1 sentinel means
            // unmapped block, decode to zeros.
            if (physical >= 0) {
                const int64_t blk_idx =
                      (int64_t) physical * BLOCK_SIZE * n_kv_heads
                    + (int64_t) tok_in_block * n_kv_heads
                    + (int64_t) kv_head_idx;
                blk_bytes = cache_bytes + blk_idx * BYTES_PER_BS256_BLOCK;
                if (lane_id == 0) {
                    __half h;
                    // Reinterpret first 2 bytes as fp16.
                    __builtin_memcpy(&h, blk_bytes, sizeof(__half));
                    scale_f = __half2float(h);
                }
            }
        }

        // Broadcast scale from lane 0 to all 32 lanes of this warp.
        // 64-bit mask literal required on RDNA4 (gfx12, wave32) per
        // amd_warp_sync_functions.h static_assert.
        scale_f = __shfl_sync(0xFFFFFFFFFFFFFFFFull, scale_f, 0, WARP_SIZE);

        // Read this lane's 4-byte qs window (8 nibbles) and 1-byte signs window.
        uint32_t qs_word = 0;
        uint8_t  signs_b = 0;
        if (blk_bytes != nullptr) {
            qs_word = *(const uint32_t *)(blk_bytes + 2 + lane_id * 4);
            signs_b = blk_bytes[2 + 128 + lane_id];  // signs start at byte 130
        }

        const int smem_row_base = row * HEAD_SIZE;
        const int smem_col_base = lane_id * ELEMS_PER_LANE;

        #pragma unroll
        for (int l = 0; l < ELEMS_PER_LANE; ++l) {
            const int idx = (qs_word >> (l * 4)) & 0xF;
            const int s   = (signs_b >> l) & 1;
            float val = lut[idx] * scale_f;
            if (s) val = -val;
            smem_dst[smem_row_base + smem_col_base + l] = __float2half(val);
        }
    }
}

}  // namespace mt_turbo_fp8
