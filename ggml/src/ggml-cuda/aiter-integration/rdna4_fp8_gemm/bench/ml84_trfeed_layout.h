// bench/ml84_trfeed_layout.h — MAD-305 ML8_4 (4.5 bpw) packed weight layout
// for the frozen fp8 trfeed WMMA kernel family (trfeed_kernels.h).
//
// Motivation (see the task's measured numbers, RESULT.md-style): the frozen
// fp8 `gemm_fp8_trfeed<32,1>` decode tile streams 89 MB of B at 506 GB/s
// (memory-bound) because B is one fp8 byte/element. ML8_4 stores B as a
// 4-bit centroid INDEX/element (+ a per-(64-K-group, N) fp32 scale + a tiny
// per-group 16-entry e4m3 LUT), i.e. ~4.5 bits/weight vs 8 — the whole point
// of this file is to define a byte layout for those indices that (a) a
// plain (non-transposing) global load can read directly, one dword per
// lane, matching exactly the operand grouping the frozen kernel's
// `global_load_tr_b64` fragment load produces, so a cheap LUT expand
// (trfeed_ml84_kernels.h) rebuilds the identical v2i32 fp8 fragment the
// frozen WMMA consumes, and (b) is a PROVABLY-the-same permutation as the
// existing fp8 `B_shuf` tile layout (trfeed_common.h / gemm_blockscale.hip's
// bs_trperm), so the prefill path can cheaply re-expand ML8_4 straight into
// `B_shuf` and reuse the unchanged frozen `gemm_fp8_trfeed<128,2>` kernel
// (see rdna4_expand_ml84_to_trfeed below).
//
// Nothing in this header touches gfx12-only intrinsics (WMMA / tr-load) —
// it is pure byte-address arithmetic + e4m3 software conversion, so (like
// gemm_blockscale.hip's bs_trperm/preshuffle kernels) it compiles for every
// offload arch, host included.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

#include "trfeed_common.h"           // b_tile_offset(kt,nt,NT), trperm(L,s) — the frozen fp8 B_shuf addressing

// ggml-common.h's struct/macro section is gated on one of the
// GGML_COMMON_DECL_* macros being defined before it's included (see
// ggml-cuda/common.cuh's own `#define GGML_COMMON_DECL_HIP` /
// `#include "ggml-common.h"` pair) -- without one of those defined first,
// the whole file compiles to an empty shell (no block_ml8_4, no QK_ML8).
// This directory's .hip files are always a HIP compile (no dependency on
// the ggml build's GGML_USE_HIP switch), so define it unconditionally; only
// define it if some other TU hasn't already (defends against multiple
// translation units in the same link step, e.g. via a future ggml.h include).
#ifndef GGML_COMMON_DECL_HIP
#define GGML_COMMON_DECL_HIP
#endif
#include "../../../ggml-common.h"    // block_ml8_4, QK_ML8 (=64, the on-disk ml8-4 K-group width)

// ─────────────────────────────────────────────────────────────────────────
// (kl, nl) <-> (lane L, slot s) within one 16(K) x 16(N) tile.
//
// IDENTICAL formula to gemm_blockscale.hip's `bs_lane_slot` / the inverse of
// trfeed_common.h's `preshuffle_B` forward rule (kl = ((L>>4)&1)*8+s,
// nl = L&15). Re-deriving it here (rather than #include-ing a .hip TU) keeps
// this header self-contained; it is copied verbatim, not reinvented, so the
// B_nib layout below and the fp8 B_shuf layout provably assign the SAME
// (k,n) element to the SAME lane L / slot s of the SAME tile.
// ─────────────────────────────────────────────────────────────────────────
__host__ __device__ inline void ml84_lane_slot(int kl, int nl, int* L, int* s) {
    *L = nl | (((kl >> 3) & 1) << 4);
    *s = kl & 7;
}

// Byte address, within the (kt,nt) tile of an NT-wide tile grid, of the
// FIRST of the 4 bytes lane L's dword occupies in the B_nib buffer (see the
// "ML84_TRFEED B_nib" contract below for why it's 4, not 8, bytes/lane).
__host__ __device__ inline size_t ml84_tile_nib_base(int kt, int nt, int NT) {
    return b_tile_offset(kt, nt, NT) / 2;   // 128 nibble-bytes per 16x16 tile (256 fp8 bytes / 2)
}

// ─────────────────────────────────────────────────────────────────────────
// ML84_TRFEED packed layout.
//
// (a) B_nib: the frozen fp8 B_shuf layout (same b_tile_offset(kt,nt,NT)
//     tile addressing) with each fp8 byte replaced by its 4-bit LUT index,
//     two indices packed per byte. Unlike B_shuf (consumed by the
//     hardware-transposing `global_load_tr_b64`, so its byte at trperm(L,s)
//     is scattered across the whole 256-byte tile), B_nib is read by a
//     PLAIN load: the decode kernel already knows, at compile time, which
//     lane wants which 8 elements (slots s=0..7, same (kl,nl) assignment as
//     the fp8 path via ml84_lane_slot), so B_nib packs those 8 indices
//     CONTIGUOUSLY at that lane's own 4-byte dword — elements s=0..3 as the
//     LOW nibbles of the dword's 4 bytes, elements s=4..7 as the HIGH
//     nibbles (matching the frozen kernel's tr_load8 v2i32{x,y} split:
//     x = bytes 0-3, y = bytes 4-7 — see trfeed_ml84_kernels.h). So:
//       nib_dword_addr(kt,nt,NT,L) = ml84_tile_nib_base(kt,nt,NT) + L*4
//     and B_nib is N*K/2 bytes total (half the fp8 B_shuf's N*K).
// (b) b_scale_g: fp32 [K/64][N] row-major, b_scale_g[g*N+n] = scale for
//     column n, K-group g (g = k/64, QK_ML8=64 groups). This is EXACTLY
//     what ml8.cu's `ggml_cuda_ml8_repack_blocks` already produces (its
//     `dst_b_scale` output, `(n_groups_k, N)` row-major) — kept unchanged,
//     just reused as this layout's scale table.
// (c) lut: F8_E4M3 [K/64][16], 16 bytes/group, as stored by the model
//     (ggml_type F8_E4M3 sidecar tensor, see ggml-turbo-quant.c's
//     dequantize_row_ml8_4_with_lut). Values are OCP e4m3 (torch
//     float8_e4m3fn) by construction of the calibration pipeline, |c| <= 1.
// ─────────────────────────────────────────────────────────────────────────

// Nibble-index bijection: 0..N*K-1 <-> B_nib's nibble stream. pos = 2*byte
// offset + (0 = low nibble, 1 = high nibble). Used only for the host/bench
// round-trip self-check (ml84_trfeed_pos_to_nk vs. this forward map); the
// production packer/expander kernels iterate directly over (n,k) and don't
// need pos as an intermediate (see ml84_trfeed_pack_host below and
// gemm_ml84_prod.hip's device packer / expander).
__host__ __device__ inline size_t ml84_trfeed_nk_to_pos(int n, int k, int N, int /*K*/) {
    const int NT = N / 16;
    const int kt = k / 16, nt = n / 16, kl = k % 16, nl = n % 16;
    int L, s;
    ml84_lane_slot(kl, nl, &L, &s);
    const size_t byte_off = ml84_tile_nib_base(kt, nt, NT) + (size_t) L * 4 + (size_t) (s & 3);
    const int hi = (s >> 2) & 1;
    return 2 * byte_off + (size_t) hi;
}

// Inverse of the above: pos (0..N*K-1) -> (n,k). Written independently of
// ml84_trfeed_nk_to_pos (not by literally undoing its arithmetic line by
// line) so the bench's round-trip check (loop every pos of a small shape,
// assert ml84_trfeed_nk_to_pos(pos_to_nk(pos)) == pos) is a real bijection
// check, not a tautology.
__host__ __device__ inline void ml84_trfeed_pos_to_nk(size_t pos, int N, int /*K*/, int* n, int* k) {
    const int NT = N / 16;
    const int hi = (int) (pos & 1);
    const size_t byte_off = pos >> 1;
    const size_t tile_nib_bytes = 128;                 // 256 fp8 bytes / 2 per 16x16 tile
    const size_t tile_idx = byte_off / tile_nib_bytes;
    const size_t within = byte_off % tile_nib_bytes;
    const int L = (int) (within / 4);
    const int byte_in_lane = (int) (within % 4);
    const int s = hi ? (byte_in_lane + 4) : byte_in_lane;
    const int kt = (int) (tile_idx / (size_t) NT);
    const int nt = (int) (tile_idx % (size_t) NT);
    const int kl = ((L >> 4) & 1) * 8 + s;
    const int nl = L & 15;
    *k = kt * 16 + kl;
    *n = nt * 16 + nl;
}

// ─────────────────────────────────────────────────────────────────────────
// e4m3 <-> fp32, host+device. Same convention (torch float8_e4m3fn / OCP,
// via hip_fp8.h's __hip_fp8_e4m3) the rest of this directory's benches use
// (see bench/gemm_trfeed_prod_bench.hip's enc()/dec()).
// ─────────────────────────────────────────────────────────────────────────
__host__ __device__ inline uint8_t ml84_encode_e4m3(float x) {
    __hip_fp8_e4m3 v(x);
    return *reinterpret_cast<uint8_t*>(&v);
}
__host__ __device__ inline float ml84_decode_e4m3(uint8_t b) {
    __hip_fp8_e4m3 v;
    *reinterpret_cast<uint8_t*>(&v) = b;
    return (float) v;
}

// Set/get a 4-bit nibble in a packed byte array by nibble-index `pos`
// (2*byte_off + hi), lo-nibble-first within a byte (matching block_ml8_4's
// on-disk convention, ggml-common.h).
__host__ __device__ inline void ml84_set_nibble(uint8_t* buf, size_t pos, uint8_t idx4) {
    const size_t byte_off = pos >> 1;
    if (pos & 1) buf[byte_off] = (uint8_t) ((buf[byte_off] & 0x0Fu) | ((idx4 & 0x0Fu) << 4));
    else         buf[byte_off] = (uint8_t) ((buf[byte_off] & 0xF0u) | (idx4 & 0x0Fu));
}
__host__ __device__ inline uint8_t ml84_get_nibble(const uint8_t* buf, size_t pos) {
    const uint8_t b = buf[pos >> 1];
    return (pos & 1) ? (uint8_t) ((b >> 4) & 0x0Fu) : (uint8_t) (b & 0x0Fu);
}

// ─────────────────────────────────────────────────────────────────────────
// HOST reference packer: on-disk block_ml8_4 array, layout `w[N][n_groups_k]`
// (row-major per output column n, n_groups_k = K/QK_ML8 groups per column —
// the SAME `src_blocks` convention `ggml_cuda_ml8_repack_blocks`/
// `ml8_repack_kernel` consume in ml8.cu) -> B_nib (N*K/2 bytes,
// ML84_TRFEED nibble layout) + b_scale_g (fp32 [K/64][N] row-major,
// identical to ml8_repack_kernel's `dst_b_scale` output).
//
// Used by the bench (gemm_ml84_bench.hip) to build ground-truth packed
// weights without needing the device packer below. Not on any hot path.
// ─────────────────────────────────────────────────────────────────────────
inline void ml84_trfeed_pack_host(const block_ml8_4* w, int N, int K, uint8_t* B_nib, float* b_scale_g) {
    const int n_groups_k = K / QK_ML8;
    std::memset(B_nib, 0, (size_t) N * (size_t) K / 2);
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < n_groups_k; ++g) {
            const block_ml8_4& blk = w[(size_t) n * (size_t) n_groups_k + (size_t) g];
            b_scale_g[(size_t) g * (size_t) N + (size_t) n] = blk.scale;
            for (int i = 0; i < QK_ML8 / 2; ++i) {
                const uint8_t packed = blk.qs[i];
                const uint8_t lo_idx = packed & 0x0Fu;
                const uint8_t hi_idx = (packed >> 4) & 0x0Fu;
                const int k_lo = g * QK_ML8 + 2 * i;
                const int k_hi = k_lo + 1;
                ml84_set_nibble(B_nib, ml84_trfeed_nk_to_pos(n, k_lo, N, K), lo_idx);
                ml84_set_nibble(B_nib, ml84_trfeed_nk_to_pos(n, k_hi, N, K), hi_idx);
            }
        }
    }
}
