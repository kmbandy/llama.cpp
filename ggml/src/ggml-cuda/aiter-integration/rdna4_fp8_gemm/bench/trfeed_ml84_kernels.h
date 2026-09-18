// bench/trfeed_ml84_kernels.h — MAD-305 ML8_4 (4.5 bpw) decode/prefill kernel.
//
// This is a SEPARATE COPY of the frozen `gemm_fp8_trfeed<TBM,TWAVES_M>` body
// (trfeed_kernels.h) — NOT an edit to that file, following the same
// copying discipline as bench/trfeed_splitk_kernels.h. The A-tile LDS fill,
// the WMMA sequence's shape, and the epilogue's per-lane (row,col) mapping
// are all byte-for-byte the frozen body's; the differences are confined to:
//   (i)   the B feed: a plain 4-byte load from the ML84_TRFEED `B_nib`
//         buffer (ml84_trfeed_layout.h) + a 4-register LUT expand, instead
//         of `tr_load8` from a pre-shuffled fp8 `B_shuf`;
//   (ii)  a second accumulator set `acc_g`, zeroed every 64-K group and
//         folded into `acc` (scaled by b_scale_g) at each group's end —
//         BK=32 divides the QK_ML8=64 group width, so a group is exactly
//         two K-tile iterations of the frozen loop;
//   (iii) the epilogue stores fp32 directly (`a_scale[m]*acc[..]` — the
//         b_scale is already folded in step (ii)), not bf16.
// Everything else (LDS A-fill indexing, `row_a`/`colhi` lane maps, the WMMA
// builtin call itself, the `__syncthreads()` placement, the epilogue's
// `e_col`/`e_rowbase` scratch layout and row mask against M) is copied
// verbatim from trfeed_kernels.h's `gemm_fp8_trfeed<TBM,TWAVES_M>`.
//
// HISTORY (round 2->3, coordinator-measured on a 9070 XT): an earlier
// version of this file additionally staged b_scale_g/lut in LDS per
// ML84_CHUNK_GROUPS-group chunk inside a separate `gemm_ml84_trfeed` kernel,
// on the theory that exposed per-group global loads explained a measured
// 0.31ms-vs-0.176ms gap against the frozen fp8 kernel. Measured result: NO
// improvement (0.300-0.305 ms at N=17408, same as unstaged) -- the latency
// theory was wrong, so that kernel and its launcher are REMOVED (round 3).
// What actually wins is split-K (below): unstaged, N=17408 n_splits=1
// measured 0.194 ms (258 GB/s) -- already within 10% of the frozen fp8
// kernel's 0.176 ms -- and N=5120/K=17408 n_splits=4 measured 0.177 ms
// (283 GB/s), beating the fp8 path's own split-K at the same shape (0.224
// ms). `gemm_ml84_trfeed_splitk` below is now the ONLY decode kernel body;
// `rdna4_gemm_ml84_trfeed_decode` (gemm_ml84_prod.hip) just calls it with
// n_splits=1 via the ATOMIC=false template path (plain store, no memset,
// grid.z=1 -- functionally a non-split launch, sharing one body with true
// split-K instead of keeping two copies).
//
// Also instantiated at TBM=128,TWAVES_M=2 (the frozen prefill tile
// geometry) by rdna4_gemm_ml84_trfeed_prefill for the "in-kernel LUT expand
// instead of expander+frozen-fp8" prefill experiment (gemm_ml84_prod.hip);
// the template parameters were always kept generic for exactly this reason.
#pragma once

#include "trfeed_kernels.h"        // BM,BN,BK,WAVES_M,WAVES_N,WAVE_SIZE,KSTEPS,float8_t,v2i32,v8f32,b_tile_offset
#include "ml84_trfeed_layout.h"    // ml84_tile_nib_base; QK_ML8 (via ggml-common.h)

#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)

// ─────────────────────────────────────────────────────────────────────────
// LUT expand: rebuild the 8-fp8-byte WMMA B fragment (the same v2i32{x,y}
// the frozen kernel's tr_load8 returns: x = elements 0-3, y = elements 4-7)
// from a 4-byte nibble dword `w` (2 packed indices/byte, low-nibble-first
// per element pair — see ml84_trfeed_layout.h's B_nib contract) and the
// group's 16-entry e4m3 LUT, held uniformly across all lanes in 4 uint32
// registers L0..L3 (L0 = lut bytes 0-3 = indices 0-3, ..., L3 = indices
// 12-15).
//
// PERM-SELECTOR CONVENTION ASSUMED (uncertain — could not verify against
// hardware/ISA docs or a compile in this sandbox; flagged in the task
// report): `__builtin_amdgcn_perm(S0, S1, sel)` lowers to `v_perm_b32`
// treating the two 32-bit operands as the 64-bit concatenation {S0:S1} with
// S1 occupying the LOW 4 bytes (selector value 0-3) and S0 the HIGH 4 bytes
// (selector value 4-7) — i.e. argument order is (high-half, low-half), not
// (low-half, high-half). Under that convention,
// `__builtin_amdgcn_perm(L1, L0, sel)` selects: sel 0-3 -> L0 (indices 0-3),
// sel 4-7 -> L1 (indices 4-7) — which is what we want for a 0..7 gather.
// A second call over L2/L3 with the selector's low 3 bits (`sel & 0x07`)
// covers indices 8-15, and the two results are blended on selector bit 3
// (`(sel>>3)&1`, replicated to a full-byte mask) to get the full 0..15
// gather. If this convention is backwards on the actual toolchain, the fix
// is a one-line argument swap in both `__builtin_amdgcn_perm` calls below —
// isolated here, not spread through the kernel.
__device__ inline uint32_t ml84_lut_gather4(uint32_t sel, uint32_t L0, uint32_t L1, uint32_t L2, uint32_t L3) {
    const uint32_t lo = __builtin_amdgcn_perm(L1, L0, sel);                 // sel in 0..7 -> L0/L1
    const uint32_t hi = __builtin_amdgcn_perm(L3, L2, sel & 0x07070707u);   // sel-8 in 0..7 -> L2/L3
    const uint32_t mask = ((sel >> 3) & 0x01010101u) * 0xFFu;              // 0xFF per byte where sel>=8
    return (hi & mask) | (lo & ~mask);
}

// Expand one lane's 4-byte B_nib dword `w` into the frozen kernel's v2i32 B
// fragment: x = elements s=0..3 (low nibbles of w's 4 bytes as selectors),
// y = elements s=4..7 (high nibbles).
__device__ inline v2i32 ml84_expand_frag(uint32_t w, uint32_t L0, uint32_t L1, uint32_t L2, uint32_t L3) {
    const uint32_t sel_lo = w & 0x0F0F0F0Fu;         // elements 0-3
    const uint32_t sel_hi = (w >> 4) & 0x0F0F0F0Fu;  // elements 4-7
    const uint32_t x = ml84_lut_gather4(sel_lo, L0, L1, L2, L3);
    const uint32_t y = ml84_lut_gather4(sel_hi, L0, L1, L2, L3);
    return v2i32{(int) x, (int) y};
}

// One K-group's worth of WMMA work (2 BK=32 K-tiles), reading the group's
// LUT out of 4 uniform uint32 registers L0..L3 (loaded from global by the
// caller -- see gemm_ml84_trfeed_splitk below). `As`/`Asv` (the A tile) and
// the WMMA loop itself are otherwise identical to the frozen fp8 trfeed
// body. Factored out so gemm_ml84_trfeed_splitk's ATOMIC=false (decode,
// n_splits=1) and ATOMIC=true (true split-K) instantiations, and the
// TBM=128,TWAVES_M=2 prefill instantiation, all share this exact body.
template <int TBM, int TWAVES_M>
__device__ inline void ml84_trfeed_group_body(
    const float8_t* __restrict__ A, const uint8_t* __restrict__ B_nib,
    float8_t* __restrict__ As, const uint32_t L0, const uint32_t L1,
    const uint32_t L2, const uint32_t L3,
    int tm, int tn, int wave_m, int wave_n, int lane, int NT, int K, int g,
    v8f32 (&acc_g)[(TBM / TWAVES_M) / 16][(BN / WAVES_N) / 16]) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    const int tid = threadIdx.x;
    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);

    for (int mi = 0; mi < TFRAGS_M; ++mi)
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            acc_g[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    #pragma unroll
    for (int half = 0; half < 2; ++half) {         // two BK=32 K-tiles per 64-wide group
        const int k0 = g * QK_ML8 + half * BK;
        // ---- A fill: BYTE-FOR-BYTE the frozen body ----
        constexpr int AVEC = TBM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv;
            int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        __syncthreads();

        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[TFRAGS_M], fb[TFRAGS_N];
            for (int mi = 0; mi < TFRAGS_M; ++mi) {
                int lds_row = (wave_m * TFRAGS_M + mi) * 16 + row_a;
                fa[mi] = *reinterpret_cast<const v2i32*>(As + lds_row * BK + kbase + colhi * 8);
            }
            for (int ni = 0; ni < TFRAGS_N; ++ni) {
                int nt = (tn + (wave_n * TFRAGS_N + ni) * 16) / 16;
                // ---- (i) B feed: plain load + LUT expand, replacing tr_load8 ----
                const size_t nib_base = ml84_tile_nib_base(kt, nt, NT);
                const uint32_t w = *reinterpret_cast<const uint32_t*>(B_nib + nib_base + (size_t) lane * 4);
                fb[ni] = ml84_expand_frag(w, L0, L1, L2, L3);
            }
            for (int mi = 0; mi < TFRAGS_M; ++mi)
                for (int ni = 0; ni < TFRAGS_N; ++ni)
                    acc_g[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc_g[mi][ni]);
        }
        __syncthreads();                              // A LDS reuse barrier -- frozen, unchanged
    }
}

// Epilogue for gemm_ml84_trfeed_splitk's two ATOMIC instantiations.
// ATOMIC=false: plain store (grid.z==1, i.e. the "decode" non-split launch).
// ATOMIC=true: atomicAdd into a caller-zeroed C_f32 (true multi-way
// split-K), matching how trfeed_splitk_kernels.h's epilogue differs from
// trfeed_kernels.h's only in that one respect. A bool template param rather
// than a device lambda/functor -- matches this codebase's existing style
// (no device lambdas elsewhere in these kernel headers) and avoids relying
// on HIP's extended-lambda device-lambda support.
template <int TBM, int TWAVES_M, bool ATOMIC>
__device__ inline void ml84_trfeed_epilogue(
    v8f32 (&acc)[(TBM / TWAVES_M) / 16][(BN / WAVES_N) / 16],
    int tm, int tn, int wave_m, int wave_n, int lane, int wid, int M, int N,
    const float* __restrict__ a_scale, float* __restrict__ C_f32) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    __shared__ float scratch[TWAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < TFRAGS_M; ++mi) {
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
            int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * TFRAGS_N + ni) * 16;
            for (int t = lane; t < 256; t += WAVE_SIZE) {
                int gr = row0 + t / 16, gc = col0 + t % 16;
                if (gr < M && gc < N) {
                    const float v = a_scale[gr] * ws[t];
                    if (ATOMIC) atomicAdd(&C_f32[gr * N + gc], v);
                    else        C_f32[gr * N + gc] = v;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// The ONLY decode/prefill kernel body (round 3: replaces the removed
// LDS-staged gemm_ml84_trfeed -- see the file header HISTORY note). Splits
// are on 64-K GROUP boundaries (not raw K-tiles), since the per-group scale
// fold must see a group's BOTH BK=32 halves before folding -- a split
// boundary mid-group would either double-fold or drop half a group's
// contribution. blockIdx.z selects this WG's group range
// [z*groups_per_split, min((z+1)*groups_per_split, n_groups_k)).
//
// ATOMIC=false, groups_per_split=n_groups_k, grid.z=1: the "decode, no
// split" case (rdna4_gemm_ml84_trfeed_decode) -- functionally a plain
// non-split launch (one workgroup walks the WHOLE K range and stores once),
// just sharing this one body instead of keeping a separate near-duplicate
// kernel around. ATOMIC=true: true multi-way split-K
// (rdna4_gemm_ml84_trfeed_decode_splitk with n_splits>1), atomicAdd-ing into
// a caller-zeroed C_f32, mirroring trfeed_splitk_kernels.h's own epilogue
// diff from its non-split counterpart. No LDS scale/LUT staging (measured
// to give zero win at the plain decode's full-K-range group count, and each
// split's own group range is only ever a few groups wide at n_splits>1
// anyway -- see rdna4_ml84_trfeed_splitk_default_splits's >=2-groups/split
// floor) -- simplicity wins, exactly the same trade the fp8
// trfeed_splitk_kernels.h made.
//
// Also instantiated at TBM=128,TWAVES_M=2 (ATOMIC=false, groups_per_split=
// n_groups_k, grid.z=1) by rdna4_gemm_ml84_trfeed_prefill for the in-kernel
// LUT-expand prefill experiment -- the frozen prefill tile geometry, same
// body, no new kernel needed.
// ─────────────────────────────────────────────────────────────────────────
template <int TBM, int TWAVES_M, bool ATOMIC>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_ml84_trfeed_splitk(const float8_t* __restrict__ A, const uint8_t* __restrict__ B_nib,
                        const uint8_t* __restrict__ lut, float* __restrict__ C_f32,
                        const float* __restrict__ a_scale, const float* __restrict__ b_scale_g,
                        int M, int N, int K, int groups_per_split) {
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * BK];
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;
    const int n_groups_k = K / QK_ML8;
    const int g_lo = blockIdx.z * groups_per_split;
    const int g_hi = min((blockIdx.z + 1) * groups_per_split, n_groups_k);

    v8f32 acc[TFRAGS_M][TFRAGS_N], acc_g[TFRAGS_M][TFRAGS_N];
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    for (int g = g_lo; g < g_hi; ++g) {
        const uint32_t L0 = *reinterpret_cast<const uint32_t*>(lut + (size_t) g * 16 + 0);
        const uint32_t L1 = *reinterpret_cast<const uint32_t*>(lut + (size_t) g * 16 + 4);
        const uint32_t L2 = *reinterpret_cast<const uint32_t*>(lut + (size_t) g * 16 + 8);
        const uint32_t L3 = *reinterpret_cast<const uint32_t*>(lut + (size_t) g * 16 + 12);

        ml84_trfeed_group_body<TBM, TWAVES_M>(A, B_nib, As, L0, L1, L2, L3,
                                               tm, tn, wave_m, wave_n, lane, NT, K, g, acc_g);

        const int e_col = lane & 0xF;
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            const int col0 = tn + (wave_n * TFRAGS_N + ni) * 16;
            const int gc = col0 + e_col;
            const float scale = b_scale_g[(size_t) g * N + gc];
            for (int mi = 0; mi < TFRAGS_M; ++mi)
                acc[mi][ni] = acc[mi][ni] + scale * acc_g[mi][ni];
        }
    }

    // ---- epilogue: plain store (ATOMIC=false) or atomicAdd into a
    // caller-zeroed C_f32 (ATOMIC=true) ----
    ml84_trfeed_epilogue<TBM, TWAVES_M, ATOMIC>(
        acc, tm, tn, wave_m, wave_n, lane, wid, M, N, a_scale, C_f32);
}

#else
template <int TBM, int TWAVES_M, bool ATOMIC>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_ml84_trfeed_splitk(const float8_t* __restrict__ A, const uint8_t* __restrict__ B_nib,
                        const uint8_t* __restrict__ lut, float* __restrict__ C_f32,
                        const float* __restrict__ a_scale, const float* __restrict__ b_scale_g,
                        int M, int N, int K, int groups_per_split);
#endif
