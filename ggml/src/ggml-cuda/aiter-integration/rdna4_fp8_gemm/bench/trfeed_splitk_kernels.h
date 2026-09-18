// bench/trfeed_splitk_kernels.h — MAD-305 split-K decode kernel.
//
// Problem: at M<=32 the frozen `gemm_fp8_trfeed<32,1>` launches grid
// (N/128, 1) -- only N/128 workgroups (40-136 for the production N range
// 5120..17408), each one streaming the WHOLE K dimension serially. Measured
// 0.186 ms/call at M=2, N=17408, K=5120 (~160 GB/s effective weight
// streaming on a 640 GB/s card): far too few workgroups to saturate the
// memory system at decode's tiny M.
//
// Fix: split K across blockIdx.z. Each workgroup streams only a K-slice
// [z*k_tiles_per_split, min((z+1)*k_tiles_per_split, K/BK)) and atomically
// accumulates its partial a_scale*b_scale-scaled product into an fp32
// output (C_f32 must be zeroed by the caller before launch -- see
// rdna4_gemm_fp8_trfeed_splitk in ../gemm_trfeed_prod.hip). fp32 output also
// drops the bf16 conversion kernel the non-split-K production path runs
// after the GEMM.
//
// This is a SEPARATE COPY of the frozen `gemm_fp8_trfeed<TBM,TWAVES_M>` body
// in trfeed_kernels.h -- NOT an edit to that file. The main K-tile loop (A
// LDS fill, `global_load_tr` B feed, WMMA sequence) is IDENTICAL to the
// frozen body; the only changes are (1) the loop bounds, restricted to this
// workgroup's K-tile split via blockIdx.z, and (2) the epilogue, which
// stores via `atomicAdd` into fp32 C instead of a plain bf16 store (still
// applying a_scale[m]*b_scale[n] exactly as the frozen epilogue does, and
// still masking rows against M). trfeed_kernels.h's helper constants/types
// (BM, BN, BK, WAVES_N, WAVE_SIZE, KSTEPS, float8_t, v2i32, v8f32, tr_load8,
// b_tile_offset) are reused via #include, not redefined here.
#pragma once

#include "trfeed_kernels.h"

// Kernel body compiled only in the host pass and the gfx1201 device pass,
// matching trfeed_kernels.h's guard convention exactly (see that header's
// comment on why: other offload-arch device-code passes sharing the fat
// binary, e.g. gfx1030, must never see the gfx12-only WMMA/global_load_tr
// intrinsics or the gfx12 atomic-fadd path below).
#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)

template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed_splitk(const float8_t* __restrict__ A, const uint8_t* __restrict__ B_shuf,
                       float* __restrict__ C_f32,
                       const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                       int M, int N, int K, int k_tiles_per_split) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * BK];                // A still staged in LDS (wide read)
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    // Split-K range: this workgroup only walks K-tiles
    // [blockIdx.z*k_tiles_per_split, min((blockIdx.z+1)*k_tiles_per_split, K/BK)).
    // Everything else in the main loop below is byte-for-byte the frozen
    // gemm_fp8_trfeed<TBM,TWAVES_M> body from trfeed_kernels.h.
    const int total_k_tiles = K / BK;
    const int kt_lo = blockIdx.z * k_tiles_per_split;
    const int kt_hi = min((blockIdx.z + 1) * k_tiles_per_split, total_k_tiles);

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int kt0 = kt_lo; kt0 < kt_hi; ++kt0) {
        const int k0 = kt0 * BK;
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
                fb[ni] = tr_load8(B_shuf + b_tile_offset(kt, nt, NT) + lane * 8);   // one global_load_tr
            }
            for (int mi = 0; mi < TFRAGS_M; ++mi)
                for (int ni = 0; ni < TFRAGS_N; ++ni)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc[mi][ni]);
        }
        __syncthreads();                              // A LDS reuse barrier
    }

    // ---- epilogue: apply a_scale[m]*b_scale[n] exactly as the frozen
    // epilogue does, but atomically ADD the partial-K product into fp32 C
    // (zeroed by the caller before launch) instead of overwriting a bf16
    // C -- each of the k_tiles_per_split-wide K-slices computed by the
    // different blockIdx.z values for this (tm,tn) tile must sum, not
    // clobber each other. Row mask against M kept, identical to the frozen
    // epilogue's `if (gr < M && gc < N)`.
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
                    float v = ws[t] * a_scale[gr] * b_scale[gc];
                    atomicAdd(&C_f32[gr * N + gc], v);
                }
            }
        }
    }
}

#else
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed_splitk(const float8_t* __restrict__ A, const uint8_t* __restrict__ B_shuf,
                       float* __restrict__ C_f32,
                       const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                       int M, int N, int K, int k_tiles_per_split);
#endif
