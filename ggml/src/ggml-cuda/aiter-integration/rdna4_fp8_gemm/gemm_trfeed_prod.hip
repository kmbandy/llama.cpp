// gemm_trfeed_prod.hip — MAD-305 Phase 5 (round 3) production entry point.
//
// Implements rdna4_gemm_fp8_trfeed (gemm_capi.h) by launching the FROZEN
// Phase-1 "trfeed" kernel verbatim from bench/trfeed_kernels.h -- the SAME
// header bench/gemm_trfeed_bench.hip includes, so THIS translation unit and
// the bench compile byte-identical device code for gemm_fp8_trfeed<TBM,TWAVES_M>
// (acc[4][4] TBM=128 for prefill, acc[2][4] TBM=32 for decode/verify -- see
// TBM_DECODE below; BN=128/BK=32, both instantiations of the SAME frozen
// template). DO NOT add scale folding, double buffering, or any other
// instruction to that kernel here or in the header: every such attempt on
// this kernel family regressed 4-20x (RESULT.md Phase 2; gemm_blockscale.hip's
// two abandoned in-loop-fold kernels measured 7-35 TF vs this kernel's 131 TF
// at production shapes).
//
// gfx1201-only, matching gemm_blockscale.hip's convention: the kernel body
// uses `global_load_tr_b64` + the gfx12 fp8 WMMA builtin, both gfx12
// matrix-core-only intrinsics that don't exist for other offload-arch passes
// potentially built into the same fat binary (e.g. gfx1030). The launcher
// below is guarded by `#if defined(__gfx1201__)` -- defined only during the
// gfx1201 device-code compilation pass -- so every other pass (and the
// host-only pass) compiles the `#else` build-safety stub instead of failing
// to lower an unknown intrinsic. `supports_op` (ggml-cuda.cu) gates
// GGML_OP_FP8_MUL_MAT's rdna4-layout path to RDNA4 devices, so the stub is
// never taken at runtime.
#include <hip/hip_runtime.h>
#include "bench/trfeed_kernels.h"
#include "bench/trfeed_splitk_kernels.h"
#include "gemm_capi.h"
#include <algorithm>
#include <cstdint>

// Decode/verify M tile: TBM=32, TWAVES_M=1 -> TWAVES=TWAVES_M*WAVES_N=2,
// TBLOCK=TWAVES*WAVE_SIZE=64, TFRAGS_M=(TBM/TWAVES_M)/16=(32/1)/16=2,
// TFRAGS_N=(BN/WAVES_N)/16=(128/2)/16=4 (N-side identical to the prefill
// tile -- BN/WAVES_N are template-global constants, unaffected by
// TBM/TWAVES_M). Per-K-tile A fill: AVEC=TBM*BK/16=32*32/16=64, BKv=BK/16=2,
// so the `for (e=tid; e<AVEC; e+=TBLOCK)` loop is APT=AVEC/TBLOCK=64/64=1 --
// every derived constant divides cleanly (no remainder anywhere in the
// template body for this instantiation); this is a valid, distinct
// instantiation of the SAME frozen template, not a new kernel body.
constexpr int TBM_DECODE = 32, TWAVES_M_DECODE = 1;

// Host launcher: NOT arch-guarded (the __gfx1201__ macro only exists in the device
// pass); the kernel bodies in bench/trfeed_kernels.h carry the per-arch guard.
hipError_t rdna4_gemm_fp8_trfeed(const void* A, const void* B_shuf, void* C_bf16,
                                 const float* a_scale, const float* b_scale,
                                 int M_pad, int N, int K, hipStream_t stream) {
    if (M_pad <= 0 || N <= 0 || K <= 0 || (N % BN) != 0 || (K % BK) != 0) {
        return hipErrorInvalidValue;
    }
    // M_pad == TBM_DECODE (32) selects the decode/verify tile (the caller
    // pads M<=32 up to exactly 32, see gemm_capi.h); every other M_pad must
    // be an exact multiple of the prefill tile's BM=128.
    if (M_pad == TBM_DECODE) {
        constexpr int TBLOCK = TWAVES_M_DECODE * WAVES_N * WAVE_SIZE;
        const dim3 grid(N / BN, M_pad / TBM_DECODE);
        gemm_fp8_trfeed<TBM_DECODE, TWAVES_M_DECODE><<<grid, TBLOCK, 0, stream>>>(
            (const float8_t*) A, (const uint8_t*) B_shuf, (__hip_bfloat16*) C_bf16,
            a_scale, b_scale, M_pad, N, K);
        return hipGetLastError();
    }
    if ((M_pad % BM) != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int TBM = BM, TWAVES_M = WAVES_M;   // the frozen 128x128/4-wave prefill tile
    constexpr int TBLOCK = TWAVES_M * WAVES_N * WAVE_SIZE;
    const dim3 grid(N / BN, M_pad / TBM);
    gemm_fp8_trfeed<TBM, TWAVES_M><<<grid, TBLOCK, 0, stream>>>(
        (const float8_t*) A, (const uint8_t*) B_shuf, (__hip_bfloat16*) C_bf16,
        a_scale, b_scale, M_pad, N, K);
    return hipGetLastError();
}

// fp32-output launcher (MAD-305, freeze lifted 2026-09-18): SAME grid/tile
// rules as rdna4_gemm_fp8_trfeed above (M_pad==32 selects the decode/verify
// tile, else M_pad must be an exact multiple of BM=128 for the prefill
// tile) -- copied verbatim from the bf16 launcher, not re-derived -- just
// instantiating gemm_fp8_trfeed<TBM,TWAVES_M,float> instead of the default
// <TBM,TWAVES_M> (== <TBM,TWAVES_M,__hip_bfloat16>), and threading M_valid
// (the caller's true M) through so the epilogue's float branch never writes
// past the caller's un-padded fp32 destination. Removes the
// convert_unary<bf16,float> pass the bf16 path's caller otherwise needs.
hipError_t rdna4_gemm_fp8_trfeed_f32(const uint8_t* a_fp8, const uint8_t* b_shuf, float* c_f32,
                                     const float* a_scale, const float* b_scale,
                                     int M_pad, int M_valid, int N, int K, hipStream_t stream) {
    if (M_pad <= 0 || N <= 0 || K <= 0 || (N % BN) != 0 || (K % BK) != 0) {
        return hipErrorInvalidValue;
    }
    if (M_valid <= 0 || M_valid > M_pad) {
        return hipErrorInvalidValue;
    }
    if (M_pad == TBM_DECODE) {
        constexpr int TBLOCK = TWAVES_M_DECODE * WAVES_N * WAVE_SIZE;
        const dim3 grid(N / BN, M_pad / TBM_DECODE);
        gemm_fp8_trfeed<TBM_DECODE, TWAVES_M_DECODE, float><<<grid, TBLOCK, 0, stream>>>(
            (const float8_t*) a_fp8, b_shuf, c_f32, a_scale, b_scale, M_pad, N, K, M_valid);
        return hipGetLastError();
    }
    if ((M_pad % BM) != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int TBM = BM, TWAVES_M = WAVES_M;   // the frozen 128x128/4-wave prefill tile
    constexpr int TBLOCK = TWAVES_M * WAVES_N * WAVE_SIZE;
    const dim3 grid(N / BN, M_pad / TBM);
    gemm_fp8_trfeed<TBM, TWAVES_M, float><<<grid, TBLOCK, 0, stream>>>(
        (const float8_t*) a_fp8, b_shuf, c_f32, a_scale, b_scale, M_pad, N, K, M_valid);
    return hipGetLastError();
}

// SwiGLU-epilogue launcher (MAD-305 fused-FFN task): SAME grid/tile rules as
// rdna4_gemm_fp8_trfeed_f32 above, but N_fused is the interleaved gate/up
// column count (see trfeed_kernels.h's SwiGLU epilogue comment and
// rdna4_expand_ml84_pair_to_trfeed in gemm_ml84_prod.hip for how B_shuf_fused
// / b_scale_fused are built) and c_f32 is [M_valid, N_fused/2] -- HALF the
// fused width, one value per (gate,up) pair. Only the prefill tile is wired
// here (the decode M_pad==32 tile is never fused -- see the caller,
// ml8.cu's fusion match, which only takes this path for M_pad>32); the
// TBM_DECODE branch below still exists so a future decode-side fusion has
// somewhere to plug in, but is untested and not called by anything yet.
hipError_t rdna4_gemm_fp8_trfeed_swiglu_f32(const uint8_t* a_fp8, const uint8_t* b_shuf_fused,
                                            float* c_f32, const float* a_scale, const float* b_scale_fused,
                                            int M_pad, int M_valid, int N_fused, int K, hipStream_t stream) {
    if (M_pad <= 0 || N_fused <= 0 || K <= 0 || (N_fused % BN) != 0 || (K % BK) != 0) {
        return hipErrorInvalidValue;
    }
    if (M_valid <= 0 || M_valid > M_pad) {
        return hipErrorInvalidValue;
    }
    if (M_pad == TBM_DECODE) {
        constexpr int TBLOCK = TWAVES_M_DECODE * WAVES_N * WAVE_SIZE;
        const dim3 grid(N_fused / BN, M_pad / TBM_DECODE);
        gemm_fp8_trfeed<TBM_DECODE, TWAVES_M_DECODE, float, /*SwiGLU=*/true><<<grid, TBLOCK, 0, stream>>>(
            (const float8_t*) a_fp8, b_shuf_fused, c_f32, a_scale, b_scale_fused, M_pad, N_fused, K, M_valid);
        return hipGetLastError();
    }
    if ((M_pad % BM) != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int TBM = BM, TWAVES_M = WAVES_M;   // the frozen 128x128/4-wave prefill tile
    constexpr int TBLOCK = TWAVES_M * WAVES_N * WAVE_SIZE;
    const dim3 grid(N_fused / BN, M_pad / TBM);
    gemm_fp8_trfeed<TBM, TWAVES_M, float, /*SwiGLU=*/true><<<grid, TBLOCK, 0, stream>>>(
        (const float8_t*) a_fp8, b_shuf_fused, c_f32, a_scale, b_scale_fused, M_pad, N_fused, K, M_valid);
    return hipGetLastError();
}

// ─────────────────────────────────────────────────────────────────────────
// Decode split-K variant (gemm_capi.h): only the M_pad==32 decode/verify
// tile has few enough workgroups (N/128 = 40..136 at production N) to leave
// the GPU parallelism-starved streaming the whole K serially (measured
// ~160 GB/s effective weight-bandwidth at M=2, N=17408, K=5120 on a
// 640 GB/s card). Splitting K across blockIdx.z multiplies the workgroup
// count by n_splits at the cost of atomic-add reduction into an fp32
// output; see bench/trfeed_splitk_kernels.h for the kernel itself (a
// separate copy of the frozen gemm_fp8_trfeed<32,1> body, not an edit to
// it). Not used for the prefill tile (M_pad%128==0): that shape already
// dispatches N/128 * M_pad/128 workgroups, plenty to saturate the GPU.
//
// Heuristic default n_splits: aim for >= 256 total workgroups (N/128 * n_splits)
// while keeping each split at least 4 K-tiles wide (128 K-elements) so a
// split's own A-fill/WMMA loop still amortizes its LDS fill overhead.
int rdna4_trfeed_splitk_default_splits(int N, int K) {
    if (N <= 0 || K <= 0) {
        return 1;
    }
    const int n_wg_base = N / BN;               // workgroups the M<=32 tile launches with n_splits=1
    const int k_tiles   = K / BK;
    if (n_wg_base <= 0 || k_tiles <= 0) {
        return 1;
    }
    // Measured on the 9070 XT (bench/gemm_trfeed_prod_bench, M<=32 tile):
    //   N=17408 K=5120  (136 WGs): frozen 0.176 ms; splits 1/2/4/8/16 =
    //     0.227/0.273/0.397/0.665/1.147 ms -- never split.
    //   N=5120  K=17408 (40 WGs):  frozen 0.478 ms; splits 1/2/4/8/16 =
    //     0.484/0.281/0.224/0.292/0.433 ms -- 4 wins (2.1x).
    // So the win exists only while the base launch is workgroup-starved
    // (< ~128 WGs on 32 CUs) and the atomic epilogue's cost grows with
    // splits x N; aim for ~160 WGs total, rounded to nearest, never more
    // than ceil(160/base). The caller treats a return of 1 as "use the
    // non-split bf16 frozen path".
    if (n_wg_base >= 128) {
        return 1;
    }
    const int target_total_wg = 160;
    int want = (target_total_wg + n_wg_base / 2) / n_wg_base;   // round(160 / n_wg_base)
    const int max_splits = std::max(1, k_tiles / 4);            // >= 4 K-tiles/split
    return std::min(std::max(want, 1), max_splits);
}

hipError_t rdna4_gemm_fp8_trfeed_splitk(const void* A, const void* B_shuf, float* C_f32,
                                        const float* a_scale, const float* b_scale,
                                        int M_pad, int N, int K, int n_splits, hipStream_t stream) {
    if (M_pad != TBM_DECODE || N <= 0 || (N % BN) != 0 || K <= 0 || (K % BK) != 0) {
        return hipErrorInvalidValue;
    }
    const int k_tiles = K / BK;
    if (n_splits < 1) {
        n_splits = 1;
    }
    if (n_splits > k_tiles) {
        n_splits = k_tiles;   // never split finer than one K-tile per workgroup
    }
    const int k_tiles_per_split = (k_tiles + n_splits - 1) / n_splits;

    // Atomics accumulate onto zero -- zero the WHOLE M_pad*N output (not just
    // the caller's true-M rows) so the padded decode rows the frozen
    // A-tile fill still reads/writes never contain garbage; the epilogue's
    // row mask against the true M then decides which of those zeroed/
    // accumulated rows the caller actually reads back.
    const hipError_t zc = hipMemsetAsync(C_f32, 0, (size_t) M_pad * (size_t) N * sizeof(float), stream);
    if (zc != hipSuccess) {
        return zc;
    }

    constexpr int TBLOCK = TWAVES_M_DECODE * WAVES_N * WAVE_SIZE;
    const dim3 grid(N / BN, M_pad / TBM_DECODE, n_splits);
    gemm_fp8_trfeed_splitk<TBM_DECODE, TWAVES_M_DECODE><<<grid, TBLOCK, 0, stream>>>(
        (const float8_t*) A, (const uint8_t*) B_shuf, C_f32, a_scale, b_scale,
        M_pad, N, K, k_tiles_per_split);
    return hipGetLastError();
}

