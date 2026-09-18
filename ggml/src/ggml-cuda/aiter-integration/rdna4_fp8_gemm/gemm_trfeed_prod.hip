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
#include "gemm_capi.h"

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

