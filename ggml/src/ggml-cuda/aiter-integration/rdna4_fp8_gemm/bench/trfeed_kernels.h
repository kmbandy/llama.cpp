// bench/trfeed_kernels.h — MAD-305 Phase 1/5 FROZEN kernel bodies, extracted
// VERBATIM (byte-for-byte, no instruction changed) from gemm_trfeed_bench.hip
// so that bench and production translation units compile the IDENTICAL kernel
// object code from one source of truth. DO NOT EDIT the K-loop or epilogue math
// of any kernel below without re-running the bench oracle + perf gates.
//
// Production freezes on gemm_fp8_trfeed<128,2> (acc[4][4], TBM=128, BN=128,
// BK=32, 4 waves): the Phase-1 "winner" (140.3 TF @4096^3, bit-exact vs the
// byte-gather baseline, RESULT.md Phase 1). gemm_fp8_trfeed_rb is kept alongside
// (also oracle-PASS, same epilogue) since the register-blocked variant was the
// other named candidate; swapping the production entry point to it is a
// one-line change in ../gemm_trfeed_prod.hip because both bodies live here,
// unmodified, byte-identical to what the bench measures.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include "trfeed_common.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
// rocwmma's config static-asserts on non-WMMA offload archs (gfx1030 device
// pass in the multi-arch libggml-hip build); it is only needed where the kernel
// bodies are compiled (host pass + gfx1201 device pass).
#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
#else
typedef __hip_fp8_e4m3 float8_t;   // declaration-only pass: just needs the type name
#endif

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

// Same tile geometry as the verified kernel (gemm_wmma.hip): 128x128 tile, BK=32, 4 waves.
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WAVES_M = 2, WAVES_N = 2, WAVES = WAVES_M * WAVES_N;
constexpr int WAVE_SIZE = 32, BLOCK_THREADS = WAVES * WAVE_SIZE;
constexpr int FRAGS_M = (BM / WAVES_M) / 16;   // 2
constexpr int FRAGS_N = (BN / WAVES_N) / 16;   // 2
constexpr int KSTEPS  = BK / 16;               // 2

// Kernel BODIES are compiled only in the host pass (launch stubs need the
// definitions) and in the gfx1201 device pass; other offload archs in the same
// fat binary (gfx1030) get declarations only, so the gfx12-only intrinsics
// below are never lowered for them. Nothing inside this block is modified.
#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)
// gfx12 global transpose-load, 8-bit element / b64 form (one wave32 lane gets 8 bytes).
__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}

// ---------------- baseline: byte-gather B feed (B staged in LDS) ----------------
static __global__ void __launch_bounds__(BLOCK_THREADS)   // static: header is now included by two prod TUs
gemm_fp8_baseline(const float8_t* __restrict__ A, const float8_t* __restrict__ B,
                  __hip_bfloat16* __restrict__ C,
                  const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                  int M, int N, int K) {
    const int tm = blockIdx.y * BM;
    const int tn = blockIdx.x * BN;

    __shared__ float8_t As[BM * BK];   // [BM][BK], ld=BK (K-inner: A reads wide)
    __shared__ float8_t Bs[BK * BN];   // [BK][BN], ld=BN (N-inner: fragment read must gather)

    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;

    v8f32 acc[FRAGS_M][FRAGS_N];
    for (int mi = 0; mi < FRAGS_M; ++mi)
        for (int ni = 0; ni < FRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    const int4* Bv = reinterpret_cast<const int4*>(B);
    int4*  Asv = reinterpret_cast<int4*>(As);
    int4*  Bsv = reinterpret_cast<int4*>(Bs);
    uint8_t* Bsb = reinterpret_cast<uint8_t*>(Bs);

    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int AVEC = BM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += BLOCK_THREADS) {
            int r = e / BKv, c = e % BKv;
            int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        constexpr int BVEC = BK * BN / 16, BNv = BN / 16;
        for (int e = tid; e < BVEC; e += BLOCK_THREADS) {
            int r = e / BNv, c = e % BNv;
            int gk = k0 + r, gc = tn + c * 16;
            Bsv[e] = Bv[(gk * N + gc) / 16];
        }
        __syncthreads();

        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;   // A-map (§7.12)
        const int col_b = lane & 0xF, rowhi = (lane >> 4) & 1;   // B-map (§7.12)
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16;
            v2i32 fa[FRAGS_M], fb[FRAGS_N];
            for (int mi = 0; mi < FRAGS_M; ++mi) {
                int lds_row = (wave_m * FRAGS_M + mi) * 16 + row_a;
                fa[mi] = *reinterpret_cast<const v2i32*>(As + lds_row * BK + kbase + colhi * 8);
            }
            for (int ni = 0; ni < FRAGS_N; ++ni) {
                int lds_col = (wave_n * FRAGS_N + ni) * 16 + col_b;
                const uint8_t* bp = Bsb + (kbase + rowhi * 8) * BN + lds_col;
                uint32_t lo = 0, hi = 0;
                #pragma unroll
                for (int p = 0; p < 4; ++p) {
                    lo |= (uint32_t)bp[p * BN]       << (p * 8);
                    hi |= (uint32_t)bp[(p + 4) * BN] << (p * 8);
                }
                fb[ni] = v2i32{(int)lo, (int)hi};
            }
            for (int mi = 0; mi < FRAGS_M; ++mi)
                for (int ni = 0; ni < FRAGS_N; ++ni)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    // ---- epilogue (manual scratch, §7.12 C/D map) ----
    __shared__ float scratch[WAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < FRAGS_M; ++mi) {
        for (int ni = 0; ni < FRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
            int row0 = tm + (wave_m * FRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * FRAGS_N + ni) * 16;
            for (int t = lane; t < 256; t += WAVE_SIZE) {
                int gr = row0 + t / 16, gc = col0 + t % 16;
                if (gr < M && gc < N) {
                    float v = ws[t] * a_scale[gr] * b_scale[gc];
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed: B fed via global_load_tr_b64 from pre-shuffled global ------------
// Templated on the M-tiling so bm128 (TBM=128,TWAVES_M=2) and the larger-M amortization
// variant bm256 (TBM=256,TWAVES_M=4) share one body. N-tiling fixed: BN=128, WAVES_N=2.
// Larger TBM serves each B column tile across more M-rows -> fewer distinct B-tile DRAM
// fetches per output tile (amortizes B re-fetch).
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                __hip_bfloat16* __restrict__ C,
                const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                int M, int N, int K) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * BK];                // A still staged in LDS (wide read)
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += BK) {
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
                fb[ni] = tr_load8(Bshuf + b_tile_offset(kt, nt, NT) + lane * 8);   // one global_load_tr
            }
            for (int mi = 0; mi < TFRAGS_M; ++mi)
                for (int ni = 0; ni < TFRAGS_N; ++ni)
                    acc[mi][ni] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        fa[mi], fb[ni], acc[mi][ni]);
        }
        __syncthreads();                              // A LDS reuse barrier
    }

    // ---- epilogue (identical to baseline) ----
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
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed_db: software-pipelined double-buffer (the gfx1201 async lever) --------
// gfx1201 lacks global_load_lds (vmem->LDS direct) AND the gfx1250 async loads, so we get the
// CDNA double-buffer OVERLAP via software pipelining: issue the NEXT K-tile's A global loads into
// registers at the top of the loop (they go in-flight, tracked by vmcnt), run the current tile's
// WMMAs while memory is outstanding, then store the prefetched A into the OTHER LDS buffer (the
// s_waitcnt vmcnt lands here). Ping-pong As[2] removes the single-buffer __syncthreads stall that
// forced A loads to complete before any compute. B stays direct-from-global (Phase 1 win).
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed_db(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                   __hip_bfloat16* __restrict__ C,
                   const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                   int M, int N, int K) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    constexpr int AVEC = TBM * BK / 16, BKv = BK / 16;
    constexpr int APT  = AVEC / TBLOCK;            // A int4 loads per thread per K-tile
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[2][TBM * BK];           // ping-pong A buffers
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16, NTILES = K / BK;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4 areg[APT];

    // prologue: prefetch tile 0 A -> regs -> As[0]
    #pragma unroll
    for (int i=0;i<APT;++i){ int e=tid+i*TBLOCK; int r=e/BKv,c=e%BKv; int gr=tm+r,gk=c*16; areg[i]=Av[(gr*K+gk)/16]; }
    #pragma unroll
    for (int i=0;i<APT;++i){ int e=tid+i*TBLOCK; reinterpret_cast<int4*>(As[0])[e]=areg[i]; }
    __syncthreads();

    const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
    for (int t = 0; t < NTILES; ++t) {
        const int buf = t & 1, nbuf = (t + 1) & 1, k0 = t * BK;
        // issue next-tile A loads into registers (in flight, overlaps the WMMAs below)
        if (t + 1 < NTILES) {
            const int nk0 = (t + 1) * BK;
            #pragma unroll
            for (int i=0;i<APT;++i){ int e=tid+i*TBLOCK; int r=e/BKv,c=e%BKv; int gr=tm+r,gk=nk0+c*16; areg[i]=Av[(gr*K+gk)/16]; }
        }
        #pragma unroll
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[TFRAGS_M], fb[TFRAGS_N];
            #pragma unroll
            for (int mi=0;mi<TFRAGS_M;++mi){ int lds_row=(wave_m*TFRAGS_M+mi)*16+row_a;
                fa[mi]=*reinterpret_cast<const v2i32*>(As[buf]+lds_row*BK+kbase+colhi*8); }
            #pragma unroll
            for (int ni=0;ni<TFRAGS_N;++ni){ int nt=(tn+(wave_n*TFRAGS_N+ni)*16)/16;
                fb[ni]=tr_load8(Bshuf+b_tile_offset(kt,nt,NT)+lane*8); }
            #pragma unroll
            for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni)
                acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi],fb[ni],acc[mi][ni]);
        }
        // store prefetched next-tile A into the other buffer (waitcnt vmcnt resolves here)
        if (t + 1 < NTILES) {
            #pragma unroll
            for (int i=0;i<APT;++i){ int e=tid+i*TBLOCK; reinterpret_cast<int4*>(As[nbuf])[e]=areg[i]; }
        }
        __syncthreads();
    }

    // ---- epilogue (identical to baseline) ----
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
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed_bslds: stage transposed B in LDS (attack the diagnosed B-feed wall) ----
// Diagnosis: trfeed walls at 46% because B is global_load_tr'd from L2 on the WMMA critical path
// EVERY WMMA (A is LDS-resident, reused 16x; B isn't). Fix: cooperatively global_load_tr the whole
// block's B tile into LDS ONCE per K-tile, in fragment-contiguous layout, then the WMMA loop reads
// B from LDS WIDE (one ds_load_b64, like A) — the hardware transpose makes the LDS store contiguous
// (no byte-scatter, the thing that killed feedwidth_proto). Halves B global ops and takes them off
// the matrix-core critical path. global_load_tr is wave-collective, so the fill is organized
// per-wave (all 32 lanes participate per fragment-tile).
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed_bslds(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                      __hip_bfloat16* __restrict__ C,
                      const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                      int M, int N, int K) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    constexpr int BNT      = BN / 16;              // N fragment-tiles in the block (8)
    constexpr int BFT      = KSTEPS * BNT;          // B fragment-tiles per K-tile (16)
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * BK];
    __shared__ uint8_t  Bsf[BFT * 256];            // block's B tile, fragment-contiguous (4KB)
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += BK) {
        // ---- A fill (wide, K-inner) ----
        constexpr int AVEC = TBM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv; int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        // ---- B fill: each wave global_load_tr's whole fragment-tiles into LDS (wave-collective) ----
        for (int ft = wid; ft < BFT; ft += TWAVES) {
            int k_sub = ft / BNT, n_sub = ft % BNT;
            int kt = (k0 + k_sub * 16) / 16, nt = (tn + n_sub * 16) / 16;
            v2i32 v = tr_load8(Bshuf + b_tile_offset(kt, nt, NT) + lane * 8);
            *reinterpret_cast<v2i32*>(Bsf + ft * 256 + lane * 8) = v;
        }
        __syncthreads();

        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16;
            v2i32 fa[TFRAGS_M], fb[TFRAGS_N];
            for (int mi=0;mi<TFRAGS_M;++mi){ int lds_row=(wave_m*TFRAGS_M+mi)*16+row_a;
                fa[mi]=*reinterpret_cast<const v2i32*>(As+lds_row*BK+kbase+colhi*8); }
            for (int ni=0;ni<TFRAGS_N;++ni){ int ft = kk*BNT + (wave_n*TFRAGS_N+ni);
                fb[ni]=*reinterpret_cast<const v2i32*>(Bsf + ft*256 + lane*8); }   // wide LDS read
            for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni)
                acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi],fb[ni],acc[mi][ni]);
        }
        __syncthreads();
    }

    // ---- epilogue (identical to baseline) ----
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
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed_bk: deepen the K-tile to amortize per-tile fill + syncthreads overhead --
// Diagnosis: bm256/db/bslds all REGRESSED -> the wall isn't operand latency (the direct global_load_tr
// already hides B behind the 16-acc WMMA ILP); it's the per-K-tile fill + 2 __syncthreads, paid 128x
// for K=4096 at BK=32. Deepening BK accumulates MORE WMMAs into the SAME acc[4][4] registers per
// fill/sync (occupancy unchanged — acc size is independent of BK), amortizing that overhead.
template <int TBM, int TWAVES_M, int TBK>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed_bk(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                   __hip_bfloat16* __restrict__ C,
                   const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                   int M, int N, int K) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    constexpr int TKSTEPS  = TBK / 16;
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * TBK];
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += TBK) {
        constexpr int AVEC = TBM * TBK / 16, BKv = TBK / 16;
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv; int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        __syncthreads();
        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        for (int kk = 0; kk < TKSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[TFRAGS_M], fb[TFRAGS_N];
            for (int mi=0;mi<TFRAGS_M;++mi){ int lds_row=(wave_m*TFRAGS_M+mi)*16+row_a;
                fa[mi]=*reinterpret_cast<const v2i32*>(As+lds_row*TBK+kbase+colhi*8); }
            for (int ni=0;ni<TFRAGS_N;++ni){ int nt=(tn+(wave_n*TFRAGS_N+ni)*16)/16;
                fb[ni]=tr_load8(Bshuf+b_tile_offset(kt,nt,NT)+lane*8); }
            for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni)
                acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi],fb[ni],acc[mi][ni]);
        }
        __syncthreads();
    }

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
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed_wg: vary the WORKGROUP wave-grid to trade reuse for occupancy ----------
// Diagnosis: 4 feed/buffer levers flat-to-negative -> wall is the 166-VGPR acc[4][4] capping
// occupancy. Test the OTHER direction: more/smaller waves over the SAME 128x128 tile. TWM x TWN
// waves, each owns (128/TWM)x(128/TWN), so FRAGS shrink and acc[FRAGS_M][FRAGS_N] shrinks ->
// higher occupancy, but each operand load feeds fewer WMMAs (less reuse). Occupancy-bound -> up;
// reuse/issue-bound -> down.
template <int TWM, int TWN>
__global__ void __launch_bounds__(TWM * TWN * WAVE_SIZE)
gemm_fp8_trfeed_wg(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                   __hip_bfloat16* __restrict__ C,
                   const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                   int M, int N, int K) {
    constexpr int WGBM = 128, WGBN = 128;
    constexpr int TWAVES = TWM * TWN, TBLOCK = TWAVES * WAVE_SIZE;
    constexpr int WFRAGS_M = (WGBM / TWM) / 16, WFRAGS_N = (WGBN / TWN) / 16;
    const int tm = blockIdx.y * WGBM, tn = blockIdx.x * WGBN;
    __shared__ float8_t As[WGBM * BK];
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / TWN, wave_n = wid % TWN, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    v8f32 acc[WFRAGS_M][WFRAGS_N];
    for (int mi=0;mi<WFRAGS_M;++mi) for (int ni=0;ni<WFRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int AVEC = WGBM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv; int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        __syncthreads();
        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            v2i32 fa[WFRAGS_M], fb[WFRAGS_N];
            for (int mi=0;mi<WFRAGS_M;++mi){ int lds_row=(wave_m*WFRAGS_M+mi)*16+row_a;
                fa[mi]=*reinterpret_cast<const v2i32*>(As+lds_row*BK+kbase+colhi*8); }
            for (int ni=0;ni<WFRAGS_N;++ni){ int nt=(tn+(wave_n*WFRAGS_N+ni)*16)/16;
                fb[ni]=tr_load8(Bshuf+b_tile_offset(kt,nt,NT)+lane*8); }
            for (int mi=0;mi<WFRAGS_M;++mi) for (int ni=0;ni<WFRAGS_N;++ni)
                acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi],fb[ni],acc[mi][ni]);
        }
        __syncthreads();
    }

    __shared__ float scratch[TWAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < WFRAGS_M; ++mi) {
        for (int ni = 0; ni < WFRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s) ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
            int row0 = tm + (wave_m * WFRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * WFRAGS_N + ni) * 16;
            for (int t = lane; t < 256; t += WAVE_SIZE) {
                int gr = row0 + t / 16, gc = col0 + t % 16;
                if (gr < M && gc < N) {
                    float v = ws[t] * a_scale[gr] * b_scale[gc];
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

// ---------------- trfeed_rb: register-block the whole K-tile (separate loads from a WMMA burst) --
// Diagnosis: wall is WMMA-issue-rate (occupancy tested both ways, all feed levers flat). The 307
// ceiling issues WMMAs back-to-back with operands fixed in registers. Mirror that: hoist ALL of a
// K-tile's fa + fb into registers FIRST, then issue all WMMAs as one burst (loads no longer
// interleave with each WMMA group). Costs ~32 extra VGPR for the staged operands.
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed_rb(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                   __hip_bfloat16* __restrict__ C,
                   const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                   int M, int N, int K) {
    constexpr int TWAVES   = TWAVES_M * WAVES_N;
    constexpr int TBLOCK   = TWAVES * WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BN / WAVES_N) / 16;
    const int tm = blockIdx.y * TBM, tn = blockIdx.x * BN;
    __shared__ float8_t As[TBM * BK];
    const int tid = threadIdx.x, wid = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N, wave_n = wid % WAVES_N, lane = tid % WAVE_SIZE;
    const int NT = N / 16;

    v8f32 acc[TFRAGS_M][TFRAGS_N];
    for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni) acc[mi][ni]=v8f32{0,0,0,0,0,0,0,0};

    const int4* Av = reinterpret_cast<const int4*>(A);
    int4* Asv = reinterpret_cast<int4*>(As);
    for (int k0 = 0; k0 < K; k0 += BK) {
        constexpr int AVEC = TBM * BK / 16, BKv = BK / 16;
        for (int e = tid; e < AVEC; e += TBLOCK) {
            int r = e / BKv, c = e % BKv; int gr = tm + r, gk = k0 + c * 16;
            Asv[e] = Av[(gr * K + gk) / 16];
        }
        __syncthreads();
        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        // hoist ALL operands for the K-tile into registers
        v2i32 fa[KSTEPS][TFRAGS_M], fb[KSTEPS][TFRAGS_N];
        #pragma unroll
        for (int kk=0;kk<KSTEPS;++kk){ int kbase=kk*16, kt=(k0+kbase)/16;
            #pragma unroll
            for (int mi=0;mi<TFRAGS_M;++mi){ int lds_row=(wave_m*TFRAGS_M+mi)*16+row_a;
                fa[kk][mi]=*reinterpret_cast<const v2i32*>(As+lds_row*BK+kbase+colhi*8); }
            #pragma unroll
            for (int ni=0;ni<TFRAGS_N;++ni){ int nt=(tn+(wave_n*TFRAGS_N+ni)*16)/16;
                fb[kk][ni]=tr_load8(Bshuf+b_tile_offset(kt,nt,NT)+lane*8); }
        }
        // then one WMMA burst
        #pragma unroll
        for (int kk=0;kk<KSTEPS;++kk)
            #pragma unroll
            for (int mi=0;mi<TFRAGS_M;++mi) for (int ni=0;ni<TFRAGS_N;++ni)
                acc[mi][ni]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[kk][mi],fb[kk][ni],acc[mi][ni]);
        __syncthreads();
    }

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
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

#else
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * WAVES_N * WAVE_SIZE)
gemm_fp8_trfeed(const float8_t* __restrict__ A, const uint8_t* __restrict__ Bshuf,
                __hip_bfloat16* __restrict__ C,
                const float* __restrict__ a_scale, const float* __restrict__ b_scale,
                int M, int N, int K);
#endif
