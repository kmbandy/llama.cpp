// gemm_trfeed_bench.hip — MAD-305 Phase 1 (unsupervised normal-HIP launch).
//
// Thesis: the real fp8 GEMM walls at ~69-90 TF (22-29% of the proven 307 TF WMMA ceiling)
// because the B operand fragment is byte-gathered out of LDS (8 strided ds_load_u8 ahead of
// every WMMA) while A reads wide. Phase 0 verified (bit-exact) that gfx1201's hardware
// transpose-load `global_load_tr_b64` delivers the §7.12 B fragment in ONE instruction when B
// is pre-shuffled (B is static weights -> repack once, free at runtime; see
// bench/global_load_tr_contract.md). This bench runs the SAME 128x128/BK=32/4-wave tiling two
// ways: the byte-gather baseline (B staged in LDS) vs `gemm_fp8_trfeed` (A still LDS wide-read,
// B fed directly from a pre-shuffled global buffer via global_load_tr_b64, NO LDS for B).
// Correctness is gated against a CPU e4m3 oracle at 256^3 (A != B^T so a layout bug can't hide);
// throughput + baseline/trfeed agreement is measured at a compute-bound 4096^3.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>
#include <rocwmma/rocwmma.hpp>
#include "trfeed_common.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
using namespace rocwmma;

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

// Same tile geometry as the verified kernel (gemm_wmma.hip): 128x128 tile, BK=32, 4 waves.
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WAVES_M = 2, WAVES_N = 2, WAVES = WAVES_M * WAVES_N;
constexpr int WAVE_SIZE = 32, BLOCK_THREADS = WAVES * WAVE_SIZE;
constexpr int FRAGS_M = (BM / WAVES_M) / 16;   // 2
constexpr int FRAGS_N = (BN / WAVES_N) / 16;   // 2
constexpr int KSTEPS  = BK / 16;               // 2

// gfx12 global transpose-load, 8-bit element / b64 form (one wave32 lane gets 8 bytes).
__device__ inline v2i32 tr_load8(const uint8_t* p) {
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}

// ---------------- baseline: byte-gather B feed (B staged in LDS) ----------------
__global__ void __launch_bounds__(BLOCK_THREADS)
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

// ---------------- host ----------------
#define HCK(x) do{ hipError_t e=(x); if(e!=hipSuccess){ \
  fprintf(stderr,"HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 3; } }while(0)

static uint8_t enc(float x){ __hip_fp8_e4m3 v(x); return *reinterpret_cast<uint8_t*>(&v); }
static float   dec(uint8_t b){ __hip_fp8_e4m3 v; *reinterpret_cast<uint8_t*>(&v)=b; return (float)v; }

// Distinct A/B patterns (A != B^T) so a transpose bug cannot hide. Values in [-2,2] step 0.5,
// all exact in e4m3.
static float patA(int i,int k){ return (float)(((i*7 + k*3) % 9) - 4) * 0.5f; }
static float patB(int k,int j){ return (float)(((k*5 + j*2) % 9) - 4) * 0.5f; }

static int build_io(int M,int N,int K,
                    std::vector<uint8_t>&Ah,std::vector<uint8_t>&Bh){
    Ah.resize((size_t)M*K); Bh.resize((size_t)K*N);
    for(int i=0;i<M;i++)for(int k=0;k<K;k++) Ah[(size_t)i*K+k]=enc(patA(i,k));
    for(int k=0;k<K;k++)for(int j=0;j<N;j++) Bh[(size_t)k*N+j]=enc(patB(k,j));
    return 0;
}

static float time_baseline(const float8_t*A,const float8_t*B,__hip_bfloat16*C,
                           const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
    for(int i=0;i<5;i++) gemm_fp8_baseline<<<grid,BLOCK_THREADS>>>(A,B,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_baseline<<<grid,BLOCK_THREADS>>>(A,B,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                         const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed_db(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_db<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_db<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed_bslds(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                               const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_bslds<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_bslds<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M, int TBK>
static float time_trfeed_bk(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_bk<TBM,TWAVES_M,TBK><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_bk<TBM,TWAVES_M,TBK><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TWM, int TWN>
static float time_trfeed_wg(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+128-1)/128,(M+128-1)/128);
    constexpr int TBLOCK = TWM*TWN*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_wg<TWM,TWN><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_wg<TWM,TWN><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

template <int TBM, int TWAVES_M>
static float time_trfeed_rb(const float8_t*A,const uint8_t*Bshuf,__hip_bfloat16*C,
                            const float*as,const float*bs,int M,int N,int K,int iters){
    dim3 grid((N+BN-1)/BN,(M+TBM-1)/TBM);
    constexpr int TBLOCK = TWAVES_M*WAVES_N*WAVE_SIZE;
    for(int i=0;i<5;i++) gemm_fp8_trfeed_rb<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K); // warmup
    hipDeviceSynchronize();
    hipEvent_t s,e; hipEventCreate(&s); hipEventCreate(&e);
    hipEventRecord(s);
    for(int i=0;i<iters;i++) gemm_fp8_trfeed_rb<TBM,TWAVES_M><<<grid,TBLOCK>>>(A,Bshuf,C,as,bs,M,N,K);
    hipEventRecord(e); hipEventSynchronize(e);
    float ms=0; hipEventElapsedTime(&ms,s,e);
    hipEventDestroy(s); hipEventDestroy(e);
    return ms/iters;
}

int main(){
    hipDeviceProp_t p; hipGetDeviceProperties(&p,0);
    printf("device: %s  CUs=%d\n", p.name, p.multiProcessorCount);
    const double CEIL = 307.0, HIPBLASLT = 143.0;

    // ---------- correctness @ 256^3 (the Phase-1 oracle gate) ----------
    {
        const int M=256,N=256,K=256;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        std::vector<float> oref((size_t)M*N,0.f);
        for(int i=0;i<M;i++)for(int j=0;j<N;j++){
            float acc=0; for(int k=0;k<K;k++) acc+=dec(Ah[(size_t)i*K+k])*dec(Bh[(size_t)k*N+j]);
            oref[(size_t)i*N+j]=acc;   // a_scale=b_scale=1
        }
        float8_t *A,*B; uint8_t* Bshuf; __hip_bfloat16*C; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&B,Bh.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&C,(size_t)M*N*2));
        HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(B,Bh.data(),Bh.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N>M?N:M,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
        auto check=[&](const char*name)->float{
            std::vector<uint16_t> Chost((size_t)M*N);
            HCK(hipMemcpy(Chost.data(),C,(size_t)M*N*2,hipMemcpyDeviceToHost));
            float mx=0;
            for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t bits=(uint32_t)Chost[t]<<16; float gv; __builtin_memcpy(&gv,&bits,4);
                float rv=oref[t]; float e=fabsf(gv-rv)/(fabsf(rv)+1e-3f);
                if(e>mx)mx=e;
            }
            printf("  %-9s max_rel_err=%.4f  %s\n",name,mx,mx<0.03f?"PASS":"*** FAIL ***");
            return mx;
        };
        gemm_fp8_baseline<<<grid,BLOCK_THREADS>>>(A,B,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float eb=check("baseline");
        gemm_fp8_trfeed<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float et=check("trfeed128");
        dim3 grid256((N+BN-1)/BN,(M+256-1)/256);
        gemm_fp8_trfeed<256,4><<<grid256,256>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float et2=check("trfeed256");
        gemm_fp8_trfeed_db<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ed=check("trfeed_db");
        gemm_fp8_trfeed_bslds<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float el=check("trfeed_bsl");
        gemm_fp8_trfeed_bk<128,2,64><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ek64=check("trfeed_bk64");
        gemm_fp8_trfeed_bk<128,2,128><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ek128=check("trfeed_bk128");
        gemm_fp8_trfeed_wg<4,4><<<grid,512>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ew44=check("trfeed_wg44");
        gemm_fp8_trfeed_wg<2,4><<<grid,256>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float ew24=check("trfeed_wg24");
        gemm_fp8_trfeed_rb<128,2><<<grid,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize());
        float erb=check("trfeed_rb");
        if(eb>=0.03f||et>=0.03f||et2>=0.03f||ed>=0.03f||el>=0.03f||ek64>=0.03f||ek128>=0.03f||ew44>=0.03f||ew24>=0.03f||erb>=0.03f){ printf("CORRECTNESS GATE FAILED — perf numbers below are meaningless.\n"); }
        hipFree(A);hipFree(B);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
    }

    // ---------- perf @ 4096^3 ----------
    {
        const int M=4096,N=4096,K=4096;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        float8_t *A,*B; uint8_t* Bshuf; __hip_bfloat16*Cb,*Ct,*Ct2,*Cd,*Cl; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&B,Bh.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&Cb,(size_t)M*N*2)); HCK(hipMalloc(&Ct,(size_t)M*N*2));
        HCK(hipMalloc(&Ct2,(size_t)M*N*2)); HCK(hipMalloc(&Cd,(size_t)M*N*2)); HCK(hipMalloc(&Cl,(size_t)M*N*2));
        HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(B,Bh.data(),Bh.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        double gflop = 2.0*M*N*K/1e12;
        float msb  = time_baseline(A,B,Cb,as,bs,M,N,K,30);
        float mst  = time_trfeed<128,2>(A,Bshuf,Ct ,as,bs,M,N,K,30);
        float mst2 = time_trfeed<256,4>(A,Bshuf,Ct2,as,bs,M,N,K,30);
        float msd  = time_trfeed_db<128,2>(A,Bshuf,Cd,as,bs,M,N,K,30);
        float msl  = time_trfeed_bslds<128,2>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msk64 = time_trfeed_bk<128,2,64>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msk128= time_trfeed_bk<128,2,128>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msw44 = time_trfeed_wg<4,4>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msw24 = time_trfeed_wg<2,4>(A,Bshuf,Cl,as,bs,M,N,K,30);
        float msrb  = time_trfeed_rb<128,2>(A,Bshuf,Cl,as,bs,M,N,K,30);
        double tfb=gflop/(msb/1e3), tft=gflop/(mst/1e3), tft2=gflop/(mst2/1e3),
               tfd=gflop/(msd/1e3), tfl=gflop/(msl/1e3),
               tfk64=gflop/(msk64/1e3), tfk128=gflop/(msk128/1e3),
               tfw44=gflop/(msw44/1e3), tfw24=gflop/(msw24/1e3), tfrb=gflop/(msrb/1e3);
        printf("\n=== perf @ %d^3 (ceiling = %.0f TF, hipBLASLt = %.0f TF) ===\n",M,CEIL,HIPBLASLT);
        printf("  baseline (byte-gather)    : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msb,tfb,100*tfb/CEIL,tfb/HIPBLASLT);
        printf("  trfeed bm128 (global_tr)  : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               mst,tft,100*tft/CEIL,tft/HIPBLASLT);
        printf("  trfeed bm256 (amortize B) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               mst2,tft2,100*tft2/CEIL,tft2/HIPBLASLT);
        printf("  trfeed_db bm128 (dbl-buf) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msd,tfd,100*tfd/CEIL,tfd/HIPBLASLT);
        printf("  trfeed_bslds bm128 (B@LDS): %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msl,tfl,100*tfl/CEIL,tfl/HIPBLASLT);
        printf("  trfeed_bk64 bm128 (BK=64) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msk64,tfk64,100*tfk64/CEIL,tfk64/HIPBLASLT);
        printf("  trfeed_bk128 bm128(BK=128): %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msk128,tfk128,100*tfk128/CEIL,tfk128/HIPBLASLT);
        printf("  trfeed_wg 4x4 (16w,acc2x2): %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msw44,tfw44,100*tfw44/CEIL,tfw44/HIPBLASLT);
        printf("  trfeed_wg 2x4 (8w,acc4x2) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msw24,tfw24,100*tfw24/CEIL,tfw24/HIPBLASLT);
        printf("  trfeed_rb bm128 (reg-blk) : %7.3f ms  %6.1f TF  (%.0f%% of 307, %.2fx 143)\n",
               msrb,tfrb,100*tfrb/CEIL,tfrb/HIPBLASLT);
        printf("  speedup bk64 %.2fx bk128 %.2fx wg44 %.2fx wg24 %.2fx rb %.2fx (vs bm128)\n",
               tfk64/tft,tfk128/tft,tfw44/tft,tfw24/tft,tfrb/tft);
        // real-shape agreement: all trfeed variants vs baseline must match (same math).
        std::vector<uint16_t> Hb((size_t)M*N),Ht((size_t)M*N),Ht2((size_t)M*N),Hd((size_t)M*N),Hl((size_t)M*N);
        HCK(hipMemcpy(Hb.data(),Cb,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Ht.data(),Ct,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Ht2.data(),Ct2,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Hd.data(),Cd,(size_t)M*N*2,hipMemcpyDeviceToHost));
        HCK(hipMemcpy(Hl.data(),Cl,(size_t)M*N*2,hipMemcpyDeviceToHost));
        auto maxdiff=[&](std::vector<uint16_t>&H)->double{ double md=0;
            for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t a=(uint32_t)Hb[t]<<16,b=(uint32_t)H[t]<<16; float fa,fb;
                __builtin_memcpy(&fa,&a,4);__builtin_memcpy(&fb,&b,4);
                double d=fabs((double)fa-fb); if(d>md)md=d;
            } return md; };
        double md=maxdiff(Ht), md2=maxdiff(Ht2), mdd=maxdiff(Hd), mdl=maxdiff(Hl);
        printf("  bm128 vs baseline max|Δ|  = %.4g  %s\n",md ,md <1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  bm256 vs baseline max|Δ|  = %.4g  %s\n",md2,md2<1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  db    vs baseline max|Δ|  = %.4g  %s\n",mdd,mdd<1e-2?"(match)":"(*** MISMATCH ***)");
        printf("  bslds vs baseline max|Δ|  = %.4g  %s\n",mdl,mdl<1e-2?"(match)":"(*** MISMATCH ***)");
        hipFree(A);hipFree(B);hipFree(Bshuf);hipFree(Cb);hipFree(Ct);hipFree(Ct2);hipFree(Cd);hipFree(Cl);hipFree(as);hipFree(bs);
    }

    // ---------- Phase-3 FREE PREVIEW: acc-tile reuse sweep — does a BIGGER accumulator
    //   tile raise WMMA/cycle? acc[4][4] (TBM=128) is the 140-TF/7.35-WMMA-per-cyc winner.
    //   Raising FRAGS_M amortizes operand loads over more WMMAs (reuse 2.00->2.22->2.40
    //   WMMA/load) but costs VGPR -> lower static occupancy. All three fit the 256-VGPR
    //   static-wave cap (acc[8][8]=512 would not -> that one needs dyn-VGPR). Dims are
    //   divisible by lcm(128,160,192)=1920 so there is NO edge-tile waste: useful-TF ==
    //   WMMA issue rate, and WMMA/cyc = (TF/307)*15.9 exactly (clock pinned, same ceiling).
    //   NOTE: the trfeed A-fill is unguarded, so TBM MUST divide M/N (hence 1920/3840). ----
    {
        // -- oracle @ 1920x1920x256 (1920 divisible by 128/160/192 -> A-fill in-bounds) --
        const int M=1920,N=1920,K=256;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        std::vector<float> oref((size_t)M*N,0.f);
        for(int i=0;i<M;i++)for(int j=0;j<N;j++){
            float acc=0; for(int k=0;k<K;k++) acc+=dec(Ah[(size_t)i*K+k])*dec(Bh[(size_t)k*N+j]);
            oref[(size_t)i*N+j]=acc;   // a_scale=b_scale=1
        }
        float8_t *A; uint8_t* Bshuf; __hip_bfloat16*C; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&C,(size_t)M*N*2)); HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        auto chk=[&](const char*name)->float{
            std::vector<uint16_t> H((size_t)M*N);
            HCK(hipMemcpy(H.data(),C,(size_t)M*N*2,hipMemcpyDeviceToHost));
            float mx=0; for(size_t t=0;t<(size_t)M*N;t++){
                uint32_t b=(uint32_t)H[t]<<16; float gv; __builtin_memcpy(&gv,&b,4);
                float rv=oref[t], e=fabsf(gv-rv)/(fabsf(rv)+1e-3f); if(e>mx)mx=e; }
            printf("  oracle %-9s max_rel_err=%.4f  %s\n",name,mx,mx<0.03f?"PASS":"*** FAIL ***");
            return mx; };
        printf("\n=== acc-tile reuse sweep — oracle @ %d^3 ===\n",M);
        { dim3 g((N+BN-1)/BN,(M+128-1)/128); gemm_fp8_trfeed<128,2><<<g,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize()); }
        float o4=chk("acc[4][4]");
        { dim3 g((N+BN-1)/BN,(M+160-1)/160); gemm_fp8_trfeed<160,2><<<g,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize()); }
        float o5=chk("acc[5][4]");
        { dim3 g((N+BN-1)/BN,(M+192-1)/192); gemm_fp8_trfeed<192,2><<<g,128>>>(A,Bshuf,C,as,bs,M,N,K); HCK(hipDeviceSynchronize()); }
        float o6=chk("acc[6][4]");
        hipFree(A);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
        if(o4>=0.03f||o5>=0.03f||o6>=0.03f) printf("  acc-tile ORACLE FAILED — perf below is meaningless.\n");
    }
    {
        // -- perf @ 3840x3840x4096 (divisible by 128/160/192; compute-bound) --
        const int M=3840,N=3840,K=4096;
        std::vector<uint8_t> Ah,Bh; build_io(M,N,K,Ah,Bh);
        std::vector<uint8_t> Bshufh((size_t)K*N,0); preshuffle_B(Bh.data(),Bshufh.data(),K,N);
        float8_t *A; uint8_t* Bshuf; __hip_bfloat16*C; float*as,*bs;
        HCK(hipMalloc(&A,Ah.size())); HCK(hipMalloc(&Bshuf,Bshufh.size()));
        HCK(hipMalloc(&C,(size_t)M*N*2)); HCK(hipMalloc(&as,M*4)); HCK(hipMalloc(&bs,N*4));
        HCK(hipMemcpy(A,Ah.data(),Ah.size(),hipMemcpyHostToDevice));
        HCK(hipMemcpy(Bshuf,Bshufh.data(),Bshufh.size(),hipMemcpyHostToDevice));
        std::vector<float> ones(N,1.f);
        HCK(hipMemcpy(as,ones.data(),M*4,hipMemcpyHostToDevice));
        HCK(hipMemcpy(bs,ones.data(),N*4,hipMemcpyHostToDevice));
        double gflop=2.0*M*N*K/1e12;
        float m4=time_trfeed<128,2>(A,Bshuf,C,as,bs,M,N,K,30);
        float m5=time_trfeed<160,2>(A,Bshuf,C,as,bs,M,N,K,30);
        float m6=time_trfeed<192,2>(A,Bshuf,C,as,bs,M,N,K,30);
        double t4=gflop/(m4/1e3),t5=gflop/(m5/1e3),t6=gflop/(m6/1e3);
        const double WPC=15.9;                       // WMMA/cyc at the 307 TF ceiling
        printf("\n=== acc-tile reuse sweep — perf @ %d^3 (ceiling %.0f TF = %.1f WMMA/cyc) ===\n",M,CEIL,WPC);
        printf("  acc[4][4] TBM=128 (128 acc-VGPR, reuse 2.00): %7.3f ms  %6.1f TF  (%2.0f%% of 307)  %.2f WMMA/cyc\n",
               m4,t4,100*t4/CEIL,WPC*t4/CEIL);
        printf("  acc[5][4] TBM=160 (160 acc-VGPR, reuse 2.22): %7.3f ms  %6.1f TF  (%2.0f%% of 307)  %.2f WMMA/cyc\n",
               m5,t5,100*t5/CEIL,WPC*t5/CEIL);
        printf("  acc[6][4] TBM=192 (192 acc-VGPR, reuse 2.40): %7.3f ms  %6.1f TF  (%2.0f%% of 307)  %.2f WMMA/cyc\n",
               m6,t6,100*t6/CEIL,WPC*t6/CEIL);
        printf("  slope acc[4]->acc[6]: %+.1f TF  (%+.2f WMMA/cyc)\n", t6-t4, WPC*(t6-t4)/CEIL);
        printf("  VERDICT: %s\n", t6>t4*1.02 ? "RISES -> reuse lever bends the curve; extrapolate acc[8][8] (needs dyn-VGPR)"
                                 : (t6<t4*0.98 ? "FALLS -> static occupancy loss outweighs reuse; dyn-VGPR REQUIRED to realize big tiles"
                                              : "FLAT -> reuse gain ~cancels static occupancy loss; dyn-VGPR needed to break the tie"));
        hipFree(A);hipFree(Bshuf);hipFree(C);hipFree(as);hipFree(bs);
    }
    return 0;
}
