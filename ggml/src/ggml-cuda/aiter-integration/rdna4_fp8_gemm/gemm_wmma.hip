#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include "gemm_capi.h"
using namespace rocwmma;

// LDS-tiled WMMA core (P1, Task 4): each block computes a BM x BN output tile from
// multiple WMMA fragments, staging A (BM x BK) and B (BK x BN) through LDS so every
// loaded fp8 element feeds many MACs (the reuse that the naive per-tile kernel lacks).
//
// Layout (matches the oracle-proven naive kernel exactly — same fragment types/ld):
//   matrix_a row_major from As, ld = BK  (As is row-major [BM][BK])
//   matrix_b row_major from Bs, ld = BN  (Bs is row-major [BK][BN])
//   A[M,K] and B[K,N] are row-major in global; the global->LDS copy preserves that.
//
// Tunables: Task 6 sweeps these. Constraints baked in below assume BM/BN multiples
// of 16 (WMMA M/N) and BK a multiple of 16 (WMMA K). With BM=BN=128, BK=32 and 4 waves
// (WAVES_M=WAVES_N=2), each wave owns a 64x64 region = 4x4 = 16 accumulator fragments
// (NACC=16 keeps the matrix unit's ILP saturated; ~166 VGPRs, no spills, occ 9 waves/SIMD).
// Each loaded A/B element feeds BN/16 = BM/16 = 8 MACs along the free dim (the reuse lever).
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WAVES_M = 2, WAVES_N = 2;
constexpr int WAVES = WAVES_M * WAVES_N;            // 4 waves/block
constexpr int WAVE_SIZE = 32;                        // RDNA4 wave32
constexpr int BLOCK_THREADS = WAVES * WAVE_SIZE;     // 128 lanes
constexpr int FRAGS_M = (BM / WAVES_M) / 16;         // 2 acc frags down M per wave
constexpr int FRAGS_N = (BN / WAVES_N) / 16;         // 2 acc frags across N per wave
constexpr int KSTEPS = BK / 16;                      // 2 WMMA-K steps per LDS tile

__global__ void __launch_bounds__(BLOCK_THREADS)
gemm_fp8_tiled(const float8_t* __restrict__ A, const float8_t* __restrict__ B,
               __hip_bfloat16* __restrict__ C,
               const float* __restrict__ a_scale, const float* __restrict__ b_scale,
               int M, int N, int K) {
    const int tm = blockIdx.y * BM;   // block C tile row origin (M)
    const int tn = blockIdx.x * BN;   // block C tile col origin (N)

    __shared__ float8_t As[BM * BK];  // row-major [BM][BK], ld = BK
    __shared__ float8_t Bs[BK * BN];  // row-major [BK][BN], ld = BN

    const int tid    = threadIdx.x;
    const int wid    = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N;
    const int wave_n = wid % WAVES_N;

    fragment<accumulator, 16, 16, 16, float> acc[FRAGS_M][FRAGS_N];
    for (int mi = 0; mi < FRAGS_M; ++mi)
        for (int ni = 0; ni < FRAGS_N; ++ni)
            fill_fragment(acc[mi][ni], 0.0f);

    const float8_t f8zero = (float8_t)0.0f;

    // Fully-interior tile? (true for every block of the bench shapes — all multiples of
    // BM/BN/BK). Lets us use vectorized 16B (int4) global->LDS copies. Each As/Bs row is a
    // contiguous run in global (A row in K, B row in N), and gr*K / gk*N are 16B-aligned
    // since K,N % 16 == 0 and k0,tn step by BK,BN. Ragged tiles take the scalar bounds path.
    const bool full_mn = (tm + BM <= M) && (tn + BN <= N);

    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool full = full_mn && (k0 + BK <= K);
        if (full) {
            // Vectorized: int4 = 16 fp8 bytes. As: BM*BK/16 vecs; Bs: BK*BN/16 vecs.
            const int4* Av = reinterpret_cast<const int4*>(A);
            const int4* Bv = reinterpret_cast<const int4*>(B);
            int4* Asv = reinterpret_cast<int4*>(As);
            int4* Bsv = reinterpret_cast<int4*>(Bs);
            constexpr int AVEC = BM * BK / 16;          // int4 count for As
            constexpr int BVEC = BK * BN / 16;          // int4 count for Bs
            constexpr int BKv = BK / 16;                // int4 per A row
            constexpr int BNv = BN / 16;                // int4 per B row
            for (int e = tid; e < AVEC; e += BLOCK_THREADS) {
                int r = e / BKv, c = e % BKv;           // row within tile, vec-col
                int gr = tm + r, gk = k0 + c * 16;
                Asv[e] = Av[(gr * K + gk) / 16];
            }
            for (int e = tid; e < BVEC; e += BLOCK_THREADS) {
                int r = e / BNv, c = e % BNv;           // k-row within tile, vec-col
                int gk = k0 + r, gc = tn + c * 16;
                Bsv[e] = Bv[(gk * N + gc) / 16];
            }
        } else {
            // Scalar bounds-safe path: zero-pad ragged M/N/K edges so out-of-range WMMA
            // lanes contribute 0 (oracle's 16x16x16 / 256^3 cases).
            for (int e = tid; e < BM * BK; e += BLOCK_THREADS) {
                int r = e / BK, c = e % BK;
                int gr = tm + r, gk = k0 + c;
                As[e] = (gr < M && gk < K) ? A[gr * K + gk] : f8zero;
            }
            for (int e = tid; e < BK * BN; e += BLOCK_THREADS) {
                int r = e / BN, c = e % BN;
                int gk = k0 + r, gc = tn + c;
                Bs[e] = (gk < K && gc < N) ? B[gk * N + gc] : f8zero;
            }
        }
        __syncthreads();

        // Cache this wave's A/B fragments per K-step, then MAC the FRAGS_M x FRAGS_N grid.
        for (int kk = 0; kk < KSTEPS; ++kk) {
            fragment<matrix_a, 16, 16, 16, float8_t, row_major> fa[FRAGS_M];
            fragment<matrix_b, 16, 16, 16, float8_t, row_major> fb[FRAGS_N];
            for (int mi = 0; mi < FRAGS_M; ++mi) {
                int lds_row = (wave_m * FRAGS_M + mi) * 16;      // row within As tile
                load_matrix_sync(fa[mi], As + lds_row * BK + kk * 16, BK);
            }
            for (int ni = 0; ni < FRAGS_N; ++ni) {
                int lds_col = (wave_n * FRAGS_N + ni) * 16;      // col within Bs tile
                load_matrix_sync(fb[ni], Bs + (kk * 16) * BN + lds_col, BN);
            }
            for (int mi = 0; mi < FRAGS_M; ++mi)
                for (int ni = 0; ni < FRAGS_N; ++ni)
                    mma_sync(acc[mi][ni], fa[mi], fb[ni], acc[mi][ni]);
        }
        __syncthreads();
    }

    // Epilogue: each wave writes its FRAGS_M x FRAGS_N frags. Scale via an LDS f32
    // scratch (one 16x16 region per wave), then a_scale[row]*b_scale[col] -> bf16.
    __shared__ float scratch[WAVES][16 * 16];
    float* ws = scratch[wid];
    for (int mi = 0; mi < FRAGS_M; ++mi) {
        for (int ni = 0; ni < FRAGS_N; ++ni) {
            store_matrix_sync(ws, acc[mi][ni], 16, mem_row_major);
            // 32 lanes cooperatively scale+cast this 16x16 frag.
            int row0 = tm + (wave_m * FRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * FRAGS_N + ni) * 16;
            for (int t = tid % WAVE_SIZE; t < 256; t += WAVE_SIZE) {
                int gr = row0 + t / 16, gc = col0 + t % 16;
                if (gr < M && gc < N) {
                    float v = ws[t] * a_scale[gr] * b_scale[gc];
                    C[gr * N + gc] = (__hip_bfloat16)v;
                }
            }
        }
    }
}

extern "C" void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                                       const float* a_scale, const float* b_scale,
                                       int M, int N, int K, hipStream_t stream) {
    // 2D grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles. Ceil-div so ragged
    // small oracle shapes (16x16x16, 256^3) still cover the full output.
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_fp8_tiled<<<grid, BLOCK_THREADS, 0, stream>>>(
        (const float8_t*)A, (const float8_t*)B, (__hip_bfloat16*)C, a_scale, b_scale, M, N, K);
}
