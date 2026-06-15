#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cassert>
#include "gemm_capi.h"
using namespace rocwmma;

// Native vectors for the raw WMMA path: A/B operand = 2 int32 per lane (8 fp8 bytes); the
// accumulator/result = 8 float32 per lane (the fragment register layouts the gfx12 fp8 WMMA
// intrinsic consumes/produces, per ISA §7.12).
typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

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
// (NACC keeps the matrix unit's ILP saturated; ~166 VGPRs, no spills, occ 9 waves/SIMD).
// Each loaded A/B element feeds BN/16 = BM/16 = 8 MACs along the free dim (the reuse lever).
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int WAVES_M = 2, WAVES_N = 2;
constexpr int WAVES = WAVES_M * WAVES_N;            // 4 waves/block
constexpr int WAVE_SIZE = 32;                        // RDNA4 wave32
constexpr int BLOCK_THREADS = WAVES * WAVE_SIZE;     // 128 lanes
constexpr int FRAGS_M = (BM / WAVES_M) / 16;         // 2 acc frags down M per wave
constexpr int FRAGS_N = (BN / WAVES_N) / 16;         // 2 acc frags across N per wave
constexpr int KSTEPS = BK / 16;                      // 2 WMMA-K steps per LDS tile
// number of WMMA accumulator fragments each wave keeps live
[[maybe_unused]] constexpr int NACC = FRAGS_M * FRAGS_N;

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
    const int lane   = tid % WAVE_SIZE;  // wave-local lane (0-31)

    // Native accumulator: the gfx12 fp8 WMMA intrinsic consumes/produces v8f32 per lane
    // (8 result floats per lane, §7.12 C/D map). We accumulate in these directly and only
    // build a rocWMMA fragment for the store_matrix_sync epilogue.
    v8f32 acc[FRAGS_M][FRAGS_N];
    for (int mi = 0; mi < FRAGS_M; ++mi)
        for (int ni = 0; ni < FRAGS_N; ++ni)
            acc[mi][ni] = v8f32{0,0,0,0,0,0,0,0};

    const float8_t f8zero = static_cast<float8_t>(0.0f);

    // Fully-interior tile? (true for every block of the bench shapes — all multiples of
    // BM/BN/BK). Lets us use vectorized 16B (int4) global->LDS copies. Each As/Bs row is a
    // contiguous run in global (A row in K, B row in N), and gr*K / gk*N are 16B-aligned
    // since K,N % 16 == 0 and k0,tn step by BK,BN. Ragged tiles take the scalar bounds path.
    const bool full_mn = (tm + BM <= M) && (tn + BN <= N);

    // vectorized aliases of the fp8 operands/LDS tiles (loop-invariant)
    const int4* Av = reinterpret_cast<const int4*>(A);
    const int4* Bv = reinterpret_cast<const int4*>(B);
    int4* Asv = reinterpret_cast<int4*>(As);
    int4* Bsv = reinterpret_cast<int4*>(Bs);

    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool full = full_mn && (k0 + BK <= K);
        if (full) {
            // Vectorized: int4 = 16 fp8 bytes. As: BM*BK/16 vecs; Bs: BK*BN/16 vecs.
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

        // Raw-intrinsic inner loop (Task 6). A operand is read WIDE (one ds_load_b64 of the
        // lane's 8 contiguous K-bytes) from row-major As, exploiting the §7.12 fp8 A-map:
        //   lane -> row = lane&0xF, col_hi = (lane>>4)&1; the lane's 2 VGPRs hold the 8
        //   contiguous K-bytes [col_hi*8 .. col_hi*8+7] of that row.
        // B operand (row-major Bs) is gathered per-byte (its 8 K-bytes for a fixed N are
        // BN-strided -> not contiguous); same byte traffic as the rocWMMA baseline. We feed
        // the raw __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 intrinsic directly and
        // accumulate into the rocWMMA accumulator's native vector (acc.x), reusing its
        // store_matrix_sync epilogue. The §7.12 B-map: lane -> col = lane&0xF,
        //   row_hi = (lane>>4)&1; the 2 VGPRs hold K-bytes [row_hi*8 .. row_hi*8+7] of col.
        const int row_a  = lane & 0xF;
        const int colhi  = (lane >> 4) & 1;
        const int col_b  = lane & 0xF;
        const int rowhi  = (lane >> 4) & 1;
        for (int kk = 0; kk < KSTEPS; ++kk) {
            const int kbase = kk * 16;
            v2i32 fa[FRAGS_M];
            v2i32 fb[FRAGS_N];
            for (int mi = 0; mi < FRAGS_M; ++mi) {
                int lds_row = (wave_m * FRAGS_M + mi) * 16 + row_a;
                // 8 contiguous K-bytes -> one ds_load_b64
                fa[mi] = *reinterpret_cast<const v2i32*>(As + lds_row * BK + kbase + colhi * 8);
            }
            for (int ni = 0; ni < FRAGS_N; ++ni) {
                int lds_col = (wave_n * FRAGS_N + ni) * 16 + col_b;
                // NOTE: read RAW BYTES (reinterpret), not (uint8_t)float8_t — the latter does
                // a numeric fp8->int conversion and corrupts the operand.
                const uint8_t* bp = reinterpret_cast<const uint8_t*>(Bs)
                                    + (kbase + rowhi * 8) * BN + lds_col;
                // 8 K-bytes for this fixed N, BN-strided -> per-byte gather (as baseline)
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

    // Epilogue: stage each frag through an LDS f32 scratch (one 16x16 region per wave) so the
    // 256 scaled bf16 outputs are written to global COALESCED (row-major runs). Writing
    // straight from the v8f32 accumulator instead scatters 8 BN-strided stores per lane and
    // is ~3x slower. We refill the scratch from acc via the §7.12 C/D map (lane holds column
    // col=lane&0xF, rows ((lane>>4)&1)*8+slot) instead of store_matrix_sync (same layout, no
    // rocWMMA accumulator-fragment dependency in the hot path).
    __shared__ float scratch[WAVES][16 * 16];
    float* ws = scratch[wid];
    const int e_col = lane & 0xF;
    const int e_rowbase = ((lane >> 4) & 1) * 8;
    for (int mi = 0; mi < FRAGS_M; ++mi) {
        for (int ni = 0; ni < FRAGS_N; ++ni) {
            #pragma unroll
            for (int s = 0; s < 8; ++s)
                ws[(e_rowbase + s) * 16 + e_col] = acc[mi][ni][s];
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

extern "C" void rdna4_gemm_fp8_forward(const void* A, const void* B, void* C,
                                       const float* a_scale, const float* b_scale,
                                       int M, int N, int K, hipStream_t stream) {
    // 2D grid: blockIdx.x -> N tiles, blockIdx.y -> M tiles. Ceil-div so ragged
    // small oracle shapes (16x16x16, 256^3) still cover the full output.
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    assert(K % 16 == 0 && N % 16 == 0);
    gemm_fp8_tiled<<<grid, BLOCK_THREADS, 0, stream>>>(
        (const float8_t*)A, (const float8_t*)B, (__hip_bfloat16*)C, a_scale, b_scale, M, N, K);
}
