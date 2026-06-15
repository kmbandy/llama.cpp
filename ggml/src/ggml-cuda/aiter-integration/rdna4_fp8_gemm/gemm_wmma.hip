#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cassert>
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

// Unified tiled WMMA core, templated on the B front-end.
//   ML8 == false : plain fp8. B is float8_t[K,N] row-major; b_scale is per-col b_scale[N];
//                  applied as a single epilogue multiply (B scale is constant across K).
//   ML8 == true  : 4-bit LUT-dequant. B_idx is packed nibbles [K/2,N] uint8 (lo-first);
//                  centroids_fp8 is [n_groups_k,16]; b_scale is per-(group,N) [n_groups_k,N].
//                  The global->LDS B fill unpacks a nibble and looks up an LDS-resident
//                  16-entry fp8 centroid LUT (staged once per K-group) into the same Bs tile.
//                  Because the B scale varies per K-group within the K-reduction, it is
//                  FOLDED PER K-GROUP: at each group boundary we scale that group's partial
//                  WMMA accumulator by b_scale[g,n] and add into a register fp32 accumulator,
//                  then re-zero the WMMA acc for the next group (the fp8 path keeps a single
//                  epilogue multiply, unchanged). The A fill, WMMA compute and accumulator
//                  structure are IDENTICAL to the fp8 path.
template <bool ML8>
__device__ __forceinline__ void gemm_tiled_body(
    const float8_t* __restrict__ A, const void* __restrict__ B,
    __hip_bfloat16* __restrict__ C,
    const float* __restrict__ a_scale, const float* __restrict__ b_scale,
    const float8_t* __restrict__ centroids, int group_size, int M, int N, int K) {
    const int tm = blockIdx.y * BM;   // block C tile row origin (M)
    const int tn = blockIdx.x * BN;   // block C tile col origin (N)

    __shared__ float8_t As[BM * BK];  // row-major [BM][BK], ld = BK
    __shared__ float8_t Bs[BK * BN];  // row-major [BK][BN], ld = BN
    __shared__ float8_t lut[16];      // ml8: this K-group's 16 fp8 centroids (staged once/group)
    // ml8: this K-group's BN b_scale values for the tile's columns [tn, tn+BN), staged once per
    // group (coalesced) so the per-group fold reads LDS, not global, on every group boundary.
    __shared__ float bscale_s[ML8 ? BN : 1];

    const int tid    = threadIdx.x;
    const int wid    = tid / WAVE_SIZE;
    const int wave_m = wid / WAVES_N;
    const int wave_n = wid % WAVES_N;
    const int lane   = tid % WAVE_SIZE;  // wave-local lane (0-31)

    fragment<accumulator, 16, 16, 16, float> acc[FRAGS_M][FRAGS_N];
    for (int mi = 0; mi < FRAGS_M; ++mi)
        for (int ni = 0; ni < FRAGS_N; ++ni)
            fill_fragment(acc[mi][ni], 0.0f);

    // ml8 register scaled accumulator. The rocWMMA gfx1201 wave32 16x16x16 f32 accumulator
    // layout (probed) is deterministic: lane l, element i -> (row = i + (l/16)*8, col = l%16).
    // The COLUMN is constant across a lane's 8 elements (= l%16), so the per-(group,col) B scale
    // is ONE scalar per (frag,lane) -> we fold it in registers with NO store_matrix_sync and NO
    // LDS scaled accumulator (both of which blew up gfx1201's LDS/occupancy). facc holds the
    // sum over K-groups of (group WMMA partial) * b_scale[g, col]. Sized 1 for the fp8 path.
    constexpr int EPL = ML8 ? 8 : 1;   // acc elements per lane (16*16/32)
    float facc[ML8 ? FRAGS_M : 1][ML8 ? FRAGS_N : 1][EPL];
    if (ML8) {
        for (int mi = 0; mi < FRAGS_M; ++mi)
            for (int ni = 0; ni < FRAGS_N; ++ni)
                for (int i = 0; i < EPL; ++i) facc[mi][ni][i] = 0.0f;
    }

    const float8_t f8zero = static_cast<float8_t>(0.0f);

    // Fully-interior tile? (true for every block of the bench shapes — all multiples of
    // BM/BN/BK). Lets us use vectorized 16B (int4) global->LDS copies. Each As/Bs row is a
    // contiguous run in global (A row in K, B row in N), and gr*K / gk*N are 16B-aligned
    // since K,N % 16 == 0 and k0,tn step by BK,BN. Ragged tiles take the scalar bounds path.
    const bool full_mn = (tm + BM <= M) && (tn + BN <= N);

    // vectorized aliases of the fp8 operands/LDS tiles (loop-invariant)
    const int4* Av = reinterpret_cast<const int4*>(A);
    const float8_t* Bf8 = reinterpret_cast<const float8_t*>(B);      // fp8 path
    const uint8_t*  Bidx = reinterpret_cast<const uint8_t*>(B);      // ml8 path (packed nibbles)
    const int4* Bv = reinterpret_cast<const int4*>(B);
    int4* Asv = reinterpret_cast<int4*>(As);
    int4* Bsv = reinterpret_cast<int4*>(Bs);

    int cur_group = -1;   // ml8: which K-group's centroids are currently staged in lut[]

    for (int k0 = 0; k0 < K; k0 += BK) {
        const bool full = full_mn && (k0 + BK <= K);

        // ----- A fill (IDENTICAL to fp8 path) -----
        if (full) {
            constexpr int AVEC = BM * BK / 16;          // int4 count for As
            constexpr int BKv = BK / 16;                // int4 per A row
            for (int e = tid; e < AVEC; e += BLOCK_THREADS) {
                int r = e / BKv, c = e % BKv;           // row within tile, vec-col
                int gr = tm + r, gk = k0 + c * 16;
                Asv[e] = Av[(gr * K + gk) / 16];
            }
        } else {
            for (int e = tid; e < BM * BK; e += BLOCK_THREADS) {
                int r = e / BK, c = e % BK;
                int gr = tm + r, gk = k0 + c;
                As[e] = (gr < M && gk < K) ? A[gr * K + gk] : f8zero;
            }
        }

        // ----- B fill -----
        if (!ML8) {
            // plain fp8: copy bytes (vectorized when interior, scalar+zero-pad on edges).
            if (full) {
                constexpr int BVEC = BK * BN / 16;
                constexpr int BNv  = BN / 16;
                for (int e = tid; e < BVEC; e += BLOCK_THREADS) {
                    int r = e / BNv, c = e % BNv;
                    int gk = k0 + r, gc = tn + c * 16;
                    Bsv[e] = Bv[(gk * N + gc) / 16];
                }
            } else {
                for (int e = tid; e < BK * BN; e += BLOCK_THREADS) {
                    int r = e / BN, c = e % BN;
                    int gk = k0 + r, gc = tn + c;
                    Bs[e] = (gk < K && gc < N) ? Bf8[gk * N + gc] : f8zero;
                }
            }
        } else {
            // ml8: stage this K-group's 16 centroids into LDS once per group, then
            // unpack a 4-bit index per (k-row,n) and look up the fp8 centroid -> Bs.
            const int g = k0 / group_size;
            if (g != cur_group) {
                if (tid < 16) lut[tid] = centroids[g * 16 + tid];
                // Stage this group's BN b_scale columns once (coalesced) for the fold.
                for (int e = tid; e < BN; e += BLOCK_THREADS) {
                    int gc = tn + e;
                    bscale_s[e] = (gc < N) ? b_scale[g * N + gc] : 0.0f;
                }
                cur_group = g;
            }
            __syncthreads();   // lut[] + bscale_s visible to all lanes
            // Stage the 16-entry LUT into registers (avoids repeated LDS reads in the hot loop).
            float8_t lutr[16];
            #pragma unroll
            for (int i = 0; i < 16; ++i) lutr[i] = lut[i];
            if (full) {
                // Vectorized: one packed byte (gk/2, gc) holds BOTH k-rows gk(lo) & gk+1(hi),
                // so load 16 packed bytes (uint4) per thread -> 16 cols x 2 k-rows of Bs.
                // Bidx is [K/2,N]; packed row pr = k0/2 + r maps to Bs rows 2r (lo) & 2r+1 (hi).
                const uint4* Bidxv = reinterpret_cast<const uint4*>(Bidx);
                constexpr int PR    = BK / 2;            // packed rows per tile (=16)
                constexpr int BNv   = BN / 16;           // uint4 groups across N (=8)
                constexpr int NPACK = PR * BNv;          // total uint4 loads (=128 == threads)
                for (int e = tid; e < NPACK; e += BLOCK_THREADS) {
                    int r = e / BNv, c = e % BNv;        // packed-row within tile, vec-col
                    int gpr = (k0 >> 1) + r, gc = tn + c * 16;
                    uint4 packed = Bidxv[(gpr * N + gc) / 16];
                    const uint8_t* pb = reinterpret_cast<const uint8_t*>(&packed);
                    float8_t* lo = &Bs[(2 * r)     * BN + c * 16];
                    float8_t* hi = &Bs[(2 * r + 1) * BN + c * 16];
                    #pragma unroll
                    for (int j = 0; j < 16; ++j) {
                        uint8_t byte = pb[j];
                        lo[j] = lutr[byte & 0x0F];
                        hi[j] = lutr[byte >> 4];
                    }
                }
            } else {
                for (int e = tid; e < BK * BN; e += BLOCK_THREADS) {
                    int r = e / BN, c = e % BN;          // k-row within tile, col within tile
                    int gk = k0 + r, gc = tn + c;
                    if (gk < K && gc < N) {
                        // B_idx packed [K/2,N] lo-first: byte (gk/2, gc); lo nibble is gk even.
                        uint8_t byte = Bidx[(gk >> 1) * N + gc];
                        int idx = (gk & 1) ? (byte >> 4) : (byte & 0x0F);
                        Bs[e] = lutr[idx];
                    } else {
                        Bs[e] = f8zero;
                    }
                }
            }
        }
        __syncthreads();

        // ----- WMMA compute (IDENTICAL to fp8 path) -----
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

        // ----- ml8: fold this K-group's per-(group,n) B scale at the group boundary -----
        // group_size is a multiple of BK, so a boundary always lands on a BK boundary. Each lane
        // scales its 8 accumulator elements (all in column l%16) by the single scalar b_scale[g,
        // col], adds into the register scaled accumulator, then re-zeros the WMMA acc for the
        // next K-group. No LDS / store_matrix_sync — the column is constant across a lane's elts.
        if (ML8 && ((k0 + BK) % group_size == 0 || k0 + BK >= K)) {
            const int col_lane = lane % 16;           // this lane's accumulator column (probed)
            for (int mi = 0; mi < FRAGS_M; ++mi) {
                for (int ni = 0; ni < FRAGS_N; ++ni) {
                    // tile-local column -> staged b_scale (loaded once per group, coalesced).
                    float bs = bscale_s[(wave_n * FRAGS_N + ni) * 16 + col_lane];
                    #pragma unroll
                    for (int i = 0; i < EPL; ++i)
                        facc[mi][ni][i] += acc[mi][ni].x[i] * bs;
                    fill_fragment(acc[mi][ni], 0.0f);   // reset for the next K-group
                }
            }
            // Barrier so no thread re-stages bscale_s (next group) while a lagging thread is
            // still reading this group's bscale_s in the fold above (WAR safety).
            __syncthreads();
        }
    }

    // Epilogue: write each wave's FRAGS_M x FRAGS_N frags.
    //   fp8 : scale via LDS f32 scratch (store_matrix_sync), a_scale[row]*b_scale[col] -> bf16.
    //   ml8 : the already-B-scaled register accumulator * a_scale[row] -> bf16, written via the
    //         probed accumulator layout (lane l, elt i -> row = i + (l/16)*8, col = l%16). No
    //         store_matrix_sync needed, so the ml8 path adds no LDS over the fp8 path.
    __shared__ float scratch[WAVES][16 * 16];
    float* ws = scratch[wid];
    for (int mi = 0; mi < FRAGS_M; ++mi) {
        for (int ni = 0; ni < FRAGS_N; ++ni) {
            int row0 = tm + (wave_m * FRAGS_M + mi) * 16;
            int col0 = tn + (wave_n * FRAGS_N + ni) * 16;
            if (!ML8) {
                store_matrix_sync(ws, acc[mi][ni], 16, mem_row_major);
                for (int t = lane; t < 256; t += WAVE_SIZE) {
                    int gr = row0 + t / 16, gc = col0 + t % 16;
                    if (gr < M && gc < N) {
                        float v = ws[t] * a_scale[gr] * b_scale[gc];
                        C[gr * N + gc] = (__hip_bfloat16)v;
                    }
                }
            } else {
                const int row_base = row0 + (lane / 16) * 8;
                const int gc = col0 + (lane % 16);
                #pragma unroll
                for (int i = 0; i < EPL; ++i) {
                    int gr = row_base + i;
                    if (gr < M && gc < N) {
                        float v = facc[mi][ni][i] * a_scale[gr];
                        C[gr * N + gc] = (__hip_bfloat16)v;
                    }
                }
            }
        }
    }
}

__global__ void __launch_bounds__(BLOCK_THREADS)
gemm_fp8_tiled(const float8_t* __restrict__ A, const float8_t* __restrict__ B,
               __hip_bfloat16* __restrict__ C,
               const float* __restrict__ a_scale, const float* __restrict__ b_scale,
               int M, int N, int K) {
    gemm_tiled_body<false>(A, B, C, a_scale, b_scale, nullptr, 0, M, N, K);
}

__global__ void __launch_bounds__(BLOCK_THREADS)
gemm_ml8_tiled(const float8_t* __restrict__ A, const uint8_t* __restrict__ B_idx,
               __hip_bfloat16* __restrict__ C,
               const float* __restrict__ a_scale, const float8_t* __restrict__ centroids,
               const float* __restrict__ b_group_scale, int M, int N, int K, int group_size) {
    gemm_tiled_body<true>(A, B_idx, C, a_scale, b_group_scale, centroids, group_size, M, N, K);
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

extern "C" void rdna4_gemm_ml8_forward(const void* A, const void* B_idx, void* C,
                                       const float* a_scale, const void* centroids_fp8,
                                       const float* b_group_scale,
                                       int M, int N, int K, int group_size, hipStream_t stream) {
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    assert(K % 16 == 0 && N % 16 == 0);
    // group_size must be a multiple of BK so each K-group's per-(group,n) scale aligns with
    // a BK tile boundary (the scale is folded at the boundary, never mid-tile).
    assert(group_size % BK == 0 && K % group_size == 0);
    gemm_ml8_tiled<<<grid, BLOCK_THREADS, 0, stream>>>(
        (const float8_t*)A, (const uint8_t*)B_idx, (__hip_bfloat16*)C,
        a_scale, (const float8_t*)centroids_fp8, b_group_scale, M, N, K, group_size);
}
