// gemm_blockscale.hip — MAD-305 Phase 5, production GGML_TYPE_ML8_FP8 block-scale
// GEMM. Reuses the Phase-1 "trfeed" design verified in bench/gemm_trfeed_bench.hip
// (128x128/BK=32/4-wave tile, B fed straight from global via `global_load_tr_b64`
// out of a pre-shuffled weight buffer, A staged wide in LDS) and adds the
// per-32-wide-K-group block-scale epilogue fold described in gemm_capi.h.
//
// BK is fixed at 32 == the activation/weight scale group width (G=32), so the
// per-K-tile WMMA accumulator IS the per-group partial: no group ever spans a
// K-tile boundary (unlike the ml8 4-bit LUT path in gemm_wmma.hip, whose
// group_size is a *multiple* of BK and folds only at group boundaries).
//
// Two tile instantiations share this body:
//   TBM=128, TWAVES_M=2  (4 waves, acc[4][4] per wave)   — M > 32 (prefill).
//   TBM=32,  TWAVES_M=1  (2 waves, acc[2][4] per wave)   — M <= 32 (decode /
//                                                           speculative verify).
// Both are correct for every M in [1, 4096]: A rows >= M are read as zero
// (never touching out-of-bounds activation memory) and C rows >= M are never
// stored.
//
// gfx1201-only (`global_load_tr_b64` + the gfx12 fp8 WMMA builtin are gfx12
// matrix-core intrinsics): the kernel bodies and their host launchers are
// guarded by `#if defined(__gfx1201__)` so other offload-arch device-code
// passes (e.g. gfx1030, built into the same fat binary) compile this
// translation unit down to an inert stub instead of failing on an unknown
// builtin. The byte-shuffle kernels below don't touch gfx12-specific
// intrinsics and compile for every arch.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstddef>
#include "gemm_capi.h"

typedef int   v2i32 __attribute__((ext_vector_type(2)));
typedef float v8f32 __attribute__((ext_vector_type(8)));

// ─────────────────────────────────────────────────────────────────────────
// global_load_tr_b64 fragment-tile shuffle (verified bench/global_load_tr_contract.md).
// Pure byte permutation — no gfx12-specific instructions, compiles for any arch.
// ─────────────────────────────────────────────────────────────────────────

__host__ __device__ static inline int bs_trperm(int L, int s) {   // L in 0..31, s in 0..7
    int base = (L & 7) + ((L >> 3) & 1) * 32 + ((L >> 4) & 1) * 128;
    return base + (s & 3) * 8 + ((s >> 2) & 1) * 64;
}

__host__ __device__ static inline size_t bs_tile_offset(int kt, int nt, int NT) {
    return (size_t) (kt * NT + nt) * 256;   // 256 bytes per 16x16 fp8 tile
}

// Given tile-local (kl, nl) in [0,16)x[0,16), the (lane L, slot s) that the
// forward preshuffle_B rule (trfeed_common.h) places Blogical[kl*16+nl] into:
// kl = ((L>>4)&1)*8 + s, nl = L&15  =>  L = nl | (((kl>>3)&1)<<4), s = kl&7.
__host__ __device__ static inline void bs_lane_slot(int kl, int nl, int * L, int * s) {
    *L = nl | (((kl >> 3) & 1) << 4);
    *s = kl & 7;
}

// One thread per (k, n) logical element; grid-stride loop. src is the
// transposed-e4m3 [K,N] row-major matrix (K outer, N inner — same convention
// as ml8_fp8_repack_kernel's b_fp8 in ml8.cu); dst is the tile-major
// `global_load_tr_b64` fragment layout (KT*NT tiles of 256 bytes).
__global__ void rdna4_ml8fp8_preshuffle_kernel(
    const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int K, int N) {
    const size_t total = (size_t) K * (size_t) N;
    const int NT = N / 16;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int k = (int) (idx / (size_t) N);
        const int n = (int) (idx % (size_t) N);
        const int kt = k / 16, nt = n / 16, kl = k % 16, nl = n % 16;
        int L, s;
        bs_lane_slot(kl, nl, &L, &s);
        dst[bs_tile_offset(kt, nt, NT) + bs_trperm(L, s)] = src[idx];
    }
}

// Inverse of the above: gather the tile-major fragment layout back into the
// transposed [K,N] row-major byte matrix.
__global__ void rdna4_ml8fp8_unshuffle_kernel(
    const uint8_t * __restrict__ src, uint8_t * __restrict__ dst, int K, int N) {
    const size_t total = (size_t) K * (size_t) N;
    const int NT = N / 16;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int k = (int) (idx / (size_t) N);
        const int n = (int) (idx % (size_t) N);
        const int kt = k / 16, nt = n / 16, kl = k % 16, nl = n % 16;
        int L, s;
        bs_lane_slot(kl, nl, &L, &s);
        dst[idx] = src[bs_tile_offset(kt, nt, NT) + bs_trperm(L, s)];
    }
}

extern "C" hipError_t rdna4_preshuffle_b_ml8fp8(
    const void* b_transposed, void* b_shuf, int K, int N, hipStream_t stream) {
    if (K <= 0 || N <= 0 || (K % 16) != 0 || (N % 16) != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int TPB = 256;
    const size_t total = (size_t) K * (size_t) N;
    const int grid = (int) ((total + TPB - 1) / TPB > 65535u ? 65535u : (total + TPB - 1) / TPB);
    rdna4_ml8fp8_preshuffle_kernel<<<grid, TPB, 0, stream>>>(
        (const uint8_t*) b_transposed, (uint8_t*) b_shuf, K, N);
    return hipGetLastError();
}

extern "C" hipError_t rdna4_unshuffle_b_ml8fp8(
    const void* b_shuf, void* b_transposed, int K, int N, hipStream_t stream) {
    if (K <= 0 || N <= 0 || (K % 16) != 0 || (N % 16) != 0) {
        return hipErrorInvalidValue;
    }
    constexpr int TPB = 256;
    const size_t total = (size_t) K * (size_t) N;
    const int grid = (int) ((total + TPB - 1) / TPB > 65535u ? 65535u : (total + TPB - 1) / TPB);
    rdna4_ml8fp8_unshuffle_kernel<<<grid, TPB, 0, stream>>>(
        (const uint8_t*) b_shuf, (uint8_t*) b_transposed, K, N);
    return hipGetLastError();
}

// ─────────────────────────────────────────────────────────────────────────
// Production block-scale GEMM.
//
// The kernel TEMPLATE below is compiled unconditionally (host pass + every
// device-arch pass, e.g. gfx1030 alongside gfx1201, in the same fat binary):
// only the inner WMMA-compute block that uses gfx12-only intrinsics
// (`global_load_tr_b64`, the gfx12 fp8 WMMA builtin) is guarded by
// `#if defined(__gfx1201__)` -- that macro is defined ONLY during the
// gfx1201 device-code compilation pass, so every other arch's device pass
// (and the host-only pass, which never defines any __gfxNNNN__ macro) takes
// the `#else` stub (accumulator stays zero) instead of failing to lower an
// unknown target intrinsic. `supports_op` (ggml-cuda.cu) gates this whole op
// to RDNA4 devices, so the stub path is a build-safety net, never taken at
// runtime.
// ─────────────────────────────────────────────────────────────────────────

constexpr int BS_BN = 128, BS_BK = 32;          // N-tile / K-tile (== scale group width)
constexpr int BS_WAVES_N = 2;
constexpr int BS_WAVE_SIZE = 32;
constexpr int BS_KSTEPS = BS_BK / 16;           // 2 WMMA-K steps per K-tile

#if defined(__gfx1201__)
__device__ static inline v2i32 bs_tr_load8(const uint8_t* p) {
    auto g = reinterpret_cast<__attribute__((address_space(1))) v2i32*>(
                 reinterpret_cast<uintptr_t>(const_cast<uint8_t*>(p)));
    return __builtin_amdgcn_global_load_tr_b64_v2i32(g);
}
#endif

// Vectorized LDS fill for one K-slice of A (WIDTH bytes wide), shared by both
// kernels below. NEVER falls back to a per-byte scalar gather: the caller's
// row stride is always a multiple of 4 (K%32==0 for ML8_FP8, K%128==0 for
// FP8_B128 both force it via 32K*33/32-style arithmetic -- see the callers),
// so when it isn't ALSO a multiple of 16 (true for several real production
// K-split shapes, e.g. FP8_B128 ffn_down K=4224/1920/12416 -> row stride
// K+K/32 = 4356/1980/12804, none %16==0) this uses a 4-byte (dword)
// vectorized load instead of int4 -- 4x fewer instructions than scalar, not
// 16x fewer like int4 but still far from the "flat per-K-tile wall" a
// per-byte gather caused. `dst` is TBM*WIDTH bytes of shared memory,
// row-major [TBM][WIDTH]. Rows >= M are zero-filled without ever touching
// a_packed (M is the true token count; the activation buffer has no padding
// rows to safely over-read).
template <int TBM, int WIDTH, int TBLOCK>
__device__ inline void bs_fill_a(
    uint8_t * __restrict__ dst, const uint8_t * __restrict__ a_packed,
    int stride_am_bytes, int tm, int k0, int M, int tid, bool aligned16) {
    if (aligned16) {
        constexpr int VEC = TBM * WIDTH / 16, WV = WIDTH / 16;
        int4 * dstv = reinterpret_cast<int4 *>(dst);
        for (int e = tid; e < VEC; e += TBLOCK) {
            const int r = e / WV, c = e % WV;
            const int gr = tm + r;
            dstv[e] = (gr < M)
                ? *reinterpret_cast<const int4 *>(a_packed + (size_t) gr * (size_t) stride_am_bytes + k0 + c * 16)
                : int4{0, 0, 0, 0};
        }
    } else {
        constexpr int VEC = TBM * WIDTH / 4, WV = WIDTH / 4;
        uint32_t * dstv = reinterpret_cast<uint32_t *>(dst);
        for (int e = tid; e < VEC; e += TBLOCK) {
            const int r = e / WV, c = e % WV;
            const int gr = tm + r;
            dstv[e] = (gr < M)
                ? *reinterpret_cast<const uint32_t *>(a_packed + (size_t) gr * (size_t) stride_am_bytes + k0 + c * 4)
                : 0u;
        }
    }
}

// TBM/TWAVES_M select the M-tile: (128,2) for prefill, (32,1) for decode/verify.
// BN=128/WAVES_N=2 fixed (N tile geometry mirrors the Phase-1 trfeed winner).
template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * BS_WAVES_N * BS_WAVE_SIZE)
gemm_ml8fp8_blockscale_kernel(
    const uint8_t * __restrict__ a_packed, int stride_am_bytes,
    const uint8_t * __restrict__ b_shuf, const __half * __restrict__ b_scale_f16,
    float * __restrict__ c, int M, int N, int K) {

    constexpr int TWAVES   = TWAVES_M * BS_WAVES_N;
    constexpr int TBLOCK   = TWAVES * BS_WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BS_BN / BS_WAVES_N) / 16;

    const int tm = blockIdx.y * TBM;
    const int tn = blockIdx.x * BS_BN;
    [[maybe_unused]] const int NT = N / 16;   // only read on the gfx1201 WMMA-compute path below

    __shared__ uint8_t As[TBM * BS_BK];   // row-major [TBM][BK] raw e4m3, K-inner (wide A read)

    const int tid = threadIdx.x, wid = tid / BS_WAVE_SIZE;
    const int wave_m = wid / BS_WAVES_N, wave_n = wid % BS_WAVES_N, lane = tid % BS_WAVE_SIZE;
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;   // WMMA C-fragment map

    // Running fp32 scaled accumulator (register only — this is the GEMM's real
    // output, summed across K-tiles/scale-groups).
    float regacc[TFRAGS_M][TFRAGS_N][8];
    #pragma unroll
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            #pragma unroll
            for (int s = 0; s < 8; ++s) regacc[mi][ni][s] = 0.0f;

    // stride_am_bytes is a->ne[0] in bytes; K%32==0 makes it a multiple of 4
    // always, but only a multiple of 16 (needed for the int4 A-fill below)
    // when K%128==0 -- true for every production shape in the contract
    // (5120/4224/1920/12416/3072/17408 are all %128==0), not guaranteed for
    // an arbitrary K%32==0 test shape. Decided once (uniform across the
    // block/grid -- same for every thread), so this if costs nothing in the
    // hot loop itself, just picks which fill loop runs.
    const bool a_row_16b_aligned = (stride_am_bytes % 16) == 0;

    for (int k0 = 0; k0 < K; k0 += BS_BK) {
        const int g = k0 / BS_BK;   // scale-group index (BK == group width == 32)

        // ---- A fill: wide, K-inner (mirrors gemm_trfeed_bench.hip's int4
        // AVEC loop -- vectorized, never a per-byte scalar gather, which was
        // costing ~16x the instructions of the trfeed hot loop's own A fill
        // and turned out to be the actual "fixed per-K-tile" wall, not the
        // scale fold). Rows >= M are never read (M is the true token count,
        // no padding rows exist in the activation buffer) -- zero-filled.
        bs_fill_a<TBM, BS_BK, TBLOCK>(As, a_packed, stride_am_bytes, tm, k0, M, tid, a_row_16b_aligned);
        __syncthreads();

        // ---- per-K-tile (== per-scale-group) scale-slice preload. Once per
        // tile, NOT per (mi,ni): b_scale's row segment (one scalar per
        // (ni,lane) -- the WMMA C-fragment column is constant across a
        // lane's 8 elements) and a_scale's column slice (one scalar per
        // (mi,lane,s) -- the row varies per s). No global loads happen
        // inside the (mi,ni) fold loop below.
        float bsc[TFRAGS_N];
        #pragma unroll
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            const int gc = tn + (wave_n * TFRAGS_N + ni) * 16 + e_col;
            bsc[ni] = (gc < N) ? __half2float(b_scale_f16[(size_t) g * (size_t) N + gc]) : 0.0f;
        }
        float asc[TFRAGS_M][8];
        #pragma unroll
        for (int mi = 0; mi < TFRAGS_M; ++mi) {
            const int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int gr = row0 + e_rowbase + s;
                asc[mi][s] = (gr < M)
                    ? *reinterpret_cast<const float*>(
                          a_packed + (size_t) gr * (size_t) stride_am_bytes + K + g * 4)
                    : 0.0f;
            }
        }

#if defined(__gfx1201__)
        // Hoist the K-tile's WMMA operands into small register arrays ONCE
        // (TFRAGS_M/TFRAGS_N x KSTEPS fragments -- e.g. 4x2 + 4x4x2 = 8+8
        // v2i32 for the TBM=128 tile), exactly like gemm_trfeed_bench.hip's
        // trfeed_rb variant. Then process the (mi,ni) fragment grid ONE AT A
        // TIME with a single reusable 8-float WMMA accumulator `tmp`,
        // immediately folding it into the persistent `regacc` and discarding
        // it -- so the extra live state on top of `regacc` (the kernel's
        // real, unavoidable running output) is ~8 floats + the small fa/fb
        // arrays, not a second full acc[TFRAGS_M][TFRAGS_N] set. The
        // previous version kept BOTH a full `tileacc[TFRAGS_M][TFRAGS_N]`
        // AND `regacc` live across the fold, doubling the resident
        // accumulator footprint (256 VGPR + spills on the TBM=128 tile) for
        // no reason -- that was the actual "fixed per-K-tile" wall, not the
        // scale multiply itself.
        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        v2i32 fa[TFRAGS_M][BS_KSTEPS], fb[TFRAGS_N][BS_KSTEPS];
        #pragma unroll
        for (int kk = 0; kk < BS_KSTEPS; ++kk) {
            const int kbase = kk * 16, kt = (k0 + kbase) / 16;
            #pragma unroll
            for (int mi = 0; mi < TFRAGS_M; ++mi) {
                const int lds_row = (wave_m * TFRAGS_M + mi) * 16 + row_a;
                fa[mi][kk] = *reinterpret_cast<const v2i32*>(As + lds_row * BS_BK + kbase + colhi * 8);
            }
            #pragma unroll
            for (int ni = 0; ni < TFRAGS_N; ++ni) {
                const int nt = (tn + (wave_n * TFRAGS_N + ni) * 16) / 16;
                // N need not be a multiple of BS_BN (only of 16); guard the
                // last, partial N-tile so we never global_load_tr past the
                // packed buffer's NT tiles.
                fb[ni][kk] = (nt < NT) ? bs_tr_load8(b_shuf + bs_tile_offset(kt, nt, NT) + lane * 8)
                                       : v2i32{0, 0};
            }
        }
        #pragma unroll
        for (int mi = 0; mi < TFRAGS_M; ++mi) {
            #pragma unroll
            for (int ni = 0; ni < TFRAGS_N; ++ni) {
                v8f32 tmp = v8f32{0, 0, 0, 0, 0, 0, 0, 0};
                #pragma unroll
                for (int kk = 0; kk < BS_KSTEPS; ++kk) {
                    tmp = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi][kk], fb[ni][kk], tmp);
                }
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    regacc[mi][ni][s] += tmp[s] * asc[mi][s] * bsc[ni];
                }
            }
        }
#else
        // Non-gfx1201 device pass (e.g. gfx1030 built into the same fat
        // binary): global_load_tr_b64 / the gfx12 fp8 WMMA builtin don't
        // exist for this target. supports_op() never routes here at
        // runtime; regacc simply doesn't accumulate this tile, so this
        // stays a build-safety stub, not a silently-wrong fast path.
        (void) lane; (void) asc; (void) bsc;
#endif
        __syncthreads();   // WAR on As before the next iteration's fill
    }

    // ---- epilogue: direct fp32 store, no LDS scratch needed (register accumulator).
    #pragma unroll
    for (int mi = 0; mi < TFRAGS_M; ++mi) {
        const int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
        #pragma unroll
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            const int col0 = tn + (wave_n * TFRAGS_N + ni) * 16;
            const int gc = col0 + e_col;
            if (gc >= N) continue;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int gr = row0 + e_rowbase + s;
                if (gr < M) {
                    c[(size_t) gr * (size_t) N + gc] = regacc[mi][ni][s];
                }
            }
        }
    }
}

template <int TBM, int TWAVES_M>
static hipError_t launch_blockscale(
    const void* a_packed, int stride_am_bytes, const void* b_shuf, const void* b_scale_f16,
    float* c, int M, int N, int K, hipStream_t stream) {
    constexpr int TWAVES = TWAVES_M * BS_WAVES_N;
    constexpr int TBLOCK = TWAVES * BS_WAVE_SIZE;
    const dim3 grid((N + BS_BN - 1) / BS_BN, (M + TBM - 1) / TBM);
    gemm_ml8fp8_blockscale_kernel<TBM, TWAVES_M><<<grid, TBLOCK, 0, stream>>>(
        (const uint8_t*) a_packed, stride_am_bytes,
        (const uint8_t*) b_shuf, (const __half*) b_scale_f16,
        c, M, N, K);
    return hipGetLastError();
}

extern "C" hipError_t rdna4_gemm_ml8fp8_blockscale(
    const void* a_packed, int stride_am_bytes,
    const void* b_shuf, const void* b_scale_f16,
    float* c, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || (K % 32) != 0 || (N % 16) != 0) {
        return hipErrorInvalidValue;
    }
    // BM=32/TWAVES_M=1 for decode/speculative-verify batches (M<=32); BM=128/
    // TWAVES_M=2 (the Phase-1 trfeed winner's tile) otherwise. Both are exact
    // for every M via the A-load/C-store masking above -- this is a throughput
    // choice (small-M occupancy), not a correctness one.
    if (M <= 32) {
        return launch_blockscale<32, 1>(a_packed, stride_am_bytes, b_shuf, b_scale_f16, c, M, N, K, stream);
    }
    return launch_blockscale<128, 2>(a_packed, stride_am_bytes, b_shuf, b_scale_f16, c, M, N, K, stream);
}

// ─────────────────────────────────────────────────────────────────────────
// GGML_TYPE_FP8_B128 block-scale GEMM (G=128 scale groups, "round 2").
//
// RDNA shares the WMMA and VALU issue port; folding every 32 K (the ML8_FP8
// kernel above, 8 WMMA/wave between scale multiplies) pays ~150% VALU
// overhead on top of the WMMAs themselves -- measured 34-37 TF vs trfeed's
// 131 TF. Folding every 128 K (32 WMMA/wave between scale multiplies, this
// kernel) cuts the per-WMMA VALU overhead to ~1/6th. FP8_B128's on-disk
// block width IS 128, so this reuses the SAME weight layout
// (FP8_B128_LAYOUT_RDNA4, ml8.cu) and the SAME activation format
// (GGML_OP_FP8_QUANT_ROT's G=128 default output) -- no new on-disk format.
//
// Because the block's N-tile is BN=128 == FP8_B128's own scale-tile width,
// b_scale is now literally ONE fp32 scalar per (block, K-group) shared by
// EVERY fragment in the block -- not a per-column slice like the G=32
// kernel needed. a_scale is still one fp32 per row per K-group (unavoidable:
// rows differ per WMMA C-fragment element), preloaded once per K-group and
// reused across all N sub-fragments, same as before.
//
// Tile/fold structure per 128-wide K-group: stage the FULL 128-wide A slice
// for the block's BM rows in LDS in one bs_fill_a call (BM x 128 bytes: 16KB
// at BM=128), then process the (ni, mi) fragment grid with B hoisted PER ni
// (v2i32 fb[4 sub-tiles][2 k-steps] = 8 v2i32, reused across every mi in the
// block, discarded once the next ni starts) and A read fresh from LDS per
// WMMA step (LDS reads are cheap, unlike B's global_load_tr) -- exactly the
// "hoist fb per ni, loop mi inner" fallback shape for when hoisting the
// WHOLE ni x mi grid's B operands doesn't fit under 256 VGPR.
constexpr int BS_G128           = 128;                 // FP8_B128 scale-group width
constexpr int BS_KT_PER_GROUP   = BS_G128 / BS_BK;      // 4 (32-wide sub-tiles per 128-group)

template <int TBM, int TWAVES_M>
__global__ void __launch_bounds__(TWAVES_M * BS_WAVES_N * BS_WAVE_SIZE)
gemm_fp8b128_blockscale_kernel(
    const uint8_t * __restrict__ a_packed, int stride_am_bytes,
    const uint8_t * __restrict__ b_shuf, const float * __restrict__ b_scale_f32,
    float * __restrict__ c, int M, int N, int K) {

    constexpr int TWAVES   = TWAVES_M * BS_WAVES_N;
    constexpr int TBLOCK   = TWAVES * BS_WAVE_SIZE;
    constexpr int TFRAGS_M = (TBM / TWAVES_M) / 16;
    constexpr int TFRAGS_N = (BS_BN / BS_WAVES_N) / 16;

    const int tm = blockIdx.y * TBM;
    const int tn = blockIdx.x * BS_BN;
    const int n_tiles = N / BS_BN;              // == N/128; caller asserts N%128==0
    [[maybe_unused]] const int NT = N / 16;
    const int n_tile = blockIdx.x;              // this block's fixed FP8_B128 scale-tile index

    __shared__ uint8_t As[TBM * BS_G128];       // row-major [TBM][128], 16 KiB at TBM=128

    const int tid = threadIdx.x, wid = tid / BS_WAVE_SIZE;
    const int wave_m = wid / BS_WAVES_N, wave_n = wid % BS_WAVES_N, lane = tid % BS_WAVE_SIZE;
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;

    float regacc[TFRAGS_M][TFRAGS_N][8];
    #pragma unroll
    for (int mi = 0; mi < TFRAGS_M; ++mi)
        #pragma unroll
        for (int ni = 0; ni < TFRAGS_N; ++ni)
            #pragma unroll
            for (int s = 0; s < 8; ++s) regacc[mi][ni][s] = 0.0f;

    // See bs_fill_a: stride is always %4==0 (K%128==0 here) but only
    // sometimes %16==0 (e.g. K=4224/1920/12416 -> stride=4356/1980/12804,
    // none %16==0) -- real FP8_B128 K-split shapes hit this, so the dword
    // fallback (never scalar-byte) matters here too, not just in theory.
    const bool a_row_16b_aligned = (stride_am_bytes % 16) == 0;

    for (int kg = 0; kg < K; kg += BS_G128) {
        const int kb = kg / BS_G128;   // FP8_B128 scale-group index

        bs_fill_a<TBM, BS_G128, TBLOCK>(As, a_packed, stride_am_bytes, tm, kg, M, tid, a_row_16b_aligned);
        __syncthreads();

        // One scalar per (block, K-group) -- the whole 128-wide N tile this
        // block owns shares one FP8_B128 weight scale by construction
        // (N%128==0 asserted at dispatch, so BN==128 IS the scale tile).
        const float bsc = b_scale_f32[(size_t) kb * (size_t) n_tiles + (size_t) n_tile];

        // a_scale: one fp32 per row per 128-K group, preloaded once per
        // K-group and reused across every ni sub-fragment below.
        float asc[TFRAGS_M][8];
        #pragma unroll
        for (int mi = 0; mi < TFRAGS_M; ++mi) {
            const int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int gr = row0 + e_rowbase + s;
                asc[mi][s] = (gr < M)
                    ? *reinterpret_cast<const float*>(
                          a_packed + (size_t) gr * (size_t) stride_am_bytes + K + kb * 4)
                    : 0.0f;
            }
        }

#if defined(__gfx1201__)
        const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
        #pragma unroll
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            const int nt = (tn + (wave_n * TFRAGS_N + ni) * 16) / 16;
            v2i32 fb[BS_KT_PER_GROUP][BS_KSTEPS];   // 4x2 = 8 v2i32, live only for this ni
            #pragma unroll
            for (int kt = 0; kt < BS_KT_PER_GROUP; ++kt) {
                #pragma unroll
                for (int kk = 0; kk < BS_KSTEPS; ++kk) {
                    const int kt_glob = kg / 16 + kt * BS_KSTEPS + kk;   // absolute 16-wide tile idx
                    fb[kt][kk] = (nt < NT) ? bs_tr_load8(b_shuf + bs_tile_offset(kt_glob, nt, NT) + lane * 8)
                                           : v2i32{0, 0};
                }
            }
            #pragma unroll
            for (int mi = 0; mi < TFRAGS_M; ++mi) {
                v8f32 tmp = v8f32{0, 0, 0, 0, 0, 0, 0, 0};
                const int lds_row = (wave_m * TFRAGS_M + mi) * 16 + row_a;
                #pragma unroll
                for (int kt = 0; kt < BS_KT_PER_GROUP; ++kt) {
                    #pragma unroll
                    for (int kk = 0; kk < BS_KSTEPS; ++kk) {
                        const int kbase = kt * BS_BK + kk * 16;
                        const v2i32 fa = *reinterpret_cast<const v2i32*>(
                            As + lds_row * BS_G128 + kbase + colhi * 8);
                        tmp = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa, fb[kt][kk], tmp);
                    }
                }
                #pragma unroll
                for (int s = 0; s < 8; ++s) {
                    regacc[mi][ni][s] += tmp[s] * asc[mi][s] * bsc;
                }
            }
        }
#else
        (void) lane; (void) asc; (void) bsc;
#endif
        __syncthreads();   // WAR on As before the next iteration's fill
    }

    // ---- epilogue: direct fp32 store (identical shape to the ML8_FP8 kernel).
    #pragma unroll
    for (int mi = 0; mi < TFRAGS_M; ++mi) {
        const int row0 = tm + (wave_m * TFRAGS_M + mi) * 16;
        #pragma unroll
        for (int ni = 0; ni < TFRAGS_N; ++ni) {
            const int col0 = tn + (wave_n * TFRAGS_N + ni) * 16;
            const int gc = col0 + e_col;
            if (gc >= N) continue;
            #pragma unroll
            for (int s = 0; s < 8; ++s) {
                const int gr = row0 + e_rowbase + s;
                if (gr < M) {
                    c[(size_t) gr * (size_t) N + gc] = regacc[mi][ni][s];
                }
            }
        }
    }
}

template <int TBM, int TWAVES_M>
static hipError_t launch_fp8b128_blockscale(
    const void* a_packed, int stride_am_bytes, const void* b_shuf, const float* b_scale_f32,
    float* c, int M, int N, int K, hipStream_t stream) {
    constexpr int TWAVES = TWAVES_M * BS_WAVES_N;
    constexpr int TBLOCK = TWAVES * BS_WAVE_SIZE;
    const dim3 grid((N + BS_BN - 1) / BS_BN, (M + TBM - 1) / TBM);
    gemm_fp8b128_blockscale_kernel<TBM, TWAVES_M><<<grid, TBLOCK, 0, stream>>>(
        (const uint8_t*) a_packed, stride_am_bytes,
        (const uint8_t*) b_shuf, b_scale_f32,
        c, M, N, K);
    return hipGetLastError();
}

// Tile selection is purely a function of M (any M in [1, 4096+], including
// M=1..40 decode/speculative-verify batches and a prompt's ragged last
// ubatch) -- see rdna4_gemm_ml8fp8_blockscale's comment, identical reasoning.
extern "C" hipError_t rdna4_gemm_fp8b128_blockscale(
    const void* a_packed, int stride_am_bytes,
    const void* b_shuf, const float* b_scale_f32,
    float* c, int M, int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0 || (K % 128) != 0 || (N % 128) != 0) {
        return hipErrorInvalidValue;
    }
    if (M <= 32) {
        return launch_fp8b128_blockscale<32, 1>(a_packed, stride_am_bytes, b_shuf, b_scale_f32, c, M, N, K, stream);
    }
    return launch_fp8b128_blockscale<128, 2>(a_packed, stride_am_bytes, b_shuf, b_scale_f32, c, M, N, K, stream);
}
