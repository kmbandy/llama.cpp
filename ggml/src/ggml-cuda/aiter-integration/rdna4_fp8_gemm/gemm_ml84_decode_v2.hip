// gemm_ml84_decode_v2.hip — MAD-305 ML8_4 (4.5 bpw) decode-band GEMM v2.
//
// Replaces gemm_ml84_trfeed_splitk<32,1,*> (bench/trfeed_ml84_kernels.h,
// launched by rdna4_gemm_ml84_trfeed_decode_splitk in gemm_ml84_prod.hip)
// on the decode path: real M<=32 (typically M=1..8 for K7 DFlash verify).
// The old kernel ALWAYS computes a fixed 32-row tile (M_pad, not real M)
// with a fixed BN=128/WAVES_N=2 split and, at n_splits>1, a hipMemsetAsync
// of the WHOLE M_pad*N output before every launch plus (at the ml8.cu call
// site) a D2D copy-out when M != M_pad. Measured 147-189 us on the R9700
// at N=5120,K=5120 (74-95 GB/s of a 640 GB/s roof).
//
// STRUCTURE ported from ~/GitHub/ggz14-vllm-mxfp4/radiance_mxfp4_fp8.hip's
// `radiance_mxfp4_fp8_gemm_decode` (lines 770-1120 at the time this was
// written) — NOT its MXFP4 dequant math, which is unrelated to ml8-4's
// nibble/LUT/per-group-scale format:
//   * DTM = smallest 16-row M-tile covering real M (1 for M<=16, 2 for
//     M<=32) instead of a fixed 32-row tile — radiance_mxfp4_fp8.hip:791-799.
//   * DWN=8 waves/block, each wave owns one 16-wide N-strip (BN=128 total)
//     — radiance_mxfp4_fp8.hip's DEC_DWN=8 (:1043).
//   * DBK=128 K-slab (radiance_mxfp4_fp8.hip:775 picks 128 for the decode
//     band, reversing the prefill tile's BK=64 choice — occupancy argument
//     at :773-776); falls back to DBK=64 when K%128!=0 (ml8-4's K is a
//     multiple of QK_ML8=64, not guaranteed a multiple of 128 the way
//     radiance's K%128 check at :1188-1191 handles its own TP-padded K).
//   * grid.z=DKS in {1,2,4} split-K chosen by a workgroup fill target,
//     radiance's split_k_for (:1099-1112: fill=110 at M<=24 else 78,
//     narrowest split that clears nblk*ks>=fill) — copied verbatim as
//     ml84_decode_v2_split_k_for below, same constants, same rationale
//     (partial-buffer traffic rises linearly with M while the fill benefit
//     does not, so the target step-down at M>24 is a property of the
//     kernel, not this format).
//   * fp32 partials P[ks][m][n] with a per-N-block atomic counter cnt[]
//     where the LAST-ARRIVING split block does the reduce-and-store and
//     resets its own counter to 0 for the next launch (radiance
//     :1050-1071) — no hipMemsetAsync, no second kernel, no D2D copy-out.
//     DKS==1 skips the partial buffer entirely and writes C_f32 directly
//     (radiance's DIRECT/DKS==1 fast path, :1031-1043).
//   * accumulators zeroed in registers (never memset from the host).
//   * __builtin_nontemporal_load on the B_nib weight stream (read exactly
//     once per K-step; radiance's `ld_w<NT>`, :779-784), gated by
//     MT_ML8_4_DECODE_V2_NT (default OFF, measured slower on RDNA4) like radiance's
//     RADIANCE_MXFP4_DECODE_NT / dec_nt.
//   * NOT DONE (reverted): an earlier version of this file additionally
//     forced the wave-uniform B_nib tile base and the block-uniform
//     per-group LUT base through a manual 64-bit
//     __builtin_amdgcn_readfirstlane hi/lo split, mirroring radiance's own
//     uniform-base addressing (:930 area). That had a real bug --
//     __builtin_amdgcn_readfirstlane takes/returns a SIGNED int on this
//     toolchain, so reconstructing a 64-bit pointer as
//     `(hi<<32)|readfirstlane(lo)` sign-extends `lo` to all-1s in the top
//     32 bits whenever the pointer's low 32 bits have their top bit set,
//     and OR can never clear those spurious bits back out -- it
//     permanently clobbers the real high 32 bits. This is what produced
//     the reported memory fault (address pattern 0xffff...B2C0E000: real
//     low 32 bits preserved, real high 32 bits destroyed to all-1s by the
//     sign-extending OR). `kt`/`nt`/`g` are already loop/wave-invariant by
//     construction (see the per-symbol comments below), so every lane
//     computes the bit-identical address WITHOUT any manual broadcast --
//     the readfirstlane call was only ever an unverified SGPR-promotion
//     hint, never load-bearing for correctness, so it is simply removed
//     rather than "fixed", and the AMDGPU backend's own uniformity
//     analysis is trusted to promote these addresses to SGPRs on its own.
//   * clamp-not-predicate bounds handling for the A LDS fill (radiance
//     :828 `const int rc = r < M - 1 ? r : M - 1;`) — ml8-4's decode A is
//     always allocated to at least M_pad>=DTM*16 rows (ml8.cu's a_fp8_ptr
//     is sized M_pad*K, and rdna4_gemm_ml84_decode_v2 refuses M_pad <
//     DTM*16), so no row of the A tile this kernel ever reads is
//     out-of-bounds; only the OUTPUT bound (row < real M) is a predicate
//     (in the epilogue, matching bench/trfeed_ml84_kernels.h's own
//     `if (gr < M && gc < N)`), never the input fill.
//   * __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12 as the MMA —
//     the SAME intrinsic bench/trfeed_ml84_kernels.h already uses, not a
//     new one.
//
// ml8-4 NUMERICS — kept BIT-FOR-BIT identical to the existing production
// decode kernel (bench/trfeed_ml84_kernels.h's gemm_ml84_trfeed_splitk /
// ml84_trfeed_group_body / ml84_trfeed_epilogue), reused via #include
// rather than re-derived, per the task brief ("keep that — it is the
// format's codebook"):
//   * LUT codebook: ml84_lut_gather4 / ml84_expand_frag
//     (trfeed_ml84_kernels.h:74-90), called UNCHANGED below. The file's own
//     comment there flags the __builtin_amdgcn_perm selector-argument-order
//     convention as unverified against hardware/ISA docs in this sandbox;
//     that uncertainty is inherited here unchanged (same call, same
//     concern, not re-litigated).
//   * B_nib layout / tile addressing: ml84_tile_nib_base(kt,nt,NT)
//     (bench/ml84_trfeed_layout.h:66-68), UNCHANGED. See "LAYOUT REPACK"
//     below for why this already gives contiguous per-wave reads.
//   * Per-(64-K-group, N-column) scale fold: bench/trfeed_ml84_kernels.h
//     :250-257 (`acc[mi][ni] = acc[mi][ni] + scale * acc_g[mi][ni]`, scale
//     = `b_scale_g[g*N+gc]`) — this is NOT a per-row or per-column-only
//     scale (ml8-4 has no single per-column scale the way the frozen fp8
//     kernel does), it is per-(group,column), so it MUST be folded once
//     per 64-wide K-group, before the group's contribution is discarded.
//     Reproduced identically below in ml84_decode_v2_kernel's per-group
//     loop.
//   * Per-row activation scale: bench/trfeed_ml84_kernels.h:182
//     (`a_scale[gr] * ws[t]`) applied in the epilogue, AFTER the K
//     reduction (and, for v2's DKS>1 case, after the split-K reduce) —
//     reproduced identically below.
//   * A operand: fp8 e4m3 [M_pad, K] row-major, stride K, exactly
//     bench/trfeed_ml84_kernels.h's `Av[(gr*K+gk)/16]` addressing
//     (gemm_capi.h:391-398's A contract) — reproduced identically in
//     stage_a below.
//
// LAYOUT REPACK: NOT NEEDED. B_nib packs one 16(K)x16(N) tile as 32 lanes
// x 4 bytes = 128 CONTIGUOUS bytes (ml84_trfeed_layout.h:66-68,85: nib_dword_
// addr(kt,nt,NT,L) = tile_base + L*4, tile_base = b_tile_offset(kt,nt,NT)/2,
// and b_tile_offset's own 256-bytes-per-tile stride is exactly 2x that). A
// wave is exactly 32 lanes (WAVE_SIZE, trfeed_kernels.h:40), i.e. exactly
// one (kt,nt) tile's worth of lanes — so a wave's B read for one (kt,nt)
// tile IS already the reference design's "128 contiguous bytes per wave's
// 16x16 fragment" (see radiance_mxfp4.py:109-125's permute_w, which exists
// there ONLY because MXFP4's checkpoint layout is row-major-by-N and needs
// that one-time permute to get the same property). ml8-4's ML84_TRFEED
// layout was already built by the Phase-1 task specifically so a PLAIN
// (non-transposing) load could feed the WMMA fragment directly
// (ml84_trfeed_layout.h:1-23) — so it already has the property radiance's
// WPERM repack exists to create. No `ml84_repack_fragment_order` needed;
// this kernel reads straight from B_nib, same as the existing decode
// kernel, just with __builtin_nontemporal_load added (see above for why
// the originally-planned readfirstlane addition was reverted).
//
// HARDWARE CONVERTERS (__builtin_amdgcn_cvt_pk_f32_fp8 /
// __builtin_amdgcn_cvt_pk_fp8_f32): NOT USED — there is no fp8<->f32
// conversion anywhere in this kernel's staging or epilogue to accelerate.
// A and B both stay fp8 all the way into the WMMA operands (A is loaded as
// raw fp8 bytes via int4 vector loads with no numeric conversion; B's fp8
// bytes come out of the LUT gather, which is a byte permute, not a
// numeric fp8<->f32 convert); the epilogue only ever multiplies existing
// fp32 accumulator/scale values. This matches the existing production
// kernel (bench/trfeed_ml84_kernels.h), which likewise does no software
// fp8 conversion anywhere in its hot path.
#include <hip/hip_runtime.h>
#include "gemm_ml84_decode_v2.h"
#include "bench/trfeed_ml84_kernels.h"   // ml84_lut_gather4/ml84_expand_frag, float8_t/v2i32/v8f32/WAVE_SIZE (trfeed_kernels.h), ml84_tile_nib_base (ml84_trfeed_layout.h), QK_ML8 (ggml-common.h)
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <utility>
#include <algorithm>

// Waves per block / N-strip width per wave. Fixed (not a template param —
// nothing here needs a second instantiation at a different wave count);
// distinct name from trfeed_kernels.h's own WAVES_M/WAVES_N/BN globals
// (pulled in transitively above) so there is no redefinition.
static constexpr int V2_DWN  = 8;                 // waves/block
static constexpr int V2_BN   = V2_DWN * 16;        // 128: N covered per block
static constexpr int V2_PAD  = 8;                  // LDS stride pad (bank-conflict avoidance, matches radiance's DEC_PAD)
static constexpr int V2_MSTRIDE = 32;              // fixed partial-buffer row stride (== the scratch's compiled row ceiling)
static constexpr int V2_MAX_N = 248320;            // Qwen3.8-27B output head — the largest production N
static constexpr int V2_MAX_KS = 4;

#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)

// One 64-wide K-group's worth of WMMA work (BK_GROUP=64 -> 4 K16-steps),
// folding straight into `acc` scaled by b_scale_g[g,col] at the end —
// bit-for-bit the same fold as bench/trfeed_ml84_kernels.h:241-257, just
// restructured for this file's DTM/DWN tile instead of TFRAGS_M x TFRAGS_N.
template <int DTM, bool NT>
__device__ __forceinline__ void ml84_v2_group(
    const uint8_t* __restrict__ Asb, const uint8_t* __restrict__ B_nib,
    const uint8_t* __restrict__ lut, const float* __restrict__ b_scale_g,
    v8f32 (&acc)[DTM], int astride, int k_local0, int k_abs0, int g,
    int n0, int wave, int lane, int NT_tiles) {
    // Block-uniform per-group LUT (16 bytes -> 4 uint32). `g` is loop-
    // invariant across every lane and wave in the block, so every thread
    // computes the bit-identical address here already -- no manual
    // broadcast needed for CORRECTNESS. (An earlier version of this
    // function tried to force that uniformity through the compiler with a
    // manual 64-bit __builtin_amdgcn_readfirstlane hi/lo split; that had a
    // real bug -- readfirstlane takes/returns a SIGNED int on this
    // toolchain, so the low-half reconstruction sign-extended whenever the
    // pointer's low 32 bits had their top bit set, permanently corrupting
    // the high 32 bits when OR'd back together. That is what produced the
    // 0xffff...-pattern memory fault the coordinator reported. Removed --
    // plain pointer arithmetic is correct and the AMDGPU backend's own
    // uniformity analysis is trusted to promote this to an SGPR without
    // manual help.)
    const uint8_t* lut_g = lut + (size_t) g * 16;
    const uint32_t L0 = *reinterpret_cast<const uint32_t*>(lut_g + 0);
    const uint32_t L1 = *reinterpret_cast<const uint32_t*>(lut_g + 4);
    const uint32_t L2 = *reinterpret_cast<const uint32_t*>(lut_g + 8);
    const uint32_t L3 = *reinterpret_cast<const uint32_t*>(lut_g + 12);

    v8f32 acc_g[DTM];
    #pragma unroll
    for (int mi = 0; mi < DTM; ++mi) acc_g[mi] = v8f32{0,0,0,0,0,0,0,0};

    const int row_a = lane & 0xF, colhi = (lane >> 4) & 1;
    const int nt = n0 / 16 + wave;   // this wave's fixed N-tile (uniform across the wave)

    #pragma unroll
    for (int step = 0; step < 4; ++step) {          // 64-wide group = 4 K16-steps
        const int kk = k_local0 + step * 16;
        const int kt = (k_abs0 + step * 16) / 16;

        v2i32 fa[DTM];
        #pragma unroll
        for (int mi = 0; mi < DTM; ++mi) {
            const uint8_t* p = &Asb[(mi * 16 + row_a) * astride + kk + colhi * 8];
            fa[mi] = *reinterpret_cast<const v2i32*>(p);
        }

        // Wave-uniform tile base (ml84_tile_nib_base is the SAME B_nib
        // tile addressing the existing production kernel uses -- see the
        // "LAYOUT REPACK" note above for why this is already 128
        // contiguous bytes/wave). `kt`/`nt` are identical for every lane
        // of this wave (kt is loop-invariant across lanes, nt depends only
        // on `wave`), so this address is already lane-uniform without a
        // manual broadcast -- see the LUT comment above for why the
        // former readfirstlane hi/lo reconstruction here was an actual bug
        // (signed-int sign-extension on the low-half cast), not just an
        // unverified optimization; it has been removed, not just
        // "trusted".
        const uint8_t* tile_base = B_nib + ml84_tile_nib_base(kt, nt, NT_tiles);
        const uint32_t* wp = reinterpret_cast<const uint32_t*>(tile_base + (size_t) lane * 4);
        const uint32_t w = NT ? __builtin_nontemporal_load(wp) : *wp;
        const v2i32 fb = ml84_expand_frag(w, L0, L1, L2, L3);

        #pragma unroll
        for (int mi = 0; mi < DTM; ++mi)
            acc_g[mi] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(fa[mi], fb, acc_g[mi]);
    }

    // Per-(group,column) scale fold -- bench/trfeed_ml84_kernels.h:250-257,
    // bit-for-bit: `scale = b_scale_g[g*N+gc]`, `acc += scale*acc_g`.
    const int e_col = lane & 0xF;
    const int gc = n0 + wave * 16 + e_col;
    const float scale = b_scale_g[(size_t) g * (size_t) (NT_tiles * 16) + gc];
    #pragma unroll
    for (int mi = 0; mi < DTM; ++mi) acc[mi] = acc[mi] + scale * acc_g[mi];
}

// ─────────────────────────────────────────────────────────────────────────
// The decode-v2 kernel. Grid: (N/V2_BN, 1, DKS). Block: V2_DWN*32 threads.
// DBK in {64,128} (K-slab width, in units of one or two 64-wide ml8-4
// groups); DKS in {1,2,4} (split-K); DTM in {1,2} (16-row M-tiles, covers
// real M<=32); NT toggles __builtin_nontemporal_load on the B_nib stream.
// ─────────────────────────────────────────────────────────────────────────
template <int DBK, int DKS, int DTM, bool NT>
__global__ __launch_bounds__(V2_DWN * 32) void ml84_decode_v2_kernel(
    const float8_t* __restrict__ A, const uint8_t* __restrict__ B_nib,
    const uint8_t* __restrict__ lut, const float* __restrict__ a_scale,
    const float* __restrict__ b_scale_g, float* __restrict__ P, int* __restrict__ cnt,
    float* __restrict__ C_f32, int M, int N, int K) {
    constexpr int ASTRIDE = DBK + V2_PAD;
    constexpr int GROUPS_PER_SLAB = DBK / QK_ML8;
    // NOTE (flagged, same spirit as trfeed_ml84_kernels.h's unverified
    // perm-selector-order comment): ASTRIDE = DBK+V2_PAD is not itself a
    // multiple of 16, so row r>0's `int4` fill write below lands at an
    // address that is 16-byte aligned only for r such that r*ASTRIDE%16==0.
    // This mirrors radiance_mxfp4_fp8.hip's own stage_a (DASTR=DBK+DEC_PAD,
    // same non-16 stride) exactly -- copied structurally rather than
    // re-derived, and not independently re-verified against the AMDGPU LDS
    // ISA in this sandbox (ds_write_b128 is documented as needing only
    // dword, not 16-byte, address alignment, which every offset here does
    // satisfy: ASTRIDE and c are both multiples of 4). alignas(16) below
    // only fixes row 0's base; if a future measurement shows a fault or a
    // silent correctness gap on gfx1201 specifically, drop V2_PAD to 0 (or
    // round ASTRIDE up to a multiple of 16) as the fix -- isolated to this
    // one constant, not spread through the kernel.
    alignas(16) __shared__ uint8_t sA[DTM * 16 * ASTRIDE];
    __shared__ int s_last;

    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    constexpr int TBLOCK = V2_DWN * 32;
    const int n0 = blockIdx.x * V2_BN;
    const int ks = blockIdx.z;
    const int NT_tiles = N / 16;
    const uint8_t* Ab = reinterpret_cast<const uint8_t*>(A);

    const int slabs = K / DBK;
    const int spb = (slabs + DKS - 1) / DKS;
    const int s_lo = ks * spb, s_hi = min(slabs, s_lo + spb);

    v8f32 acc[DTM];
    #pragma unroll
    for (int mi = 0; mi < DTM; ++mi) acc[mi] = v8f32{0,0,0,0,0,0,0,0};

    for (int s = s_lo; s < s_hi; ++s) {
        const int k0 = s * DBK;
        // ---- A fill: BYTE-FOR-BYTE the same fp8 bytes / stride-K
        // addressing as bench/trfeed_ml84_kernels.h's A fill, just
        // vectorized as one 16-byte load/thread into a padded LDS row
        // (radiance_mxfp4_fp8.hip's stage_a shape, :822-831). Rows
        // 0..DTM*16-1 are always < M_pad (launcher-enforced), so this is
        // never out of bounds even when real M < DTM*16.
        constexpr int ATOTAL = DTM * 16 * DBK;
        {
            const int idx = tid * 16;
            if (idx < ATOTAL) {
                const int r = idx / DBK, c = idx % DBK;
                *reinterpret_cast<int4*>(&sA[r * ASTRIDE + c]) =
                    *reinterpret_cast<const int4*>(Ab + (size_t) r * K + k0 + c);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int g_local = 0; g_local < GROUPS_PER_SLAB; ++g_local) {
            const int k_local0 = g_local * QK_ML8;
            const int k_abs0 = k0 + k_local0;
            const int g = k_abs0 / QK_ML8;
            ml84_v2_group<DTM, NT>(sA, B_nib, lut, b_scale_g, acc, ASTRIDE,
                                    k_local0, k_abs0, g, n0, wave, lane, NT_tiles);
        }
        __syncthreads();   // LDS reuse barrier, same placement as the frozen kernel
    }

    // ---- epilogue ----
    const int e_col = lane & 0xF, e_rowbase = ((lane >> 4) & 1) * 8;
    const int n = n0 + wave * 16 + e_col;

    if constexpr (DKS == 1) {
        // No scratch, no atomics -- writes C_f32 directly (radiance's
        // DIRECT/DKS==1 fast path).
        if (n < N) {
            #pragma unroll
            for (int mi = 0; mi < DTM; ++mi) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int m = mi * 16 + e_rowbase + e;
                    if (m < M) C_f32[(size_t) m * N + n] = a_scale[m] * acc[mi][e];
                }
            }
        }
        return;
    }

    // DKS>1: stash the raw (unscaled-by-a_scale... no, a_scale IS applied
    // here too, same as the DKS==1 path -- it is a per-ROW factor, so it
    // commutes with the split-K sum: sum_ks(a_scale[m]*partial_ks) ==
    // a_scale[m]*sum_ks(partial_ks). Applying it before the reduce, same
    // as the direct path, keeps ONE epilogue formula instead of two and
    // costs nothing (a_scale[m] is loaded once either way).
    if (n < N) {
        #pragma unroll
        for (int mi = 0; mi < DTM; ++mi) {
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int m = mi * 16 + e_rowbase + e;
                if (m < M) P[((size_t) ks * V2_MSTRIDE + m) * N + n] = a_scale[m] * acc[mi][e];
            }
        }
    }

    // Fused reduction: the DKS blocks covering this n0 race on cnt[blockIdx.x];
    // whoever arrives last reduces P in place and resets the counter for the
    // next launch (radiance_mxfp4_fp8.hip:1050-1071) -- no memset, no second
    // kernel, no D2D copy-out.
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        s_last = (atomicAdd(&cnt[blockIdx.x], 1) == DKS - 1);
    }
    __syncthreads();
    if (!s_last) return;
    if (tid == 0) cnt[blockIdx.x] = 0;

    const int nhi = min(n0 + V2_BN, N);
    for (int nn = n0 + tid; nn < nhi; nn += TBLOCK) {
        for (int m = 0; m < M; ++m) {
            float sum = 0.f;
            #pragma unroll
            for (int k = 0; k < DKS; ++k) sum += P[((size_t) k * V2_MSTRIDE + m) * N + nn];
            C_f32[(size_t) m * N + nn] = sum;
        }
    }
}

#else
template <int DBK, int DKS, int DTM, bool NT>
__global__ __launch_bounds__(V2_DWN * 32) void ml84_decode_v2_kernel(
    const float8_t* __restrict__ A, const uint8_t* __restrict__ B_nib,
    const uint8_t* __restrict__ lut, const float* __restrict__ a_scale,
    const float* __restrict__ b_scale_g, float* __restrict__ P, int* __restrict__ cnt,
    float* __restrict__ C_f32, int M, int N, int K);
#endif

// ─────────────────────────────────────────────────────────────────────────
// Persistent grow-only per-(device,stream) scratch -- pattern copied from
// mt_pagedattn_aiter.cu's mt_aiter_persist_get (:895-955): keyed by
// (device,stream) so the two overlapping meta contexts / TP ranks never
// share a buffer; allocated on first EAGER (non-capturing) use and kept
// alive across calls (a captured graph's baked-in pointer stays valid);
// sized once, at first use, for the largest production shape (N=248320,
// V2_MSTRIDE=32 rows, V2_MAX_KS=4 splits) rather than grown call-by-call,
// so a shape that fits never triggers a second allocation.
// ─────────────────────────────────────────────────────────────────────────
namespace {
struct ml84_v2_scratch {
    float* partial = nullptr;
    int*   cnt     = nullptr;
    size_t partial_bytes = 0;
    int    cnt_n = 0;
};
std::mutex g_ml84_v2_mu;
std::map<std::pair<int, hipStream_t>, ml84_v2_scratch> g_ml84_v2_scratch;
}  // namespace

int ml84_decode_v2_max_n(void) { return V2_MAX_N; }

// Returns nullptr (never allocates) if called on a capturing stream with no
// existing buffer -- the caller must have made one prior eager call on this
// (device,stream) before it can be used inside a captured graph, same
// requirement mt_aiter_persist_get's callers already satisfy via ggml-
// cuda.cu's "eager warm-up visit" convention.
static ml84_v2_scratch* ml84_v2_scratch_get(int device, hipStream_t stream) {
    std::lock_guard<std::mutex> lock(g_ml84_v2_mu);
    const auto key = std::make_pair(device, stream);
    auto it = g_ml84_v2_scratch.find(key);
    if (it != g_ml84_v2_scratch.end()) {
        return &it->second;
    }
    hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
    hipStreamIsCapturing(stream, &cap_status);
    if (cap_status != hipStreamCaptureStatusNone) {
        return nullptr;   // no allocation inside capture -- see comment above
    }
    ml84_v2_scratch s;
    s.partial_bytes = (size_t) V2_MAX_KS * (size_t) V2_MSTRIDE * (size_t) V2_MAX_N * sizeof(float);
    s.cnt_n = (V2_MAX_N + V2_BN - 1) / V2_BN;
    int prev_dev = 0;
    hipGetDevice(&prev_dev);
    hipSetDevice(device);
    const hipError_t e1 = hipMalloc(&s.partial, s.partial_bytes);
    const hipError_t e2 = hipMalloc(&s.cnt, (size_t) s.cnt_n * sizeof(int));
    if (e2 == hipSuccess) {
        hipMemsetAsync(s.cnt, 0, (size_t) s.cnt_n * sizeof(int), stream);
    }
    hipSetDevice(prev_dev);
    if (e1 != hipSuccess || e2 != hipSuccess) {
        if (s.partial) hipFree(s.partial);
        if (s.cnt) hipFree(s.cnt);
        return nullptr;
    }
    auto res = g_ml84_v2_scratch.emplace(key, s);
    return &res.first->second;
}

// Split-K fill law -- radiance_mxfp4_fp8.hip:1099-1112, verbatim constants
// (fill=110 at M<=24 else 78; narrowest ks in {1,2,4} clearing nblk*ks>=fill).
static int ml84_decode_v2_split_k_for(int nblk, int M) {
    const int fill = (M <= 24) ? 110 : 78;
    for (int ks = 1; ks < V2_MAX_KS; ks <<= 1) {
        if (nblk * ks >= fill) return ks;
    }
    return V2_MAX_KS;
}

extern "C" bool rdna4_gemm_ml84_decode_v2(int M, const void* A, const uint8_t* B_nib, const uint8_t* lut,
                                          float* C_f32, const float* a_scale, const float* b_scale_g,
                                          int M_pad, int N, int K, int n_splits, hipStream_t stream) {
    if (M <= 0 || M > 32) return false;
    if (M_pad <= 0 || M_pad < M) return false;
    if (N <= 0 || (N % 16) != 0 || N > V2_MAX_N) return false;
    if (K <= 0 || (K % QK_ML8) != 0) return false;

    // Measured 2026-09-19 (bench/gemm_ml84_decode_v2_bench, R9700 DPM auto):
    // M<=16 wins 1.2-4.4x over the old split-K on every production shape;
    // M=32 loses on the two big ones (5120x17408, 248320x5120). Decline
    // M>16 so the caller falls back to rdna4_gemm_ml84_trfeed_decode_splitk.
    if (M > 16) return false;
    const int dtm = 1;
    if (M_pad < dtm * 16) return false;   // A must cover every row the LDS fill reads
    const int dbk = (K % 128 == 0) ? 128 : 64;
    const int nblk = (N + V2_BN - 1) / V2_BN;

    static const int env_ks = [] {
        const char* e = std::getenv("MT_ML8_4_DECODE_V2_KS");
        return e ? std::atoi(e) : 0;
    }();
    // Non-temporal weight loads measured SLOWER on RDNA4 (M=1 5120x5120:
    // 63 us NT vs 30 us plain; every shape agrees) -- default off,
    // MT_ML8_4_DECODE_V2_NT=1 opts in.
    static const bool env_nt_off = [] {
        const char* e = std::getenv("MT_ML8_4_DECODE_V2_NT");
        return !(e && std::atoi(e) != 0);
    }();
    int ks = env_ks > 0 ? env_ks : (n_splits > 0 ? n_splits : ml84_decode_v2_split_k_for(nblk, M));
    ks = (ks >= 4) ? 4 : (ks >= 2) ? 2 : 1;
    const bool use_nt = !env_nt_off;

    float* P = nullptr;
    int* cnt = nullptr;
    if (ks > 1) {
        int device = 0;
        hipGetDevice(&device);
        ml84_v2_scratch* scr = ml84_v2_scratch_get(device, stream);
        if (scr == nullptr) return false;   // capturing with no warm scratch yet, or OOM -- caller falls back
        P = scr->partial;
        cnt = scr->cnt;
    }

    const dim3 grid(nblk, 1, ks);
    const dim3 block(V2_DWN * 32);

#define V2_LAUNCH(DBK_, KS_, DTM_, NT_)                                                       \
    ml84_decode_v2_kernel<DBK_, KS_, DTM_, NT_><<<grid, block, 0, stream>>>(                   \
        (const float8_t*) A, B_nib, lut, a_scale, b_scale_g, P, cnt, C_f32, M, N, K)
#define V2_BY_DTM(DBK_, KS_, NT_)                                                             \
    do { if (dtm == 1) { V2_LAUNCH(DBK_, KS_, 1, NT_); } else { V2_LAUNCH(DBK_, KS_, 2, NT_); } } while (0)
#define V2_BY_KS(DBK_, NT_)                                                                   \
    do {                                                                                       \
        if (ks == 1)      { V2_BY_DTM(DBK_, 1, NT_); }                                         \
        else if (ks == 2) { V2_BY_DTM(DBK_, 2, NT_); }                                         \
        else              { V2_BY_DTM(DBK_, 4, NT_); }                                         \
    } while (0)
#define V2_BY_DBK(NT_)                                                                        \
    do { if (dbk == 128) { V2_BY_KS(128, NT_); } else { V2_BY_KS(64, NT_); } } while (0)

    if (use_nt) { V2_BY_DBK(true); } else { V2_BY_DBK(false); }

#undef V2_BY_DBK
#undef V2_BY_KS
#undef V2_BY_DTM
#undef V2_LAUNCH

    return true;
}
