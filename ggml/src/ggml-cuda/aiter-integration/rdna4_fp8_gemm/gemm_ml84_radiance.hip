// gemm_ml84_radiance.hip — PORT of radiance's prefill GEMM kernel onto
// ml8-4 numerics. See gemm_ml84_radiance.h for the design writeup.
//
// PORTED VERBATIM from ~/GitHub/ggz14-vllm-mxfp4/radiance_mxfp4_fp8.hip
// (the copy that produced the 3512 tok/s measurement; diffed against this
// repo's own /home/kmbandy/GitHub/vllm-radiance/radiance_mxfp4_fp8.hip and
// found identical in every kernel BODY -- the two files differ only in
// launcher-level paroquant/merged-linear plumbing added on top, none of
// which touches a kernel body):
//
//   radiance_mxfp4_fp8.hip:217-224  tile-shape #defines (TM,WM,WN,NWAVE,
//                                   NTHREADS,BMF,BNF_OF,BNF) -- copied
//                                   verbatim, renamed RADP_* to avoid
//                                   colliding with any other TU's macros
//                                   of the same generic names when linked
//                                   together (this file is its own TU;
//                                   macros do not cross TUs, but the
//                                   rename documents the port 1:1).
//   radiance_mxfp4_fp8.hip:524-528  radiance_lds_barrier() -- copied
//                                   verbatim (renamed radp_lds_barrier),
//                                   same 3 intrinsics, same ordering.
//   radiance_mxfp4_fp8.hip:530-758  radiance_mxfp4_fp8_gemm_atiled<TN,WPERM,LBK>
//                                   -- the WPERM=true instantiation only
//                                   (see below for why), copied verbatim
//                                   line-for-line EXCEPT:
//                                     (a) the W-staging pointer arithmetic
//                                         (Wp/pk address computation) is
//                                         re-derived for this repo's
//                                         ML84_TRFEED B_nib tile layout
//                                         instead of radiance's own
//                                         fragment-order MXFP4 weight
//                                         buffer -- "B-source plumbing"
//                                         per the task directive. Both
//                                         layouts use the IDENTICAL
//                                         per-(16x16 K,N)-tile,
//                                         one-uint32-per-lane packing
//                                         (low nibbles = elements 0-3,
//                                         high nibbles = elements 4-7 --
//                                         see ml84_trfeed_layout.h's B_nib
//                                         contract vs radiance's WPERM
//                                         comment at line 234-237 of the
//                                         source), so the pk[]/wq
//                                         extraction and the FINAL
//                                         interleave line
//                                         (`sW[...] = {perm(bo,be,...),
//                                         perm(bo,be,...)}`) are BYTE-FOR-
//                                         BYTE unchanged; only the tile's
//                                         base-address formula differs
//                                         (kt-major/nt-minor here vs
//                                         nt-major/kt-minor there, per
//                                         each buffer's own layout).
//                                     (b) the staging conversion itself:
//                                         radiance's kMag[d] 3-bit
//                                         magnitude lookup + explicit sign
//                                         OR (lines 665-677, the E2M1 ->
//                                         e4m3 conversion) is replaced by
//                                         a 16-way LUT gather against the
//                                         T[g][n] table this file's prep
//                                         kernel builds, using the SAME
//                                         two-v_perm-plus-blend technique
//                                         bench/trfeed_ml84_kernels.h's
//                                         ml84_lut_gather4 uses (that
//                                         function's own header comment
//                                         explains the perm-selector
//                                         convention this reuses). The
//                                         16-entry table is now built per
//                                         K-slab in LDS from the weight's
//                                         own lut and b_scale_g (inlined
//                                         in the kernel) instead of read
//                                         from a precomputed T. The
//                                         sMag/kMag shared-memory constant
//                                         table and the per-row wref/wsv
//                                         reference-byte arrays (radiance's
//                                         MX-block-exponent machinery, not
//                                         applicable to ml8-4's arbitrary
//                                         fp32 group scale) are removed;
//                                         nothing else in the K-loop moves.
//                                     (c) the epilogue's per-column scale
//                                         factor `rf[j] = 2^Wref[n]`
//                                         (radiance's E8M0 exponent
//                                         reconstruction, line 727) becomes
//                                         `rf[j] = colscale[n]` (a direct
//                                         fp32 read) -- SAME variable name,
//                                         SAME place in the epilogue,
//                                         SAME multiply against the WMMA
//                                         accumulator and the per-token
//                                         a_scale two lines later. This is
//                                         the ONLY epilogue change, exactly
//                                         as the task directive specifies.
//                                     (d) the pb1/pb2/astride merged-linear
//                                         partition-select block (source
//                                         lines 548-555) is kept VERBATIM
//                                         -- this port's launcher always
//                                         passes the disabling values
//                                         (1<<30, 1<<30, 0), so it is dead
//                                         computation, exactly as it is on
//                                         radiance's own non-partitioned
//                                         `launch_at` entry point.
//                                   The WPERM=false ("checkpoint [N,K/2]
//                                   row-major") branch is NOT ported: it
//                                   addresses radiance's own on-disk
//                                   weight layout, which does not exist
//                                   for ml8-4 (our only packed weight
//                                   layout is the ML84_TRFEED tile format,
//                                   which is what WPERM=true already
//                                   assumes). This matches what is
//                                   actually dispatched in production:
//                                   ggz14's serve-mxfp4.sh sets
//                                   RADIANCE_MXFP4_WPERM=1 by default
//                                   (forced to 0 only under TP padding,
//                                   which does not apply to this port's
//                                   target shapes) -- see the task report
//                                   for the grep trail.
//   radiance_mxfp4_fp8.hip:1495    radiance_silu_mul_quant's TILED store
//                                   index, `(mt*kst+(g>>1))*32+lrow+16*(g&1)`
//                                   -- the arithmetic ported (not the
//                                   silu/quant compute around it, which
//                                   this repo has no use for) into
//                                   radp_retile_a_kernel below as the
//                                   inverse of the atiled kernel's own A
//                                   read (abase[i] + aoff + k0*16 + st*256,
//                                   lines 566-572/628-634), confirmed to
//                                   agree by construction (both index the
//                                   same `AT + (mt*ksteps_g+kt)*256 +
//                                   lane*8 + byte` cell).
//
// Nothing here is a new kernel design: every tile shape, LDS stride, wave
// assignment, double-buffering absence (radiance's atiled kernel has none
// -- A comes straight from global per the header comment at
// radiance_mxfp4_fp8.hip:628), sched_barrier/lds-barrier placement and
// fragment load is the ORIGINAL's, unchanged.
#include <hip/hip_runtime.h>
#include "gemm_ml84_radiance.h"
// bench/trfeed_ml84_kernels.h pulls in ml84_trfeed_layout.h (ml84_decode_e4m3/
// encode_e4m3, QK_ML8, block_ml8_4, ml84_tile_nib_base, b_tile_offset) AND
// gemm_ml84_decode_v2.hip's OWN validated ml84_lut_gather4 (trfeed_ml84_kernels.h:74-79)
// -- included directly, not hand-copied, per the fix requested after the
// bench reported wrong output: use the EXACT hardware-proven gather, not a
// hand-retyped copy of it, to remove any doubt about textual drift.
#include "bench/trfeed_ml84_kernels.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <vector>

typedef float floatx8_radp __attribute__((ext_vector_type(8)));
typedef int int2_radp __attribute__((ext_vector_type(2)));
typedef unsigned int uint4_radp __attribute__((ext_vector_type(4)));
typedef unsigned int uint2_radp __attribute__((ext_vector_type(2)));

// ── radiance_mxfp4_fp8.hip:217-224, verbatim, renamed RADP_* ──────────────
#define RADP_TM 4
#define RADP_WM 4
#define RADP_WN 2
#define RADP_NWAVE (RADP_WM * RADP_WN)
#define RADP_NTHREADS (RADP_NWAVE * 32)
#define RADP_BMF (RADP_WM * RADP_TM * 16)
#define RADP_BNF_OF(tn) (RADP_WN * (tn) * 16)
#define RADP_PAD 8

// ── radiance_mxfp4_fp8.hip:524-528, verbatim ──────────────────────────────
static __device__ __forceinline__ void radp_lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

#if defined(__gfx1201__) || !defined(__HIP_DEVICE_COMPILE__)

// staging-step replacement: 16-way LUT gather. Uses trfeed_ml84_kernels.h's
// OWN ml84_lut_gather4 (lines 74-79) directly -- NOT a hand copy -- since
// that function is the one validated correct on hardware by
// gemm_ml84_decode_v2.hip (v2_orcl ~1e-5..2e-3 across every shape).

// ── radiance_mxfp4_fp8.hip:530-758, ported. Template drops WPERM (only the
// true instantiation is ported, see header comment above) and keeps LBK so
// the TN=2/TN=4 tile-size dependent K-slab width (128 vs 64) is preserved
// verbatim from the original's RAD_AT macro (radiance_mxfp4_fp8.hip:1680:
// `(TN_ == 2 ? 128 : 64)`). ──────────────────────────────────────────────
template <int TN, int LBK>
__global__ __launch_bounds__(RADP_NTHREADS) void radp_gemm_ml84_atiled(
    const unsigned char *__restrict__ AT, const unsigned char *__restrict__ B_nib,
    const unsigned char *__restrict__ lut, const float *__restrict__ b_scale_g,
    const float *__restrict__ colscale,
    const float *__restrict__ As, float *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + RADP_PAD;
  constexpr int BNF_T = RADP_WN * TN * 16;
  __shared__ unsigned char sW[BNF_T * LWSTR];
  // sT is double-buffered: the K loop builds slab k0+LBK's table (sT[buf^1])
  // during slab k0's WMMA phase, so the loop's existing trailing barrier
  // publishes it for the next iteration -- no extra barrier needed. The build
  // is split in two: prefetch_sT issues the global loads (b_scale_g, the 8-byte
  // lut dword pair, the colscale inverse) early in the iteration so the
  // latency hides behind staging + WMMA; convert_sT does the VALU decode/encode
  // and the LDS store afterwards (pf_bsg/pf_lw are written at the top and
  // consumed at the bottom of the same iteration).
  __shared__ unsigned char sT[2][(LBK / QK_ML8) * BNF_T * 16];

  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int wm = wave / RADP_WN, wn = wave % RADP_WN;
  const int col = lane & 15, kb8 = (lane >> 4) * 8;
  const int m0 = blockIdx.y * RADP_BMF, n0 = blockIdx.x * BNF_T;
  const int NT_tiles = N / 16;   // B_nib's own tile grid width (ml84_tile_nib_base's NT)
  // radiance_mxfp4_fp8.hip:548-555, verbatim (dead unless the caller opts
  // into paroquant partitioning; this port's launcher always disables it
  // via pb1=pb2=1<<30, astride=0, matching radiance's own `launch_at`).
  {
    const int p_ = n0 >= pb2 ? 2 : (n0 >= pb1 ? 1 : 0);
    AT += (size_t)p_ * astride;
    As += (size_t)p_ * M;
  }
  const int ksteps_g = K / 16;
  const int Mt = (M + 15) >> 4;

  // A tile bases, one per M-fragment: radiance_mxfp4_fp8.hip:562-572, verbatim.
  const unsigned char *abase[RADP_TM];
#pragma unroll
  for (int i = 0; i < RADP_TM; ++i) {
    int mt = (m0 >> 4) + wm * RADP_TM + i; mt = mt < Mt - 1 ? mt : Mt - 1;
    abase[i] = AT + (size_t)__builtin_amdgcn_readfirstlane(mt * ksteps_g * 256);
  }
  const unsigned int aoff = lane * 8;

  // W staging geometry, hoisted out of the K loop: radiance_mxfp4_fp8.hip:
  // 574-609's WPERM branch, verbatim (this port has no WPERM=false path).
  constexpr int KSTEPS_T = LBK / 16, NTILES_T = BNF_T / 16;
  constexpr int TOT_SLOTS = NTILES_T * KSTEPS_T * 32;
  constexpr int WSLOTS = (TOT_SLOTS >= RADP_NTHREADS * 4) ? 4 : 2;
  constexpr int NW = TOT_SLOTS / (RADP_NTHREADS * WSLOTS);
  static_assert(NW * RADP_NTHREADS * WSLOTS == TOT_SLOTS, "WPERM slot groups must tile");
  int wrow[NW], wcol[NW], wrc[NW], wkst[NW], wlanec[NW];
#pragma unroll
  for (int it = 0; it < NW; ++it) {
    const int sl = it * RADP_NTHREADS * WSLOTS + tid * WSLOTS;
    const int lane_ = sl & 31, rest = sl >> 5;
    const int kst = rest % KSTEPS_T, ntl = rest / KSTEPS_T;
    wrow[it] = ntl * 16 + (lane_ & 15);
    wcol[it] = kst * 16 + (lane_ >> 4) * 8;          // kloc
    const int gn = n0 + wrow[it];
    wrc[it] = gn < N - WSLOTS ? gn : N - WSLOTS;      // global clamped column
    wlanec[it] = (lane_ & 16) | (wrc[it] & 15);       // == B_nib's tile lane L
    wkst[it] = kst;
  }

  floatx8_radp acc[RADP_TM][TN];
#pragma unroll
  for (int i = 0; i < RADP_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][j][e] = 0.f;

  // Per-slab 16-byte codebook table in LDS, built from the ml8-4 weight's
  // OWN lut (shared across all N) + per-(group,column) b_scale_g. Same
  // bit-identical formula as the former radp_table_kernel (now inlined).
  // Double-buffered + split: the loop PREFETCHES slab k0+LBK's b_scale_g
  // and lut bytes (global loads issued at the TOP of the iteration, so the
  // latency hides behind staging + WMMA), then CONVERTs them (VALU
  // decode/encode + LDS store) at the BOTTOM. The loop's existing trailing
  // barrier publishes sT[buf^1] for the next iteration -- no extra barrier.
  constexpr int GS = LBK / QK_ML8;
  constexpr int HALF_TABLES = GS * BNF_T * 2;   // 8-entry halves of the 16-byte tables
  static_assert(HALF_TABLES == RADP_NTHREADS, "one half-table per thread");
  const int h = tid;
  // Per-thread half-table geometry, hoisted out of the K loop (k0-independent).
  const int c0  = (h & 1) * 8;                  // entries c0..c0+7
  const int nl  = (h >> 1) % BNF_T;
  const int gl  = (h >> 1) / BNF_T;
  int n = n0 + nl; n = n < N ? n : N - 1;
  const float inv = colscale[N + n];            // colscale[N..2N) = 1/denom (prep)
  // Decoded-centroid cache: the 16 fp8 codebook entries for a K-group decode
  // to the SAME f32 values for every column (the codebook `lut` is indexed
  // by K-group only, per the header comment -- "one 16-byte codebook per
  // K-group of 64", NOT per column). The old per-thread code had all
  // BNF_T*2 threads sharing a (group,half) independently re-run
  // __builtin_amdgcn_cvt_f32_fp8 on the SAME 8 bytes -- BNF_T-way redundant
  // decode work AND BNF_T-way redundant `lut` global loads. Table sized by
  // GS (K-slab-local group count, 1 or 2) alone, not by N or K*N, so this
  // stays tiny (<=128 B) regardless of the GEMM shape -- see goal note "a
  // table sized by N or by K alone is fine". Single-buffered (not double
  // like sT[2][..]): unlike sT, which is read by the sW-staging step in the
  // FOLLOWING iteration and so must survive across the iteration boundary,
  // sCent is written near the top of an iteration and consumed by that same
  // iteration's convert_sT() call below the loop's existing mid-loop
  // radp_lds_barrier() -- no cross-iteration lifetime, so no second buffer
  // is needed (verified: the next iteration's decode_sT() write cannot
  // execute until this iteration's trailing barrier has passed, by which
  // point this iteration's convert_sT() read has already happened).
  __shared__ float sCent[GS][16];
  // Per-slab global loads, issued early so the latency hides behind staging + WMMA.
  float      pf_bsg;      // prefetched b_scale_g[g][n] for the slab being prefetched
  uint2_radp pf_lw;       // prefetched 8 codebook bytes lut[g*16 + c0 .. +7] (nl==0 threads only)
  auto prefetch_sT = [&](int kslab) {
    const int g = kslab / QK_ML8 + gl;
    pf_bsg = b_scale_g[(size_t)g * (size_t)N + n];
    // Only ONE (gl,c0) pair per group/half needs the raw codebook bytes --
    // the thread at nl==0 decodes for the whole column-tile (see decode_sT).
    // Skipping this global load on the other (BNF_T-1) threads per (gl,c0)
    // cuts the redundant `lut` VMEM traffic the old code issued on every
    // one of the RADP_NTHREADS threads.
    if (nl == 0) pf_lw = *(const uint2_radp *)(lut + (size_t)g * 16 + c0);
  };
  // Decode this group/half's 8 codebook bytes to f32 ONCE (nl==0 threads
  // only) and publish into sCent for every column's thread to read back.
  // Same hardware fp8->f32 conversion as before (v_cvt_f32_fp8, byte select
  // 0..3), just executed by 2*GS threads instead of RADP_NTHREADS -- the
  // decoded VALUES are identical (decode depends only on the group's lut
  // bytes, never on the column), so this changes nothing about convert_sT's
  // output, only how many threads redundantly compute it.
  auto decode_sT = [&]() {
    if (nl != 0) return;
    const float f0 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[0], 0), f1 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[0], 1);
    const float f2 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[0], 2), f3 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[0], 3);
    const float f4 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[1], 0), f5 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[1], 1);
    const float f6 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[1], 2), f7 = __builtin_amdgcn_cvt_f32_fp8(pf_lw[1], 3);
    sCent[gl][c0 + 0] = f0; sCent[gl][c0 + 1] = f1; sCent[gl][c0 + 2] = f2; sCent[gl][c0 + 3] = f3;
    sCent[gl][c0 + 4] = f4; sCent[gl][c0 + 5] = f5; sCent[gl][c0 + 6] = f6; sCent[gl][c0 + 7] = f7;
  };
  // convert_sT is split into two independent HALVES (convert_lo: this
  // thread's elements 0-3 -> o0; convert_hi: elements 4-7 -> o1) that used
  // to be one lambda issued as a single ~20-VALU-op block AFTER the whole
  // WMMA step loop had finished (see the call-site history below). Bench
  // data showed that placement cost 4-5% of kernel time even after the
  // decode-hoist above removed the redundant `lut` loads/decode -- the
  // convert work was a serial VALU block sitting between the WMMA loop and
  // the trailing barrier, not overlapped with anything. Splitting it lets
  // each half be issued from WITHIN the step loop, between two WMMA
  // steps' `__builtin_amdgcn_sched_barrier(0)` fences (which block reorder
  // only AT that fence, not between one step's WMMA and a later step's
  // fence), so the scheduler can interleave these VALU ops with the
  // matrix-core instruction stream instead of appending them after it.
  // Math and byte layout are UNCHANGED from the original single lambda --
  // same per-element clamp (see the rejected-optimization note kept on
  // convert_lo below), same 2-call cvt_pk_fp8_f32 pairing, same
  // destination bytes -- only WHEN each half's instructions are issued
  // changes.
  auto convert_lo = [&](unsigned char *dstT) {
    const float s = pf_bsg * inv;
    // Read this thread's already-decoded centroids back from sCent
    // (broadcast LDS reads: every thread's address depends only on (gl,c0),
    // so the BNF_T threads sharing a (group,half) all hit the SAME 2
    // addresses per wave -- no bank conflict, no re-decode).
    //
    // NOT hoisted: clamping s itself (s_c = min(s,448)) instead of each
    // product was evaluated per the goal note's proof obligation and
    // REJECTED -- it is not bit-identical. Centroids are bounded to
    // [-1,1] (scripts/calibration/centroid_quantizer.py fits Lloyd-Max
    // centroids on rows normalized by their own per-row abs-max, so
    // |x_norm|<=1 by construction; centroids lie in the convex hull of
    // that data and the subsequent E4M3 snap cannot push a value <=1.0
    // past the 1.0 rounding boundary up to 1.0625), so |f[e]*s| <= s
    // always. But when s > 448 (a real case: s = b_scale_g*inv can exceed
    // 448 for a group whose OWN lut_amax is well under 1 while a
    // DIFFERENT group sets colscale[n]'s max), per-entry clamping keeps
    // f[e]*s for f[e] with small |f[e]| (which never approached 448 in
    // the first place) UNCHANGED, whereas clamping s first would rescale
    // that same small product down to f[e]*448 -- a different, wrong
    // value. Example: f[e]=0.1, s=500 -> per-entry clamp keeps 50; a
    // clamped-s version would give 44.8. So the per-entry clamp is kept
    // (one v_med3_f32 per element, same as before).
    const float f0 = fminf(fmaxf(sCent[gl][c0 + 0] * s, -448.f), 448.f);
    const float f1 = fminf(fmaxf(sCent[gl][c0 + 1] * s, -448.f), 448.f);
    const float f2 = fminf(fmaxf(sCent[gl][c0 + 2] * s, -448.f), 448.f);
    const float f3 = fminf(fmaxf(sCent[gl][c0 + 3] * s, -448.f), 448.f);
    // scale, then encode pairs with the hardware f32->fp8 pack (v_cvt_pk_fp8_f32, RNE): (lo, hi, old, write_hi)
    // v_cvt_pk_fp8_f32 does NOT saturate (|x| > 448 -> NaN byte); the
    // software __hip_fp8_e4m3 the old table used clamps to +-448, hence
    // the per-element clamp above.
    uint32_t o0 = __builtin_amdgcn_cvt_pk_fp8_f32(f0, f1, 0u, false);
    o0          = __builtin_amdgcn_cvt_pk_fp8_f32(f2, f3, o0, true);
    *(uint32_t *)(&dstT[((gl * BNF_T) + nl) * 16 + c0 + 0]) = o0;
  };
  auto convert_hi = [&](unsigned char *dstT) {
    const float s = pf_bsg * inv;   // same s as convert_lo -- see its comment for why per-entry clamp is kept
    const float f4 = fminf(fmaxf(sCent[gl][c0 + 4] * s, -448.f), 448.f);
    const float f5 = fminf(fmaxf(sCent[gl][c0 + 5] * s, -448.f), 448.f);
    const float f6 = fminf(fmaxf(sCent[gl][c0 + 6] * s, -448.f), 448.f);
    const float f7 = fminf(fmaxf(sCent[gl][c0 + 7] * s, -448.f), 448.f);
    uint32_t o1 = __builtin_amdgcn_cvt_pk_fp8_f32(f4, f5, 0u, false);
    o1          = __builtin_amdgcn_cvt_pk_fp8_f32(f6, f7, o1, true);
    *(uint32_t *)(&dstT[((gl * BNF_T) + nl) * 16 + c0 + 4]) = o1;
  };
  // One-time prologue: prefetch + decode + convert slab 0's table before the
  // loop. This is the only place that needs an EXTRA barrier (between
  // decode_sT's publish and the converts' read) -- inside the K loop below,
  // decode_sT()'s publish for slab k0+LBK is placed before the loop's
  // existing mid-loop radp_lds_barrier() (the one that already gates the
  // sW build), so that barrier does double duty and the converts need no
  // additional synchronization. The prologue runs once per launch (no WMMA
  // to interleave with yet), so convert_lo/convert_hi just run back to
  // back here; this one extra barrier is not on the per-slab critical path.
  prefetch_sT(0);
  decode_sT();
  radp_lds_barrier();
  convert_lo(sT[0]);
  convert_hi(sT[0]);
  radp_lds_barrier();
  for (int k0 = 0, buf = 0; k0 < K; k0 += LBK, buf ^= 1) {
    // Prefetch the NEXT slab's global loads first so their latency hides
    // behind this slab's staging + WMMA (consumed by convert_sT at the
    // bottom). decode_sT() runs right after: it only needs pf_lw (nl==0
    // threads), and its sCent write is published by the mid-loop barrier
    // below -- no separate barrier needed for it.
    if (k0 + LBK < K) { prefetch_sT(k0 + LBK); decode_sT(); }
    // A fragments straight from global: radiance_mxfp4_fp8.hip:628-634, verbatim.
    int2_radp af[RADP_TM][NS];
#pragma unroll
    for (int i = 0; i < RADP_TM; ++i)
#pragma unroll
      for (int st = 0; st < NS; ++st)
        af[i][st] = *(const int2_radp *)(abase[i] + (aoff + (unsigned int)(k0 * 16) + (unsigned int)(st * 256)));

    // ---- W tile -> LDS: (a) B-source addressing re-derived for B_nib's
    // kt-major tile layout (ml84_trfeed_layout.h's b_tile_offset), in place
    // of radiance's nt-major fragment buffer -- see header comment (a).
    // (b) staging conversion replaced by the T-table LUT gather -- header
    // comment (b). (c) the FINAL INTERLEAVE PERMUTATION IS DROPPED --
    // see below, this is the fix for the "one quarter of each dot product
    // right" bug the orchestrator found on hardware.
    //
    // Radiance's own `perm(bo,be,0x05010400u)/perm(bo,be,0x07030602u)`
    // (radiance_mxfp4_fp8.hip:675-677) exists to UN-SCRAMBLE its
    // CHECKPOINT's own on-disk packing: there, 4 consecutive ROW-MAJOR
    // bytes b0..b0+3 hold LOW nibbles at k=2*b0,2*b0+2,2*b0+4,2*b0+6 (ev,
    // EVEN k's) and HIGH nibbles at k=2*b0+1,...,+7 (od, ODD k's) --
    // i.e. radiance's "ev"/"od" are the EVEN/ODD-strided halves of one
    // 8-wide k range, and the perm interleaves them byte-by-byte
    // (be0,bo0,be1,bo1,...) back into SEQUENTIAL k order for LDS.
    //
    // ml84-trfeed's B_nib is packed completely differently (see
    // ml84_trfeed_layout.h's B_nib contract / ml84_lane_slot): elements
    // s=0..3 are already the LOW nibbles of 4 CONSECUTIVE bytes in
    // SEQUENTIAL k order (k=half*8+0,+1,+2,+3, not even-strided), and
    // s=4..7 are the HIGH nibbles, ALSO sequential (k=half*8+4,+5,+6,+7).
    // So sel_lo/be is already [k+0,k+1,k+2,k+3] and sel_hi/bo is already
    // [k+4,k+5,k+6,k+7] -- applying radiance's even/odd-interleave perm to
    // ALREADY-sequential data scrambles it: verified by hand and by a
    // byte-level replica (be=[100..103], bo=[104..107] ->
    // perm output [100,104,101,105,102,106,103,107], i.e. only byte
    // positions 0 and 7 land correctly -- exactly the observed "2 of 8
    // terms right per WMMA step" / 0.2515 correlation the orchestrator's
    // error-structure analysis measured. The fix is a straight
    // concatenation instead of an interleave: {be, bo} written directly
    // gives sW bytes [k+0..k+7] in order, matching what the UNCHANGED
    // consuming WMMA read (wf[j][0]=sW[...0..3], wf[j][1]=sW[...4..7])
    // already assumes. Everything else (pk/wq extraction) unchanged from
    // radiance_mxfp4_fp8.hip:637-674. ----
    uint4_radp pk[NW];
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      const int nt = wrc[it] >> 4;
      const int kt = k0 / 16 + wkst[it];
      const size_t nib_base = ((size_t)kt * NT_tiles + nt) * 128;   // ml84_tile_nib_base(kt,nt,NT_tiles)
      const unsigned char *src = B_nib + nib_base + (size_t)wlanec[it] * 4;
      if constexpr (WSLOTS == 4) pk[it] = *(const uint4_radp *)src;
      else { const uint2_radp v = *(const uint2_radp *)src; pk[it] = uint4_radp{v[0], v[1], 0u, 0u}; }
    }
#pragma unroll
    for (int it = 0; it < NW; ++it) {
      // g is constant across this whole it (QK_ML8=64 spans exactly
      // LBK/64 slabs' worth of k-tiles; wkst[it] only ever moves the
      // k-tile within ONE k0 slab, so (k0+wkst*16)/64 is loop-invariant
      // in wkst whenever LBK==64, and correctly walks 2 groups per slab
      // when LBK==128 -- same "blk" pattern radiance's own
      // non-WPERM/WPERM-scale read used at line 646/652, just at group
      // width 64 instead of MX's 32).
      const int gl = (wkst[it] * 16) / QK_ML8;     // group within slab
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        const unsigned char *Tqn = &sT[buf][((gl * BNF_T) + (wrow[it] + q)) * 16];
        const uint32_t L0 = *(const uint32_t *)(Tqn + 0);
        const uint32_t L1 = *(const uint32_t *)(Tqn + 4);
        const uint32_t L2 = *(const uint32_t *)(Tqn + 8);
        const uint32_t L3 = *(const uint32_t *)(Tqn + 12);
        const unsigned int wq = pk[it][q];
        const unsigned int sel_lo = wq & 0x0F0F0F0Fu;         // elements s=0..3
        const unsigned int sel_hi = (wq >> 4) & 0x0F0F0F0Fu;  // elements s=4..7
        const unsigned int be = ml84_lut_gather4(sel_lo, L0, L1, L2, L3);
        const unsigned int bo = ml84_lut_gather4(sel_hi, L0, L1, L2, L3);
        // Straight concatenation (NOT radiance's even/odd interleave --
        // see the comment above): be already holds k+0..k+3 in its 4
        // bytes, bo already holds k+4..k+7, so writing {be, bo} places
        // sW[...0..7] = [k+0,k+1,...,k+7] directly.
        *(uint2_radp *)(&sW[(wrow[it] + q) * LWSTR + wcol[it]]) = uint2_radp{be, bo};
      }
    }
    radp_lds_barrier();

    // radiance_mxfp4_fp8.hip:703-718, verbatim, EXCEPT for the two
    // convert_lo/convert_hi calls interleaved at steps 0 and 1 (see the
    // lambda-definition comment above for why): NS is LBK/16, i.e. 4 for
    // TN=4/LBK=64 and 8 for TN=2/LBK=128, so step 1 always exists and both
    // calls land in the FIRST HALF of the step loop, leaving the remaining
    // NS-2 steps' WMMA instructions (2 or 6 of them) after each call as
    // overlap window -- vs. the old placement where the whole convert_sT
    // body ran strictly after ALL NS steps, back to back with nothing to
    // hide its latency behind. Each call sits between step i's WMMA (just
    // issued above it in program order) and step i+1's `sched_barrier(0)`
    // (below it), i.e. inside one scheduling region together with step i's
    // WMMA -- sched_barrier(0) blocks reorder AT that fence, not between a
    // completed step and a later step's fence, so the backend is free to
    // interleave/dual-issue these VALU ops against the surrounding
    // matrix-core instructions instead of them forming their own serial
    // block. Guarded the same way convert_sT was (only on non-final slabs).
#pragma unroll
    for (int step = 0; step < NS; ++step) {
      const int kk = step * 16 + kb8;
      int2_radp wf[TN];
#pragma unroll
      for (int j = 0; j < TN; ++j) {
        const unsigned char *p = &sW[(wn * TN * 16 + j * 16 + col) * LWSTR + kk];
        wf[j][0] = *(const int *)p; wf[j][1] = *(const int *)(p + 4);
      }
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int i = 0; i < RADP_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af[i][step], wf[j], acc[i][j]);
      if (k0 + LBK < K) {
        if (step == 0) convert_lo(sT[buf ^ 1]);
        if (step == 1) convert_hi(sT[buf ^ 1]);
      }
    }
    // sT[buf^1] is now fully written (both halves, for every (gl,c0)
    // thread) by the interleaved calls above; the trailing barrier below
    // publishes it for the next iteration -- no extra barrier (same
    // double-buffering contract as before, just filled earlier).
    radp_lds_barrier();
  }

  // Epilogue: radiance_mxfp4_fp8.hip:722-757, verbatim except (c) --
  // rf[j] is colscale[n] (a direct fp32 read) instead of 2^Wref[n].
  float rf[TN];
#pragma unroll
  for (int j = 0; j < TN; ++j) {
    const int n = n0 + wn * TN * 16 + j * 16 + col;
    rf[j] = (n < N) ? colscale[n] : 0.f;
  }
  {
    float *__restrict__ Cb = C + (size_t)(m0 + wm * RADP_TM * 16) * N;
    const float *__restrict__ Asb = As + m0 + wm * RADP_TM * 16;
    const bool full = (m0 + wm * RADP_TM * 16 + (RADP_TM - 1) * 16 + kb8 + 7 < M) &&
                      (n0 + wn * TN * 16 + (TN - 1) * 16 + col < N);
    if (full) {
#pragma unroll
      for (int i = 0; i < RADP_TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int r = i * 16 + kb8 + e;
            const int n = n0 + wn * TN * 16 + j * 16 + col;
            Cb[r * N + n] = acc[i][j][e] * rf[j] * Asb[r];
          }
      return;
    }
  }
#pragma unroll
  for (int i = 0; i < RADP_TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m0 + wm * RADP_TM * 16 + i * 16 + kb8 + e;
        const int n = n0 + wn * TN * 16 + j * 16 + col;
        if (m < M && n < N) C[(size_t)m * N + n] = acc[i][j][e] * rf[j] * As[m];
      }
}

#endif   // __gfx1201__ || host pass

// ─────────────────────────────────────────────────────────────────────────
// Retile-A kernel: ported arithmetic from radiance_silu_mul_quant's TILED
// store (radiance_mxfp4_fp8.hip:1495), see header comment. Pure byte
// shuffle -- no fp8 math, no gfx12-only intrinsics -- so this compiles for
// every offload arch (like ml84_trfeed_pack_host/ml84_pack_kernel).
// ─────────────────────────────────────────────────────────────────────────
__global__ void radp_retile_a_kernel(const uint8_t *__restrict__ A, uint8_t *__restrict__ AT,
                                      int M, int K, int Mt) {
    const int KT = K >> 4;
    const size_t total = (size_t)Mt * (size_t)KT * 32;
    for (size_t tix = (size_t)blockIdx.x * blockDim.x + threadIdx.x; tix < total;
         tix += (size_t)gridDim.x * blockDim.x) {
        const int L = (int)(tix % 32);
        const size_t rest = tix / 32;
        const int kt = (int)(rest % (size_t)KT);
        const int mt = (int)(rest / (size_t)KT);
        const int lrow = L & 15, half = (L >> 4) & 1;
        const int r = mt * 16 + lrow;
        if (r >= M) continue;   // pad row: left at the memset-zero AT started with
        const size_t at_base = ((size_t)mt * KT + kt) * 256 + (size_t)L * 8;
        const size_t a_base = (size_t)r * K + kt * 16 + half * 8;
#pragma unroll
        for (int b = 0; b < 8; ++b) AT[at_base + b] = A[a_base + b];
    }
}

bool rdna4_gemm_ml84_radiance_retile_a(int M, int K, const uint8_t* A_rowmajor_fp8,
                                        uint8_t* AT_out, hipStream_t stream) {
    if (M <= 0 || K <= 0 || (K % 128) != 0) return false;
    const int Mt = (M + 15) >> 4;
    const size_t at_bytes = (size_t)Mt * 16 * (size_t)K;
    if (hipMemsetAsync(AT_out, 0, at_bytes, stream) != hipSuccess) return false;
    const int KT = K >> 4;
    const size_t total = (size_t)Mt * (size_t)KT * 32;
    constexpr int TPB = 256;
    const int blocks = (int)std::min<size_t>((total + TPB - 1) / TPB, 65535 * 8u);
    radp_retile_a_kernel<<<blocks, TPB, 0, stream>>>(A_rowmajor_fp8, AT_out, M, K, Mt);
    return hipGetLastError() == hipSuccess;
}

void rdna4_ml84_radiance_retile_a_host(int M, int K, const uint8_t* A_rowmajor_fp8, uint8_t* AT_out) {
    const int Mt = (M + 15) >> 4;
    const int KT = K >> 4;
    std::memset(AT_out, 0, (size_t)Mt * 16 * (size_t)K);
    for (int mt = 0; mt < Mt; ++mt)
        for (int kt = 0; kt < KT; ++kt)
            for (int L = 0; L < 32; ++L) {
                const int lrow = L & 15, half = (L >> 4) & 1;
                const int r = mt * 16 + lrow;
                if (r >= M) continue;
                const size_t at_base = ((size_t)mt * KT + kt) * 256 + (size_t)L * 8;
                const size_t a_base = (size_t)r * K + kt * 16 + half * 8;
                for (int b = 0; b < 8; ++b) AT_out[at_base + b] = A_rowmajor_fp8[a_base + b];
            }
}

// ─────────────────────────────────────────────────────────────────────────
// Prep: colscale (bit-identical formula to gemm_ml84_prod.hip's
// ml84_scale_colmax_kernel) then T[g][n][16] (bit-identical formula to
// ml84_expand_kernel's per-element requantization, gemm_ml84_prod.hip:280,
// just written out per-centroid instead of per-(n,k) element since every
// element of a (group,column) shares the same 16 outputs).
// ─────────────────────────────────────────────────────────────────────────
__global__ void radp_colscale_kernel(const float* __restrict__ b_scale_g, const uint8_t* __restrict__ lut,
                                      float* __restrict__ colscale_out, int N, int n_groups_k) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float mx = 0.f;
    for (int g = 0; g < n_groups_k; ++g) {
        float lut_amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) lut_amax = fmaxf(lut_amax, fabsf(ml84_decode_e4m3(lut[(size_t)g * 16 + i])));
        mx = fmaxf(mx, b_scale_g[(size_t)g * (size_t)N + n] * lut_amax);
    }
    const float cs = fmaxf(mx / 448.0f, 1e-12f);
    colscale_out[n]     = cs;
    colscale_out[N + n] = 1.0f / (cs > 0.f ? cs : 1.f);   // inverse for the per-slab table build
}

bool rdna4_gemm_ml84_radiance_prep(const uint8_t* lut, const float* b_scale_g,
                                    int N, int K, float* colscale_out, hipStream_t stream) {
    if (N <= 0 || K <= 0 || (K % QK_ML8) != 0 || (N % 16) != 0) return false;
    const int n_groups_k = K / QK_ML8;
    constexpr int TPB = 256;
    radp_colscale_kernel<<<(N + TPB - 1) / TPB, TPB, 0, stream>>>(b_scale_g, lut, colscale_out, N, n_groups_k);
    return hipGetLastError() == hipSuccess;
}

void rdna4_ml84_radiance_prep_host(const uint8_t* lut, const float* b_scale_g,
                                    int N, int K, float* colscale_out) {
    const int n_groups_k = K / QK_ML8;
    std::vector<float> lut_amax_g(n_groups_k, 0.f);
    for (int g = 0; g < n_groups_k; ++g) {
        float mx = 0.f;
        for (int i = 0; i < 16; ++i) mx = std::max(mx, std::fabs(ml84_decode_e4m3(lut[(size_t)g * 16 + i])));
        lut_amax_g[g] = mx;
    }
    for (int n = 0; n < N; ++n) {
        float mx = 0.f;
        for (int g = 0; g < n_groups_k; ++g) mx = std::max(mx, b_scale_g[(size_t)g * (size_t)N + n] * lut_amax_g[g]);
        const float cs = std::max(mx / 448.0f, 1e-12f);
        colscale_out[n]     = cs;
        colscale_out[N + n] = 1.0f / (cs > 0.f ? cs : 1.f);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Launcher: radiance_mxfp4_fp8.hip:1660-1693's launch_at_impl/launch_at,
// verbatim shape (K%128 check, tn4_min_m env, grid/block math), B/scale
// arguments swapped for T/colscale/B_nib per the format change.
// ─────────────────────────────────────────────────────────────────────────
bool rdna4_gemm_ml84_radiance(int M, const void* A_fp8, const float* a_scale, const uint8_t* B_nib,
                               const uint8_t* lut, const float* b_scale_g, const float* colscale, float* C_f32,
                               int N, int K, hipStream_t stream) {
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (K % 128) return false;          // radiance_mxfp4_fp8.hip:1663, "tiled A needs K % 128 == 0"
    if (K % QK_ML8) return false;       // group granularity
    if (N % 16) return false;
    static const int tn4_min_m = [] {
        const char* e = std::getenv("RADIANCE_MXFP4_TN4_MIN_M");
        return e ? std::atoi(e) : 2048;
    }();
    const dim3 block(RADP_NTHREADS);
    const auto *AT = static_cast<const unsigned char*>(A_fp8);
    if (M >= tn4_min_m && N >= RADP_BNF_OF(4)) {
        constexpr int B_ = RADP_BNF_OF(4);
        const dim3 grid((N + B_ - 1) / B_, (M + RADP_BMF - 1) / RADP_BMF);
        hipLaunchKernelGGL((radp_gemm_ml84_atiled<4, 64>), grid, block, 0, stream,
                            AT, B_nib, lut, b_scale_g, colscale, a_scale, C_f32, M, N, K, 1 << 30, 1 << 30, 0L);
    } else {
        constexpr int B_ = RADP_BNF_OF(2);
        const dim3 grid((N + B_ - 1) / B_, (M + RADP_BMF - 1) / RADP_BMF);
        hipLaunchKernelGGL((radp_gemm_ml84_atiled<2, 128>), grid, block, 0, stream,
                            AT, B_nib, lut, b_scale_g, colscale, a_scale, C_f32, M, N, K, 1 << 30, 1 << 30, 0L);
    }
    return hipGetLastError() == hipSuccess;
}
