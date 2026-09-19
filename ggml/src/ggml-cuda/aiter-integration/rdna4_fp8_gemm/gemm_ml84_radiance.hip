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
    const unsigned char *__restrict__ T, const float *__restrict__ colscale,
    const float *__restrict__ As, float *__restrict__ C, int M, int N, int K,
    int pb1, int pb2, long astride) {
  constexpr int NS = LBK / 16;
  constexpr int LWSTR = LBK + RADP_PAD;
  constexpr int BNF_T = RADP_WN * TN * 16;
  __shared__ unsigned char sW[BNF_T * LWSTR];

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

  for (int k0 = 0; k0 < K; k0 += LBK) {
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
      const int g = (k0 + wkst[it] * 16) / QK_ML8;
#pragma unroll
      for (int q = 0; q < WSLOTS; ++q) {
        const int n = wrc[it] + q;
        const unsigned char *Tqn = T + ((size_t)g * N + n) * 16;
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

    // radiance_mxfp4_fp8.hip:703-718, verbatim.
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
    }
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
    colscale_out[n] = fmaxf(mx / 448.0f, 1e-12f);
}

__global__ void radp_table_kernel(const float* __restrict__ b_scale_g, const uint8_t* __restrict__ lut,
                                   const float* __restrict__ colscale, uint8_t* __restrict__ T_out,
                                   int N, int n_groups_k) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t)n_groups_k * (size_t)N;
    if (idx >= total) return;
    const int n = (int)(idx % (size_t)N);
    const int g = (int)(idx / (size_t)N);
    const uint8_t* lut_g = lut + (size_t)g * 16;
    const float bsg = b_scale_g[(size_t)g * (size_t)N + n];
    const float denom = colscale[n] > 0.f ? colscale[n] : 1.f;
    uint8_t* Tg = T_out + idx * 16;
#pragma unroll
    for (int c = 0; c < 16; ++c) {
        const float centroid = ml84_decode_e4m3(lut_g[c]);
        Tg[c] = ml84_encode_e4m3(centroid * bsg / denom);
    }
}

bool rdna4_gemm_ml84_radiance_prep(const uint8_t* B_nib, const uint8_t* lut, const float* b_scale_g,
                                    int N, int K, uint8_t* T_out, float* colscale_out, hipStream_t stream) {
    (void)B_nib;   // not read -- see header comment
    if (N <= 0 || K <= 0 || (K % QK_ML8) != 0 || (N % 16) != 0) return false;
    const int n_groups_k = K / QK_ML8;
    constexpr int TPB = 256;
    radp_colscale_kernel<<<(N + TPB - 1) / TPB, TPB, 0, stream>>>(b_scale_g, lut, colscale_out, N, n_groups_k);
    if (hipGetLastError() != hipSuccess) return false;
    const size_t total = (size_t)n_groups_k * (size_t)N;
    const int blocks = (int)std::min<size_t>((total + TPB - 1) / TPB, 65535 * 8u);
    radp_table_kernel<<<blocks, TPB, 0, stream>>>(b_scale_g, lut, colscale_out, T_out, N, n_groups_k);
    return hipGetLastError() == hipSuccess;
}

void rdna4_ml84_radiance_prep_host(const uint8_t* lut, const float* b_scale_g,
                                    int N, int K, uint8_t* T_out, float* colscale_out) {
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
        colscale_out[n] = std::max(mx / 448.0f, 1e-12f);
    }
    for (int g = 0; g < n_groups_k; ++g) {
        for (int n = 0; n < N; ++n) {
            const float denom = colscale_out[n] > 0.f ? colscale_out[n] : 1.f;
            const float bsg = b_scale_g[(size_t)g * (size_t)N + n];
            uint8_t* Tg = T_out + ((size_t)g * N + n) * 16;
            for (int c = 0; c < 16; ++c) {
                const float centroid = ml84_decode_e4m3(lut[(size_t)g * 16 + c]);
                Tg[c] = ml84_encode_e4m3(centroid * bsg / denom);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Launcher: radiance_mxfp4_fp8.hip:1660-1693's launch_at_impl/launch_at,
// verbatim shape (K%128 check, tn4_min_m env, grid/block math), B/scale
// arguments swapped for T/colscale/B_nib per the format change.
// ─────────────────────────────────────────────────────────────────────────
bool rdna4_gemm_ml84_radiance(int M, const void* A_fp8, const float* a_scale, const uint8_t* B_nib,
                               const uint8_t* T, const float* colscale, float* C_f32, int N, int K,
                               hipStream_t stream) {
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
                            AT, B_nib, T, colscale, a_scale, C_f32, M, N, K, 1 << 30, 1 << 30, 0L);
    } else {
        constexpr int B_ = RADP_BNF_OF(2);
        const dim3 grid((N + B_ - 1) / B_, (M + RADP_BMF - 1) / RADP_BMF);
        hipLaunchKernelGGL((radp_gemm_ml84_atiled<2, 128>), grid, block, 0, stream,
                            AT, B_nib, T, colscale, a_scale, C_f32, M, N, K, 1 << 30, 1 << 30, 0L);
    }
    return hipGetLastError() == hipSuccess;
}
