// r4d_gdn: the gated-delta-net chunked scan, gfx1201 WMMA. Entry point
// r4d_gdn_chunk_scan_k128_v128_c64_bf16 -- head_k 128, head_v 128, chunk 64, bf16 in.
// Named for the algorithm rather than the serving phase: the chunked scan is what prefill and
// chunked prefill both run, while decode runs the recurrent update, which is a different kernel.
// It replaces FLA's recompute_w_u_fwd + chunk_gated_delta_rule_fwd_h + chunk_fwd_o: the
// recurrent state lives in WMMA accumulators across the whole chunk loop, so h / w / u /
// v_new are never written to or read from HBM.
//
// Decay is carried on a per-chunk reference c = (g_first + g_last)/2 rather than on
// g_last, which lets the (i,j) decay factor separate exactly:
//     e^{g_i-g_j} = e^{g_i-c} * e^{c-g_j}          (j <= i, so the product is <= 1)
// Folding the left half into the score matrix and the right half into V' means the
// UNGATED Vd is never needed -- one LDS buffer and two barriers disappear, and the
// 16 per-lane __expf in the score step collapse to one multiply by a precomputed row
// scale. Splitting at the midpoint bounds each half by e^{span/2}; a g_last reference
// would need e^{span}, and measured spans already reach 64 of the 88 fp32 allows.
//
//   ge[j]=e^{g_j}   rs[i]=scale*e^{g_i-c}   gv[j]=e^{c-g_j}   lam=e^{g_last-c}
//   U  = K@S            W  = beta*(V - ge*U)      V' = gv*(A@W)
//   P  = Q@K^T          sA = rs*tril(P)
//   O  = scale*ge*(Q@S) + sA@V'
//   S  = lam*(e^c*S + V'^T@K)
// A is FLA's solve_tril output (decay already folded in), so no decay enters W or V'.
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_wmma.h"
// TRIL: skip GEMM tiles above the diagonal (A from solve_tril is unit lower
// triangular, and sA is masked). 2.4% slower, and left off; the reason is
// structural rather than tuning: the waves rendezvous at a workgroup barrier, so only
// the critical path matters. Every wave computes 2 P tiles; only 10 of the 16 are
// needed, but 10 tiles over 8 waves still leaves some wave holding 2 -- the barrier's
// arrival time does not move, and the skipped waves just wait longer. On top of that
// the guards stop the compiler batching fragment loads ahead of the WMMA chain.
// Correctness is unaffected either way; the flag is kept so this stays reproducible.
#ifndef TRIL
#define TRIL 0
#endif
// LDSBAR: __syncthreads() is an ALL-address-space fence, so the barrier that follows
// the global O store compiles to s_wait_storecnt_dscnt 0x0 -- every wave waits for its
// output stores to reach cache before an LDS-only rendezvous. Scoping the fence to
// "local" keeps the required LDS ordering and drops the storecnt wait.
#ifndef LDSBAR
#define LDSBAR 1
#endif
#if LDSBAR
#define BAR() do {                                                        \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");       \
    __builtin_amdgcn_s_barrier();                                         \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");       \
  } while (0)
#else
#define BAR() __syncthreads()
#endif

#define BT   64
#define KD  128
#define VD  128
#define BV   64                 // V per workgroup; also halves the Q@K^T redundancy
#ifndef KP
#define KP  132                 // row pitch of the [.][KD] buffers, in shorts
#endif
#ifndef AP
#define AP   68                 // row pitch of the [.][BT] buffers, in shorts
#endif
#define WP  AP
// NW: waves per workgroup. The [4 token][4 v] tile grid is fixed by BT and BV, so the
// wave count decides how many tiles each wave carries: 8 waves take two, 16 take one.
// One tile per wave halves the live accumulators (and the state slice), which is what
// this kernel is short of -- at NW=8 it sits on 242 VGPRs and keeping one more address
// live spills 146 of them.
#ifndef NW
#define NW    8
#endif
#define NTHR (NW * 32)
#if NW == 8
#define NTV   2                 // n tiles per wave in the main / V' / O loops
#elif NW == 16
#define NTV   1
#else
#error "NW must be 8 or 16"
#endif
// SASG: how a wave's slice of the 8 k-tile x 4 v-tile state grid is shaped. The A
// operand of the state update is K TRANSPOSED, which costs eight scalar LDS reads per
// fragment against one for an untransposed one, so the shape that reads the FEWEST
// k-tiles per wave wins: one k-tile x four v-tiles reads 4 fragT + 16 fragP per chunk
// where two x two reads 8 + 8. Pricing the transposed reads at all (ABL=7) says they are
// 3.0% of the kernel.
// Slower: 237.6 us against 196.7. Reading half as many transposed fragments does not
// pay for holding four V' fragments live per k step -- the shape that reads fewer
// k-tiles reads more of everything else. Left at 0 (two k-tiles x two v-tiles).
#ifndef SASG
#define SASG 0
#endif
#if SASG && NW == 8
#define SKT   1                 // state k-tiles per wave
#define SVT   4                 // state v-tiles per wave
#elif SASG && NW == 16
#define SKT   1
#define SVT   2
#elif NW == 8
#define SKT   2
#define SVT   2
#else
#define SKT   1
#define SVT   2
#endif
#define NST (SKT * SVT)         // state tiles per wave

struct Smem {
  unsigned short k[BT][KP];     // A: K   /  B: K^T for P=Q@K^T
  unsigned short A[BT][AP];     // B-source for sA (written as [i][j])
  unsigned short S[BV][KP];     // B: state, V-major (matches FLA's [V,K] layout)
  unsigned short W[BV][WP];     // B: W (transposed)
  unsigned short D[BV][AP];     // staged V as [token][v]; then V' as [v][token].
                                // Both live inside one chunk and never overlap: V dies
                                // in the W update, V' is written after barrier (2).
  float ge[BT], gv[BT], rs[BT], bt[BT];
};

// LDSO: hold each LDS row pointer as one opaque 32-bit byte offset (see fragO).
#ifndef LDSO
#define LDSO 1
#endif
#ifndef SCHED
#define SCHED 5
#endif
#ifndef SB          // k steps per sched group in the V' and O+= loops
#define SB 2
#endif
#ifndef SS          // k steps per sched group in the state update
#define SS 1
#endif
// GO: address the global q / A fragments as a 32-bit element offset off a uniform base
// (fragGo) instead of a per-lane 64-bit pointer (rowG + fragP). See r4d_gdn_wmma.h --
// the pointer form makes LLVM rebuild v_mad_co_i64_i32 + v_lshlrev_b64 + a 64-bit add
// per fragment at this kernel's register pressure.
#ifndef GO
#define GO 1
#endif
// LDSO2: the same treatment for the V' / O / state loops' LDS rows -- one opaque byte
// offset per buffer, so every k step is an immediate on ds_load_2addr_b32 (LDSO already
// does this for the main loop's five fragments).
// It SPILLS at NW=8 (242 VGPRs already, 146 spilled, 2.5x slower) and is only usable
// when a wave carries one tile.
#ifndef LDSO2
#define LDSO2 (NW == 16)
#endif
// WIDE: stage k and v eight shorts per lane (global_load_b128) instead of four, and
// address them as a 32-bit element offset off a chunk-uniform base. Halves the staging
// loads, and with a wave-uniform row (PFV >= 1) the address is one v_add on an SALU row
// offset instead of v_mad_co_i64_i32 + v_lshlrev_b64 + a 64-bit add.
#ifndef WIDE
#define WIDE 1
#endif
// PFIN: issue the next chunk's six staging loads INSIDE the main WMMA loop, one per k step,
// instead of as a block before it. Nothing orders them -- they read global memory and write
// registers consumed a whole chunk later -- and as a block they land in the 267-instruction
// stretch between the main loop's last WMMA and the V' loop's first, which has no WMMA to hide
// their addressing behind.
// Slower: 206.4 us against 196.2. The loads themselves are one v_min and an access, but
// interleaving them costs the main loop's WMMA schedule more than the block costs in the gap.
// Left at 0. (Nor does an in-loop bf16 rounding trick exist to shrink that gap: gfx1201 has no
// v_add_nc_u64, so two floats cannot be rounded with one add, and v_pk_add_u16 does not carry
// between halves -- which IS the rounding.)
#ifndef PFIN
#define PFIN 0
#endif
// APRE: issue the V' loop's four A fragments (global) BEFORE the main WMMA loop instead
// of at the top of the V' loop. They are read immediately after barrier (2), so their HBM
// latency is otherwise fully exposed; hoisted, they land during the main loop's 48 WMMAs.
// Costs 16 VGPRs.
// Slower: 255.7 us against 220.5. It does not spill (227 VGPRs) but the sixteen
// registers it keeps live across the main loop cost more than the latency it hides, and
// the V' loop's sched group already batches those loads a whole step ahead. Left at 0.
#ifndef APRE
#define APRE 0
#endif
// ABL: timing-only ablation ladder -- every nonzero value produces WRONG results and
// exists solely to price one stage. 1 = drop the state-update WMMAs, 2 = drop O += sA@V',
// 3 = drop V' = A@W, 4 = drop P from the main loop, 5 = drop U, 6 = drop O^T,
// 7 = read the state update's K^T operand with a normal (untransposed) fragP, which has
// the same WMMA count and 2 LDS reads per fragment instead of 8.
#ifndef ABL
#define ABL 0
#endif
// BFR: which bf16 packs round. 1 = all (safe), 2 = intermediates truncate but the
// recurrent state and the kernel output still round, 0 = all truncate.
// Rejected: truncating is 1.5% (BFR=2) / 2.2% (BFR=0) faster but takes the fp64-oracle
// error from 3.30e-03 (FLA parity) to 9.02e-03 / 1.17e-02, and the state error from
// 2.4e-03 to 5.1e-03 -- W/sA/V' feed the recurrence, so even BFR=2 pollutes the state.
// Left at 1.
#ifndef BFR
#define BFR 1
#endif
// PRIO: raise this wave's issue priority across the WMMA-dense region so it is not
// starved by waves in their memory phase.
#ifndef PRIO
#define PRIO 0
#endif
// SID: give each WMMA loop its own sched_group_barrier syncid, so one loop's group
// pattern cannot be satisfied by another loop's instructions when the scheduler puts
// them in the same region.
#ifndef SID
#define SID 0
#endif
// NOVM: leave the global q/A loads out of the group pattern and let the scheduler
// place them freely.
#ifndef NOVM
#define NOVM 0
#endif
#if SID
#define SIDF 1
#define SIDV 2
#define SIDO 3
#define SIDS 4
#else
#define SIDF 0
#define SIDV 0
#define SIDO 0
#define SIDS 0
#endif
#if PRIO
#define PRIO_HI() __builtin_amdgcn_s_setprio(1)
#define PRIO_LO() __builtin_amdgcn_s_setprio(0)
#else
#define PRIO_HI() ((void)0)
#define PRIO_LO() ((void)0)
#endif
#if BFR == 1
#define F2BF4I f2bf4                 // W, sA, V' -- consumed inside one chunk
#define F2BF4O f2bf4                 // s.S and o -- carried across chunks / returned
#elif BFR == 2
#define F2BF4I f2bf4t
#define F2BF4O f2bf4
#else
#define F2BF4I f2bf4t
#define F2BF4O f2bf4t
#endif
// LDS reads and WMMAs per k step of the main loop, for the sched groups below:
// one A fragment (K row mt) shared by every tile, plus S and K per n tile.
// One ds_load_2addr_b64 per LDS fragment (see fragP in r4d_gdn_wmma.h) -- these counts
// are what the sched groups below pattern on, and a count that does not match what the
// loop actually contains produces a WORSE schedule than no pattern at all.
#define MLDS (1 + 2 * NTV)
#define MWMA (3 * NTV)
// byte offset of buffer[base + lo][4*hi] from the base of the shared block
#define ROWO(buf, pitch, base) \
  ((unsigned)((const char*)&(buf)[0][0] - (const char*)&s) + \
   (unsigned)((((base) + lo) * (pitch) + 4 * hi) * 2))

__global__ __launch_bounds__(NTHR) void r4d_gdn_chunk_scan_kernel(
    const unsigned short* __restrict__ q, const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v, const unsigned short* __restrict__ Amat,
    const float* __restrict__ gcs, const float* __restrict__ beta,
    const float* __restrict__ h0, unsigned short* __restrict__ o,
    float* __restrict__ ht, const int* __restrict__ cu, int H, int Hg, float scale)
{
  __shared__ Smem s;
  const int bv = blockIdx.x, hv = blockIdx.y, nq = blockIdx.z;
  const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
  const int hq = hv / (H / Hg), v0 = bv * BV;
  const int tb = cu[nq], te = cu[nq + 1];
  const int nchunk = (te - tb + BT - 1) / BT;
  const int lo = lane & 15, hi = lane >> 4;

#if NW == 8
  const int mt = w >> 1, ntb = (w & 1) * 2;                 // [BT][BV]: 2 tiles/wave
#else
  const int mt = w >> 2, ntb = (w & 3);                     // [BT][BV]: 1 tile/wave
#endif
  // The state accumulator is held TRANSPOSED, C[m=k-dim][n=v]. WMMA A and B fragments
  // share a lane layout, so swapping the operands of the state update is free -- and it
  // makes k-dim the per-lane contiguous axis, so the per-chunk writeback to s.S is four
  // packed 8-short stores instead of 32 scalar ds_store_b16.
#if SASG && NW == 8
  const int kt0 = w, vt0 = 0;                                // [KD][BV]: 1 x 4 tiles/wave
#elif SASG && NW == 16
  const int kt0 = (w >> 1), vt0 = (w & 1) * 2;               // 1 x 2
#elif NW == 8
  const int kt0 = (w >> 1) * 2, vt0 = (w & 1) * 2;           // 2 x 2
#else
  const int kt0 = (w >> 1), vt0 = (w & 1) * 2;               // 1 x 2
#endif

#if WIDE
#define PFW 8                   // shorts staged per lane per step
#define pfv_t uint4
#define PFZERO make_uint4(0u, 0u, 0u, 0u)
#else
#define PFW 4
#define pfv_t uint2
#define PFZERO make_uint2(0u, 0u)
#endif
#define NKB (BT * KD / (NTHR * PFW))
#define NVB (BT * BV / (NTHR * PFW))
  pfv_t kb[NKB], vb[NVB];
  // Staging index, split into a WAVE-UNIFORM row and a loop-invariant lane column.
  // Written as `i = tid*4 + u*NTHR*4; r = i>>7` the row is opaquely divergent to LLVM,
  // so each of the 12 loads rebuilt a full 64-bit address (v_mad_co_i64_i32 +
  // v_lshlrev_b64 + 64-bit add) with the bounds test as exec-mask manipulation --
  // ~13 instructions per 8-byte load. readfirstlane makes the row provably uniform, so
  // the row address is SALU and the vector part is one loop-invariant register.
  // Identities (exact, since u*NTHR*4 is a multiple of both 128 and 64):
  //   k: r = (tid>>5) + NW*u          c = (tid*4) & 127
  //   v: r = 2*(tid>>5) + vh + 2*NW*u  c = (tid*4) & 63   vh = (tid>>4) & 1
  const int wu = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int kc = (tid * PFW) & (KD - 1), vc = (tid * PFW) & (BV - 1);
#if WIDE
  const int kr0 = 2 * wu + ((tid >> 4) & 1), vr0 = 4 * wu + ((tid >> 3) & 3);
#define KRU (2 * NW)
#define VRU (4 * NW)
#else
  const int kr0 = wu, vr0 = 2 * wu + ((tid >> 4) & 1);
#define KRU NW
#define VRU (2 * NW)
#endif
  // Rows past the chunk end are CLAMPED, not zeroed. Nothing they reach is live:
  //   * beta[t>=nval] = 0, so W = beta*(V - ge*U) is 0 whatever V and U hold;
  //   * gv[t>=nval]  = 0, and V' is scaled by gv[t] as it is stored, so the state
  //     update's contraction over t already drops those columns;
  //   * rs[i>=nval]  = 0 and the tril mask kills j>i, so sA is clean both ways;
  //   * the O store is guarded by tok < nval.
  // A clamped row is real (finite) data, so no NaN can enter through the 0 factors.
  // This removes 12 selects, the exec-mask manipulation around each address, and the
  // two zero registers -- and since the row is wave-uniform the clamp is pure SALU.
// PFV: 0 = original (divergent index, predicated)   1 = wave-uniform + predicate
//      2 = wave-uniform + clamp                      3 = divergent index + clamp
// Measured round-robin in one process (4-arm, bit-identical outputs): 0 = 245.0 us,
// 1 = 215.5, 2 = 246.0, and with WIDE: 1 = 211.1, 2 = 202.4. The clamp only pays once
// the load is wide -- at PFW=4 it saves the selects but doubles nothing else, while at
// PFW=8 it also removes the exec-mask manipulation around a 128-bit load.
#ifndef PFV
#define PFV 4
#endif
// 4 = the clamp applied to a HOISTED per-lane byte offset. Once a lane stages eight
// shorts the staging row is no longer wave-uniform (lanes 0-15 and 16-31 are different
// rows), so `r * (Hg*KD)` is a v_mul_lo_u32 per load per chunk. Clamping the OFFSET
// instead of the row is the same value -- both add the same column -- and lets the whole
// product be computed once, outside the chunk loop.
#if PFV == 0 || PFV == 3
#define PFROWK (i / KD)
#define PFROWV (i / BV)
#define PFCOLK (i & (KD - 1))
#define PFCOLV (i % BV)
#define PFIDX  int i = tid * PFW + u * NTHR * PFW;
#else
#define PFROWK (kr0 + KRU * u)
#define PFROWV (vr0 + VRU * u)
#define PFCOLK kc
#define PFCOLV vc
#define PFIDX
#endif
#if PFV == 0 || PFV == 1
#define PFK(r, pnv, e) ((r) < (pnv) ? (e) : PFZERO)
#define PFCLAMP(r, prl) (r)
#else
#define PFK(r, pnv, e) (e)
#define PFCLAMP(r, prl) min((r), (prl))
#endif
#if PFV == 4
#define PFHOIST()                                                                    \
  const unsigned krs = (unsigned)(Hg * KD), vrs = (unsigned)(H * VD);                \
  unsigned koff0[NKB], voff0[NVB];                                                   \
  _Pragma("unroll") for (int u = 0; u < NKB; ++u)                                    \
      koff0[u] = (unsigned)(kr0 + KRU * u) * krs + (unsigned)kc;                     \
  _Pragma("unroll") for (int u = 0; u < NVB; ++u)                                    \
      voff0[u] = (unsigned)(vr0 + VRU * u) * vrs + (unsigned)vc;
#else
#define PFHOIST()
#endif
#define PFSTORE(dst, val)                                                            \
  do {                                                                               \
    unsigned short* dp_ = (dst);                                                     \
    *(uint2*)dp_ = make_uint2((val).x, (val).y);                                      \
    if (PFW == 8) *(uint2*)(dp_ + 4) = make_uint2(*(&(val).x + 2), *(&(val).x + 3));  \
  } while (0)
#if PFV == 4
// the chunk-uniform half, so the loads themselves are one v_min plus the access
#define PFBASE(CH)                                                                   \
    const int pc0_ = tb + (CH) * BT, prl_ = min(BT, te - pc0_) - 1;                  \
    const unsigned short* kb_ = k + ((size_t)pc0_ * Hg + hq) * KD;                   \
    const unsigned short* vb_ = v + ((size_t)pc0_ * H + hv) * VD + v0;               \
    const unsigned kcl = (unsigned)prl_ * krs + (unsigned)kc;                        \
    const unsigned vcl = (unsigned)prl_ * vrs + (unsigned)vc;
#define PFONE(u)                                                                     \
    do {                                                                             \
      if ((u) < NKB) kb[(u) < NKB ? (u) : 0] =                                       \
        *(const pfv_t*)(kb_ + min(koff0[(u) < NKB ? (u) : 0], kcl));                 \
      else if ((u) - NKB < NVB) vb[(u) >= NKB ? (u) - NKB : 0] =                     \
        *(const pfv_t*)(vb_ + min(voff0[(u) >= NKB ? (u) - NKB : 0], vcl));          \
    } while (0)
#define PREFETCH(CH)                                                                 \
  {                                                                                  \
    PFBASE(CH)                                                                       \
    _Pragma("unroll") for (int u = 0; u < NKB + NVB; ++u) PFONE(u);                  \
  }
#else
#define PREFETCH(CH)                                                                 \
  {                                                                                  \
    int pc0 = tb + (CH) * BT, pnv = min(BT, te - pc0), prl = pnv - 1;                \
    const unsigned short* kb_ = k + ((size_t)pc0 * Hg + hq) * KD;                   \
    const unsigned short* vb_ = v + ((size_t)pc0 * H + hv) * VD + v0;               \
    _Pragma("unroll") for (int u = 0; u < NKB; ++u) {                                \
      PFIDX int r0 = PFROWK, r = PFCLAMP(r0, prl);                                   \
      kb[u] = PFK(r0, pnv,                                                           \
        *(const pfv_t*)(kb_ + (unsigned)(r * (Hg * KD)) + PFCOLK));                  \
    }                                                                                \
    _Pragma("unroll") for (int u = 0; u < NVB; ++u) {                                \
      PFIDX int r0 = PFROWV, r = PFCLAMP(r0, prl);                                   \
      vb[u] = PFK(r0, pnv,                                                           \
        *(const pfv_t*)(vb_ + (unsigned)(r * (H * VD)) + PFCOLV));                   \
    }                                                                                \
  }
#endif

  v8f St[NST];                  // state: NSK k-tiles x 2 v-tiles
#define KT(t) (kt0 + (t) / SVT)
#define VT(t) (vt0 + (t) % SVT)
#define SROW(t) ((size_t)(v0 + VT(t) * 16 + lo) * KD + KT(t) * 16 + 8 * hi)
  {
    const float* p = h0 + ((size_t)(nq * H + hv) * VD) * KD;
#pragma unroll
    for (int t = 0; t < NST; ++t) {
      const float4 a = *(const float4*)(p + SROW(t));
      const float4 b = *(const float4*)(p + SROW(t) + 4);
      St[t][0]=a.x; St[t][1]=a.y; St[t][2]=a.z; St[t][3]=a.w;
      St[t][4]=b.x; St[t][5]=b.y; St[t][6]=b.z; St[t][7]=b.w;
    }
  }
#define SWRITE()                                                                      \
  _Pragma("unroll") for (int t = 0; t < NST; ++t) {                                     \
    unsigned short* sp = &s.S[VT(t) * 16 + lo][KT(t) * 16 + 8 * hi];                  \
    *(uint2*)sp       = F2BF4O(St[t][0], St[t][1], St[t][2], St[t][3]);                \
    *(uint2*)(sp + 4) = F2BF4O(St[t][4], St[t][5], St[t][6], St[t][7]);                \
  }
  SWRITE();
  PFHOIST();
  BAR();
  PREFETCH(0);

  const char* const sbase = (const char*)&s;
  for (int ci = 0; ci < nchunk; ++ci) {
    const int c0 = tb + ci * BT, nval = min(BT, te - c0);
    const int lim = nval - 1;
    const unsigned short* qg = q + ((size_t)c0 * Hg + hq) * KD;
    const unsigned short* ag = Amat + ((size_t)c0 * H + hv) * BT;
    const size_t qrs = (size_t)Hg * KD, ars = (size_t)H * BT;
    const float gl = gcs[(size_t)(c0 + lim) * H + hv];
    const float cref = 0.5f * (gcs[(size_t)c0 * H + hv] + gl);

    // Issued before the staging store so the LDS writes below cover their latency:
    // only waves 0-1 (tid < BT) run this, and every other wave waits for them at (1).
    float ggr = 0.f, btr = 0.f;
    if (tid < BT) {
      bool ok = tid < nval;
      ggr = ok ? gcs[(size_t)(c0 + tid) * H + hv] : 0.f;
      btr = ok ? beta[(size_t)(c0 + tid) * H + hv] : 0.f;
    }
    { // consume the previous iteration's prefetch: the s_wait_loadcnt for these
      // loads now sits a whole chunk of compute after they were issued
#pragma unroll
      for (int u = 0; u < NKB; ++u) { PFIDX PFSTORE(&s.k[PFROWK][PFCOLK], kb[u]); }
#pragma unroll
      for (int u = 0; u < NVB; ++u) { PFIDX PFSTORE(&s.D[PFROWV][PFCOLV], vb[u]); }
    }
    if (tid < BT) {
      bool ok = tid < nval;
      s.bt[tid] = btr;
      s.ge[tid] = __expf(ggr);
      // The split form e^{g_i-c}.e^{c-g_j} is one exp per token instead of eight per tile, but each
      // half leaves range on its own where the PRODUCT never does: cref sits at the chunk midpoint,
      // so each half spans +/-(gate span)/2, and a span past ~176 sends one to +INF while the other
      // underflows to 0. Both consumers are bounded -- rs[i].gv[j] = scale.e^{g_i-g_j} under the
      // tril mask, and gv[t].lam = e^{gl-g_t} on the state path -- so clamping the growing half
      // leaves the shrinking one at or near 0 and the term is attenuated instead of 0 * INF = NaN.
      //
      // Attenuated, not exact. What leaves range is the distance from cref, not g_i-g_j, so on a
      // span-200 chunk the last token's own diagonal -- and its e^{gl-g_t} ~ 1 weight into the
      // state, which the next chunk reads -- are damped by e^{80-(cref-g_t)}. This bounds the
      // damage; it does not remove the cause. The cause is splitting a weight that is provably
      // <= 1 into a huge x tiny pair: stage V' in fp32, or apply e^{gl-g_t} on the state path.
      //
      // 80 rather than 88 because the binding ceiling is not fp32's: V' = gv[t].(A@W)[t] is staged
      // in bf16 (s.D) and then accumulated over BT tokens into V'^T @ K, so gv must leave room for
      // |A@W| under 3.4e38 -- e^88 = 1.6e38 leaves a factor of two and still produced NaN. The
      // value is measured, not assumed: WikiText-2 PPL over 208,539 tokens, no other mitigation --
      // 70 -> 8.3841, 80 -> 8.3706, 83 -> 8.3728, reference 8.3335, stock 653586. Where both
      // halves fit (every ordinary chunk) nothing changes.
      s.gv[tid] = ok ? __expf(fminf(cref - ggr, 80.0f)) : 0.f;
      s.rs[tid] = ok ? scale * __expf(fminf(ggr - cref, 80.0f)) : 0.f;
    }
    BAR();                                                   // (1)
#if PFIN && PFV == 4
    PFBASE(min(ci + 1, nchunk - 1))          // issued one per k step in the loop below
#else
    PREFETCH(min(ci + 1, nchunk - 1));       // issue only; consumed next iteration
#endif

#if APRE
    v8bf apre[BT / 16];
    {
#if GO
      const unsigned aoh = rowGo((unsigned)ars, mt * 16, lim, lane);
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) apre[sk] = fragGo(ag, aoh + sk * 16);
#else
      const unsigned short* pah = rowG(ag, ars, mt * 16, lim, lane);
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) apre[sk] = fragP(pah + sk * 16);
#endif
    }
#endif
    // ---- U = K@S, O^T = S@Q^T and P^T = K@Q^T share the q and k fragments.
    // tril: a whole P tile is masked away when its j-tile is above the i-tile, and the
    // O += sA@V' below skips the same tiles, so they need neither computing nor storing.
    v8f O[NTV], u[NTV], p[NTV];
    bool doP[NTV];
#pragma unroll
    for (int i = 0; i < NTV; ++i) {
      O[i] = (v8f){}; u[i] = (v8f){}; p[i] = (v8f){};
      doP[i] = (TRIL ? ntb + i <= mt : true);
    }
    PRIO_HI();
    {
#if LDSO
      const char* sb = sbase;
      unsigned oSn[NTV], okn[NTV];
      unsigned okm = ROWO(s.k, KP, mt * 16);
      LDSOPAQUE(okm);
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        oSn[i] = ROWO(s.S, KP, (ntb + i) * 16);
        okn[i] = ROWO(s.k, KP, (ntb + i) * 16);
        LDSOPAQUE(oSn[i]); LDSOPAQUE(okn[i]);
      }
#define pSnf(i, ko) fragO(sb, oSn[i] + (ko) * 2u)
#define pkmf(ko)    fragO(sb, okm + (ko) * 2u)
#define pknf(i, ko) fragO(sb, okn[i] + (ko) * 2u)
#else
      const unsigned short* pSn[NTV];
      const unsigned short* pkn[NTV];
      const unsigned short* pkm = rowP(&s.k[0][0], KP, mt * 16, lane);
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        pSn[i] = rowP(&s.S[0][0], KP, (ntb + i) * 16, lane);
        pkn[i] = rowP(&s.k[0][0], KP, (ntb + i) * 16, lane);
      }
#define pSnf(i, ko) fragP(pSn[i] + (ko))
#define pkmf(ko)    fragP(pkm + (ko))
#define pknf(i, ko) fragP(pkn[i] + (ko))
#endif
#if GO
      const unsigned qo = rowGo((unsigned)qrs, mt * 16, lim, lane);
#define pqf(ko) fragGo(qg, qo + (ko))
#else
      const unsigned short* pq  = rowG(qg, qrs, mt * 16, lim, lane);
#define pqf(ko) fragP(pq + (ko))
#endif
#pragma unroll
      for (int sk = 0; sk < KD / 16; ++sk) {
        const int ko = sk * 16;
        v8bf ak = pkmf(ko), aq = pqf(ko);
#if PFIN && PFV == 4
        if (sk < NKB + NVB) PFONE(sk);
#endif
#pragma unroll
        for (int i = 0; i < NTV; ++i) {
          v8bf b = pSnf(i, ko);
#if ABL != 5
          u[i] = mma(ak, b, u[i]);
#endif
#if ABL != 6
          O[i] = mma(b, aq, O[i]);                     // O^T: m = v-tile, n = token-tile
#endif
#if ABL != 4
          if (doP[i]) p[i] = mma(pknf(i, ko), aq, p[i]);   // P^T: m = j
#endif
        }
      }
#if SCHED
      // Ask for one k step's LDS reads to issue against the previous step's WMMAs.
      // Per step: 5 LDS fragments = 10 ds_read2_b32, 1 global q fragment = 2 loads,
      // 6 WMMA. Without this the scheduler sinks each read onto its consumer and
      // every WMMA is preceded by a full s_wait_dscnt 0x0 drain.
#if SCHED == 4
      // Explicit one-step software pipeline: ask for step i+1's operand reads to be
      // issued BEFORE step i's WMMAs, so the wait in front of each WMMA group is for
      // loads that have had a full WMMA group to land.
      __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);
      __builtin_amdgcn_sched_group_barrier(0x100, MLDS, 0);
#pragma unroll
      for (int i = 0; i < KD / 16 - 1; ++i) {
        __builtin_amdgcn_sched_group_barrier(0x008, MWMA, 0);  // WMMA step i
        __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);     // reads step i+1
        __builtin_amdgcn_sched_group_barrier(0x100, MLDS, 0);
      }
      __builtin_amdgcn_sched_group_barrier(0x008, MWMA, 0);
#elif SCHED == 5 || SCHED == 6 || SCHED == 7
      // GRAN k steps per group. Coarser is better here: the group barrier PINS the
      // order at every boundary, so a per-step pattern over-constrains a loop the
      // scheduler can already interleave -- it just needs to be told not to sink every
      // read onto its consumer.
#if SCHED == 5
#define GRAN 2
#elif SCHED == 6
#define GRAN 4
#else
#define GRAN 8
#endif
#pragma unroll
      for (int i = 0; i < KD / (16 * GRAN); ++i) {
#if NOVM == 0
        __builtin_amdgcn_sched_group_barrier(0x020, 2 * GRAN, SIDF);
#endif
        __builtin_amdgcn_sched_group_barrier(0x100, MLDS * GRAN, SIDF);
        __builtin_amdgcn_sched_group_barrier(0x008, MWMA * GRAN, SIDF);
      }
#else
#pragma unroll
      for (int i = 0; i < KD / 16; ++i) {
#if SCHED == 3                       // finer: rotate reads through the WMMA stream
        _Pragma("unroll") for (int j = 0; j < 3; ++j) {
          __builtin_amdgcn_sched_group_barrier(0x100, MLDS / 2, 0);
          __builtin_amdgcn_sched_group_barrier(0x008, MWMA / 2, 0);
        }
#else
        __builtin_amdgcn_sched_group_barrier(0x020, 2, 0);      // VMEM read (q)
        __builtin_amdgcn_sched_group_barrier(0x100, MLDS, 0);  // DS read
        __builtin_amdgcn_sched_group_barrier(0x008, MWMA, 0);  // WMMA
#endif
      }
#endif
#endif
    }
    PRIO_LO();
    {
      float g = scale * s.ge[mt * 16 + lo];             // token index is constant per lane
#pragma unroll
      for (int i = 0; i < NTV; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) O[i][e] *= g;
    }
    // W = beta * (V - ge*U)   (beta folded here: (A*beta_col)@W == A@(beta_row*W))
    {
      float wa[NTV][8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {           // beta / ge read once for both tiles
        const int tok = mt * 16 + 8 * hi + e;
        const float f = s.bt[tok], g = s.ge[tok];
#pragma unroll
        for (int i = 0; i < NTV; ++i)
          wa[i][e] = f * (bf2f(s.D[tok][(ntb + i) * 16 + lo]) - g * u[i][e]);
      }
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        unsigned short* wp = &s.W[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
        *(uint2*)wp       = F2BF4I(wa[i][0], wa[i][1], wa[i][2], wa[i][3]);
        *(uint2*)(wp + 4) = F2BF4I(wa[i][4], wa[i][5], wa[i][6], wa[i][7]);
      }
    }
    // sA = rs * tril(P): one multiply per element, no exp
    {
      const int ii = mt * 16 + lo;            // now constant per lane: one rs load
      const float r = s.rs[ii];
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        const int j0 = (ntb + i) * 16 + 8 * hi;
        float a[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) a[e] = (j0 + e <= ii) ? p[i][e] * r : 0.f;
        if (doP[i]) {
          *(uint2*)&s.A[ii][j0]     = F2BF4I(a[0], a[1], a[2], a[3]);
          *(uint2*)&s.A[ii][j0 + 4] = F2BF4I(a[4], a[5], a[6], a[7]);
        }
      }
    }
    BAR();                                                   // (2)

    // ---- V' = gv * (A @ W)  -> D (transposed store is one contiguous pair of dwords)
    {
      v8f d[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) d[i] = (v8f){};
#if APRE
#define paf(ko) apre[(ko) / 16]
#elif GO
      const unsigned ao = rowGo((unsigned)ars, mt * 16, lim, lane);
#define paf(ko) fragGo(ag, ao + (ko))
#else
      const unsigned short* pa = rowG(ag, ars, mt * 16, lim, lane);
#define paf(ko) fragP(pa + (ko))
#endif
#if LDSO2
      unsigned own[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) { own[i] = ROWO(s.W, WP, (ntb + i) * 16);
                                      LDSOPAQUE(own[i]); }
#define pwnf(i, ko) fragO(sbase, own[i] + (ko) * 2u)
#else
      const unsigned short* pwn[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) pwn[i] = rowP(&s.W[0][0], WP, (ntb + i) * 16, lane);
#define pwnf(i, ko) fragP(pwn[i] + (ko))
#endif
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        if (TRIL && sk > mt) break;         // A is unit lower triangular
        const int ko = sk * 16;
        v8bf a = paf(ko);
#if ABL != 3
#pragma unroll
        for (int i = 0; i < NTV; ++i) d[i] = mma(a, pwnf(i, ko), d[i]);
#endif
      }
#if SCHED >= 2                  // per group: SB steps x (1 global A frag, NTV LDS, NTV WMMA)
#pragma unroll
      for (int i = 0; i < BT / (16 * SB); ++i) {
#if NOVM == 0 && APRE == 0
        __builtin_amdgcn_sched_group_barrier(0x020, 2 * SB, SIDV);
#endif
        __builtin_amdgcn_sched_group_barrier(0x100, NTV * SB, SIDV);
        __builtin_amdgcn_sched_group_barrier(0x008, NTV * SB, SIDV);
      }
#endif
      float gvv[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) gvv[e] = s.gv[mt * 16 + 8 * hi + e];
#pragma unroll
      for (int i = 0; i < NTV; ++i) {
        unsigned short* dp = &s.D[(ntb + i) * 16 + lo][mt * 16 + 8 * hi];
        *(uint2*)dp       = F2BF4I(d[i][0]*gvv[0], d[i][1]*gvv[1],
                                   d[i][2]*gvv[2], d[i][3]*gvv[3]);
        *(uint2*)(dp + 4) = F2BF4I(d[i][4]*gvv[4], d[i][5]*gvv[5],
                                   d[i][6]*gvv[6], d[i][7]*gvv[7]);
      }
    }
    BAR();                                                   // (3)

    // ---- O += sA @ V' ; store.  V' also feeds the state update, unchanged.
    {
#if LDSO2
      unsigned oA = ROWO(s.A, AP, mt * 16);                    // sA as [n=i][k=j]
      unsigned odn[NTV];
      LDSOPAQUE(oA);
#pragma unroll
      for (int i = 0; i < NTV; ++i) { odn[i] = ROWO(s.D, AP, (ntb + i) * 16);
                                      LDSOPAQUE(odn[i]); }
#define pAf(ko)     fragO(sbase, oA + (ko) * 2u)
#define pdnf(i, ko) fragO(sbase, odn[i] + (ko) * 2u)
#else
      const unsigned short* pA = rowP(&s.A[0][0], AP, mt * 16, lane);
      const unsigned short* pdn[NTV];
#pragma unroll
      for (int i = 0; i < NTV; ++i) pdn[i] = rowP(&s.D[0][0], AP, (ntb + i) * 16, lane);
#define pAf(ko)     fragP(pA + (ko))
#define pdnf(i, ko) fragP(pdn[i] + (ko))
#endif
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        if (TRIL && sk > mt) break;         // sA is masked to lower triangular
        const int ko = sk * 16;
        v8bf b = pAf(ko);
#if ABL != 2
#pragma unroll
        for (int i = 0; i < NTV; ++i) O[i] = mma(pdnf(i, ko), b, O[i]);
#endif
      }
#if SCHED >= 2                  // per group: SB steps x (1 + NTV LDS frags, NTV WMMA)
#pragma unroll
      for (int i = 0; i < BT / (16 * SB); ++i) {
        __builtin_amdgcn_sched_group_barrier(0x100, (1 + NTV) * SB, SIDO);
        __builtin_amdgcn_sched_group_barrier(0x008, NTV * SB, SIDO);
      }
#endif
    }
    {
      int tok = mt * 16 + lo;
      if (tok < nval) {
        unsigned short* op = o + ((size_t)(c0 + tok) * H + hv) * VD + v0;
#pragma unroll
        for (int i = 0; i < NTV; ++i) {
          unsigned short* q2 = op + (ntb + i) * 16 + 8 * hi;
          *(uint2*)q2       = F2BF4O(O[i][0], O[i][1], O[i][2], O[i][3]);
          *(uint2*)(q2 + 4) = F2BF4O(O[i][4], O[i][5], O[i][6], O[i][7]);
        }
      }
    }

    // ---- S = lam * (e^c * S + V'^T @ K)   (B needs K as [dim][token] -> strided fragT)
    {
      const float ec = __expf(cref), lam = __expf(gl - cref);
#pragma unroll
      for (int t = 0; t < NST; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] *= ec;
#if LDSO2
      unsigned ovn[SVT];
#pragma unroll
      for (int j = 0; j < SVT; ++j) { ovn[j] = ROWO(s.D, AP, (vt0 + j) * 16);
                                      LDSOPAQUE(ovn[j]); }                 // V': n = v
#define pvnf(j, ko) fragO(sbase, ovn[j] + (ko) * 2u)
#else
      const unsigned short* pvn[SVT];
#pragma unroll
      for (int j = 0; j < SVT; ++j) pvn[j] = rowP(&s.D[0][0], AP, (vt0 + j) * 16, lane);
#define pvnf(j, ko) fragP(pvn[j] + (ko))
#endif
#pragma unroll
      for (int sk = 0; sk < BT / 16; ++sk) {
        v8bf bv[SVT];
#pragma unroll
        for (int j = 0; j < SVT; ++j) bv[j] = pvnf(j, sk * 16);
#pragma unroll
        for (int ki = 0; ki < SKT; ++ki) {
#if ABL == 7
          v8bf ak = fragP(rowP(&s.k[0][0], KP, (kt0 + ki) * 16, lane) + sk * 16);
#else
          v8bf ak = fragT(&s.k[0][0], KP, (kt0 + ki) * 16, sk * 16, lane);  // K^T: m = k-dim
#endif
#if ABL != 1
#pragma unroll
          for (int j = 0; j < SVT; ++j)
            St[ki * SVT + j] = mma(ak, bv[j], St[ki * SVT + j]);
#else
          (void)ak;
#endif
        }
#if ABL == 1
#pragma unroll
        for (int j = 0; j < SVT; ++j) (void)bv[j];
#endif
      }
#if SCHED >= 2   // per group: SS steps x (SKT fragT = 8 scalar reads each + SVT fragP, NST WMMA)
#pragma unroll
      for (int i = 0; i < BT / (16 * SS); ++i) {
        __builtin_amdgcn_sched_group_barrier(0x100, (8 * SKT + SVT) * SS, SIDS);
        __builtin_amdgcn_sched_group_barrier(0x008, NST * SS, SIDS);
      }
#endif
#pragma unroll
      for (int t = 0; t < NST; ++t)
#pragma unroll
        for (int e = 0; e < 8; ++e) St[t][e] *= lam;
    }
    // no barrier: s.S aliases nothing, and barrier (2) already proves every wave
    // finished its U/O reads of s.S this chunk
    SWRITE();
    BAR();                                                   // (4)
  }

  float* p = ht + ((size_t)(nq * H + hv) * VD) * KD;
#pragma unroll
  for (int t = 0; t < NST; ++t) {
    *(float4*)(p + SROW(t))     = make_float4(St[t][0], St[t][1], St[t][2], St[t][3]);
    *(float4*)(p + SROW(t) + 4) = make_float4(St[t][4], St[t][5], St[t][6], St[t][7]);
  }
}

extern "C" int r4d_gdn_chunk_scan_k128_v128_c64_bf16(
    const void* q, const void* k, const void* v, const void* A, const void* g,
    const void* beta, const void* h0, void* o, void* ht, const void* cu,
    int N, int H, int Hg, int K, int V, int bt, float scale, void* stream)
{
  if (K != KD || V != VD || bt != BT) return -1;
  dim3 grid(V / BV, H, N), blk(NTHR);
  r4d_gdn_chunk_scan_kernel<<<grid, blk, 0, (hipStream_t)stream>>>(
      (const unsigned short*)q, (const unsigned short*)k, (const unsigned short*)v,
      (const unsigned short*)A, (const float*)g, (const float*)beta, (const float*)h0,
      (unsigned short*)o, (float*)ht, (const int*)cu, H, Hg, scale);
  return (int)hipGetLastError();
}

// The geometry baked into the kernel above, so a caller can test a model's GDN shape against it
// instead of hardcoding a copy of these three numbers.
extern "C" void r4d_gdn_dims(int* head_k, int* head_v, int* chunk) {
  if (head_k) *head_k = KD;
  if (head_v) *head_v = VD;
  if (chunk)  *chunk  = BT;
}
