// r4d_gdn_kkt_solve_k128_c64_bf16.hip -- the gated-delta-net chunk preamble in one kernel:
//
//     A = (I + strict_lower(diag(beta) . K K^T . e^{g_i - g_j}))^{-1}     per chunk of 64
//
// It replaces FLA's chunk_scaled_dot_kkt_fwd + solve_tril, whose intermediate is a 64x64 fp32
// matrix per (chunk, head) written to HBM and read straight back -- 16 KB in and 16 KB out per
// chunk-head, for a matrix that is dead the moment the inverse is formed. Here the gram never
// leaves LDS, and only the bf16 inverse is stored.
//
// Two structural wins over the pair it replaces, both from doing the whole thing at once:
//
//   * THE GRAM IS COMPUTED ONCE PER K HEAD, NOT ONCE PER V HEAD. K K^T does not depend on beta or
//     g, and this model has three v heads per k head, so FLA computes the same 64x64x128 matmul
//     three times. One workgroup owns a (chunk, k head) and applies each v head's scales to the
//     gram it already holds in registers.
//   * THE GRAM IS COMPUTED IN ITS UPPER TRIANGLE. K K^T is symmetric, so the strictly LOWER half
//     of A can be read out of the upper tiles -- and that orientation is the one where a WMMA
//     accumulator's eight elements are eight CONSECUTIVE columns of one row of A, i.e. one
//     16-byte LDS store instead of eight 2-byte ones.
//
// The decay is applied as e^{g_i-g_j} PER ELEMENT, not split across a per-chunk reference. The
// split form (e^{g_i-c} . e^{c-g_j}) is one exp per token instead of eight per tile, and it is what
// the chunked scan uses -- but it only holds while the chunk's gate span stays inside fp32's
// exponent range, and each half can overflow on its own even where the product cannot. Here the
// difference is always <= 0 (g decreases along the chunk and only i > j is kept), so the direct
// form cannot overflow at any span, and the exps it costs are eight per tile against a kernel
// whose cost is memory.
//
// The inverse is blocked 16 -> 32 -> 64. The four diagonal blocks are inverted by fp32 forward
// substitution (one wave each, one lane per column), which is both cheaper than a WMMA Neumann
// series at this size and exact in fp32 -- FLA does the same thing for the same reason. The two
// 32-merges and the 64-merge are WMMA, using the identity
//
//     [[P, 0], [R, Q]]^-1 = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]]
//
// Every buffer is named for the layout it holds because the WMMA contract fixes it:
// mma(X, Y) computes X . Y^T from two row-major, k-contiguous operands, so a product's left
// operand must be row-major and its right operand must be stored transposed. An accumulator
// writes its transpose contiguously and its row-major form scattered, which is why the
// intermediates below are staged transposed and the final M is assembled row-major (it is stored
// to HBM by rows).
#include <hip/hip_runtime.h>
#include "r4d.h"
#include "r4d_gdn_wmma.h"

#define BT   64                 // chunk
#define KD  128                 // head_k
#define NBK   4                 // 16x16 blocks per chunk edge
#ifndef AP
#define AP   68                 // row pitch of the [BT][BT] buffers, in shorts. 136 B: 8-byte
#endif                          // aligned for the paired stores, and 136 mod 128 has gcd 8 with
                                // 128, so 16 rows still cover every bank
#ifndef NW
#define NW    4
#endif
#define NTHR (NW * 32)
#define NTILE 10                // upper-triangular tiles of a 4x4 tiling
#define MAXT  ((NTILE + NW - 1) / NW)

#ifndef LDSBAR
#define LDSBAR 1
#endif
// ABL: ablation levels, WRONG ON PURPOSE except at 0 -- they exist to price the parts.
//   1 = stop after the gram (store the scaled, masked A instead of its inverse)
//   2 = skip the diagonal substitution (identity blocks), keep both merges
//   3 = skip both merges, keep the substitution
#ifndef ABL
#define ABL 0
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

// LDS is what limits how many of these run per CU, and this kernel spends most of its time waiting
// on memory -- so a buffer is worth keeping only if it buys more than the occupancy it costs. The
// inverse's transposed form does not: it is needed as the right operand of three fragments per
// head, and fragT rebuilds those from the row-major copy for eight scalar reads each.
// ... and by the same argument the inverse does not need a buffer of its own either. Each stage
// reads the block of A it is about to overwrite BEFORE writing it, and the stages are separated by
// workgroup barriers, so the inverse can be formed in place:
//   diagonal p    reads A[p][p], writes it after the substitution has finished reading
//   32-merge      reads A10 and the two diagonal inverses, writes A10
//   64-merge      reads R = A[32:,:32] into registers everywhere, barriers, then writes it
// That leaves one 64x64 buffer plus a 32x32 scratch: 11 KB, where the first version wanted 31.
struct KkSmem {
  unsigned short A[BT][AP];     // A on the way in, its inverse on the way out
  unsigned short TMP[32][36];   // one 32x32 intermediate, transposed
  float gb[BT], bt[BT];         // the chunk's gate cumsum and beta, per token
};

// The (mt, nt) of the NTILE upper-triangular tiles, walked out arithmetically rather than read
// from a __constant__ table: the table form is one byte load per tile from a segment this kernel
// is otherwise never touching, and it is wave-uniform arithmetic either way.
__device__ __forceinline__ void kkt_tile(int ti, int* mt, int* nt) {
  int m = 0, n = ti;
  while (n >= NBK - m) { n -= (NBK - m); ++m; }
  *mt = m; *nt = n + m;
}

// OCC: minimum waves per SIMD to compile for. This kernel is memory wait, not arithmetic
// (~2000 instructions per wave against its wall clock), so occupancy is its lever -- and at
// HG_RATIO=3 it lands on 190 VGPRs, which caps it at 8 waves/SIMD where its 11.5 KB of LDS
// would allow 10. Asking for 10 costs spills; whether that trade pays is measured, not assumed.
// Slower: OCC=10 gets there (144 VGPRs) but spills 148 bytes per lane and runs 39.9 us
// against 25.4. Left at 0.
#ifndef OCC
#define OCC 0
#endif
template <int HG_RATIO>
__global__
#if OCC
__launch_bounds__(NTHR, OCC)
#else
__launch_bounds__(NTHR)
#endif
void r4d_gdn_kkt_solve_kernel(const unsigned short* __restrict__ k,
                              const float* __restrict__ beta,
                              const float* __restrict__ g,
                              unsigned short* __restrict__ Ao,
                              const int* __restrict__ cu,
                              int N, int H, int Hg)
{
  __shared__ KkSmem s;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int lo = lane & 15, hi = lane >> 4;
  const int hg = blockIdx.y;

  // Resolve the flat chunk id to (sequence, chunk). The grid is launched with an upper bound on
  // the total chunk count, so this also decides which workgroups have nothing to do -- it costs
  // N+1 scalar loads and saves the host a device-to-host sync it would otherwise need to size the
  // grid exactly.
  int n = -1, t0 = 0, bos = 0, rows = 0;
  {
    int acc = 0, want = blockIdx.x;
    for (int i = 0; i < N; ++i) {
      const int b = cu[i], e = cu[i + 1];
      const int nc = (e - b + BT - 1) / BT;
      if (want < acc + nc) { n = i; bos = b; t0 = (want - acc) * BT;
                             rows = min(BT, e - b - t0); break; }
      acc += nc;
    }
    if (n < 0) return;
  }

  // ---- zero the parts no store ever reaches: the upper tiles of A, and everything in M / MT
  // outside the blocks the inverse writes. Done once; the write pattern is the same for every
  // head, so it cannot be dirtied between heads.
  // 16 bytes per thread per step, not 2: as shorts this is 34 ds_store_b16 per thread for a
  // buffer that is 8704 bytes and 16-byte aligned.
  for (int i = tid; i < BT * AP / 8; i += NTHR) ((uint4*)&s.A[0][0])[i] = make_uint4(0, 0, 0, 0);

  // ---- the gram, once for all HG_RATIO v heads of this k head -----------------------------
  const unsigned short* kp = k + ((size_t)(bos + t0) * Hg + hg) * KD;
  const unsigned krow = (unsigned)(Hg * KD);
  v8f P[MAXT];
#pragma unroll
  for (int t = 0; t < MAXT; ++t) P[t] = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int t = 0; t < MAXT; ++t) {
    const int ti = warp + t * NW;
    if (ti < NTILE) {
      int mt, nt; kkt_tile(ti, &mt, &nt);
      // One 32-bit element offset per operand row, hoisted out of the k loop. The pointer form
      // (fragG) rebuilds v_mad_co_i64_i32 + v_lshlrev_b64 + a 64-bit add for EVERY one of the
      // sixteen fragments, because the size_t row stride forces 64-bit arithmetic and LLVM will
      // not keep the address live across the loop.
      const unsigned om = rowGo(krow, mt * 16, rows - 1, lane);
      const unsigned on = rowGo(krow, nt * 16, rows - 1, lane);
#pragma unroll
      for (int sk = 0; sk < KD / 16; ++sk)
        P[t] = mma(fragGo(kp, om + sk * 16), fragGo(kp, on + sk * 16), P[t]);
    }
  }

  for (int hh = 0; hh < HG_RATIO; ++hh) {
    const int hv = hg * HG_RATIO + hh;

    // ---- per-head gate and beta for this chunk
    if (tid < BT) {
      const int t = min(tid, rows - 1);
      s.gb[tid] = g[(size_t)(bos + t0 + t) * H + hv];
      s.bt[tid] = beta[(size_t)(bos + t0 + t) * H + hv];
    }
    BAR();

    // ---- scale, mask and place the gram as the strictly lower half of A ---------------------
    // Tile (mt, nt) holds P[m][n]; the element written is A[n][m] = rs[n] gv[m] P[m][n], which is
    // in the lower triangle exactly when n > m. n is fixed per lane and m runs over the eight
    // accumulator elements, so this is one row of A and one 16-byte store.
#pragma unroll
    for (int t = 0; t < MAXT; ++t) {
      const int ti = warp + t * NW;
      if (ti < NTILE) {
        int mt, nt; kkt_tile(ti, &mt, &nt);
        const int i = nt * 16 + lo;                      // row of A
        const int j0 = mt * 16 + 8 * hi;                 // first column of A
        const bool live = (i < rows);
        const float bi = live ? s.bt[i] : 0.0f;
        const float gi = live ? s.gb[i] : 0.0f;
        float vv[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int j = j0 + e;
          // i > j means g_i <= g_j, so this exponent is never positive whatever the span is --
          // but ONLY for a live row. On a padding row (i >= rows) gi is forced to 0 above while
          // gb[j] keeps its real, negative cumsum, so gi - gb[j] = -gb[j] is large POSITIVE,
          // __expf overflows to +INF, and bi * INF = 0 * INF = NaN. Those NaNs sit in padding
          // rows of the 64x64 tile, and the blocked inverse below merges the whole tile with
          // WMMA, so they reach live rows and poison the chunk's output for the head.
          // Measured on Qwen3.8-27B at TP2: 16 non-finite values in A, which chunk_scan then
          // spreads to one entire head_v-wide head of the layer output.
          const float d = (live && i > j && j < rows) ? (gi - s.gb[j]) : -INFINITY;
          vv[e] = bi * __expf(d) * P[t][e];
        }
        *(uint2*)(&s.A[i][j0])     = f2bf4(vv[0], vv[1], vv[2], vv[3]);
        *(uint2*)(&s.A[i][j0 + 4]) = f2bf4(vv[4], vv[5], vv[6], vv[7]);
      }
    }
    BAR();

#if ABL == 1
    {
      unsigned short* op = Ao + ((size_t)(bos + t0) * H + hv) * BT;
      const size_t orow = (size_t)H * BT;
      for (int r = tid / 8; r < rows; r += NTHR / 8)
        *(uint4*)(op + (size_t)r * orow + (tid & 7) * 8) = *(const uint4*)(&s.A[r][(tid & 7) * 8]);
    }
    if (hh + 1 < HG_RATIO) BAR();
    continue;
#endif
    // ---- 1. the four diagonal blocks, by fp32 forward substitution --------------------------
    // D = (I + N)^-1 solves D = I - N D one row at a time. Lane l owns column l of D, so the
    // dot product over j < i is 16 registers deep and N[i][j] is a broadcast LDS read.
    for (int p = warp; p < NBK; p += NW) {
      const int b0 = p * 16;
      float d[16];
#if ABL == 2
#pragma unroll
      for (int i = 0; i < 16; ++i) d[i] = (lo == i) ? 1.0f : 0.0f;
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = f2bf(d[i]);
        *(uint2*)(&s.MT[b0 + lo][b0])      = f2bf4(d[0], d[1], d[2], d[3]);
        *(uint2*)(&s.MT[b0 + lo][b0 + 4])  = f2bf4(d[4], d[5], d[6], d[7]);
        *(uint2*)(&s.MT[b0 + lo][b0 + 8])  = f2bf4(d[8], d[9], d[10], d[11]);
        *(uint2*)(&s.MT[b0 + lo][b0 + 12]) = f2bf4(d[12], d[13], d[14], d[15]);
      }
      continue;
#endif
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        float acc = 0.0f;
#pragma unroll
        for (int j = 0; j < 16; ++j)
          if (j < i) acc += bf2f(s.A[b0 + i][b0 + j]) * d[j];
        d[i] = ((lo == i) ? 1.0f : 0.0f) - acc;
      }
      if (lane < 16) {
#pragma unroll
        for (int i = 0; i < 16; ++i) s.A[b0 + i][b0 + lo] = f2bf(d[i]);
      }
    }
    BAR();

    // ---- 2. the two 32-merges: M10 = -D1 A10 D0 --------------------------------------------
#if ABL != 3
    for (int blk = warp; blk < 2; blk += NW) {
      const int r0 = blk * 32;                            // 32-block origin
      // X = A10 . D0     (left A10 row-major, right D0 transposed = MT)
      v8f X = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      X = mma(frag(&s.A[0][0], AP, r0 + 16, r0, lane),
              fragT(&s.A[0][0], AP, r0, r0, lane), X);
      // stage X transposed for the next product. The two merges run on two waves against the
      // same scratch, so each owns 16 rows of it.
      *(uint2*)(&s.TMP[blk * 16 + lo][8 * hi])     = f2bf4(X[0], X[1], X[2], X[3]);
      *(uint2*)(&s.TMP[blk * 16 + lo][8 * hi + 4]) = f2bf4(X[4], X[5], X[6], X[7]);
      // fragT reads M column-wise below, so the merge results only need the row-major copy
      __builtin_amdgcn_wave_barrier();
      // M10 = -D1 . X    (left D1 row-major = M, right X transposed = TMP)
      v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
      Y = mma(frag(&s.A[0][0], AP, r0 + 16, r0 + 16, lane),
              frag(&s.TMP[0][0], 36, blk * 16, 0, lane), Y);
      const int i = r0 + 16 + 8 * hi;                     // rows of M10
      const int j = r0 + lo;                              // column of M10
#pragma unroll
      for (int e = 0; e < 8; ++e) s.A[i + e][j] = f2bf(-Y[e]);
    }
#endif
    BAR();

#if ABL != 3
    // ---- 3. the 64-merge: M[32:,:32] = -Q^-1 (R P^-1) --------------------------------------
    // Y = R . P^-1, R = A[32:,:32] row-major, P^-1 transposed = MT[:32,:32]
    for (int t = warp; t < 4; t += NW) {
      const int mt = t >> 1, nt = t & 1;
      v8f Y = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Y = mma(frag(&s.A[0][0], AP, 32 + mt * 16, sk * 16, lane),
                fragT(&s.A[0][0], AP, nt * 16, sk * 16, lane), Y);
      *(uint2*)(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi])     = f2bf4(Y[0], Y[1], Y[2], Y[3]);
      *(uint2*)(&s.TMP[nt * 16 + lo][mt * 16 + 8 * hi + 4]) = f2bf4(Y[4], Y[5], Y[6], Y[7]);
    }
    BAR();                          // every wave has read R into TMP; now it may be overwritten
    // Z = -Q^-1 . Y, Q^-1 = A[32:,32:] row-major, Y transposed = TMP
    for (int t = warp; t < 4; t += NW) {
      const int mt = t >> 1, nt = t & 1;
      v8f Z = (v8f){0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int sk = 0; sk < 2; ++sk)
        Z = mma(frag(&s.A[0][0], AP, 32 + mt * 16, 32 + sk * 16, lane),
                frag(&s.TMP[0][0], 36, nt * 16, sk * 16, lane), Z);
      const int i = 32 + mt * 16 + 8 * hi;
      const int j = nt * 16 + lo;
#pragma unroll
      for (int e = 0; e < 8; ++e) s.A[i + e][j] = f2bf(-Z[e]);
    }
#endif
    BAR();

    // ---- store: one row of the inverse is 64 bf16, contiguous in HBM -----------------------
    {
      unsigned short* op = Ao + ((size_t)(bos + t0) * H + hv) * BT;
      const size_t orow = (size_t)H * BT;
      for (int r = tid / 8; r < rows; r += NTHR / 8) {
        const int c8 = (tid & 7) * 8;
        *(uint4*)(op + (size_t)r * orow + c8) = *(const uint4*)(&s.A[r][c8]);
      }
    }
    if (hh + 1 < HG_RATIO) BAR();
  }
}

extern "C" int r4d_gdn_kkt_solve_k128_c64_bf16(const void* k, const void* beta, const void* g,
                                               void* A, const void* cu, int N, int T, int H,
                                               int Hg, int K, int bt, void* stream)
{
  if (K != KD || bt != BT) return -1;
  if (H % Hg) return -2;
  const int nchunk_ub = (T + BT - 1) / BT + N;   // sum of per-sequence chunk counts, bounded
  dim3 grid(nchunk_ub, Hg), blk(NTHR);
#define KK_LAUNCH(R)                                                                          \
  hipLaunchKernelGGL((r4d_gdn_kkt_solve_kernel<R>), grid, blk, 0, (hipStream_t)stream,        \
                     (const unsigned short*)k, (const float*)beta, (const float*)g,           \
                     (unsigned short*)A, (const int*)cu, N, H, Hg)
  switch (H / Hg) {
    case 1: KK_LAUNCH(1); break;
    case 2: KK_LAUNCH(2); break;
    case 3: KK_LAUNCH(3); break;
    case 4: KK_LAUNCH(4); break;
    case 6: KK_LAUNCH(6); break;
    case 8: KK_LAUNCH(8); break;
    default: return -3;
  }
#undef KK_LAUNCH
  return (int)hipGetLastError();
}
