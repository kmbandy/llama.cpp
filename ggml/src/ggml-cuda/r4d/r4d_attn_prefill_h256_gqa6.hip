// R4D prefill. Transposed core: S^T = K.Q^T, one lane owns one query row, ROWS = 16*NWARPS,
// 128-VGPR accumulator. The query is bf16 and the KV cache is fp8 or bf16; the 16-bit operand
// format for both matmuls is f16 (see r4d_dt16.h for why).
//
// Everything that measured far off the optimum was deleted rather than left behind a flag: the
// two-term-e4m3 query and P legs, the single-term fp8 P leg (16% faster, 15x less accurate), the
// paired 8-bit LDS read that only those needed, the S-split, and a layout permutation that CONTIG
// made redundant. What is left is the two families that ship: all-16-bit (bf16 KV) and 16/8 mixed
// (bf16 query, fp8 KV).
//
// The surviving options. Each is a measured step and each stays separable, so any one of them can
// be ablated again after a geometry change:
//
//   LAZY  (always on, and the reason this file exists)
//         The obvious form rescales all 128 accumulator VGPRs by alpha on EVERY m-tile. Here the
//         softmax runs against a REFERENCE max instead of a running one: p = 2^(s - m_ref + SHIFT)
//         with m_ref updated -- and the accumulator rescaled -- only when some row's max exceeds
//         m_ref by more than the P format can absorb (DT16::GROW octaves). Accuracy is unchanged:
//         eager rescaling by alpha and lazy accumulation at a shifted scale produce the SAME
//         relative weighting in an f32 accumulator.
//   F16   f16 rather than bf16 as the 16-bit operand format. gfx1201 has NO bf16 convert, so an
//         f32->bf16 is ~6 VALU of software RTNE; v_cvt_pkrtz_f16_f32 does two f32->f16 in ONE.
//         Both are exact for these operands (fp8 has 3 mantissa bits, bf16 has 7, f16 has 10).
//   MSKIP Causal masking applied only on m-tiles that straddle the limit. Interior tiles pay
//         nothing.
//   FOLDQ scale * k_descale * log2(e) folded into the query ONCE at load.
//   CONTIG The k axis of a WMMA is only ever CONTRACTED, so relabelling it on BOTH operands is
//         free; reading eight contiguous elements at 8*h gives one aligned ds_read_b128 per
//         fragment, deletes 16 address adds, and makes the P fragment lane-local.
//   PF/SGB Prefetch PF fragments before the matching PF WMMAs, with a hard scheduling barrier
//         between the two groups. Without the barrier PF is INERT.
//   DOT2  v_dot2_f32_f16 row sum off the packed f16 P: 4 instructions for 8, and the denominator
//         becomes the p values the PV matmul actually used.
//   BTS/KWIDE/GPRE  the tile-staging set -- see the comments on each define.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#define O_F16     1
#define O_MSKIP   2
// QK8/PV8: the deleted 8-bit arms, rebuilt as SEPARABLE opt-in bits for long-context serving.
// The dispatch header records why they were deleted: single-term e4m3 Q measured 2.58e-2 against
// the fp32 oracle vs 2.3e-3 for the f16 path. That is a KERNEL-level verdict; these bits exist to
// let the END-TASK verdict (ppl + GSM8K, the fp8 all-reduce playbook) be measured on a build
// where each leg can be flipped alone. Both require the fp8 KV cache (KVP == 0).
//   QK8: Q quantized e4m3 once at load (the fold makes it ~0.1-0.5, well inside e4m3), K stays
//        RAW BYTES in LDS -- the fp8->f16 upconvert in storeK disappears, sK halves, and the QK
//        WMMA runs at the 2x fp8 issue rate.
//   PV8: P packed e4m3 instead of f16 (v_cvt_pk_fp8_f32, same instruction count as pkrtz), V
//        stays raw bytes, PV WMMA at 2x. GROW tightens to 8 octaves (e4m3 max 448 = 2^8.8) and
//        DOT2 is off -- the denominator sums the pre-quantization p, which the old probe measured
//        indistinguishable at these error levels.
#define O_QK8    32
#define O_PV8    64
// A K prefetch across the m-tile loop (the GPREV pattern applied to K) was re-measured under the
// 8-bit legs on 2026-08-29, on the theory that the register headroom they free (qf8 is 32 VGPRs
// against qf's 64) removes the spill that killed upstream's three attempts. It still LOSES:
// -5% at an 8k chunk, -1% at 40k, bit-identical output -- the live range across the loop is the
// cost, not the spill. Do not build it a fifth time.
#define O_FOLDQ   8
#define O_CONTIG  512
#define O_PF2    1024
#define O_PF4    2048
#define O_PF8    3072
#define O_SGB    4096
#define O_DOT2  16384
// BTS: the block table entries for a tile are TILE/BS values at a WAVE-UNIFORM index, so they
// belong in SGPRs. Without this, every staging pass is `global_load_b32` (block table),
// `s_wait_loadcnt 0x0`, fourteen instructions of address arithmetic, `global_load_b64` (the KV),
// `s_wait_loadcnt 0x0` -- TWO serialised memory round trips, inside a barrier-bounded phase with
// nothing else to run. A scalar load hoisted to the top of the tile collapses that to one.
#define O_BTS   262144
// KWIDE: one 16-byte K fetch per thread instead of two 8-byte ones. 768 threads x 16 B is exactly
// a 48-key tile, so the staging loop becomes straight-line: half the global loads, half the block
// table lookups, half the dependency chains.
// The naive assignment (key = tid/16) makes a warp write ONE row at a 32-byte stride, which is a
// 4-way LDS bank conflict: bank = 8*ch mod 32 takes only four values. Interleaving two keys across
// the warp instead -- key = 2*warp + (lane&1), ch = (lane&31)>>1 -- gives eight distinct 4-bank
// groups covering all 32 banks, i.e. the conflict-free minimum, and the global side still reads two
// contiguous 256-byte runs.
#define O_KWIDE 524288
// GPREV: fetch tile ti+1's V into REGISTERS immediately after the second barrier, so its memory
// latency is covered by tile ti's 96 WMMAs and only the LDS stores stay between the barriers.
// V ONLY. The same prefetch for K was built three ways -- as its own bit, issued at the tail of
// the body so its live range missed the m-tile loop, and as a register-free scalar
// `s_prefetch_data` -- and all three LOST. At 240 VGPRs the allocator spills a second prefetch,
// and a spilled prefetch is waited on immediately, which is not a prefetch at all.
#define O_GPREV 1048576
// LDSB: an LDS-scoped barrier instead of __syncthreads(). See lds_barrier() in r4d_common.h --
// the default fence invalidates the vector cache twice per tile for a synchronisation that only
// ever orders LDS.
#define O_LDSB  2097152

typedef int v2i32_r4d __attribute__((ext_vector_type(2)));   // one fp8 WMMA operand (8 e4m3)
// pack 4 floats -> 4 e4m3 bytes in element order (RNE, saturating), 2 instructions
__device__ __forceinline__ uint32_t pk_fp8x4(float a, float b, float c, float d) {
    const uint32_t w = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0u, false);
    return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, w, true);
}

// bf16 dword -> 16-bit dword in the target format, optionally scaled. Used once per kernel on Q.
template<class D, int FOLDQ, int F16>
__device__ __forceinline__ uint32_t r4d_qcvt(uint32_t w, float s) {
    if (!FOLDQ) return F16 ? D::from_bf16w(w) : w;
    return D::pk(__builtin_bit_cast(float, w << 16) * s,
                 __builtin_bit_cast(float, w & 0xffff0000u) * s);
}

template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_prefill_kernel(const R4DArgs a)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16) && !(OPT & O_PV8)) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)   ? 1 : 0;
    constexpr int GPREV = (OPT & O_GPREV) ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB)  ? 1 : 0;
    constexpr int KWIDE = (OPT & O_KWIDE) ? 1 : 0;
    constexpr int QK8   = ((OPT & O_QK8) && !KVP) ? 1 : 0;
    constexpr int PV8   = ((OPT & O_PV8) && !KVP) ? 1 : 0;
    static_assert(!(OPT & (O_QK8 | O_PV8)) || !KVP, "8-bit legs need the fp8 KV cache");
    static_assert(!(OPT & O_QK8) || KWIDE, "QK8 raw-byte staging is written for the wide K fetch");
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int ROWS    = NWARPS * 16;
    constexpr int BLOCK_Q = ROWS / GQA;
    constexpr int NKS     = HEAD_DIM / 16;
    constexpr int MT      = TILE / 16;
    constexpr int KSTR    = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR    = TILE + (CTG ? 8 : 4);
    constexpr int NTHR    = NWARPS * 32;
    constexpr int NB      = TILE / BS;                       // block-table entries per tile
    constexpr int KEL     = KWIDE ? 16 : 8;                  // K elements fetched per thread-pass
    constexpr int KITEMS  = TILE * (HEAD_DIM / KEL);          // K passes needed for one tile
    constexpr int KPASS   = (KITEMS + NTHR - 1) / NTHR;        // passes per thread
    constexpr int KREGS   = KPASS * (KVP ? 2 : 1);             // uint4 of raw K per thread
    constexpr int NTR     = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS   = (NTR + NWARPS - 1) / NWARPS;        // uint4 of raw V per thread
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = PV8 ? 8.0f : D::GROW;   // e4m3 max is 448 = 2^8.8
    // KPASS was pinned to 1 when the wide fetch was written for the 48-key tile (768 threads x
    // 16 B is exactly that tile); the fetch/store loops always iterated KPASS and the index map
    // extends verbatim, so larger tiles just take more passes. Bound it to keep kreg honest.
    static_assert(!KWIDE || KPASS <= 2, "the wide K fetch covers at most two passes");
    static_assert((!QK8 && !PV8) || CTG, "the 8-bit fragment maps are written for CONTIG");

    // The K and V tiles are BYTES under the 8-bit legs and the allocation is sized accordingly
    // -- this is what unlocks TILE > 48 (bf16 tiles at TILE 64 need 68 KB > the 64 KB limit;
    // byte tiles need 35 KB, and TILE 96 still fits at 52 KB), and at TILE 48 it takes the
    // fp8-leg LDS from 54 KB to 27 KB, i.e. TWO resident workgroups per WGP instead of one.
    // Strides are padded 8 (bytes or shorts alike) so the 16-row fragment reads spread across
    // all 32 banks.
    constexpr int KSTR8 = HEAD_DIM + 8, VSTR8 = TILE + 8;
    __shared__ uint16_t sK[QK8 ? (TILE * KSTR8 + 1) / 2 : TILE * KSTR];
    __shared__ uint16_t sV16[PV8 ? (HEAD_DIM * VSTR8 + 1) / 2 : HEAD_DIM * VSTR];
    uint8_t* const sK8 = (uint8_t*)sK;
    uint8_t* const sV8 = (uint8_t*)sV16;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;

    const int qb = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const int qpos = qb * BLOCK_Q + qi;
    const bool live = (qpos < a.q_len);
    const int qrow = live ? qpos : (a.q_len - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;

    const float kdesc = a.k_descale ? a.k_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float vdesc = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    frag16 qf[NKS];
    v2i32_r4d qf8[QK8 ? NKS : 1];
    {
        const uint16_t* qp = (const uint16_t*)a.q
                           + ((size_t)(seq * a.q_len + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            // CONTIG reads the same 8 d as ONE run; the legacy map splits them 4+4.
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            if constexpr (QK8) {
                // e4m3 Q, folded scale first: the fold puts |q| in ~[0.05, 1], comfortably inside
                // e4m3, and its error is relative so the fold costs nothing extra.
                const float m_ = FOLDQ ? mul : 1.0f;
                #define R4D_QB(w) __builtin_bit_cast(float, (w) << 16) * m_, \
                                  __builtin_bit_cast(float, (w) & 0xffff0000u) * m_
                qf8[t] = (v2i32_r4d){(int)pk_fp8x4(R4D_QB(l0.x), R4D_QB(l0.y)),
                                     (int)pk_fp8x4(R4D_QB(h0.x), R4D_QB(h0.y))};
                #undef R4D_QB
            } else {
                qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                         r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                              make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                         r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
            }
        }
    }

    const int klimit = ctx - a.q_len + qpos;

    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int qmax_cta   = min(qb * BLOCK_Q + BLOCK_Q - 1, a.q_len - 1);
    const int klimit_cta = ctx - a.q_len + qmax_cta;
    const int ntiles = (min(klimit_cta + 1, ctx) + TILE - 1) / TILE;

    // ---- staging, split into a global FETCH (registers) and an LDS STORE ------------------------
    // Splitting them is what lets GPRE move the fetch out from between the barriers; without GPRE
    // the two run back to back and the generated code is the same shape as it was before GPRE.
    const int vg = lane >> 3, vj = lane & 7;
    uint4 kreg[KREGS], vreg[VREGS];

    auto fetchK = [&](int k0, const int* blks) {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                const int kk  = min(k0 + key, ctx - 1);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM);
                if (KVP) {
                    const uint16_t* g = (const uint16_t*)a.kv + o + ch * KEL;
                    #pragma unroll
                    for (int q = 0; q < (KVP ? (KEL / 8) : 1); ++q)
                        kreg[p * (KEL / 8) + q] = *(const uint4*)(g + 8 * q);
                } else if (KWIDE) {
                    kreg[p] = *(const uint4*)((const uint8_t*)a.kv + o + ch * 16);
                } else {
                    const uint2 r = *(const uint2*)((const uint8_t*)a.kv + o + ch * 8);
                    kreg[p] = make_uint4(r.x, r.y, 0, 0);
                }
            }
        }
    };

    auto fetchV = [&](int k0, const int* blks) {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int kk = min(k0 + kg * 8 + vj, ctx - 1);
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride
                               + (size_t)(kk % BS) * (2 * HEAD_DIM) + HEAD_DIM + d0;
                const void* p = KVP ? (const void*)((const uint16_t*)a.kv + o)
                                    : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeK = [&]() {
        #pragma unroll
        for (int p = 0; p < KPASS; ++p) {
            const int i = tid + p * NTHR;
            if (KITEMS % NTHR == 0 || i < KITEMS) {
                const int key = KWIDE ? (2 * (i >> 5) + (i & 1)) : (i / (HEAD_DIM / KEL));
                const int ch  = KWIDE ? ((i & 31) >> 1) : (i - key * (HEAD_DIM / KEL));
                if (KWIDE && key >= TILE) continue;
                if constexpr (QK8) {
                    // K stays e4m3: no upconvert, one b128 store, sK footprint halves.
                    *(uint4*)(&sK8[key * KSTR8 + ch * KEL]) = kreg[p];
                    continue;
                }
                uint16_t* kd = &sK[key * KSTR + ch * KEL];
                #pragma unroll
                for (int q = 0; q < KEL / 8; ++q) {
                    uint32_t w4[4];
                    if (KVP) {
                        const uint4 r = kreg[p * (KEL / 8) + q];
                        w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                        w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                    } else {
                        const uint4 r = kreg[p];
                        fp8x8_to_16x4w<D>(q ? make_uint2(r.z, r.w) : make_uint2(r.x, r.y), w4);
                    }
                    *(uint2*)(kd + 8 * q)     = make_uint2(w4[0], w4[1]);
                    *(uint2*)(kd + 8 * q + 4) = make_uint2(w4[2], w4[3]);
                }
            }
        }
    };

    auto storeV = [&]() {
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int d0 = dg * (KVP ? 32 : 64) + vg * (KVP ? 8 : 16);
                const uint4 v = vreg[r];
                if (KVP) {
                    uint16_t* dst = &sV16[(d0 + vj) * VSTR + kg * 8];
                    *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
                    *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
                } else {
                    // tr_b128 transposes 16-BIT lanes, so each transposed short is two fp8 from
                    // ADJACENT d; v_perm_b32 splits them back into two d rows.
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
                    if constexpr (PV8) {
                        *(uint2*)(&sV8[(d0 + 2 * vj)     * VSTR8 + kg * 8]) = make_uint2(lo0, lo1);
                        *(uint2*)(&sV8[(d0 + 2 * vj + 1) * VSTR8 + kg * 8]) = make_uint2(hi0, hi1);
                        continue;
                    }
                    uint32_t wl[4], wh[4];
                    fp8x8_to_16x4w<D>(make_uint2(lo0, lo1), wl);
                    fp8x8_to_16x4w<D>(make_uint2(hi0, hi1), wh);
                    uint16_t* a0 = &sV16[(d0 + 2 * vj)     * VSTR + kg * 8];
                    uint16_t* a1 = &sV16[(d0 + 2 * vj + 1) * VSTR + kg * 8];
                    *(uint2*)(a0)     = make_uint2(wl[0], wl[1]);
                    *(uint2*)(a0 + 4) = make_uint2(wl[2], wl[3]);
                    *(uint2*)(a1)     = make_uint2(wh[0], wh[1]);
                    *(uint2*)(a1 + 4) = make_uint2(wh[2], wh[3]);
                }
            }
        }
    };

    // The block table for tile ti is fetched during tile ti-1, so the scalar load has an entire
    // store phase and two barriers to complete instead of stalling the KV address chain.
    int blk[NB];
    if (BTS) {
        #pragma unroll
        for (int i = 0; i < NB; ++i) blk[i] = bt[i];
    }
    if (GPREV) fetchV(0, blk);

    for (int ti = 0; ti < ntiles; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < ntiles) {
            const int tb = (ti + 1) * NB;
            #pragma unroll
            for (int i = 0; i < NB; ++i) blkn[i] = bt[tb + i];
        }
        if (LDSB) lds_barrier(); else __syncthreads();
        fetchK(k0, blk);
        if (!GPREV) fetchV(k0, blk);
        storeV();              // V first: its data is already in registers, so it covers K's wait
        storeK();
        if (LDSB) lds_barrier(); else __syncthreads();
        // The next tile's global loads are issued HERE, before the 96 WMMAs below, and the wait
        // that consumes them does not appear until after the next barrier.
        if (GPREV && ti + 1 < ntiles) fetchV(k0 + TILE, blkn);

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            if constexpr (QK8) {
                // Same PF discipline, half the fragment bytes, 2x the WMMA issue rate.
                const uint8_t* krow8 = &sK8[(mt * 16 + c) * KSTR8] + 8 * h;
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    v2i32_r4d kb8[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        kb8[j] = *(const v2i32_r4d*)(krow8 + 16 * (g + j));
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        s = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(kb8[j], qf8[g + j], s);
                }
            } else {
                const uint16_t* krow = &sK[(mt * 16 + c) * KSTR] + (CTG ? 8 * h : 0);
                // PF fragments are LOADED before the matching PF WMMAs are ISSUED, so the wait in
                // front of each WMMA becomes `s_wait_dscnt PF-1` instead of a full drain.
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 kb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* kp = krow + 16 * (g + j);
                        if (CTG) {
                            const uint4 u = *(const uint4*)kp;
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h),
                                          *(const uint2*)(kp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
                }
            }
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;

            const int kbase = k0 + mt * 16 + 8 * h;
            if (!MSKIP || wave_any(kbase + 7 > klimit)) {
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] = (kbase + e <= klimit) ? sc[e] : -INFINITY;
            }

            float smax = sc[0];
            #pragma unroll
            for (int e = 1; e < 8; ++e) smax = fmaxf(smax, sc[e]);
            if (wave_any(smax > m_ref + PGROW)) {
                const float rmax  = fmaxf(smax, swap16(smax));
                const float mnew  = fmaxf(fmaxf(m_ref, rmax), -3.0e38f);
                const float alpha = ex2(m_ref - mnew);
                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= alpha;
                l_i *= alpha;
                m_ref = mnew;
                off   = PSHIFT - mnew;
            }

            float pf[8], lsum = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const float p = ex2(sc[e] + off);
                pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;

            if constexpr (PV8) {
                const v2i32_r4d p8 = (v2i32_r4d){(int)pk_fp8x4(pf[0], pf[1], pf[2], pf[3]),
                                                 (int)pk_fp8x4(pf[4], pf[5], pf[6], pf[7])};
                const uint8_t* vbase8 = &sV8[c * VSTR8 + mt * 16 + 8 * h];
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    v2i32_r4d vb8[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        vb8[j] = *(const v2i32_r4d*)(vbase8 + (size_t)(g + j) * 16 * VSTR8);
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j)
                        acc[g + j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(vb8[j], p8, acc[g + j]);
                }
            } else {
                uint32_t pw[4];
                #pragma unroll
                for (int e = 0; e < 8; e += 2) pw[e >> 1] = D::pk(pf[e], pf[e + 1]);
                if (DOT2) {
                    const v2hf one = {(__fp16)1.0f, (__fp16)1.0f};
                    #pragma unroll
                    for (int e = 0; e < 4; ++e)
                        l_i = __builtin_amdgcn_fdot2(__builtin_bit_cast(v2hf, pw[e]), one, l_i, false);
                }
                const uint2 lo  = make_uint2(pw[0], pw[1]);
                const uint2 hi2 = make_uint2(pw[2], pw[3]);
                // QK leaves lane (c,h) holding keys 8h+e. Under CONTIG that IS the B fragment, so
                // the cross-half exchange the two-runs-of-four map needed is simply not there.
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);
                } else {
                    const uint2 y = swap16_u2p((h == 0) ? hi2 : lo);
                    p16 = (h == 0) ? D::mk(lo, y) : D::mk(y, hi2);
                }
                const uint16_t* vbase = &sV16[c * VSTR + mt * 16] + (CTG ? 8 * h : 0);
                #pragma unroll
                for (int g = 0; g < NKS; g += PF) {
                    frag16 vb[PF];
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) {
                        const uint16_t* vp = vbase + (g + j) * 16 * VSTR;
                        if (CTG) {
                            const uint4 u = *(const uint4*)vp;
                            vb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            vb[j] = D::mk(*(const uint2*)(vp + 4 * h),
                                          *(const uint2*)(vp + 8 + 4 * h));
                        }
                    }
                    if (SGB) sched_barrier();
                    #pragma unroll
                    for (int j = 0; j < PF; ++j) acc[g + j] = D::wmma(vb[j], p16, acc[g + j]);
                }
            }
        }
        if (BTS) {
            #pragma unroll
            for (int i = 0; i < NB; ++i) blk[i] = blkn[i];
        }
    }

    if (!live) return;
    l_i += swap16(l_i);
    const float inv = (l_i > 0.0f) ? (vdesc / l_i) : 0.0f;
    uint16_t* op = (uint16_t*)a.out
                 + ((size_t)(seq * a.q_len + qpos) * a.q_heads + qhead) * HEAD_DIM;
    #pragma unroll
    for (int dt = 0; dt < NKS; ++dt) {
        uint32_t tw[4];
        #pragma unroll
        for (int e = 0; e < 8; e += 2)
            tw[e >> 1] = (uint32_t)f32_to_bf16(acc[dt][e] * inv)
                       | ((uint32_t)f32_to_bf16(acc[dt][e + 1] * inv) << 16);
        *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
    }
}
