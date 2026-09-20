// R4D decode. The prefill core (lazy reference-max softmax, f16 operands, folded query scale,
// contiguous-k fragments, prefetch + scheduling barrier, dot2 row sum) on the split-KV MTP-verify
// shape, plus two decode-only changes:
//
//   KLDS0  reads K straight from the paged cache, including when that cache is fp8. The
//          widening moves from once-per-workgroup to once-per-warp; it costs VALU on a leg that is
//          bandwidth-bound and buys back the whole sK allocation, which at TILE 16 is 8.3 KB of
//          the 14.5 KB that sets resident-CTA count. This is the argmin at every depth.
//   PF16   split-KV partials as f16 instead of f32. The partial buffer is written once and read
//          back once, and at 16K context that round trip is ~14 MB against ~17 MB of KV -- i.e.
//          nearly half the leg's DRAM traffic. Storing O/l rather than O bounds the value by
//          max|v|, so f16 cannot overflow and its 11-bit mantissa costs ~5e-4 relative.
//
// The fp8-P leg and the layout permutation CONTIG replaced are gone from this file too; see
// r4d_attn_prefill_h256_gqa6.hip for the option set and the reasoning behind each bit.
#include "r4d.h"
#include "r4d_common.h"
#include "r4d_dt16.h"

#define O_F16    1
#define O_MSKIP  2
#define O_FOLDQ  8
#define O_KLDS0 16
#define O_PF16  32
#define O_CONTIG 512
#define O_PF2   1024
#define O_PF4   2048
#define O_PF8   3072
#define O_SGB   4096
#define O_DOT2 16384
// The tile-staging set, shared with r4d_attn_prefill_h256_gqa6.hip -- see the comments there.
//   BTS   block table entries for the tile in SGPRs, loaded a tile ahead. On this leg it also
//         removes a dependent global load from INSIDE the m-tile loop: with KLDS0 the K address
//         comes from bt[kk/BS] on every m-tile.
//   GPREV tile ti+1's V fetched into registers before the m-tile loop.
//   LDSB  an LDS-scoped barrier, so __syncthreads() stops invalidating the vector cache.
#define O_BTS   262144
#define O_GPREV 1048576
#define O_LDSB  2097152

__host__ __device__ __forceinline__ int r4d_attn_tiles(int ctx, int tile) { return (ctx + tile - 1) / tile; }
__host__ __device__ __forceinline__ int r4d_attn_tps(int ctx, int tile, int splits) {
    int t = r4d_attn_tiles(ctx, tile); return (t + splits - 1) / splits;
}
__host__ __device__ __forceinline__ int r4d_attn_used(int ctx, int tile, int splits) {
    int t = r4d_attn_tiles(ctx, tile), tps = r4d_attn_tps(ctx, tile, splits); return (t + tps - 1) / tps;
}

template<int NWARPS, int TILE, int HEAD_DIM, int GQA, int BS, int KVP, int OPT>
__global__ __launch_bounds__(NWARPS * 32)
void r4d_attn_decode_kernel(const R4DArgs a, int splits)
{
    constexpr int F16   = (OPT & O_F16)   ? 1 : 0;
    constexpr int MSKIP = (OPT & O_MSKIP) ? 1 : 0;
    constexpr int FOLDQ = (OPT & O_FOLDQ) ? 1 : 0;
    constexpr int KLDS  = (OPT & O_KLDS0) ? 0 : 1;
    constexpr int PF16  = (OPT & O_PF16)  ? 1 : 0;
    constexpr int CTG   = (OPT & O_CONTIG) ? 1 : 0;
    constexpr int PF    = 1 << ((OPT >> 10) & 3);
    constexpr int SGB   = (OPT & O_SGB) ? 1 : 0;
    // turbo4 (KVP == 2) excludes DOT2 for the same reason prefill's PV8 does: the row-sum has to
    // be taken over UNSCALED P (the per-key V scale folds in afterwards, see the PV step below),
    // and DOT2 sums the ALREADY-PACKED p fragment that also feeds the PV matmul -- there is no
    // single packed fragment that is both the pre-fold sum input and the post-fold PV input. The
    // plain lsum accumulation (the !DOT2 path already below) has no such conflict.
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16) && KVP != 2) ? 1 : 0;
    constexpr int BTS   = (OPT & O_BTS)  ? 1 : 0;
    constexpr int LDSB  = (OPT & O_LDSB) ? 1 : 0;
    typedef DT16<F16> D;
    typedef typename D::frag frag16;

    constexpr int NKS   = HEAD_DIM / 16;
    constexpr int MT    = TILE / 16;
    constexpr int KSTR  = HEAD_DIM + (CTG ? 8 : 4);
    constexpr int VSTR  = TILE + (CTG ? 8 : 4);
    constexpr int NTHR  = NWARPS * 32;
    constexpr int NB    = TILE / BS;
    constexpr int NTR   = (TILE / 8) * (HEAD_DIM / (KVP == 1 ? 32 : 64));
    constexpr int VREGS = (NTR + NWARPS - 1) / NWARPS;
    // A V prefetch costs VREGS x 4 VGPRs and this leg already carries a 128-VGPR accumulator and a
    // 64-VGPR query. Measured: at two uint4 (fp8 KV, TILE 16) it is worth 6-8% at depth; at four
    // (fp8 TILE 32, or bf16 TILE 16, whose V is twice the bytes) it spills and LOSES 10%. So the
    // geometry decides, not the flag.
    // turbo4 (KVP == 2) never prefetches: its V item count (NTQ below) and per-thread register
    // count (VQREGS) are unrelated to VREGS <= 2, so this is a fresh geometry the measurement
    // above says nothing about -- ship the correct, unprefetched form first.
    constexpr int GPREV = ((OPT & O_GPREV) && VREGS <= 2 && KVP != 2) ? 1 : 0;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;
    // turbo4: one thread owns a 4-key x 4-d V item, same map as the prefill turbo4 fetchV/storeV
    // -- but prefill's NTHR is pinned to exactly TILE/4 * HEAD_DIM/4 threads (one item per
    // thread, no loop); decode's NWARPS varies with q_len*gqa, so here NTQ items are looped over
    // VQREGS passes per thread instead.
    constexpr int NTQ    = (TILE / 4) * (HEAD_DIM / 4);
    constexpr int VQREGS = (NTQ + NTHR - 1) / NTHR;
    static_assert(KVP != 2 || (TILE % 4 == 0 && HEAD_DIM % 4 == 0),
                  "turbo4 V load body assumes 4-key x 4-d items");
    // turbo4's K load (below) is written only for KLDS0 (K read straight from the paged cache,
    // never staged into sK) -- the only K-staging mode D_OPT actually uses. The dead `if (KLDS)`
    // branch below still has to compile for a KVP == 2 instantiation; it is not implemented for
    // turbo4 and stays unreachable behind this assert.
    static_assert(KVP != 2 || !KLDS, "turbo4 K load is only written for KLDS0");

    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];
    __shared__ uint16_t sV16[HEAD_DIM * VSTR];
    __shared__ float sKs[KVP == 2 ? TILE : 1];   // turbo4: per-key K scale for this m-tile
    __shared__ float sVs[KVP == 2 ? TILE : 1];   // turbo4: per-key V scale for this m-tile

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c = lane & 15, h = lane >> 4;

    const int sp = blockIdx.x, kvh = blockIdx.y, seq = blockIdx.z;
    const int ctx = a.seqused_k[seq];
    if (ctx <= 0) return;

    const int ntl  = r4d_attn_tiles(ctx, TILE);
    const int tps  = r4d_attn_tps(ctx, TILE, splits);
    const int t_lo = sp * tps;
    if (t_lo >= ntl) return;
    const int t_hi = min(t_lo + tps, ntl);

    const int row = warp * 16 + c;
    const int qi = row / GQA, hi = row - qi * GQA;
    const bool live = (qi < a.q_len);
    const int qrow = live ? qi : (a.q_len - 1);
    const int qhead = kvh * GQA + hi;

    const int* __restrict__ bt = a.block_table + (size_t)seq * a.max_blocks;
    const float kdesc = a.k_descale ? a.k_descale[seq * a.kv_heads + kvh] : 1.0f;
    const float mul   = a.scale * kdesc * 1.44269504089f;

    uint32_t kL[4] = {0,0,0,0}, vL[4] = {0,0,0,0};   // turbo4: 16 e4m3 centroids as 4 dwords
    if constexpr (KVP == 2) {
        const uint4 kl = *(const uint4*)a.k_lut; kL[0]=kl.x; kL[1]=kl.y; kL[2]=kl.z; kL[3]=kl.w;
        const uint4 vl = *(const uint4*)a.v_lut; vL[0]=vl.x; vL[1]=vl.y; vL[2]=vl.z; vL[3]=vl.w;
    }
    // kL/vL are consumed by fetchV/storeV and the K load below when KVP == 2; for KVP != 2 they
    // stay zero-initialized and genuinely unused, hence the cast.
    if constexpr (KVP != 2) { (void)kL; (void)vL; }

    frag16 qf[NKS];
    {
        const uint16_t* qp = (const uint16_t*)a.q
                           + ((size_t)(seq * a.q_len + qrow) * a.q_heads + qhead) * HEAD_DIM;
        #pragma unroll
        for (int t = 0; t < NKS; ++t) {
            const uint2 l0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h : 4 * h));
            const uint2 h0 = *(const uint2*)(qp + 16 * t + (CTG ? 8 * h + 4 : 8 + 4 * h));
            qf[t] = D::mk(make_uint2(r4d_qcvt<D, FOLDQ, F16>(l0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(l0.y, mul)),
                          make_uint2(r4d_qcvt<D, FOLDQ, F16>(h0.x, mul),
                                     r4d_qcvt<D, FOLDQ, F16>(h0.y, mul)));
        }
    }

    const int klimit = ctx - a.q_len + qrow;
    float m_ref = -3.0e38f;
    float off   = PSHIFT + 3.0e38f;
    float l_i   = 0.0f;
    // turbo4 (KVP == 2): running V-scale reference -- see the prefill kernel. P is folded with
    // sVs[key] / v_ref (<= 1) so the f16 P fragment stays under its PGROW budget (2^14 < 65504);
    // acc is rescaled when v_ref grows and the partial written below is multiplied back by it.
    float v_ref = 0.0f;
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int vg = lane >> 3, vj = lane & 7;
    uint4 vreg[VREGS];
    // turbo4: one thread's 4 keys x 4 d-item, a completely different map from vreg's
    // transposing-load one (see fetchV/storeV below), so it gets its own register array.
    uint32_t vreg2[KVP == 2 ? 4 * VQREGS : 1];
    float    vscale2[KVP == 2 ? 4 * VQREGS : 1];

    auto fetchV = [&](int k0, const int* blks) {
        if constexpr (KVP == 2) {
            // turbo4: V has its own record buffer (a.v_cache) and its own item map -- one thread
            // per (key-quad, d-quad), 4 keys per item, looped VQREGS times to cover NTQ items
            // (prefill's turbo4 fetchV assumes exactly one item per thread; decode's NTHR does
            // not always divide NTQ, hence the loop). Nibble-packed data can't use the
            // transposing global loads the other KVP legs use, so this is scalar loads,
            // transposed at store time instead of load time.
            #pragma unroll
            for (int r = 0; r < VQREGS; ++r) {
                const int idx = tid + r * NTHR;
                if (NTQ % NTHR == 0 || idx < NTQ) {
                    const int kq = idx % (TILE / 4), dq = idx / (TILE / 4);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const int kk  = min(k0 + 4 * kq + t, ctx - 1);
                        const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                        const uint8_t* rec = (const uint8_t*)a.v_cache
                                           + (size_t)blk * a.kv_block_stride
                                           + (size_t)(kk % BS) * a.kv_slot_stride
                                           + (size_t)kvh * a.kv_head_stride;
                        // 4 nibbles, elements 4dq..4dq+3.
                        const uint32_t nib16 = *(const uint16_t*)(rec + 2 + 2 * dq);
                        const uint32_t sgn   = (rec[130 + dq / 2] >> ((dq & 1) * 4)) & 0xFu;
                        vreg2[4 * r + t] = nib16 | (sgn << 16);
                        if (dq == 0) vscale2[4 * r + t] = r4d_f16_to_f32(*(const uint16_t*)rec);
                    }
                }
            }
            return;
        }
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP == 1 ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int kk = min(k0 + kg * 8 + vj, ctx - 1);
                const int d0 = dg * (KVP == 1 ? 32 : 64) + vg * (KVP == 1 ? 8 : 16);
                const int blk = BTS ? blks[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)blk * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride
                               + (size_t)(kk % BS) * (2 * HEAD_DIM) + HEAD_DIM + d0;
                const void* p = KVP == 1 ? (const void*)((const uint16_t*)a.kv + o)
                                         : (const void*)((const uint8_t*)a.kv + o);
                vreg[r] = __builtin_bit_cast(uint4, load_tr_b128(p));
            }
        }
    };

    auto storeV = [&]() {
        if constexpr (KVP == 2) {
            // turbo4: dequantise this thread's up-to-4 items (kq,dq), each 4 keys x 4 d, via the
            // e4m3 LUT, then transpose the 4x4 byte matrix (element-major, from fetchV) into
            // sV16's d-major layout -- same transpose prefill's storeV does into sV8, except
            // decode has no PV8 leg, so every transposed row of 4 e4m3 bytes is immediately
            // widened to 4 16-bit values with the SAME fp8x8_to_16x4w the fp8 leg uses (called on
            // one dword at a time: it widens 8 fp8 in 2 dwords to 8 16-bit in 4 dwords, so a
            // single 4-byte row uses only its first dword's output, w4[0]/w4[1]).
            #pragma unroll
            for (int r = 0; r < VQREGS; ++r) {
                const int idx = tid + r * NTHR;
                if (NTQ % NTHR == 0 || idx < NTQ) {
                    const int kq = idx % (TILE / 4), dq = idx / (TILE / 4);
                    uint32_t g[4];
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        const uint32_t nib16 = vreg2[4 * r + t] & 0xFFFFu;
                        const uint32_t sgn   = (vreg2[4 * r + t] >> 16) & 0xFu;
                        // 4 nibbles -> one index per byte, element order.
                        const uint32_t sel = (nib16 & 0x000Fu)
                                           | ((nib16 & 0x00F0u) << 4)
                                           | ((nib16 & 0x0F00u) << 8)
                                           | ((nib16 & 0xF000u) << 12);
                        g[t] = r4d_lut_gather4(sel, vL[0], vL[1], vL[2], vL[3]) | r4d_sign_spread4(sgn);
                    }
                    // g[t] byte e = element (4dq+e) of key (4kq+t); out_e byte t = element
                    // (4dq+e) of key (4kq+t) -- a 4x4 byte transpose, same as prefill's.
                    #pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const uint32_t out_e = ((g[0] >> (8 * e)) & 0xFFu)
                                             | (((g[1] >> (8 * e)) & 0xFFu) << 8)
                                             | (((g[2] >> (8 * e)) & 0xFFu) << 16)
                                             | (((g[3] >> (8 * e)) & 0xFFu) << 24);
                        uint32_t w4[4];
                        fp8x8_to_16x4w<D>(make_uint2(out_e, 0u), w4);
                        *(uint2*)(&sV16[(4 * dq + e) * VSTR + 4 * kq]) = make_uint2(w4[0], w4[1]);
                    }
                    if (dq == 0) {
                        #pragma unroll
                        for (int t = 0; t < 4; ++t) sVs[4 * kq + t] = vscale2[4 * r + t];
                    }
                }
            }
            return;
        }
        #pragma unroll
        for (int r = 0; r < VREGS; ++r) {
            const int idx = warp + r * NWARPS;
            if (NTR % NWARPS == 0 || idx < NTR) {
                constexpr int DG = HEAD_DIM / (KVP == 1 ? 32 : 64);
                const int kg = idx / DG, dg = idx - kg * DG;
                const int d0 = dg * (KVP == 1 ? 32 : 64) + vg * (KVP == 1 ? 8 : 16);
                const uint4 v = vreg[r];
                if constexpr (KVP == 1) {
                    uint16_t* dst = &sV16[(d0 + vj) * VSTR + kg * 8];
                    *(uint2*)(dst)     = make_uint2(D::from_bf16w(v.x), D::from_bf16w(v.y));
                    *(uint2*)(dst + 4) = make_uint2(D::from_bf16w(v.z), D::from_bf16w(v.w));
                } else {
                    const uint32_t lo0 = byte_gather(v.x, v.y, 0), lo1 = byte_gather(v.z, v.w, 0);
                    const uint32_t hi0 = byte_gather(v.x, v.y, 1), hi1 = byte_gather(v.z, v.w, 1);
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

    int blk[NB];
    if (BTS) {
        #pragma unroll
        for (int i = 0; i < NB; ++i) blk[i] = bt[t_lo * NB + i];
    }
    if (GPREV) fetchV(t_lo * TILE, blk);

    for (int ti = t_lo; ti < t_hi; ++ti) {
        const int k0 = ti * TILE;
        int blkn[NB];
        if (BTS && ti + 1 < t_hi) {
            const int tb = (ti + 1) * NB;
            #pragma unroll
            for (int i = 0; i < NB; ++i) blkn[i] = bt[tb + i];
        }
        if (LDSB) lds_barrier(); else __syncthreads();

        if (KLDS) {
            #pragma unroll 1
            for (int i = tid; i < TILE * (HEAD_DIM / 8); i += NTHR) {
                const int key = i / (HEAD_DIM / 8), ch = i - key * (HEAD_DIM / 8);
                const int kk = min(k0 + key, ctx - 1);
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                const size_t o = (size_t)kb * a.kv_block_stride
                               + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM);
                uint16_t* kd = &sK[key * KSTR];
                uint32_t w4[4];
                // KVP == 2 never reaches here (see the static_assert above): this branch stays
                // the bf16/fp8 pair it always was rather than gaining an unreachable turbo4 leg.
                if constexpr (KVP == 1) {
                    const uint4 r = *(const uint4*)((const uint16_t*)a.kv + o + ch * 8);
                    w4[0] = D::from_bf16w(r.x); w4[1] = D::from_bf16w(r.y);
                    w4[2] = D::from_bf16w(r.z); w4[3] = D::from_bf16w(r.w);
                } else {
                    fp8x8_to_16x4w<D>(*(const uint2*)((const uint8_t*)a.kv + o + ch * 8), w4);
                }
                *(uint2*)(kd + ch * 8)     = make_uint2(w4[0], w4[1]);
                *(uint2*)(kd + ch * 8 + 4) = make_uint2(w4[2], w4[3]);
            }
        }
        if (!GPREV) fetchV(k0, blk);
        storeV();
        if (LDSB) lds_barrier(); else __syncthreads();
        if (GPREV && ti + 1 < t_hi) fetchV(k0 + TILE, blkn);
        float inv_vref = 0.0f;
        if constexpr (KVP == 2) {
            float vsmax = 0.0f;
            #pragma unroll
            for (int i = 0; i < TILE; ++i) vsmax = fmaxf(vsmax, sVs[i]);
            if (vsmax > v_ref) {                    // uniform across the CTA: sVs is tile-wide
                const float r = (v_ref > 0.0f) ? (v_ref / vsmax) : 0.0f;
                #pragma unroll
                for (int dt = 0; dt < NKS; ++dt)
                    #pragma unroll
                    for (int e = 0; e < 8; ++e) acc[dt][e] *= r;
                v_ref = vsmax;
            }
            inv_vref = (v_ref > 0.0f) ? (1.0f / v_ref) : 0.0f;
        }

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            const uint8_t* grow = nullptr;
            if (!KLDS) {
                const int kk = min(k0 + mt * 16 + c, ctx - 1);
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                if constexpr (KVP == 2) {
                    // turbo4: K record is 162 B, byte-addressed with a byte-unit stride (see
                    // r4d.h) -- no *(2*HEAD_DIM) element stride like the fp8/bf16 legs. Lane c
                    // owns key (k0 + mt*16 + c)'s whole record; its fp16 scale (rec[0..1]) is the
                    // same for every t below, so read it once here into sKs, same index the QK
                    // step below reads it back at (mt*16 + 8*h + e is the OUTPUT key; this lane's
                    // OWN key mt*16+c is what it just fetched the scale for).
                    grow = (const uint8_t*)a.kv + (size_t)kb * a.kv_block_stride
                         + (size_t)(kk % BS) * a.kv_slot_stride + (size_t)kvh * a.kv_head_stride;
                    sKs[mt * 16 + c] = r4d_f16_to_f32(*(const uint16_t*)grow);
                } else {
                    grow = (const uint8_t*)a.kv + ((size_t)kb * a.kv_block_stride
                         + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM))
                         * (KVP == 1 ? 2 : 1);
                }
            }
            // turbo4: every lane in the warp just wrote sKs[mt*16 + <its own c>], covering all
            // TILE keys (c spans 0..15 across the warp's 32 lanes); the QK step below reads
            // sKs at a DIFFERENT index (the output key, not this lane's own), so the write has
            // to be visible workgroup-wide before that read.
            if constexpr (KVP == 2) { if (LDSB) lds_barrier(); else __syncthreads(); }
            v8f s = (v8f){0,0,0,0,0,0,0,0};
            #pragma unroll
            for (int g = 0; g < NKS; g += PF) {
                frag16 kb[PF];
                #pragma unroll
                for (int j = 0; j < PF; ++j) {
                    const int t = g + j;
                    if (KLDS) {
                        const uint16_t* kp = &sK[(mt * 16 + c) * KSTR + 16 * t];
                        if (CTG) {
                            const uint4 u = *(const uint4*)(kp + 8 * h);
                            kb[j] = D::mk(make_uint2(u.x, u.y), make_uint2(u.z, u.w));
                        } else {
                            kb[j] = D::mk(*(const uint2*)(kp + 4 * h), *(const uint2*)(kp + 8 + 4 * h));
                        }
                    } else if constexpr (KVP == 1) {
                        const uint16_t* kp = (const uint16_t*)grow + 16 * t;
                        const uint2 l0 = *(const uint2*)(kp + (CTG ? 8 * h : 4 * h));
                        const uint2 h0 = *(const uint2*)(kp + (CTG ? 8 * h + 4 : 8 + 4 * h));
                        kb[j] = D::mk(make_uint2(D::from_bf16w(l0.x), D::from_bf16w(l0.y)),
                                      make_uint2(D::from_bf16w(h0.x), D::from_bf16w(h0.y)));
                    } else if constexpr (KVP == 2) {
                        // turbo4: this lane's own key (grow) holds elements [16*t+8*h, +8) as 4
                        // nibble bytes (8 indices) + 1 sign byte -- dequantise via the e4m3 LUT
                        // with the same gather/perm/sign-spread sequence the prefill storeK
                        // turbo4 branch uses (there it does 16 elements/2 halves at once from a
                        // wide cooperative K fetch; here it is 8 elements/1 half per lane,
                        // straight from the paged record), then feed the SAME fp8x8_to_16x4w
                        // conversion the fp8 leg below uses unchanged.
                        const uint16_t* nib = (const uint16_t*)(grow + 2 + 8 * t + 4 * h);
                        const uint32_t n4  = (uint32_t)nib[0] | ((uint32_t)nib[1] << 16);
                        const uint32_t sgn = *(const uint8_t*)(grow + 130 + 2 * t + h);
                        const uint32_t ge = r4d_lut_gather4(n4 & 0x0F0F0F0Fu, kL[0], kL[1], kL[2], kL[3]);
                        const uint32_t go = r4d_lut_gather4((n4 >> 4) & 0x0F0F0F0Fu, kL[0], kL[1], kL[2], kL[3]);
                        const uint32_t o0 = __builtin_amdgcn_perm(go, ge, 0x05010400u) | r4d_sign_spread4(sgn & 0xFu);
                        const uint32_t o1 = __builtin_amdgcn_perm(go, ge, 0x07030602u) | r4d_sign_spread4((sgn >> 4) & 0xFu);
                        uint32_t w4[4];
                        fp8x8_to_16x4w<D>(make_uint2(o0, o1), w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    } else {
                        const uint8_t* kp = grow + 16 * t;
                        uint32_t w4[4];
                        // CONTIG makes this ONE 8-byte load where the split map needs two 4-byte.
                        const uint2 raw = CTG ? *(const uint2*)(kp + 8 * h)
                                              : make_uint2(*(const uint32_t*)(kp + 4 * h),
                                                           *(const uint32_t*)(kp + 8 + 4 * h));
                        fp8x8_to_16x4w<D>(raw, w4);
                        kb[j] = D::mk(make_uint2(w4[0], w4[1]), make_uint2(w4[2], w4[3]));
                    }
                }
                if (SGB) sched_barrier();
                #pragma unroll
                for (int j = 0; j < PF; ++j) s = D::wmma(kb[j], qf[g + j], s);
            }
            float sc[8];
            #pragma unroll
            for (int e = 0; e < 8; ++e) sc[e] = FOLDQ ? s[e] : s[e] * mul;
            if constexpr (KVP == 2) {
                // turbo4: S = q . (lut[idx]*sign) accumulated per key; the per-key fp16 scale
                // multiplies the score for that key, before the mask (same place/order as prefill).
                #pragma unroll
                for (int e = 0; e < 8; ++e) sc[e] *= sKs[mt * 16 + 8 * h + e];
            }

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
                const float p = ex2(sc[e] + off); pf[e] = p;
                if (!DOT2) lsum += p;
            }
            if (!DOT2) l_i += lsum;
            if constexpr (KVP == 2) {
                // turbo4: fold the per-key V scale into P AFTER the row-sum (softmax
                // normalisation above used unscaled P) and BEFORE the PV accumulation below.
                // DOT2 is forced off for KVP == 2 (see its definition), so l_i above always came
                // from the unscaled lsum, never from a packed fragment built after this fold.
                #pragma unroll
                for (int e = 0; e < 8; ++e) pf[e] *= sVs[mt * 16 + 8 * h + e] * inv_vref;
            }

            {
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
                frag16 p16;
                if (CTG) {
                    p16 = D::mk(lo, hi2);          // QK already left the keys lane-local
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
    if constexpr (KVP == 2) {
        // turbo4: undo the v_ref normalisation so the partial is in V's true units for the merge.
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt)
            #pragma unroll
            for (int e = 0; e < 8; ++e) acc[dt][e] *= v_ref;
    }
    const float lrow = l_i + swap16(l_i);
    const int tok = seq * a.q_len + qrow;
    const size_t idx = (size_t)(tok * a.q_heads + qhead) * splits + sp;
    const size_t nrow = (size_t)a.num_seqs * a.q_len * a.q_heads * splits;
    if (PF16) {
        // O/l in f16: bounded by max|v|, so no overflow, and the merge weight becomes w*l.
        uint16_t* so = (uint16_t*)a.scratch;
        float* sm = (float*)(so + nrow * HEAD_DIM);
        float* sl = sm + nrow;
        uint16_t* op = so + idx * HEAD_DIM;
        const float inv = (lrow > 0.0f) ? (1.0f / lrow) : 0.0f;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) {
            uint32_t tw[4];
            #pragma unroll
            for (int e = 0; e < 8; e += 2) tw[e >> 1] = DT16<1>::pk(acc[dt][e] * inv,
                                                                    acc[dt][e + 1] * inv);
            *(uint4*)(op + dt * 16 + 8 * h) = make_uint4(tw[0], tw[1], tw[2], tw[3]);
        }
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    } else {
        float* so = (float*)a.scratch;
        float* sm = so + nrow * HEAD_DIM;
        float* sl = sm + nrow;
        float* op = so + idx * HEAD_DIM;
        #pragma unroll
        for (int dt = 0; dt < NKS; ++dt) *(v8f*)(op + dt * 16 + 8 * h) = acc[dt];
        if (h == 0) { sm[idx] = m_ref; sl[idx] = lrow; }
    }
}

// Merge. PF16=1 reads normalised f16 partials and folds l into the weight; PF16=0 is the plain
// path (unnormalised f32 partials, l summed separately). Both walk the segments in FIXED INDEX
// ORDER so a request's rounding never depends on CTA scheduling.
template<int HEAD_DIM, int CHAINS, int PF16>
__global__ __launch_bounds__(HEAD_DIM)
void r4d_attn_splitkv_combine_kernel(const R4DArgs a, int splits, int tile)
{
    extern __shared__ float sw[];
    __shared__ float red[HEAD_DIM];
    const int job = blockIdx.x, d = threadIdx.x;
    const int tok = job / a.q_heads, seq = tok / a.q_len;
    const int ctx = a.seqused_k[seq];
    // A padded batch slot has no context at all. Its output is discarded by the caller, but the
    // segment count below would divide by zero, so leave zeros and stop. `ctx` is block-uniform,
    // so every thread takes this branch together and the barriers further down stay matched.
    if (ctx <= 0) { ((uint16_t*)a.out)[(size_t)job * HEAD_DIM + d] = 0; return; }
    const int u = r4d_attn_used(ctx, tile, splits);
    const size_t nrow = (size_t)a.num_seqs * a.q_len * a.q_heads * splits;
    const size_t base = (size_t)job * splits;

    const float* sm; const float* sl;
    if (PF16) { const uint16_t* so = (const uint16_t*)a.scratch;
                sm = (const float*)(so + nrow * HEAD_DIM); sl = sm + nrow; }
    else      { const float* so = (const float*)a.scratch;
                sm = so + nrow * HEAD_DIM; sl = sm + nrow; }

    float gmax = -INFINITY;
    for (int s = d; s < u; s += HEAD_DIM) gmax = fmaxf(gmax, sm[base + s]);
    red[d] = gmax; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] = fmaxf(red[d], red[d + off]);
        __syncthreads();
    }
    gmax = red[0]; __syncthreads();

    float tot = 0.0f;
    for (int s = d; s < u; s += HEAD_DIM) {
        const float mm = sm[base + s];
        float w = (mm <= -3.0e38f) ? 0.0f : ex2(mm - gmax);
        if (PF16) { w *= sl[base + s]; sw[s] = w; tot += w; }
        else      { sw[s] = w;         tot += w * sl[base + s]; }
    }
    red[d] = tot; __syncthreads();
    for (int off = HEAD_DIM / 2; off; off >>= 1) {
        if (d < off) red[d] += red[d + off];
        __syncthreads();
    }
    tot = red[0]; __syncthreads();

    float ch[CHAINS];
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) ch[i] = 0.0f;
    int s = 0;
    if (PF16) {
        const uint16_t* op = (const uint16_t*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], (float)__builtin_bit_cast(
                    __fp16, op[(size_t)(s + i) * HEAD_DIM]), ch[i]);
        }
        for (; s < u; ++s)
            ch[0] = fmaf(sw[s], (float)__builtin_bit_cast(__fp16, op[(size_t)s * HEAD_DIM]), ch[0]);
    } else {
        const float* op = (const float*)a.scratch + base * HEAD_DIM + d;
        for (; s + CHAINS <= u; s += CHAINS) {
            #pragma unroll
            for (int i = 0; i < CHAINS; ++i)
                ch[i] = fmaf(sw[s + i], op[(size_t)(s + i) * HEAD_DIM], ch[i]);
        }
        for (; s < u; ++s) ch[0] = fmaf(sw[s], op[(size_t)s * HEAD_DIM], ch[0]);
    }
    float av = 0.0f;
    #pragma unroll
    for (int i = 0; i < CHAINS; ++i) av += ch[i];

    const int kvh = (job % a.q_heads) / (a.q_heads / a.kv_heads);
    const float vd = a.v_descale ? a.v_descale[seq * a.kv_heads + kvh] : 1.0f;
    ((uint16_t*)a.out)[(size_t)job * HEAD_DIM + d] =
        f32_to_bf16(tot > 0.0f ? av * vd / tot : 0.0f);
}
