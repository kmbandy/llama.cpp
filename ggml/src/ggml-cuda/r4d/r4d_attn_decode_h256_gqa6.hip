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
    constexpr int DOT2  = ((OPT & O_DOT2) && (OPT & O_F16)) ? 1 : 0;
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
    constexpr int NTR   = (TILE / 8) * (HEAD_DIM / (KVP ? 32 : 64));
    constexpr int VREGS = (NTR + NWARPS - 1) / NWARPS;
    // A V prefetch costs VREGS x 4 VGPRs and this leg already carries a 128-VGPR accumulator and a
    // 64-VGPR query. Measured: at two uint4 (fp8 KV, TILE 16) it is worth 6-8% at depth; at four
    // (fp8 TILE 32, or bf16 TILE 16, whose V is twice the bytes) it spills and LOSES 10%. So the
    // geometry decides, not the flag.
    constexpr int GPREV = ((OPT & O_GPREV) && VREGS <= 2) ? 1 : 0;
    constexpr float PSHIFT = D::SHIFT;
    constexpr float PGROW  = D::GROW;

    __shared__ uint16_t sK[KLDS ? TILE * KSTR : 1];
    __shared__ uint16_t sV16[HEAD_DIM * VSTR];

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
    v8f acc[NKS];
    #pragma unroll
    for (int t = 0; t < NKS; ++t) acc[t] = (v8f){0,0,0,0,0,0,0,0};

    const int vg = lane >> 3, vj = lane & 7;
    uint4 vreg[VREGS];

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
                if (KVP) {
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

        #pragma unroll 1
        for (int mt = 0; mt < MT; ++mt) {
            const uint8_t* grow = nullptr;
            if (!KLDS) {
                const int kk = min(k0 + mt * 16 + c, ctx - 1);
                const int kb = BTS ? blk[(kk - k0) / BS] : bt[kk / BS];
                grow = (const uint8_t*)a.kv + ((size_t)kb * a.kv_block_stride
                     + (size_t)kvh * a.kv_head_stride + (size_t)(kk % BS) * (2 * HEAD_DIM))
                     * (KVP ? 2 : 1);
            }
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
                    } else if (KVP) {
                        const uint16_t* kp = (const uint16_t*)grow + 16 * t;
                        const uint2 l0 = *(const uint2*)(kp + (CTG ? 8 * h : 4 * h));
                        const uint2 h0 = *(const uint2*)(kp + (CTG ? 8 * h + 4 : 8 + 4 * h));
                        kb[j] = D::mk(make_uint2(D::from_bf16w(l0.x), D::from_bf16w(l0.y)),
                                      make_uint2(D::from_bf16w(h0.x), D::from_bf16w(h0.y)));
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
