// R4D attention dispatch. One translation unit so the KV-dtype variants share the device code:
// each is a template instantiation, not a copy. The entry points are named for the geometry they
// are compiled for (head_dim 256, 6 queries per KV head, paged block 16) because that geometry is
// a hard gate -- everything else is rejected below rather than run.
//
// The query is bf16 in every variant. Quantising it measured an order of magnitude worse against
// the fp32 oracle (2.58e-02 vs ~2.3e-03), and the two-term e4m3 form that DOES reach f16 accuracy
// is 4-10% slower end to end -- so the 8-bit query arms are gone rather than kept behind a flag.
// What ships is the two families that differ only in KV cache dtype: bf16 KV (all 16-bit) and fp8
// KV (16/8 mixed, the production configuration).
//
// `opt` is a bitmask of the measured optimisation steps, all of them on:
//     prefill 3955211 = f16 operands | mask skip | folded query scale | contiguous-k fragments
//                     | prefetch 4 + scheduling barrier | dot2 row sum | block table in SGPRs
//                     | wide K fetch | V prefetch across the m-tile loop | LDS-scoped barrier
//     decode  3430971 = the same, minus the wide K fetch (its K never reaches LDS), plus reading
//                       K straight from the paged cache (bit 16) and f16 split-KV partials (32).
// Each bit is defined and justified where it is used, in r4d_attn_prefill_h256_gqa6.hip and
// r4d_attn_decode_h256_gqa6.hip.
// They are kept separable so any one of them can be ablated again after a geometry change.
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

// The geometry every kernel in this translation unit is compiled for. HEAD_DIM and GQA appear in
// the entry-point names; BLOCK_SIZE does not, because it is 16 for the whole library and so
// discriminates nothing between them.
#define A_HEAD_DIM   256
#define A_GQA          6
#define A_BLOCK_SIZE  16
#define A_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold

extern "C" void r4d_attn_dims(int* head_dim, int* gqa, int* block_size, int* max_decode_rows) {
    if (head_dim)        *head_dim        = A_HEAD_DIM;
    if (gqa)             *gqa             = A_GQA;
    if (block_size)      *block_size      = A_BLOCK_SIZE;
    if (max_decode_rows) *max_decode_rows = A_MAX_DECODE_ROWS;
}

// ---- prefill ---------------------------------------------------------------------------------
// LDS, head_dim 256, 16-bit K and V: sK = TILE*264*2, sV = 256*(TILE+8)*2. bf16 KV at TILE 64
// would need 68096 B > 65536, so the geometry stops at 48.
#define P_NWARPS 24
#define P_TILE   48
#define P_OPT    3955211

// KVP: 0 = fp8 e4m3 cache, 1 = bf16 cache, 2 = turbo4_fp8_bs256 cache (K/V separate 162-byte
// records + 16-entry e4m3 LUTs, step A of the ml8-4 load stage -- prefill only). Reached only
// through the named wrappers below.
template <int KVP>
static int prefill_launch(const R4DArgs* a, hipStream_t s) {
    if constexpr (KVP == 2) {
        if (!a->v_cache || !a->k_lut || !a->v_lut || a->kv_slot_stride <= 0) return -5;
    }
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    constexpr int BQ = (P_NWARPS * 16) / A_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    // R4D_ATTN_FP8: 0 = the shipped f16 legs; 1 = O_QK8, 2 = O_PV8, 3 = both. fp8 KV only.
    // Opt-in and read once: the 8-bit legs trade oracle accuracy (1.7e-3 -> 1.7e-2/2.6e-2/3.1e-2
    // row relRMSE) for WMMA issue rate (+10%/+24%/+56% at a hot 8k chunk, +18% at a cold 40k
    // chunk, ~/mxfp4_work/tier8). Whether that accuracy trade is FREE at the end task is exactly
    // what the serving gates (ppl + GSM8K paired) decide; kernel-level parity already decided the
    // default, which is why these are bits and not the baseline.
    int fp8mode = 0;
    if constexpr (KVP != 1) {
        static const int m = [] {
            const char* e = getenv("R4D_ATTN_FP8");
            return e ? atoi(e) : 0;
        }();
        fp8mode = m;
    }
#define P_LAUNCH(OPTX)                                                                            \
    do {                                                                                          \
        hipLaunchKernelGGL((r4d_attn_prefill_kernel<P_NWARPS, P_TILE, A_HEAD_DIM, A_GQA,          \
                                                    A_BLOCK_SIZE, KVP, OPTX>),                    \
                           grid, dim3(P_NWARPS * 32), 0, s, *a);                                  \
    } while (0)
    if constexpr (KVP == 2) {
        // turbo4_fp8_bs256 always dequantises straight into the 8-bit K/V legs; R4D_ATTN_FP8 is
        // not consulted (there is no 16-bit leg to fall back to for this cache format), so only
        // the QK8|PV8 variant is instantiated -- avoids compiling the other three fp8mode arms
        // that r4d_attn_prefill_h256_gqa6.hip's static_assert(KVP != 2 || (QK8 && PV8)) rejects.
        P_LAUNCH(P_OPT | O_QK8 | O_PV8);
    } else if constexpr (KVP != 1) {
        if (fp8mode == 3)      P_LAUNCH(P_OPT | O_QK8 | O_PV8);
        else if (fp8mode == 2) P_LAUNCH(P_OPT | O_PV8);
        else if (fp8mode == 1) P_LAUNCH(P_OPT | O_QK8);
        else                   P_LAUNCH(P_OPT);
    } else {
        P_LAUNCH(P_OPT);
    }
#undef P_LAUNCH
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<0>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_turbo4kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<2>(a, s);
}
extern "C" int r4d_attn_prefill_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_launch<1>(a, s);
}

// ---- decode ----------------------------------------------------------------------------------
#define D_TILE 16
#define D_OPT  3430971

// Split law. The first version, splits ~ 16*cbrt(tiles/programs), OVERSHOOTS: at 256K/TILE16 it
// asks for 322 and clamps to 256, where a sweep of {32,64,128,256} put the argmin at 64 at EVERY
// depth (0.985 ms vs 1.112 at the law's 256, 256K/bf16 KV). Extra splits do not buy parallelism
// once the device is full -- they buy partial-buffer traffic, which is written and read back.
//
// So the law is a machine-fill target divided by the grid units that are NOT segments, snapped
// down to an instantiated power of two, then clamped so segments never get thinner than 2 tiles.
// The argmin is depth-invariant, which is why there is no context term: only the thin-segment
// clamp depends on length.
static int decode_splits(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + D_TILE - 1) / D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;              // CU-fill target over the non-segment grid units
    int s = 128;                                // ceiling: 256 measured worse at head_dim 256
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;                  // >= 2 tiles per segment
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

// Same for both cache dtypes: the split law does not read the dtype, and the partials are f16
// either way (opt bit 32).
extern "C" long r4d_attn_decode_h256_gqa6_scratch_bytes(const R4DArgs* a) {
    const int  splits = decode_splits(a);
    const long rows   = (long)a->num_seqs * a->q_len * a->q_heads;
    return (D_OPT & 32) ? rows * splits * (long)(a->head_dim * 2 + 8)       // f16 O/l + m + l
                        : rows * splits * (long)(a->head_dim + 2) * 4L;     // f32 O partials + m + l
}

template <int KVP>
static int decode_launch(const R4DArgs* a, hipStream_t s) {
    if constexpr (KVP == 2) {
        if (!a->v_cache || !a->k_lut || !a->v_lut || a->kv_slot_stride <= 0) return -5;
    }
    if (a->head_dim != A_HEAD_DIM || a->block_size != A_BLOCK_SIZE) return -1;
    if (a->q_heads / a->kv_heads != A_GQA) return -2;
    const int rows = a->q_len * A_GQA;
    if (rows > A_MAX_DECODE_ROWS) return -4;    // verify width <= 10 at gqa 6
    const int nw = (rows + 15) / 16;            // gqa-aligned height; no dead m-tiles issued
    const int splits = decode_splits(a);

#define D_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, D_TILE, A_HEAD_DIM, A_GQA, A_BLOCK_SIZE,    \
                                               KVP, D_OPT>),                                  \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
    CHK();

    const int jobs3 = a->num_seqs * a->q_len * a->q_heads;
    if (D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, 1>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<A_HEAD_DIM, 4, 0>), dim3(jobs3),
                           dim3(256), splits * sizeof(float), s, *a, splits, D_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_decode_h256_gqa6_fp8kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<0>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_bf16kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<1>(a, s);
}
extern "C" int r4d_attn_decode_h256_gqa6_turbo4kv(const R4DArgs* a, hipStream_t s) {
    return decode_launch<2>(a, s);
}
