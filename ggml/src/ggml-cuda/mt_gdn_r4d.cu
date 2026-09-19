// mt_gdn_r4d — R4D-backed path for GGML_OP_GATED_DELTA_NET.
//
// See mt_gdn_r4d.cuh for the dispatch contract. This file's job is translating between ggml's
// gated_delta_net op contract (gated_delta_net.cu:ggml_cuda_op_gated_delta_net_impl) and libr4d's
// GDN C ABI (ggml-cuda/r4d/r4d.h) — layout algebra plus f32<->bf16 casts, no allocations or syncs
// on the hot path (persistent grow-only scratch, same pattern as mt_pagedattn_r4d.cu).
//
// ── The layout algebra ───────────────────────────────────────────────────────────────────────
//
// ggml's op contract (gated_delta_net.cu):
//   q,k   : ne = [K, Hg, T, N]  f32, contiguous
//   v     : ne = [V, H,  T, N]  f32, contiguous
//   g     : ne = [1 or V, H, T, N] f32 (ne0==1: scalar/GDA gate; ne0==V: per-channel/KDA gate)
//   beta  : ne = [1, H, T, N]   f32
//   state : ne = [V, V, H, N]   f32   (r4d.h calls the two V-sized axes K and V; see below)
//   dst   : ne = [row_elems*T*N + state tail], f32 — first V*H*T*N floats are the attention-like
//           output in the SAME [V,H,T,N] layout as v; the tail (when K_snapshots<=1, the only case
//           this path serves) is exactly one more state-shaped block, S_v*S_v*H*N floats.
//
// For a CONTIGUOUS ggml tensor, ne0 is the fastest-varying axis. So q/k are laid out, fastest to
// slowest, as (K, Hg, T, N) — i.e. for fixed N,T the K-then-Hg-then-T ordering libr4d calls
// "[T, Hg, K]" (K fastest) falls out for free, and concatenating N sequences back-to-back (ggml's
// outermost/slowest axis) is exactly what libr4d's cu_seqlens-addressed "T_total" axis wants. Same
// argument for v against libr4d's "[T, H, V]" (up to the head PERMUTATION below). g/beta have ne0
// collapsed to 1 (GDA) so H is the fastest real axis — exactly libr4d's "[T, H]" (H fastest)
// reading, again up to the permutation. So q, k, v, g, beta need only a dtype cast (f32->bf16 for
// q/k/v; g/beta are fp32 on both sides already) plus, for v/g/beta/state/output, the head
// permutation below — no transpose, no other per-token indexing.
//
// state is the one place two same-sized axes (both S_v=128) could hide a transpose bug, so this
// was checked directly against the recurrent kernel's own comment (gated_delta_net.cu, the
// autoregressive kernel): "state is stored transposed: M[col][i] = S[i][col], row col is
// contiguous" with `curr_state += ... + col * S_v` (col = ne1, i = ne0) — i.e. ne0 is S's ROW index
// (the key/K axis) and ne1 is S's COLUMN index (the value/V axis), row-major with the K axis
// contiguous. r4d.h's h0/ht are documented as "[N, H, V, K]" (K fastest) — the same K-fastest,
// V-next ordering. So state needs no transform on its (V,K) axes, only the same H-axis
// permutation as v/g/beta/output.
//
// ── The GQA head-convention permutation (MAD-406 follow-up) ────────────────────────────────────
//
// Qwen3.8-27B's linear-attention hparams are linear_num_key_heads=16, linear_num_value_heads=48
// (a GQA repeat R=3), linear_key_head_dim=linear_value_head_dim=128 — see
// /home/kmbandy/GitHub/vllm-radiance/models/Qwen3.8-27B-*/config.json. K=128/V=128 match libr4d's
// fixed k128/v128/c64 geometry exactly (r4d_gdn_dims). Production runs with
// cparams.fused_gdn_ar && fused_gdn_ch both true (llama-context.cpp), so qwen35.cpp's
// build_layer_attn_linear does NOT pre-broadcast q_conv/k_conv to H heads — this op receives the
// raw Hg=16, H=48 tensors on every call, and the in-op GQA repeat is load-bearing.
//
// ggml and libr4d use different value-head -> key/query-head mappings:
//   - ggml (gated_delta_net.cu's autoregressive kernel): `iq1 = h_idx % neqk1` — value-head h maps
//     to key/query-head (h mod Hg). Confirmed by reading gated_delta_net_cuda directly.
//   - libr4d (r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip): `hq = hv / (H / Hg)` — value-head hv
//     maps to key/query-head (hv / R), R = H/Hg.
// These disagree whenever R > 1 (Hg < H): passing ggml's raw value-head order straight through to
// libr4d would pair the wrong key/query head with 2 of every 3 value heads (R=3 here).
//
// Fix: relabel every value-head-indexed tensor (v, g, beta, state h0/ht, the op's own output) by
// the permutation
//     perm(h) = R*(h mod Hg) + (h / Hg),        R = H / Hg
// before/after calling libr4d. Proof this fixes it: libr4d's key/query-head for hv'=perm(h) is
//     hv' / R = (h mod Hg)                              (since 0 <= h/Hg < R, floor divides out)
// which is exactly ggml's own key/query-head for h. So after relabeling, libr4d's blocked
// convention reproduces ggml's interleaved one exactly. perm() is a bijection on [0,H) with an
// R-way structure (it maps head h -> hv' = R*(h%Hg) + h/Hg); applying it going in (index by ggml
// h, write at hv'=perm(h)) and applying the SAME forward formula going out (index by ggml h, read
// from hv'=perm(h)) are inverses of each other by construction, so no separate inverse formula is
// needed anywhere in this file. When Hg==H (R=1), perm(h) = h — the identity, so the Hg==H case
// (still reachable if fused_gdn_ar/ch are ever off) needs no special-casing.
//
// q and k are indexed by Hg (key/query-head count) and never touched by this permutation — see the
// coordinator's derivation this file implements. The permutation is applied to v/g/beta during
// input staging (folded into the existing f32->bf16 cast and cumsum kernels — they already touch
// every element, so a permuted destination index is free) and to h0 with a small copy kernel
// (fp32, no cast). It is applied to the chunk_scan output (o) and final state (ht) during output
// destaging, in the same kernels that already do the bf16->f32 cast / state copy.
//
// ── Precondition this file relies on: q/k arrive already L2-normalized ─────────────────────────
//
// r4d_gdn_kkt_solve_k128_c64_bf16 and r4d_gdn_chunk_scan_k128_v128_c64_bf16 do NOT L2-normalize
// q/k themselves (verified: no norm/rsqrt code in either .hip file) -- unlike libr4d's own
// conv_prep (r4d_gdn_conv_w4_h128_bf16.hip: "const float inv = 1.0f / sqrtf(ss + 1e-6f); //
// L2NORM_EPS, as in the kernel it replaces", applied to q/k, NOT v) and unlike
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state, which does its own internal L2-norm. This
// file relies entirely on ggml having already normalized src_q/src_k before this op runs --
// qwen35.cpp calls build_gdn_l2_norm() on q_conv/k_conv UNCONDITIONALLY, before the
// fused_gdn_ar/ch repeat-or-not branch, so this holds in every configuration this op is reachable
// from, permuted GQA case included. Skipping this precondition (e.g. in a test harness that feeds
// raw random q/k straight to these kernels) does not fail gracefully: the chunk KKT triangular
// solve (I + strict_lower(diag(beta) K K^T e^dg))^-1 is only well-conditioned for the |k|=1 regime
// the algorithm was designed for, and un-normalized k with a large K.K^T can blow the forward
// substitution up by many orders of magnitude across a 64-token chunk while leaving a naive
// per-token reference (no matrix inversion) numerically unaffected -- exactly the failure mode
// this integration hit before tests/test-r4d-gdn.hip.cpp was fixed to normalize its synthetic
// q/k the same way (see that file's l2_normalize_rows()).
//
// ── What this file actually computes ─────────────────────────────────────────────────────────
//
// 1. Flat f32->bf16 casts of q, k (no permutation: indexed by Hg).
// 2. A permuting f32->bf16 cast of v: dst[(gt*H + perm(h))*V + v] = bf16(src[(gt*H+h)*V+v]).
// 3. A combined kernel that, per (n,h), walks t computing BOTH the per-chunk (64-token, reset at
//    each sequence's own chunk boundary) inclusive cumsum of raw g, and a straight copy of beta —
//    writing both into the H axis at perm(h) instead of h. libr4d wants g "already summed along
//    the chunk" (r4d.h); ggml's own gate tensor is the raw per-token log-decay, never cumsum'd,
//    because ggml's own kernels are a plain serial recurrence that never needed a materialized
//    prefix sum.
// 4. A permuting copy of h0 (fp32, no cast) into H0_PERM at perm(h).
// 5. cu_seqlens[i] = i*T (uniform per-seq token count T — ggml's op batches all N sequences to the
//    same T by construction, unlike the paged-attention op, so there is no ragged case here).
// 6. r4d_gdn_kkt_solve_k128_c64_bf16, then r4d_gdn_chunk_scan_k128_v128_c64_bf16, called with the
//    REAL Hg/H (16/48) — libr4d's own GQA repeat now agrees with ggml's because every value-head
//    tensor it sees is already relabeled.
// 7. A permuting bf16->f32 cast of the chunk_scan output back into dst's head block:
//    dst[(gt*H+h)*V+v] = f32(out_bf16[(gt*H+perm(h))*V+v]).
// 8. A permuting copy of ht (fp32, no cast) from HT_PERM back into dst's state tail at h.
//
// ── Eligibility: T must be a whole multiple of the 64-token chunk (empirically required) ───────
//
// r4d_gdn_chunk_scan_k128_v128_c64_bf16's h0/ht carry across calls is bit-exact ONLY when every
// call's token count is a multiple of the 64-token chunk (verified with
// tests/test-r4d-gdn.hip.cpp's split-vs-whole check: splitting one T=100 call into two calls at
// the 64-token boundary reproduces the whole-call output and final state bit-for-bit; splitting at
// 50 or 36 -- NOT chunk-aligned -- diverges by 0.09-2.8 absolute, growing the further the split sits
// from a chunk boundary). This is a property of the chunked WY-transform itself, not a marshalling
// bug: the algorithm's "chunk" is only self-contained (state-in, state-out, no cross-chunk
// dependency past the state) at its own boundary, and libr4d has no path for continuing a *partial*
// chunk's forward substitution from an intermediate state.
//
// So this file takes the R4D path ONLY when n_tokens (this call's T, which is per-sequence and
// uniform across the op's N sequences by construction -- see the layout algebra above) is a
// multiple of 64. A single-token decode call (T==1) fails this trivially and is never sent to
// chunk_scan; ggml's own gated_delta_net_cuda (the plain per-token autoregressive kernel) already
// serves decode exactly, one token at a time, with no chunk-size constraint, so declining here
// costs nothing decode was ever going to get from this path. A non-64-aligned PREFILL tail (e.g. a
// ubatch of 1000 tokens where a prior chunk-scan call already consumed 960 and 40 remain) also
// declines and falls through to the same exact per-token kernel for that call.
//
// ── The state carry across ubatches, in ggml's own terms ────────────────────────────────────────
//
// dst->src[5] (src_state, this call's h0) and dst's state tail (this call's ht) ARE the persistent
// recurrent-state cache slice for this op's sequences across ubatches (mctx_cur->get_s_l(il) in
// qwen35.cpp's build_layer_attn_linear/build_rs -- the same tensor gated_delta_net.cu's own
// autoregressive/chunked-prefill kernels read as `curr_state` and write as `state`, this file just
// being another producer/consumer of that same slice). This file's own round trip through that
// slice -- h0_d -(permute, fp32, no cast)-> h0_perm -> chunk_scan -> ht_perm -(permute, fp32, no
// cast)-> ht_d -- is bit-exact BY CONSTRUCTION regardless of chunk alignment: the permutation is a
// pure data relabeling (a copy at a different offset, no arithmetic), so it introduces no rounding
// on either side. The alignment requirement above is entirely about what chunk_scan itself computes
// into ht_perm, not about anything this file's own plumbing does to it. Consequence: for a run of
// consecutive 64-aligned ubatches (in particular, the common production case of 1024-token
// ubatches -- 1024 = 16*64), the state this file writes back is bit-for-bit what a single call
// over the whole span would have produced, so chaining aligned ubatches composes exactly.

#include "common.cuh"
#include "mt_gdn_r4d.cuh"

#ifdef GGML_HIP_R4D

#include "r4d/ggml-r4d.h"

#include <array>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace {

// value-head permutation: ggml's (h mod Hg) key/query-head convention <-> libr4d's (hv / R) one.
// R = H/Hg. Self-inverse under "apply forward both ways" — see the file header proof.
__host__ __device__ __forceinline__ int r4d_gdn_perm_head(int h, int Hg, int R) {
    return R * (h % Hg) + (h / Hg);
}

// ─────────────────────────────────────────────────────────────────────────
// Persistent, grow-only, capture-safe scratch. Same reasoning as
// mt_pagedattn_r4d.cu's r4d_persist_get (MAD-288) — duplicated rather than
// shared because that's a different, independently-evolving translation
// unit. Keyed by (device, stream) so overlapping meta contexts on one
// device (GGML_META_OVERLAP) never share a buffer.
struct r4d_gdn_persist_buf {
    void * ptr   = nullptr;
    size_t bytes = 0;
};
enum r4d_gdn_persist_slot {
    R4D_GDN_Q_BF16 = 0,
    R4D_GDN_K_BF16,
    R4D_GDN_V_BF16,       // value-head-permuted
    R4D_GDN_OUT_BF16,     // value-head-permuted (libr4d's own order)
    R4D_GDN_G_CUMSUM,     // f32, value-head-permuted, [T_total, H]
    R4D_GDN_BETA_PERM,    // f32, value-head-permuted, [T_total, H]
    R4D_GDN_H0_PERM,      // f32, value-head-permuted, [N, H, V, K]
    R4D_GDN_HT_PERM,      // f32, value-head-permuted, [N, H, V, K] (libr4d's output, pre-unpermute)
    R4D_GDN_A_SCRATCH,    // bf16, [T_total, H, 64] — kkt_solve output / chunk_scan input
    R4D_GDN_CU_SEQLENS,   // int32, N+1
    R4D_GDN_PERSIST_COUNT
};
static std::mutex g_r4d_gdn_persist_mutex;
static std::map<std::pair<int, cudaStream_t>, std::array<r4d_gdn_persist_buf, R4D_GDN_PERSIST_COUNT>> g_r4d_gdn_persist;

template <typename T>
static T * r4d_gdn_persist_get(int device, cudaStream_t stream, r4d_gdn_persist_slot slot, size_t n_elems) {
    const size_t need = n_elems * sizeof(T);
    std::lock_guard<std::mutex> lock(g_r4d_gdn_persist_mutex);
    r4d_gdn_persist_buf & b = g_r4d_gdn_persist[std::make_pair(device, stream)][slot];
    if (need > b.bytes) {
        size_t bytes = std::max(need, b.bytes * 2);
        bytes = (bytes + (1u << 20) - 1) & ~(size_t) ((1u << 20) - 1);
        void * ptr = nullptr;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        b.ptr   = ptr;
        b.bytes = bytes;
    }
    return (T *) b.ptr;
}

// ─────────────────────────────────────────────────────────────────────────
// Kernel A: flat elementwise f32 -> bf16 cast over a contiguous buffer of n_elems floats. Used
// for q/k, which are indexed by Hg and never permuted.
__global__ void r4d_gdn_cast_f32_to_bf16_kernel(const float * __restrict__ src, nv_bfloat16 * __restrict__ dst,
                                                 size_t n_elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// Kernel B: f32 -> bf16 cast of a [T_total, H, V] buffer, permuting the H axis: src index h ->
// dst index perm(h). Used for v (ggml order in -> libr4d order out).
__global__ void r4d_gdn_cast_v_permute_kernel(const float * __restrict__ src, nv_bfloat16 * __restrict__ dst,
                                               size_t T_total, int H, int Hg, int R, int V) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = T_total * (size_t) H * (size_t) V;
    if (i >= total) {
        return;
    }
    const int    v_  = (int) (i % V);
    const size_t tmp = i / V;
    const int    h   = (int) (tmp % H);
    const size_t gt  = tmp / H;
    const int    hp  = r4d_gdn_perm_head(h, Hg, R);
    dst[(gt * H + hp) * (size_t) V + v_] = __float2bfloat16(src[i]);
}

// Kernel C: bf16 -> f32 cast of a [T_total, H, V] buffer, permuting the H axis the OTHER way:
// dst index h reads src index perm(h). Used for the chunk_scan output (libr4d order in -> ggml
// order out, straight into dst's head block). Same perm() formula as kernel B — see the file
// header proof for why one formula serves both directions.
__global__ void r4d_gdn_cast_o_unpermute_kernel(const nv_bfloat16 * __restrict__ src, float * __restrict__ dst,
                                                 size_t T_total, int H, int Hg, int R, int V) {
    const size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = T_total * (size_t) H * (size_t) V;
    if (j >= total) {
        return;
    }
    const int    v_  = (int) (j % V);
    const size_t tmp = j / V;
    const int    h   = (int) (tmp % H);
    const size_t gt  = tmp / H;
    const int    hp  = r4d_gdn_perm_head(h, Hg, R);
    dst[j] = __bfloat162float(src[(gt * H + hp) * (size_t) V + v_]);
}

// Kernel D: per-chunk (64-token, reset at each sequence's own boundary) inclusive prefix sum of
// raw g, PLUS a straight copy of beta, both permuting the H axis (h -> perm(h)) as they go. One
// thread per (seq, ggml-head); each thread walks its own T-length column sequentially (T is at
// most a few thousand, this runs once per GDN op call — acceptable for a first cut, not on any
// inner hot loop the profiler flagged).
__global__ void r4d_gdn_g_beta_permute_kernel(const float * __restrict__ g_in, const float * __restrict__ beta_in,
                                               float * __restrict__ g_out, float * __restrict__ beta_out,
                                               int N, int T, int H, int Hg, int R, int chunk) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (h >= H) {
        return;
    }
    const int hp = r4d_gdn_perm_head(h, Hg, R);
    const float * gcol_in  = g_in    + (size_t) n * T * H + h;
    const float * bcol_in  = beta_in + (size_t) n * T * H + h;
    float *       gcol_out = g_out   + (size_t) n * T * H + hp;
    float *       bcol_out = beta_out+ (size_t) n * T * H + hp;
    float acc = 0.0f;
    for (int t = 0; t < T; ++t) {
        if (t % chunk == 0) {
            acc = 0.0f;
        }
        acc += gcol_in[(size_t) t * H];
        gcol_out[(size_t) t * H] = acc;
        bcol_out[(size_t) t * H] = bcol_in[(size_t) t * H];
    }
}

// Kernel E: permuting fp32 copy of a [N, H, V, K] state buffer, H axis h -> perm(h) (used for
// h0, going in) or perm(h) -> h (used for ht, coming out — same formula, see the file header
// proof). One block per (n, h_src); threads stride over the V*K elements of that head's block.
__global__ void r4d_gdn_state_permute_kernel(const float * __restrict__ src, float * __restrict__ dst,
                                              int N, int H, int Hg, int R, int VK, bool src_is_ggml_order) {
    const int n = blockIdx.y;
    const int h = blockIdx.x;
    if (n >= N || h >= H) {
        return;
    }
    const int hp = r4d_gdn_perm_head(h, Hg, R);
    const size_t src_off = src_is_ggml_order ? (((size_t) n * H + h)  * VK) : (((size_t) n * H + hp) * VK);
    const size_t dst_off = src_is_ggml_order ? (((size_t) n * H + hp) * VK) : (((size_t) n * H + h)  * VK);
    for (int i = threadIdx.x; i < VK; i += blockDim.x) {
        dst[dst_off + i] = src[src_off + i];
    }
}

// Kernel F: cu_seqlens[i] = i*T for i in [0, N]. Single-thread, allocation- and sync-free — capture
// safe (same reasoning as mt_pagedattn_r4d.cu's r4d_build_cu_seqlens_and_seqused_kernel).
__global__ void r4d_gdn_build_cu_seqlens_kernel(int32_t * __restrict__ cu, int N, int T) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    for (int i = 0; i <= N; ++i) {
        cu[i] = i * T;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Strided kernels for ggml_cuda_gdn_r4d_prefix (the production, use_prefill_chunked entry point):
// unlike ggml_cuda_op_gated_delta_net_r4d above, the source tensors here are the FULL n_tokens-
// length buffers (q_d/k_d/v_d/g_d/b_d), not a standalone P-token tensor, and the destination is
// the FULL dst buffer with its own dst_seq_stride (n_tokens) distinct from P — so every read/write
// needs the real strides rather than a flat index. Same permutation, same math, just addressed via
// sq*/sv*/sb*/dst_seq_stride instead of assuming a canonical product-of-dims layout.

// Kernel G: f32 -> bf16 cast of q or k's first P tokens of every sequence, no permutation
// (indexed by Hg). One thread per (n, t, h, k) element (flattened).
__global__ void r4d_gdn_cast_qk_strided_kernel(
        const float * __restrict__ src, nv_bfloat16 * __restrict__ dst,
        int64_t n_seqs, int64_t P, int Hg, int K_dim,
        int64_t s1, int64_t s2, int64_t s3) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_seqs * P * Hg * K_dim;
    if (i >= total) {
        return;
    }
    const int    k   = (int) (i % K_dim);
    size_t       tmp = i / K_dim;
    const int    h   = (int) (tmp % Hg);
    tmp /= Hg;
    const int    t   = (int) (tmp % P);
    const int64_t n  = tmp / P;
    dst[i] = __float2bfloat16(src[(size_t) n * s3 + (size_t) t * s2 + (size_t) h * s1 + k]);
}

// Kernel H: f32 -> bf16 cast of v's first P tokens of every sequence, permuting the H axis
// h -> perm(h) into the canonical (flat) destination layout kkt_solve/chunk_scan expect.
__global__ void r4d_gdn_cast_v_permute_strided_kernel(
        const float * __restrict__ src, nv_bfloat16 * __restrict__ dst,
        int64_t n_seqs, int64_t P, int H, int Hg, int R, int V,
        int64_t s1, int64_t s2, int64_t s3) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_seqs * P * H * V;
    if (i >= total) {
        return;
    }
    const int    v_  = (int) (i % V);
    size_t       tmp = i / V;
    const int    h   = (int) (tmp % H);
    tmp /= H;
    const int    t   = (int) (tmp % P);
    const int64_t n  = tmp / P;
    const int    hp  = r4d_gdn_perm_head(h, Hg, R);
    const float val = src[(size_t) n * s3 + (size_t) t * s2 + (size_t) h * s1 + v_];
    dst[((size_t) n * P + t) * H * V + (size_t) hp * V + v_] = __float2bfloat16(val);
}

// Kernel I: bf16 -> f32 cast of the chunk_scan output (canonical, permuted [n_seqs,P,H,V] layout)
// back into dst_d's strided [0,P) rows, unpermuting the H axis (same perm() formula both ways).
__global__ void r4d_gdn_cast_o_unpermute_strided_kernel(
        const nv_bfloat16 * __restrict__ src, float * __restrict__ dst,
        int64_t n_seqs, int64_t P, int H, int Hg, int R, int V, int64_t dst_seq_stride) {
    const size_t j = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) n_seqs * P * H * V;
    if (j >= total) {
        return;
    }
    const int    v_  = (int) (j % V);
    size_t       tmp = j / V;
    const int    h   = (int) (tmp % H);
    tmp /= H;
    const int    t   = (int) (tmp % P);
    const int64_t n  = tmp / P;
    const int    hp  = r4d_gdn_perm_head(h, Hg, R);
    const nv_bfloat16 val = src[((size_t) n * P + t) * H * V + (size_t) hp * V + v_];
    dst[(size_t) n * dst_seq_stride * H * V + (size_t) t * H * V + (size_t) h * V + v_] = __bfloat162float(val);
}

// Kernel J: per-chunk (64-token, reset at the sequence's own start -- always t==0 here, the
// prefix always begins at the sequence's first token) inclusive prefix sum of raw g, PLUS a
// straight copy of beta, both reading via g/beta's shared strides and permuting the H axis
// (h -> perm(h)) into the canonical destination. One thread per (seq, ggml-head).
__global__ void r4d_gdn_g_beta_permute_strided_kernel(
        const float * __restrict__ g_in, const float * __restrict__ beta_in,
        float * __restrict__ g_out, float * __restrict__ beta_out,
        int64_t n_seqs, int64_t P, int H, int Hg, int R, int chunk,
        int64_t s1, int64_t s2, int64_t s3) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n = blockIdx.y;
    if (h >= H || n >= n_seqs) {
        return;
    }
    const int hp = r4d_gdn_perm_head(h, Hg, R);
    const size_t base_in  = (size_t) n * s3 + (size_t) h * s1;
    float * gcol_out    = g_out    + ((size_t) n * P) * H + hp;
    float * bcol_out    = beta_out + ((size_t) n * P) * H + hp;
    float acc = 0.0f;
    for (int64_t t = 0; t < P; ++t) {
        if (t % chunk == 0) {
            acc = 0.0f;
        }
        acc += g_in[base_in + (size_t) t * s2];
        gcol_out[(size_t) t * H] = acc;
        bcol_out[(size_t) t * H] = beta_in[base_in + (size_t) t * s2];
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Diagnostics — MAD_R4D_LOG=1.
bool r4d_gdn_log_enabled() {
    static const bool e = [] { const char * v = std::getenv("MAD_R4D_LOG"); return v && v[0] == '1'; }();
    return e;
}
void r4d_gdn_log_once(int T, int N, int H, int Hg, int kkt_rc, int scan_rc) {
    if (!r4d_gdn_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::tuple<int, int, int, int>> seen;
    const auto key = std::make_tuple(T, N, H, Hg);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
        return;
    }
    std::fprintf(stderr,
        "[mt_gdn_r4d] T=%d N=%d H=%d Hg=%d path=prefill(chunked, %d chunks) kkt_rc=%d scan_rc=%d\n",
        T, N, H, Hg, T / 64, kkt_rc, scan_rc);
}
void r4d_gdn_log_reject_once(const char * reason, const char * fmt, ...) {
    if (!r4d_gdn_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::string> seen;
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(reason).second) {
        return;
    }
    char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    std::fprintf(stderr, "[mt_gdn_r4d] REJECT reason=%s %s\n", reason, buf);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
// Runtime gate
// ─────────────────────────────────────────────────────────────────────────
bool r4d_gdn_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_USE_R4D_GDN");
        const bool   opted_in = env != nullptr && env[0] == '1';
        return opted_in && ggml_cuda_r4d_available();
    }();
    return enabled;
}

// ─────────────────────────────────────────────────────────────────────────
// Dispatch entry
// ─────────────────────────────────────────────────────────────────────────
bool ggml_cuda_op_gated_delta_net_r4d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_q     = dst->src[0];
    const ggml_tensor * src_k     = dst->src[1];
    const ggml_tensor * src_v     = dst->src[2];
    const ggml_tensor * src_g     = dst->src[3];
    const ggml_tensor * src_beta  = dst->src[4];
    const ggml_tensor * src_state = dst->src[5];

    const int64_t S_v      = src_v->ne[0];   // head_v_dim
    const int64_t H        = src_v->ne[1];   // value-head count
    const int64_t T        = src_v->ne[2];   // tokens per seq
    const int64_t N        = src_v->ne[3];   // sequences
    const int64_t K_dim    = src_q->ne[0];   // head_k_dim
    const int64_t Hg       = src_q->ne[1];   // query/key-head count

    // ── Eligibility (return false => caller falls through to the existing implementation) ──────
    const bool kda = (src_g->ne[0] == S_v);
    if (kda) {
        r4d_gdn_log_reject_once("kda_gate", "src_g->ne[0]=%lld (per-channel/KDA gate not supported; "
                                 "libr4d GDN kernels take a scalar-per-head gate only)", (long long) src_g->ne[0]);
        return false;
    }
    if (K_dim != 128 || S_v != 128) {
        r4d_gdn_log_reject_once("head_dim", "head_k_dim=%lld head_v_dim=%lld (need 128/128)",
                                 (long long) K_dim, (long long) S_v);
        return false;
    }
    if (Hg <= 0 || H <= 0 || H % Hg != 0) {
        r4d_gdn_log_reject_once("gqa_repeat_not_integral",
                                 "Hg=%lld H=%lld (H must be a positive integer multiple of Hg for the "
                                 "head-convention permutation to be well-defined)",
                                 (long long) Hg, (long long) H);
        return false;
    }
    const int R = (int) (H / Hg);   // GQA repeat; R==1 (Hg==H) makes perm() the identity
    const int K_snapshots = ggml_get_op_params_i32(dst, 0);
    if (K_snapshots > 1) {
        r4d_gdn_log_reject_once("keep_rs_snapshots", "op_params[0]=%d (K-snapshot fusion not supported)",
                                 K_snapshots);
        return false;
    }
    if (!ggml_is_contiguous(src_q) || !ggml_is_contiguous(src_k) || !ggml_is_contiguous(src_v) ||
        !ggml_is_contiguous(src_g) || !ggml_is_contiguous(src_beta) || !ggml_is_contiguous(src_state)) {
        r4d_gdn_log_reject_once("non_contiguous", "one of q/k/v/g/beta/state is not fully contiguous");
        return false;
    }
    if (src_q->type != GGML_TYPE_F32 || src_k->type != GGML_TYPE_F32 || src_v->type != GGML_TYPE_F32 ||
        src_g->type != GGML_TYPE_F32 || src_beta->type != GGML_TYPE_F32 || src_state->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32) {
        r4d_gdn_log_reject_once("dtype", "expected f32 throughout (this op's only supported dtype)");
        return false;
    }
    if (N <= 0 || T <= 0 || H <= 0) {
        r4d_gdn_log_reject_once("degenerate_shape", "N=%lld T=%lld H=%lld", (long long) N, (long long) T, (long long) H);
        return false;
    }
    // T (this call's per-sequence token count -- ne[2], uniform across all N sequences by
    // construction, see the layout algebra in the file header) must be a whole multiple of
    // libr4d's 64-token chunk: chunk_scan's h0/ht carry is only bit-exact at that boundary
    // (measured directly, see the file header's "Eligibility: T must be a whole multiple of the
    // 64-token chunk" section) -- not a marshalling issue, a property of the chunked algorithm.
    // T==1 (single-token decode) fails this trivially and always falls through here; ggml's own
    // gated_delta_net_cuda already serves decode exactly, one token at a time.
    if (T % 64 != 0) {
        r4d_gdn_log_reject_once("tail_not_chunk_aligned",
                                 "T=%lld is not a multiple of 64 (libr4d's chunk_scan state carry is "
                                 "only exact at chunk boundaries; includes T==1 decode)", (long long) T);
        return false;
    }

    // Capture-safety warm-up: same reasoning as mt_pagedattn_r4d.cu — the first-ever call must be
    // eager (not under HIP graph capture) so a getenv()-backed memo inside libr4d's launchers (if
    // any) resolves outside capture. This op never mutates persistent state before this check, so
    // deferring here is always a clean, side-effect-free `return false`.
    static std::atomic<bool> g_r4d_gdn_warmed_up{false};
    cudaStream_t stream = ctx.stream();
    if (!g_r4d_gdn_warmed_up.load(std::memory_order_acquire)) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        const bool is_capturing = cudaStreamIsCapturing(stream, &cap) == cudaSuccess
                                   && cap != cudaStreamCaptureStatusNone;
        if (is_capturing) {
            r4d_gdn_log_reject_once("capture_warmup_deferral",
                                     "first-ever call landed inside HIP graph capture; deferring warm-up");
            return false;
        }
        int dims_head_k, dims_head_v, dims_chunk;
        r4d_gdn_dims(&dims_head_k, &dims_head_v, &dims_chunk);
        GGML_ASSERT(dims_head_k == 128 && dims_head_v == 128 && dims_chunk == 64 &&
                    "R4D GDN library geometry does not match this integration's assumptions");
    }

    // ── Past this point: no allocation/sync-unsafe surprises, but also nothing irreversible has
    // happened yet (unlike mt_pagedattn_r4d, there is no shared-cache scatter) — a launch-time
    // rejection below is still treated as a hard abort rather than a fallback, because dst may
    // already be partially written by the time libr4d's second kernel runs. ─────────────────────
    const int dev = ctx.device;
    const int64_t T_total = T * N;
    const int     bt      = 64;
    const int     VK      = (int) (S_v * K_dim);

    const float * q_d    = (const float *) src_q->data;
    const float * k_d    = (const float *) src_k->data;
    const float * v_d    = (const float *) src_v->data;
    const float * g_d    = (const float *) src_g->data;
    const float * beta_d = (const float *) src_beta->data;
    const float * h0_d   = (const float *) src_state->data;
    float *       dst_d  = (float *) dst->data;
    float *       ht_d   = dst_d + S_v * H * T * N;   // state tail, same layout as h0 (K_snapshots<=1)

    nv_bfloat16 * q_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_Q_BF16,    (size_t) T_total * Hg * K_dim);
    nv_bfloat16 * k_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_K_BF16,    (size_t) T_total * Hg * K_dim);
    nv_bfloat16 * v_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_V_BF16,    (size_t) T_total * H  * S_v);
    nv_bfloat16 * out_bf16  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_OUT_BF16,  (size_t) T_total * H  * S_v);
    float *       g_cumsum  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_G_CUMSUM,  (size_t) T_total * H);
    float *       beta_perm = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_BETA_PERM, (size_t) T_total * H);
    float *       h0_perm   = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_H0_PERM,   (size_t) N * H * VK);
    float *       ht_perm   = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_HT_PERM,   (size_t) N * H * VK);
    nv_bfloat16 * A_scratch  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_A_SCRATCH, (size_t) T_total * H * bt);
    int32_t *     cu_seqlens = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_CU_SEQLENS, (size_t) N + 1);

    constexpr int CAST_THREADS = 256;

    // q, k: flat cast, no permutation (indexed by Hg).
    {
        const size_t n = (size_t) T_total * Hg * K_dim;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_f32_to_bf16_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(q_d, q_bf16, n);
        r4d_gdn_cast_f32_to_bf16_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(k_d, k_bf16, n);
    }
    // v: permuting cast, ggml order -> libr4d order.
    {
        const size_t n = (size_t) T_total * H * S_v;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_v_permute_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            v_d, v_bf16, (size_t) T_total, (int) H, (int) Hg, R, (int) S_v);
    }
    // g (cumsum) + beta: permuting, combined.
    {
        const dim3 grid((unsigned) ((H + 63) / 64), (unsigned) N);
        r4d_gdn_g_beta_permute_kernel<<<grid, 64, 0, stream>>>(
            g_d, beta_d, g_cumsum, beta_perm, (int) N, (int) T, (int) H, (int) Hg, R, bt);
    }
    // h0: permuting fp32 copy, ggml order -> libr4d order.
    {
        const dim3 grid((unsigned) H, (unsigned) N);
        r4d_gdn_state_permute_kernel<<<grid, 256, 0, stream>>>(
            h0_d, h0_perm, (int) N, (int) H, (int) Hg, R, VK, /*src_is_ggml_order=*/true);
    }
    r4d_gdn_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(cu_seqlens, (int) N, (int) T);

    const float scale = 1.0f / sqrtf((float) S_v);

    const int kkt_rc = r4d_gdn_kkt_solve_k128_c64_bf16(
        k_bf16, beta_perm, g_cumsum, A_scratch, cu_seqlens,
        (int) N, (int) T_total, (int) H, (int) Hg, (int) K_dim, bt, stream);
    if (kkt_rc != 0) {
        GGML_ABORT("mt_gdn_r4d: r4d_gdn_kkt_solve_k128_c64_bf16 rejected shape (rc=%d, N=%lld T=%lld "
                   "H=%lld Hg=%lld K=%lld)", kkt_rc, (long long) N, (long long) T_total, (long long) H,
                   (long long) Hg, (long long) K_dim);
    }

    const int scan_rc = r4d_gdn_chunk_scan_k128_v128_c64_bf16(
        q_bf16, k_bf16, v_bf16, A_scratch, g_cumsum, beta_perm, h0_perm, out_bf16, ht_perm, cu_seqlens,
        (int) N, (int) H, (int) Hg, (int) K_dim, (int) S_v, bt, scale, stream);
    if (scan_rc != 0) {
        GGML_ABORT("mt_gdn_r4d: r4d_gdn_chunk_scan_k128_v128_c64_bf16 rejected shape (rc=%d, N=%lld "
                   "T=%lld H=%lld Hg=%lld K=%lld V=%lld)", scan_rc, (long long) N, (long long) T_total,
                   (long long) H, (long long) Hg, (long long) K_dim, (long long) S_v);
    }

    // o: permuting cast back, libr4d order -> ggml order, straight into dst's head block.
    {
        const size_t n = (size_t) T_total * H * S_v;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_o_unpermute_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            out_bf16, dst_d, (size_t) T_total, (int) H, (int) Hg, R, (int) S_v);
    }
    // ht: permuting fp32 copy back, libr4d order -> ggml order, into dst's state tail.
    {
        const dim3 grid((unsigned) H, (unsigned) N);
        r4d_gdn_state_permute_kernel<<<grid, 256, 0, stream>>>(
            ht_perm, ht_d, (int) N, (int) H, (int) Hg, R, VK, /*src_is_ggml_order=*/false);
    }

    g_r4d_gdn_warmed_up.store(true, std::memory_order_release);
    r4d_gdn_log_once((int) T, (int) N, (int) H, (int) Hg, kkt_rc, scan_rc);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Production entry: ggml_cuda_op_gated_delta_net_impl's use_prefill_chunked branch (see
// mt_gdn_r4d.cuh for the full contract this implements).
// ─────────────────────────────────────────────────────────────────────────
bool ggml_cuda_gdn_r4d_prefix(
        ggml_backend_cuda_context & ctx, ggml_tensor * /*dst*/,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * prefix_state_out,
        int64_t S_v, int64_t H, int64_t P, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t dst_seq_stride, cudaStream_t stream) {
    const int64_t Hg = neqk1;
    const int64_t H_ = H;

    // ── Eligibility (return false => caller falls through to the existing implementation for
    // the WHOLE [0,T0) window, unchanged -- this function has not touched anything yet). ───────
    if (S_v != 128) {
        r4d_gdn_log_reject_once("prefix_head_dim", "S_v=%lld (need 128)", (long long) S_v);
        return false;
    }
    if (n_seqs <= 0 || Hg <= 0 || H_ <= 0 || H_ % Hg != 0) {
        r4d_gdn_log_reject_once("prefix_gqa_repeat_not_integral",
                                 "n_seqs=%lld H=%lld Hg=%lld", (long long) n_seqs, (long long) H_, (long long) Hg);
        return false;
    }
    if (rq3 != 1) {
        // MTP-verify batch-broadcast (q/k have fewer "sequences" than v/state, see
        // ggml_cuda_op_gated_delta_net_impl's `rq3 = nev3/neq3`): this file's strided kernels
        // below assume `n` indexes q/k/v/g/beta identically, which only holds at rq3==1. Declined
        // rather than risked; the existing kernels handle this case already via iq3=fastdiv.
        r4d_gdn_log_reject_once("prefix_mtp_batch_repeat", "rq3=%lld (need 1)", (long long) rq3);
        return false;
    }
    if (P <= 0 || P % 64 != 0) {
        r4d_gdn_log_reject_once("prefix_not_chunk_aligned", "P=%lld (need a positive multiple of 64)",
                                 (long long) P);
        return false;
    }
    // Head-contiguity only: sq1/sv1/sb1 are used as raw per-element offsets within a token's row
    // (h*s1 + k, h*s1 + v_, h*s1) in r4d_gdn_cast_qk_strided_kernel / r4d_gdn_cast_v_permute_
    // strided_kernel / r4d_gdn_g_beta_permute_strided_kernel -- the ONE thing those kernels
    // actually assume is that a head's own K (or V) elements are contiguous (unit stride), which
    // is exactly sq1==S_v / sv1==S_v / sb1==1. The per-token stride (sq2/sv2/sb2) and per-sequence
    // stride (sq3/sv3/sb3) are used AS GIVEN in every read (n*s3 + t*s2 + h*s1 + ...) -- these
    // kernels never assume a canonical product-of-dims value for them, so a fused-projection view
    // (q|k|v sharing one buffer, sv2 = S_v*(Hg+Hg+H) rather than S_v*H, as seen in production:
    // sv2=10240 = 128*(16+16+48)) is handled correctly and must NOT be rejected here. Previously
    // this check also required sq2==S_v*Hg / sv2==S_v*H / sb2==H, which was stricter than what the
    // kernels need and rejected every real production call (MAD-406 follow-up).
    if (sq1 != S_v || sv1 != S_v || sb1 != 1) {
        r4d_gdn_log_reject_once("prefix_head_not_contiguous",
                                 "sq1=%lld sv1=%lld sb1=%lld (need S_v=%lld / S_v / 1)",
                                 (long long) sq1, (long long) sv1, (long long) sb1, (long long) S_v);
        return false;
    }

    const int R = (int) (H_ / Hg);

    // Same capture-safety warm-up as ggml_cuda_op_gated_delta_net_r4d -- shared static, since
    // both functions ultimately touch the same libr4d launchers and the memo they warm is
    // process-global, not per-entry-point.
    static std::atomic<bool> g_r4d_gdn_prefix_warmed_up{false};
    if (!g_r4d_gdn_prefix_warmed_up.load(std::memory_order_acquire)) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        const bool is_capturing = cudaStreamIsCapturing(stream, &cap) == cudaSuccess
                                   && cap != cudaStreamCaptureStatusNone;
        if (is_capturing) {
            r4d_gdn_log_reject_once("prefix_capture_warmup_deferral",
                                     "first-ever call landed inside HIP graph capture; deferring warm-up");
            return false;
        }
        int dims_head_k, dims_head_v, dims_chunk;
        r4d_gdn_dims(&dims_head_k, &dims_head_v, &dims_chunk);
        GGML_ASSERT(dims_head_k == 128 && dims_head_v == 128 && dims_chunk == 64 &&
                    "R4D GDN library geometry does not match this integration's assumptions");
    }

    // ── Past this point: commit. ────────────────────────────────────────────────────────────
    const int dev = ctx.device;
    const int64_t T_total = P * n_seqs;
    const int     bt      = 64;
    const int     VK      = (int) (S_v * S_v);

    nv_bfloat16 * q_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_Q_BF16,    (size_t) T_total * Hg * S_v);
    nv_bfloat16 * k_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_K_BF16,    (size_t) T_total * Hg * S_v);
    nv_bfloat16 * v_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_V_BF16,    (size_t) T_total * H_ * S_v);
    nv_bfloat16 * out_bf16  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_OUT_BF16,  (size_t) T_total * H_ * S_v);
    float *       g_cumsum  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_G_CUMSUM,  (size_t) T_total * H_);
    float *       beta_perm = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_BETA_PERM, (size_t) T_total * H_);
    float *       h0_perm   = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_H0_PERM,   (size_t) n_seqs * H_ * VK);
    float *       ht_perm   = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_HT_PERM,   (size_t) n_seqs * H_ * VK);
    nv_bfloat16 * A_scratch  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_A_SCRATCH, (size_t) T_total * H_ * bt);
    int32_t *     cu_seqlens = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_CU_SEQLENS, (size_t) n_seqs + 1);

    constexpr int CAST_THREADS = 256;

    // q, k: strided cast, no permutation.
    {
        const size_t n = (size_t) T_total * Hg * S_v;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_qk_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            q_d, q_bf16, n_seqs, P, (int) Hg, (int) S_v, sq1, sq2, sq3);
        r4d_gdn_cast_qk_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            k_d, k_bf16, n_seqs, P, (int) Hg, (int) S_v, sq1, sq2, sq3);
    }
    // v: strided permuting cast.
    {
        const size_t n = (size_t) T_total * H_ * S_v;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_v_permute_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            v_d, v_bf16, n_seqs, P, (int) H_, (int) Hg, R, (int) S_v, sv1, sv2, sv3);
    }
    // g (cumsum) + beta: strided, permuting, combined.
    {
        const dim3 grid((unsigned) ((H_ + 63) / 64), (unsigned) n_seqs);
        r4d_gdn_g_beta_permute_strided_kernel<<<grid, 64, 0, stream>>>(
            g_d, b_d, g_cumsum, beta_perm, n_seqs, P, (int) H_, (int) Hg, R, bt, sb1, sb2, sb3);
    }
    // h0: permuting fp32 copy (canonical -- ggml's state tensor is always fully contiguous,
    // guaranteed by ggml_cuda_op_gated_delta_net_impl's own is_contiguous(src_state) assert).
    {
        const dim3 grid((unsigned) H_, (unsigned) n_seqs);
        r4d_gdn_state_permute_kernel<<<grid, 256, 0, stream>>>(
            s_d, h0_perm, (int) n_seqs, (int) H_, (int) Hg, R, VK, /*src_is_ggml_order=*/true);
    }
    r4d_gdn_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(cu_seqlens, (int) n_seqs, (int) P);

    const int kkt_rc = r4d_gdn_kkt_solve_k128_c64_bf16(
        k_bf16, beta_perm, g_cumsum, A_scratch, cu_seqlens,
        (int) n_seqs, (int) T_total, (int) H_, (int) Hg, (int) S_v, bt, stream);
    if (kkt_rc != 0) {
        GGML_ABORT("mt_gdn_r4d: r4d_gdn_kkt_solve_k128_c64_bf16 (prefix) rejected shape (rc=%d, "
                   "n_seqs=%lld P=%lld H=%lld Hg=%lld)", kkt_rc, (long long) n_seqs, (long long) P,
                   (long long) H_, (long long) Hg);
    }
    const int scan_rc = r4d_gdn_chunk_scan_k128_v128_c64_bf16(
        q_bf16, k_bf16, v_bf16, A_scratch, g_cumsum, beta_perm, h0_perm, out_bf16, ht_perm, cu_seqlens,
        (int) n_seqs, (int) H_, (int) Hg, (int) S_v, (int) S_v, bt, scale, stream);
    if (scan_rc != 0) {
        GGML_ABORT("mt_gdn_r4d: r4d_gdn_chunk_scan_k128_v128_c64_bf16 (prefix) rejected shape (rc=%d, "
                   "n_seqs=%lld P=%lld H=%lld Hg=%lld)", scan_rc, (long long) n_seqs, (long long) P,
                   (long long) H_, (long long) Hg);
    }

    // o: strided permuting cast back, straight into dst_d's [0,P) rows.
    {
        const size_t n = (size_t) T_total * H_ * S_v;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_o_unpermute_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            out_bf16, dst_d, n_seqs, P, (int) H_, (int) Hg, R, (int) S_v, dst_seq_stride);
    }
    // ht: permuting fp32 copy back into prefix_state_out (canonical [n_seqs,H,V,K]).
    {
        const dim3 grid((unsigned) H_, (unsigned) n_seqs);
        r4d_gdn_state_permute_kernel<<<grid, 256, 0, stream>>>(
            ht_perm, prefix_state_out, (int) n_seqs, (int) H_, (int) Hg, R, VK, /*src_is_ggml_order=*/false);
    }

    g_r4d_gdn_prefix_warmed_up.store(true, std::memory_order_release);
    return true;
}

void r4d_gdn_prefix_log(int64_t P, int64_t n_tokens, int K, bool handled) {
    if (!r4d_gdn_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::tuple<int64_t, int64_t, int>> seen;
    const auto key = std::make_tuple(P, n_tokens, K);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
        return;
    }
    std::fprintf(stderr,
        "[mt_gdn_r4d] use_prefill_chunked P=%lld n_tokens=%lld K=%d r4d_prefix=%s\n",
        (long long) P, (long long) n_tokens, K, handled ? "yes" : "no");
}

#endif  // GGML_HIP_R4D
