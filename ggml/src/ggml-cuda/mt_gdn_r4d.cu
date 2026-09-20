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
//
// ── ggml <-> radiance GDN op mapping (MAD-406 op-mapping task, 2026-09-19) ─────────────────────
//
// radiance (radiance_gdn.py's forward_core_fused, "both" path)         | ggml (qwen35.cpp build_layer_attn_linear)
// ----------------------------------------------------------------------+----------------------------------------------
// conv_prep() -> r4d_gdn_conv_prep_w4_h128_bf16 (ONE kernel):          | FIVE separate ops/graph steps:
//   causal conv (width 4) + SiLU                                       |  ggml_ssm_conv() (GGML_OP_SSM_CONV) + ggml_silu()
//   q/k/v split                                                        |  ggml_view_4d() x3 (q_conv/k_conv/v_conv, free views)
//   q/k L2-norm                                                        |  build_gdn_l2_norm() (RMS-style norm, no learned scale)
//   g = -exp(A_log)*softplus(a+dt_bias), per-chunk cumsum              |  ggml_add/ggml_softplus/ggml_mul on a SEPARATE branch:
//                                                                       |    alpha = build_lora_mm(ssm_alpha, cur); + ssm_dt;
//                                                                       |    softplus; * ssm_a  -- never touches conv_x at all
//   beta = sigmoid(b)                                                  |  beta = sigmoid(build_lora_mm(ssm_beta, cur))
//   (cumsum of g is folded into the same kernel; ggml's own g is raw)  |  no cumsum node -- gated_delta_net.cu's own kernels
//                                                                       |    (and this file's g_beta_permute kernels) do it
// kkt_solve() -> r4d_gdn_kkt_solve_k128_c64_bf16                       |  no ggml equivalent -- part of GATED_DELTA_NET's
//                                                                       |    internal chunked-prefill computation
// fused_prefill() -> r4d_gdn_chunk_scan_k128_v128_c64_bf16             |  GGML_OP_GATED_DELTA_NET (gated_delta_net.cu),
//                                                                       |    use_prefill_chunked branch -> THIS file's
//                                                                       |    ggml_cuda_gdn_r4d_prefix (already wired)
// output_norm() -> r4d_gdn_gated_rmsnorm_h128_bf16 (ONE kernel):       |  build_norm_gated() (qwen35.cpp): FOUR separate ops
//   rms(x) . w . silu(z), one row per (token, head)                    |    GGML_OP_RMS_NORM, GGML_OP_MUL (weight), GGML_OP_SILU
//                                                                       |    (on z), GGML_OP_MUL (elementwise with silu(z))
//
// ── The conv-side hook: investigated, DECLINES (op boundary, not a marshalling gap) ────────────
//
// GGML_OP_SSM_CONV's own src[] is exactly two tensors: dst->src[0] = conv_x (the pre-conv
// [conv_channels, n_seq_tokens, n_seqs] activation -- the concatenated q|k|v channels BEFORE the
// conv, i.e. radiance's `x`/mixed_qkv) and dst->src[1] = the conv1d weight (ggml_ssm_conv() in
// ggml.c). ggml-cuda.cu's existing SSM_CONV+bias+SiLU fusion detector additionally threads two
// SIBLING nodes into ggml_cuda_op_ssm_conv (bias_add_node, silu_dst) -- but those are the bias
// tensor and the SiLU destination, not the gate inputs.
//
// r4d_gdn_conv_prep_w4_h128_bf16 (r4d.h) ALSO needs `a`, `b` (raw pre-activation alpha/beta,
// [n_seq_tokens, num_v_heads], one value per token per head) and `A_log`, `dt_bias`
// (model.layers[il].ssm_a / ssm_dt, one value per head). In ggml's graph these come from a
// completely disjoint branch of qwen35.cpp's build_layer_attn_linear:
//     alpha = build_lora_mm(ssm_alpha, cur, ...);  alpha_biased = alpha + ssm_dt;
//     alpha_softplus = softplus(alpha_biased);     gate = alpha_softplus * ssm_a;   (= -A_log.exp()*softplus(...))
//     beta = sigmoid(build_lora_mm(ssm_beta, cur, ...));
// computed straight off `cur` (the layer's hidden-state input), NEVER routed through
// conv_input/ssm_conv1d at all -- so at the point GGML_OP_SSM_CONV executes, `a`/`b` may not even
// have been computed yet (ggml's scheduler is free to order independent subgraphs either way), and
// even if they had been, this op's dispatch contract (ggml_cuda_op_ssm_conv(ctx, dst,
// bias_add_node, silu_dst) in ggml-cuda.cu) has no slot for them.
//
// Making them reachable would require: (a) ggml-cuda.cu's fusion-detection pass recognising the
// SSM_CONV -> alpha/beta chain as one fusable unit and passing the extra tensors through (a
// dispatcher change), and/or (b) qwen35.cpp restructuring build_layer_attn_linear so the gate
// computation is graph-adjacent to the conv node in a way the detector can find it (a graph-builder
// change). Both are outside this integration's file ownership (mt_gdn_r4d.cu/.cuh, a gate line in
// ssm-conv.cu, tests/test-r4d-gdn.hip.cpp) and outside "adapters only, no kernels beyond trivial
// cast/copy helpers" scope regardless. So ggml_cuda_op_ssm_conv_r4d below is wired (the gate line
// in ssm-conv.cu calls it whenever MAD_USE_R4D_GDN_CONV=1) but always declines, logging the reason
// once under MAD_R4D_LOG=1 -- a clean, side-effect-free `return false` exactly like every other
// eligibility miss in this file, so the existing ggml_cuda_op_ssm_conv body runs unchanged.
//
// ── What was done instead: cut redundant launches from the adapter's own overhead ────────────────
//
// libr4d's kkt_solve/chunk_scan C ABI (r4d.h) takes dense pointers, not strided views -- neither
// entry point has a stride argument for q/k/v/A/g/beta/h0/ht, unlike conv_prep (which takes
// `xpitch`/`ab_stride` because it is meant to read straight out of a strided qkv projection
// buffer). So this file cannot hand kkt_solve/chunk_scan a strided view directly no matter how the
// glue is organized -- every input still needs a real cast-and-repack pass into libr4d's own dense
// bf16 layout; the only thing this pass could reduce was the NUMBER of launches doing that
// repacking, not the repacking itself.
//
// First pass (MAD-406 follow-up): q and k were cast via two separate invocations of the same
// strided kernel (kernel G) even though they share identical strides (sq1/sq2/sq3) and never
// interact. r4d_gdn_cast_qk_combined_strided_kernel (and its flat counterpart,
// r4d_gdn_cast_qk_combined_kernel, for ggml_cuda_op_gated_delta_net_r4d's non-production entry)
// folded both into one launch, addressed by a leading "which" bit over double the element count.
//
// Second pass (glue-reduction follow-up, this comment's revision): v's permuting cast (kernel H)
// was still a separate launch from the qk-combined one even though production always calls this
// op with q/k/v as three strided views into ONE fused qkv projection buffer (sv2 = S_v*(2*Hg+H),
// matching kernel G3's per_qk/per_v split below exactly) -- there was no remaining reason for the
// GPU to see them as two dispatches. r4d_gdn_cast_qkv_combined_strided_kernel (kernel G3) merges
// qk-combined and v-permute into ONE launch over the concatenated q|k|v index range; and kernel F
// (cu_seqlens, a <<<1,1>>> launch whose only job is to fill an N+1-entry int32 array the h0-permute
// launch could trivially do itself) was folded into kernel E's (h0 permute-in) own grid as kernel
// F2, spending one extra grid column instead of a whole extra launch. Together these remove TWO of
// the ~6 kernel launches this file used to issue per GDN layer per non-primed chunked-prefill call
// (qkv-cast-and-permute, g+beta, h0-permute-and-cu_seqlens, kkt_solve, chunk_scan, o-unpermute-
// cast, ht-permute-out is now 7 launches where kkt_solve/chunk_scan/ht-permute-out were already
// there and o-unpermute-cast is skippable under RAW_OUT -- down from 9 before this pass, 6 of them
// on the ggml side of the libr4d call and now 4). Neither pass changes the per-element math any
// kernel performs (same read, same cast, same permutation formula, same write, just addressed out
// of a wider launch) -- see tests/test-r4d-gdn.hip.cpp's run_adapter_overhead_timing() for the
// measured per-stage breakdown at the production N=1/T=1024/H=48/Hg=16 shape.
//
// What could NOT be removed within this file's ownership: the g+beta kernel (already one launch,
// but the cumsum it computes is inherently a serial per-(seq,head) scan over up to 1024 tokens, so
// it cannot be folded into the qkv-cast launch's flat element-parallel indexing without either
// giving up the running accumulator or launching a second pass); and, per libr4d's dense-pointer
// ABI above, no launch that feeds kkt_solve/chunk_scan can be eliminated by passing a stride
// instead -- libr4d does not accept one there.

#include "common.cuh"
#include "mt_gdn_r4d.cuh"

#ifdef GGML_HIP_R4D

#include "r4d/ggml-r4d.h"

#include <algorithm>
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
#include <vector>


// ── MAD_R4D_HOSTPROF=1: host CPU time spent inside this adapter's entry points (decode host-cost
// investigation, chain 314). Totals printed at exit. Zero cost when unset (one branch).
#include <chrono>
#include <atomic>
namespace {
struct r4d_hostprof_slot { const char * name; std::atomic<long long> ns{0}; std::atomic<long long> n{0}; };
r4d_hostprof_slot g_r4d_hp[6] = {{"concat_elide"},{"conv_prep_fusion"},{"prefix"},{"tail"},{"query_primed"},{"take_primed"}};
bool r4d_hostprof_enabled() { static const bool e = [] { const char * v = std::getenv("MAD_R4D_HOSTPROF"); return v && *v && *v != '0'; }(); return e; }
struct r4d_hostprof_scope {
    int i; std::chrono::steady_clock::time_point t0; bool on;
    explicit r4d_hostprof_scope(int idx) : i(idx), on(r4d_hostprof_enabled()) { if (on) t0 = std::chrono::steady_clock::now(); }
    ~r4d_hostprof_scope() { if (on) { g_r4d_hp[i].ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); g_r4d_hp[i].n += 1; } }
};
struct r4d_hostprof_report { ~r4d_hostprof_report() { if (!r4d_hostprof_enabled()) return;
    for (auto & s : g_r4d_hp) fprintf(stderr, "[r4d-hostprof] %-18s calls=%lld total=%.1f ms avg=%.1f us\n", s.name, (long long) s.n.load(), s.ns.load() / 1e6, s.n.load() ? s.ns.load() / 1e3 / s.n.load() : 0.0); } } g_r4d_hostprof_report;
} // namespace
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
    // MAD-406 (R4D GDN conv_prep, task 3) — conv_prep's own inputs/outputs, all in GGML head
    // order (conv_prep never sees the GQA permutation; that is applied to its outputs afterward,
    // same as every other producer this file permutes).
    R4D_GDN_CP_X_STAGE,   // bf16, [T_total, conv_channels] — new tokens only, channel-fastest
    R4D_GDN_CP_CSTATE,    // bf16, [N, conv_channels, 3] — seeded from ggml's own SSM_CONV src[0]
                          // history rows every call; never read back across calls (task 2)
    R4D_GDN_CP_WGT_BF16,  // bf16, [conv_channels, 4] — conv1d weight, cast only (layout already matches)
    R4D_GDN_CP_HAS_INIT,  // uint8, [N] — always 1 (see task 2: history is always seeded from ggml)
    R4D_GDN_CP_CACHE_IDX, // int32, [N] — identity (slot i == sequence i; no cross-call persistence)
    R4D_GDN_CP_V_GGML,    // bf16, [T_total, H, V] — conv_prep's v output, GGML head order
    R4D_GDN_CP_G_GGML,    // f32,  [T_total, H] — conv_prep's g output (already chunk-cumsum'd), GGML order
    R4D_GDN_CP_BETA_GGML, // f32,  [T_total, H] — conv_prep's beta output, GGML order
    // MAD-406 follow-up (chain 251, ggml_cuda_gdn_r4d_tail) — small, K-row-sized scratch for the
    // DFlash snapshot tail. K is always small (the vLLM/DFlash draft width, ~8), so these are
    // tiny compared to everything above.
    R4D_GDN_TAIL_AB_PERM,  // f32, [2, K, H] — a_raw/b_raw's tail rows, permuted to libr4d head order
    R4D_GDN_TAIL_O_PERM,   // RETIRED (RAW_OUT K-tail follow-up): ggml_cuda_gdn_r4d_tail now writes
                           // its raw output straight into R4D_GDN_OUT_BF16's tail rows instead of
                           // this standalone scratch -- slot kept (unused) so no other enum value
                           // shifts index.
    R4D_GDN_TAIL_CU,       // int32, [2] — {0, K}
    R4D_GDN_TAIL_SIDX,     // int32, [K] — snapshot slot map, sidx[t] = K-1-t
    // MAD-406 follow-up (2026-09-20): A_log/dt_bias, permuted to libr4d head order. Needed ONLY
    // here -- conv_prep reads A_log/dt_bias in GGML order (matching its own un-permuted internal
    // head indexing), but r4d_gdn_recurrent_update_k128_v128_bf16_fp32state indexes A_log[hv]/
    // dt_bias[hv] by hv, the SAME libr4d-order index it uses for v/output (r4d_gdn_recurrent_
    // update_k128_v128_bf16_fp32state.hip:115) -- see ggml_cuda_gdn_r4d_tail's header comment.
    R4D_GDN_TAIL_ALOG_PERM, // f32, [H]
    R4D_GDN_TAIL_DTB_PERM,  // f32, [H]
    // 2026-09-20 follow-up (state head-order bug): the recurrent_update kernel indexes its
    // `state` argument by hv, the SAME libr4d-blocked head index it uses for q/k/v/A_log/dt_bias
    // -- but state_d (the real ggml snapshot cache ggml_cuda_gdn_r4d_tail is handed) is in GGML
    // head order end to end (every other consumer of it -- the plain autoregressive kernel's
    // curr_state/state, a later decode step's own recurrent read -- addresses it by GGML h_idx).
    // Feeding/reading state_d's bytes directly under an hv-indexed kernel silently swaps entire
    // heads' worth of [V,K] state, exactly the same bug class already fixed for A_log/dt_bias
    // (see r4d_gdn_permute_alog_dtbias_kernel's comment) but for the state tensor instead. Fix:
    // never let the recurrent_update kernel touch state_d directly -- run it against this
    // libr4d-ordered scratch (permute the GGML-order seed IN before the kernel, permute all K
    // written slots back to GGML order OUT after), same [K(=slot), H, V*K] contiguous layout the
    // kernel's own slot_stride=H*VK/head_stride=VK addressing already assumes.
    R4D_GDN_TAIL_STATE_SCRATCH, // f32, [K, H, S_v*S_v] -- libr4d head order, one slot per row
    // MAD_USE_R4D_GDN_VERIFY=1 (chain 270 follow-up): scratch for staging ggml's OWN q/k/v/g/beta
    // (via the SAME strided cast/permute kernels ggml_cuda_gdn_r4d_prefix uses when NOT primed)
    // so r4d_gdn_verify_compare can diff them directly against the primed scratch, in the same
    // libr4d/bf16 representation on both sides -- see that function.
    R4D_GDN_VERIFY_Q_BF16,
    R4D_GDN_VERIFY_K_BF16,
    R4D_GDN_VERIFY_V_BF16,
    R4D_GDN_VERIFY_G_CUMSUM,
    R4D_GDN_VERIFY_BETA_PERM,
    // MAD-406 follow-up (chain 277/278, 2026-09-20): scratch destinations for running alpha_
    // mulmat/beta_mulmat early (at SSM_CONV's dispatch turn) WITHOUT writing into their own
    // allocator-assigned dst -- see ggml_cuda_try_gdn_conv_prep_fusion's "commit" section for the
    // full root-cause writeup (running a node's real computation into its own too-early dst
    // aliases whatever the allocator still considers live at that graph position).
    R4D_GDN_ALPHA_SCRATCH, // f32, [H, T, Nseq] flat (same total count as alpha_mulmat/a_raw)
    R4D_GDN_BETA_SCRATCH,  // f32, [H, T, Nseq] flat
    // MAD_USE_R4D_GDN_RAW_OUT=1 follow-up: head_src[h] = perm(h) for pattern C's "_r4d" kernel
    // variant (rdna4_ml8_qrot_gated_norm_tiled_r4d) to read directly -- rebuilt (cheap, H<=~64
    // ints) every call that publishes a raw-output record; not persisted across calls.
    R4D_GDN_HEAD_SRC,      // int32, [H]
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
// MAD-406 (R4D GDN conv_prep, task 3) — the hand-off between ggml_cuda_try_gdn_conv_prep_fusion
// (runs at GGML_OP_SSM_CONV's dispatch turn, EARLIER in cgraph->nodes than the GATED_DELTA_NET
// node it feeds) and ggml_cuda_gdn_r4d_prefix (called later, from gated_delta_net.cu's
// use_prefill_chunked branch, for that SAME GATED_DELTA_NET node). Both run on the SAME stream
// within one cgraph dispatch pass, so "primed at SSM_CONV time, consumed at GDN time" is safe
// without any extra synchronisation — the GPU serialises same-stream launches in issue order,
// and ggml-cuda.cu's dispatch loop issues SSM_CONV's turn strictly before GDN's.
//
// Keyed by (device, stream) like the persist scratch above; the entry identifies which
// GATED_DELTA_NET node it was computed for (by pointer — cgraph tensors are stable for the
// lifetime of one build/compute pass) and the exact geometry consumers must see before they may
// trust the scratch instead of re-deriving from their own q_d/k_d/v_d/g_d/b_d pointers.
//
// MAD-406 follow-up (chain 251): conv_prep primes for the WHOLE original SSM_CONV token count T
// (e.g. 4096), but gated_delta_net.cu's use_prefill_chunked branch splits that into a chunked-
// prefill portion [0, T0) (T0 = T - K, K = the DFlash snapshot-tail length) and a K-token tail —
// so a CONSUMER now asks for P <= T (not P == T), and there are TWO consumers of the SAME primed
// entry: ggml_cuda_gdn_r4d_prefix (the [0, T0) chunk_scan portion) and ggml_cuda_gdn_r4d_tail (the
// K-token snapshot tail, run from the SAME buffers' rows [T0, T0+K)). Neither one erases the
// entry by itself any more (a previous version did, and consuming it for the prefix portion left
// nothing for the tail to find) — the caller (gated_delta_net.cu) releases it explicitly, once,
// after BOTH portions of this GATED_DELTA_NET call are done (r4d_gdn_conv_prep_release_primed).
//
// a_raw_data/b_raw_data/ab_stride/a_log_data/dt_bias_data are captured here too: they are real,
// valid ggml tensor data regardless of the fusion (a_raw/b_raw are RESHAPE views over
// alpha_mulmat/beta_mulmat's output, and RESHAPE never moves data — see the fusion detector's own
// comment on this — so their buffers are valid whether or not their own graph nodes' "compute"
// steps were skipped), so ggml_cuda_gdn_r4d_tail can read the K tail rows of raw a/b directly
// instead of needing its own primed scratch for them (conv_prep's own scratch only holds the
// ALREADY-GATED, cumsum'd g/beta — not useful to a kernel that computes its own gate from raw
// a/b, like r4d_gdn_recurrent_update_k128_v128_bf16_fp32state does).
struct r4d_gdn_conv_prep_primed {
    const ggml_tensor * gdn    = nullptr;
    int64_t              T      = 0;
    int64_t              n_seqs = 0;
    int64_t              H      = 0;
    int64_t              Hg     = 0;
    const void *          a_raw_data   = nullptr;
    const void *          b_raw_data   = nullptr;
    int64_t               ab_stride    = 0;
    const void *          a_log_data   = nullptr;
    const void *          dt_bias_data = nullptr;
    // MAD-406 follow-up (chain 279, 2026-09-20 verify bisection): the REAL a_raw/b_raw ggml
    // tensors (not just their .data, which -- when alpha_needs_exec/beta_needs_exec -- is OUR
    // scratch, not theirs). In MAD_USE_R4D_GDN_VERIFY=1 mode, alpha_mulmat/beta_mulmat's normal
    // turn still runs later (their nodes are never skipped there), and it happens strictly BEFORE
    // GATED_DELTA_NET's own turn (GDN topologically depends on them) -- so by the time
    // r4d_gdn_verify_compare runs, a_raw_tensor->data/b_raw_tensor->data hold the REAL,
    // independently-computed values, letting that function cross-check our early scratch
    // snapshot against them directly.
    const ggml_tensor *   a_raw_tensor = nullptr;
    const ggml_tensor *   b_raw_tensor = nullptr;
};
static std::mutex g_r4d_gdn_primed_mutex;
static std::map<std::pair<int, cudaStream_t>, r4d_gdn_conv_prep_primed> g_r4d_gdn_primed;

static void r4d_gdn_conv_prep_mark_primed(int device, cudaStream_t stream, const ggml_tensor * gdn,
                                           int64_t T, int64_t n_seqs, int64_t H, int64_t Hg,
                                           const void * a_raw_data, const void * b_raw_data, int64_t ab_stride,
                                           const void * a_log_data, const void * dt_bias_data,
                                           const ggml_tensor * a_raw_tensor, const ggml_tensor * b_raw_tensor) {
    std::lock_guard<std::mutex> lock(g_r4d_gdn_primed_mutex);
    g_r4d_gdn_primed[std::make_pair(device, stream)] = r4d_gdn_conv_prep_primed{
        gdn, T, n_seqs, H, Hg, a_raw_data, b_raw_data, ab_stride, a_log_data, dt_bias_data,
        a_raw_tensor, b_raw_tensor};
}

// Returns true (and copies the entry into *out, WITHOUT erasing it — see the struct's header
// comment: two independent consumers share one primed entry per call now) iff a primed entry
// exists for (device, stream) whose gdn pointer matches and whose span covers at least
// `P + K_tail` rows with matching n_seqs/H/Hg. A geometry MISMATCH on a matching gdn pointer is a
// design-invariant violation, not a normal miss — it would mean the two sides of this hand-off
// disagree about this call's shape — so it aborts loudly instead of silently consuming a
// mismatched buffer.
static bool r4d_gdn_conv_prep_take_primed(int device, cudaStream_t stream, const ggml_tensor * gdn,
                                           int64_t P, int64_t K_tail, int64_t n_seqs, int64_t H, int64_t Hg,
                                           r4d_gdn_conv_prep_primed * out) {
    r4d_hostprof_scope r4d_hp_(5);
    std::lock_guard<std::mutex> lock(g_r4d_gdn_primed_mutex);
    auto it = g_r4d_gdn_primed.find(std::make_pair(device, stream));
    if (it == g_r4d_gdn_primed.end() || it->second.gdn != gdn) {
        return false;
    }
    const r4d_gdn_conv_prep_primed p = it->second;
    GGML_ASSERT(p.T >= P + K_tail && p.n_seqs == n_seqs && p.H == H && p.Hg == Hg &&
                "mt_gdn_r4d: conv_prep primed a GATED_DELTA_NET call with a different geometry "
                "(or too few rows) than the one it is now being consumed for");
    if (out != nullptr) {
        *out = p;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Kernel A2: combined flat f32 -> bf16 cast of q AND k in ONE launch (same n_elems, no
// permutation -- q/k are indexed by Hg and never permuted). Top half of the index range [0,n) is
// q, [n,2n) is k. Used by ggml_cuda_op_gated_delta_net_r4d (the non-production entry) in place of
// two separate flat-cast launches (kernel A/A2's predecessor, removed).
__global__ void r4d_gdn_cast_qk_combined_kernel(
        const float * __restrict__ q_src, const float * __restrict__ k_src,
        nv_bfloat16 * __restrict__ q_dst, nv_bfloat16 * __restrict__ k_dst, size_t n_elems) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * n_elems) {
        return;
    }
    if (i < n_elems) {
        q_dst[i] = __float2bfloat16(q_src[i]);
    } else {
        const size_t j = i - n_elems;
        k_dst[j] = __float2bfloat16(k_src[j]);
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

// Kernel head-src (MAD_USE_R4D_GDN_RAW_OUT=1 follow-up): fills head_src[h] = perm(h) for pattern
// C's "_r4d" kernel variant, which has no notion of Hg/R itself (only ggml's own head count
// n_heads=H is meaningful to it) -- see r4d_gdn_raw_out's doc comment in mt_gdn_r4d.cuh. Same
// perm() formula as every other permuting kernel in this file (r4d_gdn_perm_head above).
__global__ void r4d_gdn_build_head_src_kernel(int32_t * __restrict__ head_src, int H, int Hg, int R) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    head_src[h] = r4d_gdn_perm_head(h, Hg, R);
}

// ─────────────────────────────────────────────────────────────────────────
// Strided kernels for ggml_cuda_gdn_r4d_prefix (the production, use_prefill_chunked entry point):
// unlike ggml_cuda_op_gated_delta_net_r4d above, the source tensors here are the FULL n_tokens-
// length buffers (q_d/k_d/v_d/g_d/b_d), not a standalone P-token tensor, and the destination is
// the FULL dst buffer with its own dst_seq_stride (n_tokens) distinct from P — so every read/write
// needs the real strides rather than a flat index. Same permutation, same math, just addressed via
// sq*/sv*/sb*/dst_seq_stride instead of assuming a canonical product-of-dims layout.

// Kernel G2: combined strided f32 -> bf16 cast of q AND k's first P tokens of every sequence in
// ONE launch (no permutation, indexed by Hg; q/k share the same strides in production -- both are
// views into one fused qkv projection buffer). Top half of the index range [0,per) is q, bottom
// half [per,2*per) is k -- see kernel A2 above for the same split, strided instead of flat.
// Replaces what used to be two separate per-tensor strided-cast launches in
// ggml_cuda_gdn_r4d_prefix (the production entry) with one.
__global__ void r4d_gdn_cast_qk_combined_strided_kernel(
        const float * __restrict__ q_src, const float * __restrict__ k_src,
        nv_bfloat16 * __restrict__ q_dst, nv_bfloat16 * __restrict__ k_dst,
        int64_t n_seqs, int64_t P, int Hg, int K_dim,
        int64_t s1, int64_t s2, int64_t s3) {
    const size_t per = (size_t) n_seqs * P * Hg * K_dim;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * per) {
        return;
    }
    const bool   is_k = i >= per;
    const size_t idx  = is_k ? (i - per) : i;
    const int    k    = (int) (idx % K_dim);
    size_t       tmp  = idx / K_dim;
    const int    h    = (int) (tmp % Hg);
    tmp /= Hg;
    const int     t = (int) (tmp % P);
    const int64_t n = tmp / P;
    const float * src = is_k ? k_src : q_src;
    nv_bfloat16 * dst  = is_k ? k_dst : q_dst;
    dst[idx] = __float2bfloat16(src[(size_t) n * s3 + (size_t) t * s2 + (size_t) h * s1 + k]);
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

// Kernel G3 (glue-reduction follow-up): combined strided f32 -> bf16 cast+permute of q, k, AND v
// in ONE launch, replacing kernel G2 (qk) + kernel H (v) as ggml_cuda_gdn_r4d_prefix's (the
// production, non-primed entry) input-staging step. q/k share one set of strides (sq1/sq2/sq3,
// indexed by Hg heads, no permutation, same as kernel G2); v uses its own strides (sv1/sv2/sv3,
// indexed by H heads, permuted h -> perm(h), same as kernel H). Index space is partitioned into
// three contiguous ranges: [0,per_qk) = q, [per_qk,2*per_qk) = k, [2*per_qk,2*per_qk+per_v) = v,
// where per_qk = n_seqs*P*Hg*S_v and per_v = n_seqs*P*H*S_v -- exactly the per-token element count
// of a fused qkv projection buffer (production's sv2 = S_v*(2*Hg+H) is this same total per token),
// so a single 1-D launch over the whole range keeps every thread's read/write pattern identical,
// element for element, to the two kernels it replaces (each thread still computes exactly one
// destination element the same way one of the old kernels would have) -- this is a launch-count
// reduction only, not a numerics change. Removes one kernel-launch boundary (~3-8us on gfx1201)
// from every non-primed call to ggml_cuda_gdn_r4d_prefix; see that function and this file's header
// comment ("What was done instead") for the running launch-count tally.
__global__ void r4d_gdn_cast_qkv_combined_strided_kernel(
        const float * __restrict__ q_src, const float * __restrict__ k_src, const float * __restrict__ v_src,
        nv_bfloat16 * __restrict__ q_dst, nv_bfloat16 * __restrict__ k_dst, nv_bfloat16 * __restrict__ v_dst,
        int64_t n_seqs, int64_t P, int Hg, int H, int R, int S_v,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3) {
    const size_t per_qk = (size_t) n_seqs * P * Hg * S_v;
    const size_t per_v  = (size_t) n_seqs * P * H  * S_v;
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * per_qk + per_v) {
        return;
    }
    if (i < 2 * per_qk) {
        const bool   is_k = i >= per_qk;
        const size_t idx  = is_k ? (i - per_qk) : i;
        const int    k    = (int) (idx % S_v);
        size_t       tmp  = idx / S_v;
        const int    h    = (int) (tmp % Hg);
        tmp /= Hg;
        const int     t = (int) (tmp % P);
        const int64_t n = tmp / P;
        const float * src = is_k ? k_src : q_src;
        nv_bfloat16 * dst  = is_k ? k_dst : q_dst;
        dst[idx] = __float2bfloat16(src[(size_t) n * sq3 + (size_t) t * sq2 + (size_t) h * sq1 + k]);
        return;
    }
    const size_t j   = i - 2 * per_qk;
    const int    v_  = (int) (j % S_v);
    size_t       tmp = j / S_v;
    const int    h   = (int) (tmp % H);
    tmp /= H;
    const int     t  = (int) (tmp % P);
    const int64_t n  = tmp / P;
    const int    hp  = r4d_gdn_perm_head(h, Hg, R);
    const float val = v_src[(size_t) n * sv3 + (size_t) t * sv2 + (size_t) h * sv1 + v_];
    v_dst[((size_t) n * P + t) * H * S_v + (size_t) hp * S_v + v_] = __float2bfloat16(val);
}

// Kernel F2 (glue-reduction follow-up): fused variant of kernel E (h0 permute-in) that ALSO
// builds cu_seqlens in the SAME launch, removing kernel F's separate <<<1,1>>> launch from
// ggml_cuda_gdn_r4d_prefix's hot path. Grid is (H+1, N): the extra H'th column (blockIdx.x==H)
// does no per-head permute work; only thread 0 of block (H,0) writes the whole cu_seqlens[0..N]
// array (N -- the ubatch's sequence count -- is always tiny, so a serial loop here costs nothing
// next to the O(H*VK) work every other block in the same launch does). Every other block's work
// is byte-for-byte identical to kernel E's.
__global__ void r4d_gdn_state_permute_in_and_cu_seqlens_kernel(
        const float * __restrict__ src, float * __restrict__ dst, int32_t * __restrict__ cu,
        int N, int H, int Hg, int R, int VK, int T) {
    const int n = blockIdx.y;
    const int h = blockIdx.x;
    if (h == H) {
        if (n == 0 && threadIdx.x == 0) {
            for (int i = 0; i <= N; ++i) {
                cu[i] = i * T;
            }
        }
        return;
    }
    if (n >= N || h >= H) {
        return;
    }
    const int hp = r4d_gdn_perm_head(h, Hg, R);
    const size_t src_off = ((size_t) n * H + h)  * VK;
    const size_t dst_off = ((size_t) n * H + hp) * VK;
    for (int i = threadIdx.x; i < VK; i += blockDim.x) {
        dst[dst_off + i] = src[src_off + i];
    }
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
// MAD-406 (R4D GDN conv_prep, task 3) — small cast/copy helpers around
// r4d_gdn_conv_prep_w4_h128_bf16 (adapters/plumbing only, per this integration's scope: no new
// algorithms, just relabeling and dtype casts identical in kind to kernels B/D/E above).

// Kernel M: permuting bf16 -> bf16 copy of v. conv_prep already wrote v as bf16 in GGML head
// order (see r4d_gdn_conv_w4_h128_bf16.hip: v's head index there mirrors the packed qkv channel
// layout qwen35.cpp's own views use) — this only relabels the H axis into libr4d's blocked
// convention, no cast, no other arithmetic. Same perm() as kernel B/C.
__global__ void r4d_gdn_permute_v_bf16_kernel(const nv_bfloat16 * __restrict__ src,
                                               nv_bfloat16 * __restrict__ dst,
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
    dst[(gt * H + hp) * (size_t) V + v_] = src[i];
}

// Kernel N: permuting fp32 copy of g and beta, WITHOUT re-summing g — conv_prep already produced
// the per-chunk (64-token) inclusive cumsum internally (r4d_gdn_conv_w4_h128_bf16.hip's own wave
// scan, chunk width CP_BT==64, the same 64 libr4d's kkt_solve/chunk_scan chunk on), so this is a
// straight H-axis relabel of both tensors, one flat launch. Only reachable when this call's T is
// itself a multiple of 64 (ggml_cuda_try_gdn_conv_prep_fusion's own gate), so conv_prep's chunk
// boundaries and libr4d's chunk boundaries line up identically -- no re-derivation needed.
__global__ void r4d_gdn_permute_g_beta_ggml_kernel(const float * __restrict__ g_in,
                                                    const float * __restrict__ beta_in,
                                                    float * __restrict__ g_out, float * __restrict__ beta_out,
                                                    size_t T_total, int H, int Hg, int R) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = T_total * (size_t) H;
    if (i >= total) {
        return;
    }
    const int    h  = (int) (i % H);
    const size_t gt = i / H;
    const int    hp = r4d_gdn_perm_head(h, Hg, R);
    g_out[gt * H + hp]    = g_in[i];
    beta_out[gt * H + hp] = beta_in[i];
}

// Kernel O: relabel ggml's SSM_CONV src[0] (conv_input, the ggml_concat of the d_conv-1 history
// rows and this call's new tokens — build_conv_state, delta-net-base.cpp:449-472) into the two
// buffers r4d_gdn_conv_prep_w4_h128_bf16 wants, in ONE pass:
//   - x_stage[T_total, C]      bf16, channel-fastest — the NEW tokens only (skip the history rows)
//   - cstate[N, C, 3]          bf16, channel-major, tok-fastest — the d_conv-1 history rows
// ggml's conv_input is [tok, channel, seq] contiguous (ne0 = tok, the FASTEST axis — see
// build_conv_state), i.e. transposed from conv_prep's channel-fastest expectation: this kernel is
// exactly that transpose, plus the f32 -> bf16 cast. task 2's cstate design: cstate is always
// freshly rebuilt from ggml's own (always-correct, rollback-proof) history rows every call, so
// there is no cross-call persistent state and no has_init/cache_idx bookkeeping to get wrong on a
// context-checkpoint restore — see this file's ggml_cuda_try_gdn_conv_prep_fusion for the caller.
// One thread per (seq, channel); each thread walks its own (W + T)-length row sequentially (same
// "not a hot inner loop, first cut" reasoning as kernel D/J above).
// Perf follow-up (rocprofv3, production 16k prefill): this kernel used to ALSO write x_stage in
// the same per-(n,c) thread's serial loop below (the "xrow" loop, T up to 2048 iterations). That
// loop's reads (col[W+t], t the inner index) are contiguous PER THREAD (conv_input's tok axis is
// its fastest axis) but STRIDED ACROSS THE WARP (adjacent threads = adjacent c = addresses
// ci_ne0 elements apart) -- every warp-wide read at a given t touched up to 32 separate cache
// lines instead of one. Measured at ~530us/call vs an expected ~250us for a straight
// bandwidth-bound 84MB-read/42MB-write cast+transpose -- i.e. this loop alone cost the difference.
// Fixed by moving the x_stage half of the job to r4d_gdn_conv_prep_stage_x_transpose_kernel below
// (a standard shared-memory TILED transpose: coalesced reads AND coalesced writes, at the cost of
// one bank-conflict-free shared-memory shuffle instead of a strided global one). This kernel now
// handles ONLY the cstate history rows (W == d_conv-1, e.g. 3 elements per (n,c) -- negligible
// bytes either way, so its serial-loop pattern was never the hot part).
__global__ void r4d_gdn_conv_prep_stage_kernel(
        const float * __restrict__ conv_input, int64_t ci_ne0, int64_t ci_ne1,
        nv_bfloat16 * __restrict__ cstate,
        int N, int C, int W) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (c >= C || n >= N) {
        return;
    }
    const float * col = conv_input + (size_t) n * ci_ne0 * ci_ne1 + (size_t) c * ci_ne0;
    for (int i = 0; i < W; ++i) {
        cstate[((size_t) n * C + c) * W + i] = __float2bfloat16(col[i]);
    }
}

// Kernel O2 (perf follow-up): tiled transpose+cast of conv_input's [W, W+T) per-channel,
// tok-fastest rows into x_stage's [T_total, C] channel-fastest layout -- the part kernel O used
// to do with a strided global read per warp (see that kernel's comment above for the measured
// cost). Classic shared-memory tile transpose: the LOAD indexes threads by token (conv_input's
// own fastest axis) so a warp's read at fixed channel is contiguous; the STORE (after
// __syncthreads()) indexes threads by channel (x_stage's fastest axis) so a warp's write at fixed
// token is contiguous too; the transpose itself happens entirely inside the padded ([][TILE_DIM+1])
// shared-memory tile, which needs no coalescing (bank-conflict-free by construction of the +1
// pad). Produces byte-identical output to kernel O's old xrow loop -- same source elements, same
// f32->bf16 cast, same destination addresses, only the access pattern differs.
template <int TILE_DIM>
__global__ void r4d_gdn_conv_prep_stage_x_transpose_kernel(
        const float * __restrict__ conv_input, int64_t ci_ne0, int64_t ci_ne1,
        nv_bfloat16 * __restrict__ x_stage, int T, int C, int W) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    const int n  = blockIdx.z;
    const int t0 = blockIdx.x * TILE_DIM; // token tile origin (conv_input's fastest axis)
    const int c0 = blockIdx.y * TILE_DIM; // channel tile origin
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int c_in = c0 + ty;
    const int t_in = t0 + tx;
    if (c_in < C && t_in < T) {
        tile[ty][tx] = conv_input[(size_t) n * ci_ne0 * ci_ne1 + (size_t) c_in * ci_ne0 + (W + t_in)];
    }
    __syncthreads();

    const int c_out = c0 + tx;
    const int t_out = t0 + ty;
    if (c_out < C && t_out < T) {
        x_stage[(size_t) n * (size_t) T * C + (size_t) t_out * C + c_out] =
            __float2bfloat16(tile[tx][ty]);
    }
}

// Kernel P: flat f32 -> bf16 cast, no relabeling — ssm_conv1d's weight tensor (ne = [kernel_size,
// conv_channels], kernel-tap FASTEST) already matches r4d_gdn_conv_prep_w4_h128_bf16's expected
// [channel][tap] (tap-fastest, CP_W==kernel_size==4) byte layout exactly (see ggml_ssm_conv's
// weight layout vs. r4d_gdn_conv_w4_h128_bf16.hip's `wgt[(d0+c)*CP_W + i]` indexing) — so this is
// a pure elementwise cast, no transpose.
__global__ void r4d_gdn_cast_flat_bf16_kernel(const float * __restrict__ src, nv_bfloat16 * __restrict__ dst,
                                               size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

// Kernel Q: cache_idx[i] = i for i in [0, N) — identity slot map (task 2: cstate has exactly N
// slots, one per THIS call's sequences, no cross-call persistence). Single-thread, alloc/sync
// free, same reasoning as kernel F.
__global__ void r4d_gdn_build_iota_kernel(int32_t * __restrict__ out, int N) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    for (int i = 0; i < N; ++i) {
        out[i] = i;
    }
}

// Kernel S (MAD-406 follow-up, chain 251): permute the K tail rows of a_raw/b_raw (raw pre-
// activation alpha/beta -- real, valid ggml tensor data regardless of the conv_prep fusion, see
// r4d_gdn_conv_prep_primed's header comment) from GGML head order into libr4d's blocked
// convention, into a small [K, H] scratch pair. r4d_gdn_recurrent_update_k128_v128_bf16_fp32state
// indexes its `a`/`b` by hv (libr4d order) directly -- unlike conv_prep, which reads a/b in GGML
// order and permutes only its OUTPUTS afterward (same as every other producer in this file), this
// kernel's inputs themselves need the permutation applied first. a_in/b_in are read at their own
// ab_stride (== H, elements) starting at token T0 -- so `t` here is 0-based within the K-token
// tail window, and the source row is (T0+t).
__global__ void r4d_gdn_permute_ab_tail_kernel(
        const float * __restrict__ a_in, const float * __restrict__ b_in, int64_t ab_stride,
        float * __restrict__ a_out, float * __restrict__ b_out,
        int64_t T0, int K, int H, int Hg, int R) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (h >= H || t >= K) {
        return;
    }
    const int    hp  = r4d_gdn_perm_head(h, Hg, R);
    const size_t src = (size_t) (T0 + t) * ab_stride + h;
    a_out[(size_t) t * H + hp] = a_in[src];
    b_out[(size_t) t * H + hp] = b_in[src];
}

// Kernel U (MAD-406 follow-up, 2026-09-20): permute the per-head A_log/dt_bias (H elements each)
// from GGML head order into libr4d's blocked convention. r4d_gdn_recurrent_update_k128_v128_
// bf16_fp32state indexes A_log[hv]/dt_bias[hv] by hv (r4d_gdn_recurrent_update_k128_v128_bf16_
// fp32state.hip:115) -- the SAME libr4d-order index it uses for v/output (both explicitly
// permuted already) -- so feeding it GGML-order A_log/dt_bias mismatches almost every head's
// decay constant against almost every other head's data (perm_head has only 2 fixed points out
// of 48 at this integration's R=3 geometry), corrupting the state update per-head while leaving
// the register-computed output "close enough" to pass a loose tolerance on some runs -- this was
// the actual cause of the primed-tail composition's final-state mismatch (chain 251 follow-up,
// 2026-09-20): stage (b)'s single-step output matched but its WRITTEN state didn't, because the
// output and the write use the identical post-update h[] register (same address formula for
// read and write, ruling out a layout/transpose bug -- see mt_gdn_r4d.cu's own analysis) so
// only the DECAY INPUT itself, not the read/write address, could explain a content-only
// mismatch. Trivial, H elements, single small launch.
__global__ void r4d_gdn_permute_alog_dtbias_kernel(
        const float * __restrict__ alog_in, const float * __restrict__ dtb_in,
        float * __restrict__ alog_out, float * __restrict__ dtb_out, int H, int Hg, int R) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= H) {
        return;
    }
    const int hp = r4d_gdn_perm_head(h, Hg, R);
    alog_out[hp] = alog_in[h];
    dtb_out[hp]  = dtb_in[h];
}

// Kernel T: sidx[t] = K-t for t in [0, K) -- the snapshot slot map
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state's own `ssm_state_indices` parameter reads
// (see ggml_cuda_gdn_r4d_tail's header comment for the full derivation and the +1/base-pointer
// shift this pairs with). This is gated_delta_net_cuda's own `target_slot = n_tokens - 1 - t`
// snapshot mapping ("slot 0 = most recent state, slot s = s tokens back") SHIFTED BY ONE: the
// recurrent_update kernel treats slot/index 0 as NULL_BLOCK_ID (`if (si <= 0) return;` on the
// read, `if (so > 0)` guarding the write -- read directly from
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip, not assumed) and silently DROPS any
// write whose index is <= 0 -- so target_slot 0 (the FINAL state, the one this whole tail exists
// to produce) would never be written if sidx used ggml's raw 0-based slot numbers directly. Every
// value here is in [1, K], never 0, and ggml_cuda_gdn_r4d_tail passes `state_d -
// state_slot_stride` (not `state_d`) as this kernel's `state` base pointer so that index i lands
// on physical ggml slot (i-1) == target_slot(t) exactly -- single-thread, alloc/sync free.
__global__ void r4d_gdn_build_tail_sidx_kernel(int32_t * __restrict__ sidx, int K) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    for (int t = 0; t < K; ++t) {
        sidx[t] = K - t;
    }
}

// (MAD-406 follow-up, chain 226) A prior version of this file carried a hand-rolled
// r4d_gdn_gate_mulmat_kernel/_supported/_run stand-in here for computing alpha_mulmat/beta_mulmat
// early, restricted to unquantized F32/BF16 weights -- removed: the real checkpoint's
// ssm_alpha.weight/ssm_beta.weight are Q8_0 (declined by that stand-in on every layer) and it was
// a new kernel, against this integration's "adapters only, no kernels beyond trivial cast/copy
// helpers" scope in the first place. ggml_cuda_try_gdn_conv_prep_fusion below now calls the real
// backend dispatch instead, via ggml_cuda_compute_node_now (mt_gdn_r4d.cuh, defined in
// ggml-cuda.cu -- an exported wrapper around that file's own static per-node executor), which
// handles Q8_0 (and anything else the backend supports) correctly because it IS the backend.
//
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

// r4d_gdn_conv_prep_query_primed and r4d_gdn_conv_prep_release_primed need EXTERNAL linkage
// (mt_gdn_r4d.cuh declares them for gated_delta_net.cu) -- defined here, right after the
// anonymous namespace above closes, rather than inside it alongside mark_primed/take_primed
// (which stay `static`/file-local, called only from this file).

// Non-consuming lookup: true iff a primed entry exists for (device, stream) whose gdn pointer
// matches AND whose primed span covers at least `min_T` rows (n_seqs/H/Hg must match exactly).
// Used by gated_delta_net.cu to decide, BEFORE calling ggml_cuda_gdn_r4d_prefix, whether this
// call may skip the usual 64-token-alignment floor on P (safe only when the WHOLE [0, T0) span
// will be served in one un-split chunk_scan call, which is exactly when it is primed for at
// least T0 rows) — and internally by ggml_cuda_gdn_r4d_prefix's own eligibility gate for the same
// reason. Pure inspection, no side effects.
bool r4d_gdn_conv_prep_query_primed(int device, cudaStream_t stream, const ggml_tensor * gdn,
                                     int64_t min_T, int64_t n_seqs, int64_t H, int64_t Hg) {
    r4d_hostprof_scope r4d_hp_(4);
    std::lock_guard<std::mutex> lock(g_r4d_gdn_primed_mutex);
    auto it = g_r4d_gdn_primed.find(std::make_pair(device, stream));
    if (it == g_r4d_gdn_primed.end() || it->second.gdn != gdn) {
        return false;
    }
    const r4d_gdn_conv_prep_primed & p = it->second;
    return p.T >= min_T && p.n_seqs == n_seqs && p.H == H && p.Hg == Hg;
}

// Explicit release, called by gated_delta_net.cu once BOTH the [0,T0) chunk_scan portion (via
// ggml_cuda_gdn_r4d_prefix) and the K-token tail (via ggml_cuda_gdn_r4d_tail, when it ran) of one
// GATED_DELTA_NET call are done — never automatic any more (see r4d_gdn_conv_prep_primed's
// header comment above). Erasing a key that is not present (nothing was primed, or it was
// already released) is a no-op.
void r4d_gdn_conv_prep_release_primed(int device, cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_r4d_gdn_primed_mutex);
    g_r4d_gdn_primed.erase(std::make_pair(device, stream));
}

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
// Raw-output handoff (MAD_USE_R4D_GDN_RAW_OUT=1) -- see mt_gdn_r4d.cuh's r4d_gdn_raw_out doc
// comment for the full three-step protocol. Two small maps, thread_local like every other
// per-graph-compute piece of state this integration keeps (ggml-cuda.cu's g_fused_qrot_skip, the
// verify-plan maps, etc.): `g_r4d_gdn_raw_wanted` (PLAN intent) and `g_r4d_gdn_raw_records`
// (published records, consumed by pattern C's own turn). Both are reset by
// ggml_cuda_gdn_r4d_reset_raw_out_state(), called once per graph compute from ggml-cuda.cu's PLAN
// prepass before it re-marks the new graph -- tensor pointers are only stable within one cgraph
// build, so leftovers from a previous compute must never be trusted.
namespace {
thread_local std::unordered_map<const ggml_tensor *, bool>              g_r4d_gdn_raw_wanted;  // dst -> skip_fp32
thread_local std::unordered_map<const ggml_tensor *, r4d_gdn_raw_out>   g_r4d_gdn_raw_records;
} // namespace

bool r4d_gdn_raw_out_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_USE_R4D_GDN_RAW_OUT");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

void ggml_cuda_gdn_r4d_reset_raw_out_state() {
    g_r4d_gdn_raw_wanted.clear();
    g_r4d_gdn_raw_records.clear();
}

void ggml_cuda_gdn_r4d_mark_raw_wanted(const ggml_tensor * gdn_dst, bool skip_fp32) {
    if (!r4d_gdn_raw_out_enabled()) {
        return; // feature off -- never mark, so ggml_cuda_gdn_r4d_prefix's lookup always misses
    }
    g_r4d_gdn_raw_wanted[gdn_dst] = skip_fp32;
}

bool ggml_cuda_gdn_r4d_raw_wanted(const ggml_tensor * gdn_dst, bool * skip_fp32_out) {
    auto it = g_r4d_gdn_raw_wanted.find(gdn_dst);
    if (it == g_r4d_gdn_raw_wanted.end()) {
        return false;
    }
    if (skip_fp32_out) {
        *skip_fp32_out = it->second;
    }
    return true;
}

void ggml_cuda_gdn_r4d_publish_raw_out(const ggml_tensor * gdn_dst, const r4d_gdn_raw_out & rec) {
    g_r4d_gdn_raw_records[gdn_dst] = rec;
}

bool ggml_cuda_gdn_r4d_take_raw_out(const ggml_tensor * gdn_dst, r4d_gdn_raw_out * out) {
    auto it = g_r4d_gdn_raw_records.find(gdn_dst);
    if (it == g_r4d_gdn_raw_records.end()) {
        return false;
    }
    *out = it->second;
    g_r4d_gdn_raw_records.erase(it);
    return true;
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

    // q, k: flat cast, no permutation (indexed by Hg) -- ONE combined launch (kernel A2) instead
    // of two separate r4d_gdn_cast_f32_to_bf16_kernel calls.
    {
        const size_t n = (size_t) T_total * Hg * K_dim;
        const size_t blocks = (2 * n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_qk_combined_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(q_d, k_d, q_bf16, k_bf16, n);
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
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d, float * prefix_state_out,
        int64_t S_v, int64_t H, int64_t P, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, int64_t dst_seq_stride, int64_t K_tail, cudaStream_t stream) {
    r4d_hostprof_scope r4d_hp_(2);
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
    // MAD-406 follow-up (chain 251): the 64-alignment floor exists only because P used to be a
    // PREFIX that a later, separate call would continue from (kkt_solve/chunk_scan's h0/ht carry
    // is only bit-exact at a 64-token boundary when SPLIT across two calls -- see the file header
    // comment's "Eligibility" section) -- it is NOT a limitation of a single, whole-span
    // chunk_scan call (tests/test-r4d-gdn.hip.cpp's T=100 case runs one un-split call over a
    // non-64-multiple T and matches the CPU reference for both output and final state). When
    // conv_prep already primed buffers covering this call's whole [0, P+K_tail) span, this WILL
    // be exactly one un-split call (see gated_delta_net.cu's use_prefill_chunked branch: P is set
    // to T0 itself, not floor(T0/64)*64, whenever primed), so the alignment floor does not apply.
    // Decode-shaped early-out (chain 312 tg regression follow-up): conv_prep can only ever have
    // primed a call whose FULL span (P+K_tail) is >= 64 -- ggml_cuda_try_gdn_conv_prep_fusion's
    // own T<64 gate never runs conv_prep below that -- so P+K_tail<64 is guaranteed to fail the
    // P%64==0-or-primed check below regardless of what r4d_gdn_conv_prep_query_primed would say.
    // Skip that call (a mutex lock + map lookup) entirely in that case: decode-sized calls
    // (q_len 1-8, DFlash verify batches) hit this on EVERY layer EVERY step, all on the decode
    // critical path, for a query whose answer is always "not primed" here.
    if (P + K_tail < 64) {
        r4d_gdn_log_reject_once("prefix_decode_shaped",
                                 "P=%lld K_tail=%lld (span < 64, cannot be primed, decode shape)",
                                 (long long) P, (long long) K_tail);
        return false;
    }
    const bool primed_avail = r4d_gdn_conv_prep_query_primed(ctx.device, stream, dst, P + K_tail, n_seqs, H_, Hg);
    if (P <= 0 || (P % 64 != 0 && !primed_avail)) {
        r4d_gdn_log_reject_once("prefix_not_chunk_aligned",
                                 "P=%lld K_tail=%lld (need a positive multiple of 64, or a primed "
                                 "whole-span buffer)", (long long) P, (long long) K_tail);
        return false;
    }
    // Head-contiguity only: sq1/sv1/sb1 are used as raw per-element offsets within a token's row
    // (h*s1 + k, h*s1 + v_, h*s1) in r4d_gdn_cast_qk_combined_strided_kernel / r4d_gdn_cast_v_permute_
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
    // RAW_OUT K-tail follow-up: sized for (P+K_tail) rows, not just P, so ggml_cuda_gdn_r4d_tail
    // (called after this function returns, for the SAME dst) can write its own [P,P+K_tail) rows
    // into the SAME buffer in the SAME libr4d head-blocked layout instead of a separate scratch --
    // see that function's own comment. K_tail==0 makes this identical to the old T_total*H_*S_v
    // sizing. r4d_gdn_persist_get only grows (never shrinks/reallocs-and-drops on a smaller ask),
    // so ggml_cuda_gdn_r4d_tail's own later persist_get for this same slot, at (P+K_tail)*H_*S_v
    // or less, safely returns this exact allocation without touching the rows this function is
    // about to write.
    nv_bfloat16 * out_bf16  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_OUT_BF16,
                                                                (size_t) (P + K_tail) * n_seqs * H_ * S_v);
    float *       g_cumsum  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_G_CUMSUM,  (size_t) T_total * H_);
    float *       beta_perm = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_BETA_PERM, (size_t) T_total * H_);
    float *       h0_perm   = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_H0_PERM,   (size_t) n_seqs * H_ * VK);
    float *       ht_perm   = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_HT_PERM,   (size_t) n_seqs * H_ * VK);
    nv_bfloat16 * A_scratch  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_A_SCRATCH, (size_t) T_total * H_ * bt);
    int32_t *     cu_seqlens = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_CU_SEQLENS, (size_t) n_seqs + 1);

    constexpr int CAST_THREADS = 256;

    // MAD-406 (R4D GDN conv_prep, task 3): if ggml_cuda_try_gdn_conv_prep_fusion already ran
    // conv_prep for THIS exact GATED_DELTA_NET call (same dst pointer, same P/n_seqs/H/Hg — see
    // r4d_gdn_conv_prep_take_primed's header comment), q_bf16/k_bf16/v_bf16/g_cumsum/beta_perm
    // already hold live data in it, and q_d/k_d/v_d/g_d/b_d point at ggml tensors that were never
    // computed (their producing nodes were the ones conv_prep's fusion skipped) -- reading them
    // here would be reading uninitialized memory. Skip the cast/permute/cumsum kernels below
    // entirely in that case; everything from cu_seqlens onward is identical either way. Does NOT
    // erase the primed entry (chain 251 follow-up: ggml_cuda_gdn_r4d_tail, called after this
    // function returns for the SAME dst, needs to find it too) -- gated_delta_net.cu releases it
    // explicitly once both portions of this call are done.
    const bool primed = r4d_gdn_conv_prep_take_primed(dev, stream, dst, P, K_tail, n_seqs, H_, Hg, nullptr);
    if (!primed) {
    // q, k, v: ONE combined strided cast+permute launch (kernel G3) instead of two (kernel G2 for
    // q/k, kernel H for v) -- see kernel G3's comment above for why one launch over the whole
    // q|k|v index range is exactly the same per-element work as the two launches it replaces.
    {
        const size_t per_qk = (size_t) T_total * Hg * S_v;
        const size_t per_v  = (size_t) T_total * H_ * S_v;
        const size_t total  = 2 * per_qk + per_v;
        const size_t blocks = (total + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_qkv_combined_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            q_d, k_d, v_d, q_bf16, k_bf16, v_bf16, n_seqs, P, (int) Hg, (int) H_, R, (int) S_v,
            sq1, sq2, sq3, sv1, sv2, sv3);
    }
    // g (cumsum) + beta: strided, permuting, combined.
    {
        const dim3 grid((unsigned) ((H_ + 63) / 64), (unsigned) n_seqs);
        r4d_gdn_g_beta_permute_strided_kernel<<<grid, 64, 0, stream>>>(
            g_d, b_d, g_cumsum, beta_perm, n_seqs, P, (int) H_, (int) Hg, R, bt, sb1, sb2, sb3);
    }
    } // !primed
    // h0: permuting fp32 copy (canonical -- ggml's state tensor is always fully contiguous,
    // guaranteed by ggml_cuda_op_gated_delta_net_impl's own is_contiguous(src_state) assert) --
    // fused with building cu_seqlens (kernel F2) into the same launch, removing kernel F's
    // separate <<<1,1>>> launch.
    {
        const dim3 grid((unsigned) (H_ + 1), (unsigned) n_seqs);
        r4d_gdn_state_permute_in_and_cu_seqlens_kernel<<<grid, 256, 0, stream>>>(
            s_d, h0_perm, cu_seqlens, (int) n_seqs, (int) H_, (int) Hg, R, VK, (int) P);
    }

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

    // Raw-output handoff (MAD_USE_R4D_GDN_RAW_OUT=1): if pattern C's PLAN step marked `dst` as
    // wanting libr4d's raw output AND this call covers the op's ENTIRE per-sequence span in one
    // shot (no K_tail, P == dst_seq_stride) -- a partial call's out_bf16 only ever holds ITS OWN
    // [0,P) rows, not the [P, dst_seq_stride) remainder some other kernel is about to write, so
    // publishing here would hand pattern C's _r4d kernel a buffer missing rows it needs -- publish
    // a record pointing straight at out_bf16 (still in libr4d's own blocked head order; no extra
    // copy) and, unless the PLAN step also wants the fp32 copy kept (MT_ML8_4_RADIANCE_VERIFY=1),
    // skip kernel I's write below entirely. See mt_gdn_r4d.cuh's r4d_gdn_raw_out doc comment.
    bool       raw_skip_fp32 = false;
    const bool raw_wanted    = r4d_gdn_raw_out_enabled() && ggml_cuda_gdn_r4d_raw_wanted(dst, &raw_skip_fp32);
    const bool raw_full_span = K_tail == 0 && P == dst_seq_stride;
    const bool raw_publish   = raw_wanted && raw_full_span;
    if (raw_publish) {
        int32_t * head_src = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_HEAD_SRC, (size_t) H_);
        const unsigned hs_blocks = (unsigned) ((H_ + 63) / 64);
        r4d_gdn_build_head_src_kernel<<<hs_blocks, 64, 0, stream>>>(head_src, (int) H_, (int) Hg, R);
        r4d_gdn_raw_out rec;
        rec.o_bf16    = out_bf16;
        rec.o_nb_head = (size_t) S_v * sizeof(nv_bfloat16);
        rec.o_nb_tok  = (size_t) H_ * (size_t) S_v * sizeof(nv_bfloat16);
        rec.head_src  = head_src;
        rec.head_dim  = (int) S_v;
        rec.n_heads   = (int) H_;
        ggml_cuda_gdn_r4d_publish_raw_out(dst, rec);
    }
    // o: strided permuting cast back, straight into dst_d's [0,P) rows -- skipped only when the
    // raw record just published above will be consumed instead of it (i.e. raw_publish AND the
    // PLAN step didn't ask to keep the fp32 copy too).
    if (!(raw_publish && raw_skip_fp32)) {
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

// ─────────────────────────────────────────────────────────────────────────
// MAD-406 follow-up (chain 251): the DFlash K-token snapshot tail, served from conv_prep's primed
// buffers when they cover this call, instead of gated_delta_net.cu's own launch_gated_delta_net
// <false,true> reading q_d/k_d/v_d/g_d/b_d -- which, whenever conv_prep's fusion ran for this
// layer, are exactly the tensors that fusion's identity-skip left UNWRITTEN (see
// ggml_cuda_try_gdn_conv_prep_fusion's header comment). Called from gated_delta_net.cu's
// use_prefill_chunked branch, right where it would otherwise launch that tail kernel.
//
// ── Design, n_seqs == 1 ONLY (see the guard below) ──────────────────────────────────────────
//
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state (r4d.h; vendored, tested in
// tests/test-r4d-gdn.hip.cpp's run_recurrent_update_path) is libr4d's decode-step kernel: for one
// "item" n, it walks T SEQUENTIAL tokens (cu[n+1]-cu[n]), reading its initial state ONCE from
// slot `sidx[n*stride + (naccept ? naccept[n]-1 : 0)]` and, for every token t, writing the
// updated state into slot `sidx[n*stride + t]` IF that slot index is > 0
// (r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip's own kernel body, read directly rather
// than assumed). With N=1 item and T=K tokens, this is EXACTLY gated_delta_net_cuda's own K-tail
// loop (same per-token recurrence, same "slot 0 = most recent, slot s = s tokens back" mapping --
// gated_delta_net.cu's `target_slot = n_tokens - 1 - t`) in ONE launch instead of a per-token
// kernel: matching target_slot(t) = K-1-t bit-for-bit -- EXCEPT the recurrent_update kernel
// treats slot/index 0 as NULL_BLOCK_ID (`if (si <= 0) return;` on the read, `if (so > 0)` guarding
// every write -- read directly from r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip's
// kernel body, not assumed) and silently DROPS a write whose index is <= 0. Passing ggml's raw
// 0-based target_slot as sidx would therefore never write target_slot 0 -- the FINAL state, the
// one this whole tail exists to produce (this was exactly the chain-251 follow-up bug: "final
// state (slot 0)" came back all-zero, max_rel==1.0, because the t=K-1 write was silently
// dropped). Fix: shift every index up by one -- sidx[t] = K-t (kernel T,
// r4d_gdn_build_tail_sidx_kernel), values in [1,K], never 0 -- and pass `state_d -
// state_slot_stride` (NOT `state_d`) as this kernel's `state` base pointer below, so index i
// lands on physical ggml slot (i-1) == target_slot(t) exactly. PRE-SEED slot K-1 (sidx[0]=K,
// which the kernel reads as its initial state AND overwrites in place for t=0 -- the read and the
// t=0 write target the SAME slot) with prefill_state_out, the state gated_delta_net.cu's own
// chunk_scan-over-[0,T0) call (or, when primed, ggml_cuda_gdn_r4d_prefix's single whole-T0-span
// chunk_scan call) already computed -- this seed is written at the REAL (unshifted) physical
// slot K-1 address, since it goes through state_d directly, not through the kernel's own shifted
// addressing. That is exactly the state the FIRST tail token (global token T0) needs to start
// from, and the kernel's cross-token dependency (its state, `h[]`, lives in registers across the
// WHOLE t-loop within one launch) then carries it forward through the rest of the K tokens with
// no further host-side chaining needed -- one launch, not K.
//
// q/k inputs: primed Q_BF16/K_BF16 rows [T0, T0+K), Hg-indexed -- NEVER permuted anywhere in this
// integration (see the file header proof), so these are usable as-is. They ARE, however, already
// L2-NORMALIZED by conv_prep, while this kernel L2-normalizes q/k ITSELF internally (unlike
// chunk_scan/kkt_solve, which need pre-normalized input -- see the file header's "Precondition"
// section). Re-normalizing an already-unit-norm (up to bf16 rounding) vector by its own ~1 norm
// is very close to a no-op -- a bounded, small perturbation, not a structural error -- and this
// integration accepts it rather than storing a second, pre-norm copy of q/k nothing else needs
// (a real kernel change to skip normalization would be needed to remove it entirely, out of this
// integration's "adapters only" scope). tests/test-r4d-gdn.hip.cpp's own recurrent_update check
// already runs at a 0.08 absolute/relative tolerance for the analogous reason (bf16 + internal
// norm/gate math, not exact by construction); the tail composition test uses a comparable one.
//
// v input: primed V_BF16 rows [T0, T0+K) -- already in libr4d's blocked head order (same
// permutation chunk_scan needs), which is also what this kernel's `v[(tok*H+hv)*V+row]` indexing
// wants (hk = hv/(H/Hg), same GQA convention as chunk_scan). Used as-is, no extra permutation.
//
// a/b inputs: NOT taken from conv_prep's scratch (which only holds the already-gated, cumsum'd
// g/beta -- useless to a kernel that computes gate math itself from RAW a/b) -- taken directly
// from a_raw/b_raw's own real ggml buffers (r4d_gdn_conv_prep_primed.a_raw_data/b_raw_data,
// captured at prime time; valid regardless of the fusion, see that struct's comment), permuted
// into libr4d head order for just the K tail rows (kernel S, r4d_gdn_permute_ab_tail_kernel) --
// this kernel indexes a/b by hv (libr4d order), unlike conv_prep which reads them in GGML order.
//
// A_log/dt_bias: likewise permuted to libr4d order (kernel U, r4d_gdn_permute_alog_dtbias_kernel)
// -- MISSING this (2026-09-20 follow-up) was the actual bug behind the primed-tail composition's
// final-state mismatch: r4d_gdn_recurrent_update_k128_v128_bf16_fp32state.hip:115 indexes
// A_log[hv]/dt_bias[hv] by hv, the SAME libr4d-order index used for v/output, not GGML order (the
// order conv_prep correctly uses A_log/dt_bias in, since conv_prep's own hv IS the GGML packed-
// channel head index). Feeding GGML-order A_log/dt_bias here paired almost every head's decay
// constant with the wrong head's q/k/v/state (perm_head has only 2 fixed points out of 48 at
// R=3) -- corrupting the WRITTEN state per-head while the register-computed single-step OUTPUT
// (built from that same corrupted math) still landed inside a loose tolerance on the test's
// random seed, producing exactly the observed "output PASS, state FAIL, content wrong not
// mapping" signature.
//
// output: written by the kernel in libr4d's blocked head order (same as v); unpermuted back into
// dst_d's [T0, T0+K) rows by the existing r4d_gdn_cast_o_unpermute_strided_kernel, called flat
// (n_seqs=1, so its per-sequence term is inert regardless of dst_seq_stride's value).
//
// state: state_d/state_slot_stride are the SAME variables gated_delta_net.cu already computed for
// its own (unprimed) tail call -- this writes into the identical snapshot cache, same slots, so a
// later consumer (the DFlash draft) sees byte-identical addressing either way. BUT state_d is in
// GGML head order (every other reader of it is -- the plain autoregressive kernel's
// curr_state/state, a later decode step's own recurrent read), while
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state addresses `state` by hv, the SAME libr4d-
// blocked index it uses for q/k/v/A_log/dt_bias/output (r4d_gdn_recurrent_update_k128_v128_
// bf16_fp32state.hip:115/152/210) -- so, same bug class as the A_log/dt_bias fix just above but
// for the state tensor, the kernel is run against a [K,H,S_v*S_v] libr4d-order SCRATCH
// (R4D_GDN_TAIL_STATE_SCRATCH) instead: prefill_state_out (GGML order) is permuted into the
// scratch's slot K-1 before the launch, and after it every one of the K written scratch slots is
// permuted back into state_d's real slots (r4d_gdn_state_permute_kernel, K standing in for its
// own "N" -- state_d's canonical n_seqs==1 layout is byte-identical to that function's [N,H,VK]
// h0/ht shape, GGML_ASSERT'd below). state_head_stride = S_v*S_v matches gated_delta_net_cuda's
// own implicit per-head stride (`state_out_offset = (sequence*H+h_idx)*S_v*S_v`, sequence=0 for
// n_seqs==1) on BOTH sides of the permute.
//
// Returns true iff it fully wrote dst_d's [T0,T0+K) rows and state_d's K snapshot slots -- a
// clean, side-effect-free `false` (falls back to gated_delta_net.cu's own launch_gated_delta_net
// <false,true>) whenever n_seqs != 1 (the primed buffers are laid out [T_total = T*n_seqs] with
// sequences concatenated; a straight [T0,T0+K) row window is only correct for n_seqs==1 -- for
// n_seqs>1 each sequence's own T-row block would need its own K-tail handling and its own
// per-sequence sidx addressing, which this function does not implement) or conv_prep never primed
// (enough of) this call.
bool ggml_cuda_gdn_r4d_tail(
        ggml_backend_cuda_context & ctx, ggml_tensor * gdn_dst,
        const float * prefill_state_out, float * state_d, int64_t state_slot_stride,
        float * dst_d, int64_t S_v, int64_t H, int64_t Hg, int64_t T0, int64_t K,
        int64_t n_seqs, float scale, int64_t dst_seq_stride, cudaStream_t stream) {
    r4d_hostprof_scope r4d_hp_(3);
    if (n_seqs != 1 || K <= 0 || T0 < 0) {
        return false;
    }
    // Decode-shaped early-out (chain 312 tg regression follow-up): conv_prep can only ever have
    // primed a call whose FULL span (T0+K) is >= 64 (ggml_cuda_try_gdn_conv_prep_fusion's own
    // T<64 gate never runs conv_prep below that) -- so T0+K<64 is guaranteed to make the
    // r4d_gdn_conv_prep_take_primed lookup below return false regardless. Skip that call (a
    // mutex lock + map lookup) entirely in that case: a standalone decode/verify-sized
    // GATED_DELTA_NET call that still reaches this function (T0=0, K=q_len<=8, e.g. a DFlash
    // verify batch) hit this on every layer every step, all on the decode critical path, for a
    // query whose answer is always "not primed" here.
    if (T0 + K < 64) {
        return false;
    }
    // The writeback below (r4d_gdn_state_permute_kernel treating state_d as [K, H, S_v*S_v]
    // contiguous, K standing in for its own "N") is only byte-correct when state_d's real
    // slot/head strides actually equal that shape -- true for n_seqs==1's
    // state_out_offset = (sequence*H+h_idx)*S_v*S_v formula (gated_delta_net.cu), guaranteed by
    // the n_seqs!=1 guard just above, but asserted rather than assumed since a future caller
    // could pass a padded/non-canonical stride the way vLLM does for other paged caches here.
    GGML_ASSERT(state_slot_stride == H * S_v * S_v &&
                "mt_gdn_r4d: ggml_cuda_gdn_r4d_tail's state_d layout is not the canonical "
                "[H, S_v*S_v]-per-slot shape the head-order writeback assumes");
    const int dev = ctx.device;
    r4d_gdn_conv_prep_primed p{};
    if (!r4d_gdn_conv_prep_take_primed(dev, stream, gdn_dst, T0, K, n_seqs, H, Hg, &p)) {
        return false;
    }
    const int R = (int) (H / Hg);

    // Same scratch slots ggml_cuda_gdn_r4d_prefix already filled for this call's [0, p.T) rows;
    // T0+K <= p.T is guaranteed by take_primed's own assert above, so [T0, T0+K) is in bounds.
    nv_bfloat16 * q_bf16 = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_Q_BF16, (size_t) p.T * Hg * S_v);
    nv_bfloat16 * k_bf16 = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_K_BF16, (size_t) p.T * Hg * S_v);
    nv_bfloat16 * v_bf16 = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_V_BF16, (size_t) p.T * H  * S_v);

    // RAW_OUT K-tail follow-up: write recurrent_update's raw output directly into out_bf16's
    // [T0, T0+K) rows instead of the old standalone R4D_GDN_TAIL_O_PERM scratch. Same libr4d
    // head-blocked [K,H,V] layout either way (recurrent_update's `o` output convention is
    // unchanged -- see this function's header comment: "output: written by the kernel in
    // libr4d's blocked head order, same as v"), so this is a destination-pointer change only, not
    // a numerics change: o_tail[...] gets byte-identical values to what o_perm[...] used to hold.
    // Sized at (T0+K)*H*S_v <= what ggml_cuda_gdn_r4d_prefix already allocated for this exact call
    // (P=T0, K_tail=K there) -- r4d_gdn_persist_get only grows on a LARGER ask, so this call never
    // reallocates (and never drops) prefix's already-written [0,T0) rows. This is the ordering
    // this function already depended on before this change (it consumes prefix's
    // prefill_state_out), so no new ordering assumption is introduced.
    nv_bfloat16 * out_bf16 = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_OUT_BF16,
                                                               (size_t) (T0 + K) * H * S_v);
    nv_bfloat16 * o_tail   = out_bf16 + (size_t) T0 * H * S_v;

    float *       ab_perm = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_TAIL_AB_PERM, (size_t) 2 * K * H);
    int32_t *     cu_tail = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_TAIL_CU, 2);
    int32_t *     sidx_tail = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_TAIL_SIDX, (size_t) K);
    float *       a_perm  = ab_perm;
    float *       b_perm  = ab_perm + (size_t) K * H;
    float *       alog_perm = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_TAIL_ALOG_PERM, (size_t) H);
    float *       dtb_perm  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_TAIL_DTB_PERM,  (size_t) H);
    const int64_t VK = S_v * S_v;
    // 2026-09-20 follow-up (state head-order bug -- see the enum comment above
    // R4D_GDN_TAIL_STATE_SCRATCH): r4d_gdn_recurrent_update_k128_v128_bf16_fp32state addresses
    // its `state` argument by hv, the libr4d-blocked head index (same as v/output/A_log/dt_bias),
    // but state_d is ggml's own snapshot cache and stays in GGML head order everywhere else it is
    // read/written (the plain autoregressive kernel's curr_state/state, a later decode step's own
    // read). Never let the kernel touch state_d directly -- stage a [K, H, VK] scratch in libr4d
    // order, contiguous per slot (slot_stride = H*VK, head_stride = VK, exactly the addressing
    // the kernel already assumes), and permute the one real state in (seed) and the K real states
    // back out (writeback) at the boundary, the same head relabeling already applied to
    // A_log/dt_bias/a/b/v/output.
    float * state_scratch = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_TAIL_STATE_SCRATCH,
                                                        (size_t) K * H * VK);
    const int64_t state_scratch_slot_stride = H * VK;

    constexpr int TAIL_THREADS = 64;
    {
        const dim3 grid((unsigned) ((H + TAIL_THREADS - 1) / TAIL_THREADS), (unsigned) K);
        r4d_gdn_permute_ab_tail_kernel<<<grid, TAIL_THREADS, 0, stream>>>(
            (const float *) p.a_raw_data, (const float *) p.b_raw_data, p.ab_stride,
            a_perm, b_perm, T0, (int) K, (int) H, (int) Hg, R);
    }
    // A_log/dt_bias: permute to libr4d head order too (kernel U) -- see this function's header
    // comment and kernel U's own comment for the full derivation of why this is required (unlike
    // a/b, which are permuted by kernel S above already).
    {
        const int blocks = (int) ((H + TAIL_THREADS - 1) / TAIL_THREADS);
        r4d_gdn_permute_alog_dtbias_kernel<<<blocks, TAIL_THREADS, 0, stream>>>(
            (const float *) p.a_log_data, (const float *) p.dt_bias_data,
            alog_perm, dtb_perm, (int) H, (int) Hg, R);
    }
    // Seed physical slot K-1 of state_scratch (sidx[0], read as the initial state AND overwritten
    // in place for t=0) with the state after T0 tokens -- see this function's header comment for
    // why that slot is exactly right. prefill_state_out is GGML head order; state_scratch is
    // libr4d order, so this goes through the same permute kernel ggml_cuda_gdn_r4d_prefix uses
    // for h0/ht (N=1 "sequence" == this one slot) instead of a flat memcpy.
    {
        const dim3 grid((unsigned) H, 1);
        r4d_gdn_state_permute_kernel<<<grid, 256, 0, stream>>>(
            prefill_state_out, state_scratch + (size_t) (K - 1) * state_scratch_slot_stride,
            /*N=*/1, (int) H, (int) Hg, R, (int) VK, /*src_is_ggml_order=*/true);
    }
    r4d_gdn_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(cu_tail, 1, (int) K);
    r4d_gdn_build_tail_sidx_kernel<<<1, 1, 0, stream>>>(sidx_tail, (int) K);

    // sidx values are 1-based (kernel T: K-t, never 0 -- see this function's header comment for
    // why) so the kernel's own NULL_BLOCK_ID convention (index <= 0 means "no write") never
    // fires; shifting the base pointer back by one slot-stride makes index i land on physical
    // scratch slot (i-1). This computed pointer is never itself dereferenced (every read/write
    // inside the kernel adds so*st_slot with so in [1,K], so the lowest address ever touched is
    // exactly state_scratch) -- only used as an arithmetic base, same trick vLLM-style
    // NULL_BLOCK_ID paged addressing relies on elsewhere in this vendored library.
    float * state_base = state_scratch - state_scratch_slot_stride;
    const int rc = r4d_gdn_recurrent_update_k128_v128_bf16_fp32state(
        q_bf16 + (size_t) T0 * Hg * S_v, k_bf16 + (size_t) T0 * Hg * S_v, v_bf16 + (size_t) T0 * H * S_v,
        a_perm, b_perm, /*ab_stride=*/H, /*ab_is_bf16=*/0,
        alog_perm, dtb_perm,
        state_base, state_scratch_slot_stride, /*state_head_stride=*/VK,
        o_tail, cu_tail, sidx_tail, /*indices_stride=*/K, /*num_accepted=*/nullptr,
        /*z_gate=*/nullptr, /*norm_weight=*/nullptr, /*norm_eps=*/0.0f, /*norm_act=*/0,
        /*N=*/1, (int) H, (int) Hg, (int) S_v, (int) S_v, scale, /*softplus_thr=*/20.0f, stream);
    if (rc != 0) {
        GGML_ABORT("mt_gdn_r4d: r4d_gdn_recurrent_update_k128_v128_bf16_fp32state (tail) rejected "
                   "shape (rc=%d, H=%lld Hg=%lld K=%lld)", rc, (long long) H, (long long) Hg, (long long) K);
    }

    // Writeback: state_scratch's K physical slots are all in libr4d head order (every one of
    // them gets written exactly once -- sidx[t]=K-t for t in [0,K) covers physical slots
    // K-1..0). state_d's per-slot layout is [H, VK] contiguous with slot_stride/head_stride ==
    // H*VK/VK (state_slot_stride passed in IS exactly H*VK*n_seqs with n_seqs==1, matching
    // gated_delta_net_cuda's own state_out_offset formula -- see this function's header comment),
    // which is byte-for-byte the same "N sequences of [H,VK]" shape r4d_gdn_state_permute_kernel
    // already handles for h0/ht -- so K stands in for N here, one call permutes every slot back
    // to the GGML order every other state_d consumer expects.
    {
        const dim3 grid((unsigned) H, (unsigned) K);
        r4d_gdn_state_permute_kernel<<<grid, 256, 0, stream>>>(
            state_scratch, state_d, (int) K, (int) H, (int) Hg, R, (int) VK, /*src_is_ggml_order=*/false);
    }

    // MAD_USE_R4D_GDN_RAW_OUT=1 follow-up (K-tail coverage, perf pass): decide the raw-publish
    // gate BEFORE the fp32 unpermute-cast so the fp32 write can actually be skipped when it is --
    // rocprofv3 production profiling showed this kernel running unconditionally even with RAW_OUT
    // published+consumed (121ms/720 launches), because an earlier version of this function always
    // ran it and only decided raw-publish afterward. Gate, exactly mirroring
    // ggml_cuda_gdn_r4d_prefix's own raw_publish/raw_skip_fp32 pattern:
    //   raw_wanted    -- PLAN marked this dst (MAD_USE_R4D_GDN_RAW_OUT=1, pattern C structurally
    //                    confirmed the fuse for this dst -- ggml_cuda_gdn_r4d_mark_raw_wanted).
    //   raw_full_span -- this call's tail covers the op's ENTIRE per-sequence span
    //                    (T0+K == dst_seq_stride); a partial tail must never publish a record
    //                    spanning rows that were not actually written (should not arise given
    //                    this function's own n_seqs==1/primed preconditions, checked defensively
    //                    anyway, same invariant every other publish site in this file holds to).
    //   raw_skip_fp32 -- the skip_fp32 flag from the SAME PLAN mark (false under
    //                    MT_ML8_4_RADIANCE_VERIFY=1, which wants the fp32 copy kept too so it can
    //                    diff the two kernels against each other -- see r4d_gdn_raw_out's doc
    //                    comment in mt_gdn_r4d.cuh).
    // out_bf16[0, T0+K) is a complete, contiguous span in libr4d order by this point (this
    // function just wrote its own [T0,T0+K) rows into the SAME buffer ggml_cuda_gdn_r4d_prefix
    // wrote [0,T0) into -- see o_tail's comment above), which is what makes publishing HERE (as
    // opposed to ggml_cuda_gdn_r4d_prefix, which only ever sees its own [0,T0) portion and so can
    // only safely publish immediately when there is no tail at all) correct. Ordering this relies
    // on: ggml_cuda_gdn_r4d_prefix has already run for this exact dst by the time this function
    // is called -- not a NEW assumption, this function already required that (it consumes
    // prefix's own prefill_state_out as its seed state, above).
    bool       raw_skip_fp32 = false;
    const bool raw_wanted    = r4d_gdn_raw_out_enabled() && ggml_cuda_gdn_r4d_raw_wanted(gdn_dst, &raw_skip_fp32);
    const bool raw_full_span = (T0 + K) == dst_seq_stride;
    const bool raw_publish   = raw_wanted && raw_full_span;
    if (!(raw_publish && raw_skip_fp32)) {
        const size_t n      = (size_t) K * H * S_v;
        const size_t blocks = (n + 255) / 256;
        r4d_gdn_cast_o_unpermute_strided_kernel<<<(unsigned) blocks, 256, 0, stream>>>(
            o_tail, dst_d + (size_t) T0 * S_v * H, /*n_seqs=*/1, /*P=*/K, (int) H, (int) Hg, R,
            (int) S_v, dst_seq_stride);
    }
    if (raw_publish) {
        int32_t * head_src = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_HEAD_SRC, (size_t) H);
        const unsigned hs_blocks = (unsigned) ((H + 63) / 64);
        r4d_gdn_build_head_src_kernel<<<hs_blocks, 64, 0, stream>>>(head_src, (int) H, (int) Hg, R);
        r4d_gdn_raw_out rec;
        rec.o_bf16    = out_bf16;
        rec.o_nb_head = (size_t) S_v * sizeof(nv_bfloat16);
        rec.o_nb_tok  = (size_t) H * (size_t) S_v * sizeof(nv_bfloat16);
        rec.head_src  = head_src;
        rec.head_dim  = (int) S_v;
        rec.n_heads   = (int) H;
        ggml_cuda_gdn_r4d_publish_raw_out(gdn_dst, rec);
    }

    if (r4d_gdn_log_enabled()) {
        static std::atomic<bool> logged_once{false};
        bool expected = false;
        if (logged_once.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "[mt_gdn_r4d] conv_prep tail: fused (T0=%lld K=%lld H=%lld Hg=%lld)\n",
                         (long long) T0, (long long) K, (long long) H, (long long) Hg);
        }
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// MAD_USE_R4D_GDN_VERIFY=1 (chain 270 follow-up). Diffs the r4d/primed path (already run into
// scratch by the caller) against the real ggml result for the SAME GATED_DELTA_NET call, on the
// host -- this is a diagnostic path only (reads whole buffers back to host with plain
// cudaMemcpy), never on the production dispatch path, so simplicity is prioritised over speed.
namespace {

__host__ __forceinline__ float r4d_gdn_verify_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t) h << 16;
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

struct r4d_gdn_verify_stats {
    float  max_abs = 0.0f;
    double sum_abs = 0.0;
    size_t n       = 0;
    // Coordinator-requested (2026-09-20 follow-up): the SCALE of the values themselves, not just
    // the diff -- max(|a|,|b|) over every element -- so a small max_abs diff can be told apart
    // from "just bf16 rounding" (diff << scale) vs a real mismatch (diff ~ scale).
    float  max_val = 0.0f;
    float mean_abs() const { return n ? (float) (sum_abs / (double) n) : 0.0f; }
};

r4d_gdn_verify_stats r4d_gdn_verify_diff_f32(const float * a_dev, const float * b_dev, size_t n, cudaStream_t stream) {
    std::vector<float> a(n), b(n);
    if (n > 0) {
        CUDA_CHECK(cudaMemcpyAsync(a.data(), a_dev, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), b_dev, n * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    r4d_gdn_verify_stats s;
    s.n = n;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(a[i] - b[i]);
        s.max_abs = std::max(s.max_abs, d);
        s.sum_abs += (double) d;
        s.max_val = std::max({s.max_val, std::fabs(a[i]), std::fabs(b[i])});
    }
    return s;
}

r4d_gdn_verify_stats r4d_gdn_verify_diff_bf16(const nv_bfloat16 * a_dev, const nv_bfloat16 * b_dev, size_t n, cudaStream_t stream) {
    std::vector<uint16_t> a(n), b(n);
    if (n > 0) {
        CUDA_CHECK(cudaMemcpyAsync(a.data(), a_dev, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(b.data(), b_dev, n * sizeof(uint16_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    r4d_gdn_verify_stats s;
    s.n = n;
    for (size_t i = 0; i < n; ++i) {
        const float av = r4d_gdn_verify_bf16_to_f32(a[i]), bv = r4d_gdn_verify_bf16_to_f32(b[i]);
        s.max_val = std::max({s.max_val, std::fabs(av), std::fabs(bv)});
        const float d = std::fabs(av - bv);
        s.max_abs = std::max(s.max_abs, d);
        s.sum_abs += (double) d;
    }
    return s;
}

} // namespace

void r4d_gdn_verify_compare(
        ggml_backend_cuda_context & ctx, ggml_tensor * gdn_dst,
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3, int64_t neqk1,
        const float * dst_d, const float * state_d, int64_t state_slot_stride,
        const float * verify_dst, const float * verify_state, int64_t verify_slot_stride,
        const float * verify_prefix_state,
        int64_t S_v, int64_t H, int64_t T0, int64_t K, int64_t n_seqs, int64_t n_tokens, int64_t P,
        bool cache_present, int64_t cache_slot_stride, cudaStream_t stream) {
    const int     dev = ctx.device;
    const int64_t Hg  = neqk1;
    const int     R   = (int) (H / Hg);

    // Layer identification: this file has no direct layer index, only the GATED_DELTA_NET
    // tensor's own name (cb(cur, "...", il) conventions elsewhere in this codebase suffix ggml
    // tensor names with "-<il>") -- parsed on a best-effort basis; "?" if unparseable rather than
    // guessing wrong.
    std::string layer_tag = "?";
    {
        const char * nm = ggml_get_name(gdn_dst);
        if (nm != nullptr) {
            const char * dash = std::strrchr(nm, '-');
            if (dash != nullptr && dash[1] != '\0') {
                bool all_digits = true;
                for (const char * p = dash + 1; *p; ++p) {
                    if (*p < '0' || *p > '9') { all_digits = false; break; }
                }
                if (all_digits) {
                    layer_tag = dash + 1;
                }
            }
        }
    }

    static std::mutex          verify_mu;
    static std::atomic<int>    verify_call_count{0};
    static float               worst_out_max_abs = -1.0f;
    const int call_idx  = verify_call_count.fetch_add(1);
    const bool is_layer0 = (call_idx == 0);

    // Fetched once, reused by every comparison below that needs the primed entry's stored
    // pointers (a_raw/b_raw/A_log/dt_bias data, and -- chain 279 follow-up -- the REAL a_raw/
    // b_raw ggml_tensor pointers for cross-checking our early-exec scratch against their later,
    // independently-computed values).
    r4d_gdn_conv_prep_primed p{};
    const bool have_primed = r4d_gdn_conv_prep_take_primed(dev, stream, gdn_dst, T0, K, n_seqs, H, Hg, &p);

    // ---- chain 279 follow-up: bisect the TAIL specifically, on the real production tensors. ---
    // (1) the state the tail is SEEDED with (r4d's own T0-token prefix state, `verify_prefix_
    // state`) vs ggml's own slot K-1 -- ggml's K-tail kernel's OWN `n_tokens` parameter at that
    // call site is `tail`=K (NOT the outer n_tokens), so its `target_slot = n_tokens-1-t` = K-1-t
    // for local step t in [0,K); t=0 is GLOBAL token T0 (tok_offset=T0), so slot K-1 is written
    // exactly once, in place, as "prefix state advanced by ONE tail token" -- but slot K-1's
    // INITIAL content, before that in-place update, is read from `curr_state` (== prefill_
    // state_out, the T0-token prefix state) UNCHANGED for the FIRST iteration's read -- so
    // ggml's OWN slot K-1, if this were compared BEFORE its in-place write landed, would equal
    // the prefix state; after the write (which is what state_d actually holds by the time this
    // function runs), slot K-1 = state after token T0. r4d's own tail applies the IDENTICAL
    // read-then-overwrite to its OWN scratch slot K-1 (see ggml_cuda_gdn_r4d_tail's header
    // comment) -- so the apples-to-apples check is verify_prefix_state (BEFORE any tail token)
    // against ggml's slot K-1 read from a state_d SNAPSHOT taken before its own tail ran, which
    // this function cannot do (state_d is already post-tail by now) -- what IS checkable here is
    // whether the two sides' PREFIX computation over [0,T0) agrees at all, which the state-after-
    // T0 comparison below (against the once-removed slot K-1) still bounds: a big mismatch here
    // means the prefix state disagreement dominates over anything the tail itself could add.
    r4d_gdn_verify_stats prefix_vs_slotK1;
    if (K > 0) {
        prefix_vs_slotK1 = r4d_gdn_verify_diff_f32(
            verify_prefix_state, state_d + (size_t) (K - 1) * state_slot_stride, (size_t) S_v * S_v * H, stream);
    }

    // ---- output: [0,T0) prefix rows vs [T0,T0+K) tail rows, separately (a composed check could
    // hide a small tail regression under a large T0 -- same lesson as the earlier primed-tail
    // composition test). ------------------------------------------------------------------------
    const r4d_gdn_verify_stats out_prefix = r4d_gdn_verify_diff_f32(
        dst_d, verify_dst, (size_t) T0 * S_v * H, stream);
    r4d_gdn_verify_stats out_tail;
    if (K > 0) {
        out_tail = r4d_gdn_verify_diff_f32(
            dst_d + (size_t) T0 * S_v * H, verify_dst + (size_t) T0 * S_v * H, (size_t) K * S_v * H, stream);
    }

    // ---- state snapshot slots 0..K-1: each against ggml's SAME slot, AND (chain 279 follow-up,
    // item 3) against ggml's slot K-1-s -- a slot-order reversal would show up as slot_stats[s]
    // being large while slot_reversed_stats[s] (r4d slot s vs ggml slot K-1-s) is small. ---------
    std::vector<r4d_gdn_verify_stats> slot_stats, slot_reversed_stats;
    for (int64_t s = 0; s < K; ++s) {
        slot_stats.push_back(r4d_gdn_verify_diff_f32(
            state_d + (size_t) s * state_slot_stride, verify_state + (size_t) s * verify_slot_stride,
            (size_t) S_v * S_v * H, stream));
        slot_reversed_stats.push_back(r4d_gdn_verify_diff_f32(
            state_d + (size_t) (K - 1 - s) * state_slot_stride, verify_state + (size_t) s * verify_slot_stride,
            (size_t) S_v * S_v * H, stream));
    }

    // ---- primed inputs vs ggml: stage ggml's OWN q/k/v/g/beta with the SAME strided cast/
    // permute kernels ggml_cuda_gdn_r4d_prefix uses when NOT primed, then diff directly against
    // the ALREADY-PRIMED scratch (same libr4d/bf16 representation on both sides, [0,T0) rows). --
    constexpr int CAST_THREADS = 256;
    const size_t qk_n = (size_t) T0 * Hg * S_v;
    const size_t v_n  = (size_t) T0 * H * S_v;
    const size_t gb_n = (size_t) T0 * H;

    nv_bfloat16 * ggml_q  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_VERIFY_Q_BF16, qk_n);
    nv_bfloat16 * ggml_k  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_VERIFY_K_BF16, qk_n);
    nv_bfloat16 * ggml_v  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_VERIFY_V_BF16, v_n);
    float *       ggml_g  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_VERIFY_G_CUMSUM, gb_n);
    float *       ggml_b  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_VERIFY_BETA_PERM, gb_n);

    {
        const size_t blocks = (2 * qk_n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_qk_combined_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            q_d, k_d, ggml_q, ggml_k, n_seqs, T0, (int) Hg, (int) S_v, sq1, sq2, sq3);
    }
    {
        const size_t blocks = (v_n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_v_permute_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            v_d, ggml_v, n_seqs, T0, (int) H, (int) Hg, R, (int) S_v, sv1, sv2, sv3);
    }
    {
        const dim3 grid((unsigned) ((H + 63) / 64), (unsigned) n_seqs);
        r4d_gdn_g_beta_permute_strided_kernel<<<grid, 64, 0, stream>>>(
            g_d, b_d, ggml_g, ggml_b, n_seqs, T0, (int) H, (int) Hg, R, /*chunk=*/64, sb1, sb2, sb3);
    }

    nv_bfloat16 * primed_q = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_Q_BF16, qk_n);
    nv_bfloat16 * primed_k = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_K_BF16, qk_n);
    nv_bfloat16 * primed_v = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_V_BF16, v_n);
    float *       primed_g = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_G_CUMSUM, gb_n);
    float *       primed_b = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_BETA_PERM, gb_n);

    const r4d_gdn_verify_stats q_stats = r4d_gdn_verify_diff_bf16(primed_q, ggml_q, qk_n, stream);
    const r4d_gdn_verify_stats k_stats = r4d_gdn_verify_diff_bf16(primed_k, ggml_k, qk_n, stream);
    const r4d_gdn_verify_stats v_stats = r4d_gdn_verify_diff_bf16(primed_v, ggml_v, v_n, stream);
    const r4d_gdn_verify_stats g_stats = r4d_gdn_verify_diff_f32(primed_g, ggml_g, gb_n, stream);
    const r4d_gdn_verify_stats b_stats = r4d_gdn_verify_diff_f32(primed_b, ggml_b, gb_n, stream);
    // NOTE: "conv output vs ggml's SSM_CONV output" (also requested) is NOT checked here -- the
    // SSM_CONV tensor itself is not reachable from this call (a different, earlier cgraph node;
    // gated_delta_net.cu's use_prefill_chunked branch only has GATED_DELTA_NET's own src
    // pointers, not the graph topology ggml_cuda_try_gdn_conv_prep_fusion walked to find
    // SSM_CONV). q/k/v above are conv_prep's OWN post-conv/post-norm outputs, so a real
    // conv-stage bug would still show up there (and in g/beta, which also derive from the same
    // conv_prep call), just without pinpointing "conv specifically" vs "split/norm/gate specifically".

    // ---- (2) chain 279 follow-up: the SAME "primed vs ggml" staging, but for the TAIL rows
    // [T0,T0+K) specifically -- reuses the SAME R4D_GDN_VERIFY_* scratch slots (grow-only, so
    // requesting the smaller K-row size here just reuses the already-larger [0,T0) buffer; the
    // [0,T0) values above have already been read out, so overwriting is safe) with the base
    // pointers offset by T0 tokens, exactly like ggml_cuda_gdn_r4d_tail's own addressing. ---------
    r4d_gdn_verify_stats q_tail_stats, k_tail_stats, v_tail_stats;
    if (K > 0) {
        const size_t qk_tail_n = (size_t) K * Hg * S_v;
        const size_t v_tail_n  = (size_t) K * H * S_v;
        nv_bfloat16 * ggml_q_tail = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_VERIFY_Q_BF16, qk_tail_n);
        nv_bfloat16 * ggml_k_tail = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_VERIFY_K_BF16, qk_tail_n);
        nv_bfloat16 * ggml_v_tail = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_VERIFY_V_BF16, v_tail_n);
        {
            const size_t blocks = (2 * qk_tail_n + CAST_THREADS - 1) / CAST_THREADS;
            r4d_gdn_cast_qk_combined_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
                q_d + (size_t) T0 * sq2, k_d + (size_t) T0 * sq2, ggml_q_tail, ggml_k_tail,
                n_seqs, K, (int) Hg, (int) S_v, sq1, sq2, sq3);
        }
        {
            const size_t blocks = (v_tail_n + CAST_THREADS - 1) / CAST_THREADS;
            r4d_gdn_cast_v_permute_strided_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
                v_d + (size_t) T0 * sv2, ggml_v_tail, n_seqs, K, (int) H, (int) Hg, R, (int) S_v, sv1, sv2, sv3);
        }
        // primed_q/k/v (from the [0,T0) staging above) are the SAME buffers ggml_cuda_gdn_r4d_
        // tail itself reads at offset T0 (they cover the whole primed span, T rows, not just T0) --
        // compare the tail's own [T0,T0+K) slice directly against ggml's just-staged tail rows.
        q_tail_stats = r4d_gdn_verify_diff_bf16(primed_q + (size_t) T0 * Hg * S_v, ggml_q_tail, qk_tail_n, stream);
        k_tail_stats = r4d_gdn_verify_diff_bf16(primed_k + (size_t) T0 * Hg * S_v, ggml_k_tail, qk_tail_n, stream);
        v_tail_stats = r4d_gdn_verify_diff_bf16(primed_v + (size_t) T0 * H * S_v, ggml_v_tail, v_tail_n, stream);
    }

    // ---- (3) chain 279 follow-up: our EARLY-EXEC scratch for a_raw/b_raw (used when alpha_
    // needs_exec/beta_needs_exec fired, mt_gdn_r4d.cu's fusion "commit" section) vs the REAL
    // a_raw/b_raw tensors, now valid (verify mode never skips alpha_mulmat/beta_mulmat -- their
    // real turn runs later, strictly before GATED_DELTA_NET's own turn, so a_raw_tensor->data is
    // genuine ground truth by the time this function runs). Trivially zero when the early-exec
    // path did NOT fire for this call (p.a_raw_data == p.a_raw_tensor->data already). -------------
    r4d_gdn_verify_stats a_raw_vs_real, b_raw_vs_real;
    if (have_primed && p.a_raw_tensor != nullptr && p.b_raw_tensor != nullptr) {
        const size_t n_elems = (size_t) ggml_nelements(p.a_raw_tensor);
        a_raw_vs_real = r4d_gdn_verify_diff_f32(
            (const float *) p.a_raw_data, (const float *) p.a_raw_tensor->data, n_elems, stream);
        b_raw_vs_real = r4d_gdn_verify_diff_f32(
            (const float *) p.b_raw_data, (const float *) p.b_raw_tensor->data, n_elems, stream);
    }

    const bool worth_logging = is_layer0;
    bool new_worst = false;
    {
        std::lock_guard<std::mutex> lock(verify_mu);
        if (out_prefix.max_abs > worst_out_max_abs) {
            worst_out_max_abs = out_prefix.max_abs;
            new_worst = true;
        }
    }

    if (worth_logging || new_worst) {
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY layer=%s%s: T=%lld T0=%lld K=%lld P=%lld n_seqs=%lld "
            "state_slot_stride=%lld cache_slot_stride=%lld cache=%s\n",
            layer_tag.c_str(), is_layer0 ? " (layer 0)" : (new_worst ? " (new worst)" : ""),
            (long long) n_tokens, (long long) T0, (long long) K, (long long) P, (long long) n_seqs,
            (long long) state_slot_stride, (long long) cache_slot_stride, cache_present ? "yes" : "no");
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY   output [0,T0): max_abs=%.6f mean_abs=%.6f | tail [T0,T0+K): max_abs=%.6f mean_abs=%.6f\n",
            out_prefix.max_abs, out_prefix.mean_abs(), out_tail.max_abs, out_tail.mean_abs());
        for (int64_t s = 0; s < K; ++s) {
            std::fprintf(stderr,
                "[mt_gdn_r4d] VERIFY   state slot %lld: max_abs=%.6f mean_abs=%.6f | vs ggml slot %lld "
                "(reversal check): max_abs=%.6f mean_abs=%.6f\n",
                (long long) s, slot_stats[s].max_abs, slot_stats[s].mean_abs(),
                (long long) (K - 1 - s), slot_reversed_stats[s].max_abs, slot_reversed_stats[s].mean_abs());
        }
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY   prefix state (r4d, after T0 tokens) vs ggml slot K-1 (post-tail-"
            "step-0, see this function's header comment for why these are one tail-token apart, "
            "not an exact match by construction): max_abs=%.6f mean_abs=%.6f\n",
            prefix_vs_slotK1.max_abs, prefix_vs_slotK1.mean_abs());
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY   tail rows [T0,T0+K) primed vs ggml: q max_abs=%.6f k max_abs=%.6f "
            "v max_abs=%.6f\n",
            q_tail_stats.max_abs, k_tail_stats.max_abs, v_tail_stats.max_abs);
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY   a_raw/b_raw early-exec scratch vs the REAL (later-computed) "
            "tensor: a max_abs=%.6f mean_abs=%.6f | b max_abs=%.6f mean_abs=%.6f (zero means the "
            "early-exec scratch path did not fire for this call, or it matches perfectly)\n",
            a_raw_vs_real.max_abs, a_raw_vs_real.mean_abs(), b_raw_vs_real.max_abs, b_raw_vs_real.mean_abs());
        if (K > 0) {
            std::vector<int32_t> sidx_host((size_t) K);
            int32_t * sidx_dev = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_TAIL_SIDX, (size_t) K);
            CUDA_CHECK(cudaMemcpyAsync(sidx_host.data(), sidx_dev, (size_t) K * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::string sidx_str;
            for (int64_t s = 0; s < K; ++s) {
                sidx_str += std::to_string(sidx_host[s]);
                if (s + 1 < K) sidx_str += ",";
            }
            std::fprintf(stderr, "[mt_gdn_r4d] VERIFY   tail sidx values (1-based, kernel T's K-t mapping): [%s]\n",
                         sidx_str.c_str());
        }
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY   primed vs ggml (bf16/f32 cast+permute, [0,T0) rows): "
            "q max_abs=%.6f k max_abs=%.6f v max_abs=%.6f g(cumsum) max_abs=%.6f beta max_abs=%.6f\n",
            q_stats.max_abs, k_stats.max_abs, v_stats.max_abs, g_stats.max_abs, b_stats.max_abs);
        // Coordinator-requested (2026-09-20 follow-up): v's scale, to tell "just bf16 rounding"
        // (diff << scale) apart from a real mismatch (diff ~ scale) -- v max_abs=0.25 alone
        // doesn't say which.
        std::fprintf(stderr,
            "[mt_gdn_r4d] VERIFY   v scale check: max_abs=%.6f mean_abs=%.6f vs max_val(either side)=%.6f "
            "(bf16 relative precision ~1/256 of scale -> expect max_abs ~%.6f if this is just rounding)\n",
            v_stats.max_abs, v_stats.mean_abs(), v_stats.max_val, v_stats.max_val / 256.0f);
        // Coordinator-requested: g[t] for t=0,1,63,64,65,4087, head 0 (GGML head order), from
        // BOTH sides -- ggml_g/primed_g are stored at the PERMUTED (libr4d) head position, so
        // ggml head 0 lives at column perm_head(0,Hg,R) in both buffers.
        {
            const int hp0 = r4d_gdn_perm_head(0, (int) Hg, R);
            const int64_t print_ts[] = { 0, 1, 63, 64, 65, T0 - 1 };
            for (int64_t t : print_ts) {
                if (t < 0 || t >= T0) {
                    continue;
                }
                float ggml_val = 0.0f, primed_val = 0.0f;
                CUDA_CHECK(cudaMemcpyAsync(&ggml_val, ggml_g + (size_t) t * H + hp0, sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&primed_val, primed_g + (size_t) t * H + hp0, sizeof(float),
                                           cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::fprintf(stderr,
                    "[mt_gdn_r4d] VERIFY   g[t=%lld, head0]: ggml=%.6f primed(conv_prep)=%.6f diff=%.6f\n",
                    (long long) t, ggml_val, primed_val, std::fabs(ggml_val - primed_val));
            }
        }
        // Coordinator-requested (chain 276 follow-up, 2026-09-20): conv_prep's per-token gate is
        // ~1000x too small at some tokens but NOT by a constant ratio across t (1199, 1056, 44.2,
        // 4.6, 6.5, 20.6 at t=0,1,63,64,65,4087) -- a UNIFORM dtype/scale error on A_log alone
        // would multiply the whole cumsum by one constant factor (cumsum is linear), so a
        // non-constant ratio argues for a value-level bug in dt_bias or a_raw specifically (both
        // sit INSIDE the nonlinear softplus, so a wrong dt_bias distorts the ratio differently at
        // every token depending on that token's own `a`) rather than a blanket A_log dtype/scale
        // issue. Print A_log[0]/dt_bias[0]/a_raw[t=0,h=0] as
        // read from the EXACT production pointers conv_prep was given (p.a_log_data/dt_bias_data/
        // a_raw_data), both as f32 (the interpretation r4d_gdn_conv_prep_w4_h128_bf16's own
        // signature declares -- A_log/dt_bias are `const float*` unconditionally in that kernel,
        // NOT gated by ab_is_bf16, which only governs `a`/`b`) and as a pair of bf16 halves (in
        // case the resident data is secretly bf16-packed despite the tensor's own GGML_TYPE_F32
        // label), next to the CPU-computed gate from the f32 interpretation.
        {
            if (have_primed) {
                uint32_t alog_raw = 0, dtb_raw = 0, a_raw_raw = 0;
                CUDA_CHECK(cudaMemcpyAsync(&alog_raw, p.a_log_data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&dtb_raw, p.dt_bias_data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(&a_raw_raw, (const char *) p.a_raw_data, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                float alog_f32, dtb_f32, a_f32;
                std::memcpy(&alog_f32, &alog_raw, sizeof(float));
                std::memcpy(&dtb_f32, &dtb_raw, sizeof(float));
                std::memcpy(&a_f32, &a_raw_raw, sizeof(float));
                const float alog_bf_lo = r4d_gdn_verify_bf16_to_f32((uint16_t) (alog_raw & 0xffff));
                const float alog_bf_hi = r4d_gdn_verify_bf16_to_f32((uint16_t) (alog_raw >> 16));
                const float dtb_bf_lo  = r4d_gdn_verify_bf16_to_f32((uint16_t) (dtb_raw & 0xffff));
                const float dtb_bf_hi  = r4d_gdn_verify_bf16_to_f32((uint16_t) (dtb_raw >> 16));
                const float a_bf_lo    = r4d_gdn_verify_bf16_to_f32((uint16_t) (a_raw_raw & 0xffff));
                const float a_bf_hi    = r4d_gdn_verify_bf16_to_f32((uint16_t) (a_raw_raw >> 16));
                const float sp = [](float x) {
                    if (x > 20.0f) return x;
                    return (x > 0.0f) ? x + std::log1p(std::exp(-x)) : std::log1p(std::exp(x));
                }(a_f32 + dtb_f32);
                const float cpu_gate = -std::exp(alog_f32) * sp;
                std::fprintf(stderr,
                    "[mt_gdn_r4d] VERIFY   gate inputs (production pointers, t=0 h=0): "
                    "A_log[0] f32=%.6f (as bf16 pair: lo=%.6f hi=%.6f) "
                    "dt_bias[0] f32=%.6f (as bf16 pair: lo=%.6f hi=%.6f) "
                    "a_raw[0,0] f32=%.6f (as bf16 pair: lo=%.6f hi=%.6f) "
                    "ab_stride=%lld -> CPU gate(f32)=-exp(A_log)*softplus(a+dt_bias)=%.6f\n",
                    alog_f32, alog_bf_lo, alog_bf_hi, dtb_f32, dtb_bf_lo, dtb_bf_hi,
                    a_f32, a_bf_lo, a_bf_hi, (long long) p.ab_stride, cpu_gate);
            }
        }
    }
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

void r4d_gdn_remainder_log(int64_t remainder_len, int64_t n_seqs, bool used_chunked, int64_t n_pieces) {
    if (!r4d_gdn_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::tuple<int64_t, int64_t, bool, int64_t>> seen;
    const auto key = std::make_tuple(remainder_len, n_seqs, used_chunked, n_pieces);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
        return;
    }
    if (used_chunked) {
        std::fprintf(stderr,
            "[mt_gdn_r4d] r4d-prefix remainder len=%lld n_seqs=%lld path=chunked(gated_delta_net_chunked_cuda) "
            "n_pieces=%lld\n",
            (long long) remainder_len, (long long) n_seqs, (long long) n_pieces);
    } else {
        std::fprintf(stderr,
            "[mt_gdn_r4d] r4d-prefix remainder len=%lld n_seqs=%lld path=autoregressive(launch_gated_delta_net)\n",
            (long long) remainder_len, (long long) n_seqs);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Conv-side hook (GGML_OP_SSM_CONV) — see this file's header comment ("The conv-side hook:
// investigated, DECLINES") for the full reasoning. This gate is a SEPARATE opt-in from
// MAD_USE_R4D_GDN: it exists so the gate line in ssm-conv.cu can be exercised (and logged)
// independently of whether the GATED_DELTA_NET prefix path is on.
// ─────────────────────────────────────────────────────────────────────────
bool r4d_gdn_conv_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_USE_R4D_GDN_CONV");
        const bool   opted_in = env != nullptr && env[0] == '1';
        return opted_in && ggml_cuda_r4d_available();
    }();
    return enabled;
}

// MAD_USE_R4D_GDN_VERIFY=1 (chain 270 follow-up, 2026-09-20): production (conv_prep+prefix both
// on) gave a wrong first token despite the synthetic primed-tail composition test passing --
// meaning the divergence is in something the synthetic test does not reproduce (real strides,
// real conv/gate data, or a real-graph-only interaction), not the tail math itself. With this on,
// ggml_cuda_try_gdn_conv_prep_fusion still runs conv_prep and primes the hand-off exactly as
// today, but identity-skips NOTHING -- every ggml node (SSM_CONV, silu, views, rms_norm, scale,
// gate-prep, and GATED_DELTA_NET itself) computes for real, into its own real buffers -- and
// gated_delta_net.cu's use_prefill_chunked branch additionally runs the r4d/primed path into a
// separate scratch copy (see r4d_gdn_verify_compare) and diffs it against the real ggml result on
// the host, logging once for layer 0 and once for whichever layer has the worst output error seen
// so far (the last such log line, once the whole model has run, is the true worst layer). A
// SEPARATE opt-in from both MAD_USE_R4D_GDN and MAD_USE_R4D_GDN_CONV -- meant to be run WITH both
// of those already on, to see where their combination disagrees with the unmodified path.
bool r4d_gdn_verify_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_USE_R4D_GDN_VERIFY");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

bool ggml_cuda_op_ssm_conv_r4d(
        ggml_backend_cuda_context & /*ctx*/, ggml_tensor * /*dst*/,
        ggml_tensor * /*bias_add_node*/, ggml_tensor * /*silu_dst*/) {
    // r4d_gdn_conv_prep_w4_h128_bf16 needs `a`/`b` (raw pre-activation alpha/beta) and
    // `A_log`/`dt_bias`, none of which are reachable from GGML_OP_SSM_CONV's own src[0]/src[1] or
    // from the bias_add_node/silu_dst siblings ggml-cuda.cu's existing fusion detector already
    // threads through -- this per-OP hook's contract is structurally too narrow for conv_prep and
    // always will be regardless of graph shape (unlike ggml_cuda_try_gdn_conv_prep_fusion below,
    // which walks the WHOLE cgraph from SSM_CONV and does reach a/b/A_log/dt_bias, and does fuse
    // conv_prep for real once its own eligibility gates pass -- see that function). This hook is
    // wired only so ssm-conv.cu's gate line has something to call under MAD_USE_R4D_GDN_CONV=1;
    // it declines unconditionally and silently (no log line here -- see the graph-level detector
    // for conv_prep's real eligibility reporting, which is what actually matters operationally).
    return false;
}

// ═════════════════════════════════════════════════════════════════════════
// GRAPH-LEVEL conv_prep fusion detector (MAD-406 follow-up, 2026-09-19)
//
// ggml_cuda_op_ssm_conv_r4d above is a per-OP hook: it sees only GGML_OP_SSM_CONV's own two srcs
// and declines because a/b/A_log/dt_bias aren't among them. This detector operates one level up,
// on the whole cgraph, exactly like ggml_cuda_try_gdn_cache_fusion (ggml-cuda.cu) -- so it CAN see
// a/b/A_log/dt_bias, by walking from the GATED_DELTA_NET node's own srcs backward through the
// alpha/beta branch. Call-site snippet for the ggml-cuda.cu owner (mirrors the gdn_cache_fusion
// call site exactly; place it BEFORE the existing `{GGML_OP_SSM_CONV, GGML_OP_UNARY}` SiLU-fusion
// check at ggml-cuda.cu:7257 so it gets first refusal, same reasoning as the ml8_4_radiance
// patterns' placement comment):
//
//     if (node->op == GGML_OP_SSM_CONV) {
//         const int nodes_to_skip = ggml_cuda_try_gdn_conv_prep_fusion(cgraph, i, *cuda_ctx);
//         if (nodes_to_skip > 0) {
//             return nodes_to_skip;
//         }
//     }
//
// ── Matched op sequence (Qwen3.8 GDN layer, qwen35.cpp build_layer_attn_linear) ────────────────
//
// Found by walking BACKWARD from the GATED_DELTA_NET node's own srcs (q,k,v,g,beta,state at
// src[0..5] -- ggml_gated_delta_net(ctx0,q,k,v,g,b,s,K), delta-net-base.cpp:567), not by assuming
// fixed forward offsets from SSM_CONV: the alpha/beta branch (qwen35.cpp:636-654) is built in the
// SOURCE before the conv branch (:663-707), and this file cannot assume without inspection which
// way ggml_build_forward_expand's DFS post-order (which visits src[0..5] in ORDER, so q's whole
// ancestor chain -- including SSM_CONV -- would be visited/pushed before g's) places them in the
// actual cgraph -- backward-tracing from GDN's srcs is correct regardless of that ordering, and a
// separate check (below) verifies the ordering assumption the fusion NEEDS rather than presuming
// it silently.
//
//   v  (src[2]): VIEW v_conv (:694, src[0]=SILU) -- fed to GDN directly, no l2norm.
//   q  (src[0]): SCALE (build_gdn_l2_norm's ggml_scale) <- RMS_NORM (ggml_rms_norm) <-
//                VIEW q_conv (:682, src[0]=SILU).
//   k  (src[1]): same shape as q, from VIEW k_conv (:688).
//   SILU (:672, ggml_silu) <- SSM_CONV (:669, ggml_ssm_conv) -- src[0]=conv_input, src[1]=conv_kernel.
//     Qwen3.8's ssm_conv1d has no bias tensor (llama-model.h's ssm_conv1d_b field exists but
//     qwen35.cpp never create_tensors or reads it, confirmed by grep), so this is the 1-skip
//     {SSM_CONV, UNARY(SILU)} shape (ggml-cuda.cu:7257), not the 2-skip ADD+SILU one.
//   gate (src[3]): RESHAPE (:654) <- MUL (:651, gate = softplus * ssm_a) <- {SOFTPLUS (:648,
//                  ggml_softplus) <- ADD (:647, alpha_biased = alpha + ssm_dt) <- RESHAPE (:644,
//                  "a_raw") <- MUL_MAT (:643, build_lora_mm(ssm_alpha, cur)) ; ssm_a leaf (:287,
//                  LLM_TENSOR_SSM_A_NOSCAN)}. ssm_dt (:286) is the ADD's other operand.
//   beta (src[4]): SIGMOID (:640) <- RESHAPE (:637, "b_raw") <- MUL_MAT (:636,
//                  build_lora_mm(ssm_beta, cur)).
//
// a_raw/b_raw (the RESHAPE nodes right after each MUL_MAT, BEFORE dt_bias/softplus/sigmoid) are
// exactly r4d_gdn_conv_prep_w4_h128_bf16's `a`/`b` parameters -- the kernel recomputes
// dt_bias/softplus/-exp(A_log)/sigmoid internally from them plus A_log/dt_bias, so add/softplus/
// mul/reshape(gate)/sigmoid(beta) are what get skipped, NOT the alpha/beta MUL_MATs themselves
// (those must still run -- conv_prep consumes their OUTPUT, a_raw/b_raw, as input; it does not
// reproduce the projection GEMM).
//
// ssm_a's role, per qwen35.cpp:651's own comment ("-A_log.exp() * softplus"): it is a LEAF weight
// ALREADY holding -exp(A_log) (folded at conversion time -- LLM_TENSOR_SSM_A_NOSCAN, "a version of
// SSM_A used for MUL instead of SSM_SCAN", llama-arch.cpp:859), not the raw A_log
// r4d_gdn_conv_prep_w4_h128_bf16's ABI wants (it computes exp(A_log) internally --
// r4d_gdn_conv_w4_h128_bf16.hip:117, `__expf(A_log[hv])`). This file has NO source for a raw A_log
// tensor anywhere in the graph -- recovering it as log(-ssm_a) would be numerically fine (ssm_a's
// values are, by construction, in (-inf, 0)) but is an EXTRA, unverified transform this detector
// does not perform; see the two decline gates below, either of which fires first in practice.
//
// ── How the two structural blockers below were resolved (MAD-406 follow-up, this pass) ─────────
//
// 1. SEQUENCING, RESOLVED: r4d_gdn_conv_prep_w4_h128_bf16 needs a_raw/b_raw's data ALREADY
//    COMPUTED (it reads them as plain device pointers -- it does not run the alpha/beta
//    projection GEMMs itself). Production graphs (confirmed on real chains, MAD-406 follow-up
//    chain 226) place alpha_mulmat/beta_mulmat AFTER SSM_CONV in cgraph->nodes -- the DFS-order
//    reasoning above turned out to predict the wrong direction for this build. Since this
//    detector cannot reorder ggml's own build (out of scope: no ggml-cuda.cu/graph-builder
//    changes), it instead computes alpha_mulmat/beta_mulmat ITSELF, right here, before conv_prep
//    -- see the sequencing gate below, via ggml_cuda_compute_node_now (mt_gdn_r4d.cuh, defined in
//    ggml-cuda.cu): an exported wrapper the ggml-cuda.cu owner added around that file's own
//    static per-node executor (ggml_cuda_compute_forward), so this runs the REAL backend
//    dispatch for the node -- correct for any dtype the backend supports (production's
//    ssm_alpha.weight/ssm_beta.weight are Q8_0; an earlier version of this fix hand-rolled an
//    F32/BF16-only stand-in here, which declined on exactly that checkpoint and was also a new
//    kernel against this integration's own "adapters only" scope -- removed). This still declines
//    ("conv_prep_alpha_beta_not_yet_computed") if they cannot be found at all before GDN (a real
//    structural miss), and separately declines ("conv_prep_alpha/beta_mulmat_srcs_not_ready") if
//    their own inputs aren't ready yet -- never a silently wrong result.
//
// 2. CACHE PROTOCOL, RESOLVED: r4d_gdn_conv_prep_w4_h128_bf16's `cstate`/`cache_idx`/`has_init`
//    parameters (r4d.h) are a genuinely different conv-state-carry protocol from ggml's -- the
//    kernel reads the previous (width-1) tokens from a slot addressed by `cache_idx[n]`
//    (radiance/vLLM's persistent block-cache convention) gated by `has_init`, and wants `x` to be
//    JUST the new tokens. ggml's build_conv_state (delta-net-base.cpp:449) does the same job by
//    ggml_concat()-ing the prior state (from build_rs, a gather against the recurrent-memory
//    cache) ONTO the new tokens ON THE GGML SIDE, so GGML_OP_SSM_CONV's own src[0] (conv_input)
//    already carries its (kernel_size-1) history rows baked into one padded tensor. Task 2's
//    resolution: don't bridge the two protocols -- SKIP cache_idx/has_init's persistent-slot
//    semantics entirely. conv_input's first (kernel_size-1) rows per sequence ARE, by
//    construction, exactly the history conv_prep wants (real prior state when build_rs found one,
//    or ggml's own zero-init for a fresh sequence -- either way already correct and already
//    reflects any prior context-checkpoint ROLLBACK, since conv_input is rebuilt from whatever
//    state build_rs currently reports every single call). So this detector stages a fresh,
//    call-scoped cstate from those rows every time (r4d_gdn_conv_prep_stage_kernel), passes
//    has_init=true unconditionally (a true history of zeros behaves identically to has_init=false
//    inside the kernel), and cache_idx=identity (this call's N sequences, N cstate slots, no
//    meaning beyond this one launch) -- there is no persistent state to keep consistent across
//    calls or restores, so there is nothing for a rollback to get wrong.
//
// Net effect: with (1) satisfied for a real graph (the alpha/beta MUL_MATs already ran) and this
// call's own token count a multiple of 64 (chunk-alignment; see the function body), this detector
// now runs conv_prep for real and primes ggml_cuda_gdn_r4d_prefix's scratch (see
// r4d_gdn_conv_prep_mark_primed/take_primed above) instead of always returning 0.
// ═════════════════════════════════════════════════════════════════════════
namespace {

// Direct tensor-pointer use-count lookup (ggml_node_get_use_count's body, ggml-impl.h, takes a
// node_idx; this is the same lookup keyed by tensor identity instead, since this detector finds
// candidate tensors by walking src pointers backward from GATED_DELTA_NET, not by node_idx).
int32_t r4d_gdn_tensor_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

// True iff t is used exactly once (by the one node this detector already expects to be its sole
// consumer -- checked separately via src-pointer identity) and is not a graph output that some
// caller outside this fusion still needs materialized.
bool r4d_gdn_single_internal_use(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    return r4d_gdn_tensor_use_count(cgraph, t) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
}

bool r4d_gdn_is_unary(const ggml_tensor * t, ggml_unary_op op) {
    return t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == op;
}

// O(window) linear scan for tensor t's own index in cgraph->nodes, restricted to [lo, hi). Used a
// handful of times per fusion ATTEMPT (once per GDN layer's SSM_CONV, not a hot inner loop), so
// this is not worth a hash map.
int r4d_gdn_find_node_idx(const ggml_cgraph * cgraph, const ggml_tensor * t, int lo, int hi) {
    hi = std::min(hi, cgraph->n_nodes);
    for (int j = std::max(lo, 0); j < hi; ++j) {
        if (cgraph->nodes[j] == t) {
            return j;
        }
    }
    return -1;
}

// MAD-406 follow-up (chain 226): true iff tensor `t` is safe to READ at cgraph index `node_idx`
// (i.e. by the time this detector's own trigger, SSM_CONV, dispatches) -- either a leaf (op NONE:
// an already-resident weight or a graph input, never itself scheduled as a compute node) or a
// node whose OWN cgraph index is strictly less than node_idx (already computed by an earlier turn
// in this same dispatch pass). Walks through VIEW-style chains via view_src, since a view's own
// node index can differ from the underlying storage it reads/writes -- what actually needs to be
// "ready" is whatever owns the bytes, not the view node itself. Pure inspection, no side effects.
bool r4d_gdn_src_ready(const ggml_cgraph * cgraph, const ggml_tensor * t, int node_idx) {
    const ggml_tensor * cur = t;
    while (cur != nullptr) {
        if (cur->op == GGML_OP_NONE) {
            return true; // leaf: resident weight/input data, no compute turn to wait on
        }
        const int idx = r4d_gdn_find_node_idx(cgraph, cur, 0, cgraph->n_nodes);
        if (idx >= 0 && idx < node_idx) {
            return true;
        }
        if (cur->view_src != nullptr && cur->view_src != cur) {
            cur = cur->view_src;
            continue;
        }
        return false;
    }
    return false;
}

// Kernel V (concat-elision follow-up, chain 313): writes the SAME bytes build_conv_state's
// state-update ggml_cpy would have written (delta-net-base.cpp:496/558, the n_rs_seq==0 single-cpy
// case only), straight from qkv_mixed (the concat's own src[1]'s transpose-view underlying
// tensor) instead of from concat's materialized output. Addressing, derived by hand from
// ggml_cpy_impl (ggml.c: `result = ggml_view_tensor(b); result->src[0]=a; result->src[1]=b` --
// i.e. cpy_node IS a view of the destination b, with a's own linear/logical iteration order
// (ne0 fastest) mapped onto b's linear iteration order since only ggml_nelements(a)==
// ggml_nelements(b) is required, not matching shapes):
//   a = conv_state_last = ggml_view_3d(conv_input, W, C, Nseq, ...) -- ne=[W,C,Nseq], W fastest,
//       reading conv_input's LAST W token-columns, which -- since conv_input's own token axis is
//       [history(W) | qkv_mixed's T tokens] concatenated -- are exactly qkv_mixed's own last W
//       tokens (columns [T-W, T)).
//   b = conv_state_update = ggml_view_2d(conv_states_all, row_count=W*C, Nseq, ...) -- cpy_node
//       IS this view (same data/nb), so cpy_node->data/nb[1] are b's real, already-resolved
//       destination address/per-sequence stride.
//   a's linear order is (w fastest, then c, then n) -> b's row_count-index for (w,c) is c*W+w
//       (the SAME order a was iterated in), confirmed by ggml_cpy needing only matching ELEMENT
//       COUNTS, not matching axis shapes.
// One thread per (n, c), looping the tiny W dimension (same "not a hot inner loop" reasoning as
// kernel O/D/J) -- this data is a few KB regardless of T.
__global__ void r4d_gdn_concat_elide_state_update_kernel(
        const float * __restrict__ qkv_mixed, int64_t T, int C, int W,
        uint8_t * __restrict__ dst_base, int64_t dst_nb1, int Nseq) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y;
    if (c >= C || n >= Nseq) {
        return;
    }
    float * dst_row = (float *) (dst_base + (size_t) n * dst_nb1);
    for (int w = 0; w < W; ++w) {
        const int64_t t = (T - W) + w;
        const float val = qkv_mixed[(size_t) n * T * C + (size_t) t * C + c];
        dst_row[(size_t) c * W + w] = val;
    }
}

} // namespace

// MAD-406 follow-up (chain 313, concat elision): runs at a GGML_OP_CONCAT node's OWN turn --
// EARLIER in cgraph->nodes than the SSM_CONV/GATED_DELTA_NET pair it feeds -- because by the time
// ggml_cuda_try_gdn_conv_prep_fusion runs (at SSM_CONV's own, later turn), the main per-node
// dispatch loop has ALREADY executed everything at earlier indices, concat (and its state-update
// cpy) included; ggml_cuda_mark_fused_skip only pre-empts nodes whose turn has not come yet, so
// retiring concat is only possible from a hook anchored at concat's OWN index.
//
// Design: do the MINIMUM structural work here (locate the SSM_CONV this concat feeds, locate its
// state-update cpy consumer(s), and confirm concat has no OTHER consumer), then hand the REAL
// eligibility decision to ggml_cuda_try_gdn_conv_prep_fusion itself -- called EARLY, anchored at
// SSM_CONV's real index, from this earlier vantage point. This reuses 100% of that function's
// existing structural pattern match + geometry/dtype/T-alignment gates with ZERO duplicated logic
// (the actual risk here: two independently-written eligibility checks disagreeing about whether a
// call is fusable, with concat already retired one way and SSM_CONV expecting it the other way).
// The a/b-mulmat "needs early exec" logic inside that function already tolerates being invoked
// from an earlier cgraph position (it decides via `mulmat_idx >= node_idx`, which is simply MORE
// often true from here -- exactly what that logic exists to handle). And per the structural
// detection this file's conv_prep stage kernels now do (see ggml_cuda_try_gdn_conv_prep_fusion's
// "have_direct_producers" block), a successful fire reads qkv_mixed/conv_states DIRECTLY and never
// touches conv_input's bytes at all -- which is what makes concat's own compute provably dead
// once this fires, and therefore safe to skip.
//
// n_rs_seq>0 (the K-cpy sliding-window case, delta-net-base.cpp ~511-558): declines outright,
// logged once -- that loop's comments already document it as unsafe to reorder/batch without test
// coverage this integration cannot provide; concat and every cpy in that chain are left running
// exactly as today.
int ggml_cuda_try_gdn_concat_elide(
        const ggml_cgraph * cgraph, int node_idx, ggml_backend_cuda_context & ctx) {
    r4d_hostprof_scope r4d_hp_(0);
    if (!r4d_gdn_conv_enabled() || !r4d_gdn_enabled()) {
        return 0;
    }
    if (node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * concat = cgraph->nodes[node_idx];
    if (concat->op != GGML_OP_CONCAT) {
        return 0;
    }

    // Locate the SSM_CONV this concat feeds (delta-net-base.cpp emits concat, then the cpy(s),
    // then returns to qwen35.cpp which calls ggml_ssm_conv -- so SSM_CONV's index is a handful of
    // nodes after concat's, never far).
    constexpr int WINDOW = 32;
    int ssm_conv_idx = -1;
    for (int j = node_idx + 1; j < cgraph->n_nodes && j <= node_idx + WINDOW; ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_SSM_CONV && cgraph->nodes[j]->src[0] == concat) {
            ssm_conv_idx = j;
            break;
        }
    }
    if (ssm_conv_idx < 0) {
        return 0; // not our pattern -- some other CONCAT node, silent decline
    }

    // Locate every ggml_cpy node in (node_idx, ssm_conv_idx) whose src[0] is a VIEW rooted at
    // concat (conv_state_last -> conv_state_update, delta-net-base.cpp:496 or the loop at 543-559).
    std::vector<int> cpy_idxs;
    for (int j = node_idx + 1; j < ssm_conv_idx; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_CPY && n->src[0] != nullptr && n->src[0]->op == GGML_OP_VIEW &&
            n->src[0]->src[0] == concat) {
            cpy_idxs.push_back(j);
        }
    }
    if (cpy_idxs.empty()) {
        return 0; // structural mismatch -- not build_conv_state's shape, decline silently
    }
    if (cpy_idxs.size() > 1) {
        // n_rs_seq > 0: the K-way sliding-window cpy chain (delta-net-base.cpp ~511-536's own
        // comment: reordering/batching this is unsupported, untested territory). Leave concat and
        // every cpy running exactly as today.
        r4d_gdn_log_reject_once("concat_elide_multi_cpy",
                                 "n_rs_seq>0 sliding-window cpy chain (%zu cpys) -- declining concat elision",
                                 cpy_idxs.size());
        return 0;
    }
    const ggml_tensor * cpy_node = cgraph->nodes[cpy_idxs[0]];

    // concat's ONLY consumers must be this ssm_conv (src[0]) and this one cpy's VIEW (src[0] of
    // the cpy) -- use_count counts DIRECT consumers, i.e. ssm_conv and the view node, exactly 2.
    // Any other consumer means something else in the graph still needs concat's real output --
    // decline conservatively rather than risk it.
    if (r4d_gdn_tensor_use_count(cgraph, concat) != 2) {
        return 0;
    }
    // Destination must be contiguous per-row (ggml_view_2d's own construction guarantees this in
    // practice; asserted rather than assumed since this write bypasses ggml_cpy's own general
    // strided-copy path) -- declines rather than risking a wrong byte order otherwise.
    if (!ggml_is_contiguous(cpy_node)) {
        r4d_gdn_log_reject_once("concat_elide_cpy_dst_not_contiguous",
                                 "state-update cpy destination is not contiguous -- declining");
        return 0;
    }

    // Run the REAL eligibility check + conv_prep fusion EARLY, anchored at ssm_conv's real index.
    // Return codes (see that function's own comment): 0 = declined (decode-shaped T<64, dtype
    // mismatch, whatever) -- concat and its cpy MUST run normally, nothing touched. 1 = fired but
    // fell back to reading conv_input directly (its own structural "have_direct_producers" check
    // failed) -- concat's real output IS still needed, so it is NOT safe to elide even though
    // conv_prep otherwise succeeded. Only 2 (fired AND used the direct producers) makes concat's
    // own compute provably dead.
    const int fired = ggml_cuda_try_gdn_conv_prep_fusion(cgraph, ssm_conv_idx, ctx);
    if (fired != 2) {
        return 0;
    }

    // Re-derive W/C/T/Nseq independently (single-line formulas, same as conv_prep_fusion's own)
    // to write the state-update bytes ourselves before skipping the cpy.
    const ggml_tensor * conv_wgt = cgraph->nodes[ssm_conv_idx]->src[1];
    const ggml_tensor * qkv_view = concat->src[1];
    const ggml_tensor * qkv_flat = (qkv_view != nullptr && qkv_view->op == GGML_OP_TRANSPOSE) ? qkv_view->src[0] : nullptr;
    if (conv_wgt == nullptr || qkv_flat == nullptr) {
        // Should be unreachable (fired==2 means conv_prep_fusion's own have_direct_producers
        // check just verified this same shape) -- but never write into the persistent cache off
        // an unverified pointer, and never skip concat without also being able to correctly
        // satisfy the cpy (concat's buffer must exist for the cpy to read if we don't write its
        // destination ourselves). Decline everything: concat, its view, and the cpy all run
        // exactly as today (conv_prep already computed correctly off the direct producers either
        // way, so declining elision here costs perf, not correctness).
        return 0;
    }
    const int64_t W    = conv_wgt->ne[0] - 1;
    const int     C    = (int) conv_wgt->ne[1];
    const int64_t T    = qkv_flat->ne[1];
    const int     Nseq = (int) qkv_flat->ne[2];

    {
        const dim3 grid((unsigned) ((C + 63) / 64), (unsigned) Nseq);
        r4d_gdn_concat_elide_state_update_kernel<<<grid, 64, 0, ctx.stream()>>>(
            (const float *) qkv_flat->data, T, C, (int) W,
            (uint8_t *) cpy_node->data, cpy_node->nb[1], Nseq);
    }

    ggml_cuda_mark_fused_skip(concat);
    ggml_cuda_mark_fused_skip(concat->src[1]);       // the transpose view node
    ggml_cuda_mark_fused_skip(cgraph->nodes[cpy_idxs[0]]->src[0]); // conv_state_last view node
    ggml_cuda_mark_fused_skip(cgraph->nodes[cpy_idxs[0]]);         // the cpy node itself

    if (r4d_gdn_log_enabled()) {
        static std::atomic<bool> logged_once{false};
        bool expected = false;
        if (logged_once.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr, "[mt_gdn_r4d] concat_elide: fired (T=%lld C=%d W=%lld Nseq=%d)\n",
                         (long long) T, C, (long long) W, Nseq);
        }
    }
    return 1;
}

int ggml_cuda_try_gdn_conv_prep_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_backend_cuda_context & ctx) {
    r4d_hostprof_scope r4d_hp_(1);
    if (!r4d_gdn_conv_enabled()) {
        return 0;
    }
    // conv_prep skips the silu/q/k/v/g/beta producers; GDN must consume the
    // primed buffers via ggml_cuda_gdn_r4d_prefix. Without MAD_USE_R4D_GDN=1
    // that prefix is off and GDN would read uncomputed srcs.
    if (!r4d_gdn_enabled()) {
        return 0;
    }
    if (node_idx < 0 || node_idx >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
    if (ssm_conv->op != GGML_OP_SSM_CONV) {
        return 0;
    }
    // Decode-shaped early-out (chain 312 tg regression follow-up, rocprofv3: MAD_USE_R4D_GDN=1
    // decode was 9% slower than plain ggml with ZERO measurable r4d GPU launches -- the cost was
    // CPU-dispatch-side graph-walk overhead on the decode critical path). Compute this call's
    // token count DIRECTLY off ssm_conv's own src[0]/src[1] -- conv_input->ne[0] ==
    // (conv_kernel_size-1)+T, conv_wgt->ne[0] == conv_kernel_size, the SAME invariant this
    // function's own conv_input-shape check further down already asserts -- instead of paying the
    // ~15-comparison structural pattern match plus the WINDOW=64 forward scan for
    // GATED_DELTA_NET just to reach the identical T<64 decline below (which needs `gdn` found
    // first to read T off gq->ne[2]). Every decode-sized call (q_len 1-8, DFlash verify batches)
    // hits this on EVERY layer EVERY step, so skipping the walk here removes real per-token CPU
    // latency. Conservative: a missing/malformed shape here just falls through to the slower,
    // fully-validated path below instead of guessing -- this is a fast-path SHORTCUT to the same
    // decline, never a new decision.
    if (ssm_conv->src[0] != nullptr && ssm_conv->src[1] != nullptr && ssm_conv->src[1]->ne[0] > 0) {
        const int64_t w_plus_t = ssm_conv->src[0]->ne[0];
        const int64_t conv_w   = ssm_conv->src[1]->ne[0] - 1;
        const int64_t t_guess  = w_plus_t - conv_w;
        if (t_guess > 0 && t_guess < 64) {
            return 0; // identical silent decline to the T<64 check further below, walk skipped
        }
    }
    auto decline = [](const char * reason) {
        r4d_gdn_log_reject_once(reason, "conv_prep graph-fusion declined");
        return 0;
    };

    // ── locate GATED_DELTA_NET within a bounded forward window ────────────────────────────────
    constexpr int WINDOW = 64; // generous: the whole matched chain is ~18 nodes
    const ggml_tensor * gdn     = nullptr;
    int                 gdn_idx = -1;
    for (int j = node_idx + 1; j < cgraph->n_nodes && j <= node_idx + WINDOW; ++j) {
        if (cgraph->nodes[j]->op == GGML_OP_GATED_DELTA_NET) {
            gdn     = cgraph->nodes[j];
            gdn_idx = j;
            break;
        }
    }
    if (gdn == nullptr) {
        return decline("conv_prep_no_gdn_in_window");
    }

    // ── GDN geometry eligibility (same shape this file's other entry points already require) ──
    const ggml_tensor * gq = gdn->src[0];
    const ggml_tensor * gk = gdn->src[1];
    const ggml_tensor * gv = gdn->src[2];
    const ggml_tensor * gg = gdn->src[3];
    const ggml_tensor * gb = gdn->src[4];
    const ggml_tensor * gs = gdn->src[5];
    if (gg->ne[0] == gv->ne[0]) { // KDA (per-channel gate); libr4d GDN kernels want scalar-per-head
        return decline("conv_prep_kda_gate");
    }
    if (gq->ne[0] != 128 || gv->ne[0] != 128) {
        return decline("conv_prep_head_dim");
    }
    if (gv->ne[1] % gq->ne[1] != 0) {
        return decline("conv_prep_gqa_repeat_not_integral");
    }
    if (gdn->type != GGML_TYPE_F32 || gq->type != GGML_TYPE_F32 || gk->type != GGML_TYPE_F32 ||
        gv->type != GGML_TYPE_F32 || gg->type != GGML_TYPE_F32 || gb->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(gs)) {
        return decline("conv_prep_dtype_or_state_layout");
    }
    if (ssm_conv->src[1] == nullptr || ssm_conv->src[1]->ne[0] != 4) {
        return decline("conv_prep_conv_width");
    }

    // ── v branch: VIEW v_conv, src[0] == SILU, feeds gdn directly ─────────────────────────────
    if (gv->op != GGML_OP_VIEW) {
        return decline("conv_prep_v_not_view");
    }
    const ggml_tensor * silu = gv->src[0];
    if (!r4d_gdn_is_unary(silu, GGML_UNARY_OP_SILU)) {
        return decline("conv_prep_no_silu");
    }
    if (silu->src[0] != ssm_conv) {
        return decline("conv_prep_silu_not_ssm_conv"); // e.g. a bias ADD sits between them
    }
    if (!r4d_gdn_single_internal_use(cgraph, ssm_conv)) {
        return decline("conv_prep_ssm_conv_shared"); // some other consumer needs ssm_conv's dst
    }
    if (r4d_gdn_tensor_use_count(cgraph, silu) != 3) {
        return decline("conv_prep_silu_fanout"); // must be exactly q_conv/k_conv/v_conv
    }
    if (!r4d_gdn_single_internal_use(cgraph, gv)) {
        return decline("conv_prep_v_shared");
    }

    // ── q/k branches: SCALE <- RMS_NORM <- VIEW(src[0]==silu), each exclusively used ──────────
    auto match_qk = [&](const ggml_tensor * scale, const ggml_tensor ** out_view) -> bool {
        if (scale->op != GGML_OP_SCALE || !r4d_gdn_single_internal_use(cgraph, scale)) {
            return false;
        }
        const ggml_tensor * rms = scale->src[0];
        if (rms == nullptr || rms->op != GGML_OP_RMS_NORM || !r4d_gdn_single_internal_use(cgraph, rms)) {
            return false;
        }
        const ggml_tensor * view = rms->src[0];
        if (view == nullptr || view->op != GGML_OP_VIEW || view->src[0] != silu ||
            !r4d_gdn_single_internal_use(cgraph, view)) {
            return false;
        }
        *out_view = view;
        return true;
    };
    const ggml_tensor * q_view = nullptr;
    const ggml_tensor * k_view = nullptr;
    if (!match_qk(gq, &q_view)) {
        return decline("conv_prep_q_chain");
    }
    if (!match_qk(gk, &k_view)) {
        return decline("conv_prep_k_chain");
    }
    if (q_view == k_view || q_view == gv || k_view == gv) {
        return decline("conv_prep_qkv_views_alias");
    }

    // ── gate (alpha) branch: RESHAPE <- MUL <- SOFTPLUS <- ADD <- RESHAPE("a_raw") <- MUL_MAT ──
    const ggml_tensor * gate_reshape = gg;
    if (gate_reshape->op != GGML_OP_RESHAPE || !r4d_gdn_single_internal_use(cgraph, gate_reshape)) {
        return decline("conv_prep_gate_reshape");
    }
    const ggml_tensor * gate_mul = gate_reshape->src[0];
    if (gate_mul == nullptr || gate_mul->op != GGML_OP_MUL || !r4d_gdn_single_internal_use(cgraph, gate_mul)) {
        return decline("conv_prep_gate_mul");
    }
    const ggml_tensor * softplus  = gate_mul->src[0];
    const ggml_tensor * a_log_leaf = gate_mul->src[1];
    if (!r4d_gdn_is_unary(softplus, GGML_UNARY_OP_SOFTPLUS) ||
        !r4d_gdn_single_internal_use(cgraph, softplus)) {
        return decline("conv_prep_softplus");
    }
    if (a_log_leaf == nullptr || a_log_leaf->op != GGML_OP_NONE) {
        return decline("conv_prep_ssm_a_not_leaf"); // expected model.layers[il].ssm_a
    }
    const ggml_tensor * alpha_biased = softplus->src[0];
    if (alpha_biased == nullptr || alpha_biased->op != GGML_OP_ADD ||
        !r4d_gdn_single_internal_use(cgraph, alpha_biased)) {
        return decline("conv_prep_alpha_add");
    }
    const ggml_tensor * a_raw    = alpha_biased->src[0];
    const ggml_tensor * dt_bias  = alpha_biased->src[1];
    if (a_raw == nullptr || a_raw->op != GGML_OP_RESHAPE || !r4d_gdn_single_internal_use(cgraph, a_raw)) {
        return decline("conv_prep_a_raw");
    }
    if (dt_bias == nullptr || dt_bias->op != GGML_OP_NONE) {
        return decline("conv_prep_dt_bias_not_leaf"); // expected model.layers[il].ssm_dt
    }
    const ggml_tensor * alpha_mulmat = a_raw->src[0];
    if (alpha_mulmat == nullptr || alpha_mulmat->op != GGML_OP_MUL_MAT) {
        return decline("conv_prep_alpha_mulmat");
    }

    // ── beta branch: SIGMOID <- RESHAPE("b_raw") <- MUL_MAT ────────────────────────────────────
    const ggml_tensor * beta_sigmoid = gb;
    if (!r4d_gdn_is_unary(beta_sigmoid, GGML_UNARY_OP_SIGMOID) ||
        !r4d_gdn_single_internal_use(cgraph, beta_sigmoid)) {
        return decline("conv_prep_beta_sigmoid");
    }
    const ggml_tensor * b_raw = beta_sigmoid->src[0];
    if (b_raw == nullptr || b_raw->op != GGML_OP_RESHAPE || !r4d_gdn_single_internal_use(cgraph, b_raw)) {
        return decline("conv_prep_b_raw");
    }
    const ggml_tensor * beta_mulmat = b_raw->src[0];
    if (beta_mulmat == nullptr || beta_mulmat->op != GGML_OP_MUL_MAT) {
        return decline("conv_prep_beta_mulmat");
    }

    // ── gate 1 (SEQUENCING, MAD-406 follow-up chain 226): a_raw/b_raw's MUL_MAT producers need
    // to be COMPUTED by the time conv_prep reads a_raw/b_raw's data -- but real production graphs
    // place alpha_mulmat/beta_mulmat AFTER SSM_CONV in cgraph->nodes (ggml's own build order is
    // not something this adapter controls; the DFS-order reasoning in this file's header comment
    // was wrong about which way it falls). Search the WHOLE span up to GDN (not just [0,node_idx),
    // which is where the old, always-declining version of this gate looked) -- alpha_mulmat/
    // beta_mulmat MUST be somewhere before gdn_idx topologically (GDN depends on them via
    // a_raw/b_raw), so failing to find them at all here is a real structural miss, not just late
    // placement. ─────────────────────────────────────────────────────────────────────────────────
    const int alpha_mulmat_idx = r4d_gdn_find_node_idx(cgraph, alpha_mulmat, 0, gdn_idx);
    const int beta_mulmat_idx  = r4d_gdn_find_node_idx(cgraph, beta_mulmat, 0, gdn_idx);
    if (alpha_mulmat_idx < 0 || beta_mulmat_idx < 0) {
        return decline("conv_prep_alpha_beta_not_yet_computed");
    }
    // If a mulmat's index is >= node_idx, it has NOT run yet at SSM_CONV's own dispatch turn --
    // this function must compute it itself, via ggml_cuda_compute_node_now (mt_gdn_r4d.cuh,
    // defined in ggml-cuda.cu -- the real backend dispatch for this node, correct for whatever
    // dtype the weight actually is, Q8_0 included), before conv_prep can read a_raw/b_raw. That
    // is only safe once the mulmat's OWN inputs (the weight leaf and the in_proj activation
    // `cur`) are themselves already resident/computed by node_idx (r4d_gdn_src_ready, walking
    // through view_src) -- ggml_cuda_compute_node_now still assumes ITS srcs are ready, same as
    // the normal per-node dispatch loop would at that node's real turn.
    const bool alpha_needs_exec = alpha_mulmat_idx >= node_idx;
    const bool beta_needs_exec  = beta_mulmat_idx  >= node_idx;
    if (alpha_needs_exec) {
        if (!r4d_gdn_src_ready(cgraph, alpha_mulmat->src[0], node_idx) ||
            !r4d_gdn_src_ready(cgraph, alpha_mulmat->src[1], node_idx)) {
            return decline("conv_prep_alpha_mulmat_srcs_not_ready");
        }
    }
    if (beta_needs_exec) {
        if (!r4d_gdn_src_ready(cgraph, beta_mulmat->src[0], node_idx) ||
            !r4d_gdn_src_ready(cgraph, beta_mulmat->src[1], node_idx)) {
            return decline("conv_prep_beta_mulmat_srcs_not_ready");
        }
    }

    // Collect matched producers as an EXPLICIT INDEX SET (not a contiguous
    // span). Production graphs interleave unrelated real nodes between
    // SSM_CONV and GATED_DELTA_NET; requiring idxs to cover [node_idx, gdn_idx)
    // with no gaps was declining every production call
    // (conv_prep_skip_range_not_contiguous). Skip-by-identity via
    // ggml_cuda_mark_fused_skip, same as the radiance fusion patterns.
    // When a mulmat needed early exec, its index is inside (node_idx, gdn_idx)
    // and must be skipped so the normal loop does not dispatch it a second
    // time. When it did NOT need early exec (already computed before node_idx),
    // it is outside this window and is not added.
    std::vector<int> idxs;
    idxs.reserve(18);
    idxs.push_back(node_idx); // ssm_conv itself
    const ggml_tensor * to_place[] = {
        silu, gv, q_view, gq->src[0] /*rms_q*/, gq, k_view, gk->src[0] /*rms_k*/, gk,
        gate_reshape, gate_mul, softplus, alpha_biased, a_raw,
        beta_sigmoid, b_raw,
    };
    for (const ggml_tensor * t : to_place) {
        const int idx = r4d_gdn_find_node_idx(cgraph, t, node_idx + 1, gdn_idx);
        if (idx < 0) {
            return decline("conv_prep_matched_node_out_of_window");
        }
        idxs.push_back(idx);
    }
    if (alpha_needs_exec) {
        idxs.push_back(alpha_mulmat_idx);
    }
    if (beta_needs_exec) {
        idxs.push_back(beta_mulmat_idx);
    }
    std::sort(idxs.begin(), idxs.end());
    idxs.erase(std::unique(idxs.begin(), idxs.end()), idxs.end());

    // ── gate 2 (A_LOG SIDECAR, task 4): r4d_gdn_conv_prep_w4_h128_bf16 needs the RAW A_log (it
    // computes exp(A_log) itself, r4d_gdn_conv_w4_h128_bf16.hip: `__expf(A_log[hv])`); a_log_leaf
    // (== gate_mul->src[1], ggml's ssm_a leaf) already holds -exp(A_log) folded in at conversion
    // time (LLM_TENSOR_SSM_A_NOSCAN) -- the WRONG thing for this kernel. The GDN layer builder
    // (qwen35.cpp build_layer_attn_linear) exposes the derived sidecar
    // (llama_model_build_ssm_a_log_sidecars, llama-model.cpp, task 1) as ssm_conv's src[2] -- an
    // extra, otherwise-unused src slot every other SSM_CONV consumer (CPU, plain CUDA/HIP) already
    // ignores -- so it is found here by construction, no cgraph-visibility trick needed. ─────────
    const ggml_tensor * a_log_raw = ssm_conv->src[2];
    if (a_log_raw == nullptr || a_log_raw->type != GGML_TYPE_F32 || a_log_raw->buffer == nullptr ||
        ggml_nelements(a_log_raw) != ggml_nelements(a_log_leaf)) {
        return decline("conv_prep_no_a_log_sidecar");
    }

    // ── task 3: the cache-protocol gate that used to be unconditional is gone (task 2's design:
    // cstate is always rebuilt from ggml's own SSM_CONV src[0] history rows, so there is no
    // separate protocol to bridge) -- what remains is ordinary per-call shape/layout validation.
    const int64_t K_dim = gq->ne[0];
    const int64_t V_dim = gv->ne[0];
    const int64_t Hg    = gq->ne[1];
    const int64_t H_    = gv->ne[1];
    const int64_t T     = gq->ne[2];
    const int64_t Nseq  = gq->ne[3];
    if (H_ % Hg != 0) {
        return decline("conv_prep_gqa_repeat_not_integral2");
    }
    // MAD-406 follow-up (chain 251): the primed buffers are laid out [T_total = T*Nseq] with
    // sequences concatenated back-to-back (sequence n occupies rows [n*T, (n+1)*T)). Both
    // consumers of this hand-off now serve a WHOLE per-sequence span in one shot -- the [0,T0)
    // chunk_scan portion (ggml_cuda_gdn_r4d_prefix, when primed) and the K-token tail
    // (ggml_cuda_gdn_r4d_tail) -- by taking a straight [0,T0) / [T0,T0+K) row window of the
    // buffer, which is only correct when there is exactly ONE sequence (Nseq==1): for Nseq>1 the
    // same row window would need a per-sequence base offset (n*T) that neither consumer computes
    // today. Declining here for Nseq>1 keeps this fusion's OLD (pre-chain-251), still-correct
    // behavior available for those calls -- their gq/gk/gv/gg/gb never go unwritten because
    // conv_prep simply never skips producing them in the first place.
    if (Nseq != 1) {
        return decline(("conv_prep_multi_seq_unsupported Nseq=" + std::to_string(Nseq)).c_str());
    }
    // Only fuse when this call's own token count is chunk-aligned. gated_delta_net.cu's
    // use_prefill_chunked branch computes P as the largest 64-token-aligned PREFIX of THIS SAME
    // T; when T%64==0, P==T exactly, which is the ONLY case the primed hand-off below
    // (r4d_gdn_conv_prep_take_primed) can safely serve -- its scratch is laid out per-sequence-
    // length T, and a mismatched P != T would misalign every sequence after the first in
    // kkt_solve/chunk_scan's own cu_seqlens. A non-64-aligned T declines here, and
    // ggml_cuda_gdn_r4d_prefix keeps computing its own casts from ggml's own q/k/v/g/beta tensors
    // (which this fusion never having run means WERE actually computed), exactly as today.
    if (T > 0 && T < 64 && Nseq > 0) {
        // The decode graphs (T = num_accepted spec tokens or a plain 1-token step, seen in
        // production as T=1/2/4/7/8 -- always < the 64-token chunk, never chunk-aligned by
        // construction, never going to become eligible no matter what else about the call
        // changes): this is not a miss to report, it is simply the wrong kind of call for this
        // fusion (ggml_cuda_gdn_r4d_prefix declines it identically, and always did, for the same
        // reason -- decode already runs ggml's own per-token kernel). Silent, no log line.
        return 0;
    }
    if (T <= 0 || T % 64 != 0 || Nseq <= 0) {
        return decline(("conv_prep_not_chunk_aligned T=" + std::to_string(T) + " Nseq=" + std::to_string(Nseq)).c_str());
    }

    // conv_input (ssm_conv->src[0]) must be exactly build_conv_state's shape/layout
    // (delta-net-base.cpp:449-472): f32, contiguous, [W+T, C, N] (W = conv_kernel_size - 1).
    const ggml_tensor * conv_input = ssm_conv->src[0];
    const ggml_tensor * conv_wgt   = ssm_conv->src[1];
    const int64_t W = conv_wgt->ne[0] - 1; // conv_kernel_size - 1 (== 3 for width 4)
    const int64_t C = conv_wgt->ne[1];     // conv_channels
    if (conv_input == nullptr || conv_input->type != GGML_TYPE_F32 || !ggml_is_contiguous(conv_input) ||
        conv_input->ne[0] != W + T || conv_input->ne[1] != C || conv_input->ne[2] != Nseq ||
        conv_input->data == nullptr) {
        return decline("conv_prep_conv_input_shape");
    }
    // Concat-elision follow-up (chain 313): conv_input is USUALLY build_conv_state's
    // ggml_concat(conv_states_reshaped, ggml_transpose(qkv_mixed), 0) (delta-net-base.cpp:449-472)
    // -- if this exact structural shape holds, read the two TRUE producers directly instead of
    // conv_input (concat's own materialized output): conv_states_reshaped (concat->src[0], f32,
    // contiguous, [W,C,Nseq] -- IDENTICAL addressing to the cstate stage kernel's existing
    // (conv_input, W+T-wide) reads, just with a W-wide row instead) and qkv_mixed (the transpose
    // view's OWN src[0], concat->src[1]->src[0] -- f32, contiguous, [C,T,Nseq], CHANNEL-fastest,
    // i.e. the EXACT layout x_stage itself wants, so staging it needs nothing but a flat
    // elementwise cast, no transpose at all). This makes conv_prep's own work independent of
    // whether concat ever actually runs, which is what lets ggml_cuda_try_gdn_concat_elide skip
    // it. Any mismatch (a model/config where conv_input isn't built this exact way) falls back to
    // the always-correct, concat-output-reading path (kernel O / kernel O2 tiled transpose) added
    // in the previous perf pass -- this is a strict fast-path addition, never a new decision about
    // whether conv_prep itself is eligible.
    const ggml_tensor * qkv_mixed_direct    = nullptr;
    const ggml_tensor * conv_states_direct  = nullptr;
    bool                have_direct_producers = false;
    if (conv_input->op == GGML_OP_CONCAT && conv_input->src[0] != nullptr && conv_input->src[1] != nullptr) {
        const ggml_tensor * cs_view  = conv_input->src[0];
        const ggml_tensor * qkv_view = conv_input->src[1];
        if (cs_view->type == GGML_TYPE_F32 && ggml_is_contiguous(cs_view) && cs_view->data != nullptr &&
            cs_view->ne[0] == W && cs_view->ne[1] == C && cs_view->ne[2] == Nseq &&
            qkv_view->op == GGML_OP_TRANSPOSE && qkv_view->src[0] != nullptr) {
            const ggml_tensor * qkv_flat = qkv_view->src[0];
            if (qkv_flat->type == GGML_TYPE_F32 && ggml_is_contiguous(qkv_flat) && qkv_flat->data != nullptr &&
                qkv_flat->ne[0] == C && qkv_flat->ne[1] == T && qkv_flat->ne[2] == Nseq) {
                qkv_mixed_direct      = qkv_flat;
                conv_states_direct    = cs_view;
                have_direct_producers = true;
            }
        }
    }
    if (!ggml_is_contiguous(conv_wgt) || conv_wgt->data == nullptr) {
        return decline("conv_prep_conv_weight_layout");
    }
    // a_raw/b_raw: conv_prep reads them as a flat [T, H] row-major array (ab_stride = H_,
    // ab_is_bf16 = 0) -- i.e. it only cares that the CONTIGUOUS buffer's linear layout is
    // H-fastest-then-T-then-N, not how ggml's RESHAPE happened to factor that count into ne[].
    // qwen35.cpp's build_layer_attn_linear reshapes alpha to 3D ([num_v_heads, n_seq_tokens,
    // n_seqs], H fastest) but beta to 4D ([1, num_v_heads, n_seq_tokens, n_seqs] -- see
    // build_layer_attn_linear's `beta = ggml_reshape_4d(...)` vs `alpha = ggml_reshape_3d(...)`)
    // -- both are RESHAPE views over the SAME MUL_MAT output (H-fastest by construction, the
    // matmul's own M axis), and ggml_reshape never moves data, so both land at the identical flat
    // offset t*H_+h for a given (t,h) regardless of the ne[] split (production hit exactly this:
    // a_raw=[48,4096,1], b_raw=[1,48,4096], REJECT conv_prep_alpha_beta_shape before this fix --
    // same H*T*Nseq element count, same flat layout, different factoring). So the only things
    // that actually matter are dtype/contiguity/residency and the TOTAL element count -- checking
    // ne[0..2] individually was stricter than what the kernel needs and rejected this layout.
    if (a_raw->type != GGML_TYPE_F32 || !ggml_is_contiguous(a_raw) || a_raw->data == nullptr ||
        ggml_nelements(a_raw) != H_ * T * Nseq ||
        b_raw->type != GGML_TYPE_F32 || !ggml_is_contiguous(b_raw) || b_raw->data == nullptr ||
        ggml_nelements(b_raw) != H_ * T * Nseq) {
        return decline(("conv_prep_alpha_beta_shape T=" + std::to_string(T) + " H=" + std::to_string(H_) +
                        " a_raw=[" + std::to_string(a_raw->ne[0]) + "," + std::to_string(a_raw->ne[1]) + "," + std::to_string(a_raw->ne[2]) +
                        "] type=" + std::to_string((int) a_raw->type) + " cont=" + std::to_string((int) ggml_is_contiguous(a_raw)) +
                        " data=" + std::to_string(a_raw->data != nullptr) +
                        " b_raw=[" + std::to_string(b_raw->ne[0]) + "," + std::to_string(b_raw->ne[1]) + "," + std::to_string(b_raw->ne[2]) + "]").c_str());
    }
    if (dt_bias->ne[0] != H_ || a_log_leaf->ne[0] != H_ || dt_bias->data == nullptr) {
        return decline("conv_prep_gate_weight_shape");
    }

    // ── past this point: commit -- launch conv_prep and prime the hand-off to
    // ggml_cuda_gdn_r4d_prefix (mt_gdn_r4d.cuh's contract for this function: every return above
    // this line is a clean, side-effect-free decline; every kernel launch below is unconditional
    // once reached, same invariant ggml_cuda_gdn_r4d_prefix's own "past this point: commit" holds
    // to). ──────────────────────────────────────────────────────────────────────────────────────
    const int          dev    = ctx.device;
    cudaStream_t       stream = ctx.stream();
    const int          R      = (int) (H_ / Hg);
    const int64_t      T_total = T * Nseq;

    // MAD-406 follow-up (chain 277/278, 2026-09-20): compute alpha_mulmat/beta_mulmat ourselves
    // via the REAL backend dispatch (ggml_cuda_compute_node_now, mt_gdn_r4d.cuh -- an exported
    // wrapper the ggml-cuda.cu owner added around that file's own static ggml_cuda_compute_
    // forward, run on ctx.stream()) -- but into OUR OWN PERSISTENT SCRATCH, never into the node's
    // own allocator-assigned dst. Root cause of the actual production bug (chain 277/278 verify:
    // conv_prep computed the RIGHT gate from the a/b it was given -- CPU-checked, exact match --
    // but ggml's OWN a/b at the SAME (token,head) were completely different values, e.g.
    // a[0,0]~+9.4 vs the ~-1.83 conv_prep actually saw): ggml's allocator assigns alpha_mulmat's
    // dst buffer based on its POSITION in the graph (its real turn, node index 58 in that trace) --
    // running it early (at SSM_CONV's turn, node index 49) writes into memory the allocator may
    // still consider live for an EARLIER-turn tensor at that point (conv_input, a q/k/v view,
    // ...), and/or that memory gets legitimately overwritten by whatever real node 50..57 owns it
    // before conv_prep ever reads it back. Writing a node's result into ITS OWN dst ahead of its
    // real turn is unsafe in general -- this exact bug class already cost 3 rounds on the fusion-
    // skip side (identity-skip vs contiguous-range) before it was fixed there.
    //
    // Fix: give ggml_cuda_compute_node_now a TEMPORARY, STACK-LOCAL COPY of the node with `.data`
    // repointed at scratch we own (R4D_GDN_ALPHA_SCRATCH/BETA_SCRATCH) -- ne/nb/op/op_params/src
    // all copied verbatim from the real node, so the SAME computation runs (reading the SAME real
    // srcs), just landing somewhere nothing else can alias. ggml_cuda_compute_forward (the
    // function this call ultimately reaches) dispatches purely off node->op and reads node->src[]/
    // ->data/->ne/->nb/->op_params for the MUL_MAT case (`ggml_cuda_mul_mat(ctx, dst->src[0],
    // dst->src[1], dst)`, ggml-cuda.cu) -- it does not need the node to be present in `cgraph` or
    // consult `->buffer` for validity, so a copy never inserted into any graph is safe to pass.
    // conv_prep and the tail then read a_scratch/b_scratch DIRECTLY (never a_raw->data/b_raw->
    // data in this case -- those still point at the REAL, not-yet-computed dst) with the same
    // flat [T,H] (H-fastest) row-major layout a_raw/b_raw's own contract already established
    // (ab_stride = H_) is all that matters, regardless of ne[] factoring.
    //
    // Whichever of the normal per-node dispatch loop's OWN turn for alpha_mulmat/beta_mulmat comes
    // later never runs a second (redundant) execution when this fusion actually fired (non-verify
    // mode: their nodes are in the identity-skip set); in MAD_USE_R4D_GDN_VERIFY=1 mode they are
    // NOT skipped and DO run again later, at their real turn, into their real dst -- which is now
    // completely safe, since this early pass never touched that memory.
    const void * a_data = a_raw->data;
    const void * b_data = b_raw->data;
    if (alpha_needs_exec || beta_needs_exec) {
        const size_t ab_elems = (size_t) H_ * T * Nseq;
        if (alpha_needs_exec) {
            float * a_scratch = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_ALPHA_SCRATCH, ab_elems);
            ggml_tensor alpha_copy = *alpha_mulmat;
            alpha_copy.data = a_scratch;
            if (!ggml_cuda_compute_node_now(ctx, &alpha_copy)) {
                GGML_ABORT("mt_gdn_r4d: ggml_cuda_compute_node_now failed for alpha_mulmat (node %d) -- "
                           "conv_prep fusion cannot safely proceed with a_raw unwritten", alpha_mulmat_idx);
            }
            a_data = a_scratch;
        }
        if (beta_needs_exec) {
            float * b_scratch = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_BETA_SCRATCH, ab_elems);
            ggml_tensor beta_copy = *beta_mulmat;
            beta_copy.data = b_scratch;
            if (!ggml_cuda_compute_node_now(ctx, &beta_copy)) {
                GGML_ABORT("mt_gdn_r4d: ggml_cuda_compute_node_now failed for beta_mulmat (node %d) -- "
                           "conv_prep fusion cannot safely proceed with b_raw unwritten", beta_mulmat_idx);
            }
            b_data = b_scratch;
        }
    }
    if ((alpha_needs_exec || beta_needs_exec) && r4d_gdn_log_enabled()) {
        static std::atomic<bool> logged_once{false};
        bool expected = false;
        if (logged_once.compare_exchange_strong(expected, true)) {
            std::fprintf(stderr,
                "[mt_gdn_r4d] conv_prep: fused (T=%lld, a/b mulmat idx %d/%d > ssm_conv idx %d)\n",
                (long long) T, alpha_mulmat_idx, beta_mulmat_idx, node_idx);
        }
    }

    nv_bfloat16 * q_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_Q_BF16,    (size_t) T_total * Hg * K_dim);
    nv_bfloat16 * k_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_K_BF16,    (size_t) T_total * Hg * K_dim);
    nv_bfloat16 * v_ggml    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_CP_V_GGML, (size_t) T_total * H_ * V_dim);
    float *       g_ggml    = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_CP_G_GGML,       (size_t) T_total * H_);
    float *       beta_ggml = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_CP_BETA_GGML,    (size_t) T_total * H_);
    nv_bfloat16 * v_bf16    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_V_BF16,     (size_t) T_total * H_ * V_dim);
    float *       g_cumsum  = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_G_CUMSUM,         (size_t) T_total * H_);
    float *       beta_perm = r4d_gdn_persist_get<float>(dev, stream, R4D_GDN_BETA_PERM,        (size_t) T_total * H_);
    nv_bfloat16 * x_stage   = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_CP_X_STAGE,   (size_t) T_total * C);
    nv_bfloat16 * cstate    = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_CP_CSTATE,    (size_t) Nseq * C * W);
    nv_bfloat16 * wgt_bf16  = r4d_gdn_persist_get<nv_bfloat16>(dev, stream, R4D_GDN_CP_WGT_BF16,  (size_t) C * (W + 1));
    uint8_t *     has_init  = r4d_gdn_persist_get<uint8_t>(dev, stream, R4D_GDN_CP_HAS_INIT,      (size_t) Nseq);
    int32_t *     cache_idx = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_CP_CACHE_IDX,     (size_t) Nseq);
    int32_t *     cu_seqlens = r4d_gdn_persist_get<int32_t>(dev, stream, R4D_GDN_CU_SEQLENS,      (size_t) Nseq + 1);

    constexpr int CAST_THREADS = 256;

    // conv weight: flat cast (layout already matches r4d_gdn_conv_prep_w4_h128_bf16's expected
    // [channel][tap] byte order -- see kernel P).
    {
        const size_t n      = (size_t) C * (W + 1);
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_cast_flat_bf16_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            (const float *) conv_wgt->data, wgt_bf16, n);
    }
    if (have_direct_producers) {
        // Concat-elision follow-up: read the two TRUE producers directly -- makes conv_prep's
        // own work independent of whether concat ever runs (see the structural-detection block
        // above). cstate: SAME kernel O, SAME addressing formula, just given conv_states_direct's
        // own (W-wide, not W+T-wide) row instead of conv_input's -- conv_states_direct's layout
        // ([W,C,Nseq] contiguous, W fastest) is byte-identical in STRUCTURE to conv_input's own
        // history-row region, just without the extra T columns after it. x_stage: qkv_mixed_direct
        // is [C,T,Nseq] contiguous with C (channel) already the FASTEST axis -- exactly x_stage's
        // own layout -- so staging it is a flat elementwise cast (kernel P /
        // r4d_gdn_cast_flat_bf16_kernel), not a transpose at all.
        {
            const dim3 grid((unsigned) ((C + 63) / 64), (unsigned) Nseq);
            r4d_gdn_conv_prep_stage_kernel<<<grid, 64, 0, stream>>>(
                (const float *) conv_states_direct->data, conv_states_direct->ne[0], conv_states_direct->ne[1],
                cstate, (int) Nseq, (int) C, (int) W);
        }
        {
            const size_t n      = (size_t) T_total * C;
            const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
            r4d_gdn_cast_flat_bf16_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
                (const float *) qkv_mixed_direct->data, x_stage, n);
        }
    } else {
        // Fallback: conv_input's own shape didn't match build_conv_state's concat pattern exactly
        // -- read conv_input (concat's materialized output) as before (kernel O for cstate, kernel
        // O2 tiled transpose for x_stage). Always correct regardless of topology; just does not
        // enable concat elision.
        {
            const dim3 grid((unsigned) ((C + 63) / 64), (unsigned) Nseq);
            r4d_gdn_conv_prep_stage_kernel<<<grid, 64, 0, stream>>>(
                (const float *) conv_input->data, conv_input->ne[0], conv_input->ne[1],
                cstate, (int) Nseq, (int) C, (int) W);
        }
        {
            constexpr int TILE_DIM = 32;
            const dim3 block(TILE_DIM, TILE_DIM);
            const dim3 grid((unsigned) ((T + TILE_DIM - 1) / TILE_DIM),
                             (unsigned) ((C + TILE_DIM - 1) / TILE_DIM),
                             (unsigned) Nseq);
            r4d_gdn_conv_prep_stage_x_transpose_kernel<TILE_DIM><<<grid, block, 0, stream>>>(
                (const float *) conv_input->data, conv_input->ne[0], conv_input->ne[1],
                x_stage, (int) T, (int) C, (int) W);
        }
    }
    // has_init is always "true" (task 2): the history rows just staged into cstate are ALWAYS the
    // correct ones for this call -- ggml's build_rs already zero-initializes a fresh sequence's
    // conv-state slot, so a genuinely-fresh sequence's staged history is already all-zero and
    // conv_prep treats has_init=1 with all-zero history identically to has_init=0. This is also
    // what makes rollback (an earlier recurrent-state snapshot being restored) automatically
    // correct: this cstate is rebuilt from conv_input -- which IS the post-rollback ggml state --
    // on every single call, never carried across calls.
    CUDA_CHECK(cudaMemsetAsync(has_init, 1, (size_t) Nseq, stream));
    // cache_idx is the identity map: this cstate has exactly Nseq slots, one per THIS call's
    // sequences, with no meaning beyond this one launch (task 2: no cross-call persistence).
    r4d_gdn_build_iota_kernel<<<1, 1, 0, stream>>>(cache_idx, (int) Nseq);
    r4d_gdn_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(cu_seqlens, (int) Nseq, (int) T);

    const int cp_rc = r4d_gdn_conv_prep_w4_h128_bf16(
        x_stage, /*xpitch=*/C, wgt_bf16, /*bias=*/nullptr, cstate,
        /*cs_seq=*/(long) C * W, /*cs_dim=*/(long) W, /*cs_tok=*/1L,
        cache_idx, /*ci_stride=*/1L, has_init,
        a_data, b_data, /*ab_stride=*/(long) H_, /*ab_is_bf16=*/0,
        a_log_raw->data, dt_bias->data,
        q_bf16, k_bf16, v_ggml, g_ggml, beta_ggml, cu_seqlens,
        (int) Nseq, (int) T, (int) H_, (int) Hg, (int) K_dim, (int) V_dim, (int) (W + 1),
        /*softplus_thr=*/20.0f, stream);
    if (cp_rc != 0) {
        // r4d_gdn_conv_prep_w4_h128_bf16 rejects a shape none of the checks above anticipated --
        // a plain decline (no side effect outside this file's own scratch has been committed:
        // dst/gdn/the graph are all still untouched) rather than the GGML_ABORT this file uses
        // for kkt_solve/chunk_scan (those run only after THIS file's own geometry invariants,
        // proven over many production calls, are already established -- conv_prep's is new here).
        return decline("conv_prep_kernel_rejected_shape");
    }

    // v/g/beta: relabel ggml's head order into libr4d's blocked convention (kernels M/N) -- see
    // this file's header comment for why this permutation still applies even after conv_prep:
    // conv_prep splits q/k/v straight off the packed qkv channel layout qwen35.cpp's own views
    // use, which is ggml's head order, not libr4d's.
    {
        const size_t n      = (size_t) T_total * H_ * V_dim;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_permute_v_bf16_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            v_ggml, v_bf16, (size_t) T_total, (int) H_, (int) Hg, R, (int) V_dim);
    }
    {
        const size_t n      = (size_t) T_total * H_;
        const size_t blocks = (n + CAST_THREADS - 1) / CAST_THREADS;
        r4d_gdn_permute_g_beta_ggml_kernel<<<(unsigned) blocks, CAST_THREADS, 0, stream>>>(
            g_ggml, beta_ggml, g_cumsum, beta_perm, (size_t) T_total, (int) H_, (int) Hg, R);
    }

    // q_bf16/k_bf16/v_bf16/g_cumsum/beta_perm now hold exactly what ggml_cuda_gdn_r4d_prefix
    // would otherwise have computed itself from q_d/k_d/v_d/g_d/b_d -- prime the hand-off so that
    // function (the [0,T0) chunk_scan portion) AND ggml_cuda_gdn_r4d_tail (the K-token snapshot
    // tail, chain 251 follow-up) skip their own cast/permute/cumsum kernels for this SAME gdn
    // call. a_data/b_data (a_raw/b_raw's data when their mulmat was already computed at
    // node_idx, OR our own scratch when it had to be run early -- see above) and a_log_raw/
    // dt_bias are captured here too, for the tail (see r4d_gdn_conv_prep_primed's header comment
    // for why those specifically, not more scratch). Capturing a_data/b_data rather than always
    // a_raw->data/b_raw->data matters: when alpha_needs_exec/beta_needs_exec, a_raw->data still
    // points at the REAL (not-yet-computed-this-pass) dst -- only a_data/b_data (our scratch)
    // hold the actual values conv_prep just used.
    r4d_gdn_conv_prep_mark_primed(dev, stream, gdn, T, Nseq, H_, Hg,
                                   a_data, b_data, /*ab_stride=*/H_,
                                   a_log_raw->data, dt_bias->data, a_raw, b_raw);

    // MAD_USE_R4D_GDN_VERIFY=1 (chain 270 follow-up): identity-skip NOTHING -- every matched node
    // still computes for real, into its own real buffer, so gated_delta_net.cu's verify block can
    // diff the r4d/primed path (run separately, into scratch) against genuine ggml ground truth.
    // conv_prep itself and the priming above still ran unconditionally -- only the skip is
    // withheld. Returning 0 (not 1) tells the caller nothing was fused, which is correct: nothing
    // WAS skipped.
    const bool verify = r4d_gdn_verify_enabled();
    if (!verify) {
        for (int idx : idxs) {
            ggml_cuda_mark_fused_skip(cgraph->nodes[idx]);
        }
    }
    if (r4d_gdn_log_enabled()) {
        static std::mutex mu;
        static std::set<std::tuple<int64_t, int64_t, int64_t, int64_t>> seen;
        const auto key = std::make_tuple(T, Nseq, H_, Hg);
        std::lock_guard<std::mutex> lock(mu);
        if (seen.insert(key).second) {
            if (verify) {
                std::fprintf(stderr,
                    "[mt_gdn_r4d] conv_prep: fused T=%lld N=%lld H=%lld Hg=%lld (identity-skip "
                    "SUPPRESSED -- MAD_USE_R4D_GDN_VERIFY=1)\n",
                    (long long) T, (long long) Nseq, (long long) H_, (long long) Hg);
            } else {
                std::fprintf(stderr,
                    "[mt_gdn_r4d] conv_prep: fused T=%lld N=%lld H=%lld Hg=%lld (identity-skip %zu nodes)\n",
                    (long long) T, (long long) Nseq, (long long) H_, (long long) Hg, idxs.size());
            }
        }
    }
    // Return value encodes MORE than "fired" for ggml_cuda_try_gdn_concat_elide's benefit: 2 means
    // fired AND read the direct producers (conv_input/concat's bytes were never touched, so
    // concat is provably dead and safe to elide); 1 means fired but fell back to reading
    // conv_input (concat's real output) -- concat is NOT safe to elide in that case. The plain
    // SSM_CONV call site (ggml-cuda.cu) only ever checks `!= 0`, so this distinction is invisible
    // to it and changes nothing about the existing, already-tested SSM_CONV-anchored path.
    if (verify) {
        return 0;
    }
    return have_direct_producers ? 2 : 1; // fired iff nodes were actually skipped; skip itself is via g_fused_qrot_skip
}

#endif  // GGML_HIP_R4D
