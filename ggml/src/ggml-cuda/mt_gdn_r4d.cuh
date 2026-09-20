#pragma once

// mt_gdn_r4d — R4D-backed path for GGML_OP_GATED_DELTA_NET (the Gated DeltaNet recurrence
// Qwen3.8/Qwen3Next's hybrid linear-attention layers use).
//
// Active only when ggml-hip is built with GGML_HIP_R4D=ON (gfx1201 offload arch requested; see
// ggml/src/ggml-hip/CMakeLists.txt) AND the runtime flag MAD_USE_R4D_GDN=1 is set AND the current
// HIP device is actually gfx1201 at runtime (ggml_cuda_r4d_available(), ggml-cuda/r4d/ggml-r4d.h).
// Mirrors the mt_pagedattn_r4d gate pattern (mt_pagedattn_r4d.cuh) but is a SEPARATE env var and a
// separate compiled-in gate: the two ops (attention, GDN) are independent and either can be
// enabled without the other.
//
// R4D (ggml-cuda/r4d/, vendored from radiance-libr4d commit b9e42ab-rx6) supplies hand-tuned
// bf16 Gated DeltaNet kernels fixed at head_k=128, head_v=128, chunk=64 (r4d.h). Qwen3.8-27B's
// linear-attention hparams (linear_key_head_dim=128, linear_value_head_dim=128,
// linear_num_key_heads=16, linear_num_value_heads=48) match K=128/V=128 exactly; the head COUNTS
// differ (16 vs 48, a GQA repeat R=3), which production always exercises (cparams.fused_gdn_ar &&
// fused_gdn_ch are both true by default, so this op receives the raw Hg=16/H=48 tensors on every
// call, never pre-broadcast). ggml and libr4d disagree on which key/query head a given value head
// maps to under GQA (interleaved vs. blocked) — mt_gdn_r4d.cu relabels every value-head-indexed
// tensor by the fixing permutation perm(h) = R*(h mod Hg) + (h/Hg) before/after calling libr4d;
// see that file's header comment for the derivation and proof. Hg==H (R=1) is still served as the
// identity case.
//
// This op is a pure "core GDN" adapter: only the chunked scan (r4d_gdn_kkt_solve_k128_c64_bf16 +
// r4d_gdn_chunk_scan_k128_v128_c64_bf16) is wired. The vendored conv/gate-prep and gated-rmsnorm
// kernels are NOT used — ggml's own graph already runs the causal conv, silu, q/k L2-norm, gate
// and beta-sigmoid, and the gated RMSNorm as separate ops on either side of GGML_OP_GATED_DELTA_NET
// (see gated_delta_net.cu / src/models/qwen35.cpp:build_layer_attn_linear), and none of those map
// 1:1 onto this op's tensor boundary without graph changes. r4d_gdn_recurrent_update_k128_v128_
// bf16_fp32state is likewise left unwired: its C ABI takes RAW pre-activation alpha/beta plus
// A_log/dt_bias and computes the softplus/exp/sigmoid gate itself, but this op's src[3]/src[4]
// (g, beta) are already the fully-computed gate and beta — the per-layer A_log/dt_bias weights
// this kernel would need to invert that transform are not among this op's src tensors at all.
// The chunked kernel pair takes already-computed (g, beta) directly and handles a 1-token chunk
// exactly like any other, so it is used for BOTH decode (n_tokens==1) and prefill uniformly;
// see the .cu file's header comment for the full reasoning.
//
// MAD-406 (R4D integration, GDN half).

#include "common.cuh"

// MAD_USE_R4D_GDN_RAW_OUT=1 follow-up: what ggml-cuda.cu's ml8-4 radiance pattern C needs to read
// libr4d's raw bf16 chunk_scan output DIRECTLY (its own per-call persistent scratch, head-blocked
// order) instead of the fp32, ggml-order tensor ggml_cuda_gdn_r4d_prefix normally un-permutes it
// into (kernel I, r4d_gdn_cast_o_unpermute_strided_kernel in mt_gdn_r4d.cu -- measured ~930us per
// 1024 tokens). Defined outside the GGML_HIP_R4D guard so ggml-cuda.cu can reference the type
// unconditionally; every function taking/returning it is a no-op (false / untouched) when
// GGML_HIP_R4D is not compiled in, same as every other declaration in this header.
struct r4d_gdn_raw_out {
    const void *    o_bf16;    // libr4d's raw chunk_scan output for this call, bf16, [T_total, n_heads, head_dim]
                                // (head-blocked/libr4d order -- head_src[] below maps ggml head -> this order)
    size_t          o_nb_head; // byte stride between heads within one token's row (== head_dim * sizeof(bf16))
    size_t          o_nb_tok;  // byte stride between tokens (== n_heads * head_dim * sizeof(bf16))
    const int32_t * head_src;  // device int32[n_heads]: head_src[h] = the libr4d head slot for ggml head h
    int             head_dim;
    int             n_heads;
};

#ifdef GGML_HIP_R4D

// MAD-406 follow-up (chain 226). Owned/defined in ggml-cuda.cu (that file's owner added it): an
// exported wrapper around ggml-cuda.cu's own static per-node executor
// (ggml_cuda_compute_forward), so a fusion detector living outside that file -- like this one's
// ggml_cuda_try_gdn_conv_prep_fusion -- can run the REAL backend dispatch for one graph node
// "early" (before that node's own turn in the normal per-node dispatch loop) without this file
// reimplementing any part of that dispatch itself. Runs ggml_cuda_compute_forward(ctx, node) on
// ctx.stream() and returns its result; `node` must have every one of its own srcs already
// ready (resident leaf, or already computed earlier in this same dispatch pass) -- same
// precondition the normal dispatch loop relies on at that node's real turn, just checked by the
// caller instead of by graph order. Declared here, defined in ggml-cuda.cu.
bool ggml_cuda_compute_node_now(ggml_backend_cuda_context & ctx, ggml_tensor * node);

// Register `node` in ggml-cuda.cu's g_fused_qrot_skip so the main graph-walk
// loop skips it by identity (not a contiguous index range). Used by
// ggml_cuda_try_gdn_conv_prep_fusion after a successful fuse -- production
// graphs interleave unrelated real nodes between SSM_CONV and GDN.
void ggml_cuda_mark_fused_skip(const ggml_tensor * node);

// Runtime gate. Reads env var MAD_USE_R4D_GDN once (cached) and ANDs it with
// ggml_cuda_r4d_available() (compiled in AND current device is gfx1201). Compiled out entirely
// when GGML_HIP_R4D is undefined.
bool r4d_gdn_enabled();

// R4D-path dispatch entry for GGML_OP_GATED_DELTA_NET (the plain, non-fused-cache entry point
// only — ggml_cuda_op_gated_delta_net_fused_cache's K>1 snapshot fusion is out of scope here and
// this function declines whenever the op asks for it; see the .cu file).
//
// NOT on the production dispatch path: llama-graph.cpp/ggml-cuda.cu's GDN cache-fusion pass
// (ggml_cuda_try_gdn_cache_fusion, ggml-cuda.cu:5136) rewrites GGML_OP_GATED_DELTA_NET nodes to
// call ggml_cuda_op_gated_delta_net_fused_cache -> ggml_cuda_op_gated_delta_net_impl(ctx, dst,
// &cache) DIRECTLY, bypassing the plain ggml_cuda_op_gated_delta_net entry this function used to
// gate (measured: MAD_USE_R4D_GDN=1 produced zero mt_gdn_r4d log lines on a real chain run). The
// gate was removed from ggml_cuda_op_gated_delta_net for that reason; see
// ggml_cuda_gdn_r4d_prefix below for the entry point that IS on the production path (called from
// inside ggml_cuda_op_gated_delta_net_impl's use_prefill_chunked branch). This function is kept,
// compiled, and correct for the one caller that still reaches the plain entry (ggml-cuda.cu:4063,
// a GGML_OP_GATED_DELTA_NET node the fusion pass did not rewrite — e.g. a warm-up/reserve build),
// just not gated there anymore, so it is presently unreachable until something re-adds that call.
//
// Returns true iff this call fully handled the op (dst is completely written) — the caller must
// `return` immediately in that case. Returns false to signal a pure eligibility miss (geometry,
// dtype, gate shape, GQA shape) — dst is untouched and the caller should fall through to the
// existing (non-R4D) implementation exactly as if this function had never been called. Unlike
// mt_pagedattn_r4d, this op never mutates shared/persistent state (no KV-cache scatter) before
// deciding eligibility, so there is no "past this point we commit" boundary: every rejection here
// is a clean, side-effect-free `return false`.
bool ggml_cuda_op_gated_delta_net_r4d(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// R4D-path entry for the PRODUCTION dispatch: ggml_cuda_op_gated_delta_net_impl's
// use_prefill_chunked branch (gated_delta_net.cu) calls this to run libr4d's chunked scan over the
// largest 64-token-aligned PREFIX [0, P) of that branch's [0, T0) chunked-prefill window — see
// gated_delta_net.cu's use_prefill_chunked branch and mt_gdn_r4d.cu's header comment for the full
// P/T0/tail composition (r4d prefix, then the existing plain autoregressive kernel over the
// unaligned [P, T0) remainder when T0 > P, then the existing K-snapshot tail unchanged).
//
// q_d/k_d/v_d/g_d/b_d/s_d are the SAME full-tensor base pointers and sq*/sv*/sb*/neqk1/rq3 the
// SAME strides ggml_cuda_op_gated_delta_net_impl already computed for its own kernels — this
// function reads only the first P tokens of each sequence via those strides, exactly like the
// tail kernel reads its own [T0, n_tokens) window via tok_offset. dst_d is the FULL dst buffer;
// dst_seq_stride is the FULL per-sequence token stride (n_tokens, not P) — rows [0, P) of each
// sequence are written, matching launch_gated_delta_net's own tok_offset=0 addressing.
// prefix_state_out receives the state after exactly P tokens, in the same contiguous
// [S_v, S_v, H, n_seqs] layout ggml's own state tensors use (h0/ht), unpermuted.
//
// Returns true iff it fully wrote dst_d's [0,P) rows and prefix_state_out (a clean, side-effect-
// free `return false` otherwise — every eligibility check runs before any kernel launch, same
// invariant as ggml_cuda_op_gated_delta_net_r4d above). Eligibility (checked internally, each
// logged once under MAD_R4D_LOG=1 on rejection): S_v==128, H%Hg==0, rq3==1 (no MTP-verify
// batch-broadcast; declined rather than risked), P a positive multiple of 64, and HEAD
// CONTIGUITY ONLY on q/k/v/g/beta: sq1==S_v, sv1==S_v, sb1==1 — the one thing the strided kernels
// below actually assume (a head's own K/V elements are unit-stride within a token's row). The
// per-token and per-sequence strides (sq2/sq3, sv2/sv3, sb2/sb3) are used exactly as given in
// every read and are NOT required to be any canonical product-of-dims value — production runs
// with q/k/v as strided views into one fused qkv projection buffer (e.g. sv2 = S_v*(2*Hg+H), not
// S_v*H), which an earlier, stricter version of this check wrongly rejected (MAD-406 follow-up:
// that version required sq2==S_v*Hg / sv2==S_v*H / sb2==H, which no real production call ever
// satisfied).
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
        float scale, int64_t dst_seq_stride,
        // K_tail (chain 251 follow-up): the length of the DFlash snapshot tail this SAME
        // GATED_DELTA_NET call will separately run via ggml_cuda_gdn_r4d_tail after this function
        // returns (0 if there is none, K<=1). Used only to (a) let this function's eligibility
        // gate know a primed hand-off must cover P+K_tail rows, not just P, before it may skip
        // the 64-token-alignment floor on P, and (b) pass to r4d_gdn_conv_prep_take_primed's own
        // geometry assert for the same reason -- this function itself never touches the tail
        // rows.
        int64_t K_tail, cudaStream_t stream);

// MAD-406 follow-up (chain 251). Runs the K-token DFlash snapshot tail straight from conv_prep's
// primed buffers (see mt_gdn_r4d.cu's r4d_gdn_conv_prep_primed struct and this function's own
// definition for the full design: it is libr4d's decode-step kernel,
// r4d_gdn_recurrent_update_k128_v128_bf16_fp32state, run as ONE N=1/T=K item with the snapshot
// slot map inverted to match gated_delta_net_cuda's own "slot 0 = most recent, slot s = s tokens
// back" mapping byte-for-byte). Called from gated_delta_net.cu's use_prefill_chunked branch in
// place of its own launch_gated_delta_net<false,true> tail call, ONLY when that branch's
// ggml_cuda_gdn_r4d_prefix call for the SAME dst returned true AND was itself primed (i.e. the
// chunk_scan portion also came from conv_prep's buffers, not from q_d/k_d/v_d/g_d/b_d) -- those
// are the exact conditions under which q_d/k_d/v_d/g_d/b_d are NOT valid to read for the tail
// either. `prefill_state_out` is the state after exactly T0 tokens (same buffer
// ggml_cuda_gdn_r4d_prefix or the existing wmma/scalar prefill kernel already wrote); `state_d`/
// `state_slot_stride` are the SAME variables gated_delta_net.cu already computed for its own
// (unprimed) tail call -- this writes into the identical snapshot cache. `dst_d` is the FULL dst
// buffer; `dst_seq_stride` is the FULL per-sequence token stride (n_tokens), unused in practice
// since this function only ever runs at n_seqs==1.
//
// Returns true iff it fully wrote dst_d's [T0,T0+K) rows and state_d's K snapshot slots -- a
// clean, side-effect-free `false` (caller must fall back to its own launch_gated_delta_net
// <false,true> exactly as before) whenever n_seqs != 1 or conv_prep did not prime (enough of)
// this call.
bool ggml_cuda_gdn_r4d_tail(
        ggml_backend_cuda_context & ctx, ggml_tensor * gdn_dst,
        const float * prefill_state_out, float * state_d, int64_t state_slot_stride,
        float * dst_d, int64_t S_v, int64_t H, int64_t Hg, int64_t T0, int64_t K,
        int64_t n_seqs, float scale, int64_t dst_seq_stride, cudaStream_t stream);

// Non-consuming lookup (mt_gdn_r4d.cu): true iff conv_prep primed buffers exist for this exact
// `gdn` node covering at least `min_T` rows with matching (n_seqs, H, Hg). Lets
// gated_delta_net.cu decide, BEFORE calling ggml_cuda_gdn_r4d_prefix, whether this call may run
// the WHOLE [0, T0) span in one un-chunk-aligned call (primed) instead of the usual
// floor(T0/64)*64-prefix-plus-remainder composition (not primed) -- pass n_tokens (T0 + K_tail)
// as min_T to check the whole call, not just the chunk_scan portion.
bool r4d_gdn_conv_prep_query_primed(int device, cudaStream_t stream, const ggml_tensor * gdn,
                                     int64_t min_T, int64_t n_seqs, int64_t H, int64_t Hg);

// Releases the primed entry for (device, stream) (mt_gdn_r4d.cu) -- call once, after BOTH the
// chunk_scan portion (ggml_cuda_gdn_r4d_prefix) and the K-tail (ggml_cuda_gdn_r4d_tail, when it
// ran) of one primed GATED_DELTA_NET call are done. A no-op if nothing is primed.
void r4d_gdn_conv_prep_release_primed(int device, cudaStream_t stream);

// Logs (MAD_R4D_LOG=1, once per distinct (P, n_tokens, K)) which composition
// ggml_cuda_op_gated_delta_net_impl's use_prefill_chunked branch took: r4d handled [0,P) or not.
void r4d_gdn_prefix_log(int64_t P, int64_t n_tokens, int K, bool handled);

// Logs (MAD_R4D_LOG=1, once per distinct (remainder_len, n_seqs, used_chunked, n_pieces)) which
// kernel the use_prefill_chunked branch's [P, T0) remainder (after an r4d prefix hit) ran through:
// the short-block chunked kernel (gated_delta_net_chunked_cuda, ~52 us/launch), possibly split
// into n_pieces <= GGML_CUDA_GDN_CHUNK_MAX-sized pieces chained through scratch state (n_pieces==1
// means one launch covered the whole remainder; n_pieces==0 means the per-token autoregressive
// kernel ran instead -- used_chunked is false in that case).
void r4d_gdn_remainder_log(int64_t remainder_len, int64_t n_seqs, bool used_chunked, int64_t n_pieces);

// ─────────────────────────────────────────────────────────────────────────
// Conv-side hook: GGML_OP_SSM_CONV, gated by MAD_USE_R4D_GDN_CONV=1 (a SEPARATE env var from
// MAD_USE_R4D_GDN — either can be on without the other).
//
// Investigated for MAD-406 op-mapping task: libr4d's r4d_gdn_conv_prep_w4_h128_bf16 (r4d.h) fuses
// the causal conv + SiLU + q/k/v split + q/k L2-norm + gate prep (g = -exp(A_log)*softplus(a+
// dt_bias) with its per-chunk cumsum, beta = sigmoid(b)) into ONE kernel — exactly what radiance's
// conv_prep() (radiance_gdn.py) calls before kkt_solve. Its C ABI needs, beyond SSM_CONV's own
// two srcs (x = conv_x, wgt = conv weight): `bias` (SSM_CONV already receives an optional fused
// bias via the bias_add_node param — usable), `a`/`b` (the RAW pre-activation alpha/beta, one
// value per (token,head), read by qwen35.cpp's build_layer_attn_linear via
// `build_lora_mm(ssm_alpha, cur, ...)` / `build_lora_mm(ssm_beta, cur, ...)` — a graph branch off
// `cur` directly, never routed through ssm_conv1d at all) and `A_log`/`dt_bias` (per-layer weight
// tensors, model.layers[il].ssm_a / ssm_dt).
//
// GGML_OP_SSM_CONV's own src[] is exactly two tensors (dst->src[0] = conv_x, dst->src[1] = conv
// weight; see ggml_ssm_conv() in ggml.c) — `a`, `b`, `A_log`, `dt_bias` are NOT among them, NOT
// reachable through bias_add_node/silu_dst (the only sibling nodes ggml-cuda.cu's existing
// SSM_CONV+bias+SiLU fusion detector already threads through to this call), and the op dispatch
// contract this adapter is scoped to (mt_gdn_r4d.cu/.cuh + a gate line in ssm-conv.cu) does not
// extend to ggml-cuda.cu's fusion-detection pass or to qwen35.cpp's graph builder — both out of
// this integration's file ownership. So this hook can only ever recognise that it is ineligible:
// it is wired (the gate line in ssm-conv.cu calls it unconditionally when the env is on) but
// r4d_gdn_conv_prep_available() below always returns false today, with the reason logged once
// under MAD_R4D_LOG=1. See mt_gdn_r4d.cu's header comment for the full op-mapping table and the
// overhead reduction (combined q/k cast kernel) taken instead.
//
// Contract: returns true iff it fully wrote `dst` (bias/SiLU fused per bias_add_node/silu_dst,
// same as the plain path) AND populated the persistent per-layer scratch this file already keeps
// (R4D_GDN_Q_BF16 etc.) with libr4d-ready bf16 q/k/v + fp32 g/beta so a later
// ggml_cuda_gdn_r4d_prefix call for the SAME layer's GATED_DELTA_NET node can skip its own
// cast/permute/cumsum and consume that scratch directly. Returns false (dst untouched) to mean
// "not eligible" — the caller must fall through to the existing ggml_cuda_op_ssm_conv body
// exactly as if this had never been called, and ggml_cuda_gdn_r4d_prefix keeps computing its own
// casts from the op's own f32 srcs as it does today.
bool r4d_gdn_conv_enabled();
bool ggml_cuda_op_ssm_conv_r4d(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        ggml_tensor * bias_add_node, ggml_tensor * silu_dst);

// MAD_USE_R4D_GDN_VERIFY=1 (chain 270 follow-up, 2026-09-20). A SEPARATE opt-in from both
// MAD_USE_R4D_GDN and MAD_USE_R4D_GDN_CONV, meant to be run WITH both already on: production gave
// a wrong first token with conv_prep+prefix enabled despite the synthetic primed-tail composition
// test passing, so this diffs the r4d/primed path against genuine, un-skipped ggml computation on
// real graphs. With it set: ggml_cuda_try_gdn_conv_prep_fusion still runs conv_prep and primes
// the hand-off, but identity-skips nothing (every node computes for real); gated_delta_net.cu's
// use_prefill_chunked branch additionally runs the r4d/primed path into scratch and calls
// r4d_gdn_verify_compare below to diff it against the real result. Checked internally, so callers
// need no separate gate of their own.
bool r4d_gdn_verify_enabled();

// Diffs the r4d/primed path's scratch outputs (verify_dst/verify_state, already computed by the
// caller via ggml_cuda_gdn_r4d_prefix + ggml_cuda_gdn_r4d_tail into scratch of the same shape as
// the real dst_d/state_d) against the real ggml result for the SAME GATED_DELTA_NET call
// (dst_d/state_d, computed by gated_delta_net.cu's normal, un-primed path since verify mode never
// lets the r4d path touch them) -- and separately re-derives ggml's own q/k/v/g/beta (via the
// same strided cast/permute kernels ggml_cuda_gdn_r4d_prefix uses when not primed) to diff
// against conv_prep's primed scratch directly. Logs once for layer 0 (parsed from gdn_dst's own
// ggml tensor name) and once per NEW worst-output-error layer seen so far (the last such line
// once the whole model has run is the true worst layer). Diagnostic only (reads whole buffers
// back to the host with plain cudaMemcpy) -- never call this outside MAD_USE_R4D_GDN_VERIFY=1.
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
        bool cache_present, int64_t cache_slot_stride, cudaStream_t stream);

// ─────────────────────────────────────────────────────────────────────────
// Raw-output handoff for ggml-cuda.cu's ml8-4 radiance pattern C "_r4d" fusion variant
// (rdna4_ml8_qrot_gated_norm_tiled_r4d, radiance_quant.h). Protocol (see r4d_gdn_raw_out's own
// doc comment above for the struct):
//
//   1. ggml-cuda.cu's pattern-C PLAN pass -- a whole-cgraph prepass run once per graph compute,
//      BEFORE any node has executed -- calls ggml_cuda_gdn_r4d_mark_raw_wanted(gdn_dst, skip_fp32)
//      for every GATED_DELTA_NET dst tensor whose consuming RMS_NORM chain it has structurally
//      confirmed (shapes/types/op_params only, no data read -- safe pre-execution) will be fused
//      by pattern C. `skip_fp32` is false under MT_ML8_4_RADIANCE_VERIFY=1 (verify mode wants the
//      fp32 path produced too, to diff the two kernels against each other).
//   2. ggml_cuda_gdn_r4d_prefix, at its own (earlier-in-cgraph-order) turn, calls
//      ggml_cuda_gdn_r4d_raw_wanted(dst, &skip_fp32). If wanted AND this call covers the GDN op's
//      ENTIRE per-sequence span in one shot (no K_tail, P == dst_seq_stride -- a partial call's
//      out_bf16 only holds ITS OWN rows, not the whole tensor pattern C's kernel needs), it calls
//      ggml_cuda_gdn_r4d_publish_raw_out(dst, rec) and skips kernel I's fp32 write iff skip_fp32.
//      A partial-coverage call never publishes and never skips fp32, regardless of what was
//      marked -- correctness never depends on stitching a raw record together from two calls.
//   3. Pattern C's EXEC step (this same RMS_NORM's real turn, later in cgraph order) calls
//      ggml_cuda_gdn_r4d_take_raw_out(gdn_dst, &rec) to consume the record. Present => call
//      rdna4_ml8_qrot_gated_norm_tiled_r4d with it instead of reading gdn_dst's (possibly
//      unwritten, when skip_fp32 was honored) fp32 data. Absent => fall back to the plain fp32
//      path exactly as before this feature existed (e.g. this call didn't take the r4d path at
//      all, or covered only part of the span).
//
// Env gate: MAD_USE_R4D_GDN_RAW_OUT=1 (default off; independent of MAD_USE_R4D_GDN/_CONV/_VERIFY).
// Every entry point below is a cheap/no-op when it's off.
bool r4d_gdn_raw_out_enabled();

// Clears BOTH the "wanted" marks and any (should-be-impossible, defensively swept) leftover
// published records. Called once per graph compute, from ggml-cuda.cu's PLAN prepass, before it
// re-marks the new graph -- tensor pointers are only stable for the lifetime of one cgraph build,
// so entries from a previous compute must never be trusted.
void ggml_cuda_gdn_r4d_reset_raw_out_state();

// PLAN-time (step 1 above). A no-op when r4d_gdn_raw_out_enabled() is false.
void ggml_cuda_gdn_r4d_mark_raw_wanted(const ggml_tensor * gdn_dst, bool skip_fp32);

// EXEC-time, called by ggml_cuda_gdn_r4d_prefix only (step 2 above) -- NOT by pattern C directly.
// True iff `gdn_dst` was marked for this graph compute and no record has been published for it
// yet; `skip_fp32_out` (if non-null) receives the flag from that mark.
bool ggml_cuda_gdn_r4d_raw_wanted(const ggml_tensor * gdn_dst, bool * skip_fp32_out);

// EXEC-time, called by ggml_cuda_gdn_r4d_prefix only (step 2 above): publishes a raw-output
// record for `gdn_dst`, consumed once by ggml_cuda_gdn_r4d_take_raw_out below. Must only be
// called when this call covers the GDN op's whole span in one shot -- see the protocol comment.
void ggml_cuda_gdn_r4d_publish_raw_out(const ggml_tensor * gdn_dst, const r4d_gdn_raw_out & rec);

// EXEC-time, called by ggml-cuda.cu's pattern C only (step 3 above): consumes (erases) the
// raw-output record for `gdn_dst`, if any. Returns true and fills `out` iff one was present.
bool ggml_cuda_gdn_r4d_take_raw_out(const ggml_tensor * gdn_dst, r4d_gdn_raw_out * out);

// ─────────────────────────────────────────────────────────────────────────
// Graph-level conv_prep fusion detector (MAD-406 follow-up). Unlike ggml_cuda_op_ssm_conv_r4d
// above (a per-op hook that only sees GGML_OP_SSM_CONV's own two srcs), this operates on the
// whole cgraph starting at a GGML_OP_SSM_CONV node, walking backward from the GATED_DELTA_NET
// node it feeds to find the alpha/beta gate-prep branch too -- see mt_gdn_r4d.cu's header comment
// above this function's definition for the full matched op sequence (with qwen35.cpp line
// numbers) and the geometry/eligibility checks. Runs r4d_gdn_conv_prep_w4_h128_bf16 for real when
// every check passes (task 1/MAD-406 follow-up: llama_model_build_ssm_a_log_sidecars supplies the
// raw A_log the kernel needs via ssm_conv's src[2], see qwen35.cpp's build_layer_attn_linear;
// task 2: the kernel's cstate/cache_idx/has_init protocol is bridged by rebuilding cstate from
// ggml's OWN SSM_CONV src[0] history rows on every call rather than carrying persistent state --
// see r4d_gdn_conv_prep_stage_kernel's comment in the .cu file), writing bf16 q/k/v + fp32 g/beta
// straight into the persistent scratch ggml_cuda_gdn_r4d_prefix consumes (via the
// r4d_gdn_conv_prep_mark_primed/take_primed hand-off, keyed by the GATED_DELTA_NET tensor pointer
// this call feeds) so that function skips its own cast/permute/cumsum for the SAME call. Declines
// (a clean, side-effect-free 0, same invariant every other eligibility miss in this file holds
// to) whenever the structural match fails, the sidecar is missing, or this call's own token count
// isn't a multiple of 64 -- see the .cu file for the exact gates. On success, every
// matched producer is registered in g_fused_qrot_skip via ggml_cuda_mark_fused_skip
// (identity skip -- production graphs interleave unrelated nodes between SSM_CONV
// and GATED_DELTA_NET, so a contiguous index count is the wrong tool). Returns
// nonzero as an internal "this pattern fired" signal; ggml_cuda_try_fuse
// translates that into a literal 0 for the main graph-walk loop, which then
// skips the SSM_CONV anchor via the identity set. Returns 0 to decline (a
// clean, side-effect-free no-op -- nothing is read or written on a 0 return).
//
// Call site (ggml-cuda.cu, before the existing SSM_CONV+SILU fusion):
//
//     if (node->op == GGML_OP_SSM_CONV) {
//         if (ggml_cuda_try_gdn_conv_prep_fusion(cgraph, i, *cuda_ctx) != 0) {
//             return 0;
//         }
//     }
//
// Env gate: MAD_USE_R4D_GDN_CONV=1 (same flag ggml_cuda_op_ssm_conv_r4d uses, via
// r4d_gdn_conv_enabled() -- checked internally, so callers need no separate gate of their own).
int ggml_cuda_try_gdn_conv_prep_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_backend_cuda_context & ctx);

// MAD-406 follow-up (chain 313, concat elision). Call site (ggml-cuda.cu, placed next to the
// existing SSM_CONV hook, BEFORE it in node order since GGML_OP_CONCAT's own turn always precedes
// the SSM_CONV it feeds):
//
//     if (node->op == GGML_OP_CONCAT) {
//         if (ggml_cuda_try_gdn_concat_elide(cgraph, i, *cuda_ctx) != 0) {
//             return 0;
//         }
//     }
//
// Runs at a GGML_OP_CONCAT node's own dispatch turn (earlier than ggml_cuda_try_gdn_conv_prep_
// fusion's own SSM_CONV-anchored turn) -- see this function's definition in mt_gdn_r4d.cu for why
// that is required (identity-skip cannot retroactively un-execute an already-dispatched node) and
// for the full elision design: it locates the SSM_CONV this concat feeds and the state-update cpy
// consumer(s) build_conv_state emits (delta-net-base.cpp:449-559), declines outright (logged once)
// on the n_rs_seq>0 multi-cpy sliding-window shape, then hands the actual eligibility decision to
// ggml_cuda_try_gdn_conv_prep_fusion (called EARLY, anchored at the real SSM_CONV index) so there
// is exactly one, already-existing, already-tested eligibility check -- never two independently-
// written ones that could disagree about whether a given call is fusable. On success it writes the
// state-update cpy's destination bytes itself (straight from qkv_mixed, the concat's own operand)
// and adds concat + its view/cpy chain to the identity-skip set. Env-gated internally (both
// MAD_USE_R4D_GDN and MAD_USE_R4D_GDN_CONV); a 0 return is a clean, side-effect-free decline, same
// invariant every other hook in this file holds to.
int ggml_cuda_try_gdn_concat_elide(
        const ggml_cgraph * cgraph, int node_idx, ggml_backend_cuda_context & ctx);

#else  // GGML_HIP_R4D undefined — stub out

inline void ggml_cuda_mark_fused_skip(const ggml_tensor *) {}
inline bool r4d_gdn_enabled() { return false; }
inline bool ggml_cuda_op_gated_delta_net_r4d(ggml_backend_cuda_context &, ggml_tensor *) {
    // Unreachable: caller must check r4d_gdn_enabled() first, and it is always false here.
    return false;
}
inline bool r4d_gdn_conv_enabled() { return false; }
inline bool ggml_cuda_op_ssm_conv_r4d(ggml_backend_cuda_context &, ggml_tensor *,
                                       ggml_tensor *, ggml_tensor *) {
    return false;
}
inline int ggml_cuda_try_gdn_conv_prep_fusion(
        const ggml_cgraph *, int, ggml_backend_cuda_context &) {
    return 0;
}
inline int ggml_cuda_try_gdn_concat_elide(
        const ggml_cgraph *, int, ggml_backend_cuda_context &) {
    return 0;
}
inline bool r4d_gdn_raw_out_enabled() { return false; }
inline void ggml_cuda_gdn_r4d_reset_raw_out_state() {}
inline void ggml_cuda_gdn_r4d_mark_raw_wanted(const ggml_tensor *, bool) {}
inline bool ggml_cuda_gdn_r4d_raw_wanted(const ggml_tensor *, bool *) { return false; }
inline void ggml_cuda_gdn_r4d_publish_raw_out(const ggml_tensor *, const r4d_gdn_raw_out &) {}
inline bool ggml_cuda_gdn_r4d_take_raw_out(const ggml_tensor *, r4d_gdn_raw_out *) { return false; }
inline bool ggml_cuda_gdn_r4d_prefix(
        ggml_backend_cuda_context &, ggml_tensor *,
        const float *, const float *, const float *, const float *, const float *, const float *,
        float *, float *, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, float, int64_t, int64_t, cudaStream_t) {
    return false;
}
inline bool ggml_cuda_gdn_r4d_tail(
        ggml_backend_cuda_context &, ggml_tensor *,
        const float *, float *, int64_t,
        float *, int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, float, int64_t, cudaStream_t) {
    return false;
}
inline bool r4d_gdn_conv_prep_query_primed(int, cudaStream_t, const ggml_tensor *,
                                            int64_t, int64_t, int64_t, int64_t) {
    return false;
}
inline void r4d_gdn_conv_prep_release_primed(int, cudaStream_t) {}
inline bool r4d_gdn_verify_enabled() { return false; }
inline void r4d_gdn_verify_compare(
        ggml_backend_cuda_context &, ggml_tensor *,
        const float *, const float *, const float *, const float *, const float *,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        const float *, const float *, int64_t,
        const float *, const float *, int64_t,
        const float *,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        bool, int64_t, cudaStream_t) {}
inline void r4d_gdn_prefix_log(int64_t, int64_t, int, bool) {}
inline void r4d_gdn_remainder_log(int64_t, int64_t, bool, int64_t) {}

#endif
