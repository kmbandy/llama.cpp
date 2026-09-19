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

#ifdef GGML_HIP_R4D

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
        float scale, int64_t dst_seq_stride, cudaStream_t stream);

// Logs (MAD_R4D_LOG=1, once per distinct (P, n_tokens, K)) which composition
// ggml_cuda_op_gated_delta_net_impl's use_prefill_chunked branch took: r4d handled [0,P) or not.
void r4d_gdn_prefix_log(int64_t P, int64_t n_tokens, int K, bool handled);

#else  // GGML_HIP_R4D undefined — stub out

inline bool r4d_gdn_enabled() { return false; }
inline bool ggml_cuda_op_gated_delta_net_r4d(ggml_backend_cuda_context &, ggml_tensor *) {
    // Unreachable: caller must check r4d_gdn_enabled() first, and it is always false here.
    return false;
}
inline bool ggml_cuda_gdn_r4d_prefix(
        ggml_backend_cuda_context &, ggml_tensor *,
        const float *, const float *, const float *, const float *, const float *, const float *,
        float *, float *, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
        int64_t, int64_t, float, int64_t, cudaStream_t) {
    return false;
}
inline void r4d_gdn_prefix_log(int64_t, int64_t, int, bool) {}

#endif
