// mt_pagedattn_r4d — R4D-backed path for GGML_OP_PAGED_ATTN_MT.
//
// See mt_pagedattn_r4d.cuh for the dispatch contract and mt_pagedattn.cu:~1794 for the call site
// (checked immediately before the AITER gate, since both are mutually-exclusive whole-op takeovers
// keyed off the KV cache's ggml type).
//
// R4D (ggml-cuda/r4d/r4d.h) speaks its own tensor conventions, different from this op's:
//   - Q is bf16, laid out (num_seqs*q_len, q_heads, head_dim) — SLOT-indexed (row i*q_len+r is
//     seq slot i's r'th query token), not packed to only the active seqs.
//   - Our q (dst->src[0]) is F16, packed (head_dim, n_heads, sum of active q_lens) — only the
//     active seqs' rows exist, back-to-back in seq-slot order.
//   - R4D's KV cache is fp8-e4m3, (num_blocks, kv_heads, block_size, 2*head_dim) with K then V
//     per slot — this is exactly what the new GGML_TYPE_R4D_FP8_KV cache type holds (k_cache is
//     the COMBINED K|V tensor; v_cache is allocated but unused, matching the AITER TURBO4_FP8
//     precedent of one tensor doing double duty).
//   - R4D wants one int32 per seq SLOT (seqused_k) giving that slot's total context INCLUDING its
//     query tokens, 0 for a slot that is not live this call — rows with seqused_k<=0 come back
//     zero (decode's split-KV combine kernel zero-fills them; prefill skips them outright, per
//     r4d.h's seqused_k doc comment).
//
// So this file's real job, on top of calling into R4D, is translating between "packed, F16,
// active-seqs-only" (this op's contract) and "slot-indexed, bf16, every slot" (R4D's contract) —
// the expand/compact kernels below — plus scattering K_cur/V_cur into R4D's fp8 layout first
// (mt_pagedattn_r4d_scatter.cuh, owned by another agent working this same integration).
//
// ── The "shared q_len" assumption, and how it is checked without a device readback ──────────
//
// R4D's q buffer is q_len*num_seqs rows: ALL live seqs must share one q_len (r4d.h: "All seqs
// share ONE q_len"). Our op's contract lets q_lens vary per seq (dst->src[5], device-resident,
// no host mirror at op-execution time) — so per-call eligibility for this path hinges on knowing,
// FROM THE HOST, without reading device memory, that every active seq's q_len equals some single
// value.
//
// op_params[4] (max_q_len) is populated host-side at graph-build time by
// src/llama-graph.cpp:3565:
//     cur->op_params[4] = ubatch.equal_seqs() ? (int32_t) ubatch.n_seq_tokens
//                                             : (int32_t) ubatch.n_tokens;
// i.e. it is the TRUE per-seq q_len only when ubatch.equal_seqs() was true at build time (every
// seq in the ubatch has exactly n_seq_tokens query tokens); otherwise it is n_tokens, the TOTAL
// token count across all seqs — not a per-seq value at all. Nothing in op_params says which case
// produced it.
//
// This file distinguishes the two WITHOUT reading q_lens back from the device: op_params[6]
// (n_seqs_active, src/llama-graph.cpp:649, "the REAL number of live/active sequences in this
// ubatch") and q->ne[2] (this op's total active-query-token count, i.e. the packed row count —
// see the AITER path's `num_q_tokens = k_cur->ne[2]` for the equivalent quantity, mirrored here
// off q itself since head_dim/n_heads also come from q) are BOTH host-visible. If
//     op_params[4] * num_active == q->ne[2]
// then the "every active seq has exactly op_params[4] query tokens" reading is the ONLY reading
// consistent with the observed total, because q_lens are always >= 0 for a live-batch op (a
// non-uniform split summing to the same total, or n_tokens itself standing in as q_len, would
// require num_active to divide unevenly or be off by the same coincidence for every batch this op
// ever sees — treated as impossible here). When ubatch.equal_seqs() was in fact false at build
// time, op_params[4] holds n_tokens, and the equality above holds only in the degenerate
// num_active==1 case (where equal_seqs is moot anyway) — every other case fails the check and this
// path declines cleanly. This is the host-side route the task calls for in preference to a device
// uniformity kernel, and it is a strictly necessary condition for R4D's assumption, checked at
// zero device-read cost.
//
// num_active itself is op_params[6] (src/llama-graph.cpp:649, same 0-means-unset convention),
// clamped to block_tables->ne[1] (the cache's static n_seq_max) exactly as the AITER path does
// (mt_pagedattn_aiter.cu:1056-1058, "num_seqs_dispatch") — a cache can never legitimately have
// more live sequences than it was constructed with.
//
// ── Fixed: op_params[4] under tensor-split-attn, and non-uniform q_lens ─────────────────────
//
// Two gaps used to live here, both now closed at the source (llm_graph_input_attn_kv::
// update_paged_attn_q_lens(), src/llama-graph.cpp): (a) under tensor-split-attn, op_params[4]
// used to be written with a plain `cur->op_params[4] = ...` field assignment on the meta/world
// tensor at graph-build time only, so a per-device clone materialized afterward (warm-up/reserve
// build, or a reused-not-rebuilt graph) could see it frozen at 0 forever; (b) whenever
// ubatch.equal_seqs() was false at build time there was no host-visible way at all to tell
// "every active seq happens to share one q_len" (e.g. two 1-token decode seqs) from "seqs
// genuinely have different q_lens" — both looked like op_params[4]==n_tokens.
//
// update_paged_attn_q_lens() now runs every set_input (same per-ubatch, per-device-clone
// propagation as op_params[5]/[6], via ggml_backend_meta_buffer_set_op_param_i32,
// ggml-backend-meta.cpp:774) and computes uniformity straight from the host-side q_lens mirror
// the paged cache already holds — no device readback. It pushes:
//   op_params[4] = the shared q_len IF every active seq (q_lens[s] > 0) has the same q_len, ELSE
//                  the ubatch's total token count (an always-safe, always-oversized upper bound)
//                  — deliberately NOT 0 in the non-uniform case: this keeps op_params[4]'s value
//                  semantics identical to build_attn's old graph-build-time assignment
//                  (equal_seqs() ? n_seq_tokens : n_tokens), so mt_pagedattn.cu's tile/decode
//                  gates (which read op_params[4] purely as that upper bound, owned by another
//                  agent, out of scope here) see no behavior change. op_params[7] bit0 below is
//                  the ONLY uniformity signal this file (or any consumer) should branch on —
//                  op_params[4] alone is never enough to tell uniform from non-uniform anymore.
//   op_params[7] = flags: bit0 = q_lens uniform (op_params[4] holds the exact shared value; when
//                  clear, op_params[4] is only a safe upper bound), bit1 = pure decode (every
//                  active seq has q_len==1), bit2 = n_seq_max > 8 (op_params[8..15] unset)
//   op_params[8..15] = per-seq q_len for cache slots 0..7 (0 for an inactive slot), when
//                  n_seq_max <= 8
//
// The eligibility gate below decides uniform-vs-per-seq from flags ALONE, never from op_params[4]:
// bit0 set -> single uniform-q_len launch exactly as before; bit0 clear and n_seq_max <= 8 ->
// per-seq dispatch (one libr4d launch per active sequence, see the "Per-seq dispatch" section
// below); bit2 set (n_seq_max > 8, non-uniform) -> decline, no host-visible way to give R4D a
// single q_len and per-seq q_lens weren't packed.
// flags==0 (old graph / cold clone never touched by update_paged_attn_q_lens()) falls back to
// the pre-existing op_params[4]/[6]/q->ne[2] derivation, kept verbatim below.
//
// ── turbo4kv mode: serving GGML_TYPE_TURBO4_FP8_BS256 (MAD_USE_R4D_ATTN_TURBO4=1) ──────────────
//
// Everything above this point was written against GGML_TYPE_R4D_FP8_KV, R4D's own combined-K|V
// fp8 cache. This adapter can ALSO serve the production KV cache, GGML_TYPE_TURBO4_FP8_BS256, via
// r4d_attn_{prefill,decode}_h256_gqa6_turbo4kv — the same R4D library, a different pair of kernel
// entries compiled for turbo4's on-disk layout. Gated by r4d_turbo4_enabled() (env
// MAD_USE_R4D_ATTN_TURBO4) independently of MAD_USE_R4D, and tracked throughout the dispatch
// function by the local `turbo4` bool. Three differences from the R4D_FP8_KV path, all confined
// to the places this file's helper `turbo4` bool is read:
//   (1) Cache layout: turbo4 keeps K and V in SEPARATE cache buffers (dst->src[1]/src[2] — v_cache,
//       unused for R4D_FP8_KV, is live here) of 162-byte records per (paged block, slot, kv head):
//       fp16 per-vector scale, 128 bytes of 4-bit centroid indices, 32 sign bytes, addressed at
//       byte offset ((block*16 + slot_in_block)*n_kv_heads + kv_head) * 162. R4DArgs' KVP=2 fields
//       (v_cache, k_lut, v_lut, kv_slot_stride) carry this layout in; kv_block_stride/kv_head_stride
//       switch from R4D_FP8_KV's ELEMENT strides to BYTE strides for this mode (r4d.h). The scatter
//       itself is NOT this file's own kernel — it calls back into the AITER path's exported
//       mt_aiter_scatter_kv_turbo4_fp8_launch (mt_pagedattn_aiter.cu) so both adapters write bit-
//       identical records, rather than maintaining a second copy of that kernel here.
//   (2) Per-(layer, K|V) centroid LUTs: mt_turbo_fp8::get_lut_device_ptr(il, KV_K/KV_V), il parsed
//       from k_cache's tensor name via the AITER path's exported mt_aiter_parse_layer_from_kv_cache_
//       name — same registry, same parser, same layer binding the AITER path uses for this cache
//       type, so a mixed AITER/R4D deployment can never disagree about which LUTs a layer gets.
//   (3) Optional Hadamard Q pre-rotation: when mt_turbo_fp8::hadamard_required() is true, K was
//       FWHT-rotated at scatter time (inside mt_aiter_scatter_kv_turbo4_fp8_launch); Q must be
//       rotated identically before attention for (QH)·(KH)^T = QK^T to hold. Done once, up front,
//       over the WHOLE packed Q buffer (both the uniform and per-seq launch bodies below read out
//       of the same rotated copy, at their own offsets) — mirrors the AITER path's own Q rotation
//       exactly. V is never rotated. R4D_FP8_KV never takes this branch: it has no Hadamard mode.
// Nothing else changes: eligibility's shape/GQA/q_len gates, the expand/compact/cast kernels, the
// uniform-vs-per-seq split, and MAD_R4D_ATTN_CALLFIX all apply identically to both cache types.

#include "common.cuh"
#include "mt_pagedattn_r4d.cuh"

#ifdef GGML_HIP_R4D

#include "r4d/ggml-r4d.h"
#include "mt_pagedattn_r4d_scatter.cuh"
// turbo4kv mode (MAD_USE_R4D_ATTN_TURBO4=1): reuses the AITER path's turbo4_fp8 scatter kernel and
// layer-name parser (mt_pagedattn_aiter.cu) so both adapters write the exact same on-disk record
// layout, plus the centroid-LUT registry and the Hadamard Q pre-rotation helper it depends on. See
// this file's header comment for the mode's layout and rotation contract.
#include "mt_pagedattn_aiter.cuh"
#include "mt_turbo_fp8_lut_registry.h"
#include "turbo_fp8_hadamard.cuh"

#include <algorithm>
#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace mt {

// ─────────────────────────────────────────────────────────────────────────
// Runtime gate
// ─────────────────────────────────────────────────────────────────────────
bool r4d_backend_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_USE_R4D");
        const bool   opted_in = env != nullptr && env[0] == '1';
        return opted_in && ggml_cuda_r4d_available();
    }();
    return enabled;
}

// MAD_USE_R4D_ATTN_TURBO4=1: let this adapter also serve GGML_TYPE_TURBO4_FP8_BS256 (the
// production KV cache type), via the r4d_*_turbo4kv entries and the AITER path's turbo4_fp8
// scatter kernel (mt_aiter_scatter_kv_turbo4_fp8_launch, mt_pagedattn_aiter.cu). Gated separately
// from r4d_backend_enabled() (MAD_USE_R4D) since this mode has its own, independently-rolled-out
// eligibility. Only meaningful when built with GGML_HIP_AITER — the scatter launch this mode calls
// into only exists for real then (mt_pagedattn_aiter.cuh's stub otherwise).
bool r4d_turbo4_enabled() {
#ifdef GGML_HIP_AITER
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_USE_R4D_ATTN_TURBO4");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
#else
    return false;
#endif
}

namespace {

// ─────────────────────────────────────────────────────────────────────────
// Persistent, grow-only, capture-safe scratch — same pattern (and same
// reasoning: MAD-288, mt_pagedattn_aiter.cu:895-955) as mt_aiter_persist_get,
// duplicated here rather than shared because that file is owned by another
// agent working concurrently on this same integration. Keyed by
// (device, stream) so the two overlapping meta contexts on one device
// (GGML_META_OVERLAP) never share a buffer.
struct r4d_persist_buf {
    void * ptr   = nullptr;
    size_t bytes = 0;
};
enum r4d_persist_slot {
    R4D_PERSIST_Q_BF16 = 0,   // expanded, slot-indexed bf16 Q: (num_seqs*q_len, q_heads, head_dim)
    R4D_PERSIST_OUT_BF16,     // R4D's bf16 output, same shape as Q above
    R4D_PERSIST_SEQUSED_K,    // int32, one per seq slot
    R4D_PERSIST_CU_SEQLENS,   // int32, num_seqs+1 — prefix sum of q_lens (packed-row offsets)
    R4D_PERSIST_DECODE_SCRATCH, // split-KV partials, decode only, sized in bytes
    R4D_PERSIST_Q_ROT,       // turbo4kv + mt_turbo_fp8::hadamard_required() only: FWHT-rotated
                             // copy of the whole packed F16 Q buffer (K was rotated at scatter
                             // time by mt_aiter_scatter_kv_turbo4_fp8_launch)
    // MAD_R4D_ATTN_CALLFIX=1 only (see r4d_callfix_enabled() below): compact grid.z from the
    // cache's static n_seq_max down to the actual live-sequence count, mirroring radiance's
    // R4DAttentionMetadataBuilder._plan() which only ever launches with num_seqs==num_active.
    // See mt_pagedattn_r4d.cu's header-adjacent comment and tests/test-r4d-attn.hip.cpp's
    // "production prefill bench" for the investigation this closes.
    R4D_PERSIST_SEQ_MAP,            // int32, num_active — compacted slot j -> original slot id
    R4D_PERSIST_SEQUSED_COMPACT,    // int32, num_active — seqused_k reindexed by compacted j
    R4D_PERSIST_BLOCK_TABLE_COMPACT, // int32, num_active*max_blocks — block_table rows reindexed by j
    R4D_PERSIST_COUNT
};
static std::mutex g_r4d_persist_mutex;
static std::map<std::pair<int, cudaStream_t>, std::array<r4d_persist_buf, R4D_PERSIST_COUNT>> g_r4d_persist;

// MAD-LAB 2026-09-20 diag: WP_ALLOC_LOG=1 attribution for this grow-only
// cache, mirroring mt_aiter_persist_alloc_log's format/reasoning exactly
// (mt_pagedattn_aiter.cu) so the same journal grep finds both. wp_alloc_log
// has internal linkage in ggml-cuda.cu and is not reachable from this TU,
// hence the local twin. NOTE (unlike the aiter twin): this path (a) has no
// MT_AITER_PERSIST_MAX_MB-style cap -- a single slot can grow without bound
// if n_elems ever tracks something unbounded (e.g. context length rather
// than a fixed num_seqs/head_dim shape) -- and (b) still uses the OLD
// doubling growth (max(need, b.bytes*2)) that the aiter twin's 2026-09-20
// comment (mt_pagedattn_aiter.cu) says permanently overshoots and, on top
// of that, LEAKS the old buffer on every grow step (b.ptr is overwritten
// below with no cudaFree of the previous allocation -- same bug the aiter
// twin used to have before that fix). Diagnostic-only here: no behavior
// change, since this file is gated behind MAD_USE_R4D=1/GGML_HIP_R4D and a
// fix belongs with whoever owns this integration (see the "duplicated here
// rather than shared" comment above r4d_persist_buf).
static void r4d_persist_alloc_log(int device, size_t old_bytes, size_t new_bytes) {
    static const bool enabled = [] {
        const char * e = std::getenv("WP_ALLOC_LOG");
        return e != nullptr && e[0] == '1';
    }();
    if (!enabled) {
        return;
    }
    struct timespec ts; clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tmv; localtime_r(&ts.tv_sec, &tmv);
    std::fprintf(stderr, "wp alloc-log %02d:%02d:%02d.%03ld r4d_persist_grow device=%d size=%.1fMiB extra=%.1fMiB\n",
                 tmv.tm_hour, tmv.tm_min, tmv.tm_sec, ts.tv_nsec / 1000000, device,
                 new_bytes / 1048576.0, old_bytes / 1048576.0);
}

template <typename T>
static T * r4d_persist_get(int device, cudaStream_t stream, r4d_persist_slot slot, size_t n_elems) {
    const size_t need = n_elems * sizeof(T);
    std::lock_guard<std::mutex> lock(g_r4d_persist_mutex);
    r4d_persist_buf & b = g_r4d_persist[std::make_pair(device, stream)][slot];
    if (need > b.bytes) {
        size_t bytes = std::max(need, b.bytes * 2);
        bytes = (bytes + (1u << 20) - 1) & ~(size_t) ((1u << 20) - 1);
        void * ptr = nullptr;
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        r4d_persist_alloc_log(device, b.bytes, bytes);
        b.ptr   = ptr;
        b.bytes = bytes;
    }
    return (T *) b.ptr;
}

// ─────────────────────────────────────────────────────────────────────────
// Kernel A: build cu_seqlens (prefix sum of q_lens, packed-row offsets) and
// seqused_k (context_lens gated by liveness) in one pass. num_seqs is small
// (the cache's static n_seq_max) so a single-thread sequential scan is
// cheap and, crucially, allocation- and sync-free — capture safe.
__global__ void r4d_build_cu_seqlens_and_seqused_kernel(
        const int32_t * __restrict__ q_lens,
        const int32_t * __restrict__ context_lens,
        int32_t * __restrict__ cu_seqlens,
        int32_t * __restrict__ seqused_k,
        int num_seqs) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int32_t acc = 0;
    for (int i = 0; i < num_seqs; ++i) {
        cu_seqlens[i] = acc;
        const int32_t ql = q_lens[i];
        if (ql > 0) {
            acc += ql;
            seqused_k[i] = context_lens[i];
        } else {
            seqused_k[i] = 0;
        }
    }
    cu_seqlens[num_seqs] = acc;
}

// float <-> bf16 helpers, same intrinsics ggml_cuda_cast uses (convert.cuh) for f16<->bf16.
__device__ __forceinline__ nv_bfloat16 r4d_f16_to_bf16(__half h) {
    return __float2bfloat16(__half2float(h));
}
__device__ __forceinline__ __half r4d_bf16_to_f16(nv_bfloat16 b) {
    return __float2half(__bfloat162float(b));
}

// Kernel B: expand — packed F16 Q (active seqs only, back-to-back in seq-slot order) into R4D's
// slot-indexed bf16 Q (num_seqs*q_len rows, one q_len-sized block per slot regardless of
// liveness). grid = (num_seqs, q_len); each block copies one destination row
// (q_heads*head_dim elements). Inactive slots (q_lens[seq]<=0) are left untouched — R4D's
// seqused_k<=0 for that slot makes its content irrelevant (skipped outright in prefill,
// zero-filled by the split-KV combine in decode — see r4d.h).
__global__ void r4d_expand_q_kernel(
        const __half * __restrict__ q_packed,   // (sum_active_q_lens, q_heads, head_dim)
        nv_bfloat16 * __restrict__  q_slot,      // (num_seqs*q_len, q_heads, head_dim)
        const int32_t * __restrict__ q_lens,
        const int32_t * __restrict__ cu_seqlens,
        int q_len, int row_elems) {
    const int seq = blockIdx.x;
    const int r   = blockIdx.y;
    if (q_lens[seq] <= 0) {
        return;
    }
    const long src_row = (long) cu_seqlens[seq] + r;
    const long dst_row = (long) seq * q_len + r;
    const __half   * src = q_packed + src_row * row_elems;
    nv_bfloat16    * dst = q_slot   + dst_row * row_elems;
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        dst[e] = r4d_f16_to_bf16(src[e]);
    }
}

// Kernel C: compact — the reverse of kernel B, R4D's slot-indexed bf16 output back into this op's
// packed F16 dst (active seqs only). Inactive slots contribute no output rows (the packed layout
// has none for them) so they are simply never visited.
__global__ void r4d_compact_out_kernel(
        const nv_bfloat16 * __restrict__ out_slot,  // (num_seqs*q_len, q_heads, head_dim)
        __half * __restrict__            out_packed, // (sum_active_q_lens, q_heads, head_dim)
        const int32_t * __restrict__ q_lens,
        const int32_t * __restrict__ cu_seqlens,
        int q_len, int row_elems) {
    const int seq = blockIdx.x;
    const int r   = blockIdx.y;
    if (q_lens[seq] <= 0) {
        return;
    }
    const long src_row = (long) seq * q_len + r;
    const long dst_row = (long) cu_seqlens[seq] + r;
    const nv_bfloat16 * src = out_slot   + src_row * row_elems;
    __half            * dst = out_packed + dst_row * row_elems;
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        dst[e] = r4d_bf16_to_f16(src[e]);
    }
}

// Kernel D/E: per-seq mode casts. With num_seqs==1 (one libr4d launch per active sequence, see
// the "Per-seq dispatch" section below), R4D's slot-indexed and this op's packed row layouts
// coincide — there is only one "slot" and it's live — so no cu_seqlens/q_lens skip-inactive-
// slot logic is needed, just a straight elementwise cast over that sequence's own q_len_s rows
// (offsets computed host-side, see the per-seq launch loop).
__global__ void r4d_cast_f16_to_bf16_kernel(
        const __half * __restrict__ src, nv_bfloat16 * __restrict__ dst, int row_elems, int n_rows) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const __half   * s = src + (long) row * row_elems;
    nv_bfloat16    * d = dst + (long) row * row_elems;
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        d[e] = r4d_f16_to_bf16(s[e]);
    }
}
__global__ void r4d_cast_bf16_to_f16_kernel(
        const nv_bfloat16 * __restrict__ src, __half * __restrict__ dst, int row_elems, int n_rows) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const nv_bfloat16 * s = src + (long) row * row_elems;
    __half             * d = dst + (long) row * row_elems;
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        d[e] = r4d_bf16_to_f16(s[e]);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// MAD_R4D_ATTN_CALLFIX=1 (default off) — grid.z compaction, A/B-gated.
//
// mt_pagedattn_r4d.cu always passes R4DArgs.num_seqs = block_tables->ne[1], the paged cache's
// STATIC n_seq_max (see the uniform-path comment at its use below and the identical, older
// precedent in mt_pagedattn_aiter.cu's num_seqs_dispatch, ~:1021-1034) — required because R4D's
// block_table / seqused_k / slot-indexed Q are addressed by absolute seq-SLOT id, not compacted,
// and a live sequence can sit at any slot. radiance_r4d_attn.py's R4DAttentionMetadataBuilder
// never has this problem: its `_plan()` and `R4DAttentionImpl.forward()` pass num_seqs = the
// ACTUAL live-sequence count for that launch (vLLM's own block_table/seq_lens tensors are already
// packed to only the live requests), so grid.z there is never padded with idle, immediately-
// returning workgroups (r4d_attn_prefill_kernel: "if (ctx <= 0) return;"). tests/test-r4d-attn.
// hip.cpp's "production prefill bench" isolated exactly this field as the one remaining,
// verifiable difference between the two call sites once max_ctx/splits/scratch (all read-but-
// unused by prefill) and kv strides/descales/q_len were confirmed identical.
//
// This is deliberately NOT the always-on behavior: shrinking grid.z to the live count while still
// addressing block_table/seqused_k by absolute slot index would read the WRONG sequences'
// context whenever live slots are not packed at 0..num_active-1 (not guaranteed by llama.cpp's
// paged cache in general). So instead of narrowing grid.z directly, this gate REMAPS: it builds a
// compacted (block_table, seqused_k) — one row per LIVE slot, in the same slot-ascending order the
// packed Q/out layout already uses — via r4d_build_compact_seqmap_kernel +
// r4d_gather_block_table_kernel, and expands/compacts Q/out against that compacted indexing via
// r4d_expand_q_compact_kernel / r4d_compact_out_compact_kernel. R4DArgs.num_seqs then becomes the
// true live count, matching radiance's grid.z exactly, with no risk of cross-slot misreads.
//
// Off by default (MAD_R4D_ATTN_CALLFIX unset or not "1") so the existing, always-correct padded
// path stays the default; set it to A/B this specific fix in isolation.
bool r4d_callfix_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("MAD_R4D_ATTN_CALLFIX");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

// Single-thread sequential scan (num_seqs is small — the cache's static n_seq_max — so this is
// cheap, alloc- and sync-free, and capture-safe, same reasoning as
// r4d_build_cu_seqlens_and_seqused_kernel above). Writes, for each live slot in ascending slot
// order, its original slot id into seq_map[j] and its context length into seqused_compact[j].
// j runs 0..num_active-1 by construction (num_active live slots among num_seqs), matching the
// order the packed Q/out rows are already laid out in (cu_seqlens/expand_q iterate slots the same
// way).
__global__ void r4d_build_compact_seqmap_kernel(
        const int32_t * __restrict__ q_lens,
        const int32_t * __restrict__ context_lens,
        int32_t * __restrict__ seq_map,
        int32_t * __restrict__ seqused_compact,
        int num_seqs) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int32_t j = 0;
    for (int s = 0; s < num_seqs; ++s) {
        if (q_lens[s] > 0) {
            seq_map[j]          = s;
            seqused_compact[j]  = context_lens[s];
            ++j;
        }
    }
}

// Reindex block_table rows from original slot id to compacted j. grid = (num_active); each block
// copies one row (max_blocks ints) via seq_map[j] -> original slot id.
__global__ void r4d_gather_block_table_kernel(
        const int32_t * __restrict__ block_table_full,
        const int32_t * __restrict__ seq_map,
        int32_t * __restrict__       block_table_compact,
        int max_blocks) {
    const int j = blockIdx.x;
    const int s = seq_map[j];
    const int32_t * __restrict__ src = block_table_full     + (size_t) s * (size_t) max_blocks;
    int32_t *       __restrict__ dst = block_table_compact  + (size_t) j * (size_t) max_blocks;
    for (int b = threadIdx.x; b < max_blocks; b += blockDim.x) {
        dst[b] = src[b];
    }
}

// r4d_expand_q_kernel's compacted twin: same packed-F16 -> slot-indexed-bf16 job, but the
// destination slot is the compacted index j (blockIdx.x, 0..num_active-1) rather than the
// original seq slot id — seq_map[j] gives the original id needed to read q_lens/cu_seqlens (the
// packed source layout is still keyed by original slot id, unchanged by this gate).
__global__ void r4d_expand_q_compact_kernel(
        const __half * __restrict__ q_packed,
        nv_bfloat16 * __restrict__  q_slot,
        const int32_t * __restrict__ q_lens,
        const int32_t * __restrict__ cu_seqlens,
        const int32_t * __restrict__ seq_map,
        int q_len, int row_elems) {
    const int j   = blockIdx.x;
    const int r   = blockIdx.y;
    const int seq = seq_map[j];
    if (q_lens[seq] <= 0) {
        return; // defensive; seq_map only ever names live slots by construction
    }
    const long src_row = (long) cu_seqlens[seq] + r;
    const long dst_row = (long) j * q_len + r;
    const __half   * src = q_packed + src_row * row_elems;
    nv_bfloat16    * dst = q_slot   + dst_row * row_elems;
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        dst[e] = r4d_f16_to_bf16(src[e]);
    }
}

// r4d_compact_out_kernel's compacted twin — the reverse of the kernel above.
__global__ void r4d_compact_out_compact_kernel(
        const nv_bfloat16 * __restrict__ out_slot,
        __half * __restrict__            out_packed,
        const int32_t * __restrict__ q_lens,
        const int32_t * __restrict__ cu_seqlens,
        const int32_t * __restrict__ seq_map,
        int q_len, int row_elems) {
    const int j   = blockIdx.x;
    const int r   = blockIdx.y;
    const int seq = seq_map[j];
    if (q_lens[seq] <= 0) {
        return;
    }
    const long src_row = (long) j * q_len + r;
    const long dst_row = (long) cu_seqlens[seq] + r;
    const nv_bfloat16 * src = out_slot   + src_row * row_elems;
    __half            * dst = out_packed + dst_row * row_elems;
    for (int e = threadIdx.x; e < row_elems; e += blockDim.x) {
        dst[e] = r4d_bf16_to_f16(src[e]);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Diagnostics — MAD_R4D_LOG=1: log once per distinct (q_len, num_seqs, max_ctx) the path taken.
bool r4d_log_enabled() {
    static const bool e = [] { const char * v = std::getenv("MAD_R4D_LOG"); return v && v[0] == '1'; }();
    return e;
}
void r4d_log_once(int q_len, int num_seqs, int max_ctx, bool decode, int splits, long scratch_bytes, int rc,
                   const char * kernel_name) {
    if (!r4d_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::tuple<int, int, int>> seen;
    const auto key = std::make_tuple(q_len, num_seqs, max_ctx);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
        return;
    }
    std::fprintf(stderr,
        "[mt_pagedattn_r4d] mode=uniform q_len=%d num_seqs=%d max_ctx=%d path=%s kernel=%s splits=%d scratch_bytes=%ld rc=%d\n",
        q_len, num_seqs, max_ctx, decode ? "decode" : "prefill", kernel_name, splits, scratch_bytes, rc);
}

// MAD_R4D_LOG=1: log once per distinct (num_seqs_static, num_active) pair that MAD_R4D_ATTN_CALLFIX
// actually compacted (i.e. num_active < num_seqs_static — nothing to log when the cache's static
// n_seq_max already equals the live count, since the gate is then a no-op).
void r4d_log_callfix_once(int num_seqs_static, int num_active) {
    if (!r4d_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::pair<int, int>> seen;
    const auto key = std::make_pair(num_seqs_static, num_active);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
        return;
    }
    std::fprintf(stderr, "r4d attn callfix: num_seqs %d -> %d\n", num_seqs_static, num_active);
}

// Per-seq mode: one libr4d launch per active sequence, each with its own q_len/decode-vs-prefill
// choice, so there's no single (q_len, decode, rc) to summarize — log the batch shape once per
// distinct (num_seqs, n_launched, max_ctx) instead.
void r4d_log_once_per_seq(int num_seqs, int n_launched, int max_ctx) {
    if (!r4d_log_enabled()) {
        return;
    }
    static std::mutex mu;
    static std::set<std::tuple<int, int, int>> seen;
    const auto key = std::make_tuple(num_seqs, n_launched, max_ctx);
    std::lock_guard<std::mutex> lock(mu);
    if (!seen.insert(key).second) {
        return;
    }
    std::fprintf(stderr,
        "[mt_pagedattn_r4d] mode=per-seq n=%d num_seqs=%d max_ctx=%d\n",
        n_launched, num_seqs, max_ctx);
}

// Diagnostics — MAD_R4D_LOG=1: log ONCE PER DISTINCT REASON why a call was rejected at the
// eligibility gate (as opposed to r4d_log_once above, which logs accepted calls). Without this,
// a TP config where every call falls through silently gives no host-side signal at all about
// which check failed -- the CUDA-side reader has no other way to report it, since a `return
// false` here is indistinguishable, from the caller's perspective, from "this op just isn't
// R4D-eligible by design" (e.g. a non-R4D cache type on every other model). Keyed by the reason
// string alone (not the values) so a hot loop that fails the same check every call logs exactly
// once, not once per distinct value combination.
void r4d_log_reject_once(const char * reason, const char * fmt, ...) {
    if (!r4d_log_enabled()) {
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
    std::fprintf(stderr, "[mt_pagedattn_r4d] REJECT reason=%s %s\n", reason, buf);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
// Dispatch entry
// ─────────────────────────────────────────────────────────────────────────
bool ggml_cuda_op_paged_attn_mt_r4d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_cache       = dst->src[1];
    // v_cache (dst->src[2]) is allocated but unused for GGML_TYPE_R4D_FP8_KV — k_cache holds the
    // combined K|V tensor (see this file's header comment and mt_pagedattn_r4d.cuh). For turbo4kv
    // mode (GGML_TYPE_TURBO4_FP8_BS256) it IS used: K and V live in separate cache buffers there.
    const ggml_tensor * v_cache       = dst->src[2];
    const ggml_tensor * block_tables  = dst->src[3];
    const ggml_tensor * context_lens  = dst->src[4];
    const ggml_tensor * q_lens        = dst->src[5];
    const ggml_tensor * k_cur         = dst->src[6];
    const ggml_tensor * v_cur         = dst->src[7];
    const ggml_tensor * slot_mapping  = dst->src[8];

    const float * op_params_f = (const float *)(dst->op_params);
    const float   scale               = op_params_f[0];
    const int32_t block_size          = ((const int32_t *)(op_params_f + 1))[0];
    const int32_t max_bps             = ((const int32_t *)(op_params_f + 2))[0];
    const int32_t n_kv_heads          = ((const int32_t *)(op_params_f + 3))[0];
    const int32_t max_q_len_param     = ((const int32_t *)(op_params_f + 4))[0];
    const int32_t max_ctx_len_param   = ((const int32_t *)(op_params_f + 5))[0];
    const int32_t n_seqs_active_param = ((const int32_t *)(op_params_f + 6))[0];
    // op_params[7]/[8..15]: llm_graph_input_attn_kv::update_paged_attn_q_lens()
    // (src/llama-graph.cpp) — see this file's header comment for the full contract. flags bit0 =
    // q_lens uniform (op_params[4] meaningful), bit1 = pure decode, bit2 = n_seq_max > 8
    // (op_params[8..15] not populated). 0 means "unset" (old graph / cold clone), same convention
    // as op_params[4]/[5]/[6].
    const int32_t flags = ((const int32_t *)(op_params_f + 7))[0];
    int32_t per_seq_q_len[8];
    for (int s = 0; s < 8; ++s) {
        per_seq_q_len[s] = ((const int32_t *)(op_params_f + 8 + s))[0];
    }

    const int head_dim  = (int) q->ne[0];
    const int n_heads    = (int) q->ne[1];
    const int total_q_tokens = (int) q->ne[2];
    // block_tables->ne[1] is the paged cache's STATIC n_seq_max (see the identical comment in
    // mt_pagedattn_aiter.cu:1021-1034) — the correct bound for R4D's num_seqs, which indexes
    // block_table/seqused_k/the expanded Q buffer by SLOT, not by a compacted live-seq count.
    const int num_seqs   = (int) block_tables->ne[1];

    // ── Eligibility (return false => caller falls through to the existing paths) ──────────────
    // turbo4kv mode (MAD_USE_R4D_ATTN_TURBO4=1): also accept the production KV cache type,
    // GGML_TYPE_TURBO4_FP8_BS256, alongside the existing R4D_FP8_KV cache. `turbo4` is read
    // everywhere below that the two modes' cache layouts / R4DArgs fields / kernel entries differ
    // (scatter, Q rotation, args, launch selection).
    const bool turbo4 = k_cache->type == GGML_TYPE_TURBO4_FP8_BS256 && r4d_turbo4_enabled();
    if (k_cache->type != GGML_TYPE_R4D_FP8_KV && !turbo4) {
        r4d_log_reject_once("k_cache_type",
                             "k_cache->type=%d (need GGML_TYPE_R4D_FP8_KV=%d, or GGML_TYPE_TURBO4_FP8_BS256=%d "
                             "with MAD_USE_R4D_ATTN_TURBO4=1)",
                             (int) k_cache->type, (int) GGML_TYPE_R4D_FP8_KV, (int) GGML_TYPE_TURBO4_FP8_BS256);
        return false;
    }
    if (head_dim != 256 || block_size != 16) {
        r4d_log_reject_once("head_dim_block_size", "head_dim=%d block_size=%d (need 256/16)",
                             head_dim, block_size);
        return false;
    }
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0 || n_heads / n_kv_heads != 6) {
        r4d_log_reject_once("gqa_ratio", "n_heads=%d n_kv_heads=%d (need n_heads/n_kv_heads==6)",
                             n_heads, (int) n_kv_heads);
        return false;
    }
    // num_active: same clamp-to-static-n_seq_max policy as the AITER path's num_seqs_dispatch
    // (mt_pagedattn_aiter.cu:1056-1058) — op_params[6]==0 means unset, falls back to the static
    // count (safe: it can only ever UNDER-count when unset, since the fallback assumes every slot
    // might be active). Computed before the op_params[4] check below because the op_params[4]==0
    // fallback needs it.
    const int num_active = n_seqs_active_param > 0
        ? (n_seqs_active_param < num_seqs ? (int) n_seqs_active_param : num_seqs)
        : num_seqs;
    if (num_active <= 0) {
        r4d_log_reject_once("num_active_zero", "num_active=%d num_seqs(static n_seq_max)=%d",
                             num_active, num_seqs);
        return false;
    }

    bool use_per_seq = false;
    int  q_len       = 0;
    if (flags != 0) {
        // New-style op_params (llm_graph_input_attn_kv::update_paged_attn_q_lens()) — uniform-vs-
        // per-seq is decided from flags bit0 ALONE, never from op_params[4]'s value: op_params[4]
        // is non-zero in both cases now (the shared q_len when uniform, else the ubatch's total
        // token count as a safe upper bound — kept that way so mt_pagedattn.cu's tile/decode
        // gates, which read op_params[4] as that upper bound, see no behavior change), so it can
        // no longer distinguish the two cases by itself.
        if (flags & 1) {
            // bit0: every active seq shares one q_len — op_params[4] holds it directly (exact,
            // not just an upper bound) when this bit is set, re-pushed per-ubatch (and
            // per-device-clone) by update_paged_attn_q_lens(), so it's correct under
            // tensor-split-attn too.
            q_len = (int) max_q_len_param;
            if (q_len <= 0) {
                r4d_log_reject_once("uniform_q_len_nonpositive",
                                     "flags=0x%x op_params[4]=%d", (unsigned) flags, (int) max_q_len_param);
                return false;
            }
        } else if (flags & 4) {
            // bit2: n_seq_max > 8, so op_params[8..15] couldn't hold one slot each, and q_lens
            // are not uniform (bit0 unset) — no host-visible single q_len, and the per-seq
            // values needed for per-seq dispatch (below) weren't packed either. Decline cleanly.
            r4d_log_reject_once("too_many_slots_nonuniform",
                                 "n_seq_max=%d (static) > 8 and q_lens not uniform; per-seq q_lens "
                                 "not packed into op_params[8..15]", num_seqs);
            return false;
        } else {
            // Non-uniform, n_seq_max <= 8: op_params[8..15] (per_seq_q_len[], read above) carries
            // every active seq's own q_len. Dispatch R4D once per active sequence instead of once
            // for the whole batch — see the "Per-seq dispatch" section below.
            use_per_seq = true;
        }
    } else {
        // flags==0: old graph / cold clone — op_params[7] never got a value (materialized before
        // this integration existed, or executed before its first set_input; same "0 means unset"
        // convention as op_params[4]/[5]/[6]). Fall back to the pre-existing host-side derivation
        // from op_params[4]/[6]/q->ne[2] (no device readback), kept for graphs built before this
        // fix — see the (now historical) argument in the header comment.
        if (max_q_len_param > 0) {
            q_len = (int) max_q_len_param;
            if ((long) q_len * (long) num_active != (long) total_q_tokens) {
                r4d_log_reject_once("q_len_product_mismatch",
                                     "op_params[4]=%d num_active=%d total_q_tokens=%d (product != total)",
                                     (int) max_q_len_param, num_active, total_q_tokens);
                return false;
            }
        } else {
            if (n_seqs_active_param > 0 && num_active > 0 && total_q_tokens % num_active == 0) {
                q_len = total_q_tokens / num_active;
            } else if (n_seqs_active_param <= 0 && num_seqs == 1) {
                // op_params[6] is ALSO unset (0 means unset, same convention) AND the cache is
                // single-slot (block_tables->ne[1]==1, so num_active can only ever be 1 here) —
                // every query token in this call belongs to that one slot.
                q_len = total_q_tokens;
            } else {
                r4d_log_reject_once("q_len_unset_undecidable",
                    "op_params[4]=0 op_params[6]=%d num_active=%d total_q_tokens=%d num_seqs(static)=%d "
                    "(cannot derive a shared q_len without a device readback)",
                    (int) n_seqs_active_param, num_active, total_q_tokens, num_seqs);
                return false;
            }
            if (q_len <= 0) {
                r4d_log_reject_once("q_len_derived_nonpositive",
                    "derived q_len=%d num_active=%d total_q_tokens=%d", q_len, num_active, total_q_tokens);
                return false;
            }
        }
    }
    // Decode-vs-prefill selection (q_len*6 <= 64) happens below, after commit; both shapes are
    // otherwise eligible here. In per-seq mode the same selection is made independently per
    // active sequence, using its own q_len.

    // ── Capture safety (MAD-406 warm-up) ───────────────────────────────────────────────────────
    // r4d's prefill launcher memoizes a getenv() in a function-local static on its first call
    // (see the task contract this file was written against). A first call that happens to land
    // inside HIP graph capture would bake a possibly-wrong memoized value into every future
    // replay. So: the very first call into this function, ever, must be eager (not under
    // capture) — once that eager warm-up call has succeeded, every later call (captured or not)
    // is fine, because the memo is already resolved.
    static std::atomic<bool> g_r4d_warmed_up{false};
    cudaStream_t stream = ctx.stream();
    if (!g_r4d_warmed_up.load(std::memory_order_acquire)) {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        const bool is_capturing = cudaStreamIsCapturing(stream, &cap) == cudaSuccess
                                   && cap != cudaStreamCaptureStatusNone;
        if (is_capturing) {
            // Defer: let this graph fall back to the existing paths. A later eager call (e.g. the
            // ggml-cuda.cu "eager warm-up visit" pattern already used for PAGED_ATTN_MT, or simply
            // this op's first ever non-captured invocation) will warm R4D up for good.
            r4d_log_reject_once("capture_warmup_deferral",
                                 "first-ever call landed inside HIP graph capture; deferring warm-up");
            return false;
        }
        // Cheap, host-only, no allocation/sync — safe to call before the real first launch either
        // way; also serves as an explicit touch of the R4D ABI before committing to it.
        int dims_head, dims_gqa, dims_bs, dims_max_rows;
        r4d_attn_dims(&dims_head, &dims_gqa, &dims_bs, &dims_max_rows);
        GGML_ASSERT(dims_head == 256 && dims_gqa == 6 && dims_bs == 16 &&
                    "R4D library geometry does not match this integration's assumptions");
    }

    // ── Past this point we COMMIT: no more `return false`. Once the scatter below runs, the cache
    // is in R4D's fp8 layout and the other paths' kernels can no longer read it. ─────────────────
    const int dev = ctx.device;
    const int n_tokens = (int) k_cur->ne[2];

    // turbo4kv mode: the layer index the centroid-LUT registry keys on, parsed from the k_cache
    // tensor's name (llama_kv_cache names it "cache_k_l<N>") — same parser the AITER path uses,
    // exported non-static for this purpose (mt_pagedattn_aiter.cuh). Not meaningful/used otherwise.
    const int il = turbo4 ? mt_aiter_parse_layer_from_kv_cache_name(k_cache->name) : -1;
    GGML_ASSERT((!turbo4 || il >= 0) &&
                "mt_pagedattn_r4d: turbo4kv mode failed to parse layer index from k_cache tensor name");

    // Debug-only overread guard, mirroring the aiter path's check (mt_pagedattn_aiter.cu:1093-1131):
    // everything below reads exactly total_q_tokens*n_heads*head_dim F16 elements out of q.
    GGML_ASSERT((size_t) total_q_tokens * (size_t) n_heads * (size_t) head_dim * sizeof(__half)
                    <= ggml_nbytes(q) &&
                "mt_pagedattn_r4d: q tensor too small for (total_q_tokens, n_heads, head_dim)");

    // ── 1. Fused scatter: K_cur/V_cur (F16) -> the cache's paged fp8 layout, via slot_mapping.
    //       Must run before the attention call below (same ordering as the AITER path: scatter,
    //       then attend against the just-written cache). turbo4kv reuses the AITER path's scatter
    //       kernel (mt_pagedattn_aiter.cu) so both adapters write the identical on-disk layout;
    //       the R4D_FP8_KV path keeps its own combined-K|V scatter (mt_pagedattn_r4d_scatter.cuh).
    if (turbo4) {
        mt_aiter_scatter_kv_turbo4_fp8_launch(
            k_cache->data, v_cache->data,
            (const __half *) k_cur->data, (const __half *) v_cur->data,
            (const int32_t *) slot_mapping->data,
            n_tokens, n_kv_heads, head_dim, block_size, il, stream);
    } else {
        mt_r4d_scatter_kv(
            (const half *) k_cur->data, (const half *) v_cur->data,
            (uint8_t *) k_cache->data,
            (const int32_t *) slot_mapping->data,
            (const int32_t *) q_lens->data,
            num_seqs, n_tokens, n_kv_heads, head_dim, block_size, stream);
    }

    // ── 2. cu_seqlens (packed-row prefix sum) + seqused_k (context_lens gated by liveness) ──────
    int32_t * cu_seqlens_ptr = r4d_persist_get<int32_t>(dev, stream, R4D_PERSIST_CU_SEQLENS, (size_t) num_seqs + 1);
    int32_t * seqused_k_ptr  = r4d_persist_get<int32_t>(dev, stream, R4D_PERSIST_SEQUSED_K,  (size_t) num_seqs);
    r4d_build_cu_seqlens_and_seqused_kernel<<<1, 1, 0, stream>>>(
        (const int32_t *) q_lens->data, (const int32_t *) context_lens->data,
        cu_seqlens_ptr, seqused_k_ptr, num_seqs);

    const int row_elems = n_heads * head_dim;
    const int max_ctx   = max_ctx_len_param > 0 ? (int) max_ctx_len_param : (int) max_bps * block_size;

    // ── turbo4kv + Hadamard: K was FWHT-rotated at scatter time by mt_aiter_scatter_kv_turbo4_fp8_
    // launch above (whenever the LUT registry says hadamard_required()) — (QH)·(KH)^T = QK^T only
    // holds if Q is rotated identically before attention. Mirrors the AITER path's Q pre-rotation
    // exactly (mt_pagedattn_aiter.cu:~1583-1618): copy the WHOLE packed Q buffer into scratch, FWHT
    // it in place, then feed that copy (not q->data) to whichever launch body runs below — both the
    // uniform and per-seq bodies read out of this same packed buffer, just at different offsets, so
    // one rotation upfront covers both. V is never rotated.
    const __half * q_data = (const __half *) q->data;
    if (turbo4 && mt_turbo_fp8::hadamard_required()) {
        const size_t q_elts = (size_t) total_q_tokens * (size_t) n_heads * (size_t) head_dim;
        __half * q_rot = r4d_persist_get<__half>(dev, stream, R4D_PERSIST_Q_ROT, q_elts);
        CUDA_CHECK(cudaMemcpyAsync(q_rot, q->data, q_elts * sizeof(__half), cudaMemcpyDeviceToDevice, stream));
        const cudaError_t herr = mt_turbo_fp8_fwht_half(stream, q_rot, total_q_tokens * n_heads, head_dim, head_dim);
        if (herr != cudaSuccess) {
            GGML_ABORT("mt_pagedattn_r4d: mt_turbo_fp8_fwht_half(Q) launch failed: %s", cudaGetErrorString(herr));
        }
        q_data = q_rot;
    }

    if (!use_per_seq) {
        // ── MAD_R4D_ATTN_CALLFIX gate: compact grid.z from the static n_seq_max down to the
        // actual live count — see r4d_callfix_enabled()'s comment above for why this exists and
        // why it is a remap rather than a plain narrowing. A no-op (r4d_num_seqs == num_seqs)
        // whenever the cache's static n_seq_max already equals num_active.
        const bool do_compact = r4d_callfix_enabled() && num_active < num_seqs;
        const int  r4d_num_seqs = do_compact ? num_active : num_seqs;
        if (do_compact) {
            r4d_log_callfix_once(num_seqs, num_active);
        }

        int32_t *      seq_map_ptr             = nullptr;
        const int32_t * seqused_active_ptr     = seqused_k_ptr;
        const int *     block_table_active_ptr = (const int *) block_tables->data;
        if (do_compact) {
            seq_map_ptr = r4d_persist_get<int32_t>(dev, stream, R4D_PERSIST_SEQ_MAP, (size_t) num_active);
            int32_t * seqused_compact_ptr = r4d_persist_get<int32_t>(dev, stream, R4D_PERSIST_SEQUSED_COMPACT, (size_t) num_active);
            int32_t * block_table_compact_ptr = r4d_persist_get<int32_t>(
                dev, stream, R4D_PERSIST_BLOCK_TABLE_COMPACT, (size_t) num_active * (size_t) max_bps);

            r4d_build_compact_seqmap_kernel<<<1, 1, 0, stream>>>(
                (const int32_t *) q_lens->data, (const int32_t *) context_lens->data,
                seq_map_ptr, seqused_compact_ptr, num_seqs);
            {
                const int threads = std::min(256, max_bps);
                r4d_gather_block_table_kernel<<<num_active, threads, 0, stream>>>(
                    (const int32_t *) block_tables->data, seq_map_ptr, block_table_compact_ptr, max_bps);
            }

            seqused_active_ptr     = seqused_compact_ptr;
            block_table_active_ptr = (const int *) block_table_compact_ptr;
        }

        // ── 3. Expand packed F16 Q -> slot-indexed bf16 Q ─────────────────────────────────────
        const size_t   slot_rows = (size_t) r4d_num_seqs * (size_t) q_len;
        nv_bfloat16 *  q_bf16    = r4d_persist_get<nv_bfloat16>(dev, stream, R4D_PERSIST_Q_BF16, slot_rows * (size_t) row_elems);
        nv_bfloat16 *  out_bf16  = r4d_persist_get<nv_bfloat16>(dev, stream, R4D_PERSIST_OUT_BF16, slot_rows * (size_t) row_elems);
        {
            const dim3 grid((unsigned) r4d_num_seqs, (unsigned) q_len);
            const int  threads = std::min(256, row_elems);
            if (do_compact) {
                r4d_expand_q_compact_kernel<<<grid, threads, 0, stream>>>(
                    q_data, q_bf16,
                    (const int32_t *) q_lens->data, cu_seqlens_ptr, seq_map_ptr, q_len, row_elems);
            } else {
                r4d_expand_q_kernel<<<grid, threads, 0, stream>>>(
                    q_data, q_bf16,
                    (const int32_t *) q_lens->data, cu_seqlens_ptr, q_len, row_elems);
            }
        }

        // ── 4. Fill R4DArgs ────────────────────────────────────────────────────────────────────
        R4DArgs args{};
        args.q             = q_bf16;
        args.kv            = k_cache->data;
        args.block_table   = block_table_active_ptr;
        args.seqused_k     = seqused_active_ptr;
        args.out           = out_bf16;
        args.k_descale     = nullptr;  // NULL => 1.0 (r4d.h)
        args.v_descale     = nullptr;
        args.q_descale     = nullptr;  // unused: query is bf16
        args.scratch       = nullptr;  // filled below for decode
        args.num_seqs      = r4d_num_seqs;
        args.q_len         = q_len;
        args.q_heads       = n_heads;
        args.kv_heads      = n_kv_heads;
        args.head_dim      = head_dim;
        args.block_size    = block_size;
        args.max_blocks    = max_bps;
        if (turbo4) {
            // Separate K/V caches, 162-byte records per (block, slot, kv head) — see this file's
            // header comment and mt_scatter_kv_turbo4_fp8_aiter_kernel (mt_pagedattn_aiter.cu) for
            // the exact layout. Strides are in BYTES for this mode (r4d.h).
            args.kv              = k_cache->data;
            args.v_cache         = v_cache->data;
            args.k_lut           = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_K);
            args.v_lut           = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_V);
            GGML_ASSERT(args.k_lut && args.v_lut &&
                        "mt_pagedattn_r4d: turbo4kv centroid LUT lookup returned null");
            args.kv_block_stride = 16L * (long) n_kv_heads * 162L;
            args.kv_slot_stride  = (long) n_kv_heads * 162L;
            args.kv_head_stride  = 162L;
        } else {
            // kv layout: (num_blocks, kv_heads, block_size, 2*head_dim), fp8 e4m3, K then V per
            // slot — strides in ELEMENTS (r4d.h). block_size/head_dim already gated to 16/256.
            args.kv_block_stride = (long) n_kv_heads * (long) block_size * (long) (2 * head_dim);
            args.kv_head_stride  = (long) block_size * (long) (2 * head_dim);
        }
        args.scale         = scale;
        args.splits        = 0;  // let R4D's split law choose
        args.max_ctx       = max_ctx;

        const bool is_decode = (q_len * 6) <= 64;
        long scratch_bytes = 0;
        if (is_decode) {
            scratch_bytes = r4d_attn_decode_h256_gqa6_scratch_bytes(&args);
            if (scratch_bytes > 0) {
                args.scratch = r4d_persist_get<uint8_t>(dev, stream, R4D_PERSIST_DECODE_SCRATCH, (size_t) scratch_bytes);
            }
        }

        // ── 5. Launch ──────────────────────────────────────────────────────────────────────────
        const char * kernel_name = turbo4
            ? (is_decode ? "r4d_attn_decode_h256_gqa6_turbo4kv" : "r4d_attn_prefill_h256_gqa6_turbo4kv")
            : (is_decode ? "r4d_attn_decode_h256_gqa6_fp8kv"    : "r4d_attn_prefill_h256_gqa6_fp8kv");
        const int rc = turbo4
            ? (is_decode ? r4d_attn_decode_h256_gqa6_turbo4kv(&args, stream)
                         : r4d_attn_prefill_h256_gqa6_turbo4kv(&args, stream))
            : (is_decode ? r4d_attn_decode_h256_gqa6_fp8kv(&args, stream)
                         : r4d_attn_prefill_h256_gqa6_fp8kv(&args, stream));

        r4d_log_once(q_len, r4d_num_seqs, args.max_ctx, is_decode, args.splits, scratch_bytes, rc, kernel_name);

        if (rc != 0) {
            // No fallback here: the scatter above has already committed the cache to R4D's fp8
            // layout, so the other paths can no longer read it correctly. A geometry rejection
            // this late means the eligibility gate above let through a shape R4D itself refuses
            // (a bug in that gate, not a runtime condition to route around) — abort loudly rather
            // than produce silently-wrong attention output.
            GGML_ABORT("mt_pagedattn_r4d: %s launch rejected shape (rc=%d, q_len=%d num_seqs=%d "
                       "n_heads=%d n_kv_heads=%d max_ctx=%d max_blocks=%d)",
                       kernel_name, rc, q_len, r4d_num_seqs, n_heads, n_kv_heads, args.max_ctx, max_bps);
        }

        g_r4d_warmed_up.store(true, std::memory_order_release);

        // ── 6. Compact slot-indexed bf16 output -> packed F16 dst ─────────────────────────────
        {
            const dim3 grid((unsigned) r4d_num_seqs, (unsigned) q_len);
            const int  threads = std::min(256, row_elems);
            if (do_compact) {
                r4d_compact_out_compact_kernel<<<grid, threads, 0, stream>>>(
                    out_bf16, (__half *) dst->data,
                    (const int32_t *) q_lens->data, cu_seqlens_ptr, seq_map_ptr, q_len, row_elems);
            } else {
                r4d_compact_out_kernel<<<grid, threads, 0, stream>>>(
                    out_bf16, (__half *) dst->data,
                    (const int32_t *) q_lens->data, cu_seqlens_ptr, q_len, row_elems);
            }
        }
    } else {
        // ── Per-seq dispatch ───────────────────────────────────────────────────────────────────
        // R4D's kernels only accept one shared q_len per launch (r4d.h: "All seqs share ONE
        // q_len"), so when the active seqs' q_lens differ we run libr4d once per active sequence
        // instead of once for the whole batch. Each launch uses num_seqs=1: with only one live
        // "slot", R4D's slot-indexed Q/out layout and this op's packed (active-seqs-only) layout
        // coincide exactly, so no expand/compact skip-inactive-slot logic is needed — just a
        // straight elementwise cast over that sequence's own rows. Row offsets come from a
        // host-side prefix sum over the per-seq q_lens already sitting in op_params[8..15]
        // (per_seq_q_len[]) — no device readback, mirroring how the uniform path derives q_len.
        // cu_seqlens_ptr (built above) is unused here; seqused_k_ptr is reused by indexing +s,
        // since it already holds one context length per cache slot.
        int32_t row_off[9] = {0};
        for (int s = 0; s < num_seqs; ++s) {
            const int32_t ql = per_seq_q_len[s] > 0 ? per_seq_q_len[s] : 0;
            row_off[s + 1] = row_off[s] + ql;
        }
        GGML_ASSERT(row_off[num_seqs] == total_q_tokens &&
                    "mt_pagedattn_r4d: per-seq q_lens (op_params[8..15]) don't sum to q->ne[2]");

        int n_launched = 0;
        for (int s = 0; s < num_seqs; ++s) {
            const int32_t q_len_s = per_seq_q_len[s];
            if (q_len_s <= 0) {
                continue;  // slot not live this call
            }
            ++n_launched;

            const long     row_base = row_off[s];
            const __half * q_src    = q_data                     + row_base * (long) row_elems;
            __half *       out_dst  = (__half *)       dst->data + row_base * (long) row_elems;

            const size_t  rows     = (size_t) q_len_s;
            nv_bfloat16 * q_bf16   = r4d_persist_get<nv_bfloat16>(dev, stream, R4D_PERSIST_Q_BF16,   rows * (size_t) row_elems);
            nv_bfloat16 * out_bf16 = r4d_persist_get<nv_bfloat16>(dev, stream, R4D_PERSIST_OUT_BF16, rows * (size_t) row_elems);

            {
                const int threads = std::min(256, row_elems);
                r4d_cast_f16_to_bf16_kernel<<<(unsigned) q_len_s, threads, 0, stream>>>(
                    q_src, q_bf16, row_elems, q_len_s);
            }

            R4DArgs args_s{};
            args_s.q             = q_bf16;
            args_s.kv            = k_cache->data;
            args_s.block_table   = (const int *) block_tables->data + (size_t) s * (size_t) max_bps;
            args_s.seqused_k     = seqused_k_ptr + s;
            args_s.out           = out_bf16;
            args_s.k_descale     = nullptr;
            args_s.v_descale     = nullptr;
            args_s.q_descale     = nullptr;
            args_s.scratch       = nullptr;
            args_s.num_seqs      = 1;
            args_s.q_len         = q_len_s;
            args_s.q_heads       = n_heads;
            args_s.kv_heads      = n_kv_heads;
            args_s.head_dim      = head_dim;
            args_s.block_size    = block_size;
            args_s.max_blocks    = max_bps;
            if (turbo4) {
                args_s.kv              = k_cache->data;
                args_s.v_cache         = v_cache->data;
                args_s.k_lut           = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_K);
                args_s.v_lut           = mt_turbo_fp8::get_lut_device_ptr(il, mt_turbo_fp8::KV_V);
                GGML_ASSERT(args_s.k_lut && args_s.v_lut &&
                            "mt_pagedattn_r4d: turbo4kv centroid LUT lookup returned null");
                args_s.kv_block_stride = 16L * (long) n_kv_heads * 162L;
                args_s.kv_slot_stride  = (long) n_kv_heads * 162L;
                args_s.kv_head_stride  = 162L;
            } else {
                args_s.kv_block_stride = (long) n_kv_heads * (long) block_size * (long) (2 * head_dim);
                args_s.kv_head_stride  = (long) block_size * (long) (2 * head_dim);
            }
            args_s.scale         = scale;
            args_s.splits        = 0;
            args_s.max_ctx       = max_ctx;

            const bool is_decode_s = (q_len_s * 6) <= 64;
            long scratch_bytes_s = 0;
            if (is_decode_s) {
                scratch_bytes_s = r4d_attn_decode_h256_gqa6_scratch_bytes(&args_s);
                if (scratch_bytes_s > 0) {
                    args_s.scratch = r4d_persist_get<uint8_t>(dev, stream, R4D_PERSIST_DECODE_SCRATCH, (size_t) scratch_bytes_s);
                }
            }

            const char * kernel_name_s = turbo4
                ? (is_decode_s ? "r4d_attn_decode_h256_gqa6_turbo4kv" : "r4d_attn_prefill_h256_gqa6_turbo4kv")
                : (is_decode_s ? "r4d_attn_decode_h256_gqa6_fp8kv"    : "r4d_attn_prefill_h256_gqa6_fp8kv");
            const int rc_s = turbo4
                ? (is_decode_s ? r4d_attn_decode_h256_gqa6_turbo4kv(&args_s, stream)
                               : r4d_attn_prefill_h256_gqa6_turbo4kv(&args_s, stream))
                : (is_decode_s ? r4d_attn_decode_h256_gqa6_fp8kv(&args_s, stream)
                               : r4d_attn_prefill_h256_gqa6_fp8kv(&args_s, stream));

            if (rc_s != 0) {
                // Same reasoning as the uniform path's abort: the scatter has already committed
                // the cache to R4D's fp8 layout, so there's no falling back partway through.
                GGML_ABORT("mt_pagedattn_r4d: %s launch rejected shape (per-seq slot=%d, rc=%d, "
                           "q_len=%d n_heads=%d n_kv_heads=%d max_ctx=%d max_blocks=%d)",
                           kernel_name_s, s, rc_s, q_len_s, n_heads, n_kv_heads, args_s.max_ctx, max_bps);
            }

            {
                const int threads = std::min(256, row_elems);
                r4d_cast_bf16_to_f16_kernel<<<(unsigned) q_len_s, threads, 0, stream>>>(
                    out_bf16, out_dst, row_elems, q_len_s);
            }
        }

        g_r4d_warmed_up.store(true, std::memory_order_release);
        r4d_log_once_per_seq(num_seqs, n_launched, max_ctx);
    }

    return true;
}

}  // namespace mt

#endif  // GGML_HIP_R4D
