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

#include "common.cuh"
#include "mt_pagedattn_r4d.cuh"

#ifdef GGML_HIP_R4D

#include "r4d/ggml-r4d.h"
#include "mt_pagedattn_r4d_scatter.cuh"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
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
    R4D_PERSIST_COUNT
};
static std::mutex g_r4d_persist_mutex;
static std::map<std::pair<int, cudaStream_t>, std::array<r4d_persist_buf, R4D_PERSIST_COUNT>> g_r4d_persist;

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

// ─────────────────────────────────────────────────────────────────────────
// Diagnostics — MAD_R4D_LOG=1: log once per distinct (q_len, num_seqs, max_ctx) the path taken.
bool r4d_log_enabled() {
    static const bool e = [] { const char * v = std::getenv("MAD_R4D_LOG"); return v && v[0] == '1'; }();
    return e;
}
void r4d_log_once(int q_len, int num_seqs, int max_ctx, bool decode, int splits, long scratch_bytes, int rc) {
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
        "[mt_pagedattn_r4d] q_len=%d num_seqs=%d max_ctx=%d path=%s splits=%d scratch_bytes=%ld rc=%d\n",
        q_len, num_seqs, max_ctx, decode ? "decode" : "prefill", splits, scratch_bytes, rc);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────
// Dispatch entry
// ─────────────────────────────────────────────────────────────────────────
bool ggml_cuda_op_paged_attn_mt_r4d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_cache       = dst->src[1];
    // v_cache (dst->src[2]) is allocated but unused for GGML_TYPE_R4D_FP8_KV — k_cache holds the
    // combined K|V tensor (see this file's header comment and mt_pagedattn_r4d.cuh).
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

    const int head_dim  = (int) q->ne[0];
    const int n_heads    = (int) q->ne[1];
    const int total_q_tokens = (int) q->ne[2];
    // block_tables->ne[1] is the paged cache's STATIC n_seq_max (see the identical comment in
    // mt_pagedattn_aiter.cu:1021-1034) — the correct bound for R4D's num_seqs, which indexes
    // block_table/seqused_k/the expanded Q buffer by SLOT, not by a compacted live-seq count.
    const int num_seqs   = (int) block_tables->ne[1];

    // ── Eligibility (return false => caller falls through to the existing paths) ──────────────
    if (k_cache->type != GGML_TYPE_R4D_FP8_KV) {
        return false;
    }
    if (head_dim != 256 || block_size != 16) {
        return false;
    }
    if (n_kv_heads <= 0 || n_heads % n_kv_heads != 0 || n_heads / n_kv_heads != 6) {
        return false;
    }
    // max_q_len_param==0 means "unset" (a cold graph executed before its first set_input, or a
    // ubatch that has not gone through set_input at all yet) — same convention as op_params[5]/[6]
    // elsewhere in this op. Nothing safe to conclude about q_len without it.
    if (max_q_len_param <= 0) {
        return false;
    }
    // num_active: same clamp-to-static-n_seq_max policy as the AITER path's num_seqs_dispatch
    // (mt_pagedattn_aiter.cu:1056-1058) — op_params[6]==0 means unset, falls back to the static
    // count (safe: it can only ever UNDER-count when unset, since the fallback assumes every slot
    // might be active).
    const int num_active = n_seqs_active_param > 0
        ? (n_seqs_active_param < num_seqs ? (int) n_seqs_active_param : num_seqs)
        : num_seqs;
    if (num_active <= 0) {
        return false;
    }
    const int q_len = (int) max_q_len_param;
    // The uniform-q_len check this file's header comment derives: op_params[4] can only be read
    // as "every active seq's q_len" when this equality holds. See that comment for the full
    // argument (src/llama-graph.cpp:3565, :649).
    if ((long) q_len * (long) num_active != (long) total_q_tokens) {
        return false;
    }
    // Decode-vs-prefill selection (q_len*6 <= 64) happens below, after commit; both shapes are
    // otherwise eligible here.

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

    // Debug-only overread guard, mirroring the aiter path's check (mt_pagedattn_aiter.cu:1093-1131):
    // everything below reads exactly total_q_tokens*n_heads*head_dim F16 elements out of q.
    GGML_ASSERT((size_t) total_q_tokens * (size_t) n_heads * (size_t) head_dim * sizeof(__half)
                    <= ggml_nbytes(q) &&
                "mt_pagedattn_r4d: q tensor too small for (total_q_tokens, n_heads, head_dim)");

    // ── 1. Fused scatter: K_cur/V_cur (F16) -> R4D's fp8 (num_blocks, kv_heads, 16, 2*head_dim)
    //       layout, via slot_mapping. Must run before the attention call below (same ordering as
    //       the AITER path: scatter, then attend against the just-written cache).
    mt_r4d_scatter_kv(
        (const half *) k_cur->data, (const half *) v_cur->data,
        (uint8_t *) k_cache->data,
        (const int32_t *) slot_mapping->data,
        (const int32_t *) q_lens->data,
        num_seqs, n_tokens, n_kv_heads, head_dim, block_size, stream);

    // ── 2. cu_seqlens (packed-row prefix sum) + seqused_k (context_lens gated by liveness) ──────
    int32_t * cu_seqlens_ptr = r4d_persist_get<int32_t>(dev, stream, R4D_PERSIST_CU_SEQLENS, (size_t) num_seqs + 1);
    int32_t * seqused_k_ptr  = r4d_persist_get<int32_t>(dev, stream, R4D_PERSIST_SEQUSED_K,  (size_t) num_seqs);
    r4d_build_cu_seqlens_and_seqused_kernel<<<1, 1, 0, stream>>>(
        (const int32_t *) q_lens->data, (const int32_t *) context_lens->data,
        cu_seqlens_ptr, seqused_k_ptr, num_seqs);

    // ── 3. Expand packed F16 Q -> slot-indexed bf16 Q ─────────────────────────────────────────
    const int      row_elems = n_heads * head_dim;
    const size_t   slot_rows = (size_t) num_seqs * (size_t) q_len;
    nv_bfloat16 *  q_bf16    = r4d_persist_get<nv_bfloat16>(dev, stream, R4D_PERSIST_Q_BF16, slot_rows * (size_t) row_elems);
    nv_bfloat16 *  out_bf16  = r4d_persist_get<nv_bfloat16>(dev, stream, R4D_PERSIST_OUT_BF16, slot_rows * (size_t) row_elems);
    {
        const dim3 grid((unsigned) num_seqs, (unsigned) q_len);
        const int  threads = std::min(256, row_elems);
        r4d_expand_q_kernel<<<grid, threads, 0, stream>>>(
            (const __half *) q->data, q_bf16,
            (const int32_t *) q_lens->data, cu_seqlens_ptr, q_len, row_elems);
    }

    // ── 4. Fill R4DArgs ────────────────────────────────────────────────────────────────────────
    R4DArgs args{};
    args.q             = q_bf16;
    args.kv            = k_cache->data;
    args.block_table   = (const int *) block_tables->data;
    args.seqused_k     = seqused_k_ptr;
    args.out           = out_bf16;
    args.k_descale     = nullptr;  // NULL => 1.0 (r4d.h)
    args.v_descale     = nullptr;
    args.q_descale     = nullptr;  // unused: query is bf16
    args.scratch       = nullptr;  // filled below for decode
    args.num_seqs      = num_seqs;
    args.q_len         = q_len;
    args.q_heads       = n_heads;
    args.kv_heads      = n_kv_heads;
    args.head_dim      = head_dim;
    args.block_size    = block_size;
    args.max_blocks    = max_bps;
    // kv layout: (num_blocks, kv_heads, block_size, 2*head_dim), fp8 e4m3, K then V per slot —
    // strides in ELEMENTS (r4d.h). block_size and head_dim are already gated to 16/256 above.
    args.kv_block_stride = (long) n_kv_heads * (long) block_size * (long) (2 * head_dim);
    args.kv_head_stride  = (long) block_size * (long) (2 * head_dim);
    args.scale         = scale;
    args.splits        = 0;  // let R4D's split law choose
    // max_ctx: prefer the host-known per-ubatch bound (op_params[5], MAD-378); 0 (unset — a cold
    // graph before its first set_input) falls back to the cache's full allocated-capacity bound,
    // same convention the AITER path uses (mt_pagedattn_aiter.cu:1412-1413).
    args.max_ctx       = max_ctx_len_param > 0 ? (int) max_ctx_len_param : (int) max_bps * block_size;

    const bool is_decode = (q_len * 6) <= 64;
    long scratch_bytes = 0;
    if (is_decode) {
        scratch_bytes = r4d_attn_decode_h256_gqa6_scratch_bytes(&args);
        if (scratch_bytes > 0) {
            args.scratch = r4d_persist_get<uint8_t>(dev, stream, R4D_PERSIST_DECODE_SCRATCH, (size_t) scratch_bytes);
        }
    }

    // ── 5. Launch ──────────────────────────────────────────────────────────────────────────────
    const int rc = is_decode
        ? r4d_attn_decode_h256_gqa6_fp8kv(&args, stream)
        : r4d_attn_prefill_h256_gqa6_fp8kv(&args, stream);

    r4d_log_once(q_len, num_seqs, args.max_ctx, is_decode, args.splits, scratch_bytes, rc);

    if (rc != 0) {
        // No fallback here: the scatter above has already committed the cache to R4D's fp8
        // layout, so the other paths can no longer read it correctly. A geometry rejection this
        // late means the eligibility gate above let through a shape R4D itself refuses (a bug in
        // that gate, not a runtime condition to route around) — abort loudly rather than produce
        // silently-wrong attention output.
        GGML_ABORT("mt_pagedattn_r4d: %s launch rejected shape (rc=%d, q_len=%d num_seqs=%d "
                   "n_heads=%d n_kv_heads=%d max_ctx=%d max_blocks=%d)",
                   is_decode ? "r4d_attn_decode_h256_gqa6_fp8kv" : "r4d_attn_prefill_h256_gqa6_fp8kv",
                   rc, q_len, num_seqs, n_heads, n_kv_heads, args.max_ctx, max_bps);
    }

    g_r4d_warmed_up.store(true, std::memory_order_release);

    // ── 6. Compact slot-indexed bf16 output -> packed F16 dst ─────────────────────────────────
    {
        const dim3 grid((unsigned) num_seqs, (unsigned) q_len);
        const int  threads = std::min(256, row_elems);
        r4d_compact_out_kernel<<<grid, threads, 0, stream>>>(
            out_bf16, (__half *) dst->data,
            (const int32_t *) q_lens->data, cu_seqlens_ptr, q_len, row_elems);
    }

    return true;
}

}  // namespace mt

#endif  // GGML_HIP_R4D
