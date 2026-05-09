// mt_pagedattn — paged attention kernel implementation.
//
// See mt_pagedattn.cuh for layout and threading model docs.
// Design adapted from vLLM (Apache 2.0). Code is independent.

#include "mt_pagedattn.cuh"

#include <cmath>
#include <cstdio>
#include <hip/hip_cooperative_groups.h>

namespace cg = cooperative_groups;

namespace mt {

// MAD-114: scatter→attn cache-flush barrier. The view-aliasing fix
// (scatter's result aliases K cache) gives the scheduler a real RAW edge,
// and __threadfence_system at scatter kernel exit is the strongest GPU
// memory fence available — but neither, alone or together, makes
// scatter's K/V cache writes visible to a same-stream attn kernel that
// follows them on HIP/RDNA (gfx1201). Empirically tested:
//   - Same-stream submission ordering: insufficient
//   - cudaEventRecord + cudaStreamWaitEvent on the same stream: no-op
//   - __threadfence_system inside the kernel: insufficient
// Only host-side cudaStreamSynchronize works, which is illegal in CUDA
// graph capture.
//
// Workaround: insert a cudaMemsetAsync of a single byte to a tiny
// scratch buffer between scatter and attn. This captures as a
// cudaGraphAddMemsetNode in the captured graph, which on RDNA triggers
// a more aggressive hardware-level cache invalidation than a kernel→
// kernel dep alone (see ROCm/hip#3887 thread; the user there resolved
// a near-identical bug after fixing their graph-capture setup, and
// existing AMD docs note the memset-as-barrier pattern). Capture-safe.
//
// thread_local — ggml-backend dispatches sequentially per backend
// thread, so each thread gets its own scratch byte; lazy-init, never
// destroyed (lifetime = process).
static thread_local void * paged_kv_scratch = nullptr;

static void * paged_kv_scratch_get() {
    if (paged_kv_scratch == nullptr) {
        CUDA_CHECK(cudaMalloc(&paged_kv_scratch, 1));
    }
    return paged_kv_scratch;
}

// ───────────────────────── helpers ─────────────────────────

// Warp-level reduce across 32 lanes (HIP wavefront is 64 on some
// GPUs but ggml-cuda's WARP_SIZE is fixed at 32 — works correctly on
// gfx1xxx because __shfl_xor_sync over the active mask).
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        v += __shfl_xor_sync(0xffffffffu, v, mask, WARP_SIZE);
    }
    return v;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_max(T v) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
        v = max(v, __shfl_xor_sync(0xffffffffu, v, mask, WARP_SIZE));
    }
    return v;
}

// Block-level sum: warp reduce, write per-warp partials to smem,
// last warp reduces. One __syncthreads. red_smem must be sized to at
// least NUM_WARPS floats.
template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float v, float * red_smem) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    v = warp_reduce_sum(v);
    if (lane == 0) red_smem[warp] = v;
    __syncthreads();

    float partial = (lane < NUM_WARPS) ? red_smem[lane] : 0.0f;
    if (warp == 0) {
        partial = warp_reduce_sum(partial);
        if (lane == 0) red_smem[0] = partial;
    }
    __syncthreads();
    return red_smem[0];
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_max(float v, float * red_smem) {
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    v = warp_reduce_max(v);
    if (lane == 0) red_smem[warp] = v;
    __syncthreads();

    float partial = (lane < NUM_WARPS) ? red_smem[lane] : -INFINITY;
    if (warp == 0) {
        partial = warp_reduce_max(partial);
        if (lane == 0) red_smem[0] = partial;
    }
    __syncthreads();
    return red_smem[0];
}

// ──────────────────── kernel ────────────────────
//
// Threading: each thread block handles one (head, seq) pair, running
// over all query tokens of that seq's batch slice. NUM_THREADS threads
// cooperate per token using warp shuffles for QK and per-row softmax
// + V@logits accumulation. Designed for HEAD_SIZE divisible by
// NUM_THREADS for the V-accumulator stride; we assert this in dispatch.
//
// Per-query iteration:
//   1. Each thread loads HEAD_SIZE/NUM_THREADS elements of Q into
//      registers (q_reg).
//   2. Walk K blocks in logical order: for each block, for each
//      token-in-block, compute partial Q·K (each thread contributes
//      its slice), block-reduce-sum to get the full QK score, store
//      into smem logits[token_idx]. Track running max during this
//      pass for online-softmax stability.
//   3. After all blocks: subtract max, exp, block-reduce-sum the
//      exp_sum, normalize logits.
//   4. Walk V blocks: for each token-in-block, multiply its logit by
//      this thread's slice of V row, accumulate. Final accumulator
//      is this thread's slice of the output.
//   5. Write output slice to global memory.
//
// Memory access pattern:
//   K cache laid out as [num_blocks, n_kv_heads, HEAD_SIZE/x, BLOCK_SIZE, x]
//   For a given (block, kv_head) and our thread reading element
//   d in [0, HEAD_SIZE), the offset within the block is:
//     (d / x) * BLOCK_SIZE * x  +  token_in_block * x  +  (d % x)
//   With x = 16/sizeof(scalar_t), neighboring threads (different d)
//   read adjacent x-elements — coalesced.
//
//   V cache: [num_blocks, n_kv_heads, HEAD_SIZE, BLOCK_SIZE]
//   For (block, kv_head, head_dim_d), all BLOCK_SIZE tokens are
//   contiguous. Accumulating this thread's d means reading V at
//   (kv_head_idx * HEAD_SIZE + d) * BLOCK_SIZE + token_in_block.
//
// GQA: n_heads can be > n_kv_heads. kv_head = head_idx / (n_heads /
// n_kv_heads). n_heads % n_kv_heads == 0 enforced at dispatch.

template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS,
          int PARTITION_SIZE>
__global__ void mt_paged_attention_kernel(
    scalar_t       * __restrict__ out,
    const scalar_t * __restrict__ q,
    cache_t        * __restrict__ k_cache,    // writable (fused scatter writes here)
    cache_t        * __restrict__ v_cache,    // writable (fused scatter writes here)
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
    const scalar_t * __restrict__ k_cur,      // [head_dim, n_kv_heads, n_tokens]
    const scalar_t * __restrict__ v_cur,      // [head_dim, n_kv_heads, n_tokens]
    const int32_t  * __restrict__ slot_mapping, // [n_tokens]
    int             max_blocks_per_seq,
    int             n_kv_heads,
    int             n_heads,
    float           scale) {
    const int head_idx = blockIdx.x;
    const int seq_idx  = blockIdx.y;
    const int tid      = threadIdx.x;

    constexpr int NUM_WARPS         = NUM_THREADS / WARP_SIZE;
    constexpr int VEC_PER_THREAD    = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int K_X               = 16 / sizeof(cache_t);  // K interleave width
    static_assert(BLOCK_SIZE > 0 && (BLOCK_SIZE & (BLOCK_SIZE - 1)) == 0, "BLOCK_SIZE must be power of 2");
    static_assert(HEAD_SIZE % K_X == 0, "HEAD_SIZE must be divisible by K_X");
    static_assert(NUM_THREADS % WARP_SIZE == 0, "NUM_THREADS must be multiple of WARP_SIZE");

    const int kv_head_idx       = head_idx / (n_heads / n_kv_heads);
    const int q_len             = q_lens[seq_idx];
    const int ctx_len_after_q   = context_lens[seq_idx];   // total tokens in seq's context AFTER this batch's Q is applied
    const int * seq_block_table = block_tables + seq_idx * max_blocks_per_seq;

    // MAD-114 verify-fix: trace EVERY kernel invocation.
    if (head_idx == 0 && seq_idx == 0 && tid == 0) {
        printf("[KDBG] q_len=%d ctx=%d slot[0..3]=%d %d %d %d\n",
               q_len, ctx_len_after_q,
               q_len > 0 ? slot_mapping[0] : -1,
               q_len > 1 ? slot_mapping[1] : -1,
               q_len > 2 ? slot_mapping[2] : -1,
               q_len > 3 ? slot_mapping[3] : -1);
    }


    // ── Phase 1: scatter K_cur/V_cur into the K/V cache ───────────────────
    //
    // MAD-114: fused scatter+attn. Doing the scatter inside this kernel,
    // separated from the attn math by a grid-wide cooperative-groups sync,
    // sidesteps the HIP runtime bug (ROCm/hip#3882, #3887) where same-
    // stream inter-kernel ordering isn't enforced for the scatter→attn
    // pair on RDNA — the bug only matters across kernel boundaries.
    // grid.sync() is an in-kernel hardware barrier across all blocks.
    //
    // To avoid redundant writes (n_heads/n_kv_heads blocks would otherwise
    // race writing identical values to the same slots), only the FIRST
    // head_idx in each kv_head group does the scatter:
    //   head_idx % (n_heads / n_kv_heads) == 0 → scatter
    // The grid.sync() after scatter then makes the writes visible to ALL
    // blocks (including the non-scattering ones) before the attn math.
    {
        const size_t seq_q_offset = 0;  // v1: single-seq batches start at 0
        const int    heads_per_kv   = n_heads / n_kv_heads;
        const bool   is_scatterer   = (head_idx % heads_per_kv) == 0;

        // Only the FIRST head_idx in each kv_head group scatters — avoids
        // redundant writes across n_heads/n_kv_heads blocks. After the
        // grid.sync below, ALL blocks (scatterers + non-scatterers) see
        // the cache values consistently.
        if (is_scatterer)
        for (int t = 0; t < q_len; ++t) {
            const int global_token_idx = (int)(seq_q_offset + t);
            const int slot = slot_mapping[global_token_idx];
            if (slot < 0) continue;  // padding

            const int block_idx     = slot / BLOCK_SIZE;
            const int slot_in_block = slot % BLOCK_SIZE;

            const size_t src_base = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                                  + (size_t) kv_head_idx * HEAD_SIZE;

            #pragma unroll
            for (int v = 0; v < VEC_PER_THREAD; ++v) {
                const int d = tid + v * NUM_THREADS;
                if (d < HEAD_SIZE) {
                    const scalar_t k_val = k_cur[src_base + (size_t) d];
                    const scalar_t v_val = v_cur[src_base + (size_t) d];

                    const int dim_outer = d / K_X;
                    const int dim_inner = d % K_X;
                    const size_t k_idx = (size_t) block_idx * n_kv_heads * HEAD_SIZE * BLOCK_SIZE
                                       + (size_t) kv_head_idx * HEAD_SIZE * BLOCK_SIZE
                                       + (size_t) dim_outer * BLOCK_SIZE * K_X
                                       + (size_t) slot_in_block * K_X
                                       + (size_t) dim_inner;
                    k_cache[k_idx] = k_val;

                    const size_t v_idx = (size_t) block_idx * n_kv_heads * HEAD_SIZE * BLOCK_SIZE
                                       + (size_t) kv_head_idx * HEAD_SIZE * BLOCK_SIZE
                                       + (size_t) d * BLOCK_SIZE
                                       + (size_t) slot_in_block;
                    v_cache[v_idx] = v_val;
                }
            }
        }
        // Grid-wide barrier: all blocks wait here until ALL scatter writes
        // (across all blocks) are globally visible. Requires the kernel to
        // be launched via hipLaunchCooperativeKernel.
        cg::this_grid().sync();
    }
    // ── Phase 2: attention math (uses the just-scattered cache) ───────────

    // Shared memory layout:
    //   [0 .. NUM_WARPS)        — red_smem (reduction scratch)
    //   [NUM_WARPS .. ...)      — logits buffer (per query-token, max ctx)
    extern __shared__ float smem[];
    float * red_smem = smem;
    float * logits   = smem + NUM_WARPS;

    // Per-token iteration.
    for (int qi = 0; qi < q_len; ++qi) {
        // q_pos is the absolute position of this query token in the
        // sequence (0-indexed). Causal mask: only attend to k tokens
        // with kj <= q_pos.
        const int q_pos = (ctx_len_after_q - q_len) + qi;
        const int valid_ctx = q_pos + 1;  // tokens [0, valid_ctx) are visible

        // Load Q slice into registers, scale-applied.
        // Q layout matches ggml's natural [head_dim, n_heads, n_tokens]
        // (head_dim fastest, n_tokens slowest in memory). For seq 0,
        // head H, query token Q, dim D:
        //   offset = (Q * n_heads + H) * HEAD_SIZE + D
        // (Multi-seq batches concat sequentially in the n_tokens axis;
        // qi here is the local-to-this-seq index, so we add the seq's
        // start offset = sum of preceding q_lens. v1 single-seq path
        // has seq_q_offset == 0.)
        scalar_t q_reg[VEC_PER_THREAD];
        const size_t seq_q_offset = 0;  // v1: single-seq batches start at 0
#pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                const size_t q_off = ((seq_q_offset + (size_t) qi) * n_heads + head_idx) * HEAD_SIZE + d;
                q_reg[v] = q[q_off];
            } else {
                q_reg[v] = scalar_t(0);
            }
        }
        GGML_UNUSED(seq_idx);  // multi-seq offset accounting is a follow-up

        // ── Pass 1: QK + running max + write to logits[] ──
        //
        // ALL threads cooperate on ONE token at a time: each thread
        // contributes its slice of d to partial_qk, then block_reduce_sum
        // (which contains __syncthreads) aggregates across the block.
        // This pattern requires every thread to enter the loop the same
        // number of iterations as valid_ctx — divergent threads at the
        // syncthreads is undefined behavior in HIP/CUDA. The previous
        // strided-by-tid loop was wrong for two reasons: (a) different
        // threads were aggregating partial_qks for DIFFERENT tokens
        // (meaningless sum) and (b) when valid_ctx < NUM_THREADS some
        // threads skipped the syncthreads → garbage output.
        float qk_max = -INFINITY;

        for (int token = 0; token < valid_ctx; ++token) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];

            float partial_qk = 0.0f;
            if (physical != kInvalidBlockTableEntry) {
#pragma unroll
                for (int v = 0; v < VEC_PER_THREAD; ++v) {
                    const int d = tid + v * NUM_THREADS;
                    if (d < HEAD_SIZE) {
                        const int xi = d / K_X;
                        const int xj = d % K_X;
                        const size_t k_off = ((size_t) physical * n_kv_heads + kv_head_idx) * (HEAD_SIZE / K_X) * BLOCK_SIZE * K_X
                                           + (size_t) xi * BLOCK_SIZE * K_X
                                           + (size_t) tok_in_block * K_X
                                           + xj;
                        const float k_val = (float) k_cache[k_off];
                        partial_qk += (float) q_reg[v] * k_val;
                    }
                }
            }
            // All threads call block_reduce_sum — meets __syncthreads.
            const float qk = (physical == kInvalidBlockTableEntry)
                ? -INFINITY
                : block_reduce_sum<NUM_WARPS>(partial_qk, red_smem) * scale;

            if (tid == 0) {
                logits[token] = qk;
            }
            qk_max = max(qk_max, qk);
        }

        // ── Block-reduce max across all threads' qk_max ──
        qk_max = block_reduce_max<NUM_WARPS>(qk_max, red_smem);

        // ── Pass 2: exp(qk - max), sum, normalize ──
        float exp_sum = 0.0f;
        for (int token = tid; token < valid_ctx; token += NUM_THREADS) {
            const float e = __expf(logits[token] - qk_max);
            logits[token] = e;
            exp_sum += e;
        }
        exp_sum = block_reduce_sum<NUM_WARPS>(exp_sum, red_smem);
        const float inv_sum = 1.0f / (exp_sum + 1e-6f);

        for (int token = tid; token < valid_ctx; token += NUM_THREADS) {
            logits[token] *= inv_sum;
        }
        __syncthreads();

        // ── Pass 3: V @ logits accumulation ──
        // Each thread accumulates VEC_PER_THREAD output rows.
        float acc[VEC_PER_THREAD];
#pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) acc[v] = 0.0f;

        for (int token = 0; token < valid_ctx; ++token) {
            const int logical_block = token / BLOCK_SIZE;
            const int tok_in_block  = token % BLOCK_SIZE;
            const int physical      = seq_block_table[logical_block];
            if (physical == kInvalidBlockTableEntry) continue;
            const float w = logits[token];

#pragma unroll
            for (int v = 0; v < VEC_PER_THREAD; ++v) {
                const int d = tid + v * NUM_THREADS;
                if (d < HEAD_SIZE) {
                    const size_t v_off = ((size_t) physical * n_kv_heads + kv_head_idx) * HEAD_SIZE * BLOCK_SIZE
                                       + (size_t) d * BLOCK_SIZE
                                       + tok_in_block;
                    acc[v] += w * (float) v_cache[v_off];
                }
            }
        }

        // ── Write output slice ──
        // Output mirrors Q's layout: ggml [head_dim, n_heads, n_tokens]
        // (same shape as q since the constructor copies q->ne).
#pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                const size_t out_off = ((seq_q_offset + (size_t) qi) * n_heads + head_idx) * HEAD_SIZE + d;
                out[out_off] = (scalar_t) acc[v];
            }
        }

        __syncthreads();  // ensure smem reuse safe between qi iterations
    }
}

// ──────────────────── dispatch ────────────────────

// op_params layout (set by graph builder when emitting GGML_OP_PAGED_ATTN_MT):
//   [0]: float scale
//   [1]: int32_t block_size
//   [2]: int32_t max_blocks_per_seq
//   [3]: int32_t n_kv_heads
//
// src tensors:
//   src[0] = Q     [head_size, n_heads, sum(q_lens), 1]   — packed across seqs
//   src[1] = K cache [paged layout, F16, mutated by this op]
//   src[2] = V cache [paged layout, F16, mutated by this op]
//   src[3] = block_tables [max_blocks_per_seq, num_seqs]
//   src[4] = context_lens [num_seqs]
//   src[5] = q_lens       [num_seqs]
//   src[6] = K_cur        [head_dim, n_kv_heads, n_tokens]   F16  ← fused scatter
//   src[7] = V_cur        [head_dim, n_kv_heads, n_tokens]   F16  ← fused scatter
//   src[8] = slot_mapping [n_tokens]                          I32  ← fused scatter
// dst:
//   out [head_size, n_heads, sum(q_lens), 1]
//
// MAD-114: src[6..8] make this op the SINGLE handler for both KV cache
// writes AND attention reads. Doing both phases inside one kernel
// (separated by __syncthreads()) sidesteps the HIP runtime bug
// where same-stream inter-kernel ordering isn't enforced — see the
// kernel header comment. The legacy ggml_paged_kv_update_mt op is no
// longer needed and has been removed.
//
// For the v1 single-batch case sum(q_lens) collapses to q_len * num_seqs
// when all seqs in the batch have the same q_len (typical decode batch).

template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int BLOCK_SIZE>
static void launch_paged_attn(
    scalar_t       * out,
    const scalar_t * q,
    cache_t        * k_cache,
    cache_t        * v_cache,
    const int32_t  * block_tables,
    const int32_t  * context_lens,
    const int32_t  * q_lens,
    const scalar_t * k_cur,
    const scalar_t * v_cur,
    const int32_t  * slot_mapping,
    int             num_seqs,
    int             n_heads,
    int             n_kv_heads,
    int             max_blocks_per_seq,
    int             max_ctx_len,
    float           scale,
    cudaStream_t    stream) {
    constexpr int NUM_THREADS = 128;
    constexpr int NUM_WARPS   = NUM_THREADS / WARP_SIZE;

    dim3 grid(n_heads, num_seqs);
    dim3 block(NUM_THREADS);

    // smem: NUM_WARPS reduction floats + max_ctx_len logits floats.
    const size_t smem_bytes = (NUM_WARPS + max_ctx_len) * sizeof(float);

    // Defensive: catch the LDS-overflow case loudly. AMD GPUs typically
    // expose 64 KiB shared memory per block (SM/CU); NVIDIA newer SMs
    // 100+ KiB. The dispatcher's `cudaErrorInvalidArgument` for
    // oversized smem looks identical to many other failure modes —
    // surface a clear error here so users know to either (a) reduce
    // per-slot context, (b) drop --kv-tier-paged-blocks, or (c) wait
    // on the chunked-attention rewrite (see docs/MAD-NN).
    if (smem_bytes > 65536) {
        GGML_LOG_ERROR("mt::paged_attn: requested smem %zu B exceeds 64 KiB LDS limit "
                       "(max_ctx_len=%d). The current paged kernel doesn't support "
                       "ctx > ~16k per attention call. Reduce -c/--parallel so "
                       "n_ctx_seq * 4 bytes fits, or drop --kv-tier-paged-blocks.\n",
                       smem_bytes, max_ctx_len);
        GGML_ABORT("mt::paged_attn smem overflow — see docs for chunked-attention plan");
    }

    // MAD-114: cooperative launch — the kernel's grid.sync() between
    // scatter and attn phases requires this. Without it, grid.sync()
    // is a no-op (or worse, deadlocks).
    void * args[] = {
        (void *) &out,
        (void *) &q,
        (void *) &k_cache,
        (void *) &v_cache,
        (void *) &block_tables,
        (void *) &context_lens,
        (void *) &q_lens,
        (void *) &k_cur,
        (void *) &v_cur,
        (void *) &slot_mapping,
        (void *) &max_blocks_per_seq,
        (void *) &n_kv_heads,
        (void *) &n_heads,
        (void *) &scale,
    };
    using kernel_t = void (*)(
        scalar_t *, const scalar_t *,
        cache_t *, cache_t *,
        const int32_t *, const int32_t *, const int32_t *,
        const scalar_t *, const scalar_t *, const int32_t *,
        int, int, int, float);
    kernel_t kernel_ptr = &mt_paged_attention_kernel<
        scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, /*PARTITION_SIZE=*/0>;
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (const void *) kernel_ptr,
        grid, block, args, smem_bytes, stream));

}

void ggml_cuda_op_paged_attn_mt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_cache       = dst->src[1];
    const ggml_tensor * v_cache       = dst->src[2];
    const ggml_tensor * block_tables  = dst->src[3];
    const ggml_tensor * context_lens  = dst->src[4];
    const ggml_tensor * q_lens        = dst->src[5];
    const ggml_tensor * k_cur         = dst->src[6];
    const ggml_tensor * v_cur         = dst->src[7];
    const ggml_tensor * slot_mapping  = dst->src[8];

    const float * op_params_f = (const float *)(dst->op_params);
    const float   scale       = op_params_f[0];
    const int32_t block_size  = ((const int32_t *)(op_params_f + 1))[0];
    const int32_t max_bps     = ((const int32_t *)(op_params_f + 2))[0];
    const int32_t n_kv_heads  = ((const int32_t *)(op_params_f + 3))[0];

    const int head_size = q->ne[0];
    const int n_heads   = q->ne[1];
    const int num_seqs  = block_tables->ne[1];

    GGML_ASSERT(n_heads % n_kv_heads == 0 && "n_heads must be divisible by n_kv_heads");
    GGML_ASSERT(q->type == GGML_TYPE_F16 && "PagedAttn v1 supports F16 Q only");
    GGML_ASSERT(k_cache->type == GGML_TYPE_F16 && "PagedAttn v1 supports F16 K cache only");
    GGML_ASSERT(v_cache->type == GGML_TYPE_F16 && "PagedAttn v1 supports F16 V cache only");
    GGML_ASSERT(k_cur && k_cur->type == GGML_TYPE_F16 && "PagedAttn fused: K_cur must be F16");
    GGML_ASSERT(v_cur && v_cur->type == GGML_TYPE_F16 && "PagedAttn fused: V_cur must be F16");
    GGML_ASSERT(slot_mapping && slot_mapping->type == GGML_TYPE_I32 && "PagedAttn fused: slot_mapping must be I32");
    GGML_ASSERT(k_cur->ne[1] == n_kv_heads);
    GGML_ASSERT(v_cur->ne[1] == n_kv_heads);
    GGML_ASSERT(slot_mapping->ne[0] == k_cur->ne[2]);

    // For smem sizing we need the longest context in this batch.
    // Cheap upper bound: max_blocks_per_seq * block_size.
    const int max_ctx_len = max_bps * block_size;

    cudaStream_t stream = ctx.stream();

    // Dispatch on (head_size, block_size). Add cases as models need.
    auto run = [&](auto head_size_const, auto block_size_const) {
        constexpr int HS = decltype(head_size_const)::value;
        constexpr int BS = decltype(block_size_const)::value;
        launch_paged_attn<__half, __half, HS, BS>(
            (__half *) dst->data,
            (const __half *) q->data,
            (__half *) k_cache->data,
            (__half *) v_cache->data,
            (const int32_t *) block_tables->data,
            (const int32_t *) context_lens->data,
            (const int32_t *) q_lens->data,
            (const __half *) k_cur->data,
            (const __half *) v_cur->data,
            (const int32_t *) slot_mapping->data,
            num_seqs, n_heads, n_kv_heads, max_bps, max_ctx_len,
            scale, stream);
    };

    // Most common cases first; fall through to a runtime error for
    // unsupported (head_size, block_size) so we discover them loudly.
    if (head_size == 128 && block_size == 16) {
        run(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 64 && block_size == 16) {
        run(std::integral_constant<int, 64>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 256 && block_size == 16) {
        run(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 128 && block_size == 32) {
        run(std::integral_constant<int, 128>{}, std::integral_constant<int, 32>{});
    } else {
        GGML_ABORT("mt_paged_attn: unsupported (head_size=%d, block_size=%d) — add a template instantiation",
                   head_size, block_size);
    }
}

}  // namespace mt
