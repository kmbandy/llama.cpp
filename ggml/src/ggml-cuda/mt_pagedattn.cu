// mt_pagedattn — paged attention kernel implementation.
//
// See mt_pagedattn.cuh for layout and threading model docs.
// Design adapted from vLLM (Apache 2.0). Code is independent.

#include "mt_pagedattn.cuh"

#include <cmath>
#include <cstdio>

namespace mt {

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
    const cache_t  * __restrict__ k_cache,
    const cache_t  * __restrict__ v_cache,
    const int32_t  * __restrict__ block_tables,
    const int32_t  * __restrict__ context_lens,
    const int32_t  * __restrict__ q_lens,
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
//   src[1] = K cache [paged layout]
//   src[2] = V cache [paged layout]
//   src[3] = block_tables [max_blocks_per_seq, num_seqs]
//   src[4] = context_lens [num_seqs]
//   src[5] = q_lens       [num_seqs]
// dst:
//   out [head_size, n_heads, sum(q_lens), 1]
//
// For the v1 single-batch case sum(q_lens) collapses to q_len * num_seqs
// when all seqs in the batch have the same q_len (typical decode batch).

template <typename scalar_t, typename cache_t,
          int HEAD_SIZE, int BLOCK_SIZE>
static void launch_paged_attn(
    scalar_t       * out,
    const scalar_t * q,
    const cache_t  * k_cache,
    const cache_t  * v_cache,
    const int32_t  * block_tables,
    const int32_t  * context_lens,
    const int32_t  * q_lens,
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

    mt_paged_attention_kernel<scalar_t, cache_t, HEAD_SIZE, BLOCK_SIZE, NUM_THREADS, /*PARTITION_SIZE=*/0>
        <<<grid, block, smem_bytes, stream>>>(
            out, q, k_cache, v_cache,
            block_tables, context_lens, q_lens,
            max_blocks_per_seq, n_kv_heads, n_heads, scale);
}

void ggml_cuda_op_paged_attn_mt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q             = dst->src[0];
    const ggml_tensor * k_cache       = dst->src[1];
    const ggml_tensor * v_cache       = dst->src[2];
    const ggml_tensor * block_tables  = dst->src[3];
    const ggml_tensor * context_lens  = dst->src[4];
    const ggml_tensor * q_lens        = dst->src[5];

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
            (const __half *) k_cache->data,
            (const __half *) v_cache->data,
            (const int32_t *) block_tables->data,
            (const int32_t *) context_lens->data,
            (const int32_t *) q_lens->data,
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

// ─── Phase 3.4b-2: paged K/V scatter ──────────────────────────────────────
//
// Writes K_cur/V_cur into the block-indexed cache at the positions given by
// slot_mapping. Layout matches the attention kernel above (interleaved K,
// transposed V) — see mt_pagedattn.cuh for the layout contract.

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int X>
__global__ void mt_reshape_and_cache_kernel(
    const scalar_t * __restrict__ k_cur,        // [head_dim, n_kv_heads, n_tokens]
    const scalar_t * __restrict__ v_cur,        // [head_dim, n_kv_heads, n_tokens]
    scalar_t       * __restrict__ k_cache,      // [num_blocks, n_kv_heads, head_dim/x, block_size, x]
    scalar_t       * __restrict__ v_cache,      // [num_blocks, n_kv_heads, head_dim, block_size]
    const int32_t  * __restrict__ slot_mapping, // [n_tokens]
    int n_kv_heads,
    int n_tokens) {

    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int dim_idx   = threadIdx.x;

    if (token_idx >= n_tokens || head_idx >= n_kv_heads || dim_idx >= HEAD_SIZE) {
        return;
    }

    const int slot = slot_mapping[token_idx];
    if (slot < 0) {
        // Padding token — skip.
        return;
    }

    const int block_idx     = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    // K_cur / V_cur: ne[0]=head_dim (fast), ne[1]=n_kv_heads, ne[2]=n_tokens.
    // Flat = token_idx * (n_kv_heads * HEAD_SIZE) + head_idx * HEAD_SIZE + dim_idx.
    const int src_idx = token_idx * (n_kv_heads * HEAD_SIZE) + head_idx * HEAD_SIZE + dim_idx;
    const scalar_t k_val = k_cur[src_idx];
    const scalar_t v_val = v_cur[src_idx];

    // K cache layout: [num_blocks, n_kv_heads, HEAD_SIZE/X, BLOCK_SIZE, X]
    // Index = block * (n_kv_heads * HEAD_SIZE * BLOCK_SIZE)
    //       + head  * (HEAD_SIZE * BLOCK_SIZE)
    //       + dim_outer * (BLOCK_SIZE * X)
    //       + slot_in_block * X
    //       + dim_inner
    const int dim_outer = dim_idx / X;
    const int dim_inner = dim_idx % X;
    const int k_idx = block_idx * (n_kv_heads * HEAD_SIZE * BLOCK_SIZE)
                    + head_idx  * (HEAD_SIZE * BLOCK_SIZE)
                    + dim_outer * (BLOCK_SIZE * X)
                    + slot_in_block * X
                    + dim_inner;
    k_cache[k_idx] = k_val;

    // V cache layout: [num_blocks, n_kv_heads, HEAD_SIZE, BLOCK_SIZE]
    // Index = block * (n_kv_heads * HEAD_SIZE * BLOCK_SIZE)
    //       + head  * (HEAD_SIZE * BLOCK_SIZE)
    //       + dim_idx * BLOCK_SIZE
    //       + slot_in_block
    const int v_idx = block_idx * (n_kv_heads * HEAD_SIZE * BLOCK_SIZE)
                    + head_idx  * (HEAD_SIZE * BLOCK_SIZE)
                    + dim_idx   * BLOCK_SIZE
                    + slot_in_block;
    v_cache[v_idx] = v_val;
}

template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE>
static void launch_paged_kv_update(
    const scalar_t * k_cur,
    const scalar_t * v_cur,
    scalar_t       * k_cache,
    scalar_t       * v_cache,
    const int32_t  * slot_mapping,
    int n_kv_heads,
    int n_tokens,
    cudaStream_t stream) {
    constexpr int X = 16 / sizeof(scalar_t);  // 8 for fp16
    static_assert(HEAD_SIZE % X == 0, "HEAD_SIZE must divide evenly into vector groups of x");

    dim3 grid(n_tokens, n_kv_heads);
    dim3 block(HEAD_SIZE);

    mt_reshape_and_cache_kernel<scalar_t, HEAD_SIZE, BLOCK_SIZE, X>
        <<<grid, block, 0, stream>>>(
            k_cur, v_cur, k_cache, v_cache, slot_mapping,
            n_kv_heads, n_tokens);
}

void ggml_cuda_op_paged_kv_update_mt(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_cur        = dst->src[0];
    const ggml_tensor * v_cur        = dst->src[1];
    const ggml_tensor * k_cache      = dst->src[2];
    const ggml_tensor * v_cache      = dst->src[3];
    const ggml_tensor * slot_mapping = dst->src[4];

    const int32_t * params_i32 = (const int32_t *)(dst->op_params);
    const int32_t   block_size = params_i32[0];
    const int32_t   n_kv_heads = params_i32[1];

    // K_cur shape: [head_dim, n_kv_heads, n_tokens]
    const int head_size = (int) k_cur->ne[0];
    const int n_tokens  = (int) k_cur->ne[2];

    GGML_ASSERT(k_cur->type == GGML_TYPE_F16 && "PagedKVUpdate v1: F16 K_cur only");
    GGML_ASSERT(v_cur->type == GGML_TYPE_F16);
    GGML_ASSERT(k_cache->type == GGML_TYPE_F16);
    GGML_ASSERT(v_cache->type == GGML_TYPE_F16);
    GGML_ASSERT(slot_mapping->type == GGML_TYPE_I32);
    GGML_ASSERT(slot_mapping->ne[0] == n_tokens);

    if (n_tokens == 0) {
        return;  // nothing to scatter
    }

    cudaStream_t stream = ctx.stream();

    auto run = [&](auto head_size_const, auto block_size_const) {
        constexpr int HS = decltype(head_size_const)::value;
        constexpr int BS = decltype(block_size_const)::value;
        launch_paged_kv_update<__half, HS, BS>(
            (const __half *) k_cur->data,
            (const __half *) v_cur->data,
            (__half *)       k_cache->data,
            (__half *)       v_cache->data,
            (const int32_t *) slot_mapping->data,
            n_kv_heads, n_tokens, stream);
    };

    // Mirror the attention dispatch's (head_size, block_size) matrix.
    if (head_size == 128 && block_size == 16) {
        run(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 64 && block_size == 16) {
        run(std::integral_constant<int, 64>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 256 && block_size == 16) {
        run(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
    } else if (head_size == 128 && block_size == 32) {
        run(std::integral_constant<int, 128>{}, std::integral_constant<int, 32>{});
    } else {
        GGML_ABORT("mt_paged_kv_update: unsupported (head_size=%d, block_size=%d) — add a template instantiation",
                   head_size, block_size);
    }
}

}  // namespace mt
