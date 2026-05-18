// mt_pagedattn_aiter.cu — AITER-backed paged attention path. See the header
// for design notes.
//
// Only compiled when ggml-hip is built with -DGGML_HIP_AITER=ON. The
// non-AITER builds get the inline no-op stubs from the header.

#include "mt_pagedattn_aiter.cuh"

#ifdef GGML_HIP_AITER

#include "common.cuh"

#include <cstdlib>
#include <cstdio>
#include <atomic>
#include <mutex>

// The runtime AITER wrapper. Lives in aiter-integration's static library
// (libaiter_triton_aot.a), linked into ggml-hip when GGML_HIP_AITER=ON.
// Header propagated via aiter_triton_aot's PUBLIC target_include_directories.
#include "mt_aiter_unified_attn.h"

namespace mt {

// ─────────────────────────────────────────────────────────────────────────
// AITER-format scatter kernel (F16 cache only for v1)
//
// Layout: [num_blocks, block_size, n_kv_heads, head_size], no interleaving.
// Equivalent to vLLM/AITER's `unified_attention` K/V cache shape — keeps the
// scatter dead-simple compared to ggml's vectorized K layout.
// ─────────────────────────────────────────────────────────────────────────
template <typename scalar_t, int HEAD_SIZE, int BLOCK_SIZE, int NUM_THREADS>
__global__ void mt_scatter_kv_kernel_aiter(
    scalar_t       * __restrict__ k_cache,
    scalar_t       * __restrict__ v_cache,
    const scalar_t * __restrict__ k_cur,        // [head_dim, n_kv_heads, n_tokens]
    const scalar_t * __restrict__ v_cur,        // [head_dim, n_kv_heads, n_tokens]
    const int32_t  * __restrict__ slot_mapping, // [n_tokens]
    const int32_t  * __restrict__ q_lens,       // [num_seqs]
    int             n_kv_heads) {

    const int kv_head_idx = blockIdx.x;
    const int seq_idx     = blockIdx.y;
    const int tid         = threadIdx.x;

    constexpr int VEC_PER_THREAD = (HEAD_SIZE + NUM_THREADS - 1) / NUM_THREADS;

    const int q_len = q_lens[seq_idx];
    // Per-seq offset into the packed k_cur/v_cur tensor — seq tokens are
    // concatenated in seq_id order on the ne[2] axis.
    size_t seq_q_offset = 0;
    for (int s = 0; s < seq_idx; ++s) seq_q_offset += (size_t) q_lens[s];

    for (int t = 0; t < q_len; ++t) {
        const int global_token_idx = (int)(seq_q_offset + t);
        const int slot = slot_mapping[global_token_idx];
        if (slot < 0) continue;  // padding token

        const int block_idx     = slot / BLOCK_SIZE;
        const int slot_in_block = slot % BLOCK_SIZE;

        const size_t src_base = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                              + (size_t) kv_head_idx     * HEAD_SIZE;
        const size_t dst_base = (size_t) block_idx     * BLOCK_SIZE * n_kv_heads * HEAD_SIZE
                              + (size_t) slot_in_block * n_kv_heads * HEAD_SIZE
                              + (size_t) kv_head_idx   * HEAD_SIZE;

        #pragma unroll
        for (int v = 0; v < VEC_PER_THREAD; ++v) {
            const int d = tid + v * NUM_THREADS;
            if (d < HEAD_SIZE) {
                k_cache[dst_base + d] = k_cur[src_base + d];
                v_cache[dst_base + d] = v_cur[src_base + d];
            }
        }
    }
}

// Build the AITER `query_start_len` cu-seqlens tensor [num_seqs+1] on device
// from q_lens [num_seqs]. Tiny — one thread block.
__global__ void mt_build_cu_seqlens_kernel(
    int32_t       * __restrict__ cu_seqlens,
    const int32_t * __restrict__ q_lens,
    int             num_seqs) {
    if (threadIdx.x != 0) return;  // 1 thread; trivial sequential prefix sum
    int32_t acc = 0;
    cu_seqlens[0] = 0;
    for (int s = 0; s < num_seqs; ++s) {
        acc += q_lens[s];
        cu_seqlens[s + 1] = acc;
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Runtime gate
// ─────────────────────────────────────────────────────────────────────────
bool aiter_backend_enabled() {
    static std::atomic<int> cached{-1};  // -1 = unset, 0 = off, 1 = on
    int v = cached.load(std::memory_order_relaxed);
    if (v < 0) {
        const char * env = std::getenv("MAD_USE_AITER");
        v = (env && *env && env[0] != '0') ? 1 : 0;
        cached.store(v, std::memory_order_relaxed);
        if (v) {
            std::fprintf(stderr, "mt_pagedattn: AITER backend ENABLED (MAD_USE_AITER=%s)\n", env);
        }
    }
    return v == 1;
}

// ─────────────────────────────────────────────────────────────────────────
// One-time-allocated 1.0f device buffer for q/k/v/out descale (unquantized
// path passes ones).
// ─────────────────────────────────────────────────────────────────────────
static float * descale_ones_device() {
    static float * ptr = nullptr;
    static std::atomic<bool> ready{false};
    if (ready.load(std::memory_order_acquire)) return ptr;
    // Init under a coarse lock — rare, on first AITER call.
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (ready.load(std::memory_order_relaxed)) return ptr;
    cudaMalloc((void**) &ptr, sizeof(float));
    const float one = 1.0f;
    cudaMemcpy(ptr, &one, sizeof(float), cudaMemcpyHostToDevice);
    ready.store(true, std::memory_order_release);
    return ptr;
}

// ─────────────────────────────────────────────────────────────────────────
// AITER dispatch entry
// ─────────────────────────────────────────────────────────────────────────
void ggml_cuda_op_paged_attn_mt_aiter(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
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

    const int head_size      = (int) q->ne[0];
    const int n_heads        = (int) q->ne[1];
    const int num_seqs       = (int) block_tables->ne[1];
    const int num_q_tokens   = (int) k_cur->ne[2];

    // Shape gate — the wrapper builds a Triton signature from these at first
    // call and the runtime registry compiles a matching kernel. We just need
    // to make sure dtypes and basic GQA invariants hold; the actual
    // dimensions are passed through.
    GGML_ASSERT(q->type == GGML_TYPE_F16 && "AITER backend requires F16 Q");
    GGML_ASSERT(k_cache->type == GGML_TYPE_F16 && "AITER backend requires F16 KV cache for v1");
    GGML_ASSERT(n_heads % n_kv_heads == 0 && "n_heads must be divisible by n_kv_heads");
    // Triton signature uses HEAD_SIZE_PADDED = next pow2 of HEAD_SIZE; for now
    // we only support power-of-2 head sizes (covers 64/128/256 — all common
    // LLM head_dim values).
    GGML_ASSERT(head_size > 0 && (head_size & (head_size - 1)) == 0 && "AITER backend requires power-of-2 head_size");

    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = head_size;
    shape.num_q_heads  = n_heads;
    shape.num_kv_heads = n_kv_heads;
    shape.block_size   = block_size;

    cudaStream_t stream = ctx.stream();

    // ── 1. Scatter K_cur/V_cur into AITER-layout cache ──
    // Dispatch on common (head_size, block_size) at compile time so the
    // kernel can unroll its inner loops. New shapes get added here when a new
    // model needs them.
    {
        constexpr int NUM_THREADS = 128;
        dim3 grid(n_kv_heads, num_seqs);
        dim3 block(NUM_THREADS);
        auto launch_aiter_scatter = [&](auto HS_const, auto BS_const) {
            constexpr int HS = decltype(HS_const)::value;
            constexpr int BS = decltype(BS_const)::value;
            mt_scatter_kv_kernel_aiter<__half, HS, BS, NUM_THREADS>
                <<<grid, block, 0, stream>>>(
                    (__half*) k_cache->data,
                    (__half*) v_cache->data,
                    (const __half*) k_cur->data,
                    (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data,
                    (const int32_t*) q_lens->data,
                    n_kv_heads);
        };
        if (head_size == 128 && block_size == 16) {
            launch_aiter_scatter(std::integral_constant<int, 128>{}, std::integral_constant<int, 16>{});
        } else if (head_size == 64 && block_size == 16) {
            launch_aiter_scatter(std::integral_constant<int, 64>{}, std::integral_constant<int, 16>{});
        } else if (head_size == 256 && block_size == 16) {
            launch_aiter_scatter(std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
        } else {
            GGML_ABORT("AITER scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
        }
    }

    // ── 2. Allocate AITER workspace + cu_seqlens ──
    const size_t segm_out_n   = mt_aiter_uattn_segm_output_bytes(&shape, num_q_tokens) / sizeof(float);
    const size_t segm_ms_n    = mt_aiter_uattn_segm_max_bytes(&shape, num_q_tokens)    / sizeof(float);
    ggml_cuda_pool_alloc<float>   segm_out_buf(ctx.pool(), segm_out_n);
    ggml_cuda_pool_alloc<float>   segm_max_buf(ctx.pool(), segm_ms_n);
    ggml_cuda_pool_alloc<float>   segm_exp_buf(ctx.pool(), segm_ms_n);
    ggml_cuda_pool_alloc<int32_t> cu_seqlens_buf(ctx.pool(), (size_t)(num_seqs + 1));

    mt_build_cu_seqlens_kernel<<<1, 1, 0, stream>>>(
        cu_seqlens_buf.get(), (const int32_t*) q_lens->data, num_seqs);

    // ── 3. Launch AITER attention via the runtime wrapper ──
    mt_aiter_uattn_args_t args = {};
    args.shape        = shape;
    args.q            = q->data;
    args.k_cache      = k_cache->data;
    args.v_cache      = v_cache->data;
    args.out          = dst->data;
    args.segm_output  = segm_out_buf.get();
    args.segm_max     = segm_max_buf.get();
    args.segm_expsum  = segm_exp_buf.get();
    args.block_tables = (const int32_t*) block_tables->data;
    args.seq_lens     = (const int32_t*) context_lens->data;
    args.query_start_len = cu_seqlens_buf.get();

    float * ones = descale_ones_device();
    args.q_descale   = ones;
    args.k_descale   = ones;
    args.v_descale   = ones;
    args.out_scale   = ones;

    args.scale              = scale;
    args.num_seqs           = num_seqs;
    args.num_q_tokens       = num_q_tokens;
    args.block_table_stride = max_bps;
    args.q_stride_0         = (int64_t) n_heads * head_size;
    args.output_stride_0    = args.q_stride_0;
    args.k_stride_0         = (int64_t) block_size * n_kv_heads * head_size;
    args.k_stride_1         = (int64_t) n_kv_heads * head_size;
    args.k_stride_2         = head_size;
    args.v_stride_0         = args.k_stride_0;
    args.v_stride_1         = args.k_stride_1;
    args.v_stride_2         = args.k_stride_2;

    hipError_t err = mt_aiter_unified_attn(stream, &args);
    if (err != hipSuccess) {
        GGML_ABORT("mt_aiter_unified_attn launch failed: %s", hipGetErrorString(err));
    }
}

}  // namespace mt

#endif  // GGML_HIP_AITER
