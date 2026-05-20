// mt_pagedattn_aiter.cu — AITER-backed paged attention path. See the header
// for design notes.
//
// Only compiled when ggml-hip is built with -DGGML_HIP_AITER=ON. The
// non-AITER builds get the inline no-op stubs from the header.

#include "mt_pagedattn_aiter.cuh"

#ifdef GGML_HIP_AITER

#include "common.cuh"
#include "turbo-quant.cuh"   // MAD-199: TURBO_CENTROIDS_{3,4}BIT, turbo_nearest_centroid_{3,4}bit

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

// ─────────────────────────────────────────────────────────────────────────
// AITER-layout TURBO3 scatter (MAD-199).
//
// Block geometry: 128 threads per block (one block per (token, kv_head,
// qb_idx)). Pipeline mirrors mt_scatter_kv_turbo3_0_kernel in
// mt_pagedattn.cu — same load, parallel L2 norm, normalize, nearest-3-bit
// centroid, pack qs (32-byte) + signs (16-byte), reconstruction norm. The
// ONLY difference is the destination block-index math: AITER layout is
// `[num_paged_blocks, BLOCK_SIZE, n_kv_heads, N_QBLOCKS_PER_TOKEN]` while
// paged-tile uses `[num_paged_blocks, n_kv_heads, BLOCK_SIZE, ...]`. Same
// content per element, different memory ordering — the Triton load
// helpers in unified_attention.py expect THIS layout.
//
// As with paged-tile, RHT is intentionally skipped: TURBO_CENTROIDS_3BIT
// is Lloyd-Max for N(0, 1/d) which matches normalized K vectors directly.
// ─────────────────────────────────────────────────────────────────────────
template <int HEAD_SIZE, int BLOCK_SIZE>
__launch_bounds__(QK_TURBO3)
__global__ void mt_scatter_kv_turbo3_aiter_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    int             n_kv_heads) {

    constexpr int Q_BLOCK             = QK_TURBO3;            // 128
    constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;  // 1 at HS=128
    constexpr int N_WARPS             = Q_BLOCK / WARP_SIZE;  // 4
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO3");
    static_assert(Q_BLOCK == 128, "this kernel assumes QK_TURBO3 == 128");

    const int j                = threadIdx.x;
    const int global_token_idx = blockIdx.x;
    const int y_idx            = blockIdx.y;
    const int kv_select        = blockIdx.z;   // 0 = K, 1 = V
    const int kv_head_idx      = y_idx / N_QBLOCKS_PER_TOKEN;
    const int qb_idx           = y_idx % N_QBLOCKS_PER_TOKEN;

    const int slot = slot_mapping[global_token_idx];
    if (slot < 0) return;

    const int paged_block   = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    const int    d   = qb_idx * Q_BLOCK + j;
    const __half * src = (kv_select == 0) ? k_cur : v_cur;
    const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                         + (size_t) kv_head_idx     * HEAD_SIZE
                         + (size_t) d;

    // AITER-layout block index: [paged_block, slot_in_block, kv_head, qb_idx]
    void * dst_buf = (kv_select == 0) ? k_cache : v_cache;
    const int64_t block_ib =
          ((int64_t) paged_block * BLOCK_SIZE * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) slot_in_block * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) kv_head_idx * N_QBLOCKS_PER_TOKEN)
        + (int64_t) qb_idx;
    block_turbo3_0 * blk = (block_turbo3_0 *) dst_buf + block_ib;

    __shared__ float x[Q_BLOCK];
    x[j] = __half2float(src[src_off]);
    __syncthreads();

    __shared__ float warp_accum[N_WARPS];
    {
        float v_sq = x[j] * x[j];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v_sq += __shfl_xor_sync(0xffffffffu, v_sq, offset);
        }
        if (j % WARP_SIZE == 0) warp_accum[j / WARP_SIZE] = v_sq;
    }
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < N_WARPS; ++w) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    x[j] *= inv_norm;
    __syncthreads();

    const float   rv  = x[j];
    const uint8_t idx = turbo_nearest_centroid_3bit(rv);  // 0..7

    // Pack qs: 4 elements per byte, low 2 bits each (per-warp).
    {
        const int     lane    = j % WARP_SIZE;
        const int     warp_id = j / WARP_SIZE;
        const uint8_t my2     = idx & 0x3;
        uint8_t byte_val      = my2 << ((lane & 3) * 2);
        byte_val |= __shfl_xor_sync(0xffffffffu, byte_val, 1);
        byte_val |= __shfl_xor_sync(0xffffffffu, byte_val, 2);
        if ((lane & 3) == 0) {
            blk->qs[warp_id * (WARP_SIZE / 4) + lane / 4] = byte_val;
        }
    }

    // Pack signs: 8 elements per byte, high 1 bit each (per-warp).
    {
        const int     lane    = j % WARP_SIZE;
        const int     warp_id = j / WARP_SIZE;
        const uint8_t my1     = (idx >> 2) & 0x1;
        uint8_t bits          = my1 << (lane & 7);
        bits |= __shfl_xor_sync(0xffffffffu, bits, 1);
        bits |= __shfl_xor_sync(0xffffffffu, bits, 2);
        bits |= __shfl_xor_sync(0xffffffffu, bits, 4);
        if ((lane & 7) == 0) {
            blk->signs[warp_id * (WARP_SIZE / 8) + lane / 8] = bits;
        }
    }

    // Reconstruction norm: ||centroid·norm|| should equal ||K||; correct norm
    // if drift > 1e-10.
    {
        const float c = TURBO_CENTROIDS_3BIT[idx];
        float rc = c * c;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            rc += __shfl_xor_sync(0xffffffffu, rc, offset);
        }
        if (j % WARP_SIZE == 0) warp_accum[j / WARP_SIZE] = rc;
    }
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < N_WARPS; ++w) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }
}

// ─────────────────────────────────────────────────────────────────────────
// AITER-layout TURBO4 scatter (MAD-199). Same shape as turbo3 above; only
// the index packing (4-bit nibble) and block struct (block_turbo4_0) differ.
// ─────────────────────────────────────────────────────────────────────────
template <int HEAD_SIZE, int BLOCK_SIZE>
__launch_bounds__(QK_TURBO4)
__global__ void mt_scatter_kv_turbo4_aiter_kernel(
    void           * __restrict__ k_cache,
    void           * __restrict__ v_cache,
    const __half   * __restrict__ k_cur,
    const __half   * __restrict__ v_cur,
    const int32_t  * __restrict__ slot_mapping,
    int             n_kv_heads) {

    constexpr int Q_BLOCK             = QK_TURBO4;            // 128
    constexpr int N_QBLOCKS_PER_TOKEN = HEAD_SIZE / Q_BLOCK;
    constexpr int N_WARPS             = Q_BLOCK / WARP_SIZE;  // 4
    static_assert(HEAD_SIZE % Q_BLOCK == 0, "HEAD_SIZE must be divisible by QK_TURBO4");
    static_assert(Q_BLOCK == 128, "this kernel assumes QK_TURBO4 == 128");

    const int j                = threadIdx.x;
    const int global_token_idx = blockIdx.x;
    const int y_idx            = blockIdx.y;
    const int kv_select        = blockIdx.z;
    const int kv_head_idx      = y_idx / N_QBLOCKS_PER_TOKEN;
    const int qb_idx           = y_idx % N_QBLOCKS_PER_TOKEN;

    const int slot = slot_mapping[global_token_idx];
    if (slot < 0) return;

    const int paged_block   = slot / BLOCK_SIZE;
    const int slot_in_block = slot % BLOCK_SIZE;

    const int    d   = qb_idx * Q_BLOCK + j;
    const __half * src = (kv_select == 0) ? k_cur : v_cur;
    const size_t src_off = (size_t) global_token_idx * n_kv_heads * HEAD_SIZE
                         + (size_t) kv_head_idx     * HEAD_SIZE
                         + (size_t) d;

    void * dst_buf = (kv_select == 0) ? k_cache : v_cache;
    const int64_t block_ib =
          ((int64_t) paged_block * BLOCK_SIZE * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) slot_in_block * n_kv_heads * N_QBLOCKS_PER_TOKEN)
        + ((int64_t) kv_head_idx * N_QBLOCKS_PER_TOKEN)
        + (int64_t) qb_idx;
    block_turbo4_0 * blk = (block_turbo4_0 *) dst_buf + block_ib;

    __shared__ float x[Q_BLOCK];
    x[j] = __half2float(src[src_off]);
    __syncthreads();

    __shared__ float warp_accum[N_WARPS];
    {
        float v_sq = x[j] * x[j];
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            v_sq += __shfl_xor_sync(0xffffffffu, v_sq, offset);
        }
        if (j % WARP_SIZE == 0) warp_accum[j / WARP_SIZE] = v_sq;
    }
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < N_WARPS; ++w) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? (1.0f / grp_norm) : 0.0f;

    x[j] *= inv_norm;
    __syncthreads();

    const float   rv  = x[j];
    const uint8_t idx = turbo_nearest_centroid_4bit(rv);  // 0..15

    // Pack qs: 2 nibbles per byte (warp-cooperative).
    {
        const int      lane            = j % WARP_SIZE;
        const uint8_t  my_nibble       = idx & 0xF;
        const uint8_t  partner_nibble  = __shfl_sync(0xffffffffu, my_nibble, lane ^ 1);
        if ((j & 1) == 0) {
            blk->qs[j / 2] = my_nibble | (partner_nibble << 4);
        }
    }

    {
        const float c = TURBO_CENTROIDS_4BIT[idx];
        float rc = c * c;
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            rc += __shfl_xor_sync(0xffffffffu, rc, offset);
        }
        if (j % WARP_SIZE == 0) warp_accum[j / WARP_SIZE] = rc;
    }
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < N_WARPS; ++w) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? (grp_norm / recon_norm) : grp_norm;

    if (j == 0) {
        blk->norm  = __float2half(corrected_norm);
        blk->rnorm = __float2half(0.0f);  // reserved/unused in 4-bit mode
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
    // call and the runtime registry compiles a matching kernel.
    GGML_ASSERT(q->type == GGML_TYPE_F16 && "AITER backend requires F16 Q");
    GGML_ASSERT(k_cache->type == v_cache->type && "AITER backend requires K and V cache to be the same type");
    GGML_ASSERT(n_heads % n_kv_heads == 0 && "n_heads must be divisible by n_kv_heads");
    GGML_ASSERT(head_size > 0 && (head_size & (head_size - 1)) == 0 && "AITER backend requires power-of-2 head_size");

    // MAD-199: route the cache ggml_type to the wrapper's mt_aiter_cache_type
    // enum. Triton-side dispatch on this value selects the right K/V load +
    // dequant path inside kernel_unified_attention_3d.
    int cache_type;
    switch (k_cache->type) {
        case GGML_TYPE_F16:      cache_type = MT_AITER_CACHE_F16;    break;
        case GGML_TYPE_TURBO3_0: cache_type = MT_AITER_CACHE_TURBO3; break;
        case GGML_TYPE_TURBO4_0: cache_type = MT_AITER_CACHE_TURBO4; break;
        default:
            GGML_ABORT("AITER backend: unsupported KV cache type %d (only F16, TURBO3_0, TURBO4_0)", (int) k_cache->type);
    }

    mt_aiter_uattn_shape_t shape {};
    shape.head_size    = head_size;
    shape.num_q_heads  = n_heads;
    shape.num_kv_heads = n_kv_heads;
    shape.block_size   = block_size;
    shape.cache_type   = cache_type;

    cudaStream_t stream = ctx.stream();

    // ── 1. Scatter K_cur/V_cur into AITER-layout cache ──
    // Dispatch on (cache_type, head_size, block_size) at compile time so the
    // kernel can unroll its inner loops. Add instantiations here when a new
    // model needs an unsupported shape.
    if (cache_type == MT_AITER_CACHE_F16) {
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
            GGML_ABORT("AITER F16 scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
        }
    } else {
        // MAD-199: TURBO3 / TURBO4 scatter. 128 threads per block (= QK_TURBO).
        // Grid: (num_tokens, n_kv_heads * N_QBLOCKS_PER_TOKEN, 2_for_K_and_V).
        // The kernel templates derive N_QBLOCKS_PER_TOKEN from HEAD_SIZE.
        const int n_qblocks_per_token = head_size / 128;  // QK_TURBO3 == QK_TURBO4 == 128
        dim3 grid(num_q_tokens, n_kv_heads * n_qblocks_per_token, 2);
        dim3 block(128);

        if (cache_type == MT_AITER_CACHE_TURBO3) {
            if (head_size == 128 && block_size == 16) {
                mt_scatter_kv_turbo3_aiter_kernel<128, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else if (head_size == 256 && block_size == 16) {
                mt_scatter_kv_turbo3_aiter_kernel<256, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else {
                GGML_ABORT("AITER TURBO3 scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
            }
        } else {  // MT_AITER_CACHE_TURBO4
            if (head_size == 128 && block_size == 16) {
                mt_scatter_kv_turbo4_aiter_kernel<128, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else if (head_size == 256 && block_size == 16) {
                mt_scatter_kv_turbo4_aiter_kernel<256, 16><<<grid, block, 0, stream>>>(
                    k_cache->data, v_cache->data,
                    (const __half*) k_cur->data, (const __half*) v_cur->data,
                    (const int32_t*) slot_mapping->data, n_kv_heads);
            } else {
                GGML_ABORT("AITER TURBO4 scatter: add a (head_size=%d, block_size=%d) instantiation", head_size, block_size);
            }
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
