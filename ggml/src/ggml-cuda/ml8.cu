// ml8.cu — GGML_TYPE_ML8_4 on-device repack for the HIP backend.
// See ml8.cuh for the contract and motivation. MAD-223 Phase G.4.d.

#include "ml8.cuh"

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include "ggml.h"
#include "common.cuh"
#include "convert.cuh"
#include "mt_ml8_gemm.h"
#include "turbo_fp8_hadamard.cuh"  // G.6.f: FWHT for rotation H_b leg

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <vector>

// G.6.g.C: debug hooks to dump rotation input + ml8_mul_mat output to /tmp
// for Python-side bit-equivalence comparison. Set env var ML8_DUMP=1 to
// enable. First-call-only; the static atomics track which dumps have fired.
namespace {
std::atomic<bool> g_ml8_dump_rot_done    {false};
std::atomic<bool> g_ml8_dump_rotdst_done {false};
std::atomic<bool> g_ml8_dump_mm_done     {false};
std::atomic<bool> g_ml8_dump_quant_done  {false};

void ml8_dump_u8(const char * path, const uint8_t * d_ptr, size_t n_elems,
                cudaStream_t stream, int ndim, const int64_t * shape) {
    std::vector<uint8_t> host(n_elems);
    cudaMemcpyAsync(host.data(), d_ptr, n_elems, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    FILE * f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "[ml8-dump] open %s failed\n", path); return; }
    std::fwrite(&ndim, sizeof(int32_t), 1, f);
    std::fwrite(shape, sizeof(int64_t), (size_t) ndim, f);
    std::fwrite(host.data(), 1, n_elems, f);
    std::fclose(f);
    std::fprintf(stderr, "[ml8-dump] wrote %s  ndim=%d  n=%zu\n", path, ndim, n_elems);
}

bool ml8_dump_enabled() {
    static const bool e = (std::getenv("ML8_DUMP") != nullptr);
    return e;
}

void ml8_dump_fp32(const char * path, const float * d_ptr, size_t n_elems,
                  cudaStream_t stream, int ndim, const int64_t * shape) {
    std::vector<float> host(n_elems);
    cudaMemcpyAsync(host.data(), d_ptr, n_elems * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    FILE * f = std::fopen(path, "wb");
    if (!f) { std::fprintf(stderr, "[ml8-dump] open %s failed\n", path); return; }
    // Header: int32 ndim, int64 * shape, then fp32 data
    std::fwrite(&ndim, sizeof(int32_t), 1, f);
    std::fwrite(shape, sizeof(int64_t), (size_t) ndim, f);
    std::fwrite(host.data(), sizeof(float), n_elems, f);
    std::fclose(f);
    std::fprintf(stderr, "[ml8-dump] wrote %s  ndim=%d  n=%zu\n", path, ndim, n_elems);
}
} // namespace

// On-disk per-block layout: 4-byte fp32 scale, then QK_ML8/2 = 32 packed
// nibble bytes covering 64 K-elements. sizeof(block_ml8_4) == 36.
static constexpr int ML8_BLOCK_BYTES   = (int) sizeof(block_ml8_4);
static constexpr int ML8_GROUP_NIBBLES = QK_ML8 / 2;   // == 32

// One thread per (n, g) pair. Reads the (4-byte scale + 32-byte nibbles)
// block from the on-disk row-major (N, n_groups_k * 36) layout and
// scatters into the separated (b_packed[K/2, N], b_scale[n_groups_k, N])
// layout. group_size is currently always QK_ML8 = 64 (ML8_GROUP_NIBBLES).
//
// Memory pattern: source reads are coalesced per warp (consecutive n
// threads → consecutive 36-byte blocks in memory). Destination writes
// are strided by N for b_packed and by N for b_scale, which is the
// price we pay for the [K/2, N] / [n_groups_k, N] layout the kernel
// downstream consumes — done once at load, never on the inference path.
static __global__ void ml8_repack_kernel(
    const uint8_t * __restrict__ src,        // (N, n_groups_k * 36) bytes
    uint8_t       * __restrict__ b_packed,   // (K/2, N) row-major
    float         * __restrict__ b_scale,    // (n_groups_k, N) row-major
    int N,
    int n_groups_k) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (n >= N || g >= n_groups_k) {
        return;
    }

    const uint8_t * blk = src
        + (size_t) n * (size_t) n_groups_k * (size_t) ML8_BLOCK_BYTES
        + (size_t) g * (size_t) ML8_BLOCK_BYTES;

    // Scale: 4 bytes at the start of the block.
    float scale;
    memcpy(&scale, blk, sizeof(float));
    b_scale[(size_t) g * (size_t) N + (size_t) n] = scale;

    // Nibbles: 32 bytes after the scale, covering K-half rows
    // [g * ML8_GROUP_NIBBLES, (g + 1) * ML8_GROUP_NIBBLES).
    const uint8_t * nibbles      = blk + sizeof(float);
    const int       k_half_base  = g * ML8_GROUP_NIBBLES;
    #pragma unroll
    for (int j = 0; j < ML8_GROUP_NIBBLES; ++j) {
        b_packed[((size_t) (k_half_base + j)) * (size_t) N + (size_t) n] = nibbles[j];
    }
}

void ggml_cuda_ml8_repack_blocks(
    cudaStream_t stream,
    const void * src_blocks,
    void *       dst_b_packed,
    float *      dst_b_scale,
    int32_t      N,
    int32_t      K,
    int32_t      group_size) {

    GGML_ASSERT(group_size == QK_ML8 && "ml8-4 repack only supports group_size == QK_ML8 (64)");
    GGML_ASSERT(N > 0);
    GGML_ASSERT(K > 0);
    GGML_ASSERT(K % group_size == 0);

    const int n_groups_k = K / group_size;

    constexpr int BLOCK_N = 64;
    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, n_groups_k, 1);
    const dim3 block(BLOCK_N, 1, 1);

    ml8_repack_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t *) src_blocks,
        (uint8_t *)       dst_b_packed,
        dst_b_scale,
        N,
        n_groups_k);
}

// ─────────────────────────────────────────────────────────────────────
// Cache layer.
// ─────────────────────────────────────────────────────────────────────

namespace {

struct cache_entry_t {
    ml8_weight_repack_t info;
};

std::mutex                                            g_ml8_cache_mu;
std::unordered_map<const void *, cache_entry_t>       g_ml8_cache;

} // namespace

const ml8_weight_repack_t * ggml_cuda_ml8_get_or_repack(
    cudaStream_t        stream,
    const ggml_tensor * w) {

    if (w == nullptr || w->data == nullptr) {
        return nullptr;
    }
    if (w->type != GGML_TYPE_ML8_4) {
        return nullptr;
    }

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    if (K <= 0 || N <= 0 || K % QK_ML8 != 0) {
        return nullptr;
    }
    const int32_t group_size = QK_ML8;
    const int32_t n_groups_k = K / group_size;

    const void * key = w->data;

    {
        std::lock_guard<std::mutex> lock(g_ml8_cache_mu);
        auto it = g_ml8_cache.find(key);
        if (it != g_ml8_cache.end()) {
            return &it->second.info;
        }
    }

    // Allocate device side buffers. These live until clear_cache() or
    // process exit.
    void *  d_b_packed = nullptr;
    float * d_b_scale  = nullptr;

    const size_t b_packed_bytes = (size_t) (K / 2) * (size_t) N;
    const size_t b_scale_bytes  = (size_t) n_groups_k * (size_t) N * sizeof(float);

    cudaError_t err = cudaMalloc(&d_b_packed, b_packed_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8] cudaMalloc(b_packed=%zu) failed: %s\n",
                b_packed_bytes, cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMalloc((void **) &d_b_scale, b_scale_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8] cudaMalloc(b_scale=%zu) failed: %s\n",
                b_scale_bytes, cudaGetErrorString(err));
        cudaFree(d_b_packed);
        return nullptr;
    }

    ggml_cuda_ml8_repack_blocks(
        stream,
        w->data,
        d_b_packed,
        d_b_scale,
        N,
        K,
        group_size);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8] repack kernel launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_b_packed);
        cudaFree(d_b_scale);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_ml8_cache_mu);
    // Re-check in case another thread raced us. If so, free ours and
    // return the winner's.
    auto it = g_ml8_cache.find(key);
    if (it != g_ml8_cache.end()) {
        cudaFree(d_b_packed);
        cudaFree(d_b_scale);
        return &it->second.info;
    }
    cache_entry_t entry{};
    entry.info.b_packed   = d_b_packed;
    entry.info.b_scale    = d_b_scale;
    entry.info.N          = N;
    entry.info.K          = K;
    entry.info.n_groups_k = n_groups_k;
    entry.info.group_size = group_size;
    auto [ins_it, _ins_ok] = g_ml8_cache.emplace(key, entry);
    return &ins_it->second.info;
}

void ggml_cuda_ml8_clear_cache(void) {
    std::lock_guard<std::mutex> lock(g_ml8_cache_mu);
    for (auto & kv : g_ml8_cache) {
        cudaFree(kv.second.info.b_packed);
        cudaFree(kv.second.info.b_scale);
    }
    g_ml8_cache.clear();
}

// ─────────────────────────────────────────────────────────────────────
// Per-row activation fp32 → e4m3 + scale.
// ─────────────────────────────────────────────────────────────────────

// Standard (non-fnuz) e4m3: bias = 7, m_bits = 3, max representable
// = ±448 (S.1111.110 = 2^8 × 1.75). NaN encoding = S.1111.111. Mirrors
// quantize_row_f8_e4m3_ref in ggml-turbo-quant.c (round-to-nearest-even,
// saturate at ±448). Pulled into a device function here because the
// existing ggml_cuda_fp32_to_ue4m3 in common.cuh targets NVFP4 sub-block
// scales on Blackwell (different format, different range) and isn't
// usable on RDNA.
static __device__ __forceinline__ uint8_t ml8_fp32_to_e4m3(float xv) {
    uint32_t bits;
    memcpy(&bits, &xv, 4);
    const uint32_t sign  = (bits >> 31) & 1u;
    const uint32_t exp_b = (bits >> 23) & 0xFFu;
    const uint32_t mant  = bits & 0x7FFFFFu;

    // NaN or Inf input → e4m3 NaN (S.1111.111).
    if (exp_b == 0xFFu) {
        return (uint8_t)((sign << 7) | 0x7Fu);
    }
    // Zero (and fp32 subnormals, which underflow to e4m3 zero).
    if (exp_b == 0) {
        return (uint8_t)(sign << 7);
    }

    const int32_t e_un = (int32_t) exp_b - 127;

    // Saturate to ±448 = e=15, m=6.
    if (e_un >= 9 || (e_un == 8 && mant >= 0x600000u)) {
        return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
    }

    if (e_un >= -6) {
        const uint32_t e_e4m3 = (uint32_t)(e_un + 7);
        const uint32_t guard  = (mant >> 19) & 1u;
        const uint32_t sticky = (mant & ((1u << 19) - 1)) != 0 ? 1u : 0u;
        const uint32_t lsb    = (mant >> 20) & 1u;
        uint32_t       m_e4m3 = (mant >> 20) & 0x7u;
        if (guard && (sticky || lsb)) m_e4m3 += 1;
        uint32_t e_out = e_e4m3;
        if (m_e4m3 == 8) {
            m_e4m3 = 0;
            e_out += 1;
            // G.6.g.C BUGFIX (2026-05-26): was `e_out >= 15`, which prematurely
            // saturated valid e=15, m=0..6 values (256, 288, ..., 448) to ±448.
            // Only e>15 (= e_real > 8) overflows the E4M3 finite range. The
            // m=7 NaN slot is handled by the `m_e4m3 == 7` guard below. This
            // bug cost ~+0.33 PPL on Cell E vs the Python kernel reference.
            if (e_out > 15) {
                return (uint8_t)((sign << 7) | (0xFu << 3) | 0x6u);
            }
        }
        if (e_out == 15 && m_e4m3 == 7) m_e4m3 = 6;
        return (uint8_t)((sign << 7) | (e_out << 3) | m_e4m3);
    }

    // Subnormal e4m3: |x| < 2^-6. m = round(|x| * 2^9) ∈ {0..7}.
    const int32_t shift = 23 - (e_un + 9);
    if (shift > 31) {
        return (uint8_t)(sign << 7);
    }
    const uint32_t implicit = (1u << 23) | mant;
    const uint32_t guard    = (implicit >> (shift - 1)) & 1u;
    const uint32_t sticky   = (implicit & ((1u << (shift - 1)) - 1)) != 0 ? 1u : 0u;
    uint32_t       m_e4m3   = implicit >> shift;
    const uint32_t lsb      = m_e4m3 & 1u;
    if (guard && (sticky || lsb)) m_e4m3 += 1;
    if (m_e4m3 >= 8) {
        // Rounded into smallest normal e4m3 (e=1, m=0).
        return (uint8_t)((sign << 7) | (1u << 3));
    }
    return (uint8_t)((sign << 7) | m_e4m3);
}

// E4M3 max representable value. Used to compute per-row scale such
// that `x / scale` lies in roughly [-448, +448].
static constexpr float ML8_FP8_E4M3_MAX = 448.0f;

// Epsilon to avoid divide-by-zero on all-zero rows. Picked so the
// scale stays representable in fp32 while making the cast a no-op
// (every element rounds to fp8 zero).
static constexpr float ML8_ACT_SCALE_EPS = 1e-12f;

// One block per row M. Each block:
//   1. Cooperatively reads K fp32 elements, computing per-thread |x|max.
//   2. Block-reduces to row absmax via shared memory.
//   3. Thread 0 writes a_scale[m] = absmax / 448 (with epsilon).
//   4. All threads quantize their slice: a_fp8[m, k] = e4m3(x / scale).
//
// ─────────────────────────────────────────────────────────────────────
// G.6.h: ml8 GEMV kernel for M=1 (decode hot path).
//
// At M=1 the standard ml8 mul_mat path pads M up to 16 and feeds the
// Triton blockscale gemm, which spends 15/16 of its compute on
// zero-padded rows. rocprofv3 (2026-05-26) showed the ml8 gemm at 70.6%
// of decode GPU time, 4× off memory-bandwidth ceiling.
//
// This naïve GEMV is the first-pass correctness target:
//   - 1 block per N-tile of size BN=64 output columns
//   - 1 thread per output column (no K-cooperative reduction yet)
//   - Each thread does the full K reduction, reading the same a[K]
//     and the per-K-group centroid LUT + scale.
//   - a[K] cached in LDS (loaded once per block, shared by all 64
//     threads = 64 output cols).
//
// Inputs:
//   a       : [K]                fp32 post-rotation activation
//   b_pack  : [K/2, N]           uint8 packed nibbles (lo at k=even, hi at k=odd)
//   b_scale : [n_groups_k, N]    fp32 per-(group, col) scale
//   lut     : [n_groups_k, 16]   fp8 e4m3 centroid LUT per K-group
//   c       : [N]                fp32 output
//
// Per-thread inner loop unrolls the 32-byte K-group as 32 nibble pairs.
// ─────────────────────────────────────────────────────────────────────

// GEMV tile: BN output columns × K_COOP threads per column.
// Block size = BN * K_COOP = 256 threads. 4-way K-cooperative reduction
// per output column splits the K loop across 4 threads, then merges via
// shared memory. The K-split must align to group boundaries (K_COOP must
// divide n_groups_k cleanly; for QK_ML8=64 and our K values 2560/9216,
// n_groups_k = 40/144 — both divisible by 4).
// G.6.h sweep: kernel is templated on <BN, K_COOP, USE_LDS_A, LAYOUT>.
// Dispatch reads env vars ML8_GEMV_BN / ML8_GEMV_K_COOP / ML8_GEMV_LDS_A /
// ML8_GEMV_LAYOUT and routes to the matching instantiation. After the
// sweep picks a winner, collapse to a single non-templated kernel.
//
// LAYOUT semantics:
//   LAYOUT=0 (cross_warp): tid = n_local + k_part * BN. K_COOP threads
//     reducing one col span multiple waves → must use LDS reduction.
//   LAYOUT=1 (within_warp): tid = k_part + n_local * K_COOP. K_COOP
//     threads for one col are consecutive lane IDs within a wave → can
//     use __shfl_xor for in-register reduction.

static __device__ __forceinline__ float ml8_fp8_e4m3_to_fp32(uint8_t b) {
    // Standard E4M3: bias=7, m=3 bits. NaN at S.1111.111.
    const uint32_t sign = (b >> 7) & 1u;
    const uint32_t exp_b = (b >> 3) & 0xFu;
    const uint32_t mant = b & 0x7u;
    if (exp_b == 0) {
        // Zero or subnormal (e_real = -6, no implicit leading 1).
        const float v = (float) mant * (1.0f / 64.0f) * (1.0f / 64.0f); // mant * 2^-6 * 2^-3 = mant/4096
        return sign ? -v : v;
    }
    if (exp_b == 15 && mant == 7) {
        return __builtin_nanf("");
    }
    const int e_real = (int) exp_b - 7;
    const float frac = 1.0f + (float) mant * (1.0f / 8.0f);
    float v;
    // scalbnf is fp32-clean for our exponent range (e_real ∈ [-6, 8]).
    v = frac * exp2f((float) e_real);
    return sign ? -v : v;
}

// Templated GEMV kernel — sweepable on (BN, K_COOP, USE_LDS_A, LAYOUT).
//   LAYOUT=0: cross_warp index, LDS reduce.
//   LAYOUT=1: within_warp index, __shfl_xor reduce.
template <int BN, int K_COOP, bool USE_LDS_A, int LAYOUT>
static __global__ void ml8_gemv_tpl(
    const float   * __restrict__ a,        // [K]
    const uint8_t * __restrict__ b_pack,   // [K/2, N]
    const float   * __restrict__ b_scale,  // [n_groups_k, N]
    const uint8_t * __restrict__ lut,      // [n_groups_k, 16] fp8 e4m3
    float         * __restrict__ c,        // [N]
    int K, int N, int n_groups_k) {

    constexpr int TPB = BN * K_COOP;
    const int tid = threadIdx.x;

    int n_local, k_part;
    if (LAYOUT == 0) {
        n_local = tid % BN;
        k_part  = tid / BN;
    } else {
        n_local = tid / K_COOP;
        k_part  = tid % K_COOP;
    }
    const int n_base = blockIdx.x * BN;
    const int n      = n_base + n_local;

    const int groups_per_thread = n_groups_k / K_COOP;
    const int g_start = k_part * groups_per_thread;
    const int g_end   = g_start + groups_per_thread;

    // Optional LDS cache for activations.
    extern __shared__ float s_mem[];
    float * s_a = USE_LDS_A ? s_mem : nullptr;
    if (USE_LDS_A) {
        for (int kk = tid; kk < K; kk += TPB) {
            s_a[kk] = a[kk];
        }
        __syncthreads();
    }

    float acc = 0.0f;
    if (n < N) {
        for (int g = g_start; g < g_end; g++) {
            const float scale_gn = b_scale[g * N + n];
            const uint8_t * lut_g = lut + g * 16;
            const int k_base = g * 64;
            float group_acc = 0.0f;
            #pragma unroll
            for (int p = 0; p < 32; p++) {
                const int k = k_base + p * 2;
                const uint8_t byte = b_pack[(k / 2) * N + n];
                const uint8_t lo_idx = byte & 0x0F;
                const uint8_t hi_idx = (byte >> 4) & 0x0F;
                const float c_lo = ml8_fp8_e4m3_to_fp32(lut_g[lo_idx]);
                const float c_hi = ml8_fp8_e4m3_to_fp32(lut_g[hi_idx]);
                const float a_lo = USE_LDS_A ? s_a[k]     : a[k];
                const float a_hi = USE_LDS_A ? s_a[k + 1] : a[k + 1];
                group_acc += a_lo * c_lo;
                group_acc += a_hi * c_hi;
            }
            acc += group_acc * scale_gn;
        }
    }

    if (K_COOP == 1) {
        if (n < N) c[n] = acc;
        return;
    }

    if (LAYOUT == 1) {
        // Within-warp reduce via __shfl_xor across K_COOP lanes (lane stride 1).
        #pragma unroll
        for (int off = K_COOP / 2; off > 0; off >>= 1) {
            acc += __shfl_xor(acc, off, K_COOP);
        }
        if (k_part == 0 && n < N) c[n] = acc;
    } else {
        // Cross-warp reduce via LDS.
        // Reuse s_mem when USE_LDS_A is false; otherwise allocate after s_a.
        extern __shared__ float s_mem2[];
        float * s_partial = USE_LDS_A ? (s_mem2 + K) : s_mem2;
        s_partial[n_local * K_COOP + k_part] = acc;
        __syncthreads();
        if (k_part == 0 && n < N) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < K_COOP; i++) {
                sum += s_partial[n_local * K_COOP + i];
            }
            c[n] = sum;
        }
    }
}

// Dispatch helper: returns true if a matching template was launched.
// We enumerate a curated set of (BN, K_COOP, USE_LDS_A, LAYOUT) tuples;
// the sweep harness sets these via env vars and we route accordingly.
#define ML8_GEMV_DISPATCH(BN, KC, LDS, LAYOUT)                                          \
    if (bn_v == (BN) && kc_v == (KC) && lds_v == (LDS) && layout_v == (LAYOUT)) {       \
        constexpr int TPB = (BN) * (KC);                                                \
        const size_t shmem = (size_t) ((LDS) ? K : 0) * sizeof(float)                   \
                           + (size_t) ((KC) > 1 && (LAYOUT) == 0 ? (BN) * (KC) : 0)     \
                             * sizeof(float);                                           \
        ml8_gemv_tpl<(BN),(KC),(LDS),(LAYOUT)><<<                                       \
            dim3((N + (BN) - 1) / (BN)), dim3(TPB), shmem, stream>>>(                   \
                a, b_pack, b_scale, lut, c, K, N, n_groups_k);                          \
        return true;                                                                    \
    }

static bool ml8_gemv_dispatch_env(
    cudaStream_t stream, const float * a, const uint8_t * b_pack,
    const float * b_scale, const uint8_t * lut, float * c,
    int K, int N, int n_groups_k) {

    auto env_int = [](const char * name, int def) {
        const char * s = std::getenv(name);
        if (!s) return def;
        return std::atoi(s);
    };
    // G.6.h M1 sweep winner (2026-05-26): BN=16, K_COOP=8, LDS=0, LAYOUT=0
    // → 30.66 t/s decode on Qwen3.5-4B Cell E. Env vars override for
    // continued M2/M3 experimentation (vector loads, fp8 intrinsics, etc.).
    const int bn_v     = env_int("ML8_GEMV_BN", 16);
    const int kc_v     = env_int("ML8_GEMV_K_COOP", 8);
    const int lds_v    = env_int("ML8_GEMV_LDS_A", 0);
    const int layout_v = env_int("ML8_GEMV_LAYOUT", 0);

    // BN ∈ {8,16,32,64,128} × K_COOP ∈ {1,2,4,8} × LDS_A ∈ {0,1} × LAYOUT ∈ {0,1}
    // Pruned: K_COOP=1 ignores LAYOUT (use LAYOUT=0); BN*K_COOP must be ≤ 1024.

    // K_COOP=1 family (no reduction; LAYOUT irrelevant — pass 0).
    ML8_GEMV_DISPATCH(  8, 1, 0, 0); ML8_GEMV_DISPATCH(  8, 1, 1, 0);
    ML8_GEMV_DISPATCH( 16, 1, 0, 0); ML8_GEMV_DISPATCH( 16, 1, 1, 0);
    ML8_GEMV_DISPATCH( 32, 1, 0, 0); ML8_GEMV_DISPATCH( 32, 1, 1, 0);
    ML8_GEMV_DISPATCH( 64, 1, 0, 0); ML8_GEMV_DISPATCH( 64, 1, 1, 0);
    ML8_GEMV_DISPATCH(128, 1, 0, 0); ML8_GEMV_DISPATCH(128, 1, 1, 0);

    // K_COOP=2
    ML8_GEMV_DISPATCH(  8, 2, 0, 0); ML8_GEMV_DISPATCH(  8, 2, 0, 1);
    ML8_GEMV_DISPATCH(  8, 2, 1, 0); ML8_GEMV_DISPATCH(  8, 2, 1, 1);
    ML8_GEMV_DISPATCH( 16, 2, 0, 0); ML8_GEMV_DISPATCH( 16, 2, 0, 1);
    ML8_GEMV_DISPATCH( 16, 2, 1, 0); ML8_GEMV_DISPATCH( 16, 2, 1, 1);
    ML8_GEMV_DISPATCH( 32, 2, 0, 0); ML8_GEMV_DISPATCH( 32, 2, 0, 1);
    ML8_GEMV_DISPATCH( 32, 2, 1, 0); ML8_GEMV_DISPATCH( 32, 2, 1, 1);
    ML8_GEMV_DISPATCH( 64, 2, 0, 0); ML8_GEMV_DISPATCH( 64, 2, 0, 1);
    ML8_GEMV_DISPATCH( 64, 2, 1, 0); ML8_GEMV_DISPATCH( 64, 2, 1, 1);
    ML8_GEMV_DISPATCH(128, 2, 0, 0); ML8_GEMV_DISPATCH(128, 2, 0, 1);
    ML8_GEMV_DISPATCH(128, 2, 1, 0); ML8_GEMV_DISPATCH(128, 2, 1, 1);

    // K_COOP=4
    ML8_GEMV_DISPATCH(  8, 4, 0, 0); ML8_GEMV_DISPATCH(  8, 4, 0, 1);
    ML8_GEMV_DISPATCH(  8, 4, 1, 0); ML8_GEMV_DISPATCH(  8, 4, 1, 1);
    ML8_GEMV_DISPATCH( 16, 4, 0, 0); ML8_GEMV_DISPATCH( 16, 4, 0, 1);
    ML8_GEMV_DISPATCH( 16, 4, 1, 0); ML8_GEMV_DISPATCH( 16, 4, 1, 1);
    ML8_GEMV_DISPATCH( 32, 4, 0, 0); ML8_GEMV_DISPATCH( 32, 4, 0, 1);
    ML8_GEMV_DISPATCH( 32, 4, 1, 0); ML8_GEMV_DISPATCH( 32, 4, 1, 1);
    ML8_GEMV_DISPATCH( 64, 4, 0, 0); ML8_GEMV_DISPATCH( 64, 4, 0, 1);
    ML8_GEMV_DISPATCH( 64, 4, 1, 0); ML8_GEMV_DISPATCH( 64, 4, 1, 1);
    ML8_GEMV_DISPATCH(128, 4, 0, 0); ML8_GEMV_DISPATCH(128, 4, 0, 1);
    ML8_GEMV_DISPATCH(128, 4, 1, 0); ML8_GEMV_DISPATCH(128, 4, 1, 1);

    // K_COOP=8
    ML8_GEMV_DISPATCH(  8, 8, 0, 0); ML8_GEMV_DISPATCH(  8, 8, 0, 1);
    ML8_GEMV_DISPATCH(  8, 8, 1, 0); ML8_GEMV_DISPATCH(  8, 8, 1, 1);
    ML8_GEMV_DISPATCH( 16, 8, 0, 0); ML8_GEMV_DISPATCH( 16, 8, 0, 1);
    ML8_GEMV_DISPATCH( 16, 8, 1, 0); ML8_GEMV_DISPATCH( 16, 8, 1, 1);
    ML8_GEMV_DISPATCH( 32, 8, 0, 0); ML8_GEMV_DISPATCH( 32, 8, 0, 1);
    ML8_GEMV_DISPATCH( 32, 8, 1, 0); ML8_GEMV_DISPATCH( 32, 8, 1, 1);
    ML8_GEMV_DISPATCH( 64, 8, 0, 0); ML8_GEMV_DISPATCH( 64, 8, 0, 1);
    ML8_GEMV_DISPATCH( 64, 8, 1, 0); ML8_GEMV_DISPATCH( 64, 8, 1, 1);
    // BN=128, K_COOP=8 = 1024 threads — at block max but legal.
    ML8_GEMV_DISPATCH(128, 8, 0, 0); ML8_GEMV_DISPATCH(128, 8, 0, 1);
    ML8_GEMV_DISPATCH(128, 8, 1, 0); ML8_GEMV_DISPATCH(128, 8, 1, 1);

    std::fprintf(stderr, "[ml8-gemv] no template matches BN=%d K_COOP=%d LDS_A=%d LAYOUT=%d\n",
                 bn_v, kc_v, lds_v, layout_v);
    return false;
}

// (dispatch lives in ml8_gemv_dispatch_env above)

// blockDim.x is fixed at ML8_ACT_QUANT_TPB. We assume K ≥ 1 (caller
// asserts K > 0) but allow K not divisible by TPB — guarded by stride
// loop.
static constexpr int ML8_ACT_QUANT_TPB = 256;

static __global__ void ml8_quantize_activations_kernel(
    const float * __restrict__ src,        // [M, K] row-major
    uint8_t     * __restrict__ a_fp8,      // [M, K] row-major
    float       * __restrict__ a_scale,    // [M]
    int K) {

    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    const float   * row_in  = src   + (size_t) m * (size_t) K;
    uint8_t       * row_out = a_fp8 + (size_t) m * (size_t) K;

    // Stage 1: per-thread local absmax across the row.
    float local_max = 0.0f;
    for (int k = tid; k < K; k += ML8_ACT_QUANT_TPB) {
        const float v = fabsf(row_in[k]);
        local_max = fmaxf(local_max, v);
    }

    // Stage 2: block reduction via shared memory.
    __shared__ float s_red[ML8_ACT_QUANT_TPB];
    s_red[tid] = local_max;
    __syncthreads();
    #pragma unroll
    for (int off = ML8_ACT_QUANT_TPB / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        }
        __syncthreads();
    }
    const float row_absmax = s_red[0];

    // Scale: absmax / 448, floored to epsilon so dividing zero-rows
    // doesn't blow up. Thread 0 writes; everyone uses the same value.
    const float scale     = fmaxf(row_absmax * (1.0f / ML8_FP8_E4M3_MAX), ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    if (tid == 0) {
        a_scale[m] = scale;
    }

    // Stage 3: quantize.
    for (int k = tid; k < K; k += ML8_ACT_QUANT_TPB) {
        row_out[k] = ml8_fp32_to_e4m3(row_in[k] * inv_scale);
    }
}

void ggml_cuda_ml8_quantize_activations(
    cudaStream_t  stream,
    const float * src_fp32,
    void *        dst_a_fp8,
    float *       dst_a_scale,
    int32_t       M,
    int32_t       K) {

    GGML_ASSERT(M > 0);
    GGML_ASSERT(K > 0);
    GGML_ASSERT(src_fp32   != nullptr);
    GGML_ASSERT(dst_a_fp8  != nullptr);
    GGML_ASSERT(dst_a_scale != nullptr);

    const dim3 grid((unsigned) M, 1, 1);
    const dim3 block(ML8_ACT_QUANT_TPB, 1, 1);

    ml8_quantize_activations_kernel<<<grid, block, 0, stream>>>(
        src_fp32,
        (uint8_t *) dst_a_fp8,
        dst_a_scale,
        K);
}

// ─────────────────────────────────────────────────────────────────────
// GGML_OP_ML8_MUL_MAT HIP dispatch.
// ─────────────────────────────────────────────────────────────────────

void ggml_cuda_op_ml8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {

    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * cent = dst->src[1];
    const ggml_tensor * x    = dst->src[2];

    GGML_ASSERT(w    != nullptr && cent != nullptr && x != nullptr);
    GGML_ASSERT(w->type    == GGML_TYPE_ML8_4);
    GGML_ASSERT(cent->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(cent));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    const int32_t M = (int32_t) x->ne[1];

    GGML_ASSERT(x->ne[0]   == K);
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT(dst->ne[1] == M);
    GGML_ASSERT(K % QK_ML8         == 0);
    GGML_ASSERT(N % MT_ML8_BLOCK_SIZE_N == 0);

    const int32_t group_size  = QK_ML8;
    const int32_t n_groups_k  = K / group_size;
    const int32_t n_centroids = 16;
    GGML_ASSERT(cent->ne[0] == n_centroids);
    GGML_ASSERT(cent->ne[1] == n_groups_k);

    cudaStream_t stream = ctx.stream();

    // ── 1. Repack weights (cached after first call for this w).
    const ml8_weight_repack_t * repack = ggml_cuda_ml8_get_or_repack(stream, w);
    GGML_ASSERT(repack != nullptr);

    // ── 1b. M=1 GEMV path (G.6.h). Default ON after sweep landed winner
    // BN=16, K_COOP=8, LDS=0, LAYOUT=0 → 30.66 t/s decode (1.51× Triton M=16
    // path at 20.30 t/s, 60% of f16 reference 50.89 t/s). Set ML8_NO_GEMV=1
    // to disable and fall back to the Triton blockscale path (kept for A/B).
    static const bool ml8_no_gemv = (std::getenv("ML8_NO_GEMV") != nullptr);
    if (M == 1 && !ml8_no_gemv) {
        const bool ok = ml8_gemv_dispatch_env(
            stream,
            (const float *)   x->data,
            (const uint8_t *) repack->b_packed,
            (const float *)   repack->b_scale,
            (const uint8_t *) cent->data,
            (float *)         dst->data,
            K, N, n_groups_k);
        if (ok) return;
        // fall through to Triton path if dispatch missed
    }

    // ── 2. Pad M to a multiple of the tuned tier's BLOCK_SIZE_M.
    // Pick the same config the dispatch will pick (decode for M<=16, prefill
    // otherwise) so M_pad % cfg.bm == 0 after padding. Pre-paged paths
    // (M = 1..16) align to 16; prefill (M > 16) aligns to 128.
    const mt_ml8_tuned_cfg pad_cfg = ml8_pick_config(M, K, N);
    const int32_t M_pad = ((M + pad_cfg.bm - 1) / pad_cfg.bm) * pad_cfg.bm;

    // Padded fp32 activations (M_pad × K). Zero-pad the trailing rows
    // so they produce zero output — the dst slice ignores them anyway.
    ggml_cuda_pool_alloc<float> x_padded(ctx.pool());
    const float * x_src;
    if (M_pad == M) {
        x_src = (const float *) x->data;
    } else {
        x_padded.alloc((size_t) M_pad * (size_t) K);
        CUDA_CHECK(cudaMemsetAsync(x_padded.get(), 0,
            (size_t) M_pad * (size_t) K * sizeof(float), stream));
        CUDA_CHECK(cudaMemcpyAsync(x_padded.get(), x->data,
            (size_t) M * (size_t) K * sizeof(float),
            cudaMemcpyDeviceToDevice, stream));
        x_src = x_padded.get();
    }

    // ── 3. Quantize fp32 → fp8 + per-row scale.
    ggml_cuda_pool_alloc<uint8_t> a_fp8(ctx.pool(),    (size_t) M_pad * (size_t) K);
    ggml_cuda_pool_alloc<float>   a_scale(ctx.pool(), (size_t) M_pad);

    // G.6.g.C: dump pre-quant fp32 activation that the kernel will see.
    if (ml8_dump_enabled() && !g_ml8_dump_quant_done.load()) {
        const int64_t shp[2] = { (int64_t) K, (int64_t) M_pad };
        ml8_dump_fp32("/tmp/ml8_hip_x_prequant.bin", x_src,
                      (size_t) M_pad * (size_t) K, stream, 2, shp);
    }

    ggml_cuda_ml8_quantize_activations(
        stream,
        x_src,
        a_fp8.get(),
        a_scale.get(),
        M_pad,
        K);

    // G.6.g.C: dump fp8 quantized activations + per-row scale on first call.
    if (ml8_dump_enabled() && !g_ml8_dump_quant_done.exchange(true)) {
        const int64_t shp_fp8[2]   = { (int64_t) K,     (int64_t) M_pad };
        const int64_t shp_scale[1] = { (int64_t) M_pad };
        ml8_dump_u8("/tmp/ml8_hip_a_fp8.bin", a_fp8.get(),
                    (size_t) M_pad * (size_t) K, stream, 2, shp_fp8);
        ml8_dump_fp32("/tmp/ml8_hip_a_scale.bin", a_scale.get(),
                      (size_t) M_pad, stream, 1, shp_scale);
    }

    // ── 4. Allocate bf16 output (M_pad × N) and launch mt_ml8_gemm.
    ggml_cuda_pool_alloc<nv_bfloat16> c_bf16(ctx.pool(), (size_t) M_pad * (size_t) N);

    mt_ml8_gemm_args_t args{};
    args.shape.N           = N;
    args.shape.K           = K;
    args.shape.group_size  = group_size;
    args.shape.n_centroids = n_centroids;

    args.a_fp8             = a_fp8.get();
    args.b_packed          = repack->b_packed;
    args.c                 = c_bf16.get();

    args.a_scale_fp32      = a_scale.get();
    args.b_scale_fp32      = repack->b_scale;
    args.centroid_lut_fp8  = cent->data;

    args.M                 = M_pad;

    args.stride_am         = K;  args.stride_ak       = 1;
    args.stride_bk         = N;  args.stride_bn       = 1;
    args.stride_cm         = N;  args.stride_cn       = 1;
    args.stride_ascale_m   = 1;
    args.stride_bscale_k   = N;  args.stride_bscale_n = 1;
    args.stride_lut_k      = n_centroids;

    const hipError_t gemm_rc = mt_ml8_gemm(stream, &args);
    GGML_ASSERT(gemm_rc == hipSuccess && "mt_ml8_gemm dispatch failed");

    // ── 5. Convert first M rows of bf16 [M_pad, N] → fp32 [M, N] into dst.
    // Row-major layout means the first M*N bf16 elements correspond
    // exactly to the first M output rows; the trailing (M_pad - M)*N
    // bf16 elements are the padded rows we discard.
    const to_fp32_cuda_t bf16_to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
    GGML_ASSERT(bf16_to_fp32 != nullptr);
    bf16_to_fp32(c_bf16.get(), (float *) dst->data,
                 (size_t) M * (size_t) N, stream);

    // G.6.g.C: dump final mul_mat output on first call.
    if (ml8_dump_enabled() && !g_ml8_dump_mm_done.exchange(true)) {
        const int64_t shape[2] = { (int64_t) N, (int64_t) M };
        ml8_dump_fp32("/tmp/ml8_hip_y_out.bin", (const float *) dst->data,
                      (size_t) M * (size_t) N, stream, 2, shape);
    }
}

// ─────────────────────────────────────────────────────────────────────
// GGML_OP_ML8_APPLY_ROTATION HIP dispatch.
// G.4.g: original naïve O(b²) dense matmul (46 ms/call, 92% of GPU time).
// G.6.f: rewritten as row-wise FWHT + small H_a^T multiply (~100x less
//   compute on the H_b leg). H_b is the Sylvester orthogonal Hadamard, so
//   X @ H_b == row-wise FWHT(X) normalized by 1/sqrt(b_dim) — exactly what
//   mt_turbo_fp8_fwht (turbo_fp8_hadamard.cuh) produces.
// ─────────────────────────────────────────────────────────────────────

// One block per token, blockDim.x = b_dim. Each thread l computes the
// `a_dim` outputs in column l: Y[token][k][l] = sum_i H_a[i,k] * Z[token][i][l].
// a_dim is small (5 for gate/up, 9 for down) — fits in registers.
static __global__ void ml8_h_a_left_multiply_kernel(
    const float * __restrict__ z,     // [n_tokens, a_dim, b_dim] row-major (post-FWHT)
    const float * __restrict__ h_a,   // [a_dim, a_dim] row-major
    float       * __restrict__ y,     // [n_tokens, a_dim, b_dim] row-major
    int a_dim,
    int b_dim) {
    const int t = blockIdx.x;
    const int l = threadIdx.x;
    if (l >= b_dim) return;

    const size_t token_offset = (size_t) t * a_dim * b_dim;
    const float * zt = z + token_offset;
    float       * yt = y + token_offset;

    // Load z[t][i][l] for all i into registers. a_dim ≤ 16 in practice
    // (gate/up=5, down=9 for Qwen3.5-4B; bound is generous).
    float z_col[16];
    for (int i = 0; i < a_dim; i++) {
        z_col[i] = zt[i * b_dim + l];
    }

    // Y[t][k][l] = sum_i H_a[i, k] * Z[t][i][l]
    for (int k = 0; k < a_dim; k++) {
        float s = 0.0f;
        for (int i = 0; i < a_dim; i++) {
            s += h_a[i * a_dim + k] * z_col[i];
        }
        yt[k * b_dim + l] = s;
    }
}

void ggml_cuda_op_ml8_apply_rotation(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {

    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * h_a = dst->src[1];

    GGML_ASSERT(x   != nullptr && h_a != nullptr);
    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(h_a->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(h_a));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * pp    = (const int32_t *) dst->op_params;
    const int32_t   a_dim = pp[0];
    const int32_t   b_dim = pp[1];
    const int32_t   d_dim = a_dim * b_dim;

    GGML_ASSERT(a_dim > 0 && a_dim <= 16 && "a_dim must fit in z_col register array");
    GGML_ASSERT(b_dim > 0 && (b_dim & (b_dim - 1)) == 0 && "b_dim must be power of 2");
    GGML_ASSERT(b_dim >= 16 && b_dim <= 1024 && "b_dim must be supported by FWHT kernel (16..1024)");
    GGML_ASSERT(x->ne[0]   == d_dim);
    GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);
    GGML_ASSERT(dst->ne[0] == d_dim && dst->ne[1] == x->ne[1]);

    cudaStream_t stream = ctx.stream();
    const int n_tokens = (int) x->ne[1];
    const size_t total_elems = (size_t) n_tokens * (size_t) d_dim;

    // G.6.g.C: dump rotation input (pre-rotation activations) on first call.
    if (ml8_dump_enabled() && !g_ml8_dump_rot_done.exchange(true)) {
        const int64_t shape[2] = { (int64_t) d_dim, (int64_t) n_tokens };
        ml8_dump_fp32("/tmp/ml8_hip_x_in.bin", (const float *) x->data,
                      total_elems, stream, 2, shape);
    }

    // (rotation kernel runs below; output dump happens after the kernel returns)

    // Step 1: copy X into a scratch Z buffer (FWHT is in-place).
    ggml_cuda_pool_alloc<float> z_buf(ctx.pool(), total_elems);
    CUDA_CHECK(cudaMemcpyAsync(z_buf.get(), x->data,
        total_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));

    // Step 2: row-wise FWHT on Z. Each (token, i) slice of length b_dim
    // becomes (X @ H_b)[token][i] (orthogonal Hadamard, normalized).
    CUDA_CHECK(mt_turbo_fp8_fwht(stream, z_buf.get(),
        n_tokens * a_dim, b_dim, b_dim));

    // Step 3: small left-multiply Y = H_a^T @ Z per token.
    const dim3 grid((unsigned) n_tokens, 1, 1);
    const dim3 block((unsigned) b_dim,   1, 1);
    ml8_h_a_left_multiply_kernel<<<grid, block, 0, stream>>>(
        z_buf.get(),
        (const float *) h_a->data,
        (float *) dst->data,
        a_dim,
        b_dim);

    // G.6.g.C: dump rotation output (post-FWHT + H_a^T) on first call.
    if (ml8_dump_enabled() && !g_ml8_dump_rotdst_done.exchange(true)) {
        const int64_t shape[2] = { (int64_t) d_dim, (int64_t) n_tokens };
        ml8_dump_fp32("/tmp/ml8_hip_x_rotated.bin", (const float *) dst->data,
                      total_elems, stream, 2, shape);
    }
}
