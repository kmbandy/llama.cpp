// ml8.cu — GGML_TYPE_ML8_4 on-device repack for the HIP backend.
// See ml8.cuh for the contract and motivation. MAD-223 Phase G.4.d.

#include "ml8.cuh"

#define GGML_COMMON_DECL_CUDA
#include "ggml-common.h"

#include "ggml.h"
#include "ggml-ml8.h"    // FP8_B128 phase 2: GGML_FP8_QUANT_ROT_KIND_* constants
#include "common.cuh"
#include "convert.cuh"
#include "dequantize.cuh"
#ifdef GGML_HIP_AITER
// The ml8 GEMM dispatch goes through the AITER Triton-AOT kernels. Their headers
// only live on the include path when ggml-hip is configured with -DGGML_HIP_AITER=ON
// (see ggml-hip/CMakeLists.txt). Gate the includes so ml8.cu compiles on any build
// WITHOUT that toolchain — ml8 inference is then unavailable (calibration-only /
// cross-arch builds), but the rest of ggml-hip (repack, rotation, the pager) builds.
#include "mt_ml8_gemm.h"
#include "mt_ml8_moe_gemm.h"       // G.7: ml8 MoE GEMM Triton wrapper
#include "mt_fp8_b128_gemm.h"      // FP8_B128 phase 2: AITER preshuffle GEMM
#endif // GGML_HIP_AITER
// MAD-305 Phase 5: hand-written gfx1201 WMMA "trfeed" block-scale GEMM for
// GGML_TYPE_ML8_FP8 (production integration). Pure HIP, no Triton AOT
// dependency — included unconditionally (unlike the AITER wrapper headers
// above) because the RDNA4 packed-weight LAYOUT is chosen at load time
// (ggml_cuda_ml8_inplace_set, below), which runs regardless of whether
// ggml-hip was configured with -DGGML_HIP_AITER=ON. Only the GEMM COMPUTE
// dispatch (ggml_cuda_op_fp8_mul_mat) is AITER-gated, matching FP8_B128.
#include "aiter-integration/rdna4_fp8_gemm/gemm_capi.h"
// ML8_4 RDNA4_TRFEED packed-layout addressing (ml84_trfeed_nk_to_pos /
// ml84_get_nibble / ml84_set_nibble) -- shared, byte-for-byte, with
// rdna4_pack_ml84_trfeed's own packer kernel (gemm_ml84_prod.hip) so the
// unpack path below (ml84_trfeed_unpack_kernel) is provably the same
// bijection, not an independently-derived inverse. Pure host+device byte
// arithmetic (no gfx12 intrinsics), so this include is unconditional like
// gemm_capi.h above. NOTE: this header's own `#include "../../../ggml-common.h"`
// only resolves once ggml-hip/CMakeLists.txt adds the rdna4_fp8_gemm directory
// itself to the include path (added alongside gemm_ml84_prod.hip; see that
// CMakeLists.txt comment) -- the standalone bench/build.sh has always passed
// this same directory via -I for exactly that reason.
#include "aiter-integration/rdna4_fp8_gemm/bench/ml84_trfeed_layout.h"
#include "turbo_fp8_hadamard.cuh"  // G.6.f: FWHT for rotation H_b leg

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>
#include <algorithm>

// G.6.g.C: debug hooks to dump rotation input + ml8_mul_mat output to /tmp
// for Python-side bit-equivalence comparison. Set env var ML8_DUMP=1 to
// enable. First-call-only; the static atomics track which dumps have fired.
namespace {
// ML8_DUMP harness: dumps the first ML8_DUMP_N (env, default 1) rotation /
// mul_mat calls to /tmp/ml8_hip_{rot,mm}<i>_*.bin and appends one line per
// call to /tmp/ml8_hip_index.txt (node name, weight name, shape) so a
// Python check can recompute each from the GGUF's own weights.
std::atomic<int> g_ml8_dump_rot_n   {0};
std::atomic<int> g_ml8_dump_mm_n    {0};
int ml8_dump_limit() {
    static const int n = std::getenv("ML8_DUMP_N") ? std::atoi(std::getenv("ML8_DUMP_N")) : 1;
    return n;
}
void ml8_dump_index(const char * kind, int i, const char * node, const char * w, int64_t K, int64_t N, int64_t M, int a_dim, int b_dim, bool has_h_a) {
    FILE * f = std::fopen("/tmp/ml8_hip_index.txt", "a");
    if (!f) return;
    std::fprintf(f, "%s %d node=%s w=%s K=%lld N=%lld M=%lld a_dim=%d b_dim=%d h_a=%d\n", kind, i, node, w ? w : "-", (long long) K, (long long) N, (long long) M, a_dim, b_dim, (int) has_h_a);
    std::fclose(f);
}

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
// MAD-305 ML8_4 (4.5 bpw) GEMM weight LAYOUT switch (generic "triton" /
// RDNA4 "trfeed"), the ML8_4 sibling of ML8_FP8_GEMM_LAYOUT_*/FP8_B128_LAYOUT_*
// further below. Both layouts share the exact same footprint
// (ggml_cuda_ml8_inplace_alloc_size's ML8_4 formula, (K/2)*N nibbles +
// (K/64)*N fp32 scales -- rdna4_pack_ml84_trfeed's B_nib is a pure
// tile-shuffled repacking of the same nibble stream, and b_scale_g is
// byte-for-byte the SAME [K/64,N] fp32 table ggml_cuda_ml8_repack_blocks's
// TRITON layout already produces -- see gemm_capi.h's ML84_TRFEED comment):
//   TRITON (was the only layout): b_packed is the straight [K/2,N] nibble
//     transpose (ml8_repack_kernel's layout) -- read by mt_ml8_gemm's
//     WEIGHT_FORMAT=1 LUT path, the M=1 GEMV kernel (ml8_gemv_dispatch_env)
//     and the native GET_ROWS packed gather (ml8_packed_get_rows_kernel).
//   RDNA4_TRFEED (new default): b_packed is B_nib, the SAME nibble stream
//     tile-shuffled into the frozen fp8 trfeed kernel's 16(K)x16(N)
//     addressing (rdna4_pack_ml84_trfeed, gemm_capi.h) -- read by
//     rdna4_gemm_ml84_trfeed_decode_splitk (M<=32) and, via
//     rdna4_expand_ml84_to_trfeed + the UNCHANGED rdna4_gemm_fp8_trfeed,
//     for M>32 prefill. 9070 XT (2026-09-17 bench, gemm_capi.h): decode
//     0.158ms vs Triton's 0.304ms at N=17408/K=5120; prefill M=2048 132 TF
//     vs Triton's 57 TF.
// Read ONCE (static), same rationale as the sibling layout switches: a live
// env flip would desync already-packed weights from a dispatch expecting
// the other layout. Recorded per-tensor in ml8_weight_repack_t::layout.
// MT_ML8_4_LAYOUT=triton restores the old path; MT_ML8_4_LAYOUT=trfeed (or
// unset) is the default. The RDNA4_TRFEED prefill expander needs whole
// 128-wide N tiles (rdna4_expand_ml84_to_trfeed / rdna4_gemm_fp8_trfeed both
// require N%128==0); a tensor whose N isn't a multiple of 128 (e.g. a TP
// N-slice sliced to a 16-multiple) is packed TRITON regardless of the env
// choice -- see ml8_4_layout_for_tensor below, which both repack paths
// (in-place ggml_cuda_ml8_inplace_set and the cache-copy
// ggml_cuda_ml8_get_or_repack) call to decide a given tensor's layout once,
// at pack time, so ml8_mul_mat_core's dispatch can just follow
// repack->layout with no further shape checks.
// ─────────────────────────────────────────────────────────────────────
enum {
    ML8_4_LAYOUT_TRITON       = 0,
    ML8_4_LAYOUT_RDNA4_TRFEED = 1,
};

static int32_t ml8_4_env_layout() {
    static const int32_t layout = [] {
        const char * e = getenv("MT_ML8_4_LAYOUT");
        if (e != nullptr && std::strcmp(e, "triton") == 0) {
            return (int32_t) ML8_4_LAYOUT_TRITON;
        }
        if (e != nullptr && std::strcmp(e, "trfeed") != 0 && std::strlen(e) > 0) {
            fprintf(stderr, "[ml8-4] MT_ML8_4_LAYOUT=%s not recognized, using 'trfeed'\n", e);
        }
        return (int32_t) ML8_4_LAYOUT_RDNA4_TRFEED;   // default: gfx1201 trfeed decode/prefill kernels
    }();
    return layout;
}

static int32_t ml8_4_layout_for_tensor(int32_t N) {
    if (ml8_4_env_layout() == ML8_4_LAYOUT_RDNA4_TRFEED && N % 128 == 0) {
        return ML8_4_LAYOUT_RDNA4_TRFEED;
    }
    return ML8_4_LAYOUT_TRITON;
}

// Pack the on-disk ML8_4 blocks into `dst_packed`/`dst_scale` according to
// `layout`. TRITON reuses the existing straight [K/2,N] repack kernel;
// RDNA4_TRFEED calls the out-of-place device packer (rdna4_pack_ml84_trfeed,
// gemm_capi.h) directly against `staging` (the on-disk bytes) -- no
// intermediate buffer needed, unlike ML8_FP8's RDNA4 layout, since the
// packer already reads block_ml8_4 rows straight from `staging`.
static void ml8_4_pack_for_layout(
    cudaStream_t stream, const uint8_t * staging, uint8_t * dst_packed, float * dst_scale,
    int32_t N, int32_t K, int32_t n_groups_k, int32_t layout) {
    if (layout == ML8_4_LAYOUT_TRITON) {
        constexpr int BLOCK_N = 64;
        const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, n_groups_k, 1);
        const dim3 block(BLOCK_N, 1, 1);
        ml8_repack_kernel<<<grid, block, 0, stream>>>(staging, dst_packed, dst_scale, N, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const hipError_t rc = rdna4_pack_ml84_trfeed(staging, N, K, dst_packed, dst_scale, stream);
    CUDA_CHECK(cudaGetLastError());
    GGML_ASSERT(rc == hipSuccess && "rdna4_pack_ml84_trfeed failed");
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

// ML8_FP8 (scaled-fp8) repack cache. Here info.b_packed holds raw e4m3 bytes
// [K, N] (no nibbles). Keyed purely by device pointer, exactly like the ml8-4
// cache above: in production a weight's w->data is stable for the model's
// lifetime, so the pointer is a valid key and lookups are a free hash hit on
// the hot path (α/β run every token). Stale-pointer aliasing — a freed buffer's
// address reused for a different weight — is handled by the invariant that
// ggml_cuda_ml8_clear_cache() runs whenever a CUDA device buffer is freed
// (wired into ggml_backend_cuda_buffer_free_buffer), so no entry outlives the
// buffer its key points into. That also covers test-backend-ops, which frees
// and recycles device buffers across cases.
std::mutex                                       g_ml8_fp8_cache_mu;
std::unordered_map<const void *, cache_entry_t>  g_ml8_fp8_cache;

// FP8_B128 (design 4(a)) cache-copy fallback repack, mirroring g_ml8_fp8_cache
// above: used when a FP8_B128 weight isn't in-place eligible (WP_ML8_INPLACE=0
// or N not a multiple of 128). info.b_packed is the preshuffled [N,K] byte
// permutation, info.b_scale is the fp32 [K/128, N/128] scale table.
std::mutex                                       g_fp8_b128_cache_mu;
std::unordered_map<const void *, cache_entry_t>  g_fp8_b128_cache;

// MAD-305 Phase 5 (round 2): the legacy WP_ML8_FP8_LEGACY=1 plain-MUL_MAT
// path (ggml_cuda_op_ml8_fp8_mul_mat below) dispatches the mt_ml8_gemm
// Triton WEIGHT_FORMAT=0 kernel, which only understands the original
// straight-[K,N]-transpose "triton" packed layout -- but MT_ML8_FP8_GEMM now
// defaults to "rdna4" (fragment-tile shuffled), so that weight's ACTUAL
// packed bytes are usually not what the Triton kernel expects. Rather than
// abort, cache a built-once [K,N] "triton view" (an on-demand
// rdna4_unshuffle_b_ml8fp8 of the rdna4-packed bytes) keyed by the weight's
// device pointer, exactly like the repack caches above -- so the legacy path
// pays the unshuffle cost once per weight, not once per token.
std::mutex                                       g_ml8_fp8_triton_view_mu;
std::unordered_map<const void *, void *>         g_ml8_fp8_triton_view;

// In-place (load-time) ML8_FP8 repack registry, keyed by the tensor's own
// device pointer. `staging` holds the on-disk block bytes while a tensor is
// being written in pieces; once `received == nbytes` the repack kernel
// scatters staging -> info.{b_packed,b_scale} (both inside the tensor's
// allocation), staging is freed and `packed` flips to true.
struct inplace_entry_t {
    ml8_weight_repack_t info;
    ggml_type           type;       // GGML_TYPE_ML8_4 or GGML_TYPE_ML8_FP8
    size_t              nbytes;     // on-disk block bytes (ggml_nbytes)
    uint8_t *           staging;    // device, nbytes; null once packed
    size_t              received;
    bool                packed;
};
std::mutex                                        g_ml8_inplace_mu;
std::unordered_map<const void *, inplace_entry_t> g_ml8_inplace;

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

    // Load-time in-place repack: the tensor's own allocation is the kernel layout.
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(key);
        if (it != g_ml8_inplace.end() && it->second.packed) {
            return &it->second.info;
        }
    }

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

    if (ggml_cuda_wp_vram_log_enabled()) {
        size_t fb = 0, tot = 0; int dev = -1; (void) cudaGetDevice(&dev); (void) cudaMemGetInfo(&fb, &tot);
        fprintf(stderr, "wp vram-budget: ml8_repack %.1f MiB on device %d (free before %.1f MiB) K=%d N=%d\n",
                (b_packed_bytes + b_scale_bytes) / 1048576.0, dev, fb / 1048576.0, (int) K, (int) N); fflush(stderr);
    }
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

    // MAD-305: layout-aware, mirroring the in-place path (ggml_cuda_ml8_inplace_set)
    // -- test-backend-ops and any WP_ML8_INPLACE=0 build take this cache-copy
    // path, and both must produce the same packed layout the dispatch expects.
    const int32_t layout = ml8_4_layout_for_tensor(N);
    ml8_4_pack_for_layout(stream, (const uint8_t *) w->data, (uint8_t *) d_b_packed, d_b_scale,
        N, K, n_groups_k, layout);

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
    entry.info.layout     = layout;
    auto [ins_it, _ins_ok] = g_ml8_cache.emplace(key, entry);
    return &ins_it->second.info;
}

void ggml_cuda_ml8_clear_cache(void) {
    {
        std::lock_guard<std::mutex> lock(g_ml8_cache_mu);
        for (auto & kv : g_ml8_cache) {
            cudaFree(kv.second.info.b_packed);
            cudaFree(kv.second.info.b_scale);
        }
        g_ml8_cache.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_ml8_fp8_cache_mu);
        for (auto & kv : g_ml8_fp8_cache) {
            cudaFree(kv.second.info.b_packed);
            cudaFree(kv.second.info.b_scale);
        }
        g_ml8_fp8_cache.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_fp8_b128_cache_mu);
        for (auto & kv : g_fp8_b128_cache) {
            cudaFree(kv.second.info.b_packed);
            cudaFree(kv.second.info.b_scale);
        }
        g_fp8_b128_cache.clear();
    }
    {
        std::lock_guard<std::mutex> lock(g_ml8_fp8_triton_view_mu);
        for (auto & kv : g_ml8_fp8_triton_view) {
            cudaFree(kv.second);
        }
        g_ml8_fp8_triton_view.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────
// ML8_FP8 (scaled-fp8) repack — no LUT.
//
// On-disk per-block layout: 2-byte fp16 scale, then QK_ML8_FP8 = 32 raw
// OCP e4m3fn weight bytes covering 32 K-elements. sizeof(block_ml8_fp8)==34.
// Rows are [N, K] laid out as per-row sequences of n_groups_k blocks.
//
// The WF=0 Triton path wants B as raw e4m3 [K, N] (transposed, same dtype
// as A) plus a fp16 per-(K-group, N) scale [n_groups_k, N] — copied through
// verbatim from the on-disk fp16 scale (no widen/narrow round-trip), so the
// packed layout stays 8.5 bpw. So this repack is the FP8 sibling of
// ml8_repack_kernel: copy the e4m3 byte straight through (no 4-bit unpack,
// no centroid) and the fp16 group scale straight through (no widen).
// group_size is QK_ML8_FP8 = 32.
// ─────────────────────────────────────────────────────────────────────
static constexpr int ML8_FP8_BLOCK_BYTES = (int) sizeof(block_ml8_fp8);  // 34

// One thread per (n, g) pair. Reads the (2-byte fp16 scale + 32 e4m3 bytes)
// block from the on-disk row-major (N, n_groups_k * 34) layout and scatters
// into the separated (b_fp8[K, N], b_scale[n_groups_k, N]) layout.
static __global__ void ml8_fp8_repack_kernel(
    const uint8_t * __restrict__ src,        // (N, n_groups_k * 34) bytes
    uint8_t       * __restrict__ b_fp8,      // (K, N) row-major raw e4m3
    __half        * __restrict__ b_scale,    // (n_groups_k, N) row-major, fp16
    int N,
    int n_groups_k) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (n >= N || g >= n_groups_k) {
        return;
    }

    const uint8_t * blk = src
        + (size_t) n * (size_t) n_groups_k * (size_t) ML8_FP8_BLOCK_BYTES
        + (size_t) g * (size_t) ML8_FP8_BLOCK_BYTES;

    // Scale: 2-byte fp16 at the start of the block, copied through verbatim
    // (no widen to fp32 — the packed layout keeps fp16 scales at 8.5 bpw).
    __half scale_h;
    memcpy(&scale_h, blk, sizeof(__half));
    b_scale[(size_t) g * (size_t) N + (size_t) n] = scale_h;

    // Weights: 32 raw e4m3 bytes after the scale, covering K-rows
    // [g * QK_ML8_FP8, (g + 1) * QK_ML8_FP8). Copied straight through.
    const uint8_t * qs     = blk + sizeof(uint16_t);
    const int       k_base = g * QK_ML8_FP8;
    #pragma unroll
    for (int j = 0; j < QK_ML8_FP8; ++j) {
        b_fp8[((size_t) (k_base + j)) * (size_t) N + (size_t) n] = qs[j];
    }
}

// ─────────────────────────────────────────────────────────────────────
// MAD-305 Phase 5 — ML8_FP8 GEMM weight LAYOUT switch (generic "triton" /
// RDNA4 "trfeed"), the ML8_FP8 sibling of the FP8_B128_LAYOUT_* switch further
// below. Both layouts share the exact same footprint
// (ggml_cuda_ml8_inplace_alloc_size's ML8_FP8 formula, K*N + (K/32)*N fp16 —
// the RDNA4 preshuffle is a byte permutation of the same K*N weight bytes,
// and the fp16 [K/32,N] scale table is IDENTICAL in both layouts, upcast to
// fp32 only inside the GEMM's per-K-tile scale fold):
//   TRITON (selectable, was the only layout before Phase 5): b_packed is the
//     raw e4m3 weight TRANSPOSED to [K,N] row-major (ml8_fp8_repack_kernel's
//     layout) — read by the mt_ml8_gemm Triton WEIGHT_FORMAT=0 kernel.
//   RDNA4 (new default): b_packed is that SAME transposed [K,N] byte matrix,
//     further pre-shuffled into `global_load_tr_b64` fragment-tile order
//     (rdna4_fp8_gemm/gemm_capi.h::rdna4_preshuffle_b_ml8fp8) — read by
//     rdna4_gemm_ml8fp8_blockscale (gemm_blockscale.hip), the hand-written
//     gfx1201 WMMA kernel that hits hipBLASLt parity (RESULT.md).
// Read ONCE (static) for the same reason as MT_FP8_B128_LAYOUT: flipping the
// env var mid-process would desync already-packed weights from a dispatch
// expecting the other layout. Recorded per-tensor in ml8_weight_repack_t::layout.
// ─────────────────────────────────────────────────────────────────────
enum {
    ML8_FP8_GEMM_LAYOUT_TRITON = 0,
    ML8_FP8_GEMM_LAYOUT_RDNA4  = 1,
};

static int32_t ml8_fp8_gemm_current_layout() {
    static const int32_t layout = [] {
        const char * e = getenv("MT_ML8_FP8_GEMM");
        if (e != nullptr && std::strcmp(e, "triton") == 0) {
            return (int32_t) ML8_FP8_GEMM_LAYOUT_TRITON;
        }
        if (e != nullptr && std::strcmp(e, "rdna4") != 0 && std::strlen(e) > 0) {
            fprintf(stderr, "[ml8-fp8] MT_ML8_FP8_GEMM=%s not recognized, using 'rdna4'\n", e);
        }
        return (int32_t) ML8_FP8_GEMM_LAYOUT_RDNA4;   // default: our own trfeed WMMA kernel
    }();
    return layout;
}

// Pack the on-disk ML8_FP8 blocks into `dst_packed`/`dst_scale` according to
// `layout`. Scale is always the straight fp16 [n_groups_k,N] copy (identical
// bytes in both layouts); the weight bytes either land directly in `dst_packed`
// (TRITON, [K,N] transpose) or are staged through a temporary [K,N] buffer and
// then shuffled into `dst_packed` (RDNA4, fragment-tile order — device-side,
// no host round trip, via rdna4_preshuffle_b_ml8fp8).
static void ml8_fp8_pack_for_layout(
    cudaStream_t stream, const uint8_t * staging, uint8_t * dst_packed, __half * dst_scale,
    int32_t N, int32_t K, int32_t n_groups_k, int32_t layout) {
    constexpr int BLOCK_N = 64;
    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, n_groups_k, 1);
    const dim3 block(BLOCK_N, 1, 1);
    if (layout == ML8_FP8_GEMM_LAYOUT_TRITON) {
        ml8_fp8_repack_kernel<<<grid, block, 0, stream>>>(staging, dst_packed, dst_scale, N, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    uint8_t * tmp_kn = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &tmp_kn, (size_t) K * (size_t) N));
    ml8_fp8_repack_kernel<<<grid, block, 0, stream>>>(staging, tmp_kn, dst_scale, N, n_groups_k);
    CUDA_CHECK(cudaGetLastError());
    const hipError_t rc = rdna4_preshuffle_b_ml8fp8(tmp_kn, dst_packed, K, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(tmp_kn);
    GGML_ASSERT(rc == hipSuccess && "rdna4_preshuffle_b_ml8fp8 failed");
}

// Returns a TRITON-layout [K,N] view of `repack`'s weight bytes, regardless
// of which layout it's actually packed in -- straight-through if it's
// already TRITON, else a cached, built-once rdna4_unshuffle_b_ml8fp8. `key`
// is the owning weight's device pointer (same key ggml_cuda_ml8_fp8_get_or_repack
// uses). Used only by the legacy WP_ML8_FP8_LEGACY=1 plain-MUL_MAT Triton
// dispatch (ggml_cuda_op_ml8_fp8_mul_mat) -- the production path
// (GGML_OP_FP8_MUL_MAT) reads whatever layout the weight is actually packed
// in directly, no view needed.
static const void * ml8_fp8_triton_view(
    cudaStream_t stream, const void * key, const ml8_weight_repack_t * repack, int32_t K, int32_t N) {
    if (repack->layout == ML8_FP8_GEMM_LAYOUT_TRITON) {
        return repack->b_packed;
    }
    {
        std::lock_guard<std::mutex> lock(g_ml8_fp8_triton_view_mu);
        auto it = g_ml8_fp8_triton_view.find(key);
        if (it != g_ml8_fp8_triton_view.end()) {
            return it->second;
        }
    }
    void * tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&tmp, (size_t) K * (size_t) N));
    const hipError_t rc = rdna4_unshuffle_b_ml8fp8(repack->b_packed, tmp, K, N, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    GGML_ASSERT(rc == hipSuccess && "rdna4_unshuffle_b_ml8fp8 (legacy triton view) failed");

    std::lock_guard<std::mutex> lock(g_ml8_fp8_triton_view_mu);
    auto it = g_ml8_fp8_triton_view.find(key);
    if (it != g_ml8_fp8_triton_view.end()) {
        cudaFree(tmp);   // another thread raced us
        return it->second;
    }
    g_ml8_fp8_triton_view.emplace(key, tmp);
    return tmp;
}

// Cache-keyed ML8_FP8 repack. Mirrors ggml_cuda_ml8_get_or_repack but for
// the scaled-fp8 weight: b_packed holds raw e4m3 bytes [K, N], b_scale holds
// fp32 [n_groups_k, N]. group_size is QK_ML8_FP8 (32).
static const ml8_weight_repack_t * ggml_cuda_ml8_fp8_get_or_repack(
    cudaStream_t        stream,
    const ggml_tensor * w) {

    if (w == nullptr || w->data == nullptr) {
        return nullptr;
    }
    if (w->type != GGML_TYPE_ML8_FP8) {
        return nullptr;
    }

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    if (K <= 0 || N <= 0 || K % QK_ML8_FP8 != 0) {
        return nullptr;
    }
    const int32_t group_size = QK_ML8_FP8;
    const int32_t n_groups_k = K / group_size;

    const void * key = w->data;

    // Load-time in-place repack: the tensor's own allocation already holds
    // the kernel layout, so there is nothing to build and nothing to cache.
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(key);
        if (it != g_ml8_inplace.end() && it->second.packed) {
            return &it->second.info;
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_ml8_fp8_cache_mu);
        auto it = g_ml8_fp8_cache.find(key);
        if (it != g_ml8_fp8_cache.end()) {
            return &it->second.info;
        }
    }

    void *   d_b_fp8   = nullptr;
    __half * d_b_scale = nullptr;

    const size_t b_fp8_bytes   = (size_t) K * (size_t) N;            // [K, N] raw e4m3
    const size_t b_scale_bytes = (size_t) n_groups_k * (size_t) N * sizeof(__half);

    if (ggml_cuda_wp_vram_log_enabled()) {
        size_t fb = 0, tot = 0; int dev = -1; (void) cudaGetDevice(&dev); (void) cudaMemGetInfo(&fb, &tot);
        fprintf(stderr, "wp vram-budget: ml8_repack_fp8 %.1f MiB on device %d (free before %.1f MiB) K=%d N=%d\n",
                (b_fp8_bytes + b_scale_bytes) / 1048576.0, dev, fb / 1048576.0, (int) K, (int) N); fflush(stderr);
    }
    cudaError_t err = cudaMalloc(&d_b_fp8, b_fp8_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8-fp8] cudaMalloc(b_fp8=%zu) failed: %s\n",
                b_fp8_bytes, cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMalloc((void **) &d_b_scale, b_scale_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8-fp8] cudaMalloc(b_scale=%zu) failed: %s\n",
                b_scale_bytes, cudaGetErrorString(err));
        cudaFree(d_b_fp8);
        return nullptr;
    }

    const int32_t layout = ml8_fp8_gemm_current_layout();
    ml8_fp8_pack_for_layout(
        stream, (const uint8_t *) w->data, (uint8_t *) d_b_fp8, d_b_scale,
        N, K, n_groups_k, layout);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8-fp8] repack kernel launch failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_b_fp8);
        cudaFree(d_b_scale);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_ml8_fp8_cache_mu);
    // Re-check in case another thread raced us. If so, free ours and
    // return the winner's.
    auto it = g_ml8_fp8_cache.find(key);
    if (it != g_ml8_fp8_cache.end()) {
        cudaFree(d_b_fp8);
        cudaFree(d_b_scale);
        return &it->second.info;
    }
    cache_entry_t entry{};
    entry.info.b_packed   = d_b_fp8;     // raw e4m3 [K, N] (TRITON) or trfeed-shuffled (RDNA4)
    entry.info.b_scale    = d_b_scale;
    entry.info.N          = N;
    entry.info.K          = K;
    entry.info.n_groups_k = n_groups_k;
    entry.info.group_size = group_size;
    entry.info.layout     = layout;
    auto [ins_it, _ins_ok] = g_ml8_fp8_cache.emplace(key, entry);
    return &ins_it->second.info;
}

// ─────────────────────────────────────────────────────────────────────
// ML8_FP8 in-place repack (see ml8.cuh).
// ─────────────────────────────────────────────────────────────────────

// Inverse of ml8_fp8_repack_kernel: gather the kernel layout back into the
// on-disk {fp16 scale, 32 e4m3} blocks. One thread per (n, g).
static __global__ void ml8_fp8_unpack_kernel(
    const uint8_t * __restrict__ b_fp8,      // (K, N) row-major raw e4m3
    const __half  * __restrict__ b_scale,    // (n_groups_k, N) row-major, fp16
    uint8_t       * __restrict__ dst,        // (N, n_groups_k * 34) bytes
    int N,
    int n_groups_k) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (n >= N || g >= n_groups_k) {
        return;
    }

    uint8_t * blk = dst
        + (size_t) n * (size_t) n_groups_k * (size_t) ML8_FP8_BLOCK_BYTES
        + (size_t) g * (size_t) ML8_FP8_BLOCK_BYTES;

    // Copied through verbatim — b_scale is already fp16, no round-trip.
    const __half scale_h = b_scale[(size_t) g * (size_t) N + (size_t) n];
    memcpy(blk, &scale_h, sizeof(__half));

    uint8_t * qs     = blk + sizeof(__half);
    const int k_base = g * QK_ML8_FP8;
    #pragma unroll
    for (int j = 0; j < QK_ML8_FP8; ++j) {
        qs[j] = b_fp8[((size_t) (k_base + j)) * (size_t) N + (size_t) n];
    }
}

// Inverse of ml8_fp8_pack_for_layout: reconstruct the on-disk block bytes
// (dst, (N, n_groups_k*34)) from a packed ML8_FP8 entry of the given layout.
// RDNA4 layout first unshuffles b_shuf back into the [K,N] transpose
// (device-side, no host round trip, via rdna4_unshuffle_b_ml8fp8) and then
// runs the same ml8_fp8_unpack_kernel the TRITON layout uses directly.
static void ml8_fp8_unpack_for_layout(
    cudaStream_t stream, const uint8_t * src_packed, const __half * src_scale, uint8_t * dst,
    int32_t N, int32_t K, int32_t n_groups_k, int32_t layout) {
    constexpr int BLOCK_N = 64;
    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, n_groups_k, 1);
    const dim3 block(BLOCK_N, 1, 1);
    if (layout == ML8_FP8_GEMM_LAYOUT_TRITON) {
        ml8_fp8_unpack_kernel<<<grid, block, 0, stream>>>(src_packed, src_scale, dst, N, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    uint8_t * tmp_kn = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &tmp_kn, (size_t) K * (size_t) N));
    const hipError_t rc = rdna4_unshuffle_b_ml8fp8(src_packed, tmp_kn, K, N, stream);
    GGML_ASSERT(rc == hipSuccess && "rdna4_unshuffle_b_ml8fp8 failed");
    ml8_fp8_unpack_kernel<<<grid, block, 0, stream>>>(tmp_kn, src_scale, dst, N, n_groups_k);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaFree(tmp_kn);
}

// Inverse of ml8_repack_kernel (ML8_4): gather nibbles + fp32 scale back into
// the on-disk block_ml8_4 {float scale; uint8_t qs[32]} blocks.
static __global__ void ml8_unpack_kernel(
    const uint8_t * __restrict__ b_packed,   // (K/2, N)
    const float   * __restrict__ b_scale,    // (n_groups_k, N)
    uint8_t       * __restrict__ dst,        // (N, n_groups_k * 36) bytes
    int N,
    int n_groups_k) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (n >= N || g >= n_groups_k) {
        return;
    }
    uint8_t * blk = dst + ((size_t) n * (size_t) n_groups_k + (size_t) g) * sizeof(block_ml8_4);
    const float scale = b_scale[(size_t) g * (size_t) N + (size_t) n];
    memcpy(blk, &scale, sizeof(float));
    uint8_t * qs = blk + sizeof(float);
    const int k_half_base = g * ML8_GROUP_NIBBLES;
    #pragma unroll
    for (int j = 0; j < ML8_GROUP_NIBBLES; ++j) {
        qs[j] = b_packed[((size_t) (k_half_base + j)) * (size_t) N + (size_t) n];
    }
}

// Inverse of ml84_pack_kernel (gemm_ml84_prod.hip): gather B_nib nibbles +
// b_scale_g back into the on-disk block_ml8_4 {float scale; uint8_t qs[32]}
// blocks. One thread per (n, g), mirroring ml8_unpack_kernel's TRITON
// inverse above but reading through ml84_trfeed_nk_to_pos's tile-shuffled
// addressing (bench/ml84_trfeed_layout.h) instead of a straight [K/2,N]
// stride.
static __global__ void ml84_trfeed_unpack_kernel(
    const uint8_t * __restrict__ B_nib,      // ML84_TRFEED nibble layout
    const float   * __restrict__ b_scale_g,  // (n_groups_k, N) row-major
    uint8_t       * __restrict__ dst,        // (N, n_groups_k * 36) bytes
    int N, int K, int n_groups_k) {

    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int g = blockIdx.y;
    if (n >= N || g >= n_groups_k) {
        return;
    }
    uint8_t * blk = dst + ((size_t) n * (size_t) n_groups_k + (size_t) g) * sizeof(block_ml8_4);
    const float scale = b_scale_g[(size_t) g * (size_t) N + (size_t) n];
    memcpy(blk, &scale, sizeof(float));
    uint8_t * qs = blk + sizeof(float);
    #pragma unroll
    for (int i = 0; i < QK_ML8 / 2; ++i) {
        const int k_lo = g * QK_ML8 + 2 * i;
        const int k_hi = k_lo + 1;
        const uint8_t lo_idx = ml84_get_nibble(B_nib, ml84_trfeed_nk_to_pos(n, k_lo, N, K));
        const uint8_t hi_idx = ml84_get_nibble(B_nib, ml84_trfeed_nk_to_pos(n, k_hi, N, K));
        qs[i] = (uint8_t) ((hi_idx << 4) | lo_idx);
    }
}

// Inverse of ml8_4_pack_for_layout: reconstruct the on-disk block bytes
// (dst, (N, n_groups_k*36)) from a packed ML8_4 entry of the given layout.
static void ml8_4_unpack_for_layout(
    cudaStream_t stream, const uint8_t * src_packed, const float * src_scale, uint8_t * dst,
    int32_t N, int32_t K, int32_t n_groups_k, int32_t layout) {
    constexpr int BLOCK_N = 64;
    const dim3 grid((N + BLOCK_N - 1) / BLOCK_N, n_groups_k, 1);
    const dim3 block(BLOCK_N, 1, 1);
    if (layout == ML8_4_LAYOUT_TRITON) {
        ml8_unpack_kernel<<<grid, block, 0, stream>>>(src_packed, src_scale, dst, N, n_groups_k);
    } else {
        ml84_trfeed_unpack_kernel<<<grid, block, 0, stream>>>(src_packed, src_scale, dst, N, K, n_groups_k);
    }
    CUDA_CHECK(cudaGetLastError());
}

// FP8_B128 phase 2: on-disk block is { ggml_half d; uint8_t qs[128]; }, 130
// bytes / 128 elems (== block_fp8_b128 in ggml-common.h). blck_size 128.
static constexpr int FP8_B128_BLOCK_SIZE  = 128;
static constexpr int FP8_B128_BLOCK_BYTES = (int) sizeof(block_fp8_b128);  // 130

// ─────────────────────────────────────────────────────────────────────
// FP8_B128 weight LAYOUT switch (generic default / preshuffle A-B).
//
// Two packed byte layouts share the same total footprint
// (ggml_cuda_ml8_inplace_alloc_size's FP8_B128 formula) and the same
// [K/128, N/128] fp32 scale table — they differ only in how the e4m3
// weight bytes [0, N*K) are permuted:
//   GENERIC:     b_packed = e4m3 weight TRANSPOSED to [K, N] row-major
//                (exactly the ML8_FP8 WEIGHT_FORMAT=0 b_packed convention —
//                see ml8_fp8_repack_kernel above); GEMM dispatch launches
//                the generic `_gemm_a8w8_blockscale_kernel` (WF=0) via
//                mt_fp8_b128_gemm_generic. Faster on gfx1201 (measured
//                4.36ms vs 5.2-9.5ms for preshuffle at K=5120 N=17408
//                M=2048) so this is the default.
//   PRESHUFFLE:  b_packed = AITER shuffle_weight(layout=(16,16)) permutation
//                (see the comment below); GEMM dispatch launches
//                `_gemm_a8w8_blockscale_preshuffle_kernel` via mt_fp8_b128_gemm.
//                Kept selectable for A/B via MT_FP8_B128_LAYOUT=preshuffle.
//
// Read ONCE (static): the layout decides the packed byte layout at load
// time, so flipping the env var mid-process would desync already-packed
// weights from a dispatch that expects the other layout. Each packed
// tensor's chosen layout is also recorded in ml8_weight_repack_t::layout
// so the GEMM dispatch and the unpack paths (get_tensor/cpy_tensor via
// ggml_cuda_ml8_inplace_get, and the GET_ROWS fallback via
// ggml_cuda_ml8_inplace_fp8_b128_unpack_to_device) always pick the kernel
// matching how that tensor was actually packed.
// ─────────────────────────────────────────────────────────────────────
enum {
    FP8_B128_LAYOUT_GENERIC    = 0,
    FP8_B128_LAYOUT_PRESHUFFLE = 1,
    // MAD-305 Phase 5 (round 2): B [K,N] transposed then pre-shuffled into
    // the same global_load_tr_b64 fragment-tile layout ML8_FP8's rdna4
    // layout uses (the shuffle is a pure e4m3-byte permutation, independent
    // of which on-disk quant format the bytes came from) + the SAME
    // [K/128,N/128] fp32 scale table as GENERIC/PRESHUFFLE. Read by the
    // hand-written gfx1201 WMMA "trfeed" kernel (rdna4_gemm_fp8b128_blockscale,
    // gemm_blockscale.hip) with a 128-wide (== FP8_B128_BLOCK_SIZE) scale
    // fold instead of ML8_FP8's 32-wide one -- RDNA shares the WMMA and VALU
    // issue port, so folding every 32 K (8 WMMA/wave) pays ~150% VALU
    // overhead; folding every 128 K (32 WMMA/wave) cuts that to ~1/6 and
    // hits hipBLASLt-parity (RESULT.md). SUPERSEDED as the default by
    // FP8_B128_LAYOUT_RDNA4_TRFEED below (round 3): its consumer
    // (rdna4_gemm_fp8b128_blockscale, gemm_blockscale.hip) is abandoned --
    // every in-loop scale-fold width tried (32, 128) spilled/VALU-starved to
    // 7-35 TF. Kept selectable (MT_FP8_B128_LAYOUT=rdna4_tile) for A/B only;
    // no default-path code reads it anymore.
    FP8_B128_LAYOUT_RDNA4      = 2,
    // MAD-305 Phase 5 (round 3, DEFAULT) — same B [K,N] transpose + the SAME
    // global_load_tr_b64 fragment-tile preshuffle as FP8_B128_LAYOUT_RDNA4
    // above (rdna4_preshuffle_b_ml8fp8 is reused unchanged for the weight
    // bytes), but the scale table is now fp32 b_scale[N] -- ONE scalar per
    // output row n, not a [K/128,N/128] tile table -- because the frozen
    // Phase-1 "trfeed" kernel (rdna4_gemm_fp8_trfeed, gemm_capi.h) applies
    // a_scale[m] * b_scale[n] ONLY in the epilogue, after the full K
    // reduction (no in-loop fold at all). The converter guarantees the
    // on-disk per-(n, k-group) block scale is replicated identically across
    // every k-group of a given row n (see fp8_b128_pack_scale_row_kernel),
    // so the packer reads it once from block (n, kblock 0). Paired with the
    // GGML_OP_FP8_QUANT_ROT "per-row" (G=0) activation contract: a->ne[0] ==
    // K + 4 (a single fp32 a_scale per row, not per K-group) -- distinct
    // from GENERIC/PRESHUFFLE/RDNA4's a->ne[0] == K + K/32 "block packing"
    // contract, which those three keep serving unchanged.
    FP8_B128_LAYOUT_RDNA4_TRFEED = 3,
};

static int32_t fp8_b128_current_layout() {
    static const int32_t layout = [] {
        const char * e = getenv("MT_FP8_B128_LAYOUT");
        if (e != nullptr && std::strcmp(e, "preshuffle") == 0) {
            return (int32_t) FP8_B128_LAYOUT_PRESHUFFLE;
        }
        if (e != nullptr && std::strcmp(e, "generic") == 0) {
            return (int32_t) FP8_B128_LAYOUT_GENERIC;
        }
        // Escape hatch to the abandoned round-2 in-loop-fold kernel (A/B only,
        // no production dispatch reads FP8_B128_LAYOUT_RDNA4 by default anymore).
        if (e != nullptr && std::strcmp(e, "rdna4_tile") == 0) {
            return (int32_t) FP8_B128_LAYOUT_RDNA4;
        }
        if (e != nullptr && std::strcmp(e, "rdna4") != 0 && std::strlen(e) > 0) {
            fprintf(stderr, "[fp8_b128] MT_FP8_B128_LAYOUT=%s not recognized, using 'rdna4'\n", e);
        }
        return (int32_t) FP8_B128_LAYOUT_RDNA4_TRFEED;   // default: frozen trfeed WMMA kernel
    }();
    return layout;
}

bool ggml_cuda_fp8_b128_layout_is_per_row(void) {
    return fp8_b128_current_layout() == FP8_B128_LAYOUT_RDNA4_TRFEED;
}

// ─────────────────────────────────────────────────────────────────────
// FP8_B128 in-place pack/unpack (design 4(a)).
//
// Packed layout (bytes [0, N*K)): the raw e4m3 qs bytes of the on-disk
// [N, K] weight, permuted into AITER's shuffle_weight(layout=(16,16))
// order: for row-block nb16 (16 rows) and 32-wide K-chunk kc, a 512-byte
// chunk at offset (nb16*(K/32) + kc)*512 holds, in [half(2)][n_in(16)][k_in(16)]
// order, W[nb16*16 + n_in, kc*32 + half*16 + k_in]'s raw e4m3 byte.
// (PRESHUFFLE layout only — see FP8_B128_LAYOUT_* above for GENERIC.)
//
// Packed layout (bytes [N*K, N*K + 4*(K/128)*(N/128))): fp32 scale table in
// kernel order [K/128 (kb, outer), N/128 (tile_n, inner)]: w_scale[kb*(N/128)
// + tile_n] = f32(on-disk fp16 `d` of block (row tile_n*128, k-group kb)).
// The converter replicates `d` across all 128 rows of a tile; the packer
// reads it once from the tile's first row (see WP_FP8B128_CHECK_SCALE
// below for a debug assert that the other 127 rows agree). SAME order and
// kernel for both GENERIC and PRESHUFFLE layouts.
// ─────────────────────────────────────────────────────────────────────

// One thread per (n, k) output byte. Grid-stride loop over N*K bytes.
static __global__ void fp8_b128_pack_weight_kernel(
    const uint8_t * __restrict__ src,   // on-disk (N, n_groups_k * 130) bytes
    uint8_t       * __restrict__ dst,   // preshuffled [N*K] bytes
    int N, int K, int n_groups_k) {

    const size_t total = (size_t) N * (size_t) K;
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int n = (int) (idx / (size_t) K);
        const int k = (int) (idx % (size_t) K);
        const int g = k / FP8_B128_BLOCK_SIZE;
        const uint8_t qbyte = src[(size_t) n * row_bytes + (size_t) g * FP8_B128_BLOCK_BYTES
                                   + sizeof(ggml_half) + (k % FP8_B128_BLOCK_SIZE)];

        const int nb16  = n / 16;
        const int n_in  = n % 16;
        const int kc    = k / 32;
        const int half  = (k % 32) / 16;
        const int k_in  = k % 16;
        const size_t dst_off = ((size_t) nb16 * (size_t) (K / 32) + (size_t) kc) * 512
                              + (size_t) half * 256 + (size_t) n_in * 16 + (size_t) k_in;
        dst[dst_off] = qbyte;
    }
}

// One thread per (kb, tile_n) scale-table entry.
static __global__ void fp8_b128_pack_scale_kernel(
    const uint8_t * __restrict__ src,     // on-disk blocks
    float         * __restrict__ b_scale, // [K/128, N/128], kb outer, tile_n inner
    int N, int K, int n_groups_k) {

    const int tiles_n = N / FP8_B128_BLOCK_SIZE;
    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int total   = n_groups_k * tiles_n;
    if (idx >= total) return;
    const int kb = idx / tiles_n;
    const int tn = idx % tiles_n;
    const int n0 = tn * FP8_B128_BLOCK_SIZE;

    __half d;
    memcpy(&d, src + (size_t) n0 * (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES
                    + (size_t) kb * (size_t) FP8_B128_BLOCK_BYTES, sizeof(__half));
    b_scale[(size_t) kb * (size_t) tiles_n + (size_t) tn] = __half2float(d);
}

// MAD-305 Phase 5 (round 3) — FP8_B128_LAYOUT_RDNA4_TRFEED's scale table:
// fp32 b_scale[N], one scalar per output row n. The converter replicates the
// on-disk `d` across every one of a row's n_groups_k blocks (a DIFFERENT
// invariant from fp8_b128_pack_scale_kernel's "same tile-of-128-rows" one
// above), so the packer reads block (n, kblock 0) only. One thread per n.
static __global__ void fp8_b128_pack_scale_row_kernel(
    const uint8_t * __restrict__ src,     // on-disk blocks, (N, n_groups_k*130) bytes
    float         * __restrict__ b_scale, // [N]
    int N, int n_groups_k) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    __half d;
    memcpy(&d, src + (size_t) n * (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES, sizeof(__half));
    b_scale[n] = __half2float(d);
}

// Inverse of fp8_b128_pack_scale_row_kernel: broadcast each row's fp32 scale
// (narrowed to fp16) into every one of that row's n_groups_k on-disk blocks.
static __global__ void fp8_b128_unpack_scale_row_kernel(
    const float * __restrict__ b_scale, uint8_t * __restrict__ dst,
    int N, int n_groups_k) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const __half d = __float2half(b_scale[n]);
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    for (int g = 0; g < n_groups_k; g++) {
        memcpy(dst + (size_t) n * row_bytes + (size_t) g * FP8_B128_BLOCK_BYTES, &d, sizeof(__half));
    }
}

// Debug-only (WP_FP8B128_CHECK_SCALE=1): warns if a tile's 128 rows don't
// actually share the same on-disk scale for a given K-group — expected for
// converter-produced weights (the CONVERTER INVARIANT), NOT guaranteed for
// ad-hoc test-quantized tensors (test-backend-ops' test_fp8_mul_mat
// quantizes each row independently — see ggml-turbo-quant.c
// quantize_row_fp8_b128_ref). One thread per (tile_n, kb) tile.
static __global__ void fp8_b128_check_scale_kernel(
    const uint8_t * __restrict__ src, int N, int K, int n_groups_k) {
    const int tn = blockIdx.x;
    const int kb = blockIdx.y;
    const int n0 = tn * FP8_B128_BLOCK_SIZE;
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    uint16_t d0;
    memcpy(&d0, src + (size_t) n0 * row_bytes + (size_t) kb * FP8_B128_BLOCK_BYTES, 2);
    for (int r = 1; r < FP8_B128_BLOCK_SIZE && (n0 + r) < N; r++) {
        uint16_t dr;
        memcpy(&dr, src + (size_t) (n0 + r) * row_bytes + (size_t) kb * FP8_B128_BLOCK_BYTES, 2);
        if (dr != d0) {
            printf("[fp8_b128] WARNING: tile scale mismatch tile_n=%d (rows %d..%d) kb=%d: "
                   "row %d scale bits 0x%04x != row %d scale bits 0x%04x\n",
                   tn, n0, n0 + FP8_B128_BLOCK_SIZE - 1, kb, n0 + r, (unsigned) dr, n0, (unsigned) d0);
        }
    }
}

// Inverse of fp8_b128_pack_weight_kernel: gather the preshuffled bytes back
// into the on-disk block_fp8_b128 qs[] layout.
static __global__ void fp8_b128_unpack_weight_kernel(
    const uint8_t * __restrict__ packed, // preshuffled [N*K] bytes
    uint8_t       * __restrict__ dst,    // on-disk (N, n_groups_k * 130) bytes
    int N, int K, int n_groups_k) {

    const size_t total = (size_t) N * (size_t) K;
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int n = (int) (idx / (size_t) K);
        const int k = (int) (idx % (size_t) K);
        const int nb16  = n / 16;
        const int n_in  = n % 16;
        const int kc    = k / 32;
        const int half  = (k % 32) / 16;
        const int k_in  = k % 16;
        const size_t src_off = ((size_t) nb16 * (size_t) (K / 32) + (size_t) kc) * 512
                              + (size_t) half * 256 + (size_t) n_in * 16 + (size_t) k_in;
        const uint8_t qbyte = packed[src_off];
        const int g = k / FP8_B128_BLOCK_SIZE;
        dst[(size_t) n * row_bytes + (size_t) g * FP8_B128_BLOCK_BYTES
            + sizeof(ggml_half) + (k % FP8_B128_BLOCK_SIZE)] = qbyte;
    }
}

// ─────────────────────────────────────────────────────────────────────
// FP8_B128 GENERIC layout pack/unpack: byte [0, N*K) is the e4m3 weight
// TRANSPOSED to [K, N] row-major — the same b_packed convention the
// ML8_FP8 WEIGHT_FORMAT=0 path uses (see ml8_fp8_repack_kernel above) —
// instead of the AITER preshuffle permutation. The [K/128, N/128] fp32
// scale table (fp8_b128_pack_scale_kernel / fp8_b128_unpack_scale_kernel)
// is unchanged and shared with the preshuffle layout.
// ─────────────────────────────────────────────────────────────────────

// One thread per (n, k) output byte. Grid-stride loop over N*K bytes.
static __global__ void fp8_b128_pack_weight_generic_kernel(
    const uint8_t * __restrict__ src,   // on-disk (N, n_groups_k * 130) bytes
    uint8_t       * __restrict__ dst,   // transposed [K, N] bytes, row-major over K
    int N, int K, int n_groups_k) {

    const size_t total = (size_t) N * (size_t) K;
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int n = (int) (idx / (size_t) K);
        const int k = (int) (idx % (size_t) K);
        const int g = k / FP8_B128_BLOCK_SIZE;
        const uint8_t qbyte = src[(size_t) n * row_bytes + (size_t) g * FP8_B128_BLOCK_BYTES
                                   + sizeof(ggml_half) + (k % FP8_B128_BLOCK_SIZE)];
        dst[(size_t) k * (size_t) N + (size_t) n] = qbyte;
    }
}

// Inverse of fp8_b128_pack_weight_generic_kernel: gather the transposed
// [K, N] bytes back into the on-disk block_fp8_b128 qs[] layout.
static __global__ void fp8_b128_unpack_weight_generic_kernel(
    const uint8_t * __restrict__ packed, // transposed [K, N] bytes, row-major over K
    uint8_t       * __restrict__ dst,    // on-disk (N, n_groups_k * 130) bytes
    int N, int K, int n_groups_k) {

    const size_t total = (size_t) N * (size_t) K;
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t) gridDim.x * blockDim.x) {
        const int n = (int) (idx / (size_t) K);
        const int k = (int) (idx % (size_t) K);
        const uint8_t qbyte = packed[(size_t) k * (size_t) N + (size_t) n];
        const int g = k / FP8_B128_BLOCK_SIZE;
        dst[(size_t) n * row_bytes + (size_t) g * FP8_B128_BLOCK_BYTES
            + sizeof(ggml_half) + (k % FP8_B128_BLOCK_SIZE)] = qbyte;
    }
}

// Inverse of fp8_b128_pack_scale_kernel: broadcast each tile's fp32 scale
// (narrowed back to fp16) into all 128 on-disk block `d` fields it covers.
static __global__ void fp8_b128_unpack_scale_kernel(
    const float * __restrict__ b_scale, uint8_t * __restrict__ dst,
    int N, int K, int n_groups_k) {

    const int tiles_n = N / FP8_B128_BLOCK_SIZE;
    const int idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int total   = n_groups_k * tiles_n;
    if (idx >= total) return;
    const int kb = idx / tiles_n;
    const int tn = idx % tiles_n;
    const __half d = __float2half(b_scale[(size_t) kb * (size_t) tiles_n + (size_t) tn]);
    const size_t row_bytes = (size_t) n_groups_k * (size_t) FP8_B128_BLOCK_BYTES;
    for (int r = 0; r < FP8_B128_BLOCK_SIZE; r++) {
        const int n = tn * FP8_B128_BLOCK_SIZE + r;
        memcpy(dst + (size_t) n * row_bytes + (size_t) kb * FP8_B128_BLOCK_BYTES, &d, sizeof(__half));
    }
}

// Pack the on-disk FP8_B128 blocks (`staging`, on-disk (N, n_groups_k*130)
// bytes) into `dst_packed` (N*K bytes) + `dst_scale` (fp32 [K/128,N/128])
// according to `layout`. The scale table is IDENTICAL across all three
// layouts (fp8_b128_pack_scale_kernel doesn't care how the weight bytes
// are permuted); only the weight-byte packing differs:
//   GENERIC:     straight [K,N] transpose.
//   PRESHUFFLE:  AITER shuffle_weight(16,16).
//   RDNA4:       [K,N] transpose (reusing the GENERIC kernel as a pure
//                transpose step into a temp buffer), then shuffled into the
//                trfeed fragment-tile layout via rdna4_preshuffle_b_ml8fp8
//                (device-side, no host round trip -- the same byte-shuffle
//                ML8_FP8's rdna4 layout uses; it only permutes e4m3 bytes,
//                agnostic to which quant format produced them).
static void fp8_b128_pack_for_layout(
    cudaStream_t stream, const uint8_t * staging, uint8_t * dst_packed, float * dst_scale,
    int32_t N, int32_t K, int32_t n_groups_k, int32_t layout) {
    constexpr int TPB = 256;
    const size_t total_bytes = (size_t) N * (size_t) K;
    const int grid_x = (int) std::min<size_t>((total_bytes + TPB - 1) / TPB, (size_t) 65535);
    if (layout == FP8_B128_LAYOUT_PRESHUFFLE) {
        fp8_b128_pack_weight_kernel<<<grid_x, TPB, 0, stream>>>(staging, dst_packed, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
    } else if (layout == FP8_B128_LAYOUT_GENERIC) {
        fp8_b128_pack_weight_generic_kernel<<<grid_x, TPB, 0, stream>>>(staging, dst_packed, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
    } else {
        uint8_t * tmp_kn = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &tmp_kn, (size_t) K * (size_t) N));
        fp8_b128_pack_weight_generic_kernel<<<grid_x, TPB, 0, stream>>>(staging, tmp_kn, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        const hipError_t rc = rdna4_preshuffle_b_ml8fp8(tmp_kn, dst_packed, K, N, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(tmp_kn);
        GGML_ASSERT(rc == hipSuccess && "rdna4_preshuffle_b_ml8fp8 (fp8_b128) failed");
    }
    if (layout == FP8_B128_LAYOUT_RDNA4_TRFEED) {
        // Weight bytes: [0, N*K) preshuffled e4m3, identical to the RDNA4
        // (round 2) branch above -- both fall into the `else` arm. Scale:
        // fp32[N], one scalar per output row (see fp8_b128_pack_scale_row_kernel).
        // Buffer-fit invariant (gemm_capi.h): the in-place allocator sizes
        // this tensor's data buffer to fit the on-disk (N, n_groups_k*130)
        // footprint (130*N*K/128 bytes); this layout only ever needs
        // N*K + 4*N bytes, which is always <= that for FP8_B128_BLOCK_SIZE=128.
        GGML_ASSERT((size_t) 130 * (size_t) N * (size_t) K / 128 >=
                     (size_t) N * (size_t) K + 4 * (size_t) N &&
                     "FP8_B128_LAYOUT_RDNA4_TRFEED: N*K+4N exceeds the in-place buffer footprint");
        fp8_b128_pack_scale_row_kernel<<<(N + 255) / 256, 256, 0, stream>>>(staging, dst_scale, N, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int tiles_n = N / FP8_B128_BLOCK_SIZE;
    const int scale_total = n_groups_k * tiles_n;
    fp8_b128_pack_scale_kernel<<<(scale_total + 255) / 256, 256, 0, stream>>>(staging, dst_scale, N, K, n_groups_k);
    CUDA_CHECK(cudaGetLastError());
    if (getenv("WP_FP8B128_CHECK_SCALE") != nullptr) {
        const dim3 grid_chk((unsigned) tiles_n, (unsigned) n_groups_k, 1);
        fp8_b128_check_scale_kernel<<<grid_chk, 1, 0, stream>>>(staging, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Inverse of fp8_b128_pack_for_layout: reconstruct the on-disk block bytes
// (dst, (N, n_groups_k*130)) from a packed FP8_B128 entry of the given layout.
static void fp8_b128_unpack_for_layout(
    cudaStream_t stream, const uint8_t * src_packed, const float * src_scale, uint8_t * dst,
    int32_t N, int32_t K, int32_t n_groups_k, int32_t layout) {
    constexpr int TPB = 256;
    const size_t total_bytes = (size_t) N * (size_t) K;
    const int grid_x = (int) std::min<size_t>((total_bytes + TPB - 1) / TPB, (size_t) 65535);
    if (layout == FP8_B128_LAYOUT_PRESHUFFLE) {
        fp8_b128_unpack_weight_kernel<<<grid_x, TPB, 0, stream>>>(src_packed, dst, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
    } else if (layout == FP8_B128_LAYOUT_GENERIC) {
        fp8_b128_unpack_weight_generic_kernel<<<grid_x, TPB, 0, stream>>>(src_packed, dst, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
    } else {
        uint8_t * tmp_kn = nullptr;
        CUDA_CHECK(cudaMalloc((void **) &tmp_kn, (size_t) K * (size_t) N));
        const hipError_t rc = rdna4_unshuffle_b_ml8fp8(src_packed, tmp_kn, K, N, stream);
        GGML_ASSERT(rc == hipSuccess && "rdna4_unshuffle_b_ml8fp8 (fp8_b128) failed");
        fp8_b128_unpack_weight_generic_kernel<<<grid_x, TPB, 0, stream>>>(tmp_kn, dst, N, K, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(tmp_kn);
    }
    if (layout == FP8_B128_LAYOUT_RDNA4_TRFEED) {
        fp8_b128_unpack_scale_row_kernel<<<(N + 255) / 256, 256, 0, stream>>>(src_scale, dst, N, n_groups_k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const int tiles_n = N / FP8_B128_BLOCK_SIZE;
    const int scale_total = n_groups_k * tiles_n;
    fp8_b128_unpack_scale_kernel<<<(scale_total + 255) / 256, 256, 0, stream>>>(src_scale, dst, N, K, n_groups_k);
    CUDA_CHECK(cudaGetLastError());
}

// Cache-keyed FP8_B128 repack (second-copy fallback), mirroring
// ggml_cuda_ml8_fp8_get_or_repack above. Used by GGML_OP_FP8_MUL_MAT when
// the weight isn't in-place eligible (WP_ML8_INPLACE=0, or N not a multiple
// of 128 — e.g. some test-backend-ops shapes) so the AITER preshuffle GEMM
// still has somewhere to read the preshuffled layout + scale table from.
// (g_fp8_b128_cache_mu / g_fp8_b128_cache declared near g_ml8_fp8_cache above.)
static const ml8_weight_repack_t * ggml_cuda_fp8_b128_get_or_repack(
    cudaStream_t stream, const ggml_tensor * w) {

    if (w == nullptr || w->data == nullptr || w->type != GGML_TYPE_FP8_B128) {
        return nullptr;
    }
    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    if (K <= 0 || N <= 0 || K % FP8_B128_BLOCK_SIZE != 0 || N % FP8_B128_BLOCK_SIZE != 0) {
        return nullptr;
    }
    const int32_t n_groups_k = K / FP8_B128_BLOCK_SIZE;
    const int32_t tiles_n    = N / FP8_B128_BLOCK_SIZE;
    const void * key = w->data;

    // Prefer the in-place registry if this weight happens to be packed
    // there already (e.g. it WAS eligible and WP_ML8_INPLACE wasn't 0).
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(key);
        if (it != g_ml8_inplace.end() && it->second.packed && it->second.type == GGML_TYPE_FP8_B128) {
            return &it->second.info;
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_fp8_b128_cache_mu);
        auto it = g_fp8_b128_cache.find(key);
        if (it != g_fp8_b128_cache.end()) {
            return &it->second.info;
        }
    }

    void *  d_packed = nullptr;
    float * d_scale  = nullptr;
    const size_t packed_bytes = (size_t) K * (size_t) N;
    const int32_t layout_for_alloc = fp8_b128_current_layout();
    // FP8_B128_LAYOUT_RDNA4_TRFEED's scale table is fp32[N] (one scalar per
    // output row), NOT the [K/128,N/128] tile table the other three layouts
    // share -- and N can exceed n_groups_k*tiles_n for K < N (e.g. K=5120,
    // N=17408: n_groups_k*tiles_n = 40*136 = 5440 < 17408), so reusing the
    // tile-table formula here would under-allocate and let
    // fp8_b128_pack_scale_row_kernel write past the buffer.
    const size_t scale_bytes = (layout_for_alloc == FP8_B128_LAYOUT_RDNA4_TRFEED)
        ? (size_t) N * sizeof(float)
        : (size_t) n_groups_k * (size_t) tiles_n * sizeof(float);

    cudaError_t err = cudaMalloc(&d_packed, packed_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[fp8_b128] cudaMalloc(packed=%zu) failed: %s\n", packed_bytes, cudaGetErrorString(err));
        return nullptr;
    }
    err = cudaMalloc((void **) &d_scale, scale_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[fp8_b128] cudaMalloc(scale=%zu) failed: %s\n", scale_bytes, cudaGetErrorString(err));
        cudaFree(d_packed);
        return nullptr;
    }

    const int32_t layout = layout_for_alloc;
    fp8_b128_pack_for_layout(stream, (const uint8_t *) w->data, (uint8_t *) d_packed, d_scale,
        N, K, n_groups_k, layout);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[fp8_b128] repack kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_packed);
        cudaFree(d_scale);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_fp8_b128_cache_mu);
    auto it = g_fp8_b128_cache.find(key);
    if (it != g_fp8_b128_cache.end()) {
        cudaFree(d_packed);
        cudaFree(d_scale);
        return &it->second.info;
    }
    cache_entry_t entry{};
    entry.info.b_packed   = d_packed;
    entry.info.b_scale    = d_scale;
    entry.info.N          = N;
    entry.info.K          = K;
    entry.info.n_groups_k = n_groups_k;
    entry.info.group_size = FP8_B128_BLOCK_SIZE;
    entry.info.layout     = layout;
    auto [ins_it, _ins_ok] = g_fp8_b128_cache.emplace(key, entry);
    return &ins_it->second.info;
}

bool ggml_cuda_ml8_inplace_eligible(const ggml_tensor * t) {
    static const bool disabled = [] {
        const char * e = getenv("WP_ML8_INPLACE");
        return e != nullptr && atoi(e) == 0;
    }();
    if (disabled || t == nullptr || t->view_src != nullptr) {
        return false;
    }
    if (t->type != GGML_TYPE_ML8_FP8 && t->type != GGML_TYPE_ML8_4 && t->type != GGML_TYPE_FP8_B128) {
        return false;
    }
    const bool is_fp8_b128 = t->type == GGML_TYPE_FP8_B128;
    const int    qk  = t->type == GGML_TYPE_ML8_4 ? QK_ML8
                      : is_fp8_b128               ? FP8_B128_BLOCK_SIZE
                                                    : QK_ML8_FP8;
    const size_t bsz = t->type == GGML_TYPE_ML8_4 ? sizeof(block_ml8_4)
                      : is_fp8_b128               ? (size_t) FP8_B128_BLOCK_BYTES
                                                    : (size_t) ML8_FP8_BLOCK_BYTES;
    if (t->ne[2] != 1 || t->ne[3] != 1 || t->ne[0] <= 0 || t->ne[1] <= 0 || t->ne[0] % qk != 0) {
        return false;
    }
    if (t->ne[0] > INT32_MAX || t->ne[1] > INT32_MAX) {
        return false;
    }
    // FP8_B128's preshuffled packed layout (design 4(a)) needs whole 128x128
    // N x K tiles for the [K/128, N/128] scale table; N not a multiple of
    // 128 (e.g. some test-backend-ops shapes) falls back to the cache-copy
    // repack path in ggml_cuda_fp8_b128_get_or_repack instead of in-place.
    if (is_fp8_b128 && t->ne[1] % 128 != 0) {
        return false;
    }
    // The trfeed layout stores N*K preshuffled bytes + fp32 [N] row scales; that
    // only fits the on-disk footprint (130*N*K/128) when K >= 256. Smaller K
    // (test shapes only) takes the cache-copy path.
    if (is_fp8_b128 && fp8_b128_current_layout() == FP8_B128_LAYOUT_RDNA4_TRFEED && t->ne[0] < 256) {
        return false;
    }
    // The rdna4 preshuffle (16x16 fragment tiles) needs N % 16 == 0 and K % 16 == 0
    // for ML8_FP8 weights; anything else (small GET_ROWS test tables) stays unpacked.
    if (t->type == GGML_TYPE_ML8_FP8 && ml8_fp8_gemm_current_layout() == ML8_FP8_GEMM_LAYOUT_RDNA4 &&
        (t->ne[1] % 16 != 0 || t->ne[0] % 16 != 0)) {
        return false;
    }
    // Contiguous rows of whole blocks (the loader never hands us anything else).
    return t->nb[0] == bsz && t->nb[1] == (size_t) (t->ne[0] / qk) * bsz;
}

size_t ggml_cuda_ml8_inplace_alloc_size(const ggml_tensor * t) {
    const size_t K = (size_t) t->ne[0];
    const size_t N = (size_t) t->ne[1];
    if (t->type == GGML_TYPE_ML8_4) {
        // nibbles [K/2, N] + fp32 scales [K/64, N] == 4.5 bpw, same as on disk
        return K * N / 2 + (K / QK_ML8) * N * sizeof(float);
    }
    if (t->type == GGML_TYPE_FP8_B128) {
        // preshuffled raw e4m3 [N, K] (byte-for-byte permutation of the
        // on-disk qs bytes, same total count) + fp32 scale table
        // [K/128, N/128]. Always smaller than ggml_nbytes(t) (== N*K +
        // 2*N*(K/128) on-disk) since 4*(N/128) < 2*N — see design 4(a).
        return K * N + 4 * (K / FP8_B128_BLOCK_SIZE) * (N / FP8_B128_BLOCK_SIZE);
    }
    // raw e4m3 [K, N] + fp16 scales [K/32, N] == 8.5 bpw, same as on disk
    // (== ggml_nbytes(t)).
    return K * N + (K / QK_ML8_FP8) * N * sizeof(__half);
}

bool ggml_cuda_ml8_inplace_is_packed(const void * data) {
    std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
    auto it = g_ml8_inplace.find(data);
    return it != g_ml8_inplace.end() && it->second.packed;
}

void ggml_cuda_ml8_inplace_alias(const void * src_data, const void * dst_data) {
    std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
    auto it = g_ml8_inplace.find(src_data);
    if (it == g_ml8_inplace.end() || !it->second.packed) {
        return;
    }
    inplace_entry_t e = it->second;
    const ptrdiff_t delta = (const char *) dst_data - (const char *) src_data;
    e.info.b_packed = (void *) ((char *) e.info.b_packed + delta);
    e.info.b_scale  = (void *) ((char *) e.info.b_scale  + delta);
    g_ml8_inplace[dst_data] = e;
}

void ggml_cuda_ml8_inplace_forget_range(const void * base, size_t size) {
    const char * lo = (const char *) base;
    const char * hi = lo + size;
    std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
    for (auto it = g_ml8_inplace.begin(); it != g_ml8_inplace.end(); ) {
        const char * p = (const char *) it->first;
        if (p >= lo && p < hi) {
            if (it->second.staging != nullptr) {
                cudaFree(it->second.staging);
            }
            it = g_ml8_inplace.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_cuda_ml8_inplace_set(
    cudaStream_t  stream,
    ggml_tensor * t,
    const void *  data,
    size_t        offset,
    size_t        size,
    size_t        n_copies,
    size_t        stride_tensor,
    size_t        stride_data) {

    GGML_ASSERT(ggml_cuda_ml8_inplace_eligible(t));
    const bool    is_ml8_4    = t->type == GGML_TYPE_ML8_4;
    const bool    is_fp8_b128 = t->type == GGML_TYPE_FP8_B128;
    const int32_t K          = (int32_t) t->ne[0];
    const int32_t N          = (int32_t) t->ne[1];
    const int32_t group_size = is_ml8_4 ? QK_ML8 : is_fp8_b128 ? FP8_B128_BLOCK_SIZE : QK_ML8_FP8;
    const int32_t n_groups_k = K / group_size;
    // FP8_B128's preshuffled bytes are also K*N (a byte permutation, same
    // count as raw e4m3 [K,N]) — same packed_sz formula as ML8_FP8.
    const size_t  packed_sz  = is_ml8_4 ? (size_t) K * (size_t) N / 2 : (size_t) K * (size_t) N;
    const size_t  nbytes     = ggml_nbytes(t);
    GGML_ASSERT(n_copies >= 1);
    GGML_ASSERT(offset + (n_copies - 1) * stride_tensor + size <= nbytes);

    std::unique_lock<std::mutex> lock(g_ml8_inplace_mu);
    inplace_entry_t & e = g_ml8_inplace[t->data];
    if (e.info.b_packed == nullptr) {
        e.info.b_packed   = t->data;
        e.info.b_scale    = (void *) ((char *) t->data + packed_sz);
        e.info.N          = N;
        e.info.K          = K;
        e.info.n_groups_k = n_groups_k;
        e.info.group_size = group_size;
        e.info.layout     = is_fp8_b128 ? fp8_b128_current_layout()
                          : (t->type == GGML_TYPE_ML8_FP8) ? ml8_fp8_gemm_current_layout()
                          : is_ml8_4 ? ml8_4_layout_for_tensor(N)
                          : 0;
        e.type            = t->type;
        e.nbytes          = nbytes;
        e.staging         = nullptr;
        e.received        = 0;
        e.packed          = false;
    }
    if (e.packed) {
        // Re-writing an already packed tensor: start over from a fresh staging.
        e.packed   = false;
        e.received = 0;
    }
    if (e.staging == nullptr) {
        CUDA_CHECK(cudaMalloc((void **) &e.staging, nbytes));
    }
    uint8_t * staging = e.staging;
    lock.unlock();

    if (n_copies == 1 || (stride_tensor == size && stride_data == size)) {
        CUDA_CHECK(cudaMemcpyAsync(staging + offset, data, size * n_copies, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(staging + offset, stride_tensor, data, stride_data, size, n_copies,
                                     cudaMemcpyHostToDevice, stream));
    }

    lock.lock();
    e.received += size * n_copies;
    const bool complete = e.received >= nbytes;
    if (complete) {
        if (is_fp8_b128) {
            fp8_b128_pack_for_layout(stream, staging, (uint8_t *) e.info.b_packed, (float *) e.info.b_scale,
                N, K, n_groups_k, e.info.layout);
            CUDA_CHECK(cudaGetLastError());
        } else if (is_ml8_4) {
            // MAD-305: layout-aware (RDNA4_TRFEED default on gfx1201, TRITON
            // via MT_ML8_4_LAYOUT=triton or a non-128-multiple N).
            ml8_4_pack_for_layout(
                stream, staging, (uint8_t *) e.info.b_packed, (float *) e.info.b_scale,
                N, K, n_groups_k, e.info.layout);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // GGML_TYPE_ML8_FP8: layout-aware (MAD-305 Phase 5) -- TRITON packs
            // straight into e.info.b_packed; RDNA4 stages through a temp [K,N]
            // buffer and shuffles into the trfeed fragment-tile layout.
            ml8_fp8_pack_for_layout(
                stream, staging, (uint8_t *) e.info.b_packed, (__half *) e.info.b_scale,
                N, K, n_groups_k, e.info.layout);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    lock.unlock();

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (complete) {
        lock.lock();
        CUDA_CHECK(cudaFree(staging));
        e.staging  = nullptr;
        e.received = 0;
        e.packed   = true;
    }
}

void ggml_cuda_ml8_inplace_get(
    cudaStream_t        stream,
    const ggml_tensor * t,
    void *              data,
    size_t              offset,
    size_t              size) {

    ml8_weight_repack_t info;
    size_t nbytes;
    ggml_type type;
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(t->data);
        GGML_ASSERT(it != g_ml8_inplace.end() && it->second.packed &&
            "ml8 in-place tensor read before it was fully written");
        info   = it->second.info;
        nbytes = it->second.nbytes;
        type   = it->second.type;
    }
    GGML_ASSERT(offset + size <= nbytes);

    uint8_t * tmp = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &tmp, nbytes));
    if (type == GGML_TYPE_FP8_B128) {
        fp8_b128_unpack_for_layout(stream, (const uint8_t *) info.b_packed, (const float *) info.b_scale, tmp,
            info.N, info.K, info.n_groups_k, info.layout);
    } else if (type == GGML_TYPE_ML8_4) {
        ml8_4_unpack_for_layout(stream, (const uint8_t *) info.b_packed, (const float *) info.b_scale, tmp,
            info.N, info.K, info.n_groups_k, info.layout);
    } else {
        // GGML_TYPE_ML8_FP8: layout-aware (MAD-305 Phase 5).
        ml8_fp8_unpack_for_layout(
            stream, (const uint8_t *) info.b_packed, (const __half *) info.b_scale, tmp,
            info.N, info.K, info.n_groups_k, info.layout);
    }
    CUDA_CHECK(cudaMemcpyAsync(data, tmp + offset, size, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(tmp));
}

// Device-side variant of ggml_cuda_ml8_inplace_get: unpacks into a fresh
// device buffer instead of copying to a host pointer. Used by the generic
// GET_ROWS dequant fallback (getrows.cu) so it can read the on-disk block
// layout without a host round-trip. Returns nullptr if `t->data` isn't a
// fully-packed FP8_B128 entry.
void * ggml_cuda_ml8_inplace_fp8_b128_unpack_to_device(
    cudaStream_t stream, const ggml_tensor * t) {

    ml8_weight_repack_t info;
    size_t nbytes;
    ggml_type type;
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(t->data);
        if (it == g_ml8_inplace.end() || !it->second.packed) {
            return nullptr;
        }
        info   = it->second.info;
        nbytes = it->second.nbytes;
        type   = it->second.type;
    }
    if (type != GGML_TYPE_FP8_B128) {
        return nullptr;
    }

    uint8_t * tmp = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &tmp, nbytes));
    fp8_b128_unpack_for_layout(stream, (const uint8_t *) info.b_packed, (const float *) info.b_scale, tmp,
        info.N, info.K, info.n_groups_k, info.layout);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return tmp;
}

// ML8_FP8 sibling of ggml_cuda_ml8_inplace_fp8_b128_unpack_to_device above:
// unpacks a packed in-place GGML_TYPE_ML8_FP8 entry into a freshly cudaMalloc'd
// device buffer of on-disk block_ml8_fp8 bytes, layout-aware (MAD-305 Phase 5
// -- mirrors how FP8_B128's GENERIC/PRESHUFFLE pair is handled). Used by the
// GET_ROWS dequant fallback (getrows.cu) when the fast packed-layout kernel
// (ggml_cuda_ml8_inplace_get_rows) declines an RDNA4-layout weight. Caller
// owns the returned pointer and must cudaFree it. Returns nullptr if
// `t->data` isn't a fully-packed ML8_FP8 entry.
void * ggml_cuda_ml8_inplace_ml8fp8_unpack_to_device(
    cudaStream_t stream, const ggml_tensor * t) {

    ml8_weight_repack_t info;
    size_t nbytes;
    ggml_type type;
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(t->data);
        if (it == g_ml8_inplace.end() || !it->second.packed) {
            return nullptr;
        }
        info   = it->second.info;
        nbytes = it->second.nbytes;
        type   = it->second.type;
    }
    if (type != GGML_TYPE_ML8_FP8) {
        return nullptr;
    }

    uint8_t * tmp = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &tmp, nbytes));
    ml8_fp8_unpack_for_layout(
        stream, (const uint8_t *) info.b_packed, (const __half *) info.b_scale, tmp,
        info.N, info.K, info.n_groups_k, info.layout);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return tmp;
}

// get_rows over the packed layout: row n of the logical [N, K] weight is
// column n of b_fp8 [K, N]. One block per output row, threads stride K.
template <typename dst_t>
static __global__ void ml8_fp8_packed_get_rows_kernel(
    const uint8_t * __restrict__ b_fp8,     // (K, N)
    const __half  * __restrict__ b_scale,   // (K/32, N), fp16
    const int32_t * __restrict__ ids,
    dst_t         * __restrict__ dst,
    int K, int N,
    int64_t ne10, int64_t ne11,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1,  size_t nb2,  size_t nb3) {

    const int64_t i10 = blockIdx.x;
    const int64_t i11 = blockIdx.y;
    const int64_t i12 = blockIdx.z;
    if (i10 >= ne10 || i11 >= ne11) {
        return;
    }
    const int32_t n = *(const int32_t *) ((const char *) ids + i10*nb10 + i11*nb11 + i12*nb12);
    if (n < 0 || n >= N) {
        return;
    }
    dst_t * out = (dst_t *) ((char *) dst + i10*nb1 + i11*nb2 + i12*nb3);
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float scale = __half2float(b_scale[(size_t) (k / QK_ML8_FP8) * N + n]);
        const float v = ggml_cuda_e4m3fn_to_fp32(b_fp8[(size_t) k * N + n]) * scale;
        out[k] = ggml_cuda_cast<dst_t>(v);
    }
}

bool ggml_cuda_ml8_inplace_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    if (src0->type != GGML_TYPE_ML8_FP8) {
        return false;
    }
    ml8_weight_repack_t info;
    {
        std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
        auto it = g_ml8_inplace.find(src0->data);
        if (it == g_ml8_inplace.end() || !it->second.packed) {
            return false;
        }
        info = it->second.info;
    }
    // ml8_fp8_packed_get_rows_kernel below reads b_fp8 as a straight [K,N]
    // transpose (TRITON layout). An RDNA4-layout weight's bytes are
    // fragment-tile-shuffled, not [K,N]-contiguous -- fall through to the
    // caller's ggml_cuda_ml8_inplace_ml8fp8_unpack_to_device fallback instead
    // of reading garbage through this fast path.
    if (info.layout != ML8_FP8_GEMM_LAYOUT_TRITON) {
        return false;
    }
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(src0->ne[2] == 1 && src0->ne[3] == 1);

    const dim3 grid((unsigned) src1->ne[0], (unsigned) src1->ne[1], (unsigned) src1->ne[2]);
    const dim3 block(256, 1, 1);
    cudaStream_t stream = ctx.stream();
    switch (dst->type) {
        case GGML_TYPE_F32:
            ml8_fp8_packed_get_rows_kernel<float><<<grid, block, 0, stream>>>(
                (const uint8_t *) info.b_packed, (const __half *) info.b_scale, (const int32_t *) src1->data, (float *) dst->data,
                info.K, info.N, src1->ne[0], src1->ne[1], src1->nb[0], src1->nb[1], src1->nb[2],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        case GGML_TYPE_F16:
            ml8_fp8_packed_get_rows_kernel<half><<<grid, block, 0, stream>>>(
                (const uint8_t *) info.b_packed, (const __half *) info.b_scale, (const int32_t *) src1->data, (half *) dst->data,
                info.K, info.N, src1->ne[0], src1->ne[1], src1->nb[0], src1->nb[1], src1->nb[2],
                dst->nb[1], dst->nb[2], dst->nb[3]);
            break;
        default:
            GGML_ABORT("ml8_fp8 packed get_rows: unsupported dst type %s", ggml_type_name(dst->type));
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
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

// MT_FP8_TRFEED_F32OUT=0 restores the bf16-store + convert path of the
// frozen trfeed prefill GEMM (A/B against the fp32 epilogue).
static bool ml8_trfeed_f32out_disabled() {
    static const bool off = [] {
        const char * e = std::getenv("MT_FP8_TRFEED_F32OUT");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    return off;
}

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
// shared memory. Groups are strided over the K_COOP threads, so n_groups_k
// need not divide evenly (TP K-slices do not).
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

    // K groups are strided across the K_COOP threads of a column so any
    // n_groups_k is covered exactly once (a contiguous n_groups_k / K_COOP
    // split silently dropped the remainder — TP K-slices such as 12544 or
    // 4992 give 196 / 78 groups, and K=256 gives fewer groups than threads).

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
        for (int g = k_part; g < n_groups_k; g += K_COOP) {
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
    const float * __restrict__ src,        // [M_valid, K] row-major
    uint8_t     * __restrict__ a_fp8,      // [M, K] row-major
    float       * __restrict__ a_scale,    // [M]
    int K,
    int M_valid) {

    const int m = blockIdx.x;
    const int tid = threadIdx.x;

    uint8_t       * row_out = a_fp8 + (size_t) m * (size_t) K;

    // GEMM M-padding row: emit zero fp8 + epsilon scale without touching
    // src (which only has M_valid rows). Identical output to the old
    // zero-padded-staging path: quantize(0) = 0x00, absmax 0 → eps scale.
    if (m >= M_valid) {
        for (int k = tid; k < K; k += ML8_ACT_QUANT_TPB) {
            row_out[k] = 0;
        }
        if (tid == 0) {
            a_scale[m] = ML8_ACT_SCALE_EPS;
        }
        return;
    }

    const float   * row_in  = src   + (size_t) m * (size_t) K;

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
    int32_t       K,
    int32_t       M_valid) {

    GGML_ASSERT(M > 0);
    GGML_ASSERT(K > 0);
    GGML_ASSERT(M_valid > 0 && M_valid <= M);
    GGML_ASSERT(src_fp32   != nullptr);
    GGML_ASSERT(dst_a_fp8  != nullptr);
    GGML_ASSERT(dst_a_scale != nullptr);

    const dim3 grid((unsigned) M, 1, 1);
    const dim3 block(ML8_ACT_QUANT_TPB, 1, 1);

    ml8_quantize_activations_kernel<<<grid, block, 0, stream>>>(
        src_fp32,
        (uint8_t *) dst_a_fp8,
        dst_a_scale,
        K,
        M_valid);
}

// ─────────────────────────────────────────────────────────────────────
// G.6.d — fused rotation+quantize GEMM prologue.
//
// One block per output row m: load x[m] into LDS, run the H_b FWHT on each
// of the a_dim slices of length b_dim, apply the small H_a^T left-multiply
// in registers, then absmax-reduce / e4m3-quantize the rotated row straight
// into (a_fp8, a_scale). Replaces the per-GEMM chain
//   memcpy(z) → mt_turbo_fp8_fwht → ml8_h_a_left_multiply →
//   [pad memset+memcpy] → ml8_quantize_activations
// with a single launch.
//
// Bit-equivalence to the unfused chain, piece by piece:
//   * butterfly: same pairing (partner = tid ^ stride) and same lower/upper
//     (a+b) / (partner−self) assignment as mt_turbo_fp8_fwht_kernel, same
//     stage order, same final ×rsqrtf(b_dim) normalize;
//   * H_a^T: same sequential-i accumulation as ml8_h_a_left_multiply_kernel;
//   * quantize: fmaxf absmax is exact regardless of reduction shape, then
//     the same scale/eps/e4m3 math as ml8_quantize_activations_kernel.
//
// blockDim.x = b_dim (pow2, 16..1024); dynamic LDS = K fp32 (gated by
// ggml_cuda_ml8_can_fuse_rot_mm to fit with the static reduce array).
// Rows m ≥ M_valid are GEMM padding: zero fp8 + eps scale, src not read.
static __global__ void ml8_fused_rot_quant_kernel(
    const float * __restrict__ x,        // [M_valid, K] row-major, pre-rotation
    const float * __restrict__ h_a,      // [a_dim, a_dim] row-major
    uint8_t     * __restrict__ a_fp8,    // [M, K] row-major
    float       * __restrict__ a_scale,  // [M]
    int K,
    int a_dim,
    int b_dim,
    int M_valid) {

    extern __shared__ float s_z[];       // K floats: slice a at s_z[a*b_dim ..]
    __shared__ float s_red[1024];        // absmax reduce, blockDim ≤ 1024

    const int m   = blockIdx.x;
    const int tid = threadIdx.x;         // lane l in [0, b_dim)

    uint8_t * row_out = a_fp8 + (size_t) m * (size_t) K;

    if (m >= M_valid) {
        for (int k = tid; k < K; k += b_dim) {
            row_out[k] = 0;
        }
        if (tid == 0) {
            a_scale[m] = ML8_ACT_SCALE_EPS;
        }
        return;
    }

    const float * row_in = x + (size_t) m * (size_t) K;
    for (int k = tid; k < K; k += b_dim) {
        s_z[k] = row_in[k];
    }
    __syncthreads();

    // FWHT per a-slice. Read both pair elements for every slice into
    // registers before any write (the read/sync/write/sync schedule of
    // mt_turbo_fp8_fwht_kernel, with the slice loop hoisted inside).
    float r_new[16];                     // a_dim ≤ 16, gated by can_fuse
    for (int stride = 1; stride < b_dim; stride <<= 1) {
        const int partner = tid ^ stride;
        for (int a = 0; a < a_dim; a++) {
            const float v = s_z[a * b_dim + tid];
            const float p = s_z[a * b_dim + partner];
            r_new[a] = ((tid & stride) == 0) ? (v + p) : (p - v);
        }
        __syncthreads();
        for (int a = 0; a < a_dim; a++) {
            s_z[a * b_dim + tid] = r_new[a];
        }
        __syncthreads();
    }

    // Normalize, then Y[k][l] = sum_i H_a[i, k] * Z[i][l] (per-thread lane).
    const float inv_sqrt_b = rsqrtf((float) b_dim);
    float z_col[16];
    for (int i = 0; i < a_dim; i++) {
        z_col[i] = s_z[i * b_dim + tid] * inv_sqrt_b;
    }
    float y_col[16];
    float local_max = 0.0f;
    for (int k = 0; k < a_dim; k++) {
        float s = 0.0f;
        for (int i = 0; i < a_dim; i++) {
            s += h_a[i * a_dim + k] * z_col[i];
        }
        y_col[k] = s;
        local_max = fmaxf(local_max, fabsf(s));
    }

    s_red[tid] = local_max;
    __syncthreads();
    for (int off = b_dim / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        }
        __syncthreads();
    }

    const float scale     = fmaxf(s_red[0] * (1.0f / ML8_FP8_E4M3_MAX), ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    if (tid == 0) {
        a_scale[m] = scale;
    }

    for (int k = 0; k < a_dim; k++) {
        row_out[k * b_dim + tid] = ml8_fp32_to_e4m3(y_col[k] * inv_scale);
    }
}

// ─────────────────────────────────────────────────────────────────────
// MAD-3xx — fused block_hadamard rotation + per-row quantize (sibling of
// ml8_fused_rot_quant_kernel above, with the H_a^T left-multiply stage
// removed). block_hadamard (Q = I_a ⊗ H_b) has no cross-a_dim mixing, so
// unlike the kronecker kernel this does NOT need a per-thread a_dim-sized
// register array (which would have to be sized for the largest a_dim seen
// in practice, e.g. 38 or 98 for the 4864/12544-wide K-split shards) — every
// output element only ever depends on its own b_dim-wide slice. That lets
// this kernel support ANY a_dim, at the cost of doing the FWHT one a-slice
// at a time (a_dim * 2*log2(b_dim) syncthreads instead of the kronecker
// kernel's 2*log2(b_dim) syncs shared across all slices) — a deliberate
// trade of some sync overhead for no register-array bound. Replaces the
// unfused chain memcpy(z) → mt_turbo_fp8_fwht → [pad] → quantize with one
// launch, same as the kronecker fused kernel does for its case.
//
// blockDim.x = b_dim (pow2, 16..1024); dynamic LDS = K fp32 (same budget
// gate as the kronecker kernel — see ggml_cuda_op_fp8_quant_rot's dispatch).
// Rows m >= M_valid are GEMM padding: zero fp8 + eps scale, src not read.
static __global__ void ml8_fused_blockhad_quant_kernel(
    const float * __restrict__ x,        // [M_valid, K] row-major, pre-rotation
    uint8_t     * __restrict__ a_fp8,    // [M, K] row-major
    float       * __restrict__ a_scale,  // [M]
    int K,
    int a_dim,
    int b_dim,
    int M_valid) {

    extern __shared__ float s_z[];       // K floats: slice a at s_z[a*b_dim ..]
    __shared__ float s_red[1024];        // absmax reduce, blockDim <= 1024

    const int m   = blockIdx.x;
    const int tid = threadIdx.x;         // lane l in [0, b_dim)

    uint8_t * row_out = a_fp8 + (size_t) m * (size_t) K;

    if (m >= M_valid) {
        for (int k = tid; k < K; k += b_dim) {
            row_out[k] = 0;
        }
        if (tid == 0) {
            a_scale[m] = ML8_ACT_SCALE_EPS;
        }
        return;
    }

    const float * row_in = x + (size_t) m * (size_t) K;
    for (int k = tid; k < K; k += b_dim) {
        s_z[k] = row_in[k];
    }
    __syncthreads();

    // FWHT, one a-slice at a time (see the kernel-level comment above for
    // why this trades sync count for no a_dim register-array bound). Each
    // slice's butterfly is the exact same pairing/assignment/stage order as
    // mt_turbo_fp8_fwht_kernel and ml8_fused_rot_quant_kernel's per-a loop.
    for (int a = 0; a < a_dim; a++) {
        float * slice = s_z + a * b_dim;
        for (int stride = 1; stride < b_dim; stride <<= 1) {
            const int partner = tid ^ stride;
            const float v = slice[tid];
            const float p = slice[partner];
            const float newval = ((tid & stride) == 0) ? (v + p) : (p - v);
            __syncthreads();
            slice[tid] = newval;
            __syncthreads();
        }
    }

    // Normalize, then per-row absmax over the WHOLE rotated row (all a_dim
    // slices) — no H_a mixing, so we read straight out of s_z.
    const float inv_sqrt_b = rsqrtf((float) b_dim);
    float local_max = 0.0f;
    for (int a = 0; a < a_dim; a++) {
        const float v = s_z[a * b_dim + tid] * inv_sqrt_b;
        s_z[a * b_dim + tid] = v;
        local_max = fmaxf(local_max, fabsf(v));
    }

    s_red[tid] = local_max;
    __syncthreads();
    for (int off = b_dim / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        }
        __syncthreads();
    }

    const float scale     = fmaxf(s_red[0] * (1.0f / ML8_FP8_E4M3_MAX), ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    if (tid == 0) {
        a_scale[m] = scale;
    }

    for (int a = 0; a < a_dim; a++) {
        row_out[a * b_dim + tid] = ml8_fp32_to_e4m3(s_z[a * b_dim + tid] * inv_scale);
    }
}

// ─────────────────────────────────────────────────────────────────────
// GGML_OP_ML8_MUL_MAT HIP dispatch.
// ─────────────────────────────────────────────────────────────────────

#ifdef GGML_HIP_AITER
// Shared core for the plain and fused ML8_MUL_MAT dispatch. `x` is the fp32
// activation input; when `h_a` is non-null, `x` is the PRE-rotation tensor
// (rot->src[0]) and the fused rotation+quantize prologue runs instead of the
// plain quantize (G.6.d). Shape/gate validation for the fused case happens
// in ggml_cuda_ml8_can_fuse_rot_mm before the graph picks this path.
// ---------------------------------------------------------------------------
// ML8_4 prefill expander prefetch (see the call site in ml8_mul_mat_core).
//
// Two device buffers per GPU (B_shuf fp8 [N*K] + b_scale [N], sized to the
// largest weight seen), a side stream, and per-slot events:
//   ready[s] -- recorded on the side stream after the expand into slot s
//   done[s]  -- recorded on the compute stream after the GEMM that read slot s
// Successor prediction: next_of[w] = the prefill ML8_4 weight that followed w
// last time (the graph order is static per ubatch, so this is exact from the
// second ubatch on; a miss just expands synchronously on the compute stream).
// Acquire(w): if slot s was prefetched for w -> compute stream waits ready[s];
// else expand w into the free slot on the compute stream. Release(w): record
// done[s]; look up w_next; if it has a repack in TRFEED layout, side stream
// waits done[other] then expands w_next into the other slot, records ready.
// Disabled while the compute stream is being captured into a graph (the
// cross-stream fork/join would need explicit capture plumbing) and by
// MT_ML8_4_PREFETCH=0.
// ---------------------------------------------------------------------------
struct ml8_expand_prefetch_slot {
    uint8_t *      b_shuf  = nullptr;
    float *        b_scale = nullptr;
    size_t         cap_nk  = 0;      // bytes allocated for b_shuf
    size_t         cap_n   = 0;      // floats allocated for b_scale
    const void *   w       = nullptr; // weight prefetched into this slot (nullptr = none)
    bool           pending = false;   // expand enqueued on side stream, ready[] valid
    cudaEvent_t    ready   = nullptr;
    cudaEvent_t    done    = nullptr;
    bool           done_valid = false;
};

// What a prefill ML8_4 GEMM needs to expand its weight: the (weight, LUT
// slice, packed layout) triple. cent_data is per NODE (lut_group_off under
// TP), so it is recorded per call, not derived from the weight.
struct ml8_expand_prefetch_src {
    const ggml_tensor *         w         = nullptr;
    const uint8_t *             cent_data = nullptr;
    const ml8_weight_repack_t * repack    = nullptr;
    int                         N = 0, K = 0;
};

struct ml8_expand_prefetch_state {
    int                       device = -1;
    cudaStream_t              side   = nullptr;
    ml8_expand_prefetch_slot  slot[2];
    const void *              last_w = nullptr;
    std::unordered_map<const void *, ml8_expand_prefetch_src> next_of;
};

static ml8_expand_prefetch_state * ml8_expand_prefetch_get(int device) {
    static const bool disabled = [] {
        const char * e = std::getenv("MT_ML8_4_PREFETCH");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    if (disabled) {
        return nullptr;
    }
    static ml8_expand_prefetch_state states[GGML_CUDA_MAX_DEVICES];
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);
    ml8_expand_prefetch_state & st = states[device];
    if (st.device < 0) {
        st.device = device;
        CUDA_CHECK(cudaStreamCreateWithFlags(&st.side, cudaStreamNonBlocking));
        for (auto & sl : st.slot) {
            CUDA_CHECK(cudaEventCreateWithFlags(&sl.ready, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&sl.done,  cudaEventDisableTiming));
        }
    }
    return &st;
}

static void ml8_expand_prefetch_reserve(ml8_expand_prefetch_slot & sl, size_t nk, size_t n) {
    if (sl.cap_nk < nk) {
        if (sl.b_shuf) { CUDA_CHECK(cudaFree(sl.b_shuf)); }
        CUDA_CHECK(cudaMalloc(&sl.b_shuf, nk));
        sl.cap_nk = nk;
    }
    if (sl.cap_n < n) {
        if (sl.b_scale) { CUDA_CHECK(cudaFree(sl.b_scale)); }
        CUDA_CHECK(cudaMalloc(&sl.b_scale, n * sizeof(float)));
        sl.cap_n = n;
    }
}

// Returns the slot holding w's expansion, valid on `stream` after this call.
static int ml8_expand_prefetch_acquire(
    ml8_expand_prefetch_state * st, const ggml_tensor * w, const uint8_t * cent_data,
    const ml8_weight_repack_t * repack, int N, int K, cudaStream_t stream) {
    for (int s = 0; s < 2; s++) {
        ml8_expand_prefetch_slot & sl = st->slot[s];
        if (sl.w == w && sl.pending) {
            CUDA_CHECK(cudaStreamWaitEvent(stream, sl.ready, 0));
            sl.pending = false;
            return s;
        }
    }
    // miss: pick the slot not holding last_w's expansion... any slot whose
    // pending expand we won't need. Prefer a slot with no pending expand.
    int s = 0;
    if (st->slot[0].pending && !st->slot[1].pending) { s = 1; }
    else if (st->slot[0].pending && st->slot[1].pending) {
        // both pending (prediction went wrong twice): drain the side stream
        // ordering by waiting on both readies, reuse slot 0
        CUDA_CHECK(cudaStreamWaitEvent(stream, st->slot[0].ready, 0));
        CUDA_CHECK(cudaStreamWaitEvent(stream, st->slot[1].ready, 0));
        st->slot[0].pending = st->slot[1].pending = false;
    }
    ml8_expand_prefetch_slot & sl = st->slot[s];
    // the compute stream is FIFO, so any earlier GEMM that read this slot is
    // ordered before this expand; a pending side-stream expand into it is
    // ordered by the ready wait above.
    if (sl.pending) { CUDA_CHECK(cudaStreamWaitEvent(stream, sl.ready, 0)); sl.pending = false; }
    ml8_expand_prefetch_reserve(sl, (size_t) N * (size_t) K, (size_t) N);
    const hipError_t exp_rc = rdna4_expand_ml84_to_trfeed(
        (const uint8_t *) repack->b_packed, cent_data, (const float *) repack->b_scale,
        N, K, sl.b_shuf, sl.b_scale, stream);
    GGML_ASSERT(exp_rc == hipSuccess && "rdna4_expand_ml84_to_trfeed dispatch failed");
    sl.w = w;
    return s;
}

static void ml8_expand_prefetch_release(
    ml8_expand_prefetch_state * st, int s, const ml8_expand_prefetch_src & cur, cudaStream_t stream) {
    const ggml_tensor * w = cur.w;
    ml8_expand_prefetch_slot & sl = st->slot[s];
    CUDA_CHECK(cudaEventRecord(sl.done, stream));
    sl.done_valid = true;

    // learn the successor of the previous weight, then predict ours
    if (st->last_w != nullptr && st->last_w != w) {
        st->next_of[st->last_w] = cur;
    }
    st->last_w = w;

    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        return;
    }
    auto it = st->next_of.find(w);
    if (it == st->next_of.end()) {
        return;
    }
    const ml8_expand_prefetch_src & nx = it->second;
    const ggml_tensor * wn = nx.w;
    const int o = 1 - s;
    ml8_expand_prefetch_slot & os = st->slot[o];
    if (os.pending) {
        return; // already prefetched something into the other slot
    }
    const int Nn = nx.N;
    const int Kn = nx.K;
    ml8_expand_prefetch_reserve(os, (size_t) Nn * (size_t) Kn, (size_t) Nn);
    // the side stream must not overwrite slot o before the GEMM that last
    // read it (recorded in done[o]) has finished
    if (os.done_valid) {
        CUDA_CHECK(cudaStreamWaitEvent(st->side, os.done, 0));
    }
    const hipError_t exp_rc = rdna4_expand_ml84_to_trfeed(
        (const uint8_t *) nx.repack->b_packed, nx.cent_data, (const float *) nx.repack->b_scale,
        Nn, Kn, os.b_shuf, os.b_scale, st->side);
    GGML_ASSERT(exp_rc == hipSuccess && "rdna4_expand_ml84_to_trfeed (prefetch) dispatch failed");
    CUDA_CHECK(cudaEventRecord(os.ready, st->side));
    os.w = wn;
    os.pending = true;
}

static void ml8_mul_mat_core(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst,
    const ggml_tensor *         x,
    const ggml_tensor *         h_a,
    int32_t                     a_dim,
    int32_t                     b_dim) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * cent = dst->src[1];

    GGML_ASSERT(w    != nullptr && cent != nullptr && x != nullptr);
    GGML_ASSERT(w->type    == GGML_TYPE_ML8_4);
    GGML_ASSERT(cent->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_I8);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(cent));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    // M = total columns across ALL batch dims, not just ne[1]. qwen35's ssm_out
    // feeds a 3D input [K, n_seq_tokens, n_seqs] (reshape_3d in the delta-net); with
    // M=ne[1] only the first sequence is computed and the rest are garbage — the
    // chunk-1-good / rest-explode signature. 2D inputs have ne[2]=ne[3]=1 so this is
    // unchanged. Mirrors the ml8_apply_rotation fix (n_tokens = ne[1]*ne[2]*ne[3]).
    const int32_t M = (int32_t) (x->ne[1] * x->ne[2] * x->ne[3]);

    // MAD-3xx activation fusion: x is EITHER the legacy raw fp32 activation
    // [K, M] (this function quantizes it below) OR the pre-quantized per-row
    // I8 output of ggml_fp8_quant_rot(..., G=0), [K+4, M] — bytes [0,M*K) are
    // every row's e4m3 A bytes back-to-back (row m at byte m*K), then bytes
    // [M*K, M*K+4*M) are fp32 a_scale[m] at byte 4*m (see the
    // GGML_OP_FP8_QUANT_ROT doc comment in ggml.h). The two are mutually
    // exclusive with h_a (the legacy fused-rotation path): a pre-quantized x
    // has already had its rotation applied by FP8_QUANT_ROT.
    const bool x_prequant = (x->type == GGML_TYPE_I8);
    if (x_prequant) {
        GGML_ASSERT(h_a == nullptr &&
            "pre-quantized activation path is mutually exclusive with the h_a fused-rotation path");
        GGML_ASSERT(x->ne[0] == K + 4 &&
            "pre-quantized x must be the ggml_fp8_quant_rot(..., G=0) per-row output");
    } else {
        GGML_ASSERT(x->ne[0] == K);
    }
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT((int64_t) dst->ne[1] * dst->ne[2] * dst->ne[3] == (int64_t) M);
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(dst));
    GGML_ASSERT(K % QK_ML8         == 0);
    GGML_ASSERT(N % MT_ML8_BLOCK_SIZE_N == 0);

    const int32_t group_size  = QK_ML8;
    const int32_t n_groups_k  = K / group_size;
    const int32_t n_centroids = 16;
    GGML_ASSERT(cent->ne[0] == n_centroids);
    // lut_group_off (op_params[0] on the ML8_MUL_MAT node `dst`): first
    // centroid K-group this node reads. Under tensor parallelism w holds
    // only a K-slice while cent is mirrored in full, so cent->ne[1] may
    // exceed n_groups_k — see ggml.h.
    const int32_t lut_group_off = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(lut_group_off >= 0 && (int64_t) lut_group_off + n_groups_k <= cent->ne[1]);
    const uint8_t * cent_data = (const uint8_t *) cent->data + (size_t) lut_group_off * n_centroids;

    if (h_a != nullptr) {
        GGML_ASSERT(h_a->type == GGML_TYPE_F32 && ggml_is_contiguous(h_a));
        GGML_ASSERT(a_dim > 0 && a_dim <= 16);
        GGML_ASSERT((int64_t) a_dim * (int64_t) b_dim == (int64_t) K);
        GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);
    }

    cudaStream_t stream = ctx.stream();

    // ── 1. Repack weights (cached after first call for this w).
    const ml8_weight_repack_t * repack = ggml_cuda_ml8_get_or_repack(stream, w);
    GGML_ASSERT(repack != nullptr);

    // ── 1b. M=1 GEMV path (G.6.h). Default ON after sweep landed winner
    // BN=16, K_COOP=8, LDS=0, LAYOUT=0 → 30.66 t/s decode (1.51× Triton M=16
    // path at 20.30 t/s, 60% of f16 reference 50.89 t/s). Set ML8_NO_GEMV=1
    // to disable and fall back to the Triton blockscale path (kept for A/B).
    // GEMV reads repack->b_packed as a straight [K/2,N] nibble stride --
    // only valid for the TRITON layout; RDNA4_TRFEED's B_nib is tile-shuffled
    // and falls through to the M<=32 decode-splitk branch below instead.
    static const bool ml8_no_gemv = (std::getenv("ML8_NO_GEMV") != nullptr);
    if (M == 1 && !ml8_no_gemv && h_a == nullptr && !x_prequant && repack->layout == ML8_4_LAYOUT_TRITON) {
        const bool ok = ml8_gemv_dispatch_env(
            stream,
            (const float *)   x->data,
            (const uint8_t *) repack->b_packed,
            (const float *)   repack->b_scale,
            cent_data,
            (float *)         dst->data,
            K, N, n_groups_k);
        if (ok) return;
        // fall through to Triton path if dispatch missed
    }

    // ── 2. Pad M to a multiple of the tuned tier's BLOCK_SIZE_M.
    // Pick the same config the dispatch will pick (decode for M<=16, prefill
    // otherwise) so M_pad % cfg.bm == 0 after padding. Pre-paged paths
    // (M = 1..16) align to 16; prefill (M > 16) aligns to 128.
    // RDNA4_TRFEED uses the frozen trfeed kernel's OWN tile rule instead
    // (gemm_capi.h): M<=32 -> the (32,1) decode/verify tile exactly (not a
    // round_up(M,16) -- the frozen kernel's A-tile fill is unguarded against
    // M), M>32 -> round_up(M,128) prefill tile.
    const int32_t M_pad = (repack->layout == ML8_4_LAYOUT_RDNA4_TRFEED)
        ? ((M <= 32) ? 32 : ((M + 127) / 128) * 128)
        : [&] {
              const mt_ml8_tuned_cfg pad_cfg = ml8_pick_config(M, K, N);
              return ((M + pad_cfg.bm - 1) / pad_cfg.bm) * pad_cfg.bm;
          }();

    // ── 3. Obtain fp8 activation + per-row scale at M_pad rows.
    //
    // Pre-quantized (x_prequant): x already IS the fp8+scale pair (produced
    // upstream by FP8_QUANT_ROT, G=0) — no quantize kernel, no allocation, no
    // launch at all when the caller's M is already M_pad-aligned (the common
    // prefill case). When M isn't tile-aligned (ragged prefill ubatch /
    // decode), pad into a small scratch buffer with a zero-memset + 2 D2D
    // memcpys — the same pattern ggml_cuda_op_fp8_mul_mat's per-row (G=0)
    // branch uses to pad an unpadded FP8_QUANT_ROT output for FP8_B128/
    // ML8_FP8 GEMMs; the copies here move only K+4 bytes/row (fp8+scale)
    // instead of a fresh quantize pass, and never move the pre-rotation fp32
    // activation at all (that memcpy is what this whole change eliminates).
    //
    // Legacy (!x_prequant): quantize fp32 → fp8 + per-row scale here, same as
    // before — either the h_a!=nullptr fused rotate+quantize prologue or the
    // plain quantize kernel. M-padding is folded into those kernels (rows ≥
    // M emit zero fp8 + eps scale), so no zero-padded fp32 staging copy of x
    // is needed in that branch.
    ggml_cuda_pool_alloc<uint8_t> a_fp8_scratch(ctx.pool());
    ggml_cuda_pool_alloc<float>   a_scale_scratch(ctx.pool());
    const uint8_t * a_fp8_ptr;
    const float   * a_scale_ptr;

    // G.6.g.C dump harness only covers the legacy fp32 path (it dumps the
    // pre-quant fp32 activation, which doesn't exist when x is already
    // pre-quantized).
    const int dump_i = (!x_prequant && ml8_dump_enabled()) ? g_ml8_dump_mm_n.load() : ml8_dump_limit();
    const bool dump_this = dump_i < ml8_dump_limit();
    char dump_path[128];

    if (x_prequant) {
        const uint8_t * qs_base    = (const uint8_t *) x->data;
        const float   * scale_base = (const float *) ((const uint8_t *) x->data + (size_t) M * (size_t) K);
        if (M_pad == M) {
            a_fp8_ptr   = qs_base;
            a_scale_ptr = scale_base;
        } else {
            a_fp8_scratch.alloc((size_t) M_pad * (size_t) K);
            a_scale_scratch.alloc((size_t) M_pad);
            CUDA_CHECK(cudaMemsetAsync(a_fp8_scratch.get(), 0, (size_t) M_pad * (size_t) K, stream));
            CUDA_CHECK(cudaMemsetAsync(a_scale_scratch.get(), 0, (size_t) M_pad * sizeof(float), stream));
            CUDA_CHECK(cudaMemcpyAsync(a_fp8_scratch.get(), qs_base, (size_t) M * (size_t) K,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(a_scale_scratch.get(), scale_base, (size_t) M * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            a_fp8_ptr   = a_fp8_scratch.get();
            a_scale_ptr = a_scale_scratch.get();
        }
    } else {
        a_fp8_scratch.alloc((size_t) M_pad * (size_t) K);
        a_scale_scratch.alloc((size_t) M_pad);

        const float * x_src = (const float *) x->data;

        // G.6.g.C: dump pre-quant fp32 activation that the kernel will see.
        // (The fused-rotation path never dumps: can_fuse gates on ML8_DUMP off.)
        if (dump_this) {
            const int64_t shp[2] = { (int64_t) K, (int64_t) M };
            std::snprintf(dump_path, sizeof(dump_path), "/tmp/ml8_hip_mm%d_x_prequant.bin", dump_i);
            ml8_dump_fp32(dump_path, x_src, (size_t) M * (size_t) K, stream, 2, shp);
            ml8_dump_index("mm", dump_i, dst->name, w->name, K, N, M, a_dim, b_dim, h_a != nullptr);
        }

        if (h_a != nullptr) {
            // G.6.d fused prologue: FWHT + H_a^T + quantize in one launch.
            const dim3   grid((unsigned) M_pad, 1, 1);
            const dim3   block((unsigned) b_dim, 1, 1);
            const size_t lds_bytes = (size_t) K * sizeof(float);
            ml8_fused_rot_quant_kernel<<<grid, block, lds_bytes, stream>>>(
                x_src,
                (const float *) h_a->data,
                a_fp8_scratch.get(),
                a_scale_scratch.get(),
                K, a_dim, b_dim, M);
        } else {
            ggml_cuda_ml8_quantize_activations(
                stream,
                x_src,
                a_fp8_scratch.get(),
                a_scale_scratch.get(),
                M_pad,
                K,
                M);
        }

        // G.6.g.C: dump fp8 quantized activations + per-row scale on first call.
        if (dump_this) {
            const int64_t shp_fp8[2]   = { (int64_t) K,     (int64_t) M_pad };
            const int64_t shp_scale[1] = { (int64_t) M_pad };
            std::snprintf(dump_path, sizeof(dump_path), "/tmp/ml8_hip_mm%d_a_fp8.bin", dump_i);
            ml8_dump_u8(dump_path, a_fp8_scratch.get(), (size_t) M_pad * (size_t) K, stream, 2, shp_fp8);
            std::snprintf(dump_path, sizeof(dump_path), "/tmp/ml8_hip_mm%d_a_scale.bin", dump_i);
            ml8_dump_fp32(dump_path, a_scale_scratch.get(), (size_t) M_pad, stream, 1, shp_scale);
        }

        a_fp8_ptr   = a_fp8_scratch.get();
        a_scale_ptr = a_scale_scratch.get();
    }

    // ── 4 (RDNA4_TRFEED). Default dispatch on gfx1201: decode/verify
    // (M_pad==32) via the split-K nibble-native kernel, no fp8 expansion;
    // prefill (M_pad>32) via the expander + UNCHANGED frozen fp8 trfeed
    // kernel. See gemm_capi.h's ML84_TRFEED section for both contracts.
    if (repack->layout == ML8_4_LAYOUT_RDNA4_TRFEED) {
        GGML_ASSERT(N % 128 == 0 && "ML8_4_LAYOUT_RDNA4_TRFEED requires N%128==0 "
            "(ml8_4_layout_for_tensor should have picked TRITON otherwise)");
        if (M_pad == 32) {
            // Decode/verify: fp32 output straight from the split-K kernel, no
            // bf16 intermediate (see rdna4_gemm_ml84_trfeed_decode_splitk's
            // contract). MT_ML8_4_SPLITS overrides the heuristic split count
            // for A/B; unset uses rdna4_ml84_trfeed_splitk_default_splits.
            static const int splits_override = [] {
                const char * e = std::getenv("MT_ML8_4_SPLITS");
                return e ? std::atoi(e) : 0;
            }();
            const int n_splits = splits_override > 0 ? splits_override
                                                      : rdna4_ml84_trfeed_splitk_default_splits(N, K);
            ggml_cuda_pool_alloc<float> c_pad(ctx.pool());
            float * c_ptr;
            if (M == M_pad) {
                c_ptr = (float *) dst->data;
            } else {
                c_pad.alloc((size_t) M_pad * (size_t) N);
                c_ptr = c_pad.get();
            }
            const hipError_t rc = rdna4_gemm_ml84_trfeed_decode_splitk(
                a_fp8_ptr, (const uint8_t *) repack->b_packed, cent_data,
                c_ptr, a_scale_ptr, (const float *) repack->b_scale,
                M_pad, N, K, n_splits, stream);
            GGML_ASSERT(rc == hipSuccess && "rdna4_gemm_ml84_trfeed_decode_splitk dispatch failed");
            if (c_ptr != (float *) dst->data) {
                CUDA_CHECK(cudaMemcpyAsync((float *) dst->data, c_ptr, (size_t) M * (size_t) N * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
            }
            return;
        }

        // Prefill: re-expand the 4.5bpw ML8_4 weight into the frozen fp8
        // trfeed kernel's B_shuf + per-column b_scale, then reuse
        // rdna4_gemm_fp8_trfeed UNCHANGED. Transient pool scratch -- the
        // largest weight in this model (N=17408,K=5120) is 89 MB, freed back
        // to the pool the moment this call returns.
        // Expander prefetch (2026-09-18): the expander is memory-bound (0.40 s
        // of a 7.2 s 8k prefill, serial before every compute-bound GEMM). The
        // graph's ML8_MUL_MAT order is identical every ubatch, so after the
        // first pass we know which weight follows this one and expand it on a
        // side stream while this GEMM runs (double-buffered, event-fenced).
        // MT_ML8_4_PREFETCH=0 disables (synchronous pool-scratch expand).
        ml8_expand_prefetch_state * pf = ml8_expand_prefetch_get(ctx.device);
        const uint8_t * b_shuf_ptr      = nullptr;
        const float   * b_scale_out_ptr = nullptr;
        ggml_cuda_pool_alloc<uint8_t> b_shuf(ctx.pool());
        ggml_cuda_pool_alloc<float>   b_scale_out(ctx.pool());
        int use_slot = -1;
        if (pf != nullptr) {
            use_slot = ml8_expand_prefetch_acquire(pf, w, cent_data, repack, N, K, stream);
            b_shuf_ptr      = pf->slot[use_slot].b_shuf;
            b_scale_out_ptr = pf->slot[use_slot].b_scale;
        } else {
            b_shuf.alloc((size_t) N * (size_t) K);
            b_scale_out.alloc((size_t) N);
            const hipError_t exp_rc = rdna4_expand_ml84_to_trfeed(
                (const uint8_t *) repack->b_packed, cent_data, (const float *) repack->b_scale,
                N, K, b_shuf.get(), b_scale_out.get(), stream);
            GGML_ASSERT(exp_rc == hipSuccess && "rdna4_expand_ml84_to_trfeed dispatch failed");
            b_shuf_ptr      = b_shuf.get();
            b_scale_out_ptr = b_scale_out.get();
        }

        // fp32 epilogue straight into dst (M_valid = M rows): no bf16
        // scratch, no convert launch. Measured 2026-09-18 (out/gemm_trfeed_prod_bench,
        // R9700): bit-exact vs the fp32 reference and 2-8% faster than the
        // bf16 store; the convert_unary<bf16,float> it replaces was 0.385 s
        // of a 7.2 s 8k prefill. MT_FP8_TRFEED_F32OUT=0 restores the bf16 path.
        if (!ml8_trfeed_f32out_disabled()) {
            const hipError_t gemm_rc_trfeed = rdna4_gemm_fp8_trfeed_f32(
                (const uint8_t *) a_fp8_ptr, b_shuf_ptr, (float *) dst->data, a_scale_ptr, b_scale_out_ptr,
                M_pad, M, N, K, stream);
            GGML_ASSERT(gemm_rc_trfeed == hipSuccess && "rdna4_gemm_fp8_trfeed_f32 dispatch failed");
        } else {
            ggml_cuda_pool_alloc<nv_bfloat16> c_bf16_trfeed(ctx.pool(), (size_t) M_pad * (size_t) N);
            const hipError_t gemm_rc_trfeed = rdna4_gemm_fp8_trfeed(
                a_fp8_ptr, b_shuf_ptr, c_bf16_trfeed.get(), a_scale_ptr, b_scale_out_ptr, M_pad, N, K, stream);
            GGML_ASSERT(gemm_rc_trfeed == hipSuccess && "rdna4_gemm_fp8_trfeed dispatch failed");

            const to_fp32_cuda_t bf16_to_fp32_trfeed = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
            GGML_ASSERT(bf16_to_fp32_trfeed != nullptr);
            bf16_to_fp32_trfeed(c_bf16_trfeed.get(), (float *) dst->data, (size_t) M * (size_t) N, stream);
        }
        if (pf != nullptr) {
            ml8_expand_prefetch_src cur;
            cur.w = w; cur.cent_data = cent_data; cur.repack = repack; cur.N = N; cur.K = K;
            ml8_expand_prefetch_release(pf, use_slot, cur, stream);
        }
        return;
    }

    // ── 4. Allocate bf16 output (M_pad × N) and launch mt_ml8_gemm.
    ggml_cuda_pool_alloc<nv_bfloat16> c_bf16(ctx.pool(), (size_t) M_pad * (size_t) N);

    mt_ml8_gemm_args_t args{};
    args.shape.N             = N;
    args.shape.K             = K;
    args.shape.group_size    = group_size;
    args.shape.n_centroids   = n_centroids;
    args.shape.weight_format = 1;  // ml8-4 LUT path

    args.a_fp8             = a_fp8_ptr;
    args.b_packed          = repack->b_packed;
    args.c                 = c_bf16.get();

    args.a_scale_fp32      = a_scale_ptr;
    args.b_scale_fp32      = repack->b_scale;
    args.centroid_lut_fp8  = cent_data;

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
    if (dump_this) {
        const int64_t shape[2] = { (int64_t) N, (int64_t) M };
        std::snprintf(dump_path, sizeof(dump_path), "/tmp/ml8_hip_mm%d_y_out.bin", dump_i);
        ml8_dump_fp32(dump_path, (const float *) dst->data, (size_t) M * (size_t) N, stream, 2, shape);
        g_ml8_dump_mm_n.fetch_add(1);
    }
}
#endif // GGML_HIP_AITER

void ggml_cuda_op_ml8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {
#ifndef GGML_HIP_AITER
    // ml8 inference dispatches through the AITER Triton-AOT GEMM, only built with
    // -DGGML_HIP_AITER=ON. Without it ml8 inference is unavailable (the box that
    // calibrates ml8 weights and the box that runs them can differ — gfx1201 runs).
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("ml8 mul_mat inference requires ggml-hip built with -DGGML_HIP_AITER=ON");
#else
    ml8_mul_mat_core(ctx, dst, dst->src[2], /*h_a=*/nullptr, 0, 0);
#endif // GGML_HIP_AITER
}

bool ggml_cuda_ml8_can_fuse_rot_mm(
    const ggml_tensor * rot,
    const ggml_tensor * mm) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(rot); GGML_UNUSED(mm);
    return false;
#else
    static const bool no_fuse = (std::getenv("ML8_NO_FUSE") != nullptr);
    if (no_fuse || ml8_dump_enabled()) {  // ML8_DUMP harness expects the unfused chain
        return false;
    }
    if (rot == nullptr || mm == nullptr ||
        rot->op != GGML_OP_ML8_APPLY_ROTATION || mm->op != GGML_OP_ML8_MUL_MAT ||
        mm->src[2] != rot) {
        return false;
    }
    // block_hadamard (h_a == NULL) never feeds an ML8_MUL_MAT in practice —
    // that op family is ML8_4-only, block_hadamard targets ML8_FP8's plain
    // MUL_MAT — but guard explicitly: the fused path unconditionally reads
    // rot->src[1] as H_a and would silently drop the rotation if it were
    // NULL (ml8_mul_mat_core's `if (h_a != nullptr)` quantize branch).
    if (rot->src[1] == nullptr) {
        return false;
    }
    const ggml_tensor * x = rot->src[0];
    if (x == nullptr || x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x)) {
        return false;
    }
    const int32_t * pp    = (const int32_t *) rot->op_params;
    const int32_t   a_dim = pp[0];
    const int32_t   b_dim = pp[1];
    if (a_dim <= 0 || a_dim > 16) {                          // z/y register arrays
        return false;
    }
    if (b_dim < 16 || b_dim > 1024 || (b_dim & (b_dim - 1)) != 0) {
        return false;
    }
    const int64_t K = (int64_t) a_dim * (int64_t) b_dim;
    if (x->ne[0] != K || mm->src[0] == nullptr || mm->src[0]->ne[0] != K) {
        return false;
    }
    // dynamic K-fp32 LDS + the kernel's static 4KB reduce array must fit.
    if (K * sizeof(float) + 1024 * sizeof(float) > 64 * 1024) {
        return false;
    }
    // M == 1 decode keeps the unfused GEMV fast path.
    const int64_t M = x->ne[1] * x->ne[2] * x->ne[3];
    return M > 1;
#endif // GGML_HIP_AITER
}

void ggml_cuda_op_ml8_mul_mat_fused(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor *         rot,
    ggml_tensor *               dst) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(ctx); GGML_UNUSED(rot); GGML_UNUSED(dst);
    GGML_ABORT("ml8 mul_mat inference requires ggml-hip built with -DGGML_HIP_AITER=ON");
#else
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        fprintf(stderr, "[ml8-fuse] rotation+mul_mat fusion ACTIVE (first hit: %s)\n",
                dst->name);
    }
    const int32_t * pp = (const int32_t *) rot->op_params;
    ml8_mul_mat_core(ctx, dst, rot->src[0], rot->src[1], pp[0], pp[1]);
#endif // GGML_HIP_AITER
}

// ─────────────────────────────────────────────────────────────────────
// MAD-305 fused-FFN task (2026-09-18): {ML8_MUL_MAT(gate), ML8_MUL_MAT(up),
// GLU(swiglu)} -> one fused GEMM (gemm_capi.h's rdna4_gemm_fp8_trfeed_swiglu_f32
// over rdna4_expand_ml84_pair_to_trfeed's fused B_shuf). See ml8.cuh for the
// full contract; this mirrors ggml_cuda_ml8_can_fuse_rot_mm /
// ggml_cuda_op_ml8_mul_mat_fused's split (cheap boolean gate + a separate
// execute function) one node-pattern up.
// ─────────────────────────────────────────────────────────────────────

bool ggml_cuda_ml8_can_fuse_ffn_swiglu(
    const ggml_tensor * mm_gate,
    const ggml_tensor * mm_up,
    const ggml_tensor * glu) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(mm_gate); GGML_UNUSED(mm_up); GGML_UNUSED(glu);
    return false;
#else
    // Default OFF (2026-09-18): the fused kernel is exact but measures a wash on the R9700 --
    // 5.07 ms vs 4.72 (two GEMMs) + 0.28 (GLU kernel) at M=2048/N_half=17408/K=5120 -- because
    // the SwiGLU epilogue costs +42 VGPRs (occupancy 9 -> 7 waves/SIMD). Enable with
    // MT_ML8_FFN_FUSE=1; revisit with the bf16-output epilogue (halves the store bytes).
    static const bool no_fuse = [] {
        const char * e = std::getenv("MT_ML8_FFN_FUSE");
        return e == nullptr || std::strcmp(e, "1") != 0;
    }();
    if (no_fuse || ml8_dump_enabled()) {   // ML8_DUMP harness expects the unfused chain
        return false;
    }
    if (mm_gate == nullptr || mm_up == nullptr || glu == nullptr) {
        return false;
    }
    if (mm_gate->op != GGML_OP_ML8_MUL_MAT || mm_up->op != GGML_OP_ML8_MUL_MAT || glu->op != GGML_OP_GLU) {
        return false;
    }
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU) {
        return false;
    }
    if (ggml_get_op_params_i32(glu, 1) != 0) {   // swapped
        return false;
    }
    // ggml_swiglu(a,b) always builds src0=a (SiLU'd), src1=b (multiplied) --
    // the caller (ggml-cuda.cu's fusion match) is responsible for handing
    // this function mm_gate/mm_up in the order that matches glu's actual
    // src0/src1 (the graph may emit the two ML8_MUL_MAT nodes in either
    // index order relative to which is gate vs up); this function does not
    // itself try both orders, it just verifies the one it was given.
    if (glu->src[0] != mm_gate || glu->src[1] != mm_up) {
        return false;
    }
    const ggml_tensor * x = mm_gate->src[2];
    if (x == nullptr || x != mm_up->src[2]) {
        return false;   // must be literally the same shared activation tensor
    }
    if (x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_I8) {
        return false;
    }
    if (!ggml_is_contiguous(x)) {
        return false;
    }
    const ggml_tensor * w_gate = mm_gate->src[0];
    const ggml_tensor * w_up   = mm_up->src[0];
    const ggml_tensor * c_gate = mm_gate->src[1];
    const ggml_tensor * c_up   = mm_up->src[1];
    if (w_gate == nullptr || w_up == nullptr || c_gate == nullptr || c_up == nullptr) {
        return false;
    }
    if (w_gate->type != GGML_TYPE_ML8_4 || w_up->type != GGML_TYPE_ML8_4) {
        return false;
    }
    if (w_gate->ne[0] != w_up->ne[0] || w_gate->ne[1] != w_up->ne[1]) {
        return false;   // K, N must match between gate and up
    }
    const int32_t K      = (int32_t) w_gate->ne[0];
    const int32_t N_half = (int32_t) w_gate->ne[1];
    if (K <= 0 || K % QK_ML8 != 0 || N_half <= 0 || N_half % 64 != 0) {
        return false;
    }
    // Same lut_group_off (op_params[0]) -- fusing across mismatched TP
    // K-slice offsets would silently mix centroid tables.
    if (ggml_get_op_params_i32(mm_gate, 0) != ggml_get_op_params_i32(mm_up, 0)) {
        return false;
    }
    // Both weights must land in the RDNA4_TRFEED layout at this N (a TP
    // N-slice not a multiple of 128 packs TRITON instead and this fusion
    // does not apply -- ml8_4_layout_for_tensor is the same pure function
    // ggml_cuda_ml8_get_or_repack itself consults, so this is exactly the
    // layout the repack will actually produce, not a guess).
    if (ml8_4_layout_for_tensor(N_half) != ML8_4_LAYOUT_RDNA4_TRFEED) {
        return false;
    }
    // M_pad rule mirrors ml8_mul_mat_core's RDNA4_TRFEED branch: M<=32 stays
    // the decode/verify split-K path (never fused here), M>32 is the 128-wide
    // prefill tile this fusion targets.
    const int64_t M = x->ne[1] * x->ne[2] * x->ne[3];
    if (M <= 32) {
        return false;
    }
    return true;
#endif // GGML_HIP_AITER
}

void ggml_cuda_op_ml8_ffn_gate_up_swiglu(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor *         mm_gate,
    const ggml_tensor *         mm_up,
    ggml_tensor *               glu_dst) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(ctx); GGML_UNUSED(mm_gate); GGML_UNUSED(mm_up); GGML_UNUSED(glu_dst);
    GGML_ABORT("ml8 mul_mat inference requires ggml-hip built with -DGGML_HIP_AITER=ON");
#else
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        fprintf(stderr, "[ml8-fuse] ffn gate/up swiglu fusion ACTIVE (first hit: %s)\n", glu_dst->name);
    }

    const ggml_tensor * w_gate = mm_gate->src[0];
    const ggml_tensor * c_gate = mm_gate->src[1];
    const ggml_tensor * w_up   = mm_up->src[0];
    const ggml_tensor * c_up   = mm_up->src[1];
    const ggml_tensor * x      = mm_gate->src[2];

    GGML_ASSERT(w_gate->type == GGML_TYPE_ML8_4 && w_up->type == GGML_TYPE_ML8_4);
    GGML_ASSERT(c_gate->type == GGML_TYPE_F8_E4M3 && c_up->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_I8);
    GGML_ASSERT(ggml_is_contiguous(w_gate) && ggml_is_contiguous(w_up));
    GGML_ASSERT(ggml_is_contiguous(x) && ggml_is_contiguous(glu_dst));

    const int32_t K      = (int32_t) w_gate->ne[0];
    const int32_t N_half = (int32_t) w_gate->ne[1];
    GGML_ASSERT(w_up->ne[0] == K && w_up->ne[1] == N_half);
    const int32_t M = (int32_t) (x->ne[1] * x->ne[2] * x->ne[3]);

    const bool x_prequant = (x->type == GGML_TYPE_I8);
    if (x_prequant) {
        GGML_ASSERT(x->ne[0] == K + 4 &&
            "pre-quantized x must be the ggml_fp8_quant_rot(..., G=0) per-row output");
    } else {
        GGML_ASSERT(x->ne[0] == K);
    }
    GGML_ASSERT(glu_dst->type == GGML_TYPE_F32);
    GGML_ASSERT((int64_t) glu_dst->ne[0] == N_half);
    GGML_ASSERT((int64_t) glu_dst->ne[1] * glu_dst->ne[2] * glu_dst->ne[3] == (int64_t) M);

    const int32_t group_size  = QK_ML8;
    const int32_t n_groups_k  = K / group_size;
    const int32_t n_centroids = 16;
    GGML_ASSERT(c_gate->ne[0] == n_centroids && c_up->ne[0] == n_centroids);
    const int32_t lut_group_off = ggml_get_op_params_i32(mm_gate, 0);
    GGML_ASSERT(lut_group_off == ggml_get_op_params_i32(mm_up, 0));
    GGML_ASSERT(lut_group_off >= 0 && (int64_t) lut_group_off + n_groups_k <= c_gate->ne[1]
                                    && (int64_t) lut_group_off + n_groups_k <= c_up->ne[1]);
    const uint8_t * cent_gate_data = (const uint8_t *) c_gate->data + (size_t) lut_group_off * n_centroids;
    const uint8_t * cent_up_data   = (const uint8_t *) c_up->data   + (size_t) lut_group_off * n_centroids;

    cudaStream_t stream = ctx.stream();

    const ml8_weight_repack_t * repack_gate = ggml_cuda_ml8_get_or_repack(stream, w_gate);
    const ml8_weight_repack_t * repack_up   = ggml_cuda_ml8_get_or_repack(stream, w_up);
    GGML_ASSERT(repack_gate != nullptr && repack_up != nullptr);
    GGML_ASSERT(repack_gate->layout == ML8_4_LAYOUT_RDNA4_TRFEED && repack_up->layout == ML8_4_LAYOUT_RDNA4_TRFEED &&
        "ggml_cuda_ml8_can_fuse_ffn_swiglu should have already rejected a non-RDNA4_TRFEED weight");

    // M padding: same rule as ml8_mul_mat_core's RDNA4_TRFEED prefill branch
    // (M>32 -> round_up(M,128); this fusion never sees M<=32, see
    // ggml_cuda_ml8_can_fuse_ffn_swiglu).
    GGML_ASSERT(M > 32);
    const int32_t M_pad = ((M + 127) / 128) * 128;

    // Shared activation: quantize/pad ONCE for both gate and up -- the
    // un-fused chain pays this cost TWICE (once inside each
    // ml8_mul_mat_core call on the same x), so this fusion also removes
    // that redundant work, not only the two GEMM stores + GLU pass.
    ggml_cuda_pool_alloc<uint8_t> a_fp8_scratch(ctx.pool());
    ggml_cuda_pool_alloc<float>   a_scale_scratch(ctx.pool());
    const uint8_t * a_fp8_ptr;
    const float   * a_scale_ptr;
    if (x_prequant) {
        const uint8_t * qs_base    = (const uint8_t *) x->data;
        const float   * scale_base = (const float *) ((const uint8_t *) x->data + (size_t) M * (size_t) K);
        if (M_pad == M) {
            a_fp8_ptr   = qs_base;
            a_scale_ptr = scale_base;
        } else {
            a_fp8_scratch.alloc((size_t) M_pad * (size_t) K);
            a_scale_scratch.alloc((size_t) M_pad);
            CUDA_CHECK(cudaMemsetAsync(a_fp8_scratch.get(), 0, (size_t) M_pad * (size_t) K, stream));
            CUDA_CHECK(cudaMemsetAsync(a_scale_scratch.get(), 0, (size_t) M_pad * sizeof(float), stream));
            CUDA_CHECK(cudaMemcpyAsync(a_fp8_scratch.get(), qs_base, (size_t) M * (size_t) K,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(a_scale_scratch.get(), scale_base, (size_t) M * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            a_fp8_ptr   = a_fp8_scratch.get();
            a_scale_ptr = a_scale_scratch.get();
        }
    } else {
        a_fp8_scratch.alloc((size_t) M_pad * (size_t) K);
        a_scale_scratch.alloc((size_t) M_pad);
        ggml_cuda_ml8_quantize_activations(
            stream, (const float *) x->data, a_fp8_scratch.get(), a_scale_scratch.get(), M_pad, K, M);
        a_fp8_ptr   = a_fp8_scratch.get();
        a_scale_ptr = a_scale_scratch.get();
    }

    // Dual-source expand: gate+up ML84_TRFEED -> ONE fused B_shuf + fused
    // per-column b_scale (rdna4_expand_ml84_pair_to_trfeed, gemm_ml84_prod.hip).
    // No prefetch double-buffering for this path yet (unlike
    // ml8_mul_mat_core's single-weight prefill path): the existing
    // ml8_expand_prefetch_* machinery is keyed on one weight per slot, and
    // this expander reads two per call -- wiring a pair-aware prefetch is
    // left to a follow-up (noted in the task report), not done here.
    const int32_t N_fused = 2 * N_half;
    ggml_cuda_pool_alloc<uint8_t> b_shuf_fused(ctx.pool(), (size_t) N_fused * (size_t) K);
    ggml_cuda_pool_alloc<float>   b_scale_fused(ctx.pool(), (size_t) N_fused);
    const hipError_t exp_rc = rdna4_expand_ml84_pair_to_trfeed(
        (const uint8_t *) repack_gate->b_packed, cent_gate_data, (const float *) repack_gate->b_scale,
        (const uint8_t *) repack_up->b_packed,   cent_up_data,   (const float *) repack_up->b_scale,
        N_half, K, b_shuf_fused.get(), b_scale_fused.get(), stream);
    GGML_ASSERT(exp_rc == hipSuccess && "rdna4_expand_ml84_pair_to_trfeed dispatch failed");

    // ONE fused GEMM: silu(gate)*up straight into glu_dst->data, fp32
    // [M, N_half] -- no intermediate GEMM output, no separate GLU kernel.
    const hipError_t gemm_rc = rdna4_gemm_fp8_trfeed_swiglu_f32(
        a_fp8_ptr, b_shuf_fused.get(), (float *) glu_dst->data, a_scale_ptr, b_scale_fused.get(),
        M_pad, M, N_fused, K, stream);
    GGML_ASSERT(gemm_rc == hipSuccess && "rdna4_gemm_fp8_trfeed_swiglu_f32 dispatch failed");
#endif // GGML_HIP_AITER
}

// ─────────────────────────────────────────────────────────────────────
// GGML_OP_ML8_GET_ROWS — native 4-bit token-embedding gather.
//
// One CUDA block per gathered row; threads stride over the row's K-groups.
// For each group: read the per-block fp32 scale + 32 packed nibbles from the
// native block_ml8_4 layout, index the shared per-group centroid LUT (16 fp8
// e4m3 each), dequant = centroid * scale, write K fp32. No AITER GEMM, no
// repack — this is a pure gather so it works on any CUDA/HIP build. Mirrors
// the CPU ggml_compute_forward_ml8_get_rows math exactly (same LUT, same
// lo-nibble-first ordering, same e4m3→fp32 helper).
// ─────────────────────────────────────────────────────────────────────
static __global__ void ml8_get_rows_kernel(
    const block_ml8_4 * __restrict__ w,    // [N rows][n_groups_k blocks] native layout
    const uint8_t     * __restrict__ lut,  // [n_groups_k, 16] fp8 e4m3 (flat g*16+i)
    const int32_t     * __restrict__ ids,  // [nr] contiguous
    float             * __restrict__ y,    // [nr, K] row-major (K contiguous per row)
    int K, int N, int n_groups_k, int64_t nr) {

    const int64_t i = blockIdx.x;          // gathered-row index
    if (i >= nr) return;

    const int32_t row = ids[i];
    // out-of-range ids would read garbage rows; clamp defensively to 0.
    const int32_t row_safe = (row >= 0 && row < N) ? row : 0;

    const block_ml8_4 * w_row = w + (int64_t) row_safe * n_groups_k;
    float             * y_row = y + i * (int64_t) K;

    for (int g = threadIdx.x; g < n_groups_k; g += blockDim.x) {
        const block_ml8_4 * blk   = &w_row[g];
        const float         scale = blk->scale;
        const uint8_t     * lut_g = lut + (int64_t) g * 16;
        const int           k_base = g * QK_ML8;
        #pragma unroll
        for (int p = 0; p < QK_ML8 / 2; p++) {
            const uint8_t byte = blk->qs[p];
            const uint8_t lo   = byte & 0x0F;
            const uint8_t hi   = (byte >> 4) & 0x0F;
            y_row[k_base + p * 2]     = ml8_fp8_e4m3_to_fp32(lut_g[lo]) * scale;
            y_row[k_base + p * 2 + 1] = ml8_fp8_e4m3_to_fp32(lut_g[hi]) * scale;
        }
    }
}

// Packed-layout sibling of ml8_get_rows_kernel: row n is column n of the
// nibble matrix [K/2, N]; threads stride K.
static __global__ void ml8_packed_get_rows_kernel(
    const uint8_t * __restrict__ b_packed,  // (K/2, N)
    const float   * __restrict__ b_scale,   // (n_groups_k, N)
    const uint8_t * __restrict__ lut,       // [n_groups_k, 16]
    const int32_t * __restrict__ ids,
    float         * __restrict__ y,         // [nr, K]
    int K, int N, int n_groups_k, int64_t nr) {

    const int64_t i = blockIdx.x;
    if (i >= nr) return;
    const int32_t row = ids[i];
    const int32_t n   = (row >= 0 && row < N) ? row : 0;
    float * y_row = y + i * (int64_t) K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const int     g    = k / QK_ML8;
        const uint8_t byte = b_packed[(size_t) (k / 2) * (size_t) N + (size_t) n];
        const uint8_t idx  = (k & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
        y_row[k] = ml8_fp8_e4m3_to_fp32(lut[(int64_t) g * 16 + idx]) * b_scale[(size_t) g * (size_t) N + (size_t) n];
    }
    GGML_UNUSED(n_groups_k);
}

void ggml_cuda_op_ml8_get_rows(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * cent = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(w != nullptr && cent != nullptr && ids != nullptr);
    GGML_ASSERT(w->type    == GGML_TYPE_ML8_4);
    GGML_ASSERT(cent->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(cent));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    GGML_ASSERT(K % QK_ML8 == 0);
    const int32_t n_groups_k = K / QK_ML8;
    GGML_ASSERT(cent->ne[0] == 16);
    GGML_ASSERT(cent->ne[1] == n_groups_k);
    GGML_ASSERT(dst->ne[0] == K);

    const int64_t nr = ggml_nelements(ids);
    GGML_ASSERT(ggml_nrows(dst) == nr);
    if (nr == 0) {
        return;
    }

    cudaStream_t stream = ctx.stream();

    const block_ml8_4 * w_d   = (const block_ml8_4 *) w->data;
    const uint8_t     * lut_d = (const uint8_t     *) cent->data;
    const int32_t     * ids_d = (const int32_t     *) ids->data;
    float             * y_d   = (float             *) dst->data;

    const int threads = (n_groups_k < 256) ? ((n_groups_k + 31) / 32) * 32 : 256;
    const dim3 grid((unsigned) nr);

    {
        // In-place packed weight (kernel layout) -> gather from the packed form.
        ml8_weight_repack_t info;
        bool packed = false;
        {
            std::lock_guard<std::mutex> lock(g_ml8_inplace_mu);
            auto it = g_ml8_inplace.find(w->data);
            if (it != g_ml8_inplace.end() && it->second.packed) {
                info = it->second.info; packed = true;
            }
        }
        if (packed && info.layout == ML8_4_LAYOUT_TRITON) {
            ml8_packed_get_rows_kernel<<<grid, dim3(256), 0, stream>>>(
                (const uint8_t *) info.b_packed, (const float *) info.b_scale, lut_d, ids_d, y_d, K, N, n_groups_k, nr);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (packed) {
            // RDNA4_TRFEED layout: ml8_packed_get_rows_kernel's straight
            // [K/2,N] stride would read the wrong bytes against a
            // tile-shuffled B_nib. Not on any hot path (token_embd is Q8_0
            // in our model; this only guards a dense ML8_4 weight that
            // happens to also be GET_ROWS'd) -- unpack once into a temp
            // on-disk-layout buffer and reuse the raw-block gather kernel.
            uint8_t * tmp = nullptr;
            CUDA_CHECK(cudaMalloc((void **) &tmp, (size_t) N * (size_t) n_groups_k * sizeof(block_ml8_4)));
            ml8_4_unpack_for_layout(stream, (const uint8_t *) info.b_packed, (const float *) info.b_scale, tmp,
                N, K, n_groups_k, info.layout);
            ml8_get_rows_kernel<<<grid, dim3(threads > 0 ? threads : 32), 0, stream>>>(
                (const block_ml8_4 *) tmp, lut_d, ids_d, y_d, K, N, n_groups_k, nr);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaFree(tmp);
            return;
        }
    }
    ml8_get_rows_kernel<<<grid, dim3(threads > 0 ? threads : 32), 0, stream>>>(
        w_d, lut_d, ids_d, y_d, K, N, n_groups_k, nr);
}

// ─────────────────────────────────────────────────────────────────────
// No-LUT FP8-WMMA mul_mat for scaled-fp8 weights (GGML_TYPE_ML8_FP8).
//
// ML8_FP8 weights are a single self-contained tensor: per-32-element fp16
// scale + raw OCP e4m3fn bytes. Unlike ML8_4 there is NO centroid sidecar,
// so this op has just src[0]=w, src[1]=x. It routes through the SAME Triton
// kernel as ML8_4 but with WEIGHT_FORMAT=0: B is the raw e4m3 weight fragment
// fed straight into tl.dot, and the per-group fp32 scale is applied in the
// fp32 epilogue (accumulator += tl.dot(a, b) * a_scale * b_scale). The M=1
// GEMV fast-path is ml8-LUT-specific and NOT used here.
// MAD Task 11.
// ─────────────────────────────────────────────────────────────────────
void ggml_cuda_op_ml8_fp8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("ml8 mul_mat inference requires ggml-hip built with -DGGML_HIP_AITER=ON");
#else
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * x = dst->src[1];

    GGML_ASSERT(w != nullptr && x != nullptr);
    GGML_ASSERT(w->type   == GGML_TYPE_ML8_FP8);
    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    // M folds in all batch dims (see the ML8_4 mul_mat above) so a 3D activation
    // [K, n_tokens, n_seqs] computes every sequence, not just the first.
    const int32_t M = (int32_t) (x->ne[1] * x->ne[2] * x->ne[3]);

    GGML_ASSERT(x->ne[0]   == K);
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT((int64_t) dst->ne[1] * dst->ne[2] * dst->ne[3] == (int64_t) M);
    GGML_ASSERT(K % QK_ML8_FP8       == 0);
    if (N % MT_ML8_BLOCK_SIZE_N != 0) {
        fprintf(stderr, "[ml8-fp8] %s: N=%d is not a multiple of %d (K=%d M=%d)\n", w->name, (int) N, MT_ML8_BLOCK_SIZE_N, (int) K, (int) M);
        fflush(stderr);
    }
    GGML_ASSERT(N % MT_ML8_BLOCK_SIZE_N == 0);

    const int32_t group_size = QK_ML8_FP8;          // 32
    const int32_t n_groups_k = K / group_size;

    cudaStream_t stream = ctx.stream();

    // ── 1. Repack weights (cached after first call for this w).
    //   b_packed = raw e4m3 [K, N]; b_scale = fp32 [n_groups_k, N].
    const ml8_weight_repack_t * repack = ggml_cuda_ml8_fp8_get_or_repack(stream, w);
    GGML_ASSERT(repack != nullptr);
    // MAD-305 Phase 5: ML8_FP8 gained a second packed layout (RDNA4 trfeed,
    // now the default -- see ml8_fp8_gemm_current_layout()). mt_ml8_gemm below
    // is the Triton WEIGHT_FORMAT=0 kernel and only understands the original
    // straight [K,N] transpose. This is the WP_ML8_FP8_LEGACY=1 plain-MUL_MAT
    // path (llama-ml8-registry.cpp) kept for A/B comparison -- it must keep
    // working under the new default layout, so read through a cached,
    // built-once [K,N] "triton view" of the weight instead of the raw
    // (possibly RDNA4-shuffled) repack->b_packed. Straight-through (no copy,
    // no cache entry) when the weight actually IS TRITON-packed
    // (MT_ML8_FP8_GEMM=triton).
    const void * b_packed_triton = ml8_fp8_triton_view(stream, w->data, repack, K, N);

    // ── 2. Pad M to a multiple of the tuned tier's BLOCK_SIZE_M (same as ML8_4).
    const mt_ml8_tuned_cfg pad_cfg = ml8_pick_config(M, K, N);
    const int32_t M_pad = ((M + pad_cfg.bm - 1) / pad_cfg.bm) * pad_cfg.bm;

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

    // ── 3. Quantize fp32 → fp8 + per-row scale (identical to ML8_4 path).
    ggml_cuda_pool_alloc<uint8_t> a_fp8(ctx.pool(),    (size_t) M_pad * (size_t) K);
    ggml_cuda_pool_alloc<float>   a_scale(ctx.pool(), (size_t) M_pad);
    ggml_cuda_ml8_quantize_activations(
        stream, x_src, a_fp8.get(), a_scale.get(), M_pad, K, /*M_valid=*/M_pad);

    // ── 4. bf16 output (M_pad × N) and launch mt_ml8_gemm with WEIGHT_FORMAT=0.
    ggml_cuda_pool_alloc<nv_bfloat16> c_bf16(ctx.pool(), (size_t) M_pad * (size_t) N);

    mt_ml8_gemm_args_t args{};
    args.shape.N             = N;
    args.shape.K             = K;
    args.shape.group_size    = group_size;
    args.shape.n_centroids   = 16;   // ignored under WF=0, but kept in the cache key
    args.shape.weight_format = 0;    // scaled-fp8 baseline (no LUT)

    args.a_fp8             = a_fp8.get();
    args.b_packed          = (void *) b_packed_triton;   // raw e4m3 [K, N] (TRITON view)
    args.c                 = c_bf16.get();

    args.a_scale_fp32      = a_scale.get();
    args.b_scale_fp32      = repack->b_scale;
    // WF=0 contract: centroid_lut_ptr is never dereferenced (the LUT branch is
    // DCE'd), but the kernel param list still binds it — pass a non-null dummy
    // (reuse b_scale) so the launcher arg-slot is valid, and stride_lut_k = 0.
    args.centroid_lut_fp8  = repack->b_scale;

    args.M                 = M_pad;

    args.stride_am         = K;  args.stride_ak       = 1;
    // WF=0 B is [K, N] (NOT [K/2, N]): stride_bk = N over full K rows.
    args.stride_bk         = N;  args.stride_bn       = 1;
    args.stride_cm         = N;  args.stride_cn       = 1;
    args.stride_ascale_m   = 1;
    args.stride_bscale_k   = N;  args.stride_bscale_n = 1;
    args.stride_lut_k      = 0;

    const hipError_t gemm_rc = mt_ml8_gemm(stream, &args);
    if (gemm_rc != hipSuccess) {
        const mt_ml8_tuned_cfg cfg = ml8_pick_config(M_pad, K, N);
        fprintf(stderr, "[ml8-fp8] mt_ml8_gemm failed: %s (M=%d M_pad=%d K=%d N=%d w=%s cfg bm=%d bn=%d gsm=%d nw=%d)\n",
                hipGetErrorString(gemm_rc), (int) M, (int) M_pad, (int) K, (int) N, w->name,
                cfg.bm, cfg.bn, cfg.gsm, cfg.nw);
        fflush(stderr);
    }
    GGML_ASSERT(gemm_rc == hipSuccess && "mt_ml8_gemm (fp8 WF=0) dispatch failed");

    // ── 5. Convert first M rows of bf16 [M_pad, N] → fp32 [M, N] into dst.
    const to_fp32_cuda_t bf16_to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
    GGML_ASSERT(bf16_to_fp32 != nullptr);
    bf16_to_fp32(c_bf16.get(), (float *) dst->data,
                 (size_t) M * (size_t) N, stream);
#endif // GGML_HIP_AITER
}

// ─────────────────────────────────────────────────────────────────────
// GGML_OP_ML8_APPLY_ROTATION HIP dispatch.
// G.4.g: original naïve O(b²) dense matmul (46 ms/call, 92% of GPU time).
// G.6.f: rewritten as row-wise FWHT + small H_a^T multiply (~100x less
//   compute on the H_b leg). H_b is the Sylvester orthogonal Hadamard, so
//   X @ H_b == row-wise FWHT(X) normalized by 1/sqrt(b_dim) — exactly what
//   mt_turbo_fp8_fwht (turbo_fp8_hadamard.cuh) produces.
// MAD-266: h_a == NULL selects block_hadamard (Q = I_a ⊗ H_b) for the
//   tensor-parallel ML8_FP8 path — the FWHT leg is unchanged, the H_a^T
//   left-multiply (and its a_dim <= 16 register-array limit) is simply
//   skipped, so block count a_dim is unbounded (e.g. 4864/128 = 38).
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
    const ggml_tensor * h_a = dst->src[1];   // NULL => block_hadamard (Q = I_a ⊗ H_b)

    GGML_ASSERT(x   != nullptr);
    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t * pp    = (const int32_t *) dst->op_params;
    const int32_t   a_dim = pp[0];
    const int32_t   b_dim = pp[1];
    const int32_t   d_dim = a_dim * b_dim;

    GGML_ASSERT(a_dim > 0 && "a_dim must be positive");
    GGML_ASSERT(b_dim > 0 && (b_dim & (b_dim - 1)) == 0 && "b_dim must be power of 2");
    GGML_ASSERT(b_dim >= 16 && b_dim <= 1024 && "b_dim must be supported by FWHT kernel (16..1024)");
    GGML_ASSERT(x->ne[0]   == d_dim);
    GGML_ASSERT(dst->ne[0] == d_dim && dst->ne[1] == x->ne[1]);

    if (h_a != nullptr) {
        GGML_ASSERT(h_a->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(h_a));
        // a_dim <= 16 only for the H_a leg: ml8_h_a_left_multiply_kernel keeps
        // a per-thread z_col[16] register array. block_hadamard skips that
        // kernel entirely, so it has no such bound (a_dim = in_features/b_dim,
        // e.g. 38 or 98 in practice).
        GGML_ASSERT(a_dim <= 16 && "a_dim must fit in z_col register array (kronecker path)");
        GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);
    }

    cudaStream_t stream = ctx.stream();
    // MAD-244: rotation is per-row; the "n_tokens" the kernel needs is the
    // total number of rows = product of all dims except ne[0]. For dense
    // input [d_dim, n_tokens] this equals ne[1]; for MoE input
    // [d_dim, n_used, n_tokens] it equals ne[1] * ne[2]. Without this
    // generalization the kernel only rotates the first ne[1] rows and leaves
    // the rest unrotated — silently corrupting MoE inference.
    const int n_tokens = (int) (x->ne[1] * x->ne[2] * x->ne[3]);
    const size_t total_elems = (size_t) n_tokens * (size_t) d_dim;

    // G.6.g.C: dump rotation input (pre-rotation activations) on first call.
    const int  rot_dump_i    = ml8_dump_enabled() ? g_ml8_dump_rot_n.load() : ml8_dump_limit();
    const bool rot_dump_this = rot_dump_i < ml8_dump_limit();
    char rot_dump_path[128];
    if (rot_dump_this) {
        const int64_t shape[2] = { (int64_t) d_dim, (int64_t) n_tokens };
        std::snprintf(rot_dump_path, sizeof(rot_dump_path), "/tmp/ml8_hip_rot%d_x_in.bin", rot_dump_i);
        ml8_dump_fp32(rot_dump_path, (const float *) x->data, total_elems, stream, 2, shape);
        ml8_dump_index("rot", rot_dump_i, dst->name, nullptr, d_dim, 0, n_tokens, a_dim, b_dim, h_a != nullptr);
    }

    // (rotation kernel runs below; output dump happens after the kernel returns)

    if (h_a == nullptr) {
        // block_hadamard: Q = I_a ⊗ H_b, no H_a leg. FWHT runs directly on a
        // copy of X in dst — no separate Z scratch buffer or left-multiply
        // kernel needed.
        CUDA_CHECK(cudaMemcpyAsync(dst->data, x->data,
            total_elems * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        CUDA_CHECK(mt_turbo_fp8_fwht(stream, (float *) dst->data,
            n_tokens * a_dim, b_dim, b_dim));
    } else {
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
    }

    // G.6.g.C: dump rotation output (post-FWHT + H_a^T) on first call.
    if (rot_dump_this) {
        const int64_t shape[2] = { (int64_t) d_dim, (int64_t) n_tokens };
        std::snprintf(rot_dump_path, sizeof(rot_dump_path), "/tmp/ml8_hip_rot%d_x_rotated.bin", rot_dump_i);
        ml8_dump_fp32(rot_dump_path, (const float *) dst->data, total_elems, stream, 2, shape);
        g_ml8_dump_rot_n.fetch_add(1);
    }
}

// ═════════════════════════════════════════════════════════════════════════
// FP8_B128 phase 2 — GGML_OP_FP8_QUANT_ROT (design 4(c)).
// ═════════════════════════════════════════════════════════════════════════

// Bit-exact port of quantize_row_f8_e4m3_ref (ggml-turbo-quant.c) — the CPU
// oracle test-backend-ops compares FP8_QUANT_ROT against byte-for-byte
// (kind=0 tolerance is ~1e-7 on the raw I8 output). Same rollover rule as
// ml8_fp32_to_e4m3 (`e_out > 15`; the CPU codec was fixed to match).
static __device__ __forceinline__ uint8_t fp8_quant_rot_f32_to_e4m3(float xv) {
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
            // Same rollover rule as the CPU reference (see comment
            // above) -- deliberately NOT the `> 15` fix ml8_fp32_to_e4m3 uses.
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
        return (uint8_t)((sign << 7) | (1u << 3));
    }
    return (uint8_t)((sign << 7) | m_e4m3);
}

// Per-128-group e4m3 quantize into the packed I8 row layout (design section
// 3(a)): row m's output bytes [0,K) are the e4m3 qs, [K, K+K/32) are the
// n_groups fp32 group scales. One block per (row, group), 128 threads (one
// per element in the group) — bit-identical math to
// ggml_compute_forward_fp8_quant_rot's per-group loop: same amax reduction
// (max is associative/exact regardless of reduction order), scale computed
// as a REAL division `amax / 448.0f` (not a multiply by a precomputed
// reciprocal — those are not bit-identical in fp32) then clamped to 1e-12,
// inv_scale = 1/scale computed once and applied by multiplication (mirroring
// the CPU reference's `scaled[i] = grp[i] * inv_scale` exactly, rather than
// dividing by scale per element), and fp8_quant_rot_f32_to_e4m3 (NOT the
// shared ml8_fp32_to_e4m3 — see its comment) for the final rounding step.
// G = scale group width (op_params[3]: 32 or 128); one block of G threads per (row, group).
template <int G>
static __global__ void fp8_quant_pack_kernel(
    const float * __restrict__ x,   // [n_rows, K] row-major, post-rotation
    int8_t      * __restrict__ y,   // [n_rows, K + 4*n_groups]
    int K, int n_groups) {

    const int row = blockIdx.x;
    const int g   = blockIdx.y;
    const int tid = threadIdx.x;    // 0..G-1

    const int row_out = K + n_groups * (int) sizeof(float);
    const float * grp = x + (size_t) row * (size_t) K + (size_t) g * G;

    __shared__ float s_red[G];
    const float v = grp[tid];
    s_red[tid] = fabsf(v);
    __syncthreads();
    #pragma unroll
    for (int off = G / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        }
        __syncthreads();
    }
    // amax / 448.0f: real division, matching ops.cpp's `amax / 448.0f`
    // exactly (a multiply by a precomputed 1/448 constant rounds
    // differently in the last bit for some amax values).
    const float scale     = fmaxf(s_red[0] / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;

    uint8_t * qs     = (uint8_t *) (y + (size_t) row * (size_t) row_out);
    float   * scales = (float *) (qs + K);
    if (tid == 0) {
        scales[g] = scale;
    }
    qs[g * G + tid] = fp8_quant_rot_f32_to_e4m3(v * inv_scale);
}

// MAD-305 Phase 5 (round 3) — GGML_OP_FP8_QUANT_ROT "per-row" mode (G=0, no
// longer an alias of 128; this is the HIP side of the contract the CPU op
// already landed). Unlike fp8_quant_pack_kernel's per-(row,group) grid (whose
// output row stride K+4*n_groups interleaves each row's own scale right
// after its own qs), this mode's output is a GLOBAL split, NOT interleaved:
// `y`'s bytes [0, n_rows*K) are every row's e4m3 A bytes back-to-back (row m
// at byte m*K), then bytes [n_rows*K, n_rows*K + 4*n_rows) are fp32
// a_scale[m] at byte 4*m -- exactly what ggml_cuda_op_fp8_mul_mat's per-row
// dispatch branch above reads.
//
// One block of THREADS=256 threads per row, float4-vectorized: K%32==0
// guarantees K%4==0 so every row start is 16-byte aligned (row byte offset
// row*K*4 is always a multiple of 16), letting both the absmax pass and the
// quantize pass load 4 fp32 at a time instead of one (rocprof showed the
// scalar version at 0.22 ms/call for M=2048 K=5120 -- ~4x a memory-bound
// op's expected time). The quantize pass ALSO packs 4 e4m3 bytes into one
// uint32_t store instead of 4 separate byte stores. Same amax-then-quantize
// math as fp8_quant_pack_kernel (real division by 448, eps-clamp, inv_scale
// multiply, fp8_quant_rot_f32_to_e4m3 rounding, associative max) so it stays
// byte-exact vs the CPU reference -- float4 changes ONLY the memory access
// granularity, not the per-element arithmetic or its ordering-independent
// reduction.
template <int THREADS>
static __global__ void fp8_quant_pack_row_kernel(
    const float * __restrict__ x,   // [n_rows, K] row-major, post-rotation
    int8_t      * __restrict__ y,   // [0,n_rows*K) qs, then [n_rows*K,+4*n_rows) a_scale
    int K, int n_rows) {

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int K4  = K / 4;
    const float4 * __restrict__ xr4 = reinterpret_cast<const float4 *>(x + (size_t) row * (size_t) K);

    __shared__ float s_red[THREADS];
    float local_max = 0.f;
    for (int k4 = tid; k4 < K4; k4 += THREADS) {
        const float4 v = xr4[k4];
        local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
    s_red[tid] = local_max;
    __syncthreads();
    #pragma unroll
    for (int off = THREADS / 2; off > 0; off >>= 1) {
        if (tid < off) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        }
        __syncthreads();
    }
    // Same real-division + eps-clamp as fp8_quant_pack_kernel (bit-identical
    // to the CPU reference's `amax / 448.0f`, not a reciprocal multiply).
    const float scale     = fmaxf(s_red[0] / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;

    if (tid == 0) {
        float * scale_ptr = (float *) (y + (size_t) n_rows * (size_t) K + (size_t) row * sizeof(float));
        *scale_ptr = scale;
    }
    uint32_t * __restrict__ qs4 = reinterpret_cast<uint32_t *>(y + (size_t) row * (size_t) K);
    for (int k4 = tid; k4 < K4; k4 += THREADS) {
        const float4 v = xr4[k4];
        const uint32_t packed =
              (uint32_t) fp8_quant_rot_f32_to_e4m3(v.x * inv_scale)
            | ((uint32_t) fp8_quant_rot_f32_to_e4m3(v.y * inv_scale) << 8)
            | ((uint32_t) fp8_quant_rot_f32_to_e4m3(v.z * inv_scale) << 16)
            | ((uint32_t) fp8_quant_rot_f32_to_e4m3(v.w * inv_scale) << 24);
        qs4[k4] = packed;
    }
}

// ─────────────────────────────────────────────────────────────────────
// MAD-3xx round 4 (V2) — single per-row (G=0) fused rotate+quantize family
// for BOTH kronecker and block_hadamard, replacing:
//   * ml8_fused_rot_quant_kernel      (kronecker, K-sized LDS)
//   * ml8_fused_blockhad_quant_kernel (block_hadamard, K-sized LDS)
//   * the generic memcpy + mt_turbo_fp8_fwht + fp8_quant_pack_row_kernel
//     three-launch chain (block_hadamard K too large for the old LDS gate)
//
// Root cause of the old kernels' 54-244 GB/s: LDS usage scales with K (the
// WHOLE row lives in shared memory), which (a) caps occupancy long before
// bandwidth saturates and (b) simply doesn't fit for K=17408 (17408*4 +
// 4096 > 64KB), forcing that shape onto the slow 3-launch generic path with
// a full extra D2D memcpy of the 42MB input.
//
// Fix: blockDim.x = b_dim * ROWS_PER_BLOCK (one thread == one (row-in-block,
// lane) pair, same lane-owns-column convention as the old fused kernels so
// the kronecker H_a^T stage stays register-local exactly as before), and
// the FWHT butterfly stages through a LDS scratch sized ONLY
// b_dim*ROWS_PER_BLOCK floats (<=1024 floats = 4KB), reused across all
// a_dim slice iterations instead of holding all a_dim slices simultaneously.
// LDS is therefore O(b_dim), NEVER O(K) -- the gate on K disappears entirely
// for this path. ROWS_PER_BLOCK = min(8, 1024/b_dim) packs multiple rows
// into one block for the b_dim=128 shapes (6144, 17408) to keep bytes/block
// (and therefore concurrent DRAM requests) high even though each row is
// "only" 24-70KB.
//
// Register footprint: each thread keeps a per-thread array of the row's
// a_dim values at its own lane (one float per slice) live across the whole
// kernel -- this is what lets quantization run without ever re-reading X or
// spilling the rotated row back to DRAM (the old generic path's second
// full-tensor round trip). MAX_A bounds this array at compile time; two
// instantiations are provided (16 and 160) so shapes with small a_dim (the
// kronecker path, and small block_hadamard a_dim) don't pay for registers
// they don't use. a_dim=136 (K=17408, the (c) shape from the problem
// statement) needs ~136 live floats/thread in the MAX_A=160 instantiation --
// a real amount of VGPR pressure that likely limits occupancy to very few
// waves/CU; this is a knowingly-untested trade (no GPU available to profile
// from this box) accepted because the alternative (spilling the row to DRAM
// between the absmax pass and the quantize pass) provably cannot hit the
// 600 GB/s effective target -- see the design-rationale comment in the
// dispatcher below for the traffic accounting.
//
// Quantize math: real division by 448 (not a reciprocal multiply) and
// fp8_quant_rot_f32_to_e4m3 (the FP8_QUANT_ROT-specific rounding function,
// NOT the shared ml8_fp32_to_e4m3 the OLD fused kernels used) -- matches
// fp8_quant_pack_row_kernel above exactly, which is the generic per-row
// path's own quantizer and therefore the thing this kernel must agree with
// for kind=NONE-adjacent behavior and for the CPU oracle's rounding rule.
// Reduction order (row-segment tree max, one slice at a time through LDS
// instead of the old kernels' whole-row-at-once butterfly) is a legitimate
// fp32 reordering vs. both the CPU reference and the old GPU kernels; the
// FP8_QUANT_ROT KRONECKER/BLOCK_HADAMARD test tolerance (1e-4 nmse on
// dequantized values, not raw bytes -- see test_fp8_quant_rot::err) exists
// exactly to absorb this.
template <int MAX_A, bool HAS_HA>
static __global__ void ml8_fp8_qrot_v2_kernel(
    const float * __restrict__ x,        // [n_rows, K] row-major, pre-rotation
    const float * __restrict__ h_a,      // [a_dim, a_dim] row-major, or nullptr when !HAS_HA
    uint8_t     * __restrict__ a_fp8,    // [n_rows, K] row-major
    float       * __restrict__ a_scale,  // [n_rows]
    int K, int a_dim, int b_dim, int n_rows, int rows_per_block) {

    // Dynamic shared mem: [0, blockDim) = per-slice butterfly scratch,
    // [blockDim, 2*blockDim) = absmax reduce scratch. Both sized to
    // blockDim.x (== b_dim*rows_per_block), never to K.
    extern __shared__ float smem[];
    float * s_slice = smem;
    float * s_red   = smem + blockDim.x;

    const int tid          = threadIdx.x;
    const int row_in_block = tid / b_dim;
    const int lane         = tid % b_dim;
    const int base         = row_in_block * b_dim;
    const int row          = blockIdx.x * rows_per_block + row_in_block;
    const bool row_valid   = row < n_rows;

    const float inv_sqrt_b = rsqrtf((float) b_dim);

    // Per-thread register-resident state: this thread's own lane across all
    // a_dim slices. Invalid (padding) rows still run the full butterfly
    // schedule (with a dummy zero input) so every thread in the block hits
    // the same sequence of __syncthreads() -- only the final write is guarded.
    float reg[MAX_A];
    for (int a = 0; a < a_dim; a++) {
        s_slice[tid] = row_valid ? x[(size_t) row * (size_t) K + (size_t) a * b_dim + lane] : 0.0f;
        __syncthreads();
        // In-place FWHT on this row's b_dim-wide segment [base, base+b_dim)
        // of s_slice -- same pairing/assignment/stage order as
        // mt_turbo_fp8_fwht_kernel (partner = lane ^ stride; lower gets
        // v+p, upper gets p-v), just done one slice at a time through a
        // b_dim-sized (not K-sized) scratch buffer.
        for (int stride = 1; stride < b_dim; stride <<= 1) {
            const int partner_tid = base + (lane ^ stride);
            const float v = s_slice[tid];
            const float p = s_slice[partner_tid];
            __syncthreads();
            s_slice[tid] = ((lane & stride) == 0) ? (v + p) : (p - v);
            __syncthreads();
        }
        reg[a] = s_slice[tid] * inv_sqrt_b;
        // No extra sync needed here: every cross-thread read of s_slice[*]
        // for this slice already happened before the last __syncthreads()
        // above, and the next iteration only ever writes s_slice[tid] (this
        // thread's own slot) before any thread reads it again.
    }

    // H_a^T left-multiply (kronecker only) or pass-through (block_hadamard):
    // same index convention as ml8_h_a_left_multiply_kernel /
    // ml8_fused_rot_quant_kernel (Y[k] = sum_i H_a[i,k] * Z[i]).
    float local_max = 0.0f;
    float y[HAS_HA ? MAX_A : 1];
    if constexpr (HAS_HA) {
        for (int k = 0; k < a_dim; k++) {
            float s = 0.0f;
            for (int i = 0; i < a_dim; i++) {
                s += h_a[i * a_dim + k] * reg[i];
            }
            y[k] = s;
            local_max = fmaxf(local_max, fabsf(s));
        }
    } else {
        for (int a = 0; a < a_dim; a++) {
            local_max = fmaxf(local_max, fabsf(reg[a]));
        }
    }

    // Per-row (per-segment) absmax tree reduction: rows are laid out as
    // contiguous, power-of-2-sized b_dim segments of tid-space, so a
    // strided tree reduction bounded by `lane < off` never crosses a row
    // boundary.
    s_red[tid] = local_max;
    __syncthreads();
    for (int off = b_dim / 2; off > 0; off >>= 1) {
        if (lane < off) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + off]);
        }
        __syncthreads();
    }

    if (!row_valid) {
        return;
    }

    // Real division (not a reciprocal multiply) + eps-clamp, matching
    // fp8_quant_pack_row_kernel / the CPU reference's `amax / 448.0f`.
    const float scale     = fmaxf(s_red[base] / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    if (lane == 0) {
        a_scale[row] = scale;
    }

    uint8_t * row_out = a_fp8 + (size_t) row * (size_t) K;
    if constexpr (HAS_HA) {
        for (int k = 0; k < a_dim; k++) {
            row_out[k * b_dim + lane] = fp8_quant_rot_f32_to_e4m3(y[k] * inv_scale);
        }
    } else {
        for (int a = 0; a < a_dim; a++) {
            row_out[a * b_dim + lane] = fp8_quant_rot_f32_to_e4m3(reg[a] * inv_scale);
        }
    }
}

// Host launcher: picks ROWS_PER_BLOCK (packs multiple rows into one block
// when b_dim is small, so bytes-in-flight per block stays high even for
// small-K shapes) and the dynamic LDS size (always O(b_dim), never O(K)).
template <int MAX_A, bool HAS_HA>
static void ml8_launch_qrot_v2(
    cudaStream_t stream,
    const float * x, const float * h_a,
    uint8_t * a_fp8, float * a_scale,
    int K, int a_dim, int b_dim, int n_rows) {
    const int rows_per_block = std::max(1, std::min(8, 1024 / b_dim));
    const int block          = b_dim * rows_per_block;
    const int grid           = (n_rows + rows_per_block - 1) / rows_per_block;
    const size_t smem_bytes  = (size_t) 2 * block * sizeof(float);
    ml8_fp8_qrot_v2_kernel<MAX_A, HAS_HA><<<grid, block, smem_bytes, stream>>>(
        x, h_a, a_fp8, a_scale, K, a_dim, b_dim, n_rows, rows_per_block);
}



// Packed hardware fp32 -> e4m3 (gfx12 v_cvt_pk_fp8_f32, RNE): validated bit-identical to
// fp8_quant_rot_f32_to_e4m3 over 4M values incl. +-448, zeros and the subnormal edges
// (2026-09-18, scratch probe cvt2.hip). Inputs are clamped to +-448 first (the row scale
// guarantees |v*inv_scale| <= 448 up to fp32 rounding; the clamp is 2 v_med3 per pair).
static __device__ __forceinline__ uint32_t fp8_quant_rot_pack4_hw(float a, float b, float c, float d) {
#if defined(__gfx1200__) || defined(__gfx1201__)
    a = fminf(fmaxf(a, -448.0f), 448.0f); b = fminf(fmaxf(b, -448.0f), 448.0f);
    c = fminf(fmaxf(c, -448.0f), 448.0f); d = fminf(fmaxf(d, -448.0f), 448.0f);
    const uint32_t lo = (uint32_t) __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
    const uint32_t hi = (uint32_t) __builtin_amdgcn_cvt_pk_fp8_f32(c, d, 0, false);
    return (lo & 0xFFFFu) | (hi << 16);
#else
    return (uint32_t) fp8_quant_rot_f32_to_e4m3(a) | ((uint32_t) fp8_quant_rot_f32_to_e4m3(b) << 8)
         | ((uint32_t) fp8_quant_rot_f32_to_e4m3(c) << 16) | ((uint32_t) fp8_quant_rot_f32_to_e4m3(d) << 24);
#endif
}
// ---------------------------------------------------------------------------
// FP8_QUANT_ROT per-row (G=0) V3 -- wave-shuffle FWHT, register-resident row.
//
// Why a third version (2026-09-18, rocprofv3 on the R9700, 2048-row ubatch):
// V2 (LDS-staged, one __syncthreads per butterfly stage per slice) measured
// 139 GB/s at K=17408 (136 slices x 7 stages x 2 barriers = 1904 barriers
// per row). The transform itself is trivial; the barriers are the cost.
//
// V3 layout: one WAVE owns one b_dim-wide Hadamard block. Lane l holds the
// E = b_dim/32 CONTIGUOUS elements [l*E, l*E+E) of that block, loaded as
// float4s (consecutive lanes -> consecutive 16E bytes -> fully coalesced).
// Butterfly stages with stride < E pair elements inside the lane's own
// register array; stages with stride >= E pair lane l with lane l^(stride/E)
// via __shfl_xor -- no LDS, no barriers. Same pairing/assignment as
// mt_turbo_fp8_fwht_kernel (lower partner gets v+p, upper gets p-v, ascending
// stride order, one 1/sqrt(b) normalize at the end), so the numerics differ
// from the LDS kernels only by fp32 reassociation.
//
//   BLOCK_HADAMARD: a workgroup of NW waves owns one row; wave w owns blocks
//     w, w+NW, ... (<= MAXAW of them) and keeps them in registers
//     (MAXAW*E floats/lane: K=17408 -> 34*4 = 136, K=6144 -> 12*4 = 48).
//     Row absmax = lane max -> wave reduce -> NW floats through LDS ->
//     quantize straight from registers. One read of x, one write of the
//     fp8 row. Zero barriers except the single absmax exchange.
//   KRONECKER: one wave per row (NW=1), all a_dim <= MAXAW blocks resident
//     (a=5, b=1024 -> 5*32 = 160 floats/lane), so the H_a^T mix across
//     blocks (Y[k] = sum_i H_a[i,k] Z[i], same index convention as
//     ml8_h_a_left_multiply_kernel) is register-local, then absmax + quant.
// Everything is compile-time bounded (b_dim, NW, MAXAW are template
// parameters) so the arrays stay in VGPRs; the dispatcher picks the
// instantiation and falls through to V2 for shapes without one.
// ---------------------------------------------------------------------------
template <int B, int NW, int MAXAW, bool HAS_HA>
__launch_bounds__(32 * NW)
static __global__ void ml8_fp8_qrot_v3_kernel(
    const float * __restrict__ x,        // [n_rows, K]
    const float * __restrict__ h_a,      // [a_dim, a_dim] or nullptr
    uint8_t     * __restrict__ a_fp8,    // [n_rows, K]
    float       * __restrict__ a_scale,  // [n_rows]
    int K, int a_dim, int n_rows) {

    constexpr int E = B / 32;            // elements per lane per block
    static_assert(E >= 1 && (E & (E - 1)) == 0, "b_dim must be a power of two >= 32");
    static_assert(!HAS_HA || NW == 1, "kronecker mixing needs the whole row in one wave");

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    // BLOCK_HADAMARD: one row per workgroup, waves stride over its blocks.
    // KRONECKER: one row per wave (NW == 1 so wave == 0 and the workgroup
    // is one wave); grid.x indexes rows in both cases.
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }
    const float * xrow = x + (size_t) row * (size_t) K;

    float v[MAXAW][E];
    float local_max = 0.0f;

    #pragma unroll
    for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;     // block index within the row
        if (a < a_dim) {
            const float * blk = xrow + (size_t) a * B + lane * E;
            #pragma unroll
            for (int e = 0; e < E; e += 4) {
                if constexpr (E >= 4) {
                    const float4 t = *reinterpret_cast<const float4 *>(blk + e);
                    v[j][e] = t.x; v[j][e + 1] = t.y; v[j][e + 2] = t.z; v[j][e + 3] = t.w;
                } else {
                    #pragma unroll
                    for (int q = 0; q < E; q++) { v[j][e + q] = blk[e + q]; }
                }
            }
            // in-lane stages: stride < E
            #pragma unroll
            for (int stride = 1; stride < E; stride <<= 1) {
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    if ((e & stride) == 0) {
                        const float lo = v[j][e];
                        const float hi = v[j][e + stride];
                        v[j][e]          = lo + hi;
                        v[j][e + stride] = lo - hi;
                    }
                }
            }
            // cross-lane stages: stride >= E -> partner lane l ^ (stride/E)
            #pragma unroll
            for (int ls = 1; ls < 32; ls <<= 1) {
                const bool upper = (lane & ls) != 0;
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const float mine = v[j][e];
                    const float p    = __shfl_xor_sync(0xffffffff, mine, ls, 32);
                    v[j][e] = upper ? (p - mine) : (mine + p);
                }
            }
            const float inv_sqrt_b = rsqrtf((float) B);
            #pragma unroll
            for (int e = 0; e < E; e++) {
                v[j][e] *= inv_sqrt_b;
                if constexpr (!HAS_HA) {
                    local_max = fmaxf(local_max, fabsf(v[j][e]));
                }
            }
        }
    }

    if constexpr (HAS_HA) {
        // Y[k] = sum_i H_a[i,k] * Z[i] per element; a_dim <= MAXAW, all
        // blocks of the row are in this wave (NW == 1).
        float y[MAXAW][E];
        #pragma unroll
        for (int k = 0; k < MAXAW; k++) {
            if (k < a_dim) {
                #pragma unroll
                for (int e = 0; e < E; e++) { y[k][e] = 0.0f; }
                #pragma unroll
                for (int i = 0; i < MAXAW; i++) {
                    if (i < a_dim) {
                        const float h = h_a[i * a_dim + k];
                        #pragma unroll
                        for (int e = 0; e < E; e++) { y[k][e] = fmaf(h, v[i][e], y[k][e]); }
                    }
                }
                #pragma unroll
                for (int e = 0; e < E; e++) { local_max = fmaxf(local_max, fabsf(y[k][e])); }
            }
        }
        #pragma unroll
        for (int k = 0; k < MAXAW; k++) {
            #pragma unroll
            for (int e = 0; e < E; e++) { v[k][e] = y[k][e]; }
        }
    }

    // row absmax: wave reduce, then across the NW waves through LDS.
    local_max = warp_reduce_max<32>(local_max);
    float row_max = local_max;
    if constexpr (NW > 1) {
        __shared__ float s_max[NW];
        if (lane == 0) { s_max[wave] = local_max; }
        __syncthreads();
        row_max = s_max[0];
        #pragma unroll
        for (int w = 1; w < NW; w++) { row_max = fmaxf(row_max, s_max[w]); }
    }
    // Real division + eps clamp, matching fp8_quant_pack_row_kernel / CPU ref.
    const float scale     = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    if (threadIdx.x == 0) {
        a_scale[row] = scale;
    }

    uint8_t * orow = a_fp8 + (size_t) row * (size_t) K;
    #pragma unroll
    for (int j = 0; j < MAXAW; j++) {
        const int a = wave + j * NW;
        if (a < a_dim) {
            uint8_t * ob = orow + (size_t) a * B + lane * E;
            #pragma unroll
            for (int e = 0; e < E; e += 4) {
                if constexpr (E >= 4) {
                    const uint32_t packed = fp8_quant_rot_pack4_hw(v[j][e] * inv_scale, v[j][e + 1] * inv_scale, v[j][e + 2] * inv_scale, v[j][e + 3] * inv_scale);
                    *reinterpret_cast<uint32_t *>(ob + e) = packed;
                } else {
                    #pragma unroll
                    for (int q = 0; q < E; q++) { ob[e + q] = fp8_quant_rot_f32_to_e4m3(v[j][e + q] * inv_scale); }
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// FP8_QUANT_ROT per-row KRONECKER V4 (2026-09-18): the V3 one-wave-per-row
// kronecker instantiation measured 0.373 ms/call at K=5120 (a=5, b=1024,
// 2048 rows) on the router chain -- SLOWER than the old LDS kernel (0.215):
// one wave holding 160 floats/lane is latency-bound with only 2048 waves.
// V4 uses a 4-wave workgroup per row: lane L (0..127) owns the E = b/128
// contiguous elements [E*L, E*L+E) of EVERY a-slice (E = 8 at b = 1024, two
// float4 loads per slice), so the H_a^T mix across slices stays lane-local.
// FWHT-b: strides < E in registers, strides E..16E across lanes of the wave
// via __shfl_xor (5 stages), and the last two stages (32E, 64E = partner
// lanes L^32, L^64, L^96 in the other waves) as ONE LDS exchange + a 4-point
// butterfly in registers. Optional fused RMSNorm (norm_w != nullptr): the
// row's sum of squares is reduced first from the same registers, so the
// kernel reads the RESIDUAL row and the norm weight instead of the normed
// tensor (x_norm = x * rsqrt(mean(x^2) + eps) * w), same math as
// rms_norm_f32 + the MUL it fuses.
// ---------------------------------------------------------------------------
template <int B, int MAXA, bool HAS_HA, bool FUSE_NORM>
__launch_bounds__(128)
static __global__ void ml8_fp8_qrot_v4_kernel(
    const float * __restrict__ x, const float * __restrict__ h_a,
    const float * __restrict__ norm_w, float norm_eps,
    uint8_t * __restrict__ a_fp8, float * __restrict__ a_scale,
    int K, int a_dim, int n_rows) {
    constexpr int NT = 128;              // threads per row
    constexpr int E  = B / NT;           // contiguous elements per lane per slice
    static_assert(E >= 4 && (E & (E - 1)) == 0, "b_dim must be 512 or 1024 here");
    __shared__ float s_x[MAXA * B];      // whole row for the cross-wave exchange
    __shared__ float s_red[4];
    __shared__ float s_ha[MAXA * MAXA];  // H_a staged once (global loads inside the mix loop
                                         // serialized on latency: 0.30 ms/call before this)

    const int row  = blockIdx.x;
    const int L    = threadIdx.x;
    const int lane = L & 31;
    const int wave = L >> 5;
    if (row >= n_rows) { return; }
    const float * xrow = x + (size_t) row * (size_t) K;
    if constexpr (HAS_HA) {
        if (L < a_dim * a_dim) { s_ha[L] = h_a[L] * rsqrtf((float) B); }
    }

    float v[MAXA][E];
    // ---- load (+ fused RMSNorm) ----
    float ss = 0.0f;
    #pragma unroll
    for (int a = 0; a < MAXA; a++) {
        if (a < a_dim) {
            const float * blk = xrow + (size_t) a * B + L * E;
            #pragma unroll
            for (int e = 0; e < E; e += 4) {
                const float4 t = *reinterpret_cast<const float4 *>(blk + e);
                v[a][e] = t.x; v[a][e + 1] = t.y; v[a][e + 2] = t.z; v[a][e + 3] = t.w;
            }
            if constexpr (FUSE_NORM) {
                #pragma unroll
                for (int e = 0; e < E; e++) { ss += v[a][e] * v[a][e]; }
            }
        }
    }
    if constexpr (FUSE_NORM) {
        ss = warp_reduce_sum<32>(ss);
        if (lane == 0) { s_red[wave] = ss; }
        __syncthreads();
        const float tot = s_red[0] + s_red[1] + s_red[2] + s_red[3];
        const float rms = rsqrtf(tot / (float) K + norm_eps);
        __syncthreads();   // s_red reused below
        #pragma unroll
        for (int a = 0; a < MAXA; a++) {
            if (a < a_dim) {
                const float * wb = norm_w + (size_t) a * B + L * E;
                #pragma unroll
                for (int e = 0; e < E; e++) { v[a][e] = v[a][e] * rms * wb[e]; }
            }
        }
    }
    // ---- FWHT-B per slice ----
    #pragma unroll
    for (int a = 0; a < MAXA; a++) {
        if (a < a_dim) {
            #pragma unroll
            for (int stride = 1; stride < E; stride <<= 1) {       // in-lane
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    if ((e & stride) == 0) {
                        const float lo = v[a][e], hi = v[a][e + stride];
                        v[a][e] = lo + hi; v[a][e + stride] = lo - hi;
                    }
                }
            }
            #pragma unroll
            for (int ls = 1; ls < 32; ls <<= 1) {                  // in-wave: strides E..16E
                const bool upper = (lane & ls) != 0;
                #pragma unroll
                for (int e = 0; e < E; e++) {
                    const float mine = v[a][e];
                    const float p    = __shfl_xor_sync(0xffffffff, mine, ls, 32);
                    v[a][e] = upper ? (p - mine) : (mine + p);
                }
            }
            // cross-wave: strides 32E and 64E -> lanes L^32 (wave^1) and L^64 (wave^2).
            // float4 LDS traffic (scalar 32 B-strided accesses were 8-way bank conflicted).
            #pragma unroll
            for (int e = 0; e < E; e += 4) {
                *reinterpret_cast<float4 *>(s_x + a * B + L * E + e) = make_float4(v[a][e], v[a][e + 1], v[a][e + 2], v[a][e + 3]);
            }
        }
    }
    __syncthreads();
    #pragma unroll
    for (int a = 0; a < MAXA; a++) {
        if (a < a_dim) {
            const float * r1 = s_x + a * B + (L ^ 32) * E;
            const float * r2 = s_x + a * B + (L ^ 64) * E;
            const float * r3 = s_x + a * B + (L ^ 96) * E;
            const bool up1 = (wave & 1) != 0;   // stage 32E: upper partner gets p - v
            const bool up2 = (wave & 2) != 0;   // stage 64E
            #pragma unroll
            for (int e = 0; e < E; e += 4) {
                const float4 p1 = *reinterpret_cast<const float4 *>(r1 + e);
                const float4 p2 = *reinterpret_cast<const float4 *>(r2 + e);
                const float4 p3 = *reinterpret_cast<const float4 *>(r3 + e);
                const float a1[4] = { p1.x, p1.y, p1.z, p1.w };
                const float a2[4] = { p2.x, p2.y, p2.z, p2.w };
                const float a3[4] = { p3.x, p3.y, p3.z, p3.w };
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    // stage 32E on (v, p1) and on (p2, p3); then stage 64E on the two results
                    const float t0  = up1 ? (a1[q] - v[a][e + q]) : (v[a][e + q] + a1[q]);
                    const float t0p = up1 ? (a3[q] - a2[q])       : (a2[q] + a3[q]);
                    v[a][e + q] = up2 ? (t0p - t0) : (t0 + t0p);
                }
            }
        }
    }
    const float inv_sqrt_b = rsqrtf((float) B);
    float local_max = 0.0f;
    if constexpr (HAS_HA) {
        float y[MAXA][E];
        #pragma unroll
        for (int k = 0; k < MAXA; k++) {
            if (k < a_dim) {
                #pragma unroll
                for (int e = 0; e < E; e++) { y[k][e] = 0.0f; }
                #pragma unroll
                for (int i = 0; i < MAXA; i++) {
                    if (i < a_dim) {
                        const float h = s_ha[i * a_dim + k];
                        #pragma unroll
                        for (int e = 0; e < E; e++) { y[k][e] = fmaf(h, v[i][e], y[k][e]); }
                    }
                }
                #pragma unroll
                for (int e = 0; e < E; e++) { local_max = fmaxf(local_max, fabsf(y[k][e])); }
            }
        }
        #pragma unroll
        for (int k = 0; k < MAXA; k++) {
            #pragma unroll
            for (int e = 0; e < E; e++) { v[k][e] = y[k][e]; }
        }
    } else {
        #pragma unroll
        for (int a = 0; a < MAXA; a++) {
            if (a < a_dim) {
                #pragma unroll
                for (int e = 0; e < E; e++) { v[a][e] *= inv_sqrt_b; local_max = fmaxf(local_max, fabsf(v[a][e])); }
            }
        }
    }
    local_max = warp_reduce_max<32>(local_max);
    if (lane == 0) { s_red[wave] = local_max; }
    __syncthreads();
    const float row_max = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));
    const float scale     = fmaxf(row_max / ML8_FP8_E4M3_MAX, ML8_ACT_SCALE_EPS);
    const float inv_scale = 1.0f / scale;
    if (L == 0) { a_scale[row] = scale; }
    uint8_t * orow = a_fp8 + (size_t) row * (size_t) K;
    #pragma unroll
    for (int a = 0; a < MAXA; a++) {
        if (a < a_dim) {
            uint8_t * ob = orow + (size_t) a * B + L * E;
            #pragma unroll
            for (int e = 0; e < E; e += 4) {
                const uint32_t packed = fp8_quant_rot_pack4_hw(v[a][e] * inv_scale, v[a][e + 1] * inv_scale, v[a][e + 2] * inv_scale, v[a][e + 3] * inv_scale);
                *reinterpret_cast<uint32_t *>(ob + e) = packed;
            }
        }
    }
}

// V4 launcher: kronecker (b in {512, 1024}, a <= 8) and block_hadamard with the same b.
static bool ml8_launch_qrot_v4(
    cudaStream_t stream, bool kronecker,
    const float * x, const float * h_a, const float * norm_w, float norm_eps,
    uint8_t * a_fp8, float * a_scale, int K, int a_dim, int b_dim, int n_rows) {
    const dim3 grid((unsigned) n_rows);
    const bool fuse = norm_w != nullptr;
#define ML8_QROT_V4(B_, MAXA_, HA_) \
    if (fuse) { ml8_fp8_qrot_v4_kernel<B_, MAXA_, HA_, true ><<<grid, 128, 0, stream>>>(x, h_a, norm_w, norm_eps, a_fp8, a_scale, K, a_dim, n_rows); } \
    else      { ml8_fp8_qrot_v4_kernel<B_, MAXA_, HA_, false><<<grid, 128, 0, stream>>>(x, h_a, nullptr, 0.0f,    a_fp8, a_scale, K, a_dim, n_rows); } \
    return true;
    if (b_dim == 1024) {
        if (kronecker  && a_dim <= 5) { ML8_QROT_V4(1024, 5, true)  }
        if (kronecker  && a_dim <= 8) { ML8_QROT_V4(1024, 8, true)  }
        if (!kronecker && a_dim <= 8) { ML8_QROT_V4(1024, 8, false) }
    }
    if (b_dim == 512) {
        if (kronecker  && a_dim <= 8)  { ML8_QROT_V4(512, 8, true)   }
        if (!kronecker && a_dim <= 16) { ML8_QROT_V4(512, 16, false) }
    }
#undef ML8_QROT_V4
    return false;
}

// Returns true if a V3 instantiation covers (kind, a_dim, b_dim) and launched it.
static bool ml8_launch_qrot_v3(
    cudaStream_t stream, bool kronecker,
    const float * x, const float * h_a, uint8_t * a_fp8, float * a_scale,
    int K, int a_dim, int b_dim, int n_rows) {
    const dim3 grid((unsigned) n_rows);
    if (kronecker) {
        // one wave per row; a_dim*E floats/lane resident
        if (b_dim == 1024 && a_dim <= 5) {
            ml8_fp8_qrot_v3_kernel<1024, 1, 5, true><<<grid, 32, 0, stream>>>(x, h_a, a_fp8, a_scale, K, a_dim, n_rows);
            return true;
        }
        if (b_dim == 1024 && a_dim <= 8) {
            ml8_fp8_qrot_v3_kernel<1024, 1, 8, true><<<grid, 32, 0, stream>>>(x, h_a, a_fp8, a_scale, K, a_dim, n_rows);
            return true;
        }
        if (b_dim == 512 && a_dim <= 16) {
            ml8_fp8_qrot_v3_kernel<512, 1, 16, true><<<grid, 32, 0, stream>>>(x, h_a, a_fp8, a_scale, K, a_dim, n_rows);
            return true;
        }
        if (b_dim == 256 && a_dim <= 16) {
            ml8_fp8_qrot_v3_kernel<256, 1, 16, true><<<grid, 32, 0, stream>>>(x, h_a, a_fp8, a_scale, K, a_dim, n_rows);
            return true;
        }
        return false;
    }
    if (b_dim == 128) {
        // Keep <= ~12 blocks (48 floats) per lane resident: more waves per
        // row, shorter per-wave dependency chains, more bytes in flight.
        static const int nw_env = [] { const char * e = std::getenv("MT_FP8_QROT_V3_NW"); return e ? std::atoi(e) : 0; }();
        if (a_dim <= 16 && nw_env == 0) {
            ml8_fp8_qrot_v3_kernel<128, 4, 4, false><<<grid, 32 * 4, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true;
        }
        const int nw = nw_env ? nw_env : 16;   // measured 2026-09-18 K=17408: NW=4 0.744 ms, 8 0.832, 16 0.463 (359 GB/s)
        #define ML8_QROT_V3_B128(NW_) \
            if (nw == NW_) { \
                const int per_wave = (a_dim + NW_ - 1) / NW_; \
                if (per_wave <= 4)  { ml8_fp8_qrot_v3_kernel<128, NW_,  4, false><<<grid, 32 * NW_, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; } \
                if (per_wave <= 12) { ml8_fp8_qrot_v3_kernel<128, NW_, 12, false><<<grid, 32 * NW_, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; } \
                if (per_wave <= 34) { ml8_fp8_qrot_v3_kernel<128, NW_, 34, false><<<grid, 32 * NW_, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; } \
                return false; \
            }
        ML8_QROT_V3_B128(4)
        ML8_QROT_V3_B128(8)
        ML8_QROT_V3_B128(16)
        #undef ML8_QROT_V3_B128
        return false;
    }
    if (b_dim == 64) {
        constexpr int NW = 4;
        const int per_wave = (a_dim + NW - 1) / NW;
        if (per_wave <= 8)  { ml8_fp8_qrot_v3_kernel<64, NW,  8, false><<<grid, 32 * NW, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; }
        if (per_wave <= 32) { ml8_fp8_qrot_v3_kernel<64, NW, 32, false><<<grid, 32 * NW, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; }
        return false;
    }
    if (b_dim == 32) {
        constexpr int NW = 4;
        const int per_wave = (a_dim + NW - 1) / NW;
        if (per_wave <= 16) { ml8_fp8_qrot_v3_kernel<32, NW, 16, false><<<grid, 32 * NW, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; }
        if (per_wave <= 64) { ml8_fp8_qrot_v3_kernel<32, NW, 64, false><<<grid, 32 * NW, 0, stream>>>(x, nullptr, a_fp8, a_scale, K, a_dim, n_rows); return true; }
        return false;
    }
    return false;
}

// MT_FP8_QROT_V3=0 forces V2 (then V2's own switch applies) for A/B.
static bool ggml_cuda_fp8_qrot_v3_disabled() {
    static const bool off = [] {
        const char * e = std::getenv("MT_FP8_QROT_V3");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    return off;
}

// Largest a_dim the block_hadamard V2 path's "big" register-array
// instantiation supports before falling back to the old paths. Chosen to
// comfortably cover the production TP K-split shard widths seen elsewhere
// in this file (a_dim=98 for 12544, a_dim=136 for 17408) with headroom;
// above this the register footprint of a MAX_A-sized per-thread array is
// judged not worth it vs. just falling through to the generic path.
static constexpr int ML8_QROT_V2_BLOCKHAD_MAX_A = 160;

// MT_FP8_QROT_V2=0 forces the old (round-3) per-row paths for A/B testing
// against this round-4 kernel; unset or any other value keeps V2 on (the
// default).
static bool ggml_cuda_fp8_qrot_v2_disabled() {
    static const bool off = [] {
        const char * e = std::getenv("MT_FP8_QROT_V2");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    return off;
}

// RMS_NORM -> MUL(w) -> FP8_QUANT_ROT fusion (2026-09-18): the V4 quant kernel reads the
// residual row, normalizes it in registers and quantizes -- the normed f32 tensor is never
// written or re-read (42 MB each way per call at 2048x5120) and the rms_norm launch goes
// away. Only when the normed tensor has no other consumer (ggml_can_fuse guarantees that
// at the call site) and the shape has a V4 instantiation. Returns false to let the caller
// fall back to the separate ops. MT_FP8_QROT_NORM_FUSE=0 disables.
bool ggml_cuda_op_fp8_quant_rot_fused_norm(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * rms_norm, const ggml_tensor * mul, ggml_tensor * dst) {
    static const bool off = [] {
        const char * e = std::getenv("MT_FP8_QROT_NORM_FUSE");
        return e != nullptr && std::strcmp(e, "0") == 0;
    }();
    if (off || ggml_cuda_fp8_qrot_v3_disabled()) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    if (dst->src[0] != mul || mul->src[0] != rms_norm) {
        return false;                       // only the (norm, w) operand order
    }
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(w) ||
        w->ne[0] != x->ne[0] || ggml_nelements(w) != w->ne[0]) {
        return false;                       // weight must be a plain [K] broadcast row
    }
    const ggml_tensor * h_a = dst->src[1];
    const int32_t * pp    = (const int32_t *) dst->op_params;
    const int32_t   a_dim = pp[0], b_dim = pp[1], kind = pp[2], G_raw = pp[3];
    if (G_raw != 0 || kind == GGML_FP8_QUANT_ROT_KIND_NONE) {
        return false;
    }
    const int64_t K      = x->ne[0];
    const int64_t n_rows = x->ne[1] * x->ne[2] * x->ne[3];
    if ((int64_t) a_dim * b_dim != K || dst->ne[0] != K + (int64_t) sizeof(float)) {
        return false;
    }
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    uint8_t * out_qs    = (uint8_t *) dst->data;
    float   * out_scale = (float *) ((uint8_t *) dst->data + (size_t) n_rows * (size_t) K);
    const bool kron = kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER;
    if (kron && (h_a == nullptr || h_a->type != GGML_TYPE_F32 || !ggml_is_contiguous(h_a))) {
        return false;
    }
    const bool ok = ml8_launch_qrot_v4(ctx.stream(), kron,
        (const float *) x->data, h_a ? (const float *) h_a->data : nullptr,
        (const float *) w->data, eps, out_qs, out_scale, (int) K, a_dim, b_dim, (int) n_rows);
    if (ok) {
        CUDA_CHECK(cudaGetLastError());
    }
    return ok;
}

void ggml_cuda_op_fp8_quant_rot(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {

    const ggml_tensor * x   = dst->src[0];
    const ggml_tensor * h_a = dst->src[1];

    GGML_ASSERT(x != nullptr);
    GGML_ASSERT(x->type   == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I8);
    GGML_ASSERT(ggml_is_contiguous(x));

    const int32_t * pp     = (const int32_t *) dst->op_params;
    const int32_t   a_dim  = pp[0];
    const int32_t   b_dim  = pp[1];
    const int32_t   kind   = pp[2];
    const int32_t   G_raw  = pp[3];   // 0 = per-row (this mode), else the block width (32/128)
    GGML_ASSERT(G_raw == 0 || G_raw == 32 || G_raw == 128);

    const int64_t K = x->ne[0];
    const bool per_row = (G_raw == 0);
    // Per-row mode only needs K%32==0 (no scale-group alignment requirement);
    // the block-packing modes need K%G==0 for a whole number of groups.
    GGML_ASSERT(per_row ? (K % 32 == 0) : (K % G_raw == 0));
    const int64_t n_groups = per_row ? 0 : (K / G_raw);
    const int64_t row_out  = per_row ? (K + (int64_t) sizeof(float))
                                      : (K + n_groups * (int64_t) sizeof(float));
    GGML_ASSERT(dst->ne[0] == row_out);

    const int64_t n_rows = x->ne[1] * x->ne[2] * x->ne[3];
    GGML_ASSERT(dst->ne[1] == x->ne[1] && dst->ne[2] == x->ne[2] && dst->ne[3] == x->ne[3]);

    cudaStream_t stream = ctx.stream();

    // MAD-3xx round 4 (V2) — single fused kernel family for BOTH per-row
    // kinds, replacing all three round-3 paths below (kronecker fused,
    // block_hadamard fused, and the generic 3-launch chain block_hadamard's
    // large-K shapes fell back to). See ml8_fp8_qrot_v2_kernel's comment for
    // why its LDS is O(b_dim) instead of O(K) -- that's what lets it cover
    // the K=17408 shape the round-3 fused block_hadamard kernel couldn't.
    //
    // Traffic accounting for why this MUST stay a single launch with the
    // rotated row held on-chip (registers), not round-tripped through DRAM:
    // a K=17408, n_rows=2048 row is 42MB in / 10.5MB out (52.5MB "useful").
    // At the 600 GB/s effective floor that's an 87.5us budget. Any design
    // that re-reads X a second time (to avoid holding the rotated row
    // on-chip between the absmax pass and the quantize pass) already moves
    // >=84MB of physical traffic before even writing the output, which caps
    // effective throughput at <350 GB/s regardless of how fast the reads
    // are -- provably short of the target. So the row's rotated values must
    // survive on-chip (in per-thread registers here) from the absmax pass
    // straight into the quantize pass.
    //
    // Gate: KRONECKER needs a_dim<=16 (H_a register array, same bound the
    // round-3 kernel already had); BLOCK_HADAMARD needs a_dim<=160 (the V2
    // kernel's own register-array bound -- see ML8_QROT_V2_BLOCKHAD_MAX_A).
    // b_dim must be the power-of-two 16..1024 the FWHT butterfly requires.
    // Anything outside this (plus kind==NONE, which never rotates) falls
    // through to the round-3 paths below unchanged. MT_FP8_QROT_V2=0 forces
    // that fallback unconditionally, for A/B comparison against this kernel.
    if (per_row && !ggml_cuda_fp8_qrot_v2_disabled() &&
        b_dim >= 16 && b_dim <= 1024 && (b_dim & (b_dim - 1)) == 0 &&
        ((kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER && a_dim > 0 && a_dim <= 16) ||
         (kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD && a_dim > 0 && a_dim <= ML8_QROT_V2_BLOCKHAD_MAX_A))) {
        GGML_ASSERT((int64_t) a_dim * (int64_t) b_dim == K);
        uint8_t * out_qs    = (uint8_t *) dst->data;
        float   * out_scale = (float *) ((uint8_t *) dst->data + (size_t) n_rows * (size_t) K);

        if (kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER) {
            GGML_ASSERT(h_a != nullptr && h_a->type == GGML_TYPE_F32 && ggml_is_contiguous(h_a));
            GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);
        }
        if (!ggml_cuda_fp8_qrot_v3_disabled() &&
            ml8_launch_qrot_v4(stream, kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER,
                               (const float *) x->data, h_a ? (const float *) h_a->data : nullptr,
                               nullptr, 0.0f, out_qs, out_scale, (int) K, a_dim, b_dim, (int) n_rows)) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (!ggml_cuda_fp8_qrot_v3_disabled() &&
            ml8_launch_qrot_v3(stream, kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER,
                               (const float *) x->data, h_a ? (const float *) h_a->data : nullptr,
                               out_qs, out_scale, (int) K, a_dim, b_dim, (int) n_rows)) {
            CUDA_CHECK(cudaGetLastError());
            return;
        }

        if (kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER) {
            ml8_launch_qrot_v2<16, true>(
                stream, (const float *) x->data, (const float *) h_a->data,
                out_qs, out_scale, (int) K, a_dim, b_dim, (int) n_rows);
        } else {
            GGML_ASSERT(h_a == nullptr);
            if (a_dim <= 16) {
                ml8_launch_qrot_v2<16, false>(
                    stream, (const float *) x->data, nullptr,
                    out_qs, out_scale, (int) K, a_dim, b_dim, (int) n_rows);
            } else {
                ml8_launch_qrot_v2<ML8_QROT_V2_BLOCKHAD_MAX_A, false>(
                    stream, (const float *) x->data, nullptr,
                    out_qs, out_scale, (int) K, a_dim, b_dim, (int) n_rows);
            }
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // MAD-305 Phase 5 (round 3) — per-row (G=0) + KRONECKER fusion: skip the
    // separate FWHT + H_a^T launches (mt_turbo_fp8_fwht_kernel measured
    // 1.5s / 14427 calls on the router chain) entirely by reusing the
    // EXISTING ml8_fused_rot_quant_kernel (already the fused prologue for
    // ml8_mul_mat_core's h_a!=nullptr path): it does FWHT-over-a_dim-slices
    // + H_a^T-left-multiply + per-row absmax + e4m3 quantize in ONE launch,
    // and already produces the exact two outputs this contract needs
    // (uint8_t a_fp8[M,K] + separate float a_scale[M]) -- just point them at
    // dst->data and dst->data + n_rows*K instead of two ml8_mul_mat_core
    // scratch buffers. Gated identically to ggml_cuda_ml8_can_fuse_rot_mm's
    // existing LDS check (K fp32 dynamic + the kernel's 1024-fp32 static
    // reduce array must fit 64KB) and a_dim<=16 (its register array size);
    // kind==NONE has no rotation at all (this fused kernel unconditionally
    // rotates, so it cannot serve that case) and falls through to the
    // generic path below, which dispatches the vectorized
    // fp8_quant_pack_row_kernel above for per_row instead of a second
    // separate quantize launch. BLOCK_HADAMARD gets its OWN fused fast path
    // (ml8_fused_blockhad_quant_kernel) right below this one -- only an
    // over-large K (LDS budget) falls all the way through to generic.
    if (per_row && kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER && a_dim > 0 && a_dim <= 16 &&
        (size_t) K * sizeof(float) + 1024 * sizeof(float) <= 64 * 1024) {
        GGML_ASSERT(h_a != nullptr && h_a->type == GGML_TYPE_F32 && ggml_is_contiguous(h_a));
        GGML_ASSERT(b_dim >= 16 && b_dim <= 1024 && (b_dim & (b_dim - 1)) == 0);
        GGML_ASSERT((int64_t) a_dim * (int64_t) b_dim == K);
        GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);

        const dim3   grid((unsigned) n_rows, 1, 1);
        const dim3   block((unsigned) b_dim, 1, 1);
        const size_t lds_bytes = (size_t) K * sizeof(float);
        ml8_fused_rot_quant_kernel<<<grid, block, lds_bytes, stream>>>(
            (const float *) x->data,
            (const float *) h_a->data,
            (uint8_t *) dst->data,                                                  // a_fp8 [n_rows, K]
            (float *) ((uint8_t *) dst->data + (size_t) n_rows * (size_t) K),        // a_scale [n_rows]
            (int) K, a_dim, b_dim, (int) n_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // MAD-3xx — per-row (G=0) + BLOCK_HADAMARD fusion, sibling of the
    // KRONECKER fast path above: reuses ml8_fused_blockhad_quant_kernel
    // (FWHT-per-a-slice + per-row absmax + e4m3 quantize in ONE launch, no
    // H_a leg) instead of the generic path's memcpy + mt_turbo_fp8_fwht +
    // fp8_quant_pack_row_kernel three-launch chain. No a_dim<=16 bound here
    // (see the kernel's own comment) — only the LDS budget gates it, same
    // check as the KRONECKER branch. An over-large K falls through to the
    // generic path below.
    if (per_row && kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD &&
        (size_t) K * sizeof(float) + 1024 * sizeof(float) <= 64 * 1024) {
        GGML_ASSERT(h_a == nullptr);
        GGML_ASSERT(b_dim >= 16 && b_dim <= 1024 && (b_dim & (b_dim - 1)) == 0);
        GGML_ASSERT(a_dim > 0 && (int64_t) a_dim * (int64_t) b_dim == K);

        const dim3   grid((unsigned) n_rows, 1, 1);
        const dim3   block((unsigned) b_dim, 1, 1);
        const size_t lds_bytes = (size_t) K * sizeof(float);
        ml8_fused_blockhad_quant_kernel<<<grid, block, lds_bytes, stream>>>(
            (const float *) x->data,
            (uint8_t *) dst->data,                                                  // a_fp8 [n_rows, K]
            (float *) ((uint8_t *) dst->data + (size_t) n_rows * (size_t) K),        // a_scale [n_rows]
            (int) K, a_dim, b_dim, (int) n_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const float * rotated_src = (const float *) x->data;
    ggml_cuda_pool_alloc<float> z_buf(ctx.pool());

    if (kind != GGML_FP8_QUANT_ROT_KIND_NONE) {
        GGML_ASSERT(b_dim >= 16 && b_dim <= 1024 && (b_dim & (b_dim - 1)) == 0);
        GGML_ASSERT(a_dim > 0 && (int64_t) a_dim * (int64_t) b_dim == K);

        z_buf.alloc((size_t) n_rows * (size_t) K);
        CUDA_CHECK(cudaMemcpyAsync(z_buf.get(), x->data,
            (size_t) n_rows * (size_t) K * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        // Same FWHT primitive ggml_cuda_op_ml8_apply_rotation uses for both
        // the kronecker and block_hadamard H_b legs.
        CUDA_CHECK(mt_turbo_fp8_fwht(stream, z_buf.get(),
            (int) (n_rows * a_dim), b_dim, b_dim));

        if (kind == GGML_FP8_QUANT_ROT_KIND_KRONECKER) {
            GGML_ASSERT(h_a != nullptr && h_a->type == GGML_TYPE_F32 && ggml_is_contiguous(h_a));
            GGML_ASSERT(a_dim <= 16 && "a_dim must fit ml8_h_a_left_multiply_kernel's register array");
            GGML_ASSERT(h_a->ne[0] == a_dim && h_a->ne[1] == a_dim);
            // In-place: ml8_h_a_left_multiply_kernel only reads/writes the
            // column its own thread owns (l = threadIdx.x), so z and y may
            // alias — see ggml_cuda_op_ml8_apply_rotation's out-of-place use
            // for the same kernel; aliasing here just skips one buffer.
            const dim3 grid((unsigned) n_rows, 1, 1);
            const dim3 block((unsigned) b_dim, 1, 1);
            ml8_h_a_left_multiply_kernel<<<grid, block, 0, stream>>>(
                z_buf.get(), (const float *) h_a->data, z_buf.get(), a_dim, b_dim);
            CUDA_CHECK(cudaGetLastError());
        } else {
            GGML_ASSERT(kind == GGML_FP8_QUANT_ROT_KIND_BLOCK_HADAMARD);
            GGML_ASSERT(h_a == nullptr);
        }
        rotated_src = z_buf.get();
    } else {
        GGML_ASSERT(h_a == nullptr);
    }

    static const bool log_shapes = (std::getenv("FP8_B128_LOG") != nullptr);
    if (log_shapes) {
        static std::mutex log_mtx;
        static std::unordered_map<std::string, int> seen;
        char buf[96];
        std::snprintf(buf, sizeof(buf), "qrot/%d/%d/%d/%d", (int) n_rows, (int) K, (int) a_dim, kind);
        std::lock_guard<std::mutex> lk(log_mtx);
        if (seen.emplace(buf, 1).second) {
            fprintf(stderr, "[fp8_b128] FP8_QUANT_ROT rows=%d K=%d a_dim=%d b_dim=%d kind=%d\n",
                    (int) n_rows, (int) K, a_dim, b_dim, kind);
        }
    }

    if (per_row) {
        constexpr int THREADS = 256;
        const dim3 grid((unsigned) n_rows, 1, 1);
        const dim3 block(THREADS, 1, 1);
        fp8_quant_pack_row_kernel<THREADS><<<grid, block, 0, stream>>>(
            rotated_src, (int8_t *) dst->data, (int) K, (int) n_rows);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const dim3 grid((unsigned) n_rows, (unsigned) n_groups, 1);
    const dim3 block((unsigned) G_raw, 1, 1);
    if (G_raw == 32) {
        fp8_quant_pack_kernel<32><<<grid, block, 0, stream>>>(
            rotated_src, (int8_t *) dst->data, (int) K, (int) n_groups);
    } else {
        fp8_quant_pack_kernel<128><<<grid, block, 0, stream>>>(
            rotated_src, (int8_t *) dst->data, (int) K, (int) n_groups);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ═════════════════════════════════════════════════════════════════════════
// FP8_B128 phase 2 — GGML_OP_FP8_MUL_MAT (design 4(b)).
// ═════════════════════════════════════════════════════════════════════════

void ggml_cuda_op_fp8_mul_mat(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {
#ifndef GGML_HIP_AITER
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("fp8_b128 mul_mat inference requires ggml-hip built with -DGGML_HIP_AITER=ON");
#else
    const ggml_tensor * w = dst->src[0];
    const ggml_tensor * a = dst->src[1];

    GGML_ASSERT(w != nullptr && a != nullptr);
    GGML_ASSERT(w->type   == GGML_TYPE_FP8_B128 || w->type == GGML_TYPE_ML8_FP8);
    GGML_ASSERT(a->type   == GGML_TYPE_I8);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(w));
    GGML_ASSERT(ggml_is_contiguous(a));
    GGML_ASSERT(ggml_is_contiguous(dst));

    // MAD-305 Phase 5 — GGML_TYPE_ML8_FP8 weight: dispatch to the hand-written
    // gfx1201 "trfeed" block-scale WMMA kernel (rdna4_fp8_gemm/gemm_blockscale.hip)
    // instead of the AITER Triton FP8_B128 path below.
    if (w->type == GGML_TYPE_ML8_FP8) {
        const int32_t K = (int32_t) w->ne[0];
        const int32_t N = (int32_t) w->ne[1];
        const int32_t M = (int32_t) (a->ne[1] * a->ne[2] * a->ne[3]);

        GGML_ASSERT(K % QK_ML8_FP8 == 0);            // scale group width == 32
        GGML_ASSERT(N % 16 == 0);
        const int32_t n_groups = K / QK_ML8_FP8;
        GGML_ASSERT(a->ne[0] == K + n_groups * (int32_t) sizeof(float));
        GGML_ASSERT(dst->ne[0] == N);
        GGML_ASSERT((int64_t) dst->ne[1] * dst->ne[2] * dst->ne[3] == (int64_t) M);

        cudaStream_t stream = ctx.stream();

        // ggml_cuda_ml8_fp8_get_or_repack checks the in-place registry first
        // (the common case: a converter-produced weight already packed at
        // load time by ggml_cuda_ml8_inplace_set) and falls back to a
        // cache-copy repack otherwise (WP_ML8_INPLACE=0). Either way the
        // returned entry's `layout` says which packed byte layout b_packed
        // is actually in (MT_ML8_FP8_GEMM=rdna4|triton, decided once at pack
        // time — see ml8_fp8_gemm_current_layout()).
        const ml8_weight_repack_t * repack = ggml_cuda_ml8_fp8_get_or_repack(stream, w);
        GGML_ASSERT(repack != nullptr && "ml8_fp8 weight repack failed (bad shape or OOM)");
        GGML_ASSERT(repack->layout == ML8_FP8_GEMM_LAYOUT_RDNA4 &&
            "GGML_OP_FP8_MUL_MAT on a GGML_TYPE_ML8_FP8 weight requires the RDNA4 trfeed "
            "packed layout (MT_ML8_FP8_GEMM=triton weights only run through GGML_OP_MUL_MAT / "
            "ggml_cuda_op_ml8_fp8_mul_mat, the mt_ml8_gemm Triton path)");

        // src1 (a) row m: K raw e4m3 bytes, then n_groups fp32 per-32-group
        // activation scales — exactly rdna4_gemm_ml8fp8_blockscale's `a_packed`
        // contract. stride_am_bytes is a->ne[0] in bytes (I8, 1 byte/elem).
        const int stride_am_bytes = (int) a->nb[1];
        GGML_ASSERT(stride_am_bytes == (int) a->ne[0]);

        const hipError_t rc = rdna4_gemm_ml8fp8_blockscale(
            a->data, stride_am_bytes, repack->b_packed, repack->b_scale,
            (float *) dst->data, M, N, K, stream);
        GGML_ASSERT(rc == hipSuccess && "rdna4_gemm_ml8fp8_blockscale dispatch failed");
        return;
    }

    GGML_ASSERT(w->type   == GGML_TYPE_FP8_B128);

    // MAD-305 Phase 5 (round 3, DEFAULT) — GGML_OP_FP8_QUANT_ROT "per-row"
    // (G=0) activation contract: a->ne[0] == K + 4 (a SINGLE fp32 a_scale per
    // row, not one per K-group -- see ml8.cuh/gemm_capi.h). Distinct from the
    // K + K/32 "block packing" contract the GENERIC/PRESHUFFLE/RDNA4 (round 2)
    // branches below keep serving unchanged. Dispatches to the FROZEN Phase-1
    // "trfeed" kernel (rdna4_gemm_fp8_trfeed): it applies a_scale[m]*b_scale[n]
    // ONLY in the epilogue (no in-loop fold), measured 131 TF at production
    // shapes vs the abandoned in-loop-fold kernels' 7-35 TF (gemm_capi.h).
    if (a->ne[0] == w->ne[0] + (int64_t) sizeof(float)) {
        const int32_t K = (int32_t) w->ne[0];
        const int32_t N = (int32_t) w->ne[1];
        const int32_t M = (int32_t) (a->ne[1] * a->ne[2] * a->ne[3]);

        GGML_ASSERT(K % 32 == 0);
        GGML_ASSERT(N % 128 == 0);
        GGML_ASSERT(dst->ne[0] == N);
        GGML_ASSERT((int64_t) dst->ne[1] * dst->ne[2] * dst->ne[3] == (int64_t) M);

        cudaStream_t stream = ctx.stream();
        const ml8_weight_repack_t * repack = ggml_cuda_fp8_b128_get_or_repack(stream, w);
        GGML_ASSERT(repack != nullptr && "fp8_b128 weight repack failed (bad shape or OOM)");
        GGML_ASSERT(repack->layout == FP8_B128_LAYOUT_RDNA4_TRFEED &&
            "GGML_OP_FP8_MUL_MAT per-row (G=0) activation packing requires the "
            "FP8_B128_LAYOUT_RDNA4_TRFEED weight layout (MT_FP8_B128_LAYOUT default); "
            "the K+K/32 block-packing activation contract routes through the "
            "generic/preshuffle/rdna4-tile branches below instead");

        // Per-row (G=0) contract is a GLOBAL split, not a per-row interleave:
        // ggml's own nb[1] for a [K+4, M] I8 tensor is K+4 (standard
        // contiguous stride) but that is NOT how the bytes are actually laid
        // out -- ne[0]=K+4 only communicates total size. The real layout
        // (ggml_cuda_op_fp8_quant_rot's per-row kernel below, matching the
        // CPU op) is: bytes [0, M_total*K) = all rows' e4m3 A bytes
        // back-to-back (row m at byte m*K, M_total = ne1*ne2*ne3 -- M here,
        // since a is 2-D per this dispatch), then bytes [M_total*K,
        // M_total*K + 4*M_total) = fp32 a_scale[m] at byte 4*m. So address
        // both segments directly off a->data; never read a->nb[1] for this
        // layout.
        const float * a_scale_src =
            (const float *) ((const uint8_t *) a->data + (size_t) M * (size_t) K);

        // Pad M up to a multiple of BM=128: the frozen kernel's A-tile LDS
        // fill is UNGUARDED against M (only the C-store epilogue masks), so
        // M_pad must be an exact multiple of the kernel's M tile, not merely
        // of 16 (see gemm_capi.h). Covers M=1 decode through ragged prefill
        // ubatches identically.
        // Decode/verify tile (gemm_fp8_trfeed<32,1>) for M<=32 -- round_up(M,32)
        // is always exactly 32 in that range, which is how gemm_trfeed_prod.hip
        // tells the two tile instantiations apart; M>32 keeps the prefill
        // tile (gemm_fp8_trfeed<128,2>), M_pad = round_up(M,128).
        constexpr int32_t BM_TRFEED_DECODE  = 32;
        constexpr int32_t BM_TRFEED_PREFILL = 128;
        const int32_t M_pad = (M <= BM_TRFEED_DECODE)
            ? BM_TRFEED_DECODE
            : ((M + BM_TRFEED_PREFILL - 1) / BM_TRFEED_PREFILL) * BM_TRFEED_PREFILL;

        ggml_cuda_pool_alloc<uint8_t> a_pad(ctx.pool());
        ggml_cuda_pool_alloc<float>   a_scale_pad(ctx.pool());
        const void  * a_ptr;
        const float * a_scale_ptr;
        if (M_pad == M) {
            a_ptr       = a->data;
            a_scale_ptr = a_scale_src;
        } else {
            a_pad.alloc((size_t) M_pad * (size_t) K);
            a_scale_pad.alloc((size_t) M_pad);
            CUDA_CHECK(cudaMemsetAsync(a_pad.get(), 0, (size_t) M_pad * (size_t) K, stream));
            CUDA_CHECK(cudaMemsetAsync(a_scale_pad.get(), 0, (size_t) M_pad * sizeof(float), stream));
            CUDA_CHECK(cudaMemcpyAsync(a_pad.get(), a->data, (size_t) M * (size_t) K,
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(a_scale_pad.get(), a_scale_src, (size_t) M * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));
            a_ptr       = a_pad.get();
            a_scale_ptr = a_scale_pad.get();
        }

        // MAD-305 decode split-K (2026-09-17): for WORKGROUP-STARVED decode
        // shapes only. The M_pad==32 frozen tile launches N/128 workgroups
        // walking the whole K serially. bench/gemm_trfeed_prod_bench on the
        // 9070 XT: at N=17408/K=5120 (136 WGs) that path already streams B at
        // ~506 GB/s (~80% of peak) and every split count is slower (atomic
        // fp32 epilogue + memset cost more than the parallelism buys); at
        // N=5120/K=17408 (40 WGs, ffn_down/attn_output/ssm_out K-slices)
        // it does 0.478 ms and rdna4_gemm_fp8_trfeed_splitk with 4 splits
        // does 0.224 ms (2.1x). rdna4_trfeed_splitk_default_splits encodes
        // that rule and returns 1 when the base launch is not starved, in
        // which case the bf16 frozen path below is used unchanged. Split-K
        // writes fp32 directly (no bf16 intermediate / convert kernel).
        // MT_FP8_TRFEED_SPLITK=0 disables; MT_FP8_TRFEED_SPLITS=<n> forces a
        // split count (A/B only).
        static const bool splitk_enabled = [] {
            const char * e = std::getenv("MT_FP8_TRFEED_SPLITK");
            return e == nullptr || std::strcmp(e, "0") != 0;
        }();
        static const int splitk_env_override = [] {
            const char * e = std::getenv("MT_FP8_TRFEED_SPLITS");
            return e ? std::atoi(e) : 0;
        }();
        const int n_splits = !splitk_enabled || M_pad != BM_TRFEED_DECODE ? 1
            : (splitk_env_override > 0 ? splitk_env_override : rdna4_trfeed_splitk_default_splits(N, K));

        if (n_splits > 1) {
            // dst is fp32 [N, M] with only the true M rows, but the kernel
            // writes a full M_pad(==32)-row tile: go through a pooled fp32
            // scratch unless M == M_pad, then copy the first M*N floats
            // (dst's rows are a prefix, row-major).
            ggml_cuda_pool_alloc<float> c_f32_pad(ctx.pool());
            float * c_ptr;
            if (M == M_pad) {
                c_ptr = (float *) dst->data;
            } else {
                c_f32_pad.alloc((size_t) M_pad * (size_t) N);
                c_ptr = c_f32_pad.get();
            }
            const hipError_t rc = rdna4_gemm_fp8_trfeed_splitk(
                a_ptr, repack->b_packed, c_ptr, a_scale_ptr, (const float *) repack->b_scale,
                M_pad, N, K, n_splits, stream);
            GGML_ASSERT(rc == hipSuccess && "rdna4_gemm_fp8_trfeed_splitk dispatch failed");
            if (c_ptr != (float *) dst->data) {
                CUDA_CHECK(cudaMemcpyAsync((float *) dst->data, c_ptr, (size_t) M * (size_t) N * sizeof(float),
                                           cudaMemcpyDeviceToDevice, stream));
            }
            return;
        }

        // fp32 epilogue into dst (see the ML8_4 prefill call site above).
        if (!ml8_trfeed_f32out_disabled()) {
            const hipError_t rc = rdna4_gemm_fp8_trfeed_f32(
                (const uint8_t *) a_ptr, (const uint8_t *) repack->b_packed, (float *) dst->data,
                a_scale_ptr, (const float *) repack->b_scale, M_pad, M, N, K, stream);
            GGML_ASSERT(rc == hipSuccess && "rdna4_gemm_fp8_trfeed_f32 dispatch failed");
            return;
        }
        ggml_cuda_pool_alloc<nv_bfloat16> c_bf16(ctx.pool(), (size_t) M_pad * (size_t) N);
        const hipError_t rc = rdna4_gemm_fp8_trfeed(
            a_ptr, repack->b_packed, c_bf16.get(), a_scale_ptr, (const float *) repack->b_scale,
            M_pad, N, K, stream);
        GGML_ASSERT(rc == hipSuccess && "rdna4_gemm_fp8_trfeed dispatch failed");

        // Convert the first M rows of bf16 [M_pad, N] -> fp32 [M, N] into
        // dst->data; row-major layout means those are the first M*N
        // contiguous bf16 elements (same idiom as ml8_mul_mat_core above).
        const to_fp32_cuda_t bf16_to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        GGML_ASSERT(bf16_to_fp32 != nullptr);
        bf16_to_fp32(c_bf16.get(), (float *) dst->data, (size_t) M * (size_t) N, stream);
        return;
    }

    const int32_t K = (int32_t) w->ne[0];
    const int32_t N = (int32_t) w->ne[1];
    const int32_t M = (int32_t) (a->ne[1] * a->ne[2] * a->ne[3]);

    GGML_ASSERT(K % 128 == 0);
    const int32_t n_groups = K / 128;
    GGML_ASSERT(a->ne[0] == K + n_groups * (int32_t) sizeof(float));
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT((int64_t) dst->ne[1] * dst->ne[2] * dst->ne[3] == (int64_t) M);
    // N%128==0 (not just %16): the [K/128,N/128] scale table (shared by all
    // three packed layouts) is indexed by n/128 -- already the supports_op
    // gate for FP8_B128 (ggml-cuda.cu), asserted again here for defense in
    // depth against a caller that skipped it.
    GGML_ASSERT(N % 128 == 0);

    cudaStream_t stream = ctx.stream();

    // ── 1. Look up the packed weight. ggml_cuda_fp8_b128_get_or_repack checks
    // the in-place registry first (the common case for a converter-produced
    // model weight) and falls back to a cache-copy repack otherwise
    // (WP_ML8_INPLACE=0, or N not a multiple of 128 — some test-backend-ops
    // shapes).
    const ml8_weight_repack_t * repack = ggml_cuda_fp8_b128_get_or_repack(stream, w);
    GGML_ASSERT(repack != nullptr && "fp8_b128 weight repack failed (bad shape or OOM)");

    static const bool log_shapes = (std::getenv("FP8_B128_LOG") != nullptr);
    if (log_shapes) {
        static std::mutex log_mtx;
        static std::unordered_map<std::string, int> seen;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "mm/%d/%d/%d/%d", M, N, K, repack->layout);
        std::lock_guard<std::mutex> lk(log_mtx);
        if (seen.emplace(buf, 1).second) {
            fprintf(stderr, "[fp8_b128] FP8_MUL_MAT M=%d N=%d K=%d layout=%s\n", M, N, K,
                    repack->layout == FP8_B128_LAYOUT_RDNA4      ? "rdna4"
                    : repack->layout == FP8_B128_LAYOUT_PRESHUFFLE ? "preshuffle" : "generic");
        }
    }

    // ── 2. Launch the GEMM matching how the weight was packed.
    //   rdna4:       our own hand-written gfx1201 WMMA "trfeed" kernel
    //                (gemm_blockscale.hip), G=128 scale-group fold. Default.
    //   generic:     offs_cm < M, offs_cn < N (see gemm_ml8.py
    //                _gemm_a8w8_blockscale_kernel's c_mask)
    //   preshuffle:  offs_cm < M, offs_cn < N (design 4(b)) plus %M-wrapped
    //                loads
    // All three mask their C store (no M padding needed).
    if (repack->layout == FP8_B128_LAYOUT_RDNA4) {
        // src1 (a) row m: K raw e4m3 bytes, then n_groups fp32 per-128-group
        // activation scales -- exactly rdna4_gemm_fp8b128_blockscale's
        // `a_packed` contract. stride_am_bytes is a->ne[0] in bytes.
        const int stride_am_bytes = (int) a->nb[1];
        GGML_ASSERT(stride_am_bytes == (int) a->ne[0]);
        const hipError_t rc = rdna4_gemm_fp8b128_blockscale(
            a->data, stride_am_bytes, repack->b_packed, (const float *) repack->b_scale,
            (float *) dst->data, M, N, K, stream);
        GGML_ASSERT(rc == hipSuccess && "rdna4_gemm_fp8b128_blockscale dispatch failed");
    } else if (repack->layout == FP8_B128_LAYOUT_PRESHUFFLE) {
        mt_fp8_b128_gemm_args_t args{};
        args.N = N;
        args.K = K;
        args.M = M;
        args.a_packed   = a->data;
        args.b_preshuffled = repack->b_packed;
        args.b_scale    = repack->b_scale;
        args.c          = dst->data;

        args.stride_am = K + K / 32;            // (K + K/32) elements, matches a->ne[0]
        args.stride_ak = 1;
        args.stride_bn = K * 16;
        args.stride_bk = 1;
        args.stride_cm = N;
        args.stride_cn = 1;
        args.stride_ascale_m = (K + K / 32) / 4;  // a_scale is embedded at byte offset K
        args.stride_ascale_k = 1;
        args.stride_bscale_k = N / 128;
        args.stride_bscale_n = 1;

        const hipError_t rc = mt_fp8_b128_gemm(stream, &args);
        GGML_ASSERT(rc == hipSuccess && "mt_fp8_b128_gemm dispatch failed");
    } else {
        mt_fp8_b128_gemm_generic_args_t args{};
        args.N = N;
        args.K = K;
        args.M = M;
        args.a_packed     = a->data;
        args.b_transposed = repack->b_packed;   // e4m3 [K, N] row-major
        args.b_scale      = repack->b_scale;
        args.c            = dst->data;

        args.stride_am = K + K / 32;            // (K + K/32) elements, matches a->ne[0]
        args.stride_ak = 1;
        args.stride_bk = N;                     // B is [K, N] row-major
        args.stride_bn = 1;
        args.stride_cm = N;
        args.stride_cn = 1;
        args.stride_ascale_m = (K + K / 32) / 4;  // a_scale is embedded at byte offset K
        args.stride_ascale_k = 1;                 // real per-128-K-group scale stride
        args.stride_bscale_k = N / 128;
        args.stride_bscale_n = 1;

        const hipError_t rc = mt_fp8_b128_gemm_generic(stream, &args);
        GGML_ASSERT(rc == hipSuccess && "mt_fp8_b128_gemm_generic dispatch failed");
    }
#endif // GGML_HIP_AITER
}

// ═════════════════════════════════════════════════════════════════════════
// MAD-223 G.7 — MoE path.
// ═════════════════════════════════════════════════════════════════════════

// ─── Per-expert weight repack ──────────────────────────────────────────

void ggml_cuda_ml8_repack_blocks_moe(
    cudaStream_t stream,
    const void * src_blocks,
    void *       dst_b_packed,
    float *      dst_b_scale,
    int32_t      N,
    int32_t      K,
    int32_t      group_size,
    int32_t      n_experts) {

    GGML_ASSERT(group_size == QK_ML8);
    GGML_ASSERT(N > 0 && K > 0 && n_experts > 0);
    GGML_ASSERT(K % group_size == 0);

    const int32_t n_groups_k = K / group_size;
    const size_t src_bytes_per_expert      = (size_t) N * (size_t) n_groups_k * (size_t) ML8_BLOCK_BYTES;
    const size_t b_packed_bytes_per_expert = (size_t) (K / 2) * (size_t) N;
    const size_t b_scale_elems_per_expert  = (size_t) n_groups_k * (size_t) N;

    for (int32_t e = 0; e < n_experts; ++e) {
        const uint8_t * src_e = (const uint8_t *) src_blocks + (size_t) e * src_bytes_per_expert;
        uint8_t       * pkd_e = (uint8_t *)       dst_b_packed + (size_t) e * b_packed_bytes_per_expert;
        float         * scl_e = dst_b_scale + (size_t) e * b_scale_elems_per_expert;
        ggml_cuda_ml8_repack_blocks(stream, src_e, pkd_e, scl_e, N, K, group_size);
    }
}

namespace {

struct moe_cache_entry_t {
    ml8_weight_repack_moe_t info;
};

std::mutex                                              g_ml8_moe_cache_mu;
std::unordered_map<const void *, moe_cache_entry_t>     g_ml8_moe_cache;

} // namespace

const ml8_weight_repack_moe_t * ggml_cuda_ml8_get_or_repack_moe(
    cudaStream_t        stream,
    const ggml_tensor * w) {

    if (w == nullptr || w->data == nullptr || w->type != GGML_TYPE_ML8_4) {
        return nullptr;
    }
    const int32_t K         = (int32_t) w->ne[0];
    const int32_t N         = (int32_t) w->ne[1];
    const int32_t n_experts = (int32_t) w->ne[2];
    if (K <= 0 || N <= 0 || n_experts <= 0 || K % QK_ML8 != 0) {
        return nullptr;
    }
    const int32_t group_size = QK_ML8;
    const int32_t n_groups_k = K / group_size;

    // MAD-244: streaming (no per-weight cache) repack. The unbounded cache
    // version eats ~150 MB per MoE-expert tensor and OOMs at 35B+ scale
    // (40 layers × 3 = ~18 GB total). For the AOS legacy path we use ONE
    // shared b_packed/b_scale buffer pair sized to the largest weight seen
    // so far and re-repack on every call. This makes the path A/B-comparable
    // with ML8_4_SOA without exhausting VRAM. Slower than caching for
    // inference, fine for PPL validation.
    static std::mutex g_buf_mu;
    static void *      g_buf_packed         = nullptr;
    static float *     g_buf_scale          = nullptr;
    static size_t      g_buf_packed_cap     = 0;
    static size_t      g_buf_scale_cap      = 0;
    static ml8_weight_repack_moe_t g_buf_info{};

    const size_t b_packed_bytes = (size_t) n_experts * (size_t) (K / 2) * (size_t) N;
    const size_t b_scale_bytes  = (size_t) n_experts * (size_t) n_groups_k * (size_t) N * sizeof(float);

    std::lock_guard<std::mutex> lock(g_buf_mu);
    if ((b_packed_bytes > g_buf_packed_cap || b_scale_bytes > g_buf_scale_cap) && ggml_cuda_wp_vram_log_enabled()) {
        size_t fb = 0, tot = 0; int dev = -1; (void) cudaGetDevice(&dev); (void) cudaMemGetInfo(&fb, &tot);
        fprintf(stderr, "wp vram-budget: ml8_moe_regrow %.1f MiB on device %d (free before %.1f MiB, old cap %.1f MiB)\n",
                (b_packed_bytes + b_scale_bytes) / 1048576.0, dev, fb / 1048576.0, (g_buf_packed_cap + g_buf_scale_cap) / 1048576.0); fflush(stderr);
    }
    if (b_packed_bytes > g_buf_packed_cap) {
        if (g_buf_packed) cudaFree(g_buf_packed);
        g_buf_packed = nullptr;
        cudaError_t err = cudaMalloc(&g_buf_packed, b_packed_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ml8-moe] cudaMalloc(b_packed=%zu) failed: %s\n",
                    b_packed_bytes, cudaGetErrorString(err));
            g_buf_packed_cap = 0;
            return nullptr;
        }
        g_buf_packed_cap = b_packed_bytes;
    }
    if (b_scale_bytes > g_buf_scale_cap) {
        if (g_buf_scale) cudaFree(g_buf_scale);
        g_buf_scale = nullptr;
        cudaError_t err = cudaMalloc((void **) &g_buf_scale, b_scale_bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "[ml8-moe] cudaMalloc(b_scale=%zu) failed: %s\n",
                    b_scale_bytes, cudaGetErrorString(err));
            g_buf_scale_cap = 0;
            return nullptr;
        }
        g_buf_scale_cap = b_scale_bytes;
    }

    ggml_cuda_ml8_repack_blocks_moe(
        stream, w->data, g_buf_packed, g_buf_scale, N, K, group_size, n_experts);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[ml8-moe] repack kernel launch failed: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    g_buf_info.b_packed   = g_buf_packed;
    g_buf_info.b_scale    = g_buf_scale;
    g_buf_info.N          = N;
    g_buf_info.K          = K;
    g_buf_info.n_groups_k = n_groups_k;
    g_buf_info.group_size = group_size;
    g_buf_info.n_experts  = n_experts;
    (void) g_ml8_moe_cache_mu;
    (void) g_ml8_moe_cache;
    return &g_buf_info;
}

// ─── Output scatter kernel (sorted bf16 → dst fp32 via InvGather) ──────
//
// Y_sorted [n_total, N] bf16, dst [N, n_used, n_tokens] fp32.
// One thread per (n, pair) pair; pair = t*n_used + s.
static __global__ void ml8_moe_scatter_kernel(
    const nv_bfloat16 * __restrict__ y_sorted,
    const int32_t     * __restrict__ inv_gather,   // [n_pairs] sorted_pos
    float             * __restrict__ dst,
    int32_t N,
    int32_t n_pairs) {

    const int32_t n    = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t pair = blockIdx.y;
    if (n >= N || pair >= n_pairs) return;

    const int32_t sorted_pos = inv_gather[pair];
    const nv_bfloat16 v = y_sorted[(size_t) sorted_pos * (size_t) N + (size_t) n];
    dst[(size_t) pair * (size_t) N + (size_t) n] = (float) v;
}

// ─── GGML_OP_ML8_MUL_MAT_ID dispatch ────────────────────────────────────

void ggml_cuda_op_ml8_mul_mat_id(
    ggml_backend_cuda_context & ctx,
    ggml_tensor *               dst) {
#ifndef GGML_HIP_AITER
    // ml8 MoE inference dispatches through the AITER Triton-AOT MoE GEMM, only built
    // with -DGGML_HIP_AITER=ON. Unavailable on builds without that toolchain.
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("ml8 mul_mat_id (MoE) inference requires ggml-hip built with -DGGML_HIP_AITER=ON");
#else
    const ggml_tensor * w    = dst->src[0];
    const ggml_tensor * cent = dst->src[1];
    const ggml_tensor * x    = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];

    GGML_ASSERT(w && cent && x && ids);
    GGML_ASSERT(w->type == GGML_TYPE_ML8_4 || w->type == GGML_TYPE_ML8_4_SOA);
    GGML_ASSERT(cent->type == GGML_TYPE_F8_E4M3);
    GGML_ASSERT(x->type    == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(x));
    GGML_ASSERT(ggml_is_contiguous(ids));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const int32_t K         = (int32_t) w->ne[0];
    const int32_t N         = (int32_t) w->ne[1];
    const int32_t n_experts = (int32_t) w->ne[2];
    const int32_t n_used    = (int32_t) x->ne[1];
    const int32_t n_tokens  = (int32_t) x->ne[2];
    const int32_t n_pairs   = n_used * n_tokens;

    GGML_ASSERT(K % QK_ML8 == 0);
    const int32_t group_size  = QK_ML8;
    const int32_t n_groups_k  = K / group_size;
    const int32_t n_centroids = 16;
    GGML_ASSERT(cent->ne[0] == n_centroids);
    GGML_ASSERT(cent->ne[1] == n_groups_k);
    GGML_ASSERT(cent->ne[2] == n_experts);
    GGML_ASSERT(ids->ne[0] == n_used);
    GGML_ASSERT(ids->ne[1] == n_tokens);
    GGML_ASSERT(dst->ne[0] == N);
    GGML_ASSERT(dst->ne[1] == n_used);
    GGML_ASSERT(dst->ne[2] == n_tokens);
    GGML_ASSERT(N % MT_ML8_MOE_BLOCK_N == 0);

    cudaStream_t stream = ctx.stream();

    // ── 1. Per-expert weight access. Two paths:
    //
    //   * GGML_TYPE_ML8_4_SOA — the GGUF stores the kernel-native SOA layout
    //     directly (per expert: K/2 × N bytes of b_packed followed by
    //     n_groups_k × N × 4 bytes of b_scale). We just compute the two
    //     pointers and the per-expert byte stride, no runtime repack, no
    //     cache, no extra VRAM. This is the only path used by GGUFs written
    //     after MAD-244.
    //
    //   * GGML_TYPE_ML8_4 (legacy AOS blocks) — fall back to the
    //     runtime repack cache. Kept for compatibility with pre-MAD-244
    //     MoE GGUFs; preferred for tests/dense models that already use the
    //     block layout. Note: at full 35B+ MoE scale this path can exhaust
    //     VRAM (cache grows to 18+ GB) — see the SOA design doc.
    const void *  w_packed_ptr     = nullptr;
    const float * w_scale_ptr      = nullptr;
    int32_t       stride_w_e_runtime    = 0;
    int32_t       stride_w_bs_e_runtime = 0;

    if (w->type == GGML_TYPE_ML8_4_SOA) {
        const size_t b_packed_bytes_e = (size_t)(K / 2) * (size_t) N;
        const size_t b_scale_bytes_e  = (size_t) n_groups_k * (size_t) N * sizeof(float);
        const size_t per_expert_bytes = b_packed_bytes_e + b_scale_bytes_e;
        GGML_ASSERT(per_expert_bytes % sizeof(float) == 0
                    && "ML8_4_SOA per-expert payload must be float-aligned");

        const uint8_t * base = (const uint8_t *) w->data;
        w_packed_ptr           = base;                                       // expert 0's b_packed
        w_scale_ptr            = (const float *)(base + b_packed_bytes_e);   // expert 0's b_scale
        stride_w_e_runtime     = (int32_t) per_expert_bytes;                 // bytes between experts in b_packed
        stride_w_bs_e_runtime  = (int32_t)(per_expert_bytes / sizeof(float)); // fp32 elements between experts in b_scale
    } else {
        const ml8_weight_repack_moe_t * repack = ggml_cuda_ml8_get_or_repack_moe(stream, w);
        GGML_ASSERT(repack != nullptr);
        w_packed_ptr          = repack->b_packed;
        w_scale_ptr           = repack->b_scale;
        stride_w_e_runtime    = (K / 2) * N;
        stride_w_bs_e_runtime = n_groups_k * N;
    }

    // ── 2. Build routing tensors host-side from ids.
    // Mirrors the ggml-cuda mmq.cu pattern: download ids, bin by expert,
    // upload routing tensors. Cheap because n_pairs is small (≤ ctx × top_k).
    constexpr int32_t BM = MT_ML8_MOE_BLOCK_M;
    std::vector<int32_t> h_ids(n_pairs);
    CUDA_CHECK(cudaMemcpyAsync(h_ids.data(), ids->data,
        (size_t) n_pairs * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<int32_t> h_hist(n_experts, 0);
    for (int32_t i = 0; i < n_pairs; ++i) {
        const int32_t e = h_ids[i];
        GGML_ASSERT(e >= 0 && e < n_experts);
        h_hist[e] += 1;
    }
    // Pad each expert's chunk to BM, build offsets (chunk starts).
    std::vector<int32_t> h_hist_padded(n_experts, 0);
    std::vector<int32_t> h_offs(n_experts, 0);
    int32_t cumulative = 0;
    for (int32_t e = 0; e < n_experts; ++e) {
        h_offs[e] = cumulative;
        h_hist_padded[e] = ((h_hist[e] + BM - 1) / BM) * BM;
        cumulative += h_hist_padded[e];
    }
    const int32_t n_total = cumulative;
    const int32_t grid_m  = n_total / BM;
    const int32_t grid_n  = N / MT_ML8_MOE_BLOCK_N;

    // Bin-sort (s, t) flat indices by expert. The kernel does
    //   X_row = GatherIndx[sorted_pos] / N_EXPTS_ACT
    // (see kernels/moe_op_gemm_ml8.py line ~375) — designed for the case
    // where X stores ONE row per token and pair_idx = token * N_EXPTS_ACT + s.
    // Our X is quantized per-pair (because the post-swiglu down input is
    // genuinely per-pair, not per-token replicated), so we want the kernel
    // to recover the literal pair index. Multiplying the stored gather value
    // by n_used makes the kernel's division a no-op and gives pair_idx back.
    std::vector<int32_t> h_gather(n_total, 0);   // padding slots get safe value (kernel masks via hist)
    std::vector<int32_t> h_inv   (n_pairs, 0);
    std::vector<int32_t> counter (n_experts, 0);
    for (int32_t i = 0; i < n_pairs; ++i) {
        const int32_t e = h_ids[i];
        const int32_t pos = h_offs[e] + counter[e];
        h_gather[pos] = i * n_used;   // see N_EXPTS_ACT division above
        h_inv[i]      = pos;
        counter[e]    += 1;
    }
    // ExptData entries: one per grid_m block. (block_within_expert << 16) | expt_id
    std::vector<int32_t> h_edata(grid_m, 0);
    int32_t block_cursor = 0;
    for (int32_t e = 0; e < n_experts; ++e) {
        const int32_t n_blocks_e = h_hist_padded[e] / BM;
        for (int32_t b = 0; b < n_blocks_e; ++b) {
            h_edata[block_cursor++] = (b << 16) | e;
        }
    }
    GGML_ASSERT(block_cursor == grid_m);

    // Upload routing buffers via pool.
    ggml_cuda_pool_alloc<int32_t> d_hist  (ctx.pool(), (size_t) n_experts);
    ggml_cuda_pool_alloc<int32_t> d_offs  (ctx.pool(), (size_t) n_experts);
    ggml_cuda_pool_alloc<int32_t> d_edata (ctx.pool(), (size_t) grid_m);
    ggml_cuda_pool_alloc<int32_t> d_gather(ctx.pool(), (size_t) n_total);
    ggml_cuda_pool_alloc<int32_t> d_inv   (ctx.pool(), (size_t) n_pairs);
    CUDA_CHECK(cudaMemcpyAsync(d_hist.get(),   h_hist.data(),
        n_experts * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_offs.get(),   h_offs.data(),
        n_experts * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_edata.get(),  h_edata.data(),
        grid_m * sizeof(int32_t),    cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_gather.get(), h_gather.data(),
        n_total * sizeof(int32_t),   cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_inv.get(),    h_inv.data(),
        n_pairs * sizeof(int32_t),   cudaMemcpyHostToDevice, stream));

    // ── 3. Quantize x [n_pairs, K] → fp8 + per-row scale.
    ggml_cuda_pool_alloc<uint8_t> a_fp8  (ctx.pool(), (size_t) n_pairs * (size_t) K);
    ggml_cuda_pool_alloc<float>   a_scale(ctx.pool(), (size_t) n_pairs);
    ggml_cuda_ml8_quantize_activations(
        stream, (const float *) x->data, a_fp8.get(), a_scale.get(), n_pairs, K, /*M_valid=*/n_pairs);

    // ── 4. Allocate sorted bf16 output [n_total, N] and launch wrapper.
    ggml_cuda_pool_alloc<nv_bfloat16> y_sorted(ctx.pool(), (size_t) n_total * (size_t) N);

    mt_ml8_moe_gemm_args_t args{};
    args.shape.N                      = N;
    args.shape.K                      = K;
    args.shape.group_size             = group_size;
    args.shape.n_centroids            = n_centroids;
    args.shape.n_experts              = n_experts;
    args.shape.n_expts_act            = n_used;
    args.shape.apply_swiglu           = 0;
    args.shape.activation_reduction_n = 1;
    args.shape.add_residual           = 0;
    args.shape.per_row_x_scale        = 1;
    args.shape.even_k                 = 1;
    args.shape.mask_k_limit           = K;
    args.shape.upcast_indices         = 0;
    args.shape.has_bias               = 0;
    args.shape.has_gammas             = 0;
    args.shape.has_x_static_scale     = 0;
    args.shape.has_w_static_scale     = 0;
    args.shape.has_quant_static_scale = 0;

    args.y                  = y_sorted.get();
    args.x_fp8              = a_fp8.get();
    args.w_packed           = const_cast<void *>(w_packed_ptr);
    args.x_scale_fp32       = a_scale.get();
    args.w_scale_fp32       = const_cast<float *>(w_scale_ptr);
    // No lut_group_off here: MoE (mul_mat_id) weights are not K-split under
    // tensor parallelism in this scheme, only dense ML8_MUL_MAT weights are —
    // see ml8_mul_mat_core above.
    args.centroid_lut_fp8   = cent->data;
    args.bias               = nullptr;
    args.gammas             = nullptr;
    args.x_static_scale     = nullptr;
    args.w_static_scale     = nullptr;
    args.quant_static_scale = nullptr;
    args.alpha              = 0.0f;
    args.limit              = 0.0f;
    args.gather_indx        = d_gather.get();
    args.expt_hist          = d_hist.get();
    args.expt_offs          = d_offs.get();
    args.expt_offs_sum      = nullptr;
    args.expt_data          = d_edata.get();
    args.M                  = n_total;
    args.grid_m             = grid_m;
    args.grid_n             = grid_n;

    // Strides (mirrors the test's layout):
    args.stride_y_k        = 0;
    args.stride_y_m        = N;
    args.stride_y_n        = 1;
    args.stride_x_m        = K;
    args.stride_x_k        = 1;
    args.stride_x_bs_m     = 1;
    args.stride_x_bs_k     = 0;
    args.stride_w_e        = stride_w_e_runtime;
    args.stride_w_k        = N;
    args.stride_w_n        = 1;
    args.stride_w_bs_e     = stride_w_bs_e_runtime;
    args.stride_w_bs_k     = N;
    args.stride_w_bs_n     = 1;
    args.stride_b_e        = 0;
    args.stride_lut_expert = n_groups_k * n_centroids;
    args.stride_lut_k      = n_centroids;

    const hipError_t rc = mt_ml8_moe_gemm(stream, &args);
    GGML_ASSERT(rc == hipSuccess && "mt_ml8_moe_gemm dispatch failed");

    // ── 5. Scatter sorted bf16 output → dst fp32 [N, n_used, n_tokens].
    constexpr int BLOCK_NX = 64;
    const dim3 sgrid((N + BLOCK_NX - 1) / BLOCK_NX, (unsigned) n_pairs, 1);
    const dim3 sblock(BLOCK_NX, 1, 1);
    ml8_moe_scatter_kernel<<<sgrid, sblock, 0, stream>>>(
        y_sorted.get(), d_inv.get(), (float *) dst->data, N, n_pairs);
    CUDA_CHECK(cudaGetLastError());
#endif // GGML_HIP_AITER
}
