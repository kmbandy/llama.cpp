// mt_fp8_b128_gemm.cpp — runtime-shape FP8_B128 preshuffle GEMM wrapper.
// Mirrors mt_ml8_gemm.cpp's pattern (build signature -> Registry::get_or_compile
// -> hipModuleLaunchKernel with manually-packed args[]), targeting
// kernels/gemm_ml8.py::_gemm_a8w8_blockscale_preshuffle_kernel.
//
// FP8_B128 phase 2 (design doc section 4(b)).

#include "mt_fp8_b128_gemm.h"
#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

std::string detect_hip_target() {
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) return "hip:unknown:32";
    hipDeviceProp_t prop {};
    if (hipGetDeviceProperties(&prop, dev) != hipSuccess) return "hip:unknown:32";
    std::string arch = prop.gcnArchName;
    auto colon = arch.find(':');
    if (colon != std::string::npos) arch = arch.substr(0, colon);
    const bool cdna = (arch.size() >= 4 && arch[0] == 'g' && arch[1] == 'f' &&
                       arch[2] == 'x' && arch[3] == '9');
    return std::string("hip:") + arch + (cdna ? ":64" : ":32");
}

// Build the Triton signature for `_gemm_a8w8_blockscale_preshuffle_kernel`
// (post LOCAL PATCH: @triton.heuristics removed, cache_modifier arg removed
// — see gemm_ml8.py). Kernel param order (after those patches):
//
//   a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
//   stride_am, stride_ak, stride_bn, stride_bk, stride_ck, stride_cm, stride_cn,
//   stride_ascale_m, stride_ascale_k, stride_bscale_k, stride_bscale_n,
//   GROUP_K, GROUP_N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
//   NUM_KSPLIT, SPLITK_BLOCK_SIZE, EVEN_K, GRID_MN
//
// ":1" divisibility hints get value-specialized to a compile-time constant 1
// by Triton's AOT/JIT front end and DROPPED from the runtime arg list (same
// mechanism mt_ml8_gemm.cpp documents at #185) — stride_ak, stride_bk,
// stride_cn, stride_ascale_k, stride_bscale_n are always 1 (contiguous rows)
// so we hint them and drop them from the launch args below. stride_ck is
// always multiplied by pid_k == 0 (NUM_KSPLIT == 1) so its value never
// matters — left as a plain runtime i32 (always passed as 0) rather than
// risk a bad divisibility hint.
std::string build_signature_fp8_b128(int32_t M, int32_t N, int32_t K,
                                     const mt_fp8_b128_tuned_cfg & cfg) {
    const int32_t grid_mn = ((M + cfg.bm - 1) / cfg.bm) * ((N + cfg.bn - 1) / cfg.bn);
    const int     even_k  = (K % MT_FP8_B128_BLOCK_SIZE_K == 0) ? 1 : 0;
    // a_ptr's 16-byte alignment hint depends on the packed activation row
    // stride (K + K/32) being a multiple of 16 (design 4(b)).
    const bool a_aligned16 = ((K + K / 32) % 16) == 0;
    const char * a_dtype = a_aligned16 ? "*fp8e4nv:16" : "*fp8e4nv";

    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "%s, *fp8e4nv:16, *fp32:16, *fp32, *fp32:16, "
        "i32, i32:16, i32:16, "
        "i32, i32:1, i32:16, i32:1, i32, i32:16, i32:1, "
        "i32, i32:1, i32, i32:1, "
        "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d",
        a_dtype,
        MT_FP8_B128_GROUP_K, MT_FP8_B128_GROUP_N,
        cfg.bm, cfg.bn, MT_FP8_B128_BLOCK_SIZE_K,
        (M > 32) ? mt_fp8_b128_env_int("MT_FP8_GSM", MT_FP8_B128_GROUP_SIZE_M) : MT_FP8_B128_GROUP_SIZE_M,
        MT_FP8_B128_NUM_KSPLIT, K /* SPLITK_BLOCK_SIZE == K */,
        even_k, grid_mn);
    return buf;
}

struct CachedHandle {
    const aiter::KernelHandle * handle   = nullptr;
    hipError_t                  init_err = hipSuccess;
};

struct ShapeKey {
    int32_t N, K;
    int32_t m_tier;   // 0 = M<=32 (BM=16), 1 = M>32 (BM=64)
    int32_t device;
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey & k) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix((uint64_t) k.N); mix((uint64_t) k.K); mix((uint64_t) k.m_tier); mix((uint64_t) k.device);
        return (size_t) h;
    }
};
struct ShapeKeyEq {
    bool operator()(const ShapeKey & a, const ShapeKey & b) const noexcept {
        return a.N == b.N && a.K == b.K && a.m_tier == b.m_tier && a.device == b.device;
    }
};

using HandleMap = std::unordered_map<ShapeKey, CachedHandle, ShapeKeyHash, ShapeKeyEq>;

HandleMap  & get_handle_map() { static HandleMap m; return m; }
std::mutex & get_cache_mutex() { static std::mutex mu; return mu; }

ShapeKey shape_to_key(int32_t N, int32_t K, int32_t M) {
    int dev = 0;
    (void) hipGetDevice(&dev);
    return ShapeKey { N, K, (M <= 32) ? 0 : 1, dev };
}

hipError_t ensure_initialized(int32_t N, int32_t K, int32_t M,
                              const mt_fp8_b128_tuned_cfg & cfg,
                              const aiter::KernelHandle ** out_handle) {
    std::lock_guard<std::mutex> g(get_cache_mutex());
    HandleMap & m = get_handle_map();
    const ShapeKey key = shape_to_key(N, K, M);
    auto it = m.find(key);
    if (it != m.end()) {
        *out_handle = it->second.handle;
        return it->second.init_err;
    }

    const std::string target = detect_hip_target();
    const std::string sig    = build_signature_fp8_b128(M, N, K, cfg);

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    aiter::KernelSpec spec {
        MT_ML8_KERNEL_SOURCE,   // same vendored file as mt_ml8_gemm (gemm_ml8.py)
        "_gemm_a8w8_blockscale_preshuffle_kernel",
        target, sig,
        MT_FP8_B128_NUM_WARPS, MT_FP8_B128_NUM_STAGES,
    };
    spec.waves_per_eu         = (M > 32) ? mt_fp8_b128_env_int("MT_FP8_WPE", MT_FP8_B128_WAVES_PER_EU) : MT_FP8_B128_WAVES_PER_EU;
    spec.matrix_instr_nonkdim = MT_FP8_B128_MATRIX_INSTR_NONKDIM;

    if (const char * s = std::getenv("MT_ML8_NUM_WARPS"))  { int v = std::atoi(s); if (v > 0 && v <= 32) spec.num_warps  = v; }
    if (const char * s = std::getenv("MT_ML8_NUM_STAGES")) { int v = std::atoi(s); if (v > 0 && v <= 8 ) spec.num_stages = v; }

    CachedHandle c{};
    c.handle = reg.get_or_compile(spec);
    if (!c.handle) {
        std::fprintf(stderr, "mt_fp8_b128_gemm: kernel compile failed for N=%d K=%d (M-tier=%d)\n",
                     N, K, key.m_tier);
        c.init_err = hipErrorInvalidValue;
    } else {
        c.init_err = hipSuccess;
    }
    m.emplace(key, c);
    *out_handle = c.handle;
    return c.init_err;
}

}  // namespace

extern "C" hipError_t mt_fp8_b128_gemm(hipStream_t stream, const mt_fp8_b128_gemm_args_t * args) {
    if (!args) return hipErrorInvalidValue;

    const mt_fp8_b128_tuned_cfg cfg = mt_fp8_b128_pick_config(args->M);

    static const bool gemm_log = (std::getenv("FP8_B128_LOG") != nullptr);
    if (gemm_log) {
        static std::mutex log_mtx;
        static std::unordered_map<std::string, int> seen;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%d/%d/%d", args->M, args->K, args->N);
        std::lock_guard<std::mutex> lk(log_mtx);
        if (seen.emplace(buf, 1).second) {
            std::fprintf(stderr, "[fp8-b128-gemm] M=%-5d K=%-6d N=%-6d -> bm=%d bn=%d wpe=%d mnk=%d\n",
                args->M, args->K, args->N, cfg.bm, cfg.bn,
                MT_FP8_B128_WAVES_PER_EU, MT_FP8_B128_MATRIX_INSTR_NONKDIM);
        }
    }

    if (args->K % MT_FP8_B128_BLOCK_SIZE_K != 0) {
        std::fprintf(stderr, "mt_fp8_b128_gemm: K (%d) must be a multiple of %d\n",
                     args->K, MT_FP8_B128_BLOCK_SIZE_K);
        return hipErrorInvalidValue;
    }

    const aiter::KernelHandle * handle = nullptr;
    hipError_t init_rc = ensure_initialized(args->N, args->K, args->M, cfg, &handle);
    if (init_rc != hipSuccess) return init_rc;
    if (!handle) return hipErrorInvalidValue;

    const unsigned int grid_x =
        (unsigned int) ((int64_t) ((args->M + cfg.bm - 1) / cfg.bm)
                       * (int64_t) ((args->N + cfg.bn - 1) / cfg.bn)
                       * MT_FP8_B128_NUM_KSPLIT);

    hipDeviceptr_t a_ptr  = (hipDeviceptr_t) args->a_packed;
    hipDeviceptr_t b_ptr  = (hipDeviceptr_t) args->b_preshuffled;
    hipDeviceptr_t c_ptr  = (hipDeviceptr_t) args->c;
    // a_scale lives inside the packed activation row at byte offset K.
    hipDeviceptr_t as_ptr = (hipDeviceptr_t) ((const uint8_t *) args->a_packed + args->K);
    hipDeviceptr_t bs_ptr = (hipDeviceptr_t) args->b_scale;

    int32_t M = args->M, N = args->N, K = args->K;
    int32_t stride_am = args->stride_am;
    int32_t stride_bn = args->stride_bn;
    int32_t stride_ck = 0;
    int32_t stride_cm = args->stride_cm;
    int32_t stride_ascale_m = args->stride_ascale_m;
    int32_t stride_bscale_k = args->stride_bscale_k;

    if (args->stride_ak != 1 || args->stride_bk != 1 || args->stride_cn != 1 ||
        args->stride_ascale_k != 1 || args->stride_bscale_n != 1) {
        std::fprintf(stderr, "mt_fp8_b128_gemm: non-unit inner stride"
                     " (ak=%d bk=%d cn=%d ascale_k=%d bscale_n=%d)"
                     " — signature specializes these to 1\n",
                     args->stride_ak, args->stride_bk, args->stride_cn,
                     args->stride_ascale_k, args->stride_bscale_n);
        return hipErrorInvalidValue;
    }

    hipDeviceptr_t p_global_scratch  = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_profile_scratch = (hipDeviceptr_t) nullptr;

    // Runtime arg order matches the compiled kernel's non-constexpr param
    // list: declared order minus the ":1"-hinted (dropped) stride_ak,
    // stride_bk, stride_cn, stride_ascale_k, stride_bscale_n.
    void * kernel_args[] = {
        &a_ptr, &b_ptr, &c_ptr, &as_ptr, &bs_ptr,
        &M, &N, &K,
        &stride_am,
        &stride_bn,
        &stride_ck,
        &stride_cm,
        &stride_ascale_m,
        &stride_bscale_k,
        &p_global_scratch, &p_profile_scratch,
    };

    return handle->launch(stream, grid_x, 1, 1, kernel_args);
}

extern "C" void mt_fp8_b128_gemm_reset_cache(void) {
    std::lock_guard<std::mutex> g(get_cache_mutex());
    get_handle_map().clear();
}

// ═════════════════════════════════════════════════════════════════════════
// GENERIC layout GEMM (default) — launches the non-preshuffle
// `_gemm_a8w8_blockscale_kernel` (WEIGHT_FORMAT=0) against the FP8_B128
// generic packed layout. See mt_fp8_b128_gemm.h for the full rationale.
// ═════════════════════════════════════════════════════════════════════════

namespace {

// Config for the generic-layout kernel: BM128/BN64/GSM4/nw4 prefill (M>16),
// BM16/BN64/GSM1/nw4 decode (M<=16) — the ml8_pick_config() "generic
// shapes" winners (mt_ml8_gemm.h), reused here because this is the exact
// same underlying Triton kernel/tile shape space. MT_FP8_BM/BN/GSM are the
// same diagnostic override names the preshuffle path's tuned_cfg uses
// (mt_fp8_b128_gemm.h); num_warps stays fixed at 4 for both tiers per the
// measured recipe but MT_ML8_NUM_WARPS still overrides it (below).
//
// LOCAL (FP8_B128 BLOCK_SIZE_K=32, 2026-09-17): `bk` is BLOCK_SIZE_K for
// this generic-layout kernel launch specifically — GROUP_K/GROUP_N stay
// fixed at 128 (the block-128 scale table shape), decoupled from
// BLOCK_SIZE_K by gemm_ml8.py LOCAL PATCH #7. Default 32: measured 4.36ms
// vs >=5.0ms for every BK=128 tile we tried at K=5120 N=17408 M=2048 (BK=128
// forces GROUP_K==BLOCK_SIZE_K so it spills / under-occupies). MT_FP8_BK
// lets the orchestrator A/B 32 vs 128 without a rebuild.
struct mt_fp8_b128_generic_cfg {
    int32_t bm, bn, gsm, nw, bk;
};

int32_t mt_fp8_b128_generic_block_k() {
    const int32_t bk = mt_fp8_b128_env_int("MT_FP8_BK", 32);
    return (bk == 32 || bk == 128) ? bk : 32;
}

mt_fp8_b128_generic_cfg mt_fp8_b128_generic_pick_config(int32_t M) {
    const bool prefill = (M > 16);
    // ml8_pick_config()'s "generic shapes" winners (mt_ml8_gemm.h): BM=128/
    // BN=64/GSM=4 prefill, BM=16/BN=64/GSM=1 decode. BLOCK_SIZE_N (64) is
    // deliberately smaller than GROUP_N (128, the scale-table tile size) —
    // the kernel's b_scale addressing (offs_bn // GROUP_N) still resolves
    // to one tile per 64-wide N-block since 64 | 128.
    const int32_t bm  = mt_fp8_b128_env_int("MT_FP8_BM",  prefill ? 128 : 16);
    const int32_t bn  = mt_fp8_b128_env_int("MT_FP8_BN",  64);
    const int32_t gsm = mt_fp8_b128_env_int("MT_FP8_GSM", prefill ? 4 : 1);
    const int32_t bk  = mt_fp8_b128_generic_block_k();
    return mt_fp8_b128_generic_cfg{ bm, bn, gsm, 4, bk };
}

// Build the Triton signature for `_gemm_a8w8_blockscale_kernel` with
// WEIGHT_FORMAT=0, targeting the FP8_B128 generic packed layout (as opposed
// to mt_ml8_gemm.cpp's build_signature_ml8, which targets this SAME kernel
// but for the ml8-4/ml8-fp8 dense paths — different GROUP_N, different
// b_scale dtype, and a real (not dropped) stride_ascale_k; see the header
// comment on mt_fp8_b128_gemm_generic_args_t). Kernel param order (see
// kernels/gemm_ml8.py::_gemm_a8w8_blockscale_kernel):
//
//   a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr, M, N, K,
//   stride_am, stride_ak, stride_bk, stride_bn, stride_ck, stride_cm, stride_cn,
//   stride_ascale_m, stride_ascale_k, stride_bscale_k, stride_bscale_n,
//   GROUP_K, GROUP_N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
//   NUM_KSPLIT, SPLITK_BLOCK_SIZE, EVEN_K, GRID_MN, num_stages,
//   WEIGHT_FORMAT, N_CENTROIDS, centroid_lut_ptr, stride_lut_k
//
// ":1"-hinted args (stride_ak, stride_bn, stride_cn, stride_bscale_n) are
// value-specialized to compile-time 1 by Triton and DROPPED from the
// runtime arg list — all four are always-1 in our packed layouts
// (contiguous rows). stride_ascale_k is deliberately left UNHINTED (a real
// runtime i32, always passed as 1) rather than ":1"-dropped: the activation
// scale table's per-128-K-group entries genuinely advance by one fp32
// element per group (a real, positive stride value the kernel's
// `tl.assume(stride_ascale_k > 0)` needs at launch); how often the kernel
// re-reads it per K iteration depends on BLOCK_SIZE_K vs GROUP_K (see
// gemm_ml8.py LOCAL PATCH #7), not on this stride's value.
//
// LOCAL (FP8_B128 BLOCK_SIZE_K=32): BLOCK_SIZE_K is now `cfg.bk` (32 by
// default, 128 via MT_FP8_BK=128) instead of the fixed
// MT_FP8_B128_BLOCK_SIZE_K(=128) — GROUP_K/GROUP_N stay 128 regardless
// (gemm_ml8.py LOCAL PATCH #7 decouples them when GROUP_K % BLOCK_SIZE_K
// == 0). EVEN_K must therefore check K % cfg.bk, not K % GROUP_K.
// Integer-arg hints matter: triton.tools.compile only specializes what the
// signature says, and a plain i32 where the JIT would have inferred :16 / :1
// costs vectorized loads (measured here: 256 VGPR + spills -> 218, no spills).
// Every hint below is derived from the actual value, so it is always truthful;
// M's hint is part of the handle cache key (see GenericShapeKey::m16).
// Never emit ":1" here: Triton treats an i32:1 hint as a constexpr 1 and DROPS
// the argument from the launch list, and the launch code below passes every
// runtime arg unconditionally (M=1 decode with ":1" shifted all later args and
// memory-faulted). Only the divisibility hint is safe to derive dynamically.
static const char * hint_i32(int32_t v) {
    if (v % 16 == 0)  return "i32:16";
    return "i32";
}

std::string build_signature_fp8_b128_generic(int32_t M, int32_t N, int32_t K,
                                             const mt_fp8_b128_generic_cfg & cfg) {
    const int32_t stride_am       = K + K / 32;          // packed activation row (bytes)
    const int32_t stride_ascale_m = stride_am / 4;       // in floats
    const int32_t stride_bscale_k = N / MT_FP8_B128_GROUP_N;
    const char * a_dtype = (stride_am % 16 == 0) ? "*fp8e4nv:16" : "*fp8e4nv";
    // No M padding — the kernel masks the C store (offs_cm < M, offs_cn < N).
    const int32_t grid_mn = ((M + cfg.bm - 1) / cfg.bm) * ((N + cfg.bn - 1) / cfg.bn);
    const int     even_k  = (K % cfg.bk == 0) ? 1 : 0;

    char buf[1536];
    std::snprintf(buf, sizeof(buf),
        // a_ptr, b_ptr, c_ptr, a_scale_ptr, b_scale_ptr
        "%s, *fp8e4nv:16, *fp32:16, *fp32, *fp32:16, "
        // M, N, K
        "%s, i32:16, i32:16, "
        // stride_am, stride_ak(=1), stride_bk(=N), stride_bn(=1), stride_ck(=0), stride_cm(=N), stride_cn(=1)
        "%s, i32:1, i32:16, i32:1, i32:16, i32:16, i32:1, "
        // stride_ascale_m, stride_ascale_k (=1), stride_bscale_k (=N/128), stride_bscale_n (=1)
        "%s, i32:1, %s, i32:1, "
        // GROUP_K, GROUP_N, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M,
        // NUM_KSPLIT, SPLITK_BLOCK_SIZE(=K), EVEN_K, GRID_MN, num_stages
        "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
        // WEIGHT_FORMAT=0, N_CENTROIDS (ignored), centroid_lut_ptr (dummy), stride_lut_k
        "%d, %d, *fp8e4nv:16, i32",
        a_dtype,
        hint_i32(M),
        hint_i32(stride_am),
        hint_i32(stride_ascale_m), hint_i32(stride_bscale_k),
        /*GROUP_K=*/MT_FP8_B128_GROUP_K, /*GROUP_N=*/MT_FP8_B128_GROUP_N,
        /*BLOCK_SIZE_M=*/cfg.bm, /*BLOCK_SIZE_N=*/cfg.bn, /*BLOCK_SIZE_K=*/cfg.bk,
        /*GROUP_SIZE_M=*/cfg.gsm,
        /*NUM_KSPLIT=*/MT_FP8_B128_NUM_KSPLIT, /*SPLITK_BLOCK_SIZE=*/K,
        /*EVEN_K=*/even_k,
        /*GRID_MN=*/grid_mn,
        /*num_stages=*/MT_FP8_B128_NUM_STAGES,
        /*WEIGHT_FORMAT=*/0, /*N_CENTROIDS=*/1);
    return buf;
}

struct GenericCachedHandle {
    const aiter::KernelHandle * handle   = nullptr;
    hipError_t                  init_err = hipSuccess;
};

struct GenericShapeKey {
    int32_t N, K;
    int32_t m_tier;   // 0 = decode (M<=16, BM=16), 1 = prefill (M>16, BM=128)
    int32_t device;
    int32_t bk;       // BLOCK_SIZE_K (MT_FP8_BK: 32 default, or 128) — part of
                       // the compiled signature, so it must be part of the key.
    int32_t m_hint;   // hint class of M in the signature: 0 plain, 16 (%16)
    int32_t m_tiles;  // cdiv(M, BM): GRID_MN is a constexpr in the signature and
                      // remap_xcd() is only a bijection when it matches the launch grid
};
struct GenericShapeKeyHash {
    size_t operator()(const GenericShapeKey & k) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix((uint64_t) k.N); mix((uint64_t) k.K); mix((uint64_t) k.m_tier); mix((uint64_t) k.device);
        mix((uint64_t) k.bk); mix((uint64_t) k.m_hint); mix((uint64_t) k.m_tiles);
        return (size_t) h;
    }
};
struct GenericShapeKeyEq {
    bool operator()(const GenericShapeKey & a, const GenericShapeKey & b) const noexcept {
        return a.N == b.N && a.K == b.K && a.m_tier == b.m_tier && a.device == b.device &&
               a.bk == b.bk && a.m_hint == b.m_hint && a.m_tiles == b.m_tiles;
    }
};

using GenericHandleMap = std::unordered_map<GenericShapeKey, GenericCachedHandle, GenericShapeKeyHash, GenericShapeKeyEq>;

GenericHandleMap & get_generic_handle_map() { static GenericHandleMap m; return m; }
std::mutex       & get_generic_cache_mutex() { static std::mutex mu; return mu; }

GenericShapeKey generic_shape_to_key(int32_t N, int32_t K, int32_t M, int32_t bk, int32_t bm) {
    int dev = 0;
    (void) hipGetDevice(&dev);
    const int32_t m_hint = (M % 16 == 0) ? 16 : 0;
    return GenericShapeKey { N, K, (M <= 16) ? 0 : 1, dev, bk, m_hint, (M + bm - 1) / bm };
}

hipError_t ensure_initialized_generic(int32_t N, int32_t K, int32_t M,
                                      const mt_fp8_b128_generic_cfg & cfg,
                                      const aiter::KernelHandle ** out_handle) {
    std::lock_guard<std::mutex> g(get_generic_cache_mutex());
    GenericHandleMap & m = get_generic_handle_map();
    const GenericShapeKey key = generic_shape_to_key(N, K, M, cfg.bk, cfg.bm);
    auto it = m.find(key);
    if (it != m.end()) {
        *out_handle = it->second.handle;
        return it->second.init_err;
    }

    const std::string target = detect_hip_target();
    const std::string sig = build_signature_fp8_b128_generic(M, N, K, cfg);

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    aiter::KernelSpec spec {
        MT_ML8_KERNEL_SOURCE,   // same vendored file as mt_ml8_gemm / mt_fp8_b128_gemm (gemm_ml8.py)
        "_gemm_a8w8_blockscale_kernel",
        target, sig,
        cfg.nw, MT_FP8_B128_NUM_STAGES,
    };

    if (const char * s = std::getenv("MT_ML8_NUM_WARPS"))  { int v = std::atoi(s); if (v > 0 && v <= 32) spec.num_warps  = v; }
    if (const char * s = std::getenv("MT_ML8_NUM_STAGES")) { int v = std::atoi(s); if (v > 0 && v <= 8 ) spec.num_stages = v; }

    GenericCachedHandle c{};
    c.handle = reg.get_or_compile(spec);
    if (!c.handle) {
        std::fprintf(stderr, "mt_fp8_b128_gemm_generic: kernel compile failed for N=%d K=%d (M-tier=%d)\n",
                     N, K, key.m_tier);
        c.init_err = hipErrorInvalidValue;
    } else {
        c.init_err = hipSuccess;
    }
    m.emplace(key, c);
    *out_handle = c.handle;
    return c.init_err;
}

}  // namespace

extern "C" hipError_t mt_fp8_b128_gemm_generic(hipStream_t stream, const mt_fp8_b128_gemm_generic_args_t * args) {
    if (!args) return hipErrorInvalidValue;

    const mt_fp8_b128_generic_cfg cfg = mt_fp8_b128_generic_pick_config(args->M);

    static const bool gemm_log = (std::getenv("FP8_B128_LOG") != nullptr);
    if (gemm_log) {
        static std::mutex log_mtx;
        static std::unordered_map<std::string, int> seen;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "generic/%d/%d/%d", args->M, args->K, args->N);
        std::lock_guard<std::mutex> lk(log_mtx);
        if (seen.emplace(buf, 1).second) {
            std::fprintf(stderr, "[fp8-b128-gemm] layout=generic M=%-5d K=%-6d N=%-6d -> bm=%d bn=%d bk=%d gsm=%d nw=%d\n",
                args->M, args->K, args->N, cfg.bm, cfg.bn, cfg.bk, cfg.gsm, cfg.nw);
        }
    }

    if (args->K % cfg.bk != 0) {
        std::fprintf(stderr, "mt_fp8_b128_gemm_generic: K (%d) must be a multiple of BLOCK_SIZE_K (%d)\n",
                     args->K, cfg.bk);
        return hipErrorInvalidValue;
    }

    const aiter::KernelHandle * handle = nullptr;
    hipError_t init_rc = ensure_initialized_generic(args->N, args->K, args->M, cfg, &handle);
    if (init_rc != hipSuccess) return init_rc;
    if (!handle) return hipErrorInvalidValue;

    // No M padding — the kernel masks the C store (offs_cm < M, offs_cn < N).
    const unsigned int grid_x =
        (unsigned int) ((int64_t) ((args->M + cfg.bm - 1) / cfg.bm)
                       * (int64_t) ((args->N + cfg.bn - 1) / cfg.bn)
                       * MT_FP8_B128_NUM_KSPLIT);

    hipDeviceptr_t a_ptr = (hipDeviceptr_t) args->a_packed;
    hipDeviceptr_t b_ptr = (hipDeviceptr_t) args->b_transposed;
    hipDeviceptr_t c_ptr = (hipDeviceptr_t) args->c;
    // a_scale lives inside the packed activation row at byte offset K (same
    // convention as the preshuffle path).
    hipDeviceptr_t as_ptr = (hipDeviceptr_t) ((const uint8_t *) args->a_packed + args->K);
    hipDeviceptr_t bs_ptr = (hipDeviceptr_t) args->b_scale;
    // Dummy non-null centroid LUT pointer — WEIGHT_FORMAT=0 DCEs the branch
    // that reads it, but the positional slot must still bind to something.
    hipDeviceptr_t lut_ptr = bs_ptr;

    if (args->stride_ak != 1 || args->stride_bn != 1 || args->stride_cn != 1 ||
        args->stride_bscale_n != 1) {
        std::fprintf(stderr, "mt_fp8_b128_gemm_generic: non-unit inner stride"
                     " (ak=%d bn=%d cn=%d bscale_n=%d) — signature specializes these to 1\n",
                     args->stride_ak, args->stride_bn, args->stride_cn, args->stride_bscale_n);
        return hipErrorInvalidValue;
    }
    if (args->stride_ascale_k != 1) {
        std::fprintf(stderr, "mt_fp8_b128_gemm_generic: stride_ascale_k must be 1 (got %d) — "
                     "the generic kernel's per-128-K-group activation scale advances by exactly "
                     "one fp32 element per K iteration\n", args->stride_ascale_k);
        return hipErrorInvalidValue;
    }

    int32_t M = args->M, N = args->N, K = args->K;
    int32_t stride_am       = args->stride_am;
    int32_t stride_bk       = args->stride_bk;
    int32_t stride_ck       = 0;
    int32_t stride_cm       = args->stride_cm;
    int32_t stride_ascale_m = args->stride_ascale_m;
    int32_t stride_ascale_k = args->stride_ascale_k;   // real runtime arg, always 1
    int32_t stride_bscale_k = args->stride_bscale_k;
    int32_t stride_lut_k    = 0;

    hipDeviceptr_t p_global_scratch  = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_profile_scratch = (hipDeviceptr_t) nullptr;

    // Runtime arg order matches the compiled kernel's non-constexpr param
    // list: declared order minus the ":1"-hinted (dropped) stride_ak,
    // stride_bn, stride_cn, stride_bscale_n.
    void * kernel_args[] = {
        &a_ptr, &b_ptr, &c_ptr, &as_ptr, &bs_ptr,
        &M, &N, &K,
        &stride_am,
        &stride_bk,
        &stride_ck,
        &stride_cm,
        &stride_ascale_m,
        &stride_ascale_k,
        &stride_bscale_k,
        &lut_ptr, &stride_lut_k,
        &p_global_scratch, &p_profile_scratch,
    };

    return handle->launch(stream, grid_x, 1, 1, kernel_args);
}

extern "C" void mt_fp8_b128_gemm_generic_reset_cache(void) {
    std::lock_guard<std::mutex> g(get_generic_cache_mutex());
    get_generic_handle_map().clear();
}
