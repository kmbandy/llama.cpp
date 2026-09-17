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
