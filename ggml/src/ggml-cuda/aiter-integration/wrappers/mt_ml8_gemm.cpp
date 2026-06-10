// mt_ml8_gemm.cpp — runtime-shape ml8-4 dense GEMM wrapper.
//
// Mirrors mt_aiter_unified_attn.cpp's pattern:
//   - build Triton signature from runtime shape at first call
//   - call aiter::Registry::get_or_compile() → KernelHandle
//   - cache the handle per-shape (MAD-223 G.4.a: was single-shape, now keyed
//     by (N, K, group_size, n_centroids) so gate/up/down with different
//     shapes can coexist in one process)
//   - launch via hipModuleLaunchKernel with manually-packed args[]
//
// MAD-223 Phase C.2 (orig), Phase G.4.a (multi-shape cache).

#include "mt_ml8_gemm.h"
#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

// Build the Triton signature for `_gemm_a8w8_blockscale_kernel` with
// WEIGHT_FORMAT=1 (ml8 LUT path).
//
// Arg order (35 positional args in the kernel sig, with strides as runtime
// `i32` and meta as compile-time constexpr literals):
//
//   1.  a_ptr               *fp8e4nv:16
//   2.  b_ptr               *i8:16          (uint8 packed nibbles)
//   3.  c_ptr               *bf16:16
//   4.  a_scale_ptr         *fp32:16
//   5.  b_scale_ptr         *fp32:16
//   6-8.  M, N, K           i32 / constexpr-N,K / runtime-M
//   9-19. 11 strides         i32 runtime
//   20.   GROUP_K            constexpr int = group_size
//   21.   GROUP_N            constexpr int = 1
//   22-24.BLOCK_SIZE_M/N/K   constexpr int
//   25.   GROUP_SIZE_M       constexpr int = 1
//   26.   NUM_KSPLIT         constexpr int = 1
//   27.   SPLITK_BLOCK_SIZE  constexpr int = K
//   28.   EVEN_K             constexpr bool (heuristic; we pass 1 = True)
//   29.   GRID_MN            constexpr int (heuristic; we compute it)
//   30.   cache_modifier     constexpr str ""
//   31.   num_stages         constexpr int = 1
//   32.   WEIGHT_FORMAT      constexpr int = 1 (ml8 LUT)
//   33.   N_CENTROIDS        constexpr int (= shape.n_centroids)
//   34.   centroid_lut_ptr   *fp8e4nv:16
//   35.   stride_lut_k       i32 runtime
//
// Note: M is RUNTIME (passed as i32 scalar) — different shape M values reuse
// the same compiled binary (good for variable batch size). N, K are constexpr
// (baked into the signature) — different (N, K) tuples need separate compiles.
std::string build_signature_ml8(const mt_ml8_gemm_shape_t & s, int32_t runtime_M,
                                const mt_ml8_tuned_cfg & cfg) {
    const int32_t group_size = s.group_size;
    const int32_t block_size_k = group_size;  // kernel constraint GROUP_K == BLOCK_K
    const int32_t grid_mn = (runtime_M / cfg.bm) * (s.N / cfg.bn);
    // EVEN_K: K must be a multiple of BLOCK_SIZE_K.
    const int even_k = (s.K % block_size_k == 0) ? 1 : 0;

    // WEIGHT_FORMAT switch (see mt_ml8_gemm_shape_t::weight_format):
    //   WF=1 (ml8-4 LUT): arg #2 (b_ptr) is *i8:16 (packed uint8 nibbles).
    //   WF=0 (ml8-fp8):   arg #2 (b_ptr) is *fp8e4nv:16 (raw e4m3, same dtype
    //                     as A, fed straight to tl.dot). The trailing
    //                     centroid_lut_ptr/stride_lut_k args remain in the
    //                     signature (the kernel param list still lists them;
    //                     the body branch that reads them is DCE'd), so the
    //                     launcher still binds those positional slots — ml8.cu
    //                     passes a non-null dummy lut pointer + stride_lut_k=0.
    const char * b_dtype       = (s.weight_format == 0) ? "*fp8e4nv:16" : "*i8:16";
    const int32_t weight_format = s.weight_format;

    char buf[2048];
    // Integer arg hints (i32:16 = divisible by 16, i32:1 = the value 1).
    // The torch-Triton JIT specializes these automatically and emits
    // vectorized global loads; triton.tools.compile only gets what the
    // signature says — plain i32 cost a 5-7x slowdown at M=512 (#185).
    // Guarantees come from ml8.cu: M padded to %16, K % 64 == 0, N % 16
    // == 0, all tensors contiguous → unit inner strides, outer strides
    // = K or N. Arg order: M, N, K, am, ak, bk, bn, 0, cm, cn, 1, 0,
    // bscale_k, bscale_n.
    std::snprintf(buf, sizeof(buf),
        "*fp8e4nv:16, %s, *bf16:16, *fp32:16, *fp32:16, "
        "i32:16, i32:16, i32:16, "
        "i32:16, i32:1, i32:16, i32:1, i32:16, i32:16, i32:1, i32:1, i32:16, i32:16, i32:1, "
        // GROUP_K, GROUP_N, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_SIZE_M,
        // NUM_KSPLIT, SPLITK_BLOCK_SIZE, EVEN_K, GRID_MN, num_stages
        "%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, "
        // ml8 additions: WEIGHT_FORMAT, N_CENTROIDS, centroid_lut_ptr, stride_lut_k
        "%d, %d, *fp8e4nv:16, i32",
        b_dtype,                     // arg #2 b_ptr dtype (WF-dependent)
        group_size,                  // GROUP_K
        MT_ML8_GROUP_N,              // GROUP_N (= 1)
        cfg.bm,                      // BLOCK_SIZE_M  (G.6.a tuned)
        cfg.bn,                      // BLOCK_SIZE_N  (G.6.a tuned)
        block_size_k,                // BLOCK_SIZE_K (= group_size)
        cfg.gsm,                     // GROUP_SIZE_M  (G.6.a tuned)
        MT_ML8_NUM_KSPLIT,           // NUM_KSPLIT
        s.K,                         // SPLITK_BLOCK_SIZE = K (since NUM_KSPLIT=1)
        even_k,                      // EVEN_K
        grid_mn,                     // GRID_MN
        MT_ML8_NUM_STAGES,           // num_stages
        weight_format,               // WEIGHT_FORMAT (0 = fp8 baseline, 1 = LUT)
        s.n_centroids);              // N_CENTROIDS
    return buf;
}

struct CachedHandle {
    const aiter::KernelHandle * handle   = nullptr;
    hipError_t                  init_err = hipSuccess;
};

// Hash + equality for using mt_ml8_gemm_shape_t as a map key. EVEN_K and GRID_MN
// are derived from shape fields in build_signature_ml8(); they don't enter the
// key because they're functions of shape (so two callers with same shape get the
// same handle whether or not their runtime_M happens to fall in the same EVEN_K
// regime — M is a runtime arg, not a constexpr in the signature).
struct ShapeKey {
    int32_t N, K, group_size, n_centroids;
    int32_t weight_format;  // 0=fp8 baseline, 1=ml8-4 LUT — different b_ptr dtype
    int32_t prefill;        // 0=decode (M<=16), 1=prefill (M>16) — tuned configs differ
};
struct ShapeKeyHash {
    size_t operator()(const ShapeKey & k) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        auto mix = [&](uint64_t v) { h ^= v; h *= 1099511628211ULL; };
        mix((uint64_t) k.N);
        mix((uint64_t) k.K);
        mix((uint64_t) k.group_size);
        mix((uint64_t) k.n_centroids);
        mix((uint64_t) k.weight_format);
        mix((uint64_t) k.prefill);
        return (size_t) h;
    }
};
struct ShapeKeyEq {
    bool operator()(const ShapeKey & a, const ShapeKey & b) const noexcept {
        return a.N == b.N && a.K == b.K && a.group_size == b.group_size
            && a.n_centroids == b.n_centroids && a.weight_format == b.weight_format
            && a.prefill == b.prefill;
    }
};

using HandleMap = std::unordered_map<ShapeKey, CachedHandle, ShapeKeyHash, ShapeKeyEq>;

HandleMap  & get_handle_map() {
    static HandleMap m;
    return m;
}
std::mutex & get_cache_mutex() {
    static std::mutex mu;
    return mu;
}

static ShapeKey shape_to_key(const mt_ml8_gemm_shape_t & s, int32_t M) {
    return ShapeKey { s.N, s.K, s.group_size, s.n_centroids,
                      s.weight_format, (M > 16) ? 1 : 0 };
}

// Returns the cached handle for (shape, M-tier), JIT-compiling on first sight.
// Decode and prefill tiers cache separately because their tuned BLOCK sizes
// differ — different constexprs → different kernel binary.
hipError_t ensure_initialized(const mt_ml8_gemm_shape_t & shape, int32_t runtime_M,
                              const mt_ml8_tuned_cfg & cfg,
                              const aiter::KernelHandle ** out_handle) {
    std::lock_guard<std::mutex> g(get_cache_mutex());
    HandleMap & m   = get_handle_map();
    const ShapeKey key = shape_to_key(shape, runtime_M);
    auto it = m.find(key);
    if (it != m.end()) {
        *out_handle = it->second.handle;
        return it->second.init_err;
    }

    const std::string target = detect_hip_target();
    const std::string sig    = build_signature_ml8(shape, runtime_M, cfg);

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    aiter::KernelSpec spec {
        MT_ML8_KERNEL_SOURCE,
        "_gemm_a8w8_blockscale_kernel",
        target, sig, cfg.nw /*num_warps*/, MT_ML8_NUM_STAGES,
    };

    // Env overrides win over tuned defaults (for ad-hoc experimentation).
    int env_nw = cfg.nw, env_ns = MT_ML8_NUM_STAGES;
    if (const char * s = std::getenv("MT_ML8_NUM_WARPS"))  { int v = std::atoi(s); if (v > 0 && v <= 32) env_nw = v; }
    if (const char * s = std::getenv("MT_ML8_NUM_STAGES")) { int v = std::atoi(s); if (v > 0 && v <= 8 ) env_ns = v; }
    spec.num_warps  = env_nw;
    spec.num_stages = env_ns;

    CachedHandle c{};
    c.handle = reg.get_or_compile(spec);
    if (!c.handle) {
        std::fprintf(stderr, "mt_ml8_gemm: kernel compile failed for shape "
                     "N=%d K=%d gs=%d nc=%d\n",
                     shape.N, shape.K, shape.group_size, shape.n_centroids);
        c.init_err = hipErrorInvalidValue;
    } else {
        c.init_err = hipSuccess;
    }
    m.emplace(key, c);
    *out_handle = c.handle;
    return c.init_err;
}

}  // namespace


extern "C" hipError_t mt_ml8_gemm(hipStream_t stream, const mt_ml8_gemm_args_t * args) {
    if (!args) return hipErrorInvalidValue;

    // G.6.b: pick tuned config per (M, K, N). Block sizes vary by tier and shape.
    const mt_ml8_tuned_cfg cfg = ml8_pick_config(args->M, args->shape.K, args->shape.N);

    // ML8_GEMM_LOG=1: print each unique (M, K, N) once with its chosen config —
    // the ground truth for which shapes ride tuned configs vs Phase-A defaults.
    static const bool gemm_log = (std::getenv("ML8_GEMM_LOG") != nullptr);
    if (gemm_log) {
        static std::mutex log_mtx;
        static std::unordered_map<std::string, int> seen;
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%d/%d/%d/%d",
                      args->M, args->shape.K, args->shape.N, args->shape.weight_format);
        std::lock_guard<std::mutex> lk(log_mtx);
        if (seen.emplace(buf, 1).second) {
            std::fprintf(stderr,
                "[ml8-gemm] M=%-5d K=%-6d N=%-6d wf=%d -> bm=%d bn=%d gsm=%d nw=%d\n",
                args->M, args->shape.K, args->shape.N, args->shape.weight_format,
                cfg.bm, cfg.bn, cfg.gsm, cfg.nw);
        }
    }

    // Validate constraints against the chosen config's block sizes.
    if (args->M % cfg.bm != 0) {
        std::fprintf(stderr,
            "mt_ml8_gemm: M (%d) must be multiple of BLOCK_SIZE_M (%d) for tuned cfg "
            "[K=%d N=%d]\n",
            args->M, cfg.bm, args->shape.K, args->shape.N);
        return hipErrorInvalidValue;
    }
    if (args->shape.N % cfg.bn != 0) {
        std::fprintf(stderr,
            "mt_ml8_gemm: shape.N (%d) must be multiple of BLOCK_SIZE_N (%d) for tuned cfg\n",
            args->shape.N, cfg.bn);
        return hipErrorInvalidValue;
    }
    if (args->shape.K % args->shape.group_size != 0) {
        std::fprintf(stderr,
            "mt_ml8_gemm: shape.K (%d) must be multiple of group_size (%d)\n",
            args->shape.K, args->shape.group_size);
        return hipErrorInvalidValue;
    }

    const aiter::KernelHandle * handle = nullptr;
    hipError_t init_rc = ensure_initialized(args->shape, args->M, cfg, &handle);
    if (init_rc != hipSuccess) return init_rc;
    if (!handle) return hipErrorInvalidValue;

    // Compute grid (1D dispatch). Uses the tuned config's block sizes.
    const unsigned int grid_x =
        (unsigned int)((args->M / cfg.bm)
                       * (args->shape.N / cfg.bn)
                       * MT_ML8_NUM_KSPLIT);

    // Stack-local copies for stable pointer-to-arg slots.
    // Arg layout matches the kernel's RUNTIME args (excluding constexprs which
    // are baked at AOT/JIT compile from the signature).
    //
    // Runtime args (in kernel signature order):
    //   1. a_ptr            (hipDeviceptr_t)
    //   2. b_ptr            (hipDeviceptr_t)
    //   3. c_ptr            (hipDeviceptr_t)
    //   4. a_scale_ptr      (hipDeviceptr_t)
    //   5. b_scale_ptr      (hipDeviceptr_t)
    //   6. M                (int32_t)
    //   7. N                (int32_t)  ← passed as runtime even though constexpr
    //                                    in Python (we baked it in the sig; HIP
    //                                    runtime args still expect it positionally)
    //   8. K                (int32_t)
    //   9. stride_am        (int32_t)
    //   10. stride_ak
    //   11. stride_bk
    //   12. stride_bn
    //   13. stride_ck       (= 0 for NUM_KSPLIT=1)
    //   14. stride_cm
    //   15. stride_cn
    //   16. stride_ascale_m
    //   17. stride_ascale_k (= 0 for single K-group access)
    //   18. stride_bscale_k
    //   19. stride_bscale_n
    //   20. centroid_lut_ptr (hipDeviceptr_t)
    //   21. stride_lut_k     (int32_t)
    //
    // NOTE: Triton's AOT for non-constexpr args produces a launcher with
    // exactly these positions. The runtime compiler's compile_aiter_kernel.py
    // does the same (it uses the SAME triton.tools.compile call). For
    // constexpr-baked args (N, K, all the block sizes, etc.), they do NOT
    // appear in the runtime arg list — they're embedded in the binary.
    //
    // HOWEVER — there's a wrinkle: M, N, K are positioned RIGHT after the
    // pointer args in the source. The kernel signature treats M as runtime
    // (i32) and N, K as runtime BUT constexpr-baked via our signature string
    // (specifically the signature places N and K as literal int values when
    // they're meant to be runtime). For safety, we pass them all as runtime
    // i32 since the kernel reads them as runtime values inside the body.
    //
    // After more careful reading: our signature build_signature_ml8 declares
    // positions 6-8 as `i32, i32, i32` (M, N, K) — all runtime. Constexprs
    // start at position 20 (GROUP_K) which the kernel reads as compile-time.
    // So the runtime arg list is everything up to position 19 + the LUT
    // pointer + stride_lut_k.

    hipDeviceptr_t a_ptr     = (hipDeviceptr_t)args->a_fp8;
    hipDeviceptr_t b_ptr     = (hipDeviceptr_t)args->b_packed;
    hipDeviceptr_t c_ptr     = (hipDeviceptr_t)args->c;
    hipDeviceptr_t as_ptr    = (hipDeviceptr_t)args->a_scale_fp32;
    hipDeviceptr_t bs_ptr    = (hipDeviceptr_t)args->b_scale_fp32;
    hipDeviceptr_t lut_ptr   = (hipDeviceptr_t)args->centroid_lut_fp8;
    int32_t M                = args->M;
    int32_t N                = args->shape.N;
    int32_t K                = args->shape.K;
    int32_t stride_am        = args->stride_am;
    int32_t stride_ak        = args->stride_ak;
    int32_t stride_bk        = args->stride_bk;
    int32_t stride_bn        = args->stride_bn;
    int32_t stride_ck        = 0;
    int32_t stride_cm        = args->stride_cm;
    int32_t stride_cn        = args->stride_cn;
    int32_t stride_ascale_m  = args->stride_ascale_m;
    int32_t stride_ascale_k  = 0;
    int32_t stride_bscale_k  = args->stride_bscale_k;
    int32_t stride_bscale_n  = args->stride_bscale_n;
    int32_t stride_lut_k     = args->stride_lut_k;
    // Triton 3.7+ appends two scratch pointers (null) after the user args
    // (see triton/tools/compile.py — the generated C launcher unconditionally
    // packs &global_scratch and &profile_scratch as the final two arg slots).
    hipDeviceptr_t p_global_scratch  = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_profile_scratch = (hipDeviceptr_t) nullptr;

    // #185: the signature value-specializes the unit strides (i32:1 →
    // constexpr), so stride_ak / stride_bn / stride_cn / stride_ascale_m /
    // stride_bscale_n are baked into the binary and DROPPED from the runtime
    // arg list. They must actually be 1 — enforced here, guaranteed by ml8.cu
    // (contiguous tensors only).
    if (stride_ak != 1 || stride_bn != 1 || stride_cn != 1 ||
        stride_ascale_m != 1 || stride_bscale_n != 1) {
        std::fprintf(stderr, "mt_ml8_gemm: non-unit inner stride (ak=%d bn=%d cn=%d asm=%d bsn=%d)"
                     " — signature specializes these to 1\n",
                     stride_ak, stride_bn, stride_cn, stride_ascale_m, stride_bscale_n);
        return hipErrorInvalidValue;
    }

    void * kernel_args[] = {
        &a_ptr, &b_ptr, &c_ptr, &as_ptr, &bs_ptr,
        &M, &N, &K,
        &stride_am,
        &stride_bk,
        &stride_ck, &stride_cm,
        &stride_ascale_k,
        &stride_bscale_k,
        &lut_ptr, &stride_lut_k,
        &p_global_scratch, &p_profile_scratch,
    };

    return handle->launch(stream, grid_x, 1, 1, kernel_args);
}

extern "C" void mt_ml8_gemm_reset_cache(void) {
    std::lock_guard<std::mutex> g(get_cache_mutex());
    get_handle_map().clear();
}
