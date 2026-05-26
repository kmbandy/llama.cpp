// mt_ml8_moe_gemm.cpp — runtime-shape ml8-4 MoE GEMM wrapper.
//
// Sibling of mt_ml8_gemm.cpp. Same dispatch pattern:
//   - build Triton signature from runtime shape at first call
//   - call aiter::Registry::get_or_compile() → KernelHandle
//   - cache the handle (one shape per process)
//   - launch via hipModuleLaunchKernel with manually-packed args[]
//
// Load-bearing pattern (MAD-243 / KG fact triton-37-trailing-scratch):
// kernel_args[] MUST end with &p_global_scratch, &p_profile_scratch (both
// nullptr) — Triton 3.7+'s C launcher (compile.py:185) unconditionally reads
// these from the args array. Omitting them = wild reads = host SIGSEGV.
//
// MAD-223 Phase C.3 / MAD-244.

#include "mt_ml8_moe_gemm.h"
#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

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

// Build the Triton signature for _moe_gemm_a8w8_blockscale with WEIGHT_FORMAT=1.
//
// Runtime args (in source-declaration order, AFTER constexprs are stripped):
//   1.  Y                    *bf16:16
//   2.  stride_y_k           i32
//   3.  stride_y_m           i32
//   4.  stride_y_n           i32
//   5.  X                    *fp8e4nv:16
//   6.  stride_x_m           i32
//   7.  stride_x_k           i32
//   8.  XBlockScale          *fp32:16
//   9.  stride_x_bs_m        i32
//   10. stride_x_bs_k        i32
//   11. W                    *i8:16
//   12. stride_w_e           i32
//   13. stride_w_k           i32
//   14. stride_w_n           i32
//   15. WBlockScale          *fp32:16
//   16. stride_w_bs_e        i32
//   17. stride_w_bs_k        i32
//   18. stride_w_bs_n        i32
//   19. X_static_scale       *fp32
//   20. W_static_scale       *fp32
//   21. Quant_static_scale   *fp32
//   22. B (bias)             *bf16
//   23. stride_b_e           i32
//   24. Gammas               *fp32
//   25. N                    i32
//   26. K                    i32
//   27. GatherIndx           *i32
//   28. ExptHist             *i32
//   29. ExptOffs             *i32
//   30. ExptOffsSum          *i32
//   31. ExptData             *i32
//   32. grid_m               i32
//   33. grid_n               i32
//   34. alpha                fp32   ← runtime fp32 scalar (between constexprs in source)
//   35. limit                fp32   ← runtime fp32 scalar
//   36. centroid_lut_ptr     *fp8e4nv:16
//   37. stride_lut_expert    i32
//   38. stride_lut_k         i32
//
// Constexprs (baked into signature as literal values; order matches the
// source kernel's declaration order, EXCLUDING runtime args):
//   APPLY_SWIGLU, ACTIVATION_REDUCTION_N, ADD_RESIDUAL, N_EXPTS_ACT,
//   BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M,
//   BLOCKSCALE_M, BLOCKSCALE_N, BLOCKSCALE_K,
//   XCD_SWIZZLE, EVEN_K, MASK_K_LIMIT, SPLIT_K,
//   UPCAST_INDICES, PER_ROW_X_SCALE, WEIGHT_FORMAT (=1), N_CENTROIDS
std::string build_signature_moe(const mt_ml8_moe_gemm_shape_t & s) {
    const int32_t block_k     = s.group_size;            // kernel constraint
    const int32_t blockscale_m = s.per_row_x_scale ? 1 : MT_ML8_MOE_BLOCK_M;
    const int32_t blockscale_n = 1;                       // matches ml8 calibration
    const int32_t blockscale_k = s.group_size;            // kernel asserts BLOCKSCALE_K == BLOCK_K

    char buf[4096];
    std::snprintf(buf, sizeof(buf),
        // Runtime args interspersed in source order. Triton's signature
        // string lists ALL params (constexprs and runtime) in source order;
        // constexprs are literal int/float values, runtime args are dtype
        // strings.
        //
        // 4 (Y + 3 strides) + 3 (X + 2 strides) + 3 (XBlockScale + 2 strides)
        "*bf16:16, i32, i32, i32, "                       // Y, stride_y_k, stride_y_m, stride_y_n
        "*fp8e4nv:16, i32, i32, "                         // X, stride_x_m, stride_x_k
        "*fp32:16, i32, i32, "                            // XBlockScale, stride_x_bs_m, stride_x_bs_k
        // 4 (W + 3 strides) + 4 (WBlockScale + 3 strides)
        "*i8:16, i32, i32, i32, "                         // W, stride_w_e/k/n
        "*fp32:16, i32, i32, i32, "                       // WBlockScale, stride_w_bs_e/k/n
        // 3 nullable static scales + bias (ptr + stride) + Gammas
        "*fp32, *fp32, *fp32, "                           // X/W/Quant_static_scale
        "*bf16, i32, "                                    // B (bias), stride_b_e
        "*fp32, "                                         // Gammas
        // shape dims
        "i32, i32, "                                      // N, K
        // routing
        "*i32, *i32, *i32, *i32, *i32, "                  // GatherIndx, ExptHist, ExptOffs, ExptOffsSum, ExptData
        "i32, i32, "                                      // grid_m, grid_n
        // ── APPLY_SWIGLU constexpr ──
        "%d, "
        // alpha, limit (runtime fp32 scalars, interleaved between constexprs in source)
        "fp32, fp32, "
        // ── ACTIVATION_REDUCTION_N, ADD_RESIDUAL, N_EXPTS_ACT constexprs ──
        "%d, %d, %d, "
        // ── BLOCK_M/N/K, GROUP_M constexprs ──
        "%d, %d, %d, %d, "
        // ── BLOCKSCALE_M/N/K constexprs ──
        "%d, %d, %d, "
        // ── XCD_SWIZZLE, EVEN_K, MASK_K_LIMIT, SPLIT_K constexprs ──
        "%d, %d, %d, %d, "
        // ── UPCAST_INDICES, PER_ROW_X_SCALE constexprs ──
        "%d, %d, "
        // ── LOCAL PATCH #6 feature-present flags (5): HAS_BIAS, HAS_GAMMAS,
        // HAS_X_STATIC_SCALE, HAS_W_STATIC_SCALE, HAS_QUANT_STATIC_SCALE ──
        "%d, %d, %d, %d, %d, "
        // ── WEIGHT_FORMAT (=1) + N_CENTROIDS constexprs ──
        "1, %d, "
        // ── ml8 LUT runtime args (per LOCAL PATCH #2): ptr + 2 strides ──
        "*fp8e4nv:16, i32, i32",
        s.apply_swiglu,
        s.activation_reduction_n,
        s.add_residual,
        s.n_expts_act,
        MT_ML8_MOE_BLOCK_M,
        MT_ML8_MOE_BLOCK_N,
        block_k,
        MT_ML8_MOE_GROUP_M,
        blockscale_m,
        blockscale_n,
        blockscale_k,
        MT_ML8_MOE_XCD_SWIZZLE,
        s.even_k,
        s.mask_k_limit,
        MT_ML8_MOE_SPLIT_K,
        s.upcast_indices,
        s.per_row_x_scale,
        s.has_bias,
        s.has_gammas,
        s.has_x_static_scale,
        s.has_w_static_scale,
        s.has_quant_static_scale,
        s.n_centroids);
    return buf;
}

struct CachedHandle {
    mt_ml8_moe_gemm_shape_t     shape       = {};
    const aiter::KernelHandle * handle      = nullptr;
    bool                        initialized = false;
    hipError_t                  init_err    = hipSuccess;
};

CachedHandle & get_cached() {
    static CachedHandle c;
    return c;
}

hipError_t ensure_initialized(const mt_ml8_moe_gemm_shape_t & shape) {
    CachedHandle & c = get_cached();
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (c.initialized) {
        if (std::memcmp(&c.shape, &shape, sizeof(shape)) != 0) {
            std::fprintf(stderr,
                "mt_ml8_moe_gemm: shape changed across calls. "
                "Single-process MoE cache supports one shape only. "
                "Call mt_ml8_moe_gemm_reset_cache() between shapes.\n");
            return hipErrorInvalidValue;
        }
        return c.init_err;
    }

    const std::string target = detect_hip_target();
    const std::string sig    = build_signature_moe(shape);

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    aiter::KernelSpec spec {
        MT_ML8_MOE_KERNEL_SOURCE,
        "_moe_gemm_a8w8_blockscale",
        target, sig, 4 /*num_warps*/, MT_ML8_MOE_NUM_STAGES,
    };

    int env_nw = 4, env_ns = MT_ML8_MOE_NUM_STAGES;
    if (const char * s = std::getenv("MT_ML8_MOE_NUM_WARPS"))  { int v = std::atoi(s); if (v > 0 && v <= 32) env_nw = v; }
    if (const char * s = std::getenv("MT_ML8_MOE_NUM_STAGES")) { int v = std::atoi(s); if (v > 0 && v <= 8 ) env_ns = v; }
    spec.num_warps  = env_nw;
    spec.num_stages = env_ns;

    c.handle = reg.get_or_compile(spec);
    if (!c.handle) {
        std::fprintf(stderr, "mt_ml8_moe_gemm: kernel compile failed\n");
        c.init_err = hipErrorInvalidValue;
    } else {
        c.init_err = hipSuccess;
    }
    c.shape       = shape;
    c.initialized = true;
    return c.init_err;
}

}  // namespace


extern "C" hipError_t mt_ml8_moe_gemm(hipStream_t stream, const mt_ml8_moe_gemm_args_t * args) {
    if (!args) return hipErrorInvalidValue;

    if (args->M % MT_ML8_MOE_BLOCK_M != 0) {
        std::fprintf(stderr,
            "mt_ml8_moe_gemm: M (%d) must be multiple of %d\n",
            args->M, MT_ML8_MOE_BLOCK_M);
        return hipErrorInvalidValue;
    }
    if (args->shape.N % MT_ML8_MOE_BLOCK_N != 0) {
        std::fprintf(stderr,
            "mt_ml8_moe_gemm: shape.N (%d) must be multiple of %d\n",
            args->shape.N, MT_ML8_MOE_BLOCK_N);
        return hipErrorInvalidValue;
    }
    if (args->shape.K % args->shape.group_size != 0) {
        std::fprintf(stderr,
            "mt_ml8_moe_gemm: shape.K (%d) must be multiple of group_size (%d)\n",
            args->shape.K, args->shape.group_size);
        return hipErrorInvalidValue;
    }

    hipError_t init_rc = ensure_initialized(args->shape);
    if (init_rc != hipSuccess) return init_rc;

    CachedHandle & c = get_cached();
    if (!c.handle) return hipErrorInvalidValue;

    // Grid: total_tiles * SPLIT_K (1D dispatch; matches Phase B.5 test)
    const unsigned int grid_x =
        (unsigned int)(args->grid_m * args->grid_n * MT_ML8_MOE_SPLIT_K);

    // Stack-local copies for stable pointer-to-arg slots.
    hipDeviceptr_t p_y           = (hipDeviceptr_t)args->y;
    hipDeviceptr_t p_x           = (hipDeviceptr_t)args->x_fp8;
    hipDeviceptr_t p_xbs         = (hipDeviceptr_t)args->x_scale_fp32;
    hipDeviceptr_t p_w           = (hipDeviceptr_t)args->w_packed;
    hipDeviceptr_t p_wbs         = (hipDeviceptr_t)args->w_scale_fp32;
    hipDeviceptr_t p_xsstatic    = (hipDeviceptr_t)args->x_static_scale;
    hipDeviceptr_t p_wsstatic    = (hipDeviceptr_t)args->w_static_scale;
    hipDeviceptr_t p_qsstatic    = (hipDeviceptr_t)args->quant_static_scale;
    hipDeviceptr_t p_bias        = (hipDeviceptr_t)args->bias;
    hipDeviceptr_t p_gammas      = (hipDeviceptr_t)args->gammas;
    hipDeviceptr_t p_gather      = (hipDeviceptr_t)args->gather_indx;
    hipDeviceptr_t p_ehist       = (hipDeviceptr_t)args->expt_hist;
    hipDeviceptr_t p_eoffs       = (hipDeviceptr_t)args->expt_offs;
    hipDeviceptr_t p_eoffs_sum   = (hipDeviceptr_t)args->expt_offs_sum;
    hipDeviceptr_t p_edata       = (hipDeviceptr_t)args->expt_data;
    hipDeviceptr_t p_lut         = (hipDeviceptr_t)args->centroid_lut_fp8;

    int32_t stride_y_k       = args->stride_y_k;
    int32_t stride_y_m       = args->stride_y_m;
    int32_t stride_y_n       = args->stride_y_n;
    int32_t stride_x_m       = args->stride_x_m;
    int32_t stride_x_k       = args->stride_x_k;
    int32_t stride_x_bs_m    = args->stride_x_bs_m;
    int32_t stride_x_bs_k    = args->stride_x_bs_k;
    int32_t stride_w_e       = args->stride_w_e;
    int32_t stride_w_k       = args->stride_w_k;
    int32_t stride_w_n       = args->stride_w_n;
    int32_t stride_w_bs_e    = args->stride_w_bs_e;
    int32_t stride_w_bs_k    = args->stride_w_bs_k;
    int32_t stride_w_bs_n    = args->stride_w_bs_n;
    int32_t stride_b_e       = args->stride_b_e;
    int32_t N                = args->shape.N;
    int32_t K                = args->shape.K;
    int32_t grid_m           = args->grid_m;
    int32_t grid_n           = args->grid_n;
    float   alpha            = args->alpha;
    float   limit            = args->limit;
    int32_t stride_lut_expt  = args->stride_lut_expert;
    int32_t stride_lut_k     = args->stride_lut_k;

    // Triton 3.7+ trailing scratch pointers (load-bearing — MAD-243 rule).
    hipDeviceptr_t p_global_scratch  = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_profile_scratch = (hipDeviceptr_t) nullptr;

    void * kernel_args[] = {
        // Y + strides (4)
        &p_y, &stride_y_k, &stride_y_m, &stride_y_n,
        // X + strides (3)
        &p_x, &stride_x_m, &stride_x_k,
        // XBlockScale + strides (3)
        &p_xbs, &stride_x_bs_m, &stride_x_bs_k,
        // W + strides (4)
        &p_w, &stride_w_e, &stride_w_k, &stride_w_n,
        // WBlockScale + strides (4)
        &p_wbs, &stride_w_bs_e, &stride_w_bs_k, &stride_w_bs_n,
        // static scales (3)
        &p_xsstatic, &p_wsstatic, &p_qsstatic,
        // Bias (1 ptr + 1 stride) + Gammas (1 ptr)
        &p_bias, &stride_b_e, &p_gammas,
        // N, K (2)
        &N, &K,
        // routing (5 ptrs)
        &p_gather, &p_ehist, &p_eoffs, &p_eoffs_sum, &p_edata,
        // grid_m, grid_n (2)
        &grid_m, &grid_n,
        // alpha, limit (2 runtime fp32 scalars)
        &alpha, &limit,
        // ml8 LUT runtime args (3)
        &p_lut, &stride_lut_expt, &stride_lut_k,
        // Triton 3.7+ trailing scratch (2)
        &p_global_scratch, &p_profile_scratch,
    };

    return c.handle->launch(stream, grid_x, 1, 1, kernel_args);
}

extern "C" void mt_ml8_moe_gemm_reset_cache(void) {
    CachedHandle & c = get_cached();
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    c.shape       = {};
    c.handle      = nullptr;
    c.initialized = false;
    c.init_err    = hipSuccess;
}
