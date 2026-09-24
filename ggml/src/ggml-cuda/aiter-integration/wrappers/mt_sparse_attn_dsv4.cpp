// mt_sparse_attn_dsv4.cpp — see header for design notes.
//
// Mirrors mt_ml8_gemm.cpp's pattern: build a Triton signature (here fixed,
// not shape-derived -- this op is bound to one model shape), call
// aiter::Registry::get_or_compile() once per kernel (prefill / decode),
// cache the handles, and launch via hipModuleLaunchKernel with manually
// packed args[] -- remembering Triton's two trailing implicit scratch-pointer
// kernarg slots (verified via `llvm-readobj --notes kernel.hsaco` in Stage 2;
// see scratchpad/sparse-attn/test_aot_ctypes.py's comment for how that was
// found).
//
// Signatures / configs below are copy-pasted from the exact strings Stage 2
// validated stand-alone (scratchpad/sparse-attn/test_aot_ctypes.py +
// scratchpad/sparse-attn/aot-out/{prefill_v3,decode}/meta.json) -- not
// re-derived here.

#include "mt_sparse_attn_dsv4.h"
#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

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

const aiter::KernelHandle * get_prefill_handle() {
    static std::mutex mu;
    static const aiter::KernelHandle * handle = nullptr;
    static bool tried = false;
    std::lock_guard<std::mutex> g(mu);
    if (tried) return handle;
    tried = true;

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    // Stage 2 v3 signature: q_stride_d/kv_stride_d/out_stride_d hinted ":1"
    // (baked to the compile-time constant 1 -- required for the fix to the
    // narrow-load AOT trap), num_heads=64/head_dim=512 baked as bare
    // literals (same fix), the rest of the strides hinted ":16" (all
    // provably divisible for this fixed shape).
    aiter::KernelSpec spec;
    spec.source_path = MT_SPARSE_ATTN_DSV4_PREFILL_SOURCE;
    spec.kernel_name  = "_sparse_attn_prefill_kernel";
    spec.target       = detect_hip_target();
    spec.signature    =
        "*fp16:16, *fp16:16, *i32:16, *i32:16, *fp32:16, *fp16:16, "
        "i64:16, i64:16, i64:1, i64:16, i64:1, i64:16, i64:16, i64:1, "
        "64, 512, i32, fp32, 1, 32, 512, 16";
    spec.num_warps            = 4;
    spec.num_stages           = 1;
    spec.matrix_instr_nonkdim = 16;

    handle = reg.get_or_compile(spec);
    if (!handle) {
        std::fprintf(stderr, "mt_sparse_attn_dsv4: prefill kernel compile failed\n");
    }
    return handle;
}

const aiter::KernelHandle * get_decode_handle() {
    static std::mutex mu;
    static const aiter::KernelHandle * handle = nullptr;
    static bool tried = false;
    std::lock_guard<std::mutex> g(mu);
    if (tried) return handle;
    tried = true;

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    // KV_SPLITS=1 baked (Stage 2 engineering decision: single-CTA path is
    // T-independent and only ~30% slower than the split-K path's best case,
    // and avoids needing a T-keyed family of AOT specializations for
    // KV_SPLITS, which varies with T in the split-K formula). Every stride /
    // shape / scale argument is baked as a literal (bare numbers, not
    // "i64:N" hints) since they're all compile-time-fixed for this shape --
    // see the runtime_args introspection in the Stage 2 debugging notes:
    // only the 10 pointers + total_pages remain non-constexpr.
    aiter::KernelSpec spec;
    spec.source_path = MT_SPARSE_ATTN_DSV4_DECODE_SOURCE;
    spec.kernel_name  = "_pa_decode_sparse";
    spec.target       = detect_hip_target();
    spec.signature    =
        "*fp16:16, *fp16:16, *fp32:16, *i32:16, *i32:16, *fp32:16, *fp32:16, "
        "*fp32:16, *fp32:16, *fp16:16, i32, "
        "32768, 512, 1, 512, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "
        "32768, 512, 1, 64, 512, 1, 0.04419417382415922, "
        "16, 512, 16, 1, 0, 64, 1, 1, 4";
    spec.num_warps  = 4;
    spec.num_stages = 2;

    handle = reg.get_or_compile(spec);
    if (!handle) {
        std::fprintf(stderr, "mt_sparse_attn_dsv4: decode kernel compile failed\n");
    }
    return handle;
}

}  // namespace

// kv_indptr ([0, n_idx, 2*n_idx, ..., n_tokens*n_idx], I32) is built in the
// ggml graph by the caller, so launching needs no host-side allocation, copy
// or stream sync -- safe under HIP graph capture.
extern "C" hipError_t mt_sparse_attn_dsv4(hipStream_t stream,
                                           const struct mt_sparse_attn_dsv4_args_t * args) {
    if (args->n_tokens <= 0) return hipSuccess;

    const int32_t * kv_indptr = args->kv_indptr;

    hipDeviceptr_t p_global_scratch  = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_profile_scratch = (hipDeviceptr_t) nullptr;

    if (args->n_tokens >= MT_SPARSE_ATTN_DSV4_PREFILL_MIN_T) {
        const aiter::KernelHandle * h = get_prefill_handle();
        if (!h) return hipErrorInvalidValue;

        hipDeviceptr_t q_ptr          = (hipDeviceptr_t) args->q;
        hipDeviceptr_t kv_ptr         = (hipDeviceptr_t) args->k_all;
        hipDeviceptr_t kv_indices_ptr = (hipDeviceptr_t) args->kv_indices;
        hipDeviceptr_t kv_indptr_ptr  = (hipDeviceptr_t) kv_indptr;
        hipDeviceptr_t attn_sink_ptr  = (hipDeviceptr_t) args->attn_sink;
        hipDeviceptr_t out_ptr        = (hipDeviceptr_t) args->out;
        int64_t q_stride_t  = MT_SPARSE_ATTN_DSV4_NUM_HEADS * MT_SPARSE_ATTN_DSV4_HEAD_DIM;
        int64_t q_stride_h  = MT_SPARSE_ATTN_DSV4_HEAD_DIM;
        int64_t kv_stride_n = MT_SPARSE_ATTN_DSV4_HEAD_DIM;
        int64_t out_stride_t = q_stride_t;
        int64_t out_stride_h = q_stride_h;
        int32_t num_kv = args->n_kv;
        float   scale  = args->scale;

        void * kernel_args[] = {
            &q_ptr, &kv_ptr, &kv_indices_ptr, &kv_indptr_ptr, &attn_sink_ptr, &out_ptr,
            &q_stride_t, &q_stride_h, &kv_stride_n, &out_stride_t, &out_stride_h,
            &num_kv, &scale,
            &p_global_scratch, &p_profile_scratch,
        };
        // grid = (n_tokens, cdiv(num_heads=64, BLOCK_H=32)=2, 1)
        return h->launch(stream, (unsigned) args->n_tokens, 2, 1, kernel_args);
    } else {
        const aiter::KernelHandle * h = get_decode_handle();
        if (!h) return hipErrorInvalidValue;

        hipDeviceptr_t q_ptr          = (hipDeviceptr_t) args->q;
        hipDeviceptr_t kv_ptr         = (hipDeviceptr_t) args->k_all;
        hipDeviceptr_t kv_scales_ptr  = (hipDeviceptr_t) nullptr;  // dummy: QUANT_KV=0 baked, dead code
        hipDeviceptr_t kv_indices_ptr = (hipDeviceptr_t) args->kv_indices;
        hipDeviceptr_t kv_indptr_ptr  = (hipDeviceptr_t) kv_indptr;
        hipDeviceptr_t dummy_mla_ptr  = (hipDeviceptr_t) args->out;  // m/l/acc partials: dead code (KV_SPLITS=1 baked)
        hipDeviceptr_t attn_sink_ptr  = (hipDeviceptr_t) args->attn_sink;
        hipDeviceptr_t out_ptr        = (hipDeviceptr_t) args->out;
        int32_t total_pages = args->n_kv;

        void * kernel_args[] = {
            &q_ptr, &kv_ptr, &kv_scales_ptr, &kv_indices_ptr, &kv_indptr_ptr,
            &dummy_mla_ptr, &dummy_mla_ptr, &dummy_mla_ptr,
            &attn_sink_ptr, &out_ptr,
            &total_pages,
            &p_global_scratch, &p_profile_scratch,
        };
        // grid = (n_tokens, cdiv(num_heads=64, BLOCK_H=16)=4, KV_SPLITS=1)
        return h->launch(stream, (unsigned) args->n_tokens, 4, 1, kernel_args);
    }
}
