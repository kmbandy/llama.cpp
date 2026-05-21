// mt_aiter_unified_attn.cpp — runtime-shape AITER wrapper.
//
// The wrapper takes the model's shape at runtime (head_size, num_q_heads,
// num_kv_heads, block_size) and builds the Triton signature from those at
// first call. The runtime registry compiles a matching kernel (or hits the
// disk cache on a warm restart) and we launch via hipModuleLaunchKernel with
// a manually-packed args[] array.
//
// Shape is captured at first call and asserted-equal on subsequent calls —
// one process, one shape. (Future: per-shape handle map if we ever serve
// multiple models from one process.)
//
// MAD-188.

#include "mt_aiter_unified_attn.h"
#include "aiter_runtime_compiler.h"

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace {

// Derive the Triton target string ("hip:<arch>:<wave_size>") from the
// currently-active HIP device.
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

// Build the 3D-kernel Triton signature for the given model shape.
// Substitutions vs. the AITER signature template:
//   pos 17 → num_q_heads
//   pos 18 → num_queries_per_kv (= num_q_heads / num_kv_heads)
//   pos 21 → head_size (query_stride_1 constexpr)
//   pos 23 → block_size (BLOCK_SIZE constexpr)
//   pos 25 → head_size (HEAD_SIZE constexpr)
//   pos 26 → head_size (HEAD_SIZE_PADDED constexpr; assumes head_size is pow2)
std::string build_signature_3d(const mt_aiter_uattn_shape_t & s) {
    // MAD-199: K/V cache pointer dtype depends on cache_type. F16 stays
    // `*fp16:16` (upstream signature); turbo3/turbo4 switch to `*i8:166` byte
    // pointers and bake CACHE_TYPE=1/2 as the kernel constexpr — both branches
    // hash to distinct AOT artifacts (or distinct runtime-compile cache keys).
    const char * kv_ptr_dtype;
    int          cache_type_val;
    switch (s.cache_type) {
        case MT_AITER_CACHE_F16:
            kv_ptr_dtype   = "*fp16:16";
            cache_type_val = 0;
            break;
        case MT_AITER_CACHE_TURBO3:
            kv_ptr_dtype   = "*i8:166";
            cache_type_val = 1;
            break;
        case MT_AITER_CACHE_TURBO4:
            kv_ptr_dtype   = "*i8:166";
            cache_type_val = 2;
            break;
        // MAD-214: turbo-FP8 family. Production-wired variants only (BS=256).
        // BS<256 variants throw at compile time via tl.static_assert in
        // unified_attention.py — MAD-215 wires those.
        case MT_AITER_CACHE_TURBO3_FP8_BS256:
            kv_ptr_dtype   = "*i8:166";
            cache_type_val = 14;
            break;
        case MT_AITER_CACHE_TURBO4_FP8_BS256:
            kv_ptr_dtype   = "*i8:166";
            cache_type_val = 24;
            break;
        case MT_AITER_CACHE_TURBO5_FP8_BS256:
            kv_ptr_dtype   = "*i8:166";
            cache_type_val = 34;
            break;
        default:
            kv_ptr_dtype   = "*fp16:16";
            cache_type_val = 0;
            break;
    }

    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "*fp32:16, *fp32, *fp32, *fp16:16, %s, %s, *fp32, *i32, *i32, "
        "*fp32, *fp16, fp32, *fp32, *fp32, *fp32, fp32, "
        "%d, %d, "                          // num_q_heads, num_queries_per_kv
        "i64, i64, %d, i64, "               // block_table_stride, q_stride_0, query_stride_1=head_size, qq_bias_stride_0
        "%d, %d, %d, %d, "                  // BLOCK_SIZE, TILE_SIZE, HEAD_SIZE, HEAD_SIZE_PADDED
        "0, 0, 0, 0, 0, "                   // USE_ALIBI / QQ / SOFTCAP / SINKS / SLIDING_WINDOW
        "i64, i64, i64, 1, i64, i64, i64, 1, "  // k/v cache strides (last is constexpr=1; for turbo these args are present but unused — helper computes byte strides internally)
        "*i32, %d, i32, %d, %d, 1, %d, "    // query_start_len, BLOCK_Q, num_seqs(runtime), BLOCK_M, NUM_SEGMENTS, ALL_DECODE, CACHE_TYPE (MAD-199)
        "*i8:16, *i8:16",                     // MAD-214: centroids_k_ptr, centroids_v_ptr (None-safe for non-FP8)
        kv_ptr_dtype,                           // K cache pointer dtype
        kv_ptr_dtype,                           // V cache pointer dtype
        s.num_q_heads,                          // num_q_heads constexpr
        s.num_q_heads / s.num_kv_heads,         // num_queries_per_kv constexpr
        s.head_size,                            // query_stride_1 = head_size constexpr
        s.block_size,                           // BLOCK_SIZE constexpr
        MT_AITER_UATTN_TILE_SIZE,               // TILE_SIZE constexpr
        s.head_size,                            // HEAD_SIZE constexpr
        s.head_size,                            // HEAD_SIZE_PADDED (assumes head_size is pow2)
        MT_AITER_UATTN_BLOCK_Q,                 // BLOCK_Q constexpr
        MT_AITER_UATTN_BLOCK_M,                 // BLOCK_M constexpr
        MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ,    // NUM_SEGMENTS_PER_SEQ constexpr
        cache_type_val);                        // CACHE_TYPE constexpr (MAD-199)
    return buf;
}

// MAD-199 chunk D3: 2D-kernel signature for prefill dispatch.
// kernel_unified_attention_2d is single-pass-per-(kv_head, q_block) — no
// split-K, no reduce phase. Better for prefill (q_len >> 1) where each Q
// block has plenty of work; the 3D kernel's split-K reduction overhead
// dominates instead of helping at this size.
//
// MAD-203: BLOCK_M / BLOCK_Q are now parameters so the caller can request
// either the "base" prefill spec (16/2) or the "large" prefill spec (64/8)
// per the upstream host dispatcher's max_seqlen_q >= 256 rule.
//
// Same K/V cache pointer dtype switch as build_signature_3d (cache_type
// drives *fp16:16 vs *i8:166 and the CACHE_TYPE constexpr).
std::string build_signature_2d(const mt_aiter_uattn_shape_t & s,
                                int block_m, int block_q) {
    const char * kv_ptr_dtype;
    int          cache_type_val;
    switch (s.cache_type) {
        case MT_AITER_CACHE_F16:               kv_ptr_dtype = "*fp16:16"; cache_type_val =  0; break;
        case MT_AITER_CACHE_TURBO3:            kv_ptr_dtype = "*i8:166";   cache_type_val =  1; break;
        case MT_AITER_CACHE_TURBO4:            kv_ptr_dtype = "*i8:166";   cache_type_val =  2; break;
        // MAD-214: turbo-FP8 family (BS=256 production variants only;
        // BS<256 covered by MAD-215). Numeric values match mt_aiter_cache_type.
        case MT_AITER_CACHE_TURBO3_FP8_BS256:  kv_ptr_dtype = "*i8:166";   cache_type_val = 14; break;
        case MT_AITER_CACHE_TURBO4_FP8_BS256:  kv_ptr_dtype = "*i8:166";   cache_type_val = 24; break;
        case MT_AITER_CACHE_TURBO5_FP8_BS256:  kv_ptr_dtype = "*i8:166";   cache_type_val = 34; break;
        default:                               kv_ptr_dtype = "*fp16:16"; cache_type_val =  0; break;
    }
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "*fp16:16, *fp16:16, %s, %s, *fp32, *i32, *i32, *fp32, *fp16, "  // out, q, k, v, sink, bt, sl, alibi, qq_bias
        "fp32, *fp32, *fp32, *fp32, *fp32, fp32, "                       // scale, q/k/v_descale, out_scale, softcap
        "%d, %d, "                                                       // num_q_heads, num_queries_per_kv
        "i64, i64, %d, i64, %d, i64, "                                   // bt_stride, q_stride_0, q_stride_1=head_size, out_stride_0, out_stride_1=head_size, qq_bias_stride_0
        "%d, %d, %d, %d, "                                               // BLOCK_SIZE, TILE_SIZE, HEAD_SIZE, HEAD_SIZE_PADDED
        "0, 0, 0, 0, 0, "                                                // USE_ALIBI / QQ / SOFTCAP / SINKS / SLIDING_WINDOW
        "i64, i64, i64, 1, i64, i64, i64, 1, "                           // k/v cache strides (last is constexpr=1)
        "*i32, %d, i32, %d, "                                            // query_start_len, BLOCK_Q, num_seqs(runtime), BLOCK_M
        "-448.0, 448.0, 0, %d, "                                         // FP8_MIN, FP8_MAX, ALL_DECODE=0 (prefill), CACHE_TYPE
        "*i8:16, *i8:16",                                                  // MAD-214: centroids_k_ptr, centroids_v_ptr
        kv_ptr_dtype, kv_ptr_dtype,
        s.num_q_heads, s.num_q_heads / s.num_kv_heads,
        s.head_size, s.head_size,           // query_stride_1, output_stride_1
        s.block_size,                       // BLOCK_SIZE
        MT_AITER_UATTN_TILE_SIZE,           // TILE_SIZE
        s.head_size, s.head_size,           // HEAD_SIZE, HEAD_SIZE_PADDED
        block_q,                            // BLOCK_Q
        block_m,                            // BLOCK_M
        cache_type_val);                    // CACHE_TYPE
    return buf;
}

std::string build_signature_reduce(const mt_aiter_uattn_shape_t & s) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "*fp16:16, *fp32:16, *fp32, *fp32, *i32, i32, "
        "%d, "                              // num_query_heads
        "*fp32, i64, %d, i64, "             // out_scale, output_stride_0, output_stride_1=head_size, block_table_stride
        "%d, %d, %d, "                      // TILE_SIZE, HEAD_SIZE, HEAD_SIZE_PADDED
        "*i32, %d, %d, "                    // query_start_len, BLOCK_Q, NUM_SEGMENTS
        "-448.0, 448.0",                    // FP8_MIN, FP8_MAX
        s.num_q_heads,
        s.head_size,
        MT_AITER_UATTN_TILE_SIZE,
        s.head_size,
        s.head_size,
        MT_AITER_UATTN_BLOCK_Q,
        MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ);
    return buf;
}

struct CachedHandles {
    mt_aiter_uattn_shape_t      shape       = {};
    const aiter::KernelHandle * h_3d        = nullptr;
    const aiter::KernelHandle * h_reduce    = nullptr;
    const aiter::KernelHandle * h_2d        = nullptr;  // MAD-199 D3: base prefill (BLOCK_M=16, BLOCK_Q=2)
    const aiter::KernelHandle * h_2d_large  = nullptr;  // MAD-203:    large prefill (BLOCK_M=64, BLOCK_Q=8)
    bool                        initialized = false;
    hipError_t                  init_err    = hipSuccess;
};

CachedHandles & get_cached() {
    static CachedHandles c;
    return c;
}

// Initialize on first call: build signatures from the shape we're handed and
// request kernel handles from the registry. Subsequent calls must use the
// same shape — assert and abort otherwise.
hipError_t ensure_initialized(const mt_aiter_uattn_shape_t & shape) {
    CachedHandles & c = get_cached();
    static std::mutex mu;
    std::lock_guard<std::mutex> g(mu);
    if (c.initialized) {
        // Sanity: shape must match. If a single process ever needs multiple
        // shapes we'll upgrade to a per-shape handle map, but for now this is
        // a guardrail against silent misdispatch.
        if (std::memcmp(&c.shape, &shape, sizeof(shape)) != 0) {
            std::fprintf(stderr,
                "mt_aiter_unified_attn: shape changed across calls (was %d/%d/%d/%d, "
                "now %d/%d/%d/%d). Single-process AITER cache supports one shape only.\n",
                c.shape.head_size, c.shape.num_q_heads, c.shape.num_kv_heads, c.shape.block_size,
                shape.head_size,   shape.num_q_heads,   shape.num_kv_heads,   shape.block_size);
            return hipErrorInvalidValue;
        }
        return c.init_err;
    }

    const std::string target  = detect_hip_target();
    const std::string sig_3d  = build_signature_3d(shape);
    const std::string sig_red = build_signature_reduce(shape);

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    aiter::KernelSpec spec_3d {
        AITER_KERNEL_SOURCE_DEFAULT,
        "kernel_unified_attention_3d",
        target, sig_3d, 4, 1,
    };
    c.h_3d = reg.get_or_compile(spec_3d);

    aiter::KernelSpec spec_reduce {
        AITER_KERNEL_SOURCE_DEFAULT,
        "reduce_segments",
        target, sig_red, 4, 1,
    };
    c.h_reduce = reg.get_or_compile(spec_reduce);

    // MAD-199 D3: 2D base prefill spec (BLOCK_M=16, BLOCK_Q=2).
    const std::string sig_2d = build_signature_2d(shape, MT_AITER_UATTN_BLOCK_M, MT_AITER_UATTN_BLOCK_Q);
    aiter::KernelSpec spec_2d {
        AITER_KERNEL_SOURCE_DEFAULT,
        "kernel_unified_attention_2d",
        target, sig_2d, 4, 1,
    };
    c.h_2d = reg.get_or_compile(spec_2d);

    // MAD-203: 2D large-prefill spec (BLOCK_M=64, BLOCK_Q=8). Per the
    // upstream host dispatcher's large-prefill branch (max_seqlen_q >= 256):
    // 4× LDS reuse vs the base spec. Single best win expected from MAD-203.
    // Requires num_queries_per_kv == 8 for BLOCK_Q=8 to be valid; assert.
    if (shape.num_kv_heads > 0 &&
        (shape.num_q_heads / shape.num_kv_heads) == 8) {
        const std::string sig_2d_large = build_signature_2d(
            shape, MT_AITER_UATTN_BLOCK_M_LARGE, MT_AITER_UATTN_BLOCK_Q_LARGE);
        aiter::KernelSpec spec_2d_large {
            AITER_KERNEL_SOURCE_DEFAULT,
            "kernel_unified_attention_2d",
            target, sig_2d_large, 4, 1,
        };
        c.h_2d_large = reg.get_or_compile(spec_2d_large);
    } else {
        // For non-8-GQA shapes, fall back to the base spec for large prefill too.
        // BLOCK_Q must equal BLOCK_M / num_queries_per_kv; with our hardcoded
        // BLOCK_Q_LARGE=8 this only matches GQA=8. Generalizing is MAD-203 phase 2.
        c.h_2d_large = c.h_2d;
    }

    c.shape       = shape;
    c.initialized = true;
    if (!c.h_3d || !c.h_reduce || !c.h_2d || !c.h_2d_large) {
        std::fprintf(stderr,
            "mt_aiter_unified_attn: registry could not compile/load kernels "
            "(3d=%p, reduce=%p, 2d=%p, 2d_large=%p, target=%s, h=%d nq=%d nkv=%d bs=%d ct=%d)\n",
            (const void*)c.h_3d, (const void*)c.h_reduce, (const void*)c.h_2d, (const void*)c.h_2d_large,
            target.c_str(),
            shape.head_size, shape.num_q_heads, shape.num_kv_heads, shape.block_size, shape.cache_type);
        c.init_err = hipErrorInvalidImage;
    }
    return c.init_err;
}

}  // anonymous namespace

hipError_t mt_aiter_unified_attn(hipStream_t stream,
                                  const mt_aiter_uattn_args_t *a) {
    hipError_t init_err = ensure_initialized(a->shape);
    if (init_err != hipSuccess) return init_err;
    const CachedHandles & c = get_cached();
    if (!c.h_3d || !c.h_reduce || !c.h_2d || !c.h_2d_large) return hipErrorInvalidImage;

    // MAD-199 D3 + MAD-203: three-way dispatch.
    //
    // 3D split-K is only a win for short q (decode); for prefill the
    // per-segment overhead and reduce-segments pass dominate. Within the
    // 2D path, switching to BLOCK_M=64 / BLOCK_Q=8 (large-prefill spec)
    // gives 4× LDS reuse vs the base BLOCK_M=16 spec, but BLOCK_Q=8 only
    // pays off when each Q block has ≥256 tokens to chew through (matches
    // upstream's max_seqlen_q >= 256 cutover).
    const int32_t avg_q_len    = (a->num_seqs > 0) ? (a->num_q_tokens / a->num_seqs) : 1;
    const bool    use_2d       = (avg_q_len >= MT_AITER_UATTN_BLOCK_Q);
    const bool    use_2d_large = use_2d && (avg_q_len >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD);

    // ── 3D split-K phase ───────────────────────────────────────────────────
    hipDeviceptr_t p_segm_out    = (hipDeviceptr_t) a->segm_output;
    hipDeviceptr_t p_segm_max    = (hipDeviceptr_t) a->segm_max;
    hipDeviceptr_t p_segm_expsum = (hipDeviceptr_t) a->segm_expsum;
    hipDeviceptr_t p_q           = (hipDeviceptr_t) a->q;
    hipDeviceptr_t p_k           = (hipDeviceptr_t) a->k_cache;
    hipDeviceptr_t p_v           = (hipDeviceptr_t) a->v_cache;
    hipDeviceptr_t p_sink        = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_bt          = (hipDeviceptr_t) a->block_tables;
    hipDeviceptr_t p_sl          = (hipDeviceptr_t) a->seq_lens;
    hipDeviceptr_t p_alibi       = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_qq_bias     = (hipDeviceptr_t) nullptr;
    float          scale_f       = a->scale;
    hipDeviceptr_t p_qd          = (hipDeviceptr_t) a->q_descale;
    hipDeviceptr_t p_kd          = (hipDeviceptr_t) a->k_descale;
    hipDeviceptr_t p_vd          = (hipDeviceptr_t) a->v_descale;
    float          softcap_f     = 0.0f;
    int64_t        bts           = a->block_table_stride;
    int64_t        qs0           = a->q_stride_0;
    int64_t        qqs0          = 0;
    int64_t        ks0           = a->k_stride_0;
    int64_t        ks1           = a->k_stride_1;
    int64_t        ks2           = a->k_stride_2;
    int64_t        vs0           = a->v_stride_0;
    int64_t        vs1           = a->v_stride_1;
    int64_t        vs2           = a->v_stride_2;
    hipDeviceptr_t p_cu          = (hipDeviceptr_t) a->query_start_len;
    int32_t        num_seqs      = a->num_seqs;
    // MAD-214: turbo-FP8 centroid LUT pointers. NULL'd defensively for non-FP8
    // cache types — kernel branch dead-eliminates the deref so safe regardless,
    // but explicit NULL avoids passing struct-uninit garbage to the kernel.
    hipDeviceptr_t p_centroids_k = (hipDeviceptr_t) (
        mt_aiter_cache_is_turbo_fp8(a->shape.cache_type) ? a->centroids_k : nullptr);
    hipDeviceptr_t p_centroids_v = (hipDeviceptr_t) (
        mt_aiter_cache_is_turbo_fp8(a->shape.cache_type) ? a->centroids_v : nullptr);
    // Triton 3.7+ appends two scratch pointers (null) after the user args.
    hipDeviceptr_t p_global_scratch  = (hipDeviceptr_t) nullptr;
    hipDeviceptr_t p_profile_scratch = (hipDeviceptr_t) nullptr;

    const int32_t num_q_tokens = a->num_q_tokens > 0 ? a->num_q_tokens : num_seqs;

    // MAD-199 D3: Prefill path — single-pass 2D kernel, no segm bufs, no reduce.
    // MAD-203: select between base (BLOCK_M=16, BLOCK_Q=2) and large (BLOCK_M=64,
    // BLOCK_Q=8) spec based on avg_q_len.
    if (use_2d) {
        hipDeviceptr_t p_out = (hipDeviceptr_t) a->out;
        hipDeviceptr_t p_os  = (hipDeviceptr_t) a->out_scale;
        int64_t        os0   = a->output_stride_0;

        void *args_2d[] = {
            &p_out, &p_q, &p_k, &p_v,
            &p_sink,
            &p_bt, &p_sl,
            &p_alibi, &p_qq_bias,
            &scale_f,
            &p_qd, &p_kd, &p_vd, &p_os,
            &softcap_f,
            &bts, &qs0, &os0, &qqs0,
            &ks0, &ks1, &ks2,
            &vs0, &vs1, &vs2,
            &p_cu, &num_seqs,
            &p_centroids_k, &p_centroids_v,  // MAD-214: turbo-FP8 per-layer LUT pointers
            &p_global_scratch, &p_profile_scratch,
        };

        // 2D grid: (n_kv_heads, num_q_blocks). num_q_blocks = num_q_tokens/BLOCK_Q + num_seqs
        // per upstream's upper-bound formula (see unified_attention_host_reference.py).
        // BLOCK_Q differs between base (2) and large (8) specs — must match the
        // selected handle.
        const int32_t block_q_for_grid = use_2d_large
            ? MT_AITER_UATTN_BLOCK_Q_LARGE
            : MT_AITER_UATTN_BLOCK_Q;
        const aiter::KernelHandle * h_2d_selected = use_2d_large ? c.h_2d_large : c.h_2d;

        unsigned int g2_x = (unsigned int) a->shape.num_kv_heads;
        unsigned int g2_y = (unsigned int)(num_q_tokens / block_q_for_grid + num_seqs);
        unsigned int g2_z = 1;
        return h_2d_selected->launch(stream, g2_x, g2_y, g2_z, args_2d);
    }

    void *args_3d[] = {
        &p_segm_out, &p_segm_max, &p_segm_expsum,
        &p_q, &p_k, &p_v,
        &p_sink,
        &p_bt, &p_sl,
        &p_alibi, &p_qq_bias,
        &scale_f,
        &p_qd, &p_kd, &p_vd,
        &softcap_f,
        &bts, &qs0, &qqs0,
        &ks0, &ks1, &ks2,
        &vs0, &vs1, &vs2,
        &p_cu, &num_seqs,
        &p_centroids_k, &p_centroids_v,  // MAD-214: turbo-FP8 per-layer LUT pointers
        &p_global_scratch, &p_profile_scratch,
    };

    // Grid for 3D — mirrors AITER's host dispatcher:
    //   gX = num_q_tokens / BLOCK_Q + num_seqs  (q-block index space)
    //   gY = num_kv_heads
    //   gZ = NUM_SEGMENTS_PER_SEQ
    // For pure decode (q_len=1 each), num_q_tokens == num_seqs so this
    // collapses to num_seqs/BLOCK_Q + num_seqs (the POC formula).
    unsigned int g3_x = (unsigned int)(num_q_tokens / MT_AITER_UATTN_BLOCK_Q + num_seqs);
    unsigned int g3_y = (unsigned int) a->shape.num_kv_heads;
    unsigned int g3_z = (unsigned int) MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ;

    hipError_t err = c.h_3d->launch(stream, g3_x, g3_y, g3_z, args_3d);
    if (err != hipSuccess) return err;

    // ── reduce_segments phase ──────────────────────────────────────────────
    hipDeviceptr_t p_out = (hipDeviceptr_t) a->out;
    hipDeviceptr_t p_os  = (hipDeviceptr_t) a->out_scale;
    int64_t        os0   = a->output_stride_0;
    int64_t        rbts  = a->block_table_stride;

    void *args_reduce[] = {
        &p_out,
        &p_segm_out, &p_segm_max, &p_segm_expsum,
        &p_sl, &num_seqs,
        &p_os, &os0, &rbts,
        &p_cu,
        &p_global_scratch, &p_profile_scratch,
    };

    unsigned int gr_x = (unsigned int) num_q_tokens;
    unsigned int gr_y = (unsigned int) a->shape.num_q_heads;
    unsigned int gr_z = 1;

    return c.h_reduce->launch(stream, gr_x, gr_y, gr_z, args_reduce);
}

size_t mt_aiter_uattn_segm_output_bytes(const mt_aiter_uattn_shape_t * shape, int num_q_tokens) {
    return (size_t)num_q_tokens
         * (size_t)shape->num_q_heads
         * (size_t)MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ
         * (size_t)shape->head_size
         * sizeof(float);
}

size_t mt_aiter_uattn_segm_max_bytes(const mt_aiter_uattn_shape_t * shape, int num_q_tokens) {
    return (size_t)num_q_tokens
         * (size_t)shape->num_q_heads
         * (size_t)MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ
         * sizeof(float);
}

size_t mt_aiter_uattn_segm_expsum_bytes(const mt_aiter_uattn_shape_t * shape, int num_q_tokens) {
    return mt_aiter_uattn_segm_max_bytes(shape, num_q_tokens);
}
