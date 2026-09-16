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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

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

// FP8 WMMA (v_wmma_f32_16x16x16_fp8_fp8) exists on gfx1200/gfx1201/gfx1250.
// gfx1030 (RDNA2) has neither FP8 WMMA nor v_dot4_i32_i8. The kernel falls
// back to dequant-to-f16 + packed f16 tl.dot on those targets.
bool target_has_fp8_wmma(const std::string & target) {
    return target.find("gfx120") != std::string::npos
        || target.find("gfx125") != std::string::npos;
}

// Build the 3D-kernel Triton signature for the given model shape.
// Substitutions vs. the AITER signature template:
//   pos 17 → num_q_heads
//   pos 18 → num_queries_per_kv (= num_q_heads / num_kv_heads)
//   pos 21 → head_size (query_stride_1 constexpr)
//   pos 23 → block_size (BLOCK_SIZE constexpr)
//   pos 25 → head_size (HEAD_SIZE constexpr)
//   pos 26 → head_size (HEAD_SIZE_PADDED constexpr; assumes head_size is pow2)
// MAD-2026-09-11 vectorized loads: the AOT signature must carry Triton's
// ":16" divisibility specialization on the stride arguments, exactly as the
// JIT would infer it, or the compiler cannot prove 16-byte alignment of the
// K/V/Q/out tile addresses and emits one 2-byte load per element
// (global_load_ushort / d16_b16). Measured on the production 2D spec:
// gfx1030 f16 96 x ushort -> 12 x dwordx4, gfx1201 f16 256 x d16 -> 32 x b128.
// q/out stride_0 = num_q_heads*head_size and every f16 K/V cache stride is a
// multiple of head_size, so head_size % 16 == 0 makes them all 16-divisible;
// the launch path asserts the runtime values. Turbo cache types leave the
// (unused) K/V stride args unhinted.
static inline bool mt_aiter_q_stride_div16(const mt_aiter_uattn_shape_t & s) {
    return s.head_size % 16 == 0 && ((int64_t) s.num_q_heads * s.head_size) % 16 == 0;
}
static inline bool mt_aiter_kv_stride_div16(const mt_aiter_uattn_shape_t & s) {
    return s.cache_type == MT_AITER_CACHE_F16 && s.head_size % 16 == 0;
}
static inline const char * mt_aiter_i64_sig(bool div16) { return div16 ? "i64:16" : "i64"; }

std::string build_signature_3d(const mt_aiter_uattn_shape_t & s, int use_fp8_wmma,
                                int use_fp8_loader_v2 = 0, int all_decode = 1) {
    // MAD-199: K/V cache pointer dtype depends on cache_type. F16 stays
    // `*fp16:16` (upstream signature); turbo3/turbo4 switch to `*i8:16` byte
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
            kv_ptr_dtype   = "*i8:16";
            cache_type_val = 1;
            break;
        case MT_AITER_CACHE_TURBO4:
            kv_ptr_dtype   = "*i8:16";
            cache_type_val = 2;
            break;
        // MAD-214: turbo-FP8 family. Production-wired variants only (BS=256).
        // BS<256 variants throw at compile time via tl.static_assert in
        // unified_attention.py — MAD-215 wires those.
        case MT_AITER_CACHE_TURBO3_FP8_BS256:
            kv_ptr_dtype   = "*i8:16";
            cache_type_val = 14;
            break;
        case MT_AITER_CACHE_TURBO4_FP8_BS256:
            kv_ptr_dtype   = "*i8:16";
            cache_type_val = 24;
            break;
        case MT_AITER_CACHE_TURBO5_FP8_BS256:
            kv_ptr_dtype   = "*i8:16";
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
        "i64, %s, %d, i64, "               // block_table_stride, q_stride_0 (":16" when divisible), query_stride_1=head_size, qq_bias_stride_0
        "%d, %d, %d, %d, "                  // BLOCK_SIZE, TILE_SIZE, HEAD_SIZE, HEAD_SIZE_PADDED
        "0, 0, 0, 0, 0, "                   // USE_ALIBI / QQ / SOFTCAP / SINKS / SLIDING_WINDOW
        "%s, %s, %s, 1, %s, %s, %s, 1, "  // k/v cache strides (":16" for f16; last is constexpr=1; for turbo these args are present but unused — helper computes byte strides internally)
        "*i32, %d, i32, %d, %d, %d, %d, %d, %d, " // query_start_len, BLOCK_Q, num_seqs, BLOCK_M, NUM_SEGMENTS, ALL_DECODE, CACHE_TYPE, USE_FP8_WMMA, USE_FP8_LOADER_V2
        "*i8:16, *i8:16",                     // MAD-214: centroids_k_ptr, centroids_v_ptr (None-safe for non-FP8)
        kv_ptr_dtype,                           // K cache pointer dtype
        kv_ptr_dtype,                           // V cache pointer dtype
        s.num_q_heads,                          // num_q_heads constexpr
        s.num_q_heads / s.num_kv_heads,         // num_queries_per_kv constexpr
        mt_aiter_i64_sig(mt_aiter_q_stride_div16(s)),  // q_stride_0 divisibility
        s.head_size,                            // query_stride_1 = head_size constexpr
        s.block_size,                           // BLOCK_SIZE constexpr
        MT_AITER_UATTN_TILE_SIZE,               // TILE_SIZE constexpr
        s.head_size,                            // HEAD_SIZE constexpr
        s.head_size,                            // HEAD_SIZE_PADDED (assumes head_size is pow2)
        mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)),  // k cache strides 0..2
        mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)),  // v cache strides 0..2
        MT_AITER_UATTN_BLOCK_Q,                 // BLOCK_Q constexpr
        MT_AITER_UATTN_BLOCK_M,                 // BLOCK_M constexpr
        MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ,    // NUM_SEGMENTS_PER_SEQ constexpr
        all_decode,                             // ALL_DECODE constexpr (MAD-2026-09-12: was hardcoded 1)
        cache_type_val,                         // CACHE_TYPE constexpr (MAD-199)
        use_fp8_wmma,                           // USE_FP8_WMMA constexpr (0 on gfx1030)
        use_fp8_loader_v2);                     // USE_FP8_LOADER_V2 constexpr (MAD-2026-09-11 fp8-loader-v2, opt-in)
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
// drives *fp16:16 vs *i8:16 and the CACHE_TYPE constexpr).
std::string build_signature_2d(const mt_aiter_uattn_shape_t & s,
                                int block_m, int block_q, int use_fp8_wmma,
                                int tile_size = MT_AITER_UATTN_TILE_SIZE,
                                int use_fp8_loader_v2 = 0) {
    const char * kv_ptr_dtype;
    int          cache_type_val;
    switch (s.cache_type) {
        case MT_AITER_CACHE_F16:               kv_ptr_dtype = "*fp16:16"; cache_type_val =  0; break;
        case MT_AITER_CACHE_TURBO3:            kv_ptr_dtype = "*i8:16";   cache_type_val =  1; break;
        case MT_AITER_CACHE_TURBO4:            kv_ptr_dtype = "*i8:16";   cache_type_val =  2; break;
        // MAD-214: turbo-FP8 family (BS=256 production variants only;
        // BS<256 covered by MAD-215). Numeric values match mt_aiter_cache_type.
        case MT_AITER_CACHE_TURBO3_FP8_BS256:  kv_ptr_dtype = "*i8:16";   cache_type_val = 14; break;
        case MT_AITER_CACHE_TURBO4_FP8_BS256:  kv_ptr_dtype = "*i8:16";   cache_type_val = 24; break;
        case MT_AITER_CACHE_TURBO5_FP8_BS256:  kv_ptr_dtype = "*i8:16";   cache_type_val = 34; break;
        default:                               kv_ptr_dtype = "*fp16:16"; cache_type_val =  0; break;
    }
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
        "*fp16:16, *fp16:16, %s, %s, *fp32, *i32, *i32, *fp32, *fp16, "  // out, q, k, v, sink, bt, sl, alibi, qq_bias
        "fp32, *fp32, *fp32, *fp32, *fp32, fp32, "                       // scale, q/k/v_descale, out_scale, softcap
        "%d, %d, "                                                       // num_q_heads, num_queries_per_kv
        "i64, %s, %d, %s, %d, i64, "                                     // bt_stride, q_stride_0, q_stride_1=head_size, out_stride_0, out_stride_1=head_size, qq_bias_stride_0 (":16" on q/out stride_0 when divisible)
        "%d, %d, %d, %d, "                                               // BLOCK_SIZE, TILE_SIZE, HEAD_SIZE, HEAD_SIZE_PADDED
        "0, 0, 0, 0, 0, "                                                // USE_ALIBI / QQ / SOFTCAP / SINKS / SLIDING_WINDOW
        "%s, %s, %s, 1, %s, %s, %s, 1, "                                 // k/v cache strides (":16" for f16; last is constexpr=1)
        "*i32, %d, i32, %d, "                                            // query_start_len, BLOCK_Q, num_seqs(runtime), BLOCK_M
        "-448.0, 448.0, 0, %d, %d, %d, "                                 // FP8_MIN, FP8_MAX, ALL_DECODE=0, CACHE_TYPE, USE_FP8_WMMA, USE_FP8_LOADER_V2
        "*i8:16, *i8:16",                                                  // MAD-214: centroids_k_ptr, centroids_v_ptr
        kv_ptr_dtype, kv_ptr_dtype,
        s.num_q_heads, s.num_q_heads / s.num_kv_heads,
        mt_aiter_i64_sig(mt_aiter_q_stride_div16(s)), s.head_size,   // q_stride_0 divisibility, query_stride_1
        mt_aiter_i64_sig(mt_aiter_q_stride_div16(s)), s.head_size,   // out_stride_0 divisibility, output_stride_1
        s.block_size,                       // BLOCK_SIZE
        tile_size,                          // TILE_SIZE (per-call override; MAD-2026-09-11 gfx1030 fix)
        s.head_size, s.head_size,           // HEAD_SIZE, HEAD_SIZE_PADDED
        mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)),  // k cache strides 0..2
        mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)), mt_aiter_i64_sig(mt_aiter_kv_stride_div16(s)),  // v cache strides 0..2
        block_q,                            // BLOCK_Q
        block_m,                            // BLOCK_M
        cache_type_val,                     // CACHE_TYPE
        use_fp8_wmma,                       // USE_FP8_WMMA (0 on gfx1030)
        use_fp8_loader_v2);                 // USE_FP8_LOADER_V2 (MAD-2026-09-11 fp8-loader-v2, opt-in)
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

// MAD-2026-09-11 fp8-predequant: signature for dequant_turbo4_fp8_bs256_to_f16_2d.
// Fixed to the turbo4_fp8 BS=256 (IDX_BITS=4) family — only cache type this
// pre-pass is wired for (see MT_AITER_CACHE_TURBO4_FP8_BS256 gate below).
std::string build_signature_dequant(const mt_aiter_uattn_shape_t & s) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
        "*i8:16, *i8:16, *fp16:16, *fp16:16, "  // k_fp8, v_fp8, k_f16 (scratch), v_f16 (scratch)
        // MAD-2026-09-12 predequant-scratch: scratch_block_tables inserted
        // right after block_tables — see predequant-scratch-0912.txt.
        "*i32, *i32, *i32, *i8:16, *i8:16, "    // block_tables, scratch_block_tables, seq_lens, centroids_k, centroids_v
        "i64, "                                  // block_table_stride
        "%d, %d, %d, %d",                        // n_kv_heads, BLOCK_SIZE, HEAD_SIZE, BYTES_PER_BLOCK (constexpr)
        s.num_kv_heads, s.block_size, s.head_size,
        162 /* BYTES_PER_BLOCK for turbo4_fp8 BS=256, IDX_BITS=4 */);
    return buf;
}

struct CachedHandles {
    mt_aiter_uattn_shape_t      shape         = {};
    const aiter::KernelHandle * h_3d          = nullptr;  // ALL_DECODE=1 (one q-token per seq)
    // MAD-2026-09-12 dispatch-fix: second 3D handle, ALL_DECODE=0, for
    // multi-query-token batches (MTP verify / DFlash draft-check) now routed
    // to the 3D split-K kernel by the occupancy-driven dispatch predicate
    // instead of unconditionally to the 2D base kernel. See
    // mt_aiter_uattn_should_use_2d() and the dispatch in
    // mt_aiter_unified_attn() below.
    const aiter::KernelHandle * h_3d_md       = nullptr;  // ALL_DECODE=0 (find_seq_idx addressing)
    const aiter::KernelHandle * h_reduce      = nullptr;
    const aiter::KernelHandle * h_2d          = nullptr;  // base prefill (BLOCK_M=16, BLOCK_Q=2)
    const aiter::KernelHandle * h_2d_large    = nullptr;  // large prefill (BLOCK_M=8*GQA, BLOCK_Q=8)
    int                         block_q_large = MT_AITER_UATTN_BLOCK_Q_LARGE;
    bool                        initialized   = false;
    hipError_t                  init_err      = hipSuccess;

    // MAD-2026-09-11 fp8-predequant (gfx1030 2D-large-prefill only). See
    // build_signature_dequant() / ensure_initialized() / the dispatch in
    // mt_aiter_unified_attn() below for the full path. `predequant_enabled`
    // is decided once at init time (arch + cache_type + env override);
    // whether a given call actually TAKES the path additionally depends on
    // use_2d_large and args->num_scratch_blocks > 0 (MAD-2026-09-12
    // predequant-scratch), both only known per-call.
    bool                         predequant_enabled = false;
    const aiter::KernelHandle  * h_dequant          = nullptr;  // dequant_turbo4_fp8_bs256_to_f16_2d
    const aiter::KernelHandle  * h_2d_large_f16     = nullptr;  // F16 shadow of h_2d_large, same cache key
                                                                  // a real F16 call of this shape would use
    int                          block_q_large_f16  = MT_AITER_UATTN_BLOCK_Q_LARGE;
    // MAD-2026-09-12 garbage-16k fix: lazily grown f16 scratch K/V paged
    // caches for the predequant path. CachedHandles itself is keyed ONLY by
    // physical device ordinal (get_cached() below) — it is shared by EVERY
    // ggml_backend_cuda_context that ever runs on this device, which in
    // practice means both the target llama_context AND any independent
    // draft llama_context (common/speculative.cpp's ctx_dft, e.g. the MTP
    // "nextn" head, which calls llama_encode()/llama_decode() on its OWN
    // llama_context and therefore its OWN hipStream_t — see
    // ggml_backend_cuda_init()/new_pool_for_device(), each call gets a fresh
    // ggml_backend_cuda_context with its own streams[] array). A single
    // shared scratch_k/scratch_v buffer written and read by two independent
    // streams with no cross-stream ordering is a genuine data race: ctx_dft's
    // own turbo4_fp8 2D-large prefill (it uses the same paged cache type,
    // just a smaller SWA ring) can dequant-write into the SAME physical
    // scratch buffer ctx_tgt's prefill is concurrently dequant-writing/
    // reading on its own stream, with nothing but scratch_mu (which only
    // guards the malloc/free bookkeeping below, not kernel execution)
    // between them. Bigger prompts -> longer-running dequant/attention
    // kernels -> a wider overlap window -> more likely to lose the race,
    // which matches the observed size-dependence (garbage above ~32 blocks,
    // clean below). Fix: key the scratch buffers by the ISSUING STREAM, not
    // just the device, so two contexts on the same device never share the
    // same physical scratch memory. Kernel handles (h_2d/h_3d/.../h_dequant)
    // are stateless compiled code and stay safely shared across streams —
    // only this mutable buffer needed isolating.
    struct ScratchBuf {
        void   *k      = nullptr;
        void   *v      = nullptr;
        size_t  blocks = 0;
        bool    logged = false;
    };
    std::mutex                                  scratch_mu;
    std::unordered_map<hipStream_t, ScratchBuf> scratch_by_stream;
};

// Per-DEVICE handle cache.
//
// A KernelHandle owns a HIP module loaded into one device's context, and the
// Triton target is derived from the active device's gcnArchName. A single
// process-wide cache therefore pins every later call to whichever device
// happened to run attention first: under tensor parallelism the second card
// reused the first card's modules and the launch failed with
// "invalid device ordinal" (and the second arch was never even compiled for).
//
// Key by device ordinal. References into the map stay valid across rehash and
// entries are only ever inserted, so handing out a reference under the lock and
// using it after is safe.
CachedHandles & get_cached() {
    static std::mutex mu_map;
    static std::unordered_map<int, CachedHandles> per_device;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        dev = 0;
    }
    std::lock_guard<std::mutex> g(mu_map);
    return per_device[dev];
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
                "now %d/%d/%d/%d). The AITER cache supports one shape per device.\n",
                c.shape.head_size, c.shape.num_q_heads, c.shape.num_kv_heads, c.shape.block_size,
                shape.head_size,   shape.num_q_heads,   shape.num_kv_heads,   shape.block_size);
            return hipErrorInvalidValue;
        }
        return c.init_err;
    }

    const std::string target  = detect_hip_target();
    const int use_fp8_wmma    = target_has_fp8_wmma(target) ? 1 : 0;
    // fp8-loader-v2 (MAD-2026-09-11): opt-in compact-vectorized-load variant
    // of the turbo4_fp8 (IDX_BITS=4, CACHE_TYPE=24) K/V loaders — see
    // kernels/unified_attention.py's USE_FP8_LOADER_V2 comment for the full
    // rationale (targets the per-element address/instruction bloat that
    // num_warps=8 alone could not remove, per fp8-kernel-why-0911.txt §3/§4).
    // Plumbed exactly like MT_AITER_GFX1201_NUM_WARPS8 below: an opt-in env
    // var, off by default, baked into the signature string (and therefore
    // the AOT/runtime-compile cache key) the same way USE_FP8_WMMA is, so
    // the v1 and v2 loader variants compile to and live in distinct cache
    // entries and can be A/B'd without a rebuild. The kernel itself only
    // acts on this flag when IDX_BITS==4 (CACHE_TYPE==24); for other
    // turbo-FP8 cache types the flag is still baked into the signature (its
    // own cache slot) but the kernel body silently falls back to the v1
    // loader for them.
    // Default ON since 2026-09-11: measured bit-identical to the v1 loader
    // (tests/test_aiter_turbo_fp8_smoke on gfx1201 and gfx1030) and +33%
    // prefill at 16k on the 27B TP shape (634 -> 846 tok/s). Set
    // MT_AITER_FP8_LOADER_V2=0 to fall back to the v1 loader.
    int use_fp8_loader_v2 = 1;
    if (const char * s = std::getenv("MT_AITER_FP8_LOADER_V2")) { use_fp8_loader_v2 = std::atoi(s) != 0 ? 1 : 0; }
    const std::string sig_3d  = build_signature_3d(shape, use_fp8_wmma, use_fp8_loader_v2);
    const std::string sig_red = build_signature_reduce(shape);

    aiter::Registry & reg = aiter::Registry::instance();
    reg.set_compile_script(AITER_COMPILE_SCRIPT_DEFAULT);

    // MAD-232 perf-sweep env overrides for num_warps / num_stages. Defaults
    // are 4/1 (matching RDNA generic Triton heuristic; gfx1201 has no
    // arch-specific defaults per third_party/amd/backend/compiler.py). The
    // FP8 path may benefit from different values; sweepable via env without
    // recompile. Each (nw, ns) pair gets its own JIT cache slot.
    //
    // MAD-2026-09-11 gfx1030 prefill tiling fix: on gfx1030 (RDNA2, no WMMA,
    // USE_FP8_WMMA=0 -> the tl.dot fallback lowers to a manual per-lane FMA
    // accumulation instead of a matrix-core reduction) the production 2D
    // large-prefill spec (BLOCK_Q=10, BLOCK_M=64, head_size=256) was measured
    // offline (Triton AOT compile + AMDGPU code-object metadata, no GPU
    // touched) to already spill heavily at num_warps=4:
    //   vgpr_count=256 (HW per-wave cap, hit regardless of tile), vgpr_spill
    //   _count=1836, sgpr_spill_count=28, private_segment_fixed_size=4980B/
    //   lane. The fp8 2D-large spec spills even harder (6800B/lane).
    // Raising num_warps to 8 for gfx1030 ONLY spreads the same BLOCK_M x
    // HEAD_SIZE_PADDED fp32 accumulator tile over twice as many physical
    // lanes without changing BLOCK_M/BLOCK_Q/the KV-tile loop or the
    // per-output-row accumulation order, and cut spilling sharply in offline
    // recompiles of the exact production signature:
    //   BLOCK_Q=10/BLOCK_M=64 nw4->nw8: vgpr_spill 1836->755 (-59%),
    //     scratch/lane 4980B->2876B (-42%), sgpr_spill 28->25.
    //   BLOCK_Q=5/BLOCK_M=32  nw4->nw8: vgpr_spill  815->246 (-70%),
    //     scratch/lane 3056B->984B (-68%).
    // vgpr_count stays pinned at the 256 HW cap in every configuration
    // tested (gfx1030's per-wave VGPR file), so waves/SIMD occupancy is not
    // reduced by this change; LDS usage (32KB/workgroup) is unaffected
    // (num_warps only changes threads-per-workgroup, not the tile shape or
    // shared-memory footprint). The same num_warps=8 recompile against
    // gfx1201 (WMMA path, USE_FP8_WMMA=1) also strictly improved (spill
    // 132->0 VGPRs at the identical BLOCK_Q=10/BLOCK_M=64 signature) but is
    // deliberately NOT changed here — this fix is scoped to gfx1030 only, so
    // gfx1201's kernel and launch config are byte-for-byte unchanged unless
    // a future patch chooses to also raise its default.
    // MT_AITER_NUM_WARPS/_NUM_STAGES env overrides still take precedence, so
    // the old default remains reachable for A/B sweeps without a rebuild.
    const bool is_gfx1030 = target.find("gfx1030") != std::string::npos;
    int env_nw = is_gfx1030 ? 8 : 4;
    int env_ns = 1;
    if (const char * s = std::getenv("MT_AITER_NUM_WARPS"))  { int v = std::atoi(s); if (v > 0 && v <= 32) env_nw = v; }
    // Card-scoped variant so a TP process can raise only the gfx1030 rank
    // (the 2026-09-11 zero-spill point is gfx1030 TILE=8 + 16 warps).
    if (is_gfx1030) {
        if (const char * s = std::getenv("MT_AITER_GFX1030_NUM_WARPS")) { int v = std::atoi(s); if (v > 0 && v <= 32) env_nw = v; }
    }
    if (const char * s = std::getenv("MT_AITER_NUM_STAGES")) { int v = std::atoi(s); if (v > 0 && v <= 8 ) env_ns = v; }

    // MAD-2026-09-11 gfx1201 num_warps=8 option — OPT-IN, NOT the default.
    // Separate, independently gated hunk (env var, not a target-string
    // default flip like gfx1030's above) purely so the orchestrator can A/B
    // it without touching gfx1030's behavior or gfx1201's default. Offline
    // recompile of the identical BQ=10/BM=64/TILE=32 gfx1201 WMMA-path spec
    // (USE_FP8_WMMA=1) at num_warps=8 eliminated its small residual spill
    // entirely: vgpr_spill_count 132->0, private_segment_fixed_size
    // 532B->0B/lane, vgpr_count 256->219 (i.e. it no longer even needs the
    // per-wave cap). Strictly better statically at this one tile, but left
    // off by default because (a) it wasn't asked for by the original task
    // scope (gfx1030 only) and (b) the WMMA-path occupancy/operand-staging
    // interaction at 8 warps was not otherwise audited here — only this one
    // (BQ=10,BM=64,TILE=32) signature was measured.
    const bool is_gfx1201 = target.find("gfx1201") != std::string::npos;
    if (is_gfx1201 && std::getenv("MT_AITER_GFX1201_NUM_WARPS8") != nullptr) {
        env_nw = 8;
    }

    // MAD-214 Phase 1F-D: 3d + reduce kernels now handle FP8 too (IS_TURBO_FP8
    // branches mirrored from the 2d kernel). Compile unconditionally.
    aiter::KernelSpec spec_3d {
        AITER_KERNEL_SOURCE_DEFAULT,
        "kernel_unified_attention_3d",
        target, sig_3d, env_nw, env_ns,
    };
    c.h_3d = reg.get_or_compile(spec_3d);

    // MAD-2026-09-12 dispatch-fix: ALL_DECODE=0 sibling of h_3d. Same shape/
    // target/num_warps/num_stages/cache-type/fp8-loader-v2 selection, only
    // the ALL_DECODE literal differs (0 instead of 1) — kernel_unified_
    // attention_3d already has fully general find_seq_idx/q_block_local_idx
    // addressing for this case (kernels/unified_attention.py:1664-1679), it
    // was just never instantiated with ALL_DECODE=0 before this patch.
    const std::string sig_3d_md = build_signature_3d(shape, use_fp8_wmma, use_fp8_loader_v2, /*all_decode=*/0);
    aiter::KernelSpec spec_3d_md {
        AITER_KERNEL_SOURCE_DEFAULT,
        "kernel_unified_attention_3d",
        target, sig_3d_md, env_nw, env_ns,
    };
    c.h_3d_md = reg.get_or_compile(spec_3d_md);

    aiter::KernelSpec spec_reduce {
        AITER_KERNEL_SOURCE_DEFAULT,
        "reduce_segments",
        target, sig_red, env_nw, env_ns,
    };
    c.h_reduce = reg.get_or_compile(spec_reduce);

    // MAD-199 D3: 2D base prefill spec (BLOCK_M=16, BLOCK_Q=2).
    const std::string sig_2d = build_signature_2d(
        shape, MT_AITER_UATTN_BLOCK_M, MT_AITER_UATTN_BLOCK_Q, use_fp8_wmma,
        MT_AITER_UATTN_TILE_SIZE, use_fp8_loader_v2);
    aiter::KernelSpec spec_2d {
        AITER_KERNEL_SOURCE_DEFAULT,
        "kernel_unified_attention_2d",
        target, sig_2d, env_nw, env_ns,
    };
    c.h_2d = reg.get_or_compile(spec_2d);

    // MAD-203: 2D large-prefill. BLOCK_M = next_pow2(BLOCK_Q_LARGE*GQA) and
    // BLOCK_Q = BLOCK_M/GQA — 64/8 for GQA=8, 64/10 for GQA=6. BLOCK_M must be
    // a multiple of 16 for WMMA AND a power of two, because the kernel does
    // tl.arange(0, BLOCK_M) and Triton rejects a non-power-of-two extent at
    // compile time. 8*GQA=48 satisfied the %16 check but not that one, which is
    // what produced the rc=256 compile failure on gfx1201.
    int block_m_large = mt_aiter_uattn_block_m_large(shape.num_q_heads, shape.num_kv_heads);
    int block_q_large = mt_aiter_uattn_block_q_large(shape.num_q_heads, shape.num_kv_heads);
    // gfx1030 A/B knob (2026-09-11): MT_AITER_GFX1030_BLOCK_M=<pow2 >= 16>
    // overrides the large-prefill BLOCK_M (BLOCK_Q follows as BLOCK_M/GQA);
    // the offline sweep found BM32/BQ5 spill-free at TILE 8 with 8 warps.
    if (is_gfx1030) {
        if (const char * t = std::getenv("MT_AITER_GFX1030_BLOCK_M")) {
            const int v = std::atoi(t);
            const int gqa = shape.num_q_heads / shape.num_kv_heads;
            if (v >= 16 && (v & (v - 1)) == 0 && gqa > 0 && v / gqa >= 1) {
                block_m_large = v;
                block_q_large = v / gqa;
            }
        }
    }
    const bool block_m_pow2 = block_m_large > 0
        && (block_m_large & (block_m_large - 1)) == 0;
    const bool large_ok = block_m_large >= 16 && (block_m_large % 16 == 0) && block_m_pow2
        && block_q_large >= 1
        && (block_m_large != MT_AITER_UATTN_BLOCK_M
            || block_q_large != MT_AITER_UATTN_BLOCK_Q);
    // MAD-2026-09-11 gfx1030 zero/near-zero-spill tiling fix: the 2D
    // large-prefill spec (BLOCK_Q=10, BLOCK_M=64, HEAD_SIZE=256) is the
    // kernel that carries prefill under TP with the 6900XT holding a
    // fraction of the KV heads. Offline AOT recompiles of this EXACT
    // production signature (Triton 3.8.0, target=hip:gfx1030:32, code-
    // object amdhsa.kernels metadata read directly, no GPU touched) swept
    // TILE_SIZE (the KV-tile-length constexpr, MT_AITER_UATTN_TILE_SIZE=32
    // globally today) at num_warps in {4,8}:
    //   BQ=10 BM=64 TILE=32 nw=4 (today's prod default): vgpr_spill=1836,
    //     sgpr_spill=28,  scratch/lane=4980B  <- baseline
    //   BQ=10 BM=64 TILE=32 nw=8 (this patch's v1 hunk, above):
    //     vgpr_spill=755,  sgpr_spill=25, scratch/lane=2876B (-59%/-42%)
    //   BQ=10 BM=64 TILE=16 nw=8 (THIS HUNK, gfx1030 default below):
    //     vgpr_spill=245,  sgpr_spill=12, scratch/lane= 984B (-87%/-80%
    //     vs. true baseline) — BLOCK_M/BLOCK_Q UNCHANGED from MAD-203, so
    //     the full GQA=6 Q-row tile-reuse rationale is preserved; only the
    //     KV-tile trip count doubles (32 tokens/iter -> 16 tokens/iter).
    //   BQ=10 BM=64 TILE=4  nw=8: vgpr_spill=0, scratch=0 (TRUE ZERO) but
    //     TILE=4 < BLOCK_SIZE=16 means 8x the loop trip count of
    //     production and repeated redundant physical_block_idx/page
    //     lookups per paged block — not applied, flagged as the "spill-
    //     free but likely to hurt" option (see report.txt table).
    //   BQ=5  BM=32 TILE=8  nw=8: vgpr_spill=0, scratch=0 (TRUE ZERO,
    //     f16) / vgpr_spill=1, scratch=8B (fp8 — the turbo-FP8 LUT dequant
    //     pushes it 1 VGPR past zero at the identical tile, see report.txt
    //     "fp8 dequant fallback" section) — narrows BLOCK_M in half
    //     (less GQA reuse per K/V load) AND halves TILE_SIZE again vs the
    //     TILE=4 option's already-small tile. Also not applied by default;
    //     documented as the best true-zero-spill fallback if TILE=16/nw=8
    //     still isn't enough once measured on real hardware.
    // Chose TILE=16/nw=8 as the default: smallest single-parameter change
    // (TILE_SIZE only, BLOCK_M/BLOCK_Q untouched) that gets spill within
    // "a few hundred bytes of scratch," not the most aggressive option.
    // MT_AITER_UATTN_TILE_SIZE (the 3d/reduce/base-2d spec's global
    // default) is intentionally NOT changed — only the 2d_large spec's
    // TILE_SIZE is overridden, and only on gfx1030.
    // gfx1030 KV tile for the large-prefill spec: 16 by default (tolerance-level,
    // eye-tested 2026-09-11). MT_AITER_GFX1030_TILE=8 selects the zero-spill
    // point found by the 2026-09-11 offline sweep (BM64/BQ10/TILE8 with
    // MT_AITER_NUM_WARPS=16: 0 VGPR spills, 0 scratch, both cache types);
    // any power of two in [8, 64] is accepted, baked into the AOT signature.
    int tile_size_large = is_gfx1030 ? 16 : MT_AITER_UATTN_TILE_SIZE;
    if (is_gfx1030) {
        if (const char * t = std::getenv("MT_AITER_GFX1030_TILE")) {
            const int v = std::atoi(t);
            if (v >= 8 && v <= 64 && (v & (v - 1)) == 0) { tile_size_large = v; }
        }
    }
    bool large_compiled = false;
    if (large_ok) {
        const std::string sig_2d_large = build_signature_2d(
            shape, block_m_large, block_q_large, use_fp8_wmma, tile_size_large,
            use_fp8_loader_v2);
        aiter::KernelSpec spec_2d_large {
            AITER_KERNEL_SOURCE_DEFAULT,
            "kernel_unified_attention_2d",
            target, sig_2d_large, env_nw, env_ns,
        };
        c.h_2d_large   = reg.get_or_compile(spec_2d_large);
        large_compiled = (c.h_2d_large != nullptr);
    }
    if (large_compiled) {
        c.block_q_large = block_q_large;
    } else {
        // The large tile is an optimisation, never a correctness requirement:
        // the base 16/2 spec handles every prefill shape. A failed or skipped
        // large-tile compile must degrade to it, not abort the server — losing
        // prefill throughput beats refusing to serve.
        if (large_ok) {
            std::fprintf(stderr,
                "mt_aiter_unified_attn: large-prefill spec (BLOCK_M=%d BLOCK_Q=%d) failed to "
                "compile on %s — falling back to the base %d/%d tile. Prefill will be slower.\n",
                block_m_large, block_q_large, target.c_str(),
                MT_AITER_UATTN_BLOCK_M, MT_AITER_UATTN_BLOCK_Q);
        }
        c.h_2d_large    = c.h_2d;
        c.block_q_large = MT_AITER_UATTN_BLOCK_Q;
    }

    // MAD-2026-09-11 fp8-predequant. Default ON when: gfx1030 (no FP8 WMMA —
    // gfx1201/gfx125x already run the fp8 kernel via WMMA and see no benefit
    // here), cache_type is turbo4_fp8 BS=256 (the only family this pre-pass
    // is wired for), and the large-prefill spec actually compiled (a
    // fallback to the base 16/2 tile means the geometry this pre-pass was
    // sized for isn't what's being launched this call — skip it, the fp8
    // kernel's existing in-kernel dequant still runs correctly on the base
    // tile). MT_AITER_FP8_PREDEQUANT=0 forces off (any arch); =1 forces on
    // (any arch, e.g. to A/B the R9700 once it's usable) provided the
    // cache_type/large-tile preconditions still hold — the env only overrides
    // the is_gfx1030 arch check, not the cache_type or large_compiled gates,
    // since those aren't just perf knobs: the kernel below hard-requires
    // turbo4_fp8 BS=256 (tl.static_assert HEAD_SIZE==256 in the loader) and a
    // real large-tile handle to shadow.
    int predequant_env = -1;  // -1 = unset
    if (const char * s = std::getenv("MT_AITER_FP8_PREDEQUANT")) { predequant_env = std::atoi(s) != 0 ? 1 : 0; }
    const bool predequant_arch_ok = (predequant_env == 1) || (predequant_env == -1 && is_gfx1030);
    const bool predequant_wanted  = predequant_env != 0 && predequant_arch_ok
        && shape.cache_type == MT_AITER_CACHE_TURBO4_FP8_BS256
        && large_compiled;

    if (predequant_wanted) {
        const std::string sig_dequant = build_signature_dequant(shape);
        aiter::KernelSpec spec_dequant {
            AITER_KERNEL_SOURCE_DEFAULT,
            "dequant_turbo4_fp8_bs256_to_f16_2d",
            target, sig_dequant, env_nw, env_ns,
        };
        c.h_dequant = reg.get_or_compile(spec_dequant);

        // F16 shadow of the large-prefill spec: same shape/tile/target/
        // num_warps/num_stages/use_fp8_wmma/use_fp8_loader_v2 as the
        // production fp8 large-prefill spec above, but cache_type = F16 —
        // this MUST produce the identical signature (and therefore the
        // identical registry cache key) that a real F16 call of this same
        // geometry would build, so the operator can verify it lands on an
        // existing entry rather than silently compiling a near-duplicate.
        mt_aiter_uattn_shape_t shadow_shape = shape;
        shadow_shape.cache_type = MT_AITER_CACHE_F16;
        const std::string sig_2d_large_f16 = build_signature_2d(
            shadow_shape, block_m_large, block_q_large, use_fp8_wmma,
            tile_size_large, use_fp8_loader_v2);
        aiter::KernelSpec spec_2d_large_f16 {
            AITER_KERNEL_SOURCE_DEFAULT,
            "kernel_unified_attention_2d",
            target, sig_2d_large_f16, env_nw, env_ns,
        };
        c.h_2d_large_f16    = reg.get_or_compile(spec_2d_large_f16);
        c.block_q_large_f16 = block_q_large;

        c.predequant_enabled = (c.h_dequant != nullptr) && (c.h_2d_large_f16 != nullptr);
        if (!c.predequant_enabled) {
            std::fprintf(stderr,
                "mt_aiter_unified_attn: fp8-predequant kernels failed to compile on %s "
                "(dequant=%p, f16_shadow=%p) — falling back to the in-kernel fp8 dequant path.\n",
                target.c_str(), (const void*)c.h_dequant, (const void*)c.h_2d_large_f16);
        }
    }

    std::fprintf(stderr,
        "mt_aiter_unified_attn: target=%s USE_FP8_WMMA=%d USE_FP8_LOADER_V2=%d GQA=%d "
        "2d_large BLOCK_M=%d BLOCK_Q=%d (base 16/2)\n",
        target.c_str(), use_fp8_wmma, use_fp8_loader_v2,
        mt_aiter_uattn_gqa(shape.num_q_heads, shape.num_kv_heads),
        large_compiled ? block_m_large : MT_AITER_UATTN_BLOCK_M,
        large_compiled ? block_q_large : MT_AITER_UATTN_BLOCK_Q);

    c.shape       = shape;
    c.initialized = true;
    const bool kernels_ok = c.h_3d && c.h_3d_md && c.h_reduce && c.h_2d && c.h_2d_large;
    if (!kernels_ok) {
        std::fprintf(stderr,
            "mt_aiter_unified_attn: registry could not compile/load kernels "
            "(3d=%p, 3d_md=%p, reduce=%p, 2d=%p, 2d_large=%p, target=%s, h=%d nq=%d nkv=%d bs=%d ct=%d)\n",
            (const void*)c.h_3d, (const void*)c.h_3d_md, (const void*)c.h_reduce, (const void*)c.h_2d, (const void*)c.h_2d_large,
            target.c_str(),
            shape.head_size, shape.num_q_heads, shape.num_kv_heads, shape.block_size, shape.cache_type);
        c.init_err = hipErrorInvalidImage;
    }
    return c.init_err;
}

// MAD-2026-09-12 predequant-scratch + garbage-16k fix: lazily grow the f16
// scratch K/V paged cache for THIS ISSUING STREAM to at least
// `num_scratch_blocks` slots. Never shrinks — a later call needing fewer
// slots just reuses the existing (larger) buffer, so the high-water mark is
// the largest single call's compacted slot count seen so far (e.g. the
// largest prefill batch), NOT the paged cache's total physical block
// capacity — see predequant-scratch-0912.txt for why that distinction
// matters (512 MiB/cache x2 permanently vs. a few tens of MiB scaled to the
// actual prefill).
//
// Keyed by `stream`, not just by device (garbage-16k-0912.txt): CachedHandles
// is shared per PHYSICAL DEVICE across every ggml_backend_cuda_context that
// ever runs on it — in particular both the target llama_context and any
// independent draft llama_context (common/speculative.cpp's ctx_dft, e.g.
// the MTP "nextn" head), each of which owns its own hipStream_t even on the
// same device. Before this fix a single scratch_k/scratch_v pair was shared
// by both: ctx_dft's own turbo4_fp8 2D-large prefill (same cache type, just
// a smaller SWA-ring context) could dequant-write into the exact buffer
// ctx_tgt's prefill was concurrently writing/reading on its own stream, with
// nothing serializing the two streams against each other — scratch_mu only
// ever guarded the malloc/free bookkeeping, never actual kernel execution.
// Keying the buffer by stream gives each context its own physical scratch
// memory, so there is nothing left to race on.
//
// Returns false (leaving any previous buffer for this stream untouched) on
// allocation failure; the caller falls back to the in-kernel fp8 dequant
// path for that call rather than aborting.
bool ensure_predequant_scratch(CachedHandles & c, const mt_aiter_uattn_shape_t & shape,
                                int32_t num_scratch_blocks, hipStream_t stream,
                                void ** out_k, void ** out_v) {
    std::lock_guard<std::mutex> g(c.scratch_mu);
    CachedHandles::ScratchBuf & sb = c.scratch_by_stream[stream];
    if ((size_t) num_scratch_blocks <= sb.blocks && sb.k && sb.v) {
        *out_k = sb.k;
        *out_v = sb.v;
        return true;
    }
    const size_t bytes_per_cache = (size_t) num_scratch_blocks * (size_t) shape.block_size
        * (size_t) shape.num_kv_heads * (size_t) shape.head_size * sizeof(uint16_t);
    void * new_k = nullptr;
    void * new_v = nullptr;
    if (hipMalloc(&new_k, bytes_per_cache) != hipSuccess) return false;
    if (hipMalloc(&new_v, bytes_per_cache) != hipSuccess) { (void) hipFree(new_k); return false; }
    // hipFree implicitly device-synchronizes, so any in-flight kernel on
    // THIS stream (or any other) that was still reading/writing the old
    // buffer for this stream has necessarily drained first — safe, and this
    // regrowth path is rare (grow-only, high-water mark) so the cost is
    // acceptable.
    if (sb.k) (void) hipFree(sb.k);
    if (sb.v) (void) hipFree(sb.v);
    sb.k      = new_k;
    sb.v      = new_v;
    sb.blocks = (size_t) num_scratch_blocks;
    *out_k = new_k;
    *out_v = new_v;
    if (!sb.logged) {
        int dev = 0;
        (void) hipGetDevice(&dev);
        std::fprintf(stderr,
            "mt_aiter_unified_attn: fp8-predequant path active on device %d stream=%p "
            "(num_scratch_blocks=%d, scratch=%zu B/cache x2)\n",
            dev, (void*) stream, num_scratch_blocks, bytes_per_cache);
        sb.logged = true;
    }
    return true;
}

// MAD-2026-09-12 dispatch-fix: per-device CU count, queried once and cached.
// Used only by mt_aiter_uattn_should_use_2d()'s occupancy test below —
// separate from CachedHandles/get_cached() because this must be callable
// from mt_pagedattn_aiter.cu's workspace-allocation gate BEFORE
// ensure_initialized() has necessarily run for this device (workspace is
// sized before mt_aiter_unified_attn() — and therefore ensure_initialized()
// — is ever called).
int cached_cu_count_for_current_device() {
    static std::mutex mu;
    static std::unordered_map<int, int> per_device;
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) {
        dev = 0;
    }
    std::lock_guard<std::mutex> g(mu);
    auto it = per_device.find(dev);
    if (it != per_device.end()) {
        return it->second;
    }
    hipDeviceProp_t prop {};
    int cu = 0;
    if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
        cu = prop.multiProcessorCount;
    }
    per_device[dev] = cu;
    return cu;
}

}  // anonymous namespace

// MAD-2026-09-12 dispatch-fix (decode-depth-scaling-0912.txt): occupancy-
// driven 2D/3D dispatch predicate — see the declaration comment in
// mt_aiter_unified_attn.h and the long comment above mt_aiter_uattn_use_2d()
// there. Mirrors upstream's use_2d_kernel + program-count formula
// (kernels/unified_attention_host_reference.py:33-129), minus the
// sliding_window/max_seqlen_k<=512 short-context clause (neither dispatch
// site here currently threads sliding-window or a real max_seqlen_k through
// to this call, and skipping that clause only ever biases the choice toward
// 3D, which is never worse than 2D at low context — 3D degrades gracefully
// to a single, mostly-empty segment when context is tiny).
int mt_aiter_uattn_should_use_2d(int num_q_tokens, int num_seqs, int num_kv_heads) {
    // Env override, read once. Matches the read-once-via-function-local-
    // static style already used elsewhere in this file (e.g.
    // aiter_backend_enabled(), mt_aiter_scan_block_table's `on`) rather than
    // std::call_once — a first-call data race between threads reading the
    // same env value is benign (same result either way) and this file
    // doesn't otherwise synchronize env reads.
    static int force = -1;  // -1 = not yet checked, 0 = no override, 1 = force 3D, 2 = force 2D
    if (force < 0) {
        const char * e = std::getenv("MT_AITER_UATTN_FORCE_2D");
        force = e ? (std::atoi(e) != 0 ? 2 : 1) : 0;
    }
    if (force == 2) { return 1; }
    if (force == 1) { return 0; }

    const int avg_q_len = mt_aiter_uattn_avg_q_len(num_q_tokens, num_seqs);
    // 2D-large prefill path unchanged (task spec): avg_q_len >= 256 always
    // takes the wide 2D-large tile regardless of occupancy.
    if (avg_q_len >= MT_AITER_UATTN_LARGE_PREFILL_THRESHOLD) {
        return 1;
    }

    const int kv_heads        = num_kv_heads > 0 ? num_kv_heads : 1;
    const int num_2d_prgms    = (num_q_tokens / MT_AITER_UATTN_BLOCK_Q + num_seqs) * kv_heads;
    const int cu_count        = cached_cu_count_for_current_device();
    const int target_num_prgms = (cu_count > 0 ? cu_count : 1) * 4;

    return (num_2d_prgms > target_num_prgms) ? 1 : 0;
}

hipError_t mt_aiter_unified_attn(hipStream_t stream,
                                  const mt_aiter_uattn_args_t *a) {
    hipError_t init_err = ensure_initialized(a->shape);
    if (init_err != hipSuccess) return init_err;
    // Non-const: the fp8-predequant path below lazily grows this stream's
    // scratch entry (c.scratch_by_stream[stream]) under c.scratch_mu on
    // (possibly) every call, not just the first.
    CachedHandles & c = get_cached();
    if (!c.h_2d || !c.h_2d_large || !c.h_3d || !c.h_3d_md || !c.h_reduce) return hipErrorInvalidImage;

    // MAD-199 D3 + MAD-203: three-way dispatch.
    //
    // 3D split-K is only a win for short q (decode); for prefill the
    // per-segment overhead and reduce-segments pass dominate. Within the
    // 2D path, switching to BLOCK_M=64 / BLOCK_Q=8 (large-prefill spec)
    // gives 4× LDS reuse vs the base BLOCK_M=16 spec, but BLOCK_Q=8 only
    // pays off when each Q block has ≥256 tokens to chew through (matches
    // upstream's max_seqlen_q >= 256 cutover).
    const int32_t avg_q_len    = mt_aiter_uattn_avg_q_len(a->num_q_tokens, a->num_seqs);
    // MAD-2026-09-12 dispatch-fix: occupancy-driven predicate (see
    // mt_aiter_uattn_should_use_2d() above and mt_aiter_unified_attn.h) in
    // place of the old avg_q_len>=BLOCK_Q token-count proxy. MUST use the
    // same shape.num_kv_heads the caller will pass — mt_pagedattn_aiter.cu's
    // workspace-allocation gate calls this exact function with the same
    // three arguments.
    const bool    use_2d       = mt_aiter_uattn_should_use_2d(a->num_q_tokens, a->num_seqs, a->shape.num_kv_heads) != 0;
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
    // The AOT spec was compiled with ":16" divisibility on these strides
    // (see mt_aiter_q_stride_div16 / mt_aiter_kv_stride_div16); a runtime
    // value that breaks that promise would make the vectorized loads read
    // misaligned memory, so refuse loudly instead.
    {
        const mt_aiter_uattn_shape_t & sh = a->shape;
        const bool q_div  = mt_aiter_q_stride_div16(sh);
        const bool kv_div = mt_aiter_kv_stride_div16(sh);
        const bool ok = (!q_div || (qs0 % 16 == 0 && a->output_stride_0 % 16 == 0)) &&
                        (!kv_div || (ks0 % 16 == 0 && ks1 % 16 == 0 && ks2 % 16 == 0 &&
                                     vs0 % 16 == 0 && vs1 % 16 == 0 && vs2 % 16 == 0));
        if (!ok) {
            std::fprintf(stderr, "mt_aiter_unified_attn: stride divisibility promise violated "
                         "(q0=%lld out0=%lld k=%lld/%lld/%lld v=%lld/%lld/%lld)\n",
                         (long long) qs0, (long long) a->output_stride_0, (long long) ks0, (long long) ks1,
                         (long long) ks2, (long long) vs0, (long long) vs1, (long long) vs2);
            std::abort();
        }
    }
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
            ? c.block_q_large
            : MT_AITER_UATTN_BLOCK_Q;
        const aiter::KernelHandle * h_2d_selected = use_2d_large ? c.h_2d_large : c.h_2d;

        unsigned int g2_x = (unsigned int) a->shape.num_kv_heads;
        unsigned int g2_y = (unsigned int)(num_q_tokens / block_q_for_grid + num_seqs);
        unsigned int g2_z = 1;

        // MAD-2026-09-11 fp8-predequant, MAD-2026-09-12 predequant-scratch:
        // 2D-large-prefill only. Dequant the production turbo4_fp8 K/V cache
        // into an f16 scratch paged cache — COMPACTED block indices
        // (a->scratch_block_tables), not the physical ones, since
        // predequant-scratch-0912.txt — standard f16 layout otherwise, then
        // launch the UNMODIFIED F16 2D-large kernel against the scratch
        // cache (indexed by that same compacted table) — bit-identical to
        // the fp8 in-kernel-dequant path by construction (see
        // dequant_turbo4_fp8_bs256_to_f16_2d's docstring in
        // kernels/unified_attention.py). Falls through to the normal fp8
        // launch below if scratch growth fails for this call.
        void * scratch_k_this_stream = nullptr;
        void * scratch_v_this_stream = nullptr;
        if (use_2d_large && c.predequant_enabled && a->num_scratch_blocks > 0 && a->scratch_block_tables
            && ensure_predequant_scratch(c, a->shape, a->num_scratch_blocks, stream,
                                          &scratch_k_this_stream, &scratch_v_this_stream)) {
            hipDeviceptr_t p_k_f16 = (hipDeviceptr_t) scratch_k_this_stream;
            hipDeviceptr_t p_v_f16 = (hipDeviceptr_t) scratch_v_this_stream;
            int64_t        bt_stride_dq = a->block_table_stride;
            // MAD-2026-09-12 predequant-scratch: compacted [num_seqs,
            // block_table_stride] table (prefix[seq]+slot, or -1) built on
            // the host from seq_lens/block_size — where the dequant kernel
            // WRITES, as opposed to p_bt (physical), where it READS.
            hipDeviceptr_t p_bt_scratch = (hipDeviceptr_t) a->scratch_block_tables;

            void *args_dequant[] = {
                &p_k, &p_v,               // fp8 source caches (a->k_cache / a->v_cache)
                &p_k_f16, &p_v_f16,       // f16 scratch destination caches
                &p_bt, &p_bt_scratch, &p_sl,
                &p_centroids_k, &p_centroids_v,
                &bt_stride_dq,
                &p_global_scratch, &p_profile_scratch,
            };
            unsigned int gd_x = (unsigned int) num_seqs;
            unsigned int gd_y = (unsigned int) a->block_table_stride;
            unsigned int gd_z = (unsigned int) a->shape.num_kv_heads;
            hipError_t dq_err = c.h_dequant->launch(stream, gd_x, gd_y, gd_z, args_dequant);
            if (dq_err == hipSuccess) {
                // F16-shadow strides: the scratch cache is a fresh, tightly
                // packed [num_blocks_allocated, BLOCK_SIZE, n_kv_heads,
                // HEAD_SIZE] f16 buffer (ensure_predequant_scratch), so the
                // element strides are the standard contiguous f16 formula —
                // identical to what mt_pagedattn_aiter.cu computes for a real
                // F16 k_cache/v_cache of this shape.
                int64_t ks0_f16 = (int64_t) a->shape.block_size * a->shape.num_kv_heads * a->shape.head_size;
                int64_t ks1_f16 = (int64_t) a->shape.num_kv_heads * a->shape.head_size;
                int64_t ks2_f16 = (int64_t) a->shape.head_size;
                // A real F16 call passes NULL centroids (mt_aiter_cache_is_turbo_fp8
                // is false for cache_type F16) — match that exactly rather than
                // relying on CACHE_TYPE=0 dead-code elimination to make the fp8
                // pointers harmless.
                hipDeviceptr_t p_centroids_null = (hipDeviceptr_t) nullptr;

                // MAD-2026-09-12 predequant-scratch: the F16 shadow kernel
                // reads K/V through the SAME table it loaded from — since
                // k_f16/v_f16 above are the scratch caches (compacted
                // indexing), this launch must index them with p_bt_scratch,
                // not the physical p_bt used for the real fp8 cache.
                void *args_2d_f16[] = {
                    &p_out, &p_q, &p_k_f16, &p_v_f16,
                    &p_sink,
                    &p_bt_scratch, &p_sl,
                    &p_alibi, &p_qq_bias,
                    &scale_f,
                    &p_qd, &p_kd, &p_vd, &p_os,
                    &softcap_f,
                    &bts, &qs0, &os0, &qqs0,
                    &ks0_f16, &ks1_f16, &ks2_f16,
                    &ks0_f16, &ks1_f16, &ks2_f16,  // v strides == k strides (identical scratch layout)
                    &p_cu, &num_seqs,
                    &p_centroids_null, &p_centroids_null,  // NULL, matching a real F16 call
                    &p_global_scratch, &p_profile_scratch,
                };
                unsigned int gf_y = (unsigned int)(num_q_tokens / c.block_q_large_f16 + num_seqs);
                return c.h_2d_large_f16->launch(stream, g2_x, gf_y, g2_z, args_2d_f16);
            }
            // Dequant launch failed — fall through to the normal fp8 2D-large
            // launch below rather than propagating a spurious error; the
            // in-kernel dequant path is still fully correct.
        }

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

    // MAD-2026-09-12 dispatch-fix: pick ALL_DECODE=1 vs ALL_DECODE=0 3D
    // handle. ALL_DECODE=1 (seq_idx = program_id(0) directly, cur_batch_
    // query_len hardcoded to 1 — kernels/unified_attention.py:1657-1663) is
    // only valid when every one of the num_seqs slots contributes EXACTLY
    // one query token this call. `num_q_tokens == num_seqs` is a necessary
    // AND (for this codebase's actual traffic shapes) sufficient proxy for
    // that: pure decode always satisfies it (each live seq contributes 1
    // token, num_seqs counts only live-per-tensor-shape slots that all
    // decode together); MTP/DFlash verify batches never satisfy it (every
    // live seq contributes num_q_tokens/num_seqs > 1 tokens uniformly, so
    // the sum != num_seqs whenever depth > 1); and — this is the par-4 fault
    // review finding, see par-4-fault-report.txt candidate #2 — a partially-
    // masked decode call (num_seqs fixed at n_seq_max by block_tables'
    // tensor shape, but fewer than n_seq_max sequences actually live this
    // step, so num_q_tokens < num_seqs) ALSO now correctly falls through to
    // the ALL_DECODE=0 handle instead of ALL_DECODE=1.
    //   This matters beyond just picking the "more correct" kernel: with
    // ALL_DECODE=1's cur_batch_query_len hardcoded to 1, a dead/masked seq
    // slot's cu_seqlens entry (query_start_len_ptr[seq_idx], built by
    // mt_build_cu_seqlens_kernel from that slot's own q_len — 0 for a dead
    // slot) is a REPEAT of the previous live slot's boundary, not a fresh
    // one; the hardcoded query_mask_0 = (query_pos < 1) then evaluates TRUE
    // for that dead slot's "query" anyway (query_pos=0 < cur_batch_query_len
    // forced to 1), so the kernel treats a phantom query token as real: it
    // reads Q at that repeated boundary offset — up to one row PAST the end
    // of the real [num_q_tokens, ...] Q tensor when the dead slot is the
    // last one scanned — and, worse, WRITES segm_output/segm_max/segm_expsum
    // at that same one-past-the-end row, i.e. an out-of-bounds heap write
    // into whatever pool allocation follows the (num_q_tokens-sized) segm_*
    // workspace. That is a plausible mechanism for the kind of heap
    // corruption that later surfaces as an unrelated HSA_STATUS_ERROR_
    // MEMORY_FAULT. Routing this case to ALL_DECODE=0 (which derives
    // cur_batch_query_len from the REAL cu_seqlens diff via find_seq_idx, so
    // a dead slot's query_len is genuinely 0 and query_mask_0 is correctly
    // all-False for it) closes this off entirely rather than papering over
    // it with an extra clamp.
    const bool all_decode = (num_q_tokens == num_seqs);
    const aiter::KernelHandle * h_3d_selected = all_decode ? c.h_3d : c.h_3d_md;

    unsigned int g3_x;
    if (all_decode) {
        // ALL_DECODE grid: one q-block (= one query token) per sequence
        // (PR #2888).
        g3_x = (unsigned int) num_seqs;
    } else {
        // ALL_DECODE=0 grid: upstream's axis order/formula — total_num_q_
        // blocks = num_q_tokens/BLOCK_Q + num_seqs (upper bound on
        // sum_i[ceil(query_len[i]/BLOCK_Q)]; see the derivation comment in
        // unified_attention_host_reference.py's unified_attention()). This
        // is a DIFFERENT axis-0 quantity than num_seqs — they only coincide
        // when every sequence contributes exactly one q-block, which is
        // exactly the invariant `all_decode` above checks.
        g3_x = (unsigned int)(num_q_tokens / MT_AITER_UATTN_BLOCK_Q + num_seqs);
    }
    unsigned int g3_y = (unsigned int) a->shape.num_kv_heads;
    unsigned int g3_z = (unsigned int) MT_AITER_UATTN_NUM_SEGMENTS_PER_SEQ;

    hipError_t err = h_3d_selected->launch(stream, g3_x, g3_y, g3_z, args_3d);
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
