#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"
#include "mmid.cuh"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {
// Thread-local side channel for routing-aware MMQ on consolidated MoE
// (MAD-88 Phase 2). The weight-pager eval callback sets this just
// before a MUL_MAT_ID op runs, supplying a device-resident array of
// per-expert weight pointers. ggml_cuda_mul_mat_q reads it via
// take_routed_expert_ptrs() (which clears it) and threads it into the
// mmq_args struct.
//
// Lifetime: the caller (eval cb) owns the array memory and guarantees
// it stays valid until the kernel completes. Single-shot — the
// dispatcher clears it after read, so a missed-set or a pure
// MUL_MAT (no MoE) op uses the default nullptr (legacy path,
// bit-identical to pre-MAD-88).
thread_local const void * const * tls_routed_expert_ptrs = nullptr;
thread_local std::vector<const void * const *> tls_routed_expert_ptrs_queue;
thread_local size_t tls_routed_expert_ptrs_queue_i = 0;
std::atomic<uint64_t> g_routed_expert_ptrs_set{0};
std::atomic<uint64_t> g_routed_expert_ptrs_consumed{0};
std::atomic<uint64_t> g_routed_expert_ptrs_discarded_unconsumed{0};

// MAD-230 follow-up: GGML's CUDA streams are created with
// cudaStreamNonBlocking (common.cuh:1439), so they do NOT implicitly
// serialize with the default (NULL) stream that a synchronous hipMemcpy
// uses. Without this side channel, the weight pager's eval_cb has to
// hipDeviceSynchronize() to flush the compute stream before each
// host→device write of the expert-pointer array — a heavy per-MoE-op
// stall — and even then there's a torn-pointer race window that produced
// the near-null GPU faults during MoE prefill. Exposing the compute
// stream to the eval_cb lets it use hipMemcpyAsync(stream), which is
// stream-ordered with the MMQ kernels and removes both the race and
// the device-wide sync.
thread_local void * tls_wp_compute_stream[GGML_CUDA_MAX_DEVICES] = { nullptr };
}  // namespace

void ggml_cuda_set_routed_expert_ptrs(const void * const * ptr) {
    if (tls_routed_expert_ptrs != nullptr) {
        g_routed_expert_ptrs_discarded_unconsumed.fetch_add(1, std::memory_order_relaxed);
    }
    if (ptr != nullptr) {
        g_routed_expert_ptrs_set.fetch_add(1, std::memory_order_relaxed);
    }
    tls_routed_expert_ptrs = ptr;
}

void ggml_cuda_queue_routed_expert_ptrs(const void * const * ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (tls_routed_expert_ptrs_queue_i > 0 &&
            tls_routed_expert_ptrs_queue_i == tls_routed_expert_ptrs_queue.size()) {
        tls_routed_expert_ptrs_queue.clear();
        tls_routed_expert_ptrs_queue_i = 0;
    }
    g_routed_expert_ptrs_set.fetch_add(1, std::memory_order_relaxed);
    tls_routed_expert_ptrs_queue.push_back(ptr);
}

const void * const * ggml_cuda_take_routed_expert_ptrs() {
    if (tls_routed_expert_ptrs_queue_i < tls_routed_expert_ptrs_queue.size()) {
        const void * const * p = tls_routed_expert_ptrs_queue[tls_routed_expert_ptrs_queue_i++];
        g_routed_expert_ptrs_consumed.fetch_add(1, std::memory_order_relaxed);
        return p;
    }
    const void * const * p = tls_routed_expert_ptrs;
    if (p != nullptr) {
        g_routed_expert_ptrs_consumed.fetch_add(1, std::memory_order_relaxed);
    }
    tls_routed_expert_ptrs = nullptr;
    return p;
}

bool ggml_cuda_has_routed_expert_ptrs() {
    // Non-consuming peek. The MUL_MAT_ID dispatcher uses this to decide
    // whether to bypass kernel paths (mmvq, mmvf, mmf) that don't support
    // routing-aware paging and force the MMQ path which does.
    return tls_routed_expert_ptrs != nullptr ||
        tls_routed_expert_ptrs_queue_i < tls_routed_expert_ptrs_queue.size();
}

void ggml_cuda_discard_routed_expert_ptrs() {
    const size_t queued = tls_routed_expert_ptrs_queue.size() - tls_routed_expert_ptrs_queue_i;
    if (queued > 0) {
        g_routed_expert_ptrs_discarded_unconsumed.fetch_add(
            queued, std::memory_order_relaxed);
    }
    tls_routed_expert_ptrs_queue.clear();
    tls_routed_expert_ptrs_queue_i = 0;
    if (tls_routed_expert_ptrs != nullptr) {
        g_routed_expert_ptrs_discarded_unconsumed.fetch_add(1, std::memory_order_relaxed);
    }
    tls_routed_expert_ptrs = nullptr;
}

void ggml_cuda_get_routed_expert_ptrs_stats(uint64_t * set, uint64_t * consumed, uint64_t * discarded_unconsumed) {
    if (set != nullptr) {
        *set = g_routed_expert_ptrs_set.load(std::memory_order_relaxed);
    }
    if (consumed != nullptr) {
        *consumed = g_routed_expert_ptrs_consumed.load(std::memory_order_relaxed);
    }
    if (discarded_unconsumed != nullptr) {
        *discarded_unconsumed = g_routed_expert_ptrs_discarded_unconsumed.load(std::memory_order_relaxed);
    }
}

const void * const * ggml_cuda_resolve_mul_mat_id_expert_ptrs(const ggml_tensor * dst) {
    const void * const * p = ggml_cuda_take_routed_expert_ptrs();
    if (p == nullptr) {
        p = (const void * const *) ggml_mul_mat_id_get_expert_ptrs(dst);
    }
    const int32_t n_as = ggml_mul_mat_id_get_expert_ptrs_n_as(dst);
    if (n_as > 0 && p == nullptr) {
        GGML_ABORT("MUL_MAT_ID %s missing expert_ptrs n_as=%d",
                   (dst != nullptr && dst->name[0] != '\0') ? dst->name : "<unnamed>",
                   n_as);
    }
    return p;
}

static bool ggml_cuda_wp_routing_guard_enabled() {
    static const bool enabled = []() {
        const char * env = std::getenv("WP_ROUTING_GUARD");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

void ggml_cuda_wp_routing_guard_check(
        const char * path, const ggml_tensor * src0, const ggml_tensor * ids, const ggml_tensor * dst,
        const void * const * expert_ptrs) {
    if (!ggml_cuda_wp_routing_guard_enabled()) {
        return;
    }
    if (ids != nullptr && expert_ptrs != nullptr) {
        return;
    }

    fprintf(stderr,
            "WP_ROUTING_GUARD: %s routed op would read src0->data directly: src0=%s src0_data=%p ids=%p dst=%s expert_ptrs=%p\n",
            path,
            src0 != nullptr ? src0->name : "<null>",
            src0 != nullptr ? src0->data : nullptr,
            (const void *) ids,
            dst != nullptr ? dst->name : "<null>",
            (const void *) expert_ptrs);
    GGML_ABORT("WP_ROUTING_GUARD routed expert pointer invariant failed");
}

void ggml_cuda_set_wp_compute_stream(int device, void * stream) {
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
        tls_wp_compute_stream[device] = stream;
    }
}

void * ggml_cuda_get_wp_compute_stream(int device) {
    if (device >= 0 && device < GGML_CUDA_MAX_DEVICES) {
        return tls_wp_compute_stream[device];
    }
    return nullptr;
}

static void ggml_cuda_mul_mat_q_switch_type(ggml_backend_cuda_context & ctx, const mmq_args & args, cudaStream_t stream, const ggml_prec prec_src1) {
    switch (args.type_x) {
        case GGML_TYPE_Q1_0:
            mul_mat_q_case<GGML_TYPE_Q1_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q2_0:
            mul_mat_q_case<GGML_TYPE_Q2_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_0:
            mul_mat_q_case<GGML_TYPE_Q4_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_1:
            mul_mat_q_case<GGML_TYPE_Q4_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_0:
            mul_mat_q_case<GGML_TYPE_Q5_0>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_1:
            mul_mat_q_case<GGML_TYPE_Q5_1>(ctx, args, stream);
            break;
        case GGML_TYPE_Q8_0:
            mul_mat_q_case<GGML_TYPE_Q8_0>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_Q2_K:
            mul_mat_q_case<GGML_TYPE_Q2_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q3_K:
            mul_mat_q_case<GGML_TYPE_Q3_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q4_K:
            mul_mat_q_case<GGML_TYPE_Q4_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q5_K:
            mul_mat_q_case<GGML_TYPE_Q5_K>(ctx, args, stream);
            break;
        case GGML_TYPE_Q6_K:
            mul_mat_q_case<GGML_TYPE_Q6_K>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_IQ1_S:
            mul_mat_q_case<GGML_TYPE_IQ1_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            mul_mat_q_case<GGML_TYPE_IQ2_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            mul_mat_q_case<GGML_TYPE_IQ2_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ2_S:
            mul_mat_q_case<GGML_TYPE_IQ2_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            mul_mat_q_case<GGML_TYPE_IQ3_XXS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ3_S:
            mul_mat_q_case<GGML_TYPE_IQ3_S>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            mul_mat_q_case<GGML_TYPE_IQ4_XS>(ctx, args, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            mul_mat_q_case<GGML_TYPE_IQ4_NL>(ctx, args, stream);
            break;
// -----------------------------------------------------------------------
        case GGML_TYPE_MXFP4:
            // src1 at Q4 uses the native FP4 instructions, which are Blackwell-only
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_MXFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_MXFP4>(ctx, args, stream);
            break;
        case GGML_TYPE_NVFP4:
            if (prec_src1 == GGML_PREC_Q4) {
                mul_mat_q_case<GGML_TYPE_NVFP4, GGML_PREC_Q4>(ctx, args, stream);
                break;
            }
            mul_mat_q_case<GGML_TYPE_NVFP4>(ctx, args, stream);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

// GGML_MMQ_DEBUG_SYNC=1: synchronize and check after every stage of the MUL_MAT_ID MMQ path and
// print the launch geometry, to localize device faults on GPUs the sanitizers no longer support
bool ggml_cuda_mmq_debug_sync_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_MMQ_DEBUG_SYNC");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

void ggml_cuda_mmq_debug_sync(const char * stage, cudaStream_t stream) {
    if (!ggml_cuda_mmq_debug_sync_enabled()) {
        return;
    }
    const cudaError_t err = cudaStreamSynchronize(stream);
    const cudaError_t last = cudaGetLastError();
    fprintf(stderr, "[mmq-debug] stage %s: sync=%s last=%s\n", stage, cudaGetErrorString(err), cudaGetErrorString(last));
    if (err != cudaSuccess || last != cudaSuccess) {
        GGML_ABORT("mmq debug: device fault after stage %s", stage);
    }
}

// overrides the src1 precision requested by the graph, "auto" keeps the requested one
static ggml_prec ggml_cuda_mmq_get_prec_env() {
    const char * env_c = getenv("GGML_CUDA_MMQ_PREC");
    if (env_c == nullptr) {
        return GGML_PREC_UNDEFINED;
    }
    std::string env_cpp = env_c;
    for (char & c : env_cpp) {
        c = std::tolower(c);
    }
    if (env_cpp == "q4") {
        return GGML_PREC_Q4;
    }
    if (env_cpp == "q8") {
        return GGML_PREC_Q8;
    }
    if (env_cpp != "auto") {
        GGML_LOG_WARN("%s: Unknown value for GGML_CUDA_MMQ_PREC: '%s'. Available: 'q4', 'q8', 'auto'.\n", __func__, env_cpp.c_str());
    }
    return GGML_PREC_UNDEFINED;
}

// src1 is quantized to Q8_1 unless the FP4 types can use 4-bit activations, in which case they
// default to the native W4A4 instructions on Blackwell.
static ggml_prec ggml_cuda_mmq_get_prec_src1(const ggml_tensor * src0, const ggml_tensor * dst, const int cc) {
    static const ggml_prec prec_env = ggml_cuda_mmq_get_prec_env();

    ggml_prec prec = prec_env;
    if (prec == GGML_PREC_UNDEFINED) {
        prec = (ggml_prec) ggml_get_op_params_i32(dst, 3);
    }

    // Q4 only for the FP4 types on Blackwell
    GGML_ASSERT(prec == GGML_PREC_UNDEFINED || prec == GGML_PREC_Q8 || prec == GGML_PREC_Q4);
    const bool can_use_q4 = (src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4) && blackwell_mma_available(cc);
    if (prec == GGML_PREC_Q8 || !can_use_q4) {
        return GGML_PREC_Q8;
    }
    return GGML_PREC_Q4;
}

void ggml_cuda_mul_mat_q(
        ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids, ggml_tensor * dst,
        bool force_mm_id) {
    GGML_ASSERT(        src1->type == GGML_TYPE_F32);
    GGML_ASSERT(        dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(!ids || ids->type  == GGML_TYPE_I32); // Optional, used for batched GGML_MUL_MAT_ID.

    GGML_TENSOR_BINARY_OP_LOCALS;

    cudaStream_t stream = ctx.stream();
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    const size_t ts_src0 = ggml_type_size(src0->type);
    const size_t ts_src1 = ggml_type_size(src1->type);
    const size_t ts_dst  = ggml_type_size(dst->type);

    GGML_ASSERT(        nb00       == ts_src0);
    GGML_ASSERT(        nb10       == ts_src1);
    GGML_ASSERT(        nb0        == ts_dst);
    GGML_ASSERT(!ids || ids->nb[0] == ggml_type_size(ids->type));

    const char  * src0_d = (const char  *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       *  dst_d = (float       *)  dst->data;

    // If src0 is a temporary compute buffer, clear any potential padding.
    //
    // SKIP when routing-aware paging is active for this op: the consolidated
    // MoE parent's src0->data is a placeholder pointer (pool base, not real
    // tensor storage), and the kernel will read per-expert pointers from the
    // expert_ptrs side channel instead. Writing past placeholder with size_data
    // bytes faults the GPU (near-null offset since placeholder may be 0-based
    // within the pool view).
    const bool routing_was_set = ggml_cuda_has_routed_expert_ptrs() ||
        ggml_mul_mat_id_get_expert_ptrs_n_as(dst) > 0;

    if (!routing_was_set &&
        ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        const size_t size_data  = ggml_nbytes(src0);
        const size_t size_alloc = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
        if (size_alloc > size_data) {
            GGML_ASSERT(ggml_is_contiguously_allocated(src0));
            GGML_ASSERT(!src0->view_src);
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + size_data, 0, size_alloc - size_data, stream));
        }
    } else if (routing_was_set) {
        static int s_dump = 0;
        if (s_dump < 4) {
            fprintf(stderr, "[mmq DIAG] routing active: src0=%s data=%p buf=%p nbytes=%zu\n",
                    src0->name, src0->data, (void*)src0->buffer, ggml_nbytes(src0));
            ++s_dump;
        }
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    const int64_t s01 = src0->nb[1] / ts_src0;
    const int64_t s1  =  dst->nb[1] / ts_dst;
    const int64_t s02 = src0->nb[2] / ts_src0;
    const int64_t s2  =  dst->nb[2] / ts_dst;
    const int64_t s03 = src0->nb[3] / ts_src0;
    const int64_t s3  =  dst->nb[3] / ts_dst;

    const bool fallback = ne01 % 128 != 0;

    const ggml_prec prec_src1 = ggml_cuda_mmq_get_prec_src1(src0, dst, cc);

    const bool use_native_fp4 = prec_src1 == GGML_PREC_Q4;
    const size_t y_block_size       = use_native_fp4 ? sizeof(block_fp4_mmq) : sizeof(block_q8_1_mmq);
    const size_t y_values_per_block = use_native_fp4 ? QK_FP4_MMQ            : QK8_1_MMQ;

    // Pull any routing-aware expert pointer array set by the weight-pager
    // eval callback. take_*() clears the TLS, so this op consumes it
    // exactly once. nullptr (default) is the legacy bit-identical path.
    const void * const * routed_expert_ptrs = ggml_cuda_resolve_mul_mat_id_expert_ptrs(dst);
    if (routing_was_set) {
        ggml_cuda_wp_routing_guard_check("MMQ", src0, ids, dst, routed_expert_ptrs);
    }

    // Validation harness for MAD-88 Phase 2 (kernel hook correctness).
    // When WP_MMQ_VALIDATE_EXPERT_PTRS=1 is set in the environment, build
    // a "golden" expert_ptrs array of pointers that match exactly what the
    // legacy x + c*stride_channel_x address computation produces. Both
    // paths must yield bit-identical output — token sequences from the
    // same prompt + seed should match between WP_MMQ_VALIDATE_EXPERT_PTRS
    // off and on. Only fires for MUL_MAT_ID (ids != nullptr); regular
    // MUL_MAT has nchannels_x==1 so the test is meaningless.
    static const bool s_validate_expert_ptrs = []() {
        const char * env = std::getenv("WP_MMQ_VALIDATE_EXPERT_PTRS");
        return env != nullptr && env[0] == '1';
    }();
    ggml_cuda_pool_alloc<const void *> validate_expert_ptrs_buf(ctx.pool());
    if (s_validate_expert_ptrs && ids != nullptr && routed_expert_ptrs == nullptr) {
        const int64_t n_experts = src0->ne[2];
        if (n_experts > 1) {
            std::vector<const void *> host_ptrs((size_t) n_experts);
            const char * base = (const char *) src0->data;
            const size_t expert_stride_bytes = src0->nb[2];
            for (int64_t e = 0; e < n_experts; ++e) {
                host_ptrs[(size_t) e] = base + (size_t) e * expert_stride_bytes;
            }
            validate_expert_ptrs_buf.alloc((size_t) n_experts);
            CUDA_CHECK(cudaMemcpyAsync(validate_expert_ptrs_buf.ptr, host_ptrs.data(),
                                       (size_t) n_experts * sizeof(const void *),
                                       cudaMemcpyHostToDevice, stream));
            routed_expert_ptrs = validate_expert_ptrs_buf.ptr;
        }
    }

    if (!ids) {
        const size_t nbytes_src1_q8_1 = ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block +
            ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne11) * sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
        ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
        if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
            src1_scale.alloc(ne13*ne12*ne11);
        }

        {
            const int64_t s11 = src1->nb[1] / ts_src1;
            const int64_t s12 = src1->nb[2] / ts_src1;
            const int64_t s13 = src1->nb[3] / ts_src1;
            if (use_native_fp4) {
                static constexpr size_t align_float8 = 32;
                const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
                static_assert(sizeof(block_fp4_mmq) == 4 * sizeof(block_q8_1));
                quantize_mmq_fp4_cuda(src1_d, nullptr, src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13, ne10_padded,
                                        ne11, ne12, ne13, stream);

            } else {
                quantize_mmq_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded,
                                       ne11, ne12, ne13, stream);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Stride depends on quantization format
        const int64_t s12 = use_native_fp4 ?
                                ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
        const int64_t s13 = ne12*s12;

        mmq_args args = {
            src0_d, src0->type, (const int *) src1_q8_1.ptr, nullptr, nullptr, dst_d,
            src0->type == GGML_TYPE_NVFP4 && use_native_fp4 ? src1_scale.ptr : nullptr,
            ne00, ne01, ne1, s01, ne11, s1,
            ne02, ne12, s02, s12, s2,
            ne03, ne13, s03, s13, s3,
            ne1, ne1};
        args.expert_ptrs = routed_expert_ptrs;
        ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
        return;
    }

    GGML_ASSERT(ne13 == 1);
    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;
    GGML_ASSERT(ne1 == n_expert_used);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> ids_dst(ctx.pool(), ne_get_rows);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);

    // gate/up activations are broadcast across experts (ne11 == 1): quantize each token once and
    // scatter to its slots. ids_src1 then holds the inverse map (token slot -> compact row).
    const bool dedup_bcast = ne11 == 1 && n_expert_used > 1;

    {
        GGML_ASSERT(ids->nb[0] == ggml_element_size(ids));
        const int si1  = ids->nb[1] / ggml_element_size(ids);
        const int sis1 = nb12 / nb11;

        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data, ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, ne12, n_expert_used, ne11, si1, sis1, /*write_inverse =*/ dedup_bcast, stream);
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_mmq_debug_sync("mm_ids_helper", stream);
    }

    const size_t nbytes_src1_q8_1 = ne12*n_expert_used*ne10_padded * y_block_size/y_values_per_block +
        ggml_cuda_mmq_get_J_max(src0->type, fallback, cc, ne12) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), nbytes_src1_q8_1);
    ggml_cuda_pool_alloc<float> src1_scale(ctx.pool());
    if (src0->type == GGML_TYPE_NVFP4 && use_native_fp4) {
        src1_scale.alloc(ne12*n_expert_used);
    }

    const int64_t ne11_flat = ne12*n_expert_used;
    const int64_t ne12_flat = 1;
    const int64_t ne13_flat = 1;

    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;

        if (use_native_fp4) {
            static constexpr size_t align_float8 = 32;
            const bool use_aligned_float8 = ggml_cuda_is_aligned(src1, align_float8);
            if (dedup_bcast) {
                quantize_scatter_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10,
                                        /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
            } else {
                quantize_mmq_fp4_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src1_scale.ptr, src0->type, use_aligned_float8, ne10, s11, s12, s13,
                                        ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
            }
        } else if (dedup_bcast) {
            quantize_scatter_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10,
                                    /*stride_token=*/s12, ne10_padded, ne12, ne11_flat, n_expert_used, stream);
        } else {
            quantize_mmq_q8_1_cuda(src1_d, ids_src1.get(), src1_q8_1.get(), src0->type, ne10, s11, s12, s13,
                                   ne10_padded, ne11_flat, ne12_flat, ne13_flat, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        ggml_cuda_mmq_debug_sync("quantize", stream);
    }

    static_assert(QK_FP4_MMQ == 8 * QK_MXFP4, "QK_FP4_MMQ needs to be 8 * QK_MXFP4");
    const int64_t s12 = use_native_fp4 ? ne11 * ne10_padded * sizeof(block_fp4_mmq) / (QK_FP4_MMQ * sizeof(int)) :
                                         ne11 * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    const int64_t s13 = ne12*s12;

    // Each expert only sees ne12*n_expert_used/ne02 tokens on average.
    // On RDNA3 and RDNA4 it is faster to pick the tile size against this value instead of ne12.
    int64_t ncols_opt = ne12;
    if (GGML_CUDA_CC_IS_RDNA3(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
        ncols_opt = (ne12*n_expert_used + ne02 - 1) / ne02;
    }

    // Note that ne02 is used instead of ne12 because the number of y channels determines the z dimension of the CUDA grid.
    mmq_args args = {
        src0_d, src0->type, (const int *) src1_q8_1.get(), ids_dst.get(), expert_bounds.get(), dst_d,
        src1_scale.ptr,
        ne00, ne01, ne_get_rows, s01, ne_get_rows, s1,
        ne02, ne02, s02, s12, s2,
        ne03, ne13, s03, s13, s3,
        ne12, ncols_opt};
    args.force_mm_id = force_mm_id;
    args.expert_ptrs = routed_expert_ptrs;

    ggml_cuda_mul_mat_q_switch_type(ctx, args, stream, prec_src1);
    ggml_cuda_mmq_debug_sync("mul_mat_q(ids)", stream);
}

bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts, bool force_mm_id) {
#ifdef GGML_CUDA_FORCE_CUBLAS
    return false;
#endif // GGML_CUDA_FORCE_CUBLAS

    bool mmq_supported;

    switch (type) {
        case GGML_TYPE_Q1_0:
        case GGML_TYPE_Q2_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
// -------------------------------------------------
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
// -------------------------------------------------
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ4_NL:
// -------------------------------------------------
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_NVFP4:
            mmq_supported = true;
            break;
        default:
            mmq_supported = false;
            break;
    }

    if (!mmq_supported) {
        return false;
    }

    // MMQ tiles require at least 48 KiB per-block shared memory; fall back to BLAS otherwise.
    {
        const int    id    = ggml_cuda_get_device();
        const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
        if (smpbo < 48 * 1024) {
            return false;
        }
    }

    if (force_mm_id) {
        return true;
    }

    if (turing_mma_available(cc)) {
        return true;
    }

    if (ggml_cuda_highest_compiled_arch(cc) < GGML_CUDA_CC_DP4A) {
        // for MoE, mmq is faster even without native dp4a
        // TODO: check if cards older than pascal might benefit from this as well
        return cc >= GGML_CUDA_CC_PASCAL && n_experts > 0;
    }

#ifdef GGML_CUDA_FORCE_MMQ
    return true;
#endif //GGML_CUDA_FORCE_MMQ

    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
        return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    if (amd_mfma_available(cc)) {
        // As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3 and hipblaslt (via ROCBLAS_USE_HIPBLASLT)
        // performs better but is currently suffering from a crash on this architecture.
        // TODO: Revisit when hipblaslt is fixed on CDNA3
        if (GGML_CUDA_CC_IS_CDNA3(cc)) {
            return true;
        }
        if (n_experts > 64 || ne11 <= 128) {
            return true;
        }
        if (type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q4_1 || type == GGML_TYPE_Q5_0 || type == GGML_TYPE_Q5_1) {
            return true;
        }
        if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
            return true;
        }
        return false;
    }

    if (amd_wmma_available(cc)) {
        if (GGML_CUDA_CC_IS_RDNA3(cc)) {
            // High expert counts are almost always better on MMQ due to
            //     the synchronization overhead in the cuBLAS/hipBLAS path:
            // https://github.com/ggml-org/llama.cpp/pull/18202
            if (n_experts >= 64) {
                return true;
            }

            // For some quantization types MMQ can have lower peak TOPS than hipBLAS
            //     so it's only faster for sufficiently small batch sizes:
            switch (type) {
                case GGML_TYPE_Q2_K:
                    return ne11 <= 128;
                case GGML_TYPE_Q6_K:
                    return ne11 <= (GGML_CUDA_CC_IS_RDNA3_0(cc) ? 128 : 256);
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                    return GGML_CUDA_CC_IS_RDNA3_5(cc) || ne11 <= 128;
                default:
                    return true;
            }
        }

        // For RDNA4 MMQ is consistently faster than dequantization + hipBLAS:
        // https://github.com/ggml-org/llama.cpp/pull/18537#issuecomment-3706422301
        return true;
    }

    // gfx80x (Tonga/Fiji/Polaris, GGML_CUDA_CC_GCN4 and earlier) have no hardware dp4a
    // (v_dot4): ggml_cuda_dp4a falls back to fully-scalar byte emulation, which makes MMQ
    // ~8x slower than dequantization + hipBLAS for prefill-sized batches (rocBLAS ships
    // gfx803 fp16/fp32 Tensile kernels). Restrict MMQ to small (decode) batches only.
    if (GGML_CUDA_CC_IS_GCN(cc) && cc <= GGML_CUDA_CC_GCN4) {
        return ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    }

    // gfx900 (Vega 10), gfx909, and gfx90c lack native dp4a, losing to dequant + hipBLAS
    // for dense matrices; keep MMQ only for MoE, where the
    // hipBLAS path is much slower.
    if (cc == GGML_CUDA_CC_VEGA || GGML_CUDA_CC_IS_GCN_APU(cc)) {
        return n_experts > 0;
    }

    // MUSA: the MMQ kernels compute wrong values on PH1 (MTT S5000).
    if (cc == GGML_CUDA_CC_PH1) {
        return false;
    }

    return (!GGML_CUDA_CC_IS_CDNA(cc)) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
}

// mmq's q8_1 D4 activation layout (same as quantize_mmq_q8_1 for MXFP4), plus one
// extra grid column (blockIdx.x == ne1) that turns the e4m3 codebook into int8
// tables, so ML8_4 wide batches take exactly MXFP4's two launches.
static __global__ void ml8_4_quantize_mmq_q8_1_d4(
        const float * __restrict__ x, void * __restrict__ vy, const int64_t ne00, const int64_t s01,
        const int64_t ne0, const int ne1,
        const uint8_t * __restrict__ lut, int8_t * __restrict__ lut_q, float * __restrict__ lut_d, const int n_groups) {
    if ((int) blockIdx.x == ne1) {
        const int g = blockIdx.y*blockDim.x + threadIdx.x;
        if (g < n_groups) {
            int8_t t[16];
            lut_d[g] = ml8_4_group_to_q8(lut + (int64_t) g*16, t);
            *(int4 *) (lut_q + (int64_t) g*16) = *(const int4 *) t;
        }
        return;
    }

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;
    if (i0 >= ne0) {
        return;
    }
    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ;
    const int64_t iqs     = i0 % QK8_1_MMQ;

    const float4 xi = i0 < ne00 ? x4[(blockIdx.x*s01 + i0)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 32/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);

    const int64_t ib = k_block*ne1 + blockIdx.x;
    ((char4 *) y[ib].qs)[iqs/4] = q;
    if (iqs % 32 == 0) {
        y[ib].d4[iqs/32] = 1.0f / d_inv;
    }
}

void ggml_cuda_ml8_4_mul_mat_q(
        ggml_backend_cuda_context & ctx, const void * vx, const uint8_t * lut,
        const float * x, const int64_t x_stride, float * dst, const int64_t dst_stride,
        const int64_t ncols_x, const int64_t nrows_x, const int64_t ncols_y) {
    cudaStream_t stream = ctx.stream();
    const int    cc     = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const bool   fallback = nrows_x % 128 != 0;
    const int    n_groups = (int) (ncols_x / QK_ML8);

    const int64_t ne10_padded = GGML_PAD(ncols_x, MATRIX_ROW_PADDING);
    const size_t  nbytes_y    = ncols_y*ne10_padded * sizeof(block_q8_1_mmq)/QK8_1_MMQ +
        ggml_cuda_mmq_get_J_max(GGML_TYPE_ML8_4, fallback, cc, ncols_y) * sizeof(block_q8_1_mmq);
    ggml_cuda_pool_alloc<char>   y_q8_1(ctx.pool(), nbytes_y);
    ggml_cuda_pool_alloc<int8_t> lut_q(ctx.pool(), (size_t) n_groups * 16);
    ggml_cuda_pool_alloc<float>  lut_d(ctx.pool(), (size_t) n_groups);
    {
        const int64_t block_num_y = (ne10_padded + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
        const dim3 num_blocks((unsigned) (ncols_y + 1), (unsigned) block_num_y, 1);
        ml8_4_quantize_mmq_q8_1_d4<<<num_blocks, CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 0, stream>>>(
            x, y_q8_1.get(), ncols_x, x_stride, ne10_padded, (int) ncols_y, lut, lut_q.get(), lut_d.get(), n_groups);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t s12 = ncols_y * ne10_padded * sizeof(block_q8_1) / (QK8_1 * sizeof(int));
    mmq_args args = {
        (const char *) vx, GGML_TYPE_ML8_4, (const int *) y_q8_1.ptr, nullptr, nullptr, dst, nullptr,
        ncols_x, nrows_x, ncols_y, ncols_x / QK_ML8, ncols_y, dst_stride,
        1, 1, 0, s12, 0,
        1, 1, 0, s12, 0,
        ncols_y, ncols_y};
    args.ml8_lut_q = lut_q.get();
    args.ml8_lut_d = lut_d.get();
    mul_mat_q_case<GGML_TYPE_ML8_4>(ctx, args, stream);
}
