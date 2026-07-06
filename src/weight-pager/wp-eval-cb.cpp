#include "wp-eval-cb.h"
#include "wp-pager.h"

#include "ggml.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
// Forward decl of the ggml-cuda side channel — the actual symbol lives in
// libggml-hip.so and we link against it. Avoids dragging the full
// ggml-cuda/mmq.cuh into libllama's wp-eval-cb compilation unit.
extern "C++" void                  ggml_cuda_set_routed_expert_ptrs(const void * const * ptr);
extern "C++" const void * const *  ggml_cuda_take_routed_expert_ptrs();
extern "C++" void *                ggml_cuda_get_wp_compute_stream();
#endif

#include <cstdlib>       // getenv
#include <cstring>
#include <limits>        // numeric_limits — MAD-232 advise sentinel
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace wp {

namespace {
// Diagnostic counters. Logged only when WP_EVAL_DEBUG=1 is set in the
// environment. First few ops get verbose output; afterwards we suppress
// to keep logs readable.
struct DebugState {
    int  ops_seen   = 0;       // total ops the callback fired on (ask=true)
    int  ops_with_pages    = 0;  // ops that had at least one paged source
    int  patches_total     = 0;  // total src->data overwrites
    int  views_patched     = 0;  // of those, how many were view tensors
    int  ensures_failed    = 0;  // ensure() returned null
    int  mmid_ops_seen     = 0;  // GGML_OP_MUL_MAT_ID ops total (ask=true)
    int  mmid_consolidated = 0;  // of those, src[0] resolved to a consolidated parent
    static constexpr int kVerboseLimit = 200;  // log details for first N ops only
    int  ops_no_paged_with_weight_src = 0;  // ops where eval_cb saw a src whose name has "weight" but find_page missed
};
DebugState g_debug;
}  // namespace

bool weight_pager_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    // Only act on the pre-execution call. The post-execution call is
    // informational and would re-trigger the same lookups.
    if (!ask)             return true;
    if (t == nullptr)     return true;
    auto * pager = (WeightPager *) user_data;
    if (pager == nullptr) return true;

#if defined(GGML_USE_HIP)
    // MAD-230: discard any stale routed_expert_ptrs TLS that wasn't
    // consumed by a CUDA kernel on the previous op. eval_cb fires
    // for EVERY op the scheduler dispatches, including ops that
    // land on non-CUDA backends (CPU fallback for unusual shapes,
    // multi-backend split graphs, etc). If a MoE MUL_MAT_ID op gets
    // assigned to a non-CUDA backend, eval_cb's routing block below
    // sets the TLS but no CUDA kernel consumes it. The next CUDA op
    // — which may be a totally unrelated non-MoE op — then sees a
    // stale TLS, and any kernel that peeks via
    // ggml_cuda_has_routed_expert_ptrs() (or any path that calls
    // take_) gets a pointer to expert_ptrs that have nothing to do
    // with the current op, leading to near-null GPU faults during
    // decode. Defensively consume here so the slate is clean before
    // we (maybe) set it again for this op.
    ggml_cuda_take_routed_expert_ptrs();
#endif

    // MAD-231: drain pins from the PREVIOUS op's pages now that the GPU
    // has had a full eval-cb-cycle of latency to finish reading them.
    // Conservative: this window is a superset of the actual GPU read
    // (the previous op was dispatched on the compute stream and any
    // subsequent prefetch H2D is enqueued on the same stream → ordered),
    // but the pin set is small (handful of pages per op) so the over-
    // protection is free.
    //
    // Single-threaded: ggml's scheduler dispatches eval_cb on one thread,
    // so no synchronization is needed for s_pinned_pages_prev_op.
    static std::vector<int> s_pinned_pages_prev_op;
    for (int prev_page : s_pinned_pages_prev_op) {
        pager->unpin_page(prev_page);
    }
    s_pinned_pages_prev_op.clear();

    // Diagnostic: detect MUL_MAT_ID ops and check whether their weight
    // source is a consolidated MoE parent. This is the entry point for
    // routing-aware paging (MAD-88 Phase 2 part 2). Currently informational
    // only — without the kernel-side scatter variant the per-expert paging
    // can't replace the consolidated tensor read, so this just counts and
    // reports what would be needed.
    if (t->op == GGML_OP_MUL_MAT_ID) {
        ++g_debug.mmid_ops_seen;
        if (t->src[0] != nullptr) {
            const int weight_page = pager->find_page(ggml_get_name(t->src[0]));
            if (weight_page >= 0) {
                const auto & meta = pager->page_meta(weight_page);
                if (meta.is_consolidated) {
                    ++g_debug.mmid_consolidated;

                    // Count sub-experts of this parent (contiguous insertion
                    // order — see PageCatalog::add_consolidated_experts).
                    int n_subs = 0;
                    for (int i = weight_page + 1; i < pager->n_pages(); ++i) {
                        const auto & sub = pager->page_meta(i);
                        if (!sub.is_sub_expert || sub.parent_page_idx != weight_page) {
                            break;
                        }
                        ++n_subs;
                    }

                    if (g_debug.mmid_consolidated <= 4) {
                        LLAMA_LOG_INFO("[wp::eval_cb] MUL_MAT_ID over consolidated tensor '%s' (parent=%d, %d sub-experts)\n",
                                       ggml_get_name(t->src[0]), weight_page, n_subs);
                    }

#if defined(GGML_USE_HIP)
                    // MAD-88 Phase 2-6: routing-aware paging.
                    //
                    // Read the indices tensor (t->src[2], shape
                    // [n_expert_used, n_tokens, n_seqs]), build the unique
                    // active expert set, ensure() each active sub-expert
                    // page, populate a device-side pointer array, and
                    // hand it to the kernel via the TLS side channel.
                    //
                    // The ensure()d slot pointers are valid for the
                    // duration of this op — the kernel reads them before
                    // we get the next eval_cb invocation, and the wp pool
                    // doesn't evict slots that were just ensure()d (LRU
                    // tracks insertion order).
                    //
                    // Single-buffer device pointer cache: we reuse a
                    // statically-allocated device array sized for
                    // kMaxExperts. cudaMemcpyAsync + kernel launch on the
                    // same stream serialise correctly — no race with the
                    // previous op.
                    struct ggml_tensor * idx_tensor = t->src[2];
                    if (n_subs > 0 && idx_tensor != nullptr) {
                        constexpr int kMaxExperts = 256;
                        if (n_subs > kMaxExperts) {
                            LLAMA_LOG_WARN("[wp::eval_cb] consolidated tensor has %d experts > kMaxExperts=%d, skipping routing\n",
                                           n_subs, kMaxExperts);
                        } else {
                            // Lazy device-buffer init.
                            static const void * * s_dev_expert_ptrs = nullptr;
                            if (s_dev_expert_ptrs == nullptr) {
                                hipError_t alloc_err = hipMalloc(&s_dev_expert_ptrs,
                                                                 kMaxExperts * sizeof(const void *));
                                if (alloc_err != hipSuccess) {
                                    LLAMA_LOG_WARN("[wp::eval_cb] hipMalloc for expert_ptrs failed: %s\n",
                                                   hipGetErrorString(alloc_err));
                                    s_dev_expert_ptrs = nullptr;
                                }
                            }

                            if (s_dev_expert_ptrs != nullptr) {
                                // Pick up the GGML CUDA compute stream so all
                                // host↔device transfers below are stream-ordered
                                // with the kernels that produce / consume them.
                                // GGML creates compute streams with
                                // cudaStreamNonBlocking (common.cuh:1439), so a
                                // synchronous hipMemcpy on the default stream
                                // does NOT serialize with them — that race window
                                // produced near-null GPU faults during MoE
                                // prefill (MAD-230). nullptr falls back to
                                // legacy sync behaviour for safety, but a
                                // properly-initialised cuda backend always
                                // provides the stream.
                                hipStream_t wp_stream =
                                    (hipStream_t) ggml_cuda_get_wp_compute_stream();

                                // MAD-230 follow-up: periodic compute-stream
                                // drain so the GPU's command processor can
                                // schedule graphics-ring frames between MoE
                                // bursts. Without this, decode on a fast MoE
                                // model (gpt-oss-20b at 70-80 t/s → ~5500 MoE
                                // ops/sec) saturates the compute ring densely
                                // enough that the graphics ring times out on
                                // a display-attached GPU (MODE1 reset → system
                                // restart). Yielding every N MoE ops creates
                                // frame-rate-equivalent windows for the
                                // compositor without significantly impacting
                                // throughput (stream sync is bounded to this
                                // stream, not device-wide). Tunable via
                                // WP_YIELD_EVERY_N_OPS; 0 disables.
                                static const int s_yield_every = []() {
                                    const char * env = std::getenv("WP_YIELD_EVERY_N_OPS");
                                    if (env == nullptr) return 32;
                                    char * end = nullptr;
                                    long v = std::strtol(env, &end, 10);
                                    return (end != env && v >= 0) ? (int) v : 32;
                                }();
                                static thread_local int s_yield_ctr = 0;
                                if (wp_stream != nullptr && s_yield_every > 0) {
                                    if (++s_yield_ctr >= s_yield_every) {
                                        s_yield_ctr = 0;
                                        hipStreamSynchronize(wp_stream);
                                    }
                                }

                                // Read indices to host. Stream-ordered async
                                // D2H + stream sync waits for the router-output
                                // kernel that produced idx_tensor->data without
                                // a device-wide stall. (sync hipMemcpy on the
                                // default stream is NOT a sufficient barrier
                                // against a non-blocking compute stream — see
                                // above.)
                                const int64_t n_indices = ggml_nelements(idx_tensor);
                                std::vector<int32_t> host_indices((size_t) n_indices, 0);

                                hipError_t mc_err;
                                if (wp_stream != nullptr) {
                                    mc_err = hipMemcpyAsync(host_indices.data(),
                                                            idx_tensor->data,
                                                            (size_t) n_indices * sizeof(int32_t),
                                                            hipMemcpyDeviceToHost,
                                                            wp_stream);
                                    if (mc_err == hipSuccess) {
                                        mc_err = hipStreamSynchronize(wp_stream);
                                    }
                                } else {
                                    mc_err = hipMemcpy(host_indices.data(),
                                                       idx_tensor->data,
                                                       (size_t) n_indices * sizeof(int32_t),
                                                       hipMemcpyDeviceToHost);
                                }
                                if (mc_err == hipSuccess) {
                                    // Build active expert set first so we can
                                    // pipeline the page-ins. (MAD-88 Phase 9c.)
                                    std::vector<const void *> host_ptrs((size_t) n_subs, nullptr);
                                    std::unordered_set<int> active;
                                    for (int32_t idx : host_indices) {
                                        if (idx < 0 || idx >= n_subs) continue;
                                        active.insert((int) idx);
                                    }

                                    // Pass 1: fire async prefetch for every
                                    // active expert. With io_uring (depth 4),
                                    // multiple preads can be in flight at
                                    // once. ensure() in pass 2 will see slots
                                    // already reserved + in-flight, and
                                    // wait_for completion instead of doing
                                    // a fresh sync pread.
                                    //
                                    // MAD-235: prefer the atomic batch path
                                    // (one io_uring_submit syscall for the
                                    // whole expert set). WP_BATCH_PREFETCH=0
                                    // reverts to per-expert loop for A/B
                                    // measurement / regression rollback.
                                    static int s_batch_prefetch_env = -1;
                                    if (s_batch_prefetch_env < 0) {
                                        const char * env = std::getenv("WP_BATCH_PREFETCH");
                                        s_batch_prefetch_env = (env != nullptr && env[0] == '0') ? 0 : 1;
                                    }
                                    bool batch_ok = false;
                                    if (s_batch_prefetch_env != 0) {
                                        std::vector<int> active_pages;
                                        active_pages.reserve(active.size());
                                        for (int e : active) {
                                            active_pages.push_back(weight_page + 1 + e);
                                        }
                                        batch_ok = pager->prefetch_pages_batch(active_pages);
                                    }
                                    if (!batch_ok) {
                                        // Either batch was disabled OR scheduler
                                        // refused (queue full / capacity tight).
                                        // Fall back to per-expert prefetch — best-
                                        // effort, ignores individual failures.
                                        for (int e : active) {
                                            const int sub_page_idx = weight_page + 1 + e;
                                            pager->prefetch_page(sub_page_idx);
                                        }
                                    }
                                    // Drive the prefetch state machine forward
                                    // so submitted reads get out the door
                                    // before we start blocking on completions.
                                    pager->tick();

                                    // Pass 2: ensure() each, harvesting slot
                                    // pointers. For pages whose prefetch is
                                    // already reaped, this is a fast cache
                                    // hit; for in-flight pages it waits on
                                    // the async completion; for unloaded
                                    // pages it falls back to sync.
                                    int n_ensures = 0;
                                    void * first_active_slot = nullptr;
                                    for (int e : active) {
                                        const int sub_page_idx = weight_page + 1 + e;
                                        void * slot = pager->ensure(sub_page_idx);
                                        if (slot != nullptr) {
                                            host_ptrs[(size_t) e] = slot;
                                            if (first_active_slot == nullptr) {
                                                first_active_slot = slot;
                                            }
                                            ++n_ensures;
                                            // MAD-231: pin the slot so a later prefetch
                                            // alloc_slot in this same eval_cb can't evict
                                            // it. Unpinned in the NEXT eval_cb (above).
                                            pager->pin_page(sub_page_idx);
                                            s_pinned_pages_prev_op.push_back(sub_page_idx);
                                        }
                                    }
                                    // Safety: fill INACTIVE expert slots with a non-null
                                    // sentinel (first active slot) so a kernel that reads
                                    // expert_ptrs[inactive_idx] gets a valid (wrong) pointer
                                    // instead of NULL-faulting. If the kernel correctly only
                                    // reads active indices this is dead memory; if it doesn't
                                    // we'll see wrong logits but no fault, which is recoverable.
                                    if (first_active_slot != nullptr) {
                                        for (size_t i = 0; i < host_ptrs.size(); ++i) {
                                            if (host_ptrs[i] == nullptr) {
                                                host_ptrs[i] = first_active_slot;
                                            }
                                        }
                                    }

                                    // Write the per-expert pointer array via
                                    // stream-ordered async H2D. s_dev_expert_ptrs
                                    // is a STATIC buffer shared by every MoE op
                                    // in this forward pass, but stream ordering
                                    // makes that safe: on this same compute
                                    // stream, the previous MMQ kernel's read
                                    // completes before this memcpy executes,
                                    // and the next MMQ kernel's read happens
                                    // after this memcpy completes. No device-
                                    // wide sync (previously hipDeviceSynchronize)
                                    // and no torn-pointer race (the previous
                                    // sync-on-default-stream design had one;
                                    // MAD-230). For pageable host memory,
                                    // hipMemcpyAsync H2D does an internal
                                    // staging copy before returning, so
                                    // host_ptrs going out of scope at end of
                                    // eval_cb is safe.
                                    if (wp_stream != nullptr) {
                                        hipMemcpyAsync(s_dev_expert_ptrs,
                                                       host_ptrs.data(),
                                                       (size_t) n_subs * sizeof(const void *),
                                                       hipMemcpyHostToDevice,
                                                       wp_stream);
                                    } else {
                                        // Legacy fallback if the cuda backend
                                        // didn't publish a stream (shouldn't
                                        // happen with a properly initialised
                                        // GGML CUDA backend).
                                        hipDeviceSynchronize();
                                        hipMemcpy(s_dev_expert_ptrs,
                                                  host_ptrs.data(),
                                                  (size_t) n_subs * sizeof(const void *),
                                                  hipMemcpyHostToDevice);
                                    }
                                    ggml_cuda_set_routed_expert_ptrs(s_dev_expert_ptrs);

                                    if (g_debug.mmid_consolidated <= 4) {
                                        LLAMA_LOG_INFO("[wp::eval_cb] routed: %d/%zu unique active experts ensured\n",
                                                       n_ensures, active.size());
                                    }

                                    // Patch the consolidated parent's buffer so ggml-cuda's
                                    // mul_mat_id assertions don't NULL-deref. The kernel
                                    // never reads parent->data when routed_expert_ptrs is
                                    // set, but ggml_cuda_mul_mat_id dereferences
                                    // src0->buffer->buft on entry to check for split
                                    // buffers (ggml-cuda.cu:2667). init_weight_pager left
                                    // src0->buffer = nullptr for paged tensors, so without
                                    // this patch we NULL-deref before reaching the
                                    // routing-aware dispatcher gate.
                                    ggml_backend_buffer_t pool_buf = pager->pool_buf();
                                    if (t->src[0]->buffer == nullptr && pool_buf != nullptr) {
                                        t->src[0]->buffer = pool_buf;
                                    }

                                    // MAD-88 Phase 9a: same-layer prefetch.
                                    // gate / up / down at one MoE layer all share
                                    // the same active expert set (the router runs
                                    // once per layer, before any of them). When we
                                    // process the first MUL_MAT_ID over a sister
                                    // parent, fire async prefetches for the OTHER
                                    // sister parents' same expert sub-pages so by
                                    // the time their MUL_MAT_IDs fire they're
                                    // either cache hits or already in flight.
                                    //
                                    // Sister discovery is O(catalog) on first
                                    // hit per parent; cached after that.
                                    static std::unordered_map<int, std::vector<int>> s_sister_cache;
                                    auto sister_it = s_sister_cache.find(weight_page);
                                    if (sister_it == s_sister_cache.end()) {
                                        std::vector<int> sisters;
                                        const int my_block = meta.block_idx;
                                        for (int i = 0; i < pager->n_pages(); ++i) {
                                            if (i == weight_page) continue;
                                            const auto & p = pager->page_meta(i);
                                            if (!p.is_consolidated) continue;
                                            if (p.block_idx != my_block) continue;
                                            sisters.push_back(i);
                                        }
                                        sister_it = s_sister_cache.emplace(weight_page, std::move(sisters)).first;
                                    }
                                    for (int sister_parent : sister_it->second) {
                                        for (int e : active) {
                                            if (e < 0 || e >= n_subs) continue;
                                            const int sister_sub = sister_parent + 1 + e;
                                            pager->prefetch_page(sister_sub);
                                        }
                                    }

                                    // MAD-233 — cross-layer N+K MoE expert prefetch.
                                    //
                                    // Project the CURRENT layer's active expert set forward to
                                    // layers [block+1, block+K]. We can't know the true active
                                    // set for future layers (the router decides per layer), but
                                    // empirical Qwen3-MoE has 40-50% expert reuse across
                                    // consecutive tokens — locality enough that prefetching
                                    // the same indices means most ensure()s for the next layer
                                    // are cache hits, and the rest at least overlap NVMe I/O
                                    // with this layer's compute (5 ms per MoE layer).
                                    //
                                    // Safety: MAD-231 slot pinning protects this layer's
                                    // in-flight slots from eviction by the prefetch alloc_slot.
                                    // Without that, the prefetch could evict a slot the current
                                    // op is still reading. With MAD-231 the worst case is
                                    // "prefetch couldn't find an unpinned slot, batch refused,
                                    // we don't get the win" — never corruption.
                                    //
                                    // WP_NEXT_LAYER_PREFETCH_K env tunes lookahead (default 1,
                                    // 0 disables). Cache the parent list per source-parent for
                                    // O(1) reuse across all 3 MUL_MAT_IDs of a layer.
                                    static int s_next_k = -1;
                                    if (s_next_k < 0) {
                                        const char * env = std::getenv("WP_NEXT_LAYER_PREFETCH_K");
                                        s_next_k = env ? std::atoi(env) : 1;
                                        if (s_next_k < 0) s_next_k = 0;
                                        LLAMA_LOG_INFO("[wp::eval_cb] WP_NEXT_LAYER_PREFETCH_K=%d "
                                                       "(0=disabled, requires MAD-231 pinning)\n",
                                                       s_next_k);
                                    }
                                    if (s_next_k > 0) {
                                        // Cache: weight_page -> vector of consolidated parent
                                        // indices for blocks [my_block+1, my_block+s_next_k].
                                        static std::unordered_map<int, std::vector<int>> s_next_layer_parents_cache;
                                        auto next_it = s_next_layer_parents_cache.find(weight_page);
                                        if (next_it == s_next_layer_parents_cache.end()) {
                                            std::vector<int> next_parents;
                                            const int my_block = meta.block_idx;
                                            for (int i = 0; i < pager->n_pages(); ++i) {
                                                const auto & p = pager->page_meta(i);
                                                if (!p.is_consolidated) continue;
                                                if (p.block_idx <= my_block) continue;
                                                if (p.block_idx >  my_block + s_next_k) continue;
                                                next_parents.push_back(i);
                                            }
                                            next_it = s_next_layer_parents_cache.emplace(
                                                weight_page, std::move(next_parents)).first;
                                        }

                                        // Build a single batched prefetch covering every
                                        // (future_parent, active_expert) pair. The batch path
                                        // (MAD-235) collapses to one io_uring_submit syscall
                                        // and either queues the whole set atomically or refuses
                                        // — refusal falls back to per-page prefetch which is
                                        // best-effort.
                                        if (!next_it->second.empty()) {
                                            std::vector<int> future_pages;
                                            future_pages.reserve(next_it->second.size() * active.size());
                                            for (int future_parent : next_it->second) {
                                                for (int e : active) {
                                                    if (e < 0 || e >= n_subs) continue;
                                                    future_pages.push_back(future_parent + 1 + e);
                                                }
                                            }
                                            if (!future_pages.empty()) {
                                                pager->mark_cross_layer_prefetch_candidates(future_pages);
                                                const bool batch_ok = pager->prefetch_pages_batch(future_pages);
                                                if (!batch_ok) {
                                                    // Per-page fallback (best-effort, ignores
                                                    // individual failures — eviction will sort
                                                    // it out and ensure() on the next layer
                                                    // falls back to sync if nothing landed).
                                                    for (int fp : future_pages) {
                                                        pager->prefetch_page(fp);
                                                    }
                                                }
                                                if (g_debug.mmid_consolidated <= 4) {
                                                    LLAMA_LOG_INFO("[wp::eval_cb] cross-layer prefetch: %zu pages "
                                                                   "(parents=%zu, K=%d, batch=%d)\n",
                                                                   future_pages.size(), next_it->second.size(),
                                                                   s_next_k, (int) batch_ok);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
#endif // GGML_USE_HIP
                }
            }
        }
    }

    // Step 1: walk t->src[] and collect distinct paged-weight page indices.
    // A source counts as paged if either its own name or its view_src's
    // name is in the catalog. View tensors fall in the second category;
    // gallocr initialises their data to (char*)1 + view_offs (a sentinel
    // address), so we MUST overwrite their data before the op runs — bug
    // B-P1 in docs/dev/memory-tier-bug-catalog.md.
    int  page_indices[GGML_MAX_SRC];
    int  n_page_indices = 0;
    int  highest_page   = -1;

    // DIAGNOSTIC (MAD-230 MoE near-null fault hunt): catch any src with
    // a sentinel-shaped data pointer (gallocr's (char*)1 + view_offs).
    // If a view of a paged consolidated parent reaches this op without
    // being patched, its data lives at 0x1 + view_offs → near-null fault
    // on kernel read. Log it BEFORE the standard skip path so we see it.
    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = t->src[i];
        if (src == nullptr) break;
        const uintptr_t data_addr = (uintptr_t) src->data;
        if (data_addr != 0 && data_addr < 0x10000000ULL) {
            const char * vsrc_name = src->view_src ? ggml_get_name(src->view_src) : "(none)";
            int psrc_page  = pager->find_page(ggml_get_name(src));
            int psrc_vpage = src->view_src ? pager->find_page(vsrc_name) : -1;
            LLAMA_LOG_WARN("[wp::eval_cb][LOW_ADDR_DATA] op=%s op_name=\"%s\" src[%d]=\"%s\" "
                           "data=0x%lx view_offs=%zu view_src=\"%s\" "
                           "src_page=%d view_src_page=%d (consolidated? src=%d vsrc=%d)\n",
                           ggml_op_name(t->op), ggml_get_name(t), i, ggml_get_name(src),
                           (unsigned long) data_addr, src->view_offs, vsrc_name,
                           psrc_page, psrc_vpage,
                           (psrc_page  >= 0 ? (int) pager->page_meta(psrc_page).is_consolidated  : -1),
                           (psrc_vpage >= 0 ? (int) pager->page_meta(psrc_vpage).is_consolidated : -1));
            std::fflush(stderr);
        }
    }

    for (int i = 0; i < GGML_MAX_SRC; ++i) {
        struct ggml_tensor * src = t->src[i];
        if (src == nullptr) break;

        int page_idx = pager->find_page(ggml_get_name(src));
        if (page_idx < 0 && src->view_src != nullptr) {
            page_idx = pager->find_page(ggml_get_name(src->view_src));
        }
        if (page_idx < 0) continue;

        // MAD-88: skip consolidated MoE parents in the standard ensure()
        // path. The parent is metadata-only (no slot allocated, full
        // tensor size exceeds per-expert staging buffer). For MUL_MAT_ID
        // the routing-aware block above ensures the active sub-experts.
        // For any other op that references the consolidated tensor by
        // name (rare in practice — only views from the model loader
        // itself), the kernel reads from src->data which still points
        // at the placeholder. That's a known limitation; if it bites a
        // real model we'd need to ensure ALL sub-experts for that op.
        if (pager->page_meta(page_idx).is_consolidated) {
            continue;
        }

        // Dedupe: two views of the same weight in one op resolve to the
        // same page; we only need to ensure() once.
        bool already = false;
        for (int j = 0; j < n_page_indices; ++j) {
            if (page_indices[j] == page_idx) { already = true; break; }
        }
        if (!already) {
            page_indices[n_page_indices++] = page_idx;
            if (page_idx > highest_page) highest_page = page_idx;
        }
    }

    ++g_debug.ops_seen;
    if (n_page_indices == 0) {
        // Diagnostic: did any src LOOK LIKE a weight tensor that we should have found?
        // Helps surface name-mismatch / catalog-miss bugs.
        bool had_weight_looking_src = false;
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            struct ggml_tensor * s = t->src[i];
            if (s == nullptr) break;
            const char * nm = ggml_get_name(s);
            if (nm && std::strstr(nm, "weight") != nullptr) {
                had_weight_looking_src = true;
                break;
            }
        }
        if (had_weight_looking_src) {
            if (g_debug.ops_no_paged_with_weight_src < 16) {
                std::string srcs;
                for (int i = 0; i < GGML_MAX_SRC; ++i) {
                    if (t->src[i] == nullptr) break;
                    if (i > 0) srcs += ", ";
                    char buf[128];
                    std::snprintf(buf, sizeof(buf), "%s@%p(buf=%p)",
                                  ggml_get_name(t->src[i]), t->src[i]->data,
                                  (void*)t->src[i]->buffer);
                    srcs += buf;
                }
                LLAMA_LOG_WARN("[wp::eval_cb][MISS] op=%s name=\"%s\" srcs=[%s]\n",
                               ggml_op_name(t->op), ggml_get_name(t), srcs.c_str());
            }
            ++g_debug.ops_no_paged_with_weight_src;
        }
        // Periodic summary so we know eval_cb is alive even when no patches happen
        if ((g_debug.ops_seen % 500) == 0) {
            LLAMA_LOG_WARN("[wp::eval_cb][SUM] ops_seen=%d ops_with_pages=%d patches=%d miss_w=%d fails=%d\n",
                           g_debug.ops_seen, g_debug.ops_with_pages,
                           g_debug.patches_total, g_debug.ops_no_paged_with_weight_src,
                           g_debug.ensures_failed);
        }
        return true;
    }
    ++g_debug.ops_with_pages;

    // MAD-232: posix_fadvise(WILLNEED) for the next K layers' paged tensors.
    // Warms NVMe→page-cache while THIS layer's compute runs, so by the time
    // the eval-cb reaches layer N+1's ensure() the bytes are already in
    // page cache (memcpy at ~10 GB/s vs cold pread at ~500 MB/s QD=1).
    //
    // Idempotent: the kernel deduplicates overlapping advise hints, and our
    // sentinel `s_last_advised_block` skips re-issuing for the same boundary.
    //
    // Wrap detection: when block_idx drops below `s_last_advised_block` we
    // assume a new forward pass began (the scheduler revisits block 0). Reset
    // the sentinel so we re-advise from the current block. This handles both
    // decode steps and any future op-reordering the scheduler does.
    //
    // RAM cost: each advised range is `size` bytes of page-cache pressure.
    // Tune via WP_FADVISE_LOOKAHEAD (default 2, 0 disables).
    {
        static int s_advise_k          = -2;  // -2 = unread env
        static int s_last_advised_block = -1;
        if (s_advise_k == -2) {
            const char * env = std::getenv("WP_FADVISE_LOOKAHEAD");
            s_advise_k = env ? std::atoi(env) : 2;
            if (s_advise_k < 0) s_advise_k = 0;
            LLAMA_LOG_INFO("[wp::eval_cb] WP_FADVISE_LOOKAHEAD=%d (0=disabled)\n", s_advise_k);
        }
        if (s_advise_k > 0) {
            int min_block = std::numeric_limits<int>::max();
            for (int j = 0; j < n_page_indices; ++j) {
                const int b = pager->page_meta(page_indices[j]).block_idx;
                if (b >= 0 && b < min_block) min_block = b;
            }
            if (min_block != std::numeric_limits<int>::max()) {
                if (min_block < s_last_advised_block) {
                    // New forward pass — sentinel wraps. Re-advise from current.
                    s_last_advised_block = -1;
                }
                if (min_block > s_last_advised_block) {
                    pager->advise_layer_lookahead(min_block, s_advise_k);
                    s_last_advised_block = min_block;
                }
            }
        }
    }

    // Step 2: page each one in (waiting on prefetch if in flight, sync
    // fallback otherwise) and patch the matching src tensors.
    ggml_backend_buffer_t pool_buf = pager->pool_buf();
    int  patches_this_op = 0;
    int  views_this_op   = 0;

    for (int j = 0; j < n_page_indices; ++j) {
        const int    page_idx = page_indices[j];
        void       * vram     = pager->ensure(page_idx);
        if (vram == nullptr) {
            ++g_debug.ensures_failed;
            // ensure() logs the failure; we can't make progress on this op.
            // Returning false from the callback would abort scheduling; we
            // continue and let the kernel fail with whatever pointer is in
            // src->data. This matches the previous pager's behaviour and
            // keeps debugging signal local to the failing op.
            continue;
        }
        // MAD-231: pin the slot so a subsequent prefetch alloc_slot in
        // tick() (or in a later op's pre-cb) cannot evict it while the
        // GPU is still reading from it. Unpinned at the top of the NEXT
        // eval_cb invocation.
        pager->pin_page(page_idx);
        s_pinned_pages_prev_op.push_back(page_idx);

        const std::string & page_name = pager->page_meta(page_idx).tensor_name;

        // Patch every src whose direct name OR view_src's name matches
        // this page.
        for (int i = 0; i < GGML_MAX_SRC; ++i) {
            struct ggml_tensor * src = t->src[i];
            if (src == nullptr) break;

            if (std::strcmp(ggml_get_name(src), page_name.c_str()) == 0) {
                src->data   = vram;
                src->buffer = pool_buf;
                ++patches_this_op;
                continue;
            }
            if (src->view_src != nullptr &&
                std::strcmp(ggml_get_name(src->view_src), page_name.c_str()) == 0) {
                // Sentinel overwrite (B-P1): gallocr left
                // src->data = (char*)1 + src->view_offs.
                src->data   = (char *) vram + src->view_offs;
                src->buffer = pool_buf;
                ++patches_this_op;
                ++views_this_op;
            }
        }
    }

    g_debug.patches_total += patches_this_op;
    g_debug.views_patched += views_this_op;
    // Always-on diagnostics for the first N paged ops so we can see what
    // the eval cb is doing without relying on env var. Use WARN level so
    // llama-cli's default log filter doesn't suppress it.
    if (g_debug.ops_with_pages <= DebugState::kVerboseLimit) {
        LLAMA_LOG_WARN("[wp::eval_cb][%d]: op=%s op_name=\"%s\" n_pages=%d patches=%d views=%d (cum: patches=%d views=%d fails=%d miss_w=%d)\n",
                        g_debug.ops_with_pages, ggml_op_name(t->op),
                        ggml_get_name(t),
                        n_page_indices, patches_this_op, views_this_op,
                        g_debug.patches_total, g_debug.views_patched, g_debug.ensures_failed,
                        g_debug.ops_no_paged_with_weight_src);
    } else if (g_debug.ops_with_pages == DebugState::kVerboseLimit + 1) {
        LLAMA_LOG_WARN("[wp::eval_cb] suppressing further per-op logs after first %d paged ops\n",
                       DebugState::kVerboseLimit);
    }

    // Step 3: drive the prefetch pipeline forward.
    //
    // We deliberately do NOT submit a next-page prefetch here — that
    // calls pool_.alloc_slot() which can evict an LRU slot, including
    // one we just patched src->data into for this op. The op would then
    // read corrupted VRAM. A correct implementation needs slot refcounts
    // (pin while an op references the slot) or a "current op set" the
    // pool refuses to evict. Phase 1e work item.
    pager->tick();
    (void) highest_page;  // suppress unused warning; will be used once eviction-safe prefetch lands

    return true;
}

}  // namespace wp
