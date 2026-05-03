#include "wp-eval-cb.h"
#include "wp-pager.h"

#include "ggml.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
// Forward decl of the ggml-cuda side channel — the actual symbol lives in
// libggml-hip.so and we link against it. Avoids dragging the full
// ggml-cuda/mmq.cuh into libllama's wp-eval-cb compilation unit.
extern "C++" void ggml_cuda_set_routed_expert_ptrs(const void * const * ptr);
#endif

#include <cstdlib>       // getenv
#include <cstring>
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
    static constexpr int kVerboseLimit = 8;  // log details for first N ops only
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
                                // Read indices to host. hipMemcpy is
                                // synchronous-w.r.t.-host and serialises
                                // against any prior async work on the
                                // source memory's stream — the explicit
                                // hipDeviceSynchronize that used to live
                                // here forced a device-wide stall between
                                // every MoE op (240/token), preventing
                                // any cross-op GPU pipelining. Drop it;
                                // hipMemcpy waits as needed. Phase 9b.
                                const int64_t n_indices = ggml_nelements(idx_tensor);
                                std::vector<int32_t> host_indices((size_t) n_indices, 0);

                                hipError_t mc_err = hipMemcpy(host_indices.data(),
                                                              idx_tensor->data,
                                                              (size_t) n_indices * sizeof(int32_t),
                                                              hipMemcpyDeviceToHost);
                                if (mc_err == hipSuccess) {
                                    // Build active expert set + slot ptrs.
                                    std::vector<const void *> host_ptrs((size_t) n_subs, nullptr);
                                    std::unordered_set<int> active;
                                    int n_ensures = 0;
                                    for (int32_t idx : host_indices) {
                                        if (idx < 0 || idx >= n_subs) continue;
                                        if (!active.insert((int) idx).second) continue;
                                        const int sub_page_idx = weight_page + 1 + idx;
                                        void * slot = pager->ensure(sub_page_idx);
                                        if (slot != nullptr) {
                                            host_ptrs[(size_t) idx] = slot;
                                            ++n_ensures;
                                        }
                                    }

                                    // Async copy host pointer table to
                                    // device. The kernel will read after
                                    // this completes (same stream order).
                                    hipMemcpy(s_dev_expert_ptrs,
                                              host_ptrs.data(),
                                              (size_t) n_subs * sizeof(const void *),
                                              hipMemcpyHostToDevice);
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
    if (n_page_indices == 0) return true;
    ++g_debug.ops_with_pages;

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
        LLAMA_LOG_INFO("[wp::eval_cb][%d]: op=%s op_name=\"%s\" n_pages=%d patches=%d views=%d (cum: patches=%d views=%d fails=%d)\n",
                        g_debug.ops_with_pages, ggml_op_name(t->op),
                        ggml_get_name(t),
                        n_page_indices, patches_this_op, views_this_op,
                        g_debug.patches_total, g_debug.views_patched, g_debug.ensures_failed);
    } else if (g_debug.ops_with_pages == DebugState::kVerboseLimit + 1) {
        LLAMA_LOG_INFO("[wp::eval_cb] suppressing further per-op logs after first %d paged ops\n",
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
