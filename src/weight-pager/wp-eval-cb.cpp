#include "wp-eval-cb.h"
#include "wp-pager.h"

#include "ggml.h"
#include "llama-impl.h"  // LLAMA_LOG_*

#include <cstdlib>       // getenv
#include <cstring>
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
        LLAMA_LOG_ERROR("[DIAG] wp::eval_cb[%d]: op=%s op_name=\"%s\" n_pages=%d patches=%d views=%d (cum: patches=%d views=%d fails=%d)\n",
                        g_debug.ops_with_pages, ggml_op_name(t->op),
                        ggml_get_name(t),
                        n_page_indices, patches_this_op, views_this_op,
                        g_debug.patches_total, g_debug.views_patched, g_debug.ensures_failed);
    } else if (g_debug.ops_with_pages == DebugState::kVerboseLimit + 1) {
        LLAMA_LOG_ERROR("[DIAG] wp::eval_cb: suppressing further per-op logs after first %d paged ops\n",
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
