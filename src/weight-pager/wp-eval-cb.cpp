#include "wp-eval-cb.h"
#include "wp-pager.h"

#include "ggml.h"

#include <cstring>
#include <vector>

namespace wp {

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

    if (n_page_indices == 0) return true;

    // Step 2: page each one in (waiting on prefetch if in flight, sync
    // fallback otherwise) and patch the matching src tensors.
    ggml_backend_buffer_t pool_buf = pager->pool_buf();

    for (int j = 0; j < n_page_indices; ++j) {
        const int    page_idx = page_indices[j];
        void       * vram     = pager->ensure(page_idx);
        if (vram == nullptr) {
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
                continue;
            }
            if (src->view_src != nullptr &&
                std::strcmp(ggml_get_name(src->view_src), page_name.c_str()) == 0) {
                // Sentinel overwrite (B-P1): gallocr left
                // src->data = (char*)1 + src->view_offs.
                src->data   = (char *) vram + src->view_offs;
                src->buffer = pool_buf;
            }
        }
    }

    // Step 3: drive the prefetch pipeline forward and submit a hint for
    // the next page after the highest one we just used. Pages are
    // catalog-ordered (typically by layer), so "next page after highest"
    // is a reasonable heuristic for sequential weight access through a
    // forward pass. Phase 1d may swap this for a more sophisticated walk
    // once we have profiling data.
    pager->tick();
    if (highest_page >= 0 && highest_page + 1 < pager->n_pages()) {
        pager->prefetch_page(highest_page + 1);
    }

    return true;
}

}  // namespace wp
