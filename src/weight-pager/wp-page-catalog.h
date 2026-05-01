#pragma once

// PageCatalog — metadata for tensors that the weight pager manages.
//
// Pure bookkeeping. No I/O, no GPU. One entry per tensor that participates
// in paging. Insertion order is preserved so prefetch heuristics keyed on
// "next page" are stable across runs.
//
// Catalog is populated once at model load by the model loader integration
// (Phase 1d), then queried (read-mostly) thereafter. Mutation after init
// is not supported; the catalog is a snapshot of the GGUF layout.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace wp {

struct PageMeta {
    std::string tensor_name;   // ggml tensor name, e.g. "blk.0.attn_q.weight"
    uint16_t    file_idx;      // index into the model's file array (split GGUFs)
    uint64_t    file_offset;   // absolute byte offset within that file
    size_t      size;          // tensor payload size in bytes
};

// Insertion-ordered, read-mostly map: name -> PageMeta.
// Page indices are stable for the lifetime of the catalog.
class PageCatalog {
public:
    PageCatalog() = default;

    // Add a page. Must not be called after build() / once querying begins.
    // Returns the assigned page index.
    int add(const std::string & name, uint16_t file_idx,
            uint64_t file_offset, size_t size);

    // Number of registered pages.
    int size() const { return (int) pages_.size(); }

    // Lookup by name. Returns -1 if not present.
    int find(const std::string & name) const;

    // Index access. Caller must ensure 0 <= idx < size().
    const PageMeta & at(int idx) const { return pages_[idx]; }

    // Maximum payload size across all pages — used by PoolAllocator to size
    // its slot stride.
    size_t max_page_size() const { return max_size_; }

    // Clear all entries. Intended for tests / re-init paths.
    void clear();

private:
    std::vector<PageMeta>                 pages_;
    std::unordered_map<std::string, int>  name_to_idx_;
    size_t                                max_size_ = 0;
};

}  // namespace wp
