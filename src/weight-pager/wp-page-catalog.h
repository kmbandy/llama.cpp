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

// FFN role bitmask values for PageMeta::expert_role_mask. A consolidated
// MoE tensor packs all experts for one role; a per-expert tensor names a
// single (role, expert) pair. Either way the role is single-bit per tensor.
constexpr uint8_t ROLE_UP   = 1u << 0;
constexpr uint8_t ROLE_GATE = 1u << 1;
constexpr uint8_t ROLE_DOWN = 1u << 2;

struct PageMeta {
    std::string tensor_name;   // ggml tensor name, e.g. "blk.0.attn_q.weight"
    uint16_t    file_idx;      // index into the model's file array (split GGUFs)
    uint64_t    file_offset;   // absolute byte offset within that file
    size_t      size;          // tensor payload size in bytes

    // Layer / MoE classification (parsed from tensor_name in PageCatalog::add).
    // Useful for MoE-aware prefetch and eviction policy. See MAD-88.
    int16_t  block_idx        = -1;     // 0..n_layer-1, -1 if non-block tensor
    int16_t  expert_idx       = -1;     // 0..n_expert-1; -1 for consolidated or non-expert
    uint8_t  expert_role_mask = 0;      // ROLE_UP / ROLE_GATE / ROLE_DOWN
    bool     is_expert        = false;  // true for any MoE expert weight tensor
    bool     is_consolidated  = false;  // true if one tensor holds all experts of a role
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

    // True if any page is an MoE expert tensor — i.e. the model is sparse
    // MoE and downstream policy can use routing-aware prefetch / eviction.
    bool has_experts() const { return n_expert_pages_ > 0; }

    // Number of expert pages registered.
    int  n_expert_pages() const { return n_expert_pages_; }

    // All page indices for a given layer block (0..n_layer-1). Empty if no
    // pages match. Intended for layer-level prefetch heuristics.
    std::vector<int> pages_for_block(int block_idx) const;

    // All page indices for a (block, expert) pair. For consolidated MoE
    // tensors expert_idx is -1; pass -1 to match those. Intended for
    // routing-aware prefetch driven by GGML_OP_MUL_MAT_ID gating output.
    std::vector<int> pages_for_expert(int block_idx, int expert_idx) const;

    // Clear all entries. Intended for tests / re-init paths.
    void clear();

private:
    std::vector<PageMeta>                 pages_;
    std::unordered_map<std::string, int>  name_to_idx_;
    size_t                                max_size_ = 0;
    int                                   n_expert_pages_ = 0;
};

}  // namespace wp
