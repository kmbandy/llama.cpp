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
#include <map>
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
    int16_t  expert_idx       = -1;     // 0..n_expert-1; -1 for consolidated parent or non-expert
    uint8_t  expert_role_mask = 0;      // ROLE_UP / ROLE_GATE / ROLE_DOWN
    bool     is_expert        = false;  // true for any MoE expert weight tensor
    bool     is_consolidated  = false;  // true if this entry holds all experts of a role
                                        // (parent meta — has children but no own slot)
    bool     is_sub_expert    = false;  // true if this entry is a synthesized sub-page
                                        // (one expert of a consolidated parent)
    int      parent_page_idx  = -1;     // for sub-experts: index of consolidated parent

    // MAD-236 — always-resident pin. When true, this entry represents a
    // tensor whose bytes live in caller-owned VRAM (regular ggml-allocated
    // buffer for e.g. token_embd, output_norm, router weights). The pager
    // does NOT allocate a pool slot for it, does NOT read from the file —
    // ensure(page_idx) just returns `resident_ptr`. Useful for tracking
    // total VRAM (paged + resident) in one telemetry view and for letting
    // future mixed-mode workloads (attn-pinned + FFN-paged) be expressed
    // through the same lookup API.
    //
    // NOT to be confused with PoolAllocator's slot pin_count (MAD-231).
    // That's a refcount on slot-level eviction protection; this is a
    // catalog-level flag that the page has no slot at all.
    bool     is_pinned        = false;
    void *   resident_ptr     = nullptr;  // device ptr; ignored unless is_pinned
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

    // Add a consolidated MoE expert tensor as a parent + N sub-page entries
    // (one per expert). The parent has the original tensor name; sub-pages
    // have synthetic names "<name>#expert.<E>" so the eval-callback's
    // name-based lookup can resolve a specific (block, expert) directly.
    //
    // The parent's PageMeta has is_consolidated=true with no slot allocated
    // by the pool — it's pure metadata. Each sub-page has is_sub_expert=true,
    // parent_page_idx pointing at the parent, and a per-expert offset/size
    // (file_offset + e * (size / n_experts), size / n_experts).
    //
    // Returns the page index of the FIRST sub-expert. The parent is at
    // (first_sub - 1) since insertion is in-order. Subsequent experts are
    // at first_sub + e for 0 <= e < n_experts.
    int add_consolidated_experts(const std::string & name, uint16_t file_idx,
                                 uint64_t file_offset, size_t total_size,
                                 int n_experts);

    // MAD-236 — register a tensor whose bytes already live in caller-owned
    // VRAM (e.g. token_embd from the model loader's regular ggml buffer).
    // The pager does NOT allocate a pool slot or read from disk for these
    // entries — ensure(page_idx) just returns `device_ptr`. Useful for
    // mixed paged/resident workloads and for unified telemetry.
    //
    // file_idx / file_offset are unused for pinned entries; defaults of
    // (0, 0) are stored so the PageMeta layout stays uniform. `size` is
    // tracked for the telemetry counters.
    //
    // Returns the assigned page index.
    int add_pinned(const std::string & name, void * device_ptr, size_t bytes);

    // Number of registered pages.
    int size() const { return (int) pages_.size(); }

    // Lookup by name. Returns -1 if not present.
    int find(const std::string & name) const;

    enum class RemapStatus {
        Ok,
        NotFound,      // no such page in this catalog
        NotPageable,   // pinned or a consolidated parent — never reads a file
        SizeMismatch,  // blob disagrees with the model's tensor geometry
    };

    // Repoint an existing page at a different (file_idx, file_offset) without
    // changing anything else about it. This is how wp-repack's expert-major
    // blobs are adopted: the catalog is built from the source GGUFs as usual,
    // then each repacked expert page is redirected into its blob. Everything
    // downstream (pool, transport, prefetch, eval-cb) only ever consumes the
    // (file_idx, file_offset, size) triple, so nothing else needs to know.
    //
    // Must be called BEFORE init() — the fd table is handed to the pager
    // there, and `file_idx` indexes into it.
    RemapStatus remap_source(const std::string & name, uint16_t file_idx,
                             uint64_t file_offset, size_t size);

    // Index access. Caller must ensure 0 <= idx < size().
    const PageMeta & at(int idx) const { return pages_[idx]; }

    // Maximum payload size across all pages — used by PoolAllocator to size
    // its slot stride.
    size_t max_page_size() const { return max_size_; }

    // MAD-420 — page-size histogram for size-class slot pre-carving.
    //
    // Returns size-in-bytes -> number-of-pages for the SLOTTABLE set only:
    // pages that can occupy a pool slot, i.e. is_expert && !is_pinned &&
    // !is_consolidated. Pinned pages live in caller-owned VRAM and never
    // touch the pool; consolidated parents are pure metadata (their
    // sub-expert children are the slottable units). Non-expert dense pages
    // are excluded too — they are either pinned or routed through the
    // resident-dense path, never paged.
    //
    // The keys are RAW per-page payload sizes (un-aligned). The pool's
    // pre-carve solver aligns each key up to slot_alignment_ to get the
    // actual class stride, so the histogram is independent of the device
    // buffer-type alignment that the pool settles at init time.
    //
    // Empty when the model has no slottable expert pages (dense model, or a
    // pager populated only with pinned entries). Callers must handle that.
    std::map<size_t, int> page_size_histogram() const;

    // Per-size, per-block slottable page counts: size -> (block_idx -> count).
    // Same page filter as page_size_histogram (expert, not pinned, not the
    // consolidated parent); keys are likewise RAW un-aligned payload sizes.
    //
    // This is what the pool's pre-carve solver needs to compute a per-class
    // PIN FLOOR. A whole ensure_batch is pinned at once and alloc_slot will
    // not evict a pinned slot, so a class must have at least as many slots as
    // the largest number of its pages any single block owns -- otherwise a
    // wide batch over that block exhausts the class and the allocator aborts.
    // Demand share alone cannot see this: on GLM-5.2 the largest class is
    // 1.75% of all pages but 256 of them live in one block.
    std::map<size_t, std::map<int, int>> page_size_layer_counts() const;

    // True if any page is an MoE expert tensor — i.e. the model is sparse
    // MoE and downstream policy can use routing-aware prefetch / eviction.
    bool has_experts() const { return n_expert_pages_ > 0; }

    // Number of expert pages registered.
    int  n_expert_pages() const { return n_expert_pages_; }

    // MAD-236 — pinned (always-resident) page telemetry.
    bool   has_pinned()    const { return n_pinned_pages_ > 0; }
    int    n_pinned_pages() const { return n_pinned_pages_; }
    size_t pinned_bytes()  const { return pinned_bytes_; }

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
    // MAD-236 — pinned (always-resident) tracking.
    int                                   n_pinned_pages_ = 0;
    size_t                                pinned_bytes_   = 0;
};

}  // namespace wp
