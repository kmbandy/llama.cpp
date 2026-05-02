#include "wp-page-catalog.h"

#include <cstdlib>
#include <cstring>

namespace wp {

namespace {

// Parse "blk.<N>." prefix. On success, fills block_idx and writes the
// substring after the prefix to *rest. Returns false if the name doesn't
// have a recognizable block prefix.
bool parse_block_prefix(const std::string & name, int16_t & block_idx, std::string & rest) {
    static const char prefix[] = "blk.";
    constexpr size_t prefix_len = sizeof(prefix) - 1;
    if (name.size() <= prefix_len || std::memcmp(name.data(), prefix, prefix_len) != 0) {
        return false;
    }

    const size_t num_start = prefix_len;
    const size_t dot       = name.find('.', num_start);
    if (dot == std::string::npos || dot == num_start) {
        return false;
    }

    // strtol: tolerate well-formed integers without throwing.
    const std::string num_str = name.substr(num_start, dot - num_start);
    char * end_ptr = nullptr;
    const long n   = std::strtol(num_str.c_str(), &end_ptr, 10);
    if (end_ptr == nullptr || *end_ptr != '\0' || n < 0 || n > INT16_MAX) {
        return false;
    }

    block_idx = static_cast<int16_t>(n);
    rest      = name.substr(dot + 1);
    return true;
}

// Detect MoE expert tensors. Recognizes:
//   "ffn_<role>_exps.weight"    -> consolidated (expert_idx stays -1)
//   "ffn_<role>.<E>.weight"     -> per-expert
// Both Qwen3-MoE and Mixtral-style naming are covered. Any other shape
// (including the common per-block dense ffn weights like
// "ffn_up.weight") is treated as non-expert.
void classify_expert(const std::string & rest, PageMeta & m) {
    struct role_entry {
        const char * prefix;
        size_t       len;
        uint8_t      mask;
    };
    static const role_entry roles[] = {
        // Order matters: longer match first when prefixes share a head
        // (none currently do, but cheap insurance).
        { "ffn_gate", 8, ROLE_GATE },
        { "ffn_down", 8, ROLE_DOWN },
        { "ffn_up",   6, ROLE_UP   },
    };

    for (const auto & r : roles) {
        if (rest.size() <= r.len) continue;
        if (std::memcmp(rest.data(), r.prefix, r.len) != 0) continue;

        // Consolidated form: "ffn_<role>_exps."
        static const char cons[] = "_exps.";
        constexpr size_t  cons_len = sizeof(cons) - 1;
        if (rest.size() >= r.len + cons_len &&
            std::memcmp(rest.data() + r.len, cons, cons_len) == 0) {
            m.is_expert        = true;
            m.is_consolidated  = true;
            m.expert_role_mask = r.mask;
            // expert_idx stays -1 — tensor packs all experts.
            return;
        }

        // Per-expert form: "ffn_<role>.<E>."
        if (rest[r.len] == '.') {
            const size_t num_start = r.len + 1;
            const size_t next_dot  = rest.find('.', num_start);
            if (next_dot == std::string::npos || next_dot == num_start) {
                return;
            }
            const std::string num_str = rest.substr(num_start, next_dot - num_start);
            char * end_ptr = nullptr;
            const long e   = std::strtol(num_str.c_str(), &end_ptr, 10);
            if (end_ptr == nullptr || *end_ptr != '\0' || e < 0 || e > INT16_MAX) {
                return;
            }
            m.is_expert        = true;
            m.is_consolidated  = false;
            m.expert_role_mask = r.mask;
            m.expert_idx       = static_cast<int16_t>(e);
            return;
        }
        // Anything else after the role prefix means it's a dense FFN weight,
        // not an expert — leave is_expert = false.
        return;
    }
}

}  // namespace

int PageCatalog::add(const std::string & name, uint16_t file_idx,
                     uint64_t file_offset, size_t size) {
    const int idx = static_cast<int>(pages_.size());
    PageMeta m;
    m.tensor_name = name;
    m.file_idx    = file_idx;
    m.file_offset = file_offset;
    m.size        = size;

    std::string rest;
    if (parse_block_prefix(name, m.block_idx, rest)) {
        classify_expert(rest, m);
    }

    if (m.is_expert) {
        ++n_expert_pages_;
    }

    pages_.push_back(std::move(m));
    name_to_idx_.emplace(name, idx);
    if (size > max_size_) {
        max_size_ = size;
    }
    return idx;
}

int PageCatalog::add_consolidated_experts(const std::string & name, uint16_t file_idx,
                                          uint64_t file_offset, size_t total_size,
                                          int n_experts) {
    if (n_experts <= 1) {
        // Degenerate case — caller should have used add() directly.
        return add(name, file_idx, file_offset, total_size);
    }
    if (total_size % static_cast<size_t>(n_experts) != 0) {
        // Non-uniform experts aren't expected for the consolidated layout
        // we target. Bail to non-consolidated behaviour rather than silently
        // sub-divide unevenly. Caller can detect this via the returned
        // single-page result (no sub_expert children registered).
        return add(name, file_idx, file_offset, total_size);
    }

    // Step 1: register the parent (consolidated) page. No slot will be
    // allocated for it by the pool — it's pure metadata so the eval
    // callback can resolve "this op uses src[0] = consolidated tensor"
    // and walk to the children.
    PageMeta parent;
    parent.tensor_name     = name;
    parent.file_idx        = file_idx;
    parent.file_offset     = file_offset;
    parent.size            = total_size;
    parent.is_consolidated = true;
    // Re-run the standard parser to fill block_idx + expert_role_mask on
    // the parent, but is_expert stays false (the parent isn't paged itself
    // — only its children are).
    {
        std::string rest;
        if (parse_block_prefix(name, parent.block_idx, rest)) {
            classify_expert(rest, parent);
            // classify_expert sets is_expert / is_consolidated / role —
            // we want role + block but NOT to count the parent as an
            // expert page (children are the slottable units).
            parent.is_expert = false;
        }
    }

    const int parent_idx = static_cast<int>(pages_.size());
    pages_.push_back(std::move(parent));
    name_to_idx_.emplace(name, parent_idx);
    // Note: parent's `size` is the full consolidated size; we deliberately
    // do NOT update max_size_ from the parent — only sub-experts allocate
    // slots, and the per-expert size is what determines slot stride.

    // Step 2: register N sub-pages, one per expert. Each gets a synthetic
    // name so it's directly findable by name. Per-expert offset = base +
    // e * per_expert_size; per-expert size = total_size / n_experts.
    const size_t per_expert_size = total_size / static_cast<size_t>(n_experts);
    const int    first_sub_idx   = static_cast<int>(pages_.size());

    for (int e = 0; e < n_experts; ++e) {
        PageMeta sub = pages_[parent_idx];  // copy to inherit block_idx + role
        sub.tensor_name      = name + "#expert." + std::to_string(e);
        sub.file_offset      = file_offset + static_cast<uint64_t>(e) * per_expert_size;
        sub.size             = per_expert_size;
        sub.expert_idx       = static_cast<int16_t>(e);
        sub.is_expert        = true;
        sub.is_consolidated  = false;
        sub.is_sub_expert    = true;
        sub.parent_page_idx  = parent_idx;

        const int sub_idx = static_cast<int>(pages_.size());
        pages_.push_back(std::move(sub));
        name_to_idx_.emplace(pages_.back().tensor_name, sub_idx);
        ++n_expert_pages_;
    }

    if (per_expert_size > max_size_) {
        max_size_ = per_expert_size;
    }
    return first_sub_idx;
}

int PageCatalog::find(const std::string & name) const {
    const auto it = name_to_idx_.find(name);
    return it == name_to_idx_.end() ? -1 : it->second;
}

std::vector<int> PageCatalog::pages_for_block(int block_idx) const {
    std::vector<int> out;
    if (block_idx < 0) return out;
    out.reserve(8);
    for (size_t i = 0; i < pages_.size(); ++i) {
        if (pages_[i].block_idx == block_idx) {
            out.push_back(static_cast<int>(i));
        }
    }
    return out;
}

std::vector<int> PageCatalog::pages_for_expert(int block_idx, int expert_idx) const {
    std::vector<int> out;
    if (block_idx < 0) return out;
    out.reserve(3);  // typically up + gate + down
    for (size_t i = 0; i < pages_.size(); ++i) {
        const auto & m = pages_[i];
        if (!m.is_expert) continue;
        if (m.block_idx != block_idx) continue;
        if (m.expert_idx != expert_idx) continue;
        out.push_back(static_cast<int>(i));
    }
    return out;
}

void PageCatalog::clear() {
    pages_.clear();
    name_to_idx_.clear();
    max_size_       = 0;
    n_expert_pages_ = 0;
}

}  // namespace wp
