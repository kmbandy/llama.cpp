#include "weight-pager/wp-router.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace wp {

const char * const ROUTER_EXPERT_PATTERN     = "ffn_(up|gate|down)_exps\\.";
const char * const ROUTER_SHEXP_PATTERN      = "ffn_(up|gate|down)_shexp\\.";
// NOTE 2026-07-31: this said ffn_exp_probs_b, which NEVER MATCHED ANY MODEL --
// llama-arch.cpp:419 names the GGUF tensor "blk.%d.exp_probs_b"; only the C++
// field (llama-model.h:348) carries the ffn_ prefix. The router bias therefore
// fell through to the dense catch-all and landed on the RESIDENT card, forcing a
// TB3 crossing per layer per token -- exactly what this island exists to avoid.
// Matching the real name also still covers an ffn_-prefixed variant by substring.
// FFN island on paging GPU: keeps MoE block intra-device (R9700). Residual then
// only crosses TB3 into/out of the attention island, not per-op mid-FFN.
const char * const ROUTER_FFN_ISLAND_PATTERN =
        "(ffn_norm\\.|ffn_gate_inp\\.|exp_probs_b\\.|ffn_gate_tid2eid\\.|hc_ffn_)";
const char * const ROUTER_TOKEN_EMBD_PATTERN = "token_embd\\.";
const char * const ROUTER_DENSE_PATTERN      = ".*";

// Keep in lockstep with ROUTER_EXPERT_PATTERN. Substring matching rather than
// <regex> because this runs per tensor during load.
bool is_routed_expert_name(const char * name) {
    const char * p = name ? std::strstr(name, "ffn_") : nullptr;
    if (p == nullptr) {
        return false;
    }
    return std::strstr(p, "ffn_up_exps.")   != nullptr ||
           std::strstr(p, "ffn_gate_exps.") != nullptr ||
           std::strstr(p, "ffn_down_exps.") != nullptr;
}

bool parse_block_index(const char * name, int & block_idx) {
    if (name == nullptr || std::strncmp(name, "blk.", 4) != 0) {
        return false;
    }
    const char * digits = name + 4;
    if (!std::isdigit((unsigned char) *digits)) {
        return false;
    }
    char *          end = nullptr;
    const long long v   = std::strtoll(digits, &end, 10);
    if (end == digits || *end != '.' || v < 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    block_idx = (int) v;
    return true;
}

bool ResidentExpertPlan::covers_block(int block_idx) const {
    return std::binary_search(layers_.begin(), layers_.end(), block_idx);
}

bool ResidentExpertPlan::covers_tensor(const char * name) const {
    if (layers_.empty() || !is_routed_expert_name(name)) {
        return false;
    }
    int block_idx = -1;
    return parse_block_index(name, block_idx) && covers_block(block_idx);
}

void ResidentExpertPlan::rebuild_pattern() {
    pattern_.clear();
    if (layers_.empty()) {
        return;
    }
    // Anchored on "blk\." so a block index can never match a substring of a
    // longer index (e.g. block 1 matching "blk.17.").
    // Anchored: the override list is matched with regex_search (a SUBSTRING
    // match) while covers_tensor() requires a "blk." PREFIX. Without the ^
    // a hypothetical name like "blk.5.<extra>.ffn_up_exps.weight" would be
    // claimed resident by covers_tensor but routed to the paging device by
    // the override, putting the two out of agreement. No current arch emits
    // such a name; anchoring makes the equivalence structural anyway.
    pattern_ = "^blk\\.(";
    for (size_t i = 0; i < layers_.size(); ++i) {
        if (i > 0) {
            pattern_ += '|';
        }
        pattern_ += std::to_string(layers_[i]);
    }
    pattern_ += ")\\.ffn_(up|gate|down)_exps\\.";
}

std::string ResidentExpertPlan::describe() const {
    if (layers_.empty()) {
        return "none";
    }
    // Collapse to ranges so a 40-block plan does not print 40 numbers.
    std::string out;
    size_t      i = 0;
    while (i < layers_.size()) {
        size_t j = i;
        while (j + 1 < layers_.size() && layers_[j + 1] == layers_[j] + 1) {
            ++j;
        }
        if (!out.empty()) {
            out += ',';
        }
        out += std::to_string(layers_[i]);
        if (j > i) {
            out += '-' + std::to_string(layers_[j]);
        }
        i = j + 1;
    }
    return out;
}

ResidentExpertPlan ResidentExpertPlan::from_blocks(const std::vector<LayerExpertBytes> & per_layer,
                                                   const std::vector<int> &              blocks) {
    ResidentExpertPlan plan;

    // Deduplicate BEFORE accumulating: a repeated block would otherwise count
    // its bytes more than once, and that total is what the caller checks
    // against free VRAM.
    std::vector<int> wanted = blocks;
    std::sort(wanted.begin(), wanted.end());
    wanted.erase(std::unique(wanted.begin(), wanted.end()), wanted.end());

    for (int b : wanted) {
        for (const LayerExpertBytes & l : per_layer) {
            if (l.block_idx == b && l.bytes > 0) {
                plan.layers_.push_back(b);
                plan.bytes_ += l.bytes;
                break;
            }
        }
    }
    plan.rebuild_pattern();   // layers_ is already sorted and unique
    return plan;
}

ResidentExpertRequest parse_resident_expert_request(const char * value) {
    ResidentExpertRequest req;
    if (value == nullptr || value[0] == '\0') {
        return req;
    }

    std::string v(value);
    // trim
    const size_t b = v.find_first_not_of(" \t");
    const size_t e = v.find_last_not_of(" \t");
    v = (b == std::string::npos) ? std::string() : v.substr(b, e - b + 1);

    std::string lower = v;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return (char) std::tolower(c); });

    // NOTE: "0" is deliberately NOT an off-switch. A bare number is a BLOCK
    // index everywhere else here ("5" is block 5), so letting "0" mean off
    // would silently disable the feature for the one block a user is most
    // likely to name first, while the command line still looked configured.
    if (lower.empty() || lower == "off" || lower == "none") {
        return req;
    }
    // "auto" and byte sizes used to mean "fill this device with whole blocks".
    // REMOVED by decision: on a model far larger than the fleet's VRAM a spare
    // device holds only a few percent of the blocks, and because whole blocks
    // are byte-neutral that buys only a few percent of the expert traffic. The
    // mode looked configured while being nearly worthless. Reject loudly rather
    // than silently reinterpreting "12GiB" as a block range.
    if (lower == "auto" || lower.find_first_not_of("0123456789,- \t") != std::string::npos) {
        throw std::invalid_argument(
            "resident-experts: expected block ranges like \"0-6,20-22\", or \"off\"; got '" + v +
            "' (budget/auto fill was removed -- whole-block residency buys only a few percent on a "
            "model this size, so blocks must be named deliberately)");
    }

    // Block range list: "0-6,20-22" or "5" or "5,9".
    std::vector<int> blocks;
    size_t           start = 0;
    while (start <= v.size()) {
        const size_t      comma = v.find(',', start);
        const std::string item  = v.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty()) {
            throw std::invalid_argument("resident-experts: empty range in '" + v + "'");
        }
        const size_t dash = item.find('-');
        long long    first = 0;
        long long    last  = 0;
        try {
            // stoll stops at the first non-digit and does NOT report leftovers,
            // so "5x" would parse as 5 and "1-2-3" as 1-2. Require that each
            // number consumed its whole field.
            size_t used_first = 0, used_last = 0;
            if (dash == std::string::npos) {
                first = last = std::stoll(item, &used_first);
                if (used_first != item.size()) {
                    throw std::invalid_argument(
                        "resident-experts: trailing characters in block '" + item + "'");
                }
            } else {
                const std::string first_s = item.substr(0, dash);
                const std::string last_s  = item.substr(dash + 1);
                first = std::stoll(first_s, &used_first);
                last  = std::stoll(last_s,  &used_last);
                if (used_first != first_s.size() || used_last != last_s.size()) {
                    throw std::invalid_argument(
                        "resident-experts: trailing characters in block range '" + item + "'");
                }
            }
        } catch (const std::exception &) {
            throw std::invalid_argument("resident-experts: invalid block range '" + item + "'");
        }
        if (first < 0 || last < first || last > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("resident-experts: invalid block range '" + item + "'");
        }
        for (long long i = first; i <= last; ++i) {
            blocks.push_back((int) i);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    req.enabled = true;
    req.blocks  = std::move(blocks);
    return req;
}

static std::vector<int> parse_explicit_block_ranges(
        const std::string & value, const char * option_name) {
    std::vector<int> blocks;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string item =
            value.substr(start, comma == std::string::npos
                ? std::string::npos : comma - start);
        if (item.empty()) {
            throw std::invalid_argument(
                std::string(option_name) + ": empty range in '" + value + "'");
        }
        const size_t dash = item.find('-');
        long long first = 0;
        long long last  = 0;
        try {
            size_t used_first = 0;
            size_t used_last  = 0;
            if (dash == std::string::npos) {
                first = last = std::stoll(item, &used_first);
                if (used_first != item.size()) {
                    throw std::invalid_argument("trailing characters");
                }
            } else {
                const std::string first_s = item.substr(0, dash);
                const std::string last_s  = item.substr(dash + 1);
                first = std::stoll(first_s, &used_first);
                last  = std::stoll(last_s, &used_last);
                if (used_first != first_s.size() || used_last != last_s.size()) {
                    throw std::invalid_argument("trailing characters");
                }
            }
        } catch (const std::exception &) {
            throw std::invalid_argument(
                std::string(option_name) + ": invalid block range '" + item + "'");
        }
        if (first < 0 || last < first ||
            last > std::numeric_limits<int>::max()) {
            throw std::invalid_argument(
                std::string(option_name) + ": invalid block range '" + item + "'");
        }
        for (long long i = first; i <= last; ++i) {
            blocks.push_back((int) i);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    return blocks;
}

static std::string block_expert_pattern(const std::vector<int> & blocks) {
    std::string pattern = "blk\\.(";
    for (size_t i = 0; i < blocks.size(); ++i) {
        if (i > 0) {
            pattern += '|';
        }
        pattern += std::to_string(blocks[i]);
    }
    pattern += ")\\.ffn_(up|gate|down)_exps\\.";
    return pattern;
}

std::vector<DeviceLayerRequest> parse_device_layer_request(const char * value) {
    std::vector<DeviceLayerRequest> out;
    if (value == nullptr || value[0] == '\0') {
        return out;
    }

    std::string input(value);
    size_t start = 0;
    while (start <= input.size()) {
        const size_t semi = input.find(';', start);
        const std::string group =
            input.substr(start, semi == std::string::npos
                ? std::string::npos : semi - start);
        const size_t colon = group.find(':');
        if (group.empty() || colon == std::string::npos ||
            colon == 0 || colon + 1 == group.size()) {
            throw std::invalid_argument(
                "device-layers: expected DEVICE:BLOCKS group, got '" + group + "'");
        }
        DeviceLayerRequest req;
        req.device = group.substr(0, colon);
        req.blocks = parse_explicit_block_ranges(
            group.substr(colon + 1), "device-layers");
        out.push_back(std::move(req));
        if (semi == std::string::npos) {
            break;
        }
        start = semi + 1;
    }
    return out;
}

void DeviceLayerPlan::add(
        std::string device, std::vector<int> blocks,
        ggml_backend_buffer_type_t buft) {
    if (device.empty() || blocks.empty() || buft == nullptr) {
        throw std::invalid_argument("device-layers: empty device, blocks, or buft");
    }
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
    for (int block : blocks) {
        if (covers_block(block)) {
            throw std::invalid_argument(
                "device-layers: block " + std::to_string(block) +
                " assigned more than once");
        }
    }
    Entry entry;
    entry.device  = std::move(device);
    entry.blocks  = std::move(blocks);
    entry.pattern = block_expert_pattern(entry.blocks);
    entry.buft    = buft;
    entries_.push_back(std::move(entry));
}

bool DeviceLayerPlan::covers_block(int block_idx) const {
    for (const Entry & entry : entries_) {
        if (std::binary_search(entry.blocks.begin(), entry.blocks.end(), block_idx)) {
            return true;
        }
    }
    return false;
}

ggml_backend_buffer_type_t DeviceLayerPlan::buft_for_tensor(
        const char * name, ggml_backend_buffer_type_t fallback) const {
    if (!is_routed_expert_name(name)) {
        return fallback;
    }
    int block_idx = -1;
    if (!parse_block_index(name, block_idx)) {
        return fallback;
    }
    for (const Entry & entry : entries_) {
        if (std::binary_search(entry.blocks.begin(), entry.blocks.end(), block_idx)) {
            return entry.buft;
        }
    }
    return fallback;
}

std::vector<llama_model_tensor_buft_override> build_router_overrides(
        ggml_backend_buffer_type_t paging_buft,
        ggml_backend_buffer_type_t resident_buft,
        ggml_backend_buffer_type_t cpu_buft,
        const llama_model_tensor_buft_override * user_overrides,
        bool emit_dense_catch_all,
        ggml_backend_buffer_type_t island_buft,
        const ResidentExpertPlan * resident_experts,
        const DeviceLayerPlan * device_layers) {
    std::vector<llama_model_tensor_buft_override> out;

    ggml_backend_buffer_type_t shexp_ffn_buft = island_buft != nullptr ? island_buft : paging_buft;

    // 0) Resident-expert blocks -> island device, NOT paged. Must come first:
    //    rule 1 matches every routed expert and first-match-wins, so emitting
    //    this afterwards would be a silent no-op.
    if (resident_experts != nullptr && !resident_experts->empty() && island_buft != nullptr) {
        out.push_back({ resident_experts->pattern().c_str(), island_buft });
    }

    // 1) Explicit paged layer bands -> their named device.
    if (device_layers != nullptr) {
        for (const DeviceLayerPlan::Entry & entry : device_layers->entries()) {
            out.push_back({ entry.pattern.c_str(), entry.buft });
        }
    }

    // 2) Remaining routed experts -> primary paging device.
    out.push_back({ ROUTER_EXPERT_PATTERN, paging_buft });

    // 3) Shared expert -> paging device, always-resident (not in paged set);
    //    or island device when the FFN-island role is configured.
    out.push_back({ ROUTER_SHEXP_PATTERN, shexp_ffn_buft });

    // 4) FFN island dense -> paging (T4: fewer TB3 intermediate activations);
    //    or island device when the FFN-island role is configured.
    out.push_back({ ROUTER_FFN_ISLAND_PATTERN, shexp_ffn_buft });

    // 5) token_embd -> CPU when available (row gather; frees eGPU for draft/attn).
    if (cpu_buft != nullptr) {
        out.push_back({ ROUTER_TOKEN_EMBD_PATTERN, cpu_buft });
    }

    // 6) User overrides before dense catch-all so they are never shadowed.
    if (user_overrides != nullptr) {
        for (const auto * o = user_overrides; o->pattern != nullptr; ++o) {
            out.push_back(*o);
        }
    }

    // 7) Everything else dense (attention, lm_head, attn norms, ...)
    //    -> resident / attention-island GPU.
    if (emit_dense_catch_all) {
        out.push_back({ ROUTER_DENSE_PATTERN, resident_buft });
    }
    out.push_back({ nullptr, nullptr });
    return out;
}

} // namespace wp
