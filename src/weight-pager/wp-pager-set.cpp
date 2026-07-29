#include "weight-pager/wp-pager-set.h"

#include "llama-impl.h"

#include <stdexcept>

namespace wp {

WeightPagerSet::~WeightPagerSet() {
    log_page_in_summary();
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        LLAMA_LOG_WARN(
            "wp::WeightPagerSet detailed summary follows for device=%s index=%d\n",
            it->device_name.c_str(), it->device_idx);
        it->pager->shutdown();
    }
}

WeightPager & WeightPagerSet::add_pager(
        ggml_backend_buffer_type_t buft, int device_idx, std::string device_name) {
    Entry entry;
    entry.pager       = std::make_unique<WeightPager>();
    entry.buft        = buft;
    entry.device_idx  = device_idx;
    entry.device_name = std::move(device_name);
    entries_.push_back(std::move(entry));
    return *entries_.back().pager;
}

void WeightPagerSet::build_routes(size_t expected_total_pages) {
    routes_.clear();
    block_owners_.clear();

    size_t total = 0;
    for (Entry & entry : entries_) {
        WeightPager * pager = entry.pager.get();
        for (int page_idx = 0; page_idx < pager->n_pages(); ++page_idx) {
            const PageMeta & meta = pager->page_meta(page_idx);
            auto inserted = routes_.emplace(meta.tensor_name, Route{pager, page_idx});
            if (!inserted.second) {
                throw std::runtime_error(
                    "weight pager: page appears in more than one device partition: " +
                    meta.tensor_name);
            }
            if (meta.block_idx >= 0) {
                auto owner = block_owners_.emplace((int) meta.block_idx, pager);
                if (!owner.second && owner.first->second != pager) {
                    throw std::runtime_error(
                        "weight pager: one block spans multiple pager devices: block " +
                        std::to_string(meta.block_idx));
                }
            }
            ++total;
        }
    }
    if (total != expected_total_pages || routes_.size() != expected_total_pages) {
        throw std::runtime_error(
            "weight pager: partition page count mismatch (expected " +
            std::to_string(expected_total_pages) + ", got " + std::to_string(total) + ")");
    }
}

WeightPagerSet::Route WeightPagerSet::find_page(const char * name) const {
    if (name == nullptr) {
        return {};
    }
    thread_local std::string key;
    key.assign(name);
    return find_page(key);
}

WeightPagerSet::Route WeightPagerSet::find_page(const std::string & name) const {
    const auto it = routes_.find(name);
    return it == routes_.end() ? Route{} : it->second;
}

WeightPager * WeightPagerSet::primary() {
    return entries_.empty() ? nullptr : entries_.front().pager.get();
}

const WeightPager * WeightPagerSet::primary() const {
    return entries_.empty() ? nullptr : entries_.front().pager.get();
}

WeightPager * WeightPagerSet::pager_for_block(int block_idx) {
    const auto it = block_owners_.find(block_idx);
    return it == block_owners_.end() ? primary() : it->second;
}

void WeightPagerSet::tick_all() {
    for (Entry & entry : entries_) {
        entry.pager->tick();
    }
}

void WeightPagerSet::mark_routing_boundaries(const struct ggml_cgraph * gf) {
    for (Entry & entry : entries_) {
        entry.pager->mark_routing_boundaries(gf);
    }
}

void WeightPagerSet::set_draft_window(int n_draft) {
    for (Entry & entry : entries_) {
        entry.pager->set_draft_window(n_draft);
    }
}

int WeightPagerSet::prefetch_hot_experts(
        const int32_t * tokens, int n_tokens, int source) {
    int total = 0;
    for (Entry & entry : entries_) {
        total += entry.pager->prefetch_hot_experts(tokens, n_tokens, source);
    }
    return total;
}

int WeightPagerSet::note_sampled_token(int32_t token) {
    int total = 0;
    for (Entry & entry : entries_) {
        total += entry.pager->note_sampled_token(token);
    }
    return total;
}

bool WeightPagerSet::draft_oracle_should_run() {
    bool run = false;
    for (Entry & entry : entries_) {
        run = entry.pager->draft_oracle_should_run() || run;
    }
    return run;
}

int WeightPagerSet::flush_sample_oracle_at_fa() {
    int total = 0;
    for (Entry & entry : entries_) {
        total += entry.pager->flush_sample_oracle_at_fa();
    }
    return total;
}

int WeightPagerSet::prefetch_sticky_hot_experts() {
    int total = 0;
    for (Entry & entry : entries_) {
        total += entry.pager->prefetch_sticky_hot_experts();
    }
    return total;
}

void WeightPagerSet::register_tid2eid_host(
        int block_idx, int n_expert_used, int n_vocab, const int32_t * table) {
    for (Entry & entry : entries_) {
        entry.pager->register_tid2eid_host(
            block_idx, n_expert_used, n_vocab, table);
    }
}

bool WeightPagerSet::xlayer_prefetch_enabled() const {
    for (const Entry & entry : entries_) {
        if (entry.pager->xlayer_prefetch_enabled()) {
            return true;
        }
    }
    return false;
}

bool WeightPagerSet::host_prefetch_enabled() const {
    for (const Entry & entry : entries_) {
        if (entry.pager->host_prefetch_enabled()) {
            return true;
        }
    }
    return false;
}

bool WeightPagerSet::host_prefetch_async_enabled() const {
    for (const Entry & entry : entries_) {
        if (entry.pager->host_prefetch_async_enabled()) {
            return true;
        }
    }
    return false;
}

bool WeightPagerSet::predictors_have_router(int block_idx) const {
    for (const Entry & entry : entries_) {
        if (!entry.pager->predictor_has_router(block_idx)) {
            return false;
        }
    }
    return true;
}

void WeightPagerSet::note_router_weight(
        int block_idx, const float * weights, int n_expert, int n_embd) {
    for (Entry & entry : entries_) {
        entry.pager->note_router_weight(
            block_idx, weights, n_expert, n_embd);
    }
}

void WeightPagerSet::submit_xlayer_prefetch(
        const float * hidden, int from_layer) {
    for (Entry & entry : entries_) {
        entry.pager->submit_xlayer_prefetch(hidden, from_layer);
    }
}

void WeightPagerSet::submit_host_prefetch(
        const float * hidden, int n_embd, int from_layer, bool async) {
    for (Entry & entry : entries_) {
        if (async && entry.pager->host_prefetch_async_enabled()) {
            entry.pager->submit_host_prefetch_async(
                hidden, n_embd, from_layer);
        } else {
            entry.pager->submit_host_prefetch(hidden, from_layer);
        }
    }
}

void WeightPagerSet::log_page_in_summary() const {
    if (entries_.empty()) {
        return;
    }
    uint64_t page_ins = 0;
    uint64_t evictions = 0;
    uint64_t prefetch_hits = 0;
    uint64_t prefetch_misses = 0;
    uint64_t sync_fallbacks = 0;
    uint64_t io_bytes = 0;
    for (const Entry & entry : entries_) {
        const WeightPager::Stats & stats = entry.pager->stats();
        page_ins        += stats.page_ins;
        evictions       += stats.evictions;
        prefetch_hits   += stats.prefetch_hits;
        prefetch_misses += stats.prefetch_misses;
        sync_fallbacks  += stats.sync_fallbacks;
        io_bytes        += stats.io_bytes;
        LLAMA_LOG_WARN(
            "wp::WeightPagerSet page_ins device=%s index=%d: %lu\n",
            entry.device_name.c_str(), entry.device_idx,
            (unsigned long) stats.page_ins);
    }
    LLAMA_LOG_WARN(
        "wp::WeightPagerSet aggregate: devices=%zu page_ins=%lu evictions=%lu "
        "prefetch_hits=%lu prefetch_misses=%lu sync_fallbacks=%lu io_bytes=%lu\n",
        entries_.size(), (unsigned long) page_ins, (unsigned long) evictions,
        (unsigned long) prefetch_hits, (unsigned long) prefetch_misses,
        (unsigned long) sync_fallbacks, (unsigned long) io_bytes);
}

} // namespace wp
