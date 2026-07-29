#pragma once

#include "weight-pager/wp-pager.h"

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace wp {

class WeightPagerSet {
public:
    struct Route {
        WeightPager * pager    = nullptr;
        int           page_idx = -1;

        explicit operator bool() const {
            return pager != nullptr && page_idx >= 0;
        }
    };

    struct Entry {
        std::unique_ptr<WeightPager> pager;
        ggml_backend_buffer_type_t   buft       = nullptr;
        int                          device_idx = -1;
        std::string                  device_name;
    };

    WeightPagerSet() = default;
    ~WeightPagerSet();

    WeightPagerSet(const WeightPagerSet &)             = delete;
    WeightPagerSet & operator=(const WeightPagerSet &) = delete;

    WeightPager & add_pager(
        ggml_backend_buffer_type_t buft, int device_idx, std::string device_name);
    void build_routes(size_t expected_total_pages);

    Route find_page(const char * name) const;
    Route find_page(const std::string & name) const;

    WeightPager *       primary();
    const WeightPager * primary() const;
    size_t              size() const { return entries_.size(); }
    bool                empty() const { return entries_.empty(); }

    const std::vector<Entry> & entries() const { return entries_; }
    std::vector<Entry> &       entries()       { return entries_; }

    WeightPager * pager_for_block(int block_idx);

    // Advance EVERY pager's prefetch pipeline, not just the one owning the
    // current op. A pager whose band the token is not currently in is exactly
    // the one with spare time to prefetch; ticking only the active pager
    // freezes it precisely when it should be working.
    void tick_all();

    void mark_routing_boundaries(const struct ggml_cgraph * gf);
    void set_draft_window(int n_draft);
    int  prefetch_hot_experts(const int32_t * tokens, int n_tokens, int source);
    int  note_sampled_token(int32_t token);
    bool draft_oracle_should_run();
    int  flush_sample_oracle_at_fa();
    int  prefetch_sticky_hot_experts();
    void register_tid2eid_host(
        int block_idx, int n_expert_used, int n_vocab, const int32_t * table);

    bool xlayer_prefetch_enabled() const;
    bool host_prefetch_enabled() const;
    bool host_prefetch_async_enabled() const;
    bool predictors_have_router(int block_idx) const;
    void note_router_weight(
        int block_idx, const float * weights, int n_expert, int n_embd);
    void submit_xlayer_prefetch(const float * hidden, int from_layer);
    void submit_host_prefetch(
        const float * hidden, int n_embd, int from_layer, bool async);

private:
    void log_page_in_summary() const;

    std::vector<Entry>                    entries_;
    std::unordered_map<std::string, Route> routes_;
    std::unordered_map<int, WeightPager *> block_owners_;
};

} // namespace wp
