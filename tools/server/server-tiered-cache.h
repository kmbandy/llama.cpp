#pragma once

#include "llama-kv-cache-tiered.h"
#include "server-common.h"

#include <memory>
#include <mutex>
#include <unordered_map>

// Tiered cache manager for server slots
struct server_tiered_cache {
    // Per-slot tier manager
    struct slot_tier_manager {
        std::unique_ptr<llama_kv_cache_tiered> tiered_cache;
        llama_tier_stats stats;
        bool initialized = false;

        void reset() {
            tiered_cache.reset();
            stats.reset();
            initialized = false;
        }
    };

    server_tiered_cache() : enabled(false) {}
    server_tiered_cache(const common_params& params);
    ~server_tiered_cache();

    // Initialize tier manager for a slot
    bool init_slot(int slot_id, const llama_model& model);

    // Get tier manager for a slot
    slot_tier_manager* get_slot_manager(int slot_id);

    // Evict tokens from a slot
    bool evict_from_slot(int slot_id, uint32_t n_tokens);

    // Migrate tokens between tiers for a slot
    bool migrate_in_slot(int slot_id, const std::vector<llama_pos>& positions,
                         llama_cache_tier from_tier, llama_cache_tier to_tier);

    // Get statistics for a slot
    llama_tier_stats get_slot_stats(int slot_id);

    // Get global statistics
    struct global_stats {
        uint64_t total_evictions = 0;
        uint64_t total_migrations = 0;
        uint64_t total_cache_hits = 0;
        uint64_t total_cache_misses = 0;
        double total_migration_latency_us = 0.0;

        void reset() {
            total_evictions = 0;
            total_migrations = 0;
            total_cache_hits = 0;
            total_cache_misses = 0;
            total_migration_latency_us = 0.0;
        }
    };

    global_stats get_global_stats();

    // Reset all statistics
    void reset_stats();

    // Check if tiered cache is enabled
    bool is_enabled() const { return enabled; }

private:
    bool enabled = false;
    std::unordered_map<int, slot_tier_manager> slot_managers;
    global_stats stats_;
    mutable std::mutex mutex;

    common_params params;
    std::string ssd_path;
    llama_eviction_policy eviction_policy;
    llama_cache_compression compression;
    float attention_threshold;
};
