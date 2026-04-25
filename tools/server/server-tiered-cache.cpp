#include "server-tiered-cache.h"

#include "server-common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>

server_tiered_cache::server_tiered_cache(const common_params& params)
    : params(params) {
    // Check if tiered cache is enabled
    enabled = params.kv_tiered_enabled;

    if (enabled) {
        // Extract SSD path from params
        ssd_path = params.kv_tier_ssd_path;
        if (ssd_path.empty()) {
            // Default SSD path if not specified
            ssd_path = "./tiered-cache";
        }

        // Extract eviction policy
        eviction_policy = llama_eviction_policy(params.kv_tier_eviction_policy);

        // Extract compression type
        compression = llama_cache_compression(params.kv_tier_compression);

        // Extract attention threshold
        attention_threshold = params.kv_tier_attention_threshold;

        stats_.reset();
    }
}

server_tiered_cache::~server_tiered_cache() {
    // Cleanup all slot managers
    std::lock_guard<std::mutex> lock(mutex);
    slot_managers.clear();
}

bool server_tiered_cache::init_slot(int slot_id, const llama_model& model, struct llama_context * lctx) {
    if (!enabled) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex);

    // Check if slot already has a manager
    auto it = slot_managers.find(slot_id);
    if (it != slot_managers.end()) {
        // Already initialized
        return true;
    }

    // Create new slot manager
    slot_tier_manager manager;

    // Create tiered cache configuration
    llama_tier_config config;
    config.hot_percent = params.kv_tier_hot_pct;
    config.warm_percent = params.kv_tier_warm_pct;
    config.cold_percent = params.kv_tier_cold_pct;
    config.total_ctx = params.kv_tier_total_ctx > 0 ? params.kv_tier_total_ctx : params.n_ctx;
    config.warm_device = params.kv_warm_device;

    // Create tiered cache instance
    manager.tiered_cache = std::make_unique<llama_kv_cache_tiered>(
        model,
        config,
        ssd_path,
        eviction_policy,
        compression,
        attention_threshold
    );

    // Initialize the tiered cache (warm slots reserved, buffers come after layer wiring)
    if (!manager.tiered_cache->init()) {
        return false;
    }

    // Wire per-layer K/V tensor pointers for actual data movement
    if (lctx) {
        auto * mem = llama_get_memory(lctx);
        auto * kv  = dynamic_cast<llama_kv_cache *>(mem);
        if (kv) {
            manager.tiered_cache->set_kv_layers_from_cache(kv);
        } else {
            SRV_WRN("tiered cache: could not cast memory to llama_kv_cache for slot %d — metadata-only mode\n", slot_id);
        }
    }

    manager.initialized = true;
    manager.stats.reset();
    slot_managers.emplace(slot_id, std::move(manager));

    return true;
}

server_tiered_cache::slot_tier_manager* server_tiered_cache::get_slot_manager(int slot_id) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = slot_managers.find(slot_id);
    if (it == slot_managers.end()) {
        return nullptr;
    }

    return &it->second;
}

bool server_tiered_cache::evict_from_slot(int slot_id, uint32_t n_tokens, uint32_t n_hot_positions) {
    if (!enabled) {
        return false;
    }

    auto* manager = get_slot_manager(slot_id);
    if (!manager || !manager->initialized) {
        return false;
    }

    // Populate metadata so eviction scoring has candidates to work with
    if (n_hot_positions > 0) {
        manager->tiered_cache->track_hot_range(n_hot_positions);
    }

    // Evict tokens using the tiered cache
    bool result = manager->tiered_cache->evict_tokens(n_tokens, TIER_HOT);

    if (result) {
        // Update global stats
        std::lock_guard<std::mutex> lock(mutex);
        stats_.total_evictions += n_tokens;
    }

    return result;
}

bool server_tiered_cache::migrate_in_slot(int slot_id,
                                           const std::vector<llama_pos>& positions,
                                           llama_cache_tier from_tier,
                                           llama_cache_tier to_tier) {
    if (!enabled) {
        return false;
    }

    auto* manager = get_slot_manager(slot_id);
    if (!manager || !manager->initialized) {
        return false;
    }

    // Migrate tokens between tiers
    bool result = manager->tiered_cache->migrate_tokens(positions, from_tier, to_tier);

    if (result) {
        // Update global stats
        std::lock_guard<std::mutex> lock(mutex);
        stats_.total_migrations += positions.size();
    }

    return result;
}

llama_tier_stats server_tiered_cache::get_slot_stats(int slot_id) {
    if (!enabled) {
        return llama_tier_stats{};
    }

    auto* manager = get_slot_manager(slot_id);
    if (!manager || !manager->initialized) {
        return llama_tier_stats{};
    }

    return manager->tiered_cache->get_stats();
}

server_tiered_cache::global_stats server_tiered_cache::get_global_stats() {
    std::lock_guard<std::mutex> lock(mutex);
    return stats_;
}

void server_tiered_cache::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex);
    stats_.reset();
}
