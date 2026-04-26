#include "server-tiered-cache.h"

#include "server-common.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>

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

        // Extract semantic threshold and top_k
        semantic_threshold = params.kv_semantic_threshold;
        semantic_top_k = params.kv_semantic_top_k;

        stats_.reset();
    }

    // Load semantic embedding model if specified
    if (!params.kv_semantic_index.empty()) {
        auto sem_params = llama_model_default_params();
        sem_params.n_gpu_layers = 0; // CPU only
        sem_params.progress_callback = NULL;

        sem_model = llama_model_load_from_file(params.kv_semantic_index.c_str(), sem_params);
        if (!sem_model) {
            SRV_ERR("failed to load semantic index model: %s\n", params.kv_semantic_index.c_str());
            return;
        }

        auto ctx_params = llama_context_default_params();
        ctx_params.n_ctx = 512;
        ctx_params.embeddings = true;
        ctx_params.n_batch = 512;
        ctx_params.n_ubatch = 512;

        sem_ctx = llama_init_from_model(sem_model, ctx_params);
        if (!sem_ctx) {
            LOG_ERR("failed to create context for semantic index model\n");
            llama_model_free(sem_model);
            sem_model = nullptr;
            return;
        }

        SRV_INF("semantic KV index loaded: %s (CPU)\n", params.kv_semantic_index.c_str());
    }
}

server_tiered_cache::~server_tiered_cache() {
    // Cleanup all slot managers
    std::lock_guard<std::mutex> lock(mutex);
    slot_managers.clear();

    // Cleanup semantic embedding model
    if (sem_ctx) {
        llama_free(sem_ctx);
        sem_ctx = nullptr;
    }
    if (sem_model) {
        llama_model_free(sem_model);
        sem_model = nullptr;
    }
}

std::vector<float> server_tiered_cache::embed(const std::string & text) {
    if (!sem_ctx || !sem_model) {
        return {};
    }

    // Tokenize the input text
    auto * vocab = llama_model_get_vocab(sem_model);
    std::vector<llama_token> tokens(256);
    int32_t n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                      tokens.data(), (int32_t)tokens.capacity(),
                                      true, true);
    if (n_tokens < 0) {
        // Tokenization failed, try without add_special
        n_tokens = llama_tokenize(vocab, text.c_str(), (int32_t)text.size(),
                                  tokens.data(), (int32_t)tokens.capacity(),
                                  false, false);
        if (n_tokens < 0) {
            return {};
        }
    }
    tokens.resize(n_tokens);

    if (tokens.empty()) {
        return {};
    }

    // Create a batch and decode to get embeddings
    auto batch = llama_batch_get_one(tokens.data(), n_tokens);
    int32_t result = llama_decode(sem_ctx, batch);
    if (result != 0) {
        SRV_WRN("embed: llama_decode failed with code %d\n", result);
        llama_batch_free(batch);
        return {};
    }
    llama_batch_free(batch);

    // Extract embeddings for sequence 0
    float * embd_ptr = llama_get_embeddings_seq(sem_ctx, 0);
    if (!embd_ptr) {
        return {};
    }

    // Get embedding dimension
    int32_t n_embd = llama_model_n_embd(sem_model);
    std::vector<float> embedding(embd_ptr, embd_ptr + n_embd);

    // L2-normalize the embedding vector
    float norm = 0.0f;
    for (int32_t i = 0; i < n_embd; ++i) {
        norm += embedding[i] * embedding[i];
    }
    norm = std::sqrt(norm);
    if (norm > 1e-9f) {
        for (int32_t i = 0; i < n_embd; ++i) {
            embedding[i] /= norm;
        }
    }

    return embedding;
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

    // Store model pointer for detokenization
    detokenize_model = &model;

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
        
        // If semantic index is enabled, compute fingerprint for evicted tokens
        if (sem_enabled() && detokenize_model && n_tokens > 0) {
            // Create positions vector for the evicted tokens
            std::vector<llama_pos> positions(n_tokens);
            for (uint32_t i = 0; i < n_tokens; i++) {
                positions[i] = i;  // Simplified - should track actual positions from tiered cache
            }
            
            // Detokenize to get text for embedding
            std::string text;
            auto * vocab = llama_model_get_vocab(detokenize_model);
            for (auto pos : positions) {
                std::vector<char> buf(64);
                int len = llama_token_to_piece(vocab, pos, buf.data(), buf.size(), 0, false);
                if (len > 0) {
                    text.append(buf.data(), len);
                }
            }
            
            // Compute embedding
            auto embedding = embed(text);
            if (!embedding.empty()) {
                // Add fingerprint to tiered cache
                manager->tiered_cache->add_fingerprint(positions, embedding, TIER_WARM);
                
                // Save fingerprints to disk
                if (!fingerprints_path.empty()) {
                    manager->tiered_cache->save_fingerprints_to_disk(fingerprints_path + "/fingerprints.bin");
                }
            }
        }
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

std::vector<llama_kv_cache_tiered::PrefetchHint>
server_tiered_cache::get_prefetch_hints(int slot_id, const std::string& input_text, int top_k) {
    if (!sem_enabled()) {
        return {};
    }
    
    // Embed the input text
    auto embedding = embed(input_text);
    if (embedding.empty()) {
        return {};
    }
    
    // Get the slot manager
    auto* manager = get_slot_manager(slot_id);
    if (!manager || !manager->initialized) {
        return {};
    }
    
    // Score fingerprints and return hints
    return manager->tiered_cache->score_fingerprints(embedding, top_k, semantic_threshold);
}
