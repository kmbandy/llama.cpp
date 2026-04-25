#pragma once

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-eviction-policy.h"

#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <mutex>
#include <chrono>
#include <unordered_map>

#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
#endif

// Tier types for KV cache
enum llama_cache_tier {
    TIER_HOT,   // VRAM - currently used context
    TIER_WARM,  // RAM - recently used context
    TIER_COLD   // SSD - less frequently accessed context
};

// Compression types for cold tier storage
enum llama_cache_compression {
    COMPRESSION_NONE,
    COMPRESSION_INT4,    // 4-bit quantized (4x space reduction)
    COMPRESSION_INT8,    // 8-bit quantized (2x space reduction)
    COMPRESSION_LZ4,     // LZ4 compression
    COMPRESSION_QUANTIZED // Model-native quantization
};

// Tier configuration
struct llama_tier_config {
    float hot_percent;   // Percentage of total context for hot tier (VRAM)
    float warm_percent;  // Percentage of total context for warm tier (RAM)
    float cold_percent;  // Percentage of total context for cold tier (SSD)

    uint32_t total_ctx;  // Total context size

    uint32_t hot_capacity() const { return uint32_t(total_ctx * hot_percent / 100.0f); }
    uint32_t warm_capacity() const { return uint32_t(total_ctx * warm_percent / 100.0f); }
    uint32_t cold_capacity() const { return uint32_t(total_ctx * cold_percent / 100.0f); }

    // HIP device index for warm tier (-1 = RAM/SSD fallback, 1 = 6900XT)
    int warm_device = -1;

    // Default configuration: 25% hot, 25% warm, 50% cold
    static llama_tier_config default_config(uint32_t total_ctx) {
        return {25.0f, 25.0f, 50.0f, total_ctx};
    }
};

// SSD storage format for cold tier
struct llama_ssd_storage_format {
    // Magic header for validation
    static constexpr uint32_t MAGIC = 0x4B565443; // "KVTC"
    static constexpr uint32_t VERSION = 1;

    // File header structure
    struct file_header {
        uint32_t magic;
        uint32_t version;
        uint32_t n_layers;
        uint32_t n_embd_k;
        uint32_t n_embd_v;
        uint32_t n_tokens;
        uint32_t compression_type;
        uint32_t layer_offset;  // Starting token position for this layer
        float attention_threshold;
        uint64_t index_offset;  // Offset to index section
        uint32_t index_size;    // Size of index in bytes
    };

    // Layer index entry for quick lookup
    struct layer_index_entry {
        uint32_t layer;
        uint32_t n_tokens;
        uint64_t file_offset;  // Byte offset in file
        uint32_t data_size;    // Size of layer data in bytes
    };

    // Token range for a layer
    struct token_range {
        llama_pos start;
        llama_pos end;
        uint32_t n_tokens;
    };

    // Write KV cache to SSD
    bool write(const std::string& path,
               const float* k_data,
               const float* v_data,
               uint32_t n_tokens,
               uint32_t n_layers,
               uint32_t n_embd_k,
               uint32_t n_embd_v,
               llama_cache_compression compression = COMPRESSION_INT4);

    // Read KV cache from SSD
    bool read(const std::string& path,
              float* k_data,
              float* v_data,
              uint32_t n_tokens,
              uint32_t n_layers,
              uint32_t n_embd_k,
              uint32_t n_embd_v);

    // Build index file for fast lookup
    bool build_index(const std::string& index_path,
                      const std::string& data_path,
                      uint32_t n_layers);

    // Load index from file
    bool load_index(const std::string& index_path);

    // Save index to file
    bool save_index(const std::string& index_path);

    // Get layer data offset from index
    uint64_t get_layer_offset(uint32_t layer) const;

    // Check if layer is in index
    bool has_layer(uint32_t layer) const;

private:
    // Layer index for fast lookup
    std::vector<layer_index_entry> layer_index;

    // Helper for quantization
    std::vector<uint8_t> quantize_int4(const float* data, uint32_t n_elements);
    std::vector<uint8_t> quantize_int8(const float* data, uint32_t n_elements);
    bool dequantize_int4(const uint8_t* data, float* out, uint32_t n_elements);
    bool dequantize_int8(const uint8_t* data, float* out, uint32_t n_elements);
};

// Tier statistics for monitoring
struct llama_tier_stats {
    uint32_t hot_tokens;
    uint32_t warm_tokens;
    uint32_t cold_tokens;
    uint64_t eviction_count;
    uint64_t cache_hits;
    uint64_t cache_misses;
    double total_migration_latency_us;

    void reset() {
        hot_tokens = 0;
        warm_tokens = 0;
        cold_tokens = 0;
        eviction_count = 0;
        cache_hits = 0;
        cache_misses = 0;
        total_migration_latency_us = 0.0;
    }

    double get_hit_rate() const {
        uint64_t total = cache_hits + cache_misses;
        return total > 0 ? double(cache_hits) / total : 0.0;
    }
};

// Main tiered cache class
class llama_kv_cache_tiered {
public:
    // Constructor
    llama_kv_cache_tiered(
        const llama_model& model,
        const llama_tier_config& config,
        const std::string& ssd_path,
        llama_eviction_policy eviction_policy = HYBRID,
        llama_cache_compression compression = COMPRESSION_INT4,
        float attention_threshold = 0.1f);

    // Destructor
    ~llama_kv_cache_tiered();

    // Tier management
    bool init();
    bool resize(uint32_t new_total_ctx);
    bool set_eviction_policy(llama_eviction_policy policy);
    bool set_attention_threshold(float threshold);
    // Wire real KV layer tensors for actual data movement
    void set_kv_layers_from_cache(llama_kv_cache * cache);

    // Eviction interface
    bool evict_tokens(uint32_t n_tokens_to_evict, llama_cache_tier from_tier);
    // Pre-register hot positions 0..n_hot-1 so eviction candidates are available
    void track_hot_range(uint32_t n_hot);
    
    // Migration with tensor handles - requires actual KV cache tensor pointers
    bool migrate_tokens(const std::vector<llama_pos>& positions,
                        llama_cache_tier from_tier,
                        llama_cache_tier to_tier,
                        const ggml_tensor* k_tensor = nullptr,
                        const ggml_tensor* v_tensor = nullptr);

    // Migration optimization
    bool batch_migrate_tokens(const std::vector<llama_pos>& positions,
                               llama_cache_tier from_tier,
                               llama_cache_tier to_tier,
                               const ggml_tensor* k_tensor = nullptr,
                               const ggml_tensor* v_tensor = nullptr);
    
    bool prefetch_tokens(const std::vector<llama_pos>& positions,
                        llama_cache_tier target_tier);

    // Token access
    bool contains_token(llama_pos pos) const;
    bool load_token(llama_pos pos, llama_cache_tier target_tier);

    // Statistics
    llama_tier_stats get_stats() const;
    void reset_stats();

    // Configuration getters
    const llama_tier_config& get_config() const { return config; }
    llama_eviction_policy get_eviction_policy() const { return eviction_policy; }
    float get_attention_threshold() const { return attention_threshold; }

private:
    // Tier capacities
    llama_tier_config config;

    // Eviction policy
    llama_eviction_policy eviction_policy;

    // SSD storage path
    std::string ssd_path;

    // Compression type for cold tier
    llama_cache_compression compression;

    // Attention threshold for eviction
    float attention_threshold;

    // Token metadata store
    llama_token_metadata_store token_metadata;

    // Statistics
    llama_tier_stats stats;

    // Thread safety
    mutable std::mutex mutex;

    // Per-layer KV tensor info — populated by set_kv_layers_from_cache()
    // K layout: [n_embd_k_gqa, kv_size, n_stream] — token p is contiguous at row p
    // V layout: [n_embd_v_gqa, kv_size, n_stream] — when v_trans=true, token p's
    //   element j is at flat index (j*kv_size + p), requiring scatter/gather
    struct TieredKVLayer {
        ggml_tensor * k     = nullptr;   // raw K tensor in VRAM (or host)
        ggml_tensor * v     = nullptr;   // raw V tensor in VRAM (or host)
        bool v_trans        = false;     // true = V stored transposed (default)
        int64_t kv_size     = 0;         // n_ctx_max (= k->ne[1])
        size_t k_bytes      = 0;         // bytes per token for K (contiguous row)
        size_t v_bytes      = 0;         // bytes per token for V (de-transposed)
        uint8_t * warm_k    = nullptr;   // warm_slots * k_bytes, host RAM
        uint8_t * warm_v    = nullptr;   // warm_slots * v_bytes, host RAM (de-transposed)
#ifdef GGML_USE_HIP
        void * warm_k_dev   = nullptr;   // optional: device (eGPU) warm K buffer
        void * warm_v_dev   = nullptr;   // optional: device (eGPU) warm V buffer
#endif
    };
    std::vector<TieredKVLayer> kv_layers;

    // Warm slot occupancy (shared across all layers)
    struct WarmSlot { bool occupied = false; llama_pos pos = -1; };
    std::vector<WarmSlot>              warm_slots;
    std::unordered_map<llama_pos, int> warm_pos_to_slot;
    int  warm_alloc_slot();
    void warm_free_slot(llama_pos pos);

    // Per-layer warm copy helpers: VRAM ↔ RAM (or VRAM ↔ device)
    bool warm_copy_to_host(uint32_t il, int slot, llama_pos pos);
    bool warm_copy_from_host(uint32_t il, int slot, llama_pos pos);
#ifdef GGML_USE_HIP
    bool warm_copy_to_dev(uint32_t il, int slot, llama_pos pos);
    bool warm_copy_from_dev(uint32_t il, int slot, llama_pos pos);
#endif

    // Helper methods
    std::string get_ssd_file_path(llama_pos pos) const;
    bool save_to_ssd(const std::vector<llama_pos>& positions, const ggml_tensor* k_tensor, const ggml_tensor* v_tensor, bool is_device_data = false);
    bool load_from_ssd(const std::vector<llama_pos>& positions, ggml_tensor* k_tensor, ggml_tensor* v_tensor, bool to_device = false);
    
    // Tier capacity getters
    uint32_t get_tier_capacity(llama_cache_tier tier) const;
};
