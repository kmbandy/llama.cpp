#include "llama-kv-cache-tiered.h"

#include "llama-impl.h"
#include "llama-io.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <cmath>
#include <limits>

#ifdef GGML_USE_HIP
#include <hip/hip_runtime.h>
#endif

//
// llama_ssd_storage_format
//

// Quantize float32 to int4 (4-bit)
std::vector<uint8_t> llama_ssd_storage_format::quantize_int4(const float* data, uint32_t n_elements) {
    std::vector<uint8_t> result((n_elements + 1) / 2);  // 2 values per byte

    for (uint32_t i = 0; i < n_elements; i += 2) {
        // Quantize to 4-bit range [-8, 7]
        int32_t v0 = std::max(-8, std::min(7, int(std::round(data[i] * 15.0f))));
        int32_t v1 = std::max(-8, std::min(7, int(std::round(data[i + 1] * 15.0f))));

        // Pack into single byte
        result[i / 2] = uint8_t((v0 & 0x0F) | ((v1 & 0x0F) << 4));
    }

    return result;
}

// Quantize float32 to int8 (8-bit)
std::vector<uint8_t> llama_ssd_storage_format::quantize_int8(const float* data, uint32_t n_elements) {
    std::vector<uint8_t> result(n_elements);

    for (uint32_t i = 0; i < n_elements; i++) {
        // Quantize to int8 range [-128, 127]
        result[i] = uint8_t(std::max(-128, std::min(127, int(std::round(data[i] * 127.0f)))) + 128);
    }

    return result;
}

// Dequantize int4 to float32
bool llama_ssd_storage_format::dequantize_int4(const uint8_t* data, float* out, uint32_t n_elements) {
    for (uint32_t i = 0; i < (n_elements + 1) / 2; i++) {
        uint8_t byte = data[i];
        int32_t v0 = (byte & 0x0F) - 8;  // Lower 4 bits
        int32_t v1 = ((byte >> 4) & 0x0F) - 8;  // Upper 4 bits

        out[2 * i] = float(v0) / 15.0f;
        if (2 * i + 1 < n_elements) {
            out[2 * i + 1] = float(v1) / 15.0f;
        }
    }

    return true;
}

// Dequantize int8 to float32
bool llama_ssd_storage_format::dequantize_int8(const uint8_t* data, float* out, uint32_t n_elements) {
    for (uint32_t i = 0; i < n_elements; i++) {
        out[i] = float(data[i] - 128) / 127.0f;
    }

    return true;
}

bool llama_ssd_storage_format::write(const std::string& path,
                                       const float* k_data,
                                       const float* v_data,
                                       uint32_t n_tokens,
                                       uint32_t n_layers,
                                       uint32_t n_embd_k,
                                       uint32_t n_embd_v,
                                       llama_cache_compression compression) {
    std::ofstream file(path, std::ofstream::binary);
    if (!file) {
        return false;
    }

    // Calculate data size based on compression
    size_t element_size = sizeof(float);
    if (compression == COMPRESSION_INT4) {
        element_size = sizeof(uint8_t) / 2;  // 4-bit = 0.5 bytes per element
    } else if (compression == COMPRESSION_INT8) {
        element_size = sizeof(uint8_t);  // 8-bit = 1 byte per element
    }

    size_t total_elements = n_tokens * n_layers * (n_embd_k + n_embd_v);
    size_t compressed_size = total_elements * element_size;

    // Write header
    file_header header;
    header.magic = MAGIC;
    header.version = VERSION;
    header.n_layers = n_layers;
    header.n_embd_k = n_embd_k;
    header.n_embd_v = n_embd_v;
    header.n_tokens = n_tokens;
    header.compression_type = uint32_t(compression);
    header.layer_offset = 0;
    header.attention_threshold = 0.0f;
    header.index_offset = 0;  // Will be filled later
    header.index_size = 0;    // Will be filled later

    file.write(reinterpret_cast<char*>(&header), sizeof(file_header));

    // Write data with compression
    if (compression == COMPRESSION_INT4) {
        // Quantize and write K data
        auto k_quant = quantize_int4(k_data, n_tokens * n_layers * n_embd_k);
        uint32_t k_size = uint32_t(k_quant.size());
        file.write(reinterpret_cast<char*>(&k_size), sizeof(uint32_t));
        file.write(reinterpret_cast<char*>(k_quant.data()), k_quant.size());

        // Quantize and write V data
        auto v_quant = quantize_int4(v_data, n_tokens * n_layers * n_embd_v);
        uint32_t v_size = uint32_t(v_quant.size());
        file.write(reinterpret_cast<char*>(&v_size), sizeof(uint32_t));
        file.write(reinterpret_cast<char*>(v_quant.data()), v_quant.size());
    } else if (compression == COMPRESSION_INT8) {
        // Quantize and write K data
        auto k_quant = quantize_int8(k_data, n_tokens * n_layers * n_embd_k);
        uint32_t k_size = uint32_t(k_quant.size());
        file.write(reinterpret_cast<char*>(&k_size), sizeof(uint32_t));
        file.write(reinterpret_cast<char*>(k_quant.data()), k_quant.size());

        // Quantize and write V data
        auto v_quant = quantize_int8(v_data, n_tokens * n_layers * n_embd_v);
        uint32_t v_size = uint32_t(v_quant.size());
        file.write(reinterpret_cast<char*>(&v_size), sizeof(uint32_t));
        file.write(reinterpret_cast<char*>(v_quant.data()), v_quant.size());
    } else {
        // No compression - write raw floats
        size_t k_size = n_tokens * n_layers * n_embd_k * sizeof(float);
        size_t v_size = n_tokens * n_layers * n_embd_v * sizeof(float);
        file.write(const_cast<char*>(reinterpret_cast<const char*>(k_data)), k_size);
        file.write(const_cast<char*>(reinterpret_cast<const char*>(v_data)), v_size);
    }

    return file.good();
}

bool llama_ssd_storage_format::read(const std::string& path,
                                      float* k_data,
                                      float* v_data,
                                      uint32_t n_tokens,
                                      uint32_t n_layers,
                                      uint32_t n_embd_k,
                                      uint32_t n_embd_v) {
    std::ifstream file(path, std::ifstream::binary);
    if (!file) {
        return false;
    }

    // Read and validate header
    file_header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(file_header));

    if (header.magic != MAGIC || header.version != VERSION) {
        return false;
    }

    // Read data based on compression type
    if (header.compression_type == COMPRESSION_INT4) {
        // Read K data
        uint32_t k_size;
        file.read(reinterpret_cast<char*>(&k_size), sizeof(uint32_t));
        std::vector<uint8_t> k_quant(k_size);
        file.read(reinterpret_cast<char*>(k_quant.data()), k_size);
        dequantize_int4(k_quant.data(), k_data, n_tokens * n_layers * n_embd_k);

        // Read V data
        uint32_t v_size;
        file.read(reinterpret_cast<char*>(&v_size), sizeof(uint32_t));
        std::vector<uint8_t> v_quant(v_size);
        file.read(reinterpret_cast<char*>(v_quant.data()), v_size);
        dequantize_int4(v_quant.data(), v_data, n_tokens * n_layers * n_embd_v);
    } else if (header.compression_type == COMPRESSION_INT8) {
        // Read K data
        uint32_t k_size;
        file.read(reinterpret_cast<char*>(&k_size), sizeof(uint32_t));
        std::vector<uint8_t> k_quant(k_size);
        file.read(reinterpret_cast<char*>(k_quant.data()), k_size);
        dequantize_int8(k_quant.data(), k_data, n_tokens * n_layers * n_embd_k);

        // Read V data
        uint32_t v_size;
        file.read(reinterpret_cast<char*>(&v_size), sizeof(uint32_t));
        std::vector<uint8_t> v_quant(v_size);
        file.read(reinterpret_cast<char*>(v_quant.data()), v_size);
        dequantize_int8(v_quant.data(), v_data, n_tokens * n_layers * n_embd_v);
    } else {
        // No compression - read raw floats
        size_t k_size = n_tokens * n_layers * n_embd_k * sizeof(float);
        size_t v_size = n_tokens * n_layers * n_embd_v * sizeof(float);
        file.read(reinterpret_cast<char*>(k_data), k_size);
        file.read(reinterpret_cast<char*>(v_data), v_size);
    }

    return file.good();
}

bool llama_ssd_storage_format::build_index(const std::string& index_path,
                                            const std::string& data_path,
                                            uint32_t n_layers) {
    // Build layer index for fast lookup
    std::ifstream data_file(data_path, std::ifstream::binary);
    if (!data_file) {
        return false;
    }

    // Read header to get file structure
    file_header header;
    data_file.read(reinterpret_cast<char*>(&header), sizeof(file_header));

    // Build index entries for each layer
    layer_index.clear();
    for (uint32_t layer = 0; layer < n_layers; layer++) {
        layer_index_entry entry;
        entry.layer = layer;
        entry.file_offset = sizeof(file_header) + layer * (header.n_embd_k + header.n_embd_v) * sizeof(float);
        entry.n_tokens = header.n_tokens;
        entry.data_size = header.n_tokens * (header.n_embd_k + header.n_embd_v) * sizeof(float);
        layer_index.push_back(entry);
    }

    // Save index to file
    return save_index(index_path);
}

bool llama_ssd_storage_format::load_index(const std::string& index_path) {
    std::ifstream file(index_path, std::ifstream::binary);
    if (!file) {
        return false;
    }

    // Read number of entries
    uint32_t n_entries;
    file.read(reinterpret_cast<char*>(&n_entries), sizeof(uint32_t));

    // Read each entry
    layer_index.clear();
    for (uint32_t i = 0; i < n_entries; i++) {
        layer_index_entry entry;
        file.read(reinterpret_cast<char*>(&entry), sizeof(layer_index_entry));
        layer_index.push_back(entry);
    }

    return file.good();
}

bool llama_ssd_storage_format::save_index(const std::string& index_path) {
    std::ofstream file(index_path, std::ofstream::binary);
    if (!file) {
        return false;
    }

    // Write number of entries
    uint32_t n_entries = uint32_t(layer_index.size());
    file.write(reinterpret_cast<char*>(&n_entries), sizeof(uint32_t));

    // Write each entry
    for (const auto& entry : layer_index) {
        file.write(reinterpret_cast<const char*>(&entry), sizeof(layer_index_entry));
    }

    return file.good();
}

uint64_t llama_ssd_storage_format::get_layer_offset(uint32_t layer) const {
    for (const auto& entry : layer_index) {
        if (entry.layer == layer) {
            return entry.file_offset;
        }
    }
    return 0;  // Not found
}

bool llama_ssd_storage_format::has_layer(uint32_t layer) const {
    for (const auto& entry : layer_index) {
        if (entry.layer == layer) {
            return true;
        }
    }
    return false;
}

//
// llama_kv_cache_tiered
//

llama_kv_cache_tiered::llama_kv_cache_tiered(
    const llama_model& model,
    const llama_tier_config& config,
    const std::string& ssd_path,
    llama_eviction_policy eviction_policy,
    llama_cache_compression compression,
    float attention_threshold)
    : config(config),
      eviction_policy(eviction_policy),
      ssd_path(ssd_path),
      compression(compression),
      attention_threshold(attention_threshold) {
    stats.reset();
}

llama_kv_cache_tiered::~llama_kv_cache_tiered() {
    // Cleanup
}

bool llama_kv_cache_tiered::init() {
    // Create SSD directory if it doesn't exist
    std::filesystem::path ssd_dir(ssd_path);
    if (!std::filesystem::exists(ssd_dir)) {
        std::filesystem::create_directories(ssd_dir);
    }

#ifdef GGML_USE_HIP
    if (config.warm_device >= 0) {
        // Allocate warm tier buffers on 6900XT (or whatever warm_device is)
        int prev_dev = 0;
        hipGetDevice(&prev_dev);
        hipSetDevice(config.warm_device);

        // Each warm slot holds n_kv_heads * head_dim elements per position.
        // We allocate config.warm_ctx slots; each slot is warm_elem_bytes bytes
        // for K and the same for V.
        // warm_elem_bytes is set by the caller via set_warm_elem_bytes().
        if (warm_elem_bytes > 0 && config.warm_capacity() > 0) {
            warm_dev_capacity = warm_elem_bytes * config.warm_capacity();
            hipError_t err_k = hipMalloc(&warm_k_dev, warm_dev_capacity);
            hipError_t err_v = hipMalloc(&warm_v_dev, warm_dev_capacity);
            if (err_k != hipSuccess || err_v != hipSuccess) {
                LLAMA_LOG_WARN("%s: hipMalloc warm tier on device %d failed, falling back to SSD\n",
                               __func__, config.warm_device);
                if (warm_k_dev) { hipFree(warm_k_dev); warm_k_dev = nullptr; }
                if (warm_v_dev) { hipFree(warm_v_dev); warm_v_dev = nullptr; }
                warm_dev_capacity = 0;
            } else {
                warm_slots.assign(config.warm_capacity(), WarmSlot{});
                LLAMA_LOG_INFO("%s: warm tier on device %d: %.1f MB K + %.1f MB V\n",
                               __func__, config.warm_device,
                               warm_dev_capacity / 1e6, warm_dev_capacity / 1e6);
            }
        }
        hipSetDevice(prev_dev);
    }
#endif

    return true;
}

bool llama_kv_cache_tiered::resize(uint32_t new_total_ctx) {
    std::lock_guard<std::mutex> lock(mutex);
    config.total_ctx = new_total_ctx;
    return true;
}

bool llama_kv_cache_tiered::set_eviction_policy(llama_eviction_policy policy) {
    std::lock_guard<std::mutex> lock(mutex);
    eviction_policy = policy;
    return true;
}

bool llama_kv_cache_tiered::set_attention_threshold(float threshold) {
    std::lock_guard<std::mutex> lock(mutex);
    attention_threshold = threshold;
    return true;
}

bool llama_kv_cache_tiered::evict_tokens(uint32_t n_tokens_to_evict, llama_cache_tier from_tier) {
    std::lock_guard<std::mutex> lock(mutex);

    // Get eviction candidates based on policy
    auto candidates = token_metadata.get_eviction_candidates(
        eviction_policy,
        n_tokens_to_evict);

    // Evict tokens
    for (auto pos : candidates) {
        // Move to cold tier or remove
        // Implementation depends on tier architecture
    }

    stats.eviction_count += candidates.size();
    return true;
}

#ifdef GGML_USE_HIP
int llama_kv_cache_tiered::warm_alloc_slot() {
    for (int i = 0; i < (int)warm_slots.size(); i++) {
        if (!warm_slots[i].occupied) {
            warm_slots[i].occupied = true;
            return i;
        }
    }
    return -1;
}

void llama_kv_cache_tiered::warm_free_slot(llama_pos pos) {
    auto it = warm_pos_to_slot.find(pos);
    if (it != warm_pos_to_slot.end()) {
        int slot = it->second;
        warm_slots[slot].occupied = false;
        warm_slots[slot].pos = -1;
        warm_pos_to_slot.erase(it);
    }
}

bool llama_kv_cache_tiered::warm_copy_to_device(int slot, const void * k_host, const void * v_host, size_t nbytes) {
    if (!warm_k_dev || !warm_v_dev || slot < 0 || (size_t)slot >= warm_slots.size()) return false;
    int prev_dev = 0;
    hipGetDevice(&prev_dev);
    hipSetDevice(config.warm_device);
    size_t offset = (size_t)slot * nbytes;
    bool ok = (hipMemcpy((char*)warm_k_dev + offset, k_host, nbytes, hipMemcpyHostToDevice) == hipSuccess) &&
              (hipMemcpy((char*)warm_v_dev + offset, v_host, nbytes, hipMemcpyHostToDevice) == hipSuccess);
    hipSetDevice(prev_dev);
    return ok;
}

bool llama_kv_cache_tiered::warm_copy_from_device(int slot, void * k_host, void * v_host, size_t nbytes) {
    if (!warm_k_dev || !warm_v_dev || slot < 0 || (size_t)slot >= warm_slots.size()) return false;
    int prev_dev = 0;
    hipGetDevice(&prev_dev);
    hipSetDevice(config.warm_device);
    size_t offset = (size_t)slot * nbytes;
    bool ok = (hipMemcpy(k_host, (char*)warm_k_dev + offset, nbytes, hipMemcpyDeviceToHost) == hipSuccess) &&
              (hipMemcpy(v_host, (char*)warm_v_dev + offset, nbytes, hipMemcpyDeviceToHost) == hipSuccess);
    hipSetDevice(prev_dev);
    return ok;
}
#endif  // GGML_USE_HIP

bool llama_kv_cache_tiered::migrate_tokens(const std::vector<llama_pos>& positions,
                                            llama_cache_tier from_tier,
                                            llama_cache_tier to_tier,
                                            const ggml_tensor* k_tensor,
                                            const ggml_tensor* v_tensor) {
    auto start = std::chrono::high_resolution_clock::now();

    // Migration logic based on tier direction
    // - Hot (VRAM) to Warm (RAM): copy with compression
    // - Warm (RAM) to Cold (SSD): compress and write to SSD
    // - Cold (SSD) to Warm/Hot: decompress and load to RAM/VRAM
    
    // Check if we have tensor pointers for actual data movement
    if (k_tensor == nullptr || v_tensor == nullptr) {
        // Metadata-only migration (no actual data movement)
        for (auto pos : positions) {
            token_metadata.record_access(pos);
        }
    } else {
        // Actual data migration with tensor pointers
        // Determine source and destination based on tier
        bool is_hot_to_warm = (from_tier == TIER_HOT && to_tier == TIER_WARM);
        bool is_warm_to_cold = (from_tier == TIER_WARM && to_tier == TIER_COLD);
        bool is_cold_to_warm = (from_tier == TIER_COLD && to_tier == TIER_WARM);
        bool is_cold_to_hot = (from_tier == TIER_COLD && to_tier == TIER_HOT);
        
        if (is_warm_to_cold) {
            // Warm→Cold: serialize to SSD
            save_to_ssd(positions, k_tensor, v_tensor, false);
        } else if (is_hot_to_warm) {
#ifdef GGML_USE_HIP
            if (warm_k_dev && warm_elem_bytes > 0) {
                // R9700 VRAM → host staging → 6900XT VRAM
                size_t nbytes = warm_elem_bytes;
                std::vector<uint8_t> k_buf(nbytes), v_buf(nbytes);
                // copy device→host on current (hot) device
                hipMemcpy(k_buf.data(), k_tensor->data, nbytes, hipMemcpyDeviceToHost);
                hipMemcpy(v_buf.data(), v_tensor->data, nbytes, hipMemcpyDeviceToHost);
                for (auto pos : positions) {
                    int slot = warm_alloc_slot();
                    if (slot < 0) {
                        // warm tier full — spill to SSD
                        save_to_ssd({pos}, k_tensor, v_tensor, true);
                        continue;
                    }
                    warm_copy_to_device(slot, k_buf.data(), v_buf.data(), nbytes);
                    warm_slots[slot].pos = pos;
                    warm_pos_to_slot[pos] = slot;
                }
            } else
#endif
            {
                save_to_ssd(positions, k_tensor, v_tensor, true);
            }
        } else if (is_cold_to_warm || is_cold_to_hot) {
            load_from_ssd(positions, const_cast<ggml_tensor*>(k_tensor), const_cast<ggml_tensor*>(v_tensor), false);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.total_migration_latency_us += duration.count();

    return true;
}

bool llama_kv_cache_tiered::batch_migrate_tokens(const std::vector<llama_pos>& positions,
                                                  llama_cache_tier from_tier,
                                                  llama_cache_tier to_tier,
                                                  const ggml_tensor* k_tensor,
                                                  const ggml_tensor* v_tensor) {
    auto start = std::chrono::high_resolution_clock::now();

    // Batch migration optimization:
    // - Group tokens by layer for efficient I/O
    // - Compress data before transfer if moving to colder tier
    // - Decompress data when loading from colder tier
    
    if (k_tensor == nullptr || v_tensor == nullptr) {
        // Metadata-only batch migration
        for (auto pos : positions) {
            token_metadata.record_access(pos);
        }
    } else {
        // Batch migration with actual data movement
        // Group by layer for efficient I/O
        // For AMD ROCm/HIP: use hipMemcpy for device-to-host or host-to-device transfers
        
        bool is_warm_to_cold = (from_tier == TIER_WARM && to_tier == TIER_COLD);
        bool is_hot_to_warm = (from_tier == TIER_HOT && to_tier == TIER_WARM);
        
        if (is_warm_to_cold) {
            save_to_ssd(positions, k_tensor, v_tensor, false);
        } else if (is_hot_to_warm) {
#ifdef GGML_USE_HIP
            if (warm_k_dev && warm_elem_bytes > 0) {
                size_t nbytes = warm_elem_bytes;
                std::vector<uint8_t> k_buf(nbytes), v_buf(nbytes);
                hipMemcpy(k_buf.data(), k_tensor->data, nbytes, hipMemcpyDeviceToHost);
                hipMemcpy(v_buf.data(), v_tensor->data, nbytes, hipMemcpyDeviceToHost);
                for (auto pos : positions) {
                    int slot = warm_alloc_slot();
                    if (slot < 0) {
                        save_to_ssd({pos}, k_tensor, v_tensor, true);
                        continue;
                    }
                    warm_copy_to_device(slot, k_buf.data(), v_buf.data(), nbytes);
                    warm_slots[slot].pos = pos;
                    warm_pos_to_slot[pos] = slot;
                }
            } else
#endif
            {
                save_to_ssd(positions, k_tensor, v_tensor, true);
            }
        } else {
            load_from_ssd(positions, const_cast<ggml_tensor*>(k_tensor), const_cast<ggml_tensor*>(v_tensor), false);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    stats.total_migration_latency_us += duration.count();

    return true;
}

bool llama_kv_cache_tiered::prefetch_tokens(const std::vector<llama_pos>& positions,
                                             llama_cache_tier target_tier) {
    // Prefetch optimization:
    // - Pre-load likely-needed tokens before they're requested
    // - Use attention patterns to predict which tokens will be needed
    // - Batch prefetch requests for efficiency

    // TODO: Implement attention-based prediction for prefetch
    // TODO: Implement batch prefetch with priority queue

    return true;
}

bool llama_kv_cache_tiered::contains_token(llama_pos pos) const {
    std::lock_guard<std::mutex> lock(mutex);
    return token_metadata.get_metadata(pos) != nullptr;
}

bool llama_kv_cache_tiered::load_token(llama_pos pos, llama_cache_tier target_tier) {
    // Check if token is in cold tier and needs to be loaded
    std::string ssd_file = get_ssd_file_path(pos);
    if (std::filesystem::exists(ssd_file)) {
        // Load from SSD
        stats.cache_hits++;
        return true;
    }
    stats.cache_misses++;
    return false;
}

llama_tier_stats llama_kv_cache_tiered::get_stats() const {
    std::lock_guard<std::mutex> lock(mutex);
    return stats;
}

void llama_kv_cache_tiered::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex);
    stats.reset();
}

std::string llama_kv_cache_tiered::get_ssd_file_path(llama_pos pos) const {
    // Generate unique file path for token position
    return ssd_path + "/token_" + std::to_string(pos) + ".bin";
}

bool llama_kv_cache_tiered::save_to_ssd(const std::vector<llama_pos>& positions,
                                         const ggml_tensor* k_tensor,
                                         const ggml_tensor* v_tensor,
                                         bool is_device_data) {
    // Save KV cache data to SSD using the SSD storage format
    if (positions.empty() || k_tensor == nullptr || v_tensor == nullptr) {
        return false;
    }
    
    // Calculate number of tokens and layer info
    uint32_t n_tokens = uint32_t(positions.size());
    uint32_t n_layers = 32;  // Default placeholder - should come from model config
    uint32_t n_embd_k = k_tensor->ne[0];
    uint32_t n_embd_v = v_tensor->ne[0];
    
    // Calculate buffer size for device-to-host copy if needed
    // This is a simplified calculation - actual implementation should use model-specific sizes
    size_t buffer_size = n_tokens * n_embd_k * sizeof(float);
    size_t buffer_size_v = n_tokens * n_embd_v * sizeof(float);
    
    // Use the SSD storage format to write data
    llama_ssd_storage_format ssd;
    std::string base_path = ssd_path + "/cache";
    
#ifdef GGML_USE_HIP
    // If data is on device, we need to copy to host first
    if (is_device_data) {
        // Allocate host buffers for staging device data
        std::vector<float> k_host_buf(n_tokens * n_embd_k);
        std::vector<float> v_host_buf(n_tokens * n_embd_v);
        
        // Copy device data to host buffer using hipMemcpy
        hipError_t err_k = hipMemcpy(k_host_buf.data(),
                                      (const void*)k_tensor->data,
                                      buffer_size,
                                      hipMemcpyDeviceToHost);
        hipError_t err_v = hipMemcpy(v_host_buf.data(),
                                      (const void*)v_tensor->data,
                                      buffer_size_v,
                                      hipMemcpyDeviceToHost);
        
        if (err_k != hipSuccess || err_v != hipSuccess) {
            return false;
        }
        
        // Write K and V data for each position using host-staged data
        for (auto pos : positions) {
            std::string file_path = base_path + "_pos_" + std::to_string(pos) + ".bin";
            bool success = ssd.write(file_path,
                                     k_host_buf.data(),
                                     v_host_buf.data(),
                                     n_tokens, n_layers, n_embd_k, n_embd_v, compression);
            if (!success) {
                return false;
            }
        }
        return true;
    }
#endif
    
    // Non-device data path (RAM/SSD) - use tensor data directly
    const float* k_data = (const float*)k_tensor->data;
    const float* v_data = (const float*)v_tensor->data;
    
    // Write K and V data for each position
    for (auto pos : positions) {
        std::string file_path = base_path + "_pos_" + std::to_string(pos) + ".bin";
        bool success = ssd.write(file_path, k_data, v_data, n_tokens, n_layers, n_embd_k, n_embd_v, compression);
        if (!success) {
            return false;
        }
    }
    
    return true;
}

bool llama_kv_cache_tiered::load_from_ssd(const std::vector<llama_pos>& positions,
                                           ggml_tensor* k_tensor,
                                           ggml_tensor* v_tensor,
                                           bool to_device) {
    // Load KV cache data from SSD using the SSD storage format
    if (positions.empty() || k_tensor == nullptr || v_tensor == nullptr) {
        return false;
    }
    
    // Calculate number of tokens and layer info
    uint32_t n_tokens = uint32_t(positions.size());
    uint32_t n_layers = 32;  // Default placeholder - should come from model config
    uint32_t n_embd_k = k_tensor->ne[0];
    uint32_t n_embd_v = v_tensor->ne[0];
    
    // Calculate buffer size for host-to-device copy if needed
    size_t buffer_size = n_tokens * n_embd_k * sizeof(float);
    size_t buffer_size_v = n_tokens * n_embd_v * sizeof(float);
    
    // Use the SSD storage format to read data
    llama_ssd_storage_format ssd;
    std::string base_path = ssd_path + "/cache";
    
#ifdef GGML_USE_HIP
    // If destination is device, we need to copy from host to device
    if (to_device) {
        // Allocate host buffer for staging before device copy
        std::vector<float> k_host_buf(n_tokens * n_embd_k);
        std::vector<float> v_host_buf(n_tokens * n_embd_v);
        
        // Read K and V data for each position into host buffer
        for (auto pos : positions) {
            std::string file_path = base_path + "_pos_" + std::to_string(pos) + ".bin";
            if (!ssd.read(file_path, k_host_buf.data(), v_host_buf.data(), n_tokens, n_layers, n_embd_k, n_embd_v)) {
                return false;
            }
        }
        
        // Copy host data to device using hipMemcpy
        hipError_t err_k = hipMemcpy((void*)k_tensor->data,
                                      k_host_buf.data(),
                                      buffer_size,
                                      hipMemcpyHostToDevice);
        hipError_t err_v = hipMemcpy((void*)v_tensor->data,
                                      v_host_buf.data(),
                                      buffer_size_v,
                                      hipMemcpyHostToDevice);
        
        return (err_k == hipSuccess && err_v == hipSuccess);
    }
#endif
    
    // Non-device path - read directly to tensor data
    float* k_data = (float*)k_tensor->data;
    float* v_data = (float*)v_tensor->data;
    
    // Read K and V data for each position
    for (auto pos : positions) {
        std::string file_path = base_path + "_pos_" + std::to_string(pos) + ".bin";
        if (!ssd.read(file_path, k_data, v_data, n_tokens, n_layers, n_embd_k, n_embd_v)) {
            return false;
        }
    }
    
    return true;
}
