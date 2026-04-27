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
    for (auto & tl : kv_layers) {
        delete[] tl.warm_k;
        delete[] tl.warm_v;
#ifdef GGML_USE_HIP
        if (tl.warm_k_dev) { hipFree(tl.warm_k_dev); }
        if (tl.warm_v_dev) { hipFree(tl.warm_v_dev); }
#endif
    }
}

bool llama_kv_cache_tiered::init() {
    // Create SSD directory if it doesn't exist
    std::filesystem::path ssd_dir(ssd_path);
    if (!std::filesystem::exists(ssd_dir)) {
        std::filesystem::create_directories(ssd_dir);
    }

    // Warm slot tracking — per-layer buffers allocated later in set_kv_layers_from_cache()
    if (config.warm_capacity() > 0) {
        warm_slots.assign(config.warm_capacity(), WarmSlot{});
        LLAMA_LOG_INFO("%s: warm tier: %u slots reserved (buffers wired after KV layers available)\n",
                       __func__, config.warm_capacity());
    }
    if (config.cold_capacity() > 0) {
        LLAMA_LOG_INFO("%s: cold tier: %u slots reserved (SSD path: %s)\n",
                       __func__, config.cold_capacity(), ssd_path.c_str());
    }

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

void llama_kv_cache_tiered::track_hot_range(uint32_t n_hot) {
    std::lock_guard<std::mutex> lock(mutex);
    for (uint32_t p = 0; p < n_hot; p++) {
        token_metadata.record_access((llama_pos)p);
    }
    stats.hot_tokens = n_hot;
}

bool llama_kv_cache_tiered::evict_tokens(uint32_t n_tokens_to_evict, llama_cache_tier from_tier) {
    std::lock_guard<std::mutex> lock(mutex);

    auto candidates = token_metadata.get_eviction_candidates(eviction_policy, n_tokens_to_evict);
    if (candidates.empty()) {
        return false;
    }

    // Apply semantic eviction weighting if we have a current query embedding
    if (!current_query_emb.empty() && !fingerprints.empty()) {
        std::vector<std::pair<llama_pos, float>> candidates_with_similarity;
        
        for (auto pos : candidates) {
            float similarity = 0.0f; // Default similarity if no fingerprint found
            
            // Find fingerprint for this position
            for (const auto& fp : fingerprints) {
                bool found = false;
                for (auto fp_pos : fp.positions) {
                    if (fp_pos == pos) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    // Compute cosine similarity (dot product since both are L2-normalized)
                    similarity = 0.0f;
                    for (size_t i = 0; i < current_query_emb.size() && i < fp.embedding.size(); i++) {
                        similarity += current_query_emb[i] * fp.embedding[i];
                    }
                    break;
                }
            }
            
            candidates_with_similarity.emplace_back(pos, similarity);
        }
        
        // Reorder candidates: positions with similarity < 0.3 evicted first,
        // positions with similarity > 0.65 moved to back of eviction queue
        std::sort(candidates_with_similarity.begin(), candidates_with_similarity.end(),
                  [](const auto& a, const auto& b) {
                      if (a.second > 0.65f) return false;  // Keep high similarity at back
                      if (b.second > 0.65f) return true;   // Keep high similarity at back
                      if (a.second < 0.3f) return true;    // Evict low similarity first
                      if (b.second < 0.3f) return false;   // Evict low similarity first
                      return a.second < b.second;          // Otherwise by similarity
                  });
        
        // Update candidates with reordered positions
        candidates.clear();
        for (const auto& candidate : candidates_with_similarity) {
            candidates.push_back(candidate.first);
        }
        
        // Count semantic eviction saves
        uint32_t saves = 0;
        for (const auto& candidate : candidates_with_similarity) {
            if (candidate.second > 0.65f) {
                saves++;
            }
        }
        stats.semantic_eviction_saves += saves;
    }

    std::vector<llama_pos> to_warm, to_cold;

    // Route to warm if slots available (RAM or GPU), otherwise cold
    int free_warm = 0;
    for (const auto & s : warm_slots) {
        if (!s.occupied) free_warm++;
    }
    for (auto pos : candidates) {
        if (free_warm > 0) {
            to_warm.push_back(pos);
            free_warm--;
        } else {
            to_cold.push_back(pos);
        }
    }

    // migrate_tokens handles null k_tensor/v_tensor as metadata-only (no-copy) path
    if (!to_warm.empty()) {
        migrate_tokens(to_warm, TIER_HOT, TIER_WARM);
        stats.warm_tokens += (uint32_t)to_warm.size();
        stats.hot_tokens   = stats.hot_tokens >= (uint32_t)to_warm.size()
                           ? stats.hot_tokens - (uint32_t)to_warm.size() : 0;
        LLAMA_LOG_INFO("%s: evicted %zu hot->warm\n", __func__, to_warm.size());
    }
    if (!to_cold.empty()) {
        migrate_tokens(to_cold, TIER_HOT, TIER_COLD);
        stats.cold_tokens += (uint32_t)to_cold.size();
        stats.hot_tokens   = stats.hot_tokens >= (uint32_t)to_cold.size()
                           ? stats.hot_tokens - (uint32_t)to_cold.size() : 0;
        LLAMA_LOG_INFO("%s: evicted %zu hot->cold\n", __func__, to_cold.size());
    }

    stats.eviction_count += candidates.size();
    return true;
}

// Warm slot management — unconditional (works for RAM and GPU warm tiers)
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

// Copy one token's K data from VRAM (or host) into warm slot — K is always contiguous
bool llama_kv_cache_tiered::warm_copy_to_host(uint32_t il, int slot, llama_pos pos) {
    if (il >= kv_layers.size() || slot < 0 || (size_t)slot >= warm_slots.size()) return false;
    const auto & tl = kv_layers[il];
    if (!tl.k || !tl.warm_k || !tl.v || !tl.warm_v) return false;

    const size_t elem_k = ggml_element_size(tl.k);
    const size_t n_embd_k = (size_t)tl.k->ne[0];
    uint8_t * dst_k = tl.warm_k + (size_t)slot * tl.k_bytes;

    // K: contiguous row at pos — single copy
#ifdef GGML_USE_HIP
    hipMemcpy(dst_k, (uint8_t *)tl.k->data + (size_t)pos * tl.k_bytes, tl.k_bytes, hipMemcpyDeviceToHost);
#else
    memcpy(dst_k, (uint8_t *)tl.k->data + (size_t)pos * tl.k_bytes, tl.k_bytes);
#endif

    const size_t elem_v = ggml_element_size(tl.v);
    const size_t n_embd_v = (size_t)tl.v->ne[0];
    uint8_t * dst_v = tl.warm_v + (size_t)slot * tl.v_bytes;

    if (!tl.v_trans) {
        // V: also contiguous when not transposed
#ifdef GGML_USE_HIP
        hipMemcpy(dst_v, (uint8_t *)tl.v->data + (size_t)pos * tl.v_bytes, tl.v_bytes, hipMemcpyDeviceToHost);
#else
        memcpy(dst_v, (uint8_t *)tl.v->data + (size_t)pos * tl.v_bytes, tl.v_bytes);
#endif
    } else {
        // V transposed: element j at (j*kv_size + pos)*elem_v — gather into contiguous warm buf
        // Use hipMemcpy2D (src column → dst row): copies n_embd_v elements, each elem_v bytes apart
#ifdef GGML_USE_HIP
        hipMemcpy2D(dst_v,                                          // dst (contiguous)
                    elem_v,                                          // dst pitch
                    (uint8_t *)tl.v->data + (size_t)pos * elem_v,  // src (column pos)
                    (size_t)tl.kv_size * elem_v,                    // src pitch (stride between rows)
                    elem_v,                                          // width per row
                    n_embd_v,                                        // number of rows
                    hipMemcpyDeviceToHost);
#else
        for (size_t j = 0; j < n_embd_v; j++) {
            memcpy(dst_v + j * elem_v,
                   (uint8_t *)tl.v->data + ((size_t)j * (size_t)tl.kv_size + (size_t)pos) * elem_v,
                   elem_v);
        }
#endif
    }
    return true;
}

// Restore one token's K/V from warm slot back into the hot VRAM tensor
bool llama_kv_cache_tiered::warm_copy_from_host(uint32_t il, int slot, llama_pos pos) {
    if (il >= kv_layers.size() || slot < 0 || (size_t)slot >= warm_slots.size()) return false;
    const auto & tl = kv_layers[il];
    if (!tl.k || !tl.warm_k || !tl.v || !tl.warm_v) return false;

    const size_t elem_k = ggml_element_size(tl.k);
    const size_t n_embd_k = (size_t)tl.k->ne[0];
    const uint8_t * src_k = tl.warm_k + (size_t)slot * tl.k_bytes;

#ifdef GGML_USE_HIP
    hipMemcpy((uint8_t *)tl.k->data + (size_t)pos * tl.k_bytes, src_k, tl.k_bytes, hipMemcpyHostToDevice);
#else
    memcpy((uint8_t *)tl.k->data + (size_t)pos * tl.k_bytes, src_k, tl.k_bytes);
#endif

    const size_t elem_v = ggml_element_size(tl.v);
    const size_t n_embd_v = (size_t)tl.v->ne[0];
    const uint8_t * src_v = tl.warm_v + (size_t)slot * tl.v_bytes;

    if (!tl.v_trans) {
#ifdef GGML_USE_HIP
        hipMemcpy((uint8_t *)tl.v->data + (size_t)pos * tl.v_bytes, src_v, tl.v_bytes, hipMemcpyHostToDevice);
#else
        memcpy((uint8_t *)tl.v->data + (size_t)pos * tl.v_bytes, src_v, tl.v_bytes);
#endif
    } else {
        // Scatter: warm contiguous → V transposed column at pos
#ifdef GGML_USE_HIP
        hipMemcpy2D((uint8_t *)tl.v->data + (size_t)pos * elem_v,  // dst column pos
                    (size_t)tl.kv_size * elem_v,                     // dst pitch
                    src_v,                                            // src (contiguous)
                    elem_v,                                           // src pitch
                    elem_v,                                           // width
                    n_embd_v,                                         // height
                    hipMemcpyHostToDevice);
#else
        for (size_t j = 0; j < n_embd_v; j++) {
            memcpy((uint8_t *)tl.v->data + ((size_t)j * (size_t)tl.kv_size + (size_t)pos) * elem_v,
                   src_v + j * elem_v,
                   elem_v);
        }
#endif
    }
    return true;
}

#ifdef GGML_USE_HIP
// VRAM (R9700) → device VRAM (6900XT warm tier): K contiguous, V gather/scatter
bool llama_kv_cache_tiered::warm_copy_to_dev(uint32_t il, int slot, llama_pos pos) {
    if (il >= kv_layers.size() || slot < 0 || (size_t)slot >= warm_slots.size()) return false;
    const auto & tl = kv_layers[il];
    if (!tl.warm_k_dev || !tl.warm_v_dev || !tl.k || !tl.v) return false;
    int prev_dev = 0; hipGetDevice(&prev_dev); hipSetDevice(config.warm_device);
    // K: contiguous D2D copy
    hipMemcpy((uint8_t *)tl.warm_k_dev + (size_t)slot * tl.k_bytes,
              (uint8_t *)tl.k->data   + (size_t)pos  * tl.k_bytes,
              tl.k_bytes, hipMemcpyDeviceToDevice);
    const size_t ev = ggml_element_size(tl.v);
    const size_t nv = (size_t)tl.v->ne[0]; // n_embd_v_gqa
    if (!tl.v_trans) {
        hipMemcpy((uint8_t *)tl.warm_v_dev + (size_t)slot * tl.v_bytes,
                  (uint8_t *)tl.v->data   + (size_t)pos  * tl.v_bytes,
                  tl.v_bytes, hipMemcpyDeviceToDevice);
    } else {
        // Gather column pos into contiguous warm buf
        hipMemcpy2D((uint8_t *)tl.warm_v_dev + (size_t)slot * tl.v_bytes, ev,
                    (uint8_t *)tl.v->data    + (size_t)pos  * ev,
                    (size_t)tl.kv_size * ev, ev, nv, hipMemcpyDeviceToDevice);
    }
    hipSetDevice(prev_dev);
    return true;
}

bool llama_kv_cache_tiered::warm_copy_from_dev(uint32_t il, int slot, llama_pos pos) {
    if (il >= kv_layers.size() || slot < 0 || (size_t)slot >= warm_slots.size()) return false;
    const auto & tl = kv_layers[il];
    if (!tl.warm_k_dev || !tl.warm_v_dev || !tl.k || !tl.v) return false;
    int prev_dev = 0; hipGetDevice(&prev_dev); hipSetDevice(config.warm_device);
    hipMemcpy((uint8_t *)tl.k->data    + (size_t)pos  * tl.k_bytes,
              (uint8_t *)tl.warm_k_dev + (size_t)slot * tl.k_bytes,
              tl.k_bytes, hipMemcpyDeviceToDevice);
    const size_t ev = ggml_element_size(tl.v);
    const size_t nv = (size_t)tl.v->ne[0];
    if (!tl.v_trans) {
        hipMemcpy((uint8_t *)tl.v->data    + (size_t)pos  * tl.v_bytes,
                  (uint8_t *)tl.warm_v_dev + (size_t)slot * tl.v_bytes,
                  tl.v_bytes, hipMemcpyDeviceToDevice);
    } else {
        hipMemcpy2D((uint8_t *)tl.v->data    + (size_t)pos  * ev,
                    (size_t)tl.kv_size * ev,
                    (uint8_t *)tl.warm_v_dev + (size_t)slot * tl.v_bytes, ev,
                    ev, nv, hipMemcpyDeviceToDevice);
    }
    hipSetDevice(prev_dev);
    return true;
}
#endif  // GGML_USE_HIP

void llama_kv_cache_tiered::set_kv_layers_from_cache(llama_kv_cache * cache) {
    if (!cache) return;
    std::lock_guard<std::mutex> lock(mutex);

    uint32_t n = cache->get_num_kv_layers();
    bool vtrans = cache->is_v_transposed();
    int64_t kvsz = (int64_t)cache->get_size();

    kv_layers.resize(n);
    for (uint32_t il = 0; il < n; il++) {
        auto & tl = kv_layers[il];
        tl.k      = cache->get_layer_k_raw(il);
        tl.v      = cache->get_layer_v_raw(il);
        tl.v_trans = vtrans;
        tl.kv_size = kvsz;
        // Use nb[1] (the actual row stride) for per-token byte size so that
        // block-quantized types (turbo4, Q4_K, etc.) are sized correctly.
        // ggml_element_size returns the block size, not bytes-per-element, so
        // ne[0]*element_size is wildly wrong for quantized caches.
        if (tl.k) tl.k_bytes = tl.k->nb[1];
        if (tl.v) tl.v_bytes = tl.v->nb[1];
    }

    // (Re)allocate per-layer warm buffers using already-reserved slot count
    uint32_t n_warm = (uint32_t)warm_slots.size();
    if (n_warm == 0 && config.warm_capacity() > 0) {
        n_warm = config.warm_capacity();
        warm_slots.assign(n_warm, WarmSlot{});
    }

    for (auto & tl : kv_layers) {
        delete[] tl.warm_k;
        delete[] tl.warm_v;
        tl.warm_k = tl.warm_v = nullptr;
        // Skip RAM staging buffers when a GPU warm device is configured — the
        // GPU VRAM path (warm_k_dev/warm_v_dev) is used instead, so allocating
        // n_warm * kv_bytes of RAM here would just trigger OOM for large contexts.
#ifndef GGML_USE_HIP
        if (n_warm > 0 && tl.k_bytes > 0) tl.warm_k = new uint8_t[n_warm * tl.k_bytes]();
        if (n_warm > 0 && tl.v_bytes > 0) tl.warm_v = new uint8_t[n_warm * tl.v_bytes]();
#else
        if (config.warm_device < 0) {
            if (n_warm > 0 && tl.k_bytes > 0) tl.warm_k = new uint8_t[n_warm * tl.k_bytes]();
            if (n_warm > 0 && tl.v_bytes > 0) tl.warm_v = new uint8_t[n_warm * tl.v_bytes]();
        }
#endif

#ifdef GGML_USE_HIP
        if (config.warm_device >= 0 && n_warm > 0) {
            hipSetDevice(config.warm_device);
            if (tl.warm_k_dev) hipFree(tl.warm_k_dev);
            if (tl.warm_v_dev) hipFree(tl.warm_v_dev);
            tl.warm_k_dev = tl.warm_v_dev = nullptr;
            if (tl.k_bytes > 0) hipMalloc(&tl.warm_k_dev, n_warm * tl.k_bytes);
            if (tl.v_bytes > 0) hipMalloc(&tl.warm_v_dev, n_warm * tl.v_bytes);
        }
#endif
    }

    size_t total_mb = 0;
    for (const auto & tl : kv_layers)
        total_mb += (tl.k_bytes + tl.v_bytes) * n_warm;
    total_mb /= (1024*1024);

#ifdef GGML_USE_HIP
    if (config.warm_device >= 0) {
        LLAMA_LOG_INFO("%s: wired %u KV layers (v_trans=%d, kv_size=%lld), warm GPU: ~%zu MiB (%u slots x %u layers)\n",
                       __func__, n, (int)vtrans, (long long)kvsz, total_mb, n_warm, n);
    } else {
        LLAMA_LOG_INFO("%s: wired %u KV layers (v_trans=%d, kv_size=%lld), warm RAM: ~%zu MiB (%u slots x %u layers)\n",
                       __func__, n, (int)vtrans, (long long)kvsz, total_mb, n_warm, n);
    }
#else
    LLAMA_LOG_INFO("%s: wired %u KV layers (v_trans=%d, kv_size=%lld), warm RAM: ~%zu MiB (%u slots x %u layers)\n",
                   __func__, n, (int)vtrans, (long long)kvsz, total_mb, n_warm, n);
#endif
#ifdef GGML_USE_HIP
    if (config.warm_device >= 0 && n_warm > 0 && !kv_layers.empty() && kv_layers[0].warm_k_dev) {
        LLAMA_LOG_INFO("llama_kv_cache_tiered:      ROCm%d KV warm buffer size = %6.2f MiB (%u slots x %u layers)\n",
                       config.warm_device, (float)total_mb, n_warm, n);
    }
#endif
}

bool llama_kv_cache_tiered::migrate_tokens(const std::vector<llama_pos>& positions,
                                            llama_cache_tier from_tier,
                                            llama_cache_tier to_tier,
                                            const ggml_tensor* /*unused_k*/,
                                            const ggml_tensor* /*unused_v*/) {
    auto start = std::chrono::high_resolution_clock::now();
    bool success = true;

    if (kv_layers.empty() || warm_slots.empty()) {
        // Metadata-only path: layer tensors not yet wired (set_kv_layers_from_cache not called)
        for (auto pos : positions) {
            token_metadata.record_access(pos);
        }
    } else {
        bool is_hot_to_warm  = (from_tier == TIER_HOT  && to_tier == TIER_WARM);
        bool is_warm_to_cold = (from_tier == TIER_WARM  && to_tier == TIER_COLD);
        bool is_cold_to_hot  = (from_tier == TIER_COLD  && to_tier == TIER_HOT);
        bool is_cold_to_warm = (from_tier == TIER_COLD  && to_tier == TIER_WARM);
        bool is_warm_to_hot  = (from_tier == TIER_WARM  && to_tier == TIER_HOT);

        if (is_hot_to_warm) {
            // VRAM → RAM (or eGPU): copy K/V for every layer and every evicted position
            for (auto pos : positions) {
                int slot = warm_alloc_slot();
                if (slot < 0) {
                    LLAMA_LOG_WARN("%s: warm full, dropping pos %d (cold SSD TODO)\n", __func__, pos);
                    token_metadata.record_access(pos);
                    continue;
                }
                warm_slots[slot].pos = pos;
                warm_pos_to_slot[pos] = slot;

                for (uint32_t il = 0; il < (uint32_t)kv_layers.size(); il++) {
#ifdef GGML_USE_HIP
                    if (kv_layers[il].warm_k_dev) {
                        success &= warm_copy_to_dev(il, slot, pos);
                    } else {
                        success &= warm_copy_to_host(il, slot, pos);
                    }
#else
                    success &= warm_copy_to_host(il, slot, pos);
#endif
                }
                token_metadata.record_access(pos);
            }
            if (success)
                LLAMA_LOG_INFO("%s: hot→warm: %zu positions x %zu layers\n",
                               __func__, positions.size(), kv_layers.size());
        } else if (is_warm_to_hot) {
            // RAM (or eGPU) → VRAM: restore
            for (auto pos : positions) {
                auto it = warm_pos_to_slot.find(pos);
                if (it == warm_pos_to_slot.end()) continue;
                int slot = it->second;
                for (uint32_t il = 0; il < (uint32_t)kv_layers.size(); il++) {
#ifdef GGML_USE_HIP
                    if (kv_layers[il].warm_k_dev) {
                        success &= warm_copy_from_dev(il, slot, pos);
                    } else {
                        success &= warm_copy_from_host(il, slot, pos);
                    }
#else
                    success &= warm_copy_from_host(il, slot, pos);
#endif
                }
                warm_free_slot(pos);
                token_metadata.record_access(pos);
            }
        } else if (is_warm_to_cold || is_cold_to_hot || is_cold_to_warm) {
            // SSD paths: TODO — per-layer serialization
            for (auto pos : positions) {
                token_metadata.record_access(pos);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    stats.total_migration_latency_us +=
        (double)std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return success;
}

bool llama_kv_cache_tiered::batch_migrate_tokens(const std::vector<llama_pos>& positions,
                                                  llama_cache_tier from_tier,
                                                  llama_cache_tier to_tier,
                                                  const ggml_tensor* k_tensor,
                                                  const ggml_tensor* v_tensor) {
    auto start = std::chrono::high_resolution_clock::now();

    // Delegate to migrate_tokens which handles the real per-layer data movement
    return migrate_tokens(positions, from_tier, to_tier);
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

void llama_kv_cache_tiered::add_fingerprint(const std::vector<llama_pos>& positions,
                                             const std::vector<float>& embedding,
                                             llama_cache_tier tier) {
    std::lock_guard<std::mutex> lock(mutex);
    
    semantic_fingerprint fp;
    fp.positions = positions;
    fp.embedding = embedding;
    fp.tier = tier;
    fp.turn = fingerprint_turn++;
    
    // Cap at 1000 entries - evict oldest when over cap
    if (fingerprints.size() >= 1000) {
        fingerprints.erase(fingerprints.begin());
    }
    
    fingerprints.push_back(fp);
}

std::vector<llama_kv_cache_tiered::PrefetchHint> llama_kv_cache_tiered::score_fingerprints(
    const std::vector<float>& query_emb, int top_k, float threshold) const {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::vector<PrefetchHint> results;
    
    if (fingerprints.empty() || query_emb.empty()) {
        return results;
    }
    
    // Compute cosine similarity between query_emb and each fingerprint
    // Since both are L2-normalized, cosine similarity = dot product
    for (const auto& fp : fingerprints) {
        // Compute dot product
        float dot_product = 0.0f;
        size_t min_size = std::min(query_emb.size(), fp.embedding.size());
        for (size_t i = 0; i < min_size; ++i) {
            dot_product += query_emb[i] * fp.embedding[i];
        }
        
        // Check if above threshold
        if (dot_product >= threshold) {
            PrefetchHint hint;
            hint.positions = fp.positions;
            hint.score = dot_product;
            hint.current_tier = fp.tier;
            results.push_back(hint);
        }
    }
    
    // Sort by score descending
    std::sort(results.begin(), results.end(),
              [](const PrefetchHint& a, const PrefetchHint& b) {
                  return a.score > b.score;
              });
    
    // Return top_k results
    if (static_cast<int>(results.size()) > top_k) {
        results.resize(top_k);
    }
    
    return results;
}

void llama_kv_cache_tiered::set_current_query_embedding(const std::vector<float>& emb) {
    std::lock_guard<std::mutex> lock(mutex);
    current_query_emb = emb;
}

bool llama_kv_cache_tiered::save_fingerprints_to_disk(const std::string& path) {
    std::lock_guard<std::mutex> lock(mutex);
    
    std::ofstream file(path, std::ofstream::binary);
    if (!file) {
        return false;
    }
    
    // Write number of entries
    uint32_t n_entries = uint32_t(fingerprints.size());
    file.write(reinterpret_cast<char*>(&n_entries), sizeof(uint32_t));
    
    // Write each fingerprint
    for (const auto& fp : fingerprints) {
        // Write positions count and positions
        uint32_t n_pos = uint32_t(fp.positions.size());
        file.write(reinterpret_cast<const char*>(&n_pos), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(fp.positions.data()), n_pos * sizeof(llama_pos));
        
        // Write embedding dimension and embedding
        uint32_t n_embd = uint32_t(fp.embedding.size());
        file.write(reinterpret_cast<const char*>(&n_embd), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(fp.embedding.data()), n_embd * sizeof(float));
        
        // Write tier and turn
        file.write(reinterpret_cast<const char*>(&fp.tier), sizeof(llama_cache_tier));
        file.write(reinterpret_cast<const char*>(&fp.turn), sizeof(uint64_t));
    }
    
    return file.good();
}

bool llama_kv_cache_tiered::load_fingerprints_from_disk(const std::string& path) {
    std::ifstream file(path, std::ifstream::binary);
    if (!file) {
        return false;
    }
    
    // Read number of entries
    uint32_t n_entries;
    file.read(reinterpret_cast<char*>(&n_entries), sizeof(uint32_t));
    
    // Clear existing fingerprints
    fingerprints.clear();
    
    // Read each fingerprint
    for (uint32_t i = 0; i < n_entries; i++) {
        semantic_fingerprint fp;
        
        // Read positions
        uint32_t n_pos;
        file.read(reinterpret_cast<char*>(&n_pos), sizeof(uint32_t));
        fp.positions.resize(n_pos);
        file.read(reinterpret_cast<char*>(fp.positions.data()), n_pos * sizeof(llama_pos));
        
        // Read embedding
        uint32_t n_embd;
        file.read(reinterpret_cast<char*>(&n_embd), sizeof(uint32_t));
        fp.embedding.resize(n_embd);
        file.read(reinterpret_cast<char*>(fp.embedding.data()), n_embd * sizeof(float));
        
        // Read tier and turn
        file.read(reinterpret_cast<char*>(&fp.tier), sizeof(llama_cache_tier));
        file.read(reinterpret_cast<char*>(&fp.turn), sizeof(uint64_t));
        
        fingerprints.push_back(fp);
    }
    
    return file.good();
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
