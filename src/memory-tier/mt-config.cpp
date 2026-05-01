#include "mt-config.h"

#include <cstdio>

namespace mt {

bool validate(const TieredConfig & cfg, std::string * err_out) {
    auto fail = [&](const char * msg) -> bool {
        if (err_out) *err_out = msg;
        return false;
    };

    if (cfg.hot_pct < 0.0f || cfg.hot_pct > 100.0f)   return fail("hot_pct out of [0, 100]");
    if (cfg.warm_pct < 0.0f || cfg.warm_pct > 100.0f) return fail("warm_pct out of [0, 100]");
    if (cfg.cold_pct < 0.0f || cfg.cold_pct > 100.0f) return fail("cold_pct out of [0, 100]");

    const float sum = cfg.hot_pct + cfg.warm_pct + cfg.cold_pct;
    if (sum < 99.0f || sum > 101.0f) {
        return fail("hot_pct + warm_pct + cold_pct must sum to ~100");
    }

    if (cfg.attention_threshold < 0.0f || cfg.attention_threshold > 1.0f) {
        return fail("attention_threshold out of [0, 1]");
    }

    if (cfg.semantic_threshold < 0.0f || cfg.semantic_threshold > 1.0f) {
        return fail("semantic_threshold out of [0, 1]");
    }

    if (cfg.semantic_top_k < 0) {
        return fail("semantic_top_k must be non-negative");
    }

    return true;
}

namespace {
const char * eviction_name(EvictionPolicy p) {
    switch (p) {
        case EvictionPolicy::LRU:       return "LRU";
        case EvictionPolicy::LFU:       return "LFU";
        case EvictionPolicy::Attention: return "Attention";
        case EvictionPolicy::Hybrid:    return "Hybrid";
    }
    return "?";
}
const char * compression_name(Compression c) {
    switch (c) {
        case Compression::None:      return "none";
        case Compression::Int4:      return "int4";
        case Compression::Int8:      return "int8";
        case Compression::LZ4:       return "lz4";
        case Compression::Quantized: return "quantized";
    }
    return "?";
}
}  // namespace

std::string describe(const TieredConfig & cfg) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "tiers=[hot %.0f%%, warm %.0f%%, cold %.0f%%] total_ctx=%u "
                  "policy=%s compression=%s attn_thresh=%.2f warm_dev=%d "
                  "semantic=%s ssd=%s",
                  cfg.hot_pct, cfg.warm_pct, cfg.cold_pct, cfg.total_ctx,
                  eviction_name(cfg.eviction), compression_name(cfg.compression),
                  cfg.attention_threshold, cfg.warm_device,
                  cfg.semantic_index.empty() ? "off" : cfg.semantic_index.c_str(),
                  cfg.ssd_path.c_str());
    return std::string(buf);
}

}  // namespace mt
