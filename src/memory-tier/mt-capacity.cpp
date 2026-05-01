#include "mt-capacity.h"

#include <algorithm>

namespace mt {

namespace {
constexpr float kPressureThreshold = 0.80f;  // start evicting when hot ≥ 80%
constexpr float kPressureTarget    = 0.60f;  // evict down to ~60%

uint32_t pct_of(uint32_t total, float pct) {
    if (total == 0 || pct <= 0.0f) return 0;
    return (uint32_t)(((double) total) * (double) pct / 100.0);
}
}  // namespace

TierCapacityManager::TierCapacityManager(const TieredConfig & cfg) : cfg_(cfg) {}

void TierCapacityManager::reconfigure(const TieredConfig & cfg) {
    std::lock_guard<std::mutex> lk(mu_);
    cfg_ = cfg;
    // Counters preserved (the actual tokens didn't move); high-water marks
    // also preserved so cross-config telemetry survives.
}

void TierCapacityManager::on_insert_hot(uint32_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    s_.hot_tokens += n;
    s_.hot_high_water = std::max(s_.hot_high_water, s_.hot_tokens);
}

void TierCapacityManager::on_remove_hot(uint32_t n) {
    std::lock_guard<std::mutex> lk(mu_);
    s_.hot_tokens = (n >= s_.hot_tokens) ? 0 : (s_.hot_tokens - n);
}

void TierCapacityManager::on_migrate(uint32_t n, Tier from, Tier to) {
    if (n == 0 || from == to) return;
    std::lock_guard<std::mutex> lk(mu_);

    auto add_to = [&](Tier t, uint32_t k) {
        switch (t) {
            case Tier::Hot:  s_.hot_tokens  += k; s_.hot_high_water  = std::max(s_.hot_high_water,  s_.hot_tokens);  break;
            case Tier::Warm: s_.warm_tokens += k; s_.warm_high_water = std::max(s_.warm_high_water, s_.warm_tokens); break;
            case Tier::Cold: s_.cold_tokens += k; s_.cold_high_water = std::max(s_.cold_high_water, s_.cold_tokens); break;
        }
    };
    auto sub_from = [&](Tier t, uint32_t k) {
        switch (t) {
            case Tier::Hot:  s_.hot_tokens  = (k >= s_.hot_tokens)  ? 0 : (s_.hot_tokens  - k); break;
            case Tier::Warm: s_.warm_tokens = (k >= s_.warm_tokens) ? 0 : (s_.warm_tokens - k); break;
            case Tier::Cold: s_.cold_tokens = (k >= s_.cold_tokens) ? 0 : (s_.cold_tokens - k); break;
        }
    };

    sub_from(from, n);
    add_to(to, n);

    if (from == Tier::Hot  && to == Tier::Warm) s_.hot_to_warm_count   += n;
    if (from == Tier::Hot  && to == Tier::Cold) s_.hot_to_cold_count   += n;
    if (from == Tier::Warm && to == Tier::Cold) s_.warm_to_cold_count  += n;
    if (from == Tier::Cold && to == Tier::Warm) s_.cold_to_warm_count  += n;
    if (from == Tier::Warm && to == Tier::Hot)  s_.warm_to_hot_count   += n;
    if (from == Tier::Cold && to == Tier::Hot)  s_.cold_to_hot_count   += n;
}

bool TierCapacityManager::hot_pressure() const {
    std::lock_guard<std::mutex> lk(mu_);
    const uint32_t cap = pct_of(cfg_.total_ctx, cfg_.hot_pct);
    if (cap == 0) return false;
    return (float) s_.hot_tokens >= (float) cap * kPressureThreshold;
}

uint32_t TierCapacityManager::recommended_evict_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    const uint32_t cap = pct_of(cfg_.total_ctx, cfg_.hot_pct);
    if (cap == 0) return 0;
    const uint32_t threshold = (uint32_t)((float) cap * kPressureThreshold);
    if (s_.hot_tokens < threshold) return 0;
    const uint32_t target = (uint32_t)((float) cap * kPressureTarget);
    return s_.hot_tokens > target ? (s_.hot_tokens - target) : 0;
}

uint32_t TierCapacityManager::hot_capacity()  const {
    std::lock_guard<std::mutex> lk(mu_);
    return pct_of(cfg_.total_ctx, cfg_.hot_pct);
}
uint32_t TierCapacityManager::warm_capacity() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pct_of(cfg_.total_ctx, cfg_.warm_pct);
}
uint32_t TierCapacityManager::cold_capacity() const {
    std::lock_guard<std::mutex> lk(mu_);
    return pct_of(cfg_.total_ctx, cfg_.cold_pct);
}

TierCapacityManager::Stats TierCapacityManager::snapshot() const {
    std::lock_guard<std::mutex> lk(mu_);
    return s_;
}

void TierCapacityManager::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    s_ = Stats{};
}

}  // namespace mt
