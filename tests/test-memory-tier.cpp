// Unit tests for the mt::* foundation modules under src/memory-tier/.
//
// Follows the same lightweight style as test-weight-pager.cpp: each
// TEST_FN runs subtests and returns a failure count; main sums them
// and exits non-zero if any failed.
//
// Phase 2a coverage: TieredConfig, TierCapacityManager, TokenMetadataStore,
// INT4/INT8 quant round-trips. Movers, KvtcStore, semantic index, and
// the wrapper class itself land in later phases.

#include "memory-tier/mt-config.h"
#include "memory-tier/mt-capacity.h"
#include "memory-tier/mt-eviction.h"
#include "memory-tier/mt-quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <thread>
#include <vector>

#define EXPECT(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d): %s\n", __func__, __LINE__, (msg)); \
        ++fails; \
    } \
} while (0)

#define EXPECT_EQ_INT(actual, expected, msg) do { \
    if ((long long)(actual) != (long long)(expected)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d): %s — got %lld, expected %lld\n", \
                     __func__, __LINE__, (msg), (long long)(actual), (long long)(expected)); \
        ++fails; \
    } \
} while (0)

#define EXPECT_NEAR(actual, expected, tol, msg) do { \
    const double a = (double)(actual); \
    const double e = (double)(expected); \
    if (std::fabs(a - e) > (tol)) { \
        std::fprintf(stderr, "  FAIL: %s (line %d): %s — got %g, expected %g (tol %g)\n", \
                     __func__, __LINE__, (msg), a, e, (double)(tol)); \
        ++fails; \
    } \
} while (0)

// ---------------------------------------------------------------------------
// TieredConfig
// ---------------------------------------------------------------------------

static int test_config_validate() {
    int fails = 0;
    std::string err;

    // Default config should be valid (25/25/50).
    mt::TieredConfig c;
    EXPECT(mt::validate(c, &err), "default config validates");

    // Sum != 100 fails.
    c.hot_pct = 30; c.warm_pct = 30; c.cold_pct = 30;
    EXPECT(!mt::validate(c, &err), "sum != 100 rejected");

    // Negative percentage fails.
    c = {}; c.hot_pct = -1;
    EXPECT(!mt::validate(c, &err), "negative pct rejected");

    // Out-of-range threshold fails.
    c = {}; c.attention_threshold = 1.5f;
    EXPECT(!mt::validate(c, &err), "attention_threshold > 1 rejected");

    c = {}; c.semantic_threshold = -0.1f;
    EXPECT(!mt::validate(c, &err), "negative semantic_threshold rejected");

    c = {}; c.semantic_top_k = -3;
    EXPECT(!mt::validate(c, &err), "negative top_k rejected");

    // Sum within ±1% tolerance is OK.
    c = {}; c.hot_pct = 25.4f; c.warm_pct = 25.4f; c.cold_pct = 49.7f;  // 100.5
    EXPECT(mt::validate(c, &err), "sum 100.5 ok (within tolerance)");

    return fails;
}

static int test_config_describe() {
    int fails = 0;
    mt::TieredConfig c;
    c.total_ctx = 131072;
    const std::string s = mt::describe(c);
    EXPECT(s.find("hot 25%")  != std::string::npos, "describe contains hot_pct");
    EXPECT(s.find("warm 25%") != std::string::npos, "describe contains warm_pct");
    EXPECT(s.find("cold 50%") != std::string::npos, "describe contains cold_pct");
    EXPECT(s.find("Hybrid")   != std::string::npos, "describe names policy");
    EXPECT(s.find("int4")     != std::string::npos, "describe names compression");
    return fails;
}

// ---------------------------------------------------------------------------
// TierCapacityManager
// ---------------------------------------------------------------------------

static int test_capacity_basic() {
    int fails = 0;
    mt::TieredConfig cfg;
    cfg.total_ctx = 1000;          // hot=250, warm=250, cold=500
    mt::TierCapacityManager cap(cfg);

    EXPECT_EQ_INT(cap.hot_capacity(),  250, "hot_capacity");
    EXPECT_EQ_INT(cap.warm_capacity(), 250, "warm_capacity");
    EXPECT_EQ_INT(cap.cold_capacity(), 500, "cold_capacity");

    EXPECT(!cap.hot_pressure(),                     "no pressure on empty");
    EXPECT_EQ_INT(cap.recommended_evict_count(), 0, "no eviction recommended");

    cap.on_insert_hot(100);
    auto s = cap.snapshot();
    EXPECT_EQ_INT(s.hot_tokens,     100, "after insert 100");
    EXPECT_EQ_INT(s.hot_high_water, 100, "high_water tracks");
    EXPECT(!cap.hot_pressure(), "100/250 (40%) below 80% threshold");

    cap.on_insert_hot(110);  // 210/250 = 84% -> pressure
    EXPECT(cap.hot_pressure(), "210/250 above pressure threshold");
    const uint32_t to_evict = cap.recommended_evict_count();
    EXPECT(to_evict > 0,        "evict count positive under pressure");
    EXPECT(to_evict <= 100,     "evict count bounded sensibly");

    return fails;
}

static int test_capacity_migrate() {
    int fails = 0;
    mt::TieredConfig cfg;
    cfg.total_ctx = 1000;
    mt::TierCapacityManager cap(cfg);

    cap.on_insert_hot(200);
    cap.on_migrate(50, mt::TierCapacityManager::Tier::Hot,
                       mt::TierCapacityManager::Tier::Warm);

    auto s = cap.snapshot();
    EXPECT_EQ_INT(s.hot_tokens,         150, "hot decreased");
    EXPECT_EQ_INT(s.warm_tokens,        50,  "warm increased");
    EXPECT_EQ_INT(s.hot_to_warm_count,  50,  "h->w counter");

    cap.on_migrate(30, mt::TierCapacityManager::Tier::Warm,
                       mt::TierCapacityManager::Tier::Cold);
    s = cap.snapshot();
    EXPECT_EQ_INT(s.warm_tokens,         20, "warm after w->c");
    EXPECT_EQ_INT(s.cold_tokens,         30, "cold after w->c");
    EXPECT_EQ_INT(s.warm_to_cold_count,  30, "w->c counter");

    cap.on_migrate(15, mt::TierCapacityManager::Tier::Cold,
                       mt::TierCapacityManager::Tier::Hot);
    s = cap.snapshot();
    EXPECT_EQ_INT(s.hot_tokens,         165, "hot after c->h restore");
    EXPECT_EQ_INT(s.cold_tokens,         15, "cold after c->h");
    EXPECT_EQ_INT(s.cold_to_hot_count,   15, "c->h counter");

    return fails;
}

// ---------------------------------------------------------------------------
// TokenMetadataStore + eviction policies
// ---------------------------------------------------------------------------

static int test_eviction_lru() {
    int fails = 0;
    mt::TokenMetadataStore store;

    store.add(1);
    store.add(2);
    store.add(3);

    // Sleep is unavoidable for time-based scoring; keep it tiny.
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    store.record_access(2);  // bump pos 2 to MRU
    store.record_access(3);  // bump pos 3 to MRU (latest)

    auto candidates = store.get_eviction_candidates(mt::EvictionPolicy::LRU, 3);
    EXPECT_EQ_INT(candidates.size(), 3, "returned 3 candidates");
    EXPECT_EQ_INT(candidates[0], 1, "LRU = pos 1 (oldest access)");
    return fails;
}

static int test_eviction_lfu() {
    int fails = 0;
    mt::TokenMetadataStore store;

    store.add(1);
    store.add(2);
    store.add(3);
    store.record_access(2);
    store.record_access(2);
    store.record_access(2);
    store.record_access(3);

    // pos 1: 0 accesses, pos 3: 1, pos 2: 3
    auto candidates = store.get_eviction_candidates(mt::EvictionPolicy::LFU, 3);
    EXPECT_EQ_INT(candidates.size(), 3, "LFU 3 candidates");
    EXPECT_EQ_INT(candidates[0], 1, "LFU first = least frequent (pos 1)");
    EXPECT_EQ_INT(candidates[2], 2, "LFU last = most frequent (pos 2)");
    return fails;
}

static int test_eviction_attention() {
    int fails = 0;
    mt::TokenMetadataStore store;

    store.add(1, 0.9f);
    store.add(2, 0.1f);
    store.add(3, 0.5f);

    auto candidates = store.get_eviction_candidates(mt::EvictionPolicy::Attention, 3);
    EXPECT_EQ_INT(candidates.size(), 3, "attention 3 candidates");
    EXPECT_EQ_INT(candidates[0], 2, "lowest attention evicted first");
    EXPECT_EQ_INT(candidates[2], 1, "highest attention kept");
    return fails;
}

static int test_eviction_count_clamp() {
    int fails = 0;
    mt::TokenMetadataStore store;
    store.add(1); store.add(2);
    auto c = store.get_eviction_candidates(mt::EvictionPolicy::LRU, 100);
    EXPECT_EQ_INT(c.size(), 2, "clamp count to store size");

    auto c0 = store.get_eviction_candidates(mt::EvictionPolicy::LRU, 0);
    EXPECT_EQ_INT(c0.size(), 0, "count=0 returns empty");

    return fails;
}

// ---------------------------------------------------------------------------
// Quantization round-trip
// ---------------------------------------------------------------------------

static int test_quant_int4_round_trip() {
    int fails = 0;

    // Test the FULL [-8, +7] range divided by 7 = approx [-1.143, +1].
    // Inputs are pre-snapped to representable values so round-trip is
    // exact (within float precision).
    std::vector<float> in;
    for (int q = -7; q <= 7; ++q) in.push_back((float) q / 7.0f);

    auto packed = mt::quantize_int4(in.data(), in.size());
    EXPECT_EQ_INT(packed.size(), (in.size() + 1) / 2, "int4 packed size");

    std::vector<float> out(in.size());
    bool ok = mt::dequantize_int4(packed.data(), out.data(), out.size());
    EXPECT(ok, "dequantize_int4 returns true");

    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i], in[i], 1e-6, "int4 round-trip exact for representable values");
    }

    // Edge case: -1 saturates to -7/7 (we don't use the -8 codepoint to
    // avoid asymmetric saturation behaviour).
    float e_in[1]  = { -1.0f };
    auto e_packed = mt::quantize_int4(e_in, 1);
    float e_out[1] = { 0.0f };
    EXPECT(mt::dequantize_int4(e_packed.data(), e_out, 1), "edge -1 dequant");
    EXPECT_NEAR(e_out[0], -1.0f, 0.05f, "edge -1 within 5% (saturation tolerance)");

    // Odd length: write 3 floats, read 3 floats. The 4th nibble is unused.
    float o_in[3] = { 0.0f, 0.5f, -0.5f };
    auto o_packed = mt::quantize_int4(o_in, 3);
    EXPECT_EQ_INT(o_packed.size(), 2, "odd length packs to ceil(n/2)");
    float o_out[3] = { 99, 99, 99 };
    EXPECT(mt::dequantize_int4(o_packed.data(), o_out, 3), "odd-len dequant");
    EXPECT_NEAR(o_out[0],  0.0f, 0.1f, "odd[0]");
    EXPECT_NEAR(o_out[1],  0.5f, 0.1f, "odd[1]");
    EXPECT_NEAR(o_out[2], -0.5f, 0.1f, "odd[2]");

    // Negative: null inputs.
    EXPECT(!mt::dequantize_int4(nullptr, o_out, 3), "null src rejected");
    EXPECT(!mt::dequantize_int4(o_packed.data(), nullptr, 3), "null dst rejected");

    return fails;
}

static int test_quant_int8_round_trip() {
    int fails = 0;

    std::vector<float> in;
    for (int q = -127; q <= 127; q += 8) in.push_back((float) q / 127.0f);

    auto packed = mt::quantize_int8(in.data(), in.size());
    EXPECT_EQ_INT(packed.size(), in.size(), "int8 packed size = n");

    std::vector<float> out(in.size());
    EXPECT(mt::dequantize_int8(packed.data(), out.data(), out.size()), "dequant_int8");

    for (size_t i = 0; i < in.size(); ++i) {
        EXPECT_NEAR(out[i], in[i], 1e-5, "int8 round-trip");
    }

    return fails;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
    int total_fails = 0;

    struct named_test {
        const char * name;
        int (*fn)();
    };
    named_test tests[] = {
        { "config_validate",        test_config_validate        },
        { "config_describe",        test_config_describe        },
        { "capacity_basic",         test_capacity_basic         },
        { "capacity_migrate",       test_capacity_migrate       },
        { "eviction_lru",           test_eviction_lru           },
        { "eviction_lfu",           test_eviction_lfu           },
        { "eviction_attention",     test_eviction_attention     },
        { "eviction_count_clamp",   test_eviction_count_clamp   },
        { "quant_int4_round_trip",  test_quant_int4_round_trip  },
        { "quant_int8_round_trip",  test_quant_int8_round_trip  },
    };

    for (const auto & t : tests) {
        std::fprintf(stderr, "RUN  test_%s\n", t.name);
        const int f = t.fn();
        std::fprintf(stderr, "%s test_%s (%d failure%s)\n",
                     f == 0 ? "PASS" : "FAIL", t.name, f, f == 1 ? "" : "s");
        total_fails += f;
    }

    std::fprintf(stderr, "\n=== %s: %d total failures ===\n",
                 total_fails == 0 ? "PASS" : "FAIL", total_fails);
    return total_fails == 0 ? 0 : 1;
}
