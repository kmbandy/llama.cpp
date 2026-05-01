#pragma once

// TierCapacityManager — per-tier token counting + eviction triggers.
//
// Pure bookkeeping. Tracks how many tokens currently live in each tier
// (hot / warm / cold), and exposes a single predicate the wrapper uses
// to decide when to start evicting from hot. High-water marks are kept
// for telemetry.
//
// Threadsafe via internal mutex. Counters are updated via on_*()
// hooks called from the move path — this class does NOT trigger any
// migrations itself; it only signals when a migration would be a good
// idea.

#include "mt-config.h"

#include <cstdint>
#include <mutex>

namespace mt {

class TierCapacityManager {
public:
    struct Stats {
        uint32_t hot_tokens        = 0;
        uint32_t warm_tokens       = 0;
        uint32_t cold_tokens       = 0;
        uint32_t hot_high_water    = 0;
        uint32_t warm_high_water   = 0;
        uint32_t cold_high_water   = 0;
        uint64_t hot_to_warm_count = 0;
        uint64_t hot_to_cold_count = 0;
        uint64_t warm_to_cold_count = 0;
        uint64_t cold_to_warm_count = 0;
        uint64_t warm_to_hot_count  = 0;
        uint64_t cold_to_hot_count  = 0;
    };

    explicit TierCapacityManager(const TieredConfig & cfg);

    // Reconfigure (e.g. on resize). Resets high-water marks.
    void reconfigure(const TieredConfig & cfg);

    // ---- mutation hooks (called by the mover) ----

    // n tokens were just added to the hot tier (e.g. a prompt token's KV).
    void on_insert_hot(uint32_t n);

    // n tokens were just removed from the hot tier without migration
    // (e.g. context shift / seq erase). Use on_migrate for tier moves.
    void on_remove_hot(uint32_t n);

    // Set the hot-tier count to an absolute value (e.g. resync from
    // the inner cache's seq state). Updates high-water mark. Does not
    // touch warm/cold counters or migration totals.
    void set_hot_tokens(uint32_t n);

    // n tokens migrated from one tier to another. Updates both counters
    // and the per-direction migration counters.
    enum class Tier { Hot, Warm, Cold };
    void on_migrate(uint32_t n, Tier from, Tier to);

    // ---- queries ----

    // True iff hot-tier usage is above the eviction trigger (currently 80%
    // of hot capacity). Wrapper polls this in init_batch and starts
    // evicting when true.
    bool hot_pressure() const;

    // Number of tokens that should be evicted to bring hot back to
    // a comfortable level (target = 60% of hot capacity). Returns 0 if
    // no pressure.
    uint32_t recommended_evict_count() const;

    // Configured capacities, derived from cfg.
    uint32_t hot_capacity()  const;
    uint32_t warm_capacity() const;
    uint32_t cold_capacity() const;

    // Snapshot the current stats. Threadsafe.
    Stats snapshot() const;

    // Reset all counters. Tests + diagnostics.
    void reset();

private:
    mutable std::mutex mu_;
    TieredConfig       cfg_;
    Stats              s_;
};

}  // namespace mt
