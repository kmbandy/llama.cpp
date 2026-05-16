#pragma once

// TieredConfig — configuration for the tiered KV cache.
//
// Pure data: percentages, paths, policy enums. Populated by the
// integration code (server or context init) from common_params and
// passed to llama_memory_tiered's constructor.
//
// Single-source-of-truth for tier knobs. The corresponding CLI flags
// live in common/arg.cpp under --kv-tier-* (with deprecation aliases
// for the old --tier-* / --kv-warm-device / --kv-semantic-* names).

#include <cstdint>
#include <string>

namespace mt {

// Eviction policy for hot-tier overflow.
//
// Numeric values match the legacy CLI flag (--kv-tier-eviction-policy 0/1/2/3)
// so users with existing scripts get the same behaviour without rebinding.
enum class EvictionPolicy : int {
    LRU       = 0,  // least-recently-used
    LFU       = 1,  // least-frequently-used
    Attention = 2,  // attention score (lowest aggregate attention evicted first)
    Hybrid    = 3,  // weighted combination of LRU + LFU + attention (default)
};

// Compression for cold-tier (SSD) storage.
//
// Numeric values match --kv-tier-compression 0..4. INT4 is the default
// because it's the only one currently round-trip-tested in mt-quant.
enum class Compression : int {
    None      = 0,
    Int4      = 1,
    Int8      = 2,
    LZ4       = 3,  // reserved; not yet implemented
    Quantized = 4,  // reserved; native model quant
};

struct TieredConfig {
    // Tier capacities, expressed as percentages of total_ctx.
    // hot_pct + warm_pct + cold_pct must sum to ~100; the exact rounding
    // policy is up to the consumer (typically capacity_manager).
    float hot_pct  = 25.0f;
    float warm_pct = 25.0f;
    float cold_pct = 50.0f;

    // Total context tokens budgeted across all tiers. The actual
    // llama_context's n_ctx is the hot-tier capacity; this field carries
    // the full budget so the tier system knows how much warm + cold to
    // pre-reserve. 0 = derive from llama_context's n_ctx (no fan-out).
    uint32_t total_ctx = 0;

    // Cold-tier SSD location. Relative paths resolved against $PWD by
    // the tier system at init time.
    std::string ssd_path = "./tiered-cache";

    // Eviction selection policy.
    EvictionPolicy eviction = EvictionPolicy::Hybrid;

    // Cold-tier compression. None = straight FP copies (large but lossless),
    // Int4/Int8 = quantize on write, dequantize on read.
    Compression compression = Compression::Int4;

    // Eviction-policy threshold for ATTENTION / HYBRID modes. Tokens with
    // average attention below this fraction are preferred for eviction.
    // [0.0, 1.0]. 0.1 = bottom 10% of attention scores.
    float attention_threshold = 0.1f;

    // HIP/CUDA device index hosting the warm tier. -1 = warm tier in host
    // RAM (the simple default). >= 0 = warm tier in VRAM on that device
    // (e.g. an eGPU used as overflow). The new tier system does not yet
    // implement device-warm; the field is preserved for forward compat.
    int warm_device = -1;

    // Pin the host-RAM warm tier in physical memory via mlock() / VirtualLock()
    // so the kernel cannot page it out to swap under memory pressure. Default
    // ON because the failure mode is silent and catastrophic: at long context
    // on RAM-constrained hosts, warm-tier reads turn into swap-disk reads,
    // collapsing prefill throughput to ~10% of expected. Disable only if the
    // host cannot grant the lock (RLIMIT_MEMLOCK / CAP_IPC_LOCK) and the lock
    // failure is acceptable for the workload. When mlock fails, allocation
    // still succeeds and a warning is logged with the remediation steps.
    bool warm_mlock = true;

    // Optional semantic-similarity prefetch. Non-empty enables it.
    std::string semantic_index;
    float       semantic_threshold = 0.65f;
    int         semantic_top_k     = 5;

    // Paged-blocks refactor (PHASE 2a, opt-in, off by default).
    //
    // When true, llama_memory_tiered allocates a mt::BlockPool +
    // mt::BlockTable alongside the existing position-keyed bookkeeping.
    // Phase 2a only INSTANTIATES the new structures and logs init —
    // no live wiring yet. Phase 2b will start using BlockTable on the
    // write side; Phase 2c on the read side; Phase 2d makes paged
    // the default and removes the position-keyed paths.
    //
    // block_size is fixed at 16 tokens for now (matches vLLM default).
    // Configurable in a later phase if benchmarks show otherwise.
    bool     paged_blocks    = false;
    uint32_t paged_block_size = 16;
};

// Returns true iff cfg's percentages sum within tolerance and all numeric
// fields are in valid ranges. On failure, populates *err_out (if non-null)
// with a human-readable reason.
bool validate(const TieredConfig & cfg, std::string * err_out = nullptr);

// Pretty-print the active config to a single log line. Useful at init.
std::string describe(const TieredConfig & cfg);

}  // namespace mt
