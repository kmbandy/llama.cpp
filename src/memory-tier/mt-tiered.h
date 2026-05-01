#pragma once

// llama_memory_tiered — a memory backend that wraps an existing
// llama_memory_i (the inner cache) and adds tier-aware eviction
// (hot VRAM <-> warm RAM <-> cold SSD) on top.
//
// Phase 2d intentionally lands as PASSTHROUGH ONLY — every virtual
// method delegates to the inner cache. This guarantees that simply
// inserting the wrapper into the model-load path doesn't change
// behaviour. Tier policy hooks land in subsequent phase-2d
// sub-iterations:
//
//   2d-pt   passthrough only (this commit)
//   2d-evict  intercept seq_rm / seq_*ctx ops to record token metadata
//             and start hot->warm eviction at capacity pressure
//   2d-cold   wire the KvtcStore into the eviction pipeline
//   2d-recur  wire the RecurrentStateMover for hybrid models
//   2d-sem    optional semantic prefetch hints
//
// Each sub-iteration is a separate commit and can be reverted in
// isolation if it regresses correctness.
//
// The wrapper composes (does not inherit from) the foundation modules
// so each subsystem stays independently testable.

#include "mt-config.h"
#include "mt-capacity.h"
#include "mt-eviction.h"
#include "mt-mover-attn.h"
#include "mt-mover-recurrent.h"
#include "mt-kvtc-store.h"
#include "mt-semantic.h"

#include "llama-memory.h"

#include <memory>

namespace mt {

class llama_memory_tiered_context;  // fwd

class llama_memory_tiered : public llama_memory_i {
public:
    // Take ownership of the inner cache. The wrapper's lifetime is the
    // outer model's; inner_ is destroyed when the wrapper is.
    //
    // n_seq_max sets the upper bound for the seq_id polling loop used by
    // capacity tracking. Pass cparams.n_seq_max from the dispatch site.
    explicit llama_memory_tiered(llama_memory_ptr     inner,
                                 const TieredConfig & cfg,
                                 uint32_t             n_seq_max);
    ~llama_memory_tiered() override;

    llama_memory_tiered(const llama_memory_tiered &)             = delete;
    llama_memory_tiered & operator=(const llama_memory_tiered &) = delete;

    // ---- llama_memory_i ----
    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags)       override;

    // ---- tier-specific ----
    const TieredConfig & config()        const { return cfg_; }
    llama_memory_i *     inner_for_test() const { return inner_.get(); }

    // Accessors used by mt::llama_memory_tiered_context. Made public so
    // the context wrapper doesn't need to be a friend; they're intended
    // for internal use only.
    TierCapacityManager & capacity()        { return capacity_; }
    TokenMetadataStore  & eviction()        { return eviction_; }
    KvtcStore           & store()           { return store_; }
    SemanticIndex       & semantic()        { return semantic_; }
    AttentionMover      & mover_attn()      { return mover_attn_; }
    RecurrentStateMover & mover_recur()     { return mover_recur_; }

private:
    // Poll inner_->seq_pos_min / seq_pos_max across 0..n_seq_max_, sum
    // active token counts, push the result into capacity_, and refresh
    // eviction_ position metadata for any newly observed tokens. Logs
    // a one-shot message when hot_pressure() flips from off to on.
    void update_tier_state();

    llama_memory_ptr      inner_;
    TieredConfig          cfg_;
    uint32_t              n_seq_max_;

    TierCapacityManager   capacity_;
    TokenMetadataStore    eviction_;
    AttentionMover        mover_attn_;
    RecurrentStateMover   mover_recur_;
    KvtcStore             store_;
    SemanticIndex         semantic_;

    bool                  store_initialized_  = false;
    bool                  pressure_announced_ = false;  // edge-trigger for hot_pressure logging
};

}  // namespace mt
