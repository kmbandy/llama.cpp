#include "mt-tiered.h"
#include "mt-context.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <cassert>

namespace mt {

llama_memory_tiered::llama_memory_tiered(llama_memory_ptr     inner,
                                         const TieredConfig & cfg,
                                         uint32_t             n_seq_max)
    : inner_(std::move(inner)),
      cfg_(cfg),
      n_seq_max_(n_seq_max == 0 ? 1u : n_seq_max),
      capacity_(cfg) {
    if (!inner_) {
        LLAMA_LOG_ERROR("mt::llama_memory_tiered: null inner cache\n");
        return;
    }

    std::string err;
    if (!validate(cfg_, &err)) {
        LLAMA_LOG_WARN("mt::llama_memory_tiered: config invalid (%s) — proceeding with passthrough\n",
                       err.c_str());
    }

    // Cold tier opens lazily — defer KvtcStore::init until the first
    // eviction would actually use it. This keeps "tiering enabled but
    // never reached cold pressure" runs from creating an unused file.
    // (Phase 2d sub-iteration that wires eviction will set this.)

    LLAMA_LOG_INFO("mt::llama_memory_tiered: %s n_seq_max=%u\n",
                   describe(cfg_).c_str(), n_seq_max_);

    // Snapshot the inner view once at init so we know whether the inner
    // backend is tierable. Backends that don't override make_tier_view
    // (or that have no layers / no recurrent state) return an empty view —
    // the wrapper still functions as passthrough but tier movement is a
    // no-op for them.
    const auto view = inner_->make_tier_view();
    LLAMA_LOG_INFO("mt::llama_memory_tiered: tier view: attn_layers=%zu recur_seqs=%zu%s\n",
                   view.attn_layers.size(),
                   view.recur_seqs.size(),
                   view.empty() ? " (not tierable; will run as passthrough)" : "");
}

llama_memory_tiered::~llama_memory_tiered() {
    if (store_initialized_) {
        store_.shutdown();
    }
}

// ---------------------------------------------------------------------------
// Phase 2d-pt: passthrough delegation. Each method forwards to inner_.
// Tier-aware behaviour is added in subsequent sub-iterations.
// ---------------------------------------------------------------------------

// Resync our tier bookkeeping from inner_'s ground truth.
//
// Walks 0..n_seq_max_, sums (seq_pos_max - seq_pos_min + 1) for every
// active sequence, pushes the total into capacity_.set_hot_tokens, and
// records each observed position in eviction_ so the policy scorer
// has data to work with when it's queried in 2d-evict-move. Cheap —
// virtual dispatch only, no allocation in the inner caches' impls.
void llama_memory_tiered::update_tier_state() {
    if (!inner_) return;

    uint32_t total = 0;
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq_max_; ++s) {
        const auto lo = inner_->seq_pos_min(s);
        const auto hi = inner_->seq_pos_max(s);
        if (lo < 0 || hi < lo) continue;
        const uint32_t n = (uint32_t)(hi - lo + 1);
        total += n;

        // Add positions to the eviction store. add() is idempotent on
        // the position key, so re-adding is a no-op. record_access()
        // bumps the LRU/LFU counters so the policy can score these.
        for (llama_pos p = lo; p <= hi; ++p) {
            eviction_.record_access(p);
        }
    }

    capacity_.set_hot_tokens(total);

    const bool now_pressured = capacity_.hot_pressure();
    if (now_pressured && !pressure_announced_) {
        const auto stats = capacity_.snapshot();
        LLAMA_LOG_WARN("mt::llama_memory_tiered: hot pressure reached "
                       "(hot=%u cap=%u recommended_evict=%u) — eviction not yet implemented in 2d-evict-meta\n",
                       stats.hot_tokens, capacity_.hot_capacity(),
                       capacity_.recommended_evict_count());
        pressure_announced_ = true;
    } else if (!now_pressured && pressure_announced_) {
        pressure_announced_ = false;  // re-arm; next flip-on logs again
    }
}

// init_* return the *inner* context directly rather than our wrapper.
//
// Why: the graph builder (src/llama-graph.cpp) downcasts the memory_context
// to its concrete type with static_cast (e.g. to llama_memory_hybrid_iswa_context
// or llama_memory_recurrent_context). That cast is undefined behaviour when
// the dynamic type is anything else, so any wrapper context — even one that
// pure-passthroughs every method — corrupts the cast and SIGSEGVs.
//
// Tier interception therefore happens at the llama_memory_i level (here on
// init_batch entry, plus the seq_* mutators) rather than at the
// llama_memory_context_i level. Hot->warm eviction in 2d-evict-move will run
// at init_batch boundary before delegating; cold prefetch can run before
// init_batch returns. apply()-level hooks (if needed later) will require a
// different mechanism — likely changing the graph downcasts to go through a
// virtual accessor.
llama_memory_context_ptr llama_memory_tiered::init_batch(llama_batch_allocr & balloc,
                                                         uint32_t             n_ubatch,
                                                         bool                 embd_all) {
    update_tier_state();
    return inner_->init_batch(balloc, n_ubatch, embd_all);
}

llama_memory_context_ptr llama_memory_tiered::init_full() {
    return inner_->init_full();
}

llama_memory_context_ptr llama_memory_tiered::init_update(llama_context * lctx, bool optimize) {
    return inner_->init_update(lctx, optimize);
}

bool llama_memory_tiered::get_can_shift() const {
    return inner_ ? inner_->get_can_shift() : false;
}

void llama_memory_tiered::clear(bool data) {
    if (inner_) inner_->clear(data);
    capacity_.reset();
    eviction_.clear();
    semantic_.clear();
    pressure_announced_ = false;
    // Note: KvtcStore is NOT cleared on clear() — its file persists for
    // the run. clear() corresponds to seq_rm-everything, not shutdown.
}

bool llama_memory_tiered::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!inner_) return false;
    const bool ok = inner_->seq_rm(seq_id, p0, p1);
    if (ok) {
        // Drop the removed positions from the eviction store; resync hot
        // count from inner state to absorb any whole-seq reshuffling
        // (seq_rm with p0=-1 wipes the whole seq).
        if (p0 >= 0 && p1 > p0) {
            for (llama_pos p = p0; p < p1; ++p) {
                eviction_.remove(p);
            }
        }
        update_tier_state();
    }
    return ok;
}

void llama_memory_tiered::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                                  llama_pos p0, llama_pos p1) {
    if (inner_) inner_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_tiered::seq_keep(llama_seq_id seq_id) {
    if (inner_) inner_->seq_keep(seq_id);
}

void llama_memory_tiered::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (inner_) inner_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_tiered::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (inner_) inner_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_tiered::seq_pos_min(llama_seq_id seq_id) const {
    return inner_ ? inner_->seq_pos_min(seq_id) : -1;
}

llama_pos llama_memory_tiered::seq_pos_max(llama_seq_id seq_id) const {
    return inner_ ? inner_->seq_pos_max(seq_id) : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_tiered::memory_breakdown() const {
    return inner_ ? inner_->memory_breakdown()
                  : std::map<ggml_backend_buffer_type_t, size_t>{};
}

void llama_memory_tiered::state_write(llama_io_write_i & io,
                                       llama_seq_id          seq_id,
                                       llama_state_seq_flags flags) const {
    if (inner_) inner_->state_write(io, seq_id, flags);
}

void llama_memory_tiered::state_read(llama_io_read_i & io,
                                      llama_seq_id          seq_id,
                                      llama_state_seq_flags flags) {
    if (inner_) inner_->state_read(io, seq_id, flags);
}

}  // namespace mt
