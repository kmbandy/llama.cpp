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
    tier_view_ = inner_->make_tier_view();
    size_t base_layers = 0, swa_layers = 0;
    for (const auto & c : tier_view_.attn_caches) {
        (c.is_swa ? swa_layers : base_layers) += c.layers.size();
    }
    LLAMA_LOG_INFO("mt::llama_memory_tiered: tier view: attn_caches=%zu "
                   "(base_layers=%zu swa_layers=%zu) recur_seqs=%zu%s\n",
                   tier_view_.attn_caches.size(),
                   base_layers, swa_layers,
                   tier_view_.recur_seqs.size(),
                   tier_view_.empty() ? " (not tierable; will run as passthrough)" : "");
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

// Lazy-allocate the warm-tier host staging buffer.
//
// Sizing: warm capacity tokens × Σ_layers(k_row_bytes + v_row_bytes).
// For Qwen3.6-27B at ctx=8192 with warm=25%, 16 attn layers and ~2KB per
// row this is ~16 MB — totally reasonable for host RAM. For 97B at ctx
// 131072 this could climb into the GBs and we'd want pinned + paged
// allocation; for now plain new uint8_t[].
//
// Returns false if the inner cache has no attention layers (recurrent-only
// models — Phase 2d-recur will own those).
bool llama_memory_tiered::ensure_warm_staging() {
    if (warm_initialized_) return true;

    // Count restorable (non-SWA) layers. We only allocate warm storage
    // for those — SWA layers can't be restored (their distance mask hides
    // any position older than n_swa back from the head) so backing them
    // up is wasted RAM.
    size_t restorable_layers = 0;
    for (const auto & c : tier_view_.attn_caches) {
        if (!c.is_swa) restorable_layers += c.layers.size();
    }
    if (restorable_layers == 0) {
        if (!no_attn_warned_) {
            LLAMA_LOG_WARN("mt::llama_memory_tiered: no restorable (non-SWA) attention "
                           "layers in tier view; hot->warm eviction is a no-op for this model\n");
            no_attn_warned_ = true;
        }
        return false;
    }

    warm_capacity_ = capacity_.warm_capacity();
    if (warm_capacity_ == 0) {
        LLAMA_LOG_WARN("mt::llama_memory_tiered: warm_capacity=0; eviction disabled\n");
        return false;
    }

    // Layout: each restorable layer gets a [K slab | V slab] block,
    // each sized warm_cap * row_bytes. Indexed by a flat "restorable
    // layer index" — we walk the same iteration order in evict_hot_to_warm.
    warm_layer_off_.resize(restorable_layers);
    warm_layer_v_off_.resize(restorable_layers);
    size_t cursor = 0;
    size_t flat_idx = 0;
    for (const auto & c : tier_view_.attn_caches) {
        if (c.is_swa) continue;
        for (const auto & a : c.layers) {
            warm_layer_off_[flat_idx]   = cursor;
            warm_layer_v_off_[flat_idx] = cursor + (size_t) warm_capacity_ * a.k_row_bytes;
            cursor                     += (size_t) warm_capacity_ * (a.k_row_bytes + a.v_row_bytes);
            ++flat_idx;
        }
    }
    warm_buf_.assign(cursor, 0);

    warm_free_slots_.reserve(warm_capacity_);
    for (uint32_t s = warm_capacity_; s-- > 0; ) {
        warm_free_slots_.push_back((int) s);
    }

    LLAMA_LOG_INFO("mt::llama_memory_tiered: warm staging: capacity=%u tokens, "
                   "buffer=%.1f MiB across %zu restorable layers (skipped %zu SWA layers)\n",
                   warm_capacity_, (double) warm_buf_.size() / (1024.0 * 1024.0),
                   restorable_layers,
                   tier_view_.attn_layer_count() - restorable_layers);

    warm_initialized_ = true;
    return true;
}

// Back up positions [p0, p1) of seq_id from hot to warm. Driven by
// seq_rm: upstream is about to delete those positions, so this is the
// last chance to capture their K/V data for later restoration.
//
// Slot lookup uses a freshly snapshotted InnerView (the cell layout
// changes as the cache evolves; the cached tier_view_ from construction
// has stale cells). We re-snapshot per call — cheap relative to the
// hipMemcpy work that follows.
//
// SWA caches are skipped: their distance mask hides any restored
// position older than n_swa back from the head, so backing them up
// would be wasted work.
uint32_t llama_memory_tiered::backup_seq_rm_range(llama_seq_id seq_id,
                                                   llama_pos    p0,
                                                   llama_pos    p1) {
    if (!inner_ || p0 < 0 || p1 <= p0) return 0;
    if (!ensure_warm_staging()) return 0;

    // Fresh cell snapshot. We only need the cell occupancy maps; the
    // K/V tensor pointers in tier_view_ are stable for the cache's
    // lifetime, so we keep using those.
    const auto fresh = inner_->make_tier_view();

    // Pair restorable caches in tier_view_ (which has K/V pointers and
    // matched warm_layer_off_) with their corresponding fresh cell
    // snapshot. Indices line up because both views walk the same
    // make_tier_view dispatch path; we assert that defensively.
    if (fresh.attn_caches.size() != tier_view_.attn_caches.size()) {
        LLAMA_LOG_WARN("mt::backup_seq_rm_range: cache topology changed (%zu->%zu); "
                       "skipping backup to avoid mis-mapped slots\n",
                       tier_view_.attn_caches.size(), fresh.attn_caches.size());
        return 0;
    }

    uint32_t backed_up = 0;
    size_t flat_layer_idx = 0;

    for (size_t ci = 0; ci < tier_view_.attn_caches.size(); ++ci) {
        const auto & cache_view = tier_view_.attn_caches[ci];
        const auto & cells      = fresh.attn_caches[ci].cells;

        if (cache_view.is_swa) {
            // Layers in this cache aren't in warm staging — flat_layer_idx
            // doesn't advance.
            continue;
        }

        // Find slots whose (pos, seq_id) match the seq_rm range.
        // O(kv_size) per call — acceptable; kv_size is typically <= ctx
        // and seq_rm is a control-plane event not a hot loop.
        for (uint32_t slot = 0; slot < cells.size(); ++slot) {
            const auto & cs = cells[slot];
            if (cs.pos < p0 || cs.pos >= p1) continue;
            if (cs.seq_id != seq_id) continue;

            // Skip if we've already backed this position up.
            if (evicted_to_warm_.count(cs.pos)) continue;

            if (warm_free_slots_.empty()) {
                LLAMA_LOG_DEBUG("mt::backup_seq_rm_range: warm full at %u; "
                                "remaining positions in this seq_rm will be lost "
                                "(2d-evict-cold will spill warm to SSD)\n",
                                warm_capacity_);
                goto warm_full;
            }

            const int warm_slot = warm_free_slots_.back();
            warm_free_slots_.pop_back();

            // Copy this slot's K/V across every layer in this cache.
            bool ok = true;
            size_t layer_idx_in_warm = flat_layer_idx;
            for (const auto & layer : cache_view.layers) {
                uint8_t * k_dst = warm_buf_.data()
                                  + warm_layer_off_[layer_idx_in_warm]
                                  + (size_t) warm_slot * layer.k_row_bytes;
                uint8_t * v_dst = warm_buf_.data()
                                  + warm_layer_v_off_[layer_idx_in_warm]
                                  + (size_t) warm_slot * layer.v_row_bytes;
                if (!mover_attn_.evict_k(layer, slot, k_dst) ||
                    !mover_attn_.evict_v(layer, slot, v_dst)) {
                    ok = false;
                    break;
                }
                ++layer_idx_in_warm;
            }

            if (ok) {
                warm_pos_to_slot_.emplace(cs.pos, (uint32_t) warm_slot);
                evicted_to_warm_.insert(cs.pos);
                ++backed_up;
            } else {
                warm_free_slots_.push_back(warm_slot);
                LLAMA_LOG_WARN("mt::backup_seq_rm_range: mover failed at slot %u "
                               "(pos %d); aborting this seq_rm's backup\n",
                               slot, cs.pos);
                goto mover_failed;
            }
        }

        // Advance flat_layer_idx by the number of layers we just iterated.
        flat_layer_idx += cache_view.layers.size();
    }

warm_full:
mover_failed:

    if (backed_up > 0) {
        capacity_.on_migrate(backed_up,
                             TierCapacityManager::Tier::Hot,
                             TierCapacityManager::Tier::Warm);
        LLAMA_LOG_INFO("mt::backup_seq_rm_range: seq=%d range=[%d,%d) "
                       "backed up %u positions (warm now %u/%u)\n",
                       seq_id, p0, p1, backed_up,
                       warm_capacity_ - (uint32_t) warm_free_slots_.size(),
                       warm_capacity_);
    }

    return backed_up;
}

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

        // Record accesses for positions still resident in hot. Skip
        // ones we've already moved to warm — re-adding would put them
        // back in the eviction policy's candidate pool and we'd churn.
        for (llama_pos p = lo; p <= hi; ++p) {
            if (evicted_to_warm_.count(p) == 0) {
                eviction_.record_access(p);
            }
        }
    }

    capacity_.set_hot_tokens(total);

    // Pressure logging is now informational only — the actual eviction
    // trigger has moved to seq_rm-time (see backup_seq_rm_range). This
    // matches what tier KV can actually do under llama.cpp's existing
    // graph contract: capture data when upstream is about to delete it,
    // restore on demand. Pressure-poll-triggered eviction would require
    // freeing hot slots without upstream knowing, which would either
    // strand attention reads or require graph changes per architecture.
    const bool now_pressured = capacity_.hot_pressure();
    if (now_pressured && !pressure_announced_) {
        const auto stats = capacity_.snapshot();
        LLAMA_LOG_INFO("mt::llama_memory_tiered: hot pressure reached "
                       "(hot=%u cap=%u) — eviction fires on next seq_rm in "
                       "this range\n",
                       stats.hot_tokens, capacity_.hot_capacity());
        pressure_announced_ = true;
    } else if (!now_pressured && pressure_announced_) {
        pressure_announced_ = false;
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
    warm_pos_to_slot_.clear();
    evicted_to_warm_.clear();
    if (warm_initialized_) {
        // Refill the free-slot stack to its full warm capacity.
        warm_free_slots_.clear();
        warm_free_slots_.reserve(warm_capacity_);
        for (uint32_t s = warm_capacity_; s-- > 0; ) {
            warm_free_slots_.push_back((int) s);
        }
    }
    // Note: KvtcStore is NOT cleared on clear() — its file persists for
    // the run. clear() corresponds to seq_rm-everything, not shutdown.
}

bool llama_memory_tiered::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!inner_) return false;

    // Tier backup: capture K/V for positions [p0, p1) BEFORE delegating
    // to the inner cache. After inner_->seq_rm the slots are reusable
    // and reading from them returns whatever the next batch wrote.
    //
    // p0 < 0 / p1 < 0 are sentinels meaning "from the beginning" /
    // "to the end". We don't try to resolve those into concrete bounds
    // for backup — those wholesale-wipe calls are typically clear/reset
    // events where restoration isn't useful, and computing concrete
    // bounds requires polling the inner. Skip backup for sentinels;
    // capture happens for explicit ranges only (which is what context
    // shift produces).
    if (p0 >= 0 && p1 > p0) {
        backup_seq_rm_range(seq_id, p0, p1);
    }

    const bool ok = inner_->seq_rm(seq_id, p0, p1);
    if (ok) {
        // Drop the removed positions from the eviction store. We keep
        // the warm copies — those are now the only source of truth for
        // restoration. (warm_pos_to_slot_ is unchanged here; we only
        // free a warm slot when the warm copy itself becomes obsolete,
        // e.g. via clear() or when the same position is re-added then
        // re-removed.)
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
