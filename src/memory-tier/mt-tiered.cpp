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

// Migrate up to n victim positions from hot to warm. Returns count moved.
//
// Only positions NOT already in warm are eligible. Eviction order is
// determined by the configured policy (LRU / LFU / Attention / Hybrid).
// On success, capacity_ migration counters are bumped and warm_pos_to_slot_
// is updated. The inner cache is NOT seq_rm'd — this is a backup-style
// eviction; the hot copy stays valid until the upstream lifecycle frees it
// (via context shift or explicit seq_rm). Future restoration support
// (2d-evict-restore) will rely on warm_pos_to_slot_ to find the backup
// data when seq_rm fires.
uint32_t llama_memory_tiered::evict_hot_to_warm(uint32_t n) {
    if (n == 0) return 0;
    if (!ensure_warm_staging()) return 0;

    // Ask the policy for more candidates than we need so we can skip
    // already-evicted ones without a second policy pass.
    const uint32_t over_request = n + (uint32_t) evicted_to_warm_.size();
    auto candidates = eviction_.get_eviction_candidates(cfg_.eviction, over_request);

    uint32_t moved = 0;
    for (auto pos : candidates) {
        if (moved >= n) break;
        if (warm_free_slots_.empty()) {
            // Warm tier full. 2d-evict-cold will spill to KvtcStore here.
            LLAMA_LOG_DEBUG("mt::llama_memory_tiered: warm full at %u; deferring "
                            "spill until 2d-evict-cold lands\n", warm_capacity_);
            break;
        }
        if (evicted_to_warm_.count(pos)) continue;

        const int slot = warm_free_slots_.back();
        warm_free_slots_.pop_back();

        bool ok = true;
        size_t flat_idx = 0;
        for (const auto & c : tier_view_.attn_caches) {
            if (c.is_swa) continue;
            for (const auto & layer : c.layers) {
                uint8_t * k_dst = warm_buf_.data() + warm_layer_off_[flat_idx]
                                  + (size_t) slot * layer.k_row_bytes;
                uint8_t * v_dst = warm_buf_.data() + warm_layer_v_off_[flat_idx]
                                  + (size_t) slot * layer.v_row_bytes;
                // NOTE: this still passes `pos` as the slot index — that's
                // wrong on rotating-window layouts. 2d-evict-restore-bk
                // (next commit) replaces this with a proper position->slot
                // mapping using cache.cells. For now this matches the
                // pre-restructure behaviour so this commit is no-behavior-
                // change.
                ok = ok && mover_attn_.evict_k(layer, pos, k_dst);
                ok = ok && mover_attn_.evict_v(layer, pos, v_dst);
                if (!ok) goto eviction_failed;
                ++flat_idx;
            }
        }
eviction_failed:;

        if (ok) {
            warm_pos_to_slot_.emplace(pos, (uint32_t) slot);
            evicted_to_warm_.insert(pos);
            ++moved;
        } else {
            // mover failed — return the slot. (HIP error already logged.)
            warm_free_slots_.push_back(slot);
        }
    }

    if (moved > 0) {
        capacity_.on_migrate(moved, TierCapacityManager::Tier::Hot,
                                    TierCapacityManager::Tier::Warm);
        LLAMA_LOG_INFO("mt::llama_memory_tiered: evicted %u tokens hot->warm "
                       "(warm now %u/%u)\n",
                       moved, warm_capacity_ - (uint32_t) warm_free_slots_.size(),
                       warm_capacity_);
    }

    return moved;
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

    const bool now_pressured = capacity_.hot_pressure();
    if (now_pressured && !pressure_announced_) {
        const auto stats = capacity_.snapshot();
        LLAMA_LOG_WARN("mt::llama_memory_tiered: hot pressure reached "
                       "(hot=%u cap=%u recommended_evict=%u)\n",
                       stats.hot_tokens, capacity_.hot_capacity(),
                       capacity_.recommended_evict_count());
        pressure_announced_ = true;
    } else if (!now_pressured && pressure_announced_) {
        pressure_announced_ = false;  // re-arm; next flip-on logs again
    }

    if (now_pressured) {
        const uint32_t want = capacity_.recommended_evict_count();
        if (want > 0) evict_hot_to_warm(want);
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
    const bool ok = inner_->seq_rm(seq_id, p0, p1);
    if (ok) {
        // Drop the removed positions from the eviction store and the
        // warm tier. seq_rm with p0=-1 wipes the whole seq — the
        // resync at the end picks up that case.
        if (p0 >= 0 && p1 > p0) {
            for (llama_pos p = p0; p < p1; ++p) {
                eviction_.remove(p);
                auto it = warm_pos_to_slot_.find(p);
                if (it != warm_pos_to_slot_.end()) {
                    warm_free_slots_.push_back((int) it->second);
                    warm_pos_to_slot_.erase(it);
                    evicted_to_warm_.erase(p);
                    capacity_.on_remove_warm(1);
                }
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
