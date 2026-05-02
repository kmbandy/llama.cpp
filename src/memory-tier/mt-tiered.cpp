#include "mt-tiered.h"
#include "mt-context.h"

#include "llama-impl.h"  // LLAMA_LOG_*

#include <algorithm>
#include <cassert>
#include <unistd.h>  // getpid

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

// Lazy-open the cold-tier KvtcStore. Path comes from cfg_.ssd_path,
// which the CLI populates from --kv-tier-ssd-path. We append a unique
// per-process filename so multiple wrapper instances (e.g. tests) in
// the same directory don't collide.
bool llama_memory_tiered::ensure_cold_store() {
    if (store_initialized_) return true;
    if (capacity_.cold_capacity() == 0) {
        return false;
    }

    // Use cfg_.ssd_path as a directory hint; cold file lives inside it.
    std::string path = cfg_.ssd_path;
    if (path.empty()) {
        path = "./tiered-cache";  // matches TieredConfig default
    }
    if (!path.empty() && path.back() != '/') {
        path += "/";
    }
    path += "mt-cold-";
    path += std::to_string((long) ::getpid());
    path += ".kvtc";

    if (!store_.init(path, cfg_.compression)) {
        LLAMA_LOG_WARN("mt::llama_memory_tiered: KvtcStore::init(%s) failed; "
                       "cold tier disabled\n", path.c_str());
        return false;
    }
    store_initialized_ = true;
    LLAMA_LOG_INFO("mt::llama_memory_tiered: cold tier opened at %s "
                   "(compression=%d)\n", path.c_str(), (int) cfg_.compression);
    return true;
}

// Spill one warm entry to cold. FIFO: the oldest warm entry goes first.
// For each restorable layer, write the K and V bytes to KvtcStore keyed
// by (flat_layer_idx, position). On success: free the warm slot, drop
// from warm_pos_to_slot_, add to cold_positions_, decrement warm count
// and increment cold count via on_migrate.
bool llama_memory_tiered::spill_one_to_cold() {
    if (!ensure_cold_store()) return false;
    if (warm_insertion_order_.empty()) return false;

    const llama_pos victim_pos = warm_insertion_order_.front();
    auto it = warm_pos_to_slot_.find(victim_pos);
    if (it == warm_pos_to_slot_.end()) {
        // Stale FIFO entry (e.g. forgotten via forget_warm) — skip.
        warm_insertion_order_.pop_front();
        return spill_one_to_cold();
    }
    const uint32_t warm_slot = it->second;

    // Write K/V for every restorable layer.
    bool ok = true;
    size_t flat_idx = 0;
    for (const auto & c : tier_view_.attn_caches) {
        if (c.is_swa) continue;
        for (const auto & layer : c.layers) {
            const uint8_t * k_src = warm_buf_.data()
                                    + warm_layer_off_[flat_idx]
                                    + (size_t) warm_slot * layer.k_row_bytes;
            const uint8_t * v_src = warm_buf_.data()
                                    + warm_layer_v_off_[flat_idx]
                                    + (size_t) warm_slot * layer.v_row_bytes;
            if (!store_.write_attn_k((int) flat_idx, victim_pos, k_src, layer.k_row_bytes) ||
                !store_.write_attn_v((int) flat_idx, victim_pos, v_src, layer.v_row_bytes)) {
                ok = false;
                break;
            }
            ++flat_idx;
        }
        if (!ok) break;
    }

    if (!ok) {
        LLAMA_LOG_WARN("mt::spill_one_to_cold: KvtcStore write failed for pos %d; "
                       "leaving in warm\n", victim_pos);
        return false;
    }

    // Free the warm slot, drop from warm bookkeeping, add to cold.
    warm_free_slots_.push_back((int) warm_slot);
    warm_pos_to_slot_.erase(it);
    evicted_to_warm_.erase(victim_pos);
    warm_insertion_order_.pop_front();
    cold_positions_.insert(victim_pos);

    capacity_.on_migrate(1, TierCapacityManager::Tier::Warm,
                            TierCapacityManager::Tier::Cold);
    return true;
}

// Pull one position from cold into warm. Reads K/V for every restorable
// layer from KvtcStore into a freshly-allocated warm slot. If warm is
// full, recursively spills another entry to cold to make room.
//
// On success: warm_pos_to_slot_ has the position, cold_positions_ does
// not, capacity counters reflect the migration. On failure (KvtcStore
// read error or permanent warm-full), returns false.
bool llama_memory_tiered::load_one_from_cold(llama_pos pos) {
    if (cold_positions_.find(pos) == cold_positions_.end()) {
        return false;
    }
    if (!store_initialized_) return false;

    // Need a free warm slot. If none, evict another warm entry to cold first.
    if (warm_free_slots_.empty()) {
        if (!spill_one_to_cold()) {
            LLAMA_LOG_WARN("mt::load_one_from_cold: warm full and spill failed; "
                           "cannot load pos %d\n", pos);
            return false;
        }
    }
    const int warm_slot = warm_free_slots_.back();
    warm_free_slots_.pop_back();

    // Read K/V for every restorable layer.
    bool ok = true;
    size_t flat_idx = 0;
    for (const auto & c : tier_view_.attn_caches) {
        if (c.is_swa) continue;
        for (const auto & layer : c.layers) {
            uint8_t * k_dst = warm_buf_.data()
                              + warm_layer_off_[flat_idx]
                              + (size_t) warm_slot * layer.k_row_bytes;
            uint8_t * v_dst = warm_buf_.data()
                              + warm_layer_v_off_[flat_idx]
                              + (size_t) warm_slot * layer.v_row_bytes;
            if (!store_.read_attn_k((int) flat_idx, pos, k_dst, layer.k_row_bytes) ||
                !store_.read_attn_v((int) flat_idx, pos, v_dst, layer.v_row_bytes)) {
                ok = false;
                break;
            }
            ++flat_idx;
        }
        if (!ok) break;
    }

    if (!ok) {
        warm_free_slots_.push_back(warm_slot);
        LLAMA_LOG_WARN("mt::load_one_from_cold: KvtcStore read failed for pos %d\n", pos);
        return false;
    }

    // Promote to warm.
    warm_pos_to_slot_[pos] = (uint32_t) warm_slot;
    evicted_to_warm_.insert(pos);
    warm_insertion_order_.push_back(pos);
    cold_positions_.erase(pos);

    capacity_.on_migrate(1, TierCapacityManager::Tier::Cold,
                            TierCapacityManager::Tier::Warm);
    return true;
}

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

std::vector<float> llama_memory_tiered::embed_text(const std::string & text) {
    if (cfg_.semantic_index.empty()) {
        return {};
    }
    if (!embed_model_) {
        embed_model_ = std::make_unique<EmbeddingModel>(cfg_.semantic_index);
    }
    return embed_model_->embed(text);
}

bool llama_memory_tiered::has_warm_recurrent(llama_seq_id seq_id) const {
    return warm_recur_buf_.find(seq_id) != warm_recur_buf_.end();
}

bool llama_memory_tiered::restore_recurrent_from_warm(llama_seq_id seq_id) {
    auto it = warm_recur_buf_.find(seq_id);
    if (it == warm_recur_buf_.end()) {
        return false;  // no backup
    }

    // Allocate a fresh slot in the inner recurrent cache.
    const int slot = inner_->mt_restore_recurrent_slot(seq_id);
    if (slot < 0) {
        LLAMA_LOG_WARN("mt::restore_recurrent_from_warm: no free slot for seq %d "
                       "(cache full or backend has no recurrent state)\n", seq_id);
        return false;
    }

    // Refresh the view to find the freshly-allocated RecurrentStateView.
    // We need the per-layer pointers and the slot index it picked.
    const auto fresh = inner_->make_tier_view();
    const RecurrentStateView * target = nullptr;
    for (const auto & rs : fresh.recur_seqs) {
        if (rs.seq_id == seq_id && rs.seq_slot == slot) {
            target = &rs;
            break;
        }
    }
    if (!target) {
        LLAMA_LOG_WARN("mt::restore_recurrent_from_warm: inner allocated slot %d "
                       "but make_tier_view didn't surface it for seq %d\n",
                       slot, seq_id);
        return false;
    }

    if (!mover_recur_.restore_seq(*target, it->second.data())) {
        LLAMA_LOG_WARN("mt::restore_recurrent_from_warm: mover failed for seq %d\n", seq_id);
        return false;
    }

    LLAMA_LOG_INFO("mt::restore_recurrent_from_warm: seq %d restored "
                   "(%.1f MiB) into slot %d\n",
                   seq_id, (double) it->second.size() / (1024.0 * 1024.0), slot);

    // Drop the warm copy. Recurrent state is per-seq atomic — a future
    // wipe will back it up fresh.
    warm_recur_buf_.erase(it);

    capacity_.on_migrate(1, TierCapacityManager::Tier::Warm,
                            TierCapacityManager::Tier::Hot);
    return true;
}

// ---- public tier-restore API ----

bool llama_memory_tiered::has_warm(llama_seq_id /*seq_id*/, llama_pos position) const {
    // NOTE: seq_id is currently ignored — warm_pos_to_slot_ / cold_positions_
    // are keyed by position only. This is correct for single-seq workloads
    // (the most common case, including llama-cli single-turn and most
    // server slots). Multi-seq separation requires keying by (seq, pos)
    // and is a documented follow-up.
    //
    // "warm" in the API name is historical — this query returns true for
    // anything in warm OR cold, since restore_from_warm transparently
    // demand-loads from cold via load_one_from_cold.
    return warm_pos_to_slot_.find(position) != warm_pos_to_slot_.end()
        || cold_positions_.find(position)   != cold_positions_.end();
}

uint32_t llama_memory_tiered::restore_from_warm(llama_seq_id                   seq_id,
                                                  const std::vector<llama_pos> & positions) {
    if (!inner_ || !warm_initialized_ || positions.empty()) return 0;

    uint32_t restored = 0;
    for (auto pos : positions) {
        // Demand-load from cold if the position is there but not in warm.
        if (warm_pos_to_slot_.find(pos) == warm_pos_to_slot_.end() &&
            cold_positions_.find(pos)   != cold_positions_.end()) {
            if (!load_one_from_cold(pos)) continue;
        }

        auto it = warm_pos_to_slot_.find(pos);
        if (it == warm_pos_to_slot_.end()) continue;  // not in warm or cold
        const uint32_t warm_slot = it->second;

        // Ask the inner cache for a free slot tagged with this position.
        // For iswa caches this returns a slot in the base cache only;
        // make_tier_view's first non-SWA AttentionCacheSnapshot is the
        // one that owns this slot.
        const int inner_slot = inner_->mt_restore_tag_slot(seq_id, pos);
        if (inner_slot < 0) {
            LLAMA_LOG_WARN("mt::restore_from_warm: no free slot in inner cache for "
                           "pos %d (seq %d)\n", pos, seq_id);
            break;
        }

        // Walk the same restorable-layer order ensure_warm_staging used
        // when laying out warm_buf_. Per-cache layers are contiguous in
        // the warm slab; flat_idx is the global non-SWA layer index.
        bool ok = true;
        size_t flat_idx = 0;
        for (const auto & c : tier_view_.attn_caches) {
            if (c.is_swa) continue;
            for (const auto & layer : c.layers) {
                const uint8_t * src_k = warm_buf_.data()
                                        + warm_layer_off_[flat_idx]
                                        + (size_t) warm_slot * layer.k_row_bytes;
                const uint8_t * src_v = warm_buf_.data()
                                        + warm_layer_v_off_[flat_idx]
                                        + (size_t) warm_slot * layer.v_row_bytes;
                if (!mover_attn_.restore_k(layer, src_k, inner_slot) ||
                    !mover_attn_.restore_v(layer, src_v, inner_slot)) {
                    ok = false;
                    break;
                }
                ++flat_idx;
            }
            if (!ok) break;
        }

        if (ok) {
            ++restored;
            // Keep the warm copy: the same chunk may be requested again
            // (e.g. another seq with the same prefix). forget_warm()
            // is the explicit release.
        } else {
            LLAMA_LOG_WARN("mt::restore_from_warm: mover failed restoring pos %d "
                           "into slot %d; some layers may be inconsistent\n",
                           pos, inner_slot);
            // We can't easily roll back the inner_slot tagging without
            // exposing more API surface — leave it. In practice failure
            // here means HIP error and the whole context is suspect.
        }
    }

    if (restored > 0) {
        capacity_.on_migrate(restored,
                             TierCapacityManager::Tier::Warm,
                             TierCapacityManager::Tier::Hot);
        // Refresh tier metadata: the restored positions are now in hot
        // and policy scoring should consider them.
        update_tier_state();
        LLAMA_LOG_INFO("mt::restore_from_warm: restored %u positions for seq %d "
                       "(warm now %u/%u)\n",
                       restored, seq_id,
                       warm_capacity_ - (uint32_t) warm_free_slots_.size(),
                       warm_capacity_);
    }
    return restored;
}

void llama_memory_tiered::record_chunk_fingerprint(std::vector<llama_pos> positions,
                                                     std::vector<float>     embedding,
                                                     SemanticIndex::Tier    tier) {
    semantic_.add_fingerprint(std::move(positions), std::move(embedding), tier);
}

std::vector<SemanticIndex::Hint>
llama_memory_tiered::find_similar_chunks(const std::vector<float> & query_embedding,
                                          int                        top_k,
                                          float                      threshold) const {
    return semantic_.score(query_embedding, top_k, threshold);
}

uint32_t llama_memory_tiered::restore_semantic(llama_seq_id              seq_id,
                                                 const std::vector<float> & query_embedding,
                                                 int                        top_k,
                                                 float                      threshold) {
    auto hints = find_similar_chunks(query_embedding, top_k, threshold);
    if (hints.empty()) return 0;

    // Flatten hint position lists. Drop positions already resident in
    // hot (we'd otherwise double-allocate slots) and dedupe across
    // hints in case two semantic matches overlap.
    std::vector<llama_pos> wanted;
    std::unordered_set<llama_pos> seen;
    for (const auto & h : hints) {
        for (auto p : h.positions) {
            if (seen.insert(p).second && has_warm(seq_id, p)) {
                wanted.push_back(p);
            }
        }
    }
    if (wanted.empty()) return 0;

    // Conflict resolution: if the inner cache currently holds any of
    // our target positions (e.g. post-shift content occupying the
    // slots whose original tokens we're trying to restore), we must
    // free them before restoration. Calling seq_rm here trips our
    // own backup hook, so the displaced content lands in warm and
    // becomes recoverable in turn — symmetric, no data loss.
    //
    // We coalesce wanted into contiguous ranges so seq_rm is called
    // O(ranges) times, not O(positions). Sort first.
    std::vector<llama_pos> sorted_wanted = wanted;
    std::sort(sorted_wanted.begin(), sorted_wanted.end());

    // Snapshot inner cells once and check which wanted positions
    // collide with currently-occupied (seq_id) slots.
    const auto fresh = inner_->make_tier_view();
    std::unordered_set<llama_pos> in_inner;
    for (const auto & c : fresh.attn_caches) {
        if (c.is_swa) continue;
        for (const auto & cs : c.cells) {
            if (cs.pos >= 0 && cs.seq_id == seq_id) {
                in_inner.insert(cs.pos);
            }
        }
    }

    // Walk sorted_wanted, build contiguous conflict ranges, seq_rm them.
    size_t i = 0;
    while (i < sorted_wanted.size()) {
        if (!in_inner.count(sorted_wanted[i])) { ++i; continue; }
        size_t j = i + 1;
        while (j < sorted_wanted.size() &&
               sorted_wanted[j] == sorted_wanted[j-1] + 1 &&
               in_inner.count(sorted_wanted[j])) {
            ++j;
        }
        // [i, j) is a contiguous run of colliding positions.
        seq_rm(seq_id, sorted_wanted[i], sorted_wanted[j-1] + 1);
        i = j;
    }

    LLAMA_LOG_INFO("mt::restore_semantic: %zu hints -> %zu positions to restore "
                   "(seq %d, top_k=%d, threshold=%.2f)\n",
                   hints.size(), wanted.size(), seq_id, top_k, threshold);

    return restore_from_warm(seq_id, wanted);
}

void llama_memory_tiered::forget_warm(const std::vector<llama_pos> & positions) {
    for (auto pos : positions) {
        auto it = warm_pos_to_slot_.find(pos);
        if (it != warm_pos_to_slot_.end()) {
            warm_free_slots_.push_back((int) it->second);
            warm_pos_to_slot_.erase(it);
            evicted_to_warm_.erase(pos);
            // Also lazy-purge from FIFO; spill_one_to_cold tolerates
            // stale entries by re-trying.
            capacity_.on_remove_warm(1);
        }
        if (cold_positions_.erase(pos) > 0 && store_initialized_) {
            // Drop cold entries for every restorable layer.
            size_t flat_idx = 0;
            for (const auto & c : tier_view_.attn_caches) {
                if (c.is_swa) continue;
                for (size_t li = 0; li < c.layers.size(); ++li) {
                    store_.erase_attn((int) flat_idx, pos);
                    ++flat_idx;
                }
            }
            // No on_remove_cold helper yet; capacity counter is best-effort
            // because cold size isn't load-bearing for any current logic.
        }
    }
}

// Back up the entire recurrent state for seq_id before upstream's
// seq_rm wipes it. Migration unit is whole-sequence, not per-position
// — recurrent state (Mamba/DeltaNet/RWKV/linear-attention) is an
// accumulated function of all positions, so partial-range copies
// don't make architectural sense. Driven by seq_rm hooks the same
// way as the attention path.
//
// Per-seq host buffer layout matches RecurrentStateMover::evict_seq:
//     [r tensors of all layers concatenated]
//     [s tensors of all layers concatenated]
//
// For Qwen3.5-REAP-97B (36 recurrent layers) this is ~149 MiB per
// seq. For most workloads only one or two seqs are active at once;
// the buffer count is bounded by n_seq_max.
bool llama_memory_tiered::backup_seq_rm_recurrent(llama_seq_id seq_id) {
    if (!inner_) return false;

    // Fresh view to find this seq's RecurrentStateView. Recurrent
    // state moves around as seqs come and go; the cached tier_view_
    // from ctor is stale.
    const auto fresh = inner_->make_tier_view();
    if (fresh.recur_seqs.empty()) return false;  // not a recurrent model

    const RecurrentStateView * found = nullptr;
    for (const auto & rs : fresh.recur_seqs) {
        if (rs.seq_id == seq_id) {
            found = &rs;
            break;
        }
    }
    if (!found) return false;  // seq has no active recurrent state

    const size_t total = found->r_bytes_total + found->s_bytes_total;
    if (total == 0) return false;

    // Skip if we already have a backup for this seq from a prior
    // wipe (forget_warm clears it; otherwise it's stale-but-present).
    if (warm_recur_buf_.count(seq_id)) return false;

    auto & buf = warm_recur_buf_[seq_id];
    buf.assign(total, 0);
    if (!mover_recur_.evict_seq(*found, buf.data())) {
        warm_recur_buf_.erase(seq_id);
        LLAMA_LOG_WARN("mt::backup_seq_rm_recurrent: mover failed for seq %d "
                       "(%zu bytes); recurrent state will be lost\n",
                       seq_id, total);
        return false;
    }

    LLAMA_LOG_INFO("mt::backup_seq_rm_recurrent: seq %d backed up "
                   "(%.1f MiB across %zu layers)\n",
                   seq_id, (double) total / (1024.0 * 1024.0),
                   found->layers.size());
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

            // If warm is full, try to make room by spilling to cold.
            // If that also fails (no cold storage / SSD error), give up
            // on the rest of this seq_rm — the remaining positions are
            // lost.
            if (warm_free_slots_.empty() && !spill_one_to_cold()) {
                LLAMA_LOG_DEBUG("mt::backup_seq_rm_range: warm full and cold "
                                "spill unavailable; remaining positions lost\n");
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
                warm_insertion_order_.push_back(cs.pos);
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
    warm_insertion_order_.clear();
    cold_positions_.clear();
    warm_recur_buf_.clear();
    if (warm_initialized_) {
        // Refill the free-slot stack to its full warm capacity.
        warm_free_slots_.clear();
        warm_free_slots_.reserve(warm_capacity_);
        for (uint32_t s = warm_capacity_; s-- > 0; ) {
            warm_free_slots_.push_back((int) s);
        }
    }
    // KvtcStore's index is dropped, but its file persists with stale
    // payloads. We don't reclaim the disk space here — clear() can
    // happen mid-run, and re-opening the file on next backup re-uses
    // the descriptor and just appends. (This means clear() does leak
    // some on-disk space until shutdown; acceptable for a tier cache.)
    //
    // The store_'s in-memory index is what matters for lookup; we wipe
    // that by closing and re-init-on-demand.
    if (store_initialized_) {
        store_.shutdown();
        store_initialized_ = false;
    }
}

bool llama_memory_tiered::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (!inner_) return false;

    // Tier backup: capture K/V for positions [p0, p1) BEFORE delegating
    // to the inner cache. After inner_->seq_rm the slots are reusable
    // and reading from them returns whatever the next batch wrote.
    //
    // p0 < 0 / p1 < 0 are sentinels meaning "from the beginning" /
    // "to the end" (whole-seq wipe). For attention, those typically map
    // to clear/reset events where per-position restoration isn't useful.
    // For recurrent state the opposite is true — whole-seq wipes are
    // exactly when per-seq state needs to be captured before it's lost.
    if (p0 >= 0 && p1 > p0) {
        backup_seq_rm_range(seq_id, p0, p1);
    }

    // Recurrent state: capture the seq's whole r+s state on whole-seq
    // wipes (sentinel p0/p1) and on partial wipes that include the seq
    // tail. The implementation conservatively backs up on every seq_rm
    // call against this seq_id — it's idempotent (no-op when already
    // backed up) and the cost is bounded by n_seq_max.
    if (seq_id >= 0) {
        backup_seq_rm_recurrent(seq_id);
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
