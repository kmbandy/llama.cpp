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
#include "mt-embed.h"
#include "mt-eviction.h"
#include "mt-mover-attn.h"
#include "mt-mover-recurrent.h"
#include "mt-kvtc-store.h"
#include "mt-semantic.h"

#include "llama-memory.h"

#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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

    // ---- public tier-restore API ----
    //
    // Used by upstream (server-context's slot prefix-matching, or test
    // harnesses) to learn what positions are recoverable from warm and
    // to trigger restoration before submitting a batch.

    // True iff the wrapper has a warm-tier copy of (seq_id, position).
    // Caller can include such positions in the cached-prefix length
    // when computing what tokens to send to llama_decode.
    bool has_warm(llama_seq_id seq_id, llama_pos position) const;

    // Restore the listed positions from warm to the inner cache. Returns
    // the count actually restored. Caller is responsible for ensuring
    // the inner cache doesn't already hold those positions (otherwise
    // duplicates would be created); the typical caller already knows
    // this from its own prefix-match logic.
    //
    // After successful restore, the warm slot remains held — the same
    // chunk can be restored again in a future query without re-fetching.
    // Use forget_warm to release it.
    uint32_t restore_from_warm(llama_seq_id                  seq_id,
                                const std::vector<llama_pos> & positions);

    // Release warm-tier copies of the given (seq_id, positions). Used
    // after the upstream lifecycle determines a chunk will not be needed
    // again (or to make room when warm is full). After this, has_warm()
    // returns false for those (seq_id, position) pairs.
    void forget_warm(llama_seq_id seq_id, const std::vector<llama_pos> & positions);

    // ---- public semantic-restore API ----
    //
    // Caller provides L2-normalized embeddings; the wrapper stores them
    // in its internal SemanticIndex and uses them to find chunks that
    // semantically match a future query. The embedding model itself
    // lives outside the wrapper (loaded by the integration layer —
    // typically a small CPU model like bge-small or nomic-embed-text).

    // Record a fingerprint for a chunk. positions identifies the cells
    // covered by the chunk; embedding should be L2-normalized (caller's
    // responsibility). tier records where the chunk currently lives so
    // the policy can prefer chunks already cheap to fetch.
    void record_chunk_fingerprint(std::vector<llama_pos> positions,
                                  std::vector<float>     embedding,
                                  SemanticIndex::Tier    tier);

    // Find the top-K semantically similar chunks to query_embedding,
    // filtered to cosine similarity >= threshold. Returns hints in
    // descending-score order.
    std::vector<SemanticIndex::Hint>
    find_similar_chunks(const std::vector<float> & query_embedding,
                        int                        top_k     = 5,
                        float                      threshold = 0.65f) const;

    // Convenience: find_similar_chunks + restore_from_warm in one call.
    // Returns total positions actually restored. Skips hints whose
    // positions are already in the inner cache (caller can avoid that
    // double-allocation by querying has_warm before restore).
    uint32_t restore_semantic(llama_seq_id              seq_id,
                              const std::vector<float> & query_embedding,
                              int                        top_k     = 5,
                              float                      threshold = 0.65f);

    // Restore the warm-tier recurrent state for seq_id back into the
    // inner cache. Allocates a fresh recurrent slot via the inner
    // cache's mt_restore_recurrent_slot, then copies the stored r/s
    // bytes via RecurrentStateMover. Returns true on success; false
    // if seq_id has no warm recurrent backup, the cache is full, or
    // the mover fails. The warm copy is dropped after a successful
    // restore (single-shot: per-seq state is unique, no reuse).
    bool restore_recurrent_from_warm(llama_seq_id seq_id);

    // Diagnostic: does seq_id have a warm-tier recurrent state backup
    // available? Pairs with restore_recurrent_from_warm so callers can
    // check before attempting a restore.
    bool has_warm_recurrent(llama_seq_id seq_id) const;

    // **Proactive eviction backup** — public entry to the same backup
    // pipeline that runs at seq_rm time, but without requiring an actual
    // seq_rm. Use case: hybrid models (Qwen3.6, future DeepSeek V4) auto-
    // disable ctx_shift because their recurrent state can't support
    // partial seq_rm. Without this entry, mt::'s backup never fires for
    // those architectures and the cold tier stays empty.
    //
    // Caller (the server-side proactive trigger when slot capacity
    // crosses ~80%) supplies the position range to back up — typically
    // the OLDEST 20% of the slot's tokens. Backup writes attention K/V
    // to warm; when warm is full, spills to cold via KvtcStore. Returns
    // the count actually backed up.
    //
    // After backup completes, frees the hot-tier attention slots so the
    // reclaimed positions can be reused by subsequent prefill — that's
    // the whole point of "proactive eviction." For hybrids we reach past
    // the wrapper to mem_attn->seq_rm directly to leave recurrent state
    // alone (upstream's hybrid::seq_rm aborts on partial ranges since
    // recurrent can't be sliced positionally). For pure-attention models
    // inner_->seq_rm works as usual.
    uint32_t backup_proactive(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    // Physical attention cache cell count from the inner view (base,
    // non-SWA). For hybrid models this is much smaller than the user-
    // facing context size — recurrent layers carry the long context, so
    // the attention KV cache is sized for a sliding window. Server-side
    // eviction triggers should compare against THIS, not slot.n_ctx,
    // otherwise the trigger fires too late and the inner cache 500s.
    // Returns 0 if pure-recurrent or the inner backend isn't tierable.
    // Cached from tier_view_ at init — cheap, no allocation.
    uint32_t physical_attn_cells() const;

    // MAD-134: warm the bge-small embedding model at construction time
    // so the first user prompt doesn't pay the lazy-load cost
    // (~200ms cold). Called from the ctor when semantic_index is set.
    // Failures are non-fatal — lazy path still works.
    void warmup_embed_();

    // Compute an L2-normalized embedding for `text` via the embedding
    // model loaded from cfg_.semantic_index. Lazy-initializes the model
    // on first call. Returns empty vector if the model failed to load
    // or the path was empty.
    //
    // Typical use: caller feeds the original chunk text at backup time
    // to record_chunk_fingerprint, and the new query text to
    // restore_semantic. The wrapper itself does NOT compute fingerprints
    // automatically — token-IDs-to-text decoding requires upstream
    // (e.g. server-context's slot.prompt.tokens) since the inner cache
    // doesn't store the original token IDs.
    std::vector<float> embed_text(const std::string & text);

private:
    // Poll inner_->seq_pos_min / seq_pos_max across 0..n_seq_max_, sum
    // active token counts, push the result into capacity_, and refresh
    // eviction_ position metadata for any newly observed tokens. Logs
    // a one-shot message when hot_pressure() flips from off to on.
    void update_tier_state();

    // Allocate the warm-tier host staging buffer the first time it's
    // needed. Sized for warm_capacity tokens × Σ(k_row + v_row) over
    // all attention layers, populated from the cached tier view.
    // Returns false if there are no attention layers to back up
    // (recurrent-only models — handled in 2d-recur).
    bool ensure_warm_staging();

    // Open KvtcStore on the configured SSD path the first time cold
    // is needed. Returns false if cold_capacity is 0 or KvtcStore::init
    // fails.
    bool ensure_cold_store();

    // Spill ONE warm entry to cold to make room. Picks the oldest by
    // FIFO (warm_insertion_order_). Writes K and V for every restorable
    // layer to KvtcStore, frees the warm slot, records the position in
    // cold_positions_. Returns true on success.
    bool spill_one_to_cold();

    // Pull a (seq_id, position) from cold into a free warm slot. Used
    // when restore is requested for a position that lives in cold. May
    // trigger spill_one_to_cold internally if warm is full. Returns true
    // if the position is now in warm and warm_pos_to_slot_[seq_id] has
    // its slot.
    bool load_one_from_cold(llama_seq_id seq_id, llama_pos pos);

    // Back up positions [p0, p1) of seq_id from hot to warm before
    // upstream's seq_rm frees them. Refreshes the cell snapshot to map
    // each position to its current slot, copies K/V from every non-SWA
    // attention cache via AttentionMover, and records the position->slot
    // mapping in warm_pos_to_slot_ so the future restore path can find
    // the backup data. Returns the count actually backed up. Skips
    // positions already in warm (e.g. seq_rm called twice on the same
    // range — second call is a no-op).
    uint32_t backup_seq_rm_range(llama_seq_id seq_id, llama_pos p0, llama_pos p1);

    // Back up the entire recurrent state for seq_id before upstream's
    // seq_rm wipes it. Migration unit is whole-sequence (the state is
    // an accumulated function of all positions so partial-range copies
    // don't make sense). Stores a contiguous host buffer of r+s data
    // keyed by seq_id in warm_recur_buf_. Returns true iff something
    // was actually backed up (false when the model has no recurrent
    // state, or the seq has no active state in the cache).
    bool backup_seq_rm_recurrent(llama_seq_id seq_id);

    llama_memory_ptr      inner_;
    TieredConfig          cfg_;
    uint32_t              n_seq_max_;

    TierCapacityManager   capacity_;
    TokenMetadataStore    eviction_;
    AttentionMover        mover_attn_;
    RecurrentStateMover   mover_recur_;
    KvtcStore             store_;
    SemanticIndex         semantic_;

    // Cached tier view captured at construction. Pointers stay stable
    // for the lifetime of inner_ — see mt-inner-access.h.
    InnerView             tier_view_;

    // Lazy-loaded embedding model (only constructed when
    // cfg_.semantic_index is non-empty). nullptr otherwise.
    std::unique_ptr<EmbeddingModel> embed_model_;

    // Warm-tier host staging. Lazily allocated on first eviction.
    // Per-attn-layer base offsets:
    //   warm_buf_[layer_off_[i]] ... + warm_cap * k_row_bytes_[i]   = K slab
    //   ... + warm_cap * v_row_bytes_[i]                            = V slab
    std::vector<uint8_t>  warm_buf_;
    std::vector<size_t>   warm_layer_off_;   // byte offset of layer i's K slab
    std::vector<size_t>   warm_layer_v_off_; // byte offset of layer i's V slab within layer
    uint32_t              warm_capacity_ = 0;
    bool                  warm_initialized_ = false;

    // ─── Per-seq tier metadata ───────────────────────────────────────
    // All vectors are sized to n_seq_max_ at construction. Index by
    // seq_id, then by pos (for the maps/sets). With multi-seq enabled
    // (--parallel > 1), two sequences can independently hold position 5
    // without their tier state colliding.
    //
    // The slot pool (warm_free_slots_) and the global FIFO
    // (warm_insertion_order_) remain shared across seqs because the
    // underlying warm_buf_ is a single shared slot space — disambiguation
    // happens via the per-seq maps that point into it.

    // pos -> slot index for positions currently in warm, scoped per seq.
    std::vector<std::unordered_map<llama_pos, uint32_t>> warm_pos_to_slot_;

    // Free slot stack. Slots are seq-agnostic — any seq can own any slot.
    std::vector<int>                        warm_free_slots_;

    // Per-seq mirror of warm_pos_to_slot_'s key set, kept for fast O(1)
    // skip in update_tier_state.
    std::vector<std::unordered_set<llama_pos>> evicted_to_warm_;

    // GLOBAL FIFO queue of (seq_id, pos) in insertion order, used to pick
    // spill victims when warm fills. Order is global because eviction
    // policy is "oldest across all seqs first" — a long-running seq A
    // shouldn't be sheltered from spill just because seq B has activity.
    // Deque so removal of restored entries from the front (after
    // load_one_from_cold) is cheap.
    std::deque<std::pair<llama_seq_id, llama_pos>> warm_insertion_order_;

    // Positions currently stored in cold tier (KvtcStore), scoped per
    // seq. Disjoint from warm_pos_to_slot_[seq]'s key set — a (seq, pos)
    // is in either warm or cold, never both. (load_one_from_cold removes
    // from cold and adds to warm; spill_one_to_cold does the inverse.)
    std::vector<std::unordered_set<llama_pos>> cold_positions_;

    // Per-seq recurrent state backups. Each entry is a contiguous host
    // buffer holding the full r+s state (layout per RecurrentStateMover:
    // all r layers concatenated, then all s layers). Sized to the seq's
    // r_bytes_total + s_bytes_total at backup time.
    std::unordered_map<llama_seq_id, std::vector<uint8_t>> warm_recur_buf_;

    bool                  store_initialized_  = false;
    bool                  pressure_announced_ = false;  // edge-trigger for hot_pressure logging
    bool                  no_attn_warned_     = false;  // one-shot: "no attn layers, can't migrate"
};

}  // namespace mt
