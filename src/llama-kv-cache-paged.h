#pragma once

// llama_kv_cache_paged — paged KV cache for the mt:: tier system.
//
// Owns GPU-resident block storage in vLLM-style layout:
//   K cache (per layer): [num_blocks, n_kv_heads, head_dim/x, block_size, x]
//   V cache (per layer): [num_blocks, n_kv_heads, head_dim, block_size]
// where x = 16/sizeof(scalar_t). The kernel in
// ggml/src/ggml-cuda/mt_pagedattn.{cuh,cu} reads this layout directly.
//
// This is the "for real B" implementation of the paged refactor — replaces
// the standard llama_kv_cache when --kv-tier-paged-blocks is set. The
// existing position-keyed mt:: tier wrapper still exists but is bypassed
// when this cache is in use.
//
// Design adapted from vLLM (Apache 2.0). Code is independent.
//
// PHASE 3.3 SCOPE:
//   - Constructor + GPU buffer allocation
//   - Per-batch block allocation + slot mapping
//   - llama_memory_i interface implementation (init_batch, seq_rm,
//     seq_cp, etc.)
//   - llama_kv_cache_paged_context for graph integration
//
// LIMITATIONS (will lift in later phases):
//   - F16 K/V only (no quantized cache yet)
//   - No prefix caching (no BlockHashToBlockMap)
//   - No CPU offload tier (warm tier inside paged context not yet wired)
//   - State persistence (state_write/state_read) — placeholder warns
//
// MULTI-SEQ STATUS:
//   compute_slot_mapping + init_batch* support n_seq_max > 1 — tokens
//   are bucketed per seq via ubatch->seq_id[i][0] and routed to the
//   appropriate logical block list. The earlier defensive assert is
//   gone; the cache works correctly with --parallel N for any N up to
//   n_seq_max_.

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include "memory-tier/mt-block-pool.h"
#include "memory-tier/mt-block-table.h"
#include "memory-tier/mt-semantic.h"

#include <functional>
#include <string>
#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_hparams;

class llama_kv_cache_paged_context;

class llama_kv_cache_paged : public llama_memory_i {
public:
    // Per-layer storage handles. Each layer gets a K and V buffer in
    // vLLM-style block layout, allocated as ggml tensors on the GPU
    // backend.
    struct layer_storage {
        ggml_tensor * k = nullptr;  // bytes, sized n_blocks * block_size * n_kv_heads * head_dim * sizeof(half)
        ggml_tensor * v = nullptr;  // same size, transposed-V layout

        uint32_t n_kv_heads  = 0;
        uint32_t head_dim    = 0;
    };

    using layer_filter_cb = std::function<bool(int32_t /*il*/)>;

    llama_kv_cache_paged(
            const llama_model & model,
            ggml_backend_buffer_type_t buft,
            uint32_t                   n_blocks_total,   // total physical blocks (GPU pool size)
            uint32_t                   block_size,       // tokens per block
            uint32_t                   n_seq_max,
            uint32_t                   max_blocks_per_seq,
            // MAD-120: host-side warm-tier capacity in blocks. 0 = disabled
            // (legacy single-pool behavior). When > 0, the cache can evict
            // GPU blocks to host RAM when the GPU pool fills, and auto-fault
            // them back in when needed for attention. Sized by --kv-tiered
            // warm_pct in the create_memory caller.
            uint32_t                   n_warm_blocks = 0,
            // MAD-121: cold-tier (SSD) capacity in blocks. 0 = disabled.
            // Backed by per-layer files at {ssd_path}/paged/L{il}.{k,v}.bin.
            // When the warm pool fills, the LRU warm block is spilled to a
            // free cold slot before further hot evictions proceed.
            uint32_t                   n_cold_blocks = 0,
            // Filesystem directory under which the cold-tier files are
            // created. Ignored when n_cold_blocks == 0. The "/paged"
            // subdir is created on demand.
            std::string                ssd_path = std::string(),
            // MAD-130: when true, skip the O_TRUNC on cold-tier files
            // and load the in-memory cold index from the sidecar file
            // at `${ssd_path}/paged/instance-${INSTANCE_ID}/index.bin`.
            // Lets the server resume cold-tier contents across a clean
            // restart. Default false (legacy behavior: fresh truncation).
            bool                       cold_resume = false,
            // MAD-131: per-instance ID. Cold-tier files live under
            // `${ssd_path}/paged/instance-${INSTANCE_ID}/`. Empty →
            // use the process pid as a string. Required to allow
            // multiple llama-server processes to share one ssd_path
            // without colliding on cold-tier files.
            std::string                instance_id = std::string(),
            // MAD-131: cap cold pool to this many MiB total (K+V across
            // all attn layers). 0 = no cap (size from cold percentage).
            // Lets operators bound SSD wear per instance.
            uint32_t                   cold_budget_mb = 0,
            // Optional per-layer filter. Returns true for layers that should
            // get K/V allocation in the paged pool. Recurrent layers in
            // hybrid models should be filtered OUT (their state lives in
            // mem_recr instead). nullptr = include all layers (back-compat).
            layer_filter_cb            filter = nullptr,
            // KV element types. Default F16 = current paged behavior.
            ggml_type                  type_k = GGML_TYPE_F16,
            ggml_type                  type_v = GGML_TYPE_F16);

    ~llama_kv_cache_paged() override;

    // Accessors used by the graph builder + kernel dispatch.
    uint32_t          block_size()           const { return block_size_; }
    uint32_t          n_blocks_total()       const { return n_blocks_total_; }
    uint32_t          n_layers()             const { return (uint32_t) layers_.size(); }
    uint32_t          n_seq_max()            const { return n_seq_max_; }
    uint32_t          max_blocks_per_seq()   const { return max_blocks_per_seq_; }
    const layer_storage & layer(uint32_t il) const { return layers_[il]; }

    // The block_table tensor is wired into ggml_paged_attn_mt for every
    // attention call; kept resident, updated by prepare_batch_tensors.
    ggml_tensor * block_table_tensor() const { return block_table_; }

    // MAD-114: context_lens / q_lens are NOT persistent any more — they
    // were causing a write-after-read hazard across forward passes (the
    // hybrid model issues prefill + decode back-to-back; the persistent
    // tensors got overwritten by pass-B's prepare while pass-A's late
    // attention layers were still reading them, on HIP/RDNA where
    // same-stream kernel ordering isn't honored). Now allocated per-graph
    // by build_attn_inp_kv_paged_impl + populated by set_input from these
    // host mirrors.
    const int32_t * h_context_lens_data() const { return h_context_lens_.data(); }
    size_t          h_context_lens_size() const { return h_context_lens_.size(); }
    const int32_t * h_q_lens_data()       const { return h_q_lens_.data(); }
    size_t          h_q_lens_size()       const { return h_q_lens_.size(); }

    // ─── llama_memory_i ───
    llama_memory_context_ptr init_batch(llama_batch_allocr & balloc,
                                         uint32_t             n_ubatch,
                                         bool                 embd_all) override;
    llama_memory_context_ptr init_full() override;
    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }  // paged doesn't need ctx shift

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id, llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // Allocate enough blocks to fit `n_new_tokens` for `seq_id`, append
    // them to the seq's logical block list. Returns false if the GPU
    // pool is exhausted (caller should evict OR this becomes admission
    // control later).
    bool ensure_blocks_for(llama_seq_id seq_id, uint32_t n_new_tokens);

    // Phase 3.4b-3: per-batch slot resolution. For each token i in
    // `ubatch`, write into out[i] the physical slot index that token
    // should land in: `physical_block_idx * block_size + slot_in_block`.
    // out must have at least ubatch->n_tokens entries. Returns false +
    // leaves out unspecified if any token resolves to an unallocated
    // logical block (loader bug — ensure_blocks_for should have run
    // first). For v1 single-seq, all tokens are assumed seq 0.
    bool compute_slot_mapping(const llama_ubatch * ubatch, int32_t * out) const;

    // Phase 3.6: for hybrid composition. Takes pre-built ubatches (e.g.
    // from a `balloc.split_equal()` driven by the recurrent half) and
    // does only the cache-side bookkeeping (block allocation + per-batch
    // tensor uploads). Mirrors init_batch's body but skips the splitting
    // step. Returns a paged context wrapping the same ubatches.
    llama_memory_context_ptr init_batch_with_ubatches(std::vector<llama_ubatch> ubatches);

    // ─── MAD-120: hot↔warm tier ───
    //
    // Move ONE GPU-resident block to host (warm) memory. After the call
    // the block is no longer on GPU; the seq's logical block list points
    // at the new CPU physical and reads need restore_block_from_warm to
    // bring it back. Returns false if the warm pool is full (caller
    // should escalate to cold or refuse the request).
    bool evict_block_to_warm(llama_seq_id seq_id, uint32_t logical_block);

    // Move ONE warm-resident block back to GPU. After the call the seq's
    // logical block list points at the new GPU physical and the kernel
    // can read it. Returns false if the GPU pool is full (caller should
    // evict another block first or refuse the request).
    bool restore_block_from_warm(llama_seq_id seq_id, uint32_t logical_block);

    // Pick an LRU GPU-resident logical block across all seqs and evict it
    // to warm. Used by ensure_blocks_for as the reactive trigger. Returns
    // true if a block was evicted, false if no GPU-resident block was
    // eligible (e.g. all seqs hold only their own most-recent block, or
    // warm is also full).
    bool evict_lru_to_warm();

    // ─── MAD-120 Phase 2: whole-slot preemption ───
    //
    // Move ALL of seq_id's GPU-resident blocks to warm in one go. Used by
    // the scheduler's admission control: when the active set wouldn't fit
    // in hot, the picked victim seq has its entire context paged out so
    // the remaining seqs can run. Returns the number of blocks moved (0
    // if the seq had nothing on GPU, negative on error). Idempotent:
    // calling twice with no intervening restore is a no-op.
    int  evict_seq_to_warm(llama_seq_id seq_id);

    // Bring ALL of seq_id's warm-resident blocks back to GPU. Caller must
    // first ensure GPU pool has space (the fault-in pre-pass handles this
    // by evict_lru_to_warm of OTHER seqs). Returns the number of blocks
    // restored, or negative on error (e.g. GPU pool ran out mid-restore).
    int  restore_seq_from_warm(llama_seq_id seq_id);

    // Count how many of seq_id's logical blocks are currently GPU-resident
    // vs warm-resident. Used by can_admit() to compute the working-set
    // delta when admitting a slot that's currently fully or partially in
    // warm.
    uint32_t n_gpu_blocks_for(llama_seq_id seq_id)  const;
    uint32_t n_warm_blocks_for(llama_seq_id seq_id) const;

    // Admission predicate. Given a set of seq_ids already accepted into
    // the upcoming batch (`accepted`) and a candidate `seq_id` that wants
    // to join (with `n_new_tokens` of fresh tokens it'll add this batch),
    // return true iff the union's hot-block requirement fits within the
    // hot pool. Used by the scheduler before adding a slot's tokens.
    //
    // The candidate's needed hot-block count is:
    //     ceil((seq_pos_max + n_new_tokens + 1) / block_size_)
    // i.e. all blocks the kernel will need to read+write for this seq.
    //
    // accepted seqs contribute their already-needed blocks. The total
    // must be ≤ n_blocks_total_ (hot capacity).
    bool can_admit(llama_seq_id seq_id, uint32_t n_new_tokens,
                   const std::vector<llama_seq_id> & accepted) const;

    // Public accessors for tests / dispatch decisions.
    uint32_t n_warm_blocks() const { return n_warm_blocks_; }
    bool     warm_enabled()  const { return n_warm_blocks_ > 0; }

    // ─── MAD-121: cold (SSD) tier ───
    //
    // Same shape as the warm-tier evict/restore but the data lives on
    // disk. Cold storage is O(1)-addressed by a fixed-size slot index;
    // see cold_*_ members below. Restoration is cold→hot directly
    // (saves the warm→hot hop when warm is full).
    bool evict_block_to_cold(llama_seq_id seq_id, uint32_t logical_block);
    bool restore_block_from_cold(llama_seq_id seq_id, uint32_t logical_block);

    // Pick an LRU warm-resident block and spill it to cold, freeing one
    // CPU slot. Used as the escalation when the warm pool is full and
    // we still need to evict a hot block. Returns true if a warm block
    // was spilled, false if no warm-resident block exists or cold is
    // full / disabled.
    bool evict_lru_warm_to_cold();

    uint32_t n_cold_blocks() const { return n_cold_blocks_; }
    bool     cold_enabled()  const { return n_cold_blocks_ > 0; }

    // ─── MAD-125: BGE-small semantic prefetch ───
    //
    // The cache holds an optional per-(seq, lblock) fingerprint store.
    // Server populates it at backup time (one BGE embedding per block);
    // at prefill arrival the server queries restore_semantic_paged with
    // a query embedding, and this method scores against the fingerprints,
    // picks the top-K matches above a cosine threshold, and faults each
    // matching block back from warm/cold to hot before kernel dispatch.
    //
    // Lifecycle: fingerprints follow blocks. Whole-seq wipe (clear,
    // seq_rm with full range) drops them; per-seq removal happens
    // automatically. No FIFO cap — memory grows with active context.

    // Record the fingerprint for one paged block. embedding should be
    // L2-normalized; tier annotates current location (informational).
    void record_paged_block_fingerprint(llama_seq_id        seq_id,
                                         uint32_t            lblock,
                                         std::vector<float>  embedding,
                                         mt::SemanticIndex::Tier tier);

    // Score the seq's paged-block fingerprints against query_embedding,
    // restore the top-K above threshold from warm (and cold as fallback)
    // back to hot. Returns the count of blocks actually restored. Logs
    // hit-rate (restored / requested) for MAD-122 acceptance criterion.
    uint32_t restore_semantic_paged(llama_seq_id              seq_id,
                                     const std::vector<float> & query_embedding,
                                     int                        top_k     = 5,
                                     float                      threshold = 0.65f);

    // Diagnostic: how many fingerprints currently held.
    size_t n_paged_fingerprints() const { return paged_semantic_.size(); }

    // MAD-129: O(1) check whether (seq_id, lblock) already has a
    // fingerprint. Used by the server's prefill-time write trigger to
    // skip blocks already fingerprinted on prior turns.
    bool has_paged_fingerprint(llama_seq_id seq_id, uint32_t lblock) const {
        return paged_semantic_.has_fingerprint(seq_id, lblock);
    }

    // MAD-130: persist the cold-tier index to a sidecar file at
    // `${cold_path_}/paged/index.bin`. Called by the server on graceful
    // shutdown or as part of /slots/save. Returns false on I/O error.
    // No-op when cold tier is disabled.
    bool save_cold_index_sidecar() const;

    // MAD-130: persist the BlockSemanticIndex (paged-block fingerprints)
    // to a sidecar file at the given path. Thin forwarders so the
    // server can save fingerprints alongside state_write without poking
    // at private members.
    bool save_paged_fingerprints(const std::string & path) const {
        return paged_semantic_.save_to_disk(path);
    }
    bool load_paged_fingerprints(const std::string & path) {
        return paged_semantic_.load_from_disk(path);
    }

private:
    friend class llama_kv_cache_paged_context;

    // Per-seq tracking. seq_pos_max is the highest position written for
    // the seq (inclusive). seq_pos_min stays 0 unless seq_rm carved off
    // the head.
    struct seq_state {
        llama_pos pos_min = -1;
        llama_pos pos_max = -1;
    };

    // Allocate and upload the block_table / context_lens / q_lens
    // tensors to the GPU for the current batch. Called by init_batch
    // before the graph runs.
    void prepare_batch_tensors();

    // Multi-seq accounting for one ubatch: bucket tokens per seq, allocate
    // blocks for each affected seq, update seq_states_ pos tracking and
    // h_q_lens_. Returns false if any seq's block allocation fails (caller
    // should surface FAILED_PREPARE). Used by both init_batch and
    // init_batch_with_ubatches.
    bool apply_ubatch_to_state(const llama_ubatch & ub);

    // MAD-120: walk the ubatch's seqs and ensure all logical blocks
    // referenced (positions 0..max_pos for each seq) are GPU-resident.
    // For any block currently in warm, evict-LRU-non-needed-then-restore.
    // Returns false if the GPU pool cannot fit all needed blocks (e.g.
    // multiple seqs whose total active ctx exceeds the hot capacity).
    // No-op if warm tier is disabled.
    bool fault_in_warm_blocks_for_batch(const llama_ubatch & ub);

    // MAD-130: load the cold-tier sidecar from disk and rebuild
    // cold_slot_for_ + cold_pool_free_ + cold_in_use_. Called from the
    // ctor when cold_resume=true and the sidecar exists. Returns false
    // if the sidecar is missing/corrupt/mismatched-config — caller
    // should treat as "start fresh."
    bool load_cold_index_sidecar_(const std::string & path);

    // MAD-128: CoW any blocks being written this ubatch that are shared
    // (refcount > 1 from a prior seq_cp). For each (seq, lblock) pair
    // touched by a write in `ub`: if that physical block is shared,
    // allocate a fresh GPU block, copy the existing data into it, swap
    // the seq's table entry to point at the new block, and decrement
    // the old block's refcount. After this returns, every write target
    // is uniquely-owned and the kernel can write without corrupting
    // other sequences sharing the same prefix. Returns false if a CoW
    // alloc failed and eviction couldn't free a block.
    bool cow_writes_for_ubatch(const llama_ubatch & ub);

    const llama_model &        model_;
    ggml_backend_buffer_type_t buft_;
    uint32_t                   n_blocks_total_;
    uint32_t                   n_warm_blocks_      = 0;   // MAD-120: host-side warm-tier capacity
    uint32_t                   block_size_;
    uint32_t                   n_seq_max_;
    uint32_t                   max_blocks_per_seq_;
    ggml_type                  type_k_ = GGML_TYPE_F16;
    ggml_type                  type_v_ = GGML_TYPE_F16;

    // Per-layer K/V storage (GPU).
    std::vector<layer_storage> layers_;

    // MAD-120: per-layer host-side warm storage. Indexed [layer][cpu_block_idx].
    // cpu_block_idx = (cpu_physical - n_blocks_total_), i.e. the CPU pool's
    // 0-based offset within its half of the block_id space. Each block holds
    // K and V data laid out identically to the GPU layout (same byte size
    // per block; one hipMemcpy moves the whole thing).
    //
    // Sized at construction: n_warm_blocks_ × k_bytes_per_block_ for K, same
    // shape for V. Empty if n_warm_blocks_ == 0 (warm tier disabled).
    std::vector<std::vector<uint8_t>> warm_k_;  // [n_layers][n_warm_blocks * k_bytes_per_block]
    std::vector<std::vector<uint8_t>> warm_v_;  // [n_layers][n_warm_blocks * v_bytes_per_block]
    size_t                            k_bytes_per_block_ = 0;
    size_t                            v_bytes_per_block_ = 0;

    // MAD-121: cold-tier (SSD) storage. One file per (layer, K|V) at
    // {ssd_path}/paged/L{il}.{k,v}.bin, sized to n_cold_blocks ×
    // bytes_per_block at startup. A cold "slot" is a fixed offset
    // `cold_idx * bytes_per_block` into those files; cold_idx is in
    // [0, n_cold_blocks_) and allocated from cold_pool_free_ (LIFO
    // stack). Storage is bounded — no churn growth like an append-only
    // store would have.
    //
    // cold_slot_for_[seq_id][lblock] holds the assigned cold_idx when
    // the (seq, lblock) is cold-resident; kInvalidColdIdx otherwise.
    // table_'s mapping for cold-resident blocks is kInvalidBlockId
    // (the GPU/CPU physical was freed back to the pool when we spilled).
    //
    // For compression: turbo4 / Q8_0 caches store raw bytes (already
    // quantized at the cache layer; ~4x / ~2x smaller than F16). F16
    // cache cold compression is a follow-up (INT4 quantization on
    // write, dequantize on read) — not yet wired.
    static constexpr uint32_t kInvalidColdIdx = ~0u;
    uint32_t                  n_cold_blocks_ = 0;
    uint32_t                  cold_in_use_   = 0;
    std::string               cold_path_;

    // MAD-131: per-instance subdir + flock-based double-start protection.
    // instance_id_ defaults to the process pid as a string; --instance-id
    // overrides for deterministic restarts. cold_lock_fd_ holds the
    // OS-level lock on `${cold_path_}/paged/instance-${id}/.lock` for
    // the lifetime of the cache; the .pid file is informational. Both
    // get cleaned up by the destructor.
    std::string               instance_id_;
    std::string               cold_lock_path_;
    std::string               cold_pid_path_;
    int                       cold_lock_fd_  = -1;
    std::vector<int>          cold_fd_k_;            // [n_layers] fd for K cold file (-1 = none)
    std::vector<int>          cold_fd_v_;            // [n_layers] fd for V cold file
    std::vector<uint32_t>     cold_pool_free_;       // free cold_idx stack
    std::vector<std::vector<uint32_t>> cold_slot_for_;  // [seq_id][lblock] -> cold_idx

    // Backing ggml context + buffer for layer storage.
    ggml_context *      ctx_storage_   = nullptr;
    ggml_backend_buffer_t buf_storage_ = nullptr;

    // Block-table machinery (independent from mt::, since this owns the
    // cache outright). Single-pool: GPU only for v1.
    mt::BlockPool          pool_;
    mt::BlockTable         table_;
    mt::BlockSemanticIndex paged_semantic_;  // MAD-125

    // Per-seq position tracking.
    std::vector<seq_state> seq_states_;

    // Graph-input tensors (block_table / context_lens / q_lens). These
    // live in CPU + GPU mirror (CPU side written each batch, GPU copy
    // synchronized before kernel launch).
    ggml_context *        ctx_inputs_      = nullptr;
    ggml_backend_buffer_t buf_inputs_      = nullptr;
    ggml_tensor *         block_table_    = nullptr;  // [max_blocks_per_seq, n_seq_max] i32
    // MAD-114: context_lens / q_lens are no longer persistent — they're
    // allocated per-graph by the graph builder and populated by set_input
    // from h_context_lens_ / h_q_lens_ below.

    // Host-side mirrors for the input tensors. Updated by
    // prepare_batch_tensors then copied to GPU.
    std::vector<int32_t> h_block_table_;   // size = max_blocks_per_seq * n_seq_max
    std::vector<int32_t> h_context_lens_;  // size = n_seq_max
    std::vector<int32_t> h_q_lens_;        // size = n_seq_max
};

// Per-batch context — what the graph builder consumes during
// init_batch / next() / apply().
class llama_kv_cache_paged_context : public llama_memory_context_i {
public:
    llama_kv_cache_paged_context(llama_kv_cache_paged *           parent,
                                   std::vector<llama_ubatch>       ubatches,
                                   llama_memory_status              status = LLAMA_MEMORY_STATUS_SUCCESS);

    // MAD-124: success-path ctor that takes per-ubatch q_lens / context_lens
    // snapshots so apply() can install the right values before each ubatch's
    // graph runs. The two snapshot vectors must each have ubatches.size()
    // entries, indexed by ubatch order.
    llama_kv_cache_paged_context(llama_kv_cache_paged *               parent,
                                   std::vector<llama_ubatch>           ubatches,
                                   std::vector<std::vector<int32_t>>   q_lens_per_ubatch,
                                   std::vector<std::vector<int32_t>>   context_lens_per_ubatch,
                                   llama_memory_status                 status = LLAMA_MEMORY_STATUS_SUCCESS);

    bool                 next() override;
    bool                 apply() override;
    const llama_ubatch & get_ubatch() const override;
    llama_memory_status  get_status() const override { return status_; }

    // Phase 3.4b: graph builder needs to reach the parent cache to fetch
    // the per-batch input tensors (block_table / context_lens / q_lens)
    // that ggml_paged_attn_mt consumes.
    llama_kv_cache_paged * parent() const { return parent_; }

private:
    llama_kv_cache_paged *    parent_;
    std::vector<llama_ubatch> ubatches_;
    // MAD-124: per-ubatch q_lens / context_lens snapshots. apply() copies
    // q_lens_per_ubatch_[i_ubatch_] / context_lens_per_ubatch_[i_ubatch_]
    // into parent_->h_q_lens_ / h_context_lens_ before each ubatch's graph
    // runs, so set_input uploads the right per-ubatch values. Without this,
    // h_q_lens_ ends up holding only the LAST ubatch's counts after
    // init_batch's eager loop, and earlier ubatches' graphs read stale
    // q_lens — corrupting multi-seq prefill that splits across ubatches.
    std::vector<std::vector<int32_t>> q_lens_per_ubatch_;
    std::vector<std::vector<int32_t>> context_lens_per_ubatch_;
    size_t                    i_ubatch_ = 0;
    llama_memory_status       status_;
};
