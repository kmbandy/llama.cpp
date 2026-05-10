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

#include <functional>
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

    // Public accessors for tests / dispatch decisions.
    uint32_t n_warm_blocks() const { return n_warm_blocks_; }
    bool     warm_enabled()  const { return n_warm_blocks_ > 0; }

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

    // Backing ggml context + buffer for layer storage.
    ggml_context *      ctx_storage_   = nullptr;
    ggml_backend_buffer_t buf_storage_ = nullptr;

    // Block-table machinery (independent from mt::, since this owns the
    // cache outright). Single-pool: GPU only for v1.
    mt::BlockPool   pool_;
    mt::BlockTable  table_;

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
