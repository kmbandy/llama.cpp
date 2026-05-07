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
//   - n_seq_max constrained by the defensive assert in mt::

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include "memory-tier/mt-block-pool.h"
#include "memory-tier/mt-block-table.h"

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

    llama_kv_cache_paged(
            const llama_model & model,
            ggml_backend_buffer_type_t buft,
            uint32_t                   n_blocks_total,   // total physical blocks (GPU pool size)
            uint32_t                   block_size,       // tokens per block
            uint32_t                   n_seq_max,
            uint32_t                   max_blocks_per_seq);

    ~llama_kv_cache_paged() override;

    // Accessors used by the graph builder + kernel dispatch.
    uint32_t          block_size()           const { return block_size_; }
    uint32_t          n_blocks_total()       const { return n_blocks_total_; }
    uint32_t          n_layers()             const { return (uint32_t) layers_.size(); }
    uint32_t          n_seq_max()            const { return n_seq_max_; }
    uint32_t          max_blocks_per_seq()   const { return max_blocks_per_seq_; }
    const layer_storage & layer(uint32_t il) const { return layers_[il]; }

    // The 3 input tensors the graph wires into ggml_paged_attn_mt for
    // every attention call. Kept resident; updated by prepare_batch.
    ggml_tensor * block_table_tensor() const { return block_table_; }
    ggml_tensor * context_lens_tensor() const { return context_lens_; }
    ggml_tensor * q_lens_tensor()      const { return q_lens_; }

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

    const llama_model &        model_;
    ggml_backend_buffer_type_t buft_;
    uint32_t                   n_blocks_total_;
    uint32_t                   block_size_;
    uint32_t                   n_seq_max_;
    uint32_t                   max_blocks_per_seq_;

    // Per-layer K/V storage.
    std::vector<layer_storage> layers_;

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
    ggml_tensor *         context_lens_   = nullptr;  // [n_seq_max] i32
    ggml_tensor *         q_lens_         = nullptr;  // [n_seq_max] i32

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
    size_t                    i_ubatch_ = 0;
    llama_memory_status       status_;
};
