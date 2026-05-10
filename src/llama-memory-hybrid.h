#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-paged.h"
#include "llama-memory.h"
#include "llama-memory-recurrent.h"

#include <memory>
#include <vector>

//
// llama_memory_hybrid
//

// utilizes instances of llama_memory_recurrent and llama_kv_cache to
//   support models where each layer may be either attention-based or recurrent

class llama_memory_hybrid : public llama_memory_i {
public:
    llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn = nullptr,
    const layer_filter_cb & filter_recr = nullptr,
                            /* Phase 3.6: paged attn (mt::). When n_blocks > 0
                               the attention sub-cache is constructed as
                               llama_kv_cache_paged instead of llama_kv_cache.
                               kv_size / n_pad / n_swa / swa_type are ignored
                               in the paged path. block_size and n_blocks come
                               from cparams.kv_tier_paged_blocks; see
                               llama-model.cpp create_memory for sizing. */
                 uint32_t   paged_n_blocks = 0,
                 uint32_t   paged_block_size = 16,
                 uint32_t   paged_max_blocks_per_seq = 0,
                 // MAD-120: host-side warm-tier capacity for the paged cache.
                 // 0 = warm tier disabled (legacy behavior). Sized by
                 // --kv-tiered warm_pct in create_memory.
                 uint32_t   paged_n_warm_blocks = 0,
                 // MAD-121: cold-tier (SSD) capacity + path. 0 = cold disabled.
                 uint32_t   paged_n_cold_blocks = 0,
                 std::string paged_ssd_path     = std::string());

    ~llama_memory_hybrid() = default;

    //
    // llama_memory_i
    //

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
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    // tier-system inner view — composite (attn + recurrent).
    mt::InnerView make_tier_view() const override;
    int mt_restore_tag_slot(llama_seq_id seq_id, llama_pos position) override;
    int mt_restore_recurrent_slot(llama_seq_id seq_id) override;

    //
    // llama_memory_hybrid specific API
    //

    llama_kv_cache * get_mem_attn() const;
    llama_memory_recurrent * get_mem_recr() const;

    // Phase 3.6: paged-attn variant. When the hybrid is constructed with
    // paged_n_blocks > 0, mem_attn_paged holds the cache and mem_attn is
    // null. is_paged_attn() returns true; get_mem_attn_paged() returns
    // the paged cache for the graph builder to thread tensors from.
    bool                     is_paged_attn() const { return mem_attn_paged != nullptr; }
    llama_kv_cache_paged *   get_mem_attn_paged() const { return mem_attn_paged.get(); }

private:
    const llama_hparams & hparams;

    // Mutually exclusive: either mem_attn (regular kv) is non-null, or
    // mem_attn_paged (Phase 3.6 paged variant) is non-null. is_paged_attn()
    // distinguishes; everything that touches the attn sub-cache must
    // branch on it.
    const std::unique_ptr<llama_kv_cache>           mem_attn;
    const std::unique_ptr<llama_kv_cache_paged>     mem_attn_paged;
    const std::unique_ptr<llama_memory_recurrent>   mem_recr;

    // Internal: returns whichever of mem_attn / mem_attn_paged is live,
    // upcast to the polymorphic base. Used by methods that only need
    // llama_memory_i functionality (clear, seq_*, state_*, memory_breakdown)
    // and don't care which concrete cache is behind it.
    llama_memory_i * attn_base() const;
};

class llama_memory_hybrid_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // init failure
    explicit llama_memory_hybrid_context(llama_memory_status status);

    // init full
    explicit llama_memory_hybrid_context(llama_memory_hybrid * mem);

    // init update
    explicit llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize);

    // init success
    llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches);

    // Phase 3.6: init success — paged attn variant. The attn sub-context
    // is pre-built (returned by mem_attn_paged->init_batch) instead of
    // being constructed from sinfos. Mutually exclusive with the regular
    // sinfos-based ctor.
    llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
        llama_memory_context_ptr    ctx_attn_paged,
        std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_context() = default;

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // TurboQuant: delegate to the KV cache context
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

    //
    // llama_memory_hybrid_context
    //

    const llama_kv_cache_context * get_attn() const;
    const llama_memory_recurrent_context * get_recr() const;

    // Phase 3.6: when the underlying hybrid was built with paged attn,
    // ctx_attn is a llama_kv_cache_paged_context — get_attn() above will
    // return null (it static_casts to the regular type) and callers must
    // use get_attn_paged() instead. Returns null when attn is regular.
    const llama_kv_cache_paged_context * get_attn_paged() const;

private:
    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_attn;
    const llama_memory_context_ptr ctx_recr;

    const llama_memory_status status;
};
