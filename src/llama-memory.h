#pragma once

#include "llama.h"
#include "memory-tier/mt-inner-access.h"  // mt::InnerView (returned by value from make_tier_view)

#include <map>
#include <memory>
#include <functional>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;

    // tiered KV cache (Phase 2 rewrite). When kv_tier_enabled is false the
    // rest of these fields are ignored and the inner cache is returned as-is.
    bool         kv_tier_enabled;
    float        kv_tier_hot_pct;
    float        kv_tier_warm_pct;
    float        kv_tier_cold_pct;
    const char * kv_tier_ssd_path;
    int32_t      kv_tier_eviction_policy;
    int32_t      kv_tier_compression;
    float        kv_tier_attention_threshold;
    int32_t      kv_tier_warm_device;
    int32_t      kv_tier_total_ctx;
    const char * kv_tier_semantic_index;
    float        kv_tier_semantic_threshold;
    int32_t      kv_tier_semantic_topk;
    bool         kv_tier_paged_blocks;       // Phase 2a opt-in
    int32_t      kv_tier_paged_block_size;   // tokens/block (0 => default 16)
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;

    // TurboQuant: get rotation tensors for pre-rotate-queries optimization
    // Returns null for non-turbo memory types. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_rot_forward() const { return nullptr; }
    virtual ggml_tensor * get_turbo_rot_inverse() const { return nullptr; }

    // TurboQuant InnerQ: get per-channel scale_inv tensor for Q/V equalization
    // Returns nullptr when InnerQ is not active. Override in KV cache contexts.
    virtual ggml_tensor * get_turbo_innerq_scale_inv() const { return nullptr; }
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    //
    // Inner-access protocol for the tiered KV wrapper.
    //
    // Default returns an empty view, meaning "this backend does not
    // participate in tiering". Concrete backends (llama_kv_cache,
    // llama_kv_cache_iswa, llama_memory_recurrent,
    // llama_memory_hybrid_iswa) override to expose per-layer K/V tensors
    // and per-sequence recurrent state. The returned view is a non-owning
    // snapshot; the pointers must remain stable for the lifetime of the
    // backend (or until the next structural change).
    //
    // Defined in src/memory-tier/mt-inner-access.h. mt::InnerView is a
    // value type (vectors of POD-ish structs) — returning by value lets
    // this header avoid including the mt header.
    //
    virtual mt::InnerView make_tier_view() const;

    // Find a free slot in the cache and tag it with (position, seq_id).
    // After this returns, the slot is reserved (cells consider it filled)
    // but the K/V tensor data at that slot is whatever was there before;
    // the caller (the tier wrapper) immediately copies the backed-up
    // K/V via AttentionMover, completing the restoration.
    //
    // Returns the slot index on success, or -1 if no free slot is
    // available. Default: -1 (backend doesn't support tier restoration).
    //
    // For composite caches (iswa, hybrid), the slot is in the
    // restorable sub-cache (base attention only). The wrapper uses
    // make_tier_view's attn_caches list (skipping is_swa entries) to
    // know which layers to fill; the non-SWA layers all live in the
    // physical cache that owns this slot.
    virtual int mt_restore_tag_slot(llama_seq_id seq_id, llama_pos position);

    // Recurrent counterpart: allocate a fresh recurrent cell for seq_id
    // and wire it as the seq's tail, so the next make_tier_view will
    // surface a RecurrentStateView for this seq pointing at the
    // newly-allocated slot. The wrapper then copies the backed-up
    // r/s state into that slot via RecurrentStateMover.
    //
    // Returns the slot (cell index) on success, or -1 on failure.
    // Default: -1 (backend has no recurrent state). Composite caches
    // (hybrid, hybrid_iswa) forward to their recurrent sub-cache.
    virtual int mt_restore_recurrent_slot(llama_seq_id seq_id);
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
