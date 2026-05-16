#include "llama-memory-hybrid.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

//
// llama_memory_hybrid
//

// Helper: pick the right attn buffer type for the paged path. The paged
// cache currently allocates on the model's first device (matches Phase 3.4a
// non-hybrid sizing). For hybrid models we mirror this — the recurrent
// memory continues to use the regular per-layer placement.
static ggml_backend_buffer_type_t paged_attn_buft(const llama_model & model) {
    GGML_ASSERT(!model.devices.empty() && "paged attn: model has no devices");
    return ggml_backend_dev_buffer_type(model.devices[0].dev);
}

llama_memory_hybrid::llama_memory_hybrid(
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
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* paged */
                 uint32_t   paged_n_blocks,
                 uint32_t   paged_block_size,
                 uint32_t   paged_max_blocks_per_seq,
                 uint32_t   paged_n_warm_blocks,
                 uint32_t   paged_n_cold_blocks,
                 std::string paged_ssd_path,
                 bool       paged_cold_resume,
                 std::string paged_instance_id,
                 uint32_t   paged_cold_budget_mb) :
    hparams(model.hparams),
    mem_attn(paged_n_blocks > 0 ? nullptr : new llama_kv_cache(
        model,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recurrent(il); }
            : filter_attn,
        nullptr
    )),
    mem_attn_paged(paged_n_blocks > 0 ? new llama_kv_cache_paged(
        model,
        paged_attn_buft(model),
        paged_n_blocks,
        paged_block_size,
        n_seq_max,
        paged_max_blocks_per_seq,
        paged_n_warm_blocks,
        paged_n_cold_blocks,
        paged_ssd_path,
        paged_cold_resume,
        paged_instance_id,
        paged_cold_budget_mb,
        // Filter out recurrent layers — paged cache only carries attention
        // K/V; recurrent state lives in mem_recr. Without this the cache
        // pre-allocates K/V for ALL layers (including recurrent ones that
        // never use it), wasting up to ~7×n_layer worth of VRAM.
        filter_attn ? filter_attn : llama_kv_cache_paged::layer_filter_cb(
            [&](int32_t il) { return !hparams.is_recurrent(il); }),
        type_k,
        type_v
    ) : nullptr),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        rs_size,
        n_seq_max,
        n_rs_seq,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recurrent(il); }
            : filter_recr
    )) {
    GGML_UNUSED(v_trans);
    GGML_UNUSED(kv_size); GGML_UNUSED(n_pad); GGML_UNUSED(n_swa);
    GGML_UNUSED(swa_type); GGML_UNUSED(unified);
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    // Phase 3.6: paged attn path. The recurrent half requires equal_seqs
    // ubatches (asserts in llama-memory-recurrent.cpp), so we do the
    // split here using the same balloc.split_equal pattern as the
    // regular hybrid path, then feed the pre-built ubatches to both the
    // paged cache (block alloc + tensor uploads) and the recurrent
    // prepare. The paged cache exposes init_batch_with_ubatches for
    // exactly this composition.
    if (is_paged_attn()) {
        balloc.split_reset();
        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ub = embd_all
                ? balloc.split_seq(n_ubatch)
                : balloc.split_equal(n_ubatch, /*sequential=*/true);
            if (ub.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ub));
        }
        if (balloc.get_n_used() < balloc.get_n_tokens() || ubatches.empty()) {
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        if (!mem_recr->prepare(ubatches)) {
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches (paged hybrid)\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        // ubatches passed by-value to the paged init helper; copy first
        // so we still own the vector for the hybrid context.
        std::vector<llama_ubatch> ubatches_copy = ubatches;
        auto paged_ctx_ptr = mem_attn_paged->init_batch_with_ubatches(std::move(ubatches_copy));
        if (paged_ctx_ptr->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            return std::make_unique<llama_memory_hybrid_context>(paged_ctx_ptr->get_status());
        }
        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(paged_ctx_ptr), std::move(ubatches));
    }

    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);
                ubatch = balloc.split_equal(n_ubatch, !unified);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

// Phase 3.6: helper — returns the polymorphic-base pointer for whichever
// attn cache is non-null. Lets the simple delegating methods below stay
// branch-free for ops that exist on the llama_memory_i interface.
llama_memory_i * llama_memory_hybrid::attn_base() const {
    return mem_attn_paged ? (llama_memory_i *) mem_attn_paged.get()
                          : (llama_memory_i *) mem_attn.get();
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent. Paged returns false
    // (its get_can_shift is hard-coded false in v1).
    return attn_base()->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    attn_base()->clear(data);
    mem_recr->clear(data);
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }
    return attn_base()->seq_rm(seq_id, p0, p1);
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    attn_base()->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    attn_base()->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    attn_base()->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    attn_base()->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(attn_base()->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(attn_base()->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = attn_base()->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    return mb;
}

int llama_memory_hybrid::mt_restore_tag_slot(llama_seq_id seq_id, llama_pos position) {
    // Paged cache's restore semantics are TBD — return -1 ("nothing
    // restored") for now so the tier system treats it as a miss.
    if (mem_attn_paged) return -1;
    return mem_attn ? mem_attn->mt_restore_tag_slot(seq_id, position) : -1;
}

int llama_memory_hybrid::mt_restore_recurrent_slot(llama_seq_id seq_id) {
    return mem_recr ? mem_recr->mt_restore_recurrent_slot(seq_id) : -1;
}

mt::InnerView llama_memory_hybrid::make_tier_view() const {
    // Paged cache exposes its own InnerView through llama_memory_i too,
    // but for the tier system the paged path is its own world — return
    // an empty attn view for now and let the recurrent half flow through.
    mt::InnerView view = mem_attn ? mem_attn->make_tier_view() : mt::InnerView{};
    if (mem_recr) {
        auto recr_view = mem_recr->make_tier_view();
        view.recur_seqs.insert(view.recur_seqs.end(),
                               std::make_move_iterator(recr_view.recur_seqs.begin()),
                               std::make_move_iterator(recr_view.recur_seqs.end()));
    }
    return view;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        attn_base()->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        attn_base()->state_read(io, seq_id, flags);
    }
    mem_recr->state_read(io, seq_id, flags);
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->is_paged_attn()
             ? mem->get_mem_attn_paged()->init_full()
             : mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->is_paged_attn()
             ? mem->get_mem_attn_paged()->init_update(lctx, optimize)
             : mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
        llama_memory_context_ptr    ctx_attn_paged,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    ctx_attn(std::move(ctx_attn_paged)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_forward() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_forward() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_rot_inverse() const {
    return ctx_attn ? ctx_attn->get_turbo_rot_inverse() : nullptr;
}

ggml_tensor * llama_memory_hybrid_context::get_turbo_innerq_scale_inv() const {
    return ctx_attn ? ctx_attn->get_turbo_innerq_scale_inv() : nullptr;
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}

const llama_kv_cache_paged_context * llama_memory_hybrid_context::get_attn_paged() const {
    return dynamic_cast<const llama_kv_cache_paged_context *>(ctx_attn.get());
}
