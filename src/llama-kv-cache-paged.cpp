#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {
// Re-declare locally to avoid pulling the cuh header into a non-CUDA
// .cpp. Statically asserted in the .cu side to match the kernel's
// expected sentinel.
constexpr int32_t kInvalidBlockTableEntry = -1;
}

llama_kv_cache_paged::llama_kv_cache_paged(
        const llama_model & model,
        ggml_backend_buffer_type_t buft,
        uint32_t                   n_blocks_total,
        uint32_t                   block_size,
        uint32_t                   n_seq_max,
        uint32_t                   max_blocks_per_seq)
    : model_(model),
      buft_(buft),
      n_blocks_total_(n_blocks_total),
      block_size_(block_size),
      n_seq_max_(n_seq_max),
      max_blocks_per_seq_(max_blocks_per_seq) {

    GGML_ASSERT(block_size_ > 0 && (block_size_ & (block_size_ - 1)) == 0);
    GGML_ASSERT(n_blocks_total_ > 0);
    GGML_ASSERT(n_seq_max_ > 0);
    GGML_ASSERT(max_blocks_per_seq_ > 0);

    // Pool + table init.
    pool_.init(n_blocks_total_, /*n_cpu=*/0, /*watermark=*/0.05f);
    table_.init(n_seq_max_, block_size_);
    seq_states_.assign(n_seq_max_, seq_state{});

    // ── Per-layer K/V storage ──
    //
    // Need per-layer dimensions from the model. For PHASE 3.3, derive
    // them from hparams; later phases will support per-layer variation
    // (hybrid models with mixed attention/recurrent layers).
    const auto & hparams = model.hparams;
    const uint32_t n_layer    = hparams.n_layer;
    const uint32_t head_dim   = hparams.n_embd_head_v(/*il=*/0);
    const uint32_t n_kv_heads = hparams.n_head_kv(/*il=*/0);

    GGML_ASSERT(head_dim > 0 && "head_dim must be > 0");
    GGML_ASSERT(n_kv_heads > 0 && "n_kv_heads must be > 0");

    // Each layer's K and V buffer is a flat 1D byte tensor sized for
    // the vLLM-style block layout. Kernel reinterprets via raw
    // pointer math (we don't need ggml's multi-dim metadata here).
    //
    // Per-block bytes = block_size * n_kv_heads * head_dim * sizeof(__half)
    const size_t bytes_per_block = (size_t) block_size_ * n_kv_heads * head_dim * sizeof(uint16_t);
    const size_t total_layer_bytes = (size_t) n_blocks_total_ * bytes_per_block;

    // ggml_context for the layer storage tensors.
    {
        ggml_init_params iparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * (size_t)(2 * n_layer + 8),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_storage_ = ggml_init(iparams);
        if (!ctx_storage_) {
            throw std::runtime_error("llama_kv_cache_paged: failed to allocate ggml_context for layer storage");
        }
    }

    layers_.resize(n_layer);
    for (uint32_t il = 0; il < n_layer; ++il) {
        // 1D F16 tensor sized for the full block pool's worth of K (or V).
        // Kernel computes offsets within using vLLM layout.
        const int64_t n_elements = (int64_t) total_layer_bytes / (int64_t) sizeof(uint16_t);

        ggml_tensor * k = ggml_new_tensor_1d(ctx_storage_, GGML_TYPE_F16, n_elements);
        ggml_tensor * v = ggml_new_tensor_1d(ctx_storage_, GGML_TYPE_F16, n_elements);
        ggml_set_name(k, ("paged_k_l" + std::to_string(il)).c_str());
        ggml_set_name(v, ("paged_v_l" + std::to_string(il)).c_str());

        layers_[il].k          = k;
        layers_[il].v          = v;
        layers_[il].n_kv_heads = n_kv_heads;
        layers_[il].head_dim   = head_dim;
    }

    // Allocate the backend buffer for layer storage.
    buf_storage_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_storage_, buft_);
    if (!buf_storage_) {
        throw std::runtime_error("llama_kv_cache_paged: failed to allocate backend buffer for layer storage");
    }
    ggml_backend_buffer_clear(buf_storage_, 0);

    LLAMA_LOG_INFO("llama_kv_cache_paged: allocated %u layers × %.1f MiB = %.1f MiB total "
                   "(n_blocks=%u, block_size=%u, n_kv_heads=%u, head_dim=%u)\n",
                   n_layer,
                   (double) total_layer_bytes / (1024.0 * 1024.0),
                   (double)(total_layer_bytes * 2 * n_layer) / (1024.0 * 1024.0),
                   n_blocks_total_, block_size_, n_kv_heads, head_dim);

    // ── Input tensors (block_table, context_lens, q_lens) ──
    //
    // These live on the GPU but are written from the host each batch.
    // We keep host mirrors and ggml_backend_tensor_set them on prepare.
    {
        ggml_init_params iparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 8,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_inputs_ = ggml_init(iparams);
        if (!ctx_inputs_) {
            throw std::runtime_error("llama_kv_cache_paged: failed to allocate ggml_context for inputs");
        }
    }

    block_table_  = ggml_new_tensor_2d(ctx_inputs_, GGML_TYPE_I32, max_blocks_per_seq_, n_seq_max_);
    context_lens_ = ggml_new_tensor_1d(ctx_inputs_, GGML_TYPE_I32, n_seq_max_);
    q_lens_       = ggml_new_tensor_1d(ctx_inputs_, GGML_TYPE_I32, n_seq_max_);
    ggml_set_name(block_table_,  "paged_block_table");
    ggml_set_name(context_lens_, "paged_context_lens");
    ggml_set_name(q_lens_,       "paged_q_lens");

    buf_inputs_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_inputs_, buft_);
    if (!buf_inputs_) {
        throw std::runtime_error("llama_kv_cache_paged: failed to allocate backend buffer for input tensors");
    }
    ggml_backend_buffer_clear(buf_inputs_, 0);

    // Host mirrors.
    h_block_table_.assign((size_t) max_blocks_per_seq_ * n_seq_max_, kInvalidBlockTableEntry);
    h_context_lens_.assign(n_seq_max_, 0);
    h_q_lens_.assign(n_seq_max_, 0);
}

// Workaround for the kInvalidBlockId vs i32 issue — kernel expects
// signed sentinel (-1), pool uses unsigned (~0u). Keep them
// numerically equal in 32-bit twos-complement.
static_assert((int32_t) mt::kInvalidBlockId == kInvalidBlockTableEntry,
              "mt::kInvalidBlockId must reinterpret to -1 in i32 for kernel compat");

llama_kv_cache_paged::~llama_kv_cache_paged() {
    if (buf_storage_) ggml_backend_buffer_free(buf_storage_);
    if (buf_inputs_)  ggml_backend_buffer_free(buf_inputs_);
    if (ctx_storage_) ggml_free(ctx_storage_);
    if (ctx_inputs_)  ggml_free(ctx_inputs_);
}

bool llama_kv_cache_paged::ensure_blocks_for(llama_seq_id seq_id, uint32_t n_new_tokens) {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;
    if (n_new_tokens == 0) return true;

    const uint32_t cur_blocks  = table_.num_blocks(seq_id);
    const uint32_t cur_pos_max = (uint32_t) std::max<llama_pos>(0, seq_states_[seq_id].pos_max + 1);
    const uint32_t need_pos    = cur_pos_max + n_new_tokens;
    const uint32_t need_blocks = (need_pos + block_size_ - 1) / block_size_;

    if (need_blocks <= cur_blocks) return true;
    const uint32_t to_alloc = need_blocks - cur_blocks;

    if (cur_blocks + to_alloc > max_blocks_per_seq_) {
        LLAMA_LOG_WARN("llama_kv_cache_paged::ensure_blocks_for: seq %d needs %u blocks but "
                       "max_blocks_per_seq is %u — request rejected\n",
                       seq_id, cur_blocks + to_alloc, max_blocks_per_seq_);
        return false;
    }

    for (uint32_t i = 0; i < to_alloc; ++i) {
        const uint32_t bid = pool_.alloc_gpu();
        if (bid == mt::kInvalidBlockId) {
            LLAMA_LOG_WARN("llama_kv_cache_paged::ensure_blocks_for: GPU pool exhausted "
                           "(seq %d, need %u more, free %zu)\n",
                           seq_id, to_alloc - i, pool_.n_free_gpu());
            return false;
        }
        table_.append_block(seq_id, bid);
    }
    return true;
}

bool llama_kv_cache_paged::compute_slot_mapping(const llama_ubatch * ubatch, int32_t * out) const {
    // For v1 single-seq (n_seq_max == 1), every token belongs to seq 0.
    // The seq's logical block list is in table_; for token i at position
    // pos = ubatch->pos[i]:
    //   logical_block_idx = pos / block_size_
    //   physical_block    = table_.get_physical(0, logical_block_idx)
    //   slot_in_block     = pos % block_size_
    //   slot_mapping[i]   = physical_block * block_size_ + slot_in_block
    //
    // Multi-seq (n_seq_max > 1) is a follow-up: walk ubatch->seq_id[i][0]
    // per token instead of pinning seq 0.
    GGML_ASSERT(n_seq_max_ == 1 && "compute_slot_mapping: multi-seq not yet implemented");

    const uint32_t n_blk = table_.num_blocks(/*seq_id=*/0);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos[i];
        if (pos < 0) {
            out[i] = -1;  // padding token
            continue;
        }
        const uint32_t logical = (uint32_t) pos / block_size_;
        const uint32_t slot    = (uint32_t) pos % block_size_;
        if (logical >= n_blk) {
            // ensure_blocks_for should have allocated enough — bug.
            return false;
        }
        const uint32_t physical = table_.get_physical(/*seq_id=*/0, logical);
        if (physical == mt::kInvalidBlockId) {
            return false;
        }
        out[i] = (int32_t)(physical * block_size_ + slot);
    }
    return true;
}

void llama_kv_cache_paged::prepare_batch_tensors() {
    // Write per-seq block table + context lens + q lens into host
    // mirrors, then copy to backend.
    std::fill(h_block_table_.begin(), h_block_table_.end(), kInvalidBlockTableEntry);
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        const uint32_t n_blk = table_.num_blocks((llama_seq_id) s);
        for (uint32_t b = 0; b < n_blk; ++b) {
            // Layout is [max_blocks_per_seq, n_seq_max] in ggml-style
            // ne[0]=max_bps fastest-changing, ne[1]=n_seq_max.
            h_block_table_[(size_t) s * max_blocks_per_seq_ + b] =
                (int32_t) table_.get_physical((llama_seq_id) s, b);
        }
        h_context_lens_[s] = (int32_t) std::max<llama_pos>(0, seq_states_[s].pos_max + 1);
        // q_lens is filled by init_batch when it knows the ubatch shape.
    }

    ggml_backend_tensor_set(block_table_,  h_block_table_.data(),  0,
                            sizeof(int32_t) * h_block_table_.size());
    ggml_backend_tensor_set(context_lens_, h_context_lens_.data(), 0,
                            sizeof(int32_t) * h_context_lens_.size());
    ggml_backend_tensor_set(q_lens_,       h_q_lens_.data(),       0,
                            sizeof(int32_t) * h_q_lens_.size());
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch_with_ubatches(
        std::vector<llama_ubatch> ubatches) {
    if (ubatches.empty()) {
        return std::make_unique<llama_kv_cache_paged_context>(
            this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
    }
    for (const auto & ub : ubatches) {
        if (!ensure_blocks_for(/*seq_id=*/0, ub.n_tokens)) {
            return std::make_unique<llama_kv_cache_paged_context>(
                this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        h_q_lens_[0] = (int32_t) ub.n_tokens;
        seq_states_[0].pos_max += (llama_pos) ub.n_tokens;
        if (seq_states_[0].pos_min < 0) seq_states_[0].pos_min = 0;
    }
    prepare_batch_tensors();
    return std::make_unique<llama_kv_cache_paged_context>(
        this, std::move(ubatches), LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool /*embd_all*/) {
    // Split the batch into ubatches (1 per slot for v1, since
    // n_seq_max=1 and we don't yet do parallel batching).
    std::vector<llama_ubatch> ubatches;

    // Pull ubatches from the balloc until exhausted.
    while (true) {
        llama_ubatch ub = balloc.split_simple(n_ubatch);
        if (ub.n_tokens == 0) break;
        ubatches.push_back(std::move(ub));
    }

    if (ubatches.empty()) {
        return std::make_unique<llama_kv_cache_paged_context>(
            this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
    }

    // Allocate blocks for each ubatch's tokens.
    for (const auto & ub : ubatches) {
        // For v1 single-seq: all tokens belong to seq 0.
        // (Real multi-seq case requires walking ub.seq_id_unq[] etc.
        // and bucketing — punted to multi-seq phase.)
        if (!ensure_blocks_for(/*seq_id=*/0, ub.n_tokens)) {
            return std::make_unique<llama_kv_cache_paged_context>(
                this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        // Update q_lens host mirror for this batch's contribution.
        h_q_lens_[0] = (int32_t) ub.n_tokens;
        // Bump pos_max optimistically — the graph will write into the
        // newly allocated blocks at positions [pos_max+1, pos_max+n_tokens].
        seq_states_[0].pos_max += (llama_pos) ub.n_tokens;
        if (seq_states_[0].pos_min < 0) seq_states_[0].pos_min = 0;
    }

    prepare_batch_tensors();

    return std::make_unique<llama_kv_cache_paged_context>(
        this, std::move(ubatches), LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    // Worst-case batch — used during compute buffer sizing. Return
    // an empty context with status SUCCESS; the graph will be built
    // for the maximum batch size separately.
    return std::make_unique<llama_kv_cache_paged_context>(
        this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    return std::make_unique<llama_kv_cache_paged_context>(
        this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
}

void llama_kv_cache_paged::clear(bool /*data*/) {
    // Free all blocks back to the pool, reset table + seq states.
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        std::vector<uint32_t> freed = table_.clear_seq((llama_seq_id) s);
        for (uint32_t bid : freed) {
            if (bid != mt::kInvalidBlockId) pool_.free_block(bid);
        }
        seq_states_[s] = seq_state{};
    }
    pool_.reset();
    table_.reset();
    std::fill(h_block_table_.begin(), h_block_table_.end(), kInvalidBlockTableEntry);
    std::fill(h_context_lens_.begin(), h_context_lens_.end(), 0);
    std::fill(h_q_lens_.begin(), h_q_lens_.end(), 0);
    LLAMA_LOG_DEBUG("llama_kv_cache_paged::clear: all blocks released, all seqs reset\n");
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;

    // Whole-seq wipe: free all blocks, reset state.
    if (p0 < 0 || p1 < 0) {
        std::vector<uint32_t> freed = table_.clear_seq(seq_id);
        for (uint32_t bid : freed) {
            if (bid != mt::kInvalidBlockId) pool_.free_block(bid);
        }
        seq_states_[seq_id] = seq_state{};
        LLAMA_LOG_DEBUG("llama_kv_cache_paged::seq_rm: whole-seq wipe seq=%d, freed %zu blocks\n",
                        seq_id, freed.size());
        return true;
    }

    // Partial wipe: tail truncation only for v1 (carving out middle
    // ranges requires more bookkeeping; rare in practice). If p1 ==
    // current pos_max+1, treat as tail truncation: free blocks
    // entirely past the truncation point.
    const llama_pos cur_max = seq_states_[seq_id].pos_max;
    if (p0 > cur_max) return true;  // nothing to do

    // Compute new pos_max after truncation.
    const llama_pos new_max = p0 - 1;

    // Drop blocks whose entire range is past p0.
    const uint32_t keep_blocks = (uint32_t)(new_max + 1 + (llama_pos) block_size_ - 1) / block_size_;
    if (keep_blocks > table_.num_blocks(seq_id)) return true;  // nothing to drop

    // Walk blocks past keep_blocks and free.
    while (table_.num_blocks(seq_id) > keep_blocks) {
        std::vector<uint32_t> tmp = table_.clear_seq(seq_id);  // CAUTION: clears all
        // Re-append the kept ones.
        for (uint32_t i = 0; i < keep_blocks && i < tmp.size(); ++i) {
            table_.append_block(seq_id, tmp[i]);
        }
        for (uint32_t i = keep_blocks; i < tmp.size(); ++i) {
            if (tmp[i] != mt::kInvalidBlockId) pool_.free_block(tmp[i]);
        }
        break;
    }

    seq_states_[seq_id].pos_max = new_max;
    return true;
}

void llama_kv_cache_paged::seq_cp(llama_seq_id /*src*/, llama_seq_id /*dst*/,
                                    llama_pos /*p0*/, llama_pos /*p1*/) {
    // CoW between sequences — not supported in v1.
    LLAMA_LOG_WARN("llama_kv_cache_paged::seq_cp: not implemented in v1 — no-op\n");
}

void llama_kv_cache_paged::seq_keep(llama_seq_id /*seq_id*/) {
    // No-op for v1 (no shared blocks to release).
}

void llama_kv_cache_paged::seq_add(llama_seq_id /*seq_id*/, llama_pos /*p0*/,
                                     llama_pos /*p1*/, llama_pos /*shift*/) {
    // Position shifts (for ctx-shift) — not supported (get_can_shift = false).
    LLAMA_LOG_WARN("llama_kv_cache_paged::seq_add: ctx shift not supported on paged cache\n");
}

void llama_kv_cache_paged::seq_div(llama_seq_id /*seq_id*/, llama_pos /*p0*/,
                                     llama_pos /*p1*/, int /*d*/) {
    // Same as seq_add — position-mod ops not supported.
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return -1;
    return seq_states_[seq_id].pos_min;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return -1;
    return seq_states_[seq_id].pos_max;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> out;
    if (buf_storage_) out[buft_] += ggml_backend_buffer_get_size(buf_storage_);
    if (buf_inputs_)  out[buft_] += ggml_backend_buffer_get_size(buf_inputs_);
    return out;
}

void llama_kv_cache_paged::state_write(llama_io_write_i & /*io*/, llama_seq_id /*seq_id*/,
                                          llama_state_seq_flags /*flags*/) const {
    // State persistence — punted to a later phase.
    LLAMA_LOG_WARN("llama_kv_cache_paged::state_write: not implemented — state will not persist\n");
}

void llama_kv_cache_paged::state_read(llama_io_read_i & /*io*/, llama_seq_id /*seq_id*/,
                                         llama_state_seq_flags /*flags*/) {
    LLAMA_LOG_WARN("llama_kv_cache_paged::state_read: not implemented — state will not load\n");
}

// ─── llama_kv_cache_paged_context ───

llama_kv_cache_paged_context::llama_kv_cache_paged_context(
        llama_kv_cache_paged *           parent,
        std::vector<llama_ubatch>        ubatches,
        llama_memory_status              status)
    : parent_(parent),
      ubatches_(std::move(ubatches)),
      status_(status) {}

bool llama_kv_cache_paged_context::next() {
    if (++i_ubatch_ >= ubatches_.size()) return false;
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // For paged cache the per-batch input tensors are populated in
    // init_batch (host mirrors copied to GPU before the graph runs).
    // No additional per-step apply needed in v1.
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    GGML_ASSERT(i_ubatch_ < ubatches_.size());
    return ubatches_[i_ubatch_];
}
