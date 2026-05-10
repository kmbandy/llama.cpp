#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

// MAD-121: cold slot-id helpers — file-scope so fault_in_warm_blocks_for_batch
// can read the cold_slot_for_ vector. kInvalidColdIdx (~0u) means
// "not cold-resident".
namespace {
constexpr uint32_t kInvalidColdIdx = ~0u;
inline void mark_cold(std::vector<std::vector<uint32_t>> & slot_for,
                       llama_seq_id seq, uint32_t lblock, uint32_t cold_idx) {
    if ((size_t) seq >= slot_for.size()) slot_for.resize(seq + 1);
    auto & row = slot_for[seq];
    if (lblock >= row.size()) row.resize(lblock + 1, kInvalidColdIdx);
    row[lblock] = cold_idx;
}
inline uint32_t cold_slot(const std::vector<std::vector<uint32_t>> & slot_for,
                           llama_seq_id seq, uint32_t lblock) {
    if ((size_t) seq >= slot_for.size()) return kInvalidColdIdx;
    const auto & row = slot_for[seq];
    if (lblock >= row.size()) return kInvalidColdIdx;
    return row[lblock];
}
inline bool is_cold(const std::vector<std::vector<uint32_t>> & slot_for,
                     llama_seq_id seq, uint32_t lblock) {
    return cold_slot(slot_for, seq, lblock) != kInvalidColdIdx;
}
}  // namespace

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
        uint32_t                   max_blocks_per_seq,
        uint32_t                   n_warm_blocks,
        uint32_t                   n_cold_blocks,
        std::string                ssd_path,
        layer_filter_cb            filter,
        ggml_type                  type_k,
        ggml_type                  type_v)
    : model_(model),
      buft_(buft),
      n_blocks_total_(n_blocks_total),
      n_warm_blocks_(n_warm_blocks),
      block_size_(block_size),
      n_seq_max_(n_seq_max),
      max_blocks_per_seq_(max_blocks_per_seq),
      type_k_(type_k),
      type_v_(type_v),
      n_cold_blocks_(n_cold_blocks),
      cold_path_(std::move(ssd_path)) {

    GGML_ASSERT(block_size_ > 0 && (block_size_ & (block_size_ - 1)) == 0);
    GGML_ASSERT(n_blocks_total_ > 0);
    GGML_ASSERT(n_seq_max_ > 0);
    GGML_ASSERT(max_blocks_per_seq_ > 0);

    // Pool + table init. CPU pool is the warm tier (MAD-120). 0 means
    // warm-tier disabled (legacy single-pool behavior).
    pool_.init(n_blocks_total_, n_warm_blocks_, /*watermark=*/0.05f);
    table_.init(n_seq_max_, block_size_);
    seq_states_.assign(n_seq_max_, seq_state{});

    // ── Per-layer K/V storage ──
    //
    // For hybrid models a `filter` is supplied that returns true only for
    // attention layers — recurrent layers' state lives in mem_recr. We
    // still keep `layers_` indexed by model layer index so
    // `cache.layer(il)` works directly from the graph; non-attn entries
    // stay default-constructed (k/v == nullptr) and never queried.
    const auto & hparams = model.hparams;
    const uint32_t n_layer    = hparams.n_layer;
    const uint32_t head_dim   = hparams.n_embd_head_v(/*il=*/0);
    const uint32_t n_kv_heads = hparams.n_head_kv(/*il=*/0);

    GGML_ASSERT(head_dim > 0 && "head_dim must be > 0");
    GGML_ASSERT(n_kv_heads > 0 && "n_kv_heads must be > 0");

    // Per-element byte cost: f16 = 2 bytes, q8_0 = 17 bytes per 32-element
    // block ≈ 0.53 byte/elt, etc. ggml_type_size + ggml_blck_size give the
    // exact ratio for any quant. n_elements is total values per layer; we
    // round n_blocks_total slot count up to a multiple of the quant block
    // size so partial-block edge cases don't underrun the buffer.
    auto round_to_block = [](size_t n, size_t blk) {
        return blk > 1 ? ((n + blk - 1) / blk) * blk : n;
    };
    const size_t k_blk = (size_t) ggml_blck_size(type_k_);
    const size_t v_blk = (size_t) ggml_blck_size(type_v_);
    const size_t per_token_per_layer_elts = (size_t) n_kv_heads * head_dim;
    const size_t k_n_elts_per_layer = round_to_block(
        (size_t) n_blocks_total_ * block_size_ * per_token_per_layer_elts, k_blk);
    const size_t v_n_elts_per_layer = round_to_block(
        (size_t) n_blocks_total_ * block_size_ * per_token_per_layer_elts, v_blk);
    const size_t k_bytes_per_layer = ggml_row_size(type_k_, (int64_t) k_n_elts_per_layer);
    const size_t v_bytes_per_layer = ggml_row_size(type_v_, (int64_t) v_n_elts_per_layer);

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
    uint32_t n_attn_layers = 0;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (filter && !filter((int32_t) il)) {
            // Recurrent (or otherwise filtered-out) layer — leave entry
            // empty. The graph builder skips these and routes their state
            // through the recurrent half of the hybrid memory.
            layers_[il].k = nullptr;
            layers_[il].v = nullptr;
            continue;
        }
        ggml_tensor * k = ggml_new_tensor_1d(ctx_storage_, type_k_, (int64_t) k_n_elts_per_layer);
        ggml_tensor * v = ggml_new_tensor_1d(ctx_storage_, type_v_, (int64_t) v_n_elts_per_layer);
        ggml_set_name(k, ("paged_k_l" + std::to_string(il)).c_str());
        ggml_set_name(v, ("paged_v_l" + std::to_string(il)).c_str());

        layers_[il].k          = k;
        layers_[il].v          = v;
        layers_[il].n_kv_heads = n_kv_heads;
        layers_[il].head_dim   = head_dim;
        ++n_attn_layers;
    }

    // Allocate the backend buffer for layer storage.
    buf_storage_ = ggml_backend_alloc_ctx_tensors_from_buft(ctx_storage_, buft_);
    if (!buf_storage_) {
        throw std::runtime_error("llama_kv_cache_paged: failed to allocate backend buffer for layer storage");
    }
    ggml_backend_buffer_clear(buf_storage_, 0);

    const double per_layer_mib = (double)(k_bytes_per_layer + v_bytes_per_layer) / (1024.0 * 1024.0);
    LLAMA_LOG_INFO("llama_kv_cache_paged: allocated %u/%u attn layers × %.1f MiB (K+V) = %.1f MiB total "
                   "(n_blocks=%u, block_size=%u, n_kv_heads=%u, head_dim=%u, type_k=%s, type_v=%s)\n",
                   n_attn_layers, n_layer, per_layer_mib,
                   per_layer_mib * (double) n_attn_layers,
                   n_blocks_total_, block_size_, n_kv_heads, head_dim,
                   ggml_type_name(type_k_), ggml_type_name(type_v_));

    // ── MAD-120: per-layer host-side warm storage ──
    //
    // Per-block sizes derived from the per-layer totals: each block is
    // 1/n_blocks_total of the layer's K (or V) buffer. We use these as
    // the granularity for hipMemcpy on evict/restore. Stored in the
    // class so the methods can compute offsets without re-deriving.
    k_bytes_per_block_ = k_bytes_per_layer / n_blocks_total_;
    v_bytes_per_block_ = v_bytes_per_layer / n_blocks_total_;

    if (n_warm_blocks_ > 0) {
        warm_k_.resize(n_layer);
        warm_v_.resize(n_layer);
        size_t warm_bytes_total = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!layers_[il].k) continue;  // filtered-out layer (recurrent)
            warm_k_[il].assign((size_t) n_warm_blocks_ * k_bytes_per_block_, 0);
            warm_v_[il].assign((size_t) n_warm_blocks_ * v_bytes_per_block_, 0);
            warm_bytes_total += warm_k_[il].size() + warm_v_[il].size();
        }
        LLAMA_LOG_INFO("llama_kv_cache_paged: warm tier enabled — %u host blocks × "
                       "%.1f KiB/block per layer × %u attn layers = %.1f MiB host warm storage\n",
                       n_warm_blocks_,
                       (double)(k_bytes_per_block_ + v_bytes_per_block_) / 1024.0,
                       n_attn_layers,
                       (double) warm_bytes_total / (1024.0 * 1024.0));
    }

    // MAD-121: cold tier (SSD) — fixed-slot per-layer files. Each
    // attention layer gets two files (K and V), sized to
    // n_cold_blocks × bytes_per_block. Cold slots are O(1)-addressed
    // by `cold_idx * bytes_per_block`. Storage is bounded — no churn
    // growth. For turbo4 / Q8_0 caches the bytes are already
    // quantized at the cache layer; F16 cache stores raw fp16 bytes
    // (compression for that case is a follow-up).
    if (n_cold_blocks_ > 0) {
        if (cold_path_.empty()) {
            LLAMA_LOG_WARN("llama_kv_cache_paged: cold tier requested with empty ssd_path; disabling cold\n");
            n_cold_blocks_ = 0;
        } else {
            const std::string subdir = cold_path_ + "/paged";
            (void) ::mkdir(cold_path_.c_str(), 0700);
            (void) ::mkdir(subdir.c_str(),    0700);

            cold_fd_k_.assign(n_layer, -1);
            cold_fd_v_.assign(n_layer, -1);
            const off_t k_file_bytes = (off_t) n_cold_blocks_ * (off_t) k_bytes_per_block_;
            const off_t v_file_bytes = (off_t) n_cold_blocks_ * (off_t) v_bytes_per_block_;
            uint64_t total_bytes = 0;
            bool ok = true;
            for (uint32_t il = 0; il < n_layer && ok; ++il) {
                if (!layers_[il].k) continue;
                const std::string kpath = subdir + "/L" + std::to_string(il) + ".k.bin";
                const std::string vpath = subdir + "/L" + std::to_string(il) + ".v.bin";
                const int fk = ::open(kpath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
                const int fv = ::open(vpath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
                if (fk < 0 || fv < 0 ||
                    ::ftruncate(fk, k_file_bytes) < 0 ||
                    ::ftruncate(fv, v_file_bytes) < 0) {
                    LLAMA_LOG_WARN("llama_kv_cache_paged: cold tier file open/truncate failed at %s; disabling\n",
                                   kpath.c_str());
                    if (fk >= 0) ::close(fk);
                    if (fv >= 0) ::close(fv);
                    ok = false;
                    break;
                }
                cold_fd_k_[il] = fk;
                cold_fd_v_[il] = fv;
                total_bytes += (uint64_t)(k_file_bytes + v_file_bytes);
            }

            if (!ok) {
                for (int fd : cold_fd_k_) if (fd >= 0) ::close(fd);
                for (int fd : cold_fd_v_) if (fd >= 0) ::close(fd);
                cold_fd_k_.clear();
                cold_fd_v_.clear();
                n_cold_blocks_ = 0;
            } else {
                // Free-stack of cold slot ids.
                cold_pool_free_.reserve(n_cold_blocks_);
                for (uint32_t i = n_cold_blocks_; i > 0; --i) {
                    cold_pool_free_.push_back(i - 1);
                }
                cold_slot_for_.assign(n_seq_max_, std::vector<uint32_t>());
                LLAMA_LOG_INFO("llama_kv_cache_paged: cold tier enabled — %u blocks × "
                               "%.1f KiB/block (K+V) × %u attn layers = %.1f MiB on %s\n",
                               n_cold_blocks_,
                               (double)(k_bytes_per_block_ + v_bytes_per_block_) / 1024.0,
                               n_attn_layers,
                               (double) total_bytes / (1024.0 * 1024.0),
                               subdir.c_str());
            }
        }
    }

    // ── Input tensor (block_table) ──
    //
    // Lives on the GPU but written from the host each batch. We keep a
    // host mirror and ggml_backend_tensor_set it on prepare. context_lens
    // and q_lens used to live here too but were moved per-graph (MAD-114).
    {
        ggml_init_params iparams = {
            /*.mem_size   =*/ ggml_tensor_overhead() * 4,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_inputs_ = ggml_init(iparams);
        if (!ctx_inputs_) {
            throw std::runtime_error("llama_kv_cache_paged: failed to allocate ggml_context for inputs");
        }
    }

    block_table_  = ggml_new_tensor_2d(ctx_inputs_, GGML_TYPE_I32, max_blocks_per_seq_, n_seq_max_);
    ggml_set_name(block_table_,  "paged_block_table");

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
    // MAD-121: close cold-tier fds.
    for (int fd : cold_fd_k_) if (fd >= 0) ::close(fd);
    for (int fd : cold_fd_v_) if (fd >= 0) ::close(fd);
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
        uint32_t bid = pool_.alloc_gpu();

        // MAD-120: GPU pool full → try evicting LRU to warm and retry.
        // Spin until either alloc succeeds or no eligible victim remains.
        while (bid == mt::kInvalidBlockId && warm_enabled()) {
            if (!evict_lru_to_warm()) {
                break;  // no eligible GPU block to evict (all in hot tail)
            }
            bid = pool_.alloc_gpu();
        }

        if (bid == mt::kInvalidBlockId) {
            LLAMA_LOG_WARN("llama_kv_cache_paged::ensure_blocks_for: GPU pool exhausted "
                           "(seq %d, need %u more, free gpu=%zu cpu=%zu, warm_enabled=%d)\n",
                           seq_id, to_alloc - i,
                           pool_.n_free_gpu(), pool_.n_free_cpu(),
                           (int) warm_enabled());
            return false;
        }
        table_.append_block(seq_id, bid);
    }
    return true;
}

// ─── MAD-120: hot↔warm tier methods ───
//
// Block-granular host RAM eviction. Each evict copies one paged block's
// worth of K and V data via ggml_backend_tensor_get to host buffers
// owned by this cache. Restore is symmetric. The block_table_ is the
// authoritative mapping from (seq, lblock) → physical block id; physical
// IDs in [0, n_blocks_total_) are GPU-resident, IDs >= n_blocks_total_
// are CPU/warm-resident (offset in warm_k_/warm_v_ is physical -
// n_blocks_total_).

bool llama_kv_cache_paged::evict_block_to_warm(llama_seq_id seq_id, uint32_t logical_block) {
    if (!warm_enabled()) return false;
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;
    if (logical_block >= table_.num_blocks(seq_id)) return false;

    const uint32_t gpu_physical = table_.get_physical(seq_id, logical_block);
    if (gpu_physical == mt::kInvalidBlockId) return false;
    if (!pool_.is_gpu(gpu_physical)) return false;  // already in warm

    uint32_t cpu_physical = pool_.alloc_cpu();
    if (cpu_physical == mt::kInvalidBlockId) {
        // MAD-121: warm pool full → escalate to cold. Pick any
        // warm-resident block, spill it to SSD, then retry.
        if (cold_enabled() && evict_lru_warm_to_cold()) {
            cpu_physical = pool_.alloc_cpu();
        }
        if (cpu_physical == mt::kInvalidBlockId) {
            return false;
        }
    }

    const uint32_t cpu_idx = cpu_physical - n_blocks_total_;
    const size_t gpu_k_off  = (size_t) gpu_physical * k_bytes_per_block_;
    const size_t gpu_v_off  = (size_t) gpu_physical * v_bytes_per_block_;
    const size_t warm_k_off = (size_t) cpu_idx     * k_bytes_per_block_;
    const size_t warm_v_off = (size_t) cpu_idx     * v_bytes_per_block_;

    // Move data for every attn layer.
    for (uint32_t il = 0; il < layers_.size(); ++il) {
        const auto & layer = layers_[il];
        if (!layer.k) continue;
        ggml_backend_tensor_get(layer.k,
                                warm_k_[il].data() + warm_k_off,
                                gpu_k_off, k_bytes_per_block_);
        ggml_backend_tensor_get(layer.v,
                                warm_v_[il].data() + warm_v_off,
                                gpu_v_off, v_bytes_per_block_);
    }

    // Atomic flip: table now maps (seq, lblock) to the CPU physical;
    // free the GPU physical back to the pool.
    table_.swap_block(seq_id, logical_block, cpu_physical);
    pool_.free_block(gpu_physical);
    return true;
}

bool llama_kv_cache_paged::restore_block_from_warm(llama_seq_id seq_id, uint32_t logical_block) {
    if (!warm_enabled()) return false;
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;
    if (logical_block >= table_.num_blocks(seq_id)) return false;

    const uint32_t cpu_physical = table_.get_physical(seq_id, logical_block);
    if (cpu_physical == mt::kInvalidBlockId) return false;
    if (pool_.is_gpu(cpu_physical)) return true;  // already on GPU, no-op

    const uint32_t gpu_physical = pool_.alloc_gpu();
    if (gpu_physical == mt::kInvalidBlockId) {
        // GPU full — caller (fault-in pre-pass) is responsible for
        // making room before invoking restore.
        return false;
    }

    const uint32_t cpu_idx = cpu_physical - n_blocks_total_;
    const size_t gpu_k_off  = (size_t) gpu_physical * k_bytes_per_block_;
    const size_t gpu_v_off  = (size_t) gpu_physical * v_bytes_per_block_;
    const size_t warm_k_off = (size_t) cpu_idx     * k_bytes_per_block_;
    const size_t warm_v_off = (size_t) cpu_idx     * v_bytes_per_block_;

    for (uint32_t il = 0; il < layers_.size(); ++il) {
        const auto & layer = layers_[il];
        if (!layer.k) continue;
        ggml_backend_tensor_set(layer.k,
                                warm_k_[il].data() + warm_k_off,
                                gpu_k_off, k_bytes_per_block_);
        ggml_backend_tensor_set(layer.v,
                                warm_v_[il].data() + warm_v_off,
                                gpu_v_off, v_bytes_per_block_);
    }

    table_.swap_block(seq_id, logical_block, gpu_physical);
    pool_.free_block(cpu_physical);
    return true;
}

bool llama_kv_cache_paged::fault_in_warm_blocks_for_batch(const llama_ubatch & ub) {
    if (!warm_enabled() && !cold_enabled()) return true;

    // Step 1: identify all (seq, lblock) pairs the kernel will read for
    // this ubatch. Causal attention reads positions 0..max_pos per seq,
    // so we need every block in [0, max_pos / block_size_].
    std::vector<llama_pos> max_pos_per_seq(n_seq_max_, -1);
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        const llama_pos pos = ub.pos[i];
        if (pos < 0) continue;
        llama_seq_id sid = 0;
        if (ub.seq_id && ub.seq_id[i] && ub.n_seq_id && ub.n_seq_id[i] > 0) {
            sid = ub.seq_id[i][0];
        }
        if (sid < 0 || (uint32_t) sid >= n_seq_max_) continue;
        if (pos > max_pos_per_seq[sid]) max_pos_per_seq[sid] = pos;
    }

    // Build the "needed" set as a mask per seq for fast eviction-protection
    // queries. needed[sid] is the highest needed lblock for that seq;
    // any lblock <= needed[sid] for that seq is protected from eviction.
    std::vector<uint32_t> needed_max_lblock(n_seq_max_, UINT32_MAX);
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        if (max_pos_per_seq[s] >= 0) {
            needed_max_lblock[s] = (uint32_t) max_pos_per_seq[s] / block_size_;
        }
    }

    auto is_protected = [&](llama_seq_id sid, uint32_t lblock) {
        if (sid < 0 || (uint32_t) sid >= n_seq_max_) return false;
        const uint32_t cap = needed_max_lblock[sid];
        return cap != UINT32_MAX && lblock <= cap;
    };

    // Local LRU-with-protection eviction helper.
    auto evict_lru_protected = [&]() -> bool {
        constexpr uint32_t MIN_HOT_TAIL = 1;
        llama_seq_id victim_seq          = -1;
        uint32_t     victim_lblock       = 0;
        uint32_t     victim_seq_count    = 0;
        for (uint32_t s = 0; s < n_seq_max_; ++s) {
            const llama_seq_id sid = (llama_seq_id) s;
            const uint32_t n_blk = table_.num_blocks(sid);
            if (n_blk <= MIN_HOT_TAIL) continue;
            const uint32_t stop = n_blk - MIN_HOT_TAIL;
            uint32_t gpu_count = 0;
            uint32_t oldest_lb = UINT32_MAX;
            for (uint32_t lb = 0; lb < stop; ++lb) {
                if (is_protected(sid, lb)) continue;
                const uint32_t physical = table_.get_physical(sid, lb);
                if (physical == mt::kInvalidBlockId) continue;
                if (!pool_.is_gpu(physical)) continue;
                ++gpu_count;
                if (lb < oldest_lb) oldest_lb = lb;
            }
            if (gpu_count > victim_seq_count) {
                victim_seq        = sid;
                victim_lblock     = oldest_lb;
                victim_seq_count  = gpu_count;
            }
        }
        if (victim_seq < 0) return false;
        return evict_block_to_warm(victim_seq, victim_lblock);
    };

    // Step 2: walk needed (seq, lblock) pairs; fault in any not in GPU.
    uint32_t restored = 0;
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        const uint32_t cap = needed_max_lblock[s];
        if (cap == UINT32_MAX) continue;
        const llama_seq_id sid = (llama_seq_id) s;

        for (uint32_t lb = 0; lb <= cap; ++lb) {
            const uint32_t physical = table_.get_physical(sid, lb);
            const bool     in_cold  = is_cold(cold_slot_for_, sid, lb);

            if (in_cold) {
                // Cold-resident: physical is kInvalidBlockId (was freed
                // when we spilled to cold). Need to allocate GPU + read
                // from kvtcstore.
            } else {
                if (physical == mt::kInvalidBlockId) continue;
                if (pool_.is_gpu(physical)) continue;  // already hot
                // Warm-resident: existing path.
            }

            // Make GPU room. Try alloc; if full, evict LRU non-protected,
            // retry until we either get a slot or run out of victims.
            uint32_t free_gpu = (uint32_t) pool_.n_free_gpu();
            while (free_gpu == 0) {
                if (!evict_lru_protected()) {
                    LLAMA_LOG_WARN("fault_in_warm_blocks_for_batch: cannot make GPU room "
                                   "for seq %d lblock %u — all GPU blocks are protected. "
                                   "Pool too small for this batch's active ctx.\n",
                                   sid, lb);
                    return false;
                }
                free_gpu = (uint32_t) pool_.n_free_gpu();
            }

            const bool ok = in_cold
                ? restore_block_from_cold(sid, lb)
                : restore_block_from_warm(sid, lb);
            if (!ok) {
                LLAMA_LOG_WARN("fault_in_warm_blocks_for_batch: restore failed for "
                               "seq %d lblock %u (tier=%s)\n",
                               sid, lb, in_cold ? "cold" : "warm");
                return false;
            }
            ++restored;
        }
    }

    if (restored > 0) {
        LLAMA_LOG_DEBUG("fault_in_warm_blocks_for_batch: restored %u warm blocks "
                        "(pool free: gpu=%zu cpu=%zu)\n",
                        restored, pool_.n_free_gpu(), pool_.n_free_cpu());
    }
    return true;
}

bool llama_kv_cache_paged::evict_lru_to_warm() {
    if (!warm_enabled()) return false;

    // Simple LRU policy v0.1: pick the OLDEST (lowest logical index)
    // GPU-resident block from the seq with the most GPU-resident
    // non-tail blocks. Skip the most-recent block per seq so the
    // in-flight token's slot is preserved.
    constexpr uint32_t MIN_HOT_TAIL = 1;

    llama_seq_id victim_seq          = -1;
    uint32_t     victim_lblock       = 0;
    uint32_t     victim_seq_gpu_count = 0;

    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        const llama_seq_id sid = (llama_seq_id) s;
        const uint32_t n_blk = table_.num_blocks(sid);
        if (n_blk <= MIN_HOT_TAIL) continue;
        const uint32_t stop = n_blk - MIN_HOT_TAIL;

        uint32_t gpu_count = 0;
        uint32_t oldest_gpu_lblock = UINT32_MAX;
        for (uint32_t lb = 0; lb < stop; ++lb) {
            const uint32_t physical = table_.get_physical(sid, lb);
            if (physical == mt::kInvalidBlockId) continue;
            if (!pool_.is_gpu(physical)) continue;
            ++gpu_count;
            if (lb < oldest_gpu_lblock) oldest_gpu_lblock = lb;
        }

        if (gpu_count > victim_seq_gpu_count) {
            victim_seq           = sid;
            victim_lblock        = oldest_gpu_lblock;
            victim_seq_gpu_count = gpu_count;
        }
    }

    if (victim_seq < 0) return false;
    return evict_block_to_warm(victim_seq, victim_lblock);
}

// MAD-120 Phase 2: whole-slot preemption helpers ──────────────────────

uint32_t llama_kv_cache_paged::n_gpu_blocks_for(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return 0;
    const uint32_t n = table_.num_blocks(seq_id);
    uint32_t cnt = 0;
    for (uint32_t lb = 0; lb < n; ++lb) {
        const uint32_t phys = table_.get_physical(seq_id, lb);
        if (phys != mt::kInvalidBlockId && pool_.is_gpu(phys)) ++cnt;
    }
    return cnt;
}

uint32_t llama_kv_cache_paged::n_warm_blocks_for(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return 0;
    const uint32_t n = table_.num_blocks(seq_id);
    uint32_t cnt = 0;
    for (uint32_t lb = 0; lb < n; ++lb) {
        const uint32_t phys = table_.get_physical(seq_id, lb);
        if (phys != mt::kInvalidBlockId && !pool_.is_gpu(phys)) ++cnt;
    }
    return cnt;
}

int llama_kv_cache_paged::evict_seq_to_warm(llama_seq_id seq_id) {
    if (!warm_enabled()) return -1;
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return -1;
    const uint32_t n = table_.num_blocks(seq_id);
    int moved = 0;
    for (uint32_t lb = 0; lb < n; ++lb) {
        const uint32_t phys = table_.get_physical(seq_id, lb);
        if (phys == mt::kInvalidBlockId) continue;
        if (!pool_.is_gpu(phys)) continue;  // already warm
        if (!evict_block_to_warm(seq_id, lb)) {
            // Warm pool full or unexpected error; bail. Caller decides
            // how to proceed (cold tier when wired, or refuse the
            // candidate that triggered preemption).
            LLAMA_LOG_WARN("evict_seq_to_warm: stopped at seq=%d lblock=%u after "
                           "moving %d block(s) (warm pool full?)\n",
                           seq_id, lb, moved);
            return moved > 0 ? moved : -1;
        }
        ++moved;
    }
    LLAMA_LOG_DEBUG("evict_seq_to_warm: seq=%d evicted %d block(s)\n", seq_id, moved);
    return moved;
}

int llama_kv_cache_paged::restore_seq_from_warm(llama_seq_id seq_id) {
    if (!warm_enabled()) return -1;
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return -1;
    const uint32_t n = table_.num_blocks(seq_id);
    int moved = 0;
    for (uint32_t lb = 0; lb < n; ++lb) {
        const uint32_t phys = table_.get_physical(seq_id, lb);
        if (phys == mt::kInvalidBlockId) continue;
        if (pool_.is_gpu(phys)) continue;  // already on GPU
        if (!restore_block_from_warm(seq_id, lb)) {
            LLAMA_LOG_WARN("restore_seq_from_warm: stopped at seq=%d lblock=%u after "
                           "restoring %d block(s) (GPU pool full?)\n",
                           seq_id, lb, moved);
            return moved > 0 ? moved : -1;
        }
        ++moved;
    }
    LLAMA_LOG_DEBUG("restore_seq_from_warm: seq=%d restored %d block(s)\n", seq_id, moved);
    return moved;
}

bool llama_kv_cache_paged::evict_block_to_cold(llama_seq_id seq_id, uint32_t lblock) {
    if (!cold_enabled()) return false;
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;
    if (lblock >= table_.num_blocks(seq_id)) return false;
    if (cold_pool_free_.empty()) {
        LLAMA_LOG_WARN("evict_block_to_cold: cold pool full (%u in use); refusing\n", cold_in_use_);
        return false;
    }

    const uint32_t phys = table_.get_physical(seq_id, lblock);
    if (phys == mt::kInvalidBlockId) return false;

    const bool from_gpu = pool_.is_gpu(phys);
    const uint32_t cpu_idx = from_gpu ? UINT32_MAX : (phys - n_blocks_total_);

    const uint32_t cold_idx = cold_pool_free_.back();
    const off_t k_off_cold = (off_t) cold_idx * (off_t) k_bytes_per_block_;
    const off_t v_off_cold = (off_t) cold_idx * (off_t) v_bytes_per_block_;

    std::vector<uint8_t> kbuf(k_bytes_per_block_);
    std::vector<uint8_t> vbuf(v_bytes_per_block_);

    for (uint32_t il = 0; il < layers_.size(); ++il) {
        const auto & layer = layers_[il];
        if (!layer.k) continue;

        if (from_gpu) {
            const size_t k_off = (size_t) phys * k_bytes_per_block_;
            const size_t v_off = (size_t) phys * v_bytes_per_block_;
            ggml_backend_tensor_get(layer.k, kbuf.data(), k_off, k_bytes_per_block_);
            ggml_backend_tensor_get(layer.v, vbuf.data(), v_off, v_bytes_per_block_);
        } else {
            const size_t k_off = (size_t) cpu_idx * k_bytes_per_block_;
            const size_t v_off = (size_t) cpu_idx * v_bytes_per_block_;
            std::memcpy(kbuf.data(), warm_k_[il].data() + k_off, k_bytes_per_block_);
            std::memcpy(vbuf.data(), warm_v_[il].data() + v_off, v_bytes_per_block_);
        }

        const ssize_t wk = ::pwrite(cold_fd_k_[il], kbuf.data(), k_bytes_per_block_, k_off_cold);
        const ssize_t wv = ::pwrite(cold_fd_v_[il], vbuf.data(), v_bytes_per_block_, v_off_cold);
        if (wk != (ssize_t) k_bytes_per_block_ || wv != (ssize_t) v_bytes_per_block_) {
            LLAMA_LOG_WARN("evict_block_to_cold: pwrite failed at layer=%u cold_idx=%u (wk=%zd wv=%zd)\n",
                           il, cold_idx, wk, wv);
            return false;
        }
    }

    cold_pool_free_.pop_back();
    table_.swap_block(seq_id, lblock, mt::kInvalidBlockId);
    pool_.free_block(phys);
    mark_cold(cold_slot_for_, seq_id, lblock, cold_idx);
    ++cold_in_use_;
    return true;
}

bool llama_kv_cache_paged::restore_block_from_cold(llama_seq_id seq_id, uint32_t lblock) {
    if (!cold_enabled()) return false;
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;
    if (lblock >= table_.num_blocks(seq_id)) return false;
    const uint32_t cold_idx = cold_slot(cold_slot_for_, seq_id, lblock);
    if (cold_idx == kInvalidColdIdx) return false;

    const uint32_t gpu_phys = pool_.alloc_gpu();
    if (gpu_phys == mt::kInvalidBlockId) return false;

    const off_t k_off_cold = (off_t) cold_idx * (off_t) k_bytes_per_block_;
    const off_t v_off_cold = (off_t) cold_idx * (off_t) v_bytes_per_block_;

    std::vector<uint8_t> kbuf(k_bytes_per_block_);
    std::vector<uint8_t> vbuf(v_bytes_per_block_);

    for (uint32_t il = 0; il < layers_.size(); ++il) {
        const auto & layer = layers_[il];
        if (!layer.k) continue;

        const ssize_t rk = ::pread(cold_fd_k_[il], kbuf.data(), k_bytes_per_block_, k_off_cold);
        const ssize_t rv = ::pread(cold_fd_v_[il], vbuf.data(), v_bytes_per_block_, v_off_cold);
        if (rk != (ssize_t) k_bytes_per_block_ || rv != (ssize_t) v_bytes_per_block_) {
            LLAMA_LOG_WARN("restore_block_from_cold: pread short at layer=%u cold_idx=%u (rk=%zd rv=%zd)\n",
                           il, cold_idx, rk, rv);
            pool_.free_block(gpu_phys);
            return false;
        }

        const size_t k_off = (size_t) gpu_phys * k_bytes_per_block_;
        const size_t v_off = (size_t) gpu_phys * v_bytes_per_block_;
        ggml_backend_tensor_set(layer.k, kbuf.data(), k_off, k_bytes_per_block_);
        ggml_backend_tensor_set(layer.v, vbuf.data(), v_off, v_bytes_per_block_);
    }

    table_.swap_block(seq_id, lblock, gpu_phys);
    mark_cold(cold_slot_for_, seq_id, lblock, kInvalidColdIdx);
    cold_pool_free_.push_back(cold_idx);
    if (cold_in_use_ > 0) --cold_in_use_;
    return true;
}

bool llama_kv_cache_paged::evict_lru_warm_to_cold() {
    if (!cold_enabled() || cold_pool_free_.empty()) return false;

    // Pick any warm-resident (seq, lblock). We don't track warm-LRU
    // explicitly — first hit is fine for v1 (the warm tier itself is
    // already a cold-er-than-hot tier; the order within warm matters
    // less). Skip the seq's tail block to avoid evicting actively-
    // written data.
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        const uint32_t n = table_.num_blocks((llama_seq_id) s);
        if (n == 0) continue;
        const uint32_t tail = n - 1;
        for (uint32_t lb = 0; lb < n; ++lb) {
            if (lb == tail) continue;
            const uint32_t phys = table_.get_physical((llama_seq_id) s, lb);
            if (phys == mt::kInvalidBlockId) continue;
            if (pool_.is_gpu(phys)) continue;  // we want a WARM victim
            if (evict_block_to_cold((llama_seq_id) s, lb)) {
                return true;
            }
        }
    }
    return false;
}

bool llama_kv_cache_paged::can_admit(llama_seq_id           seq_id,
                                       uint32_t              n_new_tokens,
                                       const std::vector<llama_seq_id> & accepted) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;

    // Candidate seq's hot-block need = ceil((seq_pos_max + n_new + 1) / block).
    // seq_pos_max is -1 for an unwritten seq, so adjust.
    const llama_pos cand_pos_max = seq_states_[seq_id].pos_max;
    const uint64_t  cand_total_tokens =
        (cand_pos_max < 0 ? 0 : (uint64_t)(cand_pos_max + 1)) + (uint64_t) n_new_tokens;
    const uint32_t cand_blocks = (uint32_t)
        ((cand_total_tokens + block_size_ - 1) / block_size_);

    uint32_t total = cand_blocks;
    for (llama_seq_id sid : accepted) {
        if (sid < 0 || (uint32_t) sid >= n_seq_max_) continue;
        if (sid == seq_id) continue;  // candidate isn't double-counted
        const llama_pos pmax = seq_states_[sid].pos_max;
        if (pmax < 0) continue;
        const uint32_t b = (uint32_t)
            (((uint64_t)(pmax + 1) + block_size_ - 1) / block_size_);
        total += b;
    }

    // n_blocks_total_ is the hot pool capacity. Headroom of a couple
    // blocks for new-block allocation churn during this batch.
    const uint32_t headroom = 2;
    const uint32_t budget   = (n_blocks_total_ > headroom) ? (n_blocks_total_ - headroom) : n_blocks_total_;
    return total <= budget;
}

bool llama_kv_cache_paged::compute_slot_mapping(const llama_ubatch * ubatch, int32_t * out) const {
    // For each token i at (seq_id = ubatch->seq_id[i][0], pos = ubatch->pos[i]):
    //   logical_block_idx = pos / block_size_
    //   physical_block    = table_.get_physical(seq_id, logical_block_idx)
    //   slot_in_block     = pos % block_size_
    //   slot_mapping[i]   = physical_block * block_size_ + slot_in_block
    //
    // Multi-seq: each seq has its own logical→physical block list in
    // `table_`; tokens from different seqs in the same ubatch route to
    // different physical pages. ensure_blocks_for must have already been
    // called per-seq (init_batch handles that).
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_pos pos = ubatch->pos[i];
        if (pos < 0) {
            out[i] = -1;  // padding token
            continue;
        }
        // Resolve seq_id. ubatch->seq_id[i] is a small array; entry [0] is
        // the primary seq for this token. For tokens shared across seqs
        // (e.g. CoW), the underlying cache writes only the primary slot
        // and downstream attention sees both — matches non-paged behavior.
        llama_seq_id seq_id = 0;
        if (ubatch->seq_id && ubatch->seq_id[i] && ubatch->n_seq_id && ubatch->n_seq_id[i] > 0) {
            seq_id = ubatch->seq_id[i][0];
        }
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) {
            out[i] = -1;
            continue;
        }
        const uint32_t logical = (uint32_t) pos / block_size_;
        const uint32_t slot    = (uint32_t) pos % block_size_;
        const uint32_t n_blk   = table_.num_blocks(seq_id);
        if (logical >= n_blk) {
            // ensure_blocks_for should have allocated enough — caller bug.
            return false;
        }
        const uint32_t physical = table_.get_physical(seq_id, logical);
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
    // MAD-114: context_lens and q_lens are no longer uploaded here.
    // They're per-graph now (allocated in build_attn_inp_kv_paged_impl,
    // populated by the corresponding set_input from these host mirrors).
    // h_context_lens_ / h_q_lens_ are still populated above so set_input
    // can read them.
}

bool llama_kv_cache_paged::apply_ubatch_to_state(const llama_ubatch & ub) {
    // Per-seq: count tokens, find max position. Sized to n_seq_max_ for
    // direct indexing — n_seq_max_ is bounded (typical 4–32).
    std::vector<uint32_t>  tokens_per_seq(n_seq_max_, 0);
    std::vector<llama_pos> max_pos_per_seq(n_seq_max_, -1);
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        const llama_pos pos = ub.pos[i];
        if (pos < 0) continue;  // padding
        llama_seq_id seq_id = 0;
        if (ub.seq_id && ub.seq_id[i] && ub.n_seq_id && ub.n_seq_id[i] > 0) {
            seq_id = ub.seq_id[i][0];
        }
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) continue;
        tokens_per_seq[seq_id]++;
        if (pos > max_pos_per_seq[seq_id]) max_pos_per_seq[seq_id] = pos;
    }


    // Allocate blocks per-seq sized to the max position written. Note: we
    // don't use ensure_blocks_for's "pos_max + n_new" arithmetic here
    // because in multi-seq prefill the positions in `pos[]` are absolute
    // (e.g. seq A might be at positions 0..63 while seq B is at 100..107
    // in the same ubatch). We instead allocate enough blocks to fit
    // max_pos_per_seq[s] for each affected seq.
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        if (tokens_per_seq[s] == 0) continue;
        const llama_pos new_max = max_pos_per_seq[s];
        const llama_pos cur_max = seq_states_[s].pos_max;
        if (new_max > cur_max) {
            const uint32_t n_new = (uint32_t)(new_max - cur_max);
            if (!ensure_blocks_for((llama_seq_id) s, n_new)) {
                return false;
            }
        }
    }

    // Commit per-seq state.
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        h_q_lens_[s] = (int32_t) tokens_per_seq[s];
        if (tokens_per_seq[s] > 0) {
            if (max_pos_per_seq[s] > seq_states_[s].pos_max) {
                seq_states_[s].pos_max = max_pos_per_seq[s];
            }
            if (seq_states_[s].pos_min < 0) seq_states_[s].pos_min = 0;
        }
    }

    // MAD-120: fault in any warm-tier blocks the kernel will read this
    // batch. Must run AFTER ensure_blocks_for above (so new tokens have
    // GPU slots allocated) and BEFORE prepare_batch_tensors (so the
    // h_block_table_ snapshot reflects post-fault-in physicals). No-op
    // when warm tier is disabled.
    if (warm_enabled() && !fault_in_warm_blocks_for_batch(ub)) {
        return false;
    }
    return true;
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch_with_ubatches(
        std::vector<llama_ubatch> ubatches) {
    if (ubatches.empty()) {
        return std::make_unique<llama_kv_cache_paged_context>(
            this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
    }
    std::fill(h_q_lens_.begin(), h_q_lens_.end(), 0);

    // MAD-124: capture per-ubatch snapshots of (q_lens, context_lens) right
    // after each apply_ubatch_to_state, so the per-graph paged_q_lens /
    // paged_context_lens tensors can be uploaded with the correct values
    // for THIS ubatch (instead of the eager loop's final-state leakage).
    std::vector<std::vector<int32_t>> q_lens_snap;
    std::vector<std::vector<int32_t>> ctx_lens_snap;
    q_lens_snap.reserve(ubatches.size());
    ctx_lens_snap.reserve(ubatches.size());

    for (const auto & ub : ubatches) {
        if (!apply_ubatch_to_state(ub)) {
            return std::make_unique<llama_kv_cache_paged_context>(
                this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        q_lens_snap.emplace_back(h_q_lens_.begin(), h_q_lens_.end());
        std::vector<int32_t> cl(n_seq_max_);
        for (uint32_t s = 0; s < n_seq_max_; ++s) {
            cl[s] = (int32_t) std::max<llama_pos>(0, seq_states_[s].pos_max + 1);
        }
        ctx_lens_snap.push_back(std::move(cl));
    }
    prepare_batch_tensors();
    return std::make_unique<llama_kv_cache_paged_context>(
        this, std::move(ubatches),
        std::move(q_lens_snap), std::move(ctx_lens_snap),
        LLAMA_MEMORY_STATUS_SUCCESS);
}

llama_memory_context_ptr llama_kv_cache_paged::init_batch(
        llama_batch_allocr & balloc, uint32_t n_ubatch, bool /*embd_all*/) {
    // Split the batch into ubatches. Each ubatch may contain tokens from
    // multiple seqs; apply_ubatch_to_state buckets per-seq and routes
    // block allocation accordingly.
    std::vector<llama_ubatch> ubatches;

    while (true) {
        llama_ubatch ub = balloc.split_simple(n_ubatch);
        if (ub.n_tokens == 0) break;
        ubatches.push_back(std::move(ub));
    }

    if (ubatches.empty()) {
        return std::make_unique<llama_kv_cache_paged_context>(
            this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_NO_UPDATE);
    }

    std::fill(h_q_lens_.begin(), h_q_lens_.end(), 0);

    // MAD-124: per-ubatch snapshots; see init_batch_with_ubatches.
    std::vector<std::vector<int32_t>> q_lens_snap;
    std::vector<std::vector<int32_t>> ctx_lens_snap;
    q_lens_snap.reserve(ubatches.size());
    ctx_lens_snap.reserve(ubatches.size());

    for (const auto & ub : ubatches) {
        if (!apply_ubatch_to_state(ub)) {
            return std::make_unique<llama_kv_cache_paged_context>(
                this, std::vector<llama_ubatch>{}, LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }
        q_lens_snap.emplace_back(h_q_lens_.begin(), h_q_lens_.end());
        std::vector<int32_t> cl(n_seq_max_);
        for (uint32_t s = 0; s < n_seq_max_; ++s) {
            cl[s] = (int32_t) std::max<llama_pos>(0, seq_states_[s].pos_max + 1);
        }
        ctx_lens_snap.push_back(std::move(cl));
    }

    prepare_batch_tensors();

    return std::make_unique<llama_kv_cache_paged_context>(
        this, std::move(ubatches),
        std::move(q_lens_snap), std::move(ctx_lens_snap),
        LLAMA_MEMORY_STATUS_SUCCESS);
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

    // Match the regular kv_cache convention: p0 < 0 → from 0; p1 < 0 →
    // to max. The server uses p1 = -1 to mean "to end", NOT "wipe
    // everything" — earlier code treated p1<0 as a whole-seq wipe and
    // produced position drift between the cache and the slot manager.
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();

    const llama_pos cur_max = seq_states_[seq_id].pos_max;

    // Whole-seq wipe when the range covers everything.
    if (p0 == 0 && p1 > cur_max) {
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
    // ranges requires more bookkeeping; rare in practice).
    if (p0 > cur_max) return true;  // nothing to do
    // Middle wipes (p1 < cur_max+1 with p0 > 0) aren't supported in v1.
    // The server uses these for partial speculative-decode rollback;
    // until we add support, treat as tail truncation from p0 (drops
    // some valid positions but doesn't crash).
    GGML_UNUSED(p1);

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

llama_kv_cache_paged_context::llama_kv_cache_paged_context(
        llama_kv_cache_paged *               parent,
        std::vector<llama_ubatch>           ubatches,
        std::vector<std::vector<int32_t>>   q_lens_per_ubatch,
        std::vector<std::vector<int32_t>>   context_lens_per_ubatch,
        llama_memory_status                 status)
    : parent_(parent),
      ubatches_(std::move(ubatches)),
      q_lens_per_ubatch_(std::move(q_lens_per_ubatch)),
      context_lens_per_ubatch_(std::move(context_lens_per_ubatch)),
      status_(status) {}

bool llama_kv_cache_paged_context::next() {
    if (++i_ubatch_ >= ubatches_.size()) return false;
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // MAD-124: install per-ubatch q_lens / context_lens into the parent's
    // host mirrors before this ubatch's graph runs. set_input on the per-
    // graph paged_q_lens / paged_context_lens tensors will then upload the
    // right values for THIS ubatch (instead of whatever was last written
    // during init_batch's eager loop).
    if (i_ubatch_ < q_lens_per_ubatch_.size()) {
        const auto & qsnap = q_lens_per_ubatch_[i_ubatch_];
        const auto & csnap = context_lens_per_ubatch_[i_ubatch_];
        const size_t n = parent_->h_q_lens_.size();
        for (size_t s = 0; s < n; ++s) {
            parent_->h_q_lens_[s]       = (s < qsnap.size()) ? qsnap[s] : 0;
            parent_->h_context_lens_[s] = (s < csnap.size()) ? csnap[s] : 0;
        }
    }
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    GGML_ASSERT(i_ubatch_ < ubatches_.size());
    return ubatches_[i_ubatch_];
}
