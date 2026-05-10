#include "llama-kv-cache-paged.h"

#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-hparams.h"
#include "ggml-backend.h"

#include <cassert>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/file.h>
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
        bool                       cold_resume,
        std::string                instance_id,
        uint32_t                   cold_budget_mb,
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
            // MAD-131: per-instance subdir + lockfile. Default instance
            // ID is the process pid as a string; operator override via
            // --instance-id makes restarts deterministic.
            instance_id_ = instance_id.empty() ? std::to_string(::getpid()) : instance_id;

            // MAD-131: budget cap on cold pool size. Convert MiB → blocks
            // (across K+V per attention layer). Cap n_cold_blocks_
            // BEFORE building the file paths so the truncation/sizing
            // arithmetic uses the capped count.
            if (cold_budget_mb > 0) {
                const size_t bytes_budget = (size_t) cold_budget_mb * 1024ull * 1024ull;
                const size_t bytes_per_block_per_layer = k_bytes_per_block_ + v_bytes_per_block_;
                if (bytes_per_block_per_layer == 0 || n_attn_layers == 0) {
                    LLAMA_LOG_WARN("llama_kv_cache_paged: cold-budget-mb=%u set but layer sizing "
                                   "is unknown; ignoring cap\n", cold_budget_mb);
                } else {
                    const size_t bytes_per_block = bytes_per_block_per_layer * n_attn_layers;
                    const uint32_t max_blocks_for_budget = (uint32_t) std::min<size_t>(
                        UINT32_MAX, bytes_budget / bytes_per_block);
                    if (max_blocks_for_budget < n_cold_blocks_) {
                        LLAMA_LOG_INFO("llama_kv_cache_paged: cold-budget-mb=%u caps cold pool "
                                       "to %u blocks (was %u)\n",
                                       cold_budget_mb, max_blocks_for_budget, n_cold_blocks_);
                        n_cold_blocks_ = max_blocks_for_budget;
                    }
                }
            }
            if (n_cold_blocks_ == 0) {
                // Budget reduced cold to zero; skip file setup.
                LLAMA_LOG_WARN("llama_kv_cache_paged: cold-budget-mb capped pool to 0 — "
                               "cold tier effectively disabled\n");
                goto cold_setup_done;
            }

            const std::string subdir =
                cold_path_ + "/paged/instance-" + instance_id_;
            (void) ::mkdir(cold_path_.c_str(),                         0700);
            (void) ::mkdir((cold_path_ + "/paged").c_str(),            0700);
            (void) ::mkdir(subdir.c_str(),                             0700);

            // MAD-131: lockfile + .pid for double-start refusal.
            // flock(LOCK_EX | LOCK_NB) gives clean OS-level mutual
            // exclusion. The .pid file is informational — read it on
            // lock failure to give the operator a useful diagnostic.
            const std::string lock_path = subdir + "/.lock";
            const std::string pid_path  = subdir + "/.pid";
            cold_lock_path_ = lock_path;
            cold_pid_path_  = pid_path;
            cold_lock_fd_ = ::open(lock_path.c_str(), O_RDWR | O_CREAT, 0600);
            if (cold_lock_fd_ < 0) {
                throw std::runtime_error("llama_kv_cache_paged: cannot open lockfile " + lock_path);
            }
            if (::flock(cold_lock_fd_, LOCK_EX | LOCK_NB) != 0) {
                std::string holder_pid = "(unknown)";
                std::ifstream pf(pid_path);
                if (pf) std::getline(pf, holder_pid);
                ::close(cold_lock_fd_);
                cold_lock_fd_ = -1;
                throw std::runtime_error(
                    "llama_kv_cache_paged: instance " + instance_id_ +
                    " is already in use by pid " + holder_pid +
                    " (lockfile " + lock_path + " held). Use a different "
                    "--instance-id, or stop the holder, or rm the lockfile "
                    "if it's stale.");
            }
            // Write our pid into .pid so future failed-acquires can
            // diagnose. Best-effort; not fatal if write fails.
            {
                std::ofstream pf(pid_path, std::ios::trunc);
                if (pf) pf << ::getpid() << "\n";
            }

            cold_fd_k_.assign(n_layer, -1);
            cold_fd_v_.assign(n_layer, -1);
            const off_t k_file_bytes = (off_t) n_cold_blocks_ * (off_t) k_bytes_per_block_;
            const off_t v_file_bytes = (off_t) n_cold_blocks_ * (off_t) v_bytes_per_block_;
            uint64_t total_bytes = 0;
            bool ok = true;
            // MAD-130: cold_resume=true → open WITHOUT O_TRUNC so existing
            // bytes survive. We'll load the in-memory index from the
            // sidecar after the layer files are open.
            const int open_flags = cold_resume
                ? (O_RDWR | O_CREAT)            // preserve contents
                : (O_RDWR | O_CREAT | O_TRUNC); // legacy: fresh
            for (uint32_t il = 0; il < n_layer && ok; ++il) {
                if (!layers_[il].k) continue;
                const std::string kpath = subdir + "/L" + std::to_string(il) + ".k.bin";
                const std::string vpath = subdir + "/L" + std::to_string(il) + ".v.bin";
                const int fk = ::open(kpath.c_str(), open_flags, 0600);
                const int fv = ::open(vpath.c_str(), open_flags, 0600);
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
                // Free-stack of cold slot ids. Initialize as all-free;
                // the cold-resume sidecar load below will mark in-use
                // slots and pull them off the free stack.
                cold_pool_free_.reserve(n_cold_blocks_);
                for (uint32_t i = n_cold_blocks_; i > 0; --i) {
                    cold_pool_free_.push_back(i - 1);
                }
                cold_slot_for_.assign(n_seq_max_, std::vector<uint32_t>());

                // MAD-130: cold-resume — load the in-memory index from
                // the sidecar. Sidecar absent → log warning but proceed
                // (the on-disk bytes exist but we don't know what's
                // in them; first-write of any slot will overwrite).
                if (cold_resume) {
                    const std::string sidecar_path = subdir + "/index.bin";
                    if (!load_cold_index_sidecar_(sidecar_path)) {
                        LLAMA_LOG_WARN("llama_kv_cache_paged: --kv-tier-cold-resume requested but "
                                       "sidecar %s missing or invalid — starting with empty cold "
                                       "index (existing file bytes ignored)\n",
                                       sidecar_path.c_str());
                    }
                }

                LLAMA_LOG_INFO("llama_kv_cache_paged: cold tier enabled — %u blocks × "
                               "%.1f KiB/block (K+V) × %u attn layers = %.1f MiB on %s%s "
                               "(instance=%s)\n",
                               n_cold_blocks_,
                               (double)(k_bytes_per_block_ + v_bytes_per_block_) / 1024.0,
                               n_attn_layers,
                               (double) total_bytes / (1024.0 * 1024.0),
                               subdir.c_str(),
                               cold_resume ? " (resume mode)" : "",
                               instance_id_.c_str());
            }
        }
    }
cold_setup_done:;

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

// MAD-132: enforce the single-threading contract. First mutator call
// captures std::this_thread::get_id(); subsequent calls assert match.
// Release builds compile this to a no-op via assert().
//
// Method is const to allow calling from const methods that mutate
// internal caches (none today, but const-correctness for future).
// The captured_thread_id_ field is mutable for the same reason.
void llama_kv_cache_paged::check_thread_id_() const {
#ifndef NDEBUG
    const std::thread::id self = std::this_thread::get_id();
    if (captured_thread_id_ == std::thread::id()) {
        captured_thread_id_ = self;
    } else {
        assert(captured_thread_id_ == self &&
               "llama_kv_cache_paged: cross-thread mutation detected. "
               "This cache is single-threaded; see header doc block "
               "(MAD-132 / Epic A4). Add explicit synchronization or "
               "use a worker queue if async access is required.");
    }
#endif
}

// MAD-132: timestamp helper (microseconds since epoch). Used for
// preempt-fairness victim selection.
static uint64_t now_us_() {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

llama_kv_cache_paged::~llama_kv_cache_paged() {
    // MAD-121: close cold-tier fds.
    for (int fd : cold_fd_k_) if (fd >= 0) ::close(fd);
    for (int fd : cold_fd_v_) if (fd >= 0) ::close(fd);
    // MAD-131: release the per-instance lockfile + remove .pid. flock
    // releases automatically when the fd closes; .pid is informational
    // and best-effort cleaned (next start with same instance_id will
    // overwrite anyway).
    if (cold_lock_fd_ >= 0) {
        ::close(cold_lock_fd_);
        cold_lock_fd_ = -1;
    }
    if (!cold_pid_path_.empty()) {
        (void) ::unlink(cold_pid_path_.c_str());
    }
    if (buf_storage_) ggml_backend_buffer_free(buf_storage_);
    if (buf_inputs_)  ggml_backend_buffer_free(buf_inputs_);
    if (ctx_storage_) ggml_free(ctx_storage_);
    if (ctx_inputs_)  ggml_free(ctx_inputs_);
}

bool llama_kv_cache_paged::ensure_blocks_for(llama_seq_id seq_id, uint32_t n_new_tokens) {
    check_thread_id_();
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
                if (pool_.refcount(physical) > 1) continue;  // MAD-128: skip shared
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

// MAD-128: CoW any blocks being written this ubatch that are shared
// (refcount > 1 from a prior seq_cp). For each unique (seq, lblock)
// touched by a write in `ub`: if that physical's refcount > 1, allocate
// a fresh GPU block, copy K/V from the shared block into it, swap the
// seq's table entry, decrement the old block's refcount.
//
// Why this is needed: seq_cp shares physical blocks via refcount. If
// either sequence then writes to a shared block (e.g. partial last
// block at the share boundary), the kernel's K/V scatter would corrupt
// the OTHER sequence's view of that block. CoW preserves the invariant
// that each (seq, lblock) writable cell is uniquely owned.
//
// Called from apply_ubatch_to_state AFTER fault-in (so the block is
// GPU-resident before we copy) and BEFORE prepare_batch_tensors (so
// the uploaded block_table reflects the new physicals).
bool llama_kv_cache_paged::cow_writes_for_ubatch(const llama_ubatch & ub) {
    if (ub.n_tokens == 0) return true;

    // Collect unique (seq, lblock) pairs touched by writes this ubatch.
    // Using a set: small N (<= n_tokens), de-dup'd lookups.
    std::vector<std::pair<llama_seq_id, uint32_t>> touched;
    touched.reserve(ub.n_tokens);
    for (uint32_t i = 0; i < ub.n_tokens; ++i) {
        const llama_pos pos = ub.pos[i];
        if (pos < 0) continue;
        llama_seq_id sid = 0;
        if (ub.seq_id && ub.seq_id[i] && ub.n_seq_id && ub.n_seq_id[i] > 0) {
            sid = ub.seq_id[i][0];
        }
        if (sid < 0 || (uint32_t) sid >= n_seq_max_) continue;
        touched.emplace_back(sid, (uint32_t) pos / block_size_);
    }
    std::sort(touched.begin(), touched.end());
    touched.erase(std::unique(touched.begin(), touched.end()), touched.end());

    uint32_t n_cowed = 0;
    for (auto [sid, lblock] : touched) {
        if (lblock >= table_.num_blocks(sid)) continue;  // freshly allocated by ensure_blocks_for; refcount=1
        const uint32_t physical = table_.get_physical(sid, lblock);
        if (physical == mt::kInvalidBlockId) continue;
        if (pool_.refcount(physical) <= 1) continue;       // not shared
        if (!pool_.is_gpu(physical)) continue;             // CoW only on GPU-resident; warm/cold writes go through fault-in path

        // Allocate a fresh GPU block. If the pool is full, try evicting
        // an LRU GPU block to warm to make room (only if warm enabled).
        uint32_t new_phys = pool_.alloc_gpu();
        while (new_phys == mt::kInvalidBlockId && warm_enabled()) {
            if (!evict_lru_to_warm()) break;
            new_phys = pool_.alloc_gpu();
        }
        if (new_phys == mt::kInvalidBlockId) {
            LLAMA_LOG_ERROR("llama_kv_cache_paged::cow_writes_for_ubatch: GPU pool "
                            "exhausted attempting CoW for seq=%d lblock=%u — "
                            "write would corrupt shared block. Refusing batch.\n",
                            sid, lblock);
            return false;
        }

        // GPU-to-GPU copy via host bounce buffer (no native ggml D2D
        // primitive; the get/set pair routes through host memory but the
        // block is small — k+v_bytes_per_block, ~17 KiB at turbo4).
        std::vector<uint8_t> kbuf(k_bytes_per_block_);
        std::vector<uint8_t> vbuf(v_bytes_per_block_);
        const size_t k_off_old = (size_t) physical * k_bytes_per_block_;
        const size_t v_off_old = (size_t) physical * v_bytes_per_block_;
        const size_t k_off_new = (size_t) new_phys * k_bytes_per_block_;
        const size_t v_off_new = (size_t) new_phys * v_bytes_per_block_;
        for (uint32_t il = 0; il < layers_.size(); ++il) {
            const auto & layer = layers_[il];
            if (!layer.k) continue;
            ggml_backend_tensor_get(layer.k, kbuf.data(), k_off_old, k_bytes_per_block_);
            ggml_backend_tensor_get(layer.v, vbuf.data(), v_off_old, v_bytes_per_block_);
            ggml_backend_tensor_set(layer.k, kbuf.data(), k_off_new, k_bytes_per_block_);
            ggml_backend_tensor_set(layer.v, vbuf.data(), v_off_new, v_bytes_per_block_);
        }

        // Swap seq's table entry to the fresh block; decrement old.
        // The old block's refcount drops by 1 (from N to N-1). If N was
        // 2, the old block now has refcount 1 (the OTHER seq still owns
        // it, no one freed). If N was higher, more shares remain.
        // Fingerprint follows the seq's logical block — same content,
        // new physical, no fingerprint update needed.
        table_.swap_block(sid, lblock, new_phys);
        pool_.free_block(physical);
        ++n_cowed;
    }

    if (n_cowed > 0) {
        LLAMA_LOG_DEBUG("llama_kv_cache_paged::cow_writes_for_ubatch: CoW'd %u "
                        "shared block(s) for upcoming writes (pool free: gpu=%zu)\n",
                        n_cowed, pool_.n_free_gpu());
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
            // MAD-128: skip shared blocks (refcount > 1) — evicting them
            // doesn't free GPU space (other seqs still hold the physical),
            // so the eviction-retry caller would loop. Shared blocks stay
            // GPU-resident until either the other seq frees its reference
            // or its own write triggers CoW.
            if (pool_.refcount(physical) > 1) continue;
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
        // MAD-132: cold full → escalate by dropping the oldest cold
        // block to make room. The dropped seq's K/V is gone (its table
        // entry was already kInvalidBlockId from prior cold spill);
        // future kernel reads of that block return -INFINITY logit (=
        // zero attention contribution), same as middle-wipe holes.
        // This is the last-resort policy before refusing the eviction
        // and surfacing FAILED_PREPARE up the call chain.
        if (!drop_oldest_cold_block()) {
            LLAMA_LOG_WARN("evict_block_to_cold: cold pool full (%u in use) AND drop_oldest "
                           "failed; refusing eviction. Caller should fall back to keeping "
                           "the block in warm or returning a 503 to the client.\n",
                           cold_in_use_);
            return false;
        }
        LLAMA_LOG_INFO("evict_block_to_cold: cold pool was full; dropped oldest cold block "
                       "to make room for seq=%d lblock=%u\n", seq_id, lblock);
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

// MAD-125: BGE-small semantic prefetch — record + restore.

void llama_kv_cache_paged::record_paged_block_fingerprint(
        llama_seq_id            seq_id,
        uint32_t                lblock,
        std::vector<float>      embedding,
        mt::SemanticIndex::Tier tier) {
    paged_semantic_.add_fingerprint(seq_id, lblock, std::move(embedding), tier);
}

uint32_t llama_kv_cache_paged::restore_semantic_paged(
        llama_seq_id              seq_id,
        const std::vector<float> & query_embedding,
        int                        top_k,
        float                      threshold) {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return 0;

    auto hints = paged_semantic_.score(seq_id, query_embedding, top_k, threshold);
    if (hints.empty()) return 0;

    uint32_t restored      = 0;
    uint32_t requested     = 0;
    uint32_t already_hot   = 0;
    uint32_t unmapped      = 0;
    uint32_t restore_fail  = 0;

    for (const auto & h : hints) {
        ++requested;
        if (h.lblock >= table_.num_blocks(seq_id)) { ++unmapped; continue; }
        const uint32_t phys = table_.get_physical(seq_id, h.lblock);
        if (phys == mt::kInvalidBlockId) { ++unmapped; continue; }

        if (pool_.is_gpu(phys)) { ++already_hot; continue; }

        // Try warm first; if warm restore fails (e.g. block is in cold)
        // fall back to cold restore. Both are no-ops if the tier isn't
        // configured.
        bool ok = warm_enabled() && restore_block_from_warm(seq_id, h.lblock);
        if (!ok && cold_enabled()) {
            ok = restore_block_from_cold(seq_id, h.lblock);
        }
        if (ok) ++restored;
        else    ++restore_fail;
    }

    // MAD-122 acceptance criterion #5: hit-rate logging. The
    // restored/requested ratio is the prefetch-effectiveness signal.
    LLAMA_LOG_INFO("llama_kv_cache_paged::restore_semantic_paged: seq %d — %zu hints "
                   "(top_k=%d, threshold=%.2f), restored %u/%u "
                   "(hit-rate %.0f%%, %u already hot, %u unmapped, %u failed)\n",
                   seq_id, hints.size(), top_k, threshold,
                   restored, requested,
                   requested == 0 ? 0.0f : 100.0f * (float) restored / (float) requested,
                   already_hot, unmapped, restore_fail);

    return restored;
}

// MAD-132: drop the oldest cold block — last-resort escalation when
// cold pool is full but eviction must continue. Walk cold_slot_for_
// in seq-then-lblock order and pick the first in-use entry. This is
// "any" not strictly "oldest" — the cold tier doesn't track per-slot
// age. Acceptable for v1: the cold spillover is itself age-ordered
// (LRU warm → cold), so the lowest-indexed cold slots are typically
// the oldest evictions. True LRU tracking is a follow-up.
//
// Effect: the dropped block's data is gone. The owning seq's table
// entry stays kInvalidBlockId (already marked when the block was
// spilled to cold); future kernel reads contribute -INFINITY logit
// → 0 attention weight, same as middle-wipe holes. The seq sees a
// hole at that block position; correctness is preserved (no garbage
// reads), recall is degraded (the dropped K/V can't be attended to).
bool llama_kv_cache_paged::drop_oldest_cold_block() {
    if (!cold_enabled()) return false;

    // Scan for any in-use cold slot.
    for (uint32_t s = 0; s < (uint32_t) cold_slot_for_.size(); ++s) {
        auto & row = cold_slot_for_[s];
        for (uint32_t lb = 0; lb < (uint32_t) row.size(); ++lb) {
            if (row[lb] == kInvalidColdIdx) continue;
            const uint32_t cold_idx = row[lb];
            row[lb] = kInvalidColdIdx;
            cold_pool_free_.push_back(cold_idx);
            if (cold_in_use_ > 0) --cold_in_use_;
            LLAMA_LOG_INFO("llama_kv_cache_paged::drop_oldest_cold_block: dropped "
                           "(seq=%u, lblock=%u, cold_idx=%u) — owning seq sees a "
                           "hole at this block (kernel handles via -INFINITY logit)\n",
                           s, lb, cold_idx);
            return true;
        }
    }
    return false;
}

// MAD-132: idle-priority victim selection for whole-slot preemption.
// Picks the seq with the smallest (oldest) last_active_us, excluding
// any seq in exclude_seqs and any seq that has zero blocks (nothing
// to preempt). Returns -1 if no eligible victim exists.
llama_seq_id llama_kv_cache_paged::pick_preempt_victim(
        const std::vector<llama_seq_id> & exclude_seqs) const {
    auto excluded = [&](llama_seq_id sid) {
        for (auto e : exclude_seqs) if (e == sid) return true;
        return false;
    };

    llama_seq_id victim = -1;
    uint64_t     oldest = UINT64_MAX;
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        const llama_seq_id sid = (llama_seq_id) s;
        if (excluded(sid)) continue;
        if (table_.num_blocks(sid) == 0) continue;            // no blocks
        if (n_gpu_blocks_for(sid) == 0) continue;             // already preempted
        const uint64_t last = seq_states_[s].last_active_us;
        if (last < oldest) {
            oldest = last;
            victim = sid;
        }
    }
    return victim;
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
    const uint64_t now = now_us_();
    for (uint32_t s = 0; s < n_seq_max_; ++s) {
        h_q_lens_[s] = (int32_t) tokens_per_seq[s];
        if (tokens_per_seq[s] > 0) {
            if (max_pos_per_seq[s] > seq_states_[s].pos_max) {
                seq_states_[s].pos_max = max_pos_per_seq[s];
            }
            if (seq_states_[s].pos_min < 0) seq_states_[s].pos_min = 0;
            // MAD-132: stamp last-active so preempt-fairness picks
            // genuinely idle seqs over actively-batched ones.
            seq_states_[s].last_active_us = now;
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

    // MAD-128: CoW any blocks being written this ubatch that are shared
    // (refcount > 1 from a prior seq_cp). Must run AFTER fault-in (so
    // the block is GPU-resident before we attempt to copy its data) and
    // BEFORE prepare_batch_tensors (so the table snapshot reflects the
    // CoW'd physicals). No-op when no shared blocks exist.
    if (!cow_writes_for_ubatch(ub)) {
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
    check_thread_id_();
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
    paged_semantic_.clear();  // MAD-125: drop all per-seq fingerprints
    std::fill(h_block_table_.begin(), h_block_table_.end(), kInvalidBlockTableEntry);
    std::fill(h_context_lens_.begin(), h_context_lens_.end(), 0);
    std::fill(h_q_lens_.begin(), h_q_lens_.end(), 0);
    LLAMA_LOG_DEBUG("llama_kv_cache_paged::clear: all blocks released, all seqs reset\n");
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    check_thread_id_();
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max_) return false;

    // Match the regular kv_cache convention: p0 < 0 → from 0; p1 < 0 →
    // to max. The server uses p1 = -1 to mean "to end", NOT "wipe
    // everything" — earlier code treated p1<0 as a whole-seq wipe and
    // produced position drift between the cache and the slot manager.
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    if (p1 <= p0) return true;  // empty range

    const llama_pos cur_max = seq_states_[seq_id].pos_max;

    // Whole-seq wipe when the range covers everything.
    if (p0 == 0 && p1 > cur_max) {
        std::vector<uint32_t> freed = table_.clear_seq(seq_id);
        for (uint32_t bid : freed) {
            if (bid != mt::kInvalidBlockId) pool_.free_block(bid);
        }
        seq_states_[seq_id] = seq_state{};
        // MAD-125: fingerprints from the prior task can't match the new
        // one's K/V — wipe them so semantic restore doesn't fault in stale
        // blocks. Mirrors the wrapper's whole-seq-wipe behavior.
        const size_t n_finger = paged_semantic_.size(seq_id);
        paged_semantic_.remove_seq(seq_id);
        LLAMA_LOG_DEBUG("llama_kv_cache_paged::seq_rm: whole-seq wipe seq=%d, freed %zu blocks, dropped %zu paged-block fingerprints\n",
                        seq_id, freed.size(), n_finger);
        return true;
    }

    if (p0 > cur_max) return true;  // nothing in range

    // MAD-128: block-aligned partial wipe. Handles both tail truncate
    // (p1 > cur_max) and middle wipe (p1 <= cur_max) uniformly.
    //
    // For each block whose [b_start, b_end) intersects [p0, p1):
    //   - Wholly covered (p0 <= b_start AND p1 >= b_end): free the
    //     physical, mark table entry as kInvalidBlockId, drop fingerprint.
    //     The mt_paged_attention_kernel handles invalid entries by
    //     contributing -INFINITY to attention logits → zero weight after
    //     softmax → no contribution. So freed blocks read as "no
    //     attention," not as garbage.
    //   - Partially overlapped (p0 inside the block OR p1 inside the
    //     block): leave the block alone. The unwiped slots within keep
    //     stale K/V that the kernel WILL read. Caller should round their
    //     range to block boundaries for clean wipes; we log a clear
    //     warning so sub-block partial wipes don't silently corrupt.
    //
    // Sub-block per-slot zeroing requires either a per-block valid-bitmask
    // consulted by the kernel mask, or a layout-aware per-slot zero
    // primitive. Both are deferred (separate ticket if a real consumer
    // needs sub-block precision).
    const llama_pos p1_clamped = std::min(p1, (llama_pos)(cur_max + 1));
    const uint32_t bsize = block_size_;
    const uint32_t n_blocks_seq = table_.num_blocks(seq_id);

    uint32_t blocks_freed           = 0;
    uint32_t blocks_partial_skipped = 0;

    for (uint32_t lblock = 0; lblock < n_blocks_seq; ++lblock) {
        const llama_pos b_start = (llama_pos) lblock * (llama_pos) bsize;
        const llama_pos b_end   = b_start + (llama_pos) bsize;

        if (b_end <= p0 || b_start >= p1_clamped) continue;  // no overlap

        const bool wholly_covered = (p0 <= b_start) && (p1_clamped >= b_end);
        if (wholly_covered) {
            const uint32_t physical = table_.get_physical(seq_id, lblock);
            if (physical != mt::kInvalidBlockId) {
                pool_.free_block(physical);
                table_.swap_block(seq_id, lblock, mt::kInvalidBlockId);
                paged_semantic_.remove_block(seq_id, lblock);
                ++blocks_freed;
            }
        } else {
            ++blocks_partial_skipped;
        }
    }

    if (blocks_partial_skipped > 0) {
        LLAMA_LOG_WARN("llama_kv_cache_paged::seq_rm: range [%d,%d) is not "
                       "block-aligned (block_size=%u). %u block(s) wholly "
                       "freed; %u block(s) partially overlapped — those keep "
                       "stale K/V in the unwiped slots and the kernel will "
                       "attend to them. Round caller's range to block "
                       "boundaries for clean wipes.\n",
                       p0, p1, bsize, blocks_freed, blocks_partial_skipped);
    }

    // Update pos_max iff the wipe touches the tail (p1 covers past cur_max).
    // For middle wipes (p1_clamped <= cur_max), pos_max stays put — holes
    // in the block table represent the wiped middle. For tail truncate,
    // shrink pos_max to the position just before the wipe started.
    if (p1 > cur_max) {
        seq_states_[seq_id].pos_max = (llama_pos)(p0 - 1);
        if (seq_states_[seq_id].pos_max < 0) seq_states_[seq_id].pos_min = -1;
    }

    LLAMA_LOG_DEBUG("llama_kv_cache_paged::seq_rm: seq=%d range=[%d,%d) "
                    "blocks_freed=%u blocks_partial=%u pos_max=%d\n",
                    seq_id, p0, p1, blocks_freed, blocks_partial_skipped,
                    seq_states_[seq_id].pos_max);
    return true;
}

void llama_kv_cache_paged::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst,
                                    llama_pos p0, llama_pos p1) {
    check_thread_id_();
    // MAD-128: block-aligned CoW. Wholly-covered blocks of src in [p0, p1)
    // are SHARED into dst's table (refcount bumped in BlockPool). Future
    // writes that target a shared block trigger CoW in
    // cow_writes_for_ubatch (called from apply_ubatch_to_state).
    //
    // Sub-block partials are not shared (would require partial-block CoW
    // which doesn't exist) — they're skipped with a warning, mirroring
    // seq_rm's sub-block behavior.
    if (seq_id_src < 0 || (uint32_t) seq_id_src >= n_seq_max_) return;
    if (seq_id_dst < 0 || (uint32_t) seq_id_dst >= n_seq_max_) return;
    if (seq_id_src == seq_id_dst) return;
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<llama_pos>::max();
    if (p1 <= p0) return;

    const llama_pos src_max = seq_states_[seq_id_src].pos_max;
    if (p0 > src_max) return;  // src has nothing in range
    const llama_pos p1_clamped = std::min(p1, (llama_pos)(src_max + 1));

    // Wipe dst's existing range first so its old blocks are properly
    // refcount-released (recursive call uses the new partial seq_rm).
    seq_rm(seq_id_dst, p0, p1_clamped);

    const uint32_t bsize = block_size_;
    const uint32_t n_blocks_src = table_.num_blocks(seq_id_src);

    uint32_t blocks_shared          = 0;
    uint32_t blocks_partial_skipped = 0;

    for (uint32_t lblock = 0; lblock < n_blocks_src; ++lblock) {
        const llama_pos b_start = (llama_pos) lblock * (llama_pos) bsize;
        const llama_pos b_end   = b_start + (llama_pos) bsize;

        if (b_end <= p0 || b_start >= p1_clamped) continue;
        const bool wholly_covered = (p0 <= b_start) && (p1_clamped >= b_end);
        if (!wholly_covered) {
            ++blocks_partial_skipped;
            continue;
        }

        const uint32_t physical = table_.get_physical(seq_id_src, lblock);
        if (physical == mt::kInvalidBlockId) continue;  // hole in src

        // Bump refcount on src's physical and install in dst's table.
        pool_.bump_ref(physical);

        // Pad dst's table with kInvalidBlockId for any logical gaps
        // before this block, then either swap into an existing slot or
        // append.
        while (table_.num_blocks(seq_id_dst) < lblock) {
            table_.append_block(seq_id_dst, mt::kInvalidBlockId);
        }
        if (lblock < table_.num_blocks(seq_id_dst)) {
            table_.swap_block(seq_id_dst, lblock, physical);
        } else {
            table_.append_block(seq_id_dst, physical);
        }

        ++blocks_shared;
    }

    if (blocks_partial_skipped > 0) {
        LLAMA_LOG_WARN("llama_kv_cache_paged::seq_cp: range [%d,%d) is not "
                       "block-aligned (block_size=%u). %u block(s) shared via "
                       "CoW; %u block(s) partially overlapped — those are NOT "
                       "shared. Round caller's range to block boundaries.\n",
                       p0, p1, bsize, blocks_shared, blocks_partial_skipped);
    }

    // Update dst's pos_max if the copy extended its tail.
    if (p1_clamped - 1 > seq_states_[seq_id_dst].pos_max) {
        seq_states_[seq_id_dst].pos_max = p1_clamped - 1;
        if (seq_states_[seq_id_dst].pos_min < 0) seq_states_[seq_id_dst].pos_min = 0;
    }

    LLAMA_LOG_DEBUG("llama_kv_cache_paged::seq_cp: src=%d dst=%d range=[%d,%d) "
                    "blocks_shared=%u blocks_partial=%u\n",
                    seq_id_src, seq_id_dst, p0, p1,
                    blocks_shared, blocks_partial_skipped);
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

// MAD-130: cold-tier sidecar persistence.
//
// File format (CIDX v1) at `${cold_path_}/paged/index.bin`:
//   uint32  magic   = 0x58444943  ("CIDX")
//   uint32  version = 1
//   uint32  n_layers       (sanity check vs current cache)
//   uint32  n_cold_blocks  (sanity check vs current cache)
//   uint32  n_in_use
//   For each in-use entry:
//     int32  seq_id
//     uint32 lblock
//     uint32 cold_idx
//
// Save: called explicitly via save_cold_index_sidecar() (server
// shutdown handler or /slots/save). Atomic via write-to-tmp + rename.
//
// Load: called from ctor when cold_resume=true. On any error (missing
// file, bad magic, mismatched config), returns false and the caller
// proceeds with an empty cold index.

namespace {
constexpr uint32_t kColdIndexMagic   = 0x58444943;  // "CIDX"
constexpr uint32_t kColdIndexVersion = 1;
}

bool llama_kv_cache_paged::save_cold_index_sidecar() const {
    if (!cold_enabled() || cold_path_.empty()) return true;  // no-op

    const std::string subdir   = cold_path_ + "/paged/instance-" + instance_id_;
    const std::string sidecar  = subdir + "/index.bin";
    const std::string tmp_path = sidecar + ".tmp";

    std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
    if (!f) {
        LLAMA_LOG_WARN("llama_kv_cache_paged::save_cold_index_sidecar: open(%s) failed\n",
                       tmp_path.c_str());
        return false;
    }

    auto write_u32 = [&](uint32_t v) { f.write((const char *) &v, sizeof(v)); };
    auto write_i32 = [&](int32_t  v) { f.write((const char *) &v, sizeof(v)); };

    write_u32(kColdIndexMagic);
    write_u32(kColdIndexVersion);
    const uint32_t n_layers = (uint32_t) layers_.size();
    write_u32(n_layers);
    write_u32(n_cold_blocks_);
    write_u32(cold_in_use_);

    // Walk cold_slot_for_ and write each in-use (seq, lblock, cold_idx).
    uint32_t written = 0;
    for (uint32_t s = 0; s < (uint32_t) cold_slot_for_.size(); ++s) {
        const auto & row = cold_slot_for_[s];
        for (uint32_t lb = 0; lb < (uint32_t) row.size(); ++lb) {
            if (row[lb] == kInvalidColdIdx) continue;
            write_i32((int32_t) s);
            write_u32(lb);
            write_u32(row[lb]);
            ++written;
        }
    }
    f.close();
    if (written != cold_in_use_) {
        LLAMA_LOG_WARN("llama_kv_cache_paged::save_cold_index_sidecar: counted %u in-use entries "
                       "but cold_in_use_ tracker says %u — sidecar may be inconsistent\n",
                       written, cold_in_use_);
    }
    if (::rename(tmp_path.c_str(), sidecar.c_str()) != 0) {
        LLAMA_LOG_WARN("llama_kv_cache_paged::save_cold_index_sidecar: rename %s -> %s failed\n",
                       tmp_path.c_str(), sidecar.c_str());
        return false;
    }
    LLAMA_LOG_INFO("llama_kv_cache_paged::save_cold_index_sidecar: wrote %u entries to %s\n",
                   written, sidecar.c_str());
    return true;
}

bool llama_kv_cache_paged::load_cold_index_sidecar_(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    auto read_u32 = [&](uint32_t & v) -> bool {
        f.read((char *) &v, sizeof(v));
        return (bool) f;
    };
    auto read_i32 = [&](int32_t & v) -> bool {
        f.read((char *) &v, sizeof(v));
        return (bool) f;
    };

    uint32_t magic = 0, version = 0, saved_n_layers = 0, saved_n_cold = 0, saved_n_in_use = 0;
    if (!read_u32(magic) || !read_u32(version)) return false;
    if (magic != kColdIndexMagic || version != kColdIndexVersion) {
        LLAMA_LOG_WARN("load_cold_index_sidecar: bad header (magic=%08x ver=%u)\n", magic, version);
        return false;
    }
    if (!read_u32(saved_n_layers) || !read_u32(saved_n_cold) || !read_u32(saved_n_in_use)) {
        return false;
    }
    if (saved_n_layers != (uint32_t) layers_.size() || saved_n_cold != n_cold_blocks_) {
        LLAMA_LOG_WARN("load_cold_index_sidecar: config mismatch (saved_n_layers=%u current=%zu, "
                       "saved_n_cold=%u current=%u) — rejecting sidecar\n",
                       saved_n_layers, layers_.size(), saved_n_cold, n_cold_blocks_);
        return false;
    }

    // Walk in-use entries. Mark each cold slot in cold_slot_for_, and
    // rebuild cold_pool_free_ minus the in-use slots.
    std::vector<bool> in_use(n_cold_blocks_, false);
    for (uint32_t i = 0; i < saved_n_in_use; ++i) {
        int32_t  sid = 0;
        uint32_t lb = 0, cold_idx = 0;
        if (!read_i32(sid) || !read_u32(lb) || !read_u32(cold_idx)) return false;
        if (sid < 0 || (uint32_t) sid >= n_seq_max_) continue;
        if (cold_idx >= n_cold_blocks_) continue;
        mark_cold(cold_slot_for_, (llama_seq_id) sid, lb, cold_idx);
        in_use[cold_idx] = true;
    }

    // Rebuild free stack — only push slots NOT in use.
    cold_pool_free_.clear();
    cold_pool_free_.reserve(n_cold_blocks_);
    for (uint32_t i = n_cold_blocks_; i > 0; --i) {
        if (!in_use[i - 1]) cold_pool_free_.push_back(i - 1);
    }
    cold_in_use_ = saved_n_in_use;

    LLAMA_LOG_INFO("llama_kv_cache_paged::load_cold_index_sidecar: loaded %u in-use cold "
                   "entries (free pool: %zu)\n",
                   saved_n_in_use, cold_pool_free_.size());
    return true;
}

// MAD-130: state persistence for the paged cache.
//
// On-disk format (PAGS v1):
//
//   HEADER (76 bytes):
//     magic            uint32  = 0x53474150 ("PAGS" little-endian)
//     version          uint32  = 1
//     n_seq_max        uint32  (config: must match on read)
//     block_size       uint32  (config: must match on read)
//     n_blocks_total   uint32  (config: GPU pool size; must match on read)
//     n_warm_blocks    uint32  (config: must match on read)
//     n_cold_blocks    uint32  (config: must match on read)
//     n_layers         uint32  (config: total, including filtered-out)
//     k_bytes_per_block uint64 (config sanity)
//     v_bytes_per_block uint64 (config sanity)
//     type_k           uint32  (ggml_type)
//     type_v           uint32  (ggml_type)
//     flags            uint32  (reserved)
//
//   PER-SEQ SECTION (one per saved seq):
//     seq_present      uint8  (0 = end of seq list; 1 = entry follows)
//     seq_id           int32
//     pos_min          int32
//     pos_max          int32
//     num_blocks       uint32
//     For each lblock in [0, num_blocks):
//       tier_origin    uint8  (0=hole, 1=hot, 2=warm, 3=cold)
//       if hot or warm:
//         For each restorable layer (filter applied at construction):
//           K bytes (k_bytes_per_block)
//           V bytes (v_bytes_per_block)
//       if cold:
//         cold_idx     uint32  (slot in the per-layer .bin files)
//
//   FINGERPRINT SECTION:
//     n_fingerprints   uint32
//     For each: seq_id (int32), lblock (uint32), tier (uint8),
//               embedding_dim (uint32), floats[embedding_dim]
//
// Validation: header config fields must EXACTLY match the runtime
// cache (n_blocks_total, block_size, type_k, type_v, n_layers,
// k/v_bytes_per_block). Mismatch → throw runtime_error with a clear
// message; caller's /slots/restore returns the error to the client.

namespace {
constexpr uint32_t kPagedStateMagic   = 0x53474150;  // "PAGS"
constexpr uint32_t kPagedStateVersion = 1;

enum class PagedTierTag : uint8_t {
    Hole = 0,
    Hot  = 1,
    Warm = 2,
    Cold = 3,
};
}  // namespace

void llama_kv_cache_paged::state_write(llama_io_write_i & io, llama_seq_id seq_id,
                                          llama_state_seq_flags /*flags*/) const {
    // ── Header ──
    io.write(&kPagedStateMagic,   sizeof(kPagedStateMagic));
    io.write(&kPagedStateVersion, sizeof(kPagedStateVersion));
    io.write(&n_seq_max_,         sizeof(n_seq_max_));
    io.write(&block_size_,        sizeof(block_size_));
    io.write(&n_blocks_total_,    sizeof(n_blocks_total_));
    io.write(&n_warm_blocks_,     sizeof(n_warm_blocks_));
    io.write(&n_cold_blocks_,     sizeof(n_cold_blocks_));
    const uint32_t n_layers = (uint32_t) layers_.size();
    io.write(&n_layers,           sizeof(n_layers));
    const uint64_t k_bpb = (uint64_t) k_bytes_per_block_;
    const uint64_t v_bpb = (uint64_t) v_bytes_per_block_;
    io.write(&k_bpb,              sizeof(k_bpb));
    io.write(&v_bpb,              sizeof(v_bpb));
    const uint32_t type_k_u = (uint32_t) type_k_;
    const uint32_t type_v_u = (uint32_t) type_v_;
    io.write(&type_k_u,           sizeof(type_k_u));
    io.write(&type_v_u,           sizeof(type_v_u));
    const uint32_t flags_reserved = 0;
    io.write(&flags_reserved,     sizeof(flags_reserved));

    // ── Per-seq section ──
    auto write_one_seq = [&](llama_seq_id sid) {
        if (sid < 0 || (uint32_t) sid >= n_seq_max_) return;
        const uint32_t n_blocks_seq = table_.num_blocks(sid);
        // Skip empty seqs (no blocks).
        if (n_blocks_seq == 0) return;

        const uint8_t present = 1;
        io.write(&present, sizeof(present));
        const int32_t sid_i32 = (int32_t) sid;
        io.write(&sid_i32, sizeof(sid_i32));
        const int32_t pos_min = (int32_t) seq_states_[sid].pos_min;
        const int32_t pos_max = (int32_t) seq_states_[sid].pos_max;
        io.write(&pos_min, sizeof(pos_min));
        io.write(&pos_max, sizeof(pos_max));
        io.write(&n_blocks_seq, sizeof(n_blocks_seq));

        std::vector<uint8_t> kbuf(k_bytes_per_block_);
        std::vector<uint8_t> vbuf(v_bytes_per_block_);

        for (uint32_t lb = 0; lb < n_blocks_seq; ++lb) {
            const uint32_t physical = table_.get_physical(sid, lb);
            PagedTierTag tag = PagedTierTag::Hole;
            if (physical != mt::kInvalidBlockId) {
                tag = pool_.is_gpu(physical) ? PagedTierTag::Hot : PagedTierTag::Warm;
            } else if (cold_enabled() && cold_slot(cold_slot_for_, sid, lb) != kInvalidColdIdx) {
                tag = PagedTierTag::Cold;
            }
            const uint8_t tag_u8 = (uint8_t) tag;
            io.write(&tag_u8, sizeof(tag_u8));

            if (tag == PagedTierTag::Hot) {
                const size_t k_off_gpu = (size_t) physical * k_bytes_per_block_;
                const size_t v_off_gpu = (size_t) physical * v_bytes_per_block_;
                for (uint32_t il = 0; il < layers_.size(); ++il) {
                    const auto & layer = layers_[il];
                    if (!layer.k) continue;
                    ggml_backend_tensor_get(layer.k, kbuf.data(), k_off_gpu, k_bytes_per_block_);
                    ggml_backend_tensor_get(layer.v, vbuf.data(), v_off_gpu, v_bytes_per_block_);
                    io.write(kbuf.data(), k_bytes_per_block_);
                    io.write(vbuf.data(), v_bytes_per_block_);
                }
            } else if (tag == PagedTierTag::Warm) {
                const uint32_t cpu_idx = physical - n_blocks_total_;
                const size_t k_off_cpu = (size_t) cpu_idx * k_bytes_per_block_;
                const size_t v_off_cpu = (size_t) cpu_idx * v_bytes_per_block_;
                for (uint32_t il = 0; il < layers_.size(); ++il) {
                    if (!layers_[il].k) continue;
                    io.write(warm_k_[il].data() + k_off_cpu, k_bytes_per_block_);
                    io.write(warm_v_[il].data() + v_off_cpu, v_bytes_per_block_);
                }
            } else if (tag == PagedTierTag::Cold) {
                const uint32_t cold_idx = cold_slot(cold_slot_for_, sid, lb);
                io.write(&cold_idx, sizeof(cold_idx));
            }
            // Hole: no extra bytes.
        }
    };

    if (seq_id < 0) {
        for (uint32_t s = 0; s < n_seq_max_; ++s) write_one_seq((llama_seq_id) s);
    } else {
        write_one_seq(seq_id);
    }
    const uint8_t end_marker = 0;
    io.write(&end_marker, sizeof(end_marker));

    // ── Fingerprint section ──
    // BlockSemanticIndex doesn't expose iteration; for now we only
    // serialize fingerprint COUNT here, with the actual data left to
    // BlockSemanticIndex::save_to_disk (separate sidecar file). The
    // sidecar approach matches the legacy SemanticIndex pattern and
    // keeps state_write self-contained.
    const uint32_t n_fingerprints_total = (uint32_t) paged_semantic_.size();
    io.write(&n_fingerprints_total, sizeof(n_fingerprints_total));
    // Fingerprints themselves are written via paged_semantic.save_to_disk,
    // called by the server's /slots/save handler alongside this state_write.

    LLAMA_LOG_INFO("llama_kv_cache_paged::state_write: wrote %zu bytes "
                   "(seq_id=%d, n_fingerprints=%u)\n",
                   io.n_bytes(), seq_id, n_fingerprints_total);
}

void llama_kv_cache_paged::state_read(llama_io_read_i & io, llama_seq_id seq_id,
                                         llama_state_seq_flags /*flags*/) {
    check_thread_id_();
    // ── Header validation ──
    uint32_t magic, version;
    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));
    if (magic != kPagedStateMagic) {
        throw std::runtime_error("llama_kv_cache_paged::state_read: bad magic — not a PAGS state file");
    }
    if (version != kPagedStateVersion) {
        throw std::runtime_error("llama_kv_cache_paged::state_read: unsupported version " +
                                 std::to_string(version));
    }

    uint32_t saved_n_seq_max, saved_block_size, saved_n_blocks_total;
    uint32_t saved_n_warm, saved_n_cold, saved_n_layers;
    uint64_t saved_k_bpb, saved_v_bpb;
    uint32_t saved_type_k, saved_type_v, saved_flags;
    io.read(&saved_n_seq_max,      sizeof(saved_n_seq_max));
    io.read(&saved_block_size,     sizeof(saved_block_size));
    io.read(&saved_n_blocks_total, sizeof(saved_n_blocks_total));
    io.read(&saved_n_warm,         sizeof(saved_n_warm));
    io.read(&saved_n_cold,         sizeof(saved_n_cold));
    io.read(&saved_n_layers,       sizeof(saved_n_layers));
    io.read(&saved_k_bpb,          sizeof(saved_k_bpb));
    io.read(&saved_v_bpb,          sizeof(saved_v_bpb));
    io.read(&saved_type_k,         sizeof(saved_type_k));
    io.read(&saved_type_v,         sizeof(saved_type_v));
    io.read(&saved_flags,          sizeof(saved_flags));

    auto require = [](bool cond, const char * what, auto saved, auto current) {
        if (!cond) {
            std::string msg = std::string("llama_kv_cache_paged::state_read: ") + what +
                              " mismatch — saved=" + std::to_string(saved) +
                              " current=" + std::to_string(current) +
                              " (state file from a different cache config)";
            throw std::runtime_error(msg);
        }
    };
    require(saved_n_seq_max == n_seq_max_,           "n_seq_max",      saved_n_seq_max,      n_seq_max_);
    require(saved_block_size == block_size_,         "block_size",     saved_block_size,     block_size_);
    require(saved_n_blocks_total == n_blocks_total_, "n_blocks_total", saved_n_blocks_total, n_blocks_total_);
    require(saved_n_warm == n_warm_blocks_,          "n_warm_blocks",  saved_n_warm,         n_warm_blocks_);
    require(saved_n_cold == n_cold_blocks_,          "n_cold_blocks",  saved_n_cold,         n_cold_blocks_);
    require(saved_n_layers == (uint32_t) layers_.size(), "n_layers",   saved_n_layers,       (uint32_t) layers_.size());
    require(saved_k_bpb == (uint64_t) k_bytes_per_block_, "k_bytes_per_block", saved_k_bpb, (uint64_t) k_bytes_per_block_);
    require(saved_v_bpb == (uint64_t) v_bytes_per_block_, "v_bytes_per_block", saved_v_bpb, (uint64_t) v_bytes_per_block_);
    require(saved_type_k == (uint32_t) type_k_, "type_k", saved_type_k, (uint32_t) type_k_);
    require(saved_type_v == (uint32_t) type_v_, "type_v", saved_type_v, (uint32_t) type_v_);

    std::vector<uint8_t> kbuf(k_bytes_per_block_);
    std::vector<uint8_t> vbuf(v_bytes_per_block_);

    // ── Per-seq sections ──
    uint32_t n_seqs_loaded = 0;
    uint32_t n_blocks_loaded = 0;
    while (true) {
        uint8_t present = 0;
        io.read(&present, sizeof(present));
        if (present == 0) break;

        int32_t saved_sid = 0;
        io.read(&saved_sid, sizeof(saved_sid));
        // If caller asked for a specific seq_id, remap saved data into
        // that slot (allows /slots/restore into a different slot ID).
        const llama_seq_id load_sid = (seq_id < 0) ? (llama_seq_id) saved_sid : seq_id;
        if (load_sid < 0 || (uint32_t) load_sid >= n_seq_max_) {
            throw std::runtime_error("llama_kv_cache_paged::state_read: seq_id out of range");
        }

        // Wipe the existing seq before reading. Honors refcount via
        // free_block (shared blocks decrement, free at 0).
        seq_rm(load_sid, 0, std::numeric_limits<llama_pos>::max());

        int32_t pos_min = 0, pos_max = 0;
        uint32_t n_blocks_seq = 0;
        io.read(&pos_min,      sizeof(pos_min));
        io.read(&pos_max,      sizeof(pos_max));
        io.read(&n_blocks_seq, sizeof(n_blocks_seq));

        for (uint32_t lb = 0; lb < n_blocks_seq; ++lb) {
            uint8_t tag_u8 = 0;
            io.read(&tag_u8, sizeof(tag_u8));
            const PagedTierTag tag = (PagedTierTag) tag_u8;

            if (tag == PagedTierTag::Hole) {
                table_.append_block(load_sid, mt::kInvalidBlockId);
                continue;
            }

            if (tag == PagedTierTag::Hot) {
                const uint32_t phys = pool_.alloc_gpu();
                if (phys == mt::kInvalidBlockId) {
                    throw std::runtime_error("llama_kv_cache_paged::state_read: GPU pool exhausted "
                                             "while restoring hot block");
                }
                const size_t k_off_gpu = (size_t) phys * k_bytes_per_block_;
                const size_t v_off_gpu = (size_t) phys * v_bytes_per_block_;
                for (uint32_t il = 0; il < layers_.size(); ++il) {
                    if (!layers_[il].k) continue;
                    io.read(kbuf.data(), k_bytes_per_block_);
                    io.read(vbuf.data(), v_bytes_per_block_);
                    ggml_backend_tensor_set(layers_[il].k, kbuf.data(), k_off_gpu, k_bytes_per_block_);
                    ggml_backend_tensor_set(layers_[il].v, vbuf.data(), v_off_gpu, v_bytes_per_block_);
                }
                table_.append_block(load_sid, phys);
            } else if (tag == PagedTierTag::Warm) {
                if (!warm_enabled()) {
                    throw std::runtime_error("llama_kv_cache_paged::state_read: warm tier disabled but "
                                             "saved state has warm blocks");
                }
                const uint32_t phys = pool_.alloc_cpu();
                if (phys == mt::kInvalidBlockId) {
                    throw std::runtime_error("llama_kv_cache_paged::state_read: CPU pool exhausted "
                                             "while restoring warm block");
                }
                const uint32_t cpu_idx = phys - n_blocks_total_;
                const size_t k_off_cpu = (size_t) cpu_idx * k_bytes_per_block_;
                const size_t v_off_cpu = (size_t) cpu_idx * v_bytes_per_block_;
                for (uint32_t il = 0; il < layers_.size(); ++il) {
                    if (!layers_[il].k) continue;
                    io.read(warm_k_[il].data() + k_off_cpu, k_bytes_per_block_);
                    io.read(warm_v_[il].data() + v_off_cpu, v_bytes_per_block_);
                }
                table_.append_block(load_sid, phys);
            } else if (tag == PagedTierTag::Cold) {
                if (!cold_enabled()) {
                    throw std::runtime_error("llama_kv_cache_paged::state_read: cold tier disabled but "
                                             "saved state has cold blocks");
                }
                uint32_t cold_idx = 0;
                io.read(&cold_idx, sizeof(cold_idx));
                if (cold_idx >= n_cold_blocks_) {
                    throw std::runtime_error("llama_kv_cache_paged::state_read: cold_idx out of range");
                }
                // Mark as cold-resident; the actual bytes live in the
                // cold-tier files (which require --kv-tier-cold-resume
                // to survive a restart — separate part of MAD-130).
                mark_cold(cold_slot_for_, load_sid, lb, cold_idx);
                table_.append_block(load_sid, mt::kInvalidBlockId);
                ++cold_in_use_;
            }
            ++n_blocks_loaded;
        }

        seq_states_[load_sid].pos_min = pos_min;
        seq_states_[load_sid].pos_max = pos_max;
        ++n_seqs_loaded;
    }

    // ── Fingerprint count (data lives in sidecar) ──
    uint32_t n_fingerprints_meta = 0;
    io.read(&n_fingerprints_meta, sizeof(n_fingerprints_meta));

    LLAMA_LOG_INFO("llama_kv_cache_paged::state_read: loaded %u seq(s), %u block(s); "
                   "fingerprint sidecar reports %u entries (load via "
                   "BlockSemanticIndex::load_from_disk).\n",
                   n_seqs_loaded, n_blocks_loaded, n_fingerprints_meta);
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
