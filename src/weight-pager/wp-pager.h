#pragma once

// WeightPager — facade composing PageCatalog + FileIOLayer + PoolAllocator
// + GpuTransport + PrefetchScheduler into a single subsystem the rest of
// llama.cpp talks to.
//
// Lifetime:
//   1. Construct (cheap).
//   2. add_page() N times during model load (typically from the model
//      loader integration in Phase 1d).
//   3. init() once after the catalog is built. Pool, transport, prefetch
//      scheduler all come up here. GGML_CUDA_DISABLE_GRAPHS is snapshotted
//      and forced to "1" unless WP_HIP_GRAPHS=1 (ggml's hipGraph capture
//      bakes tensor->data pointers; the eval callback's per-step rewrites
//      need MAD-P1 graph update handling). The original env-var state is
//      restored on shutdown - fixes B-P5.
//   4. ensure() / prefetch_next() / tick() during inference, called from
//      the eval callback adapter.
//   5. shutdown() (or destructor) tears everything down in reverse order.

#include "wp-page-catalog.h"
#include "wp-file-io.h"
#include "wp-host-tier.h"
#include "wp-pool.h"
#include "wp-gpu-transport.h"
#include "wp-prefetch.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Forward-declared unconditionally: WeightPager holds an io_uring* member in
// every build so its layout does not depend on LLAMA_HAVE_IO_URING (see below).
struct io_uring;

struct ggml_backend_buffer;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;

struct ggml_backend_buffer_type;
typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;

struct ggml_cgraph;
struct ggml_tensor;

namespace wp {

class WeightPager {
public:
    struct Config {
        int  n_slots         = 0;     // size of the VRAM ring; -1 / 0 = auto (one per layer)
        int  prefetch_depth  = 4;     // PrefetchScheduler queue depth
        int  io_uring_depth  = 0;     // FileIOLayer SQ depth; 0 = prefetch_depth
        bool prefer_async_io = true;  // try io_uring for stage 1 before SyncPread
    };

    struct Stats {
        uint64_t page_ins                       = 0;
        uint64_t evictions                      = 0;
        uint64_t prefetch_hits                  = 0;
        uint64_t prefetch_misses                = 0;
        uint64_t sync_fallbacks                 = 0;
        uint64_t io_bytes                       = 0;
        double   io_seconds                     = 0.0;
        uint64_t lru_walk_hot_skips             = 0;
        uint64_t lru_walk_pinned_skips          = 0;
        uint64_t dense_prefetch_submitted       = 0;
        uint64_t cross_layer_prefetch_submitted = 0;
        uint64_t cross_layer_hit_in_ensure      = 0;
        uint64_t host_tier_hits                 = 0;
        uint64_t routing_ptrs_set                  = 0;
        uint64_t routing_ptrs_consumed             = 0;
        uint64_t routing_ptrs_discarded_unconsumed = 0;
        // Draft-as-paging-oracle (speculative draft model -> expert cache).
        uint64_t draft_prefetch_calls              = 0;
        uint64_t draft_prefetch_pages_submitted    = 0;
        uint64_t draft_prefetch_pages_resident     = 0; // already loaded at draft fire
        uint64_t draft_retain_pins                 = 0; // pages pinned for draft window
        uint64_t draft_tid2eid_pages               = 0; // pages from hash-layer token map
        uint64_t draft_cold_pages                  = 0; // union cold count at draft fire
        uint64_t draft_tid2eid_cold                = 0; // tid2eid pages that needed I/O
        uint64_t draft_tid2eid_hits_in_ensure      = 0; // ensure of last draft tid2eid page
        uint64_t draft_oracle_skips                = 0; // adaptive: skipped DFlash fire
        uint64_t draft_window_opens                = 0; // set_draft_window(n>0)
        uint64_t draft_window_closes               = 0; // set_draft_window(0)
        uint64_t draft_prefetch_queue_blocked      = 0; // waves aborted: free_q==0 after drain
        uint64_t draft_prefetch_harvested          = 0; // Done prefetches committed before submit
        uint64_t draft_hot_records                 = 0; // MMID history snaps (metrics only)
        // Oracle precision (hash-layer tid2eid pages only).
        // pred = pages marked at last oracle fire; actual = record_active pages
        // in tid2eid blocks. tp = actual ∩ pred; fn = actual \ pred; fp finalized
        // at next fire as pred never seen in actual.
        uint64_t oracle_sample_fires               = 0; // llama_wp_on_sampled_token
        uint64_t oracle_draft_fires                = 0; // draft-token fires (cold submit path)
        uint64_t oracle_pred_pages                 = 0; // sum |predicted set| at fires
        uint64_t oracle_actual_hash_pages          = 0; // hash-layer pages in routing snaps
        uint64_t oracle_tp                         = 0; // predicted and later routed
        uint64_t oracle_fn                         = 0; // routed hash page not predicted
        uint64_t oracle_fp                         = 0; // predicted never routed before next fire
        uint64_t oracle_pages_submitted            = 0; // cold hash pages queued by sample path
        uint64_t oracle_pages_free_slot            = 0; // of those, took free pool slots
        uint64_t oracle_pages_evict_slot           = 0; // of those, recycled stale-hash / MoE LRU
        uint64_t oracle_hash_slots_freed           = 0; // stale hash pages released for oracle
        uint64_t oracle_protect_pins               = 0; // temp MoE-history pins during MoE LRU
        uint64_t oracle_sticky_pins                = 0; // sample-oracle sticky retain installs
        uint64_t sticky_l2_pins                    = 0; // current sticky L2 pin refs
        uint64_t sticky_l2_hits_in_ensure          = 0; // ensure of sticky L2 page
        uint64_t sticky_l2_promotions              = 0; // pages added to sticky L2 set
        uint64_t sticky_l2_demotions               = 0; // pages removed from sticky L2 set
        uint64_t sticky_spec_fires                 = 0; // FA-window sticky/hot prefetch calls
        uint64_t sticky_spec_pages_submitted       = 0; // cold pages submitted in those fires
        uint64_t sticky_spec_pages_resident        = 0; // already resident at fire
        // ensure_batch multi-QD bursts (P2P path)
        uint64_t ensure_batch_calls                = 0;
        uint64_t ensure_batch_pages                = 0; // cold misses issued in batches
        uint64_t ensure_batch_max_n                = 0; // largest concurrent miss set
        uint64_t ensure_batch_bytes                = 0;
        double   ensure_batch_seconds              = 0.0; // submit+wait wall
        double   ensure_batch_submit_seconds       = 0.0;
        double   ensure_batch_wait_seconds         = 0.0;
        uint64_t ensure_batch_timeouts             = 0; // wait_for_req returned non-Ok
        uint64_t ensure_batch_n_sub_sum            = 0; // sum of submit_batch return
    };

    WeightPager() = default;
    ~WeightPager();

    WeightPager(const WeightPager &)             = delete;
    WeightPager & operator=(const WeightPager &) = delete;

    // MAD-236 — register a tensor whose bytes already live in caller-owned
    // VRAM (e.g. token_embd from the regular model loader buffer). The
    // pager doesn't allocate a pool slot or read from disk for these —
    // ensure(page_idx) returns `device_ptr` directly. Useful for unified
    // VRAM telemetry (paged + resident in one view) and for mixed-mode
    // workloads where some weights stay pinned and others page.
    //
    // Must be called BEFORE init(), like add_page(). Returns the page index.
    int register_pinned(const std::string & name, void * device_ptr, size_t bytes);

    // Catalog population. Must be called before init().
    //
    // n_experts > 1 marks a consolidated MoE expert tensor (Qwen3-MoE
    // style: blk.<N>.ffn_<role>_exps.weight packs all experts as a 3D
    // tensor with ne[2] == n_experts). The catalog will register one
    // sub-page per expert with synthetic names of the form
    // "<base_name>#expert.<E>", letting the pager and prefetch scheduler
    // page individual experts rather than the full consolidated tensor.
    // (MAD-88 Phase 2.) Use n_experts = 1 (default) for non-MoE tensors
    // and Mixtral-style per-expert tensors, which are already separate
    // pages by name.
    int add_page(const std::string & name,
                 uint16_t            file_idx,
                 uint64_t            file_offset,
                 size_t              size,
                 int                 n_experts = 1,
                 ggml_backend_buffer_type_t buft = nullptr);

    // Initialise the pool, transport, file-io layer, and prefetch scheduler.
    //
    // `fds` are pre-prepared file descriptors (one per split GGUF file).
    // The caller should obtain them via dup_clear_o_direct() to fix B-P3.
    // The pager takes ownership of the fds; they are closed on shutdown.
    //
    // Paging is single-device by contract: under WP_RESIDENT_DENSE only routed
    // experts page, and they all live on the paging device, so one pool keyed
    // by that device's buffer type is sufficient. Dense/attention weights are
    // resident on the (possibly different) attention device and never page.
    //
    // `device_buft` is the buffer type for the device the pager will
    // allocate VRAM on. Should match devices_used.front().
    bool init(const Config &             cfg,
              ggml_backend_buffer_type_t device_buft,
              int                        device_idx,
              std::vector<int>           fds,
              const std::vector<int> &   devices_used);

    // Tear down in reverse order. Restores GGML_CUDA_DISABLE_GRAPHS to its
    // pre-init value if init forced it. Safe to call multiple times.
    void shutdown();

    // Lookup helpers.
    int    find_page(const std::string & name) const { return catalog_.find(name); }
    // Allocation-free overload for the eval-cb hot path: ggml tensor names
    // arrive as const char* (t->name), and building a std::string temporary
    // per lookup heap-allocs for names > SSO (e.g. "blk.23.ffn_gate.weight").
    // eval_cb calls find_page ~thousands of times per decoded token, so that
    // alloc/free churn dominated decode (WP_PROFILE_EVAL). A thread_local
    // scratch string reuses its buffer via assign() → no alloc after warmup.
    // Overload resolution routes char* callers here automatically.
    int    find_page(const char * name) const {
        thread_local std::string key;
        key.assign(name);
        return catalog_.find(key);
    }
    int    n_pages()                            const { return catalog_.size(); }
    int    catalog_n_expert_pages()             const { return catalog_.n_expert_pages(); }
    size_t max_page_size()                      const { return catalog_.max_page_size(); }
    bool   is_initialized()                     const { return initialized_; }
    bool   hip_graphs_enabled()                 const { return hip_graphs_enabled_; }
    bool   async_ensure_enabled()               const { return async_ensure_enabled_; }
    const Stats & stats() const;
    bool   batch_safe() const;
    void   mark_routing_boundaries(const struct ggml_cgraph * gf);
    bool   is_routing_break(const struct ggml_tensor * t) const {
        return routing_break_tensors_.count(t) != 0;
    }
    // VRAM arena size in bytes — used by the WP_PAGED_BATCH reactive auto-break
    // to bound a batch range's pinned working set below what fits in the pool.
    size_t pool_arena_bytes() const { return pool_.pool_size(); }
    uint64_t sync_fallback_count() const { return stats_.sync_fallbacks; }
    int    loaded_pages() const;
    int    pending_prefetches() const { return prefetch_.pending(); }
    bool   async_prefetch_enabled() const { return cfg_.prefer_async_io; }

    // Ensure a page is in VRAM, returning the slot pointer. Synchronous
    // fallback if the page is not (yet) prefetched. Returns nullptr if
    // page_idx is out of range or any underlying op fails.
    void * ensure(int page_idx);

    // Batch-ensure a set of pages with all cold-miss reads issued CONCURRENTLY
    // (Colibri pattern). Each miss's slot is reserved AND pinned up front so no
    // read in the batch can evict a sibling's not-yet-loaded slot — the eviction
    // window that collapsed effective io_uring queue depth to ~1 under decode
    // pressure. On the P2P (direct-to-device) IO path the misses are submitted in
    // one io_uring batch (true QD=N) reading straight into the VRAM slots.
    // Fills out_ptrs[i] with the slot pointer for pages[i] (nullptr on failure /
    // pool exhaustion), and out_pinned with every page this call pinned — the
    // CALLER must record those and unpin them in the next eval callback (matches
    // the per-op pin lifecycle used by ensure()+pin_page).
    void ensure_batch(const std::vector<int> & page_indices,
                      std::vector<void *>     & out_ptrs,
                      std::vector<int>        & out_pinned);

    // WP_ASYNC_ENSURE handoff. ensure() stashes the transfer event here when
    // it returns before stage 2 has completed; the eval callback takes it,
    // queues a compute-stream wait, and releases it after that op completes.
    int  take_async_transfer_event(int page_idx);
    bool enqueue_async_transfer_wait(int event_handle, void * stream);
    bool synchronize_async_transfer_event(int event_handle);
    void finish_async_transfer_event(int page_idx, int event_handle);
    void release_async_transfer_event(int event_handle);

    // Submit a prefetch hint for a page. No-op if the page is already in
    // flight or already loaded. Errors are logged but do not propagate —
    // the eval callback's ensure() will fall back to sync on miss.
    // allow_evict=false: free pool slots only (sample oracle; no thrash).
    bool prefetch_page(int page_idx, bool count_dense_prefetch = false,
                       bool allow_evict = true);

    // MAD-235 — batch-prefetch N pages atomically. Reserves N pool slots
    // up-front, builds the file-IO batch, issues one batched submit. If
    // any step fails (pool can't supply N slots, scheduler rejects the
    // batch), returns false and leaves no partial state. Caller falls
    // back to per-page prefetch_page() on false.
    //
    // Skips page indices that are already resident or already in flight
    // — those are no-ops, not failures. Returns true iff every NEEDED
    // request was queued (or no requests were needed).
    // allow_evict=false: free pool slots only (sample oracle).
    bool prefetch_pages_batch(const std::vector<int> & page_indices,
                              bool count_dense_prefetch = false,
                              bool allow_evict = true);

    // Hint the kernel (via POSIX_FADV_WILLNEED on the file_io layer) that
    // we will soon need every paged tensor in layers [block_idx+1,
    // block_idx+k]. Cheap (one syscall per range), idempotent at kernel
    // level. Subsequent reads against these ranges hit warm page cache
    // instead of cold NVMe.
    //
    // No-op when not initialized, when k <= 0, or when block_idx < 0.
    //
    // The eval callback invokes this once per layer boundary in the forward
    // pass. RAM cost: each advised range is `size` bytes of page cache
    // pressure; on RAM-tight systems keep k low or disable via
    // WP_FADVISE_LOOKAHEAD=0.
    void advise_layer_lookahead(int block_idx, int k);

    // MAD-233 aggregate instrumentation. The eval callback marks candidate
    // pages before issuing cross-layer prefetches; successful scheduler
    // submissions and later ensure-time hits are folded into Stats.
    void mark_cross_layer_prefetch_candidates(const std::vector<int> & page_indices);

    // --- Draft-as-paging-oracle -------------------------------------------
    // Hash-layer tid2eid(token) is the hard signal (DS4 layers 0..H). Softmax
    // MMID history is metrics-only (low cross-token locality measured).
    // set_draft_window / prefetch_hot_experts pin tid2eid pages across the
    // draft->target gap. draft_oracle_should_run() gates running the draft
    // model at all when the pool is already warm (adaptive skip).
    void record_active_expert_pages(const std::vector<int> & page_indices);
    // Host copy of blk.N.ffn_gate_tid2eid (I32, layout [n_vocab][n_expert_used]).
    // Call once per hash layer after model tensors are loaded.
    void register_tid2eid_host(int block_idx, int n_expert_used, int n_vocab,
                               const int32_t * host_data);
    // source: 0 = sample (ground truth next input), 1 = draft (speculative)
    int  prefetch_hot_experts(const int32_t * tokens = nullptr, int n_tokens = 0,
                              int source = 1);
    // After target samples token T, record T for tid2eid and (default) flush
    // cold I/O immediately: free slots first, then capped protected LRU
    // (WP_SAMPLE_ORACLE_MAX=16, WP_SAMPLE_ORACLE_EVICT=0 free-only).
    // WP_SAMPLE_ORACLE_EAGER=0 defers submit to layer-0 FA only.
    int  note_sampled_token(int32_t token);
    int  flush_sample_oracle_at_fa();
    // Drop sample-oracle sticky pins once decode is past hash layers so MoE
    // regains full pool capacity (hash pages stay LRU-hot from ensure hits).
    void release_sample_sticky_if_past_hash(int block_idx);
    // Speculative expert page-in for the FA/dense window: top sticky scores +
    // recent target-routing history. Call from eval_cb on FLASH_ATTN so R9700
    // NVMe work can run under eGPU attention. Returns cold pages submitted.
    int  prefetch_sticky_hot_experts();
    void set_draft_window(int n_draft);
    bool draft_window_active() const { return draft_window_ > 0; }
    int  draft_window() const { return draft_window_; }
    // Adaptive: false when last fires found no cold tid2eid pages (warm pool).
    // WP_DRAFT_ADAPTIVE=0 forces always true. Counts as skip when false.
    bool draft_oracle_should_run();

    // Drive the prefetch pipeline forward. Idempotent and non-blocking.
    void tick();

    // Pin / unpin the slot currently backing `page_idx` so eviction can't
    // reclaim it while an in-flight op references the VRAM. Refcounted via
    // PoolAllocator::pin_slot — safe under overlapping ops that touch the
    // same page. No-op if page_idx is out of range or the page is not
    // currently resident (no slot to pin).
    //
    // Lifecycle in the eval-callback (MAD-231): call pin_page on each page
    // the op references after ensure() succeeds; call unpin_page in the
    // NEXT eval-callback before submitting any new prefetches. This window
    // is a superset of the actual GPU-side read but stream ordering on the
    // compute stream guarantees correctness for the conservative bound.
    void pin_page(int page_idx);
    void unpin_page(int page_idx);

    // Backing buffer for the pool — used by the eval-cb adapter when
    // patching tensor->buffer (B-P4 requires a valid ggml backend buffer).
    ggml_backend_buffer_t pool_buf() const { return pool_.vram_buf(); }
    ggml_backend_buffer_t pool_buf(int page_idx) const;

    // Slot-and-page metadata (read-only public view).
    const PageMeta & page_meta(int page_idx) const { return catalog_.at(page_idx); }

    // Where a page currently lives in the pool, or -1 if not loaded.
    int slot_for_page(int page_idx) const;

    // MAD-P1: graph-lifetime pins for slots captured by CUDA/HIP graphs.
    // Replaces the pin set for graph_key with the slots currently backing
    // page_indices, unpinning the old exact slots first. No-op unless
    // WP_HIP_GRAPHS=1.
    void update_graph_pins(const void * graph_key, const std::vector<int> & page_indices);
    bool try_add_graph_pin_page(const void * graph_key, int page_idx, std::vector<int> & page_indices) const;

    // Stable slot base address for capture. Valid for the pool lifetime.
    void * slot_base_for_capture(int slot_idx) const { return pool_.slot_base_for_capture(slot_idx); }

private:
    // Internal helper: synchronous page-in (used by ensure() on miss).
    // Reads the page's bytes via FileIOLayer (sync path), copies to VRAM,
    // and zeros the padding. Returns the slot index or -1 on failure.
    // reuse_slot >= 0: read into that caller-owned (typically pinned) slot
    // instead of allocating a fresh one, and do NOT release it on failure — the
    // caller owns its lifecycle. reuse_slot < 0 keeps the original behavior
    // (alloc a slot, release it on any error). Returns the slot index or -1.
    int  page_in_sync_(int page_idx, int reuse_slot = -1);

    // Resolve a slot index to a VRAM pointer.
    void * slot_ptr_(int slot_idx) const { return pool_.slot_ptr(slot_idx); }

    // PoolAllocator's eviction callback — clears page_to_slot_[evicted].
    void on_pool_evict_(int slot_idx);
    void ensure_slot_map_(int slot_idx);

    void log_stats_summary();
    void record_page_in_(size_t bytes, double seconds);
    void restore_disable_graphs_env_();
    void release_graph_pins_();
    int  graph_pin_max_slots_() const;
    int  graph_pin_slot_count_except_(const void * graph_key) const;

    // Catalog of all pages. Built before init().
    PageCatalog catalog_;

    // Monotonic req_id source for the pager's OWN direct file_io_ submissions
    // (page_in_sync_ and ensure_batch). The FileIOLayer is shared with the
    // PrefetchScheduler, whose req_ids come from its own low counter; the high
    // bit here keeps the two spaces disjoint so a prefetch completion can never
    // be miscredited as a pager read on the shared ring's demux buffer.
    static constexpr uint64_t kPagerReqIdBit = (uint64_t) 1 << 62;
    uint64_t next_io_req_id_ = kPagerReqIdBit;

    // Owned subsystems.
    std::unique_ptr<FileIOLayer> file_io_;
    PoolAllocator                pool_;
    GpuTransport                 transport_;
    PrefetchScheduler            prefetch_;
    std::unique_ptr<HostTier>    host_tier_;

    // page_idx -> slot_idx (or -1). Set both for in-flight prefetches and
    // for committed (data-ready) pages — distinguished by page_loaded_.
    std::vector<int> page_to_slot_;
    // page_idx -> true iff the slot's data is committed (sync page-in done
    // OR prefetch stage 2 completed and reaped). False means slot is
    // reserved but the bytes aren't there yet.
    std::vector<bool> page_loaded_;
    std::vector<bool> cross_layer_prefetch_candidate_;
    std::vector<std::chrono::steady_clock::time_point> prefetch_started_at_;
    std::vector<int> page_async_event_;
    std::vector<ggml_backend_buffer_type_t> page_buft_;
    // Reverse map: slot_idx -> page_idx (or -1 if free). Used by the
    // eviction callback to clear page_to_slot_ / page_loaded_ correctly.
    std::vector<int> slot_to_page_;

    Config cfg_;
    bool   initialized_ = false;
    bool   hip_graphs_enabled_ = false;
    bool   async_ensure_enabled_ = false;

    // MMID active-set history (metrics / optional future priors only).
    static constexpr int kHotExpertCap  = 768;
    static constexpr int kHotHistoryMax = 256;
    std::vector<std::vector<int>> hot_expert_history_;
    // Pages we pin_page()'d for the open draft window; unpinned on clear.
    std::vector<int> draft_retain_pages_;
    // Last draft fire's tid2eid pages; ensure hits counted while marked.
    std::vector<bool> draft_tid2eid_mark_;
    // Precision set: pages predicted at last oracle fire; cleared on next fire.
    std::vector<bool> oracle_pred_mark_;
    std::vector<bool> oracle_pred_hit_; // seen in record_active since last fire
    // Sticky L2 keeps hash-hot resident pages pinned across skipped draft fires.
    std::vector<uint32_t> sticky_l2_score_;
    std::vector<bool> sticky_l2_mark_;
    std::vector<bool> sticky_l2_pinned_;
    std::vector<int> sticky_l2_pages_;
    bool sticky_l2_enabled_ = false;
    bool sticky_l2_stats_ = false;
    int  sticky_l2_max_pages_ = 32;
    int  sticky_l2_hits_since_refresh_ = 0;
    int draft_window_ = 0;
    // Pending sample-oracle token; flushed at sample (default) or layer-0 FA
    // when WP_SAMPLE_ORACLE_EAGER=0.
    int32_t pending_sample_token_ = -1;
    bool    pending_sample_flushed_ = true;
    // Sticky retain of sample-oracle hash pages across MoE until next sample.
    // Independent of draft_retain so set_draft_window(0) does not drop them
    // mid-window (server clears draft at sample boundary only after use).
    std::vector<int> sample_sticky_pages_;
    // Adaptive draft skip state.
    int draft_warm_streak_   = 0;
    int draft_oracle_fires_  = 0;
    int last_tid2eid_n_      = 0;
    int last_tid_cold_       = 0;
    uint64_t hits_at_last_fire_ = 0; // draft_tid2eid_hits_in_ensure snapshot
    float last_hit_ratio_    = 0.f;  // ensure hits / tid2eid pages for last window

    void note_draft_tid2eid_ensure_(int page_idx);
    bool draft_retain_contains_(int page_idx) const;
    void sticky_l2_refresh_(const char * reason);
    void sticky_l2_refresh_if_due_(const char * reason);
    void release_sticky_l2_();
    void oracle_begin_prediction_(const std::vector<int> & pages);
    void oracle_finalize_fp_();
    bool is_tid2eid_hash_page_(int page_idx) const;
    // Commit Done prefetches -> page_loaded_ and free scheduler slots.
    // Without this, depth-4 Done slots block all further draft submits.
    int  harvest_ready_prefetches_();
    // Submit cold pages in waves, draining/harvesting between waves.
    // allow_evict=false: free pool slots only.
    int  submit_cold_waves_(const std::vector<int> & cold, bool allow_evict = true);
    // Temp-pin recent MoE routing history so sample-oracle LRU skips them.
    // Returns pages that were pinned (caller must unpin_page each).
    std::vector<int> pin_oracle_protect_set_();
    // Free up to n_need slots by releasing unpinned *stale* hash-layer pages
    // (not in keep_pages). Does not touch MoE working-set pages. Returns how
    // many slots were freed.
    int free_stale_hash_slots_(int n_need, const std::vector<int> & keep_pages);
    // Pin up to cap resident/in-flight sample-oracle pages until next fire.
    void clear_sample_sticky_();
    void install_sample_sticky_(const std::vector<int> & tid_pages, int cap);

    // Hash-layer token->expert tables (host). Indexed by block_idx.
    struct Tid2EidTable {
        int block_idx      = -1;
        int n_expert_used  = 0;
        int n_vocab        = 0;
        std::vector<int32_t> data; // [token * n_expert_used + k]
    };
    std::vector<Tid2EidTable> tid2eid_tables_;

    void clear_draft_retain_();
    void clear_draft_tid2eid_mark_();
    void union_push_(std::vector<int> & dst, int page_idx, int cap) const;
    void collect_tid2eid_pages_(const int32_t * tokens, int n_tokens,
                                std::vector<int> & out, int cap) const;

    mutable Stats stats_;

    // GGML_CUDA_DISABLE_GRAPHS lifecycle (B-P5).
    bool        env_was_present_ = false;
    bool        env_disable_graphs_forced_ = false;
    std::string env_prior_value_;

    // MAD-P1 graph_key -> exact pool slots pinned for captured graph args.
    // Stored as slots, not pages, so unpin releases the same refcounts even
    // if a page later moves to another slot.
    std::unordered_map<const void *, std::vector<int>> graph_pin_slots_;
    std::unordered_set<const struct ggml_tensor *> routing_break_tensors_;
    struct {
        int n_nodes = -1;
        const void * first = nullptr;
        const void * last = nullptr;
    } routing_sig_;

    // Shared pinned staging buffer for page_in_sync_. Allocated once at
    // init, sized to max_page_size, reused across every sync page-in.
    // Pinning a fresh buffer per call (the original design) costs
    // hundreds of ms per allocation for hundred-MB tensors and dominates
    // the paging path for dense layers. Single shared buffer matches
    // the OLD pager's pool.pinned_staging behaviour.
    void * sync_staging_       = nullptr;
    size_t sync_staging_size_  = 0;
    bool   sync_staging_pinned_ = false;  // true if hipHostMalloc, false if malloc fallback

    // Multi-QD host bounce for ensure_batch (WP_ENSURE_BATCH_HOST=1):
    // Cold random buffered ~1.1 GB/s; O_DIRECT multi-QD ~6.2 GB/s on SN850X.
    // GGUF tensor offs are not 512-aligned, so each read uses an O_DIRECT
    // bounce (align-down offset, pad size) then H2D of the payload slice.
    std::vector<void *> ensure_host_bufs_;
    size_t              ensure_host_buf_bytes_ = 0;
    bool                ensure_host_bufs_pinned_ = false;
    std::vector<int>    ensure_odirect_fds_; // parallel to file_io fds; -1 = unused
    struct EnsureODirectReadJob {
        int      fd     = -1;
        uint64_t off    = 0;
        size_t   size   = 0;
        void *   dst    = nullptr;
        bool     done   = false;
        bool     ok     = false;
        int      err    = 0;
    };
    std::vector<std::thread>          ensure_odirect_workers_;
    std::deque<EnsureODirectReadJob *> ensure_odirect_queue_;
    std::mutex                       ensure_odirect_mu_;
    std::condition_variable          ensure_odirect_cv_;
    std::condition_variable          ensure_odirect_done_cv_;
    bool                             ensure_odirect_workers_stop_ = false;
    // Kept unconditional (NOT #if LLAMA_HAVE_IO_URING): a struct layout in a
    // shared header must never depend on a per-TU macro. LLAMA_HAVE_IO_URING is
    // PRIVATE to the llama target, so test TUs that include this header lack it;
    // guarding these members would make sizeof(WeightPager) differ between
    // libllama and the test, corrupting stack/heap pagers. Used only under the
    // guard in wp-pager.cpp; harmlessly unused when io_uring is unavailable.
    struct io_uring *    ensure_odirect_ring_ = nullptr;
    std::vector<uint8_t> ensure_odirect_fd_registered_;
    bool                 ensure_odirect_ring_files_registered_ = false;
    bool ensure_host_bufs_ready_(size_t n, size_t page_bytes);
    void free_ensure_host_bufs_();
    int  ensure_odirect_fd_(int file_idx);
    int  ensure_odirect_worker_count_(size_t n_jobs) const;
    bool ensure_odirect_workers_ready_(int n_workers);
    void shutdown_ensure_odirect_workers_();
    void ensure_odirect_worker_loop_();
#if defined(LLAMA_HAVE_IO_URING) && defined(__linux__)
    bool ensure_odirect_ring_ready_(size_t n_entries);
    bool ensure_odirect_fixed_fd_ready_(int file_idx, int fd, size_t n_entries);
    void shutdown_ensure_odirect_ring_();
#endif
};

// File-range descriptor for advise_prefetch — one per paged tensor in the
// catalog walk. Exposed at namespace scope so unit tests can validate the
// catalog-walk logic (compute_advise_ranges) without standing up a real
// WeightPager.
struct AdviseRange {
    uint16_t fd_idx;
    uint64_t offset;
    size_t   size;
};

// Walk `catalog` for every page whose block_idx is in [block_idx+1,
// block_idx+k] AND whose size > 0 (i.e. excludes consolidated parents,
// which have no own bytes). Skips negative inputs. Caller owns the
// returned vector. Pure function — no I/O, no global state. Unit-tested
// directly without a real pager.
std::vector<AdviseRange> compute_advise_ranges(const PageCatalog & catalog,
                                                int                 block_idx,
                                                int                 k);

}  // namespace wp
