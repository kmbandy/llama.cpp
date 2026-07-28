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
#include "wp-host-prefetch.h"
#include "wp-host-tier.h"
#include "wp-pool.h"
#include "wp-gpu-transport.h"
#include "wp-prefetch.h"
#include "wp-router-predictor.h"

#include <atomic>
#include <functional>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <map>
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

// Explicit opt-in resolver, kept CPU-testable without a HIP pager instance.
bool wp_pipeline_promotions_enabled();

// Achieved-concurrency accounting for the HOST O_DIRECT read-worker pool
// (see ensure_odirect_worker_loop_ in wp-pager.cpp). ensure_batch_max_n /
// ensure_batch_avg_n (Stats, below) only count how many jobs were QUEUED
// per batch -- they say nothing about how many pread()s were genuinely
// executing at once. This tracker answers that question directly:
// begin() is called by a worker immediately before issuing its pread(),
// end() immediately after that pread() returns (success or failure).
//
// Deliberately lock-free -- three independent atomics, no mutex -- because
// this sits on the hot read path and any lock here would itself become the
// bottleneck it exists to diagnose.
//
//   peak()    -- the highest in-flight count observed by ANY begin() call
//                across the whole run (a running high-water mark).
//   average() -- the in-flight count is SAMPLED at the instant each
//                begin() call happens (i.e. how many reads -- including
//                the one just starting -- are in flight at that moment),
//                and average() is the mean of those per-begin samples.
//                This is a call-weighted average of queue depth at
//                read-start, NOT a time-weighted average over wall-clock
//                (a time-weighted version would need a timestamp per event
//                plus an integral, which is not worth the added cost or
//                complexity here). It is cheap -- one atomic fetch_add per
//                begin() -- and is defensible for the question it exists
//                to answer: are ~9 queued jobs actually running ~9 deep,
//                or effectively serializing? A pool that truly serializes
//                reports avg == 1.0 (every read starts finding itself
//                alone in flight); a pool that overlaps N reads at a time
//                reports avg close to N.
struct EnsureODirectInFlightTracker {
    std::atomic<int64_t> current_{0};
    std::atomic<int64_t> peak_{0};
    std::atomic<int64_t> sample_sum_{0};
    std::atomic<int64_t> samples_{0};

    // Call immediately before issuing pread(). Returns the post-increment
    // in-flight count (informational; callers may ignore it).
    int64_t begin() {
        const int64_t n = current_.fetch_add(1, std::memory_order_relaxed) + 1;
        sample_sum_.fetch_add(n, std::memory_order_relaxed);
        samples_.fetch_add(1, std::memory_order_relaxed);
        int64_t prev = peak_.load(std::memory_order_relaxed);
        while (n > prev && !peak_.compare_exchange_weak(prev, n, std::memory_order_relaxed)) {
            // prev is updated to the current value by compare_exchange_weak
            // on failure; loop retries until we win or another begin() has
            // already pushed the peak >= n.
        }
        return n;
    }

    // Call immediately after pread() returns (success or failure).
    void end() {
        current_.fetch_sub(1, std::memory_order_relaxed);
    }

    int64_t current() const { return current_.load(std::memory_order_relaxed); }
    int64_t peak()    const { return peak_.load(std::memory_order_relaxed); }
    double  average() const {
        const int64_t n = samples_.load(std::memory_order_relaxed);
        return n > 0 ? (double) sample_sum_.load(std::memory_order_relaxed) / (double) n : 0.0;
    }
};

class WeightPager {
public:
    struct Config {
        int  n_slots         = 0;     // size of the VRAM ring; -1 / 0 = auto (one per layer)
        int  prefetch_depth  = 4;     // PrefetchScheduler queue depth
        int  io_uring_depth  = 0;     // FileIOLayer SQ depth; 0 = prefetch_depth
        bool prefer_async_io = true;  // try io_uring for stage 1 before SyncPread

        // Force every pool slot offset to a multiple of this. Vulkan needs the
        // quant block size here (ggml_type_size of the paged tensors) because
        // its matmul indexes the weight buffer as an array of quant blocks, so
        // a slot that is not block-aligned cannot be addressed. 1 = no extra
        // constraint, which is what HIP/CUDA want: they use raw byte pointers
        // and their alignment is already validated.
        size_t block_alignment = 1;
    };

    struct Stats {
        uint64_t page_ins                       = 0;
        uint64_t evictions                      = 0;
        uint64_t prefetch_hits                  = 0;
        uint64_t prefetch_misses                = 0;
        uint64_t sync_fallbacks                 = 0;
        uint64_t batch_slot_exhaustions         = 0;  // ensure_batch: no slot AND sync fallback failed -> NULL out_ptr
        // page_ins attribution by call site. page_ins is the SUM of these plus
        // the ensure_batch bursts; measured 2026-07-27 that ~36% of page_ins on
        // laguna do NOT come from ensure_batch and run at the drive's QD1 rate
        // (0.64 GB/s vs the batch path's 2.27), consuming ~67% of the I/O time.
        // These name which serial path is responsible.
        uint64_t page_ins_ensure_async          = 0;  // ensure(): prefetch completion harvested
        uint64_t page_ins_ensure_sync           = 0;  // ensure(): sync fallback after prefetch miss
        uint64_t page_ins_prefetch_reap         = 0;  // bulk prefetch reap path
        uint64_t page_ins_sync_direct           = 0;  // page_in_sync_ called directly
        // Which CALLER invoked page_in_sync_. The page_ins_* counters above
        // attribute by FUNCTION, which named the what but not the who: 25,431
        // of 70,104 page_ins went through page_in_sync_ and the total was
        // IDENTICAL across four arms whose batch width differed 2.8x, so the
        // serial population is a fixed set of pages, not a queue-depth artifact.
        uint64_t pis_from_ensure                = 0;  // ensure() sync fallback
        uint64_t pis_vk_host                    = 0;  // vulkan HOST path per-miss
        uint64_t pis_host_path                  = 0;  // WP_ENSURE_BATCH_HOST per-miss
        uint64_t pis_nonhip                     = 0;  // non-HIP/CUDA build path
        uint64_t pis_tier_pre                   = 0;  // pre-pipeline HostTier hit
        uint64_t pis_read_failed                = 0;  // P2P read/padding failed
        uint64_t pis_tier_promo                 = 0;  // tier promotion request
        uint64_t pis_serial_batch               = 0;  // whole batch went serial
        uint64_t io_bytes                       = 0;
        double   io_seconds                     = 0.0;
        uint64_t lru_walk_hot_skips             = 0;
        uint64_t lru_walk_pinned_skips          = 0;
        uint64_t dense_prefetch_submitted       = 0;
        uint64_t cross_layer_prefetch_submitted = 0;
        uint64_t cross_layer_hit_in_ensure      = 0;
        uint64_t speculative_evicted_unused     = 0; // xlayer: prefetched, evicted, never demanded
        uint64_t xlayer_predict_calls           = 0; // submit_xlayer_prefetch invocations
        uint64_t xlayer_pred_pages              = 0; // predicted sister pages before filters
        uint64_t xlayer_resident_skips          = 0; // predicted pages already resident/in-flight
        uint64_t xlayer_blocked_budget          = 0; // submits skipped: speculative cap reached
        uint64_t xlayer_blocked_free_queue      = 0; // submits skipped: scheduler queue full
        uint64_t xlayer_bootstrap_allocs        = 0; // WP_SPEC_BOOTSTRAP: speculative slots seeded by eviction
        uint64_t demand_trimmed_by_reserve      = 0; // demand pages withheld to protect the reserve
        uint64_t xlayer_harvest_calls           = 0; // harvests run before the speculative gate
        uint64_t xlayer_harvested_pages         = 0; // Done prefetches committed+reaped by them
        uint64_t host_tier_hits                 = 0;
        uint64_t host_tier_stores               = 0; // evicted pages copied VRAM->RAM
        uint64_t host_tier_store_fail           = 0; // D2H refused or failed
        uint64_t host_prefetch_enqueued         = 0;
        uint64_t host_prefetch_dropped          = 0;
        uint64_t host_prefetch_read             = 0;
        uint64_t host_prefetch_read_fail        = 0;
        uint64_t host_prefetch_skipped          = 0;
        // Soft-prefetch policy (HostTier path): predictions held until N strikes
        // and/or trimmed by the per-wave byte budget.
        uint64_t host_prefetch_strike_held      = 0;
        uint64_t host_prefetch_budget_trim      = 0;
        // HostTier speculative sub-tier (WP_HOST_SPEC_TIER). The ratio that
        // matters is promotions vs evicted_unused: it is the RAM prefetcher's
        // precision, measured on pages that actually reached the tier.
        // CRITICAL-PATH PROFILE of the inline prefetch block in the eval cb.
        // cb_prefetch_cpu_ms  = work on the eval thread (scalar GEMV, f32
        //                       convert, softmax) -- this is the tax a side
        //                       thread would remove.
        // cb_prefetch_wall_ms - cb_prefetch_cpu_ms = blocked/descheduled, i.e.
        //                       hipStreamSynchronize.
        // Thread CPU time is load-immune; wall clock on this box is not
        // (20-23% within-arm spread, decision da055d88).
        uint64_t cb_prefetch_calls              = 0;
        double   cb_prefetch_wall_ms            = 0.0;
        double   cb_prefetch_cpu_ms             = 0.0;
        uint64_t host_predict_calls             = 0;
        double   host_predict_cpu_ms            = 0.0; // the router GEMV alone
        uint64_t host_spec_resident             = 0; // unconfirmed predictions in RAM now
        uint64_t host_spec_evicted_unused       = 0; // predictions thrown away unused
        uint64_t host_spec_promotions           = 0; // predictions a demand hit confirmed
        uint64_t ensure_batch_host_hits         = 0; // P2P/path misses served from HostTier
        // MAD: O_DIRECT alignment fixed to the filesystem's actual block size
        // (was hardcoded 512, wrong for e.g. btrfs's 4096) -- see
        // resolve_odirect_alignment()/compute_odirect_read_plan() below. This
        // counter fires when the padded, aligned read for a page would not
        // fit the sized bounce buffer; previously that case was a silent
        // `continue` that fell back to the slower sync path with no signal.
        uint64_t ensure_batch_host_odirect_cap_skips = 0;
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
        // ensure_batch multi-QD bursts. ensure_batch_submit_seconds/wait_seconds
        // carry their ORIGINAL P2P-path meaning: submit is io_uring enqueue
        // time, wait is completion-wait time. The HOST O_DIRECT pthread-pool
        // path (WP_ENSURE_BATCH_HOST=1) also writes these two fields, as
        // aliases, for backward compatibility -- but on that path submit is
        // actually the whole storage-read wall-clock and wait is actually the
        // H2D copy phase. Do not read submit/wait as "P2P path" numbers; the
        // ensure_batch_host_* fields below are the ones that name the
        // HOST-path phases correctly and should be used for HOST-path analysis.
        uint64_t ensure_batch_calls                = 0;
        uint64_t ensure_batch_pages                = 0; // cold misses issued in batches
        uint64_t ensure_batch_max_n                = 0; // largest concurrent *real storage*
                                                         // submission set (excludes HostTier hits)
        uint64_t ensure_batch_bytes                = 0;
        // Denominator for ensure_batch_gb_s: storage-read-phase wall time
        // only, comparable across transports. On the HOST path this
        // excludes the H2D copy phase; the P2P path has no separate H2D
        // phase (direct_to_device reads land in the VRAM slot directly),
        // so its submit+wait wall already qualifies. Batches with zero
        // real storage bytes (pure HostTier hits) contribute no time here.
        double   ensure_batch_seconds              = 0.0;
        double   ensure_batch_submit_seconds       = 0.0;
        double   ensure_batch_wait_seconds         = 0.0;
        uint64_t ensure_batch_timeouts             = 0; // wait_for_req returned non-Ok
        // Real storage submissions only. Both paths must exclude HostTier
        // hits: on HOST this is the count of jobs actually enqueued to the
        // O_DIRECT worker pool (n_submitted); on P2P it is submit_batch's
        // return value (n_sub). HostTier hits never reach either queue.
        uint64_t ensure_batch_n_sub_sum            = 0;
        uint64_t ensure_batch_window_pressure_fallbacks = 0;
        // Transport identity: which ensure_batch branch actually served
        // each batch's reads this run. More than one can be nonzero in a
        // single run (e.g. a HOST-path read failing over to the per-page
        // sync fallback), so all three are counted rather than keeping
        // only the last path taken.
        uint64_t ensure_batch_host_path_batches    = 0; // served by HOST O_DIRECT pthread pool
        uint64_t ensure_batch_p2p_path_batches     = 0; // served by P2P direct_to_device
        uint64_t ensure_batch_serial_path_batches  = 0; // served by serial sync fallback
        // HOST O_DIRECT pthread-pool path only (WP_ENSURE_BATCH_HOST=1): the
        // five phases of ensure_batch's HOST branch, named for what each
        // actually measures. Unset (0.0) on the P2P and serial-fallback paths.
        double   ensure_batch_host_jobs_seconds      = 0.0; // building the job list (HostTier lookup + fd resolution)
        double   ensure_batch_host_prep_seconds      = 0.0; // computing O_DIRECT alignment/bounce params
        double   ensure_batch_host_enqueue_seconds   = 0.0; // enqueueing jobs to the worker queue
        double   ensure_batch_host_read_wait_seconds = 0.0; // blocked until all reads complete
        double   ensure_batch_host_h2d_seconds       = 0.0; // H2D copy phase (mixed: promotion + fresh)
        // P2P direct-to-device path: names deliberately mirror the HOST
        // phases so arm-to-arm logs are comparable. H2D is structurally zero
        // on a successful direct read, but remains explicit in the summary.
        double   ensure_batch_p2p_jobs_seconds       = 0.0;
        double   ensure_batch_p2p_prep_seconds       = 0.0;
        double   ensure_batch_p2p_enqueue_seconds    = 0.0;
        double   ensure_batch_p2p_read_wait_seconds  = 0.0;
        double   ensure_batch_p2p_h2d_seconds        = 0.0;
        uint64_t ensure_batch_p2p_fresh_count        = 0;
        uint64_t ensure_batch_p2p_inflight_peak      = 0;
        double   ensure_batch_p2p_inflight_avg_at_read_start = 0.0;
        // MAD-P4 follow-up: ensure_batch_host_h2d_seconds above mixes two
        // different kinds of H2D copy -- a HostTier RAM->VRAM promotion
        // (page already read once from storage, now just moving host RAM
        // bytes to a VRAM slot) and a fresh storage read's H2D (first time
        // this page's bytes land in VRAM this run). Differencing separate
        // aggregate phases to infer a per-promotion cost is unreliable (a
        // 36% spread was observed across two runs); these fields measure
        // each kind directly instead. Both are 0.0 unless host_tier_ is
        // enabled. Reported distinctly per call site (ensure_batch's HOST
        // O_DIRECT path vs page_in_sync_) since the two paths batch their
        // H2D copies differently and their per-page cost could differ.
        uint64_t ensure_batch_host_promotion_count      = 0;   // pages promoted RAM->VRAM in ensure_batch HOST path
        double   ensure_batch_host_promotion_h2d_seconds = 0.0; // their H2D copy time only
        // Of ensure_batch_host_promotion_count above, how many sourced their
        // H2D straight from the pinned HostTier arena via borrow() (2026-07-25
        // zero-copy design) instead of lookup()'s memcpy-into-bounce-buffer
        // fallback (taken when the arena isn't HIP-pinned). Lets a session
        // tell the fast path from the fallback in a log instead of inferring
        // it from indirect evidence.
        uint64_t ensure_batch_host_zerocopy_promotions   = 0;
        uint64_t tier_promotion_async_enqueued            = 0;
        uint64_t tier_promotion_sync_enqueued             = 0;
        uint64_t tier_promotion_event_pool_exhausted      = 0;
        // The completion-fence subset above, separately visible for diagnosis.
        double   tier_promotion_fence_seconds             = 0.0;
        uint64_t ensure_batch_host_fresh_count           = 0;   // fresh storage reads' H2D in ensure_batch HOST path
        double   ensure_batch_host_fresh_h2d_seconds     = 0.0; // their H2D copy time only
        uint64_t page_in_sync_promotion_count            = 0;   // pages promoted RAM->VRAM in page_in_sync_
        double   page_in_sync_promotion_h2d_seconds      = 0.0; // their H2D copy time only (transport_.stage_in)
        uint64_t page_in_sync_zerocopy_promotions         = 0;   // HostTier borrow() source, not lookup() copy
        uint64_t page_in_sync_fresh_count                = 0;   // fresh storage reads' H2D in page_in_sync_
        double   page_in_sync_fresh_h2d_seconds          = 0.0; // their H2D copy time only (transport_.stage_in)
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
                              bool allow_evict = true,
                              bool speculative = false);

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

    // Append the sister page indices (gate/up/down) for (block_idx, expert_idx)
    // to `out`, using the reverse index built in init(). O(1) replacement for
    // PageCatalog::pages_for_expert on the cross-layer prefetch hot path.
    void expert_sister_pages(int block_idx, int expert_idx, std::vector<int> & out) const;

    // --- Cross-layer prefetch (WP_PREFETCH_XLAYER) --------------------------
    // Store a host f32 copy of layer L's ffn_gate_inp router weight so the
    // RouterPredictor can score future layers' experts from a live residual.
    void note_router_weight(int block_idx, const float * W, int n_expert, int n_embd);
    bool predictor_has_router(int block_idx) const;
    // Predict layer from_layer+1..+K experts from residual h and speculatively
    // prefetch their sister pages. No-op unless WP_PREFETCH_XLAYER is set.
    void submit_xlayer_prefetch(const float * h, int from_layer);
    bool xlayer_prefetch_enabled() const { return xlayer_prefetch_enabled_; }
    void submit_host_prefetch(const float * h, int from_layer);
    // Accumulate one execution of the inline eval-cb prefetch block. Called
    // from the block's scope guard so it records on every exit path.
    void note_cb_prefetch_cost(double wall_ms, double cpu_ms) {
        ++stats_.cb_prefetch_calls;
        stats_.cb_prefetch_wall_ms += wall_ms;
        stats_.cb_prefetch_cpu_ms  += cpu_ms;
    }
    bool host_prefetch_enabled() const { return host_prefetcher_ != nullptr; }

    // Scheduler queue slots that DEMAND prefetch may not consume, so that
    // speculative submits are not starved by demand contention for one shared
    // pool (WP_SPEC_RESERVE). 0 = off = previous behaviour exactly.
    int  spec_reserve() const { return spec_reserve_; }

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
    // Base of the slot pool. On Vulkan this is the backend's fixed pointer
    // sentinel, so slot_ptr() - pool_base() is the true buffer offset.
    void *                pool_base() const { return pool_.pool_base(); }
    // True only when the pool actually lives in a Vulkan buffer. Callers must
    // gate Vulkan-specific behaviour on THIS, not on #if defined(GGML_USE_VULKAN)
    // — a build with several backends compiled in (e.g. build-army: CUDA +
    // Vulkan + RPC) defines the macro on every run regardless of the device
    // in use.
    bool                  is_vulkan() const { return transport_.is_vulkan(); }
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
    struct TierPromotionRequest {
        int page;
        int slot;
    };
    struct TierPromotion {
        int                    page;
        HostTier::BorrowHandle  borrow;
        int                    event;
    };
    using TierPromotionEnqueue = std::function<bool(void *, const void *, size_t, size_t, int &)>;

    // Borrows sources, enqueues copies, and retains generation handles until
    // release_tier_promotions_ is called after the caller's completion fence.
    // Requests which cannot enqueue deliberately remain absent from `queued`.
    void enqueue_tier_promotions_(const std::vector<TierPromotionRequest> & requests,
                                  std::vector<TierPromotion> & queued,
                                  const TierPromotionEnqueue & enqueue,
                                  bool transport_events);
    bool synchronize_tier_promotions_(const std::vector<TierPromotion> & queued);
    void release_tier_promotions_(std::vector<TierPromotion> & queued);
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

    // (block_idx, expert_idx) -> sister page indices (gate/up/down), built
    // once in init() from catalog_. Mirrors PageCatalog::pages_for_expert but O(1).
    std::map<std::pair<int,int>, std::vector<int>> expert_page_index_;

    // Queue slots withheld from demand for speculative use (WP_SPEC_RESERVE).
    int  spec_reserve_ = 0;

    // Harvest finished prefetches before the speculative capacity gate
    // (WP_SPEC_REAP). Done-but-unreaped slots hold queue capacity that no
    // reservation can reclaim. 0 = off = previous behaviour exactly.
    bool spec_reap_ = false;
    bool spec_keep_tier_ = false;  // harvest keeps pages speculative (touch_lru, not mark_used)

    // Cross-layer prefetch (WP_PREFETCH_XLAYER). predictor_ holds host f32
    // copies of each layer's ffn_gate_inp; config knobs parsed once in init().
    RouterPredictor predictor_;
    bool xlayer_prefetch_enabled_ = false;
    int  xlayer_lookahead_k_      = 2;
    int  xlayer_topk_             = 16;
    int  xlayer_max_slots_        = 0;   // 0 => n_slots/4, set in init()
    // Confidence gate for VRAM speculation (WP_PREFETCH_MIN_CONF /
    // _CONF_STEP). The gate shipped 2026-07-22 on the HOST path only; the
    // xlayer path called predict() without the argument, so it defaulted to
    // 0.0 and pulled every expert in top-M regardless of routing probability.
    // Default 0.0f keeps that behaviour until explicitly set.
    float xlayer_min_conf_        = 0.0f;
    float xlayer_conf_step_       = 0.0f;
    int  n_layer_                 = 0;   // max catalog block_idx + 1
    int  host_prefetch_lookahead_ = 2;
    int  host_prefetch_topm_      = 16;
    float host_prefetch_min_conf_ = 0.0f;
    // Soft host-prefetch policy (decaying horizon + thrash guards).
    // conf for distance d: min(0.99, min_conf + (d-1)*conf_step).
    // M for distance d: topm for d==1, max(1, topm >> (d-1)) otherwise.
    float  host_prefetch_conf_step_     = 0.10f;
    int    host_prefetch_strikes_needed_ = 2;   // 1 = enqueue on first conf pass
    size_t host_prefetch_bytes_budget_  = 64ull << 20; // 0 = unlimited per wave
    // Per-page prediction strike counts for the 2-strike gate (catalog-sized).
    std::vector<uint8_t> host_prefetch_strikes_;

    // Monotonic req_id source for the pager's OWN direct file_io_ submissions
    // (page_in_sync_ and ensure_batch). The FileIOLayer is shared with the
    // PrefetchScheduler, whose req_ids come from its own low counter; the high
    // bit here keeps the two spaces disjoint so a prefetch completion can never
    // be miscredited as a pager read on the shared ring's demux buffer.
    static constexpr uint64_t kPagerReqIdBit = (uint64_t) 1 << 62;
    uint64_t next_io_req_id_ = kPagerReqIdBit;

    // Owned subsystems.
    std::unique_ptr<FileIOLayer> file_io_;
    std::unique_ptr<FileIOLayer> host_prefetch_file_io_;
    PoolAllocator                pool_;
    GpuTransport                 transport_;
    PrefetchScheduler            prefetch_;
    std::unique_ptr<HostTier>    host_tier_;
    std::unique_ptr<HostPrefetcher> host_prefetcher_;
    uint64_t next_host_prefetch_req_id_ = 1;

    // page_idx -> slot_idx (or -1). Set both for in-flight prefetches and
    // for committed (data-ready) pages — distinguished by page_loaded_.
    std::vector<int> page_to_slot_;
    // page_idx -> true iff the slot's data is committed (sync page-in done
    // OR prefetch stage 2 completed and reaped). False means slot is
    // reserved but the bytes aren't there yet.
    std::vector<bool> page_loaded_;
    // Per-page access histogram (WP_PAGE_HIST=1). page_access_ counts how many
    // times a page was REQUESTED (ensure / ensure_batch), page_pagein_ how many
    // times it actually had to be read. A page requested every token but always
    // resident costs nothing; the pin candidates are high-access AND high-pagein.
    std::vector<uint32_t> page_access_;
    std::vector<uint32_t> page_pagein_;
    uint64_t              page_hist_total_accesses_ = 0;
    bool                  page_hist_enabled_        = false;
    void log_page_histogram_() const;
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
    // WP_PIPELINE_PROMOTIONS defaults on; setting it to 0 selects the
    // synchronous promotion route for A/B control and required configurations.
    bool   pipeline_promotions_enabled_ = true;

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
    // GGUF tensor offs are not filesystem-block-aligned, so each read uses an
    // O_DIRECT bounce (align-down offset, pad size) then H2D of the payload
    // slice. The alignment authority is the FILESYSTEM's block size (e.g.
    // btrfs = 4096), not the device's logical_block_size (512) -- using the
    // device value under-aligns on filesystems with a larger block size and
    // causes read amplification (measured 2.49x on btrfs). See
    // resolve_odirect_alignment() / compute_odirect_read_plan().
    std::vector<void *> ensure_host_bufs_;
    size_t              ensure_host_buf_bytes_ = 0;
    bool                ensure_host_bufs_pinned_ = false;
    // Arena came from GpuTransport::host_alloc (Vulkan-registered host memory),
    // so it must be released through host_free, not hipHostFree/std::free. Kept
    // separate from ensure_host_bufs_pinned_ because that flag gates HIP-only
    // zero-copy promotion paths that do not apply here.
    bool                ensure_host_bufs_vk_pinned_ = false;
    std::vector<int>    ensure_odirect_fds_; // parallel to file_io fds; -1 = unused
    // Parallel to ensure_odirect_fds_: the resolved O_DIRECT alignment (bytes,
    // power of two) and cached file size for each file_idx, populated once in
    // ensure_odirect_fd_() alongside opening the fd. 0 = not yet resolved.
    std::vector<size_t>   ensure_odirect_align_;
    std::vector<uint64_t> ensure_odirect_filesize_;
    // Running max of ensure_odirect_align_ across all resolved shards -- used
    // to size the shared bounce buffers (ensure_host_bufs_ready_) before any
    // individual file_idx's alignment may be known yet. Defaults to 4096
    // (the fallback alignment) until the first shard resolves, which already
    // covers the common btrfs case; self-corrects (triggers reallocation) if
    // a later-resolved shard needs a larger alignment.
    size_t                ensure_odirect_align_max_ = 0;
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
    // Achieved-concurrency counter for the HOST O_DIRECT path (see
    // EnsureODirectInFlightTracker above). Incremented/decremented in
    // ensure_odirect_worker_loop_ around each job's pread(); read only
    // at log time in log_stats_summary().
    EnsureODirectInFlightTracker     ensure_odirect_inflight_;
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

// Resolve the O_DIRECT alignment to use for a shard given its filesystem's
// reported block size (statfs/fstatfs f_bsize). This is the alignment
// AUTHORITY for O_DIRECT: the device's logical_block_size is NOT sufficient
// -- a filesystem (e.g. btrfs, f_bsize=4096) can require coarser alignment
// than the underlying device reports (512), and misaligned O_DIRECT reads
// silently amplify (measured 2.49x: 221.9 GB delivered vs 82.7 GB buffered
// for identical 89.24 GB of requested pages). Guards: never returns a value
// below kLogicalBlockFloor; requires a power of two; falls back to
// kFallbackAlign if the input is <= 0 or not a power of two after flooring.
// Pure function -- no syscalls, unit-testable directly.
size_t resolve_odirect_alignment(long fs_bsize);

// The aligned, EOF-clamped O_DIRECT read window for one payload read of
// [off, off+size) bytes.
struct OdirectReadPlan {
    uint64_t base;    // align-down start offset for the O_DIRECT pread
    size_t   prefix;  // payload's byte offset within the bounce buffer
    size_t   nbytes;  // bytes to actually request this read
};

// Compute the O_DIRECT read plan for a payload at [off, off+size) against a
// filesystem alignment `align` (must be a power of two -- see
// resolve_odirect_alignment) and a shard of `file_size` bytes. `file_size ==
// 0` means "unknown / unresolved" and disables EOF clamping.
//
// Pads [off, off+size) out to `align`-aligned boundaries on both ends, then
// clamps the padded tail so the request never reads past `file_size`: on
// O_DIRECT, a padded read that overruns EOF returns EIO rather than a short
// read (the pre-existing bug this also fixes -- it fired 3x/run at 512
// alignment and gets strictly worse at 4096, since the overrun can grow to
// up to align-1 bytes). The payload itself is always fully covered even if
// `file_size` is inconsistent with `off+size`.
//
// Pure function -- no I/O, no global state. Unit-tested directly without a
// real pager or a real file.
OdirectReadPlan compute_odirect_read_plan(uint64_t off, size_t size, size_t align, uint64_t file_size);

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

// Build a (block_idx, expert_idx) -> sister page indices reverse index from a
// catalog. Mirrors PageCatalog::pages_for_expert. Pure function, unit-testable
// without a live pager.
void build_expert_page_index(const PageCatalog & catalog,
                             std::map<std::pair<int,int>, std::vector<int>> & out);

}  // namespace wp
