#pragma once

#include "pipe-expert-dispatcher.h"
#include "pipe-hash-oracle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <thread>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace pipe_expert_dispatcher {

class ngram_hint_table;

class graph_dispatcher {
  public:
    // last_no_defer_layer: last main-graph MoE layer that must not leave
    // deferred work pending. Pass hparams.n_layer()-1 so NextN/MTP worker
    // layers are not treated as the fold successor. -1 = worker HELLO max.
    graph_dispatcher(const std::string & endpoints,
                     int32_t             n_embd,
                     int32_t             n_ff_exp,
                     int32_t             n_expert,
                     int32_t             n_expert_used,
                     int32_t             last_no_defer_layer = -1,
                     int32_t             phantom_token = -1);
    ~graph_dispatcher();

    graph_dispatcher(const graph_dispatcher &)             = delete;
    graph_dispatcher & operator=(const graph_dispatcher &) = delete;

    // WP_DISPATCH_CHUNKS as the dispatcher resolved it (1 = single dispatch);
    // the graph builder decides per-layer chunking from this.
    int dispatch_chunks() const { return remote.dispatch_chunks(); }

    ggml_tensor * build(ggml_context * ctx,
                        ggml_tensor *  activations,
                        ggml_tensor *  selected_experts,
                        ggml_tensor *  weights,
                        int32_t        layer,
                        float          swiglu_clamp,
                        bool            chunked = false);

    // Split of build() so a sibling GPU op (shared expert) can sit between
    // the worker send and the worker recv. build_issue sends; after_issue
    // pins the shexp *input* (scale+acc_inplace of 0*issued[0]) so the FFN
    // cannot start before send; build_wait recvs. build() remains combined.
    //
    // build_wait depends ONLY on the issue node. Joining wait to shexp as a
    // graph src forced the scheduler to finish shexp (and copy it to host)
    // before recv — GPU idle for the whole RPC. The ggml_add of wait+shexp
    // is the join; wait's compute does not read shexp (GGML_UNUSED).
    // after_issue the shexp INPUT, not the output: pinning the output left
    // the FFN in an earlier GPU split that finished before wait.
    ggml_tensor * build_issue(ggml_context * ctx,
                              ggml_tensor *  activations,
                              ggml_tensor *  selected_experts,
                              ggml_tensor *  weights,
                              int32_t        layer,
                              float          swiglu_clamp,
                              int32_t        chunk_index = 0,
                              int32_t        chunk_count = 1,
                              ggml_tensor *  issue_dependency = nullptr,
                              ggml_tensor *  full_activations = nullptr,
                              ggml_tensor *  full_selected_experts = nullptr,
                              ggml_tensor *  full_weights = nullptr,
                              int64_t        token_offset = 0);
    ggml_tensor * after_issue(ggml_context * ctx, ggml_tensor * tensor, int32_t layer);
    ggml_tensor * build_wait(ggml_context * ctx, int32_t layer);

    void begin_graph_build(ggml_context * ctx);
    bool begin_chunked_issue_build(ggml_context * ctx, int32_t layer);
    void note_wait_expanded(ggml_context * ctx, int32_t layer);

    // ---- hash-layer prefetch ------------------------------------------------
    //
    // Register the host copy of blk.<layer>.ffn_gate_tid2eid. LOAD TIME ONLY --
    // call before the first decode and never again. See hash_oracle.
    void register_hash_layer(int32_t         layer,
                             int32_t         n_expert_used,
                             int32_t         n_vocab,
                             const int32_t * data);

    bool   has_hash_oracle()    const { return !oracle_.empty(); }
    size_t hash_oracle_layers() const { return oracle_.layers().size(); }

    // Drop every registered table, disabling hints. For the loader to call when
    // it cannot register the WHOLE hash block -- see hash_oracle::clear().
    void clear_hash_oracle() { oracle_.clear(); }

    // For every registered hash layer, resolve `tokens` to expert ids and offer
    // them to the workers that will be asked for them. Advisory: sends no
    // request, awaits nothing, and never throws -- a failed hint costs a page-in
    // the run was going to pay anyway.
    //
    // MUST NOT be called with a dispatch in flight: it writes to the same
    // sockets. Call it between decodes, not from inside the graph.
    //
    // Returns the number of hint frames sent.
    // n_certain of the leading tokens are ones the target WILL process; the rest
    // are predicted. They are sent as separate frames so the worker can price
    // them differently -- see pipe_hint_provenance.
    //
    // `conf` (optional, n_tokens long) is the probability each token is real.
    // The leading n_certain entries are ignored -- a committed token is 1.0 by
    // definition -- so this only prices the PREDICTED tail. Without it every
    // predicted token counts as certain, which is what made the predicted frame
    // an undifferentiated union that any downstream cap had to truncate by
    // expert id. See WP_PREFETCH_CONF_MIN / WP_PREFETCH_TOPM.
    size_t prefetch_for_tokens(const int32_t * tokens, size_t n_tokens,
                               size_t n_certain = SIZE_MAX,
                               const float * conf = nullptr);

    // Minimum per-expert confidence for a PREDICTED hint (WP_PREFETCH_CONF_MIN,
    // default 0.4 -- the draft head's own conf_min). CERTAIN is never gated.
    static float predicted_conf_min();

    // Cap on PREDICTED experts per layer frame, highest confidence first
    // (WP_PREFETCH_TOPM, default 6 = n_expert_used). 0 = uncapped.
    static size_t predicted_top_m();

    // Append the raw token batch to the WP_PREDICT_CAPTURE stream.
    void note_batch_tokens(const int32_t * tokens, size_t n_tokens) noexcept;

    // Score static token-conditioned counts before graph compute and offer one
    // PREDICTED top-M set per table layer. Wide prefill batches are skipped.
    size_t prefetch_ngram_for_tokens(const int32_t * tokens, size_t n_tokens) noexcept;

    const prefetch_hint_stats & hint_stats() const { return remote.get_prefetch_hint_stats(); }

    // WP_PREFETCH_HINT=1. DEFAULT OFF: this changes what goes on the wire, and a
    // bare run must stay byte-identical to the config of record.
    static bool hint_enabled();

    // Lag-1 expert reuse. The experts this layer just dispatched are a ~40%
    // predictor of the NEXT token on the SAME layer (measured 2026-07-19,
    // lag-1 overlap 0.399 vs 0.023 chance). Recorded at compute(); flushed
    // at the next decode's pre-graph hint site.
    // WP_HINT_REUSE_LAST=1 to arm (default off). WP_HINT_REUSE_PAGES (default
    // 32) caps the flush in expert-pages, soonest layer first, skipping hash
    // layers 0..H (tid2eid is already exact). A layer is all-or-nothing.
    static bool reuse_last_enabled();
    void note_dispatched_experts(int32_t layer,
                                 const std::vector<pipe_expert_assignment> & assignments,
                                 uint32_t n_tokens) noexcept;
    size_t flush_reuse_hints() noexcept;
    void clear_reuse_hints() noexcept;

    // ---- cross-layer predicted prefetch -------------------------------------
    //
    // Register a host f32 copy of blk.<layer>.ffn_gate_inp (w, [n_expert rows x
    // n_embd], the layer's router) and blk.<layer>.ffn_exp_probs_b (b,
    // [n_expert]). LOAD TIME ONLY, same contract as register_hash_layer. The
    // dims must match the dispatcher's n_embd/n_expert or the call throws.
    void register_router_layer(int32_t layer, int32_t n_expert, int32_t n_embd,
                               const float * w, const float * b);

    bool   has_router_oracle()    const { return !routers_.empty(); }
    size_t router_oracle_layers() const { return routers_.size(); }

    // Drop every registered router, disabling predicted hints. All or nothing,
    // for the same reason as clear_hash_oracle().
    void clear_router_oracle() { routers_.clear(); }

    // WP_HINT_ROUTER2=M (default 0 = off, clamp 1..16): at layer L-2,
    // apply layer L's router to the step activations, max-pool scores over
    // token positions, and offer the top-M experts for L as PREDICTED.
    static int router2_topm();
    // WP_PREDICT_MAX_TOKENS=n (default 16): skip capture for wider batches.
    static int predict_max_tokens();

    // Softmax probability floor for router-predicted hints (WP_HINT_ROUTER2_CONF,
    // default 0.10 -- the value the whole-expert pager settled on). 0 = no gate.
    static float router2_conf_min();
    // How many layers ahead to predict (WP_HINT_ROUTER2_K, default 1 = just the
    // L+2 target this path has always had). Depth d gets a HALVED top-M and a
    // RAISED floor, because prediction quality decays with distance and the
    // whole-expert measurements showed the tail is where the wasted bytes are.
    static int   router2_lookahead();
    // Added to the floor per extra layer of depth (WP_HINT_ROUTER2_CONF_STEP,
    // default 0.05).
    static float router2_conf_step();

    // WP_HINT_QUEUE_DEPTH=N (default 0/unset): 0 keeps the legacy one-slot
    // mailbox byte-identical (a snapshot the predictor thread has not yet
    // drained is silently overwritten -- see enqueue_prediction). N>0 gives
    // the spine-to-predictor handoff a bounded FIFO of depth N instead, so a
    // snapshot is dropped only when N are already queued, and that loss is
    // counted (pred_queue_overflow()) instead of silent. Depth does not
    // change which snapshot is scored first within a decode (see
    // pred_snapshot_taken_) -- only whether a scored-but-not-yet-drained
    // snapshot from an earlier decode can survive to be drained instead of
    // being clobbered.
    static size_t pred_queue_depth();

    size_t n_workers() const;
    bool failed() const noexcept;
    std::string failure_message() const;
    void begin_decode() noexcept;
    void end_decode() noexcept;
    uint64_t decode_dispatch_ns() const noexcept { return decode_ns_total_; }

    struct spine_profile_stats {
        uint64_t ns_dispatch_issue_total = 0;
        uint64_t ns_dispatch_wait_total = 0;
        uint64_t ns_between_issue_and_wait_total = 0;
        uint64_t ns_before_first_issue = 0;
        uint64_t ns_after_last_wait = 0;
        uint64_t ns_gaps = 0;
        size_t n_layers = 0;
    };

    void spine_profile_begin(std::chrono::steady_clock::time_point begin) noexcept;
    spine_profile_stats spine_profile_end(std::chrono::steady_clock::time_point end) noexcept;

  private:
    struct op_context;

    struct router_layer {
        std::vector<float> w;   // [n_expert][n_embd], expert-major
        std::vector<float> b;   // [n_expert]
    };

    void latch_failure(const char * message) noexcept;
    void layer_trace_issue_return(int32_t layer, std::chrono::steady_clock::time_point time) noexcept;
    void layer_trace_wait_entry(int32_t layer, std::chrono::steady_clock::time_point time) noexcept;
    void spine_profile_issue_begin(std::chrono::steady_clock::time_point time) noexcept;
    void spine_profile_issue_end(int32_t layer, std::chrono::steady_clock::time_point time) noexcept;
    void spine_profile_wait_begin(int32_t layer, std::chrono::steady_clock::time_point time) noexcept;
    void spine_profile_wait_end(std::chrono::steady_clock::time_point time) noexcept;
    void write_layer_trace(int32_t layer) noexcept;
    bool is_phantom_row(int64_t token) const noexcept;
    void zero_phantom_rows(std::vector<float> & result, int64_t n_tokens, int64_t n_embd,
                           int64_t token_offset = 0) const noexcept;

    // Predicted-hint pipeline, two halves so the router GEMM never touches the
    // dispatch critical path (measured 2026-08-07: synchronous scoring cost
    // +9.4s/run against -3.0s of worker wait bought -- the hints pay only if
    // the math is free):
    //
    // FIRST SNAPSHOT WINS. enqueue_prediction is called once per MoE layer;
    // only the first call this decode is scored. Later layers are dropped on
    // purpose so K lookahead is from the START of the decode (soonest
    // consume), not from whichever layer the scorer happened to be free for.
    // Latest-wins was a race: 40-55% of snapshots dropped, and the survivors
    // were middle/late layers with no lead (measured 2026-08-19).
    // The scorer applies routers layer+2 .. layer+1+K and parks top-M unions.
    // flush_predicted_hints() sends soonest-first up to WP_HINT_ROUTER2_PAGES
    // per decode. Neither half ever throws.
    void enqueue_prediction(int32_t layer, const std::vector<float> & activations,
                            int64_t n_tokens) noexcept;
    void flush_predicted_hints() noexcept;
    void predictor_loop();
    // Prefill whole-slice L+1 CERTAIN hints. Advisory; never throws.
    // WP_PREFILL_LAYER_AHEAD=1 and n_tokens > width. Worker catalog path is
    // the fetch engine; these frames keep the spine/worker hint counters in
    // the same A/B and stay off the decode spec queue (worker skips enqueue
    // when the frame is larger than the decode generation cap of 16).
    void prefetch_layer_ahead(int32_t layer, uint32_t n_tokens) noexcept;

    // WP_PREDICT_CAPTURE=<prefix>: append (layer, h, selected experts) records
    // to <prefix>.<pid>.bin -- the dispatch-path routing capture that replaces
    // the pager-only WP_CAPTURE_ROUTING (whose file a GLM run overwrote; the
    // pid suffix is the fopen("w") lesson). Never throws.
    void capture_routing(const char * prefix, int32_t layer,
                         const std::vector<float> & activations,
                         const ggml_tensor * selected_experts,
                         int64_t n_tokens, int64_t n_expert_used) noexcept;

    static void compute(ggml_tensor *       dst,
                        const ggml_tensor * activations,
                        const ggml_tensor * selected_experts,
                        const ggml_tensor * weights,
                        int                 ith,
                        int                 nth,
                        void *              userdata);
    static void compute_issue(ggml_tensor *       dst,
                              const ggml_tensor * activations,
                              const ggml_tensor * selected_experts,
                              const ggml_tensor * weights,
                              int                 ith,
                              int                 nth,
                              void *              userdata);
    static void compute_wait(ggml_tensor *       dst,
                             const ggml_tensor * issued,
                             const ggml_tensor * shexp,
                             int                 ith,
                             int                 nth,
                             void *              userdata);

    dispatcher                                     remote;
    // Hash-layer tid2eid tables. Written only by register_hash_layer() at load
    // time; read-only and reentrant afterwards, which is why it needs no lock.
    hash_oracle                                    oracle_;
    // Scratch for prefetch_for_tokens, reused so a per-decode hint costs no
    // allocation. Not thread safe -- see the "no dispatch in flight" contract.
    std::vector<int32_t>                           hint_experts_;
    // Scratch for the ranked lookup behind the confidence gate. Same
    // no-allocation, dispatch-thread-only contract as hint_experts_.
    std::vector<hash_oracle::ranked_expert>        hint_ranked_;
    // Last expert set hinted per layer. The same token set is now offered from
    // several points in one step (draft start, post-draft, and the ubatch), and
    // without this each would re-send an identical frame that the worker can
    // only discard. Suppressing the repeat costs one compare and keeps
    // hint_stats honest about how much was really offered.
    // Trade-off, on purpose: if a set recurs after its pages were evicted, the
    // repeat is skipped and one prefetch opportunity is lost. Strictly fewer
    // frames, never more reads.
    std::map<int32_t, std::vector<int32_t>>        last_hint_;
    // Router weights for predicted hints. Written only by register_router_layer()
    // at load time; read-only afterwards (same no-lock rationale as oracle_).
    std::map<int32_t, router_layer>                routers_;
    // Last PREDICTED set sent per target layer -- same repeat-suppression
    // trade-off as last_hint_, kept separate because the sets never coincide
    // with the hash-layer CERTAIN ones. Dispatch thread only.
    std::map<int32_t, std::vector<int32_t>>        last_pred_hint_;
    // N-gram hints have their own repeat history so router2 results do not
    // make an unchanged token-table result look new on the next step.
    std::map<int32_t, std::vector<int32_t>>        last_ngram_hint_;
    // Last dispatched expert set per layer, waiting to be offered as PREDICTED
    // for the next decode. Separate from last_pred_hint_ so a reuse frame is
    // not mistaken for a router2 frame (they can coincide, and the worker
    // would discard the second anyway).
    struct reuse_set {
        uint32_t             n_tokens = 1;
        std::vector<int32_t> experts;
    };
    std::map<int32_t, reuse_set>                   last_dispatched_;
    std::map<int32_t, std::vector<int32_t>>        last_reuse_hint_;
    // Scorer thread <-> dispatch thread handoff. pred_mutex_ guards the four
    // fields below; the scorer owns its own scratch.
    std::mutex                                     pred_mutex_;
    std::condition_variable                        pred_cv_;
    // First-snapshot inbox: only the earliest layer this decode is scored.
    // Later enqueues increment pred_dropped_ and leave the inbox alone.
    struct pred_job {
        int32_t            layer    = -1;
        int64_t            n_tokens = 0;
        std::vector<float> activations;
        bool               valid    = false;
    };
    // Legacy one-slot mailbox. Used verbatim (unconditional overwrite, no
    // count) when pred_queue_depth() == 0, for byte-identical default
    // behaviour. When the depth knob is set, pred_queue_ below is used
    // instead and this stays empty.
    pred_job                                       pred_inbox_;
    // Bounded FIFO used when WP_HINT_QUEUE_DEPTH > 0. push_back on
    // enqueue_prediction's dispatch thread, pop_front on the predictor
    // thread; both hold pred_mutex_ while touching it.
    std::deque<pred_job>                           pred_queue_;
    // High-water mark of pred_queue_.size(), i.e. the deepest backlog the
    // predictor ever fell behind by. Read this to size WP_HINT_QUEUE_DEPTH:
    // if the hwm keeps hitting the configured depth, the predictor is
    // structurally too slow for the offer rate at this K, and a deeper queue
    // only trades loss for staleness -- see pred_queue_overflow_.
    size_t                                         pred_queue_hwm_ = 0;
    struct pred_result {
        uint32_t             n_tokens = 0;
        std::vector<int32_t> experts;
    };
    // Ready sets awaiting flush, keyed by target layer (a newer set for the
    // same target overwrites -- same staleness argument).
    std::map<int32_t, pred_result>                 pred_ready_;
    // Expert-pages router2 has already offered this decode. WP_HINT_ROUTER2_PAGES
    // is a PER-DECODE cap (default 16), not per flush -- flush runs once per
    // MoE layer and a per-call cap never bound (measured 2026-08-19: +23k
    // spec_pageins when the 16-page cap reset 43x per token).
    size_t                                         router2_pages_this_decode_ = 0;
    bool                                           pred_snapshot_taken_ = false;
    // Prediction-cadence census (2026-08-19). offered = enqueue_prediction calls
    // (once per layer per forward pass); dropped = snapshots the latest-wins
    // mailbox overwrote before the scorer could take them; scored = snapshots
    // the scorer actually processed. offered - dropped should equal scored.
    // A high drop rate means the predictor is running far below once-per-layer
    // and the layers it does predict are chosen by thread timing.
    std::atomic<uint64_t>                          pred_offered_{0};
    std::atomic<uint64_t>                          pred_dropped_{0};
    std::atomic<uint64_t>                          pred_scored_{0};
    // Queue-path counters (WP_HINT_QUEUE_DEPTH > 0 only; stay 0 otherwise).
    // pred_queue_overflow_ is the loss pred_dropped_ could never see: a
    // snapshot that passed the same-decode first-wins gate but arrived while
    // pred_queue_depth() entries were already queued and undrained. That is
    // the count that should track down to ~0 as the knob is raised to cover
    // the real consumer lag; it does not track K, only depth vs. drain rate.
    std::atomic<uint64_t>                          pred_queue_overflow_{0};

  public:
    // Exposed so the decode path can log the cadence at end_decode.
    uint64_t pred_offered() const { return pred_offered_.load(std::memory_order_relaxed); }
    uint64_t pred_dropped() const { return pred_dropped_.load(std::memory_order_relaxed); }
    uint64_t pred_scored()  const { return pred_scored_.load(std::memory_order_relaxed); }
    uint64_t pred_queue_overflow() const { return pred_queue_overflow_.load(std::memory_order_relaxed); }
    // Snapshot of the high-water mark. Not atomic -- only ever read from the
    // dispatch thread at end_decode(), same thread that (via pred_mutex_)
    // updates it, so a lock here would be for a race that cannot happen on
    // this read; kept simple rather than adding a fifth atomic.
    size_t pred_queue_high_water() const { return pred_queue_hwm_; }

  private:
    // MAD-LAB DS4-Flash pipeline-streams: per-dispatcher socket exclusivity.
    // A single-stream server never needed this -- exactly one thread ever
    // drove a `graph_dispatcher`'s decode, so compute()/compute_issue()/
    // compute_wait() (the ggml custom-op callbacks that do the actual
    // socket I/O -- see below) and the hint-send entry points
    // (prefetch_for_tokens/prefetch_ngram_for_tokens/flush_reuse_hints/
    // flush_predicted_hints, called from llama_context::decode() and from
    // inside compute()/compute_issue() respectively) were implicitly
    // serialized by there being only one caller thread.
    //
    // Pipeline-streams breaks that implicit invariant WITHOUT anyone
    // calling this dispatcher's sockets from two DIFFERENT stream threads:
    // a speculative draft context (ctx_dft) BORROWS its target's dispatcher
    // (src/llama-context.cpp:622-660, params.ctx_other), so ctx_dft2's
    // expert_dispatch IS THE SAME graph_dispatcher OBJECT as ctx_tgt2's.
    // common_speculative_draft() (called from
    // tools/server/server-context.cpp's pre_decode(), which this server
    // runs for BOTH streams sequentially on the MAIN thread) can invoke
    // llama_decode(ctx_dft2, ...) -- see the multiple llama_decode(ctx_dft,
    // ...) call sites in common/speculative.cpp -- which runs THIS SAME
    // dispatcher's compute()/compute_issue()/compute_wait() callbacks on
    // the MAIN thread, while stream_b_thread_ can simultaneously be
    // running llama_decode(ctx_tgt2, ...) against the identical dispatcher
    // object. Two threads, one dispatcher, its own sockets -- exactly the
    // "MUST NOT be called with a dispatch in flight: it writes to the same
    // sockets" hazard prefetch_for_tokens()'s doc comment already named,
    // now reachable via the draft-context path instead of a hint call
    // racing a dispatch on one thread.
    //
    // Fix: one mutex per dispatcher instance, held for the full duration of
    // every entry point that touches `remote` (the socket-owning
    // connection): compute()/compute_issue()/compute_wait() (the real
    // dispatch + publish path) and prefetch_for_tokens()/
    // prefetch_ngram_for_tokens()/flush_reuse_hints()/flush_predicted_hints()
    // (the hint-send path). Recursive because flush_predicted_hints() is
    // called from INSIDE compute()/compute_issue() (same thread, nested) --
    // a plain mutex would self-deadlock there. This does not serialize
    // stream A against stream B (separate graph_dispatcher instances,
    // separate mutexes) -- it only serializes a dispatcher against itself,
    // which is what "per-dispatcher socket exclusivity" means. Correctness
    // over hint throughput: a hint call that loses the race just declines
    // (mirrors the existing in_flight-declines-a-hint pattern) or, for the
    // rarer case of a dispatch itself blocking briefly on a same-dispatcher
    // hint flush already in progress, pays a short wait -- never a torn
    // frame on the wire.
    mutable std::recursive_mutex                  io_mutex_;

    bool                                           pred_stop_ = false;
    std::thread                                    pred_thread_;
    std::atomic<bool>                              pred_thread_started_{ false };
    std::unique_ptr<ngram_hint_table>              ngram_table_;
    int32_t                                        ngram_top_m_ = 0;
    int32_t                                        phantom_token_ = -1;
    std::vector<uint8_t>                           phantom_rows_;
    // WP_PREDICT_CAPTURE stream; opened on first record, closed in the dtor.
    FILE *                                         capture_file_ = nullptr;
    std::atomic<uint64_t>                          next_seq_id{ 1 };
    std::map<int32_t, std::unique_ptr<op_context>> op_contexts;
    ggml_context *                                 graph_build_ctx_ = nullptr;
    int32_t                                         last_chunked_issue_layer_ = -1;
    int32_t                                         last_expanded_wait_layer_ = -1;
    std::atomic<bool>                              failed_{ false };
    mutable std::mutex                             failure_mutex_;
    std::string                                    failure_message_;
    bool                                           collect_stats_ = false;
    FILE *                                         forward_log_ = nullptr;
    FILE *                                         layer_trace_ = nullptr;
    struct layer_trace_record {
        uint64_t dense_ns = 0;
        std::chrono::steady_clock::time_point dense_started{};
        bool dense_active = false;
    };
    std::map<int32_t, layer_trace_record>          layer_traces_;
    struct spine_profile_record {
        std::chrono::steady_clock::time_point begin{};
        std::chrono::steady_clock::time_point issue_started{};
        std::chrono::steady_clock::time_point first_issue{};
        std::chrono::steady_clock::time_point last_wait_end{};
        std::map<int32_t, std::chrono::steady_clock::time_point> issue_ended;
        spine_profile_stats stats{};
        bool active = false;
        bool issue_active = false;
        bool have_first_issue = false;
        bool have_last_wait = false;
    };
    spine_profile_record                            spine_profile_;
    std::chrono::steady_clock::time_point          decode_t0_{};
    bool                                           decode_active_ = false;
    size_t                                         decode_layers_ = 0;
    uint64_t                                       decode_ns_pack_ = 0;
    uint64_t                                       decode_ns_issue_ = 0;
    uint64_t                                       decode_ns_wait_ = 0;
    uint64_t                                       decode_ns_unpack_ = 0;
    uint64_t                                       decode_ns_total_ = 0;
    uint64_t                                       decode_ns_fold_overlapped_ = 0;
    uint64_t                                       decode_n_partials_folded_early_ = 0;
    uint64_t                                       decode_first_await_in_flight_ = 0;
    // Max over the decode's layers of dispatch_stats::max_in_flight -- see the
    // comment on that field. Distinct from decode_first_await_in_flight_'s
    // average: this is a peak-of-peaks, so it catches a single layer's
    // cross-layer overlap even if most layers show none.
    size_t                                         decode_max_in_flight_ = 0;
    // n_tokens of the ubatch this decode call is serving: >1 = prefill, 1 = decode.
    // Recorded in compute() (where n_tokens is already in hand) rather than passed
    // through begin_decode(), so the RAII scope in llama-context.cpp is untouched.
    uint32_t                                       decode_n_tokens_ = 0;
    std::vector<worker_dispatch_stats>             decode_workers_;
};

}  // namespace pipe_expert_dispatcher
