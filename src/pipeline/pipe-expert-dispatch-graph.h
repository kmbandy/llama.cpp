#pragma once

#include "pipe-expert-dispatcher.h"
#include "pipe-hash-oracle.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
                     int32_t             last_no_defer_layer = -1);
    ~graph_dispatcher();

    graph_dispatcher(const graph_dispatcher &)             = delete;
    graph_dispatcher & operator=(const graph_dispatcher &) = delete;

    ggml_tensor * build(ggml_context * ctx,
                        ggml_tensor *  activations,
                        ggml_tensor *  selected_experts,
                        ggml_tensor *  weights,
                        int32_t        layer,
                        float          swiglu_clamp);

    // Split of build() so a sibling GPU op (shared expert) can sit between
    // the worker send and the worker recv. build_issue sends; after_issue
    // adds a 0-valued dependence on that send; build_wait recvs and
    // produces the MoE residual. build() remains the combined path.
    ggml_tensor * build_issue(ggml_context * ctx,
                              ggml_tensor *  activations,
                              ggml_tensor *  selected_experts,
                              ggml_tensor *  weights,
                              int32_t        layer,
                              float          swiglu_clamp);
    ggml_tensor * after_issue(ggml_context * ctx, ggml_tensor * tensor, int32_t layer);
    ggml_tensor * build_wait(ggml_context * ctx, ggml_tensor * shexp, int32_t layer);

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

    size_t n_workers() const;
    bool failed() const noexcept;
    std::string failure_message() const;
    void begin_decode() noexcept;
    void end_decode() noexcept;

    // Scheduler callbacks in llama-context.cpp delimit the dense interval
    // between the split dispatch issue and wait nodes.
    void layer_trace_dense_begin(int32_t layer) noexcept;
    void layer_trace_dense_end(int32_t layer) noexcept;

  private:
    struct op_context;

    struct router_layer {
        std::vector<float> w;   // [n_expert][n_embd], expert-major
        std::vector<float> b;   // [n_expert]
    };

    void latch_failure(const char * message) noexcept;
    void write_layer_trace(int32_t layer) noexcept;

    // Predicted-hint pipeline, two halves so the router GEMM never touches the
    // dispatch critical path (measured 2026-08-07: synchronous scoring cost
    // +9.4s/run against -3.0s of worker wait bought -- the hints pay only if
    // the math is free):
    //
    // enqueue_prediction() copies this layer's dispatch activations into a
    // latest-wins slot for the scorer thread and returns immediately. The
    // scorer applies router_{layer+2}, max-pools over the token positions, and
    // parks the top-M result. flush_predicted_hints() -- called from compute() on
    // the dispatch thread BEFORE dispatch, where the sockets are quiet
    // (in_flight == 0 under WP_DEFER_K=0) -- sends whatever is parked. The
    // hint therefore goes out at the NEXT layer's dispatch: one layer of the
    // two-layer lead is spent on the handoff.
    // Neither half ever throws: a hint carries no correctness weight, so no
    // predictor failure may latch the dispatcher.
    void enqueue_prediction(int32_t layer, const std::vector<float> & activations,
                            int64_t n_tokens) noexcept;
    void flush_predicted_hints() noexcept;
    void predictor_loop();

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
    // Scorer thread <-> dispatch thread handoff. pred_mutex_ guards the four
    // fields below; the scorer owns its own scratch.
    std::mutex                                     pred_mutex_;
    std::condition_variable                        pred_cv_;
    // Latest-wins inbox: a stale activation snapshot is worthless (its target
    // layer is about to be dispatched anyway), so a new enqueue overwrites.
    struct pred_job {
        int32_t            layer    = -1;
        int64_t            n_tokens = 0;
        std::vector<float> activations;
        bool               valid    = false;
    };
    pred_job                                       pred_inbox_;
    struct pred_result {
        uint32_t             n_tokens = 0;
        std::vector<int32_t> experts;
    };
    // Ready sets awaiting flush, keyed by target layer (a newer set for the
    // same target overwrites -- same staleness argument).
    std::map<int32_t, pred_result>                 pred_ready_;
    // Prediction-cadence census (2026-08-19). offered = enqueue_prediction calls
    // (once per layer per forward pass); dropped = snapshots the latest-wins
    // mailbox overwrote before the scorer could take them; scored = snapshots
    // the scorer actually processed. offered - dropped should equal scored.
    // A high drop rate means the predictor is running far below once-per-layer
    // and the layers it does predict are chosen by thread timing.
    std::atomic<uint64_t>                          pred_offered_{0};
    std::atomic<uint64_t>                          pred_dropped_{0};
    std::atomic<uint64_t>                          pred_scored_{0};

  public:
    // Exposed so the decode path can log the cadence at end_decode.
    uint64_t pred_offered() const { return pred_offered_.load(std::memory_order_relaxed); }
    uint64_t pred_dropped() const { return pred_dropped_.load(std::memory_order_relaxed); }
    uint64_t pred_scored()  const { return pred_scored_.load(std::memory_order_relaxed); }

  private:
    bool                                           pred_stop_ = false;
    std::thread                                    pred_thread_;
    std::atomic<bool>                              pred_thread_started_{ false };
    std::unique_ptr<ngram_hint_table>              ngram_table_;
    int32_t                                        ngram_top_m_ = 0;
    // WP_PREDICT_CAPTURE stream; opened on first record, closed in the dtor.
    FILE *                                         capture_file_ = nullptr;
    std::atomic<uint64_t>                          next_seq_id{ 1 };
    std::map<int32_t, std::unique_ptr<op_context>> op_contexts;
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
    std::chrono::steady_clock::time_point          decode_t0_{};
    bool                                           decode_active_ = false;
    size_t                                         decode_layers_ = 0;
    uint64_t                                       decode_ns_pack_ = 0;
    uint64_t                                       decode_ns_issue_ = 0;
    uint64_t                                       decode_ns_wait_ = 0;
    uint64_t                                       decode_ns_unpack_ = 0;
    uint64_t                                       decode_ns_total_ = 0;
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
