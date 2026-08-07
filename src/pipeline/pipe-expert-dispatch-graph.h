#pragma once

#include "pipe-expert-dispatcher.h"
#include "pipe-hash-oracle.h"

#include <atomic>
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
    size_t prefetch_for_tokens(const int32_t * tokens, size_t n_tokens,
                               size_t n_certain = SIZE_MAX);

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

    // WP_PREDICT_AHEAD=k (default 0 = off): at layer L, apply layer L+k's
    // router to L's dispatch activations and hint the union of each token's
    // top-M experts with PREDICTED provenance -- a constant k-layer lead for
    // every hinted layer at one router GEMM per dispatched layer. The
    // approximation error is the h_L -> h_{L+k} drift (measured 2026-07-19:
    // rank-1 0.973 at k=1); k>=2 decay is measured from WP_PREDICT_CAPTURE.
    static int predict_ahead();
    // WP_PREDICT_TOPM=m (default 3, clamp 1..8): experts kept per token.
    static int predict_topm();
    // WP_PREDICT_MAX_TOKENS=n (default 16): skip prediction AND capture for
    // wider batches. This is the prefill gate -- speculative reads during the
    // prefill sweep are pure contention (LATE 84-100%), and a 2048-row router
    // GEMM per layer is not free either.
    static int predict_max_tokens();

    size_t n_workers() const;
    bool failed() const noexcept;
    std::string failure_message() const;
    void begin_decode() noexcept;
    void end_decode() noexcept;

  private:
    struct op_context;

    struct router_layer {
        std::vector<float> w;   // [n_expert][n_embd], expert-major
        std::vector<float> b;   // [n_expert]
    };

    void latch_failure(const char * message) noexcept;

    // Predicted-hint pipeline, two halves so the router GEMM never touches the
    // dispatch critical path (measured 2026-08-07: synchronous scoring cost
    // +9.4s/run against -3.0s of worker wait bought -- the hints pay only if
    // the math is free):
    //
    // enqueue_prediction() copies this layer's dispatch activations into a
    // latest-wins slot for the scorer thread and returns immediately. The
    // scorer applies router_{layer+k}, takes the per-token top-M union, and
    // parks the result. flush_predicted_hints() -- called from compute() on
    // the dispatch thread BEFORE dispatch, where the sockets are quiet
    // (in_flight == 0 under WP_DEFER_K=0) -- sends whatever is parked. The
    // hint therefore goes out at the NEXT layer's dispatch: one layer of the
    // k-layer lead is spent on the handoff; pick WP_PREDICT_AHEAD accordingly.
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

    dispatcher                                     remote;
    // Hash-layer tid2eid tables. Written only by register_hash_layer() at load
    // time; read-only and reentrant afterwards, which is why it needs no lock.
    hash_oracle                                    oracle_;
    // Scratch for prefetch_for_tokens, reused so a per-decode hint costs no
    // allocation. Not thread safe -- see the "no dispatch in flight" contract.
    std::vector<int32_t>                           hint_experts_;
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
    // Ready sets awaiting flush, keyed by target layer (a newer set for the
    // same target overwrites -- same staleness argument).
    std::map<int32_t, std::vector<int32_t>>        pred_ready_;
    bool                                           pred_stop_ = false;
    std::thread                                    pred_thread_;
    std::atomic<bool>                              pred_thread_started_{ false };
    // WP_PREDICT_CAPTURE stream; opened on first record, closed in the dtor.
    FILE *                                         capture_file_ = nullptr;
    std::atomic<uint64_t>                          next_seq_id{ 1 };
    std::map<int32_t, std::unique_ptr<op_context>> op_contexts;
    std::atomic<bool>                              failed_{ false };
    mutable std::mutex                             failure_mutex_;
    std::string                                    failure_message_;
    bool                                           collect_stats_ = false;
    bool                                           decode_active_ = false;
    size_t                                         decode_layers_ = 0;
    uint64_t                                       decode_ns_pack_ = 0;
    uint64_t                                       decode_ns_issue_ = 0;
    uint64_t                                       decode_ns_wait_ = 0;
    uint64_t                                       decode_ns_unpack_ = 0;
    uint64_t                                       decode_ns_total_ = 0;
    uint64_t                                       decode_first_await_in_flight_ = 0;
    // n_tokens of the ubatch this decode call is serving: >1 = prefill, 1 = decode.
    // Recorded in compute() (where n_tokens is already in hand) rather than passed
    // through begin_decode(), so the RAII scope in llama-context.cpp is untouched.
    uint32_t                                       decode_n_tokens_ = 0;
    std::vector<worker_dispatch_stats>             decode_workers_;
};

}  // namespace pipe_expert_dispatcher
