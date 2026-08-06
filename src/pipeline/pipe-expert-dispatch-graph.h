#pragma once

#include "pipe-expert-dispatcher.h"
#include "pipe-hash-oracle.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
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

    size_t n_workers() const;
    bool failed() const noexcept;
    std::string failure_message() const;
    void begin_decode() noexcept;
    void end_decode() noexcept;

  private:
    struct op_context;

    void latch_failure(const char * message) noexcept;

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
