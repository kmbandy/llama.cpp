#pragma once

#include "pipe-expert-dispatcher.h"

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
