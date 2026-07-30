#pragma once

#include "pipe-expert-dispatcher.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>

struct ggml_context;
struct ggml_tensor;

namespace pipe_expert_dispatcher {

class graph_dispatcher {
  public:
    graph_dispatcher(const std::string & endpoints,
                     int32_t             n_embd,
                     int32_t             n_ff_exp,
                     int32_t             n_expert,
                     int32_t             n_expert_used);
    ~graph_dispatcher();

    graph_dispatcher(const graph_dispatcher &)             = delete;
    graph_dispatcher & operator=(const graph_dispatcher &) = delete;

    ggml_tensor * build(ggml_context * ctx,
                        ggml_tensor *  activations,
                        ggml_tensor *  selected_experts,
                        ggml_tensor *  weights,
                        int32_t        layer);

    size_t n_workers() const;

  private:
    struct op_context;

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
};

}  // namespace pipe_expert_dispatcher
