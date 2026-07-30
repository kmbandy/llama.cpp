#pragma once

#include "pipe-protocol.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pipe_expert_dispatcher {

struct endpoint {
    std::string host;
    int         port = 0;
    // The machine key groups workers that share shard storage. An empty key
    // defaults to host.
    std::string machine;
};

struct worker_info {
    std::string          endpoint;
    std::string          machine;
    int32_t              expert_first = -1;
    int32_t              expert_last  = -1;
    uint32_t             n_slots      = 0;
    std::vector<int32_t> layers;
    std::string          shard_identity;
};

struct worker_dispatch_stats {
    std::string endpoint;
    size_t      n_experts = 0;
};

struct dispatch_stats {
    size_t                             workers_used          = 0;
    size_t                             requests_issued       = 0;
    size_t                             first_await_in_flight = 0;
    bool                               first_await_recorded  = false;
    std::vector<worker_dispatch_stats> workers;
};

class dispatcher {
  public:
    explicit dispatcher(const std::vector<endpoint> & endpoints);
    ~dispatcher();

    dispatcher(const dispatcher &)             = delete;
    dispatcher & operator=(const dispatcher &) = delete;
    dispatcher(dispatcher &&) noexcept;
    dispatcher & operator=(dispatcher &&) noexcept;

    // Activations are F16 on the wire. Router weights remain F32, and the
    // returned reduced block is F32. Calls on one dispatcher are serialized.
    std::vector<float> dispatch(int32_t                                     layer,
                                uint64_t                                    seq_id,
                                uint32_t                                    n_tokens,
                                const std::vector<uint16_t> &               activations,
                                const std::vector<pipe_expert_assignment> & assignments);

    int32_t                          n_embd() const;
    int32_t                          n_expert() const;
    int32_t                          n_expert_used() const;
    const std::string &              model_identity() const;
    const std::vector<worker_info> & workers() const;

    size_t                 in_flight_requests() const;
    const dispatch_stats & last_dispatch_stats() const;

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

}  // namespace pipe_expert_dispatcher
