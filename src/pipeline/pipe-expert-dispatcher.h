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
    uint64_t    ns_wait = 0;
    uint64_t    n_requests = 0;
    uint64_t    n_experts_total = 0;
};

struct dispatch_stats {
    size_t                             workers_used          = 0;
    size_t                             requests_issued       = 0;
    size_t                             first_await_in_flight = 0;
    bool                               first_await_recorded  = false;
    uint64_t                           ns_pack               = 0;
    uint64_t                           ns_issue              = 0;
    uint64_t                           ns_wait               = 0;
    uint64_t                           ns_unpack             = 0;
    uint64_t                           ns_total              = 0;
    std::vector<worker_dispatch_stats> workers;
};

// Cumulative prefetch-hint counters. MECHANISM counters, not outcome ones:
// they answer "did the spine actually offer work" and nothing more. The outcome
// lives on the worker (n_pagein / bytes_read), because only the worker knows
// whether a hinted expert was already resident.
struct prefetch_hint_stats {
    // Hint frames handed to the transport.
    uint64_t n_frames = 0;
    // Expert ids across those frames (a frame carries >= 1).
    uint64_t n_experts = 0;
    // Frames the transport refused. A hint is advisory, so a failed send is
    // counted and swallowed rather than thrown -- but a nonzero value here means
    // every measurement in the run is understating what was offered.
    uint64_t n_send_failed = 0;
    // Calls that produced nothing because the layer has no tid2eid table.
    uint64_t n_no_oracle = 0;
    // Calls skipped because worker choice is not predictable this run
    // (WP_DISPATCH_STATIC_ASSIGN=0). Hinting the wrong worker costs a read on
    // the worker that was hinted AND a cold read on the one that was used, so
    // this path declines rather than guesses.
    uint64_t n_skipped_dynamic = 0;
    // Calls declined because a request was still outstanding on the sockets
    // (WP_DEFER_K > 0). Interleaving a hint there would desynchronise a stream
    // whose reader matches responses by seq_id.
    uint64_t n_skipped_in_flight = 0;
};

// Cumulative expert-deferral mechanism counters (spec section 4).
struct deferral_stats {
    // Experts placed in the deferred set (cumulative over the process).
    uint64_t n_deferred = 0;
    // Deferred partials that arrived after their fold point (bug counter).
    uint64_t n_deferred_late = 0;
    // Nanoseconds spent with zero requests in flight (gap time).
    uint64_t ns_gap = 0;
    // Device busy fraction over the last begin_decode/end_decode window,
    // from /proc/diskstats io_ticks. -1 if unavailable.
    double nvme_util_pct = -1.0;
    // Runtime gate value: number of experts computed immediately per token.
    // 0 means feature off (defer nothing).
    int defer_k = 0;
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
    //
    // When WP_DEFER_K > 0 and this is not the last routed layer, the lowest
    // weight experts are issued but not awaited; their partials are folded
    // into the next dispatch's return value. Deferred reads are always issued
    // before this call returns.
    // swiglu_clamp: hparams.swiglu_clamp_exp[layer], <= 0 = no clamp. Travels
    // with every request because the spine's clamped SwiGLU is unreachable on
    // the dispatch path. See pipe_expert_dispatch_req::swiglu_clamp.
    std::vector<float> dispatch(int32_t                                     layer,
                                uint64_t                                    seq_id,
                                uint32_t                                    n_tokens,
                                const std::vector<float> &                  activations,
                                const std::vector<pipe_expert_assignment> & assignments,
                                float                                       swiglu_clamp);

    // Offer `experts` on `layer` to the workers that will actually be asked for
    // them, as PIPE_EXPERT_PREFETCH_HINT frames. `experts` must be ascending and
    // unique (what hash_oracle::experts_for produces).
    //
    // ADVISORY, NOT A REQUEST. No response is awaited, in_flight is untouched,
    // and a transport failure is counted rather than thrown -- a dropped hint
    // costs a page-in the run was going to pay anyway, so it must never be able
    // to fail a decode. Safe to call between dispatches; never call it while a
    // request is in flight on the same socket.
    //
    // Returns the number of frames sent.
    size_t send_prefetch_hints(int32_t layer, const std::vector<int32_t> & experts,
                               uint32_t provenance = PIPE_HINT_CERTAIN);

    const prefetch_hint_stats & get_prefetch_hint_stats() const;

    int32_t                          n_embd() const;
    int32_t                          n_ff_exp() const;
    int32_t                          n_expert() const;
    int32_t                          n_expert_used() const;
    const std::string &              model_identity() const;
    const std::vector<worker_info> & workers() const;

    size_t                 in_flight_requests() const;
    const dispatch_stats & last_dispatch_stats() const;
    const deferral_stats & get_deferral_stats() const;
    int                    defer_k() const;
    int32_t                last_no_defer_layer() const;

    // Layer that must not leave deferred work pending (no successor in the
    // main graph to fold into). Pass hparams.n_layer()-1 from the host so
    // NextN/MTP layers advertised by workers are not treated as the fold
    // successor of the main stack. -1 keeps the worker-advertised max.
    void set_last_no_defer_layer(int32_t layer) noexcept;

    // Sample NVMe util window and reset per-window gap accounting.
    void begin_deferral_window() noexcept;
    void end_deferral_window() noexcept;

    // Drain any still-pending deferred partials (e.g. end of decode). Counts
    // them as late if they missed their fold point. Returns the folded sum
    // (may be empty if nothing was pending).
    std::vector<float> drain_deferred();

  private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

}  // namespace pipe_expert_dispatcher
