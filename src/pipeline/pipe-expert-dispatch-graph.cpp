#include "pipe-expert-dispatch-graph.h"

#include "ggml-cpu.h"
#include "ggml.h"
#include "llama-impl.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace pipe_expert_dispatcher {
namespace {

using dispatch_clock = std::chrono::steady_clock;

bool dispatch_stats_enabled() {
    const char * value = std::getenv("WP_DISPATCH_STATS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

uint64_t elapsed_ns(dispatch_clock::time_point begin, dispatch_clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

int parse_port(const std::string & text) {
    int        port   = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), port);
    if (parsed.ec != std::errc() || parsed.ptr != text.data() + text.size() || port <= 0 || port > 65535) {
        throw std::invalid_argument("--expert-dispatch expects host:port[,host:port...]");
    }
    return port;
}

std::vector<endpoint> parse_endpoints(const std::string & text) {
    std::vector<endpoint> result;
    size_t                begin = 0;
    while (begin < text.size()) {
        const size_t      comma = text.find(',', begin);
        const std::string item  = text.substr(begin, comma == std::string::npos ? std::string::npos : comma - begin);
        const size_t      colon = item.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == item.size()) {
            throw std::invalid_argument("--expert-dispatch expects host:port[,host:port...]");
        }
        result.push_back({
            item.substr(0, colon),
            parse_port(item.substr(colon + 1)),
            "",
        });
        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    if (result.empty() || begin == text.size()) {
        throw std::invalid_argument("--expert-dispatch expects host:port[,host:port...]");
    }
    return result;
}

}  // namespace

struct graph_dispatcher::op_context {
    graph_dispatcher * owner = nullptr;
    int32_t            layer = -1;
    // hparams.swiglu_clamp_exp[layer]; <= 0 means no clamp. See the note on
    // pipe_expert_dispatch_req::swiglu_clamp -- the spine's own clamped SwiGLU
    // is unreachable on the dispatch path, so the worker must apply it and the
    // limit has to travel with every request.
    float              swiglu_clamp = 0.0f;
};

graph_dispatcher::graph_dispatcher(const std::string & endpoints,
                                   int32_t             n_embd,
                                   int32_t             n_ff_exp,
                                   int32_t             n_expert,
                                   int32_t             n_expert_used,
                                   int32_t             last_no_defer_layer) :
    remote(parse_endpoints(endpoints)),
    collect_stats_(dispatch_stats_enabled()) {
    if (remote.n_embd() != n_embd || remote.n_ff_exp() != n_ff_exp || remote.n_expert() != n_expert ||
        remote.n_expert_used() != n_expert_used) {
        throw std::runtime_error("expert dispatcher workers do not match the model MoE dimensions");
    }

    if (last_no_defer_layer >= 0) {
        remote.set_last_no_defer_layer(last_no_defer_layer);
    }

    // Spec §3: print the ACTUAL runtime value at WARN (logger threshold 3 filters
    // INFO; libllama WARN maps to 2 and passes). A gate whose value you cannot
    // see in the log has cost this project three separate retracted measurement sets.
    // Default 0 = feature off = defer nothing. Do NOT add this to struct Options
    // (ABI mismatch 2026-07-30); env var read at startup only.
    const int k = remote.defer_k();
    LLAMA_LOG_WARN(
        "expert dispatch: WP_DEFER_K=%d (%s; 0=off/defer nothing; K=immediate experts per token) "
        "last_no_defer_layer=%d\n",
        k,
        k <= 0 ? "OFF" : "ON",
        remote.last_no_defer_layer());
}

graph_dispatcher::~graph_dispatcher() {
    // Best-effort drain so a short-lived context does not leave workers hanging.
    try {
        (void) remote.drain_deferred();
    } catch (...) {
        // Destructor must not throw; workers will drop on socket close.
    }
}

ggml_tensor * graph_dispatcher::build(ggml_context * ctx,
                                      ggml_tensor *  activations,
                                      ggml_tensor *  selected_experts,
                                      ggml_tensor *  weights,
                                      int32_t        layer,
                                      float          swiglu_clamp) {
    if (layer < 0) {
        throw std::invalid_argument("expert dispatch requires a non-negative layer");
    }
    if (activations->type != GGML_TYPE_F32 || selected_experts->type != GGML_TYPE_I32 ||
        weights->type != GGML_TYPE_F32) {
        throw std::invalid_argument("expert dispatch requires F32 activations, I32 expert ids, and F32 weights");
    }

    auto & context = op_contexts[layer];
    if (!context) {
        context.reset(new op_context{ this, layer });
    }
    // Assign every call, not just on creation: op_contexts is cached per layer
    // for the lifetime of the dispatcher, so a create-only assignment would pin
    // whatever value the first graph build happened to see.
    context->swiglu_clamp = swiglu_clamp;
    return ggml_map_custom3(ctx, activations, selected_experts, weights, compute, 1, context.get());
}

size_t graph_dispatcher::n_workers() const {
    return remote.workers().size();
}

bool graph_dispatcher::hint_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("WP_PREFETCH_HINT");
        return value != nullptr && value[0] == '1';
    }();
    return enabled;
}

void graph_dispatcher::register_hash_layer(int32_t         layer,
                                           int32_t         n_expert_used,
                                           int32_t         n_vocab,
                                           const int32_t * data) {
    oracle_.register_layer(layer, n_expert_used, n_vocab, remote.n_expert(), data);
}

size_t graph_dispatcher::prefetch_for_tokens(const int32_t * tokens, size_t n_tokens) {
    if (!hint_enabled() || oracle_.empty() || tokens == nullptr || n_tokens == 0) {
        return 0;
    }
    size_t sent = 0;
    for (int32_t layer : oracle_.layers()) {
        if (!oracle_.experts_for(layer, tokens, n_tokens, hint_experts_) ||
            hint_experts_.empty()) {
            continue;
        }
        // Same set as last time for this layer: the worker would resolve it to
        // pages it already holds and discard it. Skip the frame.
        std::vector<int32_t> & previous = last_hint_[layer];
        if (previous == hint_experts_) {
            continue;
        }
        previous = hint_experts_;
        // Swallowing here is deliberate and is the property that makes this safe
        // to leave enabled: a hint carries no correctness weight, so no failure
        // inside it may reach the decode. A broken socket surfaces on the next
        // real dispatch, which is where it belongs.
        try {
            sent += remote.send_prefetch_hints(layer, hint_experts_);
        } catch (...) {
            return sent;
        }
    }
    return sent;
}

void graph_dispatcher::latch_failure(const char * message) noexcept {
    try {
        std::lock_guard<std::mutex> lock(failure_mutex_);
        if (failed_.load(std::memory_order_relaxed)) {
            return;
        }
        failure_message_ = message;
        failed_.store(true, std::memory_order_release);
    } catch (...) {
        failed_.store(true, std::memory_order_release);
    }
}

bool graph_dispatcher::failed() const noexcept {
    return failed_.load(std::memory_order_acquire);
}

std::string graph_dispatcher::failure_message() const {
    std::lock_guard<std::mutex> lock(failure_mutex_);
    return failure_message_;
}

void graph_dispatcher::begin_decode() noexcept {
    remote.begin_deferral_window();
    if (!collect_stats_) {
        return;
    }
    decode_active_                       = true;
    decode_layers_                       = 0;
    decode_ns_pack_                      = 0;
    decode_ns_issue_                     = 0;
    decode_ns_wait_                      = 0;
    decode_ns_unpack_                    = 0;
    decode_ns_total_                     = 0;
    decode_first_await_in_flight_        = 0;
    decode_n_tokens_                     = 0;
    decode_workers_.clear();
    for (const worker_info & worker : remote.workers()) {
        decode_workers_.push_back({ worker.endpoint });
    }
}

void graph_dispatcher::end_decode() noexcept {
    // Last-layer path should have left nothing pending; if anything remains it
    // missed its fold point (bug). Drain and count as late inside the dispatcher.
    try {
        const std::vector<float> leftover = remote.drain_deferred();
        if (!leftover.empty()) {
            LLAMA_LOG_WARN(
                "expert dispatch: drained %zu deferred partial values after decode "
                "(n_deferred_late will reflect this; last layer must not defer)\n",
                leftover.size());
        }
    } catch (const std::exception & error) {
        latch_failure(error.what());
    } catch (...) {
        latch_failure("expert dispatch failed while draining deferred partials");
    }

    remote.end_deferral_window();

    const deferral_stats & dstats = remote.get_deferral_stats();
    // Always log mechanism counters at WARN so they are visible under the
    // llama-server default logger threshold (3); INFO is filtered.
    LLAMA_LOG_WARN(
        "expert dispatch deferral: WP_DEFER_K=%d nvme_util_pct=%.1f ns_gap=%.2f ms "
        "n_deferred=%llu n_deferred_late=%llu\n",
        dstats.defer_k,
        dstats.nvme_util_pct,
        dstats.ns_gap * 1.0e-6,
        (unsigned long long) dstats.n_deferred,
        (unsigned long long) dstats.n_deferred_late);

    // Same treatment for the prefetch hint, and for the same reason: these are
    // the spine's side of the mechanism, and they are what tells a run that
    // "prefetch did nothing" apart from "prefetch was never offered". Silent
    // when the feature is off, so a config-of-record run's log is unchanged.
    if (hint_enabled()) {
        const prefetch_hint_stats & hstats = remote.get_prefetch_hint_stats();
        LLAMA_LOG_WARN(
            "expert dispatch prefetch hint: layers=%zu frames=%llu experts=%llu "
            "send_failed=%llu no_route=%llu skip_dynamic=%llu skip_in_flight=%llu\n",
            oracle_.layers().size(),
            (unsigned long long) hstats.n_frames,
            (unsigned long long) hstats.n_experts,
            (unsigned long long) hstats.n_send_failed,
            (unsigned long long) hstats.n_no_oracle,
            (unsigned long long) hstats.n_skipped_dynamic,
            (unsigned long long) hstats.n_skipped_in_flight);
    }

    if (!collect_stats_ || !decode_active_) {
        return;
    }
    decode_active_ = false;
    if (decode_layers_ == 0) {
        return;
    }

    const double ns_to_ms = 1.0e-6;
    // phase= and n_tokens= (2026-08-03): this block is emitted once per
    // llama_context::decode(), so prefill and decode blocks were previously
    // distinguishable only by their position in the log. With a real prompt and
    // n_ubatch chunking there are several prefill blocks, and any tooling that
    // assumed "block 1 is prefill" silently mis-attributed the rest.
    // ms/tok is the cross-phase-comparable column: prefill amortises, decode does not.
    LLAMA_LOG_WARN(
        "expert dispatch: phase=%s n_tokens=%u layers=%zu pack=%.2f ms issue=%.2f ms wait=%.2f ms "
        "unpack=%.2f ms total=%.2f ms (%.3f ms/tok) first_await_in_flight avg=%.1f (workers=%zu)\n",
        // 3-way: n_tokens > 1 alone would call every speculative-verify batch a
        // prefill, and with DSpark on that is ~97% of them by count.
        decode_n_tokens_ >= 64 ? "PREFILL" : (decode_n_tokens_ > 1 ? "verify" : "decode"),
        decode_n_tokens_,
        decode_layers_,
        decode_ns_pack_ * ns_to_ms,
        decode_ns_issue_ * ns_to_ms,
        decode_ns_wait_ * ns_to_ms,
        decode_ns_unpack_ * ns_to_ms,
        decode_ns_total_ * ns_to_ms,
        decode_ns_total_ * ns_to_ms / (double) std::max<uint32_t>(1, decode_n_tokens_),
        (double) decode_first_await_in_flight_ / (double) decode_layers_,
        n_workers());
    for (const worker_dispatch_stats & worker : decode_workers_) {
        const double avg_wait_ms = worker.n_requests == 0
            ? 0.0
            : worker.ns_wait * ns_to_ms / (double) worker.n_requests;
        LLAMA_LOG_WARN(
            "expert dispatch worker %s requests=%llu experts=%llu wait=%.2f ms (avg %.2f ms/req)\n",
            worker.endpoint.c_str(),
            (unsigned long long) worker.n_requests,
            (unsigned long long) worker.n_experts_total,
            worker.ns_wait * ns_to_ms,
            avg_wait_ms);
    }
}

void graph_dispatcher::compute(ggml_tensor *       dst,
                               const ggml_tensor * activations,
                               const ggml_tensor * selected_experts,
                               const ggml_tensor * weights,
                               int                 ith,
                               int                 nth,
                               void *              userdata) {
    graph_dispatcher * owner = nullptr;
    try {
        if (ith != 0 || nth != 1) {
            throw std::runtime_error("expert dispatch custom op must run as one CPU task");
        }
        op_context * context = static_cast<op_context *>(userdata);
        if (context == nullptr || context->owner == nullptr) {
            throw std::runtime_error("expert dispatch custom op has no dispatcher");
        }
        owner = context->owner;
        if (owner->failed()) {
            ggml_set_zero(dst);
            return;
        }
        const bool collect_stats = owner->collect_stats_ && owner->decode_active_;
        const dispatch_clock::time_point total_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

        const int64_t n_embd        = activations->ne[0];
        const int64_t n_tokens      = activations->ne[1];
        const int64_t n_expert_used = selected_experts->ne[0];
        // Weights must be [1, n_expert_used, n_tokens] — a bare [n_tokens] 1-D
        // tensor PASSES ggml_can_repeat against [n_embd, n_tokens] (because
        // ne[i] % src1->ne[i] == 0) and then broadcasts along the WRONG AXIS.
        const bool shapes_match =
            n_embd == owner->remote.n_embd() && n_tokens > 0 &&
            activations->ne[2] == 1 && activations->ne[3] == 1 &&
            selected_experts->ne[1] == n_tokens &&
            selected_experts->ne[2] == 1 && selected_experts->ne[3] == 1 &&
            weights->ne[0] == 1 && weights->ne[1] == n_expert_used &&
            weights->ne[2] == n_tokens && weights->ne[3] == 1 &&
            ggml_are_same_shape(dst, activations);
        if (!shapes_match) {
            throw std::runtime_error("expert dispatch custom op input shapes do not match");
        }

        // f32 STRAIGHT THROUGH as of PIPE_VERSION 4. This used to round to f16
        // here, which put a ~3e-4 relative error on the input of EVERY routed
        // expert on EVERY layer while attention, the shared expert and the
        // residual all kept f32. See pipe_expert_dispatch_req::activations.
        std::vector<float> wire_activations((size_t) n_embd * (size_t) n_tokens);
        const dispatch_clock::time_point pack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        if (ggml_is_contiguous(activations)) {
            std::memcpy(wire_activations.data(), activations->data,
                        wire_activations.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < wire_activations.size(); ++i) {
                wire_activations[i] = ggml_get_f32_1d(activations, (int) i);
            }
        }
        const dispatch_clock::time_point pack_end =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

        std::map<int32_t, pipe_expert_assignment> by_expert;
        for (int64_t token = 0; token < n_tokens; ++token) {
            std::set<int32_t> token_experts;
            for (int64_t slot = 0; slot < n_expert_used; ++slot) {
                const int     index  = (int) (token * n_expert_used + slot);
                const int32_t expert = ggml_get_i32_1d(selected_experts, index);
                if (!token_experts.insert(expert).second) {
                    throw std::runtime_error("expert dispatch received a repeated expert for one token");
                }
                auto inserted = by_expert.emplace(expert, pipe_expert_assignment{});
                if (inserted.second) {
                    inserted.first->second.expert_id = expert;
                    inserted.first->second.weights.resize((size_t) n_tokens, 0.0f);
                }
                inserted.first->second.weights[(size_t) token] = ggml_get_f32_1d(weights, index);
            }
        }

        std::vector<pipe_expert_assignment> assignments;
        assignments.reserve(by_expert.size());
        for (auto & entry : by_expert) {
            assignments.push_back(std::move(entry.second));
        }

        const uint64_t           seq_id = owner->next_seq_id.fetch_add(1, std::memory_order_relaxed);
        // dispatch() issues deferred reads before returning, waits only for
        // immediate experts, and folds the previous layer's deferred partials
        // into the returned block (residual path for layer N+1).
        const std::vector<float> result =
            owner->remote.dispatch(context->layer, seq_id, (uint32_t) n_tokens, wire_activations, assignments,
                                   context->swiglu_clamp);
        const dispatch_stats & layer_stats = owner->remote.last_dispatch_stats();
        const dispatch_clock::time_point unpack_start =
            collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
        if (ggml_is_contiguous(dst) && dst->type == GGML_TYPE_F32) {
            std::memcpy(dst->data, result.data(), result.size() * sizeof(float));
        } else {
            for (size_t i = 0; i < result.size(); ++i) {
                ggml_set_f32_1d(dst, (int) i, result[i]);
            }
        }

        if (collect_stats) {
            const dispatch_clock::time_point total_end = dispatch_clock::now();
            ++owner->decode_layers_;
            owner->decode_n_tokens_ = (uint32_t) n_tokens;
            owner->decode_ns_pack_ += elapsed_ns(pack_start, pack_end);
            owner->decode_ns_issue_ += layer_stats.ns_issue;
            owner->decode_ns_wait_ += layer_stats.ns_wait;
            owner->decode_ns_unpack_ += elapsed_ns(unpack_start, total_end);
            owner->decode_ns_total_ += elapsed_ns(total_start, total_end);
            owner->decode_first_await_in_flight_ += layer_stats.first_await_in_flight;
            for (const worker_dispatch_stats & worker : layer_stats.workers) {
                for (worker_dispatch_stats & decode_worker : owner->decode_workers_) {
                    if (decode_worker.endpoint != worker.endpoint) {
                        continue;
                    }
                    decode_worker.ns_wait += worker.ns_wait;
                    decode_worker.n_requests += worker.n_requests;
                    decode_worker.n_experts_total += worker.n_experts_total;
                    break;
                }
            }
        }
    } catch (const std::exception & error) {
        if (owner != nullptr) {
            owner->latch_failure(error.what());
        }
        ggml_set_zero(dst);
    } catch (...) {
        if (owner != nullptr) {
            owner->latch_failure("expert dispatch custom op failed with an unknown exception");
        }
        ggml_set_zero(dst);
    }
}

}  // namespace pipe_expert_dispatcher
