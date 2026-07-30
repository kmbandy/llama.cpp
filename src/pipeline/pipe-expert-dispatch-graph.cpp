#include "pipe-expert-dispatch-graph.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <charconv>
#include <map>
#include <set>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace pipe_expert_dispatcher {
namespace {

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
};

graph_dispatcher::graph_dispatcher(const std::string & endpoints,
                                   int32_t             n_embd,
                                   int32_t             n_ff_exp,
                                   int32_t             n_expert,
                                   int32_t             n_expert_used) :
    remote(parse_endpoints(endpoints)) {
    if (remote.n_embd() != n_embd || remote.n_ff_exp() != n_ff_exp || remote.n_expert() != n_expert ||
        remote.n_expert_used() != n_expert_used) {
        throw std::runtime_error("expert dispatcher workers do not match the model MoE dimensions");
    }
}

graph_dispatcher::~graph_dispatcher() = default;

ggml_tensor * graph_dispatcher::build(ggml_context * ctx,
                                      ggml_tensor *  activations,
                                      ggml_tensor *  selected_experts,
                                      ggml_tensor *  weights,
                                      int32_t        layer) {
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
    return ggml_map_custom3(ctx, activations, selected_experts, weights, compute, 1, context.get());
}

size_t graph_dispatcher::n_workers() const {
    return remote.workers().size();
}

void graph_dispatcher::compute(ggml_tensor *       dst,
                               const ggml_tensor * activations,
                               const ggml_tensor * selected_experts,
                               const ggml_tensor * weights,
                               int                 ith,
                               int                 nth,
                               void *              userdata) {
    if (ith != 0 || nth != 1) {
        throw std::runtime_error("expert dispatch custom op must run as one CPU task");
    }
    op_context * context = static_cast<op_context *>(userdata);
    if (context == nullptr || context->owner == nullptr) {
        throw std::runtime_error("expert dispatch custom op has no dispatcher");
    }

    const int64_t n_embd        = activations->ne[0];
    const int64_t n_tokens      = activations->ne[1];
    const int64_t n_expert_used = selected_experts->ne[0];
    const bool shapes_match =
        n_embd == context->owner->remote.n_embd() && n_tokens > 0 &&
        activations->ne[2] == 1 && activations->ne[3] == 1 &&
        selected_experts->ne[1] == n_tokens &&
        selected_experts->ne[2] == 1 && selected_experts->ne[3] == 1 &&
        weights->ne[0] == 1 && weights->ne[1] == n_expert_used &&
        weights->ne[2] == n_tokens && weights->ne[3] == 1 &&
        ggml_are_same_shape(dst, activations);
    if (!shapes_match) {
        throw std::runtime_error("expert dispatch custom op input shapes do not match");
    }

    std::vector<uint16_t> wire_activations((size_t) n_embd * (size_t) n_tokens);
    for (size_t i = 0; i < wire_activations.size(); ++i) {
        wire_activations[i] = (uint16_t) ggml_fp32_to_fp16(ggml_get_f32_1d(activations, (int) i));
    }

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

    graph_dispatcher *       owner  = context->owner;
    const uint64_t           seq_id = owner->next_seq_id.fetch_add(1, std::memory_order_relaxed);
    const std::vector<float> result =
        owner->remote.dispatch(context->layer, seq_id, (uint32_t) n_tokens, wire_activations, assignments);
    for (size_t i = 0; i < result.size(); ++i) {
        ggml_set_f32_1d(dst, (int) i, result[i]);
    }
}

}  // namespace pipe_expert_dispatcher
