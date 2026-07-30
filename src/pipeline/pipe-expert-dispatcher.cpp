#include "pipe-expert-dispatcher.h"

#include "ggml.h"
#include "pipe-transport.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pipe_expert_dispatcher {
namespace {

std::string endpoint_label(const endpoint & value) {
    return value.host + ":" + std::to_string(value.port);
}

std::string assignment_experts(const std::vector<pipe_expert_assignment> & assignments) {
    std::ostringstream stream;
    for (size_t i = 0; i < assignments.size(); ++i) {
        if (i != 0) {
            stream << ",";
        }
        stream << assignments[i].expert_id;
    }
    return stream.str();
}

}  // namespace

struct dispatcher::impl {
    struct worker {
        endpoint                                 target;
        worker_info                              info;
        pipe_expert_hello                        hello;
        pipe_socket_ptr                          socket;
        std::vector<std::pair<int32_t, int32_t>> resident_lru;
    };

    struct planned_request {
        size_t                              worker_index = 0;
        std::vector<pipe_expert_assignment> assignments;
        std::vector<uint8_t>                payload;
    };

    std::vector<worker>                                 workers;
    std::vector<worker_info>                            public_workers;
    std::map<int32_t, std::vector<std::vector<size_t>>> routes;
    std::map<std::string, size_t>                       machine_cursor;
    dispatch_stats                                      stats;
    size_t                                              in_flight     = 0;
    int32_t                                             n_embd        = 0;
    int32_t                                             n_ff_exp      = 0;
    int32_t                                             n_expert      = 0;
    int32_t                                             n_expert_used = 0;
    std::string                                         model_identity;
    bool                                                usable = true;

    explicit impl(const std::vector<endpoint> & endpoints) {
        if (endpoints.empty()) {
            throw std::invalid_argument("expert dispatcher requires at least one worker endpoint");
        }
        if (!pipe_transport_init()) {
            throw std::runtime_error("expert dispatcher failed to initialize TCP transport");
        }

        std::set<std::string> seen_endpoints;
        for (endpoint target : endpoints) {
            if (target.host.empty() || target.port <= 0 || target.port > 65535) {
                throw std::invalid_argument("expert dispatcher has an invalid worker endpoint");
            }
            if (target.machine.empty()) {
                target.machine = target.host;
            }
            const std::string label = endpoint_label(target);
            if (!seen_endpoints.insert(label).second) {
                throw std::invalid_argument("expert dispatcher repeats worker " + label);
            }

            worker connected;
            connected.target        = target;
            connected.info.endpoint = label;
            connected.info.machine  = target.machine;
            connected.socket        = pipe_socket_t::connect(target.host.c_str(), target.port);
            if (!connected.socket) {
                throw std::runtime_error("expert dispatcher failed to connect to worker " + label);
            }

            pipe_frame_type      type;
            uint64_t             seq_id = 0;
            std::vector<uint8_t> payload;
            if (!pipe_recv_frame(*connected.socket, type, seq_id, payload)) {
                throw std::runtime_error("expert dispatcher worker " + label + " died before HELLO");
            }
            if (type == PIPE_ERROR) {
                const pipe_error error = pipe_decode_error(payload.data(), payload.size());
                throw std::runtime_error("expert dispatcher worker " + label + " rejected HELLO: " + error.msg);
            }
            if (type != PIPE_HELLO || seq_id != 0) {
                throw std::runtime_error("expert dispatcher worker " + label + " sent an invalid HELLO frame");
            }
            try {
                connected.hello = pipe_decode_expert_hello(payload.data(), payload.size());
            } catch (const std::exception & error) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " has an invalid HELLO: " + error.what());
            }
            if (connected.hello.role != PIPE_EXPERT_ROLE_WORKER) {
                throw std::runtime_error("expert dispatcher peer " + label + " is not an expert worker");
            }

            if (workers.empty()) {
                n_embd         = connected.hello.n_embd;
                n_ff_exp       = connected.hello.n_ff_exp;
                n_expert       = connected.hello.n_expert;
                n_expert_used  = connected.hello.n_expert_used;
                model_identity = connected.hello.model_identity;
            } else if (connected.hello.n_embd != n_embd || connected.hello.n_ff_exp != n_ff_exp ||
                       connected.hello.n_expert != n_expert || connected.hello.n_expert_used != n_expert_used ||
                       connected.hello.model_identity != model_identity) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " does not match the first worker's model identity and hparams");
            }

            connected.info.expert_first = connected.hello.expert_first;
            connected.info.expert_last  = connected.hello.expert_last;
            connected.info.n_slots      = connected.hello.n_slots;
            connected.info.layers       = connected.hello.layers;

            pipe_expert_hello client = connected.hello;
            client.role              = PIPE_EXPERT_ROLE_CLIENT;
            client.expert_first      = -1;
            client.expert_last       = -1;
            client.n_slots           = 0;
            client.layers.clear();
            payload = pipe_encode_expert_hello(client);
            if (!pipe_send_frame(*connected.socket, PIPE_HELLO, 0, payload.data(), payload.size())) {
                throw std::runtime_error("expert dispatcher failed to send HELLO to worker " + label);
            }

            if (!pipe_send_frame(*connected.socket, PIPE_PING, 0, nullptr, 0)) {
                throw std::runtime_error("expert dispatcher failed to confirm HELLO with worker " + label);
            }
            if (!pipe_recv_frame(*connected.socket, type, seq_id, payload)) {
                throw std::runtime_error("expert dispatcher worker " + label + " died during HELLO");
            }
            if (type == PIPE_ERROR) {
                const pipe_error error = pipe_decode_error(payload.data(), payload.size());
                throw std::runtime_error("expert dispatcher worker " + label + " rejected HELLO: " + error.msg);
            }
            if (type != PIPE_PONG || seq_id != 0) {
                throw std::runtime_error("expert dispatcher worker " + label + " did not confirm the expert HELLO");
            }

            public_workers.push_back(connected.info);
            workers.push_back(std::move(connected));
        }

        build_routes();
    }

    void build_routes() {
        std::set<int32_t> claimed_layers;
        for (const worker & value : workers) {
            claimed_layers.insert(value.hello.layers.begin(), value.hello.layers.end());
        }
        for (int32_t layer : claimed_layers) {
            std::vector<std::vector<size_t>> layer_routes((size_t) n_expert);
            for (int32_t expert = 0; expert < n_expert; ++expert) {
                std::set<std::string> machines;
                for (size_t i = 0; i < workers.size(); ++i) {
                    const worker & value = workers[i];
                    if (expert < value.hello.expert_first || expert > value.hello.expert_last ||
                        std::find(value.hello.layers.begin(), value.hello.layers.end(), layer) ==
                            value.hello.layers.end()) {
                        continue;
                    }
                    layer_routes[(size_t) expert].push_back(i);
                    machines.insert(value.target.machine);
                }
                if (layer_routes[(size_t) expert].empty()) {
                    throw std::runtime_error("expert dispatcher coverage gap for layer " + std::to_string(layer) +
                                             " expert " + std::to_string(expert));
                }
                if (machines.size() != 1) {
                    throw std::runtime_error("expert dispatcher expert " + std::to_string(expert) + " on layer " +
                                             std::to_string(layer) + " is advertised by more than one machine");
                }
            }
            routes.emplace(layer, std::move(layer_routes));
        }
    }

    bool is_resident(size_t worker_index, int32_t layer, int32_t expert) const {
        const std::pair<int32_t, int32_t>                key(layer, expert);
        const std::vector<std::pair<int32_t, int32_t>> & lru = workers[worker_index].resident_lru;
        return std::find(lru.begin(), lru.end(), key) != lru.end();
    }

    size_t choose_worker(int32_t                     layer,
                         int32_t                     expert,
                         const std::vector<size_t> & candidates,
                         const std::vector<size_t> & assigned_counts) {
        bool any_resident = false;
        for (size_t candidate : candidates) {
            any_resident = any_resident || is_resident(candidate, layer, expert);
        }

        size_t              best_count = (size_t) -1;
        uint32_t            best_slots = 0;
        std::vector<size_t> tied;
        for (size_t candidate : candidates) {
            if (any_resident && !is_resident(candidate, layer, expert)) {
                continue;
            }
            const size_t   count = assigned_counts[candidate];
            const uint32_t slots = workers[candidate].hello.n_slots;
            if (count < best_count || (count == best_count && slots > best_slots)) {
                best_count = count;
                best_slots = slots;
                tied.clear();
                tied.push_back(candidate);
            } else if (count == best_count && slots == best_slots) {
                tied.push_back(candidate);
            }
        }

        const std::string & machine = workers[candidates.front()].target.machine;
        size_t &            cursor  = machine_cursor[machine];
        const size_t        chosen  = tied[cursor % tied.size()];
        ++cursor;
        return chosen;
    }

    void update_residency(size_t worker_index, int32_t layer, const std::vector<pipe_expert_assignment> & assignments) {
        worker & value = workers[worker_index];
        for (const pipe_expert_assignment & assignment : assignments) {
            const std::pair<int32_t, int32_t> key(layer, assignment.expert_id);
            auto found = std::find(value.resident_lru.begin(), value.resident_lru.end(), key);
            if (found != value.resident_lru.end()) {
                value.resident_lru.erase(found);
            }
            value.resident_lru.push_back(key);
            while (value.resident_lru.size() > value.hello.n_slots) {
                value.resident_lru.erase(value.resident_lru.begin());
            }
        }
    }

    void invalidate() {
        usable    = false;
        in_flight = 0;
        for (worker & value : workers) {
            value.socket.reset();
        }
    }

    pipe_frame_type await_response(planned_request & request, uint64_t wanted_seq_id, std::vector<uint8_t> & payload) {
        if (!stats.first_await_recorded) {
            stats.first_await_recorded  = true;
            stats.first_await_in_flight = in_flight;
        }
        pipe_frame_type type;
        uint64_t        seq_id = 0;
        worker &        value  = workers[request.worker_index];
        if (!pipe_recv_frame(*value.socket, type, seq_id, payload)) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " died while computing expert(s) " + assignment_experts(request.assignments));
        }
        if (in_flight == 0) {
            throw std::runtime_error("expert dispatcher in-flight counter underflow");
        }
        --in_flight;
        if (seq_id != wanted_seq_id) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint + " returned sequence " +
                                     std::to_string(seq_id) + " while awaiting " + std::to_string(wanted_seq_id));
        }
        return type;
    }

    std::vector<float> dispatch(int32_t                                     layer,
                                uint64_t                                    seq_id,
                                uint32_t                                    n_tokens,
                                const std::vector<uint16_t> &               activations,
                                const std::vector<pipe_expert_assignment> & assignments) {
        if (!usable) {
            throw std::runtime_error("expert dispatcher cannot be reused after a worker or protocol failure");
        }
        const auto route_it = routes.find(layer);
        if (route_it == routes.end()) {
            throw std::invalid_argument("expert dispatcher has no workers for layer " + std::to_string(layer));
        }
        const uint64_t activation_count = (uint64_t) n_tokens * (uint64_t) n_embd;
        if (n_tokens == 0 || activation_count != activations.size()) {
            throw std::invalid_argument("expert dispatcher activation shape does not match n_tokens and n_embd");
        }
        if (assignments.empty()) {
            throw std::invalid_argument("expert dispatcher requires at least one activated expert");
        }

        std::set<int32_t> seen_experts;
        for (const pipe_expert_assignment & assignment : assignments) {
            if (assignment.expert_id < 0 || assignment.expert_id >= n_expert ||
                !seen_experts.insert(assignment.expert_id).second || assignment.weights.size() != n_tokens) {
                throw std::invalid_argument("expert dispatcher has an invalid or repeated expert assignment");
            }
        }

        std::vector<planned_request> by_worker(workers.size());
        std::vector<size_t>          assigned_counts(workers.size(), 0);
        for (size_t i = 0; i < workers.size(); ++i) {
            by_worker[i].worker_index = i;
        }
        for (const pipe_expert_assignment & assignment : assignments) {
            const std::vector<size_t> & candidates = route_it->second[(size_t) assignment.expert_id];
            const size_t chosen = choose_worker(layer, assignment.expert_id, candidates, assigned_counts);
            by_worker[chosen].assignments.push_back(assignment);
            ++assigned_counts[chosen];
        }

        std::vector<planned_request> requests;
        for (planned_request & request : by_worker) {
            if (request.assignments.empty()) {
                continue;
            }
            pipe_expert_dispatch_req wire_request;
            wire_request.layer       = layer;
            wire_request.n_tokens    = n_tokens;
            wire_request.assignments = request.assignments;
            wire_request.activations = activations;
            request.payload          = pipe_encode_expert_dispatch_req(wire_request);
            requests.push_back(std::move(request));
        }

        stats              = {};
        stats.workers_used = requests.size();
        for (const planned_request & request : requests) {
            stats.workers.push_back({
                workers[request.worker_index].info.endpoint,
                request.assignments.size(),
            });
        }

        try {
            for (planned_request & request : requests) {
                worker & value = workers[request.worker_index];
                if (!pipe_send_frame(*value.socket, PIPE_EXPERT_DISPATCH_REQ, seq_id, request.payload.data(),
                                     request.payload.size())) {
                    throw std::runtime_error("expert dispatcher failed to send expert(s) " +
                                             assignment_experts(request.assignments) + " to worker " +
                                             value.info.endpoint);
                }
                ++in_flight;
                ++stats.requests_issued;
            }

            std::vector<float> result((size_t) activation_count, 0.0f);
            for (planned_request & request : requests) {
                std::vector<uint8_t>  payload;
                const pipe_frame_type type  = await_response(request, seq_id, payload);
                worker &              value = workers[request.worker_index];
                if (type == PIPE_ERROR) {
                    const pipe_error error = pipe_decode_error(payload.data(), payload.size());
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " rejected expert(s) " + assignment_experts(request.assignments) +
                                             " on layer " + std::to_string(layer) + " with code " +
                                             std::to_string(error.code) + ": " + error.msg);
                }
                if (type != PIPE_EXPERT_PARTIAL) {
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned frame type " + std::to_string((uint32_t) type) +
                                             " for expert(s) " + assignment_experts(request.assignments));
                }

                pipe_expert_partial partial;
                try {
                    partial = pipe_decode_expert_partial(payload.data(), payload.size(), n_embd);
                } catch (const std::exception & error) {
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned an invalid partial for expert(s) " +
                                             assignment_experts(request.assignments) + ": " + error.what());
                }
                if (partial.layer != layer || partial.n_tokens != n_tokens || partial.partial.size() != result.size()) {
                    throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                             " returned the wrong partial shape for expert(s) " +
                                             assignment_experts(request.assignments));
                }
                for (size_t i = 0; i < result.size(); ++i) {
                    result[i] += ggml_fp16_to_fp32((ggml_fp16_t) partial.partial[i]);
                }
            }

            for (const planned_request & request : requests) {
                update_residency(request.worker_index, layer, request.assignments);
            }
            return result;
        } catch (...) {
            invalidate();
            throw;
        }
    }
};

dispatcher::dispatcher(const std::vector<endpoint> & endpoints) : pimpl(new impl(endpoints)) {}

dispatcher::~dispatcher() = default;

dispatcher::dispatcher(dispatcher &&) noexcept = default;

dispatcher & dispatcher::operator=(dispatcher &&) noexcept = default;

std::vector<float> dispatcher::dispatch(int32_t                                     layer,
                                        uint64_t                                    seq_id,
                                        uint32_t                                    n_tokens,
                                        const std::vector<uint16_t> &               activations,
                                        const std::vector<pipe_expert_assignment> & assignments) {
    return pimpl->dispatch(layer, seq_id, n_tokens, activations, assignments);
}

int32_t dispatcher::n_embd() const {
    return pimpl->n_embd;
}

int32_t dispatcher::n_expert() const {
    return pimpl->n_expert;
}

int32_t dispatcher::n_expert_used() const {
    return pimpl->n_expert_used;
}

const std::string & dispatcher::model_identity() const {
    return pimpl->model_identity;
}

const std::vector<worker_info> & dispatcher::workers() const {
    return pimpl->public_workers;
}

size_t dispatcher::in_flight_requests() const {
    return pimpl->in_flight;
}

const dispatch_stats & dispatcher::last_dispatch_stats() const {
    return pimpl->stats;
}

}  // namespace pipe_expert_dispatcher
