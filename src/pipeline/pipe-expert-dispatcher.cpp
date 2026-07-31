#include "pipe-expert-dispatcher.h"

#include "ggml.h"
#include "pipe-transport.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pipe_expert_dispatcher {
namespace {

using dispatch_clock = std::chrono::steady_clock;

bool dispatch_stats_enabled() {
    const char * value = std::getenv("WP_DISPATCH_STATS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

// WP_DEFER_K = number of experts computed immediately per token.
// Unset / empty / non-positive => feature off (defer nothing).
int parse_wp_defer_k() {
    const char * value = std::getenv("WP_DEFER_K");
    if (value == nullptr || value[0] == '\0') {
        return 0;
    }
    char * end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || parsed < 0 || parsed > 1000000L) {
        return 0;
    }
    return (int) parsed;
}

uint64_t elapsed_ns(dispatch_clock::time_point begin, dispatch_clock::time_point end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
}

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

// Sum of io_ticks (ms) over all whole-disk nvme devices (not partitions).
// Returns false if none found or /proc/diskstats unreadable.
bool sample_nvme_io_ticks(uint64_t & io_ticks_ms_sum) {
    std::ifstream in("/proc/diskstats");
    if (!in) {
        return false;
    }
    io_ticks_ms_sum = 0;
    bool any = false;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream fields(line);
        std::string        major;
        std::string        minor;
        std::string        name;
        fields >> major >> minor >> name;
        if (name.rfind("nvme", 0) != 0) {
            continue;
        }
        // Whole devices look like nvme0n1; partitions look like nvme0n1p1.
        if (name.find('p') != std::string::npos) {
            // Could be nvme0n1p2 — skip partitions. Also reject names with no 'n'
            // handled by the prefix check above.
            const size_t npos = name.find('n');
            if (npos != std::string::npos && name.find('p', npos) != std::string::npos) {
                continue;
            }
        }
        // Fields after name: 11 stats then io_ticks (0-based index 12 of full line tokens).
        uint64_t values[11];
        bool     ok = true;
        for (int i = 0; i < 11; ++i) {
            if (!(fields >> values[i])) {
                ok = false;
                break;
            }
        }
        uint64_t io_ticks = 0;
        if (!ok || !(fields >> io_ticks)) {
            continue;
        }
        io_ticks_ms_sum += io_ticks;
        any = true;
    }
    return any;
}

// Split assignments into immediate (top K by router weight per token) and deferred.
// An expert may appear in both with complementary per-token weight masks.
// When defer_k <= 0, all experts are immediate.
void split_immediate_deferred(const std::vector<pipe_expert_assignment> & assignments,
                              uint32_t                                   n_tokens,
                              int                                        defer_k,
                              std::vector<pipe_expert_assignment> &      immediate,
                              std::vector<pipe_expert_assignment> &      deferred,
                              size_t &                                   n_deferred_count) {
    immediate.clear();
    deferred.clear();
    n_deferred_count = 0;

    if (defer_k <= 0 || assignments.empty()) {
        immediate = assignments;
        return;
    }

    // per_token_immediate[token] = set of expert ids that are immediate for that token
    std::vector<std::set<int32_t>> per_token_immediate((size_t) n_tokens);

    for (uint32_t token = 0; token < n_tokens; ++token) {
        struct ranked {
            int32_t expert_id;
            float   weight;
        };
        std::vector<ranked> ranked_experts;
        ranked_experts.reserve(assignments.size());
        for (const pipe_expert_assignment & assignment : assignments) {
            const float w = assignment.weights[(size_t) token];
            if (w != 0.0f) {
                ranked_experts.push_back({ assignment.expert_id, w });
            }
        }
        std::stable_sort(ranked_experts.begin(), ranked_experts.end(),
                         [](const ranked & a, const ranked & b) {
                             if (a.weight != b.weight) {
                                 return a.weight > b.weight;
                             }
                             return a.expert_id < b.expert_id;
                         });
        const size_t keep = std::min((size_t) defer_k, ranked_experts.size());
        for (size_t i = 0; i < keep; ++i) {
            per_token_immediate[(size_t) token].insert(ranked_experts[i].expert_id);
        }
    }

    for (const pipe_expert_assignment & assignment : assignments) {
        pipe_expert_assignment imm;
        pipe_expert_assignment def;
        imm.expert_id = assignment.expert_id;
        def.expert_id = assignment.expert_id;
        imm.weights.assign((size_t) n_tokens, 0.0f);
        def.weights.assign((size_t) n_tokens, 0.0f);
        bool any_imm = false;
        bool any_def = false;
        for (uint32_t token = 0; token < n_tokens; ++token) {
            const float w = assignment.weights[(size_t) token];
            if (w == 0.0f) {
                continue;
            }
            if (per_token_immediate[(size_t) token].count(assignment.expert_id) != 0) {
                imm.weights[(size_t) token] = w;
                any_imm                     = true;
            } else {
                def.weights[(size_t) token] = w;
                any_def                     = true;
            }
        }
        if (any_imm) {
            immediate.push_back(std::move(imm));
        }
        if (any_def) {
            deferred.push_back(std::move(def));
            ++n_deferred_count;
        }
    }

    // Safety: never leave a token with zero immediate experts when K >= 1.
    // (Can happen if K is set but a token selected fewer than K — already handled
    // by keep = min(K, size). If all went deferred somehow, fall back.)
    if (immediate.empty() && !assignments.empty()) {
        immediate = assignments;
        deferred.clear();
        n_deferred_count = 0;
    }
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
        dispatch_clock::time_point          issued_at;
    };

    // Deferred requests issued at layer N, collected at layer N+1's dispatch.
    struct pending_deferred_batch {
        int32_t                      layer    = -1;
        uint64_t                     seq_id   = 0;
        uint32_t                     n_tokens = 0;
        std::vector<planned_request> requests;
        // Set when the successor layer begins collecting — anything still
        // outstanding after the successor has already returned is late.
        bool                         fold_opened = false;
        bool                         fold_closed = false;
    };

    std::vector<worker>                                 workers;
    std::vector<worker_info>                            public_workers;
    std::map<int32_t, std::vector<std::vector<size_t>>> routes;
    std::map<std::string, size_t>                       machine_cursor;
    dispatch_stats                                      stats;
    deferral_stats                                      deferral;
    pending_deferred_batch                              pending_def;
    size_t                                              in_flight     = 0;
    int32_t                                             n_embd        = 0;
    int32_t                                             n_ff_exp      = 0;
    int32_t                                             n_expert      = 0;
    int32_t                                             n_expert_used = 0;
    int32_t                                             last_routed_layer = -1;
    int                                                 defer_k_value = 0;
    std::string                                         model_identity;
    bool                                                poisoned = false;
    bool                                                collect_stats = false;

    // Gap accounting: time spent with in_flight == 0.
    bool                       gap_at_zero = false;
    dispatch_clock::time_point gap_zero_since{};
    bool                       window_active = false;
    uint64_t                   window_io_ticks_begin = 0;
    dispatch_clock::time_point window_wall_begin{};
    bool                       window_sample_ok = false;

    explicit impl(const std::vector<endpoint> & endpoints) :
        collect_stats(dispatch_stats_enabled()) {
        defer_k_value     = parse_wp_defer_k();
        deferral.defer_k  = defer_k_value;

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

            connected.info.expert_first   = connected.hello.expert_first;
            connected.info.expert_last    = connected.hello.expert_last;
            connected.info.n_slots        = connected.hello.n_slots;
            connected.info.layers         = connected.hello.layers;
            connected.info.shard_identity = connected.hello.shard_identity;

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

            if (!pipe_recv_frame(*connected.socket, type, seq_id, payload)) {
                throw std::runtime_error("expert dispatcher worker " + label + " died during HELLO");
            }
            if (type != PIPE_EXPERT_HELLO_ACK || seq_id != 0) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " sent an invalid expert HELLO acknowledgement");
            }
            pipe_expert_hello_ack ack;
            try {
                ack = pipe_decode_expert_hello_ack(payload.data(), payload.size());
            } catch (const std::exception & error) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " sent an invalid expert HELLO acknowledgement: " + error.what());
            }
            if (!ack.accepted) {
                throw std::runtime_error("expert dispatcher worker " + label +
                                         " rejected HELLO: " + ack.reason);
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
        last_routed_layer = claimed_layers.empty() ? -1 : *claimed_layers.rbegin();
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

    void note_in_flight_delta(int delta) {
        if (delta > 0) {
            if (gap_at_zero) {
                deferral.ns_gap += elapsed_ns(gap_zero_since, dispatch_clock::now());
                gap_at_zero = false;
            }
            in_flight += (size_t) delta;
            return;
        }
        if (delta < 0) {
            const size_t dec = (size_t) (-delta);
            if (in_flight < dec) {
                throw std::runtime_error("expert dispatcher in-flight counter underflow");
            }
            in_flight -= dec;
            if (in_flight == 0 && !gap_at_zero) {
                gap_at_zero    = true;
                gap_zero_since = dispatch_clock::now();
            }
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

    void poison() {
        poisoned  = true;
        in_flight = 0;
        gap_at_zero = false;
        pending_def = {};
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
        note_in_flight_delta(-1);
        if (seq_id != wanted_seq_id) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint + " returned sequence " +
                                     std::to_string(seq_id) + " while awaiting " + std::to_string(wanted_seq_id));
        }
        return type;
    }

    std::vector<planned_request> plan_requests(int32_t                                     layer,
                                               uint32_t                                    n_tokens,
                                               const std::vector<uint16_t> &               activations,
                                               const std::vector<pipe_expert_assignment> & assignments,
                                               const std::vector<std::vector<size_t>> &   layer_routes,
                                               std::vector<size_t> &                       assigned_counts) {
        std::vector<planned_request> by_worker(workers.size());
        for (size_t i = 0; i < workers.size(); ++i) {
            by_worker[i].worker_index = i;
        }
        for (const pipe_expert_assignment & assignment : assignments) {
            const std::vector<size_t> & candidates = layer_routes[(size_t) assignment.expert_id];
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
        return requests;
    }

    void issue_requests(std::vector<planned_request> & requests, uint64_t seq_id) {
        for (planned_request & request : requests) {
            worker & value = workers[request.worker_index];
            if (collect_stats) {
                request.issued_at = dispatch_clock::now();
            }
            if (!pipe_send_frame(*value.socket, PIPE_EXPERT_DISPATCH_REQ, seq_id, request.payload.data(),
                                 request.payload.size())) {
                throw std::runtime_error("expert dispatcher failed to send expert(s) " +
                                         assignment_experts(request.assignments) + " to worker " +
                                         value.info.endpoint);
            }
            note_in_flight_delta(+1);
            ++stats.requests_issued;
        }
    }

    void accumulate_partial(std::vector<float> &             result,
                            planned_request &                request,
                            uint64_t                         seq_id,
                            int32_t                          layer,
                            uint32_t                         n_tokens,
                            dispatch_clock::time_point *     last_response) {
        std::vector<uint8_t>  payload;
        const pipe_frame_type type = await_response(request, seq_id, payload);
        if (collect_stats && last_response != nullptr) {
            *last_response = dispatch_clock::now();
            // per-request wait is only tracked for the primary wait loop via stats.workers
        }
        worker & value = workers[request.worker_index];
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
        // Partial carries (layer, n_tokens); token identity is the layout of
        // partial[token * n_embd + dim]. Do not rely on arrival ordering across
        // workers — each partial is a full [n_tokens * n_embd] block.
        if (partial.layer != layer || partial.n_tokens != n_tokens || partial.partial.size() != result.size()) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " returned the wrong partial shape for expert(s) " +
                                     assignment_experts(request.assignments) +
                                     " (layer=" + std::to_string(partial.layer) +
                                     " want=" + std::to_string(layer) +
                                     " n_tokens=" + std::to_string(partial.n_tokens) +
                                     " want_n_tokens=" + std::to_string(n_tokens) + ")");
        }
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += ggml_fp16_to_fp32((ggml_fp16_t) partial.partial[i]);
        }
    }

    // Collect previously-issued deferred partials and sum them. Marks late if
    // the fold point for this batch was already closed (successor returned).
    std::vector<float> collect_pending_deferred(bool mark_fold_open) {
        if (pending_def.requests.empty()) {
            return {};
        }
        if (mark_fold_open) {
            pending_def.fold_opened = true;
        }
        const size_t           n_values = (size_t) pending_def.n_tokens * (size_t) n_embd;
        std::vector<float>     fold(n_values, 0.0f);
        const int32_t          layer  = pending_def.layer;
        const uint64_t         seq_id = pending_def.seq_id;
        const uint32_t         n_tok  = pending_def.n_tokens;
        std::vector<planned_request> requests = std::move(pending_def.requests);
        pending_def.requests.clear();

        for (planned_request & request : requests) {
            // If the successor layer already returned without this partial, it is late.
            if (pending_def.fold_closed) {
                ++deferral.n_deferred_late;
            }
            accumulate_partial(fold, request, seq_id, layer, n_tok, nullptr);
            update_residency(request.worker_index, layer, request.assignments);
        }
        pending_def = {};
        return fold;
    }

    std::vector<float> dispatch(int32_t                                     layer,
                                uint64_t                                    seq_id,
                                uint32_t                                    n_tokens,
                                const std::vector<uint16_t> &               activations,
                                const std::vector<pipe_expert_assignment> & assignments) {
        if (poisoned) {
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

        try {
            // Fold previous layer's deferred partials into this layer's residual
            // contribution (added to the returned MoE block). Collect BEFORE
            // issuing this layer so TCP response order stays simple: each
            // worker's deferred response from layer N is fully drained before
            // layer N+1 requests go out. Deferred reads were already in flight
            // across the inter-layer gap (attention + router of this layer).
            std::vector<float> folded_prev = collect_pending_deferred(/*mark_fold_open=*/true);

            // Decide whether this layer may leave experts deferred.
            // Last routed layer has no successor — do not defer (fold would fall off).
            const bool may_defer =
                defer_k_value > 0 &&
                layer != last_routed_layer &&
                n_tokens >= 1;  // prefill and decode both handled via per-token top-K

            std::vector<pipe_expert_assignment> immediate;
            std::vector<pipe_expert_assignment> deferred;
            size_t                              n_def = 0;
            if (may_defer) {
                split_immediate_deferred(assignments, n_tokens, defer_k_value, immediate, deferred, n_def);
            } else {
                immediate = assignments;
            }
            deferral.n_deferred += n_def;

            const dispatch_clock::time_point issue_start =
                collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};

            // Shared assigned_counts so residency balancing sees both sets.
            std::vector<size_t> assigned_counts(workers.size(), 0);
            std::vector<planned_request> imm_requests =
                plan_requests(layer, n_tokens, activations, immediate, route_it->second, assigned_counts);
            std::vector<planned_request> def_requests =
                deferred.empty()
                    ? std::vector<planned_request>{}
                    : plan_requests(layer, n_tokens, activations, deferred, route_it->second, assigned_counts);

            stats              = {};
            stats.workers_used = imm_requests.size() + def_requests.size();
            for (const planned_request & request : imm_requests) {
                stats.workers.push_back({
                    workers[request.worker_index].info.endpoint,
                    request.assignments.size(),
                    0,
                    1,
                    request.assignments.size(),
                });
            }

            // THE critical ordering: issue deferred reads BEFORE waiting on
            // immediate, and before this call returns. Issue-then-return is the
            // whole mechanism — issuing deferred when the next layer begins
            // reproduces the gap this is meant to close.
            // Per-worker send order: for each worker, immediate then deferred
            // would require interleaving. We send all immediate first, then all
            // deferred. If the same worker is in both, its socket sees imm then
            // def — await order for that worker is imm first (we only await imm
            // now; def stays pending).
            issue_requests(imm_requests, seq_id);
            if (!def_requests.empty()) {
                // Use a distinct seq_id band so a demux mistake surfaces as an
                // error rather than a silent mix-up. Deferred seq = original |
                // high bit... actually keep same seq_id so await_response for
                // deferred later matches what the worker echoes. Workers echo
                // the request seq_id; both batches share this layer's seq_id.
                // Ordering on the socket disambiguates.
                issue_requests(def_requests, seq_id);
            }
            if (collect_stats) {
                stats.ns_issue = elapsed_ns(issue_start, dispatch_clock::now());
            }

            std::vector<float> result((size_t) activation_count, 0.0f);
            // Fold previous deferred into this layer's output (residual path).
            if (!folded_prev.empty()) {
                if (folded_prev.size() != result.size()) {
                    // Token count changed between layers (shouldn't happen in
                    // one decode step). Count as late and drop rather than
                    // corrupt — shapes must match.
                    deferral.n_deferred_late += 1;
                } else {
                    for (size_t i = 0; i < result.size(); ++i) {
                        result[i] += folded_prev[i];
                    }
                }
            }

            const dispatch_clock::time_point wait_start =
                collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
            dispatch_clock::time_point last_response;
            for (size_t request_index = 0; request_index < imm_requests.size(); ++request_index) {
                planned_request & request = imm_requests[request_index];
                const dispatch_clock::time_point before = collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
                accumulate_partial(result, request, seq_id, layer, n_tokens, &last_response);
                if (collect_stats) {
                    stats.workers[request_index].ns_wait = elapsed_ns(request.issued_at, last_response);
                    (void) before;
                }
                update_residency(request.worker_index, layer, request.assignments);
            }
            if (collect_stats && !imm_requests.empty()) {
                stats.ns_wait = elapsed_ns(wait_start, last_response);
            }

            // Stash deferred requests — do NOT wait. They stay in flight across
            // the next layer's attention/router work.
            if (!def_requests.empty()) {
                // If we somehow already had pending deferred, the previous
                // collect should have drained it. Anything still here is a bug.
                if (!pending_def.requests.empty()) {
                    // Force-collect and mark late before overwriting.
                    pending_def.fold_closed = true;
                    (void) collect_pending_deferred(/*mark_fold_open=*/false);
                }
                pending_def.layer       = layer;
                pending_def.seq_id      = seq_id;
                pending_def.n_tokens    = n_tokens;
                pending_def.requests    = std::move(def_requests);
                pending_def.fold_opened = false;
                pending_def.fold_closed = false;
            }

            return result;
        } catch (...) {
            poison();
            throw;
        }
    }

    void begin_window() noexcept {
        window_active     = true;
        window_sample_ok  = sample_nvme_io_ticks(window_io_ticks_begin);
        window_wall_begin = dispatch_clock::now();
        // Reset per-window gap; keep cumulative n_deferred / n_deferred_late.
        deferral.ns_gap        = 0;
        deferral.nvme_util_pct = -1.0;
        gap_at_zero            = (in_flight == 0);
        if (gap_at_zero) {
            gap_zero_since = window_wall_begin;
        }
    }

    void end_window() noexcept {
        if (!window_active) {
            return;
        }
        window_active = false;
        if (gap_at_zero) {
            deferral.ns_gap += elapsed_ns(gap_zero_since, dispatch_clock::now());
            gap_at_zero = false;
        }
        if (window_sample_ok) {
            uint64_t end_ticks = 0;
            if (sample_nvme_io_ticks(end_ticks)) {
                const double wall_ms =
                    std::chrono::duration<double, std::milli>(dispatch_clock::now() - window_wall_begin).count();
                if (wall_ms > 0.0) {
                    const double busy_ms = (double) (end_ticks - window_io_ticks_begin);
                    // Average util across devices is not well-defined when
                    // summing ticks; with one device (the usual case) this is
                    // exact. Cap at 100.
                    double pct = 100.0 * busy_ms / wall_ms;
                    if (pct < 0.0) {
                        pct = 0.0;
                    }
                    if (pct > 100.0) {
                        pct = 100.0;
                    }
                    deferral.nvme_util_pct = pct;
                }
            }
        }
    }

    std::vector<float> drain() {
        if (pending_def.requests.empty()) {
            return {};
        }
        // Anything still pending at drain has missed its fold point.
        pending_def.fold_closed = true;
        return collect_pending_deferred(/*mark_fold_open=*/false);
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

int32_t dispatcher::n_ff_exp() const {
    return pimpl->n_ff_exp;
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

const deferral_stats & dispatcher::get_deferral_stats() const {
    return pimpl->deferral;
}

int dispatcher::defer_k() const {
    return pimpl->defer_k_value;
}

void dispatcher::begin_deferral_window() noexcept {
    pimpl->begin_window();
}

void dispatcher::end_deferral_window() noexcept {
    pimpl->end_window();
}

std::vector<float> dispatcher::drain_deferred() {
    return pimpl->drain();
}

}  // namespace pipe_expert_dispatcher
