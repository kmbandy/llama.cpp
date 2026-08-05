#include "pipe-expert-dispatcher.h"

#include "ggml.h"
#include "pipe-transport.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// poll() for concurrent harvest of worker responses (POSIX; the dispatcher
// already assumes POSIX sockets).
#include <poll.h>
#include <cerrno>

namespace pipe_expert_dispatcher {
namespace {

using dispatch_clock = std::chrono::steady_clock;

bool dispatch_stats_enabled() {
    const char * value = std::getenv("WP_DISPATCH_STATS");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool speed_split_enabled() {
    const char * value = std::getenv("WP_DISPATCH_SPEED_SPLIT");
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
// Field layout matches /tmp/qd_sample.py and the kernel docs:
//   0-based tokens: [0]=major [1]=minor [2]=name ... [12]=io_ticks [13]=weighted_ms
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
        std::vector<std::string> tok;
        std::string              t;
        while (fields >> t) {
            tok.push_back(t);
        }
        if (tok.size() < 13) {
            continue;
        }
        const std::string & name = tok[2];
        if (name.rfind("nvme", 0) != 0) {
            continue;
        }
        // Whole devices: nvme0n1. Partitions: nvme0n1p1 / nvme0n1p2 — skip.
        // Match "p" only after the namespace digit so we do not reject nvme0n1.
        {
            const size_t npos = name.find('n');
            if (npos != std::string::npos && name.find('p', npos) != std::string::npos) {
                continue;
            }
        }
        // io_ticks is token[12] (1-based field 13). Do NOT read 11 values after
        // name and then one more — that lands on token[14], which is nearly
        // static under pure-read loads and yields util% ≈ 0.0.
        char * end = nullptr;
        const unsigned long long io_ticks = std::strtoull(tok[12].c_str(), &end, 10);
        if (end == tok[12].c_str() || *end != '\0') {
            continue;
        }
        io_ticks_ms_sum += (uint64_t) io_ticks;
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
    static constexpr size_t speed_estimate_min_samples = 3;
    static constexpr size_t speed_estimate_window       = 8;
    static constexpr size_t speed_estimate_min_spread   = 2;

    struct speed_sample {
        size_t n = 0;
        double wait_ms = 0.0;
    };

    struct worker {
        endpoint                                 target;
        worker_info                              info;
        pipe_expert_hello                        hello;
        pipe_socket_ptr                          socket;
        std::vector<std::pair<int32_t, int32_t>> resident_lru;
        std::array<speed_sample, speed_estimate_window> speed_history{};
        size_t                                   speed_history_next      = 0;
        size_t                                   speed_samples            = 0;
        size_t                                   speed_n_spread            = 0;
        double                                   estimated_fixed_ms        = 0.0;
        double                                   estimated_ms_per_expert   = 0.0;
        bool                                     speed_fit_valid            = false;
    };

    struct planned_request {
        size_t                              worker_index = 0;
        std::vector<pipe_expert_assignment> assignments;
        std::vector<uint8_t>                payload;
        dispatch_clock::time_point          issued_at;
        uint64_t                            wait_ns = 0;
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
    // Host-provided last main-graph MoE layer that must not defer (no successor).
    // -1 => fall back to last_routed_layer from worker HELLO. Must be set to
    // hparams.n_layer()-1 so NextN/MTP layers (e.g. blk.78) advertised by
    // workers are not mistaken for the fold successor of the main stack.
    int32_t                                             last_no_defer_layer = -1;
    int                                                 defer_k_value = 0;
    std::string                                         model_identity;
    bool                                                poisoned = false;
    bool                                                collect_stats = false;
    bool                                                speed_split   = false;
    bool                                                stats_logging = false;

    // Per-request wire log; see accumulate_partial. Off unless WP_DISPATCH_REQ_LOG
    // is set, so it costs nothing in a normal run.
    FILE *                                              req_log_ = [] {
        const char * p = std::getenv("WP_DISPATCH_REQ_LOG");
        return (p != nullptr && p[0] != '\0') ? fopen(p, "w") : (FILE *) nullptr;
    }();

    // Gap accounting: time spent with in_flight == 0.
    bool                       gap_at_zero = false;
    dispatch_clock::time_point gap_zero_since{};
    bool                       window_active = false;
    uint64_t                   window_io_ticks_begin = 0;
    dispatch_clock::time_point window_wall_begin{};
    bool                       window_sample_ok = false;

    explicit impl(const std::vector<endpoint> & endpoints) :
                speed_split(speed_split_enabled()) {
        stats_logging = dispatch_stats_enabled();
        collect_stats = stats_logging || speed_split;
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
        // *** STATIC ASSIGNMENT (default ON, 2026-08-04). REPRODUCIBILITY FIX. ***
        // The balancing path below chooses from residency, in-request assigned_counts,
        // and a rotating machine_cursor -- all of which move with batch width and
        // request history. On this fleet THREE workers on THREE DIFFERENT BACKENDS
        // (CUDA 1070, Vulkan RX480, CPU) all advertise experts 0..84, so the same
        // expert could execute on a different backend from one run to the next. The
        // comment on harvest_partials already recorded the consequence: "Worker
        // ASSIGNMENT is already timing-dependent -- ~35% of requests differ between
        // identical runs". Combined with the f16 subtotals (now fixed) that silently
        // changed generated text at temperature 0 whenever the speculative draft
        // length changed.
        // Even with f32 partials a moving partition still re-associates the sum, so
        // for BITWISE reproducibility the assignment must be a PURE FUNCTION of
        // (layer, expert). A mixing hash keeps the spread without the state.
        // WP_DISPATCH_STATIC_ASSIGN=0 restores the old load-balancing behaviour --
        // faster in principle (it can prefer a worker that already holds the page)
        // but NOT reproducible. Do not turn it off for any run whose OUTPUT matters.
        static const bool s_static_assign = [] {
            const char * e = std::getenv("WP_DISPATCH_STATIC_ASSIGN");
            return e == nullptr || e[0] != '0';   // default ON
        }();
        if (s_static_assign && candidates.size() > 1) {
            // splitmix64 on (layer, expert): deterministic, well-spread, no state.
            uint64_t h = ((uint64_t) (uint32_t) layer << 32) ^ (uint32_t) expert;
            h += 0x9E3779B97F4A7C15ull;
            h  = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ull;
            h  = (h ^ (h >> 27)) * 0x94D049BB133111EBull;
            h ^=  h >> 31;
            return candidates[(size_t) (h % (uint64_t) candidates.size())];
        }
        if (s_static_assign) {
            return candidates.front();
        }

        bool any_resident = false;
        for (size_t candidate : candidates) {
            any_resident = any_resident || is_resident(candidate, layer, expert);
        }

        size_t              best_count      = (size_t) -1;
        uint32_t            best_slots      = 0;
        double              best_projection = 0.0;
        std::vector<size_t> tied;
        bool                use_speed = speed_split;
        if (use_speed) {
            for (size_t candidate : candidates) {
                if (any_resident && !is_resident(candidate, layer, expert)) {
                    continue;
                }
                if (workers[candidate].speed_samples < speed_estimate_min_samples ||
                    !workers[candidate].speed_fit_valid) {
                    use_speed = false;
                    break;
                }
            }
        }
        for (size_t candidate : candidates) {
            if (any_resident && !is_resident(candidate, layer, expert)) {
                continue;
            }
            const size_t   count = assigned_counts[candidate];
            const uint32_t slots = workers[candidate].hello.n_slots;
            if (!use_speed) {
                if (count < best_count || (count == best_count && slots > best_slots)) {
                    best_count = count;
                    best_slots = slots;
                    tied.clear();
                    tied.push_back(candidate);
                } else if (count == best_count && slots == best_slots) {
                    tied.push_back(candidate);
                }
                continue;
            }

            const double projection = workers[candidate].estimated_ms_per_expert * (count + 1);
            if (tied.empty() || projection < best_projection ||
                (projection == best_projection && slots > best_slots)) {
                best_projection = projection;
                best_slots      = slots;
                tied.clear();
                tied.push_back(candidate);
            } else if (projection == best_projection && slots == best_slots) {
                tied.push_back(candidate);
            }
        }

        const std::string & machine = workers[candidates.front()].target.machine;
        size_t &            cursor  = machine_cursor[machine];
        const size_t        chosen  = tied[cursor % tied.size()];
        ++cursor;
        return chosen;
    }

    void update_speed_estimate(const planned_request & request) {
        if (!speed_split || !collect_stats || request.assignments.empty() || request.wait_ns == 0) {
            return;
        }
        worker & value = workers[request.worker_index];
        value.speed_history[value.speed_history_next] = {
            request.assignments.size(), request.wait_ns * 1.0e-6,
        };
        value.speed_history_next = (value.speed_history_next + 1) % speed_estimate_window;
        if (value.speed_samples < speed_estimate_window) {
            ++value.speed_samples;
        }

        size_t min_n = (size_t) -1;
        size_t max_n = 0;
        double sum_n = 0.0;
        double sum_wait = 0.0;
        for (size_t i = 0; i < value.speed_samples; ++i) {
            const speed_sample & sample = value.speed_history[i];
            min_n = std::min(min_n, sample.n);
            max_n = std::max(max_n, sample.n);
            sum_n += (double) sample.n;
            sum_wait += sample.wait_ms;
        }
        value.speed_n_spread = max_n - min_n;
        value.speed_fit_valid = value.speed_samples >= speed_estimate_min_samples &&
                                value.speed_n_spread >= speed_estimate_min_spread;
        if (!value.speed_fit_valid) {
            return;
        }

        const double mean_n = sum_n / (double) value.speed_samples;
        const double mean_wait = sum_wait / (double) value.speed_samples;
        double       sxx = 0.0;
        double       sxy = 0.0;
        for (size_t i = 0; i < value.speed_samples; ++i) {
            const double dn = (double) value.speed_history[i].n - mean_n;
            sxx += dn * dn;
            sxy += dn * (value.speed_history[i].wait_ms - mean_wait);
        }
        const double slope = sxy / sxx;
        if (!(slope > 0.0)) {
            value.speed_fit_valid = false;
            return;
        }
        value.estimated_ms_per_expert = slope;
        value.estimated_fixed_ms = mean_wait - slope * mean_n;
    }

    void update_speed_estimates(const std::vector<planned_request> & requests) {
        for (const planned_request & request : requests) {
            update_speed_estimate(request);
        }
    }

    void log_speed_state(const std::vector<size_t> & assigned_counts) const {
        if (!speed_split || !stats_logging) {
            return;
        }
        for (size_t i = 0; i < workers.size(); ++i) {
            std::fprintf(stderr, "expert dispatch speed worker %s a=%.4f ms b=%.4f ms/expert samples=%zu n-spread=%zu assigned=%zu\n",
                         workers[i].info.endpoint.c_str(), workers[i].estimated_fixed_ms,
                         workers[i].estimated_ms_per_expert, workers[i].speed_samples,
                         workers[i].speed_n_spread, assigned_counts[i]);
        }
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
                                               std::vector<size_t> &                       assigned_counts,
                                               float                                       swiglu_clamp) {
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
            wire_request.swiglu_clamp = swiglu_clamp;
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

    // Receive ONE partial and decode it into `out` (does NOT accumulate). Split
    // out of accumulate_partial so the caller can harvest partials in ARRIVAL
    // order while still summing them in a FIXED order -- see harvest_partials.
    void receive_partial(std::vector<float> &             out,
                         size_t                           n_values,
                         planned_request &                request,
                         uint64_t                         seq_id,
                         int32_t                          layer,
                         uint32_t                         n_tokens,
                         dispatch_clock::time_point *     last_response) {
        std::vector<uint8_t>  payload;
        // WP_DISPATCH_REQ_LOG=path: one line per request --
        //   layer n_tokens worker_index n_experts ns_before_await ns_blocked
        //
        // n_tokens added 2026-08-03: prefill (>1) vs decode (==1). Without it the
        // spine-side wire timings could not be split by phase either, so joining
        // against the worker log to get `wire = ns_blocked - worker_service` gave
        // one blended number across two workloads that differ by ~500x in tokens.
        //
        // ns_before_await is issue -> the moment we START awaiting this request;
        // the spine is doing its own work (issuing others, awaiting an earlier
        // worker) during it. ns_blocked is the recv itself. THE SPLIT IS THE
        // POINT: per-worker `wait` in the existing stats is issue -> consumed,
        // so for a worker awaited second or third it silently includes time the
        // spine spent on the first, which is why those waits sum to 287 ms
        // against a 156 ms dispatch wall. Only ns_blocked is time genuinely
        // spent waiting on the wire and the worker.
        //
        // Join offline against the worker's own WP_REQ_LOG ns_wall (same request
        // order per worker) to get wire = ns_blocked - worker_service.
        const auto wp_await_t0 = req_log_ != nullptr ? dispatch_clock::now()
                                                     : dispatch_clock::time_point();
        const pipe_frame_type type = await_response(request, seq_id, payload);
        if (req_log_ != nullptr) {
            const auto now = dispatch_clock::now();
            fprintf(req_log_, "%d %u %zu %zu %llu %llu\n", layer, n_tokens,
                    request.worker_index, request.assignments.size(),
                    (unsigned long long) elapsed_ns(request.issued_at, wp_await_t0),
                    (unsigned long long) elapsed_ns(wp_await_t0, now));
            fflush(req_log_);
        }
        if (collect_stats && last_response != nullptr) {
            *last_response = dispatch_clock::now();
            // per-request wait is only tracked for the primary wait loop via stats.workers
        }
        if (speed_split) {
            request.wait_ns = elapsed_ns(request.issued_at, dispatch_clock::now());
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
        if (partial.layer != layer || partial.n_tokens != n_tokens || partial.partial.size() != n_values) {
            throw std::runtime_error("expert dispatcher worker " + value.info.endpoint +
                                     " returned the wrong partial shape for expert(s) " +
                                     assignment_experts(request.assignments) +
                                     " (layer=" + std::to_string(partial.layer) +
                                     " want=" + std::to_string(layer) +
                                     " n_tokens=" + std::to_string(partial.n_tokens) +
                                     " want_n_tokens=" + std::to_string(n_tokens) + ")");
        }
        // partials arrive as f32 (PIPE_VERSION 2) -- no conversion, no rounding.
        out.assign(partial.partial.begin(), partial.partial.end());
        GGML_ASSERT(out.size() == n_values);
    }

    // Original behaviour, kept for the deferred-fold path: receive and add.
    void accumulate_partial(std::vector<float> &             result,
                            planned_request &                request,
                            uint64_t                         seq_id,
                            int32_t                          layer,
                            uint32_t                         n_tokens,
                            dispatch_clock::time_point *     last_response) {
        std::vector<float> one;
        receive_partial(one, result.size(), request, seq_id, layer, n_tokens, last_response);
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] += one[i];
        }
    }

    // Harvest a layer's partials AS THEY ARRIVE rather than in fixed worker
    // order, then sum them in fixed order.
    //
    // WHY. Measured 2026-08-02: 149.19 of the 155.8 ms/token dispatch wall is
    // spent genuinely blocked, but the spine awaited worker 0, then 1, then 2,
    // so a worker that had already answered sat unread until its turn came. The
    // per-request log showed worker 1 blocking 9.6 us -- its response had been
    // sitting in the socket the whole time. Per layer that cost ~1.6 ms beyond
    // the slowest worker's own service, ~69 ms/token.
    //
    // WHY SUM IN FIXED ORDER. Floating-point addition is not associative, so
    // summing in arrival order would make the result depend on network timing.
    // Buffering costs 3 x n_embd floats and removes a source of run-to-run
    // variance rather than adding one. (Worker ASSIGNMENT is already timing-
    // dependent -- ~35% of requests differ between identical runs -- but that is
    // no reason to add a second such source here.)
    void harvest_partials(std::vector<float> &             result,
                          std::vector<planned_request> &   requests,
                          uint64_t                         seq_id,
                          int32_t                          layer,
                          uint32_t                         n_tokens,
                          dispatch_clock::time_point *     last_response) {
        const size_t n = requests.size();
        if (n == 0) {
            return;
        }
        std::vector<std::vector<float>> partials(n);
        std::vector<char>               done(n, 0);
        size_t                          remaining = n;

        while (remaining > 0) {
            // Poll set = the FIRST outstanding request per socket. Two requests
            // sharing a worker must stay in FIFO order on that socket, and
            // await_response's seq_id check would throw if they were reordered.
            std::vector<struct pollfd> pfds;
            std::vector<size_t>        idx;
            std::set<int>              seen;
            for (size_t i = 0; i < n; ++i) {
                if (done[i]) {
                    continue;
                }
                const int fd = workers[requests[i].worker_index].socket->poll_fd();
                if (fd < 0) {
                    pfds.clear();
                    idx.clear();
                    break;
                }
                if (!seen.insert(fd).second) {
                    continue;
                }
                struct pollfd p;
                p.fd      = fd;
                p.events  = POLLIN;
                p.revents = 0;
                pfds.push_back(p);
                idx.push_back(i);
            }
            if (pfds.empty()) {
                // No pollable descriptor: fall back to the original fixed-order
                // await so this can never be worse than what it replaced.
                for (size_t i = 0; i < n; ++i) {
                    if (done[i]) {
                        continue;
                    }
                    receive_partial(partials[i], result.size(), requests[i], seq_id,
                                    layer, n_tokens, last_response);
                    done[i] = 1;
                    --remaining;
                }
                break;
            }
            const int r = ::poll(pfds.data(), (nfds_t) pfds.size(), -1);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                throw std::runtime_error(std::string("expert dispatcher poll failed: ") +
                                         std::strerror(errno));
            }
            for (size_t k = 0; k < pfds.size(); ++k) {
                if (pfds[k].revents == 0) {
                    continue;
                }
                const size_t i = idx[k];
                receive_partial(partials[i], result.size(), requests[i], seq_id,
                                layer, n_tokens, last_response);
                done[i] = 1;
                --remaining;
            }
        }

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < result.size(); ++j) {
                result[j] += partials[i][j];
            }
        }
    }

    // Collect previously-issued deferred partials and sum them. Caller must
    // already have issued the current layer's requests so N-1 deferred overlaps
    // N's in-flight reads. Marks late if the fold point was already closed.
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
            update_speed_estimate(request);
            update_residency(request.worker_index, layer, request.assignments);
        }
        pending_def = {};
        return fold;
    }

    std::vector<float> dispatch(int32_t                                     layer,
                                uint64_t                                    seq_id,
                                uint32_t                                    n_tokens,
                                const std::vector<uint16_t> &               activations,
                                const std::vector<pipe_expert_assignment> & assignments,
                                float                                       swiglu_clamp) {
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
            // Decide whether this layer may leave experts deferred.
            // The last main-graph MoE layer has no successor to fold into — do
            // not defer it. Prefer the host-provided last_no_defer_layer
            // (hparams.n_layer()-1) over the worker HELLO max: workers also
            // advertise NextN/MTP layers (e.g. 78) that the main graph never
            // dispatches, which previously left every token's last main MoE
            // layer deferred and drained as n_deferred_late at end_decode.
            const int32_t no_defer_layer =
                last_no_defer_layer >= 0 ? last_no_defer_layer : last_routed_layer;
            // Decode only (n_tokens == 1). Prefill shrinks the last layer via
            // get_rows(out_ids) so a deferred partial from layer L with the full
            // prefill width cannot fold into layer L+1's MoE output — that was a
            // silent drop + n_deferred_late path. Spec allows disabling prefill
            // rather than half-working it.
            const bool may_defer =
                defer_k_value > 0 &&
                layer != no_defer_layer &&
                n_tokens == 1;

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
            // Note: plan runs BEFORE collect_pending_deferred, so choose_worker
            // does not yet see residency updates from the previous layer's
            // deferred drain. Residency affinity is a heuristic, not a
            // correctness property; this only shifts when deferred experts
            // refresh the LRU relative to the prior order.
            std::vector<size_t> assigned_counts(workers.size(), 0);
            std::vector<planned_request> imm_requests =
                plan_requests(layer, n_tokens, activations, immediate, route_it->second, assigned_counts,
                              swiglu_clamp);
            std::vector<planned_request> def_requests =
                deferred.empty()
                    ? std::vector<planned_request>{}
                    : plan_requests(layer, n_tokens, activations, deferred, route_it->second, assigned_counts,
                                    swiglu_clamp);

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

            // Occupancy ordering (fetch-against-fetch):
            //   1. issue layer N immediate AND deferred   <- first
            //   2. await layer N-1 deferred               <- overlaps with (1)
            //   3. await layer N immediate
            //   4. fold N-1 deferred into result, return
            // Issuing N before collecting N-1 is the whole occupancy win: N-1's
            // deferred reads stay in flight while N's reads are also in flight.
            // Collecting first would serialise them (wait N-1 def, then issue N)
            // and leave nvme_util_pct unchanged.
            //
            // Per-worker send order within a layer: all immediate requests, then
            // all deferred. Both batches share this layer's seq_id; TCP FIFO per
            // socket plus await_response's seq_id check disambiguate. Do not
            // invent a separate seq_id band — a mismatch throws loudly.
            issue_requests(imm_requests, seq_id);
            if (!def_requests.empty()) {
                issue_requests(def_requests, seq_id);
            }
            if (collect_stats) {
                stats.ns_issue = elapsed_ns(issue_start, dispatch_clock::now());
            }

            // Drain previous layer's deferred partials now that layer N is in
            // flight. Safe with existing wire format: on any worker socket that
            // carries both, N-1 deferred frames were SENT before N's frames, so
            // they ARRIVE first (TCP FIFO). await_response validates seq_id and
            // throws on mismatch — do not weaken that check.
            std::vector<float> folded_prev = collect_pending_deferred(/*mark_fold_open=*/true);

            std::vector<float> result((size_t) activation_count, 0.0f);
            // Fold previous deferred into this layer's output (residual path).
            // Partials carry (layer, token) via pending_def.layer + layout; the
            // vectors must agree on n_tokens * n_embd. A mismatch means the
            // fold crossed a token-count boundary (e.g. last-layer get_rows
            // shrink for out_ids during prefill). Never silent-drop: count late
            // and leave a clear trail. Callers that need late==0 must not defer
            // across that boundary (correct last_no_defer_layer + decode n_tokens).
            if (!folded_prev.empty()) {
                if (folded_prev.size() != result.size()) {
                    ++deferral.n_deferred_late;
                    // Consume is already done (partials were awaited in
                    // collect_pending_deferred); we refuse to add a mis-shaped
                    // block into the residual. This is an explicit accounting
                    // path, not a second silent approximation.
                } else {
                    for (size_t i = 0; i < result.size(); ++i) {
                        result[i] += folded_prev[i];
                    }
                }
            }

            const dispatch_clock::time_point wait_start =
                collect_stats ? dispatch_clock::now() : dispatch_clock::time_point{};
            dispatch_clock::time_point last_response;
            // WP_DISPATCH_HARVEST=1 opts in to as-ready harvesting.
            //
            // DEFAULT OFF, AND THAT IS A MEASURED DECISION, not caution. Measured
            // 2026-08-02, load-matched back-to-back: 4.197 (off) vs 4.231 (on)
            // tok/s, i.e. +0.8%, inside noise. The mechanism DOES work -- summed
            // blocked time falls 152.72 -> 11.85 ms/token, every recv finds its
            // data already waiting -- but the time simply moves into the poll
            // wait, because the workers were ALREADY overlapping. Sum over layers
            // of the MAX worker service is 74.97 ms/token against a 155.8 ms
            // dispatch wall, so ~81 ms/token is overhead that is NOT await
            // ordering and this does not recover it. Keep the code (it is the
            // instrument that measured wire latency directly: with harvest on,
            // before_await minus worker service gives ~0.57-0.65 ms/request on
            // the remote link and ~20 us on the R9700 loopback) but do not pay
            // its complexity by default.
            static const bool harvest = [] {
                const char * e = std::getenv("WP_DISPATCH_HARVEST");
                return e != nullptr && e[0] == '1';
            }();
            if (harvest) {
                harvest_partials(result, imm_requests, seq_id, layer, n_tokens, &last_response);
                for (size_t request_index = 0; request_index < imm_requests.size(); ++request_index) {
                    planned_request & request = imm_requests[request_index];
                    if (collect_stats) {
                        stats.workers[request_index].ns_wait =
                            elapsed_ns(request.issued_at, last_response);
                    }
                    update_residency(request.worker_index, layer, request.assignments);
                }
            } else {
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
            }
            if (collect_stats && !imm_requests.empty()) {
                stats.ns_wait = elapsed_ns(wait_start, last_response);
            }
            update_speed_estimates(imm_requests);
            log_speed_state(assigned_counts);

            // Stash deferred requests — do NOT wait. They stay in flight until
            // the next layer issues and then collects them (fetch-against-fetch).
            if (!def_requests.empty()) {
                // After the collect above, pending must be empty. Anything still
                // here is a bug — force-collect and mark late before overwriting.
                if (!pending_def.requests.empty()) {
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
                                        const std::vector<pipe_expert_assignment> & assignments,
                                        float                                       swiglu_clamp) {
    return pimpl->dispatch(layer, seq_id, n_tokens, activations, assignments, swiglu_clamp);
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

int32_t dispatcher::last_no_defer_layer() const {
    return pimpl->last_no_defer_layer >= 0 ? pimpl->last_no_defer_layer : pimpl->last_routed_layer;
}

void dispatcher::set_last_no_defer_layer(int32_t layer) noexcept {
    pimpl->last_no_defer_layer = layer;
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
