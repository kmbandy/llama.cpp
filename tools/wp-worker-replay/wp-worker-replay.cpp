// wp-worker-replay: cheapest-falsifying-probe stream generator for
// lever-queue item #9 (cross-batch pipelining / two-connection overlap).
//
// A standalone client that speaks the worker wire protocol directly
// (pipe-protocol.h / pipe-transport.h -- the same public API
// tests/test-wp-expert-worker.cpp uses) and replays SYNTHETIC dispatch
// requests against an already-running wp-expert-worker. It does not run a
// spine and does not need one: it reads the worker's own HELLO to learn its
// real layer list and real expert range, then issues requests for those
// layers/experts -- exactly the same "ask the real resident worker to do
// real page-in and real GPU compute against synthetic activations" trick
// wp-expert-worker.cpp's own WP_WARMUP path already uses internally.
//
// The point of THIS tool is to run two of these concurrently against one
// worker (started with WP_WORKER_MULTI_CONN=2) and compare the per-stream
// service rate against one running alone -- if worker/NVMe utilization does
// not move off the idle floor with two concurrent streams, serialization
// binds somewhere other than "only one TCP connection at a time" and
// fine-grained double-buffering is not worth building yet.
//
// Usage:
//   llama-wp-worker-replay <host> <port> <n_requests> [n_tokens] [label]
//
// n_tokens defaults to 32 (a middling prefill-ish width). label is an
// arbitrary tag printed with the summary line so the orchestrator can tell
// streams apart in combined output (default: "<host>:<port>").
//
// Prints ONE summary line per run to stdout:
//   wp-replay label=<label> n=<done> wall_ms=<total> mean_ms=<mean>
//     p50_ms=<p50> p95_ms=<p95> min_ms=<min> max_ms=<max> rps=<req/s>
// plus one "wp-replay error: ..." line and a non-zero exit code on failure.

#include "pipe-protocol.h"
#include "pipe-transport.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;

pipe_socket_ptr connect_with_retry(const std::string & host, int port, int attempts) {
    for (int i = 0; i < attempts; ++i) {
        pipe_socket_ptr socket = pipe_socket_t::connect(host.c_str(), port);
        if (socket) {
            return socket;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }
    throw std::runtime_error("failed to connect to " + host + ":" + std::to_string(port));
}

// Build a synthetic dispatch request for `layer`, cycling through the
// worker's advertised expert range so successive requests touch different
// experts (and therefore different pages) rather than hammering one
// already-resident slot forever.
pipe_expert_dispatch_req make_request(
        const pipe_expert_hello & hello, int32_t layer, uint32_t n_tokens,
        int32_t expert_offset, int32_t n_experts_per_req) {
    pipe_expert_dispatch_req req;
    req.layer        = layer;
    req.n_tokens      = n_tokens;
    req.swiglu_clamp = 10.0f;   // non-zero so the clamp ops get built, matches WP_WARMUP
    const int32_t first = hello.expert_first;
    const int32_t last  = hello.expert_last;
    const int32_t span  = (first >= 0 && last >= first) ? (last - first + 1) : 0;
    if (span > 0) {
        const int32_t n = std::min(n_experts_per_req, span);
        for (int32_t i = 0; i < n; ++i) {
            pipe_expert_assignment a;
            a.expert_id = first + (int32_t) ((expert_offset + i) % span);
            a.weights.assign((size_t) n_tokens, 0.5f);
            req.assignments.push_back(std::move(a));
        }
    }
    req.activations.assign((size_t) n_tokens * (size_t) hello.n_embd, 0.01f);
    return req;
}

struct Stats {
    std::vector<double> latency_ms;

    void print(const std::string & label, double wall_ms) const {
        if (latency_ms.empty()) {
            std::printf("wp-replay label=%s n=0 wall_ms=%.3f (no completed requests)\n",
                        label.c_str(), wall_ms);
            return;
        }
        std::vector<double> sorted = latency_ms;
        std::sort(sorted.begin(), sorted.end());
        const size_t n = sorted.size();
        double sum = 0.0;
        for (double v : sorted) { sum += v; }
        const double mean = sum / (double) n;
        const double p50  = sorted[n / 2];
        const double p95  = sorted[(size_t) (0.95 * (double) (n - 1))];
        const double rps  = wall_ms > 0.0 ? (double) n / (wall_ms / 1000.0) : 0.0;
        std::printf(
            "wp-replay label=%s n=%zu wall_ms=%.3f mean_ms=%.3f p50_ms=%.3f "
            "p95_ms=%.3f min_ms=%.3f max_ms=%.3f rps=%.3f\n",
            label.c_str(), n, wall_ms, mean, p50, p95, sorted.front(), sorted.back(), rps);
        std::fflush(stdout);
    }
};

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <host> <port> <n_requests> [n_tokens] [label]\n", argv[0]);
        return 2;
    }
    const std::string host   = argv[1];
    const int port           = std::atoi(argv[2]);
    const long n_requests    = std::strtol(argv[3], nullptr, 10);
    const uint32_t n_tokens  = (argc > 4) ? (uint32_t) std::strtoul(argv[4], nullptr, 10) : 32;
    const std::string label  = (argc > 5) ? argv[5] : (host + ":" + std::to_string(port));

    if (!pipe_transport_init()) {
        std::fprintf(stderr, "wp-replay error: pipe_transport_init failed\n");
        return 1;
    }

    try {
        pipe_socket_ptr socket = connect_with_retry(host, port, 400);

        // Worker speaks first: PIPE_HELLO with its own advertised shape.
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        if (!pipe_recv_frame(*socket, type, seq_id, payload) || type != PIPE_HELLO) {
            throw std::runtime_error("did not receive worker HELLO");
        }
        const pipe_expert_hello worker_hello =
            pipe_decode_expert_hello(payload.data(), payload.size());
        if (worker_hello.layers.empty() || worker_hello.expert_first < 0) {
            throw std::runtime_error("worker advertised no layers/experts to replay against");
        }

        // Echo the worker's own hparams/identity back as the client HELLO --
        // validate_client_hello() in wp-expert-worker.cpp requires an exact
        // match, and the worker is the only source of truth for what its own
        // shard actually contains.
        pipe_expert_hello client_hello = worker_hello;
        client_hello.role = PIPE_EXPERT_ROLE_CLIENT;
        const std::vector<uint8_t> hello_payload = pipe_encode_expert_hello(client_hello);
        if (!pipe_send_frame(*socket, PIPE_HELLO, 0, hello_payload.data(), hello_payload.size())) {
            throw std::runtime_error("failed to send client HELLO");
        }
        if (!pipe_recv_frame(*socket, type, seq_id, payload) || type != PIPE_EXPERT_HELLO_ACK) {
            throw std::runtime_error("did not receive HELLO ack");
        }
        const pipe_expert_hello_ack ack =
            pipe_decode_expert_hello_ack(payload.data(), payload.size());
        if (!ack.accepted) {
            throw std::runtime_error("worker rejected HELLO: " + ack.reason);
        }

        // n_experts_per_req: touch as many experts per request as the worker
        // advertises using (n_expert_used), capped so we do not build a
        // request wider than the worker's own shard range. Mirrors what a
        // real dispatch round would send for one token's routed set.
        const int32_t n_experts_per_req =
            worker_hello.n_expert_used > 0 ? worker_hello.n_expert_used : 4;

        Stats stats;
        stats.latency_ms.reserve((size_t) n_requests);
        const auto wall_start = clock_t_::now();
        for (long i = 0; i < n_requests; ++i) {
            const int32_t layer = worker_hello.layers[(size_t) i % worker_hello.layers.size()];
            const pipe_expert_dispatch_req req =
                make_request(worker_hello, layer, n_tokens, (int32_t) i, n_experts_per_req);
            const std::vector<uint8_t> req_payload = pipe_encode_expert_dispatch_req(req);

            const auto t0 = clock_t_::now();
            if (!pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, (uint64_t) i + 1,
                                 req_payload.data(), req_payload.size())) {
                throw std::runtime_error("send failed at request " + std::to_string(i));
            }
            if (!pipe_recv_frame(*socket, type, seq_id, payload)) {
                throw std::runtime_error("recv failed at request " + std::to_string(i));
            }
            const double ms = std::chrono::duration<double, std::milli>(
                clock_t_::now() - t0).count();
            if (type == PIPE_ERROR) {
                throw std::runtime_error(
                    "worker returned PIPE_ERROR at request " + std::to_string(i));
            }
            if (type != PIPE_EXPERT_PARTIAL) {
                throw std::runtime_error(
                    "unexpected frame type at request " + std::to_string(i));
            }
            stats.latency_ms.push_back(ms);
        }
        const double wall_ms = std::chrono::duration<double, std::milli>(
            clock_t_::now() - wall_start).count();
        stats.print(label, wall_ms);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "wp-replay error: %s\n", e.what());
        return 1;
    }
    return 0;
}
