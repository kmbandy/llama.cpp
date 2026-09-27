// wp-worker-verify: send ONE real (not synthetic) dispatch request to an
// already-running wp-expert-worker and dump the resulting partial buffer to
// a file, so a test harness can compare it against an independent reference.
//
// Unlike wp-worker-replay (which replays SYNTHETIC activations/weights to
// measure throughput), this tool's activations and per-expert routing
// weights come from the caller (a plain-text request file) -- it exists so
// tests/wp_forge/test_ml8_e2e.py can drive the real worker binary with real
// data and check the actual computed numbers against a numpy reference,
// without reimplementing the wire protocol (pipe-protocol.h) in Python.
//
// Usage:
//   llama-wp-worker-verify <host> <port> <request_file> <output_file>
//
// request_file (whitespace-separated text):
//   line 1: layer n_tokens n_embd swiglu_clamp
//   line 2: n_assignments
//   then n_assignments lines, each: expert_id w_0 w_1 ... w_{n_tokens-1}
//   then n_tokens*n_embd float activations (row-major [n_tokens, n_embd]),
//   whitespace/newline separated (any layout is fine, only order matters).
//
// output_file: n_tokens*n_embd raw little-endian float32 values (the
// decoded PIPE_EXPERT_PARTIAL's `partial` buffer), no header -- read with
// e.g. numpy.fromfile(path, dtype="<f4").
//
// On any error, prints "wp-worker-verify error: ..." to stderr and returns
// a non-zero exit code; writes nothing to output_file in that case.

#include "pipe-protocol.h"
#include "pipe-transport.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

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

struct ParsedRequest {
    int32_t layer = -1;
    uint32_t n_tokens = 0;
    int32_t n_embd = 0;
    float swiglu_clamp = 0.0f;
    pipe_expert_dispatch_req req;
};

ParsedRequest read_request(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open request file: " + path);
    }
    ParsedRequest out;
    long n_tokens_signed = 0;
    if (!(f >> out.layer >> n_tokens_signed >> out.n_embd >> out.swiglu_clamp)) {
        throw std::runtime_error("request file: bad header line");
    }
    out.n_tokens = (uint32_t) n_tokens_signed;
    long n_assign = 0;
    if (!(f >> n_assign) || n_assign < 0) {
        throw std::runtime_error("request file: bad assignment count");
    }
    out.req.layer = out.layer;
    out.req.n_tokens = out.n_tokens;
    out.req.swiglu_clamp = out.swiglu_clamp;
    out.req.assignments.reserve((size_t) n_assign);
    for (long i = 0; i < n_assign; ++i) {
        pipe_expert_assignment a;
        if (!(f >> a.expert_id)) {
            throw std::runtime_error("request file: bad expert_id in assignment " + std::to_string(i));
        }
        a.weights.resize(out.n_tokens);
        for (uint32_t t = 0; t < out.n_tokens; ++t) {
            if (!(f >> a.weights[t])) {
                throw std::runtime_error("request file: bad weight in assignment " + std::to_string(i));
            }
        }
        out.req.assignments.push_back(std::move(a));
    }
    const size_t n_acts = (size_t) out.n_tokens * (size_t) out.n_embd;
    out.req.activations.resize(n_acts);
    for (size_t i = 0; i < n_acts; ++i) {
        if (!(f >> out.req.activations[i])) {
            throw std::runtime_error("request file: expected " + std::to_string(n_acts) +
                                      " activation floats, ran out at index " + std::to_string(i));
        }
    }
    return out;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr,
            "usage: %s <host> <port> <request_file> <output_file>\n", argv[0]);
        return 2;
    }
    const std::string host        = argv[1];
    const int port                = std::atoi(argv[2]);
    const std::string request_path = argv[3];
    const std::string output_path  = argv[4];

    if (!pipe_transport_init()) {
        std::fprintf(stderr, "wp-worker-verify error: pipe_transport_init failed\n");
        return 1;
    }

    try {
        const ParsedRequest parsed = read_request(request_path);

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

        // Echo the worker's own hparams/identity back as the client HELLO --
        // validate_client_hello() in wp-expert-worker.cpp requires an exact
        // match (see wp-worker-replay.cpp, same pattern).
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

        const std::vector<uint8_t> req_payload = pipe_encode_expert_dispatch_req(parsed.req);
        if (!pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 1, req_payload.data(), req_payload.size())) {
            throw std::runtime_error("send failed");
        }
        if (!pipe_recv_frame(*socket, type, seq_id, payload)) {
            throw std::runtime_error("recv failed");
        }
        if (type == PIPE_ERROR) {
            const pipe_error err = pipe_decode_error(payload.data(), payload.size());
            throw std::runtime_error("worker returned PIPE_ERROR: " + err.msg);
        }
        if (type != PIPE_EXPERT_PARTIAL) {
            throw std::runtime_error("unexpected frame type " + std::to_string((uint32_t) type));
        }
        const pipe_expert_partial partial =
            pipe_decode_expert_partial(payload.data(), payload.size(), parsed.n_embd);
        const size_t want = (size_t) parsed.n_tokens * (size_t) parsed.n_embd;
        if (partial.partial.size() != want) {
            throw std::runtime_error("decoded partial has " + std::to_string(partial.partial.size()) +
                                      " floats, expected " + std::to_string(want));
        }

        std::ofstream out(output_path, std::ios::binary);
        if (!out) {
            throw std::runtime_error("cannot open output file: " + output_path);
        }
        out.write(reinterpret_cast<const char *>(partial.partial.data()),
                  (std::streamsize) (partial.partial.size() * sizeof(float)));
        if (!out) {
            throw std::runtime_error("failed to write output file: " + output_path);
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "wp-worker-verify error: %s\n", e.what());
        return 1;
    }
    return 0;
}
