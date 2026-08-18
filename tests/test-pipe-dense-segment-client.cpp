#include "pipe-channel.h"
#include "pipe-dense-segment-client.h"
#include "pipe-dense-segment-manifest.h"
#include "pipe-transport.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

static int g_failed = 0;

#define CHECK(cond)                                                             \
    do {                                                                        \
        if (!(cond)) {                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failed;                                                         \
        }                                                                       \
    } while (0)

namespace {

constexpr const char * MODEL_SHA =
    "sha256:3333333333333333333333333333333333333333333333333333333333333333";
constexpr const char * ARTIFACT_A_SHA =
    "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char * ARTIFACT_B_SHA =
    "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

int reserve_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) throw std::runtime_error("socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(fd, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0) {
        close(fd);
        throw std::runtime_error("bind failed");
    }
    socklen_t length = sizeof(address);
    if (getsockname(fd, reinterpret_cast<sockaddr *>(&address), &length) != 0) {
        close(fd);
        throw std::runtime_error("getsockname failed");
    }
    const int port = ntohs(address.sin_port);
    close(fd);
    return port;
}

std::string resign(std::string value) {
    const std::string checksum = pipe_dense_segment::manifest_checksum(value);
    const std::string key = "\"checksum\":\"";
    const size_t begin = value.find(key);
    const size_t value_begin = begin + key.size();
    const size_t value_end = value.find('"', value_begin);
    value.replace(value_begin, value_end - value_begin, checksum);
    return value;
}

// As manifest(), but the remote segment declares one interior tap. Layer 2 is the first
// layer of that segment's band [2, 3], so it is a legal tap for it and NOT computable by
// the head, which owns [0, 1] -- exactly the production shape.
pipe_dense_segment::manifest manifest_with_taps(int port) {
    const std::string json = resign(
        "{\"format\":\"llama.cpp.dense-segment-manifest\",\"version\":1,"
        "\"model_identity_sha256\":\"" + std::string(MODEL_SHA) + "\","
        "\"n_layer\":4,\"n_embd\":4,\"wire_precision\":\"f32\",\"segments\":["
        "{\"id\":3,\"layer_first\":0,\"layer_last\":1,\"host\":\"127.0.0.1\",\"port\":" +
        std::to_string(port) + ",\"stage_gguf\":\"head.gguf\",\"device\":\"CPU\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_A_SHA) + "\"},"
        "{\"id\":7,\"layer_first\":2,\"layer_last\":3,\"host\":\"127.0.0.1\",\"port\":" +
        std::to_string(port) + ",\"stage_gguf\":\"tail.gguf\",\"device\":\"CPU\","
        "\"tap_layers\":[2],"
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_B_SHA) + "\"}],\"checksum\":\"pending\"}");
    return pipe_dense_segment::parse_manifest(json);
}

pipe_dense_segment::manifest manifest(int port) {
    const std::string json = resign(
        "{\"format\":\"llama.cpp.dense-segment-manifest\",\"version\":1,"
        "\"model_identity_sha256\":\"" + std::string(MODEL_SHA) + "\","
        "\"n_layer\":4,\"n_embd\":4,\"wire_precision\":\"f32\",\"segments\":["
        "{\"id\":3,\"layer_first\":0,\"layer_last\":1,\"host\":\"127.0.0.1\",\"port\":" +
        std::to_string(port) + ",\"stage_gguf\":\"head.gguf\",\"device\":\"CPU\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_A_SHA) + "\"},"
        "{\"id\":7,\"layer_first\":2,\"layer_last\":3,\"host\":\"127.0.0.1\",\"port\":" +
        std::to_string(port) + ",\"stage_gguf\":\"tail.gguf\",\"device\":\"CPU\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_B_SHA) + "\"}],\"checksum\":\"pending\"}");
    return pipe_dense_segment::parse_manifest(json);
}

struct fake_worker {
    uint32_t expected_vocab = 7;
    uint32_t reuse_n_past = UINT32_MAX;
    uint32_t n_past = 0;
    // negotiated at HELLO; terminal_width is n_vocab under LOGITS, n_embd under HIDDEN
    uint32_t terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;
    uint32_t terminal_width = 4;
    // interior taps negotiated at HELLO; echoed and then served on every forward
    std::vector<uint32_t> tap_layers;
    // nextn sideband negotiated at HELLO: 0 unless the head declared need
    uint32_t nextn_width = 0;
    // when true, report the sideband as bit-identical to the terminal hidden state and
    // ship it ONCE -- the client must rebuild it transparently
    bool alias_nextn = false;
    std::vector<std::string> events;

    int serve(int port, std::promise<void> & ready) {
        pipe_socket_ptr server = pipe_socket_t::create_server("127.0.0.1", port);
        if (!server) throw std::runtime_error("server creation failed");
        ready.set_value();
        pipe_socket_ptr socket = server->accept();
        if (!socket) throw std::runtime_error("accept failed");
        pipe_channel::channel channel(std::move(socket), "loopback client");
        bool hello_done = false;
        try {
        for (;;) {
            pipe_channel::received_frame frame;
            if (!pipe_channel::channel::harvest({ &channel }, frame, -1)) continue;
            if (!hello_done) {
                const pipe_segment_hello hello =
                    pipe_decode_segment_hello(frame.payload.data(), frame.payload.size());
                if (frame.type != PIPE_SEGMENT_HELLO || hello.segment_id != 7 ||
                    hello.n_embd != 4 || hello.n_vocab != expected_vocab) {
                    channel.send_frame(PIPE_ERROR, frame.seq_id,
                        pipe_encode_error({ PIPE_ERR_HELLO, "manifest HELLO mismatch" }));
                    channel.flush();
                    return 1;
                }
                // logits-on-head: echo the negotiated kind and report the width it
                // implies (n_vocab for LOGITS, n_embd for HIDDEN)
                terminal_kind  = hello.terminal_kind;
                terminal_width = hello.terminal_kind == PIPE_SEGMENT_TERMINAL_LOGITS
                    ? expected_vocab : 4u;
                // interior taps: echo the requested list and report the row width
                tap_layers = hello.tap_layers;
                // nextn sideband: answer with the width the head's declared need
                // implies. This fixture is the tail, so its capability is n_embd.
                nextn_width = hello.nextn_need != 0 ? 4u : 0u;
                {
                    pipe_segment_hello_ack ack{ PIPE_SEGMENT_HELLO_VERSION, true, expected_vocab, 16,
                                                terminal_kind, terminal_width, "" };
                    ack.tap_layers = tap_layers;
                    ack.tap_width  = tap_layers.empty() ? 0u : 4u;
                    ack.nextn_width = nextn_width;
                    channel.send_frame(PIPE_SEGMENT_HELLO_ACK, 0, pipe_encode_segment_hello_ack(ack));
                }
                channel.flush();
                hello_done = true;
                continue;
            }
            if (frame.type == PIPE_SEGMENT_CTRL) {
                const pipe_segment_ctrl control =
                    pipe_decode_segment_ctrl(frame.payload.data(), frame.payload.size());
                events.push_back(control.control == PIPE_SEGMENT_CTRL_RESET ? "reset" :
                    control.control == PIPE_SEGMENT_CTRL_KV_TRIM ? "trim" : "reuse");
                if (control.control == PIPE_SEGMENT_CTRL_RESET) n_past = 0;
                if (control.control == PIPE_SEGMENT_CTRL_KV_TRIM) n_past = control.n_past;
                const uint32_t reported = control.control == PIPE_SEGMENT_CTRL_PROMPT_REUSE &&
                    reuse_n_past != UINT32_MAX ? reuse_n_past : n_past;
                channel.send_frame(PIPE_SEGMENT_CTRL_ACK, frame.seq_id,
                    pipe_encode_segment_ctrl_ack({ PIPE_SEGMENT_CTRL_ACK_VERSION, control.control,
                        control.session_id, control.cache_epoch, PIPE_SEGMENT_CTRL_APPLIED, reported }));
                channel.flush();
                continue;
            }
            if (frame.type == PIPE_SEGMENT_FWD_REQ) {
                const pipe_segment_fwd_req request =
                    pipe_decode_segment_fwd_req(frame.payload.data(), frame.payload.size(), 4);
                events.push_back("forward");
                n_past += request.n_tokens;
                pipe_segment_fwd_resp response;
                response.session_id = request.session_id;
                response.seq_id = request.seq_id;
                response.n_tokens = request.n_tokens;
                response.output_width = terminal_width;
                response.nextn_width = nextn_width;
                response.activations.resize((size_t) request.n_tokens * terminal_width, 3.0f);
                if (nextn_width > 0) {
                    if (alias_nextn) {
                        response.nextn_aliased = 1;
                    } else {
                        response.nextn.resize((size_t) request.n_tokens * nextn_width, 5.0f);
                    }
                }
                response.tap_width = tap_layers.empty() ? 0u : 4u;
                response.n_taps    = (uint32_t) tap_layers.size();
                response.taps.resize((size_t) request.n_tokens * 4 * tap_layers.size(), 9.0f);
                channel.send_frame(PIPE_SEGMENT_FWD_RESP, frame.seq_id,
                    pipe_encode_segment_fwd_resp(response));
                channel.flush();
                continue;
            }
            return 1;
        }
        } catch (const std::runtime_error & error) {
            if (std::string(error.what()).find("peer closed") != std::string::npos) return 0;
            throw;
        }
    }
};

template<typename F>
void run_worker(fake_worker & worker, F && test) {
    const int port = reserve_port();
    std::promise<void> ready;
    std::future<void> ready_future = ready.get_future();
    std::promise<int> result;
    std::future<int> result_future = result.get_future();
    std::thread thread([&] {
        try {
            result.set_value(worker.serve(port, ready));
        } catch (...) {
            try { result.set_exception(std::current_exception()); } catch (...) {}
        }
    });
    ready_future.get();
    test(manifest(port));
    thread.join();
    CHECK(result_future.get() == 0);
}

template <typename F>
void run_worker_with_taps(fake_worker & worker, F && test) {
    const int port = reserve_port();
    std::promise<void> ready;
    std::future<void> ready_future = ready.get_future();
    std::promise<int> result;
    std::future<int> result_future = result.get_future();
    std::thread thread([&] {
        try {
            result.set_value(worker.serve(port, ready));
        } catch (...) {
            try { result.set_exception(std::current_exception()); } catch (...) {}
        }
    });
    ready_future.get();
    test(manifest_with_taps(port));
    thread.join();
    CHECK(result_future.get() == 0);
}

void test_roundtrip_and_controls() {
    fake_worker worker;
    run_worker(worker, [&](const auto & value) {
        // need_nextn = true: the draft-mtp arm, the only one that reads the sideband
        pipe_dense_segment_client::client client(value, 7, true);
        // logits-on-head is the default: the terminal returns n_embd, not n_vocab
        CHECK(client.terminal_kind() == PIPE_SEGMENT_TERMINAL_HIDDEN);
        CHECK(client.terminal_width() == 4);
        CHECK(client.nextn_width() == 4);
        CHECK(client.recurrent_snapshots() == 16);
        client.reset(9, 4);
        CHECK(client.prompt_reuse(9, 4, 0,
            pipe_dense_segment_client::prompt_identity(nullptr, 0)));
        client.begin_forward(9, 12, 2, {0, 1}, {1, 2, 3, 4, 5, 6, 7, 8});
        const pipe_segment_fwd_resp response = client.finish_forward();
        CHECK(response.activations.size() == 8); // 2 tokens * n_embd 4
        CHECK(response.nextn.size() == 8);
        const std::vector<int32_t> prefix = { 1, 2 };
        CHECK(client.prompt_reuse(9, 4, 2,
            pipe_dense_segment_client::prompt_identity(prefix.data(), prefix.size())));
        client.trim(9, 4, 1);
    });
    CHECK(worker.events == std::vector<std::string>({ "reset", "reuse", "forward", "reuse", "trim" }));
}

// NEXTN WIRE DEDUP (PIPE v12), head side.
void test_nextn_not_needed() {
    fake_worker worker;
    run_worker(worker, [&](const auto & value) {
        // need_nextn = false: draft-dspark (the production default) and no-spec. The
        // head must negotiate the sideband away entirely.
        pipe_dense_segment_client::client client(value, 7, false);
        CHECK(client.nextn_width() == 0);
        client.reset(9, 4);
        client.begin_forward(9, 12, 2, {0, 1}, {1, 2, 3, 4, 5, 6, 7, 8});
        const pipe_segment_fwd_resp response = client.finish_forward();
        CHECK(response.activations.size() == 8);
        // zero nextn floats -- and therefore zero nextn bytes on the wire
        CHECK(response.nextn_width == 0);
        CHECK(response.nextn.empty());
    });
    // the tail must have been told, not merely ignored afterwards
    CHECK(worker.nextn_width == 0);
}

void test_nextn_aliased_reconstruction() {
    fake_worker worker;
    worker.alias_nextn = true;
    run_worker(worker, [&](const auto & value) {
        pipe_dense_segment_client::client client(value, 7, true);
        CHECK(client.nextn_width() == 4);
        client.reset(9, 4);
        client.begin_forward(9, 12, 2, {0, 1}, {1, 2, 3, 4, 5, 6, 7, 8});
        const pipe_segment_fwd_resp response = client.finish_forward();
        // The tail shipped ONE copy. Everything above this line must still see a full
        // nextn run, bit-identical to the terminal hidden state -- the saving is on the
        // wire, not in the API.
        CHECK(response.nextn_aliased == 1);
        CHECK(response.nextn.size() == 8);
        CHECK(response.nextn == response.activations);
    });
}

void test_interior_taps() {
    fake_worker worker;
    run_worker_with_taps(worker, [&](const auto & value) {
        pipe_dense_segment_client::client client(value, 7, false);
        // no taps have been harvested before a forward runs
        CHECK(client.taps().empty());
        client.reset(9, 4);
        client.begin_forward(9, 12, 2, {0, 1}, {1, 2, 3, 4, 5, 6, 7, 8});
        const pipe_segment_fwd_resp response = client.finish_forward();
        CHECK(response.activations.size() == 8);
        // the tap must survive finish_forward()'s per-hop response overwrite
        CHECK(client.taps().size() == 1);
        CHECK(client.taps()[0].layer == 2);
        CHECK(client.taps()[0].width == 4);
        CHECK(client.taps()[0].rows.size() == 8); // 2 tokens * n_embd 4
        CHECK(client.taps()[0].rows[0] == 9.0f);
    });
}

void test_prompt_reuse_disagreement() {
    fake_worker worker;
    worker.reuse_n_past = 1;
    run_worker(worker, [&](const auto & value) {
        pipe_dense_segment_client::client client(value, 7, false);
        client.reset(9, 4);
        const std::vector<int32_t> prefix = { 1, 2 };
        CHECK(!client.prompt_reuse(9, 4, 2,
            pipe_dense_segment_client::prompt_identity(prefix.data(), prefix.size())));
    });
}

void test_hello_validation_failure() {
    fake_worker worker;
    worker.expected_vocab = 8;
    const int port = reserve_port();
    std::promise<void> ready;
    std::future<void> ready_future = ready.get_future();
    std::promise<int> result;
    std::future<int> result_future = result.get_future();
    std::thread thread([&] {
        try { result.set_value(worker.serve(port, ready)); }
        catch (...) { try { result.set_exception(std::current_exception()); } catch (...) {} }
    });
    ready_future.get();
    bool rejected = false;
    try {
        pipe_dense_segment_client::client client(manifest(port), 7, false);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    CHECK(rejected);
    thread.join();
    CHECK(result_future.get() == 1);
}

} // namespace

int main() {
    if (!pipe_transport_init()) return 1;
    test_roundtrip_and_controls();
    test_nextn_not_needed();
    test_nextn_aliased_reconstruction();
    test_interior_taps();
    test_prompt_reuse_disagreement();
    test_hello_validation_failure();
    pipe_transport_shutdown();
    if (g_failed == 0) {
        std::printf("test-pipe-dense-segment-client: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-pipe-dense-segment-client: %d check(s) failed\n", g_failed);
    return 1;
}
