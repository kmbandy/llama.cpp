#include "arg.h"
#include "common.h"
#include "pipe-dense-segment-manifest.h"
#include "pipe-transport.h"
#include "wp-segment-worker.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
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
    "sha256:1111111111111111111111111111111111111111111111111111111111111111";
constexpr const char * ARTIFACT_SHA =
    "sha256:2222222222222222222222222222222222222222222222222222222222222222";

int reserve_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
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
    if (begin == std::string::npos) {
        throw std::runtime_error("test fixture has no checksum");
    }
    const size_t value_begin = begin + key.size();
    const size_t value_end = value.find('"', value_begin);
    value.replace(value_begin, value_end - value_begin, checksum);
    return value;
}

std::string manifest_json(int port) {
    return resign(
        "{\"format\":\"llama.cpp.dense-segment-manifest\",\"version\":1,"
        "\"model_identity_sha256\":\"" + std::string(MODEL_SHA) + "\","
        "\"n_layer\":8,\"n_embd\":4,\"wire_precision\":\"f32\",\"segments\":["
        "{\"id\":3,\"layer_first\":0,\"layer_last\":3,\"host\":\"127.0.0.1\",\"port\":" +
        std::to_string(port) + ",\"stage_gguf\":\"stages/head.gguf\",\"device\":\"HIP0\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_SHA) + "\"},"
        "{\"id\":7,\"layer_first\":4,\"layer_last\":7,\"host\":\"127.0.0.1\",\"port\":" +
        std::to_string(port) + ",\"stage_gguf\":\"/models/tail.gguf\",\"device\":\"HIP1\","
        "\"artifact_sha256\":\"" + std::string(ARTIFACT_SHA) + "\"}],\"checksum\":\"pending\"}");
}

class stub_runtime final : public wp_segment_worker::runtime {
public:
    bool forward(const pipe_segment_fwd_req & request, pipe_segment_fwd_resp & response,
                 uint32_t nextn_width, std::string &) override {
        events.push_back("forward");
        // Record what serve_connection() negotiated, so a test can assert the runtime was
        // told the truth and not just that the frame came out the right size.
        last_nextn_width = nextn_width;
        n_past_value += request.n_tokens;
        response.session_id = request.session_id;
        response.seq_id = request.seq_id;
        response.n_tokens = request.n_tokens;
        response.activations.resize((size_t) request.n_tokens * 4);
        for (size_t i = 0; i < response.activations.size(); ++i) {
            response.activations[i] = request.activations[i] + 1.0f;
        }
        // Interior taps: one [n_tokens, n_embd] block per configured tap, as the real
        // runtime produces. serve_connection() fills in the widths from the config.
        if (!tap_layers.empty()) {
            response.taps.assign((size_t) request.n_tokens * 4 * tap_layers.size(), 0.0f);
            for (size_t i = 0; i < response.taps.size(); ++i) {
                response.taps[i] = 500.0f + (float) i;
            }
        }
        // NEXTN SIDEBAND, exactly as llama_runtime does it: serialize ONLY what the
        // connection negotiated. `alias_nextn` stands in for the real worker's memcmp
        // against the terminal hidden state.
        if (nextn_width > 0) {
            response.nextn_width = nextn_width;
            if (alias_nextn) {
                response.nextn_aliased = 1;
            } else {
                response.nextn.assign((size_t) request.n_tokens * nextn_width, 5.0f);
            }
        }
        return true;
    }

    // mirrors the segment's configured taps, as llama_runtime is constructed in run()
    std::vector<uint32_t> tap_layers;
    // when true, report the sideband as bit-identical to `activations` and ship none
    bool     alias_nextn = false;
    uint32_t last_nextn_width = UINT32_MAX;

    void reset() override {
        events.push_back("reset");
        n_past_value = 0;
    }

    bool trim(uint32_t n_past, std::string &) override {
        events.push_back("trim");
        if (n_past > n_past_value) {
            return false;
        }
        n_past_value = n_past;
        return true;
    }

    uint32_t n_past() const override {
        return n_past_value;
    }

    uint32_t n_past_value = 0;
    std::vector<std::string> events;
};

pipe_segment_hello hello() {
    pipe_segment_hello value;
    value.segment_id = 3;
    value.layer_first = 0;
    value.layer_last = 3;
    value.model_identity_sha256 = MODEL_SHA;
    value.n_embd = 4;
    value.n_vocab = 32;
    value.wire_precision = PIPE_SEGMENT_WIRE_F32;
    value.terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;
    return value;
}

wp_segment_worker::service_config service_config() {
    return { 3, 0, 3, MODEL_SHA, 4, 32, 4, 0, 16,
        PIPE_SEGMENT_CAP_FWD | PIPE_SEGMENT_CAP_RESET | PIPE_SEGMENT_CAP_KV_TRIM |
        PIPE_SEGMENT_CAP_PROMPT_REUSE | PIPE_SEGMENT_CAP_RECURRENT,
        PIPE_SEGMENT_TERMINAL_HIDDEN, /*is_terminal =*/ false };
}

// A segment configured to extract one interior tap, for the tap negotiation tests.
// Layer 2 is inside this fixture's band [0, 3].
wp_segment_worker::service_config tap_service_config() {
    wp_segment_worker::service_config config = service_config();
    config.tap_layers = { 2 };
    return config;
}

// A tail that serves logits-on-head only, for the terminal-kind negotiation test.
wp_segment_worker::service_config terminal_service_config() {
    wp_segment_worker::service_config config = service_config();
    config.is_terminal = true;
    config.terminal_kind = PIPE_SEGMENT_TERMINAL_HIDDEN;
    return config;
}

// A TAIL that can produce a nextn sideband, for the nextn negotiation tests. Only a tail
// has one; nextn_width here is the capability, not the decision.
wp_segment_worker::service_config nextn_service_config() {
    wp_segment_worker::service_config config = terminal_service_config();
    config.nextn_width = 4; // n_embd_out on this fixture
    return config;
}

pipe_channel::received_frame receive(pipe_channel::channel & channel) {
    pipe_channel::received_frame frame;
    CHECK(pipe_channel::channel::harvest({ &channel }, frame, 5000));
    return frame;
}

void test_resolve_segment() {
    const int port = reserve_port();
    const std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("test-wp-segment-worker-" + std::to_string(getpid()));
    std::filesystem::create_directories(root / "stages");
    const std::filesystem::path manifest_path = root / "manifest.json";
    {
        FILE * file = std::fopen(manifest_path.string().c_str(), "wb");
        CHECK(file != nullptr);
        if (file != nullptr) {
            const std::string json = manifest_json(port);
            CHECK(std::fwrite(json.data(), 1, json.size(), file) == json.size());
            std::fclose(file);
        }
    }

    const wp_segment_worker::resolved_segment resolved =
        wp_segment_worker::resolve_segment({ manifest_path, 3, 16 });
    CHECK(resolved.segment.id == 3);
    CHECK(resolved.segment.device() == "HIP0");
    CHECK(resolved.stage_gguf == (root / "stages/head.gguf").lexically_normal());

    const wp_segment_worker::resolved_segment mid =
        wp_segment_worker::resolve_segment({ manifest_path, 7, 16 });
    CHECK(mid.segment.layer_first == 4);
    CHECK(mid.stage_gguf == "/models/tail.gguf");

    bool missing_rejected = false;
    try {
        wp_segment_worker::resolve_segment({ manifest_path, 99, 16 });
    } catch (const std::runtime_error &) {
        missing_rejected = true;
    }
    CHECK(missing_rejected);
    std::filesystem::remove_all(root);
}

void test_fit_off() {
    char default_arg0[] = "test-wp-segment-worker";
    char * default_argv[] = { default_arg0 };
    common_params defaults;
    defaults.model.path = "stage.gguf";
    defaults.fit_params = false;
    CHECK(common_params_parse(1, default_argv, defaults, LLAMA_EXAMPLE_COMPLETION, nullptr));
    CHECK(!defaults.fit_params);

    char arg0[] = "test-wp-segment-worker";
    char arg1[] = "--fit";
    char arg2[] = "off";
    char * argv[] = { arg0, arg1, arg2 };
    common_params params;
    params.model.path = "stage.gguf";
    CHECK(common_params_parse(3, argv, params, LLAMA_EXAMPLE_COMPLETION, nullptr));
    CHECK(params.model.path == "stage.gguf");
    CHECK(!params.fit_params);
}

void test_service_roundtrip(const pipe_segment_hello & hello, const wp_segment_worker::service_config & config) {
    const int port = reserve_port();
    std::promise<void> ready;
    std::future<void> ready_future = ready.get_future();
    std::promise<int> result;
    std::future<int> result_future = result.get_future();
    stub_runtime runtime;
    runtime.tap_layers = config.tap_layers;
    std::thread thread([&]() {
        try {
            pipe_socket_ptr server = pipe_socket_t::create_server("127.0.0.1", port);
            if (!server) {
                throw std::runtime_error("server creation failed");
            }
            ready.set_value();
            pipe_socket_ptr socket = server->accept();
            if (!socket) {
                throw std::runtime_error("server accept failed");
            }
            pipe_channel::channel channel(std::move(socket), "loopback client");
            result.set_value(wp_segment_worker::serve_connection(channel, config, runtime));
        } catch (...) {
            try {
                result.set_exception(std::current_exception());
            } catch (...) {
            }
        }
    });
    ready_future.get();

    {
        pipe_channel::channel channel({ "127.0.0.1", port });
        channel.send_frame(PIPE_SEGMENT_HELLO, 0, pipe_encode_segment_hello(hello));
        channel.flush();
        const pipe_channel::received_frame hello_ack = receive(channel);
        CHECK(hello_ack.type == PIPE_SEGMENT_HELLO_ACK);
        CHECK(pipe_decode_segment_hello_ack(hello_ack.payload.data(), hello_ack.payload.size()).accepted);

        const pipe_segment_ctrl reset = {
            PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_RESET, 9, 4, 0, "",
        };
        channel.send_frame(PIPE_SEGMENT_CTRL, 11, pipe_encode_segment_ctrl(reset));
        channel.flush();
        const pipe_channel::received_frame reset_ack = receive(channel);
        CHECK(reset_ack.type == PIPE_SEGMENT_CTRL_ACK);
        CHECK(pipe_decode_segment_ctrl_ack(reset_ack.payload.data(), reset_ack.payload.size()).n_past == 0);

        pipe_segment_fwd_req request;
        request.session_id = 9;
        request.seq_id = 12;
        request.n_tokens = 2;
        request.n_pos_per_token = 1;
        request.n_seqs = 1;
        request.positions = { 0, 1 };
        request.seq_token_counts = { 2 };
        request.activations = { 1, 2, 3, 4, 5, 6, 7, 8 };
        channel.send_frame(PIPE_SEGMENT_FWD_REQ, 12, pipe_encode_segment_fwd_req(request, 4));
        channel.flush();
        const pipe_channel::received_frame forward = receive(channel);
        CHECK(forward.type == PIPE_SEGMENT_FWD_RESP);
        // Decode with what THIS config negotiated: serve_connection() stamps the tap
        // width and count from the config, so a harness that assumed "no taps" would be
        // rejected by the header check -- which is the check doing its job.
        const int32_t tap_width = config.tap_layers.empty() ? 0 : config.n_embd;
        const pipe_segment_fwd_resp response =
            pipe_decode_segment_fwd_resp(forward.payload.data(), forward.payload.size(), 4, 0,
                                         tap_width, (int32_t) config.tap_layers.size());
        CHECK(response.session_id == 9);
        CHECK(response.seq_id == 12);
        CHECK(response.activations == std::vector<float>({ 2, 3, 4, 5, 6, 7, 8, 9 }));
        // and the taps themselves must arrive: one [n_tokens, n_embd] block per tap,
        // carrying what stub_runtime produced
        CHECK(response.n_taps == (uint32_t) config.tap_layers.size());
        CHECK(response.taps.size() == (size_t) response.n_tokens * tap_width * config.tap_layers.size());
        if (!config.tap_layers.empty()) {
            CHECK(response.tap_width == (uint32_t) config.n_embd);
            CHECK(response.taps.front() == 500.0f);
        }

        const pipe_segment_ctrl reuse = {
            PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_PROMPT_REUSE, 9, 4, 1, "sha256:prefix",
        };
        channel.send_frame(PIPE_SEGMENT_CTRL, 13, pipe_encode_segment_ctrl(reuse));
        channel.flush();
        const pipe_channel::received_frame reuse_ack = receive(channel);
        CHECK(pipe_decode_segment_ctrl_ack(reuse_ack.payload.data(), reuse_ack.payload.size()).n_past == 2);

        const pipe_segment_ctrl trim = {
            PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_KV_TRIM, 9, 4, 1, "",
        };
        channel.send_frame(PIPE_SEGMENT_CTRL, 14, pipe_encode_segment_ctrl(trim));
        channel.flush();
        const pipe_channel::received_frame trim_ack = receive(channel);
        const pipe_segment_ctrl_ack trim_result =
            pipe_decode_segment_ctrl_ack(trim_ack.payload.data(), trim_ack.payload.size());
        CHECK(trim_result.status == PIPE_SEGMENT_CTRL_APPLIED);
        CHECK(trim_result.n_past == 1);
    }

    thread.join();
    CHECK(result_future.get() == 0);
    CHECK(runtime.events == std::vector<std::string>({ "reset", "forward", "trim" }));
}

// NEXTN WIRE DEDUP (PIPE v12). Drives one HELLO + RESET + forward against a tail that
// CAN produce a sideband, and reports what the ACK negotiated and how many bytes the
// forward response actually cost. Returns the encoded response payload so the caller can
// compare frame sizes -- the saving is bytes, and only bytes prove it.
struct nextn_probe {
    uint32_t             ack_nextn_width = UINT32_MAX;
    uint32_t             runtime_nextn_width = UINT32_MAX;
    size_t               response_bytes = 0;
    pipe_segment_fwd_resp response;
};

nextn_probe run_nextn_probe(uint32_t nextn_need, bool alias_nextn) {
    const wp_segment_worker::service_config config = nextn_service_config();
    const int port = reserve_port();
    std::promise<void> ready;
    std::future<void> ready_future = ready.get_future();
    std::promise<int> result;
    std::future<int> result_future = result.get_future();
    stub_runtime runtime;
    runtime.alias_nextn = alias_nextn;
    std::thread thread([&]() {
        try {
            pipe_socket_ptr server = pipe_socket_t::create_server("127.0.0.1", port);
            if (!server) {
                throw std::runtime_error("server creation failed");
            }
            ready.set_value();
            pipe_socket_ptr socket = server->accept();
            if (!socket) {
                throw std::runtime_error("server accept failed");
            }
            pipe_channel::channel channel(std::move(socket), "loopback client");
            result.set_value(wp_segment_worker::serve_connection(channel, config, runtime));
        } catch (...) {
            try {
                result.set_exception(std::current_exception());
            } catch (...) {
            }
        }
    });
    ready_future.get();

    nextn_probe probe;
    {
        pipe_channel::channel channel({ "127.0.0.1", port });
        pipe_segment_hello value = hello();
        value.nextn_need = nextn_need;
        channel.send_frame(PIPE_SEGMENT_HELLO, 0, pipe_encode_segment_hello(value));
        channel.flush();
        const pipe_channel::received_frame ack_frame = receive(channel);
        CHECK(ack_frame.type == PIPE_SEGMENT_HELLO_ACK);
        const pipe_segment_hello_ack ack =
            pipe_decode_segment_hello_ack(ack_frame.payload.data(), ack_frame.payload.size());
        CHECK(ack.accepted);
        probe.ack_nextn_width = ack.nextn_width;

        const pipe_segment_ctrl reset = {
            PIPE_SEGMENT_CTRL_VERSION, PIPE_SEGMENT_CTRL_RESET, 9, 4, 0, "",
        };
        channel.send_frame(PIPE_SEGMENT_CTRL, 11, pipe_encode_segment_ctrl(reset));
        channel.flush();
        CHECK(receive(channel).type == PIPE_SEGMENT_CTRL_ACK);

        pipe_segment_fwd_req request;
        request.session_id = 9;
        request.seq_id = 12;
        request.n_tokens = 2;
        request.n_pos_per_token = 1;
        request.n_seqs = 1;
        request.positions = { 0, 1 };
        request.seq_token_counts = { 2 };
        request.activations = { 1, 2, 3, 4, 5, 6, 7, 8 };
        channel.send_frame(PIPE_SEGMENT_FWD_REQ, 12, pipe_encode_segment_fwd_req(request, 4));
        channel.flush();
        const pipe_channel::received_frame forward = receive(channel);
        CHECK(forward.type == PIPE_SEGMENT_FWD_RESP);
        probe.response_bytes = forward.payload.size();
        // decode with exactly what the ACK negotiated -- a tail that ignored the need
        // flag fails here rather than desynchronising the stream
        probe.response = pipe_decode_segment_fwd_resp(
            forward.payload.data(), forward.payload.size(), 4,
            (int32_t) probe.ack_nextn_width, 0, 0);
    }

    thread.join();
    CHECK(result_future.get() == 0);
    probe.runtime_nextn_width = runtime.last_nextn_width;
    return probe;
}

void test_nextn_negotiation() {
    // (a) NO NEED -- the dspark and no-spec arms. The tail can produce a sideband and
    // must still put ZERO nextn bytes on the wire, and the runtime must be told so.
    const nextn_probe none = run_nextn_probe(/*nextn_need =*/ 0, /*alias_nextn =*/ false);
    CHECK(none.ack_nextn_width == 0);
    CHECK(none.runtime_nextn_width == 0);
    CHECK(none.response.nextn_width == 0);
    CHECK(none.response.nextn.empty());
    CHECK(none.response.nextn_aliased == 0);

    // (b) NEED, distinct payload -- draft-mtp where the sideband is genuinely different
    // data. The full run rides along.
    const nextn_probe full = run_nextn_probe(/*nextn_need =*/ 1, /*alias_nextn =*/ false);
    CHECK(full.ack_nextn_width == 4);
    CHECK(full.runtime_nextn_width == 4);
    CHECK(full.response.nextn_width == 4);
    CHECK(full.response.nextn == std::vector<float>({ 5, 5, 5, 5, 5, 5, 5, 5 }));
    CHECK(full.response.nextn_aliased == 0);

    // (c) NEED, but bit-identical to the terminal hidden state -- the production HIDDEN
    // case. One copy on the wire; the head rebuilds the other.
    const nextn_probe aliased = run_nextn_probe(/*nextn_need =*/ 1, /*alias_nextn =*/ true);
    CHECK(aliased.ack_nextn_width == 4);
    CHECK(aliased.response.nextn_aliased == 1);
    CHECK(aliased.response.nextn.empty());

    // The dedup must be exactly one [n_tokens, nextn_width] f32 block cheaper than
    // shipping both, and the no-need arm must cost the same as the aliased one.
    CHECK(full.response_bytes - aliased.response_bytes == 2 * 4 * sizeof(float));
    CHECK(none.response_bytes == aliased.response_bytes);
}

void test_hello_mismatch(const wp_segment_worker::service_config & config,
                         const pipe_segment_hello & mismatch) {
    const int port = reserve_port();
    std::promise<void> ready;
    std::future<void> ready_future = ready.get_future();
    std::promise<int> result;
    std::future<int> result_future = result.get_future();
    stub_runtime runtime;
    std::thread thread([&]() {
        try {
            pipe_socket_ptr server = pipe_socket_t::create_server("127.0.0.1", port);
            if (!server) {
                throw std::runtime_error("server creation failed");
            }
            ready.set_value();
            pipe_socket_ptr socket = server->accept();
            if (!socket) {
                throw std::runtime_error("server accept failed");
            }
            pipe_channel::channel channel(std::move(socket), "loopback client");
            result.set_value(wp_segment_worker::serve_connection(channel, config, runtime));
        } catch (...) {
            try {
                result.set_exception(std::current_exception());
            } catch (...) {
            }
        }
    });
    ready_future.get();

    {
        pipe_channel::channel channel({ "127.0.0.1", port });
        channel.send_frame(PIPE_SEGMENT_HELLO, 0, pipe_encode_segment_hello(mismatch));
        channel.flush();
        const pipe_channel::received_frame error = receive(channel);
        CHECK(error.type == PIPE_ERROR);
        CHECK(pipe_decode_error(error.payload.data(), error.payload.size()).code == PIPE_ERR_HELLO);
    }

    thread.join();
    CHECK(result_future.get() == 1);
}

} // namespace

int main() {
    common_init();
    if (!pipe_transport_init()) {
        std::fprintf(stderr, "test-wp-segment-worker: transport initialization failed\n");
        return 1;
    }
    test_resolve_segment();
    test_fit_off();
    test_service_roundtrip(hello(), service_config());
    auto mid_hello = hello();
    mid_hello.segment_id = 7;
    mid_hello.layer_first = 4;
    mid_hello.layer_last = 7;
    auto mid_config = service_config();
    mid_config.segment_id = 7;
    mid_config.layer_first = 4;
    mid_config.layer_last = 7;
    test_service_roundtrip(mid_hello, mid_config);

    // a MIDDLE segment returns hidden state either way, so it must accept the
    // legacy logits-on-tail negotiation without complaint
    auto logits_hello = mid_hello;
    logits_hello.terminal_kind = PIPE_SEGMENT_TERMINAL_LOGITS;
    test_service_roundtrip(logits_hello, mid_config);

    {
        pipe_segment_hello mismatch = hello();
        mismatch.layer_last = 2;
        test_hello_mismatch(service_config(), mismatch);
    }
    {
        // a TERMINAL segment must reject the kind it cannot serve -- the payload
        // widths differ by 48x and neither side could detect the mix-up later
        pipe_segment_hello mismatch = hello();
        mismatch.terminal_kind = PIPE_SEGMENT_TERMINAL_LOGITS;
        test_hello_mismatch(terminal_service_config(), mismatch);
    }
    // nextn wire dedup: need/no-need serialization both ways, plus the aliased case
    test_nextn_negotiation();
    {
        // interior taps: a segment configured to extract them serves a matching request
        pipe_segment_hello tap_hello = hello();
        tap_hello.tap_layers = { 2 };
        test_service_roundtrip(tap_hello, tap_service_config());
    }
    {
        // ...and must reject a head that asks for taps it was not configured to extract.
        // Serving nothing instead would leave the head reading a stale buffer, which
        // still yields bit-identical verified tokens -- undetectable downstream.
        pipe_segment_hello mismatch = hello();
        mismatch.tap_layers = { 2 };
        test_hello_mismatch(service_config(), mismatch);
    }
    {
        // the converse: a segment that extracts taps must reject a head expecting none,
        // otherwise it would ship blocks the head never accounts for
        test_hello_mismatch(tap_service_config(), hello());
    }
    pipe_transport_shutdown();
    if (g_failed == 0) {
        std::printf("test-wp-segment-worker: all tests passed\n");
        return 0;
    }
    std::fprintf(stderr, "test-wp-segment-worker: %d check(s) failed\n", g_failed);
    return 1;
}
