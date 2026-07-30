#include "ggml.h"
#include "pipe-protocol.h"
#include "pipe-transport.h"
#include "wp-expert-worker.h"

#include <nlohmann/json.hpp>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

static constexpr int N_EMBD   = 32;
static constexpr int N_FF_EXP = 32;
static constexpr int LAYER    = 3;
static constexpr int N_TOKENS = 2;

void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct TempDir {
    fs::path path;

    TempDir() {
        std::string pattern = (fs::temp_directory_path() / "wp-expert-worker-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char * result = mkdtemp(writable.data());
        if (result == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        path = result;
    }

    ~TempDir() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

void write_json(const fs::path & path, const json & value) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to create " + path.string());
    }
    output << value.dump(2) << '\n';
}

std::vector<float> make_matrix(int expert, int role) {
    std::vector<float> values((size_t) N_EMBD * N_FF_EXP);
    for (int row = 0; row < N_FF_EXP; ++row) {
        for (int col = 0; col < N_EMBD; ++col) {
            const int pattern = (row * 7 + col * 3 + expert * 5 + role * 11) % 19 - 9;
            float value = 0.006f * pattern;
            if (row == col) {
                value += 0.08f + expert * 0.01f + role * 0.005f;
            }
            values[(size_t) row * N_EMBD + col] = value;
        }
    }
    return values;
}

struct Fixture {
    fs::path descriptor;
    fs::path manifest;
    std::map<std::pair<int, std::string>, std::vector<float>> weights;
};

Fixture make_fixture(const fs::path & dir) {
    Fixture fixture;
    fixture.descriptor = dir / "synthetic.expert-descriptor.json";
    fixture.manifest   = dir / "synthetic-experts-manifest.json";
    const fs::path sidecar = dir / "synthetic-00001-of-00001.wpi.json";
    const fs::path blob    = dir / "synthetic-00001-of-00001.wpb";

    constexpr uint64_t role_bytes = (uint64_t) N_EMBD * N_FF_EXP * sizeof(float);
    constexpr uint64_t page_bytes = role_bytes * 3;
    static_assert(page_bytes % 4096 == 0, "synthetic expert page must be O_DIRECT aligned");

    json groups = json::array();
    std::ofstream blob_output(blob, std::ios::binary);
    if (!blob_output) {
        throw std::runtime_error("failed to create synthetic blob");
    }
    uint64_t offset = 0;
    for (int expert = 0; expert < 4; ++expert) {
        json members = json::array();
        for (const auto & role : {
                 std::make_pair(std::string("up"), 1),
                 std::make_pair(std::string("gate"), 2),
                 std::make_pair(std::string("down"), 4) }) {
            const int role_index =
                role.first == "up" ? 0 : (role.first == "gate" ? 1 : 2);
            std::vector<float> matrix = make_matrix(expert, role_index);
            fixture.weights.emplace(
                std::make_pair(expert, role.first), matrix);
            blob_output.write(
                reinterpret_cast<const char *>(matrix.data()),
                (std::streamsize) (matrix.size() * sizeof(float)));
            members.push_back({
                { "role_mask", role.second },
                { "size", role_bytes },
                { "offset", offset },
                { "catalog_name",
                  "blk.3.ffn_" + role.first + "." + std::to_string(expert) + ".weight" },
                { "source_tensor_name", "synthetic." + role.first },
                { "source_file_idx", 0 },
                { "source_file_offset", offset },
            });
            offset += role_bytes;
        }
        groups.push_back({
            { "block_idx", LAYER },
            { "expert_idx", expert },
            { "member_count", 3 },
            { "members", std::move(members) },
        });
    }
    blob_output.close();
    require(offset == page_bytes * 4, "synthetic blob size mismatch");

    const json identity = {
        { "algorithm", "sha256" },
        { "value", "synthetic-expert-worker-test" },
    };
    const json role_shape = { N_EMBD, N_FF_EXP };
    const auto role_desc = [&](const char * role) {
        return json{
            { "ggml_type", (int) GGML_TYPE_F32 },
            { "ggml_type_name", ggml_type_name(GGML_TYPE_F32) },
            { "shape", role_shape },
            { "bytes_per_expert", role_bytes },
            { "source_tensor_name", std::string("synthetic.") + role },
        };
    };
    write_json(fixture.descriptor, {
        { "format", "llama.cpp.weight-pager.expert-descriptor" },
        { "version", 1 },
        { "source_model",
          {
              { "input_model", "synthetic.gguf" },
              { "model_files", { "synthetic.gguf" } },
              { "architecture", "synthetic" },
              { "name", "synthetic" },
          } },
        { "shard_manifest_identity", identity },
        { "retained_expert_range", { { "first", 0 }, { "last", 3 } } },
        { "hparams",
          {
              { "n_layer", 4 },
              { "n_embd", N_EMBD },
              { "n_ff_exp", N_FF_EXP },
              { "n_expert", 4 },
              { "n_expert_used", 2 },
              { "activation", "silu" },
          } },
        { "layers",
          {
              {
                  { "layer", LAYER },
                  { "roles",
                    {
                        { "gate", role_desc("gate") },
                        { "up", role_desc("up") },
                        { "down", role_desc("down") },
                    } },
              },
          } },
    });

    write_json(sidecar, {
        { "format", "llama.cpp.weight-pager.expert-shard-index" },
        { "version", 1 },
        { "blob_file", blob.filename().string() },
        { "shard_index", 0 },
        { "shard_count", 1 },
        { "layer_first", LAYER },
        { "layer_last", LAYER },
        { "group_count", 4 },
        { "blob_bytes", offset },
        { "content_hash", identity },
        { "model_files", { "synthetic.gguf" } },
        { "groups", std::move(groups) },
    });

    write_json(fixture.manifest, {
        { "format", "llama.cpp.weight-pager.expert-shard-manifest" },
        { "version", 1 },
        { "input_model", "synthetic.gguf" },
        { "model_files", { "synthetic.gguf" } },
        { "sharding_mode", "expert-index-range" },
        { "retained_expert_range", { { "first", 0 }, { "last", 3 } } },
        { "total_group_count", 4 },
        { "total_blob_bytes", offset },
        { "shard_count", 1 },
        { "content_hash", identity },
        { "shards",
          {
              {
                  { "blob_file", blob.filename().string() },
                  { "index_file", sidecar.filename().string() },
                  { "shard_index", 0 },
                  { "layer_first", LAYER },
                  { "layer_last", LAYER },
                  { "group_count", 4 },
                  { "blob_bytes", offset },
                  { "content_hash", identity },
              },
          } },
    });
    return fixture;
}

int reserve_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        throw std::runtime_error("socket failed");
    }
    sockaddr_in address{};
    address.sin_family      = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port        = 0;
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

pipe_socket_ptr connect_with_retry(int port) {
    for (int attempt = 0; attempt < 200; ++attempt) {
        pipe_socket_ptr socket = pipe_socket_t::connect("127.0.0.1", port);
        if (socket) {
            return socket;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error("failed to connect to worker");
}

std::vector<float> reference(
        const Fixture & fixture,
        const std::vector<float> & activation,
        const std::vector<pipe_expert_assignment> & assignments) {
    std::vector<float> result((size_t) N_TOKENS * N_EMBD, 0.0f);
    std::vector<float> gate(N_FF_EXP);
    std::vector<float> up(N_FF_EXP);
    std::vector<float> hidden(N_FF_EXP);
    std::vector<float> down(N_EMBD);
    for (const pipe_expert_assignment & assignment : assignments) {
        const auto & gate_weight =
            fixture.weights.at({ assignment.expert_id, "gate" });
        const auto & up_weight =
            fixture.weights.at({ assignment.expert_id, "up" });
        const auto & down_weight =
            fixture.weights.at({ assignment.expert_id, "down" });
        for (int token = 0; token < N_TOKENS; ++token) {
            const float * input = activation.data() + (size_t) token * N_EMBD;
            for (int row = 0; row < N_FF_EXP; ++row) {
                gate[row] = 0.0f;
                up[row]   = 0.0f;
                for (int col = 0; col < N_EMBD; ++col) {
                    gate[row] += gate_weight[(size_t) row * N_EMBD + col] * input[col];
                    up[row]   += up_weight[(size_t) row * N_EMBD + col] * input[col];
                }
                hidden[row] = gate[row] / (1.0f + std::exp(-gate[row])) * up[row];
            }
            for (int row = 0; row < N_EMBD; ++row) {
                down[row] = 0.0f;
                for (int col = 0; col < N_FF_EXP; ++col) {
                    down[row] +=
                        down_weight[(size_t) row * N_FF_EXP + col] * hidden[col];
                }
                result[(size_t) token * N_EMBD + row] +=
                    assignment.weights[token] * down[row];
            }
        }
    }
    return result;
}

void run_test() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    wp_expert_worker::Options options;
    options.shard_manifest = fixture.manifest;
    options.descriptor     = fixture.descriptor;
    options.device         = "CPU";
    options.listen_host    = "127.0.0.1";
    options.listen_port    = port;
    options.slots          = 2;
    options.once           = true;

    int server_result = -1;
    std::exception_ptr server_error;
    std::thread server([&]() {
        try {
            server_result = wp_expert_worker::run(options);
        } catch (...) {
            server_error = std::current_exception();
        }
    });

    try {
        pipe_socket_ptr socket = connect_with_retry(port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive worker HELLO");
        require(type == PIPE_HELLO, "worker did not send HELLO");
        pipe_expert_hello worker_hello =
            pipe_decode_expert_hello(payload.data(), payload.size());
        require(worker_hello.role == PIPE_EXPERT_ROLE_WORKER, "worker HELLO role mismatch");
        require(worker_hello.expert_first == 0, "worker HELLO expert first mismatch");
        require(worker_hello.expert_last == 3, "worker HELLO expert last mismatch");
        require(!worker_hello.model_identity.empty(), "worker HELLO model identity is empty");
        require(worker_hello.shard_identity == "sha256:synthetic-expert-worker-test",
                "worker HELLO shard identity mismatch");
        require(worker_hello.layers == std::vector<int32_t>{ LAYER }, "worker HELLO layers mismatch");

        pipe_expert_hello client_hello = worker_hello;
        client_hello.role         = PIPE_EXPERT_ROLE_CLIENT;
        client_hello.expert_first = -1;
        client_hello.expert_last  = -1;
        client_hello.n_slots      = 0;
        client_hello.layers.clear();
        payload = pipe_encode_expert_hello(client_hello);
        require(pipe_send_frame(
            *socket, PIPE_HELLO, 0, payload.data(), payload.size()), "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO acknowledgement");
        require(type == PIPE_EXPERT_HELLO_ACK && seq_id == 0, "worker did not acknowledge HELLO");
        const pipe_expert_hello_ack ack =
            pipe_decode_expert_hello_ack(payload.data(), payload.size());
        require(ack.accepted, "worker rejected matching HELLO");

        std::vector<float> input((size_t) N_TOKENS * N_EMBD);
        pipe_expert_dispatch_req request;
        request.layer    = LAYER;
        request.n_tokens = N_TOKENS;
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = ((int) (i % 13) - 6) * 0.07f;
            const ggml_fp16_t half = ggml_fp32_to_fp16(input[i]);
            request.activations.push_back((uint16_t) half);
            input[i] = ggml_fp16_to_fp32(half);
        }
        request.assignments = {
            { 0, { 0.5f, 0.0f } },
            { 1, { -0.25f, 0.75f } },
            { 2, { 0.0f, 0.4f } },
        };
        require(request.assignments.size() > 2, "batching regression did not exceed model top-k");
        const std::vector<float> expected =
            reference(fixture, input, request.assignments);

        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(
            *socket, PIPE_EXPERT_DISPATCH_REQ, 42,
            payload.data(), payload.size()), "failed to send dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive partial");
        require(type == PIPE_EXPERT_PARTIAL, "worker did not return a partial");
        require(seq_id == 42, "partial sequence id mismatch");
        const pipe_expert_partial response =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        require(response.layer == LAYER, "partial layer mismatch");
        require(response.n_tokens == N_TOKENS, "partial token count mismatch");
        require(response.partial.size() == expected.size(), "partial shape mismatch");
        for (size_t i = 0; i < expected.size(); ++i) {
            const float actual =
                ggml_fp16_to_fp32((ggml_fp16_t) response.partial[i]);
            const float tolerance = 0.002f + 0.01f * std::fabs(expected[i]);
            if (std::fabs(actual - expected[i]) > tolerance) {
                throw std::runtime_error(
                    "partial mismatch at " + std::to_string(i) +
                    ": actual=" + std::to_string(actual) +
                    " expected=" + std::to_string(expected[i]));
            }
        }

        pipe_expert_dispatch_req rejected = request;
        rejected.assignments = { { 4, { 1.0f, 1.0f } } };
        payload = pipe_encode_expert_dispatch_req(rejected);
        require(pipe_send_frame(
            *socket, PIPE_EXPERT_DISPATCH_REQ, 43,
            payload.data(), payload.size()), "failed to send rejected dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive rejection");
        require(type == PIPE_ERROR, "out-of-range expert was not rejected");
        require(seq_id == 43, "rejection sequence id mismatch");
        const pipe_error error = pipe_decode_error(payload.data(), payload.size());
        require(error.code == PIPE_ERR_EXPERT_RANGE, "wrong rejection error code");

        socket.reset();
    } catch (...) {
        server.join();
        throw;
    }
    server.join();
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    if (server_result != 0) {
        throw std::runtime_error("worker returned failure");
    }

    options.listen_port = reserve_port();
    server_result       = -1;
    server_error        = nullptr;
    std::thread reject_server([&]() {
        try {
            server_result = wp_expert_worker::run(options);
        } catch (...) {
            server_error = std::current_exception();
        }
    });
    try {
        pipe_socket_ptr socket = connect_with_retry(options.listen_port);
        pipe_frame_type type;
        uint64_t        seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive reject-test HELLO");
        pipe_expert_hello client =
            pipe_decode_expert_hello(payload.data(), payload.size());
        client.role           = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first   = -1;
        client.expert_last    = -1;
        client.n_slots        = 0;
        client.layers.clear();
        client.model_identity = "sha256:different-logical-model";
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(
                    *socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send mismatched client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO rejection");
        require(type == PIPE_EXPERT_HELLO_ACK && seq_id == 0, "worker did not explicitly reject HELLO");
        const pipe_expert_hello_ack ack =
            pipe_decode_expert_hello_ack(payload.data(), payload.size());
        require(!ack.accepted, "worker accepted a different logical model");
        require(ack.reason.find("model identity mismatch") != std::string::npos,
                "HELLO rejection did not explain the model mismatch");
        socket.reset();
    } catch (...) {
        reject_server.join();
        throw;
    }
    reject_server.join();
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    if (server_result == 0) {
        throw std::runtime_error("worker accepted a mismatched HELLO");
    }
}

} // namespace

int main() {
    try {
        run_test();
        std::cout << "test-wp-expert-worker: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-expert-worker: " << error.what() << '\n';
        return 1;
    }
}
