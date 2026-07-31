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
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
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
static constexpr int OTHER_LAYER = 4;
static constexpr int N_TOKENS = 2;
static constexpr uint64_t ROLE_BYTES =
    (uint64_t) N_EMBD * N_FF_EXP * sizeof(float);
static constexpr uint64_t PAGE_BYTES = ROLE_BYTES * 3;

static_assert(PAGE_BYTES % 4096 == 0, "synthetic expert page must be O_DIRECT aligned");

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
            { "bytes_per_expert", ROLE_BYTES },
            { "source_tensor_name", std::string("synthetic.") + role },
        };
    };
    const auto layer_desc = [&](int layer) {
        return json{
            { "layer", layer },
            { "roles",
              {
                  { "gate", role_desc("gate") },
                  { "up", role_desc("up") },
                  { "down", role_desc("down") },
              } },
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
              { "n_layer", 5 },
              { "n_embd", N_EMBD },
              { "n_ff_exp", N_FF_EXP },
              { "n_expert", 4 },
              { "n_expert_used", 2 },
              { "activation", "silu" },
          } },
        { "layers",
          { layer_desc(LAYER), layer_desc(OTHER_LAYER) } },
    });

    json shards = json::array();
    uint64_t total_blob_bytes = 0;
    int shard_index = 0;
    for (int layer : { LAYER, OTHER_LAYER }) {
        const std::string stem =
            "synthetic-0000" + std::to_string(shard_index + 1) +
            "-of-00002";
        const fs::path sidecar = dir / (stem + ".wpi.json");
        const fs::path blob    = dir / (stem + ".wpb");
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
                    { "size", ROLE_BYTES },
                    { "offset", offset },
                    { "catalog_name",
                      "blk." + std::to_string(layer) + ".ffn_" +
                      role.first + "." + std::to_string(expert) + ".weight" },
                    { "source_tensor_name", "synthetic." + role.first },
                    { "source_file_idx", 0 },
                    { "source_file_offset", offset },
                });
                offset += ROLE_BYTES;
            }
            groups.push_back({
                { "block_idx", layer },
                { "expert_idx", expert },
                { "member_count", 3 },
                { "members", std::move(members) },
            });
        }
        blob_output.close();
        require(offset == PAGE_BYTES * 4,
                "synthetic shard size mismatch");
        write_json(sidecar, {
            { "format", "llama.cpp.weight-pager.expert-shard-index" },
            { "version", 1 },
            { "blob_file", blob.filename().string() },
            { "shard_index", shard_index },
            { "shard_count", 2 },
            { "layer_first", layer },
            { "layer_last", layer },
            { "group_count", 4 },
            { "blob_bytes", offset },
            { "content_hash", identity },
            { "model_files", { "synthetic.gguf" } },
            { "groups", std::move(groups) },
        });
        shards.push_back({
            { "blob_file", blob.filename().string() },
            { "index_file", sidecar.filename().string() },
            { "shard_index", shard_index },
            { "layer_first", layer },
            { "layer_last", layer },
            { "group_count", 4 },
            { "blob_bytes", offset },
            { "content_hash", identity },
        });
        total_blob_bytes += offset;
        ++shard_index;
    }

    write_json(fixture.manifest, {
        { "format", "llama.cpp.weight-pager.expert-shard-manifest" },
        { "version", 1 },
        { "input_model", "synthetic.gguf" },
        { "model_files", { "synthetic.gguf" } },
        { "sharding_mode", "expert-index-range" },
        { "retained_expert_range", { { "first", 0 }, { "last", 3 } } },
        { "total_group_count", 8 },
        { "total_blob_bytes", total_blob_bytes },
        { "shard_count", 2 },
        { "content_hash", identity },
        { "shards", std::move(shards) },
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

class IoTracker {
public:
    IoTracker() {
        hooks.read_started = [this](int, int) {
            std::unique_lock<std::mutex> lock(mutex);
            ++current;
            ++started;
            peak = std::max(peak, current);
            cv.notify_all();
            if (barrier_target > 0 &&
                !cv.wait_for(lock, std::chrono::seconds(5), [&]() {
                    return peak >= barrier_target;
                })) {
                throw std::runtime_error(
                    "expert reads did not reach the required concurrency");
            }
        };
        hooks.read_finished = [this](int, int) {
            std::lock_guard<std::mutex> lock(mutex);
            --current;
            cv.notify_all();
        };
        hooks.staging_borrowed = [this]() {
            std::lock_guard<std::mutex> lock(mutex);
            ++borrows;
        };
        hooks.slot_reserved = [this](int, int, int slot) {
            std::lock_guard<std::mutex> lock(mutex);
            reserved_slots.push_back(slot);
        };
    }

    void reset(int target) {
        std::lock_guard<std::mutex> lock(mutex);
        require(current == 0, "read tracker reset with a read in flight");
        barrier_target = target;
        started        = 0;
        peak           = 0;
        borrows        = 0;
        reserved_slots.clear();
    }

    int read_count() {
        std::lock_guard<std::mutex> lock(mutex);
        return started;
    }

    int peak_reads() {
        std::lock_guard<std::mutex> lock(mutex);
        return peak;
    }

    int staging_borrows() {
        std::lock_guard<std::mutex> lock(mutex);
        return borrows;
    }

    std::vector<int> reservations() {
        std::lock_guard<std::mutex> lock(mutex);
        return reserved_slots;
    }

    wp_expert_worker::TestHooks hooks;

private:
    std::mutex              mutex;
    std::condition_variable cv;
    int                     barrier_target = 0;
    int                     current        = 0;
    int                     started        = 0;
    int                     peak           = 0;
    int                     borrows        = 0;
    std::vector<int>        reserved_slots;
};

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

const wp_expert_worker::SlotClass & find_class(
        const wp_expert_worker::ResourcePlan & plan,
        uint64_t size) {
    const auto found = std::find_if(
        plan.slot_classes.begin(), plan.slot_classes.end(),
        [&](const wp_expert_worker::SlotClass & slot_class) {
            return slot_class.size == size;
        });
    if (found == plan.slot_classes.end()) {
        throw std::runtime_error("missing planned size class");
    }
    return *found;
}

void test_glm_size_class_plan() {
    static constexpr uint64_t SMALL = 12091392;
    static constexpr uint64_t LARGE = 16318464;
    static constexpr uint64_t MID   = 13959168;
    static constexpr uint64_t TAIL  = 13664256;
    static constexpr int EXPERTS = 256;

    std::vector<wp_expert_worker::ResourcePage> pages;
    pages.reserve((size_t) 76 * EXPERTS);
    for (int layer = 3; layer <= 78; ++layer) {
        const uint64_t size =
            layer == 8 ? LARGE :
            (layer >= 75 && layer <= 77) ? MID :
            layer == 78 ? TAIL : SMALL;
        for (int expert = 0; expert < EXPERTS; ++expert) {
            pages.push_back({ layer, size });
        }
    }

    const wp_expert_worker::ResourcePlan plan =
        wp_expert_worker::plan_resources(
            pages, 1600, 2 * LARGE);
    const wp_expert_worker::ResourcePlan default_plan =
        wp_expert_worker::plan_resources(pages, 1600);
    require(plan.size_classes, "GLM distribution did not produce size classes");
    require(plan.slot_classes.size() == 4, "GLM distribution did not produce four classes");
    require(plan.slot_count > plan.requested_slots,
            "size classes did not recover slots from the mixed-quant distribution");
    require(plan.device_bytes <= plan.device_budget_bytes,
            "size-class plan exceeded the device budget");
    require(plan.staging_buffers == 2, "host budget did not set staging concurrency");
    require(default_plan.staging_buffers == 16,
            "default staging concurrency is not QD16");
    require(default_plan.staging_bytes == 16 * LARGE,
            "default staging allocation bytes mismatch");

    const auto & small = find_class(plan, SMALL);
    const auto & large = find_class(plan, LARGE);
    const auto & mid   = find_class(plan, MID);
    const auto & tail  = find_class(plan, TAIL);
    require(small.pages == 71 * EXPERTS, "small-class demand count mismatch");
    require(large.pages == EXPERTS, "large-class demand count mismatch");
    require(mid.pages == 3 * EXPERTS, "mid-class demand count mismatch");
    require(tail.pages == EXPERTS, "tail-class demand count mismatch");
    for (const auto * slot_class : { &small, &large, &mid, &tail }) {
        require(slot_class->pin_floor == EXPERTS,
                "size-class pin floor missed the worst-case layer");
        require(slot_class->slots >= slot_class->pin_floor,
                "size-class allocation fell below its pin floor");
    }
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
    options.slots             = 4;
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.host_victim_bytes = 8 * PAGE_BYTES;
    options.once           = true;
    IoTracker tracker;
    options.test_hooks = &tracker.hooks;

    wp_expert_worker::Options large_options = options;
    large_options.slots = 1600;
    const wp_expert_worker::ResourcePlan large_resources =
        wp_expert_worker::inspect_resources(large_options);
    require(large_resources.requested_slots == 1600,
            "large worker resource accounting lost requested slot count");
    require(large_resources.staging_buffers == 2,
            "large slot count changed staging concurrency");
    require(large_resources.staging_bytes == 2 * PAGE_BYTES,
            "large slot count changed staging allocation bytes");

    const wp_expert_worker::ResourcePlan resources =
        wp_expert_worker::inspect_resources(options);
    require(resources.staging_buffers == 2,
            "two-buffer host budget did not bound staging concurrency");

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
        require(worker_hello.layers ==
                    std::vector<int32_t>{ LAYER, OTHER_LAYER },
                "worker HELLO layers mismatch");

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
            { 3, { 0.3f, -0.2f } },
        };
        require(request.assignments.size() > (size_t) resources.staging_buffers,
                "dispatch did not exceed staging concurrency");

        const auto dispatch_and_check =
            [&](uint64_t expected_seq,
                const pipe_expert_dispatch_req & dispatch) {
                const std::vector<float> expected =
                    reference(fixture, input, dispatch.assignments);
                payload = pipe_encode_expert_dispatch_req(dispatch);
                require(pipe_send_frame(
                    *socket, PIPE_EXPERT_DISPATCH_REQ, expected_seq,
                    payload.data(), payload.size()),
                    "failed to send dispatch");
                require(pipe_recv_frame(*socket, type, seq_id, payload),
                        "failed to receive partial");
                if (type == PIPE_ERROR) {
                    const pipe_error error =
                        pipe_decode_error(payload.data(), payload.size());
                    throw std::runtime_error(
                        "worker dispatch failed: " + error.msg);
                }
                require(type == PIPE_EXPERT_PARTIAL,
                        "worker did not return a partial");
                require(seq_id == expected_seq,
                        "partial sequence id mismatch");
                const pipe_expert_partial response =
                    pipe_decode_expert_partial(
                        payload.data(), payload.size(), N_EMBD);
                require(response.layer == dispatch.layer,
                        "partial layer mismatch");
                require(response.n_tokens == N_TOKENS,
                        "partial token count mismatch");
                require(response.partial.size() == expected.size(),
                        "partial shape mismatch");
                for (size_t i = 0; i < expected.size(); ++i) {
                    const float actual =
                        ggml_fp16_to_fp32(
                            (ggml_fp16_t) response.partial[i]);
                    const float tolerance =
                        0.002f + 0.01f * std::fabs(expected[i]);
                    if (std::fabs(actual - expected[i]) > tolerance) {
                        throw std::runtime_error(
                            "partial mismatch at " + std::to_string(i) +
                            ": actual=" + std::to_string(actual) +
                            " expected=" + std::to_string(expected[i]));
                    }
                }
            };

        pipe_expert_dispatch_req warm = request;
        warm.layer = OTHER_LAYER;
        tracker.reset(2);
        dispatch_and_check(40, warm);
        require(tracker.read_count() == 4,
                "cold request did not issue one read per miss");
        require(tracker.peak_reads() == 2,
                "cold request did not saturate two staging buffers");
        require(tracker.staging_borrows() == 4,
                "staging buffers did not recycle for excess misses");

        pipe_expert_dispatch_req seed = request;
        seed.layer = LAYER;
        seed.assignments.resize(1);
        tracker.reset(1);
        dispatch_and_check(41, seed);
        require(tracker.read_count() == 1 && tracker.peak_reads() == 1,
                "single-miss seed dispatch read accounting mismatch");

        request.layer = LAYER;
        tracker.reset(2);
        dispatch_and_check(42, request);
        require(tracker.read_count() == 3,
                "mixed request did not preserve its resident hit");
        require(tracker.peak_reads() == 2,
                "mixed request did not overlap its misses");
        const std::vector<int> reservations = tracker.reservations();
        require(reservations.size() == 3,
                "mixed request reserved the wrong number of miss slots");
        const std::set<int> unique_slots(
            reservations.begin(), reservations.end());
        require(unique_slots.size() == reservations.size(),
                "later miss evicted an earlier in-flight slot");

        tracker.reset(0);
        dispatch_and_check(43, request);
        require(tracker.read_count() == 0,
                "all-hit request issued an expert read");
        require(tracker.staging_borrows() == 0,
                "all-hit request borrowed staging");
        require(tracker.reservations().empty(),
                "all-hit request reserved a miss slot");

        tracker.reset(0);
        dispatch_and_check(44, warm);
        require(tracker.read_count() == 0,
                "host victim hit issued an expert read");
        require(tracker.staging_borrows() == 0,
                "host victim hit borrowed staging");

        pipe_expert_dispatch_req rejected = request;
        rejected.assignments = { { 4, { 1.0f, 1.0f } } };
        payload = pipe_encode_expert_dispatch_req(rejected);
        require(pipe_send_frame(
            *socket, PIPE_EXPERT_DISPATCH_REQ, 45,
            payload.data(), payload.size()), "failed to send rejected dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive rejection");
        require(type == PIPE_ERROR, "out-of-range expert was not rejected");
        require(seq_id == 45, "rejection sequence id mismatch");
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

void test_default_off_multi_expert_request() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.once              = true;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive default-off worker HELLO");
        require(type == PIPE_HELLO && seq_id == 0,
                "default-off worker did not send HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(
                    *socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send default-off client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive default-off HELLO acknowledgement");
        require(type == PIPE_EXPERT_HELLO_ACK && seq_id == 0 &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "default-off worker rejected matching HELLO");

        pipe_expert_dispatch_req request;
        request.layer = LAYER;
        request.n_tokens = N_TOKENS;
        request.activations.resize((size_t) N_TOKENS * N_EMBD);
        request.assignments = {
            { 0, { 0.5f, 0.25f } },
            { 1, { 0.5f, 0.75f } },
        };
        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(
                    *socket, PIPE_EXPERT_DISPATCH_REQ, 50,
                    payload.data(), payload.size()),
                "failed to send default-off multi-expert dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive default-off multi-expert partial");
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 50,
                "default-off multi-expert dispatch did not complete");
        socket.reset();
    } catch (...) {
        server.join();
        throw;
    }
    server.join();
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "default-off worker returned failure");
}

} // namespace

int main() {
    try {
        test_glm_size_class_plan();
        run_test();
        test_default_off_multi_expert_request();
        std::cout << "test-wp-expert-worker: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-expert-worker: " << error.what() << '\n';
        return 1;
    }
}
