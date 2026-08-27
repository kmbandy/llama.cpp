#include "ggml.h"
#include "ggml-cpu.h"
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

void test_slice_device_member_layout() {
    // These are the allocation sizes returned by CUDA for the 256-wide DS4
    // MXFP4 slice: up/gate have ne0=4096, while down gets one padded 256-wide
    // quantized row. The blob itself still contains only the three raw members.
    const uint64_t up_gate_bytes = ggml_row_size(GGML_TYPE_MXFP4, 4096) * 256;
    const uint64_t down_bytes = ggml_row_size(GGML_TYPE_MXFP4, 256) * 4096;
    const uint64_t down_alloc = down_bytes + ggml_row_size(GGML_TYPE_MXFP4, 256);
    const std::vector<wp_expert_worker::DeviceMemberLayout> layout =
        wp_expert_worker::plan_device_member_layout(
            { up_gate_bytes, up_gate_bytes, down_alloc }, 128);

    require(up_gate_bytes == 557056 && down_bytes == 557056,
            "DS4 256-wide MXFP4 raw member size changed");
    require(layout.size() == 3 && layout[0].offset == 0 &&
                layout[1].offset == up_gate_bytes &&
                layout[2].offset == 2 * up_gate_bytes,
            "slice device members are not independently placed");
    const uint64_t slot_bytes = layout.back().offset + layout.back().size;
    require(slot_bytes == 1671304 && slot_bytes > 3 * up_gate_bytes,
            "slice slot does not contain CUDA down-row padding");

    const wp_expert_worker::ResourcePlan resources =
        wp_expert_worker::plan_resources(
            { { LAYER, slot_bytes, false, 3 * up_gate_bytes } }, 1,
            3 * up_gate_bytes);
    require(resources.slot_classes.size() == 1 &&
                resources.slot_classes[0].size >= slot_bytes,
            "slice size class does not cover padded member allocations");
    require(resources.staging_buffer_bytes == 3 * up_gate_bytes,
            "slice staging must hold raw blob bytes, not device padding");
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
            // f32 straight through as of PIPE_VERSION 4 -- no f16 round-trip, so
            // the reference input and the wire value are now bit-identical.
            input[i] = ((int) (i % 13) - 6) * 0.07f;
            request.activations.push_back(input[i]);
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
                    // partial is std::vector<float> as of PIPE_VERSION 2. This
                    // used to read it as ggml_fp16_to_fp32((ggml_fp16_t) x),
                    // and ggml_fp16_t is uint16_t -- so the float was TRUNCATED
                    // to an integer first and 0.35f came back as exactly 0.0f.
                    // The assertion could then only pass while every expected
                    // value sat within tolerance of zero, i.e. it was vacuous.
                    const float actual = response.partial[i];
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

        pipe_expert_dispatch_req other = request;
        other.layer = OTHER_LAYER;
        tracker.reset(2);
        dispatch_and_check(40, other);
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
        dispatch_and_check(44, other);
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
        const pipe_expert_partial monolithic =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);

        pipe_expert_dispatch_begin begin;
        begin.layer = request.layer;
        begin.n_tokens = request.n_tokens;
        begin.assignments = request.assignments;
        begin.swiglu_clamp = request.swiglu_clamp;
        payload = pipe_encode_expert_dispatch_begin(begin);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_BEGIN, 51,
                                payload.data(), payload.size()),
                "failed to send split dispatch BEGIN");
        pipe_expert_dispatch_acts acts;
        acts.activations = request.activations;
        payload = pipe_encode_expert_dispatch_acts(acts);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_ACTS, 51,
                                payload.data(), payload.size()),
                "failed to send split dispatch ACTS");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_PARTIAL && seq_id == 51,
                "split dispatch did not complete");
        const pipe_expert_partial split =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        require(split.layer == monolithic.layer && split.n_tokens == monolithic.n_tokens &&
                    split.partial == monolithic.partial,
                "split dispatch partial differs from monolithic dispatch");
        payload = pipe_encode_expert_dispatch_begin(begin);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_BEGIN, 52,
                                payload.data(), payload.size()),
                "failed to send pending split BEGIN");
        // Closing here exercises the pending-batch destructor path. The worker
        // must abandon the batch and release every slot pin before serve_connection exits.
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

// Records WHICH pages were read, and lets the test block until an asynchronous
// a speculative page-in has landed. IoTracker only counts reads; this one needs identities,
// because the whole question is which page got evicted.
struct ReadLog {
    wp_expert_worker::TestHooks hooks;

    ReadLog() {
        hooks.read_started = [this](int layer, int expert) {
            std::lock_guard<std::mutex> lock(mutex);
            reads.emplace_back(layer, expert);
            cv.notify_all();
        };
    }

    bool wait_for_total(size_t n) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(10),
                           [&]() { return reads.size() >= n; });
    }

    size_t total() {
        std::lock_guard<std::mutex> lock(mutex);
        return reads.size();
    }

    size_t count_of(int layer, int expert) {
        std::lock_guard<std::mutex> lock(mutex);
        size_t n = 0;
        for (const auto & read : reads) {
            n += (read.first == layer && read.second == expert) ? 1 : 0;
        }
        return n;
    }

  private:
    std::mutex                        mutex;
    std::condition_variable           cv;
    std::vector<std::pair<int, int>>  reads;
};

// A prefetch hint must (a) actually read the page during the idle window, so the
// dispatch that follows is a hit, and (b) NEVER evict a page demand has touched.
//
// (b) is the one that matters. The pool is filled to capacity with one demand
// page and three speculative ones, then one more is read so an eviction is
// forced. A page stamped from the prefetch band takes the OLDEST SPECULATIVE one; a page
// stamped with a fresh LRU tick would take the DEMAND page instead. The two
// behaviours differ by exactly one extra read of (LAYER, 0), which steps 4 and 6
// pin from both directions.
//
// slots is 4, not 2: the pool refuses a budget below the largest single layer
// request, so 4 is the floor for this fixture. Eviction pressure comes from
// using a second layer rather than from starving the pool.
void test_prefetch_spec_pagein_and_eviction_order(const char * lease) {
    require(setenv("WP_EXPERT_SPEC_PAGEIN", "1", 1) == 0, "failed to arm speculative page-in");
    // Lease OFF for this test. It pins the ORIGINAL two-band invariant -- a
    // speculative page is always the first victim and can never displace a
    // demand-touched one -- which is what kept layers 3+ identical to the digit
    // through every arm. WP_EXPERT_SPEC_LEASE>0 deliberately relaxes exactly
    // that, and is covered separately below.
    require(setenv("WP_EXPERT_SPEC_LEASE", lease, 1) == 0, "failed to set the speculative lease");
    // Host landing left ARMED on purpose, with no host tier configured. A
    // predicted hint must then FALL BACK to the VRAM path -- if it instead goes
    // onto the host queue it can never drain, and the prediction is silently
    // discarded while the arm still looks like it ran. Step 8 below fails with
    // "the predicted hint did not read" when that regresses.
    require(setenv("WP_EXPERT_SPEC_HOST", "1", 1) == 0, "failed to arm host landing");
    const bool leased = std::string(lease) != "0";
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    ReadLog reads;
    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;   // the floor for this fixture; see the note above
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.once              = true;
    options.test_hooks        = &reads.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive spec worker HELLO");
        require(type == PIPE_HELLO && seq_id == 0, "spec worker did not send HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send spec client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "spec worker rejected matching HELLO");

        const auto dispatch_one = [&](int32_t layer, int32_t expert, uint64_t seq) {
            pipe_expert_dispatch_req request;
            request.layer       = layer;
            request.n_tokens    = N_TOKENS;
            request.activations.resize((size_t) N_TOKENS * N_EMBD);
            request.assignments = { { expert, std::vector<float>(N_TOKENS, 0.5f) } };
            std::vector<uint8_t> buf = pipe_encode_expert_dispatch_req(request);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, seq, buf.data(), buf.size()),
                    "failed to send spec-test dispatch");
            std::vector<uint8_t> reply;
            pipe_frame_type reply_type;
            uint64_t reply_seq = 0;
            require(pipe_recv_frame(*socket, reply_type, reply_seq, reply),
                    "failed to receive spec-test partial");
            require(reply_type == PIPE_EXPERT_PARTIAL && reply_seq == seq,
                    "spec-test dispatch did not complete");
        };

        const auto hint_p = [&](int32_t layer, std::vector<int32_t> experts,
                                uint32_t provenance) {
            pipe_expert_prefetch_hint frame;
            frame.layer      = layer;
            frame.provenance = provenance;
            frame.expert_ids = std::move(experts);
            std::vector<uint8_t> buf = pipe_encode_expert_prefetch_hint(frame);
            require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0, buf.data(), buf.size()),
                    "failed to send prefetch hint");
        };
        const auto hint = [&](int32_t layer, std::vector<int32_t> experts) {
            hint_p(layer, std::move(experts), PIPE_HINT_CERTAIN);
        };

        // 1. DEMAND (LAYER, 0). One read; its slot enters the demand band.
        dispatch_one(LAYER, 0, 100);
        require(reads.count_of(LAYER, 0) == 1, "demand dispatch did not read its page exactly once");
        require(reads.total() == 1, "demand dispatch read more than its own page");

        // 2. SPECULATE (LAYER, 1..3). The pool is now full: 1 demand + 3 prefetched.
        //    Ascending order on the wire, so (LAYER,1) is the OLDEST speculative page.
        hint(LAYER, { 1, 2, 3 });
        require(reads.wait_for_total(4), "prefetch hints did not page in during the idle window");
        require(reads.count_of(LAYER, 1) == 1 && reads.count_of(LAYER, 2) == 1 &&
                    reads.count_of(LAYER, 3) == 1,
                "the speculative read took the wrong pages");

        // 3. SPECULATE one more page. Every slot is valid, so this MUST evict.
        //    Prefetch band => victim is (LAYER,1), the oldest speculative page.
        //    Fresh tick    => victim would be (LAYER,0), the demand page.
        hint(OTHER_LAYER, { 0 });
        require(reads.wait_for_total(5), "the forcing prefetch hint did not read");
        require(reads.count_of(OTHER_LAYER, 0) == 1, "the forcing speculative read took the wrong page");

        // 4. THE ASSERTION. (LAYER,0) was demanded and never re-demanded, so it
        //    must still be resident: no second read of it.
        dispatch_one(LAYER, 0, 101);
        if (leased) {
            // THE LEASE'S WHOLE POINT, AND ITS WHOLE COST. A leased speculative
            // page outranks a cold demand page, so the demand page is what goes.
            // That is pool pollution by definition -- bounded to the lease
            // window.
            require(reads.count_of(LAYER, 0) == 2,
                    "the lease did not protect the speculative pages -- demand page survived");
        } else {
            require(reads.count_of(LAYER, 0) == 1,
                    "a prefetch evicted a demand-touched page -- the prefetch LRU band is not holding");
        }

        // 5. And it paid off: (LAYER,3) was speculatively read and never evicted, so
        //    demanding it must not read again.
        dispatch_one(LAYER, 3, 102);
        if (!leased) {
            require(reads.count_of(LAYER, 3) == 1,
                    "a speculatively paged-in expert was not reused by the dispatch that followed");
        }

        // 6. (LAYER,1) is the page that should have gone, so demanding it reads
        //    again. This pins WHICH page was evicted, not merely that one was.
        // Only pinned for lease=0. With a lease the pool is 4 slots against a
        // demand page plus three leased speculative ones, so SOMETHING must go
        // once the demand page is re-read -- the lease reorders candidates, it
        // never removes them. Which leased page goes is not determined by this
        // sequence, and asserting it would be pinning an accident.
        if (!leased) {
            dispatch_one(LAYER, 1, 103);
            require(reads.count_of(LAYER, 1) == 2,
                    "the oldest speculative page was not the victim -- eviction order within the prefetch band is wrong");
        }

        // 8. PROVENANCE PRICES RESIDENCY. A CERTAIN page and a PREDICTED one,
        //    then enough eviction pressure to outlive the PREDICTED lease (4)
        //    but not the CERTAIN one (64). The guess must be the victim.
        //
        //    The demand pages cycling below are unleased and carry a use count,
        //    so they rank ABOVE an expired speculative page -- which is what
        //    makes the expired prediction, and only it, the thing that goes.
        //
        //    Discriminating by construction: with the two leases EQUAL the victim
        //    falls to tick order, and the CERTAIN page was hinted first, so it
        //    holds the older tick and goes instead. Run with
        //    WP_EXPERT_SPEC_LEASE_PREDICTED=64 and this fails.
        if (leased) {
            // Baselines captured BEFORE each hint. Reading reads.total() as the
            // argument to wait_for_total is a race: if the read has already
            // landed the baseline is already incremented and the wait is for one
            // more that never comes.
            const size_t before_certain = reads.total();
            hint_p(OTHER_LAYER, { 1 }, PIPE_HINT_CERTAIN);
            require(reads.wait_for_total(before_certain + 1), "the certain hint did not read");
            const size_t certain_after_hint = reads.count_of(OTHER_LAYER, 1);
            const size_t before_predicted = reads.total();
            hint_p(OTHER_LAYER, { 2 }, PIPE_HINT_PREDICTED);
            require(reads.wait_for_total(before_predicted + 1), "the predicted hint did not read");
            const size_t predicted_after_hint = reads.count_of(OTHER_LAYER, 2);

            // Enough evictions to expire the predicted lease of 4 AND then keep
            // evicting, because expiry only makes a page ELIGIBLE -- something
            // still has to come along and take it. Five was not enough: the lease
            // ran out on the last eviction and nothing followed.
            for (uint64_t seq = 200; seq < 216; ++seq) {
                dispatch_one(LAYER, (int32_t) (seq % 4), seq);
            }

            dispatch_one(OTHER_LAYER, 1, 220);
            dispatch_one(OTHER_LAYER, 2, 221);
            const size_t certain_reread   = reads.count_of(OTHER_LAYER, 1) - certain_after_hint;
            const size_t predicted_reread = reads.count_of(OTHER_LAYER, 2) - predicted_after_hint;
            require(predicted_reread > certain_reread,
                    "a PREDICTED page outlived a CERTAIN one -- provenance is not pricing residency");
        }

        // Use-count ranking is a separate concern from the lease, and a live
        // lease reorders these victims, so this pins the policy on its own.
        if (!leased) {
            // 7. USE COUNT BEATS RECENCY. Ask for (LAYER,0) repeatedly so its count
            //    climbs well above everything else, then touch three other pages so
            //    it is the LEAST RECENTLY used of the four. Under LRU it is the next
            //    victim. Under use-count ranking it is the last thing to go.
            for (uint64_t seq = 110; seq < 116; ++seq) {
                dispatch_one(LAYER, 0, seq);
            }
            require(reads.count_of(LAYER, 0) == 1, "repeated demand re-read a resident page");
            dispatch_one(LAYER, 1, 120);
            dispatch_one(LAYER, 2, 121);
            dispatch_one(LAYER, 3, 122);
            // (LAYER,0) is now the oldest of the four by tick and the hottest by use.
            dispatch_one(OTHER_LAYER, 0, 123);   // forces one eviction
            dispatch_one(LAYER, 0, 124);
            require(reads.count_of(LAYER, 0) == 1,
                    "the most-used page was evicted -- ranking is falling back to pure recency");
        }

        socket.reset();
    } catch (...) {
        server.join();
        unsetenv("WP_EXPERT_SPEC_PAGEIN");
        unsetenv("WP_EXPERT_SPEC_LEASE");
        unsetenv("WP_EXPERT_SPEC_HOST");
        // The worker's own failure is the useful one. Without this, a worker
        // that never started surfaces only as "failed to connect", which points
        // at the socket instead of at the reason.
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        throw;
    }
    server.join();
    require(unsetenv("WP_EXPERT_SPEC_PAGEIN") == 0, "failed to disarm speculative page-in");
    require(unsetenv("WP_EXPERT_SPEC_LEASE") == 0, "failed to clear the speculative lease");
    require(unsetenv("WP_EXPERT_SPEC_HOST") == 0, "failed to clear the host-landing flag");
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "spec worker returned failure");
}

// With WP_EXPERT_SPEC_PAGEIN unset (the default) a hint must be accepted and ignored:
// no read, no eviction, so a run is byte-for-byte the config of record while
// still reporting what the spine offered.
// WP_HINT_LOG must be set BEFORE the worker is constructed -- the FILE * is
// initialised once, in a member initialiser. Unset on the way out so the flag
// does not leak into the tests that follow.
struct ScopedEnv {
    const char * name;
    ScopedEnv(const char * n, const std::string & value) : name(n) {
        setenv(name, value.c_str(), 1);
    }
    ~ScopedEnv() { unsetenv(name); }
};

// A PREDICTED hint must land in HOST RAM and take no VRAM slot at all.
//
// The lease can only make a guess give a slot up AFTER taking it; landing in the
// host arena means it never competes for one. The check is therefore about what
// is STILL RESIDENT in VRAM after the guess arrives, not about read counts --
// the read happens either way, it is the slot that must not be spent.
void test_predicted_hint_lands_in_host_ram() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    require(setenv("WP_EXPERT_SPEC_PAGEIN", "1", 1) == 0, "failed to arm speculative page-in");
    require(setenv("WP_EXPERT_SPEC_HOST", "1", 1) == 0, "failed to arm host landing");

    ReadLog reads;
    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.host_victim_bytes = 8 * PAGE_BYTES;   // room for the guesses
    options.once              = true;
    options.test_hooks        = &reads.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK, "worker did not acknowledge HELLO");

        const auto dispatch_one = [&](int32_t layer, int32_t expert, uint64_t seq) {
            pipe_expert_dispatch_req request;
            request.layer       = layer;
            request.n_tokens    = N_TOKENS;
            request.activations.resize((size_t) N_TOKENS * N_EMBD);
            request.assignments = { { expert, std::vector<float>(N_TOKENS, 0.5f) } };
            std::vector<uint8_t> buf = pipe_encode_expert_dispatch_req(request);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, seq, buf.data(), buf.size()),
                    "failed to send dispatch");
            require(pipe_recv_frame(*socket, type, seq_id, buf) && type == PIPE_EXPERT_PARTIAL,
                    "dispatch did not complete");
        };
        const auto hint_p = [&](int32_t layer, std::vector<int32_t> experts, uint32_t prov) {
            pipe_expert_prefetch_hint frame;
            frame.layer      = layer;
            frame.provenance = prov;
            frame.expert_ids = std::move(experts);
            std::vector<uint8_t> buf = pipe_encode_expert_prefetch_hint(frame);
            require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0, buf.data(), buf.size()),
                    "failed to send prefetch hint");
        };

        // Fill VRAM with four demand pages, then predict four more. If a guess
        // took a slot, one of the demand pages would be gone and re-dispatching
        // it would read again.
        for (int32_t e = 0; e < 4; ++e) {
            dispatch_one(LAYER, e, 100 + e);
        }
        require(reads.total() == 4, "the four demand pages did not read exactly once each");

        hint_p(OTHER_LAYER, { 0, 1, 2, 3 }, PIPE_HINT_PREDICTED);
        require(reads.wait_for_total(8), "the predicted hints never read");

        for (int32_t e = 0; e < 4; ++e) {
            dispatch_one(LAYER, e, 200 + e);
        }
        require(reads.total() == 8,
                "a PREDICTED page displaced a demand page from VRAM -- it should have "
                "landed in host RAM and taken no slot");

        // And it is genuinely reachable: demanding one promotes from the host
        // arena rather than reading NVMe again.
        dispatch_one(OTHER_LAYER, 0, 300);
        require(reads.count_of(OTHER_LAYER, 0) == 1,
                "a predicted page was re-read from disk instead of promoted from host RAM");
        socket.reset();
    } catch (...) {
        server.join();
        unsetenv("WP_EXPERT_SPEC_PAGEIN");
        unsetenv("WP_EXPERT_SPEC_HOST");
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        throw;
    }
    server.join();
    require(unsetenv("WP_EXPERT_SPEC_PAGEIN") == 0, "failed to disarm speculative page-in");
    require(unsetenv("WP_EXPERT_SPEC_HOST") == 0, "failed to clear host landing");
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "host-landing worker returned failure");
}

// THE HINT LOG MUST NOT FILE A SPECULATIVE READ AS A DEMAND ONE.
//
// A speculative page-in is logged "S" at submit. Its harvest runs through the
// same drain_one_read as a demand batch, and until 2026-08-06 that drain ALSO
// wrote a "D" line for it -- so every speculative read appeared twice, once as
// the cost and once as the demand read it existed to prevent. Under the async
// path the harvest runs INSIDE ensure_batch, i.e. AFTER the current request's
// "R" line, so the classifier saw S..R..D for the same page and filed a USED
// page as LATE. That artifact is what made asynchronous speculative reads look
// like they had made the used-rate worse (686 -> 431 on identical behaviour).
//
// The invariant pinned here: a page that was speculatively read and then HIT by
// the dispatch that follows produces exactly one S and NO D; a page the
// dispatch had to read on demand produces exactly one D.
void test_spec_pagein_logs_s_not_d() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    const fs::path hint_log = temp.path / "hint.txt";
    const ScopedEnv hint_log_env("WP_HINT_LOG", hint_log.string());
    const ScopedEnv spec_env("WP_EXPERT_SPEC_PAGEIN", "1");

    ReadLog reads;
    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.once              = true;
    options.test_hooks        = &reads.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK,
                "worker did not acknowledge HELLO");

        const auto dispatch_one = [&](int32_t layer, int32_t expert, uint64_t seq) {
            pipe_expert_dispatch_req request;
            request.layer       = layer;
            request.n_tokens    = N_TOKENS;
            request.activations.resize((size_t) N_TOKENS * N_EMBD);
            request.assignments = { { expert, std::vector<float>(N_TOKENS, 0.5f) } };
            std::vector<uint8_t> buf = pipe_encode_expert_dispatch_req(request);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, seq, buf.data(), buf.size()),
                    "failed to send dispatch");
            require(pipe_recv_frame(*socket, type, seq_id, buf) && type == PIPE_EXPERT_PARTIAL,
                    "dispatch did not complete");
        };

        // 1. A demand read: exactly the event a "D" line is FOR.
        dispatch_one(LAYER, 0, 100);
        require(reads.count_of(LAYER, 0) == 1, "the demand dispatch did not read its page");

        // 2. A speculative read of (LAYER, 1) in the idle window.
        pipe_expert_prefetch_hint hint;
        hint.layer      = LAYER;
        hint.expert_ids = { 1 };
        payload = pipe_encode_expert_prefetch_hint(hint);
        require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0, payload.data(), payload.size()),
                "failed to send prefetch hint");
        require(reads.wait_for_total(2), "the hinted page was never speculatively read");

        // 3. The dispatch that uses it. Whether the read is still in flight
        //    (ensure_batch waits for it) or already harvested by the idle pump,
        //    the batch is retired before this reply arrives -- so by the time
        //    the log is read below, a spurious harvest-side "D" would be there.
        dispatch_one(LAYER, 1, 101);
        require(reads.count_of(LAYER, 1) == 1,
                "the speculatively read page was re-read by the dispatch that used it");

        // Read the log with the socket still open, exactly like the disarmed
        // test: nothing below may depend on a clean close.
        std::vector<std::string> lines;
        {
            std::ifstream in(hint_log);
            require(in.good(), "WP_HINT_LOG was never created");
            for (std::string line; std::getline(in, line); ) {
                lines.push_back(line);
            }
        }
        std::vector<std::string> s_lines, d_lines;
        for (const std::string & line : lines) {
            if (!line.empty() && line[0] == 'S') s_lines.push_back(line);
            if (!line.empty() && line[0] == 'D') d_lines.push_back(line);
        }
        require(s_lines.size() == 1 &&
                    s_lines.front() == "S " + std::to_string(LAYER) + " 1",
                "the speculative read was not logged as exactly one S line");
        // THE ASSERTION THIS TEST EXISTS FOR. The only demand read in this
        // sequence is (LAYER, 0); a second D line means the harvest of the
        // speculative batch logged its page-in as a demand read.
        require(d_lines.size() == 1,
                "a speculative page-in was ALSO logged as a demand read -- "
                "the classifier will file every used speculative page as LATE");
        require(d_lines.front() == "D " + std::to_string(LAYER) + " 0",
                "the demand D line is for the wrong page");
        socket.reset();
    } catch (...) {
        server.join();
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        throw;
    }
    server.join();
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "spec-log worker returned failure");
}

void test_prefetch_hint_without_spec_reads_nothing() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    // The counters must be DURABLE, not merely printed. report_prefetch_hints()
    // writes to stderr only on a clean close and the harness SIGKILLs workers,
    // so arm 1 (2026-08-05) produced no foreign_expert number at all -- the one
    // counter that proves spine and worker resolve (layer, expert) through the
    // same static hash.
    const fs::path hint_log = temp.path / "hint.txt";
    const ScopedEnv hint_log_env("WP_HINT_LOG", hint_log.string());

    ReadLog reads;
    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.once              = true;
    options.test_hooks        = &reads.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK,
                "worker did not acknowledge HELLO");

        pipe_expert_prefetch_hint hint;
        hint.layer      = LAYER;
        hint.expert_ids = { 0, 1, 2, 3 };
        payload = pipe_encode_expert_prefetch_hint(hint);
        require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0, payload.data(), payload.size()),
                "failed to send prefetch hint");

        // A malformed hint must not kill the session either -- the next dispatch
        // still has to work. Truncated payload: valid header, missing ids.
        const std::vector<uint8_t> truncated(8, 0);
        require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0,
                                truncated.data(), truncated.size()),
                "failed to send malformed prefetch hint");

        pipe_expert_dispatch_req request;
        request.layer       = LAYER;
        request.n_tokens    = N_TOKENS;
        request.activations.resize((size_t) N_TOKENS * N_EMBD);
        request.assignments = { { 0, std::vector<float>(N_TOKENS, 0.5f) } };
        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 7, payload.data(), payload.size()),
                "failed to send dispatch after hints");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "the session did not survive an ignored and a malformed hint");
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 7,
                "dispatch after hints did not complete");

        // Exactly the one page the DISPATCH needed. Nothing speculated.
        require(reads.total() == 1, "a hint read pages with speculative page-in disarmed");
        require(reads.count_of(LAYER, 0) == 1, "the dispatch did not read its own page");

        // READ THE LOG WITH THE SOCKET STILL OPEN. That is the whole point: the
        // worker has not closed, so report_prefetch_hints() cannot have run, and
        // anything on disk here would equally have survived a SIGKILL. Checking
        // after the join would pass even with the old print-on-close-only code.
        std::vector<std::string> lines;
        {
            std::ifstream in(hint_log);
            require(in.good(), "WP_HINT_LOG was never created");
            for (std::string line; std::getline(in, line); ) {
                lines.push_back(line);
            }
        }
        const auto tagged = [&](char tag) {
            std::vector<std::string> out;
            for (const std::string & line : lines) {
                if (!line.empty() && line[0] == tag) {
                    out.push_back(line);
                }
            }
            return out;
        };
        const std::vector<std::string> c = tagged('C');
        const std::vector<std::string> h = tagged('H');
        const std::vector<std::string> r = tagged('R');
        const std::vector<std::string> d = tagged('D');
        const std::vector<std::string> s = tagged('S');

        // One counter line per hint frame: the valid one, then the malformed one.
        require(c.size() == 2, "WP_HINT_LOG did not have one flushed counter line per hint frame");
        require(c.back().find("frames=1 experts=4") != std::string::npos,
                "WP_HINT_LOG counters do not match the hints that were sent");
        require(c.back().find("malformed=1") != std::string::npos,
                "WP_HINT_LOG did not record the malformed hint");
        // The routing-agreement check itself: every hinted expert belongs to this
        // worker's shard, so both foreign counters must be zero.
        require(c.back().find("foreign_layer=0 foreign_expert=0") != std::string::npos,
                "WP_HINT_LOG reported a foreign layer or expert for an in-shard hint");

        // The ids, which are what make mispredict and late separable at all. A
        // counter can say 4 experts were offered; only H says WHICH, and only R
        // says which were then actually selected.
        require(h.size() == 1, "WP_HINT_LOG did not record the hinted expert ids");
        require(h.front() == "H " + std::to_string(LAYER) + " 0 1 2 3",
                "WP_HINT_LOG hinted ids do not match the frame that was sent");
        require(r.size() == 1, "WP_HINT_LOG did not record the dispatch reference stream");
        require(r.front() == "R " + std::to_string(LAYER) + " 0",
                "WP_HINT_LOG reference ids do not match the dispatch that was sent");

        // With speculation disarmed the ONE page read must be a demand read, and
        // there must be no speculative read at all. This is the arm-1 invariant
        // -- hints on, reads unchanged -- now checkable from the log itself.
        require(s.empty(), "WP_HINT_LOG recorded a speculative page-in with speculation disarmed");
        require(d.size() == 1, "WP_HINT_LOG did not record the dispatch's demand page-in");
        require(d.front() == "D " + std::to_string(LAYER) + " 0",
                "WP_HINT_LOG demand page-in does not match the page the dispatch needed");
        // Ordering is the property the single stream exists to preserve: the
        // prediction must precede the reference, which must precede the read it
        // provokes. Without this, "late" is not derivable from the file.
        const auto index_of = [&](const std::string & want) {
            for (size_t i = 0; i < lines.size(); ++i) {
                if (lines[i] == want) return (long) i;
            }
            return -1L;
        };
        require(index_of(h.front()) < index_of(r.front()) &&
                    index_of(r.front()) < index_of(d.front()),
                "WP_HINT_LOG events are out of order: hint must precede reference must precede read");
        socket.reset();
    } catch (...) {
        server.join();
        throw;
    }
    server.join();
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "worker returned failure after ignored hints");
}

// WP_EXPERT_SPEC_MAX_INFLIGHT: the pump gate that used to hard-serialize the
// speculative page-in path to one batch at a time is now configurable. This
// pins two things: (a) the DEFAULT (the env var unset) is byte-identical to
// the old behaviour -- reads never overlap -- and (b) raising the cap to N
// really does let N batches read concurrently, not just N pages queued that
// still drain one at a time.
//
// Concurrency is proven with a gate hook, not a timing guess: read_started
// blocks (up to a short bound) until `target` reads are simultaneously
// inside it. If the implementation only ever has one batch in flight, the
// second read_started call can never fire while the first is still blocked
// there, so peak is pinned at 1 by construction, not by luck. The bound is
// short (not the unbounded wait a real deadlock-detector would use) because
// a hook that fails to unblock only means the test's own assertion on `peak`
// fails afterward -- read_started's caller already wraps this in a try/catch
// (see read_worker), so timing out here can never crash the test binary.
struct ConcurrencyGate {
    std::mutex              mutex;
    std::condition_variable cv;
    int                     current        = 0;
    int                     peak           = 0;
    int                     barrier_target = 0;
    std::chrono::milliseconds wait_limit{400};
    wp_expert_worker::TestHooks hooks;

    ConcurrencyGate() {
        hooks.read_started = [this](int, int) {
            std::unique_lock<std::mutex> lock(mutex);
            ++current;
            peak = std::max(peak, current);
            cv.notify_all();
            if (barrier_target > 0) {
                // No throw on timeout, unlike IoTracker's barrier: this test
                // reads `peak` afterward instead of treating "did not reach
                // the target" as itself the failure, so a plain timeout here
                // must be harmless, not fatal.
                cv.wait_for(lock, wait_limit,
                            [&]() { return peak >= barrier_target; });
            }
        };
        hooks.read_finished = [this](int, int) {
            std::lock_guard<std::mutex> lock(mutex);
            --current;
            cv.notify_all();
        };
    }

    int peak_reads() {
        std::lock_guard<std::mutex> lock(mutex);
        return peak;
    }
};

void test_spec_max_inflight(const char * env_value, int hinted_experts,
                            int barrier_target, int expected_peak,
                            const char * failure_message) {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    require(setenv("WP_EXPERT_SPEC_PAGEIN", "1", 1) == 0, "failed to arm speculative page-in");
    require(setenv("WP_EXPERT_SPEC_CHUNK", "1", 1) == 0,
            "failed to pin the spec chunk to one page per submit");
    if (env_value != nullptr) {
        require(setenv("WP_EXPERT_SPEC_MAX_INFLIGHT", env_value, 1) == 0,
                "failed to set WP_EXPERT_SPEC_MAX_INFLIGHT");
    } else {
        unsetenv("WP_EXPERT_SPEC_MAX_INFLIGHT");   // exercise the true default
    }

    ConcurrencyGate gate;
    gate.barrier_target = barrier_target;
    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;   // floor for this fixture -- see the note above
    // Each in-flight read holds a staging buffer for its duration (see
    // StagingPool::borrow() in read_worker) -- the staging pool, not
    // WP_EXPERT_SPEC_MAX_INFLIGHT, would otherwise be the concurrency ceiling
    // actually being measured. Size it to comfortably clear barrier_target so
    // this test proves the BATCH cap, not an unrelated buffer shortage.
    options.host_budget_bytes = (uint64_t) (barrier_target + 1) * PAGE_BYTES;
    options.once              = true;
    options.test_hooks        = &gate.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK, "worker did not acknowledge HELLO");

        std::vector<int32_t> experts;
        for (int32_t e = 0; e < hinted_experts; ++e) {
            experts.push_back(e);
        }
        pipe_expert_prefetch_hint frame;
        frame.layer      = LAYER;
        frame.provenance = PIPE_HINT_CERTAIN;
        frame.expert_ids = experts;
        payload = pipe_encode_expert_prefetch_hint(frame);
        require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0, payload.data(), payload.size()),
                "failed to send prefetch hint");

        // Give the idle pump real time to run: it only fires between pipe
        // frames (see await_request), so there is nothing to block on here --
        // the gate itself is what makes this deterministic rather than the
        // sleep. The sleep only needs to outlast wait_limit plus however many
        // pump ticks it takes to submit `hinted_experts` one-page batches.
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        socket.reset();
    } catch (...) {
        server.join();
        unsetenv("WP_EXPERT_SPEC_PAGEIN");
        unsetenv("WP_EXPERT_SPEC_CHUNK");
        unsetenv("WP_EXPERT_SPEC_MAX_INFLIGHT");
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        throw;
    }
    server.join();
    require(unsetenv("WP_EXPERT_SPEC_PAGEIN") == 0, "failed to disarm speculative page-in");
    require(unsetenv("WP_EXPERT_SPEC_CHUNK") == 0, "failed to clear the spec chunk override");
    require(unsetenv("WP_EXPERT_SPEC_MAX_INFLIGHT") == 0,
            "failed to clear WP_EXPERT_SPEC_MAX_INFLIGHT");
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "spec-max-inflight worker returned failure");
    require(gate.peak_reads() == expected_peak, failure_message);
}


// REGRESSION: a demand dispatch for a page that is CURRENTLY being read by an
// in-flight speculative batch must wait for that read and reuse it -- never
// hang, never throw, never read the page twice. This is the exact path
// investigated for the s0 (ROCm, 9 MiB pages, 22.9 GiB pool) production
// failure: s0 is the leg most likely to still have a speculative batch in
// flight when the next demand request lands (wide pages -> slow reads), so it
// is the only leg that reliably exercises ensure_batch's demand-path
// interlock against a REAL in-flight read, even at the default
// WP_EXPERT_SPEC_MAX_INFLIGHT=1. s1/s2's narrower pages read fast enough that
// the interlock's blocking branch was essentially never taken there.
//
// The hook artificially stalls the speculative read for exactly the page this
// test then demands, so the demand dispatch is GUARANTEED to observe
// spec_in_flight_for(page)==true and take the targeted-wait branch in
// spec_pagein_poll(false, page) -- the branch the existing spec tests never
// force, because their reads always finish before the next request lands.
struct DelayedReadLog {
    std::mutex                       mutex;
    std::condition_variable          cv;
    std::vector<std::pair<int, int>> reads;
    int                               delay_layer  = -1;
    int                               delay_expert = -1;
    std::chrono::milliseconds        delay{0};
    bool                              started_signal = false;
    wp_expert_worker::TestHooks      hooks;

    DelayedReadLog() {
        hooks.read_started = [this](int layer, int expert) {
            bool is_target = false;
            {
                std::lock_guard<std::mutex> lock(mutex);
                reads.emplace_back(layer, expert);
                is_target = (layer == delay_layer && expert == delay_expert);
                if (is_target) {
                    started_signal = true;
                }
            }
            cv.notify_all();
            // Deliberately OUTSIDE the lock: this stalls the reader thread to
            // hold the window open, not the bookkeeping that other threads
            // (there are none here but the dispatch thread reading `reads`)
            // need to make progress.
            if (is_target && delay.count() > 0) {
                std::this_thread::sleep_for(delay);
            }
        };
    }

    bool wait_for_start(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, timeout, [&]() { return started_signal; });
    }

    size_t count_of(int layer, int expert) {
        std::lock_guard<std::mutex> lock(mutex);
        size_t n = 0;
        for (const auto & r : reads) {
            n += (r.first == layer && r.second == expert) ? 1 : 0;
        }
        return n;
    }
};

void test_demand_dispatch_waits_for_inflight_spec_batch() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    require(setenv("WP_EXPERT_SPEC_PAGEIN", "1", 1) == 0, "failed to arm speculative page-in");
    require(setenv("WP_EXPERT_SPEC_CHUNK", "1", 1) == 0,
            "failed to pin the spec chunk to one page per submit");
    // WP_EXPERT_SPEC_MAX_INFLIGHT deliberately left UNSET: this test's whole
    // point is that the DEFAULT (cap=1) path is safe against a real in-flight
    // batch, which is the premise the s0 failure put in question.
    unsetenv("WP_EXPERT_SPEC_MAX_INFLIGHT");

    DelayedReadLog reads;
    reads.delay_layer  = LAYER;
    reads.delay_expert = 1;
    reads.delay        = std::chrono::milliseconds(600);

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    options.slots             = 4;   // floor for this fixture -- see the note above
    options.host_budget_bytes = 2 * PAGE_BYTES;
    options.once              = true;
    options.test_hooks        = &reads.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK, "worker did not acknowledge HELLO");

        // 1. Hint expert 1 on LAYER. The idle pump submits it as a one-page
        //    speculative batch; the hook stalls that read for 600 ms once it
        //    starts, so the batch is GUARANTEED still in flight when step 3's
        //    demand dispatch for the same page lands.
        pipe_expert_prefetch_hint frame;
        frame.layer      = LAYER;
        frame.provenance = PIPE_HINT_CERTAIN;
        frame.expert_ids = { 1 };
        payload = pipe_encode_expert_prefetch_hint(frame);
        require(pipe_send_frame(*socket, PIPE_EXPERT_PREFETCH_HINT, 0, payload.data(), payload.size()),
                "failed to send prefetch hint");

        // 2. Confirm the speculative read genuinely started (staging
        //    borrowed, read_started fired) before demanding the same page --
        //    otherwise this test would not be exercising the interlock at all.
        require(reads.wait_for_start(std::chrono::milliseconds(2000)),
                "the speculative read for (LAYER, 1) never started");

        // 3. THE REGRESSION CHECK. Demand the SAME page while its speculative
        //    read is still stalled inside the hook. ensure_batch's interlock
        //    must detect spec_in_flight_for(page)==true and block via
        //    spec_pagein_poll(false, page) until that read lands, then reuse
        //    it as a hit -- not hang, not throw (which would unwind
        //    serve_connection and close the socket, exactly the s0 symptom:
        //    the client's recv/send would fail with no exception logged), and
        //    not issue a second read of the same page.
        pipe_expert_dispatch_req request;
        request.layer       = LAYER;
        request.n_tokens    = N_TOKENS;
        request.activations.resize((size_t) N_TOKENS * N_EMBD);
        request.assignments = { { 1, std::vector<float>(N_TOKENS, 0.5f) } };
        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 100, payload.data(), payload.size()),
                "failed to send the demand dispatch that overlaps the in-flight spec batch");

        // Generous: must clear the 600 ms stall plus the read/H2D/compute
        // itself. If ensure_batch's interlock deadlocked or the connection
        // was closed out from under us, this recv will fail or time out --
        // exactly the symptom under investigation, so failing loudly here
        // (rather than hanging the test suite forever) is deliberate.
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "worker closed the connection instead of answering the demand "
                "dispatch that overlapped an in-flight speculative batch -- "
                "this is the s0 failure mode");
        if (type == PIPE_ERROR) {
            const pipe_error error = pipe_decode_error(payload.data(), payload.size());
            throw std::runtime_error(
                "demand dispatch overlapping an in-flight spec batch failed: " + error.msg);
        }
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 100,
                "demand dispatch overlapping an in-flight spec batch did not complete");

        // 4. And it must have been ONE read, not two: the interlock exists
        //    precisely so the demand path reuses the speculative read instead
        //    of racing a second one against it.
        require(reads.count_of(LAYER, 1) == 1,
                "a page already being read speculatively was read a second "
                "time by the overlapping demand dispatch");
        socket.reset();
    } catch (...) {
        server.join();
        unsetenv("WP_EXPERT_SPEC_PAGEIN");
        unsetenv("WP_EXPERT_SPEC_CHUNK");
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        throw;
    }
    server.join();
    require(unsetenv("WP_EXPERT_SPEC_PAGEIN") == 0, "failed to disarm speculative page-in");
    require(unsetenv("WP_EXPERT_SPEC_CHUNK") == 0, "failed to clear the spec chunk override");
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0,
            "worker returned failure after a demand dispatch overlapped an "
            "in-flight speculative batch");
}


// REGRESSION, PREFILL-AHEAD VARIANT: WP_PREFILL_LAYER_AHEAD submits the
// NEXT layer's ENTIRE page set as one multi-page speculative batch directly
// from dispatch()/begin_split_dispatch(), bypassing the WP_EXPERT_SPEC_CHUNK-
// bounded router queue the test above exercises. This is the shape closest to
// the s0 production failure: 9 MiB pages, WP_PREFILL_LAYER_AHEAD in play, and
// a batch big enough that some OTHER layer's demand dispatch is likely to
// still find it in flight. The single-page test above never engages this
// path at all (its requests are decode-shaped, well under
// WP_PREFILL_LAYER_AHEAD_WIDTH), so it cannot stand in for this one.
void test_demand_dispatch_waits_for_inflight_prefill_ahead_batch() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const int port = reserve_port();

    require(setenv("WP_EXPERT_SPEC_PAGEIN", "1", 1) == 0, "failed to arm speculative page-in");
    require(setenv("WP_PREFILL_LAYER_AHEAD", "1", 1) == 0,
            "failed to arm WP_PREFILL_LAYER_AHEAD");
    require(setenv("WP_PREFILL_LAYER_AHEAD_WIDTH", "1", 1) == 0,
            "failed to lower the prefill-ahead width so this test's requests qualify");
    // WP_EXPERT_SPEC_MAX_INFLIGHT deliberately left UNSET -- default cap=1,
    // the exact configuration the s0 failure report says is broken.
    unsetenv("WP_EXPERT_SPEC_MAX_INFLIGHT");

    DelayedReadLog reads;
    reads.delay_layer  = OTHER_LAYER;
    reads.delay_expert = 1;
    reads.delay        = std::chrono::milliseconds(600);

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = port;
    // Enough slots to hold LAYER's own demand pages AND every page of
    // OTHER_LAYER's full-catalog ahead-submit pinned at once.
    options.slots             = 8;
    options.host_budget_bytes = 4 * PAGE_BYTES;
    options.once              = true;
    options.test_hooks        = &reads.hooks;

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
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK, "worker did not acknowledge HELLO");

        const uint32_t prefill_tokens = 4;   // > WP_PREFILL_LAYER_AHEAD_WIDTH=1
        const auto dispatch = [&](int32_t layer, std::vector<int32_t> experts,
                                  uint64_t seq) {
            pipe_expert_dispatch_req request;
            request.layer    = layer;
            request.n_tokens = prefill_tokens;
            request.activations.resize((size_t) prefill_tokens * N_EMBD);
            for (int32_t e : experts) {
                request.assignments.push_back(
                    { e, std::vector<float>(prefill_tokens, 0.5f) });
            }
            payload = pipe_encode_expert_dispatch_req(request);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, seq,
                                    payload.data(), payload.size()),
                    "failed to send dispatch");
            require(pipe_recv_frame(*socket, type, seq_id, payload),
                    "worker closed the connection instead of answering -- "
                    "this is the s0 failure mode");
            if (type == PIPE_ERROR) {
                const pipe_error error = pipe_decode_error(payload.data(), payload.size());
                throw std::runtime_error("dispatch failed: " + error.msg);
            }
            require(type == PIPE_EXPERT_PARTIAL && seq_id == seq,
                    "dispatch did not complete");
        };

        // 1. Prefill-shaped dispatch on LAYER. dispatch()'s own ensure_batch
        //    pins LAYER's page, then submit_prefill_layer_ahead(LAYER, 4)
        //    fires and submits ALL of OTHER_LAYER's pages (experts 0..3) as
        //    ONE speculative batch, bypassing WP_EXPERT_SPEC_CHUNK entirely.
        //    The hook stalls (OTHER_LAYER, 1)'s read for 600 ms.
        dispatch(LAYER, { 0 }, 100);

        // 2. Confirm the ahead-submit really landed a read in flight for the
        //    stalled page before demanding it -- otherwise this test would
        //    not be exercising the interlock at all.
        require(reads.wait_for_start(std::chrono::milliseconds(2000)),
                "the prefill-ahead speculative read for (OTHER_LAYER, 1) "
                "never started");

        // 3. THE REGRESSION CHECK. Demand OTHER_LAYER's expert 1 while its
        //    page is still being read by the in-flight ahead-submit batch.
        //    ensure_batch's interlock must block via
        //    spec_pagein_poll(false, page) until the WHOLE 4-page batch
        //    drains, then reuse the landed page -- not hang, not throw (an
        //    escaped exception here unwinds serve_connection and closes the
        //    socket exactly like the s0 symptom), and not read it twice.
        dispatch(OTHER_LAYER, { 1 }, 101);

        require(reads.count_of(OTHER_LAYER, 1) == 1,
                "a page already being read by the prefill-ahead batch was "
                "read a second time by the overlapping demand dispatch");
        socket.reset();
    } catch (...) {
        server.join();
        unsetenv("WP_EXPERT_SPEC_PAGEIN");
        unsetenv("WP_PREFILL_LAYER_AHEAD");
        unsetenv("WP_PREFILL_LAYER_AHEAD_WIDTH");
        if (server_error) {
            std::rethrow_exception(server_error);
        }
        throw;
    }
    server.join();
    require(unsetenv("WP_EXPERT_SPEC_PAGEIN") == 0, "failed to disarm speculative page-in");
    require(unsetenv("WP_PREFILL_LAYER_AHEAD") == 0,
            "failed to clear WP_PREFILL_LAYER_AHEAD");
    require(unsetenv("WP_PREFILL_LAYER_AHEAD_WIDTH") == 0,
            "failed to clear WP_PREFILL_LAYER_AHEAD_WIDTH");
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0,
            "worker returned failure after a demand dispatch overlapped an "
            "in-flight prefill-ahead speculative batch");
}

} // namespace

static void test_scatter_compact_rows_matches_get_rows_back() {
    // Unique idx: set_rows into zeros must match get_rows_back byte-for-byte
    // on CPU, including an all-zero compact row written at dest 0 (the empty-
    // expert placeholder).
    const int n_embd = 8;
    const int n_tokens = 16;
    const int n_sel = 3;
    const int32_t idx_h[3] = {2, 0, 11};

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "failed to create scatter ggml context");

    ggml_tensor * full = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * compact = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_sel);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_sel);
    require(full->data && compact->data && idx->data, "scatter tensors must be allocated");

    std::vector<float> compact_h((size_t) n_embd * n_sel);
    for (size_t i = 0; i < compact_h.size(); ++i) {
        compact_h[i] = (i < (size_t) n_embd) ? 0.0f : 0.25f * (float) (i + 1);
    }
    std::vector<float> full_h((size_t) n_embd * n_tokens, 3.0f);
    std::memcpy(compact->data, compact_h.data(), compact_h.size() * sizeof(float));
    std::memcpy(idx->data, idx_h, sizeof(idx_h));
    std::memcpy(full->data, full_h.data(), full_h.size() * sizeof(float));

    ggml_tensor * via_back = ggml_get_rows_back(ctx.get(), compact, idx, full);
    ggml_tensor * via_set  = wp_expert_worker::scatter_compact_rows(ctx.get(), compact, idx, full);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 64, false);
    ggml_build_forward_expand(gf, via_back);
    ggml_build_forward_expand(gf, via_set);
    require(ggml_graph_compute_with_ctx(ctx.get(), gf, 1) == GGML_STATUS_SUCCESS,
            "scatter equivalence graph failed");

    const int n = n_embd * n_tokens;
    for (int i = 0; i < n; ++i) {
        const float a = ggml_get_f32_1d(via_back, i);
        const float b = ggml_get_f32_1d(via_set, i);
        if (a != b) {
            throw std::runtime_error(
                "scatter_compact_rows != get_rows_back at i=" + std::to_string(i) +
                " back=" + std::to_string(a) + " set=" + std::to_string(b));
        }
    }
    // Untouched dest rows stay 0 (not the 3.0 filler in `full`).
    require(ggml_get_f32_1d(via_set, 1 * n_embd) == 0.0f,
            "row 1 is not in idx and must stay zero");
    // idx[0]=2 maps compact row 0 (all zeros) onto dest row 2.
    require(ggml_get_f32_1d(via_set, 2 * n_embd) == 0.0f,
            "dest row 2 must receive the all-zero compact row");
    // idx[1]=0 maps compact row 1 (nonzero) onto dest row 0.
    require(ggml_get_f32_1d(via_set, 0) != 0.0f,
            "dest row 0 must receive a nonzero compact row");
}

static void test_scatter_add_compact_rows_accumulates() {
    // Production path: dest is the io result (already allocated). Two experts
    // can share a token; set_rows overwrites, so we RMW-add. Dest rows not in
    // idx stay put — no full-ubatch zero tensor.
    const int n_embd = 4;
    const int n_tokens = 8;
    const int n_sel = 2;
    const int32_t idx_h[2] = {1, 4};

    ggml_init_params params = {
        /*.mem_size   =*/ 2 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "failed to create scatter-add ggml context");

    ggml_tensor * dest = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_tokens);
    ggml_tensor * compact = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_embd, n_sel);
    ggml_tensor * idx = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_sel);
    require(dest->data && compact->data && idx->data, "scatter-add tensors must be allocated");

    std::vector<float> dest_h((size_t) n_embd * n_tokens, 1.0f);
    std::vector<float> compact_h((size_t) n_embd * n_sel, 10.0f);
    std::memcpy(dest->data, dest_h.data(), dest_h.size() * sizeof(float));
    std::memcpy(compact->data, compact_h.data(), compact_h.size() * sizeof(float));
    std::memcpy(idx->data, idx_h, sizeof(idx_h));

    ggml_tensor * out = wp_expert_worker::scatter_add_compact_rows(ctx.get(), dest, compact, idx);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(gf, out);
    require(ggml_graph_compute_with_ctx(ctx.get(), gf, 1) == GGML_STATUS_SUCCESS,
            "scatter-add graph failed");

    require(ggml_get_f32_1d(out, 1 * n_embd) == 11.0f, "idx row 1 must be 1+10");
    require(ggml_get_f32_1d(out, 4 * n_embd) == 11.0f, "idx row 4 must be 1+10");
    require(ggml_get_f32_1d(out, 0) == 1.0f, "untouched row 0 stays 1");
    require(ggml_get_f32_1d(out, 2 * n_embd) == 1.0f, "untouched row 2 stays 1");
}

static void test_partial_last_column_round_trip() {
    constexpr int n_embd = 2560;
    constexpr int n_ff_exp = 640;
    constexpr int n_tokens = 2;

    std::vector<float> hidden((size_t) n_tokens * n_ff_exp);
    std::vector<float> down((size_t) n_embd * n_ff_exp);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = ((int) (i % 17) - 8) * 0.013f;
    }
    for (size_t i = 0; i < down.size(); ++i) {
        down[i] = ((int) (i % 23) - 11) * 0.001f;
    }

    // CPU replay of a [640 -> 2560] expert down projection. Keep the
    // accumulation order used by the worker's scalar reference path.
    std::vector<float> cpu((size_t) n_tokens * n_embd, 0.0f);
    for (int token = 0; token < n_tokens; ++token) {
        for (int output = 0; output < n_embd; ++output) {
            float value = 0.0f;
            for (int input = 0; input < n_ff_exp; ++input) {
                value += down[(size_t) output * n_ff_exp + input] *
                         hidden[(size_t) token * n_ff_exp + input];
            }
            cpu[(size_t) token * n_embd + output] = value;
        }
    }

    pipe_expert_partial source;
    source.layer    = 4;
    source.n_tokens = n_tokens;
    source.dtype    = PIPE_HIDDEN_F32;
    source.partial  = cpu;
    const std::vector<uint8_t> payload = pipe_encode_expert_partial(source);
    const pipe_expert_partial decoded =
        pipe_decode_expert_partial(payload.data(), payload.size(), n_embd);

    for (int token = 0; token < n_tokens; ++token) {
        const float * expected = cpu.data() + (size_t) token * n_embd;
        const float * actual = decoded.partial.data() + (size_t) token * n_embd;
        require(std::memcmp(expected, actual, (size_t) n_embd * sizeof(float)) == 0,
                "expert partial changed during f32 encode/decode");
        require(std::memcmp(&expected[n_embd - 1], &actual[n_embd - 1], sizeof(float)) == 0,
                "expert partial last column changed during f32 encode/decode");
    }
}

static void test_decode_prefill_compute_profile() {
    // Unset / empty / missing → new defaults (min tokens 2, coalesce on, cache on).
    require(wp_expert_worker::parse_gather_min_tokens(nullptr) == 2,
            "default WP_EXPERT_GATHER_MIN_TOKENS must be 2 so decode skips gather");
    require(wp_expert_worker::parse_gather_min_tokens("") == 2,
            "empty WP_EXPERT_GATHER_MIN_TOKENS must be 2");
    require(wp_expert_worker::parse_gather_min_tokens("1") == 1,
            "WP_EXPERT_GATHER_MIN_TOKENS=1 must still be honoured for A/B");
    require(wp_expert_worker::parse_gather_min_tokens("8") == 8,
            "WP_EXPERT_GATHER_MIN_TOKENS=8 must be honoured");
    require(wp_expert_worker::parse_gather_min_tokens("0") == 1,
            "non-positive gather min tokens must clamp to 1");

    require(wp_expert_worker::parse_env_default_on(nullptr),
            "WP_EXPERT_PARAMS_COALESCE / GRAPH_CACHE default ON when unset");
    require(wp_expert_worker::parse_env_default_on(""),
            "empty value must keep the default-ON knobs on");
    require(wp_expert_worker::parse_env_default_on("1"),
            "explicit 1 must enable a default-ON knob");
    require(!wp_expert_worker::parse_env_default_on("0"),
            "explicit 0 must disable a default-ON knob");

    require(wp_expert_worker::use_expert_gather(1, false, 2, true) == false,
            "decode (n_tokens==1) must not gather at the default min of 2");
    require(wp_expert_worker::use_expert_gather(2, false, 2, true) == true,
            "verify/prefill at n_tokens==2 must still gather");
    require(wp_expert_worker::use_expert_gather(2048, false, 2, true) == true,
            "prefill must still gather");
    require(wp_expert_worker::use_expert_gather(1, false, 1, true) == true,
            "min_tokens=1 is the always-gather A/B");
    require(wp_expert_worker::use_expert_gather(64, true, 2, true) == false,
            "force_dense (WP_SELFCHECK) must disable gather");
    require(wp_expert_worker::use_expert_gather(64, false, 2, false) == false,
            "WP_EXPERT_GATHER=0 must disable gather");

    const auto empty = wp_expert_worker::compact_routing_rows({ 0.0f, 0.0f, 0.0f });
    require(empty.idx.size() == 1 && empty.idx[0] == 0 && empty.weights[0] == 0.0f,
            "all-zero routing must keep a dummy idx 0 / weight 0");
    const auto mixed = wp_expert_worker::compact_routing_rows({ 0.0f, 0.5f, 0.0f, 1.25f });
    require(mixed.idx.size() == 2 && mixed.idx[0] == 1 && mixed.idx[1] == 3,
            "compact idx must be the nonzero token positions");
    require(mixed.weights.size() == 2 && mixed.weights[0] == 0.5f && mixed.weights[1] == 1.25f,
            "compact weights must follow idx");
}

// Records every ExpertSlotPool::stripe_plan() call so a test can see how
// many stripes a given (page_size, n_pageins) actually produced. This is
// independent of the read_started/read_finished hooks, which fire once per
// PAGE regardless of stripe count and so cannot show whether striping
// actually engaged.
struct StripePlanLog {
    struct Entry {
        uint64_t page_size;
        size_t   n_pageins;
        size_t   n_stripes;
    };

    wp_expert_worker::TestHooks hooks;

    StripePlanLog() {
        hooks.stripe_planned =
            [this](uint64_t page_size, size_t n_pageins, size_t n_stripes) {
                std::lock_guard<std::mutex> lock(mutex);
                entries.push_back({ page_size, n_pageins, n_stripes });
            };
    }

    std::vector<Entry> snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return entries;
    }

private:
    std::mutex          mutex;
    std::vector<Entry>  entries;
};

// Runs one cold, single-expert dispatch (exactly one miss, one page-in of
// PAGE_BYTES -- the "lone small slice read" the sliced DECODE path sees) and
// returns the worker's partial plus every stripe_plan() call the miss made.
std::pair<std::vector<float>, std::vector<StripePlanLog::Entry>>
run_single_miss_stripe_case(const Fixture & fixture, const std::vector<float> & input) {
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
    StripePlanLog log;
    options.test_hooks = &log.hooks;

    int server_result = -1;
    std::exception_ptr server_error;
    std::thread server([&]() {
        try {
            server_result = wp_expert_worker::run(options);
        } catch (...) {
            server_error = std::current_exception();
        }
    });

    std::vector<float> partial;
    try {
        pipe_socket_ptr socket = connect_with_retry(port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "stripe-case worker did not send HELLO");
        pipe_expert_hello worker_hello =
            pipe_decode_expert_hello(payload.data(), payload.size());
        pipe_expert_hello client_hello = worker_hello;
        client_hello.role         = PIPE_EXPERT_ROLE_CLIENT;
        client_hello.expert_first = -1;
        client_hello.expert_last  = -1;
        client_hello.n_slots      = 0;
        client_hello.layers.clear();
        payload = pipe_encode_expert_hello(client_hello);
        require(pipe_send_frame(
                    *socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send stripe-case client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive stripe-case HELLO ack");
        require(type == PIPE_EXPERT_HELLO_ACK &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "stripe-case worker rejected matching HELLO");

        pipe_expert_dispatch_req request;
        request.layer    = LAYER;
        request.n_tokens = N_TOKENS;
        request.activations = input;
        // One assignment == one miss == one page-in: the lone-slice-read
        // shape this fix targets, not a dense prefill-shaped batch.
        request.assignments = { { 0, { 1.0f, 0.5f } } };
        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(
                    *socket, PIPE_EXPERT_DISPATCH_REQ, 60,
                    payload.data(), payload.size()),
                "failed to send stripe-case dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive stripe-case partial");
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 60,
                "stripe-case dispatch did not complete");
        const pipe_expert_partial response =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        partial = response.partial;
        socket.reset();
    } catch (...) {
        server.join();
        throw;
    }
    server.join();
    if (server_error) {
        std::rethrow_exception(server_error);
    }
    require(server_result == 0, "stripe-case worker returned failure");
    return { partial, log.snapshot() };
}

// THE BUG THIS PROVES: stripe_plan's split-below-this-many-bytes floor used
// to be a hardcoded 1 MiB (`kMinPart`), sized for the pre-sliced rig's
// ~12.75 MiB whole-expert page. On the sliced rig's much smaller page (this
// fixture's PAGE_BYTES stands in for it -- three roles, O_DIRECT aligned,
// exactly the shape a real slice page has), that floor silently forced
// n = total/kMinPart = 0, i.e. NO STRIPING AT ALL, for exactly the lone
// small-page decode miss the read/H2D pipeline exists to overlap.
// WP_EXPERT_STRIPE_MIN_PART reproduces both arms directly: a large value
// (1 MiB, the old fixed floor) collapses the miss to one whole-page stripe;
// a small value lets WP_EXPERT_READ_STRIPES actually engage. THE FIX MUST
// NOT CHANGE THE ANSWER: the two arms' partial results must be bit-for-bit
// identical, because striping only changes how the SAME page bytes are
// grouped into pread()/tensor_set() calls, never what ends up in the slot.
void test_stripe_min_part_restores_overlap_byte_identical() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);

    std::vector<float> input((size_t) N_TOKENS * N_EMBD);
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = ((int) (i % 13) - 6) * 0.07f;
    }

    std::vector<float> no_stripe_partial;
    std::vector<float> striped_partial;
    std::vector<StripePlanLog::Entry> no_stripe_log;
    std::vector<StripePlanLog::Entry> striped_log;
    {
        const ScopedEnv stripes("WP_EXPERT_READ_STRIPES", "2");
        {
            // Old fixed 1 MiB floor, reproduced explicitly: total/n (6144 B)
            // is far below it, so stripe_plan must fall back to n<=1.
            const ScopedEnv min_part("WP_EXPERT_STRIPE_MIN_PART", "1048576");
            auto result = run_single_miss_stripe_case(fixture, input);
            no_stripe_partial = std::move(result.first);
            no_stripe_log     = std::move(result.second);
        }
        {
            // Sliced-rig-appropriate floor: small enough that a ~12 KiB
            // synthetic page (standing in for a real ~1.5-9 MiB slice page)
            // still gets split.
            const ScopedEnv min_part("WP_EXPERT_STRIPE_MIN_PART", "2048");
            auto result = run_single_miss_stripe_case(fixture, input);
            striped_partial = std::move(result.first);
            striped_log     = std::move(result.second);
        }
    }

    require(!no_stripe_log.empty() && !striped_log.empty(),
            "stripe_plan hook did not fire for either arm");
    const auto miss_entry = [&](const std::vector<StripePlanLog::Entry> & log) {
        for (const auto & entry : log) {
            if (entry.page_size == PAGE_BYTES && entry.n_pageins == 1) {
                return entry;
            }
        }
        throw std::runtime_error("no stripe_plan() call matched the lone miss");
    };
    const StripePlanLog::Entry no_stripe = miss_entry(no_stripe_log);
    const StripePlanLog::Entry striped   = miss_entry(striped_log);

    require(no_stripe.n_stripes == 1,
            "old 1 MiB floor should still collapse the sliced page to one "
            "whole-page read -- this is the bug being fixed");
    require(striped.n_stripes > 1,
            "WP_EXPERT_STRIPE_MIN_PART did not restore striping for a "
            "small sliced-rig page-sized read");

    require(no_stripe_partial.size() == striped_partial.size(),
            "striped and non-striped responses have different shapes");
    for (size_t i = 0; i < no_stripe_partial.size(); ++i) {
        require(no_stripe_partial[i] == striped_partial[i],
                "striped read/H2D pipeline changed the computed result -- "
                "page contents must be byte-identical regardless of "
                "stripe scheduling");
    }
}

int main() {
    try {
        test_decode_prefill_compute_profile();
        test_scatter_compact_rows_matches_get_rows_back();
        test_scatter_add_compact_rows_accumulates();
        test_partial_last_column_round_trip();
        test_slice_device_member_layout();
        test_glm_size_class_plan();
        run_test();
        test_default_off_multi_expert_request();
        test_prefetch_hint_without_spec_reads_nothing();
        test_spec_pagein_logs_s_not_d();
        test_predicted_hint_lands_in_host_ram();
        test_prefetch_spec_pagein_and_eviction_order("0");
        test_prefetch_spec_pagein_and_eviction_order("64");
        test_spec_max_inflight(
            /*env_value=*/ nullptr, /*hinted_experts=*/ 2, /*barrier_target=*/ 2,
            /*expected_peak=*/ 1,
            "WP_EXPERT_SPEC_MAX_INFLIGHT default did not serialize speculative "
            "batches to one at a time");
        test_spec_max_inflight(
            /*env_value=*/ "3", /*hinted_experts=*/ 3, /*barrier_target=*/ 3,
            /*expected_peak=*/ 3,
            "WP_EXPERT_SPEC_MAX_INFLIGHT=3 did not allow three speculative "
            "batches to read concurrently");
        test_demand_dispatch_waits_for_inflight_spec_batch();
        test_demand_dispatch_waits_for_inflight_prefill_ahead_batch();
        test_stripe_min_part_restores_overlap_byte_identical();
        std::cout << "test-wp-expert-worker: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-expert-worker: " << error.what() << '\n';
        return 1;
    }
}
