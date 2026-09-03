#include "ggml.h"
#include "ggml-backend.h"
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
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
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
    for (int attempt = 0; attempt < 6000; ++attempt) { // 30 s: a ROCm worker needs ~1 s just to load libggml-hip (2026-09-02, R9700), and a miss deadlocks the test in server.join()
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
            { { LAYER, slot_bytes, false, 3 * up_gate_bytes, {} } }, 1,
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
            pages.push_back({ layer, size, false, 0, {} });
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

void test_fixture_arena_stride_alignment() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.slots             = 4;
    options.host_budget_bytes = 2 * PAGE_BYTES;

    const wp_expert_worker::ResourcePlan resources =
        wp_expert_worker::inspect_resources(options);
    const uint64_t backend_alignment =
        ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    require(backend_alignment != 0, "CPU backend returned zero arena alignment");
    require(!resources.slot_classes.empty(), "fixture worker has no slot classes");
    uint64_t arena_bytes = 0;
    for (const wp_expert_worker::SlotClass & slot_class : resources.slot_classes) {
        require(slot_class.stride >= slot_class.size,
                "arena stride is smaller than its slot class");
        require(slot_class.stride % backend_alignment == 0,
                "arena stride is not a multiple of the backend alignment");
        for (const ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32 }) {
            require(slot_class.stride % ggml_type_size(type) == 0,
                    "arena stride is not a multiple of a role type size");
        }
        arena_bytes += slot_class.stride * (uint64_t) slot_class.slots;
    }
    // device_bytes is the TRUE footprint: usable slots plus each arena's
    // reserved PAD tail (n_expert_used slots per arena, bought as extra
    // bytes when a class cannot spare them). The planned slots must fit the
    // budget; the excess must be no more than the pad tails could account for.
    uint64_t pad_bytes_max = 0;
    for (const wp_expert_worker::SlotClass & slot_class : resources.slot_classes) {
        pad_bytes_max += slot_class.stride * (uint64_t) slot_class.pad_slots *
            wp_expert_worker::test_pool_arena_count();
    }
    require(arena_bytes <= resources.device_bytes &&
                resources.device_bytes - arena_bytes <= pad_bytes_max &&
                arena_bytes <= resources.slot_budget_bytes,
            "arena slot classes do not fit their resource budget");

    const uint64_t q5_k_size = ggml_type_size(GGML_TYPE_Q5_K);
    const uint64_t q8_0_size = ggml_type_size(GGML_TYPE_Q8_0);
    const uint64_t type_alignment =
        q5_k_size / std::gcd(q5_k_size, q8_0_size) * q8_0_size;
    const uint64_t expected_stride =
        (3000 + type_alignment - 1) / type_alignment * type_alignment;
    const wp_expert_worker::ResourcePlan quantized =
        wp_expert_worker::plan_resources({
            { LAYER, 3000, false, 0, { q5_k_size } },
            { OTHER_LAYER, 3000, false, 0, { q8_0_size } },
            { OTHER_LAYER + 1, 3000, false, 0, { q5_k_size } },
            { OTHER_LAYER + 2, 3000, false, 0, { q8_0_size } },
        }, 4);
    require(quantized.slot_classes.size() == 1,
            "mixed-role fixture did not produce one slot class");
    require(quantized.slot_classes[0].stride == expected_stride &&
                quantized.slot_classes[0].stride % q5_k_size == 0 &&
                quantized.slot_classes[0].stride % q8_0_size == 0,
            "arena stride did not combine role type sizes across pages");
    require(quantized.slot_count == 2 &&
                quantized.device_bytes <= quantized.slot_budget_bytes,
            "arena stride did not reduce slots to fit the resource budget");
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

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const char * name = ggml_backend_dev_name(device);
        if (name != nullptr && (std::strstr(name, "CUDA") != nullptr ||
                                std::strstr(name, "ROCm") != nullptr ||
                                std::strstr(name, "HIP") != nullptr)) {
            backend = ggml_backend_dev_init(device, nullptr);
            break;
        }
    }
    if (backend == nullptr) {
        std::cout << "test_partial_last_column_round_trip: no CUDA/HIP backend; skipping device case\n";
        return;
    }

    const std::vector<ggml_type> types = {
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
        GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S,
    };

    for (const ggml_type type : types) {
        const int block_size = ggml_blck_size(type);
        const int n_k = (n_ff_exp + block_size - 1) / block_size * block_size;
        std::vector<float> hidden_padded((size_t) n_tokens * n_k, 0.0f);
        std::vector<float> down_padded((size_t) n_embd * n_k, 0.0f);
        for (int token = 0; token < n_tokens; ++token) {
            std::memcpy(hidden_padded.data() + (size_t) token * n_k,
                        hidden.data() + (size_t) token * n_ff_exp,
                        (size_t) n_ff_exp * sizeof(float));
        }
        for (int output_idx = 0; output_idx < n_embd; ++output_idx) {
            std::memcpy(down_padded.data() + (size_t) output_idx * n_k,
                        down.data() + (size_t) output_idx * n_ff_exp,
                        (size_t) n_ff_exp * sizeof(float));
        }

        ggml_init_params params = {
            /*.mem_size   =*/ 16 * 1024 * 1024,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(params);
        require(ctx != nullptr, "failed to create CUDA/HIP regression context");

        ggml_tensor * weights = ggml_new_tensor_2d(ctx, type, n_k, n_embd);
        ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_k, n_tokens);
        ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);
        ggml_cgraph * graph   = ggml_new_graph_custom(ctx, 4, false);
        ggml_build_forward_expand(graph, output);
        require(ggml_backend_supports_op(backend, output),
                "CUDA/HIP backend does not support quantized regression op");

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "failed to allocate CUDA/HIP regression tensors");

        const size_t row_size = ggml_row_size(type, n_k);
        std::vector<uint8_t> quantized(row_size * n_embd);
        std::vector<float> imatrix(n_k, 1.0f);
        const float * im = ggml_quantize_requires_imatrix(type) ? imatrix.data() : nullptr;
        require(ggml_quantize_chunk(type, down_padded.data(), quantized.data(),
                                    0, n_embd, n_k, im) == quantized.size(),
                "failed to quantize CUDA/HIP regression weights");
        ggml_backend_tensor_set(weights, quantized.data(), 0, quantized.size());
        ggml_backend_tensor_set(input, hidden_padded.data(), 0, hidden_padded.size() * sizeof(float));

        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
                "CUDA/HIP [640 -> 2560] regression graph failed");
        std::vector<float> device_output((size_t) n_tokens * n_embd);
        ggml_backend_tensor_get(output, device_output.data(), 0, device_output.size() * sizeof(float));
        for (int token = 0; token < n_tokens; ++token) {
            for (int output_idx = 0; output_idx < n_embd; ++output_idx) {
                require(std::isfinite(device_output[(size_t) token * n_embd + output_idx]),
                        "CUDA/HIP [640 -> 2560] regression produced a non-finite value");
            }
            require(std::isfinite(device_output[(size_t) token * n_embd + n_embd - 1]),
                    "CUDA/HIP [640 -> 2560] regression last column is non-finite");
        }

        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    }
    ggml_backend_free(backend);
}

// Prefill-shaped Q5_1 down-proj on CUDA/HIP. Chat NaNs at n_tokens≈42 (MMQ)
// with first non-finite at dim 2559. The round-trip above only covers
// n_tokens=2 (MMVQ). This dirties MATRIX_ROW_PADDING then runs n_tokens=32
// on K=640 N=2560 Q5_1. MMQ must clear that pad (not only USAGE_COMPUTE)
// or the last output column is NaN.
static void test_q5_1_down_proj_prefill_last_column() {
    constexpr int n_embd = 2560;
    constexpr int n_ff_exp = 640;
    constexpr int n_tokens = 32;

    ggml_backend_load_all();
    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }
        const char * name = ggml_backend_dev_name(device);
        if (name != nullptr && (std::strstr(name, "CUDA") != nullptr ||
                                std::strstr(name, "ROCm") != nullptr ||
                                std::strstr(name, "HIP") != nullptr)) {
            backend = ggml_backend_dev_init(device, nullptr);
            break;
        }
    }
    if (backend == nullptr) {
        std::cout << "test_q5_1_down_proj_prefill_last_column: no CUDA/HIP backend; skipping\n";
        return;
    }

    std::vector<float> hidden((size_t) n_tokens * n_ff_exp);
    std::vector<float> down((size_t) n_embd * n_ff_exp);
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = ((int) (i % 17) - 8) * 0.013f;
    }
    for (size_t i = 0; i < down.size(); ++i) {
        down[i] = ((int) (i % 23) - 11) * 0.001f;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 16 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "failed to create Q5_1 prefill context");

    ggml_tensor * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q5_1, n_ff_exp, n_embd);
    ggml_tensor * input   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff_exp, n_tokens);
    ggml_tensor * output  = ggml_mul_mat(ctx, weights, input);
    ggml_cgraph * graph   = ggml_new_graph_custom(ctx, 4, false);
    ggml_build_forward_expand(graph, output);

    const size_t weight_bytes = ggml_nbytes(weights);
    const size_t weight_alloc = ggml_backend_buft_get_alloc_size(
        ggml_backend_get_default_buffer_type(backend), weights);
    const size_t input_bytes  = ggml_nbytes(input);
    const size_t output_bytes = ggml_nbytes(output);
    require(weight_alloc >= weight_bytes, "Q5_1 alloc size smaller than nbytes");

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "failed to allocate Q5_1 prefill tensors");

    std::vector<uint8_t> quantized(weight_bytes);
    require(ggml_quantize_chunk(GGML_TYPE_Q5_1, down.data(), quantized.data(),
                                0, n_embd, n_ff_exp, nullptr) == quantized.size(),
            "failed to quantize Q5_1 down weights");

    // Dirty every byte in the backend buffer (including MATRIX_ROW_PADDING),
    // then write only ggml_nbytes of weights. Matches the worker skipping
    // init_tensor: the padded tail is not part of the shard copy.
    ggml_backend_buffer_clear(buffer, 0xFF);
    ggml_backend_tensor_set(weights, quantized.data(), 0, weight_bytes);
    ggml_backend_tensor_set(input, hidden.data(), 0, input_bytes);
    // Worker zeros the pad via an I8 tensor covering the whole slot (weight
    // tensors cannot tensor_memset past ggml_nbytes). Mirror that here.
    if (weight_alloc > weight_bytes) {
        ggml_tensor * raw = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t) weight_alloc);
        raw->buffer = weights->buffer;
        raw->data   = weights->data;
        ggml_backend_tensor_memset(raw, 0, weight_bytes, weight_alloc - weight_bytes);
    }

    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "Q5_1 [640x2560] n_tokens=32 graph failed");
    std::vector<float> device_output(output_bytes / sizeof(float));
    ggml_backend_tensor_get(output, device_output.data(), 0, output_bytes);

    int n_bad = 0;
    int first_tok = -1;
    int first_dim = -1;
    for (int token = 0; token < n_tokens; ++token) {
        for (int dim = 0; dim < n_embd; ++dim) {
            if (!std::isfinite(device_output[(size_t) token * n_embd + dim])) {
                if (n_bad == 0) {
                    first_tok = token;
                    first_dim = dim;
                }
                n_bad++;
            }
        }
    }
    if (n_bad != 0) {
        throw std::runtime_error(
            "Q5_1 [640x2560] n_tokens=32 dirty-pad produced " +
            std::to_string(n_bad) + " non-finite values; first token " +
            std::to_string(first_tok) + " dim " + std::to_string(first_dim));
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(backend);
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

    // WP_EXPERT_MM_PIN is three-way: off / on / "decode".
    using wp_expert_worker::mm_pin_mode;
    require(wp_expert_worker::parse_mm_pin_mode(nullptr) == mm_pin_mode::off,
            "unset WP_EXPERT_MM_PIN must default off");
    require(wp_expert_worker::parse_mm_pin_mode("") == mm_pin_mode::off,
            "empty WP_EXPERT_MM_PIN must be off");
    require(wp_expert_worker::parse_mm_pin_mode("0") == mm_pin_mode::off,
            "WP_EXPERT_MM_PIN=0 must be off");
    require(wp_expert_worker::parse_mm_pin_mode("1") == mm_pin_mode::on,
            "WP_EXPERT_MM_PIN=1 must be the legacy wide-request pin");
    require(wp_expert_worker::parse_mm_pin_mode("decode") == mm_pin_mode::decode,
            "WP_EXPERT_MM_PIN=decode must select the narrow-request pin");

    require(wp_expert_worker::parse_mm_pin_max_tokens(nullptr) == 8,
            "default WP_EXPERT_MM_PIN_MAX_TOKENS must be 8");
    require(wp_expert_worker::parse_mm_pin_max_tokens("") == 8,
            "empty WP_EXPERT_MM_PIN_MAX_TOKENS must be 8");
    require(wp_expert_worker::parse_mm_pin_max_tokens("1") == 1,
            "WP_EXPERT_MM_PIN_MAX_TOKENS=1 (decode only) must be honoured");
    require(wp_expert_worker::parse_mm_pin_max_tokens("0") == 1,
            "non-positive pin max tokens must clamp to 1");
    require(wp_expert_worker::parse_mm_pin_min_tokens(nullptr) == 9,
            "default WP_EXPERT_MM_PIN_MIN_TOKENS must be 9");

    // off: never pinned, whatever the shape.
    require(!wp_expert_worker::use_mm_pin(1, false, mm_pin_mode::off, 9, 8),
            "pin off must not pin decode");
    require(!wp_expert_worker::use_mm_pin(128, true, mm_pin_mode::off, 9, 8),
            "pin off must not pin a prefill chunk");

    // on: legacy behaviour, gather-path requests wider than a verify block.
    require(!wp_expert_worker::use_mm_pin(1, false, mm_pin_mode::on, 9, 8),
            "pin=1 must not pin decode (n_tokens<=8 was never pinned)");
    require(!wp_expert_worker::use_mm_pin(8, true, mm_pin_mode::on, 9, 8),
            "pin=1 must not pin a verify block");
    require(wp_expert_worker::use_mm_pin(128, true, mm_pin_mode::on, 9, 8),
            "pin=1 must pin a 128-token gather prefill chunk");
    require(wp_expert_worker::use_mm_pin(74, true, mm_pin_mode::on, 9, 8),
            "pin=1 must pin the stream4 74-token tail chunk");
    require(!wp_expert_worker::use_mm_pin(128, false, mm_pin_mode::on, 9, 8),
            "pin=1 only applies on the gather path");
    require(!wp_expert_worker::use_mm_pin(32, true, mm_pin_mode::on, 64, 8),
            "WP_EXPERT_MM_PIN_MIN_TOKENS must still gate the legacy pin");

    // decode: the complement -- narrow requests only, gather irrelevant.
    require(wp_expert_worker::use_mm_pin(1, false, mm_pin_mode::decode, 9, 8),
            "pin=decode must pin decode even though decode never gathers");
    require(wp_expert_worker::use_mm_pin(8, true, mm_pin_mode::decode, 9, 8),
            "pin=decode must pin a full 8-token spec-verify block");
    require(!wp_expert_worker::use_mm_pin(9, true, mm_pin_mode::decode, 9, 8),
            "pin=decode must not pin past the max at the default of 8");
    require(!wp_expert_worker::use_mm_pin(74, true, mm_pin_mode::decode, 9, 8),
            "pin=decode must leave the stream4 74-token tail unpinned");
    require(!wp_expert_worker::use_mm_pin(128, true, mm_pin_mode::decode, 9, 8),
            "pin=decode must leave a 128-token prefill chunk unpinned");
    require(!wp_expert_worker::use_mm_pin(0, false, mm_pin_mode::decode, 9, 8),
            "an empty request is not a decode");
    require(!wp_expert_worker::use_mm_pin(2, false, mm_pin_mode::decode, 9, 1),
            "WP_EXPERT_MM_PIN_MAX_TOKENS=1 must narrow the pin to bare decode");

    require(wp_expert_worker::parse_mm_pin_mode("decode", "Vulkan0") == mm_pin_mode::decode,
            "WP_EXPERT_MM_PIN=decode must stay decode on every device");
    require(wp_expert_worker::parse_mm_pin_mode("decode:ROCm0,CUDA0", "ROCm0") == mm_pin_mode::decode,
            "decode:allow-list must pin a listed device");
    require(wp_expert_worker::parse_mm_pin_mode("decode:ROCm0,CUDA0", "Vulkan0") == mm_pin_mode::off,
            "decode:allow-list must not pin an unlisted device");
    require(wp_expert_worker::parse_mm_pin_mode("ROCm0:decode,Vulkan0:decode", "Vulkan0") == mm_pin_mode::decode,
            "per-device mode map must honour the named mode");
    require(wp_expert_worker::parse_mm_pin_mode("ROCm0:decode,Vulkan0:decode", "CUDA0") == mm_pin_mode::off,
            "per-device mode map must default unlisted devices off");
    require(wp_expert_worker::parse_mm_pin_mode("ROCm0,CUDA0", "CUDA0") == mm_pin_mode::decode,
            "allow-list without a mode prefix must mean decode");
    require(wp_expert_worker::parse_mm_pin_mode("!Vulkan0", "ROCm0") == mm_pin_mode::decode,
            "exclusion list must pin devices not named");
    require(wp_expert_worker::parse_mm_pin_mode("!Vulkan0", "Vulkan0") == mm_pin_mode::off,
            "exclusion list must not pin the named device");

    require(!wp_expert_worker::parse_mul_mat_pin_kernel_mmvq(nullptr, "ROCm0"),
            "unset PIN_KERNEL must default MMQ");
    require(!wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("", "ROCm0"),
            "empty PIN_KERNEL must default MMQ");
    require(!wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("mmq", "ROCm0"),
            "PIN_KERNEL=mmq must be MMQ");
    require(wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("mmvq", "ROCm0"),
            "PIN_KERNEL=mmvq must be MMVQ on every device");
    require(wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("mmvq", "CUDA0"),
            "PIN_KERNEL=mmvq must be MMVQ on every device");
    require(wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("ROCm0:mmvq,ROCm1:mmvq,CUDA0:mmvq", "ROCm0"),
            "PIN_KERNEL map must enable a listed mmvq device");
    require(wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("ROCm0:mmvq,ROCm1:mmvq,CUDA0:mmvq", "CUDA0"),
            "PIN_KERNEL map must enable CUDA0:mmvq");
    require(!wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("ROCm0:mmvq,ROCm1:mmvq,CUDA0:mmvq", "Vulkan0"),
            "PIN_KERNEL map must leave Vulkan on the MMQ default");
    require(wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("ROCm0,CUDA0", "CUDA0"),
            "PIN_KERNEL allow-list must mean mmvq");
    require(!wp_expert_worker::parse_mul_mat_pin_kernel_mmvq("ROCm0,CUDA0", "Vulkan0"),
            "PIN_KERNEL allow-list must not enable an unlisted device");
    require(wp_expert_worker::mm_pin_pad_cols(1) == 8,
            "pad cols at n=1 must be 8");
    require(wp_expert_worker::mm_pin_pad_cols(8) == 8,
            "pad cols at n=8 must stay 8");
    require(wp_expert_worker::mm_pin_pad_cols(9) == 16,
            "pad cols at n=9 must round up to 16");

    const auto empty = wp_expert_worker::compact_routing_rows({ 0.0f, 0.0f, 0.0f });
    require(empty.idx.size() == 1 && empty.idx[0] == 0 && empty.weights[0] == 0.0f,
            "all-zero routing must keep a dummy idx 0 / weight 0");
    const auto mixed = wp_expert_worker::compact_routing_rows({ 0.0f, 0.5f, 0.0f, 1.25f });
    require(mixed.idx.size() == 2 && mixed.idx[0] == 1 && mixed.idx[1] == 3,
            "compact idx must be the nonzero token positions");
    require(mixed.weights.size() == 2 && mixed.weights[0] == 0.5f && mixed.weights[1] == 1.25f,
            "compact weights must follow idx");
}

static void test_batch_mmid_ids() {
    // Dense: identity ids, assignment-index columns, no pad expert.
    const std::vector<std::vector<float>> dense = {
        { 0.5f },
        { 0.25f },
        { 0.125f },
    };
    const auto d = wp_expert_worker::build_batch_mmid_ids(dense, false);
    require(d.n_experts == 3 && d.n_tokens == 1 && d.k_width == 3,
            "dense k_width must equal n_experts");
    require(!d.used_pad_expert, "dense identity ids must not invent a pad expert");
    require(d.ids.size() == 3 && d.ids[0] == 0 && d.ids[1] == 1 && d.ids[2] == 2,
            "dense ids must be identity in assignment order");
    require(d.route_w.size() == 3 && d.route_w[0] == 0.5f && d.route_w[2] == 0.125f,
            "dense route_w must follow assignment order");
    require(d.expert_rows.size() == 3 && d.expert_rows[1].size() == 1 &&
                d.expert_rows[1][0] == 0,
            "dense expert_rows must name every token");
    require(wp_expert_worker::batch_mmid_ids_valid(d) &&
                wp_expert_worker::batch_mmid_n_as(d) == 3,
            "dense ids must be unique and n_as == n_experts");

    // Gather: invert per-expert rows onto [k_width, n_tokens]. Token 0 sees
    // experts 0 then 2 (assignment order); token 1 sees only expert 2 and is
    // padded with expert index n_experts at route weight 0.
    const std::vector<std::vector<float>> gather = {
        { 0.5f, 0.0f },
        { 0.0f, 0.0f },
        { 0.1f, 0.2f },
    };
    const auto g = wp_expert_worker::build_batch_mmid_ids(gather, true);
    require(g.n_experts == 3 && g.n_tokens == 2 && g.k_width == 2,
            "gather k_width must be the max experts-per-token");
    require(g.used_pad_expert, "uneven gather rank must pad with expert n");
    require(g.ids.size() == 4 && g.ids[0] == 0 && g.ids[1] == 2 &&
                g.ids[2] == 2 && g.ids[3] == 3,
            "gather ids must pack assignment order then pad with n_experts");
    require(g.route_w.size() == 4 && g.route_w[0] == 0.5f && g.route_w[1] == 0.1f &&
                g.route_w[2] == 0.2f && g.route_w[3] == 0.0f,
            "padded gather slots must have route weight 0");
    require(g.expert_rows[0].size() == 1 && g.expert_rows[0][0] == 0,
            "expert 0's compacted row is token 0");
    require(g.expert_rows[1].size() == 1 && g.expert_rows[1][0] == 0 &&
                gather[1][0] == 0.0f,
            "all-zero expert keeps compact_routing_rows dummy idx 0");
    require(wp_expert_worker::batch_mmid_ids_valid(g),
            "one-pad gather ids must be unique and in range");
    require(wp_expert_worker::batch_mmid_n_as(g) == 4,
            "one-pad gather n_as is n_experts+1");

    const auto even = wp_expert_worker::build_batch_mmid_ids(
        { { 0.5f, 0.0f }, { 0.0f, 0.3f } }, true);
    require(even.k_width == 1 && !even.used_pad_expert &&
                even.ids[0] == 0 && even.ids[1] == 1,
            "uniform rank 1 must not pad");
    require(wp_expert_worker::batch_mmid_ids_valid(even) &&
                wp_expert_worker::batch_mmid_n_as(even) == 2,
            "uniform rank 1 n_as is n_experts");

    // Two pad slots on one token: dummy expert n once, then an unused real
    // expert. Repeating n was the scatter-quantize GPU fault.
    const std::vector<std::vector<float>> multi_pad = {
        { 0.5f, 0.0f },
        { 0.3f, 0.0f },
        { 0.1f, 0.2f },
    };
    const auto mp = wp_expert_worker::build_batch_mmid_ids(multi_pad, true);
    require(mp.n_experts == 3 && mp.n_tokens == 2 && mp.k_width == 3 &&
                mp.used_pad_expert,
            "max rank 3 with a rank-1 token must pad");
    require(mp.ids[0] == 0 && mp.ids[1] == 1 && mp.ids[2] == 2,
            "full-rank token keeps assignment order");
    require(mp.ids[3] == 2 && mp.ids[4] == 3 && mp.ids[5] == 0,
            "rank-1 token: real expert, dummy n, unused real");
    require(mp.route_w[3] == 0.2f && mp.route_w[4] == 0.0f && mp.route_w[5] == 0.0f,
            "pad slots must have route weight 0");
    require(wp_expert_worker::batch_mmid_ids_valid(mp),
            "multi-pad ids must be unique per token and < n_as");
    const size_t n_as = wp_expert_worker::batch_mmid_n_as(mp);
    require(n_as == 4, "multi-pad n_as is n_experts+1");
    for (int32_t id : mp.ids) {
        require(id >= 0 && (size_t) id < n_as, "every ids value must be < n_as");
    }
    std::vector<int64_t> bases = { 0x1000, 0x2000, 0x3000 };
    std::vector<int64_t> ptrs(n_as, 0);
    wp_expert_worker::fill_batch_mmid_expert_ptrs(
        ptrs.data(), n_as, bases.data(), mp.n_experts, mp.used_pad_expert);
    require(ptrs[0] == 0x1000 && ptrs[1] == 0x2000 && ptrs[2] == 0x3000 &&
                ptrs[3] == ptrs[0],
            "every pointer slot filled; pad expert copies expert 0");
    for (int64_t p : ptrs) {
        require(p != 0, "no empty expert pointer slot");
    }

    // Same allow-list parser as WP_EXPERT_ARENA_PREFILL.
    require(!wp_expert_worker::parse_arena_prefill_enabled(nullptr, "ROCm0"),
            "unset WP_EXPERT_BATCH_MMID must default off");
    require(wp_expert_worker::parse_arena_prefill_enabled("ROCm0,ROCm1,CUDA0", "ROCm1"),
            "BATCH_MMID allow-list must enable a named HIP/CUDA device");
    require(!wp_expert_worker::parse_arena_prefill_enabled("ROCm0,ROCm1,CUDA0", "Vulkan0"),
            "BATCH_MMID allow-list must not enable Vulkan");
    require(!wp_expert_worker::parse_arena_prefill_enabled("ROCm0,ROCm1,CUDA0", "CPU"),
            "BATCH_MMID allow-list must not enable CPU");
}

static void test_batch_mmid_association() {
    const size_t n = 6;
    const size_t pageins[] = { 0, 2, n };
    std::vector<wp_expert_worker::BatchMmidAssociation> plans;
    std::vector<size_t> chunks;
    for (size_t p : pageins) {
        plans.push_back(wp_expert_worker::plan_batch_mmid_association(
            true, n, false, false, false));
        chunks.push_back(wp_expert_worker::batch_mmid_compute_chunks(true, n, p, 4));
    }
    for (size_t i = 0; i < plans.size(); ++i) {
        require(plans[i].use_mmid,
                "BATCH_MMID on must take mmid regardless of residency");
        require(plans[i].reason == wp_expert_worker::batch_mmid_ineligible_reason::none,
                "full request is eligible");
        require(plans[i].fold_order.size() == n, "fold names every assignment");
        require(chunks[i] == 1, "BATCH_MMID must not chunk on n_pagein");
        require(plans[i].fold_order == plans[0].fold_order,
                "fold order must not depend on residency");
    }
    for (size_t i = 0; i < n; ++i) {
        require(plans[0].fold_order[i] == i, "fold is assignment-index order");
    }

    require(wp_expert_worker::batch_mmid_compute_chunks(false, n, 0, 4) == 1,
            "all-resident serial path is one chunk");
    require(wp_expert_worker::batch_mmid_compute_chunks(false, n, 2, 4) == 4,
            "legacy chunking still splits when BATCH_MMID is off");

    const auto cpu = wp_expert_worker::plan_batch_mmid_association(
        true, n, true, false, false);
    require(!cpu.use_mmid, "cpu_on_arrival cannot take GPU mul_mat_id");
    require(cpu.reason == wp_expert_worker::batch_mmid_ineligible_reason::cpu_on_arrival,
            "cpu_on_arrival is counted ineligible");
    require(cpu.fold_order == plans[0].fold_order,
            "ineligible still folds in assignment order");

    const auto sub = wp_expert_worker::plan_batch_mmid_association(
        true, n, false, false, true);
    require(!sub.use_mmid, "a sub-range must not take mmid");
    require(sub.reason == wp_expert_worker::batch_mmid_ineligible_reason::subrange,
            "sub-range is counted ineligible");
    require(sub.fold_order == plans[0].fold_order,
            "sub-range ineligible still reports canonical fold order");

    const auto dense = wp_expert_worker::plan_batch_mmid_association(
        true, n, false, true, false);
    require(!dense.use_mmid, "force_dense (selfcheck) stays per-expert");
    require(dense.reason == wp_expert_worker::batch_mmid_ineligible_reason::force_dense,
            "force_dense is counted ineligible");
}

static void test_arena_prefill_device_policy() {
    // Unset / missing must default OFF, same as parse_env_default_off.
    require(!wp_expert_worker::parse_arena_prefill_enabled(nullptr, "ROCm0"),
            "unset WP_EXPERT_ARENA_PREFILL must default off");
    require(!wp_expert_worker::parse_arena_prefill_enabled("", "ROCm0"),
            "empty WP_EXPERT_ARENA_PREFILL must default off");

    // Plain 0/1 apply to every device.
    require(!wp_expert_worker::parse_arena_prefill_enabled("0", "ROCm0"),
            "WP_EXPERT_ARENA_PREFILL=0 must disable grouped prefill everywhere");
    require(!wp_expert_worker::parse_arena_prefill_enabled("0", "Vulkan0"),
            "WP_EXPERT_ARENA_PREFILL=0 must disable grouped prefill everywhere");
    require(wp_expert_worker::parse_arena_prefill_enabled("1", "ROCm0"),
            "WP_EXPERT_ARENA_PREFILL=1 must enable grouped prefill everywhere");
    require(wp_expert_worker::parse_arena_prefill_enabled("1", "Vulkan0"),
            "WP_EXPERT_ARENA_PREFILL=1 must enable grouped prefill everywhere");

    // Allow-list: only the named devices are enabled.
    require(wp_expert_worker::parse_arena_prefill_enabled("ROCm0,CUDA0", "ROCm0"),
            "device allow-list must enable a listed device");
    require(wp_expert_worker::parse_arena_prefill_enabled("ROCm0,CUDA0", "CUDA0"),
            "device allow-list must enable a listed device");
    require(!wp_expert_worker::parse_arena_prefill_enabled("ROCm0,CUDA0", "Vulkan0"),
            "device allow-list must not enable an unlisted device");

    // Exclusion list: every device except the named ones is enabled.
    require(!wp_expert_worker::parse_arena_prefill_enabled("!Vulkan0", "Vulkan0"),
            "exclusion list must disable the named device");
    require(wp_expert_worker::parse_arena_prefill_enabled("!Vulkan0", "ROCm0"),
            "exclusion list must enable devices not named");
    require(wp_expert_worker::parse_arena_prefill_enabled("!Vulkan0", "ROCm1"),
            "exclusion list must enable devices not named");

    // Whitespace around the value and around each comma-separated name must
    // be tolerated.
    require(wp_expert_worker::parse_arena_prefill_enabled(" ROCm0 , CUDA0 ", "ROCm0"),
            "whitespace around list entries must be tolerated");
    require(wp_expert_worker::parse_arena_prefill_enabled(" ROCm0 , CUDA0 ", "CUDA0"),
            "whitespace around list entries must be tolerated");
    require(!wp_expert_worker::parse_arena_prefill_enabled(" ROCm0 , CUDA0 ", "Vulkan0"),
            "whitespace-tolerant list must still exclude unlisted devices");
    require(wp_expert_worker::parse_arena_prefill_enabled(" ! Vulkan0 ", "ROCm0"),
            "whitespace around an exclusion list must be tolerated");
    require(!wp_expert_worker::parse_arena_prefill_enabled(" ! Vulkan0 ", "Vulkan0"),
            "whitespace around an exclusion list must be tolerated");
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

static void test_prefill_mul_mat_pin_chunk_byte_identical() {
    static constexpr uint32_t TOKENS = 256;
    static constexpr uint32_t CHUNKS = 4;

    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = reserve_port();
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
        pipe_socket_ptr socket = connect_with_retry(options.listen_port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload), "failed to receive chunk-pin HELLO");
        require(type == PIPE_HELLO && seq_id == 0, "chunk-pin worker did not send HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send chunk-pin client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive chunk-pin HELLO acknowledgement");
        require(type == PIPE_EXPERT_HELLO_ACK && seq_id == 0 &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "chunk-pin worker rejected matching HELLO");

        pipe_expert_dispatch_req request;
        request.layer = LAYER;
        request.n_tokens = TOKENS;
        request.activations.resize((size_t) TOKENS * N_EMBD);
        for (size_t i = 0; i < request.activations.size(); ++i) {
            request.activations[i] = ((int) ((i * 17 + i / N_EMBD) % 31) - 15) * 0.013f;
        }
        for (int expert = 0; expert < 4; ++expert) {
            pipe_expert_assignment assignment;
            assignment.expert_id = expert;
            assignment.weights.resize(TOKENS);
            for (uint32_t token = 0; token < TOKENS; ++token) {
                assignment.weights[token] =
                    (1 + (int) ((token * 5 + expert * 3) % 17)) * 0.03125f;
            }
            request.assignments.push_back(std::move(assignment));
        }

        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 700,
                                payload.data(), payload.size()),
                "failed to send monolithic chunk-pin dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive monolithic chunk-pin partial");
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 700,
                "monolithic chunk-pin dispatch returned the wrong frame");
        const pipe_expert_partial whole =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        require(whole.n_tokens == TOKENS, "monolithic chunk-pin token count mismatch");

        std::vector<float> assembled((size_t) TOKENS * N_EMBD);
        const uint32_t chunk_rows = TOKENS / CHUNKS;
        for (uint32_t chunk_index = 0; chunk_index < CHUNKS; ++chunk_index) {
            const uint32_t token_start = chunk_index * chunk_rows;
            const uint32_t token_end = chunk_index + 1 == CHUNKS
                ? TOKENS : token_start + chunk_rows;
            pipe_expert_dispatch_chunk chunk;
            chunk.chunk_index = chunk_index;
            chunk.chunk_count = CHUNKS;
            chunk.total_tokens = TOKENS;
            chunk.token_start = token_start;
            chunk.token_end = token_end;
            chunk.request.layer = request.layer;
            chunk.request.n_tokens = token_end - token_start;
            chunk.request.swiglu_clamp = request.swiglu_clamp;
            for (const pipe_expert_assignment & assignment : request.assignments) {
                pipe_expert_assignment sliced;
                sliced.expert_id = assignment.expert_id;
                sliced.weights.assign(assignment.weights.begin() + token_start,
                                      assignment.weights.begin() + token_end);
                chunk.request.assignments.push_back(std::move(sliced));
            }
            chunk.request.activations.assign(
                request.activations.begin() + (size_t) token_start * N_EMBD,
                request.activations.begin() + (size_t) token_end * N_EMBD);

            payload = pipe_encode_expert_dispatch_chunk(chunk);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_CHUNK, 701,
                                    payload.data(), payload.size()),
                    "failed to send chunk-pin dispatch chunk");
            require(pipe_recv_frame(*socket, type, seq_id, payload),
                    "failed to receive chunk-pin partial chunk");
            require(type == PIPE_EXPERT_PARTIAL_CHUNK && seq_id == 701,
                    "chunk-pin dispatch returned the wrong frame");
            const pipe_expert_partial_chunk partial = pipe_decode_expert_partial_chunk(
                payload.data(), payload.size(), N_EMBD);
            require(partial.chunk_index == chunk_index &&
                        partial.chunk_count == CHUNKS &&
                        partial.total_tokens == TOKENS &&
                        partial.token_start == token_start &&
                        partial.token_end == token_end,
                    "chunk-pin partial range mismatch");
            std::copy(partial.partial.partial.begin(), partial.partial.partial.end(),
                      assembled.begin() + (size_t) token_start * N_EMBD);
        }

        require(whole.partial.size() == assembled.size(), "chunk-pin result shape mismatch");
        for (size_t i = 0; i < assembled.size(); ++i) {
            if (std::memcmp(&whole.partial[i], &assembled[i], sizeof(float)) != 0) {
                throw std::runtime_error(
                    "pinned prefill changed when split into token chunks at " +
                    std::to_string(i) + ": whole=" + std::to_string(whole.partial[i]) +
                    " chunked=" + std::to_string(assembled[i]));
            }
        }
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
    require(server_result == 0, "chunk-pin worker returned failure");
}

static pipe_expert_dispatch_req make_sparse_arena_prefill_request() {
    static constexpr uint32_t TOKENS = 256;
    pipe_expert_dispatch_req request;
    request.layer = LAYER;
    request.n_tokens = TOKENS;
    request.activations.resize((size_t) TOKENS * N_EMBD);
    for (size_t i = 0; i < request.activations.size(); ++i) {
        request.activations[i] =
            ((int) ((i * 17 + i / N_EMBD) % 31) - 15) * 0.013f;
    }
    for (int expert = 0; expert < 4; ++expert) {
        pipe_expert_assignment assignment;
        assignment.expert_id = expert;
        assignment.weights.resize(TOKENS, 0.0f);
        for (uint32_t token = 0; token < TOKENS; ++token) {
            const int first = (int) token % 4;
            const int second = (first + 2) % 4;
            if (expert == first || (token % 3 != 0 && expert == second)) {
                assignment.weights[token] =
                    (1 + (int) ((token * 5 + expert * 3) % 17)) * 0.03125f;
            }
        }
        request.assignments.push_back(std::move(assignment));
    }
    return request;
}

static std::vector<float> run_sparse_arena_prefill(
        const Fixture & fixture,
        const pipe_expert_dispatch_req & request,
        const char * arena_prefill) {
    const ScopedEnv arena_env("WP_EXPERT_ARENA_PREFILL", arena_prefill);
    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = reserve_port();
    // 8, not 4: every arena reserves n_expert_used PAD slots for the grouped
    // path's weight-0 filler ids, and the pool refuses to reserve them below a
    // class's pin_floor (4 here, the pages of one layer). A 4-slot pool is all
    // floor, gets no pads, and the grouped path would decline every request --
    // which is exactly what this test exists to exercise. 8 slots leave room
    // for the 2 pads AND all four pages.
    options.slots             = 8;
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

    std::vector<float> result;
    try {
        pipe_socket_ptr socket = connect_with_retry(options.listen_port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive arena-prefill HELLO");
        require(type == PIPE_HELLO && seq_id == 0,
                "arena-prefill worker did not send HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send arena-prefill client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive arena-prefill HELLO acknowledgement");
        require(type == PIPE_EXPERT_HELLO_ACK && seq_id == 0 &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "arena-prefill worker rejected matching HELLO");

        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 710,
                                payload.data(), payload.size()),
                "failed to send arena-prefill dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive arena-prefill partial");
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 710,
                "arena-prefill dispatch returned the wrong frame");
        const pipe_expert_partial partial =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        require(partial.n_tokens == request.n_tokens,
                "arena-prefill token count mismatch");
        result = partial.partial;
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
    require(server_result == 0, "arena-prefill worker returned failure");
    return result;
}

static void test_prefill_arena_grouped_chunk_byte_identical() {
    static constexpr uint32_t TOKENS = 256;
    static constexpr uint32_t CHUNKS = 4;

    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const pipe_expert_dispatch_req request = make_sparse_arena_prefill_request();
    const ScopedEnv arena_env("WP_EXPERT_ARENA_PREFILL", "1");

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = reserve_port();
    // 8, not 4: see the pad-slot note in run_sparse_arena_prefill.
    options.slots             = 8;
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
        pipe_socket_ptr socket = connect_with_retry(options.listen_port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive grouped-prefill HELLO");
        require(type == PIPE_HELLO && seq_id == 0,
                "grouped-prefill worker did not send HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send grouped-prefill client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive grouped-prefill HELLO acknowledgement");
        require(type == PIPE_EXPERT_HELLO_ACK && seq_id == 0 &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "grouped-prefill worker rejected matching HELLO");

        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 711,
                                payload.data(), payload.size()),
                "failed to send monolithic grouped-prefill dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive monolithic grouped-prefill partial");
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 711,
                "monolithic grouped-prefill dispatch returned the wrong frame");
        const pipe_expert_partial whole =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        require(whole.n_tokens == TOKENS,
                "monolithic grouped-prefill token count mismatch");

        std::vector<float> assembled((size_t) TOKENS * N_EMBD);
        const uint32_t chunk_rows = TOKENS / CHUNKS;
        for (uint32_t chunk_index = 0; chunk_index < CHUNKS; ++chunk_index) {
            const uint32_t token_start = chunk_index * chunk_rows;
            const uint32_t token_end = chunk_index + 1 == CHUNKS
                ? TOKENS : token_start + chunk_rows;
            pipe_expert_dispatch_chunk chunk;
            chunk.chunk_index = chunk_index;
            chunk.chunk_count = CHUNKS;
            chunk.total_tokens = TOKENS;
            chunk.token_start = token_start;
            chunk.token_end = token_end;
            chunk.request.layer = request.layer;
            chunk.request.n_tokens = token_end - token_start;
            chunk.request.swiglu_clamp = request.swiglu_clamp;
            for (const pipe_expert_assignment & assignment : request.assignments) {
                pipe_expert_assignment sliced;
                sliced.expert_id = assignment.expert_id;
                sliced.weights.assign(assignment.weights.begin() + token_start,
                                      assignment.weights.begin() + token_end);
                chunk.request.assignments.push_back(std::move(sliced));
            }
            chunk.request.activations.assign(
                request.activations.begin() + (size_t) token_start * N_EMBD,
                request.activations.begin() + (size_t) token_end * N_EMBD);

            payload = pipe_encode_expert_dispatch_chunk(chunk);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_CHUNK, 712,
                                    payload.data(), payload.size()),
                    "failed to send grouped-prefill dispatch chunk");
            require(pipe_recv_frame(*socket, type, seq_id, payload),
                    "failed to receive grouped-prefill partial chunk");
            require(type == PIPE_EXPERT_PARTIAL_CHUNK && seq_id == 712,
                    "grouped-prefill chunk returned the wrong frame");
            const pipe_expert_partial_chunk partial = pipe_decode_expert_partial_chunk(
                payload.data(), payload.size(), N_EMBD);
            require(partial.chunk_index == chunk_index &&
                        partial.chunk_count == CHUNKS &&
                        partial.total_tokens == TOKENS &&
                        partial.token_start == token_start &&
                        partial.token_end == token_end,
                    "grouped-prefill partial range mismatch");
            std::copy(partial.partial.partial.begin(), partial.partial.partial.end(),
                      assembled.begin() + (size_t) token_start * N_EMBD);
        }

        require(whole.partial.size() == assembled.size(),
                "grouped-prefill result shape mismatch");
        for (size_t i = 0; i < assembled.size(); ++i) {
            if (std::memcmp(&whole.partial[i], &assembled[i], sizeof(float)) != 0) {
                throw std::runtime_error(
                    "grouped prefill changed when split into token chunks at " +
                    std::to_string(i) + ": whole=" + std::to_string(whole.partial[i]) +
                    " chunked=" + std::to_string(assembled[i]));
            }
        }
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
    require(server_result == 0, "grouped-prefill worker returned failure");
}

static void test_prefill_arena_grouped_matches_gather() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const pipe_expert_dispatch_req request = make_sparse_arena_prefill_request();
    const std::vector<float> gather =
        run_sparse_arena_prefill(fixture, request, "0");
    const std::vector<float> grouped =
        run_sparse_arena_prefill(fixture, request, "1");

    require(gather.size() == grouped.size(),
            "grouped and gather prefill result shapes differ");
    double max_abs_diff = 0.0;
    double max_abs_gather = 0.0;
    for (size_t i = 0; i < gather.size(); ++i) {
        max_abs_diff = std::max(
            max_abs_diff, std::fabs((double) gather[i] - (double) grouped[i]));
        max_abs_gather = std::max(max_abs_gather, std::fabs((double) gather[i]));
    }
    const double tolerance = 1e-4 * max_abs_gather + 1e-6;
    if (max_abs_diff > tolerance) {
        throw std::runtime_error(
            "grouped prefill differs from gather: max_abs_diff=" +
            std::to_string(max_abs_diff) + " tolerance=" +
            std::to_string(tolerance));
    }
}

// ---------------------------------------------------------------------------
// PRODUCTION-GEOMETRY GROUPED PREFILL (2026-09-02)
//
// Every other grouped-prefill test in this file runs the toy fixture: 4 f32
// experts, n_embd 32, one arena, CPU. That fixture cannot see any of the three
// defects the rig probes found -- a slot stride that is not a multiple of a
// role's quant block size, duplicate slot ids inside a token's id row, or a
// backend that ignores the slab stride -- because it has one arena, one type
// and no quantisation. These tests build the real thing instead:
//
//   qwen38-next, n_embd 2560, n_ff_exp 640 width-sliced 7:3, so the per-worker
//   slice widths are 448 (main box) and 192 (2026 box); the three geometry
//   variants from
//   ~/models/qwen38-eslice-v2/q38-eslice-slice-00000-experts-manifest.json
//   (expert_slicing.geometry_variants[*].role_geometry):
//       0: gate/up q4_K, down q5_1
//       1: gate/up q5_K, down q8_0
//       2: gate/up q4_K, down q8_0
//   128-token requests (the spine streams 512-token ubatches in four chunks),
//   sparse top-k routing, several arenas per size class, and a block of
//   resident-but-unrouted experts so at least one arena draws no route at all.
//
// Selected by env:
//   WP_WORKER_TEST_BACKEND   CPU (default) / ROCm0 / CUDA0 / Vulkan0 ...
//   WP_WORKER_TEST_GEOMETRY  0 / 1 / 2, unset = all three
// ---------------------------------------------------------------------------

struct ProdGeometry {
    const char * name;
    ggml_type    gate_up;
    ggml_type    down;
};

static const ProdGeometry PROD_GEOMETRIES[3] = {
    { "gate/up q4_K + down q5_1", GGML_TYPE_Q4_K, GGML_TYPE_Q5_1 },
    { "gate/up q5_K + down q8_0", GGML_TYPE_Q5_K, GGML_TYPE_Q8_0 },
    { "gate/up q4_K + down q8_0", GGML_TYPE_Q4_K, GGML_TYPE_Q8_0 },
};

static constexpr int64_t  PROD_N_EMBD  = 2560;
static constexpr int      PROD_WIDTHS[2] = { 448, 192 };
// Production runs ~180 assignments per 128-token chunk against ~8150 slots.
// Scaled to 16 here purely for CPU wall time; what the geometry has to keep is
// (a) several arenas per size class and (b) at least one arena with no route in
// the chunk. The arena split (see WP_EXPERT_ARENA_MAX_BYTES below) and
// PROD_UNROUTED_FIRST give both.
//
// *** WHY THERE ARE FOUR LAYERS WHEN ONLY ONE IS EVER DISPATCHED. ***
// Every arena reserves n_expert_used == PROD_N_EXPERT_USED == 10 PAD slots at
// its end (never bound, never a DMA target -- they are the weight-0 filler ids
// the grouped path needs), so an arena's USABLE slots are its positions minus
// 10. plan_resources caps a class at one slot per PAGE and the pool refuses to
// reserve pads below a class's pin_floor (the largest per-layer page count), so
// a one-layer fixture with 16 pages could never reserve anything. Four layers
// of PROD_EXPERTS give 64 pages with a pin_floor of 16: four arenas, ten pads
// each, and 24 usable slots -- comfortably more than the 16 pages layer
// PROD_LAYER actually demands. Layers 1..3 are never dispatched to, so they are
// never paged in; they exist purely to widen the pool the way production's
// forty-odd layers do.
static constexpr int      PROD_EXPERTS = 16;          // experts per layer
static constexpr int      PROD_LAYERS  = 4;           // pages = experts * layers
static constexpr int      PROD_UNROUTED_FIRST = 12;   // experts 12..15 draw no route
static constexpr int      PROD_LAYER   = 0;
static constexpr uint32_t PROD_TOKENS  = 128;
static constexpr uint32_t PROD_CHUNKS  = 4;
static constexpr int      PROD_TOP_K   = 3;           // routes per token
static constexpr int      PROD_N_EXPERT_USED = 10;    // top-10, as in production

struct ProdFixture {
    fs::path descriptor;
    fs::path manifest;
    uint64_t page_bytes = 0;
};

static float prod_weight_value(int expert, int role, int64_t row, int64_t col) {
    const int64_t pattern =
        (row * 31 + col * 17 + (int64_t) expert * 13 + (int64_t) role * 7) % 23;
    return 0.02f * (float) (pattern - 11);
}

// Quantise one [n_per_row, nrows] role matrix for one expert.
static std::vector<uint8_t> prod_role_bytes(
        ggml_type type, int64_t n_per_row, int64_t nrows, int expert, int role) {
    std::vector<float> src((size_t) n_per_row * (size_t) nrows);
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < n_per_row; ++col) {
            src[(size_t) row * (size_t) n_per_row + (size_t) col] =
                prod_weight_value(expert, role, row, col);
        }
    }
    const size_t bytes = (size_t) ggml_row_size(type, n_per_row) * (size_t) nrows;
    std::vector<uint8_t> dst(bytes);
    const size_t written = ggml_quantize_chunk(
        type, src.data(), dst.data(), 0, nrows, n_per_row, nullptr);
    require(written == bytes, "production fixture quantisation size mismatch");
    return dst;
}

static ProdFixture make_production_fixture(
        const fs::path & dir, const ProdGeometry & geometry, int width) {
    ProdFixture fixture;
    fixture.descriptor = dir / "prod.expert-descriptor.json";
    fixture.manifest   = dir / "prod-experts-manifest.json";

    const json identity = {
        { "algorithm", "sha256" },
        { "value", "wp-expert-worker-production-geometry" },
    };

    const uint64_t gate_bytes =
        (uint64_t) ggml_row_size(geometry.gate_up, PROD_N_EMBD) * (uint64_t) width;
    const uint64_t up_bytes   = gate_bytes;
    const uint64_t down_bytes =
        (uint64_t) ggml_row_size(geometry.down, width) * (uint64_t) PROD_N_EMBD;
    const uint64_t payload    = up_bytes + gate_bytes + down_bytes;
    const uint64_t padded     = GGML_PAD(payload, (uint64_t) 4096);
    const uint64_t padding    = padded - payload;
    fixture.page_bytes = padded;

    const auto role_desc = [&](ggml_type type, int64_t ne0, int64_t ne1,
                               uint64_t bytes, const char * role) {
        return json{
            { "ggml_type", (int) type },
            { "ggml_type_name", ggml_type_name(type) },
            { "shape", json::array({ ne0, ne1 }) },
            { "bytes_per_expert", bytes },
            { "source_tensor_name", std::string("prod.") + role },
        };
    };

    write_json(fixture.descriptor, {
        { "format", "llama.cpp.weight-pager.expert-descriptor" },
        { "version", 1 },
        { "source_model",
          {
              { "input_model", "prod.gguf" },
              { "model_files", { "prod.gguf" } },
              { "architecture", "qwen38next" },
              { "name", "qwen38-next" },
          } },
        { "shard_manifest_identity", identity },
        { "retained_expert_range", { { "first", 0 }, { "last", PROD_EXPERTS - 1 } } },
        { "hparams",
          {
              { "n_layer", PROD_LAYERS },
              { "n_embd", PROD_N_EMBD },
              { "n_ff_exp", width },
              { "n_expert", PROD_EXPERTS },
              { "n_expert_used", PROD_N_EXPERT_USED },
              { "activation", "silu" },
          } },
        { "layers", [&]() {
              json layers = json::array();
              for (int layer = PROD_LAYER; layer < PROD_LAYER + PROD_LAYERS; ++layer) {
                  layers.push_back({
                      { "layer", layer },
                      { "roles",
                        {
                            { "gate", role_desc(geometry.gate_up, PROD_N_EMBD, width, gate_bytes, "gate") },
                            { "up",   role_desc(geometry.gate_up, PROD_N_EMBD, width, up_bytes,   "up") },
                            { "down", role_desc(geometry.down,    width, PROD_N_EMBD, down_bytes, "down") },
                        } },
                  });
              }
              return layers;
          }() },
    });

    // ONE SHARD PER LAYER: load_catalog rejects a shard index whose
    // layer_first != layer_last, so the pool-widening layers each get their own
    // blob + sidecar exactly as the real slicer emits them.
    const std::vector<uint8_t> zero_pad((size_t) padding, 0);
    std::vector<uint8_t> filler[3];
    json shards = json::array();
    uint64_t total_blob_bytes = 0;
    for (int layer = PROD_LAYER; layer < PROD_LAYER + PROD_LAYERS; ++layer) {
        const int shard_index = layer - PROD_LAYER;
        char stem_buf[64];
        std::snprintf(stem_buf, sizeof(stem_buf), "prod-%05d-of-%05d",
                      shard_index + 1, PROD_LAYERS);
        const std::string stem = stem_buf;
        const fs::path sidecar = dir / (stem + ".wpi.json");
        const fs::path blob    = dir / (stem + ".wpb");

        std::ofstream blob_output(blob, std::ios::binary);
        if (!blob_output) {
            throw std::runtime_error("failed to create production blob");
        }
        json groups = json::array();
        uint64_t offset = 0;
        for (int expert = 0; expert < PROD_EXPERTS; ++expert) {
            json members = json::array();
            // Blob order is up (mask 1), gate (mask 2), down (mask 4) -- the
            // same order the real slicer writes and load_catalog walks.
            struct RoleLayout { const char * name; uint64_t mask; ggml_type type;
                                int64_t ne0; int64_t ne1; uint64_t bytes; int role_index; };
            const RoleLayout layout[3] = {
                { "up",   1, geometry.gate_up, PROD_N_EMBD, width, up_bytes,   0 },
                { "gate", 2, geometry.gate_up, PROD_N_EMBD, width, gate_bytes, 1 },
                { "down", 4, geometry.down,    width, PROD_N_EMBD, down_bytes, 2 },
            };
            for (const RoleLayout & role : layout) {
                // Only PROD_LAYER is ever dispatched to, so the CONTENT of the
                // pool-widening layers is irrelevant: quantise PROD_LAYER for
                // real and reuse one filler set for the rest instead of paying
                // for (PROD_LAYERS-1)*PROD_EXPERTS more quantisations.
                std::vector<uint8_t> & cached = filler[role.role_index];
                std::vector<uint8_t> bytes;
                const uint8_t * data = nullptr;
                if (layer != PROD_LAYER) {
                    if (cached.empty()) {
                        cached = prod_role_bytes(role.type, role.ne0, role.ne1,
                                                 PROD_EXPERTS, role.role_index);
                    }
                    require(cached.size() == role.bytes,
                            "production role byte count mismatch");
                    data = cached.data();
                } else {
                    bytes = prod_role_bytes(role.type, role.ne0, role.ne1,
                                            expert, role.role_index);
                    require(bytes.size() == role.bytes,
                            "production role byte count mismatch");
                    data = bytes.data();
                }
                blob_output.write(reinterpret_cast<const char *>(data),
                                  (std::streamsize) role.bytes);
                members.push_back({
                    { "role_mask", role.mask },
                    { "size", role.bytes },
                    { "offset", offset },
                    { "catalog_name",
                      "blk." + std::to_string(layer) + ".ffn_" +
                      std::string(role.name) + "." + std::to_string(expert) + ".weight" },
                    { "source_tensor_name", std::string("prod.") + role.name },
                    { "source_file_idx", 0 },
                    { "source_file_offset", offset },
                });
                offset += role.bytes;
            }
            if (padding > 0) {
                blob_output.write(reinterpret_cast<const char *>(zero_pad.data()),
                                  (std::streamsize) zero_pad.size());
                offset += padding;
            }
            groups.push_back({
                { "block_idx", layer },
                { "expert_idx", expert },
                { "member_count", 3 },
                { "padding_bytes", padding },
                { "members", std::move(members) },
            });
        }
        blob_output.close();
        require(offset == padded * (uint64_t) PROD_EXPERTS,
                "production blob size mismatch");

        write_json(sidecar, {
            { "format", "llama.cpp.weight-pager.expert-shard-index" },
            { "version", 1 },
            { "blob_file", blob.filename().string() },
            { "shard_index", shard_index },
            { "shard_count", PROD_LAYERS },
            { "layer_first", layer },
            { "layer_last", layer },
            { "group_count", PROD_EXPERTS },
            { "blob_bytes", offset },
            { "content_hash", identity },
            { "model_files", { "prod.gguf" } },
            { "groups", std::move(groups) },
        });
        shards.push_back({
            { "blob_file", blob.filename().string() },
            { "index_file", sidecar.filename().string() },
            { "shard_index", shard_index },
            { "layer_first", layer },
            { "layer_last", layer },
            { "group_count", PROD_EXPERTS },
            { "blob_bytes", offset },
            { "content_hash", identity },
        });
        total_blob_bytes += offset;
    }

    write_json(fixture.manifest, {
        { "format", "llama.cpp.weight-pager.expert-shard-manifest" },
        { "version", 1 },
        { "input_model", "prod.gguf" },
        { "model_files", { "prod.gguf" } },
        { "sharding_mode", "expert-index-range" },
        { "retained_expert_range", { { "first", 0 }, { "last", PROD_EXPERTS - 1 } } },
        { "total_group_count", PROD_EXPERTS * PROD_LAYERS },
        { "total_blob_bytes", total_blob_bytes },
        { "shard_count", PROD_LAYERS },
        { "content_hash", identity },
        { "shards", std::move(shards) },
    });
    return fixture;
}

// Sparse top-3-of-12 routing over 128 tokens, with experts
// [PROD_UNROUTED_FIRST, PROD_EXPERTS) resident but never routed. Every 16th
// token routes to three CONSECUTIVE expert ids so at least one arena sees
// n_used == 3 and every other arena that token touches pads to it at weight 0.
static pipe_expert_dispatch_req make_production_request() {
    pipe_expert_dispatch_req request;
    request.layer = PROD_LAYER;
    request.n_tokens = PROD_TOKENS;
    request.swiglu_clamp = 0.0f;
    request.activations.resize((size_t) PROD_TOKENS * (size_t) PROD_N_EMBD);
    for (size_t i = 0; i < request.activations.size(); ++i) {
        request.activations[i] = ((int) ((i * 19 + i / PROD_N_EMBD) % 37) - 18) * 0.011f;
    }

    std::vector<std::vector<float>> weights(
        PROD_EXPERTS, std::vector<float>(PROD_TOKENS, 0.0f));
    for (uint32_t t = 0; t < PROD_TOKENS; ++t) {
        std::vector<int> routed;
        if (t % 16 == 0) {
            const int base = (int) ((t / 16) * 4) % PROD_UNROUTED_FIRST;
            for (int k = 0; k < PROD_TOP_K; ++k) {
                routed.push_back((base + k) % PROD_UNROUTED_FIRST);
            }
        } else {
            for (int k = 0; k < PROD_TOP_K; ++k) {
                routed.push_back(
                    (int) ((t * (uint32_t) (7 * k + 1) + 5 * (uint32_t) k) % PROD_UNROUTED_FIRST));
            }
        }
        std::sort(routed.begin(), routed.end());
        routed.erase(std::unique(routed.begin(), routed.end()), routed.end());
        for (size_t k = 0; k < routed.size(); ++k) {
            // Non-zero, well away from zero: `weight == 0.0f` is the wire's
            // "this expert is not in this token's top-k" marker (weights are
            // f32 on the wire, see pipe_expert_assignment), and the grouped
            // path uses exactly that test to split routes from pads.
            weights[routed[k]][t] = 0.125f + 0.03125f * (float) ((t + k) % 9);
        }
    }
    for (int expert = 0; expert < PROD_EXPERTS; ++expert) {
        pipe_expert_assignment assignment;
        assignment.expert_id = expert;
        assignment.weights = weights[expert];
        request.assignments.push_back(std::move(assignment));
    }
    return request;
}

struct ProdRun {
    std::vector<float> whole;
    std::vector<float> chunked;
    uint64_t           hits = 0;
    uint64_t           fallbacks = 0;
    // Grouped graph builds charged to the 128-token bucket only: sampled after
    // the whole request and its repeats, BEFORE the 32-token chunks (which are
    // a different n_tokens and legitimately build their own graph).
    uint64_t           builds = 0;
    // Slot fingerprint of the last 128-token grouped request.
    uint64_t           placement = 0;
    // PAD-SLOT accounting, sampled from the worker after it shut down.
    uint32_t           hello_slots   = 0;   // what the spine was told it has
    uint64_t           pad_slots     = 0;   // reserved per arena
    uint64_t           arena_count   = 0;
    uint64_t           planned_slots = 0;   // before the reservation
    uint64_t           usable_slots  = 0;   // after it
    uint64_t           pad_bound     = 0;   // must stay 0
};

// One worker lifetime: an optional priming request (whose ASSIGNMENT ORDER
// decides the order pages are demanded, and therefore which slot -- and which
// arena -- each expert lands in), the whole 128-token request repeated
// `repeats` extra times, then the same request as four 32-token streamed
// chunks over the same connection.
static ProdRun run_production_prefill(
        const ProdFixture & fixture,
        const pipe_expert_dispatch_req & request,
        const std::string & device,
        bool grouped,
        const pipe_expert_dispatch_req * prime = nullptr,
        int repeats = 0) {
    const ScopedEnv arena_env("WP_EXPERT_ARENA_PREFILL", grouped ? "1" : "0");
    // Force several arenas per size class: arena_bytes is floor(cap/stride)*stride
    // and stride >= page_bytes (role-type alignment pads it), so a cap of 18
    // pages puts 17 or 18 of the PROD_EXPERTS*PROD_LAYERS slots in one buffer
    // -- four arenas. Each then reserves PROD_N_EXPERT_USED == 10 PAD slots and
    // still keeps six or so USABLE ones, comfortably more than the PROD_EXPERTS
    // pages layer PROD_LAYER actually demands. Production reaches the same
    // shape through the backend's own max-allocation cap (19 arenas on ROCm0,
    // 16 on ROCm1).
    const ScopedEnv arena_cap(
        "WP_EXPERT_ARENA_MAX_BYTES", std::to_string(fixture.page_bytes * 18));

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = device;
    options.listen_host       = "127.0.0.1";
    options.listen_port       = reserve_port();
    // Headroom over PROD_EXPERTS on purpose: plan_resources caps a class at one
    // slot per page but needs the byte budget to cover stride*pages (stride is
    // padded above page_bytes), so ask for a few more max-page equivalents than
    // there are pages. What matters downstream is that the USABLE slots after
    // the pad reservation still exceed PROD_EXPERTS: every requested
    // page must stay resident for the whole connection. If a page were evicted
    // and re-paged between the 128-token whole request and the 32-token chunks
    // it could land in a DIFFERENT arena, which reorders the across-arena fold
    // and would fail the byte-identity check for a reason that has nothing to
    // do with chunking.
    options.slots             = PROD_EXPERTS * PROD_LAYERS + 8;
    options.host_budget_bytes = 4 * fixture.page_bytes;
    options.once              = true;

    wp_expert_worker::test_reset_arena_prefill_counters();

    int server_result = -1;
    std::exception_ptr server_error;
    std::thread server([&]() {
        try {
            server_result = wp_expert_worker::run(options);
        } catch (...) {
            server_error = std::current_exception();
        }
    });

    ProdRun run;
    try {
        pipe_socket_ptr socket = connect_with_retry(options.listen_port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload) && type == PIPE_HELLO,
                "production-geometry worker did not send HELLO");
        pipe_expert_hello client = pipe_decode_expert_hello(payload.data(), payload.size());
        // What the SPINE is told the pool holds. Pads must not be in it.
        run.hello_slots     = client.n_slots;
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(*socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send production-geometry client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "production-geometry worker rejected HELLO");

        // PRIME. Sent before anything is resident, so the pool hands out slots
        // in THIS request's assignment order. Reversing that order moves every
        // expert to a different slot -- and, with 4-slot arenas, to a different
        // arena -- without changing the request under test at all.
        if (prime != nullptr) {
            payload = pipe_encode_expert_dispatch_req(*prime);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 718,
                                    payload.data(), payload.size()),
                    "failed to send production-geometry priming dispatch");
            require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                        type == PIPE_EXPERT_PARTIAL && seq_id == 718,
                    "production-geometry priming returned the wrong frame");
        }

        // WARM-UP. The counters must measure a steady-state chunk, not the
        // cold one: on the very first request the pages are still being read,
        // and if a slot is not yet bound the dispatcher splits the compute into
        // WP_EXPERT_COMPUTE_CHUNKS index ranges, each of which is correctly
        // refused by the whole-request guard in compute_batch and counted as a
        // fall-back. Production sees the same thing exactly once per layer.
        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 719,
                                payload.data(), payload.size()),
                "failed to send production-geometry warm-up dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_PARTIAL && seq_id == 719,
                "production-geometry warm-up returned the wrong frame");
        // Safe here: the worker is blocked reading the next frame.
        wp_expert_worker::test_reset_arena_prefill_counters();

        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ, 720,
                                payload.data(), payload.size()),
                "failed to send production-geometry dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_PARTIAL && seq_id == 720,
                "production-geometry dispatch returned the wrong frame");
        const pipe_expert_partial whole =
            pipe_decode_expert_partial(payload.data(), payload.size(), (uint32_t) PROD_N_EMBD);
        require(whole.n_tokens == PROD_TOKENS, "production-geometry token count mismatch");
        run.whole = whole.partial;

        // REPEATS. Same n_tokens, same routing, same placement: the grouped
        // graph cache must serve every one of these without a rebuild, and the
        // answer must not move by a single bit between identical requests.
        for (int repeat = 0; repeat < repeats; ++repeat) {
            payload = pipe_encode_expert_dispatch_req(request);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_REQ,
                                    (uint64_t) (730 + repeat),
                                    payload.data(), payload.size()),
                    "failed to send production-geometry repeat dispatch");
            require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                        type == PIPE_EXPERT_PARTIAL &&
                        seq_id == (uint64_t) (730 + repeat),
                    "production-geometry repeat returned the wrong frame");
            const pipe_expert_partial again = pipe_decode_expert_partial(
                payload.data(), payload.size(), (uint32_t) PROD_N_EMBD);
            require(again.partial.size() == run.whole.size(),
                    "production-geometry repeat shape mismatch");
            for (size_t i = 0; i < run.whole.size(); ++i) {
                if (std::memcmp(&run.whole[i], &again.partial[i], sizeof(float)) != 0) {
                    throw std::runtime_error(
                        "production-geometry repeat " + std::to_string(repeat) +
                        " changed the answer at " + std::to_string(i));
                }
            }
        }
        // Safe here: the worker is blocked reading the next frame.
        run.builds    = wp_expert_worker::test_arena_prefill_builds();
        run.placement = wp_expert_worker::test_arena_prefill_placement();

        run.chunked.assign((size_t) PROD_TOKENS * (size_t) PROD_N_EMBD, 0.0f);
        const uint32_t chunk_rows = PROD_TOKENS / PROD_CHUNKS;
        for (uint32_t chunk_index = 0; chunk_index < PROD_CHUNKS; ++chunk_index) {
            const uint32_t token_start = chunk_index * chunk_rows;
            const uint32_t token_end = chunk_index + 1 == PROD_CHUNKS
                ? PROD_TOKENS : token_start + chunk_rows;
            pipe_expert_dispatch_chunk chunk;
            chunk.chunk_index  = chunk_index;
            chunk.chunk_count  = PROD_CHUNKS;
            chunk.total_tokens = PROD_TOKENS;
            chunk.token_start  = token_start;
            chunk.token_end    = token_end;
            chunk.request.layer         = request.layer;
            chunk.request.n_tokens      = token_end - token_start;
            chunk.request.swiglu_clamp  = request.swiglu_clamp;
            for (const pipe_expert_assignment & assignment : request.assignments) {
                pipe_expert_assignment sliced;
                sliced.expert_id = assignment.expert_id;
                sliced.weights.assign(assignment.weights.begin() + token_start,
                                      assignment.weights.begin() + token_end);
                chunk.request.assignments.push_back(std::move(sliced));
            }
            chunk.request.activations.assign(
                request.activations.begin() + (size_t) token_start * PROD_N_EMBD,
                request.activations.begin() + (size_t) token_end * PROD_N_EMBD);

            payload = pipe_encode_expert_dispatch_chunk(chunk);
            require(pipe_send_frame(*socket, PIPE_EXPERT_DISPATCH_CHUNK, 721,
                                    payload.data(), payload.size()),
                    "failed to send production-geometry dispatch chunk");
            require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                        type == PIPE_EXPERT_PARTIAL_CHUNK && seq_id == 721,
                    "production-geometry chunk returned the wrong frame");
            const pipe_expert_partial_chunk partial = pipe_decode_expert_partial_chunk(
                payload.data(), payload.size(), (uint32_t) PROD_N_EMBD);
            require(partial.chunk_index == chunk_index &&
                        partial.token_start == token_start &&
                        partial.token_end == token_end,
                    "production-geometry partial range mismatch");
            std::copy(partial.partial.partial.begin(), partial.partial.partial.end(),
                      run.chunked.begin() + (size_t) token_start * PROD_N_EMBD);
        }
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
    require(server_result == 0, "production-geometry worker returned failure");
    run.hits      = wp_expert_worker::test_arena_prefill_hits();
    run.fallbacks = wp_expert_worker::test_arena_prefill_fallbacks();
    run.pad_slots     = wp_expert_worker::test_pool_pad_slots_per_arena();
    run.arena_count   = wp_expert_worker::test_pool_arena_count();
    run.planned_slots = wp_expert_worker::test_pool_planned_slots();
    run.usable_slots  = wp_expert_worker::test_pool_usable_slots();
    run.pad_bound     = wp_expert_worker::test_arena_prefill_pad_bound();
    return run;
}

// ---------------------------------------------------------------------------
// PAD SLOTS ARE INVISIBLE TO EVERYTHING BUT THE GROUPED GRAPH.
//
// Every arena reserves n_expert_used slots at its end so the grouped prefill
// path always has n_expert_used DISTINCT, never-written, never-DMA'd filler
// ids per arena to pad a token row with at route weight 0. They must not show
// up as capacity anywhere: not in the pool's slot vector, not in the HELLO
// slot count the spine plans residency against, and never as the home of a
// paged-in expert.
// ---------------------------------------------------------------------------
// `expect_pads` is n_expert_used on an arm that enabled grouped prefill for the
// device and 0 on one that did not: the reservation follows WP_EXPERT_ARENA_PREFILL
// (see plan_with_pad_slots), so a gather-only device keeps every slot pageable.
static void check_prod_pool_pads(const ProdRun & run, const std::string & label,
                                 uint64_t expect_pads) {
    if (run.pad_slots != expect_pads) {
        throw std::runtime_error(
            label + "arenas reserved " + std::to_string(run.pad_slots) +
            " pad slots, expected " + std::to_string(expect_pads));
    }
    if (run.arena_count < 2) {
        throw std::runtime_error(
            label + "only " + std::to_string(run.arena_count) +
            " arena(s): the multi-arena grouping this fixture exists to exercise "
            "is not happening");
    }
    // The reservation comes OUT of the plan: usable == planned - pads*arenas.
    const uint64_t reserved = run.pad_slots * run.arena_count;
    if (run.planned_slots < reserved ||
            run.usable_slots != run.planned_slots - reserved) {
        throw std::runtime_error(
            label + "pad reservation does not balance: planned=" +
            std::to_string(run.planned_slots) + " pads=" +
            std::to_string(run.pad_slots) + "x" + std::to_string(run.arena_count) +
            " usable=" + std::to_string(run.usable_slots));
    }
    // HELLO must advertise the USABLE slots only.
    if ((uint64_t) run.hello_slots != run.usable_slots) {
        throw std::runtime_error(
            label + "HELLO advertised " + std::to_string(run.hello_slots) +
            " slots but the pool carved " + std::to_string(run.usable_slots) +
            " (pads must not be advertised)");
    }
    // ... and there must be room for every requested page, or a page would be
    // evicted mid-connection and the byte-identity checks would be measuring
    // re-paging, not chunking.
    if (run.usable_slots <= (uint64_t) PROD_EXPERTS) {
        throw std::runtime_error(
            label + "only " + std::to_string(run.usable_slots) +
            " usable slots for " + std::to_string(PROD_EXPERTS) +
            " requested pages");
    }
    // No request may ever be handed a pad slot.
    if (run.pad_bound != 0) {
        throw std::runtime_error(
            label + "a paged-in expert landed in an arena's PAD region (" +
            std::to_string(run.pad_bound) + " times)");
    }
}

static void test_prefill_arena_grouped_production_geometry() {
    const char * backend_env = std::getenv("WP_WORKER_TEST_BACKEND");
    const std::string device =
        (backend_env != nullptr && backend_env[0] != '\0') ? backend_env : "CPU";
    const char * geometry_env = std::getenv("WP_WORKER_TEST_GEOMETRY");
    int only_geometry = -1;
    if (geometry_env != nullptr && geometry_env[0] != '\0') {
        only_geometry = (int) std::strtol(geometry_env, nullptr, 10);
        require(only_geometry >= 0 && only_geometry < 3,
                "WP_WORKER_TEST_GEOMETRY must be 0, 1 or 2");
    }

    const pipe_expert_dispatch_req request = make_production_request();

    for (int g = 0; g < 3; ++g) {
        if (only_geometry >= 0 && g != only_geometry) {
            continue;
        }
        for (const int width : PROD_WIDTHS) {
            const ProdGeometry & geometry = PROD_GEOMETRIES[g];
            const std::string label = std::string("[") + device + " variant " +
                std::to_string(g) + " " + geometry.name + " width " +
                std::to_string(width) + "] ";
            std::cout << "test-wp-expert-worker: production geometry " << label << std::endl;

            TempDir temp;
            const ProdFixture fixture =
                make_production_fixture(temp.path, geometry, width);

            const ProdRun grouped =
                run_production_prefill(fixture, request, device, /* grouped = */ true);
            const ProdRun gather =
                run_production_prefill(fixture, request, device, /* grouped = */ false);

            // (0) the pad reservation is real and invisible to the spine.
            check_prod_pool_pads(grouped, label, (uint64_t) PROD_N_EXPERT_USED);
            check_prod_pool_pads(gather, label, 0);

            // (c) the grouped arm actually took the grouped path, every time.
            if (grouped.hits == 0 || grouped.fallbacks != 0) {
                throw std::runtime_error(
                    label + "grouped prefill did not take the arena path: hits=" +
                    std::to_string(grouped.hits) + " fallbacks=" +
                    std::to_string(grouped.fallbacks));
            }
            require(gather.hits == 0 && gather.fallbacks == 0,
                    (label + "gather reference unexpectedly used the arena path").c_str());

            require(grouped.whole.size() == gather.whole.size() &&
                        grouped.whole.size() ==
                            (size_t) PROD_TOKENS * (size_t) PROD_N_EMBD,
                    (label + "result shape mismatch").c_str());

            // (a) every value finite, and grouped within 1e-3 relative of gather.
            double max_rel = 0.0;
            double max_abs_gather = 0.0;
            size_t worst = 0;
            for (size_t i = 0; i < gather.whole.size(); ++i) {
                if (!std::isfinite(grouped.whole[i])) {
                    throw std::runtime_error(
                        label + "grouped prefill produced a non-finite value at " +
                        std::to_string(i));
                }
                if (!std::isfinite(gather.whole[i])) {
                    throw std::runtime_error(
                        label + "gather prefill produced a non-finite value at " +
                        std::to_string(i));
                }
                max_abs_gather = std::max(max_abs_gather, std::fabs((double) gather.whole[i]));
            }
            for (size_t i = 0; i < gather.whole.size(); ++i) {
                const double diff =
                    std::fabs((double) grouped.whole[i] - (double) gather.whole[i]);
                const double rel = diff / (max_abs_gather > 0.0 ? max_abs_gather : 1.0);
                if (rel > max_rel) {
                    max_rel = rel;
                    worst = i;
                }
            }
            if (max_rel > 1e-3) {
                throw std::runtime_error(
                    label + "grouped prefill differs from gather: max_rel=" +
                    std::to_string(max_rel) + " at " + std::to_string(worst) +
                    " grouped=" + std::to_string(grouped.whole[worst]) +
                    " gather=" + std::to_string(gather.whole[worst]));
            }

            // (b) four 32-token chunks must be BIT-identical to the 128-token
            // whole under the grouped path. This is the whole point of
            // GGML_HINT_MUL_MAT_PIN plus the fixed-order fold: a (token, slot)
            // row's bits must not depend on how the ubatch was cut.
            require(grouped.chunked.size() == grouped.whole.size(),
                    (label + "chunked result shape mismatch").c_str());
            for (size_t i = 0; i < grouped.whole.size(); ++i) {
                if (std::memcmp(&grouped.whole[i], &grouped.chunked[i], sizeof(float)) != 0) {
                    throw std::runtime_error(
                        label + "grouped prefill changed when split into " +
                        std::to_string(PROD_CHUNKS) + " token chunks at " +
                        std::to_string(i) + ": whole=" +
                        std::to_string(grouped.whole[i]) + " chunked=" +
                        std::to_string(grouped.chunked[i]));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// PLACEMENT INDEPENDENCE.
//
// 2026-09-02, gates68: the same 1326-token prompt sent three times to a fresh
// spine produced three different output md5s on the grouped path (the gather
// path is stable). The grouped fold summed per ARENA GROUP in arena order and
// then over k in packing order, so which arena the pager happened to put a
// slot in decided the association of an FP sum -- and page-ins land in
// whatever slot is free, which differs between requests.
//
// This test computes the SAME request under two genuinely different slot
// placements (achieved by priming one worker with the assignments in reverse
// order, which is the order the pool hands out slots) and demands the two
// partials be byte-identical. The placement fingerprints are asserted to
// DIFFER first, so the test cannot pass by both runs getting the same layout.
// It also asserts the graph cache stabilises: repeats of one shape must not
// keep rebuilding the graph (n_arena_build was 986 for 1013 hits on ROCm0).
// ---------------------------------------------------------------------------
static void test_prefill_arena_grouped_placement_independent() {
    const char * backend_env = std::getenv("WP_WORKER_TEST_BACKEND");
    const std::string device =
        (backend_env != nullptr && backend_env[0] != '\0') ? backend_env : "CPU";
    // One geometry, the narrow width: this test is about ordering, not about
    // quantisation coverage, and it pays for two extra worker lifetimes.
    const ProdGeometry & geometry = PROD_GEOMETRIES[0];
    const int width = PROD_WIDTHS[1];
    const std::string label =
        std::string("[") + device + " placement " + geometry.name + "] ";
    std::cout << "test-wp-expert-worker: grouped prefill placement independence "
              << label << std::endl;

    const pipe_expert_dispatch_req request = make_production_request();
    pipe_expert_dispatch_req prime = request;
    std::reverse(prime.assignments.begin(), prime.assignments.end());

    TempDir temp;
    const ProdFixture fixture = make_production_fixture(temp.path, geometry, width);

    const int repeats = 3;
    const ProdRun forward = run_production_prefill(
        fixture, request, device, /* grouped = */ true,
        /* prime = */ nullptr, repeats);
    const ProdRun reversed = run_production_prefill(
        fixture, request, device, /* grouped = */ true,
        /* prime = */ &prime, /* repeats = */ 0);

    check_prod_pool_pads(forward, label, (uint64_t) PROD_N_EXPERT_USED);
    check_prod_pool_pads(reversed, label, (uint64_t) PROD_N_EXPERT_USED);

    if (forward.hits == 0 || forward.fallbacks != 0 ||
            reversed.hits == 0 || reversed.fallbacks != 0) {
        throw std::runtime_error(
            label + "a placement arm did not take the grouped path: forward hits=" +
            std::to_string(forward.hits) + "/fallbacks=" +
            std::to_string(forward.fallbacks) + " reversed hits=" +
            std::to_string(reversed.hits) + "/fallbacks=" +
            std::to_string(reversed.fallbacks));
    }

    // NON-VACUITY: the two runs must really have placed the experts differently.
    if (forward.placement == reversed.placement) {
        throw std::runtime_error(
            label + "both runs got the SAME slot placement (fingerprint " +
            std::to_string(forward.placement) +
            "); the placement-independence assertion below would be vacuous");
    }

    require(forward.whole.size() == reversed.whole.size(),
            (label + "placement arms disagree on shape").c_str());
    for (size_t i = 0; i < forward.whole.size(); ++i) {
        if (std::memcmp(&forward.whole[i], &reversed.whole[i], sizeof(float)) != 0) {
            throw std::runtime_error(
                label + "grouped prefill changed with slot placement at " +
                std::to_string(i) + ": forward=" +
                std::to_string(forward.whole[i]) + " reversed=" +
                std::to_string(reversed.whole[i]));
        }
    }

    std::cout << "test-wp-expert-worker: " << label
              << "forward hits=" << forward.hits
              << " fallbacks=" << forward.fallbacks
              << " builds=" << forward.builds
              << " | reversed hits=" << reversed.hits
              << " fallbacks=" << reversed.fallbacks
              << " | pads=" << forward.pad_slots
              << " arenas=" << forward.arena_count
              << " planned=" << forward.planned_slots
              << " usable=" << forward.usable_slots
              << " hello_slots=" << forward.hello_slots
              << " pad_bound=" << forward.pad_bound
              << " placements " << forward.placement << " vs " << reversed.placement
              << std::endl;

    // GRAPH CACHE STABILITY: warm-up already built the 128-token bucket, so the
    // measured request plus its repeats may add at most one build.
    if (forward.builds > 1) {
        throw std::runtime_error(
            label + "grouped graph cache did not stabilise: " +
            std::to_string(repeats + 1) + " identical 128-token requests caused " +
            std::to_string(forward.builds) + " graph builds");
    }
}

// ---------------------------------------------------------------------------
// WP_EXPERT_OWNER_POLICY

// Verbatim copy of the pre-WP_EXPERT_OWNER_POLICY Worker::static_owner_for_page
// body. This is the SNAPSHOT of the old behaviour: the shipped proportional
// path must keep agreeing with it exactly, or a rerun of an old config stops
// reproducing its output.
static size_t reference_proportional_owner(
        int expert, int expert_first, int expert_last,
        const std::vector<int> & device_slots) {
    const int first = expert_first;
    const int last  = expert_last;
    const uint64_t count = last >= first ? (uint64_t) (last - first) + 1 : 0;
    const uint64_t ordinal = expert >= first ? (uint64_t) (expert - first) : 0;
    uint64_t total = 0;
    for (const int slots : device_slots) {
        total += (uint64_t) slots;
    }
    const uint64_t point = count > 0 ? ordinal * total / count : 0;
    uint64_t begin = 0;
    for (size_t i = 0; i < device_slots.size(); ++i) {
        begin += (uint64_t) device_slots[i];
        if (point < begin) {
            return i;
        }
    }
    return device_slots.size() - 1;
}

static void test_owner_policy_proportional_unchanged() {
    const std::vector<std::vector<int>> slot_sets = {
        { 1, 1 },
        { 320, 96 },            // production-ish: R9700 + Thunderbolt RX 6900 XT
        { 320, 96, 48 },        // ... plus the CPU tier
        { 7, 3, 11, 1 },
        { 5 },
    };
    const std::vector<std::pair<int, int>> ranges = {
        { 0, 127 }, { 0, 0 }, { 64, 191 }, { 10, 9 },
    };
    for (const std::vector<int> & slots : slot_sets) {
        for (const std::pair<int, int> & range : ranges) {
            for (int expert = range.first - 3; expert <= range.second + 3; ++expert) {
                const size_t expected = reference_proportional_owner(
                    expert, range.first, range.second, slots);
                const size_t actual = wp_expert_worker::proportional_owner_for_expert(
                    expert, range.first, range.second, slots);
                require(expected == actual,
                        "proportional owner map changed against its pre-policy snapshot");
            }
        }
    }
    // A hardcoded band for the production shape, so a future refactor of BOTH
    // implementations at once still trips.
    const std::vector<int> production = { 320, 96, 48 };
    const std::vector<size_t> expected_first_16 = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    };
    for (int expert = 0; expert < 16; ++expert) {
        require(wp_expert_worker::proportional_owner_for_expert(
                    expert, 0, 127, production) == expected_first_16[(size_t) expert],
                "proportional owner map moved the first expert band");
    }
    require(wp_expert_worker::proportional_owner_for_expert(88, 0, 127, production) == 0 &&
                wp_expert_worker::proportional_owner_for_expert(89, 0, 127, production) == 1 &&
                wp_expert_worker::proportional_owner_for_expert(114, 0, 127, production) == 1 &&
                wp_expert_worker::proportional_owner_for_expert(115, 0, 127, production) == 2 &&
                wp_expert_worker::proportional_owner_for_expert(127, 0, 127, production) == 2,
            "proportional owner map moved a device band boundary");

    require(wp_expert_worker::parse_owner_policy(nullptr) ==
                    wp_expert_worker::owner_policy::proportional &&
                wp_expert_worker::parse_owner_policy("") ==
                    wp_expert_worker::owner_policy::proportional &&
                wp_expert_worker::parse_owner_policy("proportional") ==
                    wp_expert_worker::owner_policy::proportional &&
                wp_expert_worker::parse_owner_policy(" hot ") ==
                    wp_expert_worker::owner_policy::hot &&
                wp_expert_worker::parse_owner_policy("nonsense") ==
                    wp_expert_worker::owner_policy::proportional,
            "WP_EXPERT_OWNER_POLICY parse is not the documented default-proportional");

    const std::vector<std::string> devices = { "ROCm0", "ROCm1", "CPU" };
    require(wp_expert_worker::parse_owner_priority(nullptr, devices) ==
                    std::vector<size_t>({ 0, 1, 2 }),
            "unset WP_EXPERT_OWNER_PRIORITY did not default to device-list order");
    require(wp_expert_worker::parse_owner_priority(" ROCm1 , CPU ", devices) ==
                    std::vector<size_t>({ 1, 2, 0 }),
            "WP_EXPERT_OWNER_PRIORITY did not append the unnamed devices in list order");
    require(wp_expert_worker::parse_owner_priority("Vulkan9,ROCm1,ROCm1", devices) ==
                    std::vector<size_t>({ 1, 0, 2 }),
            "WP_EXPERT_OWNER_PRIORITY did not drop unknown and duplicate names");
}

// The synthetic rig below stands in for the production one: three devices in
// priority order (the slow-link GPU first), two size classes, and a class the
// top-priority device cannot hold at all.
//
//   capacity[class][device]   dev0   dev1   dev2
//     class 0                   2      3      0
//     class 1                   0      1      2
//
// 12 pages: ids 0-5 in class 0, ids 6-11 in class 1, ranked in id order.
static wp_expert_worker::HotOwnerInput make_hot_owner_input() {
    wp_expert_worker::HotOwnerInput input;
    input.n_devices = 3;
    input.page_class.assign(12, 0);
    for (size_t id = 6; id < 12; ++id) {
        input.page_class[id] = 1;
    }
    input.page_static_owner.assign(12, 2);
    input.capacity = { { 2, 3, 0 }, { 0, 1, 2 } };
    input.priority = { 0, 1, 2 };
    input.ranked.resize(12);
    for (size_t id = 0; id < 12; ++id) {
        input.ranked[id] = id;
    }
    return input;
}

static void test_owner_policy_hot_packs_priority_device() {
    const wp_expert_worker::HotOwnerInput input = make_hot_owner_input();
    const wp_expert_worker::HotOwnerPlan plan = wp_expert_worker::plan_hot_owner_map(input);

    // (a) The top-priority device owns exactly its usable capacity per class,
    //     taken off the HEAD of the ranked list; the next device gets the next
    //     slice; the pages past both capacities fall back.
    const std::vector<size_t> expected_owner = {
        0, 0, 1, 1, 1, 1,   // class 0: dev0 x2, dev1 x3, then one OVERFLOW page
        1, 2, 2,            // class 1: dev0 has no capacity -> dev1 x1, dev2 x2
        1, 2, 2,            // class 1 overflow, spread over {dev1:1, dev2:2}
    };
    const std::vector<char> expected_ranked = {
        1, 1, 1, 1, 1, 0,
        1, 1, 1,
        0, 0, 0,
    };
    // Page 5 is the one the 2026-09-02 production log was about: class 0 is
    // full everywhere, so it overflows. It used to land back on dev0 -- the
    // fully-resident priority device, which cannot hold it -- because the old
    // fallback weighted by TOTAL capacity. It now goes to dev1.
    const std::vector<char> expected_overflow = {
        0, 0, 0, 0, 0, 1,
        0, 0, 0,
        1, 1, 1,
    };
    require(plan.owner == expected_owner,
            "WP_EXPERT_OWNER_POLICY=hot did not pack the priority device from the "
            "head of the ranked list");
    require(plan.from_ranked == expected_ranked,
            "WP_EXPERT_OWNER_POLICY=hot mislabelled ranked vs fallback ownership");
    require(plan.from_overflow == expected_overflow,
            "WP_EXPERT_OWNER_POLICY=hot mislabelled overflow vs residual-fill ownership");

    // Restated as the property the policy exists for: nothing is over-committed.
    std::vector<std::vector<size_t>> ranked_owned(2, std::vector<size_t>(3, 0));
    for (size_t id = 0; id < plan.owner.size(); ++id) {
        if (plan.from_ranked[id]) {
            ++ranked_owned[input.page_class[id]][plan.owner[id]];
        }
    }
    require(ranked_owned[0][0] == input.capacity[0][0] &&
                ranked_owned[0][1] == input.capacity[0][1] &&
                ranked_owned[1][1] == input.capacity[1][1] &&
                ranked_owned[1][2] == input.capacity[1][2],
            "WP_EXPERT_OWNER_POLICY=hot did not fill each device to its usable capacity");

    // (d) A page whose size class has zero capacity on the priority device
    //     skips to the next device -- never lands somewhere it cannot be
    //     resident.
    require(ranked_owned[1][0] == 0,
            "WP_EXPERT_OWNER_POLICY=hot placed a page on a device with no capacity "
            "in that page's size class");
    for (size_t id = 0; id < plan.owner.size(); ++id) {
        require(input.capacity[input.page_class[id]][plan.owner[id]] != 0,
                "WP_EXPERT_OWNER_POLICY=hot left a page owned by a device that cannot "
                "hold its size class");
    }

    // The priority order, not the device-list order, decides who gets the head
    // of the list.
    wp_expert_worker::HotOwnerInput dev1_first = input;
    dev1_first.priority = { 1, 0, 2 };
    const wp_expert_worker::HotOwnerPlan dev1_plan =
        wp_expert_worker::plan_hot_owner_map(dev1_first);
    require(dev1_plan.owner[0] == 1 && dev1_plan.owner[1] == 1 &&
                dev1_plan.owner[2] == 1 && dev1_plan.owner[3] == 0 &&
                dev1_plan.owner[4] == 0,
            "WP_EXPERT_OWNER_POLICY=hot ignored WP_EXPERT_OWNER_PRIORITY order");

    // ... and a priority order that puts the zero-capacity device first for
    // class 0 must still produce a legal map.
    wp_expert_worker::HotOwnerInput dev2_first = input;
    dev2_first.priority = { 2, 0, 1 };
    const wp_expert_worker::HotOwnerPlan dev2_plan =
        wp_expert_worker::plan_hot_owner_map(dev2_first);
    require(dev2_plan.owner[0] == 0 && dev2_plan.owner[1] == 0 &&
                dev2_plan.owner[2] == 1,
            "WP_EXPERT_OWNER_POLICY=hot did not skip the zero-capacity priority device");
    require(dev2_plan.owner[6] == 2 && dev2_plan.owner[7] == 2 &&
                dev2_plan.owner[8] == 1,
            "WP_EXPERT_OWNER_POLICY=hot did not honour priority within a size class");
}

static void test_owner_policy_hot_is_deterministic() {
    // (b) Same inputs, two independent constructions, identical maps. The
    //     policy is a pure function; a page that moved between two ROCm
    //     devices between launches would change the run's output md5.
    const wp_expert_worker::HotOwnerPlan first =
        wp_expert_worker::plan_hot_owner_map(make_hot_owner_input());
    const wp_expert_worker::HotOwnerPlan second =
        wp_expert_worker::plan_hot_owner_map(make_hot_owner_input());
    require(first.owner == second.owner && first.from_ranked == second.from_ranked,
            "WP_EXPERT_OWNER_POLICY=hot is not reproducible across constructions");

    // Unranked pages are placed too, deterministically, and pages the policy
    // does not manage keep their proportional owner.
    wp_expert_worker::HotOwnerInput unlisted = make_hot_owner_input();
    unlisted.ranked.clear();
    unlisted.page_class[11] = wp_expert_worker::HOT_OWNER_NO_CLASS;
    unlisted.page_static_owner[11] = 2;
    const wp_expert_worker::HotOwnerPlan a = wp_expert_worker::plan_hot_owner_map(unlisted);
    const wp_expert_worker::HotOwnerPlan b = wp_expert_worker::plan_hot_owner_map(unlisted);
    require(a.owner == b.owner, "hot fallback placement is not deterministic");
    require(a.owner[11] == 2,
            "hot policy moved a page it does not manage off its proportional owner");
    for (const char ranked : a.from_ranked) {
        require(ranked == 0, "hot policy claimed ranked ownership with an empty hot list");
    }
    for (const char overflowed : a.from_overflow) {
        require(overflowed == 0,
                "hot policy called a residual fill an overflow while capacity was free");
    }
    // With no hot list every page takes the fallback: class 0 spreads over
    // {2,3,0} and class 1 over {0,1,2}, proportional to capacity and stable in
    // page-id order.
    const std::vector<size_t> expected = { 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2 };
    require(a.owner == expected,
            "hot fallback did not spread unranked pages proportionally to remaining "
            "capacity in page-id order");
}

// ---------------------------------------------------------------------------
// WP_EXPERT_OWNER_OVERFLOW -- who absorbs the pages that fit NOWHERE.
//
// Production shape, 2026-09-02: 24576 pages, ~16600 usable slots, priority
// ROCm1,ROCm0,CPU. ~8000 pages must overflow, and the old "spread over TOTAL
// capacity" fallback handed 3197 of them to ROCm1 -- the 2.78 GB/s Thunderbolt
// device the policy exists to keep fully resident and page-in free.
//
// The rig below is that in miniature: one size class, three devices in
// priority order, capacity {10, 6, 2} = 18 usable against 100 pages, every
// page ranked. So 18 pages fit and 82 must overflow.
static const std::vector<std::string> & overflow_device_names() {
    static const std::vector<std::string> names = { "ROCm1", "ROCm0", "CPU" };
    return names;
}

static wp_expert_worker::HotOwnerInput make_overflow_input() {
    wp_expert_worker::HotOwnerInput input;
    input.n_devices = 3;
    input.page_class.assign(100, 0);
    input.page_static_owner.assign(100, 0);   // proportional map: everyone on ROCm1
    input.capacity = { { 10, 6, 2 } };
    input.priority = { 0, 1, 2 };
    input.ranked.resize(100);
    for (size_t id = 0; id < 100; ++id) {
        input.ranked[id] = id;
    }
    return input;
}

// Per-device owned counts, and per-device overflow counts, from a plan.
static void tally_owner_plan(const wp_expert_worker::HotOwnerPlan & plan,
                             size_t n_devices,
                             std::vector<size_t> & owned,
                             std::vector<size_t> & overflowed) {
    owned.assign(n_devices, 0);
    overflowed.assign(n_devices, 0);
    for (size_t id = 0; id < plan.owner.size(); ++id) {
        ++owned[plan.owner[id]];
        if (plan.from_overflow[id]) {
            ++overflowed[plan.owner[id]];
        }
    }
}

static void test_owner_overflow_parse() {
    const std::vector<std::string> & devices = overflow_device_names();

    require(!wp_expert_worker::parse_owner_overflow(nullptr, devices).from_env &&
                !wp_expert_worker::parse_owner_overflow("", devices).from_env &&
                !wp_expert_worker::parse_owner_overflow("   ", devices).from_env,
            "unset WP_EXPERT_OWNER_OVERFLOW did not fall back to the default list");

    const wp_expert_worker::HotOwnerOverflow weighted =
        wp_expert_worker::parse_owner_overflow(" ROCm0 : 4 , CPU:1 ", devices);
    require(weighted.from_env &&
                weighted.devices == std::vector<size_t>({ 1, 2 }) &&
                weighted.weights == std::vector<uint64_t>({ 4, 1 }),
            "WP_EXPERT_OWNER_OVERFLOW did not parse names, weights and whitespace");

    // A bare name means "weight me by my usable capacity" (weight 0), and so
    // does a weight that is not an integer. Duplicates keep the first entry.
    const wp_expert_worker::HotOwnerOverflow mixed =
        wp_expert_worker::parse_owner_overflow("CPU,ROCm0:oops,CPU:9", devices);
    require(mixed.from_env &&
                mixed.devices == std::vector<size_t>({ 2, 1 }) &&
                mixed.weights == std::vector<uint64_t>({ 0, 0 }),
            "WP_EXPERT_OWNER_OVERFLOW mishandled bare names, bad weights or duplicates");

    // Unknown names are dropped; a value that is ONLY unknown names is a typo,
    // not a list, and must fall back to the default rather than to nobody.
    const wp_expert_worker::HotOwnerOverflow partly =
        wp_expert_worker::parse_owner_overflow("Vulkan9,CPU:3", devices);
    require(partly.from_env && partly.devices == std::vector<size_t>({ 2 }) &&
                partly.weights == std::vector<uint64_t>({ 3 }),
            "WP_EXPERT_OWNER_OVERFLOW did not drop an unknown device name");
    require(!wp_expert_worker::parse_owner_overflow("Vulkan9,Nope", devices).from_env,
            "WP_EXPERT_OWNER_OVERFLOW of only-unknown names did not fall back to default");
}

static void test_owner_overflow_spares_priority_device() {
    const wp_expert_worker::HotOwnerInput input = make_overflow_input();
    const wp_expert_worker::HotOwnerPlan plan = wp_expert_worker::plan_hot_owner_map(input);

    std::vector<size_t> owned, overflowed;
    tally_owner_plan(plan, 3, owned, overflowed);

    // (a) THE POINT OF THE POLICY: the priority device ends EXACTLY full --
    //     its usable capacity, no more -- and every page it owns came off the
    //     head of the ranked list. Zero overflow, so zero page-ins.
    require(owned[0] == input.capacity[0][0],
            "overflow policy did not leave the priority device exactly full");
    require(overflowed[0] == 0,
            "overflow policy handed the fully-resident priority device overflow pages");
    for (size_t id = 0; id < plan.owner.size(); ++id) {
        require(plan.owner[id] != 0 || plan.from_ranked[id],
                "a page reached the priority device other than off the ranked list");
    }
    for (size_t id = 0; id < 10; ++id) {
        require(plan.owner[id] == 0 && plan.from_ranked[id],
                "the priority device did not take the HEAD of the ranked list");
    }

    // Ranked pass fills dev1 then dev2; 100 - 18 = 82 pages overflow.
    require(overflowed[1] + overflowed[2] == 82,
            "overflow policy did not overflow every page past total capacity");

    // (b) The overflow is spread over the DEFAULT list -- every device except
    //     the priority one -- in proportion to usable capacity in the class,
    //     6:2. The integer bands put page k on dev1 while k*8/82 < 6, i.e.
    //     k < 61.5, so 62 pages to ROCm0 and 20 to CPU.
    require(overflowed[1] == 62 && overflowed[2] == 20,
            "overflow was not spread over the default devices in capacity proportion");
    require(owned[1] == 6 + 62 && owned[2] == 2 + 20,
            "overflow tally does not add up to the owner map");

    // Ascending page id, contiguous bands: no interleaving, no hash order.
    for (size_t id = 18; id < 100; ++id) {
        require(plan.from_overflow[id] == 1, "a page past capacity was not marked overflow");
        require(plan.owner[id] == (id < 80 ? (size_t) 1 : (size_t) 2),
                "overflow bands are not contiguous in ascending page id");
    }

    // (d) INVARIANT: the priority device never owns more of a class than it
    //     has usable slots for.
    require(owned[0] <= input.capacity[0][0],
            "INVARIANT: priority device over-committed in its size class");
}

static void test_owner_overflow_explicit_weights() {
    const std::vector<std::string> & devices = overflow_device_names();

    // (b) Explicit list with weights: ROCm0 takes 4 for every 1 the CPU takes,
    //     regardless of their usable capacity (6:2 would be 3:1).
    wp_expert_worker::HotOwnerInput input = make_overflow_input();
    input.overflow = wp_expert_worker::parse_owner_overflow("ROCm0:4,CPU:1", devices);
    const wp_expert_worker::HotOwnerPlan plan = wp_expert_worker::plan_hot_owner_map(input);

    std::vector<size_t> owned, overflowed;
    tally_owner_plan(plan, 3, owned, overflowed);
    // k*5/82 < 4  <=>  k < 65.6, so 66 pages to ROCm0 and 16 to the CPU.
    require(overflowed[0] == 0 && overflowed[1] == 66 && overflowed[2] == 16,
            "WP_EXPERT_OWNER_OVERFLOW weights were not honoured");
    require(owned[0] == 10, "an explicit overflow list disturbed the ranked pass");

    // A single-device list sends the whole overflow there.
    wp_expert_worker::HotOwnerInput cpu_only = make_overflow_input();
    cpu_only.overflow = wp_expert_worker::parse_owner_overflow("CPU", devices);
    const wp_expert_worker::HotOwnerPlan cpu_plan =
        wp_expert_worker::plan_hot_owner_map(cpu_only);
    tally_owner_plan(cpu_plan, 3, owned, overflowed);
    require(overflowed[2] == 82 && overflowed[0] == 0 && overflowed[1] == 0,
            "a single-device WP_EXPERT_OWNER_OVERFLOW did not absorb the whole overflow");

    // The priority device is dropped from a list that names other devices...
    wp_expert_worker::HotOwnerInput with_priority = make_overflow_input();
    with_priority.overflow =
        wp_expert_worker::parse_owner_overflow("ROCm1:9,ROCm0:1", devices);
    const wp_expert_worker::HotOwnerPlan dropped =
        wp_expert_worker::plan_hot_owner_map(with_priority);
    tally_owner_plan(dropped, 3, owned, overflowed);
    require(overflowed[0] == 0 && overflowed[1] == 82,
            "the priority device was not dropped from a multi-device overflow list");
    require(owned[0] == 10, "INVARIANT: priority device over-committed via overflow list");

    // ... but is honoured when it is the ONLY device listed. That is an
    // operator asking for it explicitly, and it is the one case where the
    // fully-resident invariant is allowed to break.
    wp_expert_worker::HotOwnerInput only_priority = make_overflow_input();
    only_priority.overflow = wp_expert_worker::parse_owner_overflow("ROCm1", devices);
    const wp_expert_worker::HotOwnerPlan on_priority =
        wp_expert_worker::plan_hot_owner_map(only_priority);
    tally_owner_plan(on_priority, 3, owned, overflowed);
    require(overflowed[0] == 82,
            "a lone-priority WP_EXPERT_OWNER_OVERFLOW was not honoured literally");

    // A list whose devices have NO capacity in the class still avoids the
    // priority device: the weights fall back to total class capacity minus
    // the priority device, so everything lands on ROCm0.
    wp_expert_worker::HotOwnerInput no_capacity = make_overflow_input();
    no_capacity.capacity = { { 10, 6, 0 } };
    no_capacity.overflow = wp_expert_worker::parse_owner_overflow("CPU:5", devices);
    const wp_expert_worker::HotOwnerPlan salvaged =
        wp_expert_worker::plan_hot_owner_map(no_capacity);
    tally_owner_plan(salvaged, 3, owned, overflowed);
    require(overflowed[0] == 0 && overflowed[1] == 100 - 16 && overflowed[2] == 0,
            "overflow onto a zero-capacity device did not fall back off the priority "
            "device");

    // Only the priority device can hold the class at all -> better an
    // over-committed resident device than a page with nowhere to live.
    wp_expert_worker::HotOwnerInput priority_only_capacity = make_overflow_input();
    priority_only_capacity.capacity = { { 10, 0, 0 } };
    const wp_expert_worker::HotOwnerPlan last_resort =
        wp_expert_worker::plan_hot_owner_map(priority_only_capacity);
    tally_owner_plan(last_resort, 3, owned, overflowed);
    require(overflowed[0] == 90 && owned[0] == 100,
            "a class only the priority device can hold was not left on it");
}

static void test_owner_overflow_is_deterministic() {
    // (c) Two independent constructions, identical maps -- default list and
    //     explicit list alike. A page that moved between two ROCm devices
    //     between launches would change the run's output md5.
    const wp_expert_worker::HotOwnerPlan a =
        wp_expert_worker::plan_hot_owner_map(make_overflow_input());
    const wp_expert_worker::HotOwnerPlan b =
        wp_expert_worker::plan_hot_owner_map(make_overflow_input());
    require(a.owner == b.owner && a.from_ranked == b.from_ranked &&
                a.from_overflow == b.from_overflow,
            "the overflow spread is not reproducible across constructions");

    wp_expert_worker::HotOwnerInput weighted = make_overflow_input();
    weighted.overflow = wp_expert_worker::parse_owner_overflow(
        "ROCm0:4,CPU:1", overflow_device_names());
    wp_expert_worker::HotOwnerInput weighted_again = make_overflow_input();
    weighted_again.overflow = wp_expert_worker::parse_owner_overflow(
        "ROCm0:4,CPU:1", overflow_device_names());
    require(wp_expert_worker::plan_hot_owner_map(weighted).owner ==
                wp_expert_worker::plan_hot_owner_map(weighted_again).owner,
            "a weighted overflow spread is not reproducible across constructions");
}

// ---------------------------------------------------------------------------
// WP_EXPERT_OWNER_POLICY=hot AGAINST A REAL TWO-DEVICE WORKER (2026-09-02).
//
// The three tests above pin plan_hot_owner_map(), which is a pure function of a
// capacity table handed to it. They cannot see the defect that made the policy
// a no-op in production: the CAPACITY TABLE ITSELF was zero.
//
// Placement size classes are keyed on ExpertPage::size (the O_DIRECT-padded
// blob page), while a ResourcePlan's SlotClass is keyed on
// ExpertPage::device_size -- resource_pages() feeds plan_resources the device
// size. layout_sliced_pages() reserves quantized row slack on any role whose
// ne0 is not a multiple of 512 and aligns each member to the backend's buffer
// alignment, so the two numbers differ on exactly the geometries that carry
// such a role. The old table matched the two by ==, found nothing, and left
// usable capacity at 0; plan_hot_owner_map then skipped every page of that
// class and the whole map fell through to the proportional fallback.
//
// This fixture reproduces that geometry EXACTLY. Production's dominant class is
// 2,150,400 blob bytes; PROD_GEOMETRIES[0] at PROD_WIDTHS[0] is gate/up q4_K
// [2560 x 448] + down q5_1 [448 x 2560] = 2,150,400 bytes, and down's ne0 of
// 448 is not a multiple of 512, so its device layout is larger than its blob.
// On 4ab9a0fbb every assertion below reports 0.
//
// inspect_resources() builds the whole Worker -- both DeviceWorkers, the
// placement tables and the hot policy -- and returns without opening a socket,
// so this costs a construction, not a serving run.
static void test_owner_policy_hot_capacity_on_two_devices() {
    TempDir temp;
    const ProdFixture fixture =
        make_production_fixture(temp.path, PROD_GEOMETRIES[0], PROD_WIDTHS[0]);

    // Same arena split the grouped-prefill production test uses: several arenas
    // per size class, each reserving PROD_N_EXPERT_USED pad slots, so `usable`
    // is strictly below the planned count and the two cannot be confused.
    const ScopedEnv arena_cap(
        "WP_EXPERT_ARENA_MAX_BYTES", std::to_string(fixture.page_bytes * 18));
    const ScopedEnv grouped("WP_EXPERT_ARENA_PREFILL", "1");
    // Pin in seed mode, as production does: it seeds LFU heat instead of
    // reading 64 pages off disk during a construction-only test.
    const ScopedEnv pin_mode("WP_EXPERT_PIN_MODE", "seed");

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.devices           = { "CPU", "CPU" };
    options.device_slots      = { PROD_EXPERTS * PROD_LAYERS + 8,
                                  PROD_EXPERTS * PROD_LAYERS + 8 };
    options.host_budget_bytes = 4 * fixture.page_bytes;

    // ---- PASS 1: default (proportional) policy. Only the capacity table is
    //      under test here, and initialize_placement_policy() builds it
    //      whatever the ownership policy is.
    wp_expert_worker::test_reset_placement_report();
    const wp_expert_worker::ResourcePlan plan =
        wp_expert_worker::inspect_resources(options);
    const wp_expert_worker::PlacementReport report =
        wp_expert_worker::test_placement_report();

    require(report.devices.size() == 2,
            "two-device placement report did not describe two devices");
    require(!report.class_bytes.empty(),
            "two-device placement report has no size classes");

    for (size_t c = 0; c < report.class_bytes.size(); ++c) {
        for (size_t d = 0; d < report.devices.size(); ++d) {
            if (report.planned[c][d] == 0) {
                throw std::runtime_error(
                    "placement class[" + std::to_string(c) + "] (" +
                    std::to_string(report.class_bytes[c]) + " bytes) resolved to NO "
                    "slot class on device " + std::to_string(d) +
                    ": the class -> slot-class match is broken");
            }
            if (report.usable[c][d] == 0) {
                throw std::runtime_error(
                    "placement class[" + std::to_string(c) + "] (" +
                    std::to_string(report.class_bytes[c]) + " bytes) reports usable=0 "
                    "on device " + std::to_string(d) + " with planned=" +
                    std::to_string(report.planned[c][d]) +
                    ": ownership would silently degrade to the proportional map");
            }
            if (report.usable[c][d] > report.planned[c][d]) {
                throw std::runtime_error(
                    "placement class[" + std::to_string(c) +
                    "] reports more usable slots than the plan asked for");
            }
        }
    }

    // The per-class table must ACCOUNT FOR THE WHOLE POOL of device 0 --
    // inspect_resources() returns that device's plan. planned sums to what the
    // plan asked for before the pad reservation, usable to what it carved.
    uint64_t planned_total = 0;
    uint64_t usable_total  = 0;
    for (size_t c = 0; c < report.class_bytes.size(); ++c) {
        planned_total += report.planned[c][0];
        usable_total  += report.usable[c][0];
    }
    require(planned_total == (uint64_t) plan.planned_slot_count,
            "placement capacity table does not account for the planned pool");
    require(usable_total == (uint64_t) plan.slot_count,
            "placement usable-capacity table does not account for the carved pool");
    require(usable_total < planned_total,
            "the pad reservation did not come out of the pool: this fixture is no "
            "longer exercising the usable-vs-planned distinction");

    // ---- PASS 2: WP_EXPERT_OWNER_POLICY=hot, with a hot list sized to fill
    //      the priority device (device 0, since WP_EXPERT_OWNER_PRIORITY is
    //      unset and the default is device-list order) to exactly its usable
    //      capacity in every class and no further. The fallback then has zero
    //      remaining capacity on device 0 and spills the rest onto device 1,
    //      so device 0 must come out owning exactly what it can hold.
    require(report.class_bytes.size() == 1,
            "this fixture is expected to have one size class; the hot list below "
            "would need to be built per class otherwise");
    const uint64_t hot_pages = report.usable[0][0];

    const fs::path pin_path = temp.path / "hot-pins.txt";
    {
        std::ofstream pins(pin_path);
        require(pins.good(), "failed to write the hot-set pin file");
        uint64_t written = 0;
        uint64_t count   = 1000000;
        for (int layer = PROD_LAYER;
                layer < PROD_LAYER + PROD_LAYERS && written < hot_pages; ++layer) {
            for (int expert = 0;
                    expert < PROD_EXPERTS && written < hot_pages; ++expert) {
                pins << layer << ' ' << expert << "  # " << count-- << '\n';
                ++written;
            }
        }
        require(written == hot_pages,
                "the fixture has fewer pages than the priority device has slots");
    }

    const ScopedEnv policy("WP_EXPERT_OWNER_POLICY", "hot");
    const ScopedEnv pin_file("WP_EXPERT_PIN_FILE", pin_path.string());

    wp_expert_worker::test_reset_placement_report();
    wp_expert_worker::inspect_resources(options);
    const wp_expert_worker::PlacementReport hot =
        wp_expert_worker::test_placement_report();

    require(hot.hot, "WP_EXPERT_OWNER_POLICY=hot did not run on a two-device worker");
    require(!hot.priority.empty(), "hot policy reported no priority order");
    const size_t top = hot.priority.front();
    require(top == 0, "the default priority order is not the device-list order");

    for (size_t c = 0; c < hot.class_bytes.size(); ++c) {
        if (hot.owned[c][top] != hot.usable[c][top]) {
            throw std::runtime_error(
                "WP_EXPERT_OWNER_POLICY=hot left the priority device holding " +
                std::to_string(hot.owned[c][top]) + " pages of class[" +
                std::to_string(c) + "] against " + std::to_string(hot.usable[c][top]) +
                " usable slots: the ranked walk did not fill it exactly");
        }
    }
    require(hot.fully_resident.at(top) != 0,
            "WP_EXPERT_OWNER_POLICY=hot did not leave the priority device fully "
            "resident");
}

// ---------------------------------------------------------------------------
// WP_EXPERT_PIN_CLASS_PCT capping the dominant class (2026-09-03).
//
// expert_pin_class_index() is called at both of its call sites with a page's
// BLOB size (ExpertPage::size) while SlotClass::size is keyed on the DEVICE
// size (ExpertPage::device_size) -- the identical confusion e805cdc1b fixed
// for the owner-policy capacity table (see
// DeviceWorker::slot_class_index_for_page()). Exact equality never matched
// for PROD_GEOMETRIES[0]'s dominant class (2,150,400 blob vs 2,150,464 device
// bytes), so both call sites resolved it to "no class" -- past the end of
// pin_class_caps/_pinned/_skipped -- and WP_EXPERT_PIN_CLASS_PCT silently
// never capped it: the class 0 log line always read pinned=0 skipped=0 no
// matter how many class-0 pages were pinned.
//
// This drives the multi-device call site (Worker::load_pin_file(), used
// whenever devices_.size() > 1) with WP_EXPERT_PIN_CLASS_PCT set low enough
// that the fixture's single dominant class -- all 64 pages of
// PROD_GEOMETRIES[0]/PROD_WIDTHS[0] are one class, see
// test_owner_policy_hot_capacity_on_two_devices above -- has to spill some
// pages into `skipped`. On the pre-fix matching rule every assertion below
// fails: pinned and skipped both stay 0 because class_id never resolves.
static void test_pin_class_cap_resolves_by_device_size() {
    TempDir temp;
    const ProdFixture fixture =
        make_production_fixture(temp.path, PROD_GEOMETRIES[0], PROD_WIDTHS[0]);

    const ScopedEnv arena_cap(
        "WP_EXPERT_ARENA_MAX_BYTES", std::to_string(fixture.page_bytes * 18));
    const ScopedEnv grouped("WP_EXPERT_ARENA_PREFILL", "1");
    const ScopedEnv pin_mode("WP_EXPERT_PIN_MODE", "seed");
    // Cap every class at 10% of its planned slots -- low enough that pinning
    // every one of the fixture's 64 pages (all one class, split across two
    // devices) has to skip some on both devices.
    const ScopedEnv pin_pct("WP_EXPERT_PIN_CLASS_PCT", "10");

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.devices           = { "CPU", "CPU" };
    options.device_slots      = { PROD_EXPERTS * PROD_LAYERS + 8,
                                  PROD_EXPERTS * PROD_LAYERS + 8 };
    options.host_budget_bytes = 4 * fixture.page_bytes;

    const fs::path pin_path = temp.path / "class-cap-pins.txt";
    {
        std::ofstream pins(pin_path);
        require(pins.good(), "failed to write the class-cap pin file");
        for (int layer = PROD_LAYER; layer < PROD_LAYER + PROD_LAYERS; ++layer) {
            for (int expert = 0; expert < PROD_EXPERTS; ++expert) {
                pins << layer << ' ' << expert << "  # 1\n";
            }
        }
    }
    const ScopedEnv pin_file("WP_EXPERT_PIN_FILE", pin_path.string());

    wp_expert_worker::test_reset_pin_class_report();
    wp_expert_worker::inspect_resources(options);
    const wp_expert_worker::PinClassReport report =
        wp_expert_worker::test_pin_class_report();

    require(report.devices.size() == 2,
            "pin class report did not describe two devices");

    for (size_t d = 0; d < report.devices.size(); ++d) {
        if (report.class_bytes[d].empty()) {
            throw std::runtime_error(
                "pin class report has no slot classes for device " + std::to_string(d));
        }
        for (size_t c = 0; c < report.class_bytes[d].size(); ++c) {
            if (report.class_slots[d][c] == 0) {
                continue;
            }
            const uint64_t attributed = report.class_pinned[d][c] + report.class_skipped[d][c];
            if (attributed == 0) {
                throw std::runtime_error(
                    "no page was attributed to slot class " + std::to_string(c) +
                    " on device " + std::to_string(d) + ": expert_pin_class_index still "
                    "resolves the dominant class's blob size to no slot class");
            }
            if (report.class_pinned[d][c] != report.class_cap[d][c]) {
                throw std::runtime_error(
                    "class " + std::to_string(c) + " on device " + std::to_string(d) +
                    " did not pin exactly its cap (pinned=" +
                    std::to_string(report.class_pinned[d][c]) + " cap=" +
                    std::to_string(report.class_cap[d][c]) + ")");
            }
            if (report.class_skipped[d][c] == 0) {
                throw std::runtime_error(
                    "class " + std::to_string(c) + " on device " + std::to_string(d) +
                    " never spilled into skipped: WP_EXPERT_PIN_CLASS_PCT is not "
                    "actually capping the dominant class");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// assignment_groups() bucketing (2026-09-03).
//
// WP_EXPERT_OWNER_POLICY=hot's owner map interleaves devices in expert-id
// order (unlike the proportional map's contiguous ranges), and the OLD
// assignment_groups() cut a new group every time the owner changed between
// consecutive assignments -- so one interleaved layer fragmented into ~11x
// the sub-requests (measured: 22.6 -> 2.0 experts/sub-request), collapsing
// prefill 69 -> 14 t/s. The fix buckets by owning device instead of by
// contiguous run, so group count is bounded by the number of distinct
// devices regardless of how the owner map is ordered.
//
// test_bucket_assignment_groups() exercises the exact bucketing algorithm
// Worker::assignment_groups() uses (see wp-expert-worker.cpp,
// bucket_indices_by_owner), fed a synthetic owner-per-assignment sequence, so
// this does not need a live multi-device Worker.
static void test_assignment_groups_bucket_by_device() {
    // Interleaved owner map: experts alternate device 0 / 1 / 0 / 1 ..., the
    // exact shape WP_EXPERT_OWNER_POLICY=hot produces and that fragmented the
    // old contiguous-run grouping.
    const std::vector<size_t> owner_for_index = { 0, 1, 0, 1, 0, 1, 0, 1, 0 };

    const wp_expert_worker::AssignmentGroupsTestReport report =
        wp_expert_worker::test_bucket_assignment_groups(owner_for_index);

    const size_t n_distinct_owners = 2;  // devices {0, 1} both appear above
    require(report.devices.size() == n_distinct_owners,
            "interleaved owner map did not collapse to one group per device");
    require(report.indices.size() == report.devices.size(),
            "assignment groups report device/indices count mismatch");

    // Groups must be ordered by device index.
    for (size_t g = 1; g < report.devices.size(); ++g) {
        require(report.devices[g - 1] < report.devices[g],
                "assignment groups are not ordered by device index");
    }

    // Each group's indices must be strictly increasing (original assignment
    // order preserved within a device), and every index in a group must
    // actually map to that group's device in owner_for_index.
    std::vector<char> covered(owner_for_index.size(), 0);
    size_t total_indices = 0;
    for (size_t g = 0; g < report.devices.size(); ++g) {
        const size_t device = report.devices[g];
        const std::vector<size_t> & indices = report.indices[g];
        require(!indices.empty(), "assignment group has no indices");
        for (size_t k = 0; k < indices.size(); ++k) {
            if (k > 0) {
                require(indices[k - 1] < indices[k],
                        "assignment group indices are not strictly increasing");
            }
            const size_t index = indices[k];
            require(index < owner_for_index.size(),
                    "assignment group index is out of range");
            require(owner_for_index[index] == device,
                    "assignment group index does not map to its group's device");
            require(covered[index] == 0,
                    "assignment index appears in more than one group");
            covered[index] = 1;
            ++total_indices;
        }
    }
    require(total_indices == owner_for_index.size(),
            "union of assignment group indices does not cover every assignment");
    for (size_t i = 0; i < covered.size(); ++i) {
        require(covered[i] != 0,
                "an assignment index was dropped by the bucketing");
    }

    // Single-owner request: the existing single-device fast path (one group,
    // covering every assignment) must still hold.
    const std::vector<size_t> single_owner(5, /* device = */ 2);
    const wp_expert_worker::AssignmentGroupsTestReport single =
        wp_expert_worker::test_bucket_assignment_groups(single_owner);
    require(single.devices.size() == 1,
            "a single-owner request produced more than one group");
    require(single.devices[0] == 2, "single-owner group reported the wrong device");
    require(single.indices[0].size() == single_owner.size(),
            "single-owner group did not cover every assignment");
    for (size_t i = 0; i < single_owner.size(); ++i) {
        require(single.indices[0][i] == i,
                "single-owner group did not preserve original assignment order");
    }

    // Empty request: no groups (begin_split_dispatch is what turns this into
    // the {device 0, no indices} fallback group).
    const wp_expert_worker::AssignmentGroupsTestReport empty =
        wp_expert_worker::test_bucket_assignment_groups({});
    require(empty.devices.empty() && empty.indices.empty(),
            "an empty owner sequence produced a group");

    // A device index that recurs non-adjacently (e.g. 1,0,1,0,1) still
    // collapses to exactly one group per device, with device 0's two
    // occurrences kept in original relative order.
    const std::vector<size_t> owner_alt = { 1, 0, 1, 0, 1 };
    const wp_expert_worker::AssignmentGroupsTestReport alt =
        wp_expert_worker::test_bucket_assignment_groups(owner_alt);
    require(alt.devices.size() == 2, "non-adjacent recurring owners did not collapse");
    require(alt.devices[0] == 0 && alt.devices[1] == 1,
            "groups were not ordered by device index");
    require((alt.indices[0] == std::vector<size_t>{ 1, 3 }),
            "device 0's indices were not kept in original relative order");
    require((alt.indices[1] == std::vector<size_t>{ 0, 2, 4 }),
            "device 1's indices were not kept in original relative order");
}

// ---------------------------------------------------------------------------
// WP_EXPERT_FUSE_GATE_UP reason classifier (2026-09-03).
//
// On qwen38 the slicer writes members as up (mask 1), gate (mask 2), down
// (mask 4). layout_sliced_pages() follows blob offset, so up sits at 0 and
// gate follows it. fuse_gate_up_ok() wants up at gate + ggml_row_size*ne1.
// Gate and up are the same type (q4_K or q5_K) and the same slice shape;
// swiglu_clamp is 0. The classifier must report adjacency for that layout,
// not type/shape/clamp, and WP_EXPERT_FUSE_GATE_UP_LAYOUT=1 reorders to
// gate then up then down without changing device_size on this geometry.
static void test_fuse_gate_up_reason_classifier() {
    using wp_expert_worker::FuseGateUpCheck;
    using wp_expert_worker::FuseGateUpReason;
    using wp_expert_worker::classify_fuse_gate_up;
    using wp_expert_worker::format_fuse_gate_up_reason;
    using wp_expert_worker::fuse_gate_up_layout_names;
    using wp_expert_worker::plan_device_member_layout;

    const int64_t ne0 = 2560;
    const int64_t ne1 = 448;
    const int gate_up_type = (int) GGML_TYPE_Q4_K;
    const uint64_t gate_bytes =
        (uint64_t) ggml_row_size(GGML_TYPE_Q4_K, ne0) * (uint64_t) ne1;
    require(gate_bytes == 645120, "qwen38 gate/up q4_K bytes changed");

    FuseGateUpCheck ok;
    ok.swiglu_clamp = 0.0f;
    ok.use_gather = false;
    ok.gather_allowed = false;
    ok.gate_type = gate_up_type;
    ok.up_type = gate_up_type;
    ok.gate_ne0 = ne0;
    ok.gate_ne1 = ne1;
    ok.up_ne0 = ne0;
    ok.up_ne1 = ne1;
    ok.gate_device_offset = 0;
    ok.up_device_offset = gate_bytes;
    require(classify_fuse_gate_up(ok).reason == FuseGateUpReason::Ok,
            "adjacent same-type gate||up was not classified ok");
    require(format_fuse_gate_up_reason(classify_fuse_gate_up(ok)) == "ok",
            "ok reason string");

    FuseGateUpCheck clamp = ok;
    clamp.swiglu_clamp = 10.0f;
    require(classify_fuse_gate_up(clamp).reason == FuseGateUpReason::Clamp,
            "nonzero swiglu_clamp was not classified clamp");
    require(format_fuse_gate_up_reason(classify_fuse_gate_up(clamp)) == "clamp",
            "clamp reason string");

    FuseGateUpCheck gather = ok;
    gather.use_gather = true;
    gather.gather_allowed = false;
    require(classify_fuse_gate_up(gather).reason == FuseGateUpReason::Gather,
            "gather without the gather fuse flag was not classified gather");
    require(format_fuse_gate_up_reason(classify_fuse_gate_up(gather)) == "gather",
            "gather reason string");
    gather.gather_allowed = true;
    require(classify_fuse_gate_up(gather).reason == FuseGateUpReason::Ok,
            "gather with the gather fuse flag was not classified ok");

    FuseGateUpCheck gather_type = gather;
    gather_type.gather_allowed = false;
    gather_type.up_type = (int) GGML_TYPE_Q5_1;
    require(classify_fuse_gate_up(gather_type).reason == FuseGateUpReason::Type,
            "type must be reported before gather");

    FuseGateUpCheck type = ok;
    type.up_type = (int) GGML_TYPE_Q5_1;
    require(classify_fuse_gate_up(type).reason == FuseGateUpReason::Type,
            "mismatched gate/up ggml_type was not classified type");
    require(format_fuse_gate_up_reason(classify_fuse_gate_up(type)) == "type",
            "type reason string");

    FuseGateUpCheck shape = ok;
    shape.up_ne1 = 192;
    require(classify_fuse_gate_up(shape).reason == FuseGateUpReason::Shape,
            "mismatched gate/up ne1 was not classified shape");
    require(format_fuse_gate_up_reason(classify_fuse_gate_up(shape)) == "shape",
            "shape reason string");

    FuseGateUpCheck adj = ok;
    adj.gate_device_offset = gate_bytes;
    adj.up_device_offset = 0;
    const wp_expert_worker::FuseGateUpDiag adj_diag = classify_fuse_gate_up(adj);
    require(adj_diag.reason == FuseGateUpReason::Adjacency,
            "up-then-gate blob order was not classified adjacency");
    require(format_fuse_gate_up_reason(adj_diag) ==
                "adjacency:go=645120,uo=0,gate_bytes=645120",
            "adjacency reason string does not match the required format");

    FuseGateUpCheck both = adj;
    both.swiglu_clamp = 1.0f;
    require(classify_fuse_gate_up(both).reason == FuseGateUpReason::Clamp,
            "clamp must win over adjacency");

    FuseGateUpCheck adj_gather = adj;
    adj_gather.use_gather = true;
    require(classify_fuse_gate_up(adj_gather).reason == FuseGateUpReason::Adjacency,
            "adjacency must be reported before gather");

    const std::vector<std::string> blob_order = { "up", "gate", "down" };
    const std::vector<std::string> fused_order =
        fuse_gate_up_layout_names(blob_order);
    require(fused_order.size() == 3 && fused_order[0] == "gate" &&
                fused_order[1] == "up" && fused_order[2] == "down",
            "fuse layout did not put gate immediately before up");
    require(fuse_gate_up_layout_names({ "down" }) ==
                std::vector<std::string>{ "down" },
            "fuse layout changed a page with no gate/up pair");

    const uint64_t down_bytes =
        (uint64_t) ggml_row_size(GGML_TYPE_Q5_1, 448) * (uint64_t) 2560;
    const uint64_t down_slack =
        (uint64_t) ggml_row_size(GGML_TYPE_Q5_1, 512 - (448 % 512));
    const uint64_t down_alloc = down_bytes + down_slack;
    require(down_bytes == 860160 && down_slack == 48,
            "qwen38 down q5_1 bytes/slack changed");
    const auto blob_layout = plan_device_member_layout(
        { gate_bytes, gate_bytes, down_alloc }, 64);
    const auto fused_layout = plan_device_member_layout(
        { gate_bytes, gate_bytes, down_alloc }, 64);
    const uint64_t blob_device =
        blob_layout.back().offset + blob_layout.back().size;
    const uint64_t fused_device =
        fused_layout.back().offset + fused_layout.back().size;
    require(blob_device == fused_device && fused_device == 2150448,
            "fuse layout changed qwen38 448-wide device_size");
    require(fused_layout[1].offset == fused_layout[0].offset + gate_bytes,
            "fuse layout did not make up adjacent after gate");
}

static int32_t float_bits(float x) {
    int32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    return bits;
}

static bool within_one_ulp(float a, float b) {
    if (a == b) {
        return true;
    }
    const int32_t ia = float_bits(a);
    const int32_t ib = float_bits(b);
    if ((ia ^ ib) < 0) {
        return false;
    }
    return std::abs(ia - ib) <= 1;
}

// One expert, n_tokens=1, CPU backend. Per-expert path vs LAYOUT=1 fused path.
// Compares VALUES (1 ulp). The synthetic blob is up, gate, down.
static std::vector<float> cpu_one_expert_partial(
        const Fixture & fixture, bool fuse_layout) {
    const ScopedEnv fuse("WP_EXPERT_FUSE_GATE_UP", fuse_layout ? "1" : "0");
    const ScopedEnv layout("WP_EXPERT_FUSE_GATE_UP_LAYOUT", fuse_layout ? "1" : "0");
    const ScopedEnv gather("WP_EXPERT_FUSE_GATE_UP_GATHER", fuse_layout ? "1" : "0");

    wp_expert_worker::Options options;
    options.shard_manifest    = fixture.manifest;
    options.descriptor        = fixture.descriptor;
    options.device            = "CPU";
    options.listen_host       = "127.0.0.1";
    options.listen_port       = reserve_port();
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

    std::vector<float> partial;
    try {
        pipe_socket_ptr socket = connect_with_retry(options.listen_port);
        pipe_frame_type type;
        uint64_t seq_id = 0;
        std::vector<uint8_t> payload;
        require(pipe_recv_frame(*socket, type, seq_id, payload) && type == PIPE_HELLO,
                "fuse-layout CPU worker did not send HELLO");
        pipe_expert_hello client =
            pipe_decode_expert_hello(payload.data(), payload.size());
        client.role         = PIPE_EXPERT_ROLE_CLIENT;
        client.expert_first = -1;
        client.expert_last  = -1;
        client.n_slots      = 0;
        client.layers.clear();
        payload = pipe_encode_expert_hello(client);
        require(pipe_send_frame(
                    *socket, PIPE_HELLO, 0, payload.data(), payload.size()),
                "failed to send fuse-layout client HELLO");
        require(pipe_recv_frame(*socket, type, seq_id, payload) &&
                    type == PIPE_EXPERT_HELLO_ACK &&
                    pipe_decode_expert_hello_ack(payload.data(), payload.size()).accepted,
                "fuse-layout CPU worker rejected HELLO");

        std::vector<float> input((size_t) N_EMBD);
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = ((int) (i % 13) - 6) * 0.07f;
        }
        pipe_expert_dispatch_req request;
        request.layer = LAYER;
        request.n_tokens = 1;
        request.activations = input;
        request.assignments = { { 0, { 0.5f } } };
        payload = pipe_encode_expert_dispatch_req(request);
        require(pipe_send_frame(
                    *socket, PIPE_EXPERT_DISPATCH_REQ, 70,
                    payload.data(), payload.size()),
                "failed to send fuse-layout dispatch");
        require(pipe_recv_frame(*socket, type, seq_id, payload),
                "failed to receive fuse-layout partial");
        if (type == PIPE_ERROR) {
            const pipe_error error =
                pipe_decode_error(payload.data(), payload.size());
            throw std::runtime_error(
                "fuse-layout dispatch failed: " + error.msg);
        }
        require(type == PIPE_EXPERT_PARTIAL && seq_id == 70,
                "fuse-layout worker did not return a partial");
        const pipe_expert_partial response =
            pipe_decode_expert_partial(payload.data(), payload.size(), N_EMBD);
        require(response.n_tokens == 1 && response.partial.size() == (size_t) N_EMBD,
                "fuse-layout partial shape mismatch");
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
    require(server_result == 0, "fuse-layout CPU worker did not exit cleanly");
    require(!partial.empty(), "fuse-layout CPU worker returned an empty partial");
    return partial;
}

static void test_fuse_gate_up_layout_matches_split() {
    TempDir temp;
    const Fixture fixture = make_fixture(temp.path);
    const std::vector<float> split = cpu_one_expert_partial(fixture, false);
    const std::vector<float> fused = cpu_one_expert_partial(fixture, true);
    require(split.size() == fused.size() && split.size() == (size_t) N_EMBD,
            "split and fused-layout partials have different shapes");
    bool any_nonzero = false;
    for (size_t i = 0; i < split.size(); ++i) {
        any_nonzero = any_nonzero || split[i] != 0.0f || fused[i] != 0.0f;
        if (!within_one_ulp(split[i], fused[i])) {
            throw std::runtime_error(
                "fused-layout CPU partial differs from the per-expert path at " +
                std::to_string(i) + ": fused=" + std::to_string(fused[i]) +
                " split=" + std::to_string(split[i]));
        }
    }
    require(any_nonzero, "split and fused-layout partials were both zero");
}

int main() {
    try {
        require(setenv("WP_EXPERT_MM_PIN", "1", 1) == 0,
                "failed to enable expert MUL_MAT pin");
        require(setenv("WP_EXPERT_MM_PIN_MIN_TOKENS", "64", 1) == 0,
                "failed to set expert MUL_MAT pin threshold");
        require(setenv("WP_EXPERT_GATHER", "1", 1) == 0 &&
                    setenv("WP_EXPERT_GATHER_MIN_TOKENS", "2", 1) == 0,
                "failed to enable expert gather");
        test_owner_policy_proportional_unchanged();
        test_owner_policy_hot_packs_priority_device();
        test_owner_policy_hot_is_deterministic();
        test_owner_overflow_parse();
        test_owner_overflow_spares_priority_device();
        test_owner_overflow_explicit_weights();
        test_owner_overflow_is_deterministic();
        test_owner_policy_hot_capacity_on_two_devices();
        test_pin_class_cap_resolves_by_device_size();
        test_assignment_groups_bucket_by_device();
        test_decode_prefill_compute_profile();
        test_batch_mmid_ids();
        test_batch_mmid_association();
        test_arena_prefill_device_policy();
        test_prefill_arena_grouped_production_geometry();
        test_prefill_arena_grouped_placement_independent();
        test_prefill_arena_grouped_chunk_byte_identical();
        test_prefill_arena_grouped_matches_gather();
        test_prefill_mul_mat_pin_chunk_byte_identical();
        test_scatter_compact_rows_matches_get_rows_back();
        test_scatter_add_compact_rows_accumulates();
        test_partial_last_column_round_trip();
        test_q5_1_down_proj_prefill_last_column();
        test_slice_device_member_layout();
        test_fuse_gate_up_reason_classifier();
        test_fuse_gate_up_layout_matches_split();
        test_glm_size_class_plan();
        test_fixture_arena_stride_alignment();
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
