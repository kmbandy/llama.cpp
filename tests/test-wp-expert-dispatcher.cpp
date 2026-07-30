#include "ggml.h"
#include "pipe-expert-dispatcher.h"
#include "pipe-protocol.h"
#include "pipe-transport.h"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

namespace {

static constexpr int          N_EMBD         = 32;
static constexpr int          N_FF_EXP       = 32;
static constexpr int          N_EXPERT       = 4;
static constexpr int          LAYER          = 3;
static constexpr int          N_TOKENS       = 2;
static constexpr const char * MODEL_IDENTITY = "sha256:synthetic-shared-model";

void require(bool condition, const std::string & message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct temp_dir {
    fs::path path;

    temp_dir() {
        std::string       pattern = (fs::temp_directory_path() / "wp-expert-dispatcher-XXXXXX").string();
        std::vector<char> writable(pattern.begin(), pattern.end());
        writable.push_back('\0');
        char * result = mkdtemp(writable.data());
        if (result == nullptr) {
            throw std::runtime_error("mkdtemp failed");
        }
        path = result;
    }

    ~temp_dir() {
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
            float     value   = 0.006f * pattern;
            if (row == col) {
                value += 0.08f + expert * 0.01f + role * 0.005f;
            }
            values[(size_t) row * N_EMBD + col] = value;
        }
    }
    return values;
}

struct fixture {
    fs::path descriptor;
    fs::path manifest;
};

using weight_map = std::map<std::pair<int, std::string>, std::vector<float>>;

fixture make_fixture(const fs::path &    dir,
                     const std::string & name,
                     int                 expert_first,
                     int                 expert_last,
                     weight_map &        all_weights) {
    fixture result;
    result.descriptor      = dir / (name + ".expert-descriptor.json");
    result.manifest        = dir / (name + "-experts-manifest.json");
    const fs::path sidecar = dir / (name + ".wpi.json");
    const fs::path blob    = dir / (name + ".wpb");

    constexpr uint64_t role_bytes = (uint64_t) N_EMBD * N_FF_EXP * sizeof(float);
    constexpr uint64_t page_bytes = role_bytes * 3;
    static_assert(page_bytes % 4096 == 0, "synthetic expert page must be O_DIRECT aligned");

    json          groups = json::array();
    std::ofstream blob_output(blob, std::ios::binary);
    if (!blob_output) {
        throw std::runtime_error("failed to create " + blob.string());
    }
    uint64_t offset = 0;
    for (int expert = expert_first; expert <= expert_last; ++expert) {
        json members = json::array();
        for (const auto & role : { std::make_pair(std::string("up"), 1), std::make_pair(std::string("gate"), 2),
                                   std::make_pair(std::string("down"), 4) }) {
            const int          role_index       = role.first == "up" ? 0 : (role.first == "gate" ? 1 : 2);
            std::vector<float> matrix           = make_matrix(expert, role_index);
            all_weights[{ expert, role.first }] = matrix;
            blob_output.write(reinterpret_cast<const char *>(matrix.data()),
                              (std::streamsize) (matrix.size() * sizeof(float)));
            members.push_back({
                { "role_mask",          role.second                                                          },
                { "size",               role_bytes                                                           },
                { "offset",             offset                                                               },
                { "catalog_name",       "blk.3.ffn_" + role.first + "." + std::to_string(expert) + ".weight" },
                { "source_tensor_name", "synthetic." + role.first                                            },
                { "source_file_idx",    0                                                                    },
                { "source_file_offset", offset                                                               },
            });
            offset += role_bytes;
        }
        groups.push_back({
            { "block_idx",    LAYER              },
            { "expert_idx",   expert             },
            { "member_count", 3                  },
            { "members",      std::move(members) },
        });
    }
    blob_output.close();

    const json identity = {
        { "algorithm", "sha256"                 },
        { "value",     "synthetic-shared-model" },
    };
    const json role_shape = { N_EMBD, N_FF_EXP };
    const auto role_desc  = [&](const char * role) {
        return json{
            { "ggml_type",          (int) GGML_TYPE_F32              },
            { "ggml_type_name",     ggml_type_name(GGML_TYPE_F32)    },
            { "shape",              role_shape                       },
            { "bytes_per_expert",   role_bytes                       },
            { "source_tensor_name", std::string("synthetic.") + role },
        };
    };
    write_json(result.descriptor,
               {
                   { "format",                  "llama.cpp.weight-pager.expert-descriptor"             },
                   { "version",                 1                                                      },
                   { "source_model",
                    {
                         { "input_model", "synthetic.gguf" },
                         { "model_files", { "synthetic.gguf" } },
                         { "architecture", "synthetic" },
                         { "name", "synthetic" },
                     }                                                                                 },
                   { "shard_manifest_identity", identity                                               },
                   { "retained_expert_range",   { { "first", expert_first }, { "last", expert_last } } },
                   { "hparams",
                    {
                         { "n_layer", 4 },
                         { "n_embd", N_EMBD },
                         { "n_ff_exp", N_FF_EXP },
                         { "n_expert", N_EXPERT },
                         { "n_expert_used", N_EXPERT },
                         { "activation", "silu" },
                     }                                                                                 },
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
                     }                                                                                 },
    });

    const uint64_t group_count = (uint64_t) (expert_last - expert_first + 1);
    write_json(sidecar, {
                            { "format",       "llama.cpp.weight-pager.expert-shard-index" },
                            { "version",      1                                           },
                            { "blob_file",    blob.filename().string()                    },
                            { "shard_index",  0                                           },
                            { "shard_count",  1                                           },
                            { "layer_first",  LAYER                                       },
                            { "layer_last",   LAYER                                       },
                            { "group_count",  group_count                                 },
                            { "blob_bytes",   offset                                      },
                            { "content_hash", identity                                    },
                            { "model_files",  { "synthetic.gguf" }                        },
                            { "groups",       std::move(groups)                           },
    });

    write_json(result.manifest, {
                                    { "format",                "llama.cpp.weight-pager.expert-shard-manifest"         },
                                    { "version",               1                                                      },
                                    { "input_model",           "synthetic.gguf"                                       },
                                    { "model_files",           { "synthetic.gguf" }                                   },
                                    { "sharding_mode",         "expert-index-range"                                   },
                                    { "retained_expert_range", { { "first", expert_first }, { "last", expert_last } } },
                                    { "total_group_count",     group_count                                            },
                                    { "total_blob_bytes",      offset                                                 },
                                    { "shard_count",           1                                                      },
                                    { "content_hash",          identity                                               },
                                    { "shards",
                                     {
                                          {
                                              { "blob_file", blob.filename().string() },
                                              { "index_file", sidecar.filename().string() },
                                              { "shard_index", 0 },
                                              { "layer_first", LAYER },
                                              { "layer_last", LAYER },
                                              { "group_count", group_count },
                                              { "blob_bytes", offset },
                                              { "content_hash", identity },
                                          },
                                      }                                                                               },
    });
    return result;
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

struct worker_process {
    pid_t pid = -1;

    worker_process(const fixture & data, int port, const fs::path & log_path) {
        pid = fork();
        if (pid < 0) {
            throw std::runtime_error("fork failed");
        }
        if (pid == 0) {
            const int log_fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
            if (log_fd >= 0) {
                dup2(log_fd, STDOUT_FILENO);
                dup2(log_fd, STDERR_FILENO);
                close(log_fd);
            }
            const std::string listen = "127.0.0.1:" + std::to_string(port);
            execl(WP_EXPERT_WORKER_BIN, WP_EXPERT_WORKER_BIN, "--shard-manifest", data.manifest.c_str(), "--descriptor",
                  data.descriptor.c_str(), "--device", "CPU", "--listen", listen.c_str(), "--slots", "2",
                  (char *) nullptr);
            _exit(127);
        }
    }

    worker_process(const worker_process &)             = delete;
    worker_process & operator=(const worker_process &) = delete;

    worker_process(worker_process && other) noexcept : pid(other.pid) { other.pid = -1; }

    ~worker_process() {
        if (pid <= 0) {
            return;
        }
        kill(pid, SIGTERM);
        int status = 0;
        while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
    }
};

void wait_for_listener(int port, pid_t pid) {
    std::ostringstream port_text;
    port_text << std::uppercase << std::hex << std::setw(4) << std::setfill('0') << port;
    const std::string suffix = ":" + port_text.str();
    for (int attempt = 0; attempt < 1000; ++attempt) {
        int status = 0;
        if (waitpid(pid, &status, WNOHANG) == pid) {
            throw std::runtime_error("expert worker exited before listening");
        }
        std::ifstream tcp("/proc/net/tcp");
        std::string   line;
        while (std::getline(tcp, line)) {
            std::istringstream fields(line);
            std::string        slot;
            std::string        local;
            std::string        remote;
            std::string        state;
            fields >> slot >> local >> remote >> state;
            if (state == "0A" && local.size() >= suffix.size() &&
                local.compare(local.size() - suffix.size(), suffix.size(), suffix) == 0) {
                return;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    throw std::runtime_error("timed out waiting for expert worker listener");
}

std::vector<float> reference(const weight_map &                          weights,
                             const std::vector<float> &                  activation,
                             const std::vector<pipe_expert_assignment> & assignments) {
    std::vector<float> result((size_t) N_TOKENS * N_EMBD, 0.0f);
    std::vector<float> gate(N_FF_EXP);
    std::vector<float> up(N_FF_EXP);
    std::vector<float> hidden(N_FF_EXP);
    std::vector<float> down(N_EMBD);
    for (const pipe_expert_assignment & assignment : assignments) {
        const std::vector<float> & gate_weight = weights.at({ assignment.expert_id, "gate" });
        const std::vector<float> & up_weight   = weights.at({ assignment.expert_id, "up" });
        const std::vector<float> & down_weight = weights.at({ assignment.expert_id, "down" });
        for (int token = 0; token < N_TOKENS; ++token) {
            const float * input = activation.data() + (size_t) token * N_EMBD;
            for (int row = 0; row < N_FF_EXP; ++row) {
                gate[row] = 0.0f;
                up[row]   = 0.0f;
                for (int col = 0; col < N_EMBD; ++col) {
                    gate[row] += gate_weight[(size_t) row * N_EMBD + col] * input[col];
                    up[row] += up_weight[(size_t) row * N_EMBD + col] * input[col];
                }
                hidden[row] = gate[row] / (1.0f + std::exp(-gate[row])) * up[row];
            }
            for (int row = 0; row < N_EMBD; ++row) {
                down[row] = 0.0f;
                for (int col = 0; col < N_FF_EXP; ++col) {
                    down[row] += down_weight[(size_t) row * N_FF_EXP + col] * hidden[col];
                }
                result[(size_t) token * N_EMBD + row] += assignment.weights[(size_t) token] * down[row];
            }
        }
    }
    return result;
}

pipe_expert_hello fault_hello() {
    pipe_expert_hello hello;
    hello.role           = PIPE_EXPERT_ROLE_WORKER;
    hello.hidden_type    = PIPE_HIDDEN_F16;
    hello.n_embd         = N_EMBD;
    hello.n_ff_exp       = N_FF_EXP;
    hello.n_expert       = N_EXPERT;
    hello.n_expert_used  = N_EXPERT;
    hello.expert_first   = 0;
    hello.expert_last    = N_EXPERT - 1;
    hello.n_slots        = 2;
    hello.layers         = { LAYER };
    hello.model_identity = MODEL_IDENTITY;
    return hello;
}

enum class fault_mode {
    reject_range,
    reject_layer,
    die,
};

struct fault_server {
    int                port;
    fault_mode         mode;
    std::thread        thread;
    std::exception_ptr error;
    std::promise<void> ready;

    explicit fault_server(fault_mode mode) : port(reserve_port()), mode(mode) {
        std::future<void> listening = ready.get_future();
        thread                      = std::thread([this]() {
            try {
                pipe_socket_ptr server = pipe_socket_t::create_server("127.0.0.1", port);
                if (!server) {
                    throw std::runtime_error("fault server failed to listen");
                }
                ready.set_value();
                pipe_socket_ptr client = server->accept();
                if (!client) {
                    throw std::runtime_error("fault server failed to accept");
                }

                std::vector<uint8_t> payload = pipe_encode_expert_hello(fault_hello());
                require(pipe_send_frame(*client, PIPE_HELLO, 0, payload.data(), payload.size()),
                        "fault server failed to send HELLO");

                pipe_frame_type type;
                uint64_t        seq_id = 0;
                require(pipe_recv_frame(*client, type, seq_id, payload), "fault server failed to receive client HELLO");
                require(type == PIPE_HELLO, "fault server expected client HELLO");
                require(pipe_recv_frame(*client, type, seq_id, payload), "fault server failed to receive PING");
                require(type == PIPE_PING, "fault server expected PING");
                require(pipe_send_frame(*client, PIPE_PONG, seq_id, nullptr, 0), "fault server failed to send PONG");

                require(pipe_recv_frame(*client, type, seq_id, payload), "fault server failed to receive dispatch");
                require(type == PIPE_EXPERT_DISPATCH_REQ, "fault server expected dispatch");
                if (this->mode == fault_mode::die) {
                    client.reset();
                    return;
                }
                const pipe_error_code code =
                    this->mode == fault_mode::reject_range ? PIPE_ERR_EXPERT_RANGE : PIPE_ERR_EXPERT_LAYER;
                require(pipe_send_error(*client, seq_id, code,
                                        this->mode == fault_mode::reject_range ? "worker does not serve expert 2" :
                                                                                 "worker does not serve layer 3"),
                        "fault server failed to send rejection");
            } catch (...) {
                error = std::current_exception();
                try {
                    ready.set_exception(error);
                } catch (...) {
                }
            }
        });
        listening.get();
    }

    ~fault_server() {
        if (thread.joinable()) {
            thread.join();
        }
    }

    void finish() {
        if (thread.joinable()) {
            thread.join();
        }
        if (error) {
            std::rethrow_exception(error);
        }
    }
};

std::string expect_dispatch_error(fault_mode mode, int32_t expert_id) {
    fault_server                           server(mode);
    const pipe_expert_dispatcher::endpoint endpoint = {
        "127.0.0.1",
        server.port,
        "fault-machine",
    };
    pipe_expert_dispatcher::dispatcher dispatcher({ endpoint });
    std::vector<uint16_t>              activations((size_t) N_TOKENS * N_EMBD, (uint16_t) ggml_fp32_to_fp16(0.25f));
    std::string                        message;
    try {
        dispatcher.dispatch(LAYER, 100 + (uint64_t) expert_id, N_TOKENS, activations,
                            {
                                { expert_id, { 1.0f, 0.0f } }
        });
    } catch (const std::runtime_error & error) {
        message = error.what();
    }
    server.finish();
    require(!message.empty(), "faulting worker did not fail dispatch");
    require(message.find("127.0.0.1:" + std::to_string(server.port)) != std::string::npos,
            "dispatch error does not name the worker");
    require(message.find(std::to_string(expert_id)) != std::string::npos, "dispatch error does not name the expert");
    return message;
}

void run_test() {
    temp_dir      temp;
    weight_map    weights;
    const fixture shard_a = make_fixture(temp.path, "shard-a", 0, 1, weights);
    const fixture shard_b = make_fixture(temp.path, "shard-b", 2, 3, weights);

    const int                   port_a0 = reserve_port();
    const int                   port_a1 = reserve_port();
    const int                   port_b0 = reserve_port();
    const int                   port_b1 = reserve_port();
    std::vector<worker_process> processes;
    processes.emplace_back(shard_a, port_a0, temp.path / "worker-a0.log");
    processes.emplace_back(shard_a, port_a1, temp.path / "worker-a1.log");
    processes.emplace_back(shard_b, port_b0, temp.path / "worker-b0.log");
    processes.emplace_back(shard_b, port_b1, temp.path / "worker-b1.log");
    wait_for_listener(port_a0, processes[0].pid);
    wait_for_listener(port_a1, processes[1].pid);
    wait_for_listener(port_b0, processes[2].pid);
    wait_for_listener(port_b1, processes[3].pid);
    bool        gap_rejected = false;
    std::string gap_error;
    try {
        pipe_expert_dispatcher::dispatcher gap({
            { "127.0.0.1", port_a0, "machine-a" },
            { "127.0.0.1", port_a1, "machine-a" },
        });
    } catch (const std::runtime_error & error) {
        gap_error    = error.what();
        gap_rejected = gap_error.find("coverage gap for layer 3 expert 2") != std::string::npos;
    }
    require(gap_rejected, "coverage gap was not rejected at construction: " + gap_error);

    std::vector<float>    activation((size_t) N_TOKENS * N_EMBD);
    std::vector<uint16_t> activation_f16;
    activation_f16.reserve(activation.size());
    for (size_t i = 0; i < activation.size(); ++i) {
        activation[i]          = ((int) (i % 13) - 6) * 0.07f;
        const ggml_fp16_t half = ggml_fp32_to_fp16(activation[i]);
        activation_f16.push_back((uint16_t) half);
        activation[i] = ggml_fp16_to_fp32(half);
    }
    const std::vector<pipe_expert_assignment> assignments = {
        { 0, { 0.50f, 0.00f }  },
        { 1, { 0.25f, 0.75f }  },
        { 2, { 0.00f, -0.40f } },
        { 3, { -0.20f, 0.30f } },
    };
    const std::vector<float> expected = reference(weights, activation, assignments);

    pipe_expert_dispatcher::dispatcher dispatcher({
        { "127.0.0.1", port_a0, "machine-a" },
        { "127.0.0.1", port_a1, "machine-a" },
        { "127.0.0.1", port_b0, "machine-b" },
        { "127.0.0.1", port_b1, "machine-b" },
    });
    const std::vector<float>           actual = dispatcher.dispatch(LAYER, 42, N_TOKENS, activation_f16, assignments);
    require(actual.size() == expected.size(), "reduced output shape mismatch");
    for (size_t i = 0; i < expected.size(); ++i) {
        const float tolerance = 0.003f + 0.02f * std::fabs(expected[i]);
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            throw std::runtime_error("reduced output mismatch at " + std::to_string(i) + ": actual=" +
                                     std::to_string(actual[i]) + " expected=" + std::to_string(expected[i]));
        }
    }

    const pipe_expert_dispatcher::dispatch_stats & stats = dispatcher.last_dispatch_stats();
    require(stats.first_await_recorded, "first response await was not instrumented");
    require(stats.workers_used == 4, "dispatch did not use all four capable workers");
    require(stats.requests_issued == stats.workers_used, "not every worker request was issued");
    require(stats.first_await_in_flight == stats.workers_used,
            "first response was awaited before all worker requests were issued");
    require(dispatcher.in_flight_requests() == 0, "in-flight requests remain after reduction");
    require(stats.workers.size() == 4, "worker balance stats are incomplete");
    for (const pipe_expert_dispatcher::worker_dispatch_stats & worker : stats.workers) {
        require(worker.n_experts > 0 && worker.n_experts < assignments.size(),
                "a capable worker received all work or no work");
    }
    std::cout << "measured first_await_in_flight=" << stats.first_await_in_flight
              << " workers_used=" << stats.workers_used << '\n';

    const std::string range_error = expect_dispatch_error(fault_mode::reject_range, 2);
    require(range_error.find("code " + std::to_string(PIPE_ERR_EXPERT_RANGE)) != std::string::npos,
            "expert range rejection lost its protocol code");
    const std::string layer_error = expect_dispatch_error(fault_mode::reject_layer, 1);
    require(layer_error.find("code " + std::to_string(PIPE_ERR_EXPERT_LAYER)) != std::string::npos,
            "expert layer rejection lost its protocol code");
    const std::string death_error = expect_dispatch_error(fault_mode::die, 3);
    require(death_error.find("died while computing") != std::string::npos,
            "worker death did not surface as a hard transport error");
}

}  // namespace

int main() {
    try {
        run_test();
        std::cout << "test-wp-expert-dispatcher: all tests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "test-wp-expert-dispatcher: " << error.what() << '\n';
        return 1;
    }
}
