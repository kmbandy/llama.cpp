#include "server-common.h"
#include "http.h"
#include "server-models.h"
#include "server-context.h"
#include "server-stream.h"

#include "build-info.h"
#include "preset.h"
#include "download.h"
#include "fit.h"
#include "hf-cache.h"
#include "http.h"
#include "subproc.h"
#include "server-router-groups.h"
#include "server-router-ledger.h"
#include "server-router-policy.h"
#include "server-router-probe.h"

#include <cpp-httplib/httplib.h> // TODO: remove this once we use HTTP client from download.h
#include <cinttypes>
#include <iterator>
#include <fstream>
#include <optional>

#include <functional>
#include <optional>
#include <algorithm>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <atomic>
#include <chrono>
#include <queue>
#include <filesystem>
#include <random>
#include <sstream>
#include <cstring>

#ifndef _WIN32
#include <unistd.h>
extern char **environ;
#endif

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

#define DEFAULT_STOP_TIMEOUT 10 // seconds
#define ROUTER_GPU_MARGIN_BYTES (1024LL * 1024LL * 1024LL)

#define CMD_ROUTER_TO_CHILD_EXIT  "cmd_router_to_child:exit"
#define CMD_CHILD_TO_ROUTER_STATE "cmd_child_to_router:state:" // followed by json string

static constexpr const char * ROUTER_ARG_GPU          = "LLAMA_ARG_ROUTER_GPU";
static constexpr const char * ROUTER_ARG_VRAM_MB      = "LLAMA_ARG_ROUTER_VRAM_MB";
static constexpr const char * ROUTER_ARG_RAM_MB      = "LLAMA_ARG_ROUTER_RAM_MB";
static constexpr const char * ROUTER_ARG_ENV          = "LLAMA_ARG_ROUTER_ENV";
static constexpr const char * ROUTER_ARG_PINNED       = "LLAMA_ARG_ROUTER_PINNED";
static constexpr const char * ROUTER_ARG_EXCLUSIVE    = "LLAMA_ARG_ROUTER_EXCLUSIVE";
static constexpr const char * ROUTER_ARG_IDLE_TIMEOUT = "LLAMA_ARG_ROUTER_IDLE_TIMEOUT";
static constexpr const char * ROUTER_ARG_PRIORITY     = "LLAMA_ARG_ROUTER_PRIORITY";
static constexpr const char * ROUTER_LOAD_TIMEOUT     = "LLAMA_SERVER_ROUTER_LOAD_TIMEOUT";

static constexpr int DEFAULT_ROUTER_LOAD_TIMEOUT_S = 900;

// note: SIGPIPE is ignored by the server
static void request_child_exit(server_subproc & proc) {
    FILE * stdin_file = proc.sproc.stdin_file();
    if (stdin_file) {
        fprintf(stdin_file, "%s\n", CMD_ROUTER_TO_CHILD_EXIT);
        fflush(stdin_file);
    }
}

static bool router_machine_is_local(const std::string & machine); // defined with the group helpers below

// address for child process, this is needed because router may run on 0.0.0.0
// ref: https://github.com/ggml-org/llama.cpp/issues/17862
#define CHILD_ADDR "127.0.0.1"

// single-threaded, watching all child processes at once
struct server_monitor {
    server_monitor(server_models & models) : models(models) {
        th = std::thread([this]() { run(); });
    }

    ~server_monitor() {
        push({ cmd_t::QUIT, {}, "", 0, false });
        th.join();
    }

    // thread-safe
    void watch(const std::string & name, std::shared_ptr<server_subproc> proc, server_child_mode mode, int port) {
        child_t c;
        c.name = name;
        c.proc = std::move(proc);
        c.mode = mode;
        c.port = port;
        if (!c.proc->has_output()) {
            SRV_ERR("failed to get stdout/stderr of child process for name=%s\n", name.c_str());
            c.eof = true;
        }
        push({ cmd_t::WATCH, std::move(c), "", 0, false });
    }

    // thread-safe
    void stop(const std::string & name, int stop_timeout, bool send_exit) {
        push({ cmd_t::STOP, {}, name, stop_timeout, send_exit });
    }

private:
    struct child_t {
        std::string name;
        std::shared_ptr<server_subproc> proc;
        server_child_mode mode = SERVER_CHILD_MODE_NORMAL;
        int port = 0;
        std::string buf;      // partial line
        bool eof = false;     // output closed, waiting for the process to be reaped
        int64_t deadline = 0; // force-kill time in ms, 0 when no stop is pending
    };

    struct cmd_t {
        enum { WATCH, STOP, QUIT } type;
        child_t child;
        std::string name;
        int  stop_timeout;
        bool send_exit;
    };

    void push(cmd_t && cmd) {
        {
            std::lock_guard<std::mutex> lk(mu);
            cmds.push_back(std::move(cmd));
        }
        waiter.wake();
    }

    // returns true if the loop should exit
    bool handle_commands() {
        std::deque<cmd_t> batch;
        {
            std::lock_guard<std::mutex> lk(mu);
            batch.swap(cmds);
        }
        for (auto & cmd : batch) {
            switch (cmd.type) {
                case cmd_t::WATCH:
                    children.push_back(std::move(cmd.child));
                    break;
                case cmd_t::STOP:
                    // the newest child with this name is the one the registry knows
                    for (auto it = children.rbegin(); it != children.rend(); ++it) {
                        if (it->name != cmd.name) {
                            continue;
                        }
                        if (cmd.send_exit && !it->eof) {
                            request_child_exit(*it->proc);
                        }
                        it->deadline = ggml_time_ms() + (int64_t) cmd.stop_timeout * 1000;
                        break;
                    }
                    break;
                case cmd_t::QUIT:
                    return true;
            }
        }
        return false;
    }

    // read what the child wrote, forward complete lines
    void read_output(child_t & c) {
        char chunk[4096];
        while (!c.eof) {
            int n = c.proc->read_output(chunk, sizeof(chunk));
            if (n < 0) {
                c.eof = true;
                break;
            }
            if (n == 0) {
                break;
            }
            c.buf.append(chunk, (size_t) n);
            size_t start = 0;
            while (true) {
                size_t nl = c.buf.find('\n', start);
                if (nl == std::string::npos) {
                    break;
                }
                std::string line = c.buf.substr(start, nl + 1 - start);
                start = nl + 1;
                on_line(c, line);
            }
            c.buf.erase(0, start);
            if (c.buf.size() > max_line) {
                c.buf.clear(); // a child that never writes a newline must not grow this without bound
            }
        }
        if (c.eof && !c.buf.empty()) {
            on_line(c, c.buf);
            c.buf.clear();
        }
    }

    void on_line(child_t & c, const std::string & line) {
        if (string_starts_with(line, CMD_CHILD_TO_ROUTER_STATE)) {
            LOG_DBG("[%5d] %s", c.port, line.c_str()); // prevent spamming the log
            models.handle_child_state(c.name, line);
        } else {
            LOG("[%5d] %s", c.port, line.c_str()); // forward log
        }
    }

    void run() {
        while (true) {
            if (handle_commands()) {
                return;
            }

            // wait for output, a wakeup, or the next deadline;
            // a child whose output closed is polled for its exit every 50 ms
            int64_t now     = ggml_time_ms();
            int64_t timeout = -1;
            for (const auto & c : children) {
                if (c.eof) {
                    timeout = timeout < 0 ? 50 : std::min<int64_t>(timeout, 50);
                }
                if (c.deadline) {
                    int64_t d = std::max<int64_t>(0, c.deadline - now);
                    timeout = timeout < 0 ? d : std::min(timeout, d);
                }
            }
            std::vector<server_subproc *> procs;
            std::vector<child_t *>        owners;
            for (auto & c : children) {
                if (!c.eof) {
                    procs.push_back(c.proc.get());
                    owners.push_back(&c);
                }
            }
            std::vector<bool> ready;
            waiter.wait(procs, ready, timeout);
            for (size_t i = 0; i < owners.size(); i++) {
                if (ready[i]) {
                    read_output(*owners[i]);
                }
            }

            // deadlines and exits
            now = ggml_time_ms();
            for (auto it = children.begin(); it != children.end();) {
                if (it->deadline && now >= it->deadline && !it->proc->stopped.load(std::memory_order_acquire)) {
                    SRV_WRN("force-killing model instance name=%s after timeout\n", it->name.c_str());
                    it->proc->terminate();
                    it->deadline = 0;
                }
                if (it->eof && !it->proc->is_alive()) {
                    int exit_code = it->proc->join();
                    it->proc->stopped.store(true, std::memory_order_release);
                    models.on_child_exit(it->name, it->proc, it->mode, exit_code);
                    SRV_INF("instance name=%s exited with status %d\n", it->name.c_str(), exit_code);
                    it = children.erase(it);
                } else {
                    ++it;
                }
            }
        }
    }

    static constexpr size_t max_line = 1024 * 1024;

    server_models & models;
    std::mutex mu;
    std::deque<cmd_t> cmds;
    std::vector<child_t> children; // monitor thread only
    server_subproc::waiter waiter;
    std::thread th;
};

struct server_lru_sched {
    server_lru_sched(server_models & models) : models(models) {}

    bool has_capacity(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return models.base_params.models_max <= 0
            || count_running() < (size_t) models.base_params.models_max;
    }

    // returns "" if no model can be given up
    std::string pick_victim(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        std::string victim;
        int64_t victim_last_used = 0;
        for (const auto & m : models.mapping) {
            // a busy model is mid-request, one still coming up has no request to finish
            if (m.second.req_count != 0 || !m.second.meta.is_ready_or_sleep()) {
                continue;
            }
            // a group worker goes with its spine, never on its own
            if (m.second.meta.is_external()) {
                continue;
            }
            // FORK GUARD: pinned is a HARD HOLD. A pinned resident owns its GPU until a
            // human unpins it (the DSWS / weight-paging case). Evicting one pulls the card
            // out from under work that asked to keep it.
            if (m.second.meta.placement.pinned) {
                continue;
            }
            // a hold lease is the same promise, made by an orchestrator for a while
            if (models.is_held_locked(m.first)) {
                continue;
            }
            // already on its way out, or a queued request wants it
            if (models.stopping_models.count(m.first) || find(m.first)) {
                continue;
            }
            if (victim.empty() || m.second.meta.last_used < victim_last_used) {
                victim           = m.first;
                victim_last_used = m.second.meta.last_used;
            }
        }
        return victim;
    }

    // requests wanting the same model share one entry, so they all need only one slot
    // and all get unblocked by the single load that entry performs
    void join(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (entry_t * e = find(model_id)) {
            e->n_waiters++;
            SRV_INF("request for name=%s joined the queue, %d waiting\n", model_id.c_str(), e->n_waiters);
            return;
        }
        queue.push_back({ model_id, 1, false });
        SRV_INF("request for name=%s queued at position %zu\n",
                model_id.c_str(), queue.size());
    }

    void leave(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                if (--it->n_waiters <= 0) {
                    queue.erase(it); // last one waiting for this model went away
                }
                return;
            }
        }
    }

    bool queue_empty(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        return queue.empty();
    }

    // true if it is this model's turn to load, and nobody is loading it yet
    bool try_claim(std::unique_lock<std::mutex> & lk, const std::string & model_id) {
        check_lock(lk);
        if (queue.empty() || queue.front().model_id != model_id || queue.front().loading) {
            return false;
        }
        if (!has_capacity(lk)) {
            return false;
        }
        queue.front().loading = true;
        return true;
    }

    // on failure the entry is back in line; on success it stays until its waiters leave,
    // so the model coming up is never picked as a victim before they use it
    void claim_done(std::unique_lock<std::mutex> & lk, const std::string & model_id, bool ok) {
        check_lock(lk);
        if (ok) {
            return;
        }
        for (auto it = queue.begin(); it != queue.end(); ++it) {
            if (it->model_id == model_id) {
                it->loading = false;
                return;
            }
        }
    }

    // evict idle models while queued requests outnumber the slots that are free or being freed
    // caller must hold models.mutex; never blocks, so it is safe from any thread
    void tick(std::unique_lock<std::mutex> & lk) {
        check_lock(lk);
        if (models.base_params.models_max <= 0 || queue.empty()) {
            return;
        }
        int n_running  = 0;
        int n_stopping = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running() && !m.second.meta.is_external()) { // a group takes one slot
                n_running++;
                if (models.stopping_models.count(m.first)) {
                    n_stopping++;
                }
            }
        }
        int n_needed  = 0;
        int n_claimed = 0; // claimed the slot, but load() has not spawned yet
        for (const auto & e : queue) {
            if (!e.loading) {
                n_needed++;
                continue;
            }
            auto it = models.mapping.find(e.model_id);
            if (it != models.mapping.end() && !it->second.meta.is_running()) {
                n_claimed++;
            }
        }
        int n_free = models.base_params.models_max - n_running + n_stopping - n_claimed;
        while (n_free < n_needed) {
            std::string victim = pick_victim(lk);
            if (victim.empty()) {
                return; // all remaining models are busy, wait for a request to end
            }
            SRV_INF("evicting idle LRU name=%s for a queued request\n", victim.c_str());
            models.request_stop(victim);
            n_free++;
        }
    }

  private:
    struct entry_t {
        std::string model_id;
        int  n_waiters; // requests waiting for this model
        bool loading;   // one of the waiters is doing the load right now
    };

    entry_t * find(const std::string & model_id) {
        for (auto & e : queue) {
            if (e.model_id == model_id) {
                return &e;
            }
        }
        return nullptr;
    }

    void check_lock(std::unique_lock<std::mutex> & lk) {
        GGML_ASSERT(lk.owns_lock() && lk.mutex() == &models.mutex);
    }

    size_t count_running() {
        size_t count = 0;
        for (const auto & m : models.mapping) {
            if (m.second.meta.is_running() && !m.second.meta.is_external()) { // a group takes one slot
                count++;
            }
        }
        return count;
    }

    server_models & models;
    std::deque<entry_t> queue;
};

// short loopback budget for the resumable stream router to child JSON calls (probe, lookup,
// delete). distinct from params.timeout_read/write which only applies to the generation proxy
static constexpr int STREAM_LOOKUP_TIMEOUT_MS = 250;

static std::filesystem::path get_server_exec_path() {
#if defined(_WIN32)
    wchar_t buf[32768] = { 0 };  // Large buffer to handle long paths
    DWORD len = GetModuleFileNameW(nullptr, buf, _countof(buf));
    if (len == 0 || len >= _countof(buf)) {
        throw std::runtime_error("GetModuleFileNameW failed or path too long");
    }
    return std::filesystem::path(buf);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        throw std::runtime_error("_NSGetExecutablePath failed after buffer resize");
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    if (count <= 0) {
        throw std::runtime_error("failed to resolve /proc/self/exe");
    }
    return std::filesystem::path(std::string(path, count));
#endif
}

static void unset_reserved_args(common_preset & preset, bool unset_model_args) {
    preset.unset_option("LLAMA_ARG_SSL_KEY_FILE");
    preset.unset_option("LLAMA_ARG_SSL_CERT_FILE");
    preset.unset_option("LLAMA_API_KEY");
    preset.unset_option("LLAMA_ARG_API_KEY_FILE");
    preset.unset_option("LLAMA_ARG_MODELS_DIR");
    preset.unset_option("LLAMA_ARG_MODELS_MAX");
    preset.unset_option("LLAMA_ARG_MODELS_PRESET");
    preset.unset_option("LLAMA_ARG_MODELS_AUTOLOAD");
    preset.unset_option("LLAMA_ARG_MODELS_IDLE_TIMEOUT");
    preset.unset_option("LLAMA_ARG_GPUS");
    preset.unset_option("LLAMA_ARG_BOARD_URL");
    preset.unset_option("LLAMA_ARG_BOARD_TOKEN_FILE");
    preset.unset_option(ROUTER_ARG_PRIORITY);
    preset.unset_option(ROUTER_ARG_GPU);
    preset.unset_option(ROUTER_ARG_VRAM_MB);
    preset.unset_option(ROUTER_ARG_RAM_MB);
    preset.unset_option(ROUTER_ARG_ENV);
    preset.unset_option(ROUTER_ARG_PINNED);
    preset.unset_option(ROUTER_ARG_EXCLUSIVE);
    preset.unset_option(ROUTER_ARG_IDLE_TIMEOUT);
    preset.unset_option(ROUTER_ARG_KIND);
    preset.unset_option(ROUTER_ARG_DEPENDS);
    preset.unset_option(ROUTER_ARG_LAUNCH);
    preset.unset_option(ROUTER_ARG_PARK_FILE);
    preset.unset_option(ROUTER_ARG_PARK_MODE);
    preset.unset_option(ROUTER_ARG_MACHINE);
    preset.unset_option(ROUTER_ARG_STARTUP_TIMEOUT);
    preset.unset_option(ROUTER_ARG_WORKER_PORT);
    if (unset_model_args) {
        preset.unset_option("LLAMA_ARG_MODEL");
        preset.unset_option("LLAMA_ARG_MMPROJ");
        preset.unset_option("LLAMA_ARG_ALIAS");
        preset.unset_option("LLAMA_ARG_HF_REPO");
    }
}

#ifdef _WIN32
static std::string wide_to_utf8(const wchar_t * ws) {
    if (!ws || !*ws) {
        return {};
    }

    const int len = static_cast<int>(std::wcslen(ws));
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, ws, len, nullptr, 0, nullptr, nullptr);
    if (bytes == 0) {
        return {};
    }

    std::string utf8(bytes, '\0');
    WideCharToMultiByte(CP_UTF8, 0, ws, len, utf8.data(), bytes, nullptr, nullptr);

    return utf8;
}
#endif

static std::vector<std::string> get_environment() {
    std::vector<std::string> env;

#ifdef _WIN32
    LPWCH env_block = GetEnvironmentStringsW();
    if (!env_block) {
        return env;
    }
    for (LPWCH e = env_block; *e; e += wcslen(e) + 1) {
        env.emplace_back(wide_to_utf8(e));
    }
    FreeEnvironmentStringsW(env_block);
#else
    if (environ == nullptr) {
        return env;
    }
    for (char ** e = environ; *e != nullptr; e++) {
        env.emplace_back(*e);
    }
#endif

    return env;
}

static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec);

void server_model_meta::update_args(common_preset_context & ctx_preset, std::string bin_path) {
    // update params
    unset_reserved_args(preset, false);
    preset.set_option(ctx_preset, "LLAMA_ARG_HOST",  CHILD_ADDR);
    preset.set_option(ctx_preset, "LLAMA_ARG_PORT",  std::to_string(port));
    preset.set_option(ctx_preset, "LLAMA_ARG_ALIAS", name);
    if (!placement.devs.empty()) {
        std::string dev_list;
        for (const auto & dev : placement.devs) {
            if (!dev_list.empty()) {
                dev_list += ",";
            }
            dev_list += dev;
        }
        preset.set_option(ctx_preset, "LLAMA_ARG_DEVICE", dev_list);
        // Split/tensor-split only mean anything across MULTIPLE devices. `exclusive` is
        // now the default for single-GPU models too (a model owns its card), so gate this
        // on the span -- otherwise a one-GPU model gets a nonsense one-element
        // --tensor-split of the card's byte count.
        if (placement.exclusive && placement.devs.size() > 1) {
            // These are DEFAULTS for a multi-GPU span, not overrides. set_option()
            // updates an existing entry in place, so setting them unconditionally
            // silently discarded whatever the operator wrote in the preset -- which
            // made a tensor-parallel model impossible to express: `split-mode =
            // tensor` was rewritten to "layer", and an explicit `tensor-split`
            // was replaced by one derived from the cards' byte counts.
            //
            // Only fill in what the preset did not specify. An operator who names
            // a split mode or a ratio has a reason the VRAM ledger cannot see (here:
            // a tensor-parallel span whose optimal ratio is set by compute rate, not
            // capacity), and the ledger's guess must not win over it.
            auto set_if_unset = [&](const char * env, const std::string & value) {
                std::string existing;
                if (preset.get_option(env, existing) && !existing.empty()) {
                    return;
                }
                preset.set_option(ctx_preset, env, value);
            };

            set_if_unset("LLAMA_ARG_SPLIT_MODE", "layer");
            set_if_unset("LLAMA_ARG_MAIN_GPU",   "0");
            if (!placement.split.empty()) {
                std::string split_str;
                for (float v : placement.split) {
                    if (!split_str.empty()) {
                        split_str += ",";
                    }
                    split_str += std::to_string(v);
                }
                set_if_unset("LLAMA_ARG_TENSOR_SPLIT", split_str);
            }
        }
    }
    // TODO: maybe validate preset before rendering ?
    // render args
    args = preset.to_args(bin_path);

    // unified binary dispatches by subcommand, re-inject it right after the
    // binary path so the child starts as 'llama serve ...' not 'llama ...'
    const char * app_cmd = std::getenv("LLAMA_APP_CMD");
    if (app_cmd != nullptr && app_cmd[0] != '\0' && !bin_path.empty()) {
        args.insert(args.begin() + 1, app_cmd);
    }
}

void server_model_meta::update_caps() {
    try {
        common_params params;
        preset.apply_to_params(params, {
            "LLAMA_ARG_MODEL",
            "LLAMA_ARG_MODEL_URL",
            "LLAMA_ARG_MMPROJ",
            "LLAMA_ARG_MMPROJ_URL",
            "LLAMA_ARG_MMPROJ_AUTO",
            "LLAMA_ARG_HF_REPO",
            "LLAMA_ARG_HF_REPO_FILE",
        });
        params.offline = true;
        common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
        common_models_handler_apply(handler, params); // note: this won't download the model because offline=true
        if (params.no_mmproj || params.mmproj.path.empty()) {
            multimodal = { false, false };
        } else {
            multimodal = mtmd_get_cap_from_file(params.mmproj.path.c_str());
        }
    } catch (const std::exception & e) {
        LOG_WRN("failed to initialize common_params for multimodal capability detection: %s\n", e.what());
        multimodal = { false, false };
    }
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv)
            : ctx_preset(LLAMA_EXAMPLE_SERVER),
              base_params(params),
              base_env(get_environment()),
              base_preset(ctx_preset.load_from_args(argc, argv)),
              sched(std::make_unique<server_lru_sched>(*this)),
              monitor(std::make_unique<server_monitor>(*this)) {
    // propagate base params to child
    unset_reserved_args(base_preset, true);

    // do not propagate these options, but allow preset to explicitly set them
    base_preset.unset_option("LLAMA_ARG_LOG_FILE");

    // Every child this router spawns (spines, workers, estimate/probe children) carries this
    // start's generation and our PID, so a later router can tell what a dead one left behind.
    router_gen = router_generation_new();
    {
        auto set_env = [this](const std::string & key, const std::string & value) {
            const std::string prefix = key + "=";
            base_env.erase(std::remove_if(base_env.begin(), base_env.end(), [&](const std::string & e) {
                return e.compare(0, prefix.size(), prefix) == 0;
            }), base_env.end());
            base_env.push_back(prefix + value);
        };
        set_env(ROUTER_ENV_GEN, router_gen);
#ifndef _WIN32
        set_env(ROUTER_ENV_ROUTER_PID, std::to_string((int) getpid()));
#endif
    }
#ifndef _WIN32
    // Kill what a previous router generation left behind (workers do not watch their stdin, so
    // they outlive a crashed router). TERM first: a worker writes its stop snapshot on TERM.
    {
        const auto swept = router_sweep_stale_children("", router_gen, (unsigned) getuid(), (int) getpid(),
                                                       ROUTER_WORKER_QUIESCE_MS_DEFAULT + ROUTER_WORKER_KILL_GRACE_MS);
        if (!swept.empty()) {
            SRV_WRN("stopped %zu process(es) left behind by a previous router generation\n", swept.size());
        }
        SRV_INF("router generation %s\n", router_gen.c_str());
    }
#endif

    // set binary path
    try {
        bin_path = get_server_exec_path().string();
    } catch (const std::exception & e) {
        bin_path = argv[0];
        LOG_WRN("failed to get server executable path: %s\n", e.what());
        LOG_WRN("using original argv[0] as fallback: %s\n", argv[0]);
    }
    load_models();
    autoload_enabled.store(params.models_autoload, std::memory_order_relaxed);
    debug_fake_timing = !common_get_env("LLAMA_SERVER_DEBUG_FAKE_TIMING").empty();

    // The coordination board: claims for what the router loads, queueing behind sessions,
    // yielding idle GPUs to them. No --board-url: all of it is off and the router behaves as
    // before (holds, priorities and the busy-resident queue still work).
    local_machine = router_local_machine();
    if (!base_params.router_board_url.empty()) {
        std::string token;
        if (base_params.router_board_token_file.empty()) {
            SRV_WRN("%s", "--board-url without --board-token-file: board writes (claims) will be refused\n");
        } else {
            std::ifstream f(base_params.router_board_token_file);
            if (!f) {
                SRV_WRN("cannot read board token file '%s': board writes (claims) will be refused\n",
                        base_params.router_board_token_file.c_str());
            } else {
                std::stringstream ss;
                ss << f.rdbuf();
                token = string_strip(ss.str());
            }
        }
        router_board_config cfg;
        cfg.url     = base_params.router_board_url;
        cfg.token   = token;
        cfg.machine = local_machine;
        router_board_host host;
        host.residents = [this]() { return board_residents(); };
        host.yield     = [this](const std::string & name, const std::string & reason) { board_yield(name, reason); };
        host.changed   = [this]() {
            std::lock_guard<std::mutex> lk(mutex);
            bump_queue_locked();
        };
        board = std::make_unique<router_board_agent>(cfg, host);
        board->start();
        SRV_INF("coordination board %s, machine '%s', holder '%s'\n", cfg.url.c_str(), local_machine.c_str(), ROUTER_BOARD_HOLDER);
    } else {
        SRV_WRN("no --board-url: coordination board features are off (machine '%s')\n", local_machine.c_str());
    }
    queue_th = std::thread(&server_models::queue_runner_loop, this);
    // Always start the sweeper in router mode: effective timeout is per-model
    // (preset idle-timeout if set, else --models-idle-timeout). When every
    // effective value is 0 the loop is a no-op; when global is 0 but a model
    // sets idle-timeout=N that model still gets unloaded.
    idle_th = std::thread(&server_models::idle_sweeper_loop, this);
}

// Must stay OUT-OF-LINE: `sched` is a unique_ptr<server_lru_sched>, an incomplete
// type at the header. Upstream added a `= default` dtor here for exactly that
// reason; this one subsumes it and also joins the idle sweeper.
server_models::~server_models() {
    stop_threads();
    idle_stop.store(true, std::memory_order_relaxed);
    if (idle_th.joinable()) {
        idle_th.join();
    }
    // Worker groups go before the monitor. Destroyed outside the lock: a group's destructor
    // joins its thread, which may be inside a callback waiting for `mutex`. Anything still
    // running here (no unload_all() before shutdown) is SIGKILLed by the destructor.
    std::vector<std::shared_ptr<router_worker_group>> dying;
    {
        std::lock_guard<std::mutex> lk(mutex);
        for (auto & [_, rt] : groups) {
            if (rt.workers) {
                dying.push_back(std::move(rt.workers));
            }
        }
    }
    dying.clear();
}

std::optional<std::filesystem::file_time_type> server_models::get_models_preset_mtime() const {
    if (base_params.models_preset.empty()) {
        return std::nullopt;
    }

    std::error_code ec;
    const auto mtime = std::filesystem::last_write_time(base_params.models_preset, ec);
    if (ec) {
        SRV_WRN("failed to stat models preset '%s': %s\n", base_params.models_preset.c_str(), ec.message().c_str());
        return std::nullopt;
    }

    return mtime;
}

void server_models::reload_models_preset_if_changed(server_model_meta & meta) {
    const auto mtime_before = get_models_preset_mtime();
    const auto applied_mtime = models_preset_applied_mtimes.find(meta.name);
    if (!mtime_before.has_value() ||
            (applied_mtime != models_preset_applied_mtimes.end() && *mtime_before == applied_mtime->second)) {
        return;
    }

    try {
        common_presets cached_models = ctx_preset.load_from_cache();
        common_presets local_models;
        if (!base_params.models_dir.empty()) {
            local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        }

        common_preset global;
        common_presets custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        const auto mtime_after = get_models_preset_mtime();
        if (!mtime_after.has_value() || *mtime_before != *mtime_after) {
            SRV_WRN("models preset '%s' changed while reloading; keeping last-known-good preset for model '%s'\n",
                base_params.models_preset.c_str(), meta.name.c_str());
            return;
        }

        cached_models  = ctx_preset.cascade(global, cached_models);
        local_models   = ctx_preset.cascade(global, local_models);
        custom_presets = ctx_preset.cascade(global, custom_presets);

        common_presets final_presets;
        std::unordered_map<std::string, server_model_source> source_map;
        for (const auto & [name, preset] : cached_models) {
            final_presets[name] = preset;
            source_map[name] = SERVER_MODEL_SOURCE_CACHE;
        }
        for (const auto & [name, preset] : local_models) {
            final_presets[name] = preset;
            source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
        }
        for (const auto & [name, custom] : custom_presets) {
            if (final_presets.find(name) != final_presets.end()) {
                final_presets[name].merge(custom);
            } else {
                final_presets[name] = custom;
            }
            source_map[name] = SERVER_MODEL_SOURCE_PRESET;
        }
        for (auto & [name, preset] : final_presets) {
            preset.merge(base_preset);
        }

        auto it = final_presets.find(meta.name);
        if (it == final_presets.end()) {
            throw std::runtime_error("updated models preset no longer resolves model '" + meta.name + "'");
        }

        meta.preset = it->second;
        meta.source = source_map.at(meta.name);
        parse_model_placement(meta);
        meta.update_args(ctx_preset, bin_path);
        meta.update_caps();
        models_preset_applied_mtimes[meta.name] = *mtime_after;
        SRV_INF("reloaded models preset '%s' for child model '%s'\n", base_params.models_preset.c_str(), meta.name.c_str());
    } catch (const std::exception & e) {
        SRV_WRN("failed to reload models preset '%s' for child model '%s': %s; keeping last-known-good preset\n",
            base_params.models_preset.c_str(), meta.name.c_str(), e.what());
    }
}

void server_models::instance_t::request_exit() const {
    request_child_exit(*subproc);
}

void server_models::add_model(server_model_meta && meta) {
    if (mapping.find(meta.name) != mapping.end()) {
        throw std::runtime_error(string_format("model '%s' appears multiple times", meta.name.c_str()));
    }

    // check model name does not conflict with existing aliases
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(meta.name)) {
            throw std::runtime_error(string_format("model name '%s' conflicts with alias of model '%s'",
                meta.name.c_str(), key.c_str()));
        }
    }

    // parse aliases from preset's --alias option (comma-separated)
    std::string alias_str;
    if (meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
        for (auto & alias : string_split<std::string>(alias_str, ',')) {
            alias = string_strip(alias);
            if (!alias.empty()) {
                meta.aliases.insert(alias);
            }
        }
    }

    // parse tags from preset's --tags option (comma-separated)
    std::string tags_str;
    if (meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
        for (auto & tag : string_split<std::string>(tags_str, ',')) {
            tag = string_strip(tag);
            if (!tag.empty()) {
                meta.tags.insert(tag);
            }
        }
    }

    // validate aliases do not conflict with existing names or aliases
    for (const auto & alias : meta.aliases) {
        if (mapping.find(alias) != mapping.end()) {
            throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with existing model name",
                alias.c_str(), meta.name.c_str()));
        }
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(alias)) {
                throw std::runtime_error(string_format("alias '%s' for model '%s' conflicts with alias of model '%s'",
                    alias.c_str(), meta.name.c_str(), key.c_str()));
            }
        }
    }

    parse_model_placement(meta);
    meta.update_args(ctx_preset, bin_path); // render args
    if (!meta.is_external()) {
        meta.update_caps(); // a worker has no model of its own to probe
    }
    std::string name = meta.name;
    mapping[name] = instance_t{
        /* subproc */ std::make_shared<server_subproc>(),
        /* meta    */ std::move(meta)
    };
}

static int64_t parse_mb_to_bytes(const std::string & value) {
    return std::stoll(value) * 1024LL * 1024LL;
}

// Physical VRAM total of a probe, bytes; -1 if it cannot be read. For a sysfs probe
// (".../mem_info_vram_used") the sibling mem_info_vram_total is read; for "nvml:N" nvidia-smi is asked.
static int64_t read_probe_total_bytes(const std::string & vram_probe) {
    if (vram_probe.rfind("nvml:", 0) == 0) {
        const std::string idx = vram_probe.substr(strlen("nvml:"));
        const std::string cmd = "nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i " + idx + " 2>/dev/null";
        FILE * pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            return -1;
        }
        char buffer[128] = {};
        std::string out;
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            out = buffer;
        }
        pclose(pipe);
        try {
            return parse_mb_to_bytes(string_strip(out));
        } catch (...) {
            return -1;
        }
    }
    const std::string suffix = "mem_info_vram_used";
    if (vram_probe.size() < suffix.size() ||
            vram_probe.compare(vram_probe.size() - suffix.size(), suffix.size(), suffix) != 0) {
        return -1;
    }
    std::ifstream file(vram_probe.substr(0, vram_probe.size() - suffix.size()) + "mem_info_vram_total");
    int64_t total = -1;
    if (file >> total) {
        return total;
    }
    return -1;
}

// Card-wide VRAM in use, bytes; -1 if the probe cannot be read.
static int64_t read_vram_used_bytes(const server_gpu_slot & slot) {
    if (slot.vram_probe.rfind("nvml:", 0) == 0) {
        const std::string idx = slot.vram_probe.substr(strlen("nvml:"));
        const std::string cmd = "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i " + idx + " 2>/dev/null";
        FILE * pipe = popen(cmd.c_str(), "r");
        if (!pipe) {
            return -1;
        }
        char buffer[128] = {};
        std::string out;
        if (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
            out = buffer;
        }
        pclose(pipe);
        try {
            return parse_mb_to_bytes(string_strip(out));
        } catch (...) {
            return -1;
        }
    }
    std::ifstream file(slot.vram_probe);
    int64_t used = 0;
    if (file >> used) {
        return used;
    }
    return -1;
}

bool server_models::load_gpu_config(const common_preset & global_preset) {
    if (gpu_placement_enabled) {
        return true;
    }
    std::string spec = base_params.router_gpus;
    if (spec.empty()) {
        global_preset.get_option("LLAMA_ARG_GPUS", spec);
    }

    gpu_slots.clear();
    gpu_placement_enabled = false;
    if (spec.empty()) {
        return false;
    }

    for (auto entry : string_split<std::string>(spec, ',')) {
        entry = string_strip(entry);
        if (entry.empty()) {
            continue;
        }
        const size_t p0 = entry.find(':');
        const size_t p1 = p0 == std::string::npos ? std::string::npos : entry.find(':', p0 + 1);
        if (p0 == std::string::npos || p1 == std::string::npos) {
            throw std::runtime_error("invalid --gpus entry '" + entry + "', expected name:total_mb:probe");
        }
        server_gpu_slot slot;
        // "ROCm0=R9700": the device, then its name on the coordination board
        if (!router_gpu_name_split(entry.substr(0, p0), slot.dev_name, slot.board_name)) {
            throw std::runtime_error("invalid --gpus entry '" + entry + "', expected name[=board]:total_mb:probe");
        }
        slot.vram_probe = entry.substr(p1 + 1);
        // total_mb is an optional OVERRIDE of the slot total: empty or 0 means "use the
        // probe's physical total" (whole card). A positive value caps the slot below that.
        const std::string mb_str = string_strip(entry.substr(p0 + 1, p1 - p0 - 1));
        const int64_t override_bytes = mb_str.empty() ? 0 : parse_mb_to_bytes(mb_str);
        const int64_t probe_total = read_probe_total_bytes(slot.vram_probe);
        slot.total_bytes = override_bytes > 0 ? override_bytes : probe_total;
        if (override_bytes > 0 && probe_total > 0 && override_bytes < probe_total) {
            SRV_WRN("GPU slot %s: declared total %" PRId64 " MB caps the physical %" PRId64 " MB; "
                    "leave total_mb empty to use the whole card\n",
                    slot.dev_name.c_str(), override_bytes / (1024 * 1024), probe_total / (1024 * 1024));
        }
        slot.pdev = pdev_for_probe(slot.vram_probe); // "" for NVML: whole-card, no per-PID view
        if (slot.dev_name.empty() || slot.total_bytes <= 0 || slot.vram_probe.empty()) {
            throw std::runtime_error("invalid --gpus entry '" + entry + "'");
        }
        for (const auto & existing : gpu_slots) {
            if (existing.dev_name == slot.dev_name) {
                throw std::runtime_error("duplicate GPU slot '" + slot.dev_name + "'");
            }
        }
        gpu_slots.push_back(std::move(slot));
    }

    gpu_placement_enabled = !gpu_slots.empty();
    if (gpu_placement_enabled) {
        validate_gpu_slots();
        SRV_INF("router GPU placement enabled with %zu declared slots\n", gpu_slots.size());
    }
    return gpu_placement_enabled;
}

// Whether this model owns its GPU(s) outright.
//
// DEFAULT IS TRUE, deliberately. The VRAM ledger's best-fit placement will happily
// co-locate two models on one card whenever they both "fit" on paper, and a bad
// estimate then OOMs a GPU that a human may be using for something else. One model
// per GPU unless the operator explicitly says otherwise (`no-exclusive`) is the safe
// default: loading a model evicts whatever else is on that card, and nothing ever
// arrives on a card behind your back.
//
// A multi-GPU span is always exclusive regardless -- that predates this and is
// independent (it needs the whole set to tensor-split across).
bool server_models::model_wants_exclusive(const server_model_meta & meta) {
    std::string v;
    if (meta.preset.get_option(ROUTER_ARG_EXCLUSIVE, v)) {
        return common_arg_utils::is_truthy(v);
    }
    return true;
}

// Parse preset idle-timeout into meta.idle_timeout.
// -1 = inherit router --models-idle-timeout; 0 = never; >0 = seconds.
static void parse_model_idle_timeout(server_model_meta & meta) {
    meta.idle_timeout = -1;
    std::string val;
    if (!meta.preset.get_option(ROUTER_ARG_IDLE_TIMEOUT, val) || val.empty()) {
        return;
    }
    try {
        meta.idle_timeout = std::stoi(val);
        if (meta.idle_timeout < 0) {
            SRV_WRN("invalid idle-timeout '%s' for model '%s' (negative); inheriting global default\n",
                    val.c_str(), meta.name.c_str());
            meta.idle_timeout = -1;
        }
    } catch (...) {
        SRV_WRN("invalid idle-timeout '%s' for model '%s'; inheriting global default\n",
                val.c_str(), meta.name.c_str());
        meta.idle_timeout = -1;
    }
}

// Effective idle seconds for the sweeper: per-model override, else global.
// 0 means never idle-unload.
static int effective_idle_timeout_s(const server_model_meta & meta, int global_timeout_s) {
    return meta.idle_timeout >= 0 ? meta.idle_timeout : global_timeout_s;
}

// Parse preset vram-mb into meta.placement.vram_mb_override (in MiB).
// -1 = unset, fall back to spawning the estimate child.
static void parse_model_vram_mb(server_model_meta & meta) {
    meta.placement.vram_mb_override = -1;
    std::string val;
    if (!meta.preset.get_option(ROUTER_ARG_VRAM_MB, val) || val.empty()) {
        return;
    }
    try {
        const int64_t mb = std::stoll(val);
        if (mb <= 0) {
            SRV_WRN("invalid vram-mb '%s' for model '%s' (must be positive); using estimate instead\n",
                    val.c_str(), meta.name.c_str());
            return;
        }
        meta.placement.vram_mb_override = mb;
    } catch (...) {
        SRV_WRN("invalid vram-mb '%s' for model '%s'; using estimate instead\n",
                val.c_str(), meta.name.c_str());
    }
}

// Parse preset ram-mb into meta.placement.ram_mb_override (in MiB).
// -1 = unset: the RAM gate is skipped for this model (its host footprint is unknown).
static void parse_model_ram_mb(server_model_meta & meta) {
    meta.placement.ram_mb_override = -1;
    std::string val;
    if (!meta.preset.get_option(ROUTER_ARG_RAM_MB, val) || val.empty()) {
        return;
    }
    try {
        const int64_t mb = std::stoll(val);
        if (mb <= 0) {
            SRV_WRN("invalid ram-mb '%s' for model '%s' (must be positive); RAM gate disabled for it\n",
                    val.c_str(), meta.name.c_str());
            return;
        }
        meta.placement.ram_mb_override = mb;
    } catch (...) {
        SRV_WRN("invalid ram-mb '%s' for model '%s'; RAM gate disabled for it\n",
                val.c_str(), meta.name.c_str());
    }
}

// Parse the preset's `env` into meta.env_overrides. Entries are comma-separated;
// "KEY=VALUE" sets, a leading '-' ("-KEY") removes. Malformed entries are dropped
// with a warning rather than failing the load: a typo in one tuning var should not
// take a model offline.
static void parse_model_env(server_model_meta & meta) {
    meta.env_overrides.clear();
    std::string val;
    if (!meta.preset.get_option(ROUTER_ARG_ENV, val) || val.empty()) {
        return;
    }
    for (auto entry : string_split<std::string>(val, ',')) {
        entry = string_strip(entry);
        if (entry.empty()) {
            continue;
        }
        if (entry[0] == '-') {
            const std::string key = string_strip(entry.substr(1));
            // A bare "-" or a key with '=' in it is not a removal request.
            if (key.empty() || key.find('=') != std::string::npos) {
                SRV_WRN("ignoring malformed env removal '%s' for model '%s' (expected -KEY)\n",
                        entry.c_str(), meta.name.c_str());
                continue;
            }
            meta.env_overrides.push_back("-" + key);
            continue;
        }
        const size_t eq = entry.find('=');
        if (eq == std::string::npos || eq == 0) {
            SRV_WRN("ignoring malformed env entry '%s' for model '%s' (expected KEY=VALUE or -KEY)\n",
                    entry.c_str(), meta.name.c_str());
            continue;
        }
        meta.env_overrides.push_back(entry);
    }
}

void server_models::parse_model_placement(server_model_meta & meta) {
    meta.placement = {};
    std::string pinned;
    if (meta.preset.get_option(ROUTER_ARG_PINNED, pinned)) {
        meta.placement.pinned = common_arg_utils::is_truthy(pinned);
    }

    // idle-timeout is not placement, but it lives in the same preset and is
    // re-parsed whenever placement is (load + hot reload paths).
    parse_model_idle_timeout(meta);

    // priority: same capture-now hazard (stripped by unset_reserved_args())
    meta.priority = ADMISSION_PRIORITY_MIDDLE;
    {
        std::string prio;
        if (meta.preset.get_option(ROUTER_ARG_PRIORITY, prio) && !string_strip(prio).empty()) {
            const auto p = admission_priority_parse(string_strip(prio));
            if (p.has_value()) {
                meta.priority = *p;
            } else {
                SRV_WRN("invalid priority '%s' for model '%s' (expected highest, middle or lowest); using middle\n",
                        prio.c_str(), meta.name.c_str());
            }
        }
    }

    // vram-mb, likewise: capture it NOW, while the preset still has it. update_args()
    // calls unset_reserved_args() immediately after every parse_model_placement() call
    // site, which strips ROUTER_ARG_VRAM_MB from the preset in place -- so reading it
    // later (as estimate_need_bytes() used to) always missed and silently fell through
    // to the estimate child. Same hazard as idle-timeout; see the comment in reload().
    // Parsed before the `gpu.empty() || "any"` early-return below so that models without
    // an explicit slot still get their override.
    parse_model_vram_mb(meta);

    // ram-mb: same capture-now hazard as vram-mb.
    parse_model_ram_mb(meta);

    // Same capture-now hazard as vram-mb above: unset_reserved_args() strips
    // ROUTER_ARG_ENV from the preset in place right after this runs.
    parse_model_env(meta);

    // Model-group keys: same capture-now hazard (kind/depends/launch/... are stripped by
    // unset_reserved_args()). slot-autosave is NOT stripped: it is a real llama-server flag
    // and reaches the spine child through the preset as LLAMA_ARG_SLOT_AUTOSAVE.
    {
        const router_group_section sec = router_group_parse_section(meta.preset, meta.name);
        meta.kind              = sec.kind;
        meta.depends           = sec.depends;
        meta.launch            = sec.launch;
        meta.machine           = sec.machine;
        meta.gpu               = sec.gpu;
        meta.park_file         = sec.park_file;
        meta.park_mode         = sec.park_mode;
        meta.slot_autosave     = sec.slot_autosave;
        meta.startup_timeout_s = sec.startup_timeout_s;
        meta.worker_port       = sec.worker_port;
    }

    std::string gpu;
    if (meta.preset.get_option(ROUTER_ARG_GPU, gpu)) {
        gpu = string_strip(gpu);
    }
    if (gpu.empty() || gpu == "any") {
        return;
    }

    for (auto dev : string_split<std::string>(gpu, ',')) {
        dev = string_strip(dev);
        if (!dev.empty()) {
            meta.placement.devs.push_back(dev);
        }
    }
    meta.placement.exclusive = model_wants_exclusive(meta) || meta.placement.devs.size() > 1;

    std::string split_str;
    if (meta.preset.get_option("LLAMA_ARG_TENSOR_SPLIT", split_str)) {
        for (auto part : string_split<std::string>(split_str, ',')) {
            part = string_strip(part);
            if (!part.empty()) {
                meta.placement.split.push_back(std::stof(part));
            }
        }
    }
}

void server_models::validate_gpu_slots() {
    std::vector<std::string> args = { bin_path, "--list-devices" };
    std::vector<std::string> env = base_env;
    common_subproc proc;
    int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
    if (!proc.create(args, options, env)) {
        throw std::runtime_error("failed to spawn --list-devices for router GPU validation");
    }
    std::string output;
    FILE * stdout_file = proc.stdout_file();
    if (stdout_file) {
        char buffer[4096];
        while (fgets(buffer, sizeof(buffer), stdout_file) != nullptr) {
            output += buffer;
        }
    }
    const int exit_code = proc.join();
    if (exit_code != 0) {
        throw std::runtime_error("--list-devices validation child exited with status " + std::to_string(exit_code));
    }
    for (const auto & slot : gpu_slots) {
        if (output.find(slot.dev_name + ":") == std::string::npos) {
            throw std::runtime_error("configured GPU slot '" + slot.dev_name + "' was not found in --list-devices output");
        }
    }
}

int64_t server_models::read_physical_free_bytes(const server_gpu_slot & slot) const {
    return physical_free_from_used(slot, read_vram_used_bytes(slot));
}

int64_t server_models::physical_free_from_used(const server_gpu_slot & slot, int64_t used) const {
    if (used >= 0) {
        return std::max<int64_t>(0, slot.total_bytes - used);
    }
    SRV_WRN("failed to read VRAM probe '%s' for %s, trusting declared total\n",
            slot.vram_probe.c_str(), slot.dev_name.c_str());
    return slot.total_bytes;
}

// PIDs of the router's own model children. Their VRAM is accounted through the slot
// reservation, so they must be excluded from "foreign" usage (never counted twice).
std::set<int> server_models::router_child_pids_locked() const {
    std::set<int> pids;
    for (const auto & [_, inst] : mapping) {
        if (inst.subproc && !inst.subproc->stopped) {
            const int pid = inst.subproc->sproc.pid();
            if (pid > 0) {
                pids.insert(pid);
            }
        }
    }
    // group workers are router-owned too; their declared vram-mb is in the slot reservation
    for (const auto & [_, rt] : groups) {
        if (rt.workers) {
            const std::set<int> w = rt.workers->pids();
            pids.insert(w.begin(), w.end());
        }
    }
    return pids;
}

std::string server_models::group_spine_locked(const std::string & name) const {
    auto it = mapping.find(name);
    if (it != mapping.end() && it->second.meta.is_external() && !it->second.meta.group.empty()) {
        return it->second.meta.group;
    }
    return name;
}

bool server_models::same_group_locked(const std::string & a, const std::string & b) const {
    auto ia = mapping.find(a);
    auto ib = mapping.find(b);
    return ia != mapping.end() && ib != mapping.end() && !ia->second.meta.group.empty() &&
           ia->second.meta.group == ib->second.meta.group;
}

// One fdinfo scan + one router-PID set per top-level operation (a listing or a placement
// admission), shared by every slot it inspects. The scan walks all of /proc, so doing it
// per slot under the router mutex is wasteful. Skipped when no slot has a PCI address.
server_models::vram_snapshot server_models::take_vram_snapshot_locked() const {
    vram_snapshot snap;
    const bool any_pdev = std::any_of(gpu_slots.begin(), gpu_slots.end(),
        [](const server_gpu_slot & s) { return !s.pdev.empty(); });
    if (any_pdev) {
        snap.usage       = probe_fdinfo_vram("");
        snap.router_pids = router_child_pids_locked();
    }
    return snap;
}

int64_t server_models::foreign_vram_bytes_locked(const server_gpu_slot & slot, const vram_snapshot & snap) const {
    if (slot.pdev.empty()) {
        return 0;
    }
    return ledger_foreign_vram(snap.usage, snap.router_pids, slot.pdev);
}

int64_t server_models::free_ram_bytes_locked(const std::string & exclude) const {
    const int64_t headroom = (int64_t) base_params.router_ram_headroom_mb * 1024LL * 1024LL;
    int64_t free = ledger_free_ram(probe_mem_available(""), headroom);
    if (free < 0) {
        return -1;
    }
    // A model that is still loading has not faulted its host memory in yet, so
    // MemAvailable does not reflect it: hold its declared ram-mb back.
    for (const auto & [other, inst] : mapping) {
        if (other != exclude && inst.meta.status == SERVER_MODEL_STATUS_LOADING && inst.meta.placement.ram_mb_override > 0) {
            free -= inst.meta.placement.ram_mb_override * 1024LL * 1024LL;
        }
    }
    return std::max<int64_t>(0, free);
}

int64_t server_models::resident_ram_bytes_locked(const std::string & name) const {
    auto it = mapping.find(name);
    if (it == mapping.end()) {
        return 0;
    }
    const auto & inst = it->second;
    int pid = 0;
    if (inst.meta.is_external()) {
        // a worker's process belongs to its group runtime; pending / exited workers hold nothing
        auto g = groups.find(inst.meta.group);
        if (g != groups.end() && g->second.workers) {
            for (const auto & w : g->second.workers->status()) {
                if (w.name == name) {
                    if (w.state == "exited" || w.state == "pending") {
                        return 0;
                    }
                    pid = w.pid;
                }
            }
        }
    } else if (inst.subproc) {
        pid = inst.subproc->sproc.pid();
    }
    const auto mem = pid > 0 ? probe_proc_mem("", pid) : std::nullopt;
    if (mem.has_value()) {
        return mem->rss_anon + mem->rss_shmem;
    }
    if (inst.meta.placement.ram_mb_override > 0) {
        return inst.meta.placement.ram_mb_override * 1024LL * 1024LL;
    }
    return 0;
}

int64_t server_models::effective_free_bytes_locked(const server_gpu_slot & slot, const vram_snapshot & snap, int64_t sysfs_used) const {
    const ledger_slot ls = { ledger_slot_id("", slot.dev_name), slot.pdev, slot.total_bytes, slot.reserved_bytes };
    return ledger_free_vram(ls, foreign_vram_bytes_locked(slot, snap), sysfs_used);
}

int64_t server_models::effective_free_bytes_locked(const server_gpu_slot & slot, const vram_snapshot & snap) const {
    return effective_free_bytes_locked(slot, snap, read_vram_used_bytes(slot));
}

json server_models::gpu_slots_json() {
    std::lock_guard<std::mutex> lk(mutex);
    json out = json::array();
    const vram_snapshot snap = take_vram_snapshot_locked();
    for (const auto & slot : gpu_slots) {
        const int64_t foreign = foreign_vram_bytes_locked(slot, snap);
        const int64_t used    = read_vram_used_bytes(slot); // read once for both fields below
        out.push_back({
            {"name", slot.dev_name},
            {"id", ledger_slot_id("", slot.dev_name)},
            {"board_resource", board_gpu_resource(slot.dev_name)},
            {"total_bytes", slot.total_bytes},
            {"reserved_bytes", slot.reserved_bytes},
            {"physical_free_bytes", physical_free_from_used(slot, used)},
            {"foreign_mb", foreign / (1024 * 1024)},
            {"free_mb", effective_free_bytes_locked(slot, snap, used) / (1024 * 1024)},
            {"exclusive_holder", slot.exclusive_holder},
        });
    }
    return out;
}

json server_models::machines_json() {
    std::lock_guard<std::mutex> lk(mutex);
    const int64_t ram_free = free_ram_bytes_locked("");
    return json::array({ json{
        {"name", "local"},
        {"ram_available_mb", probe_mem_available("") < 0 ? -1 : probe_mem_available("") / (1024 * 1024)},
        {"ram_headroom_mb", base_params.router_ram_headroom_mb},
        {"ram_free_mb", ram_free < 0 ? -1 : ram_free / (1024 * 1024)},
    } });
}

static int find_slot_index(const std::vector<server_gpu_slot> & slots, const std::string & dev) {
    for (size_t i = 0; i < slots.size(); ++i) {
        if (slots[i].dev_name == dev) {
            return (int) i;
        }
    }
    return -1;
}

void server_models::credit_gpu_reservation_locked(const std::string & name) {
    auto it = mapping.find(name);
    if (it == mapping.end()) {
        return;
    }
    auto & placement = it->second.meta.placement;
    for (size_t i = 0; i < placement.devs.size() && i < placement.need_bytes_per_dev.size(); ++i) {
        const int slot_idx = find_slot_index(gpu_slots, placement.devs[i]);
        if (slot_idx < 0) {
            continue;
        }
        auto & slot = gpu_slots[slot_idx];
        slot.reserved_bytes = std::max<int64_t>(0, slot.reserved_bytes - placement.need_bytes_per_dev[i]);
        if (slot.exclusive_holder == name) {
            slot.exclusive_holder.clear();
        }
    }
    placement.need_bytes_per_dev.clear();
}

void server_models::reserve_gpu_placement_locked(const std::string & name, const server_model_placement & placement) {
    for (size_t i = 0; i < placement.devs.size() && i < placement.need_bytes_per_dev.size(); ++i) {
        const int slot_idx = find_slot_index(gpu_slots, placement.devs[i]);
        GGML_ASSERT(slot_idx >= 0);
        auto & slot = gpu_slots[slot_idx];
        // This is the ONLY unclamped mutation of reserved_bytes -- every release path
        // clamps at 0 -- so a negative need silently poisons the ledger and disables
        // admission control. Refuse it here rather than trusting every producer.
        const int64_t need = placement.need_bytes_per_dev[i];
        if (need < 0) {
            SRV_WRN("negative need_bytes %" PRId64 " for model %s on %s -- ignoring "
                    "(this is a bug in the placement estimate, not a config error)\n",
                    need, name.c_str(), slot.dev_name.c_str());
            continue;
        }
        slot.reserved_bytes += need;
        if (placement.exclusive) {
            slot.exclusive_holder = name;
        }
    }
}

void server_models::reconcile_gpu_reservation_locked(const std::string & name) {
    auto it = mapping.find(name);
    if (it == mapping.end() || it->second.meta.placement.devs.empty()) {
        return;
    }
    for (size_t i = 0; i < it->second.meta.placement.devs.size(); ++i) {
        const int slot_idx = find_slot_index(gpu_slots, it->second.meta.placement.devs[i]);
        if (slot_idx < 0) {
            continue;
        }
        const auto & slot = gpu_slots[slot_idx];
        SRV_INF("router GPU ledger name=%s dev=%s reserved=%" PRId64 " physical_free=%" PRId64 "\n",
                name.c_str(), slot.dev_name.c_str(), slot.reserved_bytes, read_physical_free_bytes(slot));
    }
}

// Single source of truth for "what affects the VRAM estimate": every env key
// listed here is (a) the only set applied to params inside
// estimate_need_bytes_key(), via apply_to_params(), and (b) folded verbatim
// (as its raw preset string value) into the cache key. A key that is applied
// but not hashed -- or vice versa -- was exactly the bug behind the
// tiel-35b-par3 stale-estimate incident (kv-tiered/checkpoints/cache-ram were
// applied to params but never reached the key), so keep this the one list
// either side reads from; do not hand-maintain two lists.
static const std::vector<std::string> k_estimate_env_keys = {
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_CTX_SIZE",
    "LLAMA_ARG_CACHE_TYPE_K",
    "LLAMA_ARG_CACHE_TYPE_V",
    "LLAMA_ARG_N_PARALLEL",
    "LLAMA_ARG_KV_TIERED",
    "LLAMA_ARG_CTX_CHECKPOINTS",
    "LLAMA_ARG_CACHE_RAM",
};

std::string server_models::estimate_need_bytes_key(const server_model_meta & meta) {
    common_params params;
    meta.preset.apply_to_params(params,
            std::set<std::string>(k_estimate_env_keys.begin(), k_estimate_env_keys.end()));
    std::string model_path = params.model.path;
    int64_t mtime = 0;
    if (!model_path.empty() && std::filesystem::exists(model_path)) {
        mtime = (int64_t) std::filesystem::last_write_time(model_path).time_since_epoch().count();
    }
    // "v2|" invalidates every estimate cached under the pre-fix key (which
    // silently ignored kv-tiered/checkpoints/cache-ram).
    std::string k = string_format("v2|%s|%" PRId64 "|%d|%d|%d|%d",
            model_path.c_str(), mtime, params.n_ctx, (int) params.cache_type_k,
            (int) params.cache_type_v, params.n_parallel);
    // Fold in the raw preset value (not the parsed params field) for every
    // key in k_estimate_env_keys, so a new option added to that list is
    // automatically covered here too without touching this code.
    for (const auto & env : k_estimate_env_keys) {
        std::string val;
        if (meta.preset.get_option(env, val)) {
            k += "|" + env + "=" + val;
        }
    }
    return k;
}

std::vector<int64_t> server_models::estimate_need_bytes(const server_model_meta & meta) {
    // Read the value captured by parse_model_placement(), NOT meta.preset: update_args()
    // strips ROUTER_ARG_VRAM_MB from the preset in place at registration, so by the time
    // we get here the option is always gone and every model silently fell through to the
    // estimate child below -- vram-mb was never honoured at all.
    if (meta.placement.vram_mb_override >= 0) {
        return { meta.placement.vram_mb_override * 1024LL * 1024LL };
    }

    const std::string key = estimate_need_bytes_key(meta);

    std::filesystem::path cache_dir = std::filesystem::temp_directory_path() / "llama-router-estimates";
    std::filesystem::create_directories(cache_dir);
    const std::filesystem::path cache_file = cache_dir / std::to_string(std::hash<std::string>{}(key));
    {
        std::ifstream in(cache_file);
        if (in.good()) {
            // upstream #27511: common_json parses a string and has no istream overload
            const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            json data = json::parse_no_throw(text);
            if (data.is_array()) {
                std::vector<int64_t> cached;
                for (const auto & v : data) {
                    cached.push_back(v.get<int64_t>());
                }
                if (!cached.empty()) {
                    SRV_INF("estimate cache HIT for model %s: key=%s\n", meta.name.c_str(), key.c_str());
                    return cached;
                }
            }
        }
    }
    SRV_INF("estimate cache MISS for model %s: key=%s (measuring fresh)\n", meta.name.c_str(), key.c_str());

    server_model_meta est = meta;
    est.update_args(ctx_preset, bin_path);
    std::vector<std::string> child_args = est.args;
    std::vector<std::string> child_env  = base_env;
    child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));
    child_env.push_back("LLAMA_SERVER_CHILD_MODE=estimate");

    common_subproc proc;
    int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
    if (!proc.create(child_args, options, child_env)) {
        throw std::runtime_error("failed to spawn estimate child for model " + meta.name);
    }

    std::vector<int64_t> result;
    FILE * stdout_file = proc.stdout_file();
    if (stdout_file) {
        char buffer[128 * 1024];
        while (fgets(buffer, sizeof(buffer), stdout_file) != nullptr) {
            LOG("[estimate:%s] %s", meta.name.c_str(), buffer);
            std::string line(buffer);
            if (!string_starts_with(line.c_str(), CMD_CHILD_TO_ROUTER_STATE)) {
                continue;
            }
            json data = json::parse_no_throw(line.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
            if (data.is_discarded()) {
                continue;
            }
            json payload = json_value(data, "payload", json{});
            if (payload.contains("need_bytes_per_dev") && payload["need_bytes_per_dev"].is_array()) {
                for (const auto & v : payload["need_bytes_per_dev"]) {
                    result.push_back(v.get<int64_t>());
                }
            }
        }
    }
    const int exit_code = proc.join();
    if (exit_code != 0 || result.empty()) {
        throw std::runtime_error("estimate child failed for model " + meta.name);
    }
    {
        std::ofstream out(cache_file);
        out << safe_json_to_str(json(result));
    }
    return result;
}

// Thrown out of placement when the load has to wait (a foreign board claim, a busy resident);
// load() catches it once its own lock is gone and records the queued load (on_queued()).
struct router_queue_signal {
    admission_result                         res;
    std::optional<router_board_claim_result> raced; // the claim POST that found the resource taken (already queued on it)
};

admission_result server_models::decide_admission_locked(const std::string & name, const server_model_meta & meta,
                                                        std::vector<admission_candidate> candidates, bool exclusive,
                                                        const load_options & opts) {
    bool need_ram = false;
    std::set<std::string> cand_slots;
    for (const auto & c : candidates) {
        for (const auto & r : c.ram) {
            need_ram = need_ram || r.bytes > 0;
        }
        for (const auto & v : c.vram) {
            cand_slots.insert(v.slot);
        }
    }

    // slot ids are the bare device names while the router is local-only (ledger_slot_id("", dev))
    admission_input in;
    in.alias        = name;
    in.group        = meta.group;
    in.priority     = effective_priority(opts.req, meta);
    in.machine      = admission_machine(opts.req.machine);
    in.exclusive    = exclusive;
    in.margin_bytes = ROUTER_GPU_MARGIN_BYTES;
    in.candidates   = std::move(candidates);

    // An exclusive load's VRAM is not gated, so it skips the /proc scan; otherwise one scan
    // for this whole admission, shared by every candidate slot.
    const bool gate_vram = !exclusive && !cand_slots.empty();
    const vram_snapshot snap = gate_vram ? take_vram_snapshot_locked() : vram_snapshot{};
    for (const auto & slot : gpu_slots) {
        const bool read = gate_vram && cand_slots.count(slot.dev_name) > 0;
        in.slots.push_back({ slot.dev_name, "", board_gpu_resource(slot.dev_name), read ? effective_free_bytes_locked(slot, snap) : 0 });
    }
    in.machines.push_back({ "", ROUTER_BOARD_RAM_RESOURCE, need_ram ? free_ram_bytes_locked(name) : -1 });

    for (const auto & [other, inst] : mapping) {
        if (other == name || same_group_locked(other, name) || !inst.meta.is_running()) {
            continue;
        }
        const auto & p = inst.meta.placement;
        bool on_candidate = false;
        for (const auto & dev : p.devs) {
            on_candidate = on_candidate || cand_slots.count(dev) > 0;
        }
        // a sleeping model is only cleared off the cards of an exclusive load
        if (inst.meta.status == SERVER_MODEL_STATUS_SLEEPING && !(exclusive && on_candidate)) {
            continue;
        }
        admission_resident r;
        r.name  = other;
        r.group = inst.meta.group; // spine: own name; worker: its spine -> evicted as one group
        for (size_t i = 0; i < p.devs.size(); ++i) {
            const int64_t bytes = i < p.need_bytes_per_dev.size() ? p.need_bytes_per_dev[i]
                                : p.need_bytes_per_dev.empty() ? 0 : p.need_bytes_per_dev.front();
            r.vram.push_back({ p.devs[i], bytes });
            const int idx = find_slot_index(gpu_slots, p.devs[i]);
            r.exclusive = r.exclusive || (idx >= 0 && gpu_slots[idx].exclusive_holder == other);
        }
        // host RAM is only probed when this load needs some
        r.ram       = { { "", need_ram ? resident_ram_bytes_locked(other) : 0 } };
        r.last_used = inst.meta.last_used;
        r.busy      = inst.req_count > 0;
        r.pinned    = p.pinned;
        r.held      = is_held_locked(other); // a worker is held through its spine
        in.residents.push_back(std::move(r));
    }
    if (board) {
        // the cached board state (no HTTP here); an unavailable board gives no claims
        in.claims = board->admission_claims(machine_resources_locked());
    }

    admission_result res = decide_admission(in);
    if (res.verdict != ADMISSION_QUEUE) {
        return res;
    }

    SRV_INF("router admission for %s at %s: queue (blocked=%s by='%s' on='%s')\n", name.c_str(),
            admission_priority_str(in.priority), admission_block_str(res.blocked), res.blocked_by.c_str(), res.blocked_on.c_str());
    if (router_admission_queue_action(res) == ROUTER_QUEUE_WAIT) {
        throw router_queue_signal{ res }; // load() records the queued load (lock released)
    }
    // never fits as configured, or a pin / hold is in the way: refuse now
    if (res.blocked_on_ram) {
        int64_t need = 0;
        for (const auto & r : in.candidates[res.candidate].ram) {
            need += std::max<int64_t>(0, r.bytes);
        }
        throw router_refused_error("not enough host RAM for model '" + name + "': needs " +
                                   std::to_string(need / (1024 * 1024)) + " MB, free " +
                                   std::to_string(std::max<int64_t>(0, free_ram_bytes_locked(name)) / (1024 * 1024)) +
                                   " MB after headroom, and no idle model can be evicted to make room");
    }
    switch (res.blocked) {
        case ADMISSION_BLOCK_PINNED:
            throw router_refused_error("model '" + name + "' cannot load: GPU is held by pinned model '"
                                       + res.blocked_by + "' (unpin it first)");
        case ADMISSION_BLOCK_HELD:
            throw router_refused_error("model '" + name + "' cannot load: GPU is held by model '"
                                       + res.blocked_by + "' (hold lease)");
        case ADMISSION_BLOCK_NO_CANDIDATE:
            throw router_refused_error("model '" + name + "' cannot load on machine '" + opts.req.machine
                                       + "': none of its placements is there");
        default:
            throw router_refused_error("no configured GPU slot has enough capacity for model '" + name + "'");
    }
}

admission_result server_models::admit_locked(const std::string & name, const server_model_meta & meta,
                                             const std::vector<admission_candidate> & candidates, bool exclusive,
                                             const load_options & opts, std::unique_lock<std::mutex> & lk) {
    // Claiming releases the lock, so whatever changed meanwhile (another load placed on the same
    // slot, a resident gone) is decided again; the claims just taken are ours and never block.
    // Bounded: if the picture keeps moving, go ahead with what is held (the board agent takes
    // the rest for the resident later, and releases what it does not need).
    admission_result res;
    bool claim = true;
    for (int attempt = 0; ; ++attempt) {
        res = decide_admission_locked(name, meta, candidates, exclusive, opts);
        if (!claim || attempt >= 2) {
            break;
        }
        const int rc = take_board_claims_locked(name, res.board_claims_to_take, opts, lk);
        if (rc == 0) {
            break; // nothing to claim: the decision stands
        }
        claim = rc == 1; // a claim the board refused is not retried here (the agent retries it later)
    }
    return res;
}

int server_models::take_board_claims_locked(const std::string & owner, const std::vector<std::string> & resources,
                                            const load_options & opts, std::unique_lock<std::mutex> & lk) {
    if (!board || resources.empty()) {
        return 0;
    }
    const std::vector<std::string> missing = board->missing(owner, resources);
    if (missing.empty()) {
        return 0;
    }
    if (!board->available()) {
        // board outage: load as if nothing were claimed; the agent claims for the resident later
        SRV_WRN("board unavailable: loading %s without claiming %zu resource(s)\n", owner.c_str(), missing.size());
        return 0;
    }
    const std::string queue_owner = group_spine_locked(owner); // the load that waits (a group: its spine)
    auto mit = mapping.find(owner);
    const admission_priority prio = mit != mapping.end() ? effective_priority(opts.req, mit->second.meta)
                                                         : (opts.req.priority_set ? opts.req.priority : ADMISSION_PRIORITY_MIDDLE);

    lk.unlock();
    const router_board_claim_result * raced = nullptr;
    router_board_claim_result queued_result;
    std::string raced_resource;
    bool failed = false;
    for (const auto & r : missing) {
        router_board_claim_result cr = board->acquire(owner, queue_owner, r, owner, prio);
        if (cr.ok && cr.granted) {
            SRV_INF("board: claimed %s for %s (claim %s)\n", r.c_str(), owner.c_str(), cr.claim_id.c_str());
        } else if (cr.ok && cr.queued) {
            // someone claimed it after the last poll: this load waits behind them
            queued_result  = cr;
            raced          = &queued_result;
            raced_resource = r;
            break;
        } else {
            SRV_WRN("board: could not claim %s for %s (%s); loading without the claim\n", r.c_str(), owner.c_str(), cr.error.c_str());
            failed = true;
        }
    }
    lk.lock();
    if (raced != nullptr) {
        admission_result q;
        q.verdict    = ADMISSION_QUEUE;
        q.blocked    = ADMISSION_BLOCK_CLAIM;
        q.blocked_by = raced->held_by;
        q.blocked_on = raced_resource;
        throw router_queue_signal{ q, *raced };
    }
    return failed ? 2 : 1;
}

void server_models::evict_and_wait_locked(const std::string & name, const std::vector<std::string> & victims, std::unique_lock<std::mutex> & lk) {
    for (const auto & victim : victims) {
        SRV_INF("router placement: evicting %s to make room for %s\n", victim.c_str(), name.c_str());
        auto it = mapping.find(victim);
        notify_state("evicting", victim, it != mapping.end() ? it->second.meta.placement.devs : std::vector<std::string>{},
                     "evicted to make room for " + name);
        const bool loading = it != mapping.end() && it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
        if (loading) {
            it->second.subproc->terminate();
        }
        // marks the victim stopping and hands the stop to the monitor (upstream #28555); a group
        // victim is named after its spine, which stops the whole group
        request_stop(victim, !loading);
    }
    if (victims.empty()) {
        return;
    }
    cv.wait(lk, [&]() {
        for (const auto & victim : victims) {
            auto it = mapping.find(victim);
            if (it != mapping.end() && it->second.meta.is_running()) {
                return false;
            }
        }
        return true;
    });
}

void server_models::ensure_gpu_placement(const std::string & name, server_model_meta & meta, const load_options & opts, std::unique_lock<std::mutex> & lk) {
    const server_child_mode mode = opts.mode;
    if (!gpu_placement_enabled) {
        if (mode == SERVER_CHILD_MODE_NORMAL && !meta.placement.devs.empty()) {
            throw std::runtime_error("model '" + name + "' uses router gpu= but no router GPU slot table is configured");
        }
        return;
    }
    if (mode != SERVER_CHILD_MODE_NORMAL) {
        return;
    }

    // Host RAM must fit too: a shortfall evicts idle residents (LRU) just like VRAM does.
    std::vector<admission_machine_bytes> ram_need;
    if (meta.placement.ram_mb_override > 0) {
        ram_need = { { "", meta.placement.ram_mb_override * 1024LL * 1024LL } };
    }

    if (meta.is_external() && meta.placement.devs.empty()) {
        // a worker without gpu= holds no GPU slot; only its host RAM is gated
        if (!ram_need.empty()) {
            admission_candidate c;
            c.ram = ram_need;
            const admission_result res = admit_locked(name, meta, { c }, false, opts, lk);
            evict_and_wait_locked(name, res.victims, lk);
        }
        return;
    }

    if (meta.placement.devs.empty()) {
        for (const auto & slot : gpu_slots) {
            meta.placement.devs.push_back(slot.dev_name);
        }
    }
    for (const auto & dev : meta.placement.devs) {
        if (find_slot_index(gpu_slots, dev) < 0) {
            throw std::runtime_error("model '" + name + "' references unknown GPU slot '" + dev + "'");
        }
    }
    // NOTE: this must OR with the parsed option, not overwrite it -- otherwise a
    // single-GPU model silently loses its exclusivity here and the ledger is free to
    // co-locate something alongside it.
    meta.placement.exclusive = model_wants_exclusive(meta) || meta.placement.devs.size() > 1;

    std::vector<int64_t> needs;
    if (meta.is_external()) {
        // a worker has no model to estimate: its declared vram-mb is what it reserves
        if (meta.placement.vram_mb_override < 0) {
            SRV_WRN("worker '%s' has gpu= but no vram-mb; reserving nothing on its GPU\n", name.c_str());
        }
        needs = { std::max<int64_t>(0, meta.placement.vram_mb_override) * 1024LL * 1024LL };
    } else {
        lk.unlock();
        try {
            needs = estimate_need_bytes(meta);
        } catch (...) {
            lk.lock();
            throw;
        }
        lk.lock();
    }

    if (meta.placement.exclusive) {
        // Split weights only mean something across a MULTI-GPU span. Single-GPU models are
        // exclusive by default now, so gate ONLY this on the span -- everything below
        // (need_bytes, eviction, reservation) must still run for a one-card model, or
        // exclusivity silently stops evicting and two models land on the same GPU.
        if (meta.placement.split.empty() && meta.placement.devs.size() > 1) {
            for (const auto & dev : meta.placement.devs) {
                const int slot_idx = find_slot_index(gpu_slots, dev);
                meta.placement.split.push_back((float) gpu_slots[slot_idx].total_bytes);
            }
        }
        if (needs.size() == 1) {
            const int64_t total_need = needs[0];
            int64_t total_weight = 0;
            for (const auto & dev : meta.placement.devs) {
                total_weight += gpu_slots[find_slot_index(gpu_slots, dev)].total_bytes;
            }
            meta.placement.need_bytes_per_dev.clear();
            for (const auto & dev : meta.placement.devs) {
                const auto & slot = gpu_slots[find_slot_index(gpu_slots, dev)];
                // __int128 intermediate is REQUIRED, not defensive: both factors are byte
                // counts in the 1e10 range, so total_need * total_bytes lands around 1e21
                // and overflows int64 (max 9.2e18) by ~100x for any real model on any real
                // card. That silently produced garbage shares -- including NEGATIVE ones,
                // which then flowed into reserve_gpu_placement_locked() below and drove
                // slot.reserved_bytes negative, at which point effective_free_bytes_locked()
                // computes a ledger_free LARGER than the card and admission control stops
                // binding entirely.
                const __int128 share = ((__int128) total_need * (__int128) slot.total_bytes)
                                       / (__int128) std::max<int64_t>(1, total_weight);
                meta.placement.need_bytes_per_dev.push_back((int64_t) share);
            }
        } else {
            meta.placement.need_bytes_per_dev.assign(needs.begin(), needs.begin() + std::min(needs.size(), meta.placement.devs.size()));
        }

        // one candidate: every span card. Every other resident there goes (one model per GPU).
        admission_candidate c;
        for (size_t i = 0; i < meta.placement.devs.size(); ++i) {
            const int64_t bytes = i < meta.placement.need_bytes_per_dev.size() ? meta.placement.need_bytes_per_dev[i] : 0;
            c.vram.push_back({ meta.placement.devs[i], bytes });
        }
        c.ram = ram_need;
        const admission_result res = admit_locked(name, meta, { c }, true, opts, lk);
        reserve_gpu_placement_locked(name, meta.placement);
        evict_and_wait_locked(name, res.victims, lk);
        return;
    }

    // one candidate per listed slot; admission picks the one needing the fewest/cheapest
    // victims, then the most free VRAM
    const int64_t need = needs.empty() ? 0 : needs[0];
    std::vector<admission_candidate> candidates;
    for (const auto & dev : meta.placement.devs) {
        admission_candidate c;
        c.vram = { { dev, need } };
        c.ram  = ram_need;
        candidates.push_back(std::move(c));
    }
    const admission_result res = admit_locked(name, meta, candidates, false, opts, lk);
    GGML_ASSERT(res.slots.size() == 1);

    meta.placement.devs = res.slots;
    meta.placement.need_bytes_per_dev = { need };
    reserve_gpu_placement_locked(name, meta.placement);
    evict_and_wait_locked(name, res.victims, lk);
}

void server_models::notify_sse(const std::string & event, const std::string & model_id, const json & data) {
    std::unique_ptr<server_task_result_router> result = std::make_unique<server_task_result_router>();
    result->data = {
        {"model", model_id},
        {"event", event},
    };
    if (!data.is_null()) {
        result->data["data"] = data;
    }
    SRV_DBG("notifying SSE clients about event '%s' for model '%s': %s\n", event.c_str(), model_id.c_str(), safe_json_to_str(result->data).c_str());
    sse.broadcast(std::move(result));
}

void server_models::notify_state(const std::string & event, const std::string & name, const std::vector<std::string> & slots,
                                 const std::string & reason, const json & extra) {
    json data = json::object();
    data["model"]   = name;
    data["machine"] = local_machine;
    data["slots"]   = slots;
    data["reason"]  = reason;
    if (extra.is_object()) {
        for (const auto & [k, v] : extra.items()) {
            data[k] = v;
        }
    }
    notify_sse(event, name, data);
}

//
// coordination board, holds, queued loads
//

std::string server_models::board_gpu_resource(const std::string & dev) const {
    const int idx = find_slot_index(gpu_slots, dev);
    if (idx >= 0 && !gpu_slots[idx].board_name.empty()) {
        return "gpu:" + gpu_slots[idx].board_name;
    }
    return "gpu:" + dev;
}

std::vector<std::string> server_models::board_resources_locked(const server_model_meta & meta) const {
    std::vector<std::string> out;
    if (!gpu_placement_enabled) {
        return out; // no admission, no claims
    }
    if (!meta.placement.need_bytes_per_dev.empty()) {
        for (const auto & dev : meta.placement.devs) {
            const std::string r = board_gpu_resource(dev);
            if (std::find(out.begin(), out.end(), r) == out.end()) {
                out.push_back(r);
            }
        }
    }
    if (meta.placement.ram_mb_override > 0) {
        out.push_back(ROUTER_BOARD_RAM_RESOURCE);
    }
    return out;
}

std::vector<std::string> server_models::machine_resources_locked() const {
    std::vector<std::string> out;
    for (const auto & slot : gpu_slots) {
        out.push_back(board_gpu_resource(slot.dev_name));
    }
    out.push_back(ROUTER_BOARD_RAM_RESOURCE);
    return out;
}

bool server_models::is_held_locked(const std::string & name) const {
    return holds.is_held(group_spine_locked(name), ggml_time_ms());
}

admission_priority server_models::effective_priority(const router_request_opts & req, const server_model_meta & meta) const {
    if (req.priority_set) {
        return req.priority;
    }
    if (meta.is_external() && !meta.group.empty()) {
        auto it = mapping.find(meta.group); // a worker loads at its group's priority
        if (it != mapping.end()) {
            return it->second.meta.priority;
        }
    }
    return meta.priority;
}

std::string server_models::admission_machine(const std::string & requested) const {
    // only the local machine exists until node mode: its name (or "local") means "here"
    if (requested.empty() || requested == local_machine || router_machine_is_local(requested)) {
        return "";
    }
    return requested;
}

void server_models::bump_queue_locked() {
    queue_epoch++;
    cv.notify_all();
}

std::vector<router_board_resident> server_models::board_residents() {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<router_board_resident> out;
    for (const auto & [name, inst] : mapping) {
        const bool loading = loading_owners.count(name) > 0;
        if (!inst.meta.is_running() && !loading) {
            continue;
        }
        router_board_resident r;
        r.name    = name;
        r.alive   = true;
        r.loading = loading;
        if (inst.meta.is_running()) {
            r.resources = board_resources_locked(inst.meta);
        }
        // a worker goes (and is judged) with its spine
        const std::string spine = group_spine_locked(name);
        r.stop_name = spine;
        auto sit = mapping.find(spine);
        if (sit != mapping.end()) {
            const auto & sp = sit->second;
            r.idle   = sp.meta.is_ready_or_sleep() && sp.req_count == 0 && !stopping_models.count(spine) &&
                       !loading_owners.count(spine);
            r.pinned = sp.meta.placement.pinned;
        }
        r.held = is_held_locked(name);
        out.push_back(std::move(r));
    }
    return out;
}

void server_models::board_yield(const std::string & name, const std::string & reason) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    // checked again under the lock: the agent's view is a moment old
    if (it == mapping.end() || !it->second.meta.is_ready_or_sleep() || it->second.req_count > 0 ||
            stopping_models.count(name) || it->second.meta.placement.pinned || is_held_locked(name)) {
        return;
    }
    SRV_INF("board: unloading idle %s (%s)\n", name.c_str(), reason.c_str());
    notify_state("evicting", name, it->second.meta.placement.devs, reason);
    request_stop(name, true);
}

void server_models::queue_runner_loop() {
    uint64_t seen = 0;
    std::unique_lock<std::mutex> lk(mutex);
    while (!shutting_down) {
        // retry when something moved (board change, a resident went idle or down), else every 30 s
        cv.wait_for(lk, std::chrono::seconds(30), [&]() {
            return shutting_down || (queue_epoch != seen && !queued_loads.empty());
        });
        if (shutting_down) {
            break;
        }
        seen = queue_epoch;
        if (queued_loads.empty()) {
            continue;
        }
        std::vector<std::pair<std::string, router_request_opts>> todo;
        for (const auto & [n, q] : queued_loads) {
            todo.push_back({ n, q.req });
        }
        lk.unlock();
        for (const auto & [n, req] : todo) {
            load_options o;
            o.req = req;
            try {
                load(n, o);
            } catch (const router_queued_error &) {
                // still waiting
            } catch (const std::exception & e) {
                SRV_WRN("queued load of %s gave up: %s\n", n.c_str(), e.what());
            }
        }
        lk.lock();
    }
}

void server_models::on_queued(const std::string & name, const load_options & opts, const admission_result & res,
                              const router_board_claim_result * raced) {
    router_queued_info info;
    info.model      = name;
    info.machine    = opts.req.machine;
    info.board      = res.blocked == ADMISSION_BLOCK_CLAIM;
    info.blocked_by = res.blocked_by;
    info.blocked_on = res.blocked_on;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        info.priority = it != mapping.end() ? effective_priority(opts.req, it->second.meta)
                                            : (opts.req.priority_set ? opts.req.priority : ADMISSION_PRIORITY_MIDDLE);
    }
    info.reason = info.board ? info.blocked_on + " is claimed on the board by " + info.blocked_by
                             : "model '" + info.blocked_by + "' is busy";

    if (info.board && board) {
        // join the board queue (idempotent per resource: the board keeps one slot for the router)
        router_board_claim_result jr = raced != nullptr ? *raced
                                                        : board->acquire(name, name, res.blocked_on, name, info.priority);
        if (jr.ok && jr.queued) {
            info.queue_pos = jr.position;
            if (!jr.held_by.empty()) {
                info.blocked_by = jr.held_by;
            }
        } else if (jr.ok && jr.granted) {
            std::lock_guard<std::mutex> lk(mutex);
            bump_queue_locked(); // it freed meanwhile: try again right away
        }
        if (!res.notify.empty()) {
            const bool yield = res.notify.front().kind == ADMISSION_NOTIFY_YIELD;
            std::string content = "llama-router on " + local_machine + " needs " + res.blocked_on + " to load '" + name +
                                  "' (priority " + admission_priority_str(info.priority) + ") and is queued behind your claim.";
            if (yield) {
                content += " Please release it as soon as you safely can.";
            }
            board->notify(res.notify, content);
        }
    }

    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (shutting_down) {
            throw router_refused_error("router is shutting down");
        }
        if (!info.board) {
            // place among the router's own loads waiting on the same resident
            int pos = 1;
            for (const auto & [n, q] : queued_loads) {
                if (n != name && !q.info.board && q.info.blocked_by == info.blocked_by && q.info.priority >= info.priority) {
                    pos++;
                }
            }
            info.queue_pos = pos;
        }
        auto it = queued_loads.find(name);
        changed = it == queued_loads.end() || it->second.info.queue_pos != info.queue_pos ||
                  it->second.info.blocked_by != info.blocked_by || it->second.info.blocked_on != info.blocked_on ||
                  it->second.info.priority != info.priority;
        queued_loads[name] = { info, opts.req };
    }
    if (changed) {
        SRV_INF("load of %s queued: %s (position %d)\n", name.c_str(), info.reason.c_str(), info.queue_pos);
        json extra = json::parse(router_queued_info_json(info));
        extra["waiting"] = true;
        notify_state(info.board ? "queued" : "blocked", name, res.slots, info.reason, extra);
    }
    throw router_queued_error(info);
}

void server_models::drop_queued(const std::string & name, const std::string & reason) {
    bool had = false;
    {
        std::lock_guard<std::mutex> lk(mutex);
        had = queued_loads.erase(name) > 0;
    }
    if (!had) {
        return;
    }
    if (board) {
        board->leave_queue(name);
    }
    json extra = json::object();
    extra["waiting"] = false;
    notify_state("blocked", name, {}, reason, extra);
}

bool server_models::cancel_queued(const std::string & name) {
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (!queued_loads.count(name)) {
            return false;
        }
    }
    drop_queued(name, "queued load cancelled");
    return true;
}

void server_models::stop_threads() {
    {
        std::lock_guard<std::mutex> lk(mutex);
        shutting_down = true;
        cv.notify_all();
    }
    if (queue_th.joinable()) {
        queue_th.join();
    }
    std::vector<async_load> loads;
    {
        std::lock_guard<std::mutex> lk(async_mu);
        loads.swap(async_loads);
    }
    for (auto & l : loads) {
        if (l.th.joinable()) {
            l.th.join();
        }
    }
    if (board) {
        board->stop(); // releases every claim, leaves every queue it joined
    }
}

void server_models::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mutex);
        shutting_down = true;
        queued_loads.clear(); // waiters see their load gone; nothing retries it
        cv.notify_all();
    }
    if (queue_th.joinable()) {
        queue_th.join();
    }
    unload_all();
    stop_threads();
}

json server_models::load_async(const std::string & name, const router_request_opts & req) {
    struct outcome_t {
        std::mutex              m;
        std::condition_variable cv;
        bool                    done = false;
        std::exception_ptr      err;
    };
    auto outcome = std::make_shared<outcome_t>();
    auto done    = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lk(async_mu);
        for (auto it = async_loads.begin(); it != async_loads.end();) {
            if (it->done->load()) {
                it->th.join();
                it = async_loads.erase(it);
            } else {
                ++it;
            }
        }
        {
            std::lock_guard<std::mutex> l(mutex);
            if (shutting_down) {
                throw router_refused_error("router is shutting down");
            }
        }
        load_options o;
        o.req = req;
        std::thread th([this, name, o, outcome, done]() {
            std::exception_ptr err;
            try {
                load(name, o);
            } catch (...) {
                err = std::current_exception();
            }
            {
                std::lock_guard<std::mutex> l(outcome->m);
                outcome->done = true;
                outcome->err  = err;
            }
            outcome->cv.notify_all();
            done->store(true);
        });
        async_loads.push_back({ std::move(th), done });
    }
    {
        // long enough for a queue verdict or a refusal; an estimate / eviction / group start keeps going
        std::unique_lock<std::mutex> l(outcome->m);
        outcome->cv.wait_for(l, std::chrono::milliseconds(1500), [&]() { return outcome->done; });
        if (outcome->done && outcome->err) {
            std::rethrow_exception(outcome->err);
        }
    }
    json out = json::object();
    out["model"] = name;
    auto meta = get_meta(name);
    if (meta.has_value() && !meta->queue_info.is_null()) {
        for (const auto & [k, v] : meta->queue_info.items()) {
            out[k] = v;
        }
        return out;
    }
    out["state"] = meta.has_value() && meta->is_ready_or_sleep() ? "ready" : "loading";
    return out;
}

json server_models::hold(const std::string & model, int64_t ttl_s, const std::string & owner, const std::string & lease) {
    std::lock_guard<std::mutex> lk(mutex);
    std::string name;
    if (mapping.count(model)) {
        name = model;
    } else {
        for (const auto & [key, inst] : mapping) {
            if (inst.meta.aliases.count(model)) {
                name = key;
                break;
            }
        }
    }
    if (name.empty()) {
        throw std::out_of_range("model '" + model + "' not found");
    }
    name = group_spine_locked(name); // a group is held as a whole
    std::string err;
    const std::string l = holds.hold(name, ttl_s * 1000, owner, lease, ggml_time_ms(), err);
    if (l.empty()) {
        throw std::invalid_argument(err);
    }
    SRV_INF("hold %s on %s for %" PRId64 " s (owner '%s')\n", l.c_str(), name.c_str(), ttl_s, owner.c_str());
    json out = json::object();
    out["lease"] = l;
    out["model"] = name;
    out["ttl_s"] = ttl_s;
    out["owner"] = owner;
    return out;
}

bool server_models::release_hold(const std::string & lease) {
    std::lock_guard<std::mutex> lk(mutex);
    const bool ok = holds.release(lease, ggml_time_ms());
    if (ok) {
        SRV_INF("hold %s released\n", lease.c_str());
    }
    return ok;
}

json server_models::board_json() {
    json out = json::object();
    out["enabled"] = board != nullptr;
    out["machine"] = local_machine;
    if (board) {
        out["available"] = board->available();
        json claims = json::object();
        for (const auto & [owner, res] : board->held()) {
            claims[owner] = res;
        }
        out["claims"] = claims;
    }
    json hl = json::array();
    {
        std::lock_guard<std::mutex> lk(mutex);
        const int64_t now = ggml_time_ms();
        for (const auto & h : holds.list(now)) {
            json o = json::object();
            o["lease"]        = h.lease;
            o["model"]        = h.model;
            o["owner"]        = h.owner;
            o["expires_in_s"] = (h.expires_ms - now) / 1000;
            hl.push_back(o);
        }
    }
    out["holds"] = hl;
    return out;
}

void server_models::load_models() {
    // Phase 1: load presets from all sources - pure I/O, no lock needed
    std::optional<std::filesystem::file_time_type> models_preset_loaded_mtime;
    // 1. cached models
    common_presets cached_models = ctx_preset.load_from_cache();
    SRV_TRC("Loaded %zu cached model presets from %s\n", cached_models.size(), hf_cache::get_cache_path().c_str());
    // 2. local models from --models-dir
    common_presets local_models;
    if (!base_params.models_dir.empty()) {
        local_models = ctx_preset.load_from_models_dir(base_params.models_dir);
        SRV_TRC("Loaded %zu local model presets from %s\n", local_models.size(), base_params.models_dir.c_str());
    }
    // 3. custom-path models from presets
    common_preset global = {};
    common_presets custom_presets = {};
    if (!base_params.models_preset.empty()) {
        const auto mtime_before = get_models_preset_mtime();
        custom_presets = ctx_preset.load_from_ini(base_params.models_preset, global);
        const auto mtime_after = get_models_preset_mtime();
        if (mtime_before.has_value() && mtime_after.has_value() && *mtime_before == *mtime_after) {
            models_preset_loaded_mtime = *mtime_after;
        } else {
            SRV_WRN("models preset '%s' changed while loading; it will be retried before the next child spawn\n",
                base_params.models_preset.c_str());
        }
        SRV_TRC("Loaded %zu custom model presets from %s\n", custom_presets.size(), base_params.models_preset.c_str());
    }
    load_gpu_config(global);

    // cascade, apply global preset first
    cached_models  = ctx_preset.cascade(global, cached_models);
    local_models   = ctx_preset.cascade(global, local_models);
    custom_presets = ctx_preset.cascade(global, custom_presets);

    // note: if a model exists in both cached and local, local takes precedence
    common_presets final_presets;
    std::unordered_map<std::string, server_model_source> source_map;
    for (const auto & [name, preset] : cached_models) {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_CACHE;
    }
    for (const auto & [name, preset] : local_models)  {
        final_presets[name] = preset;
        source_map[name] = SERVER_MODEL_SOURCE_MODELS_DIR;
    }
    for (const auto & [name, custom] : custom_presets) {
        if (final_presets.find(name) != final_presets.end()) {
            final_presets[name].merge(custom);
        } else {
            final_presets[name] = custom;
        }
        source_map[name] = SERVER_MODEL_SOURCE_PRESET;
    }

    // overlay router's own CLI args on top of every model preset so that
    // e.g. `llama-server --temp 0` is honoured by all child processes
    for (auto & [name, preset] : final_presets) {
        preset.merge(base_preset);
    }

    // model groups: validate the whole preset set now so a bad group fails the load with a
    // message naming the section (throws std::runtime_error)
    std::map<std::string, std::string> worker_to_spine;
    {
        std::vector<router_group_section> sections;
        sections.reserve(final_presets.size());
        for (const auto & [name, preset] : final_presets) {
            sections.push_back(router_group_parse_section(preset, name));
        }
        worker_to_spine = router_groups_resolve(sections);
    }

    auto get_source = [&](const std::string & name) {
        return source_map.count(name) ? source_map.at(name) : SERVER_MODEL_SOURCE_PRESET;
    };

    // hide cache models whose resolved file is already used by a preset with dedup-cache-models enabled
    std::set<std::string> hidden_models;
    {
        std::set<std::string> preset_paths;
        auto add_hf_path = [&preset_paths](const common_preset & preset, const char * repo_key, const char * file_key) {
            std::string hf_repo;
            if (!preset.get_option(repo_key, hf_repo) || hf_repo.empty()) {
                return;
            }
            std::string hf_file;
            preset.get_option(file_key, hf_file);
            std::string path = common_download_resolve_path(hf_repo, hf_file);
            if (!path.empty()) {
                preset_paths.insert(path);
            }
        };
        for (const auto & [name, preset] : custom_presets) {
            std::string val;
            if (!preset.get_option(COMMON_ARG_PRESET_DEDUP_CACHE_MODELS, val) || !common_arg_utils::is_truthy(val)) {
                continue;
            }
            add_hf_path(preset, "LLAMA_ARG_HF_REPO", "LLAMA_ARG_HF_FILE");
            add_hf_path(preset, "LLAMA_ARG_SPEC_DRAFT_HF_REPO", "LLAMA_ARG_SPEC_DRAFT_MODEL");
        }
        if (!preset_paths.empty()) {
            for (const auto & [name, preset] : cached_models) {
                if (get_source(name) != SERVER_MODEL_SOURCE_CACHE) {
                    continue; // merged with another source, not a pure cache entry
                }
                std::string path = common_download_resolve_path(name);
                if (!path.empty() && preset_paths.count(path)) {
                    SRV_INF("hiding cache model name=%s (deduplicated by a preset)\n", name.c_str());
                    hidden_models.insert(name);
                }
            }
        }
    }

    // Helpers that read `mapping` - must be called while holding the lock.
    auto join_set = [](const std::set<std::string> & s) {
        std::string result;
        for (const auto & v : s) {
            if (!result.empty()) result += ", ";
            result += v;
        }
        return result;
    };
    auto log_available_models = [&]() {
        SRV_INF("Available models (%zu):\n", mapping.size());
        if (mapping.empty()) {
            SRV_INF("%s", "  no models found on the system (visit https://llama.app/models for suggestions)\n");
        } else {
            for (const auto & [name, inst] : mapping) {
                const std::string source = server_model_source_to_string(inst.meta.source);

                std::string info;
                if (!inst.meta.aliases.empty()) info += " (aliases: " + join_set(inst.meta.aliases) + ")";
                if (!inst.meta.tags.empty())    info += " [tags: "    + join_set(inst.meta.tags)    + "]";

                SRV_INF("  [%10s] %s%s\n", source.c_str(), name.c_str(), info.c_str());
            }
        }
    };
    auto apply_stop_timeout = [&]() {
        for (auto & [name, inst] : mapping) {
            std::string val;
            if (inst.meta.preset.get_option(COMMON_ARG_PRESET_STOP_TIMEOUT, val)) {
                try {
                    inst.meta.stop_timeout = std::stoi(val);
                } catch (...) {
                    SRV_WRN("invalid stop-timeout value '%s' for model '%s', using default %d seconds\n",
                        val.c_str(), name.c_str(), DEFAULT_STOP_TIMEOUT);
                    inst.meta.stop_timeout = DEFAULT_STOP_TIMEOUT;
                }
            }
        }
    };
    // Log only — do NOT re-parse here. parse_model_placement() already set
    // meta.idle_timeout before update_args() strips the router-only option
    // from the child preset. Re-parsing after that would always yield -1.
    auto log_idle_timeouts = [&]() {
        for (const auto & [name, inst] : mapping) {
            if (inst.meta.idle_timeout >= 0) {
                SRV_INF("  model '%s' idle-timeout=%ds (overrides global %ds)\n",
                        name.c_str(), inst.meta.idle_timeout, base_params.models_idle_timeout);
            }
        }
    };
    auto apply_groups = [&]() {
        for (auto & [name, inst] : mapping) {
            auto it = worker_to_spine.find(name);
            if (it != worker_to_spine.end()) {
                inst.meta.group = it->second;
            } else {
                inst.meta.group = inst.meta.depends.empty() ? std::string() : name;
            }
        }
    };
    auto apply_hidden = [&]() {
        for (auto & [name, inst] : mapping) {
            inst.meta.hidden = hidden_models.count(name) > 0;
        }
    };
    // update_args() injects HOST/PORT/ALIAS, so strip them before comparing presets
    auto preset_options_for_compare = [](common_preset p) {
        p.unset_option("LLAMA_ARG_HOST");
        p.unset_option("LLAMA_ARG_PORT");
        p.unset_option("LLAMA_ARG_ALIAS");
        return p.options;
    };

    // Phase 2: acquire the lock once for all mapping mutations.
    // We temporarily release it only when calling functions that acquire it internally (unload)
    std::unique_lock<std::mutex> lk(mutex);

    if (models_preset_loaded_mtime.has_value()) {
        models_preset_applied_mtimes.clear();
        for (const auto & entry : final_presets) {
            models_preset_applied_mtimes[entry.first] = *models_preset_loaded_mtime;
        }
    }
    need_reload = false;
    bool is_first_load = mapping.empty();

    if (is_first_load) {
        // FIRST LOAD: add all models, then unlock for autoloading
        for (const auto & [name, preset] : final_presets) {
            server_model_meta meta{
                /* source        */ get_source(name),
                /* preset        */ preset,
                /* name          */ name,
                /* aliases       */ {},
                /* tags          */ {},
                /* port          */ 0,
                /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                /* last_used     */ 0,
                /* args          */ std::vector<std::string>(),
                /* loaded_info   */ {},
                /* progress      */ {},
                /* exit_code     */ 0,
                /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                /* idle_timeout  */ -1,
                /* multimodal    */ mtmd_caps{false, false},
                /* placement     */ {},
                // /* need_download */ false,
            };
            add_model(std::move(meta));
        }
        apply_stop_timeout();
        log_idle_timeouts();
        apply_hidden();
        apply_groups();
        log_available_models();

        // skipped on reload, see startup_models
        if (startup_models.has_value()) {
            std::vector<std::string> models_to_load;
            for (const auto & [name, inst] : mapping) {
                std::string val;
                if (inst.meta.is_external()) {
                    continue; // workers are brought up with their group, never as a model
                }
                if (inst.meta.preset.get_option(COMMON_ARG_PRESET_LOAD_ON_STARTUP, val) && common_arg_utils::is_truthy(val)) {
                    models_to_load.push_back(name);
                }
            }
            if (!gpu_placement_enabled && (int)models_to_load.size() > base_params.models_max) {
                throw std::runtime_error(string_format(
                    "number of models to load on startup (%zu) exceeds models_max (%d)",
                    models_to_load.size(), base_params.models_max));
            }

            // to be lazy-loaded after main() setup phase is completed
            startup_models = std::move(models_to_load);
        }

        lk.unlock();
    } else {
        // RELOAD: diff the new preset list against the current mapping and reconcile
        is_reloading = true;

        // find running models whose source was removed or whose preset changed
        std::vector<std::string> to_unload;
        for (const auto & [name, inst] : mapping) {
            if (!inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) {
                to_unload.push_back(name); // removed from source
            } else if (preset_options_for_compare(inst.meta.preset) != preset_options_for_compare(it->second)) {
                to_unload.push_back(name); // preset changed
            }
        }

        // unload() acquires the lock internally, so release before each call
        for (const auto & name : to_unload) {
            SRV_INF("(reload) unloading model name=%s (source updated or removed)\n", name.c_str());
            lk.unlock();
            unload(name);
            lk.lock();
        }

        // wait for all targeted models to reach UNLOADED; cv.wait handles unlock/relock
        cv.wait(lk, [&]() {
            for (const auto & name : to_unload) {
                auto it = mapping.find(name);
                if (it != mapping.end() && it->second.meta.is_running()) return false;
            }
            return true;
        });

        // erase models no longer in any source
        for (auto it = mapping.begin(); it != mapping.end(); ) {
            if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                ++it; // download thread is still busy, skip
            } else if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADED) {
                // download finished, safe to erase
                it = mapping.erase(it);
            } else if (final_presets.find(it->first) == final_presets.end()) {
                SRV_INF("(reload) removing model name=%s (no longer in source)\n", it->first.c_str());
                it = mapping.erase(it);
            } else {
                ++it;
            }
        }

        // update presets for non-running models still in source
        for (auto & [name, inst] : mapping) {
            if (inst.meta.is_running()) continue;
            auto it = final_presets.find(name);
            if (it == final_presets.end()) continue; // erased above

            inst.meta.preset = it->second;
            parse_model_placement(inst.meta);

            // re-parse aliases, then validate against other models
            std::set<std::string> new_aliases;
            std::string alias_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_ALIAS", alias_str) && !alias_str.empty()) {
                for (auto & alias : string_split<std::string>(alias_str, ',')) {
                    alias = string_strip(alias);
                    if (!alias.empty()) new_aliases.insert(alias);
                }
            }
            inst.meta.aliases.clear();
            for (const auto & alias : new_aliases) {
                bool conflict = false;
                for (const auto & [other_name, other_inst] : mapping) {
                    if (other_name == name) continue;
                    if (other_name == alias || other_inst.meta.aliases.count(alias)) {
                        SRV_WRN("(reload) alias '%s' for model '%s' conflicts with model '%s', skipping\n",
                            alias.c_str(), name.c_str(), other_name.c_str());
                        conflict = true;
                        break;
                    }
                }
                if (!conflict) inst.meta.aliases.insert(alias);
            }

            // re-parse tags
            inst.meta.tags.clear();
            std::string tags_str;
            if (inst.meta.preset.get_option("LLAMA_ARG_TAGS", tags_str) && !tags_str.empty()) {
                for (auto & tag : string_split<std::string>(tags_str, ',')) {
                    tag = string_strip(tag);
                    if (!tag.empty()) inst.meta.tags.insert(tag);
                }
            }

            inst.meta.exit_code = 0; // clear failed state so the model can be reloaded
            inst.meta.update_args(ctx_preset, bin_path);
            if (!inst.meta.is_external()) {
                inst.meta.update_caps();
            }
        }

        // add models that are new in this reload, load-on-startup is not honored here since a
        // reload never spawns an instance
        for (const auto & [name, preset] : final_presets) {
            if (mapping.find(name) == mapping.end()) {
                server_model_meta meta{
                    /* source        */ get_source(name),
                    /* preset        */ preset,
                    /* name          */ name,
                    /* aliases       */ {},
                    /* tags          */ {},
                    /* port          */ 0,
                    /* status        */ SERVER_MODEL_STATUS_UNLOADED,
                    /* last_used     */ 0,
                    /* args          */ std::vector<std::string>(),
                    /* loaded_info   */ {},
                    /* progress      */ {},
                    /* exit_code     */ 0,
                    /* stop_timeout  */ DEFAULT_STOP_TIMEOUT,
                    /* idle_timeout  */ -1,
                    /* multimodal    */ mtmd_caps{false, false},
                    /* placement     */ {},
                    // /* need_download */ false,
                };
                add_model(std::move(meta));
            }
        }

        apply_stop_timeout();
        log_idle_timeouts();
        apply_hidden();
        apply_groups();

        // clear reload flag under the lock, this releases the load() calls waiting on !is_reloading
        is_reloading = false;
        cv.notify_all();

        log_available_models();

        lk.unlock();

        notify_sse("models_reload", "*");
    }
}

void server_models::load_startup_models() {
    std::vector<std::string> to_load;
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (!startup_models.has_value()) {
            return; // already drained
        }
        to_load = std::move(*startup_models);
        startup_models.reset();
    }
    for (const auto & name : to_load) {
        SRV_INF("(startup) loading model %s\n", name.c_str());
        try {
            load(name);
        } catch (const router_queued_error & e) {
            // waits for its turn (a board claim / a busy model); the router loads it then
            SRV_WRN("(startup) %s\n", e.what());
        }
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loading_finished
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    if (mapping.find(name) != mapping.end()) {
        return true;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            return true;
        }
    }
    return false;
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    auto queue_info = [this](const std::string & key, const server_model_meta & meta) -> json {
        auto q = queued_loads.find(key);
        if (q == queued_loads.end() || meta.status != SERVER_MODEL_STATUS_UNLOADED) {
            return nullptr;
        }
        return json::parse(router_queued_info_json(q->second.info));
    };
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        server_model_meta out = it->second.meta;
        out.group_info = group_status_json_locked(it->first);
        out.queue_info = queue_info(it->first, out);
        return out;
    }
    for (const auto & [key, inst] : mapping) {
        if (inst.meta.aliases.count(name)) {
            server_model_meta out = inst.meta;
            out.group_info = group_status_json_locked(key);
            out.queue_info = queue_info(key, out);
            return out;
        }
    }
    return std::nullopt;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::unique_lock<std::mutex> lk(mutex);
    if (need_reload) {
        lk.unlock();
        load_models();
        lk.lock();
    }

    std::vector<server_model_meta> result;
    result.reserve(mapping.size());
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
        result.back().group_info = group_status_json_locked(name);
        auto q = queued_loads.find(name);
        if (q != queued_loads.end() && inst.meta.status == SERVER_MODEL_STATUS_UNLOADED) {
            result.back().queue_info = json::parse(router_queued_info_json(q->second.info));
        }
    }
    return result;
}

void server_models::unload_lru() {
    if (base_params.models_max <= 0) {
        return; // no limit
    }
    // remove one of the servers if we passed the models_max (least recently used - LRU)
    // 2026-08-10 upstream sync: eviction moved into upstream's server_lru_sched.
    // The fork's pinned / stopping_models guards that used to live in this loop are
    // now enforced inside server_lru_sched::pick_victim (the FORK GUARD note plus
    // upstream's stopping/queued filter). Do not reintroduce a second victim-selection
    // path here.
    std::string lru_model_name;
    {
        std::unique_lock<std::mutex> lk(mutex);
        if (sched->has_capacity(lk)) {
            return;
        }
        lru_model_name = sched->pick_victim(lk);
        if (lru_model_name.empty()) {
            return;
        }
        // Stop the victim under the SAME lock that selected it: request_stop() marks it
        // stopping, so a concurrent unload_lru() excludes it in pick_victim and cannot
        // evict a second model. pick_victim only returns ready/sleeping models, so a
        // graceful exit request is always right here.
        SRV_INF("models_max limit reached, removing LRU name=%s\n", lru_model_name.c_str());
        notify_state("evicting", lru_model_name, mapping[lru_model_name].meta.placement.devs, "models_max reached (LRU)");
        request_stop(lru_model_name, true);
    }
    // wait for unload to complete (find-based: safe if the entry was erased mid-wait,
    // unlike the previous mapping[name] which default-constructed a stray entry)
    wait(lru_model_name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED;
    });
}

// Per-model `env` from the preset, applied OVER the environment the router inherited. Entries
// REPLACE any existing definition rather than being appended: duplicate KEY= entries in envp
// resolve inconsistently across libc getenv implementations.
static void apply_env_overrides(std::vector<std::string> & env, const std::vector<std::string> & overrides, const std::string & name, bool verbose = true) {
    for (const auto & override_entry : overrides) {
        const bool remove = override_entry[0] == '-';
        const std::string key = remove
            ? override_entry.substr(1)
            : override_entry.substr(0, override_entry.find('='));
        const std::string prefix = key + "=";
        for (auto it = env.begin(); it != env.end(); ) {
            it = it->compare(0, prefix.size(), prefix) == 0 ? env.erase(it) : it + 1;
        }
        if (!remove) {
            env.push_back(override_entry);
        }
        if (verbose) {
            SRV_INF("model '%s': env %s%s\n", name.c_str(),
                    remove ? "unset " : "", remove ? key.c_str() : override_entry.c_str());
        }
    }
}

// Final env = inherited router env + preset `env`. A temp dir that points at a non-directory
// (e.g. TMPDIR=/etc/passwd) fails every tmpfile the child makes in confusing ways: refuse.
static void check_temp_dirs(const std::vector<std::string> & env, const std::string & name) {
    const std::string bad = env_temp_dir_violation(env);
    if (!bad.empty()) {
        const std::string var = bad.substr(0, bad.find('='));
        throw std::runtime_error("model '" + name + "' cannot load: " + var + " is not a directory (" + bad + ")");
    }
}

void server_models::load(const std::string & name) {
    load(name, load_options{});
}

void server_models::load(const std::string & name, const load_options & opts) {
    if (opts.custom_meta.has_value() || opts.mode != SERVER_CHILD_MODE_NORMAL) {
        load_impl(name, opts); // downloads / estimates: no admission, no board, no queue
        return;
    }
    // The owners of this load's board claims (the model; a group's workers) count as alive while
    // it runs, though they are not running yet. Unregistered on every way out, before
    // release_dead(), so a failed or queued attempt leaves no claim behind.
    std::vector<std::string> owners = { name };
    {
        std::lock_guard<std::mutex> lk(mutex);
        if (shutting_down) {
            throw router_refused_error("router is shutting down");
        }
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            owners.insert(owners.end(), it->second.meta.depends.begin(), it->second.meta.depends.end());
        }
        for (const auto & o : owners) {
            loading_owners[o]++;
        }
    }
    auto unregister = [&]() {
        std::lock_guard<std::mutex> lk(mutex);
        for (const auto & o : owners) {
            auto it = loading_owners.find(o);
            if (it != loading_owners.end() && --it->second <= 0) {
                loading_owners.erase(it);
            }
        }
        owners.clear();
    };
    bool spawned = false;
    try {
        spawned = load_impl(name, opts);
    } catch (const router_queue_signal & sig) {
        unregister();
        if (board) {
            board->release_dead(); // claims this attempt took for a placement it will not use
        }
        on_queued(name, opts, sig.res, sig.raced.has_value() ? &*sig.raced : nullptr); // throws router_queued_error
    } catch (const std::exception & e) {
        unregister();
        if (board) {
            board->release_dead();
        }
        drop_queued(name, e.what()); // a queued load that can no longer go ahead
        throw;
    }
    unregister();
    if (spawned && board) {
        board->leave_queue(name); // it was queued for a resource it got some other way
    }
}

bool server_models::load_impl(const std::string & name, const load_options & opts) {
    if (debug_fake_timing) {
        // do not hold the mutex here, other requests must keep making progress
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    if (!opts.custom_meta.has_value()) {
        if (!has_model(name)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        auto m = get_meta(name);
        if (m.has_value() && !router_model_requestable(m->kind)) {
            throw std::runtime_error("model name=" + name + " is not found");
        }
        // Refuse a load whose child env is bad before anything is evicted, in any mode (the
        // LRU eviction just below runs before placement). Checked again, final, further down.
        if (m.has_value()) {
            std::vector<std::string> env = base_env;
            apply_env_overrides(env, m->env_overrides, name, /*verbose=*/false);
            check_temp_dirs(env, name);
            if (opts.mode == SERVER_CHILD_MODE_NORMAL && !m->depends.empty()) {
                std::lock_guard<std::mutex> l(mutex);
                prepare_group_locked(name, *m, /*verbose=*/false);
            }
        }
        if (!gpu_placement_enabled) {
            unload_lru();
        }
    }

    std::unique_lock<std::mutex> lk(mutex);
    // edge case: block until any in-progress reload has finished so we always load
    // against the freshest preset and a consistent mapping state
    cv.wait(lk, [this]() { return !is_reloading; });

    // A model group: one spine plus the workers it depends on. A previous load of this group
    // may still be stopping its workers (spine already down): wait for that to finish, so two
    // generations of a worker never overlap on its port and GPU.
    const bool is_group = opts.mode == SERVER_CHILD_MODE_NORMAL && !opts.custom_meta.has_value() &&
                          mapping.count(name) && !mapping[name].meta.depends.empty();
    if (is_group) {
        cv.wait(lk, [this, &name]() {
            auto g = groups.find(name);
            return !is_reloading && (g == groups.end() || g->second.stop_done);
        });
    }

    auto meta = opts.custom_meta.has_value() ? *opts.custom_meta : mapping[name].meta;
    if (meta.status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return false;
    }

    if (!opts.custom_meta.has_value()) {
        reload_models_preset_if_changed(meta);
    }

    bool marked_loading = false;
    // a group is marked loading up front too: its workers come up before the spine is spawned,
    // and nobody else may start the same group meanwhile
    if ((gpu_placement_enabled || is_group) && opts.mode == SERVER_CHILD_MODE_NORMAL && !opts.custom_meta.has_value()) {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            it->second.meta.status = SERVER_MODEL_STATUS_LOADING;
            it->second.meta.last_used = ggml_time_ms();
            marked_loading = true;
            cv.notify_all();
        }
    }

    bool placement_reserved = false;
    auto rollback_gpu_reservation = [&]() {
        if (!placement_reserved) {
            return;
        }
        for (size_t i = 0; i < meta.placement.devs.size() && i < meta.placement.need_bytes_per_dev.size(); ++i) {
            const int slot_idx = find_slot_index(gpu_slots, meta.placement.devs[i]);
            if (slot_idx < 0) {
                continue;
            }
            auto & slot = gpu_slots[slot_idx];
            slot.reserved_bytes = std::max<int64_t>(0, slot.reserved_bytes - meta.placement.need_bytes_per_dev[i]);
            if (slot.exclusive_holder == name) {
                slot.exclusive_holder.clear();
            }
        }
        placement_reserved = false;
    };
    auto rollback_loading_status = [&]() {
        if (!marked_loading) {
            return;
        }
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.meta.status == SERVER_MODEL_STATUS_LOADING) {
            it->second.meta.status = SERVER_MODEL_STATUS_UNLOADED;
        }
        stopping_models.erase(name);
        marked_loading = false;
        cv.notify_all();
    };
    // group members: placed (and marked loading) before the start, or already started
    std::vector<std::string>             workers_marked;
    std::shared_ptr<router_worker_group> group_started;
    auto rollback_group = [&]() {
        if (group_started) {
            // the workers are up: stop them the normal way; on_group_stopped() marks them
            // unloaded, and the next load of this group waits for that
            group_started->request_stop();
            group_started.reset();
        } else {
            for (const auto & w : workers_marked) {
                auto wit = mapping.find(w);
                if (wit != mapping.end() && wit->second.meta.status != SERVER_MODEL_STATUS_UNLOADED) {
                    set_worker_status_locked(w, SERVER_MODEL_STATUS_UNLOADED); // credits its reservation
                }
            }
        }
        workers_marked.clear();
    };
    auto rollback_load_attempt = [&](void *) {
        if (!lk.owns_lock()) {
            lk.lock();
        }
        if (group_started) {
            // The workers are up and only being asked to stop (up to quiesce + grace). The spine
            // must keep counting as running -- holding its reservation, marked stopping -- until
            // they are gone, or an evictor waiting on it would load onto VRAM/RAM the workers
            // still hold. on_group_stopped() makes the final mark and credits, as on unload.
            auto it = mapping.find(name);
            if (it != mapping.end()) {
                if (placement_reserved) {
                    it->second.meta.placement = meta.placement; // credited by on_group_stopped()
                    placement_reserved = false;
                }
                auto g = groups.find(name);
                if (g != groups.end() && !stopping_models.count(name)) {
                    // not a cancel: the spine itself could not be started
                    g->second.failed = true;
                    if (g->second.reason.empty()) {
                        g->second.reason = "the spine of model group '" + name + "' failed to start";
                    }
                }
                stopping_models.insert(name);
            }
            marked_loading = false; // the spine stays LOADING + stopping until its workers exit
            rollback_group();
            cv.notify_all();
            return;
        }
        rollback_group();
        rollback_gpu_reservation();
        rollback_loading_status();
    };
    std::unique_ptr<void, decltype(rollback_load_attempt)> load_attempt_guard(reinterpret_cast<void *>(1), rollback_load_attempt);

    // The child environments are final before placement: a load that is refused for its env
    // must not have evicted anything first.
    std::vector<std::string> child_env = base_env; // carries LLAMA_ROUTER_GEN / LLAMA_ROUTER_PID
    child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));
    apply_env_overrides(child_env, meta.env_overrides, meta.name);
    check_temp_dirs(child_env, name);
    std::vector<router_worker_spec> worker_specs;
    if (is_group) {
        worker_specs = prepare_group_locked(name, meta); // launch, port, env (+ temp-dir check) of every worker
    }

    ensure_gpu_placement(name, meta, opts, lk);
    if (gpu_placement_enabled && opts.mode == SERVER_CHILD_MODE_NORMAL && !meta.placement.need_bytes_per_dev.empty()) {
        placement_reserved = true;
    }

    // Admission covers the whole group: each worker's vram-mb on its gpu slot and ram-mb on
    // this machine. A worker marked loading holds its ram-mb back from the next one's check.
    for (const auto & spec : worker_specs) {
        auto wit = mapping.find(spec.name);
        if (wit == mapping.end()) {
            throw std::runtime_error("model group '" + name + "': worker '" + spec.name + "' disappeared during the load");
        }
        server_model_meta wmeta = wit->second.meta;
        ensure_gpu_placement(spec.name, wmeta, opts, lk); // may wait (unlocked) for evictions
        wit = mapping.find(spec.name);
        if (wit == mapping.end()) {
            throw std::runtime_error("model group '" + name + "': worker '" + spec.name + "' disappeared during the load");
        }
        wit->second.meta.placement = wmeta.placement; // so set_worker_status_locked() can credit it
        set_worker_status_locked(spec.name, SERVER_MODEL_STATUS_LOADING);
        workers_marked.push_back(spec.name);
    }

    // Re-check capacity under the lock to prevent concurrent loads from
    // exceeding models_max. Without this, the window between unload_lru()
    // releasing its lock and this lock acquiring allows multiple threads to
    // each observe capacity and all proceed to load. On a capacity race (a
    // concurrent load claimed the slot unload_lru() just freed) we evict again
    // and wait for a slot rather than failing the request with a 500 — the
    // request effectively queues behind the eviction it triggers.
    // Download workers do not use models_max slots.
    if (!gpu_placement_enabled && opts.mode == SERVER_CHILD_MODE_NORMAL && base_params.models_max > 0) {
        const int64_t capacity_deadline = ggml_time_ms() + 30000; // 30s cap, then surface a retriable error
        auto count_running = [this, &name]() {
            size_t n = 0;
            for (const auto & m : mapping) {
                // a group takes one slot; a group spine marked loading by this call is not "another" model
                if (m.first != name && m.second.meta.is_running() && !m.second.meta.is_external()) {
                    n++;
                }
            }
            return n;
        };
        while (count_running() >= (size_t)base_params.models_max) {
            if (ggml_time_ms() >= capacity_deadline) {
                // genuinely no evictable slot (e.g. every resident model is pinned) —
                // surface a retriable error instead of blocking the request forever.
                throw std::runtime_error("model limit reached, try again later");
            }
            // release the lock so unload_lru() can evict + wait for the child to exit,
            // then re-take it and re-validate our own preconditions.
            lk.unlock();
            unload_lru();
            lk.lock();
            cv.wait(lk, [this]() { return !is_reloading; });
            auto it = mapping.find(name);
            // a group spine is marked loading by this call itself; a stop request cancels it
            const server_model_status own_status = marked_loading ? SERVER_MODEL_STATUS_LOADING : SERVER_MODEL_STATUS_UNLOADED;
            if (it == mapping.end() || it->second.meta.status != own_status || stopping_models.count(name)) {
                // a concurrent path took over this model while we were evicting
                SRV_INF("model %s no longer loadable after capacity wait\n", name.c_str());
                return false;
            }
            if (count_running() >= (size_t)base_params.models_max) {
                // eviction couldn't free a slot yet (all residents pinned, or a concurrent
                // load refilled it); back off briefly instead of hot-spinning to the deadline.
                cv.wait_for(lk, std::chrono::milliseconds(200));
            }
        }
    }

    // Workers first: the spine is spawned only once every worker accepts TCP.
    if (is_group) {
        if (stopping_models.count(name)) {
            throw std::runtime_error("load of model group '" + name + "' was cancelled");
        }
        auto & rt = groups[name];
        // the previous load's workers object (all stopped, see the wait above). Its destructor
        // joins its thread, which may still be finishing a callback that takes `mutex`: drop it
        // only once the lock is released below.
        std::shared_ptr<router_worker_group> retired = std::move(rt.workers);
        rt = group_runtime{};
        rt.stop_done = false;

        auto holder = std::make_shared<const router_worker_group *>(nullptr);
        router_worker_group::callbacks cb;
        cb.on_unexpected_exit = [this, name, holder](const std::string & worker, int code, const std::string & reason) {
            on_group_worker_exit(name, *holder, worker, code, reason);
        };
        cb.on_stopped = [this, name, holder]() {
            on_group_stopped(name, *holder);
        };
        auto grp = std::make_shared<router_worker_group>(name, worker_specs, std::move(cb));
        *holder = grp.get(); // published to the group thread through its lock in start()
        rt.workers = grp;

        lk.unlock();
        retired.reset();
        std::string err;
        const bool ok = grp->start(err, [this, name]() {
            std::lock_guard<std::mutex> l(mutex);
            return stopping_models.count(name) > 0; // unload() / eviction / load timeout while starting
        });
        lk.lock();
        if (!ok) {
            // start() already tore down and reaped every worker it had started
            auto g = groups.find(name);
            if (g != groups.end() && g->second.workers == grp) {
                g->second.stop_done = true;
                g->second.failed    = err.find("cancelled") == std::string::npos;
                g->second.reason    = err;
            }
            for (const auto & w : grp->status()) {
                if (!w.error.empty()) { // the worker that failed the start shows as failed
                    set_worker_status_locked(w.name, SERVER_MODEL_STATUS_UNLOADED, w.exit_code > 0 ? w.exit_code : 1);
                }
            }
            throw std::runtime_error("model group '" + name + "' failed to start: " + err);
        }
        group_started = grp;
        workers_marked.clear(); // from here on, on_group_stopped() owns the workers' status
        for (const auto & spec : worker_specs) {
            set_worker_status_locked(spec.name, SERVER_MODEL_STATUS_LOADED);
        }
        if (stopping_models.count(name)) {
            throw std::runtime_error("load of model group '" + name + "' was cancelled");
        }
        SRV_INF("group %s: all workers ready, spawning the spine\n", name.c_str());
    }

    if (shutting_down) {
        throw router_refused_error("router is shutting down");
    }
    // spawning now: no longer a queued load (its board queue slot is left by load())
    queued_loads.erase(name);

    // prepare new instance info
    instance_t inst;
    inst.meta             = meta;
    inst.meta.port        = common_http_get_free_port();
    inst.meta.status      = SERVER_MODEL_STATUS_LOADING;
    inst.meta.loaded_info = json{};
    inst.meta.last_used   = ggml_time_ms();

    if (inst.meta.port <= 0) {
        throw std::runtime_error("failed to get a port number");
    }

    inst.subproc = std::make_shared<server_subproc>();
    {
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        inst.meta.update_args(ctx_preset, bin_path); // render args

        std::vector<std::string> child_args = inst.meta.args; // copy
        // child_env: built and temp-dir checked before placement (see above)

        if (opts.mode == SERVER_CHILD_MODE_DOWNLOAD) {
            inst.meta.status = SERVER_MODEL_STATUS_DOWNLOADING;
            child_env.push_back("LLAMA_SERVER_CHILD_MODE=download");
            child_env.push_back("LLAMA_ARG_HF_REPO=" + name);
        } else if (opts.mode == SERVER_CHILD_MODE_ESTIMATE) {
            child_env.push_back("LLAMA_SERVER_CHILD_MODE=estimate");
        }

        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }
        inst.meta.args = child_args; // save for debugging

        // TODO @ngxson : maybe separate stdout and stderr in the future
        //                so that we can use stdout for commands and stderr for logging
        int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        if (!inst.subproc->sproc.create(child_args, options, child_env)) {
            load_attempt_guard.reset(); // also stops a group's workers
            throw std::runtime_error("failed to spawn server instance");
        }
        load_attempt_guard.release();
        group_started.reset(); // the spine's exit now drives the workers' stop (on_child_exit)
    }

    // old process should have exited already, but just in case, we clean it up here
    {
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.subproc && it->second.subproc->is_alive()) {
            SRV_WRN("old process for model name=%s is still alive, this is unexpected\n", name.c_str());
            it->second.subproc->terminate(); // force kill
        }
    }

    notify_sse("model_status", name, {
        {"status", server_model_status_to_string(inst.meta.status)},
    });
    if (opts.mode == SERVER_CHILD_MODE_NORMAL) {
        notify_state("loading", name, inst.meta.placement.devs, "spawned");
    }

    auto proc = inst.subproc;
    int  port = inst.meta.port;
    mapping[name] = std::move(inst);
    monitor->watch(name, proc, opts.mode, port);
    cv.notify_all();
    return true;
}

void server_models::request_stop(const std::string & name_in, bool send_exit, bool drain) {
    // a worker is stopped by stopping its group, which starts at the spine
    const std::string name = group_spine_locked(name_in);
    auto it = mapping.find(name);
    if (it == mapping.end() || stopping_models.count(name)) {
        return;
    }
    if (name != name_in) {
        const bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
        if (loading) {
            it->second.subproc->terminate();
        }
        send_exit = !loading;
    }
    stopping_models.insert(name);
    auto g = groups.find(name);
    if (g != groups.end() && drain && send_exit && it->second.req_count > 0) {
        // unload order for a group: drain the spine's requests (no new ones are routed to a
        // stopping model), then stop it; bounded by its stop-timeout (idle sweeper enforces it)
        const int bound_s = std::max(1, it->second.meta.stop_timeout);
        g->second.drain_deadline = ggml_time_ms() + (int64_t) bound_s * 1000;
        SRV_INF("group %s: draining %d in-flight request(s) before stopping (up to %d s)\n",
                name.c_str(), it->second.req_count, bound_s);
        return;
    }
    monitor->stop(name, it->second.meta.stop_timeout, send_exit);
}

void server_models::maybe_finish_drain_locked(const std::string & name, bool force) {
    auto g = groups.find(name);
    if (g == groups.end() || g->second.drain_deadline == 0) {
        return;
    }
    auto it = mapping.find(name);
    if (!force && it != mapping.end() && it->second.req_count > 0) {
        return;
    }
    g->second.drain_deadline = 0;
    if (it != mapping.end()) {
        if (it->second.req_count > 0) {
            SRV_WRN("group %s: stopping with %d request(s) still in flight (drain bound reached)\n",
                    name.c_str(), it->second.req_count);
        }
        monitor->stop(name, it->second.meta.stop_timeout, true);
    }
}

void server_models::set_worker_status_locked(const std::string & worker, server_model_status status, int exit_code) {
    auto it = mapping.find(worker);
    if (it == mapping.end()) {
        return;
    }
    auto & meta = it->second.meta;
    if (status == SERVER_MODEL_STATUS_UNLOADED && !meta.placement.need_bytes_per_dev.empty()) {
        credit_gpu_reservation_locked(worker);
    }
    meta.status    = status;
    meta.exit_code = exit_code;
    if (status != SERVER_MODEL_STATUS_UNLOADED) {
        meta.last_used = ggml_time_ms();
    }
    json data = { {"status", server_model_status_to_string(status)} };
    if (status == SERVER_MODEL_STATUS_UNLOADED) {
        data["exit_code"] = exit_code;
    }
    notify_sse("status_change", worker, data); // does not take the lock
    if (status == SERVER_MODEL_STATUS_UNLOADED && board) {
        board->wake(); // its claims go
    }
    cv.notify_all();
}

// local-only for now: a worker must be on the router's own machine
static bool router_machine_is_local(const std::string & machine) {
    if (machine.empty() || machine == "local" || machine == "localhost") {
        return true;
    }
#ifndef _WIN32
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf) - 1) == 0) {
        auto lower = [](std::string v) {
            std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char) std::tolower(c); });
            return v;
        };
        const std::string host  = lower(buf);
        const std::string want  = lower(machine);
        const std::string short_host = host.substr(0, host.find('.'));
        return want == host || want == short_host;
    }
#endif
    return false;
}

std::vector<router_worker_spec> server_models::prepare_group_locked(const std::string & name, const server_model_meta & spine_meta, bool verbose) {
    std::vector<router_worker_spec> specs;
    for (const auto & dep : spine_meta.depends) {
        auto it = mapping.find(dep);
        if (it == mapping.end() || !it->second.meta.is_external()) {
            throw std::runtime_error("model group '" + name + "': worker '" + dep + "' is not a kind=external section");
        }
        const auto & w = it->second.meta;
        if (!router_machine_is_local(w.machine)) {
            throw std::runtime_error("model group '" + name + "': worker '" + dep + "' is on machine '" + w.machine +
                                     "'; only workers on the router's own machine are supported so far");
        }
        router_launch launch;
        const std::string err = router_parse_launch(w.launch, launch);
        if (!err.empty()) {
            throw std::runtime_error("model group '" + name + "': worker '" + dep + "': " + err);
        }

        router_worker_spec spec;
        spec.name = dep;
        spec.argv = launch.argv;
        std::vector<std::string> env = base_env; // carries LLAMA_ROUTER_GEN / LLAMA_ROUTER_PID
        apply_env_overrides(env, w.env_overrides, dep, verbose);
        spec.env = router_worker_env(std::move(env), launch.env, w.park_file, router_gen);
        check_temp_dirs(spec.env, dep);

        int port = router_launch_endpoint(launch.words, spec.host);
        if (w.worker_port > 0) {
            port = w.worker_port;
        }
        if (port <= 0) {
            throw std::runtime_error("model group '" + name + "': cannot tell which port worker '" + dep +
                                     "' listens on (no --listen HOST:PORT or --port N in launch); set worker-port");
        }
        spec.port              = port;
        spec.startup_timeout_s = std::max(spine_meta.startup_timeout_s, w.startup_timeout_s);
        spec.quiesce_ms        = router_env_quiesce_ms(spec.env);
        specs.push_back(std::move(spec));
    }
    return specs;
}

json server_models::group_status_json_locked(const std::string & spine) const {
    auto it = mapping.find(spine);
    if (it == mapping.end() || it->second.meta.depends.empty()) {
        return nullptr;
    }
    const auto & meta = it->second.meta;
    auto rt = groups.find(spine);
    const bool stopping = stopping_models.count(spine) > 0;
    std::string status;
    if (meta.is_ready_or_sleep()) {
        status = stopping ? "stopping" : "ready";
    } else if (meta.status == SERVER_MODEL_STATUS_LOADING) {
        status = stopping ? "stopping" : "loading";
    } else {
        // a spine that crashed on its own (healthy workers) is a failed group too
        status = (rt != groups.end() && rt->second.failed) || meta.is_failed() ? "failed" : "unloaded";
    }
    json workers = json::array();
    if (rt != groups.end() && rt->second.workers) {
        for (const auto & w : rt->second.workers->status()) {
            workers.push_back({
                {"name", w.name},
                {"pid", w.pid},
                {"host", w.host},
                {"port", w.port},
                {"state", w.state},
                {"exit_code", w.exit_code},
                {"killed", w.killed},
                {"stop_snapshot", router_snapshot_result_str(w.snapshot)},
                {"stop_snapshot_detail", w.snapshot_detail},
                {"quiesce_timeout", w.quiesce_timeout},
                {"error", w.error},
            });
        }
    } else {
        for (const auto & dep : meta.depends) {
            workers.push_back({ {"name", dep}, {"state", "pending"}, {"stop_snapshot", "none"} });
        }
    }
    json out = { {"status", status}, {"workers", workers} };
    if (rt != groups.end() && !rt->second.reason.empty()) {
        out["reason"] = rt->second.reason;
    }
    return out;
}

void server_models::on_group_worker_exit(const std::string & spine, const router_worker_group * g, const std::string & worker, int exit_code, const std::string & reason) {
    std::unique_lock<std::mutex> lk(mutex);
    auto rt = groups.find(spine);
    if (rt == groups.end() || rt->second.workers.get() != g || rt->second.stop_done) {
        return; // a group that is already gone
    }
    // The group thread decides "unexpected" before it calls back, unlocked: an unload may have
    // started in between. A group already on its way down is not failed by a worker that exits.
    if (stopping_models.count(spine) || rt->second.pending_exit.has_value()) {
        set_worker_status_locked(worker, SERVER_MODEL_STATUS_UNLOADED, exit_code);
        maybe_finish_drain_locked(spine, true); // no point draining a group that lost a worker
        return;
    }
    rt->second.failed = true;
    if (rt->second.reason.empty()) {
        rt->second.reason = reason;
    }
    set_worker_status_locked(worker, SERVER_MODEL_STATUS_UNLOADED, exit_code == 0 ? 1 : exit_code);
    auto it = mapping.find(spine);
    if (it == mapping.end() || !it->second.meta.is_running()) {
        return;
    }
    SRV_ERR("group %s failed (%s); stopping its spine, the next request reloads the whole group\n",
            spine.c_str(), reason.c_str());
    const bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
    if (loading) {
        it->second.subproc->terminate();
    }
    request_stop(spine, !loading, /*drain=*/false);
}

void server_models::on_group_stopped(const std::string & spine, const router_worker_group * g) {
    int  code = 0;
    bool changed = false;
    std::vector<std::string> slots;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto rt = groups.find(spine);
        if (rt == groups.end() || rt->second.workers.get() != g || rt->second.stop_done) {
            return;
        }
        for (const auto & w : g->status()) {
            auto wit = mapping.find(w.name);
            if (wit != mapping.end() && wit->second.meta.status != SERVER_MODEL_STATUS_UNLOADED) {
                set_worker_status_locked(w.name, SERVER_MODEL_STATUS_UNLOADED, std::max(0, w.exit_code));
            }
        }
        code = rt->second.pending_exit.value_or(0);
        if (rt->second.failed && code == 0) {
            code = 1; // shows as failed; the next request loads the group again
        }
        rt->second.pending_exit.reset();
        rt->second.drain_deadline = 0;
        // last step of the unload order: the group is unloaded only now that its workers are gone
        auto it = mapping.find(spine);
        if (it != mapping.end()) {
            auto & meta = it->second.meta;
            if (!meta.placement.need_bytes_per_dev.empty()) {
                credit_gpu_reservation_locked(spine);
            }
            changed = meta.status != SERVER_MODEL_STATUS_UNLOADED || meta.exit_code != code;
            meta.status    = SERVER_MODEL_STATUS_UNLOADED;
            meta.exit_code = code;
            stopping_models.erase(spine);
        }
        rt->second.stop_done = true; // same critical section as the status change (see load())
        sched->tick(lk);
        bump_queue_locked();
        if (it != mapping.end()) {
            slots = it->second.meta.placement.devs;
        }
    }
    SRV_INF("group %s unloaded (spine and all workers stopped)\n", spine.c_str());
    if (changed) {
        notify_sse("status_change", spine, { {"status", "unloaded"}, {"exit_code", code} });
        notify_state("unloaded", spine, slots, code == 0 ? "stopped" : "group failed or exited with status " + std::to_string(code));
    }
    if (board) {
        board->wake();
    }
    cv.notify_all();
}

void server_models::on_child_exit(const std::string & name, const std::shared_ptr<server_subproc> & proc, server_child_mode mode, int exit_code) {
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it == mapping.end() || it->second.subproc != proc) {
            stopping_models.erase(name);
            return; // entry erased, or a newer instance took the name
        }
        if (mode == SERVER_CHILD_MODE_NORMAL) {
            auto g = groups.find(name);
            if (g != groups.end() && g->second.workers && !g->second.stop_done) {
                // unload order: the spine is down, now its workers (TERM in parallel, KILL after
                // quiesce + grace); on_group_stopped() marks the group unloaded once they are gone
                SRV_INF("group %s: spine exited with status %d, stopping its workers\n", name.c_str(), exit_code);
                g->second.pending_exit   = exit_code;
                g->second.drain_deadline = 0;
                stopping_models.insert(name); // nothing may be routed to it meanwhile
                g->second.workers->request_stop();
                return;
            }
        }
    }
    if (mode == SERVER_CHILD_MODE_DOWNLOAD) {
        // instance will be cleaned up on next load_models() call
        std::lock_guard<std::mutex> lk(mutex);
        stopping_models.erase(name);
        cv.notify_all();
    } else {
        update_status(name, {
            SERVER_MODEL_STATUS_UNLOADED,
            exit_code
        });
    }
}

void server_models::unload(const std::string & name_in) {
    std::unique_lock<std::mutex> lk(mutex);
    const std::string name = group_spine_locked(name_in); // a worker unloads its whole group
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            it->second.request_exit();
            // for convenience, we wait the status change here
            wait(lk, name, [](const server_model_meta & new_meta) {
                return new_meta.status != SERVER_MODEL_STATUS_DOWNLOADING;
            });
        } else if (it->second.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                // special case: if model is in loading state, unloading means force-killing it
                SRV_WRN("model name=%s is still loading, force-killing\n", name.c_str());
                it->second.subproc->terminate();
            }
            request_stop(name, !loading);
            // status change will be handled by the monitor
        } else {
            SRV_WRN("model instance name=%s is not running\n", name.c_str());
        }
    }
}

void server_models::unload_all() {
    std::unique_lock<std::mutex> lk(mutex);
    for (auto & [name, inst] : mapping) {
        if (inst.meta.is_external()) {
            continue; // stopped with its spine; the wait below still waits for it
        }
        if (inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
            SRV_INF("cancelling download for model name=%s\n", name.c_str());
            inst.request_exit();
        } else if (inst.meta.is_running()) {
            SRV_INF("stopping model instance name=%s\n", name.c_str());
            bool loading = inst.meta.status == SERVER_MODEL_STATUS_LOADING;
            if (loading) {
                inst.subproc->terminate();
            }
            request_stop(name, !loading);
        }
    }
    // wait for every child to exit, the monitor force-kills the ones that ignore the exit command
    cv.wait(lk, [this]() {
        for (const auto & [name, inst] : mapping) {
            if (inst.meta.is_running() || inst.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
                return false;
            }
        }
        return true;
    });
}

void server_models::update_status(const std::string & name, const update_status_args & args) {
    std::unique_lock<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    bool                     found = false;
    server_model_status      prev  = SERVER_MODEL_STATUS_UNLOADED;
    std::vector<std::string> slots;
    if (it != mapping.end()) {
        auto & meta = it->second.meta;
        found = true;
        prev  = meta.status;
        slots = meta.placement.devs;
        if (args.status == SERVER_MODEL_STATUS_UNLOADED && !meta.placement.need_bytes_per_dev.empty()) {
            credit_gpu_reservation_locked(name);
        }
        meta.status      = args.status;
        meta.exit_code   = args.exit_code;
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            stopping_models.erase(name);
        }
        if (!args.loaded_info.is_null()) {
            meta.loaded_info = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            meta.progress = args.progress;
        }
        if (args.status == SERVER_MODEL_STATUS_LOADED && !meta.placement.need_bytes_per_dev.empty()) {
            reconcile_gpu_reservation_locked(name);
        }
        // a model that comes up idle or goes down changes the slot count for queued requests
        sched->tick(lk);
        if (prev != args.status) {
            bump_queue_locked(); // queued loads may fit now
        }
    }
    // broadcast status change to SSE
    {
        json data = {
            {"status", server_model_status_to_string(args.status)},
        };
        if (args.status == SERVER_MODEL_STATUS_UNLOADED) {
            data["exit_code"] = args.exit_code;
        }
        if (!args.loaded_info.is_null()) {
            data["info"] = args.loaded_info;
        }
        if (!args.progress.is_null()) {
            data["progress"] = args.progress;
        }
        // note: notify_sse doesn't acquire the lock, so no deadlock here
        notify_sse("status_change", name, data);
    }
    if (found && args.status == SERVER_MODEL_STATUS_LOADED && prev == SERVER_MODEL_STATUS_LOADING) {
        notify_state("ready", name, slots, "");
    }
    if (found && args.status == SERVER_MODEL_STATUS_UNLOADED && prev != SERVER_MODEL_STATUS_UNLOADED) {
        notify_state("unloaded", name, slots, args.exit_code == 0 ? "stopped" : "exited with status " + std::to_string(args.exit_code));
        if (board) {
            board->wake(); // release its claims now rather than at the next poll
        }
    }
    cv.notify_all();
}

void server_models::update_download_progress(const std::string & name, const common_download_progress & progress, bool done, bool ok) {
    json curr;
    {
        std::lock_guard<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            if (done) {
                // mark the instance to be erased on next load_models() call
                it->second.meta.status = SERVER_MODEL_STATUS_DOWNLOADED;
                need_reload = true;
            } else {
                json & info = it->second.meta.loaded_info;
                if (!info.contains("progress")) {
                    info["progress"] = json{};
                }
                info["progress"][progress.url] = {
                    {"done",  progress.downloaded},
                    {"total", progress.total},
                };
                curr = it->second.meta.loaded_info; // copy
            }
        }
    }
    if (done) {
        cv.notify_all(); // notify in case unload() is waiting for download to be cancelled
        notify_sse(ok ? "download_finished" : "download_failed", name, {});
    } else {
        notify_sse("download_progress", name, curr);
    }
}

bool server_models::remove(const std::string & name) {
    // do everything under one lock acquisition; avoid get_meta() /
    // unload() because they can trigger load_models() which erases
    // transient DOWNLOADING / DOWNLOADED entries as a side-effect
    std::unique_lock<std::mutex> lk(mutex);

    auto it = mapping.find(name);
    if (it == mapping.end()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (it->second.meta.source != SERVER_MODEL_SOURCE_CACHE) {
        throw std::runtime_error("model name=" + name + " is not removable (not from cache)");
    }

    if (it->second.meta.status == SERVER_MODEL_STATUS_DOWNLOADING) {
        // cancel in-flight download
        SRV_INF("cancelling download for model name=%s\n", name.c_str());
        it->second.request_exit();
    } else if (it->second.meta.is_running()) {
        // stop running instance
        SRV_INF("stopping model instance name=%s\n", name.c_str());
        bool loading = it->second.meta.status == SERVER_MODEL_STATUS_LOADING;
        if (loading) {
            it->second.subproc->terminate();
        }
        request_stop(name, !loading);
    }

    // wait until the child is gone
    wait(lk, name, [](const server_model_meta & meta) {
        return meta.status == SERVER_MODEL_STATUS_UNLOADED
            || meta.status == SERVER_MODEL_STATUS_DOWNLOADED;
    });

    // re-find after wait - load_models() may have erased the entry during the wait
    it = mapping.find(name);
    if (it == mapping.end()) {
        // load_models() already erased the entry; we just need to clean up the cached files on disk
        lk.unlock();
        bool ok = common_download_remove(name);
        SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
        notify_sse("model_remove", name, {});
        return true;
    }

    // remove from disk (best-effort: cancelled downloads may have no cached files)
    bool ok = common_download_remove(name);
    mapping.erase(name);
    if (!ok) {
        SRV_WRN("removing model name=%s from disk returned false (no cached files?)\n", name.c_str());
    }
    SRV_INF("removing model name=%s from cache (%s)\n", name.c_str(), ok ? "succeeded" : "partial");
    notify_sse("model_remove", name, {});
    return true;
}

void server_models::wait(const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    std::unique_lock<std::mutex> lk(mutex);
    wait(lk, name, predicate);
}

void server_models::wait(std::unique_lock<std::mutex> & lk, const std::string & name, std::function<bool(const server_model_meta &)> predicate) {
    cv.wait(lk, [this, &name, &predicate]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return predicate(it->second.meta);

        }
        // model was removed from mapping by another code path (e.g. load_models()).
        // nothing left to wait for - tell the caller to proceed.
        return true;
    });
}

bool server_models::ensure_model_ready(const std::string & name, const std::function<bool()> & should_stop,
                                       const router_request_opts & req) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    bool stopping;
    bool wait_in_queue; // a queued load: `lowest` waits for it, `middle` / `highest` get told now
    {
        std::lock_guard<std::mutex> lk(mutex);
        stopping = stopping_models.count(name) > 0;
        auto it = mapping.find(name);
        wait_in_queue = router_request_waits_in_queue(it != mapping.end() ? effective_priority(req, it->second.meta)
                                                                          : (req.priority_set ? req.priority : ADMISSION_PRIORITY_MIDDLE));
    }
    load_options lo;
    lo.req = req;
    if (!stopping && meta->is_ready()) {
        return false; // ready for taking requests
    }
    if (!stopping && meta->status == SERVER_MODEL_STATUS_SLEEPING) {
        return false; // child is sleeping but still running; new request will wake it up
    }

    bool queued   = false;
    bool did_load = false;
    {
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end() && it->second.meta.status == SERVER_MODEL_STATUS_UNLOADED) {
            // the queue entry protects the model from eviction until its waiters leave
            sched->join(lk, name);
            sched->tick(lk);
            queued = true;
        }
    }

    // while queued, this is also where the load happens: the head of the queue does it
    SRV_INF("waiting until model name=%s is fully loaded...\n", name.c_str());
    std::unique_lock<std::mutex> lk(mutex);
    auto leave_queue = [this, &queued, &lk, &name]() {
        if (queued) {
            sched->leave(lk, name);
            queued = false;
        }
    };

    try {
        bool saw_loading = false;
        while (true) {
            auto it = mapping.find(name);
            if (it == mapping.end()) {
                break; // removed by another code path, nothing to wait for
            }
            if (stopping_models.count(name)) {
                // a stopping instance takes no new request, the next instance serves it
                if (!queued) {
                    sched->join(lk, name);
                    sched->tick(lk);
                    queued = true;
                }
                if (should_stop && should_stop()) {
                    throw std::runtime_error("request cancelled while waiting for model name=" + name);
                }
                cv.wait_for(lk, std::chrono::milliseconds(200));
                continue;
            }
            const server_model_status status = it->second.meta.status;

            if (status == SERVER_MODEL_STATUS_LOADED || status == SERVER_MODEL_STATUS_SLEEPING) {
                break;
            }
            if (status == SERVER_MODEL_STATUS_DOWNLOADING || status == SERVER_MODEL_STATUS_DOWNLOADED) {
                break; // do not wait on a download child
            }
            if (status == SERVER_MODEL_STATUS_UNLOADED) {
                // queued behind a board claim or a busy resident: the router retries the load itself
                auto q = queued_loads.find(name);
                if (q != queued_loads.end()) {
                    if (!wait_in_queue) {
                        throw router_queued_error(q->second.info);
                    }
                    if (should_stop && should_stop()) {
                        throw std::runtime_error("request cancelled while model name=" + name + " was queued");
                    }
                    cv.wait_for(lk, std::chrono::milliseconds(200));
                    continue;
                }
            }
            if (status == SERVER_MODEL_STATUS_LOADING) {
                saw_loading = true;
            } else if (status == SERVER_MODEL_STATUS_UNLOADED) {
                if (did_load || saw_loading) {
                    // a spawn happened and the instance came back down
                    if (it->second.meta.is_failed()) {
                        throw std::runtime_error("model name=" + name + " failed to load");
                    }
                    break; // unloaded by another code path, caller reports "not running"
                }
                if (!queued) {
                    break; // not queued, and the load someone else started fell over
                }
            }

            if (should_stop && should_stop()) {
                // if a model was evicted for us, the free slot goes to the next waiter
                throw std::runtime_error("request cancelled while waiting for model name=" + name);
            }

            // our turn: our model is at the head, and a slot really did free up
            if (status == SERVER_MODEL_STATUS_UNLOADED && sched->try_claim(lk, name)) {
                lk.unlock();
                bool ok = true;
                std::exception_ptr fatal; // what the caller has to hear instead of a retry
                try {
                    SRV_INF("slot available, loading queued model name=%s\n", name.c_str());
                    load(name, lo);
                    did_load = true;
                } catch (const router_queued_error &) {
                    ok = false; // recorded as queued: `lowest` waits for it above, the others are told
                    if (!wait_in_queue) {
                        fatal = std::current_exception();
                    }
                } catch (const router_refused_error &) {
                    ok    = false; // admission will not take it as things stand: no point retrying
                    fatal = std::current_exception();
                } catch (const std::exception & e) {
                    // lost a race for the slot, stay in line and retry
                    SRV_WRN("queued load of name=%s did not go through: %s\n", name.c_str(), e.what());
                    ok = false;
                }
                lk.lock();
                sched->claim_done(lk, name, ok);
                sched->tick(lk);
                if (fatal) {
                    std::rethrow_exception(fatal);
                }
                continue;
            }

            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    } catch (...) {
        leave_queue();
        sched->tick(lk); // a slot freed for this waiter goes to the next one
        throw;
    }
    leave_queue();

    return true;
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name, bool update_last_used, bool detached) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (!meta->is_running()) {
        throw std::invalid_argument("model name=" + name + " is not running");
    }
    {
        // Count this request as in flight for the whole life of the proxy (see the cleanup
        // chain below). last_used alone is not enough: it is stamped HERE, at request start,
        // so a 20-minute generation looks idle to the sweeper and would be unloaded
        // mid-stream. Do this regardless of update_last_used -- a health poll still keeps
        // the model alive for its duration, which is correct and costs nothing.
        std::unique_lock<std::mutex> lk(mutex);
        if (update_last_used) {
            mapping[name].meta.last_used = ggml_time_ms();
        }
        mapping[name].req_count++;
    }
    if (debug_fake_timing) {
        // sleep after req_count++, so the model counts as busy while we wait here
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
    SRV_INF("proxying request to model %s on port %d\n", name.c_str(), meta->port);
    std::string proxy_path = req.path;
    if (!req.query_string.empty()) {
        proxy_path += '?' + req.query_string;
    }
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            "http",
            CHILD_ADDR,
            meta->port,
            proxy_path,
            req.headers,
            req.body,
            req.files,
            // a detached request belongs to a replay session
            detached
                ? std::function<bool()>([]() { return false; })
                : req.should_stop,
            base_params.timeout_read,
            base_params.timeout_write
            );

    // Chain onto any cleanup the proxy already carries (do NOT clobber it). When the proxy
    // dies the request is done, so drop the in-flight count and re-stamp last_used: the
    // idle clock should start when the response ENDS, not when it began.
    auto prev_cleanup = proxy->cleanup;
    proxy->cleanup = [this, name, prev_cleanup]() {
        if (prev_cleanup) {
            prev_cleanup();
        }
        std::unique_lock<std::mutex> lk(mutex);
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            it->second.meta.last_used = ggml_time_ms();
            if (it->second.req_count > 0) {
                it->second.req_count--;
                if (it->second.req_count == 0) {
                    maybe_finish_drain_locked(name, false); // a draining group spine stops now
                    sched->tick(lk);
                    bump_queue_locked(); // a load waiting on this busy model may go now
                }
            }
        }
    };


    return proxy;
}

// Unload models that have gone quiet. This is what makes the fleet give the machine back:
// an idle model holds its whole GPU (exclusive placement) AND its host-side KV tier, which
// on a 15 GB box is gigabytes of RAM sitting there doing nothing.
//
// Timeout is per-model: preset `idle-timeout` overrides router `--models-idle-timeout`.
// Effective 0 means never idle-unload that model.
//
// Two hard rules:
//   - never unload a model with a request in flight (a generation can run for many minutes
//     with no new request, so last_used alone would kill it mid-stream);
//   - never unload a pinned model (pinning is a human saying "leave this alone").
void server_models::idle_sweeper_loop() {
    const int global_timeout_s = base_params.models_idle_timeout;
    int load_timeout_s = DEFAULT_ROUTER_LOAD_TIMEOUT_S;
    if (const char * value = std::getenv(ROUTER_LOAD_TIMEOUT); value != nullptr && value[0] != '\0') {
        try {
            load_timeout_s = std::stoi(value);
        } catch (const std::exception &) {
            SRV_WRN("invalid %s=%s; using default %ds\n",
                    ROUTER_LOAD_TIMEOUT, value, DEFAULT_ROUTER_LOAD_TIMEOUT_S);
            load_timeout_s = DEFAULT_ROUTER_LOAD_TIMEOUT_S;
        }
    }
    if (load_timeout_s < 0) {
        SRV_WRN("invalid %s=%d; using default %ds\n",
                ROUTER_LOAD_TIMEOUT, load_timeout_s, DEFAULT_ROUTER_LOAD_TIMEOUT_S);
        load_timeout_s = DEFAULT_ROUTER_LOAD_TIMEOUT_S;
    }
    SRV_INF("router idle sweeper: global idle-timeout=%ds (0=never); per-model idle-timeout overrides\n",
            global_timeout_s);
    SRV_INF("router load timeout: %ds (0=never) via %s\n",
            load_timeout_s, ROUTER_LOAD_TIMEOUT);

    while (!idle_stop.load(std::memory_order_relaxed)) {
        for (int i = 0; i < 10 && !idle_stop.load(std::memory_order_relaxed); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (idle_stop.load(std::memory_order_relaxed)) {
            break;
        }

        // (name, effective_timeout_s used for the log line)
        std::vector<std::pair<std::string, int>> victims;
        std::vector<std::pair<std::string, int>> load_victims;
        {
            std::unique_lock<std::mutex> lk(mutex);
            const int64_t now = ggml_time_ms();
            holds.prune(now); // expired leases drop silently
            // group spines waiting for in-flight requests: stop them once their drain bound passes
            for (auto & [spine, rt] : groups) {
                if (rt.drain_deadline > 0 && now >= rt.drain_deadline) {
                    maybe_finish_drain_locked(spine, true);
                }
            }
            for (const auto & [name, inst] : mapping) {
                if (!inst.meta.is_running() || stopping_models.count(name)) {
                    continue;
                }
                if (inst.meta.is_external()) {
                    continue; // idles and load-times out with its spine
                }
                if (inst.meta.placement.pinned || inst.req_count > 0) {
                    continue;
                }
                if (inst.meta.status == SERVER_MODEL_STATUS_LOADING) {
                    if (load_timeout_s > 0 && inst.meta.last_used > 0 &&
                            now - inst.meta.last_used >= (int64_t) load_timeout_s * 1000) {
                        load_victims.emplace_back(name, load_timeout_s);
                    }
                    continue;
                }
                idle_resident ir;
                ir.pinned    = inst.meta.placement.pinned;
                ir.held      = is_held_locked(name); // a hold lease: no idle unload
                ir.req_count = inst.req_count;
                ir.last_used = inst.meta.last_used; // <= 0: never served a request; left to LRU/eviction
                ir.timeout_s = effective_idle_timeout_s(inst.meta, global_timeout_s); // 0 = never idle-unload
                if (idle_unload_due(ir, now)) {
                    victims.emplace_back(name, ir.timeout_s);
                    notify_state("evicting", name, inst.meta.placement.devs, "idle > " + std::to_string(ir.timeout_s) + " s");
                }
            }
        }
        for (const auto & [name, timeout_s] : victims) {
            SRV_INF("router idle sweeper: unloading %s (idle > %ds); GPU and host KV released\n",
                    name.c_str(), timeout_s);
            try {
                unload(name);
            } catch (const std::exception & e) {
                SRV_WRN("router idle sweeper: failed to unload %s: %s\n", name.c_str(), e.what());
            }
        }
        for (const auto & [name, timeout_s] : load_victims) {
            SRV_WRN("router load timeout: unloading %s (loading > %ds)\n",
                    name.c_str(), timeout_s);
            try {
                unload(name);
            } catch (const std::exception & e) {
                SRV_WRN("router load timeout: failed to unload %s: %s\n", name.c_str(), e.what());
            }
        }
    }
}

void server_models::handle_child_state(const std::string & name, const std::string & raw_input) {
    server_state state;
    json payload;

    try {
        json data = json::parse(raw_input.substr(strlen(CMD_CHILD_TO_ROUTER_STATE)));
        state = server_state_from_str(json_value(data, "state", std::string()));
        payload = json_value(data, "payload", json{});
    } catch (const std::exception & e) {
        SRV_ERR("failed to parse child state update for name=%s: %s\n", name.c_str(), e.what());
        return;
    }

    switch (state) {
        case SERVER_STATE_DOWNLOADING:
            {
                std::string result = json_value(payload, "result", std::string());
                std::string url    = json_value(payload, "url",    std::string());
                auto request_exit = [&]() {
                    std::lock_guard<std::mutex> lk(mutex);
                    auto it = mapping.find(name);
                    if (it != mapping.end()) {
                        return it->second.request_exit();
                    }
                };
                if (result == "download_finished") {
                    update_download_progress(name, {}, true, true);
                    request_exit();
                } else if (result == "download_failed") {
                    update_download_progress(name, {}, true, false);
                    request_exit();
                } else if (!url.empty()) {
                    common_download_progress p;
                    p.url        = url;
                    p.downloaded = json_value(payload, "downloaded", (size_t)0);
                    p.total      = json_value(payload, "total", (size_t)0);
                    update_download_progress(name, p, false);
                }
            } break;
        case SERVER_STATE_LOADING:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADING,
                    0,
                    nullptr, // no loaded_info yet
                    payload,
                });
            } break;
        case SERVER_STATE_READY:
            {
                update_status(name, {
                    SERVER_MODEL_STATUS_LOADED,
                    0,
                    // note: payload can be empty if this is a wakeup from sleep
                    payload.size() > 0 ? payload : nullptr,
                    {}, // reset progress info
                });
            } break;
        case SERVER_STATE_SLEEPING:
            {
                update_status(name, { SERVER_MODEL_STATUS_SLEEPING });
            } break;
        default:
            // should never happen, but just in case
            GGML_ASSERT(false && "unexpected state from child server");
    }
}

//
// server_child
//

bool server_child::is_child() {
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    return router_port != nullptr;
}

server_child_mode server_child::get_mode() {
    const char * mode = std::getenv("LLAMA_SERVER_CHILD_MODE");
    std::string mode_str(mode ? mode : "");
    if (mode_str == "download") {
        return SERVER_CHILD_MODE_DOWNLOAD;
    } else if (mode_str == "estimate") {
        return SERVER_CHILD_MODE_ESTIMATE;
    } else {
        return SERVER_CHILD_MODE_NORMAL;
    }
}

struct server_download_state : public common_download_callback {
    server_child * self;
    std::function<bool()> should_stop;
    std::atomic<int64_t> last_progress_time{0}; // multiple files downloading in different threads
    bool is_ok = false;

    server_download_state(server_child * s) : self(s) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        self->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs
        if (now - last_progress_time.load(std::memory_order_relaxed) >= 100) {
            on_progress(p);
            last_progress_time.store(now, std::memory_order_relaxed);
        }
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

int server_child::run_download(common_params & params) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(this);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

int server_child::run_estimate(common_params & params) {
    try {
        if (params.model.path.empty()) {
            throw std::runtime_error("estimate mode requires a resolved model path");
        }
        auto mparams = common_model_params_to_llama(params);
        auto cparams = common_context_params_to_llama(params);
        std::vector<ggml_backend_dev_t> devs;
        uint32_t hp_ngl = 0;
        uint32_t hp_n_ctx_train = 0;
        uint32_t hp_n_expert = 0;
        auto data = common_get_device_memory_data(
                params.model.path.c_str(),
                &mparams,
                &cparams,
                devs,
                hp_ngl,
                hp_n_ctx_train,
                hp_n_expert,
                GGML_LOG_LEVEL_WARN);

        json names = json::array();
        json need = json::array();
        for (size_t i = 0; i < devs.size() && i < data.size(); ++i) {
            names.push_back(ggml_backend_dev_name(devs[i]));
            need.push_back((int64_t) (data[i].model + data[i].context + data[i].compute));
        }
        notify_to_router(server_state_to_str(SERVER_STATE_READY), {
            {"devices", names},
            {"need_bytes_per_dev", need},
            {"n_gpu_layers", hp_ngl},
            {"n_ctx_train", hp_n_ctx_train},
            {"n_expert", hp_n_expert},
        });
        return 0;
    } catch (const std::exception & e) {
        SRV_ERR("estimate failed: %s\n", e.what());
        return 1;
    }
}

std::thread server_child::setup(const std::function<void(int)> & shutdown_handler) {
    // setup thread for monitoring stdin
    return std::thread([shutdown_handler]() {
        // wait for EOF on stdin
        SRV_INF("%s", "child server monitoring thread started, waiting for EOF on stdin...\n");
        bool eof = false;
        while (true) {
            std::string line;
            if (!std::getline(std::cin, line)) {
                // EOF detected, that means the router server is unexpectedly exit or killed
                eof = true;
                break;
            }
            if (line.find(CMD_ROUTER_TO_CHILD_EXIT) != std::string::npos) {
                SRV_INF("%s", "exit command received, exiting...\n");
                shutdown_handler(0);
                break;
            }
        }
        if (eof) {
            SRV_INF("%s", "EOF on stdin detected, forcing shutdown...\n");
            exit(1);
        }
    });
}

void server_child::notify_to_router(const std::string & state, const json & payload) {
    json data = {
        {"state", state},
        {"payload", payload},
    };
    std::lock_guard<std::mutex> lk(mtx_stdout);
    common_log_pause(common_log_main());
    fflush(stdout);
    // the router matches the command on a line prefix, so the leading newline
    // closes whatever the logger left open on the shared pipe, down to the
    // trailing color reset that carries no newline of its own
    fprintf(stdout, "\n%s%s\n", CMD_CHILD_TO_ROUTER_STATE, safe_json_to_str(data).c_str());
    fflush(stdout);
    common_log_resume(common_log_main());
}


//
// server_models_routes
//

// RAII wrapper similar to server_response_reader, but doesn't use server_queue
static std::atomic<int> sse_client_id_counter = 0;
struct server_models_sse_client {
    server_response & queue_results;
    int client_id;
    server_models_sse_client(server_response & q)
            : queue_results(q), client_id(sse_client_id_counter.fetch_add(1, std::memory_order_relaxed)) {
        SRV_DBG("new SSE client connected, assigned client_id=%d\n", client_id);
        queue_results.add_waiting_task_id(client_id);
    }
    ~server_models_sse_client() {
        SRV_DBG("SSE client disconnected, removing client_id=%d\n", client_id);
        queue_results.remove_waiting_task_id(client_id);
    }

    // return nullptr if should_stop() is true before receiving a result
    // note: if one error is received, it will stop further processing and return error result
    server_task_result_ptr next(const std::function<bool()> & should_stop) {
        while (true) {
            static const int http_polling_seconds = 1; // check should_stop every 1 second
            server_task_result_ptr result = queue_results.recv_with_timeout({client_id}, http_polling_seconds);
            if (result == nullptr) {
                // timeout, check stop condition
                if (should_stop()) {
                    return nullptr;
                }
                // continue waiting otherwise
            } else {
                SRV_DBG("recv result for client_id=%d: %s\n", client_id, safe_json_to_str(result->to_json()).c_str());
                return result;
            }
        }
        // should not reach here
    }
};

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

// 503 + Retry-After + where the load stands, for a `middle` / `highest` request whose model is queued
static void res_queued(std::unique_ptr<server_http_res> & res, const router_queued_info & info) {
    int status = 503;
    std::string body;
    std::map<std::string, std::string> headers;
    router_queued_response(info, status, body, headers);
    res->status = status;
    res->data   = body;
    for (const auto & [k, v] : headers) {
        res->headers[k] = v;
    }
}

static bool router_validate_model(std::string & name, server_models & models, bool models_autoload, std::unique_ptr<server_http_res> & res,
                                  const router_request_opts & ro = {}, const std::function<bool()> & should_stop = nullptr) {
    if (name.empty()) {
        res_err(res, format_error_response("model name is missing from the request", ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    auto meta = models.get_meta(name);
    if (!meta.has_value()) {
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    if (!router_model_requestable(meta->kind)) {
        // a group worker is not a model: same answer as an unknown name
        res_err(res, format_error_response(string_format("model '%s' not found", name.c_str()), ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    // resolve alias to canonical model name
    name = meta->name;
    if (models_autoload) {
        try {
            models.ensure_model_ready(name, should_stop, ro);
        } catch (const router_queued_error & e) {
            res_queued(res, e.info);
            return false;
        } catch (const std::runtime_error & e) {
            res_err(res, {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 503},
            });
            return false;
        }
    } else {
        if (!meta->is_running()) {
            res_err(res, format_error_response("model is not loaded", ERROR_TYPE_INVALID_REQUEST));
            return false;
        }
    }
    return true;
}

static bool is_autoload(const common_params & params, const server_http_req & req, const server_models & models) {
    // The runtime master switch wins over everything, including an explicit
    // ?autoload=true on the request. When a human has taken the GPUs back, no request
    // gets to bring a model up behind their back.
    if (!models.autoload_enabled.load(std::memory_order_relaxed)) {
        return false;
    }
    std::string autoload = req.get_param("autoload");
    if (autoload.empty()) {
        return params.models_autoload;
    } else {
        return autoload == "true" || autoload == "1";
    }
}

// case-insensitive header lookup (HTTP header names are case-insensitive; the map is not)
static std::string header_value_ci(const std::map<std::string, std::string> & headers, const std::string & target_lower) {
    for (const auto & [hk, hv] : headers) {
        if (hk.size() != target_lower.size()) {
            continue;
        }
        bool match = true;
        for (size_t i = 0; i < hk.size(); ++i) {
            char c = hk[i];
            if (c >= 'A' && c <= 'Z') {
                c = char(c + 32);
            }
            if (c != target_lower[i]) {
                match = false;
                break;
            }
        }
        if (match) {
            return hv;
        }
    }
    return std::string();
}

// priority / machine of a request: JSON body fields `priority` / `machine` (when the body was
// parsed), else the X-Priority / X-Machine headers. A bad priority answers 400.
static bool router_request_opts_from(const server_http_req & req, const json * body, router_request_opts & out,
                                     std::unique_ptr<server_http_res> & res) {
    std::string bp;
    std::string bm;
    if (body != nullptr && body->is_object()) {
        bp = json_value(*body, "priority", std::string());
        bm = json_value(*body, "machine", std::string());
    }
    std::string err;
    if (!router_parse_request_opts(bp, header_value_ci(req.headers, "x-priority"), bm, header_value_ci(req.headers, "x-machine"), out, err)) {
        res_err(res, format_error_response(err, ERROR_TYPE_INVALID_REQUEST));
        return false;
    }
    return true;
}

// percent encode one query or path component, covers reserved chars without pulling in
// httplib::detail. used by the stream routes to forward conversation_id to children safely
static std::string encode_qs(const std::string & in) {
    std::string out;
    out.reserve(in.size() * 3);
    for (unsigned char c : in) {
        bool safe = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                 || c == '-' || c == '_' || c == '.' || c == '~';
        if (safe) {
            out.push_back(char(c));
        } else {
            char buf[4];
            std::snprintf(buf, sizeof(buf), "%%%02X", c);
            out.append(buf, 3);
        }
    }
    return out;
}

// resolve the child that owns a conversation's stream session via the conv_id -> model map
// populated when the POST was routed. single map lookup then a meta lookup, no polling, no
// parsing of the conv id. returns nullopt when nothing maps, the caller answers not found and
// the client recovers
static std::optional<server_model_meta> resolve_child_for_conv(
        server_models & models, const std::string & conversation_id) {
    if (conversation_id.empty()) {
        return std::nullopt;
    }
    auto tracked = models.conv_models.lookup(conversation_id);
    if (!tracked.has_value()) {
        return std::nullopt;
    }
    auto meta = models.get_meta(*tracked);
    if (meta.has_value() && meta->is_ready()) {
        return meta;
    }
    return std::nullopt;
}

void server_models_routes::init_routes() {
    if (!common_subproc::is_supported()) {
        throw std::runtime_error("subprocess is not enabled on this build");
    }

    this->get_router_props = [this](const server_http_req & req) {
        std::string name = req.get_param("model");
        if (name.empty()) {
            // main instance
            auto res = std::make_unique<server_http_res>();
            res_ok(res, {
                // TODO: add support for this on web UI
                {"role",                 "router"},
                {"max_instances",        params.models_max},
                {"models_autoload",      params.models_autoload},
                // this is a dummy response to make sure the UI doesn't break
                {"model_alias", "llama-server"},
                {"model_path",  "none"},
                {"default_generation_settings", {
                    {"params", json{}},
                    {"n_ctx",  0},
                }},
                // New key
                {"ui_settings",          ui_settings},
                {"build_info",           std::string(llama_build_info())},
                {"cors_proxy_enabled",   params.ui_mcp_proxy},
            });
            return res;
        }
        return proxy_get(req);
    };

    this->get_router_health = [this](const server_http_req &) {
        // Unlike the stock /health (which is unconditionally "ok"), reflect child
        // process health so a crashed/wedged child is visible to fleet monitoring.
        // The router itself is alive if it can answer at all, so top-level status
        // stays "ok" (non-breaking for monitors that only check the 200) — the body
        // carries the per-child detail, including any children that exited non-zero.
        auto res = std::make_unique<server_http_res>();
        auto all_models = models.get_all_meta();
        json by_status = json::object();
        json unhealthy = json::array();
        size_t running = 0, loading = 0, sleeping = 0, failed_count = 0;
        for (const auto & meta : all_models) {
            const std::string s = server_model_status_to_string(meta.status);
            by_status[s] = by_status.value(s, 0) + 1;
            if (meta.status == SERVER_MODEL_STATUS_LOADING)  { loading++; }
            if (meta.status == SERVER_MODEL_STATUS_SLEEPING) { sleeping++; }
            if (meta.is_running())                           { running++; }
            if (meta.is_failed()) {
                failed_count++;
                unhealthy.push_back({{"model", meta.name}, {"exit_code", meta.exit_code}});
            }
        }
        res_ok(res, {
            {"status", "ok"}, // the router process itself is up
            {"role",   "router"},
            {"children", {
                {"total",     all_models.size()},
                {"running",   running},
                {"loading",   loading},
                {"sleeping",  sleeping},
                {"failed",    failed_count},
                {"by_status", by_status},
            }},
            {"unhealthy", unhealthy}, // children that exited non-zero
        });
        return res;
    };

    this->proxy_get = [this](const server_http_req & req) {
        std::string method = "GET";
        std::string name = req.get_param("model");
        bool autoload = is_autoload(params, req, models);
        auto error_res = std::make_unique<server_http_res>();
        router_request_opts ro;
        if (!router_request_opts_from(req, nullptr, ro, error_res)) {
            return error_res;
        }
        if (!router_validate_model(name, models, autoload, error_res, ro, req.should_stop)) {
            return error_res;
        }
        if (autoload) {
            models.ensure_model_ready(name, req.should_stop, ro);
        }
        return models.proxy_request(req, method, name, false);
    };

    this->proxy_post = [this](const server_http_req & req) {
        std::string method = "POST";
        // Fast path: honor an explicit X-Model header so we avoid fully parsing a
        // possibly multi-MB body (e.g. base64 images) just to read "model" — the
        // child parses the body again anyway. Fall back to the JSON parse when the
        // header is absent, preserving behavior for existing clients.
        // priority / machine ride in the same body (or in X-Priority / X-Machine, the only
        // source when X-Model spares the parse)
        std::string name = header_value_ci(req.headers, "x-model");
        auto error_res = std::make_unique<server_http_res>();
        router_request_opts ro;
        if (name.empty()) {
            json body = json::parse(req.body);
            name = json_value(body, "model", std::string());
            if (!router_request_opts_from(req, &body, ro, error_res)) {
                return error_res;
            }
        } else if (!router_request_opts_from(req, nullptr, ro, error_res)) {
            return error_res;
        }
        bool autoload = is_autoload(params, req, models);
        // a session request (X-Conversation-Id) is not cancelled by its socket, see below
        const bool session = !server_stream_conv_id_from_headers(req.headers).empty();
        if (!router_validate_model(name, models, autoload, error_res, ro, session ? std::function<bool()>() : req.should_stop)) {
            return error_res;
        }
        // remember which child serves this conversation so the stream routes can route straight
        // to it without polling, keyed on the exact conv id from the header. registered before
        // the load wait so a stop issued while the model loads can erase the entry and cancel
        // this request instead of leaving an orphan generation
        std::string conv_id = server_stream_conv_id_from_headers(req.headers);
        uint64_t ticket = models.conv_models.remember(conv_id, name);
        // a dead socket must not cancel a session request, only a stop does (checked right below)
        auto should_stop = ticket == 0 ? req.should_stop : nullptr;
        bool waited = autoload && models.ensure_model_ready(name, should_stop, ro);
        if (ticket != 0 && !models.conv_models.alive(conv_id, ticket)) {
            SRV_INF("request for conv_id=%s cancelled while model name=%s was loading\n",
                    conv_id.c_str(), name.c_str());
            res_err(error_res, format_error_response(
                    "request cancelled by a stop while the model was loading", ERROR_TYPE_INVALID_REQUEST));
            return error_res;
        }
        // a session request that waited for a load detaches from the client socket: the
        // client may have dropped during the wait (page reload) and the session buffer must
        // still receive the generation for a later resume
        return models.proxy_request(req, method, name, true, waited && ticket != 0); // update last usage for POST request only
    };

    // POST /models/load {model, priority?, machine?} (or X-Priority / X-Machine). Answers right
    // away: 200 {state: ready} when it is up, else 202 {state: loading | queued, queue_pos,
    // blocked_by, ...}; 503 when admission refuses it. The load itself goes on in the background
    // and a queued one is retried by the router; follow it on /models/sse.
    this->post_router_models_load = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        router_request_opts ro;
        if (!router_request_opts_from(req, &body, ro, res)) {
            return res;
        }
        auto meta = models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        auto answer = [&res](int status, json out) {
            out["success"] = true;
            res->status = status;
            res->data   = safe_json_to_str(out);
        };
        if (meta->is_ready_or_sleep()) {
            json out = json::object();
            out["model"] = meta->name;
            out["state"] = "ready";
            answer(200, out);
            return res;
        }
        if (meta->status == SERVER_MODEL_STATUS_LOADING) {
            json out = json::object();
            out["model"] = meta->name;
            out["state"] = "loading";
            answer(202, out);
            return res;
        }
        try {
            json out = models.load_async(meta->name, ro);
            const bool ready = json_value(out, "state", std::string()) == "ready";
            answer(ready ? 200 : 202, out);
        } catch (const router_queued_error & e) {
            json out = json::parse(router_queued_info_json(e.info));
            out["model"] = meta->name;
            answer(202, out);
        } catch (const std::exception & e) {
            res_err(res, {
                {"message", e.what()},
                {"type", "server_error"},
                {"code", 503},
            });
        }
        return res;
    };

    // one body, two registrations: /models lists everything, /v1/models is the OAI view
    auto list_models = [this](const server_http_req & req, bool oai_listing) {
        bool reload = !req.get_param("reload", "").empty();
        if (reload) {
            models.load_models();
        }
        auto res = std::make_unique<server_http_res>();
        json models_json = json::array();
        auto all_models = models.get_all_meta();
        std::time_t t = std::time(0);
        for (const auto & meta : all_models) {
            if (meta.hidden) {
                continue; // cache model deduplicated by a preset
            }
            if (oai_listing && !router_model_in_oai_listing(meta.kind)) {
                continue;
            }
            json status {
                {"value",  server_model_status_to_string(meta.status)},
                {"args",   meta.args},
            };
            if (!meta.preset.name.empty()) {
                common_preset preset_copy = meta.preset;
                unset_reserved_args(preset_copy, false);
                preset_copy.unset_option("LLAMA_ARG_HOST");
                preset_copy.unset_option("LLAMA_ARG_PORT");
                preset_copy.unset_option("LLAMA_ARG_ALIAS");
                preset_copy.unset_option("LLAMA_ARG_TAGS");
                status["preset"] = preset_copy.to_ini();
            }
            if (meta.is_failed()) {
                status["exit_code"] = meta.exit_code;
                status["failed"]    = true;
            }
            if (!meta.group_info.is_null()) {
                // group spine: loading / ready / stopping / failed / unloaded, and per worker its
                // pid, state and which stop-snapshot line it printed (written / FAILED / timeout / none)
                status["group"] = meta.group_info;
            }
            if (!meta.queue_info.is_null()) {
                // a queued load: queue_pos, blocked_by, blocked_on, board, priority, reason
                status["value"] = "queued";
                for (const auto & [k, v] : meta.queue_info.items()) {
                    if (k != "state") {
                        status[k] = v;
                    }
                }
            }

            // pi coding agent multimodal compatibility
            json input_modalities = json::array({"text"});
            if (meta.multimodal.inp_vision) {
                input_modalities.push_back("image");
            }
            if (meta.multimodal.inp_audio) {
                input_modalities.push_back("audio");
            }
            json architecture {
                {"input_modalities",  input_modalities},
                {"output_modalities", json::array({"text"})},
            };

            json model_info = json {
                {"id",            meta.name},
                {"aliases",       meta.aliases},
                {"tags",          meta.tags},
                {"object",        "model"},    // for OAI-compat
                {"owned_by",      "llamacpp"}, // for OAI-compat
                {"created",       t},          // for OAI-compat
                {"status",        status},
                {"architecture",  architecture},
                {"source",        server_model_source_to_string(meta.source)},
                {"can_remove",    meta.source == SERVER_MODEL_SOURCE_CACHE},
                // {"need_download", meta.need_download},
                // TODO: add other fields, may require reading GGUF metadata
            };
            json placement = {
                {"devices", meta.placement.devs},
                {"split", meta.placement.split},
                {"pinned", meta.placement.pinned},
                {"exclusive", meta.placement.exclusive},
                {"need_bytes", meta.placement.need_bytes_per_dev},
            };
            model_info["placement"] = placement;
            model_info["kind"]  = meta.is_external() ? "external" : "model";
            model_info["priority"] = admission_priority_str(meta.priority); // the preset default
            model_info["group"] = meta.group.empty() ? json(nullptr) : json(meta.group);
            if (!meta.depends.empty()) {
                model_info["depends"] = meta.depends;
            }
            // -1 in the ledger means "inherit global"; clients that want the
            // actual sweeper value can use idle_timeout_effective.
            model_info["idle_timeout"] = meta.idle_timeout;
            model_info["idle_timeout_effective"] =
                effective_idle_timeout_s(meta, params.models_idle_timeout);

            // merge with loaded_info from the child process if available
            if (meta.is_running()) {
                for (auto it = meta.loaded_info.begin(); it != meta.loaded_info.end(); ++it) {
                    if (!model_info.contains(it.key())) {
                        model_info[it.key()] = it.value();
                    }
                }
            }
            models_json.push_back(model_info);
        }
        res_ok(res, {
            {"data", models_json},
            {"devices", models.gpu_slots_json()},
            {"machines", models.machines_json()},
            {"board", models.board_json()},
            {"object", "list"},
        });
        return res;
    };
    this->get_router_models     = [list_models](const server_http_req & req) { return list_models(req, false); };
    this->get_router_models_oai = [list_models](const server_http_req & req) { return list_models(req, true); };

    this->post_router_models_unload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        auto model = models.get_meta(name);
        if (!model.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!model->is_running() && model->status != SERVER_MODEL_STATUS_DOWNLOADING) {
            if (models.cancel_queued(model->name)) {
                res_ok(res, {{"success", true}, {"cancelled", true}}); // a queued load, dropped
                return res;
            }
            res_err(res, format_error_response("model is not running", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        models.unload(model->name);
        res_ok(res, {{"success", true}});
        return res;
    };

    // Master switch. POST /models/autoload {"enabled": false, "unload_all": true}
    //
    // This exists so a human can take the GPUs back without stopping the router: while
    // disabled, no request can bring a model up. `unload_all` additionally evicts what
    // is already resident, so you can go from "the fleet is using both cards" to "both
    // cards are mine" in one call -- the point being that nothing can then decide to
    // load a 29 GB model into the card you are gaming on.
    this->post_router_models_autoload = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        if (!body.contains("enabled")) {
            res_err(res, format_error_response("'enabled' (bool) is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        const bool enabled = json_value(body, "enabled", true);
        models.autoload_enabled.store(enabled, std::memory_order_relaxed);
        SRV_INF("router autoload %s by request\n", enabled ? "ENABLED" : "DISABLED");

        json unloaded = json::array();
        if (!enabled && json_value(body, "unload_all", false)) {
            for (const auto & meta : models.get_all_meta()) {
                if (meta.is_running()) {
                    models.unload(meta.name);
                    unloaded.push_back(meta.name);
                }
            }
            SRV_INF("router unloaded %zu model(s); GPUs released\n", unloaded.size());
        }
        res_ok(res, {{"autoload", enabled}, {"unloaded", unloaded}});
        return res;
    };

    // POST /models/hold {model, ttl_s, owner, lease?}: keep a model resident (no eviction, no
    // idle unload, no yield to the board queue) for ttl_s seconds; re-POST with the lease to
    // renew. -> {lease, model, ttl_s, owner}. Expired leases drop silently.
    this->post_router_models_hold = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        const std::string model = json_value(body, "model", std::string());
        const int64_t     ttl_s = json_value(body, "ttl_s", (long long) 0);
        const std::string owner = json_value(body, "owner", std::string());
        const std::string lease = json_value(body, "lease", std::string());
        try {
            res_ok(res, models.hold(model, ttl_s, owner, lease));
        } catch (const std::out_of_range & e) {
            res_err(res, format_error_response(e.what(), ERROR_TYPE_NOT_FOUND));
        } catch (const std::invalid_argument & e) {
            const bool unknown_lease = std::string(e.what()).rfind("unknown", 0) == 0;
            res_err(res, format_error_response(e.what(), unknown_lease ? ERROR_TYPE_NOT_FOUND : ERROR_TYPE_INVALID_REQUEST));
        }
        return res;
    };

    // POST /models/release {lease}
    this->post_router_models_release = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = req.body.empty() ? json::object() : json::parse(req.body);
        const std::string lease = json_value(body, "lease", std::string());
        if (lease.empty()) {
            res_err(res, format_error_response("'lease' is required", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        if (!models.release_hold(lease)) {
            res_err(res, format_error_response("unknown or expired lease '" + lease + "'", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_router_models_sse = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        auto sse_client = std::make_shared<server_models_sse_client>(models.sse);
        res->next = [this, sse_client, &req](std::string & output) -> bool {
            auto result = sse_client->next([&]() {
                return stopping.load(std::memory_order_relaxed) || req.should_stop();
            });
            if (result == nullptr) {
                return false; // client disconnected or should_stop
            }
            output = "data: " + safe_json_to_str(result->to_json()) + "\n\n";
            return true; // listen for the next event
        };
        return res;
    };

    this->post_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        json body = json::parse(req.body);
        std::string name = json_value(body, "model", std::string());
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        common_params p;
        p.model.hf_repo  = name;
        p.hf_token       = params.hf_token;

        // validate by fetching metadata
        bool ok = false;
        try {
            common_models_handler_init(p, LLAMA_EXAMPLE_SERVER);
            ok = true;
        } catch (...) {
            SRV_ERR("unknown error while validating model '%s'\n", name.c_str());
            // other exceptions will be handled by the outer ex_wrapper()
            throw;
        }

        if (!ok) {
            throw std::invalid_argument("model validation failed, unable to download");
        }

        // reject if model already exists
        if (models.has_model(name)) {
            throw std::invalid_argument("model '" + name + "' already exists");
        }

        // then, proceed with the actual download
        SRV_INF("starting download for model '%s'\n", name.c_str());
        {
            server_models::load_options load_opts;
            load_opts.mode = SERVER_CHILD_MODE_DOWNLOAD;
            load_opts.custom_meta = server_model_meta{};
            load_opts.custom_meta->source = SERVER_MODEL_SOURCE_CACHE;
            load_opts.custom_meta->name   = name;
            models.load(name, load_opts);
        }

        res_ok(res, {{"success", true}});
        return res;
    };

    this->del_router_models = [this](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();

        std::string name = req.get_param("model");
        if (name.empty()) {
            throw std::invalid_argument("model must be a non-empty string");
        }

        models.remove(name); // throws on error

        res_ok(res, {{"success", true}});
        return res;
    };

    this->router_stream_get = [this](const server_http_req & req) {
        // GET /v1/stream?conv_id=<id>&from=N. resolve the owning child from the conv_id -> model
        // map, 404 when nothing maps
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::optional<server_model_meta> owner = resolve_child_for_conv(models, conv_id);
        if (!owner.has_value()) {
            // a registered conv whose model is still loading earns a retry: the session appears
            // once the load ends and the pending request reaches the child
            auto tracked = models.conv_models.lookup(conv_id);
            auto meta = tracked.has_value() ? models.get_meta(*tracked) : std::nullopt;
            bool transient = meta.has_value() && (meta->status == SERVER_MODEL_STATUS_LOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADING ||
                                                  meta->status == SERVER_MODEL_STATUS_DOWNLOADED);
            if (transient) {
                res_err(res, format_error_response("Stream owner model is loading, retry later", ERROR_TYPE_UNAVAILABLE));
            } else {
                res_err(res, format_error_response("Stream not found or expired", ERROR_TYPE_NOT_FOUND));
            }
            return res;
        }
        std::string from = req.get_param("from");
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        if (!from.empty()) {
            child_path += "&from=" + from;
        }
        SRV_TRC("proxying stream resume to model %s on port %d, path=%s\n",
                owner->name.c_str(), owner->port, child_path.c_str());
        auto proxy = std::make_unique<server_http_proxy>(
                "GET",
                "http",
                CHILD_ADDR,
                owner->port,
                child_path,
                req.headers,
                req.body,
                req.files,
                req.should_stop,
                params.timeout_read,
                params.timeout_write);
        return std::unique_ptr<server_http_res>(std::move(proxy));
    };

    this->router_streams_lookup = [this](const server_http_req & req) {
        // POST /v1/streams/lookup. resolve each requested conv id to its owning child via the
        // map, group the ids per child, and query only the children that actually own some of
        // them instead of fanning out to every ready child. a child only answers for the ids
        // it owns, never lists anything else
        auto res = std::make_unique<server_http_res>();
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string() && !v.get<std::string>().empty()) {
                        requested.push_back(v.get<std::string>());
                    }
                }
            }
        } catch (const std::exception &) {
            res_ok(res, json::array());
            return res;
        }

        // group requested ids by the child port that owns them, drop ids that map to nothing
        std::unordered_map<int, json> per_child;
        for (const auto & cid : requested) {
            auto owner = resolve_child_for_conv(models, cid);
            if (!owner.has_value()) {
                continue;
            }
            per_child[owner->port].push_back(cid);
        }

        json aggregated = json::array();
        for (auto & [port, ids] : per_child) {
            json child_body = {{"conversation_ids", ids}};
            httplib::Client cli(CHILD_ADDR, port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Post("/v1/streams/lookup", child_body.dump(), "application/json");
            if (!resp || resp->status != 200) {
                continue;
            }
            try {
                json child_arr = json::parse(resp->body);
                if (!child_arr.is_array()) {
                    continue;
                }
                for (auto & entry : child_arr) {
                    if (entry.is_object()) {
                        aggregated.push_back(entry);
                    }
                }
            } catch (const std::exception &) {
                continue;
            }
        }
        res_ok(res, aggregated);
        return res;
    };

    this->router_stream_delete = [this](const server_http_req & req) {
        // DELETE /v1/stream?conv_id=<id>. resolve the owning child via the map and forward only to
        // it, evict_and_cancel is idempotent on the child
        auto res = std::make_unique<server_http_res>();
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            res_err(res, format_error_response("Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST));
            return res;
        }
        std::string child_path = "/v1/stream?conv_id=" + encode_qs(conv_id);
        auto owner = resolve_child_for_conv(models, conv_id);
        if (owner.has_value()) {
            httplib::Client cli(CHILD_ADDR, owner->port);
            cli.set_connection_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_read_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            cli.set_write_timeout(0, STREAM_LOOKUP_TIMEOUT_MS * 1000);
            auto resp = cli.Delete(child_path.c_str());
            (void) resp; // the child logs its own miss when the session is unknown there
        } else if (auto tracked = models.conv_models.lookup(conv_id); tracked.has_value()) {
            // the entry exists but its model is still loading: the forget below erases it,
            // which cancels the request parked in proxy_post before the generation starts
            SRV_INF("router stop for conv_id=%s while model name=%s is loading, cancelling the pending request\n",
                    conv_id.c_str(), tracked->c_str());
        } else {
            SRV_WRN("router stop for unknown conv_id=%s, no owning child in the conv map\n",
                    conv_id.c_str());
        }
        // drop the tracking entry, the session is being torn down
        models.conv_models.forget(conv_id);
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}



//
// server_http_proxy
//

// NOTE(fork): our bounded/blocking pipe_t used to live here. Upstream added its own
// server_pipe in server-common.h during the 2026-07-31 sync, so the byte-accounting
// backpressure was grafted onto that type (see its write(data, bytes) overload) and
// this duplicate was removed.

static std::string to_lower_copy(const std::string & value) {
    std::string lowered(value.size(), '\0');
    std::transform(value.begin(), value.end(), lowered.begin(), [](unsigned char c) { return std::tolower(c); });
    return lowered;
}

static bool should_strip_proxy_header(const std::string & header_name) {
    // Headers that get duplicated when router forwards child responses
    if (header_name == "server" ||
        header_name == "transfer-encoding" ||
        header_name == "content-length" || // quick fix for https://github.com/ggml-org/llama.cpp/issues/17710
        header_name == "keep-alive") {
        return true;
    }

    // Router injects CORS, child also sends them: duplicate
    if (header_name.rfind("access-control-", 0) == 0) {
        return true;
    }

    return false;
}

static std::string generate_multipart_boundary() {
    thread_local std::mt19937 gen(std::random_device{}());
    static const char chars[] = "0123456789abcdefghijklmnopqrstuvwxyz";
    std::uniform_int_distribution<> dis(0, sizeof(chars) - 2);
    std::string boundary = "----llama-cpp-proxy-";
    for (int i = 0; i < 16; i++) {
        boundary += chars[dis(gen)];
    }
    return boundary;
}

static std::string build_multipart_body(
        const json & form_fields,
        const std::map<std::string, uploaded_file> & files,
        const std::string & boundary) {
    static auto sanitize_field = [](const std::string & text) {
        std::string result;
        result.reserve(text.size());
        for (char c : text) {
            if (c != '\n' && c != '\r' && c != '"') {
                result += c;
            }
        }
        return result;
    };

    std::ostringstream body;

    for (const auto & [key, value] : form_fields.items()) {
        if (value.is_array()) {
            for (const auto & item : value) {
                body << "--" << boundary << "\r\n";
                body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
                body << "\r\n";
                if (!item.is_string()) {
                    throw std::invalid_argument("expected string");
                }
                body << item.get<std::string>() << "\r\n";
            }
        } else {
            body << "--" << boundary << "\r\n";
            body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"\r\n";
            body << "\r\n";
            if (!value.is_string()) {
                throw std::invalid_argument("expected string");
            }
            body << value.get<std::string>() << "\r\n";
        }
    }

    for (const auto & [key, file] : files) {
        body << "--" << boundary << "\r\n";
        body << "Content-Disposition: form-data; name=\"" << sanitize_field(key) << "\"";
        if (!file.filename.empty()) {
            body << "; filename=\"" << sanitize_field(file.filename) << "\"";
        }
        body << "\r\n";
        if (!file.content_type.empty()) {
            body << "Content-Type: " << sanitize_field(file.content_type) << "\r\n";
        } else {
            body << "Content-Type: application/octet-stream\r\n";
        }
        body << "\r\n";
        body.write(reinterpret_cast<const char*>(file.data.data()), file.data.size());
        body << "\r\n";
    }

    body << "--" << boundary << "--\r\n";
    return body.str();
}

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & scheme,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::map<std::string, uploaded_file> & files,
        const std::function<bool()> should_stop,
        int32_t timeout_read,
        int32_t timeout_write
        ) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::ClientImpl>(host, port);
    auto pipe = std::make_shared<server_pipe<msg_t>>();

    if (scheme == "https") {
#ifdef CPPHTTPLIB_OPENSSL_SUPPORT
        cli.reset(new httplib::SSLClient(host, port));
#else
        throw std::runtime_error("HTTPS requested but CPPHTTPLIB_OPENSSL_SUPPORT is not defined");
#endif
    }

    // setup Client
    cli->set_follow_location(true);
    cli->set_connection_timeout(timeout_read, 0); // use --timeout value instead of hardcoded 5 s
    cli->set_write_timeout(timeout_read, 0); // reversed for cli (client) vs srv (server)
    cli->set_read_timeout(timeout_write, 0);
    this->status = 500; // to be overwritten upon response
    this->cleanup_pipes = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // build the header message forwarded to the reader thread, stripping internal proxy headers
    auto make_header_msg = [](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            const auto lowered = to_lower_copy(key);
            if (should_strip_proxy_header(lowered)) {
                continue;
            }
            if (lowered == "content-type") {
                msg.content_type = value;
                continue;
            }
            msg.headers[key] = value;
        }
        return msg;
    };

    // true once response_handler has already forwarded the headers
    auto headers_sent = std::make_shared<std::atomic<bool>>(false);

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, headers_sent, make_header_msg](const httplib::Response & response) {
        headers_sent->store(true);
        // headers carry no body bytes, but go through the (msg, bytes) overload
        // so the fork's backpressure accounting sees every write.
        msg_t msg = make_header_msg(response);
        const size_t header_bytes = msg.data.size();
        return pipe->write(std::move(msg), header_bytes); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length), ""}, data_length);
    };

    // when files are present, the body was converted from multipart form data to JSON
    // we need to reconstruct the multipart body for the downstream server
    std::string effective_body = body;
    std::string override_content_type;
    bool has_files = !files.empty();

    if (has_files) {
        json form_fields = json::parse_no_throw(body);
        if (!form_fields.is_discarded()) {
            auto boundary = generate_multipart_boundary();
            effective_body = build_multipart_body(form_fields, files, boundary);
            override_content_type = "multipart/form-data; boundary=" + boundary;
        } else {
            throw std::runtime_error("failed to parse multipart form fields JSON");
        }
    }

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            const auto lowered = to_lower_copy(key);
            if (lowered == "accept-encoding") {
                // disable Accept-Encoding to avoid compressed responses
                continue;
            }
            if (lowered == "transfer-encoding") {
                // the body is already decoded
                continue;
            }
            if (lowered == "content-length") {
                // let httplib calculate Content-Length from the actual body
                continue;
            }
            if (lowered == "content-type") {
                if (has_files) {
                    // we set our own Content-Type with the new boundary
                    continue;
                }
                // when no files but the original request was multipart,
                // the body is now JSON, so correct the Content-Type
                if (value.find("multipart/form-data") != std::string::npos) {
                    override_content_type = "application/json; charset=utf-8";
                    continue;
                }
            }
            if (lowered == "host") {
                bool is_default_port = (scheme == "https" && port == 443) || (scheme == "http" && port == 80);
                const std::string url_host = common_http_format_host(host);
                req.set_header(key, is_default_port ? url_host : url_host + ":" + std::to_string(port));
            } else {
                req.set_header(key, value);
            }
        }
        req.body = effective_body;
        if (!override_content_type.empty()) {
            req.set_header("Content-Type", override_content_type);
        }
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req, headers_sent, make_header_msg]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, "", ""}, 0); // header
            std::string err_body = "proxy error: " + err_str;
            const size_t err_bytes = err_body.size();
            pipe->write({{}, 0, std::move(err_body), ""}, err_bytes); // body
        } else if (!headers_sent->load()) {
            // httplib skips response_handler for bodyless statuses like 204, send headers here instead
            msg_t hdr = make_header_msg(*result);
            const size_t hdr_bytes = hdr.data.size();
            pipe->write(std::move(hdr), hdr_bytes);
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    {
        msg_t header;
        if (pipe->read(header, should_stop)) {
            SRV_DBG("%s", "received response headers\n");
            this->status  = header.status;
            this->headers = std::move(header.headers);
            if (!header.content_type.empty()) {
                this->content_type = std::move(header.content_type);
            }
        } else {
            SRV_DBG("%s", "no response headers received (request cancelled?)\n");
        }
    }
}
