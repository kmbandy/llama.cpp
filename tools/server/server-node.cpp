#include "server-node.h"

#include "server-router-group-lifecycle.h"
#include "server-router-policy.h"
#include "server-router-probe.h"

#include "common.h"
#include "http.h" // common_http_get_free_port
#include "log.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>

#ifndef _WIN32
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace fs = std::filesystem;

#define NODE_INF(fmt, ...) LOG_INF("node: " fmt, __VA_ARGS__)
#define NODE_WRN(fmt, ...) LOG_WRN("node: " fmt, __VA_ARGS__)

// same command server-models.cpp writes to a child's stdin to make it exit gracefully
static constexpr const char * NODE_CMD_CHILD_EXIT = "cmd_router_to_child:exit";

static int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static int64_t unix_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

static bool read_whole_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

// NUL-separated blob (environ, cmdline) -> entries
static std::vector<std::string> split_nul(const std::string & blob) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start < blob.size()) {
        size_t end = blob.find('\0', start);
        if (end == std::string::npos) {
            end = blob.size();
        }
        out.push_back(blob.substr(start, end - start));
        start = end + 1;
    }
    return out;
}

static std::string proc_dir(const std::string & root, int pid) {
    return (root.empty() ? std::string() : root) + "/proc/" + std::to_string(pid);
}

// the environment of a process as NAME=VALUE entries; empty when unreadable
static std::vector<std::string> read_proc_environ(const std::string & root, int pid) {
    std::string blob;
    if (!read_whole_file(proc_dir(root, pid) + "/environ", blob)) {
        return {};
    }
    return split_nul(blob);
}

// The port a child listens on: --port / --listen in its argv; 0 if none. Never read from env:
// an inherited LLAMA_ARG_PORT is the node's own, not the child's.
static int child_port(const std::vector<std::string> & args) {
    std::string host;
    const int port = router_launch_endpoint(args, host);
    return port > 0 ? port : 0;
}

#ifndef _WIN32
static constexpr unsigned long NODE_PF_KTHREAD = 0x00200000; // task flag of a kernel thread

// state and flags of a PID from the real /proc/<pid>/stat; false when it does not exist
static bool read_real_stat(int pid, char & state, unsigned long & flags) {
    std::string stat;
    if (pid <= 0 || !read_whole_file("/proc/" + std::to_string(pid) + "/stat", stat)) {
        return false;
    }
    const size_t rp = stat.rfind(')');
    if (rp == std::string::npos || rp + 2 >= stat.size()) {
        return false;
    }
    state = stat[rp + 2];
    // after ")": state ppid pgrp session tty_nr tpgid flags ...
    std::istringstream rest(stat.substr(rp + 2));
    std::string field;
    flags = 0;
    for (int i = 0; i < 7 && (rest >> field); i++) {
        if (i == 6) {
            flags = std::strtoul(field.c_str(), nullptr, 10);
        }
    }
    return true;
}

// Before a signal (the real /proc, never the fake root: signals are real): pid is alive, not a
// zombie, and verifiably still carries LLAMA_ROUTER_GEN=gen, i.e. it is the same process and
// not a reused PID. Strict: a process whose environ cannot be read is not signalled.
static bool same_tagged_process(int pid, const std::string & gen) {
    char state = 0;
    unsigned long flags = 0;
    if (!read_real_stat(pid, state, flags) || state == 'Z' || state == 'X') {
        return false;
    }
    std::string now_gen;
    return router_env_get(read_proc_environ("", pid), ROUTER_ENV_GEN, now_gen) && now_gen == gen;
}

// Liveness of a process that is not our child (adopted, orphan): gone once its PID is free, a
// zombie, a kernel thread, or a different process (a non-empty environ without our gen). A
// process that is exiting has already dropped its memory map, so its environ reads empty for
// a while before it turns into a zombie; it still holds its files (GPU memory) then, so it
// counts as alive rather than as "exited" too early.
static bool tagged_process_alive(int pid, const std::string & gen) {
    char state = 0;
    unsigned long flags = 0;
    if (!read_real_stat(pid, state, flags) || state == 'Z' || state == 'X' || (flags & NODE_PF_KTHREAD)) {
        return false;
    }
    const std::vector<std::string> env = read_proc_environ("", pid);
    if (env.empty()) {
        return true; // exiting (or unreadable): the PID is still taken, by the same process
    }
    std::string now_gen;
    return router_env_get(env, ROUTER_ENV_GEN, now_gen) && now_gen == gen;
}
#endif

//
// config
//

std::vector<std::string> server_node_default_env() {
    std::vector<std::string> env;
#ifndef _WIN32
    if (environ != nullptr) {
        for (char ** e = environ; *e != nullptr; e++) {
            env.emplace_back(*e);
        }
    }
#endif
    // The router's reserved names (the same set unset_reserved_args keeps out of a child's
    // options: TLS, API keys, model registry, board, node flags, LLAMA_ARG_ROUTER_*), plus this
    // process's own address and model: a llama-server child that inherited LLAMA_ARG_ROUTER_NODE
    // would come up as a node, one that inherited LLAMA_ARG_PORT would bind the node's port.
    static const char * const own[] = { "LLAMA_ARG_HOST", "LLAMA_ARG_PORT", "LLAMA_ARG_MODEL",
                                         "LLAMA_ARG_MMPROJ", "LLAMA_ARG_ALIAS", "LLAMA_ARG_HF_REPO" };
    env.erase(std::remove_if(env.begin(), env.end(), [](const std::string & e) {
        const std::string key = e.substr(0, e.find('='));
        return router_is_reserved_option_key(key) ||
               std::any_of(std::begin(own), std::end(own), [&](const char * k) { return key == k; });
    }), env.end());
    return env;
}

server_node_config server_node_default_config() {
    server_node_config cfg;
    cfg.base_env = server_node_default_env();
#ifndef _WIN32
    cfg.uid      = (unsigned) getuid();
    cfg.self_pid = (int) getpid();
    std::error_code ec;
    const fs::path exe = fs::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        cfg.exe = exe.string();
    }
#endif
    return cfg;
}

//
// server_node
//

struct server_node::child {
    std::string                     name;
    std::string                     gen;
    std::unique_ptr<server_subproc> proc;      // null when adopted
    int                             pid  = 0;
    int                             port = 0;
    bool                            adopted   = false;
    bool                            exited    = false; // reaped (adopted: seen gone)
    bool                            stopping  = false;
    bool                            killed    = false;
    bool                            exit_sent = false;
    int                             exit_code = -1;
    int64_t                         kill_deadline = 0; // steady ms, 0 = none
    int64_t                         started_ms    = 0; // unix ms
    std::string                     buf;               // partial output line (reaper only)
    bool                            eof = false;
};

struct server_node::orphan {
    int         pid = 0;
    std::string name;
    std::string gen;
    int64_t     adopt_deadline = 0; // steady ms
    bool        terminating    = false;
    bool        killed         = false;
    int64_t     kill_deadline  = 0;
    bool        done           = false; // adopted or gone
};

server_node::server_node(server_node_config cfg_in) : cfg(std::move(cfg_in)) {
    th = std::thread([this]() { run(); });
}

server_node::~server_node() {
    // closing first: a spawn that has not inserted its child yet sees it under `mu` and kills
    // its child itself, so nothing lands in the table after the stop loop below
    close_events();
    {
        std::lock_guard<std::mutex> lk(mu);
        const int64_t deadline = steady_ms() + cfg.shutdown_grace_ms;
        size_t n = 0;
        for (auto & [name, c] : table) {
            if (c->exited) {
                continue;
            }
            n++;
            c->stopping = true;
#ifndef _WIN32
            if (c->adopted) {
                if (same_tagged_process(c->pid, c->gen)) {
                    kill(c->pid, SIGTERM);
                }
            } else {
                if (!c->exit_sent) {
                    FILE * f = c->proc->sproc.stdin_file();
                    if (f) {
                        fprintf(f, "%s\n", NODE_CMD_CHILD_EXIT);
                        fflush(f);
                    }
                    c->exit_sent = true;
                }
                if (c->proc->sproc.pid() > 0) {
                    kill(c->proc->sproc.pid(), SIGTERM);
                }
            }
#endif
            if (c->kill_deadline == 0 || c->kill_deadline > deadline) {
                c->kill_deadline = deadline;
            }
        }
        if (n > 0) {
            NODE_INF("stopping %zu child(ren), SIGKILL after %" PRId64 " ms\n", n, cfg.shutdown_grace_ms);
        }
    }
    waiter.wake();
    {
        std::unique_lock<std::mutex> lk(mu);
        // bounded: a child stuck in D state, or an adopted PID that keeps looking alive, must not
        // hang the exit; past the SIGKILL deadline plus a grace it is abandoned
        const auto give_up = std::chrono::steady_clock::now() +
                             std::chrono::milliseconds(std::max<int64_t>(0, cfg.shutdown_grace_ms) +
                                                       std::max<int64_t>(0, cfg.abandon_grace_ms));
        const bool all_gone = cv.wait_until(lk, give_up, [this]() {
            return std::all_of(table.begin(), table.end(), [](const auto & kv) { return kv.second->exited; });
        });
        if (!all_gone) {
            for (auto & [name, c] : table) {
                if (!c->exited) {
                    NODE_WRN("abandoning %s (pid %d): still not gone %" PRId64 " ms after SIGKILL\n",
                             name.c_str(), c->pid, cfg.abandon_grace_ms);
                    c->exited    = true;
                    c->exit_code = -1;
                }
            }
        }
        quit = true;
    }
    waiter.wake();
    th.join();
}

node_child_info server_node::info_of(const child & c) {
    node_child_info i;
    i.name       = c.name;
    i.gen        = c.gen;
    i.pid        = c.pid;
    i.port       = c.port;
    i.status     = c.exited ? "exited" : c.stopping ? "stopping" : "running";
    i.exit_code  = c.exit_code;
    i.killed     = c.killed;
    i.adopted    = c.adopted;
    i.started_ms = c.started_ms;
    return i;
}

static json info_json(const node_child_info & i) {
    json j = json::object();
    j["name"]       = i.name;
    j["gen"]        = i.gen;
    j["pid"]        = i.pid;
    j["port"]       = i.port;
    j["status"]     = i.status;
    j["exit_code"]  = i.exit_code;
    j["killed"]     = i.killed;
    j["adopted"]    = i.adopted;
    j["started_ms"] = i.started_ms;
    return j;
}

json server_node::child_event_json_locked(const child & c) const {
    json ev = info_json(info_of(c));
    ev["type"] = "child";
    return ev;
}

void server_node::push_event_locked(json ev) {
    {
        std::lock_guard<std::mutex> lk(ev_mu);
        ev["seq"] = (uint64_t) (ev_first + events.size());
        events.push_back(std::move(ev));
        while (events.size() > std::max<size_t>(1, cfg.event_backlog)) {
            events.pop_front();
            ev_first++;
        }
    }
    ev_cv.notify_all();
}

node_child_info server_node::spawn(const node_spawn_request & req) {
    if (req.name.empty() || req.name.find_first_of("\n\r") != std::string::npos) {
        throw server_node_error(400, "name must be a non-empty single-line string");
    }
    if (req.gen.empty()) {
        throw server_node_error(400, "gen must be a non-empty string");
    }
    if (req.args.empty() || req.args[0].empty()) {
        throw server_node_error(400, "args must be a non-empty argv");
    }
    if (!fs::path(req.args[0]).is_absolute()) {
        throw server_node_error(400, "args[0] must be an absolute path (PATH is never searched): " + req.args[0]);
    }

    if (req.port < 0 || req.port > 65535) {
        throw server_node_error(400, "port must be 0..65535");
    }
    std::vector<std::string> env = cfg.base_env;
    for (const auto & e : req.env) {
        const std::string bad = router_env_override_error(e);
        if (!bad.empty()) {
            throw server_node_error(400, bad);
        }
    }
    // the router's own preset `env` semantics: KEY=VALUE sets, -KEY unsets
    router_env_apply_overrides(env, req.env);
    router_env_set(env, ROUTER_ENV_GEN, req.gen);
    router_env_set(env, ROUTER_ENV_ROUTER_PID, std::to_string(cfg.self_pid));
    router_env_set(env, ROUTER_ENV_CHILD, req.name);

    const std::string bad_tmp = env_temp_dir_violation(env);
    if (!bad_tmp.empty()) {
        throw server_node_error(400, "refusing to spawn '" + req.name + "': " + bad_tmp + " is not a directory");
    }

    std::vector<std::string> args = req.args;
    int port = req.port > 0 ? req.port : child_port(args);
    if (req.alloc_port) {
        // the leader cannot know which ports are free here: pick one, put it on the command line
        port = common_http_get_free_port();
        if (port <= 0) {
            throw server_node_error(500, "no free port for '" + req.name + "'");
        }
        router_args_set_port(args, port);
    }

    {
        std::lock_guard<std::mutex> lk(mu);
        if (quit || closing_flag.load()) {
            throw server_node_error(503, "node is shutting down");
        }
        auto it = table.find(req.name);
        if ((it != table.end() && !it->second->exited) || spawning.count(req.name)) {
            throw server_node_error(409, "a child named '" + req.name + "' is already running");
        }
        spawning.insert(req.name);
    }

    // spawn without the lock: fork/exec of a big binary is not instant
    auto proc = std::make_unique<server_subproc>();
    const int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
    const bool ok = proc->sproc.create(args, options, env);
    if (ok) {
        proc->has_output(); // non-blocking pipe before the reaper reads it
    }

    node_child_info info;
    bool closed = false;
    {
        std::lock_guard<std::mutex> lk(mu);
        spawning.erase(req.name);
        if (!ok) {
            throw server_node_error(500, "failed to spawn '" + req.name + "': " + req.args[0]);
        }
        // the node started closing while we spawned: the destructor's stop loop has run (or
        // will not see this child), so it must not enter the table
        closed = quit || closing_flag.load();
        if (!closed) {
            auto c = std::make_shared<child>();
            c->name       = req.name;
            c->gen        = req.gen;
            c->pid        = proc->sproc.pid();
            c->port       = port;
            c->proc       = std::move(proc);
            c->started_ms = unix_ms();
            table[req.name] = c;
            info = info_of(*c);
            push_event_locked(child_event_json_locked(*c));
        }
    }
    if (closed) {
        proc->terminate(); // SIGKILL: it never ran as one of ours
        proc->join();
        throw server_node_error(503, "node is shutting down");
    }
    NODE_INF("spawned %s (pid %d, port %d, gen %s): %s\n", info.name.c_str(), info.pid, info.port,
             req.gen.c_str(), req.args[0].c_str());
    waiter.wake();
    return info;
}

node_child_info server_node::stop(const std::string & name, int timeout_s, const std::string & method) {
    if (method != "both" && method != "stdin" && method != "term") {
        throw server_node_error(400, "method must be one of both, stdin, term");
    }
    if (timeout_s < 0) {
        throw server_node_error(400, "timeout_s must be >= 0");
    }
    node_child_info info;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = table.find(name);
        if (it == table.end()) {
            throw server_node_error(404, "no child named '" + name + "'");
        }
        child & c = *it->second;
        if (c.exited) {
            return info_of(c);
        }
        const bool was_stopping = c.stopping;
        c.stopping = true;
#ifndef _WIN32
        if (c.adopted) {
            // no stdin to write to: an adopted process only gets signals
            if (same_tagged_process(c.pid, c.gen)) {
                kill(c.pid, SIGTERM);
            }
        } else {
            if ((method == "both" || method == "stdin") && !c.exit_sent) {
                FILE * f = c.proc->sproc.stdin_file();
                if (f) {
                    fprintf(f, "%s\n", NODE_CMD_CHILD_EXIT);
                    fflush(f);
                }
                c.exit_sent = true;
            }
            if ((method == "both" || method == "term") && c.proc->sproc.pid() > 0) {
                kill(c.proc->sproc.pid(), SIGTERM);
            }
        }
#endif
        const int64_t deadline = steady_ms() + (int64_t) timeout_s * 1000;
        if (c.kill_deadline == 0 || deadline < c.kill_deadline) {
            c.kill_deadline = deadline;
        }
        NODE_INF("stopping %s (pid %d, %s), SIGKILL in %d s\n", name.c_str(), c.pid, method.c_str(), timeout_s);
        if (!was_stopping) {
            push_event_locked(child_event_json_locked(c));
        }
        info = info_of(c);
    }
    waiter.wake();
    return info;
}

bool server_node::wait_exit(const std::string & name, int64_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mu);
    auto it = table.find(name);
    if (it == table.end()) {
        return false;
    }
    std::shared_ptr<child> c = it->second;
    return cv.wait_for(lk, std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms)), [&]() { return c->exited; });
}

node_child_info server_node::signal(const std::string & name, int sig) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = table.find(name);
    if (it == table.end()) {
        throw server_node_error(404, "no child named '" + name + "'");
    }
    child & c = *it->second;
    if (c.exited) {
        throw server_node_error(409, "child '" + name + "' is not running");
    }
#ifndef _WIN32
    // under the table lock, and only the reaper reaps (also under it): the PID is still ours
    const int pid = c.adopted ? (same_tagged_process(c.pid, c.gen) ? c.pid : 0) : c.proc->sproc.pid();
    if (pid <= 0) {
        throw server_node_error(409, "child '" + name + "' is not running");
    }
    if (kill(pid, sig) != 0) {
        throw server_node_error(400, string_format("cannot send signal %d to '%s'", sig, name.c_str()));
    }
    NODE_INF("sent signal %d to %s (pid %d)\n", sig, name.c_str(), pid);
#else
    (void) sig;
    throw server_node_error(501, "signals are not supported on Windows");
#endif
    return info_of(c);
}

size_t server_node::collect_orphans() {
#ifdef _WIN32
    return 0;
#else
    // every process that carries a router generation and whose router is not its ancestor
    // ("" never equals a real generation)
    const std::vector<int> pids = router_find_stale_children(cfg.proc_root, "", cfg.uid, cfg.self_pid);
    const int64_t now = steady_ms();
    size_t n = 0;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (int pid : pids) {
            if (std::any_of(orphan_list.begin(), orphan_list.end(), [&](const auto & o) { return o->pid == pid; })) {
                continue;
            }
            const std::vector<std::string> env = read_proc_environ(cfg.proc_root, pid);
            auto o = std::make_shared<orphan>();
            o->pid = pid;
            router_env_get(env, ROUTER_ENV_GEN, o->gen);
            router_env_get(env, ROUTER_ENV_CHILD, o->name);
            // nameless: nobody can ask for it by name, so it goes now
            o->adopt_deadline = o->name.empty() ? now : now + cfg.adopt_window_ms;
            NODE_WRN("found pid %d (name '%s', gen %s) left behind by a previous generation; %s\n", pid,
                     o->name.c_str(), o->gen.c_str(), o->name.empty() ? "stopping it" : "adoptable until the window closes");
            json ev = json::object();
            ev["type"]   = "orphan";
            ev["action"] = "found";
            ev["pid"]    = pid;
            ev["name"]   = o->name;
            ev["gen"]    = o->gen;
            push_event_locked(ev);
            orphan_list.push_back(std::move(o));
            n++;
        }
    }
    waiter.wake();
    return n;
#endif
}

node_child_info server_node::adopt(const std::string & name, const std::string & gen) {
#ifdef _WIN32
    (void) name;
    (void) gen;
    throw server_node_error(501, "adoption is not supported on Windows");
#else
    std::lock_guard<std::mutex> lk(mu);
    auto it = table.find(name);
    if (it != table.end() && !it->second->exited) {
        throw server_node_error(409, "a child named '" + name + "' is already running");
    }
    std::shared_ptr<orphan> o;
    for (auto & cand : orphan_list) {
        if (!cand->done && !cand->terminating && cand->name == name && cand->gen == gen) {
            o = cand;
            break;
        }
    }
    if (!o) {
        throw server_node_error(404, "no adoptable orphan named '" + name + "' with gen " + gen);
    }
    o->done = true;
    if (!same_tagged_process(o->pid, o->gen)) {
        throw server_node_error(404, "orphan '" + name + "' (pid " + std::to_string(o->pid) + ") is gone");
    }

    std::vector<std::string> args;
    std::string cmdline;
    if (read_whole_file(proc_dir(cfg.proc_root, o->pid) + "/cmdline", cmdline)) {
        args = split_nul(cmdline);
    }
    auto c = std::make_shared<child>();
    c->name       = name;
    c->gen        = gen;
    c->pid        = o->pid;
    c->port       = child_port(args);
    c->adopted    = true;
    c->eof        = true;
    c->started_ms = unix_ms();
    table[name] = c;
    NODE_INF("adopted %s (pid %d, gen %s)\n", name.c_str(), c->pid, gen.c_str());
    push_event_locked(child_event_json_locked(*c));
    return info_of(*c);
#endif
}

std::vector<node_child_info> server_node::children() const {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<node_child_info> out;
    for (const auto & [name, c] : table) {
        out.push_back(info_of(*c));
    }
    return out;
}

std::vector<node_orphan_info> server_node::orphans() const {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<node_orphan_info> out;
    for (const auto & o : orphan_list) {
        if (o->done) {
            continue;
        }
        out.push_back({ o->pid, o->name, o->gen, o->terminating ? "terminating" : "waiting" });
    }
    return out;
}

json server_node::state() const {
    const std::vector<node_child_info>  kids = children();
    const std::vector<node_orphan_info> orph = orphans();

    // probes run without any lock
    const std::vector<proc_vram> vram = probe_fdinfo_vram(cfg.proc_root);

    json jchildren = json::array();
    for (const auto & k : kids) {
        json j = info_json(k);
        if (k.status != "exited") {
            const auto mem = probe_proc_mem(cfg.proc_root, k.pid);
            if (mem) {
                j["rss_anon"]  = mem->rss_anon;
                j["rss_shmem"] = mem->rss_shmem;
            } else {
                j["rss_anon"]  = nullptr;
                j["rss_shmem"] = nullptr;
            }
            json jv = json::array();
            for (const auto & v : vram) {
                if (v.pid == k.pid) {
                    jv.push_back(json{ { "pdev", v.pdev }, { "bytes", v.vram_bytes } });
                }
            }
            j["vram"] = jv;
        }
        jchildren.push_back(j);
    }

    json jorphans = json::array();
    for (const auto & o : orph) {
        jorphans.push_back(json{ { "pid", o.pid }, { "name", o.name }, { "gen", o.gen }, { "state", o.state } });
    }

    json jvram = json::array();
    for (const auto & v : vram) {
        jvram.push_back(json{ { "pid", v.pid }, { "pdev", v.pdev }, { "bytes", v.vram_bytes } });
    }

    json jdevices = json::array();
    {
        std::error_code ec;
        const fs::path dir = fs::path(cfg.proc_root.empty() ? "/" : cfg.proc_root) / "sys" / "bus" / "pci" / "devices";
        std::vector<std::string> pdevs;
        for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
            if (fs::exists(it->path() / "mem_info_vram_used", ec)) {
                pdevs.push_back(it->path().filename().string());
            }
        }
        std::sort(pdevs.begin(), pdevs.end());
        for (const auto & pdev : pdevs) {
            jdevices.push_back(json{ { "pdev", pdev }, { "vram_used", probe_sysfs_vram_used(cfg.proc_root, pdev) } });
        }
    }

    json out = json::object();
    out["node_pid"]      = cfg.self_pid;
    out["exe"]           = cfg.exe;
    out["time_ms"]       = unix_ms();
    out["children"]      = jchildren;
    out["orphans"]       = jorphans;
    out["vram"]          = jvram;
    out["mem_available"] = probe_mem_available(cfg.proc_root);
    out["devices"]       = jdevices;
    return out;
}

uint64_t server_node::next_seq() const {
    std::lock_guard<std::mutex> lk(ev_mu);
    return ev_first + events.size();
}

bool server_node::wait_events(uint64_t & cursor, std::vector<json> & out, int64_t timeout_ms) const {
    std::unique_lock<std::mutex> lk(ev_mu);
    ev_cv.wait_for(lk, std::chrono::milliseconds(std::max<int64_t>(0, timeout_ms)), [&]() {
        return closing_flag.load() || cursor < ev_first + events.size();
    });
    if (closing_flag.load()) {
        return false;
    }
    const uint64_t end = ev_first + events.size();
    if (cursor > end) {
        cursor = end;
    }
    if (cursor < ev_first) {
        json lost = json::object();
        lost["type"]   = "lost";
        lost["missed"] = (uint64_t) (ev_first - cursor);
        out.push_back(lost);
        cursor = ev_first;
    }
    for (uint64_t s = cursor; s < end; s++) {
        out.push_back(events[(size_t) (s - ev_first)]);
    }
    cursor = end;
    return true;
}

json server_node::heartbeat_json() const {
    json kids = json::array();
    for (const auto & k : children()) {
        json j = json::object();
        j["name"]   = k.name;
        j["gen"]    = k.gen;
        j["pid"]    = k.pid;
        j["port"]   = k.port;
        j["status"] = k.status;
        kids.push_back(j);
    }
    json hb = json::object();
    hb["type"]     = "heartbeat";
    hb["time_ms"]  = unix_ms();
    hb["node_pid"] = cfg.self_pid;
    hb["exe"]      = cfg.exe;
    hb["next_seq"] = next_seq();
    hb["children"] = kids;
    return hb;
}

void server_node::close_events() {
    {
        std::lock_guard<std::mutex> lk(ev_mu);
        closing_flag.store(true);
    }
    ev_cv.notify_all();
}

static std::string strip_eol(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

void server_node::handle_line_locked(child & c, const std::string & raw) {
    const std::string line = strip_eol(raw);
    LOG("[%s] %s\n", c.name.c_str(), line.c_str());
    json ev = json::object();
    ev["type"] = "line";
    ev["name"] = c.name;
    ev["gen"]  = c.gen;
    ev["pid"]  = c.pid;
    ev["line"] = line;
    push_event_locked(ev);
}

void server_node::read_output_locked(child & c) {
    static constexpr size_t max_line = 1024 * 1024;
    char chunk[4096];
    while (!c.eof && c.proc) {
        const int n = c.proc->read_output(chunk, sizeof(chunk));
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
            const size_t nl = c.buf.find('\n', start);
            if (nl == std::string::npos) {
                break;
            }
            handle_line_locked(c, c.buf.substr(start, nl - start));
            start = nl + 1;
        }
        c.buf.erase(0, start);
        if (c.buf.size() > max_line) {
            c.buf.clear(); // a child that never writes a newline must not grow this without bound
        }
    }
    if (c.eof && !c.buf.empty()) {
        handle_line_locked(c, c.buf);
        c.buf.clear();
    }
}

void server_node::run() {
    while (true) {
        std::vector<std::shared_ptr<child>> watch; // keeps the procs alive while we wait unlocked
        std::vector<server_subproc *>       procs;
        int64_t                             timeout = -1;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (quit) {
                return;
            }
            const int64_t now = steady_ms();
            auto bound = [&](int64_t t) { timeout = timeout < 0 ? t : std::min(timeout, t); };
            for (const auto & [name, c] : table) {
                if (c->exited) {
                    continue;
                }
                // exits are polled, not inferred from EOF: a grandchild may hold the pipe open
                bound(c->adopted ? 200 : 100);
                if (c->kill_deadline) {
                    bound(std::max<int64_t>(0, c->kill_deadline - now));
                }
                if (!c->eof && c->proc) {
                    watch.push_back(c);
                    procs.push_back(c->proc.get());
                }
            }
            for (const auto & o : orphan_list) {
                if (!o->done) {
                    bound(200);
                }
            }
        }

        std::vector<bool> ready;
        waiter.wait(procs, ready, timeout);

        {
            std::lock_guard<std::mutex> lk(mu);
            for (size_t k = 0; k < watch.size(); k++) {
                if (k < ready.size() && ready[k]) {
                    read_output_locked(*watch[k]);
                }
            }
            const int64_t now = steady_ms();
            for (auto & [name, cp] : table) {
                child & c = *cp;
                if (c.exited) {
                    continue;
                }
                if (c.kill_deadline && now >= c.kill_deadline) {
                    c.kill_deadline = 0;
#ifndef _WIN32
                    if (c.adopted) {
                        if (same_tagged_process(c.pid, c.gen)) {
                            NODE_WRN("SIGKILL %s (pid %d, adopted): stop timeout\n", c.name.c_str(), c.pid);
                            kill(c.pid, SIGKILL);
                            c.killed = true;
                        }
                    } else
#endif
                    if (c.proc->sproc.pid() > 0) {
                        NODE_WRN("SIGKILL %s (pid %d): stop timeout\n", c.name.c_str(), c.pid);
                        c.proc->terminate();
                        c.killed = true;
                    }
                }
                bool gone = false;
                if (c.adopted) {
#ifndef _WIN32
                    gone = !tagged_process_alive(c.pid, c.gen);
#else
                    gone = true;
#endif
                } else if (!c.proc->is_alive()) {
                    read_output_locked(c); // whatever it wrote before going
                    if (!c.buf.empty()) {
                        handle_line_locked(c, c.buf);
                        c.buf.clear();
                    }
                    c.exit_code = c.proc->join(); // the only place a child is reaped
                    c.eof       = true;           // join() closed the pipe
                    gone        = true;
                }
                if (gone) {
                    c.exited        = true;
                    c.kill_deadline = 0;
                    NODE_INF("%s (pid %d) exited with status %d%s\n", c.name.c_str(), c.pid, c.exit_code,
                             c.killed ? " (killed)" : "");
                    push_event_locked(child_event_json_locked(c));
                }
            }

#ifndef _WIN32
            for (auto & o : orphan_list) {
                if (o->done) {
                    continue;
                }
                auto orphan_event = [&](const char * action) {
                    json ev = json::object();
                    ev["type"]   = "orphan";
                    ev["action"] = action;
                    ev["pid"]    = o->pid;
                    ev["name"]   = o->name;
                    ev["gen"]    = o->gen;
                    push_event_locked(ev);
                };
                if (!tagged_process_alive(o->pid, o->gen)) {
                    o->done = true;
                    orphan_event("gone");
                    continue;
                }
                if (!o->terminating && now >= o->adopt_deadline) {
                    NODE_WRN("SIGTERM orphan pid %d ('%s', gen %s): not adopted\n", o->pid, o->name.c_str(), o->gen.c_str());
                    if (same_tagged_process(o->pid, o->gen)) {
                        kill(o->pid, SIGTERM);
                    }
                    o->terminating   = true;
                    o->kill_deadline = now + cfg.orphan_kill_grace_ms;
                    orphan_event("term");
                } else if (o->terminating && !o->killed && now >= o->kill_deadline) {
                    NODE_WRN("SIGKILL orphan pid %d ('%s'): ignored SIGTERM\n", o->pid, o->name.c_str());
                    if (same_tagged_process(o->pid, o->gen)) {
                        kill(o->pid, SIGKILL);
                    }
                    o->killed = true;
                    orphan_event("kill");
                }
            }
#endif
            orphan_list.erase(std::remove_if(orphan_list.begin(), orphan_list.end(),
                                             [](const auto & o) { return o->done; }), orphan_list.end());
        }
        cv.notify_all();
    }
}

//
// HTTP layer
//

std::string server_node_read_token(const std::string & path, std::string & err) {
    err.clear();
    if (path.empty()) {
        err = "no token file given";
        return "";
    }
    std::ifstream f(path);
    if (!f) {
        err = "cannot read token file '" + path + "'";
        return "";
    }
    std::stringstream ss;
    ss << f.rdbuf();
    const std::string token = string_strip(ss.str());
    if (token.empty()) {
        err = "token file '" + path + "' is empty";
    }
    return token;
}

bool server_node_token_matches(const std::string & expected, const std::string & authorization) {
    static const std::string prefix = "bearer ";
    std::string given;
    if (authorization.size() >= prefix.size()) {
        bool is_bearer = true;
        for (size_t i = 0; i < prefix.size(); i++) {
            is_bearer = is_bearer && std::tolower((unsigned char) authorization[i]) == prefix[i];
        }
        if (is_bearer) {
            given = string_strip(authorization.substr(prefix.size()));
        }
    }
    // constant time in the token's content: every byte of the longer string is visited
    const size_t n = std::max(expected.size(), given.size());
    unsigned diff = expected.size() != given.size() ? 1u : 0u;
    for (size_t i = 0; i < n; i++) {
        const unsigned char a = i < expected.size() ? (unsigned char) expected[i] : 0;
        const unsigned char b = i < given.size() ? (unsigned char) given[i] : 0;
        diff |= (unsigned) (a ^ b);
    }
    return !expected.empty() && diff == 0;
}

std::vector<std::string> server_node_bind_hosts(const std::string & node_bind) {
    std::vector<std::string> out;
    std::string cur;
    for (size_t i = 0; i <= node_bind.size(); i++) {
        if (i == node_bind.size() || node_bind[i] == ',') {
            cur = string_strip(cur);
            if (!cur.empty()) {
                out.push_back(cur);
            }
            cur.clear();
        } else {
            cur.push_back(node_bind[i]);
        }
    }
    return out;
}

// "" when `host` resolves only to specific addresses; else why it is refused. Resolved the way
// the listener will see it (getaddrinfo, so "0", "000.0.0.0", "::0", "[::]", "::ffff:0.0.0.0"
// are all caught), not by spelling.
static std::string server_node_bind_wildcard_error(const std::string & host_in) {
    std::string host = host_in;
    if (host.size() >= 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    if (host.empty() || host == "*") {
        return "a wildcard address is refused, name the LAN / Tailscale address";
    }
#ifndef _WIN32
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * res = nullptr;
    const int rc = getaddrinfo(host.c_str(), nullptr, &hints, &res);
    if (rc != 0 || res == nullptr) {
        return std::string("cannot resolve it: ") + gai_strerror(rc);
    }
    bool wildcard = false;
    for (const addrinfo * ai = res; ai != nullptr; ai = ai->ai_next) {
        if (ai->ai_family == AF_INET) {
            const auto * a = (const sockaddr_in *) ai->ai_addr;
            wildcard = wildcard || a->sin_addr.s_addr == htonl(INADDR_ANY);
        } else if (ai->ai_family == AF_INET6) {
            const auto * a = (const sockaddr_in6 *) ai->ai_addr;
            const bool any    = IN6_IS_ADDR_UNSPECIFIED(&a->sin6_addr);
            // ::ffff:0.0.0.0 is the IPv4 wildcard in IPv6 clothing
            const bool mapped = IN6_IS_ADDR_V4MAPPED(&a->sin6_addr) &&
                                a->sin6_addr.s6_addr[12] == 0 && a->sin6_addr.s6_addr[13] == 0 &&
                                a->sin6_addr.s6_addr[14] == 0 && a->sin6_addr.s6_addr[15] == 0;
            wildcard = wildcard || any || mapped;
        }
    }
    freeaddrinfo(res);
    if (wildcard) {
        return "a wildcard address is refused, name the LAN / Tailscale address";
    }
#else
    if (host == "0.0.0.0" || host == "::") {
        return "a wildcard address is refused, name the LAN / Tailscale address";
    }
#endif
    return "";
}

std::string server_node_check_params(const common_params & params) {
    if (!params.router_node) {
        return "";
    }
    if (!params.router_board_url.empty()) {
        return "--router-node does not accept --board-url: only the leader router talks to the coordination board";
    }
    if (!params.model.path.empty() || !params.model.hf_repo.empty() || !params.model.docker_repo.empty()) {
        return "--router-node does not load a model: the leader sends every child's full argv";
    }
    if (!params.models_preset.empty()) {
        return "--router-node does not read presets (--models-preset): the leader sends every child's full argv";
    }
    if (params.node_token_file.empty()) {
        return "--router-node requires --node-token-file (the bearer token every route checks)";
    }
    std::string err;
    server_node_read_token(params.node_token_file, err);
    if (!err.empty()) {
        return "--node-token-file: " + err;
    }
    const std::vector<std::string> hosts = server_node_bind_hosts(params.node_bind);
    if (hosts.empty()) {
        return "--router-node requires --node-bind ADDR[,ADDR] (the LAN / Tailscale addresses to listen on)";
    }
    for (const auto & h : hosts) {
        if (string_ends_with(h, ".sock")) {
            return "--node-bind " + h + ": the leader reaches the node over TCP, a UNIX socket is refused";
        }
        const std::string why = server_node_bind_wildcard_error(h);
        if (!why.empty()) {
            return "--node-bind " + h + ": " + why;
        }
    }
    return "";
}

bool server_node_clamp_timeout_s(double v, int & out) {
    if (!std::isfinite(v)) {
        return false;
    }
    out = (int) std::min(3600.0, std::max(0.0, v));
    return true;
}

int server_node_parse_signal(const json & sig) {
    int n = -1;
    if (sig.is_number_integer()) {
        n = sig.get<int>();
    } else if (sig.is_string()) {
        std::string s = sig.get<std::string>();
        for (auto & ch : s) {
            ch = (char) std::toupper((unsigned char) ch);
        }
        if (!s.empty() && std::all_of(s.begin(), s.end(), [](char ch) { return std::isdigit((unsigned char) ch) != 0; })) {
            n = std::atoi(s.c_str());
        } else {
            if (s.rfind("SIG", 0) == 0) {
                s = s.substr(3);
            }
#ifndef _WIN32
            static const std::map<std::string, int> names = {
                { "HUP", SIGHUP }, { "INT", SIGINT }, { "QUIT", SIGQUIT }, { "KILL", SIGKILL },
                { "USR1", SIGUSR1 }, { "USR2", SIGUSR2 }, { "TERM", SIGTERM }, { "CONT", SIGCONT },
                { "STOP", SIGSTOP },
            };
            auto it = names.find(s);
            if (it != names.end()) {
                n = it->second;
            }
#endif
        }
    }
    return n >= 1 && n <= 64 ? n : -1;
}

static server_http_res_ptr node_json_res(int status, const json & body) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data   = safe_json_to_str(body);
    return res;
}

static server_http_res_ptr node_error_res(int status, const std::string & message) {
    json err = json::object();
    err["code"]    = status;
    err["message"] = message;
    err["type"]    = status == 401 ? "authentication_error" : status == 404 ? "not_found_error"
                   : status < 500 ? "invalid_request_error" : "server_error";
    json body = json::object();
    body["error"] = err;
    return node_json_res(status, body);
}

static std::string header_ci(const server_http_req & req, const std::string & key) {
    for (const auto & [k, v] : req.headers) {
        if (k.size() == key.size() && std::equal(k.begin(), k.end(), key.begin(), [](char a, char b) {
                return std::tolower((unsigned char) a) == std::tolower((unsigned char) b);
            })) {
            return v;
        }
    }
    return "";
}

static std::string body_string(const json & body, const char * key, bool required) {
    if (!body.contains(key) || body.at(key).is_null()) {
        if (required) {
            throw server_node_error(400, std::string(key) + " is required");
        }
        return "";
    }
    if (!body.at(key).is_string()) {
        throw server_node_error(400, std::string(key) + " must be a string");
    }
    return body.at(key).get<std::string>();
}

static json parse_body(const server_http_req & req) {
    json body = json::parse_no_throw(req.body.empty() ? "{}" : req.body);
    if (body.is_discarded() || !body.is_object()) {
        throw server_node_error(400, "request body must be a JSON object");
    }
    return body;
}

server_node_routes::server_node_routes(server_node & node, std::string token) : node(node), token(std::move(token)) {
    post_spawn = [this](const server_http_req & req) {
        const json body = parse_body(req);
        node_spawn_request r;
        r.name = body_string(body, "name", true);
        r.gen  = body_string(body, "gen", true);
        if (!body.contains("args") || !body.at("args").is_array()) {
            throw server_node_error(400, "args must be an array of strings");
        }
        for (const auto & a : body.at("args")) {
            if (!a.is_string()) {
                throw server_node_error(400, "args must be an array of strings");
            }
            r.args.push_back(a.get<std::string>());
        }
        if (body.contains("port") && !body.at("port").is_null()) {
            if (!body.at("port").is_number_integer()) {
                throw server_node_error(400, "port must be an integer");
            }
            r.port = body.at("port").get<int>();
        }
        if (body.contains("alloc_port") && !body.at("alloc_port").is_null()) {
            if (!body.at("alloc_port").is_boolean()) {
                throw server_node_error(400, "alloc_port must be a boolean");
            }
            r.alloc_port = body.at("alloc_port").get<bool>();
        }
        if (body.contains("env") && !body.at("env").is_null()) {
            const json & env = body.at("env");
            if (env.is_array()) {
                for (const auto & e : env) {
                    if (!e.is_string()) {
                        throw server_node_error(400, "env entries must be \"NAME=VALUE\" or \"-NAME\" strings");
                    }
                    r.env.push_back(e.get<std::string>());
                }
            } else if (env.is_object()) {
                for (const auto & kv : env.items()) {
                    if (kv.value().is_null()) {
                        r.env.push_back("-" + kv.key()); // {K: null} unsets K
                    } else if (kv.value().is_string()) {
                        r.env.push_back(kv.key() + "=" + kv.value().get<std::string>());
                    } else {
                        throw server_node_error(400, "env values must be strings (or null to unset)");
                    }
                }
            } else {
                throw server_node_error(400, "env must be an array of \"NAME=VALUE\" or an object");
            }
        }
        return node_json_res(200, info_json(this->node.spawn(r)));
    };

    post_stop = [this](const server_http_req & req) {
        const json body = parse_body(req);
        const std::string name = body_string(body, "name", true);
        int timeout_s = NODE_STOP_TIMEOUT_S_DEFAULT;
        if (body.contains("timeout_s") && !body.at("timeout_s").is_null()) {
            if (!body.at("timeout_s").is_number()) {
                throw server_node_error(400, "timeout_s must be a number");
            }
            if (!server_node_clamp_timeout_s(body.at("timeout_s").get<double>(), timeout_s)) {
                throw server_node_error(400, "timeout_s must be a finite number");
            }
        }
        std::string method = body_string(body, "method", false);
        if (method.empty()) {
            method = "both";
        }
        const bool wait = body.contains("wait") && body.at("wait").is_boolean() && body.at("wait").get<bool>();
        node_child_info info = this->node.stop(name, timeout_s, method);
        if (wait && info.status != "exited") {
            // the kill deadline is timeout_s; give the reaper a moment past it
            this->node.wait_exit(name, (int64_t) timeout_s * 1000 + 5000);
            for (const auto & c : this->node.children()) {
                if (c.name == name) {
                    info = c;
                }
            }
        }
        return node_json_res(200, info_json(info));
    };

    post_signal = [this](const server_http_req & req) {
        const json body = parse_body(req);
        const std::string name = body_string(body, "name", true);
        if (!body.contains("sig")) {
            throw server_node_error(400, "sig is required");
        }
        const int sig = server_node_parse_signal(body.at("sig"));
        if (sig < 0) {
            throw server_node_error(400, "unknown signal");
        }
        return node_json_res(200, info_json(this->node.signal(name, sig)));
    };

    post_adopt = [this](const server_http_req & req) {
        const json body = parse_body(req);
        const std::string name = body_string(body, "name", true);
        const std::string gen  = body_string(body, "gen", true);
        return node_json_res(200, info_json(this->node.adopt(name, gen)));
    };

    get_state = [this](const server_http_req &) {
        return node_json_res(200, this->node.state());
    };

    get_events = [this](const server_http_req & req) {
        struct sse_state {
            uint64_t cursor  = 0;
            int64_t  last_hb = 0;
        };
        auto st = std::make_shared<sse_state>();
        const std::string since = req.get_param("since");
        st->cursor = since.empty() ? this->node.next_seq() : (uint64_t) std::strtoull(since.c_str(), nullptr, 10);

        auto res = std::make_unique<server_http_res>();
        res->status       = 200;
        res->content_type = "text/event-stream";
        res->headers["Cache-Control"] = "no-cache";
        server_node * n = &this->node;
        // req outlives the stream (server-http keeps it until on_complete)
        res->next = [n, st, &req](std::string & output) -> bool {
            const int64_t hb_ms = std::max<int64_t>(100, n->config().heartbeat_ms);
            while (true) {
                if (req.should_stop() || n->closing()) {
                    return false;
                }
                const int64_t now = steady_ms();
                if (st->last_hb == 0 || now - st->last_hb >= hb_ms) {
                    // first chunk is a heartbeat: the subscriber learns next_seq and the children at once
                    st->last_hb = now;
                    output += "data: " + safe_json_to_str(n->heartbeat_json()) + "\n\n";
                    return true;
                }
                std::vector<json> evs;
                const int64_t wait = std::min<int64_t>(500, hb_ms - (now - st->last_hb));
                if (!n->wait_events(st->cursor, evs, wait)) {
                    return false;
                }
                if (!evs.empty()) {
                    for (const auto & ev : evs) {
                        output += "data: " + safe_json_to_str(ev) + "\n\n";
                    }
                    return true;
                }
            }
        };
        return res;
    };
}

void server_node_routes::register_routes(const server_http_context & http) const {
    // every route: bearer token first, then the handler; errors become JSON with their status
    auto guard = [this](const server_http_context::handler_t & h) -> server_http_context::handler_t {
        return [this, h](const server_http_req & req) -> server_http_res_ptr {
            if (!server_node_token_matches(token, header_ci(req, "Authorization"))) {
                NODE_WRN("unauthorized request to %s\n", req.path.c_str());
                return node_error_res(401, "missing or invalid bearer token");
            }
            try {
                return h(req);
            } catch (const server_node_error & e) {
                return node_error_res(e.status, e.what());
            } catch (const common_json_error & e) {
                return node_error_res(400, e.what());
            } catch (const std::invalid_argument & e) {
                return node_error_res(400, e.what());
            } catch (const std::exception & e) {
                return node_error_res(500, e.what());
            }
        };
    };
    http.post("/node/spawn",  guard(post_spawn));
    http.post("/node/stop",   guard(post_stop));
    http.post("/node/signal", guard(post_signal));
    http.post("/node/adopt",  guard(post_adopt));
    http.get ("/node/state",  guard(get_state));
    http.get ("/node/events", guard(get_events));
}
