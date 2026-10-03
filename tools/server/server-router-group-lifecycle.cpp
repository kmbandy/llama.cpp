#include "server-router-group-lifecycle.h"

#include "log.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <tuple>

#ifndef _WIN32
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool contains(const std::string & s, const char * needle) {
    return s.find(needle) != std::string::npos;
}

//
// worker output
//

const char * router_snapshot_result_str(router_snapshot_result r) {
    switch (r) {
        case ROUTER_SNAPSHOT_WRITTEN: return "written";
        case ROUTER_SNAPSHOT_FAILED:  return "FAILED";
        case ROUTER_SNAPSHOT_TIMEOUT: return "timeout";
        case ROUTER_SNAPSHOT_NONE:
        default:                      return "none";
    }
}

static std::string strip_eol(std::string s) {
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) {
        s.pop_back();
    }
    return s;
}

router_worker_line router_classify_worker_line(const std::string & raw) {
    static const char * k_written = "wp expert worker: stop snapshot written: ";
    static const char * k_failed  = "wp expert worker: stop snapshot FAILED: ";
    static const char * k_timeout = "wp expert worker: stop snapshot: quiesce timeout after ";

    const std::string line = strip_eol(raw);
    router_worker_line out;
    size_t pos;
    if ((pos = line.find(k_written)) != std::string::npos) {
        out.kind   = ROUTER_WORKER_LINE_SNAPSHOT_WRITTEN;
        out.detail = line.substr(pos + strlen(k_written));
    } else if ((pos = line.find(k_failed)) != std::string::npos) {
        out.kind   = ROUTER_WORKER_LINE_SNAPSHOT_FAILED;
        out.detail = line.substr(pos + strlen(k_failed));
    } else if ((pos = line.find(k_timeout)) != std::string::npos) {
        out.kind = ROUTER_WORKER_LINE_SNAPSHOT_TIMEOUT;
        std::string rest = line.substr(pos + strlen(k_timeout));
        const size_t comma = rest.find(',');
        out.detail = comma == std::string::npos ? rest : rest.substr(0, comma);
    } else if (contains(line, "Hip error")) {
        out.kind   = ROUTER_WORKER_LINE_HIP_ERROR;
        out.detail = line;
    }
    return out;
}

//
// launch command
//

static bool is_name_start(char c) {
    return std::isalpha((unsigned char) c) || c == '_';
}

static bool is_name_char(char c) {
    return std::isalnum((unsigned char) c) || c == '_';
}

// true if the raw text at `start` is an unquoted NAME= prefix
static bool raw_is_assignment(const std::string & s, size_t start) {
    if (start >= s.size() || !is_name_start(s[start])) {
        return false;
    }
    size_t i = start + 1;
    while (i < s.size() && is_name_char(s[i])) {
        i++;
    }
    return i < s.size() && s[i] == '=';
}

std::string router_parse_launch(const std::string & launch, router_launch & out) {
    out = router_launch{};

    struct word_t {
        std::string text;
        size_t      start = 0;
        bool        needs_shell = false;
    };
    std::vector<word_t> words;

    const size_t n = launch.size();
    size_t i = 0;
    while (i < n) {
        while (i < n && (launch[i] == ' ' || launch[i] == '\t')) {
            i++;
        }
        if (i >= n) {
            break;
        }
        word_t w;
        w.start = i;
        if (launch[i] == '#') {
            return "comments are not supported in launch";
        }
        if (launch[i] == '~') {
            w.needs_shell = true;
        }
        while (i < n && launch[i] != ' ' && launch[i] != '\t') {
            const char c = launch[i];
            if (c == '\'') {
                const size_t e = launch.find('\'', i + 1);
                if (e == std::string::npos) {
                    return "unterminated single quote in launch";
                }
                w.text += launch.substr(i + 1, e - i - 1);
                i = e + 1;
                continue;
            }
            if (c == '"') {
                i++;
                bool closed = false;
                while (i < n) {
                    const char d = launch[i];
                    if (d == '"') {
                        closed = true;
                        i++;
                        break;
                    }
                    if (d == '\\' && i + 1 < n && strchr("$`\"\\\n", launch[i + 1]) != nullptr) {
                        w.text += launch[i + 1];
                        i += 2;
                        continue;
                    }
                    if (d == '$' || d == '`') {
                        w.needs_shell = true;
                    }
                    w.text += d;
                    i++;
                }
                if (!closed) {
                    return "unterminated double quote in launch";
                }
                continue;
            }
            if (c == '\\') {
                if (i + 1 >= n) {
                    return "trailing backslash in launch";
                }
                w.text += launch[i + 1];
                i += 2;
                continue;
            }
            if (strchr("|&;<>()\n\r", c) != nullptr) {
                return std::string("'") + c + "' is not allowed in launch: it must be a single command "
                       "(no pipes, lists, redirections or subshells) because the router tracks the "
                       "worker's PID and reads its output";
            }
            if (strchr("$`*?[", c) != nullptr) {
                w.needs_shell = true;
            }
            w.text += c;
            i++;
        }
        words.push_back(std::move(w));
    }

    size_t first = 0;
    while (first < words.size() && raw_is_assignment(launch, words[first].start)) {
        if (words[first].needs_shell) {
            return "assignment '" + words[first].text + "' in launch needs shell expansion; set it with the preset 'env' key";
        }
        out.env.push_back(words[first].text);
        first++;
    }
    if (first >= words.size()) {
        return "launch has no command";
    }

    for (size_t k = first; k < words.size(); k++) {
        out.words.push_back(words[k].text);
        out.via_shell = out.via_shell || words[k].needs_shell;
    }
    if (out.via_shell) {
        // exec: the shell becomes the worker, so the PID the router tracks and signals is the worker's
        out.argv = { "/bin/sh", "-c", "exec " + launch.substr(words[first].start) };
    } else {
        // no PATH search: the router's PATH is not the worker's, and a bare name would run
        // whatever comes first there
        if (out.words[0].empty() || out.words[0][0] != '/') {
            return "command '" + out.words[0] + "' must be an absolute path (the router does not search PATH)";
        }
        out.argv = out.words;
    }
    return "";
}

static int parse_port(const std::string & s) {
    if (s.empty() || !std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; })) {
        return -1;
    }
    try {
        const int p = std::stoi(s);
        return p > 0 && p <= 65535 ? p : -1;
    } catch (...) {
        return -1;
    }
}

int router_launch_endpoint(const std::vector<std::string> & words, std::string & host) {
    host = "127.0.0.1";
    for (size_t i = 0; i < words.size(); i++) {
        const std::string & w = words[i];
        std::string listen;
        std::string port;
        if (w == "--listen" && i + 1 < words.size()) {
            listen = words[i + 1];
        } else if (w.rfind("--listen=", 0) == 0) {
            listen = w.substr(strlen("--listen="));
        } else if (w == "--port" && i + 1 < words.size()) {
            port = words[i + 1];
        } else if (w.rfind("--port=", 0) == 0) {
            port = w.substr(strlen("--port="));
        } else {
            continue;
        }
        if (!listen.empty()) {
            const size_t c = listen.rfind(':');
            if (c == std::string::npos) {
                return -1;
            }
            std::string h = listen.substr(0, c);
            if (h.size() >= 2 && h.front() == '[' && h.back() == ']') {
                h = h.substr(1, h.size() - 2);
            }
            if (!(h.empty() || h == "0.0.0.0" || h == "::" || h == "*")) {
                host = h;
            }
            return parse_port(listen.substr(c + 1));
        }
        return parse_port(port);
    }
    return -1;
}

//
// environment helpers
//

void router_env_unset(std::vector<std::string> & env, const std::string & key) {
    const std::string prefix = key + "=";
    env.erase(std::remove_if(env.begin(), env.end(), [&](const std::string & e) {
        return e.compare(0, prefix.size(), prefix) == 0;
    }), env.end());
}

void router_env_set(std::vector<std::string> & env, const std::string & key, const std::string & value) {
    router_env_unset(env, key);
    env.push_back(key + "=" + value);
}

bool router_env_get(const std::vector<std::string> & env, const std::string & key, std::string & value) {
    const std::string prefix = key + "=";
    bool found = false;
    for (const auto & e : env) {
        if (e.compare(0, prefix.size(), prefix) == 0) {
            value = e.substr(prefix.size());
            found = true;
        }
    }
    return found;
}

std::string router_env_override_error(const std::string & entry) {
    if (entry.empty()) {
        return "empty env entry";
    }
    if (entry[0] == '-') {
        const std::string key = entry.substr(1);
        if (key.empty() || key.find('=') != std::string::npos) {
            return "malformed env removal '" + entry + "' (expected -KEY)";
        }
        return "";
    }
    const size_t eq = entry.find('=');
    if (eq == std::string::npos || eq == 0) {
        return "malformed env entry '" + entry + "' (expected KEY=VALUE or -KEY)";
    }
    return "";
}

void router_env_apply_overrides(std::vector<std::string> & env, const std::vector<std::string> & overrides) {
    for (const auto & entry : overrides) {
        if (!router_env_override_error(entry).empty()) {
            continue;
        }
        if (entry[0] == '-') {
            router_env_unset(env, entry.substr(1));
        } else {
            const size_t eq = entry.find('=');
            router_env_set(env, entry.substr(0, eq), entry.substr(eq + 1));
        }
    }
}

const std::vector<std::string> & router_reserved_option_keys() {
    static const std::vector<std::string> keys = {
        "LLAMA_ARG_SSL_KEY_FILE",
        "LLAMA_ARG_SSL_CERT_FILE",
        "LLAMA_API_KEY",
        "LLAMA_ARG_API_KEY_FILE",
        "LLAMA_ARG_MODELS_DIR",
        "LLAMA_ARG_MODELS_MAX",
        "LLAMA_ARG_MODELS_PRESET",
        "LLAMA_ARG_MODELS_AUTOLOAD",
        "LLAMA_ARG_MODELS_IDLE_TIMEOUT",
        "LLAMA_ARG_MODELS_QUEUE_MAX_WAIT_S",
        "LLAMA_ARG_GPUS",
        "LLAMA_ARG_BOARD_URL",
        "LLAMA_ARG_BOARD_TOKEN_FILE",
        "LLAMA_ARG_NODE_TOKEN_FILE",
        "LLAMA_ARG_NODE_BIND",
    };
    return keys;
}

bool router_is_reserved_option_key(const std::string & key) {
    static const std::string router_prefix = "LLAMA_ARG_ROUTER_"; // ROUTER_ARG_*, LLAMA_ARG_ROUTER_NODE
    if (key.compare(0, router_prefix.size(), router_prefix) == 0) {
        return true;
    }
    const auto & keys = router_reserved_option_keys();
    return std::find(keys.begin(), keys.end(), key) != keys.end();
}

std::vector<std::string> router_worker_env(std::vector<std::string> env, const std::vector<std::string> & launch_env,
                                           const std::string & park_file, const std::string & gen) {
    for (const auto & a : launch_env) {
        const size_t eq = a.find('=');
        if (eq != std::string::npos && eq > 0) {
            router_env_set(env, a.substr(0, eq), a.substr(eq + 1));
        }
    }
    if (!park_file.empty()) {
        router_env_set(env, WP_ENV_PARK_FILE, park_file);
        router_env_set(env, WP_ENV_SEED_FROM_PARK, "1");
    }
    router_env_set(env, ROUTER_ENV_GEN, gen);
    return env;
}

int64_t router_env_quiesce_ms(const std::vector<std::string> & env) {
    std::string v;
    if (router_env_get(env, WP_ENV_QUIESCE_MS, v) && !v.empty()) {
        try {
            const long long ms = std::stoll(v);
            if (ms >= 0) {
                return ms;
            }
        } catch (...) {
        }
    }
    return ROUTER_WORKER_QUIESCE_MS_DEFAULT;
}

//
// sockets
//

bool router_tcp_accepts(const std::string & host, int port, int timeout_ms) {
#ifdef _WIN32
    (void) host;
    (void) port;
    (void) timeout_ms;
    return false;
#else
    if (port <= 0) {
        return false;
    }
    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &res) != 0 || res == nullptr) {
        return false;
    }
    bool ok = false;
    for (addrinfo * ai = res; ai != nullptr && !ok; ai = ai->ai_next) {
        const int fd = socket(ai->ai_family, ai->ai_socktype | SOCK_CLOEXEC, ai->ai_protocol);
        if (fd < 0) {
            continue;
        }
        fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
        int r = connect(fd, ai->ai_addr, ai->ai_addrlen);
        if (r == 0) {
            ok = true;
        } else if (errno == EINPROGRESS) {
            pollfd p{};
            p.fd     = fd;
            p.events = POLLOUT;
            if (poll(&p, 1, timeout_ms) == 1) {
                int       so_err = 0;
                socklen_t len    = sizeof(so_err);
                ok = getsockopt(fd, SOL_SOCKET, SO_ERROR, &so_err, &len) == 0 && so_err == 0;
            }
        }
        close(fd);
    }
    freeaddrinfo(res);
    return ok;
#endif
}

//
// generation + sweep
//

std::string router_generation_new() {
    std::random_device rd;
    std::mt19937_64    rng(((uint64_t) rd() << 32) ^ (uint64_t) rd() ^ (uint64_t) now_ms());
    uint8_t b[16];
    for (int i = 0; i < 16; i += 8) {
        const uint64_t v = rng();
        memcpy(b + i, &v, 8);
    }
    b[6] = (uint8_t) ((b[6] & 0x0f) | 0x40); // version 4
    b[8] = (uint8_t) ((b[8] & 0x3f) | 0x80); // variant 1
    char out[37];
    snprintf(out, sizeof(out),
             "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
             b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return out;
}

static bool read_whole_file(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return true;
}

static bool is_all_digits(const std::string & s) {
    return !s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; });
}

// first number after "<key>:" in a /proc/<pid>/status text; -1 if missing
static long long status_field(const std::string & status, const char * key) {
    const std::string k = std::string(key) + ":";
    size_t start = 0;
    while (start < status.size()) {
        size_t end = status.find('\n', start);
        if (end == std::string::npos) {
            end = status.size();
        }
        if (status.compare(start, k.size(), k) == 0) {
            const char * p = status.c_str() + start + k.size();
            char * stop = nullptr;
            const long long v = std::strtoll(p, &stop, 10); // skips leading blanks
            return stop == p ? -1 : v;
        }
        start = end + 1;
    }
    return -1;
}

// LLAMA_ROUTER_GEN / LLAMA_ROUTER_PID out of a NUL-separated environ blob
static void parse_router_environ(const std::string & environ_text, std::string & gen, long long & router_pid) {
    const std::string gen_prefix = std::string(ROUTER_ENV_GEN) + "=";
    const std::string pid_prefix = std::string(ROUTER_ENV_ROUTER_PID) + "=";
    size_t start = 0;
    while (start < environ_text.size()) {
        size_t end = environ_text.find('\0', start);
        if (end == std::string::npos) {
            end = environ_text.size();
        }
        const std::string kv = environ_text.substr(start, end - start);
        if (kv.compare(0, gen_prefix.size(), gen_prefix) == 0) {
            gen = kv.substr(gen_prefix.size());
        } else if (kv.compare(0, pid_prefix.size(), pid_prefix) == 0) {
            router_pid = std::atoll(kv.c_str() + pid_prefix.size());
        }
        start = end + 1;
    }
}

// router_find_stale_children(), with the generation each one carried
static std::vector<std::pair<int, std::string>> find_stale(const std::string & root, const std::string & gen, unsigned uid, int self_pid) {
    struct entry_t {
        long long   ppid = -1;
        long long   uid  = -1;
        bool        has_env = false;
        std::string gen;
        long long   router_pid = 0;
    };
    std::map<int, entry_t> procs;

    std::error_code ec;
    const fs::path proc_dir = fs::path(root.empty() ? "/" : root) / "proc";
    fs::directory_iterator it(proc_dir, ec);
    if (ec) {
        return {};
    }
    for (const auto & de : it) {
        const std::string name = de.path().filename().string();
        if (!is_all_digits(name)) {
            continue;
        }
        const int pid = std::atoi(name.c_str());
        entry_t e;
        std::string status;
        if (!read_whole_file((de.path() / "status").string(), status)) {
            continue; // gone, or not ours to read
        }
        e.ppid = status_field(status, "PPid");
        e.uid  = status_field(status, "Uid");
        std::string environ_text;
        if (read_whole_file((de.path() / "environ").string(), environ_text)) {
            e.has_env = true;
            parse_router_environ(environ_text, e.gen, e.router_pid);
        }
        procs[pid] = std::move(e);
    }

    std::vector<std::pair<int, std::string>> stale;
    for (const auto & [pid, e] : procs) {
        if (pid == self_pid || !e.has_env || e.gen.empty() || e.gen == gen || e.uid != (long long) uid) {
            continue;
        }
        // the router that started it is still its ancestor: a live router (a second instance on
        // this machine, e.g. a test run) owns it, leave it alone
        bool owned_by_live_router = false;
        if (e.router_pid > 0) {
            long long cur = e.ppid;
            for (int depth = 0; depth < 256 && cur > 0; depth++) {
                if (cur == e.router_pid) {
                    owned_by_live_router = true;
                    break;
                }
                auto p = procs.find((int) cur);
                if (p == procs.end()) {
                    break;
                }
                cur = p->second.ppid;
            }
        }
        if (!owned_by_live_router) {
            stale.emplace_back(pid, e.gen);
        }
    }
    return stale;
}

std::vector<int> router_find_stale_children(const std::string & root, const std::string & gen, unsigned uid, int self_pid) {
    std::vector<int> out;
    for (const auto & [pid, _] : find_stale(root, gen, uid, self_pid)) {
        out.push_back(pid);
    }
    return out;
}

#ifndef _WIN32
// alive per the real /proc: present and not a zombie
static bool real_pid_alive(int pid) {
    std::string stat;
    if (!read_whole_file("/proc/" + std::to_string(pid) + "/stat", stat)) {
        return false;
    }
    const size_t rp = stat.rfind(')');
    if (rp == std::string::npos || rp + 2 >= stat.size()) {
        return false;
    }
    const char state = stat[rp + 2];
    return state != 'Z' && state != 'X';
}
#endif

std::vector<int> router_sweep_stale_children(const std::string & root, const std::string & gen, unsigned uid, int self_pid, int64_t grace_ms) {
    const std::vector<std::pair<int, std::string>> stale = find_stale(root, gen, uid, self_pid);
    std::vector<int> victims;
    for (const auto & [pid, _] : stale) {
        victims.push_back(pid);
    }
#ifndef _WIN32
    if (victims.empty()) {
        return victims;
    }
    for (int pid : victims) {
        LOG_WRN("router: stopping pid %d left behind by a previous router generation (SIGTERM)\n", pid);
        kill(pid, SIGTERM);
    }
    LOG_WRN("router: sweeping %zu stale router children, waiting up to %" PRId64 " ms for them to exit\n",
            victims.size(), grace_ms);
    const int64_t deadline = now_ms() + grace_ms;
    while (now_ms() < deadline) {
        if (std::none_of(victims.begin(), victims.end(), real_pid_alive)) {
            return victims;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    for (const auto & [pid, stale_gen] : stale) {
        if (!real_pid_alive(pid)) {
            continue;
        }
        // the PID may have been reused during the wait: KILL only the same stale child
        std::string environ_text;
        std::string now_gen;
        long long   now_router_pid = 0;
        if (read_whole_file("/proc/" + std::to_string(pid) + "/environ", environ_text)) {
            parse_router_environ(environ_text, now_gen, now_router_pid);
        }
        if (now_gen != stale_gen) {
            LOG_WRN("router: pid %d no longer carries the stale generation, leaving it alone\n", pid);
            continue;
        }
        LOG_WRN("router: pid %d ignored SIGTERM for %" PRId64 " ms, sending SIGKILL\n", pid, grace_ms);
        kill(pid, SIGKILL);
    }
#endif
    return victims;
}

//
// router_worker_group
//

router_worker_group::router_worker_group(std::string group, std::vector<router_worker_spec> specs, callbacks cb, int64_t kill_grace_ms)
        : group(std::move(group)), cb(std::move(cb)), kill_grace_ms(kill_grace_ms) {
    members.resize(specs.size());
    for (size_t i = 0; i < specs.size(); i++) {
        members[i].spec = std::move(specs[i]);
    }
    th = std::thread([this]() { run(); });
}

router_worker_group::~router_worker_group() {
    {
        std::lock_guard<std::mutex> lk(mu);
        closing = true;
        for (auto & m : members) {
            if (m.spawned && !m.exited) {
                m.kill_pending = true;
            }
        }
    }
    waiter.wake();
    {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [this]() { return all_spawned_exited_locked(); });
        quit = true;
    }
    waiter.wake();
    th.join();
}

bool router_worker_group::all_spawned_exited_locked() const {
    for (const auto & m : members) {
        if (m.spawned && !m.exited) {
            return false;
        }
    }
    return true;
}

void router_worker_group::handle_line_locked(member & m, const std::string & raw, std::vector<std::pair<std::string, std::string>> & lines_out) {
    const std::string line = strip_eol(raw);
    LOG("[%s] %s\n", m.spec.name.c_str(), line.c_str());
    const router_worker_line c = router_classify_worker_line(line);
    switch (c.kind) {
        case ROUTER_WORKER_LINE_HIP_ERROR:
            if (!m.hip_error) {
                m.hip_error = true;
                m.error     = c.detail;
            }
            break;
        case ROUTER_WORKER_LINE_SNAPSHOT_WRITTEN:
            m.snapshot        = ROUTER_SNAPSHOT_WRITTEN;
            m.snapshot_detail = c.detail;
            break;
        case ROUTER_WORKER_LINE_SNAPSHOT_FAILED:
            m.snapshot        = ROUTER_SNAPSHOT_FAILED;
            m.snapshot_detail = c.detail;
            break;
        case ROUTER_WORKER_LINE_SNAPSHOT_TIMEOUT:
            m.quiesce_timeout = true;
            if (m.snapshot == ROUTER_SNAPSHOT_NONE) {
                m.snapshot        = ROUTER_SNAPSHOT_TIMEOUT;
                m.snapshot_detail = c.detail;
            }
            break;
        case ROUTER_WORKER_LINE_OTHER:
        default:
            break;
    }
    if (cb.on_line) {
        lines_out.emplace_back(m.spec.name, line);
    }
}

void router_worker_group::read_output_locked(member & m, std::vector<std::pair<std::string, std::string>> & lines_out) {
    static constexpr size_t max_line = 1024 * 1024;
    char chunk[4096];
    while (!m.eof) {
        const int n = m.proc->read_output(chunk, sizeof(chunk));
        if (n < 0) {
            m.eof = true;
            break;
        }
        if (n == 0) {
            break;
        }
        m.buf.append(chunk, (size_t) n);
        size_t start = 0;
        while (true) {
            const size_t nl = m.buf.find('\n', start);
            if (nl == std::string::npos) {
                break;
            }
            handle_line_locked(m, m.buf.substr(start, nl - start), lines_out);
            start = nl + 1;
        }
        m.buf.erase(0, start);
        if (m.buf.size() > max_line) {
            m.buf.clear();
        }
    }
    if (m.eof && !m.buf.empty()) {
        handle_line_locked(m, m.buf, lines_out);
        m.buf.clear();
    }
}

void router_worker_group::run() {
    std::vector<std::pair<std::string, std::string>> lines;
    while (true) {
        std::vector<server_subproc *> procs;
        std::vector<size_t>           owners;
        int64_t                       timeout = -1;
        {
            std::lock_guard<std::mutex> lk(mu);
            if (quit) {
                return;
            }
            const int64_t now = now_ms();
            for (size_t i = 0; i < members.size(); i++) {
                const auto & m = members[i];
                if (!m.spawned || m.exited) {
                    continue;
                }
                // exits are polled, not inferred from EOF: a grandchild may hold the pipe open
                timeout = timeout < 0 ? 100 : std::min<int64_t>(timeout, 100);
                if (m.kill_deadline) {
                    timeout = std::min<int64_t>(timeout, std::max<int64_t>(0, m.kill_deadline - now));
                }
                if (m.term_pending || m.kill_pending) {
                    timeout = 0;
                }
                if (!m.eof) {
                    procs.push_back(m.proc.get());
                    owners.push_back(i);
                }
            }
        }

        std::vector<bool> ready;
        waiter.wait(procs, ready, timeout);

        std::vector<std::tuple<std::string, int, std::string>> unexpected;
        bool fire_stopped = false;
        lines.clear();
        {
            std::lock_guard<std::mutex> lk(mu);
            for (size_t k = 0; k < owners.size(); k++) {
                if (k < ready.size() && ready[k]) {
                    read_output_locked(members[owners[k]], lines);
                }
            }
            const int64_t now = now_ms();
            for (auto & m : members) {
                if (!m.spawned || m.exited) {
                    continue;
                }
                if (m.term_pending) {
                    m.term_pending = false;
                    m.stopping     = true;
                    const int pid  = m.proc->sproc.pid(); // 0 once reaped; only this thread reaps
                    if (pid > 0) {
#ifndef _WIN32
                        LOG_INF("group %s: SIGTERM worker %s (pid %d), SIGKILL in %" PRId64 " ms\n",
                                group.c_str(), m.spec.name.c_str(), pid, m.spec.quiesce_ms + kill_grace_ms);
                        kill(pid, SIGTERM);
                        m.kill_deadline = now + m.spec.quiesce_ms + kill_grace_ms;
#else
                        m.kill_pending = true;
#endif
                    }
                }
                if (m.kill_pending || (m.kill_deadline && now >= m.kill_deadline)) {
                    m.stopping     = true;
                    m.kill_deadline = 0;
                    m.kill_pending = false;
                    if (m.proc->sproc.pid() > 0) {
                        LOG_WRN("group %s: SIGKILL worker %s (pid %d)\n", group.c_str(), m.spec.name.c_str(), m.pid);
                        m.proc->terminate();
                        m.killed = true;
                    }
                }
                if (!m.proc->is_alive()) {
                    read_output_locked(m, lines); // whatever it wrote before going
                    if (!m.buf.empty()) {
                        handle_line_locked(m, m.buf, lines);
                        m.buf.clear();
                    }
                    m.exit_code     = m.proc->join();
                    m.exited        = true;
                    m.eof           = true; // the pipe is closed by join(), never read it again
                    m.kill_deadline = 0;
                    LOG_INF("group %s: worker %s (pid %d) exited with status %d%s\n", group.c_str(),
                            m.spec.name.c_str(), m.pid, m.exit_code, m.killed ? " (killed)" : "");
                    if (armed && !stop_requested && !m.stopping && !closing) {
                        std::string reason = string_format("worker '%s' exited with status %d", m.spec.name.c_str(), m.exit_code);
                        if (!m.error.empty()) {
                            reason += ": " + m.error;
                        }
                        unexpected.emplace_back(m.spec.name, m.exit_code, reason);
                    }
                }
            }
            if (stop_requested && !stop_fired && all_spawned_exited_locked()) {
                stop_fired   = true;
                fire_stopped = !closing;
            }
        }
        cv.notify_all();

        if (cb.on_line) {
            for (const auto & [name, line] : lines) {
                cb.on_line(name, line);
            }
        }
        for (const auto & [name, code, reason] : unexpected) {
            SRV_WRN("group %s: %s\n", group.c_str(), reason.c_str());
            if (cb.on_unexpected_exit) {
                cb.on_unexpected_exit(name, code, reason);
            }
        }
        if (fire_stopped && cb.on_stopped) {
            cb.on_stopped();
        }
    }
}

void router_worker_group::teardown_failed_start(std::unique_lock<std::mutex> & lk) {
    for (auto & m : members) {
        if (m.spawned && !m.exited) {
            // a worker that came up gets the normal TERM (it may write its stop snapshot);
            // one still initialising has nothing worth saving and must not overwrite a good
            // park file with a half-built one
            if (m.ready) {
                m.term_pending = true;
            } else {
                m.kill_pending = true;
            }
        }
    }
    waiter.wake();
    cv.wait(lk, [this]() { return all_spawned_exited_locked(); });
}

bool router_worker_group::start(std::string & err, const std::function<bool()> & cancelled) {
#ifdef _WIN32
    (void) cancelled;
    err = "model groups are not supported on Windows";
    return false;
#else
    // a port already in use would make the readiness check below lie (and our worker fail to bind)
    for (const auto & m : members) {
        if (router_tcp_accepts(m.spec.host, m.spec.port, 200)) {
            err = string_format("worker '%s': %s:%d already accepts connections before the worker was started "
                                "(a leftover worker, or another service on that port)",
                                m.spec.name.c_str(), m.spec.host.c_str(), m.spec.port);
            return false;
        }
    }

    const int64_t t0 = now_ms();
    for (auto & m : members) {
        auto proc = std::make_unique<server_subproc>();
        LOG_INF("group %s: starting worker %s (expects %s:%d)\n", group.c_str(), m.spec.name.c_str(),
                m.spec.host.c_str(), m.spec.port);
        for (const auto & a : m.spec.argv) {
            LOG_INF("  %s\n", a.c_str());
        }
        // argv[0] is absolute (router_parse_launch), so no PATH search
        const int options = subprocess_option_no_window | subprocess_option_combined_stdout_stderr;
        if (!proc->sproc.create(m.spec.argv, options, m.spec.env)) {
            err = string_format("worker '%s': failed to spawn '%s'", m.spec.name.c_str(),
                                m.spec.argv.empty() ? "" : m.spec.argv[0].c_str());
            std::unique_lock<std::mutex> lk(mu);
            teardown_failed_start(lk);
            return false;
        }
        proc->has_output(); // sets the pipe non-blocking before the group thread reads it
        const int pid = proc->sproc.pid();
        {
            std::lock_guard<std::mutex> lk(mu);
            m.proc    = std::move(proc);
            m.pid     = pid;
            m.spawned = true;
        }
        waiter.wake();
    }

    std::unique_lock<std::mutex> lk(mu);
    while (true) {
        // failures the group thread saw
        for (auto & m : members) {
            if (m.exited) {
                err = string_format("worker '%s' exited with status %d before accepting connections",
                                    m.spec.name.c_str(), m.exit_code);
                if (!m.error.empty()) {
                    err += ": " + m.error;
                }
                m.error = err;
                teardown_failed_start(lk);
                return false;
            }
            if (m.hip_error) {
                err = string_format("worker '%s' failed to start: %s", m.spec.name.c_str(), m.error.c_str());
                teardown_failed_start(lk);
                return false;
            }
        }
        if (std::all_of(members.begin(), members.end(), [](const member & m) { return m.ready; })) {
            armed = true;
            LOG_INF("group %s: all %zu workers accept connections (%" PRId64 " ms)\n", group.c_str(),
                    members.size(), now_ms() - t0);
            return true;
        }

        // never call out with our lock held: cancelled() takes the router's lock
        lk.unlock();
        const bool cancel = cancelled && cancelled();
        // `ready` is only ever written by this thread, so reading it unlocked here is fine
        std::vector<bool> up(members.size(), false);
        for (size_t i = 0; i < members.size(); i++) {
            if (!cancel && !members[i].ready) {
                up[i] = router_tcp_accepts(members[i].spec.host, members[i].spec.port, 200);
            }
        }
        lk.lock();
        if (cancel) {
            err = "load cancelled while its workers were starting";
            teardown_failed_start(lk);
            return false;
        }
        for (size_t i = 0; i < members.size(); i++) {
            if (up[i] && !members[i].exited) {
                members[i].ready = true;
                LOG_INF("group %s: worker %s accepts connections on %s:%d\n", group.c_str(),
                        members[i].spec.name.c_str(), members[i].spec.host.c_str(), members[i].spec.port);
            }
        }
        const int64_t elapsed = now_ms() - t0;
        for (auto & m : members) {
            if (!m.ready && !m.exited && !m.hip_error && elapsed >= (int64_t) m.spec.startup_timeout_s * 1000) {
                err = string_format("worker '%s' did not accept connections on %s:%d within %d s",
                                    m.spec.name.c_str(), m.spec.host.c_str(), m.spec.port, m.spec.startup_timeout_s);
                m.error = err;
                teardown_failed_start(lk);
                return false;
            }
        }
        cv.wait_for(lk, std::chrono::milliseconds(200));
    }
#endif
}

void router_worker_group::request_stop() {
    {
        std::lock_guard<std::mutex> lk(mu);
        if (stop_requested) {
            return;
        }
        stop_requested = true;
        for (auto & m : members) {
            if (m.spawned && !m.exited && !m.stopping) {
                m.term_pending = true;
            }
        }
    }
    waiter.wake();
}

bool router_worker_group::wait_stopped(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lk(mu);
    auto done = [this]() { return all_spawned_exited_locked() && (!stop_requested || stop_fired); };
    if (timeout_ms < 0) {
        cv.wait(lk, done);
        return true;
    }
    return cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), done);
}

std::set<int> router_worker_group::pids() const {
    std::lock_guard<std::mutex> lk(mu);
    std::set<int> out;
    for (const auto & m : members) {
        if (m.spawned && !m.exited && m.pid > 0) {
            out.insert(m.pid);
        }
    }
    return out;
}

bool router_worker_group::all_exited() const {
    std::lock_guard<std::mutex> lk(mu);
    return all_spawned_exited_locked();
}

std::vector<router_worker_state> router_worker_group::status() const {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<router_worker_state> out;
    for (const auto & m : members) {
        router_worker_state s;
        s.name            = m.spec.name;
        s.pid             = m.pid;
        s.host            = m.spec.host;
        s.port            = m.spec.port;
        s.state           = !m.spawned ? "pending" : m.exited ? "exited" : m.stopping ? "stopping" : m.ready ? "ready" : "starting";
        s.exit_code       = m.exit_code;
        s.killed          = m.killed;
        s.snapshot        = m.snapshot;
        s.snapshot_detail = m.snapshot_detail;
        s.quiesce_timeout = m.quiesce_timeout;
        s.error           = m.error;
        out.push_back(std::move(s));
    }
    return out;
}
