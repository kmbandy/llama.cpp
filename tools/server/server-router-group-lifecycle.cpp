#include "server-router-group-lifecycle.h"

#include "log.h"
#include "server-router-node-client.h"

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

std::string router_launch_listen_host(const std::vector<std::string> & words) {
    for (size_t i = 0; i < words.size(); i++) {
        std::string listen;
        if (words[i] == "--listen" && i + 1 < words.size()) {
            listen = words[i + 1];
        } else if (words[i].rfind("--listen=", 0) == 0) {
            listen = words[i].substr(strlen("--listen="));
        } else {
            continue;
        }
        const size_t c = listen.rfind(':');
        std::string h = c == std::string::npos ? std::string() : listen.substr(0, c);
        if (h.size() >= 2 && h.front() == '[' && h.back() == ']') {
            h = h.substr(1, h.size() - 2);
        }
        if (h == "0.0.0.0" || h == "::" || h == "*") {
            h.clear();
        }
        return h;
    }
    return "";
}

static bool router_host_is_wildcard(std::string h) {
    if (h.size() >= 2 && h.front() == '[' && h.back() == ']') {
        h = h.substr(1, h.size() - 2);
    }
    return h.empty() || h == "0.0.0.0" || h == "::" || h == "*";
}

// the host a launch word list names: 1 = a host (set in `host`), 0 = none; a `--listen` without a colon
// names none either
static int router_launch_named_host(const std::vector<std::string> & words, std::string & host) {
    for (size_t i = 0; i < words.size(); i++) {
        const std::string & w = words[i];
        std::string value;
        if ((w == "--listen" || w == "--host" || w == "-H") && i + 1 < words.size()) {
            value = words[i + 1];
        } else if (w.rfind("--listen=", 0) == 0) {
            value = w.substr(strlen("--listen="));
        } else if (w.rfind("--host=", 0) == 0) {
            value = w.substr(strlen("--host="));
        } else {
            continue;
        }
        if (w == "--listen" || w.rfind("--listen=", 0) == 0) {
            const size_t c = value.rfind(':');
            value = c == std::string::npos ? std::string() : value.substr(0, c); // "host:port"
        }
        host = value;
        return 1;
    }
    return 0;
}

std::string router_remote_worker_host_fix(router_launch & launch, const std::string & node_host) {
    std::string host;
    if (router_launch_named_host(launch.words, host) != 0) {
        if (router_host_is_wildcard(host)) {
            return "its launch listens on a wildcard address ('" + host + "'): a worker on another machine has no auth, "
                   "name the node's LAN / Tailscale address (" + node_host + ") instead";
        }
        return "";
    }
    if (node_host.empty() || router_host_is_wildcard(node_host)) {
        return "the node's address is not known, cannot tell the worker which address to listen on";
    }
    // no host in the launch: bind to the node's address only
    if (launch.via_shell) {
        if (launch.argv.size() < 3) {
            return "launch has no command to add --host to";
        }
        std::string quoted = "'";
        for (char ch : node_host) {
            if (ch == '\'') {
                quoted += "'\\''";
            } else {
                quoted.push_back(ch);
            }
        }
        quoted += "'";
        launch.argv[2] += " --host " + quoted;
    } else {
        router_args_set_host(launch.argv, node_host);
    }
    launch.words.push_back("--host");
    launch.words.push_back(node_host);
    return "";
}

void router_args_set_port(std::vector<std::string> & args, int port) {
    const std::string p = std::to_string(port);
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--port" && i + 1 < args.size()) {
            args[i + 1] = p;
            return;
        }
        if (args[i].rfind("--port=", 0) == 0) {
            args[i] = "--port=" + p;
            return;
        }
    }
    args.push_back("--port");
    args.push_back(p);
}

void router_args_set_host(std::vector<std::string> & args, const std::string & host) {
    for (size_t i = 0; i < args.size(); i++) {
        if (args[i] == "--host" && i + 1 < args.size()) {
            args[i + 1] = host;
            return;
        }
        if (args[i].rfind("--host=", 0) == 0) {
            args[i] = "--host=" + host;
            return;
        }
    }
    args.push_back("--host");
    args.push_back(host);
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

router_worker_group::router_worker_group(std::string group, std::vector<router_worker_spec> specs, callbacks cb,
                                         int64_t kill_grace_ms, std::string gen_in)
        : group(std::move(group)), cb(std::move(cb)), kill_grace_ms(kill_grace_ms), gen(std::move(gen_in)) {
    st = std::make_shared<shared_state>();
    if (gen.empty() && !specs.empty()) {
        router_env_get(specs[0].env, "LLAMA_ROUTER_GEN", gen);
    }
    if (gen.empty()) {
        gen = "router";
    }
    members.resize(specs.size());
    for (size_t i = 0; i < specs.size(); i++) {
        members[i].node = specs[i].node;
        members[i].spec = std::move(specs[i]);
    }
    th = std::thread([this]() { run(); });
}

router_worker_group::~router_worker_group() {
    {
        std::unique_lock<std::mutex> lk(st->mu);
        closing = true;
        for (auto & m : members) {
            if (m.spawned && !m.exited) {
                m.kill_pending = true;
            }
        }
        st->cv.notify_all();
        wait_exits_bounded(lk, 15000);
        quit = true;
        st->cv.notify_all();
    }
    th.join();
    if (private_node) {
        private_node->shutdown();
    }
}

std::shared_ptr<router_node_link> router_worker_group::own_node() {
    if (!private_node) {
        server_node_config ncfg = server_node_default_config();
        ncfg.base_env.clear(); // the specs' env is the final environment
        router_node_link_config lc;
        lc.gen = gen;
        private_node = router_node_make_local(lc, std::move(ncfg));
        private_node->start();
    }
    return private_node;
}

bool router_worker_group::all_spawned_exited_locked() const {
    for (const auto & m : members) {
        if (m.spawned && !m.exited) {
            return false;
        }
    }
    return true;
}

void router_worker_group::wait_exits_bounded(std::unique_lock<std::mutex> & lk, int64_t bound_ms) {
    int64_t grace = 0;
    for (const auto & m : members) {
        grace = std::max(grace, m.spec.quiesce_ms + kill_grace_ms);
    }
    const bool done = st->cv.wait_for(lk, std::chrono::milliseconds(bound_ms + grace), [this]() { return all_spawned_exited_locked(); });
    if (done) {
        return;
    }
    // a node that does not answer: give up on what is left (its link reports the exit if it ever comes)
    std::vector<std::pair<std::shared_ptr<router_node_link>, std::string>> drop;
    for (auto & m : members) {
        if (m.spawned && !m.exited) {
            LOG_WRN("group %s: worker %s (pid %d) did not exit in time, giving up on it\n", group.c_str(), m.spec.name.c_str(), m.pid);
            m.exited = true;
            m.stopping = true;
            drop.emplace_back(m.node, m.spec.name);
        }
    }
    lk.unlock();
    for (const auto & [node, name] : drop) {
        if (node) {
            node->unwatch(name);
        }
    }
    lk.lock();
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

void router_worker_group::run() {
    struct command {
        size_t                            idx = 0;
        std::shared_ptr<router_node_link> node;
        std::string                       name;
        bool                              kill = false;
        int                               timeout_s = 0;
    };
    while (true) {
        std::vector<std::pair<std::string, std::string>>       lines;
        std::vector<std::tuple<std::string, int, std::string>> unexpected;
        std::vector<command>                                   cmds;
        bool                                                   fire_stopped = false;
        {
            std::unique_lock<std::mutex> lk(st->mu);
            // the node enforces TERM -> KILL as well (a node we cannot reach still does); this thread
            // adds the same deadline with millisecond precision
            auto cmd_due = [&]() {
                const int64_t now = now_ms();
                for (const auto & m : members) {
                    if (!m.spawned || m.exited || m.pid <= 0) {
                        continue;
                    }
                    if ((m.term_pending || m.kill_pending) && now >= m.retry_at) {
                        return true;
                    }
                    if (m.kill_deadline && now >= m.kill_deadline) {
                        return true;
                    }
                }
                return false;
            };
            int64_t wait = 200;
            {
                const int64_t now = now_ms();
                for (const auto & m : members) {
                    if (m.spawned && !m.exited && m.kill_deadline) {
                        wait = std::min<int64_t>(wait, std::max<int64_t>(1, m.kill_deadline - now));
                    }
                    if (m.spawned && !m.exited && (m.term_pending || m.kill_pending) && m.retry_at > now) {
                        wait = std::min<int64_t>(wait, m.retry_at - now);
                    }
                }
            }
            st->cv.wait_for(lk, std::chrono::milliseconds(wait), [&]() { return quit || !st->events.empty() || cmd_due(); });
            if (quit) {
                return;
            }

            while (!st->events.empty()) {
                const event ev = std::move(st->events.front());
                st->events.pop_front();
                member & m = members[ev.idx];
                if (!ev.exit) {
                    if (!m.exited) {
                        handle_line_locked(m, ev.line, lines);
                    }
                    continue;
                }
                if (m.exited) {
                    continue;
                }
                m.exit_code     = ev.exit_code;
                m.killed        = m.killed || ev.killed;
                m.exited        = true;
                m.kill_deadline = 0;
                m.term_pending  = false;
                m.kill_pending  = false;
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

            const int64_t now = now_ms();
            for (size_t i = 0; i < members.size(); i++) {
                member & m = members[i];
                if (!m.spawned || m.exited || m.pid <= 0 || !m.node) {
                    continue;
                }
                if (m.kill_deadline && now >= m.kill_deadline) {
                    m.kill_pending  = true;
                    m.kill_deadline = 0;
                }
                if (now < m.retry_at) {
                    continue;
                }
                if (m.kill_pending) {
                    m.kill_pending = false;
                    m.term_pending = false;
                    m.stopping     = true;
                    m.killed       = true;
                    LOG_WRN("group %s: SIGKILL worker %s (pid %d)\n", group.c_str(), m.spec.name.c_str(), m.pid);
                    command c;
                    c.idx = i; c.node = m.node; c.name = m.spec.name; c.kill = true;
                    cmds.push_back(std::move(c));
                } else if (m.term_pending) {
                    m.term_pending = false;
                    m.stopping     = true;
                    const int64_t bound_ms = m.spec.quiesce_ms + kill_grace_ms;
                    LOG_INF("group %s: SIGTERM worker %s (pid %d), SIGKILL in %" PRId64 " ms\n",
                            group.c_str(), m.spec.name.c_str(), m.pid, bound_ms);
                    m.kill_deadline = now + bound_ms;
                    command c;
                    c.idx = i; c.node = m.node; c.name = m.spec.name; c.kill = false;
                    // the node's own SIGKILL is the backstop, a second after ours
                    c.timeout_s = (int) ((bound_ms + 999) / 1000) + 1;
                    cmds.push_back(std::move(c));
                }
            }
            if (stop_requested && !stop_fired && all_spawned_exited_locked()) {
                stop_fired   = true;
                fire_stopped = !closing;
            }
        }
        st->cv.notify_all();

        // node calls with no lock held (a remote node's are HTTP)
        for (const auto & c : cmds) {
            bool retry = false;
            try {
                if (c.kill) {
                    c.node->signal(c.name, 9);
                } else {
                    c.node->stop(c.name, c.timeout_s, "term");
                }
            } catch (const server_node_error & e) {
                // 404 / 409: it is gone or already going; its exit event follows
                if (e.status != 404 && e.status != 409) {
                    LOG_WRN("group %s: %s of worker %s failed: %s (retrying)\n", group.c_str(),
                            c.kill ? "SIGKILL" : "SIGTERM", c.name.c_str(), e.what());
                    retry = true;
                }
            } catch (const std::exception & e) {
                LOG_WRN("group %s: %s of worker %s failed: %s (retrying)\n", group.c_str(),
                        c.kill ? "SIGKILL" : "SIGTERM", c.name.c_str(), e.what());
                retry = true;
            }
            if (retry) {
                std::lock_guard<std::mutex> lk(st->mu);
                member & m = members[c.idx];
                if (!m.exited) {
                    (c.kill ? m.kill_pending : m.term_pending) = true;
                    m.retry_at = now_ms() + 1000;
                }
            }
        }

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
    st->cv.notify_all();
    wait_exits_bounded(lk, 15000);
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
    for (size_t i = 0; i < members.size(); i++) {
        member & m = members[i];
        LOG_INF("group %s: starting worker %s (expects %s:%d)\n", group.c_str(), m.spec.name.c_str(),
                m.spec.host.c_str(), m.spec.port);
        for (const auto & a : m.spec.argv) {
            LOG_INF("  %s\n", a.c_str());
        }
        if (!m.node) {
            m.node = own_node();
        }
        node_spawn_request req;
        req.name = m.spec.name;
        req.gen  = gen;
        req.args = m.spec.argv; // argv[0] is absolute (router_parse_launch), so no PATH search
        req.env  = m.spec.env;
        req.port = m.spec.port;

        std::shared_ptr<shared_state> state = st;
        router_node_watch w;
        w.on_line = [state, i](const std::string & line) {
            {
                std::lock_guard<std::mutex> lk(state->mu);
                event ev;
                ev.idx  = i;
                ev.line = line;
                state->events.push_back(std::move(ev));
            }
            state->cv.notify_all();
        };
        w.on_exit = [state, i](const node_child_info & info) {
            {
                std::lock_guard<std::mutex> lk(state->mu);
                event ev;
                ev.exit      = true;
                ev.idx       = i;
                ev.exit_code = info.exit_code;
                ev.killed    = info.killed;
                state->events.push_back(std::move(ev));
            }
            state->cv.notify_all();
        };
        try {
            const node_child_info info = m.node->spawn(req, std::move(w));
            std::lock_guard<std::mutex> lk(st->mu);
            m.pid     = info.pid;
            m.spawned = true;
        } catch (const std::exception & e) {
            err = string_format("worker '%s': failed to spawn '%s': %s", m.spec.name.c_str(),
                                m.spec.argv.empty() ? "" : m.spec.argv[0].c_str(), e.what());
            std::unique_lock<std::mutex> lk(st->mu);
            teardown_failed_start(lk);
            return false;
        }
        st->cv.notify_all();
    }

    std::unique_lock<std::mutex> lk(st->mu);
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
        st->cv.wait_for(lk, std::chrono::milliseconds(200));
    }
#endif
}

void router_worker_group::request_stop() {
    {
        std::lock_guard<std::mutex> lk(st->mu);
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
    st->cv.notify_all();
}

bool router_worker_group::wait_stopped(int64_t timeout_ms) {
    std::unique_lock<std::mutex> lk(st->mu);
    auto done = [this]() { return all_spawned_exited_locked() && (!stop_requested || stop_fired); };
    if (timeout_ms < 0) {
        st->cv.wait(lk, done);
        return true;
    }
    return st->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), done);
}

std::set<int> router_worker_group::pids() const {
    std::lock_guard<std::mutex> lk(st->mu);
    std::set<int> out;
    for (const auto & m : members) {
        if (m.spawned && !m.exited && m.pid > 0) {
            out.insert(m.pid);
        }
    }
    return out;
}

bool router_worker_group::all_exited() const {
    std::lock_guard<std::mutex> lk(st->mu);
    return all_spawned_exited_locked();
}

std::vector<router_worker_state> router_worker_group::status() const {
    std::lock_guard<std::mutex> lk(st->mu);
    std::vector<router_worker_state> out;
    for (const auto & m : members) {
        router_worker_state s;
        s.name            = m.spec.name;
        s.machine         = m.node ? m.node->machine() : "";
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
