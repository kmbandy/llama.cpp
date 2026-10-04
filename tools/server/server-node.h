#pragma once

// The router node: the child-process table a machine's router keeps, and the --router-node
// HTTP API (spec §3.2) that lets the leader router drive it from another machine.
//
// server_node is the core and knows nothing about HTTP: spawn / stop / signal / adopt /
// state / events. The leader uses it in-process for its own machine; server_node_routes is
// the thin HTTP layer the --router-node daemon puts on top of it.
//
// Children:
//   - the caller gives the full argv (argv[0] absolute: PATH is never searched) and env
//     overrides; the node never consults presets of its own;
//   - final env = base env (default: this process's env, minus the node's own flags) +
//     the request's overrides ("KEY=VALUE" sets, "-KEY" unsets, as a preset `env`) + LLAMA_ROUTER_GEN=<gen>, LLAMA_ROUTER_PID=<this pid>,
//     LLAMA_ROUTER_CHILD=<name>; refused when TEMP/TMP/TMPDIR names a non-directory;
//   - every child has a stdin pipe (a llama-server child exits on its EOF, so it dies with
//     the node) and a combined stdout/stderr pipe, forwarded line by line as events;
//   - one thread reads every child's output, reaps them and enforces stop deadlines. It is the
//     only place a child is reaped, and signals are sent under the same lock only to children
//     not yet reaped, so a PID is never signalled after it was reused.
//   - the destructor stops every live child (exit command + SIGTERM, SIGKILL after
//     shutdown_grace_ms) and waits for them.
//
// Orphans: collect_orphans() (the daemon calls it at start) finds processes a previous node
// (or router) left behind: they carry LLAMA_ROUTER_GEN and the router named by their
// LLAMA_ROUTER_PID is not an ancestor. The leader may adopt one by (name, gen) within
// adopt_window_ms; the rest get SIGTERM, then SIGKILL after orphan_kill_grace_ms. An adopted
// process is not our child: no output, liveness polled from /proc, exit code unknown (-1).

#include "server-common.h" // server_subproc, json
#include "server-http.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

struct common_params;

static constexpr const char * ROUTER_ENV_CHILD = "LLAMA_ROUTER_CHILD"; // the child's name in the node's table

static constexpr int64_t NODE_HEARTBEAT_MS_DEFAULT        = 2000;
static constexpr int64_t NODE_ADOPT_WINDOW_MS_DEFAULT     = 60000;
static constexpr int64_t NODE_ORPHAN_KILL_GRACE_MS        = 40000; // worker quiesce (30 s) + 10 s
static constexpr int64_t NODE_SHUTDOWN_GRACE_MS_DEFAULT   = 40000;
static constexpr int     NODE_STOP_TIMEOUT_S_DEFAULT      = 10;

struct server_node_config {
    std::vector<std::string> base_env;          // under every child's env; server_node_default_env() by default
    std::string              proc_root;         // "" = the real /proc and /sys (probe, orphan discovery)
    unsigned                 uid      = 0;      // orphans must run as this uid (default: getuid())
    int                      self_pid = 0;      // this process (default: getpid())
    int64_t                  heartbeat_ms       = NODE_HEARTBEAT_MS_DEFAULT;
    int64_t                  adopt_window_ms    = NODE_ADOPT_WINDOW_MS_DEFAULT;
    int64_t                  orphan_kill_grace_ms = NODE_ORPHAN_KILL_GRACE_MS;
    int64_t                  shutdown_grace_ms  = NODE_SHUTDOWN_GRACE_MS_DEFAULT;
    int64_t                  abandon_grace_ms   = 10000; // after the shutdown SIGKILL: stop waiting, log, exit anyway
    size_t                   event_backlog      = 4096; // events kept for late / resuming subscribers
    bool                     log_lines          = true; // echo every child output line to the log (the --router-node daemon); an in-process node's owner logs them itself
    std::string              child_host;        // non-empty: a child spawned with alloc_port is told to `--host` this address (the daemon's first --node-bind), never a wildcard
    std::vector<std::string> exec_allow;        // --node-exec-allow DIR (repeatable): when non-empty, a spawn's argv[0] must resolve (realpath) under one of them, else 403
    std::string              exe;               // this binary (state / heartbeat `exe`): the leader runs llama-server children as it
};

// This process's environment without the router's reserved names (router_is_reserved_option_key:
// TLS, API keys, model registry, GPU slots, board / node flags, LLAMA_ARG_ROUTER_*) and without
// its own LLAMA_ARG_HOST / PORT / MODEL / MMPROJ / ALIAS / HF_REPO: a child must not come up as
// a node, bind the node's port, or inherit its keys.
std::vector<std::string> server_node_default_env();

// A config with base_env / uid / self_pid filled for this process.
server_node_config server_node_default_config();

struct node_spawn_request {
    std::string              name;
    std::string              gen;  // the leader's router generation, stamped as LLAMA_ROUTER_GEN
    std::vector<std::string> args; // full argv, args[0] absolute
    std::vector<std::string> env;  // preset-style overrides on the base env: "KEY=VALUE" sets, "-KEY" unsets
    int                      port = 0; // reported port; 0 = from --port / --listen in args (never from env)
    bool                     alloc_port = false; // the node picks a free port, sets it as `--port` in args and reports it
};

// Point-in-time view of one table entry.
struct node_child_info {
    std::string name;
    std::string gen;
    int         pid       = 0;
    int         port      = 0;         // the spawn's `port`, else --port / --listen in args; 0 if none
    std::string status;                // "running" | "stopping" | "exited"
    int         exit_code = -1;        // once exited; -1 for an adopted process
    bool        killed    = false;     // the node had to SIGKILL it
    bool        adopted   = false;     // re-adopted orphan: not our child, no output
    int64_t     started_ms = 0;        // unix ms
};

struct node_orphan_info {
    int         pid = 0;
    std::string name; // "" when it carries no LLAMA_ROUTER_CHILD (cannot be adopted)
    std::string gen;
    std::string state; // "waiting" (adoptable) | "terminating"
};

// Thrown by server_node methods; `status` is the HTTP status the routes answer with.
struct server_node_error : std::runtime_error {
    int status;
    server_node_error(int status, const std::string & msg) : std::runtime_error(msg), status(status) {}
};

class server_node {
  public:
    explicit server_node(server_node_config cfg = server_node_default_config());
    ~server_node();

    server_node(const server_node &) = delete;
    server_node & operator=(const server_node &) = delete;

    // Starts a child. Throws server_node_error: 400 bad request (empty name/gen/args,
    // relative argv[0], malformed env entry, temp-dir violation), 403 argv[0] is outside the
    // allowed directories (cfg.exec_allow, when set), 409 a live child has that
    // name, 500 spawn failure (or no free port for alloc_port). A name whose previous child
    // exited is reused.
    node_child_info spawn(const node_spawn_request & req);

    // Non-blocking. method "both" (default): the router exit command on stdin + SIGTERM;
    // "stdin": exit command only (llama-server children); "term": SIGTERM only (workers).
    // SIGKILL after timeout_s (an earlier deadline from a previous stop wins). An exited child
    // is returned as is. Throws 404 unknown name, 400 bad method.
    node_child_info stop(const std::string & name, int timeout_s = NODE_STOP_TIMEOUT_S_DEFAULT,
                         const std::string & method = "both");

    // Blocks until the child has exited or timeout_ms passes; true when it exited.
    bool wait_exit(const std::string & name, int64_t timeout_ms);

    // Sends a signal to a live child. Throws 404 unknown name, 409 not running.
    node_child_info signal(const std::string & name, int sig);

    // Takes an orphan into the table. Throws 404 no waiting orphan with that (name, gen) /
    // it is gone, 409 a live child has that name.
    node_child_info adopt(const std::string & name, const std::string & gen);

    // Finds what previous generations left behind (see top of file). Returns how many.
    size_t collect_orphans();

    std::vector<node_child_info>  children() const;
    std::vector<node_orphan_info> orphans() const;

    // Children (with RssAnon/RssShmem and VRAM per device), orphans, every PID's VRAM per
    // device (fdinfo), MemAvailable and per-device sysfs VRAM used. Probes outside the lock.
    json state() const;

    // Events: {"seq", "type", ...}. Types: "child" (state change), "line" (one output line),
    // "orphan" (sweep action). Heartbeats are made per subscriber by the caller
    // (heartbeat_json()). next_seq() is the seq the next event gets.
    uint64_t next_seq() const;
    // Waits up to timeout_ms for events with seq >= cursor; appends them to out and advances
    // cursor. A cursor older than the backlog yields one {"type": "lost", "missed": N} first.
    // Returns false once the node is closing.
    bool wait_events(uint64_t & cursor, std::vector<json> & out, int64_t timeout_ms) const;
    json heartbeat_json() const;

    // Wakes event waiters and makes wait_events() return false (daemon shutdown).
    void close_events();
    bool closing() const { return closing_flag.load(); }

    const server_node_config & config() const { return cfg; }

  private:
    struct child;
    struct orphan;

    void run();
    void push_event_locked(json ev); // takes ev_mu; callers may hold mu
    json child_event_json_locked(const child & c) const;
    void read_output_locked(child & c);
    void handle_line_locked(child & c, const std::string & line);
    static node_child_info info_of(const child & c);

    server_node_config cfg;

    mutable std::mutex      mu;   // the table; ev_mu may be taken while holding it, never the reverse
    std::condition_variable cv;   // table changes (exits, adoptions)
    std::map<std::string, std::shared_ptr<child>> table;
    std::set<std::string>   spawning; // names reserved while their spawn runs unlocked
    std::vector<std::shared_ptr<orphan>> orphan_list;
    bool                    quit = false;

    mutable std::mutex              ev_mu;
    mutable std::condition_variable ev_cv;
    std::deque<json>                events;   // seq = ev_first + index
    uint64_t                        ev_first = 1;
    std::atomic<bool>               closing_flag{false};

    server_subproc::waiter waiter;
    std::thread            th;
};

//
// HTTP layer
//

// Reads a bearer token file (whitespace stripped). Returns "" and sets err when the file is
// missing, unreadable or empty.
std::string server_node_read_token(const std::string & path, std::string & err);

// "" when the token file is not readable by group or others (mode & 077 == 0) or does not exist
// (reading it reports that); else why the daemon / leader must not start (it names chmod 600).
std::string server_node_token_file_mode_error(const std::string & path);

// Whether `argv0` is an executable under one of `allow_dirs`, both resolved by realpath (symlinks
// followed, `..` collapsed): a symlink inside an allowed dir that points out of it, or a `..`
// escape, is not allowed. Whole path components: /opt/bin2/x is not under /opt/bin. A path or a
// dir that does not resolve never matches. An empty `allow_dirs` allows nothing (the caller
// decides whether a list is in force).
bool server_node_exec_allowed(const std::string & argv0, const std::vector<std::string> & allow_dirs);

// A warning when `host` (an address or a name, as --node-bind takes it) resolves to something that is
// neither loopback nor in the Tailscale CGNAT range 100.64.0.0/10 (nor Tailscale's IPv6 ULA
// fd7a:115c:a1e0::/48): the node API is cleartext HTTP over what is meant to be a WireGuard
// network. "" when fine or when it does not resolve.
std::string server_node_bind_warning(const std::string & host);
// The address-literal part of that check (a name is resolved first): true for loopback / CGNAT / Tailscale ULA.
bool server_node_addr_is_overlay(const std::string & ip_literal);

// At most one log line per key (a peer) per interval; the table is bounded.
class server_node_log_limiter {
  public:
    explicit server_node_log_limiter(int64_t interval_ms = 10000, size_t max_keys = 1024)
        : interval_ms(interval_ms), max_keys(max_keys) {}
    // true when `key` may log now (and records it)
    bool allow(const std::string & key, int64_t now_ms);
  private:
    int64_t                       interval_ms;
    size_t                        max_keys;
    std::mutex                    mu;
    std::map<std::string, int64_t> last;
};

// Constant-time check of an "Authorization: Bearer <token>" value against the expected
// token. An empty expected token never matches.
bool server_node_token_matches(const std::string & expected, const std::string & authorization);

// Startup validation of --router-node: "" when fine, else why the node must not start.
// Rejects --board-url (only the leader talks to the board), a model to load, a missing /
// unreadable / empty --node-token-file, and a missing --node-bind or one that resolves to a
// wildcard address (0.0.0.0 / :: in any spelling) or does not resolve.
std::string server_node_check_params(const common_params & params);

// The --node-bind list (comma-separated addresses).
std::vector<std::string> server_node_bind_hosts(const std::string & node_bind);

struct server_node_routes {
    server_node_routes(server_node & node, std::string token);

    // Every route answers 401 without the right bearer token.
    //   POST /node/spawn  {name, gen, args: [..], env?: ["K=V" | "-K", ..] | {K: V | null}, port?, alloc_port?} -> child info
    //   POST /node/stop   {name, timeout_s? (0..3600, default 10), method?: both|stdin|term, wait?: bool} -> child info
    //   POST /node/signal {name, sig: 15 | "TERM" | "SIGTERM"}            -> child info
    //   POST /node/adopt  {name, gen}                                      -> child info
    //   GET  /node/state                                                   -> server_node::state()
    //   GET  /node/events[?since=<seq>]  SSE: "data: <event json>\n\n", heartbeat every heartbeat_ms
    server_http_context::handler_t post_spawn;
    server_http_context::handler_t post_stop;
    server_http_context::handler_t post_signal;
    server_http_context::handler_t post_adopt;
    server_http_context::handler_t get_state;
    server_http_context::handler_t get_events;

    void register_routes(const server_http_context & http) const;

  private:
    server_node & node;
    std::string   token;
    mutable server_node_log_limiter unauthorized_log; // one "unauthorized" line per peer per 10 s
};

// /node/stop's timeout_s: false for NaN / infinity, else clamped to 0..3600 and truncated.
bool server_node_clamp_timeout_s(double v, int & out);

// Signal number from 15, "15", "TERM" or "SIGTERM"; -1 when unknown.
int server_node_parse_signal(const json & sig);
