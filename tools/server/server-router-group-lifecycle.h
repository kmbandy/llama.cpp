#pragma once

// Model-group lifecycle for the router: the expert-worker processes of a group (the
// kind=external sections a spine `depends` on). The router owns them: it launches them
// before the spine, waits for each to accept TCP, stops them all when the group goes, and
// notices when one dies. This unit knows nothing about server_models; server-models.cpp
// drives a router_worker_group per loaded group and reacts to its callbacks.
//
// Also here: the generation marker every router child carries (LLAMA_ROUTER_GEN) and the
// startup sweep that kills children a previous router left behind.

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "server-common.h" // server_subproc

static constexpr const char * ROUTER_ENV_GEN        = "LLAMA_ROUTER_GEN";
static constexpr const char * ROUTER_ENV_ROUTER_PID = "LLAMA_ROUTER_PID";

static constexpr const char * WP_ENV_PARK_FILE      = "WP_EXPERT_PARK_FILE";
static constexpr const char * WP_ENV_SEED_FROM_PARK = "WP_EXPERT_SEED_FROM_PARK";
static constexpr const char * WP_ENV_QUIESCE_MS     = "WP_EXPERT_PARK_QUIESCE_MS";

static constexpr int64_t ROUTER_WORKER_QUIESCE_MS_DEFAULT = 30000; // the worker's own default
static constexpr int64_t ROUTER_WORKER_KILL_GRACE_MS      = 10000; // on top of quiesce, then SIGKILL

//
// worker output
//

enum router_snapshot_result {
    ROUTER_SNAPSHOT_NONE    = 0, // no stop-snapshot line seen
    ROUTER_SNAPSHOT_WRITTEN = 1,
    ROUTER_SNAPSHOT_FAILED  = 2,
    ROUTER_SNAPSHOT_TIMEOUT = 3, // quiesce timed out (the worker snapshots anyway; a later line may follow)
};

const char * router_snapshot_result_str(router_snapshot_result r);

enum router_worker_line_kind {
    ROUTER_WORKER_LINE_OTHER = 0,
    ROUTER_WORKER_LINE_HIP_ERROR,
    ROUTER_WORKER_LINE_SNAPSHOT_WRITTEN,
    ROUTER_WORKER_LINE_SNAPSHOT_FAILED,
    ROUTER_WORKER_LINE_SNAPSHOT_TIMEOUT,
};

struct router_worker_line {
    router_worker_line_kind kind = ROUTER_WORKER_LINE_OTHER;
    std::string             detail; // written: "<path> (<N> rows)"; FAILED: reason; timeout: "<ms> ms"; Hip: the line
};

// Classifies one line of worker output. Matches anywhere in the line, so a log prefix is fine:
//   "wp expert worker: stop snapshot written: <path> (<N> rows)"
//   "wp expert worker: stop snapshot FAILED: <reason>"
//   "wp expert worker: stop snapshot: quiesce timeout after <ms> ms, snapshotting anyway"
//   anything containing "Hip error"
router_worker_line router_classify_worker_line(const std::string & line);

//
// launch command
//

struct router_launch {
    std::vector<std::string> argv;   // what to exec; {"/bin/sh", "-c", "exec ..."} when via_shell
    std::vector<std::string> env;    // leading NAME=VALUE words, applied over the worker env
    std::vector<std::string> words;  // the command's words after quote removal (port lookup)
    bool                     via_shell = false;
};

// Splits a `launch` string the way /bin/sh would for a single simple command: whitespace,
// '...', "...", backslash escapes; leading NAME=VALUE words become env entries. A command
// that needs expansion ($, `, globs, leading ~) is run as `/bin/sh -c "exec <command>"`, so the
// tracked PID is still the worker's. Pipes, lists, redirections, subshells and comments are
// refused: the router must own the worker's PID and read its output.
// Without the shell the command must be an absolute path: PATH is never searched.
// Returns "" on success, else the reason.
std::string router_parse_launch(const std::string & launch, router_launch & out);

// The TCP endpoint a worker listens on, read from its words: `--listen HOST:PORT`,
// `--listen=HOST:PORT`, `--port N`, `--port=N`. A wildcard host (0.0.0.0, ::, empty) becomes
// 127.0.0.1. Returns the port, or -1 if none is given.
int router_launch_endpoint(const std::vector<std::string> & words, std::string & host);

// The host part of a `--listen HOST:PORT` / `--listen=HOST:PORT` word as written (brackets
// stripped); "" when there is none or it is a wildcard (0.0.0.0, ::, *, empty).
std::string router_launch_listen_host(const std::vector<std::string> & words);

// A worker that runs on another machine is reached over the LAN and has no auth in its protocol, so
// it must not listen on every interface. Refuses (returns the reason) a launch that names a wildcard
// listen address: `--listen 0.0.0.0:P` / `--listen=:P` / `--listen [::]:P`, `--host 0.0.0.0` /
// `--host=::` / `-H ''` (0.0.0.0, ::, *, empty). A launch that names no host at all gets
// `--host <node_host>` (the node's bind address: appended to argv, to the command string of a
// /bin/sh launch, and to words) so it listens on that address only. A named, non-wildcard host is
// left as written. Returns "" when the launch is fine (or was fixed).
std::string router_remote_worker_host_fix(router_launch & launch, const std::string & node_host);

// Sets the port of an argv: replaces the value of `--port N` / `--port=N`, else appends
// `--port N`. Used when a node allocates a child's port.
void router_args_set_port(std::vector<std::string> & args, int port);
// Same for `--host`: replaces an existing `--host X` / `--host=X`, else appends `--host X`.
void router_args_set_host(std::vector<std::string> & args, const std::string & host);

//
// environment helpers ("KEY=VALUE" lists)
//

// Replaces every definition of key (duplicates resolve differently across libcs).
void router_env_set(std::vector<std::string> & env, const std::string & key, const std::string & value);
void router_env_unset(std::vector<std::string> & env, const std::string & key);
// Last definition of key; false when unset.
bool router_env_get(const std::vector<std::string> & env, const std::string & key, std::string & value);

// One preset-style env override: "KEY=VALUE" sets KEY, "-KEY" removes it. "" when well formed,
// else why not (empty key, bare "-", "-KEY=..." , no '=').
std::string router_env_override_error(const std::string & entry);

// Applies overrides in order: "KEY=VALUE" replaces every definition of KEY, "-KEY" removes
// every definition. Malformed entries (router_env_override_error) are skipped.
void router_env_apply_overrides(std::vector<std::string> & env, const std::vector<std::string> & overrides);

// Option / env names that configure the router (or node) itself: TLS, API keys, the model
// registry, GPU slots, board and node flags. A child must never get them from the router's
// own configuration; LLAMA_ARG_ROUTER_* (per-model router keys) are reserved as well.
const std::vector<std::string> & router_reserved_option_keys();
bool router_is_reserved_option_key(const std::string & key);

// The rest of a worker's env, on top of `env` (router env + the preset `env` key): the
// launch line's leading NAME=VALUE words, then WP_EXPERT_PARK_FILE=<park_file> and
// WP_EXPERT_SEED_FROM_PARK=1 when a park file is set, then LLAMA_ROUTER_GEN=<gen>.
std::vector<std::string> router_worker_env(std::vector<std::string> env, const std::vector<std::string> & launch_env,
                                           const std::string & park_file, const std::string & gen);

// WP_EXPERT_PARK_QUIESCE_MS from the worker's final env, else the worker's default (30000).
int64_t router_env_quiesce_ms(const std::vector<std::string> & env);

//
// sockets
//

// True if host:port accepts a TCP connection within timeout_ms. POSIX only (false elsewhere).
bool router_tcp_accepts(const std::string & host, int port, int timeout_ms);

//
// router generation + stale-child sweep (Linux /proc; no-ops elsewhere)
//

// A fresh random id (uuid v4 format) for this router start.
std::string router_generation_new();

// PIDs under <root>/proc that a previous router generation left behind: the process runs as
// `uid`, its environ carries LLAMA_ROUTER_GEN with a value other than `gen`, and the router
// named by its LLAMA_ROUTER_PID is not one of its ancestors (so a second router running on
// the same machine never sweeps the children of a live one). Unreadable processes are
// skipped, and so is self_pid. Sorted.
std::vector<int> router_find_stale_children(const std::string & root, const std::string & gen, unsigned uid, int self_pid);

// router_find_stale_children(), then SIGTERM each, wait up to grace_ms for them to go (a
// worker writes its stop snapshot on TERM), SIGKILL whatever is left. Signals are real and
// liveness is read from the real /proc even when `root` points elsewhere (tests describe a
// real process in a fake tree). Returns the PIDs signalled.
std::vector<int> router_sweep_stale_children(const std::string & root, const std::string & gen, unsigned uid, int self_pid, int64_t grace_ms);

//
// the worker group
//

class router_node_link; // server-router-node-client.h

struct router_worker_spec {
    std::string              name;
    std::vector<std::string> argv;
    // env overrides ("KEY=VALUE" sets, "-KEY" unsets) on the base env of the node it runs on. The
    // group's own in-process node (no `node`) has an empty base env: this is the final environment.
    std::vector<std::string> env;
    std::string              host = "127.0.0.1"; // where the router checks that it accepts TCP
    int                      port = 0;
    int                      startup_timeout_s = 300;
    int64_t                  quiesce_ms = ROUTER_WORKER_QUIESCE_MS_DEFAULT;
    // the node it runs on (this machine's or another's); null = an in-process node of the group's own
    std::shared_ptr<router_node_link> node;
};

// Point-in-time view of one worker, for the status JSON and tests.
struct router_worker_state {
    std::string            name;
    std::string            machine;  // its node's machine ("" = this one)
    int                    pid  = 0; // 0 before spawn; the PID on its machine
    std::string            host;
    int                    port = 0;
    std::string            state;    // "pending" | "starting" | "ready" | "stopping" | "exited"
    int                    exit_code = -1;
    bool                   killed = false;          // the router (or its node) had to SIGKILL it
    router_snapshot_result snapshot = ROUTER_SNAPSHOT_NONE;
    std::string            snapshot_detail;
    bool                   quiesce_timeout = false; // the timeout line was seen (a final line may follow)
    std::string            error;                   // Hip error line, or why the start failed
};

// Owns the worker processes of one group. Each worker runs on a node (router_node_link): the
// router's own machine or another one; its output lines and its exit come back as node events,
// so a remote worker's stop-snapshot and Hip-error lines are classified exactly like a local
// one's. One thread per group applies those events, sends the TERM / KILL commands (the node
// enforces the TERM -> KILL deadline) and runs the callbacks.
//
// Callbacks run on the group thread with no lock of this class held. They must not destroy
// this object (its destructor joins that thread).
class router_worker_group {
  public:
    struct callbacks {
        // a worker exited after start() succeeded and before request_stop()
        std::function<void(const std::string & worker, int exit_code, const std::string & reason)> on_unexpected_exit;
        // after request_stop(), once every spawned worker has exited; fires exactly once
        std::function<void()> on_stopped;
        // every output line (without the newline); optional, used by tests
        std::function<void(const std::string & worker, const std::string & line)> on_line;
    };

    // gen: LLAMA_ROUTER_GEN of the workers; "" = the one in each spec's env (else "router")
    router_worker_group(std::string group, std::vector<router_worker_spec> specs, callbacks cb,
                        int64_t kill_grace_ms = ROUTER_WORKER_KILL_GRACE_MS, std::string gen = "");
    // SIGKILLs anything still running (no callbacks fire), waits for the exits (bounded when a
    // node does not answer), then joins the thread
    ~router_worker_group();

    router_worker_group(const router_worker_group &) = delete;
    router_worker_group & operator=(const router_worker_group &) = delete;

    // Blocking. Spawns every worker, then waits until each accepts TCP on its port. Fails on:
    // a port already taken before spawn, a spawn error, a worker exiting, any output line with
    // "Hip error", its startup timeout, or cancelled() returning true. On failure everything
    // already started is torn down (ready workers get TERM with the usual bound, the rest
    // KILL) and reaped before this returns false with the reason in err. Call once.
    bool start(std::string & err, const std::function<bool()> & cancelled = nullptr);

    // Non-blocking. SIGTERM every live worker in parallel; each one that is still alive after
    // its quiesce + kill grace gets SIGKILL (from its node). on_stopped fires once all have exited.
    void request_stop();

    // Blocks until every spawned worker has exited (and, if a stop was requested, on_stopped
    // has fired), or timeout_ms passes (< 0 = no limit). True when that happened.
    bool wait_stopped(int64_t timeout_ms);

    std::set<int>                    pids() const; // live workers (PIDs on their own machines)
    std::vector<router_worker_state> status() const;
    bool                             all_exited() const;

  private:
    struct event {
        bool        exit = false; // else an output line
        size_t      idx  = 0;
        std::string line;
        int         exit_code = -1;
        bool        killed    = false;
    };
    // shared with the node watch callbacks, which may outlive this object
    struct shared_state {
        std::mutex              mu; // guards everything of the group
        std::condition_variable cv;
        std::deque<event>       events;
    };

    struct member {
        router_worker_spec                spec;
        std::shared_ptr<router_node_link> node;
        bool                              spawned = false;
        bool                              ready = false;
        bool                              exited = false;
        bool                              stopping = false;
        bool                              killed = false;
        bool                              term_pending = false;
        bool                              kill_pending = false;
        int64_t                           kill_deadline = 0; // our own SIGKILL deadline (steady ms)
        int64_t                           retry_at = 0;  // a command the node did not take: try again then
        int                               pid = 0;
        int                               exit_code = -1;
        bool                              hip_error = false;
        std::string                       error;
        router_snapshot_result            snapshot = ROUTER_SNAPSHOT_NONE;
        std::string                       snapshot_detail;
        bool                              quiesce_timeout = false;
    };

    void run();
    void handle_line_locked(member & m, const std::string & line, std::vector<std::pair<std::string, std::string>> & lines_out);
    bool all_spawned_exited_locked() const;
    void teardown_failed_start(std::unique_lock<std::mutex> & lk);
    // waits (lock held via lk) until every spawned worker exited or the bound passes; then gives
    // up on the rest (unwatched, marked exited) so nothing waits forever on a node that is gone
    void wait_exits_bounded(std::unique_lock<std::mutex> & lk, int64_t bound_ms);
    std::shared_ptr<router_node_link> own_node();

    std::string group;
    callbacks   cb;
    int64_t     kill_grace_ms;
    std::string gen;

    std::shared_ptr<shared_state> st;
    std::vector<member>     members; // size fixed at construction
    bool                    armed          = false; // start() succeeded: exits are now unexpected
    bool                    stop_requested = false;
    bool                    stop_fired     = false;
    bool                    closing        = false; // destructor running: no more callbacks
    bool                    quit           = false;

    std::shared_ptr<router_node_link> private_node; // for specs without a node
    std::thread            th;
};
