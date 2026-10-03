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
// Returns "" on success, else the reason.
std::string router_parse_launch(const std::string & launch, router_launch & out);

// The TCP endpoint a worker listens on, read from its words: `--listen HOST:PORT`,
// `--listen=HOST:PORT`, `--port N`, `--port=N`. A wildcard host (0.0.0.0, ::, empty) becomes
// 127.0.0.1. Returns the port, or -1 if none is given.
int router_launch_endpoint(const std::vector<std::string> & words, std::string & host);

//
// environment helpers ("KEY=VALUE" lists)
//

// Replaces every definition of key (duplicates resolve differently across libcs).
void router_env_set(std::vector<std::string> & env, const std::string & key, const std::string & value);
void router_env_unset(std::vector<std::string> & env, const std::string & key);
// Last definition of key; false when unset.
bool router_env_get(const std::vector<std::string> & env, const std::string & key, std::string & value);

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

struct router_worker_spec {
    std::string              name;
    std::vector<std::string> argv;
    std::vector<std::string> env;  // final environment, nothing is added
    std::string              host = "127.0.0.1";
    int                      port = 0;
    int                      startup_timeout_s = 300;
    int64_t                  quiesce_ms = ROUTER_WORKER_QUIESCE_MS_DEFAULT;
};

// Point-in-time view of one worker, for the status JSON and tests.
struct router_worker_state {
    std::string            name;
    int                    pid  = 0; // 0 before spawn
    std::string            host;
    int                    port = 0;
    std::string            state;    // "pending" | "starting" | "ready" | "stopping" | "exited"
    int                    exit_code = -1;
    bool                   killed = false;          // the router had to SIGKILL it
    router_snapshot_result snapshot = ROUTER_SNAPSHOT_NONE;
    std::string            snapshot_detail;
    bool                   quiesce_timeout = false; // the timeout line was seen (a final line may follow)
    std::string            error;                   // Hip error line, or why the start failed
};

// Owns the worker processes of one group. One thread per group reads their output, reaps
// them and enforces the TERM -> KILL deadlines; it is the only thread that reaps or signals
// them, so a PID is never signalled after it was reaped (no PID-reuse race).
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

    router_worker_group(std::string group, std::vector<router_worker_spec> specs, callbacks cb,
                        int64_t kill_grace_ms = ROUTER_WORKER_KILL_GRACE_MS);
    // SIGKILLs anything still running (no callbacks fire), then joins the thread
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
    // its quiesce + kill grace gets SIGKILL. on_stopped fires once all have exited.
    void request_stop();

    // Blocks until every spawned worker has exited (and, if a stop was requested, on_stopped
    // has fired), or timeout_ms passes (< 0 = no limit). True when that happened.
    bool wait_stopped(int64_t timeout_ms);

    std::set<int>                    pids() const; // live workers
    std::vector<router_worker_state> status() const;
    bool                             all_exited() const;

  private:
    struct member {
        router_worker_spec              spec;
        std::unique_ptr<server_subproc> proc;
        std::string                     buf;          // partial line (group thread only)
        bool                            eof = false;  // group thread only
        bool                            spawned = false;
        bool                            ready = false;
        bool                            exited = false;
        bool                            stopping = false;
        bool                            killed = false;
        bool                            term_pending = false;
        bool                            kill_pending = false;
        int64_t                         kill_deadline = 0;
        int                             pid = 0;
        int                             exit_code = -1;
        bool                            hip_error = false;
        std::string                     error;
        router_snapshot_result          snapshot = ROUTER_SNAPSHOT_NONE;
        std::string                     snapshot_detail;
        bool                            quiesce_timeout = false;
    };

    void run();
    void handle_line_locked(member & m, const std::string & line, std::vector<std::pair<std::string, std::string>> & lines_out);
    void read_output_locked(member & m, std::vector<std::pair<std::string, std::string>> & lines_out);
    bool all_spawned_exited_locked() const;
    void teardown_failed_start(std::unique_lock<std::mutex> & lk);

    std::string group;
    callbacks   cb;
    int64_t     kill_grace_ms;

    mutable std::mutex      mu;
    std::condition_variable cv;
    std::vector<member>     members; // size fixed at construction
    bool                    armed          = false; // start() succeeded: exits are now unexpected
    bool                    stop_requested = false;
    bool                    stop_fired     = false;
    bool                    closing        = false; // destructor running: no more callbacks
    bool                    quit           = false;

    server_subproc::waiter waiter;
    std::thread            th;
};
