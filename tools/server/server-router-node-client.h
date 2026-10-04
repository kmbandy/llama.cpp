#pragma once

// The leader router's handle on one machine's node (spec §3): the in-process server_node for its
// own machine, the --router-node daemon's HTTP API for every other machine in machines.json.
// Every child the router runs -- llama-server spines and plain models, group workers, estimate
// and download children -- is spawned, stopped, signalled and watched through this interface, so
// local and remote children share one code path.
//
// A link:
//   - spawns a child and watches it: its output lines and its exit reach the watch callbacks, in
//     node order, on the link's event thread (no link lock held). Callbacks must be short and must
//     not call back into the link's spawn(); the router hands them to its own threads;
//   - stops / signals children: synchronously (callers on their own thread) or asynchronously
//     (queued to the link's command thread; safe under any lock, the router's mutex included);
//   - for a remote node: follows /node/events (SSE). No data for offline_ms (heartbeats come every
//     2 s) -> offline; data again -> reconcile (adopt our orphans by (name, gen), stop what nobody
//     wants, report children that are gone), then online. Watched children the node no longer has
//     get an exit with exit_code -1. A child of ours the node runs but nobody watches (a spawn whose
//     answer was lost) is stopped after two heartbeats;
//   - caches the node's probe data (/node/state: per-PID VRAM, RSS, MemAvailable, sysfs VRAM used)
//     for the ledger, refreshed every probe_ms while online.
//
// Never call the synchronous methods of a remote link with the router's mutex held: they are HTTP.

#include "server-node.h"
#include "server-router-ledger.h"
#include "server-router-probe.h"

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

static constexpr int64_t ROUTER_NODE_OFFLINE_MS_DEFAULT = 10000; // heartbeat lost
static constexpr int64_t ROUTER_NODE_PROBE_MS_DEFAULT   = 5000;  // /node/state refresh
static constexpr int     ROUTER_NODE_RETRY_AFTER_S      = 10;    // Retry-After of a 503 for an offline machine
static constexpr int64_t ROUTER_NODE_SPAWN_409_RETRY_MS = 2000;  // a spawn refused 409 (name still held) is retried this long
static constexpr int     ROUTER_NODE_RECONCILE_STOP_S   = 40;    // stop bound for what a reconcile stops (worker quiesce + grace)

//
// pure helpers
//

// "http://host:port[/]" -> base "http://host:port", host (IPv6 brackets stripped), port.
bool router_node_parse_url(const std::string & url, std::string & base, std::string & host, int & port, std::string & err);

// node child info from its JSON form ({name, gen, pid, port, status, exit_code, killed, adopted, started_ms})
node_child_info router_node_child_info_from_json(const json & j);

// What a node reported in /node/state, for the ledger.
struct router_node_probe {
    bool                           ok = false;       // a state was read
    int64_t                        mem_available = -1;
    std::vector<proc_vram>         vram;             // every PID's VRAM per device
    std::map<std::string, int64_t> sysfs_used;       // pdev -> card-wide VRAM used (-1 unreadable)
    std::map<int, proc_mem>        rss;              // pid -> RssAnon / RssShmem of the node's live children
    std::set<int>                  child_pids;       // the node's live children (router-owned: never foreign)
};
router_node_probe router_node_probe_from_state(const json & state);

// Ledger free VRAM of a slot on that node (ledger_free_vram with the node's numbers): foreign =
// VRAM on slot.pdev of PIDs that are not the node's children; sysfs from `devices`. A probe that
// was never read gives the ledger term alone.
int64_t router_node_free_vram(const router_node_probe & p, const ledger_slot & slot);
// RssAnon + RssShmem of one of the node's children; -1 when unknown
int64_t router_node_child_ram(const router_node_probe & p, int pid);

// A child as a node reports it (state `children`, heartbeat `children`).
struct router_node_seen {
    std::string name;
    std::string gen;
    int         pid       = 0;
    std::string status;      // running | stopping | exited
    int         exit_code = -1;
    bool        killed    = false;
};
std::vector<router_node_seen> router_node_seen_from_json(const json & children);

// What a link does when it (re)connects to a node (or lost events).
struct router_node_reconcile_plan {
    std::vector<std::string>      keep;       // watched, running, ours, still wanted
    std::vector<std::string>      adopt;      // our orphans (same name, gen, pid as watched) still wanted: take them back
    std::vector<std::string>      adopt_stop; // our orphans nobody wants: take them back, then stop them
    std::vector<std::string>      stop;       // live children nobody wants (another generation, unknown, unwanted)
    std::vector<router_node_seen> lost;       // watched children the node no longer runs (exit_code -1 if unknown)
};
// `watched`: name -> PID of the children the link watches; `gen`: the leader's generation;
// `wanted(name)`: the router still expects that child to run.
router_node_reconcile_plan router_node_reconcile(const std::vector<router_node_seen> & children,
                                                 const std::vector<node_orphan_info> & orphans,
                                                 const std::map<std::string, int> & watched, const std::string & gen,
                                                 const std::function<bool(const std::string &)> & wanted);

// A request for a model whose machine is offline: 503 + Retry-After, OAI error shape.
struct router_unavailable_error : std::runtime_error {
    std::string machine;
    router_unavailable_error(const std::string & machine, const std::string & msg) : std::runtime_error(msg), machine(machine) {}
};
void router_unavailable_response(const std::string & model, const std::string & machine, const std::string & reason,
                                 int & status, std::string & body, std::map<std::string, std::string> & headers);

// The members of a pool (slot ids) whose machine is online, in order; `slot_machine` maps a slot
// id to its machine ("" = local), `offline` holds the machines whose node is offline.
std::vector<std::string> router_online_slots(const std::vector<std::string> & slots,
                                             const std::function<std::string(const std::string &)> & slot_machine,
                                             const std::set<std::string> & offline);

// One `gpus=` entry: [machine/]dev[=board]:total_mb:probe. An entry without a machine, or with the
// local machine's name, is local (machine ""). A remote entry needs total_mb > 0 (the leader cannot
// read the other box's sysfs) and a probe naming the card's PCI address: "pci:0000:03:00.0" or a
// sysfs path containing it (".../0000:03:00.0/mem_info_vram_used").
struct router_gpu_spec_entry {
    std::string machine;          // "" = local
    std::string dev;
    std::string board;            // "" = dev
    int64_t     total_mb = 0;     // 0 = probe's physical total (local only)
    std::string probe;
    std::string pdev;             // remote: from the probe; local: resolved by the caller
};
// "" on success, else why the spec is bad
std::string router_parse_gpus_spec(const std::string & spec, const std::function<bool(const std::string &)> & is_local,
                                   std::vector<router_gpu_spec_entry> & out);
// the PCI address in a probe string ("pci:<addr>" or a path containing <dddd:bb:dd.f>); "" if none
std::string router_probe_pdev(const std::string & probe);

// Splits a slot id "<machine>/<dev>" ("<dev>" = local).
void router_slot_split(const std::string & id, std::string & machine, std::string & dev);

// A preset `gpu=` entry as a slot id. "<machine>/<dev>" names its machine; a bare device name is on
// `machine` (the preset's `machine=`, "" = this machine). The local machine (is_local) is dropped:
// a local slot's id is the bare device, so single-machine presets and placements read as before.
std::string router_slot_resolve(const std::string & gpu, const std::string & machine,
                                const std::function<bool(const std::string &)> & is_local);

// Heartbeat loss: a remote node that is online and silent for offline_ms goes offline; one that is
// offline and was heard from within offline_ms is back (after its reconcile). Pure: the clocks are
// arguments.
bool router_node_offline_due(bool online, int64_t last_rx_ms, int64_t now_ms, int64_t offline_ms);
bool router_node_online_due(bool online, int64_t last_rx_ms, int64_t now_ms, int64_t offline_ms);

// Which machines are offline, as the router sees them. Marking is reversible: set_online() flips it
// back and reports whether anything changed. "" (this machine) is never offline.
struct router_machine_availability {
    std::set<std::string> offline;
    bool set_online(const std::string & machine, bool online);
    bool is_offline(const std::string & machine) const { return !machine.empty() && offline.count(machine) > 0; }
    // the first of `machines` that is offline; "" when none is
    std::string first_offline(const std::vector<std::string> & machines) const;
};

// The status a model shows: "unavailable" while its machine is offline, else its own status.
std::string router_effective_status(const std::string & status, bool machine_offline);

//
// the link
//

struct router_node_watch {
    // the node accepted the spawn (info: pid, port); before any line or exit of it is delivered. Runs
    // on the thread that called spawn(), no link lock held.
    std::function<void(const node_child_info & info)> on_spawn;
    std::function<void(const std::string & line)>     on_line; // one output line, without the newline
    std::function<void(const node_child_info & info)> on_exit; // exactly once; exit_code -1 when unknown
};

struct router_node_hooks {
    std::function<void(bool online)>              on_online; // remote: the node went offline / is back (after the reconcile)
    std::function<bool(const std::string & name)> wanted;    // the router still expects this child to run
};

struct router_node_link_config {
    std::string machine;                          // its name in machines.json ("" = this machine)
    std::string gen;                              // the leader's generation (children of others are stopped)
    int64_t     heartbeat_ms = NODE_HEARTBEAT_MS_DEFAULT; // local: how often the table is checked
    int64_t     offline_ms   = ROUTER_NODE_OFFLINE_MS_DEFAULT;
    int64_t     probe_ms     = ROUTER_NODE_PROBE_MS_DEFAULT;
    int         http_timeout_ms = 10000;
};

class router_node_link {
  public:
    virtual ~router_node_link();

    router_node_link(const router_node_link &) = delete;
    router_node_link & operator=(const router_node_link &) = delete;

    const std::string & machine() const { return cfg.machine; }
    const std::string & host() const { return host_; } // where its children listen / are reached
    virtual bool remote() const = 0;
    bool online() const { return online_flag.load(); }
    std::string exe() const; // the node's own binary (llama-server children run as it); "" until known

    void set_hooks(router_node_hooks h); // before start()
    virtual void start() = 0;
    virtual void shutdown() = 0;         // joins the threads; idempotent

    // Starts a child and watches it. Throws server_node_error: the node's status (400 / 409 / 500 /
    // 503), 503 when the node is offline, 0 when it did not answer (a child it may have started
    // anyway is stopped). The watch is registered before the spawn, so no line or exit is missed.
    node_child_info spawn(const node_spawn_request & req, router_node_watch w);
    // Watches a child started earlier (by this link, before a restart of its owner); tests.
    void watch_existing(const std::string & name, int pid, router_node_watch w);
    // Drops the watch of a child without waiting for its exit (its callbacks are not called again).
    void unwatch(const std::string & name);

    // synchronous: throw server_node_error like spawn()
    virtual node_child_info stop(const std::string & name, int timeout_s, const std::string & method) = 0;
    virtual node_child_info signal(const std::string & name, int sig) = 0;
    virtual node_child_info adopt(const std::string & name, const std::string & gen) = 0;
    virtual json            state() = 0;

    // asynchronous (command thread); skipped when `name` is watched with another PID (a newer child)
    void stop_async(const std::string & name, int pid, int timeout_s, const std::string & method);
    void signal_async(const std::string & name, int pid, int sig);

    router_node_probe probe() const;
    void              refresh_probe(); // synchronous /node/state; failures leave the cache as it was

  protected:
    router_node_link(router_node_link_config cfg, std::string host);

    virtual node_child_info do_spawn(const node_spawn_request & req) = 0;
    // a spawn whose outcome is unknown (no answer): best-effort stop of whatever got that name
    virtual void after_failed_spawn(const std::string & name) { (void) name; }

    // event thread: events and heartbeats, in stream order (dispatch_mu serializes them with replays)
    void on_event(const json & ev);
    void on_heartbeat(const json & hb);
    // remote: offline after offline_ms without data; back -> reconcile, then online
    void note_rx();
    void check_offline();
    // command thread
    void post(std::function<void()> fn);
    void start_commands();
    void stop_commands();
    void reconcile();

    router_node_link_config cfg;
    std::string             host_;
    std::atomic<bool>       online_flag{false};
    std::atomic<int64_t>    last_rx_ms{0};

    mutable std::mutex mu; // watches, cursor, probe, hooks; never held across a callback or HTTP

    uint64_t cursor   = 0;     // next event seq expected (guarded by mu)
    bool     have_cursor = false;
    int      node_pid = 0;
    std::string node_exe;
    bool     reconcile_pending = false;

  private:
    struct watch_entry {
        uint64_t          id  = 0;
        int               pid = 0;     // 0: spawn in flight
        bool              seen = false; // an event of this child came through the stream
        router_node_watch w;
        std::vector<json> pending;     // events that arrived while the spawn was in flight
    };

    node_child_info spawn_once(const node_spawn_request & req, const router_node_watch & w);
    void dispatch_locked_d(const json & ev); // dispatch_mu held
    void fire_exit(const std::shared_ptr<watch_entry> & e, const std::string & name, const node_child_info & info);
    std::shared_ptr<watch_entry> take_watch(const std::string & name, int pid);

    std::mutex dispatch_mu; // one event (and its callback) at a time, in stream order
    std::map<std::string, std::shared_ptr<watch_entry>> watches;
    uint64_t next_watch_id = 1;
    std::map<std::string, int> stray; // name -> heartbeats it ran unwatched
    router_node_hooks hooks;
    router_node_probe probe_cache;

    std::thread                       cmd_th;
    std::mutex                        cmd_mu;
    std::condition_variable           cmd_cv;
    std::deque<std::function<void()>> cmd_q;
    bool                              cmd_quit = false;
};

// Per-child API key for a child on another machine (it listens on the LAN): 32 random hex chars.
std::string router_child_key_generate();
// Sets `Authorization: Bearer <key>` on a proxied request's headers, dropping whatever the client
// sent under any case of the name. An empty key leaves the headers untouched (local children).
void router_child_auth_headers(std::map<std::string, std::string> & headers, const std::string & key);
// What the exit of a child does to the model's stop flag (stopping_models) when the entry does not
// hold that child: dropped only if the entry is gone; an entry held by a newer instance (or a
// restored previous one) keeps its flag, which belongs to it, not to the exiting orphan.
inline bool router_orphan_exit_clears_stop_flag(bool entry_exists) {
    return !entry_exists;
}

// The leader's own machine: an in-process server_node (owned by the link).
std::shared_ptr<router_node_link> router_node_make_local(router_node_link_config cfg, server_node_config node_cfg);
// Another machine: its --router-node daemon at `url`, bearer `token`.
std::shared_ptr<router_node_link> router_node_make_remote(router_node_link_config cfg, const std::string & url,
                                                          const std::string & token);
