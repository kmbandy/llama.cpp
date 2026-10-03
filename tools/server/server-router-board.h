#pragma once

// The router as a citizen of the coordination board (mneme's board, reached over HTTP through
// mad-lab-mcp's /board/* routes). A claim is permission to use a resource; a busy resource
// puts the claimant in a queue. The router:
//   - claims every resource a load needs (`gpu:<board name>` per GPU slot, `ram` for host RAM),
//     holder `llama-router`, note = the model, ttl 1 h, renewed every 10 min, released on unload;
//   - queues behind sessions (a foreign claim makes admission answer `queue`) and notifies the
//     holder (`wait` at middle, `yield` at highest);
//   - yields: when a session queues for a resource the router holds, idle residents there go.
//
// Pieces:
//   - pure helpers (request priority/machine, queue-verdict mapping, JSON parsing, claim
//     expansion, yield selection), unit-tested with literals;
//   - router_board_client: one HTTP call per method, no state;
//   - router_board_agent: the cache, the claims the router holds, and the poll thread
//     (poll every 3 s, renew, release claims of residents that went away, yield). It talks to
//     the router only through router_board_host callbacks and never holds its own lock across
//     HTTP or a callback, so the router's mutex is never held during HTTP.
//
// Board outage: a failed poll marks the cache unavailable; admission then sees no foreign
// claims (loads go ahead), claims that cannot be taken are skipped at load and taken later by
// the poll thread for residents still running, and renewals / releases retry every tick.

#include "server-router-admission.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

static constexpr const char * ROUTER_BOARD_HOLDER       = "llama-router";
static constexpr const char * ROUTER_BOARD_RAM_RESOURCE = "ram";
static constexpr const char * ROUTER_BOARD_MACHINE_RES  = "machine"; // a whole-machine claim conflicts with everything
static constexpr int          ROUTER_RETRY_AFTER_S      = 10;

//
// request options: priority and machine
//

struct router_request_opts {
    admission_priority priority     = ADMISSION_PRIORITY_MIDDLE;
    bool               priority_set = false; // false: the alias default (preset `priority`) applies
    std::string        machine;              // machine override, "" = any
};

// The JSON body field wins over the header (X-Priority / X-Machine); both empty = unset.
// Returns false with `err` for a priority other than highest / middle / lowest.
bool router_parse_request_opts(const std::string & body_priority, const std::string & header_priority,
                               const std::string & body_machine, const std::string & header_machine,
                               router_request_opts & out, std::string & err);

//
// queue verdicts
//

enum router_queue_action {
    ROUTER_QUEUE_REFUSE = 0, // can never fit as configured, or a pin / hold is in the way: error now
    ROUTER_QUEUE_WAIT   = 1, // a foreign board claim or a busy resident: wait in the queue
};

// Only meaningful for a `queue` verdict (anything else maps to REFUSE).
router_queue_action router_admission_queue_action(const admission_result & res);

// A request whose model is queued: `lowest` blocks until the model is ready, `middle` and
// `highest` get 503 + Retry-After + the queue info right away.
bool router_request_waits_in_queue(admission_priority p);

// Where a queued load stands.
struct router_queued_info {
    std::string        model;
    admission_priority priority   = ADMISSION_PRIORITY_MIDDLE;
    std::string        machine;    // the requested machine override ("" = any)
    bool               board      = false; // waiting on a board claim (else on a router resident)
    int                queue_pos  = 0;     // 1-based; 0 = unknown
    std::string        blocked_by; // claim holder or resident name
    std::string        blocked_on; // board resource or slot
    std::string        reason;
};

// The load of a model cannot start now and waits in the queue (the router retries it).
struct router_queued_error : std::runtime_error {
    router_queued_info info;
    explicit router_queued_error(router_queued_info i);
};

// The load can never be admitted as things stand (capacity, no candidate, pinned, held).
struct router_refused_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// JSON object {state: "queued", queue_pos, blocked_by, blocked_on, board, priority, reason}
std::string router_queued_info_json(const router_queued_info & info);

// The 503 a `middle` / `highest` request gets while its model is queued: body (OAI error shape
// with a `queue` object) and headers (Retry-After).
void router_queued_response(const router_queued_info & info, int & status, std::string & body,
                            std::map<std::string, std::string> & headers);

//
// configuration helpers
//

// `gpus=` slot name field: "ROCm0" or "ROCm0=R9700" (device = board name). Returns false when
// either side of '=' is empty.
bool router_gpu_name_split(const std::string & field, std::string & dev, std::string & board_name);

// The `local: true` entry of machines.json ({name: {local?, ...}}); "" when none / unparsable.
std::string router_parse_local_machine(const std::string & machines_json);

// Local machine name: machines.json at `path` (default ~/.config/mad-lab-agents/machines.json
// when empty), else the short hostname, else "local".
std::string router_local_machine(const std::string & path = "");

//
// board data
//

struct router_board_claim {
    std::string        id;
    std::string        machine;
    std::string        resource;
    std::string        holder;
    std::string        note;
    admission_priority priority  = ADMISSION_PRIORITY_MIDDLE;
    bool               probation = false; // auto-claimed from the queue, not confirmed yet
};

struct router_board_queue_entry {
    std::string        id;
    std::string        machine;
    std::string        resource;
    std::string        holder; // the queue row's session_id
    admission_priority priority = ADMISSION_PRIORITY_MIDDLE;
};

struct router_board_snapshot {
    bool                                  ok = false; // the last poll succeeded
    std::vector<router_board_claim>       claims;     // active claims on this machine
    std::vector<router_board_queue_entry> queue;      // waiting entries on this machine
};

// GET /board/claims body: a JSON array of claim rows
bool router_board_parse_claims(const std::string & body, std::vector<router_board_claim> & out);
// GET /board/queue body: {"queue": [rows]}
bool router_board_parse_queue(const std::string & body, std::vector<router_board_queue_entry> & out);

struct router_board_claim_result {
    bool        ok      = false; // the board answered 2xx with a usable body
    bool        granted = false; // a live claim: claim_id
    bool        queued  = false; // joined the queue: position, held_by, queue_id
    std::string claim_id;
    int         position = 0;
    std::string held_by;
    std::string queue_id;
    int         status = 0; // HTTP status, 0 = no answer
    std::string error;
};

// POST /board/claims answer: granted = the claim row ({id, ...}); queued = {queued: true,
// position, held_by, queue_id, ...}; anything else (or a non-2xx status) is an error.
router_board_claim_result router_board_parse_claim_response(int status, const std::string & body);

// Admission claims from the cache: every active claim on the machine; holder `llama-router`
// is is_router (never blocks). A whole-machine claim stands for every resource listed in
// `machine_resources`. An unavailable cache gives none (loads go ahead).
std::vector<admission_claim> router_board_admission_claims(const router_board_snapshot & snap,
                                                           const std::vector<std::string> & machine_resources);

// Of the resources the router holds, those someone other than the router is queued for (a
// queued whole-machine entry contests every one of them).
std::set<std::string> router_board_contested(const router_board_snapshot & snap, const std::set<std::string> & held);

// A router-owned process, as the agent sees it.
struct router_board_resident {
    std::string              name;
    std::string              stop_name; // what to unload for it (a worker: its spine); "" = name
    std::vector<std::string> resources; // board resources it should hold while running
    bool                     alive   = false; // running, or its load is in progress: keeps its claims
    bool                     loading = false; // its load is in progress (claims may still change)
    bool                     idle    = false; // up, nothing in flight, not stopping
    bool                     pinned  = false;
    bool                     held    = false;
};

// Residents to unload so queued sessions get their resources: idle, not pinned, not held, and
// owning a claim (`claims_by_owner`) on a contested resource. Returns stop names, deduplicated.
std::vector<std::string> router_board_pick_yield(const std::vector<router_board_resident> & residents,
                                                 const std::map<std::string, std::set<std::string>> & claims_by_owner,
                                                 const std::set<std::string> & contested);

//
// HTTP client
//

enum router_board_rc {
    ROUTER_BOARD_OK     = 0,
    ROUTER_BOARD_GONE   = 1, // 404: the claim is not live (expired / released)
    ROUTER_BOARD_FAILED = 2, // no answer, auth error, server error
};

class router_board_client {
  public:
    // url: e.g. http://mad-lab-2026.tail322e50.ts.net:18800 (a path prefix is kept)
    router_board_client(const std::string & url, const std::string & token, int timeout_ms = 3000);

    bool list_claims(const std::string & machine, std::vector<router_board_claim> & out, std::string & err) const;
    bool list_queue(const std::string & machine, std::vector<router_board_queue_entry> & out, std::string & err) const;

    router_board_claim_result claim(const std::string & machine, const std::string & resource, const std::string & note,
                                    admission_priority priority, int ttl_hours) const;
    router_board_rc release(const std::string & claim_id, std::string & err) const;
    router_board_rc renew(const std::string & claim_id, int ttl_hours, std::string & err) const;
    router_board_rc leave_queue(const std::string & machine, const std::string & resource, std::string & err) const;
    router_board_rc notify(const std::string & claim_id, admission_notify_kind kind, const std::string & content,
                           std::string & err) const;

  private:
    std::string base;   // scheme://host:port
    std::string prefix; // path prefix, no trailing '/'
    std::string token;
    int         timeout_ms;
};

//
// agent
//

struct router_board_config {
    std::string url;
    std::string token;
    std::string machine;
    int         poll_ms    = 3000;
    int64_t     renew_ms   = 10LL * 60 * 1000;
    int         ttl_hours  = 1;
    int         timeout_ms = 3000;
};

// How the agent reaches the router. Called from the agent's thread (and from release_dead()'s
// caller) with no agent lock held; each takes the router's own lock as it needs.
struct router_board_host {
    std::function<std::vector<router_board_resident>()>                       residents;
    std::function<void(const std::string & name, const std::string & reason)> yield;   // unload this resident
    std::function<void()>                                                      changed; // board state changed: re-evaluate queued loads
};

class router_board_agent {
  public:
    router_board_agent(router_board_config cfg, router_board_host host);
    ~router_board_agent();

    // Releases what a previous router generation left on this machine (claims held by
    // `llama-router`, its queue entries), then starts the poll thread (tests drive tick()).
    void start(bool run_thread = true);
    // Joins the poll thread, releases every claim and leaves every queue it joined. Idempotent.
    void stop();

    const std::string & machine() const { return cfg.machine; }

    // cache (no HTTP)
    bool                         available() const;
    router_board_snapshot        snapshot() const;
    std::vector<admission_claim> admission_claims(const std::vector<std::string> & machine_resources) const;
    // of `resources`, those `owner` holds no claim on
    std::vector<std::string>     missing(const std::string & owner, const std::vector<std::string> & resources) const;
    std::map<std::string, std::set<std::string>> held() const; // owner -> resources

    // HTTP: never call these with a router lock held.
    //
    // Claim `resource` for `owner`. Granted -> recorded. Queued -> recorded as joined by
    // `queue_owner` (the load that waits; left with leave_queue()).
    router_board_claim_result acquire(const std::string & owner, const std::string & queue_owner,
                                      const std::string & resource, const std::string & note, admission_priority priority);
    // Tells each holder once per (claim, kind); never a `llama-router` claim.
    void notify(const std::vector<admission_notify> & list, const std::string & content);
    // `queue_owner` stopped waiting: leaves every queue only it was waiting in.
    void leave_queue(const std::string & queue_owner);
    // Releases claims whose owner is not alive (per host.residents()) and those a settled
    // resident no longer needs.
    void release_dead();
    // Wakes the poll thread now (a resident went away).
    void wake();
    // One pass: poll, release, renew, re-take, probation cleanup, yield. The thread body.
    void tick();

  private:
    struct held_claim {
        std::string owner;
        std::string resource;
        std::string claim_id;
        int64_t     renewed_ms = 0;
        int64_t     taken_ms   = 0;
    };

    void release_claims(std::vector<held_claim> victims);
    void release_dead_from(const std::vector<router_board_resident> & residents, int64_t since_ms);
    void startup_sweep();

    router_board_config cfg;
    router_board_host   host;
    router_board_client client;

    mutable std::mutex                           mu; // guards everything below; never held across HTTP or a callback
    router_board_snapshot                        cache;
    std::string                                  cache_sig;
    std::vector<held_claim>                      claims;
    std::map<std::string, std::set<std::string>> joined;   // resource -> queue owners waiting for it
    std::map<std::string, int>                   pending;  // resource -> claims in flight
    std::set<std::string>                        notified; // "claim_id:kind"
    std::map<std::string, int64_t>               retake_after; // "owner|resource" -> earliest re-claim (ms)
    bool                                         stopped = false;

    std::thread             th;
    std::mutex              wake_mu;
    std::condition_variable wake_cv;
    bool                    wake_flag = false;
    std::atomic<bool>       quit{false};
};
