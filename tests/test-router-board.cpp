// Tests for the router as a board citizen (tools/server/server-router-board.cpp): request
// priority parsing, queue-verdict mapping, the 503 a queued request gets, board JSON parsing,
// claim expansion and yield selection as literals; then the board agent against a tiny
// in-process fake of the board's REST routes (same JSON shapes as mneme's board.py behind
// mad-lab-mcp): claim on load, renew, release on unload, queueing + notify, probation confirm,
// yield on queue, outage and auth. Also hold leases (server-router-holds.cpp) and the idle-unload
// rule (server-router-policy.cpp).

#undef NDEBUG

#include "server-router-admission.h"
#include "server-router-board.h"
#include "server-router-holds.h"
#include "server-router-policy.h"

#include "json.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using json = common_json;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "FAIL line %d: %s\n", line, what);
        abort();
    }
}
#define CHECK(x) check((x), #x, __LINE__)

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

//
// fake board: GET/POST /board/claims, DELETE/PATCH /board/claims/{id}, GET /board/queue,
// POST /board/queue/leave, POST /board/notify. Write routes want the bearer token.
//

struct fake_board {
    struct claim_row {
        std::string id, machine, resource, holder, note, priority;
        int         ttl_hours = 8;
        bool        probation = false;
        bool        released  = false;
    };
    struct queue_row {
        std::string id, machine, resource, session_id, note, priority, status = "waiting";
        long        seq = 0;
    };

    std::mutex             mu;
    std::vector<claim_row> claims;
    std::vector<queue_row> queue;
    int                    n_patch = 0;
    int                    n_leave = 0;
    std::vector<std::pair<std::string, std::string>> notifies; // claim_id, kind
    std::string            token = "test-token";
    long                   next  = 1;

    httplib::Server svr;
    std::thread     th;
    int             port = 0;

    static int rank(const std::string & p) { return p == "highest" ? 0 : p == "lowest" ? 2 : 1; }
    static bool conflict(const std::string & a, const std::string & b) { return a == b || a == "machine" || b == "machine"; }

    std::string url() const { return "http://127.0.0.1:" + std::to_string(port); }

    static json claim_json(const claim_row & c) {
        json o = json::object();
        o["id"]                   = c.id;
        o["machine"]              = c.machine;
        o["resource"]             = c.resource;
        o["holder"]               = c.holder;
        o["note"]                 = c.note;
        o["eta"]                  = "";
        o["vram_estimate_mb"]     = nullptr;
        o["brick_risk"]           = false;
        o["created_at"]           = "2026-10-03 12:00:00";
        o["ttl_hours"]            = c.ttl_hours;
        o["ttl_anchor"]           = "2026-10-03 12:00:00";
        o["released_at"]          = nullptr;
        o["vram_alert_pct"]       = nullptr;
        o["ram_alert_pct"]        = nullptr;
        o["probation"]            = c.probation;
        o["probation_expires_at"] = c.probation ? json("2026-10-03 12:05:00") : json(nullptr);
        o["test_slot"]            = nullptr;
        o["test_id"]              = nullptr;
        o["priority"]             = c.priority;
        return o;
    }

    static json queue_json(const queue_row & q) {
        json o = json::object();
        o["id"]          = q.id;
        o["machine"]     = q.machine;
        o["resource"]    = q.resource;
        o["session_id"]  = q.session_id;
        o["agent_id"]    = "";
        o["note"]        = q.note;
        o["joined_at"]   = "2026-10-03 12:00:00";
        o["slot_hours"]  = 4;
        o["status"]      = q.status;
        o["resolved_at"] = nullptr;
        o["notified_at"] = nullptr;
        o["priority"]    = q.priority;
        return o;
    }

    std::vector<queue_row *> waiting(const std::string & machine, const std::string & resource) {
        std::vector<queue_row *> out;
        for (auto & q : queue) {
            if (q.status == "waiting" && q.machine == machine && (resource.empty() || q.resource == resource)) {
                out.push_back(&q);
            }
        }
        std::stable_sort(out.begin(), out.end(), [](const queue_row * a, const queue_row * b) {
            return rank(a->priority) != rank(b->priority) ? rank(a->priority) < rank(b->priority) : a->seq < b->seq;
        });
        return out;
    }

    // board.claim(): confirm our own probation claim, join the queue behind a conflict
    // (idempotent per holder), or grant. Caller holds mu.
    json do_claim(const std::string & machine, const std::string & resource, const std::string & holder,
                  const std::string & note, const std::string & priority, int ttl, bool probation = false) {
        for (auto & c : claims) {
            if (!c.released && c.machine == machine && c.resource == resource && c.holder == holder && c.probation && !probation) {
                c.probation = false;
                c.ttl_hours = ttl;
                return claim_json(c);
            }
        }
        const claim_row * blocker = nullptr;
        for (const auto & c : claims) {
            if (!c.released && c.machine == machine && c.holder != holder && conflict(resource, c.resource)) {
                blocker = &c;
                break;
            }
        }
        if (blocker != nullptr) {
            queue_row * slot = nullptr;
            for (auto & q : queue) {
                if (q.status == "waiting" && q.machine == machine && q.resource == resource && q.session_id == holder) {
                    slot = &q;
                }
            }
            if (slot == nullptr) {
                queue.push_back({ "q" + std::to_string(next), machine, resource, holder, note, priority, "waiting", next });
                next++;
                slot = &queue.back();
            }
            const std::string qid = slot->id;
            const auto w = waiting(machine, resource);
            int pos = (int) w.size();
            for (size_t i = 0; i < w.size(); ++i) {
                if (w[i]->id == qid) {
                    pos = (int) i + 1;
                }
            }
            json o = json::object();
            o["queued"]     = true;
            o["position"]   = pos;
            o["held_by"]    = blocker->holder;
            o["held_since"] = "2026-10-03 12:00:00";
            o["machine"]    = machine;
            o["resource"]   = resource;
            o["queue_id"]   = qid;
            o["message"]    = resource + " is held by " + blocker->holder;
            return o;
        }
        claim_row c;
        c.id        = "c" + std::to_string(next++);
        c.machine   = machine;
        c.resource  = resource;
        c.holder    = holder;
        c.note      = note;
        c.priority  = priority;
        c.ttl_hours = ttl;
        c.probation = probation;
        claims.push_back(c);
        return claim_json(c);
    }

    // _pop_compatible_queues(): hand freed resources to the queue head as a probation claim
    void pop(const std::string & machine) {
        for (queue_row * q : waiting(machine, "")) {
            bool blocked = false;
            for (const auto & c : claims) {
                if (!c.released && c.machine == machine && conflict(q->resource, c.resource)) {
                    blocked = true;
                }
            }
            if (blocked) {
                break;
            }
            const std::string session = q->session_id, resource = q->resource, note = q->note, prio = q->priority;
            q->status = "popped";
            do_claim(machine, resource, session, note, prio, 8, true);
        }
    }

    // test helpers
    std::string session_claim(const std::string & holder, const std::string & resource) {
        std::lock_guard<std::mutex> lk(mu);
        json r = do_claim("m1", resource, holder, "session work", "middle", 8);
        return r.contains("id") ? r.at("id").get<std::string>() : "";
    }
    bool session_queue(const std::string & holder, const std::string & resource) {
        std::lock_guard<std::mutex> lk(mu);
        json r = do_claim("m1", resource, holder, "session work", "middle", 8);
        return r.contains("queued");
    }
    void session_release(const std::string & holder) {
        std::lock_guard<std::mutex> lk(mu);
        for (auto & c : claims) {
            if (c.holder == holder) {
                c.released = true;
            }
        }
        pop("m1");
    }
    int live(const std::string & holder, const std::string & resource = "") {
        std::lock_guard<std::mutex> lk(mu);
        int n = 0;
        for (const auto & c : claims) {
            n += !c.released && c.holder == holder && (resource.empty() || c.resource == resource);
        }
        return n;
    }
    int waiting_of(const std::string & holder) {
        std::lock_guard<std::mutex> lk(mu);
        int n = 0;
        for (const auto & q : queue) {
            n += q.status == "waiting" && q.session_id == holder;
        }
        return n;
    }
    const claim_row * find_live(const std::string & holder, const std::string & resource) {
        for (const auto & c : claims) {
            if (!c.released && c.holder == holder && c.resource == resource) {
                return &c;
            }
        }
        return nullptr;
    }

    bool authed(const httplib::Request & req) const {
        return req.get_header_value("Authorization") == "Bearer " + token;
    }
    static void reply(httplib::Response & res, int status, const json & body) {
        res.status = status;
        res.set_content(body.dump(), "application/json");
    }
    static json error_body(const std::string & msg) {
        json o = json::object();
        o["error"] = msg;
        return o;
    }

    void start() {
        svr.Get("/board/claims", [this](const httplib::Request & req, httplib::Response & res) {
            std::lock_guard<std::mutex> lk(mu);
            const std::string machine = req.get_param_value("machine");
            json out = json::array();
            for (const auto & c : claims) {
                if (!c.released && (machine.empty() || c.machine == machine)) {
                    out.push_back(claim_json(c));
                }
            }
            reply(res, 200, out);
        });
        svr.Post("/board/claims", [this](const httplib::Request & req, httplib::Response & res) {
            if (!authed(req)) {
                return reply(res, 401, error_body("missing or invalid bearer token"));
            }
            std::lock_guard<std::mutex> lk(mu);
            const json b = json::parse(req.body);
            if (b.value("machine", "").empty() || b.value("resource", "").empty()) {
                return reply(res, 400, error_body("machine and resource are required"));
            }
            reply(res, 200, do_claim(b.value("machine", ""), b.value("resource", ""), b.value("holder", ""), b.value("note", ""),
                                     b.value("priority", "middle"), b.value("ttl_hours", 8)));
        });
        svr.Delete(R"(/board/claims/([^/]+))", [this](const httplib::Request & req, httplib::Response & res) {
            if (!authed(req)) {
                return reply(res, 401, error_body("missing or invalid bearer token"));
            }
            std::lock_guard<std::mutex> lk(mu);
            int n = 0;
            std::string machine;
            for (auto & c : claims) {
                if (c.id == req.matches[1].str() && !c.released) {
                    c.released = true;
                    machine    = c.machine;
                    n++;
                }
            }
            if (n > 0) {
                pop(machine);
            }
            json o = json::object();
            o["released"] = n;
            reply(res, 200, o);
        });
        svr.Patch(R"(/board/claims/([^/]+))", [this](const httplib::Request & req, httplib::Response & res) {
            if (!authed(req)) {
                return reply(res, 401, error_body("missing or invalid bearer token"));
            }
            std::lock_guard<std::mutex> lk(mu);
            const json b = json::parse(req.body);
            for (auto & c : claims) {
                if (c.id == req.matches[1].str() && !c.released) {
                    c.ttl_hours = b.value("ttl_hours", c.ttl_hours);
                    n_patch++;
                    return reply(res, 200, claim_json(c));
                }
            }
            reply(res, 404, error_body("no live claim " + req.matches[1].str()));
        });
        svr.Get("/board/queue", [this](const httplib::Request & req, httplib::Response & res) {
            std::lock_guard<std::mutex> lk(mu);
            json arr = json::array();
            for (const queue_row * q : waiting(req.get_param_value("machine"), req.get_param_value("resource"))) {
                arr.push_back(queue_json(*q));
            }
            json o = json::object();
            o["queue"] = arr;
            reply(res, 200, o);
        });
        svr.Post("/board/queue/leave", [this](const httplib::Request & req, httplib::Response & res) {
            if (!authed(req)) {
                return reply(res, 401, error_body("missing or invalid bearer token"));
            }
            std::lock_guard<std::mutex> lk(mu);
            const json b = json::parse(req.body);
            int n = 0;
            for (auto & q : queue) {
                if (q.status == "waiting" && q.machine == b.value("machine", "") && q.resource == b.value("resource", "") &&
                        q.session_id == b.value("holder", "")) {
                    q.status = "left";
                    n++;
                }
            }
            n_leave++;
            json o = json::object();
            o["left"] = n;
            reply(res, 200, o);
        });
        svr.Post("/board/notify", [this](const httplib::Request & req, httplib::Response & res) {
            if (!authed(req)) {
                return reply(res, 401, error_body("missing or invalid bearer token"));
            }
            std::lock_guard<std::mutex> lk(mu);
            const json b = json::parse(req.body);
            const std::string id = b.value("claim_id", "");
            const std::string kind = b.value("kind", "");
            if (kind != "wait" && kind != "yield") {
                return reply(res, 400, error_body("kind must be one of ('wait', 'yield')"));
            }
            for (const auto & c : claims) {
                if (c.id == id && !c.released) {
                    notifies.push_back({ id, kind });
                    json o = json::object();
                    o["enqueued"] = true;
                    return reply(res, 200, o);
                }
            }
            reply(res, 404, error_body("no live claim " + id));
        });
        port = svr.bind_to_any_port("127.0.0.1");
        CHECK(port > 0);
        th = std::thread([this]() { svr.listen_after_bind(); });
        svr.wait_until_ready();
    }

    void stop() {
        svr.stop();
        if (th.joinable()) {
            th.join();
        }
    }
};

//
// pure pieces
//

static void test_request_opts() {
    std::string err;
    router_request_opts o;
    CHECK(router_parse_request_opts("", "", "", "", o, err));
    CHECK(!o.priority_set && o.priority == ADMISSION_PRIORITY_MIDDLE && o.machine.empty());

    o = {};
    CHECK(router_parse_request_opts("lowest", "highest", "", "mad-lab-main", o, err)); // body wins
    CHECK(o.priority_set && o.priority == ADMISSION_PRIORITY_LOWEST && o.machine == "mad-lab-main");

    o = {};
    CHECK(router_parse_request_opts("", " Highest ", "mad-lab-2026", "other", o, err)); // header; body machine wins
    CHECK(o.priority == ADMISSION_PRIORITY_HIGHEST && o.machine == "mad-lab-2026");

    o = {};
    CHECK(!router_parse_request_opts("urgent", "", "", "", o, err));
    CHECK(err.find("urgent") != std::string::npos);
}

static void test_queue_mapping() {
    auto with = [](admission_block b) {
        admission_result r;
        r.verdict = ADMISSION_QUEUE;
        r.blocked = b;
        return router_admission_queue_action(r);
    };
    // only a foreign claim or a busy resident is waited out
    CHECK(with(ADMISSION_BLOCK_CLAIM) == ROUTER_QUEUE_WAIT);
    CHECK(with(ADMISSION_BLOCK_BUSY) == ROUTER_QUEUE_WAIT);
    CHECK(with(ADMISSION_BLOCK_PINNED) == ROUTER_QUEUE_REFUSE);
    CHECK(with(ADMISSION_BLOCK_HELD) == ROUTER_QUEUE_REFUSE);
    CHECK(with(ADMISSION_BLOCK_CAPACITY) == ROUTER_QUEUE_REFUSE);
    CHECK(with(ADMISSION_BLOCK_NO_CANDIDATE) == ROUTER_QUEUE_REFUSE);
}

// /v1 at lowest blocks until ready; at middle / highest a queued model answers 503 + Retry-After + queue info
static void test_v1_priority() {
    CHECK(router_request_waits_in_queue(ADMISSION_PRIORITY_LOWEST));
    CHECK(!router_request_waits_in_queue(ADMISSION_PRIORITY_MIDDLE));
    CHECK(!router_request_waits_in_queue(ADMISSION_PRIORITY_HIGHEST));

    router_queued_info info;
    info.model      = "qwen-r9700";
    info.priority   = ADMISSION_PRIORITY_MIDDLE;
    info.board      = true;
    info.queue_pos  = 2;
    info.blocked_by = "session-abc";
    info.blocked_on = "gpu:R9700";
    int status = 0;
    std::string body;
    std::map<std::string, std::string> headers;
    router_queued_response(info, status, body, headers);
    CHECK(status == 503);
    CHECK(headers.count("Retry-After") && std::stoi(headers["Retry-After"]) > 0);
    const json j = json::parse(body);
    CHECK(j.at("error").at("code").get<int>() == 503);
    const json & q = j.at("error").at("queue");
    CHECK(q.at("state").get<std::string>() == "queued");
    CHECK(q.at("queue_pos").get<int>() == 2);
    CHECK(q.at("blocked_by").get<std::string>() == "session-abc");
    CHECK(q.at("blocked_on").get<std::string>() == "gpu:R9700");

    const router_queued_error e(info);
    CHECK(std::string(e.what()).find("queued") != std::string::npos);
    CHECK(json::parse(router_queued_info_json(info)).at("priority").get<std::string>() == "middle");
}

static void test_holds() {
    router_holds h;
    std::string err;
    const std::string l = h.hold("qwen", 10000, "orchestrator", "", 1000, err);
    CHECK(!l.empty());
    CHECK(h.is_held("qwen", 5000));
    CHECK(!h.is_held("other", 5000));
    CHECK(!h.is_held("qwen", 11000)); // expired at 11000

    // renew (re-hold with the lease) before it expires
    const std::string l2 = h.hold("qwen", 10000, "orchestrator", "", 1000, err);
    CHECK(h.hold("qwen", 10000, "", l2, 9000, err) == l2);
    CHECK(h.is_held("qwen", 15000));
    CHECK(h.hold("qwen-other", 1000, "", l2, 9500, err).empty()); // a lease holds one model
    CHECK(h.hold("qwen", 0, "", "", 9500, err).empty());          // ttl must be positive

    // lease expiry: dropped silently, renewing it fails, the model is no longer held
    CHECK(h.hold("qwen", 1000, "", l2, 30000, err).empty());
    CHECK(err.find("unknown or expired") != std::string::npos);
    CHECK(h.list(30000).empty());
    CHECK(!h.is_held("qwen", 30000));

    // release
    const std::string l3 = h.hold("qwen", 60000, "o", "", 30000, err);
    CHECK(h.release(l3, 30001));
    CHECK(!h.is_held("qwen", 30002));
    CHECK(!h.release(l3, 30002));

    const std::string l4 = h.hold("m", 1000, "o", "", 0, err);
    CHECK(h.list(500).size() == 1);
    h.prune(2000);
    CHECK(h.list(2000).empty());
    CHECK(!h.release(l4, 2000));
}

// a hold blocks the idle sweeper and eviction
static void test_hold_blocks_idle_and_eviction() {
    idle_resident r;
    r.last_used = 1000;
    r.timeout_s = 60;
    CHECK(idle_unload_due(r, 1000 + 61000));
    CHECK(!idle_unload_due(r, 1000 + 30000)); // not idle long enough
    idle_resident held = r;
    held.held = true;
    CHECK(!idle_unload_due(held, 1000 + 61000));
    idle_resident pinned = r;
    pinned.pinned = true;
    CHECK(!idle_unload_due(pinned, 1000 + 61000));
    idle_resident busy = r;
    busy.req_count = 1;
    CHECK(!idle_unload_due(busy, 1000 + 61000));
    idle_resident never = r;
    never.timeout_s = 0;
    CHECK(!idle_unload_due(never, 1000 + 61000));
    idle_resident unused = r;
    unused.last_used = 0;
    CHECK(!idle_unload_due(unused, 1000 + 61000));

    const int64_t GB = 1024LL * 1024LL * 1024LL;
    admission_input in;
    in.alias      = "new";
    in.candidates = { admission_candidate{ "", { { "ROCm0", 20 * GB } }, {} } };
    in.slots      = { { "ROCm0", "", "gpu:R9700", 4 * GB } };
    admission_resident old;
    old.name      = "old";
    old.vram      = { { "ROCm0", 28 * GB } };
    old.last_used = 5;
    in.residents  = { old };
    admission_result res = decide_admission(in);
    CHECK(res.verdict == ADMISSION_EVICT_THEN_ADMIT && res.victims == std::vector<std::string>{ "old" });
    CHECK(res.board_claims_to_take == std::vector<std::string>{ "gpu:R9700" });

    in.residents[0].held = true; // under a hold lease
    res = decide_admission(in);
    CHECK(res.verdict == ADMISSION_QUEUE && res.blocked == ADMISSION_BLOCK_HELD && res.blocked_by == "old");
    CHECK(router_admission_queue_action(res) == ROUTER_QUEUE_REFUSE);
}

static void test_config_helpers() {
    std::string dev, board;
    CHECK(router_gpu_name_split("ROCm0=R9700", dev, board) && dev == "ROCm0" && board == "R9700");
    CHECK(router_gpu_name_split("ROCm1", dev, board) && dev == "ROCm1" && board.empty());
    CHECK(!router_gpu_name_split("ROCm0=", dev, board));
    CHECK(!router_gpu_name_split("=R9700", dev, board));

    CHECK(router_parse_local_machine(R"({"mad-lab-2026": {"ssh": "x"}, "mad-lab-main": {"local": true}})") == "mad-lab-main");
    CHECK(router_parse_local_machine(R"({"mad-lab-2026": {"local": false}})").empty());
    CHECK(router_parse_local_machine("not json").empty());
    CHECK(!router_local_machine("/nonexistent/machines.json").empty()); // falls back to the hostname
}

static void test_parse() {
    std::vector<router_board_claim> cl;
    CHECK(router_board_parse_claims(R"([
        {"id": "c1", "machine": "m1", "resource": "gpu:R9700", "holder": "sess-1", "note": "n", "eta": "",
         "vram_estimate_mb": null, "brick_risk": false, "created_at": "2026-10-03 12:00:00", "ttl_hours": 8,
         "ttl_anchor": "2026-10-03 12:00:00", "released_at": null, "vram_alert_pct": null, "ram_alert_pct": null,
         "probation": false, "probation_expires_at": null, "test_slot": null, "test_id": null, "priority": "highest"},
        {"id": "c2", "machine": "m1", "resource": "ram", "holder": "llama-router", "probation": true, "priority": null}
    ])", cl));
    CHECK(cl.size() == 2);
    CHECK(cl[0].id == "c1" && cl[0].holder == "sess-1" && cl[0].priority == ADMISSION_PRIORITY_HIGHEST && !cl[0].probation);
    CHECK(cl[1].probation && cl[1].priority == ADMISSION_PRIORITY_MIDDLE); // legacy null = middle
    CHECK(!router_board_parse_claims(R"({"error": "x"})", cl));

    std::vector<router_board_queue_entry> q;
    CHECK(router_board_parse_queue(R"({"queue": [
        {"id": "q1", "machine": "m1", "resource": "gpu:R9700", "session_id": "sess-2", "agent_id": "", "note": "",
         "joined_at": "2026-10-03 12:00:00", "slot_hours": 4, "status": "waiting", "resolved_at": null,
         "notified_at": null, "priority": "lowest"}]})", q));
    CHECK(q.size() == 1 && q[0].holder == "sess-2" && q[0].resource == "gpu:R9700" && q[0].priority == ADMISSION_PRIORITY_LOWEST);

    auto granted = router_board_parse_claim_response(200, R"({"id": "c9", "machine": "m1", "resource": "gpu:R9700", "holder": "llama-router", "probation": false})");
    CHECK(granted.ok && granted.granted && granted.claim_id == "c9" && !granted.queued);
    auto queued = router_board_parse_claim_response(200, R"({"queued": true, "position": 2, "held_by": "sess-1",
        "held_since": "2026-10-03 12:00:00", "machine": "m1", "resource": "gpu:R9700", "queue_id": "q7", "message": "..."})");
    CHECK(queued.ok && queued.queued && queued.position == 2 && queued.held_by == "sess-1" && queued.queue_id == "q7");
    auto denied = router_board_parse_claim_response(401, R"({"error": "missing or invalid bearer token"})");
    CHECK(!denied.ok && denied.status == 401 && denied.error.find("bearer") != std::string::npos);
    auto bad = router_board_parse_claim_response(422, R"({"detail": [{"msg": "bad priority"}]})");
    CHECK(!bad.ok);
}

static void test_claims_and_yield_selection() {
    router_board_snapshot snap;
    snap.ok     = true;
    snap.claims = {
        { "c1", "m1", "machine", "sess-1", "", ADMISSION_PRIORITY_MIDDLE, false },  // whole machine
        { "c2", "m1", "gpu:R9700", "llama-router", "qwen", ADMISSION_PRIORITY_MIDDLE, false },
    };
    const auto ac = router_board_admission_claims(snap, { "gpu:R9700", "gpu:ROCm1", "ram" });
    CHECK(ac.size() == 4);
    CHECK(std::count_if(ac.begin(), ac.end(), [](const admission_claim & c) { return c.claim_id == "c1"; }) == 3);
    CHECK(std::any_of(ac.begin(), ac.end(), [](const admission_claim & c) { return c.claim_id == "c2" && c.is_router; }));
    CHECK(std::none_of(ac.begin(), ac.end(), [](const admission_claim & c) { return c.claim_id == "c1" && c.is_router; }));
    snap.ok = false;
    CHECK(router_board_admission_claims(snap, { "gpu:R9700" }).empty()); // outage: nothing blocks
    snap.ok = true;

    snap.queue = { { "q1", "m1", "gpu:R9700", "sess-2", ADMISSION_PRIORITY_MIDDLE },
                   { "q2", "m1", "gpu:ROCm1", "llama-router", ADMISSION_PRIORITY_MIDDLE } }; // our own entry contests nothing
    const std::set<std::string> held = { "gpu:R9700", "gpu:ROCm1" };
    CHECK(router_board_contested(snap, held) == std::set<std::string>{ "gpu:R9700" });
    router_board_snapshot whole = snap;
    whole.queue = { { "q3", "m1", "machine", "sess-3", ADMISSION_PRIORITY_MIDDLE } };
    CHECK(router_board_contested(whole, held) == held);

    std::vector<router_board_resident> rs(5);
    rs[0] = { "idle", "", {}, true, false, true, false, false };
    rs[1] = { "held", "", {}, true, false, true, false, true };
    rs[2] = { "pinned", "", {}, true, false, true, true, false };
    rs[3] = { "busy", "", {}, true, false, false, false, false };
    rs[4] = { "worker", "spine", {}, true, false, true, false, false };
    const std::map<std::string, std::set<std::string>> by_owner = {
        { "idle", { "gpu:R9700" } }, { "held", { "gpu:R9700" } }, { "pinned", { "gpu:R9700" } },
        { "busy", { "gpu:R9700" } }, { "worker", { "gpu:R9700" } },
    };
    const auto y = router_board_pick_yield(rs, by_owner, { "gpu:R9700" });
    CHECK((y == std::vector<std::string>{ "idle", "spine" })); // a worker yields through its spine
    CHECK(router_board_pick_yield(rs, by_owner, { "gpu:ROCm1" }).empty());
}

//
// the agent against the fake board
//

struct fake_host {
    std::mutex                         mu;
    std::vector<router_board_resident> residents;
    std::vector<std::string>           yielded;
    std::atomic<int>                   changed{0};

    router_board_host host() {
        router_board_host h;
        h.residents = [this]() {
            std::lock_guard<std::mutex> lk(mu);
            return residents;
        };
        h.yield = [this](const std::string & name, const std::string &) {
            std::lock_guard<std::mutex> lk(mu);
            yielded.push_back(name);
        };
        h.changed = [this]() { changed++; };
        return h;
    }
    void set(std::vector<router_board_resident> rs) {
        std::lock_guard<std::mutex> lk(mu);
        residents = std::move(rs);
    }
};

static router_board_resident running(const std::string & name, std::vector<std::string> res) {
    router_board_resident r;
    r.name      = name;
    r.resources = std::move(res);
    r.alive     = true;
    r.idle      = true;
    return r;
}

static void test_agent() {
    fake_board fb;
    fb.start();
    {
        std::lock_guard<std::mutex> lk(fb.mu);
        fb.do_claim("m1", "gpu:ROCm9", "llama-router", "left by a crashed router", "middle", 1); // stale
    }

    fake_host fh;
    router_board_config cfg;
    cfg.url        = fb.url();
    cfg.token      = "test-token";
    cfg.machine    = "m1";
    cfg.renew_ms   = 0; // renew on every tick
    cfg.timeout_ms = 2000;
    router_board_agent agent(cfg, fh.host());
    agent.start(false); // startup sweep only; the test drives tick()
    CHECK(fb.live("llama-router") == 0); // what a previous router generation left is released

    // claim taken on load: holder llama-router, note = the model, ttl 1 h, the load's priority
    auto r = agent.acquire("qwen", "qwen", "gpu:R9700", "qwen", ADMISSION_PRIORITY_HIGHEST);
    CHECK(r.ok && r.granted && !r.claim_id.empty());
    {
        std::lock_guard<std::mutex> lk(fb.mu);
        const auto * c = fb.find_live("llama-router", "gpu:R9700");
        CHECK(c != nullptr && c->note == "qwen" && c->ttl_hours == 1 && c->priority == "highest" && c->id == r.claim_id);
    }
    CHECK(agent.missing("qwen", { "gpu:R9700", "ram" }) == std::vector<std::string>{ "ram" });

    // while resident: kept and renewed
    fh.set({ running("qwen", { "gpu:R9700" }) });
    sleep_ms(5);
    agent.tick();
    CHECK(agent.available());
    CHECK(fb.live("llama-router", "gpu:R9700") == 1);
    CHECK(fb.n_patch >= 1);

    // released on unload
    fh.set({});
    agent.tick();
    CHECK(fb.live("llama-router") == 0);
    CHECK(agent.held().empty());

    // a session holds gpu:ROCm1: admission queues the load on it and notifies the holder
    const std::string sess_claim = fb.session_claim("session-1", "gpu:ROCm1");
    CHECK(!sess_claim.empty());
    agent.tick();
    admission_input in;
    in.alias      = "llama-8b";
    in.priority   = ADMISSION_PRIORITY_MIDDLE;
    in.candidates = { admission_candidate{ "", { { "ROCm1", 1 } }, {} } };
    in.slots      = { { "ROCm1", "", "gpu:ROCm1", 1LL << 40 } };
    in.claims     = agent.admission_claims({ "gpu:R9700", "gpu:ROCm1", "ram" });
    const admission_result res = decide_admission(in);
    CHECK(res.verdict == ADMISSION_QUEUE && res.blocked == ADMISSION_BLOCK_CLAIM && res.blocked_by == "session-1");
    CHECK(router_admission_queue_action(res) == ROUTER_QUEUE_WAIT);
    CHECK(res.notify.size() == 1 && res.notify[0].claim_id == sess_claim && res.notify[0].kind == ADMISSION_NOTIFY_WAIT);

    // queued state: the router joins the board queue and learns its place
    auto q = agent.acquire("llama-8b", "llama-8b", "gpu:ROCm1", "llama-8b", ADMISSION_PRIORITY_MIDDLE);
    CHECK(q.ok && q.queued && !q.granted && q.position == 1 && q.held_by == "session-1");
    CHECK(fb.waiting_of("llama-router") == 1);
    agent.notify(res.notify, "llama-router needs gpu:ROCm1");
    agent.notify(res.notify, "llama-router needs gpu:ROCm1"); // once per (claim, kind)
    {
        std::lock_guard<std::mutex> lk(fb.mu);
        CHECK(fb.notifies.size() == 1 && fb.notifies[0].first == sess_claim && fb.notifies[0].second == "wait");
    }
    // never notify the router's own claims
    agent.notify({ { "c-own", ROUTER_BOARD_HOLDER, ADMISSION_NOTIFY_YIELD } }, "x");
    {
        std::lock_guard<std::mutex> lk(fb.mu);
        CHECK(fb.notifies.size() == 1);
    }

    // the session releases: the board hands the router a probation claim; the agent reports the
    // change, keeps the turn for the waiting load, and claiming confirms it in place
    const int changed_before = fh.changed.load();
    fb.session_release("session-1");
    agent.tick();
    CHECK(fh.changed.load() > changed_before);
    std::string turn;
    {
        std::lock_guard<std::mutex> lk(fb.mu);
        const auto * c = fb.find_live("llama-router", "gpu:ROCm1");
        CHECK(c != nullptr && c->probation);
        turn = c->id;
    }
    auto g = agent.acquire("llama-8b", "llama-8b", "gpu:ROCm1", "llama-8b", ADMISSION_PRIORITY_MIDDLE);
    CHECK(g.ok && g.granted && g.claim_id == turn);
    {
        std::lock_guard<std::mutex> lk(fb.mu);
        CHECK(!fb.find_live("llama-router", "gpu:ROCm1")->probation);
    }
    agent.leave_queue("llama-8b"); // nothing left to leave
    fh.set({ running("llama-8b", { "gpu:ROCm1" }) });
    sleep_ms(5);
    agent.tick();
    CHECK(fb.live("llama-router", "gpu:ROCm1") == 1);

    // yield on queue: a session queues for gpu:R9700, which four residents hold; only the idle,
    // unpinned, unheld one goes
    for (const char * n : { "y-idle", "y-held", "y-pinned", "y-busy" }) {
        CHECK(agent.acquire(n, "", "gpu:R9700", n, ADMISSION_PRIORITY_MIDDLE).granted);
    }
    CHECK(fb.session_queue("session-2", "gpu:R9700"));
    router_board_resident y_held   = running("y-held", { "gpu:R9700" });
    y_held.held                    = true;
    router_board_resident y_pinned = running("y-pinned", { "gpu:R9700" });
    y_pinned.pinned                = true;
    router_board_resident y_busy   = running("y-busy", { "gpu:R9700" });
    y_busy.idle                    = false;
    fh.set({ running("llama-8b", { "gpu:ROCm1" }), running("y-idle", { "gpu:R9700" }), y_held, y_pinned, y_busy });
    sleep_ms(5);
    agent.tick();
    {
        std::lock_guard<std::mutex> lk(fh.mu);
        CHECK(fh.yielded == std::vector<std::string>{ "y-idle" });
    }
    // y-idle unloads: its claim goes; the held / pinned / busy ones stay
    fh.set({ running("llama-8b", { "gpu:ROCm1" }), y_held, y_pinned, y_busy });
    agent.tick();
    CHECK(fb.live("llama-router", "gpu:R9700") == 3);
    {
        std::lock_guard<std::mutex> lk(fh.mu);
        CHECK(fh.yielded.size() == 1);
    }

    // a settled resident that lost its claim (board outage at load, expiry) gets it again
    fh.set({ running("llama-8b", { "gpu:ROCm1", "ram" }), y_held, y_pinned, y_busy });
    agent.tick();
    CHECK(fb.live("llama-router", "ram") == 1);

    // a wrong token: claims are refused, the load goes ahead without one
    {
        fake_host fh2;
        router_board_config c2 = cfg;
        c2.token = "wrong";
        router_board_agent bad(c2, fh2.host());
        auto br = bad.acquire("x", "x", "gpu:ROCm7", "x", ADMISSION_PRIORITY_MIDDLE);
        CHECK(!br.ok && br.status == 401);
        bad.stop();
    }

    // shutdown releases every claim and leaves every queue
    agent.stop();
    CHECK(fb.live("llama-router") == 0);
    CHECK(fb.waiting_of("llama-router") == 0);
    fb.stop();
}

static void test_agent_outage() {
    fake_host fh;
    router_board_config cfg;
    cfg.url        = "http://127.0.0.1:1"; // nothing listens
    cfg.token      = "t";
    cfg.machine    = "m1";
    cfg.timeout_ms = 500;
    router_board_agent agent(cfg, fh.host());
    agent.start(false);
    fh.set({ running("qwen", { "gpu:R9700" }) });
    agent.tick();
    CHECK(!agent.available());
    CHECK(agent.admission_claims({ "gpu:R9700", "ram" }).empty()); // admit as if nothing were claimed
    auto r = agent.acquire("qwen", "qwen", "gpu:R9700", "qwen", ADMISSION_PRIORITY_MIDDLE);
    CHECK(!r.ok && !r.granted && !r.queued);
    agent.stop();
}

int main() {
    test_request_opts();
    test_queue_mapping();
    test_v1_priority();
    test_holds();
    test_hold_blocks_idle_and_eviction();
    test_config_helpers();
    test_parse();
    test_claims_and_yield_selection();
    test_agent();
    test_agent_outage();
    printf("test-router-board: OK\n");
    return 0;
}
