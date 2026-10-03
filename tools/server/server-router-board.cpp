#include "server-router-board.h"

#include "json.h"
#include "log.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>

#ifndef _WIN32
#include <unistd.h>
#endif

using json = common_json;

#define BRD_INF(fmt, ...) LOG_INF("board: " fmt, __VA_ARGS__)
#define BRD_WRN(fmt, ...) LOG_WRN("board: " fmt, __VA_ARGS__)

static int64_t board_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string trim_lower(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        b++;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    std::string out = s.substr(b, e - b);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return out;
}

static std::string trim(const std::string & s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && std::isspace((unsigned char) s[b])) {
        b++;
    }
    while (e > b && std::isspace((unsigned char) s[e - 1])) {
        e--;
    }
    return s.substr(b, e - b);
}

//
// request options
//

bool router_parse_request_opts(const std::string & body_priority, const std::string & header_priority,
                               const std::string & body_machine, const std::string & header_machine,
                               router_request_opts & out, std::string & err) {
    const std::string p = trim_lower(!trim(body_priority).empty() ? body_priority : header_priority);
    if (!p.empty()) {
        const auto v = admission_priority_parse(p);
        if (!v.has_value()) {
            err = "invalid priority '" + p + "': expected highest, middle or lowest";
            return false;
        }
        out.priority     = *v;
        out.priority_set = true;
    }
    const std::string m = trim(!trim(body_machine).empty() ? body_machine : header_machine);
    if (!m.empty()) {
        out.machine = m;
    }
    return true;
}

//
// queue verdicts
//

router_queue_action router_admission_queue_action(const admission_result & res) {
    if (res.verdict != ADMISSION_QUEUE) {
        return ROUTER_QUEUE_REFUSE;
    }
    switch (res.blocked) {
        case ADMISSION_BLOCK_CLAIM:
        case ADMISSION_BLOCK_BUSY:
            return ROUTER_QUEUE_WAIT;
        default:
            // capacity / no candidate never fit as configured; a pin or a hold is a human /
            // orchestrator decision the router does not wait out
            return ROUTER_QUEUE_REFUSE;
    }
}

bool router_request_waits_in_queue(admission_priority p) {
    return p == ADMISSION_PRIORITY_LOWEST;
}

std::vector<std::string> router_queue_service_order(const std::vector<router_queue_item> & items) {
    std::vector<router_queue_item> sorted = items;
    std::sort(sorted.begin(), sorted.end(), [](const router_queue_item & a, const router_queue_item & b) {
        if (a.priority != b.priority) {
            return a.priority > b.priority; // ADMISSION_PRIORITY_HIGHEST is the largest
        }
        return a.since_ms != b.since_ms ? a.since_ms < b.since_ms : a.name < b.name;
    });
    std::vector<std::string> out;
    for (const auto & i : sorted) {
        out.push_back(i.name);
    }
    return out;
}

int router_queue_position(const std::vector<router_queue_item> & items, const std::string & name) {
    const auto order = router_queue_service_order(items);
    auto it = std::find(order.begin(), order.end(), name);
    return it == order.end() ? 0 : (int) (it - order.begin()) + 1;
}

router_queue_wait router_queue_wait_decision(bool cancelled, bool client_gone, int64_t waited_ms, int64_t max_wait_ms) {
    if (cancelled) {
        return ROUTER_WAIT_CANCELLED;
    }
    if (client_gone) {
        return ROUTER_WAIT_ABORTED;
    }
    if (max_wait_ms > 0 && waited_ms >= max_wait_ms) {
        return ROUTER_WAIT_TIMED_OUT;
    }
    return ROUTER_WAIT_CONTINUE;
}

router_probation_action router_board_probation_action(bool ours, bool waited_for, bool claim_in_flight) {
    if (ours || claim_in_flight) {
        return ROUTER_PROBATION_KEEP;
    }
    return waited_for ? ROUTER_PROBATION_CONFIRM : ROUTER_PROBATION_RELEASE;
}

static std::string queued_message(const router_queued_info & info) {
    std::string msg = "model '" + info.model + "' is queued";
    if (info.queue_pos > 0) {
        msg += " (position " + std::to_string(info.queue_pos) + ")";
    }
    if (!info.blocked_by.empty()) {
        msg += info.board ? " behind board claim holder '" + info.blocked_by + "'" : " behind busy model '" + info.blocked_by + "'";
    }
    if (!info.blocked_on.empty()) {
        msg += " on " + info.blocked_on;
    }
    return msg;
}

router_queued_error::router_queued_error(router_queued_info i) : std::runtime_error(queued_message(i)), info(std::move(i)) {}

static json queued_info_obj(const router_queued_info & info) {
    json o = json::object();
    o["state"]      = "queued";
    o["queue_pos"]  = info.queue_pos;
    o["blocked_by"] = info.blocked_by;
    o["blocked_on"] = info.blocked_on;
    o["board"]      = info.board;
    o["priority"]   = admission_priority_str(info.priority);
    o["machine"]    = info.machine;
    o["reason"]     = info.reason;
    return o;
}

std::string router_queued_info_json(const router_queued_info & info) {
    return queued_info_obj(info).dump();
}

void router_queued_response(const router_queued_info & info, int & status, std::string & body,
                            std::map<std::string, std::string> & headers) {
    status = 503;
    headers["Retry-After"] = std::to_string(ROUTER_RETRY_AFTER_S);
    json err = json::object();
    err["code"]    = 503;
    err["type"]    = "unavailable_error";
    err["message"] = queued_message(info);
    err["queue"]   = queued_info_obj(info);
    json out = json::object();
    out["error"] = err;
    body = out.dump();
}

//
// configuration helpers
//

bool router_gpu_name_split(const std::string & field, std::string & dev, std::string & board_name) {
    const size_t eq = field.find('=');
    if (eq == std::string::npos) {
        dev        = trim(field);
        board_name = "";
        return !dev.empty();
    }
    dev        = trim(field.substr(0, eq));
    board_name = trim(field.substr(eq + 1));
    return !dev.empty() && !board_name.empty();
}

std::string router_parse_local_machine(const std::string & machines_json) {
    try {
        const json j = json::parse(machines_json);
        if (!j.is_object()) {
            return "";
        }
        for (const auto & [name, entry] : j.items()) {
            if (entry.is_object() && entry.contains("local") && entry.at("local").is_boolean() && entry.at("local").get<bool>()) {
                return name;
            }
        }
    } catch (const std::exception &) {
        return "";
    }
    return "";
}

std::string router_local_machine(const std::string & path_in) {
    std::string path = path_in;
    if (path.empty()) {
        const char * home = std::getenv("HOME");
        if (home != nullptr && home[0] != '\0') {
            path = std::string(home) + "/.config/mad-lab-agents/machines.json";
        }
    }
    if (!path.empty()) {
        std::ifstream f(path);
        if (f) {
            std::stringstream ss;
            ss << f.rdbuf();
            const std::string name = router_parse_local_machine(ss.str());
            if (!name.empty()) {
                return name;
            }
        }
    }
#ifndef _WIN32
    char buf[256] = {};
    if (gethostname(buf, sizeof(buf) - 1) == 0 && buf[0] != '\0') {
        std::string host = buf;
        return host.substr(0, host.find('.'));
    }
#endif
    return "local";
}

//
// board data
//

static std::string jstr(const json & o, const char * key) {
    if (!o.is_object() || !o.contains(key)) {
        return "";
    }
    const json & v = o.at(key);
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    return "";
}

static bool jbool(const json & o, const char * key) {
    if (!o.is_object() || !o.contains(key)) {
        return false;
    }
    const json & v = o.at(key);
    if (v.is_boolean()) {
        return v.get<bool>();
    }
    if (v.is_number_integer()) {
        return v.get<long long>() != 0;
    }
    return false;
}

static long long jint(const json & o, const char * key) {
    if (!o.is_object() || !o.contains(key)) {
        return 0;
    }
    const json & v = o.at(key);
    if (v.is_number()) {
        return v.get<long long>();
    }
    return 0;
}

static admission_priority jprio(const json & o) {
    const auto p = admission_priority_parse(jstr(o, "priority"));
    return p.has_value() ? *p : ADMISSION_PRIORITY_MIDDLE; // legacy rows carry none: middle
}

bool router_board_parse_claims(const std::string & body, std::vector<router_board_claim> & out) {
    out.clear();
    try {
        const json j = json::parse(body);
        if (!j.is_array()) {
            return false;
        }
        for (const auto & c : j) {
            if (!c.is_object()) {
                continue;
            }
            router_board_claim cl;
            cl.id        = jstr(c, "id");
            cl.machine   = jstr(c, "machine");
            cl.resource  = jstr(c, "resource");
            cl.holder    = jstr(c, "holder");
            cl.note      = jstr(c, "note");
            cl.priority  = jprio(c);
            cl.probation = jbool(c, "probation");
            if (!cl.id.empty() && !cl.resource.empty()) {
                out.push_back(std::move(cl));
            }
        }
    } catch (const std::exception &) {
        out.clear();
        return false;
    }
    return true;
}

bool router_board_parse_queue(const std::string & body, std::vector<router_board_queue_entry> & out) {
    out.clear();
    try {
        const json j = json::parse(body);
        if (!j.is_object() || !j.contains("queue") || !j.at("queue").is_array()) {
            return false;
        }
        for (const auto & e : j.at("queue")) {
            if (!e.is_object()) {
                continue;
            }
            const std::string status = jstr(e, "status");
            if (!status.empty() && status != "waiting") {
                continue;
            }
            router_board_queue_entry q;
            q.id       = jstr(e, "id");
            q.machine  = jstr(e, "machine");
            q.resource = jstr(e, "resource");
            q.holder   = jstr(e, "session_id");
            q.priority = jprio(e);
            if (!q.resource.empty()) {
                out.push_back(std::move(q));
            }
        }
    } catch (const std::exception &) {
        out.clear();
        return false;
    }
    return true;
}

router_board_claim_result router_board_parse_claim_response(int status, const std::string & body) {
    router_board_claim_result r;
    r.status = status;
    json j;
    try {
        j = json::parse(body);
    } catch (const std::exception &) {
        r.error = "unparsable board answer (HTTP " + std::to_string(status) + ")";
        return r;
    }
    if (status < 200 || status >= 300) {
        std::string msg = jstr(j, "error");
        if (msg.empty()) {
            msg = jstr(j, "detail");
        }
        r.error = "HTTP " + std::to_string(status) + (msg.empty() ? "" : ": " + msg);
        return r;
    }
    if (!j.is_object()) {
        r.error = "unexpected board answer";
        return r;
    }
    if (jbool(j, "queued")) {
        r.ok       = true;
        r.queued   = true;
        r.position = (int) jint(j, "position");
        r.held_by  = jstr(j, "held_by");
        r.queue_id = jstr(j, "queue_id");
        return r;
    }
    r.claim_id = jstr(j, "id");
    if (r.claim_id.empty()) {
        r.error = "board answer has neither a claim id nor a queue slot";
        return r;
    }
    r.ok      = true;
    r.granted = true;
    return r;
}

std::vector<admission_claim> router_board_admission_claims(const router_board_snapshot & snap,
                                                           const std::vector<std::string> & machine_resources,
                                                           admission_priority load_priority) {
    std::vector<admission_claim> out;
    if (!snap.ok) {
        return out;
    }
    auto add = [&](const std::string & id, const std::string & resource, const std::string & holder, bool router,
                   admission_priority prio) {
        if (resource == ROUTER_BOARD_MACHINE_RES) {
            for (const auto & r : machine_resources) {
                out.push_back({ id, r, holder, router, prio });
            }
        } else {
            out.push_back({ id, resource, holder, router, prio });
        }
    };
    for (const auto & c : snap.claims) {
        add(c.id, c.resource, c.holder, c.holder == ROUTER_BOARD_HOLDER, c.priority);
    }
    if (load_priority != ADMISSION_PRIORITY_HIGHEST) {
        for (const auto & e : snap.queue) {
            if (e.holder != ROUTER_BOARD_HOLDER && e.priority >= load_priority) {
                add(ROUTER_BOARD_WAITER_PREFIX + e.id, e.resource, e.holder, false, e.priority);
            }
        }
    }
    return out;
}

std::set<std::string> router_board_contested(const router_board_snapshot & snap, const std::set<std::string> & held) {
    std::set<std::string> out;
    if (!snap.ok) {
        return out;
    }
    for (const auto & e : snap.queue) {
        if (e.holder == ROUTER_BOARD_HOLDER) {
            continue;
        }
        if (e.resource == ROUTER_BOARD_MACHINE_RES) {
            out.insert(held.begin(), held.end());
        } else if (held.count(e.resource)) {
            out.insert(e.resource);
        }
    }
    return out;
}

std::vector<std::string> router_board_pick_yield(const std::vector<router_board_resident> & residents,
                                                 const std::map<std::string, std::set<std::string>> & claims_by_owner,
                                                 const std::set<std::string> & contested) {
    std::vector<std::string> out;
    if (contested.empty()) {
        return out;
    }
    for (const auto & r : residents) {
        if (!r.alive || r.loading || !r.idle || r.pinned || r.held) {
            continue;
        }
        auto it = claims_by_owner.find(r.name);
        if (it == claims_by_owner.end()) {
            continue;
        }
        const bool hit = std::any_of(it->second.begin(), it->second.end(),
                                     [&](const std::string & res) { return contested.count(res) > 0; });
        if (!hit) {
            continue;
        }
        const std::string stop = r.stop_name.empty() ? r.name : r.stop_name;
        if (std::find(out.begin(), out.end(), stop) == out.end()) {
            out.push_back(stop);
        }
    }
    return out;
}

//
// HTTP client
//

static std::string url_encode(const std::string & in) {
    std::string out;
    for (unsigned char c : in) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out.push_back((char) c);
        } else {
            char buf[4];
            snprintf(buf, sizeof(buf), "%%%02X", c);
            out += buf;
        }
    }
    return out;
}

router_board_client::router_board_client(const std::string & url, const std::string & token, int timeout_ms)
        : token(token), timeout_ms(timeout_ms) {
    // scheme://host[:port][/prefix]; httplib takes scheme://host:port
    const size_t scheme_end = url.find("://");
    if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid --board-url '" + url + "': expected http://host:port");
    }
    const size_t path_start = url.find('/', scheme_end + 3);
    base   = path_start == std::string::npos ? url : url.substr(0, path_start);
    prefix = path_start == std::string::npos ? "" : url.substr(path_start);
    while (!prefix.empty() && prefix.back() == '/') {
        prefix.pop_back();
    }
}

namespace {
struct board_http_answer {
    int         status = 0; // 0 = no answer
    std::string body;
    std::string error;
};
}

static board_http_answer board_http(const std::string & base, int timeout_ms, const std::string & token,
                                    const std::string & method, const std::string & path, const std::string & body) {
    board_http_answer a;
    httplib::Client cli(base);
    const time_t sec  = timeout_ms / 1000;
    const time_t usec = (timeout_ms % 1000) * 1000;
    cli.set_connection_timeout(sec, usec);
    cli.set_read_timeout(sec, usec);
    cli.set_write_timeout(sec, usec);
    httplib::Headers headers;
    if (!token.empty()) {
        headers.emplace("Authorization", "Bearer " + token);
    }
    httplib::Result res;
    if (method == "GET") {
        res = cli.Get(path, headers);
    } else if (method == "POST") {
        res = cli.Post(path, headers, body, "application/json");
    } else if (method == "PATCH") {
        res = cli.Patch(path, headers, body, "application/json");
    } else if (method == "DELETE") {
        res = cli.Delete(path, headers);
    } else {
        a.error = "unsupported method " + method;
        return a;
    }
    if (!res) {
        a.error = method + " " + path + ": " + httplib::to_string(res.error());
        return a;
    }
    a.status = res->status;
    a.body   = res->body;
    if (a.status < 200 || a.status >= 300) {
        a.error = method + " " + path + ": HTTP " + std::to_string(a.status) + " " + a.body.substr(0, 200);
    }
    return a;
}

bool router_board_client::list_claims(const std::string & machine, std::vector<router_board_claim> & out, std::string & err) const {
    const auto a = board_http(base, timeout_ms, "", "GET", prefix + "/board/claims?machine=" + url_encode(machine), "");
    if (a.status != 200) {
        err = a.error;
        return false;
    }
    if (!router_board_parse_claims(a.body, out)) {
        err = "unparsable GET /board/claims answer";
        return false;
    }
    return true;
}

bool router_board_client::list_queue(const std::string & machine, std::vector<router_board_queue_entry> & out, std::string & err) const {
    const auto a = board_http(base, timeout_ms, "", "GET", prefix + "/board/queue?machine=" + url_encode(machine), "");
    if (a.status != 200) {
        err = a.error;
        return false;
    }
    if (!router_board_parse_queue(a.body, out)) {
        err = "unparsable GET /board/queue answer";
        return false;
    }
    return true;
}

router_board_claim_result router_board_client::claim(const std::string & machine, const std::string & resource,
                                                     const std::string & note, admission_priority priority, int ttl_hours) const {
    json b = json::object();
    b["machine"]   = machine;
    b["resource"]  = resource;
    b["holder"]    = ROUTER_BOARD_HOLDER;
    b["note"]      = note;
    b["ttl_hours"] = ttl_hours;
    b["priority"]  = admission_priority_str(priority);
    b["source"]    = ROUTER_BOARD_HOLDER;
    const auto a = board_http(base, timeout_ms, token, "POST", prefix + "/board/claims", b.dump());
    if (a.status == 0) {
        router_board_claim_result r;
        r.error = a.error;
        return r;
    }
    return router_board_parse_claim_response(a.status, a.body);
}

static router_board_rc rc_of(const board_http_answer & a, std::string & err) {
    if (a.status >= 200 && a.status < 300) {
        return ROUTER_BOARD_OK;
    }
    err = a.error;
    return a.status == 404 ? ROUTER_BOARD_GONE : ROUTER_BOARD_FAILED;
}

router_board_rc router_board_client::release(const std::string & claim_id, std::string & err) const {
    const auto a = board_http(base, timeout_ms, token, "DELETE", prefix + "/board/claims/" + url_encode(claim_id), "");
    return rc_of(a, err);
}

router_board_rc router_board_client::renew(const std::string & claim_id, int ttl_hours, std::string & err) const {
    json b = json::object();
    b["ttl_hours"] = ttl_hours; // re-anchors the TTL at now (board.update_claim)
    const auto a = board_http(base, timeout_ms, token, "PATCH", prefix + "/board/claims/" + url_encode(claim_id), b.dump());
    return rc_of(a, err);
}

router_board_rc router_board_client::leave_queue(const std::string & machine, const std::string & resource, std::string & err) const {
    json b = json::object();
    b["machine"]  = machine;
    b["resource"] = resource;
    b["holder"]   = ROUTER_BOARD_HOLDER;
    const auto a = board_http(base, timeout_ms, token, "POST", prefix + "/board/queue/leave", b.dump());
    return rc_of(a, err);
}

router_board_rc router_board_client::notify(const std::string & claim_id, admission_notify_kind kind,
                                            const std::string & content, std::string & err) const {
    json b = json::object();
    b["claim_id"] = claim_id;
    b["kind"]     = admission_notify_kind_str(kind);
    b["content"]  = content;
    const auto a = board_http(base, timeout_ms, token, "POST", prefix + "/board/notify", b.dump());
    return rc_of(a, err);
}

//
// agent
//

router_board_agent::router_board_agent(router_board_config cfg_in, router_board_host host_in)
        : cfg(std::move(cfg_in)), host(std::move(host_in)), client(cfg.url, cfg.token, cfg.timeout_ms) {}

router_board_agent::~router_board_agent() {
    stop();
}

void router_board_agent::start(bool run_thread) {
    startup_sweep();
    if (!run_thread) {
        return;
    }
    th = std::thread([this]() {
        while (!quit.load()) {
            tick();
            std::unique_lock<std::mutex> wl(wake_mu);
            wake_cv.wait_for(wl, std::chrono::milliseconds(cfg.poll_ms), [this]() { return wake_flag || quit.load(); });
            wake_flag = false;
        }
    });
}

void router_board_agent::stop() {
    {
        std::lock_guard<std::mutex> lk(mu);
        if (stopped) {
            return;
        }
        stopped = true;
    }
    quit.store(true);
    {
        std::lock_guard<std::mutex> wl(wake_mu);
        wake_flag = true;
    }
    wake_cv.notify_all();
    if (th.joinable()) {
        th.join();
    }
    std::vector<held_claim> all;
    std::vector<std::string> queues;
    {
        std::lock_guard<std::mutex> lk(mu);
        all.swap(claims);
        for (const auto & [res, _] : joined) {
            queues.push_back(res);
        }
        joined.clear();
        joined_prio.clear();
    }
    for (const auto & c : all) {
        std::string err;
        if (client.release(c.claim_id, err) == ROUTER_BOARD_FAILED) {
            BRD_WRN("could not release claim %s (%s for %s) at shutdown: %s\n", c.claim_id.c_str(),
                    c.resource.c_str(), c.owner.c_str(), err.c_str());
        }
    }
    for (const auto & r : queues) {
        std::string err;
        if (client.leave_queue(cfg.machine, r, err) == ROUTER_BOARD_FAILED) {
            BRD_WRN("could not leave the queue for %s at shutdown: %s\n", r.c_str(), err.c_str());
        }
    }
    if (!all.empty() || !queues.empty()) {
        BRD_INF("released %zu claim(s), left %zu queue(s)\n", all.size(), queues.size());
    }
}

void router_board_agent::startup_sweep() {
    std::vector<router_board_claim> cl;
    std::vector<router_board_queue_entry> q;
    std::string err;
    if (!client.list_claims(cfg.machine, cl, err)) {
        BRD_WRN("board unavailable at startup (%s); continuing without it until it answers\n", err.c_str());
        return;
    }
    size_t n_claims = 0;
    for (const auto & c : cl) {
        if (c.holder == ROUTER_BOARD_HOLDER) {
            std::string e;
            if (client.release(c.id, e) != ROUTER_BOARD_FAILED) {
                n_claims++;
            }
        }
    }
    std::set<std::string> queues;
    if (client.list_queue(cfg.machine, q, err)) {
        for (const auto & e : q) {
            if (e.holder == ROUTER_BOARD_HOLDER) {
                queues.insert(e.resource);
            }
        }
    }
    for (const auto & r : queues) {
        std::string e;
        client.leave_queue(cfg.machine, r, e);
    }
    if (n_claims > 0 || !queues.empty()) {
        BRD_INF("released %zu claim(s) and left %zu queue(s) a previous router left on %s\n",
                n_claims, queues.size(), cfg.machine.c_str());
    }
}

bool router_board_agent::available() const {
    std::lock_guard<std::mutex> lk(mu);
    return cache.ok;
}

router_board_snapshot router_board_agent::snapshot() const {
    std::lock_guard<std::mutex> lk(mu);
    return cache;
}

std::vector<admission_claim> router_board_agent::admission_claims(const std::vector<std::string> & machine_resources,
                                                                  admission_priority load_priority) const {
    std::lock_guard<std::mutex> lk(mu);
    return router_board_admission_claims(cache, machine_resources, load_priority);
}

std::vector<std::string> router_board_agent::missing(const std::string & owner, const std::vector<std::string> & resources) const {
    std::lock_guard<std::mutex> lk(mu);
    std::vector<std::string> out;
    for (const auto & r : resources) {
        const bool have = std::any_of(claims.begin(), claims.end(),
                                      [&](const held_claim & c) { return c.owner == owner && c.resource == r; });
        if (!have && std::find(out.begin(), out.end(), r) == out.end()) {
            out.push_back(r);
        }
    }
    return out;
}

std::map<std::string, std::set<std::string>> router_board_agent::held() const {
    std::lock_guard<std::mutex> lk(mu);
    std::map<std::string, std::set<std::string>> out;
    for (const auto & c : claims) {
        out[c.owner].insert(c.resource);
    }
    return out;
}

router_board_claim_result router_board_agent::acquire(const std::string & owner, const std::string & queue_owner,
                                                      const std::string & resource, const std::string & note,
                                                      admission_priority priority) {
    bool rejoin = false;
    {
        std::lock_guard<std::mutex> lk(mu);
        if (stopped) {
            router_board_claim_result r;
            r.error = "board agent stopped";
            return r;
        }
        pending[resource]++; // protects a probation claim of ours on it from the cleanup in tick()
        auto j = joined.find(resource);
        auto p = joined_prio.find(resource);
        rejoin = !queue_owner.empty() && j != joined.end() && !j->second.empty() && p != joined_prio.end() && priority > p->second;
    }
    if (rejoin) {
        // the board keeps the slot's first priority: leave it, the claim below joins again higher
        std::string err;
        if (client.leave_queue(cfg.machine, resource, err) == ROUTER_BOARD_FAILED) {
            BRD_WRN("could not leave the queue for %s to re-join at %s: %s\n", resource.c_str(),
                    admission_priority_str(priority), err.c_str());
        }
    }
    router_board_claim_result r = client.claim(cfg.machine, resource, note, priority, cfg.ttl_hours);
    std::lock_guard<std::mutex> lk(mu);
    if (--pending[resource] <= 0) {
        pending.erase(resource);
    }
    if (r.ok && r.granted) {
        // a probation claim the queue handed us is confirmed in place: same id
        const bool known = std::any_of(claims.begin(), claims.end(),
                                       [&](const held_claim & c) { return c.claim_id == r.claim_id; });
        if (!known) {
            const int64_t now = board_now_ms();
            claims.push_back({ owner, resource, r.claim_id, now, now });
        }
        auto j = joined.find(resource);
        if (j != joined.end() && !queue_owner.empty()) {
            j->second.erase(queue_owner); // our turn came: no longer waiting for it
            if (j->second.empty()) {
                joined.erase(j);
                joined_prio.erase(resource);
            }
        }
    } else if (r.ok && r.queued && !queue_owner.empty()) {
        joined[resource].insert(queue_owner);
        auto p = joined_prio.find(resource);
        if (p == joined_prio.end() || priority > p->second) {
            joined_prio[resource] = priority;
        }
    } else if (!r.ok) {
        BRD_WRN("claim of %s for %s failed: %s\n", resource.c_str(), owner.c_str(), r.error.c_str());
    }
    return r;
}

void router_board_agent::notify(const std::vector<admission_notify> & list, const std::string & content) {
    for (const auto & n : list) {
        if (n.holder == ROUTER_BOARD_HOLDER || n.claim_id.empty() || n.claim_id.rfind(ROUTER_BOARD_WAITER_PREFIX, 0) == 0) {
            continue; // never notify ourselves, nor a session that is only waiting in the queue
        }
        const std::string key = n.claim_id + ":" + admission_notify_kind_str(n.kind);
        {
            std::lock_guard<std::mutex> lk(mu);
            if (!notified.insert(key).second) {
                continue; // told already
            }
        }
        std::string err;
        const router_board_rc rc = client.notify(n.claim_id, n.kind, content, err);
        if (rc == ROUTER_BOARD_OK) {
            BRD_INF("notified %s (claim %s): %s\n", n.holder.c_str(), n.claim_id.c_str(), admission_notify_kind_str(n.kind));
        } else {
            BRD_WRN("notify %s of claim %s failed: %s\n", admission_notify_kind_str(n.kind), n.claim_id.c_str(), err.c_str());
            if (rc == ROUTER_BOARD_FAILED) {
                std::lock_guard<std::mutex> lk(mu);
                notified.erase(key); // try again next time
            }
        }
    }
}

void router_board_agent::leave_queue(const std::string & queue_owner) {
    std::vector<std::string> to_leave;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = joined.begin(); it != joined.end();) {
            it->second.erase(queue_owner);
            if (it->second.empty()) {
                to_leave.push_back(it->first);
                joined_prio.erase(it->first);
                it = joined.erase(it);
            } else {
                ++it;
            }
        }
    }
    for (const auto & r : to_leave) {
        std::string err;
        if (client.leave_queue(cfg.machine, r, err) == ROUTER_BOARD_FAILED) {
            BRD_WRN("could not leave the queue for %s: %s\n", r.c_str(), err.c_str());
        }
    }
}

void router_board_agent::release_claims(std::vector<held_claim> victims) {
    for (const auto & c : victims) {
        std::string err;
        const router_board_rc rc = client.release(c.claim_id, err);
        if (rc == ROUTER_BOARD_FAILED) {
            BRD_WRN("release of claim %s (%s for %s) failed, will retry: %s\n", c.claim_id.c_str(),
                    c.resource.c_str(), c.owner.c_str(), err.c_str());
            std::lock_guard<std::mutex> lk(mu);
            claims.push_back(c);
        } else {
            BRD_INF("released %s (claim %s) held for %s\n", c.resource.c_str(), c.claim_id.c_str(), c.owner.c_str());
        }
    }
}

// Claims whose owner is gone, and those a settled resident no longer needs (a pool load that
// claimed one slot, then was placed on another). Claims taken at or after `since_ms` are
// skipped: their owner may have started loading after `residents` was read.
static std::vector<size_t> dead_claims(const std::vector<router_board_resident> & residents,
                                       const std::vector<std::string> & owners, const std::vector<std::string> & resources,
                                       const std::vector<int64_t> & taken_ms, int64_t since_ms) {
    std::vector<size_t> out;
    for (size_t i = 0; i < owners.size(); ++i) {
        if (taken_ms[i] >= since_ms) {
            continue;
        }
        auto r = std::find_if(residents.begin(), residents.end(),
                              [&](const router_board_resident & x) { return x.name == owners[i]; });
        if (r == residents.end() || !r->alive) {
            out.push_back(i);
        } else if (!r->loading && std::find(r->resources.begin(), r->resources.end(), resources[i]) == r->resources.end()) {
            out.push_back(i);
        }
    }
    return out;
}

void router_board_agent::release_dead() {
    if (!host.residents) {
        return;
    }
    const int64_t since = board_now_ms();
    release_dead_from(host.residents(), since);
}

void router_board_agent::release_dead_from(const std::vector<router_board_resident> & residents, int64_t since_ms) {
    std::vector<held_claim> victims;
    {
        std::lock_guard<std::mutex> lk(mu);
        std::vector<std::string> owners;
        std::vector<std::string> resources;
        std::vector<int64_t>     taken;
        for (const auto & c : claims) {
            owners.push_back(c.owner);
            resources.push_back(c.resource);
            taken.push_back(c.taken_ms);
        }
        const auto dead = dead_claims(residents, owners, resources, taken, since_ms);
        std::set<size_t> drop(dead.begin(), dead.end());
        std::vector<held_claim> keep;
        for (size_t i = 0; i < claims.size(); ++i) {
            (drop.count(i) ? victims : keep).push_back(claims[i]);
        }
        claims.swap(keep);
    }
    release_claims(std::move(victims));
}

void router_board_agent::wake() {
    {
        std::lock_guard<std::mutex> wl(wake_mu);
        wake_flag = true;
    }
    wake_cv.notify_all();
}

static std::string snapshot_signature(const std::vector<router_board_claim> & cl, const std::vector<router_board_queue_entry> & q) {
    std::vector<std::string> parts;
    for (const auto & c : cl) {
        parts.push_back("c:" + c.id + ":" + c.resource + ":" + c.holder + (c.probation ? ":p" : ""));
    }
    for (const auto & e : q) {
        parts.push_back("q:" + e.id + ":" + e.resource + ":" + e.holder);
    }
    std::sort(parts.begin(), parts.end());
    std::string out;
    for (const auto & p : parts) {
        out += p + "|";
    }
    return out;
}

void router_board_agent::tick() {
    // 1. poll (unlocked HTTP), then swap the cache in
    std::vector<router_board_claim>       cl;
    std::vector<router_board_queue_entry> q;
    std::string err;
    const bool ok = client.list_claims(cfg.machine, cl, err) && client.list_queue(cfg.machine, q, err);
    bool changed = false;
    {
        std::lock_guard<std::mutex> lk(mu);
        const bool was_ok = cache.ok;
        if (ok) {
            const std::string sig = snapshot_signature(cl, q);
            changed     = !was_ok || sig != cache_sig;
            cache.ok     = true;
            cache.claims = cl;
            cache.queue  = q;
            cache_sig    = sig;
            for (auto it = notified.begin(); it != notified.end();) {
                const std::string id = it->substr(0, it->find(':'));
                const bool live = std::any_of(cl.begin(), cl.end(), [&](const router_board_claim & c) { return c.id == id; });
                it = live ? std::next(it) : notified.erase(it);
            }
            if (!was_ok) {
                BRD_INF("board available (%s)\n", cfg.url.c_str());
            }
        } else {
            changed = was_ok; // loads queued on a claim can now go ahead
            cache.ok = false;
            cache.claims.clear();
            cache.queue.clear();
            cache_sig.clear();
            if (was_ok) {
                BRD_WRN("board unavailable (%s); admitting loads as if nothing were claimed\n", err.c_str());
            }
        }
    }

    // 2. what the router runs (takes the router's lock; ours is not held)
    const int64_t since = board_now_ms();
    const std::vector<router_board_resident> residents = host.residents ? host.residents() : std::vector<router_board_resident>{};

    // 3. release claims of residents that went away
    release_dead_from(residents, since);

    // 4. renew what is due (PATCH ttl_hours re-anchors the TTL)
    {
        const int64_t now = board_now_ms();
        std::vector<held_claim> due;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (const auto & c : claims) {
                if (now - c.renewed_ms >= cfg.renew_ms) {
                    due.push_back(c);
                }
            }
        }
        for (const auto & c : due) {
            std::string e;
            const router_board_rc rc = client.renew(c.claim_id, cfg.ttl_hours, e);
            std::lock_guard<std::mutex> lk(mu);
            auto it = std::find_if(claims.begin(), claims.end(), [&](const held_claim & x) { return x.claim_id == c.claim_id; });
            if (it == claims.end()) {
                continue;
            }
            if (rc == ROUTER_BOARD_OK) {
                it->renewed_ms = board_now_ms();
            } else if (rc == ROUTER_BOARD_GONE) {
                BRD_WRN("claim %s (%s for %s) is gone from the board; taking it again\n", c.claim_id.c_str(),
                        c.resource.c_str(), c.owner.c_str());
                claims.erase(it);
            } else {
                BRD_WRN("renew of claim %s failed, retrying: %s\n", c.claim_id.c_str(), e.c_str());
                it->renewed_ms = board_now_ms() - cfg.renew_ms + std::min<int64_t>(cfg.renew_ms, 30000);
            }
        }
    }

    if (!ok) {
        if (changed && host.changed) {
            host.changed();
        }
        return;
    }

    // 5. take claims a settled resident should hold but does not (the board was down at its
    //    load, or a claim expired) -- unless someone else holds the resource now
    const router_board_snapshot snap = snapshot();
    for (const auto & r : residents) {
        if (!r.alive || r.loading) {
            continue;
        }
        for (const auto & res : missing(r.name, r.resources)) {
            const bool foreign = std::any_of(snap.claims.begin(), snap.claims.end(), [&](const router_board_claim & c) {
                return c.holder != ROUTER_BOARD_HOLDER && (c.resource == res || c.resource == ROUTER_BOARD_MACHINE_RES);
            });
            if (foreign) {
                continue;
            }
            const std::string key = r.name + "|" + res;
            {
                std::lock_guard<std::mutex> lk(mu);
                auto it = retake_after.find(key);
                if (it != retake_after.end() && board_now_ms() < it->second) {
                    continue; // refused a moment ago: not every tick
                }
            }
            const router_board_claim_result cr = acquire(r.name, "", res, r.name, ADMISSION_PRIORITY_MIDDLE);
            {
                std::lock_guard<std::mutex> lk(mu);
                if (cr.ok && cr.granted) {
                    retake_after.erase(key);
                } else {
                    retake_after[key] = board_now_ms() + 60000;
                }
            }
            if (cr.ok && cr.queued) {
                // lost a race for a resource a resident already uses: do not sit in the queue
                bool waited_for = false;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    waited_for = joined.count(res) > 0;
                }
                if (!waited_for) {
                    std::string e;
                    client.leave_queue(cfg.machine, res, e);
                }
                BRD_WRN("%s runs on %s, which %s claimed meanwhile\n", r.name.c_str(), res.c_str(), cr.held_by.c_str());
            }
        }
    }

    // 6. queue turns (probation claims the board handed the router): confirm them here for the
    //    queued load that waits (PROBATION_S is short, and a load retry may be stuck behind a
    //    long load), release them when nobody waits any more
    {
        struct turn_t {
            std::string             id;
            std::string             resource;
            std::string             owner;
            admission_priority      priority = ADMISSION_PRIORITY_MIDDLE;
            router_probation_action action   = ROUTER_PROBATION_KEEP;
        };
        std::vector<turn_t> turns;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (const auto & c : snap.claims) {
                if (c.holder != ROUTER_BOARD_HOLDER || !c.probation) {
                    continue;
                }
                const bool ours = std::any_of(claims.begin(), claims.end(), [&](const held_claim & x) { return x.claim_id == c.id; });
                auto j = joined.find(c.resource);
                const bool waited = j != joined.end() && !j->second.empty();
                turn_t t;
                t.id       = c.id;
                t.resource = c.resource;
                t.action   = router_board_probation_action(ours, waited, pending.count(c.resource) > 0);
                if (waited) {
                    t.owner = *j->second.begin();
                    auto p  = joined_prio.find(c.resource);
                    t.priority = p != joined_prio.end() ? p->second : ADMISSION_PRIORITY_MIDDLE;
                }
                turns.push_back(t);
            }
        }
        for (const auto & t : turns) {
            if (t.action == ROUTER_PROBATION_CONFIRM) {
                // claiming it again confirms it in place; recorded for the waiting load, which the
                // router counts as alive while it is queued
                const router_board_claim_result cr = acquire(t.owner, t.owner, t.resource, t.owner, t.priority);
                if (cr.ok && cr.granted) {
                    BRD_INF("confirmed the queue turn on %s for %s (claim %s)\n", t.resource.c_str(), t.owner.c_str(), cr.claim_id.c_str());
                    changed = true;
                }
            } else if (t.action == ROUTER_PROBATION_RELEASE) {
                std::string e;
                client.release(t.id, e);
                BRD_INF("released a queue turn on %s nobody waits for any more\n", t.resource.c_str());
            }
        }
    }

    // 7. yield: a session queued for a resource the router holds gets it as soon as the
    //    residents there are idle (pinned and held residents stay)
    {
        const auto by_owner = held();
        std::set<std::string> held_res;
        for (const auto & [_, rs] : by_owner) {
            held_res.insert(rs.begin(), rs.end());
        }
        const std::set<std::string> contested = router_board_contested(snap, held_res);
        for (const auto & name : router_board_pick_yield(residents, by_owner, contested)) {
            std::string who;
            std::string what;
            for (const auto & e : snap.queue) {
                if (e.holder != ROUTER_BOARD_HOLDER && (contested.count(e.resource) || e.resource == ROUTER_BOARD_MACHINE_RES)) {
                    who  = e.holder;
                    what = e.resource;
                    break;
                }
            }
            BRD_INF("yielding: unloading idle %s, %s is queued for %s\n", name.c_str(), who.c_str(), what.c_str());
            if (host.yield) {
                host.yield(name, "yield: " + who + " is queued for " + what);
            }
        }
    }

    if (changed && host.changed) {
        host.changed();
    }
}
