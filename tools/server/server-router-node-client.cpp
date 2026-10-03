#include "server-router-node-client.h"

#include "server-router-group-lifecycle.h"

#include "log.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <cstdlib>
#include <random>

#ifndef _WIN32
#include <signal.h>
#endif

#define NLC_INF(fmt, ...) LOG_INF("node-link: " fmt, __VA_ARGS__)
#define NLC_WRN(fmt, ...) LOG_WRN("node-link: " fmt, __VA_ARGS__)

static int64_t steady_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string jstr(const json & j, const char * k) {
    return j.is_object() && j.contains(k) && j.at(k).is_string() ? j.at(k).get<std::string>() : std::string();
}

static int64_t jint(const json & j, const char * k, int64_t def) {
    if (!j.is_object() || !j.contains(k)) {
        return def;
    }
    const json & v = j.at(k);
    if (v.is_number_integer()) {
        return v.get<long long>();
    }
    if (v.is_number()) {
        return (int64_t) v.get<double>();
    }
    return def;
}

static bool jbool(const json & j, const char * k) {
    return j.is_object() && j.contains(k) && j.at(k).is_boolean() && j.at(k).get<bool>();
}

static std::string machine_label(const std::string & m) {
    return m.empty() ? std::string("local") : m;
}

//
// pure helpers
//

bool router_node_parse_url(const std::string & url_in, std::string & base, std::string & host, int & port, std::string & err) {
    std::string url = url_in;
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    const std::string scheme = "http://";
    if (url.compare(0, scheme.size(), scheme) != 0) {
        err = "router_node URL '" + url_in + "' must start with http://";
        return false;
    }
    const std::string rest = url.substr(scheme.size());
    if (rest.empty() || rest.find('/') != std::string::npos) {
        err = "router_node URL '" + url_in + "' must be http://host:port";
        return false;
    }
    std::string h;
    std::string p;
    if (rest.front() == '[') {
        const size_t rb = rest.find(']');
        if (rb == std::string::npos || rb + 1 >= rest.size() || rest[rb + 1] != ':') {
            err = "router_node URL '" + url_in + "' must be http://[v6]:port";
            return false;
        }
        h = rest.substr(1, rb - 1);
        p = rest.substr(rb + 2);
    } else {
        const size_t c = rest.rfind(':');
        if (c == std::string::npos) {
            err = "router_node URL '" + url_in + "' has no port";
            return false;
        }
        h = rest.substr(0, c);
        p = rest.substr(c + 1);
    }
    if (h.empty() || p.empty() || !std::all_of(p.begin(), p.end(), [](char ch) { return std::isdigit((unsigned char) ch) != 0; })) {
        err = "router_node URL '" + url_in + "' must be http://host:port";
        return false;
    }
    const long v = std::strtol(p.c_str(), nullptr, 10);
    if (v <= 0 || v > 65535) {
        err = "router_node URL '" + url_in + "' has a bad port";
        return false;
    }
    base = url;
    host = h;
    port = (int) v;
    return true;
}

node_child_info router_node_child_info_from_json(const json & j) {
    node_child_info i;
    i.name       = jstr(j, "name");
    i.gen        = jstr(j, "gen");
    i.pid        = (int) jint(j, "pid", 0);
    i.port       = (int) jint(j, "port", 0);
    i.status     = jstr(j, "status");
    i.exit_code  = (int) jint(j, "exit_code", -1);
    i.killed     = jbool(j, "killed");
    i.adopted    = jbool(j, "adopted");
    i.started_ms = jint(j, "started_ms", 0);
    return i;
}

router_node_probe router_node_probe_from_state(const json & state) {
    router_node_probe p;
    if (!state.is_object()) {
        return p;
    }
    p.ok            = true;
    p.mem_available = jint(state, "mem_available", -1);
    if (state.contains("vram") && state.at("vram").is_array()) {
        for (const auto & v : state.at("vram")) {
            proc_vram pv;
            pv.pid        = (int) jint(v, "pid", 0);
            pv.pdev       = jstr(v, "pdev");
            pv.vram_bytes = jint(v, "bytes", 0);
            if (pv.pid > 0 && !pv.pdev.empty()) {
                p.vram.push_back(pv);
            }
        }
    }
    if (state.contains("devices") && state.at("devices").is_array()) {
        for (const auto & d : state.at("devices")) {
            const std::string pdev = jstr(d, "pdev");
            if (!pdev.empty()) {
                p.sysfs_used[pdev] = jint(d, "vram_used", -1);
            }
        }
    }
    if (state.contains("children") && state.at("children").is_array()) {
        for (const auto & c : state.at("children")) {
            const int pid = (int) jint(c, "pid", 0);
            if (pid <= 0 || jstr(c, "status") == "exited") {
                continue;
            }
            p.child_pids.insert(pid);
            if (c.contains("rss_anon") && c.at("rss_anon").is_number()) {
                p.rss[pid] = proc_mem{ jint(c, "rss_anon", 0), jint(c, "rss_shmem", 0) };
            }
        }
    }
    return p;
}

int64_t router_node_free_vram(const router_node_probe & p, const ledger_slot & slot) {
    if (!p.ok) {
        return ledger_free_vram(slot, 0, -1);
    }
    const int64_t foreign = ledger_foreign_vram(p.vram, p.child_pids, slot.pdev);
    auto it = slot.pdev.empty() ? p.sysfs_used.end() : p.sysfs_used.find(slot.pdev);
    return ledger_free_vram(slot, foreign, it == p.sysfs_used.end() ? -1 : it->second);
}

int64_t router_node_child_ram(const router_node_probe & p, int pid) {
    auto it = p.rss.find(pid);
    if (it == p.rss.end()) {
        return -1;
    }
    return std::max<int64_t>(0, it->second.rss_anon) + std::max<int64_t>(0, it->second.rss_shmem);
}

std::vector<router_node_seen> router_node_seen_from_json(const json & children) {
    std::vector<router_node_seen> out;
    if (!children.is_array()) {
        return out;
    }
    for (const auto & c : children) {
        router_node_seen s;
        s.name      = jstr(c, "name");
        s.gen       = jstr(c, "gen");
        s.pid       = (int) jint(c, "pid", 0);
        s.status    = jstr(c, "status");
        s.exit_code = (int) jint(c, "exit_code", -1);
        s.killed    = jbool(c, "killed");
        if (!s.name.empty()) {
            out.push_back(s);
        }
    }
    return out;
}

router_node_reconcile_plan router_node_reconcile(const std::vector<router_node_seen> & children,
                                                 const std::vector<node_orphan_info> & orphans,
                                                 const std::map<std::string, int> & watched, const std::string & gen,
                                                 const std::function<bool(const std::string &)> & wanted) {
    router_node_reconcile_plan plan;
    auto want = [&](const std::string & name) { return wanted && wanted(name); };
    std::set<std::string> covered; // watched names whose child the node still has (or gives back)

    for (const auto & c : children) {
        if (c.status == "exited") {
            continue;
        }
        auto w = watched.find(c.name);
        const bool tracked = w != watched.end() && w->second == c.pid && c.gen == gen;
        if (tracked) {
            covered.insert(c.name); // its exit (if we stop it) arrives as an event
        }
        if (tracked && want(c.name)) {
            plan.keep.push_back(c.name);
        } else if (c.status == "running") {
            plan.stop.push_back(c.name); // another generation, unknown to us, or no longer wanted
        }
    }
    for (const auto & o : orphans) {
        if (o.gen != gen || o.name.empty() || o.state != "waiting") {
            continue; // another generation's orphan: the node's own sweep stops it
        }
        auto w = watched.find(o.name);
        const bool tracked = w != watched.end() && w->second == o.pid;
        if (tracked && want(o.name)) {
            plan.adopt.push_back(o.name);
            covered.insert(o.name);
        } else {
            plan.adopt_stop.push_back(o.name);
            if (tracked) {
                covered.insert(o.name);
            }
        }
    }
    for (const auto & [name, pid] : watched) {
        if (covered.count(name)) {
            continue;
        }
        router_node_seen lost;
        lost.name   = name;
        lost.pid    = pid;
        lost.status = "exited";
        for (const auto & c : children) {
            if (c.name == name && c.pid == pid && c.status == "exited") {
                lost.exit_code = c.exit_code; // the exit event itself was missed
                lost.killed    = c.killed;
            }
        }
        plan.lost.push_back(lost);
    }
    return plan;
}

void router_unavailable_response(const std::string & model, const std::string & machine, const std::string & reason,
                                 int & status, std::string & body, std::map<std::string, std::string> & headers) {
    status = 503;
    headers["Retry-After"] = std::to_string(ROUTER_NODE_RETRY_AFTER_S);
    json err = json::object();
    err["code"]    = 503;
    err["type"]    = "unavailable_error";
    err["message"] = reason.empty() ? "model '" + model + "' is unavailable: machine '" + machine + "' is offline" : reason;
    err["model"]   = model;
    err["machine"] = machine;
    json out = json::object();
    out["error"] = err;
    body = out.dump();
}

std::vector<std::string> router_online_slots(const std::vector<std::string> & slots,
                                             const std::function<std::string(const std::string &)> & slot_machine,
                                             const std::set<std::string> & offline) {
    std::vector<std::string> out;
    for (const auto & s : slots) {
        if (!offline.count(slot_machine(s))) {
            out.push_back(s);
        }
    }
    return out;
}

void router_slot_split(const std::string & id, std::string & machine, std::string & dev) {
    const size_t slash = id.find('/');
    if (slash == std::string::npos) {
        machine.clear();
        dev = id;
    } else {
        machine = id.substr(0, slash);
        dev     = id.substr(slash + 1);
    }
}

std::string router_slot_resolve(const std::string & gpu, const std::string & machine,
                                const std::function<bool(const std::string &)> & is_local) {
    std::string m;
    std::string dev;
    const size_t slash = gpu.find('/');
    if (slash == std::string::npos) {
        m   = machine;
        dev = gpu;
    } else {
        m   = gpu.substr(0, slash);
        dev = gpu.substr(slash + 1);
    }
    if (is_local && is_local(m)) {
        m.clear();
    }
    return m.empty() ? dev : m + "/" + dev;
}

bool router_node_offline_due(bool online, int64_t last_rx_ms, int64_t now_ms, int64_t offline_ms) {
    return online && now_ms - last_rx_ms >= offline_ms;
}

bool router_node_online_due(bool online, int64_t last_rx_ms, int64_t now_ms, int64_t offline_ms) {
    return !online && now_ms - last_rx_ms < offline_ms;
}

bool router_machine_availability::set_online(const std::string & machine, bool online) {
    if (machine.empty()) {
        return false;
    }
    return online ? offline.erase(machine) > 0 : offline.insert(machine).second;
}

std::string router_machine_availability::first_offline(const std::vector<std::string> & machines) const {
    for (const auto & m : machines) {
        if (is_offline(m)) {
            return m;
        }
    }
    return "";
}

std::string router_effective_status(const std::string & status, bool machine_offline) {
    return machine_offline ? "unavailable" : status;
}

static bool is_pdev_at(const std::string & s, size_t i) {
    // dddd:bb:dd.f (hex)
    static const char * shape = "hhhh:hh:hh.h";
    if (i + 12 > s.size()) {
        return false;
    }
    for (size_t k = 0; k < 12; k++) {
        const char c = s[i + k];
        if (shape[k] == 'h') {
            if (!std::isxdigit((unsigned char) c)) {
                return false;
            }
        } else if (c != shape[k]) {
            return false;
        }
    }
    return true;
}

std::string router_probe_pdev(const std::string & probe) {
    std::string s = probe;
    if (s.compare(0, 4, "pci:") == 0) {
        s = s.substr(4);
        return s.size() == 12 && is_pdev_at(s, 0) ? s : std::string();
    }
    std::string last;
    for (size_t i = 0; i + 12 <= s.size(); i++) {
        if (is_pdev_at(s, i)) {
            last = s.substr(i, 12);
        }
    }
    return last;
}

static std::string trim_ws(const std::string & s) {
    size_t a = 0;
    size_t b = s.size();
    while (a < b && std::isspace((unsigned char) s[a])) {
        a++;
    }
    while (b > a && std::isspace((unsigned char) s[b - 1])) {
        b--;
    }
    return s.substr(a, b - a);
}

std::string router_parse_gpus_spec(const std::string & spec, const std::function<bool(const std::string &)> & is_local,
                                   std::vector<router_gpu_spec_entry> & out) {
    out.clear();
    size_t start = 0;
    while (start <= spec.size()) {
        size_t comma = spec.find(',', start);
        if (comma == std::string::npos) {
            comma = spec.size();
        }
        const std::string entry = trim_ws(spec.substr(start, comma - start));
        start = comma + 1;
        if (entry.empty()) {
            continue;
        }
        const size_t p0 = entry.find(':');
        const size_t p1 = p0 == std::string::npos ? std::string::npos : entry.find(':', p0 + 1);
        if (p0 == std::string::npos || p1 == std::string::npos) {
            return "invalid --gpus entry '" + entry + "', expected [machine/]name[=board]:total_mb:probe";
        }
        router_gpu_spec_entry e;
        std::string name = trim_ws(entry.substr(0, p0));
        const size_t slash = name.find('/');
        if (slash != std::string::npos) {
            e.machine = trim_ws(name.substr(0, slash));
            name      = trim_ws(name.substr(slash + 1));
            if (e.machine.empty()) {
                return "invalid --gpus entry '" + entry + "': empty machine before '/'";
            }
            if (is_local && is_local(e.machine)) {
                e.machine.clear(); // this machine, written out
            }
        }
        const size_t eq = name.find('=');
        e.dev   = trim_ws(eq == std::string::npos ? name : name.substr(0, eq));
        e.board = eq == std::string::npos ? std::string() : trim_ws(name.substr(eq + 1));
        if (e.dev.empty() || (eq != std::string::npos && e.board.empty()) || e.dev.find('/') != std::string::npos) {
            return "invalid --gpus entry '" + entry + "', expected [machine/]name[=board]:total_mb:probe";
        }
        const std::string mb = trim_ws(entry.substr(p0 + 1, p1 - p0 - 1));
        if (!mb.empty()) {
            if (!std::all_of(mb.begin(), mb.end(), [](char c) { return std::isdigit((unsigned char) c) != 0; })) {
                return "invalid --gpus entry '" + entry + "': total_mb must be a number";
            }
            e.total_mb = std::strtoll(mb.c_str(), nullptr, 10);
        }
        e.probe = trim_ws(entry.substr(p1 + 1));
        if (e.probe.empty()) {
            return "invalid --gpus entry '" + entry + "': no probe";
        }
        if (!e.machine.empty()) {
            if (e.total_mb <= 0) {
                return "--gpus entry '" + entry + "' is on machine '" + e.machine +
                       "': its total_mb must be given (the leader cannot read that machine's sysfs)";
            }
            e.pdev = router_probe_pdev(e.probe);
            if (e.pdev.empty()) {
                return "--gpus entry '" + entry + "' is on machine '" + e.machine +
                       "': its probe must name the card's PCI address (pci:0000:03:00.0 or a sysfs path with it)";
            }
        }
        out.push_back(e);
    }
    return "";
}

//
// the link
//

router_node_link::router_node_link(router_node_link_config cfg_in, std::string host) : cfg(std::move(cfg_in)), host_(std::move(host)) {}

router_node_link::~router_node_link() {
    stop_commands();
}

std::string router_node_link::exe() const {
    std::lock_guard<std::mutex> lk(mu);
    return node_exe;
}

void router_node_link::set_hooks(router_node_hooks h) {
    std::lock_guard<std::mutex> lk(mu);
    hooks = std::move(h);
}

router_node_probe router_node_link::probe() const {
    std::lock_guard<std::mutex> lk(mu);
    return probe_cache;
}

void router_node_link::refresh_probe() {
    json st;
    try {
        st = state();
    } catch (const std::exception & e) {
        return; // offline / failed: keep what we had
    }
    router_node_probe p = router_node_probe_from_state(st);
    std::lock_guard<std::mutex> lk(mu);
    probe_cache = std::move(p);
    const std::string exe_now = jstr(st, "exe");
    if (!exe_now.empty()) {
        node_exe = exe_now;
    }
}

node_child_info router_node_link::spawn(const node_spawn_request & req, router_node_watch w) {
    // A child that just exited (a reload, a download that hands over to the model) still holds its
    // name until its exit event has been delivered: retry a 409 briefly before giving up.
    const int64_t deadline = steady_ms() + ROUTER_NODE_SPAWN_409_RETRY_MS;
    while (true) {
        try {
            return spawn_once(req, w);
        } catch (const server_node_error & err) {
            if (err.status != 409 || steady_ms() >= deadline) {
                throw;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

node_child_info router_node_link::spawn_once(const node_spawn_request & req, const router_node_watch & w_in) {
    router_node_watch w = w_in;
    auto e = std::make_shared<watch_entry>();
    {
        std::lock_guard<std::mutex> lk(mu);
        if (watches.count(req.name)) {
            throw server_node_error(409, "a child named '" + req.name + "' is still running on " + machine_label(cfg.machine));
        }
        e->id = next_watch_id++;
        e->w  = std::move(w);
        watches[req.name] = e;
    }
    auto drop = [&]() {
        std::lock_guard<std::mutex> lk(mu);
        auto it = watches.find(req.name);
        if (it != watches.end() && it->second == e) {
            watches.erase(it);
        }
    };
    node_child_info info;
    try {
        info = do_spawn(req);
    } catch (const server_node_error & err) {
        drop();
        if (err.status == 0) {
            after_failed_spawn(req.name); // it may have started before the answer was lost
        }
        throw;
    } catch (const std::exception & err) {
        drop();
        throw server_node_error(500, err.what());
    }
    if (w.on_spawn) {
        w.on_spawn(info);
    }
    // events that came in while the spawn was in flight, in order, before any later one
    std::lock_guard<std::mutex> d(dispatch_mu);
    std::vector<json> pend;
    {
        std::lock_guard<std::mutex> lk(mu);
        e->pid = info.pid;
        pend.swap(e->pending);
    }
    for (const auto & ev : pend) {
        dispatch_locked_d(ev);
    }
    return info;
}

void router_node_link::watch_existing(const std::string & name, int pid, router_node_watch w) {
    auto e = std::make_shared<watch_entry>();
    std::lock_guard<std::mutex> lk(mu);
    e->id  = next_watch_id++;
    e->pid = pid;
    e->w   = std::move(w);
    watches[name] = e;
}

void router_node_link::unwatch(const std::string & name) {
    std::lock_guard<std::mutex> d(dispatch_mu); // no callback of it is running once this returns
    std::lock_guard<std::mutex> lk(mu);
    watches.erase(name);
}

std::shared_ptr<router_node_link::watch_entry> router_node_link::take_watch(const std::string & name, int pid) {
    std::lock_guard<std::mutex> lk(mu);
    auto it = watches.find(name);
    if (it == watches.end() || it->second->pid != pid || pid <= 0) {
        return nullptr;
    }
    auto e = it->second;
    watches.erase(it);
    return e;
}

void router_node_link::fire_exit(const std::shared_ptr<watch_entry> & e, const std::string & name, const node_child_info & info) {
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = watches.find(name);
        if (it == watches.end() || it->second != e) {
            return; // already reported, or replaced
        }
        watches.erase(it);
    }
    if (e->w.on_exit) {
        e->w.on_exit(info);
    }
}

void router_node_link::dispatch_locked_d(const json & ev) {
    const std::string type = jstr(ev, "type");
    const std::string name = jstr(ev, "name");
    const int         pid  = (int) jint(ev, "pid", 0);
    std::shared_ptr<watch_entry> e;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = watches.find(name);
        if (it != watches.end()) {
            e = it->second;
            if (e->pid == 0) {
                e->pending.push_back(ev); // its spawn has not returned yet
                return;
            }
            if (e->pid != pid) {
                e.reset(); // an earlier child with the same name
            } else {
                e->seen = true;
            }
        }
    }
    if (type == "line") {
        // not logged here: the watcher (the router's monitor / group thread) logs each line once
        if (e && e->w.on_line) {
            e->w.on_line(jstr(ev, "line"));
        }
    } else if (type == "child") {
        if (e && jstr(ev, "status") == "exited") {
            fire_exit(e, name, router_node_child_info_from_json(ev));
        }
    }
}

void router_node_link::post(std::function<void()> fn) {
    {
        std::lock_guard<std::mutex> lk(cmd_mu);
        if (cmd_quit) {
            return;
        }
        cmd_q.push_back(std::move(fn));
    }
    cmd_cv.notify_all();
}

void router_node_link::note_rx() {
    last_rx_ms.store(steady_ms());
}

void router_node_link::check_offline() {
    if (!remote() || !online_flag.load()) {
        return;
    }
    if (!router_node_offline_due(true, last_rx_ms.load(), steady_ms(), cfg.offline_ms)) {
        return;
    }
    online_flag.store(false);
    NLC_WRN("node %s: no heartbeat for %" PRId64 " ms, offline\n", cfg.machine.c_str(), cfg.offline_ms);
    router_node_hooks h;
    {
        std::lock_guard<std::mutex> lk(mu);
        h = hooks;
    }
    if (h.on_online) {
        h.on_online(false);
    }
}

void router_node_link::on_event(const json & ev) {
    std::lock_guard<std::mutex> d(dispatch_mu);
    note_rx();
    if (ev.contains("seq")) {
        const uint64_t seq = (uint64_t) jint(ev, "seq", 0);
        std::lock_guard<std::mutex> lk(mu);
        if (have_cursor && seq < cursor) {
            return; // replayed twice (a reconnect overlapped)
        }
        cursor      = seq + 1;
        have_cursor = true;
    }
    const std::string type = jstr(ev, "type");
    if (type == "lost") {
        NLC_WRN("node %s: %" PRId64 " event(s) were lost; reconciling\n", cfg.machine.c_str(), jint(ev, "missed", 0));
        bool schedule = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            schedule = !reconcile_pending;
            reconcile_pending = true;
        }
        if (schedule) {
            post([this]() { reconcile(); });
        }
        return;
    }
    if (type == "orphan") {
        NLC_INF("node %s: orphan %s pid %" PRId64 " ('%s', gen %s)\n", cfg.machine.c_str(), jstr(ev, "action").c_str(),
                jint(ev, "pid", 0), jstr(ev, "name").c_str(), jstr(ev, "gen").c_str());
        return;
    }
    if (type == "line" || type == "child") {
        dispatch_locked_d(ev);
    }
}

void router_node_link::on_heartbeat(const json & hb) {
    std::lock_guard<std::mutex> d(dispatch_mu);
    note_rx();
    const int      hb_pid = (int) jint(hb, "node_pid", 0);
    const uint64_t next   = (uint64_t) jint(hb, "next_seq", 0);
    bool restarted = false;
    bool schedule  = false;
    bool settled   = false;
    uint64_t cur   = 0;
    {
        std::lock_guard<std::mutex> lk(mu);
        const std::string exe_now = jstr(hb, "exe");
        if (!exe_now.empty()) {
            node_exe = exe_now;
        }
        if (node_pid != hb_pid) {
            restarted   = node_pid != 0;
            node_pid    = hb_pid;
            cursor      = next; // a new node counts from 1 again
            have_cursor = true;
        } else if (!have_cursor) {
            cursor      = next;
            have_cursor = true;
        }
        if ((restarted || !online_flag.load()) && !reconcile_pending) {
            reconcile_pending = true;
            schedule          = true;
        }
        settled = !reconcile_pending;
        cur     = cursor;
    }
    if (restarted) {
        NLC_WRN("node %s restarted (pid %d): reconciling its children\n", cfg.machine.c_str(), hb_pid);
    }
    if (schedule) {
        post([this]() { reconcile(); });
    }
    // Until the reconcile ran (orphans to adopt), and while events written before this heartbeat
    // are still on their way, the heartbeat's child list says nothing new.
    if (!settled || cur < next) {
        return;
    }
    const std::vector<router_node_seen> seen = router_node_seen_from_json(hb.contains("children") ? hb.at("children") : json::array());

    // a watched child the node no longer runs, whose exit event was lost
    std::vector<std::pair<std::string, std::shared_ptr<watch_entry>>> gone;
    std::vector<std::pair<std::string, int>> strays;
    {
        std::lock_guard<std::mutex> lk(mu);
        for (const auto & [name, e] : watches) {
            if (e->pid <= 0 || !e->seen) {
                continue;
            }
            auto s = std::find_if(seen.begin(), seen.end(), [&](const router_node_seen & x) { return x.name == name; });
            if (s == seen.end() || s->pid != e->pid || s->status == "exited") {
                gone.emplace_back(name, e);
            }
        }
        // ours or not, a child nobody watches is stopped once it shows up twice running
        std::map<std::string, int> next_stray;
        for (const auto & s : seen) {
            if (s.status != "running" || watches.count(s.name)) {
                continue;
            }
            const int n = stray.count(s.name) ? stray[s.name] + 1 : 1;
            if (n >= 2) {
                strays.emplace_back(s.name, s.pid);
            } else {
                next_stray[s.name] = n;
            }
        }
        stray.swap(next_stray);
    }
    for (const auto & [name, e] : gone) {
        NLC_WRN("node %s: %s (pid %d) is gone and its exit was not reported\n", cfg.machine.c_str(), name.c_str(), e->pid);
        node_child_info info;
        info.name   = name;
        info.pid    = e->pid;
        info.status = "exited";
        fire_exit(e, name, info);
    }
    for (const auto & [name, pid] : strays) {
        NLC_WRN("node %s: %s (pid %d) runs and nobody watches it: stopping it\n", cfg.machine.c_str(), name.c_str(), pid);
        stop_async(name, pid, ROUTER_NODE_RECONCILE_STOP_S, "both");
    }
}

void router_node_link::reconcile() {
    uint64_t judge_below = 0;
    {
        std::lock_guard<std::mutex> lk(mu);
        judge_below = next_watch_id; // watches registered after the state read are not judged by it
    }
    json st;
    try {
        st = state();
    } catch (const std::exception & e) {
        NLC_WRN("node %s: reconcile could not read the state: %s\n", cfg.machine.c_str(), e.what());
        std::lock_guard<std::mutex> lk(mu);
        reconcile_pending = false; // the next heartbeat tries again
        return;
    }
    router_node_hooks h;
    std::map<std::string, int> watched;
    {
        std::lock_guard<std::mutex> lk(mu);
        probe_cache = router_node_probe_from_state(st);
        const std::string exe_now = jstr(st, "exe");
        if (!exe_now.empty()) {
            node_exe = exe_now;
        }
        h = hooks;
        for (const auto & [name, e] : watches) {
            if (e->id < judge_below && e->pid > 0) {
                watched[name] = e->pid;
            }
        }
    }
    std::vector<node_orphan_info> orphans;
    if (st.contains("orphans") && st.at("orphans").is_array()) {
        for (const auto & o : st.at("orphans")) {
            orphans.push_back({ (int) jint(o, "pid", 0), jstr(o, "name"), jstr(o, "gen"), jstr(o, "state") });
        }
    }
    const auto seen = router_node_seen_from_json(st.contains("children") ? st.at("children") : json::array());
    router_node_reconcile_plan plan = router_node_reconcile(seen, orphans, watched, cfg.gen, h.wanted);

    for (const auto & name : plan.adopt) {
        try {
            const node_child_info a = adopt(name, cfg.gen);
            NLC_INF("node %s: re-adopted %s (pid %d)\n", cfg.machine.c_str(), name.c_str(), a.pid);
        } catch (const std::exception & e) {
            NLC_WRN("node %s: could not re-adopt %s: %s\n", cfg.machine.c_str(), name.c_str(), e.what());
            router_node_seen l;
            l.name   = name;
            l.pid    = watched[name];
            l.status = "exited";
            plan.lost.push_back(l);
        }
    }
    for (const auto & name : plan.adopt_stop) {
        try {
            adopt(name, cfg.gen);
            stop(name, ROUTER_NODE_RECONCILE_STOP_S, "both");
            NLC_INF("node %s: took back %s, which is no longer wanted, and stopped it\n", cfg.machine.c_str(), name.c_str());
        } catch (const std::exception & e) {
            NLC_WRN("node %s: could not stop orphan %s: %s\n", cfg.machine.c_str(), name.c_str(), e.what());
        }
    }
    for (const auto & name : plan.stop) {
        try {
            stop(name, ROUTER_NODE_RECONCILE_STOP_S, "both");
            NLC_INF("node %s: stopped %s (not wanted by this router)\n", cfg.machine.c_str(), name.c_str());
        } catch (const std::exception & e) {
            NLC_WRN("node %s: could not stop %s: %s\n", cfg.machine.c_str(), name.c_str(), e.what());
        }
    }
    if (!plan.lost.empty()) {
        std::lock_guard<std::mutex> d(dispatch_mu);
        for (const auto & l : plan.lost) {
            auto e = take_watch(l.name, l.pid);
            if (!e) {
                continue;
            }
            NLC_WRN("node %s: %s (pid %d) is gone\n", cfg.machine.c_str(), l.name.c_str(), l.pid);
            node_child_info info;
            info.name      = l.name;
            info.pid       = l.pid;
            info.status    = "exited";
            info.exit_code = l.exit_code;
            info.killed    = l.killed;
            if (e->w.on_exit) {
                e->w.on_exit(info);
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(mu);
        reconcile_pending = false;
    }
    if (remote() && router_node_online_due(online_flag.load(), last_rx_ms.load(), steady_ms(), cfg.offline_ms)) {
        online_flag.store(true);
        NLC_INF("node %s online (%zu kept, %zu re-adopted, %zu stopped, %zu gone)\n", cfg.machine.c_str(),
                plan.keep.size(), plan.adopt.size(), plan.stop.size() + plan.adopt_stop.size(), plan.lost.size());
        if (h.on_online) {
            h.on_online(true);
        }
    }
}

void router_node_link::stop_async(const std::string & name, int pid, int timeout_s, const std::string & method) {
    post([this, name, pid, timeout_s, method]() {
        {
            std::lock_guard<std::mutex> lk(mu);
            auto it = watches.find(name);
            if (pid > 0 && it != watches.end() && it->second->pid != pid) {
                return; // a newer child has that name now
            }
        }
        try {
            stop(name, timeout_s, method);
        } catch (const server_node_error & e) {
            if (e.status != 404) {
                NLC_WRN("node %s: stop of %s failed: %s\n", machine_label(cfg.machine).c_str(), name.c_str(), e.what());
            }
        } catch (const std::exception & e) {
            NLC_WRN("node %s: stop of %s failed: %s\n", machine_label(cfg.machine).c_str(), name.c_str(), e.what());
        }
    });
}

void router_node_link::signal_async(const std::string & name, int pid, int sig) {
    post([this, name, pid, sig]() {
        {
            std::lock_guard<std::mutex> lk(mu);
            auto it = watches.find(name);
            if (pid > 0 && it != watches.end() && it->second->pid != pid) {
                return;
            }
        }
        try {
            signal(name, sig);
        } catch (const server_node_error & e) {
            if (e.status != 404 && e.status != 409) {
                NLC_WRN("node %s: signal %d to %s failed: %s\n", machine_label(cfg.machine).c_str(), sig, name.c_str(), e.what());
            }
        } catch (const std::exception & e) {
            NLC_WRN("node %s: signal %d to %s failed: %s\n", machine_label(cfg.machine).c_str(), sig, name.c_str(), e.what());
        }
    });
}

void router_node_link::start_commands() {
    std::lock_guard<std::mutex> lk(cmd_mu);
    if (cmd_th.joinable()) {
        return;
    }
    cmd_quit = false;
    cmd_th = std::thread([this]() {
        int64_t last_probe = 0;
        while (true) {
            std::function<void()> fn;
            {
                std::unique_lock<std::mutex> l(cmd_mu);
                cmd_cv.wait_for(l, std::chrono::milliseconds(500), [&]() { return cmd_quit || !cmd_q.empty(); });
                if (!cmd_q.empty()) {
                    fn = std::move(cmd_q.front());
                    cmd_q.pop_front();
                } else if (cmd_quit) {
                    return; // queued commands (stops at shutdown) ran first
                }
            }
            if (fn) {
                try {
                    fn();
                } catch (const std::exception & e) {
                    NLC_WRN("node %s: command failed: %s\n", machine_label(cfg.machine).c_str(), e.what());
                }
                continue;
            }
            check_offline();
            if (remote() && online_flag.load() && steady_ms() - last_probe >= cfg.probe_ms) {
                last_probe = steady_ms();
                refresh_probe();
            }
        }
    });
}

void router_node_link::stop_commands() {
    {
        std::lock_guard<std::mutex> lk(cmd_mu);
        cmd_quit = true;
    }
    cmd_cv.notify_all();
    if (cmd_th.joinable() && cmd_th.get_id() != std::this_thread::get_id()) {
        cmd_th.join();
    }
}

//
// local: the in-process node
//

namespace {

class router_node_local_impl : public router_node_link {
  public:
    router_node_local_impl(router_node_link_config c, server_node_config ncfg)
            : router_node_link(std::move(c), "127.0.0.1") {
        node_exe = ncfg.exe;
        node     = std::make_unique<server_node>(std::move(ncfg));
        node_pid = node->config().self_pid;
        cursor   = node->next_seq();
        have_cursor = true;
        online_flag.store(true); // our own process: never offline
        note_rx();
    }

    ~router_node_local_impl() override {
        shutdown();
    }

    bool remote() const override { return false; }

    void start() override {
        start_commands();
        if (!th.joinable()) {
            th = std::thread([this]() { run(); });
        }
    }

    void shutdown() override {
        if (done.exchange(true)) {
            return;
        }
        stop_commands(); // queued stops run first
        if (node) {
            node->close_events();
        }
        if (th.joinable()) {
            th.join();
        }
        node.reset(); // stops whatever still runs (exit command + SIGTERM, SIGKILL after its grace)
    }

    node_child_info stop(const std::string & name, int timeout_s, const std::string & method) override {
        return node->stop(name, timeout_s, method);
    }
    node_child_info signal(const std::string & name, int sig) override {
        return node->signal(name, sig);
    }
    node_child_info adopt(const std::string & name, const std::string & gen) override {
        return node->adopt(name, gen);
    }
    json state() override {
        return node->state();
    }

  protected:
    node_child_info do_spawn(const node_spawn_request & req) override {
        return node->spawn(req);
    }

  private:
    void run() {
        uint64_t cur = 0;
        {
            std::lock_guard<std::mutex> lk(mu);
            cur = cursor;
        }
        int64_t last_hb = 0;
        while (true) {
            std::vector<json> evs;
            if (!node->wait_events(cur, evs, std::max<int64_t>(50, cfg.heartbeat_ms))) {
                return; // closing
            }
            for (const auto & ev : evs) {
                on_event(ev);
            }
            if (steady_ms() - last_hb >= cfg.heartbeat_ms) {
                last_hb = steady_ms();
                on_heartbeat(node->heartbeat_json());
            }
        }
    }

    std::unique_ptr<server_node> node;
    std::thread                  th;
    std::atomic<bool>            done{false};
};

//
// remote: the --router-node daemon over HTTP
//

struct http_answer {
    int         status = 0; // 0 = no answer
    std::string body;
    std::string error;
};

class router_node_remote_impl : public router_node_link {
  public:
    router_node_remote_impl(router_node_link_config c, std::string base, std::string host, std::string token)
            : router_node_link(std::move(c), std::move(host)), base(std::move(base)), token(std::move(token)) {}

    ~router_node_remote_impl() override {
        shutdown();
    }

    bool remote() const override { return true; }

    void start() override {
        start_commands();
        if (!ev_th.joinable()) {
            ev_th = std::thread([this]() { run_events(); });
        }
    }

    void shutdown() override {
        if (done.exchange(true)) {
            return;
        }
        quit.store(true);
        {
            std::lock_guard<std::mutex> lk(cli_mu);
            if (live_cli != nullptr) {
                live_cli->stop(); // ends the blocking SSE read
            }
        }
        wait_cv.notify_all();
        if (ev_th.joinable()) {
            ev_th.join();
        }
        stop_commands();
    }

    node_child_info stop(const std::string & name, int timeout_s, const std::string & method) override {
        json b = json::object();
        b["name"]      = name;
        b["timeout_s"] = timeout_s;
        b["method"]    = method;
        return info_or_throw(call("POST", "/node/stop", b.dump()), "stop " + name);
    }

    node_child_info signal(const std::string & name, int sig) override {
        json b = json::object();
        b["name"] = name;
        b["sig"]  = sig;
        return info_or_throw(call("POST", "/node/signal", b.dump()), "signal " + name);
    }

    node_child_info adopt(const std::string & name, const std::string & gen) override {
        json b = json::object();
        b["name"] = name;
        b["gen"]  = gen;
        return info_or_throw(call("POST", "/node/adopt", b.dump()), "adopt " + name);
    }

    json state() override {
        const http_answer a = call("GET", "/node/state", "");
        throw_on_error(a, "state");
        json j = json::parse_no_throw(a.body);
        if (j.is_discarded() || !j.is_object()) {
            throw server_node_error(502, "node " + cfg.machine + ": unparsable /node/state answer");
        }
        return j;
    }

  protected:
    node_child_info do_spawn(const node_spawn_request & req) override {
        if (!online()) {
            throw server_node_error(503, "node " + cfg.machine + " is offline");
        }
        json b = json::object();
        b["name"] = req.name;
        b["gen"]  = req.gen;
        json args = json::array();
        for (const auto & a : req.args) {
            args.push_back(a);
        }
        b["args"] = args;
        json env = json::array();
        for (const auto & e : req.env) {
            env.push_back(e);
        }
        b["env"] = env;
        if (req.port > 0) {
            b["port"] = req.port;
        }
        if (req.alloc_port) {
            b["alloc_port"] = true;
        }
        return info_or_throw(call("POST", "/node/spawn", b.dump()), "spawn " + req.name);
    }

    void after_failed_spawn(const std::string & name) override {
        try {
            stop(name, 0, "term");
            NLC_WRN("node %s: the spawn of %s got no answer; stopped what may have started\n", cfg.machine.c_str(), name.c_str());
        } catch (const std::exception &) {
            // unreachable: the reconcile on its return stops it (not wanted, or nobody watches it)
        }
    }

  private:
    http_answer call(const std::string & method, const std::string & path, const std::string & body) const {
        http_answer a;
        httplib::Client cli(base);
        const time_t sec  = cfg.http_timeout_ms / 1000;
        const time_t usec = (cfg.http_timeout_ms % 1000) * 1000;
        cli.set_connection_timeout(std::min<time_t>(sec, 3), sec >= 3 ? 0 : usec);
        cli.set_read_timeout(sec, usec);
        cli.set_write_timeout(sec, usec);
        httplib::Headers headers = { { "Authorization", "Bearer " + token } };
        httplib::Result res = method == "GET" ? cli.Get(path, headers) : cli.Post(path, headers, body, "application/json");
        if (!res) {
            a.error = method + " " + path + ": " + httplib::to_string(res.error());
            return a;
        }
        a.status = res->status;
        a.body   = res->body;
        return a;
    }

    void throw_on_error(const http_answer & a, const std::string & what) const {
        if (a.status == 0) {
            throw server_node_error(0, "node " + cfg.machine + " did not answer (" + what + "): " + a.error);
        }
        if (a.status < 200 || a.status >= 300) {
            json j = json::parse_no_throw(a.body);
            std::string msg;
            if (!j.is_discarded() && j.is_object() && j.contains("error")) {
                msg = jstr(j.at("error"), "message");
            }
            throw server_node_error(a.status, "node " + cfg.machine + " (" + what + "): HTTP " + std::to_string(a.status) +
                                                  (msg.empty() ? std::string() : ": " + msg));
        }
    }

    node_child_info info_or_throw(const http_answer & a, const std::string & what) const {
        throw_on_error(a, what);
        json j = json::parse_no_throw(a.body);
        if (j.is_discarded() || !j.is_object()) {
            throw server_node_error(502, "node " + cfg.machine + " (" + what + "): unparsable answer");
        }
        return router_node_child_info_from_json(j);
    }

    void run_events() {
        std::string last_error;
        while (!quit.load()) {
            std::string path = "/node/events";
            {
                std::lock_guard<std::mutex> lk(mu);
                if (have_cursor) {
                    path += "?since=" + std::to_string(cursor);
                }
            }
            httplib::Client cli(base);
            cli.set_connection_timeout(3, 0);
            // heartbeats come every 2 s: a read that waits offline_ms for a byte means the node is gone
            cli.set_read_timeout(cfg.offline_ms / 1000, (cfg.offline_ms % 1000) * 1000);
            {
                std::lock_guard<std::mutex> lk(cli_mu);
                if (quit.load()) {
                    break;
                }
                live_cli = &cli;
            }
            std::string buf;
            httplib::Headers headers = { { "Authorization", "Bearer " + token } };
            int status = 0;
            auto res = cli.Get(path, headers,
                [&](const httplib::Response & r) {
                    status = r.status;
                    return r.status == 200;
                },
                [&](const char * data, size_t len) {
                    buf.append(data, len);
                    size_t pos;
                    while ((pos = buf.find("\n\n")) != std::string::npos) {
                        const std::string chunk = buf.substr(0, pos);
                        buf.erase(0, pos + 2);
                        if (chunk.compare(0, 6, "data: ") != 0) {
                            continue;
                        }
                        const json ev = json::parse_no_throw(chunk.substr(6));
                        if (ev.is_discarded() || !ev.is_object()) {
                            continue;
                        }
                        if (jstr(ev, "type") == "heartbeat") {
                            on_heartbeat(ev);
                        } else {
                            on_event(ev);
                        }
                    }
                    return !quit.load();
                });
            {
                std::lock_guard<std::mutex> lk(cli_mu);
                live_cli = nullptr;
            }
            if (quit.load()) {
                break;
            }
            std::string err = status != 0 && status != 200 ? "HTTP " + std::to_string(status)
                            : !res ? httplib::to_string(res.error()) : "stream ended";
            if (err != last_error) {
                NLC_WRN("node %s (%s): event stream: %s; reconnecting\n", cfg.machine.c_str(), base.c_str(), err.c_str());
                last_error = err;
            }
            check_offline();
            std::unique_lock<std::mutex> wl(wait_mu);
            wait_cv.wait_for(wl, std::chrono::milliseconds(500), [&]() { return quit.load(); });
        }
    }

    std::string base;
    std::string token;

    std::thread             ev_th;
    std::atomic<bool>       quit{false};
    std::atomic<bool>       done{false};
    std::mutex              cli_mu;
    httplib::Client *       live_cli = nullptr;
    std::mutex              wait_mu;
    std::condition_variable wait_cv;
};

} // namespace

std::shared_ptr<router_node_link> router_node_make_local(router_node_link_config cfg, server_node_config node_cfg) {
    node_cfg.log_lines = false; // the link's watcher logs the lines (once)
    return std::make_shared<router_node_local_impl>(std::move(cfg), std::move(node_cfg));
}

std::shared_ptr<router_node_link> router_node_make_remote(router_node_link_config cfg, const std::string & url,
                                                          const std::string & token) {
    std::string base;
    std::string host;
    int port = 0;
    std::string err;
    if (!router_node_parse_url(url, base, host, port, err)) {
        throw std::runtime_error(err);
    }
    return std::make_shared<router_node_remote_impl>(std::move(cfg), base, host, token);
}

std::string router_child_key_generate() {
    std::random_device rd;
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (int i = 0; i < 32; i++) {
        out.push_back(hex[rd() & 0xf]);
    }
    return out;
}

void router_child_auth_headers(std::map<std::string, std::string> & headers, const std::string & key) {
    if (key.empty()) {
        return;
    }
    for (auto it = headers.begin(); it != headers.end();) {
        std::string lower = it->first;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        if (lower == "authorization") {
            it = headers.erase(it);
        } else {
            ++it;
        }
    }
    headers["Authorization"] = "Bearer " + key;
}
