// Tests for the router node (tools/server/server-node.cpp) and the machine registry
// (tools/server/server-router-machines.cpp): machines.json parsing; --router-node startup
// validation and the bearer token check; the core child table (spawn / stop / signal / state /
// events) against /bin/sh and the fake worker (tests/router-fixtures/fake-worker.py, no GPU);
// the stale-generation orphan sweep + adoption against a fake /proc tree describing real
// processes; and the /node/* HTTP layer in-process on a random localhost port.

#undef NDEBUG

#include "server-node.h"
#include "server-router-group-lifecycle.h"
#include "server-router-machines.h"

#include "common.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
extern char ** environ;
#endif

namespace fs = std::filesystem;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "FAIL line %d: %s\n", line, what);
        abort();
    }
}
#define CHECK(x) check((x), #x, __LINE__)

static bool has(const std::string & s, const std::string & needle) {
    return s.find(needle) != std::string::npos;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static void write_file(const fs::path & p, const std::string & content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << content;
}

//
// machines.json
//

static void test_machines() {
    // the live registry's shape plus router_node, unknown keys and a non-object entry
    const machines_registry reg = load_machines("tests/router-fixtures/machines/machines.json");
    CHECK(reg.ok());
    CHECK(reg.machines.size() == 2); // "laptop" is not an object: skipped
    CHECK(reg.local_machine() == "mad-lab-main");
    CHECK(reg.node_url("mad-lab-2026") == "http://192.168.1.33:8094");
    CHECK(reg.node_url("mad-lab-main").empty());
    CHECK(reg.node_url("nope").empty());
    const machine_entry * r = reg.find("mad-lab-2026");
    CHECK(r != nullptr && !r->local && r->ssh == "kmbandy@mad-lab-2026" && r->models_dir == "/mnt/nvme");
    CHECK(router_local_machine("tests/router-fixtures/machines/machines.json") == "mad-lab-main");

    // the other box's copy marks itself local; wrong-typed keys count as absent
    const machines_registry other = parse_machines(
        R"({"mad-lab-main": {"router_node": 8094, "local": "yes"}, "mad-lab-2026": {"local": true, "router_node": "http://127.0.0.1:8094"}})");
    CHECK(other.ok());
    CHECK(other.local_machine() == "mad-lab-2026");
    CHECK(other.node_url("mad-lab-main").empty());
    CHECK(other.node_url("mad-lab-2026") == "http://127.0.0.1:8094");

    // missing file / garbage: empty registry with an error, never a throw
    const machines_registry missing = load_machines("/nonexistent/machines.json");
    CHECK(!missing.ok() && missing.machines.empty() && missing.local_machine().empty());
    CHECK(!router_local_machine("/nonexistent/machines.json").empty()); // falls back to the hostname
    CHECK(!parse_machines("not json").ok());
    CHECK(!parse_machines("[1, 2]").ok());
    CHECK(router_parse_local_machine(R"({"a": {"local": false}})").empty());
}

//
// env: preset-style overrides (shared with the router's own child path) and the reserved strip
//

static bool env_has(const std::vector<std::string> & env, const std::string & key) {
    std::string v;
    return router_env_get(env, key, v);
}

static void test_env_overrides_and_reserved() {
    std::vector<std::string> env = { "A=0", "B=x", "B=y", "C=z" };
    router_env_apply_overrides(env, { "A=1", "-B", "A=2", "-", "-C=1", "NOEQ" }); // malformed ones skipped
    std::string v;
    CHECK(router_env_get(env, "A", v) && v == "2");
    CHECK(std::count_if(env.begin(), env.end(), [](const std::string & e) { return e.rfind("A=", 0) == 0; }) == 1);
    CHECK(!env_has(env, "B"));
    CHECK(router_env_get(env, "C", v) && v == "z");
    CHECK(router_env_override_error("K=V").empty() && router_env_override_error("-K").empty());
    CHECK(!router_env_override_error("").empty() && !router_env_override_error("-").empty());
    CHECK(!router_env_override_error("-K=V").empty() && !router_env_override_error("NOEQ").empty());
    CHECK(!router_env_override_error("=V").empty());

    CHECK(router_is_reserved_option_key("LLAMA_API_KEY"));
    CHECK(router_is_reserved_option_key("LLAMA_ARG_MODELS_DIR"));
    CHECK(router_is_reserved_option_key("LLAMA_ARG_ROUTER_GPU"));
    CHECK(router_is_reserved_option_key("LLAMA_ARG_ROUTER_NODE"));
    CHECK(!router_is_reserved_option_key("LLAMA_ARG_CTX_SIZE"));

#ifndef _WIN32
    // the node's base env drops the reserved names and its own address / model
    const char * stripped[] = { "LLAMA_API_KEY", "LLAMA_ARG_API_KEY_FILE", "LLAMA_ARG_SSL_KEY_FILE", "LLAMA_ARG_MODELS_DIR",
                                "LLAMA_ARG_ROUTER_GPU", "LLAMA_ARG_ROUTER_NODE", "LLAMA_ARG_NODE_BIND",
                                "LLAMA_ARG_BOARD_URL", "LLAMA_ARG_PORT", "LLAMA_ARG_HOST", "LLAMA_ARG_MODEL" };
    for (const char * k : stripped) {
        setenv(k, "leak", 1);
    }
    setenv("NODE_TEST_SAFE", "1", 1);
    setenv("LLAMA_ARG_CTX_SIZE", "4096", 1); // an ordinary child option still passes
    const std::vector<std::string> base = server_node_default_env();
    for (const char * k : stripped) {
        CHECK(!env_has(base, k));
        unsetenv(k);
    }
    CHECK(env_has(base, "NODE_TEST_SAFE") && env_has(base, "LLAMA_ARG_CTX_SIZE"));
    unsetenv("NODE_TEST_SAFE");
    unsetenv("LLAMA_ARG_CTX_SIZE");
#endif
}

//
// startup validation + token
//

static void test_params_and_token() {
    const fs::path dir = fs::temp_directory_path() / ("router-node-params-" + std::to_string((long long) now_ms()));
    fs::remove_all(dir);
    write_file(dir / "token", "  s3cret-token\n");
    write_file(dir / "empty", "\n  \n");

    std::string err;
    CHECK(server_node_read_token((dir / "token").string(), err) == "s3cret-token" && err.empty());
    CHECK(server_node_read_token((dir / "empty").string(), err).empty() && !err.empty());
    CHECK(server_node_read_token((dir / "missing").string(), err).empty() && !err.empty());

    common_params p;
    p.router_node     = true;
    p.node_token_file = (dir / "token").string();
    p.node_bind       = "127.0.0.1, 100.64.0.7";
    CHECK(server_node_check_params(p).empty());
    CHECK((server_node_bind_hosts(p.node_bind) == std::vector<std::string>{ "127.0.0.1", "100.64.0.7" }));

    // --router-node + --board-url: rejected, only the leader talks to the board
    {
        common_params q = p;
        q.router_board_url = "http://mad-lab-2026:18800";
        CHECK(has(server_node_check_params(q), "--board-url"));
    }
    {
        common_params q = p;
        q.node_token_file = (dir / "empty").string();
        CHECK(!server_node_check_params(q).empty());
        q.node_token_file = (dir / "missing").string();
        CHECK(!server_node_check_params(q).empty());
        q.node_token_file = "";
        CHECK(!server_node_check_params(q).empty());
    }
    {
        // wildcards are refused in any spelling (resolved, not string-matched)
        common_params q = p;
        for (const char * w : { "0.0.0.0", "0", "000.0.0.0", "::", "[::]", "::0", "[::0]", "0:0:0:0:0:0:0:0",
                                "::ffff:0.0.0.0", "*", "127.0.0.1,0.0.0.0" }) {
            q.node_bind = w;
            CHECK(has(server_node_check_params(q), "wildcard"));
        }
        q.node_bind = "";
        CHECK(!server_node_check_params(q).empty());
        q.node_bind = "no-such-host.invalid";
        CHECK(!server_node_check_params(q).empty());
        for (const char * ok : { "127.0.0.1", "::1", "[::1]", "localhost" }) {
            q.node_bind = ok;
            CHECK(server_node_check_params(q).empty());
        }
    }
    {
        common_params q = p;
        q.model.path = "/models/x.gguf";
        CHECK(!server_node_check_params(q).empty());
    }
    {
        common_params q; // not a node: nothing to check
        q.router_board_url = "http://x";
        CHECK(server_node_check_params(q).empty());
    }

    CHECK(server_node_token_matches("tok", "Bearer tok"));
    CHECK(server_node_token_matches("tok", "bearer tok"));
    CHECK(!server_node_token_matches("tok", "Bearer tok2"));
    CHECK(!server_node_token_matches("tok", "Bearer to"));
    CHECK(!server_node_token_matches("tok", "tok"));
    CHECK(!server_node_token_matches("tok", ""));
    CHECK(!server_node_token_matches("", "Bearer "));

    // /node/stop timeout_s: clamped to 0..3600, non-finite refused
    int t = -1;
    CHECK(server_node_clamp_timeout_s(1e12, t) && t == 3600);
    CHECK(server_node_clamp_timeout_s(-5, t) && t == 0);
    CHECK(server_node_clamp_timeout_s(2.7, t) && t == 2);
    CHECK(!server_node_clamp_timeout_s(std::numeric_limits<double>::infinity(), t));
    CHECK(!server_node_clamp_timeout_s(std::numeric_limits<double>::quiet_NaN(), t));

    CHECK(server_node_parse_signal(json(15)) == 15);
#ifndef _WIN32
    CHECK(server_node_parse_signal(json("SIGTERM")) == SIGTERM);
    CHECK(server_node_parse_signal(json("kill")) == SIGKILL);
#endif
    CHECK(server_node_parse_signal(json("9")) == 9);
    CHECK(server_node_parse_signal(json("NOPE")) == -1);
    CHECK(server_node_parse_signal(json(0)) == -1);

    fs::remove_all(dir);
}

#ifndef _WIN32

//
// helpers
//

// The test's own view, independent of the node's liveness logic: a zombie counts as dead. The
// "orphans" below are children of this test process, so they stay zombies until the test joins
// them; a real orphan is reparented to init / a subreaper and reaped there.
static bool pid_alive(int pid) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    if (!in.good()) {
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t rp = text.rfind(')');
    return rp != std::string::npos && rp + 2 < text.size() && text[rp + 2] != 'Z';
}

static int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port        = 0;
    CHECK(bind(fd, (sockaddr *) &a, sizeof(a)) == 0);
    socklen_t len = sizeof(a);
    getsockname(fd, (sockaddr *) &a, &len);
    const int port = ntohs(a.sin_port);
    close(fd);
    return port;
}

static bool have_python3() {
    return fs::exists("/usr/bin/env") && fs::exists("tests/router-fixtures/fake-worker.py") &&
           std::system("/usr/bin/env python3 -c 'pass' >/dev/null 2>&1") == 0;
}

static server_node_config test_config() {
    server_node_config cfg = server_node_default_config();
    cfg.heartbeat_ms      = 300;
    cfg.shutdown_grace_ms = 3000;
    return cfg;
}

// collects events from `cursor` until pred(all so far) holds or timeout_ms passes
template <typename P>
static bool wait_for_events(const server_node & node, uint64_t & cursor, std::vector<json> & all, P pred, int64_t timeout_ms) {
    const int64_t deadline = now_ms() + timeout_ms;
    while (now_ms() < deadline) {
        if (pred(all)) {
            return true;
        }
        std::vector<json> evs;
        node.wait_events(cursor, evs, 100);
        all.insert(all.end(), evs.begin(), evs.end());
    }
    return pred(all);
}

static bool saw_line(const std::vector<json> & evs, const std::string & name, const std::string & needle) {
    return std::any_of(evs.begin(), evs.end(), [&](const json & e) {
        return e.value("type", std::string()) == "line" && e.value("name", std::string()) == name &&
               has(e.value("line", std::string()), needle);
    });
}

static bool saw_status(const std::vector<json> & evs, const std::string & name, const std::string & status) {
    return std::any_of(evs.begin(), evs.end(), [&](const json & e) {
        return e.value("type", std::string()) == "child" && e.value("name", std::string()) == name &&
               e.value("status", std::string()) == status;
    });
}

static node_child_info find_child(const server_node & node, const std::string & name) {
    for (const auto & c : node.children()) {
        if (c.name == name) {
            return c;
        }
    }
    return {};
}

static int expect_error(const std::function<void()> & fn) {
    try {
        fn();
    } catch (const server_node_error & e) {
        return e.status;
    }
    return 0;
}

//
// core: spawn / stop / signal / state / events
//

static void test_core_spawn_stop_state() {
    server_node node(test_config());
    uint64_t cursor = node.next_seq();
    std::vector<json> evs;

    node_spawn_request r;
    r.name = "sh1";
    r.gen  = "gen-a";
    // extra words after the script are $0..; they carry the --port the node reports
    r.args = { "/bin/sh", "-c", "echo hello-node gen=$LLAMA_ROUTER_GEN child=$LLAMA_ROUTER_CHILD x=$EXTRA_VAR; exec sleep 60", "--port", "4567" };
    r.env  = { "EXTRA_VAR=42" };
    const node_child_info info = node.spawn(r);
    CHECK(info.pid > 0 && info.port == 4567 && info.status == "running" && !info.adopted);
    CHECK(pid_alive(info.pid));

    // the child's output arrives as a line event tagged with its name, env markers injected
    CHECK(wait_for_events(node, cursor, evs, [](const std::vector<json> & a) {
        return saw_line(a, "sh1", "hello-node gen=gen-a child=sh1 x=42");
    }, 5000));
    CHECK(saw_status(evs, "sh1", "running"));

    // state reports the child with its PID and its RAM
    const json st = node.state();
    bool found = false;
    for (const auto & c : st.at("children")) {
        if (c.at("name").get<std::string>() == "sh1") {
            found = true;
            CHECK(c.at("pid").get<int>() == info.pid);
            CHECK(c.at("status").get<std::string>() == "running");
            CHECK(c.at("rss_anon").is_number());
            CHECK(c.at("vram").is_array());
        }
    }
    CHECK(found);
    CHECK(st.at("mem_available").get<int64_t>() > 0);
    CHECK(st.at("devices").is_array() && st.at("vram").is_array() && st.at("orphans").is_array());

    // a live name cannot be spawned twice; bad requests are refused before anything runs
    CHECK(expect_error([&]() { node.spawn(r); }) == 409);
    node_spawn_request bad = r;
    bad.name = "rel";
    bad.args = { "sh", "-c", "true" };
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400); // relative argv[0]
    bad.args = { "/bin/true" };
    bad.env  = { "TMPDIR=/etc/passwd" };
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400); // temp-dir refusal
    bad.env  = { "NOEQUALS" };
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400);
    bad.env  = {};
    bad.gen  = "";
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400);
    CHECK(expect_error([&]() { node.stop("nope"); }) == 404);
    CHECK(expect_error([&]() { node.signal("nope", SIGTERM); }) == 404);

    // stop: sleep dies on SIGTERM, well before the kill deadline
    const node_child_info s = node.stop("sh1", 10);
    CHECK(s.status == "stopping" || s.status == "exited");
    CHECK(node.wait_exit("sh1", 5000));
    const node_child_info gone = find_child(node, "sh1");
    CHECK(gone.status == "exited" && !gone.killed);
    CHECK(!pid_alive(info.pid));
    CHECK(wait_for_events(node, cursor, evs, [](const std::vector<json> & a) { return saw_status(a, "sh1", "exited"); }, 2000));
    CHECK(expect_error([&]() { node.signal("sh1", SIGTERM); }) == 409);
    CHECK(node.stop("sh1").status == "exited"); // idempotent

    // the name is free again once its child exited
    r.args = { "/bin/sh", "-c", "exec sleep 60" };
    const node_child_info again = node.spawn(r);
    CHECK(again.pid > 0 && again.pid != info.pid);
    // node exit kills children: the destructor stops it
}

// -KEY unsets reach the child; the port is never read from env; a closing node refuses spawns
static void test_core_env_port_closing() {
    server_node_config cfg = test_config();
    cfg.base_env.push_back("NODE_TEST_UNSET=present");
    server_node node(cfg);
    uint64_t cursor = node.next_seq();
    std::vector<json> evs;

    node_spawn_request r;
    r.name = "envy";
    r.gen  = "gen-a";
    r.args = { "/bin/sh", "-c", "echo u=[$NODE_TEST_UNSET] k=[$KEEP]; exec sleep 60" };
    r.env  = { "-NODE_TEST_UNSET", "KEEP=1", "LLAMA_ARG_PORT=4321" };
    const node_child_info info = node.spawn(r);
    CHECK(info.port == 0); // LLAMA_ARG_PORT in env is not where the port comes from
    CHECK(wait_for_events(node, cursor, evs, [](const std::vector<json> & a) { return saw_line(a, "envy", "u=[] k=[1]"); }, 5000));

    node_spawn_request p = r;
    p.name = "ported";
    p.args = { "/bin/sh", "-c", "exec sleep 60" };
    p.env  = {};
    p.port = 5555;
    CHECK(node.spawn(p).port == 5555);

    node_spawn_request bad = p;
    bad.name = "bad";
    bad.port = 70000;
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400);
    bad.port = 0;
    bad.env  = { "-" };
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400);
    bad.env  = { "-A=B" };
    CHECK(expect_error([&]() { node.spawn(bad); }) == 400);

    // shutting down: no new children
    node.close_events();
    node_spawn_request late = p;
    late.name = "late";
    CHECK(expect_error([&]() { node.spawn(late); }) == 503);
    CHECK(find_child(node, "late").pid == 0);
}

static void test_core_destructor_kills_children() {
    int pid = 0;
    {
        server_node node(test_config());
        node_spawn_request r;
        r.name = "sleeper";
        r.gen  = "gen-a";
        r.args = { "/bin/sh", "-c", "exec sleep 60" };
        pid = node.spawn(r).pid;
        CHECK(pid_alive(pid));
    }
    CHECK(!pid_alive(pid));
}

static void test_core_signal_and_kill() {
    server_node node(test_config());
    uint64_t cursor = node.next_seq();
    std::vector<json> evs;

    // signal: the fake worker prints its stop-snapshot line on SIGTERM and exits 0
    node_spawn_request w;
    w.name = "worker";
    w.gen  = "gen-a";
    w.args = { "/usr/bin/env", "python3", "tests/router-fixtures/fake-worker.py", "--port", std::to_string(free_port()) };
    w.env  = { "WP_EXPERT_PARK_FILE=/tmp/node-test.park" };
    const node_child_info wi = node.spawn(w);
    CHECK(wi.port > 0);
    CHECK(wait_for_events(node, cursor, evs, [](const std::vector<json> & a) { return saw_line(a, "worker", "listening on"); }, 10000));
    CHECK(saw_line(evs, "worker", "LLAMA_ROUTER_GEN=gen-a"));
    node.signal("worker", SIGTERM);
    CHECK(wait_for_events(node, cursor, evs, [](const std::vector<json> & a) { return saw_status(a, "worker", "exited"); }, 10000));
    CHECK(saw_line(evs, "worker", "wp expert worker: stop snapshot written: /tmp/node-test.park (42 rows)"));
    const node_child_info wd = find_child(node, "worker");
    CHECK(wd.exit_code == 0 && !wd.killed);

    // stop of a child that ignores SIGTERM: SIGKILL once timeout_s passes
    node_spawn_request k = w;
    k.name = "stubborn";
    k.args = { "/usr/bin/env", "python3", "tests/router-fixtures/fake-worker.py", "--port", std::to_string(free_port()) };
    k.env  = { "FAKE_WORKER_MODE=ignore-term" };
    node.spawn(k);
    CHECK(wait_for_events(node, cursor, evs, [](const std::vector<json> & a) { return saw_line(a, "stubborn", "listening on"); }, 10000));
    const int64_t t0 = now_ms();
    node.stop("stubborn", 1, "term");
    CHECK(node.wait_exit("stubborn", 8000));
    CHECK(now_ms() - t0 >= 900);
    CHECK(find_child(node, "stubborn").killed);
}

//
// orphans: stale-generation sweep on node start, adoption by (name, gen)
//

// spawns `sleep 60` with extra env; the caller joins it
static void spawn_tagged_sleep(common_subproc & proc, const std::vector<std::string> & extra_env) {
    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        env.emplace_back(*e);
    }
    for (const auto & kv : extra_env) {
        router_env_set(env, kv.substr(0, kv.find('=')), kv.substr(kv.find('=') + 1));
    }
    CHECK(proc.create({ "/bin/sh", "-c", "exec sleep 60" }, 0, env));
}

// copies the real /proc/<pid>/{status,environ,cmdline} of a live process into a fake tree
static void mirror_proc(const fs::path & root, int pid) {
    for (const char * f : { "status", "environ", "cmdline" }) {
        std::ifstream in("/proc/" + std::to_string(pid) + "/" + f, std::ios::binary);
        CHECK(in.good());
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        write_file(root / "proc" / std::to_string(pid) / f, text);
    }
}

static void test_orphan_sweep_and_adopt() {
    common_subproc keep;
    common_subproc stale;
    common_subproc plain;
    // their router (LLAMA_ROUTER_PID) is not an ancestor: a dead node left them behind
    spawn_tagged_sleep(keep,  { "LLAMA_ROUTER_GEN=gen-old", "LLAMA_ROUTER_PID=2147480000", "LLAMA_ROUTER_CHILD=w-keep" });
    spawn_tagged_sleep(stale, { "LLAMA_ROUTER_GEN=gen-old", "LLAMA_ROUTER_PID=2147480000", "LLAMA_ROUTER_CHILD=w-old" });
    spawn_tagged_sleep(plain, { "SOMETHING=else" }); // not a router child: never touched
    const int keep_pid  = keep.pid();
    const int stale_pid = stale.pid();
    const int plain_pid = plain.pid();
    std::this_thread::sleep_for(std::chrono::milliseconds(150)); // let exec finish so environ is sleep's

    // only processes described in the fake tree are candidates: nothing else can be touched
    const fs::path root = fs::temp_directory_path() / ("router-node-orphans-" + std::to_string((long) getpid()));
    fs::remove_all(root);
    mirror_proc(root, keep_pid);
    mirror_proc(root, stale_pid);
    mirror_proc(root, plain_pid);

    {
        server_node_config cfg = test_config();
        cfg.proc_root            = root.string();
        cfg.adopt_window_ms      = 1500;
        cfg.orphan_kill_grace_ms = 3000;
        server_node node(cfg);
        uint64_t cursor = node.next_seq();
        std::vector<json> evs;

        CHECK(node.collect_orphans() == 2);
        const auto orph = node.orphans();
        CHECK(orph.size() == 2);
        CHECK(std::all_of(orph.begin(), orph.end(), [](const node_orphan_info & o) { return o.gen == "gen-old" && o.state == "waiting"; }));

        // adoption needs the right (name, gen)
        CHECK(expect_error([&]() { node.adopt("w-keep", "gen-new"); }) == 404);
        const node_child_info a = node.adopt("w-keep", "gen-old");
        CHECK(a.pid == keep_pid && a.adopted && a.status == "running" && a.gen == "gen-old");
        CHECK(expect_error([&]() { node.adopt("w-keep", "gen-old"); }) == 409);

        // the one nobody adopted is stopped once the window closes
        CHECK(wait_for_events(node, cursor, evs, [&](const std::vector<json> &) { return !pid_alive(stale_pid); }, 8000));
        CHECK(std::any_of(evs.begin(), evs.end(), [&](const json & e) {
            return e.value("type", std::string()) == "orphan" && e.value("action", std::string()) == "term" &&
                   e.value("pid", 0) == stale_pid;
        }));
        CHECK(pid_alive(keep_pid));
        CHECK(pid_alive(plain_pid));

        // the adopted one is driven like a child: state shows it, stop signals it
        // keep the state alive: a range-for over node.state().at(...) would iterate a dangling
        // reference into a destroyed temporary
        const json adopted_state = node.state();
        bool listed = false;
        for (const auto & c : adopted_state.at("children")) {
            listed = listed || (c.at("name").get<std::string>() == "w-keep" && c.at("pid").get<int>() == keep_pid &&
                                c.at("adopted").get<bool>());
        }
        CHECK(listed);
        node.stop("w-keep", 3);
        CHECK(node.wait_exit("w-keep", 5000));
        CHECK(!pid_alive(keep_pid));
        CHECK(find_child(node, "w-keep").exit_code == -1);
    }

    keep.join();
    stale.join();
    CHECK(pid_alive(plain_pid));
    plain.terminate();
    plain.join();
    fs::remove_all(root);
}

//
// HTTP layer, in-process on a random localhost port
//

static void test_http() {
    server_node_config cfg = test_config();
    cfg.base_env.push_back("NODE_HTTP_A=1");
    cfg.base_env.push_back("NODE_HTTP_B=2");
    server_node node(cfg);
    server_node_routes routes(node, "node-token");

    common_params params; // must outlive the server (its middleware refers to it)
    params.hostnames      = { "127.0.0.1" };
    params.port           = 0;
    params.ui             = false;
    params.n_threads_http = 4;
    server_http_context http;
    CHECK(http.init(params));
    routes.register_routes(http);
    CHECK(http.start());
    http.is_ready.store(true);
    const int port = http.port;
    CHECK(port > 0);

    httplib::Client cli("127.0.0.1", port);
    cli.set_read_timeout(10, 0);
    const httplib::Headers auth  = { { "Authorization", "Bearer node-token" } };
    const httplib::Headers wrong = { { "Authorization", "Bearer node-tokem" } };

    // every route wants the token
    for (const char * path : { "/node/state", "/node/events" }) {
        auto r1 = cli.Get(path);
        CHECK(r1 && r1->status == 401);
        auto r2 = cli.Get(path, wrong);
        CHECK(r2 && r2->status == 401);
    }
    for (const char * path : { "/node/spawn", "/node/stop", "/node/signal", "/node/adopt" }) {
        auto r1 = cli.Post(path, "{}", "application/json");
        CHECK(r1 && r1->status == 401);
        auto r2 = cli.Post(path, wrong, "{}", "application/json");
        CHECK(r2 && r2->status == 401);
    }
    CHECK(node.children().empty()); // nothing ran

    const uint64_t since = node.next_seq();

    // spawn
    json body = json::object();
    body["name"] = "web1";
    body["gen"]  = "gen-h";
    body["args"] = json::array({ "/bin/sh", "-c", "echo hello-over-http $GREETING; exec sleep 60" });
    json env = json::object();
    env["GREETING"] = "hi";
    body["env"] = env;
    auto sp = cli.Post("/node/spawn", auth, body.dump(), "application/json");
    CHECK(sp && sp->status == 200);
    const json spj = json::parse(sp->body);
    const int pid = spj.at("pid").get<int>();
    CHECK(pid > 0 && spj.at("port").get<int>() == 0 && spj.at("status").get<std::string>() == "running");

    // bad requests answer 400 / 409 with a JSON error
    auto dup = cli.Post("/node/spawn", auth, body.dump(), "application/json");
    CHECK(dup && dup->status == 409);
    json rel = body;
    rel["name"] = "rel";
    rel["args"] = json::array({ "sh", "-c", "true" });
    auto relr = cli.Post("/node/spawn", auth, rel.dump(), "application/json");
    CHECK(relr && relr->status == 400 && has(relr->body, "absolute"));
    auto junk = cli.Post("/node/spawn", auth, "not json", "application/json");
    CHECK(junk && junk->status == 400);

    // state reports it with its PID
    auto st = cli.Get("/node/state", auth);
    CHECK(st && st->status == 200);
    const json stj = json::parse(st->body);
    CHECK(stj.at("children").size() == 1);
    CHECK(stj.at("children")[0].at("pid").get<int>() == pid);
    CHECK(stj.at("children")[0].at("name").get<std::string>() == "web1");

    // events: replayed from `since`, carrying the child's output line, and heartbeats
    std::string stream;
    int heartbeats = 0;
    bool line_seen = false;
    auto ev = cli.Get("/node/events?since=" + std::to_string(since), auth, [&](const char * data, size_t len) {
        stream.append(data, len);
        size_t pos;
        while ((pos = stream.find("\n\n")) != std::string::npos) {
            const std::string chunk = stream.substr(0, pos);
            stream.erase(0, pos + 2);
            if (chunk.rfind("data: ", 0) != 0) {
                continue;
            }
            const json e = json::parse(chunk.substr(6));
            const std::string type = e.value("type", std::string());
            if (type == "heartbeat") {
                heartbeats++;
                CHECK(e.contains("next_seq") && e.at("children").is_array());
            } else if (type == "line" && e.value("name", std::string()) == "web1" &&
                       has(e.value("line", std::string()), "hello-over-http hi")) {
                line_seen = true;
            }
        }
        return !(line_seen && heartbeats >= 2); // the second heartbeat proves they repeat
    });
    CHECK(line_seen && heartbeats >= 2);

    // signal, then stop with wait
    json sig = json::object();
    sig["name"] = "web1";
    sig["sig"]  = "SIGUSR2"; // sh exec'd sleep: default action terminates it
    auto sg = cli.Post("/node/signal", auth, sig.dump(), "application/json");
    CHECK(sg && sg->status == 200);
    CHECK(node.wait_exit("web1", 5000));
    json stop = json::object();
    stop["name"] = "web1";
    stop["wait"] = true;
    auto so = cli.Post("/node/stop", auth, stop.dump(), "application/json");
    CHECK(so && so->status == 200 && json::parse(so->body).at("status").get<std::string>() == "exited");
    auto sg2 = cli.Post("/node/signal", auth, sig.dump(), "application/json");
    CHECK(sg2 && sg2->status == 409);

    // unsets over HTTP: "-K" in the array form, null in the object form
    {
        const uint64_t s2 = node.next_seq();
        json b = json::object();
        b["name"] = "unset";
        b["gen"]  = "gen-h";
        b["args"] = json::array({ "/bin/sh", "-c", "echo a=[$NODE_HTTP_A] b=[$NODE_HTTP_B]" });
        b["env"]  = json::array({ "-NODE_HTTP_A" });
        auto r1 = cli.Post("/node/spawn", auth, b.dump(), "application/json");
        CHECK(r1 && r1->status == 200);
        json eo = json::object();
        eo["NODE_HTTP_B"] = nullptr;
        b["name"] = "unset2";
        b["env"]  = eo;
        auto r2 = cli.Post("/node/spawn", auth, b.dump(), "application/json");
        CHECK(r2 && r2->status == 200);
        uint64_t c2 = s2;
        std::vector<json> got;
        CHECK(wait_for_events(node, c2, got, [](const std::vector<json> & a) {
            return std::any_of(a.begin(), a.end(), [](const json & e) {
                       return e.value("name", std::string()) == "unset" && has(e.value("line", std::string()), "a=[] b=[2]");
                   }) &&
                   std::any_of(a.begin(), a.end(), [](const json & e) {
                       return e.value("name", std::string()) == "unset2" && has(e.value("line", std::string()), "a=[1] b=[]");
                   });
        }, 5000));
    }

    // spawn + stop over HTTP with wait; a huge timeout_s is clamped, not overflowed
    body["name"] = "web2";
    body["args"] = json::array({ "/bin/sh", "-c", "exec sleep 60" });
    auto sp2 = cli.Post("/node/spawn", auth, body.dump(), "application/json");
    CHECK(sp2 && sp2->status == 200);
    const int pid2 = json::parse(sp2->body).at("pid").get<int>();
    stop["name"]      = "web2";
    stop["timeout_s"] = 1e12;
    auto so2 = cli.Post("/node/stop", auth, stop.dump(), "application/json");
    CHECK(so2 && so2->status == 200 && json::parse(so2->body).at("status").get<std::string>() == "exited");
    CHECK(!pid_alive(pid2));

    auto unknown = cli.Post("/node/stop", auth, R"({"name": "ghost"})", "application/json");
    CHECK(unknown && unknown->status == 404);

    node.close_events();
    http.stop();
    http.join();
}

#endif // !_WIN32

int main() {
#ifndef _WIN32
    // as llama-server does: a stop's exit command may hit a child that just closed its stdin
    signal(SIGPIPE, SIG_IGN);
#endif
    test_machines();
    test_env_overrides_and_reserved();
    test_params_and_token();
#ifndef _WIN32
    test_core_spawn_stop_state();
    test_core_destructor_kills_children();
    test_core_env_port_closing();
    test_orphan_sweep_and_adopt();
    test_http();
    if (have_python3()) {
        test_core_signal_and_kill();
    } else {
        fprintf(stderr, "python3 not found: skipping the fake-worker tests\n");
    }
#endif
    fprintf(stderr, "test-router-node: OK\n");
    return 0;
}
