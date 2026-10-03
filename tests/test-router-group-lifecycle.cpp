// Tests for the router's model-group lifecycle (tools/server/server-router-group-lifecycle.cpp):
// worker output classification, launch parsing, env injection, the stale-generation sweep
// (fake /proc tree), and router_worker_group driven against a fake worker
// (tests/router-fixtures/fake-worker.py: no GPU, no model).
//
// The spine side (server_models::load / on_child_exit) needs a real model, so it has no
// end-to-end test here; it is checked live by the controller.

#include "server-router-group-lifecycle.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
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

#undef NDEBUG
#include <cassert>

namespace fs = std::filesystem;

static bool has(const std::string & s, const std::string & needle) {
    return s.find(needle) != std::string::npos;
}

static void test_classify() {
    auto c = router_classify_worker_line("wp expert worker: stop snapshot written: /var/park/w.park (1234 rows)\n");
    assert(c.kind == ROUTER_WORKER_LINE_SNAPSHOT_WRITTEN);
    assert(c.detail == "/var/park/w.park (1234 rows)");

    // a log prefix in front is fine
    c = router_classify_worker_line("[ 8801] wp expert worker: stop snapshot FAILED: ENOSPC");
    assert(c.kind == ROUTER_WORKER_LINE_SNAPSHOT_FAILED);
    assert(c.detail == "ENOSPC");

    c = router_classify_worker_line("wp expert worker: stop snapshot: quiesce timeout after 30000 ms, snapshotting anyway");
    assert(c.kind == ROUTER_WORKER_LINE_SNAPSHOT_TIMEOUT);
    assert(c.detail == "30000 ms");

    c = router_classify_worker_line("ggml_cuda_init: Hip error: hipErrorOutOfMemory");
    assert(c.kind == ROUTER_WORKER_LINE_HIP_ERROR);

    c = router_classify_worker_line("expert worker listening on 0.0.0.0:8801");
    assert(c.kind == ROUTER_WORKER_LINE_OTHER);

    assert(std::string(router_snapshot_result_str(ROUTER_SNAPSHOT_WRITTEN)) == "written");
    assert(std::string(router_snapshot_result_str(ROUTER_SNAPSHOT_FAILED)) == "FAILED");
    assert(std::string(router_snapshot_result_str(ROUTER_SNAPSHOT_TIMEOUT)) == "timeout");
    assert(std::string(router_snapshot_result_str(ROUTER_SNAPSHOT_NONE)) == "none");
}

static void test_launch() {
    router_launch l;
    std::string host;

    // the real worker's shape: no shell, --listen HOST:PORT
    assert(router_parse_launch("/opt/bin/llama-wp-expert-worker --shard-manifest /m.json --device ROCm0 "
                               "--slots 1650 --listen 0.0.0.0:8801", l).empty());
    assert(!l.via_shell);
    assert(l.argv.size() == 9 && l.argv[0] == "/opt/bin/llama-wp-expert-worker");
    assert(router_launch_endpoint(l.words, host) == 8801 && host == "127.0.0.1");

    // quotes, escapes and leading assignments
    assert(router_parse_launch("A=1 B='x y' /bin/w --name \"two words\" it\\'s --port=9002", l).empty());
    assert((l.env == std::vector<std::string>{ "A=1", "B=x y" }));
    assert((l.argv == std::vector<std::string>{ "/bin/w", "--name", "two words", "it's", "--port=9002" }));
    assert(router_launch_endpoint(l.words, host) == 9002 && host == "127.0.0.1");

    // expansion goes through the shell, with exec so the PID stays the worker's
    assert(router_parse_launch("$HOME/bin/w --listen 10.0.0.5:7001", l).empty());
    assert(l.via_shell);
    assert((l.argv == std::vector<std::string>{ "/bin/sh", "-c", "exec $HOME/bin/w --listen 10.0.0.5:7001" }));
    assert(router_launch_endpoint(l.words, host) == 7001 && host == "10.0.0.5");

    // anything that is not a single command is refused
    assert(!router_parse_launch("/bin/w > /tmp/log 2>&1", l).empty());
    assert(!router_parse_launch("/bin/w | tee x", l).empty());
    assert(!router_parse_launch("/bin/a; /bin/b", l).empty());
    assert(!router_parse_launch("/bin/w 'unterminated", l).empty());
    assert(!router_parse_launch("A=1", l).empty());

    // no port to be found
    assert(router_parse_launch("/bin/w --shard 0", l).empty());
    assert(router_launch_endpoint(l.words, host) == -1);
}

static void test_env() {
    std::vector<std::string> env = { "PATH=/bin", "WP_EXPERT_PARK_FILE=/old", "X=1" };
    env = router_worker_env(env, { "X=2" }, "/var/park/w.park", "gen-1");
    std::string v;
    assert(router_env_get(env, "WP_EXPERT_PARK_FILE", v) && v == "/var/park/w.park");
    assert(router_env_get(env, "WP_EXPERT_SEED_FROM_PARK", v) && v == "1");
    assert(router_env_get(env, "LLAMA_ROUTER_GEN", v) && v == "gen-1");
    assert(router_env_get(env, "X", v) && v == "2");
    assert(std::count_if(env.begin(), env.end(), [](const std::string & e) { return e.rfind("WP_EXPERT_PARK_FILE=", 0) == 0; }) == 1);

    // no park file: nothing park-related is injected, the generation always is
    env = router_worker_env({ "PATH=/bin" }, {}, "", "gen-2");
    assert(!router_env_get(env, "WP_EXPERT_PARK_FILE", v));
    assert(!router_env_get(env, "WP_EXPERT_SEED_FROM_PARK", v));
    assert(router_env_get(env, "LLAMA_ROUTER_GEN", v) && v == "gen-2");

    assert(router_env_quiesce_ms({ "PATH=/bin" }) == 30000);
    assert(router_env_quiesce_ms({ "WP_EXPERT_PARK_QUIESCE_MS=5000" }) == 5000);
    assert(router_env_quiesce_ms({ "WP_EXPERT_PARK_QUIESCE_MS=junk" }) == 30000);

    const std::string g1 = router_generation_new();
    const std::string g2 = router_generation_new();
    assert(g1.size() == 36 && g1 != g2 && g1[14] == '4');
}

#ifndef _WIN32

static void write_file(const fs::path & p, const std::string & content) {
    fs::create_directories(p.parent_path());
    std::ofstream f(p, std::ios::binary);
    f << content;
}

static std::string environ_blob(const std::vector<std::string> & kv) {
    std::string out;
    for (const auto & e : kv) {
        out += e;
        out.push_back('\0');
    }
    return out;
}

static void fake_proc(const fs::path & root, int pid, int ppid, unsigned uid, const std::vector<std::string> * env) {
    const fs::path dir = root / "proc" / std::to_string(pid);
    write_file(dir / "status", "Name:\tfake\nState:\tS (sleeping)\nPPid:\t" + std::to_string(ppid) +
                               "\nUid:\t" + std::to_string(uid) + "\t" + std::to_string(uid) + "\t" +
                               std::to_string(uid) + "\t" + std::to_string(uid) + "\n");
    if (env) {
        write_file(dir / "environ", environ_blob(*env));
    }
}

static void test_find_stale() {
    const fs::path root = fs::temp_directory_path() / ("router-sweep-" + std::to_string(getpid()));
    fs::remove_all(root);
    const unsigned me = 1000;
    const std::vector<std::string> old_gen_dead = { "PATH=/bin", "LLAMA_ROUTER_GEN=old", "LLAMA_ROUTER_PID=999" };
    const std::vector<std::string> mine         = { "LLAMA_ROUTER_GEN=new", "LLAMA_ROUTER_PID=50" };
    const std::vector<std::string> plain        = { "PATH=/bin" };
    const std::vector<std::string> old_gen_live = { "LLAMA_ROUTER_GEN=old", "LLAMA_ROUTER_PID=200" };

    fake_proc(root, 1,   0,   0,  &plain);
    fake_proc(root, 50,  1,   me, &mine);         // this router (self)
    fake_proc(root, 101, 1,   me, &old_gen_dead); // orphan of a dead router: stale
    fake_proc(root, 102, 50,  me, &mine);         // our own child
    fake_proc(root, 103, 1,   0,  &old_gen_dead); // someone else's process
    fake_proc(root, 104, 1,   me, &plain);        // not a router child
    fake_proc(root, 200, 1,   me, &plain);        // another router, alive
    fake_proc(root, 105, 200, me, &old_gen_live); // its child
    fake_proc(root, 106, 105, me, &old_gen_live); // its grandchild (a worker's own subprocess)
    fake_proc(root, 107, 1,   me, nullptr);       // environ unreadable: skipped
    fake_proc(root, 108, 1,   me, &old_gen_dead); // second orphan: stale
    write_file(root / "proc" / "meminfo", "MemTotal: 1 kB\n");

    const std::vector<int> stale = router_find_stale_children(root.string(), "new", me, 50);
    assert((stale == std::vector<int>{ 101, 108 }));

    // as the old generation: its own children are not stale, and 102 (another generation) is
    // still owned by a live router that is its ancestor, so nothing qualifies
    const std::vector<int> as_old = router_find_stale_children(root.string(), "old", me, 50);
    assert(as_old.empty());

    fs::remove_all(root);
}

// spawns `sleep 60` with the given env; the caller joins it
static void spawn_sleep(common_subproc & proc, const std::vector<std::string> & extra_env) {
    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        env.emplace_back(*e);
    }
    for (const auto & kv : extra_env) {
        router_env_set(env, kv.substr(0, kv.find('=')), kv.substr(kv.find('=') + 1));
    }
    const bool ok = proc.create({ "sleep", "60" }, subprocess_option_search_user_path, env);
    assert(ok);
}

// copies the real /proc/<pid>/{status,environ} of a live process into a fake tree
static void mirror_proc(const fs::path & root, int pid) {
    for (const char * f : { "status", "environ" }) {
        std::ifstream in("/proc/" + std::to_string(pid) + "/" + f, std::ios::binary);
        assert(in.good());
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        write_file(root / "proc" / std::to_string(pid) / f, text);
    }
}

static bool pid_alive(int pid) {
    std::ifstream in("/proc/" + std::to_string(pid) + "/stat");
    if (!in.good()) {
        return false;
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t rp = text.rfind(')');
    return rp != std::string::npos && rp + 2 < text.size() && text[rp + 2] != 'Z';
}

// a real process left behind by an older generation is killed; a current one is not.
// Only processes described in the fake tree are candidates, so nothing else on this
// machine can be touched by the test.
static void test_sweep_kills_stale_child() {
    common_subproc stale;
    common_subproc current;
    // LLAMA_ROUTER_PID names a process that is not this child's ancestor: its router is gone
    spawn_sleep(stale,   { "LLAMA_ROUTER_GEN=gen-old", "LLAMA_ROUTER_PID=2147480000" });
    spawn_sleep(current, { "LLAMA_ROUTER_GEN=gen-now", "LLAMA_ROUTER_PID=2147480000" });
    const int stale_pid   = stale.pid();
    const int current_pid = current.pid();
    std::this_thread::sleep_for(std::chrono::milliseconds(100)); // let exec finish so environ is sleep's

    const fs::path root = fs::temp_directory_path() / ("router-sweep-real-" + std::to_string(getpid()));
    fs::remove_all(root);
    mirror_proc(root, stale_pid);
    mirror_proc(root, current_pid);

    const std::vector<int> killed = router_sweep_stale_children(root.string(), "gen-now", (unsigned) getuid(), (int) getpid(), 5000);
    assert((killed == std::vector<int>{ stale_pid }));
    assert(!pid_alive(stale_pid));
    stale.join();
    assert(pid_alive(current_pid));
    current.terminate();
    current.join();
    fs::remove_all(root);
}

//
// router_worker_group against the fake worker
//

static int free_port() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{};
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    a.sin_port        = 0;
    assert(bind(fd, (sockaddr *) &a, sizeof(a)) == 0);
    socklen_t len = sizeof(a);
    getsockname(fd, (sockaddr *) &a, &len);
    const int port = ntohs(a.sin_port);
    close(fd);
    return port;
}

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static router_worker_spec fake_spec(const std::string & name, const std::vector<std::string> & extra_env,
                                    const std::string & park_file = "") {
    router_worker_spec s;
    s.name = name;
    s.port = free_port();
    s.argv = { "python3", "tests/router-fixtures/fake-worker.py", "--listen", "0.0.0.0:" + std::to_string(s.port) };
    std::vector<std::string> env;
    for (char ** e = environ; *e; e++) {
        env.emplace_back(*e);
    }
    std::vector<std::string> launch_env = extra_env;
    s.env               = router_worker_env(env, launch_env, park_file, "gen-test");
    s.quiesce_ms        = router_env_quiesce_ms(s.env);
    s.startup_timeout_s = 20;
    return s;
}

struct recorder {
    std::mutex                                 mu;
    std::vector<std::string>                   lines;
    std::vector<std::pair<std::string, int>>   unexpected;
    std::atomic<int>                           stopped{0};

    router_worker_group::callbacks callbacks() {
        router_worker_group::callbacks cb;
        cb.on_line = [this](const std::string & w, const std::string & line) {
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(w + ": " + line);
        };
        cb.on_unexpected_exit = [this](const std::string & w, int code, const std::string &) {
            std::lock_guard<std::mutex> lk(mu);
            unexpected.emplace_back(w, code);
        };
        cb.on_stopped = [this]() { stopped++; };
        return cb;
    }
    bool saw(const std::string & needle) {
        std::lock_guard<std::mutex> lk(mu);
        return std::any_of(lines.begin(), lines.end(), [&](const std::string & l) { return has(l, needle); });
    }
};

static bool gone(int pid) {
    return pid > 0 && !pid_alive(pid);
}

// load order: start() returns only once every worker accepts TCP (the spine is spawned after
// start() returns), and a TERM stop records the "written" stop-snapshot line
static void test_group_start_order_and_term() {
    recorder rec;
    auto slow = fake_spec("w-slow", { "FAKE_WORKER_LISTEN_DELAY_MS=800" }, "/tmp/w-slow.park");
    auto fast = fake_spec("w-fast", {}, "/tmp/w-fast.park");
    const int slow_port = slow.port;
    const int fast_port = fast.port;
    router_worker_group g("g1", { slow, fast }, rec.callbacks());

    std::string err;
    const int64_t t0 = now_ms();
    assert(g.start(err));
    assert(err.empty());
    assert(now_ms() - t0 >= 800);
    // the moment start() returns -- when the router would spawn the spine -- both accept TCP
    assert(router_tcp_accepts("127.0.0.1", slow_port, 500));
    assert(router_tcp_accepts("127.0.0.1", fast_port, 500));
    for (const auto & w : g.status()) {
        assert(w.state == "ready" && w.pid > 0);
    }
    assert(g.pids().size() == 2);
    // router-injected env reached the worker
    assert(rec.saw("w-slow: fake worker: env WP_EXPERT_PARK_FILE=/tmp/w-slow.park WP_EXPERT_SEED_FROM_PARK=1 LLAMA_ROUTER_GEN=gen-test"));

    const auto before = g.status();
    g.request_stop();
    assert(g.wait_stopped(10000));
    assert(rec.stopped == 1);
    assert(rec.unexpected.empty()); // a requested stop is not a failure
    for (const auto & w : g.status()) {
        assert(w.state == "exited");
        assert(w.exit_code == 0);
        assert(!w.killed);
        assert(w.snapshot == ROUTER_SNAPSHOT_WRITTEN);
        assert(has(w.snapshot_detail, ".park (42 rows)"));
    }
    for (const auto & w : before) {
        assert(gone(w.pid));
    }
    assert(g.pids().empty());
}

// a worker that prints "Hip error" fails the start; nothing it started survives
static void test_group_hip_error() {
    recorder rec;
    auto ok  = fake_spec("w-ok", {});
    auto hip = fake_spec("w-hip", { "FAKE_WORKER_MODE=hip" });
    router_worker_group g("g2", { ok, hip }, rec.callbacks());
    std::string err;
    assert(!g.start(err));
    assert(has(err, "w-hip") && has(err, "Hip error"));
    assert(g.all_exited());
    for (const auto & w : g.status()) {
        assert(w.state == "exited");
        assert(gone(w.pid));
    }
    assert(rec.stopped == 0);
    assert(rec.unexpected.empty());
}

// a worker that ignores TERM is SIGKILLed after quiesce + grace
static void test_group_kill_after_bound() {
    recorder rec;
    auto stubborn = fake_spec("w-stubborn", { "FAKE_WORKER_MODE=ignore-term", "WP_EXPERT_PARK_QUIESCE_MS=200" });
    assert(stubborn.quiesce_ms == 200);
    router_worker_group g("g3", { stubborn }, rec.callbacks(), /*kill_grace_ms=*/300);
    std::string err;
    assert(g.start(err));
    const int pid = g.status()[0].pid;
    const int64_t t0 = now_ms();
    g.request_stop();
    assert(g.wait_stopped(10000));
    assert(now_ms() - t0 >= 500);
    const auto w = g.status()[0];
    assert(w.killed);
    assert(w.snapshot == ROUTER_SNAPSHOT_NONE);
    assert(gone(pid));
    assert(rec.stopped == 1);
}

// a worker dying after the group is up is reported (the router then stops the spine);
// the rest of the group is stopped by request_stop()
static void test_group_death_mid_serve() {
    recorder rec;
    auto dying = fake_spec("w-dying", { "FAKE_WORKER_MODE=die", "FAKE_WORKER_DIE_AFTER_MS=1500" });
    auto other = fake_spec("w-other", {});
    router_worker_group g("g4", { dying, other }, rec.callbacks());
    std::string err;
    assert(g.start(err));
    const int64_t deadline = now_ms() + 10000;
    while (now_ms() < deadline) {
        {
            std::lock_guard<std::mutex> lk(rec.mu);
            if (!rec.unexpected.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    {
        std::lock_guard<std::mutex> lk(rec.mu);
        assert(rec.unexpected.size() == 1);
        assert(rec.unexpected[0].first == "w-dying" && rec.unexpected[0].second == 3);
    }
    g.request_stop();
    assert(g.wait_stopped(10000));
    assert(rec.stopped == 1);
    for (const auto & w : g.status()) {
        assert(gone(w.pid));
    }
}

// startup timeout, a spawn that cannot work, a port somebody else holds, and cancellation
static void test_group_start_failures() {
    {
        recorder rec;
        auto late = fake_spec("w-late", { "FAKE_WORKER_LISTEN_DELAY_MS=5000" });
        late.startup_timeout_s = 1;
        router_worker_group g("g5", { late }, rec.callbacks());
        std::string err;
        assert(!g.start(err));
        assert(has(err, "within 1 s"));
        assert(gone(g.status()[0].pid));
    }
    {
        recorder rec;
        auto bad = fake_spec("w-bad", {});
        bad.argv = { "/nonexistent/llama-wp-expert-worker", "--port", std::to_string(bad.port) };
        router_worker_group g("g6", { bad }, rec.callbacks());
        std::string err;
        assert(!g.start(err));
        assert(has(err, "w-bad"));
        assert(g.all_exited());
    }
    {
        recorder rec;
        auto busy = fake_spec("w-busy", {});
        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{};
        a.sin_family      = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port        = htons((uint16_t) busy.port);
        assert(bind(fd, (sockaddr *) &a, sizeof(a)) == 0 && listen(fd, 4) == 0);
        router_worker_group g("g7", { busy }, rec.callbacks());
        std::string err;
        assert(!g.start(err));
        assert(has(err, "already accepts"));
        assert(g.status()[0].pid == 0); // nothing was spawned
        close(fd);
    }
    {
        recorder rec;
        auto slow = fake_spec("w-cancel", { "FAKE_WORKER_LISTEN_DELAY_MS=5000" });
        router_worker_group g("g8", { slow }, rec.callbacks());
        const int64_t t0 = now_ms();
        std::string err;
        assert(!g.start(err, [t0]() { return now_ms() - t0 > 300; }));
        assert(has(err, "cancelled"));
        assert(gone(g.status()[0].pid));
    }
}

static bool have_python3() {
    return std::system("python3 -c 'pass' >/dev/null 2>&1") == 0;
}

#endif // !_WIN32

int main() {
    test_classify();
    test_launch();
    test_env();
#ifndef _WIN32
    test_find_stale();
    test_sweep_kills_stale_child();
    if (have_python3()) {
        test_group_start_order_and_term();
        test_group_hip_error();
        test_group_kill_after_bound();
        test_group_death_mid_serve();
        test_group_start_failures();
    } else {
        fprintf(stderr, "python3 not found: skipping the fake-worker tests\n");
    }
#endif
    return 0;
}
