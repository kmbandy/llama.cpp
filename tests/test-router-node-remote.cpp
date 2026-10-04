// The router's remote node link (tools/server/server-router-node-client.cpp) against the REAL node
// daemon HTTP layer (server_node + server_node_routes, tools/server/server-node.cpp), in process on
// a random localhost port with a bearer token: spawn / output / exit / stop / signal over HTTP,
// the port the node allocates, 4xx refusals that leave nothing behind, the same-name 409 window,
// and the event stream reconnecting from the last seq after a forced disconnect without losing
// the exit. A small TCP forwarder between link and node lets the test cut the connection.

#include "server-node.h"
#include "server-router-node-client.h"
#include "server-router-group-lifecycle.h"

#include "common.h"

#include <cpp-httplib/httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <map>
#include <string>
#include <thread>
#include <vector>

#undef NDEBUG
#include <cassert>

#ifndef _WIN32

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

static const char * TOKEN = "remote-test-token";
static const char * GEN   = "gen-remote-test";

static int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

template <typename P>
static bool wait_until(P pred, int ms) {
    const int64_t deadline = now_ms() + ms;
    while (now_ms() < deadline) {
        if (pred()) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return pred();
}

struct watch_log {
    std::mutex               mu;
    std::vector<std::string> lines;
    int                      exits     = 0;
    int                      exit_code = -99;
    int                      pid       = 0;
    int                      spawn_port = -1; // the port on_spawn saw (-1: on_spawn never ran)
    int                      spawn_pid  = 0;

    router_node_watch watch() {
        router_node_watch w;
        w.on_spawn = [this](const node_child_info & info) {
            std::lock_guard<std::mutex> lk(mu);
            spawn_port = info.port;
            spawn_pid  = info.pid;
        };
        w.on_line = [this](const std::string & line) {
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(line);
        };
        w.on_exit = [this](const node_child_info & info) {
            std::lock_guard<std::mutex> lk(mu);
            exits++;
            exit_code = info.exit_code;
            pid       = info.pid;
        };
        return w;
    }
    bool has_line(const std::string & l) {
        std::lock_guard<std::mutex> lk(mu);
        for (const auto & x : lines) {
            if (x == l) {
                return true;
            }
        }
        return false;
    }
    bool wait_line(const std::string & l, int ms) {
        return wait_until([&]() { return has_line(l); }, ms);
    }
    bool wait_exit(int ms) {
        return wait_until([&]() {
            std::lock_guard<std::mutex> lk(mu);
            return exits > 0;
        }, ms);
    }
    int n_exits() {
        std::lock_guard<std::mutex> lk(mu);
        return exits;
    }
};

// Forwards 127.0.0.1:<port> to the node's port; sever() cuts every live connection and refuses
// new ones until resume().
struct tcp_proxy {
    int               lsock = -1;
    int               port  = 0;
    int               target = 0;
    std::atomic<bool> quit{false};
    std::atomic<bool> refuse{false};
    std::mutex        mu;
    std::vector<int>  fds;
    std::thread       acceptor;
    std::vector<std::thread> pumps;

    explicit tcp_proxy(int target_port) : target(target_port) {
        lsock = socket(AF_INET, SOCK_STREAM, 0);
        assert(lsock >= 0);
        sockaddr_in a = {};
        a.sin_family      = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port        = 0;
        assert(bind(lsock, (sockaddr *) &a, sizeof(a)) == 0);
        assert(listen(lsock, 16) == 0);
        socklen_t len = sizeof(a);
        assert(getsockname(lsock, (sockaddr *) &a, &len) == 0);
        port     = ntohs(a.sin_port);
        acceptor = std::thread([this]() { run(); });
    }

    ~tcp_proxy() {
        quit.store(true);
        sever();
        acceptor.join();
        for (auto & t : pumps) {
            t.join();
        }
        close(lsock);
    }

    void sever() {
        refuse.store(true);
        std::lock_guard<std::mutex> lk(mu);
        for (int fd : fds) {
            shutdown(fd, SHUT_RDWR);
        }
    }
    void resume() { refuse.store(false); }

    void run() {
        while (!quit.load()) {
            pollfd p = { lsock, POLLIN, 0 };
            if (poll(&p, 1, 100) <= 0) {
                continue;
            }
            const int c = accept(lsock, nullptr, nullptr);
            if (c < 0) {
                continue;
            }
            if (refuse.load()) {
                close(c);
                continue;
            }
            const int s = socket(AF_INET, SOCK_STREAM, 0);
            sockaddr_in a = {};
            a.sin_family      = AF_INET;
            a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            a.sin_port        = htons(target);
            if (s < 0 || connect(s, (sockaddr *) &a, sizeof(a)) != 0) {
                close(c);
                if (s >= 0) {
                    close(s);
                }
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(mu);
                fds.push_back(c);
                fds.push_back(s);
            }
            pumps.emplace_back([this, c, s]() { pump(c, s); });
        }
    }

    void pump(int c, int s) {
        char buf[4096];
        while (!quit.load()) {
            pollfd p[2] = { { c, POLLIN, 0 }, { s, POLLIN, 0 } };
            if (poll(p, 2, 100) < 0) {
                break;
            }
            bool done = false;
            for (int i = 0; i < 2 && !done; i++) {
                if (p[i].revents & (POLLIN | POLLHUP | POLLERR)) {
                    const int from = i == 0 ? c : s;
                    const int to   = i == 0 ? s : c;
                    const ssize_t n = read(from, buf, sizeof(buf));
                    if (n <= 0) {
                        done = true;
                        break;
                    }
                    ssize_t off = 0;
                    while (off < n) {
                        const ssize_t w = write(to, buf + off, (size_t) (n - off));
                        if (w <= 0) {
                            done = true;
                            break;
                        }
                        off += w;
                    }
                }
            }
            if (done) {
                break;
            }
        }
        std::lock_guard<std::mutex> lk(mu);
        for (int fd : { c, s }) {
            for (auto it = fds.begin(); it != fds.end(); ++it) {
                if (*it == fd) {
                    fds.erase(it);
                    break;
                }
            }
            close(fd);
        }
    }
};

static node_spawn_request sh(const std::string & name, const std::string & script) {
    node_spawn_request r;
    r.name = name;
    r.gen  = GEN;
    r.args = { "/bin/sh", "-c", script };
    return r;
}

static bool pid_alive(int pid) {
    return kill(pid, 0) == 0;
}

static std::shared_ptr<router_node_link> make_link(const std::string & url, const std::string & token, int64_t offline_ms = 10000) {
    router_node_link_config lc;
    lc.machine    = "box2";
    lc.gen        = GEN;
    lc.offline_ms = offline_ms;
    auto link = router_node_make_remote(lc, url, token);
    router_node_hooks hooks;
    hooks.wanted = [](const std::string &) { return true; };
    link->set_hooks(std::move(hooks));
    link->start();
    return link;
}

int main() {
    signal(SIGPIPE, SIG_IGN);

    // pure helpers: --host rewrite, per-child key, bearer injection, stop-flag decision
    {
        std::vector<std::string> a = { "srv", "--host", "0.0.0.0", "--port", "1" };
        router_args_set_host(a, "10.0.0.5");
        assert((a == std::vector<std::string>{ "srv", "--host", "10.0.0.5", "--port", "1" }));
        std::vector<std::string> b = { "srv", "--host=0.0.0.0" };
        router_args_set_host(b, "10.0.0.5");
        assert((b == std::vector<std::string>{ "srv", "--host=10.0.0.5" }));
        std::vector<std::string> c = { "srv" };
        router_args_set_host(c, "10.0.0.5");
        assert((c == std::vector<std::string>{ "srv", "--host", "10.0.0.5" }));

        const std::string k1 = router_child_key_generate();
        const std::string k2 = router_child_key_generate();
        assert(k1.size() == 32 && k1 != k2);

        std::map<std::string, std::string> h = { { "authorization", "Bearer client" }, { "AUTHORIZATION", "x" },
                                                  { "Content-Type", "application/json" } };
        router_child_auth_headers(h, k1);
        assert(h.size() == 2 && h.at("Authorization") == "Bearer " + k1 && h.at("Content-Type") == "application/json");
        std::map<std::string, std::string> h2 = { { "authorization", "Bearer client" } };
        router_child_auth_headers(h2, ""); // local child: untouched
        assert(h2.size() == 1 && h2.at("authorization") == "Bearer client");

        // an orphan's exit drops the stop flag only when its entry is gone, never a newer instance's
        assert(router_orphan_exit_clears_stop_flag(false));
        assert(!router_orphan_exit_clears_stop_flag(true));
    }

    server_node_config ncfg = server_node_default_config();
    ncfg.heartbeat_ms      = 300;
    ncfg.shutdown_grace_ms = 3000;
    ncfg.child_host        = "127.0.0.2"; // the daemon's first --node-bind: what an allocated child is told to bind
    server_node        node(ncfg);
    server_node_routes routes(node, TOKEN);

    common_params params; // outlives the server
    params.hostnames      = { "127.0.0.1" };
    params.port           = 0;
    params.ui             = false;
    params.n_threads_http = 8;
    server_http_context http;
    assert(http.init(params));
    routes.register_routes(http);
    assert(http.start());
    http.is_ready.store(true);
    const int node_port = http.port;
    assert(node_port > 0);

    {
        tcp_proxy proxy(node_port);
        const std::string url = "http://127.0.0.1:" + std::to_string(proxy.port) + "/";
        auto link = make_link(url, TOKEN);
        assert(link->remote());
        assert(wait_until([&]() { return link->online(); }, 15000));
        // the child's address is the node's host; its port comes from the node
        assert(link->host() == "127.0.0.1");
        assert(link->machine() == "box2");

        // output lines + exit (code) reach the callbacks, once
        {
            watch_log w;
            node_child_info info = link->spawn(sh("r1", "echo out-line; echo err-line >&2; exit 3"), w.watch());
            assert(info.pid > 0);
            assert(w.wait_exit(10000));
            assert(w.exit_code == 3);
            assert(w.has_line("out-line") && w.has_line("err-line"));
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            assert(w.n_exits() == 1);
        }

        // the node allocates the port: it is put on the command line and reported back
        {
            watch_log w;
            node_spawn_request r = sh("alloc", "echo \"$0 $1\"; exec sleep 30");
            r.args.push_back("--port");
            r.args.push_back("1");
            r.alloc_port = true;
            const node_child_info info = link->spawn(r, w.watch());
            assert(info.port > 0 && info.port != 1);
            {
                // regression: on_spawn must run, with the node's answer (the router records the
                // child's port/host from it; a moved-from watch once skipped it -> proxy to port 0)
                std::lock_guard<std::mutex> lk(w.mu);
                assert(w.spawn_port == info.port && w.spawn_pid == info.pid);
            }
            assert(w.wait_line("--port " + std::to_string(info.port), 10000));
            assert(node.children().size() >= 1);
            // stop (sync, term) -> exit event
            const node_child_info st = link->stop("alloc", 5, "term");
            (void) st;
            assert(w.wait_exit(10000));
            assert(!pid_alive(info.pid));
        }

        // a node-allocated child binds the node's address (never the leader's 0.0.0.0) and sees the
        // per-child key the leader put in its env
        {
            watch_log w;
            node_spawn_request r = sh("keyed", "echo \"$0 $1 $2 $3 key=$LLAMA_API_KEY\"; exec sleep 30");
            r.args.push_back("--host");
            r.args.push_back("0.0.0.0");
            r.args.push_back("--port");
            r.args.push_back("1");
            r.alloc_port = true;
            r.env.push_back("LLAMA_API_KEY=child-secret-key");
            const node_child_info info = link->spawn(r, w.watch());
            assert(info.port > 0);
            assert(w.wait_line("--host 127.0.0.2 --port " + std::to_string(info.port) + " key=child-secret-key", 10000));
            link->stop("keyed", 5, "term");
            assert(w.wait_exit(10000));
        }

        // a name that is still running: 409 after the retry window; the first child is untouched
        {
            watch_log w;
            const node_child_info info = link->spawn(sh("dup", "exec sleep 30"), w.watch());
            watch_log w2;
            const int64_t t0 = now_ms();
            bool refused = false;
            try {
                link->spawn(sh("dup", "exec sleep 30"), w2.watch());
            } catch (const server_node_error & e) {
                refused = e.status == 409;
            }
            assert(refused);
            assert(now_ms() - t0 >= ROUTER_NODE_SPAWN_409_RETRY_MS - 100);
            assert(pid_alive(info.pid) && w.n_exits() == 0);
            // signal (sync) -> exit
            link->signal("dup", 9);
            assert(w.wait_exit(10000));
            assert(w2.n_exits() == 0); // never watched
        }

        // a finished child's name is free again at once (the 409 window right after an exit)
        for (int i = 0; i < 5; i++) {
            watch_log w;
            link->spawn(sh("again", "echo round" + std::to_string(i)), w.watch());
            assert(w.wait_exit(10000));
            assert(w.has_line("round" + std::to_string(i)));
        }

        // async stop / signal (command thread)
        {
            watch_log w;
            const node_child_info info = link->spawn(sh("async1", "exec sleep 30"), w.watch());
            link->signal_async("async1", info.pid, 9);
            assert(w.wait_exit(10000));
            watch_log w2;
            const node_child_info i2 = link->spawn(sh("async2", "exec sleep 30"), w2.watch());
            link->stop_async("async2", i2.pid, 5, "term");
            assert(w2.wait_exit(10000));
        }

        // refusals: 4xx from the node is a clean failure, nothing is left watched or running
        {
            watch_log w;
            node_spawn_request bad = sh("bad", "true");
            bad.args = { "relative-binary" };
            const int64_t t0 = now_ms();
            int status = -1;
            try {
                link->spawn(bad, w.watch());
            } catch (const server_node_error & e) {
                status = e.status;
            }
            assert(status == 400);
            assert(now_ms() - t0 < ROUTER_NODE_SPAWN_409_RETRY_MS); // not retried
            node_spawn_request bad_env = sh("bad", "true");
            bad_env.env = { "TMPDIR=/etc/passwd" };
            status = -1;
            try {
                link->spawn(bad_env, w.watch());
            } catch (const server_node_error & e) {
                status = e.status;
            }
            assert(status == 400);
            assert(w.n_exits() == 0 && w.lines.empty());
            // the name was never taken
            watch_log ok;
            link->spawn(sh("bad", "echo fine"), ok.watch());
            assert(ok.wait_exit(10000) && ok.has_line("fine"));
        }

        // a wrong token never gets in: offline link -> spawn refused cleanly (503), nothing started
        {
            auto bad_link = make_link(url, "not-the-token", 1000);
            std::this_thread::sleep_for(std::chrono::milliseconds(700));
            assert(!bad_link->online());
            watch_log w;
            int status = -1;
            try {
                bad_link->spawn(sh("nope", "echo should-not-run"), w.watch());
            } catch (const server_node_error & e) {
                status = e.status;
            }
            assert(status == 503);
            bool seen = false;
            for (const auto & c : node.children()) {
                seen = seen || c.name == "nope";
            }
            assert(!seen);
            bad_link->shutdown();
        }

        // a node nobody listens on: offline, spawn refused, no hang
        {
            auto dead = make_link("http://127.0.0.1:1", TOKEN, 1000);
            watch_log w;
            int status = -1;
            const int64_t t0 = now_ms();
            try {
                dead->spawn(sh("dead", "true"), w.watch());
            } catch (const server_node_error & e) {
                status = e.status;
            }
            assert(status == 503 && now_ms() - t0 < 5000);
            dead->shutdown();
        }

        // the event stream is cut while a child runs: the link reconnects from its last seq and
        // the child's later output and its exit still arrive, once
        {
            watch_log w;
            const node_child_info info = link->spawn(sh("cut", "echo before; sleep 1; echo after; exit 7"), w.watch());
            assert(info.pid > 0);
            assert(w.wait_line("before", 10000));
            proxy.sever();
            std::this_thread::sleep_for(std::chrono::milliseconds(1800)); // the child exits meanwhile
            assert(w.n_exits() == 0);
            proxy.resume();
            assert(w.wait_exit(15000));
            assert(w.exit_code == 7);
            assert(w.has_line("after"));
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            assert(w.n_exits() == 1);
            assert(wait_until([&]() { return link->online(); }, 10000));

            // a healthy child survives a stream drop too: nothing is reported, it keeps running
            watch_log w2;
            const node_child_info i2 = link->spawn(sh("keep", "exec sleep 30"), w2.watch());
            proxy.sever();
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            proxy.resume();
            std::this_thread::sleep_for(std::chrono::milliseconds(1500));
            assert(w2.n_exits() == 0 && pid_alive(i2.pid));
            link->stop("keep", 5, "term");
            assert(w2.wait_exit(10000));
            assert(w2.n_exits() == 1);
        }

        // heartbeat loss and recovery (offline_ms 1 s): silent node -> offline + hook(false), spawns refused
        // 503; back -> reconcile -> online + hook(true). A child of the current gen the leader still wants
        // is kept (no exit reported), one it no longer wants is stopped by the reconcile.
        {
            std::atomic<int>  n_off{ 0 };
            std::atomic<int>  n_on{ 0 };
            std::atomic<bool> drop_unwanted{ false };
            router_node_link_config lc;
            lc.machine    = "box3";
            lc.gen        = GEN;
            lc.offline_ms = 1000;
            auto hb = router_node_make_remote(lc, url, TOKEN);
            router_node_hooks hooks;
            hooks.wanted    = [&](const std::string & n) { return !(n == "hb-unwanted" && drop_unwanted.load()); };
            hooks.on_online = [&](bool on) { (on ? n_on : n_off)++; };
            hb->set_hooks(std::move(hooks));
            hb->start();
            assert(wait_until([&]() { return hb->online(); }, 15000));
            assert(n_on.load() == 1 && n_off.load() == 0);

            watch_log wk;
            watch_log wu;
            const node_child_info ik = hb->spawn(sh("hb-keep", "exec sleep 30"), wk.watch());
            hb->spawn(sh("hb-unwanted", "exec sleep 30"), wu.watch());

            proxy.sever();
            assert(wait_until([&]() { return !hb->online(); }, 8000));
            assert(n_off.load() == 1);
            int status = -1;
            try {
                watch_log wx;
                hb->spawn(sh("hb-refused", "true"), wx.watch());
            } catch (const server_node_error & e) {
                status = e.status;
            }
            assert(status == 503);

            drop_unwanted.store(true);
            proxy.resume();
            assert(wait_until([&]() { return hb->online(); }, 15000));
            assert(n_on.load() == 2);
            assert(wu.wait_exit(15000)); // stopped by the reconcile
            assert(wk.n_exits() == 0 && pid_alive(ik.pid));
            hb->stop("hb-keep", 5, "term");
            assert(wk.wait_exit(10000));
            hb->shutdown();
        }

        link->shutdown();
    }

    node.close_events();
    http.stop();
    http.join();
    printf("test-router-node-remote: ok\n");
    return 0;
}

#else
int main() { return 0; }
#endif
