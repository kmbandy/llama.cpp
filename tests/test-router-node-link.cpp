// The router's node link for its own machine (tools/server/server-router-node-client.cpp): the
// in-process node core behind the node interface. A child's output lines and its exit reach the
// watch callbacks, stop / signal go through the node, and a finished child's name can be reused.

#include "server-router-node-client.h"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#undef NDEBUG
#include <cassert>

#ifndef _WIN32

struct watch_log {
    std::mutex              mu;
    std::condition_variable cv;
    std::vector<std::string> lines;
    bool                    exited    = false;
    int                     exit_code = -99;
    int                     exits     = 0;

    router_node_watch watch() {
        router_node_watch w;
        w.on_line = [this](const std::string & line) {
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(line);
            cv.notify_all();
        };
        w.on_exit = [this](const node_child_info & info) {
            std::lock_guard<std::mutex> lk(mu);
            exited    = true;
            exit_code = info.exit_code;
            exits++;
            cv.notify_all();
        };
        return w;
    }
    bool wait_exit(int ms) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::milliseconds(ms), [this]() { return exited; });
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
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::milliseconds(ms), [&]() {
            for (const auto & x : lines) {
                if (x == l) {
                    return true;
                }
            }
            return false;
        });
    }
};

static node_spawn_request sh(const std::string & name, const std::string & script) {
    node_spawn_request r;
    r.name = name;
    r.gen  = "gen-link-test";
    r.args = { "/bin/sh", "-c", script };
    return r;
}

static std::shared_ptr<router_node_link> make_link() {
    server_node_config ncfg = server_node_default_config();
    router_node_link_config lc;
    lc.gen = "gen-link-test";
    auto link = router_node_make_local(lc, std::move(ncfg));
    link->start();
    return link;
}

int main() {
    auto link = make_link();
    assert(!link->remote());
    assert(link->online());

    // output lines (stdout and stderr, newline stripped) then the exit, once, with the code
    {
        watch_log w;
        const node_child_info info = link->spawn(sh("c1", "echo out-line; echo err-line >&2; exit 3"), w.watch());
        assert(info.pid > 0);
        assert(w.wait_exit(10000));
        assert(w.exit_code == 3);
        assert(w.has_line("out-line"));
        assert(w.has_line("err-line"));
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        assert(w.exits == 1);
    }

    // the name of an exited child is reusable; a live name is refused (409)
    {
        watch_log w;
        link->spawn(sh("c1", "sleep 30"), w.watch());
        bool refused = false;
        try {
            watch_log w2;
            link->spawn(sh("c1", "sleep 30"), w2.watch());
        } catch (const server_node_error & e) {
            refused = e.status == 409;
        }
        assert(refused);
        link->stop("c1", 5, "term");
        assert(w.wait_exit(10000));
        assert(!w.has_line("never"));
    }

    // stdin exit command: the child reads it, the router never signals it
    {
        watch_log w;
        const node_child_info info = link->spawn(sh("c2", "read cmd; echo got:$cmd; exit 0"), w.watch());
        link->stop_async("c2", info.pid, 5, "stdin");
        assert(w.wait_exit(10000));
        assert(w.exit_code == 0);
        assert(w.has_line("got:cmd_router_to_child:exit"));
    }

    // SIGKILL through the async queue (a child that ignores TERM)
    {
        watch_log w;
        const node_child_info info = link->spawn(sh("c3", "trap '' TERM; echo up; while true; do sleep 1; done"), w.watch());
        assert(w.wait_line("up", 10000));
        link->signal_async("c3", info.pid, 9);
        assert(w.wait_exit(10000));
        assert(w.exit_code != 0);
    }

    // state(): the node's table
    {
        const json st = link->state();
        assert(st.contains("children"));
    }

    link->shutdown();
    return 0;
}

#else
int main() { return 0; }
#endif
