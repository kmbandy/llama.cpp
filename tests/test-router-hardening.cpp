// Tests for the final-review fixes: the bounded child-exit wait decision, the remote worker listen
// host rule, the node's token-file mode / exec allowlist / bind checks, the log limiter, and the
// replica-name guards (tools/server/server-router-policy.cpp, server-router-group-lifecycle.cpp,
// server-node.cpp).

#include "common.h"
#include "server-node.h"
#include "server-router-group-lifecycle.h"
#include "server-router-policy.h"

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

static void write_file(const fs::path & p, const std::string & content) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << content;
}

static bool contains(const std::string & s, const std::string & sub) {
    return s.find(sub) != std::string::npos;
}

static void test_wait_decision() {
    // every child down: done, whatever else is going on
    assert(router_child_wait_decide(0, 0, false, false) == ROUTER_CHILD_WAIT_EXITED);
    assert(router_child_wait_decide(0, 0, true, true) == ROUTER_CHILD_WAIT_EXITED);
    // still running on an online machine: keep waiting until the bound
    assert(router_child_wait_decide(1, 1, false, false) == ROUTER_CHILD_WAIT_PENDING);
    assert(router_child_wait_decide(2, 1, false, false) == ROUTER_CHILD_WAIT_PENDING); // one offline, one online
    assert(router_child_wait_decide(1, 1, false, true) == ROUTER_CHILD_WAIT_TIMEOUT);
    // only offline machines are left: nothing will report an exit, give up now
    assert(router_child_wait_decide(1, 0, false, false) == ROUTER_CHILD_WAIT_OFFLINE);
    assert(router_child_wait_decide(3, 0, false, true) == ROUTER_CHILD_WAIT_OFFLINE);
    // shutdown (when the caller honours it) beats everything but "all exited"
    assert(router_child_wait_decide(1, 1, true, false) == ROUTER_CHILD_WAIT_SHUTDOWN);
    assert(router_child_wait_decide(1, 0, true, true) == ROUTER_CHILD_WAIT_SHUTDOWN);

    // the bound: stop-timeout + SIGKILL grace, a group adds its workers' stop
    assert(router_child_wait_bound_ms(10, false, 10000, 40000) == 20000);
    assert(router_child_wait_bound_ms(10, true, 10000, 40000) == 60000);
    assert(router_child_wait_bound_ms(0, false, 10000, 40000) == 11000); // never below one second
}

static void test_remote_worker_host() {
    const std::string node = "100.64.0.7";
    auto fix = [&](const std::string & launch_str, router_launch & l) {
        assert(router_parse_launch(launch_str, l).empty());
        return router_remote_worker_host_fix(l, node);
    };
    // explicit wildcards are refused, in every spelling
    for (const char * wild : {
             "/opt/w --listen 0.0.0.0:7001", "/opt/w --listen=0.0.0.0:7001", "/opt/w --listen :7001", "/opt/w --listen [::]:7001",
             "/opt/w --listen *:7001", "/opt/w --host 0.0.0.0 --port 7001", "/opt/w --host=:: --port 7001" }) {
        router_launch l;
        const std::string err = fix(wild, l);
        assert(!err.empty());
        assert(contains(err, "wildcard"));
        assert(contains(err, node));
    }
    // a named host is left alone
    {
        router_launch l;
        assert(fix("/opt/w --listen 100.64.0.7:7001", l).empty());
        assert((l.argv == std::vector<std::string>{ "/opt/w", "--listen", "100.64.0.7:7001" }));
    }
    {
        router_launch l;
        assert(fix("/opt/w --host 10.0.0.9 --port 7001", l).empty());
        assert(l.argv.size() == 5);
    }
    // no host at all: bound to the node's address
    {
        router_launch l;
        assert(fix("/opt/w --port 7001", l).empty());
        assert((l.argv == std::vector<std::string>{ "/opt/w", "--port", "7001", "--host", node }));
        std::string host;
        assert(router_launch_endpoint(l.words, host) == 7001);
        assert(router_remote_worker_host_fix(l, node).empty()); // now named: stable
        assert(l.argv.size() == 5);
    }
    // a shell launch gets it on the command string
    {
        router_launch s;
        assert(router_parse_launch("/opt/w --port 7001 --x $HOME", s).empty() && s.via_shell);
        assert(router_remote_worker_host_fix(s, node).empty());
        assert(s.argv.size() == 3 && contains(s.argv[2], "--host '100.64.0.7'"));
        router_launch w;
        assert(router_parse_launch("/opt/w --listen 0.0.0.0:7001 --x $HOME", w).empty() && w.via_shell);
        assert(!router_remote_worker_host_fix(w, node).empty());
    }
    // the node's own address cannot be a wildcard or unknown
    {
        router_launch l;
        assert(router_parse_launch("/opt/w --port 7001", l).empty());
        assert(!router_remote_worker_host_fix(l, "").empty());
        assert(!router_remote_worker_host_fix(l, "0.0.0.0").empty());
    }
}

static void test_token_file_mode() {
    const fs::path dir = fs::temp_directory_path() / ("router-hardening-" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    write_file(dir / "tok", "secret\n");
    for (auto perms : { fs::perms::owner_read | fs::perms::owner_write, fs::perms::owner_read }) {
        fs::permissions(dir / "tok", perms, fs::perm_options::replace);
        assert(server_node_token_file_mode_error((dir / "tok").string()).empty());
    }
    for (auto extra : { fs::perms::group_read, fs::perms::others_read, fs::perms::group_write, fs::perms::others_write }) {
        fs::permissions(dir / "tok", fs::perms::owner_read | fs::perms::owner_write | extra, fs::perm_options::replace);
        const std::string err = server_node_token_file_mode_error((dir / "tok").string());
        assert(!err.empty());
        assert(contains(err, "chmod 600"));
    }
    // a missing file is the reader's error, not the mode check's
    assert(server_node_token_file_mode_error((dir / "missing").string()).empty());
    assert(server_node_token_file_mode_error("").empty());

    // the node refuses to start with such a token file
    {
        common_params p;
        p.router_node     = true;
        p.node_token_file = (dir / "tok").string();
        p.node_bind       = "127.0.0.1";
        const std::string err = server_node_check_params(p);
        assert(contains(err, "chmod 600"));
        fs::permissions(dir / "tok", fs::perms::owner_read | fs::perms::owner_write, fs::perm_options::replace);
        assert(server_node_check_params(p).empty());
        p.node_exec_allow = { (dir / "no-such-dir").string() };
        assert(contains(server_node_check_params(p), "--node-exec-allow"));
        p.node_exec_allow = { dir.string() };
        assert(server_node_check_params(p).empty());
    }
    fs::remove_all(dir);
}

static void test_exec_allow() {
    const fs::path dir = fs::temp_directory_path() / ("router-hardening-exec-" + std::to_string((long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::remove_all(dir);
    const fs::path allowed = dir / "bin";
    const fs::path sibling = dir / "bin2"; // shares the prefix, not the directory
    const fs::path outside = dir / "outside";
    write_file(allowed / "tool", "#!/bin/sh\n");
    write_file(allowed / "sub" / "tool2", "#!/bin/sh\n");
    write_file(sibling / "tool", "#!/bin/sh\n");
    write_file(outside / "evil", "#!/bin/sh\n");
    std::error_code ec;
    fs::create_symlink(outside / "evil", allowed / "link-out", ec);
    assert(!ec);
    fs::create_symlink(allowed / "tool", outside / "link-in", ec); // outside name, target inside
    assert(!ec);
    fs::create_directory_symlink(outside, allowed / "dir-out", ec);
    assert(!ec);

    const std::vector<std::string> allow = { allowed.string() };
    assert(server_node_exec_allowed((allowed / "tool").string(), allow));
    assert(server_node_exec_allowed((allowed / "sub" / "tool2").string(), allow));
    assert(server_node_exec_allowed((allowed / "sub" / ".." / "tool").string(), allow)); // ".." that stays inside
    assert(server_node_exec_allowed((outside / "link-in").string(), allow));             // resolves inside
    assert(server_node_exec_allowed((allowed / "tool").string(), { (dir / "nope").string(), allowed.string() }));
    assert(server_node_exec_allowed((allowed / "tool").string(), { allowed.string() + "/" })); // trailing slash

    assert(!server_node_exec_allowed((sibling / "tool").string(), allow));                // /bin2 is not under /bin
    assert(!server_node_exec_allowed((outside / "evil").string(), allow));
    assert(!server_node_exec_allowed((allowed / "link-out").string(), allow));            // symlink that points out
    assert(!server_node_exec_allowed((allowed / "dir-out" / "evil").string(), allow));    // through a symlinked dir
    assert(!server_node_exec_allowed((allowed / ".." / "outside" / "evil").string(), allow)); // ".." escape
    assert(!server_node_exec_allowed((allowed / ".." / "bin2" / "tool").string(), allow));
    assert(!server_node_exec_allowed(allowed.string(), allow));                           // the dir itself is no program under it
    assert(!server_node_exec_allowed((allowed / "missing").string(), allow));
    assert(!server_node_exec_allowed((allowed / "tool").string(), {}));
    assert(!server_node_exec_allowed((allowed / "tool").string(), { (dir / "nope").string() }));
    assert(!server_node_exec_allowed((allowed / "tool").string(), { "" }));

    // the node answers 403 and spawns nothing
    {
        server_node_config cfg = server_node_default_config();
        cfg.log_lines  = false;
        cfg.exec_allow = allow;
        server_node node(cfg);
        node_spawn_request req;
        req.name = "x";
        req.gen  = "g";
        req.args = { (outside / "evil").string() };
        try {
            node.spawn(req);
            assert(false);
        } catch (const server_node_error & e) {
            assert(e.status == 403);
        }
        assert(node.children().empty());
    }
    fs::remove_all(dir);
}

static void test_bind_warning_and_limiter() {
    assert(server_node_addr_is_overlay("127.0.0.1"));
    assert(server_node_addr_is_overlay("127.1.2.3"));
    assert(server_node_addr_is_overlay("100.64.0.1"));
    assert(server_node_addr_is_overlay("100.127.255.254"));
    assert(!server_node_addr_is_overlay("100.63.255.255"));
    assert(!server_node_addr_is_overlay("100.128.0.1"));
    assert(!server_node_addr_is_overlay("192.168.1.20"));
    assert(!server_node_addr_is_overlay("10.0.0.5"));
    assert(server_node_addr_is_overlay("::1"));
    assert(server_node_addr_is_overlay("fd7a:115c:a1e0::1"));
    assert(server_node_addr_is_overlay("::ffff:100.64.0.9"));
    assert(!server_node_addr_is_overlay("fd00::1"));
    assert(!server_node_addr_is_overlay("2001:db8::1"));
    assert(!server_node_addr_is_overlay("not an address"));

    assert(server_node_bind_warning("127.0.0.1").empty());
    assert(server_node_bind_warning("100.64.0.7").empty());
    assert(contains(server_node_bind_warning("192.168.1.20"), "Tailscale"));
    assert(server_node_bind_warning("").empty());

    server_node_log_limiter lim(10000, 4);
    assert(lim.allow("a", 1000));
    assert(!lim.allow("a", 5000));
    assert(!lim.allow("a", 10999));
    assert(lim.allow("b", 5000));       // another peer is independent
    assert(lim.allow("a", 11000));      // the interval passed
    // the table stays bounded under a flood of distinct peers
    for (int i = 0; i < 100; i++) {
        assert(lim.allow("p" + std::to_string(i), 20000 + i));
    }
}

static void test_replica_and_hold_guards() {
    assert(router_name_addressable(""));
    assert(!router_name_addressable("alias"));
    assert(router_hold_ttl_clamp(60) == 60);
    assert(router_hold_ttl_clamp(0) == 0);
    assert(router_hold_ttl_clamp(-5) == 0);
    assert(router_hold_ttl_clamp(ROUTER_HOLD_TTL_MAX_S) == ROUTER_HOLD_TTL_MAX_S);
    assert(router_hold_ttl_clamp(INT64_MAX) == ROUTER_HOLD_TTL_MAX_S);
    assert(router_hold_ttl_clamp(INT64_MAX / 1000 + 1) * 1000 > 0); // what the caller multiplies cannot wrap
}

int main() {
    test_wait_decision();
    test_remote_worker_host();
    test_token_file_mode();
    test_exec_allow();
    test_bind_warning_and_limiter();
    test_replica_and_hold_guards();
    return 0;
}
