// Pure tests for the multi-machine router (Task 10c): machine-prefixed gpus= slots, a remote slot's
// ledger from a node's /node/state payload, admission across machines, board resources per machine,
// heartbeat loss -> offline decision -> 503 / unavailable and recovery, and the reconcile decision
// table. No processes, no sockets: everything is literals.

#include "server-router-admission.h"
#include "server-router-board.h"
#include "server-router-node-client.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>

static constexpr int64_t MB = 1024LL * 1024LL;

static bool is_local_main(const std::string & m) {
    return m.empty() || m == "mad-lab-main" || m == "local";
}

static bool has(const std::vector<std::string> & v, const std::string & s) {
    for (const auto & x : v) {
        if (x == s) {
            return true;
        }
    }
    return false;
}

static void test_gpus_spec() {
    std::vector<router_gpu_spec_entry> out;

    // back-compat: unprefixed entries are local, total optional, board name optional, any probe text
    assert(router_parse_gpus_spec("ROCm0:16384:/sys/class/drm/card1/device/mem_info_vram_used", is_local_main, out).empty());
    assert(out.size() == 1 && out[0].machine.empty() && out[0].dev == "ROCm0" && out[0].board.empty());
    assert(out[0].total_mb == 16384 && out[0].probe == "/sys/class/drm/card1/device/mem_info_vram_used");

    assert(router_parse_gpus_spec("ROCm0=R9700::/sys/class/drm/card1/device/mem_info_vram_used, CUDA0:0:nvml:0", is_local_main, out).empty());
    assert(out.size() == 2);
    assert(out[0].dev == "ROCm0" && out[0].board == "R9700" && out[0].total_mb == 0 && out[0].machine.empty());
    assert(out[1].dev == "CUDA0" && out[1].probe == "nvml:0" && out[1].machine.empty());

    // another machine's card: machine prefix, board name, PCI-address probe (both forms)
    assert(router_parse_gpus_spec("mad-lab-2026/ROCm1=6900XT:16384:pci:0000:03:00.0", is_local_main, out).empty());
    assert(out.size() == 1 && out[0].machine == "mad-lab-2026" && out[0].dev == "ROCm1" && out[0].board == "6900XT");
    assert(out[0].total_mb == 16384 && out[0].pdev == "0000:03:00.0");
    assert(router_parse_gpus_spec("mad-lab-2026/ROCm1:16384:/sys/bus/pci/devices/0000:03:00.0/mem_info_vram_used", is_local_main, out).empty());
    assert(out[0].machine == "mad-lab-2026" && out[0].board.empty() && out[0].pdev == "0000:03:00.0");

    // the local machine's own name is local (machine ""), so one gpus= line can serve both boxes
    assert(router_parse_gpus_spec("mad-lab-main/ROCm0=R9700::/sys/class/drm/card1/device/mem_info_vram_used,"
                                  "mad-lab-2026/ROCm1=6900XT:16384:pci:0000:03:00.0", is_local_main, out).empty());
    assert(out.size() == 2 && out[0].machine.empty() && out[0].dev == "ROCm0" && out[0].board == "R9700");
    assert(out[1].machine == "mad-lab-2026");
    // the same line read on the other box: roles swap
    auto is_local_2026 = [](const std::string & m) { return m.empty() || m == "mad-lab-2026"; };
    assert(router_parse_gpus_spec("mad-lab-main/ROCm0=R9700:24576:pci:0000:42:00.0,"
                                  "mad-lab-2026/ROCm1=6900XT::/sys/class/drm/card1/device/mem_info_vram_used", is_local_2026, out).empty());
    assert(out[0].machine == "mad-lab-main" && out[1].machine.empty());

    // refusals
    assert(!router_parse_gpus_spec("mad-lab-2026/ROCm1:0:pci:0000:03:00.0", is_local_main, out).empty());          // remote needs a total
    assert(!router_parse_gpus_spec("mad-lab-2026/ROCm1:16384:/sys/class/drm/card1/device/mem_info_vram_used", is_local_main, out).empty()); // no PCI address
    assert(!router_parse_gpus_spec("/ROCm1:16384:pci:0000:03:00.0", is_local_main, out).empty());                  // empty machine
    assert(!router_parse_gpus_spec("ROCm0", is_local_main, out).empty());                                          // no fields
    assert(!router_parse_gpus_spec("ROCm0=:1:x", is_local_main, out).empty());                                     // empty board
    assert(!router_parse_gpus_spec("ROCm0:abc:/x", is_local_main, out).empty());                                   // total not a number
    assert(!router_parse_gpus_spec("ROCm0:1:", is_local_main, out).empty());                                       // no probe

    // the PCI address of a probe
    assert(router_probe_pdev("pci:0000:03:00.0") == "0000:03:00.0");
    assert(router_probe_pdev("/sys/bus/pci/devices/0000:42:00.0/mem_info_vram_used") == "0000:42:00.0");
    assert(router_probe_pdev("nvml:0").empty());
    assert(router_probe_pdev("/sys/class/drm/card1/device/mem_info_vram_used").empty());
}

static void test_slot_ids() {
    std::string m;
    std::string d;
    router_slot_split("ROCm0", m, d);
    assert(m.empty() && d == "ROCm0");
    router_slot_split("mad-lab-2026/ROCm1", m, d);
    assert(m == "mad-lab-2026" && d == "ROCm1");

    // preset gpu= + machine= -> slot id; this machine's slots stay bare
    assert(router_slot_resolve("ROCm0", "", is_local_main) == "ROCm0");
    assert(router_slot_resolve("ROCm0", "mad-lab-main", is_local_main) == "ROCm0");
    assert(router_slot_resolve("ROCm1", "mad-lab-2026", is_local_main) == "mad-lab-2026/ROCm1");
    assert(router_slot_resolve("mad-lab-2026/ROCm1", "", is_local_main) == "mad-lab-2026/ROCm1");
    assert(router_slot_resolve("mad-lab-2026/ROCm1", "mad-lab-2026", is_local_main) == "mad-lab-2026/ROCm1");
    assert(router_slot_resolve("mad-lab-main/ROCm0", "", is_local_main) == "ROCm0");
    assert(router_slot_resolve("ROCm0", "local", is_local_main) == "ROCm0");

    assert(ledger_slot_id("", "ROCm0") == "ROCm0");
    assert(ledger_slot_id("mad-lab-2026", "ROCm1") == "mad-lab-2026/ROCm1");
}

// a node's /node/state as the Task 9 API answers it
static json node_state() {
    return json::parse(R"({
        "node_pid": 4242, "time_ms": 1, "mem_available": 40000000000,
        "children": [
            {"name":"m1","gen":"G","pid":500,"port":8100,"status":"running","exit_code":-1,"killed":false,"adopted":false,
             "rss_anon":3000000000,"rss_shmem":1000000000,"vram":[{"pdev":"0000:03:00.0","bytes":4194304000}]},
            {"name":"old","gen":"G","pid":501,"port":8101,"status":"exited","exit_code":0,"killed":false,"adopted":false,
             "rss_anon":null,"rss_shmem":null}
        ],
        "orphans": [],
        "vram": [
            {"pid":500,"pdev":"0000:03:00.0","bytes":4194304000},
            {"pid":900,"pdev":"0000:03:00.0","bytes":2097152000},
            {"pid":901,"pdev":"0000:04:00.0","bytes":8589934592}
        ],
        "devices": [
            {"pdev":"0000:03:00.0","vram_used":7340032000},
            {"pdev":"0000:04:00.0","vram_used":8589934592}
        ]
    })");
}

static void test_remote_ledger() {
    const router_node_probe p = router_node_probe_from_state(node_state());
    assert(p.ok && p.mem_available == 40000000000LL);
    assert(p.vram.size() == 3);
    assert(p.child_pids.count(500) == 1 && p.child_pids.count(501) == 0); // an exited child is not live
    assert(p.sysfs_used.at("0000:03:00.0") == 7340032000LL);
    assert(router_node_child_ram(p, 500) == 4000000000LL); // anon + shmem
    assert(router_node_child_ram(p, 501) == -1);
    assert(router_node_child_ram(p, 12345) == -1);

    // slot 6900XT on that node: 16384 MB total, the router reserved 4000 MB for m1 (pid 500)
    ledger_slot slot = { "mad-lab-2026/ROCm1", "0000:03:00.0", 16384 * MB, 4000 * MB };
    // foreign = pid 900 only (pid 500 is the node's child: counted through router_reserved, never twice)
    assert(ledger_foreign_vram(p.vram, p.child_pids, slot.pdev) == 2097152000LL);
    // ledger term: 16384 - 4000 - 2000 = 10384 MB ; sysfs term: 16384 - 7000 = 9384 MB ; the smaller wins
    assert(router_node_free_vram(p, slot) == 16384 * MB - 7340032000LL);
    // a card nobody but the router uses reads the ledger term (sysfs reads the same)
    ledger_slot other = { "mad-lab-2026/ROCm2", "0000:04:00.0", 16384 * MB, 0 };
    assert(router_node_free_vram(p, other) == 16384 * MB - 8589934592LL);
    // no per-PID view (NVML-style slot, pdev ""): the ledger term alone
    ledger_slot whole = { "mad-lab-2026/CUDA0", "", 16384 * MB, 1000 * MB };
    assert(router_node_free_vram(p, whole) == 15384 * MB);
    // a node that was never read: the ledger term alone, never a made-up number
    router_node_probe none;
    assert(!none.ok);
    assert(router_node_free_vram(none, slot) == 16384 * MB - 4000 * MB);
    // reserved + foreign above the total: clamped at 0
    ledger_slot full = { "x/y", "0000:03:00.0", 5000 * MB, 4000 * MB };
    assert(router_node_free_vram(p, full) == 0);

    // RAM per machine: the node's MemAvailable minus headroom (same formula as the local machine's)
    assert(ledger_free_ram(p.mem_available, 4096 * MB) == 40000000000LL - 4096 * MB);
    assert(ledger_free_ram(-1, 4096 * MB) == -1); // unknown never gates

    // garbage / empty state: not ok, nothing invented
    assert(!router_node_probe_from_state(json::array()).ok);
    const router_node_probe e = router_node_probe_from_state(json::object());
    assert(e.ok && e.vram.empty() && e.mem_available == -1);
}

static admission_input two_box_input() {
    admission_input in;
    in.alias = "dsv";
    in.slots = {
        { "ROCm0",              "",       "gpu:R9700",         20000 * MB },
        { "mad-lab-2026/ROCm1", "mad-lab-2026", "gpu:6900XT@mad-lab-2026", 12000 * MB },
    };
    in.machines = {
        { "",             "ram",              -1 },
        { "mad-lab-2026", "ram@mad-lab-2026", 8000 * MB },
    };
    return in;
}

static void test_admission_remote() {
    // a pool of both boxes' cards: the one with the most free VRAM, the board claim names its machine
    {
        admission_input in = two_box_input();
        in.candidates = admission_pool_candidates(in.slots, 10000 * MB, 0);
        const admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_ADMIT && r.slots.size() == 1 && r.slots[0] == "ROCm0");
        assert(r.board_claims_to_take.size() == 1 && r.board_claims_to_take[0] == "gpu:R9700");
        // the machine override picks the other box
        in.machine = "mad-lab-2026";
        const admission_result r2 = decide_admission(in);
        assert(r2.verdict == ADMISSION_ADMIT && r2.slots[0] == "mad-lab-2026/ROCm1");
        assert(r2.board_claims_to_take.size() == 1 && r2.board_claims_to_take[0] == "gpu:6900XT@mad-lab-2026");
        // a machine nobody is on: no candidate
        in.machine = "nowhere";
        const admission_result r3 = decide_admission(in);
        assert(r3.verdict == ADMISSION_QUEUE && r3.blocked == ADMISSION_BLOCK_NO_CANDIDATE);
    }
    // a model that needs more than the remote card has free: capacity; an idle resident there makes room
    {
        admission_input in = two_box_input();
        admission_candidate c;
        c.machine = "mad-lab-2026";
        c.vram    = { { "mad-lab-2026/ROCm1", 14000 * MB } };
        in.candidates = { c };
        admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CAPACITY && r.blocked_on == "mad-lab-2026/ROCm1");

        admission_resident res;
        res.name      = "r1";
        res.vram      = { { "mad-lab-2026/ROCm1", 6000 * MB } };
        res.ram       = { { "mad-lab-2026", 0 } };
        res.last_used = 5;
        in.residents  = { res };
        r = decide_admission(in);
        assert(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims.size() == 1 && r.victims[0] == "r1");

        // a resident on the OTHER box does not free this one
        in.residents[0].vram = { { "ROCm0", 6000 * MB } };
        in.residents[0].ram  = { { "", 0 } };
        r = decide_admission(in);
        assert(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CAPACITY);
    }
    // host RAM is gated per machine, from the node's MemAvailable
    {
        admission_input in = two_box_input();
        admission_candidate c;
        c.machine = "mad-lab-2026";
        c.vram    = { { "mad-lab-2026/ROCm1", 4000 * MB } };
        c.ram     = { { "mad-lab-2026", 9000 * MB } };
        in.candidates = { c };
        admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CAPACITY && r.blocked_on_ram && r.blocked_on == "mad-lab-2026");

        admission_resident res; // idle, holds 3000 MB of that machine's RAM and none of the VRAM that matters
        res.name = "r2";
        res.vram = { { "mad-lab-2026/ROCm1", 0 } };
        res.ram  = { { "mad-lab-2026", 3000 * MB } };
        in.residents = { res };
        r = decide_admission(in);
        assert(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims.size() == 1 && r.victims[0] == "r2");
        assert(has(r.board_claims_to_take, "ram@mad-lab-2026") && has(r.board_claims_to_take, "gpu:6900XT@mad-lab-2026"));

        // unknown RAM there (node not read yet): never gates
        in.machines[1].free_ram = -1;
        in.residents.clear();
        r = decide_admission(in);
        assert(r.verdict == ADMISSION_ADMIT);
    }
    // a foreign board claim on the remote card queues the load, naming that resource
    {
        admission_input in = two_box_input();
        admission_candidate c;
        c.machine = "mad-lab-2026";
        c.vram    = { { "mad-lab-2026/ROCm1", 1000 * MB } };
        in.candidates = { c };
        in.claims.push_back({ "c1", "gpu:6900XT@mad-lab-2026", "session-x", false, ADMISSION_PRIORITY_MIDDLE });
        const admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_CLAIM);
        assert(r.blocked_by == "session-x" && r.blocked_on == "gpu:6900XT@mad-lab-2026");
    }
}

static void test_board_machine_resources() {
    assert(router_board_qualify("gpu:R9700", "", "mad-lab-main") == "gpu:R9700");
    assert(router_board_qualify("gpu:R9700", "mad-lab-main", "mad-lab-main") == "gpu:R9700");
    assert(router_board_qualify("gpu:6900XT", "mad-lab-2026", "mad-lab-main") == "gpu:6900XT@mad-lab-2026");
    assert(router_board_qualify("ram", "mad-lab-2026", "") == "ram@mad-lab-2026");

    std::string res;
    std::string machine;
    router_board_unqualify("gpu:6900XT@mad-lab-2026", "mad-lab-main", res, machine);
    assert(res == "gpu:6900XT" && machine == "mad-lab-2026");
    router_board_unqualify("gpu:R9700", "mad-lab-main", res, machine);
    assert(res == "gpu:R9700" && machine == "mad-lab-main");
    assert(router_board_res_machine("ram@mad-lab-2026") == "mad-lab-2026" && router_board_res_machine("ram").empty());

    assert(router_board_is_machine_res("machine") && router_board_is_machine_res("machine@mad-lab-2026"));
    assert(!router_board_is_machine_res("gpu:R9700") && !router_board_is_machine_res("ram@mad-lab-2026"));
    // a whole-machine claim covers that machine's resources only
    assert(router_board_claim_covers("machine@mad-lab-2026", "gpu:6900XT@mad-lab-2026"));
    assert(router_board_claim_covers("machine@mad-lab-2026", "ram@mad-lab-2026"));
    assert(!router_board_claim_covers("machine@mad-lab-2026", "gpu:R9700"));
    assert(router_board_claim_covers("machine", "gpu:R9700") && router_board_claim_covers("machine", "ram"));
    assert(!router_board_claim_covers("machine", "ram@mad-lab-2026"));
    assert(router_board_claim_covers("ram", "ram") && !router_board_claim_covers("ram", "ram@mad-lab-2026"));

    // admission claims: a whole-machine claim on the remote box stands for the remote resources only
    const std::vector<std::string> all = { "gpu:R9700", "ram", "gpu:6900XT@mad-lab-2026", "ram@mad-lab-2026" };
    router_board_snapshot snap;
    snap.ok = true;
    router_board_claim c1;
    c1.id       = "c1";
    c1.machine  = "mad-lab-2026";
    c1.resource = "machine@mad-lab-2026";
    c1.holder   = "session-x";
    snap.claims.push_back(c1);
    router_board_claim c2;
    c2.id       = "c2";
    c2.resource = "gpu:R9700";
    c2.holder   = "llama-router";
    snap.claims.push_back(c2);
    const auto claims = router_board_admission_claims(snap, all, ADMISSION_PRIORITY_MIDDLE);
    int n_remote = 0;
    int n_local  = 0;
    for (const auto & c : claims) {
        if (c.holder == "session-x") {
            assert(c.resource == "gpu:6900XT@mad-lab-2026" || c.resource == "ram@mad-lab-2026");
            n_remote++;
        } else {
            assert(c.is_router && c.resource == "gpu:R9700");
            n_local++;
        }
    }
    assert(n_remote == 2 && n_local == 1);

    // someone queued for the remote machine: only the remote resources the router holds are contested
    snap.queue.clear();
    router_board_queue_entry q;
    q.id       = "q1";
    q.machine  = "mad-lab-2026";
    q.resource = "machine@mad-lab-2026";
    q.holder   = "session-y";
    snap.queue.push_back(q);
    const std::set<std::string> held = { "gpu:R9700", "gpu:6900XT@mad-lab-2026" };
    const std::set<std::string> contested = router_board_contested(snap, held);
    assert(contested.size() == 1 && contested.count("gpu:6900XT@mad-lab-2026") == 1);
}

static void test_heartbeat_offline() {
    // a remote node silent for offline_ms goes offline; one that is already offline does not "go" again
    assert(!router_node_offline_due(true, 1000, 10999, 10000));
    assert(router_node_offline_due(true, 1000, 11000, 10000));
    assert(router_node_offline_due(true, 1000, 60000, 10000));
    assert(!router_node_offline_due(false, 1000, 60000, 10000));
    // back: heard from within the window (and not online yet)
    assert(router_node_online_due(false, 59000, 60000, 10000));
    assert(!router_node_online_due(false, 1000, 60000, 10000));
    assert(!router_node_online_due(true, 59000, 60000, 10000));

    // the router's view: reversible marking
    router_machine_availability av;
    assert(!av.is_offline("mad-lab-2026") && av.first_offline({ "mad-lab-2026" }).empty());
    assert(av.set_online("mad-lab-2026", false));   // changed
    assert(!av.set_online("mad-lab-2026", false));  // already
    assert(av.is_offline("mad-lab-2026") && !av.is_offline("mad-lab-3"));
    assert(!av.is_offline("") && !av.set_online("", false)); // this machine is never offline
    assert(av.first_offline({ "", "mad-lab-3", "mad-lab-2026" }) == "mad-lab-2026");
    assert(av.first_offline({ "", "mad-lab-3" }).empty());

    // status shown for its models, and what a request gets
    assert(router_effective_status("loaded", av.is_offline("mad-lab-2026")) == "unavailable");
    assert(router_effective_status("unloaded", av.is_offline("mad-lab-2026")) == "unavailable");
    assert(router_effective_status("loaded", av.is_offline("")) == "loaded");

    int status = 0;
    std::string body;
    std::map<std::string, std::string> headers;
    router_unavailable_response("dsv41", "mad-lab-2026", "", status, body, headers);
    assert(status == 503);
    assert(headers.count("Retry-After") == 1 && headers.at("Retry-After") == std::to_string(ROUTER_NODE_RETRY_AFTER_S));
    const json j = json::parse(body);
    assert(j.at("error").at("code") == 503 && j.at("error").at("type") == "unavailable_error");
    assert(j.at("error").at("model") == "dsv41" && j.at("error").at("machine") == "mad-lab-2026");
    assert(j.at("error").at("message").get<std::string>().find("mad-lab-2026") != std::string::npos);
    router_unavailable_response("dsv41", "mad-lab-2026", "custom reason", status, body, headers);
    assert(json::parse(body).at("error").at("message") == "custom reason");

    // pools / admission skip the offline machine's slots
    const auto slot_machine = [](const std::string & id) {
        std::string m;
        std::string d;
        router_slot_split(id, m, d);
        return m;
    };
    const std::vector<std::string> pool = { "ROCm0", "mad-lab-2026/ROCm1", "mad-lab-3/ROCm2" };
    auto live = router_online_slots(pool, slot_machine, av.offline);
    assert(live.size() == 2 && live[0] == "ROCm0" && live[1] == "mad-lab-3/ROCm2");
    // recovery: the same pool is whole again, nothing was torn down
    assert(av.set_online("mad-lab-2026", true));
    assert(!av.set_online("mad-lab-2026", true));
    live = router_online_slots(pool, slot_machine, av.offline);
    assert(live.size() == 3);
    assert(router_effective_status("loaded", av.is_offline("mad-lab-2026")) == "loaded");
}

static router_node_seen seen(const std::string & name, const std::string & gen, int pid, const std::string & status, int exit_code = -1) {
    router_node_seen s;
    s.name      = name;
    s.gen       = gen;
    s.pid       = pid;
    s.status    = status;
    s.exit_code = exit_code;
    return s;
}

static void test_reconcile_table() {
    const std::string G = "gen-now";
    const std::map<std::string, int> watched = { { "keep", 100 }, { "unwanted", 200 }, { "repid", 300 }, { "readopt", 400 },
                                                  { "orphan-unwanted", 450 }, { "gone", 500 }, { "gone-exited", 600 } };
    const std::function<bool(const std::string &)> wanted = [](const std::string & n) { return n != "unwanted" && n != "orphan-unwanted"; };

    const std::vector<router_node_seen> children = {
        seen("keep",      G, 100, "running"),               // ours, still wanted -> keep
        seen("unwanted",  G, 200, "running"),               // ours, not wanted any more -> stop
        seen("repid",     G, 301, "running"),               // same name, another PID: not what we watched -> stop; ours is lost
        seen("stranger",  G, 700, "running"),               // unknown to the leader -> stop
        seen("other-gen", "gen-old", 800, "running"),       // another generation -> stop
        seen("stopping",  "gen-old", 801, "stopping"),      // already on its way out: left alone
        seen("done",      G, 900, "exited", 0),             // exited children are not judged
        seen("gone-exited", G, 600, "exited", 7),           // watched, exited while we were away: lost with its code
    };
    const std::vector<node_orphan_info> orphans = {
        { 400, "readopt",         G,        "waiting" },      // node lost track, still wanted -> adopt
        { 450, "orphan-unwanted", G,        "waiting" },      // node lost track, not wanted -> take back, then stop
        { 460, "foreign-orphan",  G,        "waiting" },      // not watched by us: take back and stop
        { 470, "old-orphan",      "gen-old", "waiting" },     // another generation: the node's own sweep
        { 480, "dying",           G,        "terminating" },  // already being terminated
    };
    const router_node_reconcile_plan plan = router_node_reconcile(children, orphans, watched, G, wanted);

    assert(plan.keep.size() == 1 && plan.keep[0] == "keep");
    assert(plan.adopt.size() == 1 && plan.adopt[0] == "readopt");
    assert(has(plan.adopt_stop, "orphan-unwanted") && has(plan.adopt_stop, "foreign-orphan") && plan.adopt_stop.size() == 2);
    assert(has(plan.stop, "unwanted") && has(plan.stop, "repid") && has(plan.stop, "stranger") && has(plan.stop, "other-gen"));
    assert(plan.stop.size() == 4); // not "stopping", "done", "keep"
    // lost: watched but the node no longer runs (or exited unseen): never reported twice
    bool lost_gone = false;
    bool lost_exited = false;
    bool lost_repid = false;
    for (const auto & l : plan.lost) {
        if (l.name == "gone")        { lost_gone = true;   assert(l.pid == 500 && l.exit_code == -1); }
        if (l.name == "gone-exited") { lost_exited = true; assert(l.pid == 600 && l.exit_code == 7); }
        if (l.name == "repid")       { lost_repid = true;  assert(l.pid == 300); }
        assert(l.name != "keep" && l.name != "unwanted" && l.name != "readopt" && l.name != "orphan-unwanted");
    }
    assert(lost_gone && lost_exited && lost_repid && plan.lost.size() == 3);

    // an empty node and nothing watched: nothing to do
    const router_node_reconcile_plan idle = router_node_reconcile({}, {}, {}, G, wanted);
    assert(idle.keep.empty() && idle.adopt.empty() && idle.adopt_stop.empty() && idle.stop.empty() && idle.lost.empty());

    // a node that restarted lost all its children: everything watched is lost, nothing is stopped
    const router_node_reconcile_plan restarted = router_node_reconcile({}, {}, watched, G, wanted);
    assert(restarted.lost.size() == watched.size() && restarted.stop.empty() && restarted.keep.empty());

    // children and orphans are told apart by the state JSON the node answers
    const auto from_json = router_node_seen_from_json(json::parse(
        R"([{"name":"a","gen":"g","pid":1,"status":"running"},{"name":"","pid":2},{"name":"b","gen":"g","pid":3,"status":"exited","exit_code":4,"killed":true}])"));
    assert(from_json.size() == 2 && from_json[0].name == "a" && from_json[1].exit_code == 4 && from_json[1].killed);
}

int main() {
    test_gpus_spec();
    test_slot_ids();
    test_remote_ledger();
    test_admission_remote();
    test_board_machine_resources();
    test_heartbeat_offline();
    test_reconcile_table();
    printf("test-router-multimachine: ok\n");
    return 0;
}
