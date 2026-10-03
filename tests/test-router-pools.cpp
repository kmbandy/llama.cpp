// Tests for pools and replicas (tools/server/server-router-admission.*): preset parsing, replica
// names, the slots a placement may use (online, one replica per slot), load balancing across the
// ready replicas, on-demand growth, and the admission of a pool's placements (two replicas on two
// slots, no eviction of busy residents below `highest`, the machine override).

#include "server-router-admission.h"

#undef NDEBUG
#include <cassert>
#include <cstdio>
#include <set>
#include <string>
#include <vector>

static constexpr int64_t GB = 1024LL * 1024LL * 1024LL;

static admission_slot slot(const std::string & id, const std::string & machine, int64_t free) {
    return { id, machine, "gpu:" + id, free };
}

static admission_resident resident(const std::string & name, const std::string & slot_id, int64_t vram, int64_t last_used, bool busy) {
    admission_resident r;
    r.name      = name;
    r.vram      = { { slot_id, vram } };
    r.last_used = last_used;
    r.busy      = busy;
    return r;
}

static std::string machine_of(const std::vector<admission_slot> & slots, const std::string & id) {
    for (const auto & s : slots) {
        if (s.id == id) {
            return s.machine;
        }
    }
    return "";
}

// the pool's candidate list for one placement: online slots, minus the ones sibling replicas hold
static std::vector<admission_candidate> pool_candidates(const std::vector<admission_slot> & slots, const std::set<std::string> & offline_machines,
                                                        const std::set<std::string> & taken, int64_t need) {
    std::vector<std::string> all;
    for (const auto & s : slots) {
        all.push_back(s.id);
    }
    const auto usable = [&](const std::string & id) { return offline_machines.count(machine_of(slots, id)) == 0; };
    std::vector<admission_slot> keep;
    for (const auto & id : router_pool_slots(all, usable, taken)) {
        for (const auto & s : slots) {
            if (s.id == id) {
                keep.push_back(s);
            }
        }
    }
    return admission_pool_candidates(keep, need, 0);
}

int main() {
    // ---- preset keys
    {
        router_pool_spec s = router_pool_parse("", "", "");
        assert(s.err.empty() && !s.pool && s.replicas == 1 && s.gpus.empty());
        s = router_pool_parse("pinned", "1", "");
        assert(s.err.empty() && !s.pool);
        s = router_pool_parse("any", "", "");
        assert(s.err.empty() && s.pool && s.replicas == 1);
        s = router_pool_parse(" any ", " 3 ", "R9700, m2/6900XT ,");
        assert(s.err.empty() && s.pool && s.replicas == 3 && s.gpus.size() == 2 && s.gpus[0] == "R9700" && s.gpus[1] == "m2/6900XT");
        assert(!router_pool_parse("sometimes", "", "").err.empty());
        assert(!router_pool_parse("any", "0", "").err.empty());
        assert(!router_pool_parse("any", "-2", "").err.empty());
        assert(!router_pool_parse("any", "two", "").err.empty());
        assert(!router_pool_parse("any", "65", "").err.empty());
        assert(!router_pool_parse("pinned", "2", "").err.empty());   // replicas need a pool
        assert(!router_pool_parse("", "", "R9700").err.empty());     // so does an allow-list
    }

    // ---- replica names
    {
        assert(router_replica_name("qwen", 1) == "qwen");
        assert(router_replica_name("qwen", 2) == "qwen~r2");
        std::string a;
        int k = 0;
        assert(router_replica_split("qwen~r3", a, k) && a == "qwen" && k == 3);
        assert(!router_replica_split("qwen", a, k) && a == "qwen" && k == 1);
        assert(!router_replica_split("qwen~r", a, k) && k == 1);
        assert(!router_replica_split("qwen~rx", a, k) && k == 1);
        assert(!router_replica_split("qwen~r1", a, k) && k == 1);   // replica 1 is the alias
        assert(router_replica_split("a~rb~r12", a, k) && a == "a~rb" && k == 12);
    }

    // ---- the slots a placement may use
    {
        const std::vector<std::string> all = { "g0", "g1", "m2/g0" };
        const auto remote_off = [](const std::string & id) { return id.rfind("m2/", 0) != 0; };
        assert((router_pool_slots(all, nullptr, {}) == std::vector<std::string>{ "g0", "g1", "m2/g0" }));
        assert((router_pool_slots(all, remote_off, {}) == std::vector<std::string>{ "g0", "g1" }));
        assert((router_pool_slots(all, remote_off, { "g0" }) == std::vector<std::string>{ "g1" }));
        assert((router_pool_slots(all, nullptr, { "g0", "g1", "m2/g0" }).empty()));
    }

    // ---- load balancing across replicas
    {
        using S = router_replica_state;
        // one ready, idle: use it, nothing to start
        router_replica_choice c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 0 } }, true);
        assert(c.use == 0 && !c.use_new && c.grow < 0 && !c.grow_new);
        // fewest in flight wins; the first on a tie
        c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 3 }, S{ ROUTER_REPLICA_READY, 1 }, S{ ROUTER_REPLICA_READY, 1 } }, false);
        assert(c.use == 1 && c.grow < 0 && !c.grow_new);
        // an idle one beats busy ones: no growth
        c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 2 }, S{ ROUTER_REPLICA_READY, 0 } }, true);
        assert(c.use == 1 && !c.grow_new && c.grow < 0);
        // all busy, room for another: serve it now on the least busy, grow a new one
        c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 2 } }, true);
        assert(c.use == 0 && c.grow_new);
        // all busy, a replica entry is down: bring that one up
        c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 2 }, S{ ROUTER_REPLICA_DOWN, 0 } }, false);
        assert(c.use == 0 && c.grow == 1 && !c.grow_new);
        // all busy, at the limit: the busy one serves, nothing starts
        c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 2 }, S{ ROUTER_REPLICA_READY, 4 } }, false);
        assert(c.use == 0 && c.grow < 0 && !c.grow_new);
        // all busy but one is already loading: no second start
        c = router_replica_choose({ S{ ROUTER_REPLICA_READY, 2 }, S{ ROUTER_REPLICA_LOADING, 0 } }, true);
        assert(c.use == 0 && c.grow < 0 && !c.grow_new);
        // nothing ready: wait on the one loading, else load the down one, else a new one
        c = router_replica_choose({ S{ ROUTER_REPLICA_DOWN, 0 }, S{ ROUTER_REPLICA_LOADING, 0 } }, true);
        assert(c.use == 1 && c.grow < 0);
        c = router_replica_choose({ S{ ROUTER_REPLICA_DOWN, 0 } }, true);
        assert(c.use == 0 && !c.use_new);
        c = router_replica_choose({}, true);
        assert(c.use < 0 && c.use_new);
        c = router_replica_choose({}, false);
        assert(c.use < 0 && !c.use_new && !c.grow_new);
    }

    const std::vector<admission_slot> two = { slot("g0", "", 24 * GB), slot("g1", "", 24 * GB) };

    // ---- two replicas land on two slots
    {
        // first replica: both slots free, the pool picks one
        admission_input in;
        in.alias      = "pool";
        in.margin_bytes = 1 * GB;
        in.slots      = two;
        in.candidates = pool_candidates(two, {}, {}, 10 * GB);
        admission_result r1 = decide_admission(in);
        assert(r1.verdict == ADMISSION_ADMIT && r1.slots.size() == 1);
        const std::string first = r1.slots[0];

        // second replica: the first one's slot is taken (one replica per slot) -> the other slot, with no victims
        admission_input in2;
        in2.alias        = "pool~r2";
        in2.margin_bytes = 1 * GB;
        in2.slots        = two;
        for (auto & s : in2.slots) {
            if (s.id == first) {
                s.free = 14 * GB; // what the first replica left
            }
        }
        in2.candidates = pool_candidates(two, {}, { first }, 10 * GB);
        in2.residents  = { resident("pool", first, 10 * GB, 100, false) };
        const admission_result r2 = decide_admission(in2);
        assert(r2.verdict == ADMISSION_ADMIT && r2.slots.size() == 1 && r2.slots[0] != first && r2.victims.empty());

        // a third replica: no slot is free of a sibling -> no candidate, nothing is evicted
        admission_input in3;
        in3.alias        = "pool~r3";
        in3.margin_bytes = 1 * GB;
        in3.slots        = two;
        in3.candidates   = pool_candidates(two, {}, { "g0", "g1" }, 10 * GB);
        in3.residents    = { resident("pool", "g0", 10 * GB, 100, true), resident("pool~r2", "g1", 10 * GB, 101, true) };
        const admission_result r3 = decide_admission(in3);
        assert(r3.verdict == ADMISSION_QUEUE && r3.victims.empty() && r3.blocked == ADMISSION_BLOCK_NO_CANDIDATE);
    }

    // ---- a replica that needs a slot held by another model's busy resident queues below `highest`
    {
        // both cards full: g0 by busy "other", g1 by the pool's own busy first replica (a sibling: not a candidate)
        std::vector<admission_slot> full = { slot("g0", "", 2 * GB), slot("g1", "", 2 * GB) };
        for (admission_priority prio : { ADMISSION_PRIORITY_LOWEST, ADMISSION_PRIORITY_MIDDLE, ADMISSION_PRIORITY_HIGHEST }) {
            admission_input in;
            in.alias        = "pool~r2";
            in.priority     = prio;
            in.margin_bytes = 1 * GB;
            in.slots        = full;
            in.candidates   = pool_candidates(full, {}, { "g1" }, 10 * GB);
            in.residents    = { resident("other", "g0", 22 * GB, 100, true), resident("pool", "g1", 22 * GB, 101, true) };
            const admission_result r = decide_admission(in);
            if (prio == ADMISSION_PRIORITY_HIGHEST) {
                assert(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "other" });
            } else {
                assert(r.verdict == ADMISSION_QUEUE && r.victims.empty() && r.blocked == ADMISSION_BLOCK_BUSY && r.blocked_by == "other");
            }
        }
        // the same with the resident idle: even `lowest` evicts it
        admission_input in;
        in.alias        = "pool~r2";
        in.priority     = ADMISSION_PRIORITY_LOWEST;
        in.margin_bytes = 1 * GB;
        in.slots        = full;
        in.candidates   = pool_candidates(full, {}, { "g1" }, 10 * GB);
        in.residents    = { resident("other", "g0", 22 * GB, 100, false), resident("pool", "g1", 22 * GB, 101, true) };
        const admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_EVICT_THEN_ADMIT && r.victims == std::vector<std::string>{ "other" });
    }

    // ---- machine override
    {
        const std::vector<admission_slot> slots = { slot("g0", "", 10 * GB), slot("m2/g0", "m2", 40 * GB) };
        admission_input in;
        in.alias        = "pool";
        in.margin_bytes = 1 * GB;
        in.slots        = slots;
        in.candidates   = pool_candidates(slots, {}, {}, 8 * GB);
        // no override: the card with the most free VRAM
        assert(decide_admission(in).slots == std::vector<std::string>{ "m2/g0" });
        // override to the machine of the roomier slot
        in.machine = "m2";
        admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ "m2/g0" });
        // override to a machine with no slot of the pool: nothing to place on
        in.machine = "m3";
        r = decide_admission(in);
        assert(r.verdict == ADMISSION_QUEUE && r.blocked == ADMISSION_BLOCK_NO_CANDIDATE);
    }
    {
        // the override beats "most free VRAM": the pool is restricted to the machine asked for
        const std::vector<admission_slot> slots = { slot("g0", "", 40 * GB), slot("m3/g0", "m3", 12 * GB) };
        admission_input in;
        in.alias        = "pool";
        in.margin_bytes = 1 * GB;
        in.slots        = slots;
        in.candidates   = pool_candidates(slots, {}, {}, 8 * GB);
        assert(decide_admission(in).slots == std::vector<std::string>{ "g0" });
        in.machine = "m3";
        const admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ "m3/g0" });
    }

    // ---- a pool's past pick is no claim on the next: every placement starts from the online slots
    {
        const std::vector<admission_slot> slots = { slot("g0", "", 30 * GB), slot("m2/g0", "m2", 40 * GB) };
        // placement 1: both machines online, the pool takes the roomier remote slot
        admission_input in;
        in.alias        = "pool";
        in.margin_bytes = 1 * GB;
        in.slots        = slots;
        in.candidates   = pool_candidates(slots, {}, {}, 8 * GB);
        assert(decide_admission(in).slots == std::vector<std::string>{ "m2/g0" });
        // placement 2: m2 is offline now; the pool places on the local slot (it does not throw `unavailable`)
        in.candidates = pool_candidates(slots, { "m2" }, {}, 8 * GB);
        assert(in.candidates.size() == 1);
        const admission_result r = decide_admission(in);
        assert(r.verdict == ADMISSION_ADMIT && r.slots == std::vector<std::string>{ "g0" });
        // every slot offline: no candidate at all (the caller answers `unavailable`)
        assert(pool_candidates(slots, { "", "m2" }, {}, 8 * GB).empty());
    }

    printf("test-router-pools: OK\n");
    return 0;
}
